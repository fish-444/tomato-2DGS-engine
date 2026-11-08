"""Hybrid H (contract build/p1_leafseg_20261020_05/CONTRACT.md): M1 (YOLO11s-seg, leafinst) finds leaflets at a low score,
the tuned SAM 2.1 tiny v5 draws each boundary (H1 box prompt / H2 box + points), fragments are merged, duplicates
removed, then run 02's tail (`engine.finish`) runs unchanged. Existing leafseg functions are imported, never changed.

usage: python -m tomato3d.leafseg.hybrid --out build/p1_leafseg_20261020_05 --variant H1 [--images tray_top_01 ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.leafseg import engine as eng
from tomato3d.leafseg import pixels as px
from tomato3d.leafseg import rfdet
from tomato3d.leafseg import run as r2
from tomato3d.leafseg.variant_d import _dt_max, _iou_m, _rachis_fraction

B = Path("build/p1_leafseg_20261019_02")
RUN04_PARAMS = Path("build/p1_leafseg_20261019_04/parameters.json")
M1_WEIGHTS = Path("build/p1_leafinst_20261020_01/kaggle_out_M1/M1_s0_best.pt")
DEV = tuple(f"tray_top_0{i}" for i in range(1, 7)) + ("dwt_101", "dwt_102", "dwt_103")
HELD = ("tray_top_07", "tray_top_08", "tray_top_09", "dwt_104")
# ---- declared constants (contract Steps 1-3)
M1_SCORE, TILE, STRIDE, IMGSZ, TILE_NMS = 0.15, 1024, 819, 1024, 0.5
PAD = 0.15
N_NEG_M1, N_NEG_RACHIS = 4, 2
FG_SHARE, MERGE_IOU, MERGE_COVER, BIG_X_MEDIAN, FALLBACK_OVERLAP = 0.5, 0.5, 0.8, 4.0, 0.3


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- Step 1: M1 sliced inference (as the leafinst kernel)
def _iou_box(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def _crop_iou(a, b):
    x0, y0 = min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]); x1, y1 = max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])
    A = np.zeros((y1 - y0, x1 - x0), bool); Bm = np.zeros_like(A)
    A[a["box"][1] - y0:a["box"][3] - y0, a["box"][0] - x0:a["box"][2] - x0] = a["crop"]; Bm[b["box"][1] - y0:b["box"][3] - y0, b["box"][0] - x0:b["box"][2] - x0] = b["crop"]
    return float((A & Bm).sum() / max((A | Bm).sum(), 1))


def m1_detect(model, rgb: np.ndarray) -> list[dict]:
    """Tiles 1024/819 (last row/col aligned to the edge), score >= M1_SCORE, crop-based mask-IoU NMS 0.5. Full-res crops."""
    H, W = rgb.shape[:2]; ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)); xs = list(range(0, max(W - TILE, 0) + 1, STRIDE))
    if ys[-1] + TILE < H:
        ys.append(H - TILE)
    if xs[-1] + TILE < W:
        xs.append(W - TILE)
    cands = []
    for y0 in ys:
        for x0 in xs:
            tile = np.ascontiguousarray(rgb[y0:y0 + TILE, x0:x0 + TILE])
            r = model.predict(tile[:, :, ::-1], imgsz=IMGSZ, conf=M1_SCORE, iou=0.7, max_det=300, retina_masks=True, verbose=False, device="cpu")[0]
            if r.masks is None or not len(r.masks):
                continue
            pm = r.masks.data.cpu().numpy() > 0.5; ps = r.boxes.conf.cpu().numpy()
            for m, s in zip(pm, ps):
                ys_, xs_ = np.nonzero(m)
                if not len(xs_):
                    continue
                by0, by1, bx0, bx1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
                cands.append(dict(score=float(s), box=[int(x0 + bx0), int(y0 + by0), int(x0 + bx1), int(y0 + by1)], crop=m[by0:by1, bx0:bx1].copy()))
    cands.sort(key=lambda c: -c["score"]); kept = []
    for c in cands:
        if all(_iou_box(c["box"], k["box"]) == 0 or _crop_iou(c, k) <= TILE_NMS for k in kept):
            kept.append(c)
    return kept


def m1_to_json(kept: list[dict], hw) -> dict:
    out = []
    for k, c in enumerate(kept):
        full = np.zeros(hw, bool); full[c["box"][1]:c["box"][3], c["box"][0]:c["box"][2]] = c["crop"]
        out.append(dict(id=f"m{k:03d}", score=round(c["score"], 4), box=c["box"], mask=px.rle_encode(full)))
    return dict(score_min=M1_SCORE, tile=TILE, stride=STRIDE, instances=out)


# --------------------------------------------------------------------------- Step 2: SAM prompts
def sam_pick(sam, box_full, pos_full, neg_full, gshape):
    """Same selection as engine._predict (argmax predicted IoU x solidity) but allows a box-only prompt (H1)."""
    if pos_full is None or not len(pos_full):
        masks, ious, _ = sam.predictor.predict(box=np.asarray(box_full, np.float32), point_coords=None, point_labels=None, multimask_output=True)
    else:
        pts = np.vstack([pos_full, neg_full]) if len(neg_full) else np.asarray(pos_full)
        lab = np.r_[np.ones(len(pos_full)), np.zeros(len(neg_full))].astype(np.int32)
        masks, ious, _ = sam.predictor.predict(box=np.asarray(box_full, np.float32), point_coords=np.asarray(pts, np.float32), point_labels=lab, multimask_output=True)
    best = None
    for m, q in zip(masks, ious):
        g = eng._to_grid(m > 0, gshape); sc = float(q) * eng._solidity(g)
        if best is None or sc > best[1]:
            best = (g, sc, float(q))
    return best


def padded_box(box, hw):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    return [max(0.0, x0 - PAD * w), max(0.0, y0 - PAD * h), min(float(hw[1]), x1 + PAD * w), min(float(hw[0]), y1 + PAD * h)]


def prompts_h2(i: int, m1: list[dict], pbox, skel: np.ndarray, gshape, s: float, rng):
    """Positive = DT max of the M1 mask; negatives = DT maxima of <= 4 other M1 instances centred in the padded box
    (highest score first) + <= 2 rachis skeleton pixels inside the box. Coordinates in full-res pixels."""
    pos = _dt_max(m1[i]["grid"]) / s
    inside = [j for j in range(len(m1)) if j != i and pbox[0] <= m1[j]["centre"][0] <= pbox[2] and pbox[1] <= m1[j]["centre"][1] <= pbox[3]]
    inside = sorted(inside, key=lambda j: -m1[j]["score"])[:N_NEG_M1]
    neg = [_dt_max(m1[j]["grid"]) / s for j in inside]
    x0, y0, x1, y1 = [int(round(v * s)) for v in pbox]; reg = np.zeros(gshape, bool); reg[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
    rs = eng._sample_points(skel & reg, N_NEG_RACHIS, rng)
    if len(rs):
        neg.append(rs / s)
    return pos, (np.vstack(neg) if neg else np.zeros((0, 2)))


# --------------------------------------------------------------------------- Step 3: merge and clean
def clean_and_merge(cands: list[dict], m1: list[dict], F: int, fg: np.ndarray, rach_d: np.ndarray, gshape) -> tuple[list[dict], dict]:
    """cands: SAM outputs per M1 instance (dict m1=index, mask, score, iou_pred) or None entries. Returns kept entries + log."""
    log = dict(reject=dict(too_small=0, low_foreground=0, rachis_kept_as_leaf=0, merged_blade_suspect=0), merged_into=0, fallback=0, fallback_rejected=0)
    live = []
    for c in cands:
        if c is None:
            continue
        m = c["mask"]
        if m.sum() < F:
            log["reject"]["too_small"] += 1; c["drop"] = "too_small"
        elif fg[m].mean() < FG_SHARE:
            log["reject"]["low_foreground"] += 1; c["drop"] = "low_foreground"
        elif _rachis_fraction(m, rach_d, gshape) > eng.RACHIS_FRAC_MAX:
            log["reject"]["rachis_kept_as_leaf"] += 1; c["drop"] = "rachis_kept_as_leaf"
        else:
            live.append(c)
    # merged-blade suspects: > 4 x median area and >= 2 other live centres inside
    areas = np.array([c["mask"].sum() for c in live]); med = float(np.median(areas)) if len(areas) else 0.0
    cents = []
    for c in live:
        ys, xs = np.nonzero(c["mask"]); cents.append((int(ys.mean()), int(xs.mean())))
    keep2 = []
    for k, c in enumerate(live):
        inside = sum(1 for j, (cy, cx) in enumerate(cents) if j != k and c["mask"][cy, cx])
        if med and c["mask"].sum() > BIG_X_MEDIAN * med and inside >= 2:
            log["reject"]["merged_blade_suspect"] += 1; c["drop"] = "merged_blade_suspect"
        else:
            keep2.append(c)
    # same-leaflet merge, greedy by SAM predicted IoU
    keep2.sort(key=lambda c: -c["iou_pred"]); kept = []
    for c in keep2:
        a = c["mask"].sum(); host = None
        for k in kept:
            inter = (c["mask"] & k["mask"]).sum()
            if inter / max((c["mask"] | k["mask"]).sum(), 1) >= MERGE_IOU or inter / max(min(a, k["mask"].sum()), 1) >= MERGE_COVER:
                host = k; break
        if host is None:
            c["merged_from"] = [m1[c["m1"]]["id"]]; kept.append(c)
        else:
            host["merged_from"].append(m1[c["m1"]]["id"]); log["merged_into"] += 1; c["drop"] = "merged"
    # M1 fallback for instances whose SAM output was rejected or missing
    absorbed = {mid for c in kept for mid in c["merged_from"]}
    for i in sorted(range(len(m1)), key=lambda i: -m1[i]["score"]):
        if m1[i]["id"] in absorbed:
            continue
        g = m1[i]["grid"]; a = g.sum()
        if a < F:
            log["fallback_rejected"] += 1; continue
        ok = all((g & k["mask"]).sum() <= FALLBACK_OVERLAP * min(a, k["mask"].sum()) for k in kept)
        if ok:
            kept.append(dict(m1=i, mask=g, score=m1[i]["score"], iou_pred=None, source="M1_fallback", merged_from=[m1[i]["id"]])); log["fallback"] += 1
        else:
            log["fallback_rejected"] += 1
    return kept, log


# --------------------------------------------------------------------------- per photo
def run_photo(name: str, variant: str, model, sam, cfg: dict, floors: dict, out: Path, rgb: np.ndarray | None = None) -> dict:
    t0 = time.time(); doc = json.load(open(B / "proposals" / f"{name}_leafseg.json")); L = np.load(B / "layers" / f"{name}.npz")
    gshape = tuple(doc["grid_shape"]); s = doc["grid_scale"]; H, W = doc["image_size"]; fg, skel = L["fg"], L["skeleton"]
    rach_d = cv2.dilate(skel.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0; F = floors[name]; rng = np.random.default_rng(0)
    rgb = r2.load_rgb(r2.images()[name]) if rgb is None else rgb; rgb_g = cv2.resize(rgb, (gshape[1], gshape[0]), interpolation=cv2.INTER_AREA)
    # M1 (cached per photo, shared by the variants)
    mp = out / "m1" / f"{name}.json"
    if mp.exists():
        mj = json.load(open(mp))
    else:
        t1 = time.time(); mj = m1_to_json(m1_detect(model, rgb), (H, W)); mj["cpu_seconds"] = round(time.time() - t1, 1)
        mp.parent.mkdir(parents=True, exist_ok=True); json.dump(mj, open(mp, "w"))
    m1 = []
    for inst in mj["instances"]:
        full = px.rle_decode(inst["mask"]); g = cv2.resize(full.astype(np.float32), (gshape[1], gshape[0]), interpolation=cv2.INTER_AREA) >= 0.5
        if not g.any():
            continue
        m1.append(dict(id=inst["id"], score=inst["score"], box=inst["box"], grid=g, centre=[(inst["box"][0] + inst["box"][2]) / 2, (inst["box"][1] + inst["box"][3]) / 2]))
    # SAM per instance
    sam.predictor.set_image(rgb); cands = []
    for i, inst in enumerate(m1):
        pbox = padded_box(inst["box"], (H, W))
        if variant == "H1":
            r = sam_pick(sam, pbox, None, None, gshape)
        else:
            pos, neg = prompts_h2(i, m1, pbox, skel, gshape, s, rng); r = sam_pick(sam, pbox, pos, neg, gshape)
        cands.append(None if r is None or not r[0].any() else dict(m1=i, mask=r[0], score=r[1], iou_pred=r[2], source=f"{variant}_sam"))
    kept, log = clean_and_merge(cands, m1, F, fg, rach_d, gshape)
    # engine entries + v4 dets for the merged-suspect flag (as B)
    predj = json.load(open(B / "detections" / f"{name}.json")); dets = rfdet.to_dets(predj, (H, W)); comps = rfdet.compound_boxes(predj, (H, W))
    final = []
    for k, c in enumerate(sorted(kept, key=lambda c: -c["score"])):
        g = c["mask"]; ys, xs = np.nonzero(g); cx, cy = float(xs.mean() / s), float(ys.mean() / s)
        own = [d for d in dets if g[min(int(d["centre"][1] * s), gshape[0] - 1), min(int(d["centre"][0] * s), gshape[1] - 1)]]
        bb = [xs.min() / s, ys.min() / s, xs.max() / s, ys.max() / s]
        own = max(own, key=lambda d: _iou_box(bb, d["box"])) if own else dict(det=-1, conf=c["score"], cls="leaflet", centre=[cx, cy], box=bb)
        final.append(dict(id=f"H{k + 1:03d}", mask=g, score=float(c["score"]), iou_pred=c["iou_pred"], peel_rank=k, fg_mean=float(fg[g].mean()),
                          rachis_fraction=_rachis_fraction(g, rach_d, gshape), det=dict(own), source=c.get("source", f"{variant}_sam"), merged_from=c["merged_from"],
                          m1_score=m1[c["m1"]]["score"]))
    res = eng.finish(final, dets, rgb_g, fg, dict(skeleton=skel, ends=[], branches=[]), cfg, gshape, s, (H, W))
    for inst, c in zip(res["instances"], final):
        inst.update(source=c["source"], merged_from=c["merged_from"], m1_score=c["m1_score"], solidity=round(eng._solidity(c["mask"]), 4))
    rfdet.assign_compound_boxes(res["_layers"]["label"], s, res["instances"], comps); res["compound_boxes"] = comps
    # outputs
    ov = r2.overlay(res); lab = res["_layers"]["label"]
    for i, c in enumerate(final):
        cs, _ = cv2.findContours((lab == i + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if c["source"] == "M1_fallback":
            cv2.drawContours(ov, cs, -1, (40, 90, 255), 3)
        elif len(c["merged_from"]) > 1:
            cv2.drawContours(ov, cs, -1, (0, 220, 60), 3)
    for d in ("proposals", "overlays", "layers", "logs"):
        (out / variant / d).mkdir(parents=True, exist_ok=True)
    Image.fromarray(ov).save(out / variant / "overlays" / f"{name}.jpg", quality=88)
    Lr = res.pop("_layers"); np.savez_compressed(out / variant / "layers" / f"{name}.npz", label=Lr["label"].astype(np.int16), fg=fg, skeleton=skel)
    res.update(rachis=doc["rachis"], detections=len(m1), candidates=len([c for c in cands if c]), dropped=[])
    json.dump(dict(schema="leafseg-1", image=dict(doc["image"], variant=variant), source="model_proposal",
                   note=f"Hybrid {variant}: M1 instances -> SAM v5 masks -> merge/clean -> run 02 T1 tail; no accuracy claim", **res),
              open(out / variant / "proposals" / f"{name}_leafseg.json", "w"), default=float)
    insts = res["instances"]; areas = np.array([i["visible_px"] for i in insts]); sol = np.array([i["solidity"] for i in insts])
    rec = dict(split=doc["image"]["split"], variant=variant, m1_instances=len(m1), m1_cpu_seconds=mj.get("cpu_seconds"), sam_candidates=len([c for c in cands if c]),
               leaflets=len(insts), occluded=sum(i["occluded"] for i in insts), order_edges=len(res["order_edges"]), unassigned_area_fraction=round(res["unassigned_area_fraction"], 3),
               merged=sum("merged" in i["flags"] for i in insts), share_lt_1500px=round(float((areas < 1500).mean()), 3) if len(areas) else None,
               m1_fragments_merged=int(sum(len(i["merged_from"]) - 1 for i in insts)), share_solidity_lt_0_7=round(float((sol < 0.7).mean()), 3) if len(sol) else None,
               fallback=sum(i["source"] == "M1_fallback" for i in insts), compound_unassigned=sum(i["compound_leaf_id"] == "unassigned" for i in insts),
               compound_box_unassigned=sum(i["compound_box_id"] == "unassigned" for i in insts), floor_px=F, log=log, seconds=round(time.time() - t0, 1))
    json.dump(rec, open(out / variant / "logs" / f"{name}.json", "w"), indent=1, default=float)
    return rec


def b_row(name: str) -> dict:
    g = json.load(open(B / "gateA.json"))[name]; pb = json.load(open(B / "proposals" / f"{name}_leafseg.json")); ab = np.array([i["visible_px"] for i in pb["instances"]])
    sol = [eng._solidity(px.rle_decode(i["visible_segmentation"])) for i in pb["instances"]]
    return dict(leaflets=g["leaflets"], unassigned_area_fraction=g["unassigned_area_fraction"], merged=len(g["failure_candidates"]["merged"]),
                share_lt_1500px=round(float((ab < 1500).mean()), 3), share_solidity_lt_0_7=round(float(np.mean(np.array(sol) < 0.7)), 3))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--variant", choices=("H1", "H2"), required=True)
    ap.add_argument("--images", nargs="*"); a = ap.parse_args(argv); out = Path(a.out)
    from ultralytics import YOLO
    from tomato3d.topview.measure2d import load_sam
    model = YOLO(str(M1_WEIGHTS)); sam = load_sam(r2.SAMCFG); cfg = json.load(open(r2.T1CFG)); floors = json.load(open(RUN04_PARAMS))["floors"]
    gp = out / f"gate_{a.variant}.json"; gate = json.load(open(gp)) if gp.exists() else {}
    for name in (a.images or list(DEV)):
        if name in gate and (out / a.variant / "proposals" / f"{name}_leafseg.json").exists():
            continue
        rec = run_photo(name, a.variant, model, sam, cfg, floors, out); rec["B"] = b_row(name); gate[name] = rec
        print(name, {k: v for k, v in rec.items() if k not in ("log", "B")}, "| B", rec["B"], flush=True)
        json.dump(gate, open(gp, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
