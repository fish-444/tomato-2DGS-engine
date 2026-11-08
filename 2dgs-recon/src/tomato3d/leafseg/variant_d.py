"""Variant D (contract build/p1_leafseg_20261019_04/CONTRACT.md): run 02 variant B masks + D1 low-confidence recovery
+ D2 merged-mask splitting with the same tuned SAM 2.1 tiny v5. No detector query, no training.

usage: python -m tomato3d.leafseg.variant_d --base build/p1_leafseg_20261019_02 --out build/p1_leafseg_20261019_04 [--images ...]
"""

from __future__ import annotations

import argparse
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

# ---- declared thresholds (D0-D2) — recorded in parameters.json, never changed after a result
FLOOR_PCT = 25            # leaflet-size floor = this percentile of B visible_px per photo
LOWCONF = (0.10, 0.25)    # D1 candidate confidence range [lo, hi)
BOX_IN_U = 0.50           # D1: box area inside the unassigned region
MASK_IN_U = 0.70          # D1: new-mask pixels inside the unassigned region
FG_SHARE = 0.50           # foreground rule (share of pixels in B's fg layer)
D1_NMS = 0.5              # D1 candidate box NMS and mask-IoU limit
D2_COVER = 0.85           # D2: union of parts over target
D2_PAIR_IOU = 0.2         # D2: pairwise part IoU
D2_SOLIDITY_DROP = 0.05   # D2: part solidity >= target solidity - this
N_NEG_RACHIS, N_NEG_PEEL = 4, 3
ORDER = [f"tray_top_0{i}" for i in range(1, 10)] + ["dwt_101", "dwt_102", "dwt_103", "dwt_104"]


def _iou_m(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def _dt_max(region: np.ndarray):
    dt = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5); yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return np.array([[xx, yy]], np.float32)


def _rachis_fraction(m: np.ndarray, rach_d: np.ndarray, gshape) -> float:
    from tomato3d.topview.occlusion import features as t1
    med = t1.skeleton(cv2.resize(m.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) > 0)
    med = cv2.resize(med.astype(np.uint8), (gshape[1], gshape[0]), interpolation=cv2.INTER_NEAREST) > 0
    med = cv2.dilate(med.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float((med & rach_d).sum() / max(med.sum(), 1))


def low_conf_boxes(base: Path, name: str, image_hw) -> list[dict]:
    """D1.2: 0.10 <= c < 0.25 leaflet boxes from the run 02 whole-image response and (tray) the cached tile responses."""
    lo, hi = LOWCONF; out = []
    pred = json.load(open(base / "detections" / f"{name}.json"))
    sx = image_hw[1] / pred["image"]["width"]
    for p in pred["predictions"]:
        if p["class"] in rfdet.LEAFLET_CLASSES and lo <= p["confidence"] < hi:
            out.append(dict(conf=p["confidence"], cls=p["class"], src="whole",
                            box=[(p["x"] - p["width"] / 2) * sx, (p["y"] - p["height"] / 2) * sx, (p["x"] + p["width"] / 2) * sx, (p["y"] + p["height"] / 2) * sx]))
    H, W = image_hw; T, S, E = rfdet.TILE, rfdet.STRIDE, rfdet.EDGE_PX
    for f in sorted((base / "tiled" / "detections_tiled").glob(f"{name}_r*c*.json")):
        i, j = int(f.stem.split("_r")[1].split("c")[0]), int(f.stem.split("c")[-1]); y0, x0 = i * S, j * S
        pt = json.load(open(f)); tw, th = pt["image"]["width"], pt["image"]["height"]
        for p in pt["predictions"]:
            if p["class"] not in rfdet.LEAFLET_CLASSES or not (lo <= p["confidence"] < hi):
                continue
            bx0, by0, bx1, by1 = p["x"] - p["width"] / 2, p["y"] - p["height"] / 2, p["x"] + p["width"] / 2, p["y"] + p["height"] / 2
            cut = (bx0 < E and x0 > 0) or (by0 < E and y0 > 0) or (bx1 > tw - E and x0 + tw < W) or (by1 > th - E and y0 + th < H)
            if not cut:
                out.append(dict(conf=p["confidence"], cls=p["class"], src=f"tile r{i}c{j}", box=[bx0 + x0, by0 + y0, bx1 + x0, by1 + y0]))
    out.sort(key=lambda d: -d["conf"]); keep = []
    for d in out:
        if all(rfdet._iou(d["box"], k["box"]) <= D1_NMS for k in keep):
            keep.append(d)
    return keep


def recover(sam, cands: list[dict], b_masks: list[np.ndarray], U: np.ndarray, fg: np.ndarray, rach_d: np.ndarray, gshape, s: float, floor: int, log: list) -> list[dict]:
    """D1.3-D1.5."""
    added = []
    for k, d in enumerate(cands):
        x0, y0, x1, y1 = [int(round(v * s)) for v in d["box"]]; x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, gshape[1] - 1), min(y1, gshape[0] - 1)
        rec = dict(k=k, conf=d["conf"], cls=d["cls"], src=d["src"], box=d["box"])
        if x1 <= x0 or y1 <= y0:
            rec["decision"] = "rejected: empty_box"; log.append(rec); continue
        reg = np.zeros(gshape, bool); reg[y0:y1 + 1, x0:x1 + 1] = True; in_u = float((reg & U).sum() / reg.sum())
        rec["box_in_U"] = round(in_u, 3)
        if in_u < BOX_IN_U:
            rec["decision"] = "rejected: box_outside_unassigned"; log.append(rec); continue
        pos = _dt_max(reg & U)
        g, sc, q = eng._predict(sam, d["box"], pos / s, np.zeros((0, 2)), gshape)
        rec.update(sam_iou_pred=round(q, 3), area=int(g.sum()))
        why = None
        if g.sum() < floor:
            why = "too_small"
        elif (g & U).sum() / max(g.sum(), 1) < MASK_IN_U:
            why = "steals_from_B"
        elif fg[g].mean() < FG_SHARE:
            why = "low_foreground"
        elif any(_iou_m(g, m) > D1_NMS for m in b_masks) or any(_iou_m(g, a["mask"]) > D1_NMS for a in added):
            why = "nms"
        elif _rachis_fraction(g, rach_d, gshape) > eng.RACHIS_FRAC_MAX:
            why = "rachis_kept_as_leaf"
        rec["decision"] = f"rejected: {why}" if why else "accepted"; log.append(rec)
        if not why:
            added.append(dict(mask=g, score=sc, iou_pred=q, det=dict(det=1000 + k, conf=d["conf"], cls=d["cls"], centre=[(d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2], box=d["box"]),
                              fg_mean=float(fg[g].mean()), rachis_fraction=_rachis_fraction(g, rach_d, gshape), peel_rank=None, source="D1_lowconf_recovery", from_=d["src"]))
    return added


def split_target(sam, c: dict, boxes: list[dict], neighbours: np.ndarray, skel: np.ndarray, gshape, s: float, floor: int, rng) -> tuple[list[dict], dict]:
    """D2.3-D2.5 for one target; returns (parts or [], record)."""
    tm = c["mask"]; ys, xs = np.nonzero(tm); bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    rec = dict(id=c["id"], area=int(tm.sum()), n_boxes=len(boxes), solidity=round(eng._solidity(tm), 3))
    if len(boxes) < 2:
        rec["decision"] = "split_rejected: fewer_than_2_boxes"; return [], rec
    centres = np.array([b["centre"] for b in boxes]) * s

    def region(bi):
        b = boxes[bi]; x0, y0, x1, y1 = [v * s for v in b["box"]]
        x0, y0, x1, y1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
        if x1 <= x0 or y1 <= y0:
            return None
        reg = np.zeros(gshape, bool); reg[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
        return (reg, (x0, y0, x1, y1)) if (tm & reg).any() else None

    def prompt(bi, extra_neg):
        rr = region(bi)
        if rr is None:
            return None
        reg, (x0, y0, x1, y1) = rr; area = tm & reg
        pos = _dt_max(area)
        neg = [centres[[k for k in range(len(boxes)) if k != bi]]] if len(boxes) > 1 else []
        if len(neighbours):
            neg.append(neighbours)
        rs = eng._sample_points(skel & tm, N_NEG_RACHIS, rng)
        if len(rs):
            neg.append(rs)
        if len(extra_neg):
            neg.append(extra_neg)
        neg = np.vstack(neg) if neg else np.zeros((0, 2))
        g, sc, q = eng._predict(sam, [x0 / s, y0 / s, x1 / s, y1 / s], pos / s, neg / s, gshape)
        return g & tm, sc, q, reg

    first = []
    for bi in range(len(boxes)):
        r = prompt(bi, np.zeros((0, 2)))
        first.append((bi, boxes[bi]["conf"] * (r[2] if r else 0.0)))
    order = [bi for bi, _ in sorted(first, key=lambda t: -t[1])]
    parts, acc_union = [], np.zeros(gshape, bool)
    for bi in order:
        rr = region(bi)
        if rr is None:
            continue
        extra = eng._sample_points(acc_union & rr[0], N_NEG_PEEL, rng) if acc_union.any() else np.zeros((0, 2))
        r = prompt(bi, extra)
        g = r[0] & ~acc_union  # overlap goes to the earlier peeled part
        parts.append(dict(mask=g, score=r[1], iou_pred=r[2], box=boxes[bi], area=int(g.sum()), solidity=round(eng._solidity(g), 3)))
        acc_union |= g
    rec["parts"] = [dict(area=p["area"], solidity=p["solidity"], conf=round(p["box"]["conf"], 3)) for p in parts]
    why = None
    if len(parts) < 2:
        why = "fewer_than_2_parts"
    elif any(p["area"] < floor for p in parts):
        why = "part_below_floor"
    elif acc_union.sum() / max(tm.sum(), 1) < D2_COVER:
        why = "cover_below_0.85"
    elif any(_iou_m(parts[i]["mask"], parts[j]["mask"]) >= D2_PAIR_IOU for i in range(len(parts)) for j in range(i + 1, len(parts))):
        why = "pair_iou"
    elif any(p["solidity"] < rec["solidity"] - D2_SOLIDITY_DROP for p in parts):
        why = "solidity_drop"
    rec["cover"] = round(float(acc_union.sum() / max(tm.sum(), 1)), 3)
    rec["decision"] = f"split_rejected: {why}" if why else "split_accepted"
    if why:
        return [], rec
    out = []
    for n, p in enumerate(parts):
        out.append(dict(mask=p["mask"], score=p["score"], iou_pred=p["iou_pred"], det=p["box"], fg_mean=c["fg_mean"], rachis_fraction=c["rachis_fraction"],
                        peel_rank=n, source="D2_split", split_from=c["id"], id=f"{c['id']}{'abcdefghij'[n]}"))
    return out, rec


def overlay_d(res: dict, insts_meta: list[dict]) -> np.ndarray:
    out = r2.overlay(res); lab = res["_layers"]["label"]; h, w = lab.shape; yy, xx = np.mgrid[:h, :w]; cross = ((xx - yy) // 6) % 2 == 0
    for i, m in enumerate(insts_meta):
        mask = lab == i + 1
        if m.get("source") == "D2_split":
            out[mask & cross] = (out[mask & cross] * 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if m.get("source") == "D1_lowconf_recovery":
            cv2.drawContours(out, cs, -1, (40, 90, 255), 3)
        elif m.get("source") == "D2_split":
            cv2.drawContours(out, cs, -1, (0, 220, 60), 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--images", nargs="*"); a = ap.parse_args(argv)
    base, out = Path(a.base), Path(a.out); [(out / d).mkdir(parents=True, exist_ok=True) for d in ("proposals", "overlays", "layers", "d_log")]
    from tomato3d.topview.measure2d import load_sam
    sam = load_sam(r2.SAMCFG); cfg = json.load(open(r2.T1CFG)); imgs = r2.images()
    params = dict(contract=str(out / "CONTRACT.md"), base=str(base), sam_config=json.load(open(r2.SAMCFG)), t1_parameters=dict(path=str(r2.T1CFG), sha256=r2.sha(r2.T1CFG)),
                  thresholds=dict(FLOOR_PCT=FLOOR_PCT, LOWCONF=LOWCONF, BOX_IN_U=BOX_IN_U, MASK_IN_U=MASK_IN_U, FG_SHARE=FG_SHARE, D1_NMS=D1_NMS, D2_COVER=D2_COVER,
                                  D2_PAIR_IOU=D2_PAIR_IOU, D2_SOLIDITY_DROP=D2_SOLIDITY_DROP, N_NEG_RACHIS=N_NEG_RACHIS, N_NEG_PEEL=N_NEG_PEEL, RACHIS_FRAC_MAX=eng.RACHIS_FRAC_MAX),
                  held_out=list(r2.HELD_OUT), floors={})
    gate = json.load(open(out / "gateA.json")) if (out / "gateA.json").exists() else {}
    for name in [n for n in ORDER if n in imgs]:
        if (a.images and name not in a.images) or (name in gate and (out / "proposals" / f"{name}_leafseg.json").exists()):
            continue
        t0 = time.time(); rng = np.random.default_rng(0)
        doc = json.load(open(base / "proposals" / f"{name}_leafseg.json")); L = np.load(base / "layers" / f"{name}.npz")
        gshape = tuple(doc["grid_shape"]); s = doc["grid_scale"]; H, W = doc["image_size"]
        fg, skel, lab_b = L["fg"], L["skeleton"], L["label"]; rach_d = cv2.dilate(skel.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        rgb = r2.load_rgb(imgs[name]); rgb_g = cv2.resize(rgb, (gshape[1], gshape[0]), interpolation=cv2.INTER_AREA); sam.predictor.set_image(rgb)
        pred = json.load(open(base / "detections" / f"{name}.json")); dets = rfdet.to_dets(pred, (H, W)); comps = rfdet.compound_boxes(pred, (H, W))
        floor = int(np.percentile([i["visible_px"] for i in doc["instances"]], FLOOR_PCT)); params["floors"][name] = floor
        # ---- B masks as engine entries
        final = []
        for i in doc["instances"]:
            m = px.rle_decode(i["proposal_segmentation"])
            final.append(dict(id=i["id"], mask=m, score=i["score"], iou_pred=i["sam_iou_pred"], peel_rank=i["peel_rank"], fg_mean=i["fg_mean"], rachis_fraction=i["rachis_fraction"],
                              det=dict(det=None, conf=i["detector_conf"], cls=i["detector_class"], centre=None), source="model_proposal", flags_b=i["flags"]))
        # det index for each B mask: the run 02 box whose conf matches and whose centre is inside the mask (closest to centroid)
        for c in final:
            inside = [d for d in dets if abs(d["conf"] - c["det"]["conf"]) < 1e-6]
            if not inside:
                inside = [d for d in dets if c["mask"][min(int(d["centre"][1] * s), gshape[0] - 1), min(int(d["centre"][0] * s), gshape[1] - 1)]]
            d = inside[0] if inside else dict(det=-1, conf=c["det"]["conf"], cls=c["det"]["cls"], centre=[float(np.nonzero(c["mask"])[1].mean() / s), float(np.nonzero(c["mask"])[0].mean() / s)], box=None)
            c["det"] = dict(d)
        # ---- D1
        U = fg & (lab_b == 0); ncc, cc, st, _ = cv2.connectedComponentsWithStats(U.astype(np.uint8), 8)
        keep_cc = np.zeros(ncc, bool); keep_cc[1:] = st[1:, cv2.CC_STAT_AREA] >= floor; U = keep_cc[cc]
        cands = low_conf_boxes(base, name, (H, W)); d1log = [dict(floor=floor, n_components=int(keep_cc.sum()), n_candidates=len(cands))]
        added = recover(sam, cands, [c["mask"] for c in final], U, fg, rach_d, gshape, s, floor, d1log)
        # ---- D2
        centres_g = np.array([d["centre"] for d in dets]) * s
        d2log, new_final = [], []
        for c in final:
            m = c["mask"]; ys, xs = np.nonzero(m); bb = (xs.min(), ys.min(), xs.max(), ys.max())
            cin = [k for k, (cx, cy) in enumerate(centres_g) if 0 <= int(cy) < gshape[0] and 0 <= int(cx) < gshape[1] and m[int(cy), int(cx)]]
            target = "merged" in c["flags_b"] or len(cin) >= 2
            if not target:
                new_final.append(c); continue
            nb = np.array([centres_g[k] for k in range(len(dets)) if k not in cin and bb[0] <= centres_g[k][0] <= bb[2] and bb[1] <= centres_g[k][1] <= bb[3]]).reshape(-1, 2)
            parts, rec = split_target(sam, c, [dets[k] for k in cin], nb, skel, gshape, s, floor, rng)
            d2log.append(rec)
            new_final.extend(parts if parts else [c])
        for c in added:
            c["id"] = None
        allm = new_final + added
        nb_ = sum(1 for c in allm if c["source"] != "D1_lowconf_recovery")
        for k, c in enumerate([c for c in allm if c["source"] == "D1_lowconf_recovery"]):
            c["id"] = f"R{k + 1:03d}"
        rank_of = {id(c): (0 if c["source"] != "D1_lowconf_recovery" else 1) for c in allm}
        rach = dict(skeleton=skel, ends=[], branches=[])
        res = eng.finish(allm, dets, rgb_g, fg, rach, cfg, gshape, s, (H, W), rank_key=lambda i: (rank_of[id(allm[i])], -allm[i]["score"]))
        res["rachis"] = doc["rachis"]; res["detections"] = doc["detections"]; res["candidates"] = doc["candidates"]; res["dropped"] = doc["dropped"]
        for inst, c in zip(res["instances"], allm):
            inst["source"] = c["source"]; inst["split_from"] = c.get("split_from"); inst["from"] = c.get("from_")
        rfdet.assign_compound_boxes(res["_layers"]["label"], s, res["instances"], comps); res["compound_boxes"] = comps
        ov = overlay_d(res, allm); Image.fromarray(ov).save(out / "overlays" / f"{name}.jpg", quality=88)
        Lr = res.pop("_layers"); np.savez_compressed(out / "layers" / f"{name}.npz", label=Lr["label"].astype(np.int16), fg=fg, skeleton=skel)
        json.dump(dict(d1=d1log, d2=d2log), open(out / "d_log" / f"{name}.json", "w"), indent=1, default=float)
        doc_out = dict(schema="leafseg-1", image=dict(doc["image"], variant="D"), source="model_proposal",
                       note="Variant D: run 02 B masks + D1 low-confidence recovery + D2 merged splits; RLE on the working grid; no accuracy claim before human review", **res)
        json.dump(doc_out, open(out / "proposals" / f"{name}_leafseg.json", "w"), default=float)
        insts = res["instances"]; areas = np.array([i["visible_px"] for i in insts]); d1a = [c["mask"].sum() for c in added]
        gate[name] = dict(split=doc["image"]["split"], detections=doc["detections"], leaflets=len(insts), occluded=sum(i["occluded"] for i in insts),
                          unassigned_area_fraction=round(res["unassigned_area_fraction"], 3), order_edges=len(res["order_edges"]),
                          compounds=len({i["compound_leaf_id"] for i in insts} - {"unassigned"}), compound_unassigned=sum(i["compound_leaf_id"] == "unassigned" for i in insts),
                          compound_box_unassigned=sum(i["compound_box_id"] == "unassigned" for i in insts),
                          faces={f: sum(i["face"] == f for i in insts) for f in ("adaxial", "abaxial", "folded", "unknown")},
                          variety={v: sum(i["variety_group"] == v for i in insts) for v in ("silvery", "green")},
                          failure_candidates=dict(merged=[i["id"] for i in insts if "merged" in i["flags"]], split_folded=res["split_folded_candidates"],
                                                  rachis_kept_as_leaf_dropped=0, truncated_by_image=[i["id"] for i in insts if i["truncated_by_image"]]),
                          share_lt_1500px=round(float((areas < 1500).mean()), 3) if len(areas) else None, floor_px=floor,
                          d1=dict(candidates=len(cands), added=len(added), median_area=int(np.median(d1a)) if d1a else None,
                                  rejected={r: sum(1 for x in d1log[1:] if x["decision"] == f"rejected: {r}") for r in ("empty_box", "box_outside_unassigned", "too_small", "steals_from_B", "low_foreground", "nms", "rachis_kept_as_leaf")}),
                          d2=dict(attempted=len(d2log), accepted=sum(1 for x in d2log if x["decision"] == "split_accepted"),
                                  rejected={x["decision"].split(": ")[1]: sum(1 for y in d2log if y["decision"] == x["decision"]) for x in d2log if x["decision"] != "split_accepted"}),
                          seconds=round(time.time() - t0, 1))
        print(name, {k: v for k, v in gate[name].items() if k not in ("failure_candidates", "faces", "variety")}, flush=True)
        json.dump(gate, open(out / "gateA.json", "w"), indent=1); json.dump(params, open(out / "parameters.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
