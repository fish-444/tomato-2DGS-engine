"""Existing proposers before the human for the Step-6 correction (contract CONTRACT_LEAFLET_2D_SEED.md, Revision 3).

P-H2 existing H2(M1) proposals; P-RF Roboflow tomato_top_nonboard v4 boxes -> leafseg engine run_image (run 02 variant B, settings
unchanged); P-PT SAM v5 point split of merge_suspect masks; P-WS Revision 2 watershed (only on views with < 17 H2 instances).
Consensus at IoU >= 0.6 between different proposers -> auto / single_source / contested. Step A re-runs the frozen Step-2 vote with
the declared views' auto instances. Nothing is trained; SAM runs on local CPU as in run 02. The Roboflow key is read into memory
from api.txt (roboflow2 line) or ROBOFLOW_API_KEY and never printed, logged or written.

usage: python -m tomato3d.organs.auto_proposers [--step proposers|vote|all] [--out DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.leaflib import outline as ol
from tomato3d.leafseg import pixels as px
from tomato3d.organs import light_organs as LO
from tomato3d.organs import seed2d as S

ROOT = LO.ROOT
OUT = S.OUT / "attempt5_auto"
DECLARED = ["060313903", "060355657", "060440865", "060547546"]            # contract section 6 / Revision 3
PHOTOS = LO.DEFAULTS["poses"].parent / "undist_3024"
FG_MASKS = ROOT / "build/p1_recon_charuco_20261021_01/masks"
GRID = S.H2_GRID
CONS = dict(agree_iou=0.6, band_px=3)
PT = dict(sigma_px=2.0, min_distance_px=12, min_height_px=6.0, child_inside_min=0.8, cover_min=0.85, child_iou_max=0.2)
RF_CAP = 300


# ---------------------------------------------------------------- the key: memory only
def _rf_key() -> str | None:
    k = os.environ.get("ROBOFLOW_API_KEY")
    if k:
        return k
    p = ROOT.parents[1] / "api.txt"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().lower().startswith("roboflow2"):
            m = re.search(r"key\s*:\s*([A-Za-z0-9_\-]+)", line)
            return m.group(1) if m else None
    return None


def _scrub(msg: str, key: str | None) -> str:
    return msg.replace(key, "***") if key else msg


# ---------------------------------------------------------------- helpers
def iou(a, b) -> float:
    i = np.logical_and(a, b).sum()
    return float(i / max(np.logical_or(a, b).sum(), 1)) if i else 0.0


def bbox(m):
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None


def boxes_touch(a, b) -> bool:
    return a is not None and b is not None and not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def view_file(vid: str) -> str:
    return f"20261021_{vid}_iOS.jpg"


def plant_centre(masks):
    pts = [np.argwhere(m).mean(0)[::-1] for m in masks if m.any()]
    return np.array(pts).mean(0) if pts else None


def suspect_score(m, crit, centre):
    rec = ol.mask_outline(m, base_cue_px=None, centre_px=centre)
    if rec is None or any(rec[f] is None for f in S.FEATS):
        return dict(merge_suspect=False, p_merged=None, note="too_small_for_outline")
    return crit.score(rec)


@contextlib.contextmanager
def fixed_foreground(plant_grid):
    """Run 02's engine with the plant mask as foreground instead of its tray-trained pixel classifier (declared difference)."""
    orig = px.foreground_prob
    px.foreground_prob = lambda rf, F, shape: cv2.resize(plant_grid.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    try:
        yield
    finally:
        px.foreground_prob = orig


# ---------------------------------------------------------------- proposers
def p_h2(vid):
    j = json.load(open(S.OUT / "leaflet_instances" / (Path(view_file(vid)).stem + ".json"), encoding="utf-8"))
    return [dict(uid=f"H2:{x['id']}", prop="P-H2", mask=px.rle_decode(x["rle"]), score=float(x.get("score") or 0.0)) for x in j["instances"]]


def p_rf(vid, rgb, sam, cfg, key, out: Path, log):
    from tomato3d.leafseg import engine as eng
    from tomato3d.leafseg import rfdet
    path = PHOTOS / view_file(vid); cache = out / "rf_detections" / f"{vid}.json"; t0 = time.time(); calls = 0
    try:
        if not cache.exists():
            calls += 1
        pred = rfdet.fetch(path, rgb, cache, key); n_pred = len(pred.get("predictions", []))
        tiled = None
        if n_pred >= RF_CAP:
            dets, tiled = rfdet.tiled_dets(path, rgb, out / "rf_detections_tiled", vid, key); calls += tiled["tiles"]
        else:
            dets = rfdet.to_dets(pred, rgb.shape[:2])
    except Exception as e:                                                    # never let the key reach a log
        raise RuntimeError(_scrub(f"Roboflow call failed for {vid}: {type(e).__name__}: {e}", key)) from None
    t_rf = time.time() - t0
    plant = np.asarray(Image.open(FG_MASKS / (Path(view_file(vid)).stem + ".png")).resize((GRID, GRID), Image.NEAREST)) == 1
    t1 = time.time()
    with fixed_foreground(plant):
        res = eng.run_image(rgb, None, sam, None, cfg, dets=dets)
    out_m = [dict(uid=f"RF:{x['id']}", prop="P-RF", mask=px.rle_decode(x["visible_segmentation"]), score=float(x.get("score") or 0.0), det_class=x.get("detector_class")) for x in res["instances"]]
    log[vid] = dict(rf_calls_live=calls, rf_predictions=n_pred, rf_leaflet_boxes=len(dets), tiled=tiled, rf_seconds=round(t_rf, 1), sam_seconds=round(time.time() - t1, 1),
                    masks=len(out_m), dropped=dict(Counter(d["reason"] for d in res.get("dropped", []))))
    return out_m


def p_pt(parent, rgb_scale, sam, gshape):
    """Point-prompted split of one mask; returns (parts, info). Parts are clipped to the parent; overlaps go to the higher score."""
    from skimage.feature import peak_local_max
    from tomato3d.leafseg import engine as eng
    m = parent["mask"]; dt = cv2.GaussianBlur(cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5), (0, 0), PT["sigma_px"])
    pk = peak_local_max(dt, min_distance=PT["min_distance_px"], threshold_abs=PT["min_height_px"], labels=m.astype(int))
    info = dict(parent=parent["uid"], maxima=int(len(pk)))
    if len(pk) < 2:
        info["result"] = "fewer_than_2_maxima"; return [], info
    b = bbox(m); box_full = np.array([b[0], b[1], b[2], b[3]], float) / rgb_scale; pts = pk[:, ::-1].astype(np.float32) / rgb_scale
    kids = []
    for i in range(len(pts)):
        r = eng._predict(sam, box_full, pts[i:i + 1], np.delete(pts, i, 0), gshape)
        if r is not None and r[0].any():
            kids.append(dict(mask=r[0], score=float(r[1])))
    ok = len(kids) >= 2 and all((k["mask"] & m).sum() / max(k["mask"].sum(), 1) >= PT["child_inside_min"] for k in kids)
    if ok:
        u = np.zeros_like(m)
        for k in kids:
            u |= k["mask"] & m
        ok = u.sum() / max(m.sum(), 1) >= PT["cover_min"] and all(iou(a["mask"], b_["mask"]) <= PT["child_iou_max"] for i, a in enumerate(kids) for b_ in kids[i + 1:])
    if not ok:
        info["result"] = "no_tiling"; info["children"] = len(kids); return [], info
    taken = np.zeros_like(m); parts = []
    for q, k in enumerate(sorted(kids, key=lambda k: -k["score"])):
        mm = k["mask"] & m & ~taken; taken |= mm
        if mm.any():
            parts.append(dict(uid=f"PT:{parent['uid']}#{q + 1}", prop="P-PT", mask=mm, score=k["score"], parent=parent["uid"], parent_prop=parent["prop"]))
    info["result"] = "split"; info["parts"] = len(parts); return parts, info


def p_ws(vid):
    """Revision 2 watershed, only on views with < h2_min_instances H2 proposals."""
    regs = S.watershed_proposals([view_file(vid)], LO.DEFAULTS["masks"]).get(view_file(vid), [])
    return [dict(uid=f"WS:{rid}", prop="P-WS", mask=m, score=0.0) for rid, m in regs]


# ---------------------------------------------------------------- consensus
def components(pool):
    n = len(pool); parent = list(range(n)); bxs = [bbox(p["mask"]) for p in pool]; edges = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pool[i], pool[j]
            if a["prop"] == b["prop"] or not boxes_touch(bxs[i], bxs[j]):
                continue
            if a.get("parent") == b["uid"] or b.get("parent") == a["uid"]:
                continue
            v = iou(a["mask"], b["mask"])
            if v >= CONS["agree_iou"]:
                edges.append((i, j, v)); parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values()), edges


def consensus(masks_by_prop, suspects: set, parts_of: dict):
    """masks_by_prop: proposer -> masks; suspects: uids of merge_suspect H2/RF masks; parts_of: parent uid -> P-PT parts."""
    base = [m for p in ("P-H2", "P-RF", "P-WS") for m in masks_by_prop.get(p, []) if m["uid"] not in suspects]
    allowed = {k: v for k, v in parts_of.items() if v}
    for _ in range(4):                                      # drop parents whose parts are not all confirmed, until stable
        pool = base + [q for v in allowed.values() for q in v]
        groups, edges = components(pool)
        comp_of = {pool[i]["uid"]: g for g in groups for i in g}
        bad = []
        for par, parts in allowed.items():
            pprop = parts[0]["parent_prop"]
            for q in parts:
                g = comp_of[q["uid"]]
                if not any(pool[i]["prop"] not in ("P-PT", pprop) for i in g):
                    bad.append(par); break
        if not bad:
            break
        for par in bad:
            allowed.pop(par)
    auto, single = [], []
    for g in groups:
        members = [pool[i] for i in g]
        if len({m["prop"] for m in members}) >= 2:
            inter = members[0]["mask"].copy()
            for m in members[1:]:
                inter &= m["mask"]
            hi = max(members, key=lambda m: m["score"])["mask"]; k = 2 * CONS["band_px"] + 1
            band = hi & ~cv2.erode(hi.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
            auto.append(dict(mask=inter | band, sources=sorted(m["uid"] for m in members), props=sorted({m["prop"] for m in members}),
                             score=float(max(m["score"] for m in members)), iou_min=float(min([e[2] for e in edges if e[0] in g and e[1] in g] or [0]))))
        else:
            single.extend(members)
    all_masks = {m["uid"]: m for p in masks_by_prop.values() for m in p}
    contested = [dict(all_masks[u], reason=("split_unconfirmed" if parts_of.get(u) else "no_tiling_split")) for u in suspects if u not in allowed]
    auto.sort(key=lambda a: -a["score"])
    for k, a in enumerate(auto):
        a["id"] = f"A{k + 1:03d}"
    return auto, single, contested, dict(edges=len(edges), replaced_parents=sorted(allowed))


def agreement_ious(masks_by_prop):
    """Per proposer pair: for each mask of the first, best IoU with any mask of the second (and reverse)."""
    out = {}; props = [p for p in ("P-H2", "P-RF", "P-PT", "P-WS") if masks_by_prop.get(p)]
    for i, a in enumerate(props):
        for b in props[i + 1:]:
            best = []
            for x, y in ((a, b), (b, a)):
                for m in masks_by_prop[x]:
                    bx = bbox(m["mask"]); cand = [iou(m["mask"], n["mask"]) for n in masks_by_prop[y] if boxes_touch(bx, bbox(n["mask"]))]
                    best.append(max(cand) if cand else 0.0)
            v = np.array(best); out[f"{a}~{b}"] = dict(n=int(len(v)), median=float(np.median(v)), p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)), share_ge_0p6=float((v >= 0.6).mean()), values=[round(float(x), 3) for x in v])
    return out


# ---------------------------------------------------------------- run
def run_proposers(out: Path):
    from tomato3d.leafseg import run as r2
    from tomato3d.topview.measure2d import load_sam
    out.mkdir(parents=True, exist_ok=True); key = _rf_key()
    if not key:
        raise SystemExit("no Roboflow key available (ROBOFLOW_API_KEY unset and no roboflow2 line in api.txt)")
    sam = load_sam(r2.SAMCFG); cfg = json.load(open(r2.T1CFG)); crit = S.Criterion(S.library_rows()); rflog, summ = {}, {}
    for vid in DECLARED:
        t0 = time.time(); rgb = np.asarray(Image.open(PHOTOS / view_file(vid)).convert("RGB")); s = GRID / max(rgb.shape[:2]); gshape = (GRID, GRID)
        props = {"P-H2": p_h2(vid)}
        props["P-RF"] = p_rf(vid, rgb, sam, cfg, key, out, rflog)
        n_h2 = len(props["P-H2"]); props["P-WS"] = p_ws(vid) if n_h2 < S.WS["h2_min_instances"] else []
        centre = plant_centre([m["mask"] for m in props["P-H2"]])
        for p in ("P-H2", "P-RF"):
            for m in props[p]:
                m.update(suspect_score(m["mask"], crit, centre))
        suspects = {m["uid"] for p in ("P-H2", "P-RF") for m in props[p] if m.get("merge_suspect")}
        sam.predictor.set_image(rgb); parts_of, pt_info = {}, []
        for p in ("P-H2", "P-RF"):
            for m in props[p]:
                if m["uid"] in suspects:
                    parts, info = p_pt(m, s, sam, gshape); parts_of[m["uid"]] = parts; pt_info.append(info)
        props["P-PT"] = [q for v in parts_of.values() for q in v]
        auto, single, contested, cinfo = consensus(props, suspects, parts_of)
        agree = agreement_ious(props)
        enc = lambda m: dict(size=[GRID, GRID], counts=S.rle_encode(m))
        json.dump(dict(view=view_file(vid), status="R3-1b_unaccepted", note="proposer agreement, not accuracy; nothing here is human truth", consensus=CONS, pt=PT,
                       auto=[dict(id=a["id"], sources=a["sources"], props=a["props"], score=a["score"], iou_min=a["iou_min"], rle=enc(a["mask"])) for a in auto],
                       single_source=[dict(uid=m["uid"], prop=m["prop"], score=m["score"], merge_suspect=bool(m.get("merge_suspect")), rle=enc(m["mask"])) for m in single],
                       contested=[dict(uid=m["uid"], prop=m["prop"], score=m["score"], reason=m["reason"], p_merged=m.get("p_merged"), rle=enc(m["mask"])) for m in contested],
                       pt_splits=pt_info, consensus_info=cinfo), open(out / f"consensus_{vid}.json", "w", encoding="utf-8"))
        summ[vid] = dict(counts={p: len(v) for p, v in props.items()}, suspects=len(suspects), auto=len(auto), single_source=len(single), contested=len(contested),
                         auto_by_props=dict(Counter("+".join(a["props"]) for a in auto)), pt=dict(Counter(i["result"] for i in pt_info)),
                         agreement={k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in agree.items()}, seconds=round(time.time() - t0, 1))
        json.dump(dict(per_view=summ, roboflow=rflog, agreement_values={vid: agree}), open(out / f"proposers_{vid}.json", "w", encoding="utf-8"), indent=1, default=float)
        print(vid, json.dumps(summ[vid], default=float), flush=True)
    json.dump(dict(status="R3-1b_unaccepted", scale=LO.SCALE_TAG, gpu_hours=0.0, sam_device="cpu (as run 02)", declared_views=DECLARED, consensus=CONS, pt=PT,
                   per_view=summ, roboflow=dict(per_view=rflog, live_calls=sum(v["rf_calls_live"] for v in rflog.values()), seconds=round(sum(v["rf_seconds"] for v in rflog.values()), 1)),
                   note="agreement between proposers is not accuracy"), open(out / "proposers_summary.json", "w", encoding="utf-8"), indent=1, default=float)
    return summ


def run_vote(out: Path):
    """Step A: the frozen Step-2 vote with the declared views' instances = auto only."""
    S.OVERRIDE.clear()
    for vid in DECLARED:
        d = json.load(open(out / f"consensus_{vid}.json", encoding="utf-8"))
        S.OVERRIDE[view_file(vid)] = [(a["id"], px.rle_decode(a["rle"]), "auto:" + "+".join(a["props"])) for a in d["auto"]]
    sub = out / "auto_only"; sub.mkdir(parents=True, exist_ok=True); views = json.load(open(LO.DEFAULTS["views"]))["train"]; S.copy_instances(views, sub)
    a = argparse.Namespace(surfels=LO.DEFAULTS["surfels"], track_a=S.TRACK_A, poses=LO.DEFAULTS["poses"], views=LO.DEFAULTS["views"], out=sub, step1b=True, human_masks=None,
                           visibility="splat1512", pot_top=LO.POT_TOP_Z, root=list(S.C.POT_CENTRE_M), group_order="current", erode_px=0, watershed=False, masks=LO.DEFAULTS["masks"],
                           input_status="R3-1b_unaccepted")
    R = S.compute(a); R2 = S.compute(a)
    extra = dict(declared_views=DECLARED, auto_instances={vid: len(S.OVERRIDE[view_file(vid)]) for vid in DECLARED},
                 reproducible=bool((R["cl"] == R2["cl"]).all()))
    summ = S.write_outputs(R, a, sub, "R3-1b_unaccepted", "auto_only", extra); S.OVERRIDE.clear()
    print(json.dumps(dict(leaflets=summ["leaflets"], compound_leaves=summ["compound_leaves"], acceptance=summ["acceptance"], buckets=summ["buckets"]), default=float), flush=True)
    return summ


# ---------------------------------------------------------------- Step B: prefilled correction page + export converter
PAGE_PX = 1200          # = light_review.PAGE_PX; seed2d.view_instances maps page px -> grid with this width


def _poly(m, sc):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True).reshape(-1, 2) * sc
    return np.round(c, 1).tolist()


def build_page(out: Path):
    import base64
    import io
    sc = PAGE_PX / GRID; views, expected = [], {}
    for vid in DECLARED:
        d = json.load(open(out / f"consensus_{vid}.json", encoding="utf-8"))
        img = Image.open(PHOTOS / view_file(vid)).convert("RGB"); img.thumbnail((PAGE_PX, PAGE_PX)); buf = io.BytesIO(); img.save(buf, "JPEG", quality=80)
        items = []
        for a in d["auto"]:
            items.append(dict(id=a["id"], state="accepted", badges=[p.replace("P-", "") for p in a["props"]], suspect=False, pts=_poly(px.rle_decode(a["rle"]), sc)))
        for k, m in enumerate(d["single_source"]):
            items.append(dict(id=f"S{k + 1:03d}", src=m["uid"], state="pending", why="single_source", badges=[m["prop"].replace("P-", "")], suspect=bool(m.get("merge_suspect")), pts=_poly(px.rle_decode(m["rle"]), sc)))
        for k, m in enumerate(d["contested"]):
            items.append(dict(id=f"C{k + 1:03d}", src=m["uid"], state="pending", why="contested", badges=[m["prop"].replace("P-", "")], suspect=True, pts=_poly(px.rle_decode(m["rle"]), sc)))
        items = [i for i in items if i["pts"] and len(i["pts"]) >= 3]
        expected[vid] = sum(i["state"] == "pending" for i in items)
        views.append(dict(id=vid, w=img.size[0], h=img.size[1], jpg=base64.b64encode(buf.getvalue()).decode(), items=items))
    data = dict(views=views, expected_decisions=expected, page_px=PAGE_PX, grid=GRID)
    html = PAGE_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (out / "review").mkdir(parents=True, exist_ok=True); p = out / "review" / "prefilled_correction_page.html"; p.write_text(html, encoding="utf-8")
    json.dump(dict(expected_decisions=expected, total=sum(expected.values()), accepted={v["id"]: sum(i["state"] == "accepted" for i in v["items"]) for v in views}),
              open(out / "review" / "page_inputs.json", "w", encoding="utf-8"), indent=1)
    return p, expected


LABEL_VOTES = ("whole", "occluded")           # page labels that keep an instance in the vote (occluded = visible part only)
LABEL_EXCLUDES = ("fragment", "not_leaflet")   # page labels that exclude it (declared with the labels, before any answer)


def export_to_human_masks(export: dict, out: Path, labels: dict | None = None) -> dict:
    """Page decisions -> the existing mask_corrections op format (relative to the H2 base): every H2 proposal on a declared view
    is deleted, every final instance (untouched auto = keep, pending only when kept, cuts applied) is added as a polygon in
    page px. `labels` = export of the page's instance_labels collection {view|id: {label}}: whole / occluded keep the instance
    (a pending one becomes kept), fragment / not_leaflet drop it. `seed2d --human-masks <file> --repeat` then runs unchanged."""
    sc = PAGE_PX / GRID; ops = {}; labels = labels or {}
    for vid in DECLARED:
        d = json.load(open(out / f"consensus_{vid}.json", encoding="utf-8")); j = json.load(open(S.OUT / "leaflet_instances" / (Path(view_file(vid)).stem + ".json"), encoding="utf-8"))
        for x in j["instances"]:
            ops[f"{vid}|{x['id']}"] = dict(op="delete", view=vid, id=x["id"])
        base = [(a["id"], px.rle_decode(a["rle"]), "accepted") for a in d["auto"]] + [(f"S{k + 1:03d}", px.rle_decode(m["rle"]), "pending") for k, m in enumerate(d["single_source"])] + \
               [(f"C{k + 1:03d}", px.rle_decode(m["rle"]), "pending") for k, m in enumerate(d["contested"])]
        for iid, m, state in base:
            dec = export.get(f"{vid}|{iid}", {}); op = dec.get("op"); lab = (labels.get(f"{vid}|{iid}") or {}).get("label")
            if op == "delete" or lab in LABEL_EXCLUDES:
                continue
            if state == "pending" and op not in ("keep", "cut") and lab not in LABEL_VOTES:
                continue
            parts = [m]
            if op == "cut":
                mm = m.astype(np.uint8).copy()
                for c in dec.get("cuts", []):
                    cv2.polylines(mm, [(np.asarray(c, float) / sc).round().astype(np.int32).reshape(-1, 1, 2)], False, 0, thickness=3)
                n, lab = cv2.connectedComponents(mm, connectivity=4); parts = [lab == q for q in range(1, n) if (lab == q).sum() >= ol.MIN_AREA_PX]
            for q, part in enumerate(parts):
                pts = _poly(part, sc)
                if pts and len(pts) >= 3:
                    ops[f"{vid}|ADD_{iid}_{q + 1}"] = dict(op="add", view=vid, points=pts, from_auto_page=iid, **({"label": lab} if lab else {}))
        for k, o in export.items():
            if o.get("op") == "add" and o.get("view") == vid:
                ops[k] = dict(op="add", view=vid, points=o["points"])
    return ops


PAGE_HTML = r"""<title>R3-1b 자동 제안 교정</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: photo stage (wheel zoom, drag pan) + decision rail; stacks at phone width */
:root{--bg:#f4f5f2;--fg:#1d2220;--mut:#66706b;--panel:#fff;--line:#d6d9d3;--acc:#2f6f4f;--pend:#c27c0e;--bad:#b23a3a;--chip:#e9ebe6;--font:"IBM Plex Sans KR",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141816;--fg:#e4e8e5;--mut:#98a29d;--panel:#1c211f;--line:#323a36;--acc:#5fae86;--pend:#e0a43c;--bad:#e06c6c;--chip:#262c29;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141816;--fg:#e4e8e5;--mut:#98a29d;--panel:#1c211f;--line:#323a36;--acc:#5fae86;--pend:#e0a43c;--bad:#e06c6c;--chip:#262c29;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.5 var(--font);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:12px}
h1{font-size:18px;margin:0 0 4px;text-wrap:balance}.sub{color:var(--mut);font-size:12px;margin:0 0 8px}
.banner{background:#7a1f1f;color:#fff;border-radius:8px;padding:6px 10px;font-size:12px;margin-bottom:10px}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:14px}@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.stage{min-width:0}.row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
button{background:var(--chip);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:5px 9px;font:inherit;font-size:12px;cursor:pointer}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
button.lb{border-color:var(--acc)}button.lb.on{background:var(--acc)}
.photo{position:relative;width:100%;border-radius:8px;overflow:hidden;background:#000;touch-action:none;cursor:grab}
#zw{position:relative;transform-origin:0 0}
.photo img{display:block;width:100%;height:auto;user-select:none;-webkit-user-drag:none}.photo svg{position:absolute;inset:0;width:100%;height:100%}
polygon,polyline{vector-effect:non-scaling-stroke}polygon{stroke-width:2;cursor:pointer}polygon.sel{stroke-width:4}
svg text{font-family:var(--mono);paint-order:stroke;stroke:#000;stroke-width:3px;fill:#fff;pointer-events:none}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:10px}.card h2{font-size:12px;margin:0 0 6px;color:var(--mut);letter-spacing:.04em;text-transform:uppercase}
.k{font-family:var(--mono);font-variant-numeric:tabular-nums}.list{max-height:280px;overflow:auto;font-size:12px}.list div{display:flex;gap:6px;align-items:center;padding:2px 0;border-bottom:1px dashed var(--line);cursor:pointer}
.b{font-size:10px;font-family:var(--mono);border:1px solid var(--line);border-radius:4px;padding:0 3px}.pend{color:var(--pend)}.acc{color:var(--acc)}.del{color:var(--bad)}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&family=IBM+Plex+Mono&display=swap">
<h1>R3-1b 자동 제안 교정</h1>
<p class="sub">선언된 학습 뷰 4장. 초록 = 두 제안기 이상이 합의한 소엽(그대로 두면 유지). 주황 = 대기(한 제안기만 찾았거나 병합 의심) — <b>결정하지 않으면 투표하지 않습니다</b>. 합의는 정확도가 아니라 제안기끼리 같은 것을 찾았다는 뜻입니다.</p>
<div class="banner">R3-1b_unaccepted · 형태만 · 척도 미검증 · 사람 결정만 저장, 자동 결과는 바뀌지 않음</div>
<div class="wrap"><div class="stage">
<div class="row" id="tabs"></div>
<div class="row"><button id="mSel" class="on">선택</button><button id="mKeep">유지</button><button id="mDel">삭제</button><button id="mCut">자르기</button><button id="mAdd">추가</button><button id="done">선 완료</button><button id="undo">점 취소</button></div>
<div class="row" id="lbrow"><span class="sub" style="margin:0">고른 인스턴스 분류</span><button class="lb" data-lb="occluded">1 가려짐</button><button class="lb" data-lb="whole">2 온전</button><button class="lb" data-lb="fragment">3 조각</button><button class="lb" data-lb="not_leaflet">4 소엽 아님</button><button data-lb="">분류 지움</button></div>
<p class="sub" style="margin-top:-2px">가려짐·온전 = 유지(투표 포함) · 조각·소엽 아님 = 제외(투표 안 함). 키보드 1–4.</p>
<div class="row"><button id="zin">＋ 확대</button><button id="zout">－ 축소</button><button id="zreset">맞춤</button><span class="k" id="zv">100%</span><span class="sub" style="margin:0">휠 = 확대 · 드래그 = 이동 · 키 + − 0</span></div>
<div class="photo" id="photo"><div id="zw"><img id="img" alt="declared view"><svg id="ov"></svg></div></div>
<p class="sub" id="hint"></p></div>
<div><div class="card"><h2>진행</h2><div>대기 결정 <span class="k" id="pd"></span> / <span class="k" id="pt"></span> · 분류 <span class="k" id="lc"></span> · 경과 <span class="k" id="clock">00:00</span></div><div class="row" style="margin-top:6px"><button id="finish">완료 저장</button></div><p class="sub" id="store"></p></div>
<div class="card"><h2>이 뷰의 인스턴스</h2><div class="list" id="list"></div></div></div></div>
<script id="D" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('D').textContent);const S={v:0,mode:'sel',sel:null,draft:[],ops:{},lb:{},t0:Date.now(),shown:-1};let db=null;const $=id=>document.getElementById(id);
const key=(v,i)=>v+'|'+i;const LB={occluded:'가려짐',whole:'온전',fragment:'조각',not_leaflet:'소엽 아님'};const LK=['occluded','whole','fragment','not_leaflet'];
const Z={k:1,x:0,y:0,drag:false,sup:false};
async function putc(col,k,d){if(db){try{if(d===null)await db.doc(col+'/'+k).delete();else await db.doc(col+'/'+k).set(Object.assign({at:new Date().toISOString()},d));$('store').textContent='저장: artifact db';}catch(e){$('store').textContent='저장 실패: '+(e.code||e.message);}}else{try{localStorage.setItem('ap_ops',JSON.stringify(S.ops));localStorage.setItem('ap_lb',JSON.stringify(S.lb));}catch(e){}}}
const put=(k,d)=>putc('mask_corrections',k,d);
function lab(V,it){const l=S.lb[key(V.id,it.id)];return l&&l.label||'';}
function status(V,it){const k=key(V.id,it.id);const o=S.ops[k],l=lab(V,it);if(o&&o.op==='delete')return 'deleted';if(l==='fragment'||l==='not_leaflet')return 'excluded';if(l==='whole'||l==='occluded')return 'kept';if(o&&(o.op==='keep'||o.op==='cut'))return 'kept';return it.state==='accepted'?'accepted':'pending';}
function col(st){return st==='accepted'||st==='kept'?'#2fa36b':(st==='deleted'||st==='excluded')?'#888':'#e0a43c';}
function clampPan(){const W=$('photo').clientWidth,H=$('zw').offsetHeight;Z.x=Math.min(0,Math.max(W*(1-Z.k),Z.x));Z.y=Math.min(0,Math.max(H*(1-Z.k),Z.y));}
function applyZ(){clampPan();$('zw').style.transform='translate('+Z.x+'px,'+Z.y+'px) scale('+Z.k+')';$('zv').textContent=Math.round(Z.k*100)+'%';render();}
function zoomAt(f,px,py){const k2=Math.min(10,Math.max(1,Z.k*f));Z.x=px-(px-Z.x)*(k2/Z.k);Z.y=py-(py-Z.y)*(k2/Z.k);Z.k=k2;applyZ();}
function zoomCentre(f){zoomAt(f,$('photo').clientWidth/2,$('zw').offsetHeight/2);}
function zoomReset(){Z.k=1;Z.x=0;Z.y=0;applyZ();}
function render(){const V=D.views[S.v];if(S.shown!==S.v){$('img').src='data:image/jpeg;base64,'+V.jpg;S.shown=S.v;}$('ov').setAttribute('viewBox','0 0 '+V.w+' '+V.h);let s='';
 const fs=12/(($('photo').clientWidth||V.w)/V.w)/Z.k;
 V.items.forEach((it,i)=>{const st=status(V,it);const c=col(st);const l=lab(V,it);s+=`<polygon data-i="${i}" class="${S.sel===it.id?'sel':''}" points="${it.pts.map(p=>p.join(',')).join(' ')}" fill="${c}" fill-opacity="${st==='deleted'||st==='excluded'?0.08:0.25}" stroke="${it.suspect?'#000':c}" stroke-dasharray="${it.suspect||st==='deleted'||st==='excluded'?'6 4':l==='fragment'?'2 3':'none'}"><title>${it.id} · ${it.badges.join('+')}${it.suspect?' · M':''}${l?' · '+LB[l]:''} · ${st}</title></polygon>`;
  if(l){const cx=it.pts.reduce((a,p)=>a+p[0],0)/it.pts.length,cy=it.pts.reduce((a,p)=>a+p[1],0)/it.pts.length;s+=`<text x="${cx}" y="${cy}" font-size="${fs}" text-anchor="middle">${LB[l]}</text>`;}
  const o=S.ops[key(V.id,it.id)];if(o&&o.cuts)o.cuts.forEach(c=>{s+=`<polyline points="${c.map(p=>p.join(',')).join(' ')}" fill="none" stroke="#ff3030" stroke-width="3"/>`;});});
 Object.entries(S.ops).filter(([k,o])=>o.op==='add'&&o.view===V.id).forEach(([k,o])=>{s+=`<polygon points="${o.points.map(p=>p.join(',')).join(' ')}" fill="#4090ff" fill-opacity=".25" stroke="#4090ff"/>`;});
 if(S.draft.length)s+=`<polyline points="${S.draft.map(p=>p.join(',')).join(' ')}" fill="none" stroke="#ff3030" stroke-width="3" stroke-dasharray="4 3"/>`;
 $('ov').innerHTML=s;
 $('tabs').innerHTML=D.views.map((v,i)=>{const p=v.items.filter(it=>status(v,it)==='pending').length;return `<button data-v="${i}" class="${i===S.v?'on':''}">${v.id} · 대기 ${p}</button>`;}).join('');
 for(const [id,m] of [['mSel','sel'],['mKeep','keep'],['mDel','delete'],['mCut','cut'],['mAdd','add']])$(id).className=S.mode===m?'on':'';
 const cur=S.sel?(S.lb[key(V.id,S.sel)]||{}).label||'':'';document.querySelectorAll('#lbrow [data-lb]').forEach(b=>{b.classList.toggle('on',!!S.sel&&b.dataset.lb===cur&&cur!=='');});
 let tot=0,left=0,nl=0;D.views.forEach(v=>v.items.forEach(it=>{if(it.state==='pending'){tot++;if(status(v,it)==='pending')left++;}if(lab(v,it))nl++;}));$('pd').textContent=tot-left;$('pt').textContent=tot;$('lc').textContent=nl;
 $('list').innerHTML=V.items.map((it,i)=>{const st=status(V,it);const l=lab(V,it);return `<div data-i="${i}"><span class="${st==='pending'?'pend':(st==='deleted'||st==='excluded')?'del':'acc'}">●</span>${it.id} ${it.badges.map(b=>`<span class="b">${b}</span>`).join('')}${it.suspect?'<span class="b">M</span>':''}${l?`<span class="b">${LB[l]}</span>`:''}<span style="margin-left:auto">${st}</span></div>`;}).join('');
 $('hint').textContent={sel:'인스턴스를 클릭해 고른 뒤 1 가려짐 · 2 온전 · 3 조각 · 4 소엽 아님 (위 버튼도 가능).',keep:'클릭 = 유지(대기 → 투표 포함).',delete:'클릭 = 삭제 토글.',cut:'인스턴스를 고른 뒤 절단선을 클릭으로 그리고 [선 완료].',add:'빠진 소엽의 꼭짓점을 클릭하고 [선 완료].'}[S.mode];}
async function setLabel(l){const V=D.views[S.v];if(!S.sel){$('hint').textContent='먼저 인스턴스를 클릭해 고르세요.';return;}const k=key(V.id,S.sel);if(l===''){delete S.lb[k];await putc('instance_labels',k,null);}else{S.lb[k]={label:l,view:V.id,id:S.sel};await putc('instance_labels',k,S.lb[k]);}render();}
$('lbrow').onclick=e=>{const b=e.target.closest('[data-lb]');if(b)setLabel(b.dataset.lb);};
document.addEventListener('keydown',e=>{if(e.target.closest&&e.target.closest('input,textarea'))return;const n=+e.key;if(n>=1&&n<=4){setLabel(LK[n-1]);e.preventDefault();}else if(e.key==='+'||e.key==='='){zoomCentre(1.4);}else if(e.key==='-'){zoomCentre(1/1.4);}else if(e.key==='0'){zoomReset();}else if(e.key==='Escape'){S.draft=[];render();}});
$('zin').onclick=()=>zoomCentre(1.4);$('zout').onclick=()=>zoomCentre(1/1.4);$('zreset').onclick=zoomReset;
$('photo').addEventListener('wheel',e=>{e.preventDefault();const r=$('photo').getBoundingClientRect();zoomAt(Math.exp(-e.deltaY*0.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
$('photo').addEventListener('pointerdown',e=>{if(e.button>1)return;const sx=e.clientX,sy=e.clientY,ox=Z.x,oy=Z.y;Z.drag=false;
 const mv=ev=>{const dx=ev.clientX-sx,dy=ev.clientY-sy;if(!Z.drag&&Math.hypot(dx,dy)<5)return;Z.drag=true;Z.x=ox+dx;Z.y=oy+dy;clampPan();$('zw').style.transform='translate('+Z.x+'px,'+Z.y+'px) scale('+Z.k+')';};
 const up=()=>{window.removeEventListener('pointermove',mv);window.removeEventListener('pointerup',up);if(Z.drag){Z.sup=true;setTimeout(()=>{Z.sup=false;},80);Z.drag=false;}};
 window.addEventListener('pointermove',mv);window.addEventListener('pointerup',up);});
$('tabs').onclick=e=>{const t=e.target.closest('[data-v]');if(t){S.v=+t.dataset.v;S.sel=null;S.draft=[];Z.k=1;Z.x=0;Z.y=0;$('zw').style.transform='';$('zv').textContent='100%';render();}};
$('list').onclick=e=>{const t=e.target.closest('[data-i]');if(t){S.sel=D.views[S.v].items[+t.dataset.i].id;render();}};
for(const [id,m] of [['mSel','sel'],['mKeep','keep'],['mDel','delete'],['mCut','cut'],['mAdd','add']])$(id).onclick=()=>{S.mode=m;S.draft=[];render();};
$('undo').onclick=()=>{S.draft.pop();render();};
$('done').onclick=async()=>{const V=D.views[S.v];if(S.mode==='cut'&&S.sel&&S.draft.length>=2){const k=key(V.id,S.sel);const o=S.ops[k]&&S.ops[k].op==='cut'?S.ops[k]:{op:'cut',view:V.id,id:S.sel,cuts:[]};o.cuts.push(S.draft);S.ops[k]=o;await put(k,o);}
 else if(S.mode==='add'&&S.draft.length>=3){const k=V.id+'|ADD'+Date.now().toString(36);S.ops[k]={op:'add',view:V.id,points:S.draft};await put(k,S.ops[k]);}S.draft=[];render();};
$('ov').onclick=async e=>{if(Z.sup)return;const V=D.views[S.v];const r=$('ov').getBoundingClientRect();const x=Math.round((e.clientX-r.left)/r.width*V.w),y=Math.round((e.clientY-r.top)/r.height*V.h);const p=e.target.closest('polygon[data-i]');
 if(S.mode==='add'||(S.mode==='cut'&&S.sel)){S.draft.push([x,y]);render();return;}
 if(!p)return;const it=V.items[+p.dataset.i];S.sel=it.id;const k=key(V.id,it.id);
 if(S.mode==='keep'){if(S.ops[k]&&S.ops[k].op==='keep'){delete S.ops[k];await put(k,null);}else{S.ops[k]={op:'keep',view:V.id,id:it.id};await put(k,S.ops[k]);}}
 else if(S.mode==='delete'){if(S.ops[k]&&S.ops[k].op==='delete'){delete S.ops[k];await put(k,null);}else{S.ops[k]={op:'delete',view:V.id,id:it.id};await put(k,S.ops[k]);}}
 render();};
$('finish').onclick=async()=>{let tot=0,left=0;D.views.forEach(v=>v.items.forEach(it=>{if(it.state==='pending'){tot++;if(status(v,it)==='pending')left++;}}));
 const lc={};Object.values(S.lb).forEach(l=>{lc[l.label]=(lc[l.label]||0)+1;});
 if(db){try{await db.doc('session/summary').set({finished_at:new Date().toISOString(),started_at:new Date(S.t0).toISOString(),elapsed_s:Math.round((Date.now()-S.t0)/1000),pending_total:tot,pending_left:left,ops:Object.keys(S.ops).length,labels:lc});$('store').textContent='완료 저장됨';}catch(e){$('store').textContent='저장 실패: '+(e.code||e.message);}}};
setInterval(()=>{const s=Math.round((Date.now()-S.t0)/1000);$('clock').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');},1000);
window.addEventListener('resize',render);
try{S.ops=JSON.parse(localStorage.getItem('ap_ops')||'{}');S.lb=JSON.parse(localStorage.getItem('ap_lb')||'{}');}catch(e){}render();
window.__ap={state:()=>({v:S.v,sel:S.sel,labels:Object.keys(S.lb).length,z:Z.k,x:Z.x,y:Z.y}),setLabel,zoomAt,zoomCentre,zoomReset,select:(id)=>{S.sel=id;render();}};
(async()=>{try{if(window.claude&&window.claude.use){db=await claude.use('db');if(db){const q=await db.collection('mask_corrections').get();S.ops={};q.docs.forEach(d=>{S.ops[d.id]=d.data();});const q2=await db.collection('instance_labels').get();S.lb={};q2.docs.forEach(d=>{S.lb[d.id]=d.data();});$('store').textContent='저장: artifact db (불러옴)';render();}else $('store').textContent='artifact db 없음 → 이 브라우저에만 저장';}}catch(e){$('store').textContent='db 오류: '+(e.message||e);}})();
</script>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--step", choices=("proposers", "vote", "all", "page", "convert"), default="all")
    ap.add_argument("--export", type=Path, help="convert: JSON export of the prefilled page's mask_corrections {doc_id: {op, ...}}"); ap.add_argument("--to", type=Path)
    ap.add_argument("--labels", type=Path, default=None, help="convert: JSON export of the page's instance_labels {view|id: {label}}")
    a = ap.parse_args(argv)
    if a.step in ("proposers", "all"):
        run_proposers(a.out)
    if a.step in ("vote", "all"):
        run_vote(a.out)
    if a.step == "page":
        p, exp = build_page(a.out); print(p, exp)
    if a.step == "convert":
        ops = export_to_human_masks(json.load(open(a.export, encoding="utf-8")), a.out, json.load(open(a.labels, encoding="utf-8")) if a.labels else None); json.dump(ops, open(a.to, "w", encoding="utf-8"), indent=1); print(a.to, len(ops))


if __name__ == "__main__":
    main()
