"""User-labelled compound leaves: the user clicks leaflet masks per photo and gives the leaf number; the session votes into 3D.

Contract: build/p1_compound_userlabel_20261025_01/CONTRACT.md. Step 1 detector union (H2 ∪ RF ∪ SAM-auto, UNION chosen by
construction because no human-confirmed mask exists on this plant), Step 2 greedy-coverage view choice, Step 3 local labelling
page with live SAM (userlabel_page.py), Step 4 vote V-H / V-P and the held-out scoring.

usage: python -m tomato3d.organs.userlabel --step views|detect|vote|score|all
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import contract as C
from tomato3d.organs import light_organs as LO
from tomato3d.organs import seed2d as S2
from tomato3d.organs import vote2d as V2
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_compound_userlabel_20261025_01"
VOTE2D = V2.OUT
PHOTOS = AP.PHOTOS
FG_MASKS = AP.FG_MASKS
H2 = S2.H2_GRID
STATUS = "R3-1b_unaccepted"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG)
# declared (contract sections 1-5)
VIEWS = dict(n_default=4, n_max=6, extra_gain_min=0.10, azimuth_min_deg=60.0, heldout_azimuth_min_deg=45.0, top_view_horiz_mm=50.0)
SAMAUTO = dict(long_mm=(18.6 - 3 * 3.1, 18.6 + 3 * 3.1), lw=(1.0, 2.4), inside_plant_min=0.80, work_px=1536)
UNION = dict(dedup_iou=0.5, contain_iou_small=0.7, contain_min_parts=2)
VOTE = dict(V_min=1, S_min=0.60)
GATE = dict(median_iou_min=0.60, confusion_max=0.10)
VP = dict(long_px=1024, cap_minutes=90)
PAL = V2.PAL


# ---------------------------------------------------------------- helpers
def load_plant():
    P = CC.load_r31b(); sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); rows = P["rows"]
    return dict(P=P, xyz=P["xyz"], rows=rows, blade=~P["is_axis"], sc=sf["scale"][rows].astype(np.float64), tu=sf["tu"][rows].astype(np.float64), tv=sf["tv"][rows].astype(np.float64),
                nrm=P["normal"], po=json.load(open(LO.DEFAULTS["poses"])), views=json.load(open(LO.DEFAULTS["views"]))["train"])


def camera_azimuth(po, view):
    R = np.array(po["views"][view]["R_cw"]); t = np.array(po["views"][view]["t_cw"]); c = -R.T @ t
    d = c[:2] - V2.CENTRE; return float(np.degrees(np.arctan2(d[1], d[0])) % 360), float(np.hypot(*d) * 1000), c


def az_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def vid_of(view):
    return view[9:18]


def rle(m):
    return dict(size=[H2, H2], counts=S2.rle_encode(m))


def unrle(r):
    from tomato3d.leafseg import pixels as px
    return px.rle_decode(r)


# ---------------------------------------------------------------- step 2: views by greedy coverage
def choose_views(vis_matrix, views, po, params=VIEWS):
    """vis_matrix: bool (n_blade, n_views). Returns chosen list, held-out view, report."""
    az = {v: camera_azimuth(po, v) for v in views}; top = {v: az[v][1] < params["top_view_horiz_mm"] for v in views}
    def separated(v, chosen, min_deg):
        return all(top[v] or top[c] or az_diff(az[v][0], az[c][0]) >= min_deg for c in chosen)
    n = vis_matrix.shape[0]; covered = np.zeros(n, bool); chosen = []; steps = []
    for k in range(params["n_max"]):
        cands = [(int((vis_matrix[:, j] & ~covered).sum()), v) for j, v in enumerate(views) if v not in chosen and separated(v, chosen, params["azimuth_min_deg"])]
        if not cands:
            break
        gain, v = max(cands)
        if k >= params["n_default"] and gain < params["extra_gain_min"] * n:
            steps.append(dict(view=vid_of(v), gain=gain, gain_share=round(gain / n, 4), taken=False, reason="gain below 10 %")); break
        chosen.append(v); covered |= vis_matrix[:, views.index(v)]; steps.append(dict(view=vid_of(v), gain=gain, gain_share=round(gain / n, 4), taken=True, azimuth=round(az[v][0], 1), top=top[v], horiz_mm=round(az[v][1], 1)))
    hc = [(int((vis_matrix[:, j] & ~covered).sum()), int(vis_matrix[:, j].sum()), v) for j, v in enumerate(views) if v not in chosen and separated(v, chosen, params["heldout_azimuth_min_deg"])]
    held = max(hc)[2] if hc else None
    cnt = vis_matrix[:, [views.index(v) for v in chosen]].sum(1)
    rep = dict(params=params, chosen=[vid_of(v) for v in chosen], held_out=(vid_of(held) if held else None), steps=steps, blade_surfels=int(n), seen_ge1=int((cnt >= 1).sum()), seen_ge2=int((cnt >= 2).sum()),
               seen_ge1_share=round(float((cnt >= 1).mean()), 4), seen_ge2_share=round(float((cnt >= 2).mean()), 4),
               azimuths={vid_of(v): dict(azimuth=round(az[v][0], 1), horiz_mm=round(az[v][1], 1), top=top[v]) for v in chosen + ([held] if held else [])})
    return chosen, held, rep


def step_views(out: Path):
    pl = load_plant(); z = np.load(VOTE2D / "stage_a_h2id.npz"); H2id = z["H2id"]; assert list(z["views"]) == pl["views"]
    vis = (H2id[pl["blade"]] >= 0); chosen, held, rep = choose_views(vis, pl["views"], pl["po"])
    json.dump(rep, open(out / "views_chosen.json", "w", encoding="utf-8"), indent=1); print(json.dumps(rep, indent=1)); return chosen, held


# ---------------------------------------------------------------- step 1: detectors and UNION on the chosen views
def plant_px_per_mm(pl, view, H2id_col):
    K = ca.scaled_K(np.array(pl["po"]["K_undist"]), pl["po"]["undistorted_res"], H2); R = np.array(pl["po"]["views"][view]["R_cw"]); t = np.array(pl["po"]["views"][view]["t_cw"])
    z = (pl["xyz"] @ R.T + t)[:, 2]; zz = z[H2id_col >= 0]; return float(K[0, 0] * 0.001 / zz.mean()), float(zz.mean())


def shape_ok(m, px_per_mm):
    import cv2
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, None
    c = max(cnts, key=cv2.contourArea); (_, _), (a, b), _ = cv2.minAreaRect(c); L, W = max(a, b) / px_per_mm, max(min(a, b), 1e-6) / px_per_mm
    ok = SAMAUTO["long_mm"][0] <= L <= SAMAUTO["long_mm"][1] and SAMAUTO["lw"][0] <= L / W <= SAMAUTO["lw"][1]
    return ok, dict(long_mm=round(L, 1), lw=round(L / W, 2))


def p_samauto(vid, rgb_full, sam, plant, px_per_mm, log):
    """SAM 2.1 automatic masks on the 1536 working image, restricted to the plant mask and the declared leaflet size/shape."""
    t0 = time.time(); img = Image.fromarray(rgb_full); img.thumbnail((SAMAUTO["work_px"], SAMAUTO["work_px"])); rgb = np.asarray(img); hh, ww = rgb.shape[:2]
    cands = sam.automatic(rgb); out = []; drop = Counter()
    for i, c in enumerate(cands):
        m = np.zeros((H2, H2), bool); m[:hh, :ww] = c.mask[:H2, :H2]
        if m.sum() == 0: drop["empty"] += 1; continue
        if (m & plant).sum() / m.sum() < SAMAUTO["inside_plant_min"]: drop["outside_plant"] += 1; continue
        ok, sh = shape_ok(m, px_per_mm)
        if not ok: drop["size_or_shape"] += 1; continue
        out.append(dict(uid=f"SA:{i}", prop="P-SA", mask=m & plant, score=float(c.score), shape=sh))
    log[vid] = dict(candidates=len(cands), kept=len(out), dropped=dict(drop), seconds=round(time.time() - t0, 1), work_px=[ww, hh]); return out


def union_masks(masks):
    """Dedup IoU >= 0.5 (higher score wins); a kept mask containing >= 2 others (IoU-of-smaller >= 0.7) is merge_suspect and dropped for its parts."""
    order = sorted(masks, key=lambda m: -m["score"]); kept = []; dup = 0
    for m in order:
        if any(AP.iou(m["mask"], k["mask"]) >= UNION["dedup_iou"] for k in kept):
            dup += 1; continue
        kept.append(m)
    areas = [int(k["mask"].sum()) for k in kept]; suspect = set()
    for i, k in enumerate(kept):
        parts = [j for j, o in enumerate(kept) if j != i and areas[j] < areas[i] and (k["mask"] & o["mask"]).sum() / max(areas[j], 1) >= UNION["contain_iou_small"]]
        if len(parts) >= UNION["contain_min_parts"]:
            suspect.add(i); k["merge_suspect"] = True; k["parts"] = [kept[j]["uid"] for j in parts]
    final = [k for i, k in enumerate(kept) if i not in suspect]
    return final, dict(input=len(masks), duplicates=dup, merge_suspect_dropped=len(suspect), union=len(final))


def step_detect(out: Path, vids):
    from tomato3d.leafseg import run as r2
    from tomato3d.topview.measure2d import load_sam
    pl = load_plant(); z = np.load(VOTE2D / "stage_a_h2id.npz"); H2id = z["H2id"]; views = pl["views"]
    key = AP._rf_key(); sam = load_sam(r2.SAMCFG); cfg = json.load(open(r2.T1CFG)); d = out / "detect"; d.mkdir(exist_ok=True); rflog, salog, summ = {}, {}, {}
    for vid in vids:
        view = AP.view_file(vid); t0 = time.time(); rgb = np.asarray(Image.open(PHOTOS / view).convert("RGB"))
        plant = np.asarray(Image.open(FG_MASKS / (Path(view).stem + ".png")).resize((H2, H2), Image.NEAREST)) == 1
        ppm, zbar = plant_px_per_mm(pl, view, H2id[:, views.index(view)])
        props = {"P-H2": AP.p_h2(vid)}
        if key:
            try:
                props["P-RF"] = AP.p_rf(vid, rgb, sam, cfg, key, out, rflog)
            except RuntimeError as e:
                props["P-RF"] = []; rflog[vid] = dict(error=str(e))
        else:
            props["P-RF"] = []; rflog[vid] = dict(error="no key available")
        props["P-SA"] = p_samauto(vid, rgb, sam, plant, ppm, salog)
        allm = [dict(m, prop=p) for p, ms in props.items() for m in ms]
        final, ust = union_masks(allm)
        json.dump(dict(view=view, status=STATUS, px_per_mm=round(ppm, 3), plant_depth_m=round(zbar, 4), union=UNION, samauto=SAMAUTO,
                       masks=[dict(id=f"U{i + 1:03d}", uid=m["uid"], prop=m["prop"], score=round(m["score"], 4), rle=rle(m["mask"]), area_px=int(m["mask"].sum()), shape=m.get("shape")) for i, m in enumerate(final)],
                       merge_suspect=[dict(uid=m["uid"], prop=m["prop"], parts=m["parts"], rle=rle(m["mask"])) for m in allm if m.get("merge_suspect")]),
                  open(d / f"{vid}.json", "w", encoding="utf-8"))
        summ[vid] = dict(counts={p: len(v) for p, v in props.items()}, union=ust, by_prop_in_union=dict(Counter(m["prop"] for m in final)), seconds=round(time.time() - t0, 1))
        print(vid, json.dumps(summ[vid]), flush=True)
    json.dump(dict(status=STATUS, chosen_rule="UNION by construction (contract section 1 rule ii: no human-confirmed mask on this plant)", per_view=summ, roboflow=rflog, samauto=salog, sam_device="cpu"),
              open(out / "detect_summary.json", "w", encoding="utf-8"), indent=1, default=float)


# ---------------------------------------------------------------- step 4: votes
def load_labels(out: Path):
    f = out / "labels" / "leaf_labels.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else {}


def record_rle(r, recs, by_id):
    """Mask of one label record: its own rle, a detector mask by mask_id, or (user-made SAM piece re-labelled by click) the record named by mask_id."""
    if "rle" in r: return unrle(r["rle"])
    mid = r.get("mask_id")
    if mid in by_id: return unrle(by_id[mid]["rle"])
    if mid in recs and "rle" in recs[mid]: return unrle(recs[mid]["rle"])
    return None


def label_maps(out: Path, vids):
    """Per view: int16 label image (1536) of leaf numbers from the user's records (0 = none; '모름' = -1 -> no vote); plus counts."""
    maps, counts = {}, {}
    recs = load_labels(out)
    for vid in vids:
        det = json.load(open(out / "detect" / f"{vid}.json", encoding="utf-8")); by_id = {m["id"]: m for m in det["masks"]}
        # replay the append-only records: a removal record names the key (mask id or record id) it cancels
        active = {}
        for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
            if r.get("removed"):
                active.pop(r["removed"], None); continue
            active[r.get("mask_id") or r["id"]] = r
        img = np.zeros((H2, H2), np.int16); c = Counter()
        for r in active.values():
            leaf = -1 if r["leaf"] in ("모름", "unknown", None) else int(r["leaf"])
            m = record_rle(r, recs, by_id)
            if m is None: continue
            img[m] = leaf; c[leaf] += 1
        maps[vid] = img; counts[vid] = dict(c)
    return maps, counts


FILL = dict(radius_mm=6.0)
FRAGMENT = -2


def fill_gaps(lab, plant, px_per_mm, radius_mm=FILL["radius_mm"]):
    """Contract section 8 gap fill: close each leaf's pixels with a disc of radius_mm; keep new pixels inside the plant mask, outside
    every labelled pixel (any leaf, unknown, 0-mapped), and claimed by one leaf only. Returns filled image and pixels added per leaf."""
    import cv2
    r = max(1, int(round(radius_mm * px_per_mm))); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    free = plant & (lab == 0); claim = np.zeros(lab.shape, np.int16); n_claim = np.zeros(lab.shape, np.int8)
    for L in sorted(set(lab[lab > 0].tolist())):
        c = cv2.morphologyEx((lab == L).astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & free
        claim[c] = L; n_claim[c] += 1
    out = lab.copy(); ok = n_claim == 1; out[ok] = claim[ok]
    return out, {int(L): int((ok & (claim == L)).sum()) for L in sorted(set(lab[lab > 0].tolist()))}


def remap(maps, number_map):
    """Apply the user's cross-photo number map: local number -> plant leaf number. User 0 (fragment/other part) -> FRAGMENT (-2),
    unknown -> -1; both cast no vote and block the gap fill (0 stays 'no label')."""
    out = {}
    for vid, img in maps.items():
        o = np.zeros_like(img)
        for L in sorted(set(img[img != 0].tolist())):
            to = -1 if L == -1 else int(number_map.get(f"{vid}|{L}", -1))
            o[img == L] = FRAGMENT if to == 0 else to
        out[vid] = o
    return out


def local_surfel_labels(pl, maps, vids, H2id):
    """Per view: the per-photo leaf number of every visible blade surfel (0 = not under a labelled mask)."""
    n = len(pl["xyz"]); views = pl["views"]; loc = {}
    for vid in vids:
        view = AP.view_file(vid); k = views.index(view); vis = (H2id[:, k] >= 0) & pl["blade"]
        pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"]); u = pv["u"] * H2 // V2.VISP["res"]; w = pv["w"] * H2 // V2.VISP["res"]
        ok = vis & (u >= 0) & (w >= 0) & (u < H2) & (w < H2); l = np.zeros(n, np.int16); l[ok] = maps[vid][w[ok], u[ok]]; loc[vid] = l
    return loc


def propose_mapping(loc, ref):
    """Proposal only (the user decides): each photo's local leaf number -> a number of the reference photo, by the largest number of
    shared surfels; groups with no shared surfel with the reference go through the best-matching already-mapped group of another photo."""
    groups = {v: sorted(set(l[l > 0].tolist())) for v, l in loc.items()}; shared = {}
    for a in loc:
        for b in loc:
            if a == b: continue
            for A in groups[a]:
                ma = loc[a] == A
                for B in groups[b]:
                    s = int((ma & (loc[b] == B)).sum())
                    if s: shared[(a, A, b, B)] = s
    prop = {ref: {A: dict(to=A, via="reference", shared=None) for A in groups[ref]}}
    for v in loc:
        if v == ref: continue
        prop[v] = {}
        for A in groups[v]:
            c = [(s, B) for (a, A_, b, B), s in shared.items() if a == v and A_ == A and b == ref]
            if c:
                s, B = max(c); prop[v][A] = dict(to=B, via="reference", shared=s)
    for v in loc:                                                  # second pass through other mapped photos
        for A in groups[v]:
            if A in prop[v]: continue
            c = [(s, prop[b][B]["to"], b) for (a, A_, b, B), s in shared.items() if a == v and A_ == A and B in prop.get(b, {}) and b != ref]
            prop[v][A] = (dict(to=max(c)[1], via=f"via {max(c)[2]}", shared=max(c)[0]) if c else dict(to=None, via="no overlap", shared=0))
    alts = {f"{v}|{A}": sorted(((s, b, B) for (a, A_, b, B), s in shared.items() if a == v and A_ == A), reverse=True)[:4] for v in loc for A in groups[v]}
    # contract section 8 auto-confirm: best >= 3 x second (reference leaves only) and unique within the photo
    for v in loc:
        if v == ref:
            for A in prop[v]: prop[v][A]["status"] = "reference"
            continue
        best_of = Counter(p["to"] for p in prop[v].values() if p["via"] == "reference")
        for A, p in prop[v].items():
            c = sorted((s for (a, A_, b, B), s in shared.items() if a == v and A_ == A and b == ref), reverse=True)
            ratio = (c[0] / c[1]) if len(c) >= 2 else (float("inf") if c else 0.0)
            p["ratio"] = (round(ratio, 2) if np.isfinite(ratio) else None)
            p["status"] = "auto" if (p["via"] == "reference" and ratio >= MAP_AUTO_RATIO and best_of[p["to"]] == 1) else "check"
    return prop, alts, groups


MAP_AUTO_RATIO = 3.0
TOPO = dict(dh_mm=10.0, dtheta_deg=30.0, dr_mm=15.0, iqr_mm=15.0, extent_mm=1.5 * 37.2, mode_sep_mm=12.0, mode_share=0.25, order_mm=10.0)


def group_descriptors(xyz, sel):
    """3D descriptors of a group of surfels around the pot-centre vertical (contract section 8 topology check)."""
    P = xyz[sel]
    if len(P) < 5:
        return None
    h = (P[:, 2] - V2.POT_TOP) * 1000; d = P[:, :2] - V2.CENTRE; th = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360; r = np.hypot(d[:, 0], d[:, 1]) * 1000
    tm = float(np.degrees(np.arctan2(np.sin(np.radians(th)).mean(), np.cos(np.radians(th)).mean())) % 360)
    c = P.mean(0); dist = np.linalg.norm(P - c, axis=1) * 1000; ext = float(2 * (np.percentile(dist, 95)))
    # bimodality: split the heights at the largest gap of the sorted 1-mm histogram's two strongest separated modes
    hist, edges = np.histogram(h, bins=np.arange(np.floor(h.min()), np.ceil(h.max()) + 2, 2.0)); bimodal = False
    if len(hist) >= 3:
        from scipy.ndimage import gaussian_filter1d
        sm = gaussian_filter1d(hist.astype(float), 1.5); pk = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 0]
        for i in pk:
            for j in pk:
                if j > i and (edges[j] - edges[i]) >= TOPO["mode_sep_mm"]:
                    cut = edges[(i + j) // 2 + 1]; lo = (h < cut).mean()
                    if min(lo, 1 - lo) >= TOPO["mode_share"]: bimodal = True
    return dict(n=int(len(P)), h=round(float(np.median(h)), 1), h_p10=round(float(np.percentile(h, 10)), 1), theta=round(tm, 1), r=round(float(np.median(r)), 1),
                h_iqr=round(float(np.percentile(h, 75) - np.percentile(h, 25)), 1), extent=round(ext, 1), bimodal=bimodal)


def topo_flags(desc, ref_desc):
    f = []
    if desc is None:
        return ["too_few_surfels"]
    if ref_desc is not None:
        dth = abs((desc["theta"] - ref_desc["theta"] + 180) % 360 - 180)
        if abs(desc["h"] - ref_desc["h"]) > TOPO["dh_mm"] or dth > TOPO["dtheta_deg"] or abs(desc["r"] - ref_desc["r"]) > TOPO["dr_mm"]:
            f.append("match_mismatch")
    if desc["h_iqr"] > TOPO["iqr_mm"] or desc["extent"] > TOPO["extent_mm"] or desc["bimodal"]:
        f.append("over_painted")
    return f


def order_conflicts(leaf_h_p10):
    """Pairs (a < b by number) whose base heights contradict bottom -> top numbering by more than TOPO['order_mm']."""
    ks = sorted(leaf_h_p10); return [(a, b, leaf_h_p10[a], leaf_h_p10[b]) for i, a in enumerate(ks) for b in ks[i + 1:] if leaf_h_p10[a] - leaf_h_p10[b] > TOPO["order_mm"]]


def vote_views(pl, maps, vids, H2id_views, H2id):
    """V-H: one vote per labelled view where the surfel is visible and under a labelled mask (leaf > 0)."""
    n = len(pl["xyz"]); views = pl["views"]; cnt = np.zeros((n, 13), np.int32)
    for vid in vids:
        view = AP.view_file(vid); k = views.index(view); vis = H2id[:, k] >= 0
        pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"])      # pixel positions (same splat as stage A)
        u = pv["u"] * H2 // V2.VISP["res"]; w = pv["w"] * H2 // V2.VISP["res"]; ok = vis & (u >= 0) & (w >= 0) & (u < H2) & (w < H2)
        lv = np.zeros(n, np.int16); lv[ok] = maps[vid][w[ok], u[ok]]; sel = lv > 0
        np.add.at(cnt, (np.nonzero(sel)[0], lv[sel].astype(int)), 1)
    tot = cnt.sum(1); win = cnt.argmax(1); top = cnt.max(1); share = np.where(tot > 0, top / np.maximum(tot, 1), 0.0); second = np.sort(cnt, 1)[:, -2]
    lab = np.where((tot >= VOTE["V_min"]) & (share >= VOTE["S_min"]) & (top > second), win, 0).astype(np.int16); lab[~pl["blade"]] = 0
    return lab, cnt


def render_ids(pl, lab, view, H2id_col):
    """Rendered leaf-id image (1536) of the visible surfels as discs (Revision-1 radii)."""
    pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"]); res = V2.VISP["res"]; img = np.zeros((H2, H2), np.int16)
    idx = np.nonzero((H2id_col >= 0) & (lab > 0))[0]
    if not len(idx): return img
    r = np.minimum(V2.VISP["radius_sigma"] * pl["sc"][idx].max(1) * pv["K"][0, 0] / pv["z"][idx], V2.VISP["max_r_px"]) * H2 / res
    u = pv["u"][idx] * H2 / res; w = pv["w"][idx] * H2 / res; order = np.argsort(-pv["z"][idx])     # far first, near overwrites
    import cv2
    for j in order:
        cv2.circle(img, (int(round(u[j])), int(round(w[j]))), int(max(1, round(r[j]))), int(lab[idx[j]]), -1)
    return img


def score_heldout(lab_img, rendered):
    leaves = sorted(set(np.unique(lab_img).tolist()) - {0, -1}); A_any = lab_img > 0; rows = {}
    for L in leaves:
        A = lab_img == L; R_ = rendered == L; inter = (A & R_).sum(); union = (A | (R_ & A_any)).sum()
        rows[L] = dict(pixels=int(A.sum()), iou=float(inter / max(union, 1)), confusion=float(((rendered > 0) & (rendered != L) & A).sum() / max(A.sum(), 1)), unassigned=float(((rendered == 0) & A).sum() / max(A.sum(), 1)))
    ious = [r["iou"] for r in rows.values()]; conf = [r["confusion"] for r in rows.values()]
    return rows, dict(median_iou=(float(np.median(ious)) if ious else None), max_confusion=(max(conf) if conf else None),
                      gate_pass=bool(ious) and float(np.median(ious)) >= GATE["median_iou_min"] and max(conf) <= GATE["confusion_max"])


def leaf_table(pl, lab, counts, H):
    rows = []; p0 = np.array([V2.CENTRE[0], V2.CENTRE[1], V2.POT_TOP])
    for L in sorted(set(lab[lab > 0].tolist())):
        m = lab == L; rec = dict(leaf=int(L), n_surfels=int(m.sum()), count_2d_estimate=max((c.get(L, 0) for c in counts.values()), default=0))
        if m.sum() >= 3:
            P = pl["xyz"][m]; cen, w, e1, e2, nn = LO.pca_frame(P); area, length, width, n_px = LO.union_area(P, pl["tu"][m], pl["tv"][m], pl["sc"][m], cen, e1, e2)
            wts = np.pi * LO.FLOORS["ellipse_k"] ** 2 * pl["sc"][m, 0] * pl["sc"][m, 1]; nr = pl["nrm"][m] * np.where(pl["nrm"][m][:, 2:3] < 0, -1, 1); mn = (nr * wts[:, None]).sum(0); mn /= max(np.linalg.norm(mn), 1e-12)
            rec.update(area_m2=round(float(area), 8), area_rel_H2=round(float(area) / H ** 2, 6), inclination_area_weighted_deg=round(float(np.degrees(np.arccos(np.clip(mn[2], -1, 1)))), 2),
                       mean_normal=[round(float(v), 4) for v in mn], azimuth_deg=round(float(LO.deg_azimuth(cen[0] - p0[0], cen[1] - p0[1])), 1), h_min_mm=round(float(P[:, 2].min() - V2.POT_TOP) * 1000, 1),
                       h_max_mm=round(float(P[:, 2].max() - V2.POT_TOP) * 1000, 1), centroid_m=[round(float(v), 5) for v in cen])
        rows.append(rec)
    return rows


def step_vote(out: Path, grouping="user_labelled", lab_override=None):
    pl = load_plant(); z = np.load(VOTE2D / "stage_a_h2id.npz"); H2id = z["H2id"]; views = pl["views"]; H = float(pl["xyz"][:, 2].max() - V2.POT_TOP)
    vc = json.load(open(out / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]
    maps, counts = label_maps(out, chosen + [held]); labelled_views = [v for v in chosen if counts.get(v)]
    lab, cnt = (lab_override, None) if lab_override is not None else vote_views(pl, maps, labelled_views, None, H2id)
    rendered = render_ids(pl, lab, AP.view_file(held), H2id[:, views.index(AP.view_file(held))]); rows, gate = score_heldout(maps[held], rendered)
    leaves_in_2 = sorted(L for L in set(k for c in counts.values() for k in c if k > 0) if sum(1 for v in chosen if counts.get(v, {}).get(L)) >= 2)
    table = leaf_table(pl, lab, {v: counts[v] for v in chosen}, H); tag = "gate_passed" if gate["gate_pass"] else "gate_failed_viewing_only"
    res = dict(**LABELS, grouping=grouping, vote=VOTE, gate=GATE, labelled_views=labelled_views, held_out=held, label_counts=counts, held_out_rows={str(k): v for k, v in rows.items()},
               held_out_summary=gate, tag=tag, leaves_with_labels_in_ge2_views=leaves_in_2, blade_surfels=int(pl["blade"].sum()), assigned=int((lab > 0).sum()), unassigned_blade=int((pl["blade"] & (lab == 0)).sum()),
               per_leaf_surfels={int(L): int((lab == L).sum()) for L in sorted(set(lab[lab > 0].tolist()))})
    sub = out / grouping; sub.mkdir(exist_ok=True)
    json.dump(res, open(sub / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    with open(sub / "leaves.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {grouping}; {tag}; area = kind (1); count_2d_estimate = max labelled leaflets over labelling views\n")
        keys = sorted({k for r in table for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table)
    full = np.full(len(pl["xyz"]), -1); full[lab > 0] = lab[lab > 0]
    np.savez_compressed(sub / "surfel_clusters.npz", rows=pl["rows"], leaf=full, axis_kind=np.load(V2.R5_DIR / "surfel_clusters.npz")["axis_kind"])
    with open(sub / "surfel_leaf_ids.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {grouping}; {tag}; row = index into R3-1b_surfels_full.ply; leaf_id -1 = unassigned/axis\n"); f.write("row,leaf_id\n")
        for r, l in zip(pl["rows"], full): f.write(f"{r},{l}\n")
    print(json.dumps(dict(grouping=grouping, tag=tag, gate=gate, rows=rows, assigned=int((lab > 0).sum()), per_leaf=res["per_leaf_surfels"]), indent=1, default=float)); return res


def topology_report(pl, raw_maps, nmap, vids, H2id):
    """Contract section 8: descriptors per (photo, local group), reference = other labelling photos' groups of the same plant leaf."""
    loc = local_surfel_labels(pl, raw_maps, vids, H2id); rows = []
    for v in vids:
        for L in sorted(set(loc[v][loc[v] > 0].tolist())):
            rows.append(dict(photo=v, local=int(L), leaf=int(nmap.get(f"{v}|{L}", -1)), desc=group_descriptors(pl["xyz"], loc[v] == L)))
    for r in rows:
        if r["leaf"] <= 0 or r["desc"] is None:
            r["flags"] = ([] if r["desc"] is not None else ["too_few_surfels"]); continue
        others = [o["desc"] for o in rows if o["leaf"] == r["leaf"] and o["photo"] != r["photo"] and o["desc"] is not None]
        ref = None
        if others:
            th = np.radians([o["theta"] for o in others]); ref = dict(h=float(np.median([o["h"] for o in others])), r=float(np.median([o["r"] for o in others])),
                                                                    theta=float(np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360))
        r["reference"] = ref; r["flags"] = topo_flags(r["desc"], ref)
    leaf_p10 = {}
    for r in rows:
        if r["leaf"] > 0 and r["desc"] is not None:
            leaf_p10.setdefault(r["leaf"], []).append(r["desc"]["h_p10"])
    base = {L: float(np.median(v)) for L, v in leaf_p10.items()}
    return rows, dict(leaf_base_h_p10_mm=base, order_conflicts=order_conflicts(base))


def build_vote_maps(pl, raw, nmap, chosen, topo_rows, method):
    """Per labelling view: plant-leaf map for the method (V-H / V-H+fill / V-H+fill+topo) and fill statistics."""
    mapped = remap({v: raw[v] for v in chosen}, nmap); H2id = None; out, fstats = {}, {}
    flagged = {(r["photo"], r["local"]) for r in topo_rows if any(f in r["flags"] for f in ("match_mismatch", "over_painted"))}
    for v in chosen:
        m = mapped[v].copy()
        if method.endswith("topo"):
            for (pv_, L) in flagged:
                if pv_ == v: m[raw[v] == L] = FRAGMENT                                 # no vote, blocks the fill
        if "fill" in method:
            view = AP.view_file(v); plant = np.asarray(Image.open(FG_MASKS / (Path(view).stem + ".png")).resize((H2, H2), Image.NEAREST)) == 1
            ppm = STATE_PPM.get(v)
            m, added = fill_gaps(m, plant, ppm); fstats[v] = added
        out[v] = m
    return out, fstats, sorted(flagged)


STATE_PPM = {}


def step_vote_all(out: Path, dest: Path | None = None, map_file: Path | None = None):
    dest = dest or out
    pl = load_plant(); z = np.load(VOTE2D / "stage_a_h2id.npz"); H2id = z["H2id"]; views = pl["views"]; H = float(pl["xyz"][:, 2].max() - V2.POT_TOP)
    vc = json.load(open(out / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]
    nm = json.load(open(map_file or (out / "labels" / "number_map.json"), encoding="utf-8"))
    if "latest" not in nm:                                         # a session-made corrected map (contract section 8) has no versions
        nm = dict(latest=dict(map=nm["map"], at=nm["source"]))
    nmap = nm["latest"]["map"]
    raw, counts_local = label_maps(out, chosen + [held])
    for v in chosen:
        STATE_PPM[v] = plant_px_per_mm(pl, AP.view_file(v), H2id[:, views.index(AP.view_file(v))])[0]
    topo_rows, topo_sum = topology_report(pl, raw, nmap, chosen + [held], H2id)
    topo_rows_lab = [r for r in topo_rows if r["photo"] in chosen]
    held_map = remap({held: raw[held]}, nmap)[held]; held_map = np.where(held_map > 0, held_map, 0).astype(np.int16)       # scoring uses labelled leaves only
    hv = AP.view_file(held); held_col = H2id[:, views.index(hv)]
    results = {}
    for method in ("V-H+fill+topo", "V-H+fill", "V-H"):
        vm, fstats, flagged = build_vote_maps(pl, raw, nmap, chosen, topo_rows_lab, method)
        lab, cnt = vote_views(pl, vm, chosen, None, H2id)
        rendered = render_ids(pl, lab, hv, held_col); rows, gate = score_heldout(held_map, rendered)
        counts = {v: dict(Counter(int(x) for x in [nmap.get(f"{v}|{L}", -1) for L, c in counts_local[v].items() for _ in range(c)] if x > 0)) for v in chosen}
        leaves_in_2 = sorted(L for L in {k for c in counts.values() for k in c} if sum(1 for v in chosen if counts[v].get(L)) >= 2)
        tag = "gate_passed" if gate["gate_pass"] else "gate_failed_viewing_only"; grouping = {"V-H": "user_labelled", "V-H+fill": "user_labelled_fill", "V-H+fill+topo": "user_labelled_fill_topo"}[method]
        tot = cnt.sum(1); reason = np.where(~pl["blade"], "axis", np.where(lab > 0, "assigned", np.where(tot == 0, "fragment_or_other", "low_share")))
        # post-vote topology: voted 3D leaf vs each labelling view's groups of that leaf
        post = {}
        for L in sorted(set(lab[lab > 0].tolist())):
            d3 = group_descriptors(pl["xyz"], lab == L); views_d = [r["desc"] for r in topo_rows_lab if r["leaf"] == L and r["desc"]]
            post[int(L)] = dict(voted=d3, label_groups=views_d)
        table = leaf_table(pl, lab, counts, H)
        res = dict(**LABELS, grouping=grouping, method=method, vote=VOTE, fill=FILL, topo=TOPO, gate=GATE, number_map_version=nm["latest"]["at"], labelled_views=chosen, held_out=held,
                   flagged_groups_excluded=(flagged if method.endswith("topo") else []), fill_pixels=fstats, held_out_rows={str(k): v for k, v in rows.items()}, held_out_summary=gate, tag=tag,
                   leaves_with_labels_in_ge2_views=leaves_in_2, blade_surfels=int(pl["blade"].sum()), assigned=int((lab > 0).sum()), fragment_or_other=int((reason == "fragment_or_other").sum()),
                   low_share=int((reason == "low_share").sum()), per_leaf_surfels={int(L): int((lab == L).sum()) for L in sorted(set(lab[lab > 0].tolist()))}, post_vote_topology=post)
        sub = dest / grouping; sub.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(sub / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
        with open(sub / "leaves.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {grouping}; {tag}; area = kind (1); count_2d_estimate = max labelled leaflets over labelling views; untraced/unlabelled leaves absent\n")
            keys = sorted({k for r in table for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table)
        full = np.full(len(pl["xyz"]), -1); full[lab > 0] = lab[lab > 0]
        np.savez_compressed(sub / "surfel_clusters.npz", rows=pl["rows"], leaf=full, axis_kind=np.load(V2.R5_DIR / "surfel_clusters.npz")["axis_kind"], reason=reason.astype("U20"))
        with open(sub / "surfel_leaf_ids.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {grouping}; {tag}; row = index into R3-1b_surfels_full.ply; leaf_id -1 = unassigned/axis/fragment_or_other\nrow,leaf_id\n")
            for rr, l in zip(pl["rows"], full): f.write(f"{rr},{l}\n")
        Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8)).save(sub / f"rendered_heldout_{held}.png")
        results[method] = dict(gate=gate, rows=rows, assigned=res["assigned"], per_leaf=res["per_leaf_surfels"], excluded=len(res["flagged_groups_excluded"]))
        print(method, json.dumps(results[method], default=float), flush=True)
    json.dump(dict(rows=topo_rows, summary=topo_sum, rule=TOPO), open(dest / "topology_check.json", "w", encoding="utf-8"), indent=1, default=float)
    print("topology", json.dumps(dict(flagged=[(r["photo"], r["local"], r["leaf"], r["flags"]) for r in topo_rows if r["flags"]], order=topo_sum["order_conflicts"]), default=float), flush=True)
    return results


# ---------------------------------------------------------------- V-P: SAM 2 video propagation along camera rings (contract section 8)
VP_RING_GAP_DEG = 3.0


def rings_of(pl, views, exclude):
    rows = []
    for v in views:
        if vid_of(v) in exclude: continue
        R = np.array(pl["po"]["views"][v]["R_cw"]); t = np.array(pl["po"]["views"][v]["t_cw"]); c = -R.T @ t; d = c - np.array([V2.CENTRE[0], V2.CENTRE[1], V2.POT_TOP])
        rows.append((float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))), float(np.degrees(np.arctan2(d[1], d[0])) % 360), v))
    rows.sort(); rings = [[rows[0]]]
    for r in rows[1:]:
        (rings.append([r]) if r[0] - rings[-1][-1][0] > VP_RING_GAP_DEG else rings[-1].append(r))
    return [[v for _, _, v in sorted(rg, key=lambda x: x[1])] for rg in rings], {v: (e, a) for e, a, v in rows}


def segments(frames, seeds):
    """Ring-order-nearest seed for every frame (circular); {seed: frame indices ordered by signed ring offset from the seed}."""
    n = len(frames); si = [frames.index(s) for s in seeds]; owner = {}
    for i in range(n):
        owner[i] = min(si, key=lambda s: (min((i - s) % n, (s - i) % n), s))
    signed = lambda i, s: (i - s) % n if (i - s) % n <= n // 2 else -((s - i) % n)
    return {frames[s]: sorted([i for i in range(n) if owner[i] == s], key=lambda i: signed(i, s)) for s in si}


def step_vp(out: Path, dest: Path, map_file: Path, res=VP["long_px"]):
    import shutil, sys, tempfile, torch
    from tomato3d.leafseg import run as r2
    cfg = json.load(open(r2.SAMCFG))
    for k in ("runtime", "repository"):
        if cfg.get(k): sys.path.insert(0, str(Path(cfg[k])))
    from sam2.build_sam import build_sam2_video_predictor
    torch.set_num_threads(int(cfg.get("threads", 4))); t0 = time.time()
    vp = build_sam2_video_predictor(cfg["model_config"], str(Path(cfg["checkpoint"])), device="cpu")
    pl = load_plant(); views = pl["views"]; H2id = np.load(VOTE2D / "stage_a_h2id.npz")["H2id"]
    vc = json.load(open(out / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]
    nm = json.load(open(map_file, encoding="utf-8")); nmap = nm["map"] if "map" in nm else nm["latest"]["map"]
    raw, _ = label_maps(out, chosen + [held])
    for v in chosen:
        STATE_PPM[v] = plant_px_per_mm(pl, AP.view_file(v), H2id[:, views.index(AP.view_file(v))])[0]
    topo_rows, _ = topology_report(pl, raw, nmap, chosen + [held], H2id)
    seed_maps, _, _ = build_vote_maps(pl, raw, nmap, chosen, [r for r in topo_rows if r["photo"] in chosen], "V-H+fill+topo")
    rings, elaz = rings_of(pl, views, {held}); prop = {}; log = []; stopped = False
    with torch.inference_mode():
        for rg in rings:
            seeds = [AP.view_file(v) for v in chosen if AP.view_file(v) in rg]
            if not seeds:
                log.append(dict(ring_elev=[round(elaz[rg[0]][0], 1)], frames=len(rg), seeds=[], note="no seed -> nothing")); continue
            for s, order in segments(rg, seeds).items():
                if stopped: break
                k0 = order.index(rg.index(s)); frames = [rg[i] for i in order]; tmp = Path(tempfile.mkdtemp(prefix="vp_"))
                for i, v in enumerate(frames):
                    Image.open(PHOTOS / v).convert("RGB").resize((res, res), Image.BILINEAR).save(tmp / f"{i:05d}.jpg", quality=95)
                st = vp.init_state(video_path=str(tmp)); sm = seed_maps[vid_of(s)]; leaves = sorted(set(sm[sm > 0].tolist()))
                for L in leaves:
                    vp.add_new_mask(st, frame_idx=k0, obj_id=int(L), mask=np.asarray(Image.fromarray((sm == L).astype(np.uint8)).resize((res, res), Image.NEAREST)).astype(bool))
                logits = {}
                for rev in (False, True):
                    for fi, ids, lg in vp.propagate_in_video(st, start_frame_idx=k0, reverse=rev):
                        logits[fi] = (list(ids), lg[:, 0].float().cpu().numpy())
                vp.reset_state(st); shutil.rmtree(tmp, ignore_errors=True)
                for fi, (ids, lg) in logits.items():
                    v = frames[fi]
                    if v == s: continue
                    best = lg.argmax(0); pos = lg.max(0) > 0; m = np.where(pos, np.array(ids)[best], 0).astype(np.int16)
                    prop[vid_of(v)] = np.asarray(Image.fromarray(m).resize((H2, H2), Image.NEAREST)).astype(np.int16)
                log.append(dict(seed=vid_of(s), frames=len(frames), leaves=leaves, seconds=round(time.time() - t0))); print("V-P", log[-1], flush=True)
                if time.time() - t0 > VP["cap_minutes"] * 60:
                    log.append(dict(stopped="90 min cap reached")); stopped = True
    allmaps = dict(seed_maps, **prop); vids = chosen + sorted(prop)
    lab, cnt = vote_views(pl, allmaps, vids, None, H2id)
    hv = AP.view_file(held); rendered = render_ids(pl, lab, hv, H2id[:, views.index(hv)])
    held_map = remap({held: raw[held]}, nmap)[held]; held_map = np.where(held_map > 0, held_map, 0).astype(np.int16)
    rows, gate = score_heldout(held_map, rendered); tag = "gate_passed" if gate["gate_pass"] else "gate_failed_viewing_only"
    sub = dest / "user_labelled_propagated"; sub.mkdir(parents=True, exist_ok=True); H = float(pl["xyz"][:, 2].max() - V2.POT_TOP)
    tot = cnt.sum(1); reason = np.where(~pl["blade"], "axis", np.where(lab > 0, "assigned", np.where(tot == 0, "fragment_or_other", "low_share")))
    res_ = dict(**LABELS, grouping="user_labelled_propagated", vote=VOTE, vp=dict(VP, ring_gap_deg=VP_RING_GAP_DEG), log=log, propagated_views=sorted(prop), held_out=held,
                held_out_rows={str(k): v for k, v in rows.items()}, held_out_summary=gate, tag=tag, assigned=int((lab > 0).sum()), blade_surfels=int(pl["blade"].sum()),
                fragment_or_other=int((reason == "fragment_or_other").sum()), low_share=int((reason == "low_share").sum()),
                per_leaf_surfels={int(L): int((lab == L).sum()) for L in sorted(set(lab[lab > 0].tolist()))}, seconds=round(time.time() - t0))
    json.dump(res_, open(sub / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    table = leaf_table(pl, lab, {}, H)
    with open(sub / "leaves.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping user_labelled_propagated; {tag}; area = kind (1)\n")
        keys = sorted({k for r in table for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table)
    full = np.full(len(pl["xyz"]), -1); full[lab > 0] = lab[lab > 0]
    np.savez_compressed(sub / "surfel_clusters.npz", rows=pl["rows"], leaf=full, axis_kind=np.load(V2.R5_DIR / "surfel_clusters.npz")["axis_kind"], reason=reason.astype("U20"))
    with open(sub / "surfel_leaf_ids.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping user_labelled_propagated; {tag}; row = index into R3-1b_surfels_full.ply; leaf_id -1 = unassigned/axis/fragment_or_other\nrow,leaf_id\n")
        for rr, l in zip(pl["rows"], full): f.write(f"{rr},{l}\n")
    np.savez_compressed(sub / "propagated_maps.npz", **{f"v{k}": v for k, v in prop.items()})
    print("V-P result", json.dumps(dict(gate=gate, rows=rows, assigned=res_["assigned"], per_leaf=res_["per_leaf_surfels"]), default=float), flush=True)
    return res_


# ---------------------------------------------------------------- step 4b: leaflet instances inside each user leaf (A3 v2 rules, contract section 4b)
A3 = dict(overlap=0.50, centroid_mult=0.30, normal_deg=35.0, cap_mm=28.0, min_extent_mm=9.3, min_surfels=80, min_mask_surfels=15)


def mask_patch(pl, sel):
    """centroid, plane normal (n_z >= 0) and extent (max pairwise distance, via the PCA-aligned bounding diagonal sample) of a surfel set."""
    P = pl["xyz"][sel]; c = P.mean(0); w, v = np.linalg.eigh(np.cov((P - c).T)); n = v[:, 0] * (1 if v[2, 0] >= 0 else -1)
    q = P[np.random.default_rng(0).choice(len(P), min(len(P), 300), replace=False)]; d = np.linalg.norm(q[:, None] - q[None], axis=2)
    return dict(c=c, n=n, ext=float(d.max()))


def a3v2_leaf(pl, items, cap_on=True):
    """items: [(view, mask_id, surfel index array)] of ONE user leaf. Union-find with overlap / 3-D patch links, same-view cannot-link, size cap.
    Returns clusters (lists of item indices), link stats."""
    n = len(items); parent = list(range(n)); members = {i: [i] for i in range(n)}; sets = [set(it[2].tolist()) for it in items]; patches = [mask_patch(pl, it[2]) for it in items]
    find = lambda x: x if parent[x] == x else find(parent[x])
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = len(sets[i] & sets[j]) / max(min(len(sets[i]), len(sets[j])), 1)
            dc = np.linalg.norm(patches[i]["c"] - patches[j]["c"]) * 1000; lim = A3["centroid_mult"] * (patches[i]["ext"] + patches[j]["ext"]) * 1000 / 2
            ang = np.degrees(np.arccos(np.clip(abs(float(patches[i]["n"] @ patches[j]["n"])), 0, 1)))
            by_ov = ov >= A3["overlap"]; by_3d = (dc <= lim) and (ang <= A3["normal_deg"])
            if by_ov or by_3d: cand.append((0 if by_ov else 1, -ov, dc, i, j, "overlap" if by_ov else "patch_3d"))
    stats = dict(candidates=len(cand), by_overlap=0, only_patch_3d=0, rejected_same_view=0, rejected_cap=0)
    for _, _, _, i, j, kind in sorted(cand):
        a, b = find(i), find(j)
        if a == b: continue
        if {items[k][0] for k in members[a]} & {items[k][0] for k in members[b]}:
            stats["rejected_same_view"] += 1; continue
        if cap_on:
            allidx = np.unique(np.concatenate([items[k][2] for k in members[a] + members[b]])); P = pl["xyz"][allidx]
            q = P[np.random.default_rng(0).choice(len(P), min(len(P), 300), replace=False)]
            if np.linalg.norm(q[:, None] - q[None], axis=2).max() * 1000 > A3["cap_mm"]:
                stats["rejected_cap"] += 1; continue
        parent[b] = a; members[a] += members.pop(b); stats["by_overlap" if kind == "overlap" else "only_patch_3d"] += 1
    return [m for m in members.values()], stats


def step_4b(out: Path, dest: Path, map_file: Path, source="user_labelled_fill_topo"):
    """Leaflet instances inside each user leaf from the labelled masks of the chosen views plus the V-H leaf assignment (contract 4b)."""
    from tomato3d.leafseg import pixels as px
    pl = load_plant(); views = pl["views"]; H2id = np.load(VOTE2D / "stage_a_h2id.npz")["H2id"]
    vc = json.load(open(out / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]
    nm = json.load(open(map_file, encoding="utf-8")); nmap = nm["map"] if "map" in nm else nm["latest"]["map"]
    z = np.load(dest / source / "surfel_clusters.npz"); leaf3d = z["leaf"]                          # compound-leaf result (never changed here)
    maps_local, _ = label_maps(out, chosen + [held]); det = {v: {m["id"]: m for m in json.load(open(out / "detect" / f"{v}.json", encoding="utf-8"))["masks"]} for v in chosen + [held]}
    recs = load_labels(out); active = {}
    for r in sorted(recs.values(), key=lambda r: r["at"]):
        k = (r["view"], r.get("mask_id") or r["id"])
        if r.get("removed"): active.pop((r["view"], r["removed"]), None); continue
        active[k] = r
    def items_for(vid):
        its = []; view = AP.view_file(vid); k = views.index(view); vis = (H2id[:, k] >= 0) & pl["blade"]
        pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"]); u = pv["u"] * H2 // V2.VISP["res"]; w = pv["w"] * H2 // V2.VISP["res"]; ok = vis & (u >= 0) & (w >= 0) & (u < H2) & (w < H2)
        for (v, key), r in active.items():
            if v != vid: continue
            leaf = -1 if r["leaf"] in ("모름", None) else int(r["leaf"]); L = nmap.get(f"{vid}|{leaf}", -1) if leaf > 0 else -1
            if L <= 0: continue
            m = record_rle(r, recs, det[vid]); sel = np.nonzero(ok & m[np.clip(w, 0, H2 - 1), np.clip(u, 0, H2 - 1)])[0]
            if len(sel) >= A3["min_mask_surfels"]: its.append((vid, key, sel, int(L)))
        return its
    leaflet_id = {}; per_view = {v: items_for(v) for v in chosen}; res = dict(**LABELS, leaflet_instances="a3v2_within_user_leaf", a3=A3, source=source, leaves={})
    for cap_on in (True, False):
        tag = "cap28" if cap_on else "cap_off"
        for L in sorted({it[3] for v in chosen for it in per_view[v]}):
            its = [it for v in chosen for it in per_view[v] if it[3] == L]
            cl, st = a3v2_leaf(pl, its, cap_on)
            if cap_on:                                                  # per-surfel leaflet id (majority of the clusters whose masks cover it); fragments -> -1
                cnt_ = {}
                for ci, c in enumerate(cl):
                    for k in c:
                        for sidx in its[k][2].tolist(): cnt_.setdefault(sidx, Counter())[ci] += 1
                for sidx, cc in cnt_.items():
                    ci = cc.most_common(1)[0][0]; idx_ = np.unique(np.concatenate([its[k][2] for k in cl[ci]]))
                    if len(idx_) >= A3["min_surfels"]: leaflet_id[sidx] = (L, ci)
            sizes = []
            for c in cl:
                idx = np.unique(np.concatenate([its[k][2] for k in c])); P = pl["xyz"][idx]; q = P[np.random.default_rng(0).choice(len(P), min(len(P), 300), replace=False)]
                ext = float(np.linalg.norm(q[:, None] - q[None], axis=2).max() * 1000); frag = ext < A3["min_extent_mm"] or len(idx) < A3["min_surfels"]
                sizes.append(dict(n_masks=len(c), n_surfels=int(len(idx)), extent_mm=round(ext, 1), fragment=bool(frag)))
            nlab = max((sum(1 for it in per_view[v] if it[3] == L) for v in chosen), default=0)
            res["leaves"].setdefault(str(L), {})[tag] = dict(masks=len(its), clusters=len(cl), leaflets=sum(1 for s in sizes if not s["fragment"]), fragments=sum(1 for s in sizes if s["fragment"]),
                                                              max_labelled_leaflets_in_a_view=nlab, links=st, cluster_sizes=sizes)
    # held-out scoring: the held-out view's labelled leaflets as truth; clusters per leaflet and purity over labelled leaflets (same leaf only)
    hitems = items_for(held); score = {}
    for cap_on in (True, False):
        tag = "cap28" if cap_on else "cap_off"; rows = {}
        for L in sorted({it[3] for it in hitems}):
            truth = [it for it in hitems if it[3] == L]; train = [it for v in chosen for it in per_view[v] if it[3] == L]
            if not train or not truth: continue
            cl, st = a3v2_leaf(pl, train, cap_on); sets = [set(np.concatenate([train[k][2] for k in c]).tolist()) for c in cl]
            clusters_per, pur_num, pur_den = [], 0, 0
            for t in truth:
                ts = set(t[2].tolist()); hit = [i for i, s in enumerate(sets) if len(s & ts) >= 0.3 * min(len(s), len(ts))]
                clusters_per.append(len(hit))
            for i, s in enumerate(sets):                                          # purity: share of a cluster's surfels inside ONE held-out labelled leaflet
                ov = [len(s & set(t[2].tolist())) for t in truth]; tot = sum(ov)
                if tot: pur_num += max(ov); pur_den += tot
            rows[str(L)] = dict(held_out_leaflets=len(truth), clusters_per_leaflet=(round(float(np.mean(clusters_per)), 2) if clusters_per else None), purity=(round(pur_num / pur_den, 3) if pur_den else None), train_clusters=len(cl))
        mc = [r["clusters_per_leaflet"] for r in rows.values() if r["clusters_per_leaflet"] is not None]; pu = [r["purity"] for r in rows.values() if r["purity"] is not None]
        score[tag] = dict(per_leaf=rows, mean_clusters_per_leaflet=(round(float(np.mean(mc)), 2) if mc else None), mean_purity=(round(float(np.mean(pu)), 3) if pu else None),
                          targets=dict(clusters_per_leaflet_max=1.3, purity_min=0.98),
                          pass_=bool(mc) and float(np.mean(mc)) <= 1.3 and bool(pu) and float(np.mean(pu)) >= 0.98)
    res["held_out_score"] = score
    lid = np.full(len(pl["xyz"]), -1); leaf_of = np.full(len(pl["xyz"]), -1)
    keys = sorted(set(leaflet_id.values())); kmap = {k: i for i, k in enumerate(keys)}
    for sidx, k in leaflet_id.items(): lid[sidx] = kmap[k]; leaf_of[sidx] = k[0]
    np.savez_compressed(dest / "leaflet_instances_a3v2_within_user_leaf" / "surfel_leaflets.npz" if (dest / "leaflet_instances_a3v2_within_user_leaf").exists() else (dest / "surfel_leaflets_tmp.npz"),
                        rows=pl["rows"], leaflet=lid, leaf=leaf_of, keys=np.array([f"{a}-{b + 1}" for a, b in keys]))
    sub = dest / "leaflet_instances_a3v2_within_user_leaf"; sub.mkdir(parents=True, exist_ok=True)
    if (dest / "surfel_leaflets_tmp.npz").exists(): (dest / "surfel_leaflets_tmp.npz").replace(sub / "surfel_leaflets.npz")
    json.dump(res, open(sub / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps(dict(leaves={k: {t: (v["leaflets"], v["fragments"], v["max_labelled_leaflets_in_a_view"]) for t, v in d.items()} for k, d in res["leaves"].items()}, held=score), default=float)[:3000], flush=True)
    return res


# ---------------------------------------------------------------- trace-flank + height rule (contract section 9)
TRACE_RULE = dict(W_mm=18.6, dh_mm=15.0, share_min=0.50, labelled_min=200, step_mm=0.5)
HUMAN_TRACE_MAP = {"leaf3_mvnnjt1z": 7, "leaf5_mvnnmmh3": 6, "leaf6_mvnnnds1": 2}     # contract section 9 addendum (user screenshots)


def apply_trace_rule(pl, lab, traces, rule=TRACE_RULE, human_map=None):
    """Contract section 9. Returns new labels, the per-trace table, conflicts and the trace -> label map."""
    from scipy.spatial import cKDTree
    blade = pl["blade"]; z = pl["xyz"][:, 2] * 1000; geo = []
    for t in traces:
        pts = V2.CC.densify(np.asarray(t["pts"], float), rule["step_mm"] / 1000); d, i = cKDTree(pts).query(pl["xyz"])
        g = dict(key=t["key"], leaf=t["leaf"], d=d * 1000, zref=pts[i, 2] * 1000); g["flank"] = (g["d"] <= rule["W_mm"]) & (np.abs(z - g["zref"]) <= rule["dh_mm"]) & blade; geo.append(g)
    table, claim = [], {}
    for g in geo:
        l = lab[g["flank"] & (lab > 0)]; n = int(len(l)); row = dict(trace=g["key"], trace_leaf=g["leaf"], flank_surfels=int(g["flank"].sum()), labelled=n, dist={int(k): int(v) for k, v in Counter(l.tolist()).most_common(6)})
        if n >= rule["labelled_min"]:
            L, c = Counter(l.tolist()).most_common(1)[0]; row.update(majority=int(L), share=round(c / n, 3))
            if c / n >= rule["share_min"]: claim.setdefault(int(L), []).append(g["key"])
        table.append(row)
    leaf_of_trace, conflicts = {}, []
    for L, keys in claim.items():
        tl = {g["leaf"] for g in geo if g["key"] in keys}
        if len(tl) == 1:
            for k in keys: leaf_of_trace[k] = L
        else: conflicts.append(dict(label=L, traces=keys, trace_leaves=sorted(tl)))
    for k, L in (human_map or {}).items():
        leaf_of_trace[k] = L                                                  # human-stated mapping overrides the automatic one
    conflicts = [c for c in conflicts if not all(k in (human_map or {}) for k in c["traces"])]
    for row in table: row["mapped_label"] = leaf_of_trace.get(row["trace"]); row["human_stated"] = row["trace"] in (human_map or {})
    out = lab.copy(); mapped = [g for g in geo if g["key"] in leaf_of_trace]
    for L in sorted(set(leaf_of_trace.values())):                         # rule 4: height
        gs = [g for g in mapped if leaf_of_trace[g["key"]] == L]; arg = np.argmin([g["d"] for g in gs], axis=0); zref = np.choose(arg, [g["zref"] for g in gs])
        out[(out == L) & (np.abs(z - zref) > rule["dh_mm"])] = 0
    if mapped:                                                            # rule 3: flank, nearest wins
        D = np.stack([np.where(g["flank"], g["d"], np.inf) for g in mapped]); best = D.argmin(0); ok = np.isfinite(D.min(0))
        tie = np.zeros(len(best), bool)
        if D.shape[0] > 1:
            srt = np.sort(D, 0); tie = ok & (srt[1] == srt[0])
        new = np.array([leaf_of_trace[g["key"]] for g in mapped])[best]; out[ok & ~tie] = new[ok & ~tie]; out[tie] = 0
    out[~blade] = 0
    return out, table, conflicts, leaf_of_trace


def step_trace(out: Path, dest: Path, map_file: Path, human=False):
    pl = load_plant(); views = pl["views"]; H2id = np.load(VOTE2D / "stage_a_h2id.npz")["H2id"]; H = float(pl["xyz"][:, 2].max() - V2.POT_TOP)
    vc = json.load(open(out / "views_chosen.json", encoding="utf-8")); held = vc["held_out"]
    nm = json.load(open(map_file, encoding="utf-8")); nmap = nm["map"] if "map" in nm else nm["latest"]["map"]
    raw, _ = label_maps(out, vc["chosen"] + [held]); held_map = remap({held: raw[held]}, nmap)[held]; held_map = np.where(held_map > 0, held_map, 0).astype(np.int16)
    hv = AP.view_file(held); col = H2id[:, views.index(hv)]; traces = V2.CC.load_traces(V2.OUT / "traces_export" / "rachis_traces"); res_all = {}
    for base in ("user_labelled_propagated", "user_labelled_fill_topo"):
        z0 = np.load(dest / base / "surfel_clusters.npz", allow_pickle=True); lab0 = np.where(z0["leaf"] > 0, z0["leaf"], 0).astype(np.int16)
        lab, table, conflicts, lot = apply_trace_rule(pl, lab0, traces, human_map=(HUMAN_TRACE_MAP if human else None))
        rows0, g0 = score_heldout(held_map, render_ids(pl, lab0, hv, col)); rows1, g1 = score_heldout(held_map, render_ids(pl, lab, hv, col))
        tag = "gate_passed" if g1["gate_pass"] else "gate_failed_viewing_only"; name = base + ("_traceflank_h" if human else "_traceflank"); sub = dest / name; sub.mkdir(exist_ok=True)
        res = dict(**LABELS, grouping=name, rule=TRACE_RULE, base=base, trace_table=table, conflicts=conflicts, trace_to_label=lot, held_out_before=dict(rows={str(k): v for k, v in rows0.items()}, summary=g0),
                   held_out_rows={str(k): v for k, v in rows1.items()}, held_out_summary=g1, tag=tag, changed_surfels=int((lab != lab0).sum()), to_unassigned=int(((lab == 0) & (lab0 > 0)).sum()),
                   from_unassigned=int(((lab > 0) & (lab0 == 0)).sum()), relabelled=int(((lab > 0) & (lab0 > 0) & (lab != lab0)).sum()),
                   per_leaf_before={int(L): int((lab0 == L).sum()) for L in sorted(set(lab0[lab0 > 0].tolist()))}, per_leaf_surfels={int(L): int((lab == L).sum()) for L in sorted(set(lab[lab > 0].tolist()))})
        json.dump(res, open(sub / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
        table_ = leaf_table(pl, lab, {}, H)
        with open(sub / "leaves.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {name}; {tag}; area = kind (1)\n")
            keys = sorted({k for r in table_ for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table_)
        full = np.full(len(pl["xyz"]), -1); full[lab > 0] = lab[lab > 0]
        np.savez_compressed(sub / "surfel_clusters.npz", rows=pl["rows"], leaf=full, axis_kind=z0["axis_kind"])
        with open(sub / "surfel_leaf_ids.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {name}; {tag}; row = index into R3-1b_surfels_full.ply; leaf_id -1 = none\nrow,leaf_id\n")
            for rr, l in zip(pl["rows"], full): f.write(f"{rr},{l}\n")
        res_all[base] = dict(before=g0, after=g1, rows_after={k: (round(v["iou"], 2), round(v["confusion"], 2), round(v["unassigned"], 2)) for k, v in rows1.items()}, mapping=lot, conflicts=conflicts,
                             changed=res["changed_surfels"], to_unassigned=res["to_unassigned"], from_unassigned=res["from_unassigned"], relabelled=res["relabelled"])
        print(base, json.dumps(res_all[base], default=float), flush=True)
        print("table", json.dumps([{k: r[k] for k in ("trace", "trace_leaf", "labelled", "dist", "majority", "share", "mapped_label") if k in r} for r in table], default=float), flush=True)
    return res_all


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--step", choices=("views", "detect", "vote", "vp", "4b", "trace", "all"), default="all"); ap.add_argument("--map-file", type=Path, default=None); ap.add_argument("--human-map", action="store_true"); a = ap.parse_args(argv)
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    if a.step in ("views", "all", "detect"):
        if a.step == "detect" and (out / "views_chosen.json").exists():
            vc = json.load(open(out / "views_chosen.json", encoding="utf-8")); chosen, held = vc["chosen"], vc["held_out"]
        else:
            chosen, held = step_views(out); chosen = [vid_of(v) for v in chosen]; held = vid_of(held)
        if a.step in ("detect", "all"):
            step_detect(out, chosen + [held])
    if a.step == "trace":
        step_trace(out, out / a.map_file.stem.replace("number_map_", "map_"), a.map_file, human=a.human_map)
    if a.step == "4b":
        step_4b(out, out / a.map_file.stem.replace("number_map_", "map_"), a.map_file)
    if a.step == "vp":
        step_vp(out, out / a.map_file.stem.replace("number_map_", "map_"), a.map_file)
    if a.step == "vote":
        nv =len(json.load(open(out / "labels" / "number_map.json", encoding="utf-8"))["versions"])
        if a.map_file:
            step_vote_all(out, out / a.map_file.stem.replace("number_map_", "map_"), a.map_file)
        else:
            step_vote_all(out, out if nv == 1 else out / f"map_v{nv}")


if __name__ == "__main__":
    main()
