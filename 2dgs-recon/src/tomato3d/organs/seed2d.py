"""Seed 3D leaflet instancing from the 2D leaflet masks of H2(M1), then the frozen compound-leaf rule.

Contract: build/p1_leaflet_2dseed_20261024_01/CONTRACT_LEAFLET_2D_SEED.md. Track A (light_organs.py, seed clustering_only)
is read-only; this run writes beside it with `seed = 2d_masks`. Nothing is tuned: H2(M1) outputs are reused as they
are, the 3D constants are the frozen organs contract, the gates are fixed in the contract.

usage: python -m tomato3d.organs.seed2d [--out DIR] [--surfels PLY] [--track-a DIR] [--proposals DIR] [--labels DIR]
                                        [--poses JSON] [--views JSON] [--repeat]
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import contract as C
from tomato3d.organs import leaves as LV
from tomato3d.organs import skeleton as SK
from tomato3d.organs import tracking as TR
from tomato3d.organs import light_organs as LO
from tomato3d.leaflib import answers as an
from tomato3d.leaflib import outline as ol
from tomato3d.leafseg import pixels as px
from tomato3d.organs.light_viewer import TEMPLATE, _b64, palette, read_csv
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_leaflet_2dseed_20261024_01"
TRACK_A = LO.DEFAULTS["out"]
PROPOSALS = ROOT / "build/p1_recon_charuco_20261021_01/leaflets/proposals"
LABELS = ROOT / "build/p1_recon_charuco_20261021_01/leaflets/labels"
H2_GRID = 1536
# declared rules (contract section 2) and gates (section 4)
RULE = dict(knn=C.LEAF["graph_knn"], cap_mm=C.LEAF["graph_cap_mm"], n_same_min=3, n_diff_max=1, min_component=C.LEAF["min_cluster_pts"],
            link_share_min=0.5, link_count_min=20, confirm_views_min=3, confirm_frac_min=0.6)
GATES = dict(largest_leaflet_length_mm_max=25.0, leaflets_per_leaf_median=(4, 8), compound_leaves=(5, 9), leaflet_instances_min=25, merge_suspect_share_max=0.15)
SEED = "2d_masks"
# step 1b: the "single leaflet?" criterion (contract section 1b)
LIB_ANSWERS = ROOT / "build/p1_leaflib_realism_20261022_01/library/answers_resolved.json"
FEATS = ("solidity", "serration_index", "asymmetry")
CRIT = dict(p_merged_min=0.5, C=1.0, class_weight="balanced", labels={"1": "single", "3": "merged", "4": "fragment"})


# ---------------------------------------------------------------- step 1b: criterion from the human library
def library_rows():
    """Descriptors of every agreed whole / merged / fragment mask, with the library's own extraction code (library.build)."""
    ag = json.load(open(LIB_ANSWERS, encoding="utf-8"))["agreed"]; props, skels, rows = {}, {}, []
    for key, v in sorted(ag.items()):
        a = v["answers"][0]
        if a not in CRIT["labels"]:
            continue
        run, photo, mid = v["run"], v["photo"], v["mask_id"]
        if (run, photo) not in props:
            props[(run, photo)] = json.load(open(ROOT / an.RUNS[run] / "proposals" / f"{photo}_leafseg.json", encoding="utf-8"))
            skels[(run, photo)] = np.load(ROOT / an.RUNS[run] / "layers" / f"{photo}.npz")["skeleton"].astype(bool)
        d = props[(run, photo)]; inst = next(i for i in d["instances"] if i["id"] == mid); m = px.rle_decode(inst["visible_segmentation"])
        cue = ol.nearest_skeleton_point(skels[(run, photo)], m); centre = np.array([i["centroid"] for i in d["instances"]], float).mean(0)
        rec = ol.mask_outline(m, base_cue_px=cue, centre_px=centre)
        if rec is None or any(rec[f] is None for f in FEATS):
            continue
        rows.append(dict(key=key, split=v["split"], label=CRIT["labels"][a], **{f: float(rec[f]) for f in FEATS}, lw_apparent=rec["lw_apparent"], area_rel=rec["area_rel"]))
    return rows


class Criterion:
    def __init__(self, rows):
        from sklearn.linear_model import LogisticRegression
        dev = [r for r in rows if r["split"] == "development_photo"]; held = [r for r in rows if r["split"] == "heldout_photo"]
        X = np.array([[r[f] for f in FEATS] for r in dev]); y = np.array([r["label"] for r in dev])
        self.mean, self.sd = X.mean(0), X.std(0) + 1e-9
        self.m = LogisticRegression(class_weight=CRIT["class_weight"], C=CRIT["C"], max_iter=5000).fit((X - self.mean) / self.sd, y)
        self.classes = list(self.m.classes_)
        self.report = dict(features=list(FEATS), dev=dict(Counter(y.tolist())), held=dict(Counter(r["label"] for r in held)),
                           coef={c: dict(zip(FEATS, map(float, w))) for c, w in zip(self.classes, self.m.coef_)}, intercept=dict(zip(self.classes, map(float, self.m.intercept_))),
                           dev_merged=self._prf(dev), heldout_merged=self._prf(held), rule=f"merge_suspect iff P(merged) >= {CRIT['p_merged_min']}")

    def score(self, rec):
        x = (np.array([[rec[f] for f in FEATS]], float) - self.mean) / self.sd; p = self.m.predict_proba(x)[0]
        d = {f"p_{c}": float(v) for c, v in zip(self.classes, p)}; d["merge_suspect"] = bool(d.get("p_merged", 0.0) >= CRIT["p_merged_min"]); return d

    def _prf(self, rows):
        if not rows:
            return None
        pred = np.array([self.score(r)["merge_suspect"] for r in rows]); true = np.array([r["label"] == "merged" for r in rows])
        tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
        return dict(n=len(rows), tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                    flagged_single=int((pred & np.array([r["label"] == "single" for r in rows])).sum()), flagged_fragment=int((pred & np.array([r["label"] == "fragment" for r in rows])).sum()))


OVERRIDE: dict = {}          # Revision 3: {view: [(id, mask, "auto:<ids>")]} replaces a view's instances (set by auto_proposers)


def view_instances(v, corrections=None):
    """[(id, mask, source)] of one view: H2(M1) proposals, or - step 6 - those proposals with the user's corrections applied
    (delete / cut along drawn polylines / add drawn polygons; page px -> 1536 grid by the page scale), or - Revision 3 -
    the view's auto-consensus instances when OVERRIDE holds the view."""
    import cv2
    if v in OVERRIDE:
        return list(OVERRIDE[v]), dict(override="auto", instances=len(OVERRIDE[v]))
    j = json.load(open(PROPOSALS / (Path(v).stem + ".json"), encoding="utf-8"))
    out = [(i["id"], px.rle_decode(i["visible_segmentation"]), "H2(M1)") for i in j["instances"]]
    if not corrections:
        return out, None
    vid = v[9:18]; ops = {k: o for k, o in corrections.items() if o.get("view") == vid}
    if not ops:
        return out, None
    from tomato3d.organs import light_review as LR
    w0 = min(LR.PAGE_PX, j["image_size"][0]); sc = w0 / H2_GRID          # thumbnail width / grid width, as correction_data
    def grid(pts):
        return (np.asarray(pts, float) / sc).round().astype(np.int32)
    res, n_del, n_cut, n_add = [], 0, 0, 0
    for iid, m, src in out:
        o = ops.get(f"{vid}|{iid}")
        if o is None:
            res.append((iid, m, src)); continue
        if o["op"] == "delete":
            n_del += 1; continue
        if o["op"] == "cut":
            mm = m.astype(np.uint8).copy()
            for c in o.get("cuts", []):
                cv2.polylines(mm, [grid(c).reshape(-1, 1, 2)], False, 0, thickness=3)
            n, lab = cv2.connectedComponents(mm, connectivity=4); parts = [(lab == q) for q in range(1, n) if (lab == q).sum() >= ol.MIN_AREA_PX]
            for q, part in enumerate(sorted(parts, key=lambda p_: -p_.sum())):
                res.append((f"{iid}c{q + 1}", part, "human_cut")); n_cut += 1
            continue
        res.append((iid, m, src))
    for k, o in ops.items():
        if o["op"] == "add" and len(o.get("points", [])) >= 3:
            mm = np.zeros((H2_GRID, H2_GRID), np.uint8); cv2.fillPoly(mm, [grid(o["points"]).reshape(-1, 1, 2)], 1); res.append((k.split("|")[-1], mm.astype(bool), "human_add")); n_add += 1
    return res, dict(deleted=n_del, cut_parts=n_cut, added=n_add, instances=len(res))


def score_2d_proposals(views, crit, out, corrections=None, erode_px=0, extra=None):
    """Per training view: suspect flags for every proposal (plant-centre base rule) and the suspect-aware label map.
    Human instances (step 6) are never suspects: they vote first. Revision 2: `erode_px` > 0 erodes every non-suspect
    H2 mask before it votes (empty core -> un-eroded, flagged seed_uneroded); `extra` = {view: [(id, mask)]} watershed
    proposals that vote only where no H2 proposal covers >= 0.5 of them."""
    import cv2
    maps, flags, corr_stats, erode_stats, ws_stats, index = {}, {}, {}, {}, {}, {}
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)) if erode_px > 0 else None
    for v in views:
        inst, cs = view_instances(v, corrections); corr_stats[v[9:18]] = cs
        h2_union = np.zeros((H2_GRID, H2_GRID), bool)
        for _, m, _ in inst:
            h2_union |= m
        ws = (extra or {}).get(v, [])
        inst = inst + [(wid, wm, "watershed") for wid, wm in ws]
        centre = np.array([np.argwhere(m).mean(0)[::-1] for _, m, _ in inst if m.any()]).mean(0) if inst else None
        lab = np.zeros((H2_GRID, H2_GRID), np.int32); masks, fl = [], {}; vanished = 0; core_frac = []
        for k, (iid, m, src) in enumerate(inst):
            if src.startswith("human") or src.startswith("auto"):              # human (step 6) and auto (Revision 3) instances vote first
                s = dict(merge_suspect=False, p_merged=None, source=src)
            else:
                rec = ol.mask_outline(m, base_cue_px=None, centre_px=centre)
                s = crit.score(rec) if rec is not None and all(rec[f] is not None for f in FEATS) else dict(merge_suspect=False, p_merged=None, note="too_small_for_outline")
            s["base_rule"] = "plant_centre"; s["source"] = src
            vote_mask = m
            if ker is not None and src == "H2(M1)" and not s["merge_suspect"]:
                core = cv2.erode(m.astype(np.uint8), ker).astype(bool)
                if core.any():
                    vote_mask = core; s["seed_eroded"] = True; s["core_px"] = int(core.sum()); core_frac.append(core.sum() / max(m.sum(), 1))
                else:
                    s["seed_uneroded"] = True; vanished += 1
            if src == "watershed":
                s["h2_overlap"] = float((m & h2_union).sum() / max(m.sum(), 1)); s["h2_wins"] = s["h2_overlap"] >= 0.5
            fl[iid] = s; masks.append((k + 1, vote_mask, s["merge_suspect"], src, s.get("h2_wins", False)))
        for k, vm, sus, src, h2w in masks:                                     # 1 human / H2 cores
            if not sus and src != "watershed": lab[vm] = k
        for k, vm, sus, src, h2w in masks:                                     # 2 watershed, only where H2 does not win and pixels are empty
            if not sus and src == "watershed" and not h2w: lab[vm & (lab == 0)] = k
        for k, vm, sus, src, h2w in masks:                                     # 3 suspects on empty pixels
            if sus and not (src == "watershed" and h2w): lab[vm & (lab == 0)] = k
        maps[v] = lab; flags[v[9:18]] = fl; index[v] = [(iid, src) for iid, _, src in inst]
        if ker is not None:
            erode_stats[v[9:18]] = dict(proposals=sum(1 for _, _, src in inst if src == "H2(M1)"), vanished=vanished, core_fraction_mean=(float(np.mean(core_frac)) if core_frac else None))
        if ws:
            ws_stats[v[9:18]] = dict(regions=len(ws), suspect=sum(1 for wid, _ in ws if fl[wid]["merge_suspect"]), h2_wins=sum(1 for wid, _ in ws if fl[wid].get("h2_wins")))
        p = out / "leaflet_instances" / (Path(v).stem + ".json")
        if p.exists():
            d = json.load(open(p, encoding="utf-8"))
            if cs or ws:                                             # step 6 corrected list, or watershed regions appended (source recorded)
                d["instances"] = [dict(id=iid, source=src, rle=dict(size=[H2_GRID, H2_GRID], counts=rle_encode(m))) for iid, m, src in inst]
                if cs: d["corrections"] = cs
            for x in d["instances"]:
                x.update({k2: fl.get(x["id"], {}).get(k2) for k2 in ("merge_suspect", "p_merged", "p_single", "p_fragment", "base_rule", "source", "seed_eroded", "seed_uneroded", "core_px", "h2_overlap", "h2_wins")})
            json.dump(d, open(p, "w", encoding="utf-8"))
    score_2d_proposals.corr_stats = corr_stats; score_2d_proposals.erode_stats = erode_stats; score_2d_proposals.ws_stats = ws_stats; score_2d_proposals.index = index
    return maps, flags


def rle_encode(m):
    """COCO uncompressed counts (column-major), the inverse of pixels.rle_decode."""
    f = np.asarray(m, bool).flatten(order="F"); d = np.diff(np.r_[0, f.astype(np.int8), 0]); starts = np.nonzero(d == 1)[0]; ends = np.nonzero(d == -1)[0]
    counts, pos = [], 0
    for s, e in zip(starts, ends):
        counts += [int(s - pos), int(e - s)]; pos = int(e)
    counts.append(int(f.size - pos)); return counts


def instance_mask_2d(xyz_i, sc_i, po, view, res=H2_GRID, k_sigma=VIS["radius_sigma"] if "VIS" in globals() else 1.5):
    """Binary raster of an instance's surfels (disc splats) in one view on the 1536 grid, plus the projected centre of those surfels."""
    K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res); R = np.array(po["views"][view]["R_cw"]); t = np.array(po["views"][view]["t_cw"])
    xc = xyz_i @ R.T + t; z = xc[:, 2]; u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
    r = np.minimum(k_sigma * sc_i.max(1) * K[0, 0] / np.maximum(z, 1e-6), 14); m = np.zeros((res, res), bool); Rr = int(np.ceil(r.max())) if len(r) else 0
    for dy in range(-Rr, Rr + 1):
        for dx in range(-Rr, Rr + 1):
            s = (dx * dx + dy * dy) <= r * r; uu, ww = u[s] + dx, w[s] + dy; ok = (uu >= 0) & (ww >= 0) & (uu < res) & (ww < res); m[ww[ok], uu[ok]] = True
    return m, np.array([u.mean(), w.mean()])


def flag_instances_3d(xyz, sc, V, views, po, instances, crit, plant_centre_cache):
    """instances: {id: index array into xyz}. Best view = most visible surfels; descriptors of the splat raster there."""
    out = {}
    for lid, idx in instances.items():
        vis_counts = (V[idx] >= 0).sum(0); k = int(np.argmax(vis_counts)); v = views[k]; sel = idx[V[idx, k] >= 0]
        if len(sel) < 5:
            out[lid] = dict(best_view=v[9:18], n_visible=int(len(sel)), merge_suspect=False, note="too_few_visible"); continue
        m, _ = instance_mask_2d(xyz[sel], sc[sel], po, v)
        if v not in plant_centre_cache:
            allv = np.nonzero(V[:, k] >= 0)[0]; _, c = instance_mask_2d(xyz[allv[:1]], sc[allv[:1]], po, v); K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], H2_GRID)
            R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"]); xc = xyz[allv] @ R.T + t
            plant_centre_cache[v] = np.array([(K[0, 0] * xc[:, 0] / xc[:, 2] + K[0, 2]).mean(), (K[1, 1] * xc[:, 1] / xc[:, 2] + K[1, 2]).mean()])
        rec = ol.mask_outline(m, base_cue_px=None, centre_px=plant_centre_cache[v])
        s = crit.score(rec) if rec is not None and all(rec[f] is not None for f in FEATS) else dict(merge_suspect=False, p_merged=None, note="too_small_for_outline")
        s.update(best_view=v[9:18], n_visible=int(len(sel)), base_rule="plant_centre", **({f: float(rec[f]) for f in FEATS} if rec else {})); out[lid] = s
    return out


# ---------------------------------------------------------------- step 1: reuse H2(M1) instances
def copy_instances(views, out):
    d = out / "leaflet_instances"; d.mkdir(parents=True, exist_ok=True); counts = {}
    for v in views:
        j = json.load(open(PROPOSALS / (Path(v).stem + ".json"), encoding="utf-8"))
        inst = [dict(id=x["id"], score=x.get("score"), occluded=x.get("occluded"), source="H2(M1)", rle=x["visible_segmentation"]) for x in j["instances"]]
        json.dump(dict(view=v, source="H2(M1) as run on this capture 2026-10-22 (tomato3d.recon.charuco_leaflets, variant H2)", grid=H2_GRID,
                       proposal_note=j.get("note"), instances=inst), open(d / (Path(v).stem + ".json"), "w", encoding="utf-8"))
        counts[v[9:18]] = len(inst)
    return counts


def rle_decode(rle):
    h, w = rle["size"]; m = np.zeros(h * w, bool); pos = 0; val = False
    for c in rle["counts"]:
        if val:
            m[pos:pos + c] = True
        pos += c; val = not val
    return m.reshape(h, w), m.reshape(w, h).T          # row-major and COCO column-major readings


def check_label_ids(view):
    """Label id k must be instance H{k:03d}: compare the RLE of the first instance with labels == 1."""
    j = json.load(open(PROPOSALS / (Path(view).stem + ".json"), encoding="utf-8")); lab = np.asarray(Image.open(LABELS / (Path(view).stem + ".png")))
    if not j["instances"]:
        return None
    l1 = lab == 1
    return max(float((m & l1).sum() / max((m | l1).sum(), 1)) for m in rle_decode(j["instances"][0]["visible_segmentation"]))


# ---------------------------------------------------------------- step 2: votes, co-label graph, linking, confirmation
VIS = dict(mode="splat1512", res=1512, tol_m=C.R3_FUSION["vis_tol_m"], radius_sigma=1.5, max_r_px=14)     # contract revision 1


def splat_zbuffer(u, w, z, r_px, res):
    """Nearest-depth buffer with every surfel rasterised as a disc of radius r_px (pixels)."""
    zb = np.full((res, res), np.inf); R = int(min(np.ceil(r_px.max()), VIS["max_r_px"])) if len(r_px) else 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            m = (dx * dx + dy * dy) <= r_px * r_px
            uu, ww = u[m] + dx, w[m] + dy; ok = (uu >= 0) & (ww >= 0) & (uu < res) & (ww < res)
            np.minimum.at(zb, (ww[ok], uu[ok]), z[m][ok])
    return zb


def votes_per_view(xyz, po, views, res, tol, sc=None, mode="point", label_maps=None):
    """int16 (n, V): 2D id (> 0), 0 = visible on background, -1 = not visible. mode 'point' = the old point z-buffer;
    'splat' = contract revision 1 (disc splats of radius VIS['radius_sigma'] x max(s0, s1))."""
    R0 = po["undistorted_res"]; K = ca.scaled_K(np.array(po["K_undist"]), R0, res); n = len(xyz); V = np.full((n, len(views)), -1, np.int16)
    for k, v in enumerate(views):
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"])
        xc = xyz @ R.T + t; z = xc[:, 2]; u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); idx = np.nonzero(ok)[0]
        if mode == "splat":
            r_px = VIS["radius_sigma"] * sc[idx].max(1) * K[0, 0] / z[idx]
            zb = splat_zbuffer(u[idx], w[idx], z[idx], np.minimum(r_px, VIS["max_r_px"]), res)
        else:
            zb = np.full((res, res), np.inf); order = idx[np.argsort(-z[idx], kind="stable")]; zb[w[order], u[order]] = z[order]
        vis = idx[z[idx] <= zb[w[idx], u[idx]] + tol]
        lab = label_maps[v] if label_maps is not None else np.asarray(Image.open(LABELS / (Path(v).stem + ".png"))).astype(np.int32)
        V[vis, k] = lab[w[vis] * H2_GRID // res, u[vis] * H2_GRID // res]
    return V


def colabel_components(xyz_c, Vc):
    n = len(xyz_c); k = min(RULE["knn"] + 1, n); d, idx = cKDTree(xyz_c).query(xyz_c, k=k)
    rows, cols = [], []
    for j in range(1, k):
        m = d[:, j] <= RULE["cap_mm"] / 1000; rows.append(np.nonzero(m)[0]); cols.append(idx[m, j])
    i, j = np.concatenate(rows), np.concatenate(cols); keep_pair = i < j; i, j = i[keep_pair], j[keep_pair]
    A, B = Vc[i], Vc[j]; both = (A > 0) & (B > 0); same = (both & (A == B)).sum(1); diff = (both & (A != B)).sum(1)
    ok = (same >= RULE["n_same_min"]) & (diff <= RULE["n_diff_max"])
    G = coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(n, n))
    nc, lab = connected_components(G, directed=False)
    vals, cnt = np.unique(lab, return_counts=True); big = {v: q for q, v in enumerate(vals[cnt >= RULE["min_component"]])}
    comp = np.array([big.get(l, -1) for l in lab]); return comp, dict(edges=int(len(i)), kept=int(ok.sum()), components_all=int(nc), components_big=len(big))


def link_ids(Vc, comp, views):
    """(view, id) -> component when share >= 0.5 and >= 20 surfels; returns map and stats."""
    link = {}; stats = dict(nodes=0, mapped=0, view_local=0)
    for k, v in enumerate(views):
        ids = Vc[:, k]; m = ids > 0
        for i2 in np.unique(ids[m]):
            sel = m & (ids == i2); cs = comp[sel]; stats["nodes"] += 1
            cnt = Counter(cs[cs >= 0].tolist()); tot = int(sel.sum())
            if cnt:
                c, nmax = cnt.most_common(1)[0]
                if nmax >= RULE["link_count_min"] and nmax / tot >= RULE["link_share_min"]:
                    link[(k, int(i2))] = int(c); stats["mapped"] += 1; continue
            stats["view_local"] += 1
    return link, stats


def confirm(Vc, comp, link):
    """Per-surfel identity confirmation; returns final component per surfel (-1) and a reason per unavailable surfel."""
    n = len(Vc); final = np.full(n, -1); reason = np.full(n, "", dtype=object)
    for s in range(n):
        ids = Vc[s]; L = np.nonzero(ids > 0)[0]
        if len(L) == 0:
            reason[s] = "unlabelled"; continue
        if comp[s] < 0:
            reason[s] = "no_component"; continue
        if len(L) < RULE["confirm_views_min"]:
            reason[s] = "few_views"; continue
        agree = sum(1 for k in L if link.get((int(k), int(ids[k]))) == comp[s])
        if agree >= RULE["confirm_views_min"] and agree / len(L) >= RULE["confirm_frac_min"]:
            final[s] = comp[s]
        else:
            reason[s] = "conflict"
    vals, cnt = np.unique(final[final >= 0], return_counts=True); small = set(vals[cnt < RULE["min_component"]].tolist())
    for s in np.nonzero(np.isin(final, list(small)))[0]:
        final[s] = -1; reason[s] = "no_component"
    keep = sorted(set(final[final >= 0].tolist())); ren = {c: q for q, c in enumerate(keep)}
    return np.array([ren.get(c, -1) for c in final]), reason, ren


# ---------------------------------------------------------------- step 3: frozen grouping with given clusters (pipeline.run, replicated)
def group(xyz, is_axis, root, cl, order="current"):
    """Mirror of pipeline.run with externally seeded clusters. order 'current' = the current pipeline (tube-tracked trunk,
    tracking.build_tracked; apex cut only when no tube); 'pre_tracking' = the call order before the 2026-10-24 topology
    edits (SK.build + SK.main_stem), kept only to label attempts 1-2."""
    axis_idx = np.nonzero(is_axis)[0]
    if order == "current":
        sk, split = TR.build_tracked(xyz[axis_idx], root); SK.nodes(sk)
    else:
        sk = SK.build(xyz[axis_idx], root); SK.main_stem(sk); SK.nodes(sk); split = None
    cl_vertex = {}
    if len(axis_idx):
        at = cKDTree(xyz[axis_idx])
        for c in range(cl.max() + 1):
            cen = xyz[cl == c].mean(0); _, j = at.query(cen); v = int(sk.vert_of[j]); cl_vertex[c] = v if v >= 0 else 0
    leaves, side_axes, trusses, stubs = LV.structure(sk, cl_vertex); apex_cuts = 0
    for _ in range(0 if getattr(sk, "tube", None) is not None else 3):
        carrying = [nd for nd in sk.nodes if any(lf["node"] == nd["id"] for lf in leaves) or any(sa["node"] == nd["id"] for sa in side_axes)]
        if not carrying:
            break
        vlast = max(carrying[-1]["vertices"], key=sk.main_path.index); i = sk.main_path.index(vlast); term = sk.main_path[i + 1:]
        if not term:
            break
        inv = {}
        for c, v in cl_vertex.items():
            inv.setdefault(v, []).append(c)
        if not any(v in inv for v in sk.subtree(term[0])):
            break
        sk.main_path = sk.main_path[:i + 1]; SK.nodes(sk); apex_cuts += 1
        leaves, side_axes, trusses, stubs = LV.structure(sk, cl_vertex)
        for lf in leaves:
            if lf["root"] == term[0]:
                lf["flags"].append("apex_cut")
    leaf_of, cflags = LV.assign(xyz, cl, sk, leaves, side_axes, axis_idx); labels = LV.leaflet_labels(xyz, cl, sk, leaves, side_axes, leaf_of)
    return dict(sk=sk, leaves=leaves, side_axes=side_axes, trusses=trusses, stubs=stubs, leaf_of_cluster=leaf_of, cluster_flags=cflags, leaflet_labels=labels,
                axis_idx=axis_idx, apex_cuts=apex_cuts, root_far=bool(sk.root_far), nodes=len(sk.nodes), tube=split, order=order)


# ---------------------------------------------------------------- attributes (Track A definitions)
def attributes(xyz, tu, tv, sc, rows, n_seen, cl, r, agree_views):
    lfs = []
    for c in range(cl.max() + 1):
        m = cl == c; P = xyz[m]; cen, w, e1, e2, nn = LO.pca_frame(P)
        area, length, width, n_px = LO.union_area(P, tu[m], tv[m], sc[m], cen, e1, e2); X2 = np.c_[(P - cen) @ e1, (P - cen) @ e2]
        incl = float(np.degrees(np.arccos(np.clip(nn[2], -1, 1)))); planarity = float(w[0] / max(w[1], 1e-18)); flags = []
        if m.sum() < LO.FLOORS["min_surfels"]: flags.append("low_surfel_count")
        sup = float(np.median(n_seen[m]))
        if sup < LO.FLOORS["min_views"]: flags.append("few_views")
        if planarity > LO.FLOORS["max_planarity"]: flags.append("non_planar")
        flags += sorted(set(r["cluster_flags"].get(c, []))); owner = r["leaf_of_cluster"].get(c, "unattached")
        lfs.append(dict(cluster=c, rows=rows[m].tolist(), n_surfels=int(m.sum()), centroid=cen, normal=nn, e1=e1, e2=e2, inclination_deg=incl, normal_elevation_deg=90.0 - incl,
                        normal_azimuth_deg=(LO.deg_azimuth(nn[0], nn[1]) if incl >= LO.FLOORS["min_azimuth_incl_deg"] else None), area_m2=area, area_hull_m2=LO.hull_area(X2),
                        area_sum_ellipse_m2=float((np.pi * LO.FLOORS["ellipse_k"] ** 2 * sc[m, 0] * sc[m, 1]).sum()), length_m=length, width_m=width,
                        lw_ratio=(length / width if width > 0 else None), n_px=n_px, support_views=sup, support_min=int(n_seen[m].min()), planarity=planarity, flags=flags,
                        uncertain=any(f in flags for f in ("low_surfel_count", "few_views", "non_planar")), internal_leaf=(None if owner == "unattached" else owner),
                        label=r["leaflet_labels"].get(c), agree_views_median=float(np.median(agree_views[m]))))
    order = sorted(range(len(lfs)), key=lambda i: (round(lfs[i]["centroid"][2], 9), round(lfs[i]["centroid"][0], 9), round(lfs[i]["centroid"][1], 9)))
    for k, i in enumerate(order):
        lfs[i]["leaflet_id"] = f"LF{k + 1:03d}"
    lfs = [lfs[i] for i in order]; leaves = []
    for lf in r["leaves"] + [x for sa in r["side_axes"] for x in sa["leaves"]]:
        mem = [x for x in lfs if x["internal_leaf"] == lf["id"]]
        if not mem:
            continue
        A = np.array([x["area_m2"] for x in mem]); Cc = np.array([x["centroid"] for x in mem]); wts = A / A.sum() if A.sum() > 0 else np.full(len(A), 1 / len(A))
        cen = (Cc * wts[:, None]).sum(0); incl_w = float((np.array([x["inclination_deg"] for x in mem]) * wts).sum()); incl_p = None
        if len(mem) >= 3:
            _, w, _, _, nn = LO.pca_frame(Cc)
            if w[1] > 1e-12 and w[0] / w[1] < 0.9:
                incl_p = float(np.degrees(np.arccos(np.clip(nn[2], -1, 1))))
        base = r["sk"].V[lf["path"][0]]; ext = 0.0
        if len(mem) > 1:
            d = Cc[:, None, :2] - Cc[None, :, :2]; ext = float(np.sqrt((d ** 2).sum(-1)).max())
        leaves.append(dict(internal_id=lf["id"], members=mem, n_leaflets=len(mem), area_m2=float(A.sum()), inclination_area_weighted_deg=incl_w, inclination_plane_deg=incl_p,
                           centroid=cen, azimuth_deg=LO.deg_azimuth(cen[0] - base[0], cen[1] - base[1]), horizontal_extent_m=ext, support_min=int(min(x["support_min"] for x in mem)),
                           support_median=float(np.median([x["support_views"] for x in mem])), uncertain=any(x["uncertain"] or "ambiguous" in x["flags"] for x in mem),
                           path=r["sk"].V[lf["path"]], leaf_flags=list(lf["flags"]), grouping_source="auto"))
    leaves.sort(key=lambda L: (round(L["centroid"][2], 9), round(L["centroid"][0], 9)))
    for k, L in enumerate(leaves):
        L["leaf_id"] = f"CL{k + 1:02d}"
        for x in L["members"]:
            x["leaf_id"] = L["leaf_id"]
    for x in lfs:
        x.setdefault("leaf_id", "unassigned")
    return lfs, leaves


# ---------------------------------------------------------------- the run
def compute(a):
    sf = sio.read_surfels_full(a.surfels); ta = np.load(Path(a.track_a) / "surfel_labels.npz"); rows = ta["rows"]; is_axis = ta["is_axis"].astype(bool)
    xyz = sf["xyz"][rows].astype(np.float64); tu, tv, sc = sf["tu"][rows], sf["tv"][rows], sf["scale"][rows]; H = float(xyz[:, 2].max() - a.pot_top)
    po = json.load(open(a.poses)); views = json.load(open(a.views))["train"]
    crit, maps, flags2d = None, None, None
    corr = json.load(open(a.human_masks, encoding="utf-8")) if a.human_masks else None
    extra = watershed_proposals(views, Path(a.masks) if getattr(a, "masks", None) else LO.DEFAULTS["masks"]) if getattr(a, "watershed", False) else None
    if a.step1b:
        crit = Criterion(library_rows()); maps, flags2d = score_2d_proposals(views, crit, Path(a.out), corr, erode_px=getattr(a, "erode_px", 0), extra=extra)
    if a.visibility == "splat1512":
        V = votes_per_view(xyz, po, views, VIS["res"], VIS["tol_m"], sc=sc, mode="splat", label_maps=maps)
    else:
        V = votes_per_view(xyz, po, views, C.R3_FUSION["vis_res"], C.R3_FUSION["vis_tol_m"], label_maps=maps)
    n_seen = (V >= 0).sum(1); leak = dict(visible_fraction_per_view_mean=float((V >= 0).mean()), stem_on_leaflet_id_share=float((V[is_axis] > 0).sum() / max((V[is_axis] >= 0).sum(), 1)))
    cand = np.nonzero(~is_axis)[0]; Vc = V[cand]
    comp, gstats = colabel_components(xyz[cand], Vc); link, lstats = link_ids(Vc, comp, views); final, reason, ren = confirm(Vc, comp, link)
    cl = np.full(len(xyz), -1); cl[cand] = final
    agree = np.zeros(len(xyz));
    for q, s in enumerate(cand):
        ids = Vc[q]; agree[s] = sum(1 for k in np.nonzero(ids > 0)[0] if link.get((int(k), int(ids[k]))) is not None and ren.get(link[(int(k), int(ids[k]))]) == final[q])
    r = group(xyz, is_axis, np.asarray(a.root, float), cl, order=a.group_order)
    lfs, leaves = attributes(xyz, tu, tv, sc, rows, n_seen, cl, r, agree)
    flags3d = None
    if crit is not None:
        row_to_i = {int(rr): i for i, rr in enumerate(rows)}; cache = {}
        def inst_of(run_dir):
            js = json.load(open(Path(run_dir) / "leaflet_surfels.json", encoding="utf-8"))
            return {k: np.array([row_to_i[rr] for rr in v["surfel_rows"] if rr in row_to_i], int) for k, v in js.items() if not k.startswith("_")}
        flags3d = dict(this_run=flag_instances_3d(xyz, sc, V, views, po, {x["leaflet_id"]: np.array([row_to_i[rr] for rr in x["rows"]], int) for x in lfs}, crit, cache),
                       track_a=flag_instances_3d(xyz, sc, V, views, po, inst_of(a.track_a), crit, cache),
                       baseline_current_code=(flag_instances_3d(xyz, sc, V, views, po, inst_of(Path(a.out) / "baseline_current_code"), crit, cache)
                                              if (Path(a.out) / "baseline_current_code" / "leaflet_surfels.json").exists() else None))
        for x in lfs:
            f = flags3d["this_run"][x["leaflet_id"]]
            if f.get("merge_suspect"):
                x["flags"].append("merge_suspect")
    reasons = Counter(reason[final < 0].tolist()); axis_with_votes = int(((V[is_axis] > 0).sum(1) > 0).sum())
    links_out = defaultdict(list)
    for (k, i2), c in link.items():
        if c in ren:
            lid = next((x["leaflet_id"] for x in lfs if x["cluster"] == ren[c]), None)
            if lid: links_out[lid].append(f"{views[k][9:18]}:H{i2:03d}")
    buckets = dict(leaflet_assigned=int(np.isin(cl, [x["cluster"] for x in lfs if x["leaf_id"] != "unassigned"]).sum()),
                   leaflet_unassigned=int(np.isin(cl, [x["cluster"] for x in lfs if x["leaf_id"] == "unassigned"]).sum()),
                   unavailable=int((cl < 0).sum() - is_axis.sum()), axis=int(is_axis.sum()))
    return dict(H=H, rows=rows, xyz=xyz, is_axis=is_axis, cl=cl, n_seen=n_seen, leaflets=lfs, leaves=leaves, buckets=buckets, coverage_ok=sum(buckets.values()) == len(xyz),
                graph=gstats, links=lstats, unavailable_reasons=dict(reasons), axis_surfels_with_leaflet_votes=axis_with_votes, n_views=len(views), views=views, leak=leak,
                link_map={x: sorted(set(v)) for x, v in links_out.items()}, apex_cuts=r["apex_cuts"], root_far=r["root_far"], stubs=len(r["stubs"]), trusses=len(r["trusses"]),
                nodes=r["nodes"], group_order=r["order"], criterion=(crit.report if crit else None), flags2d=flags2d, flags3d=flags3d,
                corrections=(getattr(score_2d_proposals, "corr_stats", None) if corr else None),
                V=V, nrm=sf["normal"][rows].astype(np.float64), tu=tu, tv=tv, sc=sc, po=po, crit=crit, link=link, ren=ren, agree=agree,
                erode_stats=getattr(score_2d_proposals, "erode_stats", None), ws_stats=getattr(score_2d_proposals, "ws_stats", None), index2d=getattr(score_2d_proposals, "index", None))


def acceptance(R):
    lengths = [x["length_m"] * 1000 for x in R["leaflets"]]; per = [L["n_leaflets"] for L in R["leaves"]]
    big = max(lengths) if lengths else None; med = float(np.median(per)) if per else None
    return dict(largest_leaflet_length_mm=dict(value=big, gate=f"<= {GATES['largest_leaflet_length_mm_max']}", passes=(big is not None and big <= GATES["largest_leaflet_length_mm_max"])),
                leaflets_per_leaf_median=dict(value=med, gate="4-8", passes=(med is not None and GATES["leaflets_per_leaf_median"][0] <= med <= GATES["leaflets_per_leaf_median"][1])),
                compound_leaves=dict(value=len(R["leaves"]), gate="5-9", passes=GATES["compound_leaves"][0] <= len(R["leaves"]) <= GATES["compound_leaves"][1]),
                leaflet_instances=dict(value=len(R["leaflets"]), gate=">= 25", passes=len(R["leaflets"]) >= GATES["leaflet_instances_min"]),
                surfels_unavailable=dict(value=R["buckets"]["unavailable"], by_reason=R["unavailable_reasons"], gate="none (reported)"),
                merge_suspect_share=(lambda n, k: dict(value=(k / n if n else None), flagged=k, instances=n, gate="<= 0.15", passes=(n > 0 and k / n <= GATES["merge_suspect_share_max"])))(
                    len(R["leaflets"]), sum("merge_suspect" in x["flags"] for x in R["leaflets"])) if R.get("flags3d") else dict(value=None, gate="<= 0.15", passes=None, note="step 1b off"))


# ---------------------------------------------------------------- viewer with the seed toggle
def leaf_colours(run: Path, n):
    lab = np.load(run / "surfel_labels.npz"); leaves = read_csv(run / "compound_leaves.csv"); pal = dict(zip([L["leaf_id"] for L in leaves], palette(len(leaves))))
    col = np.tile(np.array((118, 124, 128), np.uint8), (n, 1)); leaf_of = lab["leaf_id"]
    for r_, c in zip(lab["rows"], lab["cl"]):
        if c >= 0:
            k = leaf_of[c]; col[r_] = pal.get(k, (235, 235, 235)) if k != "unassigned" else (235, 235, 235)
    return col, leaves, pal


def viewer_toggle(out: Path, track_a: Path, surfels: Path, status: str):
    sf = sio.read_surfels_full(surfels); n = len(sf["xyz"]); html = TEMPLATE.read_text(encoding="utf-8")
    m = re.search(r'<script id="D" type="application/json">(.*?)</script>', html, re.S); M = json.loads(m.group(1))
    col_b, leaves_b, pal_b = leaf_colours(out, n); col_a, leaves_a, pal_a = leaf_colours(track_a, n)
    lab = np.load(out / "surfel_labels.npz"); cls = np.zeros(n, np.uint8); cls[lab["rows"]] = 1
    lo = sf["xyz"].min(0); rng = np.maximum(sf["xyz"].max(0) - lo, 1e-9)
    M["discs"] = dict(n=n, n_plant=int(cls.sum()), H=float(rng[2]), lo=lo.tolist(), rng=rng.tolist(), pos=_b64(np.round((sf["xyz"] - lo) / rng * 65535).astype(np.uint16)),
                      tu=_b64(np.round(np.clip(sf["tu"], -1, 1) * 32767).astype(np.int16)), tv=_b64(np.round(np.clip(sf["tv"], -1, 1) * 32767).astype(np.int16)),
                      s=_b64(sf["scale"].astype(np.float16)), op=_b64(np.round(np.clip(sf["opacity"], 0, 1) * 255).astype(np.uint8)), col=_b64(col_b), col2=_b64(col_a), cls=_b64(cls))
    M["ctr"] = (lo + rng / 2).tolist()
    M["note"] = f"{status}: seed toggle 2d_masks (this run, {len(leaves_b)} compound leaves) vs clustering_only (Track A, {len(leaves_a)}); grey = axis / unavailable; height 0 = pot top; shape only, scale unvalidated"
    html = html[:m.start(1)] + json.dumps(M, separators=(",", ":")) + html[m.end(1):]
    title = f"{status} · 소엽 인스턴스 출처 비교 (2d_masks vs clustering_only)"
    html = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", html); html = re.sub(r'<h1 id="ttl">[^<]*</h1>', f'<h1 id="ttl">{title}</h1>', html)
    html = re.sub(r'(<button data-cm="0" class="on">)[^<]*(</button>)', r"\g<1>복엽 색\g<2>", html)
    banner = f"{status} · 2D 마스크 시드(H2(M1)) vs 3D 군집 시드 · 축·마디·화방 = unavailable · 형태만, 척도 미검증 · 계측 아님"
    html = re.sub(r'(z-index:50;background:#7a1f1f;[^>]*>)[^<]*(</div>)', lambda mm: mm.group(1) + banner + mm.group(2), html, count=1)
    def chips(leaves, pal):
        return "".join(f'<span style="display:inline-flex;align-items:center;gap:4px;margin:0 6px 4px 0;font-size:11px"><i style="width:10px;height:10px;border-radius:2px;background:rgb{pal[L["leaf_id"]]}"></i>{L["leaf_id"]} ·{L["n_leaflets"]}</span>' for L in leaves)
    ui = ('<label>소엽 인스턴스 출처</label><div class="row"><button id="sd0" class="on">2d_masks (이 실행)</button><button id="sd1">clustering_only (트랙 A)</button></div>'
          f'<div id="lg0"><label>복엽 {len(leaves_b)} (2d_masks)</label><div style="max-height:120px;overflow:auto">{chips(leaves_b, pal_b)}</div></div>'
          f'<div id="lg1" style="display:none"><label>복엽 {len(leaves_a)} (clustering_only)</label><div style="max-height:120px;overflow:auto">{chips(leaves_a, pal_a)}</div></div>')
    html = html.replace('<label>불투명도 배율', ui + '<label>불투명도 배율', 1)
    js = ("<script>(function(){const qc2=b64(D.col2);const C2=new Uint8Array(TW*TH*4);for(let i=0;i<N;i++){C2[i*4]=qc2[i*3];C2[i*4+1]=qc2[i*3+1];C2[i*4+2]=qc2[i*3+2];C2[i*4+3]=255;}"
          "function setSeed(s){tex(3,s?C2:C,false);document.getElementById('sd0').className=s?'':'on';document.getElementById('sd1').className=s?'on':'';"
          "document.getElementById('lg0').style.display=s?'none':'block';document.getElementById('lg1').style.display=s?'block':'none';if(window.__viewer)window.__viewer.draw();}"
          "document.getElementById('sd0').onclick=()=>setSeed(0);document.getElementById('sd1').onclick=()=>setSeed(1);})();</script>")
    html = html.replace("</body>", js + "</body>", 1) if "</body>" in html else html + js
    (out / "viewer").mkdir(exist_ok=True); p = out / "viewer" / "leaflet_2dseed_viewer.html"; p.write_text(html, encoding="utf-8"); return p


# ---------------------------------------------------------------- step 5 + 6 page: review of the new instances + mask correction on the declared views
CORRECTION_VIEWS = ["060313903", "060355657", "060440865", "060547546"]      # contract section 6: one training view per ring quadrant

CORR_HTML = r"""
<div class="card" style="margin-top:14px"><h2>6단계 · 마스크 교정 (대안, 선언된 4개 학습 뷰)</h2>
<p class="sub">H2(M1)의 2D 소엽 제안을 그대로 그려 두었습니다. 대부분은 <b>유지</b>하고, 두 잎이 붙은 것은 <b>자르기</b>(절단선을 클릭으로 그린 뒤 완료), 잎이 아니면 <b>삭제</b>, 빠진 소엽은 <b>추가</b>(꼭짓점 클릭 뒤 완료). 약 20분. 저장만 되고 자동 결과는 바뀌지 않습니다; 재시드는 별도 실행입니다.</p>
<div class="tabs" id="ctabs"></div>
<div class="row" style="margin-bottom:6px"><button id="cKeep" class="on">선택/유지</button><button id="cDel">삭제</button><button id="cCut">자르기</button><button id="cAdd">추가</button><button id="cDone">완료(선 닫기)</button><button id="cUndo">마지막 점 취소</button></div>
<div class="photo" id="cphoto"><img id="cimg" alt="correction photo"><svg id="cov"></svg></div>
<p class="sub" id="chint"></p><div class="list" id="clist"></div></div>
<script>
(function(){const CD=D.correction;if(!CD)return;const CS={v:0,mode:'keep',sel:null,draft:[],ops:{}};
const $=id=>document.getElementById(id);
async function cput(id,data){if(db){try{if(data===null)await db.doc('mask_corrections/'+id).delete();else await db.doc('mask_corrections/'+id).set(Object.assign({at:new Date().toISOString()},data));}catch(e){$('chint').textContent='저장 실패: '+(e.code||e.message);}}else{try{localStorage.setItem('lo_corr',JSON.stringify(CS.ops));}catch(e){}}}
async function cload(){if(db){const q=await db.collection('mask_corrections').get();q.docs.forEach(d=>{CS.ops[d.id]=d.data();});}else{try{CS.ops=JSON.parse(localStorage.getItem('lo_corr')||'{}');}catch(e){}}crender();}
function crender(){const V=CD.views[CS.v];$('cimg').src='data:image/jpeg;base64,'+V.jpg;const ov=$('cov');ov.setAttribute('viewBox','0 0 '+V.w+' '+V.h);let s='';
 V.polys.forEach((p,i)=>{const op=CS.ops[V.id+'|'+p.id];const del=op&&op.op==='delete';const col=p.suspect?'#e0a020':'#30c060';
  s+=`<polygon data-ci="${i}" class="${CS.sel===p.id?'sel':''}" points="${p.pts.map(q=>q.join(',')).join(' ')}" fill="${del?'#888':col}" fill-opacity="${del?0.1:0.22}" stroke="${del?'#888':col}" stroke-dasharray="${del?'6 4':'none'}"><title>${p.id}${p.suspect?' · merge_suspect p='+p.p:''}${op?' · '+op.op:''}</title></polygon>`;
  if(op&&op.op==='cut')op.cuts.forEach(c=>{s+=`<polyline points="${c.map(q=>q.join(',')).join(' ')}" fill="none" stroke="#ff3030" stroke-width="3"/>`;});});
 Object.entries(CS.ops).filter(([k,o])=>o.op==='add'&&o.view===V.id).forEach(([k,o])=>{s+=`<polygon data-add="${k}" points="${o.points.map(q=>q.join(',')).join(' ')}" fill="#4090ff" fill-opacity=".25" stroke="#4090ff"><title>${k} · 추가</title></polygon>`;});
 if(CS.draft.length)s+=`<polyline points="${CS.draft.map(q=>q.join(',')).join(' ')}" fill="none" stroke="#ff3030" stroke-width="3" stroke-dasharray="4 3"/>`+CS.draft.map(q=>`<circle cx="${q[0]}" cy="${q[1]}" r="5" fill="#ff3030"/>`).join('');
 ov.innerHTML=s;$('ctabs').innerHTML=CD.views.map((v,i)=>`<button data-cv="${i}" class="${i===CS.v?'on':''}">뷰 ${v.id} · 제안 ${v.polys.length}</button>`).join('');
 for(const [id,m] of [['cKeep','keep'],['cDel','delete'],['cCut','cut'],['cAdd','add']])$(id).className=CS.mode===m?'on':'';
 const n=Object.values(CS.ops);$('chint').textContent=`모드: ${CS.mode} · 선택: ${CS.sel||'-'} · 기록 ${n.length} (삭제 ${n.filter(o=>o.op==='delete').length}, 자르기 ${n.filter(o=>o.op==='cut').length}, 추가 ${n.filter(o=>o.op==='add').length})`;
 $('clist').innerHTML=Object.entries(CS.ops).filter(([k,o])=>o.view===V.id).map(([k,o])=>`<div>${k} <span class="v">${o.op} <button data-cundo="${k}" style="padding:1px 5px">지움</button></span></div>`).join('');}
$('ctabs').addEventListener('click',e=>{const t=e.target.closest('[data-cv]');if(t){CS.v=+t.dataset.cv;CS.sel=null;CS.draft=[];crender();}});
$('clist').addEventListener('click',async e=>{const t=e.target.closest('[data-cundo]');if(t){delete CS.ops[t.dataset.cundo];await cput(t.dataset.cundo,null);crender();}});
for(const [id,m] of [['cKeep','keep'],['cDel','delete'],['cCut','cut'],['cAdd','add']])$(id).onclick=()=>{CS.mode=m;CS.draft=[];crender();};
$('cUndo').onclick=()=>{CS.draft.pop();crender();};
$('cDone').onclick=async()=>{const V=CD.views[CS.v];if(CS.mode==='cut'&&CS.sel&&CS.draft.length>=2){const k=V.id+'|'+CS.sel;const o=CS.ops[k]&&CS.ops[k].op==='cut'?CS.ops[k]:{op:'cut',view:V.id,id:CS.sel,cuts:[]};o.cuts.push(CS.draft);CS.ops[k]=o;await cput(k,o);}
 else if(CS.mode==='add'&&CS.draft.length>=3){const k=V.id+'|ADD'+Date.now().toString(36);CS.ops[k]={op:'add',view:V.id,points:CS.draft};await cput(k,CS.ops[k]);}
 CS.draft=[];crender();};
$('cov').addEventListener('click',async e=>{const V=CD.views[CS.v];const r=$('cov').getBoundingClientRect();const x=Math.round((e.clientX-r.left)/r.width*V.w),y=Math.round((e.clientY-r.top)/r.height*V.h);const p=e.target.closest('polygon');
 if(CS.mode==='add'){CS.draft.push([x,y]);crender();return;}
 if(CS.mode==='cut'){if(!CS.sel){if(p&&p.dataset.ci!==undefined){CS.sel=V.polys[+p.dataset.ci].id;}crender();return;}CS.draft.push([x,y]);crender();return;}
 if(!p||p.dataset.ci===undefined)return;const id=V.polys[+p.dataset.ci].id;CS.sel=id;
 if(CS.mode==='delete'){const k=V.id+'|'+id;if(CS.ops[k]&&CS.ops[k].op==='delete'){delete CS.ops[k];await cput(k,null);}else{CS.ops[k]={op:'delete',view:V.id,id};await cput(k,CS.ops[k]);}}
 crender();});
const wait=setInterval(()=>{if(typeof db!=='undefined'){clearInterval(wait);cload();}},500);setTimeout(()=>{clearInterval(wait);if(!Object.keys(CS.ops).length)crender();},6000);crender();})();
</script>
"""


def correction_data(views_all, photos: Path, out: Path):
    import base64 as b64mod, io as iomod
    import cv2
    from tomato3d.organs import light_review as LR
    data = []
    for vid in CORRECTION_VIEWS:
        v = next(x for x in views_all if vid in x); j = json.load(open(out / "leaflet_instances" / (Path(v).stem + ".json"), encoding="utf-8"))
        img = Image.open(photos / v).convert("RGB"); img.thumbnail((LR.PAGE_PX, LR.PAGE_PX)); buf = iomod.BytesIO(); img.save(buf, "JPEG", quality=78)
        sc = img.size[0] / H2_GRID; polys = []
        for inst in j["instances"]:
            m = px.rle_decode(inst["rle"]).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea); c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2) * sc
            polys.append(dict(id=inst["id"], suspect=bool(inst.get("merge_suspect")), p=round(inst.get("p_merged") or 0, 2), pts=np.round(c, 1).tolist()))
        data.append(dict(id=vid, file=v, w=img.size[0], h=img.size[1], grid=H2_GRID, page_scale=sc, jpg=b64mod.b64encode(buf.getvalue()).decode(), polys=polys))
    return dict(views=data, note="H2(M1) proposals as editable polygons; coordinates in page px (page_scale x 1536-grid px)")


def build_review_page(out: Path, surfels: Path, poses: Path, photos: Path, views_all):
    from tomato3d.organs import light_review as LR
    data = LR.outlines(out, surfels, poses, LR.VIEWS, photos); data["correction"] = correction_data(views_all, photos, out); data["seed"] = SEED
    html = LR.HTML.replace("<title>R3-1b 소엽·복엽 검토</title>", "<title>R3-1b 2D 시드 소엽 검토·교정</title>")
    html = html.replace("<h1>R3-1b 소엽·복엽 검토</h1>", "<h1>R3-1b 2D 시드 소엽 검토·교정</h1>")
    html = html.replace('<div id="toast"></div>', CORR_HTML.replace("<script>", "<script id=\"corr\">", 1) + '<div id="toast"></div>')
    # the correction script must run after the main script (it reads D and db): move it to the end
    i = html.index('<script id="corr">'); j = html.index("</script>", i) + len("</script>"); corr = html[i:j]; html = html[:i] + html[j:] + corr
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (out / "review").mkdir(exist_ok=True); p = out / "review" / "review_page.html"; p.write_text(html, encoding="utf-8")
    json.dump(dict(review_views=[dict(id=v["id"], polygons=len(v["polys"])) for v in data["views"]], correction_views=[dict(id=v["id"], proposals=len(v["polys"]), suspect=sum(q["suspect"] for q in v["polys"])) for v in data["correction"]["views"]]),
              open(out / "review" / "review_inputs.json", "w"), indent=1)
    return p


# ---------------------------------------------------------------- Revision 2: eroded seeds, conditional growth, watershed (Itakura & Hosoi 2018, parts taken as declared)
GROW = dict(radius_mm=C.LEAF["cluster_radius_mm"], normal_window_mm=5.0, normal_deg=25.0, plane_mm=1.5, max_frac=0.5, max_rounds=50)
ERODE = dict(erode_px=6, grid=H2_GRID)
WS = dict(h2_min_instances=17, sigma_px=2.0, h_px=3.0, min_area_px=150, grid=H2_GRID, overlap_min=0.5, map_min_surfels=20)


def watershed_proposals(views, masks_dir: Path):
    """Part B: distance-transform watershed on the visible-plant mask of the views whose H2 count is below h2_min_instances."""
    import scipy.ndimage as ndi
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed
    out = {}
    for v in views:
        j = json.load(open(PROPOSALS / (Path(v).stem + ".json"), encoding="utf-8"))
        if len(j["instances"]) >= WS["h2_min_instances"]:
            continue
        mb = Image.open(Path(masks_dir) / (Path(v).stem + ".png")).resize((WS["grid"], WS["grid"]), Image.NEAREST); mask = np.asarray(mb) > 0
        dt = ndi.distance_transform_edt(mask); dts = ndi.gaussian_filter(dt, WS["sigma_px"])
        markers, _ = ndi.label(h_maxima(dts, WS["h_px"])); lab = watershed(-dts, markers, mask=mask)
        regs = []
        for q in range(1, lab.max() + 1):
            m = lab == q
            if m.sum() >= WS["min_area_px"]:
                regs.append((f"W{len(regs) + 1:03d}", m))
        out[v] = regs
    return out


def grow(R):
    """Part A2: conditional growth of the voted instances into unavailable non-axis surfels. Returns cl2, source, stats."""
    xyz, nrm, cl, is_axis = R["xyz"], R["nrm"], R["cl"].copy(), R["is_axis"]
    n = len(xyz); nn = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    cand_all = np.nonzero((cl < 0) & ~is_axis)[0]; tree = cKDTree(xyz); source = np.where(cl >= 0, "vote", np.where(is_axis, "axis", "unavailable")).astype(object)
    voted = Counter(cl[cl >= 0].tolist()); grown = Counter(); capped = set(); contested_total = 0; rounds = 0
    pairs = tree.query_pairs(GROW["radius_mm"] / 1000, output_type="ndarray")
    adj = defaultdict(list)
    for i, j in pairs:
        adj[i].append(j); adj[j].append(i)
    pending = set(cand_all.tolist())
    while rounds < GROW["max_rounds"]:
        rounds += 1; changed = 0; contested = 0; accept = {}
        for s in list(pending):
            labs = {int(cl[q]) for q in adj.get(s, []) if cl[q] >= 0}
            if not labs:
                continue
            if len(labs) > 1:
                contested += 1; continue
            L = labs.pop()
            if L in capped:
                continue
            near = [q for q in tree.query_ball_point(xyz[s], GROW["normal_window_mm"] / 1000) if cl[q] == L]
            if len(near) < 3:
                continue
            P = xyz[near]; mean_n = nn[near].mean(0); mean_n /= max(np.linalg.norm(mean_n), 1e-12)
            if np.degrees(np.arccos(min(1.0, abs(float(nn[s] @ mean_n))))) > GROW["normal_deg"]:
                continue
            c = P.mean(0); w, vec = np.linalg.eigh(np.cov((P - c).T)); pn = vec[:, 0]
            if abs(float((xyz[s] - c) @ pn)) > GROW["plane_mm"] / 1000:
                continue
            accept[s] = L
        # apply this round's accepted growth under the budget (instances are processed in id order; the budget is checked per surfel)
        for s, L in sorted(accept.items()):
            if grown[L] + 1 > GROW["max_frac"] * voted[L]:
                capped.add(L); continue
            cl[s] = L; source[s] = "grown"; grown[L] += 1; pending.discard(s); changed += 1
        contested_total = contested
        if changed == 0:
            break
    stats = dict(rounds=rounds, grown=int(sum(grown.values())), grown_share_of_blade=float(sum(grown.values()) / max((~is_axis).sum(), 1)),
                 candidates=int(len(cand_all)), still_unavailable=int(((cl < 0) & ~is_axis).sum()), contested_last_round=int(contested_total),
                 capped_instances=sorted(capped), per_instance_grown={int(k): int(v) for k, v in grown.items()})
    return cl, source, stats


def oversplit_rate(R):
    """Part B.5: watershed regions of one view that map (by >= 20 shared surfels) to the same H2 instance of another view."""
    V, views, index = R["V"], R["views"], R["index2d"]
    if not index:
        return None
    groups = Counter(); mapped = 0; total = 0; detail = []
    for k, v in enumerate(views):
        inst = index.get(v, [])
        for q, (iid, src) in enumerate(inst):
            if src != "watershed":
                continue
            total += 1; S = np.nonzero(V[:, k] == q + 1)[0]; best = None
            for k2, v2 in enumerate(views):
                if k2 == k:
                    continue
                ids = V[S, k2]; ids = ids[ids > 0]
                if len(ids) < WS["map_min_surfels"]:
                    continue
                i2, cnt = Counter(ids.tolist()).most_common(1)[0]
                i2src = index[v2][i2 - 1][1] if i2 - 1 < len(index[v2]) else "?"
                if i2src == "H2(M1)" and cnt >= WS["map_min_surfels"] and (best is None or cnt > best[2]):
                    best = (k2, int(i2), cnt)
            if best:
                mapped += 1; groups[(k, best[0], best[1])] += 1; detail.append(dict(view=v[9:18], region=iid, maps_to=f"{views[best[0]][9:18]}:{index[views[best[0]]][best[1] - 1][0]}", surfels=best[2]))
    over = sum(c for c in groups.values() if c >= 2)
    return dict(regions=total, mapped=mapped, in_oversplit_groups=over, rate=(over / mapped if mapped else None), groups_of_2plus=sum(1 for c in groups.values() if c >= 2), detail=detail[:200])


def write_outputs(R, a, out: Path, status: str, label: str, extra: dict):
    """Tables, surfels, tree, distributions, labels, acceptance for one labelled output (Revision 2 columns)."""
    out.mkdir(parents=True, exist_ok=True); input_sha = LO.sha256_file(a.surfels)
    lcsv, ccsv = LO.tables(R, status)
    lcsv = lcsv.replace("input_status R3-1b_unaccepted;", f"input_status {status}; seed {SEED}; column {label};", 1); ccsv = ccsv.replace("input_status R3-1b_unaccepted;", f"input_status {status}; seed {SEED}; column {label};", 1)
    (out / "leaflets.csv").write_text(lcsv, encoding="utf-8"); (out / "compound_leaves.csv").write_text(ccsv, encoding="utf-8")
    json.dump({x["leaflet_id"]: dict(leaf_id=x["leaf_id"], surfel_rows=x["rows"]) for x in R["leaflets"]} | {"_meta": dict(input_status=status, seed=SEED, column=label, input_sha256=input_sha)},
              open(out / "leaflet_surfels.json", "w", encoding="utf-8"))
    tree = LO.plant_tree(R, a, status, input_sha); tree["plant"]["attrs"].update(seed=SEED, column=label); json.dump(tree, open(out / "plant_tree.json", "w", encoding="utf-8"), indent=1)
    dist = LO.distributions(R, status, input_sha); dist.update(seed=SEED, column=label, note=dist["note"] + "; DO NOT consume before acceptance passes"); json.dump(dist, open(out / "distributions_microtom_synth_v2.json", "w", encoding="utf-8"), indent=1)
    src = R.get("source"); bucket = np.where(R["cl"] >= 0, 1, np.where(R["is_axis"], 4, 5)).astype(np.uint8)
    np.savez_compressed(out / "surfel_labels.npz", rows=R["rows"], bucket=bucket, cl=R["cl"], is_axis=R["is_axis"], n_seen=R["n_seen"], source=(src.astype("U12") if src is not None else np.where(R["cl"] >= 0, "vote", "unavailable")),
                        leaflet_id=np.array([next((x["leaflet_id"] for x in R["leaflets"] if x["cluster"] == c), "") for c in range(R["cl"].max() + 1)]),
                        leaf_id=np.array([next((x["leaf_id"] for x in R["leaflets"] if x["cluster"] == c), "") for c in range(R["cl"].max() + 1)]))
    acc = acceptance(R); json.dump(acc, open(out / "acceptance.json", "w", encoding="utf-8"), indent=1, default=float)
    summ = dict(input_status=status, seed=SEED, column=label, scale=LO.SCALE_TAG, input_sha256=input_sha, gpu_hours=0.0, plant_height_H_m=R["H"], views=R["n_views"],
                parameters=dict(rule=RULE, visibility=VIS, erode=ERODE, grow=GROW, watershed=WS), buckets=R["buckets"], coverage_ok=R["coverage_ok"],
                leaflets=len(R["leaflets"]), compound_leaves=len(R["leaves"]), leaflets_per_leaf=[L["n_leaflets"] for L in R["leaves"]],
                skeleton=dict(group_order=R["group_order"], nodes=R["nodes"], root_far=R["root_far"]), step2=dict(leak=R["leak"], graph=R["graph"], links=R["links"], unavailable_by_reason=R["unavailable_reasons"]),
                acceptance=acc, plausibility=LO.plausibility(R), organs_sha256={p.name: LO.sha256_file(p) for p in sorted((ROOT / "tomato3d/organs").glob("*.py"))}, **extra)
    json.dump(summ, open(out / "run_summary.json", "w", encoding="utf-8"), indent=1, default=float)
    return summ


def regroup(R, a, cl2, source):
    """Re-run the frozen grouping and the attributes on a changed cluster array (growth)."""
    r = group(R["xyz"], R["is_axis"], np.asarray(a.root, float), cl2, order=a.group_order)
    lfs, leaves = attributes(R["xyz"], R["tu"], R["tv"], R["sc"], R["rows"], R["n_seen"], cl2, r, R["agree"])
    if R.get("crit") is not None:
        row_to_i = {int(rr): i for i, rr in enumerate(R["rows"])}
        f3 = flag_instances_3d(R["xyz"], R["sc"], R["V"], R["views"], R["po"], {x["leaflet_id"]: np.array([row_to_i[rr] for rr in x["rows"]], int) for x in lfs}, R["crit"], {})
        for x in lfs:
            if f3[x["leaflet_id"]].get("merge_suspect"):
                x["flags"].append("merge_suspect")
    buckets = dict(leaflet_assigned=int(np.isin(cl2, [x["cluster"] for x in lfs if x["leaf_id"] != "unassigned"]).sum()),
                   leaflet_unassigned=int(np.isin(cl2, [x["cluster"] for x in lfs if x["leaf_id"] == "unassigned"]).sum()),
                   unavailable=int((cl2 < 0).sum() - R["is_axis"].sum()), axis=int(R["is_axis"].sum()))
    R2 = dict(R); R2.update(cl=cl2, source=source, leaflets=lfs, leaves=leaves, buckets=buckets, coverage_ok=sum(buckets.values()) == len(cl2), nodes=r["nodes"], group_order=r["order"], root_far=r["root_far"],
              flags3d=dict(this_run=f3) if R.get("crit") is not None else None)
    return R2


def run_variant(a, variant: str):
    status = a.input_status; root = Path(a.out); views = json.load(open(a.views))["train"]
    if variant == "erode_grow":
        sub = root / "attempt3_erode_grow"; sub.mkdir(parents=True, exist_ok=True); copy_instances(views, sub)
        a2 = argparse.Namespace(**vars(a)); a2.out = sub; a2.erode_px = ERODE["erode_px"]; a2.watershed = False
        R = compute(a2)
        extra = dict(erode=dict(erode_px=ERODE["erode_px"], per_view=R["erode_stats"], vanished_total=sum(e["vanished"] for e in R["erode_stats"].values())))
        s1 = write_outputs(R, a2, sub / "vote_only", status, "erode vote_only", extra)
        cl2, source, gstats = grow(R); R2 = regroup(R, a2, cl2, source)
        core_len = {x["leaflet_id"]: x["length_m"] * 1000 for x in R["leaflets"]}
        # the honesty check: a grown instance over 25 mm whose voted core was not (matched by cluster index)
        core_by_cluster = {x["cluster"]: x["length_m"] * 1000 for x in R["leaflets"]}
        exceed = [dict(leaflet_id=x["leaflet_id"], core_mm=core_by_cluster.get(x["cluster"]), grown_mm=x["length_m"] * 1000) for x in R2["leaflets"]
                  if x["length_m"] * 1000 > GATES["largest_leaflet_length_mm_max"] and (core_by_cluster.get(x["cluster"]) or 0) <= GATES["largest_leaflet_length_mm_max"]]
        for x in R2["leaflets"]:
            if x["cluster"] in gstats["capped_instances"]:
                x["flags"].append("grow_capped")
        s2 = write_outputs(R2, a2, sub / "grown", status, "erode grown", dict(erode=extra["erode"], growth=gstats, growth_exceeds_25mm=exceed,
                                                                             growth_adopted=(len(exceed) == 0), source_counts=dict(Counter(source.tolist()))))
        return dict(vote_only=s1, grown=s2)
    if variant == "watershed":
        sub = root / "attempt4_watershed"; sub.mkdir(parents=True, exist_ok=True); copy_instances(views, sub)
        a2 = argparse.Namespace(**vars(a)); a2.out = sub; a2.erode_px = ERODE["erode_px"]; a2.watershed = True
        R = compute(a2); ov = oversplit_rate(R)
        extra = dict(erode=dict(erode_px=ERODE["erode_px"], vanished_total=sum(e["vanished"] for e in R["erode_stats"].values())),
                     watershed=dict(params=WS, views=R["ws_stats"], regions_total=sum(w["regions"] for w in R["ws_stats"].values()), oversplit=ov))
        return dict(watershed=write_outputs(R, a2, sub, status, "watershed", extra))
    raise ValueError(variant)


def viewer_columns(out: Path, columns, surfels: Path, status: str):
    """One viewer with a colour set per column [(label, run_dir)]; height / normal modes kept; 0 = pot top."""
    sf = sio.read_surfels_full(surfels); n = len(sf["xyz"]); html = TEMPLATE.read_text(encoding="utf-8")
    m = re.search(r'<script id="D" type="application/json">(.*?)</script>', html, re.S); M = json.loads(m.group(1))
    cols = []; legends = []
    for label, run in columns:
        col, leaves, pal = leaf_colours(Path(run), n); cols.append(col); legends.append((label, leaves, pal))
    lab = np.load(Path(columns[0][1]) / "surfel_labels.npz"); cls = np.zeros(n, np.uint8); cls[lab["rows"]] = 1
    lo = sf["xyz"].min(0); rng = np.maximum(sf["xyz"].max(0) - lo, 1e-9)
    M["discs"] = dict(n=n, n_plant=int(cls.sum()), H=float(rng[2]), lo=lo.tolist(), rng=rng.tolist(), pos=_b64(np.round((sf["xyz"] - lo) / rng * 65535).astype(np.uint16)),
                      tu=_b64(np.round(np.clip(sf["tu"], -1, 1) * 32767).astype(np.int16)), tv=_b64(np.round(np.clip(sf["tv"], -1, 1) * 32767).astype(np.int16)),
                      s=_b64(sf["scale"].astype(np.float16)), op=_b64(np.round(np.clip(sf["opacity"], 0, 1) * 255).astype(np.uint8)), col=_b64(cols[0]), cls=_b64(cls),
                      cols=[_b64(c) for c in cols], labels=[c[0] for c in columns])
    M["ctr"] = (lo + rng / 2).tolist(); M["note"] = f"{status}: one colour set per column ({', '.join(c[0] for c in columns)}); grey = axis / unavailable; height 0 = pot top; shape only, scale unvalidated"
    html = html[:m.start(1)] + json.dumps(M, separators=(",", ":")) + html[m.end(1):]
    title = f"{status} · 소엽 인스턴스 열 비교 ({len(columns)})"
    html = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", html); html = re.sub(r'<h1 id="ttl">[^<]*</h1>', f'<h1 id="ttl">{title}</h1>', html)
    html = re.sub(r'(<button data-cm="0" class="on">)[^<]*(</button>)', r"\g<1>복엽/소엽 색\g<2>", html)
    banner = f"{status} · 열: {' / '.join(c[0] for c in columns)} · 축·마디·화방 = unavailable · 형태만, 척도 미검증 · 계측 아님"
    html = re.sub(r'(z-index:50;background:#7a1f1f;[^>]*>)[^<]*(</div>)', lambda mm: mm.group(1) + banner + mm.group(2), html, count=1)
    def chips(leaves, pal):
        return "".join(f'<span style="display:inline-flex;align-items:center;gap:4px;margin:0 6px 4px 0;font-size:11px"><i style="width:10px;height:10px;border-radius:2px;background:rgb{pal[L["leaf_id"]]}"></i>{L["leaf_id"]} ·{L["n_leaflets"]}</span>' for L in leaves)
    ui = '<label>열 (인스턴스 출처)</label><div class="row">' + "".join(f'<button id="sd{i}" class="{"on" if i == 0 else ""}">{lab_}</button>' for i, (lab_, _, _) in enumerate(legends)) + "</div>"
    ui += "".join(f'<div id="lg{i}" style="display:{"block" if i == 0 else "none"}"><label>복엽 {len(lv)} ({lab_}; 복엽 0 이면 소엽은 흰색 = 미배정)</label><div style="max-height:100px;overflow:auto">{chips(lv, pal)}</div></div>' for i, (lab_, lv, pal) in enumerate(legends))
    html = html.replace('<label>불투명도 배율', ui + '<label>불투명도 배율', 1)
    js = ("<script>(function(){const CS=D.cols.map(s=>{const q=b64(s);const C2=new Uint8Array(TW*TH*4);for(let i=0;i<N;i++){C2[i*4]=q[i*3];C2[i*4+1]=q[i*3+1];C2[i*4+2]=q[i*3+2];C2[i*4+3]=255;}return C2;});"
          "function setCol(s){tex(3,CS[s],false);CS.forEach((_,i)=>{document.getElementById('sd'+i).className=i===s?'on':'';document.getElementById('lg'+i).style.display=i===s?'block':'none';});if(window.__viewer)window.__viewer.draw();}"
          "CS.forEach((_,i)=>{document.getElementById('sd'+i).onclick=()=>setCol(i);});})();</script>")
    html = html.replace("</body>", js + "</body>", 1) if "</body>" in html else html + js
    (out / "viewer").mkdir(exist_ok=True); p = out / "viewer" / "leaflet_seed_columns_viewer.html"; p.write_text(html, encoding="utf-8"); return p


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--surfels", type=Path, default=LO.DEFAULTS["surfels"])
    ap.add_argument("--track-a", type=Path, default=TRACK_A); ap.add_argument("--poses", type=Path, default=LO.DEFAULTS["poses"]); ap.add_argument("--views", type=Path, default=LO.DEFAULTS["views"])
    ap.add_argument("--input-status", default="R3-1b_unaccepted"); ap.add_argument("--pot-top", type=float, default=LO.POT_TOP_Z); ap.add_argument("--root", type=float, nargs=3, default=list(C.POT_CENTRE_M))
    ap.add_argument("--repeat", action="store_true"); ap.add_argument("--visibility", choices=("point504", "splat1512"), default=VIS["mode"])
    ap.add_argument("--no-1b", dest="step1b", action="store_false"); ap.add_argument("--group-order", choices=("current", "pre_tracking"), default="current")
    ap.add_argument("--human-masks", type=Path, default=None, help="step 6: JSON export of the review page's mask_corrections collection {doc_id: {op, view, id, cuts|points}}")
    ap.add_argument("--variant", choices=("erode_grow", "watershed", "columns_viewer"), default=None, help="Revision 2 attempts (write into attempt3_/attempt4_ subfolders)")
    ap.add_argument("--masks", type=Path, default=LO.DEFAULTS["masks"])
    a = ap.parse_args(argv); status = a.input_status; LO.POT_TOP_Z_OF[0] = a.pot_top
    if a.variant == "columns_viewer":
        root = Path(a.out); cols = [("clustering_only", a.track_a), ("2d_masks vote", root), ("erode vote_only", root / "attempt3_erode_grow/vote_only"), ("erode grown", root / "attempt3_erode_grow/grown"), ("watershed", root / "attempt4_watershed")]
        print(viewer_columns(root, [(l, p) for l, p in cols if (Path(p) / "surfel_labels.npz").exists()], a.surfels, status)); return
    if a.variant:
        res = run_variant(a, a.variant); print(json.dumps({k: dict(leaflets=v["leaflets"], compound_leaves=v["compound_leaves"], acceptance=v["acceptance"], buckets=v["buckets"], **{kk: v[kk] for kk in ("growth", "growth_exceeds_25mm", "watershed", "erode") if kk in v}) for k, v in res.items()}, default=float, indent=0)); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); input_sha = LO.sha256_file(a.surfels)
    views = json.load(open(a.views))["train"]; counts = copy_instances(views, out); idcheck = {v[9:18]: check_label_ids(v) for v in views[:3]}
    R = compute(a); lcsv, ccsv = LO.tables(R, status); repro = None
    if a.repeat:
        R2 = compute(a); l2, c2 = LO.tables(R2, status); repro = dict(leaflets_csv_identical=l2 == lcsv, compound_leaves_csv_identical=c2 == ccsv)
    lcsv = lcsv.replace("input_status R3-1b_unaccepted;", f"input_status {status}; seed {SEED};", 1); ccsv = ccsv.replace("input_status R3-1b_unaccepted;", f"input_status {status}; seed {SEED};", 1)
    (out / "leaflets.csv").write_text(lcsv, encoding="utf-8"); (out / "compound_leaves.csv").write_text(ccsv, encoding="utf-8")
    json.dump({x["leaflet_id"]: dict(leaf_id=x["leaf_id"], surfel_rows=x["rows"]) for x in R["leaflets"]} | {"_meta": dict(input_status=status, seed=SEED, input_sha256=input_sha)},
              open(out / "leaflet_surfels.json", "w", encoding="utf-8"))
    json.dump(dict(seed=SEED, rule=RULE, graph=R["graph"], links=R["links"], leaflet_to_2d_ids=R["link_map"]), open(out / "leaflet_links.json", "w", encoding="utf-8"), indent=1)
    tree = LO.plant_tree(R, a, status, input_sha); tree["plant"]["attrs"]["seed"] = SEED; tree["plant"]["attrs"]["organ_split"] = "leaflets_and_compound_leaves_only; instances seeded from H2(M1) 2D masks"
    json.dump(tree, open(out / "plant_tree.json", "w", encoding="utf-8"), indent=1)
    dist = LO.distributions(R, status, input_sha); dist["seed"] = SEED; json.dump(dist, open(out / "distributions_microtom_synth_v2.json", "w", encoding="utf-8"), indent=1)
    np.savez_compressed(out / "surfel_labels.npz", rows=R["rows"], bucket=np.where(R["cl"] >= 0, 1, np.where(R["is_axis"], 4, 5)).astype(np.uint8), cl=R["cl"], is_axis=R["is_axis"], n_seen=R["n_seen"],
                        leaflet_id=np.array([next((x["leaflet_id"] for x in R["leaflets"] if x["cluster"] == c), "") for c in range(R["cl"].max() + 1)]),
                        leaf_id=np.array([next((x["leaf_id"] for x in R["leaflets"] if x["cluster"] == c), "") for c in range(R["cl"].max() + 1)]))
    acc = acceptance(R); json.dump(acc, open(out / "acceptance.json", "w", encoding="utf-8"), indent=1, default=float)
    if R.get("criterion"):
        json.dump(dict(contract="section 1b", features=list(FEATS), rule=CRIT, report=R["criterion"], flags_2d_per_view={v: dict(n=len(f), suspect=sum(1 for s in f.values() if s.get("merge_suspect"))) for v, f in R["flags2d"].items()},
                       flags_3d=R["flags3d"]), open(out / "criterion.json", "w", encoding="utf-8"), indent=1, default=float)
    ta = json.load(open(Path(a.track_a) / "run_summary.json"))
    bl_p = out / "baseline_current_code" / "run_summary.json"; bl = json.load(open(bl_p)) if bl_p.exists() else None
    summ = dict(input_status=status, seed=SEED, scale=LO.SCALE_TAG, input_sha256=input_sha, gpu_hours=0.0, plant_height_H_m=R["H"], views=R["n_views"],
                human_masks=(str(a.human_masks) if a.human_masks else None), corrections_applied=R.get("corrections"),
                step1=dict(source="H2(M1) reused (2026-10-22 run on this capture)", per_view_instances=counts, median_per_view=float(np.median(list(counts.values()))),
                           label_id_check_iou_first_instance=idcheck, domain_shift="trained on tray top views + dwt_101-104; these are ring views on a board"),
                step2=dict(visibility=a.visibility, leak=R["leak"], rule=RULE, graph=R["graph"], links=R["links"], unavailable_by_reason=R["unavailable_reasons"], axis_surfels_with_leaflet_votes=R["axis_surfels_with_leaflet_votes"]),
                buckets=R["buckets"], coverage_ok=R["coverage_ok"], leaflets=len(R["leaflets"]), leaflets_unassigned=sum(x["leaf_id"] == "unassigned" for x in R["leaflets"]),
                compound_leaves=len(R["leaves"]), leaflets_per_leaf=[L["n_leaflets"] for L in R["leaves"]],
                skeleton=dict(group_order=R["group_order"], nodes=R["nodes"], root_far=R["root_far"], apex_cuts=R["apex_cuts"], stubs=R["stubs"], truss_candidates=R["trusses"]),
                step1b=(dict(on=True, heldout_merged=R["criterion"]["heldout_merged"], dev_merged=R["criterion"]["dev_merged"],
                             suspect_2d=sum(sum(1 for s in f.values() if s.get("merge_suspect")) for f in R["flags2d"].values()), proposals_2d=sum(len(f) for f in R["flags2d"].values()),
                             track_a_flagged={k: v.get("merge_suspect") for k, v in R["flags3d"]["track_a"].items()},
                             baseline_flagged=({k: v.get("merge_suspect") for k, v in R["flags3d"]["baseline_current_code"].items()} if R["flags3d"]["baseline_current_code"] else None))
                        if R.get("criterion") else dict(on=False)),
                baseline_current_code=(dict(leaflets=bl["leaflets"], compound_leaves=bl["compound_leaves"], largest_mm=bl["plausibility"]["check1_largest_leaflet_length_mm"]["largest"],
                                            nodes=bl["pipeline_counts"]["nodes"], differs_from_track_a=(bl["compound_leaves"] != ta["compound_leaves"] or bl["leaflets"] != ta["leaflets"])) if bl else None),
                organs_sha256={p.name: LO.sha256_file(p) for p in sorted((ROOT / "tomato3d/organs").glob("*.py"))},
                acceptance=acc, reproducibility=repro, plausibility=LO.plausibility(R),
                compare_track_a=dict(leaflets=(ta["leaflets"], len(R["leaflets"])), compound_leaves=(ta["compound_leaves"], len(R["leaves"])),
                                     largest_leaflet_mm=(ta["plausibility"]["check1_largest_leaflet_length_mm"]["largest"], acc["largest_leaflet_length_mm"]["value"]),
                                     leaflets_per_leaf_median=(ta["plausibility"]["check3_leaflets_per_compound_leaf"].get("median"), acc["leaflets_per_leaf_median"]["value"])))
    json.dump(summ, open(out / "run_summary.json", "w", encoding="utf-8"), indent=1, default=float)
    p = viewer_toggle(out, Path(a.track_a), a.surfels, status)
    from tomato3d.organs import light_review as LR
    pr = build_review_page(out, a.surfels, a.poses, LR.PHOTOS, views); print(pr)
    print(json.dumps({k: summ[k] for k in ("buckets", "leaflets", "compound_leaves", "leaflets_per_leaf", "acceptance", "reproducibility", "step2")}, default=float, indent=0)); print(p)


if __name__ == "__main__":
    main()
