"""Addendum to the stacked-sheet retry: 3D-first over-segmentation (4a), SAM co-membership votes one photo at a time (4b), greedy merge (4c), proxy score
and a user grouping page (4d). Contract: build/p1_stacked_sheets_retry_20261026_01/CONTRACT.md, Amendment A1.

User labels are used only to score (and for the labelled proxy row), never to cut, vote or merge.

usage: python -m tomato3d.organs.oversegment --step 4a
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.organs import light_organs as LO
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vp_recluster as RC

ROOT = LO.ROOT
OUT = SR.OUT
STATUS = "R3-1b_unaccepted"; LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="oversegment_3d_first")
A = dict(normal_dev_deg=35.0, reattach_mm=3.0, kmeans_seed=0, gate_4a=0.90)


def load_base():
    pl = UL.load_plant(); b = np.nonzero(pl["blade"])[0]; xyz = pl["xyz"][b]
    zs = np.load(OUT / "sheets" / "sheets.npz"); zp = np.load(OUT / "patches" / "patches.npz"); assert np.array_equal(zs["rows"], pl["rows"][b]) and np.array_equal(zp["rows"], pl["rows"][b])
    S = dict(normal=zs["normal"].astype(float), n_sheets=zs["n_sheets"], index=zs["index"], gap=zs["gap"], peaks=zs["peaks"])
    W = np.load(SR.FIRST / "contributions.npz")["W"]; VISb = (W >= SR.P["w_min"])[b]
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; nmap = json.load(open(RC.MAP_FILE, encoding="utf-8"))["map"]
    prj = SR.projections(pl, b, chosen); umaps = SR.user_maps(chosen, nmap); lc = SR.leaflet_counts(pl, b, chosen, nmap, prj, VISb)
    return dict(pl=pl, b=b, xyz=xyz, S=S, patch=zp["patch"], VISb=VISb, chosen=chosen, held=vc["held_out"], nmap=nmap, prj=prj, umaps=umaps, lc=lc)


def split_patches(xyz, S, patch, E, p75, med):
    """4a: large patches -> normal-coherent components on the sheet edges (dropped surfels re-attached within 3 mm) -> k-means if still large."""
    from sklearn.cluster import KMeans
    ok = SR.edge_ok(xyz, S, E, SR.P["offset_mm"]); Es = E[ok]; out = np.full(len(xyz), -1); nxt = 0; stats = Counter()
    for p in np.unique(patch[patch >= 0]):
        idx = np.nonzero(patch == p)[0]
        if len(idx) <= p75:
            out[idx] = nxt; nxt += 1; stats["kept"] += 1; continue
        stats["split"] += 1; nrm = S["normal"][idx]; mn = np.median(nrm, axis=0); mn /= max(np.linalg.norm(mn), 1e-12)
        dev = np.degrees(np.arccos(np.clip(np.abs(nrm @ mn), 0, 1))); keep = dev <= A["normal_dev_deg"]; kidx = idx[keep]; stats["dropped"] += int((~keep).sum())
        pos = {g: k for k, g in enumerate(kidx)}; m = np.isin(Es[:, 0], kidx) & np.isin(Es[:, 1], kidx)
        e = np.array([[pos[i], pos[j]] for i, j in Es[m]], int).reshape(-1, 2); n = len(kidx)
        G = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr() if len(e) else coo_matrix((n, n)).tocsr()
        nc, lab = connected_components(G, directed=False) if n else (0, np.zeros(0, int)); full = np.full(len(idx), -1); full[keep] = lab
        if (~keep).any() and n:
            d, j = cKDTree(xyz[kidx]).query(xyz[idx[~keep]]); full[~keep] = np.where(d <= A["reattach_mm"] / 1000, lab[j], -1)
        for c in range(nc):
            cidx = idx[full == c]
            if len(cidx) > p75:
                k = max(2, int(round(len(cidx) / med))); km = KMeans(n_clusters=k, n_init=4, random_state=A["kmeans_seed"]).fit(xyz[cidx]); stats["kmeans"] += 1
                for q in range(k):
                    out[cidx[km.labels_ == q]] = nxt; nxt += 1
            elif len(cidx):
                out[cidx] = nxt; nxt += 1
        stats["unassigned_after_split"] += int((full < 0).sum())
    return out, dict(stats)


def purity_split(patch, S, prj, umaps, VISb):
    """Area-weighted purity overall and restricted to surfels on the upper / lower sheet of multi-sheet columns (patch majority from all its projections)."""
    rows, aw = SR.purity(patch, prj, umaps, VISb); maj = {p: r["majority"] for p, r in rows.items()}
    upper = (S["n_sheets"] > 1) & (S["index"] == S["n_sheets"] - 1); lower = (S["n_sheets"] > 1) & (S["index"] == 0); out = dict(all=aw)
    for name, sel in (("upper", upper), ("lower", lower), ("single_sheet", S["n_sheets"] == 1)):
        hit = tot = 0
        for vid, p in prj.items():
            vis = VISb[:, p["k"]] & p["inside"] & sel & (patch >= 0); ul = umaps[vid][p["wi"], p["ui"]]; m = vis & (ul > 0)
            mj = np.array([maj.get(int(x), -1) for x in patch[m]]); hit += int((mj == ul[m]).sum()); tot += int(m.sum())
        out[name] = dict(purity=(round(hit / tot, 4) if tot else None), labelled_projections=tot)
    return rows, out


def step_4a(out: Path):
    t0 = time.time(); B = load_base(); lc = B["lc"]; p75 = int(np.percentile(lc, 75)); med = int(np.median(lc))
    E = SR.edges_3mm(B["xyz"]); new, st = split_patches(B["xyz"], B["S"], B["patch"], E, p75, med)
    rows, pur = purity_split(new, B["S"], B["prj"], B["umaps"], B["VISb"]); _, pur0 = purity_split(B["patch"], B["S"], B["prj"], B["umaps"], B["VISb"])
    sizes = np.bincount(new[new >= 0]); passed = pur["all"] >= A["gate_4a"]
    res = dict(**LABELS, step="4a", thresholds=dict(leaflet_surfels_p75=p75, leaflet_surfels_median=med, labelled_leaflets=int(len(lc)), **A), split_stats=st,
               patches_before=int(len(np.unique(B["patch"][B["patch"] >= 0]))), patches_after=int(len(sizes)), unassigned_after=int((new < 0).sum()),
               patch_size=dict(median=int(np.median(sizes)), p90=int(np.percentile(sizes, 90)), max=int(sizes.max())),
               purity_before=pur0, purity_after=pur, gate_4a_passed=bool(passed), patches_with_labels=len(rows), seconds=round(time.time() - t0))
    (out / "oversegment").mkdir(exist_ok=True); json.dump(res, open(out / "oversegment" / "step4a.json", "w", encoding="utf-8"), indent=1, default=float)
    np.savez_compressed(out / "oversegment" / "patches_4a.npz", rows=B["pl"]["rows"][B["b"]], patch=new)
    print(json.dumps(res, indent=1, default=float)); return res


# ---------------------------------------------------------------- 4b-4d
B4 = dict(min_px=30, inside=0.70, overlap_max=0.01, adj_mm=4.0, min_views=3, merge_min=0.5, cannot_views=3)


def view_masks(vid, chosen):
    """Candidate masks of one photo (bool, 1536 grid): UNION for the labelling photos, stored H2(M1) proposals otherwise. User masks never."""
    from tomato3d.leafseg import pixels as px
    if vid in chosen:
        return [UL.unrle(m["rle"]) for m in json.load(open(UL.OUT / "detect" / f"{vid}.json", encoding="utf-8"))["masks"]], "UNION"
    from tomato3d.organs import seed2d as S2
    j = json.load(open(S2.PROPOSALS / f"20261021_{vid}_iOS.json", encoding="utf-8")); return [px.rle_decode(i["visible_segmentation"]) for i in j["instances"]], "H2(M1)"


def adjacency(xyz, S, patch):
    E = cKDTree(xyz).query_pairs(B4["adj_mm"] / 1000, output_type="ndarray"); i, j = E[:, 0], E[:, 1]; pi, pj = patch[i], patch[j]
    ok = (pi >= 0) & (pj >= 0) & (pi != pj); multi = (S["n_sheets"][i] > 1) & (S["n_sheets"][j] > 1); ok &= ~multi | (S["index"][i] == S["index"][j])
    a, b_ = np.minimum(pi[ok], pj[ok]), np.maximum(pi[ok], pj[ok]); return sorted(set(zip(a.tolist(), b_.tolist())))


def votes(B, patch, pairs, vis_views):
    """Per view and adjacent patch pair: +1 same / -1 different / 0, weight = smaller visible pixel count."""
    from tomato3d.organs import vote2d as V2
    pl, b = B["pl"], B["b"]; ppair = np.array(pairs, int); V = {}; used = {}
    for vid, (vis, src) in vis_views.items():
        view = f"20261021_{vid}_iOS.jpg"; pv = V2.project_view(pl["xyz"][:1], pl["sc"][:1], pl["po"], view, V2.VISP["res"])
        u, w, z = V2.project_points(B["xyz"], pv); s = UL.H2 / V2.VISP["res"]; ui = np.round(u * s).astype(int); wi = np.round(w * s).astype(int)
        ok = vis & (z > 0) & (ui >= 0) & (wi >= 0) & (ui < UL.H2) & (wi < UL.H2) & (patch >= 0)
        lin = wi[ok] * UL.H2 + ui[ok]; pp = patch[ok]; key = np.unique(pp.astype(np.int64) * (UL.H2 * UL.H2) + lin); Pk = key // (UL.H2 * UL.H2); Lk = key % (UL.H2 * UL.H2)
        npx = np.bincount(Pk, minlength=patch.max() + 1); masks, srcname = view_masks(vid, B["chosen"]); used[vid] = dict(source=srcname, masks=len(masks))
        if not masks: continue
        F = np.zeros((len(masks), patch.max() + 1))
        for mi, m in enumerate(masks):
            F[mi] = np.bincount(Pk[m.ravel()[Lk]], minlength=patch.max() + 1)
        F = F / np.maximum(npx, 1)[None]; area = np.array([m.sum() for m in masks])
        cand = np.nonzero((npx[ppair[:, 0]] >= B4["min_px"]) & (npx[ppair[:, 1]] >= B4["min_px"]))[0]; best = F.argmax(0); bestf = F.max(0); out = {}
        for e in cand:
            p, q = ppair[e]; wgt = int(min(npx[p], npx[q]))
            if ((F[:, p] >= B4["inside"]) & (F[:, q] >= B4["inside"])).any(): out[e] = (1, wgt); continue
            if bestf[p] >= B4["inside"] and bestf[q] >= B4["inside"] and best[p] != best[q]:
                inter = int((masks[best[p]] & masks[best[q]]).sum())
                if inter <= B4["overlap_max"] * min(area[best[p]], area[best[q]]): out[e] = (-1, wgt)
        V[vid] = out
    return V, used


def merge(pairs, V, npatch):
    S_ = np.zeros(len(pairs)); D_ = np.zeros(len(pairs)); nv = np.zeros(len(pairs), int); ndiff = np.zeros(len(pairs), int)
    for out in V.values():
        for e, (s, wgt) in out.items():
            nv[e] += 1
            if s > 0: S_[e] += wgt
            else: D_[e] += wgt; ndiff[e] += 1
    wgt = np.where(nv >= B4["min_views"], (S_ - D_) / np.maximum(S_ + D_, 1e-9), np.nan)
    parent = list(range(npatch)); cannot = {p: set() for p in range(npatch)}
    for e in np.nonzero(ndiff >= B4["cannot_views"])[0]:
        p, q = pairs[e]; cannot[p].add(q); cannot[q].add(p)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    members = {p: {p} for p in range(npatch)}; merged = blocked = 0
    for e in sorted(np.nonzero(np.isfinite(wgt) & (wgt >= B4["merge_min"]))[0], key=lambda e: -wgt[e]):
        a, c = find(pairs[e][0]), find(pairs[e][1])
        if a == c: continue
        if any(x in members[c] for m in members[a] for x in cannot[m]): blocked += 1; continue
        parent[c] = a; members[a] |= members.pop(c); merged += 1
    group = np.array([find(p) for p in range(npatch)])
    return group, dict(edges=len(pairs), edges_with_votes=int((nv > 0).sum()), edges_ge3_views=int((nv >= 3).sum()), views_per_edge_median=float(np.median(nv)),
                       views_per_edge_p10=float(np.percentile(nv, 10)), views_per_voted_edge_median=(float(np.median(nv[nv > 0])) if (nv > 0).any() else 0.0),
                       merged=merged, blocked_by_cannot_link=blocked, cannot_link_pairs=int((ndiff >= 3).sum())), nv, S_, D_


def leaflet_maps(chosen):
    """Per labelling photo: id map of the user's leaflet masks (any leaf number), for leaflet-level purity only."""
    recs = UL.load_labels(UL.OUT); out = {}
    for vid in chosen:
        det = {m["id"]: m for m in json.load(open(UL.OUT / "detect" / f"{vid}.json", encoding="utf-8"))["masks"]}; act = {}
        for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
            if r.get("removed"): act.pop(r["removed"], None); continue
            act[r.get("mask_id") or r["id"]] = r
        img = np.zeros((UL.H2, UL.H2), np.int32); k = 0
        for r in act.values():
            if r["leaf"] in ("모름", None): continue
            m = UL.record_rle(r, recs, det)
            if m is None: continue
            k += 1; img[m] = k
        out[vid] = img
    return out


def leaflet_metrics(group_s, B, lmaps, sel=None):
    """Leaflet-level purity, over-merges and under-merges of per-surfel group ids against the user's leaflet masks (4 labelling photos)."""
    per_gv = {}; per_leaf = {}
    for vid, p in B["prj"].items():
        vis = B["VISb"][:, p["k"]] & p["inside"] & (group_s >= 0) & (sel if sel is not None else True); lid = lmaps[vid][p["wi"], p["ui"]]; m = vis & (lid > 0)
        for g, l in zip(group_s[m], lid[m]):
            per_gv.setdefault((int(g), vid), Counter())[int(l)] += 1; per_leaf.setdefault((vid, int(l)), Counter())[int(g)] += 1
    hit = sum(c.most_common(1)[0][1] for c in per_gv.values()); tot = sum(sum(c.values()) for c in per_gv.values())
    over_g = {}
    for (g, vid), c in per_gv.items():
        n = sum(c.values())
        if n < B4["min_px"]: continue
        two = sum(1 for v in c.values() if v / n >= 0.20) >= 2; over_g.setdefault(g, []).append(two)
    over = sum(1 for g, f in over_g.items() if sum(f) >= len(f) / 2 and sum(f) > 0); groups = len(set(g for g, _ in per_gv))
    under = sum(1 for c in per_leaf.values() if sum(1 for v in c.values() if v / sum(c.values()) >= 0.10) > 2)
    return dict(leaflet_purity=(round(hit / tot, 4) if tot else None), labelled_projections=tot, groups_with_labels=groups, over_merges=over,
                over_merge_share=(round(over / max(groups, 1), 4)), under_merges=under, human_leaflet_masks=len(per_leaf))


def step_4bcd(out: Path):
    from tomato3d.organs import stacked_sheets as SS
    t0 = time.time(); B = load_base(); z4 = np.load(out / "oversegment" / "patches_4a.npz"); patch = z4["patch"]; npatch = int(patch.max()) + 1; pl = B["pl"]; b = B["b"]
    W = np.load(SR.FIRST / "contributions.npz")["W"]; views_train = pl["views"]; con = json.load(open(ROOT / "build/p1_recon_r3-1_20261022_01/contract.json"))
    _, _, nrm_, op = SS.load(); vis_views = {}
    for v in views_train:
        vid = v[9:18]
        if vid != B["held"][:9] and vid != B["held"]: vis_views[vid] = ((W[:, views_train.index(v)] >= SR.P["w_min"])[b], None)
    for v in con["heldout_fresh"]:
        w_ = SS.contributions(pl, op, v); vis_views[v[9:18]] = ((w_ >= SR.P["w_min"])[b], None)
    vis_views.pop(B["held"], None); vis_views.pop("060551789", None)
    pairs = adjacency(B["xyz"], B["S"], patch); print(f"views {len(vis_views)}, adjacent patch pairs {len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
    V, used = votes(B, patch, pairs, vis_views); print(f"votes done ({time.time() - t0:.0f}s)", flush=True)
    group_p, mst, nv, S_, D_ = merge(pairs, V, npatch); gid = {g: k for k, g in enumerate(sorted(set(group_p.tolist())))}; group_p = np.array([gid[g] for g in group_p])
    group_s = np.where(patch >= 0, group_p[np.maximum(patch, 0)], -1)
    # 4b agreement with the human numbers (patch majority from the labelling photos)
    rows4a, _ = SR.purity(patch, B["prj"], B["umaps"], B["VISb"]); maj = {p: r["majority"] for p, r in rows4a.items()}; same_ok = same_n = diff_ok = diff_n = 0
    for out_ in V.values():
        for e, (s, _) in out_.items():
            p, q = pairs[e]
            if p in maj and q in maj:
                if s > 0: same_n += 1; same_ok += int(maj[p] == maj[q])
                else: diff_n += 1; diff_ok += int(maj[p] != maj[q])
    lmaps = leaflet_maps(B["chosen"]); upper = (B["S"]["n_sheets"] > 1) & (B["S"]["index"] == B["S"]["n_sheets"] - 1); lower = (B["S"]["n_sheets"] > 1) & (B["S"]["index"] == 0)
    L_all = leaflet_metrics(group_s, B, lmaps); L_up = leaflet_metrics(group_s, B, lmaps, upper); L_lo = leaflet_metrics(group_s, B, lmaps, lower); L_patch = leaflet_metrics(patch, B, lmaps)
    # 4d proxy: majority human number per group (labelling photos) -> held-out score with the imported code
    rowsg, purg = SR.purity(group_s, B["prj"], B["umaps"], B["VISb"]); glab = {g: r["majority"] for g, r in rowsg.items()}
    lab = np.array([glab.get(int(g), 0) if g >= 0 else 0 for g in group_s], np.int16); full = np.zeros(len(pl["xyz"]), np.int16); full[b] = lab
    H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = f"20261021_{B['held']}_iOS.jpg"; col = H2id[:, pl["views"].index(hv)]
    raw, _ = UL.label_maps(UL.OUT, [B["held"]]); hm = UL.remap(raw, B["nmap"])[B["held"]]; hm = np.where(hm > 0, hm, 0).astype(np.int16)
    ren = UL.render_ids(pl, full, hv, col); rows_h, g_h = UL.score_heldout(hm, ren); cov = float(((ren > 0) & (hm > 0)).sum() / max((hm > 0).sum(), 1))
    flags = {str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rows_h.items()}
    s4a = json.load(open(out / "oversegment" / "step4a.json", encoding="utf-8"))
    passed = s4a["gate_4a_passed"] and (L_all["leaflet_purity"] or 0) >= 0.85 and L_all["over_merge_share"] <= 0.10 and g_h["median_iou"] > SR.PASS["ref_iou"] and g_h["max_confusion"] <= SR.PASS["ref_conf"]
    res = dict(**LABELS, step="4b-4d", views=len(vis_views), candidate_sources=dict(Counter(u["source"] for u in used.values())), qc_rejected_views_excluded=15, params=B4,
               patches=npatch, groups=int(len(set(group_p.tolist()))), merge=mst,
               vote_agreement=dict(same_votes=same_n, same_joins_same_human_number=(round(same_ok / same_n, 4) if same_n else None), different_votes=diff_n,
                                   different_separates_different_number=(round(diff_ok / diff_n, 4) if diff_n else None)),
               leaflet_level=dict(groups_all=L_all, upper_sheet=L_up, lower_sheet=L_lo, patches_4a_for_comparison=L_patch),
               human_leaflet_masks_per_view=round(L_all["human_leaflet_masks"] / 4, 1), leaf_number_purity_groups=purg,
               scoring_proxy_uses_human_numbers=dict(rows={str(L): v for L, v in rows_h.items()}, summary=g_h, coverage=round(cov, 4), flags=flags),
               reference_row=dict(median_iou=0.517, max_confusion=0.393, coverage=0.771), passed=bool(passed),
               tag=("gate_passed" if g_h["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(out / "oversegment" / "step4bcd.json", "w", encoding="utf-8"), indent=1, default=float)
    np.savez_compressed(out / "oversegment" / "groups_4c.npz", rows=pl["rows"][b], patch=patch, group=group_s, proxy_leaf=lab)
    print(json.dumps({k: v for k, v in res.items() if k not in ("leaflet_level",)}, indent=1, default=float)); print("leaflet", json.dumps(res["leaflet_level"], indent=1)); return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--step", choices=("4a", "4bcd"), default="4a"); a = ap.parse_args(argv)
    step_4a(a.out) if a.step == "4a" else step_4bcd(a.out)


if __name__ == "__main__":
    main()
