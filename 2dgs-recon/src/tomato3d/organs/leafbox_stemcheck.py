"""Schematic stem as a size / split / merge check for compound-leaf boxes (topology only; no stem mm values; proposals only).
Contract: build/p1_leafbox_20261029_01/stemcheck/CONTRACT.md.

usage: python -m tomato3d.organs.leafbox_stemcheck
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import light_organs as LO

R = LO.ROOT / "build"; OUT = R / "p1_leafbox_20261029_01/stemcheck"
F = dict(big=3.0, small=0.25, ds_split=0.08, side_deg=90.0, branch_mm=6.0, ds_merge=0.04, dir_deg=25.0, orphan_mm=30.0, comp_min=20, edge_mm=3.0, truth_share=0.15)


def polyline_param(poly):
    seg = np.diff(poly, axis=0); L = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(L)]
    def proj(p):
        best = (np.inf, 0.0, None)
        for i in range(len(seg)):
            t = np.clip(((p - poly[i]) @ seg[i]) / max(L[i] ** 2, 1e-12), 0, 1); q = poly[i] + t * seg[i]; d = np.linalg.norm(p - q)
            if d < best[0]: best = (d, (cum[i] + t * L[i]) / cum[-1], q)
        return best
    return proj


def comps(X, idx, r):
    if len(idx) == 0: return np.zeros(0, int), None
    E = cKDTree(X[idx]).query_pairs(r, output_type="ndarray"); m = len(idx); w = np.linalg.norm(X[idx][E[:, 0]] - X[idx][E[:, 1]], axis=1) + 1e-6 if len(E) else np.zeros(0)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])) if len(E) else ([], ([], [])), shape=(m, m)).tocsr(); return connected_components(G, directed=False)[1], G


def axis_dir(P):
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); return c, vt[0], (P - c) @ vt[0]


def az(v): return float(np.degrees(np.arctan2(v[1], v[0])))


def check(name, X, area, own_lab, rach_mask, human, proj, sigma=None):
    leaves = sorted(set(own_lab[own_lab > 0].tolist())); A = {k: float(area[own_lab == k].sum()) for k in leaves}; med = float(np.median(list(A.values()))); boxes = {}
    for k in leaves:
        ra = np.nonzero((own_lab == k) & rach_mask)[0]; own = np.nonzero(own_lab == k)[0]; src = ra if len(ra) >= F["comp_min"] else own
        c, u, t = axis_dir(X[src]); e1, e2 = c + u * t.min(), c + u * t.max(); p1, p2 = proj(e1), proj(e2); (dist, s, q), end = (p1, e1) if p1[0] <= p2[0] else (p2, e2)
        cands = []; cc, G = comps(X, ra, F["edge_mm"])
        for cid in (np.unique(cc) if len(cc) else []):
            mem = ra[cc == cid]
            if len(mem) < F["comp_min"]: continue
            cm, um, tm = axis_dir(X[mem]); a1, a2 = cm + um * tm.min(), cm + um * tm.max(); q1, q2 = proj(a1), proj(a2); (dd, ss, qq), ee = (q1, a1) if q1[0] <= q2[0] else (q2, a2)
            cands.append(dict(n=int(len(mem)), s=round(float(ss), 3), dist_mm=round(float(dd), 1), az=round(az(ee - qq), 1)))
        ybr = []
        if len(cc):
            big = Counter(cc.tolist()).most_common(1)[0][0]; sel = np.nonzero(cc == big)[0]; mem = ra[sel]; sub = G[sel][:, sel]
            d0 = dijkstra(sub, directed=False, indices=0); a = int(np.argmax(np.where(np.isfinite(d0), d0, -1))); da, pred = dijkstra(sub, directed=False, indices=a, return_predecessors=True)
            b = int(np.argmax(np.where(np.isfinite(da), da, -1))); path = [b]
            while path[-1] != a and pred[path[-1]] >= 0: path.append(int(pred[path[-1]]))
            dp = dijkstra(sub, directed=False, indices=path, min_only=True); far = int(np.argmax(np.where(np.isfinite(dp), dp, -1)))
            if np.isfinite(dp[far]) and dp[far] > F["branch_mm"]:
                for tip in (a, b, far): ybr.append(round(float(proj(X[mem[tip]])[1]), 3))
        boxes[k] = dict(area_mm2=round(A[k], 1), attach_s=round(float(s), 3), attach_dist_mm=round(float(dist), 1), orphan=bool(dist > F["orphan_mm"]), dir=u.tolist(), cands=cands, y_tips_s=ybr,
                        sigma_mm=(sigma or {}).get(k))
    flags = []
    for k, b in boxes.items():
        fa = b["area_mm2"] > F["big"] * med
        fb = any(abs(c1["s"] - c2["s"]) > F["ds_split"] and abs((c1["az"] - c2["az"] + 180) % 360 - 180) > F["side_deg"] for i, c1 in enumerate(b["cands"]) for c2 in b["cands"][i + 1:])
        fc = len(b["y_tips_s"]) == 3 and max(abs(b["y_tips_s"][i] - b["y_tips_s"][j]) for i in range(3) for j in range(i + 1, 3)) > F["ds_split"]
        if fa or fb or fc: flags.append(dict(box=k, kind="split", a=bool(fa), b=bool(fb), c=bool(fc), sigma_mm=b["sigma_mm"]))
        if b["orphan"]: flags.append(dict(box=k, kind="orphan", sigma_mm=b["sigma_mm"]))
        if b["area_mm2"] < F["small"] * med and not b["orphan"]:
            part = [j for j, o in boxes.items() if j != k and not o["orphan"] and abs(o["attach_s"] - b["attach_s"]) <= F["ds_merge"]
                    and np.degrees(np.arccos(min(1.0, abs(float(np.dot(b["dir"], o["dir"])))))) < F["dir_deg"]]
            flags.append(dict(box=k, kind=("merge" if len(part) == 1 else "orphan_merge_ambiguous"), into=(part[0] if len(part) == 1 else part), sigma_mm=b["sigma_mm"]))
    order = sorted(boxes, key=lambda k: boxes[k]["attach_s"]); inv = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order)) if order[i] > order[j]]
    inv_leaves = sorted(set(sum(map(list, inv), [])))
    # truth (scoring only)
    merged = {}; frag = {}
    for k in boxes:
        hl = human[own_lab == k]; hl = hl[hl > 0]; cnt = Counter(hl.tolist()); tot = max(len(hl), 1); merged[k] = sorted(h for h, c in cnt.items() if c / tot >= F["truth_share"])
    for h in sorted(set(human[human > 0].tolist())):
        bl = own_lab[human == h]; bl = bl[bl > 0]; cnt = Counter(bl.tolist()); tot = max(len(bl), 1); frag[h] = sorted(k for k, c in cnt.items() if c / tot >= F["truth_share"])
    true_merged = [k for k, v in merged.items() if len(v) >= 2]; fragmented = [h for h, v in frag.items() if len(v) >= 2]
    split_flagged = [f["box"] for f in flags if f["kind"] == "split"]; tp = [k for k in split_flagged if k in true_merged]; fp = [k for k in split_flagged if k not in true_merged]
    merge_flags = [f for f in flags if f["kind"] == "merge"]; mtp = 0
    for f in merge_flags:
        mk = Counter(human[own_lab == f["box"]][human[own_lab == f["box"]] > 0].tolist()).most_common(1); mo = Counter(human[own_lab == f["into"]][human[own_lab == f["into"]] > 0].tolist()).most_common(1)
        if mk and mo and mk[0][0] == mo[0][0]: mtp += 1
    return dict(set=name, median_area_mm2=round(med, 1), boxes=boxes, flags=flags, height_order_inversions=inv, height_order_leaves=inv_leaves,
                truth=dict(merged_boxes={str(k): merged[k] for k in true_merged}, fragmented_human_leaves={str(h): frag[h] for h in fragmented}),
                score=dict(split_flags=len(split_flagged), split_true=len(tp), split_false=len(fp), merged_boxes_caught=f"{len(tp)}/{len(true_merged)}", merge_flags=len(merge_flags), merge_true=mtp,
                           fragmented_human_leaves=len(fragmented)))


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    Z = np.load(R / "p1_pc_absorb_20261027_01/kaggle_data/r8pc_bundle.npz"); X = Z["xyz"]; ax = Z["axis_kind"].astype(int)
    LT = np.load(R / "engine_static_nine_20261028_01/kaggle_out/R3-1b/per_surfel_light.npz"); area = LT["area_b_silhouette_mm2"]
    H8 = np.load(R / "p1_pc_absorb_20261027_01/R8_pc/labels.npz")["label"].astype(int); HX = np.load(R / "p1_pc_absorb_20261027_01/R8_pc_rachis/labels_rachis.npz"); hrach = HX["target"] & (HX["label"] > 0)
    human = np.where(hrach, HX["label"], np.where(H8 > 0, H8, 0))
    poly = np.array(json.load(open(R / "p1_stem_gapfill_20261029_01/chain/R3-1b.json"))["chain"]["polyline_m"]); proj0 = polyline_param(poly)
    proj = lambda p: (lambda r: (r[0] * 1000.0, r[1], r[2]))(proj0(p))
    za = np.load(R / "p1_autoseed_20261029_01/R3-1b/labels_autoseed.npz"); auto = np.where(za["label"] > 0, za["label"], 0)
    res_auto = check("auto_seed_boxes", X, area, auto, ax == 2, human, proj)
    zl = np.load(R / "p1_leafbox_20261029_01/R3-1b/labels_leafbox.npz"); lb = np.where(zl["label"] > 0, zl["label"], 0); bj = json.load(open(R / "p1_leafbox_20261029_01/R3-1b/boxes.json"))
    sig = {lf["leaf_id"]: lf["petiole_base_sigma_mm"] for lf in bj["leaves"]}
    res_a = check("session_A_boxes", X, area, lb, ax == 2, human, proj, sig)
    # known blob = the auto box that holds the rachis of human traces 2, 4, 5, 6, 7 (autoseed results: each trace's top candidate); fixed 2026-10-29 — the first run identified it by the 15 % share rule, narrower than the contract
    tr_top = Counter(v["top_candidate"] for h, v in json.load(open(R / "p1_autoseed_20261029_01/R3-1b/results.json"))["reproduction"]["traces"].items() if int(h) in (2, 4, 5, 6, 7))
    blob = [str(k) for k, c in tr_top.items() if c >= 3]; blob_flagged = any(f["kind"] == "split" and str(f["box"]) in blob for f in res_auto["flags"])
    gate = "signal" if blob_flagged else "no_signal"
    out = dict(labels=["designated_not_adopted", "conditional_on_unvalidated_scale", "stem_role=display_only"], params=F, stem="stem_gapfill chain R3-1b polyline (stem_inferred_only, gap above 76.2 mm)",
               known_blob_boxes=blob, blob_flagged=blob_flagged, gate=gate, auto=res_auto, session_A=res_a, seconds=round(time.time() - t0))
    json.dump(out, open(OUT / "flags.json", "w"), indent=1, default=float)
    with open(OUT / "flags.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["set", "box", "kind", "a", "b", "c", "into", "sigma_mm"])
        for r_ in (res_auto, res_a):
            for f in r_["flags"]: wr.writerow([r_["set"], f["box"], f["kind"], f.get("a"), f.get("b"), f.get("c"), f.get("into"), f.get("sigma_mm")])
    print(json.dumps(dict(gate=gate, blob=blob, auto_score=res_auto["score"], auto_flags=res_auto["flags"], auto_inv=res_auto["height_order_leaves"], A_score=res_a["score"], A_flags=res_a["flags"], A_inv=res_a["height_order_leaves"],
                          auto_attach={k: (v["attach_s"], v["attach_dist_mm"], v["area_mm2"]) for k, v in res_auto["boxes"].items()}, A_attach={k: (v["attach_s"], v["attach_dist_mm"], v["area_mm2"]) for k, v in res_a["boxes"].items()},
                          truth_auto=res_auto["truth"], truth_A=res_a["truth"]), default=float, indent=0))


if __name__ == "__main__":
    main()
