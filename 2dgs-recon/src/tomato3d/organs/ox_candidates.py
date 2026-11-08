"""Compound-leaf candidates for an O / X review (contract build/p1_ox_candidates_20261029_01/CONTRACT.md). Runs as a Kaggle CPU kernel.

Per individual: several overlapping candidates (rachis component + attached leaflet groups) from axis-class surfels at three linking radii, merge / split rules, dedup, plausibility filter,
cap 40 with alternates grouped in slots. On R3-1b the 8 human leaves score the pool (recall) and a simulated perfect review gives an upper bound. No human label is a candidate, seed or filter.
usage (Kaggle): python ox_candidates.py   (files found under /kaggle/input: r8pc_bundle.npz, patches_filled.npz, per_surfel_light.npz, labels.npz, labels_rachis.npz, pc_absorb.py, pc_rachis.py)
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

P = dict(radii=(2.0, 3.0, 4.0), min_comp=30, split_len_mult=1.6, branch_mm=6.0, branch_min_n=10, junction_excl_mm=3.0, merge_end_mm=6.0, merge_deg=25.0, attach_mm=5.0, attach_min_contacts=3,
         dedup_iou=0.8, area_mult=3.0, rachis_len_mult=1.5, min_groups=3, cap=40, alt_iou=0.5, iou_min=0.5, gate_recall=6)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name):
    for g in (f"/kaggle/input/**/{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", name)):
        hit = glob.glob(g, recursive=True)
        if hit:
            return hit[0]
    raise SystemExit(f"{name} not found")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def components(X, idx, r):
    E = cKDTree(X[idx]).query_pairs(r, output_type="ndarray"); m = len(idx)
    G = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(m, m)).tocsr() if len(E) else coo_matrix((m, m)).tocsr()
    return connected_components(G, directed=False)[1]


def principal(Xc):
    c = Xc.mean(0)
    if len(Xc) < 3:
        return c, np.array([0.0, 0.0, 1.0]), np.zeros(len(Xc)), 0.0
    _, _, vt = np.linalg.svd(Xc - c, full_matrices=False); u = vt[0]; t = (Xc - c) @ u
    return c, u, t, float(np.linalg.norm(Xc[t.argmax()] - Xc[t.argmin()]))


def ends(Xc, u, t):
    return Xc[t.argmin()], Xc[t.argmax()]


def longest_path(Xc, r):
    """Longest geodesic path (double Dijkstra) on the graph with edges <= r; returns node indices along the path."""
    m = len(Xc)
    if m < 3:
        return np.arange(m)
    E = cKDTree(Xc).query_pairs(r, output_type="ndarray"); w = np.linalg.norm(Xc[E[:, 0]] - Xc[E[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(m, m)).tocsr()
    d0 = dijkstra(G, indices=0, directed=False); d0[~np.isfinite(d0)] = -1; a = int(d0.argmax())
    da, pred = dijkstra(G, indices=a, directed=False, return_predecessors=True); da[~np.isfinite(da)] = -1; b = int(da.argmax()); path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return np.array(path)


def merge_components(X, comps, r):
    """Merge rule (contract 1.3): nearest end points < 6 mm apart AND principal directions differ < 25 deg (acute angle)."""
    info = []
    for mem in comps:
        c, u, t, L = principal(X[mem]); e0, e1 = ends(X[mem], u, t); info.append((u, e0, e1))
    parent = list(range(len(comps)))
    def find_(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    cmin = np.cos(np.radians(P["merge_deg"]))
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            d = min(np.linalg.norm(a - b) for a in info[i][1:] for b in info[j][1:])
            if d < P["merge_end_mm"] and abs(float(info[i][0] @ info[j][0])) >= cmin:
                parent[find_(j)] = find_(i)
    groups = {}
    for i, mem in enumerate(comps):
        groups.setdefault(find_(i), []).append(mem)
    return [np.concatenate(v) for v in groups.values()]


def pool_candidates(X, ax2, r):
    """Candidates of one linking radius: components >= min_comp, merged by the merge rule, plus Y-junction trunk / branch candidates of the long ones."""
    cc = components(X, ax2, r); cnt = Counter(cc.tolist()); comps = [ax2[cc == c] for c, k in cnt.items() if k >= P["min_comp"]]
    comps = [c for c in merge_components(X, comps, r) if len(c) >= P["min_comp"]]
    lens = [principal(X[m])[3] for m in comps]; med = float(np.median(lens)) if lens else 0.0; out = []
    for mem, L in zip(comps, lens):
        out.append(dict(rachis=mem, kind="component", radius=r))
        if med and L > P["split_len_mult"] * med:                                   # long component: look for a Y-junction
            has_branch = False; Xc = X[mem]; path = longest_path(Xc, r); tree = cKDTree(Xc[path]); dd = tree.query(Xc)[0]; near = dd <= P["junction_excl_mm"]
            rest = np.nonzero(~near)[0]
            if len(rest):
                sub = components(Xc, rest, r)
                for c in set(sub.tolist()):
                    b = rest[sub == c]; ext = principal(Xc[b])[3]
                    if len(b) >= P["branch_min_n"] and ext > P["branch_mm"]:
                        out.append(dict(rachis=mem[b], kind="branch", radius=r)); has_branch = True
            if len(rest) and has_branch:
                out.append(dict(rachis=mem[near], kind="trunk", radius=r))
    return out


def build(X, axis_idx, blade_idx, group, area, stem_idx=None):
    """All candidates of one individual. X: (n, 3) mm; axis_idx / blade_idx: surfel indices; group: (n,) leaflet-group id per surfel (-1 = none); area: (n,) mm^2 per surfel."""
    raw = []
    for r in P["radii"]:
        raw += pool_candidates(X, axis_idx, r)
    log("raw candidates", len(raw), dict(Counter((c["radius"], c["kind"]) for c in raw)))
    bt = cKDTree(X[blade_idx]); items = []
    for c in raw:                                                                      # attach leaflets
        near = bt.query_ball_point(X[c["rachis"]], P["attach_mm"]); contacts = Counter()
        for k in near:
            for b in k:
                g = int(group[blade_idx[b]])
                if g >= 0:
                    contacts[g] += 1
        gs = sorted(g for g, k in contacts.items() if k >= P["attach_min_contacts"]); gm = np.isin(group, gs) if gs else np.zeros(len(X), bool)
        full = np.union1d(c["rachis"], np.nonzero(gm)[0]); _, u, t, L = principal(X[c["rachis"]])
        items.append(dict(c, groups=gs, surfels=full, rachis_len=L, area=float(area[full].sum()), set=set(full.tolist())))
    # dedup: smaller radius kept first
    items.sort(key=lambda d: (d["radius"], -d["area"])); kept = []; dropped = Counter()
    for d in items:
        if any(len(d["set"] & k["set"]) / max(len(d["set"] | k["set"]), 1) >= P["dedup_iou"] for k in kept):
            dropped["dedup"] += 1; continue
        kept.append(d)
    areas = np.array([d["area"] for d in kept]); med = float(np.median(areas)); lens = np.array([d["rachis_len"] for d in kept]); p95 = float(np.percentile(lens, 95)); keep2 = []; dropped_list = []
    for d in kept:
        why = None
        if d["area"] > P["area_mult"] * med:
            why = "area>3x median"
        elif d["rachis_len"] > P["rachis_len_mult"] * p95:
            why = "rachis length > 1.5 x P95"
        elif len(d["groups"]) < P["min_groups"]:
            why = "< 3 leaflet groups"
        if why:
            dropped[why] += 1; dropped_list.append(dict(radius=d["radius"], kind=d["kind"], rachis=int(len(d["rachis"])), area=round(d["area"], 1), reason=why))
        else:
            keep2.append(d)
    # slots + cap
    keep2.sort(key=lambda d: -d["area"]); prim = []; alts = []
    for d in keep2:
        ov = [(len(d["set"] & p["set"]) / max(len(d["set"] | p["set"]), 1), i) for i, p in enumerate(prim)]; best = max(ov) if ov else (0.0, -1)
        if best[0] > P["alt_iou"]:
            alts.append((best[1], d))
        else:
            prim.append(d)
    shown = prim[:P["cap"]]; room = P["cap"] - len(shown); cand = []
    for i, d in enumerate(shown):
        cand.append(dict(d, slot=i, alt=0))
    cnt_alt = Counter()
    for pi_, d in sorted(alts, key=lambda a: -a[1]["area"]):
        if room <= 0 or pi_ >= len(shown):
            dropped["cap"] += 1; continue
        cnt_alt[pi_] += 1; cand.append(dict(d, slot=pi_, alt=cnt_alt[pi_])); room -= 1
    stats = dict(raw=len(raw), after_dedup=len(kept), after_filter=len(keep2), slots=len(shown), shown=len(cand), dropped=dict(dropped), median_area_mm2=round(med, 1), rachis_len_p95_mm=round(p95, 1), dropped_list=dropped_list)
    return cand, stats


def iou(a, b):
    return len(a & b) / max(len(a | b), 1)


def score(cand, human, n):
    """Pool recall: human leaves with >= 1 shown candidate at IoU >= 0.5; candidates per human leaf; best candidate per human leaf (scoring only)."""
    res = {}; hl = sorted(set(human[human > 0].tolist()))
    for h in hl:
        S = set(np.nonzero(human == h)[0].tolist()); ious = [iou(c["set"], S) for c in cand]; best = int(np.argmax(ious)) if ious else -1
        res[str(h)] = dict(human_surfels=len(S), best_iou=round(max(ious), 3) if ious else 0.0, best_candidate=best, n_ge_0_5=int(sum(i >= P["iou_min"] for i in ious)), n_ge_0_3=int(sum(i >= 0.3 for i in ious)))
    rec = sum(1 for v in res.values() if v["best_iou"] >= P["iou_min"]); return res, rec, len(hl)


def run_chain(Z, st1, polylines, PA, PR, ax):
    """auto_seed.py steps 4-5 unchanged: R8_pc (set B) from the stage-1 labels, then rachis absorption with the seeds' polylines as traces."""
    n = len(Z["xyz"]); Z2 = dict(Z); Z2["stage1"] = st1.astype(np.int16); M = PA.Model(Z2); M.norm_unary()
    labn = np.nonzero(M.st1n > 0)[0]; d_, jj, v_ = PA.knn(M.tree_n, M.Xn[labn], 32, PA.P["band_mm"]); jj = np.where(v_, jj, 0)
    band = labn[(v_ & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1)]; cal_backup = PA.P["cal"]; PA.P["cal"] = tuple(sorted(set(M.st1n[M.st1n > 0].tolist())))[0::2]
    auc = M.colour_auc(band) if len(band) else {k: dict(auc=0.5) for k in ("C1", "C2", "C4")}; wcol = {k: (PA.WC[k] if (k == "C3" or auc[k]["auc"] >= PA.P["auc_min"]) else 0.0) for k in PA.WC}
    cross = (M.st1n[M.ei] > 0) & (M.st1n[M.ej] > 0) & (M.st1n[M.ei] != M.st1n[M.ej]); samee = (M.st1n[M.ei] > 0) & (M.st1n[M.ei] == M.st1n[M.ej]); xa, xb = M.edge_raw["C3"][cross], M.edge_raw["C3"][samee]
    auc3 = float(np.searchsorted(np.sort(xb), xa).mean() / len(xb)) if len(xa) and len(xb) else 0.5; wcol["C3"] = PA.WC["C3"] if auc3 >= PA.P["auc_min"] else 0.0; PA.P["cal"] = cal_backup
    g, c_ = PA.SETS["B"]; r8 = M.run(g, c_, wcol, M.st1n.copy()); lab8 = np.zeros(n, int); lab8[M.nid] = r8["lab"]
    tp, tl, ti = [], [], []
    for k, poly in polylines.items():
        q = np.asarray(poly) / 1000.0; tp.append(q); tl.append(np.full(len(q), k)); ti.append(np.full(len(q), k))
    Z3 = dict(Z2); Z3["trace_pts"] = np.vstack(tp); Z3["trace_leaf"] = np.concatenate(tl); Z3["trace_id"] = np.concatenate(ti)
    rx = PR.Rx(Z3, dict(label=lab8)); rr = rx.run(); labx = lab8.copy(); labx[rx.T] = rr["lab"][:rx.nT]
    return lab8, labx, dict(colour_weights=wcol, rounds=r8["counts"], rachis_targets=int(rx.nT), rachis_absorbed=int((rr["lab"][:rx.nT] > 0).sum()))


def polyline(Xc, bin_mm=2.0):
    c, u, t, L = principal(Xc); bins = np.floor((t - t.min()) / bin_mm).astype(int); occ = np.unique(bins); return np.array([np.median(Xc[bins == b], 0) for b in occ])


def main():
    out = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ox_out"); os.makedirs(out, exist_ok=True); t0 = time.time()
    for p_ in ("tomato3d", "tomato3d.organs"):
        sys.modules.setdefault(p_, types.ModuleType(p_))
    lo = types.ModuleType("tomato3d.organs.light_organs"); lo.ROOT = Path(out); sys.modules["tomato3d.organs.light_organs"] = lo
    PA = load_module("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load_module("tomato3d.organs.pc_rachis", find("pc_rachis.py"))
    Z = dict(np.load(find("r8pc_bundle.npz"))); X = Z["xyz"] * 1000.0; ax = Z["axis_kind"].astype(int); n = len(X); rows = Z["rows"]
    pf = np.load(find("patches_filled.npz")); LT = np.load(find("per_surfel_light.npz")); area = LT["area_b_silhouette_mm2"]
    group = np.full(n, -1, int); pos = {int(r): i for i, r in enumerate(rows)}
    for r_, g in zip(pf["rows"], pf["patch_filled"]):
        i = pos.get(int(r_))
        if i is not None and g >= 0:
            group[i] = int(g)
    H8 = np.load(find("labels.npz"))["label"].astype(int); HX = np.load(find("labels_rachis.npz")); hx = np.where(HX["target"] & (HX["label"] > 0), HX["label"], 0); human = np.where(hx > 0, hx, np.where(H8 > 0, H8, 0))
    axis_idx = np.nonzero(ax == 2)[0]; blade_idx = np.nonzero(ax == 0)[0]; log("surfels", n, "blade", len(blade_idx), "rachis-class", len(axis_idx), "stem", int((ax == 1).sum()), "groups", int(group.max() + 1))
    cand, stats = build(X, axis_idx, blade_idx, group, area); log("candidates", {k: v for k, v in stats.items() if k != "dropped_list"})
    per_h, rec, nh = score(cand, human, n); gate = bool(rec >= P["gate_recall"])
    R = dict(params=P, labels=["auto_candidates_user_verified", "not_independent_validation", "conditional_on_unvalidated_scale"], stats={k: v for k, v in stats.items() if k != "dropped_list"}, dropped_list=stats["dropped_list"],
             prechecks=dict(individual="R3-1b", human_leaves=nh, pool_recall=rec, gate_min=P["gate_recall"], gate_pass=gate, per_human_leaf=per_h, ox_route="pass" if gate else "failed_gate"))
    # pool without the cap, for the report only (how much the cap costs)
    log("pool recall", rec, "of", nh, "gate", gate)
    if True:                                                                          # simulated perfect review (upper bound), run whether or not the gate passes (reported, not used)
        best = {h: v["best_candidate"] for h, v in per_h.items() if v["best_iou"] >= P["iou_min"]}; chosen = []
        for h in sorted(best, key=lambda h: -per_h[h]["best_iou"]):
            c = best[h]
            if all(iou(cand[c]["set"], cand[o]["set"]) <= P["alt_iou"] for o in chosen):
                chosen.append(c)
        order = sorted(chosen, key=lambda c: float(X[cand[c]["rachis"]][:, 2].min())); st1 = np.zeros(n, int); polys = {}
        for k, c in enumerate(order, 1):
            sel = cand[c]["surfels"]; m = sel[ax[sel] == 0]; st1[m] = np.where(st1[m] == 0, k, st1[m]); polys[k] = polyline(X[cand[c]["rachis"]])
        covered = np.zeros(n, bool)
        for c in order:
            covered[cand[c]["surfels"]] = True
        blade = ax == 0; cov = float(area[blade & covered].sum() / area[blade].sum()) if area[blade].sum() else 0.0
        sim = dict(o_candidates=len(order), coverage_blade_area=round(cov, 4))
        if len(order) >= 2:
            lab8, labx, ch = run_chain(Z, st1, polys, PA, PR, ax); full = np.where(labx > 0, labx, 0); hl = sorted(set(human[human > 0].tolist())); al = sorted(set(full[full > 0].tolist()))
            C = np.array([[int(((human == h) & (full == a_)).sum()) for a_ in al] for h in hl]); ri, ci = linear_sum_assignment(-C); mp = {int(al[j]): int(hl[i]) for i, j in zip(ri, ci)}; both = (human > 0) & (full > 0)
            agree = float(np.mean([mp.get(int(a_), -1) == int(h) for a_, h in zip(full[both], human[both])])) if both.any() else None
            rep = {}
            for i, j in zip(ri, ci):
                inter = C[i, j]; uni = int(((human == hl[i]) | (full == al[j])).sum()); rep[str(hl[i])] = round(inter / uni if uni else 0.0, 3)
            sim.update(chain=ch, agreement_after_matching=(round(agree, 4) if agree is not None else None), labelled_in_both=int(both.sum()), iou_per_human_leaf=rep, leaves_iou_ge_0_5=int(sum(v >= 0.5 for v in rep.values())))
        R["simulated_review_upper_bound"] = sim; log("simulated review", sim)
    # outputs: candidates.json (+ npz with the surfel indices)
    off = np.cumsum([0] + [len(c["surfels"]) for c in cand]); roff = np.cumsum([0] + [len(c["rachis"]) for c in cand])
    np.savez_compressed(os.path.join(out, "candidates.npz"), surfels=np.concatenate([c["surfels"] for c in cand]), offsets=off, rachis=np.concatenate([c["rachis"] for c in cand]), rachis_offsets=roff, rows=rows)
    meta = []
    for i, c in enumerate(cand):
        Xr = X[c["rachis"]]; poly = polyline(Xr); base = poly[int(np.argmin(poly[:, 2]))]
        meta.append(dict(id=i, slot=c["slot"], alt=c["alt"], radius=c["radius"], kind=c["kind"], rachis_surfels=int(len(c["rachis"])), groups=len(c["groups"]), surfels=int(len(c["surfels"])), area_mm2=round(c["area"], 1), rachis_len_mm=round(c["rachis_len"], 1),
                         base_mm=[round(float(v), 2) for v in base], polyline_mm=[[round(float(v), 2) for v in p] for p in poly]))
    json.dump(meta, open(os.path.join(out, "candidates.json"), "w")); R["seconds"] = round(time.time() - t0); json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1, default=float); log("done", R["seconds"])


if __name__ == "__main__":
    main()
