"""P1 "base-only" seeds (contract build/p1_base_only_20261031_01/CONTRACT.md): one base point per compound leaf; the rachis blob is split by geodesic distance from the bases on the 3 mm
rachis-class graph; leaflets attach by the leaflet-direction rule (Steps L, B, A of build/p1_leaflet_dir_20261031_01/kaggle_kernel/leafdir.py, copied unchanged); growth = frozen rachis
absorption + R8_pc unchanged. R3-1b test with simulated clicks (exact + 5 perturbed runs). Runs as a Kaggle CPU kernel; human leaves are used only for the simulated clicks and the scoring.
"""

from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull, cKDTree

LABELS = ["base_click_seeds", "reference_only_scoring", "not_independent_validation", "conditional_on_unvalidated_scale", "after_checkpoint_v1"]
S = dict(min_views=2, same_share=0.70, adj_mm=3.0, normal_deg=30.0, oversize_x=2.0, end_q=0.15, width_ratio=1.15, min_leaflet=10, ray_mm=15.0, ray_step_mm=0.5, w_ray=5.0, w_ang=30.0, w_gap=10.0,
         delta=0.5, best_max=3.0, bins=4, per_side_bin=2, max_leaflets=9, term_s=0.85, term_dot=-0.5)                         # leafdir.py constants, unchanged
B = dict(graph_mm=3.0, snap_mm=4.0, max_geodesic_mm=60.0, tie_mm=3.0, perturb_mm=3.0, seeds=[20261031 + i for i in range(5)], min_rachis=3, iou_gate=0.5, leaves_gate=6, perturbed_gate=4)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name):
    for g in (f"/kaggle/input/**/{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle_data", name)):
        hit = glob.glob(g, recursive=True)
        if hit:
            return hit[0]
    raise SystemExit(f"{name} not found")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def axis_info(P):
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); u = vt[0]; t = (P - c) @ u; return c, u, t


def split_rachis(Xm, a2, bases):
    """Contract reading 2-3: snap bases to the nearest rachis-class surfel, per-base Dijkstra on the 3 mm graph, assignment rule. Returns (assign over a2 [0 = unassigned, k = base k], snap info)."""
    tree = cKDTree(Xm[a2]); E = tree.query_pairs(B["graph_mm"], output_type="ndarray"); w = np.linalg.norm(Xm[a2][E[:, 0]] - Xm[a2][E[:, 1]], axis=1); m = len(a2)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(m, m)).tocsr(); src, snap = [], []
    for k, b in enumerate(bases, 1):
        d, j = tree.query(b); src.append(int(j)); snap.append(dict(base=k, snap_mm=round(float(d), 2), snap_far=bool(d > B["snap_mm"]), surfel=int(a2[j])))
    D = np.stack([dijkstra(G, directed=False, indices=s) for s in src]); order = np.argsort(D, axis=0); best = D[order[0], np.arange(m)]
    second = D[order[1], np.arange(m)] if len(bases) > 1 else np.full(m, np.inf); assign = order[0] + 1
    bad = ~np.isfinite(best) | (best > B["max_geodesic_mm"]) | (np.isfinite(second) & (second - best < B["tie_mm"]))
    assign = np.where(bad, 0, assign); return assign, snap, best


def score(full, human, ax):
    hl = sorted(set(human[human > 0].tolist())); al = sorted(set(full[full > 0].tolist())); blade = ax == 0
    C = np.array([[int(((human == h) & (full == a) & blade).sum()) for a in al] for h in hl]) if al else np.zeros((len(hl), 0), int)
    per = {}; ri, ci = linear_sum_assignment(-C) if C.size else (np.array([], int), np.array([], int))
    for i, j in zip(ri, ci):
        h, a = hl[i], al[j]; uni = int((((human == h) | (full == a)) & blade).sum()); per[str(h)] = dict(auto_leaf=int(a), blade_iou=round(C[i, j] / uni if uni else 0.0, 3), human_blade=int(((human == h) & blade).sum()), auto_blade=int(((full == a) & blade).sum()))
    for h in hl:
        per.setdefault(str(h), dict(auto_leaf=None, blade_iou=0.0, human_blade=int(((human == h) & blade).sum()), auto_blade=0))
    mp = {int(al[j]): int(hl[i]) for i, j in zip(ri, ci)}; both = (human > 0) & (full > 0)
    agree = float(np.mean([mp.get(int(a), -1) == int(h) for a, h in zip(full[both], human[both])])) if both.any() else None
    return per, agree, mp


def main():
    out = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local_out"); os.makedirs(out, exist_ok=True); t0 = time.time()
    Z = dict(np.load(find("r8pc_bundle.npz"))); Xm = Z["xyz"] * 1000.0; N = Z["normal"]; ax = Z["axis_kind"].astype(int); sheet = Z["sheet_index"].astype(int); n = len(Xm); rows = Z["rows"]
    LT = np.load(find("per_surfel_light.npz")); area = LT["area_b_silhouette_mm2"]; light = LT["absorbed_ru_mm2"]; pri = json.load(open(find("leaflet_priors.json")))
    PF = np.load(find("patches_filled.npz")); posr = {int(r): i for i, r in enumerate(rows)}; pid = np.full(n, -1, int); pid[np.array([posr[int(r)] for r in PF["rows"]])] = PF["patch_filled"]
    V = np.load(find("votes_4b.npz")); H8 = np.load(find("labels.npz"))["label"].astype(int); HX = np.load(find("labels_rachis.npz")); hx = np.where(HX["target"] & (HX["label"] > 0), HX["label"], 0)
    human = np.where(hx > 0, hx, np.where(H8 > 0, H8, 0))
    poly = np.array(json.load(open(find("stem_chain_R3-1b.json")))["polyline_m"]) * 1000.0; stree = cKDTree(np.vstack([np.linspace(poly[i], poly[i + 1], 20) for i in range(len(poly) - 1)]))
    for p_ in ("tomato3d", "tomato3d.organs"):
        sys.modules.setdefault(p_, types.ModuleType(p_))
    lo_ = types.ModuleType("tomato3d.organs.light_organs"); lo_.ROOT = Path(out); sys.modules["tomato3d.organs.light_organs"] = lo_
    PA = load_module("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load_module("tomato3d.organs.pc_rachis", find("pc_rachis.py"))
    # ---- Step L (patches -> leaflets), identical to leafdir.py; independent of the rachis ----
    patches = sorted(set(pid[pid >= 0].tolist())); par = {p: p for p in patches}
    def root(p):
        while par[p] != p:
            par[p] = par[par[p]]; p = par[p]
        return p
    psheet = {p: Counter(sheet[pid == p].tolist()).most_common(1)[0][0] for p in patches}; bl = np.nonzero(pid >= 0)[0]; merged = 0
    for (a, b), sm, df in zip(V["pairs"], V["same"], V["different"]):
        a, b = int(a), int(b); nv = int(sm + df)
        if a not in par or b not in par or nv < S["min_views"] or sm / nv < S["same_share"] or psheet[a] != psheet[b]:
            continue
        ia = bl[pid[bl] == a]; ib = bl[pid[bl] == b]; tb = cKDTree(Xm[ib]); dd, jj = tb.query(Xm[ia], distance_upper_bound=S["adj_mm"]); ok = np.isfinite(dd)
        if not ok.any():
            continue
        ang = np.degrees(np.arccos(np.clip(np.abs((N[ia[ok]] * N[ib[jj[ok]]]).sum(1)), 0, 1)))
        if np.median(ang) <= S["normal_deg"]:
            par[root(a)] = root(b); merged += 1
    lid = np.full(n, -1, int); roots = sorted(set(root(p) for p in patches)); rid = {r_: i for i, r_ in enumerate(roots)}
    for p in patches:
        lid[pid == p] = rid[root(p)]
    nl = len(roots); larea = np.array([area[lid == l].sum() for l in range(nl)]); oversize = larea > S["oversize_x"] * pri["area_mm2"]["p95"]; log("step L leaflets", nl, "merges", merged)
    # ---- simulated base clicks (contract reading 4) ----
    a2 = np.nonzero(ax == 2)[0]; hl = sorted(set(human[human > 0].tolist())); tpts = Z["trace_pts"] * 1000.0; tleaf = Z["trace_leaf"].astype(int); bases0, derive = [], []
    for h in hl:
        m = tleaf == h
        if m.any():
            P_ = tpts[m]; d, _ = stree.query(P_); j = int(np.argmin(d)); bases0.append(P_[j]); derive.append(dict(human_leaf=int(h), source="trace point nearest to the stem polyline", dist_to_stem_mm=round(float(d[j]), 2), point_mm=[round(float(v), 2) for v in P_[j]]))
        else:
            idx = np.nonzero(human == h)[0]; d, _ = stree.query(Xm[idx]); j = int(np.argmin(d)); bases0.append(Xm[idx[j]]); derive.append(dict(human_leaf=int(h), source="no trace: human-leaf surfel nearest to the stem polyline", dist_to_stem_mm=round(float(d[j]), 2), point_mm=[round(float(v), 2) for v in Xm[idx[j]]]))
    json.dump(dict(labels=LABELS, bases=derive, perturb=dict(mm=B["perturb_mm"], seeds=B["seeds"])), open(os.path.join(out, "base_points_sim.json"), "w"), indent=1)
    ray_t = np.arange(0, S["ray_mm"] + 1e-9, S["ray_step_mm"])

    def run(bases, tag, keep_outputs=False):
        assign, snap, best = split_rachis(Xm, a2, bases); rach = {}; empty = []
        rsplit = np.zeros(n, int); rsplit[a2] = assign
        for k in range(1, len(bases) + 1):
            m = a2[assign == k]
            if len(m) < B["min_rachis"]:
                empty.append(k); continue
            c, u, t = axis_info(Xm[m]); e1, e2 = c + u * t.min(), c + u * t.max()
            if np.linalg.norm(e2 - bases[k - 1]) < np.linalg.norm(e1 - bases[k - 1]):
                u = -u; t = -t
            rach[k] = dict(idx=m, tree=cKDTree(Xm[m]), u=u, tmin=float(t.min()), tlen=float(np.ptp(t)) or 1.0, c=c)
        allr = np.nonzero(rsplit > 0)[0]; tall = cKDTree(Xm[allr]) if len(allr) else None; axes = []
        for l in range(nl):
            m = np.nonzero(lid == l)[0]
            if len(m) < S["min_leaflet"] or tall is None:
                axes.append(dict(leaflet=l, n=int(len(m)), flag="too_small")); continue
            c, u, t = axis_info(Xm[m]); _, _, vt = np.linalg.svd(Xm[m] - c, full_matrices=False); v = vt[1]; q = (Xm[m] - c) @ v
            lo, hi = np.quantile(t, S["end_q"]), np.quantile(t, 1 - S["end_q"]); E1 = t <= lo; E2 = t >= hi
            w1 = float(np.percentile(q[E1], 95) - np.percentile(q[E1], 5)) if E1.sum() >= 2 else 0.0; w2 = float(np.percentile(q[E2], 95) - np.percentile(q[E2], 5)) if E2.sum() >= 2 else 0.0
            p1, p2 = Xm[m][E1].mean(0), Xm[m][E2].mean(0); ratio = max(w1, w2) / max(min(w1, w2), 1e-6); flag = ""
            if ratio < S["width_ratio"]:
                base = p1 if tall.query(p1)[0] <= tall.query(p2)[0] else p2; flag = "base_by_proximity"
            else:
                base = p1 if w1 < w2 else p2
            bdir = (base - c) / max(np.linalg.norm(base - c), 1e-9)
            axes.append(dict(leaflet=l, n=int(len(m)), area_mm2=round(float(larea[l]), 1), cx=c[0], cy=c[1], cz=c[2], ux=u[0], uy=u[1], uz=u[2], bx=base[0], by=base[1], bz=base[2], width_ratio=round(ratio, 3),
                             flag=(flag + (";leaflet_oversize" if oversize[l] else "")).strip(";"), base_dir=bdir, base=base, c=c))
        rows_a = []
        for a_ in axes:
            if "base" not in a_ or not rach:
                rows_a.append(dict(leaflet=a_["leaflet"], status="too_small")); continue
            base, bdir = a_["base"], a_["base_dir"]; ray = base[None] + np.outer(ray_t, bdir); costs = {}
            for r, rr in rach.items():
                dray = float(rr["tree"].query(ray)[0].min()); gap, j = rr["tree"].query(base); pnear = Xm[rr["idx"][j]]; vv = pnear - base
                ang = float(np.degrees(np.arccos(np.clip((bdir @ vv) / max(np.linalg.norm(vv), 1e-9), -1, 1)))); costs[r] = dray / S["w_ray"] + ang / S["w_ang"] + float(gap) / S["w_gap"]
            o = sorted(costs, key=costs.get); b1 = o[0]; c1 = costs[b1]; c2 = costs[o[1]] if len(o) > 1 else np.inf
            st = "contested" if (c2 - c1 < S["delta"] or c1 > S["best_max"]) else "candidate"
            rr = rach[b1]; _, j = rr["tree"].query(a_["c"]); pn = Xm[rr["idx"][j]]; s = float(((pn - rr["c"]) @ rr["u"] - rr["tmin"]) / rr["tlen"]); s = min(max(s, 0.0), 1.0)
            side = int(np.sign(np.cross(rr["u"], a_["c"] - pn)[2]) or 1); term = bool(s >= S["term_s"] and float(a_["base_dir"] @ rr["u"]) < S["term_dot"])
            rows_a.append(dict(leaflet=a_["leaflet"], rachis=int(b1), cost=round(c1, 3), second_rachis=(int(o[1]) if len(o) > 1 else None), second_cost=(round(c2, 3) if np.isfinite(c2) else None), status=st, s=round(s, 3),
                               bin=min(S["bins"] - 1, int(s * S["bins"])), side=side, terminal=term))
        used = defaultdict(lambda: dict(total=0, term=0, bins=Counter()))
        for ra in sorted([r_ for r_ in rows_a if r_.get("status") == "candidate"], key=lambda r_: r_["cost"]):
            u_ = used[ra["rachis"]]; ok = u_["total"] < S["max_leaflets"]
            if ok and ra["terminal"]:
                ok = u_["term"] < 1
            elif ok:
                ok = u_["bins"][(ra["bin"], ra["side"])] < S["per_side_bin"]
            if ok:
                ra["status"] = "assigned"; u_["total"] += 1
                if ra["terminal"]:
                    u_["term"] += 1
                else:
                    u_["bins"][(ra["bin"], ra["side"])] += 1
            else:
                ra["status"] = "limit_blocked"
        seed = np.zeros(n, int)
        for k, rr in rach.items():
            seed[rr["idx"]] = k
        for ra in rows_a:
            if ra.get("status") == "assigned":
                seed[(lid == ra["leaflet"]) & (seed == 0)] = ra["rachis"]
        keep = sorted(set(seed[seed > 0].tolist())); remap = {k: i for i, k in enumerate(keep, 1)}; seed = np.array([remap.get(int(s_), 0) for s_ in seed]); final = [np.nonzero((ax == 2) & (seed == k))[0] for k in range(1, len(keep) + 1)]
        Z2 = dict(Z); Z2["stage1"] = seed.astype(np.int16); M = PA.Model(Z2); M.norm_unary(); labn = np.nonzero(M.st1n > 0)[0]
        d_, jj, v_ = PA.knn(M.tree_n, M.Xn[labn], 32, PA.P["band_mm"]); jj = np.where(v_, jj, 0); band = labn[(v_ & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1)]
        cb = PA.P["cal"]; PA.P["cal"] = tuple(sorted(set(M.st1n[M.st1n > 0].tolist())))[0::2]; auc = M.colour_auc(band) if len(band) else {k: dict(auc=0.5) for k in ("C1", "C2", "C4")}; PA.P["cal"] = cb
        wcol = {k: (PA.WC[k] if (k == "C3" or auc[k]["auc"] >= PA.P["auc_min"]) else 0.0) for k in PA.WC}; cr = (M.st1n[M.ei] > 0) & (M.st1n[M.ej] > 0) & (M.st1n[M.ei] != M.st1n[M.ej]); se = (M.st1n[M.ei] > 0) & (M.st1n[M.ei] == M.st1n[M.ej])
        xa, xb = M.edge_raw["C3"][cr], M.edge_raw["C3"][se]; auc3 = float(np.searchsorted(np.sort(xb), xa).mean() / len(xb)) if len(xa) and len(xb) else 0.5; wcol["C3"] = PA.WC["C3"] if auc3 >= PA.P["auc_min"] else 0.0
        g, c_ = PA.SETS["B"]; r8 = M.run(g, c_, wcol, M.st1n.copy()); lab = np.zeros(n, int); lab[M.nid] = r8["lab"]; node = np.zeros(n, bool); node[M.nid] = True
        tp, tl, ti = [], [], []
        for k, m in enumerate(final, 1):
            if len(m) < 2:
                continue
            P_ = Xm[m]; c0, u, t = axis_info(P_); bins = np.floor((t - t.min()) / 2.0).astype(int); pl = np.array([np.median(P_[bins == b_], 0) for b_ in np.unique(bins)]) / 1000.0
            tp.append(pl); tl.append(np.full(len(pl), k)); ti.append(np.full(len(pl), k))
        if tp:
            Z3 = dict(Z2); Z3["trace_pts"] = np.vstack(tp); Z3["trace_leaf"] = np.concatenate(tl); Z3["trace_id"] = np.concatenate(ti); rx = PR.Rx(Z3, dict(label=lab)); rr2 = rx.run(); rachl = np.zeros(n, int); rachl[rx.T] = rr2["lab"][:rx.nT]
        else:
            rachl = np.zeros(n, int)
        full = np.where((ax == 2) & (seed > 0), seed, np.where(rachl > 0, rachl, lab)); per, agree, mp = score(full, human, ax); nb = int(sum(1 for v in per.values() if v["blade_iou"] >= B["iou_gate"])); n3 = int(sum(1 for v in per.values() if v["blade_iou"] >= 0.3))
        blade_node = node & (ax == 0); cls = float(light[blade_node & (full <= 0)].sum() / max(light[blade_node].sum(), 1e-9))
        res = dict(tag=tag, rachis_candidates=len(rach), empty_bases=empty, snap=snap, leaflet_status=dict(Counter(r_["status"] for r_ in rows_a)), seeds=len(keep), per_human_leaf=per, leaves_iou_ge_0_5=nb, leaves_iou_ge_0_3=n3,
                   agreement_area_weighted=(round(agree, 4) if agree is not None else None), contested_light_share=round(cls, 4), unassigned_rachis=int((assign == 0).sum()), seconds=round(time.time() - t0))
        log(tag, {k: res[k] for k in ("rachis_candidates", "leaflet_status", "leaves_iou_ge_0_5", "leaves_iou_ge_0_3", "agreement_area_weighted", "contested_light_share")})
        extra = dict(rsplit=rsplit, axes=axes, rows_a=rows_a, rach=rach, full=full, seed=seed, mp=mp, remap=remap, node=node, lab=lab, rachl=rachl, bases=bases, best=best, assign=assign) if keep_outputs else None
        return res, extra
    R = dict(labels=LABELS, params=dict(leafdir=S, base_only=B), step_L=dict(patches=len(patches), leaflets=nl, merges=merged, oversize=int(oversize.sum())))
    res0, X0 = run([np.array(b) for b in bases0], "exact", keep_outputs=True); R["exact"] = res0; R["perturbed"] = []
    for i, sd in enumerate(B["seeds"]):
        rng = np.random.default_rng(sd); pb = [b + rng.uniform(-B["perturb_mm"], B["perturb_mm"], 3) for b in bases0]; r_, _ = run(pb, f"perturbed_{i}_seed{sd}"); r_["perturbation_seed"] = sd; R["perturbed"].append(r_)
    pass_exact = res0["leaves_iou_ge_0_5"] >= B["leaves_gate"]; pass_pert = sum(1 for r_ in R["perturbed"] if r_["leaves_iou_ge_0_5"] >= B["leaves_gate"])
    R["gate"] = dict(exact_leaves_iou_ge_0_5=res0["leaves_iou_ge_0_5"], exact_pass=bool(pass_exact), perturbed_leaves=[r_["leaves_iou_ge_0_5"] for r_ in R["perturbed"]], perturbed_pass_count=pass_pert,
                     required=f">= {B['leaves_gate']} of 8 exact and >= {B['perturbed_gate']} of 5 perturbed", base_only="pass" if (pass_exact and pass_pert >= B["perturbed_gate"]) else "fail")
    R["perturbed_spread"] = dict(leaves_ge_0_5=[r_["leaves_iou_ge_0_5"] for r_ in R["perturbed"]], leaves_ge_0_3=[r_["leaves_iou_ge_0_3"] for r_ in R["perturbed"]], agreement=[r_["agreement_area_weighted"] for r_ in R["perturbed"]],
                                 per_leaf_iou_range={h: [round(min([R["exact"]["per_human_leaf"][h]["blade_iou"]] + [r_["per_human_leaf"][h]["blade_iou"] for r_ in R["perturbed"]]), 3),
                                                       round(max([R["exact"]["per_human_leaf"][h]["blade_iou"]] + [r_["per_human_leaf"][h]["blade_iou"] for r_ in R["perturbed"]]), 3)] for h in R["exact"]["per_human_leaf"]})
    # ---- diagnostics on the exact run ----
    rows_a, rach, axes, mp = X0["rows_a"], X0["rach"], X0["axes"], X0["mp"]; amap = {r_["leaflet"]: r_ for r_ in rows_a}; diag = dict(pure=0, assigned_pure=0, correct=0, prox_pure=0, prox_correct=0, wrong=Counter()); human_rach = {h: np.nonzero((human == h) & (ax == 2))[0] for h in hl}
    htree = {h: cKDTree(Xm[v]) for h, v in human_rach.items() if len(v)}
    for a_ in axes:
        if "base" not in a_:
            continue
        l = a_["leaflet"]; m = np.nonzero((lid == l) & (ax == 0))[0]; hh = Counter(human[m].tolist()); top, k_ = hh.most_common(1)[0]
        if top > 0 and k_ / len(m) >= 0.8:
            diag["pure"] += 1; ra = amap.get(l, {})
            if ra.get("status") == "assigned":
                diag["assigned_pure"] += 1; ok = (hl[ra["rachis"] - 1] == top) if ra["rachis"] <= len(hl) else False; prox = "base_by_proximity" in a_.get("flag", ""); diag["correct"] += int(ok); diag["prox_pure"] += int(prox); diag["prox_correct"] += int(prox and ok)
                if not ok and top in htree:
                    c_ = a_["c"]; end2 = 2 * c_ - a_["base"]; diag["wrong"]["step_B (base end farther from its human rachis than the other end)" if htree[top].query(a_["base"])[0] > htree[top].query(end2)[0] else "step_A"] += 1
                elif not ok:
                    diag["wrong"]["step_A"] += 1
        else:
            ra = amap.get(l, {})
            if ra.get("status") == "assigned":
                diag["wrong"]["step_L (impure leaflet)"] += 1
    diag["wrong"] = dict(diag["wrong"]); R["diagnostics"] = dict(leaflets=nl, pure_leaflets=diag["pure"], pure_share=round(diag["pure"] / max(nl, 1), 3), assigned_pure=diag["assigned_pure"], base_direction_correct=diag["correct"],
                                                              base_direction_correct_share=round(diag["correct"] / max(diag["assigned_pure"], 1), 3), base_by_proximity_pure=diag["prox_pure"], base_by_proximity_correct=diag["prox_correct"], error_sources=diag["wrong"])
    # ---- outputs ----
    np.savez_compressed(os.path.join(out, "rachis_split.npz"), rows=rows, base_of_rachis=X0["rsplit"].astype(np.int16), best_geodesic_mm=np.pad(X0["best"], (0, 0)).astype(np.float32), bases_mm=np.array(X0["bases"]))
    np.savez_compressed(os.path.join(out, "labels_base.npz"), rows=rows, label=X0["full"].astype(np.int16), r8pc=X0["lab"].astype(np.int16), rachis_abs=X0["rachl"].astype(np.int16), node=X0["node"], seed=X0["seed"].astype(np.int16), human_leaf_of_label=json.dumps({str(k): int(v) for k, v in mp.items()}))
    with open(os.path.join(out, "assignment.csv"), "w", newline="") as fh:
        k_ = ["leaflet", "rachis", "cost", "second_rachis", "second_cost", "status", "s", "bin", "side", "terminal"]; wr = csv.DictWriter(fh, fieldnames=k_, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows_a)
    R["seconds"] = round(time.time() - t0); json.dump(R, open(os.path.join(out, "results_kernel.json"), "w"), indent=1, default=float)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10"); fig, axp = plt.subplots(figsize=(10, 10)); bi = np.nonzero(ax == 0)[0][::4]; axp.scatter(Xm[bi, 0], Xm[bi, 1], s=0.3, c="#d0d0d0")
    for k, rr in rach.items():
        axp.scatter(Xm[rr["idx"], 0], Xm[rr["idx"], 1], s=1.2, color=cmap((k - 1) % 10))
    for a_ in axes:
        if "base" not in a_:
            continue
        r_ = amap.get(a_["leaflet"], {}); col = cmap((r_["rachis"] - 1) % 10) if r_.get("status") == "assigned" else (0.5, 0.5, 0.5, 0.5)
        axp.annotate("", xy=(a_["base"][0], a_["base"][1]), xytext=(a_["c"][0], a_["c"][1]), arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    for h in hl:
        idx = np.nonzero((human == h) & (ax == 0))[0]
        if len(idx) > 10:
            hull = ConvexHull(Xm[idx][:, :2]); pts = Xm[idx][hull.vertices][:, :2]; axp.plot(*np.vstack([pts, pts[:1]]).T, color="k", lw=0.7); axp.text(*pts.mean(0), str(h), fontsize=11, ha="center")
    for k, b in enumerate(X0["bases"], 1):
        axp.scatter([b[0]], [b[1]], marker="*", s=220, color=cmap((k - 1) % 10), edgecolor="k", zorder=5)
    axp.set_aspect("equal"); axp.set_title("base-only (exact clicks): bases = stars, rachis split coloured, leaflet arrows, human leaves outlined"); fig.savefig(os.path.join(out, "top_view.png"), dpi=130); plt.close(fig)
    log("done", R["gate"], f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
