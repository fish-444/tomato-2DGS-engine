"""L-system structure rule test on R3-1b: phyllotactic helix (node azimuth + height) fitted to the base-near rachis surfels; node groups seed the frozen R8_pc and
rachis absorption; scored against the human-seeded leaves only at the end. Kaggle CPU kernel; imports pc_absorb.py / pc_rachis.py from the dataset.
Contract: build/p1_lsys_split_20261030_01/CONTRACT.md.
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
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

H = dict(Ks=(7, 8, 9), sig_phi=20.0, sig_h=8.0, d_range=(120, 150), phi_step=5, d_step=1, local_mm=6.0, base_pct=30, alt=5, amb=1.0, graph_mm=3.0, stage1_mm=3.0, iou=0.5, gate_n=6, blob=(2, 4, 5, 6, 7), blob_share=0.10, blob_min=4)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name):
    for g in (f"/kaggle/input/**/{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", name)):
        hit = glob.glob(g, recursive=True)
        if hit: return hit[0]
    raise SystemExit(f"{name} not found")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def pava(y, w):
    """Weighted isotonic (non-decreasing) regression."""
    v = list(y.astype(float)); ww = list(w.astype(float)); n_ = [1] * len(v); i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            nw = ww[i] + ww[i + 1]; nv = (v[i] * ww[i] + v[i + 1] * ww[i + 1]) / nw if nw > 0 else (v[i] + v[i + 1]) / 2
            v[i] = nv; ww[i] = nw; n_[i] += n_[i + 1]; del v[i + 1], ww[i + 1], n_[i + 1]; i = max(i - 1, 0)
        else: i += 1
    return np.repeat(v, n_)


def cdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def fit_helix(phi, h, K):
    n = len(phi); best = None; q0 = np.quantile(h, (np.arange(K) + 0.5) / K)
    for d in np.arange(H["d_range"][0], H["d_range"][1] + 1e-9, H["d_step"]):
        for p0 in np.arange(0, 360, H["phi_step"]):
            pk = (p0 + np.arange(K) * d) % 360.0; cphi = (cdiff(phi[:, None], pk[None]) / H["sig_phi"]) ** 2; hk = q0.copy()
            for _ in range(H["alt"]):
                C = cphi + ((h[:, None] - hk[None]) / H["sig_h"]) ** 2; a = C.argmin(1); cnt = np.bincount(a, minlength=K); s = np.bincount(a, weights=h, minlength=K)
                mean = np.where(cnt > 0, s / np.maximum(cnt, 1), hk); hk = pava(mean, np.maximum(cnt, 1e-6))
            C = cphi + ((h[:, None] - hk[None]) / H["sig_h"]) ** 2; cost = float(C.min(1).sum())
            if best is None or cost < best["cost"]: best = dict(K=K, phi0=float(p0), d=float(d), hk=hk.copy(), cost=cost)
    best["bic"] = best["cost"] + 2 * (K + 2) * np.log(n); return best


def main():
    out = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "lsys_out"); os.makedirs(out, exist_ok=True); t0 = time.time()
    Z = dict(np.load(find("r8pc_bundle.npz"))); X = Z["xyz"]; ax = Z["axis_kind"].astype(int); n = len(X); rows = Z["rows"]
    ch = json.load(open(find("stem_chain_R3-1b.json"), encoding="utf-8")); poly = np.array(ch["polyline_m"]); zt = float(ch["rim"]["z"])
    c = poly.mean(0); _, _, vt = np.linalg.svd(poly - c, full_matrices=False); dax = vt[0] if vt[0][2] > 0 else -vt[0]
    p0 = c + dax * ((zt - c[2]) / dax[2]); e1 = np.array([1.0, 0, 0]) - dax[0] * dax; e1 /= np.linalg.norm(e1); e2 = np.cross(dax, e1)
    R = dict(labels=["model_constrained", "reference_only_scoring", "not_independent_validation", "conditional_on_unvalidated_scale"], params=H, axis=dict(used="least-squares line through the stem display polyline (17 points)", point_on_pot_top=p0.tolist(), direction=dax.tolist(), tilt_deg=round(float(np.degrees(np.arccos(dax[2]))), 2), pot_top_z_m=zt))
    rc = np.nonzero(ax == 2)[0]; V = X[rc] - p0; hh = (V @ dax) * 1000; perp = V - np.outer(V @ dax, dax); rr = np.linalg.norm(perp, axis=1) * 1000; ph = np.degrees(np.arctan2(perp @ e2, perp @ e1)) % 360
    T = cKDTree(X[rc]); nb = T.query_ball_point(X[rc], H["local_mm"] / 1000); base = np.array([rr[i] < np.percentile(rr[idx], H["base_pct"]) if len(idx) >= 3 else False for i, idx in enumerate(nb)])
    bi = np.nonzero(base)[0]; log(f"rachis {len(rc)}, base-near {len(bi)} ({time.time() - t0:.0f}s)")
    fits = {}
    for K in H["Ks"]:
        fits[K] = fit_helix(ph[bi], hh[bi], K); log("K", K, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in fits[K].items() if k != "hk"}, f"({time.time() - t0:.0f}s)")
    Kc = min(fits, key=lambda k: fits[k]["bic"]); F = fits[Kc]; pk = (F["phi0"] + np.arange(Kc) * F["d"]) % 360.0; hk = F["hk"]
    json.dump(dict(chosen_K=Kc, fits={str(k): dict(phi0=v["phi0"], d=v["d"], heights_mm=[round(float(x), 2) for x in v["hk"]], cost=round(v["cost"], 2), bic=round(v["bic"], 2)) for k, v in fits.items()}, n_base=int(len(bi))),
              open(os.path.join(out, "helix_fit.json"), "w"), indent=1)
    # assignment
    Cb = (cdiff(ph[bi][:, None], pk[None]) / H["sig_phi"]) ** 2 + ((hh[bi][:, None] - hk[None]) / H["sig_h"]) ** 2; o = np.sort(Cb, 1); nodeb = Cb.argmin(1) + 1; ambb = (o[:, 1] - o[:, 0]) < H["amb"]
    E = T.query_pairs(H["graph_mm"] / 1000, output_type="ndarray"); L = np.linalg.norm(X[rc][E[:, 0]] - X[rc][E[:, 1]], axis=1) + 1e-9; m = len(rc)
    G = coo_matrix((np.r_[L, L], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(m, m)).tocsr(); src_nodes = bi
    dist, pred, srcs = dijkstra(G, directed=False, indices=src_nodes, min_only=True, return_predecessors=True)
    pos_b = -np.ones(m, int); pos_b[bi] = np.arange(len(bi)); node_r = np.zeros(m, int); amb_r = np.zeros(m, bool); reach = np.isfinite(dist)
    node_r[reach] = nodeb[pos_b[srcs[reach]]]; amb_r[reach] = ambb[pos_b[srcs[reach]]]
    node = np.zeros(n, int); node[rc] = np.where(amb_r, 0, node_r); amb = np.zeros(n, bool); amb[rc] = amb_r; unassigned = np.zeros(n, bool); unassigned[rc] = ~reach
    np.savez_compressed(os.path.join(out, "node_groups.npz"), rows=rows, node=node.astype(np.int16), ambiguous=amb, unassigned=unassigned, rachis_idx=rc, rachis_phi=ph, rachis_h=hh, rachis_r=rr, base_near=base)
    log("assigned", Counter(node[rc].tolist()), "ambiguous", int(amb.sum()), "unassigned", int(unassigned.sum()))
    # blades: frozen R8_pc + rachis absorption with the node groups as seeds
    for p_ in ("tomato3d", "tomato3d.organs"): sys.modules.setdefault(p_, types.ModuleType(p_))
    lo = types.ModuleType("tomato3d.organs.light_organs"); lo.ROOT = Path(out); sys.modules["tomato3d.organs.light_organs"] = lo
    PA = load_module("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load_module("tomato3d.organs.pc_rachis", find("pc_rachis.py")); Xm = X * 1000
    sm = np.nonzero(node > 0)[0]; ts = cKDTree(Xm[sm]); st1 = np.zeros(n, int); bl = np.nonzero(ax == 0)[0]
    for i, nbb in zip(bl, ts.query_ball_point(Xm[bl], H["stage1_mm"])):
        if nbb:
            ls = set(node[sm[nbb]].tolist())
            if len(ls) == 1: st1[i] = ls.pop()
    Z2 = dict(Z); Z2["stage1"] = st1.astype(np.int16); M = PA.Model(Z2); M.norm_unary(); labn = np.nonzero(M.st1n > 0)[0]
    d_, jj, v_ = PA.knn(M.tree_n, M.Xn[labn], 32, PA.P["band_mm"]); jj = np.where(v_, jj, 0); band = labn[(v_ & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1)]
    cb = PA.P["cal"]; PA.P["cal"] = tuple(sorted(set(M.st1n[M.st1n > 0].tolist())))[0::2]; auc = M.colour_auc(band) if len(band) else {k: dict(auc=0.5) for k in ("C1", "C2", "C4")}; PA.P["cal"] = cb
    wcol = {k: (PA.WC[k] if (k == "C3" or auc[k]["auc"] >= PA.P["auc_min"]) else 0.0) for k in PA.WC}; cr = (M.st1n[M.ei] > 0) & (M.st1n[M.ej] > 0) & (M.st1n[M.ei] != M.st1n[M.ej]); se = (M.st1n[M.ei] > 0) & (M.st1n[M.ei] == M.st1n[M.ej])
    xa, xb = M.edge_raw["C3"][cr], M.edge_raw["C3"][se]; auc3 = float(np.searchsorted(np.sort(xb), xa).mean() / len(xb)) if len(xa) and len(xb) else 0.5; wcol["C3"] = PA.WC["C3"] if auc3 >= PA.P["auc_min"] else 0.0
    g, c_ = PA.SETS["B"]; r8 = M.run(g, c_, wcol, M.st1n.copy()); lab = np.zeros(n, int); lab[M.nid] = r8["lab"]
    tp, tl, ti = [], [], []
    for k in range(1, Kc + 1):
        mk = node == k
        if mk.sum() < 2: continue
        P_ = Xm[mk]; cc = P_.mean(0); _, _, vv = np.linalg.svd(P_ - cc, full_matrices=False); t = (P_ - cc) @ vv[0]; bins = np.floor((t - t.min()) / 2.0).astype(int)
        pl_ = np.array([np.median(P_[bins == b_], 0) for b_ in np.unique(bins)]) / 1000.0; tp.append(pl_); tl.append(np.full(len(pl_), k)); ti.append(np.full(len(pl_), k))
    Z3 = dict(Z2); Z3["trace_pts"] = np.vstack(tp); Z3["trace_leaf"] = np.concatenate(tl); Z3["trace_id"] = np.concatenate(ti); rx = PR.Rx(Z3, dict(label=lab)); rr2 = rx.run(); rach = np.zeros(n, int); rach[rx.T] = rr2["lab"][:rx.nT]
    full = np.where(node > 0, node, np.where(rach > 0, rach, lab))
    # scoring (human-seeded labels, only here)
    H8 = np.load(find("labels.npz"))["label"].astype(int); HX = np.load(find("labels_rachis.npz")); hum_r = np.where(HX["target"] & (HX["label"] > 0), HX["label"], 0); human = np.where(hum_r > 0, hum_r, np.where(H8 > 0, H8, 0))
    hl = sorted(set(human[human > 0].tolist())); al = sorted(set(full[full > 0].tolist())); blade = ax == 0; rach_m = ax == 2
    Cm = np.array([[int(((human == h) & (full == a) & blade).sum()) for a in al] for h in hl]); ri, ci = linear_sum_assignment(-Cm); per = {}
    for i_, j_ in zip(ri, ci):
        h, a = hl[i_], al[j_]; ub = int((((human == h) | (full == a)) & blade).sum()); ur = int((((human == h) | (full == a)) & rach_m).sum()); ir = int(((human == h) & (full == a) & rach_m).sum())
        per[str(h)] = dict(node_group=int(a), blade_iou=round(Cm[i_, j_] / ub, 3) if ub else 0.0, rachis_iou=round(ir / ur, 3) if ur else 0.0)
    for h in hl: per.setdefault(str(h), dict(node_group=None, blade_iou=0.0, rachis_iou=0.0))
    npass = sum(1 for v in per.values() if v["blade_iou"] >= H["iou"]); gate = npass >= H["gate_n"]
    blob = np.isin(human, H["blob"]) & rach_m & (node > 0); bc = Counter(node[blob].tolist()); nblob = sum(1 for k, v in bc.items() if v >= H["blob_share"] * max(blob.sum(), 1))
    LT = np.load(find("per_surfel_light.npz")); light = LT["absorbed_ru_mm2"]; area = LT["area_b_silhouette_mm2"]
    R.update(chosen_K=Kc, user_count=8, d_fitted=F["d"], d_reference_deg=137.0, phi0=F["phi0"], heights_mm=[round(float(x), 2) for x in hk], fits={str(k): dict(cost=round(v["cost"], 1), bic=round(v["bic"], 1), d=v["d"], phi0=v["phi0"]) for k, v in fits.items()},
             base_near=int(len(bi)), rachis=int(len(rc)), node_sizes={str(k): int((node == k).sum()) for k in range(1, Kc + 1)}, ambiguous=dict(surfels=int(amb.sum()), share_of_rachis=round(float(amb.sum() / len(rc)), 4), light_share_of_rachis=round(float(light[amb].sum() / max(light[rc].sum(), 1e-12)), 4), light_share_of_plant=round(float(light[amb].sum() / max(light.sum(), 1e-12)), 4)),
             unassigned=int(unassigned.sum()), stage1_blade=int((st1 > 0).sum()), r8pc_rounds=r8["counts"], per_human_leaf=per, leaves_blade_iou_ge_0_5=npass, gate=("lsys_split=pass" if gate else "failed_gate"),
             merged_blob=dict(groups_with_ge_10pct=nblob, counts=dict(bc), split=bool(nblob >= H["blob_min"])), n04_secondary="not_run (n_04 has no axis labels in this form)", seconds=round(time.time() - t0))
    np.savez_compressed(os.path.join(out, "labels_lsys.npz"), rows=rows, label=full.astype(np.int16), node=node.astype(np.int16), human=human.astype(np.int16), ambiguous=amb)
    json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1, default=float)
    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10"); fig, axx = plt.subplots(figsize=(7, 7)); Vp = X[rc] - p0; xy = np.c_[Vp @ e1, Vp @ e2] * 1000; colr = [cmap((k - 1) % 10) if k > 0 else (0.7, 0.7, 0.7, 1) for k in node[rc]]
    axx.scatter(xy[:, 0], xy[:, 1], s=1, c=colr); Rm = np.percentile(rr, 99)
    for k, a in enumerate(pk, 1): axx.plot([0, Rm * np.cos(np.radians(a))], [0, Rm * np.sin(np.radians(a))], color=cmap((k - 1) % 10), lw=1.5); axx.text(1.05 * Rm * np.cos(np.radians(a)), 1.05 * Rm * np.sin(np.radians(a)), str(k))
    axx.set_aspect("equal"); axx.set_title(f"top view: rachis by node (K={Kc}, d={F['d']:.0f} deg); grey = ambiguous/unassigned"); axx.set_xlabel("mm"); fig.savefig(os.path.join(out, "topview.png"), dpi=130); plt.close(fig)
    fig, axx = plt.subplots(figsize=(8, 5)); axx.scatter(ph, hh, s=1, c=[(0.75, 0.75, 0.75)]); axx.scatter(ph[bi], hh[bi], s=2, c=[cmap((k - 1) % 10) for k in nodeb])
    for k in range(Kc): axx.scatter([pk[k]], [hk[k]], s=120, marker="x", c=[cmap(k % 10)]); axx.text(pk[k] + 3, hk[k] + 1, str(k + 1))
    axx.set_xlabel("azimuth (deg)"); axx.set_ylabel("height above pot top (mm)"); axx.set_title("base-near rachis surfels (colour = node) and fitted nodes (x); grey = other rachis"); fig.savefig(os.path.join(out, "phi_h.png"), dpi=130); plt.close(fig)
    log("done", R["gate"], "pass leaves", npass, "K", Kc, "d", F["d"], "blob groups", nblob)


if __name__ == "__main__":
    main()
