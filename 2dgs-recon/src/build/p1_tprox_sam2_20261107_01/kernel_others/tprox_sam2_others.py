"""Amendment A2: arm A (H2 leaflet groups, share 0.40, max-rule) vs T0 on the 11 other plants, Kaggle CPU. Ungraded: no human label anywhere.
Visibility = seed2d.votes_per_view splat rule (1512 px, tol 1.5 mm, radius 1.5 sigma, <= 14 px), copied; walker = frozen pc_absorb (set A) + the grouped walker of tprox_sam2.py; rachis = frozen pc_rachis."""
import glob, importlib.util, json, os, sys, time, traceback, types
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu

OUT = "/kaggle/working"; T0 = time.time(); R = {}; SHARE = 0.40; MIN_SURFELS = 5; H2 = 1536
VIS = dict(res=1512, tol_m=0.0015, radius_sigma=1.5, max_r_px=14)


def log(*a): print(time.strftime("%H:%M:%S"), f"({time.time() - T0:.0f}s)", *a, flush=True)
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)
def find(name): hit = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hit, name; return hit[0]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


for p_ in ("tomato3d", "tomato3d.organs"): sys.modules.setdefault(p_, types.ModuleType(p_))
lo = types.ModuleType("tomato3d.organs.light_organs"); lo.ROOT = Path(OUT); sys.modules["tomato3d.organs.light_organs"] = lo
PA = load_module("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load_module("tomato3d.organs.pc_rachis", find("pc_rachis.py")); G_SET, C_SET = PA.SETS["A"]


def rle_decode(rle):                                        # tomato3d/leafseg/pixels.py
    h, w = rle["size"]; flat = np.zeros(h * w, np.uint8); pos, val = 0, 0
    for c in rle["counts"]: flat[pos:pos + c] = val; pos += c; val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)


def scaled_K(K, src, dst):                                  # tomato3d/recon/charuco_a3.py
    s = dst / src; K2 = K.copy(); K2[0, 0] *= s; K2[1, 1] *= s; K2[0, 2] = (K[0, 2] + 0.5) * s - 0.5; K2[1, 2] = (K[1, 2] + 0.5) * s - 0.5; return K2


def splat_zbuffer(u, w, z, r_px, res):                      # tomato3d/organs/seed2d.py
    zb = np.full((res, res), np.inf); Rr = int(min(np.ceil(r_px.max()), VIS["max_r_px"])) if len(r_px) else 0
    for dy in range(-Rr, Rr + 1):
        for dx in range(-Rr, Rr + 1):
            m = (dx * dx + dy * dy) <= r_px * r_px; uu, ww = u[m] + dx, w[m] + dy; ok = (uu >= 0) & (ww >= 0) & (uu < res) & (ww < res); np.minimum.at(zb, (ww[ok], uu[ok]), z[m][ok])
    return zb


def visible_pixels(xyz, sc, po, v):
    """(visible surfel indices, their 1536-grid pixel) — seed2d.votes_per_view 'splat' rule."""
    res = VIS["res"]; K = scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res); Rm = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"])
    xc = xyz @ Rm.T + t; z = xc[:, 2]; u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); idx = np.nonzero(ok)[0]; r_px = VIS["radius_sigma"] * sc[idx].max(1) * K[0, 0] / z[idx]
    zb = splat_zbuffer(u[idx], w[idx], z[idx], np.minimum(r_px, VIS["max_r_px"]), res); vis = idx[z[idx] <= zb[w[idx], u[idx]] + VIS["tol_m"]]
    return vis, w[vis] * H2 // res, u[vis] * H2 // res


class UF:
    def __init__(self, n): self.p = list(range(n))
    def f(self, a):
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def groups(pid, Z, d):
    inp = json.load(open(f"{d}/inputs.json")); sf = np.load(f"{d}/surfels.npz"); bi = sf["bundle_index"]; ax0 = Z["axis_kind"].astype(int) == 0; st1 = Z["stage1"].astype(int)
    inv_b = -np.ones(len(sf["xyz"]), int); inv_b[bi] = np.arange(len(bi)); sets = []
    for v in inp["views"]:
        masks = inp["h2"].get(v, [])
        if not masks: continue
        vis, wi, ui = visible_pixels(sf["xyz"], sf["sc"], inp["po"], v); br = inv_b[vis]; keep = (br >= 0); keep[keep] &= ax0[br[keep]]; br, wi, ui = br[keep], wi[keep], ui[keep]
        for rle in masks:
            m = rle_decode(rle); mem = np.unique(br[m[wi, ui]])
            if len(mem) >= MIN_SURFELS: sets.append(mem)
    inv = {}
    for si, s in enumerate(sets):
        for r in s.tolist(): inv.setdefault(r, []).append(si)
    uf = UF(len(sets)); cand = set()
    for lst in inv.values():
        for a in range(len(lst)):
            for c in range(a + 1, len(lst)): cand.add((lst[a], lst[c]))
    for a, c in cand:
        if len(np.intersect1d(sets[a], sets[c], assume_unique=True)) / max(len(sets[a]), len(sets[c])) >= SHARE: uf.u(a, c)
    n = len(Z["rows"]); gid = -np.ones(n, int); conflict = np.zeros(n, bool)
    for r, lst in inv.items():
        roots = {uf.f(x) for x in lst}
        if len(roots) == 1: gid[r] = roots.pop()
        else: conflict[r] = True
    mixed = 0
    for g in np.unique(gid[gid >= 0]):
        m = gid == g
        if len(set(st1[m & (st1 > 0)].tolist())) > 1: gid[m] = -1; mixed += 1
    ug, gid2 = np.unique(gid, return_inverse=True); gid = np.where(gid >= 0, gid2 - (1 if ug[0] == -1 else 0), -1)
    return gid, dict(masks=len(sets), groups=int(gid.max() + 1), grouped_blade=int((gid >= 0).sum()), blade=int(ax0.sum()), coverage_blade=round(float((gid >= 0).sum() / max(ax0.sum(), 1)), 4),
                     group_conflict_surfels=int(conflict.sum()), group_mixed_seeds=mixed)


def run_grouped(M, wcol, seed0, cl):
    g, c = G_SET, C_SET; en = M.edge_norm; P = PA.P; WG, un = PA.WG, M.unary_norm
    E = g * (WG["G2"] * M.edge_raw["G2"] / en["G2"] + WG["G4"] * M.edge_raw["G4"] / en["G4"] + WG["G5"] * M.edge_raw["G5"] / en["G5"]) + c * wcol["C3"] * M.edge_raw["C3"] / en["C3"]
    w = np.exp(-E); m_ = M.m; K = int(cl.max()) + 1; S = csr_matrix((np.ones(m_), (np.arange(m_), cl)), shape=(m_, K))
    W = coo_matrix((np.r_[w, w], (np.r_[M.ei, M.ej], np.r_[M.ej, M.ei])), shape=(m_, m_)).tocsr(); Wc = (S.T @ W @ S).tocsr(); Wc.setdiag(0); Wc.eliminate_zeros()
    Lap = (diags(np.asarray(Wc.sum(1)).ravel()) - Wc).tocsr(); labc = np.zeros(K, int); sd = seed0 > 0; labc[cl[sd]] = seed0[sd]; p1c = np.where(labc > 0, 1.0, 0.0); p2c = np.zeros(K); counts = []
    for r in range(1, P["rounds"] + 1):
        lab = labc[cl]; Uc = np.nonzero(labc == 0)[0]
        if not len(Uc): break
        U_ = np.nonzero(lab == 0)[0]; prior_n = np.zeros((len(U_), len(M.leaves)))
        for q, L in enumerate(M.leaves):
            sL = np.nonzero(lab == L)[0]
            if not len(sL): continue
            dist = dijkstra(M.Glen, directed=False, indices=sL, min_only=True)[U_]; cand = dist <= M.U
            if not cand.any(): continue
            t = U_[cand]; f = M.local_feats(sL, t); g6 = M.rachis_cost(L, sL, t); nz = lambda k: np.where(np.isfinite(f[k]), f[k] / un[k], P["missing"])  # noqa: E731
            prior_n[cand, q] = np.exp(-(g * (WG["G1"] * dist[cand] / M.U + WG["G3"] * nz("G3") + WG["G6"] * g6) + c * (wcol["C1"] * nz("C1") + wcol["C2"] * nz("C2") + wcol["C4"] * nz("C4"))))
        pos = -np.ones(K, int); pos[Uc] = np.arange(len(Uc)); prior = np.zeros((len(Uc), len(M.leaves))); np.add.at(prior, pos[cl[U_]], prior_n)
        s = prior.sum(1, keepdims=True); prior = np.where(s > 0, prior / np.maximum(s, 1e-300), 0)
        Mc = np.nonzero(labc > 0)[0]; lu = splu((Lap[Uc][:, Uc] + P["gamma"] * identity(len(Uc))).tocsc()); WUM = Wc[Uc][:, Mc]; X_ = np.zeros((len(Uc), len(M.leaves)))
        for q, L in enumerate(M.leaves): X_[:, q] = lu.solve(WUM @ (labc[Mc] == L).astype(float) + P["gamma"] * prior[:, q])
        o = np.argsort(-X_, 1); a1 = X_[np.arange(len(Uc)), o[:, 0]]; a2 = X_[np.arange(len(Uc)), o[:, 1]] if X_.shape[1] > 1 else np.zeros(len(Uc))
        p1c[Uc] = a1; p2c[Uc] = a2; acc = (a1 >= P["p1"]) & (a1 - a2 >= P["margin"]); labc[Uc[acc]] = np.array(M.leaves)[o[acc, 0]]; counts.append(int(acc.sum()))
        if not acc.any(): break
    return dict(lab=labc[cl], p1=p1c[cl], p2=p2c[cl], counts=counts)


def finish(Z, M, r):
    n = len(Z["rows"]); f = np.zeros(n, int); f[M.nid] = r["lab"]; rx = PR.Rx(Z, dict(label=f)); rr = rx.run(); full = f.copy(); full[rx.T] = rr["lab"][:rx.nT]; return f, full


for d in sorted(glob.glob("/kaggle/input/**/r8pc_bundle.npz", recursive=True)):
    d = os.path.dirname(d); pid = os.path.basename(d); rec = {}
    try:
        Z = dict(np.load(f"{d}/r8pc_bundle.npz")); M = PA.Model(Z); M.norm_unary()
        labn = np.nonzero(M.st1n > 0)[0]; _, jj, v_ = PA.knn(M.tree_n, M.Xn[labn], 32, PA.P["band_mm"]); jj = np.where(v_, jj, 0); band = labn[(v_ & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1)]
        auc = M.colour_auc(band) if len(band) else {k: dict(auc=0.5) for k in ("C1", "C2", "C4")}                     # as the frozen main()
        wcol = {k: (PA.WC[k] if (k == "C3" or auc[k]["auc"] >= PA.P["auc_min"]) else 0.0) for k in PA.WC}     # set A: colour weight 0, so wcol does not enter the energies
        r0 = M.run(G_SET, C_SET, wcol, M.st1n.copy()); f0, full0 = finish(Z, M, r0)
        t0r = np.load(f"{d}/t0_r8pc.npz"); t0x = np.load(f"{d}/t0_rachis.npz"); rec["T0_match_r8pc"] = round(float((t0r["label"] == f0).mean()), 6); rec["T0_match_rachis"] = round(float((t0x["label"] == full0).mean()), 6)
        gid, gi = groups(pid, Z, d); rec.update(gi); gn = gid[M.nid]; K0 = int(gn.max()) + 1; cl = np.where(gn >= 0, gn, K0 + np.arange(M.m)); _, cl = np.unique(cl, return_inverse=True)
        rA = run_grouped(M, wcol, M.st1n.copy(), cl); fA, fullA = finish(Z, M, rA)
        ax0 = Z["axis_kind"].astype(int) == 0; both = (full0 > 0) & (fullA > 0)
        rec.update(assigned_T0=round(float((full0 > 0).mean()), 4), assigned_A=round(float((fullA > 0).mean()), 4), assigned_blade_T0=round(float((full0[ax0] > 0).mean()), 4), assigned_blade_A=round(float((fullA[ax0] > 0).mean()), 4),
                   changed_leaf_share=round(float((full0[both] != fullA[both]).mean()), 4) if both.any() else None, changed_surfels=int((full0 != fullA).sum()), rounds_T0=r0["counts"], rounds_A=rA["counts"])
        np.savez_compressed(f"{OUT}/labels_T0_{pid}.npz", rows=Z["rows"], label=full0.astype(np.int16), r8_pc=f0.astype(np.int16)); np.savez_compressed(f"{OUT}/labels_A_{pid}.npz", rows=Z["rows"], label=fullA.astype(np.int16), r8_pc=fA.astype(np.int16), gid=gid.astype(np.int32))
    except Exception:
        rec["error"] = traceback.format_exc()[-2500:]
    R[pid] = rec; save(); log(pid, {k: v for k, v in rec.items() if k != "error"}, rec.get("error", "")[-300:])
R["seconds"] = round(time.time() - T0); save()
