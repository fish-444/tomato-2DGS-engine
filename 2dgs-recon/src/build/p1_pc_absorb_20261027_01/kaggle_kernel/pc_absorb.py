"""R8_pc — stage-1 separation, then point-cloud-only absorption (geometry + surfel colour), R3-1b.
Self-contained (numpy + scipy only) so the same file runs as a Kaggle CPU kernel. No photo, SAM or V-P data is read.
Contract: build/p1_pc_absorb_20261027_01/CONTRACT.md.

usage: python pc_absorb.py [--bundle path] [--out dir] [--smoke]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

SETS = {"A": (1.0, 0.0), "B": (0.7, 0.3), "C": (0.5, 0.5)}
WG = dict(G1=0.30, G2=0.20, G3=0.20, G4=0.15, G5=0.05, G6=0.10); WC = dict(C1=0.35, C2=0.35, C3=0.20, C4=0.10)
P = dict(edge_mm=3.0, angle_deg=30.0, blade_like_normal_deg=20.0, trace_mm=3.0, k_local_normal=8, curv_k=16, curv_mm=5.0, lres_k=64, lres_mm=10.0, model_k=24, model_mm=10.0,
         quad_min=10, plane_min=3, side_mm=15.0, side_min=20, missing=3.0, ab_reg=2.0, gamma=1.0, p1=0.6, margin=0.25, rounds=5, band_mm=5.0, hide=0.20, repeats=5,
         norm_sample=2000, auc_min=0.60, cal=(1, 3, 6, 8), rep=(2, 4, 5, 7), cover_min=0.5, raster_mm=0.25, ellipse_k=1.5, cell_mm=10.0)
SRC = {0: "not_a_node", 1: "stage1", 2: "absorbed_round_1", 3: "absorbed_round_2", 4: "absorbed_round_3", 5: "absorbed_round_4", 6: "absorbed_round_5",
       10: "unpainted_contested", 11: "unpainted_not_connected", 12: "unpainted_other_layer", 13: "unpainted_low_colour_confidence", 20: "axis1", 21: "axis2_rachis"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_bundle(arg):
    if arg: return arg
    for g in ("/kaggle/input/**/r8pc_bundle.npz", os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", "r8pc_bundle.npz")):
        hit = glob.glob(g, recursive=True)
        if hit: return hit[0]
    raise SystemExit("r8pc_bundle.npz not found")


def srgb_to_lab(rgb):
    c = np.clip(rgb / 255.0, 0, 1); lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]); xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116); return np.c_[116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]


def angle_signfree(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12), 0, 1)))


def knn(tree, Q, k, r):
    d, j = tree.query(Q, k=k, distance_upper_bound=r); d = d.reshape(len(Q), -1); j = j.reshape(len(Q), -1); return d, j, np.isfinite(d)


class Model:
    def __init__(self, Z, smoke=False):
        t0 = time.time(); self.Z = Z; X = Z["xyz"] * 1000.0; n = len(X); self.n = n
        axis = Z["axis_kind"].astype(int); st1 = Z["stage1"].astype(int); sc = Z["scale"] * 1000.0; N = Z["normal"]
        if smoke:                                                   # code test only: a 60 mm cube around the plant centre
            c = np.median(X[axis == 0], 0); keep = np.all(np.abs(X - c) <= 30, 1); self.full_idx_smoke = np.nonzero(keep)[0]
        self.X, self.N, self.sc, self.axis, self.st1 = X, N, sc, axis, st1
        self.op = Z["opacity"].astype(float); self.t_sheet = float(Z["t_sheet_mm"]); self.off_lim = float(Z["offset_mm"]); self.U = float(Z["reach_mm"])
        lab = (st1 > 0) & (axis == 0); rad = np.sqrt(sc[:, 0] * sc[:, 1]); el = sc.max(1) / np.maximum(sc.min(1), 1e-9)
        self.ref = dict(radius_p25_mm=float(np.percentile(rad[lab], 25)), elong_p75=float(np.percentile(el[lab], 75)))
        tr = Z["trace_pts"] * 1000.0; self.tr_tree = cKDTree(tr)
        lt = cKDTree(X[lab]); li = np.nonzero(lab)[0]; a2 = np.nonzero(axis == 2)[0]; _, jj = lt.query(X[a2], k=P["k_local_normal"]); nb = N[li][jj]; locn = nb[:, 0].copy()
        for q in range(1, nb.shape[1]): locn += np.where(((locn * nb[:, q]).sum(1) < 0)[:, None], -nb[:, q], nb[:, q])
        like = (rad[a2] >= self.ref["radius_p25_mm"]) & (el[a2] <= self.ref["elong_p75"]) & (angle_signfree(N[a2], locn) <= P["blade_like_normal_deg"]) & (self.tr_tree.query(X[a2])[0] > P["trace_mm"])
        node = axis == 0; node[a2[like]] = True
        if smoke: m = np.zeros(n, bool); m[self.full_idx_smoke] = True; node &= m
        self.blade_like = np.zeros(n, bool); self.blade_like[a2[like]] = True
        self.nid = np.nonzero(node)[0]; self.m = len(self.nid); self.pos = -np.ones(n, int); self.pos[self.nid] = np.arange(self.m)
        Xn = X[self.nid]; Nn = N[self.nid]; self.Xn, self.Nn = Xn, Nn; self.tree_n = cKDTree(Xn)
        sidx = Z["sheet_index"].astype(int)[self.nid]; nsh = Z["n_sheets"].astype(int)[self.nid]; nsh[axis[self.nid] == 2] = 1; self.multi = nsh > 1
        E = self.tree_n.query_pairs(P["edge_mm"], output_type="ndarray"); i, j = E[:, 0], E[:, 1]; ni, nj = Nn[i], Nn[j]
        ang = np.degrees(np.arccos(np.clip(np.abs((ni * nj).sum(1)), 0, 1))); mn = ni + np.where(((ni * nj).sum(1) < 0)[:, None], -nj, nj); mn /= np.maximum(np.linalg.norm(mn, axis=1, keepdims=True), 1e-12)
        off = np.abs(((Xn[j] - Xn[i]) * mn).sum(1)); same = ~((nsh[i] > 1) & (nsh[j] > 1)) | (sidx[i] == sidx[j]); ok = (ang <= P["angle_deg"]) & (off <= self.off_lim) & same
        self.ei, self.ej, self.eang, self.eoff = i[ok], j[ok], ang[ok], off[ok]; self.elen = np.linalg.norm(Xn[self.ei] - Xn[self.ej], axis=1)
        m_ = self.m; self.Glen = coo_matrix((np.r_[self.elen, self.elen] + 1e-6, (np.r_[self.ei, self.ej], np.r_[self.ej, self.ei])), shape=(m_, m_)).tocsr()
        self.comp = connected_components(self.Glen, directed=False)[1]
        # curvature (surface variation) and colour fields
        d, jj, v = knn(self.tree_n, Xn, P["curv_k"], P["curv_mm"]); jj = np.where(v, jj, 0); w = v.astype(float); Q = Xn[jj]; cw = (Q * w[..., None]).sum(1) / np.maximum(w.sum(1), 1)[:, None]
        D_ = (Q - cw[:, None]) * w[..., None]; C_ = np.einsum("mki,mkj->mij", D_, Q - cw[:, None]) / np.maximum(w.sum(1), 1)[:, None, None]; ev = np.linalg.eigvalsh(C_)
        self.curv = np.where(v.sum(1) >= 4, ev[:, 0] / np.maximum(ev.sum(1), 1e-12), 0.0)
        lab3 = srgb_to_lab(Z["rgb"].astype(float))[self.nid]; self.Ls, self.ab = lab3[:, 0], lab3[:, 1:]; self.opn = self.op[self.nid]
        d, jj, v = knn(self.tree_n, Xn, P["lres_k"], P["lres_mm"]); jj = np.where(v, jj, 0); w = v * self.opn[jj]; self.Lres = self.Ls - (w * self.Ls[jj]).sum(1) / np.maximum(w.sum(1), 1e-9)
        A = coo_matrix((np.ones(2 * len(self.ei)), (np.r_[self.ei, self.ej], np.r_[self.ej, self.ei])), shape=(m_, m_)).tocsr(); deg = np.asarray(A.sum(1)).ravel() + 1
        s1 = (A @ self.Lres + self.Lres) / deg; s2 = (A @ self.Lres ** 2 + self.Lres ** 2) / deg; self.tex = np.sqrt(np.maximum(s2 - s1 ** 2, 0))
        self.st1n = st1[self.nid].copy(); self.st1n[axis[self.nid] != 0] = 0                        # blade-like axis2 start unlabelled
        same_leaf = (self.st1n[self.ei] > 0) & (self.st1n[self.ei] == self.st1n[self.ej])
        self.edge_raw = dict(G2=self.eang / P["angle_deg"], G4=self.eoff / self.t_sheet, G5=np.abs(self.curv[self.ei] - self.curv[self.ej]), C3=np.abs(self.Lres[self.ei] - self.Lres[self.ej]))
        self.edge_norm = {k: max(float(np.median(v_[same_leaf])), 1e-9) for k, v_ in self.edge_raw.items()}
        self.stem_xy = X[axis == 1][:, :2].mean(0) if (axis == 1).any() else Xn[:, :2].mean(0)
        self.traces = defaultdict(list)
        for t in np.unique(Z["trace_id"]):
            mk = Z["trace_id"] == t; self.traces[int(Z["trace_leaf"][mk][0])].append(tr[mk])
        self.leaves = sorted(set(self.st1n[self.st1n > 0].tolist()))
        log(f"nodes {self.m} (blade {int((axis[self.nid] == 0).sum())}, blade-like axis2 {int(self.blade_like[self.nid].sum())}), edges {len(self.ei)}, leaves {self.leaves} ({time.time() - t0:.0f}s)")

    # ---------- colour noise and normalisation ----------
    def colour_noise(self):
        d, jj, v = knn(self.tree_n, self.Xn, 16, 3.0); jj = np.where(v, jj, 0); same = v & (self.st1n[jj] == self.st1n[:, None]) & (self.st1n[:, None] > 0)
        rgb = self.Z["rgb"][self.nid].astype(float); dif = (rgb[jj] - rgb[:, None]) * same[..., None]; cnt = same.sum(1)
        sd = np.sqrt((dif ** 2).sum((1, 2)) / np.maximum(3 * cnt, 1)); ok = cnt >= 3
        return dict(local_rgb_rms_diff_0_255=round(float(np.median(sd[ok])), 2), p90=round(float(np.percentile(sd[ok], 90)), 2), L_std=round(float(np.median(np.abs(self.Lres))), 2), surfels=int(ok.sum()))

    def leaf_rachis(self, L, seeds_L):
        trs = self.traces.get(L)
        if trs: return trs
        P_ = self.Xn[seeds_L]; c = P_.mean(0); _, _, vt = np.linalg.svd(P_ - c, full_matrices=False); ax = vt[0]; base = P_[np.argmin(np.linalg.norm(P_[:, :2] - self.stem_xy, axis=1))]; pr = (P_ - base) @ ax
        if abs(pr.min()) > abs(pr.max()): ax = -ax; pr = -pr
        L_ = max(pr.max(), 1.0); return [base + np.outer(np.arange(0, L_ + 1e-9, 1.0), ax)]

    def local_feats(self, seeds_L, targets, exclude_self=False):
        """G3, C1, C2, C4 (raw) of target nodes against leaf seeds_L (node indices); NaN where < 3 seeds within 10 mm."""
        tL = cKDTree(self.Xn[seeds_L]); k = P["model_k"] + (1 if exclude_self else 0); d, jj, v = knn(tL, self.Xn[targets], k, P["model_mm"])
        if exclude_self: v &= seeds_L[np.where(v, jj, 0)] != targets[:, None]
        jj = np.where(v, jj, 0); g = seeds_L[jj]; w = v * np.maximum(self.opn[g], 1e-3); cnt = v.sum(1); sw = np.maximum(w.sum(1), 1e-9)
        Q = self.Xn[g]; c = (Q * w[..., None]).sum(1) / sw[:, None]; Dq = Q - c[:, None]; C_ = np.einsum("mki,mkj->mij", Dq * w[..., None], Dq) / sw[:, None, None]
        _, V = np.linalg.eigh(C_ + 1e-9 * np.eye(3)); nrm, e1, e2 = V[:, :, 0], V[:, :, 2], V[:, :, 1]
        u = np.einsum("mki,mi->mk", Dq, e1); vv = np.einsum("mki,mi->mk", Dq, e2); h = np.einsum("mki,mi->mk", Dq, nrm); dp = self.Xn[targets] - c
        up, vp, hp = (dp * e1).sum(1), (dp * e2).sum(1), (dp * nrm).sum(1)
        A = np.stack([u * u, u * vv, vv * vv, u, vv, np.ones_like(u)], -1); AtW = A * w[..., None]; M = np.einsum("mki,mkj->mij", AtW, A) + 1e-6 * np.eye(6)
        Ap = np.stack([up * up, up * vp, vp * vp, up, vp, np.ones_like(up)], -1)
        th_h = np.linalg.solve(M, np.einsum("mki,mk->mi", AtW, h)[..., None])[..., 0]; th_L = np.linalg.solve(M, np.einsum("mki,mk->mi", AtW, self.Ls[g])[..., None])[..., 0]
        quad = cnt >= P["quad_min"]; plane = (cnt >= P["plane_min"]) & ~quad; ok = cnt >= P["plane_min"]
        G3 = np.where(quad, np.abs(hp - (Ap * th_h).sum(1)), np.where(plane, np.abs(hp), np.nan)) / self.t_sheet
        Lmean = (w * self.Ls[g]).sum(1) / sw; C2 = np.where(quad, np.abs(self.Ls[targets] - (Ap * th_L).sum(1)), np.where(plane, np.abs(self.Ls[targets] - Lmean), np.nan))
        mu = (self.ab[g] * w[..., None]).sum(1) / sw[:, None]; Da = self.ab[g] - mu[:, None]; S_ = np.einsum("mki,mkj->mij", Da * w[..., None], Da) / sw[:, None, None] + P["ab_reg"] ** 2 * np.eye(2)
        x = self.ab[targets] - mu; C1 = np.where(ok, np.sqrt(np.einsum("mi,mij,mj->m", x, np.linalg.inv(S_), x)), np.nan)
        sL = (w * self.tex[g]).sum(1) / sw; C4 = np.where(ok, np.abs(np.log((self.tex[targets] + 1) / (sL + 1))), np.nan)
        return dict(G3=G3, C1=C1, C2=C2, C4=C4)

    def rachis_cost(self, L, seeds_L, targets):
        trs = self.leaf_rachis(L, seeds_L); pts = np.vstack(trs); tid = np.concatenate([np.full(len(t), k) for k, t in enumerate(trs)])
        arc = np.concatenate([np.r_[0, np.cumsum(np.linalg.norm(np.diff(t, axis=0), axis=1))] for t in trs]); tang = np.vstack([np.gradient(t, axis=0) if len(t) > 1 else np.array([[1.0, 0, 0]]) for t in trs])
        tree = cKDTree(pts)
        def side_of(Pq):
            d, j = tree.query(Pq); s = np.sign(np.cross(tang[j], Pq - pts[j])[:, 2]); return d, j, s
        ds, js, ss = side_of(self.Xn[seeds_L]); dt, jt, sn = side_of(self.Xn[targets]); bins = int(np.ceil(arc.max())) + 2; cost_side = np.full(len(targets), 0.5)
        for k in range(len(trs)):
            ms = tid[js] == k; mt = tid[jt] == k
            if not mt.any(): continue
            hp = np.bincount(np.clip(arc[js[ms & (ss > 0)]].astype(int), 0, bins - 1), minlength=bins); hm = np.bincount(np.clip(arc[js[ms & (ss < 0)]].astype(int), 0, bins - 1), minlength=bins)
            cp, cm = np.r_[0, np.cumsum(hp)], np.r_[0, np.cumsum(hm)]; a = arc[jt[mt]]; lo = np.clip((a - P["side_mm"]).astype(int), 0, bins); hi = np.clip((a + P["side_mm"]).astype(int) + 1, 0, bins)
            np_, nm_ = cp[hi] - cp[lo], cm[hi] - cm[lo]; both = (np_ >= P["side_min"]) & (nm_ >= P["side_min"]); maj = np.where(np_ >= nm_, 1, -1); none = (np_ + nm_) == 0
            cs = np.where(both, 0.0, np.where(none, 0.5, (sn[mt] != maj).astype(float))); cost_side[mt] = cs
        cost_d = np.where(dt > self.U, 1.0, dt / self.U); return 0.5 * cost_d + 0.5 * cost_side

    def norm_unary(self):
        rng = np.random.default_rng(0); lab = np.nonzero(self.st1n > 0)[0]; samp = rng.choice(lab, min(P["norm_sample"], len(lab)), replace=False); acc = defaultdict(list)
        for L in self.leaves:
            s = samp[self.st1n[samp] == L]
            if len(s) == 0: continue
            f = self.local_feats(np.nonzero(self.st1n == L)[0], s, exclude_self=True)
            for k, v in f.items(): acc[k].append(v[np.isfinite(v)])
        self.unary_norm = {k: max(float(np.median(np.concatenate(v))), 1e-9) for k, v in acc.items()}; return self.unary_norm

    def colour_auc(self, band):
        """AUC of each colour feature, own leaf vs nearest other leaf, on calibration-half boundary nodes (stage-1 seeds, self excluded)."""
        cal = band[np.isin(self.st1n[band], P["cal"])]; d, jj, v = knn(self.tree_n, self.Xn[cal], 32, 10.0); jj = np.where(v, jj, 0); oth = np.zeros(len(cal), int)
        for r in range(len(cal)):
            labs = self.st1n[jj[r][v[r]]]; labs = labs[(labs > 0) & (labs != self.st1n[cal[r]])]; oth[r] = Counter(labs.tolist()).most_common(1)[0][0] if len(labs) else 0
        own = defaultdict(lambda: np.full(len(cal), np.nan)); other = defaultdict(lambda: np.full(len(cal), np.nan))
        for L in self.leaves:
            mo = self.st1n[cal] == L
            if mo.any():
                f = self.local_feats(np.nonzero(self.st1n == L)[0], cal[mo], exclude_self=True)
                for k in ("C1", "C2", "C4"): own[k][mo] = f[k]
            mt = oth == L
            if mt.any():
                f = self.local_feats(np.nonzero(self.st1n == L)[0], cal[mt])
                for k in ("C1", "C2", "C4"): other[k][mt] = f[k]
        out = {}
        for k in ("C1", "C2", "C4"):
            a, b = own[k], other[k]; ok = np.isfinite(a) & np.isfinite(b); out[k] = dict(auc=round(float(((a[ok] < b[ok]) + 0.5 * (a[ok] == b[ok])).mean()) if ok.any() else 0.5, 3), n=int(ok.sum()))
        return out

    # ---------- absorption ----------
    def run(self, g, c, wcol, seed_lab0, verbose=False):
        """Random walker with priors in rounds. Returns labels (node-local), round of acceptance, p1, p2, top-2 leaves, per-round counts."""
        en = self.edge_norm; E = g * (WG["G2"] * self.edge_raw["G2"] / en["G2"] + WG["G4"] * self.edge_raw["G4"] / en["G4"] + WG["G5"] * self.edge_raw["G5"] / en["G5"]) + c * wcol["C3"] * self.edge_raw["C3"] / en["C3"]
        w = np.exp(-E); m_ = self.m; W = coo_matrix((np.r_[w, w], (np.r_[self.ei, self.ej], np.r_[self.ej, self.ei])), shape=(m_, m_)).tocsr(); deg = np.asarray(W.sum(1)).ravel(); Lap = (diags(deg) - W).tocsr()
        lab = seed_lab0.copy(); rnd = np.where(lab > 0, 0, -1); p1 = np.where(lab > 0, 1.0, 0.0); p2 = np.zeros(m_); top = np.zeros((m_, 2), int); counts = []; un = self.unary_norm
        for r in range(1, P["rounds"] + 1):
            U_ = np.nonzero(lab == 0)[0]
            if not len(U_): break
            prior = np.zeros((len(U_), len(self.leaves)))
            for q, L in enumerate(self.leaves):
                sL = np.nonzero(lab == L)[0]
                if not len(sL): continue
                dist = dijkstra(self.Glen, directed=False, indices=sL, min_only=True)[U_]; cand = dist <= self.U
                if not cand.any(): continue
                t = U_[cand]; f = self.local_feats(sL, t); g6 = self.rachis_cost(L, sL, t)
                nz = lambda k: np.where(np.isfinite(f[k]), f[k] / un[k], P["missing"])
                Eu = g * (WG["G1"] * dist[cand] / self.U + WG["G3"] * nz("G3") + WG["G6"] * g6) + c * (wcol["C1"] * nz("C1") + wcol["C2"] * nz("C2") + wcol["C4"] * nz("C4"))
                prior[cand, q] = np.exp(-Eu)
            s = prior.sum(1, keepdims=True); prior = np.where(s > 0, prior / np.maximum(s, 1e-300), 0)
            M_ = np.nonzero(lab > 0)[0]; A = (Lap[U_][:, U_] + P["gamma"] * identity(len(U_))).tocsc(); lu = splu(A); WUM = W[U_][:, M_]; X_ = np.zeros((len(U_), len(self.leaves)))
            for q, L in enumerate(self.leaves):
                X_[:, q] = lu.solve(WUM @ (lab[M_] == L).astype(float) + P["gamma"] * prior[:, q])
            o = np.argsort(-X_, 1); a1 = X_[np.arange(len(U_)), o[:, 0]]; a2 = X_[np.arange(len(U_)), o[:, 1]] if X_.shape[1] > 1 else np.zeros(len(U_))
            p1[U_] = a1; p2[U_] = a2; top[U_, 0] = np.array(self.leaves)[o[:, 0]]; top[U_, 1] = np.array(self.leaves)[o[:, 1]] if X_.shape[1] > 1 else 0
            acc = (a1 >= P["p1"]) & (a1 - a2 >= P["margin"]); lab[U_[acc]] = top[U_[acc], 0]; rnd[U_[acc]] = r; counts.append(int(acc.sum()))
            if verbose: log(f"  round {r}: accepted {int(acc.sum())} of {len(U_)}")
            if not acc.any(): break
        return dict(lab=lab, rnd=rnd, p1=p1, p2=p2, top=top, counts=counts)

    def reasons(self, res, c):
        lab = res["lab"]; has_seed = np.zeros(self.comp.max() + 1, bool); has_seed[self.comp[lab > 0]] = True; rs = np.zeros(self.m, int); un = lab == 0
        nc = un & ~has_seed[self.comp]; rs[nc & self.multi] = 12; rs[nc & ~self.multi] = 11; ct = un & has_seed[self.comp]; rs[ct] = 10
        if c > 0: rs[ct & (self.opn < 0.5)] = 13
        return rs


def union_area(P_, tu, tv, s, c, e1, e2):
    k, px = P["ellipse_k"], P["raster_mm"]; A = np.c_[(tu @ e1), (tu @ e2)] * (k * s[:, 0])[:, None]; B = np.c_[(tv @ e1), (tv @ e2)] * (k * s[:, 1])[:, None]; X = np.c_[(P_ - c) @ e1, (P_ - c) @ e2]
    r = np.sqrt((A ** 2).sum(1) + (B ** 2).sum(1)); lo = (X - r[:, None]).min(0) - px; hi = (X + r[:, None]).max(0) + px
    W = int(np.ceil((hi[0] - lo[0]) / px)) + 1; H = int(np.ceil((hi[1] - lo[1]) / px)) + 1; cov = np.zeros((H, W), bool)
    for i in range(len(P_)):
        det = A[i, 0] * B[i, 1] - A[i, 1] * B[i, 0]; i0 = int((X[i, 0] - r[i] - lo[0]) / px); i1 = int((X[i, 0] + r[i] - lo[0]) / px) + 1; j0 = int((X[i, 1] - r[i] - lo[1]) / px); j1 = int((X[i, 1] + r[i] - lo[1]) / px) + 1
        if abs(det) < 1e-12: cov[min(max(int((X[i, 1] - lo[1]) / px), 0), H - 1), min(max(int((X[i, 0] - lo[0]) / px), 0), W - 1)] = True; continue
        gx = lo[0] + (np.arange(i0, i1) + 0.5) * px - X[i, 0]; gy = lo[1] + (np.arange(j0, j1) + 0.5) * px - X[i, 1]; GX, GY = np.meshgrid(gx, gy)
        uu = (B[i, 1] * GX - B[i, 0] * GY) / det; vv = (-A[i, 1] * GX + A[i, 0] * GY) / det; cov[j0:j1, i0:i1] |= (uu * uu + vv * vv) <= 1.0
    return float(cov.sum()) * px * px


def area_mm2(Z, X, idx):
    if len(idx) == 0: return 0.0
    tu, tv, sc, N = Z["tu"], Z["tv"], Z["scale"] * 1000.0, Z["normal"]; cell = np.floor(X[idx] / P["cell_mm"]).astype(np.int64); groups = defaultdict(list)
    for i, k in zip(idx, map(tuple, cell)): groups[k].append(i)
    tot = 0.0
    for mem in groups.values():
        mem = np.array(mem)
        if len(mem) >= 3:
            c = X[mem].mean(0); w, v = np.linalg.eigh(np.cov((X[mem] - c).T) + 1e-12 * np.eye(3)); e1, e2 = v[:, 2], v[:, 1]
        else: c = X[mem].mean(0); e1 = tu[mem[0]]; e2 = np.cross(N[mem[0]], e1)
        tot += union_area(X[mem], tu[mem], tv[mem], sc[mem], c, e1, e2)
    return tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", default=None); ap.add_argument("--out", default=None); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    out = a.out or ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "r8pc_out")); os.makedirs(out, exist_ok=True)
    t0 = time.time(); Z = dict(np.load(find_bundle(a.bundle), allow_pickle=False)); M = Model(Z, smoke=a.smoke); RES = dict(status="R3-1b_unaccepted", scale="conditional_on_unvalidated_scale", no_photo_data=True, params=P,
                                                                                                             weights=dict(sets=SETS, geometry=WG, colour=WC), smoke=a.smoke)
    RES["nodes"] = dict(nodes=M.m, blade=int((M.axis[M.nid] == 0).sum()), blade_like_axis2=int(M.blade_like.sum()), blade_like_axis2_in_nodes=int(M.blade_like[M.nid].sum()), edges=int(len(M.ei)), reference=M.ref,
                        stage1_unassigned_blade=int(((M.st1 == 0) & (M.axis == 0)).sum()))
    RES["colour_noise"] = M.colour_noise(); log("colour noise", RES["colour_noise"]); RES["edge_norm"] = M.edge_norm; RES["unary_norm"] = M.norm_unary(); log("norms", RES["edge_norm"], RES["unary_norm"])
    labn = np.nonzero(M.st1n > 0)[0]; d, jj, v = knn(M.tree_n, M.Xn[labn], 32, P["band_mm"]); jj = np.where(v, jj, 0); other = (v & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1); band = labn[other]
    RES["band"] = dict(n=int(len(band)), per_leaf={str(L): int((M.st1n[band] == L).sum()) for L in M.leaves})
    RES["colour_auc"] = M.colour_auc(band); wcol = {k: (WC[k] if (k == "C3" or RES["colour_auc"][k]["auc"] >= P["auc_min"]) else 0.0) for k in WC}
    # C3 is an edge feature: its usefulness = AUC of |dLres| on cross-leaf vs same-leaf boundary edges
    cross = (M.st1n[M.ei] > 0) & (M.st1n[M.ej] > 0) & (M.st1n[M.ei] != M.st1n[M.ej]) & np.isin(M.st1n[M.ei], P["cal"]); samee = (M.st1n[M.ei] > 0) & (M.st1n[M.ei] == M.st1n[M.ej]) & np.isin(M.st1n[M.ei], P["cal"])
    xa, xb = M.edge_raw["C3"][cross], M.edge_raw["C3"][samee]
    if len(xa) and len(xb):
        rng = np.random.default_rng(0); xb_s = rng.choice(xb, min(len(xb), 20000), replace=False); r_ = np.searchsorted(np.sort(xb_s), xa); auc3 = float(r_.mean() / len(xb_s))
    else: auc3 = 0.5
    RES["colour_auc"]["C3"] = dict(auc=round(auc3, 3), n=int(len(xa)), note="edge feature: P(|dL-res| cross-leaf edge > same-leaf edge)"); wcol["C3"] = WC["C3"] if auc3 >= P["auc_min"] else 0.0
    RES["colour_weights_used_in_B_C"] = wcol; log("colour auc", RES["colour_auc"], "weights", wcol)
    json.dump(RES, open(os.path.join(out, "results.json"), "w"), indent=1, default=float)
    # self-test
    test = {}
    for sname, (g, c) in SETS.items():
        reps = []
        for rep in range(P["repeats"]):
            rng = np.random.default_rng(rep); hid = rng.choice(band, int(round(P["hide"] * len(band))), replace=False); s0 = M.st1n.copy(); s0[hid] = 0
            r = M.run(g, c, wcol, s0); got = r["lab"][hid]; true = M.st1n[hid]; row = {}
            for half, leaves in (("calibration", P["cal"]), ("report", P["rep"]), ("all", tuple(M.leaves))):
                mk = np.isin(true, leaves); acc_m = mk & (got > 0); na = int(acc_m.sum()); ok = int((got[acc_m] == true[acc_m]).sum())
                row[half] = dict(hidden=int(mk.sum()), accepted=na, correct=ok, accuracy=(ok / na if na else None), wrong_rate=((na - ok) / na if na else None), coverage=(na / mk.sum() if mk.sum() else None))
            row["per_leaf"] = {str(L): dict(hidden=int((true == L).sum()), accepted=int(((true == L) & (got > 0)).sum()), correct=int(((true == L) & (got == L)).sum())) for L in M.leaves}
            reps.append(row); log(f"test {sname} rep {rep}: cal acc {row['calibration']['accuracy']} cov {row['calibration']['coverage']} | rep acc {row['report']['accuracy']} cov {row['report']['coverage']}")
        agg = {}
        for half in ("calibration", "report", "all"):
            na = sum(r_[half]["accepted"] for r_ in reps); ok = sum(r_[half]["correct"] for r_ in reps); hd = sum(r_[half]["hidden"] for r_ in reps)
            agg[half] = dict(accuracy=(ok / na if na else None), wrong_rate=((na - ok) / na if na else None), coverage=(na / hd if hd else None), accepted=na, hidden=hd,
                             accuracy_per_repeat=[r_[half]["accuracy"] for r_ in reps])
        test[sname] = dict(pooled=agg, repeats=reps); RES["self_test"] = test; json.dump(RES, open(os.path.join(out, "results.json"), "w"), indent=1, default=float)
    # set choice on the calibration half only
    cands = [(s, test[s]["pooled"]["calibration"]) for s in SETS]; ok_cov = [x for x in cands if (x[1]["coverage"] or 0) >= P["cover_min"] and x[1]["accuracy"] is not None]
    pool = ok_cov if ok_cov else [x for x in cands if x[1]["accuracy"] is not None]; best_acc = max(x[1]["accuracy"] for x in pool)
    tied = [x for x in pool if x[1]["accuracy"] >= best_acc - 0.005]; chosen = max(tied, key=lambda x: x[1]["coverage"] or 0)[0]
    rep_ = test[chosen]["pooled"]["report"]; gate = bool(rep_["accuracy"] is not None and rep_["accuracy"] >= 0.90 and rep_["wrong_rate"] <= 0.05)
    RES["choice"] = dict(chosen=chosen, rule="calibration half: highest accuracy with coverage >= 0.5 (ties within 0.005 -> higher coverage)", reached_cover_min=bool(ok_cov), report_half=rep_, gate_pass=gate)
    RES["tag"] = "gate_passed_self_test" if gate else "gate_failed_viewing_only"; log("choice", RES["choice"])
    # final runs on the full stage-1 labels (all three for the ablation)
    finals = {}
    for sname, (g, c) in SETS.items():
        r = M.run(g, c, wcol, M.st1n.copy(), verbose=True); rs = M.reasons(r, c); finals[sname] = (r, rs)
        RES.setdefault("final", {})[sname] = dict(rounds=r["counts"], absorbed=int((r["rnd"] > 0).sum()), unpainted_nodes=int((r["lab"] == 0).sum()), reasons={SRC[k]: int((rs == k).sum()) for k in (10, 11, 12, 13)})
    n = M.n; X = M.X; z7 = Z["r7"].astype(int); st1 = M.st1
    def full_labels(r):
        f = np.zeros(n, int); f[M.nid] = r["lab"]; return f
    for sname, (r, rs) in finals.items():
        f = full_labels(r); src = np.zeros(n, np.int16); src[M.axis == 1] = 20; src[(M.axis == 2)] = 21; srcn = np.where(r["rnd"] == 0, 1, np.where(r["rnd"] > 0, 1 + r["rnd"], rs)); src[M.nid] = srcn
        p1 = np.zeros(n, np.float32); p2 = np.zeros(n, np.float32); p1[M.nid] = r["p1"]; p2[M.nid] = r["p2"]; t1 = np.zeros(n, np.int16); t2 = np.zeros(n, np.int16); t1[M.nid] = r["top"][:, 0]; t2[M.nid] = r["top"][:, 1]
        d_ = os.path.join(out, "R8_pc" if sname == chosen else f"set_{sname}"); os.makedirs(d_, exist_ok=True)
        np.savez_compressed(os.path.join(d_, "labels.npz"), rows=Z["rows"], label=f.astype(np.int16), source=src, source_names=np.array([f"{k}:{v}" for k, v in SRC.items()]), p1=p1, p2=p2, top1=t1, top2=t2,
                            stage1=st1.astype(np.int16), r7=z7.astype(np.int16), set_name=np.array(sname), node=np.isin(np.arange(n), M.nid))
    # tables for stage 1, R7, R8_pc (chosen)
    r, rs = finals[chosen]; r8 = full_labels(r); blike = M.blade_like
    def table(lab):
        bl = M.axis == 0; rows = dict(blade_painted=bl & (lab > 0), blade_unpainted=bl & (lab <= 0), axis1=M.axis == 1, axis2_all=M.axis == 2, axis2_blade_like_retested=blike, axis2_blade_like_painted=blike & (lab > 0))
        return {k: dict(count=int(v.sum()), mm2=round(area_mm2(Z, X, np.nonzero(v)[0]), 1)) for k, v in rows.items()}
    if not a.smoke:
        RES["table"] = dict(denominator=f"all {n} reconstructed surfels", stage1=table(st1), R7=table(z7), R8_pc=table(r8), total_blade_mm2=round(area_mm2(Z, X, np.nonzero(M.axis == 0)[0]), 1))
        un = np.zeros(n, bool); un[M.nid] = (r["lab"] == 0) & (r["p1"] - r["p2"] < P["margin"]); t1 = np.zeros(n, int); t1[M.nid] = r["top"][:, 0]
        RES["per_leaf_mm2"] = {str(L): dict(stage1=round(area_mm2(Z, X, np.nonzero(st1 == L)[0]), 1), R7=round(area_mm2(Z, X, np.nonzero(z7 == L)[0]), 1), R8_pc=round(area_mm2(Z, X, np.nonzero(r8 == L)[0]), 1),
                                            uncertain_near=round(area_mm2(Z, X, np.nonzero(un & (t1 == L))[0]), 1)) for L in M.leaves}
    m45 = (M.st1n == 0) & (np.sort(r["top"], 1) == np.array([4, 5])).all(1)
    RES["leaf4_5_boundary"] = dict(nodes_with_top2_4_and_5=int(m45.sum()), to_4=int((m45 & (r["lab"] == 4)).sum()), to_5=int((m45 & (r["lab"] == 5)).sum()), unpainted=int((m45 & (r["lab"] == 0)).sum()))
    both = (r8 > 0) & (z7 > 0); absd = np.zeros(n, bool); absd[M.nid] = r["rnd"] > 0
    RES["agreement_with_R7_informational"] = dict(labelled_in_both=int(both.sum()), equal=round(float((r8[both] == z7[both]).mean()), 4) if both.any() else None,
                                               absorbed_and_R7_labelled=int((absd & (z7 > 0)).sum()), absorbed_equal=round(float((r8[absd & (z7 > 0)] == z7[absd & (z7 > 0)]).mean()), 4) if (absd & (z7 > 0)).any() else None)
    RES["seconds"] = round(time.time() - t0); json.dump(RES, open(os.path.join(out, "results.json"), "w"), indent=1, default=float); log("done", RES["seconds"], "s; chosen", chosen, "gate", gate)


if __name__ == "__main__":
    main()
