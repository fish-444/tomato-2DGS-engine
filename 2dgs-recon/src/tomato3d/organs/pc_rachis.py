"""R8_pc_rachis: axis_kind 2 (rachis-like) surfels around each compound leaf absorbed into that leaf as its rachis part. Point cloud + the user's rachis traces only
(no photo / SAM / V-P). Contract: build/p1_pc_absorb_20261027_01/rachis/CONTRACT.md.

usage: python -m tomato3d.organs.pc_rachis
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from tomato3d.organs import light_organs as LO
from tomato3d.organs import pc_absorb as PA

D0 = LO.ROOT / "build/p1_pc_absorb_20261027_01"; OUT = D0 / "R8_pc_rachis"
P = dict(edge_mm=3.0, edge_scale_mm=2.0, nb3d_mm=8.0, trace_cand_mm=10.0, trace_seed_mm=3.0, w_nb3d=0.5, w_tr=0.3, w_dir=0.2, temp=3.0, d_floor_mm=0.5, gamma=1.0, p1=0.6, margin=0.25, rounds=5, neutral=0.5)
SRC = {1: "r8_pc_label", 2: "trace_seed", 3: "round_1", 4: "round_2", 5: "round_3", 6: "round_4", 7: "round_5", 10: "unpainted_contested", 11: "unpainted_not_connected", 12: "unpainted_no_candidate", 20: "axis1"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Rx:
    def __init__(self, Z, R8):
        t0 = time.time(); self.Z = Z; X = Z["xyz"] * 1000.0; self.X = X; ax = Z["axis_kind"].astype(int); lab = R8["label"].astype(int); self.lab, self.ax = lab, ax
        self.T = np.nonzero((ax == 2) & (lab <= 0))[0]; B = np.nonzero(lab > 0)[0]; self.leaves = sorted(set(lab[lab > 0].tolist())); tT = cKDTree(X[self.T]); tB = cKDTree(X[B])
        near = np.unique(np.concatenate([x for x in tB.query_ball_tree(tT, P["edge_mm"]) if x] or [np.zeros(0, int)])) if False else None
        # blade nodes: R8_pc-labelled surfels within 3 mm of a target
        d, _ = tT.query(X[B], distance_upper_bound=P["edge_mm"]); self.S = B[np.isfinite(d)]; self.nodes = np.concatenate([self.T, self.S]); self.nT = len(self.T); n_ = len(self.nodes)
        self.pos = -np.ones(len(X), int); self.pos[self.nodes] = np.arange(n_)
        pr = cKDTree(X[self.nodes]).query_pairs(P["edge_mm"], output_type="ndarray"); keep = ~((pr[:, 0] >= self.nT) & (pr[:, 1] >= self.nT)); pr = pr[keep]
        dd = np.linalg.norm(X[self.nodes][pr[:, 0]] - X[self.nodes][pr[:, 1]], axis=1); w = np.exp(-dd / P["edge_scale_mm"])
        self.W = coo_matrix((np.r_[w, w], (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])), shape=(n_, n_)).tocsr(); self.comp = connected_components(self.W, directed=False)[1]
        # 3D-surroundings share (rule 14) per target and leaf
        self.K = max(9, int(lab.max()) + 1)                                  # leaf ids up to K-1 (9 = R3-1b's 8 leaves; auto seeds may have more)
        self.nb = np.zeros((self.nT, self.K)); nbl = tB.query_ball_point(X[self.T], P["nb3d_mm"])
        for i, idx in enumerate(nbl):
            if not idx: continue
            idx = np.asarray(idx); dist = np.linalg.norm(X[B[idx]] - X[self.T[i]], axis=1); self.nb[i] = np.bincount(lab[B[idx]], weights=1.0 / np.maximum(dist, P["d_floor_mm"]), minlength=self.K)[:self.K]
        s = self.nb.sum(1, keepdims=True); self.nb = np.where(s > 0, self.nb / np.maximum(s, 1e-12), 0.0)
        # traces
        tr = Z["trace_pts"] * 1000.0; self.trace = {}; tleaf = Z["trace_leaf"].astype(int); tid = Z["trace_id"].astype(int)
        for L in sorted(set(tleaf.tolist())):
            pts, tg = [], []
            for t in sorted(set(tid[tleaf == L].tolist())):
                q = tr[tid == t]; pts.append(q); tg.append(np.gradient(q, axis=0) if len(q) > 1 else np.array([[1.0, 0, 0]]))
            pts = np.vstack(pts); tg = np.vstack(tg); tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12); self.trace[L] = (cKDTree(pts), tg)
        sc = Z["scale"] * 1000.0; tau = np.where((sc[:, 0] >= sc[:, 1])[:, None], Z["tu"], Z["tv"]); self.tau = tau[self.T]; self.trace_d = {}; self.trace_dir = {}
        for L, (tree, tg) in self.trace.items():
            dist, j = tree.query(X[self.T]); self.trace_d[L] = dist; self.trace_dir[L] = np.abs((self.tau * tg[j]).sum(1))
        allt = np.vstack([np.c_[self.trace_d[L], np.full(self.nT, L)][None] for L in self.trace]) if self.trace else None
        dm = np.stack([self.trace_d[L] for L in sorted(self.trace)]); k = dm.argmin(0); self.near_leaf = np.array(sorted(self.trace))[k]; self.near_d = dm.min(0)
        self.trace_seed = self.near_d <= P["trace_seed_mm"]
        # area helper needs the bundle arrays in mm
        log(f"targets {self.nT}, blade nodes {len(self.S)}, edges {len(pr)}, trace seeds {int(self.trace_seed.sum())}, components {self.comp.max() + 1} ({time.time() - t0:.0f}s)")

    def prior(self, hide_leaf=None):
        """Prior over leaves per target (nT x 9); candidates = nb3d > 0 or trace within 10 mm."""
        E = np.full((self.nT, self.K), np.inf); cand = np.zeros((self.nT, self.K), bool)
        for L in self.leaves:
            has_tr = L in self.trace and L != hide_leaf
            tr = np.minimum(self.trace_d[L], 10.0) / 10.0 if has_tr else np.full(self.nT, P["neutral"]); dr = np.where(self.trace_d[L] <= P["trace_cand_mm"], 1 - self.trace_dir[L], P["neutral"]) if has_tr else np.full(self.nT, P["neutral"])
            c = (self.nb[:, L] > 0) | ((self.trace_d[L] <= P["trace_cand_mm"]) if has_tr else False); cand[:, L] = c
            E[:, L] = np.where(c, P["w_nb3d"] * (1 - self.nb[:, L]) + P["w_tr"] * tr + P["w_dir"] * dr, np.inf)
        lam = np.exp(-P["temp"] * np.where(np.isfinite(E), E, 50.0)) * cand; s = lam.sum(1, keepdims=True); return np.where(s > 0, lam / np.maximum(s, 1e-300), 0.0), cand

    def run(self, hide_leaf=None):
        lam, cand = self.prior(hide_leaf); n_ = len(self.nodes); lab = np.zeros(n_, int); lab[self.nT:] = self.lab[self.S]; rnd = np.zeros(n_, int)
        ts = self.trace_seed.copy()
        if hide_leaf is not None: ts &= self.near_leaf != hide_leaf
        lab[:self.nT][ts] = self.near_leaf[ts]; rnd[:self.nT][ts] = -1; deg = np.asarray(self.W.sum(1)).ravel(); Lap = (diags(deg) - self.W).tocsr(); p1 = np.zeros(n_); p2 = np.zeros(n_); top = np.zeros((n_, 2), int); counts = []
        for r in range(1, P["rounds"] + 1):
            U = np.nonzero(lab == 0)[0]; U = U[U < self.nT]
            if not len(U): break
            M = np.nonzero(lab > 0)[0]; A = (Lap[U][:, U] + P["gamma"] * identity(len(U))).tocsc(); lu = splu(A); WUM = self.W[U][:, M]; Xp = np.zeros((len(U), len(self.leaves)))
            for q, L in enumerate(self.leaves): Xp[:, q] = lu.solve(WUM @ (lab[M] == L).astype(float) + P["gamma"] * lam[U, L])
            o = np.argsort(-Xp, 1); a1 = Xp[np.arange(len(U)), o[:, 0]]; a2 = Xp[np.arange(len(U)), o[:, 1]]; lv = np.array(self.leaves)
            p1[U] = a1; p2[U] = a2; top[U, 0] = lv[o[:, 0]]; top[U, 1] = lv[o[:, 1]]
            acc = (a1 >= P["p1"]) & (a1 - a2 >= P["margin"]) & cand[U, top[U, 0]]; lab[U[acc]] = top[U[acc], 0]; rnd[U[acc]] = r; counts.append(int(acc.sum()))
            if not acc.any(): break
        return dict(lab=lab, rnd=rnd, p1=p1, p2=p2, top=top, counts=counts, trace_seeded=ts)

    def reasons(self, res, cand):
        lab = res["lab"]; has = np.zeros(self.comp.max() + 1, bool); has[self.comp[lab > 0]] = True; rs = np.zeros(self.nT, int); un = lab[:self.nT] == 0
        rs[un & ~has[self.comp[:self.nT]]] = 11; rs[un & has[self.comp[:self.nT]]] = 10; rs[un & ~cand.any(1)] = 12; return rs


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); Z = dict(np.load(D0 / "kaggle_data/r8pc_bundle.npz")); R8 = np.load(D0 / "R8_pc/labels.npz"); M = Rx(Z, R8)
    res = dict(status="R3-1b_unaccepted", scale="conditional_on_unvalidated_scale", no_photo_data=True, params=P, targets=int(M.nT), trace_seeded_targets=int(M.trace_seed.sum()), blade_nodes=int(len(M.S)),
               targets_with_trace_within_3mm_per_leaf={str(L): int(((M.near_leaf == L) & M.trace_seed).sum()) for L in sorted(M.trace)})
    # self-test: leave one trace out
    test = {}; tot_acc = tot_ok = tot_hid = 0
    for L in sorted(M.trace):
        hid = np.nonzero(M.trace_seed & (M.near_leaf == L))[0]
        if not len(hid): continue
        r = M.run(hide_leaf=L); got = r["lab"][hid]; acc = got > 0; ok = int((got[acc] == L).sum()); na = int(acc.sum()); wrong = Counter(got[acc & (got != L)].tolist())
        test[str(L)] = dict(hidden=int(len(hid)), accepted=na, correct=ok, accuracy=(ok / na if na else None), wrong_rate=((na - ok) / na if na else None), coverage=na / len(hid), wrong_to=dict(wrong)); tot_acc += na; tot_ok += ok; tot_hid += len(hid)
        log(f"leave-out leaf {L}: hidden {len(hid)}, accepted {na}, acc {test[str(L)]['accuracy']}, cov {test[str(L)]['coverage']:.2f}, wrong to {dict(wrong)}")
    pooled = dict(hidden=tot_hid, accepted=tot_acc, correct=tot_ok, accuracy=(tot_ok / tot_acc if tot_acc else None), wrong_rate=((tot_acc - tot_ok) / tot_acc if tot_acc else None), coverage=tot_acc / max(tot_hid, 1))
    gate = bool(pooled["accuracy"] is not None and pooled["accuracy"] >= 0.90 and pooled["wrong_rate"] <= 0.05)
    res["self_test"] = dict(per_leaf=test, pooled=pooled, gate_pass=gate, note="leave-one-trace-out; truth = the user's own traces, so only the non-trace evidence is tested"); res["tag"] = "gate_passed_self_test" if gate else "gate_failed_viewing_only"
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float); log("pooled", pooled, "gate", gate)
    # final run
    r = M.run(); lam, cand = M.prior(); rs = M.reasons(r, cand); n = len(M.X); lab_full = M.lab.copy(); src = np.zeros(n, np.int16); src[M.lab > 0] = 1; src[M.ax == 1] = 20
    labT = r["lab"][:M.nT]; lab_full[M.T] = labT; srcT = np.where(r["rnd"][:M.nT] == -1, 2, np.where(r["rnd"][:M.nT] > 0, 2 + r["rnd"][:M.nT], rs)); src[M.T] = srcT
    p1 = np.zeros(n, np.float32); p2 = np.zeros(n, np.float32); t1 = np.zeros(n, np.int16); t2 = np.zeros(n, np.int16); p1[M.T] = r["p1"][:M.nT]; p2[M.T] = r["p2"][:M.nT]; t1[M.T] = r["top"][:M.nT, 0]; t2[M.T] = r["top"][:M.nT, 1]
    np.savez_compressed(OUT / "labels_rachis.npz", rows=Z["rows"], label=lab_full.astype(np.int16), source=src, source_names=np.array([f"{k}:{v}" for k, v in SRC.items()]), p1=p1, p2=p2, top1=t1, top2=t2, r8_pc=R8["label"].astype(np.int16), axis_kind=M.ax.astype(np.int8), target=np.isin(np.arange(n), M.T))
    ab = labT > 0; res["final"] = dict(rounds=r["counts"], trace_seeded=int((r["rnd"][:M.nT] == -1).sum()), absorbed_by_walker=int((r["rnd"][:M.nT] > 0).sum()), absorbed_total=int(ab.sum()), unpainted=int((~ab).sum()),
                                      reasons={SRC[k]: int((rs[~ab] == k).sum()) for k in (10, 11, 12)})
    per = {}
    for L in M.leaves:
        idx = M.T[labT == L]; um = (~ab) & (r["top"][:M.nT, 0] == L) & (r["p1"][:M.nT] > 0) & (r["p1"][:M.nT] - r["p2"][:M.nT] < P["margin"])
        per[str(L)] = dict(rachis_surfels=int(len(idx)), rachis_mm2=round(PA.area_mm2(Z, M.X, idx), 1), from_trace_seed=int(((r["rnd"][:M.nT] == -1) & (labT == L)).sum()), uncertain_unpainted=int(um.sum()), uncertain_mm2=round(PA.area_mm2(Z, M.X, M.T[um]), 1),
                           blade_mm2_R8_pc=round(PA.area_mm2(Z, M.X, np.nonzero(M.lab == L)[0]), 1))
    res["per_leaf"] = per; ax2 = M.ax == 2; rest = M.T[~ab]
    res["table"] = dict(denominator=f"all {n} reconstructed surfels", R8_pc=dict(axis2_painted=dict(count=int(((M.lab > 0) & ax2).sum()), mm2=round(PA.area_mm2(Z, M.X, np.nonzero((M.lab > 0) & ax2)[0]), 1)),
                                                                              axis2_unpainted=dict(count=int(len(M.T)), mm2=round(PA.area_mm2(Z, M.X, M.T), 1)), axis1=dict(count=int((M.ax == 1).sum()), mm2=round(PA.area_mm2(Z, M.X, np.nonzero(M.ax == 1)[0]), 1))),
                        R8_pc_rachis=dict(axis2_painted_as_blade=int(((M.lab > 0) & ax2).sum()), axis2_absorbed_as_rachis=dict(count=int(ab.sum()), mm2=round(PA.area_mm2(Z, M.X, M.T[ab]), 1)),
                                          axis2_unpainted=dict(count=int(len(rest)), mm2=round(PA.area_mm2(Z, M.X, rest), 1)), axis1=dict(count=int((M.ax == 1).sum()))))
    res["seconds"] = round(time.time() - t0); json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float); log("done", res["seconds"], "s", json.dumps(res["final"]))


if __name__ == "__main__":
    main()
