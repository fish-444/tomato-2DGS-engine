"""P1 T-prox + SAM 2 arms (build/p1_tprox_sam2_20261107_01/CONTRACT.md + A1), Kaggle CPU.
T0 = frozen T-prox bundle through the frozen pc_absorb.Model (set A) + frozen pc_rachis; A = H2-mask groups as one walker node each (derived grouped walker below,
the frozen file is imported unchanged); B = V-P agreement seeds; A+B. No human label is in this kernel."""
import glob, importlib.util, itertools, json, os, sys, time, types
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu

OUT = "/kaggle/working"; T0 = time.time(); R = {}


def find(name):
    hit = glob.glob(f"/kaggle/input/**/{name}", recursive=True); assert hit, name; return hit[0]


def log(*a): print(time.strftime("%H:%M:%S"), f"({time.time() - T0:.0f}s)", *a, flush=True)
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


for p_ in ("tomato3d", "tomato3d.organs"): sys.modules.setdefault(p_, types.ModuleType(p_))
lo = types.ModuleType("tomato3d.organs.light_organs"); lo.ROOT = Path(OUT); sys.modules["tomato3d.organs.light_organs"] = lo
PA = load_module("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load_module("tomato3d.organs.pc_rachis", find("pc_rachis.py"))
Z = dict(np.load(find("r8pc_bundle.npz"))); n = len(Z["rows"]); G_SET, C_SET = PA.SETS["A"]
M = PA.Model(Z); M.norm_unary()
labn = np.nonzero(M.st1n > 0)[0]; d_, jj, v_ = PA.knn(M.tree_n, M.Xn[labn], 32, PA.P["band_mm"]); jj = np.where(v_, jj, 0); band = labn[(v_ & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1)]
auc = M.colour_auc(band); wcol = {k: (PA.WC[k] if (k == "C3" or auc[k]["auc"] >= PA.P["auc_min"]) else 0.0) for k in PA.WC}     # as frozen main(); set A has colour weight 0
R["nodes"] = int(M.m); R["wcol"] = wcol; log("model", M.m)


def run_grouped(seed0, cl):
    """Frozen M.run with every group contracted to one node (cl: cluster id per walker node, groups share an id). Priors of members summed, then normalised."""
    g, c = G_SET, C_SET; en = M.edge_norm; P = PA.P; WG, un = PA.WG, M.unary_norm
    E = g * (WG["G2"] * M.edge_raw["G2"] / en["G2"] + WG["G4"] * M.edge_raw["G4"] / en["G4"] + WG["G5"] * M.edge_raw["G5"] / en["G5"]) + c * wcol["C3"] * M.edge_raw["C3"] / en["C3"]
    w = np.exp(-E); m_ = M.m; K = int(cl.max()) + 1; S = csr_matrix((np.ones(m_), (np.arange(m_), cl)), shape=(m_, K))
    W = coo_matrix((np.r_[w, w], (np.r_[M.ei, M.ej], np.r_[M.ej, M.ei])), shape=(m_, m_)).tocsr(); Wc = (S.T @ W @ S).tocsr(); Wc.setdiag(0); Wc.eliminate_zeros()
    Lap = (diags(np.asarray(Wc.sum(1)).ravel()) - Wc).tocsr()
    labc = np.zeros(K, int); seeded = seed0 > 0; labc[cl[seeded]] = seed0[seeded]; rnd = np.where(labc > 0, 0, -1); p1c = np.where(labc > 0, 1.0, 0.0); p2c = np.zeros(K); topc = np.zeros((K, 2), int); counts = []
    for r in range(1, P["rounds"] + 1):
        lab = labc[cl]; Uc = np.nonzero(labc == 0)[0]
        if not len(Uc): break
        U_ = np.nonzero(labc[cl] == 0)[0]; prior_n = np.zeros((len(U_), len(M.leaves)))
        for q, L in enumerate(M.leaves):
            sL = np.nonzero(lab == L)[0]
            if not len(sL): continue
            dist = dijkstra(M.Glen, directed=False, indices=sL, min_only=True)[U_]; cand = dist <= M.U
            if not cand.any(): continue
            t = U_[cand]; f = M.local_feats(sL, t); g6 = M.rachis_cost(L, sL, t)
            nz = lambda k: np.where(np.isfinite(f[k]), f[k] / un[k], P["missing"])  # noqa: E731
            Eu = g * (WG["G1"] * dist[cand] / M.U + WG["G3"] * nz("G3") + WG["G6"] * g6) + c * (wcol["C1"] * nz("C1") + wcol["C2"] * nz("C2") + wcol["C4"] * nz("C4"))
            prior_n[cand, q] = np.exp(-Eu)
        pos = -np.ones(K, int); pos[Uc] = np.arange(len(Uc)); prior = np.zeros((len(Uc), len(M.leaves))); np.add.at(prior, pos[cl[U_]], prior_n)
        s = prior.sum(1, keepdims=True); prior = np.where(s > 0, prior / np.maximum(s, 1e-300), 0)
        Mc = np.nonzero(labc > 0)[0]; A = (Lap[Uc][:, Uc] + P["gamma"] * identity(len(Uc))).tocsc(); lu = splu(A); WUM = Wc[Uc][:, Mc]; X_ = np.zeros((len(Uc), len(M.leaves)))
        for q, L in enumerate(M.leaves): X_[:, q] = lu.solve(WUM @ (labc[Mc] == L).astype(float) + P["gamma"] * prior[:, q])
        o = np.argsort(-X_, 1); a1 = X_[np.arange(len(Uc)), o[:, 0]]; a2 = X_[np.arange(len(Uc)), o[:, 1]] if X_.shape[1] > 1 else np.zeros(len(Uc))
        p1c[Uc] = a1; p2c[Uc] = a2; topc[Uc, 0] = np.array(M.leaves)[o[:, 0]]; topc[Uc, 1] = np.array(M.leaves)[o[:, 1]] if X_.shape[1] > 1 else 0
        acc = (a1 >= P["p1"]) & (a1 - a2 >= P["margin"]); labc[Uc[acc]] = topc[Uc[acc], 0]; rnd[Uc[acc]] = r; counts.append(int(acc.sum()))
        if not acc.any(): break
    return dict(lab=labc[cl], rnd=rnd[cl], p1=p1c[cl], p2=p2c[cl], top=topc[cl], counts=counts)


def finish(tag, r, seed_node, extra):
    """Full labels + frozen rachis absorption; saves labels_<tag>.npz (label = rachis-applied, as the frozen route's output)."""
    f = np.zeros(n, int); f[M.nid] = r["lab"]; rx = PR.Rx(Z, dict(label=f)); rr = rx.run(); lab_full = f.copy(); lab_full[rx.T] = rr["lab"][:rx.nT]
    p1 = np.zeros(n, np.float32); p2 = np.zeros(n, np.float32); p1[M.nid] = r["p1"]; p2[M.nid] = r["p2"]; st = np.zeros(n, np.int16); st[M.nid] = seed_node
    np.savez_compressed(f"{OUT}/labels_{tag}.npz", rows=Z["rows"], label=lab_full.astype(np.int16), r8_pc=f.astype(np.int16), p1=p1, p2=p2, stage1_used=st)
    R.setdefault("runs", {})[tag] = dict(rounds=r["counts"], r8_labelled=int((f > 0).sum()), final_labelled=int((lab_full > 0).sum()), **extra); save(); log(tag, R["runs"][tag])
    return f


def groups_cl(share_tag):
    gz = np.load(find(f"groups_{share_tag}.npz")); assert np.array_equal(gz["rows"], Z["rows"]); gid = gz["gid"][M.nid]; K0 = int(gid.max()) + 1
    cl = np.where(gid >= 0, gid, K0 + np.arange(M.m)); _, cl = np.unique(cl, return_inverse=True)
    return cl, dict(grouped_nodes=int((gid >= 0).sum()), groups=K0)


def b_seeds(first, n_min, a_min, cl=None):
    vz = np.load(find("vp_votes.npz")); assert np.array_equal(vz["rows"], Z["rows"]); V = vz["votes"][M.nid].astype(int); tot = V.sum(1); best = V.argmax(1); agree = np.where(tot > 0, V.max(1) / np.maximum(tot, 1), 0)
    sel = (M.st1n == 0) & (tot >= n_min) & (agree >= a_min) & (best == first["lab"]) & (first["p1"] >= 0.6) & (first["lab"] > 0)
    seed = M.st1n.copy(); seed[sel] = best[sel]; dropped = 0
    if cl is not None:                                                     # a group may not carry seeds of two leaves: drop the B seeds of such a group
        for c in np.unique(cl[seed > 0]):
            mem = cl == c
            if len(set(seed[mem & (seed > 0)].tolist())) > 1: drop = mem & sel; dropped += int(drop.sum()); seed[drop] = 0; sel &= ~drop
    return seed, dict(added_seeds=int(sel.sum()), dropped_in_mixed_groups=dropped)


try:
    r0 = M.run(G_SET, C_SET, wcol, M.st1n.copy()); f0 = finish("T0", r0, M.st1n.copy(), {})
    fz = np.load(find("t0_frozen_r8pc.npz")); R["T0_match_frozen_r8pc"] = round(float((fz["label"] == f0).mean()), 6); save(); log("T0 match", R["T0_match_frozen_r8pc"])
    firsts = {}
    for tag in ("s40", "s50", "s60"):
        cl, gi = groups_cl(tag); rA = run_grouped(M.st1n.copy(), cl); finish(f"A_{tag}", rA, M.st1n.copy(), gi); firsts[tag] = (rA, cl)
    for n_min, a_min in itertools.product((2, 3, 4), (0.7, 0.8, 0.9)):
        seed, bi = b_seeds(r0, n_min, a_min); rB = M.run(G_SET, C_SET, wcol, seed.copy()); finish(f"B_n{n_min}_a{int(a_min * 100)}", rB, seed, bi)
        for tag, (rA, cl) in firsts.items():
            seed2, bi2 = b_seeds(rA, n_min, a_min, cl); rAB = run_grouped(seed2.copy(), cl); finish(f"AB_{tag}_n{n_min}_a{int(a_min * 100)}", rAB, seed2, bi2)
    R["status"] = "ok"
except Exception:
    import traceback; R["error"] = traceback.format_exc()[-3000:]
R["seconds"] = round(time.time() - T0); save(); print(R.get("error", "done"))
