"""Multi-cue assignment of 3D-first groups to skeleton leaves (contract build/p1_skeleton_multicue_20261101_01/CONTRACT.md). One declared combination; ablations reported only.

usage: python -m tomato3d.organs.skeleton_multicue --step r31b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import oversegment as OS
from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs import skeleton_r31b_eval as EV

ROOT = OS.ROOT
OUT = ROOT / "build/p1_skeleton_multicue_20261101_01"
C = dict(link_mm=3.0, cross_mm=10.0, limit_mm=80.0, src_mm=3.0, layer_mm=15.0, elong=1.4, ratio=1.15, min_n=10, sam_mm=5.0, sam_min=2, best_max=2.5, gap=0.5, tail_pct=90)
W = dict(surface=1.0, euclid=1.0, layer=1.0, dir=0.5, sam=0.5)
CUES = tuple(W)
TOTAL = sum(W.values())
LABELS = ["user_skeleton_seeds", "multicue_assign", "conditional_on_unvalidated_scale", "reference_only_scoring"]


def graph(xyz, sidx, nsh):
    E = cKDTree(xyz).query_pairs(C["link_mm"] / 1000, output_type="ndarray"); i, j = E[:, 0], E[:, 1]
    L = np.linalg.norm(xyz[i] - xyz[j], axis=1) * 1000; cross = (nsh[i] > 1) & (nsh[j] > 1) & (sidx[i] != sidx[j]); L = L + cross * C["cross_mm"]
    n = len(xyz); return coo_matrix((np.r_[L, L], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()


def leaflet_base_dir(P):
    """Step B: unit vector centroid -> narrower end, or None (too small, not elongated, ambiguous)."""
    if len(P) < C["min_n"]:
        return None
    c = P.mean(0); w, v = np.linalg.eigh(np.cov((P - c).T)); u, vv = v[:, 2], v[:, 1]
    if np.sqrt(max(w[2], 0) / max(w[1], 1e-18)) < C["elong"]:
        return None
    t = (P - c) @ u; a, b = P[t <= np.quantile(t, 0.15)], P[t >= np.quantile(t, 0.85)]
    wa = np.ptp(np.percentile((a - c) @ vv, [5, 95])); wb = np.ptp(np.percentile((b - c) @ vv, [5, 95]))
    if max(wa, wb) / max(min(wa, wb), 1e-9) < C["ratio"]:
        return None
    e = (a if wa < wb else b).mean(0) - c; return e / max(np.linalg.norm(e), 1e-12)


def combine(costs, present, use=CUES):
    """costs: dict cue -> (G, K) normalised cost; present: dict cue -> (G,) bool. Returns C (G, K) with absent cues dropped and the rest scaled to TOTAL."""
    G, K = next(iter(costs.values())).shape; num = np.zeros((G, K)); wsum = np.zeros(G)
    for q in use:
        if q not in costs:
            continue
        p = present[q]; c = costs[q]; num[p] += W[q] * c[p]; wsum[p] += W[q]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (TOTAL / np.maximum(wsum, 1e-12))[:, None] * num
    out[wsum == 0] = np.inf; return out


def decide(Cm, high_tail):
    o = np.argsort(Cm, 1); b = Cm[np.arange(len(Cm)), o[:, 0]]; s = Cm[np.arange(len(Cm)), o[:, 1]] if Cm.shape[1] > 1 else np.full(len(Cm), np.inf)
    gap = np.where(high_tail, 2 * C["gap"], C["gap"]); ok = np.isfinite(b) & (b <= C["best_max"]) & (s - b >= gap); return np.where(ok, o[:, 0] + 1, 0)


def cues_r31b():
    Z = np.load(ROOT / "build/p1_pc_absorb_20261027_01/kaggle_data/r8pc_bundle.npz"); xyz = Z["xyz"]; n = len(xyz); rows = Z["rows"]
    sk = SIO.load(SIO.skeleton_dir(ROOT, "R3-1b", "2026-10-21") / "skeleton_v1.json"); back = [t for t in SIO.to_traces(sk) if t["leaf"] > 0 and len(t["pts"]) >= 2]
    nums = sorted({t["leaf"] for t in back}); K = len(nums); polys = [np.vstack([SIO.resample(np.asarray(t["pts"], float)) for t in back if t["leaf"] == L]) for L in nums]
    pf = np.load(ROOT / "build/p1_stacked_sheets_retry_20261026_01/oversegment/second_attempt/patches_filled.npz"); pos = {r: i for i, r in enumerate(rows)}
    group = np.full(n, -1); group[[pos[r] for r in pf["rows"]]] = pf["patch_filled"]
    sidx, nsh = Z["sheet_index"].astype(int), Z["n_sheets"].astype(int)
    D = np.stack([cKDTree(p).query(xyz)[0] for p in polys], 1) * 1000.0                                   # per surfel straight distance to each leaf (mm)
    src = [np.nonzero(D[:, k] <= C["src_mm"])[0] for k in range(K)]
    Gm = graph(xyz, sidx, nsh); DS = np.full((n, K), np.inf)
    for k in range(K):
        if len(src[k]):
            DS[:, k] = dijkstra(Gm, indices=src[k], min_only=True, limit=C["limit_mm"])
    sh = np.load(ROOT / "build/p1_thin_shell_20261026_01/metrics/R3-1b_per_surfel.npz"); assert np.array_equal(sh["rows"], rows); beyond = sh["beyond"].astype(float); p90 = float(np.percentile(beyond[Z["axis_kind"] == 0], C["tail_pct"]))
    V = np.load(ROOT / "build/p1_stacked_sheets_retry_20261026_01/oversegment/second_attempt/votes_4b.npz"); same = {}; diff = {}
    for (a, b), s_, d_ in zip(V["pairs"], V["same"], V["different"]):
        for x, y in ((a, b), (b, a)):
            same[(int(x), int(y))] = int(s_); diff[(int(x), int(y))] = int(d_)
    gids = np.unique(group[group >= 0]); G = len(gids); idx = {g: np.nonzero(group == g)[0] for g in gids}
    near = [set(int(g) for g in np.unique(group[(D[:, k] <= C["sam_mm"]) & (group >= 0)])) for k in range(K)]
    costs = {q: np.zeros((G, K)) for q in CUES}; present = {q: np.zeros(G, bool) for q in CUES}; dirabs = np.zeros((G, K), bool); high = np.zeros(G, bool)
    for gi, g in enumerate(gids):
        I = idx[g]; P = xyz[I]
        costs["euclid"][gi] = np.median(D[I], 0) / 25.0; present["euclid"][gi] = True
        ds = DS[I].min(0)
        if np.isfinite(ds).any():
            costs["surface"][gi] = ds / 10.0; present["surface"][gi] = True                                 # infinite for unreachable leaves -> C infinite there
        gm = np.bincount(sidx[I]).argmax(); lay = np.zeros(K); okl = False
        for k in range(K):
            q = I[np.argmin(D[I, k])]; S = src[k][np.linalg.norm(xyz[src[k]] - xyz[q], axis=1) <= C["layer_mm"] / 1000] if len(src[k]) else []
            if len(S):
                lay[k] = 0.0 if np.bincount(sidx[S]).argmax() == gm else 1.0; okl = True
            else:
                lay[k] = np.nan
        if okl:
            present["layer"][gi] = True; costs["layer"][gi] = np.nan_to_num(lay, nan=1.0)
        e = leaflet_base_dir(P)
        if e is not None:
            c0 = P.mean(0)
            for k in range(K):
                v = polys[k][np.argmin(np.linalg.norm(polys[k] - c0, axis=1))] - c0; v /= max(np.linalg.norm(v), 1e-12)
                costs["dir"][gi, k] = np.degrees(np.arccos(np.clip(e @ v, -1, 1))) / 30.0
            present["dir"][gi] = True
        tot_s = np.zeros(K); tot = np.zeros(K)
        for k in range(K):
            for h in near[k] - {int(g)}:
                if (int(g), h) in same:
                    tot_s[k] += same[(int(g), h)]; tot[k] += same[(int(g), h)] + diff[(int(g), h)]
        if (tot >= C["sam_min"]).all():
            costs["sam"][gi] = 1 - tot_s / np.maximum(tot, 1); present["sam"][gi] = True
        elif (tot >= C["sam_min"]).any():                                                              # votes for some leaves only: the others count as no shared mask evidence (cost 1)
            costs["sam"][gi] = np.where(tot >= C["sam_min"], 1 - tot_s / np.maximum(tot, 1), 1.0); present["sam"][gi] = True
        high[gi] = np.median(beyond[I]) > p90
    un = np.nonzero(group < 0)[0]; ucost = {"euclid": D[un] / 25.0, "surface": DS[un] / 10.0}; upres = {"euclid": np.ones(len(un), bool), "surface": np.isfinite(DS[un]).any(1)}
    ulay = np.zeros((len(un), K)); upl = np.zeros(len(un), bool)
    for k in range(K):
        if not len(src[k]):
            continue
        tk = cKDTree(xyz[src[k]]); nb = tk.query_ball_point(xyz[un], C["layer_mm"] / 1000)
        for r, js in enumerate(nb):
            if js:
                ulay[r, k] = 0.0 if np.bincount(sidx[src[k][js]]).argmax() == sidx[un[r]] else 1.0; upl[r] = True
            else:
                ulay[r, k] = 1.0
    ucost["layer"] = ulay; upres["layer"] = upl
    return dict(rows=rows, gids=gids, idx=idx, un=un, nums=nums, costs=costs, present=present, high=high, ucost=ucost, upres=upres, uhigh=beyond[un] > p90, p90=p90, n=n)


def labels_from(X, use):
    n = X["n"]; lab = np.zeros(n, int); nums = np.array([0] + X["nums"])
    gl = decide(combine(X["costs"], X["present"], use), X["high"])
    for g, l_ in zip(X["gids"], gl):
        lab[X["idx"][g]] = nums[l_]
    uuse = [q for q in use if q in ("surface", "euclid", "layer")]
    if uuse:
        ul = decide(combine(X["ucost"], X["upres"], uuse), X["uhigh"]); lab[X["un"]] = nums[ul]
    return lab


def step_r31b():
    X = cues_r31b(); Bs = OS.load_base(); si, hl = EV.human_projections(Bs); res = dict(labels=LABELS, params=C, weights=W, cue_presence={q: int(X["present"][q].sum()) for q in CUES}, groups=int(len(X["gids"])), p90_tail=X["p90"], runs={})
    variants = {"multicue": CUES}
    variants.update({f"only_{q}": (q,) for q in CUES}); variants.update({f"without_{q}": tuple(c for c in CUES if c != q) for q in CUES})
    labs = {}
    for name, use in variants.items():
        lab = labels_from(X, use); labs[name] = lab; s = EV.score(lab, X["rows"], Bs, si, hl)
        res["runs"][name] = dict(cues=list(use), report_half=s["report_half"], all_leaves=s["all_leaves"], per_leaf={k: v["accuracy"] for k, v in s["per_leaf"].items()}, confusion=s["confusion"] if name == "multicue" else None)
        print(name, s["report_half"]["accuracy"], s["report_half"]["contested_or_none_share"], flush=True)
    m = res["runs"]["multicue"]["report_half"]; res["gate"] = dict(baseline_distance=0.6065, need=0.6365, contested_baseline=0.2524, contested_max=0.3024, accuracy=m["accuracy"], contested=m["contested_or_none_share"],
                                                                    passed=bool(m["accuracy"] >= 0.6365 and m["contested_or_none_share"] <= 0.3024))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "cues_R3-1b.npz", gids=X["gids"], nums=np.array(X["nums"]), high=X["high"], **{f"cost_{q}": X["costs"][q] for q in CUES}, **{f"present_{q}": X["present"][q] for q in CUES})
    np.savez_compressed(OUT / "labels_multicue_R3-1b.npz", rows=X["rows"], label=labs["multicue"].astype(np.int16))
    json.dump(res, open(OUT / "r31b_multicue.json", "w", encoding="utf-8"), indent=1, default=float); figure(X, labs["multicue"], Bs, si, hl); print("gate", res["gate"]); return res


def figure(X, lab_m, Bs, si, hl):
    """Top view: projections where multicue and the distance route disagree, coloured by which one matched the human leaf."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sk = SIO.load(SIO.skeleton_dir(ROOT, "R3-1b", "2026-10-21") / "skeleton_v1.json"); full, rows, _ = EV.route_distance_r31b(Bs, sk)
    pos = {int(r): i for i, r in enumerate(X["rows"])}; brow = Bs["pl"]["rows"][Bs["b"]]; bi = np.array([pos[int(r)] for r in brow])
    m = lab_m[bi][si]; d = full[bi][si]; xyz = Bs["xyz"][si] * 1000; dis = m != d
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(xyz[:, 0], xyz[:, 1], s=1, c="#dddddd")
    for sel, col, name in ((dis & (m == hl), "#1a9850", "multicue right"), (dis & (d == hl), "#d73027", "distance right"), (dis & (m != hl) & (d != hl), "#4575b4", "both wrong")):
        ax.scatter(xyz[sel, 0], xyz[sel, 1], s=2, c=col, label=f"{name} ({int(sel.sum())})")
    ax.set_aspect("equal"); ax.legend(markerscale=6); ax.set_title("R3-1b: multicue vs distance where they disagree (top view, mm)"); fig.savefig(OUT / "r31b_disagreement.png", dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", default="r31b"); a = ap.parse_args(argv); step_r31b()


if __name__ == "__main__":
    main()
