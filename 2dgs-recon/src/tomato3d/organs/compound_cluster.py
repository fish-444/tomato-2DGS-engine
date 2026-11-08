"""Compound-leaf colouring by geometry (contract build/p1_compound_cluster_20261025_01/CONTRACT.md).

Blade points are clustered into compound leaves from geometry alone - cylindrical position about a stem reference axis,
height above the pot top, sign-free normal, local density. Not used anywhere: the organs skeleton / tracking / node chain,
pipeline.run, the MTG grouping, leaflet-instance labels (compared against only after clustering). Candidates K1-K4 are
fixed on the synthetic DEV bench, the first passing the DEV gate runs once on TEST and then on R3-1b.

usage: python -m tomato3d.organs.compound_cluster --step dev|test|r31b|viewer|all
Every output: R3-1b_unaccepted, conditional_on_unvalidated_scale, grouping geometry_clusters; rank / axis / attachment unavailable.
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import light_organs as LO
from tomato3d.organs.light_viewer import TEMPLATE, _b64, palette, read_csv

ROOT = LO.ROOT
OUT = ROOT / "build/p1_compound_cluster_20261025_01"
BENCH = ROOT / "build/p1_organ_topology_20261024_01/bench"
TRACK_A = LO.DEFAULTS["out"]
SEED_RUN = ROOT / "build/p1_leaflet_2dseed_20261024_01"
STATUS = "R3-1b_unaccepted"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="geometry_clusters", unavailable="rank, axis membership, attachment")
POT_TOP = 0.066; POT_CENTRE = np.array([0.1707, 0.1452])
# declared parameters (contract sections 3-6)
AXIS = dict(iters=300, max_tilt_deg=25.0, inlier_mm=2.0, min_inliers=200)
DENS_MM, SEED_MM, KNN, EDGE_MM = 1.5, 3.0, 10, 3.0
K1 = dict(min_blob=30, gap_mm=4.0, contact_mm=2.0, tip_share=0.10, jump_mm=3.0, grid=dict(sector_deg=(25, 35, 45), normal_deg=(30, 45), h_resid_mm=(2, 3, 4)))
K2 = dict(min_samples=10, min_frac=0.02, min_abs=50, w_incl=0.3, w_naz=0.1, w_dens=2.0, grid=dict(w_h=(1, 2)))
K3 = dict(k_range=(3, 12), iters=20, sigma_h_mm=5.0, cap=1.0, grid=dict(sector_deg=(25, 35, 45)))
GATE = dict(membership_min=0.85, count_mae_max=1.0, merged_max=0.15)
PLAUS = dict(compound_leaves=(5, 9), blobs_per_cluster=(4, 8), straddle_share=0.8)


# ---------------------------------------------------------------- inputs
def load_bench(plant_dir: Path):
    z = np.load(plant_dir / "cloud.npz"); fam = z["family"]
    return dict(xyz=z["xyz"].astype(np.float64), normal=z["normal"].astype(np.float64), is_axis=z["axis_label"] == 1, is_blade=z["axis_label"] == 0,
                truth=z["leaf"], eval_mask=(z["axis_label"] == 0) & (fam != "cotyledon"), name=plant_dir.name)


def load_r31b():
    sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); ta = np.load(TRACK_A / "surfel_labels.npz"); rows = ta["rows"]
    return dict(xyz=sf["xyz"][rows].astype(np.float64), normal=sf["normal"][rows].astype(np.float64), is_axis=ta["is_axis"].astype(bool), is_blade=~ta["is_axis"].astype(bool),
                truth=None, eval_mask=~ta["is_axis"].astype(bool), name="R3-1b", rows=rows, n_ply=len(sf["xyz"]))


# ---------------------------------------------------------------- step 1: features
def stem_axis(xyz_axis, rng=np.random.default_rng(0)):
    """RANSAC vertical-ish line through the axis points; else the vertical through the pot centre."""
    n = len(xyz_axis); best = (0, None)
    if n >= 2:
        for _ in range(AXIS["iters"]):
            i, j = rng.choice(n, 2, replace=False); d = xyz_axis[j] - xyz_axis[i]; L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            d = d / L
            if np.degrees(np.arccos(abs(d[2]))) > AXIS["max_tilt_deg"]:
                continue
            v = xyz_axis - xyz_axis[i]; dist = np.linalg.norm(v - np.outer(v @ d, d), axis=1); k = int((dist <= AXIS["inlier_mm"] / 1000).sum())
            if k > best[0]:
                best = (k, (xyz_axis[i].copy(), d * np.sign(d[2])))
    if best[0] >= AXIS["min_inliers"]:
        p0, d = best[1]; return p0, d, dict(used="ransac_axis_points", inliers=int(best[0]))
    return np.array([POT_CENTRE[0], POT_CENTRE[1], POT_TOP]), np.array([0, 0, 1.0]), dict(used="vertical_through_pot_centre", inliers=int(best[0]))


def features(P, xyz_b, nrm_b):
    p0, d, info = stem_axis(P["xyz"][P["is_axis"]])
    v = xyz_b - p0; t = v @ d; foot = p0 + np.outer(t, d); rad = xyz_b - foot
    e1 = np.cross(d, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    theta = np.degrees(np.arctan2(rad @ e2, rad @ e1)) % 360; r = np.linalg.norm(rad, axis=1); h = xyz_b[:, 2] - POT_TOP
    nn = nrm_b / np.maximum(np.linalg.norm(nrm_b, axis=1, keepdims=True), 1e-12)
    incl = np.degrees(np.arccos(np.clip(np.abs(nn[:, 2]), 0, 1))); naz = np.degrees(np.arctan2(nn[:, 1], nn[:, 0])) % 180
    tree = cKDTree(xyz_b); cnt = np.array([len(x) - 1 for x in tree.query_ball_point(xyz_b, DENS_MM / 1000)], float); dens = cnt / max(np.median(cnt), 1)
    at = cKDTree(P["xyz"][P["is_axis"]]) if P["is_axis"].any() else None; dist_axis = at.query(xyz_b)[0] if at is not None else np.full(len(xyz_b), np.nan)
    return dict(theta=theta, r=r, h=h, incl=incl, naz=naz, dens=dens, nn=nn, tree=tree, dist_axis=dist_axis, axis=info, r_mean=float(np.mean(r)))


# ---------------------------------------------------------------- helpers
def circ_mean_deg(a, period=360.0):
    x = np.radians(a * 360.0 / period); return float(np.degrees(np.arctan2(np.sin(x).mean(), np.cos(x).mean())) % 360) * period / 360.0


def circ_diff_deg(a, b, period=360.0):
    return float(abs((a - b + period / 2) % period - period / 2))


def mean_normal(nn):
    w, v = np.linalg.eigh(nn.T @ nn); return v[:, -1]


# ---------------------------------------------------------------- K1: blobs and merge
def knn_graph(xyz_b):
    k = min(KNN + 1, len(xyz_b)); dd, ii = cKDTree(xyz_b).query(xyz_b, k=k); rows, cols = [], []
    for j in range(1, k):
        m = dd[:, j] <= EDGE_MM / 1000; rows.append(np.nonzero(m)[0]); cols.append(ii[m, j])
    rows, cols = np.concatenate(rows), np.concatenate(cols); adj = defaultdict(list)
    for a, b in zip(rows, cols):
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    return adj


def blobs_k1(xyz_b, dens, F):
    """Density-valley over-segmentation: seeds = local density maxima within 3 mm, descending flood over the kNN graph."""
    adj = knn_graph(xyz_b); n = len(xyz_b); lab = np.full(n, -1)
    near3 = F["tree"].query_ball_point(xyz_b, SEED_MM / 1000)
    order = np.argsort(-dens, kind="stable"); nb = 0
    for i in order:
        best = -1; bd = -1
        for j in adj.get(i, []):
            if lab[j] >= 0 and dens[j] > bd:
                bd = dens[j]; best = lab[j]
        if best >= 0:
            lab[i] = best
        elif dens[i] >= dens[near3[i]].max():
            lab[i] = nb; nb += 1
    # small blobs join the neighbour with the most shared edges
    sizes = Counter(lab[lab >= 0].tolist())
    for b, s in list(sizes.items()):
        if s >= K1["min_blob"]:
            continue
        pts = np.nonzero(lab == b)[0]; cnt = Counter(int(lab[j]) for i in pts for j in adj.get(i, []) if lab[j] >= 0 and lab[j] != b and sizes.get(int(lab[j]), 0) >= K1["min_blob"])
        lab[pts] = cnt.most_common(1)[0][0] if cnt else -1
    keep = sorted(set(lab[lab >= 0].tolist())); ren = {b: q for q, b in enumerate(keep)}
    return np.array([ren.get(int(x), -1) for x in lab]), adj


def merge_k1(xyz_b, F, blob, adj, sector_deg, normal_deg, h_resid_mm, rng=np.random.default_rng(0)):
    nb = blob.max() + 1
    if nb == 0:
        return np.full(len(xyz_b), -1), dict(blobs=0, merges=0)
    members = {b: np.nonzero(blob == b)[0] for b in range(nb)}
    # contact between blobs: shared kNN edges and points within contact_mm
    contact = Counter()
    for i, js in adj.items():
        bi = blob[i]
        if bi < 0: continue
        for j in js:
            bj = blob[j]
            if bj >= 0 and bj != bi and bi < bj:
                contact[(int(bi), int(bj))] += 1
    trees = {b: cKDTree(xyz_b[m]) for b, m in members.items()}
    # blob-graph edges (contract: nearest points <= gap_mm), found through the blade-point tree; contact = shared kNN edges (may be 0)
    pairs = set(contact)
    for i, js in enumerate(F["tree"].query_ball_point(xyz_b, K1["gap_mm"] / 1000)):
        bi = blob[i]
        if bi < 0: continue
        for j in js:
            bj = blob[j]
            if bj >= 0 and bj != bi:
                pairs.add((int(min(bi, bj)), int(max(bi, bj))))
    edges = {}
    for (a, b) in pairs:
        c = contact.get((a, b), 0)
        da = trees[b].query(xyz_b[members[a]])[0] <= K1["contact_mm"] / 1000; db = trees[a].query(xyz_b[members[b]])[0] <= K1["contact_mm"] / 1000
        ha = F["h"][members[a]][da].mean() if da.any() else F["h"][members[a]].mean(); hb = F["h"][members[b]][db].mean() if db.any() else F["h"][members[b]].mean()
        edges[(a, b)] = dict(contact=int(c), ca=int(da.sum()), cb=int(db.sum()), dh=float(abs(ha - hb)))
    parent = list(range(nb))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def cluster_pts(root):
        return np.concatenate([members[b] for b in range(nb) if find(b) == root])
    def stats(pts):
        sel = pts if len(pts) <= 2000 else rng.choice(pts, 2000, replace=False)
        return dict(theta=circ_mean_deg(F["theta"][sel]), nrm=mean_normal(F["nn"][sel]), n=len(pts))
    def rule_ok(pa, pb, e, ra, rb):
        sa, sb = stats(pa), stats(pb)
        if circ_diff_deg(sa["theta"], sb["theta"]) > sector_deg: return False, "sector"
        if np.degrees(np.arccos(min(1.0, abs(float(sa["nrm"] @ sb["nrm"]))))) > normal_deg: return False, "normal"
        u = np.concatenate([pa, pb]); sel = u if len(u) <= 3000 else rng.choice(u, 3000, replace=False)
        A = np.c_[F["r"][sel], np.ones(len(sel))]; coef, *_ = np.linalg.lstsq(A, F["h"][sel], rcond=None); res = F["h"][sel] - A @ coef
        if np.sqrt((res ** 2).mean()) * 1000 > h_resid_mm: return False, "height"
        small = min(len(pa), len(pb)); touch = min(e["ca"], e["cb"])
        if touch < K1["tip_share"] * small and e["dh"] * 1000 > K1["jump_mm"]: return False, "tip_jump"
        return True, "ok"
    merges = 0; rejected = Counter(); changed = True; tried = set()
    while changed:
        changed = False
        # cluster-level edges: sum of blob edges between the two current clusters
        ce = defaultdict(lambda: dict(contact=0, ca=0, cb=0, dh=[]))
        for (a, b), e in edges.items():
            ra, rb = find(a), find(b)
            if ra == rb: continue
            key = (min(ra, rb), max(ra, rb)); d = ce[key]; d["contact"] += e["contact"]; d["ca"] += e["ca"]; d["cb"] += e["cb"]; d["dh"].append(e["dh"])
        for key, d in sorted(ce.items(), key=lambda kv: -kv[1]["contact"]):
            sig = (key, parent.count(key[0]), parent.count(key[1]))
            if sig in tried: continue
            tried.add(sig); e = dict(d, dh=float(np.mean(d["dh"])))
            pa, pb = cluster_pts(key[0]), cluster_pts(key[1]); ok, why = rule_ok(pa, pb, e, key[0], key[1])
            if ok:
                parent[find(key[1])] = find(key[0]); merges += 1; changed = True; break
            rejected[why] += 1
    roots = sorted({find(b) for b in range(nb)}); ren = {r: q for q, r in enumerate(roots)}
    cl = np.full(len(xyz_b), -1)
    for b in range(nb):
        cl[members[b]] = ren[find(b)]
    return cl, dict(blobs=int(nb), merges=merges, rejected=dict(rejected))


# ---------------------------------------------------------------- K2: HDBSCAN
def run_k2(F, n_blade, w_h):
    from sklearn.cluster import HDBSCAN
    X = np.c_[np.radians(F["theta"]) * F["r_mean"] * 1000, F["r"] * 1000, w_h * F["h"] * 1000, K2["w_incl"] * F["incl"], K2["w_naz"] * F["naz"], K2["w_dens"] * F["dens"]]
    mcs = max(K2["min_abs"], int(K2["min_frac"] * n_blade)); lab = HDBSCAN(min_cluster_size=mcs, min_samples=K2["min_samples"], cluster_selection_method="eom").fit_predict(X)
    return lab.astype(int), dict(min_cluster_size=mcs)


# ---------------------------------------------------------------- K3: template EM with BIC over K
def _fit_template(F, idx):
    th = circ_mean_deg(F["theta"][idx]); A = np.c_[np.ones(len(idx)), F["r"][idx] * 1000, (F["r"][idx] * 1000) ** 2]
    coef, *_ = np.linalg.lstsq(A, F["h"][idx] * 1000, rcond=None); return dict(theta=th, coef=coef)


def _cost(F, T, sector_deg):
    du = np.array([circ_diff_deg(t, T["theta"]) for t in F["theta"]]) if False else np.abs((F["theta"] - T["theta"] + 180) % 360 - 180)
    r_mm = F["r"] * 1000; hpred = T["coef"][0] + T["coef"][1] * r_mm + T["coef"][2] * r_mm ** 2
    return (du / sector_deg) ** 2 + ((F["h"] * 1000 - hpred) / K3["sigma_h_mm"]) ** 2


def run_k3(F, xyz_b, seeds, sector_deg):
    """seeds: list of index arrays (K1 blobs, largest first). Returns labels, info; K chosen by BIC over k_range."""
    n = len(xyz_b); best = None; table = []
    for K in range(K3["k_range"][0], K3["k_range"][1] + 1):
        if K > len(seeds): break
        T = [_fit_template(F, s) for s in seeds[:K]]
        for _ in range(K3["iters"]):
            C = np.stack([_cost(F, t, sector_deg) for t in T]); lab = C.argmin(0); cmin = C.min(0); lab[cmin > K3["cap"]] = -1
            newT = []
            for k in range(K):
                idx = np.nonzero(lab == k)[0]
                newT.append(_fit_template(F, idx) if len(idx) >= 10 else T[k])
            T = newT
        C = np.stack([_cost(F, t, sector_deg) for t in T]); lab = C.argmin(0); cmin = C.min(0); lab[cmin > K3["cap"]] = -1
        rss = float(cmin[lab >= 0].sum() + (lab < 0).sum() * K3["cap"]); bic = n * np.log(max(rss / n, 1e-9)) + K * 4 * np.log(n)
        table.append(dict(K=K, bic=float(bic), unassigned=int((lab < 0).sum())))
        if best is None or bic < best[0]:
            best = (bic, K, lab.copy(), T)
    if best is None:
        return np.full(n, -1), dict(K=0, bic_table=table), []
    return best[2], dict(K=best[1], bic_table=table), best[3]


def run_k4(F, blob, T, sector_deg):
    cl = np.full(len(blob), -1)
    for b in range(blob.max() + 1):
        idx = np.nonzero(blob == b)[0]; Fb = {k: F[k][idx] for k in ("theta", "r", "h")}
        cen = dict(theta=np.array([circ_mean_deg(Fb["theta"])]), r=np.array([Fb["r"].mean()]), h=np.array([Fb["h"].mean()]))
        costs = [float(_cost(cen, t, sector_deg)[0]) for t in T]; k = int(np.argmin(costs))
        if costs[k] <= K3["cap"]:
            cl[idx] = k
    return cl


# ---------------------------------------------------------------- metrics
def metrics(pred, truth, mask):
    p = pred[mask]; t = truth[mask]; n = len(p); leaves = sorted(set(t.tolist())); cls = sorted(set(p[p >= 0].tolist()))
    if not leaves:
        return None
    M = np.zeros((len(cls), len(leaves)))
    li = {l: j for j, l in enumerate(leaves)}; ci = {c: i for i, c in enumerate(cls)}
    for a, b in zip(p, t):
        if a >= 0: M[ci[a], li[b]] += 1
    if len(cls):
        r, c = linear_sum_assignment(-M); member = float(M[r, c].sum() / n)
    else:
        member = 0.0
    from sklearn.metrics import adjusted_rand_score
    ari = float(adjusted_rand_score(t, p))
    split = sum(1 for j in range(len(leaves)) if (M[:, j] >= 0.2 * M[:, j].sum()).sum() >= 2) / len(leaves) if len(cls) else 0.0
    merged = sum(1 for i in range(len(cls)) if (M[i] >= 0.2 * M[i].sum()).sum() >= 2) / max(len(cls), 1)
    return dict(n_points=int(n), n_true=len(leaves), n_pred=len(cls), count_err=len(cls) - len(leaves), count_exact=int(len(cls) == len(leaves)), membership=member, ari=ari,
                split_share=float(split), merged_share=float(merged), unassigned_share=float((p < 0).mean()))


def pooled(rows):
    w = np.array([r["n_points"] for r in rows], float); f = lambda k: float(np.sum([r[k] * r["n_points"] for r in rows]) / w.sum())
    return dict(plants=len(rows), membership=f("membership"), ari=f("ari"), split_share=f("split_share"), merged_share=f("merged_share"), unassigned_share=f("unassigned_share"),
                count_mae=float(np.mean([abs(r["count_err"]) for r in rows])), count_exact=float(np.mean([r["count_exact"] for r in rows])))


def passes(pl):
    return pl["membership"] >= GATE["membership_min"] and pl["count_mae"] <= GATE["count_mae_max"] and pl["merged_share"] <= GATE["merged_max"]


# ---------------------------------------------------------------- one plant, all candidates for a grid point set
def cluster_plant(P, cand: str, params: dict, cache: dict):
    xyz_b = P["xyz"][P["is_blade"]]; nrm_b = P["normal"][P["is_blade"]]
    if "F" not in cache:
        cache["F"] = features(P, xyz_b, nrm_b)
        cache["blob"], cache["adj"] = blobs_k1(xyz_b, cache["F"]["dens"], cache["F"])
    F, blob, adj = cache["F"], cache["blob"], cache["adj"]
    if cand == "K1":
        cl, info = merge_k1(xyz_b, F, blob, adj, params["sector_deg"], params["normal_deg"], params["h_resid_mm"])
    elif cand == "K2":
        cl, info = run_k2(F, len(xyz_b), params["w_h"])
    elif cand in ("K3", "K4"):
        sizes = Counter(blob[blob >= 0].tolist()); seeds = [np.nonzero(blob == b)[0] for b, _ in sizes.most_common()]
        cl3, info, T = run_k3(F, xyz_b, seeds, params["sector_deg"])
        cl = cl3 if cand == "K3" else run_k4(F, blob, T, params["sector_deg"])
        info = dict(info, blobs=int(blob.max() + 1))
    full = np.full(len(P["xyz"]), -1); full[np.nonzero(P["is_blade"])[0]] = cl
    return full, dict(info, axis=F["axis"])


def grid_points(cand):
    import itertools
    g = {"K1": K1["grid"], "K2": K2["grid"], "K3": K3["grid"], "K4": K3["grid"]}[cand]
    keys = list(g); return [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])]


def run_split(split: str, cands, grids: dict | None, out: Path):
    plants = [load_bench(Path(p)) for p in sorted(glob.glob(str(BENCH / split / "*")))]
    caches = {p["name"]: {} for p in plants}; results = {}
    for cand in cands:
        pts = grids[cand] if grids else grid_points(cand)
        if cand == "K4" and grids is None:
            best1 = results["K1"]["best"]["params"]
            pts = [dict(best1, **g) for g in grid_points("K3")]
        results[cand] = dict(grid=[])
        for gp in pts:
            rows = []; t0 = time.time()
            for P in plants:
                pred, info = cluster_plant(P, cand, gp, caches[P["name"]]); m = metrics(pred, P["truth"], P["eval_mask"]); m.update(plant=P["name"], info={k: v for k, v in info.items() if k != "bic_table"})
                rows.append(m)
            pl = pooled(rows); results[cand]["grid"].append(dict(params=gp, pooled=pl, passes=passes(pl), per_plant=rows, seconds=round(time.time() - t0, 1)))
            print(split, cand, gp, {k: round(v, 3) for k, v in pl.items()}, "PASS" if passes(pl) else "fail", flush=True)
        ok = [g for g in results[cand]["grid"] if g["passes"]]; pool = ok or results[cand]["grid"]
        results[cand]["best"] = max(pool, key=lambda g: g["pooled"]["membership"]); results[cand]["passes"] = bool(ok)
    json.dump(dict(split=split, gate=GATE, results=results, labels=LABELS), open(out / f"{split}_results.json", "w", encoding="utf-8"), indent=1, default=float)
    return results


# ---------------------------------------------------------------- R3-1b
def run_r31b(cand, params, out: Path, tag: str):
    P = load_r31b(); cache = {}; pred, info = cluster_plant(P, cand, params, cache); F, blob = cache["F"], cache["blob"]
    sub = out / "r31b"; sub.mkdir(parents=True, exist_ok=True); bl = np.nonzero(P["is_blade"])[0]; cl = pred[bl]
    rows = []
    for c in sorted(set(cl[cl >= 0].tolist())):
        m = cl == c; th = F["theta"][m]; th0 = circ_mean_deg(th); dth = (th - th0 + 180) % 360 - 180
        rows.append(dict(cluster_id=f"GC{c + 1:02d}", n_surfels=int(m.sum()), theta_mean_deg=round(th0, 1), theta_spread_p5_p95_deg=round(float(np.percentile(dth, 95) - np.percentile(dth, 5)), 1),
                         r_p5_mm=round(float(np.percentile(F["r"][m], 5)) * 1000, 1), r_p95_mm=round(float(np.percentile(F["r"][m], 95)) * 1000, 1), radial_length_mm=round(float(np.percentile(F["r"][m], 95) - np.percentile(F["r"][m], 5)) * 1000, 1),
                         h_min_mm=round(float(F["h"][m].min()) * 1000, 1), h_max_mm=round(float(F["h"][m].max()) * 1000, 1), mean_inclination_deg=round(float(F["incl"][m].mean()), 1),
                         blobs=int(len(set(blob[m].tolist()) - {-1})), **LABELS))
    with open(sub / "clusters.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping geometry_clusters ({cand} {tag}); rank/axis/attachment unavailable; blobs are not leaflets; mm values conditional on the unvalidated scale\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cluster_id"], lineterminator="\n"); w.writeheader(); w.writerows(rows)
    np.savez_compressed(sub / "surfel_clusters.npz", rows=P["rows"], cluster=pred, blob=np.where(P["is_blade"], -2, -1).astype(int) if False else _full(blob, bl, len(pred)), is_axis=P["is_axis"], theta=_full(F["theta"], bl, len(pred), np.nan), h=_full(F["h"], bl, len(pred), np.nan))
    n_cl = len(rows); radius = float(np.percentile(F["r"], 99)) * 1000
    plaus = dict(candidate=cand, tag=tag, params=params, axis=F["axis"], compound_leaves=dict(value=n_cl, gate="5-9", passes=PLAUS["compound_leaves"][0] <= n_cl <= PLAUS["compound_leaves"][1]),
                 azimuth_spread=dict(gate=f"<= {2 * params.get('sector_deg', 35)} deg", values={r["cluster_id"]: r["theta_spread_p5_p95_deg"] for r in rows}, all_pass=all(r["theta_spread_p5_p95_deg"] <= 2 * params.get("sector_deg", 35) for r in rows)),
                 radial_length=dict(gate=f"<= plant radius {radius:.1f} mm; reference leaflet 18.6 mm x 3 pairs = 55.8 mm", values={r["cluster_id"]: r["radial_length_mm"] for r in rows}, all_pass=all(r["radial_length_mm"] <= radius for r in rows)),
                 blobs_per_cluster=(dict(gate="median 4-8", median=float(np.median([r["blobs"] for r in rows])) if rows else None, passes=(PLAUS["blobs_per_cluster"][0] <= np.median([r["blobs"] for r in rows]) <= PLAUS["blobs_per_cluster"][1]) if rows else False) if cand in ("K1", "K4") else dict(gate="K1/K4 only")),
                 unassigned_blade_surfels=dict(value=int((cl < 0).sum()), share=float((cl < 0).mean())), **LABELS)
    # information only: leaflet instances of the two columns vs clusters
    overlap = {}
    for col, run in (("2d_masks_vote", SEED_RUN), ("clustering_only", TRACK_A)):
        lab = np.load(run / "surfel_labels.npz"); pos = {int(r): i for i, r in enumerate(P["rows"])}; res = []
        for inst, v in json.load(open(run / "leaflet_surfels.json", encoding="utf-8")).items():
            if inst.startswith("_"): continue
            idx = np.array([pos[r] for r in v["surfel_rows"] if r in pos], int); cls = pred[idx]; cls = cls[cls >= 0]
            if len(cls) == 0: res.append(dict(instance=inst, n=len(idx), majority_share=None)); continue
            cnt = Counter(cls.tolist()); top, k = cnt.most_common(1)[0]; share = k / len(cls)
            res.append(dict(instance=inst, n=int(len(idx)), majority_cluster=f"GC{top + 1:02d}", majority_share=round(share, 3), straddles=bool(share < PLAUS["straddle_share"]), clusters_touched=len(cnt)))
        overlap[col] = dict(instances=res, straddling=sum(1 for r in res if r.get("straddles")), total=len(res))
    json.dump(plaus, open(sub / "plausibility.json", "w", encoding="utf-8"), indent=1, default=float)
    json.dump(dict(note="information only, after clustering; a merged instance straddling two clusters is evidence of a merge, not accuracy", **overlap), open(sub / "instance_overlap.json", "w", encoding="utf-8"), indent=1, default=float)
    return plaus, overlap, rows


def _full(arr, idx, n, fill=-1):
    out = np.full(n, fill, dtype=np.asarray(arr).dtype if fill == -1 else float); out[idx] = arr; return out


# ---------------------------------------------------------------- viewer + review page
def build_viewer(out: Path):
    sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); n = len(sf["xyz"]); html = TEMPLATE.read_text(encoding="utf-8")
    m = re.search(r'<script id="D" type="application/json">(.*?)</script>', html, re.S); M = json.loads(m.group(1))
    z = np.load(out / "r31b" / "surfel_clusters.npz"); rows = z["rows"]; pred = z["cluster"]; is_axis = z["is_axis"]; clusters = read_csv(out / "r31b" / "clusters.csv")
    ids = [r["cluster_id"] for r in clusters]; pal = dict(zip(ids, palette(len(ids))))
    colG = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); colG[rows[is_axis]] = (70, 70, 70); idxG = np.zeros(n, np.uint16)
    for q, cid in enumerate(ids):
        sel = rows[pred == q]; colG[sel] = pal[cid]; idxG[sel] = q + 1
    # old clustering_only compound colours (Track A) for the toggle
    la = np.load(TRACK_A / "surfel_labels.npz"); leaves = read_csv(TRACK_A / "compound_leaves.csv"); palA = dict(zip([L["leaf_id"] for L in leaves], palette(len(leaves))))
    colA = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); colA[la["rows"][la["is_axis"].astype(bool)]] = (70, 70, 70); idxA = np.zeros(n, np.uint16); leafA = la["leaf_id"].tolist(); lids = [L["leaf_id"] for L in leaves]
    for c in range(la["cl"].max() + 1):
        k = leafA[c]
        if k in palA:
            sel = la["rows"][la["cl"] == c]; colA[sel] = palA[k]; idxA[sel] = lids.index(k) + 1
    cls = np.zeros(n, np.uint8); cls[rows] = 1; lo = sf["xyz"].min(0); rng = np.maximum(sf["xyz"].max(0) - lo, 1e-9)
    M["discs"] = dict(n=n, n_plant=int(cls.sum()), H=float(rng[2]), lo=lo.tolist(), rng=rng.tolist(), pos=_b64(np.round((sf["xyz"] - lo) / rng * 65535).astype(np.uint16)),
                      tu=_b64(np.round(np.clip(sf["tu"], -1, 1) * 32767).astype(np.int16)), tv=_b64(np.round(np.clip(sf["tv"], -1, 1) * 32767).astype(np.int16)),
                      s=_b64(sf["scale"].astype(np.float16)), op=_b64(np.round(np.clip(sf["opacity"], 0, 1) * 255).astype(np.uint8)), col=_b64(sf["rgb"].astype(np.uint8)), cls=_b64(cls))
    M["gc"] = dict(sources=[dict(key="geometry", label="복엽 (기하 군집)", idx=_b64(idxG), col=_b64(colG),
                                 meta=[dict(id=r["cluster_id"], n=int(r["n_surfels"]), theta=f"{r['theta_mean_deg']}° ± {float(r['theta_spread_p5_p95_deg']) / 2:.0f}", h=f"{r['h_min_mm']}–{r['h_max_mm']} mm", blobs=int(r["blobs"]), incl=r["mean_inclination_deg"], colour="#%02x%02x%02x" % pal[r["cluster_id"]]) for r in clusters]),
                            dict(key="clustering_only", label="clustering_only 복엽 (트랙 A, pre_topology_edit)", idx=_b64(idxA), col=_b64(colA),
                                 meta=[dict(id=L["leaf_id"], n=sum(int(x["n_surfels"]) for x in read_csv(TRACK_A / "leaflets.csv") if x["leaf_id"] == L["leaf_id"]), theta=L["azimuth_deg"] + "°", h="—", blobs=int(L["n_leaflets"]), incl=L["inclination_area_weighted_deg"], colour="#%02x%02x%02x" % palA[L["leaf_id"]]) for L in leaves])],
                   answers=["one leaf", "two leaves merged", "part of a leaf", "not a leaf"])
    M["ctr"] = (lo + rng / 2).tolist(); M["note"] = f"{STATUS}: compound-leaf colouring by geometry clusters (no skeleton, no leaflet labels); mid-grey = unassigned, dark grey = axis; rank / axis / attachment unavailable; {LO.SCALE_TAG}"
    html = html[:m.start(1)] + json.dumps(M, separators=(",", ":")) + html[m.end(1):]
    title = f"{STATUS} · 복엽 기하 군집"; html = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", html); html = re.sub(r'<h1 id="ttl">[^<]*</h1>', f'<h1 id="ttl">{title}</h1>', html)
    html = re.sub(r'(z-index:50;background:#7a1f1f;[^>]*>)[^<]*(</div>)', lambda mm: mm.group(1) + f"{STATUS} · 복엽 = 기하 군집(골격·소엽 라벨 미사용) · 순위·축·부착 = unavailable · 형태만, 척도 미검증 · 계측 아님" + mm.group(2), html, count=1)
    ui = ('<label>복엽 색 (출처)</label><div class="row" id="gcsrc"><button data-gs="-1" class="on">끔 (사진 색)</button><button data-gs="0">복엽 (기하 군집)</button><button data-gs="1">clustering_only</button></div>'
          '<div id="gcleg" style="display:none"><div style="font-size:11px;opacity:.8"><span style="display:inline-block;width:10px;height:10px;background:#808080"></span> unassigned &nbsp;<span style="display:inline-block;width:10px;height:10px;background:#464646"></span> axis</div>'
          '<div id="gclist" style="max-height:180px;overflow:auto;font-size:11px"></div><div id="gcpanel" style="font-size:12px;border-top:1px solid var(--line);margin-top:6px;padding-top:6px">군집을 클릭하면 정보가 여기 나옵니다.</div>'
          '<div id="gcans" style="font-size:12px;margin-top:6px"></div><p style="font-size:11px;opacity:.75;margin:4px 0 0">사람 점검 (~10분): 기하 군집마다 하나를 고르세요. 답은 db 에 저장됩니다.</p></div>')
    html = html.replace('<label>불투명도 배율', ui + '<label>불투명도 배율', 1)
    js = r"""<script>(function(){const G=M.gc;const RGB=C;let cur=-1,hl=null;const TEX={};let db=null;const ANS={};
function build(si){const s=G.sources[si];const idx=new Uint16Array(b64(s.idx).buffer);const col=b64(s.col);const C2=new Uint8Array(TW*TH*4);for(let i=0;i<N;i++){let r=col[i*3],g=col[i*3+1],b=col[i*3+2];const k=idx[i];if(k>0&&hl!==null&&hl!==k-1){r=g=b=205;}C2[i*4]=r;C2[i*4+1]=g;C2[i*4+2]=b;C2[i*4+3]=255;}return {idx,C2};}
const _um=gl.uniformMatrix4fv.bind(gl);gl.uniformMatrix4fv=function(loc,t,m){window.__lastMVP=m;return _um(loc,t,m);};
function apply(){if(cur<0){tex(3,RGB,false);document.getElementById('gcleg').style.display='none';}else{const t=build(cur);TEX[cur]=t;tex(3,t.C2,false);document.getElementById('gcleg').style.display='block';list();}document.querySelectorAll('#gcsrc button').forEach(b=>b.className=(+b.dataset.gs===cur)?'on':'');if(window.__viewer)window.__viewer.set({cmode:0});}
function list(){const s=G.sources[cur];document.getElementById('gclist').innerHTML=`<div style="margin:4px 0">${s.label} · ${s.meta.length}</div>`+s.meta.map((m,i)=>`<div data-gc="${i}" style="cursor:pointer;display:flex;gap:6px;align-items:center;padding:1px 0;${hl===i?'font-weight:600':''}"><i style="width:10px;height:10px;background:${m.colour};flex:none"></i>${m.id} <span style="opacity:.7">${m.n}${cur===0&&ANS[m.id]?' · '+ANS[m.id].answer:''}</span></div>`).join('');}
function panel(i){const s=G.sources[cur];const m=s.meta[i];document.getElementById('gcpanel').innerHTML=`<b>${m.id}</b> · ${s.label}<br>surfels ${m.n} · 방위 ${m.theta} · 높이 ${m.h}<br>blobs ${m.blobs} · 평균 경사 ${m.incl}°`;
 document.getElementById('gcans').innerHTML=cur===0?G.answers.map(a=>`<button data-ans="${a}" class="${ANS[m.id]&&ANS[m.id].answer===a?'on':''}" style="margin:2px">${a}</button>`).join(''):'';document.getElementById('gcans').dataset.id=m.id;}
document.getElementById('gcsrc').addEventListener('click',e=>{const b=e.target.closest('[data-gs]');if(!b)return;cur=+b.dataset.gs;hl=null;apply();});
document.getElementById('gclist').addEventListener('click',e=>{const d=e.target.closest('[data-gc]');if(!d)return;const i=+d.dataset.gc;hl=(hl===i)?null:i;apply();panel(i);});
document.getElementById('gcans').addEventListener('click',async e=>{const b=e.target.closest('[data-ans]');if(!b)return;const id=document.getElementById('gcans').dataset.id;ANS[id]={answer:b.dataset.ans,at:new Date().toISOString()};if(db){try{await db.doc('cluster_answers/'+id).set(ANS[id]);}catch(err){}}else{try{localStorage.setItem('gc_ans',JSON.stringify(ANS));}catch(err){}}panel(G.sources[cur].meta.findIndex(m=>m.id===id));list();});
document.querySelectorAll('#cm button').forEach(b=>b.addEventListener('click',()=>{if(+b.dataset.cm===0&&cur>=0){tex(3,TEX[cur]?TEX[cur].C2:build(cur).C2,false);}}));
cv.addEventListener('click',e=>{if(cur<0||!window.__lastMVP)return;const r=cv.getBoundingClientRect();const mx=(e.clientX-r.left)*(cv.width/r.width),my=(e.clientY-r.top)*(cv.height/r.height);const M4=window.__lastMVP;const t=TEX[cur]||build(cur);let best=-1,bd=144,bz=1e9;
 for(let i=0;i<N;i++){if(qk[i]!==1||t.idx[i]===0)continue;const x=PX[i*3],y=PX[i*3+1],z=PX[i*3+2];const cw=M4[3]*x+M4[7]*y+M4[11]*z+M4[15];if(cw<=0)continue;const cx=(M4[0]*x+M4[4]*y+M4[8]*z+M4[12])/cw,cy=(M4[1]*x+M4[5]*y+M4[9]*z+M4[13])/cw;const sx=(cx+1)/2*cv.width,sy=(1-cy)/2*cv.height;const d=(sx-mx)*(sx-mx)+(sy-my)*(sy-my);if(d<bd&&(d<bd-20||cw<bz)){bd=d;best=i;bz=cw;}}
 if(best>=0)panel(t.idx[best]-1);});
try{Object.assign(ANS,JSON.parse(localStorage.getItem('gc_ans')||'{}'));}catch(e){}
(async()=>{try{if(window.claude&&window.claude.use){db=await claude.use('db');if(db){const q=await db.collection('cluster_answers').get();q.docs.forEach(d=>{ANS[d.id]=d.data();});if(cur===0)list();}}}catch(e){}})();
window.__gc={set:(s)=>{cur=s;apply();},state:()=>({cur,hl,sources:G.sources.map(s=>s.meta.length),answers:Object.keys(ANS).length})};if(window.__viewer)window.__viewer.draw();})();</script>"""
    html = html.replace("</body>", js + "</body>", 1) if "</body>" in html else html + js
    (out / "viewer").mkdir(exist_ok=True); p = out / "viewer" / "compound_cluster_viewer.html"; p.write_text(html, encoding="utf-8"); return p


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", choices=("dev", "test", "r31b", "viewer", "all"), default="all"); ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.step in ("dev", "all"):
        run_split("dev", ["K1", "K2", "K3", "K4"], None, out)
    dev = json.load(open(out / "dev_results.json", encoding="utf-8"))["results"]
    chosen = next((c for c in ("K1", "K2", "K3", "K4") if dev[c]["passes"]), None)
    best = chosen or max(("K1", "K2", "K3", "K4"), key=lambda c: dev[c]["best"]["pooled"]["membership"]); params = dev[best]["best"]["params"]
    tag = "frozen_dev_pass" if chosen else "dev_failed"
    json.dump(dict(chosen=chosen, run_on_r31b=best, params=params, tag=tag), open(out / "frozen.json", "w", encoding="utf-8"), indent=1)
    if a.step in ("test", "all"):
        if chosen and not (out / "test_result.json").exists():
            run_split("test", [chosen], {chosen: [params]}, out); (out / "test_results.json").rename(out / "test_result.json")
        elif not chosen:
            print("TEST not run: no candidate passed DEV (contract section 5)")
    if a.step in ("r31b", "all"):
        plaus, overlap, rows = run_r31b(best, params, out, tag); print(json.dumps(dict(candidate=best, tag=tag, clusters=len(rows), plaus={k: v for k, v in plaus.items() if k in ("compound_leaves", "blobs_per_cluster", "unassigned_blade_surfels")}, straddling={k: (v["straddling"], v["total"]) for k, v in overlap.items()}), default=float))
    if a.step in ("viewer", "all"):
        print(build_viewer(out))


if __name__ == "__main__":
    main()


# ================================================================ round 2 (2026-10-25 17:47): diagnosis of the round-1 splits
def _components(xyz, cap_mm=EDGE_MM):
    """Connected components of the kNN graph (k KNN, cap) over the given points."""
    from scipy.sparse.csgraph import connected_components
    n = len(xyz)
    if n == 0:
        return np.zeros(0, int)
    k = min(KNN + 1, n); dd, ii = cKDTree(xyz).query(xyz, k=k); rows, cols = [], []
    for j in range(1, k):
        m = dd[:, j] <= cap_mm / 1000; rows.append(np.nonzero(m)[0]); cols.append(ii[m, j])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)[1]


def _geodesic_path(xyz, i0, i1):
    """Shortest path over the kNN graph (cap EDGE_MM) between two indices; returns (index list, length m) or (None, inf)."""
    from scipy.sparse.csgraph import dijkstra
    n = len(xyz); k = min(KNN + 1, n); dd, ii = cKDTree(xyz).query(xyz, k=k); r_, c_, w_ = [], [], []
    for j in range(1, k):
        m = dd[:, j] <= EDGE_MM / 1000; r_.append(np.nonzero(m)[0]); c_.append(ii[m, j]); w_.append(dd[m, j])
    G = coo_matrix((np.concatenate(w_), (np.concatenate(r_), np.concatenate(c_))), shape=(n, n)).tocsr()
    dist, pred = dijkstra(G, directed=False, indices=i0, return_predecessors=True)
    if not np.isfinite(dist[i1]):
        return None, float("inf")
    path = [i1]
    while path[-1] != i0 and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    return path[::-1], float(dist[i1])


def diagnose_round1(out: Path, k1_params, k3_params):
    """H1-H5 on DEV for every truth leaf split across >= 2 round-1 clusters (parts >= 20 % of the leaf blade points)."""
    plants = [load_bench(Path(p)) for p in sorted(glob.glob(str(BENCH / "dev" / "*")))]; rows = []
    for P in plants:
        z = np.load(BENCH / "dev" / P["name"] / "cloud.npz"); fam = z["family"]; cache = {}
        tree = json.load(open(BENCH / "dev" / P["name"] / "tree.json", encoding="utf-8"))
        stem_poly = np.array(tree["axes"][0]["tube"]["polyline"], float) / 1000 + np.r_[POT_CENTRE, POT_TOP]
        preds = {c: cluster_plant(P, c, gp, cache)[0] for c, gp in (("K1", k1_params), ("K3", k3_params))}
        F = cache["F"]; bl = np.nonzero(P["is_blade"])[0]; used_axis = F["axis"]["used"]
        nn_all = P["normal"] / np.maximum(np.linalg.norm(P["normal"], axis=1, keepdims=True), 1e-12)
        for cand, pred in preds.items():
            n_true = len({l for l in np.unique(P["truth"][P["eval_mask"]]) if l}); n_pred = len(set(pred[pred >= 0].tolist())); excess = max(n_pred - n_true, 0)
            for leaf in sorted({l for l in np.unique(P["truth"]) if l}):
                lb = np.nonzero((P["truth"] == leaf) & P["eval_mask"])[0]; cl = pred[lb]
                cnt = Counter(cl[cl >= 0].tolist()); parts = [c for c, k in cnt.items() if k >= 0.2 * len(lb)]
                if len(parts) < 2:
                    continue
                a, b = sorted(parts, key=lambda c: -cnt[c])[:2]; pa, pb = lb[cl == a], lb[cl == b]
                leaf_all = np.nonzero(P["truth"] == leaf)[0]; leaf_axis = leaf_all[np.isin(fam[leaf_all], ["rachis", "petiolule"])]
                idx_b = np.concatenate([pa, pb]); cb_ = _components(P["xyz"][idx_b]); joined_blade = bool(cb_[0] == cb_[len(pa)])
                idx_ba = np.concatenate([pa, pb, leaf_axis]); ca_ = _components(P["xyz"][idx_ba]); joined_axis = bool(ca_[0] == ca_[len(pa)])
                gap_mm = float(cKDTree(P["xyz"][pb]).query(P["xyz"][pa])[0].min() * 1000)
                ia, ib = np.searchsorted(bl, pa), np.searchsorted(bl, pb)
                dth_used = circ_diff_deg(circ_mean_deg(F["theta"][ia]), circ_mean_deg(F["theta"][ib]))
                def theta_truth(idx):
                    q = P["xyz"][idx]; seg = stem_poly[np.argmin(np.linalg.norm(stem_poly[None, :, :] - q[:, None, :], axis=2), axis=1)]
                    return np.degrees(np.arctan2(q[:, 1] - seg[:, 1], q[:, 0] - seg[:, 0])) % 360
                dth_truth = circ_diff_deg(circ_mean_deg(theta_truth(pa)), circ_mean_deg(theta_truth(pb)))
                na, nb = mean_normal(F["nn"][ia]), mean_normal(F["nn"][ib]); abs_deg = float(np.degrees(np.arccos(min(1.0, abs(float(na @ nb))))))
                T = cKDTree(P["xyz"][idx_ba]); i0, i1 = int(T.query(P["xyz"][pa].mean(0))[1]), int(T.query(P["xyz"][pb].mean(0))[1])
                path, L = _geodesic_path(P["xyz"][idx_ba], i0, i1); rel = None
                if path is not None and len(path) > 1:
                    nn_p = nn_all[idx_ba[path]]; ang = np.degrees(np.arccos(np.clip(np.abs((nn_p[1:] * nn_p[:-1]).sum(1)), 0, 1))); rel = float(ang.sum() / max(L * 1000, 1e-6))
                rows.append(dict(plant=P["name"], cand=cand, leaf=leaf, n_parts=len(parts), joined_by_blade_graph=joined_blade, joined_via_rachis=bool(joined_axis and not joined_blade),
                                 gap_mm=round(gap_mm, 1), dtheta_used_deg=round(dth_used, 1), dtheta_truth_deg=round(dth_truth, 1), abs_normal_deg=round(abs_deg, 1),
                                 rel_normal_deg_per_mm=(round(rel, 2) if rel is not None else None), excess_clusters=int(excess), axis_used=used_axis))
    def share(f, sub):
        v = [f(r) for r in sub]; return (round(float(np.mean(v)), 3) if v else None, len(v))
    def med(key, sub):
        v = [r[key] for r in sub if r[key] is not None]; return round(float(np.median(v)), 2) if v else None
    summ = {}
    for cand, sector, normal_deg in (("K1", k1_params["sector_deg"], k1_params["normal_deg"]), ("K3", k3_params["sector_deg"], None)):
        sub = [r for r in rows if r["cand"] == cand]; n = len(sub); pl = {r["plant"] for r in sub}
        summ[cand] = dict(split_leaves=n, plants_with_splits=len(pl),
                          H1_parts_joined_only_via_excluded_rachis=share(lambda r: r["joined_via_rachis"], sub), H1_parts_joined_by_blade_graph_anyway=share(lambda r: r["joined_by_blade_graph"], sub),
                          H2_gap_gt_4mm=share(lambda r: r["gap_mm"] > 4.0, sub), H2_gap_mm_median=med("gap_mm", sub),
                          H3_sector_fail_on_used_axis_but_ok_on_truth=share(lambda r: r["dtheta_used_deg"] > sector and r["dtheta_truth_deg"] <= sector, sub), H3_dtheta_used_median=med("dtheta_used_deg", sub), H3_dtheta_truth_median=med("dtheta_truth_deg", sub),
                          H4_abs_normal_gt_limit=(share(lambda r: r["abs_normal_deg"] > normal_deg, sub) if normal_deg else "n/a"), H4_abs_normal_deg_median=med("abs_normal_deg", sub), H4_rel_normal_deg_per_mm_median=med("rel_normal_deg_per_mm", sub),
                          H5_splits_coverable_by_excess_clusters=(round(float(sum(min(sum(1 for r in sub if r["plant"] == p), max(r["excess_clusters"] for r in sub if r["plant"] == p)) for p in pl) / n), 3) if n else None))
    out.mkdir(parents=True, exist_ok=True); json.dump(dict(k1=k1_params, k3=k3_params, summary=summ, rows=rows), open(out / "diagnosis_round1.json", "w", encoding="utf-8"), indent=1, default=float)
    return summ


# ================================================================ round 2: rachis connectors, A/R feature forms, R1-R4 (CONTRACT_R2.md)
OUT2 = ROOT / "build/p1_compound_cluster_20261025_02"
R2 = dict(bridge_mm=(8, 12), dir_deg=(30, 45), h_resid_mm=(2, 3), len_mult=(1.5, 2.0), att_mult=(1.0, 1.5), q1_min_inliers=60, leaf_len_prior_mm=37.2, leaflet_mm=18.6,
          local_mm=3.0, end_mm=5.0, seg_mm=10.0, max_blobs=9, sector_deg=45, max_splits=2, midrib_deg=(30, 150), terminal_sep_mm=15.0)
STEM_FAMS = ("stem", "peduncle"); RACHIS_FAMS = ("rachis", "petiolule", "pedicel")


def predicted_axis(xyz, cache_file: Path | None = None):
    """The frozen organs labeller (Track A's labeller step, no mask fusion): p_axis >= 0.6, 1.5 mm smoothing."""
    if cache_file is not None and cache_file.exists():
        return np.load(cache_file)["is_axis"]
    from tomato3d.organs import features as FE
    from tomato3d.organs import labeller as LB
    Fx, _ = FE.compute(xyz); p = LB.Labeller.load().p_axis(Fx); is_axis = FE.smooth(xyz, p >= 0.6)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache_file, is_axis=is_axis, p_axis=p.astype(np.float32))
    return is_axis


def axis_point_features(xyz_a, xyz_all_axis, H, R):
    """Per axis surfel: verticality of the local line (A), thickness relative to the median (R), h (A), h/H (R), r (A), r/R (R), density rel."""
    tree = cKDTree(xyz_all_axis); nb = tree.query_ball_point(xyz_a, R2["local_mm"] / 1000); vert = np.zeros(len(xyz_a)); thick = np.zeros(len(xyz_a)); dirs = np.zeros((len(xyz_a), 3))
    for i, js in enumerate(nb):
        Q = xyz_all_axis[js]
        if len(js) < 4:
            vert[i] = 90.0; thick[i] = np.nan; dirs[i] = [0, 0, 1]; continue
        w, v = np.linalg.eigh(np.cov((Q - Q.mean(0)).T)); d = v[:, -1]; dirs[i] = d * np.sign(d[2] if abs(d[2]) > 1e-9 else 1); vert[i] = np.degrees(np.arccos(min(1.0, abs(d[2])))); thick[i] = np.sqrt(max(w[1], 0))
    thick_rel = thick / max(np.nanmedian(thick), 1e-9); h = xyz_a[:, 2] - POT_TOP
    p0 = np.array([POT_CENTRE[0], POT_CENTRE[1], POT_TOP]); r = np.linalg.norm(xyz_a[:, :2] - p0[:2], axis=1)
    cnt = np.array([len(x) for x in nb], float); dens = cnt / max(np.median(cnt), 1)
    return dict(vert=vert, thick_rel=np.nan_to_num(thick_rel, nan=1.0), h=h, h_rel=h / max(H, 1e-9), r=r, r_rel=r / max(R, 1e-9), dens=dens, dirs=dirs)


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return None
    return float(roc_auc_score(label, score))


class Q1:
    """Stem-like vs rachis-like logistic on the chosen geometry forms (class-balanced, DEV only)."""
    def __init__(self, forms):
        self.forms = forms; self.m = None; self.mean = None; self.sd = None

    def X(self, AF):
        return np.c_[[AF[f] for f in self.forms]].T if self.forms else np.zeros((len(AF["h"]), 0))

    def fit(self, Xs, ys):
        from sklearn.linear_model import LogisticRegression
        X = np.vstack(Xs); y = np.concatenate(ys); self.mean, self.sd = X.mean(0), X.std(0) + 1e-9
        self.m = LogisticRegression(class_weight="balanced", max_iter=5000).fit((X - self.mean) / self.sd, y); return self

    def stem_like(self, AF):
        X = self.X(AF)
        return self.m.predict_proba((X - self.mean) / self.sd)[:, 1] >= 0.5 if X.shape[1] else np.zeros(len(AF["h"]), bool)


def choose_forms_q1(plants_feats):
    """AUC per form for stem (1) vs rachis-like (0) on DEV; forms: h (A) vs h_rel (R), r (A) vs r_rel (R); vert A-only, thick_rel R-only, dens R-only."""
    pairs = [("height", "h", "h_rel"), ("radius", "r", "r_rel")]; single = [("verticality", "vert"), ("thickness", "thick_rel"), ("density", "dens")]
    allF = {k: np.concatenate([pf["AF"][k] for pf in plants_feats]) for k in ("h", "h_rel", "r", "r_rel", "vert", "thick_rel", "dens")}; y = np.concatenate([pf["stem_truth"] for pf in plants_feats])
    table, forms = [], []
    for name, a, r in pairs:
        aa, ar = auc(allF[a], y), auc(allF[r], y); aa2, ar2 = max(aa, 1 - aa), max(ar, 1 - ar)
        pick = a if aa2 >= ar2 else r; best = max(aa2, ar2); keep = best >= 0.60
        table.append(dict(question="Q1", feature=name, auc_A=round(aa2, 3), auc_R=round(ar2, 3), chosen=("A" if pick == a else "R") if keep else "dropped", form=pick if keep else None))
        if keep: forms.append(pick)
    for name, f in single:
        a = auc(allF[f], y); a2 = max(a, 1 - a); keep = a2 >= 0.60
        table.append(dict(question="Q1", feature=name, auc_A=(round(a2, 3) if f == "vert" else None), auc_R=(round(a2, 3) if f != "vert" else None), chosen=(("A" if f == "vert" else "R") if keep else "dropped"), form=f if keep else None))
        if keep: forms.append(f)
    return forms, table


# ---- Q2 pair features (blob pairs within bridge reach)
def pair_features(xyz_b, F, blob, nn_b, max_mm):
    """For blob pairs whose nearest points are <= max_mm: absolute dh, h(s) residual (quadratic over both blobs along the joining line), absolute normal angle,
    relative normal change per mm along the straight join, direction continuity (blob PCA major axes), density valley ratio."""
    nb = blob.max() + 1; members = {b: np.nonzero(blob == b)[0] for b in range(nb)}; trees = {b: cKDTree(xyz_b[m]) for b, m in members.items()}
    cents = {b: xyz_b[m].mean(0) for b, m in members.items()}; ctree = cKDTree(np.array([cents[b] for b in range(nb)]))
    axes = {}
    for b, m in members.items():
        Q = xyz_b[m] - cents[b]; w, v = np.linalg.eigh(Q.T @ Q); axes[b] = v[:, -1]
    rows = {}
    for a, b in ctree.query_pairs(0.06):
        gap = float(trees[b].query(xyz_b[members[a]])[0].min())
        if gap > max_mm / 1000: continue
        ma, mb = members[a], members[b]; d = cents[b] - cents[a]; L = np.linalg.norm(d); u = d / max(L, 1e-9)
        s = np.r_[(xyz_b[ma] - cents[a]) @ u, (xyz_b[mb] - cents[a]) @ u]; h = np.r_[F["h"][ma], F["h"][mb]]
        A = np.c_[np.ones(len(s)), s, s ** 2]; coef, *_ = np.linalg.lstsq(A, h, rcond=None); resid = float(np.sqrt(((h - A @ coef) ** 2).mean()) * 1000)
        na, nb_ = mean_normal(nn_b[ma]), mean_normal(nn_b[mb]); abs_n = float(np.degrees(np.arccos(min(1.0, abs(float(na @ nb_))))))
        rel_n = abs_n / max(L * 1000, 1e-6)
        dir_c = float(np.degrees(np.arccos(min(1.0, abs(float(axes[a] @ axes[b]))))))
        dens_mid = float(np.mean(F["dens"][np.r_[ma, mb]])); mid = (cents[a] + cents[b]) / 2; near = F["tree"].query_ball_point(mid, 0.003); valley = float(len(near) / max(np.median([len(F["tree"].query_ball_point(xyz_b[i], 0.003)) for i in np.r_[ma[:50], mb[:50]]]), 1))
        rows[(a, b)] = dict(gap_mm=gap * 1000, dh_abs_mm=float(abs(F["h"][ma].mean() - F["h"][mb].mean()) * 1000), h_resid_mm=resid, normal_abs_deg=abs_n, normal_rel_deg_per_mm=rel_n, dir_deg=dir_c, dens_valley=valley, dtheta=circ_diff_deg(circ_mean_deg(F["theta"][ma]), circ_mean_deg(F["theta"][mb])))
    return rows


def choose_forms_q2(all_pairs):
    """AUC (same-leaf = 1) for height A (dh_abs, inverted) vs R (h_resid, inverted); normal A (abs angle) vs R (per mm); direction R; density R."""
    y = np.array([p["same"] for p in all_pairs], bool); g = lambda k: np.array([p[k] for p in all_pairs], float)
    def a2(k):
        v = auc(-g(k), y); return None if v is None else max(v, 1 - v)
    table, forms = [], {}
    for name, A_, R_ in (("height", "dh_abs_mm", "h_resid_mm"), ("normal", "normal_abs_deg", "normal_rel_deg_per_mm")):
        aa, ar = a2(A_), a2(R_); pick = A_ if (aa or 0) >= (ar or 0) else R_; best = max(aa or 0, ar or 0); keep = best >= 0.60
        table.append(dict(question="Q2", feature=name, auc_A=(round(aa, 3) if aa else None), auc_R=(round(ar, 3) if ar else None), chosen=(("A" if pick == A_ else "R") if keep else "dropped"), form=pick if keep else None))
        if keep: forms[name] = pick
    for name, R_ in (("direction", "dir_deg"), ("density_valley", "dens_valley"), ("azimuth", "dtheta")):
        ar = a2(R_); keep = (ar or 0) >= 0.60
        table.append(dict(question="Q2", feature=name, auc_A=None, auc_R=(round(ar, 3) if ar else None), chosen=("R" if keep else "dropped"), form=R_ if keep else None))
        if keep: forms[name] = R_
    return forms, table


# ---- candidates
def components_of(xyz, cap_mm=EDGE_MM):
    return _components(xyz, cap_mm)


def run_round2(P, cand, params, q1: Q1, forms2: dict, cache: dict, stem_override=None, axis_override=None):
    """One plant: Q1 labels, stem axis refit, graph over blade + rachis-like, bridges (Q2 tests), R2 shape splits, R3 bilateral attachment."""
    xyz = P["xyz"]; is_axis = P["is_axis"]; bl = np.nonzero(P["is_blade"])[0]; ax = np.nonzero(is_axis)[0]
    if "F" not in cache:
        cache["F"] = features(P, xyz[bl], P["normal"][bl]); cache["blob"], cache["adj"] = blobs_k1(xyz[bl], cache["F"]["dens"], cache["F"])
        H = float(np.percentile(xyz[:, 2], 99) - POT_TOP); R = float(np.percentile(np.linalg.norm(xyz[:, :2] - POT_CENTRE, axis=1), 99))
        cache["AF"] = axis_point_features(xyz[ax], xyz[ax], H, R)
        cache["stem_like"] = (stem_override if stem_override is not None else q1.stem_like(cache["AF"])) if len(ax) else np.zeros(0, bool)
        p0, d, info = (axis_override[0], axis_override[1], dict(used="override", inliers=None)) if axis_override is not None else stem_axis(xyz[ax][cache["stem_like"]]) if cache["stem_like"].sum() >= R2["q1_min_inliers"] else (np.array([POT_CENTRE[0], POT_CENTRE[1], POT_TOP]), np.array([0, 0, 1.0]), dict(used="vertical_through_pot_centre", inliers=int(cache["stem_like"].sum())))
        cache["axis_refit"] = (p0, d, info); nnb = P["normal"][bl] / np.maximum(np.linalg.norm(P["normal"][bl], axis=1, keepdims=True), 1e-12); cache["nn_b"] = nnb
        # recompute theta about the refit axis
        v = xyz[bl] - p0; t = v @ d; rad = xyz[bl] - (p0 + np.outer(t, d)); e1 = np.cross(d, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
        cache["F"]["theta"] = np.degrees(np.arctan2(rad @ e2, rad @ e1)) % 360; cache["F"]["r"] = np.linalg.norm(rad, axis=1)
    F, blob, AF, stem_like = cache["F"], cache["blob"], cache["AF"], cache["stem_like"]
    rach = ax[~stem_like]; conn = np.concatenate([bl, rach]); comp = components_of(xyz[conn]); ncomp = comp.max() + 1
    comp_b = comp[:len(bl)]
    # bridges between components within bridge_mm, Q2 tests on the two end segments
    ctree_pts = cKDTree(xyz[conn]); parent = list(range(ncomp))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    members = {c: np.nonzero(comp == c)[0] for c in range(ncomp)}; trees = {c: cKDTree(xyz[conn][m]) for c, m in members.items()}
    cents = np.array([xyz[conn][members[c]].mean(0) for c in range(ncomp)]); bridges = []; tested = 0
    for a, b in cKDTree(cents).query_pairs(0.08):
        da, ia = trees[b].query(xyz[conn][members[a]]); k = int(np.argmin(da)); gap = float(da[k])
        if gap > params["bridge_mm"] / 1000: continue
        pa, pb = conn[members[a][k]], conn[members[b][ia[k]]]; tested += 1
        # end segments: points of each component within seg_mm of its end point
        sa = conn[members[a]][trees[a].query_ball_point(xyz[pa], R2["seg_mm"] / 1000)]; sb = conn[members[b]][trees[b].query_ball_point(xyz[pb], R2["seg_mm"] / 1000)]
        def line_dir(idx):
            Q = xyz[idx] - xyz[idx].mean(0); w, v = np.linalg.eigh(Q.T @ Q); return v[:, -1]
        ok = True; why = []
        if "direction" in forms2 and len(sa) >= 4 and len(sb) >= 4:
            dd = float(np.degrees(np.arccos(min(1.0, abs(float(line_dir(sa) @ line_dir(sb))))))); ok &= dd <= params["dir_deg"]; why.append(("direction", round(dd, 1), dd <= params["dir_deg"]))
        if "height" in forms2:
            idx = np.r_[sa, sb]; u = xyz[pb] - xyz[pa]; u = u / max(np.linalg.norm(u), 1e-9); s = (xyz[idx] - xyz[pa]) @ u; h = xyz[idx, 2]
            if forms2["height"] == "h_resid_mm":
                A = np.c_[np.ones(len(s)), s, s ** 2]; coef, *_ = np.linalg.lstsq(A, h, rcond=None); val = float(np.sqrt(((h - A @ coef) ** 2).mean()) * 1000)
            else:
                val = float(abs(xyz[sa, 2].mean() - xyz[sb, 2].mean()) * 1000)
            ok &= val <= params["h_resid_mm"]; why.append((forms2["height"], round(val, 2), val <= params["h_resid_mm"]))
        if ok:
            bridges.append(dict(a=int(a), b=int(b), gap_mm=round(gap * 1000, 1), tests=why, p_a=[round(float(v), 5) for v in xyz[pa]], p_b=[round(float(v), 5) for v in xyz[pb]])); parent[find(b)] = find(a)
    comp2 = np.array([find(c) for c in comp]); roots = sorted(set(comp2.tolist())); ren = {r: i for i, r in enumerate(roots)}; leaf_of_conn = np.array([ren[c] for c in comp2])
    cl_b = leaf_of_conn[:len(bl)].copy()
    # a leaf needs >= 1 blade blob; rachis-only components are not leaves
    has_blade = {c for c in set(leaf_of_conn.tolist()) if (leaf_of_conn[:len(bl)] == c).any() and (blob[leaf_of_conn[:len(bl)] == c] >= 0).any()}
    cl_b[~np.isin(cl_b, list(has_blade))] = -1
    splits = []
    if cand in ("R3", "R4"):                                 # contract R2 section 5: blobs reached by NO rachis path are re-attached, so rachis-less components are not leaves of their own
        rach_leaf0 = leaf_of_conn[len(bl):]; has_rachis = sorted(set(rach_leaf0.tolist()))
        cl_b = np.where(np.isin(cl_b, has_rachis), cl_b, -1)
    if cand in ("R2", "R4"):
        cl_b, splits = shape_splits(xyz, bl, conn, leaf_of_conn, cl_b, blob, F, params)
    attached = 0
    if cand in ("R3", "R4"):
        cl_b, attached = bilateral_attach(xyz, bl, conn, leaf_of_conn, cl_b, blob, F, rach, params)
    keep = sorted(set(cl_b[cl_b >= 0].tolist())); ren2 = {c: i for i, c in enumerate(keep)}; cl_b = np.array([ren2.get(int(c), -1) for c in cl_b])
    full = np.full(len(xyz), -1); full[bl] = cl_b
    info = dict(axis=cache["axis_refit"][2], stem_like=int(stem_like.sum()), rachis_like=int((~stem_like).sum()), components=int(ncomp), bridges=len(bridges), bridge_pairs_tested=tested, splits=len(splits), attached=int(attached))
    return full, info, dict(bridges=bridges, splits=splits, stem_like=stem_like, leaf_of_conn=leaf_of_conn, conn=conn)


def shape_splits(xyz, bl, conn, leaf_of_conn, cl_b, blob, F, params):
    """R2: split a leaf at its weakest rachis link while it violates the shape (radial length, >= 2 terminal ends, blobs > 9, azimuth spread), at most 2 splits."""
    out = cl_b.copy(); splits = []; next_id = int(out.max()) + 1
    for c in sorted(set(out[out >= 0].tolist())):
        for _ in range(R2["max_splits"]):
            m = out == c; idx_b = bl[m]
            if m.sum() < 60: break
            rl = float(np.percentile(F["r"][m], 95) - np.percentile(F["r"][m], 5)) * 1000; nblob = len(set(blob[m].tolist()) - {-1})
            th = F["theta"][m]; th0 = circ_mean_deg(th); dth = (th - th0 + 180) % 360 - 180; spread = float(np.percentile(dth, 95) - np.percentile(dth, 5))
            viol = [v for v, bad in (("radial_length", rl > params["len_mult"] * R2["leaf_len_prior_mm"]), ("blobs", nblob > R2["max_blobs"]), ("azimuth_spread", spread > 2 * R2["sector_deg"])) if bad]
            if not viol: break
            # weakest link: cut the blade-point graph (3 mm) at the edge set whose removal leaves two biggest parts - approximate by the principal axis plane with the fewest points in a 3 mm slab
            Q = xyz[idx_b]; cq = Q.mean(0); w, v = np.linalg.eigh(np.cov((Q - cq).T)); s = (Q - cq) @ v[:, -1]
            best = None
            for q in np.linspace(np.percentile(s, 20), np.percentile(s, 80), 25):
                n_slab = int((np.abs(s - q) <= 0.0015).sum()); left = int((s < q).sum()); right = len(s) - left
                score = n_slab / max(min(left, right), 1)
                if best is None or score < best[0]: best = (score, q, n_slab)
            if best is None: break
            new = s >= best[1]
            if new.sum() < 30 or (~new).sum() < 30: break
            out[np.nonzero(m)[0][new]] = next_id; splits.append(dict(leaf=int(c), new=int(next_id), violations=viol, slab_points=best[2])); next_id += 1
    return out, splits


def bilateral_attach(xyz, bl, conn, leaf_of_conn, cl_b, blob, F, rach, params):
    """R3: blade blobs in no leaf join the leaf whose rachis line they flank bilaterally (distance, midrib angle 30-150 deg, height continuity)."""
    out = cl_b.copy(); n_att = 0; leaves = sorted(set(out[out >= 0].tolist()))
    if not leaves or not len(rach): return out, 0
    rach_leaf = leaf_of_conn[len(bl):]; lines = {}
    for c in leaves:
        pts = xyz[rach[rach_leaf == c]] if (rach_leaf == c).any() else None
        if pts is None or len(pts) < 6: continue
        cq = pts.mean(0); w, v = np.linalg.eigh(np.cov((pts - cq).T)); lines[c] = (cq, v[:, -1], pts)
    if not lines: return out, 0
    for b in sorted(set(blob[(blob >= 0) & (out < 0)].tolist())):
        idx = np.nonzero((blob == b) & (out < 0))[0]
        if len(idx) < 20: continue
        Q = xyz[bl[idx]]; cb = Q.mean(0); w, v = np.linalg.eigh(np.cov((Q - cb).T)); midrib = v[:, -1]; best = None
        for c, (cq, d, pts) in lines.items():
            proj = cq + ((cb - cq) @ d) * d; dist = float(np.linalg.norm(cb - proj)) * 1000
            if dist > params["att_mult"] * R2["leaflet_mm"]: continue
            ang = float(np.degrees(np.arccos(min(1.0, abs(float(midrib @ d)))))); ang = min(ang, 180 - ang)
            if not (R2["midrib_deg"][0] <= ang or ang >= 180 - R2["midrib_deg"][1]): pass
            if ang < R2["midrib_deg"][0]: continue
            hr = pts[np.argmin(np.linalg.norm(pts - proj, axis=1)), 2]; dh = abs(cb[2] - hr) * 1000
            if dh > params["h_resid_mm"] * 2: continue
            if best is None or dist < best[0]: best = (dist, c)
        if best is not None:
            out[idx] = best[1]; n_att += 1
    return out, n_att


def bench_round2(split: str, cands, grids: dict | None, out: Path, q1: Q1, forms2: dict):
    plants = [load_bench(Path(p)) for p in sorted(glob.glob(str(BENCH / split / "*")))]; caches = {p["name"]: {} for p in plants}; results = {}
    for P in plants:
        P["is_axis"] = predicted_axis(P["xyz"], out / "labels" / f"{split}_{P['name']}.npz"); P["is_blade"] = ~P["is_axis"] & (P["truth"] != "") | (~P["is_axis"] & P["eval_mask"])
    import itertools
    for cand in cands:
        if grids: pts = grids[cand]
        else:
            keys = ["bridge_mm", "dir_deg", "h_resid_mm"] + (["len_mult"] if cand in ("R2", "R4") else []) + (["att_mult"] if cand in ("R3", "R4") else [])
            pts = [dict(zip(keys, v)) for v in itertools.product(*[R2[k] for k in keys])]
        results[cand] = dict(grid=[])
        for gp in pts:
            rows = []; t0 = time.time()
            for P in plants:
                pred, info, _ = run_round2(P, cand, gp, q1, forms2, caches[P["name"]]); m = metrics(pred, P["truth"], P["eval_mask"]); m.update(plant=P["name"], info=info); rows.append(m)
            pl = pooled(rows); results[cand]["grid"].append(dict(params=gp, pooled=pl, passes=passes(pl), per_plant=rows, seconds=round(time.time() - t0, 1)))
            print(split, cand, gp, {k: round(v, 3) for k, v in pl.items()}, "PASS" if passes(pl) else "fail", flush=True)
        ok = [g for g in results[cand]["grid"] if g["passes"]]; pool = ok or results[cand]["grid"]
        results[cand]["best"] = max(pool, key=lambda g: g["pooled"]["membership"]); results[cand]["passes"] = bool(ok)
    json.dump(dict(split=split, gate=GATE, results=results, labels=LABELS), open(out / f"{split}_results_r2.json", "w", encoding="utf-8"), indent=1, default=float)
    return results, plants, caches


def q1_and_forms(out: Path):
    """DEV: predicted axis labels, Q1 forms by AUC, Q1 logistic, Q2 pair forms by AUC. Returns (q1, forms2, tables)."""
    plants = [load_bench(Path(p)) for p in sorted(glob.glob(str(BENCH / "dev" / "*")))]; pf = []; pairs = []
    for P in plants:
        z = np.load(BENCH / "dev" / P["name"] / "cloud.npz"); fam = z["family"]
        P["is_axis"] = predicted_axis(P["xyz"], out / "labels" / f"dev_{P['name']}.npz"); ax = np.nonzero(P["is_axis"])[0]
        H = float(np.percentile(P["xyz"][:, 2], 99) - POT_TOP); R = float(np.percentile(np.linalg.norm(P["xyz"][:, :2] - POT_CENTRE, axis=1), 99))
        AF = axis_point_features(P["xyz"][ax], P["xyz"][ax], H, R); st = np.isin(fam[ax], STEM_FAMS); rk = np.isin(fam[ax], RACHIS_FAMS); keep = st | rk
        pf.append(dict(name=P["name"], AF={k: v[keep] for k, v in AF.items() if k != "dirs"}, stem_truth=st[keep], AF_all=AF, ax=ax))
        bl = np.nonzero(~P["is_axis"])[0]; F = features(P, P["xyz"][bl], P["normal"][bl]); blob, _ = blobs_k1(P["xyz"][bl], F["dens"], F)
        nnb = P["normal"][bl] / np.maximum(np.linalg.norm(P["normal"][bl], axis=1, keepdims=True), 1e-12); pr = pair_features(P["xyz"][bl], F, blob, nnb, max(R2["bridge_mm"]))
        tl = P["truth"][bl]
        for (a, b), row in pr.items():
            la = Counter(tl[blob == a].tolist()).most_common(1)[0][0]; lb = Counter(tl[blob == b].tolist()).most_common(1)[0][0]
            if la and lb: pairs.append(dict(row, same=(la == lb), plant=P["name"]))
    forms1, t1 = choose_forms_q1(pf); forms1, t1 = apply_ks_q1(forms1, t1, pf); q1 = Q1(forms1).fit([pf_["AF"] and np.c_[[pf_["AF"][f] for f in forms1]].T for pf_ in pf], [pf_["stem_truth"] for pf_ in pf]) if forms1 else Q1([])
    prec = {}
    if forms1:
        yp = np.concatenate([q1.stem_like(pf_["AF"]) for pf_ in pf]); yt = np.concatenate([pf_["stem_truth"] for pf_ in pf])
        prec = dict(stem_precision=float((yp & yt).sum() / max(yp.sum(), 1)), stem_recall=float((yp & yt).sum() / max(yt.sum(), 1)), rachis_precision=float((~yp & ~yt).sum() / max((~yp).sum(), 1)), rachis_recall=float((~yp & ~yt).sum() / max((~yt).sum(), 1)), n=int(len(yt)))
    forms2, t2 = choose_forms_q2(pairs)
    json.dump(dict(q1_forms=forms1, q1_table=t1, q1_dev=prec, q2_forms=forms2, q2_table=t2, q2_pairs=len(pairs), q2_same=int(sum(p["same"] for p in pairs)),
                   note="photo features (brightness, colour) not in any form table: no user spot labels exist, so they stay out of every rule (contract R2 section 3)"),
              open(out / "feature_forms.json", "w", encoding="utf-8"), indent=1, default=float)
    return q1, forms2, dict(q1=t1, q2=t2, q1_dev=prec)


def main_round2(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", choices=("forms", "dev", "test", "r31b", "all"), default="all"); ap.add_argument("--out", type=Path, default=OUT2); a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    q1, forms2, tables = q1_and_forms(out); print("Q1 forms", q1.forms, tables["q1_dev"]); print("Q2 forms", forms2)
    if a.step == "forms": return
    if a.step in ("dev", "all"):
        bench_round2("dev", ["R1", "R2", "R3", "R4"], None, out, q1, forms2)
    dev = json.load(open(out / "dev_results_r2.json", encoding="utf-8"))["results"]
    chosen = next((c for c in ("R1", "R2", "R3", "R4") if dev[c]["passes"]), None); best = chosen or max(dev, key=lambda c: dev[c]["best"]["pooled"]["membership"]); params = dev[best]["best"]["params"]
    tag = "frozen_dev_pass" if chosen else "dev_failed"; json.dump(dict(chosen=chosen, run_on_r31b=best, params=params, tag=tag, q1_forms=q1.forms, q2_forms=forms2), open(out / "frozen_r2.json", "w", encoding="utf-8"), indent=1)
    if a.step in ("test", "all") and chosen and not (out / "test_result_r2.json").exists():
        bench_round2("test", [chosen], {chosen: [params]}, out, q1, forms2); (out / "test_results_r2.json").rename(out / "test_result_r2.json")
    if a.step in ("r31b", "all"):
        P = load_r31b(); cache = {}; pred, info, extra = run_round2(P, best, params, q1, forms2, cache)
        sub = out / "r31b"; sub.mkdir(exist_ok=True); F, blob = cache["F"], cache["blob"]; bl = np.nonzero(P["is_blade"])[0]; cl = pred[bl]; rows = []
        for c in sorted(set(cl[cl >= 0].tolist())):
            m = cl == c; th = F["theta"][m]; th0 = circ_mean_deg(th); dth = (th - th0 + 180) % 360 - 180; bb = sorted(set(blob[m].tolist()) - {-1})
            rows.append(dict(cluster_id=f"RC{c + 1:02d}", n_surfels=int(m.sum()), theta_mean_deg=round(th0, 1), theta_spread_p5_p95_deg=round(float(np.percentile(dth, 95) - np.percentile(dth, 5)), 1),
                             radial_length_mm=round(float(np.percentile(F["r"][m], 95) - np.percentile(F["r"][m], 5)) * 1000, 1), h_min_mm=round(float(F["h"][m].min()) * 1000, 1), h_max_mm=round(float(F["h"][m].max()) * 1000, 1),
                             mean_inclination_deg=round(float(F["incl"][m].mean()), 1), blobs=len(bb), **LABELS))
        with open(sub / "clusters.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping geometry_clusters round 2 ({best} {tag}); rank/axis/attachment unavailable; blobs are not leaflets\n")
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cluster_id"], lineterminator="\n"); w.writeheader(); w.writerows(rows)
        ax = np.nonzero(P["is_axis"])[0]; axis_kind = np.zeros(len(pred), np.int8); axis_kind[ax[extra["stem_like"]]] = 1; axis_kind[ax[~extra["stem_like"]]] = 2
        np.savez_compressed(sub / "surfel_clusters.npz", rows=P["rows"], cluster=pred, axis_kind=axis_kind, blob=_full(blob, bl, len(pred)), theta=_full(F["theta"], bl, len(pred), np.nan), h=_full(F["h"], bl, len(pred), np.nan), incl=_full(F["incl"], bl, len(pred), np.nan), dens=_full(F["dens"], bl, len(pred), np.nan))
        n_cl = len(rows); blobs_med = float(np.median([r["blobs"] for r in rows])) if rows else None
        plaus = dict(candidate=best, tag=tag, params=params, info=info, compound_leaves=dict(value=n_cl, gate="5-9", passes=5 <= n_cl <= 9), blobs_per_leaf=dict(median=blobs_med, gate="3-9", passes=(blobs_med is not None and 3 <= blobs_med <= 9)),
                     unassigned_blade_surfels=dict(value=int((cl < 0).sum()), share=float((cl < 0).mean())), bridges=extra["bridges"][:200], splits=extra["splits"], **LABELS)
        json.dump(plaus, open(sub / "plausibility.json", "w", encoding="utf-8"), indent=1, default=float)
        print(json.dumps(dict(candidate=best, tag=tag, clusters=n_cl, info=info, blobs_median=blobs_med, unassigned=plaus["unassigned_blade_surfels"]), default=float))


# ---------------------------------------------------------------- R3-1b photo features (feature map only; no rule uses them without user spot labels)
def photo_features_r31b(out: Path):
    """Per surfel (rows of Track A): median L*, a*, b* over the training views that see it (Revision-1 splat visibility of the 2D-seed run),
    sampled at the projected pixel of the undistorted 3024 px photo at 1512 px; brightness residual after a per-view linear fit of L* on
    cos(normal, view direction) (shading removal only), then a within-plant z-score; local contrast = residual minus the 3 mm ring mean;
    colour as within-capture z-scores of a*, b* and chroma. Written to r31b/photo_features.npz. Shading is never used as depth."""
    import cv2
    from PIL import Image
    from tomato3d.organs import seed2d as S
    from tomato3d.recon import charuco_a3 as ca
    P = load_r31b(); xyz = P["xyz"]; nrm = P["normal"] / np.maximum(np.linalg.norm(P["normal"], axis=1, keepdims=True), 1e-12)
    sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); sc = sf["scale"][P["rows"]]
    po = json.load(open(LO.DEFAULTS["poses"])); views = json.load(open(LO.DEFAULTS["views"]))["train"]
    V = S.votes_per_view(xyz, po, views, S.VIS["res"], S.VIS["tol_m"], sc=sc, mode="splat"); res = 1512
    K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res); n = len(xyz); L = np.full((n, len(views)), np.nan, np.float32); A = L.copy(); B = L.copy(); COS = L.copy()
    for k, v in enumerate(views):
        vis = np.nonzero(V[:, k] >= 0)[0]
        if not len(vis): continue
        img = np.asarray(Image.open(LO.DEFAULTS["poses"].parent / "undist_3024" / v).convert("RGB").resize((res, res), Image.BILINEAR)); lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"]); xc = xyz[vis] @ R.T + t; z = xc[:, 2]
        u = np.clip(np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int), 0, res - 1); w = np.clip(np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int), 0, res - 1)
        px_ = lab[w, u]; L[vis, k] = px_[:, 0] * 100 / 255; A[vis, k] = px_[:, 1] - 128; B[vis, k] = px_[:, 2] - 128
        cam = -(R.T @ t); d = cam - xyz[vis]; d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12); COS[vis, k] = np.abs((nrm[vis] * d).sum(1))
        # per-view shading fit L* ~ a + b cos, residual
        ok = np.isfinite(L[vis, k]); Xf = np.c_[np.ones(ok.sum()), COS[vis, k][ok]]; coef, *_ = np.linalg.lstsq(Xf, L[vis, k][ok], rcond=None); L[vis[ok], k] = L[vis[ok], k] - Xf @ coef
    n_vis = np.isfinite(L).sum(1); Lres = np.nanmedian(L, axis=1); a_ = np.nanmedian(A, axis=1); b_ = np.nanmedian(B, axis=1)
    def z_(x):
        m = np.isfinite(x); mu, sd = np.nanmean(x[m]), np.nanstd(x[m]) + 1e-9; return np.where(m, (x - mu) / sd, np.nan)
    tree = cKDTree(xyz); ring = tree.query_ball_point(xyz, 0.003); contrast = np.array([Lres[i] - np.nanmean(Lres[r]) if len(r) > 1 else np.nan for i, r in enumerate(ring)], np.float32)
    chroma = np.sqrt(a_ ** 2 + b_ ** 2)
    outp = out / "r31b"; outp.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp / "photo_features.npz", rows=P["rows"], n_vis=n_vis, L_resid_z=z_(Lres).astype(np.float32), contrast=contrast, a_z=z_(a_).astype(np.float32), b_z=z_(b_).astype(np.float32), chroma_z=z_(chroma).astype(np.float32),
                        note="feature map only; residual after per-view L*~cos(normal,view) fit; within-capture z-scores; no rule uses these without user spot labels")
    return dict(n=int(n), seen=int((n_vis > 0).sum()), n_vis_median=float(np.median(n_vis)))


# ---------------------------------------------------------------- round-2 viewer: rachis-connected colours, two axis greys, bridges, feature maps, spot labels, human check
def _ramp(v, lo, hi):
    """viridis-like 4-stop ramp on [lo, hi] -> uint8 rgb (NaN -> dark grey)."""
    stops = np.array([[49, 18, 92], [33, 107, 143], [51, 181, 120], [253, 232, 36]], float); t = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    out = np.zeros((len(v), 3), np.uint8); ok = np.isfinite(v); tt = t[ok] * 3; i = np.minimum(tt.astype(int), 2); f = (tt - i)[:, None]
    out[ok] = (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8); out[~ok] = (70, 70, 70); return out


def build_viewer_r2(out: Path):
    sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); n = len(sf["xyz"]); html = TEMPLATE.read_text(encoding="utf-8")
    m = re.search(r'<script id="D" type="application/json">(.*?)</script>', html, re.S); M = json.loads(m.group(1))
    z = np.load(out / "r31b" / "surfel_clusters.npz"); rows = z["rows"]; pred = z["cluster"]; ak = z["axis_kind"]; clusters = read_csv(out / "r31b" / "clusters.csv"); plaus = json.load(open(out / "r31b" / "plausibility.json", encoding="utf-8"))
    ids = [r["cluster_id"] for r in clusters]; pal = dict(zip(ids, palette(len(ids))))
    col = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); col[rows[ak == 1]] = (60, 60, 60); col[rows[ak == 2]] = (165, 165, 165); idx = np.zeros(n, np.uint16)
    for q, cid in enumerate(ids):
        sel = rows[pred == q]; col[sel] = pal[cid]; idx[sel] = q + 1
    # round-1 colours as a toggle
    z1 = np.load(OUT / "r31b" / "surfel_clusters.npz"); c1 = read_csv(OUT / "r31b" / "clusters.csv"); ids1 = [r["cluster_id"] for r in c1]; pal1 = dict(zip(ids1, palette(len(ids1))))
    col1 = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); col1[z1["rows"][z1["is_axis"].astype(bool)]] = (70, 70, 70); idx1 = np.zeros(n, np.uint16)
    for q, cid in enumerate(ids1):
        sel = z1["rows"][z1["cluster"] == q]; col1[sel] = pal1[cid]; idx1[sel] = q + 1
    # feature maps (chosen forms where a choice exists)
    ph = np.load(out / "r31b" / "photo_features.npz") if (out / "r31b" / "photo_features.npz").exists() else None
    feats = {}
    def fmap(name, vals, lo, hi, label):
        full = np.full(n, np.nan); full[rows] = vals; feats[name] = dict(label=label, col=_b64(_ramp(full, lo, hi)), lo=float(lo), hi=float(hi))
    hh = z["h"] * 1000; fmap("height", hh, 0, np.nanpercentile(hh, 99), "높이 h (A, mm 위 화분)"); fmap("incl", z["incl"], 0, 90, "법선 경사 (A, 0–90°)")
    fmap("dens", z["dens"], 0, 2, "밀도 / 국소 중앙값 (R)"); th = z["theta"]; fmap("theta", th, 0, 360, "방위 θ (재맞춤 축, A)")
    if ph is not None:
        fmap("L_resid", ph["L_resid_z"], -2.5, 2.5, "밝기 잔차 z (R, 음영 제거)"); fmap("contrast", ph["contrast"], -10, 10, "국소 대비 (R)"); fmap("chroma", ph["chroma_z"], -2.5, 2.5, "채도 z (A, 이 촬영 안)"); fmap("a_z", ph["a_z"], -2.5, 2.5, "a* z (A, 이 촬영 안)")
    cls = np.zeros(n, np.uint8); cls[rows] = 1; lo = sf["xyz"].min(0); rng = np.maximum(sf["xyz"].max(0) - lo, 1e-9)
    M["discs"] = dict(n=n, n_plant=int(cls.sum()), H=float(rng[2]), lo=lo.tolist(), rng=rng.tolist(), pos=_b64(np.round((sf["xyz"] - lo) / rng * 65535).astype(np.uint16)),
                      tu=_b64(np.round(np.clip(sf["tu"], -1, 1) * 32767).astype(np.int16)), tv=_b64(np.round(np.clip(sf["tv"], -1, 1) * 32767).astype(np.int16)),
                      s=_b64(sf["scale"].astype(np.float16)), op=_b64(np.round(np.clip(sf["opacity"], 0, 1) * 255).astype(np.uint8)), col=_b64(sf["rgb"].astype(np.uint8)), cls=_b64(cls))
    bridges = [dict(a=b["p_a"], b=b["p_b"], gap=b["gap_mm"], tests=b["tests"]) for b in plaus.get("bridges", []) if "p_a" in b]
    meta = [dict(id=r["cluster_id"], n=int(r["n_surfels"]), theta=f"{r['theta_mean_deg']}° ± {float(r['theta_spread_p5_p95_deg']) / 2:.0f}", h=f"{r['h_min_mm']}–{r['h_max_mm']} mm", blobs=int(r["blobs"]), incl=r["mean_inclination_deg"], colour="#%02x%02x%02x" % pal[r["cluster_id"]]) for r in clusters]
    meta1 = [dict(id=r["cluster_id"], n=int(r["n_surfels"]), theta=f"{r['theta_mean_deg']}°", h=f"{r['h_min_mm']}–{r['h_max_mm']} mm", blobs=int(r["blobs"]), incl=r["mean_inclination_deg"], colour="#%02x%02x%02x" % pal1[r["cluster_id"]]) for r in c1]
    M["gc"] = dict(sources=[dict(key="rachis", label="복엽 (엽축 연결)", idx=_b64(idx), col=_b64(col), meta=meta), dict(key="round1", label="복엽 (기하 군집, 1차)", idx=_b64(idx1), col=_b64(col1), meta=meta1)],
                   feats=feats, bridges=bridges, answers=["one leaf", "two leaves merged", "part of a leaf", "not a leaf"], spots=["stem", "rachis", "blade"],
                   greys=dict(stem_like="#3c3c3c", rachis_like="#a5a5a5", unassigned="#808080"), tag=plaus.get("tag"), candidate=plaus.get("candidate"))
    if (out / "r31b_r5" / "surfel_clusters.npz").exists():
        z5 = np.load(out / "r31b_r5" / "surfel_clusters.npz"); c5 = read_csv(out / "r31b_r5" / "clusters.csv"); ids5 = [r["cluster_id"] for r in c5]; pal5 = dict(zip(ids5, palette(len(ids5))))
        col5 = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); col5[z5["rows"][z5["axis_kind"] == 1]] = (60, 60, 60); col5[z5["rows"][z5["axis_kind"] == 2]] = (165, 165, 165); idx5 = np.zeros(n, np.uint16)
        for q, cid in enumerate(ids5):
            sel = z5["rows"][z5["cluster"] == q]; col5[sel] = pal5[cid]; idx5[sel] = q + 1
        M["gc"]["sources"].append(dict(key="r5", label="사용자 씨앗 R5 (US = 개발 엽축 씨앗, RC = 자동)", idx=_b64(idx5), col=_b64(col5),
                                       meta=[dict(id=r["cluster_id"], n=int(r["n_surfels"]), theta=f"{r['theta_mean_deg']}° ± {float(r['theta_spread_p5_p95_deg']) / 2:.0f}", h=f"{r['h_min_mm']}–{r['h_max_mm']} mm", blobs=int(r["blobs"]), incl=r["mean_inclination_deg"], colour="#%02x%02x%02x" % pal5[r["cluster_id"]]) for r in c5]))
    # colour set 4: 2D rachis assignment + per-surfel vote (build/p1_compound_2dvote_20261025_01, its own contract)
    v2 = ROOT / "build/p1_compound_2dvote_20261025_01"; v2btn = ""; flipped_html = ""
    if (v2 / "surfel_clusters.npz").exists():
        from tomato3d.organs import vote2d as V2
        zv = np.load(v2 / "surfel_clusters.npz", allow_pickle=True); lv = read_csv(v2 / "leaves.csv"); sc_v = json.load(open(v2 / "scores.json", encoding="utf-8"))
        colv = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); colv[zv["rows"][zv["axis_kind"] == 1]] = (60, 60, 60); colv[zv["rows"][zv["axis_kind"] == 2]] = (165, 165, 165); idxv = np.zeros(n, np.uint16)
        leaf_ids = sorted({int(r["leaf"]) for r in lv}); metav = []
        for q, L in enumerate(leaf_ids):
            sel = zv["rows"][zv["leaf"] == L]; colv[sel] = V2.PAL.get(L, (200, 200, 200)); idxv[sel] = q + 1; r = next(x for x in lv if int(x["leaf"]) == L)
            metav.append(dict(id=f"잎 {L} ({'개발' if V2.is_dev(L) else '보류'})", n=int(r["n_surfels"]), theta=f"{r.get('azimuth_deg') or '—'}°", h=f"밑동 {r['base_height_mm']} mm", blobs=(r.get("count_2d_estimate") or "—"),
                              incl=(r.get("inclination_area_weighted_deg") or "—"), colour="#%02x%02x%02x" % V2.PAL.get(L, (200, 200, 200))))
        v2btn = f'<button data-gs="{len(M["gc"]["sources"])}">2D 엽축 배정</button>'
        M["gc"]["sources"].append(dict(key="vote2d", label=f"복엽 (2D 엽축 배정; blobs 열 = 2d_count_estimate; {'게이트 통과' if sc_v.get('gate_pass') else '게이트 불합격, 보기용'})", idx=_b64(idxv), col=_b64(colv), meta=metav))
        fl = sc_v.get("flipped_traces", []); flipped_html = ('<div id="flipped" style="font-size:11px;opacity:.8">뒤집은 엽축(밑동 = 중심 쪽 끝, 저장소는 그대로): ' + (", ".join(fl) if fl else "없음") + '</div>')
    # colour sets 5-7: user-labelled compound leaves (build/p1_compound_userlabel_20261025_01, map v3 ChArUco-corrected), propagated, A3 v2 leaflets inside a leaf
    ul = ROOT / "build/p1_compound_userlabel_20261025_01/map_v3_charuco"
    if ul.exists():
        from tomato3d.organs import vote2d as V2u
        def leaf_set(folder, key, label):
            f = ul / folder / "surfel_clusters.npz"
            if not f.exists(): return None
            zu = np.load(f, allow_pickle=True); sc_ = json.load(open(ul / folder / "scores.json", encoding="utf-8")); h = sc_["held_out_summary"]
            colu = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); colu[zu["rows"][zu["axis_kind"] == 1]] = (60, 60, 60); colu[zu["rows"][zu["axis_kind"] == 2]] = (165, 165, 165); idxu = np.zeros(n, np.uint16); meta_ = []
            for q, L in enumerate(sorted(sc_["per_leaf_surfels"], key=int)):
                sel = zu["rows"][zu["leaf"] == int(L)]; c3 = V2u.PAL.get(int(L), (200, 200, 200)); colu[sel] = c3; idxu[sel] = q + 1
                meta_.append(dict(id=f"잎 {L}", n=int(sc_["per_leaf_surfels"][L]), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
            return dict(key=key, label=f"{label} (보류 사진 IoU {h['median_iou']:.2f} / 혼동 {h['max_confusion']:.2f}, {'합격' if h['gate_pass'] else '게이트 불합격, 보기용'})", idx=_b64(idxu), col=_b64(colu), meta=meta_)
        for sset in (leaf_set("user_labelled_fill_topo", "userlabel", "복엽 (사용자 라벨)"), leaf_set("user_labelled_propagated", "userlabel_vp", "복엽 (사용자 라벨 + SAM2 전파)"), leaf_set("user_labelled_propagated_traceflank_h", "userlabel_vp_trace", "복엽 (전파 + 엽축 주변·높이 규칙)")):
            if sset: v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">{sset["label"].split(" (")[0] + (" 전파" if sset["key"] == "userlabel_vp" else (" 전파+엽축·높이" if sset["key"] == "userlabel_vp_trace" else ""))}</button>'; M["gc"]["sources"].append(sset)
        fl = ul / "leaflet_instances_a3v2_within_user_leaf" / "surfel_leaflets.npz"
        if fl.exists():
            zl = np.load(fl, allow_pickle=True); keys_ = [str(k) for k in zl["keys"]]; pal_l = palette(len(keys_)); coll = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); idxl = np.zeros(n, np.uint16); meta_l = []
            for q, kk in enumerate(keys_):
                sel = zl["rows"][zl["leaflet"] == q]; coll[sel] = pal_l[q]; idxl[sel] = q + 1; meta_l.append(dict(id=f"소엽 {kk}", n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % pal_l[q]))
            sl = json.load(open(ul / "leaflet_instances_a3v2_within_user_leaf" / "scores.json", encoding="utf-8"))["held_out_score"]["cap28"]
            v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">소엽 A3 v2</button>'
            M["gc"]["sources"].append(dict(key="a3v2", label=f"소엽 (복엽 안 A3 v2, 번호-소엽; 보류 사진 소엽당 군집 {sl['mean_clusters_per_leaflet']} / 순도 {sl['mean_purity']}: 목표 ≤1.3 / ≥0.98 {'합격' if sl['pass_'] else '불합격, 보기용'})", idx=_b64(idxl), col=_b64(coll), meta=meta_l))
    # re-cluster of V-P (build/p1_vp_recluster_20261026_01, Revision 1): leaf colours, source overlay, probability map
    rc = ROOT / "build/p1_vp_recluster_20261026_01"
    if (rc / "surfel_clusters.npz").exists():
        from tomato3d.organs import vote2d as V2r
        zr = np.load(rc / "surfel_clusters.npz", allow_pickle=True); sr = json.load(open(rc / "scores.json", encoding="utf-8")); pk = sr["pick"]; hs = sr["held_out"][pk]
        colr = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); colr[zr["rows"][zr["axis_kind"] == 1]] = (60, 60, 60); colr[zr["rows"][zr["axis_kind"] == 2]] = (165, 165, 165); idxr = np.zeros(n, np.uint16); metar = []
        leaves_r = sorted(set(int(x) for x in zr["leaf"] if x > 0))
        for q, L in enumerate(leaves_r):
            sel = zr["rows"][zr["leaf"] == L]; c3 = V2r.PAL.get(L, (200, 200, 200)); colr[sel] = c3; idxr[sel] = q + 1
            metar.append(dict(id=f"잎 {L} ({hs['flags'].get(str(L), '보류 사진에 없음')})", n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">복엽 재군집</button>'
        M["gc"]["sources"].append(dict(key="recluster", label=f"복엽 (재군집, {pk}; 보류 사진 IoU {hs['summary']['median_iou']:.2f} / 혼동 {hs['summary']['max_confusion']:.2f} / 덮임 {hs['coverage']:.2f}, 게이트 불합격, 보기용)", idx=_b64(idxr), col=_b64(colr), meta=metar))
        SRC = {"anchor": ((70, 110, 200), "원래 (앵커)"), "kept": ((150, 170, 220), "원래 (유지)"), "filled": ((40, 190, 90), "채움"), "relabelled": ((230, 60, 60), "수정"), "unassigned": ((90, 90, 90), "미배정")}
        cols = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); idxs = np.zeros(n, np.uint16); metas = []
        for q, (k, (c3, lbl)) in enumerate(SRC.items()):
            sel = zr["rows"][zr["source"] == k]; cols[sel] = c3; idxs[sel] = q + 1; metas.append(dict(id=lbl, n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">채움/수정/원래</button>'
        M["gc"]["sources"].append(dict(key="recluster_src", label="채움/수정/원래 (재군집 대 V-P)", idx=_b64(idxs), col=_b64(cols), meta=metas))
        fullp = np.full(n, np.nan); fullp[zr["rows"]] = np.where(zr["leaf"] >= 0, zr["probability"], np.nan)
        M["gc"]["feats"]["recluster_p"] = dict(label="재군집 확률 (M1, 앵커 = 1)", col=_b64(_ramp(fullp, 0, 1)), lo=0.0, hi=1.0)
    # trace x label reconciliation (build/p1_trace_label_reconcile_20261026_01): leaf colours (trace numbering) + source overlay
    tr = ROOT / "build/p1_trace_label_reconcile_20261026_01"
    if (tr / "surfel_clusters.npz").exists():
        from tomato3d.organs import vote2d as V2t
        zt = np.load(tr / "surfel_clusters.npz", allow_pickle=True); st_ = json.load(open(tr / "scores.json", encoding="utf-8")); h4 = st_["held_out"]["step4"]
        colt = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); colt[zt["rows"][zt["axis_kind"] == 1]] = (60, 60, 60); colt[zt["rows"][zt["axis_kind"] == 2]] = (165, 165, 165); idxt = np.zeros(n, np.uint16); metat = []
        for q, L in enumerate(sorted(set(int(x) for x in zt["leaf"] if x > 0))):
            sel = zt["rows"][zt["leaf"] == L]; c3 = V2t.PAL.get(L, (200, 200, 200)); colt[sel] = c3; idxt[sel] = q + 1
            metat.append(dict(id=f"잎 {L} (엽축 번호{'' if L in [t for t in (1, 3, 4, 5, 6, 7, 8)] else ', 엽축 없음'})", n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">엽축·라벨 조정</button>'
        M["gc"]["sources"].append(dict(key="reconcile", label=f"복엽 (엽축·라벨 조정, 엽축 번호; 보류 사진 원래 정답 IoU {h4['original']['summary']['median_iou']:.2f} / 혼동 {h4['original']['summary']['max_confusion']:.2f}, 게이트 불합격, 보기용)", idx=_b64(idxt), col=_b64(colt), meta=metat))
        SRCT = {"renumbered_vote": ((150, 170, 220), "번호 조정 투표"), "trace_split": ((230, 140, 30), "엽축 분할"), "split_unassigned": ((120, 60, 60), "분할 후 미배정"), "fill": ((40, 190, 90), "엽축 따라 채움"), "unassigned": ((90, 90, 90), "미배정")}
        colo = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); idxo = np.zeros(n, np.uint16); metao = []
        for q, (k, (c3, lbl)) in enumerate(SRCT.items()):
            sel = zt["rows"][zt["source"] == k]; colo[sel] = c3; idxo[sel] = q + 1; metao.append(dict(id=lbl, n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">번호 조정/엽축 분할/채움</button>'
        M["gc"]["sources"].append(dict(key="reconcile_src", label="번호 조정 / 엽축 분할 / 채움", idx=_b64(idxo), col=_b64(colo), meta=metao))
    # V-P clean-up (build/p1_vp_cleanup_20261026_01): absorb detached pieces + fill gaps
    cu = ROOT / "build/p1_vp_cleanup_20261026_01"
    if (cu / "surfel_clusters.npz").exists():
        from tomato3d.organs import vote2d as V2c
        zc = np.load(cu / "surfel_clusters.npz", allow_pickle=True); sc_c = json.load(open(cu / "scores.json", encoding="utf-8")); hc = sc_c["held_out"]["ABCD"]
        colc = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); colc[zc["rows"][zc["axis_kind"] == 1]] = (60, 60, 60); colc[zc["rows"][zc["axis_kind"] == 2]] = (165, 165, 165); idxc = np.zeros(n, np.uint16); metac = []
        for q, L in enumerate(sorted(set(int(x) for x in zc["leaf"] if x > 0))):
            sel = zc["rows"][zc["leaf"] == L]; c3 = V2c.PAL.get(L, (235, 235, 235)) if L < 100 else (235, 235, 235); colc[sel] = c3; idxc[sel] = q + 1
            metac.append(dict(id=(f"잎 {L} ({hc['flags'].get(str(L), '보류 사진에 없음')})" if L < 100 else f"U{L - 100} (기하만, 잎 아님)"), n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">복엽 정리</button>'
        M["gc"]["sources"].append(dict(key="cleanup", label=f"복엽 (정리: 흡수 + 틈 메움 + 나머지 채움 + 기하 완성 D′(섬 크기 상한); 보류 사진 IoU {hc['summary']['median_iou']:.2f} / 혼동 {hc['summary']['max_confusion']:.2f}, 게이트 불합격, 보기용)", idx=_b64(idxc), col=_b64(colc), meta=metac))
        SRCC = {"kept": ((150, 170, 220), "원래"), "absorbed": ((230, 140, 30), "흡수 (다른 잎으로)"), "unset": ((120, 60, 60), "흡수 대상 없음 → 비움"), "gap_filled": ((40, 190, 90), "틈 메움"), "rest_filled": ((200, 220, 60), "나머지 채움"), "geometry_D": ((190, 90, 230), "기하 완성 D (섬 이동·덩어리 귀속)"), "unassigned": ((90, 90, 90), "미배정")}
        colo2 = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); idxo2 = np.zeros(n, np.uint16); metao2 = []
        for q, (k, (c3, lbl)) in enumerate(SRCC.items()):
            sel = zc["rows"][zc["source"] == k]; colo2[sel] = c3; idxo2[sel] = q + 1; metao2.append(dict(id=lbl, n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">흡수/메움/원래</button>'
        M["gc"]["sources"].append(dict(key="cleanup_src", label="흡수 / 메움 / 원래 (정리 대 V-P)", idx=_b64(idxo2), col=_b64(colo2), meta=metao2))
    # stacked sheets (build/p1_stacked_sheets_20261026_01): sheets, named leaves, stack violations; + vertical section slicer
    ss = ROOT / "build/p1_stacked_sheets_20261026_01"; slicer = False
    if (ss / "surfel_clusters.npz").exists():
        from tomato3d.organs import vote2d as V2s
        zz = np.load(ss / "surfel_clusters.npz", allow_pickle=True); sj = json.load(open(ss / "scores.json", encoding="utf-8")); hf = sj["held_out"]["final"]; dv = sj["development"]
        sh = zz["sheet"]; shs = sorted(set(int(x) for x in sh if x >= 0)); pal_s = palette(max(len(shs), 1)); col_s = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); idx_s = np.zeros(n, np.uint16); meta_s = []
        sizes = Counter(int(x) for x in sh if x >= 0); big = [s for s, _ in sizes.most_common(60)]
        for q, s_ in enumerate(shs):
            sel = zz["rows"][sh == s_]; col_s[sel] = pal_s[q]
        for q, s_ in enumerate(big):
            sel = zz["rows"][sh == s_]; idx_s[sel] = q + 1; meta_s.append(dict(id=f"시트 묶음 {s_}", n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % pal_s[shs.index(s_)]))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">층 분리 (시트)</button>'
        M["gc"]["sources"].append(dict(key="sheets", label=f"층 분리 (시트 묶음, 선택 {sj['choice']}; 큰 60개만 목록)", idx=_b64(idx_s), col=_b64(col_s), meta=meta_s))
        lf = zz["leaf"]; col_l = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); col_l[zz["rows"][zz["axis_kind"] == 1]] = (60, 60, 60); col_l[zz["rows"][zz["axis_kind"] == 2]] = (165, 165, 165); idx_l = np.zeros(n, np.uint16); meta_l = []
        for q, L in enumerate(sorted(set(int(x) for x in lf if x > 0))):
            sel = zz["rows"][lf == L]; c3 = V2s.PAL.get(L, (225, 225, 225)) if L < 200 else (225, 225, 225); col_l[sel] = c3; idx_l[sel] = q + 1
            meta_l.append(dict(id=(f"잎 {L} ({hf['flags'].get(str(L), '보류 사진에 없음')}, 겹침 {dv['final']['per_leaf'].get(str(L), dv['final']['per_leaf'].get(L, '—'))})" if L < 200 else f"이름 없는 시트 {L - 200}"), n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">층 분리 (잎 이름)</button>'
        M["gc"]["sources"].append(dict(key="sheets_named", label=f"층 분리 (잎 이름; 겹침 위반 {dv['ABCD_prime']['rate']:.3f} → {dv['final']['rate']:.3f}; 보류 사진 IoU {hf['summary']['median_iou']:.2f} / 혼동 {hf['summary']['max_confusion']:.2f}, {'게이트 합격' if hf['summary']['gate_pass'] else '게이트 불합격, 보기용'})", idx=_b64(idx_l), col=_b64(col_l), meta=meta_l))
        vv = zz["violation"].astype(bool); col_v = (col_l.astype(float) * 0.35).astype(np.uint8); col_v[zz["rows"][vv]] = (235, 40, 40); idx_v = np.zeros(n, np.uint16); idx_v[zz["rows"][vv]] = 1
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">겹침 위반</button>'
        M["gc"]["sources"].append(dict(key="stack_violation", label=f"겹침 위반 (빨강 = 같은 잎 이름이 2–12 mm 위/아래 층에도 있음; 최종 {dv['final']['rate']:.3f})", idx=_b64(idx_v), col=_b64(col_v),
                                       meta=[dict(id="겹침 위반", n=int(vv.sum()), theta="—", h="—", blobs="—", incl="—", colour="#eb2828")]))
        slicer = True
    # thin-shell metrics of R3-1b (build/p1_thin_shell_20261026_01): tail share map, FWHM map, "second layer" source set
    ts = ROOT / "build/p1_thin_shell_20261026_01/metrics/R3-1b_per_surfel.npz"
    if ts.exists():
        zt2 = np.load(ts); fullp = np.full(n, np.nan); fullp[zt2["rows"]] = zt2["beyond"]
        M["gc"]["feats"]["shell_tail"] = dict(label="껍질 꼬리 비율 (±1.5 mm 밖, R3-1b; 0 = 얇음)", col=_b64(_ramp(fullp, 0, 0.6)), lo=0.0, hi=0.6)
        fw_ = np.full(n, np.nan); fw_[zt2["rows"]] = zt2["fwhm"]; M["gc"]["feats"]["shell_fwhm"] = dict(label="껍질 두께 FWHM (mm, R3-1b)", col=_b64(_ramp(fw_, 0.5, 4.0)), lo=0.5, hi=4.0)
        colt2 = np.tile(np.array((70, 70, 70), np.uint8), (n, 1)); idxt2 = np.zeros(n, np.uint16); sl_ = zt2["rows"][zt2["blade"] & zt2["second_layer"]]; tn_ = zt2["rows"][zt2["blade"] & ~zt2["second_layer"]]
        colt2[sl_] = (240, 150, 30); colt2[tn_] = (90, 160, 230); idxt2[sl_] = 1; idxt2[tn_] = 2
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">껍질: 겹층 있음</button>'
        M["gc"]["sources"].append(dict(key="shell_second", label="껍질: 주황 = 가까이(2.5–10 mm)에 별개의 두 번째 층이 있음, 파랑 = 없음 (R3-1b; 표본 약 49 % 있음)", idx=_b64(idxt2), col=_b64(colt2),
                                       meta=[dict(id="두 번째 층 있음", n=int(len(sl_)), theta="—", h="—", blobs="—", incl="—", colour="#f0961e"), dict(id="없음", n=int(len(tn_)), theta="—", h="—", blobs="—", incl="—", colour="#5aa0e6")]))
    # stacked-sheet retry (build/p1_stacked_sheets_retry_20261026_01): sheet patches + patch leaf numbers
    sr = ROOT / "build/p1_stacked_sheets_retry_20261026_01/viz/surfel_clusters.npz"
    if sr.exists():
        from tomato3d.organs import vote2d as V2r2
        zr2 = np.load(sr, allow_pickle=True); mj = json.load(open(sr.parent.parent / "metrics.json", encoding="utf-8")); pt = zr2["patch"]; ps = sorted(set(int(x) for x in pt if x >= 0))
        pal_p = palette(max(len(ps), 1)); colp = np.tile(np.array((70, 70, 70), np.uint8), (n, 1)); idxp = np.zeros(n, np.uint16); metap = []; szp = Counter(int(x) for x in pt if x >= 0)
        for q, p_ in enumerate(ps):
            sel = zr2["rows"][pt == p_]; colp[sel] = pal_p[q]; idxp[sel] = q + 1; metap.append(dict(id=f"패치 {p_}", n=int(szp[p_]), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % pal_p[q]))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">층 재시도 (패치)</button>'
        M["gc"]["sources"].append(dict(key="sheet_retry_patch", label=f"층 분리 재시도: 시트 패치 {len(ps)}개 (가장 큰 패치 = 엽신 76 %, 잎 여러 장), 어두움 = 미배정", idx=_b64(idxp), col=_b64(colp), meta=sorted(metap, key=lambda m_: -m_["n"])[:40]))
        lf2 = zr2["leaf"]; coll = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); idxl2 = np.zeros(n, np.uint16); metal2 = []
        for q, L in enumerate(sorted(set(int(x) for x in lf2 if x > 0))):
            sel = zr2["rows"][lf2 == L]; c3 = V2r2.PAL.get(L, (220, 220, 220)); coll[sel] = c3; idxl2[sel] = q + 1; metal2.append(dict(id=f"잎 {L}", n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        h_ = mj["held_out"]["summary"]
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">층 재시도 (잎 번호)</button>'
        M["gc"]["sources"].append(dict(key="sheet_retry_leaf", label=f"층 분리 재시도: 패치별 투표 잎 번호 (보류 사진 IoU {h_['median_iou']:.2f} / 덮임 {mj['held_out']['coverage']:.2f}, 불합격, 보기용; 회색 = 번호 없음)", idx=_b64(idxl2), col=_b64(coll), meta=metal2))
    # 3D-first SAM merge, second attempt (oversegment/second_attempt): filled 4a patches, consensus groups, proxy leaf numbers
    sa = ROOT / "build/p1_stacked_sheets_retry_20261026_01/oversegment/second_attempt"
    if (sa / "viz_groups.npz").exists():
        from tomato3d.organs import vote2d as V2s
        zg = np.load(sa / "viz_groups.npz"); zpf = np.load(sa / "patches_filled.npz"); rs = json.load(open(sa / "result.json", encoding="utf-8")); hs_ = rs["scoring_proxy_uses_human_numbers"]["post_fill"]
        for key, arr, btn, lab_ in (("os_patch", zpf["patch_filled"], "3D 패치 (채움)", f"3D 먼저: 4a 패치 {int(zpf['patch'].max()) + 1}개 + 층 제약 채움 (미배정 {rs['unassigned']['after'] * 100:.1f} %), 어두움 = 미배정"),
                                    ("os_group", zg["post"], "합의 그룹 (2차)", f"3D 먼저 2차 시도: 합의 그룹 {rs['groups']}개 + 채움 (과합병 소엽 기준 18.2 %, 복엽 기준 3.3 %), 어두움 = 미배정")):
            us = sorted(set(int(x) for x in arr if x >= 0)); pal_s = palette(max(len(us), 1)); cs = np.tile(np.array((70, 70, 70), np.uint8), (n, 1)); ix = np.zeros(n, np.uint16); ms = []; sz = Counter(int(x) for x in arr if x >= 0)
            for q, u_ in enumerate(us):
                sel = zg["rows"][arr == u_]; cs[sel] = pal_s[q]; ix[sel] = q + 1; ms.append(dict(id=f"{u_}", n=int(sz[u_]), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % pal_s[q]))
            v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">{btn}</button>'
            M["gc"]["sources"].append(dict(key=key, label=lab_, idx=_b64(ix), col=_b64(cs), meta=sorted(ms, key=lambda m_: -m_["n"])[:40]))
        pls = zg["proxy_leaf"]; cl_ = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); il_ = np.zeros(n, np.uint16); ml_ = []
        for q, L in enumerate(sorted(set(int(x) for x in pls if x > 0))):
            sel = zg["rows"][pls == L]; c3 = V2s.PAL.get(L, (220, 220, 220)); cl_[sel] = c3; il_[sel] = q + 1; ml_.append(dict(id=f"잎 {L}", n=int(len(sel)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % c3))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">2차 잎 번호 (대리)</button>'
        M["gc"]["sources"].append(dict(key="os_leaf", label=f"3D 먼저 2차: 사람 번호 다수결로 붙인 대리 잎 번호 (scoring_proxy_uses_human_numbers; 보류 IoU {hs_['summary']['median_iou']:.2f} / 혼동 {hs_['summary']['max_confusion']:.2f} / 덮임 {hs_['coverage']:.2f}, 불합격·보기용; 회색 = 번호 없음)", idx=_b64(il_), col=_b64(cl_), meta=ml_))
    # local diffusion + contested resolve (build/p1_local_diffusion_20261027_01): V-P base, diffused lighter, absorbed lighter, still contested white
    ldr = ROOT / "build/p1_local_diffusion_20261027_01/contested_resolve"
    # R6 (whole chain) and R7 (leaf-4 crop propagation -> R6 chain); lighter = added after V-P
    for r6, key6, btn6, ttl6 in (  # R6 set left the page for size on 2026-10-27 (files kept); R7 / R7b below
                                 (ROOT / "build/p1_local_diffusion_20261027_01/vp_leaf4/labels.npz", "chain_r7", "복엽 정리 R7 (잎 4 주변 전파 재시도 → R6 사슬)", "복엽 정리 R7: 잎 4 둘레 25 mm 잘라 SAM 2 전파 재시도 → 새 V-P → R6 사슬"),
                                 ):   # R7b set left the page for size on 2026-10-27 (files kept; same held-out numbers as R6)
        if not r6.exists(): continue
        from tomato3d.organs import vote2d as V2r6
        z6 = np.load(r6); m6 = json.load(open(r6.parent / "metrics.json", encoding="utf-8")); l6 = z6["label"]; b6 = np.load(ROOT / "build/p1_compound_userlabel_20261025_01/map_v3_charuco/user_labelled_propagated/surfel_clusters.npz")
        base6 = np.where(b6["leaf"] > 0, b6["leaf"], 0); ax6 = b6["axis_kind"]; c6 = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); c6[z6["rows"][ax6 == 1]] = (60, 60, 60); c6[z6["rows"][ax6 == 2]] = (165, 165, 165); i6 = np.zeros(n, np.uint16); m6l = []
        for q, L in enumerate(sorted(set(int(x) for x in l6 if x > 0))):
            col3 = np.array(V2r6.PAL.get(L, (220, 220, 220))); keep6 = (l6 == L) & (base6 == L); add6 = (l6 == L) & (base6 != L)
            c6[z6["rows"][keep6]] = col3; c6[z6["rows"][add6]] = (col3 * 0.55 + 255 * 0.45).astype(np.uint8); i6[z6["rows"][keep6 | add6]] = q + 1
            m6l.append(dict(id=f"잎 {L} (V-P {int(keep6.sum())} + 추가 {int(add6.sum())})", n=int((l6 == L).sum()), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % tuple(col3)))
        w6 = z6["rows"][(l6 == 0) & (ax6 == 0)]; c6[w6] = (255, 255, 255); i6[w6] = len(m6l) + 1; m6l.append(dict(id="미배정 (흰색)", n=int(len(w6)), theta="—", h="—", blobs="—", incl="—", colour="#ffffff"))
        h6 = m6["final_held_out"]["summary"] if "final_held_out" in m6 else m6["held_out"]["final"]["summary"]
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">{btn6}</button>'
        M["gc"]["sources"].append(dict(key=key6, label=f"{ttl6} (연한 색 = V-P 뒤 추가, 흰색 = 미배정; 추가분 순도 {m6['pooled_purity_all_added']}; 보류 IoU {h6['median_iou']:.3f} / 혼동 {h6['max_confusion']:.3f}, 게이트 불합격·보기용)",
                                       idx=_b64(i6), col=_b64(c6), meta=m6l))
    # R8_pc: stage-1 -> point-cloud-only absorption (no photo / SAM / V-P); set B chosen on the calibration half; lighter = absorbed, white = unpainted blade
    r8f = ROOT / "build/p1_pc_absorb_20261027_01/R8_pc/labels.npz"
    if r8f.exists():
        from tomato3d.organs import vote2d as V2r8
        z8 = np.load(r8f); l8 = z8["label"].astype(int); s1_ = z8["stage1"].astype(int); r7_ = z8["r7"].astype(int); nd8 = z8["node"]; rj8 = json.load(open(ROOT / "build/p1_pc_absorb_20261027_01/results_final.json", encoding="utf-8"))
        ax8 = np.load(ROOT / "build/p1_compound_userlabel_20261025_01/map_v3_charuco/user_labelled_propagated/surfel_clusters.npz")["axis_kind"]
        c8 = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); c8[z8["rows"][ax8 == 1]] = (60, 60, 60); c8[z8["rows"][ax8 == 2]] = (165, 165, 165); i8 = np.zeros(n, np.uint16); m8 = []
        for q, L in enumerate(sorted(set(int(x) for x in l8 if x > 0))):
            c3 = np.array(V2r8.PAL.get(L, (220, 220, 220))); k8 = (l8 == L) & (s1_ == L); a8 = (l8 == L) & (s1_ != L)
            c8[z8["rows"][k8]] = c3; c8[z8["rows"][a8]] = (c3 * 0.55 + 255 * 0.45).astype(np.uint8); i8[z8["rows"][k8 | a8]] = q + 1
            m8.append(dict(id=f"잎 {L} (1단계 {int(k8.sum())} + 흡수 {int(a8.sum())})", n=int((l8 == L).sum()), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % tuple(c3)))
        w8 = z8["rows"][(l8 <= 0) & (ax8 == 0)]; c8[w8] = (255, 255, 255); i8[w8] = len(m8) + 1; m8.append(dict(id="미배정 엽신 (흰색)", n=int(len(w8)), theta="—", h="—", blobs="—", incl="—", colour="#ffffff"))
        ch8 = rj8["choice"]["report_half"]
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">점군 흡수 R8_pc (기하+색)</button>'
        M["gc"]["sources"].append(dict(key="pc_r8", label=f"점군 흡수 R8_pc: R7 1단계 → 점군만 (기하+색, 사진·SAM 없음, 세트 B) 흡수 (연한 색 = 흡수, 흰색 = 미배정 엽신; 자체 검사 보고 절반 정확도 {ch8['accuracy']:.3f} / 덮임 {ch8['coverage']:.2f}, 기준 0.90 불합격·보기용)", idx=_b64(i8), col=_b64(c8), meta=m8))
        d8 = np.full((n, 3), 70, np.uint8); d8[z8["rows"][ax8 == 1]] = (40, 40, 40); d8[z8["rows"][(ax8 == 2) & (l8 <= 0)]] = (110, 110, 110); i8d = np.zeros(n, np.uint16); m8d = []
        grp = (("새로 칠함 (R7 미배정)", (l8 > 0) & (r7_ <= 0), (60, 220, 90)), ("R7 과 다른 잎", (l8 > 0) & (r7_ > 0) & (l8 != r7_), (230, 50, 200)),
               ("R7 은 칠함 → R8 미배정", (l8 <= 0) & (r7_ > 0), (255, 160, 0)), ("둘 다 미배정", (l8 <= 0) & (r7_ <= 0) & ((ax8 == 0) | nd8), (255, 255, 255)))
        for q, (nm8, mk8, col8) in enumerate(grp):
            d8[z8["rows"][mk8]] = col8; i8d[z8["rows"][mk8]] = q + 1; m8d.append(dict(id=f"{nm8} {int(mk8.sum())}", n=int(mk8.sum()), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % col8))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">점군 흡수 R8_pc: R7 과 차이만</button>'
        M["gc"]["sources"].append(dict(key="pc_r8_diff", label="점군 흡수 R8_pc 대 R7 차이: 초록 = 새로 칠함, 자홍 = R7 과 다른 잎, 주황 = R7 은 칠했는데 R8 미배정, 흰색 = 둘 다 미배정, 어두운 회색 = 같음 (보기용)", idx=_b64(i8d), col=_b64(d8), meta=m8d))
    # R8_pc_rachis: rachis-like (axis_kind 2) surfels absorbed into their compound leaf as its rachis part (no photo / SAM); darker leaf colour = rachis, light grey = still unpainted axis2
    rxf = ROOT / "build/p1_pc_absorb_20261027_01/R8_pc_rachis/labels_rachis.npz"
    if rxf.exists():
        from tomato3d.organs import vote2d as V2x
        zx = np.load(rxf); lx = zx["label"].astype(int); axx = zx["axis_kind"]; tgx = zx["target"]; sx = zx["source"]; rjx = json.load(open(rxf.parent / "results.json", encoding="utf-8")); stx = rjx["self_test"]["pooled"]
        cx = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); cx[zx["rows"][axx == 1]] = (60, 60, 60); cx[zx["rows"][(axx == 2) & (lx <= 0)]] = (165, 165, 165); cx[zx["rows"][(lx <= 0) & (axx == 0)]] = (255, 255, 255); ix = np.zeros(n, np.uint16); mx = []
        for q, L in enumerate(sorted(set(int(v) for v in lx if v > 0))):
            c3 = np.array(V2x.PAL.get(L, (220, 220, 220))); bl = (lx == L) & ~tgx; rx_ = (lx == L) & tgx
            cx[zx["rows"][bl]] = c3; cx[zx["rows"][rx_]] = (c3 * 0.6).astype(np.uint8); ix[zx["rows"][bl | rx_]] = q + 1
            pl_ = rjx["per_leaf"].get(str(L), {}); mx.append(dict(id=f"잎 {L} (엽신 {int(bl.sum())} + 엽축 {int(rx_.sum())}" + (f" · {pl_['rachis_mm2']} mm²" if pl_ else "") + ")", n=int((lx == L).sum()), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % tuple(c3)))
        wx = zx["rows"][(lx <= 0) & (axx == 0)]; ix[wx] = len(mx) + 1; mx.append(dict(id="미배정 엽신 (흰색)", n=int(len(wx)), theta="—", h="—", blobs="—", incl="—", colour="#ffffff"))
        ux = zx["rows"][tgx & (lx <= 0)]; ix[ux] = len(mx) + 1; mx.append(dict(id="못 칠한 엽축 (연회색)", n=int(len(ux)), theta="—", h="—", blobs="—", incl="—", colour="#a5a5a5"))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">엽축 흡수 (R8_pc 위)</button>'
        M["gc"]["sources"].append(dict(key="pc_rachis", label=f"엽축 흡수 R8_pc_rachis: 복엽 주위 엽축 (rachis-like) 서펠 {int((tgx & (lx > 0)).sum())} / {int(tgx.sum())} 를 점군만으로 (3D 주위 + 사람 자취, 사진·SAM 없음) 그 잎의 엽축으로 흡수 (어두운 잎 색 = 흡수된 엽축, 연회색 = 못 칠한 엽축 {int(len(ux))}, 흰색 = 미배정 엽신; 자체 검사 정확도 {stx['accuracy']:.3f} / 오배정 {stx['wrong_rate']:.3f}, 기준 0.05 불합격·보기용)", idx=_b64(ix), col=_b64(cx), meta=mx))
        dx = np.full((n, 3), 70, np.uint8); dx[zx["rows"][axx == 1]] = (40, 40, 40); idd = np.zeros(n, np.uint16); mdx = []
        for q, (nm_x, mk_x, col_x) in enumerate((("사람 엽축 3 mm 안 (씨앗)", tgx & (sx == 2), (30, 150, 60)), ("걷기 흡수", tgx & (sx > 2) & (sx < 10) & (lx > 0), (60, 220, 90)), ("못 칠함", tgx & (lx <= 0), (255, 160, 0)))):
            dx[zx["rows"][mk_x]] = col_x; idd[zx["rows"][mk_x]] = q + 1; mdx.append(dict(id=f"{nm_x} {int(mk_x.sum())}", n=int(mk_x.sum()), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % col_x))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">엽축 흡수: 차이만</button>'
        M["gc"]["sources"].append(dict(key="pc_rachis_diff", label="엽축 흡수 차이: 진한 초록 = 사람 엽축 3 mm 안 (그 잎으로 고정한 씨앗), 밝은 초록 = 걷기로 흡수, 주황 = 못 칠함, 어두운 회색 = 그대로 (보기용)", idx=_b64(idd), col=_b64(dx), meta=mdx))
    # blue-leaf (leaf 4) proposal layers over R7 (not applied): revision 1, revision 2 with the 3D-surroundings evidence; lighter = proposed, white = still unpainted blade
    pdir = ROOT / "build/p1_local_diffusion_20261027_01/proposal_blue_leaf_2026-10-27"
    for sub_p, key_p, btn_p in (("with_3d_surroundings", "prop_blue_r2", "잎 4 둘레 제안 (3D 주위 포함)"),):   # revision-1 set left the page for size on 2026-10-27 (files kept)
        fp = (pdir / sub_p / "proposal.npz") if sub_p else (pdir / "proposal.npz")
        if not fp.exists(): continue
        from tomato3d.organs import vote2d as V2p
        zp = np.load(fp); rp = json.load(open(fp.parent / "result.json", encoding="utf-8")); lp = zp["proposed_label"]; kp = zp["proposal_kind"]; axp = zp["axis_kind"]; ow = zp["proposal_owner"]
        sel_k = (kp > 0) & ((ow == 4) if sub_p else (ow > 0)); cp = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); cp[zp["rows"][axp == 1]] = (60, 60, 60); cp[zp["rows"][axp == 2]] = (165, 165, 165); ip = np.zeros(n, np.uint16); mp = []
        lab_p = np.where(sel_k, lp, zp["r7_label"])
        for q, L in enumerate(sorted(set(int(x) for x in lab_p if x > 0))):
            c3 = np.array(V2p.PAL.get(L, (220, 220, 220))); old_ = (lab_p == L) & ~sel_k; new_ = (lab_p == L) & sel_k
            cp[zp["rows"][old_]] = c3; cp[zp["rows"][new_]] = (c3 * 0.55 + 255 * 0.45).astype(np.uint8); ip[zp["rows"][old_ | new_]] = q + 1
            mp.append(dict(id=f"잎 {L} (R7 {int(old_.sum())} + 제안 {int(new_.sum())})", n=int((lab_p == L).sum()), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % tuple(c3)))
        wp = zp["rows"][(lab_p <= 0) & (axp == 0)]; cp[wp] = (255, 255, 255); ip[wp] = len(mp) + 1; mp.append(dict(id="미배정 엽신 (흰색)", n=int(len(wp)), theta="—", h="—", blobs="—", incl="—", colour="#ffffff"))
        pc = rp["per_candidate"]["4"]; ar = pc["leaf_areas_mm2"]["4"]
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">{btn_p}</button>'
        M["gc"]["sources"].append(dict(key=key_p, label=f"{btn_p}: R7 위 제안 층, 적용 안 함 (연한 색 = 제안: 제안 서펠 {int(sel_k.sum())} (잎 4 기준 축2→엽신 {pc['proposals']['a_axis2_to_blade']}, 흰 엽신 엄격 통과 {pc['proposals']['b_white_strict']}); 잎 4 {ar['before']}→{ar['after']} mm²; 흰색 = 미배정 엽신, 진회색 = 줄기, 연회색 = 엽축; 사용자 확인 전, 게이트 불합격·보기용)",
                                       idx=_b64(ip), col=_b64(cp), meta=mp))
    # user choice 2026-10-27 ("후자로 진행"): photo-first kept in the viewer; the geometry-first set stays on disk only (page size limit 16 MB)
    # user choice 2026-10-27: photo-first -> remaining surfels by xyz distance + photo votes (R3) -> detached pieces cleaned (R4); the R2 enclosure sets left the page (files kept)
    # 2026-10-27 R7 added: the photo-first-only set left the page for size (files kept; the final set below contains it as its first step)
    for sub_, btn_, ttl_ in ():   # R5 "최종" set left the page for size on 2026-10-27 (files kept; R7 holds the same chain on the newer base)
        fz = (ldr / sub_ / "labels.npz") if sub_ else (ldr / "labels.npz")
        if not fz.exists(): continue
        from tomato3d.organs import vote2d as V2d
        zl_ = np.load(fz); mj_ = json.load(open(fz.parent / "metrics.json", encoding="utf-8")); src_ = zl_["source"]; lab_ = zl_["label"]; nm_ = [str(s) for s in zl_["source_names"]]
        ax_ = np.load(ROOT / "build/p1_compound_userlabel_20261025_01/map_v3_charuco/user_labelled_propagated/surfel_clusters.npz")["axis_kind"]
        cl_ = np.tile(np.array((128, 128, 128), np.uint8), (n, 1)); cl_[zl_["rows"][ax_ == 1]] = (60, 60, 60); cl_[zl_["rows"][ax_ == 2]] = (165, 165, 165); il_ = np.zeros(n, np.uint16); ml_ = []
        absorbed = np.isin(src_, [nm_.index(k) for k in ("diffused", "absorb_single_neighbour", "absorb_evidence", "absorb_geometry_only", "absorb_enclosed", "geo_photo", "geo_only", "moved_detached", "bridge_fill") if k in nm_])
        for q, L in enumerate(sorted(set(int(x) for x in lab_ if x > 0))):
            c3 = np.array(V2d.PAL.get(L, (220, 220, 220))); sel = zl_["rows"][(lab_ == L) & ~absorbed]; sel2 = zl_["rows"][(lab_ == L) & absorbed]
            cl_[sel] = c3; cl_[sel2] = (c3 * 0.55 + 255 * 0.45).astype(np.uint8); il_[sel] = q + 1; il_[sel2] = q + 1
            ml_.append(dict(id=f"잎 {L} (흡수·확산 +{len(sel2)})", n=int(len(sel) + len(sel2)), theta="—", h="—", blobs="—", incl="—", colour="#%02x%02x%02x" % tuple(c3)))
        wsel = zl_["rows"][np.isin(src_, [nm_.index(k) for k in ("still_contested", "unset_detached") if k in nm_]) & (lab_ <= 0)]; cl_[wsel] = (255, 255, 255); il_[wsel] = len(ml_) + 1
        ml_.append(dict(id="아직 경합 (흰색)", n=int(len(wsel)), theta="—", h="—", blobs="—", incl="—", colour="#ffffff"))
        if sub_ in ("detached_cleanup", "bridge_cleanup"): hs2 = mj_["held_out"]["after"]["summary"]; pur_ = f"이동 조각 {mj_['purity']['moved_after']} (전파 지도만 쓴 R3 순도 0.61)"
        elif "enclosure" in sub_: hs2 = mj_["held_out"]["after"]["summary"]; pur_ = mj_["purity"]["purity"]
        else: hs2 = mj_["held_out"]["base_plus_absorption"]["summary"]; pur_ = mj_["absorption_purity"].get("pooled_single_plus_evidence", mj_["absorption_purity"].get("pooled"))
        v2btn += f'<button data-gs="{len(M["gc"]["sources"])}">{btn_}</button>'
        M["gc"]["sources"].append(dict(key="resolve_" + (sub_.replace("/", "_") or "photo_first"), label=f"복엽 정리: V-P + 국소 확산 + 경합 해소 ({ttl_}; 연한 색 = 확산·흡수, 흰색 = 아직 경합; 흡수 순도 {pur_}; 보류 IoU {hs2['median_iou']:.3f} / 혼동 {hs2['max_confusion']:.3f}, 게이트 불합격·보기용)",
                                       idx=_b64(il_), col=_b64(cl_), meta=ml_))
    M["ctr"] = (lo + rng / 2).tolist(); M["note"] = f"{STATUS}: round-2 rachis-connected compound clusters ({plaus.get('candidate')} {plaus.get('tag')}); stem-like dark grey, rachis-like light grey, unassigned mid-grey; {LO.SCALE_TAG}"
    html = html[:m.start(1)] + json.dumps(M, separators=(",", ":")) + html[m.end(1):]
    title = f"{STATUS} · 복엽 엽축 연결 (2차)"; html = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", html); html = re.sub(r'<h1 id="ttl">[^<]*</h1>', f'<h1 id="ttl">{title}</h1>', html)
    html = re.sub(r'(z-index:50;background:#7a1f1f;[^>]*>)[^<]*(</div>)', lambda mm: mm.group(1) + f"{STATUS} · 복엽 = 엽축으로 연결한 기하 군집 (골격·소엽 라벨 미사용, {plaus.get('tag')}) · 순위·축·부착 = unavailable · 형태만, 척도 미검증 · 계측 아님" + mm.group(2), html, count=1)
    r5btn = '<button data-gs="2">사용자 씨앗 R5</button>' if (out / "r31b_r5" / "surfel_clusters.npz").exists() else ''
    ui = ('<label>복엽 색 (출처)</label><div class="row" id="gcsrc"><button data-gs="-1" class="on">끔 (사진 색)</button><button data-gs="0">엽축 연결 (2차)</button><button data-gs="1">기하 군집 (1차)</button>' + r5btn + v2btn + '</div>' + flipped_html +
          '<label>특징 지도 (한 번에 하나, 선택된 형식)</label><div class="row" id="fmrow"><button data-fm="" class="on">끔</button></div>'
          '<div id="gcleg" style="display:none"><div style="font-size:11px;opacity:.8"><span style="display:inline-block;width:10px;height:10px;background:#3c3c3c"></span> stem-like &nbsp;<span style="display:inline-block;width:10px;height:10px;background:#a5a5a5"></span> rachis-like &nbsp;<span style="display:inline-block;width:10px;height:10px;background:#808080"></span> unassigned · <label style="display:inline"><input type="checkbox" id="showbr" checked> 다리 표시</label></div>'
          '<div id="gclist" style="max-height:160px;overflow:auto;font-size:11px"></div><div id="gcpanel" style="font-size:12px;border-top:1px solid var(--line);margin-top:6px;padding-top:6px">군집을 클릭하면 정보가 여기 나옵니다.</div>'
          '<div id="gcans" style="font-size:12px;margin-top:6px"></div><div id="gcans1" style="font-size:11px;opacity:.75"></div><p style="font-size:11px;opacity:.75;margin:4px 0 0">사람 점검 (~10분): 2차 군집마다 하나를 고르세요 (db cluster_answers_r2).</p></div>'
          '<label>엽축 그리기 (잎 번호: 홀수 = 개발, 짝수 = 보류 — 계약 §7)</label><div class="row" id="trrow"><button id="trOn">그리기 켜기</button><input id="trLeaf" type="number" min="1" value="1" style="width:54px;font:inherit;font-size:12px" title="잎 번호"><button id="trUnc">이 구간 불확실</button><button id="trUndo">점 취소</button><button id="trDone">엽축 완료</button></div><div id="trinfo" style="font-size:11px;opacity:.8">밑동(중심 쪽)부터 끝소엽까지 서펠을 차례로 클릭 — 보이는 것만, 추측 없음. 저장 db rachis_traces.</div>'
          + '<label>표본 칠하기 (Q1 사진 특징용, ~10개씩)</label><div class="row" id="sprow"><button data-sp="">끔</button><button data-sp="stem">stem</button><button data-sp="rachis">rachis/petiole</button><button data-sp="blade">blade</button><button data-sp="erase">지우개</button></div><div id="spinfo" style="font-size:11px;opacity:.8">모드를 고른 뒤 서펠을 클릭하면 저장됩니다 (db spot_labels).</div>'
          '<div class="row" style="margin-top:4px"><label style="display:inline"><input type="checkbox" id="showmk" checked> 칠한 점·엽축 표시</label><button id="spUndo">마지막 칠한 점 지움</button></div><div style="font-size:11px;opacity:.8"><span style="color:#ff5a36">●</span> stem &nbsp;<span style="color:#ffd400">●</span> rachis &nbsp;<span style="color:#22d3ee">●</span> blade &nbsp;· 노랑 선 = 엽축(실선 홀수 / 점선 짝수), 주황 = 그리는 중</div>')
    html = html.replace('<label>불투명도 배율', ui + '<label>불투명도 배율', 1)
    js = r"""<script>(function(){const G=M.gc;const RGB=C;let cur=-1,hl=null,fm='',spot='';const TEX={};let db=null;const ANS={},SPOTS={};
function build(si){const s=G.sources[si];const idx=new Uint16Array(b64(s.idx).buffer);const col=b64(s.col);const C2=new Uint8Array(TW*TH*4);for(let i=0;i<N;i++){let r=col[i*3],g=col[i*3+1],b=col[i*3+2];const k=idx[i];if(k>0&&hl!==null&&hl!==k-1){r=g=b=205;}C2[i*4]=r;C2[i*4+1]=g;C2[i*4+2]=b;C2[i*4+3]=255;}return {idx,C2};}
function buildF(name){const col=b64(G.feats[name].col);const C2=new Uint8Array(TW*TH*4);for(let i=0;i<N;i++){C2[i*4]=col[i*3];C2[i*4+1]=col[i*3+1];C2[i*4+2]=col[i*3+2];C2[i*4+3]=255;}return C2;}
const _um=gl.uniformMatrix4fv.bind(gl);gl.uniformMatrix4fv=function(loc,t,m){window.__lastMVP=m;return _um(loc,t,m);};
function apply(){if(fm){tex(3,buildF(fm),false);document.getElementById('gcleg').style.display='none';}else if(cur<0){tex(3,RGB,false);document.getElementById('gcleg').style.display='none';}else{const t=build(cur);TEX[cur]=t;tex(3,t.C2,false);document.getElementById('gcleg').style.display='block';list();}
 document.querySelectorAll('#gcsrc button').forEach(b=>b.className=(+b.dataset.gs===cur&&!fm)?'on':'');document.querySelectorAll('#fmrow button').forEach(b=>b.className=(b.dataset.fm===fm)?'on':'');if(window.__viewer)window.__viewer.set({cmode:0});}
document.getElementById('fmrow').innerHTML='<button data-fm="" class="on">끔</button>'+Object.entries(G.feats).map(([k,f])=>`<button data-fm="${k}" title="${f.lo.toFixed(1)}–${f.hi.toFixed(1)}">${f.label}</button>`).join('');
function list(){const s=G.sources[cur];document.getElementById('gclist').innerHTML=`<div style="margin:4px 0">${s.label} · ${s.meta.length}</div>`+s.meta.map((m,i)=>`<div data-gc="${i}" style="cursor:pointer;display:flex;gap:6px;align-items:center;padding:1px 0;${hl===i?'font-weight:600':''}"><i style="width:10px;height:10px;background:${m.colour};flex:none"></i>${m.id} <span style="opacity:.7">${m.n} · blobs ${m.blobs}${cur===0&&ANS[m.id]?' · '+ANS[m.id].answer:''}</span></div>`).join('');}
function panel(i){const s=G.sources[cur];const m=s.meta[i];document.getElementById('gcpanel').innerHTML=`<b>${m.id}</b> · ${s.label}<br>surfels ${m.n} · 방위 ${m.theta} · 높이 ${m.h}<br>blobs ${m.blobs} · 평균 경사 ${m.incl}°<br>다리 ${G.bridges.length}개 (전체; 판정 특징: ${[...new Set(G.bridges.flatMap(b=>b.tests.map(t=>t[0])))].join(', ')||'—'})`;
 document.getElementById('gcans').innerHTML=cur===0?G.answers.map(a=>`<button data-ans="${a}" class="${ANS[m.id]&&ANS[m.id].answer===a?'on':''}" style="margin:2px">${a}</button>`).join(''):'';document.getElementById('gcans').dataset.id=m.id;document.getElementById('gcans1').textContent=cur===0?'1차 답(같은 자리 비교용): 1차 뷰어 db 에 답이 없음':'';}
document.getElementById('gcsrc').addEventListener('click',e=>{const b=e.target.closest('[data-gs]');if(!b)return;cur=+b.dataset.gs;fm='';hl=null;apply();});
document.getElementById('fmrow').addEventListener('click',e=>{const b=e.target.closest('[data-fm]');if(!b)return;fm=b.dataset.fm;apply();});
document.getElementById('gclist').addEventListener('click',e=>{const d=e.target.closest('[data-gc]');if(!d)return;const i=+d.dataset.gc;hl=(hl===i)?null:i;apply();panel(i);});
document.getElementById('gcans').addEventListener('click',async e=>{const b=e.target.closest('[data-ans]');if(!b)return;const id=document.getElementById('gcans').dataset.id;ANS[id]={answer:b.dataset.ans,at:new Date().toISOString()};if(db){try{await db.doc('cluster_answers_r2/'+id).set(ANS[id]);}catch(err){}}else{try{localStorage.setItem('gc2_ans',JSON.stringify(ANS));}catch(err){}}panel(G.sources[cur].meta.findIndex(m=>m.id===id));list();});
document.getElementById('sprow').addEventListener('click',e=>{const b=e.target.closest('[data-sp]');if(!b)return;spot=b.dataset.sp;document.querySelectorAll('#sprow button').forEach(x=>x.className=x.dataset.sp===spot?'on':'');document.getElementById('spinfo').textContent=spot==='erase'?'지우개: 칠한 점·엽축(선 위)·그리는 중인 점을 클릭하면 지웁니다':spot?`모드 ${spot}: 서펠을 클릭하면 저장 (현재 ${Object.keys(SPOTS).length}개)`:'끔';});
document.querySelectorAll('#cm button').forEach(b=>b.addEventListener('click',()=>{if(+b.dataset.cm===0){if(fm)tex(3,buildF(fm),false);else if(cur>=0)tex(3,TEX[cur]?TEX[cur].C2:build(cur).C2,false);}}));
function pick(e){if(!window.__lastMVP)return -1;const r=cv.getBoundingClientRect();const mx=(e.clientX-r.left)*(cv.width/r.width),my=(e.clientY-r.top)*(cv.height/r.height);const M4=window.__lastMVP;let best=-1,bd=144,bz=1e9;
 for(let i=0;i<N;i++){if(qk[i]!==1)continue;const x=PX[i*3],y=PX[i*3+1],z=PX[i*3+2];const cw=M4[3]*x+M4[7]*y+M4[11]*z+M4[15];if(cw<=0)continue;const cx=(M4[0]*x+M4[4]*y+M4[8]*z+M4[12])/cw,cy=(M4[1]*x+M4[5]*y+M4[9]*z+M4[13])/cw;const sx=(cx+1)/2*cv.width,sy=(1-cy)/2*cv.height;const d=(sx-mx)*(sx-mx)+(sy-my)*(sy-my);if(d<bd&&(d<bd-20||cw<bz)){bd=d;best=i;bz=cw;}}return best;}
let TR={on:false,pts:[],surf:[],unc:[],n:0};const TRACES={};
document.getElementById('trOn').onclick=()=>{TR.on=!TR.on;document.getElementById('trOn').className=TR.on?'on':'';document.getElementById('trinfo').textContent=TR.on?'그리기 중: 서펠을 밑동부터 클릭':'끔';};
document.getElementById('trUnc').onclick=()=>{if(TR.unc.length)TR.unc[TR.unc.length-1]=true;document.getElementById('trinfo').textContent='마지막 구간 = 불확실';};
document.getElementById('trUndo').onclick=()=>{TR.pts.pop();TR.surf.pop();TR.unc.pop();document.getElementById('trinfo').textContent='점 '+TR.pts.length;};
document.getElementById('trDone').onclick=async()=>{if(TR.pts.length<2){document.getElementById('trinfo').textContent='점이 2개 이상 필요';return;}const leaf=+document.getElementById('trLeaf').value||0;const k='leaf'+leaf+'_'+Date.now().toString(36);const rec={leaf,points:TR.pts,surfels:TR.surf,uncertain:TR.unc,at:new Date().toISOString()};TRACES[k]=rec;if(db){try{await db.doc('rachis_traces/'+k).set(rec);}catch(err){}}else{try{localStorage.setItem('gc2_traces',JSON.stringify(TRACES));}catch(err){}}TR={on:TR.on,pts:[],surf:[],unc:[],n:0};document.getElementById('trinfo').textContent='저장: 잎 '+leaf+' ('+rec.points.length+'점, '+(leaf%2?'개발':'보류')+' 절반) · 총 '+Object.keys(TRACES).length;};
cv.addEventListener('click',async e=>{const i=pick(e);if(i<0)return;if(TR.on&&spot!=='erase'){const w=[D.lo[0]+qp[i*3]/65535*D.rng[0],D.lo[1]+qp[i*3+1]/65535*D.rng[1],D.lo[2]+qp[i*3+2]/65535*D.rng[2]].map(v=>Math.round(v*1e5)/1e5);TR.pts.push(w);TR.surf.push(i);if(TR.pts.length>1)TR.unc.push(false);document.getElementById('trinfo').textContent='점 '+TR.pts.length+' (서펠 '+i+')';return;}if(spot&&spot!=='erase'){const k='s'+i;SPOTS[k]={surfel:i,label:spot,at:new Date().toISOString()};if(db){try{await db.doc('spot_labels/'+k).set(SPOTS[k]);}catch(err){}}else{try{localStorage.setItem('gc2_spots',JSON.stringify(SPOTS));}catch(err){}}document.getElementById('spinfo').textContent=`저장: 서펠 ${i} = ${spot} · `+spotCounts();return;}
 if(cur<0||fm)return;const t=TEX[cur]||build(cur);if(t.idx[i]>0)panel(t.idx[i]-1);});
try{Object.assign(ANS,JSON.parse(localStorage.getItem('gc2_ans')||'{}'));Object.assign(SPOTS,JSON.parse(localStorage.getItem('gc2_spots')||'{}'));}catch(e){}
(async()=>{try{if(window.claude&&window.claude.use){db=await claude.use('db');if(db){for(const [col,obj] of [['cluster_answers_r2',ANS],['spot_labels',SPOTS],['rachis_traces',TRACES]]){const q=await db.collection(col).get();q.docs.forEach(d=>{obj[d.id]=d.data();});}if(cur===0)list();}}}catch(e){}})();

const mk=document.createElement('canvas');mk.id='mkov';mk.style.cssText='position:fixed;pointer-events:none;z-index:5';document.body.appendChild(mk);const mkx=mk.getContext('2d');let mkKey='';
const SPC={stem:'#ff5a36',rachis:'#ffd400',blade:'#22d3ee'};
function proj(i){const M4=window.__lastMVP;if(!M4)return null;const x=PX[i*3],y=PX[i*3+1],z=PX[i*3+2];const cw=M4[3]*x+M4[7]*y+M4[11]*z+M4[15];if(cw<=0)return null;return [((M4[0]*x+M4[4]*y+M4[8]*z+M4[12])/cw+1)/2*cv.width,(1-(M4[1]*x+M4[5]*y+M4[9]*z+M4[13])/cw)/2*cv.height];}
function dot(p,col,r,ring){mkx.beginPath();mkx.arc(p[0],p[1],r,0,6.2832);mkx.fillStyle=col;mkx.fill();mkx.lineWidth=Math.max(1.5,r*0.35);mkx.strokeStyle=ring||'#fff';mkx.stroke();}
function line(arr,col,dash,w){const pp=arr.map(proj);mkx.beginPath();let st=false;for(const q of pp){if(!q){st=false;continue;}if(!st){mkx.moveTo(q[0],q[1]);st=true;}else mkx.lineTo(q[0],q[1]);}mkx.setLineDash(dash);mkx.lineWidth=w;mkx.strokeStyle=col;mkx.stroke();mkx.setLineDash([]);return pp;}
function drawMk(){const r=cv.getBoundingClientRect();mk.style.left=r.left+'px';mk.style.top=r.top+'px';mk.style.width=r.width+'px';mk.style.height=r.height+'px';if(mk.width!==cv.width)mk.width=cv.width;if(mk.height!==cv.height)mk.height=cv.height;mkx.clearRect(0,0,mk.width,mk.height);
 if(!document.getElementById('showmk').checked)return;const k=Math.max(1,cv.width/Math.max(r.width,1));const R=5*k;
 for(const t of Object.values(TRACES)){const idx=t.surfels||[];if(idx.length<2)continue;const pp=line(idx,'rgba(255,212,0,.95)',t.leaf%2?[]:[8*k,6*k],2.5*k);pp.forEach((q,j)=>{if(q)dot(q,t.uncertain&&t.uncertain[j-1]?'#fff':'#ffd400',R*0.7,'#000');});if(pp[0]){mkx.font=(13*k)+'px sans-serif';mkx.fillStyle='#fff';mkx.strokeStyle='#000';mkx.lineWidth=3*k;const lab='잎 '+t.leaf+(t.leaf%2?'':' (보류)');mkx.strokeText(lab,pp[0][0]+R,pp[0][1]-R);mkx.fillText(lab,pp[0][0]+R,pp[0][1]-R);}}
 if(TR.surf.length){const pp=line(TR.surf,'rgba(255,140,0,.95)',[4*k,3*k],2.5*k);pp.forEach(q=>{if(q)dot(q,'#ff8c00',R*0.7,'#fff');});}
 for(const sp of Object.values(SPOTS)){const q=proj(sp.surfel);if(q)dot(q,SPC[sp.label]||'#fff',R);}}
(function loop(){const M4=window.__lastMVP;const key=(M4?Array.prototype.join.call(M4,','):'')+'|'+cv.width+'x'+cv.height+'|'+Object.keys(SPOTS).length+'|'+Object.keys(TRACES).length+'|'+TR.surf.length+'|'+(document.getElementById('showmk')||{}).checked+'|'+cv.getBoundingClientRect().left;if(key!==mkKey){mkKey=key;drawMk();}requestAnimationFrame(loop);})();
function spotCounts(){const c={stem:0,rachis:0,blade:0};Object.values(SPOTS).forEach(s=>{c[s.label]=(c[s.label]||0)+1;});return 'stem '+c.stem+' · rachis '+c.rachis+' · blade '+c.blade;}
document.getElementById('spUndo').onclick=async()=>{const ks=Object.keys(SPOTS).sort((a,b)=>(SPOTS[a].at||'')<(SPOTS[b].at||'')?-1:1);const k=ks[ks.length-1];if(!k)return;delete SPOTS[k];let fail='';if(db){try{await db.doc('spot_labels/'+k).delete();}catch(err){fail=' ⚠ 저장소 삭제 실패: '+(err.code||err.message)+' — 다시 열면 되살아날 수 있음';}}else{try{localStorage.setItem('gc2_spots',JSON.stringify(SPOTS));}catch(err){}}document.getElementById('spinfo').textContent='지움 · '+spotCounts()+fail;};

async function delDoc(col,k){if(db){try{await db.doc(col+'/'+k).delete();}catch(err){document.getElementById('spinfo').textContent='⚠ 화면에서는 지웠지만 저장소 삭제 실패: '+(err.code||err.message)+' — 다시 열면 되살아날 수 있음';}}else{try{localStorage.setItem('gc2_spots',JSON.stringify(SPOTS));localStorage.setItem('gc2_traces',JSON.stringify(TRACES));}catch(err){}}}
function segDist(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1];const L=dx*dx+dy*dy;let t=L?((p[0]-a[0])*dx+(p[1]-a[1])*dy)/L:0;t=Math.max(0,Math.min(1,t));return Math.hypot(p[0]-(a[0]+t*dx),p[1]-(a[1]+t*dy));}
cv.addEventListener('click',async e=>{if(spot!=='erase')return;const r=cv.getBoundingClientRect();const k=cv.width/Math.max(r.width,1);const m=[(e.clientX-r.left)*k,(e.clientY-r.top)*k];const lim=14*k;let best=null;
 const cand=(d,kind,key,extra)=>{if(d<=lim&&(!best||d<best.d))best={d,kind,key,extra};};
 for(const [key,sp] of Object.entries(SPOTS)){const q=proj(sp.surfel);if(q)cand(Math.hypot(q[0]-m[0],q[1]-m[1]),'spot',key);}
 for(const [key,t] of Object.entries(TRACES)){const pp=(t.surfels||[]).map(proj);for(let j=0;j<pp.length;j++){if(!pp[j])continue;cand(Math.hypot(pp[j][0]-m[0],pp[j][1]-m[1]),'trace',key);if(j&&pp[j-1])cand(segDist(m,pp[j-1],pp[j]),'trace',key);}}
 for(let j=0;j<TR.surf.length;j++){const q=proj(TR.surf[j]);if(q)cand(Math.hypot(q[0]-m[0],q[1]-m[1]),'draft',String(j));}
 if(!best){document.getElementById('spinfo').textContent='지울 것이 근처에 없음 (점·선 위를 클릭)';return;}
 if(best.kind==='spot'){delete SPOTS[best.key];await delDoc('spot_labels',best.key);document.getElementById('spinfo').textContent='칠한 점 지움 · '+spotCounts();}
 else if(best.kind==='trace'){const lf=TRACES[best.key].leaf;delete TRACES[best.key];await delDoc('rachis_traces',best.key);document.getElementById('trinfo').textContent='엽축 지움: 잎 '+lf+' · 남은 '+Object.keys(TRACES).length;document.getElementById('spinfo').textContent='엽축 지움 (잎 '+lf+')';}
 else{const j=+best.key;TR.pts.splice(j,1);TR.surf.splice(j,1);if(TR.unc.length)TR.unc.splice(Math.max(0,j-1),1);document.getElementById('trinfo').textContent='그리는 중 점 지움 · 남은 '+TR.pts.length;}
});
window.__gc2={screenOf:(i)=>{const q=proj(i);if(!q)return null;const r=cv.getBoundingClientRect();return [r.left+q[0]*r.width/cv.width,r.top+q[1]*r.height/cv.height];},plantSample:(n)=>{const a=[];for(let i=0;i<N;i++)if(qk[i]===1)a.push(i);const st=Math.max(1,Math.floor(a.length/n));return a.filter((_,j)=>j%st===0).slice(0,n);},spots:()=>SPOTS,traces:()=>TRACES,trace:TR,set:(s)=>{cur=s;fm='';apply();},feat:(k)=>{fm=k;apply();},state:()=>({cur,fm,hl,sources:G.sources.map(s=>s.meta.length),feats:Object.keys(G.feats),bridges:G.bridges.length,answers:Object.keys(ANS).length,spots:Object.keys(SPOTS).length})};if(window.__viewer)window.__viewer.draw();})();</script>"""
    html = html.replace("</body>", js + "</body>", 1) if "</body>" in html else html + js
    if slicer:                                                       # vertical section slicer: dims everything outside a slab through the pot-centre vertical
        sl_ui = ('<label>단면 (세로 절단면, 화분 중심 지나는 평면)</label><div class="row"><label style="display:inline"><input type="checkbox" id="slOn"> 켜기</label>'
                 '<input id="slAz" type="range" min="0" max="179" value="0" style="width:120px"> <span id="slAzT">0°</span>'
                 '<input id="slTh" type="range" min="0.5" max="6" step="0.5" value="1.5" style="width:90px"> <span id="slThT">±1.5 mm</span></div>')
        html = html.replace('<label>불투명도 배율', sl_ui + '<label>불투명도 배율', 1)
        sl_js = (r"""<script>(function(){const CX=%f,CY=%f;const WX=new Float32Array(N),WY=new Float32Array(N);for(let i=0;i<N;i++){WX[i]=D.lo[0]+qp[i*3]/65535*D.rng[0];WY[i]=D.lo[1]+qp[i*3+1]/65535*D.rng[1];}
const SL={on:false,az:0,th:1.5};const _tex=tex;tex=function(u,a,f){if(u===3&&SL.on&&!f){const c=Math.cos(SL.az*Math.PI/180),s=Math.sin(SL.az*Math.PI/180),t=SL.th/1000;a=a.slice();for(let i=0;i<N;i++){const d=(WX[i]-CX)*(-s)+(WY[i]-CY)*c;if(Math.abs(d)>t){a[i*4]=a[i*4]*0.12;a[i*4+1]=a[i*4+1]*0.12;a[i*4+2]=a[i*4+2]*0.12;}}}return _tex(u,a,f);};
let last=null;const _set=window.__gc2.set;window.__gc2.set=(s)=>{last=s;_set(s);};function re(){const st=window.__gc2.state();if(st.fm)window.__gc2.feat(st.fm);else window.__gc2.set(st.cur);}
document.getElementById('slOn').onchange=e=>{SL.on=e.target.checked;re();};document.getElementById('slAz').oninput=e=>{SL.az=+e.target.value;document.getElementById('slAzT').textContent=SL.az+'°';if(SL.on)re();};
document.getElementById('slTh').oninput=e=>{SL.th=+e.target.value;document.getElementById('slThT').textContent='±'+SL.th+' mm';if(SL.on)re();};window.__slicer=SL;})();</script>""" % (POT_CENTRE[0], POT_CENTRE[1]))
        html = html.replace("</body>", sl_js + "</body>", 1)
    html = add_tool_a(html, sf, rows, ak, M)
    html = add_leafbox_overlay(html)
    (out / "viewer").mkdir(exist_ok=True); p = out / "viewer" / "compound_cluster_r2_viewer.html"; p.write_text(html, encoding="utf-8"); return p


def add_leafbox_overlay(html):
    """Session-A compound-leaf boxes drawn as a 2D overlay (edges, petiole base, rachis line) on the shared viewer; toggle "복엽 상자 (세션 A)". Read-only; a few KB."""
    f = ROOT / "build/p1_leafbox_20261029_01/R3-1b/boxes.json"
    if not f.exists(): return html
    from tomato3d.organs import vote2d as V2b
    bj = json.load(open(f, encoding="utf-8")); res = json.load(open(f.parent / "results.json", encoding="utf-8")); boxes = []
    for lf in bj["leaves"]:
        b = lf["box"]; c = np.array(b["centre"]); R = np.array(b["R"]); h = np.array(b["half_extent_mm"])
        cor = [(c + sx * h[0] * R[0] + sy * h[1] * R[1] + sz * h[2] * R[2]) / 1000.0 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        big = max(lf["overlap_with"], key=lambda o: o["volume_share"]) if lf["overlap_with"] else None
        boxes.append(dict(leaf=lf["leaf_id"], col="#%02x%02x%02x" % V2b.PAL.get(int(lf["leaf_id"]), (220, 220, 220)), cor=np.round(cor, 5).tolist(), poly=np.round(np.array(lf["rachis_polyline"]) / 1000.0, 5).tolist(),
                          base=np.round(np.array(lf["petiole_base_xyz"]) / 1000.0, 5).tolist(), n=lf["n_surfels"], area=lf["area_b_mm2_range"], sigma=lf["petiole_base_sigma_mm"],
                          big=(f"잎 {big['leaf_id']} {big['volume_share']:.2f}" if big else "없음")))
    note = f"세션 A 복엽 상자 (R8_pc 씨앗, 겹침 허용, 소속은 규칙으로) · {res['tag']} · 배정 {res['assigned']} / 경합 {res['contested']} · 빨간 점 = 잎자루 밑, 노란 선 = 엽축"
    ui = '<label style="display:block;margin-top:4px"><input type="checkbox" id="lbOn"> 복엽 상자 (세션 A)</label><div id="lbInfo" style="font-size:11px;opacity:.85;display:none"></div>'
    html = html.replace('<label>불투명도 배율', ui + '<label>불투명도 배율', 1)
    js = LB_JS.replace("__LB__", json.dumps(boxes, ensure_ascii=False)).replace("__NOTE__", json.dumps(note, ensure_ascii=False))
    return html.replace("</body>", js + "</body>", 1) if "</body>" in html else html + js


LB_JS = r"""<script>(function(){const LB=__LB__;const NOTE=__NOTE__;const CT=M.ctr;const gl=document.querySelector('canvas');const ov=document.createElement('canvas');ov.style.cssText='position:fixed;pointer-events:none;z-index:6';document.body.appendChild(ov);const x=ov.getContext('2d');
const E=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
function pj(p){const M4=window.__lastMVP;if(!M4)return null;const X=p[0]-CT[0],Y=-(p[1]-CT[1]),Z=-(p[2]-CT[2]);const cw=M4[3]*X+M4[7]*Y+M4[11]*Z+M4[15];if(cw<=0)return null;return [((M4[0]*X+M4[4]*Y+M4[8]*Z+M4[12])/cw+1)/2*gl.width,(1-(M4[1]*X+M4[5]*Y+M4[9]*Z+M4[13])/cw)/2*gl.height];}
let key='';function draw(){const on=document.getElementById('lbOn').checked;const r=gl.getBoundingClientRect();ov.style.left=r.left+'px';ov.style.top=r.top+'px';ov.style.width=r.width+'px';ov.style.height=r.height+'px';if(ov.width!==gl.width)ov.width=gl.width;if(ov.height!==gl.height)ov.height=gl.height;x.clearRect(0,0,ov.width,ov.height);if(!on)return;const k=Math.max(1,gl.width/Math.max(r.width,1));
for(const b of LB){const q=b.cor.map(pj);x.strokeStyle=b.col;x.lineWidth=2*k;x.beginPath();for(const [a,c] of E){if(q[a]&&q[c]){x.moveTo(q[a][0],q[a][1]);x.lineTo(q[c][0],q[c][1]);}}x.stroke();
const pp=b.poly.map(pj);x.strokeStyle='rgba(255,212,0,.95)';x.lineWidth=2*k;x.beginPath();let st=false;for(const p of pp){if(!p){st=false;continue;}if(!st){x.moveTo(p[0],p[1]);st=true;}else x.lineTo(p[0],p[1]);}x.stroke();
const bp=pj(b.base);if(bp){x.beginPath();x.arc(bp[0],bp[1],4.5*k,0,6.2832);x.fillStyle='#ff3030';x.fill();x.lineWidth=1.5*k;x.strokeStyle='#fff';x.stroke();x.font=(13*k)+'px sans-serif';x.lineWidth=3*k;x.strokeStyle='#000';x.fillStyle=b.col;x.strokeText('잎 '+b.leaf,bp[0]+6*k,bp[1]-6*k);x.fillText('잎 '+b.leaf,bp[0]+6*k,bp[1]-6*k);}}}
document.getElementById('lbOn').onchange=e=>{const i=document.getElementById('lbInfo');i.style.display=e.target.checked?'block':'none';i.innerHTML=NOTE+'<br>'+LB.map(b=>'<span style="color:'+b.col+'">■</span> 잎 '+b.leaf+': 서펠 '+b.n+', 면적 '+b.area[0]+'–'+b.area[2]+' mm², 밑 σ '+b.sigma+' mm, 최대 겹침 '+b.big).join('<br>');key='';};
(function loop(){const M4=window.__lastMVP;const r=gl.getBoundingClientRect();const k2=(M4?Array.prototype.join.call(M4,','):'')+'|'+gl.width+'x'+gl.height+'|'+r.left+'|'+r.top+'|'+document.getElementById('lbOn').checked;if(k2!==key){key=k2;draw();}requestAnimationFrame(loop);})();})();</script>"""



KS_MAX = 0.35
KS_PAIRS_Q1 = {"height": ("h", "h_rel"), "radius": ("r", "r_rel")}


def r31b_axis_features():
    P = load_r31b(); ax = np.nonzero(P["is_axis"])[0]
    H = float(np.percentile(P["xyz"][:, 2], 99) - POT_TOP); R = float(np.percentile(np.linalg.norm(P["xyz"][:, :2] - POT_CENTRE, axis=1), 99))
    return axis_point_features(P["xyz"][ax], P["xyz"][ax], H, R)


def apply_ks_q1(forms, table, pf):
    """Contract R2 section 3 / section 9: a chosen geometry form with KS(DEV, R3-1b) >= 0.35 is replaced by the other form; if both fail, the feature is dropped."""
    from scipy.stats import ks_2samp
    r3 = r31b_axis_features(); dev = {k: np.concatenate([p_["AF_all"][k] for p_ in pf]) for k in ("h", "h_rel", "r", "r_rel", "vert", "thick_rel", "dens")}
    ks = {k: float(ks_2samp(dev[k], r3[k]).statistic) for k in dev}; out_forms = []; new_table = []
    for row in table:
        row = dict(row); name = row["feature"]
        if row["form"] is None:
            new_table.append(row); continue
        f = row["form"]; row["ks_chosen"] = round(ks[f], 3)
        if ks[f] >= KS_MAX:
            other = [x for x in KS_PAIRS_Q1.get(name, ()) if x != f]
            if other and ks[other[0]] < KS_MAX:
                row.update(form=other[0], chosen=("A" if other[0] in ("h", "r", "vert") else "R"), ks_note=f"{f} KS {ks[f]:.3f} >= 0.35 -> other form {other[0]} KS {ks[other[0]]:.3f}")
            else:
                row.update(form=None, chosen="dropped_ks", ks_note=f"{f} KS {ks[f]:.3f}" + (f", {other[0]} KS {ks[other[0]]:.3f}" if other else "") + " -> both >= 0.35, dropped")
        if row["form"]:
            out_forms.append(row["form"])
        new_table.append(row)
    return out_forms, new_table


# ================================================================ R5: user-traced rachises (contract R2 section 7)
TRACE_SNAP_MM = 2.0          # trace point -> axis surfel snap (declared in section 7)
TRACE_STEP_MM = 0.5          # densification of a trace polyline for snapping / distances


def load_traces(folder: Path):
    out = []
    for f in sorted(Path(folder).glob("*.json")):
        d = json.load(open(f, encoding="utf-8")); d = d.get("data", d)
        out.append(dict(key=f.stem, leaf=int(d["leaf"]), pts=np.array(d["points"], float), surf=list(d.get("surfels", [])), unc=list(d.get("uncertain", []))))
    return out


def densify(pts, step=TRACE_STEP_MM / 1000):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def poly_dist(Q, pts):
    """Distance of points Q to a polyline; returns (distance, segment index)."""
    Q = np.atleast_2d(Q); best = np.full(len(Q), np.inf); seg = np.zeros(len(Q), int)
    if len(pts) == 1:
        return np.linalg.norm(Q - pts[0], axis=1), seg
    for j, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        ab = b - a; L = max(float(ab @ ab), 1e-18); t = np.clip(((Q - a) @ ab) / L, 0, 1); d = np.linalg.norm(Q - (a + t[:, None] * ab), axis=1)
        m = d < best; best[m] = d[m]; seg[m] = j
    return best, seg


def line_point_ls(lines):
    """Least-squares point nearest to a set of 3D lines (point, direction)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p, u in lines:
        u = u / max(np.linalg.norm(u), 1e-12); M = np.eye(3) - np.outer(u, u); A += M; b += M @ p
    return np.linalg.lstsq(A, b, rcond=None)[0]


def dist_to_line(q, p0, d):
    d = d / np.linalg.norm(d); v = np.atleast_2d(q) - p0; return np.linalg.norm(v - np.outer(v @ d, d), axis=1)


def flank_stats(labels_b, blob, cents, bl_count, polys, lim_m):
    """Blobs whose centroid lies within lim_m of any of the leaf's polylines; share of their surfels in the majority cluster."""
    d = np.min([poly_dist(cents, p)[0] for p in polys], axis=0) if len(cents) else np.zeros(0)
    sel_blobs = np.nonzero(d <= lim_m)[0]
    if not len(sel_blobs):
        return dict(blobs=0, surfels=0, assigned_share=None, majority_cluster=None, majority_share=None)
    m = np.isin(blob, sel_blobs); lab = labels_b[m]; asg = lab[lab >= 0]
    if not len(asg):
        return dict(blobs=int(len(sel_blobs)), surfels=int(m.sum()), assigned_share=0.0, majority_cluster=None, majority_share=None)
    c, k = Counter(asg.tolist()).most_common(1)[0]
    return dict(blobs=int(len(sel_blobs)), surfels=int(m.sum()), assigned_share=float(len(asg) / m.sum()), majority_cluster=int(c), majority_share=float(k / len(asg)))


def run_r5(out: Path = None):
    out = Path(out or OUT2); sub = out / "r31b_r5"; sub.mkdir(parents=True, exist_ok=True)
    q1, forms2, _ = q1_and_forms(out)
    dev = json.load(open(out / "dev_results_r2.json", encoding="utf-8"))["results"]; params = dev["R4"]["best"]["params"]
    traces = load_traces(out / "r31b" / "traces_export" / "rachis_traces"); devT = [t for t in traces if t["leaf"] % 2 == 1]; heldT = [t for t in traces if t["leaf"] % 2 == 0]
    P = load_r31b(); xyz = P["xyz"]
    # pass 1: the Q1 re-fit RANSAC axis
    c0 = {}; run_round2(P, "R4", params, q1, forms2, c0); p_r, d_r, info_r = c0["axis_refit"]
    # stem position from the development traces: lines through each trace's first two points (first point = base, as declared)
    lines = [(t["pts"][0], t["pts"][0] - t["pts"][1]) for t in devT if len(t["pts"]) >= 2]
    x_star = line_point_ls(lines) if len(lines) >= 2 else None
    axes = {"ransac_q1_refit": (p_r, d_r), "pot_centre_vertical": (np.array([POT_CENTRE[0], POT_CENTRE[1], POT_TOP]), np.array([0, 0, 1.0]))}
    if x_star is not None:
        axes["trace_fit_vertical"] = (np.array([x_star[0], x_star[1], POT_TOP]), np.array([0, 0, 1.0]))
    held_bases = np.array([t["pts"][0] for t in heldT]) if heldT else np.zeros((0, 3))
    axis_score = {k: (float(dist_to_line(held_bases, p, d).mean() * 1000) if len(held_bases) else None) for k, (p, d) in axes.items()}
    chosen = min((k for k in axes if axis_score[k] is not None), key=lambda k: axis_score[k]) if len(held_bases) else "ransac_q1_refit"
    orient = []
    for t in traces:                                                   # reported only: does the first point lie nearer the chosen axis than the last one?
        p, d = axes[chosen]; d0, d1 = dist_to_line(t["pts"][0], p, d)[0], dist_to_line(t["pts"][-1], p, d)[0]
        orient.append(dict(key=t["key"], leaf=t["leaf"], first_mm=round(d0 * 1000, 1), last_mm=round(d1 * 1000, 1), first_is_base=bool(d0 <= d1)))
    # pass 2: automatic R4 with the chosen axis (the leaves R5 does not seed)
    c1 = {}; pred4, info4, ex4 = run_round2(P, "R4", params, q1, forms2, c1, axis_override=(None if chosen == "ransac_q1_refit" else axes[chosen]))
    bl = np.nonzero(~P["is_axis"])[0]; ax = np.nonzero(P["is_axis"])[0]; stem_like = ex4["stem_like"]; rach = ax[~stem_like]; blob = c1["blob"]; F = c1["F"]
    lab4 = pred4[bl].copy(); K = int(lab4.max()) + 1 if (lab4 >= 0).any() else 0
    # seeds: rachis-like surfels within 2 mm of each development leaf's densified traces; grow through the 3 mm rachis-like graph; contested -> none
    leaves_dev = sorted({t["leaf"] for t in devT}); rtree = cKDTree(xyz[rach]) if len(rach) else None; comp_r = components_of(xyz[rach]) if len(rach) else np.zeros(0, int)
    grown = {}
    for L in leaves_dev:
        dens = np.vstack([densify(t["pts"]) for t in devT if t["leaf"] == L]); hit = sorted({j for js in rtree.query_ball_point(dens, TRACE_SNAP_MM / 1000) for j in js}) if rtree is not None else []
        grown[L] = set(np.nonzero(np.isin(comp_r, comp_r[hit]))[0].tolist()) if hit else set(); grown[L] = (grown[L], len(hit))
    allg = Counter(j for g, _ in grown.values() for j in g); contested_r = {j for j, c in allg.items() if c >= 2}
    # blade attach to development leaves: bilateral test against the traced polyline (R3 rules, frozen R4 params)
    nb = int(blob.max()) + 1; cents = np.array([xyz[bl[blob == b]].mean(0) for b in range(nb)]) if nb else np.zeros((0, 3))
    lim = params.get("att_mult", 1.5) * R2["leaflet_mm"] / 1000; p_ax, d_ax = axes[chosen]; seeded = {}
    for b in range(nb):
        idx = np.nonzero(blob == b)[0]
        if len(idx) < 20: continue
        Q = xyz[bl[idx]]; cb = cents[b]; w, v = np.linalg.eigh(np.cov((Q - cb).T)); mid = v[:, -1]; best = None
        for L in leaves_dev:
            for t in (t for t in devT if t["leaf"] == L):
                if len(t["pts"]) < 2: continue
                dd, sg = poly_dist(cb, t["pts"]); dd = float(dd[0]); sg = int(sg[0])
                if dd > lim: continue
                dirv = t["pts"][sg + 1] - t["pts"][sg]; dirv /= max(np.linalg.norm(dirv), 1e-12)
                ang = float(np.degrees(np.arccos(min(1.0, abs(float(mid @ dirv))))))
                ends = [t["pts"][0], t["pts"][-1]]; distal = max(ends, key=lambda e: dist_to_line(e, p_ax, d_ax)[0])
                terminal = np.linalg.norm(cb - distal) <= R2["leaflet_mm"] / 1000
                if ang < R2["midrib_deg"][0] and not terminal: continue
                hz = densify(t["pts"]); hr = hz[np.argmin(np.linalg.norm(hz - cb, axis=1)), 2]
                if abs(cb[2] - hr) * 1000 > 2 * params["h_resid_mm"]: continue
                if best is None or dd < best[0]: best = (dd, L)
        if best is not None:
            seeded.setdefault(best[1], []).append(b)
    lab5 = lab4.copy(); seed_id = {L: K + j for j, L in enumerate(leaves_dev)}
    for L, bs in seeded.items():
        lab5[np.isin(blob, bs)] = seed_id[L]
    keep = sorted(set(lab5[lab5 >= 0].tolist())); ren = {c: i for i, c in enumerate(keep)}; lab5r = np.array([ren.get(int(c), -1) for c in lab5])
    seeded_new = {L: ren[seed_id[L]] for L in leaves_dev if seed_id[L] in ren}
    # held-out scoring (contract section 7): Q1 on the traces, Q2 flanking blobs on R5 and on the automatic R4
    held_rows = []; atree = cKDTree(xyz[ax])
    for L in sorted({t["leaf"] for t in heldT}):
        polys = [t["pts"] for t in heldT if t["leaf"] == L]; dens = np.vstack([densify(p) for p in polys]); dd, jj = atree.query(dens); ok = dd <= TRACE_SNAP_MM / 1000
        q1_share = float((~stem_like[jj[ok]]).mean()) if ok.any() else None
        held_rows.append(dict(leaf=L, traces=len(polys), trace_points_snapped=int(ok.sum()), trace_points=int(len(dens)), q1_rachis_like_share=q1_share,
                              r5=flank_stats(lab5r, blob, cents, len(bl), polys, lim), r4_auto=flank_stats(lab4, blob, cents, len(bl), polys, lim)))
    def merged_pairs(key):
        ms = [r[key]["majority_cluster"] for r in held_rows]; pairs = [(held_rows[i]["leaf"], held_rows[j]["leaf"]) for i in range(len(ms)) for j in range(i + 1, len(ms)) if ms[i] is not None and ms[i] == ms[j]]
        return dict(pairs_total=len(ms) * (len(ms) - 1) // 2, merged_pairs=pairs)
    dev_rows = [dict(leaf=L, traces=sum(1 for t in devT if t["leaf"] == L), seed_rachis_surfels_snapped=grown[L][1], grown_rachis_surfels=len(grown[L][0] - contested_r), blobs_attached=len(seeded.get(L, [])),
                     blade_surfels=int(np.isin(blob, seeded.get(L, [])).sum())) for L in leaves_dev]
    scores = dict(status=STATUS, scale=LO.SCALE_TAG, grouping="user_seeded (development leaves) + geometry_clusters R4 (others)", r4_params=params,
                  traces=dict(total=len(traces), development_leaves=leaves_dev, held_out_leaves=sorted({t["leaf"] for t in heldT}), orientation=orient),
                  stem_axis=dict(candidates={k: dict(point=[round(float(v), 5) for v in axes[k][0]], direction=[round(float(v), 4) for v in axes[k][1]], mean_dist_to_heldout_bases_mm=(round(axis_score[k], 2) if axis_score[k] is not None else None)) for k in axes},
                                 chosen=chosen, rule="smallest mean distance to the held-out trace base ends (first points)", ransac_info=info_r),
                  contested_rachis_surfels=len(contested_r), development=dev_rows, held_out=held_rows, held_out_merges_r5=merged_pairs("r5"), held_out_merges_r4=merged_pairs("r4_auto"),
                  clusters_r4=int(K), clusters_r5=len(keep), note="held-out leaves are scored; development traces only seed; agreement with traces is human-referenced but the traces are not truth of membership")
    json.dump(scores, open(sub / "scores_r5.json", "w", encoding="utf-8"), indent=1, default=float)
    rows = []
    for c in keep:
        m = lab5r == ren[c]; th = F["theta"][m]; th0 = circ_mean_deg(th); dth = (th - th0 + 180) % 360 - 180
        Ls = [L for L, i in seeded_new.items() if i == ren[c]]
        rows.append(dict(cluster_id=(f"US{Ls[0]:02d}" if Ls else f"RC{ren[c] + 1:02d}"), n_surfels=int(m.sum()), theta_mean_deg=round(th0, 1), theta_spread_p5_p95_deg=round(float(np.percentile(dth, 95) - np.percentile(dth, 5)), 1),
                         radial_length_mm=round(float(np.percentile(F["r"][m], 95) - np.percentile(F["r"][m], 5)) * 1000, 1), h_min_mm=round(float(F["h"][m].min()) * 1000, 1), h_max_mm=round(float(F["h"][m].max()) * 1000, 1),
                         mean_inclination_deg=round(float(F["incl"][m].mean()), 1), blobs=len(set(blob[m].tolist()) - {-1}), **dict(LABELS, grouping=("user_seeded" if Ls else "geometry_clusters"))))
    with open(sub / "clusters.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {STATUS}; {LO.SCALE_TAG}; R5: US = user_seeded leaves (development traces), RC = automatic R4; rank/axis/attachment unavailable; blobs are not leaflets\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["cluster_id"], lineterminator="\n"); w.writeheader(); w.writerows(rows)
    full = np.full(len(xyz), -1); full[bl] = lab5r; axis_kind = np.zeros(len(xyz), np.int8); axis_kind[ax[stem_like]] = 1; axis_kind[ax[~stem_like]] = 2
    np.savez_compressed(sub / "surfel_clusters.npz", rows=P["rows"], cluster=full, axis_kind=axis_kind, blob=_full(blob, bl, len(xyz)), theta=_full(F["theta"], bl, len(xyz), np.nan), h=_full(F["h"], bl, len(xyz), np.nan), incl=_full(F["incl"], bl, len(xyz), np.nan), dens=_full(F["dens"], bl, len(xyz), np.nan))
    return scores


# ================================================================ tool A: two-click rachis path on the axis-surfel graph (contract R2 section 10)
TOOL_A = dict(snap_px=15, knn=10, edge_max_mm=8.0, gap_mm=3.0, gap_mult=3.0, stem_mult=1.5)


def axis_path_graph(xyz_ply, rows, axis_kind):
    """Axis surfels (ply index), their kind (1 stem-like, 2 rachis-like) and undirected kNN edges with the declared costs."""
    sel = np.nonzero(axis_kind > 0)[0]; ax = rows[sel]; kind = axis_kind[sel].astype(np.uint8); P = xyz_ply[ax]
    k = min(TOOL_A["knn"] + 1, len(P)); dd, ii = cKDTree(P).query(P, k=k); ea, eb, ec = [], [], []
    for j in range(1, k):
        m = (dd[:, j] <= TOOL_A["edge_max_mm"] / 1000) & (np.arange(len(P)) < ii[:, j])
        a = np.nonzero(m)[0]; b = ii[m, j]; d = dd[m, j]
        cost = d * np.where((kind[a] == 1) | (kind[b] == 1), TOOL_A["stem_mult"], 1.0) * np.where(d > TOOL_A["gap_mm"] / 1000, TOOL_A["gap_mult"], 1.0)
        ea.append(a); eb.append(b); ec.append(cost)
    ea, eb, ec = np.concatenate(ea), np.concatenate(eb), np.concatenate(ec)
    key = np.unique(np.c_[ea, eb], axis=0, return_index=True)[1]
    return ax.astype(np.uint32), kind, ea[key].astype(np.uint32), eb[key].astype(np.uint32), (ec[key] * 1000).astype(np.float32)


TOOL_A_UI = ('<label>엽축 그리기 2 — 두 번 클릭 (아래 잎부터 번호, 수락하면 +1)</label>'
             '<div class="row" id="t2row"><button id="axOnly">축만 보기</button><button id="t2On">도구 켜기</button>'
             '<span style="font-size:12px">잎</span><input id="t2Leaf" type="number" min="1" value="1" style="width:54px;font:inherit;font-size:12px">'
             '<button id="t2Mid">중간점 추가</button><button id="t2Unc">구간 불확실</button><button id="t2Undo">되돌리기</button><button id="t2Acc">수락</button><button id="t2Rej">거절</button></div>'
             '<div id="t2info" style="font-size:11px;opacity:.8">켜고 시작점(엽병이 줄기에 닿는 곳 또는 가장 안쪽)과 끝점(끝소엽 밑)을 클릭하면 경로가 채워집니다. 15 px 안의 축 서펠에 붙음. 구간 불확실 = 경로 위 두 점 클릭.</div>')

TOOL_A_JS = r"""
var TOOLV=0;const T2=G.tool;const AXI=new Uint32Array(b64(T2.ax).buffer);const AXK=b64(T2.kind);const EA=new Uint32Array(b64(T2.ea).buffer),EB=new Uint32Array(b64(T2.eb).buffer),EC=new Float32Array(b64(T2.ec).buffer);const NA=AXI.length;
const DEG=new Uint32Array(NA+1);for(let e=0;e<EA.length;e++){DEG[EA[e]+1]++;DEG[EB[e]+1]++;}for(let j=0;j<NA;j++)DEG[j+1]+=DEG[j];const ADJ=new Uint32Array(DEG[NA]),ADC=new Float32Array(DEG[NA]);{const f=DEG.slice(0,NA);for(let e=0;e<EA.length;e++){ADJ[f[EA[e]]]=EB[e];ADC[f[EA[e]]++]=EC[e];ADJ[f[EB[e]]]=EA[e];ADC[f[EB[e]]++]=EC[e];}}
const AXM=new Uint8Array(N);for(let j=0;j<NA;j++)AXM[AXI[j]]=1;const QK0=qk.slice();let axOnly=false;
function dijk(s,t){const dist=new Float64Array(NA).fill(Infinity),prev=new Int32Array(NA).fill(-1);dist[s]=0;const H=[[0,s]];
 const push=(x)=>{H.push(x);let i=H.length-1;while(i>0){const p=(i-1)>>1;if(H[p][0]<=H[i][0])break;[H[p],H[i]]=[H[i],H[p]];i=p;}};
 const pop=()=>{const top=H[0],last=H.pop();if(H.length){H[0]=last;let i=0;for(;;){const l=2*i+1,r=l+1;let m=i;if(l<H.length&&H[l][0]<H[m][0])m=l;if(r<H.length&&H[r][0]<H[m][0])m=r;if(m===i)break;[H[m],H[i]]=[H[i],H[m]];i=m;}}return top;};
 while(H.length){const [d,u]=pop();if(d>dist[u])continue;if(u===t)break;for(let q=DEG[u];q<DEG[u+1];q++){const v=ADJ[q],nd=d+ADC[q];if(nd<dist[v]){dist[v]=nd;prev[v]=u;push([nd,v]);}}}
 if(!isFinite(dist[t]))return null;const path=[t];while(path[path.length-1]!==s)path.push(prev[path[path.length-1]]);return path.reverse();}
var TL={on:false,mode:'path',wps:[],path:[],unc:new Set(),uncA:-1,hover:-1,hist:[]};
function snapshot(){TL.hist.push({wps:TL.wps.slice(),unc:[...TL.unc]});if(TL.hist.length>50)TL.hist.shift();}
function info(t){document.getElementById('t2info').textContent=t;}
function recompute(){TL.path=[];if(TL.wps.length<2){TOOLV++;return true;}for(let w=0;w+1<TL.wps.length;w++){const seg=dijk(TL.wps[w],TL.wps[w+1]);if(!seg){info('경로 없음: 축 서펠이 8 mm 넘게 끊겨 있음 — 다른 점을 고르거나 되돌리기');TOOLV++;return false;}TL.path=TL.path.concat(w?seg.slice(1):seg);}
 for(const k of [...TL.unc])if(k>=TL.path.length)TL.unc.delete(k);TOOLV++;return true;}
function projW(i){const M4=window.__lastMVP;if(!M4)return null;const x=PX[i*3],y=PX[i*3+1],z=PX[i*3+2];const cw=M4[3]*x+M4[7]*y+M4[11]*z+M4[15];if(cw<=0)return null;return [((M4[0]*x+M4[4]*y+M4[8]*z+M4[12])/cw+1)/2*cv.width,(1-(M4[1]*x+M4[5]*y+M4[9]*z+M4[13])/cw)/2*cv.height,cw];}
function snapAxis(cx,cy){const r=cv.getBoundingClientRect();const k=cv.width/Math.max(r.width,1);const m=[(cx-r.left)*k,(cy-r.top)*k];const lim=T2.snap_px*k;let best=-1,bd=Infinity,bz=Infinity;
 for(let j=0;j<NA;j++){const q=projW(AXI[j]);if(!q)continue;const d=Math.hypot(q[0]-m[0],q[1]-m[1]);if(d>lim)continue;if(d<bd-2*k||(Math.abs(d-bd)<=2*k&&q[2]<bz)){bd=d;bz=q[2];best=j;}}return best;}
function nearestPathIdx(j){const q0=projW(AXI[j]);let bi=-1,bd=Infinity;TL.path.forEach((n,i)=>{const q=projW(AXI[n]);if(!q||!q0)return;const d=Math.hypot(q[0]-q0[0],q[1]-q0[1]);if(d<bd){bd=d;bi=i;}});return bi;}
function world(i){return [D.lo[0]+qp[i*3]/65535*D.rng[0],D.lo[1]+qp[i*3+1]/65535*D.rng[1],D.lo[2]+qp[i*3+2]/65535*D.rng[2]].map(v=>Math.round(v*1e5)/1e5);}
function setBtn(){document.getElementById('t2On').className=TL.on?'on':'';document.getElementById('t2Mid').className=TL.mode==='mid'?'on':'';document.getElementById('t2Unc').className=TL.mode==='unc'?'on':'';document.getElementById('axOnly').className=axOnly?'on':'';}
document.getElementById('axOnly').onclick=()=>{axOnly=!axOnly;for(let i=0;i<N;i++)qk[i]=axOnly?AXM[i]:QK0[i];if(window.__viewer)window.__viewer.set({plant:axOnly});setBtn();TOOLV++;};
document.getElementById('t2On').onclick=()=>{TL.on=!TL.on;if(TL.on){TR.on=false;spot='';document.getElementById('trOn').className='';document.querySelectorAll('#sprow button').forEach(x=>x.className='');info('시작점을 클릭하세요 (잎 '+document.getElementById('t2Leaf').value+')');}else{TL.hover=-1;info('꺼짐');}setBtn();TOOLV++;};
document.getElementById('t2Mid').onclick=()=>{if(TL.path.length<2){info('먼저 시작·끝을 찍어 경로를 만드세요');return;}TL.mode=TL.mode==='mid'?'path':'mid';info(TL.mode==='mid'?'경로가 지나가야 할 점을 클릭':'');setBtn();};
document.getElementById('t2Unc').onclick=()=>{if(TL.path.length<2){info('먼저 경로를 만드세요');return;}TL.mode=TL.mode==='unc'?'path':'unc';TL.uncA=-1;info(TL.mode==='unc'?'불확실 구간의 한쪽 끝을 경로 위에서 클릭':'');setBtn();};
document.getElementById('t2Undo').onclick=()=>{const h=TL.hist.pop();if(!h){info('되돌릴 것이 없음');return;}TL.wps=h.wps;TL.unc=new Set(h.unc);TL.mode='path';TL.uncA=-1;recompute();setBtn();info('되돌림 · 점 '+TL.wps.length+'개');};
document.getElementById('t2Rej').onclick=()=>{if(!TL.wps.length)return;snapshot();TL.wps=[];TL.path=[];TL.unc=new Set();TL.mode='path';TOOLV++;setBtn();info('거절: 이 잎을 처음부터 (잎 번호 그대로)');};
document.getElementById('t2Acc').onclick=async()=>{if(TL.path.length<2){info('수락할 경로가 없음');return;}const leaf=+document.getElementById('t2Leaf').value||0;const surf=TL.path.map(j=>AXI[j]);
 const rec={leaf,points:surf.map(world),surfels:surf,uncertain:TL.path.slice(1).map((_,s)=>TL.unc.has(s)&&TL.unc.has(s+1)),method:'two_click_path',waypoints:TL.wps.map(j=>AXI[j]),at:new Date().toISOString()};
 const key='leaf'+leaf+'_'+Date.now().toString(36);TRACES[key]=rec;if(db){try{await db.doc('rachis_traces/'+key).set(rec);}catch(err){info('⚠ 저장 실패: '+(err.code||err.message));return;}}else{try{localStorage.setItem('gc2_traces',JSON.stringify(TRACES));}catch(err){}}
 TL.hist=[];TL.wps=[];TL.path=[];TL.unc=new Set();TL.mode='path';document.getElementById('t2Leaf').value=leaf+1;TOOLV++;setBtn();info('수락: 잎 '+leaf+' ('+surf.length+'점) · 다음 잎 '+(leaf+1)+' 시작점을 클릭');};
cv.addEventListener('mousemove',e=>{if(!TL.on||e.buttons)return;const j=snapAxis(e.clientX,e.clientY);if(j!==TL.hover){TL.hover=j;TOOLV++;}});
cv.addEventListener('click',e=>{if(!TL.on)return;const j=snapAxis(e.clientX,e.clientY);if(j<0){info('15 px 안에 축 서펠이 없음 — 축만 보기로 켜고 선 위를 클릭');return;}
 if(TL.mode==='mid'){snapshot();const pi=nearestPathIdx(j);let seg=0;for(let w=0;w+1<TL.wps.length;w++){const a=TL.path.indexOf(TL.wps[w]),b=TL.path.indexOf(TL.wps[w+1]);if(pi>=Math.min(a,b)&&pi<=Math.max(a,b)){seg=w;break;}}TL.wps.splice(seg+1,0,j);TL.mode='path';if(recompute())info('중간점 추가 · 경로 '+TL.path.length+'점');setBtn();return;}
 if(TL.mode==='unc'){const pi=nearestPathIdx(j);if(pi<0)return;if(TL.uncA<0){TL.uncA=pi;info('다른 쪽 끝을 클릭');TOOLV++;return;}snapshot();for(let s=Math.min(TL.uncA,pi);s<=Math.max(TL.uncA,pi);s++)TL.unc.add(s);TL.uncA=-1;TL.mode='path';TOOLV++;setBtn();info('불확실 구간 표시 · '+TL.unc.size+'점');return;}
 if(TL.wps.length>=2){info('수락·거절하거나 중간점 추가를 쓰세요');return;}snapshot();TL.wps.push(j);if(TL.wps.length===1){TOOLV++;info('끝점(끝소엽 밑)을 클릭');}else if(recompute())info('경로 '+TL.path.length+'점 — 수락 / 거절 / 중간점 추가 / 구간 불확실');});
function drawTool(k,R){if(!TL||(!TL.on&&!TL.path.length))return;const P_=TL.path.map(j=>AXI[j]);if(P_.length>1){line(P_,'rgba(0,229,255,.95)',[],3*k);for(let s=0;s+1<TL.path.length;s++)if(TL.unc.has(s)&&TL.unc.has(s+1))line([P_[s],P_[s+1]],'#ffffff',[5*k,4*k],3.5*k);}
 TL.wps.forEach(j=>{const q=proj(AXI[j]);if(q)dot(q,'#ff3df2',R*0.9,'#fff');});if(TL.uncA>=0&&TL.path[TL.uncA]!==undefined){const q=proj(AXI[TL.path[TL.uncA]]);if(q)dot(q,'#ffffff',R*0.8,'#000');}
 if(TL.on&&TL.hover>=0){const q=proj(AXI[TL.hover]);if(q){mkx.beginPath();mkx.arc(q[0],q[1],9*k,0,6.2832);mkx.lineWidth=2.5*k;mkx.strokeStyle='#ff3df2';mkx.stroke();}}}
"""


def add_tool_a(html, sf, rows, axis_kind, M):
    """Wire tool A into a built round-2 viewer html: graph data, UI row, JS, draw hook, test helpers."""
    ax, kind, ea, eb, ec = axis_path_graph(sf["xyz"], rows, axis_kind)
    tool = dict(ax=_b64(ax), kind=_b64(kind), ea=_b64(ea), eb=_b64(eb), ec=_b64(ec), snap_px=TOOL_A["snap_px"], params=TOOL_A)
    m = re.search(r'<script id="D" type="application/json">(.*?)</script>', html, re.S); D_ = json.loads(m.group(1)); D_["gc"]["tool"] = tool
    html = html[:m.start(1)] + json.dumps(D_, separators=(",", ":")) + html[m.end(1):]
    for old, new in (("<label>엽축 그리기 (잎 번호", TOOL_A_UI + "<label>엽축 그리기 (잎 번호"),
                     ("for(const sp of Object.values(SPOTS)){const q=proj(sp.surfel);if(q)dot(q,SPC[sp.label]||'#fff',R);}}", "for(const sp of Object.values(SPOTS)){const q=proj(sp.surfel);if(q)dot(q,SPC[sp.label]||'#fff',R);}if(typeof drawTool==='function')drawTool(k,R);}"),
                     ("+'|'+cv.getBoundingClientRect().left;", "+'|'+cv.getBoundingClientRect().left+'|'+(typeof TOOLV!=='undefined'?TOOLV:0);"),
                     ("window.__gc2={screenOf", TOOL_A_JS + "window.__gc2={t2:{TL:()=>TL,axi:(n)=>{const st=Math.max(1,Math.floor(NA/n));const a=[];for(let j=0;j<NA;j+=st)a.push(j);return a;},axPly:(j)=>AXI[j],dijk,snap:snapAxis,axOnly:()=>axOnly,qkSum:()=>{let c=0;for(let i=0;i<N;i++)c+=qk[i];return c;}},screenOf")):
        assert old in html, old[:50]; html = html.replace(old, new, 1)
    return html
