"""Re-cluster the V-P compound leaves with geometry + photo features: fill unlabelled blade, fix leaf-to-leaf boundaries.

Contract: build/p1_vp_recluster_20261026_01/CONTRACT.md. Anchors = V-P labels with share >= 0.8 and >= 3 views (never changed); the
rest is decided by a pair classifier (M1), Potts smoothing on the 3 mm graph by ICM (M2) and shape limits (M3), with an abstain rule.
All choices on a 5-fold spatial-block internal check (whole user-labelled leaflets hidden); the held-out photo is scored once.

usage: python -m tomato3d.organs.vp_recluster --step photo_inv|run
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import light_organs as LO
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2

ROOT = LO.ROOT
OUT = ROOT / "build/p1_vp_recluster_20261026_01"
UL_OUT = UL.OUT; MAP_FILE = UL_OUT / "labels" / "number_map_v3_charuco.json"; VP_DIR = UL_OUT / "map_v3_charuco" / "user_labelled_propagated"
STATUS = "R3-1b_unaccepted"; GROUPING = "user_labelled_propagated_reclustered"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping=GROUPING)
# declared (contract sections 1-4)
ANCHOR = dict(share_min=0.8, views_min=3)
FOLDS = dict(n=5, seed=0)
GRAPH = dict(k=10, r_mm=3.0, dens_med_mm=10.0)
CAND_MM = 37.2; LEAFLET_MM = 18.6
FORM = dict(auc_min=0.60, inv_min=0.90)
PERTURB = {"exp0.7": dict(gain=0.7), "exp1.3": dict(gain=1.3), "gam0.8": dict(gamma=0.8), "gam1.25": dict(gamma=1.25),
           "wbA": dict(wb=(1.1, 1.0, 0.9)), "wbB": dict(wb=(0.9, 1.0, 1.1))}
ABSTAIN = dict(P_min=(0.5, 0.6, 0.7), M_min=(0.1, 0.2))
POTTS = dict(lam=(0.5, 1.0, 2.0), sweeps=10, angle_deg=30.0)
GATE = UL.GATE; USABLE = dict(iou_min=0.55, conf_max=0.10)


# ---------------------------------------------------------------- photo features and their invariance (contract section 2)
def photo_features(perturb=None):
    """Same procedure as compound_cluster.photo_features_r31b (per-view L*a*b* at the projected pixel, per-view shading fit, within-capture
    z-scores), with an optional photometric perturbation applied to every photo first. Returns L_resid_z, a_z, chroma_z over Track A rows."""
    import cv2
    from tomato3d.fuse3d import surfels_io as sio
    from tomato3d.organs import seed2d as S
    from tomato3d.recon import charuco_a3 as ca
    P = CC.load_r31b(); xyz = P["xyz"]; nrm = P["normal"] / np.maximum(np.linalg.norm(P["normal"], axis=1, keepdims=True), 1e-12)
    sc = sio.read_surfels_full(LO.DEFAULTS["surfels"])["scale"][P["rows"]]; po = json.load(open(LO.DEFAULTS["poses"])); views = json.load(open(LO.DEFAULTS["views"]))["train"]
    V = S.votes_per_view(xyz, po, views, S.VIS["res"], S.VIS["tol_m"], sc=sc, mode="splat"); res = 1512
    K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res); n = len(xyz); L = np.full((n, len(views)), np.nan, np.float32); A = L.copy(); B = L.copy()
    for k, v in enumerate(views):
        vis = np.nonzero(V[:, k] >= 0)[0]
        if not len(vis): continue
        img = np.asarray(Image.open(LO.DEFAULTS["poses"].parent / "undist_3024" / v).convert("RGB").resize((res, res), Image.BILINEAR)).astype(np.float32) / 255
        if perturb:
            img = img * perturb.get("gain", 1.0)
            if "gamma" in perturb: img = np.clip(img, 0, 1) ** perturb["gamma"]
            if "wb" in perturb: img = img * np.array(perturb["wb"], np.float32)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8); lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"]); xc = xyz[vis] @ R.T + t; z = xc[:, 2]
        u = np.clip(np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int), 0, res - 1); w = np.clip(np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int), 0, res - 1)
        px_ = lab[w, u]; L[vis, k] = px_[:, 0] * 100 / 255; A[vis, k] = px_[:, 1] - 128; B[vis, k] = px_[:, 2] - 128
        cam = -(R.T @ t); d = cam - xyz[vis]; d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12); cos = np.abs((nrm[vis] * d).sum(1))
        Xf = np.c_[np.ones(len(vis)), cos]; coef, *_ = np.linalg.lstsq(Xf, L[vis, k], rcond=None); L[vis, k] = L[vis, k] - Xf @ coef
    def z_(x):
        m = np.isfinite(x); mu, sd = np.nanmean(x[m]), np.nanstd(x[m]) + 1e-9; return np.where(m, (x - mu) / sd, np.nan)
    Lres = np.nanmedian(L, axis=1); a_ = np.nanmedian(A, axis=1); b_ = np.nanmedian(B, axis=1)
    return dict(L_resid_z=z_(Lres), a_z=z_(a_), chroma_z=z_(np.sqrt(a_ ** 2 + b_ ** 2)))


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b); return float(spearmanr(a[m], b[m]).correlation) if m.sum() > 10 else float("nan")


def step_photo_inv(out: Path):
    out.mkdir(parents=True, exist_ok=True); base = photo_features(None); res = {}
    for name, p in PERTURB.items():
        t0 = time.time(); f = photo_features(p); res[name] = {k: round(spearman(base[k], f[k]), 4) for k in base}
        print(name, res[name], f"{time.time() - t0:.0f}s", flush=True)
    inv = {k: min(r[k] for r in res.values()) for k in base}
    json.dump(dict(per_perturbation=res, invariance_min=inv, rule=f"Spearman >= {FORM['inv_min']}", perturbations=PERTURB), open(out / "photo_invariance.json", "w", encoding="utf-8"), indent=1)
    np.savez_compressed(out / "photo_features_base.npz", **{k: v.astype(np.float32) for k, v in base.items()})
    print("invariance", inv, flush=True); return inv


# ---------------------------------------------------------------- inputs: V-P votes, anchors, blocks
def vp_votes(pl, H2id):
    """Recompute the V-P vote counts from its saved inputs (seed maps = V-H+fill+topo of map v3, propagated maps)."""
    vc = json.load(open(UL_OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; views = pl["views"]
    nmap = json.load(open(MAP_FILE, encoding="utf-8"))["map"]; raw, _ = UL.label_maps(UL_OUT, chosen + [held])
    for v in chosen:
        UL.STATE_PPM[v] = UL.plant_px_per_mm(pl, AP.view_file(v), H2id[:, views.index(AP.view_file(v))])[0]
    topo_rows, _ = UL.topology_report(pl, raw, nmap, chosen + [held], H2id)
    seed_maps, _, _ = UL.build_vote_maps(pl, raw, nmap, chosen, [r for r in topo_rows if r["photo"] in chosen], "V-H+fill+topo")
    pz = np.load(VP_DIR / "propagated_maps.npz"); prop = {k[1:]: pz[k] for k in pz.files}
    allmaps = dict(seed_maps, **prop); lab, cnt = UL.vote_views(pl, allmaps, chosen + sorted(prop), None, H2id)
    held_map = UL.remap({held: raw[held]}, nmap)[held]; held_map = np.where(held_map > 0, held_map, 0).astype(np.int16)
    return lab, cnt, raw, nmap, chosen, held, held_map


def anchors_of(lab, cnt, blade):
    tot = cnt.sum(1); top = cnt.max(1); share = np.where(tot > 0, top / np.maximum(tot, 1), 0)
    return blade & (lab > 0) & (share >= ANCHOR["share_min"]) & (tot >= ANCHOR["views_min"])


def blocks_of(pl, H2id, chosen, anchor, nmap):
    """Block id per anchor surfel = first user-labelled leaflet mask (labelling photos, map v3 number > 0) whose visible surfels contain it."""
    recs = UL.load_labels(UL_OUT); views = pl["views"]; block = np.full(len(pl["xyz"]), -1); bid = 0; meta = []
    for vid in chosen:
        det = {m["id"]: m for m in json.load(open(UL_OUT / "detect" / f"{vid}.json", encoding="utf-8"))["masks"]}; active = {}
        for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
            if r.get("removed"): active.pop(r["removed"], None); continue
            active[r.get("mask_id") or r["id"]] = r
        view = AP.view_file(vid); k = views.index(view); vis = H2id[:, k] >= 0
        pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"]); u = pv["u"] * UL.H2 // V2.VISP["res"]; w = pv["w"] * UL.H2 // V2.VISP["res"]
        ok = vis & anchor & (u >= 0) & (w >= 0) & (u < UL.H2) & (w < UL.H2)
        for key, r in sorted(active.items(), key=lambda kv: kv[1]["at"]):
            if r["leaf"] in ("모름", None) or int(nmap.get(f"{vid}|{int(r['leaf'])}", -1)) <= 0: continue
            m = UL.record_rle(r, recs, det)
            if m is None: continue
            sel = np.nonzero(ok & m[np.clip(w, 0, UL.H2 - 1), np.clip(u, 0, UL.H2 - 1)] & (block < 0))[0]
            if len(sel): block[sel] = bid; meta.append(dict(block=bid, photo=vid, key=key, anchors=int(len(sel)))); bid += 1
    return block, meta


def assign_folds(block, meta, n=FOLDS["n"], seed=FOLDS["seed"]):
    rng = np.random.default_rng(seed); order = rng.permutation(len(meta)); load = np.zeros(n); fold_of = {}
    for b in order:
        f = int(np.argmin(load)); fold_of[int(b)] = f; load[f] += meta[b]["anchors"]
    fold = np.full(len(block), -1); m = block >= 0; fold[m] = [fold_of[int(b)] for b in block[m]]
    return fold, load


# ---------------------------------------------------------------- graph and geometric features
def build_graph(xyz):
    tree = cKDTree(xyz); d, j = tree.query(xyz, k=GRAPH["k"] + 1, distance_upper_bound=GRAPH["r_mm"] / 1000); n = len(xyz)
    rows = np.repeat(np.arange(n), GRAPH["k"]); cols = j[:, 1:].ravel(); ds = d[:, 1:].ravel(); ok = np.isfinite(ds) & (cols < n)
    G = csr_matrix((ds[ok], (rows[ok], cols[ok])), shape=(n, n)); G = G.maximum(G.T)
    dens = np.array([len(x) for x in tree.query_ball_point(xyz, GRAPH["r_mm"] / 1000)], float)
    nb = tree.query_ball_point(xyz, GRAPH["dens_med_mm"] / 1000); densR = np.array([dens[i] / max(np.median(dens[x]), 1e-9) for i, x in enumerate(nb)])
    return G, densR


def geodesic_from(G, sources, densR, limit_m):
    """Multi-source geodesic distance and the minimum density R along the shortest path from the nearest source."""
    if not len(sources):
        n = G.shape[0]; return np.full(n, np.inf), np.zeros(n)
    dist, pred, src = dijkstra(G, indices=sources, min_only=True, return_predecessors=True, limit=limit_m)
    cont = np.full(len(dist), 0.0); order = np.argsort(dist); fin = np.isfinite(dist)
    for i in order[fin[order]]:
        p = pred[i]; cont[i] = densR[i] if p < 0 else min(densR[i], cont[p])
    return dist, cont


def leaf_stats(xyz, nrm, theta, photo, idx):
    P = xyz[idx]; c = P.mean(0); w, v = np.linalg.eigh(np.cov((P - c).T)) if len(P) > 3 else (None, np.eye(3)); e1 = v[:, -1]
    o = P[np.argmin(np.hypot(P[:, 0] - V2.CENTRE[0], P[:, 1] - V2.CENTRE[1]))]
    if np.dot(c[:2] - V2.CENTRE, e1[:2]) < 0: e1 = -e1
    along = (P - o) @ e1; lat = np.linalg.norm((P - o) - np.outer(along, e1), axis=1)
    nn = nrm[idx] * np.where(nrm[idx][:, 2:3] < 0, -1, 1); mn = nn.mean(0); mn /= max(np.linalg.norm(mn), 1e-12); th = np.radians(theta[idx])
    st = dict(base=float(np.percentile(P[:, 2], 10)), normal=mn, theta=float(np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360),
              o=o, e1=e1, along=(float(along.min()), float(along.max())), lat95=float(np.percentile(lat, 95)))
    for k, v in photo.items():
        st[k] = float(np.nanmedian(v[idx]))
    return st


FEATURES = ["height", "inclination", "azimuth", "density", "continuity", "distance", "brightness", "colour_a", "colour_chroma", "shape_along", "shape_lateral"]   # Revision 1: vp_share excluded (leaks the anchor rule)
PHOTO_FEATS = {"brightness": "L_resid_z", "colour_a": "a_z", "colour_chroma": "chroma_z"}


def pair_features(s, L, st, ctx):
    """A and R forms of every feature for surfels s against leaf L (arrays). Missing photo values -> nan."""
    xyz, nrm, theta, photo, densR = ctx["xyz"], ctx["nrm"], ctx["theta"], ctx["photo"], ctx["densR"]
    nn = nrm[s] * np.where(nrm[s][:, 2:3] < 0, -1, 1); incl = np.degrees(np.arccos(np.clip(nn[:, 2], -1, 1)))
    ang = np.degrees(np.arccos(np.clip(np.abs(nn @ st["normal"]), 0, 1))); dth = np.abs((theta[s] - st["theta"] + 180) % 360 - 180)
    along = (xyz[s] - st["o"]) @ st["e1"]; lat = np.linalg.norm((xyz[s] - st["o"]) - np.outer(along, st["e1"]), axis=1)
    exc = np.maximum(0, np.maximum(st["along"][0] - along, along - st["along"][1])) * 1000 / LEAFLET_MM
    tot = ctx["cnt"][s].sum(1); vps = np.where(tot > 0, ctx["cnt"][s, L] / np.maximum(tot, 1), 0.0)
    A = dict(height=xyz[s, 2], inclination=incl, azimuth=theta[s], brightness=photo["L_resid_z"][s], colour_a=photo["a_z"][s], colour_chroma=photo["chroma_z"][s])
    R = dict(height=(xyz[s, 2] - st["base"]) * 1000, inclination=ang, azimuth=dth, density=densR[s], continuity=ctx["cont"][L][s], distance=ctx["geo"][L][s] * 1000 / LEAFLET_MM,
             brightness=np.abs(photo["L_resid_z"][s] - st["L_resid_z"]), colour_a=np.abs(photo["a_z"][s] - st["a_z"]), colour_chroma=np.abs(photo["chroma_z"][s] - st["chroma_z"]),
             shape_along=exc, shape_lateral=lat * 1000 / LEAFLET_MM, vp_share=vps)
    return A, R


def context(pl, G, densR, photo, cnt, visible_anchor, lab, leaves):
    xyz = pl["xyz"]; theta = np.degrees(np.arctan2(xyz[:, 1] - V2.CENTRE[1], xyz[:, 0] - V2.CENTRE[0])) % 360
    ctx = dict(xyz=xyz, nrm=pl["nrm"], theta=theta, photo=photo, densR=densR, cnt=cnt, geo={}, cont={}, stats={})
    for L in leaves:
        src = np.nonzero(visible_anchor & (lab == L))[0]
        if len(src) < 5: continue
        ctx["geo"][L], ctx["cont"][L] = geodesic_from(G, src, densR, CAND_MM / 1000); ctx["stats"][L] = leaf_stats(xyz, pl["nrm"], theta, photo, src)
    return ctx


def candidate_pairs(ctx, todo):
    """[(surfel array, leaf)] for surfels in todo with the leaf's anchors within the geodesic candidate radius."""
    out = []
    for L, g in ctx["geo"].items():
        s = todo[np.isfinite(g[todo]) & (g[todo] <= CAND_MM / 1000)]
        if len(s): out.append((s, L))
    return out


def pair_table(ctx, todo, truth=None):
    rows_s, rows_L, A_, R_ = [], [], {f: [] for f in FEATURES}, {f: [] for f in FEATURES}
    for s, L in candidate_pairs(ctx, todo):
        A, R = pair_features(s, L, ctx["stats"][L], ctx); rows_s.append(s); rows_L.append(np.full(len(s), L))
        for f in FEATURES:
            A_[f].append(A.get(f, np.full(len(s), np.nan))); R_[f].append(R.get(f, np.full(len(s), np.nan)))
    if not rows_s:
        return None
    s = np.concatenate(rows_s); L = np.concatenate(rows_L)
    T = dict(s=s, L=L, A={f: np.concatenate(v) for f, v in A_.items()}, R={f: np.concatenate(v) for f, v in R_.items()})
    if truth is not None: T["y"] = (truth[s] == L).astype(int)
    return T


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(score)
    if m.sum() < 20 or len(set(y[m].tolist())) < 2: return float("nan")
    a = roc_auc_score(y[m], score[m]); return float(max(a, 1 - a))


def choose_forms(T, inv):
    table, chosen = {}, []
    for f in FEATURES:
        aA = auc(T["A"][f], T["y"]) if np.isfinite(T["A"][f]).any() else float("nan"); aR = auc(T["R"][f], T["y"]) if np.isfinite(T["R"][f]).any() else float("nan")
        best = max([(a, k) for a, k in ((aA, "A"), (aR, "R")) if np.isfinite(a)], default=(float("nan"), None)); ok = best[1] is not None and best[0] >= FORM["auc_min"]
        iv = inv.get(PHOTO_FEATS[f]) if f in PHOTO_FEATS else None
        if iv is not None and not (iv >= FORM["inv_min"]): ok = False
        table[f] = dict(auc_A=(round(aA, 3) if np.isfinite(aA) else None), auc_R=(round(aR, 3) if np.isfinite(aR) else None), form=best[1], invariance=iv, kept=bool(ok))
        if ok: chosen.append((f, best[1]))
    return table, chosen


def design(T, chosen):
    X = np.column_stack([T[k][f] for f, k in chosen]); return np.where(np.isfinite(X), X, 0.0)


def fit_model(kind, X, y):
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=0).fit(X, y)


def posterior(T, p, n):
    """Normalise pair scores over each surfel's candidates -> per-surfel (best leaf, best prob, margin) and a dict of probs."""
    probs = {}
    for s, L, q in zip(T["s"], T["L"], p):
        probs.setdefault(int(s), {})[int(L)] = float(q)
    best = np.zeros(n, int); bp = np.zeros(n); mg = np.zeros(n)
    for s, d in probs.items():
        tot = sum(d.values()) or 1.0; d2 = {k: v / tot for k, v in d.items()}; probs[s] = d2; srt = sorted(d2.items(), key=lambda kv: -kv[1])
        best[s] = srt[0][0]; bp[s] = srt[0][1]; mg[s] = srt[0][1] - (srt[1][1] if len(srt) > 1 else 0.0)
    return probs, best, bp, mg


def m1_labels(best, bp, mg, P_min, M_min, todo):
    out = np.zeros(len(best), int); ok = (bp >= P_min) & (mg >= M_min); out[todo] = np.where(ok[todo], best[todo], 0); return out


def edge_weights(G, ctx, chosen_names):
    Gc = G.tocoo(); i, j = Gc.row, Gc.col; nrm = ctx["nrm"]; nn = nrm * np.where(nrm[:, 2:3] < 0, -1, 1)
    ang = np.degrees(np.arccos(np.clip(np.abs((nn[i] * nn[j]).sum(1)), 0, 1))); e = ang / POTTS["angle_deg"]
    if "brightness" in chosen_names: e = e + np.nan_to_num(np.abs(ctx["photo"]["L_resid_z"][i] - ctx["photo"]["L_resid_z"][j]))
    if "colour_a" in chosen_names: e = e + np.nan_to_num(np.abs(ctx["photo"]["a_z"][i] - ctx["photo"]["a_z"][j]))
    w = np.exp(-e) * np.minimum(1.0, np.minimum(ctx["densR"][i], ctx["densR"][j]))
    return csr_matrix((w, (i, j)), shape=G.shape)


def icm(labels, probs, fixed, W, lam, sweeps=POTTS["sweeps"]):
    """Potts ICM: labels of non-fixed surfels with a posterior re-chosen among their candidates; 0 = abstained (no label, no pairwise term)."""
    lab = labels.copy(); free = [s for s in probs if not fixed[s] and lab[s] > 0]; indptr, indices, data = W.indptr, W.indices, W.data
    for _ in range(sweeps):
        changed = 0
        for s in free:
            nb = indices[indptr[s]:indptr[s + 1]]; wt = data[indptr[s]:indptr[s + 1]]; nl = lab[nb]; best, be = lab[s], None
            for L, p in probs[s].items():
                e = -np.log(max(p, 1e-9)) + lam * float(wt[(nl > 0) & (nl != L)].sum())
                if be is None or e < be: best, be = L, e
            if best != lab[s]: lab[s] = best; changed += 1
        if not changed: break
    return lab


def m3_shape(lab, ctx, anchor, G):
    out = lab.copy(); xyz = ctx["xyz"]
    for L, st in ctx["stats"].items():
        m = (out == L) & ~anchor
        if not m.any(): continue
        s = np.nonzero(m)[0]; along = (xyz[s] - st["o"]) @ st["e1"]; lat = np.linalg.norm((xyz[s] - st["o"]) - np.outer(along, st["e1"]), axis=1)
        bad = (along < st["along"][0] - LEAFLET_MM / 1000) | (along > st["along"][1] + LEAFLET_MM / 1000) | (lat > st["lat95"] + LEAFLET_MM / 2000); out[s[bad]] = 0
        m = (out == L) & ~anchor; idx = np.nonzero(m)[0]
        if not len(idx): continue
        sub = G[idx][:, idx]; nc, comp = connected_components(sub, directed=False)
        anc = np.nonzero((out == L) & anchor)[0]; inl = np.zeros(len(xyz), bool); inl[anc] = True
        for c in range(nc):
            mem = idx[comp == c]
            touches = bool(inl[G[mem].indices].any())
            if not touches and np.nanmin(ctx["geo"][L][mem]) * 1000 > LEAFLET_MM: out[mem] = 0
    return out


def internal_score(pred, truth, hidden):
    h = np.nonzero(hidden)[0]; p = pred[h]; t = truth[h]; c = float((p == t).mean()); w = float(((p > 0) & (p != t)).mean()); a = float((p == 0).mean())
    return dict(correct=round(c, 4), wrong=round(w, 4), abstained=round(a, 4), score=round(c - w, 4), n=int(len(h)))


def step_run(out: Path):
    t0 = time.time(); inv_f = out / "photo_invariance.json"
    while not inv_f.exists():
        print("waiting for photo invariance ...", flush=True); time.sleep(30)
    inv = json.load(open(inv_f, encoding="utf-8"))["invariance_min"]
    pl = UL.load_plant(); views = pl["views"]; H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; H = float(pl["xyz"][:, 2].max() - V2.POT_TOP)
    lab, cnt, raw, nmap, chosen_views, held, held_map = vp_votes(pl, H2id)
    zs = np.load(VP_DIR / "surfel_clusters.npz"); saved = np.where(zs["leaf"] > 0, zs["leaf"], 0); repro = bool(np.array_equal(lab, saved))
    blade = pl["blade"]; anchor = anchors_of(lab, cnt, blade); block, meta = blocks_of(pl, H2id, chosen_views, anchor, nmap); fold, load = assign_folds(block, meta)
    b = np.nonzero(blade)[0]; n = len(b); ph = np.load(CC.OUT2 / "r31b" / "photo_features.npz")
    sub = dict(xyz=pl["xyz"][b], nrm=pl["nrm"][b]); photo = {k: ph[k][b].astype(float) for k in ("L_resid_z", "a_z", "chroma_z")}
    lab_b = lab[b]; cnt_b = cnt[b]; anc_b = anchor[b]; fold_b = fold[b]; leaves = sorted(set(lab_b[lab_b > 0].tolist()))
    G, densR = build_graph(sub["xyz"]); print(f"inputs ready, V-P reproduced={repro}, anchors {int(anc_b.sum())}/{n}, blocks {len(meta)}, fold loads {load.tolist()} {time.time() - t0:.0f}s", flush=True)
    folds = []
    for f in range(FOLDS["n"]):
        hidden = anc_b & (fold_b == f); vis = anc_b & ~hidden; ctx = context(sub, G, densR, photo, cnt_b, vis, lab_b, leaves); todo = np.nonzero(~vis)[0]
        T = pair_table(ctx, todo, truth=lab_b); T["hid"] = hidden[T["s"]]; folds.append(dict(hidden=hidden, vis=vis, ctx=ctx, todo=todo, T=T))
        print(f"fold {f}: hidden {int(hidden.sum())}, pairs {len(T['s'])} {time.time() - t0:.0f}s", flush=True)
    pool = {k: np.concatenate([F["T"][k][F["T"]["hid"]] for F in folds]) for k in ("s", "L", "y")}
    for k in ("A", "R"):
        pool[k] = {f: np.concatenate([F["T"][k][f][F["T"]["hid"]] for F in folds]) for f in FEATURES}
    form_table, chosen = choose_forms(pool, inv); names = [f for f, _ in chosen]; print("forms", chosen, flush=True)
    def hidden_T(F):
        m = F["T"]["hid"]; return {"y": F["T"]["y"][m], **{k: {f: F["T"][k][f][m] for f in FEATURES} for k in ("A", "R")}}
    def train_except(kind, f):
        Ts = [hidden_T(F) for g, F in enumerate(folds) if g != f]; X = np.vstack([design(t, chosen) for t in Ts]); y = np.concatenate([t["y"] for t in Ts]); return fit_model(kind, X, y)
    W = edge_weights(G, dict(nrm=sub["nrm"], photo=photo, densR=densR), names)
    # classifier choice at P_min 0.5 / M_min 0.1 (M1)
    models, clf_scores = {}, {}
    for kind in ("logistic", "hgb"):
        preds = np.zeros(n, int); hid_all = np.zeros(n, bool); models[kind] = []
        for f, F in enumerate(folds):
            mdl = train_except(kind, f); models[kind].append(mdl); p = mdl.predict_proba(design(F["T"], chosen))[:, 1]; _, best, bp, mg = posterior(F["T"], p, n)
            m1 = m1_labels(best, bp, mg, 0.5, 0.1, F["todo"]); preds[F["hidden"]] = m1[F["hidden"]]; hid_all |= F["hidden"]
        clf_scores[kind] = internal_score(preds, lab_b, hid_all); print("classifier", kind, clf_scores[kind], flush=True)
    kind = max(clf_scores, key=lambda k: clf_scores[k]["score"])
    # grids per method on out-of-fold predictions
    post = []
    for f, F in enumerate(folds):
        p = models[kind][f].predict_proba(design(F["T"], chosen))[:, 1]; post.append(posterior(F["T"], p, n))
    grid = {}
    for P_min in ABSTAIN["P_min"]:
        for M_min in ABSTAIN["M_min"]:
            for lam in POTTS["lam"]:
                pr = {m: np.zeros(n, int) for m in ("M1", "M2", "M3")}; hid_all = np.zeros(n, bool)
                for f, F in enumerate(folds):
                    probs, best, bp, mg = post[f]; m1 = m1_labels(best, bp, mg, P_min, M_min, F["todo"]); base = np.where(F["vis"], lab_b, m1)
                    keep = {s: d for s, d in probs.items() if m1[s] > 0}; m2 = icm(base, keep, F["vis"], W, lam); m3 = m3_shape(m2, F["ctx"], F["vis"], G)
                    for m, arr in (("M1", base), ("M2", m2), ("M3", m3)): pr[m][F["hidden"]] = arr[F["hidden"]]
                    hid_all |= F["hidden"]
                grid[(P_min, M_min, lam)] = {m: internal_score(pr[m], lab_b, hid_all) for m in pr}
                print("grid", P_min, M_min, lam, {m: v["score"] for m, v in grid[(P_min, M_min, lam)].items()}, f"{time.time() - t0:.0f}s", flush=True)
    best_par = {}
    for m in ("M1", "M2", "M3"):
        cands = [(k, v[m]) for k, v in grid.items() if m != "M1" or k[2] == POTTS["lam"][0]]
        k, v = max(cands, key=lambda kv: (kv[1]["score"], -kv[1]["wrong"])); best_par[m] = dict(P_min=k[0], M_min=k[1], lam=(k[2] if m != "M1" else None), internal=v)
    # final: all anchors visible, model trained on every fold's hidden pairs, held-out photo scored once per method
    ctx = context(sub, G, densR, photo, cnt_b, anc_b, lab_b, leaves); todo = np.nonzero(~anc_b)[0]; T = pair_table(ctx, todo)
    Ts = [hidden_T(F) for F in folds]; mdl = fit_model(kind, np.vstack([design(t, chosen) for t in Ts]), np.concatenate([t["y"] for t in Ts]))
    p = mdl.predict_proba(design(T, chosen))[:, 1]; probs, best, bp, mg = posterior(T, p, n)
    finals = {}
    for m, par in best_par.items():
        m1 = m1_labels(best, bp, mg, par["P_min"], par["M_min"], todo); base = np.where(anc_b, lab_b, m1)
        if m == "M1": finals[m] = base; continue
        keep = {s: d for s, d in probs.items() if m1[s] > 0}; m2 = icm(base, keep, anc_b, W, par["lam"]); finals[m] = m2 if m == "M2" else m3_shape(m2, ctx, anc_b, G)
    hv = AP.view_file(held); col = H2id[:, views.index(hv)]; scores = {}
    def full(lb):
        a = np.zeros(len(pl["xyz"]), np.int16); a[b] = lb; return a
    for m, lb in [("V-P", lab_b)] + list(finals.items()):
        ren = UL.render_ids(pl, full(lb), hv, col); rows, g = UL.score_heldout(held_map, ren); cov = float(((ren > 0) & (held_map > 0)).sum() / max((held_map > 0).sum(), 1))
        flags = {str(L): ("usable" if r["iou"] >= USABLE["iou_min"] and r["confusion"] <= USABLE["conf_max"] else "viewing_only") for L, r in rows.items()}
        conf = {}
        for L in rows:
            A_ = held_map == L; conf[str(L)] = {str(k): int(v) for k, v in Counter(ren[A_].tolist()).most_common()}
        changes = None
        if m != "V-P":
            changes = dict(filled=int(((lab_b == 0) & (lb > 0)).sum()), relabelled=int(((lab_b > 0) & (lb > 0) & (lb != lab_b)).sum()), unset=int(((lab_b > 0) & (lb == 0)).sum()),
                           per_leaf={str(L): dict(filled=int(((lab_b == 0) & (lb == L)).sum()), relabelled_in=int(((lab_b > 0) & (lab_b != L) & (lb == L)).sum()), relabelled_out=int(((lab_b == L) & (lb > 0) & (lb != L)).sum())) for L in leaves})
        scores[m] = dict(rows={str(k): v for k, v in rows.items()}, summary=g, coverage=round(cov, 4), flags=flags, confusion_matrix=conf, changes=changes, assigned=int((lb > 0).sum()))
        print("held-out", m, round(g["median_iou"], 3), round(g["max_confusion"], 3), "cov", round(cov, 3), flags, changes and {k: v for k, v in changes.items() if k != "per_leaf"}, flush=True)
    order = sorted(scores, key=lambda m: (not scores[m]["summary"]["gate_pass"], -scores[m]["summary"]["median_iou"], scores[m]["summary"]["max_confusion"]))
    pick = next(m for m in order if m != "V-P"); tag = "gate_passed" if scores[pick]["summary"]["gate_pass"] else "gate_failed_viewing_only"
    res = dict(**LABELS, input="10-25 V-P map v3 (user_labelled_propagated); recheck step not run; original truth only", vp_reproduced=repro, anchors=int(anc_b.sum()), blade=n,
               anchor_rule=ANCHOR, blocks=len(meta), fold_loads=load.tolist(), photo_invariance=inv, forms=form_table, chosen_features=chosen, classifier_internal=clf_scores, classifier=kind,
               grid={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in grid.items()}, chosen_params=best_par, held_out=scores, pick=pick, tag=tag, seconds=round(time.time() - t0))
    json.dump(res, open(out / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    lb = finals[pick]; src = np.where(anc_b, "anchor", np.where((lab_b == 0) & (lb > 0), "filled", np.where((lab_b > 0) & (lb > 0) & (lb != lab_b), "relabelled", np.where(lb > 0, "kept", "unassigned"))))
    prob = np.where(anc_b, 1.0, np.where(lb > 0, bp, 0.0))
    with open(out / "surfel_leaf_ids_v2.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {GROUPING}; method {pick}; {tag}; row = index into R3-1b_surfels_full.ply (blade surfels only); leaf_id -1 = unassigned\nrow,leaf_id,probability,source,method\n")
        for r_, l_, p_, s_ in zip(pl["rows"][b], lb, prob, src): fh.write(f"{r_},{l_ if l_ > 0 else -1},{p_:.3f},{s_},{pick}\n")
    flags = scores[pick]["flags"]; table = UL.leaf_table(pl, full(lb), {}, H)
    for r in table: r["quality"] = flags.get(str(r["leaf"]), "not_in_held_out")
    with open(out / "leaves.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {GROUPING}; method {pick}; {tag}; area = kind (1); quality usable = held-out IoU >= 0.55 and confusion <= 0.10\n")
        keys = sorted({k for r in table for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table)
    leaf_full = np.full(len(pl["xyz"]), -1); leaf_full[b] = np.where(lb > 0, lb, -1); src_full = np.full(len(pl["xyz"]), "axis", dtype="U12"); src_full[b] = src
    prob_full = np.zeros(len(pl["xyz"])); prob_full[b] = prob
    np.savez_compressed(out / "surfel_clusters.npz", rows=pl["rows"], leaf=leaf_full, source=src_full, probability=prob_full, axis_kind=zs["axis_kind"],
                        **{f"{m}_leaf": np.where(full(v) > 0, full(v), -1) for m, v in finals.items()})
    print("pick", pick, tag, f"{time.time() - t0:.0f}s", flush=True); return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--step", choices=("photo_inv", "run"), default="run"); a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.step == "photo_inv":
        step_photo_inv(a.out)
    else:
        step_run(a.out)


if __name__ == "__main__":
    main()
