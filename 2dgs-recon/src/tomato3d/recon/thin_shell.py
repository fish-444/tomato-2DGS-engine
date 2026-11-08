"""Label-free surface metrics for a 2DGS whole-plant model (contract build/p1_thin_shell_20261026_01/CONTRACT.md §1): shell thickness (FWHM of
surfel offsets along the local plane normal), share of offsets beyond ±1.5 mm, blade sheet connectivity, and the area overcount of summed surfel
ellipses vs the rendered silhouette (engine static mode).

usage: python -m tomato3d.recon.thin_shell --ply <surfels_full.ply> --views <run_views.npz> --name <tag> [--leaf-npz <surfel_clusters.npz>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import light_organs as LO
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_thin_shell_20261026_01"
R31B = LO.DEFAULTS["surfels"]; TRACK_A = LO.DEFAULTS["out"] / "surfel_labels.npz"
M = dict(opacity_min=LO.OPACITY_MIN, pot_top=LO.POT_TOP_Z, axis_transfer_mm=3.0, samples=5000, seed=0, plane_mm=3.0, lateral_mm=3.0, range_mm=10.0,
         bin_mm=0.5, beyond_mm=1.5, graph_k=10, graph_mm=3.0, ellipse_k=1.5)


def plant(ply):
    sf = sio.read_surfels_full(ply); keep = (sf["opacity"] >= M["opacity_min"]) & (sf["xyz"][:, 2] > M["pot_top"]); rows = np.nonzero(keep)[0]
    return sf, rows


def blade_mask(xyz, own_axis=None):
    """R3-1b: Track A axis flags; another model: nearest R3-1b plant surfel's flag within 3 mm (geometry class only)."""
    if own_axis is not None:
        return ~own_axis
    sf0, rows0 = plant(R31B); ax0 = np.load(TRACK_A)["is_axis"].astype(bool); d, j = cKDTree(sf0["xyz"][rows0]).query(xyz)
    return ~((d <= M["axis_transfer_mm"] / 1000) & ax0[j])


def fwhm(off_mm):
    h, e = np.histogram(off_mm, bins=np.arange(-M["range_mm"], M["range_mm"] + M["bin_mm"], M["bin_mm"]))
    h = np.convolve(h.astype(float), np.array([0.25, 0.5, 0.25]), mode="same"); c = (e[:-1] + e[1:]) / 2; k = int(np.argmin(np.abs(c)))
    k = max(range(max(0, k - 2), min(len(h), k + 3)), key=lambda i: h[i]); half = h[k] / 2
    if h[k] <= 0: return np.nan
    i = k
    while i > 0 and h[i] > half: i -= 1
    j = k
    while j < len(h) - 1 and h[j] > half: j += 1
    lo = c[i] + (half - h[i]) / max(h[i + 1] - h[i], 1e-9) * (c[i + 1] - c[i]) if h[i] <= half else c[i]
    hi = c[j - 1] + (h[j - 1] - half) / max(h[j - 1] - h[j], 1e-9) * (c[j] - c[j - 1]) if h[j] <= half else c[j]
    return float(hi - lo)


def thickness(xyz, leaf=None):
    rng = np.random.default_rng(M["seed"]); s = rng.choice(len(xyz), min(M["samples"], len(xyz)), replace=False); t = cKDTree(xyz)
    T, beyond, pooled, per_leaf = [], [], [], {}
    for i in s:
        nb = t.query_ball_point(xyz[i], max(M["plane_mm"], M["range_mm"]) / 1000); P = xyz[nb]
        loc = P[np.linalg.norm(P - xyz[i], axis=1) <= M["plane_mm"] / 1000]
        if len(loc) < 6: continue
        w, v = np.linalg.eigh(np.cov((loc - loc.mean(0)).T)); n = v[:, 0]
        d = P - xyz[i]; a = d @ n; lat = np.linalg.norm(d - np.outer(a, n), axis=1); m = (lat <= M["lateral_mm"] / 1000) & (np.abs(a) <= M["range_mm"] / 1000)
        off = a[m] * 1000
        if len(off) < 6: continue
        f = fwhm(off); T.append(f); beyond.append(np.abs(off) > M["beyond_mm"]); pooled.append(off)
        if leaf is not None and 0 < leaf[i] < 100: per_leaf.setdefault(int(leaf[i]), []).append(f)
    allb = np.concatenate(beyond); allo = np.concatenate(pooled)
    return dict(samples=len(T), median_T_mm=round(float(np.nanmedian(T)), 3), p25_p75_T_mm=[round(float(x), 3) for x in np.nanpercentile(T, [25, 75])],
                share_beyond_1p5mm=round(float(allb.mean()), 4), pooled_fwhm_mm=round(fwhm(allo), 3),
                per_leaf={str(L): dict(n=len(v), median_T_mm=round(float(np.nanmedian(v)), 3)) for L, v in sorted(per_leaf.items())})


def connectivity(xyz):
    t = cKDTree(xyz); d, j = t.query(xyz, k=M["graph_k"] + 1, distance_upper_bound=M["graph_mm"] / 1000); n = len(xyz)
    r = np.repeat(np.arange(n), M["graph_k"]); c = j[:, 1:].ravel(); ok = np.isfinite(d[:, 1:].ravel()) & (c < n)
    G = coo_matrix((np.ones(ok.sum()), (r[ok], c[ok])), shape=(n, n)).tocsr(); nc, lab = connected_components(G, directed=False)
    return dict(components=int(nc), largest_share=round(float(np.bincount(lab).max() / n), 4))


def overcount(sf, rows, views_npz, po, res=1512):
    """Per view: sum of projected 1.5-sigma ellipse areas of plant surfels / rendered silhouette (alpha >= 0.5) area."""
    z = np.load(views_npz); K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res); xyz = sf["xyz"][rows].astype(float); nrm = sf["normal"][rows].astype(float)
    area = np.pi * M["ellipse_k"] ** 2 * sf["scale"][rows][:, 0] * sf["scale"][rows][:, 1]; out = {}
    for key in z.files:
        if not key.startswith("a_"): continue
        v = key[2:]
        if v not in po["views"]: continue
        al = z[key].astype(np.float32); sil = float((al >= 0.5).sum())
        if al.shape[0] != res: sil *= (res / al.shape[0]) ** 2
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"]); xc = xyz @ R.T + t; zc = xc[:, 2]; cam = -(R.T @ t)
        dirv = cam - xyz; dirv /= np.maximum(np.linalg.norm(dirv, axis=1, keepdims=True), 1e-12); cos = np.abs((nrm * dirv).sum(1))
        ok = zc > 0; proj = float((area[ok] * cos[ok] * (K[0, 0] / zc[ok]) ** 2).sum())
        if sil > 0: out[v] = proj / sil
    vals = np.array(list(out.values()))
    return dict(views=len(out), median_ratio=round(float(np.median(vals)), 3), p10_p90=[round(float(x), 3) for x in np.percentile(vals, [10, 90])])


def measure(ply, views_npz, leaf_npz=None, own_axis=False):
    sf, rows = plant(ply); xyz = sf["xyz"][rows].astype(float)
    ax = np.load(TRACK_A)["is_axis"].astype(bool) if own_axis else None; bl = blade_mask(xyz, ax)
    leaf = None
    if leaf_npz:
        zl = np.load(leaf_npz, allow_pickle=True); lf = np.where(zl["leaf"] > 0, zl["leaf"], 0); assert len(lf) == len(rows); leaf = lf[bl]
    po = json.load(open(LO.DEFAULTS["poses"]))
    return dict(plant_surfels=int(len(rows)), blade_surfels=int(bl.sum()), thickness=thickness(xyz[bl], leaf), connectivity=connectivity(xyz[bl]),
                overcount=(overcount(sf, rows, views_npz, po) if views_npz else None))


def second_layer_test(out_json):
    """Contract section 6: are the stacked second layers visible and photo-consistent on views never used in training?"""
    from PIL import Image
    from tomato3d.organs import vote2d as V2
    po = json.load(open(LO.DEFAULTS["poses"])); con = json.load(open(ROOT / "build/p1_recon_r3-1_20261022_01/contract.json")); held = con["heldout_fresh"]
    sf, rows = plant(R31B); xyz = sf["xyz"][rows].astype(float); rgb = sf["rgb"][rows].astype(float); sc = sf["scale"][rows].astype(float)
    nrm = sf["normal"][rows].astype(float); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    z = np.load(OUT / "metrics" / "R3-1b_per_surfel.npz"); blade = z["blade"]; second = z["second_layer"] & blade; single = blade & ~z["second_layer"] & np.isfinite(z["beyond"])
    photos = LO.DEFAULTS["poses"].parent / "undist_3024"; res = V2.VISP["res"]; n = len(rows)
    seen = np.zeros(n, int); err = [[] for _ in range(n)]; upper = np.zeros(n, int); lower = np.zeros(n, int)
    idx_b = np.nonzero(blade)[0]; tree_b = cKDTree(xyz[blade])
    for v in held:
        pv = V2.project_view(xyz, sc, po, v, res); vis = pv["vis"]
        img = np.asarray(Image.open(photos / v).convert("RGB").resize((res, res), Image.BILINEAR)).astype(float)
        col = img[np.clip(pv["w"], 0, res - 1), np.clip(pv["u"], 0, res - 1)]; e = np.abs(rgb - col).mean(1)
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"]); cam = -(R.T @ t)
        for i in np.nonzero(vis & blade)[0]:
            seen[i] += 1; err[i].append(e[i])
        for i in np.nonzero(vis & second)[0]:
            for sg in (1, -1):
                done = False
                for d in (3.0, 5.0, 7.5):
                    jj = tree_b.query_ball_point(xyz[i] + sg * d / 1000 * nrm[i], 0.001)
                    if jj:
                        j = idx_b[jj[0]]
                        if np.linalg.norm(cam - xyz[i]) <= np.linalg.norm(cam - xyz[j]): upper[i] += 1
                        else: lower[i] += 1
                        done = True; break
                if done: break
    cpath = OUT.parent / "p1_stacked_sheets_20261026_01" / "contributions.npz"; Wc = np.load(cpath)["W"] if cpath.exists() else None      # already over the plant rows
    assert Wc is None or len(Wc) == n
    def stat(m):
        have = np.nonzero(m & (seen > 0))[0]; e = np.array([np.median(err[i]) for i in have])
        return dict(n=int(m.sum()), seen_share_heldout=round(float((seen[m] > 0).mean()), 4), mean_views_seen=round(float(seen[m].mean()), 3),
                    median_colour_error=(round(float(np.median(e)), 2) if len(e) else None), n_with_error=int(len(e)),
                    train_visible_share=(round(float((Wc[m] >= 0.1).any(1).mean()), 4) if Wc is not None else None))
    lowerm = second & (lower > upper); upperm = second & (upper >= lower) & ((upper + lower) > 0)
    g = dict(stacked=stat(second), single_sheet=stat(single), stacked_lower=stat(lowerm), stacked_upper=stat(upperm))
    def verdict(a):
        b = g["single_sheet"]
        if a["median_colour_error"] is None or b["median_colour_error"] is None: return dict(verdict="undecided (no visible surfels)")
        vr = a["seen_share_heldout"] / max(b["seen_share_heldout"], 1e-9); er = a["median_colour_error"] / max(b["median_colour_error"], 1e-9)
        v = "supported" if (vr >= 0.5 and er <= 1.25) else ("ghost-like" if (vr < 0.25 or er > 1.5) else "undecided")
        return dict(verdict=v, visibility_ratio=round(vr, 3), colour_error_ratio=round(er, 3))
    r = dict(groups=g, verdict={k: verdict(g[k]) for k in ("stacked", "stacked_lower", "stacked_upper")}, heldout_views=held, rule="CONTRACT section 6")
    json.dump(r, open(out_json, "w"), indent=1); return r


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--second-layer", action="store_true"); ap.add_argument("--ply", type=Path, default=None); ap.add_argument("--views", type=Path, default=None); ap.add_argument("--name", default="x")
    ap.add_argument("--leaf-npz", type=Path, default=None); ap.add_argument("--own-axis", action="store_true"); ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.second_layer:
        print(json.dumps(second_layer_test(a.out / "metrics" / "second_layer_test.json"), indent=1)); return
    r = measure(a.ply, a.views, a.leaf_npz, a.own_axis); r.update(name=a.name, ply=str(a.ply), params=M)
    (a.out / "metrics").mkdir(parents=True, exist_ok=True); json.dump(r, open(a.out / "metrics" / f"{a.name}.json", "w"), indent=1); print(json.dumps({k: v for k, v in r.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()


def per_surfel(xyz, bl):
    """Local profile per blade surfel: share of neighbour offsets beyond ±1.5 mm, FWHM of the main peak, and a 'second layer' flag (a bump of
    >= 3 surfels beyond 2.5 mm separated from the core by a valley < 60 % of both ends; 1 mm bins). Same neighbourhood as thickness()."""
    X = xyz[bl]; t = cKDTree(X); n = len(X); beyond = np.full(n, np.nan, np.float32); fw = np.full(n, np.nan, np.float32); second = np.zeros(n, bool)
    for i in range(n):
        P = X[t.query_ball_point(X[i], M["range_mm"] / 1000 + 0.003)]; loc = P[np.linalg.norm(P - X[i], axis=1) <= M["plane_mm"] / 1000]
        if len(loc) < 6: continue
        nv = np.linalg.eigh(np.cov((loc - loc.mean(0)).T))[1][:, 0]; d = P - X[i]; a = d @ nv; lat = np.linalg.norm(d - np.outer(a, nv), axis=1)
        off = a[(lat <= M["lateral_mm"] / 1000) & (np.abs(a) <= M["range_mm"] / 1000)] * 1000
        if len(off) < 8: continue
        beyond[i] = (np.abs(off) > M["beyond_mm"]).mean(); fw[i] = fwhm(off); h = np.histogram(off, bins=np.arange(-10, 10.5, 1.0))[0]; mid = len(h) // 2
        for side in (1, -1):
            seg = h[mid:] if side > 0 else h[:mid + 1][::-1]
            for k in range(3, len(seg)):
                if seg[k] >= 3 and seg[1:k].min() < 0.6 * min(seg[0], seg[k]): second[i] = True
    full = lambda v, f: np.where(bl, 0, f).astype(v.dtype)
    return beyond, fw, second
