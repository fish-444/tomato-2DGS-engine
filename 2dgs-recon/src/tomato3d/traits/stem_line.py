"""Stem line per plant from surfels (build/traits_base_angle_v4_20261102_01/CONTRACT.md §2). Units mm, z up."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.traits.base_angle import StemLine

BAND, R_CENTRE_MM, W_MAX_MM, W_R_MM, NZ_MAX = 0.4, 30.0, 4.0, 3.0, 0.5
N_HYP, TOL_MM, TILT_MAX_DEG, MIN_INL, SEED = 2000, 3.0, 15.0, 30, 0


def heights(xyz):
    z = xyz[:, 2]; zb, zt = float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))
    return zb, zt, zb + BAND * (zt - zb)


def stem_width(xyz, tree, idx):
    """2 x p90 of horizontal distances of the surfels within 3 mm to their horizontal centroid."""
    out = np.empty(len(idx))
    for k, nb in enumerate(tree.query_ball_point(xyz[idx], W_R_MM)):
        q = xyz[nb, :2]; out[k] = 2.0 * float(np.percentile(np.linalg.norm(q - q.mean(0), axis=1), 90)) if len(nb) >= 3 else np.inf
    return out


def tilt_deg(d):
    d = np.asarray(d, float); return float(np.degrees(np.arccos(min(1.0, abs(d[2]) / np.linalg.norm(d)))))


def perp(xyz, p0, d):
    q = xyz - p0; return np.linalg.norm(q - np.outer(q @ d, d), axis=1)


def fit_stem(xyz, normal, centre_xy, tree=None):
    tree = tree if tree is not None else cKDTree(xyz)
    zb, zt, zc = heights(xyz)
    sel = np.where((xyz[:, 2] <= zc) & (np.hypot(xyz[:, 0] - centre_xy[0], xyz[:, 1] - centre_xy[1]) <= R_CENTRE_MM)
                   & (np.abs(normal[:, 2] / np.maximum(np.linalg.norm(normal, axis=1), 1e-12)) <= NZ_MAX))[0]
    w = stem_width(xyz, tree, sel) if len(sel) else np.zeros(0); cand = sel[w <= W_MAX_MM]
    res = dict(z_base=zb, z_top=zt, z_band=zc, n_candidates=int(len(cand)), cand=cand)
    if len(cand) < MIN_INL:
        return dict(res, status='stem_unknown', n_inliers=0, line=None, tilt_deg=None, inl=np.zeros(0, int))
    P = xyz[cand]; rng = np.random.default_rng(SEED); best, bd, bp = -1, None, None
    for _ in range(N_HYP):
        i, j = rng.choice(len(P), 2, replace=False); d = P[j] - P[i]; n = np.linalg.norm(d)
        if n < 1e-9 or tilt_deg(d) > TILT_MAX_DEG:
            continue
        d = d / n; k = int((perp(P, P[i], d) <= TOL_MM).sum())
        if k > best:
            best, bd, bp = k, d, P[i]
    if bd is None:
        return dict(res, status='stem_unknown', n_inliers=0, line=None, tilt_deg=None, inl=np.zeros(0, int))
    I = perp(P, bp, bd) <= TOL_MM; m = P[I].mean(0); _, _, vt = np.linalg.svd(P[I] - m); d2 = vt[0]
    p0, d = (m, d2) if tilt_deg(d2) <= TILT_MAX_DEG else (bp, bd)
    line = StemLine(p0, d); I = perp(P, line.p0, line.d) <= TOL_MM
    if I.sum() < MIN_INL:
        return dict(res, status='stem_unknown', n_inliers=int(I.sum()), line=None, tilt_deg=None, inl=np.zeros(0, int))
    return dict(res, status='fitted', n_inliers=int(I.sum()), line=line, tilt_deg=tilt_deg(line.d), inl=cand[I])


def known_line(stem_poly, z_band, step=0.5):
    """Principal axis of the true main-stem polyline (densified) within the band z <= z_band."""
    P = np.asarray(stem_poly, float); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1] + 1e-9, step); Q = np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1); Q = Q[Q[:, 2] <= z_band]
    if len(Q) < 2:
        Q = P[:2]
    m = Q.mean(0); _, _, vt = np.linalg.svd(Q - m); return StemLine(m, vt[0])


def poly_xy_at(stem_poly, z):
    """xy of the true stem polyline at height z (stem assumed monotone in z; clamped at the ends)."""
    P = np.asarray(stem_poly, float); o = np.argsort(P[:, 2]); P = P[o]
    return np.array([np.interp(z, P[:, 2], P[:, 0]), np.interp(z, P[:, 2], P[:, 1])]).T
