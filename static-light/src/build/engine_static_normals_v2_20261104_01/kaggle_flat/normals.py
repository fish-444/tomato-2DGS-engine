"""normal_mode for the static tracer (engine_static_normals_20261104_01/CONTRACT.md §1). New code only; the v1 package is untouched.
smooth(): per surfel, weighted PCA plane over same-label-set surfels within r mm (self included); < KMIN points -> raw normal kept.
noise_metric(): the Traits neighbour-normal noise (median angle to the unweighted PCA normal of neighbours within 2 mm, >= 6, 20 000 pts, seed 0)."""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.spatial import cKDTree

MODES = ('raw', 'smooth_r2', 'smooth_r3', 'smooth_r5')
RADIUS = {'smooth_r2': 2.0, 'smooth_r3': 3.0, 'smooth_r5': 5.0}
KMIN = 6


@nb.njit(cache=True)
def _moments(X, w, r):
    """Weighted moments of every point's r-neighbourhood (one label set). Returns sw, mean (n,3), cov (n,3,3), count."""
    n = X.shape[0]
    lo = np.empty(3)
    for a in range(3):
        lo[a] = X[:, a].min()
    ci = np.empty((n, 3), np.int64)
    for i in range(n):
        for a in range(3):
            ci[i, a] = int((X[i, a] - lo[a]) / r)
    ny = ci[:, 1].max() + 3; nz = ci[:, 2].max() + 3
    key = np.empty(n, np.int64)
    for i in range(n):
        key[i] = ((ci[i, 0] + 1) * ny + (ci[i, 1] + 1)) * nz + (ci[i, 2] + 1)
    order = np.argsort(key); sk = key[order]
    r2 = r * r
    sw = np.zeros(n); mean = np.zeros((n, 3)); cov = np.zeros((n, 3, 3)); cnt = np.zeros(n, np.int64)
    for i in range(n):
        s = 0.0; m0 = 0.0; m1 = 0.0; m2 = 0.0
        q00 = 0.0; q01 = 0.0; q02 = 0.0; q11 = 0.0; q12 = 0.0; q22 = 0.0; c = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    k = ((ci[i, 0] + 1 + dx) * ny + (ci[i, 1] + 1 + dy)) * nz + (ci[i, 2] + 1 + dz)
                    a0 = np.searchsorted(sk, k); a1 = np.searchsorted(sk, k + 1)
                    for t in range(a0, a1):
                        j = order[t]
                        ex = X[j, 0] - X[i, 0]; ey = X[j, 1] - X[i, 1]; ez = X[j, 2] - X[i, 2]
                        if ex * ex + ey * ey + ez * ez <= r2:
                            wj = w[j]; s += wj; c += 1
                            m0 += wj * ex; m1 += wj * ey; m2 += wj * ez
                            q00 += wj * ex * ex; q01 += wj * ex * ey; q02 += wj * ex * ez
                            q11 += wj * ey * ey; q12 += wj * ey * ez; q22 += wj * ez * ez
        cnt[i] = c; sw[i] = s
        if s > 0:
            m0 /= s; m1 /= s; m2 /= s
            mean[i, 0] = m0; mean[i, 1] = m1; mean[i, 2] = m2
            cov[i, 0, 0] = q00 / s - m0 * m0; cov[i, 0, 1] = q01 / s - m0 * m1; cov[i, 0, 2] = q02 / s - m0 * m2
            cov[i, 1, 1] = q11 / s - m1 * m1; cov[i, 1, 2] = q12 / s - m1 * m2; cov[i, 2, 2] = q22 / s - m2 * m2
            cov[i, 1, 0] = cov[i, 0, 1]; cov[i, 2, 0] = cov[i, 0, 2]; cov[i, 2, 1] = cov[i, 1, 2]
    return sw, mean, cov, cnt


def smooth(xyz_mm, raw_n, w, sets, r_mm, kmin=KMIN):
    """New receiver normals (n, 3). xyz in mm; raw_n unit; w = area (b) x opacity; sets = int label-set id per surfel.
    Returns normals, kept_raw (bool), count."""
    xyz_mm = np.ascontiguousarray(xyz_mm, float); raw_n = np.asarray(raw_n, float); w = np.ascontiguousarray(w, float)
    out = raw_n.copy(); kept = np.ones(len(raw_n), bool); count = np.zeros(len(raw_n), np.int64)
    for s in np.unique(sets):
        idx = np.nonzero(sets == s)[0]
        if len(idx) < kmin:
            count[idx] = len(idx); continue
        sw, _, cov, cnt = _moments(xyz_mm[idx], w[idx], float(r_mm))
        count[idx] = cnt
        ok = (cnt >= kmin) & (sw > 0)
        if not ok.any():
            continue
        _, vec = np.linalg.eigh(cov[ok])
        nn = vec[:, :, 0]
        sg = np.sign((nn * raw_n[idx[ok]]).sum(1)); sg[sg == 0] = 1.0
        nn = nn * sg[:, None]; nn /= np.linalg.norm(nn, axis=1, keepdims=True)
        out[idx[ok]] = nn; kept[idx[ok]] = False
    return out, kept, count


def plus_z(n):
    n = np.asarray(n, float).copy(); n[n[:, 2] < 0] *= -1
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def noise_metric(xyz, n, r=2.0, kmin=6, max_pts=20000, seed=0, with_rms=False):
    """Traits `neighbour_normal_noise` (recon_traits.py), same sampling; also returns the per-sample angles and RMS point-to-plane distance."""
    xyz = np.asarray(xyz, float); n = plus_z(n); rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), min(max_pts, len(xyz)), replace=False); T = cKDTree(xyz); ang = []; rms = []
    for i in idx:
        nb_ = T.query_ball_point(xyz[i], r)
        if len(nb_) < kmin:
            continue
        Q = xyz[nb_] - xyz[nb_].mean(0); _, sv, vt = np.linalg.svd(Q, full_matrices=False); pn = vt[-1]
        ang.append(np.degrees(np.arccos(min(1.0, abs(float(pn @ n[i]))))))
        rms.append(float(sv[-1] / math.sqrt(len(nb_))))
    ang = np.array(ang); rms = np.array(rms)
    med = float(np.median(ang)) if len(ang) else float('nan')
    return (med, ang, rms) if with_rms else (med, len(ang))


def reframe(n_new, tu):
    """v2: disc frame for a new normal. tu' = tu with its n'-component removed (renormalised), tv' = n' x tu'. Degenerate tu (parallel to n') -> any perpendicular."""
    n_new = np.asarray(n_new, float); tu2 = tu - (tu * n_new).sum(1, keepdims=True) * n_new
    nr = np.linalg.norm(tu2, axis=1, keepdims=True); bad = nr[:, 0] < 1e-9
    if bad.any():
        alt = np.cross(n_new[bad], np.array([1.0, 0.0, 0.0])); alt2 = np.cross(n_new[bad], np.array([0.0, 1.0, 0.0]))
        alt = np.where((np.linalg.norm(alt, axis=1) < 1e-6)[:, None], alt2, alt); tu2[bad] = alt; nr = np.linalg.norm(tu2, axis=1, keepdims=True)
    tu2 = tu2 / nr
    return tu2, np.cross(n_new, tu2)


def correlated_field(X_mm, cell_mm, rng):
    """v2 report-only: smooth 3-vector Gaussian field (trilinear interpolation of white noise on a lattice of `cell_mm`), unit variance per component."""
    lo = X_mm.min(0) - cell_mm; g = (X_mm - lo) / cell_mm; i0 = np.floor(g).astype(int); f = g - i0
    dims = i0.max(0) + 2; G = rng.normal(size=(dims[0], dims[1], dims[2], 3)); F = np.zeros((len(X_mm), 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                F += w[:, None] * G[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return F / F.std()


def correlated_rotate(n, tu, tv, X_mm, scale_deg, rng, cell_mm=3.0):
    """Tilt every normal by a spatially correlated tilt vector (tangent projection of the smooth field, components ~ N(0, scale_deg^2))."""
    F = correlated_field(X_mm, cell_mm, rng) * np.radians(scale_deg); t = F - (F * n).sum(1, keepdims=True) * n
    ang = np.linalg.norm(t, axis=1); axis = np.cross(n, t)
    return rotate_frames(n, tu, tv, ang, axis)


def rotate_frames(n, tu, tv, angle_rad, axis_seed_vec):
    """Rotate each frame (n, tu, tv) by angle about the unit axis = component of axis_seed_vec perpendicular to n (Rodrigues)."""
    a = axis_seed_vec - (axis_seed_vec * n).sum(1, keepdims=True) * n; a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)   # v2: zero axis (zero tilt) = identity
    c = np.cos(angle_rad)[:, None]; s = np.sin(angle_rad)[:, None]
    def rot(v):
        return v * c + np.cross(a, v) * s + a * (a * v).sum(1, keepdims=True) * (1 - c)
    n2 = rot(n); tu2 = rot(tu); tv2 = rot(tv)
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True); tu2 -= (tu2 * n2).sum(1, keepdims=True) * n2; tu2 /= np.linalg.norm(tu2, axis=1, keepdims=True)
    tv2 = np.cross(n2, tu2)
    return n2, tu2, tv2
