"""Per-leaflet curvature fit and quadric surface (CONTRACT 'Per-leaflet fit', '_04')."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

R_SMOOTH = 2.0; MIN_N = 30; RASTER = 0.25


def smooth_normals(X, N, r=R_SMOOTH):
    Y = X - X.mean(0); w, v = np.linalg.eigh(np.cov(Y.T)); pn = v[:, 0] * (1 if v[2, 0] >= 0 else -1)
    Nn = N / np.maximum(np.linalg.norm(N, axis=1)[:, None], 1e-12); Nn = Nn * np.where(Nn @ pn >= 0, 1.0, -1.0)[:, None]
    tr = cKDTree(X); nb = tr.query_ball_point(X, r); Ns = np.array([Nn[i].mean(0) for i in nb]); Ns /= np.maximum(np.linalg.norm(Ns, axis=1)[:, None], 1e-12)
    return Ns, pn


def frame(X, A, Ns, base):
    nbar = (Ns * A[:, None]).sum(0); nbar /= np.linalg.norm(nbar) or 1
    c = (X * A[:, None]).sum(0) / A.sum(); Y = X - X.mean(0); w, v = np.linalg.eigh(np.cov(Y.T)); a1 = v[:, 2]
    es = a1 - nbar * (a1 @ nbar); es /= np.linalg.norm(es) or 1
    if es @ (c - np.asarray(base)) < 0:
        es = -es
    ed = np.cross(nbar, es)
    return c, es, ed, nbar


def fit(X, A, N, base):
    """returns dict of curvature traits; None if too few surfels."""
    if len(X) < MIN_N:
        return None
    Ns, _ = smooth_normals(X, N); c, es, ed, nb = frame(X, A, Ns, base)
    s = (X - c) @ es; d = (X - c) @ ed; z = (X - c) @ nb
    M = np.c_[np.ones_like(s), s, d, 0.5 * s ** 2, 0.5 * d ** 2, s * d]; w = np.sqrt(A)
    coef, *_ = np.linalg.lstsq(M * w[:, None], z * w, rcond=None); zh = M @ coef
    zm = (A * z).sum() / A.sum(); ss_res = (A * (z - zh) ** 2).sum(); ss_tot = (A * (z - zm) ** 2).sum(); r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    L = float(np.percentile(s, 99) - np.percentile(s, 1)); W = float(np.percentile(d, 99) - np.percentile(d, 1))
    dev = np.degrees(np.arccos(np.clip(Ns @ nb, -1, 1)))
    return dict(kappa_l=float(coef[3]), kappa_w=float(coef[4]), twist_c=float(coef[5]), a0=float(coef[0]), a1=float(coef[1]), a2=float(coef[2]), r2=float(r2), n=int(len(X)),
                length_mm=L, width_mm=W, dalpha_l_deg=float(np.degrees(coef[3] * L / 2)), dalpha_w_deg=float(np.degrees(coef[4] * W / 2)),
                ndev_median_deg=float(np.median(dev)), ndev_p90_deg=float(np.percentile(dev, 90)), centroid=c.tolist(), e_s=es.tolist(), e_d=ed.tolist(), n_bar=nb.tolist(),
                rms_residual_mm=float(np.sqrt(ss_res / A.sum())))


def proj_ratio(V, F):
    V = np.asarray(V, float); F = np.asarray(F, int)
    if len(F) == 0:
        return float("nan")
    a3 = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    lo = V[:, :2].min(0) - 1; hi = V[:, :2].max(0) + 1; nx, ny = (np.ceil((hi - lo) / RASTER).astype(int) + 1)
    im = Image.new("1", (int(nx), int(ny)), 0); dr = ImageDraw.Draw(im); P = (V[:, :2] - lo) / RASTER
    for f in F:
        dr.polygon([tuple(P[i]) for i in f], fill=1)
    return float(np.array(im, dtype=bool).sum() * RASTER ** 2 / a3) if a3 > 0 else float("nan")


def mesh_area(V, F):
    V = np.asarray(V, float); F = np.asarray(F, int)
    return float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()) if len(F) else 0.0


def quadric_surface(Vmesh, F, fr, base, target_area=None):
    """alpha-mesh vertices -> (s, d) in the leaflet frame (about the centroid) -> z = quadric; (s, d) scaled about the base by k so area = target."""
    c = np.asarray(fr["centroid"]); es = np.asarray(fr["e_s"]); ed = np.asarray(fr["e_d"]); nb = np.asarray(fr["n_bar"]); V = np.asarray(Vmesh, float)
    s0 = (V - c) @ es; d0 = (V - c) @ ed; bs = (np.asarray(base) - c) @ es; bd = (np.asarray(base) - c) @ ed
    def build(k):
        s = bs + k * (s0 - bs); d = bd + k * (d0 - bd)
        z = fr["a0"] + fr["a1"] * s + fr["a2"] * d + 0.5 * fr["kappa_l"] * s ** 2 + 0.5 * fr["kappa_w"] * d ** 2 + fr["twist_c"] * s * d
        return c + np.outer(s, es) + np.outer(d, ed) + np.outer(z, nb)
    if target_area is None:
        return build(1.0), 1.0
    lo, hi = 0.05, 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mesh_area(build(mid), F) < target_area:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    k = 0.5 * (lo + hi); return build(k), k


def sample_mesh(V, F, dens, sig, rng):
    V = np.asarray(V, float); F = np.asarray(F, int); P, Nn = [], []
    for f in F:
        a, b, c = V[f]; cr = np.cross(b - a, c - a); ar = 0.5 * np.linalg.norm(cr)
        if ar <= 0:
            continue
        n = int(np.floor(ar * dens + rng.random()))
        if n == 0:
            continue
        u = rng.random(n); v = rng.random(n); m = u + v > 1; u[m] = 1 - u[m]; v[m] = 1 - v[m]; nr = cr / (2 * ar)
        P.append(a + u[:, None] * (b - a) + v[:, None] * (c - a) + nr[None] * rng.normal(0, sig, n)[:, None]); Nn.append(np.repeat(nr[None], n, 0))
    return (np.vstack(P), np.vstack(Nn)) if P else (np.zeros((0, 3)), np.zeros((0, 3)))
