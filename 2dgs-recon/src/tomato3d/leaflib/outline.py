"""Outline extraction and normalisation of one mask (declared parameters at the top), plus shape descriptors.

Normalised frame: principal axis = length, u along it from the base (0) to the tip (1), v across in units of the length;
N_POINTS points resampled by arc length, counter-clockwise, starting at the base. The base end is the end nearer the
attachment cue: the nearest rachis-skeleton pixel of the photo (`rachis_skeleton`), else the end nearer the given plant
centre (`plant_centre`); the rule used is recorded per outline.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

HOLE_MAX_FRAC = 0.02       # holes smaller than this fraction of the mask area are filled
SMOOTH_SIGMA_PX = 1.5      # Gaussian smoothing of the contour coordinates along the curve (grid px); serrations kept
N_POINTS = 64              # as the engine's leaflet_outline library
MIN_AREA_PX = 150


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); return lab == k


def fill_small_holes(mask: np.ndarray, max_frac: float = HOLE_MAX_FRAC) -> np.ndarray:
    inv = (~mask).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = mask.copy(); area = mask.sum()
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h and a < max_frac * area:     # interior hole, not touching the border
            out[lab == i] = True
    return out


def contour_of(mask: np.ndarray) -> np.ndarray:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if SMOOTH_SIGMA_PX > 0 and len(c) > 8:
        from scipy.ndimage import gaussian_filter1d
        c = np.c_[gaussian_filter1d(c[:, 0], SMOOTH_SIGMA_PX, mode="wrap"), gaussian_filter1d(c[:, 1], SMOOTH_SIGMA_PX, mode="wrap")]
    return c


def resample(c: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    P = np.vstack([c, c[:1]]); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
    s = np.linspace(0, cum[-1], n, endpoint=False); x = np.interp(s, cum, P[:, 0]); y = np.interp(s, cum, P[:, 1])
    return np.c_[x, y]


def normalise(contour_px: np.ndarray, base_cue_px=None, centre_px=None) -> dict:
    """Rotate to the principal axis, pick the base end, scale to unit length, resample. Returns points + descriptors."""
    P = np.asarray(contour_px, float); c = P.mean(0); X = P - c
    w, v = np.linalg.eigh(np.cov(X.T)); ax = v[:, int(np.argmax(w))]; per = np.array([-ax[1], ax[0]])
    s = X @ ax; t = X @ per
    ends = {"lo": c + ax * s.min(), "hi": c + ax * s.max()}
    if base_cue_px is not None:
        rule = "rachis_skeleton"; q = np.asarray(base_cue_px, float)
    elif centre_px is not None:
        rule = "plant_centre"; q = np.asarray(centre_px, float)
    else:
        rule = "narrow_end_is_tip"; q = None
    if q is not None:
        base_is_lo = np.linalg.norm(ends["lo"] - q) <= np.linalg.norm(ends["hi"] - q)
    else:
        L = s.max() - s.min(); lo_w = np.abs(t[s < s.min() + 0.15 * L]).mean(); hi_w = np.abs(t[s > s.max() - 0.15 * L]).mean(); base_is_lo = lo_w >= hi_w
    if not base_is_lo:
        s, t = -s, -t
    L = float(s.max() - s.min()); u = (s - s.min()) / L; vv = t / L
    pts = np.c_[u, vv]
    area2 = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    if area2 < 0:
        pts = pts[::-1]
    k0 = int(np.argmin(pts[:, 0])); pts = np.roll(pts, -k0, axis=0)                     # start at the base
    pts = resample(pts, N_POINTS)
    return dict(points=[[round(float(a), 5), round(float(b), 5)] for a, b in pts], base_rule=rule, length_px=L, **descriptors(pts))


def descriptors(pts: np.ndarray) -> dict:
    """Dimensionless shape descriptors of a normalised outline (u 0..1, v in units of length)."""
    P = np.asarray(pts, float); width = float(P[:, 1].max() - P[:, 1].min())
    area = float(abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])) / 2)
    hull = cv2.convexHull(P.astype(np.float32)).reshape(-1, 2); hull_area = float(abs(np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1])) / 2)
    per = float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()); hper = float(np.linalg.norm(np.diff(np.vstack([hull, hull[:1]]), axis=0), axis=1).sum())
    # asymmetry about the base-tip chord (the midrib proxy), not about the PCA axis through the centroid: a half blade is
    # nearly symmetric about its own centroid axis but strongly one-sided about its base-tip chord
    ib, it = int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0])); vb, vt = P[ib, 1], P[it, 1]
    vrel = P[:, 1] - (vb + (vt - vb) * P[:, 0])
    left = float(np.clip(-vrel, 0, None).mean()); right = float(np.clip(vrel, 0, None).mean())
    widest_u = float(P[int(np.argmax(np.abs(P[:, 1]))), 0])
    return dict(lw_apparent=1.0 / width if width else None, width_rel=width, solidity=area / hull_area if hull_area else None,
                serration_index=per / hper if hper else None, asymmetry=abs(left - right) / (left + right) if (left + right) else None,
                area_rel=area, widest_fraction=widest_u)


def mask_outline(mask_grid: np.ndarray, base_cue_px=None, centre_px=None) -> dict | None:
    m = largest_component(mask_grid)
    if m.sum() < MIN_AREA_PX:
        return None
    m = fill_small_holes(m); c = contour_of(m)
    if len(c) < 8:
        return None
    rec = normalise(c, base_cue_px, centre_px); rec["area_px"] = int(m.sum()); rec["perimeter_px"] = float(cv2.arcLength(c.astype(np.float32), True))
    return rec


def nearest_skeleton_point(skel: np.ndarray, mask: np.ndarray, max_dist_px: float = 40.0):
    """Nearest rachis-skeleton pixel to the mask boundary within max_dist_px, else None."""
    if not skel.any():
        return None
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(skel); d = dist[ys, xs]; k = int(np.argmin(d))
    return (float(xs[k]), float(ys[k])) if d[k] <= max_dist_px else None
