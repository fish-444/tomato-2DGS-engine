"""Mesh generation for plant_tree_v1 organs. The tube and blade maths are the v1 generator's (tomato3d/synth/microtom.py)
transcribed as pure functions, run in v1's build frame (y up, mm) and converted to the tree frame (z up, mm), so that a
param_blade or tube regenerates the same vertices as v1. Outline-driven blades (option B) replace v1's analytic half-width
with the half-width profile of a sampled leaflet outline.
"""

from __future__ import annotations

import math

import numpy as np

UP_B = (0.0, 1.0, 0.0)          # build frame up (v1)
D2R = math.pi / 180


# ------------------------------------------------------------------ frames
def zup_to_build(p):
    """tree frame (x, y, z; z up) -> v1 build frame (x, z, -y)."""
    return [p[0], p[2], -p[1]]


def build_to_zup(p):
    """v1 build frame (x, y, z; y up) -> tree frame (x, -z, y)."""
    return [p[0], -p[2], p[1]]


def _add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def _sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def _mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    ln = math.hypot(a[0], a[1], a[2]) or 1
    return [a[0] / ln, a[1] / ln, a[2] / ln]


def tri_area(a, b, c):
    x = _cross(_sub(b, a), _sub(c, a)); return 0.5 * math.hypot(x[0], x[1], x[2])


# ------------------------------------------------------------------ tube (v1 `tube`)
def tube_mesh(polyline_zup, radii, segs: int = 6):
    """Vertices (z up) and faces of a tube around a polyline, exactly as v1 builds it (frame-propagated normals, first
    normal from the build-frame up vector)."""
    pts = [zup_to_build(p) for p in polyline_zup]; UP = list(UP_B)
    V, F = [], []; prevN = None
    for i in range(len(pts)):
        t = _norm(_sub(pts[min(i + 1, len(pts) - 1)], pts[max(i - 1, 0)]))
        n = _norm(_sub(prevN, _mul(t, _dot(prevN, t)))) if prevN else _norm(_cross(t, UP) if abs(t[1]) < 0.9 else _cross(t, [1, 0, 0]))
        prevN = n; b = _cross(t, n)
        for j in range(segs):
            a = j / segs * 2 * math.pi
            V.append(_add(pts[i], _add(_mul(n, math.cos(a) * radii[i]), _mul(b, math.sin(a) * radii[i]))))
    for i in range(len(pts) - 1):
        for j in range(segs):
            a0 = i * segs + j; a1 = i * segs + (j + 1) % segs; b0 = a0 + segs; b1 = a1 + segs
            F.append([a0, a1, b1]); F.append([a0, b1, b0])
    return [build_to_zup(p) for p in V], F


# ------------------------------------------------------------------ blade (v1 `blade`, generalised half-width)
def v1_half_width(u: float, W: float) -> float:
    return W / 2 * math.pow(math.sin(math.pi * math.pow(u, 0.82)), 0.85) * (1 - 0.18 * u) / 0.93


def blade_mesh(base_zup, d_zup, n_zup, L, W, cup, bend, nu: int = 12, nv: int = 7, outline=None):
    """Vertices (z up), faces and one-sided area of a leaflet blade. Without `outline` this is v1's analytic blade; with
    an outline (normalised polygon, u along the midrib 0..1, v across in units of the length) the half-widths at each u
    come from the outline (left and right separately), so library leaflets keep their own margin shape."""
    base = zup_to_build(base_zup); d = _norm(zup_to_build(d_zup)); nrm = _norm(zup_to_build(n_zup))
    b = _norm(_cross(nrm, d)); V, F = [], []
    prof = outline_profile(outline, nu) if outline else None
    for i in range(nu + 1):
        u = i / nu
        for j in range(nv + 1):
            s = j / nv * 2 - 1
            if prof is None:
                off = s * v1_half_width(u, W)
            else:
                lo, hi = prof[i]                                # signed half-widths (left < 0 < right) in units of L
                off = (hi if s >= 0 else -lo) * s * L
            p = _add(base, _mul(d, u * L)); p = _add(p, _mul(b, off))
            p = _add(p, _mul(nrm, cup * W * 0.28 * s * s - bend * L * 0.22 * u * u))
            V.append(p)
    area = 0.0
    for i in range(nu):
        for j in range(nv):
            a0 = i * (nv + 1) + j; a1 = a0 + 1; b0 = a0 + nv + 1; b1 = b0 + 1
            F.append([a0, b0, b1]); F.append([a0, b1, a1])
            area += tri_area(V[a0], V[b0], V[b1]) + tri_area(V[a0], V[b1], V[a1])
    return [build_to_zup(p) for p in V], F, area


def outline_profile(outline: dict, nu: int):
    """Left/right half-width (negative/positive, units of the leaflet length) at nu+1 stations along the midrib, read
    from a normalised outline polygon (points [u, v], u in [0, 1] base->tip, v across)."""
    pts = np.asarray(outline["points"], float)
    u = np.linspace(0, 1, nu + 1); out = []
    for ui in u:
        # intersect the polygon with the line u = ui
        vs = []
        for k in range(len(pts)):
            p, q = pts[k], pts[(k + 1) % len(pts)]
            if (p[0] - ui) * (q[0] - ui) <= 0 and p[0] != q[0]:
                t = (ui - p[0]) / (q[0] - p[0]); vs.append(p[1] + t * (q[1] - p[1]))
        if len(vs) >= 2:
            out.append((float(min(vs)), float(max(vs))))
        else:
            out.append((0.0, 0.0))
    return out


# ------------------------------------------------------------------ normalised outlines from measured contours
def normalise_contour(contour_mm) -> dict:
    """A dissected-leaflet contour (mm, any orientation) -> outline polygon in the leaflet frame: u along the major axis
    from base (0) to tip (1), v across in units of the length. The tip is the narrower end (petiolule end is the wider,
    truncated one is not guaranteed, so the end with the smaller mean half-width over its last 15 % is called the tip)."""
    P = np.asarray(contour_mm, float); c = P.mean(0); X = P - c
    w, v = np.linalg.eigh(np.cov(X.T)); ax = v[:, int(np.argmax(w))]; per = np.array([-ax[1], ax[0]])
    s = X @ ax; t = X @ per; L = float(s.max() - s.min())
    lo_end = np.abs(t[s < s.min() + 0.15 * L]).mean(); hi_end = np.abs(t[s > s.max() - 0.15 * L]).mean()
    if hi_end > lo_end:                                          # tip must be the narrow end
        s, t = -s, -t
    u = (s - s.min()) / L; vv = t / L
    # keep polygon order (contours are simple polygons); ensure counter-clockwise for a stable winding
    pts = np.c_[u, vv]
    area2 = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    if area2 < 0:
        pts = pts[::-1]
    width = float(t.max() - t.min())
    return dict(kind="outline_polygon", points=[[round(float(a), 5), round(float(b), 5)] for a, b in pts],
                length_mm=L, width_mm=width, lw_ratio=L / width if width else None)


# ------------------------------------------------------------------ descriptors from stored meshes
def ring_centreline(vertices, segs: int):
    """Centre and mean radius of each ring of a tube mesh (vertices in rings of `segs`)."""
    V = np.asarray(vertices, float); n = len(V) // segs
    C = V[:n * segs].reshape(n, segs, 3); ctr = C.mean(1); rad = np.linalg.norm(C - ctr[:, None], axis=2).mean(1)
    return ctr.tolist(), rad.tolist()


def grid_descriptors(vertices, nu: int, nv: int) -> dict:
    """Base, tip, direction, chord length, max width and mean normal of a (nu+1) x (nv+1) blade grid."""
    G = np.asarray(vertices, float).reshape(nu + 1, nv + 1, 3)
    base = G[0].mean(0); tip = G[-1].mean(0); dvec = tip - base; L = float(np.linalg.norm(dvec)); d = dvec / (L or 1)
    width = float(max(np.linalg.norm(G[i, -1] - G[i, 0]) for i in range(nu + 1)))
    # normal: mean of face normals
    nrm = np.zeros(3)
    for i in range(nu):
        for j in range(nv):
            a, b, c = G[i, j], G[i + 1, j], G[i, j + 1]; nrm += np.cross(b - a, c - a)
    nn = np.linalg.norm(nrm); nrm = nrm / nn if nn else np.array([0, 0, 1.0])
    if nrm[2] < 0:
        nrm = -nrm
    bvec = np.cross(nrm, d); bvec /= (np.linalg.norm(bvec) or 1)
    return dict(base=base.tolist(), tip=tip.tolist(), d=d.tolist(), n=nrm.tolist(), b=bvec.tolist(), length_mm=L, width_mm=width)


def grid_outline(vertices, nu: int, nv: int, base, d, b):
    """Outline polygon of a blade grid in the leaflet frame (u, v in units of the chord length): boundary ring of the grid."""
    G = np.asarray(vertices, float).reshape(nu + 1, nv + 1, 3)
    ring = [G[0, j] for j in range(nv + 1)] + [G[i, nv] for i in range(1, nu + 1)] + [G[nu, j] for j in range(nv - 1, -1, -1)] + [G[i, 0] for i in range(nu - 1, 0, -1)]
    base = np.asarray(base); d = np.asarray(d); b = np.asarray(b); L = float(np.linalg.norm(G[-1].mean(0) - G[0].mean(0))) or 1.0
    return dict(kind="outline_polygon", points=[[round(float((p - base) @ d / L), 5), round(float((p - base) @ b / L), 5)] for p in ring])


def polyline_length(pts) -> float:
    P = np.asarray(pts, float); return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def polyline_turning_deg(pts) -> float:
    """Sum of turning angles (deg) along a polyline."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return 0.0
    T = np.diff(P, axis=0); T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    cosang = np.clip((T[:-1] * T[1:]).sum(1), -1, 1)
    return float(np.degrees(np.arccos(cosang)).sum())


def point_at_arclength(pts, s: float):
    P = np.asarray(pts, float); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
    s = min(max(s, 0.0), float(cum[-1])); k = int(np.searchsorted(cum, s, side="right") - 1); k = min(k, len(seg) - 1)
    t = (s - cum[k]) / (seg[k] or 1)
    return (P[k] + t * (P[k + 1] - P[k])).tolist()


def arclength_of_nearest(pts, q) -> float:
    """Arc length along the polyline of the point nearest to q."""
    P = np.asarray(pts, float); q = np.asarray(q, float); seg = np.diff(P, axis=0); ln = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(ln)]; best = (1e30, 0.0)
    for k in range(len(seg)):
        if ln[k] == 0:
            continue
        t = float(np.clip((q - P[k]) @ seg[k] / (ln[k] ** 2), 0, 1)); d = float(np.linalg.norm(P[k] + t * seg[k] - q))
        if d < best[0]:
            best = (d, float(cum[k] + t * ln[k]))
    return best[1]
