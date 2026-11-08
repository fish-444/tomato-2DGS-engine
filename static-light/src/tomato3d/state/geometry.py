"""MTG A1/A2 geometric quantities from vertex points - one implementation for 3-D and 2-D.

Definitions (``handoff/MTG_CONVENTION_2026-10-10.md`` sections 11-12):

* azimuth (12.1): direction projected on the horizontal plane, from +X toward +Y
  (counter-clockwise seen from +Z), [0, 360).
* elevation (A1.4): ``atan2(dz, hypot(dx, dy))``, + = upward, -90..+90.
* insertion angle (12.2): stem axis above the node vs petiole (P0 -> P1), 0..180.
* rachis bends (12.3): segments Petiole (P0 -> node 1), R1 (node 1 -> node 2), ...,
  R_m (node m -> rachis end). alpha_i = angle at node i between the segment ending there and
  the one starting there; **positive when the distal segment points lower** (elevation drops).
  ``rachis_curvature_deg`` = signed sum.
* theta (A1.5): angle between P0 -> P1 and P2 -> P3, P1 = node 1, P3 = last pair node,
  P2 = midpoint of the node path between P1 and P3 (derived, not stored).
* chord_to_path_ratio (MTG 6): |P0 -> P3| / path length P0 -> node 1 -> ... -> P3.

Points are arrays of shape (n, d) with d = 3 (x, y, z-up) or d = 2 (horizontal, up) for a side
view projected into image coordinates converted to (right, up). Nothing here classifies
flat/bent: thresholds are on hold (MTG 8.4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else np.asarray(v, float) / n


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    ua, ub = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    if ua is None or ub is None:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ua, ub))))))


def azimuth_deg(v: np.ndarray) -> float | None:
    """A2 12.1. ``None`` for a vertical direction (no horizontal component)."""
    v = np.asarray(v, float)
    if len(v) != 3 or math.hypot(v[0], v[1]) < 1e-12:
        return None
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def elevation_deg(v: np.ndarray) -> float | None:
    """A1.4 for 3-D; for 2-D (horizontal, up) the same atan2 rule."""
    v = np.asarray(v, float)
    if float(np.linalg.norm(v)) < 1e-12:
        return None
    horizontal = math.hypot(v[0], v[1]) if len(v) == 3 else abs(float(v[0]))
    return math.degrees(math.atan2(float(v[-1]), horizontal))


def wrap_deg(a: float) -> float:
    """Signed difference wrapped into (-180, 180]."""
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def insertion_angle_deg(stem_above_dir: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float | None:
    """A2 12.2: angle between the stem direction above the node and the petiole P0 -> P1."""
    return angle_between_deg(np.asarray(stem_above_dir, float), np.asarray(p1, float) - np.asarray(p0, float))


def bend_alphas_deg(p0: np.ndarray, nodes: Sequence[np.ndarray], rachis_end: np.ndarray) -> list[float] | None:
    """A2 12.3 signed bends at each lateral-pair node, length ``len(nodes)``."""
    if not nodes:
        return None
    pts = [np.asarray(p0, float), *(np.asarray(n, float) for n in nodes), np.asarray(rachis_end, float)]
    alphas: list[float] = []
    for i in range(1, len(pts) - 1):
        prox, dist = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
        mag = angle_between_deg(prox, dist)
        e_prox, e_dist = elevation_deg(prox), elevation_deg(dist)
        if mag is None or e_prox is None or e_dist is None:
            return None
        alphas.append(mag if e_dist <= e_prox else -mag)
    return alphas


def path_midpoint(points: Sequence[np.ndarray]) -> np.ndarray:
    """Point at half the arc length of the polyline through ``points``."""
    pts = np.asarray(points, float)
    if len(pts) == 1:
        return pts[0]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    half = s[-1] / 2.0
    k = int(np.searchsorted(s, half, side="right") - 1)
    k = min(k, len(seg) - 1)
    t = 0.0 if seg[k] < 1e-12 else (half - s[k]) / seg[k]
    return pts[k] + t * (pts[k + 1] - pts[k])


def theta_deg(p0: np.ndarray, nodes: Sequence[np.ndarray]) -> float | None:
    """A1.5 theta. Needs at least two pair nodes (P1 != P3); otherwise unavailable."""
    if len(nodes) < 2:
        return None
    p1, p3 = np.asarray(nodes[0], float), np.asarray(nodes[-1], float)
    p2 = path_midpoint(nodes)
    return angle_between_deg(p1 - np.asarray(p0, float), p3 - p2)


def chord_to_path_ratio(p0: np.ndarray, nodes: Sequence[np.ndarray]) -> float | None:
    if not nodes:
        return None
    pts = np.asarray([p0, *nodes], float)
    path = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if path < 1e-12:
        return None
    return float(np.linalg.norm(pts[-1] - pts[0])) / path


def point_at_arc_length(polyline: np.ndarray, s_mm: float) -> np.ndarray:
    """Point at arc length ``s_mm`` along ``polyline`` (clamped to its ends)."""
    pts = np.asarray(polyline, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_mm = min(max(float(s_mm), 0.0), float(s[-1]))
    k = min(int(np.searchsorted(s, s_mm, side="right") - 1), len(seg) - 1)
    t = 0.0 if seg[k] < 1e-12 else (s_mm - s[k]) / seg[k]
    return pts[k] + t * (pts[k + 1] - pts[k])


def tangent_at_arc_length(polyline: np.ndarray, s_mm: float) -> np.ndarray:
    pts = np.asarray(polyline, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    k = min(max(int(np.searchsorted(s, float(s_mm), side="right") - 1), 0), len(seg) - 1)
    return pts[k + 1] - pts[k]


def stem_direction_above(stem_points: np.ndarray, z_mm: float) -> np.ndarray:
    """Direction of the stem segment just above height ``z_mm`` (plant-local, +Z up)."""
    pts = np.asarray(stem_points, float)
    for k in range(len(pts) - 1):
        if pts[k + 1, 2] > z_mm + 1e-9:
            return pts[k + 1] - pts[k]
    return pts[-1] - pts[-2]


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0


def triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = np.asarray(verts, float)
    f = np.asarray(faces, int)
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def area_weighted_normal(verts: np.ndarray, faces: np.ndarray) -> np.ndarray | None:
    v = np.asarray(verts, float)
    f = np.asarray(faces, int)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]).sum(axis=0)
    return _unit(n)

