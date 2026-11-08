"""Leaflet outlines: normalisation, triangulation, and the bent/rolled 3D surface.

Normalised contour frame: base at the origin, midrib along ``+x``, tip at ``x = 1``,
``y`` lateral (``+y`` = leaflet left). Contours are closed polygons without a repeated
last point.

Surface: ``u`` along the midrib and ``v`` lateral, scaled by the leaflet length and width
scale, then

1. transverse roll - the blade curls about the midrib, edges going down by ``roll_deg``;
2. twist - rigid rotation about the midrib by ``twist_deg``;
3. longitudinal bend - the midrib follows a downward arc of total angle ``bend_deg``;

and finally placement in the leaflet frame ``(heading, left, up)`` at ``base_mm``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """``n`` points equally spaced along a closed polyline, starting at ``points[0]``."""
    p = np.asarray(points, float)
    loop = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, loop[:, 0]), np.interp(t, s, loop[:, 1])], axis=1)


def _polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def normalize_contour(contour_xy: np.ndarray, n_points: int, *, bins: int = 20) -> dict[str, Any]:
    """Normalise an image contour (pixels) to the leaflet frame.

    Orientation: the principal axis is the midrib. Tomato leaflets are ovate - widest
    nearer the base - so the end closer to the widest cross-section is taken as the base
    (``orientation_method='widest_point_heuristic'``).
    """
    c = np.asarray(contour_xy, float).reshape(-1, 2)
    centroid = c.mean(axis=0)
    _, _, vt = np.linalg.svd(c - centroid, full_matrices=False)
    axis, normal = vt[0], np.array([-vt[0][1], vt[0][0]])
    x = (c - centroid) @ axis
    y = (c - centroid) @ normal
    edges = np.linspace(x.min(), x.max(), bins + 1)
    widths = np.zeros(bins)
    for i in range(bins):
        sel = (x >= edges[i]) & (x <= edges[i + 1])
        widths[i] = (y[sel].max() - y[sel].min()) if sel.any() else 0.0
    widest = (np.argmax(widths) + 0.5) / bins
    if widest > 0.5:
        x = -x
        y = -y
    length = float(x.max() - x.min())
    xn = (x - x.min()) / length
    base_band = xn < 0.05
    y0 = float(np.median(y[base_band])) if base_band.any() else 0.0
    yn = (y - y0) / length
    pts = np.stack([xn, yn], axis=1)
    if _polygon_area(pts) < 0:
        pts = pts[::-1]
    start = int(np.argmin(pts[:, 0]))
    pts = np.roll(pts, -start, axis=0)
    out = resample_closed(pts, n_points)
    return dict(points=out, length_px=length, aspect=float(np.ptp(out[:, 1])),
                widest_fraction=float(min(widest, 1 - widest)),
                orientation_method="widest_point_heuristic")


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test (vectorised)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x1, y1 = polygon[:, 0][None], polygon[:, 1][None]
    x2, y2 = np.roll(polygon[:, 0], -1)[None], np.roll(polygon[:, 1], -1)[None]
    crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1)
    return crosses.sum(axis=1) % 2 == 1


def triangulate_contour(contour: np.ndarray, grid_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a normalised contour: boundary + interior grid, Delaunay, keep inside triangles."""
    from scipy.spatial import Delaunay

    c = np.asarray(contour, float)
    gx = np.linspace(c[:, 0].min(), c[:, 0].max(), grid_n + 2)[1:-1]
    gy = np.linspace(c[:, 1].min(), c[:, 1].max(), max(3, grid_n // 2) + 2)[1:-1]
    grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    spacing = min(np.diff(gx).min() if len(gx) > 1 else 1.0, np.diff(gy).min() if len(gy) > 1 else 1.0)
    grid = grid[_inside(grid, c)]
    if len(grid):
        d = np.min(np.linalg.norm(grid[:, None] - c[None], axis=2), axis=1)
        grid = grid[d > 0.35 * spacing]
    uv = np.vstack([c, grid])
    tri = Delaunay(uv).simplices
    centroids = uv[tri].mean(axis=1)
    keep = _inside(centroids, c)
    return uv, tri[keep].astype(np.int32)


def surface_points(uv: np.ndarray, leaflet: dict[str, Any]) -> np.ndarray:
    """Map normalised ``uv`` to board/plant mm on the bent, rolled, twisted leaflet surface."""
    length = float(leaflet["length_mm"])
    x = uv[:, 0] * length
    y = uv[:, 1] * length * float(leaflet["width_scale"])
    half_width = max(float(np.abs(y).max()), 1e-6)
    roll = math.radians(float(leaflet["roll_deg"]))
    if roll > 1e-6:
        k = roll / half_width
        yr = np.sin(k * y) / k
        zr = -(1.0 - np.cos(k * y)) / k
    else:
        yr, zr = y, np.zeros_like(y)
    tw = math.radians(float(leaflet["twist_deg"]))
    yt = yr * math.cos(tw) - zr * math.sin(tw)
    zt = yr * math.sin(tw) + zr * math.cos(tw)
    bend = math.radians(float(leaflet["bend_deg"]))
    if bend > 1e-6:
        k = bend / length
        xb = np.sin(k * x) / k + zt * np.sin(k * x)
        zb = -(1.0 - np.cos(k * x)) / k + zt * np.cos(k * x)
    else:
        xb, zb = x, zt
    h, lft, up = (np.asarray(leaflet[k], float) for k in ("heading", "left", "up"))
    base = np.asarray(leaflet["base_mm"], float)
    return base[None] + xb[:, None] * h[None] + yt[:, None] * lft[None] + zb[:, None] * up[None]


# --------------------------------------------------------------------------- Addendum D.1, B5c mesh relief
def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, oriented consistently with the first face (two-sided leaves)."""
    v = np.asarray(verts, float)
    f = np.asarray(faces, int)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ref = fn[np.argmax(np.linalg.norm(fn, axis=1))]
    fn = np.where((fn @ ref)[:, None] < 0, -fn, fn)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def sample_height(height_img: np.ndarray, tex_uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a height map (uint8, 0.5 = flat) at texture coords; returns -1..1 (0.25 per unit).

    Texture convention as in Blender: s along the image width, t = 0 at the bottom row.
    """
    h = height_img.astype(np.float64) / 255.0
    H, W = h.shape[:2]
    x = np.clip(tex_uv[:, 0], 0, 1) * (W - 1)
    y = (1.0 - np.clip(tex_uv[:, 1], 0, 1)) * (H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = x - x0, y - y0
    val = (h[y0, x0] * (1 - fx) * (1 - fy) + h[y0, x1] * fx * (1 - fy) + h[y1, x0] * (1 - fx) * fy
           + h[y1, x1] * fx * fy)
    return np.clip((val - 0.5) / 0.25, -1.0, 1.0)


def displace(verts: np.ndarray, faces: np.ndarray, offsets_mm: np.ndarray) -> np.ndarray:
    """Move every vertex along its normal by ``offsets_mm`` (raised blade +, vein grooves -)."""
    return np.asarray(verts, float) + vertex_normals(verts, faces) * np.asarray(offsets_mm, float)[:, None]
