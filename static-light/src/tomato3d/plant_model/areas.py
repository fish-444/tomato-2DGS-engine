"""Two areas per organ (contract section 4).

(a) summed disc area = sum pi k^2 s0 s1, k = 1.5.
(b) = (a) x r, r = median over views of union pixel area / sum of projected disc areas. Each disc is projected with the camera's
local affine Jacobian at its centre: semi-axis vectors J (k s0 tu), J (k s1 tv). Union = pixel centres inside >= 1 projected ellipse;
the sum = sum pi |det M|. The organ is rendered alone.
"""

from __future__ import annotations

import numpy as np

K_ELLIPSE = 1.5            # light_organs.FLOORS["ellipse_k"]
N_VIEWS = 8
FOCAL_TARGET_PX = 3000.0
WINDOWS = (8, 24, 72)      # half-window sizes (px) tried in turn; a disc larger than the last is drawn with its own window


def disc_area_m2(scale: np.ndarray) -> np.ndarray:
    return np.pi * K_ELLIPSE ** 2 * scale[:, 0] * scale[:, 1]


def pick_views(poses: dict, n: int = N_VIEWS) -> list[str]:
    acc = sorted(poses.get("accepted") or poses["views"])
    acc = [v for v in acc if v in poses["views"]]
    if len(acc) <= n:
        return acc
    return [acc[i] for i in np.linspace(0, len(acc) - 1, n).round().astype(int)]


def camera(poses: dict, view: str):
    K0 = np.asarray(poses["K_undist"], float); res0 = poses["undistorted_res"]
    res = int(round(res0 * FOCAL_TARGET_PX / K0[0, 0]))
    s = res / res0; K = K0.copy(); K[0, 0] *= s; K[1, 1] *= s; K[0, 2] = (K0[0, 2] + 0.5) * s - 0.5; K[1, 2] = (K0[1, 2] + 0.5) * s - 0.5
    pv = poses["views"][view]
    return np.asarray(pv["R_cw"], float), np.asarray(pv["t_cw"], float), K, res


def projected_ellipses(xyz, tu, tv, scale, R, t, K):
    """Centres (N, 2) and 2x2 axis matrices M (N, 2, 2) (columns = projected semi-axis vectors), plus depth."""
    Xc = xyz @ R.T + t; z = Xc[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    c = np.c_[fx * Xc[:, 0] / z + K[0, 2], fy * Xc[:, 1] / z + K[1, 2]]
    def J(vw):                                           # d(u, v) / d(world) applied to a world vector
        vc = vw @ R.T
        du = fx * (vc[:, 0] / z - Xc[:, 0] * vc[:, 2] / z ** 2); dv = fy * (vc[:, 1] / z - Xc[:, 1] * vc[:, 2] / z ** 2)
        return np.c_[du, dv]
    a = J(tu * (K_ELLIPSE * scale[:, :1])); b = J(tv * (K_ELLIPSE * scale[:, 1:2]))
    M = np.stack([a, b], -1)
    return c, M, z


def union_and_sum_px(c, M, z, res: int) -> tuple[float, float]:
    ok = z > 0
    c, M = c[ok], M[ok]
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    total = float(np.pi * np.abs(det).sum())
    good = np.abs(det) > 1e-9
    c, M, det = c[good], M[good], det[good]
    if len(c) == 0:
        return 0.0, total
    inv = np.stack([np.stack([M[:, 1, 1], -M[:, 0, 1]], -1), np.stack([-M[:, 1, 0], M[:, 0, 0]], -1)], 1) / det[:, None, None]
    rad = np.sqrt((M ** 2).sum(axis=(1, 2)))                        # bounding radius
    lo = np.floor(c.min(0) - rad.max() - 2).astype(int); hi = np.ceil(c.max(0) + rad.max() + 2).astype(int)
    W, H = int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)
    canvas = np.zeros((H, W), bool)
    remaining = np.ones(len(c), bool)
    for half in WINDOWS + (None,):
        if half is None:
            sel = np.nonzero(remaining)[0]
            for i in sel:                                            # rare very large discs
                h = int(np.ceil(rad[i])) + 1
                _draw(canvas, c[i:i + 1], inv[i:i + 1], h, lo)
            break
        sel = np.nonzero(remaining & (rad <= half))[0]
        for i0 in range(0, len(sel), max(1, 200000 // (2 * half + 1) ** 2)):
            idx = sel[i0:i0 + max(1, 200000 // (2 * half + 1) ** 2)]
            _draw(canvas, c[idx], inv[idx], half, lo)
        remaining[sel] = False
    return float(canvas.sum()), total


def _draw(canvas, c, inv, half, lo):
    off = np.arange(-half, half + 1)
    ox, oy = np.meshgrid(off, off); ox = ox.ravel(); oy = oy.ravel()
    base = np.round(c).astype(int)                                   # pixel centres at integer coordinates (projection convention)
    px = base[:, :1] + ox[None]; py = base[:, 1:2] + oy[None]
    dx = px - c[:, :1]; dy = py - c[:, 1:2]
    q0 = inv[:, 0, :1] * dx + inv[:, 0, 1:2] * dy; q1 = inv[:, 1, :1] * dx + inv[:, 1, 1:2] * dy
    inside = q0 * q0 + q1 * q1 <= 1.0
    xs = px[inside] - lo[0]; ys = py[inside] - lo[1]
    canvas[ys, xs] = True


def silhouette_ratio(xyz, tu, tv, scale, poses: dict, views: list[str]) -> dict:
    per = {}
    for v in views:
        R, t, K, res = camera(poses, v)
        c, M, z = projected_ellipses(xyz, tu, tv, scale, R, t, K)
        u, s = union_and_sum_px(c, M, z, res)
        per[v] = dict(union_px=u, sum_px=s, ratio=(u / s) if s > 0 else None)
    rs = [p["ratio"] for p in per.values() if p["ratio"] is not None]
    return dict(ratio=float(np.median(rs)) if rs else None, ratio_min=float(min(rs)) if rs else None, ratio_max=float(max(rs)) if rs else None,
                n_views=len(rs), per_view=per)
