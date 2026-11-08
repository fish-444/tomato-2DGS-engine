"""Step 2 features: two-scale PCA shape descriptors + local densities, and neighbourhood-majority label smoothing."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import contract as C


def compute(xyz: np.ndarray, ks=C.FEAT_KS) -> tuple[np.ndarray, np.ndarray]:
    """Returns F (n, 14) in the order of contract.FEATURES and the k=ks[0] PCA normal."""
    xyz = np.asarray(xyz, np.float64); n = len(xyz); tree = cKDTree(xyz); cols = []; nrm = None
    for k in ks:
        kk = min(k + 1, n); d, idx = tree.query(xyz, k=kk)
        P = xyz[idx] - xyz[idx].mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", P, P) / max(kk - 1, 1)
        w, v = np.linalg.eigh(cov)                              # ascending
        l3, l2, l1 = np.maximum(w[:, 0], 1e-18), np.maximum(w[:, 1], 1e-18), np.maximum(w[:, 2], 1e-18)
        cols += [(l1 - l2) / l1, (l2 - l3) / l1, l3 / l1, 2 * np.sqrt(l3) * 1000, d[:, -1] * 1000, (l1 - l3) / l1]
        if nrm is None:
            nrm = v[:, :, 0]
    for r in (0.0015, 0.003):
        cols.append(np.array([len(x) - 1 for x in tree.query_ball_point(xyz, r)], float))
    return np.stack(cols, 1).astype(np.float32), nrm.astype(np.float32)


def smooth(xyz: np.ndarray, is_axis: np.ndarray, r_mm=C.LABEL_SMOOTH_MM) -> np.ndarray:
    """One pass of neighbourhood majority within r_mm (the point itself included)."""
    tree = cKDTree(xyz); out = is_axis.copy()
    for i, nb in enumerate(tree.query_ball_point(xyz, r_mm / 1000)):
        out[i] = is_axis[nb].mean() >= 0.5
    return out
