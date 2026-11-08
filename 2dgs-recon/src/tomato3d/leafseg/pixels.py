"""Pixel-level parts of Option A (contract §3.1–3.3): pre-processing, foreground random forest, rachis skeleton, RLE."""

from __future__ import annotations

import cv2
import numpy as np

SPEC_V, SPEC_S, INPAINT_R = 0.90, 0.15, 4
CLAHE_CLIP, CLAHE_TILE = 2.5, 8
LCN_SIGMA_AT_2000 = 20.0
RF_DOWNSAMPLE, RF_ERODE_PX, RF_NEG_MARGIN_PX = 4, 3, 15
PURPLE_Z, PURPLE_SIGMA_AT_2000, RIDGE_PCT = 1.0, 15.0, 90


def scaled(v: float, shape) -> float:
    return v * max(shape[:2]) / 2000.0


def preprocess(rgb: np.ndarray) -> dict:
    """Specular inpaint, CLAHE on L*, LCN. Returns the processed RGB, Lab, HSV, LCN, specular mask."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL).astype(np.float32) / 255.0
    spec = (hsv[..., 2] > SPEC_V) & (hsv[..., 1] < SPEC_S)
    bgr = cv2.inpaint(np.ascontiguousarray(rgb[..., ::-1]), spec.astype(np.uint8) * 255, INPAINT_R, cv2.INPAINT_TELEA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE)).apply(lab[..., 0])
    rgb2 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    L = lab[..., 0].astype(np.float32) / 255.0; s = scaled(LCN_SIGMA_AT_2000, rgb.shape)
    mu = cv2.GaussianBlur(L, (0, 0), s); sd = np.sqrt(np.maximum(cv2.GaussianBlur(L * L, (0, 0), s) - mu * mu, 0))
    return dict(rgb=rgb2, lab=lab.astype(np.float32), hsv=cv2.cvtColor(rgb2, cv2.COLOR_RGB2HSV_FULL).astype(np.float32) / 255.0,
                lcn=(L - mu) / (sd + 1e-3), specular=spec)


def pixel_features(pp: dict) -> np.ndarray:
    """(H/ds, W/ds, F): Lab, HSV, LCN, local std of L* (7 px) on the RF grid; specular pixels marked NaN."""
    ds = RF_DOWNSAMPLE; L = pp["lab"][..., 0] / 255.0
    m = cv2.blur(L, (7, 7)); lsd = np.sqrt(np.maximum(cv2.blur(L * L, (7, 7)) - m * m, 0))
    F = np.dstack([pp["lab"] / 255.0, pp["hsv"], pp["lcn"][..., None], lsd[..., None]])[::ds, ::ds].astype(np.float32)
    F[pp["specular"][::ds, ::ds]] = np.nan
    return F


def training_pixels(F: np.ndarray, union_full: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    ds = RF_DOWNSAMPLE; u = union_full[::ds, ::ds].astype(np.uint8)
    pos = cv2.erode(u, np.ones((2 * (RF_ERODE_PX // ds) + 1,) * 2, np.uint8)) > 0
    far = cv2.dilate(u, np.ones((2 * (RF_NEG_MARGIN_PX // ds) + 1,) * 2, np.uint8)) == 0
    ok = np.isfinite(F).all(-1); P, N = F[pos & ok], F[far & ok]
    n = min(len(P), len(N), 60000)
    P = P[rng.choice(len(P), n, replace=False)]; N = N[rng.choice(len(N), n, replace=False)]
    return np.vstack([P, N]), np.r_[np.ones(len(P)), np.zeros(len(N))]


def train_foreground(Xs, ys, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=seed)
    return rf.fit(np.vstack(Xs), np.concatenate(ys))


def foreground_prob(rf, F: np.ndarray, full_shape) -> np.ndarray:
    h, w = F.shape[:2]; X = np.nan_to_num(F.reshape(-1, F.shape[-1]), nan=0.0)
    p = rf.predict_proba(X)[:, 1].reshape(h, w).astype(np.float32)
    return cv2.resize(p, (full_shape[1], full_shape[0]), interpolation=cv2.INTER_LINEAR)


def rachis_skeleton(pp: dict, fg: np.ndarray) -> dict:
    """Purple tint (local a* z-score) AND sato ridge (top 10 % inside the foreground) -> skeleton + skan graph."""
    from skimage.filters import sato
    from skimage.morphology import skeletonize
    a = pp["lab"][..., 1]; s = scaled(PURPLE_SIGMA_AT_2000, a.shape)
    mu = cv2.GaussianBlur(a, (0, 0), s); sd = np.sqrt(np.maximum(cv2.GaussianBlur(a * a, (0, 0), s) - mu * mu, 1e-6))
    purple = (a - mu) / sd > PURPLE_Z
    small = cv2.resize(pp["lab"][..., 0] / 255.0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ridge = cv2.resize(sato(small, sigmas=(1, 2, 3), black_ridges=False).astype(np.float32), (a.shape[1], a.shape[0]))
    thr = np.percentile(ridge[fg], RIDGE_PCT) if fg.any() else np.inf
    m = purple & (ridge > thr) & fg
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    sk = skeletonize(m)
    ends, branches = np.zeros((0, 2), int), np.zeros((0, 2), int)
    if sk.sum() > 10:
        from skan import Skeleton, summarize
        sm = summarize(Skeleton(sk), separator="_")
        deg = {}
        for _, r in sm.iterrows():
            for k in ("src", "dst"):
                key = (int(r[f"image_coord_{k}_0"]), int(r[f"image_coord_{k}_1"])); deg[key] = deg.get(key, 0) + 1
        ends = np.array([k for k, v in deg.items() if v == 1] or np.zeros((0, 2)), int)
        branches = np.array([k for k, v in deg.items() if v >= 3] or np.zeros((0, 2)), int)
    return dict(skeleton=sk, mask=m, ends=ends, branches=branches)


def rle_encode(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE (column-major counts, starting with a zero run)."""
    f = mask.astype(np.uint8).ravel(order="F")
    d = np.flatnonzero(np.diff(f)) + 1                       # positions where the value changes
    counts = np.diff(np.r_[0, d, len(f)]).tolist()
    if len(f) and f[0] == 1:                                 # COCO RLE always starts with a run of zeros
        counts = [0] + counts
    return dict(size=[int(mask.shape[0]), int(mask.shape[1])], counts=[int(c) for c in counts])


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]; flat = np.zeros(h * w, np.uint8); pos, val = 0, 0
    for c in rle["counts"]:
        flat[pos:pos + c] = val; pos += c; val ^= 1
    return flat.reshape((h, w), order="F").astype(bool)
