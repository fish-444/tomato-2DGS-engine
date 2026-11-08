"""Two-photo geometry for a cross-view "same leaflet?" question, from the board poses (OpenCV, 2026-09-29).

For two masks in two posed photos:

* ray gap: the rays through the two mask centroids should meet at the leaflet if both masks are the same leaflet;
  ``ray_gap_mm`` is their closest distance and ``ray_angle_deg`` the angle between them (small angles make the gap
  meaningless, so both are given).
* epipolar SIFT: SIFT keypoints inside each mask are matched (Lowe ratio 0.75); a match is kept when each point lies
  within EPI_TOL_PX of the other's epipolar line (lens distortion removed, essential matrix from the two poses).
  ``epi_inliers`` counts them and ``epi_fraction`` is their share of the ratio-test matches.

Photos without a board pose give NaN (the learner treats it as missing). These are features for the learned
reviewer; nothing here decides a pair.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tools.plant_structure import Cameras, closest

EPI_TOL_PX = 8.0
RATIO = 0.75
DILATE_PX = 9
NAN = float('nan')
FEATURES = ['ray_gap_mm', 'ray_angle_deg', 'sift_matches', 'epi_inliers', 'epi_fraction']


def essential(cams: Cameras, a: str, b: str) -> np.ndarray:
    ra, ta = cams.rt(a)
    rb, tb = cams.rt(b)
    Ra, Rb = cv2.Rodrigues(ra)[0], cv2.Rodrigues(rb)[0]
    R = Rb @ Ra.T
    t = tb - R @ ta
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return tx @ R


def epipolar_distance_px(cams: Cameras, a: str, b: str, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Symmetric epipolar distance (px) of matched pixel pairs pa (photo a) and pb (photo b)."""
    if not len(pa):
        return np.zeros(0)
    E = essential(cams, a, b)
    na = cv2.undistortPoints(pa.reshape(-1, 1, 2).astype(float), cams.K, cams.D).reshape(-1, 2)
    nb = cv2.undistortPoints(pb.reshape(-1, 1, 2).astype(float), cams.K, cams.D).reshape(-1, 2)
    ha, hb = np.c_[na, np.ones(len(na))], np.c_[nb, np.ones(len(nb))]
    lb = ha @ E.T          # lines in b
    la = hb @ E            # lines in a
    db = np.abs((hb * lb).sum(1)) / np.hypot(lb[:, 0], lb[:, 1])
    da = np.abs((ha * la).sum(1)) / np.hypot(la[:, 0], la[:, 1])
    return (da + db) / 2 * cams.K[0, 0]


def ray_gap(cams: Cameras, a: str, b: str, ca, cb) -> tuple[float, float]:
    Ca, da = cams.ray(a, tuple(ca))
    Cb, db = cams.ray(b, tuple(cb))
    ang = math.degrees(math.acos(min(1.0, abs(float(da @ db)))))
    if ang < 1.0:
        return NAN, ang
    p, q, _ = closest(Ca, da, Cb, db)
    return float(np.linalg.norm(p - q)), ang


class Photos:
    """Grey images and SIFT per (photo, mask set), cached."""

    def __init__(self, session: Path, capture: dict):
        self.session = Path(session)
        self.paths = {f['image_file'].lower(): self.session / f['image_path'] for f in capture['frames']}
        self._grey, self._sift = {}, {}
        self.sift = cv2.SIFT_create(nfeatures=4000)

    def grey(self, key):
        if key not in self._grey:
            self._grey = {key: cv2.cvtColor(np.asarray(Image.open(self.paths[key]).convert('RGB')), cv2.COLOR_RGB2GRAY)}
        return self._grey[key]

    def features(self, key, tag, mask):
        k = (key, tag)
        if k not in self._sift:
            m = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((DILATE_PX, DILATE_PX), np.uint8))
            kp, des = self.sift.detectAndCompute(self.grey(key), m)
            self._sift[k] = (np.array([p.pt for p in kp], float).reshape(-1, 2), des)
        return self._sift[k]


def pair_features(cams: Cameras, photos: Photos, a: str, mask_a: np.ndarray, tag_a: str,
                  b: str, mask_b: np.ndarray, tag_b: str) -> dict:
    out = {f: NAN for f in FEATURES}
    if not (cams.posed(a) and cams.posed(b)) or not mask_a.any() or not mask_b.any():
        return out
    ya, xa = np.nonzero(mask_a)
    yb, xb = np.nonzero(mask_b)
    out['ray_gap_mm'], out['ray_angle_deg'] = ray_gap(cams, a, b, (xa.mean(), ya.mean()), (xb.mean(), yb.mean()))
    pa, da = photos.features(a, tag_a, mask_a)
    pb, db = photos.features(b, tag_b, mask_b)
    if da is None or db is None or len(pa) < 2 or len(pb) < 2:
        out.update(sift_matches=0.0, epi_inliers=0.0, epi_fraction=NAN)
        return out
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2) if m.distance < RATIO * n.distance]
    if not good:
        out.update(sift_matches=0.0, epi_inliers=0.0, epi_fraction=NAN)
        return out
    ia, ib = np.array([m.queryIdx for m in good]), np.array([m.trainIdx for m in good])
    d = epipolar_distance_px(cams, a, b, pa[ia], pb[ib])
    inl = int((d <= EPI_TOL_PX).sum())
    out.update(sift_matches=float(len(good)), epi_inliers=float(inl), epi_fraction=inl / len(good))
    return out
