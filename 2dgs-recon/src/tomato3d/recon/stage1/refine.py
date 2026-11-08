"""E6 - observation-based refinement of the reconstructed elements (never of the observations, never self-fitting).

The target is ALWAYS the observation: the visible top / side masks. A reconstructed (hidden) leaflet must project
entirely onto pixels that the observation shows covered by some leaf - wherever it projects onto observed BACKGROUND
it contradicts the photo. The refinement moves only ``reconstructed_prior`` elements (size factor in
[``SCALE_MIN``, 1] - the prior range already bounds the size - and a rotation about the vertical at the slot within
+-``MAX_ROT_DEG``) to minimise that violation in the top view and in the side views; ``observed`` elements are fixed
(infinite weight). Before / after values and the residual (violating px) are stored per element.

Two-image product: the residual is a reprojection consistency count, not an IoU against a hidden truth. With a
multi-view teacher the same slot would be scored by observed-vs-structure point distances (Chamfer; Hausdorff p95),
leaf identity by B8 seed growth - not implemented here (needs the teacher data).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from tomato3d.recon.stage1.views import SideObs, TopObs, side_transform

SCALE_STEPS = (1.0, 0.9, 0.8, 0.7)
#: a flat (O7) reconstructed leaflet is edge-on in a side view; its height is approximate, so the side check accepts
#: observed occupancy within this vertical band
SIDE_HEIGHT_TOL_MM = 3.0
#: an element contradicts the photo when more than this share of its footprint falls on observed background
VIOLATION_FRAC = 0.05
SCALE_MIN = 0.7
ROT_STEPS_DEG = (0.0, -15.0, 15.0, -30.0, 30.0)
MAX_ROT_DEG = 30.0


def footprint(verts: np.ndarray, faces: np.ndarray, project, shape: tuple[int, int]) -> np.ndarray:
    uv = project(verts)
    m = np.zeros(shape, np.uint8)
    polys = np.round(uv[faces] * 16).astype(np.int32)
    cv2.fillPoly(m, list(polys), 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def _side_occupancy(sv: SideObs) -> np.ndarray:
    """Observed side occupancy dilated vertically by the height tolerance (cached on the observation object)."""
    occ = getattr(sv, "_occupancy_tol", None)
    if occ is None:
        k = max(1, int(round(SIDE_HEIGHT_TOL_MM / sv.camera.mm_per_px)))
        occ = cv2.dilate((sv.id_map > 0).astype(np.uint8), np.ones((2 * k + 1, 1), np.uint8)).astype(bool)
        sv._occupancy_tol = occ
    return occ


def violation(verts: np.ndarray, faces: np.ndarray, top: TopObs, sides: Sequence[SideObs]) -> tuple[int, int]:
    """(violating px, footprint px) summed over the top and side views."""
    occ = top.id_map > 0
    fp = footprint(verts, faces, top.camera.project, occ.shape)
    n, tot = int((fp & ~occ).sum()), int(fp.sum())
    for sv in sides:
        occ_s = _side_occupancy(sv)
        proj = lambda v, az=sv.azimuth_deg, cam=sv.camera: cam.project(side_transform(v, az)[:, :2])  # noqa: E731
        fps = footprint(verts, faces, proj, occ_s.shape)
        n, tot = n + int((fps & ~occ_s).sum()), tot + int(fps.sum())
    return n, tot


def _transform(verts: np.ndarray, pivot: np.ndarray, scale: float, rot_deg: float) -> np.ndarray:
    a = np.radians(rot_deg)
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return (verts - pivot) @ r.T * scale + pivot


def run(top: TopObs, sides: Sequence[SideObs], assembled: Mapping, completion: Mapping) -> list[dict]:
    rows = []
    by_el = {(e["element"], e["sample"]): e for e in completion["elements"]}
    for el_id, (v, f) in assembled["meshes"].items():
        el = by_el.get((el_id, assembled["sample"]))
        pivot = np.r_[np.asarray(el["xy"], float), v[:, 2].mean()]
        before, fp_before = violation(v, f, top, sides)
        best = (before, 1.0, 0.0)
        fp_best = fp_before
        if before > VIOLATION_FRAC * fp_before:
            for sc in SCALE_STEPS:
                for rot in ROT_STEPS_DEG:
                    if sc == 1.0 and rot == 0.0:
                        continue
                    viol, fp = violation(_transform(v, pivot, sc, rot), f, top, sides)
                    if viol < best[0] or (viol == best[0] and (1 - sc) + abs(rot) / 100 < (1 - best[1]) + abs(best[2]) / 100):
                        best, fp_best = (viol, sc, rot), fp
        rows.append(dict(element=el_id, compound=el["compound"], sample=assembled["sample"],
                         source_kind="reconstructed_prior", violation_px_before=before, violation_px_after=best[0],
                         scale=best[1], rotation_deg=best[2], length_before_mm=el["length_mm"],
                         length_after_mm=el["length_mm"] * best[1], area_before_mm2=el["area_mm2"],
                         area_after_mm2=el["area_mm2"] * best[1] ** 2,
                         violation_frac_before=before / max(fp_before, 1), violation_frac_after=best[0] / max(fp_best, 1),
                         contradicts_photo_before=before > VIOLATION_FRAC * fp_before,
                         contradicts_photo=best[0] > VIOLATION_FRAC * fp_best))
    return rows
