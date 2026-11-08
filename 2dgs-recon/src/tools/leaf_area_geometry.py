"""Versioned projected-area geometry; no leaf-normal or hidden-tissue inference."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from tools.leaf_light.common import StudyError, digest

METHOD = 'undistorted_pixel_footprint_v2'
LEGACY = 'hull_depth_fx_squared_v1'


@dataclass(frozen=True)
class IntegrationPolicy:
    chunk_pixels: int
    inverse_iterations: int
    inverse_tolerance_px: float
    relative_area_tolerance: float
    absolute_area_tolerance_normalized: float
    max_edge_subdivisions: int

    @classmethod
    def parse(cls, value: dict) -> IntegrationPolicy:
        try:
            p = cls(**value)
        except TypeError as exc:
            raise StudyError('Complete numerical integration policy required') from exc
        for field in ('chunk_pixels', 'inverse_iterations', 'max_edge_subdivisions'):
            if type(getattr(p, field)) is not int or getattr(p, field) < 1:
                raise StudyError(f'Invalid {field}')
        if p.max_edge_subdivisions < 2 or p.max_edge_subdivisions & (p.max_edge_subdivisions - 1):
            raise StudyError('max_edge_subdivisions must be a power of two >= 2')
        for field in ('inverse_tolerance_px', 'relative_area_tolerance', 'absolute_area_tolerance_normalized'):
            value = getattr(p, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise StudyError(f'Positive finite {field} required')
        return p


def validate_camera(matrix: np.ndarray, distortion: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    k, d = np.asarray(matrix, float), np.asarray(distortion, float).reshape(-1)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise StudyError('Invalid calibrated camera matrix')
    if not np.array_equal(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise StudyError('Only standard zero-skew OpenCV cameras are supported')
    if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
        raise StudyError('Explicit supported OpenCV distortion coefficients required')
    if len(size) != 2 or any(type(v) not in (int, np.int64, np.int32) or v <= 0 for v in size):
        raise StudyError('Positive calibrated image dimensions required')
    return k, d


def calibration_identity(k: np.ndarray, d: np.ndarray, size: tuple[int, int]) -> str:
    k, d = validate_camera(k, d, size)
    return digest(dict(camera_matrix=k.tolist(), dist_coeffs=d.tolist(), image_size_px=list(size)))


def normalized_pixel_areas(xy_px: np.ndarray, k: np.ndarray, d: np.ndarray,
                           size: tuple[int, int], policy: IntegrationPolicy,
                           pixel_size_xy: tuple[float, float] = (1., 1.)) -> tuple[np.ndarray, dict]:
    """Area at Z=1 of original-coordinate pixels; NaN means invalid/unconverged.

    xy_px are original-image centres. Explicit pixel_size_xy describes a known
    resampling transform, not an inferred scale. No crop-origin guessing/cache.
    """
    k, d = validate_camera(k, d, size)
    xy = np.asarray(xy_px, float)
    scale = np.asarray(pixel_size_xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise StudyError('Finite Nx2 original pixel centres required')
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise StudyError('Explicit positive pixel footprint scale required')
    if np.any(xy - scale / 2 < -.5) or np.any(xy + scale / 2 > np.asarray(size) - .5):
        raise StudyError('Pixel footprint exceeds calibrated image')
    result = np.full(len(xy), np.nan)
    counts = dict(inverse_failed_pixels=0, unconverged_pixels=0, max_subdivisions_used=0)
    if not np.any(d):
        result[:] = np.prod(scale) / (k[0, 0] * k[1, 1])
        return result, counts

    def boundary(centres: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(n) / n
        offsets = np.vstack((np.c_[t - .5, np.full(n, -.5)],
                             np.c_[np.full(n, .5), t - .5],
                             np.c_[.5 - t, np.full(n, .5)],
                             np.c_[np.full(n, -.5), .5 - t])) * scale
        raw = (centres[:, None, :] + offsets).reshape(-1, 1, 2)
        rays = cv2.undistortPointsIter(raw, k, d, None, None,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                     policy.inverse_iterations, policy.inverse_tolerance_px)).reshape(-1, 2)
        reproj = cv2.projectPoints(np.c_[rays, np.ones(len(rays))], np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
        errors = np.linalg.norm(reproj - raw.reshape(-1, 2), axis=1).reshape(len(centres), -1)
        uv = rays.reshape(len(centres), -1, 2)
        # Translate near zero to avoid catastrophic cancellation off-axis.
        uv = uv - uv[:, :1]
        cross = uv[..., 0] * np.roll(uv[..., 1], -1, axis=1) - uv[..., 1] * np.roll(uv[..., 0], -1, axis=1)
        areas = .5 * cross.sum(axis=1)
        valid = np.isfinite(areas) & (areas > 0) & np.isfinite(errors).all(axis=1) & (errors.max(axis=1) <= policy.inverse_tolerance_px)
        return areas, valid

    for start in range(0, len(xy), policy.chunk_pixels):
        centres = xy[start:start + policy.chunk_pixels]
        previous, valid = boundary(centres, 1)
        active = np.arange(len(centres))
        failed = ~valid
        n = 2
        while len(active) and n <= policy.max_edge_subdivisions:
            area, okay = boundary(centres[active], n)
            okay &= ~failed[active]
            done = okay & (np.abs(area - previous[active]) <= policy.absolute_area_tolerance_normalized + policy.relative_area_tolerance * np.abs(area))
            result[start + active[done]] = area[done]
            failed[active] |= ~okay
            previous[active] = area
            active = active[~done & okay]
            counts['max_subdivisions_used'] = max(counts['max_subdivisions_used'], n)
            n *= 2
        counts['inverse_failed_pixels'] += int(failed.sum())
        counts['unconverged_pixels'] += len(active)
    return result, counts


def photo_area_v2(cams: Any, key: str, hull_mm: np.ndarray, masks: list,
                  policy: IntegrationPolicy, depth_percentile: float, minimum_hull_points: int) -> dict:
    """Preserve legacy union/depth selection; integrate a new projected proxy."""
    k, d = validate_camera(cams.K, cams.D, cams.size)
    if not math.isfinite(depth_percentile) or not 0 <= depth_percentile <= 100 or type(minimum_hull_points) is not int or minimum_hull_points < 1:
        raise StudyError('Explicit valid depth policy required')
    width, height = cams.size
    hull = np.asarray(hull_mm, float)
    if hull.ndim != 2 or hull.shape[1] != 3 or not np.isfinite(hull).all():
        raise StudyError('Finite Nx3 hull required')
    if len(hull):
        projected = cams.project(key, hull)
        depth = cams.depth(key, hull)
        finite = np.isfinite(projected).all(axis=1) & np.isfinite(depth) & (depth > 0)
        # Invalid projections never enter an integer conversion or mask lookup.
        px = np.zeros(projected.shape, dtype=np.int64)
        finite &= (projected[:, 0] >= -.5) & (projected[:, 0] < width-.5) & (projected[:, 1] >= -.5) & (projected[:, 1] < height-.5)
        px[finite] = np.round(projected[finite]).astype(np.int64)
    else:
        px, depth, finite = np.empty((0, 2), int), np.empty(0), np.empty(0, bool)
    z_all = np.full((height, width), np.inf, np.float32)
    z_acc = np.full_like(z_all, np.inf)
    union = np.zeros((height, width), bool)
    accepted_union = np.zeros_like(union)
    used = no_depth = empty = 0
    for crop, mask, accepted in masks:
        if len(crop) != 4 or any(type(v) not in (int, np.int64, np.int32) for v in crop):
            raise StudyError('Integer original-image crop coordinates required')
        x0, y0, x1, y1 = crop
        m = np.asarray(mask)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or m.shape != (y1-y0, x1-x0) or m.dtype != bool:
            raise StudyError('Mask/crop coordinate mismatch; explicitly restore original resolution')
        union[y0:y1, x0:x1] |= m
        if accepted:
            accepted_union[y0:y1, x0:x1] |= m
        if not m.any():
            empty += 1
            continue
        selection = finite & (px[:, 0] >= x0) & (px[:, 0] < x1) & (px[:, 1] >= y0) & (px[:, 1] < y1)
        ids = np.flatnonzero(selection)
        ids = ids[m[px[ids, 1] - y0, px[ids, 0] - x0]]
        if len(ids) < minimum_hull_points:
            no_depth += 1
            continue
        z = np.float32(np.percentile(depth[ids], depth_percentile))
        if not np.isfinite(z) or z <= 0:
            no_depth += 1
            continue
        for array, take in ((z_all, True), (z_acc, accepted)):
            if take:
                view = array[y0:y1, x0:x1]
                view[m] = np.minimum(view[m], z)
        used += 1

    covered = np.isfinite(z_all)
    yy, xx = np.nonzero(covered)
    factors, diagnostics = normalized_pixel_areas(np.c_[xx, yy], k, d, cams.size, policy)
    valid = np.isfinite(factors)
    total = float(np.sum(factors[valid] * z_all[yy[valid], xx[valid]].astype(float)**2) / 100)
    av = valid & np.isfinite(z_acc[yy, xx])
    atotal = float(np.sum(factors[av] * z_acc[yy[av], xx[av]].astype(float)**2) / 100)
    original, accepted_original = int(union.sum()), int(accepted_union.sum())
    measured, accepted_measured = int(valid.sum()), int(av.sum())
    status = 'no_masks' if not masks else 'empty' if original == 0 else 'complete_projected_coverage' if measured == original else 'partial' if measured else 'unavailable'
    return dict(area_method=METHOD, calibration_hash=calibration_identity(k, d, cams.size),
                depth_method=dict(name='per_mask_hull_percentile', percentile=depth_percentile, minimum_points=minimum_hull_points),
                status=status, masks=len(masks), masks_used=used, empty_masks=empty, masks_without_depth=no_depth,
                original_union_px=original, used_union_px=measured, missing_union_px=original-measured,
                projected_area_cm2=total if masks and measured == original else None,
                supported_projected_area_cm2=total if measured or (masks and original == 0) else None,
                accepted_projected_area_cm2=atotal if masks and accepted_original == accepted_measured else None,
                accepted_used_px=accepted_measured, accepted_original_px=accepted_original,
                inverse_diagnostics=diagnostics, validated_metric=False, suitable_for_ppfd=False)


def view_geometry(cams: Any, key: str, grouping: dict | None) -> dict:
    """Camera centre is measured; elevation requires an explicit reference point."""
    if not cams.posed(key):
        return dict(group='unavailable', grouping_source='pose_unavailable', elevation_deg=None, camera_center_board_mm=None)
    centre = np.asarray(cams.centre(key), float)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise StudyError('Non-finite camera centre')
    result = dict(group='unavailable', grouping_source='reference_unavailable', elevation_deg=None, camera_center_board_mm=centre.tolist())
    if grouping is None:
        return result
    reference = np.asarray(grouping.get('centre_board_mm'), float)
    threshold = grouping.get('top_elevation_gt_deg')
    if reference.shape != (3,) or not np.isfinite(reference).all() or not grouping.get('source'):
        raise StudyError('Explicit board-frame grouping reference and source required')
    v = centre - reference
    if np.linalg.norm(v) == 0:
        raise StudyError('Camera centre coincides with elevation reference')
    result.update(elevation_deg=math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))), grouping_source=grouping['source'])
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -90 <= threshold <= 90:
            raise StudyError('Invalid explicit elevation threshold')
        result['group'] = 'top' if result['elevation_deg'] > threshold else 'ring'
    return result
