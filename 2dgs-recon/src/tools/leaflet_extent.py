"""Diagnostic 3D extent of one leaflet from its full-visibility 2D bboxes.

Each eligible bbox restricts the leaflet to the camera's viewing region through
that bbox. The feasible region is the intersection of those regions in the
board frame (mm). Its enclosing axis-aligned cuboid is reported separately from
the feasible polytope itself, because the cuboid includes infeasible corners.

No near/far distance, thickness, plant volume or nominal angle is ever added.
One view is a cone and stays unbounded; it is reported as such, never as a box.
Used by ``tools.rebuild_leaf_edit_session``; not a surface or PPFD occluder.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from tools.leaf_light.reconstruction import project

# Sampling and tolerance parameters of the approximation. They are recorded in
# every result so the disclosed approximation is reproducible.
EDGE_SAMPLES_PER_SIDE = 16
MIN_INSCRIBED_RADIUS_MM = 1e-3
CONTAINMENT_TOLERANCE_MM = 1e-6

METHOD = dict(
    frame='board; origin inner corner 0; +Z away from board; estimated mm',
    feasible_region='intersection of per-view viewing regions of full-visibility, identity-confirmed bboxes',
    distortion_handling=('bbox boundary sampled at %d points per side in original pixels, '
                         'undistorted with the capture camera model (cv2.undistortPoints), and '
                         'replaced by the convex hull of those normalized samples' % EDGE_SAMPLES_PER_SIDE),
    approximation_note=('The hull contains the sampled undistorted bbox outline, so the constraint is an '
                        'outer approximation except for sub-sample curvature of outward-bowing edges; '
                        'max_projected_outside_bbox_px reports the residual check.'),
    pixel_convention='bbox stored as continuous pixel edges [0,W]x[0,H]; converted to OpenCV pixel centres by -0.5 px',
    cuboid='axis-aligned in board axes; encloses the feasible polytope; includes infeasible space',
    forbidden_inputs='no near/far distance, thickness, plant volume, nominal camera angle or single-view depth',
)


class ExtentStatus:
    OK = 'bounded_feasible_region'
    NONE = 'insufficient_data_no_eligible_view'
    SINGLE = 'unbounded_single_view'
    UNBOUNDED = 'unbounded_extent'
    INCOMPATIBLE = 'incompatible_views_empty_intersection'
    DEGENERATE = 'degenerate_zero_volume_intersection'
    FAILED = 'numerical_failure'


def bbox_outline_px(box: dict) -> np.ndarray:
    """Sample the bbox outline in OpenCV pixel-centre coordinates."""
    x0, y0, x1, y1 = (float(box[k]) - 0.5 for k in ('x_min_px', 'y_min_px', 'x_max_px', 'y_max_px'))
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES_PER_SIDE, endpoint=False)
    sides = [np.column_stack((x0 + (x1 - x0) * t, np.full_like(t, y0))),
             np.column_stack((np.full_like(t, x1), y0 + (y1 - y0) * t)),
             np.column_stack((x1 - (x1 - x0) * t, np.full_like(t, y1))),
             np.column_stack((np.full_like(t, x0), y1 - (y1 - y0) * t))]
    return np.vstack(sides)


def view_halfspaces(box: dict, frame: dict, intrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A, b`` with ``A @ X <= b`` for board-frame points seen inside ``box``."""
    matrix = np.asarray(intrinsics['camera_matrix'], dtype=float)
    distortion = np.asarray(intrinsics['dist_coeffs'], dtype=float)
    normalized = cv2.undistortPoints(bbox_outline_px(box).reshape(-1, 1, 2), matrix, distortion).reshape(-1, 2)
    if not np.isfinite(normalized).all():
        raise ValueError('nonfinite_undistorted_bbox')
    hull = ConvexHull(normalized)
    ring = normalized[hull.vertices]
    rays = np.column_stack((ring, np.ones(len(ring))))
    inside_ray = np.append(ring.mean(axis=0), 1.0)
    rotation = cv2.Rodrigues(np.asarray(frame['rvec_rad'], dtype=float))[0]
    translation = np.asarray(frame['tvec_mm'], dtype=float)
    rows, limits = [], []
    normals = [np.cross(rays[i], rays[(i + 1) % len(rays)]) for i in range(len(rays))]
    normals.append(np.array([0.0, 0.0, 1.0]))  # In front of the camera.
    for normal in normals:
        normal = normal / np.linalg.norm(normal)
        if normal @ inside_ray < 0:
            normal = -normal
        # normal . (R X + t) >= 0  <=>  -(R^T normal) . X <= normal . t
        rows.append(-(rotation.T @ normal))
        limits.append(float(normal @ translation))
    return np.asarray(rows), np.asarray(limits)


def _solve(objective: np.ndarray, a: np.ndarray, b: np.ndarray):
    return linprog(objective, A_ub=a, b_ub=b, bounds=[(None, None)] * len(objective), method='highs')


def feasible_extent(views: list[tuple[dict, dict]], intrinsics: dict) -> dict:
    """Compute the viewing-region intersection for ``[(frame, bbox), ...]``.

    The caller passes only eligible views (measured pose, full visibility,
    confirmed identity). Every failure mode returns an explicit status and no
    cuboid; a cuboid is returned only for a bounded region of nonzero volume.
    """
    result = dict(status=ExtentStatus.NONE, eligible_view_count=len(views), cuboid_min_mm=None,
                  cuboid_max_mm=None, cuboid_size_mm=None, cuboid_volume_mm3=None,
                  feasible_volume_mm3=None, feasible_fraction_of_cuboid=None,
                  inscribed_radius_mm=None, feasible_vertices_mm=None, feasible_edges=None,
                  max_projected_outside_bbox_px=None, halfspaces=None, detail=None,
                  edge_samples_per_side=EDGE_SAMPLES_PER_SIDE, validated_metric=False)
    if not views:
        return result
    try:
        blocks = [view_halfspaces(box, frame, intrinsics) for frame, box in views]
    except (ValueError, QhullError, cv2.error, np.linalg.LinAlgError) as exc:
        return dict(result, status=ExtentStatus.FAILED, detail=f'invalid_view_geometry: {exc}')
    a = np.vstack([block[0] for block in blocks])
    b = np.concatenate([block[1] for block in blocks])
    result['halfspaces'] = dict(a=a.tolist(), b=b.tolist(), form='a @ X <= b, X in board mm')
    if len(views) == 1:
        # A single camera cone never bounds depth; do not solve for a fake box.
        return dict(result, status=ExtentStatus.SINGLE, detail='one view constrains direction only')
    feasibility = _solve(np.zeros(3), a, b)
    if feasibility.status == 2:
        return dict(result, status=ExtentStatus.INCOMPATIBLE, detail='viewing regions do not intersect')
    if feasibility.status != 0:
        return dict(result, status=ExtentStatus.FAILED, detail=f'feasibility_lp_status_{feasibility.status}: {feasibility.message}')
    low, high, unbounded = np.zeros(3), np.zeros(3), []
    for axis in range(3):
        for sign, store in ((1.0, low), (-1.0, high)):
            objective = np.zeros(3)
            objective[axis] = sign
            solved = _solve(objective, a, b)
            if solved.status == 3:
                unbounded.append(('+' if sign < 0 else '-') + 'xyz'[axis])
            elif solved.status != 0:
                return dict(result, status=ExtentStatus.FAILED, detail=f'extent_lp_status_{solved.status}: {solved.message}')
            else:
                store[axis] = solved.x[axis]
    if unbounded:
        return dict(result, status=ExtentStatus.UNBOUNDED, detail='unbounded directions: ' + ','.join(unbounded))
    # Largest inscribed ball distinguishes a real volume from touching regions.
    norms = np.linalg.norm(a, axis=1)
    centre = _solve(np.array([0.0, 0.0, 0.0, -1.0]), np.column_stack((a, norms)), b)
    if centre.status != 0:
        return dict(result, status=ExtentStatus.FAILED, detail=f'chebyshev_lp_status_{centre.status}')
    radius = float(centre.x[3])
    result.update(inscribed_radius_mm=radius)
    if radius < MIN_INSCRIBED_RADIUS_MM or np.any(high - low <= 2 * MIN_INSCRIBED_RADIUS_MM):
        return dict(result, status=ExtentStatus.DEGENERATE,
                    detail=f'inscribed radius {radius:.3g} mm; no volume to enclose')
    try:
        region = HalfspaceIntersection(np.column_stack((a / norms[:, None], -b / norms)), centre.x[:3])
        vertices = region.intersections
        hull = ConvexHull(vertices)
    except (QhullError, ValueError) as exc:
        return dict(result, status=ExtentStatus.FAILED, detail=f'polytope_failed: {exc}')
    edges = sorted({tuple(sorted((int(s[i]), int(s[(i + 1) % 3])))) for s in hull.simplices for i in range(3)
                    if not _coplanar_edge(hull, s, i)})
    outside_px = 0.0
    for frame, box in views:
        uv = project(vertices, frame, intrinsics) + 0.5  # back to pixel-edge coordinates
        dx = np.maximum(box['x_min_px'] - uv[:, 0], 0) + np.maximum(uv[:, 0] - box['x_max_px'], 0)
        dy = np.maximum(box['y_min_px'] - uv[:, 1], 0) + np.maximum(uv[:, 1] - box['y_max_px'], 0)
        outside_px = max(outside_px, float(np.max(np.hypot(dx, dy))))
    cuboid_volume = float(np.prod(high - low))
    result.update(status=ExtentStatus.OK, cuboid_min_mm=low.tolist(), cuboid_max_mm=high.tolist(),
                  cuboid_size_mm=(high - low).tolist(), cuboid_volume_mm3=cuboid_volume,
                  feasible_volume_mm3=float(hull.volume), feasible_fraction_of_cuboid=float(hull.volume) / cuboid_volume,
                  feasible_vertices_mm=vertices.tolist(), feasible_edges=[list(e) for e in edges],
                  max_projected_outside_bbox_px=outside_px)
    return result


def _coplanar_edge(hull: ConvexHull, simplex: np.ndarray, i: int) -> bool:
    """Drop triangulation diagonals inside a planar polytope face (display only)."""
    edge = {int(simplex[i]), int(simplex[(i + 1) % 3])}
    here = [k for k, s in enumerate(hull.simplices) if edge <= set(map(int, s))]
    if len(here) != 2:
        return False
    return bool(np.allclose(hull.equations[here[0]], hull.equations[here[1]], atol=1e-9))


def point_relation(xyz_mm: np.ndarray, extent: dict) -> dict:
    """Relate one estimated point to a cuboid without moving or clamping it."""
    if extent.get('status') != ExtentStatus.OK:
        return dict(inside_bbox_cuboid=None, inside_feasible_region=None, outside_cuboid_distance_mm=None,
                    feasible_region_violation_mm=None, x_cuboid_local_mm=None, y_cuboid_local_mm=None,
                    z_cuboid_local_mm=None)
    low, high = np.asarray(extent['cuboid_min_mm']), np.asarray(extent['cuboid_max_mm'])
    gap = np.maximum(low - xyz_mm, 0) + np.maximum(xyz_mm - high, 0)
    a, b = np.asarray(extent['halfspaces']['a']), np.asarray(extent['halfspaces']['b'])
    violation = float(np.max((a @ xyz_mm - b) / np.linalg.norm(a, axis=1)))
    local = xyz_mm - low
    return dict(inside_bbox_cuboid=bool(np.all(gap <= CONTAINMENT_TOLERANCE_MM)),
                inside_feasible_region=violation <= CONTAINMENT_TOLERANCE_MM,
                outside_cuboid_distance_mm=float(np.linalg.norm(gap)),
                feasible_region_violation_mm=max(violation, 0.0),
                x_cuboid_local_mm=float(local[0]), y_cuboid_local_mm=float(local[1]), z_cuboid_local_mm=float(local[2]))
