"""Per-frame camera pose from the ChArUco board, and turntable axis estimation.

Every pose is solved independently from one frame. No pose depends on any other frame,
on the commanded turntable angle, or on feature matching between frames - which is what
makes this rig work where structure-from-motion does not.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from tomato3d.io.schemas import (
    POSE_FAILED,
    POSE_SOLVED,
    BoardDetection,
    Intrinsics,
    Pose,
    TurntableAxis,
)

logger = logging.getLogger(__name__)

#: Minimum solved side-view poses before a turntable axis can be fitted. Three points
#: define a circle exactly; a fourth is what makes the residual meaningful.
MIN_POSES_FOR_AXIS = 4

#: Board +Z, used only to orient the fitted axis so that it points away from the board.
_BOARD_UP = np.array([0.0, 0.0, 1.0])

#: Angles this close to a full turn are the reference view itself: ``arctan2`` returns a
#: tiny negative value rather than exactly zero for it, and wrapping that into [0, 360)
#: would report a full revolution instead of none. Far below any measurable precision -
#: the axis fit residual is degrees, not nanodegrees.
_FULL_TURN_TOLERANCE_DEG = 1e-9


def _failed(n_corners_detected: int, reason: str) -> Pose:
    return Pose(
        pose_status=POSE_FAILED,
        n_corners_detected=n_corners_detected,
        failure_reason=reason,
    )


def solve_pose(
    detection: BoardDetection,
    intrinsics: Intrinsics,
    board: cv2.aruco.CharucoBoard,
    min_corners: int,
) -> Pose:
    """Solve one frame's camera pose from its ChArUco corners.

    Returns a :class:`Pose` with ``pose_status='failed'`` when fewer than ``min_corners``
    corners were detected or the solver fails. Never raises for a bad frame - an
    unsolvable frame is recorded and the session continues.

    The returned pose is in the **board frame** (origin at the board's origin inner
    corner, ``+Z`` up), not in OpenCV's sheet frame.
    """
    from tomato3d.calib.board import (
        board_frame_origin_sheet_mm,
        sheet_pose_to_board_pose_with_origin,
    )

    n_corners = detection.n_corners_detected
    if n_corners < min_corners:
        return _failed(n_corners, f"only {n_corners} corners, {min_corners} required")

    object_points_sheet_mm, image_points_px = board.matchImagePoints(
        detection.corners.reshape(-1, 1, 2), detection.ids.reshape(-1, 1)
    )
    if object_points_sheet_mm is None or len(object_points_sheet_mm) < 4:
        return _failed(n_corners, "could not match detected corners to object points")

    solved, rvec_sheet_rad, tvec_sheet_mm = cv2.solvePnP(
        object_points_sheet_mm,
        image_points_px,
        intrinsics.camera_matrix_array,
        intrinsics.dist_coeffs_array,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        return _failed(n_corners, "solvePnP did not converge")
    if float(tvec_sheet_mm.reshape(3)[2]) <= 0.0:
        # The board would be behind the camera: a mirrored planar solution.
        return _failed(n_corners, "solved pose places the board behind the camera")

    reprojected_px, _ = cv2.projectPoints(
        object_points_sheet_mm,
        rvec_sheet_rad,
        tvec_sheet_mm,
        intrinsics.camera_matrix_array,
        intrinsics.dist_coeffs_array,
    )
    residuals_px = np.linalg.norm(
        reprojected_px.reshape(-1, 2) - image_points_px.reshape(-1, 2), axis=1
    )
    reprojection_error_px = float(np.sqrt(np.mean(np.square(residuals_px))))

    rvec_rad, tvec_mm = sheet_pose_to_board_pose_with_origin(
        rvec_sheet_rad, tvec_sheet_mm, board_frame_origin_sheet_mm(board)
    )
    return Pose(
        pose_status=POSE_SOLVED,
        n_corners_detected=n_corners,
        reprojection_error_px=reprojection_error_px,
        rvec_rad=rvec_rad,
        tvec_mm=tvec_mm,
    )


def _fit_plane(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through ``(N, 3)`` points.

    Returns:
        ``(centroid_mm, unit_normal)``.
    """
    centroid_mm = points_mm.mean(axis=0)
    _, _, right_singular = np.linalg.svd(points_mm - centroid_mm)
    normal = right_singular[-1]
    return centroid_mm, normal / np.linalg.norm(normal)


def _fit_circle_2d(points_2d: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) circle fit to ``(N, 2)`` points.

    Returns:
        ``(center_2d, radius)``.
    """
    design = np.column_stack([2.0 * points_2d, np.ones(len(points_2d))])
    target = np.square(points_2d).sum(axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center_2d = solution[:2]
    radius = float(np.sqrt(solution[2] + center_2d @ center_2d))
    return center_2d, radius


def estimate_turntable_axis(poses: list[Pose]) -> TurntableAxis:
    """Fit the rotation axis from the side-view pose sequence.

    Because the board rotates and the camera does not, the camera traces a circle in the
    board frame. Fitting a plane and then a circle to the camera centres recovers the
    turntable axis.

    Used for QC and to fill ``measured_angle_deg`` only. **No downstream geometry may
    depend on it** - reconstruction uses each frame's own pose.
    """
    solved_poses = [pose for pose in poses if pose.is_solved]
    if len(solved_poses) < MIN_POSES_FOR_AXIS:
        return TurntableAxis(
            status="failed",
            n_poses_used=len(solved_poses),
            failure_reason=(
                f"{len(solved_poses)} solved poses, {MIN_POSES_FOR_AXIS} required"
            ),
        )

    camera_centers_mm = np.vstack(
        [pose.camera_center_board_mm() for pose in solved_poses]
    )
    centroid_mm, normal = _fit_plane(camera_centers_mm)
    if normal @ _BOARD_UP < 0.0:
        normal = -normal

    # An orthonormal in-plane basis. The first basis vector is taken from the first
    # camera centre so that angle zero is the first side view.
    first_offset_mm = camera_centers_mm[0] - centroid_mm
    in_plane_mm = first_offset_mm - (first_offset_mm @ normal) * normal
    in_plane_norm = float(np.linalg.norm(in_plane_mm))
    if in_plane_norm == 0.0:
        return TurntableAxis(
            status="failed",
            n_poses_used=len(solved_poses),
            failure_reason="first camera centre lies on the axis",
        )
    basis_u = in_plane_mm / in_plane_norm
    basis_v = np.cross(normal, basis_u)

    offsets_mm = camera_centers_mm - centroid_mm
    points_2d = np.column_stack([offsets_mm @ basis_u, offsets_mm @ basis_v])
    center_2d, radius_mm = _fit_circle_2d(points_2d)

    in_plane_residual_mm = np.linalg.norm(points_2d - center_2d, axis=1) - radius_mm
    out_of_plane_residual_mm = offsets_mm @ normal
    rms_fit_error_mm = float(
        np.sqrt(
            np.mean(np.square(in_plane_residual_mm) + np.square(out_of_plane_residual_mm))
        )
    )

    origin_mm = centroid_mm + center_2d[0] * basis_u + center_2d[1] * basis_v
    reference_offset_mm = camera_centers_mm[0] - origin_mm
    reference_direction = reference_offset_mm - (reference_offset_mm @ normal) * normal
    reference_norm = float(np.linalg.norm(reference_direction))
    if reference_norm == 0.0:
        return TurntableAxis(
            status="failed",
            n_poses_used=len(solved_poses),
            failure_reason="degenerate circle fit",
        )

    logger.info(
        "turntable axis from %d poses: radius %.1f mm, RMS %.2f mm",
        len(solved_poses),
        radius_mm,
        rms_fit_error_mm,
    )
    return TurntableAxis(
        status="solved",
        n_poses_used=len(solved_poses),
        origin_mm=origin_mm,
        direction=normal,
        reference_direction=reference_direction / reference_norm,
        radius_mm=radius_mm,
        rms_fit_error_mm=rms_fit_error_mm,
    )


def measured_angle_deg(pose: Pose, axis: TurntableAxis) -> float:
    """Angle of this frame about the turntable axis, in degrees within ``[0, 360)``.

    Zero is the first side view used to fit ``axis``. This is a *measured* quantity and
    the only angle any report may cite; ``nominal_angle_deg`` from the manifest is the
    commanded angle and is never used for geometry.

    Raises:
        ValueError: the pose failed or the axis was not fitted. Callers record
            ``unavailable`` rather than substituting a number.
    """
    if not pose.is_solved:
        raise ValueError("measured_angle_deg requested for a failed pose")
    if not axis.is_solved or axis.origin_mm is None:
        raise ValueError("measured_angle_deg requested with an unfitted turntable axis")

    direction = np.asarray(axis.direction, dtype=np.float64)
    reference_direction = np.asarray(axis.reference_direction, dtype=np.float64)
    offset_mm = pose.camera_center_board_mm() - np.asarray(
        axis.origin_mm, dtype=np.float64
    )
    in_plane_mm = offset_mm - (offset_mm @ direction) * direction
    if np.linalg.norm(in_plane_mm) == 0.0:
        raise ValueError("camera centre lies on the turntable axis")

    cosine = in_plane_mm @ reference_direction
    sine = np.cross(reference_direction, in_plane_mm) @ direction
    angle_rad = float(np.arctan2(sine, cosine))
    wrapped_deg = float(np.degrees(angle_rad) % 360.0)
    if wrapped_deg >= 360.0 - _FULL_TURN_TOLERANCE_DEG:
        return 0.0
    return wrapped_deg
