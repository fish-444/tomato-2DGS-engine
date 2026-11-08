"""Camera intrinsic calibration from ChArUco shots.

The calibration runs on matched object/image point pairs obtained from
``CharucoBoard.matchImagePoints``, so partially occluded shots still contribute. Nothing
is written here - the caller writes ``intrinsics_<camera_id>.json``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from tomato3d.io.images import read_image
from tomato3d.io.schemas import (
    SCHEMA_VERSION,
    BoardSpec,
    Intrinsics,
    SchemaError,
    now_isoformat,
)

logger = logging.getLogger(__name__)

#: Fewer views than this and the intrinsics are not trustworthy, so calibration refuses.
MIN_CALIBRATION_VIEWS = 15

#: A view contributing fewer corners than this adds noise rather than information.
MIN_CORNERS_PER_CALIBRATION_VIEW = 6


class CalibrationError(SchemaError):
    """Raised when intrinsics cannot be calibrated from the supplied shots."""


def _usable_views(
    image_paths: list[Path], board: cv2.aruco.CharucoBoard
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Collect object/image point pairs, and the common image size.

    Returns:
        ``(object_points, image_points, (width_px, height_px))``.
    """
    from tomato3d.calib.board import detect_board_corners

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size_px: tuple[int, int] | None = None

    for image_path in image_paths:
        image = read_image(image_path)
        if image is None:
            logger.warning("unreadable calibration image, skipped: %s", image_path)
            continue
        height_px, width_px = image.shape[:2]
        if image_size_px is None:
            image_size_px = (width_px, height_px)
        elif image_size_px != (width_px, height_px):
            raise CalibrationError(
                f"{image_path.name} is {width_px}x{height_px} but earlier shots are "
                f"{image_size_px[0]}x{image_size_px[1]}; one camera, one image size"
            )

        detection = detect_board_corners(image, board)
        if detection.n_corners_detected < MIN_CORNERS_PER_CALIBRATION_VIEW:
            logger.info(
                "%s: %d corners, below the %d needed for calibration, skipped",
                image_path.name,
                detection.n_corners_detected,
                MIN_CORNERS_PER_CALIBRATION_VIEW,
            )
            continue

        view_object_points, view_image_points = board.matchImagePoints(
            detection.corners.reshape(-1, 1, 2), detection.ids.reshape(-1, 1)
        )
        if view_object_points is None or len(view_object_points) < 4:
            logger.info("%s: could not match object points, skipped", image_path.name)
            continue
        object_points.append(np.asarray(view_object_points, dtype=np.float32))
        image_points.append(np.asarray(view_image_points, dtype=np.float32))

    if image_size_px is None:
        raise CalibrationError(f"no readable images among {len(image_paths)} paths")
    return object_points, image_points, image_size_px


def calibrate_intrinsics(
    image_paths: list[Path], spec: BoardSpec, camera_id: str
) -> Intrinsics:
    """Calibrate one camera from ChArUco shots.

    Requires at least :data:`MIN_CALIBRATION_VIEWS` usable views. Raises
    :class:`CalibrationError` with the count if fewer. Writes nothing; the caller writes
    the JSON.
    """
    from tomato3d.calib.board import build_charuco_board

    board = build_charuco_board(spec)
    object_points, image_points, image_size_px = _usable_views(image_paths, board)

    if len(object_points) < MIN_CALIBRATION_VIEWS:
        raise CalibrationError(
            f"{len(object_points)} usable calibration views out of {len(image_paths)} "
            f"images; at least {MIN_CALIBRATION_VIEWS} are required"
        )

    rms_reprojection_error_px, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size_px,
        cameraMatrix=None,
        distCoeffs=None,
    )
    logger.info(
        "calibrated %s from %d views, RMS %.4f px",
        camera_id,
        len(object_points),
        rms_reprojection_error_px,
    )

    return Intrinsics(
        schema_version=SCHEMA_VERSION,
        camera_id=camera_id,
        image_width_px=image_size_px[0],
        image_height_px=image_size_px[1],
        camera_matrix=tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(camera_matrix, dtype=np.float64)
        ),
        dist_coeffs=tuple(
            float(value) for value in np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        ),
        rms_reprojection_error_px=float(rms_reprojection_error_px),
        n_calibration_images=len(object_points),
        calibrated_at=now_isoformat(),
        distortion_model="opencv_pinhole",
    )
