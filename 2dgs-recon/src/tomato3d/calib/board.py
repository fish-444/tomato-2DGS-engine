"""ChArUco board construction, corner detection, and the sheet <-> board frame map.

Why ChArUco and not ``cv2.findChessboardCorners``
-------------------------------------------------
The plant stands in the middle of the board and occludes most of it. A plain chessboard
detector needs the *whole* grid and fails. ChArUco identifies each corner by the ArUco
markers next to it, so any visible subset is enough to solve a pose.

Frames
------
``sheet frame``
    OpenCV's own ChArUco object-point frame: origin at the printed sheet corner,
    ``+X`` right, ``+Y`` down, ``+Z`` into the board. ``board_spec.json`` gives
    ``scale_block_position_mm`` and ``pot_center_offset_mm`` in this frame.

``board frame``
    The pipeline frame, used by ``poses.csv`` and everything downstream: origin at the
    board's origin inner corner (ChArUco corner id 0), ``+X`` along the board rows,
    ``+Y`` along the board columns, ``+Z`` up and away from the board surface.

The two differ by a 180 degree rotation about ``+X`` plus a translation of one square in
each of ``X`` and ``Y``. :data:`SHEET_TO_BOARD_ROTATION` and the functions below are the
only implementation of that map.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from tomato3d.io.schemas import BoardDetection, BoardSpec, SchemaError

logger = logging.getLogger(__name__)

#: 180 degree rotation about +X. Flips ``+Y`` and ``+Z`` so that ``+Z`` points up, away
#: from the board surface, while keeping the frame right handed.
SHEET_TO_BOARD_ROTATION: np.ndarray = np.diag([1.0, -1.0, -1.0])


class BoardError(SchemaError):
    """Raised when a board cannot be built from its spec."""


def aruco_dictionary(spec: BoardSpec) -> cv2.aruco.Dictionary:
    """Resolve ``spec.aruco_dict`` (e.g. ``"DICT_4X4_50"``) to an OpenCV dictionary."""
    dictionary_id = getattr(cv2.aruco, spec.aruco_dict, None)
    if dictionary_id is None or not isinstance(dictionary_id, int):
        raise BoardError(f"unknown aruco_dict {spec.aruco_dict!r}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def build_charuco_board(spec: BoardSpec) -> cv2.aruco.CharucoBoard:
    """Build the ChArUco board described by ``board_spec.json``.

    Object points are in millimetres in the *sheet* frame, which is what OpenCV solves
    poses in. Use :func:`sheet_pose_to_board_pose` to move a solved pose into the board
    frame before writing it anywhere.

    Both the marker ids and the legacy-pattern flag are taken from the spec and set
    explicitly, never left to OpenCV defaults. Either one disagreeing with the print does
    not stop detection: markers are still found, but matched to the wrong object points,
    so the pose is wrong while every count looks healthy.
    """
    if spec.pattern != "charuco":
        raise BoardError(f"board pattern {spec.pattern!r} is not supported")
    dictionary = aruco_dictionary(spec)
    n_dictionary_markers = len(dictionary.bytesList)
    out_of_range_ids = [
        marker_id for marker_id in spec.marker_ids if marker_id >= n_dictionary_markers
    ]
    if out_of_range_ids:
        raise BoardError(
            f"{spec.aruco_dict} holds {n_dictionary_markers} markers but marker_ids "
            f"includes {out_of_range_ids}"
        )
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_size_mm,
        spec.marker_size_mm,
        dictionary,
        np.asarray(spec.marker_ids, dtype=np.int32),
    )
    board.setLegacyPattern(spec.legacy_pattern)
    return board


def _build_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    """A detector tuned for heavy occlusion and subpixel accuracy."""
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(
        board, charuco_params, detector_params, cv2.aruco.RefineParameters()
    )


def detect_board_corners(
    image: np.ndarray, board: cv2.aruco.CharucoBoard
) -> BoardDetection:
    """Subpixel ChArUco corners.

    ``BoardDetection`` carries corners ``(N, 2)`` float32, ids ``(N,)`` int32, and
    ``n_corners_detected``. Returns an empty detection rather than raising when the board
    is not visible at all - an unusable frame is a recorded fact, not an error.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise BoardError(f"expected a 2D or 3D image array, got shape {image.shape}")

    detector = _build_detector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return BoardDetection.empty()

    corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    return BoardDetection(corners=corners, ids=ids, n_corners_detected=int(ids.size))


# --------------------------------------------------------------------------------------
# Sheet frame <-> board frame
# --------------------------------------------------------------------------------------


def _sheet_origin_mm(spec: BoardSpec) -> np.ndarray:
    """Sheet-frame position of the board-frame origin (ChArUco corner id 0)."""
    corner_x_mm, corner_y_mm = spec.first_inner_corner_sheet_mm
    return np.array([corner_x_mm, corner_y_mm, 0.0], dtype=np.float64)


def board_frame_origin_sheet_mm(board: cv2.aruco.CharucoBoard) -> np.ndarray:
    """Sheet-frame position of the board-frame origin, read from the board itself.

    Lets code that holds a board but not its spec - :func:`solve_pose` - use the same
    frame map as everything else.
    """
    return np.asarray(board.getChessboardCorners()[0], dtype=np.float64).reshape(3)


def sheet_to_board_mm(points_sheet_mm: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """Map ``(N, 3)`` sheet-frame points into the board frame."""
    points = np.atleast_2d(np.asarray(points_sheet_mm, dtype=np.float64))
    return (points - _sheet_origin_mm(spec)) @ SHEET_TO_BOARD_ROTATION.T


def board_to_sheet_mm(points_board_mm: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """Map ``(N, 3)`` board-frame points back into the sheet frame."""
    points = np.atleast_2d(np.asarray(points_board_mm, dtype=np.float64))
    return points @ SHEET_TO_BOARD_ROTATION.T + _sheet_origin_mm(spec)


def sheet_pose_to_board_pose_with_origin(
    rvec_sheet_rad: np.ndarray,
    tvec_sheet_mm: np.ndarray,
    origin_sheet_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a sheet-frame pose into the board frame, given the board-frame origin.

    Both poses map their own frame to the camera frame. This is the single
    implementation; :func:`sheet_pose_to_board_pose` is the spec-driven wrapper.

    Returns:
        ``(rvec_rad, tvec_mm)`` in the board frame, each shaped ``(3,)``.
    """
    rotation_sheet, _ = cv2.Rodrigues(np.asarray(rvec_sheet_rad, dtype=np.float64))
    rotation_board = rotation_sheet @ SHEET_TO_BOARD_ROTATION
    translation_board_mm = rotation_sheet @ np.asarray(
        origin_sheet_mm, dtype=np.float64
    ).reshape(3) + np.asarray(tvec_sheet_mm, dtype=np.float64).reshape(3)
    rvec_board_rad, _ = cv2.Rodrigues(rotation_board)
    return rvec_board_rad.reshape(3), translation_board_mm.reshape(3)


def sheet_pose_to_board_pose(
    rvec_sheet_rad: np.ndarray, tvec_sheet_mm: np.ndarray, spec: BoardSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Move a pose solved in OpenCV's sheet frame into the pipeline's board frame."""
    return sheet_pose_to_board_pose_with_origin(
        rvec_sheet_rad, tvec_sheet_mm, _sheet_origin_mm(spec)
    )


# --------------------------------------------------------------------------------------
# Board-frame geometry used by masking
# --------------------------------------------------------------------------------------


def board_rectangle_board_mm(spec: BoardSpec) -> np.ndarray:
    """The four printed-sheet corners as ``(4, 3)`` board-frame points, in order."""
    sheet_corners_mm = np.array(
        [
            [0.0, 0.0, 0.0],
            [spec.board_width_mm, 0.0, 0.0],
            [spec.board_width_mm, spec.board_height_mm, 0.0],
            [0.0, spec.board_height_mm, 0.0],
        ],
        dtype=np.float64,
    )
    return sheet_to_board_mm(sheet_corners_mm, spec)


def scale_block_box_board_mm(spec: BoardSpec) -> np.ndarray:
    """The scale reference block as an ``(8, 3)`` board-frame box.

    The footprint is a square centred on ``scale_block_position_mm`` with side
    ``scale_block_footprint_mm``. When that is null the footprint falls back to one board
    square (``square_size_mm``) - which on the 7x7 / 190 mm board is only 27.14 mm, so a
    physically wider block overhangs the box and its pixels leak into the plant
    silhouette. Set ``scale_block_footprint_mm`` once the block is measured. The height is
    ``scale_block_nominal_height_mm``. The box is used only to *exclude* the block from the
    plant silhouette.

    Raises:
        BoardError: ``scale_block_nominal_height_mm`` is null. An unmeasured block has no
            box; a placeholder height would silently exclude the wrong pixels.
    """
    if spec.scale_block_nominal_height_mm is None:
        raise BoardError(
            "scale_block_nominal_height_mm is null in board_spec.json; the scale block "
            "has not been measured, so its exclusion box cannot be built"
        )
    half_side_mm = spec.scale_block_footprint_side_mm / 2.0
    center_x_mm, center_y_mm = spec.scale_block_position_mm
    footprint_sheet_mm = np.array(
        [
            [center_x_mm - half_side_mm, center_y_mm - half_side_mm, 0.0],
            [center_x_mm + half_side_mm, center_y_mm - half_side_mm, 0.0],
            [center_x_mm + half_side_mm, center_y_mm + half_side_mm, 0.0],
            [center_x_mm - half_side_mm, center_y_mm + half_side_mm, 0.0],
        ],
        dtype=np.float64,
    )
    footprint_board_mm = sheet_to_board_mm(footprint_sheet_mm, spec)
    top_board_mm = footprint_board_mm + np.array(
        [0.0, 0.0, spec.scale_block_nominal_height_mm], dtype=np.float64
    )
    return np.vstack([footprint_board_mm, top_board_mm])
