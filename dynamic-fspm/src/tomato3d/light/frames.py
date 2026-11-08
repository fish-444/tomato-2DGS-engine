"""Coordinate maps of the light calculation.

``board frame``  plant geometry, as reconstructed on the rotating board (see
                 :mod:`tomato3d.io.schemas`). Origin at ChArUco corner id 0, ``+Z`` up.
``grow frame``   the growing space under the lamps. Millimetres, ``+Z`` up.

:func:`board_to_grow_mm` is the only implementation of the board -> grow map. The board
lies flat on the shelf, so the map is a yaw about ``+Z`` plus a translation. If the board
is ever tilted in the grow space, this function - and nothing else - changes.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from tomato3d.io.schemas import SchemaError

#: A fixture whose long axis is parallel to its aim axis has no defined long side.
_DEGENERATE_AXIS_NORM: Final[float] = 1e-9


def board_to_grow_mm(
    points_board_mm: np.ndarray,
    board_origin_grow_mm: np.ndarray,
    board_yaw_rad: float,
) -> np.ndarray:
    """Map board-frame points into the grow frame.

    Args:
        points_board_mm: ``(..., 3)`` points in the board frame.
        board_origin_grow_mm: ``(3,)`` grow-frame position of the board origin.
        board_yaw_rad: rotation of board ``+X`` from grow ``+X`` toward grow ``+Y``.

    Returns:
        ``(..., 3)`` points in the grow frame.
    """
    cos_yaw, sin_yaw = np.cos(board_yaw_rad), np.sin(board_yaw_rad)
    rotation = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
    )
    points = np.asarray(points_board_mm, dtype=np.float64)
    return points @ rotation.T + np.asarray(board_origin_grow_mm, dtype=np.float64)


def aim_direction(aim_tilt_rad: float, aim_azimuth_rad: float) -> np.ndarray:
    """Unit aim vector: tilt from straight down, leaning toward the azimuth."""
    return np.array(
        [
            np.sin(aim_tilt_rad) * np.cos(aim_azimuth_rad),
            np.sin(aim_tilt_rad) * np.sin(aim_azimuth_rad),
            -np.cos(aim_tilt_rad),
        ]
    )


def fixture_axes(
    aim_tilt_rad: float, aim_azimuth_rad: float, long_axis_azimuth_rad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal ``(aim, long_axis, width_axis)`` of a fixture in the grow frame.

    The long axis is the horizontal direction ``long_axis_azimuth_rad`` made
    perpendicular to the aim axis, so tilting a bar lamp about its long side keeps the
    long side where it was.

    Raises:
        SchemaError: the long axis is parallel to the aim axis.
    """
    aim = aim_direction(aim_tilt_rad, aim_azimuth_rad)
    horizontal = np.array([np.cos(long_axis_azimuth_rad), np.sin(long_axis_azimuth_rad), 0.0])
    long_axis = horizontal - np.dot(horizontal, aim) * aim
    norm = float(np.linalg.norm(long_axis))
    if norm < _DEGENERATE_AXIS_NORM:
        raise SchemaError(
            "fixture long_axis_azimuth_deg is parallel to its aim axis; the long side is "
            "undefined"
        )
    long_axis = long_axis / norm
    width_axis = np.cross(aim, long_axis)
    return aim, long_axis, width_axis
