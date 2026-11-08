"""Plant silhouette masking.

The board is rejected **geometrically**, never by colour: the substrate is vermiculite,
which is as bright and as uniform as parts of the board, so any colour rule that removes
the board also removes the pot. Instead the board rectangle is projected from the solved
pose and everything on the board plane within its measured extent is discarded.

What survives that rejection is the image of the vertical prism standing on the board
footprint - "everything above the board plane inside the board footprint". Only inside
that region is a pixel classified as plant or not, and that classification is the one
step that does read colour.

Consequence worth knowing: anything that projects *onto* the board quad, rather than
above it, is discarded with the board. From a side view that is the pot and lower stem;
from a top view it is the **whole plant**, which is why :data:`SUPPORTED_VIEW_TYPES`
excludes top views. That is what rejecting the board by its 2D projection costs, and it
is why this boundary is kept stable - a segmenter that keeps foreground above the board
can replace :func:`segment_plant_silhouette` without touching any caller.
"""

from __future__ import annotations

import logging
from typing import Final

import cv2
import numpy as np

from tomato3d.io.schemas import BoardSpec, Intrinsics, Pose, SchemaError

logger = logging.getLogger(__name__)

MASK_FOREGROUND = 255
MASK_BACKGROUND = 0

#: How far the vertical prism is extended past the board, in image diagonals. The prism
#: over the board footprint is unbounded upward, so it is drawn out to well beyond the
#: frame and then clipped by the image itself.
_PRISM_EXTENSION_IMAGE_DIAGONALS = 2.0

#: Morphological cleanup kernel, as a fraction of the shorter image side. Structural
#: rather than tuned: it removes single-pixel speckle at any capture resolution.
_MORPH_KERNEL_FRACTION_OF_SHORT_SIDE = 0.004

#: Minimum kernel side in pixels.
_MIN_MORPH_KERNEL_PX = 3


class MaskError(SchemaError):
    """Raised when a mask cannot be built from the supplied pose."""


def _require_solved(pose: Pose) -> None:
    if not pose.is_solved:
        raise MaskError("masking requires a solved pose")


def _project_board_points(
    points_board_mm: np.ndarray, pose: Pose, intrinsics: Intrinsics
) -> np.ndarray:
    """Project ``(N, 3)`` board-frame points to ``(N, 2)`` image points."""
    projected_px, _ = cv2.projectPoints(
        np.asarray(points_board_mm, dtype=np.float64),
        np.asarray(pose.rvec_rad, dtype=np.float64).reshape(3, 1),
        np.asarray(pose.tvec_mm, dtype=np.float64).reshape(3, 1),
        intrinsics.camera_matrix_array,
        intrinsics.dist_coeffs_array,
    )
    return projected_px.reshape(-1, 2)


def _fill_polygon(
    polygon_px: np.ndarray, image_shape: tuple[int, int], *, convex: bool
) -> np.ndarray:
    """Rasterise a polygon into a binary mask, clipping coordinates to a safe range."""
    height_px, width_px = image_shape[:2]
    mask = np.zeros((height_px, width_px), dtype=np.uint8)
    if len(polygon_px) < 3 or not np.isfinite(polygon_px).all():
        return mask
    # Clip well outside the frame rather than at its edge, so that the polygon keeps its
    # shape where it crosses the border instead of being folded onto it.
    limit_px = float(max(height_px, width_px)) * (
        1.0 + _PRISM_EXTENSION_IMAGE_DIAGONALS * 2.0
    )
    clipped = np.clip(polygon_px, -limit_px, limit_px)
    points = np.round(clipped).astype(np.int32)
    if convex:
        cv2.fillConvexPoly(mask, points, MASK_FOREGROUND)
    else:
        cv2.fillPoly(mask, [points], MASK_FOREGROUND)
    return mask


def board_plane_mask(
    pose: Pose, intrinsics: Intrinsics, spec: BoardSpec, image_shape: tuple[int, int]
) -> np.ndarray:
    """Project the board rectangle into the image.

    Returns a binary mask of the board region. Purely geometric - no pixel colour is
    read, so the result depends only on the pose, the intrinsics and the board spec.
    """
    from tomato3d.calib.board import board_rectangle_board_mm

    _require_solved(pose)
    corners_px = _project_board_points(
        board_rectangle_board_mm(spec), pose, intrinsics
    )
    return _fill_polygon(corners_px, image_shape, convex=True)


#: View types :func:`segment_plant_silhouette` supports.
#:
#: Top views are excluded. Seen from above, the whole plant projects *inside* the board
#: quad, so rejecting that quad removes the plant entirely: a synthetic T0 view keeps 0 of
#: 60,077 plant pixels. An empty or partial top silhouette is worse than none - it would
#: carve the plant out of a visual hull - so no top-view mask is produced until the board
#: rejection can keep foreground that stands above the board. That is a design decision
#: still open; widening this tuple without it reintroduces the defect.
SUPPORTED_VIEW_TYPES: Final[tuple[str, ...]] = ("side",)


def view_supports_silhouette(view_type: str) -> bool:
    """Whether the current segmenter can produce a trustworthy mask for ``view_type``."""
    return view_type in SUPPORTED_VIEW_TYPES


def silhouette_blockers(spec: BoardSpec) -> list[str]:
    """Reasons :func:`segment_plant_silhouette` cannot run with this spec at all.

    Empty when segmentation is possible. This is the single statement of those
    preconditions: the segmenter raises on them, and a caller that processes many frames
    can check once instead of collecting the same failure per frame.
    """
    blockers: list[str] = []
    if spec.scale_block_nominal_height_mm is None:
        blockers.append(
            "scale_block_nominal_height_mm is null, so the scale block cannot be excluded "
            "from the silhouette"
        )
    return blockers


def scale_block_mask(
    pose: Pose, intrinsics: Intrinsics, spec: BoardSpec, image_shape: tuple[int, int]
) -> np.ndarray:
    """Project the scale reference block. Purely geometric.

    Raises:
        MaskError: the block height is null in ``board_spec.json``; the message names the
            field.
    """
    from tomato3d.calib.board import BoardError, scale_block_box_board_mm

    _require_solved(pose)
    try:
        box_board_mm = scale_block_box_board_mm(spec)
    except BoardError as exc:
        raise MaskError(str(exc)) from exc
    box_px = _project_board_points(box_board_mm, pose, intrinsics)
    if len(box_px) < 3 or not np.isfinite(box_px).all():
        return np.zeros(image_shape[:2], dtype=np.uint8)
    hull_px = cv2.convexHull(box_px.astype(np.float32)).reshape(-1, 2)
    return _fill_polygon(hull_px, image_shape, convex=True)


def workspace_prism_mask(
    pose: Pose, intrinsics: Intrinsics, spec: BoardSpec, image_shape: tuple[int, int]
) -> np.ndarray:
    """Image of the vertical prism standing on the board footprint.

    The prism is unbounded upward, so no plant-height constant is invented here: each
    footprint corner is extended along its own vertical image direction far past the
    frame, and the convex hull of the result is clipped by the image. Purely geometric.
    """
    from tomato3d.calib.board import board_rectangle_board_mm

    _require_solved(pose)
    height_px, width_px = image_shape[:2]
    footprint_board_mm = board_rectangle_board_mm(spec)
    base_px = _project_board_points(footprint_board_mm, pose, intrinsics)

    # One board square is a convenient, config-derived probe height for the direction
    # that "up" takes in the image; its magnitude cancels in the normalisation.
    probe_board_mm = footprint_board_mm + np.array(
        [0.0, 0.0, spec.square_size_mm], dtype=np.float64
    )
    probe_px = _project_board_points(probe_board_mm, pose, intrinsics)

    directions_px = probe_px - base_px
    norms_px = np.linalg.norm(directions_px, axis=1, keepdims=True)
    if not np.isfinite(norms_px).all() or (norms_px == 0.0).any():
        logger.warning("degenerate vertical direction; prism falls back to the footprint")
        return _fill_polygon(base_px, image_shape, convex=True)

    extension_px = (
        float(np.hypot(width_px, height_px)) * _PRISM_EXTENSION_IMAGE_DIAGONALS
    )
    top_px = base_px + directions_px / norms_px * extension_px
    hull_px = cv2.convexHull(
        np.vstack([base_px, top_px]).astype(np.float32)
    ).reshape(-1, 2)
    return _fill_polygon(hull_px, image_shape, convex=True)


def _excess_green(image: np.ndarray) -> np.ndarray:
    """Excess-green index ``2G - R - B`` as float32.

    Plant tissue separates from vermiculite, pot and room background on this index far
    more cleanly than on intensity. For a single-channel image there is no colour to
    use, so intensity is returned and the caller gets an intensity threshold instead.
    """
    if image.ndim == 2:
        return image.astype(np.float32)
    blue, green, red = cv2.split(image.astype(np.float32))
    return 2.0 * green - red - blue


def _otsu_within(values: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Otsu threshold computed over ``region`` only, applied to ``values``."""
    inside = values[region]
    value_min = float(inside.min())
    value_max = float(inside.max())
    if value_max <= value_min:
        return np.zeros(values.shape, dtype=bool)
    scaled = np.clip(
        (values - value_min) / (value_max - value_min) * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    threshold, _ = cv2.threshold(
        scaled[region].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return scaled > threshold


def _cleanup(mask: np.ndarray) -> np.ndarray:
    """Morphological open/close, then keep only the largest connected component."""
    height_px, width_px = mask.shape[:2]
    kernel_px = max(
        _MIN_MORPH_KERNEL_PX,
        int(round(min(height_px, width_px) * _MORPH_KERNEL_FRACTION_OF_SHORT_SIDE)),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels <= 1:
        return closed
    areas_px = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas_px)) + 1
    return np.where(labels == largest_label, MASK_FOREGROUND, MASK_BACKGROUND).astype(
        np.uint8
    )


def segment_plant_silhouette(
    image: np.ndarray, pose: Pose, intrinsics: Intrinsics, spec: BoardSpec
) -> np.ndarray:
    """Binary uint8 mask, plant = 255.

    Excludes the board region and the scale block using :func:`board_plane_mask` and the
    known ``scale_block_position_mm``. Everything above the board plane inside the board
    footprint is a plant candidate; within that candidate region an excess-green Otsu
    split plus largest-component cleanup separates plant from background.

    The colour chart is removed only when ``color_chart_on_board`` is true, because then
    the board rejection covers it. When the chart stands beside the board - the
    operational configuration - it has no position in the spec and is **not** excluded:
    from an elevated side view a chart beyond the far board edge projects into the plant
    candidate region, and its green patches can enter the silhouette.

    Raises:
        MaskError: the pose failed, or :func:`silhouette_blockers` reports a missing
            precondition (e.g. an unmeasured scale block). A frame that cannot be masked
            gets no mask - callers record that rather than guessing one.
    """
    _require_solved(pose)
    blockers = silhouette_blockers(spec)
    if blockers:
        raise MaskError("; ".join(blockers))
    image_shape = image.shape[:2]

    prism = workspace_prism_mask(pose, intrinsics, spec, image_shape) > 0
    board = board_plane_mask(pose, intrinsics, spec, image_shape) > 0
    block = scale_block_mask(pose, intrinsics, spec, image_shape) > 0
    candidate = prism & ~board & ~block

    if not candidate.any():
        logger.warning("empty plant candidate region; the board fills the frame")
        return np.zeros(image_shape, dtype=np.uint8)

    greenness = _excess_green(image)
    plant = candidate & _otsu_within(greenness, candidate)
    mask = np.where(plant, MASK_FOREGROUND, MASK_BACKGROUND).astype(np.uint8)
    return _cleanup(mask)


def silhouette_area_cv(masks: list[np.ndarray]) -> float:
    """Coefficient of variation of silhouette pixel area across side views.

    A high value means the plant moved during rotation or masking failed.

    Raises:
        ValueError: fewer than two masks, or every mask is empty, so the coefficient is
            undefined. The caller records ``unavailable``.
    """
    if len(masks) < 2:
        raise ValueError(f"silhouette_area_cv needs at least 2 masks, got {len(masks)}")
    areas_px = np.array([int(np.count_nonzero(mask)) for mask in masks], dtype=np.float64)
    mean_area_px = float(areas_px.mean())
    if mean_area_px <= 0.0:
        raise ValueError("every silhouette is empty; area CV is undefined")
    return float(areas_px.std() / mean_area_px)
