"""Per-session quality record.

``usable_for_training`` is decided here and nowhere else. A ``null`` threshold in
``config/quality_thresholds.json`` is skipped: it neither passes nor fails a session, and
no default is substituted for it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tomato3d.io.schemas import (
    UNAVAILABLE,
    MaybeFloat,
    Pose,
    QualityRecord,
    QualityThresholds,
    dump_json_model,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Validity failures
#
# Unlike a threshold, none of these is configurable and none is skipped when a threshold is
# null: each means a required Stage 1 product is missing or known to be wrong, so the
# session cannot be training-usable whatever the thresholds say.
# --------------------------------------------------------------------------------------

#: A frame's pixel size differs from the calibration's, so its pose would be solved with
#: the wrong camera matrix - wrong, yet with a small residual.
REASON_IMAGE_SIZE_MISMATCH = "image_size_mismatch_with_intrinsics"

#: An image listed in the manifest could not be decoded.
REASON_UNREADABLE_IMAGE = "unreadable_image"

#: The operational top view has no plant silhouette (unsupported, empty, or blocked).
REASON_OPERATIONAL_TOP_SILHOUETTE_UNAVAILABLE = "operational_top_view_silhouette_unavailable"

#: A side view whose pose solved has no plant silhouette (empty or blocked).
REASON_SIDE_SILHOUETTE_UNAVAILABLE = "side_view_silhouette_unavailable"

VALIDITY_REASONS: tuple[str, ...] = (
    REASON_IMAGE_SIZE_MISMATCH,
    REASON_UNREADABLE_IMAGE,
    REASON_OPERATIONAL_TOP_SILHOUETTE_UNAVAILABLE,
    REASON_SIDE_SILHOUETTE_UNAVAILABLE,
)


def pose_solved_ratio(poses: Sequence[Pose]) -> MaybeFloat:
    """Fraction of poses that solved. ``unavailable`` when there are no poses at all."""
    if not poses:
        return UNAVAILABLE
    return sum(1 for pose in poses if pose.is_solved) / len(poses)


def mean_reprojection_error_px(poses: Sequence[Pose]) -> MaybeFloat:
    """Mean reprojection error over solved poses. ``unavailable`` when none solved."""
    errors_px = [
        pose.reprojection_error_px
        for pose in poses
        if pose.is_solved and pose.reprojection_error_px is not None
    ]
    if not errors_px:
        return UNAVAILABLE
    return float(np.mean(errors_px))


def evaluate_usable_for_training(
    *,
    solved_ratio: MaybeFloat,
    mean_reprojection_error: MaybeFloat,
    silhouette_cv: MaybeFloat,
    thresholds: QualityThresholds,
) -> tuple[bool, list[str]]:
    """Gate a session against the *active* thresholds.

    Returns:
        ``(usable_for_training, failure_reasons)``. A metric that is ``unavailable``
        cannot be shown to pass, so it fails the gate with an explicit reason rather than
        being imputed. A threshold that is ``null`` is skipped entirely.
    """
    failure_reasons: list[str] = []

    checks: tuple[tuple[str, MaybeFloat, float | None, str], ...] = (
        (
            "pose_solved_ratio",
            solved_ratio,
            thresholds.min_pose_solved_ratio,
            "min",
        ),
        (
            "mean_reprojection_error_px",
            mean_reprojection_error,
            thresholds.max_mean_reprojection_error_px,
            "max",
        ),
        (
            "silhouette_area_cv",
            silhouette_cv,
            thresholds.max_silhouette_area_cv,
            "max",
        ),
    )

    for metric_name, value, limit, direction in checks:
        if limit is None:
            logger.info("threshold for %s is null; not gating on it", metric_name)
            continue
        if value == UNAVAILABLE:
            failure_reasons.append(f"{metric_name}_unavailable")
            continue
        number = float(value)
        if direction == "min" and number < limit:
            failure_reasons.append(f"{metric_name}_below_{limit}")
        elif direction == "max" and number > limit:
            failure_reasons.append(f"{metric_name}_above_{limit}")

    return (not failure_reasons), failure_reasons


def build_quality_record(
    *,
    session_id: str,
    poses: Sequence[Pose],
    silhouette_cv: MaybeFloat,
    thresholds: QualityThresholds,
    validity_failures: Sequence[str] = (),
) -> QualityRecord:
    """Assemble ``quality.json`` for one session.

    Args:
        validity_failures: Codes from :data:`VALIDITY_REASONS` observed in this run. Any
            one of them makes the session not usable for training; each is reported once,
            after the threshold reasons, in the order given.

    Stage 1 deliberately leaves ``scale_block_height_error_mm``, ``hull_voxel_count`` and
    ``depth_scale_fit_rmse_mm`` *absent* - later stages add them.

    Raises:
        ValueError: a validity code is not one of :data:`VALIDITY_REASONS`.
    """
    unknown = [code for code in validity_failures if code not in VALIDITY_REASONS]
    if unknown:
        raise ValueError(f"unknown validity failure codes: {unknown}")

    solved_ratio = pose_solved_ratio(poses)
    mean_error_px = mean_reprojection_error_px(poses)
    usable, failure_reasons = evaluate_usable_for_training(
        solved_ratio=solved_ratio,
        mean_reprojection_error=mean_error_px,
        silhouette_cv=silhouette_cv,
        thresholds=thresholds,
    )
    for code in dict.fromkeys(validity_failures):
        failure_reasons.append(code)
        usable = False
    return QualityRecord(
        session_id=session_id,
        pose_solved_ratio=solved_ratio,
        mean_reprojection_error_px=mean_error_px,
        silhouette_area_cv=silhouette_cv,
        n_images_total=len(poses),
        n_images_solved=sum(1 for pose in poses if pose.is_solved),
        usable_for_training=usable,
        failure_reasons=failure_reasons,
    )


def write_quality(path: Path, record: QualityRecord) -> None:
    """Write ``quality.json``."""
    dump_json_model(path, record)
