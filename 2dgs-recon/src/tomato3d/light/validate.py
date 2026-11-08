"""Predicted PPFD against quantum-sensor readings.

A person assigns every reading a ``split_role`` before anything is fitted:

* ``calibration`` readings characterise the lamp model. One multiplicative factor per
  scenario is fitted on them, against the predictions of the *calibration variant*
  (by default ``no_plants``, the light field without plant geometry), and applied
  unchanged to every geometry variant - so variants are compared on the same lamp model
  and none of them is tuned to its own readings.
* ``evaluation`` readings are held out. Their errors are the reported accuracy.

A quantum sensor validates incident PPFD at its sensing surface. It does not validate
leaf absorption, whole-plant capture or photosynthesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomato3d.io.schemas import (
    PPFD_READINGS_CSV_COLUMNS,
    SPLIT_ROLES,
    UNAVAILABLE,
    LightScene,
    format_maybe_float,
)
from tomato3d.io.tables import read_table
from tomato3d.light.raytrace import Receivers
from tomato3d.light.scene import LightInputError

ALL_READINGS_SUBSET = "all"
_FULL_OUTPUT_YES = "yes"


@dataclass(frozen=True)
class Reading:
    """One quantum-sensor reading, grow frame."""

    reading_id: str
    scenario_id: str
    split_role: str
    region_label: str
    point_grow_mm: np.ndarray  # (3,)
    unit_normal: np.ndarray  # (3,)
    #: ``None`` when the reading is ``unavailable``.
    measured_ppfd_umol_m2_s: float | None


@dataclass(frozen=True)
class ErrorMetrics:
    """Prediction minus measurement, over readings where both are available."""

    n_used: int
    n_measured_unavailable: int
    n_prediction_unavailable: int
    mae_umol_m2_s: float
    rmse_umol_m2_s: float
    bias_umol_m2_s: float


def _number(path: Path, index: int, row: dict[str, str], column: str) -> float:
    text = row[column].strip()
    where = f"{path.name} row {index + 2}"
    if text == UNAVAILABLE:
        raise LightInputError(f"{where}: {column} is unavailable; the reading cannot be placed")
    try:
        value = float(text)
    except ValueError as exc:
        raise LightInputError(f"{where}: {column} is not a number: {text!r}") from exc
    if not math.isfinite(value):
        raise LightInputError(f"{where}: {column} must be finite")
    return value


def load_readings(path: Path, scene: LightScene) -> list[Reading]:
    """Read ``ppfd_readings.csv`` and check it against the scene.

    Raises:
        LightInputError: a reading from another setup or an unknown scenario, a partial
            output reading, a missing position or normal, a negative PPFD or a repeated id.
    """
    readings: list[Reading] = []
    seen: set[str] = set()
    for index, row in enumerate(read_table(path, PPFD_READINGS_CSV_COLUMNS)):
        where = f"{path.name} row {index + 2}"
        reading_id = row["reading_id"].strip()
        if not reading_id or reading_id in seen:
            raise LightInputError(f"{where}: reading_id {reading_id!r} is empty or repeated")
        seen.add(reading_id)
        if row["setup_id"].strip() != scene.setup_id:
            raise LightInputError(
                f"{where}: setup_id {row['setup_id']!r} is not the scene's {scene.setup_id!r}"
            )
        scenario_id = row["scenario_id"].strip()
        if scenario_id not in scene.scenario_ids:
            raise LightInputError(
                f"{where}: scenario_id {scenario_id!r} is not in {scene.scenario_ids}"
            )
        if row["all_fixtures_full_output"].strip() != _FULL_OUTPUT_YES:
            raise LightInputError(
                f"{where}: only readings with every enabled fixture at full output are "
                "supported (all_fixtures_full_output must be 'yes')"
            )
        split_role = row["split_role"].strip()
        if split_role not in SPLIT_ROLES:
            raise LightInputError(f"{where}: split_role must be one of {SPLIT_ROLES}")
        point = np.array([_number(path, index, row, f"{a}_mm") for a in ("x", "y", "z")])
        normal = np.array([_number(path, index, row, f"normal_{a}") for a in ("x", "y", "z")])
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise LightInputError(f"{where}: the sensor normal is a zero vector")
        ppfd_text = row["ppfd_umol_m2_s"].strip()
        measured = None if ppfd_text == UNAVAILABLE else _number(path, index, row, "ppfd_umol_m2_s")
        if measured is not None and measured < 0.0:
            raise LightInputError(f"{where}: ppfd_umol_m2_s must be >= 0")
        readings.append(
            Reading(
                reading_id=reading_id,
                scenario_id=scenario_id,
                split_role=split_role,
                region_label=row["region_label"].strip() or UNAVAILABLE,
                point_grow_mm=point,
                unit_normal=normal / norm,
                measured_ppfd_umol_m2_s=measured,
            )
        )
    return readings


def readings_as_receivers(readings: list[Reading]) -> Receivers:
    """Sensors as one-sided cosine receivers that lie on no plant surface."""
    return Receivers(
        points_grow_mm=np.array([r.point_grow_mm for r in readings]).reshape(-1, 3),
        unit_normals=np.array([r.unit_normal for r in readings]).reshape(-1, 3),
        two_sided=False,
        own_triangle=np.full(len(readings), -1, dtype=np.int64),
    )


def calibration_factor(
    measured: list[float | None], predicted: np.ndarray
) -> float | None:
    """Least-squares ``k`` in ``measured = k * predicted``; ``None`` without usable pairs."""
    pairs = [
        (m, float(p)) for m, p in zip(measured, predicted) if m is not None and np.isfinite(p)
    ]
    denominator = sum(p * p for _, p in pairs)
    if not pairs or denominator == 0.0:
        return None
    return sum(m * p for m, p in pairs) / denominator


def error_metrics(measured: list[float | None], predicted: np.ndarray) -> ErrorMetrics:
    """MAE, RMSE and bias of ``predicted - measured``; unavailable values are counted, not used."""
    errors = []
    n_measured_unavailable = n_prediction_unavailable = 0
    for m, p in zip(measured, predicted):
        if m is None:
            n_measured_unavailable += 1
        elif not np.isfinite(p):
            n_prediction_unavailable += 1
        else:
            errors.append(float(p) - m)
    if not errors:
        nan = float("nan")
        return ErrorMetrics(0, n_measured_unavailable, n_prediction_unavailable, nan, nan, nan)
    array = np.asarray(errors)
    return ErrorMetrics(
        n_used=array.size,
        n_measured_unavailable=n_measured_unavailable,
        n_prediction_unavailable=n_prediction_unavailable,
        mae_umol_m2_s=float(np.abs(array).mean()),
        rmse_umol_m2_s=float(np.sqrt((array**2).mean())),
        bias_umol_m2_s=float(array.mean()),
    )


def sensor_prediction_rows(
    readings: list[Reading],
    geometry_variant_id: str,
    predicted: np.ndarray,
    factor: float | None,
) -> list[dict[str, str]]:
    rows = []
    for reading, p in zip(readings, predicted):
        calibrated = float("nan") if factor is None else factor * float(p)
        measured = reading.measured_ppfd_umol_m2_s
        rows.append(
            {
                "reading_id": reading.reading_id,
                "scenario_id": reading.scenario_id,
                "geometry_variant_id": geometry_variant_id,
                "split_role": reading.split_role,
                "region_label": reading.region_label,
                "measured_ppfd_umol_m2_s": format_maybe_float(measured),
                "predicted_ppfd_umol_m2_s": format_maybe_float(float(p)),
                "calibrated_ppfd_umol_m2_s": format_maybe_float(calibrated),
                "error_umol_m2_s": format_maybe_float(
                    None if measured is None else float(p) - measured
                ),
                "calibrated_error_umol_m2_s": format_maybe_float(
                    None if measured is None else calibrated - measured
                ),
            }
        )
    return rows


def _bin_label(low: float, high: float) -> str:
    return f"[{low:g},{high:g})"


def validation_rows(
    scenario_id: str,
    geometry_variant_id: str,
    readings: list[Reading],
    predicted: np.ndarray,
    factor: float | None,
    bin_edges: tuple[float, ...] | None,
) -> list[dict[str, str]]:
    """Error rows per split role, over all readings and, when bins are set, per bin."""
    rows = []
    calibrated = predicted * (np.nan if factor is None else factor)
    for role in SPLIT_ROLES:
        role_index = [i for i, r in enumerate(readings) if r.split_role == role]
        if not role_index:
            continue
        subsets = [(ALL_READINGS_SUBSET, role_index)]
        if bin_edges is not None:
            for low, high in zip(bin_edges, bin_edges[1:]):
                members = [
                    i
                    for i in role_index
                    if readings[i].measured_ppfd_umol_m2_s is not None
                    and low <= readings[i].measured_ppfd_umol_m2_s < high
                ]
                subsets.append((_bin_label(low, high), members))
        for label, members in subsets:
            measured = [readings[i].measured_ppfd_umol_m2_s for i in members]
            raw = error_metrics(measured, predicted[members])
            adjusted = error_metrics(measured, calibrated[members])
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "geometry_variant_id": geometry_variant_id,
                    "split_role": role,
                    "subset": label,
                    "n_used": str(raw.n_used),
                    "n_measured_unavailable": str(raw.n_measured_unavailable),
                    "n_prediction_unavailable": str(raw.n_prediction_unavailable),
                    "mae_umol_m2_s": format_maybe_float(raw.mae_umol_m2_s),
                    "rmse_umol_m2_s": format_maybe_float(raw.rmse_umol_m2_s),
                    "bias_umol_m2_s": format_maybe_float(raw.bias_umol_m2_s),
                    "calibration_factor": format_maybe_float(factor),
                    "calibrated_mae_umol_m2_s": format_maybe_float(adjusted.mae_umol_m2_s),
                    "calibrated_rmse_umol_m2_s": format_maybe_float(adjusted.rmse_umol_m2_s),
                    "calibrated_bias_umol_m2_s": format_maybe_float(adjusted.bias_umol_m2_s),
                }
            )
    return rows
