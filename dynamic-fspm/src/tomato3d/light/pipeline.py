"""One light run: scene + geometry variants (+ readings) -> every table and the summary.

For each scenario (the as-built arrangement first) and each geometry variant (the empty
space ``no_plants`` first):

1. place the variant's plant surfaces in the grow frame;
2. sample every leaf triangle and trace direct light from every fixture, with shading;
3. aggregate into the leaf light map, per-leaf, per-plant and canopy budgets;
4. predict PPFD at the scenario's sensor readings.

Then fit the lamp calibration factor on calibration readings, score every variant on the
evaluation readings, and compare geometry variants against the reference variant.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np

from tomato3d.io.schemas import (
    NO_PLANTS_VARIANT_ID,
    SCHEMA_VERSION,
    UNAVAILABLE,
    GeometryVariantRecord,
    LightRunSummary,
    LightScene,
    MaybeFloat,
)
from tomato3d.light.budget import (
    VariantBudget,
    leaf_rows,
    plant_rows,
    scenario_rows,
    variant_budget,
)
from tomato3d.light.compare import geometry_variant_rows
from tomato3d.light.geometry import (
    SurfaceSet,
    empty_surface_set,
    load_geometry,
    place_in_grow_frame,
    sample_triangles,
    triangle_unit_normals,
)
from tomato3d.light.raytrace import Occluders, Receivers, direct_light
from tomato3d.light.scene import (
    LightInputError,
    ResolvedScene,
    input_records,
    non_observed_input_names,
    resolve_scenario,
)
from tomato3d.light.validate import (
    Reading,
    calibration_factor,
    load_readings,
    readings_as_receivers,
    sensor_prediction_rows,
    validation_rows,
)

logger = logging.getLogger(__name__)

RUN_SUMMARY_FILENAME: Final[str] = "run_summary.json"
LEAF_TRIANGLE_LIGHT_FILENAME: Final[str] = "leaf_triangle_light.csv"
LEAF_LIGHT_FILENAME: Final[str] = "leaf_light.csv"
PLANT_LIGHT_FILENAME: Final[str] = "plant_light.csv"
SCENARIO_SUMMARY_FILENAME: Final[str] = "scenario_summary.csv"
SENSOR_PREDICTIONS_FILENAME: Final[str] = "sensor_predictions.csv"
VALIDATION_FILENAME: Final[str] = "validation.csv"
GEOMETRY_VARIANT_COMPARISON_FILENAME: Final[str] = "geometry_variant_comparison.csv"

_CALIBRATION_ROLE: Final[str] = "calibration"

#: What this calculation does not do. Written into every run summary.
MODEL_LIMITS: Final[tuple[str, ...]] = (
    "Direct light only: light reflected by walls, shelves, the floor or leaves is not "
    "traced, so predicted PPFD is expected to read low wherever reflected light matters.",
    "Light transmitted through a leaf keeps its direction; diffuse scattering is not "
    "modelled.",
    "One set of leaf optical properties is applied to every leaf.",
    "Lamp emission is cos(theta)**m with a hard cutoff, not a measured goniometric profile.",
    "A quantum sensor is modelled as an ideal cosine receiver at a point; it does not "
    "displace leaves.",
    "Plant geometry is the geometry supplied; surfaces hidden in every photograph are "
    "absent unless a variant infers them, and each variant states its geometry_source.",
    "Absorbed photons are not photosynthesis: PSII efficiency, CO2 assimilation, growth "
    "and yield are not computed and need separate measurements.",
)
_AMBIENT_NOT_MODELLED: Final[str] = (
    "Ambient light is present in the growing space but not modelled; sensor readings "
    "include it and predictions do not."
)


@dataclass(frozen=True)
class LightRunResult:
    """Every table of a run, as string rows in their CSV column contracts."""

    summary: LightRunSummary
    budgets: list[VariantBudget]
    triangle_rows: list[dict[str, str]] = field(default_factory=list)
    leaf_rows: list[dict[str, str]] = field(default_factory=list)
    plant_rows: list[dict[str, str]] = field(default_factory=list)
    scenario_rows: list[dict[str, str]] = field(default_factory=list)
    sensor_rows: list[dict[str, str]] = field(default_factory=list)
    validation_rows: list[dict[str, str]] = field(default_factory=list)
    variant_rows: list[dict[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


def build_occluders(surfaces: SurfaceSet, vertices_grow_mm: np.ndarray) -> Occluders:
    """Every triangle of a variant as an occluder, grouped by plant for culling."""
    if surfaces.n_triangles == 0:
        return Occluders.none()
    surface_keys = list(zip(surfaces.plant_ids, surfaces.organ_ids, surfaces.leaflet_ids))
    codes = {key: code for code, key in enumerate(dict.fromkeys(surface_keys))}
    plants = {plant: index for index, plant in enumerate(surfaces.plant_id_set)}
    return Occluders(
        vertices_grow_mm=vertices_grow_mm,
        is_leaf=surfaces.is_receiver,
        surface_code=np.array([codes[key] for key in surface_keys], dtype=np.int64),
        group=np.array([plants[plant] for plant in surfaces.plant_ids], dtype=np.int64),
    )


def leaf_receivers(
    surfaces: SurfaceSet, vertices_grow_mm: np.ndarray, subdivisions: int
) -> tuple[Receivers, np.ndarray]:
    """Equal-area samples on every leaf triangle, in triangle order.

    Returns:
        ``(receivers, receiver_triangles)``; the second is the ``(N,)`` index of the
        sampled triangles into ``surfaces``.
    """
    receiver_triangles = np.flatnonzero(surfaces.is_receiver)
    points, local_index = sample_triangles(vertices_grow_mm[receiver_triangles], subdivisions)
    own = receiver_triangles[local_index]
    normals = (
        triangle_unit_normals(vertices_grow_mm)[own]
        if receiver_triangles.size
        else np.zeros((0, 3))
    )
    return (
        Receivers(
            points_grow_mm=points.reshape(-1, 3),
            unit_normals=normals.reshape(-1, 3),
            two_sided=True,
            own_triangle=own.astype(np.int64),
        ),
        receiver_triangles,
    )


def _scene_warnings(scene: LightScene, variants: dict[str, SurfaceSet]) -> list[str]:
    warnings: list[str] = []
    as_built = resolve_scenario(scene, scene.scenario_ids[0])
    for state in as_built.fixtures:
        if state.missing_inputs:
            warnings.append(
                f"fixture {state.fixture_id}: light field unavailable, missing "
                f"{', '.join(state.missing_inputs)}"
            )
        if state.electricity_missing_reason is not None:
            warnings.append(f"electricity unavailable - {state.electricity_missing_reason}")
    optics = scene.leaf_optics
    if not optics.transmittance.is_available:
        warnings.append(
            "leaf transmittance is unavailable: PPFD on any ray that passes through a leaf "
            "is unavailable"
        )
    if not optics.absorptance.is_available:
        warnings.append("leaf absorptance is unavailable: every absorbed quantity is unavailable")
    for placement in scene.plants:
        if placement.posture_since_capture != "unchanged_checked":
            warnings.append(
                f"plant {placement.plant_id}: posture since capture is "
                f"{placement.posture_since_capture}; the photographed leaf angles may not "
                "match the plant under the lamps"
            )
    placed = {placement.plant_id for placement in scene.plants}
    for variant_id, surfaces in variants.items():
        if variant_id == NO_PLANTS_VARIANT_ID:
            continue
        absent = sorted(placed - set(surfaces.plant_id_set))
        if absent:
            warnings.append(
                f"variant {variant_id} has no geometry for placed plants {absent}: they "
                "neither receive light nor cast shade in that variant"
            )
        if not bool(surfaces.normal_known[surfaces.is_receiver].all()):
            warnings.append(
                f"variant {variant_id} has leaf triangles with unknown upper side: "
                "adaxial_incident_fraction is unavailable for them"
            )
    return warnings


def _height_warnings(resolved: ResolvedScene, variant_id: str, vertices: np.ndarray) -> list[str]:
    if vertices.size == 0:
        return []
    top_mm = float(vertices[:, :, 2].max())
    warnings = []
    for state in resolved.fixtures:
        center = state.fixture.emitting_center_grow_mm.value
        if center is not None and top_mm >= center[2]:
            warnings.append(
                f"scenario {resolved.scenario_id}, variant {variant_id}: plant surfaces reach "
                f"{top_mm:.0f} mm, at or above fixture {state.fixture_id} "
                f"({center[2]:.0f} mm); leaves above a lamp receive none of its light"
            )
    return warnings


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def run_light(
    *,
    scene: LightScene,
    scene_path: Path,
    geometry_paths: Sequence[Path],
    readings_path: Path | None,
    reference_variant_id: str | None,
    calibration_variant_id: str,
    run_id: str,
    created_at: datetime,
) -> LightRunResult:
    """Compute every table of one run.

    Raises:
        LightInputError: inconsistent inputs (unknown variant ids, unplaced plants,
            readings from another setup, ...).
        SchemaError: a file fails its schema.
    """
    variants: dict[str, SurfaceSet] = {NO_PLANTS_VARIANT_ID: empty_surface_set()}
    variants.update(load_geometry(geometry_paths))
    for label, variant_id in (
        ("reference variant", reference_variant_id),
        ("calibration variant", calibration_variant_id),
    ):
        if variant_id is not None and variant_id not in variants:
            raise LightInputError(f"{label} {variant_id!r} is not among {sorted(variants)}")
    readings: list[Reading] = [] if readings_path is None else load_readings(readings_path, scene)

    warnings = _scene_warnings(scene, variants)
    simulation = scene.simulation
    samples_per_triangle = simulation.triangle_subdivisions**2
    transmittance = scene.leaf_optics.transmittance.value

    budgets: list[VariantBudget] = []
    sensor_rows: list[dict[str, str]] = []
    scores: list[dict[str, str]] = []
    factors: dict[str, MaybeFloat] = {}
    for scenario_id in scene.scenario_ids:
        resolved = resolve_scenario(scene, scenario_id)
        scenario_readings = [r for r in readings if r.scenario_id == scenario_id]
        predictions: dict[str, np.ndarray] = {}
        for variant_id, surfaces in variants.items():
            logger.info("scenario %s, variant %s", scenario_id, variant_id)
            vertices = place_in_grow_frame(surfaces, resolved.placements)
            warnings.extend(_height_warnings(resolved, variant_id, vertices))
            occluders = build_occluders(surfaces, vertices)
            receivers, receiver_triangles = leaf_receivers(
                surfaces, vertices, simulation.triangle_subdivisions
            )
            light = direct_light(receivers, resolved.fixtures, occluders, transmittance, simulation)
            budgets.append(
                variant_budget(
                    resolved, surfaces, vertices, receiver_triangles, light, samples_per_triangle
                )
            )
            if scenario_readings:
                sensors = direct_light(
                    readings_as_receivers(scenario_readings),
                    resolved.fixtures,
                    occluders,
                    transmittance,
                    simulation,
                )
                predictions[variant_id] = sensors.ppfd_umol_m2_s.sum(axis=0)

        if not scenario_readings:
            factors[scenario_id] = UNAVAILABLE
            continue
        calibration_index = [
            i for i, r in enumerate(scenario_readings) if r.split_role == _CALIBRATION_ROLE
        ]
        factor = calibration_factor(
            [scenario_readings[i].measured_ppfd_umol_m2_s for i in calibration_index],
            predictions[calibration_variant_id][calibration_index],
        )
        if factor is None:
            warnings.append(
                f"scenario {scenario_id}: no usable calibration readings; calibrated "
                "predictions are unavailable"
            )
        factors[scenario_id] = UNAVAILABLE if factor is None else factor
        for variant_id, predicted in predictions.items():
            sensor_rows.extend(
                sensor_prediction_rows(scenario_readings, variant_id, predicted, factor)
            )
            scores.extend(
                validation_rows(
                    scenario_id,
                    variant_id,
                    scenario_readings,
                    predicted,
                    factor,
                    scene.validation.ppfd_bin_edges_umol_m2_s,
                )
            )

    records = input_records(scene)
    summary = LightRunSummary(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        created_at=created_at,
        setup_id=scene.setup_id,
        scene_path=str(scene_path),
        geometry_paths=[str(path) for path in geometry_paths],
        readings_path=None if readings_path is None else str(readings_path),
        scenario_ids=scene.scenario_ids,
        geometry_variants=[
            GeometryVariantRecord(
                geometry_variant_id=variant_id,
                geometry_sources=sorted(set(surfaces.geometry_sources)),
                plant_ids=surfaces.plant_id_set,
                n_triangles=surfaces.n_triangles,
            )
            for variant_id, surfaces in variants.items()
        ],
        reference_variant_id=reference_variant_id,
        calibration_variant_id=calibration_variant_id,
        calibration_factor_by_scenario=factors,
        inputs=records,
        non_observed_inputs=non_observed_input_names(records),
        model_limits=[
            *MODEL_LIMITS,
            *((_AMBIENT_NOT_MODELLED,) if scene.ambient_light == "present_not_modelled" else ()),
        ],
        warnings=list(dict.fromkeys(warnings)),
    )
    return LightRunResult(
        summary=summary,
        budgets=budgets,
        triangle_rows=[row for budget in budgets for row in budget.triangle_rows],
        leaf_rows=[row for budget in budgets for row in leaf_rows(budget)],
        plant_rows=[row for budget in budgets for row in plant_rows(budget)],
        scenario_rows=scenario_rows(budgets),
        sensor_rows=sensor_rows,
        validation_rows=scores,
        variant_rows=(
            [] if reference_variant_id is None
            else geometry_variant_rows(budgets, reference_variant_id)
        ),
    )
