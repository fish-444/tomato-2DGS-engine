"""Light and energy budgets: per triangle, per leaf, per plant, per scenario.

Definitions (all fixtures at full output unless the name says ``_day``):

* ``ppfd`` - incident photons per leaf area and time, umol m-2 s-1, on whichever face the
  light reaches. Area is counted once: a leaf is one surface with two faces, not two
  surfaces.
* ``dli`` - the schedule-weighted daily integral of that PPFD, mol m-2 day-1.
* ``intercepted`` - incident photons summed over leaf area, umol s-1 (or mol day-1).
* ``absorbed`` - intercepted x leaf absorptance. A photon transmitted through one leaf
  and intercepted by another is attenuated before it arrives, so absorbed photons never
  exceed emitted photons while absorptance + transmittance <= 1.
* ``occlusion_loss_fraction`` - 1 - shaded PPFD / unshaded PPFD: the share of direct
  light that other surfaces took away (the shade map).
* ``absorbed_umol_per_j`` - absorbed photons per day / electrical energy per day, both
  over the fixtures and schedule of the scenario.

Internally an unavailable value is NaN, which propagates through arithmetic; it is
written as ``unavailable`` and is never summed away or replaced by 0.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from tomato3d.io.schemas import AS_BUILT_SCENARIO_ID, UNAVAILABLE, format_maybe_float
from tomato3d.light.geometry import SurfaceSet, triangle_areas_mm2
from tomato3d.light.raytrace import DirectLight
from tomato3d.light.scene import ResolvedScene
from tomato3d.light.units import JOULES_PER_KWH, MM2_PER_M2, SECONDS_PER_HOUR, UMOL_PER_MOL

_NAN = float("nan")


@dataclass(frozen=True)
class LeafResult:
    plant_id: str
    organ_id: str
    n_triangles: int
    area_mm2: float
    ppfd_mean_umol_m2_s: float
    ppfd_min_umol_m2_s: float
    ppfd_max_umol_m2_s: float
    occlusion_loss_fraction: float
    upper_side_fraction: float
    dli_mol_m2_day: float
    intercepted_umol_s: float
    absorbed_umol_s: float
    intercepted_mol_day: float
    absorbed_mol_day: float


@dataclass(frozen=True)
class PlantResult:
    plant_id: str
    n_leaves: int
    leaf_area_mm2: float
    intercepted_umol_s: float
    absorbed_umol_s: float
    intercepted_mol_day: float
    absorbed_mol_day: float
    leaf_ppfd_cv: float


@dataclass(frozen=True)
class CanopyResult:
    n_plants: int
    n_leaves: int
    leaf_area_mm2: float
    emitted_mol_day: float
    intercepted_mol_day: float
    absorbed_mol_day: float
    intercepted_fraction_of_emitted: float
    absorbed_fraction_of_emitted: float
    electricity_kwh_day: float
    absorbed_umol_per_j: float
    leaf_ppfd_mean_umol_m2_s: float
    leaf_ppfd_min_umol_m2_s: float
    leaf_ppfd_cv: float


@dataclass(frozen=True)
class VariantBudget:
    """Everything computed for one scenario x geometry variant."""

    scenario_id: str
    geometry_variant_id: str
    triangle_rows: list[dict[str, str]]
    leaves: list[LeafResult]
    plants: list[PlantResult]
    canopy: CanopyResult


# --------------------------------------------------------------------------------------
# Small arithmetic that keeps "unavailable" honest
# --------------------------------------------------------------------------------------


def dli_mol_m2_day(ppfd_umol_m2_s: float, full_output_hours_per_day: float) -> float:
    """Daily light integral of a PPFD held for the given hours."""
    return ppfd_umol_m2_s * full_output_hours_per_day * SECONDS_PER_HOUR / UMOL_PER_MOL


def absorbed_umol_per_j(absorbed_mol_day: float, electricity_kwh_day: float) -> float:
    """Absorbed photons per electrical joule; unavailable when no energy was used."""
    if not electricity_kwh_day > 0.0:
        return _NAN
    return absorbed_mol_day * UMOL_PER_MOL / (electricity_kwh_day * JOULES_PER_KWH)


def _ratio(numerator: float, denominator: float) -> float:
    if not denominator > 0.0:
        return _NAN
    return numerator / denominator


def coefficient_of_variation(values: list[float]) -> float:
    """Population std / mean. Unavailable with fewer than two values or a zero mean."""
    if len(values) < 2:
        return _NAN
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    return _ratio(float(array.std()), mean)


def _optional(value: float | None) -> float:
    return _NAN if value is None else value


# --------------------------------------------------------------------------------------
# Budget of one scenario x variant
# --------------------------------------------------------------------------------------


def variant_budget(
    resolved: ResolvedScene,
    surfaces: SurfaceSet,
    vertices_grow_mm: np.ndarray,
    receiver_triangles: np.ndarray,
    light: DirectLight,
    samples_per_triangle: int,
) -> VariantBudget:
    """Aggregate sample PPFD into the triangle, leaf, plant and canopy tables.

    Args:
        resolved: the scenario's fixtures and optics.
        surfaces: the geometry variant.
        vertices_grow_mm: ``(T, 3, 3)`` placed triangles of ``surfaces``.
        receiver_triangles: ``(N,)`` indices of the leaf triangles that were sampled.
        light: direct PPFD at the ``N * samples_per_triangle`` samples, in triangle order.
        samples_per_triangle: equal-area samples per triangle.
    """
    n_fixtures = len(light.fixture_ids)
    n_receivers = receiver_triangles.size
    shape = (n_fixtures, n_receivers, samples_per_triangle)
    per_fixture = light.ppfd_umol_m2_s.reshape(shape).mean(axis=2)  # (F, N)
    ppfd = per_fixture.sum(axis=0)
    unoccluded = light.unoccluded_ppfd_umol_m2_s.reshape(shape).mean(axis=2).sum(axis=0)
    upper = light.upper_side_ppfd_umol_m2_s.reshape(shape).mean(axis=2).sum(axis=0)
    hours = np.array([state.full_output_hours_per_day for state in resolved.fixtures])
    dli = dli_mol_m2_day((per_fixture * hours.reshape(-1, 1)).sum(axis=0), 1.0)

    vertices = vertices_grow_mm[receiver_triangles]
    area_mm2 = triangle_areas_mm2(vertices)
    area_m2 = area_mm2 / MM2_PER_M2
    centroid = vertices.mean(axis=1)
    absorptance = _optional(resolved.scene.leaf_optics.absorptance.value)
    normal_known = surfaces.normal_known[receiver_triangles]

    with np.errstate(invalid="ignore", divide="ignore"):
        loss = np.where(unoccluded > 0.0, 1.0 - ppfd / unoccluded, np.nan)
        upper_fraction = np.where(normal_known & (ppfd > 0.0), upper / ppfd, np.nan)

    triangle_rows = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for k, t in enumerate(receiver_triangles):
        plant_id, organ_id = surfaces.plant_ids[t], surfaces.organ_ids[t]
        groups[(plant_id, organ_id)].append(k)
        triangle_rows.append(
            {
                "scenario_id": resolved.scenario_id,
                "geometry_variant_id": surfaces.geometry_variant_id,
                "plant_id": plant_id,
                "organ_id": organ_id,
                "leaflet_id": surfaces.leaflet_ids[t],
                "triangle_id": surfaces.triangle_ids[t],
                "area_mm2": format_maybe_float(area_mm2[k]),
                "centroid_x_grow_mm": format_maybe_float(centroid[k, 0]),
                "centroid_y_grow_mm": format_maybe_float(centroid[k, 1]),
                "centroid_z_grow_mm": format_maybe_float(centroid[k, 2]),
                "ppfd_full_output_umol_m2_s": format_maybe_float(ppfd[k]),
                "ppfd_unoccluded_full_output_umol_m2_s": format_maybe_float(unoccluded[k]),
                "occlusion_loss_fraction": format_maybe_float(loss[k]),
                "adaxial_incident_fraction": format_maybe_float(upper_fraction[k]),
                "dli_mol_m2_day": format_maybe_float(dli[k]),
            }
        )

    leaves: list[LeafResult] = []
    for (plant_id, organ_id), members in sorted(groups.items()):
        index = np.asarray(members)
        leaf_area_m2 = float(area_m2[index].sum())
        intercepted = float((ppfd[index] * area_m2[index]).sum())
        intercepted_day = float((dli[index] * area_m2[index]).sum())
        shaded_total = intercepted
        unshaded_total = float((unoccluded[index] * area_m2[index]).sum())
        upper_total = float((upper[index] * area_m2[index]).sum())
        leaves.append(
            LeafResult(
                plant_id=plant_id,
                organ_id=organ_id,
                n_triangles=index.size,
                area_mm2=float(area_mm2[index].sum()),
                ppfd_mean_umol_m2_s=_ratio(intercepted, leaf_area_m2),
                ppfd_min_umol_m2_s=float(ppfd[index].min()),
                ppfd_max_umol_m2_s=float(ppfd[index].max()),
                occlusion_loss_fraction=(
                    1.0 - shaded_total / unshaded_total if unshaded_total > 0.0 else _NAN
                ),
                upper_side_fraction=(
                    _ratio(upper_total, shaded_total) if bool(normal_known[index].all()) else _NAN
                ),
                dli_mol_m2_day=_ratio(intercepted_day, leaf_area_m2),
                intercepted_umol_s=intercepted,
                absorbed_umol_s=intercepted * absorptance,
                intercepted_mol_day=intercepted_day,
                absorbed_mol_day=intercepted_day * absorptance,
            )
        )

    plants: list[PlantResult] = []
    for plant_id in sorted({leaf.plant_id for leaf in leaves}):
        own = [leaf for leaf in leaves if leaf.plant_id == plant_id]
        plants.append(
            PlantResult(
                plant_id=plant_id,
                n_leaves=len(own),
                leaf_area_mm2=sum(leaf.area_mm2 for leaf in own),
                intercepted_umol_s=sum(leaf.intercepted_umol_s for leaf in own),
                absorbed_umol_s=sum(leaf.absorbed_umol_s for leaf in own),
                intercepted_mol_day=sum(leaf.intercepted_mol_day for leaf in own),
                absorbed_mol_day=sum(leaf.absorbed_mol_day for leaf in own),
                leaf_ppfd_cv=coefficient_of_variation([leaf.ppfd_mean_umol_m2_s for leaf in own]),
            )
        )

    canopy = canopy_result(resolved, surfaces, leaves)
    return VariantBudget(
        scenario_id=resolved.scenario_id,
        geometry_variant_id=surfaces.geometry_variant_id,
        triangle_rows=triangle_rows,
        leaves=leaves,
        plants=plants,
        canopy=canopy,
    )


def canopy_result(
    resolved: ResolvedScene, surfaces: SurfaceSet, leaves: list[LeafResult]
) -> CanopyResult:
    """Scenario totals: emitted, intercepted and absorbed photons, electricity."""
    emitted = sum(
        _optional(state.photon_flux_umol_s) * state.full_output_hours_per_day
        for state in resolved.fixtures
    ) * SECONDS_PER_HOUR / UMOL_PER_MOL
    electricity = sum(_optional(state.electricity_kwh_day) for state in resolved.fixtures)
    leaf_area_mm2 = sum(leaf.area_mm2 for leaf in leaves)
    intercepted_day = sum(leaf.intercepted_mol_day for leaf in leaves)
    absorbed_day = sum(leaf.absorbed_mol_day for leaf in leaves)
    intercepted_now = sum(leaf.intercepted_umol_s for leaf in leaves)
    leaf_means = [leaf.ppfd_mean_umol_m2_s for leaf in leaves]
    return CanopyResult(
        n_plants=len(surfaces.plant_id_set),
        n_leaves=len(leaves),
        leaf_area_mm2=leaf_area_mm2,
        emitted_mol_day=emitted,
        intercepted_mol_day=intercepted_day,
        absorbed_mol_day=absorbed_day,
        intercepted_fraction_of_emitted=_ratio(intercepted_day, emitted),
        absorbed_fraction_of_emitted=_ratio(absorbed_day, emitted),
        electricity_kwh_day=electricity,
        absorbed_umol_per_j=absorbed_umol_per_j(absorbed_day, electricity),
        leaf_ppfd_mean_umol_m2_s=_ratio(intercepted_now, leaf_area_mm2 / MM2_PER_M2),
        leaf_ppfd_min_umol_m2_s=min(leaf_means) if leaf_means else _NAN,
        leaf_ppfd_cv=coefficient_of_variation(leaf_means),
    )


# --------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------


def leaf_rows(budget: VariantBudget) -> list[dict[str, str]]:
    return [
        {
            "scenario_id": budget.scenario_id,
            "geometry_variant_id": budget.geometry_variant_id,
            "plant_id": leaf.plant_id,
            "organ_id": leaf.organ_id,
            "n_triangles": str(leaf.n_triangles),
            "area_mm2": format_maybe_float(leaf.area_mm2),
            "ppfd_mean_full_output_umol_m2_s": format_maybe_float(leaf.ppfd_mean_umol_m2_s),
            "ppfd_min_full_output_umol_m2_s": format_maybe_float(leaf.ppfd_min_umol_m2_s),
            "ppfd_max_full_output_umol_m2_s": format_maybe_float(leaf.ppfd_max_umol_m2_s),
            "occlusion_loss_fraction": format_maybe_float(leaf.occlusion_loss_fraction),
            "adaxial_incident_fraction": format_maybe_float(leaf.upper_side_fraction),
            "dli_mol_m2_day": format_maybe_float(leaf.dli_mol_m2_day),
            "intercepted_full_output_umol_s": format_maybe_float(leaf.intercepted_umol_s),
            "absorbed_full_output_umol_s": format_maybe_float(leaf.absorbed_umol_s),
            "absorbed_mol_day": format_maybe_float(leaf.absorbed_mol_day),
        }
        for leaf in budget.leaves
    ]


def plant_rows(budget: VariantBudget) -> list[dict[str, str]]:
    return [
        {
            "scenario_id": budget.scenario_id,
            "geometry_variant_id": budget.geometry_variant_id,
            "plant_id": plant.plant_id,
            "n_leaves": str(plant.n_leaves),
            "leaf_area_mm2": format_maybe_float(plant.leaf_area_mm2),
            "intercepted_full_output_umol_s": format_maybe_float(plant.intercepted_umol_s),
            "absorbed_full_output_umol_s": format_maybe_float(plant.absorbed_umol_s),
            "intercepted_mol_day": format_maybe_float(plant.intercepted_mol_day),
            "absorbed_mol_day": format_maybe_float(plant.absorbed_mol_day),
            "leaf_ppfd_cv": format_maybe_float(plant.leaf_ppfd_cv),
        }
        for plant in budget.plants
    ]


def scenario_rows(budgets: list[VariantBudget]) -> list[dict[str, str]]:
    """Headline numbers per scenario x variant, with the change against ``as_built``."""
    as_built = {
        budget.geometry_variant_id: budget.canopy
        for budget in budgets
        if budget.scenario_id == AS_BUILT_SCENARIO_ID
    }
    rows = []
    for budget in budgets:
        canopy = budget.canopy
        base = as_built.get(budget.geometry_variant_id)

        def delta(name: str) -> str:
            if base is None:
                return UNAVAILABLE
            return format_maybe_float(getattr(canopy, name) - getattr(base, name))

        rows.append(
            {
                "scenario_id": budget.scenario_id,
                "geometry_variant_id": budget.geometry_variant_id,
                "n_plants": str(canopy.n_plants),
                "n_leaves": str(canopy.n_leaves),
                "leaf_area_mm2": format_maybe_float(canopy.leaf_area_mm2),
                "emitted_mol_day": format_maybe_float(canopy.emitted_mol_day),
                "intercepted_mol_day": format_maybe_float(canopy.intercepted_mol_day),
                "absorbed_mol_day": format_maybe_float(canopy.absorbed_mol_day),
                "intercepted_fraction_of_emitted": format_maybe_float(
                    canopy.intercepted_fraction_of_emitted
                ),
                "absorbed_fraction_of_emitted": format_maybe_float(
                    canopy.absorbed_fraction_of_emitted
                ),
                "electricity_kwh_day": format_maybe_float(canopy.electricity_kwh_day),
                "absorbed_umol_per_j": format_maybe_float(canopy.absorbed_umol_per_j),
                "leaf_ppfd_mean_umol_m2_s": format_maybe_float(canopy.leaf_ppfd_mean_umol_m2_s),
                "leaf_ppfd_min_umol_m2_s": format_maybe_float(canopy.leaf_ppfd_min_umol_m2_s),
                "leaf_ppfd_cv": format_maybe_float(canopy.leaf_ppfd_cv),
                "delta_absorbed_mol_day_vs_as_built": delta("absorbed_mol_day"),
                "delta_electricity_kwh_day_vs_as_built": delta("electricity_kwh_day"),
                "delta_absorbed_umol_per_j_vs_as_built": delta("absorbed_umol_per_j"),
            }
        )
    return rows
