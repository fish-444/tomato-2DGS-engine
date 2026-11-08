"""Geometry variants against a reference variant.

This is where the question "how few photographs still predict the same light" is read.
The reference is usually the full multi-view reconstruction; a candidate is the same
plants rebuilt from fewer views, or simplified. For each plant the comparison reports
leaf area and absorbed photons of both, and the per-leaf PPFD difference over leaves
present in both.

Leaves are matched by ``organ_id`` within a plant. A variant that numbers leaves
differently matches nothing, and the unmatched counts say so - leaves are never paired
by guessing.

Agreement with the reference is not accuracy. Whether either variant predicts real light
is answered only by the evaluation readings in ``validation.csv``.
"""

from __future__ import annotations

import numpy as np

from tomato3d.io.schemas import NO_PLANTS_VARIANT_ID, UNAVAILABLE, format_maybe_float
from tomato3d.light.budget import LeafResult, PlantResult, VariantBudget

_NAN = float("nan")


def _leaf_ppfd_differences(
    reference: list[LeafResult], candidate: list[LeafResult]
) -> tuple[int, int, int, float, float]:
    ref_by_id = {leaf.organ_id: leaf for leaf in reference}
    cand_by_id = {leaf.organ_id: leaf for leaf in candidate}
    matched = sorted(set(ref_by_id) & set(cand_by_id))
    differences = np.array(
        [
            cand_by_id[organ].ppfd_mean_umol_m2_s - ref_by_id[organ].ppfd_mean_umol_m2_s
            for organ in matched
        ]
    )
    mae = float(np.abs(differences).mean()) if matched else _NAN
    bias = float(differences.mean()) if matched else _NAN
    return (
        len(matched),
        len(set(ref_by_id) - set(cand_by_id)),
        len(set(cand_by_id) - set(ref_by_id)),
        mae,
        bias,
    )


def _plant(budget: VariantBudget, plant_id: str) -> PlantResult | None:
    for plant in budget.plants:
        if plant.plant_id == plant_id:
            return plant
    return None


def geometry_variant_rows(
    budgets: list[VariantBudget], reference_variant_id: str
) -> list[dict[str, str]]:
    """One row per scenario x plant x candidate variant, against the reference variant."""
    rows = []
    by_key = {(b.scenario_id, b.geometry_variant_id): b for b in budgets}
    scenario_ids = list(dict.fromkeys(b.scenario_id for b in budgets))
    for scenario_id in scenario_ids:
        reference = by_key.get((scenario_id, reference_variant_id))
        if reference is None:
            continue
        for candidate in budgets:
            if (
                candidate.scenario_id != scenario_id
                or candidate.geometry_variant_id in (reference_variant_id, NO_PLANTS_VARIANT_ID)
            ):
                continue
            plant_ids = sorted(
                {p.plant_id for p in reference.plants} | {p.plant_id for p in candidate.plants}
            )
            for plant_id in plant_ids:
                ref_plant = _plant(reference, plant_id)
                cand_plant = _plant(candidate, plant_id)
                ref_leaves = [leaf for leaf in reference.leaves if leaf.plant_id == plant_id]
                cand_leaves = [leaf for leaf in candidate.leaves if leaf.plant_id == plant_id]
                matched, ref_only, cand_only, mae, bias = _leaf_ppfd_differences(
                    ref_leaves, cand_leaves
                )
                ref_absorbed = _NAN if ref_plant is None else ref_plant.absorbed_mol_day
                cand_absorbed = _NAN if cand_plant is None else cand_plant.absorbed_mol_day
                relative = (
                    (cand_absorbed - ref_absorbed) / ref_absorbed if ref_absorbed > 0.0 else _NAN
                )
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "plant_id": plant_id,
                        "reference_variant_id": reference_variant_id,
                        "candidate_variant_id": candidate.geometry_variant_id,
                        "reference_leaf_area_mm2": (
                            UNAVAILABLE
                            if ref_plant is None
                            else format_maybe_float(ref_plant.leaf_area_mm2)
                        ),
                        "candidate_leaf_area_mm2": (
                            UNAVAILABLE
                            if cand_plant is None
                            else format_maybe_float(cand_plant.leaf_area_mm2)
                        ),
                        "reference_absorbed_mol_day": format_maybe_float(ref_absorbed),
                        "candidate_absorbed_mol_day": format_maybe_float(cand_absorbed),
                        "absorbed_relative_difference": format_maybe_float(relative),
                        "n_leaves_matched": str(matched),
                        "n_leaves_reference_only": str(ref_only),
                        "n_leaves_candidate_only": str(cand_only),
                        "leaf_ppfd_mae_matched_umol_m2_s": format_maybe_float(mae),
                        "leaf_ppfd_bias_matched_umol_m2_s": format_maybe_float(bias),
                    }
                )
    return rows
