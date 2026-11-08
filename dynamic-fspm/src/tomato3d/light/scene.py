"""The lighting environment: loading, scenario resolution, emitters, schedules, energy.

A :class:`~tomato3d.io.schemas.LightScene` holds every environment variable of one
growing setup. A run evaluates the as-built arrangement and each scenario; a scenario is
the as-built scene with some values replaced, and every replaced value carries the status
``scenario`` so no hypothetical number can pass for a measured one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import BaseModel

from tomato3d.io.schemas import (
    AS_BUILT_SCENARIO_ID,
    NON_OBSERVED_INPUT_STATUSES,
    UNAVAILABLE,
    LedFixture,
    LightInputRecord,
    LightScenario,
    LightScene,
    PlantPlacement,
    ScalarInput,
    SchemaError,
    VectorInput,
    load_json_model,
)
from tomato3d.light.frames import fixture_axes
from tomato3d.light.units import WATT_HOURS_PER_KWH

#: Intensity ratio that defines the half-intensity half-angle.
_HALF_INTENSITY_RATIO: Final[float] = 0.5
_SCENARIO_STATUS: Final[str] = "scenario"


class LightInputError(SchemaError):
    """Light inputs that are individually valid but inconsistent with each other."""


@dataclass(frozen=True)
class Emitters:
    """The point emitters of one fixture, in the grow frame.

    Every point emits ``I(theta) = axial_intensity_umol_s_sr * cos(theta)**cosine_exponent``
    for ``cos(theta) >= cos_cutoff``, ``theta`` measured from ``aim``.
    """

    fixture_id: str
    positions_grow_mm: np.ndarray  # (E, 3)
    aim: np.ndarray  # (3,)
    axial_intensity_umol_s_sr: float
    cosine_exponent: float
    cos_cutoff: float


@dataclass(frozen=True)
class FixtureState:
    """One fixture as a scenario sets it up, with its schedule and energy resolved."""

    fixture: LedFixture
    #: ``None`` when an input the light field needs is unavailable.
    emitters: Emitters | None
    #: Names of the unavailable inputs that left ``emitters`` empty.
    missing_inputs: tuple[str, ...]
    #: Sum over schedule segments of ``output_fraction * hours``.
    full_output_hours_per_day: float
    #: ``None`` when a segment's power is unknown; ``electricity_missing_reason`` says why.
    electricity_kwh_day: float | None
    electricity_missing_reason: str | None

    @property
    def fixture_id(self) -> str:
        return self.fixture.fixture_id

    @property
    def photon_flux_umol_s(self) -> float | None:
        return self.fixture.photon_flux_umol_s.value


@dataclass(frozen=True)
class ResolvedScene:
    """The environment of one scenario: enabled fixtures and plant placements."""

    scene: LightScene
    scenario_id: str
    fixtures: tuple[FixtureState, ...]
    placements: dict[str, PlantPlacement]


def load_scene(path: Path) -> LightScene:
    """Read and validate a light scene JSON.

    Raises:
        SchemaError: the file is missing, is not JSON, or fails validation.
    """
    return load_json_model(path, LightScene)


# --------------------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------------------


def cosine_exponent(half_intensity_half_angle_rad: float) -> float:
    """``m`` of ``cos(theta)**m`` whose intensity halves at the given half-angle."""
    return math.log(_HALF_INTENSITY_RATIO) / math.log(math.cos(half_intensity_half_angle_rad))


def axial_intensity_umol_s_sr(
    photon_flux_umol_s: float, exponent: float, cutoff_half_angle_rad: float
) -> float:
    """``I0`` such that ``I0 * cos(theta)**m`` inside the cutoff cone emits the photon flux.

    Integrating over the cone gives ``flux = 2 pi I0 (1 - cos(theta_c)**(m+1)) / (m+1)``.
    """
    cone_fraction = 1.0 - math.cos(cutoff_half_angle_rad) ** (exponent + 1.0)
    return photon_flux_umol_s * (exponent + 1.0) / (2.0 * math.pi * cone_fraction)


def build_emitters(fixture: LedFixture) -> tuple[Emitters | None, tuple[str, ...]]:
    """Discretise a fixture into point emitters, or name the inputs that prevent it."""
    required: dict[str, ScalarInput | VectorInput] = {
        "emitting_center_grow_mm": fixture.emitting_center_grow_mm,
        "aim_tilt_deg": fixture.aim_tilt_deg,
        "aim_azimuth_deg": fixture.aim_azimuth_deg,
        "long_axis_azimuth_deg": fixture.long_axis_azimuth_deg,
        "emitting_length_mm": fixture.emitting_length_mm,
        "emitting_width_mm": fixture.emitting_width_mm,
        "photon_flux_umol_s": fixture.photon_flux_umol_s,
        "distribution.half_intensity_half_angle_deg": (
            fixture.distribution.half_intensity_half_angle_deg
        ),
    }
    missing = tuple(name for name, item in required.items() if not item.is_available)
    if missing:
        return None, missing

    aim, long_axis, width_axis = fixture_axes(
        math.radians(_value(fixture.aim_tilt_deg)),
        math.radians(_value(fixture.aim_azimuth_deg)),
        math.radians(_value(fixture.long_axis_azimuth_deg)),
    )
    n_long, n_wide = fixture.emitter_grid
    long_offsets_mm = ((np.arange(n_long) + 0.5) / n_long - 0.5) * _value(
        fixture.emitting_length_mm
    )
    wide_offsets_mm = ((np.arange(n_wide) + 0.5) / n_wide - 0.5) * _value(
        fixture.emitting_width_mm
    )
    center_mm = np.asarray(fixture.emitting_center_grow_mm.value, dtype=np.float64)
    positions_mm = (
        center_mm[None, None, :]
        + long_offsets_mm[:, None, None] * long_axis[None, None, :]
        + wide_offsets_mm[None, :, None] * width_axis[None, None, :]
    ).reshape(-1, 3)

    half_angle_rad = math.radians(_value(fixture.distribution.half_intensity_half_angle_deg))
    cutoff_rad = math.radians(fixture.distribution.cutoff_half_angle_deg)
    exponent = cosine_exponent(half_angle_rad)
    per_point_flux = _value(fixture.photon_flux_umol_s) / positions_mm.shape[0]
    return (
        Emitters(
            fixture_id=fixture.fixture_id,
            positions_grow_mm=positions_mm,
            aim=aim,
            axial_intensity_umol_s_sr=axial_intensity_umol_s_sr(
                per_point_flux, exponent, cutoff_rad
            ),
            cosine_exponent=exponent,
            cos_cutoff=math.cos(cutoff_rad),
        ),
        (),
    )


# --------------------------------------------------------------------------------------
# Schedule and electricity
# --------------------------------------------------------------------------------------


def full_output_hours_per_day(fixture: LedFixture) -> float:
    """Hours at full output that deliver the same photons as the schedule."""
    return sum(segment.output_fraction * segment.duration_hours for segment in fixture.schedule)


def electricity_kwh_day(fixture: LedFixture) -> tuple[float | None, str | None]:
    """Daily electrical energy of one fixture, or ``None`` with the reason it is unknown.

    A segment uses its own ``electrical_input_w`` when given; otherwise the fixture's
    full-output power, but only at ``output_fraction == 1``. A dimmed segment's power is
    never scaled linearly.
    """
    watt_hours = 0.0
    for segment in fixture.schedule:
        label = f"{fixture.fixture_id} {segment.start_hhmm}-{segment.end_hhmm}"
        if segment.electrical_input_w is not None:
            power = segment.electrical_input_w.value
            if power is None:
                return None, f"{label}: electrical_input_w is unavailable"
        elif segment.output_fraction == 1.0:
            power = fixture.electrical_input_w.value
            if power is None:
                return None, f"{fixture.fixture_id}: electrical_input_w is unavailable"
        else:
            return None, (
                f"{label}: dimmed to {segment.output_fraction} with no measured "
                "electrical_input_w for the segment"
            )
        watt_hours += power * segment.duration_hours
    return watt_hours / WATT_HOURS_PER_KWH, None


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------


def _value(item: ScalarInput) -> float:
    if item.value is None:  # pragma: no cover - callers check availability first
        raise LightInputError(f"required input is unavailable ({item.source})")
    return item.value


def _scenario_scalar(value: float, source: str) -> ScalarInput:
    return ScalarInput(value=value, status=_SCENARIO_STATUS, source=source)


def _scenario_vector(value: tuple[float, float, float], source: str) -> VectorInput:
    return VectorInput(value=value, status=_SCENARIO_STATUS, source=source)


def _find_scenario(scene: LightScene, scenario_id: str) -> LightScenario | None:
    if scenario_id == AS_BUILT_SCENARIO_ID:
        return None
    for scenario in scene.scenarios:
        if scenario.scenario_id == scenario_id:
            return scenario
    raise LightInputError(
        f"scenario {scenario_id!r} is not defined in setup {scene.setup_id!r}; "
        f"known: {scene.scenario_ids}"
    )


def _fixture_state(fixture: LedFixture) -> FixtureState:
    emitters, missing = build_emitters(fixture)
    kwh, reason = electricity_kwh_day(fixture)
    return FixtureState(
        fixture=fixture,
        emitters=emitters,
        missing_inputs=missing,
        full_output_hours_per_day=full_output_hours_per_day(fixture),
        electricity_kwh_day=kwh,
        electricity_missing_reason=reason,
    )


def resolve_scenario(scene: LightScene, scenario_id: str) -> ResolvedScene:
    """Apply one scenario's overrides to the as-built scene.

    Raises:
        LightInputError: the scenario is unknown, or it replaces a fixture's power while
            that fixture's schedule carries per-segment power (which would then describe
            the old lamp).
    """
    scenario = _find_scenario(scene, scenario_id)
    fixture_overrides = {} if scenario is None else {
        override.fixture_id: override for override in scenario.fixture_overrides
    }
    placement_overrides = {} if scenario is None else {
        override.plant_id: override for override in scenario.placement_overrides
    }
    source = "" if scenario is None else f"scenario {scenario.scenario_id}: {scenario.description}"

    fixtures: list[FixtureState] = []
    for fixture in scene.fixtures:
        override = fixture_overrides.get(fixture.fixture_id)
        if override is None:
            fixtures.append(_fixture_state(fixture))
            continue
        if not override.enabled:
            continue
        update: dict[str, object] = {}
        if override.emitting_center_grow_mm is not None:
            update["emitting_center_grow_mm"] = _scenario_vector(
                override.emitting_center_grow_mm, source
            )
        for name in (
            "aim_tilt_deg",
            "aim_azimuth_deg",
            "long_axis_azimuth_deg",
            "photon_flux_umol_s",
            "electrical_input_w",
        ):
            value = getattr(override, name)
            if value is not None:
                update[name] = _scenario_scalar(value, source)
        if "electrical_input_w" in update and any(
            segment.electrical_input_w is not None for segment in fixture.schedule
        ):
            raise LightInputError(
                f"scenario {scenario_id!r} replaces the power of {fixture.fixture_id!r}, "
                "whose schedule carries per-segment power for the as-built lamp"
            )
        fixtures.append(_fixture_state(fixture.model_copy(update=update)))

    placements: dict[str, PlantPlacement] = {}
    for placement in scene.plants:
        override = placement_overrides.get(placement.plant_id)
        if override is None:
            placements[placement.plant_id] = placement
            continue
        update = {}
        if override.board_origin_grow_mm is not None:
            update["board_origin_grow_mm"] = _scenario_vector(
                override.board_origin_grow_mm, source
            )
        if override.board_yaw_deg is not None:
            update["board_yaw_deg"] = _scenario_scalar(override.board_yaw_deg, source)
        placements[placement.plant_id] = placement.model_copy(update=update)

    return ResolvedScene(
        scene=scene,
        scenario_id=scenario_id,
        fixtures=tuple(fixtures),
        placements=placements,
    )


# --------------------------------------------------------------------------------------
# Provenance inventory
# --------------------------------------------------------------------------------------


def _item_key(item: object, index: int) -> str:
    for attribute in ("fixture_id", "plant_id", "scenario_id"):
        key = getattr(item, attribute, None)
        if isinstance(key, str):
            return key
    return str(index)


def input_records(scene: LightScene) -> list[LightInputRecord]:
    """Every environment variable of the as-built scene with its value and provenance."""
    records: list[LightInputRecord] = []

    def visit(name: str, node: object) -> None:
        if isinstance(node, ScalarInput):
            text = UNAVAILABLE if node.value is None else repr(node.value)
            records.append(
                LightInputRecord(name=name, value=text, status=node.status, source=node.source)
            )
        elif isinstance(node, VectorInput):
            text = UNAVAILABLE if node.value is None else json.dumps(list(node.value))
            records.append(
                LightInputRecord(name=name, value=text, status=node.status, source=node.source)
            )
        elif isinstance(node, BaseModel):
            for field_name in type(node).model_fields:
                visit(f"{name}.{field_name}" if name else field_name, getattr(node, field_name))
        elif isinstance(node, list | tuple):
            for index, item in enumerate(node):
                visit(f"{name}[{_item_key(item, index)}]", item)

    visit("", scene)
    return records


def non_observed_input_names(records: list[LightInputRecord]) -> list[str]:
    """``name (status)`` of every input that is not an observation of this rig."""
    flagged = (*NON_OBSERVED_INPUT_STATUSES, UNAVAILABLE)
    return [f"{record.name} ({record.status})" for record in records if record.status in flagged]
