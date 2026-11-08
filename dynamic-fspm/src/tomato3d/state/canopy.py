"""P1.8 - slot_id -> one rigid transform per plant instance (plant_local_mm -> canopy_mm).

Layout source (read-only): ``data/records/env_light_20261010/`` - 3 trays + grid column 4
separate, col 1 = right end by the post, row 1 = wall (rule v3, user_stated / user_confirmed).
Tray pitch, pot pitch, row pitch and soil heights were never measured, so the positions use the
``assumed_unmeasured`` values of ``p1_config.json`` and say so in ``canopy_placement.json``.

Transform: ``x_canopy = Rz(yaw_deg) @ x_plant_local + translation_mm`` (pots stand upright; yaw is
the plant-local +X direction in the canopy frame). Real plants: ``plant_id <-> slot`` is empty in
the plant registry and no pot orientation is recorded, so slot and yaw are ``unavailable`` and
real placement is blocked. Fixtures declare both.

Other-variety plants: plant-level only (MTG A1.3); omitted from the optical scene with a flag
unless a conditional proxy is explicitly requested (none is built here).

Duplicate source plants are distinct instances (``<source>#<n>``); organ ids stay the source's MTG
ids and become canopy-unique through :func:`tomato3d.state.schema.external_leaflet_id`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.state import schema as s
from tomato3d.state.topraster import rotate_z

SLOT_TABLE = Path("data/records/env_light_20261010/plant_slot_light_ratio_20261010.csv")
RATIO_RULE = "plant_slot_light_ratio_20261010.csv ratio_now (rule v3: grid row 1 = wall side)"


class CanopyError(ValueError):
    """Placement request violates the contract (unknown slot, slot used twice...)."""


def load_slots(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["slot_id"]: r for r in rows if not r["slot_id"].endswith("*")}


def slot_position_mm(slot: Mapping[str, str], cfg: Mapping[str, Any]) -> np.ndarray:
    ca = cfg["canopy_assumed"]
    tray_pitch = float(ca["tray_pitch_mm"]["value"])
    pot_pitch = float(ca["pot_pitch_mm"]["value"])
    row_pitch = float(ca["row_pitch_mm"]["value"])
    z = float(ca["soil_level_z_mm"]["value"])
    sign = cfg["canopy_frame"]["side_from_front_offset_sign"][slot["side_from_front"]]
    col = int(slot["grid_col"])
    row = int(slot["pot_row_from_back"])
    return np.array([(col - 1) * tray_pitch + sign * pot_pitch / 2.0, (row - 1) * row_pitch, z], float)


@dataclass(frozen=True)
class Placement:
    plant_instance_id: str
    source_plant_id: str
    slot_id: str | None
    yaw_deg: float | None
    translation_mm: np.ndarray | None
    variety: str
    light_ratio: str
    in_optical_scene: bool
    reason: str

    @property
    def placed(self) -> bool:
        return self.translation_mm is not None and self.yaw_deg is not None

    def matrix(self) -> np.ndarray:
        if not self.placed:
            raise CanopyError(f"{self.plant_instance_id}: not placed ({self.reason})")
        a = np.radians(self.yaw_deg)  # type: ignore[arg-type]
        m = np.eye(4)
        m[:3, :3] = [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
        m[:3, 3] = self.translation_mm  # type: ignore[assignment]
        return m

    def to_canopy(self, xyz_mm: np.ndarray) -> np.ndarray:
        if not self.placed:
            raise CanopyError(f"{self.plant_instance_id}: not placed ({self.reason})")
        return rotate_z(xyz_mm, self.yaw_deg) + self.translation_mm  # type: ignore[arg-type,operator]

    def to_plant(self, xyz_canopy_mm: np.ndarray) -> np.ndarray:
        if not self.placed:
            raise CanopyError(f"{self.plant_instance_id}: not placed ({self.reason})")
        return rotate_z(np.asarray(xyz_canopy_mm, float) - self.translation_mm, -self.yaw_deg)  # type: ignore[operator]


@dataclass(frozen=True)
class InstanceRequest:
    plant_instance_id: str
    slot_id: str | None
    yaw_deg: float | None
    variety: str  # micro_tom | other | unavailable
    slot_source: str  # user_stated | fixture_declared | unavailable
    yaw_source: str


def place(requests: Sequence[InstanceRequest], slots: Mapping[str, Mapping[str, str]],
          cfg: Mapping[str, Any]) -> list[Placement]:
    used: dict[str, str] = {}
    out = []
    for req in requests:
        source = s.source_of_instance(req.plant_instance_id)
        if req.slot_id is None:
            out.append(Placement(req.plant_instance_id, source, None, None, None, req.variety, UNAVAILABLE, False,
                                 "slot_id unavailable: plant_id <-> slot table is empty (plant registry)"))
            continue
        if req.slot_id not in slots:
            raise CanopyError(f"unknown slot {req.slot_id!r}")
        if req.slot_id in used:
            raise CanopyError(f"slot {req.slot_id} used by {used[req.slot_id]} and {req.plant_instance_id}")
        used[req.slot_id] = req.plant_instance_id
        slot = slots[req.slot_id]
        slot_variety = "micro_tom" if slot["variety"] == "dwarf" else "other"
        if req.variety not in (slot_variety, UNAVAILABLE):
            raise CanopyError(f"{req.plant_instance_id}: variety {req.variety} in a {slot_variety} slot")
        optical = slot_variety == "micro_tom"
        reason = "placed" if optical else "other-variety plant: plant-level only, omitted from the optical scene"
        if req.yaw_deg is None:
            out.append(Placement(req.plant_instance_id, source, req.slot_id, None, None, slot_variety,
                                 slot["ratio_now"], False, "yaw unavailable: no pot orientation recorded"))
            continue
        out.append(Placement(req.plant_instance_id, source, req.slot_id, float(req.yaw_deg),
                             slot_position_mm(slot, cfg), slot_variety, slot["ratio_now"], optical, reason))
    return out


def placement_document(placements: Sequence[Placement], requests: Sequence[InstanceRequest], cfg: Mapping[str, Any],
                       slot_table_sha256: str, evidence: str) -> dict[str, Any]:
    by_req = {r.plant_instance_id: r for r in requests}
    sources: dict[str, list[str]] = {}
    for p in placements:
        sources.setdefault(p.source_plant_id, []).append(p.plant_instance_id)
    instances = []
    for p in placements:
        req = by_req[p.plant_instance_id]
        instances.append(dict(
            plant_instance_id=p.plant_instance_id, source_plant_id=p.source_plant_id,
            duplicate_of_source=len(sources[p.source_plant_id]) > 1,
            sibling_instances=[i for i in sources[p.source_plant_id] if i != p.plant_instance_id],
            slot_id=p.slot_id or UNAVAILABLE, slot_id_source=req.slot_source if p.slot_id else UNAVAILABLE,
            variety=p.variety, light_ratio=p.light_ratio,
            light_ratio_status="app_ratio" if p.light_ratio != UNAVAILABLE else UNAVAILABLE,
            light_ratio_source=RATIO_RULE if p.light_ratio != UNAVAILABLE else UNAVAILABLE,
            transform=dict(
                from_frame="plant_local_mm", to_frame="canopy_mm",
                yaw_deg=p.yaw_deg if p.yaw_deg is not None else UNAVAILABLE,
                yaw_source=req.yaw_source if p.yaw_deg is not None else UNAVAILABLE,
                translation_mm=p.translation_mm.tolist() if p.translation_mm is not None else UNAVAILABLE,
                matrix_4x4=p.matrix().tolist() if p.placed else UNAVAILABLE,
                translation_source="assumed_unmeasured slot geometry (p1_config.canopy_assumed)" if p.placed
                else UNAVAILABLE),
            # V3 plants.csv field names, so P2 reads the same numbers without a conversion
            p2_plants_csv=dict(tx_mm=float(p.translation_mm[0]), ty_mm=float(p.translation_mm[1]),
                               tz_mm=float(p.translation_mm[2]), yaw_deg=p.yaw_deg,
                               transform_source=f"p1_canopy:{req.slot_source}+assumed_unmeasured_geometry")
            if p.placed else UNAVAILABLE,
            in_optical_scene=p.in_optical_scene, placement_status="placed" if p.placed else "unavailable",
            reason=p.reason))
    return dict(
        schema=s.P1_SCHEMA_VERSION, frame="canopy_mm", evidence=evidence,
        canopy_frame_definition=cfg["canopy_frame"]["definition"],
        tray_geometry={k: dict(value=v["value"], unit=v["unit"], source_kind=v["source_kind"], source_ref=v["source_ref"])
                       for k, v in cfg["canopy_assumed"].items()},
        slot_table=dict(path=SLOT_TABLE.as_posix(), sha256=slot_table_sha256),
        duplicate_instance_rule="same source plant placed twice = two instances; organ ids stay source MTG ids and "
                                "become canopy-unique as <plant_instance_id>|<vertex_id>",
        instances=instances)


def merge_triangles(placements: Mapping[str, Placement], rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Optical triangles (plant_local) -> canopy frame, keeping the external leaflet id."""
    out = []
    for r in rows:
        inst, _ = s.split_external_id(r["external_leaflet_id"])
        p = placements[inst]
        if not p.in_optical_scene:
            continue
        pts = np.array([[float(r[f"v{k}_{a}_mm"]) for a in "xyz"] for k in range(3)])
        q = p.to_canopy(pts)
        row = dict(r)
        row["frame"] = "canopy_mm"
        for k in range(3):
            for j, a in enumerate("xyz"):
                row[f"v{k}_{a}_mm"] = s.fmt(q[k, j], 4)
        out.append(row)
    return out


def split_triangles(placements: Mapping[str, Placement], rows: Sequence[Mapping[str, str]]
                    ) -> dict[str, list[dict[str, str]]]:
    """Inverse of :func:`merge_triangles`: canopy rows back to per-instance plant-local rows."""
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        inst, _ = s.split_external_id(r["external_leaflet_id"])
        pts = np.array([[float(r[f"v{k}_{a}_mm"]) for a in "xyz"] for k in range(3)])
        q = placements[inst].to_plant(pts)
        row = dict(r)
        row["frame"] = "plant_local_mm"
        for k in range(3):
            for j, a in enumerate("xyz"):
                row[f"v{k}_{a}_mm"] = s.fmt(q[k, j], 4)
        out.setdefault(inst, []).append(row)
    return out
