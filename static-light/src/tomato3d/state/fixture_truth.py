"""P1.2 - synthetic scene -> truth MTG, truth areas, truth optical surfaces, projected observations.

Input: an existing generator scene folder (``build/synthetic_dwarf_tomato_*/<cell>/scenes/scene_NNNN``
or ``build/sideprop_20261009_01/scenes/scene_NNNN``): ``plant_graph.json`` (frame
``plant_local_mm``: soil surface on the stem axis, +Z up), ``leaflet_index.json`` and the cell's
``leaflet_contours.json``. Nothing is regenerated, so L-Py is not needed.

Two products, kept apart:

* **truth** - everything the mesh determines, value status ``synthetic_truth`` (or ``derived``
  for A1.5/A2 quantities computed from truth points), row evidence ``synthetic_fixture``.
* **observation-like** - what a top camera would give: per-compound visible masks, the
  terminal leaflet by explicit membership, anonymous lateral leaflets; occluded leaflets,
  angles, 3-D positions and ranks are dropped. Its provenance is ``synthetic_gt`` and it can
  never count as photo reconstruction.

Fixture plant-local frame = the generator's plant-local frame (origin at the stem base on the soil
surface, +Z up, +X the generator's own axis). The real ``plant_local_mm`` (pot-rim plane, pot
orientation mark) is a different physical definition; fixtures never mix with real rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.state import geometry as g
from tomato3d.state import schema as s
from tomato3d.state.topraster import OCCLUSION_RULE, OrthoCamera, TopRender, render_top, rotate_z

EVIDENCE = "synthetic_fixture"
TRUTH = "synthetic_truth"
DERIVED = "derived"
#: P1 code for the Petiole vertex (MTG section 4 gives the grammar ``<owner>/<code><index>`` but
#: no petiole code; leaflet tags already use ``P<i><L|R>``, which A1.5 forbids reusing).
PETIOLE_CODE = "PT"
RACHIS_CODE = "R"
FACE_CONVENTION = "adaxial_ccw"
AREA_CONVENTION = "one_sided_lamina_full_outline"
CONFIG_PATH = Path(__file__).with_name("p1_config.json")


def load_p1_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cfg_value(cfg: dict[str, Any], section: str, name: str) -> Any:
    entry = cfg[section][name]
    if entry["value"] is None:
        raise s.P1SchemaError(f"p1_config {section}.{name} is null (inactive); the step needs it")
    return entry["value"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Scene loading
# --------------------------------------------------------------------------------------


@dataclass
class FixtureCompound:
    """One generator compound, carried with its own meshes (contours come from its own cell)."""

    gen_compound_id: str
    serial: int
    attachment_mm: np.ndarray
    rachis_points_mm: np.ndarray
    leaflets: list[dict[str, Any]]
    meshes: dict[str, tuple[np.ndarray, np.ndarray]]
    age_days: float
    gen_rank_from_bottom: int
    source_ref: str
    axis: str = "M"


@dataclass
class SideAxis:
    """A side shoot of the fixture (MTG A4): its own stem polyline, attached in the axil of a main compound."""

    axis: str
    parent_serial: int
    points_mm: np.ndarray
    source_ref: str


@dataclass
class FixturePlant:
    """A plant built from a scene, possibly modified (compound removed / added) for a sequence."""

    source_plant_id: str
    stem_points_mm: np.ndarray
    compounds: list[FixtureCompound]
    provenance: dict[str, Any] = field(default_factory=dict)
    next_serial: int = 1
    side_axes: dict[str, SideAxis] = field(default_factory=dict)

    def axis_vertex_id(self, axis: str) -> str:
        """MTG vertex of an axis: ``<plant>/A0`` for M (MTG section 4), ``<plant>/S<k>`` for side shoots (A4)."""
        return f"{self.source_plant_id}/A0" if axis == "M" else f"{self.source_plant_id}/{axis}"

    def stem_of(self, axis: str) -> np.ndarray:
        return self.stem_points_mm if axis == "M" else self.side_axes[axis].points_mm

    def compound_vertex_id(self, comp: FixtureCompound) -> str:
        return s.compound_id(self.source_plant_id, comp.serial)

    def leaflet_meshes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for comp in self.compounds:
            cid = self.compound_vertex_id(comp)
            for tag, mesh in comp.meshes.items():
                out[f"{cid}/{tag}"] = mesh
        return out


def _cell_dir(scene_dir: Path) -> Path:
    return Path(scene_dir).resolve().parent.parent


def _orient_adaxial(verts: np.ndarray, faces: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return faces wound counter-clockwise seen from the adaxial (``up``) side."""
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int32).copy()
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = n @ np.asarray(up, float) < 0
    f[flip] = f[flip][:, ::-1]
    return f


def load_plant(scene_dir: Path, source_plant_id: str) -> FixturePlant:
    """Read a generator scene into a :class:`FixturePlant` (plant-local mm, no rendering)."""
    from tomato3d.synth.contours import load_library
    from tomato3d.synth.leaflet_surface import surface_points, triangulate_contour

    scene_dir = Path(scene_dir)
    cell = _cell_dir(scene_dir)
    graph = json.loads((scene_dir / "plant_graph.json").read_text(encoding="utf-8"))
    if graph.get("frame") != "plant_local_mm":
        raise s.P1SchemaError(f"{scene_dir}: plant_graph frame {graph.get('frame')!r}, expected plant_local_mm")
    contours, _ = load_library(cell / "leaflet_contours.json")
    gen_cfg = json.loads((cell / "generator_config.json").read_text(encoding="utf-8"))
    grid_n = int(gen_cfg["constants"]["leaflet_mesh_grid_n"]["value"])
    # serials in creation order: oldest compound first (generator age_days), ties by generator id
    order = sorted(graph["compounds"], key=lambda c: (-float(c["age_days"]), c["compound_id"]))
    tri_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    compounds: list[FixtureCompound] = []
    for serial, comp in enumerate(order, start=1):
        interior = graph["interiors"][comp["compound_id"]]
        meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for leaf in interior["leaflets"]:
            ci = int(leaf["contour_index"])
            if ci not in tri_cache:
                tri_cache[ci] = triangulate_contour(contours[ci], grid_n)
            uv, faces = tri_cache[ci]
            verts = surface_points(uv, leaf)
            meshes[leaf["tag"]] = (verts, _orient_adaxial(verts, faces, np.asarray(leaf["up"], float)))
        compounds.append(
            FixtureCompound(
                gen_compound_id=comp["compound_id"],
                serial=serial,
                attachment_mm=np.asarray(comp["attachment_mm"], float),
                rachis_points_mm=np.asarray(interior["rachis_points_mm"], float),
                leaflets=list(interior["leaflets"]),
                meshes=meshes,
                age_days=float(comp["age_days"]),
                gen_rank_from_bottom=int(comp["rank_from_bottom"]),
                source_ref=f"{scene_dir.as_posix()}#{comp['compound_id']}",
            )
        )
    provenance = dict(
        scene_dir=scene_dir.as_posix(),
        plant_graph_sha256=sha256_file(scene_dir / "plant_graph.json"),
        leaflet_contours_sha256=sha256_file(cell / "leaflet_contours.json"),
        generator_config_sha256=sha256_file(cell / "generator_config.json"),
        seed=graph.get("seed"),
        generator_version=graph.get("generator_version"),
        prior_status="user_assumed_unvalidated",
        relief_displacement="not applied (mid-surface from leaflet_surface.surface_points)",
        modifications=[],
    )
    return FixturePlant(
        source_plant_id=source_plant_id,
        stem_points_mm=np.asarray(graph["stem"]["points_mm"], float),
        compounds=compounds,
        provenance=provenance,
        next_serial=len(compounds) + 1,
    )


# --------------------------------------------------------------------------------------
# Controlled modifications (tracking fixtures). Each is recorded in the provenance.
# --------------------------------------------------------------------------------------


def remove_compound(plant: FixturePlant, serial: int) -> FixturePlant:
    kept = [c for c in plant.compounds if c.serial != serial]
    if len(kept) == len(plant.compounds):
        raise s.P1SchemaError(f"no compound with serial {serial}")
    prov = dict(plant.provenance, modifications=[*plant.provenance["modifications"], f"removed C{serial:04d}"])
    return replace(plant, compounds=kept, provenance=prov)


def add_compound_from(plant: FixturePlant, donor: FixturePlant, donor_serial: int, yaw_deg: float,
                      lift_to_apex: bool = True) -> FixturePlant:
    """Add a donor compound as a NEW organ (next serial), rotated about the stem by ``yaw_deg``
    and, if ``lift_to_apex``, translated so its attachment sits at the recipient's apex."""
    comp = next(c for c in donor.compounds if c.serial == donor_serial)
    shift = np.zeros(3)
    if lift_to_apex:
        shift = plant.stem_points_mm[-1] - rotate_z(comp.attachment_mm[None], yaw_deg)[0]

    def move(p: np.ndarray) -> np.ndarray:
        return rotate_z(np.atleast_2d(p), yaw_deg) + shift

    leaflets = []
    for leaf in comp.leaflets:
        moved = dict(leaf)
        moved["base_mm"] = move(np.asarray(leaf["base_mm"], float))[0].tolist()
        for key in ("heading", "left", "up"):
            moved[key] = rotate_z(np.asarray(leaf[key], float)[None], yaw_deg)[0].tolist()
        leaflets.append(moved)
    new = FixtureCompound(
        gen_compound_id=comp.gen_compound_id,
        serial=plant.next_serial,
        attachment_mm=move(comp.attachment_mm)[0],
        rachis_points_mm=move(comp.rachis_points_mm),
        leaflets=leaflets,
        meshes={tag: (move(v), f) for tag, (v, f) in comp.meshes.items()},
        age_days=0.0,
        gen_rank_from_bottom=comp.gen_rank_from_bottom,
        source_ref=f"{comp.source_ref} (donor {donor.source_plant_id} C{donor_serial:04d}, yaw {yaw_deg} deg)",
    )
    prov = dict(plant.provenance, modifications=[
        *plant.provenance["modifications"], f"added C{plant.next_serial:04d} from {new.source_ref}"])
    return replace(plant, compounds=[*plant.compounds, new], provenance=prov, next_serial=plant.next_serial + 1)


# --------------------------------------------------------------------------------------
# Side shoots (MTG A4) - generator-built leaves on a declared side stem
# --------------------------------------------------------------------------------------


def set_side_shoot(plant: FixturePlant, axis: str, parent_serial: int, leaf_lengths_mm: list[float],
                   gen_cfg: dict[str, Any], contours: list[np.ndarray], seed: int, stem_length_mm: float,
                   stem_elevation_deg: float) -> FixturePlant:
    """Add or regrow side shoot ``axis`` in the axil of main compound ``parent_serial``.

    The side stem starts at the parent compound's attachment and points along its petiole azimuth, rising
    at ``stem_elevation_deg``; leaf ``i`` is generated by ``compound_leaf.build_compound`` (numpy backend)
    with rachis length ``leaf_lengths_mm[i]`` (fixture-declared), elevation and divergence drawn from the
    generator ranges with fixed seeds. Regrowing an existing axis keeps its compound serials (same organs,
    new sizes); new leaves get the next serials. Fixture geometry only - no Micro-Tom claim."""
    from tomato3d.synth.compound_leaf import build_compound
    from tomato3d.synth.config import constant, make_rng, range_of
    from tomato3d.synth.leaflet_surface import surface_points, triangulate_contour

    parent = next(c for c in plant.compounds if c.serial == parent_serial and c.axis == "M")
    az = math.atan2(*(parent.rachis_points_mm[1] - parent.rachis_points_mm[0])[1::-1])
    el = math.radians(stem_elevation_deg)
    direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    base = np.asarray(parent.attachment_mm, float)
    n_pts = max(len(leaf_lengths_mm), 1) + 1
    points = np.array([base + direction * stem_length_mm * k / (n_pts - 1) for k in range(n_pts)])
    existing = [c for c in plant.compounds if c.axis == axis]
    kept = [c for c in plant.compounds if c.axis != axis]
    grid_n = int(constant(gen_cfg, "leaflet_mesh_grid_n"))
    e_lo, e_hi = range_of(gen_cfg, "leaf_elevation_deg")
    d_lo, d_hi = range_of(gen_cfg, "phyllotaxis_divergence_deg")
    next_serial = plant.next_serial
    new: list[FixtureCompound] = []
    for i, length in enumerate(leaf_lengths_mm):
        rng = make_rng(seed, "p1_side_shoot", plant.source_plant_id, axis, i)
        elev = math.radians(float(rng.uniform(e_lo, e_hi)))
        leaf_az = az + math.radians(90.0 + i * float(rng.uniform(d_lo, d_hi)))
        at = points[min(i + 1, len(points) - 1)]
        comp_dict = dict(compound_id=f"{axis}L{i + 1}", attachment_mm=at.tolist(),
                         direction_unit=[math.cos(elev) * math.cos(leaf_az), math.cos(elev) * math.sin(leaf_az),
                                         math.sin(elev)],
                         rachis_arc_length_mm=float(length), left_extent_mm=float(length),
                         right_extent_mm=float(length))
        interior, _ = build_compound(gen_cfg, comp_dict, contours, seed * 100 + i, plant.source_plant_id,
                                     backend="numpy")
        meshes = {}
        for leaf in interior["leaflets"]:
            uv, faces = triangulate_contour(np.asarray(contours[int(leaf["contour_index"])], float), grid_n)
            verts = surface_points(uv, leaf)
            meshes[leaf["tag"]] = (verts, _orient_adaxial(verts, faces, np.asarray(leaf["up"], float)))
        serial = existing[i].serial if i < len(existing) else next_serial
        if i >= len(existing):
            next_serial += 1
        new.append(FixtureCompound(
            gen_compound_id=comp_dict["compound_id"], serial=serial, attachment_mm=at,
            rachis_points_mm=np.asarray(interior["rachis_points_mm"], float), leaflets=list(interior["leaflets"]),
            meshes=meshes, age_days=0.0, gen_rank_from_bottom=i + 1,
            source_ref=f"build_compound(numpy, seed={seed * 100 + i}) on side shoot {axis}", axis=axis))
    side = SideAxis(axis, parent_serial, points, f"declared side stem {stem_length_mm} mm at {stem_elevation_deg} deg")
    prov = dict(plant.provenance, modifications=[
        *plant.provenance["modifications"], f"side shoot {axis}: {len(leaf_lengths_mm)} leaves {leaf_lengths_mm} mm"])
    return replace(plant, compounds=kept + new, side_axes={**plant.side_axes, axis: side}, provenance=prov,
                   next_serial=next_serial)


def axis_of_compounds(plant: FixturePlant) -> dict[str, str]:
    return {plant.compound_vertex_id(c): c.axis for c in plant.compounds}


# --------------------------------------------------------------------------------------
# Truth geometry per compound (A1.5 / A2 points)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompoundPoints:
    p0: np.ndarray
    nodes: list[np.ndarray]
    rachis_end: np.ndarray
    terminal_tip: np.ndarray | None
    stem_above_dir: np.ndarray


def leaflet_tip(leaf: dict[str, Any], mesh: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Midrib tip: the mesh vertex farthest from the base along the leaflet heading."""
    verts = np.asarray(mesh[0], float)
    base = np.asarray(leaf["base_mm"], float)
    return verts[int(np.argmax((verts - base) @ np.asarray(leaf["heading"], float)))]


def compound_points(plant: FixturePlant, comp: FixtureCompound) -> CompoundPoints:
    rachis = comp.rachis_points_mm
    stations = sorted({round(float(l["rachis_station_mm"]), 6) for l in comp.leaflets if l["kind"] == "pair"})
    nodes = [g.point_at_arc_length(rachis, st) for st in stations]
    term = next((l for l in comp.leaflets if l["tag"] == "T"), None)
    tip = leaflet_tip(term, comp.meshes["T"]) if term is not None else None
    return CompoundPoints(
        p0=rachis[0],
        nodes=nodes,
        rachis_end=rachis[-1],
        terminal_tip=tip,
        stem_above_dir=_stem_above(plant.stem_of(comp.axis), rachis[0]),
    )


def _stem_above(stem: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Stem direction above the node nearest ``point`` (works for inclined side stems, too)."""
    stem = np.asarray(stem, float)
    k = int(np.argmin(np.linalg.norm(stem - np.asarray(point, float), axis=1)))
    k = min(k, len(stem) - 2)
    return stem[k + 1] - stem[k]


def _set(row: dict[str, str], attr: str, value: float | int | str | None, status: str, digits: int = 4) -> None:
    if value is None:
        s.set_value(row, attr, UNAVAILABLE, UNAVAILABLE)
    elif isinstance(value, str):
        s.set_value(row, attr, value, status)
    else:
        s.set_value(row, attr, s.fmt(value, digits), status)


def _set_xyz(row: dict[str, str], attr: str, xyz: np.ndarray | None, status: str) -> None:
    s.set_value(row, attr, s.fmt_triple(None if xyz is None else list(xyz)), UNAVAILABLE if xyz is None else status)


def _base_row(plant_instance_id: str, plant_id: str, capture_id: str, vertex_id: str, scale: int, cls: str,
              complex_id: str, parent_id: str, edge_type: str) -> dict[str, str]:
    row = s.blank_mtg_row()
    row.update(plant_instance_id=plant_instance_id, plant_id=plant_id, capture_id=capture_id,
               snapshot_capture_id=capture_id, vertex_id=vertex_id, scale=str(scale), complex_id=complex_id,
               parent_id=parent_id, edge_type=edge_type, display_name=UNAVAILABLE, observed="true",
               frame="plant_local_mm", evidence=EVIDENCE, **{"class": cls})
    return row


def _direction_values(row: dict[str, str], base: np.ndarray, tip: np.ndarray, status: str) -> None:
    v = np.asarray(tip, float) - np.asarray(base, float)
    _set_xyz(row, "base_xyz_mm", base, status)
    _set_xyz(row, "tip_xyz_mm", tip, status)
    _set(row, "azimuth_deg", g.azimuth_deg(v), DERIVED)
    _set(row, "elevation_deg", g.elevation_deg(v), DERIVED)


def truth_mtg(plant: FixturePlant, plant_instance_id: str, capture_id: str,
              ranks: dict[str, int], node_index: dict[str, int] | None = None) -> list[dict[str, str]]:
    """Truth MTG snapshot of one plant instance at one capture (plant-local mm).

    ``ranks`` maps compound vertex ids to U5 ranks (``tomato3d.state.ranks``; MTG A7). The generator's
    ``gen_rank_from_bottom`` is a node index, not this rank.
    """
    pid = plant.source_plant_id
    axis = f"{pid}/A0"
    rows: list[dict[str, str]] = []
    all_verts = np.vstack([v for c in plant.compounds for v, _ in c.meshes.values()])
    plant_row = _base_row(plant_instance_id, pid, capture_id, pid, 1, "Plant", UNAVAILABLE, UNAVAILABLE, UNAVAILABLE)
    _set(plant_row, "variety", "micro_tom", "assumed")
    _set(plant_row, "height_top_leaf_mm", float(all_verts[:, 2].max()), TRUTH)
    lowest = min(plant.compounds, key=lambda c: float(c.attachment_mm[2]))
    stem = plant.stem_points_mm
    above = stem[stem[:, 2] > lowest.attachment_mm[2]]
    _set(plant_row, "stem_length_mm", g.polyline_length(np.vstack([lowest.attachment_mm, above])), TRUTH)
    rows.append(plant_row)
    axis_row = _base_row(plant_instance_id, pid, capture_id, axis, 2, "Axis", pid, UNAVAILABLE, UNAVAILABLE)
    axis_row["axis"] = "M"
    _direction_values(axis_row, stem[0], stem[-1], TRUTH)
    _set(axis_row, "length_mm", g.polyline_length(stem), TRUTH)
    _set(axis_row, "order", 0, TRUTH)
    rows.append(axis_row)
    by_serial = {c.serial: c for c in plant.compounds}
    for name, side in sorted(plant.side_axes.items(), key=lambda kv: int(kv[0][1:])):
        srow = _base_row(plant_instance_id, pid, capture_id, plant.axis_vertex_id(name), 2, "Axis", pid, axis, "+")
        srow["axis"] = name
        _direction_values(srow, side.points_mm[0], side.points_mm[-1], TRUTH)
        _set(srow, "length_mm", g.polyline_length(side.points_mm), TRUTH)
        _set(srow, "order", 1, TRUTH)
        parent = by_serial.get(side.parent_serial)
        _set(srow, "parent_rank", ranks.get(plant.compound_vertex_id(parent)) if parent is not None else None, TRUTH)
        if node_index and parent is not None and plant.compound_vertex_id(parent) in node_index:
            _set(srow, "parent_node_index", node_index[plant.compound_vertex_id(parent)], TRUTH)
        rows.append(srow)
    for comp in plant.compounds:
        rows.extend(_compound_rows(plant, comp, plant_instance_id, capture_id, plant.axis_vertex_id(comp.axis), ranks,
                                   node_index or {}))
    return rows


def _compound_rows(plant: FixturePlant, comp: FixtureCompound, plant_instance_id: str, capture_id: str,
                   axis: str, ranks: dict[str, int], node_index: dict[str, int] | None = None) -> list[dict[str, str]]:
    pid = plant.source_plant_id
    cid = plant.compound_vertex_id(comp)
    pts = compound_points(plant, comp)
    term = next((l for l in comp.leaflets if l["tag"] == "T"), None)
    row = _base_row(plant_instance_id, pid, capture_id, cid, 3, "CompoundLeaf", axis, UNAVAILABLE, "+")
    tip = pts.terminal_tip if pts.terminal_tip is not None else pts.rachis_end
    _direction_values(row, pts.p0, tip, TRUTH)
    total = g.polyline_length(comp.rachis_points_mm)
    if term is not None:
        total += float(term.get("petiolule_mm", 0.0)) + float(term["length_mm"])
    _set(row, "length_mm", total, TRUTH)
    _set(row, "leaf_unfolded", "true", TRUTH)
    _set(row, "rank", ranks.get(cid), TRUTH)
    if node_index and cid in node_index and ranks.get(cid) is not None:
        _set(row, "node_index_from_base", node_index[cid], TRUTH)
    _set(row, "order", 0 if comp.axis == "M" else 1, TRUTH)
    if pts.nodes:
        _set(row, "compound_insertion_angle_deg", g.insertion_angle_deg(pts.stem_above_dir, pts.p0, pts.nodes[0]), DERIVED)
        alphas = g.bend_alphas_deg(pts.p0, pts.nodes, pts.rachis_end)
        if alphas is not None and len(alphas) > s.MAX_BEND_NODES:
            raise s.P1SchemaError(f"{cid}: {len(alphas)} bend nodes > MAX_BEND_NODES")
        for i, column in enumerate(s.ALPHA_COLUMNS, start=1):
            _set(row, column, alphas[i - 1] if alphas and i <= len(alphas) else None, DERIVED)
        _set(row, "n_bend_nodes", len(pts.nodes), DERIVED)
        _set(row, "rachis_curvature_deg", sum(alphas) if alphas else None, DERIVED)
        _set(row, "rachis_bend_theta_deg", g.theta_deg(pts.p0, pts.nodes), DERIVED)
        _set(row, "chord_to_path_ratio", g.chord_to_path_ratio(pts.p0, pts.nodes), DERIVED)
        _set(row, "petiole_base_azimuth_deg", g.azimuth_deg(pts.nodes[0] - pts.p0), DERIVED)
        if pts.terminal_tip is not None:
            _set(row, "blade_azimuth_deg", g.azimuth_deg(pts.terminal_tip - pts.nodes[0]), DERIVED)
            _set(row, "blade_azimuth_basis", "terminal_tip", DERIVED)
        else:
            _set(row, "blade_azimuth_deg", g.azimuth_deg(pts.nodes[-1] - pts.nodes[0]), DERIVED)
            _set(row, "blade_azimuth_basis", "last_pair_node", DERIVED)
    rows = [row]
    # axis segments: Petiole P0 -> node 1, then R1..Rm
    seg_points = [pts.p0, *pts.nodes, pts.rachis_end]
    previous = UNAVAILABLE
    for k in range(len(seg_points) - 1):
        code = f"{PETIOLE_CODE}1" if k == 0 else f"{RACHIS_CODE}{k}"
        cls = "Petiole" if k == 0 else "RachisSegment"
        vid = f"{cid}/{code}"
        seg = _base_row(plant_instance_id, pid, capture_id, vid, 4, cls, cid, previous,
                        "<" if previous != UNAVAILABLE else UNAVAILABLE)
        _direction_values(seg, seg_points[k], seg_points[k + 1], TRUTH)
        _set(seg, "length_mm", float(np.linalg.norm(seg_points[k + 1] - seg_points[k])), TRUTH)
        seg["node_index"] = str(k) if k > 0 else UNAVAILABLE
        rows.append(seg)
        previous = vid
    last_segment = previous
    node_of_station = {round(float(l["rachis_station_mm"]), 6) for l in comp.leaflets if l["kind"] == "pair"}
    node_index = {st: i for i, st in enumerate(sorted(node_of_station), start=1)}
    for leaf in comp.leaflets:
        rows.append(_leaflet_row(plant, comp, leaf, plant_instance_id, capture_id, cid, last_segment, node_index))
    for r in rows:
        r["axis"] = comp.axis
    return rows


def _segment_of_station(comp: FixtureCompound, station: float, node_index: dict[float, int], cid: str) -> str:
    """Rachis vertex a lateral/intercalary leaflet branches from (``+``)."""
    nodes = sorted(node_index)
    k = sum(1 for st in nodes if st <= station + 1e-6)
    return f"{cid}/{PETIOLE_CODE}1" if k == 0 else f"{cid}/{RACHIS_CODE}{k}"


def _leaflet_row(plant: FixturePlant, comp: FixtureCompound, leaf: dict[str, Any], plant_instance_id: str,
                 capture_id: str, cid: str, last_segment: str, node_index: dict[float, int]) -> dict[str, str]:
    tag = leaf["tag"]
    verts, faces = comp.meshes[tag]
    is_terminal = tag == "T"
    parent = last_segment if is_terminal else _segment_of_station(comp, float(leaf["rachis_station_mm"]), node_index, cid)
    row = _base_row(plant_instance_id, plant.source_plant_id, capture_id, f"{cid}/{tag}", 4, "Leaflet", cid,
                    parent, "<" if is_terminal else "+")
    tip = leaflet_tip(leaf, (verts, faces))
    _direction_values(row, np.asarray(leaf["base_mm"], float), tip, TRUTH)
    _set(row, "length_mm", float(leaf["length_mm"]), TRUTH)
    lateral = (verts - np.asarray(leaf["base_mm"], float)) @ np.asarray(leaf["left"], float)
    _set(row, "width_mm", float(np.ptp(lateral)), TRUTH)
    _set(row, "area_mm2", float(g.triangle_areas(verts, faces).sum()), TRUTH)
    normal = g.area_weighted_normal(verts, faces)
    _set_xyz(row, "normal_xyz", normal, TRUTH)
    tangent = g.tangent_at_arc_length(comp.rachis_points_mm, float(leaf["rachis_station_mm"]))
    _set(row, "leaflet_insertion_angle_deg", g.angle_between_deg(tangent, np.asarray(leaf["heading"], float)), DERIVED)
    row["side"] = leaf.get("side", UNAVAILABLE) or UNAVAILABLE
    if leaf["kind"] == "pair":
        idx = node_index[round(float(leaf["rachis_station_mm"]), 6)]
        row["pair_index"] = str(idx)
        row["node_index"] = str(idx)
    return row


def mtg_edges(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """``<`` and ``+`` edges from ``parent_id`` (``/`` lives in ``complex_id``)."""
    return [
        dict(plant_instance_id=r["plant_instance_id"], snapshot_capture_id=r["snapshot_capture_id"],
             parent_id=r["parent_id"], child_id=r["vertex_id"], edge_type=r["edge_type"])
        for r in rows
        if r["parent_id"] != UNAVAILABLE
    ]


# --------------------------------------------------------------------------------------
# Truth optical surfaces and areas
# --------------------------------------------------------------------------------------


def truth_surfaces(plant: FixturePlant) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Leaflet vertex id -> (vertices plant-local mm, faces adaxial_ccw)."""
    return plant.leaflet_meshes()


def render_capture(plant: FixturePlant, yaw_deg: float, camera: OrthoCamera) -> TopRender:
    """Top view of the plant with the pot turned by ``yaw_deg`` (image frame rotates with it)."""
    meshes = {lid: (rotate_z(v, yaw_deg), f) for lid, (v, f) in plant.leaflet_meshes().items()}
    return render_top(meshes, camera)


def truth_area_rows(plant: FixturePlant, plant_instance_id: str, capture_id: str, render: TopRender,
                    camera: OrthoCamera) -> list[dict[str, str]]:
    """The four area kinds per leaflet from truth (no estimate anywhere)."""
    px_area_mm2 = camera.mm_per_px ** 2
    out = []
    for lid, (verts, faces) in sorted(plant.leaflet_meshes().items()):
        vis = int(render.visible_mask(lid).sum())
        amo = int(render.amodal[lid].sum())
        row = {c: UNAVAILABLE for c in s.AREAS_COLUMNS}
        row.update(plant_instance_id=plant_instance_id, capture_id=capture_id,
                   compound_id=lid.rsplit("/", 1)[0], leaflet_id=lid, part="leaflet", obs_id=UNAVAILABLE,
                   evidence=EVIDENCE,
                   area_basis=s.AREA_BASIS_MAIN, scale_source="synthetic_ortho_camera_declared")
        row.update(visible_projected_area_px2=str(vis), visible_projected_area_px2__status=TRUTH,
                   visible_projected_area_mm2=s.fmt(vis * px_area_mm2, 3), visible_projected_area_mm2__status=TRUTH,
                   visible_projected_area_mm2_sd="0",  # declared synthetic camera: exact by construction
                   completed_projected_area_px2=str(amo), completed_projected_area_px2__status=TRUTH,
                   completed_projected_area_mm2=s.fmt(amo * px_area_mm2, 3), completed_projected_area_mm2__status=TRUTH,
                   completed_projected_area_mm2_sd="0",
                   completion_method="synthetic_truth_amodal_raster",
                   completion_coverage=s.fmt(vis / amo if amo else None, 4),
                   one_sided_lamina_area_mm2=s.fmt(float(g.triangle_areas(verts, faces).sum()), 3),
                   one_sided_lamina_area_mm2__status=TRUTH,
                   lamina_geometric_evidence="mesh_triangle_sum_plant_local")
        out.append(row)
    return out


# --------------------------------------------------------------------------------------
# Observation-like view
# --------------------------------------------------------------------------------------


@dataclass
class ObservationSet:
    """Observation rows + their masks + the fixture-only truth join (never read by the tracker)."""

    rows: list[dict[str, str]]
    masks: dict[str, np.ndarray]
    truth_join: list[dict[str, str]]


def _bbox(mask: np.ndarray) -> str:
    ys, xs = np.nonzero(mask)
    return json.dumps([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])


def _centroid(mask: np.ndarray) -> str:
    ys, xs = np.nonzero(mask)
    return json.dumps([round(float(xs.mean()), 2), round(float(ys.mean()), 2)])


def _touches_border(mask: np.ndarray) -> str:
    return "true" if (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()) else "false"


def observe_top(render: TopRender, plant_instance_id: str, capture_id: str, min_visible_px: int,
                source_id: str, merge_groups: list[list[str]] | None = None,
                split_compounds: list[str] | None = None,
                axis_of: dict[str, str] | None = None) -> ObservationSet:
    """Observation-like view of a rendered top capture.

    ``merge_groups`` / ``split_compounds`` simulate segmentation failures for tracking tests:
    a merged group becomes one compound observation, a split compound two (left/right halves of
    its leaflets). Both are recorded in ``source_id`` so the fixture never hides them.
    """
    by_compound: dict[str, list[str]] = {}
    for lid in render.ids:
        by_compound.setdefault(lid.rsplit("/", 1)[0], []).append(lid)
    groups: list[tuple[str, list[str]]] = []
    merged = {c for grp in (merge_groups or []) for c in grp}
    for grp in merge_groups or []:
        groups.append(("+".join(grp), [lid for c in grp for lid in by_compound.get(c, [])]))
    for cid, lids in sorted(by_compound.items()):
        if cid in merged:
            continue
        if cid in (split_compounds or []):
            half = sorted(lids)
            groups.append((f"{cid}~a", half[: len(half) // 2]))
            groups.append((f"{cid}~b", half[len(half) // 2:]))
        else:
            groups.append((cid, lids))
    rows: list[dict[str, str]] = []
    masks: dict[str, np.ndarray] = {}
    join: list[dict[str, str]] = []
    n = 0
    union_all = render.visible > 0
    for key, lids in groups:
        mask = np.zeros_like(union_all)
        for lid in lids:
            mask |= render.visible_mask(lid)
        if int(mask.sum()) < min_visible_px:
            continue
        n += 1
        obs_id = f"o{n:03d}"
        others = union_all & ~mask
        touches_other = bool((np.roll(mask, 1, 0) & others).any() or (np.roll(mask, 1, 1) & others).any())
        truth_axes = {(axis_of or {}).get(k.split("~")[0], UNAVAILABLE) for k in key.split("+")}
        hint = truth_axes.pop() if len(truth_axes) == 1 else UNAVAILABLE
        rows.append(_obs_row(plant_instance_id, capture_id, obs_id, "compound_leaf", source_id, mask,
                             touches_other, UNAVAILABLE, UNAVAILABLE, hint))
        masks[obs_id] = mask
        truth_ids = key.split("+") if "+" in key else [key.split("~")[0]]
        for t in truth_ids:
            join.append(dict(plant_instance_id=plant_instance_id, capture_id=capture_id, obs_id=obs_id,
                             truth_vertex_id=t, evidence=EVIDENCE))
        if "+" in key or "~" in key:
            continue  # a failed compound segmentation carries no leaflet membership
        for lid in sorted(lids):
            lmask = render.visible_mask(lid)
            if int(lmask.sum()) < min_visible_px:
                continue
            n += 1
            lobs = f"o{n:03d}"
            cls = "terminal_leaflet" if lid.endswith("/T") else "leaflet"
            rows.append(_obs_row(plant_instance_id, capture_id, lobs, cls, source_id, lmask,
                                 bool((lmask & (union_all & ~lmask)).any()), obs_id, "confirmed", hint))
            masks[lobs] = lmask
            join.append(dict(plant_instance_id=plant_instance_id, capture_id=capture_id, obs_id=lobs,
                             truth_vertex_id=lid, evidence=EVIDENCE))
    return ObservationSet(rows=rows, masks=masks, truth_join=join)


def _obs_row(plant_instance_id: str, capture_id: str, obs_id: str, obs_class: str, source_id: str,
             mask: np.ndarray, touches_other: bool, parent: str, membership: str,
             axis_hint: str = UNAVAILABLE) -> dict[str, str]:
    return dict(
        plant_instance_id=plant_instance_id, capture_id=capture_id, obs_id=obs_id, obs_class=obs_class,
        axis_hint=axis_hint, axis_hint_source="synthetic_gt" if axis_hint != UNAVAILABLE else UNAVAILABLE,
        provenance="synthetic_gt", source_id=source_id, source_score=UNAVAILABLE,
        detector_status="not_applicable", review_state="accepted", mask_ref=f"masks.json#{obs_id}",
        mask_quality="synthetic_gt_exact", area_px2=str(int(mask.sum())), bbox_xyxy_px=_bbox(mask),
        centroid_xy_px=_centroid(mask), touches_other_mask="true" if touches_other else "false",
        touches_image_border=_touches_border(mask), plant_attribution="this_plant",
        parent_obs_id=parent, membership_status=membership, frame="image_px", evidence=EVIDENCE,
    )


def observe_rendered_view(scene_dir: Path, view_dir: Path, plant: FixturePlant, plant_instance_id: str,
                          capture_id: str, min_visible_px: int) -> ObservationSet:
    """Observation-like view of an EXISTING perspective render (``visible_mask.png`` + index).

    The perspective view has no single mm-per-px, so callers keep ``scale_status = unavailable``.
    """
    import cv2

    from tomato3d.io.images import read_image

    index = json.loads((Path(scene_dir) / "leaflet_index.json").read_text(encoding="utf-8"))
    visible = read_image(Path(view_dir) / "visible_mask.png", cv2.IMREAD_UNCHANGED)
    if visible is None:
        raise s.P1SchemaError(f"cannot read {view_dir}/visible_mask.png")
    gen_to_vertex = {c.gen_compound_id: plant.compound_vertex_id(c) for c in plant.compounds}
    ids: list[str] = []
    remapped = np.zeros(visible.shape, np.uint16)
    for key, info in sorted(index.items(), key=lambda kv: int(kv[0])):
        vid = f"{gen_to_vertex[info['compound_id']]}/{info['tag']}"
        ids.append(vid)
        remapped[visible == int(key)] = len(ids)
    render = TopRender(visible=remapped, ids=ids, amodal={})
    rel = Path(view_dir).resolve()
    return observe_top(render, plant_instance_id, capture_id, min_visible_px,
                       source_id=f"synthetic_gt:{rel.parent.parent.name}/{rel.name}/visible_mask.png")


def fixture_manifest_entry(plant: FixturePlant, camera: OrthoCamera | None) -> dict[str, Any]:
    entry = dict(plant.provenance)
    entry.update(
        source_plant_id=plant.source_plant_id,
        evidence=EVIDENCE,
        frame_definition="generator plant-local: origin stem base on soil surface, +Z up",
        occlusion_rule=OCCLUSION_RULE,
        drawn_organs="leaflet laminae only (stem, rachis, petiolules not drawn)",
        raster_area_bias="cv2.fillPoly includes boundary pixels: rasterised (visible / amodal) areas are inflated "
                         "by about half a pixel along each outline (small leaflets up to ~12 %)",
        counts_as_photo_reconstruction=False,
    )
    if camera is not None:
        entry["ortho_camera"] = dict(image_size_px=camera.image_size_px, mm_per_px=camera.mm_per_px)
    return entry


def compound_serial_table(plant: FixturePlant) -> list[dict[str, str]]:
    """Generator compound id -> issued vertex id (creation order = oldest first)."""
    return [dict(source_plant_id=plant.source_plant_id, vertex_id=plant.compound_vertex_id(c),
                 generator_compound_id=c.gen_compound_id, age_days=s.fmt(c.age_days, 3),
                 source_ref=c.source_ref) for c in plant.compounds]

