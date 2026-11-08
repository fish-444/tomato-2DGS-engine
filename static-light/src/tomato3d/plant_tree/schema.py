"""plant_tree_v1 — schema constants and a validator (plain Python; no new dependency).

A tree is one JSON document (dict). Every id is unique within the document. Levels: plant > axis > node > leaf > leaflet;
trusses hang on nodes or axis ends. Geometry is explicit (polylines, points, frames); organ meshes are either regenerated
from parameters (``param_blade``, ``tube``) or stored (``grid``, ``mesh``) so that an OBJ round trip is exact.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "plant_tree_v1"
SOURCES = ("synthetic", "reconstruction", "engine")
UNITS = ("mm", "board_m")
PROVENANCE_TAGS = ("user_measured", "user_measured_shape", "human_confirmed_shape", "literature", "assumption", "reconstructed", "derived_from_mesh", "engine_state", "unavailable")
AXIS_KINDS = ("main", "side", "sympodial")
LEAFLET_TYPES = ("P", "T", "I")
SIDES = ("L", "R", "C")
ORGAN_FLAGS = ("few_views", "ambiguous", "unattached", "gap", "edge_soft", "occluded_part", "merged_suspect", "stub",
               "intercalary_as_extra_leaflet", "lateral_shoot_mapped_to_sympodial_axis", "dropped_no_engine_kind",
               "clipped_to_measured_max", "requires_declaration")
SURFACE_KINDS = ("param_blade", "grid", "outline_polygon", "mesh", "ref")
TUBE_KINDS = ("tube", "mesh")


class ValidationError(ValueError):
    pass


def new_tree(source: str, units: str, frame: str, input_status: str, plant_id: str = "plant", synthetic: bool | None = None) -> dict:
    if source not in SOURCES:
        raise ValidationError(f"source {source!r} not in {SOURCES}")
    if units not in UNITS:
        raise ValidationError(f"units {units!r} not in {UNITS}")
    return dict(schema=SCHEMA, source=source, input_status=input_status, units=units, frame=frame,
                synthetic=(source == "synthetic") if synthetic is None else bool(synthetic),
                provenance={}, plant=dict(id=plant_id, flags=[], attrs={}),
                axes=[], nodes=[], leaves=[], leaflets=[], trusses=[], cotyledons=[], params={}, notes=[])


# ------------------------------------------------------------------ validation helpers
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec3(x, where: str) -> None:
    if not (isinstance(x, (list, tuple)) and len(x) == 3 and all(_is_num(v) for v in x)):
        raise ValidationError(f"{where}: expected [x, y, z], got {x!r}")


def _polyline(x, where: str, min_pts: int = 2) -> None:
    if not isinstance(x, list) or len(x) < min_pts:
        raise ValidationError(f"{where}: polyline needs >= {min_pts} points")
    for i, p in enumerate(x):
        _vec3(p, f"{where}[{i}]")


def _flags(x, where: str) -> None:
    if not isinstance(x, list) or any(f not in ORGAN_FLAGS for f in x):
        raise ValidationError(f"{where}: flags must be a list from {ORGAN_FLAGS}; got {x!r}")


def _opt_unit(x, where: str) -> None:
    if x is not None and not (_is_num(x) and 0 <= x <= 1):
        raise ValidationError(f"{where}: expected a number in [0, 1] or null")


def _surface(s, where: str) -> None:
    if not isinstance(s, dict) or s.get("kind") not in SURFACE_KINDS:
        raise ValidationError(f"{where}: surface.kind must be one of {SURFACE_KINDS}")
    k = s["kind"]
    if k == "param_blade":
        for key in ("base", "d", "n"):
            _vec3(s.get(key), f"{where}.{key}")
        for key in ("L", "W", "cup", "bend"):
            if not _is_num(s.get(key)):
                raise ValidationError(f"{where}.{key}: number required")
        if s.get("nu", 12) < 2 or s.get("nv", 7) < 2:
            raise ValidationError(f"{where}: nu, nv >= 2")
        if "outline" in s and s["outline"] is not None:
            _outline(s["outline"], f"{where}.outline")
    elif k == "grid":
        nu, nv = s.get("nu"), s.get("nv")
        if not (isinstance(nu, int) and isinstance(nv, int) and nu >= 1 and nv >= 1):
            raise ValidationError(f"{where}: grid needs integer nu, nv")
        if len(s.get("vertices", [])) != (nu + 1) * (nv + 1):
            raise ValidationError(f"{where}: grid needs (nu+1)*(nv+1) vertices")
    elif k == "outline_polygon":
        _outline(s, where)
    elif k == "mesh":
        if not s.get("vertices") or not s.get("faces"):
            raise ValidationError(f"{where}: mesh needs vertices and faces")
    elif k == "ref":
        if not isinstance(s.get("ref"), str):
            raise ValidationError(f"{where}: ref needs a string")


def _outline(o, where: str) -> None:
    pts = o.get("points")
    if not isinstance(pts, list) or len(pts) < 3 or not all(isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_num(v) for v in p) for p in pts):
        raise ValidationError(f"{where}: outline.points needs >= 3 [u, v] pairs")
    if o.get("tag") is not None and o["tag"] not in PROVENANCE_TAGS:
        raise ValidationError(f"{where}: outline.tag {o['tag']!r} not a provenance tag")


def _tube(t, where: str) -> None:
    if not isinstance(t, dict) or t.get("kind") not in TUBE_KINDS:
        raise ValidationError(f"{where}: tube.kind must be one of {TUBE_KINDS}")
    if t["kind"] == "tube":
        _polyline(t.get("polyline"), f"{where}.polyline")
        r = t.get("radius")
        if not (isinstance(r, list) and len(r) == len(t["polyline"]) and all(_is_num(x) and x >= 0 for x in r)):
            raise ValidationError(f"{where}.radius: one radius per polyline point")
        if not (isinstance(t.get("segs", 6), int) and t.get("segs", 6) >= 3):
            raise ValidationError(f"{where}.segs: integer >= 3")
    else:
        if not t.get("vertices") or not t.get("faces"):
            raise ValidationError(f"{where}: mesh needs vertices and faces")


def validate(tree: dict) -> dict:
    """Raise ValidationError on the first problem; return the tree. Checks structure, ids, references and geometry shapes."""
    if not isinstance(tree, dict) or tree.get("schema") != SCHEMA:
        raise ValidationError(f"schema must be {SCHEMA!r}")
    if tree.get("source") not in SOURCES:
        raise ValidationError(f"source not in {SOURCES}")
    if tree.get("units") not in UNITS:
        raise ValidationError(f"units not in {UNITS}")
    for key in ("frame", "input_status"):
        if not isinstance(tree.get(key), str) or not tree[key]:
            raise ValidationError(f"{key}: non-empty string required")
    if not isinstance(tree.get("synthetic"), bool):
        raise ValidationError("synthetic: bool required")
    if tree["source"] == "synthetic" and not tree["synthetic"]:
        raise ValidationError("a synthetic tree must carry synthetic: true")
    prov = tree.get("provenance", {})
    if not isinstance(prov, dict) or any(v not in PROVENANCE_TAGS for v in prov.values()):
        raise ValidationError(f"provenance values must be from {PROVENANCE_TAGS}")
    plant = tree.get("plant", {})
    if not isinstance(plant.get("id"), str):
        raise ValidationError("plant.id required")
    _flags(plant.get("flags", []), "plant.flags")
    ids: set[str] = set()

    def uid(x: dict, where: str) -> str:
        i = x.get("id")
        if not isinstance(i, str) or not i:
            raise ValidationError(f"{where}: id required")
        if i in ids:
            raise ValidationError(f"{where}: duplicate id {i!r}")
        ids.add(i); return i

    axes = {}
    for k, a in enumerate(tree.get("axes", [])):
        w = f"axes[{k}]"; i = uid(a, w)
        if a.get("kind") not in AXIS_KINDS:
            raise ValidationError(f"{w}.kind not in {AXIS_KINDS}")
        _tube(a.get("tube"), f"{w}.tube"); _flags(a.get("flags", []), f"{w}.flags")
        if a.get("parent_axis") is not None and a["parent_axis"] not in axes:
            raise ValidationError(f"{w}.parent_axis {a['parent_axis']!r} unknown (axes must be listed parents first)")
        axes[i] = a
    if sum(a["kind"] == "main" for a in axes.values()) != 1:
        raise ValidationError("exactly one axis of kind 'main'")
    nodes = {}
    for k, n in enumerate(tree.get("nodes", [])):
        w = f"nodes[{k}]"; i = uid(n, w)
        if n.get("axis") not in axes:
            raise ValidationError(f"{w}.axis unknown")
        if not (isinstance(n.get("index"), int) and n["index"] >= 1):
            raise ValidationError(f"{w}.index: integer >= 1 (1 = first node above the cotyledon node)")
        if n.get("rank") is not None and not (isinstance(n["rank"], int) and n["rank"] >= 1):
            raise ValidationError(f"{w}.rank: integer >= 1 or null")
        _vec3(n.get("position"), f"{w}.position"); _flags(n.get("flags", []), f"{w}.flags")
        if not _is_num(n.get("s_mm")):
            raise ValidationError(f"{w}.s_mm: arc length along the axis required")
        nodes[i] = n
    for a in axes.values():
        if a.get("parent_node") is not None and a["parent_node"] not in nodes:
            raise ValidationError(f"axis {a['id']}: parent_node {a['parent_node']!r} unknown")
    leaves = {}
    for k, lf in enumerate(tree.get("leaves", [])):
        w = f"leaves[{k}]"; i = uid(lf, w)
        if lf.get("node") not in nodes:
            raise ValidationError(f"{w}.node unknown")
        if lf.get("rank") is not None and not (isinstance(lf["rank"], int) and lf["rank"] >= 1):
            raise ValidationError(f"{w}.rank: integer >= 1 or null")
        _tube(lf.get("rachis"), f"{w}.rachis")
        if lf.get("petiole") is not None:
            _tube(lf["petiole"], f"{w}.petiole")
        att = lf.get("attachments")
        if not isinstance(att, list):
            raise ValidationError(f"{w}.attachments: list required")
        for j, at in enumerate(att):
            if not _is_num(at.get("s_mm")) or not isinstance(at.get("leaflets"), list):
                raise ValidationError(f"{w}.attachments[{j}]: s_mm and leaflets required")
        for key in ("azimuth_deg", "elevation_deg"):
            if lf.get(key) is not None and not _is_num(lf[key]):
                raise ValidationError(f"{w}.{key}: number or null")
        _opt_unit(lf.get("expansion"), f"{w}.expansion"); _flags(lf.get("flags", []), f"{w}.flags")
        leaves[i] = lf
    leaflets = {}
    for k, lt in enumerate(tree.get("leaflets", [])):
        w = f"leaflets[{k}]"; i = uid(lt, w)
        if lt.get("leaf") not in leaves:
            raise ValidationError(f"{w}.leaf unknown")
        if lt.get("type") not in LEAFLET_TYPES or lt.get("side") not in SIDES:
            raise ValidationError(f"{w}: type in {LEAFLET_TYPES} and side in {SIDES} required")
        if not (isinstance(lt.get("pair"), int) and lt["pair"] >= 0):
            raise ValidationError(f"{w}.pair: integer >= 0 (0 = terminal)")
        if not _is_num(lt.get("attachment_s_mm")):
            raise ValidationError(f"{w}.attachment_s_mm required")
        _vec3(lt.get("base"), f"{w}.base")
        fr = lt.get("frame")
        if fr is not None:
            for key in ("d", "n", "b"):
                _vec3(fr.get(key), f"{w}.frame.{key}")
        for key in ("length_mm", "width_mm", "area_mm2"):
            if lt.get(key) is not None and not _is_num(lt[key]):
                raise ValidationError(f"{w}.{key}: number or null")
        if lt.get("normal") is not None:
            _vec3(lt["normal"], f"{w}.normal")
        _surface(lt.get("surface"), f"{w}.surface")
        if lt.get("petiolule") is not None:
            _tube(lt["petiolule"], f"{w}.petiolule")
        _opt_unit(lt.get("expansion"), f"{w}.expansion"); _flags(lt.get("flags", []), f"{w}.flags")
        leaflets[i] = lt
    for lf in leaves.values():
        for at in lf["attachments"]:
            for li in at["leaflets"]:
                if li not in leaflets or leaflets[li]["leaf"] != lf["id"]:
                    raise ValidationError(f"leaf {lf['id']}: attachment lists leaflet {li!r} that is not its own")
    for k, t in enumerate(tree.get("trusses", [])):
        w = f"trusses[{k}]"; uid(t, w)
        if (t.get("node") is None) == (t.get("axis") is None):
            raise ValidationError(f"{w}: exactly one of node / axis")
        if t.get("node") is not None and t["node"] not in nodes:
            raise ValidationError(f"{w}.node unknown")
        if t.get("axis") is not None and t["axis"] not in axes:
            raise ValidationError(f"{w}.axis unknown")
        _tube(t.get("peduncle"), f"{w}.peduncle")
        fl = t.get("flowers")
        if not isinstance(fl, list) or not isinstance(t.get("flower_count"), int) or t["flower_count"] != len(fl):
            raise ValidationError(f"{w}: flowers list and matching flower_count required")
        for j, f in enumerate(fl):
            uid(f, f"{w}.flowers[{j}]"); _vec3(f.get("position"), f"{w}.flowers[{j}].position")
            if f.get("mesh") is not None and not (f["mesh"].get("vertices") and f["mesh"].get("faces")):
                raise ValidationError(f"{w}.flowers[{j}].mesh: vertices and faces")
            if f.get("pedicel") is not None:
                _tube(f["pedicel"], f"{w}.flowers[{j}].pedicel")
        _flags(t.get("flags", []), f"{w}.flags")
    for k, c in enumerate(tree.get("cotyledons", [])):
        w = f"cotyledons[{k}]"; uid(c, w); _surface(c.get("surface"), f"{w}.surface")
    return tree


def organ_ids(tree: dict) -> dict[str, list[str]]:
    return {k: [x["id"] for x in tree.get(k, [])] for k in ("axes", "nodes", "leaves", "leaflets", "trusses", "cotyledons")}
