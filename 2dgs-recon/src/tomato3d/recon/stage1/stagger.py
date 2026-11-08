"""Staggered-growth TRUTH for the synthetic test (session E3-2): a generator scene re-grown with the Session G rule.

The generator scenes grow all leaflets of a compound together (``simultaneous``). This option re-sizes a loaded
scene with its OWN stagger value (independent of the reconstruction's), so the test can cross truth stagger x
reconstruction stagger without circularity on the stagger itself:

* every leaflet keeps existing (none added or removed); its length is scaled by
  ``g_leaf(age - offset) / g_leaf(age)`` (staggered vs the scene's simultaneous size at the same compound age);
* each rachis segment takes the offset of the pair it carries (petiole -> basal pair, last segment -> terminal) and
  its length is scaled by ``g_rachis(age - offset) / g_rachis(age)``; leaflet bases move along the rachis to the new
  stations, keeping their petiolule offset; the rachis polyline is truncated at its new length (its curve shape kept);
* meshes are rebuilt with the generator's own functions (contour library of the scene's cell,
  ``triangulate_contour`` + ``surface_points``) - no new renderer;
* a leaflet whose offset is not reached yet is present but tiny (G1). No primordial size is given, so it is kept at
  ``PRESENCE_FLOOR`` x its mature length - a numerical presence only (area ~ 0), not a biological value;
* compound age = node position from the apex x PHYLLOCHRON (the same declared rule the reconstruction uses; the age
  rule is not what this grid tests). Intercalary leaflets take the offset of the nearest pair.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from tomato3d.recon.stage1 import age
from tomato3d.state import fixture_truth as ft
from tomato3d.state import geometry as g


PRESENCE_FLOOR = 1e-3


def _arc_lengths(poly: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]


def _truncate(poly: np.ndarray, length: float) -> np.ndarray:
    s = _arc_lengths(poly)
    keep = poly[s < length]
    return np.vstack([keep, g.point_at_arc_length(poly, length)]) if len(keep) else poly[:1]


def stagger_plant(plant: ft.FixturePlant, scene_dir: Path, stagger_dd: float, par: dict) -> ft.FixturePlant:
    from tomato3d.synth.contours import load_library
    from tomato3d.synth.leaflet_surface import surface_points, triangulate_contour

    cell = Path(scene_dir).resolve().parent.parent
    contours, _ = load_library(cell / "leaflet_contours.json")
    grid_n = int(json.loads((cell / "generator_config.json").read_text(encoding="utf-8"))["constants"]
                 ["leaflet_mesh_grid_n"]["value"])
    from tomato3d.state import ranks as rk
    registry = {plant.compound_vertex_id(c): dict(rank=rk.rank1010_to_u5(c.gen_rank_from_bottom, shared_first=(c.axis == "M")),
                                                  node_index=c.gen_rank_from_bottom, axis=c.axis) for c in plant.compounds}
    ages = age.compound_ages(registry, par)
    tri: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    new_compounds = []
    for comp in plant.compounds:
        A = ages[plant.compound_vertex_id(comp)]["age_dd"]
        rachis = comp.rachis_points_mm
        total = float(_arc_lengths(rachis)[-1])
        stations = sorted({round(float(l["rachis_station_mm"]), 6) for l in comp.leaflets if l["kind"] == "pair"})
        n = len(stations)
        pidx = {s: i + 1 for i, s in enumerate(stations)}            # 1 = stem side (basal), n = tip side

        def off_of(leaf):
            if leaf["tag"] == "T":
                return 0.0
            st = round(float(leaf["rachis_station_mm"]), 6)
            p = pidx.get(st) or pidx[min(stations, key=lambda x: abs(x - st))] if stations else None
            return age.offset_dd("pair", p, n, stagger_dd)

        gl = lambda t: age.leaflet_length_fraction(max(t, 0.0), par)     # noqa: E731
        gr = lambda t: age.rachis_length_fraction(max(t, 0.0), par)      # noqa: E731
        ref_l, ref_r = max(gl(A), 1e-9), max(gr(A), 1e-9)
        # new stations: segment i ends at pair i (petiole -> basal pair = segment 1), last segment -> terminal
        bounds = [0.0, *stations, total]
        offs = [age.offset_dd("pair", i, n, stagger_dd) for i in range(1, n + 1)] + [0.0]
        new_bounds = [0.0]
        for (a, b), o in zip(zip(bounds[:-1], bounds[1:]), offs):
            new_bounds.append(new_bounds[-1] + (b - a) * gr(A - o) / ref_r)
        remap = dict(zip([round(x, 6) for x in bounds], new_bounds))
        new_leaflets, meshes = [], {}
        for leaf in comp.leaflets:
            f = max(gl(A - off_of(leaf)) / ref_l, PRESENCE_FLOOR)
            st = float(leaf["rachis_station_mm"]) if leaf["tag"] != "T" else total
            key = round(st, 6)
            if key in remap:
                st_new = remap[key]
            else:   # intercalary: linear inside its segment
                j = int(np.searchsorted(bounds, st)) - 1
                j = min(max(j, 0), len(bounds) - 2)
                a, b = bounds[j], bounds[j + 1]
                st_new = new_bounds[j] + (st - a) / max(b - a, 1e-9) * (new_bounds[j + 1] - new_bounds[j])
            base = np.asarray(leaf["base_mm"], float)
            shift = g.point_at_arc_length(rachis, st_new) - g.point_at_arc_length(rachis, st)
            nl = dict(leaf, length_mm=float(leaf["length_mm"]) * f, base_mm=(base + shift).tolist(),
                      rachis_station_mm=st_new, petiolule_mm=float(leaf["petiolule_mm"]) * f)
            new_leaflets.append(nl)
            ci = int(leaf["contour_index"])
            if ci not in tri:
                tri[ci] = triangulate_contour(contours[ci], grid_n)
            uv, faces = tri[ci]
            v = surface_points(uv, nl)
            meshes[leaf["tag"]] = (v, ft._orient_adaxial(v, faces, np.asarray(leaf["up"], float)))
        new_compounds.append(replace(comp, leaflets=new_leaflets, meshes=meshes,
                                     rachis_points_mm=_truncate(rachis, max(new_bounds[-1], 1e-3))))
    prov = dict(plant.provenance, modifications=[*plant.provenance.get("modifications", []),
                                                 f"staggered truth {stagger_dd} dd (session E3)"])
    return replace(plant, compounds=new_compounds, provenance=prov)
