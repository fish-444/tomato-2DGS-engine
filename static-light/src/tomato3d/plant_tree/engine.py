"""plant_tree_v1 <-> the engine (tools/vtc_development_native.rgg, `kinds`) and its observation route
(tools/vtc_development_observation.py). Read-only towards the engine: nothing here edits tools/.

Mapping (SCHEMA.md §4): plant -> Plant; main axis -> A0 (+ I<n>, N<n>); sympodial axes -> A<n>; leaf -> C<nnnn>, C/P, C/R<i>;
leaflet -> C/P<i>L|R, C/T; truss -> Truss<n>, Truss<n>/F<i>; root -> Root (mass only, never invented). Declared fallbacks:
intercalary leaflet I -> an extra leaflet on the nearest rachis segment, flag `intercalary_as_extra_leaflet`; Micro-Tom
lateral shoot S<k> -> a sympodial axis A<k> on its node's internode, flag `lateral_shoot_mapped_to_sympodial_axis`;
cotyledons / hypocotyl -> dropped, listed in dropped.json with flag `dropped_no_engine_kind`.

`to_engine_structure` writes the Stage-2 inputs the observation route validates (initial_state.csv, initial_controls.json,
mtg_vertices.csv (revision 9: node_index_from_base), mtg_edges.csv, structure_state.csv, observation_manifest.json,
plants.csv) plus `initial_state_template.csv` where the hidden variables (mass, age, reserves, root) are EMPTY and the row
is flagged `requires_declaration`. The validated initial_state.csv carries 0.0 placeholders so the route's checks can run;
they are placeholders, declared as such in the manifest, never estimates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tomato3d.plant_tree import geometry as g
from tomato3d.plant_tree.schema import new_tree, validate

KINDS = ["Plant", "Axis", "Apex", "Attachment", "CompoundLeaf", "Root", "Internode", "Petiole", "RachisSegment", "Leaflet", "TrussAxis", "Flower"]
SUPPORTED = {"Plant": 1, "Axis": 2, "CompoundLeaf": 3, "Attachment": 3, "Internode": 4, "Petiole": 4, "RachisSegment": 4, "Leaflet": 4, "Apex": 4}
STATE_COLS = ("day,plant_instance_id,source_plant_id,organ_id,parent_id,axis_id,class,true_leaf_order,birth_day,emergence_day,thermal_age_dd,"
              "mass_gDM,max_mass_gDM,x_m,y_m,z_m,tip_x_m,tip_y_m,tip_z_m,length_m,width_m,area_m2,rachis_bend_alpha_deg,azimuth_deg,elevation_deg,"
              "pair_index,side,provenance").split(",")
HIDDEN = ("thermal_age_dd", "mass_gDM", "max_mass_gDM", "birth_day", "emergence_day")
ENGINE_DEFAULT_MISMATCHES = [  # from the engine request parameters (build/vtc_dev_v3_acceptance_01, 2026-10-22); report only
    dict(parameter="FIRST_LEAVES", engine=9, measured="6-8 (n_truss {6:2, 7:1, 8:2})"),
    dict(parameter="PAIRS", engine=[1, 1, 2], measured="dissected leaf: 3 pairs + up to 2 intercalary"),
    dict(parameter="LEAF_LW", engine=2.0, measured="1.0-2.2, mean about 1.5"),
    dict(parameter="leaflet shape", engine="flat diamond (leaflet_outline hook off)", measured="28 dissected outlines, handoff/leaflet_shape_20261014"),
    dict(parameter="INTERNODE_CAP_M", engine=0.012, measured="about 6-9 mm"),
    dict(parameter="architecture", engine="sympodial A1-A3 + 4 trusses (CADENCE [2,1,0])", literature="Vicente 2015: two successive trusses, then lateral shoots"),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict], header: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header or list(rows[0])); w.writeheader(); w.writerows(rows)


def _dir(a, b):
    v = np.subtract(b, a); ln = float(np.linalg.norm(v))
    if ln == 0:
        return 0.0, 90.0, 0.0
    az = (math.degrees(math.atan2(v[1], v[0])) + 360) % 360; el = math.degrees(math.asin(max(-1, min(1, v[2] / ln))))
    return az, el, ln


# ====================================================================== tree -> engine
def to_engine_structure(tree: dict, out: Path, plant_instance_id: str = "p1", source_plant_id: str | None = None, fixture: bool = True,
                        capture_id: str = "plant_tree_v1") -> dict:
    """Write the observation-route Stage-2 inputs for one plant. Returns a report (ids, dropped organs, mapping notes)."""
    validate(tree); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pid = plant_instance_id; spid = source_plant_id or f"{tree['source']}_{tree['plant']['id']}"
    to_m = 1.0 / 1000 if tree["units"] == "mm" else 1.0; to_mm = 1.0 if tree["units"] == "mm" else 1000.0
    axes = {a["id"]: a for a in tree["axes"]}; main = next(a for a in tree["axes"] if a["kind"] == "main")
    nodes = {n["id"]: n for n in tree["nodes"]}; leaves = {lf["id"]: lf for lf in tree["leaves"]}; lts = {lt["id"]: lt for lt in tree["leaflets"]}
    main_pl = main["tube"]["polyline"] if main["tube"]["kind"] == "tube" else main["tube"]["centreline"]
    origin = np.asarray(main_pl[0], float)                                   # plant-local origin = base of the main axis
    shift_m = np.asarray(tree.get("plant", {}).get("attrs", {}).get("engine_position_m", [0.0, 0.0, 0.0]), float)
    org, rows, dropped, notes = [], [], [], []
    prov = "observation_fixture" if fixture else "conditional_hidden_state"

    def P(p):
        return (np.asarray(p, float) - origin) * to_m + shift_m

    def add(oid, parent, axis, cls, order, base, tip, pair=0, side=0, length=None, width=0.0, area=0.0, alpha=0.0, az=None, el=None, tree_id=None, flags=()):
        b = P(base); t = P(tip); a0, e0, ln = _dir(b, t)
        rows.append({"day": "0", "plant_instance_id": pid, "source_plant_id": spid, "organ_id": oid, "parent_id": parent, "axis_id": axis, "class": cls,
                     "true_leaf_order": int(order), "birth_day": 0, "emergence_day": 0 if cls in ("Leaflet", "Flower") else -1, "thermal_age_dd": 0.0,
                     "mass_gDM": 0.0, "max_mass_gDM": 0.0, "x_m": float(b[0]), "y_m": float(b[1]), "z_m": float(b[2]), "tip_x_m": float(t[0]), "tip_y_m": float(t[1]),
                     "tip_z_m": float(t[2]), "length_m": float(ln if length is None else length * to_m), "width_m": float(width * to_m), "area_m2": float(area * to_m * to_m),
                     "rachis_bend_alpha_deg": float(alpha), "azimuth_deg": float(a0 if az is None else az), "elevation_deg": float(e0 if el is None else el),
                     "pair_index": int(pair), "side": int(side), "provenance": prov})
        org.append(dict(organ_id=oid, tree_id=tree_id or oid, flags=list(flags)))
        return oid

    plant_id = add(f"{pid}/Plant", "", "", "Plant", 0, main_pl[0], main_pl[0])
    # ---- axes: main = A0; side / sympodial -> A<k> in tree order
    axis_oid = {main["id"]: f"{pid}/A0"}; k_axis = 0
    for a in tree["axes"]:
        if a["kind"] != "main":
            k_axis += 1; axis_oid[a["id"]] = f"{pid}/A{k_axis}"
    leaf_counter = 0; leaf_oid = {}; last_stem = {}; node_oid = {}
    for a in tree["axes"]:
        aid = axis_oid[a["id"]]; pl = a["tube"]["polyline"] if a["tube"]["kind"] == "tube" else a["tube"]["centreline"]
        ax_nodes = sorted([n for n in tree["nodes"] if n["axis"] == a["id"]], key=lambda n: n["index"])
        flags = []
        if a["kind"] == "main":
            parent = plant_id; order = 0
        else:
            pn = nodes[a["parent_node"]] if a.get("parent_node") else None
            parent_axis_stem = last_stem.get(a.get("parent_axis"), axis_oid.get(a.get("parent_axis"), plant_id))
            parent = f"{pid}/I{pn['index']}" if pn and f"{pid}/I{pn['index']}" in {r['organ_id'] for r in rows} else parent_axis_stem
            if a["kind"] == "sympodial" and pn is None:
                parent = parent_axis_stem                                  # engine: A<k> hangs on the last internode of the previous axis
            order = pn["index"] if pn else 0
            if a["kind"] == "side":
                flags.append("lateral_shoot_mapped_to_sympodial_axis")
        add(aid, parent, aid, "Axis", order, pl[0], pl[0], tree_id=a["id"], flags=flags)
        prev = aid; prev_pos = pl[0]
        for n in ax_nodes:
            oid = f"{pid}/I{n['index']}" if a["kind"] in ("main", "sympodial") else f"{pid}/{aid.split('/')[-1]}/I{n['index']}"
            add(oid, prev, aid, "Internode", n["index"], prev_pos, n["position"], width=(a["tube"].get("radius") or [0])[0] * 2 if a["tube"]["kind"] == "tube" else 0.0, tree_id=n["id"])
            nid = f"{pid}/N{n['index']}" if a["kind"] in ("main", "sympodial") else f"{pid}/{aid.split('/')[-1]}/N{n['index']}"
            add(nid, oid, aid, "Attachment", n["index"], n["position"], n["position"], tree_id=n["id"]); node_oid[n["id"]] = nid
            prev = oid; prev_pos = n["position"]
            for lid in n.get("leaves", []):
                lf = leaves[lid]; leaf_counter += 1; cid = f"{pid}/C{leaf_counter:04d}"; leaf_oid[lid] = cid
                rpl = lf["rachis"]["polyline"] if lf["rachis"]["kind"] == "tube" else lf["rachis"]["centreline"]
                def _pair_of(at):
                    return min([lts[i]["pair"] for i in at["leaflets"] if lts[i]["type"] == "P"] or [99])
                atts = sorted(lf["attachments"], key=lambda x: (x["s_mm"], _pair_of(x)))
                pairs = sorted([at for at in atts if any(lts[i]["type"] == "P" for i in at["leaflets"])], key=lambda x: (x["s_mm"], _pair_of(x)))
                term = [at for at in atts if any(lts[i]["type"] == "T" for i in at["leaflets"])]
                inter = [at for at in atts if any(lts[i]["type"] == "I" for i in at["leaflets"])]
                s_first = pairs[0]["s_mm"] if pairs else (term[0]["s_mm"] if term else g.polyline_length(rpl))
                pet_tip = g.point_at_arclength(rpl, s_first)
                add(cid, nid, aid, "CompoundLeaf", n["index"], n["position"], pet_tip, az=lf.get("azimuth_deg"), el=lf.get("elevation_deg"), tree_id=lid, flags=lf.get("flags", []))
                pet = add(f"{cid}/P", cid, aid, "Petiole", n["index"], rpl[0], pet_tip, width=(lf["rachis"].get("radius") or [0])[0] * 2 if lf["rachis"]["kind"] == "tube" else 0.0)
                prev_seg = pet; prev_s = s_first; seg_oid = {}
                m = len(pairs)
                for i, at in enumerate(pairs, 1):
                    s_end = pairs[i]["s_mm"] if i < m else (term[0]["s_mm"] if term else g.polyline_length(rpl))
                    sid = f"{cid}/R{i}"
                    a_prev = g.point_at_arclength(rpl, prev_s); a_end = g.point_at_arclength(rpl, s_end)
                    holder = prev_seg                                  # the segment that ends at this pair's attachment
                    add(sid, prev_seg, aid, "RachisSegment", n["index"], a_prev, a_end, pair=i)
                    seg_oid[i] = sid; prev_seg = sid; prev_s = s_end
                    for li in at["leaflets"]:
                        lt = lts[li]
                        if lt["type"] != "P":
                            continue
                        side = -1 if lt["side"] == "L" else 1
                        tip = np.asarray(lt["base"]) + np.asarray(lt["frame"]["d"]) * (lt.get("length_mm") or 0) if lt.get("frame") else lt["base"]
                        add(f"{cid}/P{i}{'L' if side < 0 else 'R'}", holder, aid, "Leaflet", n["index"], lt["base"], tip, pair=i, side=side, length=lt.get("length_mm"),
                            width=lt.get("width_mm") or 0.0, area=lt.get("area_mm2") or 0.0, tree_id=li, flags=lt.get("flags", []))
                if term:
                    lt = lts[next(i for i in term[0]["leaflets"] if lts[i]["type"] == "T")]
                    tip = np.asarray(lt["base"]) + np.asarray(lt["frame"]["d"]) * (lt.get("length_mm") or 0) if lt.get("frame") else lt["base"]
                    add(f"{cid}/T", prev_seg, aid, "Leaflet", n["index"], lt["base"], tip, pair=m + 1, side=0, length=lt.get("length_mm"), width=lt.get("width_mm") or 0.0,
                        area=lt.get("area_mm2") or 0.0, tree_id=lt["id"], flags=lt.get("flags", []))
                for at in inter:                                     # declared fallback: extra leaflet on the nearest rachis segment
                    for li in at["leaflets"]:
                        lt = lts[li]
                        if lt["type"] != "I":
                            continue
                        seg = max([i for i, p in enumerate(pairs, 1) if p["s_mm"] <= at["s_mm"]] or [1]); sid = seg_oid.get(seg, pet)
                        side = -1 if lt["side"] == "L" else 1
                        tip = np.asarray(lt["base"]) + np.asarray(lt["frame"]["d"]) * (lt.get("length_mm") or 0) if lt.get("frame") else lt["base"]
                        add(f"{cid}/I{lt['pair']}{'L' if side < 0 else 'R'}", sid, aid, "Leaflet", n["index"], lt["base"], tip, pair=seg, side=side, length=lt.get("length_mm"),
                            width=lt.get("width_mm") or 0.0, area=lt.get("area_mm2") or 0.0, tree_id=li, flags=list(lt.get("flags", [])) + ["intercalary_as_extra_leaflet"])
                        notes.append(f"{li}: intercalary leaflet mapped to {sid} (flag intercalary_as_extra_leaflet)")
        last_stem[a["id"]] = prev
        # apex at the axis end (stopped when a truss ends it)
        add(f"{aid}/apex", aid, aid, "Apex", ax_nodes[-1]["index"] if ax_nodes else 0, pl[0], pl[0], tree_id=a["id"] + "/apex")   # engine: apex = zero-length organ at its axis's base
    # ---- trusses
    for k, t in enumerate(tree["trusses"], 1):
        ppl = t["peduncle"]["polyline"] if t["peduncle"]["kind"] == "tube" else t["peduncle"]["centreline"]
        if t.get("axis"):
            aid = axis_oid[t["axis"]]; parent = last_stem.get(t["axis"], aid); order = max([n["index"] for n in tree["nodes"] if n["axis"] == t["axis"]] or [0])
        else:
            n = nodes[t["node"]]; aid = axis_oid[n["axis"]]; parent = f"{pid}/I{n['index']}" if axes[n["axis"]]["kind"] in ("main", "sympodial") else f"{pid}/{aid.split('/')[-1]}/I{n['index']}"; order = n["index"]
        tid = f"{pid}/Truss{k}"
        add(tid, parent, aid, "TrussAxis", order, ppl[0], ppl[-1], tree_id=t["id"], flags=t.get("flags", []))
        for i, fl in enumerate(t["flowers"], 1):
            base = fl["pedicel"]["polyline"][0] if fl.get("pedicel") and fl["pedicel"]["kind"] == "tube" else (fl["pedicel"]["centreline"][0] if fl.get("pedicel") else fl["position"])
            add(f"{tid}/F{i}", tid, aid, "Flower", order, base, fl["position"], pair=i, tree_id=fl["id"])
    for c in tree.get("cotyledons", []):
        dropped.append(dict(tree_id=c["id"], reason="no engine kind for cotyledon", flag="dropped_no_engine_kind"))
    if tree.get("params", {}).get("hypocotyl_mm") is not None:
        notes.append("hypocotyl length kept only as the main axis base (no engine kind)")
    # Root: mass only; nothing is invented
    add(f"{pid}/Root", plant_id, f"{pid}/A0", "Root", 0, main_pl[0], main_pl[0], flags=["requires_declaration"])
    # ---- files
    _write_csv(out / "initial_state.csv", rows, list(STATE_COLS))
    tmpl = []
    for r in rows:
        t = dict(r); [t.__setitem__(h, "") for h in HIDDEN]; t["requires_declaration"] = "mass,age,reserves,root" if r["class"] != "Plant" else "reserve_gC"; tmpl.append(t)
    _write_csv(out / "initial_state_template.csv", tmpl, list(STATE_COLS) + ["requires_declaration"])
    by_id = {r["organ_id"]: r for r in rows}
    main_aid = axis_oid[main["id"]]; stems = [r for r in rows if r["axis_id"] == main_aid and r["class"] == "Internode"]
    controls = {pid: dict(active_apex=f"{main_aid}/apex", active_axis=main_aid, last_stem=max(stems, key=lambda s: s["true_leaf_order"])["organ_id"] if stems else main_aid,
                          plant_root=plant_id, thermal_dd=0.0, produced=len(stems), leaf_order=max([s["true_leaf_order"] for s in stems] or [0]),
                          truss_count=sum(r["class"] == "TrussAxis" for r in rows), axis_leaf_count=sum(r["class"] == "CompoundLeaf" and r["axis_id"] == main_aid for r in rows),
                          stopped_apices=[f"{main_aid}/apex"] if any(t.get("axis") == main["id"] for t in tree["trusses"]) else [],
                          provenance="requires_declaration: hidden developmental state is not observable; placeholders only")}
    (out / "initial_controls.json").write_text(json.dumps(controls, indent=1), encoding="utf-8")
    vertices, edges, struct = [], [], []
    from tomato3d.state.ranks import u5_rank_of_node
    for s in rows:                                            # mirrors tools/vtc_development_observation.fixture_from_native (revision 9)
        kind = s["class"]
        if kind not in SUPPORTED:
            continue
        oid = s["organ_id"]; parent = s["parent_id"]; scale = SUPPORTED[kind]; ancestor = parent; within = ""; complex_id = "unavailable"
        while ancestor:
            a = by_id[ancestor]; a_scale = SUPPORTED.get(a["class"])
            if a_scale == scale and not within:
                within = ancestor
            if a_scale == scale - 1:
                complex_id = ancestor; break
            ancestor = a["parent_id"]
        if kind == "Internode":
            complex_id = next((a["organ_id"] for a in rows if a["class"] == "Attachment" and a["parent_id"] == oid), "unavailable")
        is_main = s["axis_id"].endswith("/A0")
        v = dict(frame="plant_local_mm", snapshot_capture_id=capture_id, vertex_id=oid, scale=scale, **{"class": "Metamer" if kind == "Attachment" else kind},
                 complex_id=complex_id, parent_id=within, edge_type="<" if kind in ("Internode", "RachisSegment") or (kind == "Leaflet" and int(s["side"]) == 0) else "+",
                 base_xyz_mm=json.dumps([float(s[k]) * 1000 for k in ("x_m", "y_m", "z_m")]), tip_xyz_mm=json.dumps([float(s[k]) * 1000 for k in ("tip_x_m", "tip_y_m", "tip_z_m")]),
                 length_mm=float(s["length_m"]) * 1000, width_mm=float(s["width_m"]) * 1000, area_mm2=float(s["area_m2"]) * 1e6,
                 azimuth_deg=float(s["azimuth_deg"]), elevation_deg=float(s["elevation_deg"]), normal_xyz="unavailable",
                 rank=u5_rank_of_node(int(s["true_leaf_order"]), shared_first=is_main) if kind == "CompoundLeaf" else "unavailable",
                 node_index_from_base=int(s["true_leaf_order"]) if kind == "CompoundLeaf" else "unavailable", order=0 if is_main else 1)
        for key in ("base_xyz_mm", "tip_xyz_mm", "length_mm", "width_mm", "area_mm2", "azimuth_deg", "elevation_deg", "normal_xyz", "rank", "node_index_from_base", "order"):
            v[key + "__status"] = "unavailable" if v[key] == "unavailable" else ("observed" if tree["source"] == "reconstruction" else "model_estimated")
        vertices.append(v)
        if within:
            edges.append(dict(snapshot_capture_id=capture_id, parent_id=within, child_id=oid, edge_type=v["edge_type"]))
        struct.append(dict(plant_id=spid, capture_id=capture_id, vertex_id=oid, part=kind, upstream_track_state="active", link_status="confirmed",
                           time_since_lights_on_min="unavailable", photoperiod_phase="unavailable", frame="plant_local_mm", scale_status="solved",
                           evidence_ref=f"plant_tree_v1:{tree['source']}:{next(o['tree_id'] for o in org if o['organ_id'] == oid)}"))
    _write_csv(out / "mtg_vertices.csv", vertices); _write_csv(out / "mtg_edges.csv", edges); _write_csv(out / "structure_state.csv", struct)
    _write_csv(out / "plants.csv", [dict(plant_instance_id=pid, source_plant_id=spid, x_m=0.0, y_m=0.0, z_m=0.0, reserve_gC=0.0, provenance="requires_declaration")])
    files = ["initial_state.csv", "initial_controls.json", "mtg_vertices.csv", "mtg_edges.csv", "structure_state.csv"]
    manifest = dict(fixture=fixture, producer="plant_tree_v1.engine.to_engine_structure", source_manifest_sha256="n/a", rotation_status="solved",
                    transform_source=f"plant_tree_v1 {tree['frame']} -> plant_local: translation by the main-axis base {origin.tolist()} ({tree['units']})",
                    scale_source="plant_tree_v1 units " + tree["units"], biology_status="not_established",
                    supplement_status="placeholders: thermal_age_dd, mass_gDM, max_mass_gDM, birth_day, reserve_gC are 0.0 placeholders (requires_declaration); see initial_state_template.csv",
                    schema_revision=9, rank_convention="U5_A7: rank = u5_rank_of_node(node_index_from_base); main stem shared first rank, side axes one rank per node",
                    organ_split=tree.get("plant", {}).get("attrs", {}).get("organ_split", "n/a"), synthetic=tree["synthetic"], input_status=tree["input_status"],
                    files={f: _sha(out / f) for f in files})
    (out / "observation_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    report = dict(plant_instance_id=pid, organs=len(rows), stage2_vertices=len(vertices), edges=len(edges), dropped=dropped, notes=notes,
                  engine_default_mismatches=ENGINE_DEFAULT_MISMATCHES, id_map={o["tree_id"]: o["organ_id"] for o in org})
    (out / "mapping_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8"); (out / "dropped.json").write_text(json.dumps(dropped, indent=1), encoding="utf-8")
    return report


# ====================================================================== engine states -> tree
def from_engine_states(states_csv: Path, day: int, plant_instance_id: str | None = None, input_status: str = "engine_state") -> dict:
    """One engine `states` snapshot (or an observation `initial_state.csv`) at `day` -> plant_tree_v1 (mm, plant_local, z up)."""
    with Path(states_csv).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["day"]) == day]
    if plant_instance_id is None:
        plant_instance_id = rows[0]["plant_instance_id"]
    rows = [r for r in rows if r["plant_instance_id"] == plant_instance_id]
    by = {r["organ_id"]: r for r in rows}; kids = defaultdict(list)
    for r in rows:
        kids[r["parent_id"]].append(r)
    tree = new_tree("engine", "mm", "plant_local_mm, z up", input_status, plant_id=plant_instance_id, synthetic=False)
    plant_row = next((r for r in rows if r["class"] == "Plant"), None)
    origin_m = [float(plant_row[k]) for k in ("x_m", "y_m", "z_m")] if plant_row else [0.0, 0.0, 0.0]
    tree["plant"]["attrs"]["engine_position_m"] = origin_m
    tree["provenance"].update(geometry="engine_state", length_mm="engine_state", width_mm="engine_state")
    tree["notes"].append(f"from engine states at day {day}; leaflet surfaces are param_blade with cup=bend=0 (the engine draws flat diamonds; the outline is not stored)")
    B = lambda r: [(float(r[k]) - o) * 1000 for k, o in zip(("x_m", "y_m", "z_m"), origin_m)]  # noqa: E731
    T = lambda r: [(float(r[k]) - o) * 1000 for k, o in zip(("tip_x_m", "tip_y_m", "tip_z_m"), origin_m)]  # noqa: E731
    axes = sorted([r for r in rows if r["class"] == "Axis"], key=lambda r: r["organ_id"])
    stems_of = defaultdict(list)
    for r in rows:
        if r["class"] == "Internode":
            stems_of[r["axis_id"]].append(r)
    for r in axes:
        aid = r["organ_id"]; short = aid.split("/")[-1]; st = sorted(stems_of[aid], key=lambda s: int(s["true_leaf_order"]))
        pl = [B(r)] + [T(s) for s in st] if st else [B(r), T(r)]
        if len(pl) < 2:
            pl = [pl[0], pl[0]]
        rad = [float(s["width_m"]) * 500 for s in st] if st else [0.0]
        rad = [rad[0]] + rad if len(rad) == len(pl) - 1 else rad[:len(pl)] + [rad[-1]] * (len(pl) - len(rad))
        parent_axis = None
        if short != "A0":
            pr = by.get(r["parent_id"]); parent_axis = pr["axis_id"] if pr else None
            parent_axis = (parent_axis.split("/")[-1] if parent_axis else None)
        tree["axes"].append(dict(id=short, kind="main" if short == "A0" else "sympodial", parent_axis=parent_axis, parent_node=None,
                                 tube=dict(kind="tube", polyline=pl, radius=rad, segs=8 if short == "A0" else 6), engine_id=aid, flags=[]))
    for a in tree["axes"]:
        if a["parent_axis"] and a["parent_axis"] not in {x["id"] for x in tree["axes"]}:
            a["parent_axis"] = None
    tree["axes"].sort(key=lambda a: (a["kind"] != "main", a["id"]))
    for r in sorted([r for r in rows if r["class"] == "Attachment"], key=lambda r: int(r["true_leaf_order"])):
        short = r["axis_id"].split("/")[-1]; idx = int(r["true_leaf_order"]); is_main = short == "A0"
        from tomato3d.state.ranks import u5_rank_of_node
        stem = by.get(r["parent_id"]); pl = next(a for a in tree["axes"] if a["id"] == short)["tube"]["polyline"]
        tree["nodes"].append(dict(id=r["organ_id"].split("/")[-1] if is_main else f"{short}_{r['organ_id'].split('/')[-1]}", axis=short, index=idx,
                                  rank=u5_rank_of_node(idx, shared_first=is_main), s_mm=g.arclength_of_nearest(pl, B(r)), position=B(r), leaves=[], trusses=[], engine_id=r["organ_id"], flags=[]))
    nid_of = {n["engine_id"]: n["id"] for n in tree["nodes"]}
    for r in sorted([r for r in rows if r["class"] == "CompoundLeaf"], key=lambda r: r["organ_id"]):
        cid = r["organ_id"]; short = r["axis_id"].split("/")[-1]; lid = cid.split("/")[-1]
        node = nid_of.get(r["parent_id"])
        if node is None:
            continue
        pet = by.get(cid + "/P"); segs = sorted([x for x in rows if x["parent_id"].startswith(cid) and x["class"] == "RachisSegment"], key=lambda x: int(x["pair_index"]))
        pl = [B(pet)] if pet else [B(r)]
        if pet:
            pl.append(T(pet))
        pl += [T(s) for s in segs]
        if len(pl) < 2:
            pl.append(T(r))
        rad = [float(pet["width_m"]) * 500 if pet else 0.5] * len(pl)
        cum = [0.0]
        for a, b in zip(pl, pl[1:]):
            cum.append(cum[-1] + float(np.linalg.norm(np.subtract(b, a))))
        lf = dict(id=lid, node=node, axis=short, rank=u5_rank_of_node(int(r["true_leaf_order"]), shared_first=short == "A0"),
                  petiole=dict(kind="tube", polyline=[B(pet), T(pet)], radius=[float(pet["width_m"]) * 500] * 2, segs=6) if pet else None,
                  rachis=dict(kind="tube", polyline=pl, radius=rad, segs=6), attachments=[], length_mm=cum[-1], azimuth_deg=float(r["azimuth_deg"]),
                  elevation_deg=float(r["elevation_deg"]), age_d=None, expansion=None, engine_id=cid, flags=[])
        next(n for n in tree["nodes"] if n["id"] == node)["leaves"].append(lid)
        att = defaultdict(list)
        for x in rows:
            if x["class"] == "Leaflet" and x["parent_id"].startswith(cid):
                tag = x["organ_id"].split("/")[-1]; typ = "T" if tag == "T" else ("I" if tag.startswith("I") else "P")
                side = "C" if typ == "T" else ("L" if int(x["side"]) < 0 else "R"); pair = 0 if typ == "T" else int(x["pair_index"])
                base = B(x); tip = T(x); d = np.subtract(tip, base); ln = float(np.linalg.norm(d)); d = (d / ln).tolist() if ln else [1.0, 0, 0]
                n = np.cross(np.cross(d, [0, 0, 1.0]), d); nn = float(np.linalg.norm(n)); n = (n / nn).tolist() if nn else [0, 0, 1.0]
                b = np.cross(n, d).tolist()
                L = float(x["length_m"]) * 1000; W = float(x["width_m"]) * 1000
                s_mm = g.arclength_of_nearest(pl, base)
                tree["leaflets"].append(dict(id=f"{lid}__{tag}", leaf=lid, type=typ, side=side, pair=pair, attachment_s_mm=s_mm, base=base, frame=dict(d=d, n=n, b=b),
                                             length_mm=L, width_mm=W, area_mm2=float(x["area_m2"]) * 1e6, normal=n,
                                             surface=dict(kind="param_blade", base=base, d=d, n=n, L=L, W=W, cup=0.0, bend=0.0, nu=12, nv=7), petiolule=None,
                                             age_d=None, expansion=None, engine_id=x["organ_id"], flags=[]))
                att[(pair if typ != "T" else 999, typ)].append((s_mm, f"{lid}__{tag}"))
        lf["attachments"] = [dict(s_mm=max(x[0] for x in v), leaflets=[x[1] for x in v]) for k, v in sorted(att.items())]
        tree["leaves"].append(lf)
    for r in sorted([r for r in rows if r["class"] == "TrussAxis"], key=lambda r: r["organ_id"]):
        tid = r["organ_id"].split("/")[-1]; short = r["axis_id"].split("/")[-1]
        flowers = [dict(id=f"{tid}_{x['organ_id'].split('/')[-1]}", position=T(x), mesh=None, pedicel=dict(kind="tube", polyline=[B(x), T(x)], radius=[0.3, 0.25], segs=4), engine_id=x["organ_id"])
                   for x in sorted([x for x in rows if x["class"] == "Flower" and x["parent_id"] == r["organ_id"]], key=lambda x: int(x["pair_index"]))]
        tree["trusses"].append(dict(id=tid, node=None, axis=short, peduncle=dict(kind="tube", polyline=[B(r), T(r)], radius=[0.7, 0.45], segs=5), flowers=flowers,
                                    flower_count=len(flowers), stage="engine", engine_id=r["organ_id"], flags=[]))
    return validate(tree)
