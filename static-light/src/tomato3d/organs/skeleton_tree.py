"""skeleton_v1 -> plant_tree_v1 (contract build/p1_skeleton_layer_20261101_01/CONTRACT.md, section 4 of the prompt). No engine change.

One `leaf` per live skeleton leaf: node at the polyline base, azimuth / elevation of the first 10 mm of the polyline, rachis length straight and along the path, `leaf_uid` kept as an attribute.
Leaf area (b) comes from the propagated labels where they exist; internodes, node heights of the main stem and masses stay `unavailable` / `requires_declaration` (the engine fills them: conditional_hidden_state;
the stem is display-only). Validated with `plant_tree.schema.validate`; `engine_round_trip` runs the existing converter and validators, nothing else.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from tomato3d.plant_tree.schema import new_tree, validate

UNAV = "unavailable"
SCALE_TAG = "conditional_on_unvalidated_scale"
DIR_MM = 10.0                                   # the first 10 mm of the polyline give azimuth / elevation


def first_direction(P_mm, length_mm=DIR_MM):
    """Unit direction of the polyline over its first `length_mm` (path distance); None for a single point."""
    P = np.asarray(P_mm, float)
    if len(P) < 2:
        return None
    acc = 0.0
    for i in range(1, len(P)):
        acc += float(np.linalg.norm(P[i] - P[i - 1]))
        if acc >= length_mm:
            break
    v = P[i] - P[0]; n = np.linalg.norm(v)
    return v / n if n > 0 else None


def az_el(d):
    return (float(math.degrees(math.atan2(d[1], d[0])) % 360.0), float(math.degrees(math.asin(max(-1.0, min(1.0, d[2])))))) if d is not None else (None, None)


def leaf_areas_from_labels(rows_label, rows_status, area_b_mm2, uids):
    """Area (b) per leaf_uid from a propagated route: only surfels with status 1 (assigned) count, so it is a lower bound when surfels are contested. uids[k] = label k+1."""
    out = {}
    for k, u in enumerate(uids):
        m = (rows_label == k + 1) & (rows_status == 1)
        out[u] = dict(area_b_mm2=float(np.asarray(area_b_mm2)[m].sum()), n_surfels=int(m.sum()))
    return out


def skeleton_to_plant_tree(sk, areas=None, area_source=None, status="R3-1b-style_unaccepted"):
    """sk: skeleton_v1 dict; areas: {leaf_uid: {area_b_mm2, n_surfels}} or None. Returns a validated plant_tree_v1 dict (units mm, board frame)."""
    tree = new_tree("reconstruction", "mm", "board frame, metres converted to mm, z up", status, plant_id=sk["plant_id"], synthetic=False)
    tree["provenance"].update(geometry="reconstructed", node_position="user_measured", internodes=UNAV, masses=UNAV, leaflets=UNAV)
    tree["plant"]["attrs"] = dict(scale=SCALE_TAG, skeleton_schema=sk["schema"], capture_date=sk["capture_date"], reconstruction_id=sk["reconstruction_id"], labels=sk["labels"],
                                  stem="display_only (no stem is observed; the main axis is a zero-length placeholder at the frame origin)", area_source=area_source or UNAV)
    main = next((a for a in sk.get("axes", []) if a["type"] == "main" and a["status"] == "active" and a.get("fit")), None)
    if main is not None:
        return _with_axis(tree, sk, main, areas, status)
    tree["axes"].append(dict(id="M", kind="main", parent_axis=None, parent_node=None, tube=dict(kind="tube", polyline=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], radius=[0.0, 0.0], segs=6),
                             flags=["requires_declaration"], attrs=dict(geometry_status=UNAV)))
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost")]
    for l in sorted(live, key=lambda l: l["rank_hint"] or 999):
        P = np.asarray(l["polyline_xyz"], float) * 1000.0; uid = l["leaf_uid"]; nid = f"N_{uid}"; lid = f"L_{uid}"
        d = first_direction(P); az, el = az_el(d); path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0; straight = float(np.linalg.norm(P[-1] - P[0])) if len(P) > 1 else 0.0
        pl = P.tolist() if len(P) >= 2 else [P[0].tolist(), P[0].tolist()]                       # a one-click leaf keeps only its base: zero-length rachis placeholder
        a = (areas or {}).get(uid)
        tree["nodes"].append(dict(id=nid, axis="M", index=int(max(l["rank_hint"], 1)), rank=None, s_mm=0.0, position=[float(x) for x in P[0]], leaves=[lid], trusses=[], flags=["requires_declaration"],
                                  attrs=dict(index_meaning="rank by base height of the skeleton (an ordering, not a node count)", position_status="user_measured (skeleton base)", s_mm_status=UNAV)))
        tree["leaves"].append(dict(id=lid, node=nid, axis="M", rank=None, petiole=None, rachis=dict(kind="tube", polyline=pl, radius=[0.0] * len(pl), segs=6), attachments=[],
                                   length_mm=(path if len(P) > 1 else None), azimuth_deg=az, elevation_deg=el, age_d=None, expansion=None, flags=["requires_declaration"],
                                   provenance=dict(rachis="user_measured" if len(P) > 1 else UNAV, blade="reconstructed" if a else UNAV),
                                   attrs=dict(leaf_uid=uid, rank_hint=l["rank_hint"], skeleton_source=l["source"], skeleton_status=l["status"], skeleton_flags=l.get("flags", []), n_clicks=l["n_clicks"],
                                              rachis_path_mm=(path if len(P) > 1 else None), rachis_straight_mm=(straight if len(P) > 1 else None), direction_source=f"first {DIR_MM:g} mm of the polyline" if d is not None else UNAV,
                                              area_b_mm2=(a["area_b_mm2"] if a else UNAV), area_n_surfels=(a["n_surfels"] if a else UNAV),
                                              area_label=("lower bound: assigned surfels of the distance_baseline route only; " + SCALE_TAG) if a else UNAV)))
    tree["notes"] += [f"skeleton_v1 -> plant_tree_v1; {status}; {SCALE_TAG}", "hidden structure (internodes, main-stem node heights, masses) is not observed: placeholders flagged requires_declaration; the engine fills them (conditional_hidden_state)"]
    return validate(tree)


def _leaf_entry(l, nid, areas, extra_flags=()):
    P = np.asarray(l["polyline_xyz"], float) * 1000.0; uid = l["leaf_uid"]; d = first_direction(P); az, el = az_el(d)
    path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0; straight = float(np.linalg.norm(P[-1] - P[0])) if len(P) > 1 else 0.0
    pl = P.tolist() if len(P) >= 2 else [P[0].tolist(), P[0].tolist()]; a = (areas or {}).get(uid)
    return dict(id=f"L_{uid}", node=nid, axis="M", rank=None, petiole=None, rachis=dict(kind="tube", polyline=pl, radius=[0.0] * len(pl), segs=6), attachments=[],
                length_mm=(path if len(P) > 1 else None), azimuth_deg=az, elevation_deg=el, age_d=None, expansion=None, flags=["requires_declaration", *extra_flags],
                provenance=dict(rachis="user_measured" if len(P) > 1 else UNAV, blade="reconstructed" if a else UNAV),
                attrs=dict(leaf_uid=uid, rank_hint=l["rank_hint"], skeleton_source=l["source"], skeleton_status=l["status"], skeleton_flags=l.get("flags", []), n_clicks=l["n_clicks"],
                           rachis_path_mm=(path if len(P) > 1 else None), rachis_straight_mm=(straight if len(P) > 1 else None), direction_source=f"first {DIR_MM:g} mm of the polyline" if d is not None else UNAV,
                           attach_distance_mm=l.get("attach_distance_mm"), area_b_mm2=(a["area_b_mm2"] if a else UNAV), area_n_surfels=(a["n_surfels"] if a else UNAV),
                           area_label=("lower bound: assigned surfels of the propagated route only; " + SCALE_TAG) if a else UNAV))


def _with_axis(tree, sk, main, areas, status):
    """Amendment A6: the fitted main axis is the main tube; nodes on it at attach_s (hypothesised nodes carry no leaf); internode provenance and lengths only with labels."""
    C = [list(map(float, p)) for p in main["fit"]["curve_mm"]]; nodes = main["nodes"]; r0 = nodes[0]["radius_mm"] if nodes else 0.0
    def prov_at(s0, s1):
        cov = sum(max(0.0, min(s1, sg["s1_mm"]) - max(s0, sg["s0_mm"])) for sg in main["segments"] if sg["provenance"] == "observed")
        return "observed" if s1 > s0 and cov >= 0.999 * (s1 - s0) else "model_completed"
    tree["provenance"].update(internodes="user_measured", node_position="user_measured")                    # schema tags only; per-internode observed / model_completed sits in node attrs
    tree["plant"]["attrs"].update(stem="user_skeleton_axis + model_completed fill (Amendment A6): per-internode provenance in node attrs; the schema-level tag user_measured covers the observed parts only")
    tree["axes"].append(dict(id="M", kind="main", parent_axis=None, parent_node=None, tube=dict(kind="tube", polyline=C, radius=[float(r0)] * len(C), segs=6), flags=["requires_declaration"],
                             attrs=dict(radius_source="model_generated (engine reference run, rank 1)", axis_uid=main["axis_uid"], geometry_status="user_skeleton_axis + model_completed", segments=main["segments"], axis_flags=main.get("flags", []), scale=SCALE_TAG)))
    leaves = {l["leaf_uid"]: l for l in sk["leaves"]}; prev = 0.0
    for n in nodes:
        nid = f"N_{main['axis_uid']}_{n['node_index']}"; ip = prov_at(prev, n["s_mm"])
        node = dict(id=nid, axis="M", index=int(n["node_index"]), rank=None, s_mm=float(n["s_mm"]), position=[float(x) for x in n["xyz_mm"]], leaves=[], trusses=[], flags=(["requires_declaration"] if n["kind"] != "observed_leaf" else []),
                    attrs=dict(kind=n["kind"], node_flags=list(n.get("flags", [])), internode_below=dict(length_mm=round(n["s_mm"] - prev, 2), provenance=ip, label=("observed_click" if ip == "observed" else "model_completed") + "; " + SCALE_TAG),
                               expected_internode_mm=n["expected_internode_mm"], radius_mm=n["radius_mm"], radius_source="model_generated", azimuth_deg=n.get("azimuth_deg")))
        if n.get("leaf_uid"):
            node["leaves"].append(f"L_{n['leaf_uid']}"); tree["leaves"].append(_leaf_entry(leaves[n["leaf_uid"]], nid, areas))
        tree["nodes"].append(node); prev = n["s_mm"]
    k = max([n["node_index"] for n in nodes] or [0])
    for l in sk["leaves"]:                                                       # unattached leaves: own node at their nearest axis point, flagged
        if l["status"] in ("active", "new", "candidate_lost") and l.get("attach_axis_uid") is None:
            k += 1; nid = f"N_unattached_{l['leaf_uid']}"; P0 = [float(x) * 1000.0 for x in l["polyline_xyz"][0]]
            tree["nodes"].append(dict(id=nid, axis="M", index=k, rank=None, s_mm=float(prev), position=P0, leaves=[f"L_{l['leaf_uid']}"], trusses=[], flags=["unattached", "requires_declaration"],
                                      attrs=dict(kind="unattached_leaf", attach_distance_mm=l.get("attach_distance_mm"))))
            tree["leaves"].append(_leaf_entry(l, nid, areas, ("unattached",)))
    tree["notes"] += [f"skeleton_v1.1 -> plant_tree_v1; {status}; {SCALE_TAG}", "main axis = user clicks + model_completed fill (Amendment A6); hypothesised nodes (fspm_node_hypothesis) carry no leaf and are not observations; lengths only with their provenance label"]
    return validate(tree)


def engine_round_trip(tree, out: Path, pid):
    """The existing converter and validators, unchanged (as plant_model.run.engine_round_trip). Returns a report, or the reason it could not run."""
    if not tree["leaves"]:
        return dict(validate_initial="not_run", round_trip_pass="not_run", status="tree has no leaf")
    from tomato3d.plant_tree import engine as PE
    from tomato3d.plant_tree.schema import validate as V
    from tools.vtc_development_observation import validate_initial
    main = next(a for a in tree["axes"] if a["kind"] == "main"); origin = np.asarray(main["tube"]["polyline"][0], float)            # the converter's plant-local origin = base of the main axis
    out = Path(out); rep = PE.to_engine_structure(tree, out, plant_instance_id="p1", source_plant_id=f"reconstruction_{pid}", fixture=False, capture_id=f"skeleton_{pid}")
    with open(out / "plants.csv", encoding="utf-8-sig") as f:
        plants = list(csv.DictReader(f))
    ret = validate_initial(out, dict(fixture=False), plants); back = V(PE.from_engine_states(out / "initial_state.csv", 0, "p1"))
    with open(out / "initial_state.csv", encoding="utf-8-sig") as f:
        st = list(csv.DictReader(f))
    cmp = []
    for lf in tree["leaves"]:
        row = next(r for r in st if r["organ_id"] == rep["id_map"][lf["id"]]); pos = [float(row[k]) * 1000 for k in ("x_m", "y_m", "z_m")]
        base = np.subtract(next(n for n in tree["nodes"] if n["id"] == lf["node"])["position"], origin); cmp.append(float(np.max(np.abs(np.subtract(pos, base)))))
    return dict(validate_initial=True, organs=rep["organs"], back_leaves=len(back["leaves"]), back_nodes=len(back["nodes"]), max_base_diff_mm=float(max(cmp)) if cmp else None, round_trip_pass=bool(max(cmp) < 1e-3) if cmp else False,
                dropped=rep.get("dropped"))
