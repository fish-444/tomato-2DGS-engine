"""summary() and compare() — the ONE set of descriptors used for reconstruction, synthetic and engine trees alike.

Per-leaf descriptors are relative (no mm claims about real plants): azimuth (deg, from the node along the first part of
the rachis, XY plane), elevation (deg of that first part), rachis curvature (turning deg per mm of rachis length),
leaflet count, rachis length, pairs. compare() matches leaves by rank and azimuth (Hungarian), then reports node-count
error, membership agreement and 1-D Wasserstein distances between the descriptor distributions.
"""

from __future__ import annotations

import math

import numpy as np

from tomato3d.plant_tree import geometry as g

DESCRIPTORS = ("azimuth_deg", "elevation_deg", "curvature_deg_per_mm", "leaflet_count", "rachis_length_mm", "pairs")


def _rachis_polyline(lf: dict) -> list:
    t = lf["rachis"]
    return t["polyline"] if t["kind"] == "tube" else t["centreline"]


def leaf_descriptors(tree: dict, lf: dict) -> dict:
    pl = np.asarray(_rachis_polyline(lf), float); L = g.polyline_length(pl)
    k = min(3, len(pl) - 1); d0 = pl[k] - pl[0]
    az = float(np.degrees(np.arctan2(d0[1], d0[0])) % 360) if np.hypot(d0[0], d0[1]) > 1e-9 else float("nan")
    el = float(np.degrees(np.arctan2(d0[2], np.hypot(d0[0], d0[1]))))
    lts = [lt for lt in tree["leaflets"] if lt["leaf"] == lf["id"]]
    return dict(id=lf["id"], axis=lf.get("axis"), rank=lf.get("rank"), azimuth_deg=az, elevation_deg=el,
                curvature_deg_per_mm=g.polyline_turning_deg(pl) / (L or 1), leaflet_count=len(lts), rachis_length_mm=L,
                pairs=max([lt["pair"] for lt in lts] + [0]), intercalary=sum(lt["type"] == "I" for lt in lts), flags=list(lf.get("flags", [])))


def summary(tree: dict) -> dict:
    axes = tree["axes"]; main = next(a for a in axes if a["kind"] == "main")
    main_nodes = [n for n in tree["nodes"] if n["axis"] == main["id"]]
    leaves = [leaf_descriptors(tree, lf) for lf in tree["leaves"]]
    per_axis = {}
    for a in axes:
        per_axis[a["id"]] = dict(kind=a["kind"], nodes=sum(n["axis"] == a["id"] for n in tree["nodes"]), leaves=sum(lf.get("axis") == a["id"] for lf in tree["leaves"]),
                                 trusses=sum(bool(t.get("axis") == a["id"] or (t.get("node") and any(n["id"] == t["node"] and n["axis"] == a["id"] for n in tree["nodes"]))) for t in tree["trusses"]))
    return dict(schema=tree["schema"], source=tree["source"], synthetic=tree["synthetic"], units=tree["units"], input_status=tree["input_status"],
                counts=dict(axes=len(axes), side_axes=sum(a["kind"] != "main" for a in axes), nodes=len(tree["nodes"]), main_nodes=len(main_nodes),
                            main_ranks=len({n.get("rank") for n in main_nodes if n.get("rank")}), leaves=len(tree["leaves"]), leaflets=len(tree["leaflets"]),
                            leaflets_P=sum(lt["type"] == "P" for lt in tree["leaflets"]), leaflets_T=sum(lt["type"] == "T" for lt in tree["leaflets"]),
                            leaflets_I=sum(lt["type"] == "I" for lt in tree["leaflets"]), trusses=len(tree["trusses"]),
                            flowers=sum(t["flower_count"] for t in tree["trusses"]), cotyledons=len(tree.get("cotyledons", []))),
                per_axis=per_axis, leaves=leaves,
                flags={f: sum(f in x.get("flags", []) for coll in ("leaves", "leaflets", "nodes", "axes", "trusses") for x in tree[coll]) for f in
                       sorted({f for coll in ("leaves", "leaflets", "nodes", "axes", "trusses") for x in tree[coll] for f in x.get("flags", [])})})


def _ang(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return 180.0
    return abs((a - b + 180) % 360 - 180)


def match_leaves(la: list[dict], lb: list[dict], rank_weight: float = 90.0, max_cost: float = 120.0) -> list[tuple[int, int, float]]:
    """Hungarian matching on cost = rank_weight * |rank difference| + azimuth difference (deg); pairs above max_cost are dropped."""
    from scipy.optimize import linear_sum_assignment
    if not la or not lb:
        return []
    C = np.zeros((len(la), len(lb)))
    for i, x in enumerate(la):
        for j, y in enumerate(lb):
            rd = abs((x.get("rank") or 0) - (y.get("rank") or 0)) if (x.get("rank") and y.get("rank")) else 1
            C[i, j] = rank_weight * rd + _ang(x["azimuth_deg"], y["azimuth_deg"]) + (0 if x.get("axis") == y.get("axis") else 60)
    r, c = linear_sum_assignment(C)
    return [(int(i), int(j), float(C[i, j])) for i, j in zip(r, c) if C[i, j] <= max_cost]


def compare(tree_a: dict, tree_b: dict) -> dict:
    """Tree A (e.g. reconstruction) against tree B (e.g. synthetic truth or another model)."""
    from scipy.stats import wasserstein_distance
    sa, sb = summary(tree_a), summary(tree_b); la, lb = sa["leaves"], sb["leaves"]
    m = match_leaves(la, lb); ia = {i for i, _, _ in m}; ib = {j for _, j, _ in m}
    num = sum(min(la[i]["leaflet_count"], lb[j]["leaflet_count"]) for i, j, _ in m)
    den = sum(max(la[i]["leaflet_count"], lb[j]["leaflet_count"]) for i, j, _ in m) + sum(x["leaflet_count"] for k, x in enumerate(la) if k not in ia) + sum(x["leaflet_count"] for k, x in enumerate(lb) if k not in ib)
    dist = {}
    for key in DESCRIPTORS:
        xa = [x[key] for x in la if x[key] is not None and not (isinstance(x[key], float) and math.isnan(x[key]))]
        xb = [x[key] for x in lb if x[key] is not None and not (isinstance(x[key], float) and math.isnan(x[key]))]
        if key == "azimuth_deg":                          # circular: compare the sorted gaps between successive leaves instead
            xa = _gaps(xa); xb = _gaps(xb)
        dist[key] = float(wasserstein_distance(xa, xb)) if xa and xb else None
    rank_hits = sum(1 for i, j, _ in m if la[i].get("rank") == lb[j].get("rank"))
    return dict(node_count=dict(a=sa["counts"]["main_nodes"], b=sb["counts"]["main_nodes"], error=sa["counts"]["main_nodes"] - sb["counts"]["main_nodes"]),
                leaf_count=dict(a=len(la), b=len(lb)), matched_leaves=len(m), unmatched_a=len(la) - len(ia), unmatched_b=len(lb) - len(ib),
                rank_agreement=(rank_hits / len(m)) if m else None, mean_match_cost=(sum(c for _, _, c in m) / len(m)) if m else None,
                leaflet_membership_agreement=(num / den) if den else None,
                leaflet_count=dict(a=sa["counts"]["leaflets"], b=sb["counts"]["leaflets"]), truss_count=dict(a=sa["counts"]["trusses"], b=sb["counts"]["trusses"]),
                descriptor_wasserstein=dist, pairs=[(la[i]["id"], lb[j]["id"], round(c, 1)) for i, j, c in m])


def _gaps(az: list[float]) -> list[float]:
    if len(az) < 2:
        return az
    s = sorted(az); return [((b - a) % 360) for a, b in zip(s, s[1:] + [s[0] + 360])]
