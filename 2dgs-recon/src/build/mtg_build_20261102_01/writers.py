"""MTG spec -> (a) plant_tree_v1 + engine Stage-2 files, (b) B-model static scenario CSV (CONTRACT §5).
Leaf geometry from the spec parameters only (theta0, dtheta, L, azimuth; engine convention vtc_development_native.rgg:206-209 +
fsp_extensions_native.rgg:19-24): petiole straight at theta0 (share PET_SHARE of L), rachis = m straight segments with elevation
theta0 - dtheta * cumulative weight (w1 = 1/(2m-1), wi = 2/(2m-1)). Leaflets: flat param_blade helpers (user_assumed shape, never a trait).
B-model rows via tools/bmodel_mtg_export.static_rows (A1 area priority) on an obs-like dict; side axes from the published branch rows
with our angles (as the B-model session's template builder)."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from tomato3d.plant_tree import engine as PE, geometry as g, obj as PO  # noqa: E402
from tomato3d.plant_tree.schema import new_tree, validate  # noqa: E402
from tools import bmodel_mtg_export as BX  # noqa: E402
from mtg_spec import PUB_1_50  # noqa: E402

PET_SHARE = 0.30            # petiole share of the compound-leaf length (shape helper, user_assumed; engine PETIOLE_M/RACHIS_M analogue)
PAIR_ANGLE_DEG = 45.0       # lateral leaflet angle from the rachis tangent (helper)
SHARES = dict(T=1.5, P=1.0, I=0.5)   # leaflet area shares (helper)
LW = 1.5                    # leaflet L/W (measured range 1.0-2.2, mean about 1.5; SCHEMA.md mismatch table)
FIRST_PAIR_S = 0.35         # first pair at 0.35 L (helper)
_KAPPA = None


def kappa():
    """blade_mesh area for L = W = 1 (flat): area(L, W) = kappa * L * W."""
    global _KAPPA
    if _KAPPA is None:
        _, _, a = g.blade_mesh([0, 0, 0], [1, 0, 0], [0, 0, 1], 1.0, 1.0, 0.0, 0.0, 12, 7); _KAPPA = float(a)
    return _KAPPA


def bend_weights(m):
    return [1 / (2 * m - 1)] + [2 / (2 * m - 1)] * (m - 1) if m >= 1 else []


def pairs_of(leaflets):
    """leaflets -> (pairs, intercalary): 5 -> (2, 0), 7 -> (3, 0), 6 -> (2, 1), 3 -> (1, 0)."""
    n = int(leaflets); p = (n - 1) // 2; return p, (n - 1) % 2


def leaf_polyline(base, az_deg, theta0, dtheta, L, m):
    """Vertices: base, petiole tip (= pair 1), pair 2.., tip. All in the vertical plane of the azimuth."""
    az = math.radians(az_deg); h = np.array([math.cos(az), math.sin(az), 0.0]); up = np.array([0, 0, 1.0])
    def d(e):
        e = math.radians(e); return h * math.cos(e) + up * math.sin(e)
    pts = [np.asarray(base, float)]; p = pts[0] + d(theta0) * (PET_SHARE * L); pts.append(p)
    seg = (1 - PET_SHARE) * L / max(m, 1); e = theta0; elevs = [theta0]
    for w in bend_weights(max(m, 1)):
        e = e - dtheta * w; p = p + d(e) * seg; pts.append(p); elevs.append(e)
    return [q.tolist() for q in pts], elevs


def add_leaf(tree, lid, node_id, axis_id, rank, base, az, theta0, dtheta, L, leaflets, area, attrs, flags=()):
    m, n_inter = pairs_of(leaflets); pl, elevs = leaf_polyline(base, az, theta0, dtheta, L, m)
    cum = [0.0]
    for a, b in zip(pl, pl[1:]):
        cum.append(cum[-1] + float(np.linalg.norm(np.subtract(b, a))))
    rad = [0.6] * len(pl)
    leaf = dict(id=lid, node=node_id, axis=axis_id, rank=rank, petiole=dict(kind="tube", polyline=[pl[0], pl[1]], radius=[0.6, 0.6], segs=6), rachis=dict(kind="tube", polyline=pl, radius=rad, segs=6),
                attachments=[], length_mm=float(cum[-1]), azimuth_deg=float(az % 360), elevation_deg=float(theta0), age_d=None, expansion=None, flags=list(flags), attrs=dict(attrs, dtheta_deg=dtheta, pairs=m, intercalary=n_inter))
    tree["leaves"].append(leaf)
    if area is None:          # structural placeholders only: zero-area leaflets keep the pairs / rachis segments for the engine mapping; no area is invented
        leaf["attrs"]["leaflets_geometry"] = "area unavailable: zero-area placeholder leaflets (structure only, requires_declaration); count from attrs.leaflets"; leaf["attrs"]["leaflets"] = leaflets
        leaf["flags"].append("requires_declaration"); area = 1e-6
    units = m * 2 * SHARES["P"] + SHARES["T"] + n_inter * SHARES["I"]; a_unit = area / units
    def blade(name, s_idx, side, typ, pair, share):
        p = np.asarray(pl[s_idx], float); t = np.asarray(pl[min(s_idx + 1, len(pl) - 1)], float) - np.asarray(pl[max(s_idx - 1, 0) if s_idx == len(pl) - 1 else s_idx], float)
        t = t / (np.linalg.norm(t) or 1); up = np.array([0, 0, 1.0]); sd = np.cross(t, up); sd = sd / (np.linalg.norm(sd) or 1)
        alpha = math.radians(PAIR_ANGLE_DEG); dvec = t if side == 0 else (t * math.cos(alpha) + sd * side * math.sin(alpha)); dvec = dvec / np.linalg.norm(dvec)
        nvec = up - dvec * float(np.dot(up, dvec)); nvec = nvec / (np.linalg.norm(nvec) or 1)
        A = share * a_unit; Lf = math.sqrt(A * LW / kappa()); Wf = Lf / LW
        lt = dict(id=f"{lid}__{name}", leaf=lid, type=typ, side="C" if side == 0 else ("L" if side > 0 else "R"), pair=pair, attachment_s_mm=float(cum[s_idx]), base=p.tolist(),
                  frame=dict(d=dvec.tolist(), n=nvec.tolist(), b=np.cross(nvec, dvec).tolist()), length_mm=Lf, width_mm=Wf, area_mm2=None, normal=nvec.tolist(),
                  surface=dict(kind="param_blade", base=p.tolist(), d=dvec.tolist(), n=nvec.tolist(), L=Lf, W=Wf, cup=0.0, bend=0.0, nu=12, nv=7), petiolule=None, age_d=None, expansion=None, flags=[])
        _, _, ar = g.blade_mesh(lt["base"], dvec.tolist(), nvec.tolist(), Lf, Wf, 0.0, 0.0, 12, 7); lt["area_mm2"] = float(ar)
        tree["leaflets"].append(lt); return lt
    atts = {}
    for j in range(1, m + 1):
        for side, nm in ((1, "L"), (-1, "R")):
            lt = blade(f"P{j}{nm}", j, side, "P", j, SHARES["P"]); atts.setdefault(("P", j), dict(s_mm=lt["attachment_s_mm"], leaflets=[]))["leaflets"].append(lt["id"])
    for q in range(n_inter):
        lt = blade(f"I{q + 1}", 1, 1 if q % 2 == 0 else -1, "I", 1, SHARES["I"]); atts.setdefault(("I", q + 1), dict(s_mm=lt["attachment_s_mm"] + 0.01, leaflets=[]))["leaflets"].append(lt["id"])
    lt = blade("T", len(pl) - 1, 0, "T", 0, SHARES["T"]); atts[("T", 0)] = dict(s_mm=lt["attachment_s_mm"], leaflets=[lt["id"]])
    leaf["attachments"] = sorted(atts.values(), key=lambda a: a["s_mm"])
    if n_inter:
        leaf["flags"].append("intercalary_as_extra_leaflet")
    return leaf


def spec_to_tree(spec):
    pid = spec["plant_id"]; obs = spec["kind"] == "observed"
    tree = new_tree("reconstruction" if obs else "synthetic", "mm", "board frame mm, z up" if obs else "plant_local_mm, z up", f"mtg_from_traits:{spec['kind']}", plant_id=pid, synthetic=not obs)
    tree["provenance"].update(geometry="reconstructed" if obs else "assumption", theta0="user_measured", dtheta="reconstructed", leaflets="user_measured", internodes="reconstructed" if obs else "assumption")
    tree["plant"]["attrs"].update(labels=spec["labels"], age_dag=spec["age_dag"], age_source=spec["age_source"], kind=spec["kind"], mtg_build="build/mtg_build_20261102_01",
                                  shape_helpers=dict(PET_SHARE=PET_SHARE, PAIR_ANGLE_DEG=PAIR_ANGLE_DEG, SHARES=SHARES, LW=LW, note="user_assumed shape helpers for triangles only, not traits"))
    if obs:
        C = [list(map(float, p)) for p in spec["axis"]["curve_mm"]]
    else:
        C = [[0.0, 0.0, 0.0]] + [[0.0, 0.0, float(n["s_mm"])] for n in spec["nodes"]]
    widths = [n["stem_width_mm"] for n in spec["nodes"]]; r0 = (widths[0] if widths else 2.0) / 2
    tree["axes"].append(dict(id="M", kind="main", parent_axis=None, parent_node=None, tube=dict(kind="tube", polyline=C, radius=[float(r0)] * len(C), segs=8), flags=[], attrs=dict(source=spec["axis"]["source"])))
    for n in spec["nodes"]:
        nid = f"N{n['k']:02d}"
        tree["nodes"].append(dict(id=nid, axis="M", index=n["k"], rank=n["k"], s_mm=float(n["s_mm"]), position=[float(x) for x in n["xyz_mm"]], leaves=[], trusses=[], flags=[],
                                  attrs=dict(internode_mm=n["internode_mm"], internode_source=n["internode_source"], stem_width_mm=n["stem_width_mm"], width_source=n["width_source"])))
    nodes = {n["index"]: n for n in tree["nodes"]}
    for lf in spec["leaves"]:
        lid = f"L{lf['k']:02d}"; nodes[lf["node_k"]]["leaves"].append(lid)
        base = lf["base_mm"] if obs else nodes[lf["node_k"]]["position"]
        attrs = {k: lf[k] for k in lf if k.endswith("_source") or k in ("uid", "skeleton_elevation_deg", "leaflets", "geometry", "theta0_sd_deg")}
        attrs["area_mm2_input"] = lf["area_mm2"]
        add_leaf(tree, lid, nodes[lf["node_k"]]["id"], "M", lf["k"], base, lf["azimuth_deg"], lf["theta0_deg"], lf["dtheta_deg"], lf["length_mm"], lf["leaflets"], lf["area_mm2"], attrs,
                 flags=[f for f in lf["flags"] if f in ("requires_declaration", "unattached", "stub")] + (["requires_declaration"] if lf["area_mm2"] is None else []))
        tree["leaves"][-1]["attrs"]["spec_flags"] = lf["flags"]
    for sa in spec.get("side_axes", []):
        sid = f"S{sa['j']}"; pn = nodes[sa["attach_k"]]; az = math.radians(sa["azimuth_deg"]); e = math.radians(sa["angle_above_h_deg"])
        dvec = np.array([math.cos(az) * math.cos(e), math.sin(az) * math.cos(e), math.sin(e)]); p = np.asarray(pn["position"], float); pl = [p.tolist()]; s = 0.0
        tree["axes"].append(dict(id=sid, kind="side", parent_axis="M", parent_node=pn["id"], tube=dict(kind="tube", polyline=None, radius=None, segs=6), flags=["lateral_shoot_mapped_to_sympodial_axis"],
                                 attrs={k: v for k, v in sa.items() if k != "leaves"}))
        for i, (il, lf) in enumerate(zip(sa["internodes_mm"], sa["leaves"]), 1):
            p = p + dvec * il; s += il; pl.append(p.tolist()); nid = f"{sid}_N{i:02d}"
            tree["nodes"].append(dict(id=nid, axis=sid, index=i, rank=i, s_mm=s, position=p.tolist(), leaves=[], trusses=[], flags=[], attrs=dict(internode_mm=il, internode_source=sa["internode_source"])))
            lid = f"{sid}_L{i:02d}"; tree["nodes"][-1]["leaves"].append(lid)
            add_leaf(tree, lid, nid, sid, i, p.tolist(), (sa["azimuth_deg"] + (180 if i % 2 == 0 else 0)) % 360, lf["theta0_deg"], lf["dtheta_deg"], lf["length_mm"], lf["leaflets"], lf["area_mm2"],
                     {k: lf[k] for k in lf if k.endswith("_source")} | dict(area_mm2_input=lf["area_mm2"]), flags=[])
            tree["leaves"][-1]["attrs"]["spec_flags"] = lf.get("flags", [])
        tree["axes"][-1]["tube"].update(polyline=pl, radius=[1.0] * len(pl))
    tree["notes"] += spec.get("notes", [])
    return validate(tree)


def write_engine(spec, out_dir: Path, name):
    tree = spec_to_tree(spec); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_engine.plant_tree_v1.json").write_text(json.dumps(tree, indent=1, default=float), encoding="utf-8")
    rep = PE.to_engine_structure(tree, out_dir / f"{name}_engine", plant_instance_id="p1", source_plant_id=f"mtg_{spec['plant_id']}", fixture=spec["kind"] != "observed", capture_id=f"mtg_build_{spec['plant_id']}")
    (out_dir / f"{name}_engine.obj").write_text(PO.to_obj(tree, header=f"mtg_from_traits {spec['plant_id']} ({spec['kind']}); mm, z up"), encoding="utf-8")
    return tree, rep


# ------------------------------------------------------------------ B-model
def _sla(block):
    try:
        return float(BX.block_value(block, "SLA FACTOR_1")) * float(BX.block_value(block, "PLANT_DENSITY")) + float(BX.block_value(block, "SLA FACTOR_2"))
    except KeyError:
        dyn, _ = BX.read_block(ROOT / "build/bmodel_mtg_20261101_01/upstream/model/input/dynamic_input/base_plant_dynamic_50.csv", BX.DYNAMIC_HEAD)
        return float(BX.block_value(dyn, "SLA FACTOR_1")) * float(BX.block_value(dyn, "PLANT_DENSITY")) + float(BX.block_value(dyn, "SLA FACTOR_2"))


def spec_to_bmodel(spec, out_csv: Path, only_leaf=None):
    """Static scenario (published 1_50 block). Leaves without area -> not written (reported). Returns (sha, rows_json)."""
    block, _ = BX.read_block(PUB_1_50, BX.STATIC_HEAD); sla = _sla(block)
    missing = [lf["uid"] for lf in spec["leaves"] if lf["area_mm2"] is None]
    if missing:
        return None, dict(status="not_written", reason="blade area unavailable for leaves " + ",".join(missing), plant=spec["plant_id"])
    obs = dict(plant_id=spec["plant_id"], axis_curve_mm=spec["axis"]["curve_mm"] if spec["kind"] == "observed" else [[0, 0, 0]] + [[0, 0, n["s_mm"]] for n in spec["nodes"]],
               stem_base_mm=spec["axis"]["base_mm"] if spec["kind"] == "observed" else [0.0, 0.0, 0.0],
               nodes=[dict(xyz_mm=n["xyz_mm"], internode_mm=n["internode_mm"], internode_provenance=n["internode_source"]) for n in spec["nodes"]],
               leaves=[dict(leaf_uid=lf["uid"], model_rank=lf["k"], length_mm=lf["length_mm"], azimuth_deg=lf["azimuth_deg"], elevation_deg=lf["theta0_deg"], area_b_mm2=lf["area_mm2"],
                            flags=list(lf["flags"]), geometry=lf["geometry"]) for lf in spec["leaves"]])
    rows, prov = BX.static_rows(obs, block, sla, only_leaf=only_leaf)
    # hand stem thickness -> internode width (column 16), template-mean sd -> column 13
    by_rank = {n["k"]: n for n in spec["nodes"]}; lf_by = {lf["k"]: lf for lf in spec["leaves"]}
    for r, p in zip(rows, prov):
        if p.get("kind") == "internode" and p.get("rank") in by_rank:
            r[16] = BX._f(by_rank[p["rank"]]["stem_width_mm"] / 1000); p["width"] = by_rank[p["rank"]]["width_source"]
        if p.get("kind") == "leaf":
            lf = lf_by[p["rank"]]; p.update(theta0_source=lf["theta0_source"], dtheta="not_representable (importer has no per-leaf curvature column; published Curvature_LEAF=90 -> straight)", area_source=lf["area_source"], length_source=lf["length_source"])
            if spec["kind"] == "template_mean" and lf.get("theta0_sd_deg"):
                r[13] = BX._f(lf["theta0_sd_deg"]); p["inclination_sd"] = "hand_measured theta0 sd (template-mean: the model draws normal(mean, sd) itself; the mean plant for light uses sd 0)"
    if spec.get("side_axes"):
        bi, bl = BX.published_branch_rows(PUB_1_50)
        for sa in spec["side_axes"]:
            ori = BX._f((sa["azimuth_deg"] + BX.ORI_OFFSET) % 360)
            for i, lf in enumerate(sa["leaves"], 1):
                b = list(bi.get(i) or bi[max(bi)]); b[2] = "xI" if i == 1 else f"I{i}"; b[8] = str(i); b[9] = "1"; b[10] = ori; b[12] = BX._f(90 - sa["angle_above_h_deg"]) if i == 1 else b[12]
                b[14] = BX._f(sa["internodes_mm"][i - 1] / 1000)
                rows.append(b); prov.append(dict(kind="side_internode", attach_rank=sa["attach_k"], rank=i, length=sa["internode_source"], inclination=sa["angle_source"] if i == 1 else "model_published"))
                q = list(bl.get(i) or bl[max(bl)]); q[3] = "xL"; q[8] = str(i); q[9] = "1"; q[12] = BX._f(90 - lf["theta0_deg"]); q[13] = "0"; q[17] = BX._f(lf["area_mm2"] / 1e6)
                rows.append(q); prov.append(dict(kind="side_leaf", attach_rank=sa["attach_k"], rank=i, inclination=lf["theta0_source"], area=lf["area_source"], length="model_published petiole/rachis (side leaf length not mapped)"))
        # side rows must follow their attach internode in the published order; the importer reads them after the main rows (B-model session's template v1 round trip PASS with this order)
    sha = BX.write_scenario(out_csv, block, rows, prov, dict(plant=spec["plant_id"], kind=spec["kind"], labels=spec["labels"], conventions="A4: ori = az + 90, inc = 90 - theta0; A1 area priority; no dtheta"))
    return sha, dict(status="written", rows=len(rows), sha256=sha)


def _side_rows_ok():
    return True


if __name__ == "__main__":
    from mtg_spec import template_mean_spec
    s = template_mean_spec(); t, rep = write_engine(s, HERE / "static", "mtg_template_mean_v2draft"); print(rep["organs"], rep["stage2_vertices"])
    print(spec_to_bmodel(s, HERE / "static/mtg_template_mean_v2draft_bmodel.csv"))
