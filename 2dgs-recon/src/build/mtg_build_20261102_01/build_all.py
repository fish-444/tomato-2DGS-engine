"""Local build (CONTRACT §5-6, light local, no JVM): every allowed plant + template mean + 16 check plants (x2 internode arms) -> both formats,
organ table with sources, keyed draws, distribution check, one-row engine test, engine round trip, triangle CSVs for the Kaggle tracer, B-model
scenario masks for the Kaggle import kernel, excluded list."""
from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from tomato3d.plant_tree import engine as PE, geometry as g, obj as PO  # noqa: E402
import mtg_spec as MS  # noqa: E402
import writers as W  # noqa: E402
import dynamic_rules as DR  # noqa: E402

STATIC = HERE / "static"; CHECK = HERE / "dynamic/check_plants"; KAG = HERE / "kaggle"; TESTS = HERE / "tests"
for d in (STATIC, CHECK, KAG / "import_ds/scn", KAG / "tri_ds", TESTS):
    d.mkdir(parents=True, exist_ok=True)
SCENE_XY = np.array([0.280, 0.210]); MEDIUM_Z = 0.060
REPORT = dict(plants={}, check_plants={}, notes=[])


# ------------------------------------------------------------------ triangles for the tracer
def triangles_csv(tree, spec, out):
    """organ,class,leaf,order,rank,x0..z2 in metres, grow frame: observed plants keep board z and move the stem-base xy to SCENE_XY (as obs.Placed);
    template / check plants (plant_local, base z = 0) put the base at MEDIUM_Z (as A7 / PlacedModel)."""
    main = next(a for a in tree["axes"] if a["kind"] == "main"); base = np.asarray(main["tube"]["polyline"][0], float)
    off = np.array([SCENE_XY[0] - base[0] / 1000, SCENE_XY[1] - base[1] / 1000, 0.0 if spec["kind"] == "observed" else MEDIUM_Z - base[2] / 1000])
    rows = []; leaf_rank = {lf["id"]: lf["rank"] for lf in tree["leaves"]}; leaf_axis = {lf["id"]: lf["axis"] for lf in tree["leaves"]}
    for o in PO.organ_objects(tree):
        V = np.asarray(o["V"], float) / 1000 + off; cls = {"stem": "Stem", "rachis": "Petiole", "petiolule": "Petiole", "truss": "Stem"}.get(o["cat"], "Leaflet")
        lf = o["leaf"]; order = 0 if (not lf or leaf_axis.get(lf) == "M") else 1; rank = leaf_rank.get(lf, 0) if lf else 0
        leaf_no = (rank if order == 0 else 100 * int(leaf_axis[lf][1:]) + rank) if lf else 0
        for f in o["F"]:
            t = V[list(f)]; rows.append([o["name"], cls, leaf_no, order, rank] + [f"{x:.9f}" for x in t.reshape(-1)])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# mtg_from_traits {spec['plant_id']} ({spec['kind']}) grow frame metres; base -> SCENE_XY; {'board z kept' if spec['kind'] == 'observed' else 'base z = MEDIUM_Z'}\n")
        w = csv.writer(fh); w.writerow(["organ", "class", "leaf", "order", "rank", "x0", "y0", "z0", "x1", "y1", "z1", "x2", "y2", "z2"]); w.writerows(rows)
    return len(rows)


# ------------------------------------------------------------------ engine round trip
def engine_round_trip(tree, stage_dir):
    back = PE.from_engine_states(stage_dir / "initial_state.csv", 0, "p1")
    A = {lf["id"]: lf for lf in tree["leaves"]}; main_in = [lf for lf in tree["leaves"] if lf["axis"] == "M"]
    rows = []; ok = True
    nodes_in = sorted([n for n in tree["nodes"] if n["axis"] == "M"], key=lambda n: n["index"]); nodes_back = sorted([n for n in back["nodes"] if n["axis"] == "A0"], key=lambda n: n["index"])
    cnt = dict(leaves_in=len(tree["leaves"]), leaves_back=len(back["leaves"]), main_nodes_in=len(nodes_in), main_nodes_back=len(nodes_back), order_same=[n["index"] for n in nodes_in] == [n["index"] for n in nodes_back])
    ok &= cnt["leaves_in"] == cnt["leaves_back"] and cnt["main_nodes_in"] == cnt["main_nodes_back"] and cnt["order_same"]
    rep = json.load(open(stage_dir / "mapping_report.json", encoding="utf-8")); inv = {v: k for k, v in rep["id_map"].items()}
    for lb in back["leaves"]:
        lid = inv.get(lb.get("engine_id"), None)
        if lid not in A:
            rows.append(dict(leaf=lb["id"], status="unmatched")); ok = False; continue
        la = A[lid]; pa = np.asarray(la["rachis"]["polyline"], float); pb = np.asarray(lb["rachis"]["polyline"], float)
        def first(pl):
            d = pl[1] - pl[0]; return math.degrees(math.atan2(d[1], d[0])) % 360, math.degrees(math.asin(d[2] / (np.linalg.norm(d) or 1)))
        az_a, el_a = first(pa); az_b, el_b = first(pb); L_a = g.polyline_length(pa); L_b = g.polyline_length(pb); t_a = g.polyline_turning_deg(pa); t_b = g.polyline_turning_deg(pb)
        r = dict(leaf=lid, axis=la["axis"], theta0_in=el_a, theta0_back=el_b, d_theta0=abs(el_a - el_b), dtheta_in=t_a, dtheta_back=t_b, d_dtheta=abs(t_a - t_b), L_in=L_a, L_back=L_b, dL_rel=abs(L_b / L_a - 1) if L_a else 0,
                 az_in=az_a, az_back=az_b, d_az=abs((az_a - az_b + 180) % 360 - 180), leaflets_in=sum(lt["leaf"] == lid for lt in tree["leaflets"]), leaflets_back=sum(lt["leaf"] == lb["id"] for lt in back["leaflets"]))
        r["pass"] = r["d_theta0"] <= 2 and r["d_dtheta"] <= 2 and r["dL_rel"] <= 0.01 and r["d_az"] <= 2 and r["leaflets_in"] == r["leaflets_back"]; ok &= r["pass"]; rows.append(r)
    gaps = lambda ls: [((b["azimuth_deg"] - a["azimuth_deg"]) % 360) for a, b in zip(ls, ls[1:])]
    ga = gaps(main_in); gb = gaps(sorted([lf for lf in back["leaves"] if lf["axis"] == "A0"], key=lambda l: int(l["id"][1:])))
    cnt["divergence_max_diff_deg"] = float(max([abs((x - y + 180) % 360 - 180) for x, y in zip(ga, gb)] or [0])); ok &= cnt["divergence_max_diff_deg"] <= 2
    return dict(counts=cnt, leaves=rows, pass_all=bool(ok))


def one_row_engine_test():
    spec = dict(plant_id="one_row_test", kind="test", labels=["test"], axis=dict(source="vertical 10 mm"), age_dag=60, age_source="test", flags=[], side_axes=[], trusses=[], notes=[],
                nodes=[dict(k=1, s_mm=10.0, xyz_mm=[0, 0, 10.0], internode_mm=10.0, internode_source="test", stem_width_mm=4.0, width_source="test")],
                leaves=[dict(k=1, uid="T-L01", node_k=1, base_mm=[0, 0, 10.0], azimuth_deg=0.0, azimuth_source="test", theta0_deg=30.0, theta0_source="test", dtheta_deg=0.0, dtheta_source="test", length_mm=40.0, length_source="test",
                             leaflets=5, leaflets_source="test", area_mm2=400.0, area_source="test", flags=[], geometry="test")])
    tree, rep = W.write_engine(spec, TESTS, "one_row")
    st = list(csv.DictReader(open(TESTS / "one_row_engine/initial_state.csv", encoding="utf-8")))
    cl = next(r for r in st if r["class"] == "CompoundLeaf"); pet = next(r for r in st if r["class"] == "Petiole"); segs = [r for r in st if r["class"] == "RachisSegment"]
    d = np.array([float(pet["tip_x_m"]) - float(pet["x_m"]), float(pet["tip_y_m"]) - float(pet["y_m"]), float(pet["tip_z_m"]) - float(pet["z_m"])])
    el = math.degrees(math.asin(d[2] / np.linalg.norm(d))); L = (float(pet["length_m"]) + sum(float(s["length_m"]) for s in segs)) * 1000
    out = dict(format="engine (plant_tree_v1 -> to_engine_structure initial_state.csv)", theta0_set=30.0, compound_leaf_elevation_deg=float(cl["elevation_deg"]), petiole_direction_elevation_deg=el, length_set_mm=40.0,
               petiole_plus_rachis_mm=L, rachis_segments=len(segs), pass_angle=abs(el - 30) <= 2 and abs(float(cl["elevation_deg"]) - 30) <= 2, pass_length=abs(L / 40 - 1) <= 0.01,
               convention="elevation above horizontal = theta0; engine COMPOUND_INSERTION_DEG = 90 - theta0 = 60 (vtc_development_native.rgg:206)")
    # B-model one-row scenario (Kaggle checks the exported geometry): rank-1 class, area 400 mm2 so the model minimum fits (CONTRACT §5)
    sha, info = W.spec_to_bmodel(spec, KAG / "import_ds/scn/one_row_test.csv")
    out["bmodel_scenario"] = dict(file="kaggle/import_ds/scn/one_row_test.csv", sha256=sha, expected="exported straight leaf elevation 30 +- 2 deg (inclination column 60), |tip - base| 40 mm +- 1 %, orientation column 90 (az 0 + 90)", info=info)
    return out


# ------------------------------------------------------------------ organ table
def organ_rows(spec, fmt_status):
    out = []
    for n, lf in zip(spec["nodes"], spec["leaves"]):
        out.append(dict(plant=spec["plant_id"], kind=spec["kind"], organ="internode", axis="M", rank=n["k"], internode_mm=n["internode_mm"], internode_source=n["internode_source"], stem_width_mm=n["stem_width_mm"], width_source=n["width_source"]))
        out.append(dict(plant=spec["plant_id"], kind=spec["kind"], organ="compound_leaf", axis="M", rank=lf["k"], uid=lf["uid"], azimuth_deg=lf["azimuth_deg"], azimuth_source=lf["azimuth_source"], theta0_deg=lf["theta0_deg"], theta0_source=lf["theta0_source"],
                        dtheta_deg=lf["dtheta_deg"], dtheta_source=lf["dtheta_source"], length_mm=lf["length_mm"], length_source=lf["length_source"], leaflets=lf["leaflets"], leaflets_source=lf["leaflets_source"], area_mm2=lf["area_mm2"], area_source=lf["area_source"],
                        skeleton_elevation_deg_metadata=lf.get("skeleton_elevation_deg"), flags=";".join(lf["flags"]), geometry=lf["geometry"], bmodel=fmt_status["bmodel"], engine=fmt_status["engine"]))
    for sa in spec.get("side_axes", []):
        out.append(dict(plant=spec["plant_id"], kind=spec["kind"], organ="side_axis", axis=f"S{sa['j']}", rank=sa["attach_k"], azimuth_deg=sa["azimuth_deg"], azimuth_source=sa["azimuth_source"], theta0_deg=sa["angle_above_h_deg"], theta0_source=sa["angle_source"],
                        internode_mm=";".join(f"{x:.2f}" for x in sa["internodes_mm"]), internode_source=sa["internode_source"]))
        for lf in sa["leaves"]:
            out.append(dict(plant=spec["plant_id"], kind=spec["kind"], organ="side_compound_leaf", axis=f"S{sa['j']}", rank=lf["i"], theta0_deg=lf["theta0_deg"], theta0_source=lf["theta0_source"], dtheta_deg=lf["dtheta_deg"], dtheta_source=lf["dtheta_source"],
                            length_mm=lf["length_mm"], length_source=lf["length_source"], leaflets=lf["leaflets"], leaflets_source=lf["leaflets_source"], area_mm2=lf["area_mm2"], area_source=lf["area_source"], flags=";".join(lf.get("flags", []))))
    for t in spec.get("trusses", []):
        out.append(dict(plant=spec["plant_id"], kind=spec["kind"], organ="truss", rank=t["j"], leaflets=t["flowers"], leaflets_source="flowers drawn RP_FLOWERS_PER_TRUSS", flags="not_placed_pending_user"))
    return out


def masks(full_csv: Path, name: str, sd0: bool):
    """One-leaf-visible scenario files for the B-model length round trip (CSV-level masks, as make_organ_scenarios)."""
    lines = full_csv.read_bytes().decode("utf-8").split("\n"); head = lines[:77]; body = lines[77:]
    rows = [(ln.split(",") + [""] * 33)[:33] if ln.strip() else None for ln in body]
    if sd0:
        for c in rows:
            if c and c[1:4] != ["", "", ""]:
                c[11] = "0" if c[11] else c[11]; c[13] = "0" if c[13] else c[13]
    leaf_idx = [i for i, c in enumerate(rows) if c and (c[2].startswith("xL") or c[3].startswith("xL"))]
    def write(fn, keep=None):
        out = []
        for i, c in enumerate(rows):
            if c is None:
                out.append(body[i]); continue
            c = list(c)
            if i in leaf_idx:
                c[9] = "1" if (keep is None or i == keep) else "0"
            out.append(",".join(c))
        (KAG / "import_ds/scn" / f"{fn}.csv").write_bytes(("\n".join(head + out)).encode("utf-8"))
    if sd0:
        write(f"{name}_sd0")
    write(f"{name}_stem", keep=-1)
    for k, i in enumerate(leaf_idx, 1):
        write(f"{name}_leaf{k:02d}", keep=i)
    return len(leaf_idx)


def run():
    organ_table = []; rt_rows = []; tri_index = {}; bm_index = {}
    def do(spec, out_dir, name, mask=False, sd0_mask=False):
        tree, rep = W.write_engine(spec, out_dir, name)
        sha, info = W.spec_to_bmodel(spec, out_dir / f"{name}_bmodel.csv")
        status = dict(engine=f"written ({rep['organs']} organs, {rep['stage2_vertices']} stage-2 vertices)", bmodel=info["status"] + ("" if info["status"] == "written" else ": " + info["reason"]))
        rt = engine_round_trip(tree, out_dir / f"{name}_engine"); rt_rows.append(dict(plant=spec["plant_id"], **rt["counts"], leaves_pass=sum(1 for r in rt["leaves"] if r.get("pass")), leaves_total=len(rt["leaves"]), pass_all=rt["pass_all"]))
        (out_dir / f"{name}_roundtrip_engine.json").write_text(json.dumps(rt, indent=1, default=float), encoding="utf-8")
        has_leaflets = any(True for _ in tree["leaflets"])
        if has_leaflets:
            tri_index[name] = triangles_csv(tree, spec, KAG / "tri_ds" / f"{name}_triangles.csv")
        if sha:
            import shutil; shutil.copy2(out_dir / f"{name}_bmodel.csv", KAG / "import_ds/scn" / f"{name}.csv"); bm_index[name] = dict(sha256=sha, masks=0)
            if mask:
                bm_index[name]["masks"] = masks(out_dir / f"{name}_bmodel.csv", name, sd0_mask)
        organ_table.extend(organ_rows(spec, status)); return tree, rep, status, rt
    # template mean
    tm = MS.template_mean_spec(); _, _, st, rt = do(tm, STATIC, "mtg_template_mean_v2draft", mask=True, sd0_mask=True); REPORT["plants"]["template_mean_v2draft"] = dict(status=st, roundtrip_engine=rt["pass_all"], leaves=len(tm["leaves"]), side_axes=len(tm["side_axes"]))
    json.dump(tm, open(STATIC / "mtg_template_mean_v2draft_spec.json", "w", encoding="utf-8"), indent=1, default=float)
    # observed plants
    for pid in MS.PLANTS:
        sp = MS.observed_spec(pid); nm = f"mtg_{pid}"; _, _, st, rt = do(sp, STATIC, nm, mask=pid in ("R3-1b", "n_04"))
        json.dump(sp, open(STATIC / f"{nm}_spec.json", "w", encoding="utf-8"), indent=1, default=float)
        REPORT["plants"][pid] = dict(status=st, roundtrip_engine=rt["pass_all"], leaves=len(sp["leaves"]), grade=MS.GRADE[pid], axis=sp["axis"]["source"], areas=sum(l["area_mm2"] is not None for l in sp["leaves"]), notes=sp["notes"], flags=sp["flags"])
    # dynamic rules + check plants
    R, ib = DR.rules(); specs = []
    for m in range(DR.N_CHECK):
        sp = DR.drawn_spec(m, ib, "beta"); specs.append(sp); nm = f"P{m + 1:02d}"; _, _, st, rt = do(sp, CHECK, nm, mask=m < 2)
        json.dump(sp, open(CHECK / f"{nm}_spec.json", "w", encoding="utf-8"), indent=1, default=float); REPORT["check_plants"][nm] = dict(status=st, roundtrip_engine=rt["pass_all"], leaves=len(sp["leaves"]), side_axes=len(sp["side_axes"]), trusses=len(sp["trusses"]), seed=sp["seed"])
    DR.write_draws(); n_draws = len(DR.DRAWS)
    dist = DR.distribution_check(specs, R)
    paired = []
    for m in range(DR.N_CHECK):
        sp = DR.drawn_spec(m, ib, "fixed375"); nm = f"P{m + 1:02d}_fixed375"; tree, rep, st, rt = do(sp, CHECK, nm)
        b = specs[m]; tb = json.load(open(CHECK / f"P{m + 1:02d}_engine.plant_tree_v1.json", encoding="utf-8"))
        def height(t):
            zs = [v[2] for lt in t["leaflets"] for v in (lt["base"],)] + [p[2] for lf in t["leaves"] for p in lf["rachis"]["polyline"]] + [p[2] for a in t["axes"] for p in a["tube"]["polyline"]]
            return max(zs)
        paired.append(dict(plant=f"P{m + 1:02d}", n_internodes=len(b["nodes"]), main_axis_mm_fixed375=b["nodes"][-1]["s_mm"] if False else sp["nodes"][-1]["s_mm"], main_axis_mm_beta=b["nodes"][-1]["s_mm"],
                           d_main_axis_mm=b["nodes"][-1]["s_mm"] - sp["nodes"][-1]["s_mm"], height_mm_fixed375=height(tree), height_mm_beta=height(tb), d_height_mm=height(tb) - height(tree),
                           **{f"node{k}_z_beta": (b["nodes"][k - 1]["s_mm"] if k <= len(b["nodes"]) else "") for k in range(1, 11)}, **{f"node{k}_z_fixed": (sp["nodes"][k - 1]["s_mm"] if k <= len(sp["nodes"]) else "") for k in range(1, 11)}))
    DR.DRAWS[:] = DR.DRAWS[:n_draws]
    with open(HERE / "dynamic/internode_paired_height.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
    # tables
    keys = []
    for r in organ_table:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(STATIC / "static_mtg_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(organ_table)
    with open(HERE / "roundtrip_engine.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rt_rows[0])); w.writeheader(); w.writerows(rt_rows)
    one = one_row_engine_test(); json.dump(one, open(TESTS / "one_row_tests.json", "w", encoding="utf-8"), indent=1, default=float)
    excl = [dict(item="t2_02,t2_04,t2_05", why="tray membership unconfirmed (addendum §3)"), dict(item="t2_01,t2_03", why="excluded by the user (LOG:1417)"), dict(item="e_02,e_03,e_06", why="no 3D-first groups / skeleton"),
            dict(item="validation trays 1,2", why="not read"), dict(item="fruits_per_plant", why="pending_user"), dict(item="7 held hand-data cells", why="not in the template"),
            dict(item="skeleton_theta0, skeleton_insertion_from_vertical", why="not_used (theta0 = hand only)"), dict(item="skeleton_divergence 112.7 / first-segment 91.9 / old 113 deg / folded 0-180 means", why="superseded_display_only / display_only"),
            dict(item="skeleton_internode 3.748 single value; v6 7.6 +- 7.6", why="replaced by the user setting range 2.00-4.00 median 3.75; v6 display_only"),
            dict(item="per-leaf skeleton traits (skeleton_traits_per_leaf_display_only.csv)", why="display_only"), dict(item="e_* B-model static MTG", why="blade area unavailable (no silhouette area for e_* surfels); engine format written with area unavailable"),
            dict(item="skeleton lateral axes", why="no leaf assignment to laterals in skeleton_v1.1; listed in the spec notes"), dict(item="trusses/flowers in static MTGs", why="node position not confirmed (pending_user); B-model static importer has no truss row")]
    with open(HERE / "excluded.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "why"]); w.writeheader(); w.writerows(excl)
    REPORT.update(internode_beta=ib, distribution_check=dict(rows=len(dist), passed=sum(r["passed"] for r in dist)), one_row_engine=one, triangles=tri_index, bmodel_scenarios=bm_index, draws=n_draws)
    json.dump(REPORT, open(HERE / "build_report.json", "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in REPORT.items() if k in ("distribution_check", "draws")}), "one_row", one["pass_angle"], one["pass_length"])
    for k, v in REPORT["plants"].items():
        print(k, v["status"]["bmodel"][:40], "| engine rt", v["roundtrip_engine"], "| leaves", v["leaves"])
    print("check plants rt", [v["roundtrip_engine"] for v in REPORT["check_plants"].values()])
    for r in dist:
        print("  dist", r["organ"], r["rank_class"], r["field"], "pass" if r["passed"] else "FAIL", round(float(r["max_dev"]), 3))


if __name__ == "__main__":
    run()
