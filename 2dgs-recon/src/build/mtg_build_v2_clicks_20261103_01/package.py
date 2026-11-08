"""Package the click-skeleton MTGs as static set v2 (CONTRACT). python package.py"""
import csv, hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; FIT = HERE.parent / "mtg_fit_surfels_20261103_01"; OLD = HERE.parent / "mtg_build_20261102_01"
sys.path.insert(0, str(FIT))
import fit_plant as FP  # noqa: E402
import run_real as RR  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for d in ("static", "dynamic", "figures"):
    (HERE / d).mkdir(exist_ok=True)
man = dict(observed={}, copied={}, checks={})
for name in ("mtg_template_mean_v2draft_spec.json", "mtg_template_mean_v2draft_engine.plant_tree_v1.json", "mtg_template_mean_v2draft_engine.obj", "mtg_template_mean_v2draft_bmodel.csv",
             "mtg_template_mean_v2draft_bmodel.rows.json"):
    shutil.copy2(OLD / "static" / name, HERE / "static" / name); man["copied"][f"static/{name}"] = dict(sha256=sha(HERE / "static" / name), same_as_old=sha(OLD / "static" / name) == sha(HERE / "static" / name))
shutil.copytree(OLD / "static/mtg_template_mean_v2draft_engine", HERE / "static/mtg_template_mean_v2draft_engine", dirs_exist_ok=True)
for p in sorted((OLD / "dynamic").glob("*.csv")):
    shutil.copy2(p, HERE / "dynamic" / p.name); man["copied"][f"dynamic/{p.name}"] = dict(sha256=sha(HERE / "dynamic" / p.name), same_as_old=sha(p) == sha(HERE / "dynamic" / p.name))
rt_rows, organ = [], []
for pid in RR.PLANTS:
    src = FIT / f"fitted_clicks/mtg_{pid}_spec_fitted.json"; spec = json.load(open(src, encoding="utf-8"))
    spec["mtg_set"] = dict(version="static_v2_click_skeleton_20261103", source_spec=str(src.relative_to(HERE.parents[1])).replace("\\", "/"), source_sha256=sha(src))
    json.dump(spec, open(HERE / f"static/mtg_{pid}_spec.json", "w", encoding="utf-8"), indent=1, default=float)
    tree, rep, bsha, info = FP.write_fitted(spec, HERE / "static", f"mtg_{pid}"); rt = RR.engine_round_trip(tree, HERE / f"static/mtg_{pid}_engine")
    old_b = FIT / f"fitted_clicks/mtg_{pid}_clicks_bmodel.csv"; new_b = HERE / f"static/mtg_{pid}_bmodel.csv"
    same = bool(bsha) and old_b.exists() and sha(old_b) == sha(new_b)
    man["observed"][pid] = dict(spec_sha256=sha(HERE / f"static/mtg_{pid}_spec.json"), engine_tree_sha256=sha(HERE / f"static/mtg_{pid}_engine.plant_tree_v1.json"), bmodel=info["status"],
                                bmodel_identical_to_kaggle_checked=same if bsha else None)
    fc = sum(1 for lf in spec["leaves"] if lf.get("clicks_status") == "rachis_from_clicks"); kept = sum(1 for lf in spec["leaves"] if lf.get("clicks_status") == "node_link_outside_cloud_kept_U")
    rt_rows.append(dict(plant=pid, leaves=len(spec["leaves"]), rachis_from_clicks=fc, kept_surfel_fit=kept, other=len(spec["leaves"]) - fc - kept, engine_roundtrip=rt["passed"], bmodel=info["status"],
                        bmodel_roundtrip=("carried_over_identical_csv (mtgb-bm-clicks-20261103 7/7 exact)" if same else ("not_written_no_area" if not bsha else "CSV_DIFFERS_rerun_needed"))))
    for n in spec["nodes"]:
        organ.append(dict(plant=pid, organ="internode", rank=n["k"], internode_mm=n["internode_mm"], internode_source=n["internode_source"], stem_width_mm=n["stem_width_mm"], width_source=n["width_source"],
                          xyz_mm=json.dumps([round(x, 3) for x in n["xyz_mm"]])))
    for lf in spec["leaves"]:
        organ.append(dict(plant=pid, organ="compound_leaf", rank=lf["k"], uid=lf["uid"], status=lf.get("clicks_status", lf.get("fit_status", "")), azimuth_deg=lf["azimuth_deg"], azimuth_source=lf["azimuth_source"],
                          theta0_deg=lf["theta0_deg"], theta0_source=lf["theta0_source"], dtheta_deg=lf["dtheta_deg"], dtheta_source=lf["dtheta_source"], length_mm=lf["length_mm"], length_source=lf["length_source"],
                          leaflets=lf["leaflets"], leaflets_source=lf["leaflets_source"], area_mm2=lf["area_mm2"], area_source=lf["area_source"]))
    shutil.copy2(FIT / f"figures/clicks_{pid}.png", HERE / f"figures/mtg_{pid}_over_surfels.png")
    print(rt_rows[-1], flush=True)
for name, rows in (("roundtrip_v2.csv", rt_rows), ("static/static_mtg_table.csv", organ)):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
man["checks"] = dict(engine_roundtrip_pass=sum(r["engine_roundtrip"] for r in rt_rows), plants=len(rt_rows), bmodel_identical=sum(1 for r in rt_rows if r["bmodel_roundtrip"].startswith("carried")),
                     copied_identical=all(v["same_as_old"] for v in man["copied"].values()))
json.dump(man, open(HERE / "MANIFEST.json", "w", encoding="utf-8"), indent=1); print(json.dumps(man["checks"]))
