"""Static set v2 _02: _01 + e_* blade areas (CONTRACT). python package_02.py"""
import csv, hashlib, json, shutil, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; V01 = ROOT / "build/mtg_build_v2_clicks_20261103_01"; AREA = ROOT / "build/engine_static_area_e_20261103_01"
FIT = ROOT / "build/mtg_fit_surfels_20261103_01"; OLD = ROOT / "build/mtg_build_20261102_01"
sys.path.insert(0, str(FIT))
import fit_plant as FP  # noqa: E402
import run_real as RR  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
E = ["e_01", "e_04", "e_05", "e_07"]; man = dict(inputs={}, copied={}, e_plants={}, checks={})
for p in sorted(AREA.glob("*.csv")) + [AREA / "HASHES.json", AREA / "CONTRACT.md"]:
    man["inputs"][str(p.relative_to(ROOT)).replace("\\", "/")] = sha(p)
# 1. copy everything from _01 except the e_* files
for p in sorted(V01.rglob("*")):
    rel = p.relative_to(V01)
    if p.is_dir() or "__pycache__" in p.parts or rel.name.startswith("RUNNING.lock") or rel.name in ("FINAL_HASHES.json", "MANIFEST.json", "CONTRACT.md", "package.py", "roundtrip_v2.csv"):
        continue
    if any(f"mtg_{e}" in str(rel) for e in E):
        continue
    (HERE / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, HERE / rel); man["copied"][str(rel).replace("\\", "/")] = sha(p) == sha(HERE / rel)
# 2. e_* with areas
rt_rows = []; K = HERE / "kaggle/import_02"; (K / "scn").mkdir(parents=True, exist_ok=True)
for pid in E:
    rows = [r for r in csv.DictReader(l for l in open(AREA / f"area_b_{pid}.csv", encoding="utf-8") if not l.startswith("#"))]
    A = {r["leaf_uid"]: float(r["area_b_mm2"]) for r in rows}; N = {r["leaf_uid"]: int(float(r["n_surfels"])) for r in rows}
    spec = json.load(open(V01 / f"static/mtg_{pid}_spec.json", encoding="utf-8")); miss = []; zero = []
    for lf in spec["leaves"]:
        u = lf["uid"]
        if u not in A:
            miss.append(u); continue
        lf["area_mm2"] = A[u]; lf["area_source"]          # 0.0 for an unlabelled leaf: the converter skips that leaf's B-model row (n_01-L14 rule), the plant file is written = "recon_area_b_ungraded_labels; conditional_on_unvalidated_scale (engine_static_area_e_20261103_01)"; lf["area_n_surfels"] = N[u]
        if A[u] <= 0:
            zero.append(u); lf["flags"] = list(lf.get("flags", [])) + ["no_area_not_in_bmodel"]
    spec["area_source"] = f"build/engine_static_area_e_20261103_01/area_b_{pid}.csv (silhouette ratio x area (a); per leaf = sum of status-1 labelled surfels, lower bound)"
    spec["labels"] = list(dict.fromkeys(spec["labels"] + ["area_b_e_from_engine_static_area_e_20261103_01"])); spec["mtg_set"] = dict(version="static_v2_click_skeleton_20261103_02", source_spec=f"build/mtg_build_v2_clicks_20261103_01/static/mtg_{pid}_spec.json")
    json.dump(spec, open(HERE / f"static/mtg_{pid}_spec.json", "w", encoding="utf-8"), indent=1, default=float)
    tree, rep, bsha, info = FP.write_fitted(spec, HERE / "static", f"mtg_{pid}"); rt = RR.engine_round_trip(tree, HERE / f"static/mtg_{pid}_engine")
    if bsha:
        shutil.copy2(HERE / f"static/mtg_{pid}_bmodel.csv", K / "scn" / f"mtg_{pid}.csv")
    man["e_plants"][pid] = dict(leaves=len(spec["leaves"]), areas_set=len(spec["leaves"]) - len(miss) - len(zero), area_zero=zero, uid_missing=miss, bmodel=info["status"], engine_roundtrip=rt["passed"], plant_area_b_mm2=sum(A.values()))
    rt_rows.append(dict(plant=pid, leaves=len(spec["leaves"]), areas_set=man["e_plants"][pid]["areas_set"], area_zero=";".join(zero), engine_roundtrip=rt["passed"], bmodel=info["status"]))
    shutil.copy2(FIT / f"figures/clicks_{pid}.png", HERE / f"figures/mtg_{pid}_over_surfels.png"); print(rt_rows[-1], flush=True)
# 3. n_*/pilot B-model files for the 11-plant Kaggle check (byte-identical copies)
for pid in ("R3-1b", "n_01", "n_02", "n_03", "n_04", "n_05", "n_06"):
    shutil.copy2(HERE / f"static/mtg_{pid}_bmodel.csv", K / "scn" / f"mtg_{pid}.csv")
# organ table: _01 rows for the untouched plants + fresh rows for e_*
org = [r for r in csv.DictReader(open(V01 / "static/static_mtg_table.csv", encoding="utf-8")) if r["plant"] not in E]
for pid in E:
    spec = json.load(open(HERE / f"static/mtg_{pid}_spec.json", encoding="utf-8"))
    for n in spec["nodes"]:
        org.append(dict(plant=pid, organ="internode", rank=n["k"], internode_mm=n["internode_mm"], internode_source=n["internode_source"], stem_width_mm=n["stem_width_mm"], width_source=n["width_source"], xyz_mm=json.dumps([round(x, 3) for x in n["xyz_mm"]])))
    for lf in spec["leaves"]:
        org.append(dict(plant=pid, organ="compound_leaf", rank=lf["k"], uid=lf["uid"], status=lf.get("clicks_status", ""), azimuth_deg=lf["azimuth_deg"], azimuth_source=lf["azimuth_source"], theta0_deg=lf["theta0_deg"], theta0_source=lf["theta0_source"],
                        dtheta_deg=lf["dtheta_deg"], dtheta_source=lf["dtheta_source"], length_mm=lf["length_mm"], length_source=lf["length_source"], leaflets=lf["leaflets"], leaflets_source=lf["leaflets_source"], area_mm2=lf["area_mm2"], area_source=lf["area_source"]))
keys = []
for r in org:
    keys += [k for k in r if k not in keys]
with open(HERE / "static/static_mtg_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(org)
with open(HERE / "roundtrip_engine_e.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rt_rows[0])); w.writeheader(); w.writerows(rt_rows)
# Kaggle dataset + kernel
shutil.copy2(OLD / "kaggle/import_ds/work_model.tar.bin", K / "work_model.tar.bin")
with tarfile.open(K / "scenarios.tar.bin", "w") as t:
    for p in sorted((K / "scn").glob("*.csv")):
        t.add(p, arcname=p.name)
(K / "MTGB_IMPORT_MANIFEST.json").write_text(json.dumps(dict(inputs={n: sha(K / n) for n in ("work_model.tar.bin", "scenarios.tar.bin")}, core_sha256="35c605d65d9b54fe64cdf93bf28ae44ae7a1b6ded03b461dedcb0156a5c22147", note="static set v2 _02: 11 plants")), encoding="utf-8")
(K / "dataset-metadata.json").write_text(json.dumps(dict(title="mtgb-v2-02-20261104", id="dolgfish/mtgb-v2-02-20261104", licenses=[dict(name="other")], isPrivate=True)), encoding="utf-8")
kd = HERE / "kaggle/kernels/mtgb-bm-v2-02-20261104"; kd.mkdir(parents=True, exist_ok=True); shutil.copy2(OLD / "kaggle/kernel_import.py", kd / "kernel.py")
(kd / "kernel-metadata.json").write_text(json.dumps(dict(id="dolgfish/mtgb-bm-v2-02-20261104", title="mtgb-bm-v2-02-20261104", code_file="kernel.py", language="python", kernel_type="script", is_private=True, enable_gpu=False, enable_tpu=False,
                                                       enable_internet=False, dataset_sources=["dolgfish/mtgb-v2-02-20261104", "dolgfish/tomato-h-input"], competition_sources=[], kernel_sources=[])), encoding="utf-8")
man["checks"] = dict(copied_identical=all(man["copied"].values()), copied_n=len(man["copied"]), e_engine_roundtrip=sum(v["engine_roundtrip"] for v in man["e_plants"].values()), bmodel_files=len(list((K / "scn").glob("*.csv"))))
json.dump(man, open(HERE / "MANIFEST.json", "w", encoding="utf-8"), indent=1); print(json.dumps(man["checks"]))
