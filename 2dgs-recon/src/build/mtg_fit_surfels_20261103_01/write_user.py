"""CONTRACT U: adopt the first full fit (user decision). Spec copies with label user_accepted_despite_synthetic_fail -> engine + B-model files via the old
build's writers -> engine round trip; B-model scenarios + manifest + kernel for the Kaggle round trip. python write_user.py"""
import csv, hashlib, json, shutil, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_plant as FP  # noqa: E402
import run_real as RR  # noqa: E402

OUT = HERE / "fitted_user"; OUT.mkdir(exist_ok=True); K = HERE / "kaggle/import_user"; (K / "scn").mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = []
for pid in RR.PLANTS:
    src = HERE / f"fitted/mtg_{pid}_spec_fitted.json"; spec = json.load(open(src, encoding="utf-8"))
    spec["labels"] = [x for x in spec["labels"] if x != "display_only_synthetic_check_not_met"] + ["user_accepted_despite_synthetic_fail"]
    spec["fit"]["adoption"] = dict(decision="user 2026-11-03 'ㅇㅇ' after seeing fit_e_01.png (CONTRACT U)", source_spec=f"fitted/mtg_{pid}_spec_fitted.json", source_sha256=sha(src),
                                  warning="synthetic check failed (theta0 13.9 deg, dtheta 16.2 deg, L 29 %); fitted theta0 / dtheta / L are plant geometry, not traits")
    json.dump(spec, open(OUT / f"mtg_{pid}_spec_fitted.json", "w", encoding="utf-8"), indent=1, default=float)
    tree, rep, bsha, info = FP.write_fitted(spec, OUT, f"mtg_{pid}_fitted"); rt = RR.engine_round_trip(tree, OUT / f"mtg_{pid}_fitted_engine")
    if bsha:
        shutil.copy2(OUT / f"mtg_{pid}_fitted_bmodel.csv", K / "scn" / f"mtg_{pid}_fitted.csv")
    rows.append(dict(plant=pid, engine_organs=rep["organs"], engine_roundtrip=rt["passed"], rt_leaves_in=rt["leaves_in"], rt_leaves_back=rt["leaves_back"], rt_worst_seg_dir_deg=rt["worst_seg_dir"],
                     rt_worst_L_rel=rt["worst_L"], rt_worst_dtheta_deg=rt["worst_dtheta"], bmodel=info["status"] + ("" if info["status"] == "written" else ": " + info.get("reason", ""))))
    print(rows[-1], flush=True)
with open(HERE / "roundtrip_engine_user.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
shutil.copy2(FP.OLD / "kaggle/import_ds/work_model.tar.bin", K / "work_model.tar.bin")
with tarfile.open(K / "scenarios.tar.bin", "w") as t:
    for p in sorted((K / "scn").glob("*.csv")):
        t.add(p, arcname=p.name)
(K / "MTGB_IMPORT_MANIFEST.json").write_text(json.dumps(dict(inputs={n: sha(K / n) for n in ("work_model.tar.bin", "scenarios.tar.bin")}, core_sha256="35c605d65d9b54fe64cdf93bf28ae44ae7a1b6ded03b461dedcb0156a5c22147",
                                                              note="CONTRACT U: user-adopted fitted MTGs, B-model Level 1")), encoding="utf-8")
(K / "dataset-metadata.json").write_text(json.dumps(dict(title="mtgb-fit-user-20261103", id="dolgfish/mtgb-fit-user-20261103", licenses=[dict(name="other")], isPrivate=True)), encoding="utf-8")
kd = HERE / "kaggle/kernels/mtgb-bm-fit-user-20261103"; kd.mkdir(parents=True, exist_ok=True); shutil.copy2(FP.OLD / "kaggle/kernel_import.py", kd / "kernel.py")
(kd / "kernel-metadata.json").write_text(json.dumps(dict(id="dolgfish/mtgb-bm-fit-user-20261103", title="mtgb-bm-fit-user-20261103", code_file="kernel.py", language="python", kernel_type="script", is_private=True, enable_gpu=False,
                                                       enable_tpu=False, enable_internet=False, dataset_sources=["dolgfish/mtgb-fit-user-20261103", "dolgfish/tomato-h-input"], competition_sources=[], kernel_sources=[])), encoding="utf-8")
print("bmodel scenarios", len(list((K / "scn").glob("*.csv"))))
