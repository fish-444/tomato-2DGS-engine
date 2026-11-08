"""Pack the two Kaggle datasets + kernel folders (CONTRACT §5 venue). Import dataset: work_model.tar.bin (copied from the B-model geo7 dataset, hash
recorded) + scenarios.tar.bin (our B-model CSVs incl. masks and the one-row test). Tri dataset: *_triangles.csv + manifest."""
import hashlib, json, shutil, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
IMP = HERE / "import_ds"; TRI = HERE / "tri_ds"
src_tar = ROOT / "build/bmodel_mtg_20261101_01/kaggle/geo7_ds/work_model.tar.bin"
shutil.copy2(src_tar, IMP / "work_model.tar.bin")
with tarfile.open(IMP / "scenarios.tar.bin", "w") as t:
    for p in sorted((IMP / "scn").glob("*.csv")):
        t.add(p, arcname=p.name)
man = dict(inputs={"work_model.tar.bin": sha(IMP / "work_model.tar.bin"), "scenarios.tar.bin": sha(IMP / "scenarios.tar.bin")}, core_sha256="35c605d65d9b54fe64cdf93bf28ae44ae7a1b6ded03b461dedcb0156a5c22147",
           work_model_source=str(src_tar.relative_to(ROOT)).replace("\\", "/"), scenarios={p.name: sha(p) for p in sorted((IMP / "scn").glob("*.csv"))}, n_scenarios=len(list((IMP / "scn").glob("*.csv"))))
(IMP / "MTGB_IMPORT_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
(IMP / "dataset-metadata.json").write_text(json.dumps(dict(title="mtgb-import-20261102", id="dolgfish/mtgb-import-20261102", licenses=[dict(name="other")], isPrivate=True)), encoding="utf-8")
tm = dict(files={p.name: sha(p) for p in sorted(TRI.glob("*_triangles.csv"))})
(TRI / "MTGB_TRI_MANIFEST.json").write_text(json.dumps(tm, indent=1), encoding="utf-8")
(TRI / "dataset-metadata.json").write_text(json.dumps(dict(title="mtgb-tri-20261102", id="dolgfish/mtgb-tri-20261102", licenses=[dict(name="other")], isPrivate=True)), encoding="utf-8")
for name, code, ds in (("mtgb-bm-import-20261102", "kernel_import.py", ["dolgfish/mtgb-import-20261102", "dolgfish/tomato-h-input"]),
                       ("mtgb-light-20261102", "kernel_light.py", ["dolgfish/mtgb-tri-20261102", "dolgfish/static-tau-bundle-20261031", "dolgfish/static-nine-bundle-20261028"])):
    k = HERE / "kernels" / name; k.mkdir(parents=True, exist_ok=True); shutil.copy2(HERE / code, k / "kernel.py")
    (k / "kernel-metadata.json").write_text(json.dumps(dict(id=f"dolgfish/{name}", title=name, code_file="kernel.py", language="python", kernel_type="script", is_private=True, enable_gpu=False, enable_tpu=False,
                                                           enable_internet=False, dataset_sources=ds, competition_sources=[], kernel_sources=[])), encoding="utf-8")
print(json.dumps(dict(import_inputs=man["inputs"], n_scenarios=man["n_scenarios"], n_tri=len(tm["files"])), indent=1))
