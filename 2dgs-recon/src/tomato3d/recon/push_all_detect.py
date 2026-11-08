"""Stage the two Kaggle datasets and the kernel for the label-view detectors (CONTRACT p1_all_compound Amendment 1). Local work is copying and pushing only.
usage: python -m tomato3d.recon.push_all_detect --stage | --push-data | --push-kernel | --status | --download"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "build/p1_all_compound_20261026_01"
SP = Path("<SCRATCH_DIR>/scratchpad")
CODE, JOB, KER = SP / "detect_code", SP / "detect_job", SP / "detect_kernel"
IDS = ["n_04", "t2_02", "n_01", "t2_05", "n_06", "n_05", "n_02"]
SLUG = "dolgfish/tomato-all-detect"


def kg(*a):
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True, encoding="utf-8", errors="replace"); return r.returncode, r.stdout + r.stderr


def stage():
    shutil.rmtree(CODE, ignore_errors=True); CODE.mkdir(parents=True)
    for d in ("tomato3d", "tools"):
        shutil.copytree(ROOT / d, CODE / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "build/p1_t1_occlusion_20261017_01/parameters.json", CODE / "t1_parameters.json")
    (CODE / "dataset-metadata.json").write_text(json.dumps(dict(title="tomato-all-detect-code", id="dolgfish/tomato-all-detect-code", licenses=[dict(name="CC0-1.0")])))
    shutil.rmtree(JOB, ignore_errors=True); JOB.mkdir(parents=True)
    for i in IDS:
        for sub in ("undist_3024", "masks", "rf_detections", "rf_detections_tiled"):
            if (B / i / sub).exists():
                shutil.copytree(B / i / sub, JOB / i / sub)
        shutil.copy(B / i / "detect_input.json", JOB / i / "detect_input.json")
    assert not any("api_key" in p.read_text(encoding="utf-8", errors="ignore") for p in JOB.rglob("*.json")), "a key-like string is in the job files"
    (JOB / "dataset-metadata.json").write_text(json.dumps(dict(title="tomato-all-detect-job", id="dolgfish/tomato-all-detect-job", licenses=[dict(name="CC0-1.0")])))
    KER.mkdir(exist_ok=True); shutil.copy(ROOT / "tomato3d/recon/kernel_all_detect.py", KER / "run.py")
    (KER / "kernel-metadata.json").write_text(json.dumps(dict(id=SLUG, title="tomato-all-detect", code_file="run.py", language="python", kernel_type="script", is_private=True, enable_gpu=True, machine_shape="NvidiaTeslaT4",
                                                              enable_internet=True, dataset_sources=["dolgfish/tomato-all-detect-code", "dolgfish/tomato-all-detect-job", "dolgfish/tomato-recon-code", "dolgfish/tomato-mask-input"], competition_sources=[], kernel_sources=[])))
    print("staged", sum(1 for _ in JOB.rglob("*") if _.is_file()), "job files")


def push_data():
    for d, slug in ((CODE, "dolgfish/tomato-all-detect-code"), (JOB, "dolgfish/tomato-all-detect-job")):
        rc, o = kg("datasets", "create", "-p", str(d), "--dir-mode", "zip"); print(slug, rc, o.strip()[-300:])
        if rc != 0 and "already" in o.lower():
            rc, o = kg("datasets", "version", "-p", str(d), "-m", "label views " + time.strftime("%m-%d %H:%M"), "--dir-mode", "zip"); print(slug, "version", rc, o.strip()[-300:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("stage", "push-data", "push-kernel", "status", "download"):
        ap.add_argument("--" + k, action="store_true")
    a = ap.parse_args()
    if a.stage: stage()
    if a.push_data: push_data()
    if a.push_kernel: print(kg("kernels", "push", "-p", str(KER)))
    if a.status: print(kg("kernels", "status", SLUG))
    if a.download:
        out = B / "kernel_out"; out.mkdir(exist_ok=True); print(kg("kernels", "output", SLUG, "-p", str(out), "-q"))
