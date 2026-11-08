"""Lane K driver: one Kaggle kernel per individual (own dataset, own kernel slug), one at a time, resumable.
Local work = copying, pushing, polling, downloading only (no PC RAM load). Waits for each cell's `prep` marker. Stop with a file STOP_K in the build folder.

usage: python -m tomato3d.organs.all3d_kaggle [--ids n_04 ...] [--layers h2,sa | rf]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from tomato3d.organs import all3d_first as A

SP = Path(os.environ.get("TEMP", "C:/Temp")) / "all3d_kaggle"
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def kg(*a):
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    return r.returncode, r.stdout + r.stderr


def slug_id(iid):
    return iid.replace("_", "")


def klog(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with open(A.OUT / "kaggle.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def stage_code():
    d = SP / "code"
    if (d / "dataset-metadata.json").exists():
        return d
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for x in ("tomato3d", "tools"):
        shutil.copytree(A.ROOT / x, d / x, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(A.ROOT / "build/p1_t1_occlusion_20261017_01/parameters.json", d / "t1_parameters.json")
    shutil.copy(A.OUT / "size_prior.json", d / "size_prior.json")
    (d / "dataset-metadata.json").write_text(json.dumps(dict(title="tomato-3dfirst-code", id="dolgfish/tomato-3dfirst-code", licenses=[dict(name="CC0-1.0")])))
    push_dataset(d, "tomato-3dfirst-code")
    return d


def push_dataset(d, slug):
    for attempt in range(4):
        rc, o = kg("datasets", "create", "-p", str(d), "--dir-mode", "zip")
        if rc != 0 and "already" in o.lower():
            rc, o = kg("datasets", "version", "-p", str(d), "-m", time.strftime("%m-%d %H:%M"), "--dir-mode", "zip")
        klog(f"dataset {slug} try {attempt + 1} rc {rc} {o.strip()[-120:]}")
        if rc == 0:
            return True
        time.sleep(120)
    return False


def run_individual(iid, layers):
    cell = A.cell(iid)
    out = cell / "cands"
    out.mkdir(exist_ok=True)
    flag = out / f"kaggle_{layers.replace(',', '_')}.done"
    if flag.exists():
        return True
    job = SP / f"job_{slug_id(iid)}"
    shutil.rmtree(job, ignore_errors=True)
    job.mkdir(parents=True)
    for sub in ("photos", "masks", "rf_detections", "rf_detections_tiled"):
        if (cell / sub).exists():
            shutil.copytree(cell / sub, job / sub)
    shutil.copy(cell / "detect_input.json", job / "detect_input.json")
    if any("api_key" in p.read_text(encoding="utf-8", errors="ignore") for p in job.rglob("*.json")):
        klog(f"{iid}: key-like string found in the job files, aborting this individual")
        return False
    js = f"tomato-3dfirst-job-{slug_id(iid)}"
    (job / "dataset-metadata.json").write_text(json.dumps(dict(title=js, id=f"dolgfish/{js}", licenses=[dict(name="CC0-1.0")])))
    if not push_dataset(job, js):
        return False
    ker = SP / f"kernel_{slug_id(iid)}"
    shutil.rmtree(ker, ignore_errors=True)
    ker.mkdir(parents=True)
    src = (A.ROOT / "tomato3d/recon/kernel_3dfirst_cands.py").read_text(encoding="utf-8")
    src = re.sub(r'^IID = ".*"$', f'IID = "{iid}"', src, flags=re.M)
    src = re.sub(r'^LAYERS = ".*"$', f'LAYERS = "{layers}"', src, flags=re.M)
    (ker / "run.py").write_text(src, encoding="utf-8")
    kslug = f"dolgfish/tomato-3dfirst-cands-{slug_id(iid)}"
    meta = dict(id=kslug, title=f"tomato-3dfirst-cands-{slug_id(iid)}", code_file="run.py", language="python", kernel_type="script", is_private=True, enable_gpu=True,
                machine_shape="NvidiaTeslaT4", enable_internet=True, dataset_sources=["dolgfish/tomato-3dfirst-code", f"dolgfish/{js}", "dolgfish/tomato-recon-code", "dolgfish/tomato-mask-input"],
                competition_sources=[], kernel_sources=[])
    (ker / "kernel-metadata.json").write_text(json.dumps(meta))
    for attempt in range(6):
        rc, o = kg("kernels", "push", "-p", str(ker))
        klog(f"push {iid} {layers} try {attempt + 1} {o.strip()[-100:]}")
        if rc == 0 and "successfully pushed" in o:
            break
        time.sleep(180)
    else:
        return False
    run_t = None
    while True:
        time.sleep(90)
        rc, o = kg("kernels", "status", kslug)
        m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", o)
        st = m.group(1) if m else None
        if st == "RUNNING" and run_t is None:
            run_t = time.time()
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
            break
        if run_t and time.time() - run_t > 90 * 60:
            klog(f"{iid}: kernel running > 90 min, left for the user to cancel")
            return False
        if (A.OUT / "STOP_K").exists():
            return False
    klog(f"{iid} {layers} kernel {st}")
    got = out / "dl"
    got.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        kg("kernels", "output", kslug, "-p", str(got), "-q")
        if list(got.rglob("*_cands.json")) or list(got.rglob("*_rf.json")):
            break
        time.sleep(120)
    n = 0
    for f in list(got.rglob("*_cands.json")) + list(got.rglob("*_rf.json")):
        shutil.copy(f, out / f.name)
        n += 1
    klog(f"{iid} {layers}: {n} candidate files downloaded")
    if st == "COMPLETE" and n:
        flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=A.ORDER)
    ap.add_argument("--layers", default="h2,sa")
    a = ap.parse_args(argv)
    SP.mkdir(parents=True, exist_ok=True)
    klog(f"lane K start ids {a.ids} layers {a.layers}")
    pending = list(a.ids)
    while pending and not (A.OUT / "STOP_K").exists():
        ready = [i for i in pending if "prep" in A.state(i)] if (A.OUT / "size_prior.json").exists() else []
        if not ready:
            time.sleep(60)
            continue
        iid = ready[0]
        stage_code()
        if not run_individual(iid, a.layers):
            klog(f"{iid}: not completed; left for a later pass")
        pending.remove(iid)
    klog("lane K end")


if __name__ == "__main__":
    main()
