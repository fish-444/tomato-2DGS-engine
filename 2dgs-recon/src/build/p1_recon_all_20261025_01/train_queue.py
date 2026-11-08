"""Detached driver for CONTRACT_ALL.md: trains the queued individuals on the Kaggle T4, ONE kernel at a time, and downloads each result.
Stop by creating a file named STOP next to this script. Resumable: an individual whose <iid>/kernel_out/out/results.json exists is skipped.
Local work is only pushing, polling and downloading (Rule 0)."""
import json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KDIR = Path("<SCRATCH_DIR>/scratchpad/all_job/train")
SLUG = "dolgfish/tomato-recon-all-train"
QUEUE = [q for q in sys.argv[1:]] or ["n_06", "n_05", "n_02", "t2_05", "n_03", "t2_04", "t2_02"]
LOG = open(HERE / "train_queue.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def kg(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"); return r.returncode, (r.stdout + r.stderr)


def status():
    _, o = kg("kernels", "status", SLUG); m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", o); return m.group(1) if m else None


def wait_done(limit_run_min=60, limit_queue_min=240):
    """Guard on RUNNING time only (a 19-minute training once showed 45+ minutes in total because it waited in Kaggle's queue)."""
    t0 = time.time(); t_run = None
    while True:
        if (HERE / "STOP").exists():
            return "STOPPED"
        st = status()
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
            return st
        if st == "RUNNING" and t_run is None:
            t_run = time.time(); log("running since now")
        if t_run is not None and time.time() - t_run > limit_run_min * 60:
            return "TIMEOUT"
        if t_run is None and time.time() - t0 > limit_queue_min * 60:
            return "TIMEOUT"
        time.sleep(60)


def download(iid):
    out = HERE / iid / "kernel_out"; out.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        rc, o = kg("kernels", "output", SLUG, "-p", str(out), "-q")
        if (out / "out" / "results.json").exists():
            return True
        time.sleep(120)
    log("download failed", iid, o[-300:]); return False


log("queue", QUEUE)
# first finish whatever is running (t2_02), and save it if its output is not yet local
st = wait_done()
log("current kernel", st)
if st == "COMPLETE" and not (HERE / "t2_02" / "kernel_out" / "out" / "results.json").exists():
    run_src = (KDIR / "run.py").read_text(encoding="utf-8")
    if 'IID = "t2_02"' in run_src:
        log("saving t2_02", download("t2_02"))
for iid in QUEUE:
    if (HERE / "STOP").exists():
        log("STOP file found"); break
    if (HERE / iid / "kernel_out" / "out" / "results.json").exists():
        log("skip", iid); continue
    src = (ROOT / "tomato3d" / "recon" / "kernel_all_train.py").read_text(encoding="utf-8")
    src = re.sub(r'^IID = ".*"$', f'IID = "{iid}"', src, flags=re.M); assert f'IID = "{iid}"' in src
    (KDIR / "run.py").write_text(src, encoding="utf-8")
    m = json.loads((KDIR / "kernel-metadata.json").read_text()); m["kernel_sources"] = ["dolgfish/tomato-recon-all-masks", "dolgfish/tomato-recon-all-points"]
    (KDIR / "kernel-metadata.json").write_text(json.dumps(m))
    for attempt in range(6):                                                                   # survive short DNS / network outages
        rc, o = kg("kernels", "push", "-p", str(KDIR)); log("push", iid, "try", attempt + 1, o.strip()[-120:])
        if rc == 0 and "successfully pushed" in o:
            break
        time.sleep(180)
    else:
        log("push failed 6 times, stopping"); break
    time.sleep(30)
    st = wait_done(); log("done", iid, st)
    if st == "COMPLETE":
        log("downloaded", iid, download(iid))
    elif st == "STOPPED":
        break
    elif st == "TIMEOUT":
        log("TIMEOUT: the kernel for", iid, "has been RUNNING for over 60 min (normal is ~20) or queued for over 4 h; queue stopped, the Kaggle session must be cancelled by the user"); break
log("queue finished")
