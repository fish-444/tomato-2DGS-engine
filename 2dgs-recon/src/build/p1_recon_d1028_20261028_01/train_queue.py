"""Detached driver for CONTRACT p1_recon_d1028_20261028_01: waits for the points kernel, then trains the individuals on the Kaggle T4, ONE kernel at a time, and downloads each result
into build/p1_recon_all_20261025_01/<iid>/kernel_out (the nine's layout). Adapted from build/p1_recon_all_20261025_01/train_queue.py (slugs, queue, kernel sources only).
Stop by creating a file named STOP next to this script. Resumable: an individual whose <iid>/kernel_out/out/results.json exists is skipped.
Local work is only pushing, polling and downloading (Rule 0). A timeout never retries: it stops the queue and says why."""
import json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT / "build" / "p1_recon_all_20261025_01"
KDIR = Path(os.environ.get("TEMP", "C:/Temp")) / "d1028_train"
SLUG = "dolgfish/tomato-recon-d1028-train"
POINTS = "dolgfish/tomato-recon-d1028-points"
QUEUE = [q for q in sys.argv[1:]] or ["e_05", "e_01", "e_04", "e_07", "e_06", "e_03", "e_02"]
LOG = open(HERE / "train_queue.log", "a", encoding="utf-8")
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def kg(*args, timeout=1500):
    try:
        r = subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV, timeout=timeout); return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def status(slug):
    _, o = kg("kernels", "status", slug, timeout=200); m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", o); return m.group(1) if m else None


def wait_done(slug, limit_run_min=60, limit_queue_min=240):
    """Guard on RUNNING time only (a kernel waiting in Kaggle's queue is not running)."""
    t0 = time.time(); t_run = None
    while True:
        if (HERE / "STOP").exists():
            return "STOPPED"
        st = status(slug)
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED"):
            return st
        if st == "RUNNING" and t_run is None:
            t_run = time.time(); log(slug, "running since now")
        if t_run is not None and time.time() - t_run > limit_run_min * 60:
            return "TIMEOUT"
        if t_run is None and time.time() - t0 > limit_queue_min * 60:
            return "TIMEOUT"
        time.sleep(60)


def download(iid):
    """Download outside the OneDrive folder first (direct downloads into the synced folder stalled before), then copy."""
    import shutil
    tmp = Path(os.environ.get("TEMP", "C:/Temp")) / "d1028_dl" / iid; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    out = DEST / iid / "kernel_out"; out.mkdir(parents=True, exist_ok=True); o = ""
    for _ in range(8):
        rc, o = kg("kernels", "output", SLUG, "-p", str(tmp), "-q")
        if (tmp / "out" / "results.json").exists():
            shutil.copytree(tmp, out, dirs_exist_ok=True); return True
        time.sleep(120)
    log("download failed", iid, o[-300:]); return False


log("queue", QUEUE)
st = wait_done(POINTS, limit_run_min=90)
log("points kernel", st)
if st != "COMPLETE":
    log("the points kernel did not complete; queue not started (look at its log, no retry here)"); sys.exit(1)
KDIR.mkdir(parents=True, exist_ok=True)
(KDIR / "kernel-metadata.json").write_text(json.dumps(dict(id=SLUG, title="tomato-recon-d1028-train", code_file="run.py", language="python", kernel_type="script", is_private=True, enable_gpu=True,
    machine_shape="NvidiaTeslaT4", enable_internet=True, dataset_sources=["dolgfish/tomato-recon-d1028-input", "dolgfish/tomato-recon-code", "dolgfish/tomato-recon-d1028-job"],
    competition_sources=[], kernel_sources=["dolgfish/tomato-recon-d1028-masks", POINTS])))
for iid in QUEUE:
    if (HERE / "STOP").exists():
        log("STOP file found"); break
    if (DEST / iid / "kernel_out" / "out" / "results.json").exists():
        log("skip", iid); continue
    src = (ROOT / "tomato3d" / "recon" / "kernel_d1028_train.py").read_text(encoding="utf-8")
    src = re.sub(r'^IID = ".*"$', f'IID = "{iid}"', src, flags=re.M); assert f'IID = "{iid}"' in src
    (KDIR / "run.py").write_text(src, encoding="utf-8")
    for attempt in range(6):                                                                   # survive short DNS / network outages
        rc, o = kg("kernels", "push", "-p", str(KDIR)); log("push", iid, "try", attempt + 1, o.strip()[-120:])
        if rc == 0 and "successfully pushed" in o:
            break
        time.sleep(180)
    else:
        log("push failed 6 times, stopping"); break
    time.sleep(30)
    st = wait_done(SLUG); log("done", iid, st)
    if st == "COMPLETE":
        log("downloaded", iid, download(iid))
    elif st == "STOPPED":
        break
    elif st == "TIMEOUT":
        log("TIMEOUT: the kernel for", iid, "ran over 60 min or waited over 4 h; queue stopped, the Kaggle session must be cancelled by the user"); break
    else:
        log("kernel ended", st, "for", iid, "- not retried automatically; continuing with the next individual")
log("queue finished")
