"""External-compute driver for the all-individuals 3D-first run (contract build/p1_all_3dfirst_20261027_01/CONTRACT.md, Amendment A3).

Local work = copy, hash, upload, download, state / status writes and the Roboflow API calls (key in memory, <= 3 streams). Every heavy step runs in a Kaggle kernel
(tomato3d/recon/kernel_3dfirst_ext.py): `size_prior` (CPU, once), `pre-<id>` (CPU: inputs, prep, geometry), `cands-<id>` (GPU: H2 + SAM-auto), `candsrf-<id>` (GPU: Roboflow drawing, pass 2).
At most 2 CPU and 1 GPU kernels at once; the first refusal / quota error stops all launching (no retry loop); a failed kernel is retried once, then the individual is `failed_external`.
GPU kernels start only after `gpu_quota.json` (remaining weekly hours, read from the account page) exists and the estimate fits. Never falls back to local compute.
State: <build>/ext_state.json (this process is its only writer), status: <build>/STATUS.md + status.log. Stop with a file STOP_EXT in the build folder.

usage: python -m tomato3d.organs.all3d_ext [--ids n_04 ...] [--unblock]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from tomato3d.organs import all3d_first as A
from tomato3d.organs import all_compound as AC

USER = "dolgfish"
SP = Path(os.environ.get("TEMP", "C:/Temp")) / "all3d_ext"
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
MAX_CPU, MAX_GPU, RF_STREAMS, POLL_S = 2, 1, 3, 90
REFUSAL = re.compile(r"quota|maximum .*session|limit|exceeded your|concurrent|too many|not allowed", re.I)
NETWORK = re.compile(r"NameResolution|getaddrinfo|ConnectionError|ConnectTimeout|ReadTimeout|timed out|Max retries exceeded|retries exceeded|Temporary failure|SSLError|RemoteDisconnected|Maximum batch CPU session count|Maximum .*concurrent", re.I)      # a full CPU-session slot (other sessions share the account) is transient: queue, retry next round, no block
EST_GPU_S_PER_VIEW = 25.0                      # first estimate only; replaced by the measured seconds of the first finished cands kernel
DL = dict(size_prior=r".*(size_prior\.json|run_info\.json|FAILED\.txt)$",
          pre=r".*(inputs\.json|state\.json|detect_input\.json|geometry\.json|geometry\.npz|worker\.log|run_info\.json|FAILED\.txt|\.rfin\.jpg)$",
          cands=r".*(_cands\.json|summary\.json|run_info\.json|FAILED\.txt)$", candsrf=r".*(_rf\.json|summary\.json|run_info\.json|FAILED\.txt)$",
          merge=r".*(groups\.npz|flags\.csv|summary\.json|merge_log\.json|run_info\.json|FAILED\.txt)$")
GPU_STAGES = {"cands", "candsrf"}
LOCK = threading.Lock()


# ---------------------------------------------------------------- small helpers
def kg(*a, timeout=1200):
    """kaggle CLI call; on timeout the whole process tree is killed (kaggle.exe is a launcher whose child keeps the pipes open) and the call reads as a network error."""
    pr = subprocess.Popen(["kaggle", *a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=ENV)
    try:
        out, _ = pr.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(pr.pid), "/T", "/F"], capture_output=True)
        try:
            out, _ = pr.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            out = ""
        return 124, f"kaggle {a[0]} {a[1]} timed out after {timeout} s (process tree killed) {(out or '')[-200:]}"
    return pr.returncode, (out or "").strip()


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def free_gb():
    return psutil.virtual_memory().available / 2 ** 30


def sid(iid):
    return iid.replace("_", "")


def sha_file(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def tree_hash(paths):
    h = hashlib.sha256()
    for root in paths:
        for f in sorted(Path(root).rglob("*")) if Path(root).is_dir() else [Path(root)]:
            if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc" and f.name != "all3d_ext.py":     # the local driver itself is not kernel code
                h.update(f.relative_to(A.ROOT).as_posix().encode()); h.update(sha_file(f).encode())
    return h.hexdigest()[:16]


def rel(p):
    return Path(p).resolve().relative_to(A.ROOT.resolve()).as_posix()


class State:
    def __init__(self):
        self.f = A.OUT / "ext_state.json"; self.d = json.load(open(self.f, encoding="utf-8")) if self.f.exists() else dict(jobs={}, datasets={}, blocked=None, rf={})

    def save(self):
        with LOCK:
            tmp = self.f.with_suffix(".tmp"); json.dump(self.d, open(tmp, "w", encoding="utf-8"), indent=1); tmp.replace(self.f)


# ---------------------------------------------------------------- datasets
def push_dataset(st, slug, d, content_hash):
    """Create the private dataset, or add a version when the staged content changed; wait until Kaggle reports it ready."""
    rec = st.d["datasets"].get(slug)
    if rec and rec.get("hash") == content_hash and rec.get("ready"):
        return True
    (d / "dataset-metadata.json").write_text(json.dumps(dict(title=slug, id=f"{USER}/{slug}", licenses=[dict(name="CC0-1.0")])))
    rc, o = kg("datasets", "status", f"{USER}/{slug}")
    exists = rc == 0 and "ready" in o.lower()
    rc, o = kg("datasets", "version", "-p", str(d), "-m", f"content {content_hash}", "--dir-mode", "zip") if exists else kg("datasets", "create", "-p", str(d), "--dir-mode", "zip")
    slog(f"dataset {slug} {'version' if exists else 'create'} rc {rc}: {o[-160:]}")
    if rc != 0:
        if REFUSAL.search(o) and not NETWORK.search(o):
            block(st, f"dataset upload refused ({slug}): {o[-200:]}")
        return False
    for _ in range(40):
        time.sleep(30); rc, o = kg("datasets", "status", f"{USER}/{slug}")
        if "ready" in o.lower():
            st.d["datasets"][slug] = dict(hash=content_hash, ready=True, at=now()); st.save(); return True
    slog(f"dataset {slug}: not ready after 20 min"); return False


def stage_mirror(slug, files):
    """files: {relative path under the project root: source path}. Staged copy (streamed) under SP/slug/root/<rel> + MIRROR.json."""
    d = SP / slug; (d / "root").mkdir(parents=True, exist_ok=True); h = hashlib.sha256()
    for r_, src in sorted(files.items()):
        dst = d / "root" / r_; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
            shutil.copyfile(src, dst)
        h.update(r_.encode()); h.update(str(Path(src).stat().st_size).encode()); h.update(sha_file(src).encode())
    json.dump(dict(files=sorted(files)), open(d / "MIRROR.json", "w", encoding="utf-8"), indent=0); return d, h.hexdigest()[:16]


def code_dataset(st):
    chash = sha_file(A.OUT / "CONTRACT.md"); slug = f"p1-all-code-{chash}"; d = SP / slug
    srcs = [A.ROOT / x for x in ("tomato3d", "tools", "tests", "config")]; th = tree_hash(srcs)
    if st.d["datasets"].get(slug, {}).get("hash") != th:
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
        for s in srcs:
            shutil.copytree(s, d / s.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
        shutil.copy(A.ROOT / "build/p1_t1_occlusion_20261017_01/parameters.json", d / "t1_parameters.json"); shutil.copy(A.OUT / "CONTRACT.md", d / "CONTRACT.md")
        json.dump(dict(contract_sha256_16=chash, code_tree=th, at=now()), open(d / "CODE_MARK.json", "w"))
    return slug if push_dataset(st, slug, d, th) else None


def r31b_files():
    from tomato3d.organs import compound_cluster as CC
    from tomato3d.organs import light_organs as LO
    from tomato3d.organs import userlabel as UL
    from tomato3d.organs import vote2d as V2
    fs = [LO.DEFAULTS["surfels"], CC.TRACK_A / "surfel_labels.npz", LO.DEFAULTS["poses"], LO.DEFAULTS["views"], V2.OUT / "stage_a_h2id.npz", UL.OUT / "views_chosen.json", UL.OUT / "labels" / "leaf_labels.json"]
    fs += [UL.OUT / "detect" / f"{v}.json" for v in json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8"))["chosen"]]
    return {rel(f): f for f in fs}


def individual_files(iid):
    pz = json.load(open(AC.POSES[iid], encoding="utf-8")); acc = pz["accepted"]; fs = {}; miss = []
    if iid in AC.E_IDS:                                                       # the 2026-10-28 batch lives in the Downloads folders; index only those
        idx = {}
        for r in sorted(AC.E_PHOTO_ROOT.glob("Photos-1-001*")):
            for f in r.glob("*.HEIC"):
                idx.setdefault(f.name, []).append(f)
    else:
        idx = AC.heic_index()
    for f in (AC.ALL / "outputs" / iid / f"R3-1b-style_{iid}_surfels_full.ply", AC.POSES[iid], AC.ALL / "k3a" / iid / "split.json", AC.OUT / iid / "views_chosen.json",
              A.ROOT / "build/p1_plant_model_20261026_01/rim" / f"{iid}.json"):
        if Path(f).exists():
            fs[rel(f)] = f
    for v in acc:
        m = AC.ALL / "k3a" / iid / "masks" / (Path(v).stem + ".png"); h = idx.get(v[:-4] + ".HEIC", [None])[0]
        if m.exists():
            fs[rel(m)] = m
        else:
            miss.append(f"mask {m.name}")
        if h is not None:
            fs[f"heic/{h.name}"] = h
        else:
            miss.append(f"heic {v[:-4]}.HEIC")
    return fs, miss, len(acc)


def rf_dataset(st, iid):
    c = A.cell(iid); files = {rel(f): f for sub in ("rf_detections", "rf_detections_tiled") if (c / sub).exists() for f in (c / sub).glob("*.json")}
    if not files:
        return None
    slug = f"tomato-3dfirst-rf-{sid(iid)}"; d, h = stage_mirror(slug, files)
    key = __import__("tomato3d.organs.auto_proposers", fromlist=["x"])._rf_key()
    if key and any(key in (d / "root" / r_).read_text(encoding="utf-8", errors="ignore") for r_ in files):
        slog(f"{iid}: key-like string in the Roboflow cache, not uploaded"); return None
    return slug if push_dataset(st, slug, d, h) else None


# ---------------------------------------------------------------- kernels
def kernel_slug(stage, iid):
    return f"tomato-3dfirst-{stage.replace('_', '')}" + (f"-{sid(iid)}" if iid else "")


def push_kernel(st, key, stage, iid, datasets, sources):
    slug = kernel_slug(stage, iid); d = SP / "kernels" / slug; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    src = (A.ROOT / "tomato3d/recon/kernel_3dfirst_ext.py").read_text(encoding="utf-8"); layers = "rf" if stage == "candsrf" else "h2,sa"
    stage_name = "cands" if stage in GPU_STAGES else stage
    for k_, v_ in (("STAGE", stage_name), ("IID", iid or ""), ("LAYERS", layers)):
        src = re.sub(rf'^{k_} = ".*"$', f'{k_} = "{v_}"', src, flags=re.M)
    (d / "run.py").write_text(src, encoding="utf-8"); gpu = stage in GPU_STAGES
    meta = dict(id=f"{USER}/{slug}", title=slug, code_file="run.py", language="python", kernel_type="script", is_private=True, enable_gpu=gpu, enable_internet=True,
                dataset_sources=[f"{USER}/{x}" for x in datasets], kernel_sources=[f"{USER}/{x}" for x in sources], competition_sources=[])
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    rc, o = kg("kernels", "push", "-p", str(d)); m = re.search(r"version (\d+)", o)
    j = st.d["jobs"].setdefault(key, dict(stage=stage, iid=iid, slug=slug, tries=0)); j["tries"] += 1
    if rc == 0 and "successfully pushed" in o.lower():
        j.update(status="running", version=(int(m.group(1)) if m else None), started=now(), t0=time.time(), gpu=gpu); st.save()
        status_line(st, iid, stage, j, "started"); return True
    slog(f"push {slug} refused or failed: {o[-240:]}")
    if NETWORK.search(o):
        j["tries"] -= 1; j["status"] = j.get("status") if j.get("status") in ("retry", "push_failed") else "queued"; slog(f"{slug}: network error, will try again next round (not counted)"); st.save(); return False
    if REFUSAL.search(o):
        j["tries"] -= 1; j["status"] = "queued"; block(st, f"kernel push refused ({slug}): {o[-200:]}")
    else:
        j["status"] = "push_failed"
    st.save(); return False


def poll(st, key):
    j = st.d["jobs"][key]; rc, o = kg("kernels", "status", f"{USER}/{j['slug']}", timeout=300)
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", o) or re.search(r'status "?([a-zA-Z_]+)"?', o); s = (m.group(1) if m else "").upper()
    if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED", "CANCELLED"):
        return s
    if j.get("gpu") and time.time() - j["t0"] > 3 * 3600:
        return "TIMEOUT_LOCAL"
    return None


def collect(st, key, final):
    j = st.d["jobs"][key]; stage, iid = j["stage"], j["iid"]; dest = (A.cell(iid) if iid else A.OUT) / "ext" / f"{stage}_v{j.get('version')}"; dest.mkdir(parents=True, exist_ok=True)
    tmp = SP / "dl" / f"{j['slug']}_v{j.get('version')}"; tmp.mkdir(parents=True, exist_ok=True)   # download outside OneDrive (direct downloads into the synced folder stalled), then copy
    for _ in range(4):
        kg("kernels", "output", f"{USER}/{j['slug']}", "-p", str(tmp), "-q", "-o", "--file-pattern", DL[stage], timeout=900)
        if list(tmp.rglob("run_info.json")) or list(tmp.rglob("FAILED.txt")):
            break
        time.sleep(60)
    for f in tmp.rglob("*"):
        if f.is_file():
            t = dest / f.relative_to(tmp); t.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, t)
    ri = next(iter(dest.rglob("run_info.json")), None); info = json.load(open(ri, encoding="utf-8")) if ri else {}
    if final == "COMPLETE" and not ri and not list(dest.rglob("FAILED.txt")):
        j["status"] = "download_pending"; j["final"] = final; st.save(); slog(f"{j['slug']}: kernel complete, download incomplete (network?) - will download again"); return
    ok = final == "COMPLETE" and info.get("status") == "complete"
    j["seconds"] = info.get("seconds"); j["finished"] = now()
    if ok:
        n = 0
        for f in dest.rglob("*"):                                                    # results into the project tree; never overwrite an existing file
            if f.is_file() and "mirror" in f.parts and "root" in f.parts:
                r_ = Path(*f.parts[f.parts.index("root", f.parts.index("mirror")) + 1:]); t = A.ROOT / r_
                if not t.exists():
                    t.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, t); n += 1
        j["status"] = "done"; j["files_in"] = n
        (A.cell(iid) / "ext" if iid else A.OUT / "ext").mkdir(parents=True, exist_ok=True)
        ((A.cell(iid) if iid else A.OUT) / "ext" / f"{stage}.done").write_text(json.dumps(dict(kernel=j["slug"], version=j.get("version"), at=now(), seconds=j.get("seconds"))))
    else:
        why = (next(iter(dest.rglob("FAILED.txt")), None)); j["error"] = (why.read_text(encoding="utf-8")[:300] if why else f"kernel status {final}")
        j["status"] = "retry" if j["tries"] < 2 else "failed_external"
    st.save(); status_line(st, iid, stage, j, "finished")


# ---------------------------------------------------------------- Roboflow lane (local API calls, key in memory only)
def rf_lane(st, iid):
    from PIL import Image
    import numpy as np
    from tomato3d.leafseg import rfdet
    from tomato3d.organs import auto_proposers as AP
    key = AP._rf_key(); c = A.cell(iid); ims = sorted((c / "rf_inputs").glob("*.rfin.jpg")); miss = []
    if not key:
        st.d["rf"][iid] = dict(status="no_key", at=now()); st.save(); return

    def one(f):
        stem = f.name[:-len(".rfin.jpg")]
        for attempt in range(3):
            try:
                img = np.asarray(Image.open(f).convert("RGB")); pred = rfdet.fetch(f, img, c / "rf_detections" / f"{stem}.json", key)
                if len(pred.get("predictions", [])) >= AP.RF_CAP:
                    rfdet.tiled_dets(f, img, c / "rf_detections_tiled", stem, key)
                return None
            except Exception as e:
                if attempt == 2:
                    return dict(view=stem, error=AP._scrub(f"{type(e).__name__}: {e}", key))
                time.sleep(20)
    (c / "rf_detections").mkdir(exist_ok=True); (c / "rf_detections_tiled").mkdir(exist_ok=True)
    with ThreadPoolExecutor(RF_STREAMS) as ex:
        miss = [r for r in ex.map(one, ims) if r]
    st.d["rf"][iid] = dict(status="done", photos=len(ims) - len(miss), missing=miss, at=now()); st.save(); slog(f"{iid} rf: {len(ims) - len(miss)} photos, {len(miss)} missing")


# ---------------------------------------------------------------- status
def slog(msg):
    line = f"{now()} {msg}"
    with LOCK, open(A.OUT / "ext.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def gpu_quota():
    f = A.OUT / "gpu_quota.json"; return json.load(open(f, encoding="utf-8")) if f.exists() else None


def gpu_used_h(st, since):
    return sum((j.get("seconds") or (time.time() - j.get("t0", time.time()))) for j in st.d["jobs"].values() if j.get("gpu") and j.get("started", "") >= since and j.get("status") != "queued") / 3600


def quota_left(st):
    q = gpu_quota()
    return None if not q else q["remaining_hours"] - gpu_used_h(st, q["read_at"])


def status_line(st, iid, stage, j, event):
    ql = quota_left(st); line = f"{now()} {iid or '_global'} {stage} v{j.get('version')} {j.get('status')} ({event}) gpu_quota_left={'%.1f h' % ql if ql is not None else 'not read'} free_ram={free_gb():.2f} GB"
    with LOCK, open(A.OUT / "status.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True); write_status(st)


def block(st, why):
    if not st.d.get("blocked"):
        st.d["blocked"] = dict(at=now(), why=why); slog(f"BLOCKED: {why}"); st.save()


def write_status(st, top=None):
    q = gpu_quota(); ql = quota_left(st); L = ["# P1 all-individuals 3D-first - external run status (Amendment A3)", "", f"updated {now()}; local free RAM {free_gb():.2f} GB; local workers 0", ""]
    if st.d.get("blocked"):
        L += [f"**Launching stopped: {st.d['blocked']['why']}** ({st.d['blocked']['at']}). Queued jobs wait; no retry loop. "
              "If the external route stays blocked: needs user approval for local fallback (the driver does not fall back).", ""]
    if top:
        L += [f"**{top}**", ""]
    L += [f"- GPU weekly quota: {'%.1f h remaining at %s (%s), %.1f h used since, ~%.1f h left' % (q['remaining_hours'], q['read_at'], q.get('source', ''), gpu_used_h(st, q['read_at']), ql) if q else 'not read yet - GPU kernels wait (gpu_quota.json)'}",
          f"- concurrent sessions: {q.get('concurrent', 'unknown') if q else 'unknown'}; this driver uses <= {MAX_CPU} CPU + {MAX_GPU} GPU", "",
          "| individual | stage | kernel | version | status | tries | seconds | note |", "|---|---|---|---|---|---|---|---|"]
    for k, j in st.d["jobs"].items():
        L.append(f"| {j['iid'] or '_global'} | {j['stage']} | {j['slug']} | {j.get('version')} | {j.get('status')} | {j.get('tries')} | {j.get('seconds') or ''} | {(j.get('error') or '')[:120]} |")
    L += ["", "Roboflow lane (local API, key in memory): " + ", ".join(f"{i} {r['status']} {r.get('photos', '')}" for i, r in st.d["rf"].items())]
    (A.OUT / "STATUS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main loop
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="*", default=A.ORDER); ap.add_argument("--unblock", action="store_true"); ap.add_argument("--no-gpu", action="store_true"); ap.add_argument("--merge-ids", nargs="*", default=None); a = ap.parse_args(argv); a.merge_ids = a.merge_ids or a.ids
    SP.mkdir(parents=True, exist_ok=True); st = State(); lock = A.OUT / "RUNNING_ext.lock"
    if lock.exists():
        pid = lock.read_text().split()[1]
        if pid.isdigit() and psutil.pid_exists(int(pid)):
            print("another driver is running", pid); return
    lock.write_text(f"pid {os.getpid()} {now()}\n")
    if a.unblock:
        st.d["blocked"] = None; st.save()
    slog(f"driver start pid {os.getpid()} ids {a.ids}"); rf_threads = {}
    try:
        code = code_dataset(st)
        if not code:
            write_status(st, "code dataset upload failed - needs user approval for local fallback or a retry"); return
        while not (A.OUT / "STOP_EXT").exists():
            J = st.d["jobs"]; running = [k for k, j in J.items() if j.get("status") == "running"]
            for k in running:                                                           # poll
                try:
                    fin = poll(st, k)
                except Exception as e:
                    slog(f"poll {k}: {type(e).__name__} (network?)"); fin = None
                if fin:
                    collect(st, k, fin)
            for k in [k for k, j in J.items() if j.get("status") == "download_pending"]:
                collect(st, k, J[k].get("final", "COMPLETE"))
            running = [k for k, j in J.items() if j.get("status") == "running"]; n_cpu = sum(1 for k in running if not J[k].get("gpu")); n_gpu = len(running) - n_cpu
            done = lambda k: J.get(k, {}).get("status") == "done"
            want = []                                                                  # (key, stage, iid)
            if not done("size_prior"):
                want.append(("size_prior", "size_prior", None))
            for iid in a.ids:
                c = A.cell(iid)
                if done("size_prior") and not done(f"pre:{iid}"):
                    want.append((f"pre:{iid}", "pre", iid))
                if done(f"pre:{iid}") and iid not in st.d["rf"] and iid not in rf_threads:
                    rf_threads[iid] = threading.Thread(target=rf_lane, args=(st, iid), daemon=True); rf_threads[iid].start(); slog(f"{iid} rf lane started")
                if done(f"pre:{iid}") and not done(f"cands:{iid}") and not a.no_gpu:
                    want.append((f"cands:{iid}", "cands", iid))
                if done(f"cands:{iid}") and st.d["rf"].get(iid, {}).get("status") == "done" and not done(f"candsrf:{iid}") and not a.no_gpu:
                    want.append((f"candsrf:{iid}", "candsrf", iid))
                if done(f"cands:{iid}") and (done(f"candsrf:{iid}") or J.get(f"candsrf:{iid}", {}).get("status") == "failed_external") and not done(f"merge:{iid}") and iid in a.merge_ids:
                    want.append((f"merge:{iid}", "merge", iid))
            for key, stage, iid in want:
                j = J.get(key, {})
                if j.get("status") in ("running", "failed_external") or st.d.get("blocked"):
                    continue
                if j.get("status") in ("retry", "push_failed") and j.get("tries", 0) >= 2:
                    j["status"] = "failed_external"; st.save(); continue
                gpu = stage in GPU_STAGES
                if (gpu and n_gpu >= MAX_GPU) or (not gpu and n_cpu >= MAX_CPU):
                    continue
                if gpu:
                    ql = quota_left(st)
                    if ql is None:
                        continue
                    nv = len(json.load(open(A.cell(iid) / "detect_input.json", encoding="utf-8"))["views"]); meas = [J[x]["seconds"] / max(J[x].get("views", 1), 1) for x in J if J[x].get("gpu") and J[x].get("status") == "done" and J[x].get("seconds")]
                    est_h = nv * (max(meas) if meas else EST_GPU_S_PER_VIEW) / 3600 + 0.15
                    if est_h > ql:
                        write_status(st, f"GPU quota: {stage} {iid} needs ~{est_h:.1f} h, ~{ql:.1f} h left. Options for the user: fewer views per individual, lower SAM-auto sampling (grid 32 -> 16), defer later individuals"); continue
                if stage == "size_prior":
                    d, h = stage_mirror("tomato-3dfirst-r31b", r31b_files()); ok = push_dataset(st, "tomato-3dfirst-r31b", d, h); ds, src = [code, "tomato-3dfirst-r31b"], []
                elif stage == "pre":
                    fs, miss, nacc = individual_files(iid)
                    if miss:
                        slog(f"{iid}: {len(miss)} inputs missing on this PC ({miss[:4]}); the kernel records them as photo errors")
                    slug = f"tomato-3dfirst-in-{sid(iid)}"; d, h = stage_mirror(slug, fs); ok = push_dataset(st, slug, d, h); ds, src = [code, slug], [kernel_slug("size_prior", None)]
                elif stage == "cands":
                    ok = True; ds, src = [code, "tomato-recon-code", "tomato-mask-input"], [kernel_slug("pre", iid), kernel_slug("size_prior", None)]
                elif stage == "merge":
                    ok = True; ds = [code, f"tomato-3dfirst-in-{sid(iid)}"]; src = [kernel_slug("pre", iid), kernel_slug("cands", iid), kernel_slug("size_prior", None)] + ([kernel_slug("candsrf", iid)] if done(f"candsrf:{iid}") else [])
                else:
                    rs = rf_dataset(st, iid); ok = bool(rs); ds, src = [code, "tomato-recon-code", "tomato-mask-input", rs or ""], [kernel_slug("pre", iid), kernel_slug("size_prior", None)]
                if not ok:
                    continue
                if push_kernel(st, key, stage, iid, ds, src):
                    if gpu:
                        J[key]["views"] = len(json.load(open(A.cell(iid) / "detect_input.json", encoding="utf-8"))["views"]); n_gpu += 1
                    else:
                        n_cpu += 1
            st.save(); write_status(st)
            alive = [k for k, j in J.items() if j.get("status") == "running"] or [t for t in rf_threads.values() if t.is_alive()]
            if not alive and (st.d.get("blocked") or not want):
                slog("nothing running and nothing to launch" + (" (blocked)" if st.d.get("blocked") else "") + "; driver ends"); break
            if not alive and want and all(J.get(k, {}).get("status") == "failed_external" or (s in GPU_STAGES and quota_left(st) is None) for k, s, _ in want):
                slog("only GPU stages left and the GPU quota is not read (or all failed); driver ends - write gpu_quota.json and restart"); break
            time.sleep(POLL_S)
    finally:
        lock.unlink(missing_ok=True); write_status(st); slog("driver end")


if __name__ == "__main__":
    main()
