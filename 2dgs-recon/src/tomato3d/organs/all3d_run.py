"""Scheduler for the all-individuals 3D-first run (contract build/p1_all_3dfirst_20261027_01/CONTRACT.md, section 1).

One detached OS process per (individual, step). Cells never share files. Before every local start the free RAM is read:
below 7 GB nothing is started and every wait is logged (reason, free RAM, engine JVM present or not), checked again every 60 s. Concurrency = 1 local worker.
It never touches another process. Resumable: a step with a `state.json` marker is skipped. Stop by creating a file named STOP in the build folder.

usage: python -m tomato3d.organs.all3d_run [--ids n_04 t2_02 ...] [--max-workers 4]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import psutil

from tomato3d.organs import all3d_first as A

FREE_START_GB = 7.0          # user rule 2026-10-27: 2.5 GB worker + 4 GB reserved for engine JVM (GroIMP) runs + margin; checked by the scheduler before EVERY local start
MAX_LOCAL = 0                # Amendment A3 (user Addendum 2, 2026-10-27 16:15): no local worker for this task; the external driver is all3d_ext.py
BUDGET_GB = dict(inputs=1.0, prep=1.0, geometry=2.5, size_prior=1.5, rf=0.5)
DEPS = dict(inputs=[], prep=["inputs"], geometry=["inputs"], rf=["prep"])
SEQ = ["inputs", "prep", "geometry"]
MAX_RF = 3                                     # Roboflow lane: at most 3 API streams at once
CREATE = 0x00000008 | 0x00000200              # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def slog(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with open(A.OUT / "scheduler.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def engine_jvm():
    """True if a GroIMP / engine JVM process is running (information for the wait log; the 7 GB rule applies either way)."""
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower().startswith("java") and "groimp" in " ".join(p.info["cmdline"] or []).lower():
                return True
        except Exception:
            continue
    return False


def free_gb():
    return psutil.virtual_memory().available / 2 ** 30


def running_workers(procs):
    return {k: p for k, p in procs.items() if p.poll() is None}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="*", default=A.ORDER); ap.add_argument("--steps", nargs="*", default=SEQ + ["rf"]); a = ap.parse_args(argv)
    A.OUT.mkdir(parents=True, exist_ok=True)
    if MAX_LOCAL == 0:
        slog(f"REFUSED: Amendment A3 sets the local worker count to 0 (steps {a.steps}); use python -m tomato3d.organs.all3d_ext"); return
    lock = A.OUT / ("RUNNING_rf.lock" if a.steps == ["rf"] else "RUNNING.lock")
    if lock.exists():
        old = lock.read_text().split()
        pid = int(old[1]) if len(old) > 1 and old[1].isdigit() else None
        if pid and psutil.pid_exists(pid):
            slog(f"another scheduler (pid {pid}) is running; stop here"); return
    lock.write_text(f"pid {os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); slog(f"scheduler start pid {os.getpid()} ids {a.ids} steps {a.steps}")
    procs, failed = {}, set()
    lane = lambda step: "rf" if step == "rf" else "main"
    is_done = lambda iid, step: (A.OUT / "size_prior.json").exists() if iid == "_global" else (step in A.state(iid, step))
    try:
        while True:
            if (A.OUT / "STOP").exists():
                slog("STOP file found"); break
            live = running_workers(procs)
            for k, p in procs.items():                                           # a finished worker without a done marker = failed (never retried automatically)
                if p.poll() is not None and k not in failed and not is_done(*k):
                    failed.add(k); slog(f"{k[0]} {k[1]}: worker ended (code {p.returncode}) without a done marker; see {k[0]}/logs/")
            busy = {(k[0], lane(k[1])) for k in live}; todo = []
            sp_done = (A.OUT / "size_prior.json").exists()
            if not sp_done and ("_global", "size_prior") not in live and ("_global", "size_prior") not in failed:
                todo.append(("_global", "size_prior"))
            n_rf = sum(1 for k in live if k[1] == "rf")
            for iid in a.ids:
                st = A.state(iid)
                for step in a.steps:                                             # first unfinished step of the main lane
                    if lane(step) != "main":
                        continue
                    if step in st or (iid, step) in failed:
                        continue
                    if (iid, "main") not in busy and all(d in st for d in DEPS[step]) and not (step == "geometry" and not sp_done):
                        todo.append((iid, step))
                    break
                if "rf" in a.steps and (iid, "rf") not in busy and (iid, "rf") not in failed and "rf" not in A.state(iid, "rf") and "prep" in st and n_rf < MAX_RF:
                    todo.append((iid, "rf")); n_rf += 1
            if not todo and not live:
                slog("all requested steps done or failed"); break
            f = free_gb(); jvm = engine_jvm(); waits = []
            for iid, step in todo:
                live = running_workers(procs); budget = BUDGET_GB[step]
                if step == "rf":                                                 # Roboflow lane: network only, not a local worker, no RAM gate beyond its own 0.5 GB
                    if sum(1 for k in live if k[1] == "rf") >= MAX_RF:
                        continue
                else:
                    n_local = sum(1 for k in live if k[1] != "rf")
                    if n_local >= MAX_LOCAL:
                        waits.append(("WAIT_LOCAL_BUSY", iid, step)); continue
                    if f < FREE_START_GB:
                        waits.append(("WAIT_LOW_RAM", iid, step)); continue
                cmd = [sys.executable, "-X", "utf8", "-m", "tomato3d.organs.all3d_first", "--step", step] + (["--id", iid] if iid != "_global" else [])
                (A.OUT / "logs").mkdir(exist_ok=True); out = open((A.OUT / "logs" / f"{step}.out.log") if iid == "_global" else (A.cell(iid) / "logs" / f"{step}.out.log"), "a", encoding="utf-8")
                procs[(iid, step)] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, creationflags=CREATE, cwd=str(A.ROOT)); slog(f"start {iid} {step} pid {procs[(iid, step)].pid} (free {f:.2f} GB, engine_jvm={jvm}, local workers {sum(1 for k in running_workers(procs) if k[1] != 'rf')})"); f -= budget
            if waits:                                                            # one line per check: every wait is logged
                reason = waits[0][0]; nxt = ", ".join(f"{i} {st}" for _, i, st in waits[:3])
                slog(f"{reason}: free {f:.2f} GB (start needs >= {FREE_START_GB} GB and no other local worker), engine_jvm={jvm}, local workers {sum(1 for k in running_workers(procs) if k[1] != 'rf')}, waiting {len(waits)}: next {nxt}")
            time.sleep(60 if (todo or live) else 5)
    finally:
        lock.unlink(missing_ok=True); slog("scheduler end")


if __name__ == "__main__":
    main()
