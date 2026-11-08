"""P1 Part A of the dz10 prompt (contract build/p1_dz10_labels_20261101_01/CONTRACT.md): "완료" on 8835 -> distance + T-prox labels for that skeleton version -> grade after 10 held-out picks.

usage: python -m tomato3d.organs.skeleton_dz10_labels --id <plant>            labels for the current skeleton version (then grade if picks exist)
       python -m tomato3d.organs.skeleton_dz10_labels --id <plant> --grade    grade only
       python -m tomato3d.organs.skeleton_dz10_labels --summary               per-plant review table (summary.md / summary.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs import skeleton_propagate as SP
from tomato3d.organs import skeleton_store as SS

ROOT = SP.ROOT
OUT = ROOT / "build/p1_dz10_labels_20261101_01"
STORE_BASE = SP.ALL3 / "user_base_click.jsonl"
TRIGGER_FROM = "2026-11-01T19:40:00"
LABELS = ["skeleton_dz10_user_confirmed", "conditional_on_unvalidated_scale", "after_checkpoint_v1"]
PLANTS = ["n_04", "t2_02", "n_05", "n_01", "n_02", "n_03", "n_06", "t2_05", "e_01", "e_04", "e_05"]


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def log(pid, *a):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"job_{pid}.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


def live(pid):
    """Re-derive the skeleton exactly as the page does (saved version), return (sk, skelver)."""
    z = np.load(SP.ALL3 / pid / "merge" / "groups.npz")
    sk, _, _ = SS.rebuild(ROOT, STORE_BASE, pid, SP.capture_date(pid), SP.reconstruction_id(pid), z["xyz"].astype(float), save=False)
    return sk, SIO.skeleton_version(sk)


def paths(pid, v):
    return OUT / f"labels_tprox_{pid}_{v}.npz", OUT / f"labels_distance_{pid}_{v}.npz", OUT / f"{pid}_{v}"


def run_labels(pid):
    lock = OUT / f"{pid}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, time.strftime("%F %T").encode()); os.close(fd)
    except FileExistsError:
        log(pid, "another label job holds", lock.name, "- nothing started"); return None
    try:
        sk, v = live(pid); ft, fd_, d = paths(pid, v)
        if ft.exists() and fd_.exists():
            log(pid, "skelver", v, "already labelled"); return grade(pid)
        if not [l for l in sk["leaves"] if l["status"] in ("active", "new") and len(l["polyline_xyz"]) >= 2]:
            log(pid, "no live line with >= 2 points: not_run"); return None
        d.mkdir(parents=True, exist_ok=True); snap = d / "skeleton_v1.json"; snap.write_text(json.dumps(sk, ensure_ascii=False, indent=1), encoding="utf-8")
        log(pid, "start skelver", v, "leaves", sum(l["status"] in ("active", "new") for l in sk["leaves"]))
        live_file = SIO.skeleton_dir(ROOT, pid, SP.capture_date(pid)) / "skeleton_v1.json"
        if SIO.skeleton_version(SIO.load(live_file)) != v:                       # the page saves after every edit; the saved file must be this version
            SIO.save(sk, ROOT)
        SP.step_distance(pid, str(snap)); src = SIO.skeleton_dir(ROOT, pid, SP.capture_date(pid)) / "labels_skeleton_distance.npz"
        shutil.copy2(src, d / "labels_skeleton_distance.npz"); shutil.copy2(src, fd_); log(pid, "distance done ->", fd_.name)
        from tomato3d.organs import skeleton_r8pc as R8
        from tomato3d.organs import skeleton_tprox as ST
        ST.OUT = d; R8.RUNS = d / "runs"; ST.skel_dir = lambda _pid, _d=d: _d                 # read the snapshot, write into the version folder (code otherwise unchanged)
        bd = ST.step_prep(pid); meta = json.load(open(bd / "bundle_meta.json", encoding="utf-8"))
        if meta["skeleton_sha256"] != SIO.sha_file(live_file) and SIO.skeleton_version(SIO.load(live_file)) != v:
            log(pid, "skeleton changed between snapshot and bundle: stop"); return None
        ST.step_push(pid); log(pid, "T-prox pushed", json.load(open(bd / "kaggle.json"))["kernel"])
        ST.step_finish(pid); shutil.copy2(d / f"labels_tprox_{pid}.npz", ft); log(pid, "T-prox done ->", ft.name)
        if live(pid)[1] != v:
            json.dump(dict(flag="skeleton_changed_during_run", skelver=v), open(d / "FLAG_skeleton_changed_during_run.json", "w")); log(pid, "flag skeleton_changed_during_run")
        return grade(pid)
    except Exception:
        log(pid, "FAILED", traceback.format_exc()); raise
    finally:
        os.replace(lock, OUT / f"{pid}.lock.done_{time.strftime('%Y%m%dT%H%M%S')}")


def retry_tprox(pid, v):
    """Amendment A1 infra_retry: re-push the bundle already built from the snapshot of skelver v (new stamp folder, same bundle bytes), then finish and grade."""
    from tomato3d.organs import skeleton_r8pc as R8
    from tomato3d.organs import skeleton_tprox as ST
    ft, fd_, d = paths(pid, v)
    if ft.exists():
        log(pid, "infra_retry: T-prox for", v, "already exists"); return grade(pid)
    old = sorted(x for x in (d / "runs" / pid).iterdir() if x.is_dir() and x.name[:2] == "20")[-1]; stamp = time.strftime("%Y%m%dT%H%M%S"); new = old.parent / stamp
    (new / "kaggle_data").mkdir(parents=True); shutil.copy2(old / "kaggle_data" / "r8pc_bundle.npz", new / "kaggle_data" / "r8pc_bundle.npz")
    meta = json.load(open(old / "bundle_meta.json", encoding="utf-8")); meta.update(stamp=stamp, infra_retry_of=old.name, labels=meta["labels"] + ["infra_retry"])
    json.dump(meta, open(new / "bundle_meta.json", "w", encoding="utf-8"), indent=1); old.rename(d / f"_infra_failed_{pid}_{old.name}")          # outside runs/<pid>: run_dir sorts '_' after digits
    ST.OUT = d; R8.RUNS = d / "runs"; ST.skel_dir = lambda _pid, _d=d: _d
    log(pid, "infra_retry start", v, "bundle from", old.name); ST.step_push(pid); log(pid, "T-prox pushed", json.load(open(new / "kaggle.json"))["kernel"])
    ST.step_finish(pid); shutil.copy2(d / f"labels_tprox_{pid}.npz", ft); log(pid, "T-prox done ->", ft.name, "(infra_retry)")
    if live(pid)[1] != v:
        log(pid, "note: live skeleton is now", live(pid)[1], "- labels stay on", v)
    return grade(pid)


def finish_retry(pid, v):
    """A1 recovery: the 11:17 retry re-pushed the original bundle from the renamed folder (run_dir sorts '_' after digits), so the kernel to poll is the original one.
    Restore its folder name, park the unused new stamp folder outside runs/, then finish (poll / download / set A / rachis / post) and grade."""
    from tomato3d.organs import skeleton_r8pc as R8
    from tomato3d.organs import skeleton_tprox as ST
    ft, fd_, d = paths(pid, v); rd = d / "runs" / pid
    for x in sorted(rd.glob("_infra_failed_*")):
        x.rename(rd / x.name.replace("_infra_failed_", "", 1))
    for x in sorted(rd.iterdir()):
        if x.is_dir() and x.name.startswith("20") and not (x / "kaggle.json").exists():
            x.rename(d / f"_unused_retry_{x.name}")
    ST.OUT = d; R8.RUNS = d / "runs"; ST.skel_dir = lambda _pid, _d=d: _d
    run = R8.run_dir(pid); log(pid, "finish_retry: polling", json.load(open(run / "kaggle.json"))["kernel"], "from", run.name)
    ST.step_finish(pid); shutil.copy2(d / f"labels_tprox_{pid}.npz", ft); log(pid, "T-prox done ->", ft.name, "(infra_retry)")
    return grade(pid)


def grade(pid):
    sk, v = live(pid); ft, fd_, d = paths(pid, v); picks = sk.get("heldout", [])
    if len(picks) < SS.HELDOUT["n"]:
        log(pid, "grade waits:", len(picks), "picks"); return None
    if not (ft.exists() and fd_.exists()):
        log(pid, "grade waits: labels for skelver", v, "missing (press 완료 after the last edit)"); return None
    res = dict(labels=LABELS, plant_id=pid, skelver=v, snapshot_sha16=sha16(d / "skeleton_v1.json"), picks=len(picks), routes={})
    for name, f in (("tprox", ft), ("distance", fd_)):
        z = np.load(f, allow_pickle=True); lab = z["label"].astype(int); st = z["status"].astype(int); uids = [str(u) for u in z["leaf_uids"]]
        uid_of = lambda s, lab=lab, st=st, uids=uids: (uids[lab[int(s)] - 1] if 0 <= int(s) < len(lab) and st[int(s)] == 1 and lab[int(s)] > 0 else None)
        res["routes"][name] = dict(file=f.name, sha16=sha16(f), **SIO.heldout_agreement(picks, uid_of))
    t, dd = res["routes"]["tprox"], res["routes"]["distance"]
    op = "distance" if (t.get("agreement") is not None and dd.get("agreement") is not None and t["agreement"] < dd["agreement"]) else "tprox"
    res.update(operating_route=op, grade=res["routes"][op]["grade"], rule="T-prox unless its agreement < distance (declared fallback)", graded_at=time.strftime("%F %T"))
    json.dump(res, open(OUT / f"grade_{pid}_{v}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float); log(pid, "graded", op, res["grade"]); return res


def summary():
    rows = []
    for pid in PLANTS:
        try:
            sk, v = live(pid)
        except Exception as e:
            rows.append(dict(plant=pid, error=type(e).__name__)); continue
        before = sum(l["status"] in ("active", "new") for l in SIO.from_click_log(STORE_BASE, pid, SP.capture_date(pid), SP.reconstruction_id(pid))["leaves"])
        log_ = SS.paths(ROOT, pid, SP.capture_date(pid))[1]; ed = [e for e in SS.read_edits(log_) if e.get("at", "") >= SIO.V.get("legacy_resplit_at", "")]; c = Counter(e["action"] for e in ed)
        act = [l for l in sk["leaves"] if l["status"] in ("active", "new")]; g = OUT / f"grade_{pid}_{v}.json"; gr = json.load(open(g, encoding="utf-8")) if g.exists() else None
        rows.append(dict(plant=pid, skelver=v, lines_before=before, lines_after=len(act), merges=c["merge"], splits=c["split"], flips=c["flip_direction"], confirms=c["confirm_direction"],
                         one_point_left=sum(len(l["polyline_xyz"]) < 2 for l in act), done_after_trigger=any(e["action"] == "done" and e.get("at", "") >= TRIGGER_FROM for e in ed),
                         heldout=len(sk.get("heldout", [])), minutes=(sk.get("minutes") or {}).get("minutes"), labelled=paths(pid, v)[0].exists() and paths(pid, v)[1].exists(),
                         tprox=(gr["routes"]["tprox"].get("agreement") if gr else None), distance=(gr["routes"]["distance"].get("agreement") if gr else None),
                         route=(gr["operating_route"] if gr else None), grade=(gr["grade"] if gr else "ungraded")))
    json.dump(rows, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    hdr = "| 개체 | 검토 전 선 | 후 | 합침 | 나눔 | 방향 뒤집기 / 확인 | 1점 선 | 완료 | 확인 점 | 분 | T-prox | distance | 운용 경로 | 등급 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(f"| {r['plant']} | {r.get('lines_before')} | {r.get('lines_after')} | {r.get('merges')} | {r.get('splits')} | {r.get('flips')} / {r.get('confirms')} | {r.get('one_point_left')} | {'예' if r.get('done_after_trigger') else '아니오'} | {r.get('heldout')} | {r.get('minutes')} | {r.get('tprox')} | {r.get('distance')} | {r.get('route') or '—'} | {r.get('grade', '—')} |\n" for r in rows)
    (OUT / "summary.md").write_text(hdr + body, encoding="utf-8"); print(hdr + body); return rows


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--id"); ap.add_argument("--grade", action="store_true"); ap.add_argument("--summary", action="store_true"); ap.add_argument("--retry-tprox", default=None, metavar="SKELVER"); ap.add_argument("--finish-retry", default=None, metavar="SKELVER"); a = ap.parse_args(argv)
    if a.summary:
        return summary()
    if a.retry_tprox:
        return retry_tprox(a.id, a.retry_tprox)
    if a.finish_retry:
        return finish_retry(a.id, a.finish_retry)
    return grade(a.id) if a.grade else run_labels(a.id)


if __name__ == "__main__":
    main()
