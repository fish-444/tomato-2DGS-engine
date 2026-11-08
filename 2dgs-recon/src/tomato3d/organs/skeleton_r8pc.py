"""Main route for the skeleton layer: skeleton -> stage-1 seeds -> frozen R8_pc (pc_absorb.py, Kaggle CPU) -> frozen rachis absorption (pc_rachis.py, local).
Contract: build/p1_skeleton_layer_20261101_01/CONTRACT.md, Amendment A2. Frozen code is imported or copied unchanged (hashes recorded); only inputs are built here.

Per individual run folder  build/p1_skeleton_layer_20261101_01/r8pc/<plant>/<stamp>/ mirrors the frozen layout: kaggle_data/r8pc_bundle.npz, R8_pc/labels.npz, R8_pc_rachis/labels_rachis.npz.

usage: python -m tomato3d.organs.skeleton_r8pc --step axis|bundle|push|poll|download|rachis|gate|score --id n_04|R3-1b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs import skeleton_propagate as SP

ROOT = SP.ROOT
B = ROOT / "build/p1_skeleton_layer_20261101_01"; RUNS = B / "r8pc"
FROZEN = ROOT / "build/p1_pc_absorb_20261027_01"
ADAPT = dict(rachis_mm=3.0, blade_mm=8.0, blade_margin_mm=5.0)
AXIS_REF_SHARE, AXIS_BAND = 0.293, 0.5
GATE = dict(agreement_min=0.95)
LABELS = SP.PROP_LABELS + ["frozen_code_reuse", "main_route_r8pc"]
ENV_UTF = dict(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def sha(p):
    return SIO.sha_file(p)[:16]


def hashes():
    return {k: sha(ROOT / v) for k, v in dict(pc_absorb="tomato3d/organs/pc_absorb.py", pc_rachis="tomato3d/organs/pc_rachis.py", compound_cluster="tomato3d/organs/compound_cluster.py", this="tomato3d/organs/skeleton_r8pc.py",
                                                frozen_bundle="build/p1_pc_absorb_20261027_01/kaggle_data/r8pc_bundle.npz", frozen_r8pc_labels="build/p1_pc_absorb_20261027_01/R8_pc/labels.npz").items()}


def run_dir(pid, stamp=None):
    d = RUNS / pid
    if stamp is None:
        subs = sorted(x for x in d.glob("*") if x.is_dir()) if d.exists() else []
        if not subs:
            raise SystemExit(f"no run folder for {pid}")
        return subs[-1]
    return d / stamp


# ---------------------------------------------------------------- axis classification (frozen Q1 on the S1 proxy, translated frame)
_Q1 = {}


def q1():
    if "q1" not in _Q1:
        from tomato3d.organs import compound_cluster as CC
        q, f2, t = CC.q1_and_forms(CC.OUT2); _Q1["q1"] = q; _Q1["CC"] = CC
    return _Q1["q1"], _Q1["CC"]


def step_axis(pid):
    """axis_kind for one of the nine: S1 proxy (`axis_like` of merge/groups.npz) -> frozen Q1 stem-like (1) / rachis-like (2); blade 0. Flag axis_share_unusual by the A2 rule."""
    g = np.load(SP.ALL3 / pid / "merge" / "groups.npz"); xyz = g["xyz"].astype(float); ax = g["axis_like"].astype(bool); q, CC = q1()
    low = xyz[xyz[:, 2] <= np.percentile(xyz[:, 2], 2)]; base = np.array([np.median(low[:, 0]), np.median(low[:, 1]), float(np.percentile(xyz[:, 2], 1))])
    X = xyz - base + np.array([CC.POT_CENTRE[0], CC.POT_CENTRE[1], CC.POT_TOP])                       # plant base -> R3-1b constants; features are otherwise translation-invariant
    H = float(np.percentile(X[:, 2], 99) - CC.POT_TOP); R = float(np.percentile(np.linalg.norm(X[:, :2] - CC.POT_CENTRE, axis=1), 99))
    ak = np.zeros(len(xyz), np.int8)
    if ax.any():
        AF = CC.axis_point_features(X[ax], X[ax], H, R); stem = q.stem_like(AF); ak[np.nonzero(ax)[0]] = np.where(stem, 1, 2)
    share = float(ax.mean()); flag = not (AXIS_REF_SHARE * (1 - AXIS_BAND) <= share <= AXIS_REF_SHARE * (1 + AXIS_BAND))
    d = RUNS / pid; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "axis_labels.npz", rows=g["rows"], axis_kind=ak)
    res = dict(labels=LABELS, plant_id=pid, axis_source="S1 PCA proxy (3D-first contract) + frozen Q1 (forms %s)" % q.forms, axis_share=round(share, 4), axis_share_r31b=AXIS_REF_SHARE,
               axis_share_unusual=flag, stem_like=int((ak == 1).sum()), rachis_like=int((ak == 2).sum()), q1=dict(forms=q.forms, coef=q.m.coef_.round(4).tolist(), intercept=q.m.intercept_.round(4).tolist()),
               translation_m=base.round(5).tolist(), hashes=hashes())
    json.dump(res, open(d / "axis.json", "w", encoding="utf-8"), indent=1); print(json.dumps({k: res[k] for k in ("plant_id", "axis_share", "axis_share_unusual", "stem_like", "rachis_like")})); return res


# ---------------------------------------------------------------- adapter seeds
def seeds(xyz_m, axis_kind, group, leaf_polys, rachis_any=False):
    """stage1 seeds: rachis-like surfels within 3 mm of one leaf's 1 mm polyline(s); 3D-first groups (whole) with median distance <= 8 mm to one leaf and >= 13 mm to every other.
    leaf_polys[k] = list of polylines (m) of leaf k + 1 (a leaf traced twice has two)."""
    K = len(leaf_polys); n = len(xyz_m); st = np.zeros(n, np.int16)
    if K == 0:
        return st, dict(rachis_seeds=0, blade_groups=0, blade_seeds=0)
    D = np.stack([cKDTree(np.vstack([SIO.resample(np.asarray(p, float)) for p in polys])).query(xyz_m)[0] for polys in leaf_polys], 1) * 1000.0
    o = np.argsort(D, 1); best = D[np.arange(n), o[:, 0]]; second = D[np.arange(n), o[:, 1]] if K > 1 else np.full(n, np.inf)
    rach = ((axis_kind == 2) | rachis_any) & (best <= ADAPT["rachis_mm"]) & (second > ADAPT["rachis_mm"]); st[rach] = o[rach, 0] + 1      # T-prox (A5): rachis_any drops the axis-label condition
    nb = 0; ng = 0
    for gid in np.unique(group[group >= 0]):
        m = (group == gid) & (axis_kind == 0)
        if not m.any():
            continue
        med = np.median(D[m], 0); oo = np.argsort(med)
        if med[oo[0]] <= ADAPT["blade_mm"] and (K == 1 or med[oo[1]] >= ADAPT["blade_mm"] + ADAPT["blade_margin_mm"]):
            st[m] = oo[0] + 1; nb += int(m.sum()); ng += 1
    return st, dict(rachis_seeds=int(rach.sum()), blade_groups=ng, blade_seeds=nb)


def traces_arrays(sk):
    pts, lf, ids = [], [], []
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new") and len(l["polyline_xyz"]) >= 2]      # same filter and order as leaf_map / seeds (a 1-point line used to shift the numbering)
    for k, l in enumerate(live):
        P = np.asarray(l["polyline_xyz"], float)
        d = SIO.resample(P, 0.001); pts.append(d); lf.append(np.full(len(d), k + 1)); ids.append(np.full(len(d), k))
    uids = [l["leaf_uid"] for l in live]
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int), uids
    return np.vstack(pts), np.concatenate(lf), np.concatenate(ids), uids


def step_bundle(pid, mode="frozen_axis"):
    """Build kaggle_data/r8pc_bundle.npz in the frozen layout for one individual (R3-1b: the frozen bundle with stage1 := adapter seeds, traces := skeleton export)."""
    from tomato3d.organs import chain_r6 as C6
    stamp = time.strftime("%Y%m%dT%H%M%S"); d = RUNS / pid / stamp; (d / "kaggle_data").mkdir(parents=True, exist_ok=True)
    if pid == "R3-1b":
        Z = dict(np.load(FROZEN / "kaggle_data/r8pc_bundle.npz")); sk = SIO.load(SIO.skeleton_dir(ROOT, "R3-1b", "2026-10-21") / "skeleton_v1.json")
        pf = np.load(ROOT / "build/p1_stacked_sheets_retry_20261026_01/oversegment/second_attempt/patches_filled.npz"); pos = {r: i for i, r in enumerate(Z["rows"])}
        group = np.full(len(Z["xyz"]), -1); group[[pos[r] for r in pf["rows"]]] = pf["patch_filled"]
        back = [t for t in SIO.to_traces(sk) if t["leaf"] > 0 and len(t["pts"]) >= 2]                      # frozen numbering: leaf = the pipeline leaf number carried by the trace (10 traces -> 8 leaves)
        nums = sorted({t["leaf"] for t in back}); assert nums == list(range(1, len(nums) + 1)), nums
        leaf_polys = [[t["pts"] for t in back if t["leaf"] == L] for L in nums]
        st, sinfo = seeds(Z["xyz"], Z["axis_kind"], group, leaf_polys, rachis_any=(mode == "prox"))
        from tomato3d.organs import compound_cluster as CC
        pts, lf, ids = [], [], []
        for t in back:                                                                                 # identical to pc_absorb_prep: densify per trace, leaf = pipeline leaf, id = trace index
            dd = CC.densify(np.asarray(t["pts"], float)); pts.append(dd); lf.append(np.full(len(dd), t["leaf"])); ids.append(np.full(len(dd), t["index"]))
        Z.update(stage1=st, r7=np.zeros(len(st), np.int16), trace_pts=np.vstack(pts), trace_leaf=np.concatenate(lf), trace_id=np.concatenate(ids)); src = dict(frozen_bundle=sha(FROZEN / "kaggle_data/r8pc_bundle.npz"), groups="second_attempt/patches_filled.npz")
        leaf_map = {L: "+".join(t["leaf_uid"] for t in back if t["leaf"] == L) for L in nums}
    else:
        g = np.load(SP.ALL3 / pid / "merge" / "groups.npz"); geo = np.load(SP.ALL3 / pid / "work" / "geometry.npz"); gj = json.load(open(SP.ALL3 / pid / "geometry.json", encoding="utf-8"))
        ax = np.load(RUNS / pid / "axis_labels.npz"); assert np.array_equal(ax["rows"], g["rows"])
        sf = sio.read_surfels_full(ROOT / f"build/p1_recon_all_20261025_01/outputs/{pid}/R3-1b-style_{pid}_surfels_full.ply"); rows = g["rows"]; n = len(rows)
        N = sf["normal"][rows].astype(float); N[geo["b"]] = geo["normal"]; sidx = np.zeros(n, np.int16); nsh = np.ones(n, np.int16); sidx[geo["b"]] = geo["index"]; nsh[geo["b"]] = geo["n_sheets"]
        sk = SIO.load(SIO.skeleton_dir(ROOT, pid, SP.capture_date(pid)) / "skeleton_v1.json"); tp, tl, tid, uids = traces_arrays(sk)
        live_polys = [np.asarray(l["polyline_xyz"], float) for l in sk["leaves"] if l["status"] in ("active", "new") and len(l["polyline_xyz"]) >= 2]
        st, sinfo = seeds(g["xyz"].astype(float), ax["axis_kind"], g["group"].astype(int), [[p] for p in live_polys], rachis_any=(mode == "prox")); ts = float(gj["t_sheet_mm"])
        Z = dict(xyz=g["xyz"].astype(np.float64), normal=N, scale=sf["scale"][rows][:, :2].astype(np.float64), tu=sf["tu"][rows], tv=sf["tv"][rows], rgb=sf["rgb"][rows].astype(np.float32), opacity=sf["opacity"][rows].astype(np.float32),
                 axis_kind=ax["axis_kind"].astype(np.int8), sheet_index=sidx, n_sheets=nsh, stage1=st, r7=np.zeros(n, np.int16), trace_pts=tp, trace_leaf=tl.astype(np.int64), trace_id=tid.astype(np.int64),
                 t_sheet_mm=np.float64(ts), offset_mm=np.float64(max(ts, 1.0)), reach_mm=np.float64(C6.U_MM), rows=rows)
        src = dict(groups=str((SP.ALL3 / pid / "merge" / "groups.npz").relative_to(ROOT)), geometry=str((SP.ALL3 / pid / "work" / "geometry.npz").relative_to(ROOT)), axis=str((RUNS / pid / "axis_labels.npz").relative_to(ROOT)))
        leaf_map = {k + 1: u for k, u in enumerate([l["leaf_uid"] for l in sk["leaves"] if l["status"] in ("active", "new") and len(l["polyline_xyz"]) >= 2])}
    np.savez_compressed(d / "kaggle_data" / "r8pc_bundle.npz", **Z)
    meta = dict(labels=LABELS, plant_id=pid, stamp=stamp, mode=mode, adapter=ADAPT, seeds=sinfo, stage1_labelled=int((Z["stage1"] > 0).sum()), leaves=len(leaf_map), leaf_map={str(k): v for k, v in leaf_map.items()}, sources=src, hashes=hashes(),
                skeleton_sha256=SIO.sha_file(SIO.skeleton_dir(ROOT, pid, "2026-10-21" if pid == "R3-1b" else SP.capture_date(pid)) / "skeleton_v1.json"))
    json.dump(meta, open(d / "bundle_meta.json", "w", encoding="utf-8"), indent=1); print(json.dumps({k: meta[k] for k in ("plant_id", "stamp", "seeds", "stage1_labelled", "leaves")})); return d


# ---------------------------------------------------------------- Kaggle (frozen pc_absorb.py copied unchanged)
def kg(*a):
    import os
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, **ENV_UTF)); return r.returncode, r.stdout + r.stderr


def slugs(pid, stamp):
    s = (pid + stamp[2:8] + stamp[9:13]).replace("_", "").lower(); return f"skr8pc-data-{s}", f"skr8pc-run-{s}"          # dataset slug != kernel title (Kaggle silently fails otherwise)


def step_push(pid, stamp=None):
    d = run_dir(pid, stamp); meta = json.load(open(d / "bundle_meta.json", encoding="utf-8")); ds, ks = slugs(pid, meta["stamp"])
    (d / "kaggle_data" / "dataset-metadata.json").write_text(json.dumps(dict(title=ds, id=f"dolgfish/{ds}", licenses=[dict(name="CC0-1.0")])))
    rc, o = kg("datasets", "create", "-p", str(d / "kaggle_data"), "--dir-mode", "zip"); print("dataset", rc, o.strip()[-120:])
    k = d / "kaggle_kernel"; k.mkdir(exist_ok=True); shutil.copy(ROOT / "tomato3d/organs/pc_absorb.py", k / "pc_absorb.py")
    (k / "kernel-metadata.json").write_text(json.dumps(dict(id=f"dolgfish/{ks}", title=ks, code_file="pc_absorb.py", language="python", kernel_type="script", is_private=True, enable_gpu=False, enable_internet=False,
                                                          dataset_sources=[f"dolgfish/{ds}"], competition_sources=[], kernel_sources=[])))
    time.sleep(20); rc, o = kg("kernels", "push", "-p", str(k)); print("kernel", rc, o.strip()[-120:]); json.dump(dict(dataset=ds, kernel=ks, pushed=time.strftime("%Y-%m-%d %H:%M:%S"), pc_absorb_sha=sha(k / "pc_absorb.py")), open(d / "kaggle.json", "w"), indent=1)


def step_poll(pid, stamp=None, wait=True):
    d = run_dir(pid, stamp); kj = json.load(open(d / "kaggle.json")); slug = "dolgfish/" + kj["kernel"]
    none = 0
    while True:
        rc, o = kg("kernels", "status", slug); m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", o); st = m.group(1) if m else None; print(time.strftime("%H:%M:%S"), st, flush=True)
        none = none + 1 if st is None else 0
        if st in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED") or not wait or none >= 8:        # 8 unreadable statuses in a row (~8 min): kernel missing or network down -> give up instead of waiting forever
            return st
        time.sleep(60)


def step_download(pid, stamp=None):
    d = run_dir(pid, stamp); kj = json.load(open(d / "kaggle.json")); slug = "dolgfish/" + kj["kernel"]; got = d / "kaggle_out"; got.mkdir(exist_ok=True)
    for _ in range(6):
        kg("kernels", "output", slug, "-p", str(got), "-q")
        lab = list(got.rglob("labels.npz"))
        if lab:
            break
        time.sleep(60)
    lab = list(got.rglob("labels.npz"))
    if not lab:
        print("no labels downloaded"); return None
    (d / "R8_pc").mkdir(exist_ok=True); shutil.copy(lab[0], d / "R8_pc" / "labels.npz")
    for f in got.rglob("results.json"):
        shutil.copy(f, d / "R8_pc" / "results.json"); break
    print("R8_pc labels:", d / "R8_pc" / "labels.npz"); return d


# ---------------------------------------------------------------- rachis absorption (frozen pc_rachis.py with the run folder as D0)
def step_rachis(pid, stamp=None):
    from tomato3d.organs import pc_rachis as PR
    d = run_dir(pid, stamp); PR.D0 = d; PR.OUT = d / "R8_pc_rachis"; PR.main(); print("rachis labels:", d / "R8_pc_rachis" / "labels_rachis.npz")


# ---------------------------------------------------------------- R3-1b gate and per-individual scoring
def step_gate(stamp=None):
    d = run_dir("R3-1b", stamp); new = np.load(d / "R8_pc" / "labels.npz"); old = np.load(FROZEN / "R8_pc" / "labels.npz"); assert np.array_equal(new["rows"], old["rows"])
    m = old["label"] > 0; agree = float((new["label"][m] == old["label"][m]).mean()); cov_new = float((new["label"] > 0).mean()); cov_old = float(m.mean())
    both = m & (new["label"] > 0); agree_both = float((new["label"][both] == old["label"][both]).mean()) if both.any() else None
    out = dict(labels=LABELS + ["simulated_from_human_traces"], agreement_over_frozen_labelled=round(agree, 4), agreement_where_both_labelled=(round(agree_both, 4) if agree_both is not None else None),
               coverage_new=round(cov_new, 4), coverage_frozen=round(cov_old, 4), gate=GATE, passed=bool(agree >= GATE["agreement_min"]), run=str(d.relative_to(ROOT)), hashes=hashes())
    if (d / "R8_pc_rachis" / "labels_rachis.npz").exists():
        nr = np.load(d / "R8_pc_rachis" / "labels_rachis.npz"); orr = np.load(FROZEN / "R8_pc_rachis" / "labels_rachis.npz"); mm = orr["label"] > 0
        out["rachis_agreement_over_frozen_labelled"] = round(float((nr["label"][mm] == orr["label"][mm]).mean()), 4)
    json.dump(out, open(B / "r31b_adapter_gate.json", "w", encoding="utf-8"), indent=1); print(json.dumps(out, indent=1)); return out


def step_score(pid, stamp=None):
    """Main route vs distance route for one individual: shares, held-out agreement (Wilson), grade, contested light share; per-leaf light only for grade A/B on the main route."""
    d = run_dir(pid, stamp); meta = json.load(open(d / "bundle_meta.json", encoding="utf-8")); lm = {int(k): v for k, v in meta["leaf_map"].items()}
    lab_f = d / "R8_pc_rachis" / "labels_rachis.npz" if (d / "R8_pc_rachis" / "labels_rachis.npz").exists() else d / "R8_pc" / "labels.npz"
    z = np.load(lab_f); lab = z["label"].astype(int); rows = z["rows"]; p1 = z["p1"] if "p1" in z.files else None
    sk = SIO.load(SIO.skeleton_dir(ROOT, pid, SP.capture_date(pid)) / "skeleton_v1.json"); pos = {int(r): i for i, r in enumerate(rows)}
    status = np.where(lab > 0, 1, 0); uid_of = lambda s: (lm.get(int(lab[pos[int(s)]])) if int(s) in pos and lab[pos[int(s)]] > 0 else None)
    ho = SIO.heldout_agreement(sk.get("heldout", []), uid_of) if sk.get("heldout") else None
    light = SP.contested_light(pid, rows, np.where(lab > 0, 1, 0)); g = SP.groups(pid)
    dist = json.load(open(SIO.skeleton_dir(ROOT, pid, SP.capture_date(pid)) / "summary.json", encoding="utf-8"))
    per_leaf = None
    if ho and ho["grade"] in ("A", "B"):
        f = SP.LIGHT_NINE / pid / "per_surfel_light.npz"
        if f.exists():
            L = np.load(f, allow_pickle=True)
            if np.array_equal(L["ply_row"], rows):
                per_leaf = {u: dict(absorbed_ru_mm2=float(L["absorbed_ru_mm2"][lab == k].sum()), area_b_mm2=float(L["area_b_silhouette_mm2"][lab == k].sum()), n=int((lab == k).sum())) for k, u in lm.items()}
    res = dict(labels=LABELS, plant_id=pid, run=str(d.relative_to(ROOT)), label_file=lab_f.name, main=dict(assigned=round(float((lab > 0).mean()), 4), contested_or_none=round(float((lab == 0).mean()), 4), heldout=ho or "pending: no held-out clicks", contested_light_share=light),
               distance=dict(assigned=dist["assigned"], contested=dist["contested"], none=dist["none"], heldout=dist["heldout"]), grade_main=(ho["grade"] if ho else "pending"),
               per_leaf_light=(per_leaf if per_leaf else ("not reported: grade " + (ho["grade"] if ho else "pending"))), axis=(json.load(open(RUNS / pid / "axis.json", encoding="utf-8")) if (RUNS / pid / "axis.json").exists() else None))
    json.dump(res, open(d / "score.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float); print(json.dumps({k: res[k] for k in ("plant_id", "main", "distance", "grade_main")}, ensure_ascii=False, default=float)); return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--id", required=True); ap.add_argument("--stamp", default=None); ap.add_argument("--mode", default="frozen_axis", choices=("frozen_axis", "prox")); a = ap.parse_args(argv)
    if a.step == "axis": step_axis(a.id)
    elif a.step == "bundle": step_bundle(a.id, a.mode)
    elif a.step == "push": step_push(a.id, a.stamp)
    elif a.step == "poll": step_poll(a.id, a.stamp)
    elif a.step == "download": step_download(a.id, a.stamp)
    elif a.step == "rachis": step_rachis(a.id, a.stamp)
    elif a.step == "gate": step_gate(a.stamp)
    elif a.step == "score": step_score(a.id, a.stamp)


if __name__ == "__main__":
    main()
