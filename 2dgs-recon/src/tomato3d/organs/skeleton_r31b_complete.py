"""Complete the R3-1b simulated skeleton with a line for every human leaf that had no trace (contract Amendment A5, label sim_skeleton_completed), then rescore once.

usage: python -m tomato3d.organs.skeleton_r31b_complete --step complete | score --run-b <stamp> --run-prox <stamp>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import oversegment as OS
from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs import skeleton_r31b_eval as EV
from tomato3d.organs import skeleton_r8pc as R8

ROOT = OS.ROOT; B = R8.B
SLICE = 0.05


def human_surfel_labels(Bs):
    """Per blade surfel (index into b): majority human leaf over its labelled front-visible projections (0 = never labelled)."""
    si, hl = EV.human_projections(Bs); votes = {}
    for i, L in zip(si.tolist(), hl.tolist()):
        votes.setdefault(i, Counter())[L] += 1
    lab = np.zeros(len(Bs["b"]), int)
    for i, c in votes.items():
        lab[i] = c.most_common(1)[0][0]
    return lab


def synthetic_line(X, axis_xy):
    """base nearest the axis -> medial points at 25/50/75 % -> tip, along the first PCA axis oriented from the base."""
    r = np.linalg.norm(X[:, :2] - axis_xy, axis=1); base = X[int(np.argmin(r))]
    w, v = np.linalg.eigh(np.cov((X - X.mean(0)).T)); d = v[:, -1]
    if (X.mean(0) - base) @ d < 0:
        d = -d
    t = (X - base) @ d; L = float(t.max()); tip = X[int(np.argmax(t))]; pts = [base]
    for f in (0.25, 0.5, 0.75):
        m = np.abs(t - f * L) <= SLICE * L
        if m.any():
            pts.append(np.median(X[m], 0))
    pts.append(tip); return np.array(pts), L


def step_complete():
    Bs = OS.load_base(); xyz = Bs["xyz"]; lab = human_surfel_labels(Bs)
    d = SIO.skeleton_dir(ROOT, "R3-1b", "2026-10-21"); sk = SIO.load(d / "skeleton_v1.json")
    have = {l["attrs"].get("pipeline_leaf") for l in sk["leaves"]}; added = []
    for L in range(1, 9):
        if L in have:
            continue
        X = xyz[lab == L]
        if len(X) < 10:
            added.append(dict(leaf=L, note=f"only {len(X)} labelled surfels: no line")); continue
        P, length = synthetic_line(X, np.asarray(CC.POT_CENTRE, float))
        lf = SIO.add_leaf(sk, P, source="sim_skeleton_completed", attrs=dict(pipeline_leaf=L, trace_index=1000 + L, trace_key=f"sim_leaf{L}", from_human_surfels=int(len(X))))
        added.append(dict(leaf=L, leaf_uid=lf["leaf_uid"], points=len(P), length_mm=round(length * 1000, 1), surfels=int(len(X))))
    sk["labels"] = list(sk["labels"]) + ["sim_skeleton_completed"]; f = SIO.save(sk, ROOT)
    res = dict(lines_per_leaf={str(L): [l["leaf_uid"] for l in sk["leaves"] if l["attrs"].get("pipeline_leaf") == L] for L in range(1, 9)}, added=added, file=str(f.relative_to(ROOT)))
    json.dump(res, open(B / "r31b_sim_completion.json", "w", encoding="utf-8"), indent=1); print(json.dumps(res, indent=1)); return res


def step_score(run_b, run_prox):
    Bs = OS.load_base(); si, hl = EV.human_projections(Bs); sk = SIO.load(SIO.skeleton_dir(ROOT, "R3-1b", "2026-10-21") / "skeleton_v1.json")
    res = dict(labels=EV.LABELS + ["sim_skeleton_completed", "reference_only_scoring"], routes={})
    full, rows, shares = EV.route_distance_r31b(Bs, sk); res["routes"]["a_distance_completed"] = dict(shares=shares, **EV.score(full, rows, Bs, si, hl))
    for name, stamp in (("b_adapter_r8pc_completed", run_b), ("t_prox_completed", run_prox)):
        if not stamp:
            continue
        dd = R8.RUNS / "R3-1b" / stamp; f = dd / "R8_pc_rachis/labels_rachis.npz"
        f = f if f.exists() else dd / "R8_pc/labels.npz"; z = np.load(f)
        res["routes"][name] = dict(run=stamp, file=f.name, coverage=round(float((z["label"] > 0).mean()), 4), **EV.score(z["label"].astype(int), z["rows"], Bs, si, hl))
    a = res["routes"]["a_distance_completed"]["report_half"]; Bv = a["accuracy"]; Bc = a["contested_or_none_share"]
    if "t_prox_completed" in res["routes"]:
        t = res["routes"]["t_prox_completed"]["report_half"]
        res["t_prox_gate"] = dict(baseline_B=Bv, need=round(Bv + 0.05, 4), accuracy=t["accuracy"], contested=t["contested_or_none_share"], contested_max=Bc,
                                  passed=bool(t["accuracy"] >= Bv + 0.05 and t["contested_or_none_share"] <= Bc))
    json.dump(res, open(B / "r31b_completed_scores.json", "w", encoding="utf-8"), indent=1, default=float)
    for k, r in res["routes"].items():
        print(k, "report", r["report_half"]["accuracy"], r["report_half"]["wilson95"], "contested", r["report_half"]["contested_or_none_share"], {L: v["accuracy"] for L, v in r["per_leaf"].items()})
    print("gate", res.get("t_prox_gate")); return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--run-b"); ap.add_argument("--run-prox"); a = ap.parse_args(argv)
    step_complete() if a.step == "complete" else step_score(a.run_b, a.run_prox)


if __name__ == "__main__":
    main()
