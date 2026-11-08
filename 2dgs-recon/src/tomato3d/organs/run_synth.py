"""Synthetic bench run: python -m tomato3d.organs.run_synth --split train|test [--truth-labels]
Writes <run>/synth_results_<split>[_truthlabels].json (pooled metrics, pass rules, per individual)."""

from __future__ import annotations

import argparse
import json

import numpy as np

from tomato3d.organs import contract as C
from tomato3d.organs import evaluate as EV
from tomato3d.organs import features as FE
from tomato3d.organs import labeller as LB
from tomato3d.organs import pipeline as PL


def truth_of(ind: str) -> dict:
    bs = json.load(open(C.RUN / "bench_summary.json"))
    return next(i for i in bs["individuals"] if i["individual"] == ind)["truth"]


def one(ind: str, truth_labels: bool, lab: LB.Labeller | None):
    z, F = LB.load_bench(ind); xyz, nrm = z["xyz"].astype(np.float64), z["normal"]
    if truth_labels:
        is_axis = z["axis_label"] == 1; is_blade = z["axis_label"] == 0
    else:
        is_axis = FE.smooth(xyz, lab.p_axis(F) >= C.LABEL_THR); is_blade = ~is_axis
    r = PL.run(xyz, nrm, is_axis, C.POT_CENTRE_M, is_blade)
    truth = truth_of(ind); c = r["counts"]
    pred_leaf = np.array([r["leaf_of_cluster"].get(k, "") if k >= 0 else "" for k in r["cl"]], dtype=object).astype(str)
    is_blade_truth = z["axis_label"] == 0
    mem = EV.membership(z["leaf"].astype(str), pred_leaf, is_blade_truth, z["obj"].astype(str))
    lf1 = EV.leaflet_f1(z["obj"].astype(str), is_blade_truth, r["cl"])
    ncl = int(r["cl"].max()) + 1
    blade_cl = [k for k in range(ncl) if is_blade_truth[r["cl"] == k].mean() >= 0.5]
    row = dict(individual=ind, node=EV.node_metrics(c["leaf_nodes"], truth["ranks"]), membership_point=mem["point"], membership_leaflet=mem["leaflet"],
               leaflet=lf1, side=EV.side_axes(c["side_axes"], truth["side_axes"]),
               ambiguous=c["ambiguous"] / max(ncl, 1), unattached=c["unattached"] / max(ncl, 1),
               unattached_blade_truth=float(np.mean([r["leaf_of_cluster"].get(k) == "unattached" for k in blade_cl])) if blade_cl else 0.0,
               counts=c, matched=mem["matched"])
    print(f"{ind:11s} nodes {c['leaf_nodes']}/{truth['ranks']}  leaves {c['leaves']}/{truth['main_leaves']}  side {c['side_axes']}/{truth['side_axes']}  "
          f"memb {mem['point']:.2f}  F1 {lf1['f1']:.2f}  clusters {ncl}/{truth['leaflets']}  unatt {row['unattached']:.2f}  stubs {c['stubs']}  trusses {c['truss_candidates']}", flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=("train", "test"), required=True); ap.add_argument("--truth-labels", action="store_true")
    ap.add_argument("--inds", nargs="*"); a = ap.parse_args(argv)
    lab = None if a.truth_labels else LB.Labeller.load()
    rows = [one(ind, a.truth_labels, lab) for ind in (a.inds or C.BENCH[a.split])]
    pooled = EV.pooled(rows)
    passes = dict(node_exact=pooled["node_exact"] >= C.PASS["node_exact_min"], node_mae=pooled["node_mae"] <= C.PASS["node_mae_max"],
                  membership=pooled["membership_point"] >= C.PASS["leaf_membership_min"], leaflet_f1=pooled["leaflet_f1"] >= C.PASS["leaflet_f1_min"])
    out = dict(**C.TAG, split=a.split, truth_labels=a.truth_labels, revision=4, pooled=pooled, pass_rules=C.PASS, passes=passes, all_pass=all(passes.values()), individuals=rows)
    name = f"synth_results_{a.split}{'_truthlabels' if a.truth_labels else ''}.json"
    json.dump(out, open(C.RUN / name, "w"), indent=1, default=float)
    print(json.dumps(dict(pooled=pooled, passes=passes), indent=1, default=float))


if __name__ == "__main__":
    main()
