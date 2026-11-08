"""R3-1b accuracy of the skeleton routes against the HUMAN leaves (contract build/p1_skeleton_layer_20261101_01/CONTRACT.md, Amendment A4; label metric_declared_after_adapter_gate).

Truth = the user's leaflet labels on the 4 training photos (vp numbering), projected on front-visible blade surfels (oversegment.load_base). Report half = human leaves 2, 4, 5, 7.
Routes: (a) distance route from the human-trace skeleton; (b) adapter + R8_pc run (R8_pc labels, and after rachis absorption); (c) frozen R8_pc / R8_pc_rachis labels. Scoring only; nothing is changed.

usage: python -m tomato3d.organs.skeleton_r31b_eval [--run 20261101T130242]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import oversegment as OS
from tomato3d.organs import skeleton_assign as SK
from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs import skeleton_r8pc as R8

ROOT = OS.ROOT; B = R8.B; FROZEN = R8.FROZEN
REPORT_LEAVES = (2, 4, 5, 7); CONFUSIONS = ((4, 5), (6, 7, 8))
LABELS = ["user_skeleton_seeds", "conditional_on_unvalidated_scale", "after_checkpoint_v1", "R3-1b_unaccepted", "metric_declared_after_adapter_gate", "scoring_only"]


def human_projections(Bs):
    """(surfel index into blade set b, human leaf) for every labelled front-visible projection over the 4 training photos."""
    si, hl = [], []
    for vid in Bs["chosen"]:
        p = Bs["prj"][vid]; vis = Bs["VISb"][:, p["k"]] & p["inside"]; ul = Bs["umaps"][vid][p["wi"], p["ui"]]; m = vis & (ul > 0)
        si.append(np.nonzero(m)[0]); hl.append(ul[m].astype(int))
    return np.concatenate(si), np.concatenate(hl)


def score(route_full, rows_full, Bs, si, hl):
    """route_full: label per bundle row (0 = contested / none). Returns the A4 metrics."""
    pos = {int(r): i for i, r in enumerate(rows_full)}; brow = Bs["pl"]["rows"][Bs["b"]]; lab_b = np.array([route_full[pos[int(r)]] for r in brow])
    got = lab_b[si]; out = {}
    def acc(mask):
        n = int(mask.sum()); k = int((got[mask] == hl[mask]).sum()); c = int((got[mask] == 0).sum())
        return dict(n=n, correct=k, accuracy=(round(k / n, 4) if n else None), wilson95=SIO.wilson(k, n), contested_or_none_share=(round(c / n, 4) if n else None))
    out["report_half"] = acc(np.isin(hl, REPORT_LEAVES)); out["all_leaves"] = acc(np.ones(len(hl), bool))
    out["per_leaf"] = {str(L): acc(hl == L) for L in sorted(set(hl.tolist()))}
    conf = {}
    for grp in CONFUSIONS:
        key = "/".join(map(str, grp)); m = np.isin(hl, grp); C = Counter(zip(hl[m].tolist(), got[m].tolist()))
        conf[key] = {f"human {h} -> route {g}": n for (h, g), n in sorted(C.items()) if n >= 20}
    out["confusion"] = conf; return out


def route_distance_r31b(Bs, sk):
    """(a): the distance rule on R3-1b with pipeline leaf numbers, over all bundle rows (blade groups whole = patches_filled; others singly)."""
    Z = np.load(FROZEN / "kaggle_data/r8pc_bundle.npz"); xyz = Z["xyz"] * 1000.0; n = len(xyz)
    pf = np.load(ROOT / "build/p1_stacked_sheets_retry_20261026_01/oversegment/second_attempt/patches_filled.npz"); pos = {r: i for i, r in enumerate(Z["rows"])}
    group = np.full(n, -1); group[[pos[r] for r in pf["rows"]]] = pf["patch_filled"]
    back = [t for t in SIO.to_traces(sk) if t["leaf"] > 0 and len(t["pts"]) >= 2]; nums = sorted({t["leaf"] for t in back})
    D = np.stack([cKDTree(np.vstack([SK.densify(np.asarray(t["pts"], float) * 1000.0) for t in back if t["leaf"] == L])).query(xyz)[0] for L in nums], 1)
    lab = np.zeros(n, int); st = np.zeros(n, int); gids = np.unique(group[group >= 0])
    order = np.argsort(group); gs = group[order]; starts = np.searchsorted(gs, gids); ends = np.searchsorted(gs, gids, side="right")
    med = np.stack([np.median(D[order[a:b]], 0) for a, b in zip(starts, ends)]); gl, gst = SK.decide(med)
    for l_, s_, a, b in zip(gl, gst, starts, ends):
        lab[order[a:b]] = l_; st[order[a:b]] = s_
    rest = group < 0; l2, s2 = SK.decide(D[rest]); lab[rest] = l2; st[rest] = s2
    full = np.where(st == 1, np.array([0] + nums)[lab], 0); return full, Z["rows"], dict(assigned=round(float((st == 1).mean()), 4), contested=round(float((st == 2).mean()), 4), none=round(float((st == 0).mean()), 4))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="20261101T130242"); a = ap.parse_args(argv)
    Bs = OS.load_base(); si, hl = human_projections(Bs); sk = SIO.load(SIO.skeleton_dir(ROOT, "R3-1b", "2026-10-21") / "skeleton_v1.json")
    res = dict(labels=LABELS, truth="human leaflet labels of the 4 training photos (vp numbering) on front-visible blade surfels", report_half_leaves=list(REPORT_LEAVES), projections=int(len(si)), routes={})
    full, rows, shares = route_distance_r31b(Bs, sk); res["routes"]["a_distance_from_trace_skeleton"] = dict(shares=shares, **score(full, rows, Bs, si, hl))
    d = R8.RUNS / "R3-1b" / a.run
    for name, f in (("b_adapter_r8pc", d / "R8_pc/labels.npz"), ("b_adapter_r8pc_rachis", d / "R8_pc_rachis/labels_rachis.npz"), ("c_frozen_r8pc", FROZEN / "R8_pc/labels.npz"), ("c_frozen_r8pc_rachis", FROZEN / "R8_pc_rachis/labels_rachis.npz")):
        if f.exists():
            z = np.load(f); res["routes"][name] = dict(file=str(f.relative_to(ROOT)), coverage=round(float((z["label"] > 0).mean()), 4), **score(z["label"].astype(int), z["rows"], Bs, si, hl))
    res["frozen_self_test_note"] = "R8_pc's 0.833 = hide-and-recover of 20 % band surfels of leaves 2/4/5/7 against its own stage-1 seeds; a different metric, not comparable to the rows above"
    json.dump(res, open(B / "r31b_route_accuracy.json", "w", encoding="utf-8"), indent=1, default=float)
    for k, r in res["routes"].items():
        print(k, "report", r["report_half"]["accuracy"], r["report_half"]["wilson95"], "all", r["all_leaves"]["accuracy"], "contested", r["report_half"]["contested_or_none_share"], "per-leaf", {L: v["accuracy"] for L, v in r["per_leaf"].items()})
    return res


if __name__ == "__main__":
    main()
