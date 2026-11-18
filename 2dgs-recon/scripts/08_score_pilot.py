#!/usr/bin/env python3
"""Step 8 - agreement of the T-Prox labels with the pilot plant's human leaf labels (the only labelled plant). Report only; this is not an accuracy for any other plant.
Human labels: example/r3-1b/human_labels/human_instances.npz = 15,650 (surfel position, human leaf) instances from the 4 labelling photos of R3-1b, projected onto its blade surfels.
Metric (as tomato3d.organs.skeleton_r31b_eval): agreement over the instances of the report half (leaves 2, 4, 5, 7) and over all leaves, Wilson 95 % interval, per leaf.
  python scripts/08_score_pilot.py --labels results/r3-1b_tprox/R8_pc_rachis/labels_rachis.npz        # T-Prox
  python scripts/08_score_pilot.py --distance                                                           # the distance-from-skeleton baseline
Surfels of another reconstruction: labels are moved by --radius-mm (default 2.0) nearest-neighbour transfer, see docs/ONEZIP_DIAGNOSIS_KO.md."""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; REPO = HERE.parent; EX = REPO / "example/r3-1b"
sys.path.insert(0, str(REPO / "src"))
from tomato3d.organs import skeleton_assign as SK, skeleton_io as SIO          # noqa: E402

REPORT = (2, 4, 5, 7)


def score(xyz_m, lab, ix, hl, radius_mm=2.0):
    d, j = cKDTree(xyz_m).query(ix); ok = d * 1000.0 <= radius_mm; got, h = lab[j][ok], hl[ok]; out = {}

    def acc(m):
        n = int(m.sum()); k = int((got[m] == h[m]).sum())
        return dict(n=n, correct=k, agreement=round(k / n, 4) if n else None, wilson95=[round(x, 4) for x in SIO.wilson(k, n)] if n else None, none_share=round(float((got[m] == 0).mean()), 4) if n else None)
    out["report_half"] = acc(np.isin(h, REPORT)); out["all_leaves"] = acc(np.ones(len(h), bool)); out["per_leaf"] = {str(L): acc(h == L) for L in range(1, 9)}
    out["instances"] = int(len(hl)); out["dropped_share"] = round(float((~ok).mean()), 4); return out


def distance_route(bundle, groups, skeleton):
    """The distance baseline: every surfel / patch goes to the leaf whose skeleton lines are nearest (skeleton_assign.decide). Same code as skeleton_r31b_eval.route_distance_r31b."""
    Z = np.load(bundle); xyz = Z["xyz"] * 1000.0; n = len(xyz); pf = np.load(groups); pos = {r: i for i, r in enumerate(Z["rows"])}; group = np.full(n, -1); group[[pos[r] for r in pf["rows"]]] = pf["patch_filled"]
    sk = SIO.load(skeleton); back = [t for t in SIO.to_traces(sk) if t["leaf"] > 0 and len(t["pts"]) >= 2]; nums = sorted({t["leaf"] for t in back})
    D = np.stack([cKDTree(np.vstack([SK.densify(np.asarray(t["pts"], float) * 1000.0) for t in back if t["leaf"] == L])).query(xyz)[0] for L in nums], 1)
    lab = np.zeros(n, int); st = np.zeros(n, int); gids = np.unique(group[group >= 0]); order = np.argsort(group); gs = group[order]; starts = np.searchsorted(gs, gids); ends = np.searchsorted(gs, gids, side="right")
    med = np.stack([np.median(D[order[a:b]], 0) for a, b in zip(starts, ends)]); gl, gst = SK.decide(med)
    for l_, s_, a, b in zip(gl, gst, starts, ends):
        lab[order[a:b]] = l_; st[order[a:b]] = s_
    rest = group < 0; l2, s2 = SK.decide(D[rest]); lab[rest] = l2; st[rest] = s2
    return np.where(st == 1, np.array([0] + nums)[lab], 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(EX / "frozen_run/R8_pc_rachis/labels_rachis.npz")); ap.add_argument("--bundle", default=str(EX / "frozen_run/kaggle_data/r8pc_bundle.npz"))
    ap.add_argument("--human", default=str(EX / "human_labels/human_instances.npz")); ap.add_argument("--distance", action="store_true"); ap.add_argument("--radius-mm", type=float, default=2.0)
    ap.add_argument("--groups", default=str(EX / "groups/patches_filled.npz")); ap.add_argument("--skeleton", default=str(EX / "skeleton/skeleton_v1.json")); ap.add_argument("--json", default=None)
    a = ap.parse_args(argv); H = np.load(a.human); Z = np.load(a.bundle); xyz = Z["xyz"].astype(float)
    lab = distance_route(a.bundle, a.groups, a.skeleton) if a.distance else np.load(a.labels)["label"].astype(int)
    if not a.distance:
        assert np.array_equal(np.load(a.labels)["rows"], Z["rows"]), "the labels do not index this bundle"
    res = score(xyz, lab, H["xyz"], H["human_leaf"].astype(int), a.radius_mm); res["route"] = "distance baseline" if a.distance else "T-Prox"
    res["labels_note"] = "report_only; pilot plant R3-1b only; conditional_on_unvalidated_scale; tprox_unverified_until_heldout"
    r, al = res["report_half"], res["all_leaves"]
    print(f"{res['route']}: report half (leaves 2,4,5,7) {r['agreement']} [{r['wilson95'][0]}, {r['wilson95'][1]}] n={r['n']} | all leaves {al['agreement']} n={al['n']}")
    print("per leaf:", {k: v["agreement"] for k, v in res["per_leaf"].items()})
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
