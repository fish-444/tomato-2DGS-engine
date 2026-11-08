"""Pilot (report only): R3-1b split vs leaflet_C_r8pc labels. python -X utf8 pilot_check.py"""
import csv, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
S = np.load(HERE / "split_R3-1b.npz", allow_pickle=True); C = np.load(ROOT / "build/p1_r31b_feature_segmentation_20261031_01/leaflet_C_r8pc.npz")
assert np.array_equal(np.asarray(S["rows"]), np.asarray(C["rows"])), "row order differs"
leaf = S["leaf"].astype(str); lt = S["leaflet"].astype(str); cl = np.asarray(C["label"]).astype(int); A = np.asarray(S["area_b"], float)
rows = []; leaves = []
for u in sorted(set(leaf) - {""}):
    m = leaf == u; pred = sorted(set(lt[m]) - {""})
    cnt = {int(c): int((cl[m] == c).sum()) for c in set(cl[m].tolist()) if c > 0}; n_c = sum(v >= 30 for v in cnt.values())
    for p in pred:
        mp = m & (lt == p); best = (0.0, None)
        for c in cnt:
            mc = m & (cl == c); iou = (mp & mc).sum() / (mp | mc).sum()
            if iou > best[0]:
                best = (float(iou), c)
        rows.append(dict(uid=u, leaflet=p, n=int(mp.sum()), area_mm2=float(A[mp].sum()), best_iou=best[0], best_label=best[1]))
    leaves.append(dict(uid=u, n_pred=len(pred), n_label_ge30=n_c, count_agree=len(pred) == n_c, unlabelled_share=float((cl[m] <= 0).mean())))
with open(HERE / "pilot_check.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(HERE / "pilot_check_leaves.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(leaves[0])); w.writeheader(); w.writerows(leaves)
R = dict(leaflets=len(rows), iou_median=float(np.median([r["best_iou"] for r in rows])), iou_ge_0_5=float(np.mean([r["best_iou"] >= 0.5 for r in rows])),
         leaves=len(leaves), count_agree=int(sum(l["count_agree"] for l in leaves)), note="report only; check set leaflet_C_r8pc (38 seeds + R8_pc, ungraded); count = labels with >= 30 surfels in the compound leaf")
json.dump(R, open(HERE / "pilot_check.json", "w", encoding="utf-8"), indent=1); print(json.dumps(R, indent=1))
