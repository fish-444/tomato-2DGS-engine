"""Report-only diagnosis of the 3D-first SAM merge results (first and second attempt): where the held-out labelled pixels go (no group / group without a number /
group whose majority number is wrong / right group but other pixel number / correct), the per-leaf IoU table, and the over-merges recounted at compound-leaf level.
No new grouping, no thresholds; reads existing outputs only.

usage: python -m tomato3d.organs.oversegment_diagnose
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from tomato3d.organs import oversegment as OS
from tomato3d.organs import oversegment_second as O2
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import userlabel as UL

D2 = O2.OUT2


def heldout_breakdown(B, group_s, glab):
    pl, b = B["pl"], B["b"]; full = np.zeros(len(pl["xyz"]), np.int16); full[b] = np.where(group_s >= 0, group_s + 1, 0)
    H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = f"20261021_{B['held']}_iOS.jpg"; col = H2id[:, pl["views"].index(hv)]
    raw, _ = UL.label_maps(UL.OUT, [B["held"]]); hm = UL.remap(raw, B["nmap"])[B["held"]]; hm = np.where(hm > 0, hm, 0)
    gimg = UL.render_ids(pl, full, hv, col).astype(int) - 1; lab = hm > 0; n = int(lab.sum()); cat = Counter(); per_group = {}
    for g in np.unique(gimg[lab & (gimg >= 0)]):
        m = lab & (gimg == g); c = Counter(hm[m].tolist()); per_group[int(g)] = dict(pixels=int(m.sum()), number=glab.get(int(g)), heldout_majority=c.most_common(1)[0][0])
    for g, r in per_group.items():
        m = lab & (gimg == g)
        if r["number"] is None: cat["group_without_number"] += r["pixels"]; continue
        right = int((hm[m] == r["number"]).sum())
        if r["number"] != r["heldout_majority"]: cat["group_majority_number_wrong"] += r["pixels"]; cat["_of_which_pixels_still_right"] += right
        else: cat["correct"] += right; cat["right_group_other_pixel_number"] += r["pixels"] - right
    cat["no_group_rendered"] = int((lab & (gimg < 0)).sum())
    share = {k: round(v / n, 4) for k, v in cat.items()}
    groups = dict(rendered=len(per_group), without_number=sum(1 for r in per_group.values() if r["number"] is None),
                  majority_wrong=sum(1 for r in per_group.values() if r["number"] is not None and r["number"] != r["heldout_majority"]))
    return dict(labelled_pixels=n, share=share, groups=groups)


def compound_overmerges(group_s, B, om):
    """Over-merged groups (leaflet rule) recounted with the user's leaf numbers instead of leaflet masks: same 20 % / half-of-views rule."""
    per_gv = {}
    for vid, p in B["prj"].items():
        vis = B["VISb"][:, p["k"]] & p["inside"] & (group_s >= 0); ul = B["umaps"][vid][p["wi"], p["ui"]]; m = vis & (ul > 0)
        for g, l in zip(group_s[m], ul[m]): per_gv.setdefault((int(g), vid), Counter())[int(l)] += 1
    flags = {}
    for (g, vid), c in per_gv.items():
        n = sum(c.values())
        if n >= OS.B4["min_px"]: flags.setdefault(g, []).append(sum(1 for v in c.values() if v / n >= 0.20) >= 2)
    comp = sorted(g for g, f in flags.items() if sum(f) >= len(f) / 2 and sum(f) > 0)
    return dict(leaflet_rule=len(om), compound_rule_all_groups=len(comp), compound_rule_among_leaflet_overmerges=len(set(comp) & set(om)),
                groups_with_labels=len(flags), compound_share=round(len(comp) / max(len(flags), 1), 4), compound_groups=comp)


def main():
    B = OS.load_base(); z2 = np.load(D2 / "viz_groups.npz"); r2 = json.load(open(D2 / "result.json", encoding="utf-8")); z1 = np.load(OS.OUT / "oversegment" / "groups_4c.npz")
    out = {}
    for name, gs, base in (("first_attempt", z1["group"], z1["group"]), ("second_pre_fill", z2["pre"], z2["pre"]), ("second_post_fill", z2["post"], z2["pre"])):
        rows, _ = SR.purity(base, B["prj"], B["umaps"], B["VISb"]); glab = {g: r["majority"] for g, r in rows.items()}
        out[name] = heldout_breakdown(B, gs, glab)
        out[name]["per_leaf"] = (r2["scoring_proxy_uses_human_numbers"]["post_fill" if name == "second_post_fill" else "pre_fill"]["rows"] if name != "first_attempt"
                                 else json.load(open(OS.OUT / "oversegment" / "step4bcd.json", encoding="utf-8"))["scoring_proxy_uses_human_numbers"]["rows"])
    om = [int(g) for g in r2["over_merge_cases"]]; out["second_post_fill_overmerges"] = compound_overmerges(z2["post"], B, om)
    res = dict(**O2.LABELS, report_only=True, note="diagnosis of existing results; no new attempt", **out)
    json.dump(res, open(D2 / "diagnosis.json", "w", encoding="utf-8"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
