"""P1 stem gap-fill (2026-10-29) — assemble the per-individual outputs from the Kaggle chain rerun (CONTRACT.md §4–§5 + Rev 3).

P1 = weak_prior → the default chain uses the run-2 priors (variant `base`, which reproduces run 2); the photo-derived taper floor and the S4 radius
variants are sensitivity runs only. Writes chain/<id>.json, nodes/<id>.csv, qa/ (overlays + <id>.html), sensitivity.csv, run2_vs_gapfill.csv.
No surfel or area is created; the plant-model folder and run-2 folder are only read.

usage: python -m tomato3d.organs.stem_gapfill
"""
from __future__ import annotations

import csv
import json
import shutil

from tomato3d.organs import light_organs as LO

D = LO.ROOT / "build/p1_stem_gapfill_20261029_01"
K = D / "kernel_out_chain"
RUN2 = LO.ROOT / "build/p1_stem_run2_20261026_01"
CHAINED = {"R3-1b": "R3-1b", "n_03": "n_03_tuned", "n_06": "n_06_tuned", "t2_02": "t2_02_tuned", "t2_05": "t2_05"}
NOT_RUN = {**{k: "no passed rim fit / medium plane (as run 2)" for k in ("n_01", "n_02", "n_04", "n_05", "t2_04")},
           **{f"e_0{i}": "no reconstruction in build/ (no input)" for i in range(1, 7)}}
TAGS = dict(scale="conditional_on_unvalidated_scale", biological_validation="not_established")


def main():
    summ = json.load(open(K / "summary.json"))
    pri = json.load(open(D / "stem_priors_summary.json", encoding="utf-8"))
    rows = {(r["variant"], r["individual"]): r for r in summ["rows"]}
    ilen = pri["all_heights_pooled_labelled"]["internode_upper_bound_ilen_span"]
    for sub in ("chain", "nodes", "qa"): (D / sub).mkdir(exist_ok=True)
    r2 = {r["individual"]: r for r in csv.DictReader(open(RUN2 / "summary.csv", encoding="utf-8"))}
    cmp_rows, sens_rows = [], []
    for ind, key in CHAINED.items():
        ch = json.load(open(K / "base/chain" / f"{key}.json", encoding="utf-8"))
        b = rows[("base", ind)]
        for s in ch["segments"]:
            s["evidence"] = "supported" if s.get("supported") else "gap_open"
        apex = ch["apex_mm"]; ub = ilen.get("max", 1.0) * apex
        nodes = list(csv.DictReader(open(K / "base/nodes" / f"{key}.csv", encoding="utf-8")))
        flagged = [n["node_id"] for n in nodes if n["internode_mm"] not in ("", "None") and float(n["internode_mm"]) > ub]
        sens = {v: {k: rows[(v, ind)].get(k) for k in ("h_lost_mm", "nodes", "supported_segments", "first_seg_radius_mm")}
                for v in ("taper_floor", "r0_x0.7", "r0_x1.3") if (v, ind) in rows}
        out = dict(individual=ind, run2_key=key, default_chain="run-2 priors (variant base; P1 = weak_prior, photo priors not adopted)",
                   prior_from_stem_photos=False, radius_from_rim_ratio=False,
                   prior_note="no rim click in the stem photos and no grid mesh size -> no mm prior; photo ratios reported only (stem_priors_summary.json)",
                   reproduces_run2=bool(abs(float(r2[key]["h_lost_mm"]) - b["h_lost_mm"]) < 0.05 and int(r2[key]["nodes"]) == b["nodes"]),
                   above_h_lost="gap_open (conditional_hidden_state polyline only; no node interpolated from priors)",
                   internode_plausibility=dict(rule="run-2 internode > max photo internode_upper_bound x stem span (ratio x apex)", max_ratio=ilen.get("max"),
                                               limit_mm=round(ub, 1), flagged_node_ids=flagged),
                   S1="no_reference", S2=("not testable" if ch["N1"]["passes"] is None else ("pass" if ch["N1"]["passes"] else "fail")), S3="no_reference",
                   sensitivity=sens, tag="stem_inferred_only", tags=TAGS, chain=ch)
        json.dump(out, open(D / "chain" / f"{ind}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        shutil.copyfile(K / "base/nodes" / f"{key}.csv", D / "nodes" / f"{ind}.csv")
        for f in (K / "base/qa").glob(f"{key}_*.png"): shutil.copyfile(f, D / "qa" / f.name)
        html = (K / "base/qa" / f"{key}.html").read_text(encoding="utf-8").replace("줄기 run2", "줄기 gap-fill (기본 = run2 사전값, weak_prior)")
        (D / "qa" / f"{ind}.html").write_text(html, encoding="utf-8")
        cmp_rows.append([ind, key, r2[key]["h_lost_mm"], b["h_lost_mm"], r2[key]["apex_mm"], b["apex_mm"], round(1 - float(r2[key]["C3_inside"]), 3),
                         b["outside_hull_share"], 1.5, b["first_seg_radius_mm"], r2[key]["nodes"], b["nodes"], len(flagged), "stem_inferred_only"])
        for v in ("base", "taper_floor", "r0_x0.7", "r0_x1.3"):
            if (v, ind) in rows:
                r = rows[(v, ind)]
                sens_rows.append([v, ind, r["r0_mm"], r["r_min_mm"], r["h_lost_mm"], round(r["h_lost_mm"] - b["h_lost_mm"], 1), r["nodes"], r["nodes"] - b["nodes"],
                                  r["supported_segments"], r["outside_hull_share"]])
    for ind, why in NOT_RUN.items():
        json.dump(dict(individual=ind, status="not_run", reason=why, tag=None, tags=TAGS), open(D / "chain" / f"{ind}.json", "w", encoding="utf-8"), indent=1)
    with open(D / "run2_vs_gapfill.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["individual", "run2_key", "run2_h_lost_mm", "gapfill_h_lost_mm", "run2_apex_mm", "gapfill_apex_mm", "run2_outside_hull", "gapfill_outside_hull",
                    "run2_base_radius_mm", "gapfill_first_seg_radius_mm", "run2_nodes", "gapfill_nodes", "internodes_over_photo_limit", "tag"]); w.writerows(cmp_rows)
    with open(D / "sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["variant", "individual", "r0_mm", "r_min_mm", "h_lost_mm", "d_h_lost_mm", "nodes", "d_nodes", "supported_segments", "outside_hull_share"]); w.writerows(sens_rows)
    print("written", len(CHAINED), "chains,", len(NOT_RUN), "not_run")


if __name__ == "__main__":
    main()
