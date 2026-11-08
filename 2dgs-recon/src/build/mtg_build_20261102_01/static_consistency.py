"""Static light consistency (CONTRACT §5; not validation): engine-format MTG geometry traced by our tracer (Kaggle results.json) vs the same plant's
frozen surfel-disc result (plant_light.json d_ex 2.4 / R3-1b tau out_k1b v1), plant level ratio + per-leaf Spearman rho for A/B plants (per-leaf disc
light = per-surfel absorbed x operating labels). Runs without the Kaggle file (disc references only); merges when results.json is present.
Stated next to every ratio: disc-vs-triangle representation difference 1.24-1.38x (LOG:1652/1657)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import mtg_spec as MS  # noqa: E402

DISC_NOTE = "disc (frozen v1 surfels) vs triangle representation differs by 1.24-1.38x in absorbed light on R3-1b (LOG:1652/1657); consistency check, not validation"


def disc_reference(pid):
    """Plant-level disc result + per-leaf absorbed / area by leaf uid (operating labels)."""
    if pid == "R3-1b":
        pj = json.load(open(ROOT / "build/engine_static_tau_20261031_01/out_k1b/R3-1b/plant_tau.json", encoding="utf-8"))["plant"]["v1"]
        plant = dict(absorbed=pj["total"]["mean"], self_shading=pj["self"]["mean"], mean_light=pj["mean_light"]["mean"], source="engine_static_tau_20261031_01/out_k1b plant_tau.json v1 (tau 0, byte-identical to frozen)")
        z = np.load(ROOT / "build/engine_coplanar_sep_20261101_01/out_k1/per_surfel_D.npz"); ab = z["Uext"] * z["K_v1"] * z["area_b_mm2"]
        per = {int(k): dict(absorbed=float(ab[z["leaf_R8pc"] == k].sum()), area=float(z["area_b_mm2"][z["leaf_R8pc"] == k].sum())) for k in np.unique(z["leaf_R8pc"]) if k > 0}
        return plant, per, "per R8_pc leaf (designated_not_adopted; display_only)"
    pj = json.load(open(ROOT / f"build/engine_static_nine_20261028_01/kaggle_out/{pid}/plant_light.json", encoding="utf-8")); L = pj["plant_level"]["2.4"]
    plant = dict(absorbed=L["absorbed"]["mean"], self_shading=L["self_shading"]["mean"], mean_light=L["relative_light_per_area"]["mean"], source="engine_static_nine_20261028_01 plant_light.json d_ex 2.4 (frozen v1)")
    z = np.load(ROOT / f"build/engine_static_nine_20261028_01/kaggle_out/{pid}/per_surfel_light.npz"); lab = np.load(MS.DZ / f"labels_{MS.ROUTE[pid]}_{pid}_{MS.SNAP[pid]}.npz")
    assert np.array_equal(z["ply_row"], lab["rows"])
    per = {}
    for k, u in enumerate([str(x) for x in lab["leaf_uids"]]):
        m = (lab["label"] == k + 1) & (lab["status"] == 1); per[u] = dict(absorbed=float(z["absorbed_ru_mm2"][m].sum()), area=float(z["area_b_silhouette_mm2"][m].sum()))
    return plant, per, f"per skeleton leaf via {MS.ROUTE[pid]} labels (assigned surfels)"


def main(results=HERE / "kaggle/results_light/results.json"):
    R = json.load(open(results, encoding="utf-8")) if Path(results).exists() else None
    rows = []
    for pid in MS.PLANTS:
        if not pid.startswith("n_") and pid != "R3-1b":
            rows.append(dict(plant=pid, grade=MS.GRADE[pid], status="not_traced: no area (e_*)")); continue
        plant, per, note = disc_reference(pid); spec = json.load(open(HERE / f"static/mtg_{pid}_spec.json", encoding="utf-8"))
        row = dict(plant=pid, grade=MS.GRADE[pid], disc_absorbed=plant["absorbed"], disc_self_shading=plant["self_shading"], disc_mean_light=plant["mean_light"], disc_source=plant["source"], per_leaf_note=note, representation_note=DISC_NOTE)
        if R and f"mtg_{pid}" in R["plants"]:
            s = R["plants"][f"mtg_{pid}"]["summary"]
            row.update(mtg_absorbed=s["total"]["mean"], mtg_self_shading=s["self"]["mean"], mtg_mean_light=s["mean_light"]["mean"], mtg_area=s["area"]["mean"], ratio_absorbed_mtg_over_disc=s["total"]["mean"] / plant["absorbed"],
                       ratio_mean_light=s["mean_light"]["mean"] / plant["mean_light"], d_self_shading=s["self"]["mean"] - plant["self_shading"])
            if pid == "R3-1b":
                pl = {lf["uid"]: lf for lf in spec["leaves"]}; sk = json.load(open(ROOT / spec["skeleton_file"], encoding="utf-8")); r8 = {l["leaf_uid"]: int(l["attrs"]["pipeline_leaf"]) for l in sk["leaves"]}
                a_mtg, a_disc = {}, {}
                for lf in spec["leaves"]:
                    key = f"leaf{lf['k']:03d}_total"
                    if key in s:
                        a_mtg[r8[lf["uid"]]] = a_mtg.get(r8[lf["uid"]], 0) + s[key]["mean"]
                ks = [k for k in sorted(a_mtg) if k in per]; x = [a_mtg[k] for k in ks]; y = [per[k]["absorbed"] for k in ks]
                row.update(per_leaf_n=len(ks), spearman_rho_absorbed=float(spearmanr(x, y).statistic) if len(ks) > 2 else None, per_leaf_status="display_only (R3-1b per R8_pc leaf, not graded)")
            else:
                ks = [lf for lf in spec["leaves"] if f"leaf{lf['k']:03d}_total" in s and lf["uid"] in per and per[lf["uid"]]["area"] > 0]
                x = [s[f"leaf{lf['k']:03d}_total"]["mean"] for lf in ks]; y = [per[lf["uid"]]["absorbed"] for lf in ks]
                xl = [s[f"leaf{lf['k']:03d}_light"]["mean"] for lf in ks]; yl = [per[lf["uid"]]["absorbed"] / per[lf["uid"]]["area"] for lf in ks]
                row.update(per_leaf_n=len(ks), spearman_rho_absorbed=float(spearmanr(x, y).statistic) if len(ks) > 2 else None, spearman_rho_light_per_area=float(spearmanr(xl, yl).statistic) if len(ks) > 2 else None,
                           per_leaf_status="per_leaf" if MS.GRADE[pid] in ("A", "B") else "display_only (grade C / ungraded)")
                with open(HERE / f"static/per_leaf_light_{pid}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(["leaf_uid", "model_rank", "mtg_absorbed", "mtg_area", "mtg_light_per_area", "disc_absorbed", "disc_area", "disc_light_per_area", "status"])
                    for lf, a, b_, c, d in zip(ks, x, y, xl, yl):
                        w.writerow([lf["uid"], lf["k"], a, s[f"leaf{lf['k']:03d}_area"]["mean"], c, b_, per[lf["uid"]]["area"], d, row["per_leaf_status"]])
        else:
            row.update(status="disc reference only (Kaggle tracer result not yet merged)")
        rows.append(row)
    if R:
        for name, v in R["plants"].items():
            if name.startswith("mtg_template") or name.startswith("P"):
                s = v["summary"]; rows.append(dict(plant=name, grade="n/a", mtg_absorbed=s["total"]["mean"], mtg_self_shading=s["self"]["mean"], mtg_mean_light=s["mean_light"]["mean"], mtg_area=s["area"]["mean"], status="no disc reference (generated plant)"))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(HERE / "static_consistency_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # paired internode light table (check plants beta vs fixed375)
    if R:
        pr = []
        for m in range(1, 17):
            a = R["plants"].get(f"P{m:02d}"); b = R["plants"].get(f"P{m:02d}_fixed375")
            if a and b:
                pr.append(dict(plant=f"P{m:02d}", absorbed_beta=a["summary"]["total"]["mean"], absorbed_fixed375=b["summary"]["total"]["mean"], d_absorbed_rel=a["summary"]["total"]["mean"] / b["summary"]["total"]["mean"] - 1,
                               self_beta=a["summary"]["self"]["mean"], self_fixed375=b["summary"]["self"]["mean"], d_self=a["summary"]["self"]["mean"] - b["summary"]["self"]["mean"], label="display_only"))
        if pr:
            with open(HERE / "dynamic/internode_paired_light.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(pr[0])); w.writeheader(); w.writerows(pr)
    print(f"{len(rows)} rows; kaggle merged: {bool(R)}")
    return rows


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
