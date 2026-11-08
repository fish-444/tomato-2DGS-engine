"""CONTRACT D: fresh synthetic check of mode D (stem + phi only; theta0 / dtheta / L fixed), same 13 plants as synth_B. python synth_D.py"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import synth_B as SB  # noqa: E402
import fit_plant as FP  # noqa: E402
from fit_lib import resample, point_seg_dist  # noqa: E402
import mtg_spec as MS  # noqa: E402


def main():
    res, stems = [], []; m = 0; main_n = 0; pooled = MS.pooled_areas()
    while main_n < 100:
        D, truth, true_axis, pid, seed, nl = SB.plant(m)
        fitted, stem, rows = FP.fit_plant_B(D, pooled, "D")
        stems.append(dict(plant=pid, seed=seed, stem_offset_median_mm=float(np.median(point_seg_dist(resample(np.asarray(stem["curve"], float), 1.0)[0], true_axis))), flags=";".join(stem["flags"])))
        for r in rows:
            t = truth[r["uid"]]; r.update(true_azimuth=t["azimuth_deg"], true_theta0=t["theta0_deg"], true_dtheta=t["dtheta_deg"], true_length=t["length_mm"])
            if "azimuth_fit" in r:
                r.update(err_azimuth=abs((r["azimuth_fit"] - t["azimuth_deg"] + 180) % 360 - 180), err_azimuth_start=abs((r["azimuth_prev"] - t["azimuth_deg"] + 180) % 360 - 180),
                         err_length_rel_kept=abs(r["length_fit"] / t["length_mm"] - 1), err_theta0_kept=abs(r["theta0_fit"] - t["theta0_deg"]))
            r["flags"] = ";".join(r["flags"]); res.append(r)
        main_n += nl; m += 1; print(pid, nl, flush=True)
    fit = [r for r in res if "err_azimuth" in r]
    S = dict(plants=m, main_leaves=main_n, leaves_optimised=len(fit), gate1_pass=sum(bool(r["gate1"]) for r in fit), median_err_azimuth=float(np.median([r["err_azimuth"] for r in fit])),
             start_median_err_azimuth=float(np.median([r["err_azimuth_start"] for r in fit])), median_stem_offset_mm=float(np.median([x["stem_offset_median_mm"] for x in stems])),
             median_err_length_rel_kept_not_gated=float(np.median([r["err_length_rel_kept"] for r in fit])), median_err_theta0_kept_not_gated=float(np.median([r["err_theta0_kept"] for r in fit])))
    S["acceptable"] = dict(azimuth=S["median_err_azimuth"] <= 10, stem=S["median_stem_offset_mm"] <= 3); S["all_acceptable"] = all(S["acceptable"].values())
    keys = []
    for r in res:
        keys += [k for k in r if k not in keys]
    with open(HERE / "synthetic/synthetic_check_D_leaves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
    json.dump(S, open(HERE / "synthetic/synthetic_check_D_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
