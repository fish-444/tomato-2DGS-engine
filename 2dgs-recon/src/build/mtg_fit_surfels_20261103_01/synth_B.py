"""CONTRACT B: fresh synthetic check of mode B (theta0 / dtheta fixed), arms B1 and B2 on the same 13 plants (seeds 20261103...).
Generator = synth.py (make_plant, surfels, labels, clicks), unchanged. python synth_B.py"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import synth as SY  # noqa: E402
import fit_plant as FP  # noqa: E402
from fit_lib import BX, resample, point_seg_dist  # noqa: E402
import mtg_spec as MS  # noqa: E402
from jrandom import u_of  # noqa: E402


def plant(m):
    sp, tree, true_axis, seed, pid = SY.make_plant(m); X, tk = SY.surfels(tree, seed); lab = SY.labels(X, tk, seed); cl = SY.clicks(tree, seed, pid)
    zz = np.arange(0, true_axis[-1, 2] + 1e-9, 10.0); ax = [true_axis[0]]
    for i, z in enumerate(zz[1:], 1):
        a = 2 * math.pi * u_of(seed, pid, f"AXIS_DIR_{i}"); o = SY.POOL_OFF[min(len(SY.POOL_OFF) - 1, int(u_of(seed, pid, f"AXIS_OFF_{i}") * len(SY.POOL_OFF)))]
        ax.append(true_axis[min(int(round(z)), len(true_axis) - 1)] + o * np.array([math.cos(a), math.sin(a), 0.0]))
    ax = np.asarray(ax)
    sk = dict(schema="skeleton_v1.1", plant_id=pid, labels=["synthetic"], leaves=[dict(leaf_uid=u, rank_hint=int(u.split("L")[-1]), status="active", polyline_xyz=(p / 1000).tolist(), n_clicks=len(p), attrs={}) for u, p in cl.items()],
              axes=[dict(type="main", status="active", fit=dict(curve_mm=ax.tolist()), segments=[dict(s0_mm=0, s1_mm=1e6, provenance="observed")], flags=[])])
    skf = HERE / f"synthetic/{pid}_skeleton.json"; skf.write_text(json.dumps(sk), encoding="utf-8")
    truth = {f"{pid}-L{lf['k']:02d}": lf for lf in sp["leaves"]}
    obs = BX.load_observed(pid, skf, {u: truth[u]["area_mm2"] for u in truth}, "synthetic truth area", age_dag=60); dth = SY.TPL.dtheta()
    spec = dict(plant_id=pid, kind="observed", labels=["synthetic"], skeleton_file=str(skf.relative_to(FP.ROOT)).replace("\\", "/"), axis=dict(curve_mm=obs["axis_curve_mm"], base_mm=obs["stem_base_mm"], source=obs["axis_source"]),
                age_dag=60, age_source="synthetic", flags=obs["flags"], nodes=[], leaves=[], side_axes=[], trusses=[], notes=[])
    for nd, lf in zip(obs["nodes"], obs["leaves"]):
        k = lf["model_rank"]; th = SY.TPL.theta0(k if k <= 9 else "all")
        spec["nodes"].append(dict(k=k, s_mm=nd["s_mm"], xyz_mm=nd["xyz_mm"], internode_mm=nd["internode_mm"], internode_source="synthetic", stem_width_mm=4.0, width_source="synthetic"))
        spec["leaves"].append(dict(k=k, uid=lf["leaf_uid"], node_k=k, base_mm=lf["base_mm"], azimuth_deg=lf["azimuth_deg"], azimuth_source="synthetic click", skeleton_elevation_deg=lf["elevation_deg"], theta0_deg=th[0], theta0_source=th[2],
                                   dtheta_deg=dth[0], dtheta_source=dth[2], length_mm=lf["length_mm"], length_source="synthetic click path", leaflets=SY.TPL.leaflets_mode(k if k <= 9 else "all"), leaflets_source="hand mode",
                                   area_mm2=lf["area_b_mm2"], area_source="truth", flags=lf["flags"], geometry=lf["geometry"]))
    uids = [lf["uid"] for lf in spec["leaves"]]; kof = {u: int(u.split("L")[-1]) for u in uids}
    D = dict(spec=spec, X=X, leaf_idx={u: np.nonzero(lab == kof[u])[0] for u in uids}, leaf_mask=lab > 0, stem_mask=np.zeros(len(X), bool), clicks={u: cl[u] for u in uids}, label_source="synthetic")
    return D, truth, true_axis, pid, seed, len(sp["leaves"])


def main():
    res = {"B1": [], "B2": []}; stems = {"B1": [], "B2": []}; m = 0; main_n = 0; pooled = MS.pooled_areas()
    while main_n < 100:
        D, truth, true_axis, pid, seed, nl = plant(m)
        for arm in ("B1", "B2"):
            fitted, stem, rows = FP.fit_plant_B(D, pooled, arm)
            stems[arm].append(dict(plant=pid, seed=seed, stem_offset_median_mm=float(np.median(point_seg_dist(resample(np.asarray(stem["curve"], float), 1.0)[0], true_axis))), flags=";".join(stem["flags"])))
            for r in rows:
                t = truth[r["uid"]]; r.update(true_azimuth=t["azimuth_deg"], true_theta0=t["theta0_deg"], true_dtheta=t["dtheta_deg"], true_length=t["length_mm"])
                if "azimuth_fit" in r:
                    r.update(err_azimuth=abs((r["azimuth_fit"] - t["azimuth_deg"] + 180) % 360 - 180), err_length_rel=abs(r["length_fit"] / t["length_mm"] - 1), err_theta0_template=abs(r["theta0_fit"] - t["theta0_deg"]),
                             err_dtheta_template=abs(r["dtheta_fit"] - t["dtheta_deg"]), truth_inside_L_bound=0.7 * r["L_ref"] <= t["length_mm"] <= 1.3 * r["L_ref"])
                r["flags"] = ";".join(r["flags"]); res[arm].append(r)
        main_n += nl; m += 1; print(pid, nl, flush=True)
    S = {}
    for arm in ("B1", "B2"):
        fit = [r for r in res[arm] if "err_azimuth" in r]
        s = dict(plants=m, main_leaves=main_n, leaves_optimised=len(fit), gate1_pass=sum(bool(r["gate1"]) for r in fit), median_err_azimuth=float(np.median([r["err_azimuth"] for r in fit])),
                 median_err_length_rel=float(np.median([r["err_length_rel"] for r in fit])), median_stem_offset_mm=float(np.median([x["stem_offset_median_mm"] for x in stems[arm]])),
                 median_err_theta0_template_not_gated=float(np.median([r["err_theta0_template"] for r in fit])), median_err_dtheta_template_not_gated=float(np.median([r["err_dtheta_template"] for r in fit])),
                 share_truth_inside_L_bound=float(np.mean([r["truth_inside_L_bound"] for r in fit])))
        s["acceptable"] = dict(azimuth=s["median_err_azimuth"] <= 10, length=s["median_err_length_rel"] <= 0.10, stem=s["median_stem_offset_mm"] <= 3); s["all_acceptable"] = all(s["acceptable"].values()); S[arm] = s
        keys = []
        for r in res[arm]:
            keys += [k for k in r if k not in keys]
        with open(HERE / f"synthetic/synthetic_check_{arm}_leaves.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res[arm])
    json.dump(S, open(HERE / "synthetic/synthetic_check_B_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
