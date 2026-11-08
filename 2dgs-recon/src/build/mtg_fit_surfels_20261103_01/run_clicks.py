"""CONTRACT G: synthetic information report, then the real plants (files, engine round trip, B-model scenarios + kernel, figures).
python run_clicks.py synth | real"""
import csv, hashlib, json, math, shutil, sys, tarfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import clicks_lib as CL  # noqa: E402
import fit_plant as FP  # noqa: E402
import fit_lib as FL  # noqa: E402
import run_real as RR  # noqa: E402
import mtg_spec as MS  # noqa: E402


def synth():
    import synth_B as SB
    rows = []; m = 0; n = 0
    while n < 100:
        D, truth, true_axis, pid, seed, nl = SB.plant(m)
        fitted, stem, _ = FP.fit_plant_B(D, MS.pooled_areas(), "D")          # stem + nodes from the same stem rule (leaf part not used)
        spec_g, r = CL.build_plant(fitted, D["clicks"], D["X"])
        for x in r:
            t = truth[x["uid"]]
            if x["status"] == "rachis_from_clicks":
                tp = np.asarray(t.get("_true_pl") or [], float)
            x.update(true_azimuth=t["azimuth_deg"], true_theta0=t["theta0_deg"], true_length=t["length_mm"])
            if "azimuth_G" in x:
                x.update(err_azimuth=abs((x["azimuth_G"] - t["azimuth_deg"] + 180) % 360 - 180), err_theta0=abs(x["theta0_G"] - t["theta0_deg"]), err_length_rel=abs(x["length_G"] / t["length_mm"] - 1))
            x["flags"] = ";".join(x["flags"]); rows.append(x)
        # shape error: mean distance of the true rachis (writer polyline of the truth) to the built curve
        import synth as SY
        sp, tree, _, _, _ = SY.make_plant(m); tl = {f"{pid}-L{lf['rank']:02d}": np.asarray(lf["rachis"]["polyline"], float) for lf in tree["leaves"]}
        for lf in spec_g["leaves"]:
            if lf.get("clicks_status") == "rachis_from_clicks":
                T = FL.resample(tl[lf["uid"]], 1.0)[0]; d = FL.point_seg_dist(T, np.asarray(lf["rachis_curve_from_clicks_mm"], float))
                next(x for x in rows if x["uid"] == lf["uid"])["shape_err_mm"] = float(d.mean())
        n += nl; m += 1; print(pid, nl, flush=True)
    keys = []
    for x in rows:
        keys += [k for k in x if k not in keys]
    with open(HERE / "synthetic/synthetic_clicks_G_leaves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    g = [x for x in rows if "err_azimuth" in x]
    S = dict(plants=m, leaves=len(rows), built_from_clicks=len(g), node_link_fail=sum(x["status"] == "node_link_outside_cloud_kept_U" for x in rows),
             median_err_azimuth=float(np.median([x["err_azimuth"] for x in g])), median_err_theta0=float(np.median([x["err_theta0"] for x in g])), median_err_length_rel=float(np.median([x["err_length_rel"] for x in g])),
             median_shape_err_mm=float(np.median([x["shape_err_mm"] for x in g if "shape_err_mm" in x])), note="information only (CONTRACT G); L stops at the last click (lower bound)")
    json.dump(S, open(HERE / "synthetic/synthetic_clicks_G_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))


def real():
    OUT = HERE / "fitted_clicks"; OUT.mkdir(exist_ok=True); K = HERE / "kaggle/import_clicks"; (K / "scn").mkdir(parents=True, exist_ok=True); rows, rt_rows = [], []
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    import figures as FG
    import matplotlib.pyplot as plt
    for pid in RR.PLANTS:
        D = FP.load_real(pid); spec_u = json.load(open(HERE / f"fitted_user/mtg_{pid}_spec_fitted.json", encoding="utf-8"))
        spec_g, r = CL.build_plant(spec_u, D["clicks"], D["X"]); spec_g["edited_from"] = f"mtg_{pid}_spec.json"
        json.dump(spec_g, open(OUT / f"mtg_{pid}_spec_fitted.json", "w", encoding="utf-8"), indent=1, default=float)
        tree, rep, bsha, info = FP.write_fitted(spec_g, OUT, f"mtg_{pid}_clicks"); rt = RR.engine_round_trip(tree, OUT / f"mtg_{pid}_clicks_engine")
        if bsha:
            shutil.copy2(OUT / f"mtg_{pid}_clicks_bmodel.csv", K / "scn" / f"mtg_{pid}_clicks.csv")
        for x in r:
            x["flags"] = ";".join(x["flags"]); rows.append(x)
        rt_rows.append(dict(plant=pid, leaves=len(r), from_clicks=sum(x["status"] == "rachis_from_clicks" for x in r), kept_U=sum(x["status"] == "node_link_outside_cloud_kept_U" for x in r),
                            unchanged=sum(x["status"] == "unchanged_one_click" for x in r), engine_roundtrip=rt["passed"], rt_worst_seg_dir_deg=rt["worst_seg_dir"], bmodel=info["status"],
                            median_curve_dist_mm=float(np.median([x["curve_median_dist_mm"] for x in r if "curve_median_dist_mm" in x])) if any("curve_median_dist_mm" in x for x in r) else None))
        print(rt_rows[-1], flush=True)
        X = D["X"][np.random.default_rng(0).choice(len(D["X"]), min(40000, len(D["X"])), replace=False)]; t0, t1 = FG.tree_of(spec_u), FG.tree_of(spec_g)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax, proj, lab in ((axs[0], (0, 2), "x-z (side)"), (axs[1], (0, 1), "x-y (top)")):
            ax.scatter(X[:, proj[0]], X[:, proj[1]], s=0.3, c="lightgray", rasterized=True); FG.draw(ax, t0, proj, "--", 0.5); FG.draw(ax, t1, proj, "-", 1.0)
            C = [np.asarray(c, float) for c in D["clicks"].values() if len(c) >= 2]
            for c in C:
                ax.plot(c[:, proj[0]], c[:, proj[1]], ".", color="crimson", ms=3)
            ax.set_aspect("equal"); ax.set_xlabel("mm"); ax.set_title(f"{pid}: adopted fit U (dashed) / rachis from clicks (solid), clicks red - {lab}", fontsize=8)
        fig.suptitle(f"{pid} rachis_from_clicks (CONTRACT G) - built {rt_rows[-1]['from_clicks']}, kept U {rt_rows[-1]['kept_U']}, unchanged {rt_rows[-1]['unchanged']}", fontsize=9)
        fig.tight_layout(); fig.savefig(HERE / f"figures/clicks_{pid}.png", dpi=130); plt.close(fig)
    for name, data in (("leaf_clicks_G.csv", rows), ("plant_clicks_G.csv", rt_rows)):
        keys = []
        for x in data:
            keys += [k for k in x if k not in keys]
        with open(HERE / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)
    shutil.copy2(FP.OLD / "kaggle/import_ds/work_model.tar.bin", K / "work_model.tar.bin")
    with tarfile.open(K / "scenarios.tar.bin", "w") as t:
        for p in sorted((K / "scn").glob("*.csv")):
            t.add(p, arcname=p.name)
    (K / "MTGB_IMPORT_MANIFEST.json").write_text(json.dumps(dict(inputs={n: sha(K / n) for n in ("work_model.tar.bin", "scenarios.tar.bin")}, core_sha256="35c605d65d9b54fe64cdf93bf28ae44ae7a1b6ded03b461dedcb0156a5c22147", note="CONTRACT G")), encoding="utf-8")
    (K / "dataset-metadata.json").write_text(json.dumps(dict(title="mtgb-clicks-20261103", id="dolgfish/mtgb-clicks-20261103", licenses=[dict(name="other")], isPrivate=True)), encoding="utf-8")
    kd = HERE / "kaggle/kernels/mtgb-bm-clicks-20261103"; kd.mkdir(parents=True, exist_ok=True); shutil.copy2(FP.OLD / "kaggle/kernel_import.py", kd / "kernel.py")
    (kd / "kernel-metadata.json").write_text(json.dumps(dict(id="dolgfish/mtgb-bm-clicks-20261103", title="mtgb-bm-clicks-20261103", code_file="kernel.py", language="python", kernel_type="script", is_private=True, enable_gpu=False,
                                                           enable_tpu=False, enable_internet=False, dataset_sources=["dolgfish/mtgb-clicks-20261103", "dolgfish/tomato-h-input"], competition_sources=[], kernel_sources=[])), encoding="utf-8")


if __name__ == "__main__":
    synth() if sys.argv[1] == "synth" else real()
