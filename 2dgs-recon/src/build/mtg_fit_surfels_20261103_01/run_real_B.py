"""CONTRACT B: real plants in mode B for both arms. An arm whose synthetic check passed -> specs + engine (Level 2) + B-model (Level 1) files in fitted_<arm>/,
engine round trip; else display_only specs only. Figures fit_<arm>_<id>.png. python run_real_B.py"""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_plant as FP  # noqa: E402
import mtg_spec as MS  # noqa: E402
import run_real as RR  # noqa: E402
import figures as FG  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SYN = json.load(open(HERE / "synthetic/synthetic_check_B_summary.json"))


def figure(pid, old, new, arm, X):
    t0, t1 = FG.tree_of(old), FG.tree_of(new); fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, proj, lab in ((axs[0], (0, 2), "x-z (side)"), (axs[1], (0, 1), "x-y (top)")):
        ax.scatter(X[:, proj[0]], X[:, proj[1]], s=0.3, c="lightgray", rasterized=True); FG.draw(ax, t0, proj, "--", 0.6); FG.draw(ax, t1, proj, "-", 1.0); ax.set_aspect("equal"); ax.set_xlabel("mm")
        ax.set_title(f"{pid} {arm}: before (dashed) / fitted (solid) - {lab}", fontsize=9)
    n_fit = sum(1 for lf in new["leaves"] if lf.get("fit_status") == "fitted"); st = new["fit"]["stem"]
    fig.suptitle(f"{pid} mode B {arm} (theta0/dtheta = hand template; phi, L, stem fitted) - leaves fitted {n_fit}, stem shift median {st.get('shift_median', 0):.1f} mm"
                 + ("" if SYN[arm]["all_acceptable"] else " - DISPLAY ONLY"), fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / f"figures/fit_{arm}_{pid}.png", dpi=130); plt.close(fig)


def main():
    pooled = MS.pooled_areas(); data = {pid: FP.load_real(pid) for pid in RR.PLANTS}
    for arm in ("B1", "B2"):
        ok = SYN[arm]["all_acceptable"]; out = HERE / f"fitted_{arm}"; out.mkdir(exist_ok=True); ks = HERE / f"kaggle/import_{arm}/scn"
        leaf_rows, stem_rows, plant_rows = [], [], []
        for pid in RR.PLANTS:
            D = data[pid]; fitted, stem, rows = FP.fit_plant_B(D, pooled, arm)
            if not ok:
                fitted["labels"].append("display_only_synthetic_check_not_met")
            fitted["edited_from"] = f"mtg_{pid}_spec.json"
            json.dump(fitted, open(out / f"mtg_{pid}_spec_fitted.json", "w", encoding="utf-8"), indent=1, default=float)
            s = {k: v for k, v in stem.items() if k not in ("curve", "band_counts")}; s.update(plant=pid, arm=arm, flags=";".join(stem["flags"])); stem_rows.append(s)
            for r in rows:
                r["flags"] = ";".join(r["flags"]); leaf_rows.append(r)
            pr = dict(plant=pid, arm=arm, leaves=len(rows), fitted=sum(r["status"] == "fitted" for r in rows), failed=sum(r["status"] == "fit_failed_kept_original" for r in rows),
                      not_fitted=sum(r["status"].startswith("not_fitted") for r in rows), level2_used=sum(1 for lf in fitted["leaves"] if lf.get("rachis_polyline_level2_mm")),
                      median_L_change_rel=float(np.median([r["length_fit"] / r["length_prev"] - 1 for r in rows if r["status"] == "fitted"])) if any(r["status"] == "fitted" for r in rows) else None,
                      median_phi_change=float(np.median([abs((r["azimuth_fit"] - r["azimuth_prev"] + 180) % 360 - 180) for r in rows if r["status"] == "fitted"])) if any(r["status"] == "fitted" for r in rows) else None,
                      stem_shift_median_mm=stem.get("shift_median"), stem_shift_max_mm=stem.get("shift_max"), stem_flags=";".join(stem["flags"]), written=ok)
            if ok:
                tree, rep, sha, info = FP.write_fitted(fitted, out, f"mtg_{pid}_fitted"); rt = RR.engine_round_trip(tree, out / f"mtg_{pid}_fitted_engine")
                pr.update(engine_roundtrip=rt["passed"], rt_worst_seg_dir=rt["worst_seg_dir"], rt_worst_L=rt["worst_L"], bmodel=info["status"])
                if sha:
                    ks.mkdir(parents=True, exist_ok=True); shutil.copy2(out / f"mtg_{pid}_fitted_bmodel.csv", ks / f"mtg_{pid}_fitted.csv")
                    json.dump(fitted, open(out / f"mtg_{pid}_fitted_spec.json", "w", encoding="utf-8"), indent=1, default=float)
            plant_rows.append(pr); print(pr, flush=True)
            old = json.load(open(FP.OLD / f"static/mtg_{pid}_spec.json", encoding="utf-8")); X = D["X"][np.random.default_rng(0).choice(len(D["X"]), min(40000, len(D["X"])), replace=False)]
            figure(pid, old, fitted, arm, X)
        for name, rowsx in ((f"leaf_fit_{arm}.csv", leaf_rows), (f"stem_fit_{arm}.csv", stem_rows), (f"plant_fit_summary_{arm}.csv", plant_rows)):
            keys = []
            for r in rowsx:
                keys += [k for k in r if k not in keys]
            with open(HERE / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rowsx)


if __name__ == "__main__":
    main()
