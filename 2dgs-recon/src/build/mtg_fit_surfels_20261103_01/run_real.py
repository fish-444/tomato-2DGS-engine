"""CONTRACT §2-3, §5: fit every observed plant; tables; fitted specs; MTG files only if the synthetic check is acceptable (§4), else display_only.
python run_real.py"""
import copy, csv, json, math, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_plant as FP  # noqa: E402
import mtg_spec as MS  # noqa: E402
from tomato3d.plant_tree import engine as PE, geometry as g  # noqa: E402

PLANTS = ["R3-1b", "n_01", "n_02", "n_03", "n_04", "n_05", "n_06", "e_01", "e_04", "e_05", "e_07"]
SYN = json.load(open(HERE / "synthetic/synthetic_check_summary.json"))
WRITE = bool(SYN["all_acceptable"]); OUT = HERE / "fitted"; KS = HERE / "kaggle/import_ds/scn"; KS.mkdir(parents=True, exist_ok=True)


def engine_round_trip(tree, stage):
    """as build_all.engine_round_trip of the old build (copied, not imported): tree -> to_engine_structure -> from_engine_states(day 0)."""
    back = PE.from_engine_states(stage / "initial_state.csv", 0, "p1"); A = {lf["id"]: lf for lf in tree["leaves"]}
    rep = json.load(open(stage / "mapping_report.json", encoding="utf-8")); inv = {v: k for k, v in rep["id_map"].items()}; ok = len(back["leaves"]) == len(tree["leaves"]); worst = dict(theta0=0.0, dtheta=0.0, L=0.0, az=0.0, seg_dir=0.0)
    for lb in back["leaves"]:
        la = A.get(inv.get(lb.get("engine_id")))
        if la is None:
            ok = False; continue
        pa = np.asarray(la["rachis"]["polyline"], float); pb = np.asarray(lb["rachis"]["polyline"], float)
        def dirs(p):
            v = np.diff(p, axis=0); v = v / np.maximum(np.linalg.norm(v, axis=1)[:, None], 1e-12); return v
        da, db = dirs(pa), dirs(pb)
        if len(da) != len(db):
            ok = False; continue
        ang = np.degrees(np.arccos(np.clip((da * db).sum(1), -1, 1))); worst["seg_dir"] = max(worst["seg_dir"], float(ang.max()))
        La, Lb = g.polyline_length(pa), g.polyline_length(pb); worst["L"] = max(worst["L"], abs(Lb / La - 1))
        worst["dtheta"] = max(worst["dtheta"], abs(g.polyline_turning_deg(pa) - g.polyline_turning_deg(pb)))
    passed = ok and worst["seg_dir"] <= 2 and worst["L"] <= 0.01 and worst["dtheta"] <= 2
    return dict(passed=bool(passed), leaves_in=len(tree["leaves"]), leaves_back=len(back["leaves"]), **{f"worst_{k}": v for k, v in worst.items()})


def main():
    pooled = MS.pooled_areas(); leaf_rows, stem_rows, plant_rows = [], [], []
    for pid in PLANTS:
        D = FP.load_real(pid); fitted, stem, rows = FP.fit_plant(D, pooled)
        if not WRITE:
            fitted["labels"].append("display_only_synthetic_check_not_met")
        fitted["edited_from"] = f"mtg_{pid}_spec.json"
        json.dump(fitted, open(OUT / f"mtg_{pid}_spec_fitted.json", "w", encoding="utf-8"), indent=1, default=float)
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stem.items()}, open(OUT / f"mtg_{pid}_stem_fit.json", "w"), indent=1, default=float)
        s = {k: v for k, v in stem.items() if k not in ("curve", "band_counts")}; s.update(plant=pid, band_counts=";".join(map(str, stem.get("band_counts", []))), flags=";".join(stem["flags"])); stem_rows.append(s)
        for r in rows:
            r["flags"] = ";".join(r["flags"]); leaf_rows.append(r)
        pr = dict(plant=pid, leaves=len(rows), fitted=sum(r["status"] == "fitted" for r in rows), failed=sum(r["status"] == "fit_failed_kept_original" for r in rows),
                  not_fitted=sum(r["status"].startswith("not_fitted") for r in rows), level2_used=sum(1 for lf in fitted["leaves"] if lf.get("rachis_polyline_level2_mm")),
                  median_theta0_change=float(np.median([r["theta0_fit"] - r["theta0_prev"] for r in rows if r["status"] == "fitted"])) if any(r["status"] == "fitted" for r in rows) else None,
                  stem_shift_median_mm=stem.get("shift_median"), stem_shift_max_mm=stem.get("shift_max"), stem_flags=";".join(stem["flags"]), written=WRITE)
        if WRITE:
            tree, rep, sha, info = FP.write_fitted(fitted, OUT, f"mtg_{pid}_fitted")
            rt = engine_round_trip(tree, OUT / f"mtg_{pid}_fitted_engine"); pr.update(engine_roundtrip=rt["passed"], **{f"rt_{k}": v for k, v in rt.items() if k != "passed"}, bmodel=info["status"])
            if sha:
                shutil.copy2(OUT / f"mtg_{pid}_fitted_bmodel.csv", KS / f"mtg_{pid}_fitted.csv"); json.dump(fitted, open(OUT / f"mtg_{pid}_fitted_spec.json", "w", encoding="utf-8"), indent=1, default=float)
        plant_rows.append(pr); print(pr, flush=True)
    for name, data in (("leaf_fit.csv", leaf_rows), ("stem_fit.csv", stem_rows), ("plant_fit_summary.csv", plant_rows)):
        keys = []
        for r in data:
            keys += [k for k in r if k not in keys]
        with open(HERE / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)


if __name__ == "__main__":
    main()
