"""CONTRACT U: B-model round trip of the adopted fitted MTGs (Level 1). Per leaf: orientation = (azimuth + 90) mod 360, inclination = 90 - theta0, area
(column copy) vs the fitted spec; counts and order exact; angles <= 2 deg; area <= 1 %. python roundtrip_bmodel_user.py <geometry dir>"""
import csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; g = Path(sys.argv[1]); out = []
for f in sorted(g.glob("mtg_*_fitted.csv_leaves.csv")):
    pid = f.name[len("mtg_"):-len("_fitted.csv_leaves.csv")]; spec = json.load(open(HERE / f"fitted_user/mtg_{pid}_spec_fitted.json", encoding="utf-8"))
    rows = sorted([r for r in csv.DictReader(open(f, encoding="utf-8")) if int(float(r["order"])) == 0], key=lambda r: int(float(r["rank"]))); exp = sorted(spec["leaves"], key=lambda l: l["k"])
    worst_o = worst_i = worst_a = 0.0; ok = len(rows) == len(exp) and [int(float(r["rank"])) for r in rows] == [l["k"] for l in exp]
    for lf, r in zip(exp, rows):
        worst_o = max(worst_o, abs((float(r["orientation_deg"]) - (lf["azimuth_deg"] + 90) % 360 + 180) % 360 - 180)); worst_i = max(worst_i, abs(float(r["inclination_deg"]) - (90 - lf["theta0_deg"])))
        if lf.get("area_mm2"):
            worst_a = max(worst_a, abs(float(r["blade_area_total_m2"]) * 1e6 / lf["area_mm2"] - 1))
    out.append(dict(plant=pid, leaves_in=len(exp), leaves_model=len(rows), counts_order_pass=ok, worst_d_orientation_deg=worst_o, worst_d_inclination_deg=worst_i, worst_d_area_rel=worst_a,
                    passed=bool(ok and worst_o <= 2 and worst_i <= 2 and worst_a <= 0.01)))
    print(out[-1])
with open(HERE / "roundtrip_bmodel_user.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
