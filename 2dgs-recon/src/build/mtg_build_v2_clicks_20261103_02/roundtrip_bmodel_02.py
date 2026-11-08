"""B-model round trip of static set v2 _02 (11 plants): per leaf orientation = (azimuth + 90) mod 360, inclination = 90 - theta0, area vs the spec; counts/order
over leaves with area > 0; tolerance 2 deg / 1 %. python roundtrip_bmodel_02.py <geometry dir>"""
import csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; g = Path(sys.argv[1]); out = []
for f in sorted(g.glob("mtg_*.csv_leaves.csv")):
    pid = f.name[len("mtg_"):-len(".csv_leaves.csv")]; spec = json.load(open(HERE / f"static/mtg_{pid}_spec.json", encoding="utf-8"))
    exp = [l for l in sorted(spec["leaves"], key=lambda l: l["k"]) if l.get("area_mm2")]
    rows = sorted([r for r in csv.DictReader(open(f, encoding="utf-8")) if int(float(r["order"])) == 0], key=lambda r: int(float(r["rank"])))
    ok = len(rows) == len(exp) and [int(float(r["rank"])) for r in rows] == [l["k"] for l in exp]; wo = wi = wa = 0.0
    for l, r in zip(exp, rows):
        wo = max(wo, abs((float(r["orientation_deg"]) - (l["azimuth_deg"] + 90) % 360 + 180) % 360 - 180)); wi = max(wi, abs(float(r["inclination_deg"]) - (90 - l["theta0_deg"])))
        wa = max(wa, abs(float(r["blade_area_total_m2"]) * 1e6 / l["area_mm2"] - 1))
    out.append(dict(plant=pid, leaves_with_area=len(exp), leaves_model=len(rows), skipped_area0=len(spec["leaves"]) - len(exp), counts_order_pass=ok, worst_d_orientation_deg=wo, worst_d_inclination_deg=wi, worst_d_area_rel=wa,
                    passed=bool(ok and wo <= 2 and wi <= 2 and wa <= 0.01)))
    print(out[-1])
with open(HERE / "roundtrip_bmodel_02.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("passed", sum(r["passed"] for r in out), "/", len(out))
