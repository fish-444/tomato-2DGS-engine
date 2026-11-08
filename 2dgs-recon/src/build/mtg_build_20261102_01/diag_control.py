"""CONTRACT A4: dtheta = 0 control. offset0 = measured chord - chord from printed values (straight) for the same 49 leaves as A3; compare with the A3 gap.
python diag_control.py <control geometry>"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from roundtrip_bmodel import obj_vertices, leaf_rows, spec_of  # noqa: E402

g = Path(sys.argv[1]); A3 = {(r["plant"], int(r["rank"])): r for r in csv.DictReader(open(HERE / "diag_print.csv", encoding="utf-8"))}
arr = lambda s: [float(x) for x in s.strip("[]").split(";") if x.strip() not in ("", "null")] if s and s != "null" else []
out = []
for name in ("mtg_template_mean_v2draft_sd0", "mtg_R3-1b", "mtg_n_04", "P01", "P02"):
    spec = spec_of(name.replace("_sd0", "")); mask = name.replace("_sd0", ""); stem = {tuple(np.round(v, 7)) for v in obj_vertices(g / f"{mask}_stem.csv.obj")}
    for lf in sorted(spec["leaves"], key=lambda l: l["k"]):
        a3 = A3.get((name, lf["k"]))
        f = g / f"{mask}_leaf{lf['k']:02d}nb.csv.obj"; t = g / f"{mask}_leaf{lf['k']:02d}nb.csv_leaves.csv"
        if a3 is None or not f.exists() or not t.exists():
            continue
        r = next(x for x in leaf_rows(t) if int(float(x["order"])) == 0 and int(float(x["rank"])) == lf["k"])
        nb = int(float(r["p_bladesNr"])); curv = float(r["p_leafCurvature"]); ang = float(r["p_anglePetiole"]); pet = float(r["p_lengthPetiole"])
        segs = arr(r["p_segmentsLength"]); pts = arr(r["p_lengthPetiolules"]); inc = arr(r["p_inclPetiolules"])
        e = 90 - ang; x = pet * math.cos(math.radians(e)); z = pet * math.sin(math.radians(e))
        for q in range(nb - 1):
            turn, L = (90 - inc[q], pts[q]) if q == nb - 2 else (curv - 90, segs[q])
            e -= turn; x += L * math.cos(math.radians(e)); z += L * math.sin(math.radians(e))
        chord_print = math.degrees(math.atan2(z, x))
        V = obj_vertices(f); Lv = np.asarray([v for v in V if tuple(np.round(v, 7)) not in stem]); b = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
        d = np.linalg.norm(Lv - b, axis=1); v = Lv[int(np.argmax(d))] - b; chord_meas = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
        gap = -float(a3["d_print"])                                           # A3: measured - printed at the leaf's dtheta
        out.append(dict(plant=name, rank=lf["k"], dtheta_A3=float(a3["dtheta"]), leafCurvature=curv, chord_measured0=chord_meas, chord_printed0=chord_print, offset0=chord_meas - chord_print,
                        gap_A3=gap, d_offset=chord_meas - chord_print - gap))
n = len(out); valid = sum(abs(x["leafCurvature"] - 90) <= 0.01 for x in out)
pre = sum(abs(x["d_offset"]) <= 1 for x in out); only_bent = sum(abs(x["offset0"]) <= 0.5 for x in out)
if valid < n:
    verdict = "control_void"
elif pre >= 0.9 * n:
    verdict = "pre_existing_offset(adapter_acts_as_declared)"
elif only_bent >= 0.9 * n:
    verdict = "offset_only_when_bent(adapter_needs_work)"
else:
    verdict = "not_separated"
S = dict(leaves=n, curvature_90=valid, within_1deg_of_A3_gap=pre, offset0_within_0_5=only_bent, verdict=verdict, median_offset0=float(np.median([x["offset0"] for x in out])),
         range_offset0=[float(min(x["offset0"] for x in out)), float(max(x["offset0"] for x in out))], median_gap_A3=float(np.median([x["gap_A3"] for x in out])),
         median_abs_d_offset=float(np.median([abs(x["d_offset"]) for x in out])), corr_offset0_gapA3=float(np.corrcoef([x["offset0"] for x in out], [x["gap_A3"] for x in out])[0, 1]))
(HERE / "diag_control.json").write_text(json.dumps(dict(summary=S, leaves=out), indent=1, default=float), encoding="utf-8")
with open(HERE / "diag_control.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(json.dumps(S, indent=1))
