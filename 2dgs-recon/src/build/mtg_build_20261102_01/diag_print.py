"""CONTRACT A3: attribute the A1 FAIL from the model's own printed per-leaf values (print-only run mtgb-bm-print-20261103).
1 adapter value: leafCurvature == 90 + dtheta/(bladesNr-2) within 0.01 deg. 2 chord from printed lengths/turns vs measured chord (A1 rule) within 0.5 deg
on >= 90 %; then printed vs predictor lengths. Also: OBJ of this run byte-identical (LF) to run 2 for every scenario.
python diag_print.py <this run geometry> <run-2 geometry>"""
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from roundtrip_bmodel import obj_vertices, leaf_rows, spec_of  # noqa: E402
from roundtrip_dtheta import SEG, BX  # noqa: E402

g, g2 = Path(sys.argv[1]), Path(sys.argv[2])
lf_hash = lambda p: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
objs = sorted(p.name for p in g.glob("*.obj")); same = sum(1 for n in objs if (g2 / n).exists() and lf_hash(g / n) == lf_hash(g2 / n))
arr = lambda s: [float(x) for x in s.strip("[]").split(";") if x.strip() not in ("", "null")] if s and s != "null" else []
out = []
for name in ("mtg_template_mean_v2draft_sd0", "mtg_R3-1b", "mtg_n_04", "P01", "P02"):
    spec = spec_of(name.replace("_sd0", "")); mask = name.replace("_sd0", ""); stem = {tuple(np.round(v, 7)) for v in obj_vertices(g / f"{mask}_stem.csv.obj")}
    for lf in sorted(spec["leaves"], key=lambda l: l["k"]):
        f = g / f"{mask}_leaf{lf['k']:02d}nb.csv.obj"; t = g / f"{mask}_leaf{lf['k']:02d}nb.csv_leaves.csv"
        if not f.exists() or not t.exists():
            continue
        r = next((x for x in leaf_rows(t) if int(float(x["order"])) == 0 and int(float(x["rank"])) == lf["k"]), None)
        if r is None:
            continue
        nb = int(float(r["p_bladesNr"])); n_inner = max(0, nb - 2)
        if n_inner == 0:
            continue
        curv = float(r["p_leafCurvature"]); ang = float(r["p_anglePetiole"]); pet = float(r["p_lengthPetiole"]); segs = arr(r["p_segmentsLength"]); pts = arr(r["p_lengthPetiolules"]); inc = arr(r["p_inclPetiolules"]); props = arr(r["p_proportionSegments"])
        curv_exp = 90 + lf["dtheta_deg"] / n_inner
        # walk with the printed values: petiole, then q = 0..nb-2: inner q -> turn (curv-90) then segmentsLength[q]; last q -> turn (90-inc[q]) then lengthPetiolules[q]
        e = 90 - ang; x = pet * math.cos(math.radians(e)); z = pet * math.sin(math.radians(e)); used = []
        for q in range(nb - 1):
            if q == nb - 2:
                turn, L = 90 - inc[q], pts[q]
            else:
                turn, L = curv - 90, segs[q]
            e -= turn; x += L * math.cos(math.radians(e)); z += L * math.sin(math.radians(e)); used.append(L)
        chord_print = math.degrees(math.atan2(z, x))
        V = obj_vertices(f); Lv = np.asarray([v for v in V if tuple(np.round(v, 7)) not in stem]); b = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
        d = np.linalg.norm(Lv - b, axis=1); v = Lv[int(np.argmax(d))] - b; chord_meas = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
        cls = BX.leaf_class(lf["k"]); assumed_segs = [float(r["rachis_total_m"]) * s for s in SEG[cls]]
        out.append(dict(plant=name, rank=lf["k"], cls=cls, bladesNr=nb, n_inner=n_inner, dtheta=lf["dtheta_deg"], leafCurvature=curv, leafCurvature_expected=curv_exp, d_curv=curv - curv_exp,
                        chord_measured=chord_meas, chord_from_printed=chord_print, d_print=chord_print - chord_meas, petiole_mm=pet * 1000, petiole_assumed_mm=float(r["petiole_m"]) * 1000,
                        segments_mm=";".join(f"{s * 1000:.2f}" for s in segs), segments_assumed_mm=";".join(f"{s * 1000:.2f}" for s in assumed_segs), last_piece_mm=pts[-1] * 1000 if pts else None,
                        last_turn=90 - inc[nb - 2] if len(inc) >= nb - 1 else None, proportionSegments=";".join(str(p) for p in props), rachis_total_mm=float(r["rachis_total_m"]) * 1000,
                        sum_segments_mm=sum(segs) * 1000))
n = len(out); a_ok = sum(abs(x["d_curv"]) <= 0.01 for x in out); g_ok = sum(abs(x["d_print"]) <= 0.5 for x in out)
if a_ok < n:
    verdict = "adapter_at_fault"
elif g_ok >= 0.9 * n:
    verdict = "predictor_lengths_cause"
else:
    verdict = "turtle_differs_from_code_reading"
seg_ratio = [x["sum_segments_mm"] / x["rachis_total_mm"] for x in out if x["rachis_total_mm"]]
S = dict(obj_identical_to_run2=f"{same}/{len(objs)}", leaves=n, curvature_value_arrives=a_ok, chord_from_printed_within_0_5=g_ok, verdict=verdict,
         median_d_print=float(np.median([x["d_print"] for x in out])), median_sum_segments_over_rachis_total=float(np.median(seg_ratio)), min_max_ratio=[float(min(seg_ratio)), float(max(seg_ratio))],
         median_last_piece_mm=float(np.median([x["last_piece_mm"] for x in out if x["last_piece_mm"] is not None])), median_last_turn=float(np.median([x["last_turn"] for x in out if x["last_turn"] is not None])),
         median_petiole_minus_assumed_mm=float(np.median([x["petiole_mm"] - x["petiole_assumed_mm"] for x in out])))
(HERE / "diag_print.json").write_text(json.dumps(dict(summary=S, leaves=out), indent=1, default=float), encoding="utf-8")
with open(HERE / "diag_print.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(json.dumps(S, indent=1))
