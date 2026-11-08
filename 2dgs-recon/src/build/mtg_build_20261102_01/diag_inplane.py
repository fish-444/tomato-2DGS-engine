"""CONTRACT A2b: lateral offset of A1's farthest vertex + in-plane chord (vertices within 1 mm of the leaf's vertical plane). Local, existing geometry."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from roundtrip_bmodel import obj_vertices, leaf_rows, spec_of  # noqa: E402
from roundtrip_dtheta import BX, predicted_chord  # noqa: E402
g = Path(sys.argv[1]); out = []
for name in ("mtg_template_mean_v2draft_sd0", "mtg_R3-1b", "mtg_n_04", "P01", "P02"):
    spec = spec_of(name.replace("_sd0", "")); mask = name.replace("_sd0", ""); stem = {tuple(np.round(v, 7)) for v in obj_vertices(g / f"{mask}_stem.csv.obj")}
    for lf in sorted(spec["leaves"], key=lambda l: l["k"]):
        f = g / f"{mask}_leaf{lf['k']:02d}nb.csv.obj"; t = g / f"{mask}_leaf{lf['k']:02d}nb.csv_leaves.csv"
        if not f.exists() or not t.exists():
            continue
        r = next((x for x in leaf_rows(t) if int(float(x["order"])) == 0 and int(float(x["rank"])) == lf["k"]), None)
        cls = BX.leaf_class(lf["k"]); n_inner = max(0, len(BX.block_value(BX.read_block(HERE.parents[1] / "build/bmodel_mtg_20261101_01/upstream/model/input/static_input/1_50.csv", BX.STATIC_HEAD)[0], f"PROPORTION_LEAFLETS_{cls.upper()}_LEAVES").split(";")) - 2)
        if r is None or n_inner == 0:
            continue
        V = obj_vertices(f); L = np.asarray([v for v in V if tuple(np.round(v, 7)) not in stem]); b = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
        az = math.radians((float(r["orientation_deg"]) - 90) % 360); h = np.array([math.cos(az), math.sin(az), 0.0]); nrm = np.array([-h[1], h[0], 0.0])
        D = L - b; dist = np.linalg.norm(D, axis=1); k = int(np.argmax(dist)); off_far = abs(float(D[k] @ nrm)) * 1000
        inp = np.abs(D @ nrm) <= 0.001; Q = D[inp & (D @ h > 0)]
        if not len(Q):
            continue
        q = Q[int(np.argmax(np.linalg.norm(Q, axis=1)))]; chord_in = math.degrees(math.atan2(q[2], float(q @ h)))
        chord_far = math.degrees(math.asin(D[k][2] / dist[k])); th = 90 - float(r["inclination_deg"])
        p_bent, _ = predicted_chord(th, float(r["petiole_m"]), float(r["rachis_total_m"]), cls, lf["dtheta_deg"])
        out.append(dict(plant=name, rank=lf["k"], cls=cls, dtheta=lf["dtheta_deg"], far_offset_mm=off_far, chord_far=chord_far, chord_inplane=chord_in, chord_pred_bent=p_bent,
                        d_far=chord_far - p_bent, d_inplane=chord_in - p_bent, inplane_pass=abs(chord_in - p_bent) <= 2))
n = len(out); n_in = sum(x["inplane_pass"] for x in out); n_off = sum(x["far_offset_mm"] > 2 for x in out)
verdict = "A1_measure_is_cause;adapter_acts_as_declared" if n and n_in >= 0.9 * n and n_off >= 0.9 * n else "not_separated"
S = dict(leaves=n, inplane_within_2deg=n_in, far_vertex_off_plane_gt_2mm=n_off, median_d_far=float(np.median([x["d_far"] for x in out])), median_d_inplane=float(np.median([x["d_inplane"] for x in out])),
         max_abs_d_inplane=float(max(abs(x["d_inplane"]) for x in out)), median_far_offset_mm=float(np.median([x["far_offset_mm"] for x in out])), verdict=verdict)
(HERE / "diag_inplane.json").write_text(json.dumps(dict(summary=S, leaves=out), indent=1, default=float), encoding="utf-8")
with open(HERE / "diag_inplane.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(json.dumps(S, indent=1))
