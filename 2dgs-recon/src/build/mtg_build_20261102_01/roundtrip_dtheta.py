"""CONTRACT A1 check of adapter A-dtheta. Per leaf of the check plants (nb masks: leaflets collapsed, cylinders stay): model chord elevation
(base -> farthest cylinder vertex, stem vertices removed) vs the chord predicted from the model's own lengths (_leaves.csv petiole_m, rachis_total_m
split by PROPORTION_SEGMENTS of the rank class, last petiolule 0.01 m) with the per-segment rule RL(dtheta / n_inner) applied (PASS <= 2 deg) and
without dtheta (must differ >= 2 deg where dtheta >= 5 and n_inner >= 1). First-10-mm elevation must stay theta0 (<= 2 deg).
python roundtrip_dtheta.py <geometry dir>"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from tools import bmodel_mtg_export as BX  # noqa: E402
from mtg_spec import PUB_1_50  # noqa: E402
from roundtrip_bmodel import obj_vertices, leaf_rows, spec_of  # noqa: E402

block, _ = BX.read_block(PUB_1_50, BX.STATIC_HEAD)
SEG = {"mid": [float(BX.block_value(block, "PROPORTION_SEGMENTS_MID_LEAVES"))], "last": [float(x) for x in BX.block_value(block, "PROPORTION_SEGMENTS_LAST_LEAVES").split(";")], "first": []}
BLADES = {c: len(BX.block_value(block, f"PROPORTION_LEAFLETS_{c.upper()}_LEAVES").split(";")) for c in ("first", "mid", "last")}


def predicted_chord(theta0, pet, rach, cls, dtheta):
    """2-D walk in the vertical plane: petiole at theta0, inner segments each turned down by dtheta/n_inner, last segment + petiolule straight."""
    n_blades = BLADES[cls]; segs = [rach * s for s in SEG[cls]]; n_inner = max(0, n_blades - 2)
    x = z = 0.0; e = theta0; x += pet * math.cos(math.radians(e)); z += pet * math.sin(math.radians(e))
    per = dtheta / n_inner if n_inner else 0.0
    for i, L in enumerate(segs):
        if i < n_inner:
            e -= per
        x += L * math.cos(math.radians(e)); z += L * math.sin(math.radians(e))
    x += 0.01 * math.cos(math.radians(e)); z += 0.01 * math.sin(math.radians(e))
    return math.degrees(math.atan2(z, x)), n_inner


def check(gdir, name):
    spec = spec_of(name.replace("_sd0", "")); lt = gdir / f"{name}.csv_leaves.csv"
    if spec is None or not lt.exists():
        return dict(scenario=name, status="missing")
    mask = name.replace("_sd0", "")                       # masks are named without _sd0 (template masks already carry sd 0)
    stem = {tuple(np.round(v, 7)) for v in obj_vertices(gdir / f"{mask}_stem.csv.obj")}
    out = dict(scenario=name, leaves=[])
    for lf in sorted(spec["leaves"], key=lambda l: l["k"]):
        f = gdir / f"{mask}_leaf{lf['k']:02d}nb.csv.obj"; t = gdir / f"{mask}_leaf{lf['k']:02d}nb.csv_leaves.csv"
        if not f.exists() or not t.exists():
            continue
        r = next((x for x in leaf_rows(t) if int(float(x["order"])) == 0 and int(float(x["rank"])) == lf["k"]), None)
        if r is None:
            continue
        V = obj_vertices(f); L = np.asarray([v for v in V if tuple(np.round(v, 7)) not in stem]); b = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
        if not len(L):
            out["leaves"].append(dict(rank=lf["k"], status="no leaf vertices")); continue
        d = np.linalg.norm(L - b, axis=1); tip = L[int(np.argmax(d))]; v = tip - b; chord = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
        near = L[(d > 1e-4) & (d < 0.010)]; v0 = (near.mean(axis=0) - b) if len(near) else v; el0 = math.degrees(math.asin(v0[2] / np.linalg.norm(v0)))
        cls = BX.leaf_class(lf["k"]); pet = float(r["petiole_m"]); rach = float(r["rachis_total_m"]); th = 90 - float(r["inclination_deg"])
        p_bent, n_inner = predicted_chord(th, pet, rach, cls, lf["dtheta_deg"]); p_straight, _ = predicted_chord(th, pet, rach, cls, 0.0)
        rec = dict(rank=lf["k"], cls=cls, n_inner=n_inner, dtheta_in=lf["dtheta_deg"], theta0_in=lf["theta0_deg"], el_first10=el0, chord_model=chord, chord_pred_bent=p_bent, chord_pred_straight=p_straight,
                   d_bent=abs(chord - p_bent), d_straight=abs(chord - p_straight), d_el0=abs(el0 - lf["theta0_deg"]))
        rec["pass"] = rec["d_bent"] <= 2 and rec["d_el0"] <= 2 and (n_inner == 0 or lf["dtheta_deg"] < 5 or rec["d_straight"] >= 2)
        rec["status"] = "dtheta_no_segment" if n_inner == 0 else ("pass" if rec["pass"] else "FAIL")
        out["leaves"].append(rec)
    out["n"] = len(out["leaves"]); out["pass"] = sum(1 for x in out["leaves"] if x.get("status") == "pass"); out["no_segment"] = sum(1 for x in out["leaves"] if x.get("status") == "dtheta_no_segment")
    out["fail"] = sum(1 for x in out["leaves"] if x.get("status") == "FAIL")
    return out


if __name__ == "__main__":
    g = Path(sys.argv[1]); R = {n: check(g, n) for n in ("mtg_template_mean_v2draft_sd0", "mtg_R3-1b", "mtg_n_04", "P01", "P02")}
    (HERE / "roundtrip_dtheta.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    with open(HERE / "roundtrip_dtheta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["scenario", "rank", "class", "n_inner_segments", "dtheta_in", "theta0_in", "el_first10_model", "chord_model", "chord_pred_with_dtheta", "chord_pred_straight", "d_bent", "d_straight", "status"])
        for n, r in R.items():
            for x in r.get("leaves", []):
                w.writerow([n, x.get("rank"), x.get("cls"), x.get("n_inner"), x.get("dtheta_in"), x.get("theta0_in"), x.get("el_first10"), x.get("chord_model"), x.get("chord_pred_bent"), x.get("chord_pred_straight"), x.get("d_bent"), x.get("d_straight"), x.get("status")])
    for n, r in R.items():
        print(n, r.get("status") or f"pass {r['pass']} / fail {r['fail']} / no_segment {r['no_segment']} of {r['n']}")
