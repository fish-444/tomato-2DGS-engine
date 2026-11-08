"""B-model round trip (CONTRACT §5): our MTG spec -> B-model scenario -> import (Kaggle mtgb-bm-import) -> model day-0 leaf table / OBJ -> compare.
Per scenario <name>: <name>.csv_leaves.csv rows (orientation, inclination, blade area) vs the spec (orientation = az + 90, inclination = 90 - theta0,
area); counts and order exact; angles <= 2 deg; area <= 1 % (static import copies the column). Length (<= 1 %) only where masks exist
(<name>_leafKK: leaf vertices = OBJ(leafKK) minus OBJ(stem); length = |farthest vertex - base|; straight leaf in the model). One-row test: elevation
30 +- 2, length 40 +- 1 %. dtheta: not representable (reported). python roundtrip_bmodel.py <geometry dir>"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def obj_vertices(p):
    V = [[float(x) for x in ln.split()[1:4]] for ln in open(p, encoding="utf-8", errors="replace") if ln.startswith("v ")]
    V = np.asarray(V, float); return V[:, [0, 2, 1]] if len(V) else V        # GroIMP y-up OBJ -> (x, y, z)


def leaf_rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def spec_of(name):
    for d in (HERE / "static", HERE / "dynamic/check_plants"):
        p = d / f"{name}_spec.json"
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    return None


def compare(gdir: Path, name: str):
    spec = spec_of(name); lt = gdir / f"{name}.csv_leaves.csv"
    if spec is None or not lt.exists():
        return dict(scenario=name, status="no spec or no model output")
    rows = leaf_rows(lt); main = sorted([r for r in rows if int(float(r["order"])) == 0], key=lambda r: int(float(r["rank"]))); side = [r for r in rows if int(float(r["order"])) == 1]
    exp_main = sorted(spec["leaves"], key=lambda l: l["k"]); exp_side = [lf for sa in spec.get("side_axes", []) for lf in sa["leaves"]]
    res = dict(scenario=name, kind=spec["kind"], n_main_in=len(exp_main), n_main_model=len(main), n_side_in=len(exp_side), n_side_model=len(side), order_pass=[int(float(r["rank"])) for r in main] == [l["k"] for l in exp_main], leaves=[])
    stem = None
    sp = gdir / f"{name}_stem.csv.obj"
    if sp.exists():
        stem = {tuple(np.round(v, 7)) for v in obj_vertices(sp)}
    for lf, r in zip(exp_main, main):
        ori_exp = (lf["azimuth_deg"] + 90) % 360; inc_exp = 90 - lf["theta0_deg"]
        d_ori = abs((float(r["orientation_deg"]) - ori_exp + 180) % 360 - 180); d_inc = abs(float(r["inclination_deg"]) - inc_exp)
        a_model = float(r["blade_area_total_m2"]) * 1e6; d_area = abs(a_model / lf["area_mm2"] - 1) if lf.get("area_mm2") else None
        rec = dict(rank=lf["k"], uid=lf["uid"], d_orientation=d_ori, d_inclination=d_inc, d_area_rel=d_area, theta0_in=lf["theta0_deg"], inclination_model=float(r["inclination_deg"]), dtheta_in=lf["dtheta_deg"], dtheta_model="not_representable")
        f = gdir / f"{name}_leaf{lf['k']:02d}.csv.obj"
        if stem is not None and f.exists():
            V = obj_vertices(f); L = np.asarray([v for v in V if tuple(np.round(v, 7)) not in stem]); b = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
            if len(L):
                d = np.linalg.norm(L - b, axis=1); tip = L[int(np.argmax(d))]; v = tip - b
                rec.update(length_model_mm=float(d.max() * 1000), length_in_mm=lf["length_mm"], d_length_rel=float(d.max() * 1000 / lf["length_mm"] - 1), elevation_model_tip=math.degrees(math.asin(v[2] / np.linalg.norm(v))))
        tol = (rec.get("dL_check") is None)
        rec["pass"] = d_ori <= 2 and d_inc <= 2 and (d_area is None or d_area <= 0.01) and (abs(rec.get("d_length_rel", 0)) <= 0.01)
        if "d_length_rel" in rec and abs(rec["d_length_rel"]) > 0.01:
            rec["fail_reason"] = "length: model minimum leaf (10 mm petiolule + terminal leaflet from area) vs observed line (A1 area priority; model limit)" if "length_short" in str(lf.get("flags")) or rec["d_length_rel"] > 0 else "length"
        res["leaves"].append(rec)
    res["leaves_pass"] = sum(1 for x in res["leaves"] if x["pass"]); res["counts_pass"] = res["n_main_in"] == res["n_main_model"] and res["n_side_in"] == res["n_side_model"] and res["order_pass"]
    res["pass_all"] = res["counts_pass"] and res["leaves_pass"] == len(res["leaves"])
    return res


def one_row(gdir: Path):
    lt = gdir / "one_row_test.csv_leaves.csv"; f = gdir / "one_row_test.csv.obj"
    if not lt.exists():
        return dict(status="no output")
    r = leaf_rows(lt)[0]; out = dict(orientation_model=float(r["orientation_deg"]), orientation_expected=90.0, inclination_model=float(r["inclination_deg"]), inclination_expected=60.0)
    if f.exists():
        V = obj_vertices(f); b = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]); d = np.linalg.norm(V - b, axis=1)
        near = V[(d > 1e-4) & (d < 0.2)]
        if len(near):
            dd = np.linalg.norm(near - b, axis=1); tip = near[int(np.argmax(dd))]; v = tip - b
            out.update(length_model_mm=float(dd.max() * 1000), length_expected_mm=40.0, elevation_model_deg=math.degrees(math.asin(v[2] / np.linalg.norm(v))), elevation_expected_deg=30.0)
    out["pass"] = abs(out["inclination_model"] - 60) <= 2 and abs(out["orientation_model"] - 90) <= 2 and abs(out.get("length_model_mm", 40) / 40 - 1) <= 0.01 and abs(out.get("elevation_model_deg", 30) - 30) <= 2
    return out


if __name__ == "__main__":
    g = Path(sys.argv[1]); names = sorted({p.name.split(".csv")[0] for p in g.glob("*.csv_leaves.csv") if not any(t in p.name for t in ("_leaf", "_stem", "_sd0", "one_row"))})
    R = dict(one_row_test=one_row(g), scenarios={n: compare(g, n) for n in names})
    sd0 = g / "mtg_template_mean_v2draft_sd0.csv_leaves.csv"
    if sd0.exists():
        import shutil; shutil.copy2(HERE / "static/mtg_template_mean_v2draft_spec.json", HERE / "static/mtg_template_mean_v2draft_sd0_spec.json")
        R["scenarios"]["mtg_template_mean_v2draft_sd0"] = compare(g, "mtg_template_mean_v2draft_sd0")
    (HERE / "roundtrip_bmodel.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    with open(HERE / "roundtrip_bmodel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["scenario", "kind", "n_main_in", "n_main_model", "n_side_in", "n_side_model", "counts_order_pass", "leaves_pass", "leaves_total", "max_d_orientation", "max_d_inclination", "max_d_area_rel", "length_checked", "length_pass", "pass_all"])
        for n, r in R["scenarios"].items():
            if "leaves" not in r:
                w.writerow([n, r.get("status", "")]); continue
            Ls = [x for x in r["leaves"] if "d_length_rel" in x]
            w.writerow([n, r["kind"], r["n_main_in"], r["n_main_model"], r["n_side_in"], r["n_side_model"], r["counts_pass"], r["leaves_pass"], len(r["leaves"]), max(x["d_orientation"] for x in r["leaves"]), max(x["d_inclination"] for x in r["leaves"]),
                        max((x["d_area_rel"] or 0) for x in r["leaves"]), len(Ls), sum(1 for x in Ls if abs(x["d_length_rel"]) <= 0.01), r["pass_all"]])
    print("one_row", R["one_row_test"]); [print(n, r.get("counts_pass"), r.get("leaves_pass"), "/", len(r.get("leaves", [])), r.get("pass_all")) for n, r in R["scenarios"].items()]
