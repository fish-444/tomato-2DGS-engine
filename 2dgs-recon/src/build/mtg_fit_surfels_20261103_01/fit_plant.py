"""One plant: stem fit -> nodes re-placed -> per-leaf Level 1 / Level 2 fit + gate -> fitted spec + table rows (CONTRACT §2-3).
Also: real-plant loader (§1) and the writers wrapper (engine Level 2 polylines through the old build's add_leaf)."""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_lib as FL  # noqa: E402
from fit_lib import W, BX, ROOT, OLD  # noqa: E402
import mtg_spec as MS  # noqa: E402  (old build, read only)

SRC = "fitted_to_surfels_ungraded"


# ------------------------------------------------------------------ real plants
def read_ply_xyz(p):
    b = Path(p).read_bytes(); i = b.find(b"end_header") + len(b"end_header\n"); head = b[:i].decode(errors="replace").splitlines()
    n = int(next(l for l in head if l.startswith("element vertex")).split()[-1])
    props = [l.split() for l in head if l.startswith("property")]
    dt = np.dtype([(q[2], {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}[q[1]]) for q in props])
    a = np.frombuffer(b, dtype=dt, count=n, offset=i)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(float) * 1000.0


def load_real(pid):
    spec = json.load(open(OLD / f"static/mtg_{pid}_spec.json", encoding="utf-8")); sk = json.load(open(ROOT / spec["skeleton_file"], encoding="utf-8"))
    uids = [lf["uid"] for lf in spec["leaves"]]
    if pid == "R3-1b":
        z = np.load(ROOT / "build/engine_coplanar_sep_20261101_01/out_k1/per_surfel_D.npz"); X = np.asarray(z["xyz"], float) * 1000.0
        r8 = {l["leaf_uid"]: int(l["attrs"]["pipeline_leaf"]) for l in sk["leaves"]}; lab = np.asarray(z["leaf_R8pc"]); ax = np.asarray(z["axis_kind"])
        leaf_idx = {u: np.nonzero((lab == r8[u]) & (ax != 1))[0] for u in uids}; leaf_mask = (lab > 0) & (ax != 1); stem_mask = ax == 1
        src = "per_surfel_D.npz leaf_R8pc (human-seeded R8_pc), axis_kind 1 = stem"
    else:
        X = read_ply_xyz(ROOT / f"build/p1_recon_all_20261025_01/outputs/{pid}/R3-1b-style_{pid}_surfels_full.ply")
        L = np.load(MS.DZ / f"labels_{MS.ROUTE[pid]}_{pid}_{MS.SNAP[pid]}.npz"); rows = np.asarray(L["rows"]); X = X[rows]
        lu = [str(u) for u in L["leaf_uids"]]; lab = np.asarray(L["label"]); st = np.asarray(L["status"])
        leaf_idx = {u: np.nonzero((lab == lu.index(u) + 1) & (st == 1))[0] for u in uids}; leaf_mask = (lab > 0) & (st == 1); stem_mask = np.zeros(len(X), bool)
        src = f"{MS.ROUTE[pid]} labels skelver {MS.SNAP[pid]} (assigned only); surfels = ply rows of the labels"
    clicks = {l["leaf_uid"]: np.asarray(l["polyline_xyz"], float) * 1000.0 for l in sk["leaves"]}
    return dict(spec=spec, X=X, leaf_idx=leaf_idx, leaf_mask=leaf_mask, stem_mask=stem_mask, clicks=clicks, label_source=src)


# ------------------------------------------------------------------ one plant
def fit_plant(D, pooled, log=print):
    spec = D["spec"]; X = D["X"]; tree = cKDTree(X); out = copy.deepcopy(spec)
    stem = FL.fit_stem(np.asarray(spec["axis"]["curve_mm"], float), X, D["leaf_mask"], tree)
    if "vertical_proxy" in spec["axis"]["source"]:
        stem["flags"].append("axis_is_vertical_proxy")
    curve = np.asarray(stem["curve"], float)
    out["axis"]["curve_prev_mm"] = spec["axis"]["curve_mm"]; out["axis"]["curve_mm"] = curve.tolist(); out["axis"]["source"] = f"{SRC} (start: {spec['axis']['source']}); flags {stem['flags']}"
    node_xyz = {}
    for n in out["nodes"]:
        p = FL.point_at_s(curve, n["s_mm"]); n["xyz_prev_mm"] = n["xyz_mm"]; n["xyz_mm"] = p.tolist(); node_xyz[n["k"]] = p
    stem_idx = np.nonzero(D["stem_mask"])[0]; nonstem = np.ones(len(X), bool); nonstem[stem_idx] = False
    rows = []
    for lf, lo in zip(spec["leaves"], out["leaves"]):
        uid = lf["uid"]; k = lf["k"]; rec = dict(plant=spec["plant_id"], k=k, uid=uid, azimuth_prev=lf["azimuth_deg"], theta0_prev=lf["theta0_deg"], dtheta_prev=lf["dtheta_deg"], length_prev=lf["length_mm"], flags=[])
        click = D["clicks"].get(uid, np.zeros((0, 3)))
        if lf.get("geometry") != "observed" or len(click) < 2:
            rec.update(status="not_fitted_model_completed"); rec["flags"].append("not_fitted_model_completed"); lo["fit_status"] = "not_fitted_model_completed"; rows.append(rec); continue
        idx = D["leaf_idx"].get(uid, np.zeros(0, int)); S = X[idx]
        if len(idx) < FL.FEW:
            near = nonstem & (FL.point_seg_dist(X, FL.resample(click, 1.0)[0]) <= FL.FEW_R) if len(X) < 200000 else nonstem
            S = np.vstack([X[np.nonzero(near)[0]], FL.resample(click, 1.0)[0]]); rec["flags"].append("few_leaf_surfels")
        area = lf["area_mm2"] if lf.get("area_mm2") else MS.pooled_area(pooled, k)[0]
        if not lf.get("area_mm2"):
            rec["flags"].append("area_helper_pooled_rank_for_fit_only")
        base = node_xyz[lf["node_k"]]; F = FL.LeafFit(base, lf["leaflets"], area, S, click, tree)
        phi0 = lf["azimuth_deg"]; L0 = lf["length_mm"]; el_sk = lf.get("skeleton_elevation_deg")
        bounds = [(phi0 - 60, phi0 + 60), (-60.0, 90.0), (0.0, 120.0), (0.7 * L0, 1.3 * L0)]
        starts = [(phi0, lf["theta0_deg"], lf["dtheta_deg"], L0)] + ([(phi0, el_sk, lf["dtheta_deg"], L0)] if el_sk is not None else [])
        r1 = F.fit1(starts, bounds); pl1 = F.pl1(r1.x); g1 = F.gate(pl1)
        rec.update(azimuth_fit=float(r1.x[0] % 360), theta0_fit=float(r1.x[1]), dtheta_fit=float(r1.x[2]), length_fit=float(r1.x[3]), cost1=float(r1.fun), n_set=int(len(S)),
                   gate1_median_mm=g1["median_dist_mm"], gate1_share5=g1["share_within_5mm"], gate1=g1["passed"])
        rec["cost_start"] = float(min(F.cost_pl(F.pl1(np.clip(np.asarray(x0, float), [b[0] for b in bounds], [b[1] for b in bounds]))) for x0 in starts))
        g0 = F.gate(F.pl1(np.clip(np.asarray(starts[0], float), [b[0] for b in bounds], [b[1] for b in bounds]))); rec.update(gate_before_median_mm=g0["median_dist_mm"], gate_before_share5=g0["share_within_5mm"])
        lo["fit_level1"] = dict(azimuth_deg=rec["azimuth_fit"], theta0_deg=rec["theta0_fit"], dtheta_deg=rec["dtheta_fit"], length_mm=rec["length_fit"], cost=rec["cost1"], gate=g1)
        if not g1["passed"]:
            rec.update(status="fit_failed_kept_original"); rec["flags"].append("fit_failed_kept_original"); lo["fit_status"] = "fit_failed_kept_original"; lo["flags"] = list(lo["flags"]) + ["fit_failed_kept_original"]
            rows.append(rec); continue
        r2, pl2 = F.fit2(r1.x, bounds[0]); g2 = F.gate(pl2)
        rec.update(cost2=float(r2.fun), gate2_median_mm=g2["median_dist_mm"], gate2_share5=g2["share_within_5mm"], gate2=g2["passed"], status="fitted")
        lo.update(azimuth_prev_deg=lf["azimuth_deg"], theta0_hand_deg=lf["theta0_deg"], dtheta_prev_deg=lf["dtheta_deg"], length_prev_mm=lf["length_mm"], base_prev_mm=lf["base_mm"],
                  azimuth_deg=rec["azimuth_fit"], theta0_deg=rec["theta0_fit"], dtheta_deg=rec["dtheta_fit"], length_mm=rec["length_fit"], base_mm=base.tolist(),
                  azimuth_source=f"{SRC} (was: {lf['azimuth_source']})", theta0_source=f"{SRC} (was: {lf['theta0_source']})", dtheta_source=f"{SRC} (was: {lf['dtheta_source']})",
                  length_source=f"{SRC} (was: {lf['length_source']})", fit_status="fitted")
        if g2["passed"]:
            lo["rachis_polyline_level2_mm"] = pl2.tolist(); lo["level2_gate"] = g2
        else:
            rec["flags"].append("level2_failed_level1_used"); lo["level2_status"] = "level2_failed_level1_used"
        rows.append(rec)
    out["labels"] = list(dict.fromkeys(list(spec["labels"]) + ["fitted_to_surfels", "skeleton_labels_ungraded"]))
    out["fit"] = dict(contract="build/mtg_fit_surfels_20261103_01/CONTRACT.md", label_source=D.get("label_source"), stem={k: v for k, v in stem.items() if k != "curve"},
                      note="fitted geometry of this plant, not a trait; theta0_hand_deg keeps the template value")
    return out, stem, rows


# ------------------------------------------------------------------ mode B (CONTRACT B): theta0 / dtheta fixed; arms B1 (L bound from path) / B2 (from node)
def fit_plant_B(D, pooled, arm):
    spec = D["spec"]; X = D["X"]; tree = cKDTree(X); out = copy.deepcopy(spec)
    stem = FL.fit_stem(np.asarray(spec["axis"]["curve_mm"], float), X, D["leaf_mask"], tree)
    if "vertical_proxy" in spec["axis"]["source"]:
        stem["flags"].append("axis_is_vertical_proxy")
    curve = np.asarray(stem["curve"], float)
    out["axis"]["curve_prev_mm"] = spec["axis"]["curve_mm"]; out["axis"]["curve_mm"] = curve.tolist(); out["axis"]["source"] = f"{SRC} (start: {spec['axis']['source']}); flags {stem['flags']}"
    node_xyz = {}
    for n in out["nodes"]:
        p = FL.point_at_s(curve, n["s_mm"]); n["xyz_prev_mm"] = n["xyz_mm"]; n["xyz_mm"] = p.tolist(); node_xyz[n["k"]] = p
    nonstem = ~np.asarray(D["stem_mask"], bool); rows = []
    for lf, lo in zip(spec["leaves"], out["leaves"]):
        uid = lf["uid"]; k = lf["k"]
        rec = dict(plant=spec["plant_id"], arm=arm, k=k, uid=uid, azimuth_prev=lf["azimuth_deg"], theta0_prev=lf["theta0_deg"], dtheta_prev=lf["dtheta_deg"], length_prev=lf["length_mm"], flags=[])
        click = D["clicks"].get(uid, np.zeros((0, 3)))
        if lf.get("geometry") != "observed" or len(click) < 2:
            rec.update(status="not_fitted_model_completed"); rec["flags"].append("not_fitted_model_completed"); lo["fit_status"] = "not_fitted_model_completed"; rows.append(rec); continue
        idx = D["leaf_idx"].get(uid, np.zeros(0, int)); S = X[idx]
        if len(idx) < FL.FEW:
            near = nonstem & (FL.point_seg_dist(X, FL.resample(click, 1.0)[0]) <= FL.FEW_R)
            S = np.vstack([X[np.nonzero(near)[0]], FL.resample(click, 1.0)[0]]); rec["flags"].append("few_leaf_surfels")
        area = lf["area_mm2"] if lf.get("area_mm2") else MS.pooled_area(pooled, k)[0]
        if not lf.get("area_mm2"):
            rec["flags"].append("area_helper_pooled_rank_for_fit_only")
        base = node_xyz[lf["node_k"]]; F = FL.LeafFit(base, lf["leaflets"], area, S, click, tree)
        phi0 = lf["azimuth_deg"]; path = float(BX.path_len(click)); ref = path if arm == "B1" else float(np.linalg.norm(click[0] - base)) + path
        bphi = (phi0 - 60, phi0 + 60); bL = (0.7 * ref, 1.3 * ref) if arm != "D" else (lf["length_mm"] - 1e-6, lf["length_mm"] + 1e-6); th, dt = lf["theta0_deg"], lf["dtheta_deg"]   # arm D (CONTRACT D): L fixed
        r1 = FL.fitB1(F, th, dt, bphi, bL, (phi0, lf["length_mm"])); pl1 = F.pl1((r1.x[0], th, dt, r1.x[1])); g1 = F.gate(pl1)
        g0 = F.gate(F.pl1((phi0, th, dt, float(np.clip(lf["length_mm"], *bL)))))
        rec.update(azimuth_fit=float(r1.x[0] % 360), length_fit=float(r1.x[1]), theta0_fit=th, dtheta_fit=dt, L_ref=ref, cost1=float(r1.fun), n_set=int(len(S)), gate1_median_mm=g1["median_dist_mm"],
                   gate1_share5=g1["share_within_5mm"], gate1=g1["passed"], gate_before_median_mm=g0["median_dist_mm"], gate_before_share5=g0["share_within_5mm"])
        lo["fit_level1"] = dict(azimuth_deg=rec["azimuth_fit"], length_mm=rec["length_fit"], cost=rec["cost1"], gate=g1, arm=arm)
        if not g1["passed"]:
            rec.update(status="fit_failed_kept_original"); rec["flags"].append("fit_failed_kept_original"); lo["fit_status"] = "fit_failed_kept_original"; rows.append(rec); continue
        r2, pl2 = FL.fitB2(F, r1.x[0], th, dt, r1.x[1], bphi); g2 = F.gate(pl2)
        rec.update(cost2=float(r2.fun), gate2_median_mm=g2["median_dist_mm"], gate2_share5=g2["share_within_5mm"], gate2=g2["passed"], status="fitted")
        lo.update(azimuth_prev_deg=lf["azimuth_deg"], length_prev_mm=lf["length_mm"], base_prev_mm=lf["base_mm"], theta0_hand_deg=lf["theta0_deg"], azimuth_deg=rec["azimuth_fit"], length_mm=rec["length_fit"],
                  base_mm=base.tolist(), azimuth_source=f"{SRC} (was: {lf['azimuth_source']})", length_source=f"{SRC} (was: {lf['length_source']})", fit_status="fitted",
                  fit_mode=f"B ({arm}): theta0/dtheta fixed at the spec values")
        if g2["passed"]:
            lo["rachis_polyline_level2_mm"] = pl2.tolist(); lo["level2_gate"] = g2
        else:
            rec["flags"].append("level2_failed_level1_used"); lo["level2_status"] = "level2_failed_level1_used"
        rows.append(rec)
    out["labels"] = list(dict.fromkeys(list(spec["labels"]) + ["fitted_to_surfels", "skeleton_labels_ungraded", f"fit_mode_B_{arm}"]))
    out["fit"] = dict(contract="build/mtg_fit_surfels_20261103_01/CONTRACT.md (amendment B)", arm=arm, label_source=D.get("label_source"), stem={k: v for k, v in stem.items() if k != "curve"},
                      note="fitted geometry of this plant (phi, L, stem); theta0 / dtheta = hand template / draft row, unchanged")
    return out, stem, rows


# ------------------------------------------------------------------ writers (old build) with Level-2 polylines
def write_fitted(spec, out_dir: Path, name):
    queue = [np.asarray(lf["rachis_polyline_level2_mm"], float) if lf.get("rachis_polyline_level2_mm") else None for lf in spec["leaves"]]
    orig = W.leaf_polyline; it = iter(queue)
    def patched(base, az, th, dt, L, m):
        q = next(it, None)
        if q is None:
            return orig(base, az, th, dt, L, m)
        return [p.tolist() for p in q], None
    W.leaf_polyline = patched
    try:
        tree, rep = W.write_engine(spec, out_dir, name)
    finally:
        W.leaf_polyline = orig
    sha, info = W.spec_to_bmodel(spec, out_dir / f"{name}_bmodel.csv")
    return tree, rep, sha, info
