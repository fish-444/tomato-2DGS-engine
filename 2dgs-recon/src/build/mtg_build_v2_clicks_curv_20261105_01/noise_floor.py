"""Noise floor (CONTRACT): engine flat leaflets sampled as surfels, true membership, same curvature fit. python -X utf8 noise_floor.py"""
import csv, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

CURV = Path(__file__).resolve().parent; LEAF = CURV.parent / "mtg_build_v2_clicks_leaflets_20261105_01"
sys.path.insert(0, str(CURV))
import curv_lib as CL  # noqa: E402
src = open(LEAF / "synthetic_check.py", encoding="utf-8").read().split("rows = []")[0].replace("Path(__file__)", "Path(r'" + str(LEAF / "x.py") + "')")
exec(src)   # defines SL, chain, sample, ARMS, GEO, REAL_AREA, MIN_LEAF, rng (seed 0), ...

rows = []
for arm in ARMS:
    z = np.load(GEO / arm / "geometry_day71.npz", allow_pickle=True); tri = np.asarray(z["tri"], float) * 1000.0; own = np.asarray(z["owner"]).astype(str); cls = np.asarray(z["cls"]).astype(str)
    by = defaultdict(list)
    for i, o in enumerate(own):
        m = re.match(r"^(.*?/C\d+)/(.+)$", o)
        if m and cls[i] in ("Leaflet", "Petiole", "RachisSegment"):
            by[m.group(1)].append(i)
    for cid in sorted(by):
        ii = np.array(by[cid]); lf = ii[cls[ii] == "Leaflet"]
        la = sum(0.5 * np.linalg.norm(np.cross(tri[i, 1] - tri[i, 0], tri[i, 2] - tri[i, 0])) for i in lf)
        if la < MIN_LEAF:
            continue
        P = [tri[i] for i in ii if own[i] == f"{cid}/P"]; Rk = sorted({own[i] for i in ii if re.match(rf"^{re.escape(cid)}/R\d+$", own[i])}, key=lambda s: int(s.rsplit("R", 1)[1]))
        if not P:
            continue
        Rs = [np.vstack([tri[i] for i in ii if own[i] == r]) for r in Rk]; curve = chain(np.vstack(P), Rs, tri[lf].reshape(-1, 3).mean(0))
        k_sc = float(np.sqrt(REAL_AREA / la)); o0 = curve[0].copy(); curve = o0 + (curve - o0) * k_sc; T_s = o0 + (tri[ii] - o0) * k_sc
        X, A, O, Nn = sample(T_s, own[ii]); fr = SL.frame(X, curve)
        for o in sorted(set(O)):
            nm = o.rsplit("/", 1)[1]
            if not (nm == "T" or re.match(r"^P\d[LR]$", nm)):
                continue
            m = O == o
            if m.sum() < CL.MIN_N:
                continue
            g = SL.leaflet_geometry(X[m], A[m], None, dict(r=fr["r"][m], s=fr["s"][m])); f = CL.fit(X[m], A[m], Nn[m], g["base"])
            rows.append(dict(arm=arm, leaf=cid, leaflet=nm, position="T" if nm == "T" else nm[:2], kappa_l=f["kappa_l"], kappa_w=f["kappa_w"], twist_c=f["twist_c"], r2=f["r2"],
                             dalpha_l_deg=f["dalpha_l_deg"], dalpha_w_deg=f["dalpha_w_deg"], ndev_median_deg=f["ndev_median_deg"], ndev_p90_deg=f["ndev_p90_deg"],
                             proj_ratio=CL.proj_ratio(g["mesh"]["vertices"], g["mesh"]["faces"]), rms_residual_mm=f["rms_residual_mm"], n=f["n"], length_mm=f["length_mm"], width_mm=f["width_mm"]))
with open(CURV / "noise_floor.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
Q = ["kappa_l", "kappa_w", "twist_c", "dalpha_l_deg", "dalpha_w_deg", "r2", "ndev_median_deg", "ndev_p90_deg", "proj_ratio", "rms_residual_mm"]
R = dict(leaflets=len(rows), note="engine flat leaflets (true membership), position noise 0.124 mm, triangle normals; truth: kappa = 0, normal deviation = 0",
         median={q: float(np.median([r[q] for r in rows])) for q in Q}, p90_abs={q: float(np.percentile(np.abs([r[q] for r in rows]), 90)) for q in Q})
json.dump(R, open(CURV / "noise_floor.json", "w", encoding="utf-8"), indent=1); print(json.dumps(R, indent=1))
