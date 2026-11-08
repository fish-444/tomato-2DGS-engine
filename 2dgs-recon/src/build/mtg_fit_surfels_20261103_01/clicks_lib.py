"""CONTRACT G: compound-leaf rachis built directly from the clicked line (node + clicks, Gaussian-smoothed along arc length, no extension)."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_lib as FL  # noqa: E402
from fit_lib import W  # noqa: E402

SIGMA = 5.0; GATE_MED = 3.0; GATE_IN = 5.0; GATE_SHARE = 0.90; SRC = "rachis_from_clicks_user_measured_ungraded"


def click_curve(node, click):
    """(curve every 1 mm from node to the last click, smoothed; s of the first click)."""
    node = np.asarray(node, float); C = np.asarray(click, float)
    if np.linalg.norm(C[-1] - node) < np.linalg.norm(C[0] - node):
        C = C[::-1]
    P = np.vstack([node, C]); R, s = FL.resample(P, 1.0)
    s_first = float(np.linalg.norm(C[0] - node))
    if len(R) >= 5:
        Rs = gaussian_filter1d(R, SIGMA, axis=0, mode="nearest"); Rs[0] = R[0]; Rs[-1] = R[-1]; R = Rs
    return R, s_first


def writer_points(R, m):
    """Curve sampled at the template piece ends: 0, 0.30 L, then m equal pieces to L."""
    Q, s = FL.resample(R, 0.25); L = s[-1]
    t = [0.0, W.PET_SHARE * L] + [W.PET_SHARE * L + (1 - W.PET_SHARE) * L * (i + 1) / max(m, 1) for i in range(m)]
    return np.stack([np.interp(t, s, Q[:, k]) for k in range(3)], 1), L


def angles(pl):
    v = pl[1] - pl[0]; az = math.degrees(math.atan2(v[1], v[0])) % 360; el = math.degrees(math.asin(np.clip(v[2] / np.linalg.norm(v), -1, 1)))
    d = np.diff(pl, axis=0); el_all = [math.degrees(math.asin(np.clip(x[2] / np.linalg.norm(x), -1, 1))) for x in d]
    return az, el, float(el_all[0] - el_all[-1]), float(FL.polyline_turning(pl)) if hasattr(FL, "polyline_turning") else None


def node_link_gate(R, s_first, tree):
    n = int(min(len(R), max(1, round(s_first)) + 1)); seg = R[:n]
    if s_first < 2.0:
        return dict(passed=True, flag="node_link_short", median_dist_mm=0.0, share_within_5mm=1.0, link_mm=s_first)
    d = tree.query(seg)[0]
    return dict(passed=bool(np.median(d) <= GATE_MED and (d <= GATE_IN).mean() >= GATE_SHARE), flag=None, median_dist_mm=float(np.median(d)), share_within_5mm=float((d <= GATE_IN).mean()), link_mm=s_first)


def build_plant(spec_u, clicks, X):
    """spec_u: adopted U spec (fitted stem + nodes). Returns (spec_g, rows)."""
    tree = cKDTree(X); out = copy.deepcopy(spec_u); rows = []; node_xyz = {n["k"]: np.asarray(n["xyz_mm"], float) for n in spec_u["nodes"]}
    for lf in out["leaves"]:
        uid = lf["uid"]; click = clicks.get(uid, np.zeros((0, 3))); rec = dict(plant=spec_u["plant_id"], k=lf["k"], uid=uid, azimuth_U=lf["azimuth_deg"], theta0_U=lf["theta0_deg"], length_U=lf["length_mm"], flags=[])
        if len(click) < 2 or lf.get("geometry") != "observed":
            rec.update(status="unchanged_one_click"); rows.append(rec); continue
        m, _ = W.pairs_of(lf["leaflets"]); R, s_first = click_curve(node_xyz[lf["node_k"]], click); g = node_link_gate(R, s_first, tree)
        rec.update(link_mm=g["link_mm"], link_median_mm=g["median_dist_mm"], link_share5=g["share_within_5mm"], link_gate=g["passed"])
        if g["flag"]:
            rec["flags"].append(g["flag"])
        if not g["passed"]:
            rec.update(status="node_link_outside_cloud_kept_U"); rec["flags"].append("node_link_outside_cloud_kept_U"); lf["clicks_status"] = "node_link_outside_cloud_kept_U"; rows.append(rec); continue
        pl, L = writer_points(R, m); v = pl[1] - pl[0]; az = math.degrees(math.atan2(v[1], v[0])) % 360; th = math.degrees(math.asin(np.clip(v[2] / np.linalg.norm(v), -1, 1)))
        d = np.diff(pl, axis=0); el = [math.degrees(math.asin(np.clip(x[2] / np.linalg.norm(x), -1, 1))) for x in d]; dth = float(el[0] - el[-1])
        dmed = float(np.median(tree.query(FL.resample(pl, 1.0)[0])[0]))
        lf.update(azimuth_U_deg=lf["azimuth_deg"], theta0_U_deg=lf["theta0_deg"], dtheta_U_deg=lf["dtheta_deg"], length_U_mm=lf["length_mm"], rachis_polyline_U_mm=lf.get("rachis_polyline_level2_mm"),
                  azimuth_deg=az, theta0_deg=th, dtheta_deg=dth, length_mm=L, rachis_polyline_level2_mm=pl.tolist(), rachis_curve_from_clicks_mm=R[::2].tolist(),
                  azimuth_source=f"{SRC} (petiole piece)", theta0_source=f"{SRC} (petiole piece, node -> 0.30 L)", dtheta_source=f"{SRC} (elevation first - last piece)", length_source=f"{SRC} (node -> last click, no extension)",
                  clicks_status="rachis_from_clicks", flags=list(lf.get("flags", [])) + ["L_to_last_click"])
        rec.update(status="rachis_from_clicks", azimuth_G=az, theta0_G=th, dtheta_G=dth, length_G=L, curve_median_dist_mm=dmed); rec["flags"].append("L_to_last_click"); rows.append(rec)
    out["labels"] = list(dict.fromkeys([x for x in out["labels"] if x != "user_accepted_despite_synthetic_fail"] + ["rachis_from_clicks", "stem_fitted_to_surfels_user_accepted"]))
    out["fit"]["clicks"] = dict(contract="CONTRACT G", sigma_mm=SIGMA, note="rachis = node + clicked line, Gaussian smoothed, no extension beyond the last click; stem and nodes from the adopted fit U")
    return out, rows
