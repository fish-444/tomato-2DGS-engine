"""Amendment 3: B traits from the user's skeleton lines (ungraded). One run; rules fixed in CONTRACT_AMENDMENT_3.md."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R))
from tomato3d.traits import recon_traits as rt

D = Path(__file__).resolve().parent
REC = json.loads((D / "skeleton_inputs_amendment3.json").read_text(encoding="utf-8"))
SKIP = {("n_01", 17), ("n_02", 17), ("e_07", 5)}
LAB = "skeleton_labels_ungraded;recon_trait;conditional_on_unvalidated_scale;after_checkpoint_v1;age_dag=60;user_stated"
rows, plants = [], []
for p, v in REC.items():
    z = np.load(R / v["file"], allow_pickle=True); g = np.load(R / v["groups"], allow_pickle=True)
    assert (z["rows"] == g["rows"]).all()
    xyz = g["xyz"] * 1000.0; ax = g["axis_like"].astype(bool); lab = z["label"]; st = z["status"]
    c = np.median(xyz[ax, :2], axis=0)
    leaves, first_is_base = [], 0
    for b, s in enumerate(z["branches"], start=1):
        P = np.array(json.loads(str(s)), float)
        if (p, b) in SKIP:
            rows.append(dict(plant=p, branch=b, status="leaf_missing_or_merged")); continue
        if len(P) < 2:
            rows.append(dict(plant=p, branch=b, status="skipped_lt2_points")); continue
        d0, d1 = np.linalg.norm(P[0, :2] - c), np.linalg.norm(P[-1, :2] - c)
        if d1 < d0: P = P[::-1]
        else: first_is_base += 1
        if rt.arclen(P)[-1] < 25:
            rows.append(dict(plant=p, branch=b, status="skipped_L_lt_25mm", L_raw_mm=float(rt.arclen(P)[-1]))); continue
        pts = xyz[(lab == b) & (st == 1) & ax]
        Q = rt.densify(P, pts)
        a = rt.arch_traits(Q)
        c0 = min(rt.CHORD_MM, a["L_mm"] / 2); _, dvec = rt.chord_elev(Q, 0, c0)
        ins = float(np.degrees(np.arccos(np.clip(dvec[2], -1, 1))))
        psi = float(np.degrees(np.arctan2(dvec[1], dvec[0])) % 360)
        r = dict(plant=p, branch=b, status="ok", labels=LAB + ";display_only", base_z_mm=float(Q[0, 2]), n_clicks=len(P), n_rachis_pts=int(len(pts)),
                 L_mm=a["L_mm"], theta0_deg=a["theta0_deg"], theta_tip_deg=a["theta_tip_deg"], dtheta_deg=a["dtheta_deg"], curv_deg_per_mm=a["curv_deg_per_mm"],
                 s_apex_rel=a["s_apex_rel"], tip_drop_mm=a["tip_drop_mm"], insertion_deg_axis_proxy=ins, base_azimuth_deg=psi,
                 arc_fit_rms_mm=a["arc"]["fit_rms_mm"], quad_fit_rms_mm=a["quad"]["fit_rms_mm"], arc_direct_tip_err_mm=a["arc"]["direct_tip_err_mm"],
                 quad_direct_tip_err_mm=a["quad"]["direct_tip_err_mm"], better=a["better"])
        rows.append(r); leaves.append(r)
    leaves.sort(key=lambda r: r["base_z_mm"])
    for k, r in enumerate(leaves, start=1): r["rank_from_base_height"] = k
    dv = rt.divergence([r["base_azimuth_deg"] for r in leaves])
    inter = np.diff([r["base_z_mm"] for r in leaves])
    L = np.array([r["L_mm"] for r in leaves]); th0 = np.array([r["theta0_deg"] for r in leaves]); dth = np.array([r["dtheta_deg"] for r in leaves])
    plants.append(dict(plant=p, labels=LAB, file=v["file"].split("/")[-1], n_branches=v["branches"], n_leaves_used=len(leaves), first_click_is_base=first_is_base,
                       theta0_mean_deg=th0.mean(), theta0_sd_deg=th0.std(ddof=1), theta_tip_mean_deg=np.mean([r["theta_tip_deg"] for r in leaves]),
                       dtheta_mean_deg=dth.mean(), dtheta_sd_deg=dth.std(ddof=1), curv_mean_deg_per_mm=np.mean([r["curv_deg_per_mm"] for r in leaves]),
                       L_mean_mm=L.mean(), L_sd_mm=L.std(ddof=1), tip_drop_mean_mm=np.mean([r["tip_drop_mm"] for r in leaves]),
                       s_apex_rel_mean=np.mean([r["s_apex_rel"] for r in leaves]), arc_better_share=np.mean([r["better"] == "arc" for r in leaves]),
                       insertion_mean_deg_axis_proxy=np.mean([r["insertion_deg_axis_proxy"] for r in leaves]),
                       divergence_circ_mean_deg_axis_proxy=dv.get("circ_mean_deg"), divergence_circ_sd_deg_axis_proxy=dv.get("circ_sd_deg"), handedness=dv.get("handedness"),
                       internode_mean_mm_axis_proxy=float(inter.mean()) if len(inter) else None, internode_median_mm_axis_proxy=float(np.median(inter)) if len(inter) else None,
                       flags="axis_proxy_untested_s3 (divergence, insertion, internode); rank_from_base_height"))
pd.DataFrame(rows).to_csv(D / "skeleton_traits_per_leaf_display_only.csv", index=False)
P = pd.DataFrame(plants); P.to_csv(D / "skeleton_traits_per_plant.csv", index=False)
pd.set_option("display.width", 250)
print(P.drop(columns=["labels", "file", "flags"]).round(2).to_string())
print(pd.DataFrame(rows).status.value_counts().to_dict())
