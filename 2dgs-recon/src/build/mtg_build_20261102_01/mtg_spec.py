"""MTG spec = the one intermediate description both writers read (CONTRACT §3). Three producers:
observed_spec(plant)  - skeleton snapshot + labels (positions, azimuths, lengths observed; theta0 hand by rank; dtheta template);
template_mean_spec()  - per-rank means only, no randomness;
drawn_spec(seed, n)   - keyed draws from the dynamic rules (check plants).
A spec is a plain dict: plant_id, labels, axis (curve_mm, base_mm, source), nodes [k, s_mm, xyz_mm, internode_mm, internode_source,
stem_width_mm, width_source], leaves [k, uid, node_k, base_mm, azimuth_deg, theta0_deg, dtheta_deg, length_mm, leaflets, area_mm2, *_source, flags,
geometry], side_axes [j, attach_k, azimuth_deg, angle_above_h_deg, internodes_mm, internode_source, leaves[...]], notes."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from tools import bmodel_mtg_export as BX  # noqa: E402
from template_rows import Template, INTERNODE_SETTING  # noqa: E402

LABELS = ["mtg_from_traits", "conditional_on_unvalidated_scale", "v2_ungraded_draft_not_frozen", "after_checkpoint_v1"]
DZ = ROOT / "build/p1_dz10_labels_20261101_01"
SNAP = {"n_01": "5e803db0e8f8", "n_02": "6e44394dd4d8", "n_03": "d267404f6ada", "n_04": "a40266c80d19", "n_05": "bcf8d6cef69a", "n_06": "64b0f9b553e9",
        "e_01": "26f27427b73b", "e_04": "73fdd1ffeead", "e_05": "524f23114149", "e_07": "16690f4f8d25"}
GRADE = {"n_01": "B", "n_02": "C", "n_03": "B", "n_04": "B", "n_05": "A", "n_06": "ungraded", "e_01": "B", "e_04": "A", "e_05": "A", "e_07": "C", "R3-1b": "n/a"}
ROUTE = {p: ("distance" if p == "n_06" else "tprox") for p in SNAP}
PLANTS = ["R3-1b"] + list(SNAP)
PUB_1_50 = ROOT / "build/bmodel_mtg_20261101_01/upstream/model/input/static_input/1_50.csv"
TPL = Template()


# ------------------------------------------------------------------ observed plants
def _areas(pid):
    if pid == "R3-1b":
        return None, "R8_pc blade surfels area_b_mm2 (per_surfel_D.npz); shared lines split by path length"
    v = SNAP[pid]; lab = np.load(DZ / f"labels_{ROUTE[pid]}_{pid}_{v}.npz"); uids = [str(u) for u in lab["leaf_uids"]]
    if not pid.startswith("n_"):
        return {u: None for u in uids}, "unavailable: no per-surfel silhouette area for e_* (static-nine area kernel not run for e_*)"
    z = np.load(ROOT / f"build/engine_static_nine_20261028_01/kaggle_out/{pid}/per_surfel_light.npz")
    assert np.array_equal(z["ply_row"], lab["rows"]), f"{pid}: label rows differ from the area rows"
    A = {u: float(z["area_b_silhouette_mm2"][(lab["label"] == k + 1) & (lab["status"] == 1)].sum()) for k, u in enumerate(uids)}
    return A, f"{ROUTE[pid]} route labels skelver {v} (assigned surfels only, lower bound) x area_b_silhouette_mm2 (static nine)"


def _r31b_areas(skf):
    sk = json.loads(skf.read_text(encoding="utf-8")); z = np.load(ROOT / "build/engine_coplanar_sep_20261101_01/out_k1/per_surfel_D.npz")
    per = {int(k): float(z["area_b_mm2"][z["leaf_R8pc"] == k].sum()) for k in np.unique(z["leaf_R8pc"]) if k > 0}
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost")]; by = {}
    for l in live:
        by.setdefault(int(l["attrs"]["pipeline_leaf"]), []).append(l)
    areas, shared = {}, set()
    for k, ls in by.items():
        L = np.array([BX.path_len(np.asarray(l["polyline_xyz"]) * 1000) for l in ls]); w = L / L.sum() if L.sum() > 0 else np.full(len(ls), 1 / len(ls))
        for l, wi in zip(ls, w):
            areas[l["leaf_uid"]] = per.get(k, 0.0) * float(wi)
            if len(ls) > 1:
                shared.add(l["leaf_uid"])
    return areas, shared


def observed_spec(pid):
    if pid == "R3-1b":
        skf = ROOT / "data/skeletons/R3-1b/2026-10-21/skeleton_v1.json"; areas, shared = _r31b_areas(skf); asrc = _areas(pid)[1]
    else:
        skf = DZ / f"{pid}_{SNAP[pid]}/skeleton_v1.json"; areas, asrc = _areas(pid); shared = set()
    obs = BX.load_observed(pid, skf, {k: v for k, v in areas.items() if v is not None}, asrc, age_dag=60)     # A1 node rule, first-10-mm direction, one-click completion
    dth, dth_sd, dth_lab, dth_note = TPL.dtheta()
    spec = dict(plant_id=pid, kind="observed", labels=LABELS + ["skeleton_labels_ungraded", f"grade_{GRADE[pid]}", "axis_" + obs["axis_source"].split(" ")[0]],
                skeleton_file=str(skf.relative_to(ROOT)).replace("\\", "/"), skeleton_sha256=obs["skeleton_sha256"], axis=dict(curve_mm=obs["axis_curve_mm"], base_mm=obs["stem_base_mm"], source=obs["axis_source"]),
                height_mm=obs["height_mm"], age_dag=60, age_source="user_stated", area_source=asrc, flags=list(obs["flags"]), nodes=[], leaves=[], side_axes=[], trusses=[], notes=[])
    for nd, lf in zip(obs["nodes"], obs["leaves"]):
        k = lf["model_rank"]; th, th_sd, th_lab, _ = TPL.theta0(k); w, w_sd, w_lab = TPL.stem_thickness(k)
        spec["nodes"].append(dict(k=k, s_mm=nd["s_mm"], xyz_mm=nd["xyz_mm"], internode_mm=nd["internode_mm"], internode_source=f"skeleton_axis_{nd['internode_provenance']}", stem_width_mm=w, width_source=w_lab))
        fl = list(lf["flags"]) + (["area_shared_split"] if lf["leaf_uid"] in shared else []) + (["theta0_from_hand_template_not_skeleton_elevation"])
        spec["leaves"].append(dict(k=k, uid=lf["leaf_uid"], node_k=k, base_mm=lf["base_mm"], azimuth_deg=lf["azimuth_deg"], azimuth_source=f"skeleton_first10mm_{lf['geometry']}",
                                   skeleton_elevation_deg=lf["elevation_deg"], theta0_deg=th, theta0_source=th_lab, dtheta_deg=dth, dtheta_source=dth_lab, length_mm=lf["length_mm"],
                                   length_source=f"skeleton_path_{lf['geometry']}", leaflets=TPL.leaflets_mode(k), leaflets_source="hand_measured_mode_by_rank",
                                   area_mm2=areas.get(lf["leaf_uid"]), area_source=("recon_area_b" if areas.get(lf["leaf_uid"]) is not None else "unavailable"), flags=fl, geometry=lf["geometry"]))
    sk = json.loads(skf.read_text(encoding="utf-8"))
    lat = [a for a in sk.get("axes", []) if a.get("type") == "lateral" and a.get("status") == "active" and a.get("fit")]
    if lat:
        spec["notes"].append(f"{len(lat)} lateral axis/axes in the skeleton not converted (no leaf assignment to laterals in skeleton_v1.1; listed, CONTRACT §3)")
    spec["notes"].append("dtheta note from the draft: " + dth_note)
    return spec


# ------------------------------------------------------------------ pooled areas (template-mean plant)
def pooled_areas():
    """Mean area (b) by model rank over graded A/B n_* plants (CONTRACT §3)."""
    pool = {}
    for pid in SNAP:
        if GRADE[pid] not in ("A", "B") or not pid.startswith("n_"):
            continue
        s = observed_spec(pid)
        for lf in s["leaves"]:
            if lf["area_mm2"] is not None:
                pool.setdefault(lf["k"], []).append(lf["area_mm2"])
    return {k: dict(mean=float(np.mean(v)), n=len(v), plants="n_01,n_03,n_04,n_05") for k, v in sorted(pool.items())}


def pooled_area(pool, k):
    ks = [x for x in pool if x <= k] or [min(pool)]; kk = max(ks)
    return pool[kk]["mean"], f"recon_pool_mean_area_by_rank(rank {kk}, n={pool[kk]['n']})" + ("" if kk == k else "; nearest_lower_rank")


def _branch_internodes():
    bi, bl = BX.published_branch_rows(PUB_1_50)
    return {k: float(bi[k][14]) * 1000 for k in sorted(bi)}, {k: float(bl[k][17]) * 1e6 for k in sorted(bl)}


# ------------------------------------------------------------------ template-mean plant
def template_mean_spec():
    n_main = int(round(TPL.main_leaves()[0])); dth, dth_sd, dth_lab, _ = TPL.dtheta(); L, L_sd, L_lab = TPL.leaf_length(); lo, hi, dlab = TPL.divergence_range(); D = (lo + hi) / 2
    pool = pooled_areas(); side = TPL.side(); bi, _ = _branch_internodes()
    spec = dict(plant_id="template_mean_v2draft", kind="template_mean", labels=LABELS + ["template_mean_plant", "no_randomness"], axis=dict(source="vertical, internode = median 3.75 mm (user_set_range)"),
                age_dag=60, age_source="user_stated", flags=[], nodes=[], leaves=[], side_axes=[], trusses=[], notes=[], pooled_areas=pool,
                divergence_deg=D, divergence_source=f"{dlab}_midpoint({lo}-{hi}); not a measurement", internode_mm=INTERNODE_SETTING["median"], internode_source=INTERNODE_SETTING["label"] + "(median)")
    z = 0.0
    for k in range(1, n_main + 1):
        z += INTERNODE_SETTING["median"]; th, th_sd, th_lab, _ = TPL.theta0(k); w, w_sd, w_lab = TPL.stem_thickness(k); a, a_lab = pooled_area(pool, k)
        spec["nodes"].append(dict(k=k, s_mm=z, xyz_mm=[0.0, 0.0, z], internode_mm=INTERNODE_SETTING["median"], internode_source=INTERNODE_SETTING["label"], stem_width_mm=w, width_source=w_lab))
        spec["leaves"].append(dict(k=k, uid=f"T-L{k:02d}", node_k=k, base_mm=[0.0, 0.0, z], azimuth_deg=((k - 1) * D) % 360, azimuth_source=spec["divergence_source"], theta0_deg=th, theta0_sd_deg=th_sd,
                                   theta0_source=th_lab, dtheta_deg=dth, dtheta_source=dth_lab, length_mm=L, length_source=L_lab + "(plant-mean row; no per-rank value)", leaflets=TPL.leaflets_mode(k),
                                   leaflets_source="hand_measured_mode_by_rank", area_mm2=a, area_source=a_lab, flags=[], geometry="template"))
    n_ax = int(round(side["axes_per_plant"])); cnt = side["attach_counts"]
    ranks = sorted(sorted((k for k in cnt if k <= n_main), key=lambda k: (-cnt[k], k))[:n_ax])
    n_sl = int(round(side["n_side_leaves"] / side["n_axes_items"])); ang = float(np.mean(side["angles"])); sl_leaflets = max(side["shares"], key=side["shares"].get)
    a_side = float(np.mean([v["mean"] for v in pool.values()]))
    for j, r in enumerate(ranks, 1):
        main_leaf = spec["leaves"][r - 1]
        spec["side_axes"].append(dict(j=j, attach_k=r, attach_source="hand_measured_most_frequent_attach_ranks", azimuth_deg=main_leaf["azimuth_deg"], azimuth_source="attached_main_leaf_azimuth(B-model session rule)",
                                      angle_above_h_deg=ang, angle_source="hand_measured_sparse(mean of 2 values 20,60)", internodes_mm=[bi.get(i, bi[max(bi)]) for i in range(1, n_sl + 1)],
                                      internode_source="model_published_1_50_branch_rows(kept; addendum §1)",
                                      leaves=[dict(i=i, theta0_deg=side["theta0"], theta0_sd_deg=side["theta0_sd"], theta0_source="hand_measured_side", dtheta_deg=dth, dtheta_source=dth_lab, length_mm=L,
                                                   length_source=L_lab + "(main-leaf row reused for side leaves; flagged)", leaflets=sl_leaflets, leaflets_source="hand_measured_side_mode",
                                                   area_mm2=a_side, area_source="recon_pool_mean_area_all_ranks(side leaves have no area source; flagged)", flags=["side_leaf_size_reused"]) for i in range(1, n_sl + 1)]))
    spec["notes"] += ["trusses: 1.5/plant, 3.56 flowers/truss in the hand data, node position not confirmed -> not written (pending_user)", "fruits not used (pending_user)",
                      f"side axes: {n_ax} at ranks {ranks}, {n_sl} leaves each (18/12 rounded), leaves per axis distribution pending_traits"]
    return spec


if __name__ == "__main__":
    s = template_mean_spec(); print(json.dumps({k: v for k, v in s.items() if k not in ("leaves", "nodes")}, indent=1, default=str)[:1500])
    for p in PLANTS:
        o = observed_spec(p); print(p, len(o["leaves"]), "leaves", o["axis"]["source"][:30], "areas", sum(l["area_mm2"] is not None for l in o["leaves"]), o["flags"])
