"""Dynamic rule file (CONTRACT §4), internode Beta setting (addendum 16:22 §1), keyed draws, check plants (16 seeds) and their distribution check.
Draw = jrandom.keyed(seed, id, key): u -> uniform / categorical / Beta ppf; z -> normal (clipped to the truncation range, flag clipped)."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta as Beta

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from jrandom import keyed  # noqa: E402
from template_rows import Template, INTERNODE_SETTING  # noqa: E402
from mtg_spec import LABELS, pooled_areas, pooled_area, _branch_internodes  # noqa: E402

TPL = Template(); OUT = HERE / "dynamic"; OUT.mkdir(exist_ok=True)
SEED0 = 20261102; N_CHECK = 16
THETA_TRUNC = (-5.0, 80.0)      # Engine v2 C2 declaration (measured range)


# ------------------------------------------------------------------ internode Beta
def internode_beta():
    c = INTERNODE_SETTING["concentration"]; lo, hi = INTERNODE_SETTING["lo"], INTERNODE_SETTING["hi"]; med_x = (INTERNODE_SETTING["median"] - lo) / (hi - lo)
    a = brentq(lambda a: Beta.median(a, c - a) - med_x, 1e-3, c - 1e-3); b = c - a
    mean = lo + (hi - lo) * a / c; var = (hi - lo) ** 2 * a * b / (c * c * (c + 1)); p5, p95 = (lo + (hi - lo) * Beta.ppf(q, a, b) for q in (0.05, 0.95))
    return dict(a=a, b=b, concentration=c, lo_mm=lo, hi_mm=hi, median_mm=lo + (hi - lo) * Beta.median(a, b), target_median_mm=INTERNODE_SETTING["median"], mean_mm=mean, sd_mm=math.sqrt(var), p5_mm=p5, p95_mm=p95,
                draw_unit="per internode (key RP_INTERNODE_MM, id P<n>/I<k>)", label=INTERNODE_SETTING["label"], method="scaled Beta on [2,4], a+b=6 fixed, a solved by brentq for median 0.875 of x=(L-2)/2")


def draw_internode(seed, id_, ib):
    u = keyed(seed, id_, "INTERNODE_MM").next_double(); return ib["lo_mm"] + (ib["hi_mm"] - ib["lo_mm"]) * float(Beta.ppf(u, ib["a"], ib["b"])), u


# ------------------------------------------------------------------ rule file
def rules():
    ib = internode_beta(); dth, dth_sd, dth_lab, _ = TPL.dtheta(); L, L_sd, L_lab = TPL.leaf_length(); lo, hi, dlab = TPL.divergence_range(); side = TPL.side(); tr = TPL.trusses(); pool = pooled_areas()
    R = []
    def add(organ, rank, field, kind, mean="", sd="", lo_="", hi_="", params="", shares="", label="", key="", unit="", engine="", bcol="", note=""):
        R.append(dict(organ=organ, rank_class=rank, field=field, distribution=kind, mean=mean, sd=sd, lo=lo_, hi=hi_, params=params, shares=shares, source_label=label, draw_key=key, draw_unit=unit,
                      engine_param=engine, bmodel_column=bcol, note=note))
    nl = TPL.main_leaves()
    add("main_stem", "plant", "main_leaves_count", "normal_rounded", nl[0], nl[1], 3, "", "", "", "hand_measured", "RP_MAIN_LEAVES", "plant", "FIRST_LEAVES (true leaves = engine + 1, U5)", "number of I/xL rows", "round(normal), min 3; dt06 had 3")
    add("main_stem", "internode", "internode_length_mm", "beta_scaled", ib["mean_mm"], ib["sd_mm"], ib["lo_mm"], ib["hi_mm"], json.dumps(dict(a=ib["a"], b=ib["b"], median=ib["target_median_mm"])), "", ib["label"], "RP_INTERNODE_MM", "internode",
        "no length input (INTERNODE_CAP_M is a cap of funded growth, per plant RP_INTERNODE_CAP_M) -> new per-internode key requested for Engine v2; static: structure files", "I row col 14 (m, static); dynamic import mass-based col 17", "user setting 16:22; not measured")
    add("main_compound_leaf", "leaf", "divergence_deg", "uniform", (lo + hi) / 2, "", lo, hi, "", "", dlab + ";literature_wild_species", "RP_DIVERGENCE_DEG (existing) x HANDEDNESS (existing, plant)", "leaf", "DIVERGENCE_DEG cumulative azimuth (rpDivergenceLo/Hi)", "xL col 10 orientation = az + 90", "handedness fixed per plant; template-mean uses 137.5 midpoint")
    for r in range(1, 10):
        th, sd, lab, row = TPL.theta0(r)
        add("main_compound_leaf", f"rank{r}", "theta0_above_horizontal_deg", "normal_truncated", th, sd, THETA_TRUNC[0], THETA_TRUNC[1], "", "", "hand_measured", "RP_THETA0_DEG", "organ", "COMPOUND_INSERTION_DEG = 90 - theta0 (rpInsertionLo/Hi replaced by rank rule)", "xL col 12 inclination = 90 - theta0", f"n_plants {row['n_plants']}, n {row['n_items']}")
        sh = TPL.leaflet_shares(r)
        add("main_compound_leaf", f"rank{r}", "leaflets_per_leaf", "categorical", "", "", "", "", "", json.dumps(sh), "hand_measured", "RP_LEAFLET_COUNT", "organ", "PAIRS -> pairs = (n-1)//2, 6 = 2 pairs + 1 intercalary (new rank rule replaces RP_PAIRS choice)", "PROPORTION_LEAFLETS_* by rank class (per-leaf count not representable)", "")
        w = TPL.row("main_stem", r, "stem_thickness")
        if w:
            add("main_stem", f"rank{r}", "stem_thickness_mm", "normal_truncated" if float(w["sd"] or 0) > 0 else "fixed", float(w["value"]), float(w["sd"] or 0), 1.0, "", "", "", "hand_measured", "RP_STEM_THICKNESS_MM", "organ", "STEM_WIDTH_M (engine: fixed per plant)", "I row col 16 (m)", f"n {w['n_items']}")
    th, sd, lab, row = TPL.theta0("all")
    add("main_compound_leaf", "rank>9", "theta0_above_horizontal_deg", "normal_truncated", th, sd, *THETA_TRUNC, "", "", "hand_measured_all_ranks", "RP_THETA0_DEG", "organ", "", "", "fallback for ranks without a hand row")
    add("main_compound_leaf", "rank>9", "leaflets_per_leaf", "categorical", "", "", "", "", "", json.dumps(TPL.leaflet_shares("all")), "hand_measured_all_ranks", "RP_LEAFLET_COUNT", "organ", "", "", "fallback")
    add("main_stem", "rank>7", "stem_thickness_mm", "fixed", TPL.stem_thickness(7)[0], "", "", "", "", "", "hand_measured_rank7_reused", "RP_STEM_THICKNESS_MM", "organ", "", "", "fallback")
    add("main_compound_leaf", "all", "dtheta_total_bend_deg", "normal_truncated", dth, dth_sd, 0.0, 90.0, "", "", dth_lab, "RP_TOTAL_BEND_DEG (existing key; bounds replace U(0,40) user_assumed)", "organ", "TOTAL_BEND_DEG (progressive weights 1/(2m-1), 2/(2m-1))", "not_representable (no curvature column)", "sd = between-plant sd; first click on leaflets -> may be underestimated (draft note)")
    add("main_compound_leaf", "all", "leaf_length_mm", "normal_truncated", L, L_sd, 10.0, "", "", "", L_lab, "RP_LEAF_LENGTH_MM", "organ", "PETIOLE_M + RACHIS_M (engine grows length from mass: static structure only; dynamic -> Engine v2 target)", "xL col 14 petiole + col 30 rachis dw (converter split)", "plant-mean row; per-rank length pending_traits")
    for k, v in pool.items():
        add("main_compound_leaf", f"rank{k}", "blade_area_mm2", "check_target_only", v["mean"], "", "", "", "", "", "recon_pool_mean_area_by_rank;conditional_on_unvalidated_scale", "", "", "not drawn (engine grows area)", "xL col 17 (static)", f"n {v['n']} plants {v['plants']}")
    add("side_axis", "plant", "side_axes_per_plant", "fixed", side["axes_per_plant"], "", "", "", "", "", "hand_measured", "", "plant", "no side-axis rule in the engine (Engine v2 C4)", "branch rows (order 1)", "6 plants, all 2")
    tot = sum(side["attach_counts"].values())
    add("side_axis", "axis", "attach_main_rank", "categorical", "", "", "", "", "", json.dumps({str(k): v / tot for k, v in sorted(side["attach_counts"].items())}), "hand_measured", "RP_SIDE_ATTACH_RANK", "organ", "", "xI row after its main rank", "2 attach_node_unknown not in the shares; drawn without replacement per plant")
    add("side_axis", "axis", "leaves_per_side_axis", "pending_traits", side["n_side_leaves"] / side["n_axes_items"], "", 1, side["leaves_max"], "", "", "pending_traits(hand: mean 1.5, max 4, shape unknown)", "RP_SIDE_LEAVES", "organ", "", "", "check plants use fixed 2 (round(1.5)), flagged")
    add("side_axis", "axis", "angle_above_horizontal_deg", "fixed", float(np.mean(side["angles"])), "", "", "", "", "", "hand_measured_sparse(2 values)", "", "", "", "xI col 12 = 90 - angle", "")
    add("side_axis", "internode", "internode_length_mm", "fixed_published", json.dumps(_branch_internodes()[0]), "", "", "", "", "", "model_published_1_50_branch_rows(kept)", "", "", "engine: none (side axes not generated)", "xI/I col 14", "addendum §1: existing values stay")
    add("side_axis_compound_leaf", "all", "theta0_above_horizontal_deg", "normal_truncated", side["theta0"], side["theta0_sd"], *THETA_TRUNC, "", "", "hand_measured", "RP_SIDE_THETA0_DEG", "organ", "", "branch xL col 12", "")
    add("side_axis_compound_leaf", "all", "leaflets_per_leaf", "categorical", "", "", "", "", "", json.dumps(side["shares"]), "hand_measured", "RP_SIDE_LEAFLET_COUNT", "organ", "", "", "")
    add("truss", "plant", "trusses_per_plant", "categorical", tr["trusses_mean"], "", "", "", "", json.dumps(truss_shares()), "hand_measured(hand_trusses.csv per plant)", "RP_TRUSSES_PER_PLANT", "plant", "CADENCE / truss rule (engine)", "none in static import", "truss node position pending_user -> not placed")
    add("truss", "truss", "flowers_per_truss", "normal_rounded", tr["flowers_mean"], tr["flowers_sd"], 1, "", "", "", "hand_measured", "RP_FLOWERS_PER_TRUSS (replaces FLOWERS U{4..8})", "organ", "flowers_per_truss hook", "FRUIT_PER_TRUSS", "")
    add("fruit", "plant", "fruits_per_plant", "not_used", 3.5, "", "", "", "", "", "pending_user", "", "", "", "", "main prompt §2: not used")
    add("leaflet", "all", "curl", "engine_default", "", "", "", "", "", "", "engine_default", "", "", "leaflet outline hook / flat diamonds", "CURVATURE_LEAFLET published 12", "failed synthetic check")
    rs = TPL.recon_size()
    for k, (m, s) in rs.items():
        add("plant", "all", f"recon_{k}_mm", "check_target_only", m, s, "", "", "", "", "recon_measured;conditional_on_unvalidated_scale", "", "", "", "", "7 plants R3-1b, n_01-n_06")
    add("plant", "all", "age_dag", "fixed", 60, "", "", "", "", "", "user_stated", "", "", "", "", "")
    with open(OUT / "mtg_dynamic_rules_v1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(R[0])); w.writeheader(); w.writerows(R)
    with open(OUT / "internode_range_setting.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ib)); w.writeheader(); w.writerow(ib)
    # additions to the random-parameter table (same columns as tools/microtom_parameters.csv; tools/ not edited)
    cols = ["key", "value", "unit", "source_kind", "source_ref", "cultivar", "uncertainty", "uncertainty_kind", "candidates", "engine_key", "engine_status", "note"]
    adds = []
    def A(key, value, unit, kind, ref, unc, ukind, ekey, note):
        adds.append(dict(key=key, value=value, unit=unit, source_kind=kind, source_ref=ref, cultivar="Micro-Tom (our plants, MTG-building data)", uncertainty=unc, uncertainty_kind=ukind, candidates="", engine_key=ekey, engine_status="requested_engine_v2", note=note))
    A("RP_INTERNODE_MM", json.dumps([ib["lo_mm"], ib["hi_mm"]]), "mm", "user_stated", "CLI_ADDENDUM_INTERNODE_RANGE_2_TO_4_MEDIAN_375_2026-11-02 (user 16:22)", f"scaled Beta a={ib['a']:.4f} b={ib['b']:.4f} median 3.75", "bounded_random", "new (per internode)", ib["label"] + "; not measured")
    A("RP_THETA0_DEG", "by rank (mtg_dynamic_rules_v1.csv)", "deg", "user_stated", "hand_leaf_measurements_t1_dt01-06 via MTG_TEMPLATE_v2 draft", "normal(mean_rank, sd_rank) truncated [-5, 80]", "bounded_random", "COMPOUND_INSERTION_DEG = 90 - theta0", "replaces RP_COMPOUND_INSERTION_DEG U(60,90) when adopted")
    A("RP_TOTAL_BEND_DEG", json.dumps([0.0, 90.0]), "deg", "user_stated", "MTG_TEMPLATE_v2 draft skeleton_dtheta_total_bend 32.764 +- 10.602 (skeleton_traits_ungraded)", "normal truncated", "bounded_random", "TOTAL_BEND_DEG", "bounds change from U(0,40) user_assumed; ungraded skeletons")
    A("RP_LEAF_LENGTH_MM", json.dumps([10.0, 200.0]), "mm", "user_stated", "MTG_TEMPLATE_v2 draft skeleton_leaf_length_L 43.745 +- 4.86", "normal truncated", "bounded_random", "new (static structure; dynamic target)", "skeleton_traits_ungraded")
    A("RP_LEAFLET_COUNT", "shares by rank (rule file)", "count", "user_stated", "hand data p_leaflets_5/6/7 by rank", "categorical", "bounded_random", "PAIRS (rank rule)", "replaces RP_PAIRS choice when adopted")
    A("RP_STEM_THICKNESS_MM", "by rank (rule file)", "mm", "user_stated", "hand data stem_thickness by rank", "normal truncated >= 1", "bounded_random", "STEM_WIDTH_M", "")
    A("RP_MAIN_LEAVES", json.dumps([3, 12]), "count", "user_stated", "hand data compound_leaves_per_main_stem 7.33 +- 2.34", "normal rounded min 3", "bounded_random", "FIRST_LEAVES", "U5: engine value = true leaves - 1")
    A("RP_SIDE_ATTACH_RANK", "shares (rule file)", "rank", "user_stated", "hand data attach_main_rank_counts", "categorical", "bounded_random", "new (no side-axis rule in the engine)", "")
    A("RP_SIDE_THETA0_DEG", json.dumps([-5.0, 80.0]), "deg", "user_stated", "hand data side_axis_compound_leaf theta0 20.56 +- 17.98", "normal truncated", "bounded_random", "new", "")
    A("RP_SIDE_LEAFLET_COUNT", json.dumps(side["shares"]), "count", "user_stated", "hand data side leaflets 5/7", "categorical", "bounded_random", "new", "")
    A("RP_SIDE_LEAVES", json.dumps([1, side["leaves_max"]]), "count", "unavailable", "hand data: mean 1.5, max 4, shape unknown", "pending_traits", "unavailable", "new", "check plants use fixed 2")
    A("RP_TRUSSES_PER_PLANT", json.dumps(truss_shares()), "count", "user_stated", "hand_trusses.csv", "categorical", "bounded_random", "CADENCE / truss rule", "node position pending_user")
    A("RP_FLOWERS_PER_TRUSS", json.dumps([1, 8]), "count", "user_stated", "hand data flowers_per_truss 3.56 +- 1.01", "normal rounded", "bounded_random", "flowers_per_truss hook", "replaces FLOWERS U{4..8} literature")
    with open(OUT / "random_parameter_table_additions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(adds)
    return R, ib


def truss_shares():
    rows = list(csv.DictReader(open(ROOT / "build/traits_recon_20261101_01/hand_trusses.csv", encoding="utf-8")))
    per = {p: 0 for p in ("dt01", "dt02", "dt03", "dt04", "dt05", "dt06")}
    for r in rows:
        per[r["plant"]] += int(float(r["n_truss"]))
    n = len(per); cnt = {}
    for v in per.values():
        cnt[v] = cnt.get(v, 0) + 1
    return {str(k): v / n for k, v in sorted(cnt.items())}


# ------------------------------------------------------------------ draws and check plants
DRAWS = []


def rec(pid, id_, key, kind, u, value, clipped=False):
    DRAWS.append(dict(plant=pid, id=id_, key=key, kind=kind, u_or_z=u, value=value, clipped=int(clipped)))


def d_normal(seed, pid, id_, key, mean, sd, lo=None, hi=None):
    z = keyed(seed, id_, key).next_gaussian(); v = mean + sd * z; c = False
    if lo is not None and v < lo:
        v, c = lo, True
    if hi is not None and v > hi:
        v, c = hi, True
    rec(pid, id_, key, "normal", z, v, c); return v


def d_uniform(seed, pid, id_, key, lo, hi):
    u = keyed(seed, id_, key).next_double(); v = lo + (hi - lo) * u; rec(pid, id_, key, "uniform", u, v); return v


def d_cat(seed, pid, id_, key, shares):
    u = keyed(seed, id_, key).next_double(); acc = 0.0; ks = sorted(shares, key=lambda k: float(k))
    for k in ks:
        acc += shares[k]
        if u < acc:
            rec(pid, id_, key, "categorical", u, k); return k
    rec(pid, id_, key, "categorical", u, ks[-1]); return ks[-1]


def drawn_spec(m, ib, internode_mode="beta"):
    """Check plant m (0-based): seed 20261102 + m, plant id P<m+1>. internode_mode 'beta' (rule) or 'fixed375' (old single value, paired comparison)."""
    seed = SEED0 + m; pid = f"P{m + 1:02d}"; tag = "" if internode_mode == "beta" else "_fixed375"
    dth, dth_sd, dth_lab, _ = TPL.dtheta(); L, L_sd, L_lab = TPL.leaf_length(); lo, hi, dlab = TPL.divergence_range(); side = TPL.side(); pool = pooled_areas(); bi, _ = _branch_internodes()
    nl = TPL.main_leaves(); n_main = max(3, int(round(d_normal(seed, pid, pid, "MAIN_LEAVES", nl[0], nl[1]))))
    hand_u = keyed(seed, pid, "HANDEDNESS").next_double(); hand = -1 if hand_u < 0.5 else 1; rec(pid, pid, "HANDEDNESS", "uniform", hand_u, hand)
    spec = dict(plant_id=pid + tag, kind="check_plant", seed=seed, labels=LABELS + ["generated_check_plant", f"internode_{internode_mode}"], axis=dict(source=f"vertical; internode {internode_mode}"), age_dag=60, age_source="user_stated",
                flags=[], nodes=[], leaves=[], side_axes=[], trusses=[], notes=[], handedness=hand)
    z = 0.0; az = 0.0
    for k in range(1, n_main + 1):
        il, u = draw_internode(seed, f"{pid}/I{k}", ib) if internode_mode == "beta" else (INTERNODE_SETTING["median"], None)
        if internode_mode == "beta":
            rec(pid, f"{pid}/I{k}", "INTERNODE_MM", "beta_scaled", u, il)
        z += il; lid = f"{pid}/L{k}"; rk = k if k <= 9 else "all"
        th_m, th_sd, th_lab, _ = TPL.theta0(rk); th = d_normal(seed, pid, lid, "THETA0_DEG", th_m, th_sd, *THETA_TRUNC)
        if k > 1:
            az = (az + hand * d_uniform(seed, pid, lid, "DIVERGENCE_DEG", lo, hi)) % 360
        n_lf = int(d_cat(seed, pid, lid, "LEAFLET_COUNT", TPL.leaflet_shares(rk)))
        dt = d_normal(seed, pid, lid, "TOTAL_BEND_DEG", dth, dth_sd, 0.0, 90.0); Lk = d_normal(seed, pid, lid, "LEAF_LENGTH_MM", L, L_sd, 10.0, None)
        w_m, w_sd, w_lab = TPL.stem_thickness(k); w = d_normal(seed, pid, f"{pid}/I{k}", "STEM_THICKNESS_MM", w_m, w_sd, 1.0, None) if w_sd > 0 else w_m
        a, a_lab = pooled_area(pool, k)
        spec["nodes"].append(dict(k=k, s_mm=z, xyz_mm=[0.0, 0.0, z], internode_mm=il, internode_source=ib["label"] if internode_mode == "beta" else "old_single_value_3.75_paired_comparison", stem_width_mm=w, width_source=w_lab + "_drawn"))
        spec["leaves"].append(dict(k=k, uid=f"{pid}-L{k:02d}", node_k=k, base_mm=[0.0, 0.0, z], azimuth_deg=az, azimuth_source=f"drawn {dlab} x handedness {hand}", theta0_deg=th, theta0_source=th_lab + "_drawn", dtheta_deg=dt, dtheta_source=dth_lab + "_drawn",
                                   length_mm=Lk, length_source=L_lab + "_drawn", leaflets=n_lf, leaflets_source="hand_measured_shares_drawn", area_mm2=a, area_source=a_lab + "(check target; not drawn)", flags=[], geometry="drawn"))
    # side axes: 2, attach ranks drawn without replacement from the shares (ranks <= n_main), 2 leaves each (pending_traits fixed)
    tot = sum(side["attach_counts"].values()); shares = {str(k): v / tot for k, v in side["attach_counts"].items() if k <= n_main}; s = sum(shares.values()); shares = {k: v / s for k, v in shares.items()}
    chosen = []
    for j in range(1, int(round(side["axes_per_plant"])) + 1):
        if not shares:
            break
        r = int(d_cat(seed, pid, f"{pid}/S{j}", "SIDE_ATTACH_RANK", shares)); chosen.append(r); shares = {k: v for k, v in shares.items() if int(k) != r}; s = sum(shares.values()) or 1; shares = {k: v / s for k, v in shares.items()}
        leaves = []
        for i in range(1, 3):
            lid = f"{pid}/S{j}/L{i}"; th = d_normal(seed, pid, lid, "SIDE_THETA0_DEG", side["theta0"], side["theta0_sd"], *THETA_TRUNC); n_lf = int(d_cat(seed, pid, lid, "SIDE_LEAFLET_COUNT", {str(k): v for k, v in side["shares"].items()}))
            dt = d_normal(seed, pid, lid, "TOTAL_BEND_DEG", dth, dth_sd, 0.0, 90.0); Lk = d_normal(seed, pid, lid, "LEAF_LENGTH_MM", L, L_sd, 10.0, None)
            leaves.append(dict(i=i, theta0_deg=th, theta0_source="hand_measured_side_drawn", dtheta_deg=dt, dtheta_source=dth_lab + "_drawn", length_mm=Lk, length_source=L_lab + "_drawn(main row reused)", leaflets=n_lf, leaflets_source="hand_measured_side_shares_drawn",
                               area_mm2=float(np.mean([v["mean"] for v in pool.values()])), area_source="recon_pool_mean_area_all_ranks(helper)", flags=["side_leaf_size_reused", "side_leaves_fixed_2_pending_traits"]))
        spec["side_axes"].append(dict(j=j, attach_k=r, attach_source="drawn RP_SIDE_ATTACH_RANK", azimuth_deg=spec["leaves"][r - 1]["azimuth_deg"], azimuth_source="attached main leaf", angle_above_h_deg=float(np.mean(side["angles"])), angle_source="hand_measured_sparse",
                                      internodes_mm=[bi.get(i, bi[max(bi)]) for i in range(1, 3)], internode_source="model_published_1_50_branch_rows(kept)", leaves=leaves))
    n_tr = int(d_cat(seed, pid, pid, "TRUSSES_PER_PLANT", truss_shares())); tr = TPL.trusses()
    spec["trusses"] = [dict(j=j, flowers=max(1, int(round(d_normal(seed, pid, f"{pid}/T{j}", "FLOWERS_PER_TRUSS", tr["flowers_mean"], tr["flowers_sd"])))), node="pending_user(not placed)") for j in range(1, n_tr + 1)]
    spec["notes"].append(f"trusses drawn ({n_tr}) but not placed: node position pending_user")
    return spec


def write_draws():
    with open(OUT / "random_draws.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(DRAWS[0])); w.writeheader(); w.writerows(DRAWS)


def distribution_check(specs, R):
    """Per-rank means / shares of the 16 check plants vs the rule file: mean within 1 SD, shares within 0.1 (main prompt §4.3)."""
    rows = []
    def chk(organ, rank, field, vals, kind, mean, sd, shares=None):
        if kind == "categorical":
            n = len(vals); obs = {k: sum(1 for v in vals if str(v) == str(k)) / n for k in shares}; worst = max(abs(obs[k] - shares[k]) for k in shares)
            rows.append(dict(organ=organ, rank_class=rank, field=field, n=n, rule=json.dumps(shares), generated=json.dumps({k: round(v, 3) for k, v in obs.items()}), criterion="shares within 0.1", max_dev=worst, passed=worst <= 0.1))
        else:
            n = len(vals); m = float(np.mean(vals)); s = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            rows.append(dict(organ=organ, rank_class=rank, field=field, n=n, rule=f"{mean} +- {sd}", generated=f"{m:.3f} +- {s:.3f}", criterion="mean within 1 SD of the rule", max_dev=abs(m - mean), passed=abs(m - mean) <= (sd if sd else 1e-9)))
    main = [lf for s in specs for lf in s["leaves"]]
    for r in range(1, 10):
        v = [lf for lf in main if lf["k"] == r]
        if len(v) >= 3:
            th, sd, _, _ = TPL.theta0(r); chk("main_compound_leaf", f"rank{r}", "theta0_above_horizontal_deg", [x["theta0_deg"] for x in v], "normal", th, sd)
            chk("main_compound_leaf", f"rank{r}", "leaflets_per_leaf", [x["leaflets"] for x in v], "categorical", None, None, {str(k): p for k, p in TPL.leaflet_shares(r).items()})
    dth, dth_sd, _, _ = TPL.dtheta(); L, L_sd, _ = TPL.leaf_length(); lo, hi, _ = TPL.divergence_range()
    chk("main_compound_leaf", "all", "dtheta_total_bend_deg", [x["dtheta_deg"] for x in main], "normal", dth, dth_sd)
    chk("main_compound_leaf", "all", "leaf_length_mm", [x["length_mm"] for x in main], "normal", L, L_sd)
    ib = internode_beta(); chk("main_stem", "internode", "internode_length_mm", [n["internode_mm"] for s in specs for n in s["nodes"]], "beta", ib["mean_mm"], ib["sd_mm"])
    nl = TPL.main_leaves(); chk("main_stem", "plant", "main_leaves_count", [len(s["leaves"]) for s in specs], "normal", nl[0], nl[1])
    div = [((b["azimuth_deg"] - a["azimuth_deg"]) * s["handedness"]) % 360 for s in specs for a, b in zip(s["leaves"], s["leaves"][1:])]
    chk("main_compound_leaf", "leaf", "divergence_deg", div, "uniform", (lo + hi) / 2, (hi - lo) / math.sqrt(12))
    side = TPL.side(); sl = [lf for s in specs for sa in s["side_axes"] for lf in sa["leaves"]]
    chk("side_axis_compound_leaf", "all", "theta0_above_horizontal_deg", [x["theta0_deg"] for x in sl], "normal", side["theta0"], side["theta0_sd"])
    chk("side_axis_compound_leaf", "all", "leaflets_per_leaf", [x["leaflets"] for x in sl], "categorical", None, None, {str(k): v for k, v in side["shares"].items()})
    chk("truss", "plant", "trusses_per_plant", [len(s["trusses"]) for s in specs], "categorical", None, None, truss_shares())
    with open(OUT / "check_plants_distribution_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    R, ib = rules(); print(json.dumps(ib, indent=1))
