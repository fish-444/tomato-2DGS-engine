"""E4 - complete hidden leaflets from Micro-Tom ratio priors, as N samples (never one answer).

Revised 2026-10-14 (session E2): the first version completed too eagerly at low occlusion and its sample spread
did not cover the real error. Two changes:

1. **Gate - positive evidence only.** A missing slot is created only where its EXPECTED FOOTPRINT is covered by
   another observed leaf. Footprints:
   - a partner slot uses the visible partner's mask mirrored across the compound axis (pairs are symmetric);
   - a missing terminal uses a disk of ``terminal_probe_radius_rel`` x the compound's top length at the axis tip.

   Rule (candidate values, priors file ``gates``): at most ``slot_max_background_frac`` of the footprint on observed
   BACKGROUND, and at least ``slot_min_covered_frac`` on another leaf. Outcomes:
   - background over the limit -> ``absent`` (the leaflet would be visible there);
   - neither limit met -> ``weak_evidence``, NOT created;
   - footprint mostly off the image -> ``unavailable``.

   A visible leaflet counts as CUT by the hull-cut evidence ``cut px / visible px`` (E1). Its per-sample cut
   probability is ``evidence / (2 x cut_evidence_frac)``, clipped to [0, 1], so weak evidence rarely completes.

2. **Uncertainty in the samples.** Each of the N samples draws:
   - the elevation of every compound half from N(estimate, measured sd) (E3: robust spread of the side-pixel
     heights, NOT fitted to the test truth);
   - a blade roll and bend arc per leaflet in [0, ``blade_curvature_max_deg``], giving area factor g(roll) x g(bend)
     with g(phi) = (phi/2) / sin(phi/2). This is the flat-blade assumption O7 expressed as uncertainty;
   - each cut flag, from its probability;
   - the ratio priors, the length/width ratio and a shape from the 28-leaflet pool (one position-free set, uniform).

3. **Leaf age (session E3, Session G rule).** With ``stagger_dd`` given, a RESTORED leaflet's size is
   mature ratio x growth fraction(position, compound age). The growth fraction is
   :func:`tomato3d.recon.stage1.age.leaflet_length_fraction` with offsets terminal 0 ... pair nearest the stem
   n x stagger, and the compound age comes from :func:`age.compound_ages`. The reference length is the mature
   middle-pair length implied by the uncut visible leaflets: observed / (ratio x their own growth fraction);
   fractions < ``MIN_REF_FRACTION`` are not used as references. Observed leaflets are never resized. Without an
   age the mature prior is used and the compound is flagged ``age_unavailable_mature_prior``.
   ``stagger_dd=None`` reproduces E2.

Other rules, unchanged from the first version:
- whole pairs missing on both sides cannot be located -> listed, never invented;
- areas never mix: kind (1) ``area_visible_mm2`` (tilt-corrected visible area; per-sample list
  ``area_visible_samples_mm2``) and kind (2) ``area_completed_mm2`` (per sample);
- ``fsp_calibration_weight`` is 1 for observed and 0 for reconstructed_prior (E7, user decision: keep 0).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from tomato3d.recon.stage1.priors import Priors
from tomato3d.recon.stage1.views import TopObs

PAIR_TOL = 0.12
PAIR_RATIO = {1: "basal_pair_rel", 2: "middle_pair_rel", 3: "distal_pair_rel"}
MAX_ELEV_DEG = 85.0
MIN_REF_FRACTION = 0.05


def _pairs(laterals: list[dict], L_axis: float) -> list[list[dict]]:
    ordered = sorted(laterals, key=lambda r: r["station_mm"])
    groups: list[list[dict]] = []
    for r in ordered:
        if groups and abs(r["station_mm"] - np.mean([x["station_mm"] for x in groups[-1]])) <= PAIR_TOL * L_axis \
                and r["side"] not in {x["side"] for x in groups[-1]}:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def _footprint_stats(top: TopObs, xy_mm: np.ndarray, exclude: set[int]) -> dict:
    """Shares of an expected footprint (plant mm points) on background / on another leaf / off the image."""
    uv = np.round(top.camera.project(np.c_[xy_mm, np.zeros(len(xy_mm))])).astype(int)
    uv = np.unique(uv, axis=0)
    h, w = top.id_map.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    n = len(uv)
    if not n:
        return dict(n_px=0, background=1.0, covered=0.0, off_image=0.0)
    lab = top.id_map[uv[inside, 1], uv[inside, 0]]
    covered = np.isin(lab, list(exclude), invert=True) & (lab > 0)
    return dict(n_px=n, background=float((lab == 0).sum()) / n, covered=float(covered.sum()) / n,
                off_image=float((~inside).sum()) / n)


def _gate(stats: dict, priors: Priors) -> str:
    if stats["off_image"] > 0.5:
        return "unavailable"
    if stats["background"] > float(priors.value("gates", "slot_max_background_frac")):
        return "absent"
    if stats["covered"] >= float(priors.value("gates", "slot_min_covered_frac")):
        return "hidden"
    return "weak_evidence"


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    return float(v[int(np.searchsorted(c, 0.5 * c[-1]))])


def _arc(phi_deg: float) -> float:
    p = math.radians(phi_deg) / 2
    return 1.0 if p < 1e-6 else p / math.sin(p)


def run(top: TopObs, tilt_out: Mapping, height_out: Mapping, priors: Priors, n_samples: int, seed: int,
        registry: Mapping[str, Mapping] | None = None, stagger_dd: float | None = None,
        age_mode: str = "age") -> dict:
    """``age_mode``: ``age`` (E3: restored size = age-rule size) | ``shrink_only`` (E3b, declared in the E3 report
    before its run: restored size = min(E2 size, age-rule size); the age rule may only make a restored leaflet
    smaller, never larger)."""
    from tomato3d.recon.stage1 import age as age_mod

    rng = np.random.default_rng(seed)
    age_par = age_mod.parameters(priors) if stagger_dd is not None else None
    ages = age_mod.compound_ages(registry or {}, age_par) if stagger_dd is not None else {}
    cut_px: dict[int, float] = {}
    for rel in height_out["relations"]:
        if rel["source"] == "occlusion":
            cut_px[rel["b"]] = cut_px.get(rel["b"], 0.0) + max(rel.get("cut_a_px", 0), rel.get("cut_b_px", 0))
    cut_scale = 2.0 * float(priors.value("gates", "cut_evidence_frac"))
    curv_max = float(priors.value("uncertainty", "blade_curvature_max_deg"))
    leaflets_by_comp: dict[str, list[dict]] = {}
    for r in tilt_out["leaflets"]:
        leaflets_by_comp.setdefault(r["compound"], []).append(r)
    elements, compounds, unavailable = [], [], []
    for geo in tilt_out["compounds"]:
        cid = geo["compound"]
        leafs = leaflets_by_comp.get(cid, [])
        if geo.get("L_h_mm") is None or geo.get("elevation_deg") is None:
            unavailable.append(dict(compound=cid, item="completion", reason=geo.get("reason") or "no compound geometry"))
            compounds.append(dict(compound=cid, area_visible_mm2=None, area_visible_samples_mm2=[None] * n_samples,
                                  area_visible_top_mm2=sum(r["area_top_mm2"] for r in leafs),
                                  area_completed_mm2=[None] * n_samples, reason="elevation unavailable"))
            continue
        L_axis = float(geo["L_h_mm"])
        base, d = np.asarray(geo["base_xy"]), np.asarray(geo["dir"])
        terminal = [r for r in leafs if r["is_terminal"]]
        laterals = [r for r in leafs if not r["is_terminal"]]
        groups = _pairs(laterals, L_axis)
        for k, grp in enumerate(groups, start=1):
            for r in grp:
                r["pair_index"] = min(k, 3)
                r["pair_order"] = k
        n_groups = max(len(groups), 1)
        A = ages.get(cid, {}).get("age_dd") if stagger_dd is not None else None
        age_flag = "mature_prior" if stagger_dd is None else (
            "age_aware" if A is not None else "age_unavailable_mature_prior")

        def grow(pair_order: int | None, terminal: bool) -> float:
            """growth fraction of a position at this compound's age (1 = mature prior, E2)"""
            if A is None:
                return 1.0
            off = 0.0 if terminal else age_mod.offset_dd("pair", pair_order, n_groups, stagger_dd)
            return age_mod.leaflet_length_fraction(A - off, age_par)
        for r in leafs:
            vis_px = r["area_top_mm2"] / top.mm2_per_px()
            r["cut_evidence"] = cut_px.get(r["label"], 0.0) / max(vis_px, 1.0)
            r["p_cut"] = float(np.clip(r["cut_evidence"] / cut_scale, 0.0, 1.0))
        # slots + gate
        slots = []
        for grp in groups:
            if len(grp) != 1:
                continue
            r = grp[0]
            ys, xs = np.nonzero(top.id_map == r["label"])
            pts = top.camera.unproject_xy(np.stack([xs, ys], 1).astype(float))
            along = (pts - base) @ d
            mirror = base + along[:, None] * d - ((pts - base) - along[:, None] * d)
            c = np.asarray(r["centroid_xy"])
            ca = (c - base) @ d
            stats = _footprint_stats(top, mirror, {r["label"]})
            slots.append(dict(kind="partner", pair_index=r["pair_index"], pair_order=r["pair_order"],
                              side="R" if r["side"] == "L" else "L",
                              xy=base + ca * d - ((c - base) - ca * d), partner=r, gate_stats=stats,
                              gate=_gate(stats, priors)))
        if not terminal:
            rad = float(priors.value("gates", "terminal_probe_radius_rel")) * L_axis
            tip = np.asarray(geo["tip_xy"])
            g = np.mgrid[-rad:rad:top.camera.mm_per_px, -rad:rad:top.camera.mm_per_px].reshape(2, -1).T
            disk = tip + g[np.hypot(g[:, 0], g[:, 1]) <= rad]
            stats = _footprint_stats(top, disk, set())
            slots.append(dict(kind="terminal", pair_index=None, pair_order=None, side="C", xy=tip, partner=None,
                              gate_stats=stats,
                              gate=_gate(stats, priors)))
        n_pairs_seen = len(groups)
        if n_pairs_seen < int(priors.value("ratios", "primary_pairs")):
            unavailable.append(dict(compound=cid, item="whole_pairs",
                                    reason=f"{n_pairs_seen} pair positions seen, prior {priors.value('ratios', 'primary_pairs')}: "
                                           "a pair missing on both sides cannot be located - not reconstructed"))
        hidden_slots = [s for s in slots if s["gate"] == "hidden"]
        vis_samples, comp_completed = [], []
        e1, e2 = geo["elevation_base_half_deg"], geo["elevation_tip_half_deg"]
        sd1 = geo.get("elevation_base_half_sd_deg") or 0.0
        sd2 = geo.get("elevation_tip_half_sd_deg") or 0.0
        for n in range(n_samples):
            en1 = float(np.clip(rng.normal(e1, sd1), -MAX_ELEV_DEG, MAX_ELEV_DEG))
            en2 = float(np.clip(rng.normal(e2, sd2), -MAX_ELEV_DEG, MAX_ELEV_DEG)) if geo["mode"] == "piecewise_2seg" \
                else en1 + (e2 - e1)
            ratio = {i: priors.draw(PAIR_RATIO[i], rng) for i in (1, 2, 3)}
            r_term = priors.draw("terminal_rel", rng)
            vis_total, lt, cutn = 0.0, {}, {}
            for r in leafs:
                e = en1 if r["station_mm"] < L_axis / 2 else en2
                ce = math.cos(math.radians(e))
                blade = _arc(rng.uniform(0, curv_max)) * _arc(rng.uniform(0, curv_max))
                a = r["area_top_mm2"] / ce * blade
                cp = r["cos_phi"] if r.get("cos_phi") is not None else 1.0
                lt[r["obs_id"]] = math.hypot(r["L_top_mm"] * cp / ce, r["L_top_mm"] * math.sqrt(max(0.0, 1 - cp ** 2))) \
                    * _arc(rng.uniform(0, curv_max))
                r["_area_n"] = a
                cutn[r["obs_id"]] = rng.random() < r["p_cut"]
                vis_total += a
            vis_samples.append(vis_total)
            # mature middle-pair length implied by each uncut visible leaflet; inverse-variance weights g^2 (an
            # absolute length error divided by the growth fraction g has variance ~ 1/g^2) - first E3 pass used a
            # plain median and over-weighted young leaflets (build/recon_e_20261014_05)
            refs, wts, refs_e2 = [], [], []
            for r in laterals + terminal:
                if cutn[r["obs_id"]]:
                    continue
                rr = r_term if r["is_terminal"] else ratio[r["pair_index"]]
                refs_e2.append(lt[r["obs_id"]] / rr)
                gr_ = grow(None, True) if r["is_terminal"] else grow(r["pair_order"], False)
                if gr_ >= MIN_REF_FRACTION:
                    refs.append(lt[r["obs_id"]] / (rr * gr_))
                    wts.append(gr_ ** 2)
            L_compound = L_axis / max(math.cos(math.radians((en1 + en2) / 2)), 1e-3)
            fallback = priors.draw("largest_leaflet_over_compound_length", rng) * L_compound
            L_mid = weighted_median(refs, wts) if refs else fallback
            L_mid_e2 = weighted_median(refs_e2, [1.0] * len(refs_e2)) if refs_e2 else fallback

            def size(rel, pair_order, terminal):
                """(chosen length, E2 length, age-rule length) of a restored position (E3b: shrink-only = min)"""
                l_e2 = L_mid_e2 * rel
                l_age = L_mid * rel * grow(None if terminal else pair_order, terminal)
                if stagger_dd is None:
                    return l_e2, l_e2, l_e2
                return (min(l_e2, l_age) if age_mode == "shrink_only" else l_age), l_e2, l_age

            total = 0.0
            for r in leafs:
                if not cutn[r["obs_id"]]:
                    continue
                shape = priors.shapes[int(rng.integers(len(priors.shapes)))]
                lw = priors.draw("length_over_width", rng)
                partner = next((x for g in groups if r in g for x in g if x is not r and not cutn[x["obs_id"]]), None)
                if partner:
                    L = l_e2 = l_age = lt[partner["obs_id"]]
                else:
                    L, l_e2, l_age = size(r_term if r["is_terminal"] else ratio[r.get("pair_index", 2)],
                                          r.get("pair_order"), r["is_terminal"])
                full = shape.fill * L * L / lw * _arc(rng.uniform(0, curv_max)) * _arc(rng.uniform(0, curv_max))
                add = max(0.0, full - r["_area_n"])
                total += add
                elements.append(dict(compound=cid, element=r["obs_id"] + "#completion", sample=n, kind="cut_completion",
                                     length_mm=L, area_mm2=add, shape_id=shape.shape_id, cut_evidence=r["cut_evidence"],
                                     length_e2_mm=l_e2, length_age_mm=l_age, pair_order=r.get("pair_order"),
                                     source_kind="reconstructed_prior", fsp_calibration_weight=0, xy=r["centroid_xy"]))
            for sl in hidden_slots:
                shape = priors.shapes[int(rng.integers(len(priors.shapes)))]
                lw = priors.draw("length_over_width", rng)
                p = sl["partner"]
                if p is not None and not cutn[p["obs_id"]]:
                    L = l_e2 = l_age = lt[p["obs_id"]]
                    why = "symmetry"
                elif sl["kind"] == "terminal":
                    (L, l_e2, l_age), why = size(r_term, None, True), "terminal_ratio"
                else:
                    (L, l_e2, l_age), why = size(ratio[sl["pair_index"]], sl["pair_order"], False), "pair_ratio"
                add = shape.fill * L * L / lw * _arc(rng.uniform(0, curv_max)) * _arc(rng.uniform(0, curv_max))
                total += add
                elements.append(dict(compound=cid, element=cid + "/" + sl["kind"] + "_" + sl["side"] + str(sl["pair_index"] or ""),
                                     sample=n, kind=sl["kind"], length_mm=L, area_mm2=add, size_from=why,
                                     length_e2_mm=l_e2, length_age_mm=l_age, pair_order=sl["pair_order"],
                                     shape_id=shape.shape_id, length_over_width=lw, gate=sl["gate"],
                                     gate_stats=sl["gate_stats"], source_kind="reconstructed_prior",
                                     fsp_calibration_weight=0, xy=sl["xy"], side=sl["side"], pair_index=sl["pair_index"]))
            comp_completed.append(total)
        vis = sum(r["area_corrected_mm2"] for r in leafs if r.get("area_corrected_mm2") is not None)
        for r in leafs:
            r.pop("_area_n", None)
            elements.append(dict(compound=cid, element=r["obs_id"], sample=None, kind="observed", length_mm=r.get("L_true_mm"),
                                 area_mm2=r.get("area_corrected_mm2"), source_kind="observed", fsp_calibration_weight=1,
                                 xy=r["centroid_xy"], cut_evidence=r["cut_evidence"], p_cut=r["p_cut"],
                                 pair_index=r.get("pair_index")))
        compounds.append(dict(compound=cid, area_visible_mm2=vis, area_visible_samples_mm2=vis_samples,
                              area_visible_top_mm2=sum(r["area_top_mm2"] for r in leafs),
                              area_completed_mm2=comp_completed,
                              slot_gates={g: sum(s["gate"] == g for s in slots)
                                          for g in ("hidden", "absent", "weak_evidence", "unavailable")},
                              n_cut_likely=sum(r["p_cut"] >= 0.5 for r in leafs), pairs_seen=n_pairs_seen,
                              age_dd=A, age_source=ages.get(cid, {}).get("age_source"), size_rule=age_flag,
                              stagger_dd=stagger_dd, age_mode=age_mode if stagger_dd is not None else None,
                              n_groups=len(groups)))
    return dict(compounds=compounds, elements=elements, unavailable=unavailable, seed=seed, n_samples=n_samples)
