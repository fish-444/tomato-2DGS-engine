"""Main / lateral axis of a skeleton_v1.1 (contract build/p1_skeleton_layer_20261101_01/CONTRACT.md, Amendment A6).

The user's stem clicks (+ the pot centre where the rim fit exists, + leaf bases moved halfway toward the stem) -> a smoothing curve; leaf bases -> nodes; gaps longer than
1.8 x the engine's internode length -> hypothesised nodes (`fspm_node_hypothesis`, never observed). Engine values are read from the reference run, nothing in the engine changes.
Everything computed here is derived data: `fit_all` is re-run whenever the skeleton is re-derived from its logs.

usage: python -m tomato3d.organs.skeleton_axis --step r31b
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree

from tomato3d.organs import skeleton_io as SIO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/p1_skeleton_layer_20261101_01/axis"
ENGINE_STATES = Path("C:/fsp_build/engine_dynamic_20261026_01/runs/R/native/working/states.csv")
A = dict(w_click=1.0, w_root=1.0, w_leaf=0.5, click_sigma_mm=1.0, lean_deg=15.0, lean_window_mm=10.0, lam_factor=4.0, lam_steps=6, cross_arc_mm=10.0, cross_gap_mm=2.0,
         unattached_mm=SIO.AXIS_V["unattached_mm"], observed_gap_mm=SIO.AXIS_V["observed_gap_mm"], gap_factor=1.8, engine_day="71", dir_mm=10.0, stem_label_mm=3.0, phyllotaxis_deg=137.5)
LABELS = ["user_skeleton_axis", "model_completed", "conditional_on_unvalidated_scale", "after_checkpoint_v1"]
OBSERVED = ("user_click", "observed_photo_click", "observed_3d_click")


@lru_cache(maxsize=1)
def engine_internodes():
    """{rank: (length_mm, radius_mm)} of the reference run R, plant p1, engine day 71 (59.9 DAG), class Internode."""
    out = {}
    with open(ENGINE_STATES, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["day"] == A["engine_day"] and r["plant_instance_id"] == "p1" and r["class"] == "Internode":
                out[int(r["true_leaf_order"])] = (float(r["length_m"]) * 1000.0, float(r["width_m"]) * 500.0)
    return dict(sorted(out.items()))


def rank_law(rank):
    E = engine_internodes(); top = max(E)
    return (*E[min(max(rank, 1), top)], rank > top)


def root_for(plant_id):
    """Root on the pot surface where a rim fit exists (R3-1b only on disk); else None (`rim_fit_unavailable`)."""
    if plant_id == "R3-1b":
        from tomato3d.organs import compound_cluster as CC
        return np.array([CC.POT_CENTRE[0], CC.POT_CENTRE[1], CC.POT_TOP], float)
    return None


# ---------------------------------------------------------------- geometry helpers
def arclen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def nearest_on_polyline(P, q):
    """(point, arc position) of the nearest point to q on polyline P (all in the same unit)."""
    s = arclen(P); best = (P[0], 0.0, np.inf)
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]; d = b - a; L2 = float(d @ d)
        t = 0.0 if L2 == 0 else min(max(float((q - a) @ d) / L2, 0.0), 1.0); x = a + t * d; dist = float(np.linalg.norm(q - x))
        if dist < best[2]:
            best = (x, s[i] + t * math.sqrt(L2), dist)
    return best[0], best[1]


def resample_curve(P, step=1.0):
    s = arclen(P); n = max(int(s[-1] // step), 1); t = np.linspace(0, s[-1], n + 1)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


def tangents(C):
    T = np.gradient(C, axis=0); return T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)


def lean_ok(C, step=1.0):
    """Direction change <= 15 deg over every 10 mm window of a 1 mm-sampled curve."""
    T = tangents(C); w = int(round(A["lean_window_mm"] / step))
    if len(C) <= w:
        return True, 0.0
    ang = np.degrees(np.arccos(np.clip((T[:-w] * T[w:]).sum(1), -1, 1))); return bool(ang.max() <= A["lean_deg"]), float(ang.max())


def self_crossing(C, step=1.0):
    tr = cKDTree(C); pairs = tr.query_pairs(A["cross_gap_mm"], output_type="ndarray")
    return bool(len(pairs) and (np.abs(pairs[:, 0] - pairs[:, 1]) * step > A["cross_arc_mm"]).any())


def spline_curve(t, X, w, s_factor=1.0):
    """t strictly increasing chord parameter (mm), X (n,3) mm, w per-anchor weights -> (1 mm-sampled curve, arc position of each anchor along it). <= 3 anchors: the polyline."""
    if len(t) <= 3:
        return resample_curve(X), arclen(X)
    ww = np.sqrt(w) / A["click_sigma_mm"]; s = len(t) * s_factor; tt = np.linspace(t[0], t[-1], max(int((t[-1] - t[0]) // 0.5), 2) + 1)
    D = np.stack([UnivariateSpline(t, X[:, k], w=ww, k=3, s=s)(tt) for k in range(3)], 1); return resample_curve(D), np.interp(t, tt, arclen(D))     # anchor s from its own parameter, not a nearest-point lookup


# ---------------------------------------------------------------- one axis
def anchors_for(axis, leaves_mm, root_mm):
    """Ordered anchors [(xyz_mm, weight, source, ref)] for one axis."""
    pts = [(np.asarray(p["xyz"], float) * 1000.0, p["source"]) for p in axis["points"]]
    clicks = [(x, s_) for x, s_ in pts if s_ in OBSERVED]
    base = []
    if axis["type"] == "main" and root_mm is not None:
        base = [(root_mm, A["w_root"], "pot_centre_auto", None)]
    prov = np.array([b[0] for b in base] + [c[0] for c in clicks])
    if len(prov) == 0:
        return []
    if len(prov) == 1:
        prov = np.vstack([prov[0], prov[0] + [0, 0, 200.0]])                  # a single point: provisional vertical line through it
    anc = base + [(x, A["w_click"], s_, None) for x, s_ in clicks]
    if axis["type"] == "main":
        for uid, P in leaves_mm.items():
            q, _ = nearest_on_polyline(prov, P[0])
            anc.append((P[0] + 0.5 * (q - P[0]), A["w_leaf"], "leaf_base_anchor", uid))
    order = np.argsort([nearest_on_polyline(prov, a[0])[1] for a in anc], kind="stable")
    anc = [anc[i] for i in order]
    if base:                                                                    # the root stays the first anchor
        anc = [a for a in anc if a[2] == "pot_centre_auto"] + [a for a in anc if a[2] != "pot_centre_auto"]
    return anc


def fit_axis(axis, leaves_mm, root_mm):
    anc = anchors_for(axis, leaves_mm, root_mm)
    if len(anc) < 2:
        return None
    X = np.array([a[0] for a in anc]); w = np.array([a[1] for a in anc]); t = arclen(X)
    keep = np.r_[True, np.diff(t) > 1e-6]; X, w, t = X[keep], w[keep], t[keep]; anc = [a for a, k in zip(anc, keep) if k]
    flags = []; sf = 1.0; C, s_anc = spline_curve(t, X, w, sf); ok, worst = lean_ok(C); steps = 0
    while not ok and steps < A["lam_steps"] and len(t) > 3:
        sf *= A["lam_factor"]; steps += 1; C, s_anc = spline_curve(t, X, w, sf); ok, worst = lean_ok(C)
    if not ok:
        flags.append("lean_limit_exceeded")
    if self_crossing(C):
        flags.append("self_crossing")
    sC = arclen(C); s_anc = np.minimum(s_anc, sC[-1]); res = np.linalg.norm(X - C[np.clip(np.round(s_anc).astype(int), 0, len(C) - 1)], axis=1)      # residual = anchor to the curve at its own parameter
    return dict(curve=C, s=sC, anchors=[dict(xyz_mm=a[0].round(3).tolist(), weight=a[1], source=a[2], leaf_uid=a[3], s_mm=round(float(sa), 2), residual_mm=round(float(r_), 3)) for a, sa, r_ in zip(anc, s_anc, res)],
                smoothing_factor=sf, smoothing_steps=steps, lean_max_deg=round(worst, 2), flags=flags)


def segments_of(fit, length_mm):
    """Segment ends = user clicks (leaf-base anchors are fit anchors, not segment ends; A6 correction): consecutive clicks <= 15 mm apart along the curve -> observed;
    every other part of [0, length] -> model_completed."""
    cs = sorted(a["s_mm"] for a in fit["anchors"] if a["source"] in OBSERVED); obs = [(a, b) for a, b in zip(cs, cs[1:]) if 0 < b - a <= A["observed_gap_mm"]]
    seg, cur = [], 0.0
    for a, b in obs:
        if a > cur + 1e-6:
            seg.append(dict(s0_mm=round(cur, 2), s1_mm=round(a, 2), provenance="model_completed"))
        seg.append(dict(s0_mm=round(max(a, cur), 2), s1_mm=round(b, 2), provenance="observed", ends=["user_click", "user_click"])); cur = b
    if length_mm > cur + 1e-6:
        seg.append(dict(s0_mm=round(cur, 2), s1_mm=round(length_mm, 2), provenance="model_completed"))
    return seg


def leaf_azimuth(P_mm, tangent):
    from tomato3d.organs.skeleton_tree import first_direction
    d = first_direction(P_mm, A["dir_mm"])
    if d is None:
        return None
    t = tangent / np.linalg.norm(tangent); v = d - (d @ t) * t; ref = np.array([1.0, 0, 0]) - t[0] * t
    if np.linalg.norm(v) < 1e-9 or np.linalg.norm(ref) < 1e-9:
        return None
    return float(math.degrees(math.atan2(float(np.cross(ref, v) @ t), float(ref @ v))) % 360.0)


def circ_stats(deg):
    if not deg:
        return None
    r = np.radians(deg); C, S = np.cos(r).mean(), np.sin(r).mean(); R = math.hypot(C, S)
    return dict(n=len(deg), mean_deg=round(math.degrees(math.atan2(S, C)) % 360, 1), sd_deg=round(math.degrees(math.sqrt(max(-2 * math.log(max(R, 1e-12)), 0))), 1))


def fit_all(sk, surfels_xyz=None, root=None):
    """Fit every active axis of sk in place; attach leaves; insert hypothesised nodes; returns a summary. root: xyz (m) of the pot centre or None."""
    axes = [a for a in sk.get("axes", []) if a["status"] == "active"]
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost")]
    for l in sk["leaves"]:
        for k in ("attach_axis_uid", "attach_s_mm", "node_index", "attach_distance_mm"):
            l[k] = None
    if not axes:
        return None
    leaves_mm = {l["leaf_uid"]: np.asarray(l["polyline_xyz"], float) * 1000.0 for l in live}; root_mm = None if root is None else np.asarray(root, float) * 1000.0
    fits = {}
    for a in sorted(axes, key=lambda a: a["type"] != "main"):
        if a["type"] == "main" and root_mm is not None and not any(p["source"] == "pot_centre_auto" for p in a["points"]):
            a["flags"] = [f for f in a.get("flags", []) if f != "rim_fit_unavailable"]
        elif a["type"] == "main" and root_mm is None:
            a["flags"] = sorted(set(a.get("flags", [])) | {"rim_fit_unavailable"})
        f = fit_axis(a, leaves_mm, root_mm)
        a["fit"] = None if f is None else {k: v for k, v in f.items() if k not in ("curve", "s")}
        if f is None:
            a["segments"], a["nodes"] = [], []; continue
        fits[a["axis_uid"]] = f; a["fit"]["curve_mm"] = f["curve"].round(2).tolist(); a["fit"]["length_mm"] = round(float(f["s"][-1]), 2)
        a["segments"] = segments_of(f, float(f["s"][-1])); a["flags"] = sorted(set(a.get("flags", [])) | set(f["flags"]))
        if a["type"] == "lateral" and a.get("parent_axis_uid") in fits:
            P = fits[a["parent_axis_uid"]]; _, i = cKDTree(P["curve"]).query(f["curve"][0]); a["attach_s_mm"] = round(float(P["s"][i]), 2)
    # leaves -> nearest fitted axis
    for l in live:
        b = leaves_mm[l["leaf_uid"]][0]; best = None
        for uid, f in fits.items():
            d, i = cKDTree(f["curve"]).query(b)
            if best is None or d < best[0]:
                best = (float(d), uid, float(f["s"][i]), int(i))
        if best is None:
            continue
        l["attach_distance_mm"] = round(best[0], 2)
        if best[0] <= A["unattached_mm"]:
            l["attach_axis_uid"] = best[1]; l["attach_s_mm"] = round(best[2], 2)
    summary = {}
    for a in axes:
        f = fits.get(a["axis_uid"])
        if f is None:
            continue
        att = sorted([l for l in live if l["attach_axis_uid"] == a["axis_uid"]], key=lambda l: l["attach_s_mm"]); C, sC = f["curve"], f["s"]; T = tangents(C)
        # bottom segment root -> first node: straight unless an observed segment covers it
        if a["type"] == "main" and att:
            s1 = att[0]["attach_s_mm"]
            if not any(sg["provenance"] == "observed" and sg["s0_mm"] < s1 for sg in a["segments"]):
                m = sC <= s1; i1 = int(np.nonzero(m)[0][-1]); start = root_mm if root_mm is not None else C[0]                     # straight from the root itself (A6 correction)
                C[: i1 + 1] = np.linspace(start, C[i1], i1 + 1); a["fit"]["curve_mm"] = C.round(2).tolist()
                a["segments"] = [dict(s0_mm=0.0, s1_mm=round(float(sC[i1]), 2), provenance="model_completed", ends=["root", "first_node"], shape="straight")] + [sg for sg in a["segments"] if sg["s0_mm"] >= sC[i1]]
        nodes = []; prev_s = 0.0 if a["type"] == "main" else None; k = 0
        for l in att:
            s_ = l["attach_s_mm"]; L, r, beyond = rank_law(k + 1)
            if prev_s is not None and (s_ - prev_s) > A["gap_factor"] * L:
                m = int(round((s_ - prev_s) / L)) - 1
                for j in range(1, m + 1):
                    sh = prev_s + j * (s_ - prev_s) / (m + 1); i = int(np.argmin(np.abs(sC - sh))); k += 1; Lh, rh, bh = rank_law(k)
                    nodes.append(dict(node_index=k, s_mm=round(float(sh), 2), xyz_mm=C[i].round(2).tolist(), leaf_uid=None, kind="fspm_node_hypothesis", leaf="unobserved", rank=k,
                                      expected_internode_mm=round(Lh, 2), radius_mm=round(rh, 2), radius_source="model_generated", flags=["fspm_node_hypothesis"] + (["rank_beyond_engine"] if bh else [])))
            k += 1; L, r, beyond = rank_law(k); i = int(np.argmin(np.abs(sC - s_)))
            az = leaf_azimuth(leaves_mm[l["leaf_uid"]], T[i])
            nodes.append(dict(node_index=k, s_mm=s_, xyz_mm=C[i].round(2).tolist(), leaf_uid=l["leaf_uid"], kind="observed_leaf", rank=k, expected_internode_mm=round(L, 2), radius_mm=round(r, 2),
                              radius_source="model_generated", azimuth_deg=(round(az, 1) if az is not None else None), flags=(["rank_beyond_engine"] if beyond else [])))
            l["node_index"] = k; prev_s = s_
        a["nodes"] = nodes
        obs_nodes = [n for n in nodes if n["kind"] == "observed_leaf" and n.get("azimuth_deg") is not None]
        div = [((b["azimuth_deg"] - c["azimuth_deg"]) % 360.0) for c, b in zip(obs_nodes, obs_nodes[1:])]
        L_obs = sum(sg["s1_mm"] - sg["s0_mm"] for sg in a["segments"] if sg["provenance"] == "observed"); L_tot = float(sC[-1])
        summary[a["axis_uid"]] = dict(type=a["type"], clicks=sum(p["source"] in OBSERVED for p in a["points"]), length_mm=round(L_tot, 1), observed_mm=round(L_obs, 1),
                                      observed_share=round(L_obs / L_tot, 3) if L_tot else None, nodes_observed=sum(n["kind"] == "observed_leaf" for n in nodes),
                                      nodes_hypothesised=sum(n["kind"] == "fspm_node_hypothesis" for n in nodes), divergence_raw=circ_stats(div),
                                      divergence_folded=circ_stats([min(d, 360 - d) for d in div]), divergence_list=[round(d, 1) for d in div], flags=a["flags"],
                                      lean_max_deg=a["fit"]["lean_max_deg"], smoothing_steps=a["fit"]["smoothing_steps"], anchor_residual_mm_max=max(x["residual_mm"] for x in a["fit"]["anchors"]))
    summary["unattached"] = [l["leaf_uid"] for l in live if l["attach_axis_uid"] is None]
    sk["axis_labels"] = LABELS; sk["axis_rules"] = dict(A, engine_states=str(ENGINE_STATES), dag="dag_assumed_59.9")
    SIO.validate_axes(sk); return summary


def stem_display_labels(sk, surfels_xyz, rows):
    """Surfels <= 3 mm from OBSERVED axis segments -> stem (display only); blade labels are never touched."""
    S = np.asarray(surfels_xyz, float) * 1000.0; lab = np.zeros(len(S), np.int8); tr = cKDTree(S)
    for a in sk.get("axes", []):
        if a["status"] != "active" or not a.get("fit"):
            continue
        C = np.asarray(a["fit"]["curve_mm"]); sC = arclen(C)
        for sg in a["segments"]:
            if sg["provenance"] != "observed":
                continue
            P = C[(sC >= sg["s0_mm"]) & (sC <= sg["s1_mm"])]
            for js in tr.query_ball_point(P, A["stem_label_mm"]):
                lab[js] = 1
    return dict(rows=np.asarray(rows), stem=lab)


# ---------------------------------------------------------------- R3-1b simulated test (A6 section 9)
def simulated_clicks_r31b():
    Z = np.load(ROOT / "build/p1_pc_absorb_20261027_01/kaggle_data/r8pc_bundle.npz"); X = Z["xyz"][Z["axis_kind"] == 1] * 1000.0
    z0 = X[:, 2].min(); b = ((X[:, 2] - z0) // 10).astype(int); out = []
    for k in np.unique(b):
        m = b == k
        if m.sum() >= 5:
            out.append(np.median(X[m], 0))
    return np.array(out) / 1000.0, Z


def step_r31b():
    import copy
    sk = SIO.load(SIO.skeleton_dir(ROOT, "R3-1b", "2026-10-21") / "skeleton_v1.json"); sk = copy.deepcopy(sk)
    clicks, Z = simulated_clicks_r31b(); ax = SIO.add_axis(sk, "main")
    for c in clicks:
        SIO.axis_add_point(sk, ax["axis_uid"], c, "user_click")
    summ = fit_all(sk, root=root_for("R3-1b"))
    per_leaf = {}
    for l in sk["leaves"]:
        L = l.get("attrs", {}).get("pipeline_leaf")
        if l["status"] not in ("active", "new") or not L:
            continue
        cur = per_leaf.get(L)
        if l["attach_distance_mm"] is not None and (cur is None or l["attach_distance_mm"] < cur["attach_distance_mm"]):
            per_leaf[L] = dict(leaf_uid=l["leaf_uid"], attach_distance_mm=l["attach_distance_mm"], attached=l["attach_axis_uid"] is not None, node_index=l["node_index"], attach_s_mm=l["attach_s_mm"])
    stem = stem_display_labels(sk, Z["xyz"], Z["rows"]); ak = Z["axis_kind"]
    res = dict(labels=LABELS + ["simulated_axis_clicks", "sim_skeleton_completed"], clicks=len(clicks), click_rule="median of frozen stem-like surfels (axis_kind 1) per 10 mm height bin with >= 5 surfels",
               root="pot_centre_auto (rim fit)", engine_internodes_day71={k: [round(v[0], 2), round(v[1], 2)] for k, v in engine_internodes().items()}, axes=summ,
               nodes_per_pipeline_leaf={str(k): v for k, v in sorted(per_leaf.items())}, pipeline_leaves_with_node=sum(v["attached"] for v in per_leaf.values()), pipeline_leaves=8,
               stem_display=dict(labelled=int(stem["stem"].sum()), of_stem_like=int((stem["stem"] == 1)[ak == 1].sum()), stem_like_total=int((ak == 1).sum())))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "r31b_axis_skeleton_v1_1.json").write_text(json.dumps(sk, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "r31b_stem_display_labels.npz", **stem); json.dump(res, open(OUT / "r31b_axis_test.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    tree_check(sk, "R3-1b", res); figure(sk, Z, clicks, "R3-1b"); print(json.dumps(res, indent=1, ensure_ascii=False)); return res


def tree_check(sk, pid, res):
    from tomato3d.organs import skeleton_tree as ST
    t = ST.skeleton_to_plant_tree(sk); (OUT / f"plant_tree_axis_{pid}.json").write_text(json.dumps(t, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    rt = ST.engine_round_trip(t, OUT / f"engine_{pid}", pid); res["plant_tree"] = dict(nodes=len(t["nodes"]), leaves=len(t["leaves"]), engine_round_trip=rt)
    json.dump(res, open(OUT / f"{'r31b' if pid == 'R3-1b' else pid}_axis_test.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)


def figure(sk, Z, clicks, pid):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    X = Z["xyz"] * 1000.0; fig, ax = plt.subplots(1, 2, figsize=(13, 7))
    for k, (i, j, nm) in enumerate(((0, 2, "side x-z"), (1, 2, "side y-z"))):
        a_ = ax[k]; a_.scatter(X[:, i], X[:, j], s=0.2, c="#dddddd"); st = Z["axis_kind"] == 1; a_.scatter(X[st, i], X[st, j], s=0.5, c="#8c6d31", label="frozen stem-like")
        for a in sk["axes"]:
            if not a.get("fit"):
                continue
            C = np.asarray(a["fit"]["curve_mm"]); sC = arclen(C)
            for sg in a["segments"]:
                m = (sC >= sg["s0_mm"]) & (sC <= sg["s1_mm"]); a_.plot(C[m, i], C[m, j], "-" if sg["provenance"] == "observed" else "--", c="#1a9850" if sg["provenance"] == "observed" else "#d73027", lw=2)
            for n in a["nodes"]:
                a_.plot(n["xyz_mm"][i], n["xyz_mm"][j], "o" if n["kind"] == "observed_leaf" else "x", c="k" if n["kind"] == "observed_leaf" else "#d73027", ms=6)
        a_.scatter(clicks[:, i] * 1000, clicks[:, j] * 1000, s=20, c="#2166ac", marker="^", label="simulated clicks"); a_.set_aspect("equal"); a_.set_title(nm)
    ax[0].legend(fontsize=8); fig.suptitle(f"{pid} axis: green solid = observed, red dashed = model_completed, x = fspm_node_hypothesis; conditional_on_unvalidated_scale")
    fig.tight_layout(); fig.savefig(OUT / f"{'r31b' if pid == 'R3-1b' else pid}_axis.png", dpi=90); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", default="r31b"); a = ap.parse_args(argv)
    step_r31b()


if __name__ == "__main__":
    main()
