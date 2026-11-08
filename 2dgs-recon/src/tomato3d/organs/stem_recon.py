"""Practical stem reconstruction (contract build/p1_stem_20261026_01/CONTRACT.md): S0 slab-core axis, S0-P shortest path with height and radius
constraints (Wang et al. 2024 §2.4.2, amendment A1), S1 stem surfels, S2 nodes, S3 diameter profile, A2 diameter at a fixed height, A3 leaf insertion
angle, checks C2-C6, overlays. CPU only; R3-1b only (the nine batch individuals have no pot-top plane).

usage: python -m tomato3d.organs.stem_recon
"""

from __future__ import annotations

import argparse
import base64
import csv
import heapq
import io
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import light_organs as LO
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_stem_20261026_01"
PLY = LO.DEFAULTS["surfels"]
POT = dict(top_z=0.066, centre=(0.1707, 0.1452), radius=0.031)
TAGS = dict(scale="conditional_on_unvalidated_scale", biological_validation="not_established")
P = dict(slab_mm=5.0, core_share=0.30, slab_min=10, apex_mult=1.5, huber=1.345, tube_mult=2.0, tube_min_mm=6.0, node_k3=3, edge_mm=12.0, radius_pen_mm=5.0,
         win_mm=15.0, win_step_mm=5.0, win_min=40, ransac_it=200, ransac_tol_mm=1.0, margin_mm=2.0, widen=1.3, leaf_near=50, merge_mm=3.0, s3_min=40,
         a2_heights_mm=(15.0, 20.0, 10.0), a2_band_mm=6.0, a3_mm=20.0, c3_share=0.95, c4_deg=30.0, c6_seed=0, visible_tag_mm=5.0, clear_share=0.40)
NONLEAF = 100


# ---------------------------------------------------------------- data
def load(ply=PLY):
    sf = sio.read_surfels_full(ply); keep = (sf["opacity"] >= LO.OPACITY_MIN) & (sf["xyz"][:, 2] > POT["top_z"]); rows = np.nonzero(keep)[0]
    nrm = sf["normal"][rows].astype(float); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return dict(rows=rows, xyz=sf["xyz"][rows].astype(float), nrm=nrm, rgb=sf["rgb"][rows].astype(float))


# ---------------------------------------------------------------- slab cores
def tight_core(xy, k):
    """Seed = the point whose k-th xy neighbour is nearest; returns (member mask, centre, radius)."""
    t = cKDTree(xy); d, _ = t.query(xy, k=min(k + 1, len(xy))); r = d[:, -1]; s = int(np.argmin(r)); rad = float(r[s])
    m = np.linalg.norm(xy - xy[s], axis=1) <= rad + 1e-12; return m, np.median(xy[m], axis=0), rad


def slabs(xyz, top=POT["top_z"]):
    zi = np.floor((xyz[:, 2] - top) * 1000 / P["slab_mm"]).astype(int); return zi


def s0_cores(xyz):
    zi = slabs(xyz); out = {}
    for s in sorted(set(zi.tolist())):
        idx = np.nonzero(zi == s)[0]
        if len(idx) < P["slab_min"]: continue
        k = max(2, int(np.ceil(P["core_share"] * len(idx)))); m, c, r = tight_core(xyz[idx, :2], k)
        out[s] = dict(centre=c, radius=r, n=int(len(idx)), z=POT["top_z"] + (s + 0.5) * P["slab_mm"] / 1000)
    return out


def huber_poly(z, v, deg=2, it=20):
    w = np.ones(len(z)); c = np.polyfit(z, v, deg, w=w)
    for _ in range(it):
        r = v - np.polyval(c, z); s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9; u = np.abs(r) / (P["huber"] * s); w = np.where(u <= 1, 1.0, 1.0 / u)
        c = np.polyfit(z, v, deg, w=np.sqrt(w))
    return c


def s0_axis(cores):
    ks = sorted(cores); foot_s = ks[0]; fr = cores[foot_s]["radius"]; apex_s = max((s for s in ks if cores[s]["radius"] < P["apex_mult"] * fr), default=foot_s)
    use = [s for s in ks if s <= apex_s]; z = np.array([cores[s]["z"] for s in use]); C = np.array([cores[s]["centre"] for s in use])
    cx = huber_poly(z, C[:, 0], 2 if len(z) >= 4 else 1); cy = huber_poly(z, C[:, 1], 2 if len(z) >= 4 else 1)
    zs = np.arange(POT["top_z"], cores[apex_s]["z"] + 1e-9, P["slab_mm"] / 1000); pl = np.c_[np.polyval(cx, zs), np.polyval(cy, zs), zs]
    foot_off = float(np.hypot(*(pl[0, :2] - POT["centre"]))); foot_replaced = foot_off > POT["radius"]
    if foot_replaced: pl[0, :2] = POT["centre"]
    hidden_below = float(cores[foot_s]["z"])
    return dict(polyline=pl, foot_offset_m=foot_off, foot_replaced=bool(foot_replaced), apex_slab=int(apex_s), foot_slab=int(foot_s), foot_core_radius=fr, hidden_below_z=hidden_below)


# ---------------------------------------------------------------- S0-P: path + height + radius constraints
def s0p_nodes(xyz):
    zi = slabs(xyz); nodes = []
    for s in sorted(set(zi.tolist())):
        idx = np.nonzero(zi == s)[0]
        if len(idx) < P["slab_min"]: continue
        k = max(2, int(np.ceil(P["core_share"] * len(idx)))); rem = idx.copy()
        for _ in range(P["node_k3"]):
            if len(rem) < k: break
            m, c, r = tight_core(xyz[rem, :2], k); nodes.append(dict(slab=int(s), xy=c, z=float(np.median(xyz[rem[m], 2])), radius=r, n=int(m.sum()))); rem = rem[~m]
    return nodes


def dijkstra(nodes, src):
    n = len(nodes); dist = np.full(n, np.inf); prev = np.full(n, -1); dist[src] = 0; h = [(0.0, src)]
    pos = np.array([[*nd["xy"], nd["z"]] for nd in nodes]); sl = np.array([nd["slab"] for nd in nodes]); rad = np.array([nd["radius"] for nd in nodes])
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]: continue
        for v in np.nonzero((np.abs(sl - sl[u]) <= 1))[0]:
            if v == u: continue
            L = float(np.linalg.norm(pos[v] - pos[u]))
            if L > P["edge_mm"] / 1000: continue
            c = L * (1 + (rad[u] + rad[v]) / 2 * 1000 / P["radius_pen_mm"])
            if d + c < dist[v]: dist[v] = d + c; prev[v] = u; heapq.heappush(h, (d + c, int(v)))
    return dist, prev, pos, sl, rad


def ransac_circle(Q, rng):
    best = (0, None)
    for _ in range(P["ransac_it"]):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        A = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]]); B = 0.5 * np.array([b @ b - a @ a, c @ c - a @ a])
        if abs(np.linalg.det(A)) < 1e-14: continue
        ctr = np.linalg.solve(A, B); r = np.linalg.norm(a - ctr); inl = np.abs(np.linalg.norm(Q - ctr, axis=1) - r) <= P["ransac_tol_mm"] / 1000
        if inl.sum() > best[0]: best = (int(inl.sum()), (ctr, inl))
    if best[1] is None: return None
    ctr, inl = best[1]; return float(np.median(np.linalg.norm(Q[inl] - ctr, axis=1)))


def s0p(xyz, cores_s0):
    nodes = s0p_nodes(xyz); low = min(nd["slab"] for nd in nodes); cand = [i for i, nd in enumerate(nodes) if nd["slab"] == low]
    root = min(cand, key=lambda i: np.hypot(*(nodes[i]["xy"] - POT["centre"]))); dist, prev, pos, sl, rad = dijkstra(nodes, root)
    reach = np.nonzero(np.isfinite(dist))[0]; top = int(reach[np.argmax(pos[reach, 2])]); path = [top]
    while prev[path[-1]] >= 0: path.append(int(prev[path[-1]]))
    path = path[::-1]; fr = cores_s0[min(cores_s0)]["radius"]; on = set(path); cut = None
    for j in range(len(path) - 1, -1, -1):                       # walking down from the top: first junction with a blade branch
        u = path[j]; nb = [v for v in range(len(nodes)) if v not in on and abs(sl[v] - sl[u]) <= 1 and np.linalg.norm(pos[v] - pos[u]) <= P["edge_mm"] / 1000]
        if any(rad[v] >= P["apex_mult"] * fr for v in nb) and j < len(path) - 1:
            cut = j; break
    kept = path[: cut + 1] if cut is not None else path
    pl = np.array([pos[i] for i in kept]); zs = np.arange(POT["top_z"], pl[-1, 2] + 1e-9, P["slab_mm"] / 1000)
    order = np.argsort(pl[:, 2]); pz = pl[order]; poly = np.c_[np.interp(zs, pz[:, 2], pz[:, 0]), np.interp(zs, pz[:, 2], pz[:, 1]), zs]
    foot_off = float(np.hypot(*(poly[0, :2] - POT["centre"]))); foot_replaced = foot_off > POT["radius"]
    if foot_replaced: poly[0, :2] = POT["centre"]
    return dict(polyline=poly, nodes=len(nodes), path_nodes=len(path), cut_at_node=(None if cut is None else int(cut)), cut_height_m=(float(pos[path[cut], 2]) if cut is not None else None),
                foot_offset_m=foot_off, foot_replaced=bool(foot_replaced), foot_core_radius=fr, raw_top_z=float(pos[top, 2]))


def radius_constraint(xyz, poly, tube_r):
    rng = np.random.default_rng(P["c6_seed"]); d, tan = axis_dist(xyz, poly); radii = []
    z0 = poly[0, 2]; z1 = poly[-1, 2]; z = z0
    while z + P["win_mm"] / 1000 <= z1 + 1e-9:
        m = (xyz[:, 2] >= z) & (xyz[:, 2] < z + P["win_mm"] / 1000) & (d <= tube_r)
        if m.sum() >= P["win_min"]:
            seg = poly[(poly[:, 2] >= z) & (poly[:, 2] <= z + P["win_mm"] / 1000)]
            dirv = (seg[-1] - seg[0]) if len(seg) >= 2 else np.array([0, 0, 1.0]); dirv = dirv / max(np.linalg.norm(dirv), 1e-12)
            e1 = np.cross(dirv, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-6 else np.cross(dirv, [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(dirv, e1)
            Q = np.c_[xyz[m] @ e1, xyz[m] @ e2]; r = ransac_circle(Q, rng)
            if r is not None: radii.append(dict(z_m=float(z), n=int(m.sum()), radius_m=r))
        z += P["win_step_mm"] / 1000
    rmin = min((w["radius_m"] for w in radii), default=None); return rmin, radii


# ---------------------------------------------------------------- geometry helpers
def axis_dist(xyz, poly):
    """Distance of points to the polyline and the unit tangent of the nearest segment."""
    best = np.full(len(xyz), np.inf); tan = np.zeros((len(xyz), 3))
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a; L2 = max(float(ab @ ab), 1e-18); t = np.clip(((xyz - a) @ ab) / L2, 0, 1); d = np.linalg.norm(xyz - (a + t[:, None] * ab), axis=1)
        m = d < best; best[m] = d[m]; tan[m] = ab / np.sqrt(L2)
    return best, tan


def curve_length(poly):
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def checks(poly, xyz, fallback_needed_out):
    foot = poly[0]; c2 = bool(np.hypot(*(foot[:2] - POT["centre"])) <= POT["radius"] + 1e-9 and foot[2] >= POT["top_z"] - 1e-9)
    hull = Delaunay(xyz); inside = hull.find_simplex(poly) >= 0; c3 = bool(inside.mean() >= P["c3_share"])
    seg = np.diff(poly, axis=0); seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
    ang = np.degrees(np.arccos(np.clip((seg[:-1] * seg[1:]).sum(1), -1, 1))) if len(seg) > 1 else np.zeros(0); c4 = bool((ang < P["c4_deg"]).all())
    fallback_needed_out["c4_max_deg"] = float(ang.max()) if len(ang) else 0.0
    return dict(C2=c2, C3=c3, C3_inside_share=round(float(inside.mean()), 3), C4=c4, C4_max_turn_deg=round(fallback_needed_out["c4_max_deg"], 1))


def axis_gap_mm(p1, p2):
    z = np.intersect1d(np.round(p1[:, 2], 5), np.round(p2[:, 2], 5))
    if not len(z): return None
    a = np.array([p1[np.argmin(np.abs(p1[:, 2] - zz)), :2] for zz in z]); b = np.array([p2[np.argmin(np.abs(p2[:, 2] - zz)), :2] for zz in z])
    return round(float(np.median(np.linalg.norm(a - b, axis=1)) * 1000), 2)


def circle_fit(Q):
    A = np.c_[2 * Q, np.ones(len(Q))]; b = (Q ** 2).sum(1); sol, *_ = np.linalg.lstsq(A, b, rcond=None); c = sol[:2]; return float(np.sqrt(max(sol[2] + c @ c, 0)))


# ---------------------------------------------------------------- main
def run(out: Path):
    t0 = time.time(); out.mkdir(parents=True, exist_ok=True)
    srcs = dict(surfels=PLY, cleanup=ROOT / "build/p1_vp_cleanup_20261026_01/surfel_clusters.npz", traces=ROOT / "build/p1_compound_2dvote_20261025_01/traces_oriented.json",
                visibility=ROOT / "build/p1_compound_2dvote_20261025_01/stage_a_h2id.npz", poses=LO.DEFAULTS["poses"], pot_spec=ROOT / "data/records/pots/pot_spec.csv",
                stem_height=ROOT / "data/records/stem_height/stem_height.csv")
    json.dump({k: dict(path=str(v), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(v)))) for k, v in srcs.items()}, open(out / "inputs.json", "w"), indent=1)
    D = load(); xyz, nrm = D["xyz"], D["nrm"]; n = len(xyz)
    # S0
    cores = s0_cores(xyz); A0 = s0_axis(cores)
    # S0-P
    AP = s0p(xyz, cores)
    # C2-C4 per variant (+ fallback)
    res = {}
    for name, A in (("S0", A0), ("S0P", AP)):
        fb = {}; c = checks(A["polyline"], xyz, fb); poly = A["polyline"]; fallback = False
        if not c["C4"]:
            fallback = True; poly = np.linspace(poly[0], poly[-1], len(poly))
        res[name] = dict(axis=A, checks=c, fallback=fallback, polyline=poly)
    # C6 (S0 rebuilt from a random half of the views; S0-P likewise)
    H = np.load(srcs["visibility"])["H2id"]; assert len(H) == n, "visibility rows differ from plant rows"
    rng = np.random.default_rng(P["c6_seed"]); half = rng.choice(H.shape[1], H.shape[1] // 2, replace=False); vis = (H[:, half] >= 0).any(1)
    c0h = s0_cores(xyz[vis]); A0h = s0_axis(c0h); APh = s0p(xyz[vis], c0h)
    res["S0"]["C6_mm"] = axis_gap_mm(res["S0"]["polyline"], A0h["polyline"]); res["S0P"]["C6_mm"] = axis_gap_mm(res["S0P"]["polyline"], APh["polyline"])
    sp = res["S0P"]; s0 = res["S0"]
    use = "S0P" if (sp["checks"]["C2"] and sp["checks"]["C3"] and sp["checks"]["C4"] and sp["C6_mm"] is not None and s0["C6_mm"] is not None and sp["C6_mm"] <= s0["C6_mm"]) else "S0"
    poly = res[use]["polyline"]; foot_r = res[use]["axis"]["foot_core_radius"]
    # S1
    d_ax, tan = axis_dist(xyz, poly); zi = slabs(xyz); core_r = np.array([cores.get(int(s), {}).get("radius", foot_r) for s in zi])
    tube_r = np.maximum(P["tube_mult"] * core_r, P["tube_min_mm"] / 1000); in_tube = d_ax <= tube_r
    rmin = None; windows = []
    if use == "S0P":
        rmin, windows = radius_constraint(xyz, poly, tube_r.max())
        if rmin is not None: in_tube &= d_ax <= rmin + P["margin_mm"] / 1000
    nd = np.abs((nrm * tan).sum(1)); blade = ~in_tube; blade_med = float(np.median(nd[blade])); zl = np.load(srcs["cleanup"], allow_pickle=True)
    leaf = np.where(zl["leaf"] > 0, zl["leaf"], 0); assert len(leaf) == n; labelled_leaf = (leaf > 0) & (leaf < NONLEAF)
    stem = in_tube & (nd < blade_med) & ~labelled_leaf; amb = in_tube & ~stem
    t = cKDTree(xyz); lin = np.zeros(n)
    for i in np.nonzero(in_tube)[0]:
        nb = t.query_ball_point(xyz[i], 0.005)
        if len(nb) >= 5:
            w = np.sort(np.linalg.eigvalsh(np.cov((xyz[nb] - xyz[nb].mean(0)).T)))[::-1]; lin[i] = (w[0] - w[1]) / max(w[0], 1e-18)
    blade_rgb = np.median(D["rgb"][blade], axis=0); cdist = np.linalg.norm(D["rgb"] - blade_rgb, axis=1)
    # S2 nodes
    nodes = []
    tr = json.load(open(srcs["traces"], encoding="utf-8")) if srcs["traces"].exists() else None
    if tr:
        from tomato3d.organs import compound_cluster as CC
        raw = {r["key"]: r for r in CC.load_traces(ROOT / "build/p1_compound_2dvote_20261025_01/traces_export/rachis_traces")}
        for r in tr["traces"]:
            pts = np.asarray(raw[r["key"]]["pts"], float); base = pts[-1] if r["flipped"] else pts[0]
            nodes.append(dict(height_mm=round((base[2] - POT["top_z"]) * 1000, 1), source="trace", ref=r["key"], leaf_trace=r["leaf"]))
    for L in sorted(set(leaf[labelled_leaf].tolist())):
        idx = np.nonzero(leaf == L)[0]; near = idx[np.argsort(d_ax[idx])[: P["leaf_near"]]]; c = xyz[near].mean(0)
        nodes.append(dict(height_mm=round((c[2] - POT["top_z"]) * 1000, 1), source="leaf_label", ref=f"leaf {L}", usable=(L in (1, 7))))
    ks = sorted(cores); rr = np.array([cores[s]["radius"] for s in ks])
    for j, s in enumerate(ks):
        nb = [rr[k] for k in range(max(0, j - 2), min(len(ks), j + 3)) if k != j]
        if nb and rr[j] > P["widen"] * np.median(nb) and cores[s]["z"] <= poly[-1, 2]: nodes.append(dict(height_mm=round((cores[s]["z"] - POT["top_z"]) * 1000, 1), source="profile", ref=f"slab {s}"))
    leaf_nodes_premerge = [x for x in nodes if x["source"] == "leaf_label" and x.get("usable")]   # A3 uses each usable leaf's own node
    merged = []
    for nd_ in sorted(nodes, key=lambda x: (x["source"], x["height_mm"])):
        if merged and merged[-1]["source"] == nd_["source"] and abs(merged[-1]["height_mm"] - nd_["height_mm"]) <= P["merge_mm"]: continue
        merged.append(nd_)
    nodes = sorted(merged, key=lambda x: x["height_mm"])
    # S3 diameter
    diam = []; last = None
    for s in ks:
        m = stem & (zi == s)
        if m.sum() >= P["s3_min"]: r = circle_fit(xyz[m, :2]); last = r; diam.append(dict(slab=int(s), z_mm=round((cores[s]["z"] - POT["top_z"]) * 1000, 1), n=int(m.sum()), radius_mm=round(r * 1000, 2), carried=False))
        else: diam.append(dict(slab=int(s), z_mm=round((cores[s]["z"] - POT["top_z"]) * 1000, 1), n=int(m.sum()), radius_mm=None, carried=True))
    valid = [d for d in diam if not d["carried"]]
    for d in diam:
        if d["carried"] and valid: d["radius_mm"] = min(valid, key=lambda v: abs(v["slab"] - d["slab"]))["radius_mm"]
    # A2 diameter at a fixed height
    a2 = None
    for hmm in P["a2_heights_mm"]:
        zc = POT["top_z"] + hmm / 1000; m = stem & (np.abs(xyz[:, 2] - zc) <= P["a2_band_mm"] / 2000)
        if m.sum() >= 40:
            Q = xyz[m, :2]; w, v = np.linalg.eigh(np.cov((Q - Q.mean(0)).T)); ex = [float(np.ptp((Q - Q.mean(0)) @ v[:, k])) for k in (1, 0)]
            a2 = dict(height_mm=hmm, n=int(m.sum()), circle_diameter_mm=round(2 * circle_fit(Q) * 1000, 2), extents_mm=[round(e * 1000, 2) for e in ex], mean_two_directions_mm=round(float(np.mean(ex)) * 1000, 2), tag="diameter_at_fixed_height"); break
    if a2 is None:
        a2 = dict(status="unavailable", reason=f"fewer than 40 stem surfels in the 6 mm band at 15, 20 and 10 mm above the pot top (lowest plant surfel at {round((xyz[:, 2].min() - POT['top_z']) * 1000, 1)} mm: the stem base is not reconstructed)",
                  counts={str(h): int((stem & (np.abs(xyz[:, 2] - (POT['top_z'] + h / 1000)) <= P['a2_band_mm'] / 2000)).sum()) for h in P["a2_heights_mm"]})
    # A3 leaf insertion angle (leaves 1 and 7)
    a3 = []
    for nd_ in leaf_nodes_premerge:
        if True:
            L = int(nd_["ref"].split()[-1]); zc = POT["top_z"] + nd_["height_mm"] / 1000; k = int(np.argmin(np.abs(poly[:, 2] - zc))); node_p = poly[k]
            tg = poly[min(k + 1, len(poly) - 1)] - poly[max(k - 1, 0)]; tg /= max(np.linalg.norm(tg), 1e-12)
            idx = np.nonzero(leaf == L)[0]; near = idx[np.linalg.norm(xyz[idx] - node_p, axis=1) <= P["a3_mm"] / 1000]
            if len(near) < 5: a3.append(dict(leaf=L, status="unavailable", reason="fewer than 5 leaf surfels within 20 mm of the node")); continue
            pv = xyz[near].mean(0) - node_p; ang = float(np.degrees(np.arccos(np.clip(pv @ tg / max(np.linalg.norm(pv), 1e-12), -1, 1))))
            a3.append(dict(leaf=L, node_height_mm=nd_["height_mm"], angle_deg=round(ang, 1), n=int(len(near)), tag="leaf_insertion_angle_estimated"))
    # tags
    c = res[use]["checks"]; c6 = res[use]["C6_mm"]; clear = float(np.mean([cores[s]["radius"] < P["apex_mult"] * foot_r for s in ks if cores[s]["z"] <= poly[-1, 2]]))
    if res[use]["fallback"] or not (c["C2"] and c["C3"] and c["C4"]): tag = "stem_inferred_only"
    elif c6 is not None and c6 <= P["visible_tag_mm"] and clear >= P["clear_share"]: tag = "stem_visible_supported"
    else: tag = "stem_partly_hidden"
    hidden_z = res[use]["axis"].get("hidden_below_z", poly[0, 2])
    height = float(poly[-1, 2] - POT["top_z"]); length = curve_length(poly)
    axis_json = dict(individual="R3-1b", input_status="R3-1b_unaccepted", variant_used=use, polyline_m=poly.round(5).tolist(), foot_m=poly[0].round(5).tolist(), apex_m=poly[-1].round(5).tolist(),
                     stem_height_mm=round(height * 1000, 1), stem_length_mm=round(length * 1000, 1), hidden_below_height_mm=round((hidden_z - POT["top_z"]) * 1000, 1),
                     core_radius_per_slab=[dict(slab=int(s), z_mm=round((cores[s]["z"] - POT["top_z"]) * 1000, 1), radius_mm=round(cores[s]["radius"] * 1000, 2), n=cores[s]["n"]) for s in ks],
                     tag=tag, clear_slab_share=round(clear, 3), checks=dict(C1="no ruler value", **c, C6_mm=c6), stem_radius_constraint_mm=(round(rmin * 1000, 2) if rmin else None),
                     tags=dict(TAGS, model_generated=True, hidden_part="conditional_hidden_state"), pot=POT, frame="board metres, z up")
    var = {k: dict(checks=v["checks"], C6_mm=v["C6_mm"], fallback=v["fallback"], stem_height_mm=round((v["polyline"][-1, 2] - POT["top_z"]) * 1000, 1), stem_length_mm=round(curve_length(v["polyline"]) * 1000, 1),
                   foot_offset_mm=round(v["axis"]["foot_offset_m"] * 1000, 1), foot_replaced=v["axis"]["foot_replaced"]) for k, v in res.items()}
    var["S0P"].update(cut_height_mm=(round((AP["cut_height_m"] - POT["top_z"]) * 1000, 1) if AP["cut_height_m"] else None), min_radius_mm=(round(rmin * 1000, 2) if rmin else None), nodes=AP["nodes"], path_nodes=AP["path_nodes"],
                      raw_top_mm=round((AP["raw_top_z"] - POT["top_z"]) * 1000, 1), radius_windows=windows)
    axis_json["variants"] = var; axis_json["diameter_at_fixed_height"] = a2; axis_json["leaf_insertion_angles"] = a3; axis_json["S4"] = "skipped: Tool A is a rachis tracer inside the R3-1b compound viewer, not reusable as is"
    for sub in ("stem_axis", "stem_nodes", "stem_surfels", "stem_diameter", "qa"): (out / sub).mkdir(exist_ok=True)
    json.dump(axis_json, open(out / "stem_axis" / "R3-1b.json", "w", encoding="utf-8"), indent=1, default=float)
    with open(out / "stem_nodes" / "R3-1b.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["node_index", "height_mm", "source", "ref", "tag"])
        for i, nd_ in enumerate(nodes): w.writerow([i, nd_["height_mm"], nd_["source"], nd_["ref"], ("model_generated;" + TAGS["scale"] + (";usable_leaf" if nd_.get("usable") else ""))])
    np.savez_compressed(out / "stem_surfels" / "R3-1b.npz", stem_estimated=D["rows"][stem], unassigned=D["rows"][amb], note=np.array("indices into R3-1b_surfels_full.ply"))
    with open(out / "stem_diameter" / "R3-1b.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slab", "z_mm", "n", "radius_mm", "carried"], lineterminator="\n"); w.writeheader(); w.writerows(diam)
    s1 = dict(tube_surfels=int(in_tube.sum()), stem_estimated=int(stem.sum()), unassigned=int(amb.sum()), stem_share=round(float(stem.sum() / max(in_tube.sum(), 1)), 3),
              ambiguous_share=round(float(amb.sum() / max(in_tube.sum(), 1)), 3), blade_median_abs_n_dot_axis=round(blade_med, 3),
              linearity_median=dict(stem=round(float(np.median(lin[stem])), 3) if stem.any() else None, unassigned=round(float(np.median(lin[amb])), 3) if amb.any() else None),
              colour_distance_median=dict(stem=round(float(np.median(cdist[stem])), 1) if stem.any() else None, blade=round(float(np.median(cdist[blade])), 1)))
    json.dump(dict(s1=s1, nodes=nodes, diameter=dict(foot=next((d for d in diam if not d["carried"]), None), near_apex=next((d for d in reversed(diam) if not d["carried"] and d["z_mm"] <= height * 1000), None))),
              open(out / "stem_extra_R3-1b.json", "w", encoding="utf-8"), indent=1, default=float)
    overlays(out, poly, nodes, hidden_z)
    print(json.dumps(dict(use=use, tag=tag, height_mm=round(height * 1000, 1), length_mm=round(length * 1000, 1), variants={k: {kk: vv for kk, vv in v.items() if kk != "radius_windows"} for k, v in var.items()},
                          s1=s1, nodes=len(nodes), a2=a2, a3=a3, seconds=round(time.time() - t0)), indent=1, default=float))
    return axis_json, var, s1, nodes, a2


def overlays(out: Path, poly, nodes, hidden_z, px=520):
    po = json.load(open(LO.DEFAULTS["poses"])); views = json.load(open(LO.DEFAULTS["views"]))["train"]; K0 = np.array(po["K_undist"]); R0 = po["undistorted_res"]
    photos = LO.DEFAULTS["poses"].parent / f"undist_{R0}"; tiles = []
    nz = [POT["top_z"] + n_["height_mm"] / 1000 for n_ in nodes]; npts = [poly[int(np.argmin(np.abs(poly[:, 2] - z)))] for z in nz]
    for v in views:
        img = Image.open(photos / v).convert("RGB"); s = px / max(img.size); img = img.resize((int(img.size[0] * s), int(img.size[1] * s))); dr = ImageDraw.Draw(img)
        K = ca.scaled_K(K0, R0, int(R0 * s)); R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"])
        def pr(X):
            xc = np.atleast_2d(X) @ R.T + t; return np.c_[K[0, 0] * xc[:, 0] / xc[:, 2] + K[0, 2], K[1, 1] * xc[:, 1] / xc[:, 2] + K[1, 2]]
        q = pr(poly)
        for a, b, za in zip(q[:-1], q[1:], poly[:-1, 2]):
            dr.line([tuple(a), tuple(b)], fill=((255, 160, 0) if za < hidden_z else (255, 40, 40)), width=3)
        for p_, n_ in zip(pr(np.array(npts)) if npts else [], nodes):
            col = {"trace": (255, 230, 0), "leaf_label": (0, 220, 255), "profile": (200, 120, 255)}[n_["source"]]; dr.ellipse([p_[0] - 4, p_[1] - 4, p_[0] + 4, p_[1] + 4], fill=col, outline=(0, 0, 0))
        img.save(out / "qa" / f"R3-1b_{v[9:18]}.png"); tiles.append(v[9:18])
    html = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>R3-1b 줄기 축</title>"
            "<style>body{margin:0;padding:12px 16px;background:#111;color:#eee;font:13px system-ui,sans-serif}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}img{width:100%;display:block}</style></head><body>"
            "<h3>R3-1b_unaccepted · 줄기 축 (빨강 = 복원된 부분에 맞춘 축, 주황 = 아래 가려진 부분 추정, conditional_hidden_state) · 점: 노랑 = 엽축 밑동, 하늘 = 잎 라벨, 보라 = 굵기 변화 · conditional_on_unvalidated_scale · 채점 안 함</h3><div class='g'>"
            + "".join(f"<figure style='margin:0'><img src='R3-1b_{t_}.png' alt='{t_}'><figcaption>{t_}</figcaption></figure>" for t_ in tiles) + "</div></body></html>")
    (out / "qa" / "R3-1b_overlay.html").write_text(html, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args(argv); axis, var, s1, nodes, a2 = run(a.out)
    stopped = ["n_01", "n_02", "n_03", "n_04", "n_05", "n_06", "t2_02", "t2_04", "t2_05"]
    with open(a.out / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["individual", "status", "variant", "stem_height_mm", "stem_length_mm", "foot_offset_mm", "C1", "C2", "C3", "C4", "C5", "C6_mm", "tag", "node_count", "ambiguous_share", "scale"])
        c = axis["checks"]
        w.writerow(["R3-1b", "processed", axis["variant_used"], axis["stem_height_mm"], axis["stem_length_mm"], var[axis["variant_used"]]["foot_offset_mm"], c["C1"], c["C2"], c["C3"], c["C4"], "overlays written (not scored)", c["C6_mm"], axis["tag"], len(nodes), s1["ambiguous_share"], TAGS["scale"]])
        for s in stopped: w.writerow([s, "stopped: pot-top plane not located (recon_meta 'not fitted in this batch'; pot_spec covers dwt_1* only)", "", "", "", "", "", "", "", "", "", "", "", "", "", TAGS["scale"]])


if __name__ == "__main__":
    main()
