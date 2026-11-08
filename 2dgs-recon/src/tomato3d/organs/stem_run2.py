"""Stem run 2 (contract build/p1_stem_run2_20261026_01/CONTRACT.md): S5 bottom-up cylinder chain scored on photo masks (E2D) and surfels (E3D),
S6 rachis-predicted nodes, S7 refit above h_lost, node checks, C3 / N1, overlays. CPU only.

usage: python -m tomato3d.organs.stem_run2 [--ids R3-1b n_03 ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay, cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import light_organs as LO

ROOT = LO.ROOT
OUT = ROOT / "build/p1_stem_run2_20261026_01"
RIM = ROOT / "build/p1_plant_model_20261026_01/rim"
ALL = ROOT / "build/p1_recon_all_20261025_01"
SRC = {"R3-1b": dict(ply=ROOT / "build/p1_recon_r3-1_contam_20261022_01/outputs/R3-1b_surfels_full.ply", poses=ROOT / "build/p1_charuco_a3_20261021_01/poses.json",
                     masks=ROOT / "build/p1_recon_r3-1_20261022_01/masks_b", photos=ROOT / "build/p1_charuco_a3_20261021_01/undist_3024"),
       "n_03": dict(poses=ALL / "stage2_new/n_03/poses.json"), "n_06": dict(poses=ALL / "stage2_new_n06/n_06/poses.json"),
       "t2_02": dict(poses=ROOT / "build/p1_recon_tomato2_20261023_01/stage2_d7/t2_02/poses.json"), "t2_05": dict(poses=ROOT / "build/p1_recon_tomato2_20261023_01/stage2_d7/t2_05/poses.json")}
for k in ("n_03", "n_06", "t2_02", "t2_05"):
    SRC[k].update(ply=ALL / f"outputs/{k}/R3-1b-style_{k}_surfels_full.ply", masks=ALL / f"k3a/{k}/masks")
TAGS = dict(scale="conditional_on_unvalidated_scale", biological_validation="not_established")
W = dict(e2d=0.5, e3d=0.3, straight=0.2)
S5 = dict(seg_mm=6.0, cone_deg=25.0, tilts=(5, 10, 15, 20, 25), az=8, r0_mm=1.5, r_min_mm=0.5, r_mult=(0.9, 1.0), e2d_acc=0.6, e3d_acc=0.5, max_unsup=3, min_w_px=2.0,
          along=7, across=5, margin_mm=1.5, perp_cos=np.cos(np.radians(60)), dens_mm=3.0, top_pct=99.5,
          branch_tilts=(40, 55, 70, 85, 100), branch_az=12, branch_segs=2)
S6 = dict(pet_lo_mm=5.0, pet_hi_mm=25.0, resid_max_mm=6.0, frag_min=80, sigma_min_mm=1.0)
S7 = dict(lam=1.0, iters=3, move_mm=1.0)
CHK = dict(c3=0.95, n1_mm=8.0, n1_min=3, short_mm=2.0, visible_share=0.60)
MASK_RES = 1512


# ---------------------------------------------------------------- inputs
def load(id_):
    s = SRC[id_]; rim = json.load(open(RIM / f"{id_}.json", encoding="utf-8")); med = float(rim["rim_z_m"]) - 0.010; ctr = np.array(rim["rim_centre_xy_m"], float)
    sf = sio.read_surfels_full(s["ply"]); keep = (sf["opacity"] >= LO.OPACITY_MIN) & (sf["xyz"][:, 2] > med); rows = np.nonzero(keep)[0]
    nrm = sf["normal"][rows].astype(float); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    po = json.load(open(s["poses"])); cams = []
    for v, d in po["views"].items():
        if id_ != "R3-1b" and not d.get("accepted", True): continue
        mf = s["masks"] / (Path(v).stem + ".png")
        if not mf.exists(): continue
        if id_ == "R3-1b":
            K = np.array(po["K_undist"], float); size = po["undistorted_res"]
        else:
            K = np.array(d["K"], float); size = d["size"][0]
        K = K.copy(); K[:2] *= MASK_RES / size
        lab = np.asarray(Image.open(mf)); m = np.isin(lab, (1, 2)) if id_ == "R3-1b" else lab > 0
        cams.append(dict(view=v, K=K, R=np.array(d["R_cw"], float), t=np.array(d["t_cw"], float), mask=m))
    return dict(id=id_, rows=rows, xyz=sf["xyz"][rows].astype(float), nrm=nrm, medium=med, centre=ctr, rim=rim, cams=cams, src=s)


# ---------------------------------------------------------------- scores
def e2d(seg_a, seg_b, r, cams):
    """Pooled share of silhouette samples inside the plant masks over views where the segment is in frame and >= 2 px wide."""
    t = np.linspace(0, 1, S5["along"]); P = seg_a[None] + t[:, None] * (seg_b - seg_a)[None]; inside = tot = 0
    for c in cams:
        xc = P @ c["R"].T + c["t"]; z = xc[:, 2]
        if (z <= 0).any(): continue
        u = c["K"][0, 0] * xc[:, 0] / z + c["K"][0, 2]; w = c["K"][1, 1] * xc[:, 1] / z + c["K"][1, 2]
        if (u < 0).any() or (w < 0).any() or (u >= MASK_RES).any() or (w >= MASK_RES).any(): continue
        half = r * c["K"][0, 0] / z
        if 2 * half.mean() < S5["min_w_px"]: continue
        d = np.array([u[-1] - u[0], w[-1] - w[0]]); nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        s = np.linspace(-1, 1, S5["across"]); U = (u[:, None] + s[None] * half[:, None] * nrm[0]).round().astype(int); V = (w[:, None] + s[None] * half[:, None] * nrm[1]).round().astype(int)
        ok = (U >= 0) & (V >= 0) & (U < MASK_RES) & (V < MASK_RES); inside += int(c["mask"][V[ok], U[ok]].sum()); tot += int(ok.sum())
    return (inside / tot if tot else 0.0), tot


def e3d(seg_a, seg_b, r, xyz, nrm, tree, rho):
    L = np.linalg.norm(seg_b - seg_a); ax = (seg_b - seg_a) / max(L, 1e-12); mid = (seg_a + seg_b) / 2
    idx = tree.query_ball_point(mid, np.hypot(L / 2, r + S5["margin_mm"] / 1000))
    if not idx: return 0.0
    idx = np.array(idx); d = xyz[idx] - seg_a; a = d @ ax; rad = np.linalg.norm(d - np.outer(a, ax), axis=1)
    m = (a >= 0) & (a <= L) & (rad <= r + S5["margin_mm"] / 1000) & (np.abs(nrm[idx] @ ax) <= S5["perp_cos"])
    exp = rho * 2 * np.pi * r * L; return float(min(1.0, m.sum() / max(exp, 1e-9)))


def cone(dirv, tilts, naz):
    e1 = np.cross(dirv, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-6 else np.cross(dirv, [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(dirv, e1); out = []
    for tdeg in tilts:
        tr = np.radians(tdeg)
        for k in range(naz):
            ph = 2 * np.pi * k / naz; out.append(np.cos(tr) * dirv + np.sin(tr) * (np.cos(ph) * e1 + np.sin(ph) * e2))
    return out


# ---------------------------------------------------------------- S5
def s5_chain(D, start_xy=None):
    xyz, nrm = D["xyz"], D["nrm"]; tree = cKDTree(xyz); cnt = np.array([len(x) for x in tree.query_ball_point(xyz[::7], S5["dens_mm"] / 1000)])
    rho = float(np.median(cnt) / (np.pi * (S5["dens_mm"] / 1000) ** 2)); top = float(np.percentile(xyz[:, 2], S5["top_pct"]))
    sx = D["centre"] if start_xy is None else start_xy
    p = np.array([sx[0], sx[1], D["medium"]]); dirv = np.array([0, 0, 1.0]); r = S5["r0_mm"] / 1000; L = S5["seg_mm"] / 1000
    segs, nodes, unsup = [], [], 0
    while p[2] < top and len(segs) < 60:
        best = None
        for dv in [dirv] + cone(dirv, S5["tilts"], S5["az"]):
            for rm in S5["r_mult"]:
                rr = max(r * rm, S5["r_min_mm"] / 1000); q = p + L * dv; a2, n2 = e2d(p, q, rr, D["cams"]); a3 = e3d(p, q, rr, xyz, nrm, tree, rho); st = float(dv @ dirv)
                sc = W["e2d"] * a2 + W["e3d"] * a3 + W["straight"] * st
                if best is None or sc > best["score"]: best = dict(start=p.tolist(), end=q.tolist(), dir=dv, radius_m=rr, E2D=round(a2, 3), E3D=round(a3, 3), straight=round(st, 3), score=round(sc, 3), samples=n2)
        sup = best["E2D"] >= S5["e2d_acc"] or best["E3D"] >= S5["e3d_acc"]; best["supported"] = bool(sup); segs.append(best)
        if sup:
            unsup = 0; bb = None                                       # branch test at this segment end
            for dv in cone(best["dir"], S5["branch_tilts"], S5["branch_az"]):
                q1 = np.array(best["end"]) + L * dv; q2 = q1 + L * dv; a1, _ = e2d(np.array(best["end"]), q1, best["radius_m"], D["cams"]); a2b, _ = e2d(q1, q2, best["radius_m"], D["cams"])
                if a1 >= S5["e2d_acc"] and a2b >= S5["e2d_acc"] and (bb is None or a1 + a2b > bb[0]): bb = (a1 + a2b, dv)
            if bb is not None:
                nodes.append(dict(xyz=best["end"], z_mm=round((best["end"][2] - D["medium"]) * 1000, 1), source="branch_seen", petiole_dir=bb[1].tolist(), sigma_lat_mm=None, sigma_z_mm=S5["seg_mm"] / 2))
        else:
            unsup += 1
        p = np.array(best["end"]); dirv = best["dir"]; r = best["radius_m"]
        if unsup >= S5["max_unsup"]: break
    last_sup = max((i for i, s in enumerate(segs) if s["supported"]), default=-1)
    h_lost = (segs[last_sup]["end"][2] if last_sup >= 0 else D["medium"]); chain = segs[: last_sup + 1]
    for s in segs: s["dir"] = list(np.round(s["dir"], 4))
    return dict(segments=segs, chain=chain, h_lost_z=h_lost, top_z=top, rho=rho, nodes_branch=nodes)


# ---------------------------------------------------------------- S6 / S7
def rachis_lines(D):
    """R3-1b only: traces (oriented) and clean-up leaves 1, 7 (line through leaflet-cluster centroids)."""
    if D["id"] != "R3-1b": return []
    from tomato3d.organs import compound_cluster as CC
    out = []; tr = json.load(open(ROOT / "build/p1_compound_2dvote_20261025_01/traces_oriented.json", encoding="utf-8"))
    raw = {r["key"]: r for r in CC.load_traces(ROOT / "build/p1_compound_2dvote_20261025_01/traces_export/rachis_traces")}
    for r in tr["traces"]:
        pts = np.asarray(raw[r["key"]]["pts"], float); pts = pts[::-1] if r["flipped"] else pts
        base = pts[0]; d = pts[0] - pts[1]; d /= max(np.linalg.norm(d), 1e-12); out.append(dict(key=r["key"], source="trace", leaf=f"trace {r['leaf']}", base=base, dir=d, resid_mm=0.0, line=pts))
    zl = np.load(ROOT / "build/p1_compound_userlabel_20261025_01/map_v3_charuco/leaflet_instances_a3v2_within_user_leaf/surfel_leaflets.npz", allow_pickle=True)
    sf = sio.read_surfels_full(SRC["R3-1b"]["ply"]); rowpos = {int(rw): i for i, rw in enumerate(zl["rows"])}
    for L in (1, 7):
        cents = []
        for k in np.unique(zl["leaflet"][zl["leaf"] == L]):
            m = (zl["leaflet"] == k)
            if m.sum() >= S6["frag_min"]: cents.append(sf["xyz"][zl["rows"][m]].mean(0))
        if len(cents) < 2: out.append(dict(key=f"label{L}", source="label", leaf=f"label {L}", rejected="fewer than 2 leaflet clusters")); continue
        C = np.array(cents, float); c0 = C.mean(0); w, v = np.linalg.eigh(np.cov((C - c0).T)); d = v[:, -1]; res = float(np.sqrt(np.mean(np.linalg.norm((C - c0) - np.outer((C - c0) @ d, d), axis=1) ** 2)) * 1000)
        axis_xy = np.array([D["centre"][0], D["centre"][1]]); dist = np.linalg.norm(C[:, :2] - axis_xy, axis=1); base = c0 + ((C[np.argmin(dist)] - c0) @ d) * d
        dd = -d if (c0 - base) @ d < 0 else d; inward = -dd
        if res > S6["resid_max_mm"]: out.append(dict(key=f"label{L}", source="label", leaf=f"label {L}", rejected=f"residual {res:.1f} mm > 6 mm")); continue
        out.append(dict(key=f"label{L}", source="label", leaf=f"label {L}", base=base, dir=inward, resid_mm=round(res, 2), line=C))
    return out


def stem_polyline(chain, upper):
    pts = [np.array(chain[0]["start"])] + [np.array(s["end"]) for s in chain] if chain else []
    return np.array(pts + [np.array(p) for p in upper[1:]]) if len(pts) else np.array(upper)


def closest(line_p, line_d, poly):
    """Closest approach between the ray p + s d (s in [lo, hi]) and a polyline; returns (point on ray, point on stem, s, distance)."""
    best = None
    for a, b in zip(poly[:-1], poly[1:]):
        for s in np.linspace(S6["pet_lo_mm"] / 1000 * 0, S6["pet_hi_mm"] / 1000, 51):
            q = line_p + s * line_d; ab = b - a; t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-18), 0, 1); c = a + t * ab; d = np.linalg.norm(q - c)
            if best is None or d < best[3]: best = (q, c, s, d)
    return best


def upper_fit(z0, p0, apex_z, targets):
    """Least-squares x(z), y(z) above h_lost: node targets (weights 1/sigma^2) + lam * second differences (mm units)."""
    zs = np.arange(z0, apex_z + 1e-9, S5["seg_mm"] / 1000)
    if len(zs) < 3: zs = np.array([z0, (z0 + apex_z) / 2, apex_z])
    n = len(zs); out = np.zeros((n, 3)); out[:, 2] = zs
    for k in range(2):
        rows, rhs = [], []
        r0 = np.zeros(n); r0[0] = 1e3; rows.append(r0); rhs.append(1e3 * p0[k] * 1000)
        for t in targets:
            j = int(np.argmin(np.abs(zs - t["z"]))); rr = np.zeros(n); wgt = 1.0 / max(t["sigma_mm"], 1e-6); rr[j] = wgt; rows.append(rr); rhs.append(wgt * t["xy"][k] * 1000)
        for j in range(1, n - 1):
            rr = np.zeros(n); rr[j - 1], rr[j], rr[j + 1] = 1, -2, 1; rows.append(np.sqrt(S7["lam"]) * rr); rhs.append(0.0)
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None); out[:, k] = sol / 1000
    return out


def s6_s7(D, s5, lines, drop=None):
    chain = s5["chain"]; last = np.array(chain[-1]["end"]) if chain else np.array([D["centre"][0], D["centre"][1], D["medium"]])
    upper = np.array([last, [last[0], last[1], s5["top_z"]]]); hist = []; nodes = []
    for it in range(S7["iters"] + 1):
        poly = stem_polyline(chain, upper); nodes = []
        for ln in lines:
            if "rejected" in ln: continue
            q, c, s, d = closest(ln["base"], ln["dir"], poly)
            nodes.append(dict(key=ln["key"], source="node_rachis_predicted:" + ln["source"], leaf=ln["leaf"], target=q, on_stem=c, z=float(c[2]), dist_mm=round(d * 1000, 2), extension_mm=round(s * 1000, 1),
                              sigma_lat_mm=max(ln["resid_mm"], S6["sigma_min_mm"]), sigma_z_mm=(S6["pet_hi_mm"] - S6["pet_lo_mm"]) / 2, petiole_dir=(-ln["dir"]).tolist()))
        if it == S7["iters"]: break
        targets = [dict(z=n_["target"][2], xy=n_["target"][:2], sigma_mm=n_["sigma_lat_mm"]) for n_ in nodes if n_["key"] != drop and n_["target"][2] >= last[2]]
        new_upper = upper_fit(last[2], last, s5["top_z"], targets) if targets else upper
        move = float(np.max(np.linalg.norm(new_upper[:, :2] - np.array([np.interp(new_upper[:, 2], upper[:, 2], upper[:, k]) for k in range(2)]).T, axis=1))) * 1000 if len(upper) >= 2 else 0.0
        hist.append(dict(iteration=it + 1, targets=len(targets), max_move_mm=round(move, 2))); upper = new_upper
        if move < S7["move_mm"]: break
    return stem_polyline(chain, upper), nodes, hist


# ---------------------------------------------------------------- checks, outputs
def lowest_core_xy(D):
    from tomato3d.organs import stem_recon as SR
    xyz = D["xyz"]; zi = np.floor((xyz[:, 2] - D["medium"]) * 1000 / 5.0).astype(int)
    for s in sorted(set(zi.tolist())):
        idx = np.nonzero(zi == s)[0]
        if len(idx) >= 10:
            k = max(2, int(np.ceil(0.30 * len(idx)))); _, c, _ = SR.tight_core(xyz[idx, :2], k); return c, int(s)
    return D["centre"], None


def run_one(id_, out, tuned=False):
    t0 = time.time(); D = load(id_); start = lowest_core_xy(D) if tuned else (None, None); s5 = s5_chain(D, start[0]); lines = rachis_lines(D); poly, nodes, hist = s6_s7(D, s5, lines)
    allnodes = s5["nodes_branch"] + nodes
    for n_ in allnodes:
        n_["z_mm"] = round((float(n_["xyz"][2] if "xyz" in n_ else n_["z"]) - D["medium"]) * 1000, 1)
    allnodes.sort(key=lambda n_: n_["z_mm"]); prev = None
    for n_ in allnodes:
        pd = np.array(n_.get("petiole_dir", [0, 0, 0]), float); az = float(np.degrees(np.arctan2(pd[1], pd[0])) % 360) if np.linalg.norm(pd[:2]) > 1e-6 else None
        n_["azimuth_deg"] = None if az is None else round(az, 1)
        n_["divergence_deg"] = (round(abs((az - prev["azimuth_deg"] + 180) % 360 - 180), 1) if (az is not None and prev and prev.get("azimuth_deg") is not None) else None)
        n_["internode_mm"] = (round(n_["z_mm"] - prev["z_mm"], 1) if prev else None); n_["possible_duplicate"] = bool(prev and n_["internode_mm"] is not None and n_["internode_mm"] < CHK["short_mm"]); prev = n_
    hull = Delaunay(D["xyz"]); samp = np.vstack([np.linspace(a, b, 6) for a, b in zip(poly[:-1], poly[1:])]); inside = float((hull.find_simplex(samp) >= 0).mean()); c3 = inside >= CHK["c3"]
    n1 = []
    for ln in lines:
        if ln.get("source") != "trace" or "rejected" in ln: continue
        full = next((n_ for n_ in nodes if n_["key"] == ln["key"]), None); _, nd2, _ = s6_s7(D, s5, lines, drop=ln["key"]); loo = next((n_ for n_ in nd2 if n_["key"] == ln["key"]), None)
        if full and loo: n1.append(dict(key=ln["key"], z_full_mm=round((full["z"] - D["medium"]) * 1000, 1), z_loo_mm=round((loo["z"] - D["medium"]) * 1000, 1), dz_mm=round(abs(full["z"] - loo["z"]) * 1000, 2)))
    n_above = sum(1 for n_ in nodes if n_["target"][2] >= s5["h_lost_z"])
    n1_med = float(np.median([x["dz_mm"] for x in n1])) if n1 else None
    n1_pass = (n1_med <= CHK["n1_mm"]) if (len(n1) >= CHK["n1_min"] and n_above > 0) else None      # no target above h_lost -> S7 cannot move -> not testable
    height = float(poly[-1, 2] - D["medium"]); sup_h = float(s5["h_lost_z"] - D["medium"]); sup_share = sup_h / max(height, 1e-9)
    if sup_share >= CHK["visible_share"] and c3: tag = "stem_visible_supported"
    elif c3 and nodes and (n1_pass is None or n1_pass): tag = "stem_chain_plus_nodes"
    else: tag = "stem_inferred_only"
    res = dict(individual=id_, attempt=("tuned: start at the lowest-slab S0 core" if tuned else "attempt 1"), start_xy=(None if start[0] is None else np.round(start[0], 5).tolist()), medium_plane_z_m=D["medium"], rim=dict(centre=D["rim"]["rim_centre_xy_m"], z=D["rim"]["rim_z_m"], side=D["rim"]["rim_side_m"]), views=len(D["cams"]), plant_surfels=int(len(D["xyz"])),
               segments=[{k: v for k, v in s.items()} for s in s5["segments"]], h_lost_mm=round(sup_h * 1000, 1), apex_mm=round(height * 1000, 1), supported_share=round(sup_share, 3), surface_density_per_m2=round(s5["rho"], 0),
               polyline_m=np.round(poly, 5).tolist(), iterations=hist, radius_tag="radius_prior", C1="no ruler value", C3=dict(inside_share=round(inside, 3), passes=bool(c3)), N1=dict(per_node=n1, median_dz_mm=n1_med, passes=n1_pass, node_targets_above_h_lost=n_above, note=("not testable: no node target above h_lost, the refit cannot move" if n_above == 0 else None)),
               N2="no ruler value", tag=tag, tags=dict(TAGS, above_h_lost="conditional_hidden_state"), seconds=round(time.time() - t0))
    for sub in ("chain", "nodes", "qa"): (out / sub).mkdir(exist_ok=True)
    json.dump(res, open(out / "chain" / f"{id_}{'_tuned' if tuned else ''}.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(out / "nodes" / f"{id_}{'_tuned' if tuned else ''}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["node_id", "z_mm", "x_m", "y_m", "z_m", "source", "sigma_lat_mm", "sigma_z_mm", "attached_leaf", "azimuth_deg", "divergence_deg", "internode_mm", "possible_duplicate"])
        for i, n_ in enumerate(allnodes):
            p = n_.get("on_stem", n_.get("xyz")); p = np.asarray(p, float)
            w.writerow([i, n_["z_mm"], round(p[0], 5), round(p[1], 5), round(p[2], 5), n_["source"], n_.get("sigma_lat_mm"), n_.get("sigma_z_mm"), n_.get("leaf", ""), n_["azimuth_deg"], n_["divergence_deg"], n_["internode_mm"], n_["possible_duplicate"]])
    D["tagname"] = id_ + ("_tuned" if tuned else ""); overlay(D, s5, poly, allnodes, lines, out)
    return res, allnodes


def overlay(D, s5, poly, nodes, lines, out, px=420):
    photos = D["src"].get("photos"); tiles = []
    for c in D["cams"]:
        if photos and (photos / c["view"]).exists():
            img = Image.open(photos / c["view"]).convert("RGB").resize((px, px))
        else:
            img = Image.fromarray((np.dstack([c["mask"]] * 3) * 120 + 40).astype(np.uint8)).resize((px, px))
        dr = ImageDraw.Draw(img); s = px / MASK_RES
        def pr(X):
            xc = np.atleast_2d(X) @ c["R"].T + c["t"]; return np.c_[c["K"][0, 0] * xc[:, 0] / xc[:, 2] + c["K"][0, 2], c["K"][1, 1] * xc[:, 1] / xc[:, 2] + c["K"][1, 2]] * s
        for sg in s5["segments"]:
            q = pr(np.array([sg["start"], sg["end"]]))
            if sg["supported"]: dr.line([tuple(q[0]), tuple(q[1])], fill=(255, 40, 40), width=3)
            else:
                for k in range(4):
                    a = q[0] + (q[1] - q[0]) * k / 4; b = q[0] + (q[1] - q[0]) * (k + 0.5) / 4; dr.line([tuple(a), tuple(b)], fill=(255, 160, 0), width=2)
        q = pr(poly)
        for k in range(len(q) - 1):
            if poly[k, 2] >= s5["h_lost_z"] - 1e-6:
                a, b = q[k], q[k + 1]
                for j in range(3):
                    aa = a + (b - a) * j / 3; bb = a + (b - a) * (j + 0.5) / 3; dr.line([tuple(aa), tuple(bb)], fill=(255, 120, 200), width=2)
        for ln in lines:
            if "rejected" in ln: continue
            q = pr(np.vstack([ln["line"], ln["base"] + ln["dir"] * S6["pet_hi_mm"] / 1000])); dr.line([tuple(x) for x in q], fill=(255, 230, 0), width=1)
        for n_ in nodes:
            p = np.asarray(n_.get("on_stem", n_.get("xyz")), float); q = pr(p)[0]; col = (0, 220, 255) if "label" in n_["source"] else ((255, 230, 0) if "trace" in n_["source"] else (190, 120, 255))
            dr.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=col, outline=(0, 0, 0))
        name = f"{D.get('tagname', D['id'])}_{Path(c['view']).stem}.png"; img.save(out / "qa" / name); tiles.append(name)
    html = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>" + D["id"] + " 줄기 run2</title>"
            "<style>body{margin:0;padding:12px 16px;background:#111;color:#eee;font:13px system-ui,sans-serif}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:6px}img{width:100%;display:block}</style></head><body>"
            f"<h3>{D['id']} · 줄기 run2 · 빨강 실선 = 근거 있는 원통 마디, 주황 점선 = 근거 없음, 분홍 점선 = h_lost 위 추정(conditional_hidden_state) · 노랑 가는 선 = 예측 엽축 · 점: 노랑 엽축, 하늘 라벨, 보라 가지 · conditional_on_unvalidated_scale · 채점 안 함</h3><div class='g'>"
            + "".join(f"<figure style='margin:0'><img src='{t}' alt='{t}'><figcaption>{t}</figcaption></figure>" for t in tiles) + "</div></body></html>")
    (out / "qa" / f"{D.get('tagname', D['id'])}.html").write_text(html, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--ids", nargs="+", default=["R3-1b", "n_03", "n_06", "t2_02", "t2_05"]); ap.add_argument("--tuned", nargs="*", default=[]); a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    files = {}
    for id_ in a.ids:
        for k, v in SRC[id_].items():
            files[f"{id_}.{k}"] = dict(path=str(v), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(v))))
        files[f"{id_}.rim"] = dict(path=str(RIM / f"{id_}.json"), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(RIM / f"{id_}.json"))))
    json.dump(files, open(a.out / "inputs.json", "w"), indent=1)
    rows = []
    for id_ in a.ids + [f"{t}+tuned" for t in a.tuned]:
        tuned = id_.endswith("+tuned"); id_ = id_.replace("+tuned", "")
        res, nodes = run_one(id_, a.out, tuned)
        id_ = id_ + ("_tuned" if tuned else "")
        print(id_, json.dumps(dict(h_lost_mm=res["h_lost_mm"], apex_mm=res["apex_mm"], supported_share=res["supported_share"], C3=res["C3"], N1=dict(res["N1"], per_node=len(res["N1"]["per_node"])), tag=res["tag"],
                                   nodes=len(nodes), segs=len(res["segments"]), seconds=res["seconds"]), default=float), flush=True)
        rows.append([id_, res["views"], res["h_lost_mm"], res["apex_mm"], res["supported_share"], res["C1"], res["C3"]["inside_share"], res["C3"]["passes"], res["N1"]["median_dz_mm"], res["N1"]["passes"], len(nodes),
                     sum(1 for n_ in nodes if n_.get("possible_duplicate")), res["tag"], TAGS["scale"], res["seconds"]])
    with open(a.out / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["individual", "views", "h_lost_mm", "apex_mm", "supported_share", "C1", "C3_inside", "C3_pass", "N1_median_dz_mm", "N1_pass", "nodes", "possible_duplicates", "tag", "scale", "seconds"]); w.writerows(rows)


if __name__ == "__main__":
    main()
