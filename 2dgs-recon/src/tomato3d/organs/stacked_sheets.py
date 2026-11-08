"""Separate stacked leaves by layer (geometry + visibility); leaf names come from labels only.

Contract: build/p1_stacked_sheets_20261026_01/CONTRACT.md. Step 0 measures vote leakage and graph bridging; the label-free stack-violation rate
is the development measure; candidates S1 (strict visibility votes), S2 (sheet graph + cannot-links), S3 (visibility signature), S4 (merge sheets,
name by label majority); A-D' re-run on the layer graph; the held-out photo is scored once for everything at the end.

usage: python -m tomato3d.organs.stacked_sheets [--step contrib|run]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import light_organs as LO
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.organs import vp_cleanup as VC
from tomato3d.organs import vp_recluster as RC
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_stacked_sheets_20261026_01"
CLEAN = VC.OUT
STATUS = "R3-1b_unaccepted"; GROUPING = "stacked_sheets"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping=GROUPING)
# declared (contract)
STACK = dict(d_min_mm=2.0, d_max_mm=12.0, step_mm=1.0, lateral_mm=1.0, fold_dot=-0.8)
TARGET_RATE = 0.02
S1 = dict(w_min=0.1, sigma_cut=3.0, max_r_px=14, slice_mm=1.0, res=1512)
S2 = dict(plane_mm=1.0, angle_deg=30.0)
S3 = dict(edge_jaccard_max=0.7, merge_sim_min=0.5)
S4 = dict(min_edges=5, angle_deg=30.0, dz_mm=2.0, name_share=0.60)
UNNAMED_BASE = 200
NONLEAF = VC.U_BASE        # ids >= 100: U clusters (100+) and unnamed sheets (200+) are never leaves
LEAKAGE = dict(gap_mm=1.5, bridge_mm=1.5)


# ---------------------------------------------------------------- inputs
def load():
    pl = UL.load_plant(); sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); rows = pl["rows"]; b = np.nonzero(pl["blade"])[0]
    nrm = sf["normal"][rows].astype(np.float64); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pl, b, nrm, sf["opacity"][rows].astype(np.float64)


def base_labels(n_full):
    z = np.load(CLEAN / "surfel_clusters.npz", allow_pickle=True); lab = np.where((z["leaf"] > 0) & (z["leaf"] < VC.U_BASE), z["leaf"], 0).astype(np.int32)
    assert len(lab) == n_full; return lab, z["axis_kind"]


# ---------------------------------------------------------------- stack relation and violations
def stack_pairs(xyz, nrm, st=STACK):
    """Non-fold and fold pairs (i < j) of the stack relation over the given points (blade subset)."""
    tree = cKDTree(xyz); pairs = set(); offs = np.arange(st["d_min_mm"], st["d_max_mm"] + 1e-9, st["step_mm"]) / 1000
    for sgn in (1, -1):
        for d in offs:
            q = xyz + sgn * d * nrm
            for i, js in enumerate(tree.query_ball_point(q, st["lateral_mm"] / 1000)):
                for j in js:
                    if j != i: pairs.add((min(i, j), max(i, j)))
    P = np.array(sorted(pairs), int).reshape(-1, 2)
    dot = (nrm[P[:, 0]] * nrm[P[:, 1]]).sum(1) if len(P) else np.zeros(0)
    return P[dot >= st["fold_dot"]], P[dot < st["fold_dot"]]


def violations(lab, P):
    """Per labelled surfel: violates if a non-fold stack partner has the same label. Returns bool array, overall rate, per-leaf rates."""
    same = (lab[P[:, 0]] > 0) & (lab[P[:, 0]] == lab[P[:, 1]]); v = np.zeros(len(lab), bool); v[P[same, 0]] = True; v[P[same, 1]] = True
    labelled = lab > 0; rate = float(v[labelled].mean()) if labelled.any() else 0.0
    per = {int(L): round(float(v[lab == L].mean()), 4) for L in sorted(set(lab[labelled].tolist()))}
    return v, round(rate, 4), per


def layer_check(lab, P, xyz, a=6, b_=4):
    """Named check 'leaf 6 over leaf 4': stack pairs with both labels in {a, b}."""
    la, lb = lab[P[:, 0]], lab[P[:, 1]]; m = np.isin(la, [a, b_]) & np.isin(lb, [a, b_]); Q = P[m]
    if not len(Q): return dict(pairs=0)
    up = np.where(xyz[Q[:, 0], 2] >= xyz[Q[:, 1], 2], Q[:, 0], Q[:, 1]); lo = np.where(up == Q[:, 0], Q[:, 1], Q[:, 0])
    same = lab[up] == lab[lo]
    return dict(pairs=int(len(Q)), same_label=int(same.sum()), upper_a_lower_b=int(((lab[up] == a) & (lab[lo] == b_)).sum()), upper_b_lower_a=int(((lab[up] == b_) & (lab[lo] == a)).sum()))


# ---------------------------------------------------------------- step 0
def leakage(pl, b, allmaps, H2id):
    """Depth gap (mm) between each voting surfel and the first hit at its pixel, over every vote of the V-P label maps."""
    views = pl["views"]; gaps = []
    for vid, m in allmaps.items():
        view = AP.view_file(vid); k = views.index(view); pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"])
        vis = np.nonzero((H2id[:, k] >= 0) & pl["blade"])[0]; u, w = pv["u"][vis], pv["w"][vis]
        lv = m[w * UL.H2 // V2.VISP["res"], u * UL.H2 // V2.VISP["res"]]; vv = vis[lv > 0]
        gaps.append((pv["z"][vv] - pv["zb"][pv["w"][vv], pv["u"][vv]]) * 1000)
    g = np.concatenate(gaps); h, e = np.histogram(g, bins=[-1e9, 0, 0.25, 0.5, 1.0, 1.5, 3, 6, 1e9])
    return dict(votes=int(len(g)), share_gt_1p5mm=round(float((g > 1.5).mean()), 4), share_0p5_1p5mm=round(float(((g > 0.5) & (g <= 1.5)).mean()), 4),
                histogram={f"{e[i]:.2g}..{e[i + 1]:.2g}": int(h[i]) for i in range(len(h))})


def plane_dist(xyz, nrm, i, j):
    d = xyz[j] - xyz[i]; return np.abs((d * nrm[i]).sum(1)) * 1000, np.abs((d * nrm[j]).sum(1)) * 1000


def bridging(xyz, nrm, G):
    C = G.tocoo(); m = C.row < C.col; i, j = C.row[m], C.col[m]; a, b_ = plane_dist(xyz, nrm, i, j); d = np.maximum(a, b_)
    return dict(edges=int(len(i)), share_gt_1p5mm=round(float((d > LEAKAGE["bridge_mm"]).mean()), 4), median_mm=round(float(np.median(d)), 3))


# ---------------------------------------------------------------- S1: rendering contribution per surfel and view
def contributions(pl, op, view, res=S1["res"]):
    """w = opacity x transmittance at the surfel's centre pixel (front-to-back in 1 mm depth slices)."""
    K = ca.scaled_K(np.array(pl["po"]["K_undist"]), pl["po"]["undistorted_res"], res); R = np.array(pl["po"]["views"][view]["R_cw"]); t = np.array(pl["po"]["views"][view]["t_cw"])
    xc = pl["xyz"] @ R.T + t; z = xc[:, 2]; zs = np.where(z > 0, z, 1e-9); u = np.round(K[0, 0] * xc[:, 0] / zs + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / zs + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); idx = np.nonzero(ok)[0]; out = np.zeros(len(z))
    sig = np.maximum(pl["sc"][idx].max(1) * K[0, 0] / z[idx], 0.3); rad = np.minimum(np.ceil(S1["sigma_cut"] * sig), S1["max_r_px"]).astype(int)
    logT = np.zeros((res, res)); order = np.argsort(z[idx], kind="stable"); zz = z[idx][order]; sl = np.floor((zz - zz[0]) * 1000 / S1["slice_mm"]).astype(int)
    bounds = np.r_[0, np.nonzero(np.diff(sl))[0] + 1, len(order)]
    for a0, a1 in zip(bounds[:-1], bounds[1:]):
        ch = idx[order[a0:a1]]; uc, wc = u[ch], w[ch]; out[ch] = op[ch] * np.exp(logT[wc, uc])
        sg = sig[order[a0:a1]]; rr = rad[order[a0:a1]]; R0 = int(rr.max())
        for dy in range(-R0, R0 + 1):
            for dx in range(-R0, R0 + 1):
                r2 = dx * dx + dy * dy; m = r2 <= rr * rr
                if not m.any(): continue
                uu, ww = uc[m] + dx, wc[m] + dy; okp = (uu >= 0) & (ww >= 0) & (uu < res) & (ww < res)
                al = np.clip(op[ch][m][okp] * np.exp(-0.5 * r2 / sg[m][okp] ** 2), 0, 0.999)
                np.add.at(logT, (ww[okp], uu[okp]), np.log1p(-al))
    return out


def step_contrib(out: Path):
    pl, b, nrm, op = load(); t0 = time.time(); W = np.zeros((len(pl["xyz"]), len(pl["views"])), np.float32)
    for k, v in enumerate(pl["views"]):
        W[:, k] = contributions(pl, op, v); print(f"contrib {k + 1}/{len(pl['views'])} {v[9:18]} visible>=w_min {(W[:, k] >= S1['w_min']).sum()} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out / "contributions.npz", W=W, views=np.array(pl["views"]))


# ---------------------------------------------------------------- votes with a visibility matrix
def vote_with_vis(pl, allmaps, VIS, rule_share=0.6):
    views = pl["views"]; n = len(pl["xyz"]); cnt = np.zeros((n, 13), np.int32)
    for vid, m in allmaps.items():
        view = AP.view_file(vid); k = views.index(view); pv = V2.project_view(pl["xyz"][:1], pl["sc"][:1], pl["po"], view, V2.VISP["res"])
        u, w, z = V2.project_points(pl["xyz"], pv); ui = np.round(u).astype(int); wi = np.round(w).astype(int); res = V2.VISP["res"]
        ok = VIS[:, k] & (ui >= 0) & (wi >= 0) & (ui < res) & (wi < res) & pl["blade"]; s = np.nonzero(ok)[0]
        lv = m[wi[s] * UL.H2 // res, ui[s] * UL.H2 // res]; sel = lv > 0; np.add.at(cnt, (s[sel], lv[sel].astype(int)), 1)
    tot = cnt.sum(1); top = cnt.max(1); share = np.where(tot > 0, top / np.maximum(tot, 1), 0); second = np.sort(cnt, 1)[:, -2]
    lab = np.where((tot >= 1) & (share >= rule_share) & (top > second), cnt.argmax(1), 0).astype(np.int32); lab[~pl["blade"]] = 0; return lab, cnt


def vp_allmaps(pl, H2id):
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; views = pl["views"]
    nmap = json.load(open(RC.MAP_FILE, encoding="utf-8"))["map"]; raw, _ = UL.label_maps(UL.OUT, chosen + [held])
    for v in chosen: UL.STATE_PPM[v] = UL.plant_px_per_mm(pl, AP.view_file(v), H2id[:, views.index(AP.view_file(v))])[0]
    topo_rows, _ = UL.topology_report(pl, raw, nmap, chosen + [held], H2id)
    seed_maps, _, _ = UL.build_vote_maps(pl, raw, nmap, chosen, [r for r in topo_rows if r["photo"] in chosen], "V-H+fill+topo")
    pz = np.load(RC.VP_DIR / "propagated_maps.npz"); prop = {k[1:]: pz[k] for k in pz.files}
    held_map = UL.remap({held: raw[held]}, nmap)[held]; held_map = np.where(held_map > 0, held_map, 0).astype(np.int16)
    return dict(seed_maps, **prop), held, held_map


# ---------------------------------------------------------------- S2-S4: sheets, cannot-links, merging, naming
def sheet_graph(xyz, nrm, G, sig_bits=None, jacc_max=None):
    C = G.tocoo(); m = C.row < C.col; i, j = C.row[m], C.col[m]; a, b_ = plane_dist(xyz, nrm, i, j)
    ang = np.degrees(np.arccos(np.clip(np.abs((nrm[i] * nrm[j]).sum(1)), 0, 1))); keep = (a <= S2["plane_mm"]) & (b_ <= S2["plane_mm"]) & (ang <= S2["angle_deg"])
    if sig_bits is not None:
        inter = np.bitwise_count(sig_bits[i] & sig_bits[j]).astype(float); uni = np.bitwise_count(sig_bits[i] | sig_bits[j]).astype(float)
        jd = np.where(uni > 0, 1 - inter / np.maximum(uni, 1), 0.0); keep &= jd <= jacc_max
    n = len(xyz); S = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n)).tocsr(); S = S.maximum(S.T)
    ns, sheet = connected_components(S, directed=False); return S, sheet, ns, (i, j, keep)


def merge_sheets(xyz, nrm, G, sheet, ns, P_stack, sig_mean=None):
    C = G.tocoo(); m = C.row < C.col; i, j = C.row[m], C.col[m]; si, sj = sheet[i], sheet[j]; cross = si != sj
    key = np.minimum(si, sj)[cross] * ns + np.maximum(si, sj)[cross]; ei, ej = i[cross], j[cross]
    order = np.argsort(key, kind="stable"); key, ei, ej = key[order], ei[order], ej[order]; uk, start, counts = np.unique(key, return_index=True, return_counts=True)
    parent = list(range(ns)); find = lambda x: x if parent[x] == x else find(parent[x])
    cl = {}
    for a, b_ in zip(sheet[P_stack[:, 0]], sheet[P_stack[:, 1]]):
        if a != b_: cl.setdefault(int(a), set()).add(int(b_)); cl.setdefault(int(b_), set()).add(int(a))
    members = {s: {s} for s in range(ns)}; stats = Counter(); merged_edges = []
    for k in np.argsort(-counts, kind="stable"):
        if counts[k] < S4["min_edges"]: stats["too_few_edges"] += 1; continue
        a, b_ = int(uk[k] // ns), int(uk[k] % ns); ra, rb = find(a), find(b_)
        if ra == rb: continue
        e0, e1 = ei[start[k]:start[k] + counts[k]], ej[start[k]:start[k] + counts[k]]
        na = nrm[e0].mean(0); nb = nrm[e1].mean(0); ang = np.degrees(np.arccos(np.clip(abs(na @ nb) / max(np.linalg.norm(na) * np.linalg.norm(nb), 1e-12), 0, 1)))
        dz = float(np.median(np.abs(xyz[e0, 2] - xyz[e1, 2]))) * 1000
        if ang > S4["angle_deg"]: stats["angle"] += 1; continue
        if dz > S4["dz_mm"]: stats["dz"] += 1; continue
        if sig_mean is not None:
            sa, sb = sig_mean[a], sig_mean[b_]; sim = float(np.minimum(sa, sb).sum() / max(np.maximum(sa, sb).sum(), 1e-12))
            if sim < S3["merge_sim_min"]: stats["signature"] += 1; continue
        if any(find(x) == rb for s in members[ra] for x in cl.get(s, ())): stats["cannot_link"] += 1; continue
        parent[rb] = ra; members[ra] |= members.pop(rb); stats["merged"] += 1; merged_edges.append((e0, e1))
    group = np.array([find(s) for s in range(ns)])[sheet]; return group, stats, merged_edges


def name_groups(group, base):
    out = np.zeros(len(group), np.int32); unnamed = 0; table = []
    for g in np.unique(group):
        m = group == g; l = base[m]; l = l[l > 0]
        if len(l) and Counter(l.tolist()).most_common(1)[0][1] / len(l) >= S4["name_share"]:
            out[m] = Counter(l.tolist()).most_common(1)[0][0]
        else:
            unnamed += 1; out[m] = UNNAMED_BASE + unnamed
        table.append(int(m.sum()))
    return out, unnamed


def layer_graph(n, S, merged_edges):
    if merged_edges:
        e0 = np.concatenate([a for a, _ in merged_edges]); e1 = np.concatenate([b for _, b in merged_edges])
        M = coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(n, n)).tocsr(); M = M.maximum(M.T); return (S + M).tocsr()
    return S


def run_cleanup(lab_named, xyz, Gl):
    lab0 = np.where(lab_named >= UNNAMED_BASE, 0, lab_named)
    a, _ = VC.absorb_detached(lab0, xyz, Gl, graph_neighbours=True); bb, _ = VC.fill_gaps(a, xyz, Gl); c, _ = VC.fill_rest(bb, Gl); d, _ = VC.complete_geometry(c, xyz, Gl, cap=True)
    d = np.where((d == 0) & (lab_named >= UNNAMED_BASE), lab_named, d); return d


# ---------------------------------------------------------------- main
def step_run(out: Path):
    t0 = time.time(); pl, b, nrm_full, op = load(); views = pl["views"]; H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; H = float(pl["xyz"][:, 2].max() - V2.POT_TOP)
    base_full, axis_kind = base_labels(len(pl["xyz"])); xyz = pl["xyz"][b]; nrm = nrm_full[b]; n = len(b); base = base_full[b]
    G, _ = RC.build_graph(xyz); P, Fold = stack_pairs(xyz, nrm); print(f"graph + stack pairs: {len(P)} pairs, {len(Fold)} folds {time.time() - t0:.0f}s", flush=True)
    allmaps, held, held_map = vp_allmaps(pl, H2id)
    step0 = dict(leakage=leakage(pl, b, allmaps, H2id), bridging=bridging(xyz, nrm, G)); print("step0", json.dumps(step0), flush=True)
    W = np.load(out / "contributions.npz")["W"]; VIS = W >= S1["w_min"]
    lab_vp_full, _ = UL.vote_views(pl, allmaps, sorted(allmaps), None, H2id); lab_vp = lab_vp_full[b]
    lab_s1_full, _ = vote_with_vis(pl, allmaps, VIS); lab_s1 = lab_s1_full[b]
    bits = np.zeros(len(pl["xyz"]), np.uint64)
    for k in range(VIS.shape[1]): bits |= (VIS[:, k].astype(np.uint64) << np.uint64(k))
    bits_b = bits[b]
    # S2 + S4 (named by A+B+C+D')
    S, sheet, ns, _ = sheet_graph(xyz, nrm, G); g24, st24, me24 = merge_sheets(xyz, nrm, G, sheet, ns, P); lab24, un24 = name_groups(g24, base)
    # S1 + S2 + S3 + S4 (named by S1)
    S3g, sheet3, ns3, _ = sheet_graph(xyz, nrm, G, bits_b, S3["edge_jaccard_max"])
    VISb = VIS[b].astype(float); sig_mean = np.zeros((ns3, VISb.shape[1]))
    np.add.at(sig_mean, sheet3, VISb); sig_mean /= np.maximum(np.bincount(sheet3, minlength=ns3)[:, None], 1)
    g1234, st1234, me1234 = merge_sheets(xyz, nrm, G, sheet3, ns3, P, sig_mean); lab1234, un1234 = name_groups(g1234, lab_s1)
    cands = dict(S1=lab_s1, S2_S4=lab24, S1_S2_S3_S4=lab1234)
    dev = {}
    for k, lab in dict(VP=lab_vp, ABCD_prime=base, **cands).items():
        lab_n = np.where(lab >= NONLEAF, 0, lab); v, rate, per = violations(lab_n, P)
        dev[k] = dict(rate=rate, per_leaf=per, labelled=int((lab_n > 0).sum()), unnamed_share=round(float((lab >= UNNAMED_BASE).mean()), 4), check_6_over_4=layer_check(lab_n, P, xyz))
        print("dev", k, rate, per, dev[k]["check_6_over_4"], flush=True)
    sheets_per_leaf = {k: {int(L): int(len(set(g[lab == L].tolist()))) for L in sorted(set(lab[(lab > 0) & (lab < NONLEAF)].tolist()))} for k, (g, lab) in dict(S2_S4=(sheet, lab24), S1_S2_S3_S4=(sheet3, lab1234)).items()}
    choice = min(cands, key=lambda k: (dev[k]["rate"], dev[k]["unnamed_share"]))
    Sx, mex = (S, me24) if choice != "S1_S2_S3_S4" else (S3g, me1234)
    Gl = layer_graph(n, Sx, mex); final = run_cleanup(cands[choice], xyz, Gl)
    fn = np.where(final >= NONLEAF, 0, final); vf, ratef, perf = violations(fn, P)
    dev["final"] = dict(rate=ratef, per_leaf=perf, labelled=int((fn > 0).sum()), unnamed_share=round(float((final >= UNNAMED_BASE).mean()), 4), u_cluster_share=round(float(((final >= NONLEAF) & (final < UNNAMED_BASE)).mean()), 4), check_6_over_4=layer_check(fn, P, xyz))
    print("dev final", ratef, perf, dev["final"]["check_6_over_4"], flush=True)
    # traces: sheets crossed (list only)
    traces = CC.load_traces(V2.OUT / "traces_export" / "rachis_traces"); cross = []
    for t in traces:
        pts = CC.densify(np.asarray(t["pts"], float), 0.0005); d, _ = cKDTree(pts).query(xyz); m = d <= 0.004
        cross.append(dict(trace=t["key"], leaf=t["leaf"], final_labels=dict(Counter(final[m].tolist()).most_common(4))))
    # held-out once, every state
    hv = AP.view_file(held); col = H2id[:, views.index(hv)]; scores = {}
    def full(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = np.where(lb >= NONLEAF, 0, lb); return f
    for k, lab in dict(VP=lab_vp, ABCD_prime=base, **cands, final=final).items():
        ren = UL.render_ids(pl, full(lab), hv, col); rows, g = UL.score_heldout(held_map, ren); cov = float(((ren > 0) & (held_map > 0)).sum() / max((held_map > 0).sum(), 1))
        flags = {str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rows.items()}
        scores[k] = dict(rows={str(L): v for L, v in rows.items()}, summary=g, coverage=round(cov, 4), flags=flags)
        print("held-out", k, round(g["median_iou"], 3), round(g["max_confusion"], 3), "cov", round(cov, 3), {L: (round(r["iou"], 2), round(r["confusion"], 2)) for L, r in rows.items()}, flush=True)
    freeze = dev["final"]["rate"] < dev["ABCD_prime"]["rate"] and scores["final"]["summary"]["median_iou"] >= scores["ABCD_prime"]["summary"]["median_iou"]
    tag = "gate_passed" if scores["final"]["summary"]["gate_pass"] else "gate_failed_viewing_only"
    res = dict(**LABELS, base="A+B+C+D' (build/p1_vp_cleanup_20261026_01)", params=dict(stack=STACK, target=TARGET_RATE, s1=S1, s2=S2, s3=S3, s4=S4), step0=step0,
               stack_pairs=int(len(P)), folds=int(len(Fold)), sheets=dict(S2=int(ns), S2_S3=int(ns3)), merge_stats=dict(S2_S4=dict(st24), S1_S2_S3_S4=dict(st1234)),
               unnamed=dict(S2_S4=un24, S1_S2_S3_S4=un1234), sheets_per_leaf=sheets_per_leaf, development=dev, choice=choice, target_met=bool(dev["final"]["rate"] <= TARGET_RATE),
               traces_cross=cross, held_out=scores, freeze=bool(freeze), tag=tag, seconds=round(time.time() - t0))
    json.dump(res, open(out / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    if freeze:
        json.dump(dict(frozen_on="2026-10-26", plant="R3-1b_unaccepted", variant=choice, stack=STACK, s1=S1, s2=S2, s3=S3, s4=S4, note="cite in the all-individuals run CONTRACT.md (that run does not exist yet)"),
                  open(out / "frozen_stacked_sheets.json", "w", encoding="utf-8"), indent=1)
    lf = np.full(len(pl["xyz"]), -1); lf[b] = np.where(final > 0, final, -1); sf_ = np.full(len(pl["xyz"]), -1); sf_[b] = (g1234 if choice == "S1_S2_S3_S4" else g24)
    vfull = np.zeros(len(pl["xyz"]), bool); vfull[b] = vf
    np.savez_compressed(out / "surfel_clusters.npz", rows=pl["rows"], leaf=lf, sheet=sf_, violation=vfull, axis_kind=axis_kind, **{f"{k}_leaf": np.where(full(v) > 0, full(v), -1) for k, v in cands.items()})
    with open(out / "surfel_leaf_ids.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {GROUPING}; variant {choice} + A-D' on layer graph; {tag}; leaf_id >= {UNNAMED_BASE} = unnamed sheet; -1 = none\nrow,leaf_id,sheet,stack_violation\n")
        for r_, l_, s_, v_ in zip(pl["rows"], lf, sf_, vfull): fh.write(f"{r_},{l_},{s_},{int(v_)}\n")
    table = UL.leaf_table(pl, full(final), {}, H); fl = scores["final"]["flags"]
    for r in table: r["quality"] = fl.get(str(r["leaf"]), "not_in_held_out"); r["stack_violation_rate"] = perf.get(r["leaf"])
    with open(out / "leaves.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {GROUPING}; {tag}; area = kind (1)\n")
        keys = sorted({k for r in table for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table)
    print("done", choice, tag, "freeze", freeze, f"{time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--step", choices=("contrib", "run"), default="run"); a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    step_contrib(a.out) if a.step == "contrib" else step_run(a.out)


if __name__ == "__main__":
    main()
