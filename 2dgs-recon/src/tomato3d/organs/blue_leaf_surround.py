"""Unpainted surfels around the blue leaf (user prompt 2026-10-27): reasons, evidence for "belongs to the blue leaf", and a separate proposal layer.
R7 labels are read only. Contract: build/p1_local_diffusion_20261027_01/proposal_blue_leaf_2026-10-27/CONTRACT.md.

Revision 2 (user 2026-10-27, rule 14): evidence also includes the reconstructed-3D surroundings (distance-weighted label share of the labelled surfels within 8 mm).

usage: python -m tomato3d.organs.blue_leaf_surround [--with3d --leaf 4]
"""

from __future__ import annotations

import colorsys
import csv
import json
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from tomato3d.organs import chain_r6 as C6
from tomato3d.organs import contested_resolve as CR
from tomato3d.organs import contested_resolve_r1 as R1
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import oversegment as OS
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2

OUT = LD.OUT / "proposal_blue_leaf_2026-10-27"; R7DIR = LD.OUT / "vp_leaf4"; STATUS = "R3-1b_unaccepted"
P = dict(hue=(170.0, 250.0), region_mm=12.0, near_mm=3.0, plane_mm=5.0, k_local=8, normal_deg=30.0, trace_mm=3.0, cand_mm=6.0, strict_total=0.6, strict_margin=0.25, cell_mm=10.0,
         nb3d_mm=8.0, nb3d_weight=0.25, nb3d_sens_mm=(4.0, 12.0), nb3d_floor_mm=0.5)
REASONS = ("detached_cleanup", "contested", "out_of_reach: gap > 3 mm", "out_of_reach: normal > 30°", "out_of_reach: offset", "out_of_reach: other sheet", "out_of_reach: connected, beyond reach",
           "outside_candidate_range", "other")
BREAKS = ("gap > 3 mm", "normal > 30°", "offset", "other sheet", "connected")


def area_mm2(pl, idx):
    """Union of surfel ellipses per 10 mm cell, each cell on its own PCA plane (light_organs.union_area); mm²."""
    if len(idx) == 0: return 0.0
    X = pl["xyz"]; cell = np.floor(X[idx] / (P["cell_mm"] / 1000)).astype(np.int64); keys = defaultdict(list)
    for i, k in zip(idx, map(tuple, cell)): keys[k].append(i)
    tot = 0.0
    for mem in keys.values():
        mem = np.array(mem)
        if len(mem) >= 3: c, _, e1, e2, _ = LO.pca_frame(X[mem])
        else:
            n = pl["nrm"][mem[0]]; e1 = pl["tu"][mem[0]]; e2 = np.cross(n, e1); c = X[mem].mean(0)
        a, _, _, _ = LO.union_area(X[mem], pl["tu"][mem], pl["tv"][mem], pl["sc"][mem, :2], c, e1, e2); tot += a
    return tot * 1e6


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12), 0, 1)))


def break_levels(xyz, S):
    """Component ids of the blade surfels on four nested graphs: 3 mm; + normal; + offset; + same sheet (= sheet graph)."""
    E = SR.edges_3mm(xyz); i, j = E[:, 0], E[:, 1]; ni, nj = S["normal"][i], S["normal"][j]
    ang = np.degrees(np.arccos(np.clip(np.abs((ni * nj).sum(1)), 0, 1))); mn = ni + np.where(((ni * nj).sum(1) < 0)[:, None], -nj, nj); mn /= np.maximum(np.linalg.norm(mn, axis=1, keepdims=True), 1e-12)
    doff = np.abs(((xyz[j] - xyz[i]) * mn).sum(1)) * 1000; multi = (S["n_sheets"][i] > 1) & (S["n_sheets"][j] > 1); same = ~multi | (S["index"][i] == S["index"][j])
    ok_a = ang <= SR.P["angle_deg"]; ok_o = doff <= LD.D["offset_mm"]; masks = (np.ones(len(E), bool), ok_a, ok_a & ok_o, ok_a & ok_o & same); n = len(xyz); out = []
    for m in masks:
        G = coo_matrix((np.ones(int(m.sum())), (i[m], j[m])), shape=(n, n)).tocsr(); out.append(connected_components(G, directed=False)[1])
    return out


def pixel_ids(pl, vids, H2id):
    """Per photo: H2-grid linear pixel of every surfel where visible (H2id >= 0, userlabel.vote_views convention), else -1."""
    out = {}
    for vid in vids:
        view = f"20261021_{vid}_iOS.jpg"; k = pl["views"].index(view); vis = H2id[:, k] >= 0; pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"])
        u = pv["u"] * UL.H2 // V2.VISP["res"]; w = pv["w"] * UL.H2 // V2.VISP["res"]; ok = vis & (u >= 0) & (w >= 0) & (u < UL.H2) & (w < UL.H2)
        out[vid] = np.where(ok, w * UL.H2 + u, -1).astype(np.int64)
    return out


def rachis_euclid(X, mem, L, T, labF, reach_mm):
    """R1 continuous rachis score with the geodesic replaced by the Euclidean distance to the leaf's trace (axis clusters have no sheet graph)."""
    t = T.get(L)
    if t is None or t["neutral"]: return R1.G["pseudo_score"]
    c = X[mem].mean(0); d, j = t["tree"].query(c); d_mm = d * 1000; s_dist = max(0.0, 1 - d_mm / reach_mm); dense = t["dense"]
    tang = dense[min(j + 1, len(dense) - 1)] - dense[max(j - 1, 0)]; tang /= max(np.linalg.norm(tang), 1e-12)
    if len(mem) >= 3: _, _, vt = np.linalg.svd(X[mem] - c, full_matrices=False); axis = vt[0]
    else: axis = X[mem[-1]] - X[mem[0]] if len(mem) > 1 else np.array([1.0, 0, 0])
    v = c - dense[j]; s_dir = float(abs(axis @ v) / max(np.linalg.norm(axis) * np.linalg.norm(v), 1e-12))
    memL = np.nonzero(labF == L)[0]; dL, jL = t["tree"].query(X[memL]); step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]; near = np.abs(step[jL] - step[j]) * 1000 <= R1.G["side_mm"]
    side_c = np.sign(np.cross(tang, v)[2]); sideL = np.sign(np.cross(np.repeat(tang[None], near.sum(), 0), X[memL[near]] - dense[jL[near]])[:, 2]) if near.any() else np.zeros(0)
    return s_dist * (0.6 * s_dir + 0.4 * (1.0 if (sideL == side_c).sum() >= R1.G["side_min"] else 0.5))


def with_nb3d(w):
    """Rule 14 weights: the four declared weights scaled by (1 - w3d), plus the 3D-surroundings term w3d (ratios of the four kept)."""
    return dict({k: v * (1 - P["nb3d_weight"]) for k, v in w.items()}, nb3d=P["nb3d_weight"])


def nb3d_share(X, mem, ltree, lidx, labF, r_mm):
    """Distance-weighted (1 / max(d, 0.5 mm)) share per leaf of the labelled reconstructed surfels within r of the unit's members."""
    acc = Counter()
    for s_ in mem:
        for j in ltree.query_ball_point(X[s_], r_mm / 1000):
            t = lidx[j]; acc[int(labF[t])] += 1.0 / max(np.linalg.norm(X[t] - X[s_]) * 1000, P["nb3d_floor_mm"])
    tot = sum(acc.values()); return {L: v / tot for L, v in acc.items()} if tot else {}


def totals(ev, cands, w, unseen, photo_first):
    if unseen and not photo_first:
        geo = [k for k in ("sheet", "rachis", "nb3d") if k in w]; sw = sum(w[k] for k in geo)
        return {L: sum(w[k] / sw * ev[k][L] for k in geo) for L in cands}
    tot = {L: sum(w[k] * ev[k][L] for k in w) for L in cands}
    return {L: min(v, CR.R["unseen_cap"]) for L, v in tot.items()} if unseen else tot


def main(with3d=False, only=None):
    global OUT
    if with3d: OUT = OUT / "with_3d_surroundings"
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]; X = pl["xyz"]; n = len(X); axisF = B["axis"]; nb = len(b)
    pos = -np.ones(n, int); pos[b] = np.arange(nb)
    z = np.load(R7DIR / "labels.npz"); labF = z["label"].astype(int); s3, s4 = z["step_3_distance_photo"], z["step_4_bridge"]; assert np.array_equal(labF[b], s4)
    # recompute the R7 diffusion (source tags in labels.npz collide, see contract)
    B["base"] = z["base_vp_new"].astype(np.int16); base_b = B["base"][b]
    G, Gp, _ = LD.sheet_graph(xyz, S); seeds, _, _ = LD.seeds_and_reach(B, Gp); leaves, C = LD.costs(G, seeds)
    dense_all = np.vstack([LD.CC.densify(t["pts"]) for t in B["traces"] if t["leaf"] > 0]); dnear, _ = cKDTree(np.vstack([xyz[base_b > 0], dense_all])).query(xyz)
    cand = (base_b == 0) & (axisF[b] == 0) & (dnear <= LD.D["candidate_mm"] / 1000); lab_new, src = C6.diffuse(B, G, leaves, C, cand, C6.U_MM)
    s1 = np.where(base_b > 0, base_b, lab_new); repro = bool(np.array_equal(s1, z["step_1_diffusion"])); print("R7 diffusion reproduced:", repro, f"({time.time() - t0:.0f}s)", flush=True)
    LV = break_levels(xyz, S)
    # photo evidence: 54 propagated R7 maps
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; B["chosen"] = chosen
    mz = np.load(R7DIR / "propagated_maps_leaf4crop.npz"); vpm = {k[1:]: mz[k] for k in mz.files if k[1:] != held and k[1:] not in chosen}
    H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; PIX = pixel_ids(pl, sorted(vpm), H2id)
    VOTES = np.zeros((n, 13), np.int32); NVIS = np.zeros(n, np.int16)
    for vid, px in PIX.items():
        ok = px >= 0; NVIS[ok] += 1; lv = np.zeros(n, int); lv[ok] = vpm[vid].ravel()[px[ok]]; m = lv > 0; np.add.at(VOTES, (np.nonzero(m)[0], lv[m]), 1)
    MASKS = {}; LEAFPX = {}
    def masks_of(vid):
        if vid not in MASKS: MASKS[vid] = OS.view_masks(vid, chosen)[0]
        return MASKS[vid]
    print(f"photo votes done ({time.time() - t0:.0f}s)", flush=True)
    # trace tracks for the rachis score (R1), on the R7 final labels
    stem_xy = X[axisF == 1][:, :2].mean(0); T = R1.rachis_tracks(B, Gp, s4, stem_xy); reach = {L: C6.U_MM for L in leaves}
    dense_tr = np.vstack([LD.CC.densify(t["pts"]) for t in B["traces"]]); tr_tree = cKDTree(dense_tr)
    Nf = pl["nrm"].astype(float).copy(); Nf[b] = S["normal"]; tree_all = cKDTree(X)
    # step 1: blue candidates
    present = sorted(set(labF[labF > 0].tolist())); blues = []
    for L in present:
        c3 = V2.PAL.get(L, (200, 200, 200)); h = colorsys.rgb_to_hsv(*(np.array(c3) / 255))[0] * 360
        if P["hue"][0] <= h <= P["hue"][1] and (only is None or L in only):
            mem = np.nonzero(labF == L)[0]; cen = X[mem].mean(0); dtr = {}
            for k in (4, 6):
                tr = [t for t in B["traces"] if t["leaf"] == k]; dtr[k] = round(float(cKDTree(np.vstack([LD.CC.densify(t["pts"]) for t in tr])).query(cen)[0] * 1000), 1) if tr else None
            blues.append(dict(leaf=L, colour="#%02x%02x%02x" % c3, hue=round(h, 1), surfels=int(len(mem)), centroid_mm=(cen * 1000).round(1).tolist(), centroid_to_trace_mm=dtr))
    print("blue candidates:", blues, flush=True)
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, tag="gate_failed_viewing_only", with_3d_surroundings=with3d, weights_user=(with_nb3d(R1.W_DECL) if with3d else R1.W_DECL), labels="R7 vp_leaf4/labels.npz (read only)", r7_diffusion_reproduced=repro, params=P, candidates=blues, per_candidate={})
    prop_label = labF.copy(); prop_kind = np.zeros(n, np.int8); prop_owner = np.zeros(n, np.int16); REASON_BY = {}
    unp_all0 = int((labF <= 0).sum()); unp_blade0 = int(((labF <= 0) & (axisF == 0)).sum())
    for cb in blues:
        Lb = cb["leaf"]; memLb = np.nonzero(labF == Lb)[0]; tLb = cKDTree(X[memLb])
        d_lb, _ = tLb.query(X); region = np.nonzero((labF <= 0) & (d_lb <= P["region_mm"] / 1000))[0]
        white = region[axisF[region] == 0]; ax1 = region[axisF[region] == 1]; ax2 = region[axisF[region] == 2]
        print(f"leaf {Lb}: region {len(region)} (white {len(white)}, axis1 {len(ax1)}, axis2 {len(ax2)})", flush=True)
        # step 2: reasons for white blade surfels
        wb = pos[white]; lb_b = pos[memLb[axisF[memLb] == 0]]; lb_b = lb_b[lb_b >= 0]
        brk = np.full(len(wb), "connected", object)
        for lvl, name in zip(LV, BREAKS[:4]):
            comp_lb = set(lvl[lb_b].tolist()); inside = np.array([lvl[k] in comp_lb for k in wb])
            brk[(brk == "connected") & ~inside] = name
        reason = np.full(len(wb), "other", object)
        det = (s3[wb] > 0) & (s4[wb] == 0); reason[det] = "detached_cleanup"
        con = ~det & (src[wb] == 3); reason[con] = "contested"
        oor = ~det & (src[wb] == 4)
        for k in np.nonzero(oor)[0]: reason[k] = "out_of_reach: " + ("connected, beyond reach" if brk[k] == "connected" else brk[k])
        notc = ~det & ~con & ~oor & ~cand[wb]; reason[notc] = "outside_candidate_range"
        novote = VOTES[white].sum(1) == 0; novis = NVIS[white] == 0
        rows2 = []
        for r in REASONS:
            m = reason == r
            if m.any(): rows2.append(dict(reason=r, count=int(m.sum()), area_mm2=round(area_mm2(pl, white[m]), 1), not_visible_in_any_of_54=int((m & novis).sum()), visible_but_no_leaf_vote=int((m & novote & ~novis).sum()), median_visible_photos=float(np.median(NVIS[white[m]]))))
        brk_tab = dict(Counter(brk.tolist()))
        # contested detail
        cdet = []; k_lb = leaves.index(Lb) if Lb in leaves else None
        for k in np.nonzero(con)[0]:
            s_ = wb[k]; o = np.argsort(C[:, s_]); l1, l2 = leaves[o[0]], leaves[o[1]]; v = VOTES[white[k]]; tv = v.sum(); vo = np.argsort(-v)
            cdet.append(dict(surfel=int(white[k]), best=l1, second=l2, gap_mm=round(float(C[o[1], s_] - C[o[0], s_]), 2), vote_total=int(tv), vote1=(int(vo[0]), round(v[vo[0]] / tv, 2)) if tv else None,
                             vote2=(int(vo[1]), round(v[vo[1]] / tv, 2)) if tv else None, sheet=f"{int(S['index'][s_])}/{int(S['n_sheets'][s_])}", blue_in_top2=bool(Lb in (l1, l2))))
        pairs = Counter(tuple(sorted((d["best"], d["second"]))) for d in cdet)
        csum = dict(n=len(cdet), blue_in_top2=int(sum(d["blue_in_top2"] for d in cdet)), top2_pairs={f"{a}-{b_}": v for (a, b_), v in pairs.most_common(8)},
                    median_gap_mm=(round(float(np.median([d["gap_mm"] for d in cdet])), 2) if cdet else None),
                    blue_vote_share_median=(round(float(np.median([VOTES[d["surfel"]][Lb] / max(VOTES[d["surfel"]].sum(), 1) for d in cdet])), 3) if cdet else None),
                    lower_or_upper_sheet=int(sum(1 for d in cdet if not d["sheet"].endswith("/1"))))
        # axis_kind 2 check
        bl = memLb[axisF[memLb] == 0]; rad_b = np.sqrt(pl["sc"][bl, 0] * pl["sc"][bl, 1]) * 1000; el_b = pl["sc"][bl, :2].max(1) / np.maximum(pl["sc"][bl, :2].min(1), 1e-12)
        ref = dict(radius_p25_mm=float(np.percentile(rad_b, 25)), elong_p75=float(np.percentile(el_b, 75)))
        c0, _, e1, e2, nrm0 = LO.pca_frame(X[memLb]); off = (X - c0) @ nrm0
        hull2 = Delaunay(np.c_[(X[memLb] - c0) @ e1, (X[memLb] - c0) @ e2][ConvexHull(np.c_[(X[memLb] - c0) @ e1, (X[memLb] - c0) @ e2]).vertices])
        all_ax2 = np.nonzero((axisF == 2) & (labF <= 0))[0]
        def ax2_check(idx, memL, refL, nrmL):
            tL = cKDTree(X[memL]); d, _ = tL.query(X[idx]); _, jj = tL.query(X[idx], k=P["k_local"]); loc = Nf[memL][jj]
            locn = loc[:, 0] + 0.0
            for q in range(1, loc.shape[1]): locn = locn + np.where(((locn * loc[:, q]).sum(1) < 0)[:, None], -loc[:, q], loc[:, q])
            rad = np.sqrt(pl["sc"][idx, 0] * pl["sc"][idx, 1]) * 1000; el = pl["sc"][idx, :2].max(1) / np.maximum(pl["sc"][idx, :2].min(1), 1e-12)
            a_loc = angle_deg(Nf[idx], locn); a_mean = angle_deg(Nf[idx], np.broadcast_to(nrmL, Nf[idx].shape)); dtr = tr_tree.query(X[idx])[0] * 1000
            blade_like = (rad >= refL["radius_p25_mm"]) & (a_loc <= P["normal_deg"]) & (dtr >= P["trace_mm"]) & (el <= refL["elong_p75"])
            return dict(d=d * 1000, rad=rad, el=el, a_loc=a_loc, a_mean=a_mean, dtr=dtr, blade_like=blade_like)
        in_fp = hull2.find_simplex(np.c_[(X[all_ax2] - c0) @ e1, (X[all_ax2] - c0) @ e2]) >= 0
        near_ax2 = all_ax2[(d_lb[all_ax2] <= P["near_mm"] / 1000) | (in_fp & (np.abs(off[all_ax2]) <= P["plane_mm"] / 1000))]
        A2 = ax2_check(near_ax2, memLb, ref, nrm0) if len(near_ax2) else None
        def q(a): return [round(float(np.percentile(a, p)), 2) for p in (25, 50, 75)] if len(a) else None
        ax2_sum = dict(n=int(len(near_ax2)), in_region_12mm=int(len(ax2)), reference=ref, blade_like=int(A2["blade_like"].sum()) if A2 else 0,
                       radius_mm_q=q(A2["rad"]) if A2 else None, blade_radius_mm_q=q(rad_b), elong_q=q(A2["el"]) if A2 else None, angle_local_deg_q=q(A2["a_loc"]) if A2 else None,
                       angle_mean_normal_deg_q=q(A2["a_mean"]) if A2 else None, trace_dist_mm_q=q(A2["dtr"]) if A2 else None,
                       each_condition=dict(radius=int((A2["rad"] >= ref["radius_p25_mm"]).sum()), normal=int((A2["a_loc"] <= P["normal_deg"]).sum()), trace=int((A2["dtr"] >= P["trace_mm"]).sum()),
                                           elong=int((A2["el"] <= ref["elong_p75"]).sum())) if A2 else None)
        blade_like_idx = near_ax2[A2["blade_like"]] if A2 else np.zeros(0, int)
        # step 3: groups -> clusters -> evidence
        groups = {r: white[reason == r] for r in REASONS if (reason == r).any()}
        groups["axis2: probably blade"] = blade_like_idx; groups["axis2: other"] = np.setdiff1d(ax2, blade_like_idx); groups["axis1"] = ax1
        coloured = labF > 0; ctree = cKDTree(X[coloured]); cidx = np.nonzero(coloured)[0]; rows3 = []; pass_b = {}; ltree, lidx = ctree, cidx
        for gname, gidx in groups.items():
            if len(gidx) == 0: continue
            if gname.startswith("axis"):
                E = cKDTree(X[gidx]).query_pairs(P["near_mm"] / 1000, output_type="ndarray"); m_ = len(gidx)
                Gc = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(m_, m_)).tocsr() if len(E) else coo_matrix((m_, m_)).tocsr(); cl_ = connected_components(Gc, directed=False)[1]
                units = [gidx[cl_ == u] for u in range(cl_.max() + 1)]
            else:
                wm = np.zeros(nb, bool); wm[pos[gidx]] = True; cid, _ = CR.clusters(xyz, S, wm); units = [b[np.nonzero(cid == u)[0]] for u in range(cid.max() + 1)]
            per = []
            for mem in units:
                blade_unit = bool((axisF[mem] == 0).all()); cands = {Lb}; pairs = defaultdict(list)
                for s_ in mem:
                    for d, j in zip(*ctree.query(X[s_], k=24, distance_upper_bound=P["cand_mm"] / 1000)):
                        if not np.isfinite(d): break
                        t = cidx[j]
                        if blade_unit and axisF[t] == 0 and not CR.sheet_pair_ok(S, pos[s_], pos[t]): continue
                        L = int(labF[t]); cands.add(L)
                        if d <= P["near_mm"] / 1000: pairs[L].append((s_, t, d))
                cands = sorted(cands); v = VOTES[mem].sum(0); vt = int(v[1:].sum()); unseen = vt == 0
                sam = Counter(); seen_sam = 0
                for vid, px in PIX.items():
                    pm = np.unique(px[mem][px[mem] >= 0])
                    if len(pm) < CR.R["min_px"]: continue
                    mk = masks_of(vid)
                    if not mk: continue
                    hit = [h for h, mm in enumerate(mk) if mm.ravel()[pm].mean() >= CR.R["inside"]]
                    if not hit: continue
                    seen_sam += 1
                    for L in cands:
                        if (vid, L) not in LEAFPX: lp_ = np.unique(px[labF == L]); LEAFPX[(vid, L)] = lp_[lp_ >= 0]
                        lp = LEAFPX[(vid, L)]
                        if len(lp) >= CR.R["min_px"] and any(mk[h].ravel()[lp].sum() >= CR.R["min_px"] for h in hit): sam[L] += 1
                st = sum(sam.values())
                ev = dict(vp={L: (v[L] / vt if vt else 0.0) for L in cands}, sam={L: (sam[L] / st if st else 0.0) for L in cands}, sheet={}, rachis={})
                for L in cands:
                    ev["sheet"][L] = CR.continuity(X, dict(normal=Nf), pairs[L])[0] if pairs.get(L) else 0.0
                    if blade_unit: ev["rachis"][L] = R1.rachis_score(xyz, pos[mem], L, T, s4, reach)[0]
                    else: ev["rachis"][L] = rachis_euclid(X, mem, L, T, labF, C6.U_MM)
                out_u = {}; WS = [("user", R1.W_DECL, False), ("photo_first", R1.W_PHOTO, True)]; evs = {}
                if with3d:
                    for r_ in (P["nb3d_mm"],) + tuple(P["nb3d_sens_mm"]):
                        sh = nb3d_share(X, mem, ltree, lidx, labF, r_); evs[r_] = dict(ev, nb3d={L: sh.get(L, 0.0) for L in cands})
                    ev = evs[P["nb3d_mm"]]; WS = [("user", with_nb3d(R1.W_DECL), False), ("photo_first", with_nb3d(R1.W_PHOTO), True)] + [(f"user_r{int(r_)}", with_nb3d(R1.W_DECL), False) for r_ in P["nb3d_sens_mm"]]
                for wname, w, pf in WS:
                    ev_ = evs[float(wname.split("_r")[1])] if "_r" in wname else ev
                    tot = totals(ev_, cands, w, unseen, pf); oth = [tot[L] for L in cands if L != Lb]; mg = tot[Lb] - (max(oth) if oth else 0.0)
                    ok = tot[Lb] >= P["strict_total"] and mg >= P["strict_margin"] and (not (unseen and not pf) or ev_["sheet"][Lb] == 1.0)
                    out_u[wname] = (tot[Lb], mg, ok, max(tot, key=tot.get))
                per.append((mem, out_u, unseen))
            def agg(wname):
                nn = np.array([len(m) for m, _, _ in per]); tt = np.array([o[wname][0] for _, o, _ in per]); mm = np.array([o[wname][1] for _, o, _ in per]); ok = np.array([o[wname][2] for _, o, _ in per])
                best_other = Counter(); [best_other.update({o[wname][3]: len(m)}) for m, o, _ in per]
                return dict(mean_total=round(float((tt * nn).sum() / nn.sum()), 3), mean_margin=round(float((mm * nn).sum() / nn.sum()), 3), n_pass=int(nn[ok].sum()), top_leaf_by_surfels=dict(best_other.most_common(3)))
            rows3.append(dict(group=gname, n=int(len(gidx)), clusters=len(per), area_mm2=round(area_mm2(pl, gidx), 1), unseen_surfels=int(sum(len(m) for m, _, u in per if u)), user=agg("user"), photo_first=agg("photo_first"),
                             **({f"sens_user_r{int(r_)}": agg(f"user_r{int(r_)}") for r_ in P["nb3d_sens_mm"]} if with3d else {})))
            pass_b[gname] = np.concatenate([m for m, o, _ in per if o["user"][2]] or [np.zeros(0, int)])
        # step 4: proposals (a) axis2 probably blade -> nearest labelled leaf within 3 mm, (b) white passing strict (user weights) -> Lb
        if len(blade_like_idx):
            d_, j_ = ctree.query(X[blade_like_idx]); ok = d_ <= P["near_mm"] / 1000; sel = blade_like_idx[ok]; prop_label[sel] = labF[cidx[j_[ok]]]; prop_kind[sel] = 1; prop_owner[sel] = Lb
        for gname, idx in pass_b.items():
            if gname.startswith("axis") or not len(idx): continue
            prop_label[idx] = Lb; prop_kind[idx] = 2; prop_owner[idx] = Lb
        prop_reason = np.full(n, -1, np.int16)                                  # per candidate: reason index (white), 100 axis1, 102 axis2 not proposed
        for r_i, r in enumerate(REASONS):
            m = white[reason == r]; prop_reason[m[prop_kind[m] == 0]] = r_i
        rest1 = ax1[prop_kind[ax1] == 0]; prop_reason[rest1] = 100; rest2 = ax2[prop_kind[ax2] == 0]; prop_reason[rest2] = 102; REASON_BY[f"reason_leaf{Lb}"] = prop_reason
        nbrs = sorted(set(labF[(d_lb <= P["region_mm"] / 1000) & (labF > 0)].tolist()))
        areas = {}
        for L in nbrs:
            a0 = area_mm2(pl, np.nonzero(labF == L)[0]); a1 = area_mm2(pl, np.nonzero(np.where(prop_owner == Lb, prop_label, labF) == L)[0]); areas[str(L)] = dict(before=round(a0, 1), after=round(a1, 1), change=round(a1 - a0, 1))
        res["per_candidate"][str(Lb)] = dict(region=dict(total=int(len(region)), white=int(len(white)), axis1=int(len(ax1)), axis2=int(len(ax2))), reasons=rows2, break_type_all_white=brk_tab,
                                             contested=csum, axis2_check=ax2_sum, groups=rows3, proposals=dict(a_axis2_to_blade=int(((prop_kind == 1) & (prop_owner == Lb)).sum()), b_white_strict=int(((prop_kind == 2) & (prop_owner == Lb)).sum())),
                                             leaf_areas_mm2=areas)
        with open(OUT / f"contested_leaf{Lb}.csv", "w", encoding="utf-8", newline="") as fh:
            if cdet:
                wr = csv.DictWriter(fh, fieldnames=list(cdet[0])); wr.writeheader(); wr.writerows(cdet)
        if A2:
            with open(OUT / f"axis2_check_leaf{Lb}.csv", "w", encoding="utf-8", newline="") as fh:
                wr = csv.writer(fh); wr.writerow(["surfel", "dist_to_leaf_mm", "radius_mm", "elongation", "angle_local_deg", "angle_mean_normal_deg", "trace_dist_mm", "probably_blade"])
                for k, s_ in enumerate(near_ax2): wr.writerow([int(s_), round(A2["d"][k], 2), round(A2["rad"][k], 3), round(A2["el"][k], 2), round(A2["a_loc"][k], 1), round(A2["a_mean"][k], 1), round(A2["dtr"][k], 2), bool(A2["blade_like"][k])])
        print(json.dumps(res["per_candidate"][str(Lb)], ensure_ascii=False, default=float)[:3000], flush=True)
    # whole-individual count of the axis_kind 2 pattern (every R7 leaf, nearest leaf, counted once)
    all_ax2 = np.nonzero((axisF == 2) & (labF <= 0))[0]; ctree = cKDTree(X[labF > 0]); cidx = np.nonzero(labF > 0)[0]; d_, j_ = ctree.query(X[all_ax2]); owner = labF[cidx[j_]]; tot_like = 0; per_leaf = {}
    for L in present:
        memL = np.nonzero(labF == L)[0]; blL = memL[axisF[memL] == 0]
        if len(blL) < 10: continue
        refL = dict(radius_p25_mm=float(np.percentile(np.sqrt(pl["sc"][blL, 0] * pl["sc"][blL, 1]) * 1000, 25)), elong_p75=float(np.percentile(pl["sc"][blL, :2].max(1) / np.maximum(pl["sc"][blL, :2].min(1), 1e-12), 75)))
        cL, _, f1, f2, nL = LO.pca_frame(X[memL]); Q = np.c_[(X[memL] - cL) @ f1, (X[memL] - cL) @ f2]; hL = Delaunay(Q[ConvexHull(Q).vertices])
        cand_ax = all_ax2[owner == L]; fp = (hL.find_simplex(np.c_[(X[cand_ax] - cL) @ f1, (X[cand_ax] - cL) @ f2]) >= 0) & (np.abs((X[cand_ax] - cL) @ nL) <= P["plane_mm"] / 1000)
        dd = cKDTree(X[memL]).query(X[cand_ax])[0]; sel = cand_ax[(dd <= P["near_mm"] / 1000) | fp]
        if not len(sel): per_leaf[str(L)] = 0; continue
        tL = cKDTree(X[memL]); _, jj = tL.query(X[sel], k=P["k_local"]); loc = Nf[memL][jj]; locn = loc[:, 0] + 0.0
        for q_ in range(1, loc.shape[1]): locn = locn + np.where(((locn * loc[:, q_]).sum(1) < 0)[:, None], -loc[:, q_], loc[:, q_])
        rad = np.sqrt(pl["sc"][sel, 0] * pl["sc"][sel, 1]) * 1000; el = pl["sc"][sel, :2].max(1) / np.maximum(pl["sc"][sel, :2].min(1), 1e-12)
        like = (rad >= refL["radius_p25_mm"]) & (angle_deg(Nf[sel], locn) <= P["normal_deg"]) & (tr_tree.query(X[sel])[0] * 1000 >= P["trace_mm"]) & (el <= refL["elong_p75"])
        per_leaf[str(L)] = int(like.sum()); tot_like += int(like.sum())
    res["whole_individual_axis2_blade_like"] = dict(total=tot_like, per_leaf=per_leaf, unlabelled_axis2_total=int(len(all_ax2)))
    unp_all1 = int((prop_label <= 0).sum())
    res["unpainted_share"] = dict(all_before=round(unp_all0 / n, 4), all_after=round(unp_all1 / n, 4), surfels_before=unp_all0, surfels_after=unp_all1, n=n,
                                  blade_unpainted_before=unp_blade0, blade_unpainted_after=int(((prop_label <= 0) & (axisF == 0)).sum()), blade_n=int((axisF == 0).sum()))
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT / "result.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    np.savez_compressed(OUT / "proposal.npz", rows=pl["rows"], r7_label=labF.astype(np.int16), proposed_label=prop_label.astype(np.int16), proposal_kind=prop_kind, proposal_owner=prop_owner, **REASON_BY,
                        reason_names=np.array(list(REASONS)), kind_names=np.array(["none", "a_axis2_to_blade", "b_white_strict"]), axis_kind=axisF.astype(np.int8))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_candidate",)}, ensure_ascii=False, default=float), flush=True)
    print(f"done ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--with3d", action="store_true"); ap.add_argument("--leaf", type=int, nargs="*", default=None); a = ap.parse_args()
    main(a.with3d, set(a.leaf) if a.leaf else None)
