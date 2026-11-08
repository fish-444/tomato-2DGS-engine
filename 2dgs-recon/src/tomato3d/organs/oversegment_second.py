"""Labelled second attempt (`tuned_on_R3-1b`) of the 3D-first SAM merge: group-level consensus merging (average linkage) + sheet-constrained fill of the
unassigned blade surfels. Contract: build/p1_stacked_sheets_retry_20261026_01/CONTRACT.md, Amendment A2. 4a patches and the 4b vote code are reused unchanged.

User labels are used only to score (and for the labelled proxy row), never to vote, merge or fill.

usage: python -m tomato3d.organs.oversegment_second
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from tomato3d.organs import light_organs as LO
from tomato3d.organs import oversegment as OS
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vp_recluster as RC

OUT2 = OS.OUT / "oversegment" / "second_attempt"
LABELS = dict(OS.LABELS, attempt="tuned_on_R3-1b", grouping="oversegment_3d_first_consensus_fill")
C1 = dict(min_pairs=2, single_pair_views=5, score_min=0.6, conflict_views=2, size_pct=95, size_mult=1.5)
C2 = dict(radius_mm=4.0, k=10, normal_deg=35.0, passes=5)
REPRO = dict(edges=880, edges_with_votes=557, same_votes=3272, different_votes=566)  # A1 3,225 / 551 = the subset of pairs with human numbers on both sides (A2 note)


def vote_views(B):
    """The same 64 views as A1 step 4b (training views minus the held-out photo, plus the fresh reconstruction held-out views)."""
    from tomato3d.organs import stacked_sheets as SS
    pl, b = B["pl"], B["b"]; W = np.load(SR.FIRST / "contributions.npz")["W"]; views = pl["views"]; con = json.load(open(OS.ROOT / "build/p1_recon_r3-1_20261022_01/contract.json"))
    _, _, _, op = SS.load(); vv = {}
    for v in views:
        vid = v[9:18]
        if vid != B["held"]: vv[vid] = ((W[:, views.index(v)] >= SR.P["w_min"])[b], None)
    for v in con["heldout_fresh"]:
        vv[v[9:18]] = ((SS.contributions(pl, op, v) >= SR.P["w_min"])[b], None)
    vv.pop(B["held"], None); vv.pop("060551789", None); return vv


def pair_counts(V, n):
    s = np.zeros(n, int); d = np.zeros(n, int)
    for out in V.values():
        for e, (v, _) in out.items():
            if v > 0: s[e] += 1
            else: d[e] += 1
    return s, d


def consensus_merge(pairs, s, d, sizes, cap):
    """Change 1: merge the group pair with the highest vote-weighted mean score among those passing rules (1)-(4); recompute after every merge."""
    n = len(sizes); gof = np.arange(n); gsize = {p: int(sizes[p]) for p in range(n)}; log = []
    voted = [e for e in range(len(pairs)) if s[e] + d[e] > 0]; conflict = {e for e in voted if d[e] >= C1["conflict_views"] and d[e] > s[e]}
    while True:
        inter = {}
        for e in voted:
            ga, gb = gof[pairs[e][0]], gof[pairs[e][1]]
            if ga != gb: inter.setdefault((min(ga, gb), max(ga, gb)), []).append(e)
        best = None; rej = Counter()
        for (ga, gb), es in inter.items():
            nv = np.array([s[e] + d[e] for e in es])
            if not (len(es) >= C1["min_pairs"] or nv.max() >= C1["single_pair_views"]): rej["support"] += 1; continue
            mean = float(sum(s[e] - d[e] for e in es) / nv.sum())
            if mean < C1["score_min"]: rej["score"] += 1; continue
            if any(e in conflict for e in es): rej["conflict"] += 1; continue
            if gsize[ga] + gsize[gb] > cap: rej["size"] += 1; continue
            if best is None or mean > best[0]: best = (mean, ga, gb, len(es))
        if best is None: return gof, log, dict(rej)
        mean, ga, gb, ne = best; gof[gof == gb] = ga; gsize[ga] += gsize.pop(gb); log.append(dict(keep=int(ga), absorb=int(gb), mean_score=round(mean, 4), pairs=ne, size=gsize[ga]))


def fill(lab, xyz, S):
    """Change 2: assign unlabelled surfels from the previous pass's labels (4 mm, same sheet index in multi-sheet columns, normal within 35 deg, majority of the 10 nearest)."""
    nrm = S["normal"] / np.maximum(np.linalg.norm(S["normal"], axis=1, keepdims=True), 1e-12); cmin = np.cos(np.radians(C2["normal_deg"])); multi = S["n_sheets"] > 1; idx = S["index"]
    cur = lab.copy(); filled_pass = np.zeros(len(lab), int); per_pass = []
    for p in range(1, C2["passes"] + 1):
        asg = np.nonzero(cur >= 0)[0]; un = np.nonzero(cur < 0)[0]
        if not len(un) or not len(asg): break
        dd, jj = cKDTree(xyz[asg]).query(xyz[un], k=C2["k"], distance_upper_bound=C2["radius_mm"] / 1000); new = cur.copy(); n = 0
        for r, u in enumerate(un):
            ok = np.isfinite(dd[r])
            if not ok.any(): continue
            nb = asg[jj[r][ok]]; m = (~(multi[u] & multi[nb]) | (idx[nb] == idx[u])) & (np.abs(nrm[nb] @ nrm[u]) >= cmin)
            c = Counter(cur[nb[m]].tolist()).most_common(2)
            if not c or (len(c) > 1 and c[0][1] == c[1][1]): continue
            new[u] = c[0][0]; filled_pass[u] = p; n += 1
        cur = new; per_pass.append(n)
        if n == 0: break
    return cur, filled_pass, per_pass


def heldout(B, group_s, glab):
    pl, b = B["pl"], B["b"]; lab = np.array([glab.get(int(g), 0) if g >= 0 else 0 for g in group_s], np.int16); full = np.zeros(len(pl["xyz"]), np.int16); full[b] = lab
    H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = f"20261021_{B['held']}_iOS.jpg"; col = H2id[:, pl["views"].index(hv)]
    raw, _ = UL.label_maps(UL.OUT, [B["held"]]); hm = UL.remap(raw, B["nmap"])[B["held"]]; hm = np.where(hm > 0, hm, 0).astype(np.int16)
    ren = UL.render_ids(pl, full, hv, col); rows, g = UL.score_heldout(hm, ren); cov = float(((ren > 0) & (hm > 0)).sum() / max((hm > 0).sum(), 1))
    flags = {str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rows.items()}
    return dict(rows={str(L): v for L, v in rows.items()}, summary=g, coverage=round(cov, 4), flags=flags), lab


def over_merged(group_s, B, lmaps):
    """Group ids counted as over-merges by OS.leaflet_metrics (same rule), with the view holding most labelled pixels."""
    per_gv = {}
    for vid, p in B["prj"].items():
        vis = B["VISb"][:, p["k"]] & p["inside"] & (group_s >= 0); lid = lmaps[vid][p["wi"], p["ui"]]; m = vis & (lid > 0)
        for g, l in zip(group_s[m], lid[m]): per_gv.setdefault((int(g), vid), Counter())[int(l)] += 1
    flags = {}; best = {}
    for (g, vid), c in per_gv.items():
        n = sum(c.values())
        if n < OS.B4["min_px"]: continue
        flags.setdefault(g, []).append(sum(1 for v in c.values() if v / n >= 0.20) >= 2)
        if n > best.get(g, ("", 0))[1]: best[g] = (vid, n)
    return {g: best[g][0] for g, f in flags.items() if sum(f) >= len(f) / 2 and sum(f) > 0}


def overmerge_image(vid, p, g, group_s, lmaps, VISb, path):
    img = Image.open(LO.DEFAULTS["poses"].parent / "undist_3024" / SR.AP.view_file(vid)).convert("RGB").resize((UL.H2, UL.H2), Image.BILINEAR)
    base = np.asarray(img).astype(float) * 0.35; a = base.copy(); b_ = base.copy(); lid = lmaps[vid]; rng = np.random.default_rng(0); pal = rng.integers(60, 255, (int(lid.max()) + 2, 3))
    vis = VISb[:, p["k"]] & p["inside"]; m = vis & (group_s == g); under = lid[p["wi"][m], p["ui"][m]]
    a[p["wi"][m], p["ui"][m]] = np.where(under[:, None] > 0, pal[under], 255)
    for L in np.unique(lid[lid > 0]): b_[lid == L] = b_[lid == L] * 0.4 + pal[L] * 0.6
    b_[p["wi"][m], p["ui"][m]] = 255
    ys, xs = p["wi"][m], p["ui"][m]; y0, y1 = max(int(ys.min()) - 60, 0), min(int(ys.max()) + 60, UL.H2); x0, x1 = max(int(xs.min()) - 60, 0), min(int(xs.max()) + 60, UL.H2)
    Image.fromarray(np.hstack([a[y0:y1, x0:x1], b_[y0:y1, x0:x1]]).clip(0, 255).astype(np.uint8)).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT2); a = ap.parse_args(argv); out = a.out; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    B = OS.load_base(); pl, b = B["pl"], B["b"]; patch = np.load(OS.OUT / "oversegment" / "patches_4a.npz")["patch"]; npatch = int(patch.max()) + 1
    pairs = OS.adjacency(B["xyz"], B["S"], patch); V, used = OS.votes(B, patch, pairs, vote_views(B)); s, d = pair_counts(V, len(pairs))
    repro = dict(edges=len(pairs), edges_with_votes=int(((s + d) > 0).sum()), same_votes=int(s.sum()), different_votes=int(d.sum()))
    print("reproduction", repro, f"({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(out / "votes_4b.npz", pairs=np.array(pairs, int), same=s, different=d, views=np.array(sorted(V)))
    if repro != REPRO:
        json.dump(dict(**LABELS, stopped="4b reproduction mismatch", reproduced=repro, expected=REPRO), open(out / "result.json", "w", encoding="utf-8"), indent=1); print("STOP: mismatch"); return
    lc = B["lc"]; cap = C1["size_mult"] * float(np.percentile(lc, C1["size_pct"])); sizes = np.bincount(patch[patch >= 0], minlength=npatch)
    gof, mlog, rej = consensus_merge(pairs, s, d, sizes, cap); gid = {g: k for k, g in enumerate(sorted(set(gof.tolist())))}; group_p = np.array([gid[g] for g in gof])
    pre = np.where(patch >= 0, group_p[np.maximum(patch, 0)], -1)
    post, fpass, per_pass = fill(pre, B["xyz"], B["S"]); patch_f, ppass, per_pass_p = fill(patch, B["xyz"], B["S"])
    print(f"merges {len(mlog)}, groups {len(gid)}, fill per pass {per_pass} ({time.time() - t0:.0f}s)", flush=True)
    lmaps = OS.leaflet_maps(B["chosen"]); S = B["S"]; upper = (S["n_sheets"] > 1) & (S["index"] == S["n_sheets"] - 1); lower = (S["n_sheets"] > 1) & (S["index"] == 0)
    def lm(gs): return dict(all=OS.leaflet_metrics(gs, B, lmaps), upper_sheet=OS.leaflet_metrics(gs, B, lmaps, upper), lower_sheet=OS.leaflet_metrics(gs, B, lmaps, lower))
    L_pre, L_post = lm(pre), lm(post)
    rows_pre, pur_pre = SR.purity(pre, B["prj"], B["umaps"], B["VISb"]); glab = {g: r["majority"] for g, r in rows_pre.items()}
    H_pre, _ = heldout(B, pre, glab); H_post, lab_post = heldout(B, post, glab)
    # fill purity: filled surfels' labelled projections vs their group's pre-fill majority number
    filled = (pre < 0) & (post >= 0); hit = tot = 0
    for vid, p in B["prj"].items():
        m = B["VISb"][:, p["k"]] & p["inside"] & filled; ul = B["umaps"][vid][p["wi"], p["ui"]]; m &= ul > 0
        mj = np.array([glab.get(int(g), -1) for g in post[m]]); hit += int((mj == ul[m]).sum()); tot += int(m.sum())
    ppg = np.bincount(group_p); om = over_merged(post, B, lmaps); assert len(om) == L_post["all"]["over_merges"]
    (out / "overmerge").mkdir(exist_ok=True)
    for g, vid in sorted(om.items()): overmerge_image(vid, B["prj"][vid], g, post, lmaps, B["VISb"], out / "overmerge" / f"group{g:03d}_{vid}.png")
    s4a = json.load(open(OS.OUT / "oversegment" / "step4a.json", encoding="utf-8")); La = L_post["all"]; g_h = H_post["summary"]
    passed = s4a["gate_4a_passed"] and (La["leaflet_purity"] or 0) >= 0.85 and La["over_merge_share"] <= 0.10 and g_h["median_iou"] > SR.PASS["ref_iou"] and g_h["max_confusion"] <= SR.PASS["ref_conf"]
    nb = len(patch); res = dict(**LABELS, contract="Amendment A2", c1=C1, c2=C2, size_cap_surfels=round(cap, 1), human_leaflet_p95=float(np.percentile(lc, 95)), reproduced_4b=repro,
                                sam_veto_addendum_applied=False, patches=npatch, merges=len(mlog), groups=len(gid), rejections_at_stop=rej,
                                patches_per_group=dict(median=float(np.median(ppg)), p90=float(np.percentile(ppg, 90)), max=int(ppg.max()), singletons=int((ppg == 1).sum()), hist=dict(Counter(ppg.tolist()))),
                                unassigned=dict(before=round(float((pre < 0).mean()), 4), after=round(float((post < 0).mean()), 4), filled_per_pass=per_pass, patch_fill_per_pass=per_pass_p, blade_surfels=nb),
                                fill_purity=dict(purity=(round(hit / tot, 4) if tot else None), labelled_projections=tot, note="reported only"),
                                leaflet_level=dict(pre_fill=L_pre, post_fill=L_post), leaf_number_purity_pre_fill=pur_pre,
                                scoring_proxy_uses_human_numbers=dict(post_fill=H_post, pre_fill=H_pre), reference_rows=dict(A_B_C_Dprime=[0.517, 0.393, 0.771], V_P=[0.546, 0.438]),
                                first_attempt=dict(groups=127, leaflet_purity=0.858, over_merge_share=0.216, heldout=[0.386, 0.383, 0.438]),
                                over_merge_cases={str(g): v for g, v in om.items()}, passed=bool(passed), tag=("gate_passed" if g_h["gate_pass"] else "gate_failed_viewing_only"),
                                merge_log=mlog, seconds=round(time.time() - t0))
    json.dump(res, open(out / "result.json", "w", encoding="utf-8"), indent=1, default=float)
    np.savez_compressed(out / "patches_filled.npz", rows=pl["rows"][b], patch=patch, patch_filled=patch_f, fill_pass=ppass)
    if passed: np.savez_compressed(out / "groups_filled.npz", rows=pl["rows"][b], patch=patch, group_of_patch=group_p, group=post, fill_pass=fpass, proxy_leaf=lab_post)
    np.savez_compressed(out / "viz_groups.npz", rows=pl["rows"][b], pre=pre, post=post, proxy_leaf=lab_post)
    print(json.dumps({k: v for k, v in res.items() if k not in ("merge_log", "leaflet_level")}, indent=1, default=float)); print("leaflet", json.dumps(res["leaflet_level"], indent=1))


if __name__ == "__main__":
    main()
