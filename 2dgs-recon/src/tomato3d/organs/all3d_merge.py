"""Steps 4b-6 of the all-individuals 3D-first run (contract build/p1_all_3dfirst_20261027_01, sections 4-7, Amendment A4: settings frozen, user choice "a" 2026-10-28).

Runs inside a Kaggle CPU kernel (stage `merge`) on one individual's `pre` output (geometry.npz, visibility_plant.npz, detect_input.json) and candidate masks (`cands/`, `cands_rf/`):
  4b  SAM votes on adjacent 4a patch pairs: +1 `same` when both patches are >= 70 % inside one candidate mask (fragment repair first), a `different` VETO only when every
      condition of contract section 5.4 holds
  5   conservative merge: >= 3 voting views, vote-weighted mean >= 0.8, no cross pair with a veto in >= 2 views, merged size <= 1.2 x p95 leaflet
  6   sheet-constrained fill with the margin rule, then flags and group confidence
Under-merge is the intended failure. Nothing here reads a human label.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.leafseg import pixels as px
from tomato3d.organs import all3d_first as A
from tomato3d.organs import all_compound as AC
from tomato3d.recon import charuco_a3 as ca

H2 = 1536
VIS_RES = 1512
P = dict(min_px=30, inside=0.70, overlap_max=0.01, adj_mm=4.0, frag_cover=0.30, touch_px=2.0, vis_patch=0.70, depth_mult=1.5,
         min_views=3, score_min=0.8, conflict_views=2, min_pairs=2, single_pair_views=5, size_mult=1.2,
         fill_radius_mm=4.0, fill_k=10, fill_normal_deg=35.0, fill_passes=5, fill_margin=2, low_vis_views=3, high_views=5, high_score=0.9, seconds_per_item=3.0)
FLAG = dict(unassigned=1, contested=2, multi_sheet_patch=4, low_vote=8, sam_conflict=16, oversize=32, low_visibility_region=64)
CONF = {"high": 2, "medium": 1, "low": 0}


# ---------------------------------------------------------------- geometry helpers
def project(xyz, po, view, res=VIS_RES):
    K = ca.scaled_K(np.array(po["K_undist"]), po["undistorted_res"], res); R = np.array(po["views"][view]["R_cw"]); t = np.array(po["views"][view]["t_cw"])
    xc = xyz @ R.T + t; z = xc[:, 2]; zs = np.where(z > 0, z, 1e-9); s = H2 / res
    return np.round((K[0, 0] * xc[:, 0] / zs + K[0, 2]) * s).astype(int), np.round((K[1, 1] * xc[:, 1] / zs + K[1, 2]) * s).astype(int), z


def adjacency_edges(xyz, S, patch):
    """Pairs of adjacent patches (3D neighbours within 4 mm, same sheet index inside multi-sheet columns) and, per pair, the surfel edges that connect them."""
    E = cKDTree(xyz).query_pairs(P["adj_mm"] / 1000, output_type="ndarray"); i, j = E[:, 0], E[:, 1]; pi, pj = patch[i], patch[j]
    ok = (pi >= 0) & (pj >= 0) & (pi != pj); multi = (S["n_sheets"][i] > 1) & (S["n_sheets"][j] > 1); ok &= ~multi | (S["index"][i] == S["index"][j])
    i, j, pi, pj = i[ok], j[ok], pi[ok], pj[ok]; a, b = np.minimum(pi, pj), np.maximum(pi, pj); key = a.astype(np.int64) * (int(patch.max()) + 1) + b
    uk, inv = np.unique(key, return_inverse=True); pairs = [(int(k // (patch.max() + 1)), int(k % (patch.max() + 1))) for k in uk]
    order = np.argsort(inv, kind="stable"); cuts = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(pairs)))]
    return pairs, i[order], j[order], cuts


# ---------------------------------------------------------------- 4b: votes in one photo
def decode_masks(path):
    if not Path(path).exists():
        return []
    return [px.rle_decode(c["rle"]) for c in json.load(open(path, encoding="utf-8"))["candidates"]]


def view_votes(xyz, patch, sizes, vis, po, view, masks, pairs, ei, ej, cuts, t_sheet_mm):
    """{pair index: +1 / -1} for one photo (contract section 5.4)."""
    u, w, z = project(xyz, po, view); npatch = len(sizes)
    ok = vis & (z > 0) & (u >= 0) & (w >= 0) & (u < H2) & (w < H2) & (patch >= 0)
    if not ok.any() or not masks:
        return {}, dict(masks=len(masks), fragments=0)
    lin = (w[ok] * H2 + u[ok]).astype(np.int64); pp = patch[ok]; key = np.unique(pp.astype(np.int64) * (H2 * H2) + lin); Pk = key // (H2 * H2); Lk = key % (H2 * H2)
    npx = np.bincount(Pk, minlength=npatch); visfrac = np.bincount(patch[vis & (patch >= 0)], minlength=npatch) / np.maximum(sizes, 1)
    flat = [m.ravel() for m in masks]
    F = np.array([np.bincount(Pk[f[Lk]], minlength=npatch) for f in flat], float) / np.maximum(npx, 1)[None]
    # fragment repair: one patch covered >= 30 % by each of two touching masks -> those masks are one leaflet in this photo
    frag = 0; parent = list(range(len(masks)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    dil = {}
    def touching(a, b):
        if a not in dil:
            dil[a] = cv2.dilate(masks[a].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return bool((dil[a] & masks[b]).any())
    for p in range(npatch):
        ms = np.nonzero(F[:, p] >= P["frag_cover"])[0]
        if 2 <= len(ms) <= 6:
            for x in range(len(ms)):
                for y in range(x + 1, len(ms)):
                    a, b = find(int(ms[x])), find(int(ms[y]))
                    if a != b and touching(int(ms[x]), int(ms[y])):
                        parent[b] = a; frag += 1
    roots = sorted(set(find(m) for m in range(len(masks))))
    if frag:
        merged = []
        for r in roots:
            mem = [m for m in range(len(masks)) if find(m) == r]
            merged.append(np.any([masks[m] for m in mem], axis=0) if len(mem) > 1 else masks[mem[0]])
        masks = merged; flat = [m.ravel() for m in masks]; F = np.array([np.bincount(Pk[f[Lk]], minlength=npatch) for f in flat], float) / np.maximum(npx, 1)[None]
    area = np.array([m.sum() for m in masks]); best = F.argmax(0); bestf = F.max(0); out = {}; ov = {}
    depth_lim = P["depth_mult"] * t_sheet_mm / 1000
    for e, (p, q) in enumerate(pairs):
        if npx[p] < P["min_px"] or npx[q] < P["min_px"]:
            continue
        if ((F[:, p] >= P["inside"]) & (F[:, q] >= P["inside"])).any():
            out[e] = 1; continue
        if not (bestf[p] >= P["inside"] and bestf[q] >= P["inside"] and best[p] != best[q]):
            continue
        if visfrac[p] < P["vis_patch"] or visfrac[q] < P["vis_patch"]:
            continue
        a, b = int(best[p]), int(best[q])
        if (a, b) not in ov:
            ov[(a, b)] = int((masks[a] & masks[b]).sum())
        if ov[(a, b)] > P["overlap_max"] * min(area[a], area[b]):
            continue
        s, t_ = cuts[e], cuts[e + 1]; i, j = ei[s:t_], ej[s:t_]; both = vis[i] & vis[j] & (z[i] > 0) & (z[j] > 0)
        if not both.any():
            continue
        touch = np.hypot((u[i] - u[j])[both], (w[i] - w[j])[both]).min() <= P["touch_px"]
        step = float(np.median(np.abs(z[i] - z[j])[both]))
        if touch and step <= depth_lim:
            out[e] = -1
    return out, dict(masks=len(masks), fragments=frag)


# ---------------------------------------------------------------- 5: conservative merge
def conservative_merge(pairs, s, d, vs, sizes, cap):
    """Average-linkage merge of the group pair with the highest vote-weighted mean score among those that pass every rule; recomputed after each merge."""
    n = len(sizes); gof = np.arange(n); gsize = {p: int(sizes[p]) for p in range(n)}; log = []
    voted = [e for e in range(len(pairs)) if s[e] + d[e] > 0]
    while True:
        inter = {}
        for e in voted:
            ga, gb = gof[pairs[e][0]], gof[pairs[e][1]]
            if ga != gb:
                inter.setdefault((min(ga, gb), max(ga, gb)), []).append(e)
        best = None; rej = Counter()
        for (ga, gb), es in inter.items():
            nv = np.array([s[e] + d[e] for e in es]); views = set().union(*[vs[e] for e in es])
            if len(views) < P["min_views"]:
                rej["views"] += 1; continue
            if not (len(es) >= P["min_pairs"] or nv.max() >= P["single_pair_views"]):
                rej["support"] += 1; continue
            mean = float(sum(s[e] - d[e] for e in es) / nv.sum())
            if mean < P["score_min"]:
                rej["score"] += 1; continue
            if any(d[e] >= P["conflict_views"] for e in es):
                rej["veto"] += 1; continue
            if gsize[ga] + gsize[gb] > cap:
                rej["size"] += 1; continue
            if best is None or mean > best[0]:
                best = (mean, ga, gb, len(es), len(views))
        if best is None:
            return gof, log, dict(rej)
        mean, ga, gb, ne, nviews = best; gof[gof == gb] = ga; gsize[ga] += gsize.pop(gb); log.append(dict(keep=int(ga), absorb=int(gb), mean_score=round(mean, 4), pairs=ne, views=nviews, size=gsize[ga]))


# ---------------------------------------------------------------- 6: fill with margin rule
def fill_margin(lab, xyz, S):
    nrm = S["normal"] / np.maximum(np.linalg.norm(S["normal"], axis=1, keepdims=True), 1e-12); cmin = np.cos(np.radians(P["fill_normal_deg"])); multi = S["n_sheets"] > 1; idx = S["index"]
    cur = lab.copy(); contested = np.zeros(len(lab), bool); per_pass = []
    for p in range(1, P["fill_passes"] + 1):
        asg = np.nonzero(cur >= 0)[0]; un = np.nonzero(cur < 0)[0]
        if not len(un) or not len(asg):
            break
        dd, jj = cKDTree(xyz[asg]).query(xyz[un], k=P["fill_k"], distance_upper_bound=P["fill_radius_mm"] / 1000); new = cur.copy(); n = 0
        for r, u in enumerate(un):
            ok = np.isfinite(dd[r])
            if not ok.any():
                continue
            nb = asg[jj[r][ok]]; m = (~(multi[u] & multi[nb]) | (idx[nb] == idx[u])) & (np.abs(nrm[nb] @ nrm[u]) >= cmin); c = Counter(cur[nb[m]].tolist()).most_common(2)
            if not c:
                continue
            second = c[1][1] if len(c) > 1 else 0
            if c[0][1] - second < P["fill_margin"]:
                contested[u] = True; continue
            new[u] = c[0][0]; contested[u] = False; n += 1
        cur = new; per_pass.append(n)
        if n == 0:
            break
    return cur, contested & (cur < 0), per_pass


# ---------------------------------------------------------------- the individual
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def run(iid, out=None):
    t0 = time.time(); c = A.cell(iid); out = Path(out or c / "merge"); out.mkdir(parents=True, exist_ok=True)
    pl = AC.load_individual(iid); geo = np.load(c / "work" / "geometry.npz"); assert np.array_equal(geo["rows"], pl["rows"]), "geometry rows differ from the surfels"
    gj = json.load(open(c / "geometry.json", encoding="utf-8")); man = json.load(open(c / "inputs.json", encoding="utf-8")); di = json.load(open(c / "detect_input.json", encoding="utf-8"))
    vz = np.load(c / "work" / "visibility_plant.npz"); n_all = int(vz["n"]); vviews = [str(v) for v in vz["views"]]
    Vall = np.unpackbits(vz["vis"], axis=0)[:n_all].astype(bool); assert np.array_equal(vz["rows"], pl["rows"])
    b = geo["b"]; xyz = pl["xyz"][b]; patch = geo["patch_4a"].astype(np.int64); S = dict(normal=geo["normal"], n_sheets=geo["n_sheets"], index=geo["index"]); Vb = Vall[b]
    nb_ = len(b); npatch = int(patch.max()) + 1; sizes = np.bincount(patch[patch >= 0], minlength=npatch); t_sheet = gj["t_sheet_mm"]; th = gj["thresholds"]; cap = th["size_cap"]
    p95_n = int(round(th["size_prior"]["p95"] * gj["density_per_mm2"]))
    held = man.get("label_heldout_excluded_from_votes"); vote_views = [v for v in man["views"] if v != held and v in vviews]
    pairs, ei, ej, cuts = adjacency_edges(xyz, S, patch); npair = len(pairs); s = np.zeros(npair, int); d = np.zeros(npair, int); vs = [set() for _ in range(npair)]; layers = Counter(); vlog = {}
    for v in vote_views:
        stem = Path(v).stem; k = vviews.index(v); masks = []
        for sub, suf in (("cands", "_cands.json"), ("cands_rf", "_rf.json")):
            ms = decode_masks(c / sub / f"{stem}{suf}")
            if ms:
                layers[sub] += len(ms); masks += ms
        res, info = view_votes(xyz, patch, sizes, Vb[:, k], pl["po"], v, masks, pairs, ei, ej, cuts, t_sheet); vlog[v] = info
        for e, sign in res.items():
            vs[e].add(k)
            if sign > 0:
                s[e] += 1
            else:
                d[e] += 1
    gof, mlog, rej = conservative_merge(pairs, s, d, vs, sizes, cap)
    lab0 = np.where(patch >= 0, gof[np.maximum(patch, 0)], -1); lab, contested, per_pass = fill_margin(lab0, xyz, S)
    # ---- flags
    flags = np.zeros(nb_, np.int16); flags[lab < 0] |= FLAG["unassigned"]; flags[contested] |= FLAG["contested"]; flags[contested] &= ~FLAG["unassigned"]
    ms_patch = np.zeros(npatch, bool)
    for g in range(npatch):
        idx_ = np.nonzero(patch == g)[0]
        if len(idx_) >= 6:
            m = S["n_sheets"][idx_] > 1; ms_patch[g] = bool(m.any() and len(np.unique(S["index"][idx_][m])) > 1)
    flags[(patch >= 0) & ms_patch[np.maximum(patch, 0)]] |= FLAG["multi_sheet_patch"]
    gsz = np.bincount(lab[lab >= 0], minlength=int(lab.max()) + 1 if (lab >= 0).any() else 1)
    flags[(lab >= 0) & (gsz[np.maximum(lab, 0)] > p95_n)] |= FLAG["oversize"]
    nvis = Vb[:, [vviews.index(v) for v in man["views"] if v in vviews]].sum(1); flags[nvis < P["low_vis_views"]] |= FLAG["low_visibility_region"]
    # evidence per group and per-patch conflict
    ev = {}; score = {}
    for e, (p, q) in enumerate(pairs):
        for g in {int(gof[p]), int(gof[q])}:
            ev.setdefault(g, set()).update(vs[e])
        if gof[p] == gof[q] and s[e] + d[e] > 0:
            g = int(gof[p]); a = score.setdefault(g, [0, 0]); a[0] += s[e] - d[e]; a[1] += s[e] + d[e]
            if d[e] > 0:
                for pp in (p, q):
                    flags[patch == pp] |= FLAG["sam_conflict"]
    low_vote_groups = [g for g in np.unique(gof) if len(ev.get(int(g), ())) < P["min_views"]]
    for g in low_vote_groups:
        flags[(lab == g)] |= FLAG["low_vote"]
    conf = {}
    for g in np.unique(lab[lab >= 0]):
        m = lab == g; any_flag = bool((flags[m] != 0).any()); sc = (score[int(g)][0] / max(score[int(g)][1], 1)) if int(g) in score else 1.0
        conf[int(g)] = "low" if any_flag else ("high" if len(ev.get(int(g), ())) >= P["high_views"] and sc >= P["high_score"] else "medium")
    # flagged regions: connected components of unassigned / contested surfels
    fl = np.nonzero(flags & (FLAG["unassigned"] | FLAG["contested"]))[0]; regions = 0
    if len(fl):
        E = cKDTree(xyz[fl]).query_pairs(P["adj_mm"] / 1000, output_type="ndarray"); nc, comp = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(fl), len(fl))), directed=False)
        regions = int(nc)
    flagged_groups = sum(1 for g, cf in conf.items() if cf == "low"); n_groups = len(conf); minutes = (n_groups + flagged_groups + regions) * P["seconds_per_item"] / 60
    # ---- outputs
    full_group = np.full(len(pl["xyz"]), -1, np.int32); full_group[b] = lab; full_conf = np.zeros(len(pl["xyz"]), np.int8); full_flags = np.zeros(len(pl["xyz"]), np.int16); full_flags[b] = flags
    for g, cf in conf.items():
        full_conf[b[lab == g]] = CONF[cf] + 1
    np.savez_compressed(out / "groups.npz", rows=pl["rows"], xyz=pl["xyz"].astype(np.float32), group=full_group, confidence=full_conf, flags=full_flags, axis_like=~geo["blade"], patch_4a=geo["patch_4a"], b=b)
    with open(out / "flags.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["flag", "bit", "group", "n_surfels", "x_m", "y_m", "z_m"])
        for name, bit in FLAG.items():
            m = (flags & bit) != 0
            if not m.any():
                continue
            if name in ("unassigned", "contested") and len(fl):
                sel = np.nonzero(m)[0]; E2 = cKDTree(xyz[sel]).query_pairs(P["adj_mm"] / 1000, output_type="ndarray"); nc2, comp2 = connected_components(coo_matrix((np.ones(len(E2)), (E2[:, 0], E2[:, 1])), shape=(len(sel), len(sel))), directed=False)
                for cc in range(nc2):
                    mm = sel[comp2 == cc]; wr.writerow([name, bit, "", len(mm), *np.round(xyz[mm].mean(0), 4)])
            else:
                for g in np.unique(lab[m]):
                    mm = np.nonzero(m & (lab == g))[0]; wr.writerow([name, bit, int(g), len(mm), *np.round(xyz[mm].mean(0), 4)])
    shares = {k: round(float(((flags & v) != 0).mean()), 4) for k, v in FLAG.items()}; cnt = Counter(conf.values())
    summ = dict(labels=A.LABELS, individual=iid, tag=A.RIM_NOTE.get(iid), settings=dict(P), flag_bits=FLAG, surfels_blade=int(nb_), patches_4a=int(npatch), adjacent_pairs=int(npair),
                voting_views=len(vote_views), held_out_view_excluded=held, pose_dropped_views=man.get("pose_dropped_views"), layers_used=dict(layers), candidate_masks_per_view_median=float(np.median([x["masks"] for x in vlog.values()])) if vlog else 0,
                fragment_repairs=int(sum(x["fragments"] for x in vlog.values())), pairs_same_votes=int(s.sum()), pairs_veto_votes=int(d.sum()), pairs_with_votes=int(((s + d) > 0).sum()), pairs_with_3_views=int((np.array([len(x) for x in vs]) >= 3).sum()),
                merges=len(mlog), rejected=rej, groups=n_groups, group_confidence=dict(cnt), unassigned_share=shares["unassigned"], flagged_surfel_shares=shares, flagged_regions=regions, flagged_groups=flagged_groups,
                confident_surfel_share=round(float(((flags == 0) & (lab >= 0)).mean()), 4), group_size=dict(median=int(np.median(list(Counter(lab[lab >= 0].tolist()).values()))) if n_groups else 0, p95_leaflet_surfels=p95_n, cap=int(cap)),
                fill_passes=per_pass, estimated_user_minutes=round(minutes, 1), crops="not produced in this stage", t_sheet_mm=t_sheet, code=dict(this=sha(__file__), geometry=sha(Path(A.__file__))), seconds=round(time.time() - t0))
    json.dump(summ, open(out / "summary.json", "w", encoding="utf-8"), indent=1, default=float)
    json.dump(dict(merges=mlog), open(out / "merge_log.json", "w", encoding="utf-8"), default=float)
    return summ
