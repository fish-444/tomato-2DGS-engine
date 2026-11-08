"""Stacked-sheet separation, retry with the corrected premise (contract build/p1_stacked_sheets_retry_20261026_01/CONTRACT.md).

Step 1 finds the sheets in each surfel's column (peaks of the normal-offset histogram), Step 2 joins surfels into sheet patches without crossing between
sheets, Step 3 checks the near sheet of every multi-sheet column against what the labelled photos show in front, Step 4 numbers each patch by the V-P vote
pooled over the patch and scores the held-out photo once with the existing code.

usage: python -m tomato3d.organs.sheet_retry
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import light_organs as LO
from tomato3d.organs import stacked_sheets as SS
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.organs import vp_recluster as RC

ROOT = LO.ROOT
OUT = ROOT / "build/p1_stacked_sheets_retry_20261026_01"
FIRST = ROOT / "build/p1_stacked_sheets_20261026_01"
STATUS = "R3-1b_unaccepted"; GROUPING = "stacked_sheets_retry"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping=GROUPING)
T_SHEET_MM = 1.521
P = dict(plane_mm=4.0, lateral_mm=4.0, range_mm=12.0, bin_mm=0.5, sigma_mm=0.5, sep_mm=2.0, mass_min=0.15,
         edge_mm=3.0, angle_deg=30.0, offset_mm=max(T_SHEET_MM, 1.0), small_pct=5, vote_share=0.6, w_min=0.1)
PASS = dict(front_min=0.85, purity_min=0.85, ref_iou=0.517, ref_conf=0.393)
SENS = dict(sep_mm=(1.5, 2.0, 2.5), offset_mult=(0.5, 1.0, 2.0))


# ---------------------------------------------------------------- Step 1: sheets in each column
def find_peaks(off_mm, sep_mm=P["sep_mm"]):
    """Peaks of the smoothed offset histogram, separation >= sep_mm, each holding >= mass_min of the column. Returns sorted peak offsets, assignment."""
    edges = np.arange(-P["range_mm"], P["range_mm"] + P["bin_mm"] / 2, P["bin_mm"]); c = (edges[:-1] + edges[1:]) / 2
    h = gaussian_filter1d(np.histogram(off_mm, bins=edges)[0].astype(float), P["sigma_mm"] / P["bin_mm"], mode="constant")
    cand = [k for k in range(len(h)) if h[k] > 0 and (k == 0 or h[k] >= h[k - 1]) and (k == len(h) - 1 or h[k] >= h[k + 1])]
    kept = []
    for k in sorted(cand, key=lambda k: -h[k]):
        if all(abs(c[k] - q) >= sep_mm for q in kept): kept.append(c[k])
    while True:
        pk = np.array(sorted(kept)) if kept else np.array([0.0]); asg = np.argmin(np.abs(off_mm[:, None] - pk[None]), 1)
        mass = np.bincount(asg, minlength=len(pk)) / max(len(off_mm), 1); keep = mass >= P["mass_min"]
        if keep.all() or keep.sum() == 0: break
        kept = list(pk[keep])
    return pk, asg


def local_sheets(xyz, sep_mm=P["sep_mm"], keep_columns=False):
    n = len(xyz); tree = cKDTree(xyz); rad = np.hypot(P["lateral_mm"], P["range_mm"]) / 1000
    nrm = np.zeros((n, 3)); nsh = np.ones(n, np.int8); idx = np.zeros(n, np.int8); gap = np.full(n, np.nan, np.float32); peaks = np.full((n, 4), np.nan, np.float32)
    cols = [None] * n if keep_columns else None
    for i in range(n):
        nb = np.array(tree.query_ball_point(xyz[i], rad)); d = xyz[nb] - xyz[i]; near = np.linalg.norm(d, axis=1) <= P["plane_mm"] / 1000
        if near.sum() < 6:
            nrm[i] = (0, 0, 1); continue
        v = np.linalg.eigh(np.cov(d[near].T))[1][:, 0]; v = v if v[2] >= 0 else -v; nrm[i] = v
        a = d @ v; lat = np.linalg.norm(d - np.outer(a, v), axis=1); m = (lat <= P["lateral_mm"] / 1000) & (np.abs(a) <= P["range_mm"] / 1000)
        off = a[m] * 1000; pk, asg = find_peaks(off, sep_mm); own = int(np.argmin(np.abs(pk)))
        nsh[i] = len(pk); idx[i] = own; peaks[i, :min(4, len(pk))] = pk[:4]
        if len(pk) > 1: gap[i] = float(np.min(np.abs(np.delete(pk, own) - pk[own])))
        if keep_columns: cols[i] = (nb[m].astype(np.int32), asg.astype(np.int8))
    return dict(normal=nrm, n_sheets=nsh, index=idx, gap=gap, peaks=peaks, columns=cols)


# ---------------------------------------------------------------- Step 2: sheet graph and patches
def edges_3mm(xyz):
    return cKDTree(xyz).query_pairs(P["edge_mm"] / 1000, output_type="ndarray")


def edge_ok(xyz, S, E, off_lim_mm):
    i, j = E[:, 0], E[:, 1]; ni, nj = S["normal"][i], S["normal"][j]
    ang = np.degrees(np.arccos(np.clip(np.abs((ni * nj).sum(1)), 0, 1))); mn = ni + np.where(((ni * nj).sum(1) < 0)[:, None], -nj, nj)
    mn /= np.maximum(np.linalg.norm(mn, axis=1, keepdims=True), 1e-12); doff = np.abs(((xyz[j] - xyz[i]) * mn).sum(1)) * 1000
    multi = (S["n_sheets"][i] > 1) & (S["n_sheets"][j] > 1); same = ~multi | (S["index"][i] == S["index"][j])
    return (ang <= P["angle_deg"]) & (doff <= off_lim_mm) & same


def patches(xyz, S, E, off_lim_mm, small_n):
    """Components of the sheet graph; patches below small_n merge (smallest first) into the neighbour with the most edges under the doubled offset limit."""
    import heapq
    n = len(xyz); ok = edge_ok(xyz, S, E, off_lim_mm); G = coo_matrix((np.ones(ok.sum()), (E[ok, 0], E[ok, 1])), shape=(n, n)).tocsr(); G = G.maximum(G.T)
    nc, lab = connected_components(G, directed=False); size = np.bincount(lab, minlength=nc).astype(int); conn = float(size.max() / n)
    ok2 = edge_ok(xyz, S, E, 2 * off_lim_mm); a, b_ = lab[E[ok2, 0]], lab[E[ok2, 1]]; cross = a != b_
    adj = [Counter() for _ in range(nc)]
    for x, y in zip(a[cross], b_[cross]):
        adj[x][y] += 1; adj[y][x] += 1
    parent = list(range(nc))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    heap = [(int(size[c]), c) for c in range(nc) if size[c] < small_n]; heapq.heapify(heap); failed = set()
    while heap:
        sz, c = heapq.heappop(heap)
        if find(c) != c or size[c] != sz or c in failed: continue
        nbr = Counter()
        for d, k in adj[c].items():
            r = find(d)
            if r != c: nbr[r] += k
        if not nbr: failed.add(c); continue
        d = nbr.most_common(1)[0][0]; parent[c] = d; size[d] += size[c]; adj[d].update(adj[c]); adj[c] = Counter()
        if size[d] < small_n: heapq.heappush(heap, (int(size[d]), d))
    roots = np.array([find(c) for c in range(nc)]); pl_ = roots[lab]; out = np.where(size[pl_] >= small_n, pl_, -1)
    uniq = {c: k for k, c in enumerate(sorted(set(out[out >= 0].tolist())))}; out = np.array([uniq.get(int(c), -1) for c in out])
    return out, dict(components=int(nc), connectivity=round(conn, 4), patches=len(uniq), unassigned=int((out < 0).sum()))


# ---------------------------------------------------------------- views, labels, projections
def projections(pl, b, views_vid):
    """Per labelling view: pixel (1536 grid) of every blade surfel and camera-space depth."""
    out = {}
    for vid in views_vid:
        view = AP.view_file(vid); pv = V2.project_view(pl["xyz"][:1], pl["sc"][:1], pl["po"], view, V2.VISP["res"])
        u, w, z = V2.project_points(pl["xyz"][b], pv); s = UL.H2 / V2.VISP["res"]
        ui = np.clip(np.round(u * s).astype(int), 0, UL.H2 - 1); wi = np.clip(np.round(w * s).astype(int), 0, UL.H2 - 1)
        inside = (z > 0) & (u >= 0) & (w >= 0) & (u < V2.VISP["res"]) & (w < V2.VISP["res"])
        out[vid] = dict(ui=ui, wi=wi, z=z, inside=inside, R=np.array(pl["po"]["views"][view]["R_cw"]), t=np.array(pl["po"]["views"][view]["t_cw"]), k=pl["views"].index(view))
    return out


def user_maps(chosen, nmap):
    raw, _ = UL.label_maps(UL.OUT, chosen); m = UL.remap(raw, nmap); return {v: np.where(m[v] > 0, m[v], 0).astype(np.int16) for v in chosen}


def leaflet_counts(pl, b, chosen, nmap, prj, VISb):
    """Front-visible blade surfels under each user leaflet mask (map v3 number > 0) of the labelling photos."""
    recs = UL.load_labels(UL.OUT); counts = []
    for vid in chosen:
        det = {m["id"]: m for m in json.load(open(UL.OUT / "detect" / f"{vid}.json", encoding="utf-8"))["masks"]}; act = {}
        for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
            if r.get("removed"): act.pop(r["removed"], None); continue
            act[r.get("mask_id") or r["id"]] = r
        p = prj[vid]; vis = VISb[:, p["k"]] & p["inside"]
        for r in act.values():
            if r["leaf"] in ("모름", None) or int(nmap.get(f"{vid}|{int(r['leaf'])}", -1)) <= 0: continue
            msk = UL.record_rle(r, recs, det)
            if msk is None: continue
            c = int((vis & msk[p["wi"], p["ui"]]).sum())
            if c > 0: counts.append(c)
    return np.array(counts)


# ---------------------------------------------------------------- Step 3 and Step 4
def patch_votes(patch, cnt_b, share=P["vote_share"]):
    lab = np.zeros(len(patch), np.int32); info = {}
    for p in np.unique(patch[patch >= 0]):
        m = patch == p; c = cnt_b[m].sum(0); tot = c.sum()
        if tot and c.max() / tot >= share and (np.sort(c)[-1] > np.sort(c)[-2]): lab[m] = int(c.argmax())
        info[int(p)] = dict(n=int(m.sum()), votes=int(tot), top=(int(c.argmax()) if tot else 0), share=(round(float(c.max() / tot), 3) if tot else None))
    return lab, info


def purity(patch, prj, umaps, VISb):
    """Per patch: user labels (> 0) at the projections of its front-visible surfels over the labelling photos."""
    per = {}
    for vid, p in prj.items():
        vis = VISb[:, p["k"]] & p["inside"]; ul = umaps[vid][p["wi"], p["ui"]]; m = vis & (ul > 0) & (patch >= 0)
        for pp, l in zip(patch[m], ul[m]): per.setdefault(int(pp), Counter())[int(l)] += 1
    rows = {pp: dict(n=sum(c.values()), majority=c.most_common(1)[0][0], purity=c.most_common(1)[0][1] / sum(c.values()), dist=dict(c)) for pp, c in per.items()}
    tot = sum(r["n"] for r in rows.values()); aw = sum(r["purity"] * r["n"] for r in rows.values()) / max(tot, 1)
    return rows, round(float(aw), 4)


def front_check(xyz_b, S, patch_lab, prj, umaps, VISb):
    """Per (multi-sheet column, labelling photo) with the column centre front-visible: near sheet = column peak nearer the camera."""
    res = dict(total=0, agree=0, upper=[0, 0], lower=[0, 0], middle=[0, 0], skipped_no_label=0, far_visible=[0, 0]); per_view = {}
    multi = np.nonzero(S["n_sheets"] > 1)[0]
    for vid, p in prj.items():
        vis = VISb[:, p["k"]] & p["inside"]; ul = umaps[vid][p["wi"], p["ui"]]; cam = -(p["R"].T @ p["t"]); a_t = a_n = 0
        for s in multi[vis[multi]]:
            nb, asg = S["columns"][s]; pk = S["peaks"][s][~np.isnan(S["peaks"][s])]
            if len(pk) < 2: continue
            pos = xyz_b[s][None] + (pk[:, None] / 1000) * S["normal"][s][None]; depth = (pos @ p["R"].T + p["t"])[:, 2]; near = int(np.argmin(depth))
            ns = nb[asg == near]; fs = nb[asg != near]
            pl_ = patch_lab[ns]; pl_ = pl_[pl_ > 0]; uu = ul[ns]; uu = uu[uu > 0]
            if not len(pl_) or not len(uu): res["skipped_no_label"] += 1; continue
            ok = Counter(pl_.tolist()).most_common(1)[0][0] == Counter(uu.tolist()).most_common(1)[0][0]
            res["total"] += 1; res["agree"] += int(ok); a_t += 1; a_n += int(ok)
            kind = "upper" if near == len(pk) - 1 else ("lower" if near == 0 else "middle"); res[kind][0] += 1; res[kind][1] += int(ok)
            res["far_visible"][0] += len(fs); res["far_visible"][1] += int(vis[fs].sum())
        per_view[vid] = dict(columns=a_t, agreement=(round(a_n / a_t, 4) if a_t else None))
    out = dict(front_agreement=(round(res["agree"] / res["total"], 4) if res["total"] else None), columns=res["total"], skipped_no_label=res["skipped_no_label"],
               upper=dict(columns=res["upper"][0], agreement=(round(res["upper"][1] / res["upper"][0], 4) if res["upper"][0] else None)),
               lower=dict(columns=res["lower"][0], agreement=(round(res["lower"][1] / res["lower"][0], 4) if res["lower"][0] else None)),
               middle=dict(columns=res["middle"][0], agreement=(round(res["middle"][1] / res["middle"][0], 4) if res["middle"][0] else None)),
               far_sheet_visible_share=(round(res["far_visible"][1] / res["far_visible"][0], 4) if res["far_visible"][0] else None), per_view=per_view)
    return out


def overlay(vid, p, patch_lab, umaps, VISb, path):
    img = Image.open(LO.DEFAULTS["poses"].parent / "undist_3024" / AP.view_file(vid)).convert("RGB").resize((UL.H2, UL.H2), Image.BILINEAR)
    a = (np.asarray(img).astype(float) * 0.35); b_ = a.copy(); vis = VISb[:, p["k"]] & p["inside"] & (patch_lab > 0)
    for L in np.unique(patch_lab[vis]):
        m = vis & (patch_lab == L); a[p["wi"][m], p["ui"][m]] = V2.PAL.get(int(L), (220, 220, 220))
    for L in np.unique(umaps[vid][umaps[vid] > 0]):
        b_[umaps[vid] == L] = b_[umaps[vid] == L] * 0.4 + np.array(V2.PAL.get(int(L), (220, 220, 220))) * 0.6
    Image.fromarray(np.hstack([a, b_]).clip(0, 255).astype(np.uint8)).resize((1536, 768)).save(path)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args(argv); out = a.out; t0 = time.time()
    for d in ("sheets", "patches", "front_check", "scores", "viz"): (out / d).mkdir(parents=True, exist_ok=True)
    pl = UL.load_plant(); b = np.nonzero(pl["blade"])[0]; xyz = pl["xyz"][b]; views = pl["views"]; H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]
    W = np.load(FIRST / "contributions.npz")["W"]; assert len(W) == len(pl["xyz"]); VISb = (W >= P["w_min"])[b]
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; nmap = json.load(open(RC.MAP_FILE, encoding="utf-8"))["map"]
    prj = projections(pl, b, chosen); umaps = user_maps(chosen, nmap); lc = leaflet_counts(pl, b, chosen, nmap, prj, VISb); small_n = int(np.percentile(lc, P["small_pct"]))
    print(f"labelled leaflets {len(lc)}, p5 surfels {small_n} ({time.time() - t0:.0f}s)", flush=True)
    # Step 1 (declared) + Step 2
    S = local_sheets(xyz, P["sep_mm"], keep_columns=True); E = edges_3mm(xyz)
    np.savez_compressed(out / "sheets" / "sheets.npz", rows=pl["rows"][b], n_sheets=S["n_sheets"], index=S["index"], gap=S["gap"], peaks=S["peaks"], normal=S["normal"].astype(np.float32))
    multi_share = float((S["n_sheets"] > 1).mean()); patch, pst = patches(xyz, S, E, P["offset_mm"], small_n)
    np.savez_compressed(out / "patches" / "patches.npz", rows=pl["rows"][b], patch=patch)
    print(f"step1/2: multi-sheet share {multi_share:.3f}, {pst} ({time.time() - t0:.0f}s)", flush=True)
    # Step 4 vote per patch
    allmaps, held2, held_map = SS.vp_allmaps(pl, H2id); assert held2 == held
    _, cnt = SS.vote_with_vis(pl, allmaps, W >= P["w_min"]); cnt_b = cnt[b]; plab, pinfo = patch_votes(patch, cnt_b)
    # Step 3
    prow, pur = purity(patch, prj, umaps, VISb); fc = front_check(xyz, S, plab, prj, umaps, VISb)
    print("front", json.dumps({k: v for k, v in fc.items() if k != "per_view"}), "purity", pur, flush=True)
    # per human leaf
    span = {}; multi_leaf = []
    for pp, r in prow.items():
        for L, c in r["dist"].items(): span.setdefault(L, {})[pp] = c
        big = [L for L, c in r["dist"].items() if c / r["n"] >= 0.20]
        if len(big) >= 2: multi_leaf.append(dict(patch=pp, n=r["n"], leaves={str(L): round(r["dist"][L] / r["n"], 3) for L in big}))
    leaf_span = {str(L): sum(1 for c in d.values() if c / sum(d.values()) >= 0.05) for L, d in sorted(span.items())}
    # held-out score (imported code) + old stack-violation rate (artefact)
    full = np.zeros(len(pl["xyz"]), np.int16); full[b] = plab; hv = AP.view_file(held); col = H2id[:, views.index(hv)]
    ren = UL.render_ids(pl, full, hv, col); rows, g = UL.score_heldout(held_map, ren); cov = float(((ren > 0) & (held_map > 0)).sum() / max((held_map > 0).sum(), 1))
    flags = {str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rows.items()}
    Pst, _ = SS.stack_pairs(xyz, pl["nrm"][b] / np.maximum(np.linalg.norm(pl["nrm"][b], axis=1, keepdims=True), 1e-12)); _, old_rate, _ = SS.violations(plab, Pst)
    print("held-out", round(g["median_iou"], 3), round(g["max_confusion"], 3), "cov", round(cov, 3), {L: (round(r["iou"], 2), round(r["confusion"], 2)) for L, r in rows.items()}, flush=True)
    passed = (fc["front_agreement"] or 0) >= PASS["front_min"] and pur >= PASS["purity_min"] and g["median_iou"] > PASS["ref_iou"] and g["max_confusion"] <= PASS["ref_conf"]
    # sensitivity (reporting only)
    sens = {}
    for sep in SENS["sep_mm"]:
        Ss = S if sep == P["sep_mm"] else local_sheets(xyz, sep)
        for mult in SENS["offset_mult"]:
            pt, st = patches(xyz, Ss, E, P["offset_mm"] * mult, small_n); _, pu = purity(pt, prj, umaps, VISb)
            sens[f"sep{sep}_x{mult}"] = dict(patches=st["patches"], unassigned=st["unassigned"], connectivity=st["connectivity"], purity=pu)
            print("sens", sep, mult, sens[f"sep{sep}_x{mult}"], flush=True)
    for vid, p in prj.items():
        overlay(vid, p, plab, umaps, VISb, out / "front_check" / f"overlay_{vid}.png")
    tag = "gate_passed" if g["gate_pass"] else "gate_failed_viewing_only"
    metrics = dict(**LABELS, params=P, t_sheet_mm=T_SHEET_MM, pass_rule=PASS, labelled_leaflets=int(len(lc)), small_patch_surfels=small_n, multi_sheet_share=round(multi_share, 4),
                   patches=pst, front_check=fc, purity_area_weighted=pur, n_patches_with_labels=len(prow), multi_leaf_patches=multi_leaf, leaf_span=leaf_span,
                   patch_votes=dict(numbered=int(sum(1 for v in pinfo.values() if v["share"] and v["share"] >= P["vote_share"])), total=len(pinfo)),
                   held_out=dict(rows={str(L): v for L, v in rows.items()}, summary=g, coverage=round(cov, 4), flags=flags),
                   reference_row=dict(median_iou=0.517, max_confusion=0.393, coverage=0.771, usable=[1, 7]), old_stack_violation_rate=old_rate,
                   passed=bool(passed), tag=tag, sensitivity=sens, seconds=round(time.time() - t0))
    json.dump(metrics, open(out / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    json.dump(dict(rows={str(L): v for L, v in rows.items()}, summary=g, coverage=cov, flags=flags), open(out / "scores" / "heldout.json", "w", encoding="utf-8"), indent=1, default=float)
    json.dump(fc, open(out / "front_check" / "front_check.json", "w", encoding="utf-8"), indent=1)
    with open(out / "patches" / "patch_table.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["patch", "surfels", "votes", "vote_top", "vote_share", "leaf", "labelled_proj", "label_majority", "purity"])
        for pp, inf in sorted(pinfo.items()):
            r = prow.get(pp, {}); w.writerow([pp, inf["n"], inf["votes"], inf["top"], inf["share"], int(plab[patch == pp][0]) if inf["n"] else 0, r.get("n", 0), r.get("majority", ""), round(r.get("purity", 0), 3) if r else ""])
    lf = np.full(len(pl["xyz"]), -1); lf[b] = np.where(plab > 0, plab, -1); pf = np.full(len(pl["xyz"]), -1); pf[b] = patch
    np.savez_compressed(out / "viz" / "surfel_clusters.npz", rows=pl["rows"], leaf=lf, patch=pf, n_sheets=np.r_[np.zeros(0)], axis_kind=np.load(V2.R5_DIR / "surfel_clusters.npz")["axis_kind"])
    print("done", "PASS" if passed else "FAIL", tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
