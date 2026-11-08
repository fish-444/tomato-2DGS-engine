"""Contested-surfel resolution, Revision R1 (user's geometry-first weights 0.25 / 0.10 / 0.35 / 0.30, continuous rachis score, geometry-only absorption of unseen
clusters). Contract: build/p1_local_diffusion_20261027_01/contested_resolve/CONTRACT.md, Revision R1. Clustering, neighbours, photo evidence and continuity are
imported unchanged from contested_resolve. Coloured surfels are never relabelled; labels never cross sheets. Human labels only score.

usage: python -m tomato3d.organs.contested_resolve_r1
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from collections import Counter

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import contested_resolve as CR
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import trace_reconcile as TR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.organs import vp_recluster as RC

OUT = CR.OUT / "geometry_first"; STATUS = "R3-1b_unaccepted"
W_DECL = dict(vp=0.25, sam=0.10, sheet=0.35, rachis=0.30); W_PHOTO = dict(vp=0.4, sam=0.3, sheet=0.2, rachis=0.1)
G = dict(geo_total=0.6, geo_margin=0.25, side_mm=15.0, side_min=20, pseudo_min=200, pseudo_score=0.3)
TAGS = ("other", "base", "diffused", "absorb_single_neighbour", "absorb_evidence", "still_contested", "absorb_geometry_only")


def pseudo_rachis(xyz, mem_L, stem_xy):
    """Segment along the PCA axis of a leaf's surfels from its base-most surfel (nearest in xy to the stem centroid) to the far end of the projection."""
    P = xyz[mem_L]; c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); ax = vt[0]
    base = P[np.argmin(np.linalg.norm(P[:, :2] - stem_xy, axis=1))]; proj = (P - base) @ ax
    if abs(proj.min()) > abs(proj.max()): ax = -ax; proj = -proj
    return np.vstack([base, base + ax * proj.max()])


def rachis_tracks(B, Gp, lab, stem_xy):
    """Per leaf: densified trace (or pseudo-rachis), plain-length geodesic from its trace-entry surfels, and whether it is a pseudo-rachis."""
    xyz = B["xyz"]; tree = cKDTree(xyz); out = {}
    for L in sorted(set(lab[lab > 0].tolist())):
        tr = [t for t in B["traces"] if t["leaf"] == L]; pseudo = not tr
        if pseudo:
            mem = np.nonzero(lab == L)[0]
            if len(mem) < G["pseudo_min"]: out[L] = dict(neutral=True, pseudo=True); continue
            dense = CC.densify(pseudo_rachis(xyz, mem, stem_xy))
        else: dense = np.vstack([CC.densify(t["pts"]) for t in tr])
        d, j = tree.query(dense); entries = np.unique(j[d <= LD.D["trace_seed_mm"] / 1000])
        geo = dijkstra(Gp, directed=False, indices=entries, min_only=True) if len(entries) else np.full(len(xyz), np.inf)
        out[L] = dict(neutral=False, pseudo=pseudo, dense=dense, geo=geo, tree=cKDTree(dense))
    return out


def rachis_score(xyz, mem, L, T, lab, reach):
    t = T.get(L)
    if t is None or t["neutral"]: return G["pseudo_score"], dict(neutral=True)
    c = xyz[mem].mean(0); rep = mem[np.argmin(np.linalg.norm(xyz[mem] - c, axis=1))]; d_mm = t["geo"][rep]
    s_dist = max(0.0, 1 - d_mm / reach.get(L, LD.D["reach_default_mm"])) if np.isfinite(d_mm) else 0.0
    dense = t["dense"]; _, j = t["tree"].query(c); p = dense[j]; tang = dense[min(j + 1, len(dense) - 1)] - dense[max(j - 1, 0)]; tang /= max(np.linalg.norm(tang), 1e-12)
    if len(mem) >= 3: _, _, vt = np.linalg.svd(xyz[mem] - c, full_matrices=False); axis = vt[0]
    else: axis = xyz[mem[-1]] - xyz[mem[0]] if len(mem) > 1 else np.array([1.0, 0, 0])
    v = c - p; s_dir = float(abs(axis @ v) / max(np.linalg.norm(axis) * np.linalg.norm(v), 1e-12))
    side_c = np.sign(np.cross(tang, v)[2]); memL = np.nonzero(lab == L)[0]; dL, jL = t["tree"].query(xyz[memL])
    step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]; near = np.abs(step[jL] - step[j]) * 1000 <= G["side_mm"]
    sideL = np.sign(np.cross(np.repeat(tang[None], near.sum(), 0), xyz[memL[near]] - dense[jL[near]])[:, 2]) if near.any() else np.zeros(0)
    s_side = 1.0 if (sideL == side_c).sum() >= G["side_min"] else 0.5
    return s_dist * (0.6 * s_dir + 0.4 * s_side), dict(geo_mm=(round(float(d_mm), 1) if np.isfinite(d_mm) else None), s_dist=round(s_dist, 3), s_dir=round(s_dir, 3), s_side=s_side, pseudo=t["pseudo"])


def decide(cands, ev, w, margin):
    if ev["unseen"]:
        gw = dict(sheet=w["sheet"] / (w["sheet"] + w["rachis"]), rachis=w["rachis"] / (w["sheet"] + w["rachis"]))
        tot = {L: gw["sheet"] * ev["sheet"][L] + gw["rachis"] * ev["rachis"][L] for L in cands}; o = sorted(tot.items(), key=lambda kv: -kv[1]); best, tb = o[0]; t2 = o[1][1] if len(o) > 1 else 0.0
        if tb >= G["geo_total"] and tb - t2 >= G["geo_margin"] and ev["sheet"][best] == 1.0: return best, "absorb_geometry_only", tot
        return None, "no_photo_view", tot
    tot = {L: sum(w[k] * ev[k][L] for k in w) for L in cands}; o = sorted(tot.items(), key=lambda kv: -kv[1]); best, tb = o[0]; t2 = o[1][1] if len(o) > 1 else 0.0
    if tb < CR.R["total_min"]: return None, "low_total", tot
    if tb - t2 < margin: return None, "tie", tot
    return best, "absorb_evidence", tot


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]
    z = np.load(CR.A2 / "labels_diffused.npz"); lab = z["label"][b].astype(int); src = z["source"][b]; white = src == 3; coloured = lab > 0
    reach = {int(L): v["reach_mm"] for L, v in json.load(open(CR.A2 / "metrics.json", encoding="utf-8"))["per_leaf"].items()}
    cid, merged = CR.clusters(xyz, S, white); nC = cid.max() + 1; NB = CR.neighbours(xyz, S, cid, lab, coloured)
    _, Gp, _ = LD.sheet_graph(xyz, S); stem_xy = pl["xyz"][B["axis"] == 1][:, :2].mean(0); T = rachis_tracks(B, Gp, lab, stem_xy)
    print(f"white {int(white.sum())} -> {nC} clusters; rachis tracks {[(L, 'pseudo' if v['pseudo'] else 'trace') for L, v in T.items()]} ({time.time() - t0:.0f}s)", flush=True)
    lower = (S["n_sheets"] > 1) & (S["index"] == 0); upper = (S["n_sheets"] > 1) & (S["index"] == S["n_sheets"] - 1)
    def sheet_of(mem): return "lower" if lower[mem].mean() > 0.5 else ("upper" if upper[mem].mean() > 0.5 else "single")
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; B["chosen"] = chosen; B["held"] = held
    from tomato3d.organs import oversegment_second as O2
    vis_views = {vid: v[0] for vid, v in O2.vote_views(B).items()}
    raw, _ = UL.label_maps(UL.OUT, chosen + [held]); nmap_v3 = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; seedm = UL.remap({v: raw[v] for v in chosen}, nmap_v3)
    vpmaps = {v: np.where(seedm[v] > 0, seedm[v], 0).astype(np.int16) for v in chosen}; zp = np.load(TR.VP_DIR / "propagated_maps.npz")
    for k in zp.files:
        if k[1:] != held and k[1:] not in vpmaps: vpmaps[k[1:]] = zp[k]
    cands = {c: sorted(NB[c]["contact"]) for c in range(nC)}
    vp_cnt, sam_cnt, seen, seen_sam = CR.photo_evidence(B, cid, lab, vis_views, vpmaps, cands); print(f"photo evidence done ({time.time() - t0:.0f}s)", flush=True)
    new = lab.copy(); tag = np.where(src == 1, 1, np.where(src == 2, 2, 0)).astype(np.int8); rows = []; EV = {}
    for c in range(nC):
        nb = NB[c]; mem = nb["members"]; cl = cands[c]
        rec = dict(cluster=c, size=len(mem), sheet=sheet_of(mem), neighbours=";".join(f"{L}:{nb['contact'][L]}" for L in cl), within6=";".join(map(str, sorted(nb["within6"]))), photos_seen=seen[c], photos_sam=seen_sam[c])
        if len(cl) == 1 and nb["within6"] == set(cl):
            new[mem] = cl[0]; tag[mem] = 3; rec.update(decision=cl[0], tag="absorb_single_neighbour", reason="rim"); rows.append(rec); continue
        if not cl:
            tag[mem] = 5; rec.update(decision=0, tag="still_contested", reason="no_neighbour"); rows.append(rec); continue
        vp_tot = sum(vp_cnt[c].values()); sam_tot = sum(sam_cnt[c][L] for L in cl)
        ev = dict(vp={L: (vp_cnt[c][L] / vp_tot if vp_tot else 0.0) for L in cl}, sam={L: (sam_cnt[c][L] / sam_tot if sam_tot else 0.0) for L in cl}, sheet={}, rachis={}, unseen=(seen[c] == 0)); det = {}
        for L in cl:
            ev["sheet"][L], d1 = CR.continuity(xyz, S, nb["pairs"][L]); ev["rachis"][L], d2 = rachis_score(xyz, mem, L, T, lab, reach); det[L] = dict(cont=d1, rach=d2)
        EV[c] = ev; best, reason, tot = decide(cl, ev, W_DECL, CR.R["margin"])
        if best is not None: new[mem] = best; tag[mem] = 6 if reason == "absorb_geometry_only" else 4; rec.update(decision=best, tag=reason, reason=reason)
        else: tag[mem] = 5; rec.update(decision=0, tag="still_contested", reason=reason)
        alt, alt_reason, _ = decide(cl, ev, W_PHOTO, CR.R["margin"]); rec.update(photo_first_decision=(alt or 0), photo_first_reason=alt_reason)
        rec.update(evidence=json.dumps({str(L): dict(vp=round(ev["vp"][L], 3), sam=round(ev["sam"][L], 3), sheet=ev["sheet"][L], rachis=round(ev["rachis"][L], 3), total=round(tot[L], 3), **det[L]) for L in cl}, ensure_ascii=False, default=float))
        rows.append(rec)
    # evaluation
    prj = SR.projections(pl, b, chosen); hit = Counter(); tot_ = Counter()
    for vid, p in prj.items():
        ul = vpmaps[vid][p["wi"], p["ui"]]
        for t in (3, 4, 6):
            m = B["VISb"][:, p["k"]] & p["inside"] & (tag == t) & (ul > 0); hit[t] += int((new[m] == ul[m]).sum()); tot_[t] += int(m.sum())
    pur = {TAGS[t]: dict(purity=(round(hit[t] / tot_[t], 4) if tot_[t] else None), labelled_projections=tot_[t]) for t in (3, 4, 6)}
    pooled = (round((hit[3] + hit[4]) / (tot_[3] + tot_[4]), 4) if (tot_[3] + tot_[4]) else None)
    truth = UL.remap({held: raw[held]}, nmap_v3)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16); H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def score(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; ren = UL.render_ids(pl, f, hv, col); rws, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        return dict(rows={str(k): v for k, v in rws.items()}, summary=g, coverage=round(cov, 4), flags={str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rws.items()})
    sb, sa = score(lab), score(new); gb, ga = sb["summary"], sa["summary"]
    passed = (pooled or 0) >= 0.85 and ga["median_iou"] >= gb["median_iou"] and ga["max_confusion"] <= gb["max_confusion"]
    sens = {}; agree = dis = 0
    for wname, ww in (("geometry_first_declared", W_DECL), ("photo_first", W_PHOTO)):
        for mg in (0.10, 0.15, 0.25):
            lb2 = new.copy(); nabs = Counter()
            for c, ev in EV.items():
                mem = NB[c]["members"]; best, reason, _ = decide(cands[c], ev, ww, mg); lb2[mem] = best if best is not None else lab[mem]; nabs[reason] += 1
            g2 = score(lb2)["summary"]; sens[f"{wname}_margin{mg}"] = dict(decisions=dict(nabs), median_iou=round(g2["median_iou"], 4), max_confusion=round(g2["max_confusion"], 4))
    for r in rows:
        if "photo_first_decision" not in r: continue
        if r["decision"] or r["photo_first_decision"]:
            if r["decision"] == r["photo_first_decision"]: agree += 1
            else: dis += 1
    by_tag = Counter(TAGS[t] for t in tag.tolist()); reasons = Counter(); reasons_s = Counter()
    for r in rows:
        if r["tag"] == "still_contested": reasons[r["reason"]] += 1; reasons_s[r["reason"]] += r["size"]
    low = [r for r in rows if r["sheet"] == "lower"]; wb = int(white.sum())
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="contested_resolve_R1_geometry_first", base="diffusion attempt 2 (V-P base)", weights=W_DECL, geometry_rules=G, white_before=wb, clusters=int(nC),
               rachis_tracks={str(L): ("neutral_<200" if v["neutral"] else ("pseudo_rachis" if v["pseudo"] else "trace")) for L, v in T.items()},
               clusters_by_tag=dict(Counter(r["tag"] for r in rows)), surfels_by_tag=dict(by_tag), white_after=int(by_tag["still_contested"]),
               share_absorbed={k: round(by_tag[k] / wb, 4) for k in ("absorb_single_neighbour", "absorb_evidence", "absorb_geometry_only", "still_contested")},
               still_white_by_reason=dict(clusters=dict(reasons), surfels=dict(reasons_s)), lower_sheet=dict(clusters=len(low), surfels=int(sum(r["size"] for r in low)), by_tag=dict(Counter(r["tag"] for r in low))),
               absorption_purity=dict(pooled_single_plus_evidence=pooled, **pur, note="absorb_geometry_only never pooled; its photo purity is usually unmeasurable"),
               held_out=dict(view=held, base=sb, base_plus_absorption=sa), reference=dict(A_B_C_Dprime=[0.517, 0.393], V_P=[0.546, 0.438], photo_first_run_section8=[0.5474, 0.4253]),
               pass_rule="pooled purity (single + evidence) >= 0.85 AND median IoU >= base AND max confusion <= base", passed=bool(passed),
               weight_set_agreement=dict(decided_clusters=agree + dis, agree=agree, disagree=dis, disagreement_share=(round(dis / (agree + dis), 4) if agree + dis else None)),
               absorbed_light_by_source="not available (no per-surfel absorbed-light file in build/)", sensitivity=sens, tag=("gate_passed" if ga["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    with open(OUT / "clusters.csv", "w", encoding="utf-8", newline="") as fh:
        keys = ["cluster", "size", "sheet", "neighbours", "within6", "photos_seen", "photos_sam", "decision", "tag", "reason", "photo_first_decision", "photo_first_reason", "evidence"]
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    fl = np.zeros(len(pl["xyz"]), np.int16); fl[b] = new; ft = np.zeros(len(pl["xyz"]), np.int8); ft[b] = tag; fc = -np.ones(len(pl["xyz"]), int); fc[b] = cid
    np.savez_compressed(OUT / "labels.npz", rows=pl["rows"], label=fl, source=ft, source_names=np.array(TAGS), cluster=fc)
    CR.images(B, lab, new, np.where(tag == 6, 4, tag), OUT / "images")
    vdir = LD.OUT / "viewer"; old = vdir / "diffusion_viewer_resolved.html"; keep = vdir / "diffusion_viewer_resolved_photofirst.html"
    if old.exists() and not keep.exists(): shutil.copy2(old, keep)
    src_v = np.where(tag == 5, 3, np.where(np.isin(tag, (3, 4, 6)), 2, np.where(tag >= 1, 1, 0))).astype(np.int8)
    info = {int(L): dict(pending=False, base_surfels=int((lab == L).sum()), diffused=int(((new == L) & (src_v == 2)).sum()), reach_mm=reach.get(int(L), "—")) for L in np.unique(new[new > 0])}
    note = (f"{STATUS} · conditional_on_unvalidated_scale · {res['tag']} — 경합 해소 R1 (기하 우선 0.25/0.10/0.35/0.30): 연한 색 = 흡수 (단일 이웃 {by_tag['absorb_single_neighbour']} + 증거 {by_tag['absorb_evidence']} + 기하만 {by_tag['absorb_geometry_only']}), "
            f"흰색 = 아직 경합 {by_tag['still_contested']} · 보류 IoU {gb['median_iou']:.3f}→{ga['median_iou']:.3f}, 혼동 {gb['max_confusion']:.3f}→{ga['max_confusion']:.3f}, 흡수 순도 {pooled} · 읽기 전용")
    LD.write_viewer(B, new.astype(np.int16), src_v, info, old, note)
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out", "sensitivity", "geometry_rules")}, indent=1, default=float, ensure_ascii=False))
    print("held-out base", gb, "after", ga); print("rows after", {k: (round(v["iou"], 3), round(v["confusion"], 3)) for k, v in sa["rows"].items()}); print("sens", json.dumps(sens))


if __name__ == "__main__":
    main()
