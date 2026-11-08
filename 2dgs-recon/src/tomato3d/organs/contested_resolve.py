"""Resolve the contested (white) surfels of diffusion attempt 2: cluster them on the sheet graph, absorb single-neighbour clusters (rim rule), and decide
multi-neighbour clusters by evidence voting (V-P photo masks 0.4, SAM co-membership 0.3, sheet continuity 0.2, rachis geometry 0.1).
Contract: build/p1_local_diffusion_20261027_01/contested_resolve/CONTRACT.md. Coloured surfels are never relabelled; labels never cross sheets. Human labels only score.

usage: python -m tomato3d.organs.contested_resolve
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import oversegment as OS
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import trace_reconcile as TR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.organs import vp_recluster as RC

OUT = LD.OUT / "contested_resolve"; A2 = LD.OUT / "second_attempt"
STATUS = "R3-1b_unaccepted"
W = dict(vp=0.4, sam=0.3, sheet=0.2, rachis=0.1); R = dict(min_px=30, inside=0.70, total_min=0.5, margin=0.15, unseen_cap=0.3, contact_mm=3.0, exclusive_mm=6.0,
                                                              cont_gap=1.5, cont_crease=35.0, broken_gap=2.5, axis_deg=60.0, small=2)
SENS = dict(margins=(0.10, 0.15, 0.25), weights={"vp0.4_sam0.3": W, "vp0.5_sam0.2": dict(vp=0.5, sam=0.2, sheet=0.2, rachis=0.1)})
TAGS = ("other", "base", "diffused", "absorb_single_neighbour", "absorb_evidence", "still_contested")


def sheet_pair_ok(S, i, j):
    multi = (S["n_sheets"][i] > 1) & (S["n_sheets"][j] > 1); return ~multi | (S["index"][i] == S["index"][j])


def clusters(xyz, S, white):
    """Step A: components of the white surfels on the sheet graph; size <= 2 join the nearest cluster on the same sheet within 3 mm."""
    E = SR.edges_3mm(xyz); E = E[SR.edge_ok(xyz, S, E, LD.D["offset_mm"])]; m = white[E[:, 0]] & white[E[:, 1]]; E = E[m]
    wi = np.nonzero(white)[0]; pos = -np.ones(len(xyz), int); pos[wi] = np.arange(len(wi)); n = len(wi)
    G = coo_matrix((np.ones(len(E)), (pos[E[:, 0]], pos[E[:, 1]])), shape=(n, n)).tocsr(); _, lab = connected_components(G, directed=False)
    cid = -np.ones(len(xyz), int); cid[wi] = lab; sizes = Counter(lab.tolist()); big = [c for c, s in sizes.items() if s > R["small"]]
    merged = 0
    if big:
        bigm = np.isin(cid, big); tree = cKDTree(xyz[bigm]); bidx = np.nonzero(bigm)[0]
        for c in [c for c, s in sizes.items() if s <= R["small"]]:
            for s_ in np.nonzero(cid == c)[0]:
                for d, j in zip(*tree.query(xyz[s_], k=8, distance_upper_bound=R["contact_mm"] / 1000)):
                    if np.isfinite(d) and sheet_pair_ok(S, s_, bidx[j]): cid[s_] = cid[bidx[j]]; merged += 1; break
    ids = {c: k for k, c in enumerate(sorted(set(cid[cid >= 0].tolist())))}; cid = np.array([ids.get(c, -1) if c >= 0 else -1 for c in cid]); return cid, merged


def neighbours(xyz, S, cid, lab, coloured):
    """Per cluster: contact pairs with coloured leaves within 3 mm on the same sheet, and leaves present within 6 mm."""
    tree = cKDTree(xyz[coloured]); cidx = np.nonzero(coloured)[0]; out = {}
    for c in range(cid.max() + 1):
        mem = np.nonzero(cid == c)[0]; pairs = defaultdict(list); within6 = set(); touch = defaultdict(set)
        for s_ in mem:
            for d, j in zip(*tree.query(xyz[s_], k=24, distance_upper_bound=R["exclusive_mm"] / 1000)):
                if not np.isfinite(d): break
                t = cidx[j]
                if not sheet_pair_ok(S, s_, t): continue
                L = int(lab[t]); within6.add(L)
                if d <= R["contact_mm"] / 1000: pairs[L].append((s_, t, d)); touch[L].add(s_)
        out[c] = dict(members=mem, pairs=dict(pairs), contact={L: len(v) for L, v in touch.items()}, within6=within6)
    return out


def continuity(xyz, S, pairs):
    i = np.array([p[0] for p in pairs]); j = np.array([p[1] for p in pairs]); gap = np.median([p[2] for p in pairs]) * 1000
    ni, nj = S["normal"][i], S["normal"][j]; crease = np.degrees(np.arccos(np.clip(np.abs((ni * nj).sum(1)), 0, 1))); mn = ni + np.where(((ni * nj).sum(1) < 0)[:, None], -nj, nj)
    mn /= np.maximum(np.linalg.norm(mn, axis=1, keepdims=True), 1e-12); off = np.abs(((xyz[j] - xyz[i]) * mn).sum(1)) * 1000; cr, of = float(np.median(crease)), float(np.median(off))
    if gap <= R["cont_gap"] and cr < R["cont_crease"] and of <= LD.T_SHEET_MM: sc = 1.0
    elif cr >= R["cont_crease"] or of > 2 * LD.T_SHEET_MM or gap > R["broken_gap"]: sc = 0.0
    else: sc = 0.5
    return sc, dict(gap=round(float(gap), 2), crease=round(cr, 1), offset=round(of, 2))


def rachis(xyz_c, traces, L, reach):
    tr = [t for t in traces if t["leaf"] == L]
    if not tr: return 0.5, None
    dense = np.vstack([CC.densify(t["pts"]) for t in tr]); c = xyz_c.mean(0); d, j = cKDTree(dense).query(c); d_mm = d * 1000
    if len(xyz_c) >= 3:
        u, s_, vt = np.linalg.svd(xyz_c - c, full_matrices=False); axis = vt[0]
    else: axis = (xyz_c[-1] - xyz_c[0]) if len(xyz_c) > 1 else np.array([1.0, 0, 0])
    v = c - dense[j]; ang = np.degrees(np.arccos(np.clip(abs(axis @ v) / max(np.linalg.norm(axis) * np.linalg.norm(v), 1e-12), 0, 1)))
    ok = d_mm <= reach.get(L, LD.D["reach_default_mm"]) and ang <= R["axis_deg"]; return (1.0 if ok else 0.0), dict(dist_mm=round(float(d_mm), 1), axis_deg=round(float(ang), 1))


def photo_evidence(B, cid, lab, vis_views, vpmaps, cands):
    """Per cluster: V-P pixel-view counts per leaf, SAM same votes per leaf, and the number of photos where it is front-visible."""
    pl, b, xyz = B["pl"], B["b"], B["xyz"]; nC = cid.max() + 1; vp_cnt = {c: Counter() for c in range(nC)}; sam_cnt = {c: Counter() for c in range(nC)}; seen = Counter(); seen_sam = Counter()
    for vid, vis in vis_views.items():
        view = f"20261021_{vid}_iOS.jpg"; pv = V2.project_view(pl["xyz"][:1], pl["sc"][:1], pl["po"], view, V2.VISP["res"]); u, w, z = V2.project_points(xyz, pv); s = UL.H2 / V2.VISP["res"]
        ui = np.round(u * s).astype(int); wi = np.round(w * s).astype(int); ok = vis & (z > 0) & (ui >= 0) & (wi >= 0) & (ui < UL.H2) & (wi < UL.H2)
        lin = wi * UL.H2 + ui; vpm = vpmaps.get(vid); masks = None
        # per-leaf visible pixel sets (for SAM) are needed only for candidates; build lazily
        leaf_px = {}
        for c in range(nC):
            m = ok & (cid == c); px = np.unique(lin[m])
            if len(px) < R["min_px"]: continue
            seen[c] += 1
            if vpm is not None:
                v_ = vpm.ravel()[px]; v_ = v_[v_ > 0]; vp_cnt[c].update(v_.tolist())
            if masks is None: masks, _ = OS.view_masks(vid, B["chosen"])
            if not masks: continue
            inside = np.array([mk.ravel()[px].mean() for mk in masks]); hit = np.nonzero(inside >= R["inside"])[0]
            if not len(hit): continue
            seen_sam[c] += 1
            for L in cands[c]:
                if L not in leaf_px:
                    ml = ok & (lab == L); leaf_px[L] = np.unique(lin[ml])
                if len(leaf_px[L]) < R["min_px"]: continue
                if any(masks[h].ravel()[leaf_px[L]].sum() >= R["min_px"] for h in hit): sam_cnt[c][L] += 1
    return vp_cnt, sam_cnt, seen, seen_sam


def decide(cands, ev, weights, margin):
    tot = {L: sum(weights[k] * ev[k][L] for k in weights) for L in cands}
    if ev["unseen"]: tot = {L: min(v, R["unseen_cap"]) for L, v in tot.items()}
    order = sorted(tot.items(), key=lambda kv: -kv[1]); best, tb = order[0]; t2 = order[1][1] if len(order) > 1 else 0.0
    if ev["unseen"]: return None, "no_photo_view", tot
    if tb < R["total_min"]: return None, "low_total", tot
    if tb - t2 < margin: return None, "tie", tot
    return best, "absorb_evidence", tot


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]
    z = np.load(A2 / "labels_diffused.npz"); lab = z["label"][b].astype(int); src = z["source"][b]; white = src == 3; coloured = lab > 0
    m2 = json.load(open(A2 / "metrics.json", encoding="utf-8")); reach = {int(L): v["reach_mm"] for L, v in m2["per_leaf"].items()}
    cid, merged = clusters(xyz, S, white); nC = cid.max() + 1; print(f"white {int(white.sum())} -> {nC} clusters (small merged {merged}) ({time.time() - t0:.0f}s)", flush=True)
    NB = neighbours(xyz, S, cid, lab, coloured)
    lower = (S["n_sheets"] > 1) & (S["index"] == 0); upper = (S["n_sheets"] > 1) & (S["index"] == S["n_sheets"] - 1)
    def sheet_of(mem): return "lower" if lower[mem].mean() > 0.5 else ("upper" if upper[mem].mean() > 0.5 else "single")
    # photos (64 QC-accepted minus held-out), visibility, V-P maps
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; B["chosen"] = chosen; B["held"] = held
    from tomato3d.organs import oversegment_second as O2
    vis_views = {vid: v[0] for vid, v in O2.vote_views(B).items()}
    raw, _ = UL.label_maps(UL.OUT, chosen + [held]); nmap_v3 = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; seedm = UL.remap({v: raw[v] for v in chosen}, nmap_v3)
    vpmaps = {v: np.where(seedm[v] > 0, seedm[v], 0).astype(np.int16) for v in chosen}; zp = np.load(TR.VP_DIR / "propagated_maps.npz")
    for k in zp.files:
        vid = k[1:]
        if vid != held and vid not in vpmaps: vpmaps[vid] = zp[k]
    vpmaps.pop(held, None); print(f"photos {len(vis_views)}, V-P maps {len([v for v in vis_views if v in vpmaps])} ({time.time() - t0:.0f}s)", flush=True)
    cands = {c: sorted(NB[c]["contact"]) for c in range(nC)}
    vp_cnt, sam_cnt, seen, seen_sam = photo_evidence(B, cid, lab, vis_views, vpmaps, cands); print(f"photo evidence done ({time.time() - t0:.0f}s)", flush=True)
    # decisions
    new = lab.copy(); tag = np.where(src == 1, 1, np.where(src == 2, 2, 0)).astype(np.int8); rows = []; EV = {}
    for c in range(nC):
        nb = NB[c]; mem = nb["members"]; cl = cands[c]; rec = dict(cluster=c, size=len(mem), sheet=sheet_of(mem), neighbours=";".join(f"{L}:{nb['contact'][L]}" for L in cl), within6=";".join(map(str, sorted(nb["within6"]))),
                                                                   photos_seen=seen[c], photos_sam=seen_sam[c])
        if len(cl) == 1 and nb["within6"] == set(cl):
            L = cl[0]; new[mem] = L; tag[mem] = 3; rec.update(decision=L, tag="absorb_single_neighbour", reason="rim"); rows.append(rec); continue
        if not cl:
            tag[mem] = 5; rec.update(decision=0, tag="still_contested", reason="no_neighbour"); rows.append(rec); continue
        vp_tot = sum(vp_cnt[c].values()); sam_tot = sum(sam_cnt[c][L] for L in cl)
        ev = dict(vp={L: (vp_cnt[c][L] / vp_tot if vp_tot else 0.0) for L in cl}, sam={L: (sam_cnt[c][L] / sam_tot if sam_tot else 0.0) for L in cl}, sheet={}, rachis={}, unseen=(seen[c] == 0))
        det = {}
        for L in cl:
            ev["sheet"][L], d1 = continuity(xyz, S, nb["pairs"][L]); ev["rachis"][L], d2 = rachis(xyz[mem], B["traces"], L, reach); det[L] = dict(cont=d1, rach=d2)
        EV[c] = ev; best, reason, tot = decide(cl, ev, W, R["margin"])
        if len(cl) == 1 and best is None and reason != "no_photo_view" and seen[c] > 0:
            pass
        if best is not None: new[mem] = best; tag[mem] = 4; rec.update(decision=best, tag="absorb_evidence", reason=reason)
        else: tag[mem] = 5; rec.update(decision=0, tag="still_contested", reason=reason)
        rec.update(evidence=json.dumps({str(L): dict(vp=round(ev["vp"][L], 3), sam=round(ev["sam"][L], 3), sheet=ev["sheet"][L], rachis=ev["rachis"][L], total=round(tot[L], 3), **det[L]) for L in cl}, ensure_ascii=False, default=float)); rows.append(rec)
    # single-neighbour clusters that failed the 6 mm exclusivity go through voting too (their only candidate competes against nothing -> margin trivially holds; they are decided by total >= 0.5)
    # evaluation: training purity, held-out
    prj = SR.projections(pl, b, chosen); hit = {3: 0, 4: 0}; tot_ = {3: 0, 4: 0}
    for vid, p in prj.items():
        ul = vpmaps[vid][p["wi"], p["ui"]]
        for t in (3, 4):
            m = B["VISb"][:, p["k"]] & p["inside"] & (tag == t) & (ul > 0); hit[t] += int((new[m] == ul[m]).sum()); tot_[t] += int(m.sum())
    pur = {("absorb_single_neighbour" if t == 3 else "absorb_evidence"): dict(purity=(round(hit[t] / tot_[t], 4) if tot_[t] else None), labelled_projections=tot_[t]) for t in (3, 4)}
    pooled = (round((hit[3] + hit[4]) / (tot_[3] + tot_[4]), 4) if (tot_[3] + tot_[4]) else None)
    truth = UL.remap({held: raw[held]}, nmap_v3)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16); H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def score(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; ren = UL.render_ids(pl, f, hv, col); rws, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        return dict(rows={str(k): v for k, v in rws.items()}, summary=g, coverage=round(cov, 4), flags={str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rws.items()})
    sb, sa = score(lab), score(new); gb, ga = sb["summary"], sa["summary"]
    passed = (pooled or 0) >= 0.85 and ga["median_iou"] >= gb["median_iou"] and ga["max_confusion"] <= gb["max_confusion"]
    # sensitivity (report only): margins and the alternative weights, re-deciding only the evidence clusters
    sens = {}
    for wname, ww in SENS["weights"].items():
        for mg in SENS["margins"]:
            lb2 = new.copy(); n_abs = 0
            for c, ev in EV.items():
                mem = NB[c]["members"]; best, reason, _ = decide(cands[c], ev, ww, mg); lb2[mem] = best if best is not None else 0
                if best is None: lb2[mem] = lab[mem]
                n_abs += int(best is not None)
            g2 = score(lb2)["summary"]; sens[f"{wname}_margin{mg}"] = dict(evidence_clusters_absorbed=n_abs, median_iou=round(g2["median_iou"], 4), max_confusion=round(g2["max_confusion"], 4))
    by_tag = Counter(TAGS[t] for t in tag.tolist()); reasons = Counter(r["reason"] for r in rows if r["tag"] == "still_contested"); reasons_surf = Counter()
    for r in rows:
        if r["tag"] == "still_contested": reasons_surf[r["reason"]] += r["size"]
    low = [r for r in rows if r["sheet"] == "lower"]
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="contested_resolve", base="diffusion attempt 2 (V-P base)", weights=W, rules=R, white_before=int(white.sum()), clusters=nC, small_merged=merged,
               clusters_by_tag=dict(Counter(r["tag"] for r in rows)), surfels_by_tag=dict(by_tag), white_after=int(by_tag["still_contested"]),
               share_absorbed=dict(single_neighbour=round(by_tag["absorb_single_neighbour"] / white.sum(), 4), evidence=round(by_tag["absorb_evidence"] / white.sum(), 4), still_white=round(by_tag["still_contested"] / white.sum(), 4)),
               still_white_by_reason=dict(clusters=dict(reasons), surfels=dict(reasons_surf)), photos=len(vis_views), vp_map_photos=len([v for v in vis_views if v in vpmaps]),
               lower_sheet=dict(clusters=len(low), surfels=int(sum(r["size"] for r in low)), by_tag=dict(Counter(r["tag"] for r in low)), by_reason=dict(Counter(r["reason"] for r in low))),
               absorption_purity=dict(pooled=pooled, **pur), held_out=dict(view=held, base=sb, base_plus_absorption=sa), reference=dict(A_B_C_Dprime=[0.517, 0.393], V_P=[0.546, 0.438]),
               pass_rule="pooled purity >= 0.85 AND median IoU >= base AND max confusion <= base", passed=bool(passed), absorbed_light_by_source="not available (no per-surfel absorbed-light file in build/)",
               sensitivity=sens, tag=("gate_passed" if ga["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    with open(OUT / "clusters.csv", "w", encoding="utf-8", newline="") as fh:
        keys = ["cluster", "size", "sheet", "neighbours", "within6", "photos_seen", "photos_sam", "decision", "tag", "reason", "evidence"]; w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    full_lab = np.zeros(len(pl["xyz"]), np.int16); full_lab[b] = new; full_tag = np.zeros(len(pl["xyz"]), np.int8); full_tag[b] = tag; full_cid = -np.ones(len(pl["xyz"]), int); full_cid[b] = cid
    np.savez_compressed(OUT / "labels.npz", rows=pl["rows"], label=full_lab, source=full_tag, source_names=np.array(TAGS), cluster=full_cid)
    images(B, lab, new, tag, OUT / "images"); write_viewer(B, lab, new, tag, res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out", "sensitivity", "rules")}, indent=1, default=float, ensure_ascii=False))
    print("held-out base", gb, "after", ga); print("rows after", {k: (round(v["iou"], 3), round(v["confusion"], 3)) for k, v in sa["rows"].items()}); print("sens", json.dumps(sens))


def images(B, lab, new, tag, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xyz = B["xyz"] * 1000; out.mkdir(exist_ok=True)
    def col(lb): c = np.full((len(lb), 3), 0.5); [c.__setitem__(lb == L, np.array(V2.PAL.get(int(L), (200, 200, 200))) / 255) for L in np.unique(lb[lb > 0])]; return c
    t5 = next(t for t in B["traces"] if t["trace_leaf"] == 5)["pts"] * 1000; t4 = next(t for t in B["traces"] if t["trace_leaf"] == 4)["pts"] * 1000; c6 = xyz[lab == 2].mean(0)
    for name, c, r in [("leaf5", t5.mean(0), 45), ("leaf4_leaf6", (t4.mean(0) + c6) / 2, 50)]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6)); m = (np.abs(xyz[:, 0] - c[0]) < r) & (np.abs(xyz[:, 1] - c[1]) < r); o = np.argsort(xyz[m, 2])
        for a, (lb, tg, ttl) in zip(ax, ((lab, None, "before (attempt 2: white = contested)"), (new, tag, "after (absorbed = leaf colour, white edge; still white = contested)"))):
            cc = col(lb[m]); wmask = (tag[m] == 5) if tg is not None else (tag[m] >= 3); cc[wmask] = 1.0
            a.scatter(xyz[m, 0][o], xyz[m, 1][o], c=cc[o], s=3, linewidths=0); a.set_facecolor("#202020"); a.set_aspect("equal"); a.set_title(ttl, fontsize=10)
            if tg is not None:
                ab = m & (tag >= 3) & (tag <= 4); a.scatter(xyz[ab, 0], xyz[ab, 1], c=col(new[ab]), s=7, edgecolors="white", linewidths=0.4)
            for t in B["traces"]: a.plot(t["pts"][:, 0] * 1000, t["pts"][:, 1] * 1000, color="#ffd400", lw=1.5); a.text(*(t["pts"][-1][:2] * 1000), f"{t['trace_leaf']}→{t['leaf']}", color="#ffd400", fontsize=8)
            a.set_xlim(c[0] - r, c[0] + r); a.set_ylim(c[1] - r, c[1] + r)
        fig.suptitle(f"{STATUS} · contested resolve · top view, {name} · gate_failed_viewing_only unless stated"); fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=110); plt.close(fig)


def write_viewer(B, lab, new, tag, res):
    """Reuse the diffusion viewer: colours = final labels; 'diffused' channel = absorbed surfels (lighter), 'contested' channel = still white."""
    src = np.where(tag == 5, 3, np.where((tag == 3) | (tag == 4), 2, np.where(tag >= 1, 1, 0))).astype(np.int8)
    info = {int(L): dict(pending=False, base_surfels=int(((lab == L)).sum()), diffused=int(((new == L) & (src == 2)).sum()), reach_mm="—") for L in np.unique(new[new > 0])}
    g = res["held_out"]["base_plus_absorption"]["summary"]; gb = res["held_out"]["base"]["summary"]
    note = (f"{STATUS} · conditional_on_unvalidated_scale · {res['tag']} — 경합(흰색) 해소: 연한 색 = 흡수된 점 (단일 이웃 {res['surfels_by_tag'].get('absorb_single_neighbour', 0)} + 증거 {res['surfels_by_tag'].get('absorb_evidence', 0)}), "
            f"흰색 = 아직 경합 {res['white_after']} · 보류 IoU {gb['median_iou']:.3f}→{g['median_iou']:.3f}, 혼동 {gb['max_confusion']:.3f}→{g['max_confusion']:.3f}, 흡수 순도 {res['absorption_purity']['pooled']} · 읽기 전용")
    LD.write_viewer(B, new.astype(np.int16), src, info, LD.OUT / "viewer" / "diffusion_viewer_resolved.html", note)


if __name__ == "__main__":
    main()
