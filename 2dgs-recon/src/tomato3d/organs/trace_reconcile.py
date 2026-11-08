"""Reconcile the two human records — rachis traces (3-D) and leaf labels (2-D clicks per photo) — to fix numbering swaps, split merge
clusters along their rachises and fill along a trace.

Contract: build/p1_trace_label_reconcile_20261026_01/CONTRACT.md. Labels and traces are read only; the per-photo renumbering is a derived
layer applied at vote time (`numbering: trace_reconciled`). The V-P propagated masks are reused (no new SAM run).

usage: python -m tomato3d.organs.trace_reconcile
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import compound_cluster as CC
from tomato3d.organs import light_organs as LO
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.organs import vp_recluster as RC

ROOT = LO.ROOT
OUT = ROOT / "build/p1_trace_label_reconcile_20261026_01"
UL_OUT = UL.OUT; MAP_FILE = UL_OUT / "labels" / "number_map_v3_charuco.json"; VP_DIR = UL_OUT / "map_v3_charuco" / "user_labelled_propagated"
TRACES = V2.OUT / "traces_export" / "rachis_traces"
STATUS = "R3-1b_unaccepted"; NUMBERING = "trace_reconciled"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, numbering=NUMBERING)
# declared (contract sections 1-5)
R_T_MM = 4.0; D_MM = 10.0; L_MAX_MM = 25.0; STEP_MM = 0.5; ID_BONUS = 0.001
FLAG = dict(split_label_share=0.10, split_labels=3, split_unassigned=0.30, contain_share=0.50)
ANCHOR = RC.ANCHOR


# ---------------------------------------------------------------- geometry of traces
def trace_near(xyz, traces, r_mm):
    """{trace key: (distance to polyline mm per surfel, bool within r)}"""
    out = {}
    for t in traces:
        pts = CC.densify(np.asarray(t["pts"], float), STEP_MM / 1000); d, _ = cKDTree(pts).query(xyz); out[t["key"]] = (d * 1000, d * 1000 <= r_mm)
    return out


def matrix_T(lab, blade, traces, near):
    rows = []
    for t in traces:
        m = near[t["key"]][1] & blade; l = lab[m]; c = Counter(l.tolist()); n = int(m.sum())
        rows.append(dict(trace=t["key"], leaf=t["leaf"], n=n, counts={int(k): int(v) for k, v in c.most_common()}, shares={int(k): round(v / max(n, 1), 3) for k, v in c.most_common()}))
    return rows


def flags_T(rows):
    dom = {}
    for r in rows:
        lab_only = {k: v for k, v in r["counts"].items() if k > 0}
        if lab_only: dom.setdefault(r["leaf"], Counter()).update({max(lab_only, key=lab_only.get): 1})
    dom = {k: c.most_common(1)[0][0] for k, c in dom.items()}
    swaps = sorted({tuple(sorted((k, j))) for k, j in dom.items() if j != k and dom.get(j) == k})
    cycles = []
    for k in dom:                                         # dominant-label map as a function on leaf numbers
        seen, cur = [], k
        while cur in dom and cur not in seen: seen.append(cur); cur = dom[cur]
        if cur == k and len(seen) >= 3 and sorted(seen) not in [sorted(c) for c in cycles]: cycles.append(seen)
    merges = {}
    for k, j in dom.items(): merges.setdefault(j, []).append(k)
    merges = {j: ks for j, ks in merges.items() if len(set(ks)) >= 2}
    splits = []
    for r in rows:
        big = [k for k, s in r["shares"].items() if k > 0 and s >= FLAG["split_label_share"]]
        if len(big) >= FLAG["split_labels"] or r["shares"].get(0, 0) >= FLAG["split_unassigned"]: splits.append(dict(trace=r["trace"], leaf=r["leaf"], labels=big, unassigned=r["shares"].get(0, 0)))
    return dict(dominant=dom, swaps=swaps, cycles=cycles, merges=merges, splits=splits)


# ---------------------------------------------------------------- per-photo click table and renumbering
def active_records(vid):
    recs = UL.load_labels(UL_OUT); act = {}
    for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
        if r.get("removed"): act.pop(r["removed"], None); continue
        act[r.get("mask_id") or r["id"]] = r
    return act, recs


def click_table(pl, H2id, vid, traces):
    import cv2
    view = AP.view_file(vid); pv = V2.project_view(pl["xyz"][:1], pl["sc"][:1], pl["po"], view, V2.VISP["res"])
    pvf = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, V2.VISP["res"]); sc = UL.H2 / V2.VISP["res"]
    ppm = UL.plant_px_per_mm(pl, view, H2id[:, pl["views"].index(view)])[0]; dpx = D_MM * ppm
    dts = {}
    for t in traces:
        u, w, z, vis = V2.trace_visibility(CC.densify(np.asarray(t["pts"], float), STEP_MM / 1000), pvf)
        if not vis.any(): continue
        ras = np.ones((UL.H2, UL.H2), np.uint8); ui = np.clip(np.round(u[vis] * sc).astype(int), 0, UL.H2 - 1); wi = np.clip(np.round(w[vis] * sc).astype(int), 0, UL.H2 - 1); ras[wi, ui] = 0
        dts.setdefault(t["leaf"], []).append(cv2.distanceTransform(ras, cv2.DIST_L2, 5))
    dts = {k: np.minimum.reduce(v) for k, v in dts.items()}
    det = {m["id"]: m for m in json.load(open(UL_OUT / "detect" / f"{vid}.json", encoding="utf-8"))["masks"]}; act, recs = active_records(vid); rows = []
    for key, r in act.items():
        if r["leaf"] in ("모름", None): continue
        m = UL.record_rle(r, recs, det)
        if m is None or not m.any(): continue
        d = {k: float(dt[m].min()) for k, dt in dts.items()}; k_best = min(d, key=d.get) if d else None
        rows.append(dict(key=key, local=int(r["leaf"]), trace=(k_best if k_best is not None and d[k_best] <= dpx else None), d_px=(round(d[k_best], 1) if k_best is not None else None)))
    return rows, dict(d_px=round(dpx, 1), visible_traces=sorted(dts))


def renumber(rows):
    """Permutation of N = local numbers ∪ trace numbers maximising click agreement (identity bonus)."""
    locs = {r["local"] for r in rows}; trs = {r["trace"] for r in rows if r["trace"] is not None}; N = sorted(locs | trs); ix = {n: i for i, n in enumerate(N)}
    M = np.zeros((len(N), len(N)))
    for r in rows:
        if r["trace"] is not None: M[ix[r["local"]], ix[r["trace"]]] += 1
    M += ID_BONUS * np.eye(len(N)); ri, ci = linear_sum_assignment(-M)
    perm = {int(N[i]): int(N[j]) for i, j in zip(ri, ci)}
    agree_before = sum(1 for r in rows if r["trace"] is not None and r["local"] == r["trace"]); agree_after = sum(1 for r in rows if r["trace"] is not None and perm[r["local"]] == r["trace"])
    return perm, dict(clicks=len(rows), with_trace=sum(1 for r in rows if r["trace"] is not None), agree_before=agree_before, agree_after=agree_after,
                      evidence={f"{int(N[i])}->{int(N[j])}": int(M[i, j]) for i, j in zip(ri, ci) if M[i, j] >= 1})


def renumber_map(perms, vids):
    """number map in the user-label format: '<photo>|<local>' -> reconciled number."""
    return {f"{v}|{L}": t for v in vids for L, t in perms[v].items()}


# ---------------------------------------------------------------- re-vote reusing the propagated masks
def propagated_renumbered(pl, chosen, held, nmap_v3, perms):
    pz = np.load(VP_DIR / "propagated_maps.npz"); prop = {k[1:]: pz[k] for k in pz.files}
    rings, _ = UL.rings_of(pl, pl["views"], {held}); seed_of = {}
    for rg in rings:
        seeds = [AP.view_file(v) for v in chosen if AP.view_file(v) in rg]
        if not seeds: continue
        for s, order in UL.segments(rg, seeds).items():
            for i in order: seed_of[UL.vid_of(rg[i])] = UL.vid_of(s)
    out, conflicts = {}, {}
    for v, m in prop.items():
        s = seed_of[v]; trans = {}
        for key, P in nmap_v3.items():
            ph, L = key.split("|")
            if ph != s or int(P) <= 0: continue
            trans.setdefault(int(P), set()).add(perms[s].get(int(L), int(L)))
        o = np.zeros_like(m)
        for P, ts in trans.items():
            if len(ts) == 1: o[m == P] = next(iter(ts))
            else: o[m == P] = -1; conflicts.setdefault(s, {})[P] = sorted(ts)
        out[v] = o
    return out, conflicts


def revote(pl, H2id, chosen, held, nmap_new, raw, prop_new):
    for v in chosen:
        UL.STATE_PPM[v] = UL.plant_px_per_mm(pl, AP.view_file(v), H2id[:, pl["views"].index(AP.view_file(v))])[0]
    topo_rows, _ = UL.topology_report(pl, raw, nmap_new, chosen + [held], H2id)
    seeds, _, _ = UL.build_vote_maps(pl, raw, nmap_new, chosen, [r for r in topo_rows if r["photo"] in chosen], "V-H+fill+topo")
    allm = dict(seeds, **prop_new); return UL.vote_views(pl, allm, chosen + sorted(prop_new), None, H2id)


# ---------------------------------------------------------------- steps 3 and 4
def full_graph(xyz):
    G, densR = RC.build_graph(xyz); return G


def step3_split(lab, blade, axis, traces, near, G):
    out = lab.copy(); info = []
    for C in sorted(set(lab[lab > 0].tolist())):
        inside = sorted({t["leaf"] for t in traces if (lambda m: m.sum() and (lab[m] == C).mean() >= FLAG["contain_share"])(near[t["key"]][1] & blade)})
        if len(inside) < 2: continue
        nodes = np.nonzero(((lab == C) & blade) | axis)[0]; idx = {n: i for i, n in enumerate(nodes)}; sub = G[nodes][:, nodes]
        D = []; offs = []
        for k in inside:
            src = [idx[n] for n in np.nonzero(np.isin(np.arange(len(lab)), nodes) & np.any([near[t["key"]][1] for t in traces if t["leaf"] == k], axis=0))[0]]
            d = dijkstra(sub, indices=src, min_only=True) if src else np.full(len(nodes), np.inf); D.append(d)
            offs.append(np.min([near[t["key"]][0][nodes] for t in traces if t["leaf"] == k], axis=0))
        D = np.array(D); offs = np.array(offs); D[offs > L_MAX_MM] = np.inf; best = D.argmin(0); ok = np.isfinite(D.min(0))
        cm = (lab[nodes] == C) & blade[nodes]; new = np.where(ok, np.array(inside)[best], 0)
        out[nodes[cm]] = new[cm]; info.append(dict(cluster=int(C), traces=inside, surfels=int(cm.sum()), to={int(k): int((new[cm] == k).sum()) for k in inside}, unassigned=int((new[cm] == 0).sum())))
    return out, info


def step4_fill(lab, blade, anchor, traces, near, G):
    out = lab.copy(); info = []
    for k in sorted({t["leaf"] for t in traces}):
        off = np.min([near[t["key"]][0] for t in traces if t["leaf"] == k], axis=0); seed = np.any([near[t["key"]][1] for t in traces if t["leaf"] == k], axis=0) & blade
        allowed = blade & (off <= L_MAX_MM) & ~(anchor & (lab != k) & (lab > 0))
        nodes = np.nonzero(allowed)[0]
        if not len(nodes) or not seed[nodes].any(): info.append(dict(leaf=k, filled=0)); continue
        nc, comp = connected_components(G[nodes][:, nodes], directed=False); keep = set(comp[seed[nodes]].tolist())
        reach = nodes[np.isin(comp, list(keep))]; f = reach[out[reach] == 0]; out[f] = k; info.append(dict(leaf=k, filled=int(len(f))))
    return out, info


# ---------------------------------------------------------------- scoring
def score_both(pl, lab, hv, col, truth_rec, truth_orig):
    ren = UL.render_ids(pl, np.where(lab > 0, lab, 0).astype(np.int16), hv, col)
    rows_r, g_r = UL.score_heldout(truth_rec, ren); cov = float(((ren > 0) & (truth_rec > 0)).sum() / max((truth_rec > 0).sum(), 1))
    P = sorted(set(ren[ren > 0].tolist())); T = sorted(set(truth_orig[truth_orig > 0].tolist()))
    M = np.array([[((ren == p) & (truth_orig == t)).sum() for t in T] for p in P]) if P and T else np.zeros((0, 0))
    ren2 = np.zeros_like(ren); match = {}
    if M.size:
        ri, ci = linear_sum_assignment(-M)
        for i, j in zip(ri, ci):
            if M[i, j] > 0: match[int(P[i])] = int(T[j]); ren2[ren == P[i]] = T[j]
        unmatched = [p for p in P if p not in match]
        for k, p in enumerate(unmatched): ren2[ren == p] = 1000 + k                 # distinct ids that never equal a truth number
    rows_o, g_o = UL.score_heldout(truth_orig, ren2)
    fl = lambda rows: {str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rows.items()}
    return dict(reconciled=dict(rows={str(k): v for k, v in rows_r.items()}, summary=g_r, coverage=round(cov, 4), flags=fl(rows_r)),
                original=dict(rows={str(k): v for k, v in rows_o.items()}, summary=g_o, flags=fl(rows_o), number_match=match)), ren


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args(argv); out = a.out; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    pl = UL.load_plant(); views = pl["views"]; H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; H = float(pl["xyz"][:, 2].max() - V2.POT_TOP); blade = pl["blade"]; axis = ~blade
    vc = json.load(open(UL_OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; photos = chosen + [held]
    nmap_v3 = json.load(open(MAP_FILE, encoding="utf-8"))["map"]; traces = CC.load_traces(TRACES)
    zs = np.load(VP_DIR / "surfel_clusters.npz"); lab0 = np.where(zs["leaf"] > 0, zs["leaf"], 0).astype(np.int16)
    near4 = trace_near(pl["xyz"], traces, R_T_MM)
    T0 = matrix_T(lab0, blade, traces, near4); F0 = flags_T(T0); print("T before", json.dumps(F0, default=str), flush=True)
    # step 1 click tables, step 2 permutations
    clicks, perms, ev = {}, {}, {}
    for v in photos:
        rows, meta = click_table(pl, H2id, v, traces); perm, e = renumber(rows); clicks[v] = dict(meta=meta, rows=rows); perms[v] = perm; ev[v] = e
        json.dump(dict(photo=v, numbering=NUMBERING, derived=True, permutation={str(k): x for k, x in perm.items()}, evidence=e, click_meta=meta, held_out=(v == held),
                       note="derived layer; user labels unchanged; applied at vote time"), open(out / f"label_renumber_{v}.json", "w", encoding="utf-8"), indent=1)
        print("photo", v, meta, e, flush=True)
    raw, _ = UL.label_maps(UL_OUT, photos); nmap_new = renumber_map(perms, photos)
    prop_new, pconf = propagated_renumbered(pl, chosen, held, nmap_v3, perms)
    lab1, cnt1 = revote(pl, H2id, chosen, held, nmap_new, raw, prop_new); lab1 = np.where(blade, lab1, 0).astype(np.int16)
    tot = cnt1.sum(1); share = np.where(tot > 0, cnt1.max(1) / np.maximum(tot, 1), 0); anchor = blade & (lab1 > 0) & (share >= ANCHOR["share_min"]) & (tot >= ANCHOR["views_min"])
    G = full_graph(pl["xyz"]); lab3, s3 = step3_split(lab1, blade, axis, traces, trace_near(pl["xyz"], traces, R_T_MM), G)
    near_full = trace_near(pl["xyz"], traces, R_T_MM); lab4, s4 = step4_fill(lab3, blade, anchor, traces, near_full, G)
    T1 = matrix_T(lab4, blade, traces, near4); F1 = flags_T(T1)
    # held-out truths
    truth_orig = UL.remap({held: raw[held]}, nmap_v3)[held]; truth_orig = np.where(truth_orig > 0, truth_orig, 0).astype(np.int16)
    truth_rec = UL.remap({held: raw[held]}, {f"{held}|{L}": t for L, t in perms[held].items()})[held]; truth_rec = np.where(truth_rec > 0, truth_rec, 0).astype(np.int16)
    hv = AP.view_file(held); col = H2id[:, views.index(hv)]; scores = {}; finals = dict(VP_before=lab0, revote=lab1, step3=lab3, step4=lab4)
    for name, lb in finals.items():
        scores[name], _ = score_both(pl, lb, hv, col, truth_rec, truth_orig)
        r, o = scores[name]["reconciled"], scores[name]["original"]
        print(f"held-out {name}: reconciled {r['summary']['median_iou']:.3f}/{r['summary']['max_confusion']:.3f} cov {r['coverage']:.3f} | original {o['summary']['median_iou']:.3f}/{o['summary']['max_confusion']:.3f}", flush=True)
    untraced = sorted(set(lab4[lab4 > 0].tolist()) - {t["leaf"] for t in traces})
    src = np.where(~blade, "axis", np.where((lab4 > 0) & (lab1 == 0), "fill", np.where((lab4 != lab1) & (lab1 > 0), np.where(lab4 > 0, "trace_split", "split_unassigned"), np.where(lab4 > 0, "renumbered_vote", "unassigned"))))
    res = dict(**LABELS, input="10-25 V-P map v3 (propagated masks reused)", traces_used=[t["key"] for t in traces], odd_even_split="retired for this step",
               T_before=T0, flags_before=F0, T_after=T1, flags_after=F1, click_tables={v: dict(meta=c["meta"], table={f"{l}->{t}": n for (l, t), n in Counter((r["local"], r["trace"]) for r in c["rows"]).items()}) for v, c in clicks.items()},
               permutations={v: {str(k): x for k, x in p.items()} for v, p in perms.items()}, renumber_evidence=ev, propagated_conflicts=pconf, step3=s3, step4=s4,
               untraced_leaves=untraced, held_out=scores, anchors=int(anchor.sum()), assigned={k: int((v > 0).sum()) for k, v in finals.items()}, seconds=round(time.time() - t0))
    best = max(["revote", "step3", "step4"], key=lambda k: (scores[k]["reconciled"]["summary"]["gate_pass"], scores[k]["reconciled"]["summary"]["median_iou"], -scores[k]["reconciled"]["summary"]["max_confusion"]))
    res["pick"] = best; res["tag"] = "gate_passed" if scores[best]["reconciled"]["summary"]["gate_pass"] else "gate_failed_viewing_only"
    json.dump(res, open(out / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    with open(out / "matrix_T.csv", "w", encoding="utf-8", newline="") as fh:
        labs = sorted({k for r in T0 + T1 for k in r["counts"]}); w = csv.writer(fh, lineterminator="\n"); w.writerow(["state", "trace", "leaf", "n"] + [f"label_{k}" for k in labs])
        for st, TT in (("before_VP", T0), ("after_step4", T1)):
            for r in TT: w.writerow([st, r["trace"], r["leaf"], r["n"]] + [r["counts"].get(k, 0) for k in labs])
    lb = finals["step4"]; leaf_full = np.where(lb > 0, lb, -1)
    with open(out / "surfel_leaf_ids_v3.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; numbering {NUMBERING}; final = step4; {res['tag']}; row = index into R3-1b_surfels_full.ply; leaf_id -1 = none\nrow,leaf_id,source\n")
        for r_, l_, s_ in zip(pl["rows"], leaf_full, src): fh.write(f"{r_},{l_},{s_}\n")
    flags = scores["step4"]["reconciled"]["flags"]; table = UL.leaf_table(pl, np.where(lb > 0, lb, 0).astype(np.int16), {}, H)
    for r in table: r["quality"] = flags.get(str(r["leaf"]), "not_in_held_out"); r["traced"] = r["leaf"] in {t["leaf"] for t in traces}
    with open(out / "leaves.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; numbering {NUMBERING}; step4; {res['tag']}; area = kind (1); quality on reconciled truth\n")
        keys = sorted({k for r in table for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table)
    np.savez_compressed(out / "surfel_clusters.npz", rows=pl["rows"], leaf=leaf_full, source=src.astype("U20"), axis_kind=zs["axis_kind"], **{f"{k}_leaf": np.where(v > 0, v, -1) for k, v in finals.items()})
    print("done", res["pick"], res["tag"], "step3", s3, "step4", s4, "untraced", untraced, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
