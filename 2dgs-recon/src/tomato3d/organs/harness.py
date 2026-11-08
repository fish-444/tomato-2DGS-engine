"""Step 2 of CONTRACT_ORGAN_TOPOLOGY.md: error-localisation harness - the FIRST failing stage per plant, with evidence.

Stages: 1 connectivity, 2 base, 3 main axis, 4 laterals, 5 nodes, 6 membership & leaflets (definitions in the contract, section 2).
Binding mode = truth axis/blade labels; predicted labels (the 2026-10-22 labeller, unchanged) are information only.

usage: python -m tomato3d.organs.harness --split dev [--labels truth|pred] [--tag baseline] [--inds dev_00 ...]
Writes <run>/harness_<split>_<labels>_<tag>.json and prints the first-failure table (markdown).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.organs import contract as C
from tomato3d.organs import evaluate as EV
from tomato3d.organs import features as FE
from tomato3d.organs import labeller as LB
from tomato3d.organs import pipeline as PL

POT_Z_M = C.POT_CENTRE_M[2]
STAGES = ["connectivity", "base", "main_axis", "laterals", "nodes", "membership"]
TH = dict(reach_min=0.99, base_stem_mm=5.0, main_stem_share_min=0.90, main_top_tol_mm=3.0, lateral_min_pts=5, membership_min=0.85, f1_min=0.70)


def junction_points(z, rec, tol_mm: float = 0.3) -> np.ndarray:
    """Indices of non-stem axis points lying inside the truth stem tube (distance to the truth main polyline <= local radius + tol)."""
    tree = json.load(open(C.TOPO_RUN / "bench" / rec["split"] / rec["name"] / "tree.json")); main = next(a for a in tree["axes"] if a["kind"] == "main")
    Q = np.asarray(main["tube"]["polyline"], float) / 1000 + np.array(C.POT_CENTRE_M); R = np.asarray(main["tube"]["radius"], float) / 1000
    xyz = z["xyz"].astype(np.float64); fam = z["family"].astype(str); cand = np.nonzero((z["axis_label"] == 1) & (fam != "stem"))[0]
    if not len(cand):
        return cand
    P = xyz[cand]; best_d = np.full(len(P), np.inf); best_r = np.zeros(len(P))
    for k in range(len(Q) - 1):
        seg = Q[k + 1] - Q[k]; ln = np.linalg.norm(seg)
        if ln == 0:
            continue
        t = np.clip((P - Q[k]) @ seg / ln ** 2, 0, 1); d = np.linalg.norm(Q[k] + t[:, None] * seg - P, axis=1); m = d < best_d
        best_d[m] = d[m]; best_r[m] = R[k] + t[m] * (R[k + 1] - R[k])
    return cand[best_d <= best_r + tol_mm / 1000]


def load(split: str, name: str):
    d = C.TOPO_RUN / "bench" / split / name; z = np.load(d / "cloud.npz", allow_pickle=False)
    bs = json.load(open(C.TOPO_RUN / f"bench_summary_{split}.json")); rec = next(r for r in bs["plants"] if r["name"] == name)
    return z, rec


def _members(r, vs):
    sk = r["sk"]; parts = [r["axis_idx"][sk.members[v]] for v in vs if len(sk.members[v])]
    return np.concatenate(parts) if parts else np.array([], int)


def _majority(arr):
    if not len(arr):
        return "", 0.0
    c = Counter(arr.tolist()).most_common(1)[0]; return c[0], c[1] / len(arr)


def stages(z, rec, r) -> dict:
    """All six stage checks (each with pass flag + evidence) and the first failing stage."""
    xyz = z["xyz"].astype(np.float64); fam = z["family"].astype(str); organ = z["organ"].astype(str); leaf_t = z["leaf"].astype(str)
    sk = r["sk"]; ax = r["axis_idx"]; is_axis = r["is_axis"]; out = {}
    # revision 2 (2026-10-24, DEV): the generator starts every branch tube ON the stem axis, so a branch's first points lie inside the
    # stem's own surface; for the stage-3 share such points (within the truth stem radius + junction_tol_mm of the truth stem polyline)
    # count as stem (organ_j = 'M'). The raw share (OBJ names as they are) is reported alongside.
    organ_j = organ.copy(); organ_j[junction_points(z, rec)] = "M"
    # ---- 1 connectivity
    A = xyz[ax]; n = len(A)
    pairs = cKDTree(A).query_pairs(C.SKEL["max_edge_mm"] / 1000, output_type="ndarray")
    ncomp, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
    reached = np.ones(n, bool); reached[sk.unreached] = False; share = float(reached.mean()) if n else 0.0
    gaps = []
    if (~reached).any() and reached.any():
        rt = cKDTree(A[reached])
        for c in np.unique(lab[~reached]):
            m = (lab == c) & ~reached; dmin = float(rt.query(A[m])[0].min() * 1000)
            gaps.append(dict(points=int(m.sum()), nearest_reached_mm=round(dmin, 1), family=_majority(fam[ax][m])[0], organ=_majority(organ[ax][m])[0]))
    gaps.sort(key=lambda g: -g["points"])
    out["connectivity"] = dict(ok=share >= TH["reach_min"], reached_share=round(share, 4), components_3mm=int(ncomp), unreached_points=int((~reached).sum()),
                               bridges=len(sk.bridges), unreached_components=gaps[:6])
    # ---- 2 base
    root_children = sk.children.get(0, []); rl = _members(r, root_children); rl_fam = _majority(fam[rl]) if len(rl) else ("", 0.0)
    stem_present = ax[fam[ax] == "stem"]; z_low_present = float(xyz[stem_present, 2].min()) if len(stem_present) else np.nan
    reached_stem = stem_present[reached[np.searchsorted(ax, stem_present)]] if len(stem_present) else stem_present
    z_low_reached = float(xyz[reached_stem, 2].min()) if len(reached_stem) else np.nan
    base_ok = rl_fam[0] == "stem" and len(reached_stem) > 0 and (z_low_reached - z_low_present) * 1000 <= TH["base_stem_mm"]
    out["base"] = dict(ok=bool(base_ok), root_linked_points=int(len(rl)), root_linked_family=rl_fam[0], root_linked_share=round(rl_fam[1], 2),
                       lowest_stem_present_mm=round((z_low_present - POT_Z_M) * 1000, 1) if len(stem_present) else None,
                       lowest_stem_reached_mm=round((z_low_reached - POT_Z_M) * 1000, 1) if len(reached_stem) else None, root_far=bool(sk.root_far))
    # ---- 3 main axis
    mp = sk.main_path; mm = _members(r, mp); stem_share = float((organ_j[mm] == "M").mean()) if len(mm) else 0.0
    stem_share_raw = float((organ[mm] == "M").mean()) if len(mm) else 0.0
    div = None
    # divergence = the vertex after the LAST main-path vertex whose members are majority stem (junction vertices mix organs and are not divergences)
    stem_maj = [i for i, v in enumerate(mp) if len(sk.members[v]) and (organ[ax[sk.members[v]]] == "M").mean() >= 0.5]
    if stem_maj and stem_maj[-1] + 1 < len(mp):
        tail = _members(r, mp[stem_maj[-1] + 1:]); o, sh = _majority(organ[tail]) if len(tail) else ("", 0.0); f, _ = _majority(fam[tail]) if len(tail) else ("", 0.0)
        v = mp[stem_maj[-1] + 1]
        div = dict(vertex=int(v), ran_into_organ=o, family=f, share=round(sh, 2), z_mm=round((sk.V[v][2] - POT_Z_M) * 1000, 1), s_mm=round(float(sk.s_mm[v]), 1),
                   tail_vertices=len(mp) - stem_maj[-1] - 1, tail_points=int(len(tail)))
    end_z = (sk.V[mp[-1]][2] - POT_Z_M) * 1000; top_z = rec["truth"]["top_main_node_z_mm"]
    # revision 1 (2026-10-24, before any stage-3 number with the fix): the share is taken over the path up to the LAST organ-carrying
    # node (contract 4.3: the terminal segment beyond it is the apex with its trusses); the terminal segment is reported separately and
    # must not be majority rachis / side stem.
    carrying = [nd for nd in sk.nodes if any(lf["node"] == nd["id"] for lf in r["leaves"]) or any(sa["node"] == nd["id"] for sa in r["side_axes"])]
    i_last = max((mp.index(v) for nd in carrying for v in nd["vertices"] if v in mp), default=len(mp) - 1)
    core = _members(r, mp[:i_last + 1]); term = _members(r, mp[i_last + 1:])
    core_share = float((organ_j[core] == "M").mean()) if len(core) else 0.0; core_share_raw = float((organ[core] == "M").mean()) if len(core) else 0.0
    term_fam, term_sh = _majority(fam[term]) if len(term) else ("", 0.0)
    term_ok = term_fam in ("", "stem", "peduncle", "pedicel")
    out["main_axis"] = dict(ok=core_share >= TH["main_stem_share_min"] and end_z >= top_z - TH["main_top_tol_mm"] and term_ok, stem_share=round(core_share, 3),
                            stem_share_raw=round(core_share_raw, 3), stem_share_full_path=round(stem_share, 3), stem_share_full_path_raw=round(stem_share_raw, 3), terminal=dict(points=int(len(term)), vertices=len(mp) - i_last - 1, family=term_fam, share=round(term_sh, 2)),
                            path_end_z_mm=round(float(end_z), 1), truth_top_node_z_mm=top_z, truth_stem_top_z_mm=rec["truth"]["stem_top_z_mm"], diverges=div,
                            path_vertices=len(mp), apex_cuts=r["counts"].get("apex_cuts", 0), tube=r["counts"].get("tube"))
    # ---- 4 laterals
    leaf_vertex = {}
    for lf in r["leaves"]:
        for v in lf["vertices"]:
            leaf_vertex[v] = lf["id"]
    main_set = set(mp); lat_rows = []
    for L in rec["truth"]["laterals"]:
        sid = L["id"]; pts = ax[(organ[ax] == sid) & (fam[ax] == "side_stem")]
        row = dict(id=sid, parent_node=L["parent_node"], side_stem_points=int(len(pts)), leaves=L["leaves"])
        if len(pts) < TH["lateral_min_pts"]:
            row["status"] = "undetectable_by_axis"; lat_rows.append(row); continue
        det = None
        for sa in r["side_axes"]:
            pm = _members(r, sa["path"])
            if len(pm) and (organ[pm] == sid).mean() >= 0.5:
                det = sa["id"]; break
        verts = sk.vert_of[np.searchsorted(ax, pts)]; verts = verts[verts >= 0]
        where = Counter("main_stem" if v in main_set else ("leaf:" + leaf_vertex[v] if v in leaf_vertex else "other") for v in verts)
        absorbed = [k for k, c in where.items() if k.startswith("leaf:") and c / max(len(verts), 1) >= 0.5]
        if det:
            row["status"] = "detected"; row["pred"] = det
        elif absorbed:
            row["status"] = "absorbed"; row["into"] = absorbed[0]
        else:
            row["status"] = "missed"
        row["side_stem_points_lie_in"] = dict(where.most_common(3))
        # where did its leaves go? (pred leaf / side-leaf ids holding the majority of each truth side leaf's blade points)
        became = {}
        for lid in L["leaves"]:
            bm = (leaf_t == lid) & (fam == "blade"); cl = r["cl"][bm]; cl = cl[cl >= 0]
            if len(cl):
                pl = Counter(r["leaf_of_cluster"].get(int(c), "unattached") for c in cl).most_common(1)[0][0]; became[lid] = pl
        row["leaves_became"] = became; lat_rows.append(row)
    n_eval = sum(x["status"] != "undetectable_by_axis" for x in lat_rows)
    out["laterals"] = dict(ok=all(x["status"] in ("detected", "undetectable_by_axis") for x in lat_rows), truth=len(lat_rows), evaluable=n_eval,
                           detected=sum(x["status"] == "detected" for x in lat_rows), missed=sum(x["status"] == "missed" for x in lat_rows),
                           absorbed=sum(x["status"] == "absorbed" for x in lat_rows), undetectable=len(lat_rows) - n_eval,
                           pred_side_axes=len(r["side_axes"]), per_lateral=lat_rows)
    # ---- 5 nodes
    pred_nodes = r["counts"]["leaf_nodes"]; truth_ranks = rec["truth"]["main_ranks"]
    node_z = [round((sk.V[nd["vertices"][0]][2] - POT_Z_M) * 1000, 1) for nd in sk.nodes
              if any(lf["node"] == nd["id"] for lf in r["leaves"]) or any(sa["node"] == nd["id"] for sa in r["side_axes"])]
    out["nodes"] = dict(ok=pred_nodes == truth_ranks, pred=int(pred_nodes), truth=int(truth_ranks), abs_err=abs(int(pred_nodes) - int(truth_ranks)),
                        pred_node_z_mm=node_z, truth_node_z_mm=rec["truth"]["main_node_z_mm"], truth_node_rank=rec["truth"]["main_node_rank"],
                        pred_leaves=len(r["leaves"]), truth_main_leaves=rec["truth"]["per_axis"]["M"]["leaves"])
    # ---- 6 membership & leaflets
    pred_leaf = np.array([r["leaf_of_cluster"].get(int(k), "") if k >= 0 else "" for k in r["cl"]], dtype=object).astype(str)
    is_blade_truth = z["axis_label"] == 0
    mem = EV.membership(leaf_t, pred_leaf, is_blade_truth, z["obj"].astype(str)); lf1 = EV.leaflet_f1(z["obj"].astype(str), is_blade_truth, r["cl"])
    ncl = int(r["cl"].max()) + 1
    out["membership"] = dict(ok=mem["point"] >= TH["membership_min"] and lf1["f1"] >= TH["f1_min"], membership_point=round(mem["point"], 3), membership_leaflet=round(mem["leaflet"], 3),
                             leaflet_f1=round(lf1["f1"], 3), clusters=ncl, unattached=round(r["counts"]["unattached"] / max(ncl, 1), 3),
                             ambiguous=round(r["counts"]["ambiguous"] / max(ncl, 1), 3), pred_leaves=mem["pred_leaves"], truth_leaves=mem["truth_leaves"])
    first = next((s for s in STAGES if not out[s]["ok"]), None)
    out["first_failure"] = first or "pass"; out["counts"] = r["counts"]
    out["metrics_row"] = dict(node=EV.node_metrics(pred_nodes, truth_ranks), membership_point=mem["point"], membership_leaflet=mem["leaflet"], leaflet=lf1,
                              side=EV.side_axes(len(r["side_axes"]), len(rec["truth"]["laterals"])), ambiguous=out["membership"]["ambiguous"], unattached=out["membership"]["unattached"])
    return out


def one(split: str, name: str, labels: str, lab=None) -> dict:
    z, rec = load(split, name); xyz = z["xyz"].astype(np.float64); nrm = z["normal"].astype(np.float64)
    if labels == "truth":
        is_axis = z["axis_label"] == 1; is_blade = z["axis_label"] == 0
    else:
        F, _ = FE.compute(xyz); is_axis = FE.smooth(xyz, lab.p_axis(F) >= C.LABEL_THR); is_blade = ~is_axis
    r = PL.run(xyz, nrm, is_axis, C.POT_CENTRE_M, is_blade)
    st = stages(z, rec, r); st["name"] = name; st["perturbations"] = rec["perturbations"]
    return st


def table(rows: list[dict]) -> str:
    L = ["| plant | perturbations | first failure | reach | base | stem share / end vs top | laterals det/miss/abs (undet) | nodes pred/truth | memb | F1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        c, b, m, l, nd, me = s["connectivity"], s["base"], s["main_axis"], s["laterals"], s["nodes"], s["membership"]
        L.append(f"| {s['name']} | {','.join(s['perturbations']) or '-'} | **{s['first_failure']}** | {c['reached_share']:.3f} ({c['unreached_points']} unr.) | "
                 f"{b['root_linked_family'] or '-'} {b['lowest_stem_reached_mm']} / {b['lowest_stem_present_mm']} mm | {m['stem_share']:.2f} / {m['path_end_z_mm']} vs {m['truth_top_node_z_mm']}"
                 f"{' -> ' + m['diverges']['ran_into_organ'] + ' @' + str(m['diverges']['z_mm']) if m['diverges'] else ''} | "
                 f"{l['detected']}/{l['missed']}/{l['absorbed']} ({l['undetectable']}) of {l['truth']} | {nd['pred']}/{nd['truth']} | {me['membership_point']:.2f} | {me['leaflet_f1']:.2f} |")
    cnt = Counter(s["first_failure"] for s in rows)
    L.append(""); L.append("first failure counts: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda kv: (STAGES + ['pass']).index(kv[0]))))
    return "\n".join(L)


def gates(rows: list[dict]) -> dict:
    n = len(rows); lat = [x for s in rows for x in s["laterals"]["per_lateral"]]; ev = [x for x in lat if x["status"] != "undetectable_by_axis"]
    pooled = EV.pooled([s["metrics_row"] for s in rows])
    g1 = sum(s["connectivity"]["ok"] for s in rows); g2 = sum(s["base"]["ok"] for s in rows); g3 = sum(s["main_axis"]["ok"] for s in rows)
    det = sum(x["status"] == "detected" for x in ev); absd = sum(x["status"] == "absorbed" for x in ev)
    return dict(G1=dict(ok=g1 == n, plants=f"{g1}/{n}"), G2=dict(ok=g2 == n, plants=f"{g2}/{n}"), G3=dict(ok=g3 >= 10 if n == 12 else g3 >= round(n * 10 / 12), plants=f"{g3}/{n}"),
                G4=dict(ok=(det / max(len(ev), 1) >= 0.70) and absd == 0, detected=det, evaluable=len(ev), absorbed=absd, undetectable=len(lat) - len(ev), share=round(det / max(len(ev), 1), 3)),
                G5=dict(ok=pooled["node_exact"] >= C.PASS["node_exact_min"] and pooled["node_mae"] <= C.PASS["node_mae_max"] and pooled["membership_point"] >= C.PASS["leaf_membership_min"] and pooled["leaflet_f1"] >= C.PASS["leaflet_f1_min"],
                        node_exact=round(pooled["node_exact"], 3), node_mae=round(pooled["node_mae"], 3), membership=round(pooled["membership_point"], 3), leaflet_f1=round(pooled["leaflet_f1"], 3)),
                pooled=pooled)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=("dev", "test"), required=True); ap.add_argument("--labels", choices=("truth", "pred"), default="truth")
    ap.add_argument("--tag", default="baseline"); ap.add_argument("--inds", nargs="*"); a = ap.parse_args(argv)
    bs = json.load(open(C.TOPO_RUN / f"bench_summary_{a.split}.json")); names = a.inds or [r["name"] for r in bs["plants"]]
    lab = LB.Labeller.load() if a.labels == "pred" else None
    rows = []
    for nm in names:
        s = one(a.split, nm, a.labels, lab); rows.append(s)
        print(f"{nm}: first failure {s['first_failure']}", flush=True)
    g = gates(rows); txt = table(rows)
    out = dict(split=a.split, labels=a.labels, tag=a.tag, thresholds=TH, skel=C.SKEL, leaf=C.LEAF, topo=getattr(C, "TOPO", None), gates=g, plants=rows)
    json.dump(out, open(C.TOPO_RUN / f"harness_{a.split}_{a.labels}_{a.tag}.json", "w"), indent=1, default=float)
    (C.TOPO_RUN / f"harness_{a.split}_{a.labels}_{a.tag}.md").write_text(txt + "\n\ngates: " + json.dumps({k: v for k, v in g.items() if k != "pooled"}, default=float) + "\n", encoding="utf-8")
    print(txt); print("gates:", json.dumps({k: v for k, v in g.items() if k != "pooled"}, default=float))


if __name__ == "__main__":
    main()
