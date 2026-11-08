"""V-P clean-up on the user's instruction: absorb detached pieces of a leaf into the nearby cluster (A), then fill gaps enclosed by one
colour (B). Contract: build/p1_vp_cleanup_20261026_01/CONTRACT.md.

usage: python -m tomato3d.organs.vp_cleanup
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import light_organs as LO
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.organs import vp_recluster as RC

ROOT = LO.ROOT
OUT = ROOT / "build/p1_vp_cleanup_20261026_01"
VP_DIR = RC.VP_DIR; MAP_FILE = RC.MAP_FILE
STATUS = "R3-1b_unaccepted"; GROUPING = "user_labelled_propagated_cleaned"
LABELS = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping=GROUPING)
DETACH_MM = 11.0; NEAR_MM = 5.0; GAP_SHARE = 0.80; GAP_EXTENT_MM = 18.6
FILL_SHARE = 0.60; FILL_MAX_MM = 37.2      # step C (contract addendum)
ISLAND_SHARE = 0.70; D_EDGE_SHARE = 0.50; D_MIN = 20; U_BASE = 100   # step D (contract addendum)
ISLAND_MAX_MM = 18.6; ISLAND_MAX_N = 300                              # D revised: size cap on islands


def components_of(G, nodes):
    if not len(nodes): return 0, np.zeros(0, int)
    return connected_components(G[nodes][:, nodes], directed=False)


def extent_mm(P):
    if len(P) < 2: return 0.0
    q = P[np.random.default_rng(0).choice(len(P), min(len(P), 300), replace=False)]
    return float(np.linalg.norm(q[:, None] - q[None], axis=2).max() * 1000)


def absorb_detached(lab, xyz, G, graph_neighbours=False):
    """Step A. lab: labels over the graph's nodes (0 = unassigned). graph_neighbours=True replaces the 5 mm Euclidean target search by the
    two-hop neighbours on G (used on a layer graph so that a piece is never handed to a leaf in another layer)."""
    out = lab.copy(); moves = []; tree_all = cKDTree(xyz)
    if graph_neighbours:
        def near(piece):
            one = set(G[piece].indices.tolist()); two = set(G[np.array(sorted(one), int)].indices.tolist()) if one else set(); return (one | two) - set(piece.tolist())
    else:
        def near(piece):
            return set(j for js in tree_all.query_ball_point(xyz[piece], NEAR_MM / 1000) for j in js) - set(piece.tolist())
    for L in sorted(set(lab[lab > 0].tolist())):
        nodes = np.nonzero(lab == L)[0]; nc, comp = components_of(G, nodes)
        if nc <= 1: continue
        sizes = np.bincount(comp); main = nodes[comp == sizes.argmax()]; tmain = cKDTree(xyz[main])
        for c in range(nc):
            if c == sizes.argmax(): continue
            piece = nodes[comp == c]; gap = float(tmain.query(xyz[piece])[0].min() * 1000)
            if gap <= DETACH_MM: continue
            nb = near(piece)
            cnt = Counter(int(lab[j]) for j in nb if lab[j] > 0 and lab[j] != L)
            to = cnt.most_common(1)[0][0] if cnt else 0; out[piece] = to
            moves.append(dict(leaf=int(L), surfels=int(len(piece)), gap_mm=round(gap, 1), to=int(to), neighbours=dict(cnt.most_common(3))))
    return out, moves


def fill_gaps(lab, xyz, G):
    """Step B: unassigned components enclosed by one label (>= 80 % of labelled neighbours) and no larger than one leaflet."""
    out = lab.copy(); fills = []; nodes = np.nonzero(lab == 0)[0]; nc, comp = components_of(G, nodes)
    for c in range(nc):
        piece = nodes[comp == c]; nb = np.unique(G[piece].indices); nl = lab[nb]; nl = nl[nl > 0]
        if not len(nl): continue
        L, k = Counter(nl.tolist()).most_common(1)[0]
        if k / len(nl) >= GAP_SHARE and extent_mm(xyz[piece]) <= GAP_EXTENT_MM:
            out[piece] = L; fills.append(dict(leaf=int(L), surfels=int(len(piece))))
    return out, fills


def fill_rest(lab, G):
    """Step C: outside-in majority fill of unassigned surfels (share >= 0.60), limited to geodesic <= 37.2 mm from the labels before C."""
    from scipy.sparse.csgraph import dijkstra
    out = lab.copy(); src = np.nonzero(lab > 0)[0]; reach = dijkstra(G, indices=src, min_only=True, limit=FILL_MAX_MM / 1000) if len(src) else np.full(len(lab), np.inf)
    cand = np.nonzero((lab == 0) & np.isfinite(reach))[0]; rounds = 0; indptr, indices = G.indptr, G.indices
    while True:
        rounds += 1; upd = {}
        for s in cand:
            if out[s] > 0: continue
            nl = out[indices[indptr[s]:indptr[s + 1]]]; nl = nl[nl > 0]
            if not len(nl): continue
            L, k = Counter(nl.tolist()).most_common(1)[0]
            if k / len(nl) >= FILL_SHARE: upd[s] = L
        if not upd: break
        for s, L in upd.items(): out[s] = L
    left = int(((out == 0)).sum())
    return out, dict(filled=int(((out > 0) & (lab == 0)).sum()), rounds=rounds, beyond_limit=int(((lab == 0) & ~np.isfinite(reach)).sum()), still_unassigned=left)


def complete_geometry(lab, xyz, G, cap=True):
    """Step D: enclosed islands take the surrounding label; remaining unassigned components join the leaf with most boundary edges
    (>= 50 %, height overlap) or become their own U cluster (ids U_BASE+)."""
    out = lab.copy(); info = dict(islands=[], components=[]); indptr, indices = G.indptr, G.indices
    def outside_labels(piece):
        ps = set(piece.tolist()); c = Counter()
        for s in piece:
            for j in indices[indptr[s]:indptr[s + 1]]:
                if j not in ps and 0 < out[j] < U_BASE: c[int(out[j])] += 1
        return c
    for L in sorted(set(out[(out > 0) & (out < U_BASE)].tolist())):
        nodes = np.nonzero(out == L)[0]; nc, comp = components_of(G, nodes)
        if nc <= 1: continue
        main = int(np.bincount(comp).argmax())
        for c in range(nc):
            if c == main: continue
            piece = nodes[comp == c]; ol = outside_labels(piece); tot = sum(ol.values())
            if tot and ol.most_common(1)[0][0] != L and ol.most_common(1)[0][1] / tot >= ISLAND_SHARE:
                to = ol.most_common(1)[0][0]; ext = extent_mm(xyz[piece])
                if cap and (ext > ISLAND_MAX_MM or len(piece) >= ISLAND_MAX_N):
                    info.setdefault("large_enclosed", []).append(dict(leaf=int(L), enclosed_by=int(to), surfels=int(len(piece)), extent_mm=round(ext, 1))); continue
                out[piece] = to; info["islands"].append(dict(leaf=int(L), to=int(to), surfels=int(len(piece)), extent_mm=round(ext, 1)))
    hr = {int(L): (float(np.percentile(xyz[out == L, 2], 5)), float(np.percentile(xyz[out == L, 2], 95))) for L in set(out[(out > 0) & (out < U_BASE)].tolist())}
    nodes = np.nonzero(out == 0)[0]; nc, comp = components_of(G, nodes); nu = 0
    for c in range(nc):
        piece = nodes[comp == c]; ol = outside_labels(piece); tot = sum(ol.values()); z0, z1 = float(xyz[piece, 2].min()), float(xyz[piece, 2].max())
        if tot:
            L, k = ol.most_common(1)[0]
            if k / tot >= D_EDGE_SHARE and L in hr and z1 >= hr[L][0] and z0 <= hr[L][1]:
                out[piece] = L; info["components"].append(dict(to=int(L), surfels=int(len(piece)), edge_share=round(k / tot, 2))); continue
        if len(piece) >= D_MIN:
            nu += 1; out[piece] = U_BASE + nu; info["components"].append(dict(to=f"U{nu}", surfels=int(len(piece)), edge_share=(round(ol.most_common(1)[0][1] / tot, 2) if tot else None)))
    info["new_clusters"] = nu; return out, info


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args(argv); out = a.out; out.mkdir(parents=True, exist_ok=True)
    pl = UL.load_plant(); views = pl["views"]; H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; H = float(pl["xyz"][:, 2].max() - V2.POT_TOP); blade = pl["blade"]
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); held = vc["held_out"]; nmap = json.load(open(MAP_FILE, encoding="utf-8"))["map"]
    raw, _ = UL.label_maps(UL.OUT, vc["chosen"] + [held]); truth = UL.remap({held: raw[held]}, nmap)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16)
    zs = np.load(VP_DIR / "surfel_clusters.npz"); base = np.where(zs["leaf"] > 0, zs["leaf"], 0).astype(np.int16)
    b = np.nonzero(blade)[0]; xyz = pl["xyz"][b]; G, _ = RC.build_graph(xyz)
    labA, moves = absorb_detached(base[b], xyz, G); labB, fills = fill_gaps(labA, xyz, G); labC, cinfo = fill_rest(labB, G); labDu, dinfo_u = complete_geometry(labC, xyz, G, cap=False); labD, dinfo = complete_geometry(labC, xyz, G, cap=True)
    hv = AP.view_file(held); col = H2id[:, views.index(hv)]; scores = {}; states = dict(VP=base[b], A=labA, AB=labB, ABC=labC, ABCD_unbounded=labDu, ABCD=labD)
    def full(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; return f
    for k, lb in states.items():
        ren = UL.render_ids(pl, full(np.where(lb >= U_BASE, 0, lb).astype(np.int16)), hv, col)      # U clusters are not leaves
        rows, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        flags = {str(L): ("usable" if r["iou"] >= RC.USABLE["iou_min"] and r["confusion"] <= RC.USABLE["conf_max"] else "viewing_only") for L, r in rows.items()}
        scores[k] = dict(rows={str(L): v for L, v in rows.items()}, summary=g, coverage=round(cov, 4), flags=flags, assigned=int((lb > 0).sum()))
        print(k, round(g["median_iou"], 3), round(g["max_confusion"], 3), "cov", round(cov, 3), {L: (round(r["iou"], 2), round(r["confusion"], 2)) for L, r in rows.items()}, flush=True)
    mv = Counter(); [mv.update({f"{m['leaf']}->{m['to']}": m["surfels"]}) for m in moves]; fl = Counter(); [fl.update({str(f["leaf"]): f["surfels"]}) for f in fills]
    tag = "gate_passed" if scores["ABCD"]["summary"]["gate_pass"] else "gate_failed_viewing_only"
    res = dict(**LABELS, base="10-25 V-P map v3", rules=dict(detach_mm=DETACH_MM, near_mm=NEAR_MM, gap_share=GAP_SHARE, gap_extent_mm=GAP_EXTENT_MM),
               moved_pieces=len(moves), moved_surfels=int(sum(m["surfels"] for m in moves)), unset_surfels=int(sum(m["surfels"] for m in moves if m["to"] == 0)), moves_by_pair=dict(mv),
               largest_moves=sorted(moves, key=lambda m: -m["surfels"])[:15], filled_components=len(fills), filled_surfels=int(sum(f["surfels"] for f in fills)), fills_by_leaf=dict(fl), step_d=dict(islands=dinfo["islands"], island_surfels=int(sum(i["surfels"] for i in dinfo["islands"])), components=dinfo["components"], new_clusters=dinfo["new_clusters"], large_enclosed=dinfo.get("large_enclosed", []), unbounded=dict(islands=len(dinfo_u["islands"]), island_surfels=int(sum(i["surfels"] for i in dinfo_u["islands"]))),
                           to_leaves=int(sum(c["surfels"] for c in dinfo["components"] if not str(c["to"]).startswith("U")))), step_c=dict(cinfo, by_leaf={str(L): int(((labC == L) & (labB == 0)).sum()) for L in sorted(set(labC[labC > 0].tolist()))}),
               held_out=scores, tag=tag)
    json.dump(res, open(out / "scores.json", "w", encoding="utf-8"), indent=1, default=float)
    lb = labD; src = np.where((labD != labC), "geometry_D", np.where((labA != base[b]) & (labA > 0), "absorbed", np.where((labA != base[b]) & (labA == 0) & (labC == 0), "unset", np.where((labB != labA), "gap_filled", np.where((labC != labB), "rest_filled", np.where(lb > 0, "kept", "unassigned"))))))
    leaf_full = np.full(len(pl["xyz"]), -1); leaf_full[b] = np.where(lb > 0, lb, -1); src_full = np.full(len(pl["xyz"]), "axis", dtype="U12"); src_full[b] = src
    np.savez_compressed(out / "surfel_clusters.npz", rows=pl["rows"], leaf=leaf_full, source=src_full, axis_kind=zs["axis_kind"])
    with open(out / "surfel_leaf_ids.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {GROUPING}; {tag}; row = index into R3-1b_surfels_full.ply; leaf_id -1 = none\nrow,leaf_id,source\n")
        for r_, l_, s_ in zip(pl["rows"], leaf_full, src_full): fh.write(f"{r_},{l_},{s_}\n")
    table = UL.leaf_table(pl, full(np.where(lb >= U_BASE, 0, lb).astype(np.int16)), {}, H); fls = scores["ABCD"]["flags"]
    for r in table: r["quality"] = fls.get(str(r["leaf"]), "not_in_held_out")
    with open(out / "leaves.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(f"# {STATUS}; {LO.SCALE_TAG}; grouping {GROUPING}; {tag}; area = kind (1)\n")
        keys = sorted({k for r in table for k in r}, key=lambda k: (k != "leaf", k)); w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows(table)
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out", "largest_moves")}, default=float), flush=True)


if __name__ == "__main__":
    main()
