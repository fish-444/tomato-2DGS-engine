"""Revision R5: detached pieces redone with a geometric-bridge test. A piece of leaf L that reaches L's main body through unlabelled blade surfels on the
sheet graph (geodesic <= 60 mm, other leaves are walls) stays with L and the bridge is filled with L; only unreachable pieces are moved (rule A) or unset.
Contract: contested_resolve/CONTRACT.md, Revision R5. Photos not used.

usage: python -m tomato3d.organs.bridge_cleanup
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import contested_resolve as CR
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import photo_then_geometry as PG
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import trace_reconcile as TR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vp_cleanup as VC

OUT = CR.OUT / "bridge_cleanup"; STATUS = "R3-1b_unaccepted"; BRIDGE_MM = 60.0; CORRIDOR_MM = 80.0
TAGS = tuple(PG.TAGS) + ("moved_detached", "unset_detached", "bridge_fill", "corridor_relabel")


def bridge_pass(xyz, S, lab, tag3, corridor=False, limit_mm=BRIDGE_MM):
    """Per leaf: far pieces (> DETACH_MM) stay if they reach the main body within limit_mm through own/unlabelled surfels (any surfel if corridor), the bridge
    taking the leaf; else rule A (two-hop majority, else unset). Returns new labels, tags (8 moved, 9 unset, 10 bridge fill, 11 corridor), log, near-piece sizes, previous labels."""
    n = len(xyz); E = SR.edges_3mm(xyz); E = E[SR.edge_ok(xyz, S, E, LD.D["offset_mm"])]; L_ = np.linalg.norm(xyz[E[:, 0]] - xyz[E[:, 1]], axis=1) * 1000
    G = coo_matrix((np.r_[L_, L_], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr(); G1 = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    new = lab.copy(); tag = tag3.copy(); log = []; tree_all = cKDTree(xyz); near_untouched = []; old_lab = lab.copy()
    for L in sorted(set(lab[lab > 0].tolist()), key=lambda L: -int((lab == L).sum())):
        nodes = np.nonzero(new == L)[0]; nc, comp = VC.components_of(G1, nodes)
        if nc <= 1: continue
        sizes = np.bincount(comp); main = nodes[comp == sizes.argmax()]
        allowed = np.ones(n, bool) if corridor else ((new == L) | (new == 0)); keep = np.nonzero(allowed)[0]; pos = -np.ones(n, int); pos[keep] = np.arange(len(keep)); sub = G[keep][:, keep]
        dist, pred, src = dijkstra(sub, directed=False, indices=pos[main], min_only=True, limit=limit_mm, return_predecessors=True)
        tmain = cKDTree(xyz[main])
        for c in range(nc):
            if c == sizes.argmax(): continue
            piece = nodes[comp == c]; gap = float(tmain.query(xyz[piece])[0].min() * 1000)
            if gap <= VC.DETACH_MM: near_untouched.append(int(len(piece))); continue          # R4's rule: near pieces are not detached
            dp = dist[pos[piece]]; k = int(np.argmin(dp))
            if np.isfinite(dp[k]):
                path = []; cur = pos[piece[k]]
                while cur >= 0 and pred[cur] >= 0: cur = pred[cur]; path.append(keep[cur])
                path = np.array(path, int); br = path[new[path] != L]
                if len(br):
                    nb = set(br.tolist())
                    for p_ in br: nb |= set(G1[p_].indices.tolist())
                    fill = np.array([q for q in nb if new[q] != L and (corridor or new[q] == 0)], int); old_lab[fill] = new[fill]; tag[fill] = np.where(new[fill] == 0, 10, 11); new[fill] = L
                else: fill = np.zeros(0, int)
                log.append(dict(leaf=int(L), surfels=int(len(piece)), gap_mm=round(gap, 1), geodesic_mm=round(float(dp[k]), 1), kept=True, bridge_filled=int(len(fill)))); continue
            near = set(j for js in tree_all.query_ball_point(xyz[piece], VC.NEAR_MM / 1000) for j in js) - set(piece.tolist())
            cnt = Counter(int(new[j]) for j in near if new[j] > 0 and new[j] != L); to = cnt.most_common(1)[0][0] if cnt else 0
            new[piece] = to; tag[piece] = 8 if to else 9; log.append(dict(leaf=int(L), surfels=int(len(piece)), gap_mm=round(gap, 1), geodesic_mm=None, kept=False, to=int(to)))
    return new, tag, log, near_untouched, old_lab


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=("unlabelled", "corridor"), default="unlabelled"); a = ap.parse_args(argv); corridor = a.mode == "corridor"
    global OUT, BRIDGE_MM
    if corridor: OUT = OUT / "corridor"; BRIDGE_MM = CORRIDOR_MM
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]
    z3 = np.load(PG.OUT / "labels.npz"); lab = z3["label"][b].astype(int); tag3 = z3["source"][b]
    new, tag, log, near_untouched, old_lab = bridge_pass(xyz, S, lab, tag3, corridor, BRIDGE_MM)
    print(f"pieces {len(log)}, kept {sum(l['kept'] for l in log)} ({time.time() - t0:.0f}s)", flush=True)
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]
    raw, _ = UL.label_maps(UL.OUT, chosen + [held]); nm = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; um = UL.remap({v: raw[v] for v in chosen}, nm); um = {v: np.where(um[v] > 0, um[v], 0) for v in chosen}
    prj = SR.projections(pl, b, chosen)
    def purity(lb, mask):
        hit = tot = 0
        for vid, p in prj.items():
            ul = um[vid][p["wi"], p["ui"]]; m = B["VISb"][:, p["k"]] & p["inside"] & mask & (ul > 0); hit += int((lb[m] == ul[m]).sum()); tot += int(m.sum())
        return (round(hit / tot, 4) if tot else None), tot
    truth = UL.remap({held: raw[held]}, nm)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16); H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def score(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; ren = UL.render_ids(pl, f, hv, col); rws, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        return dict(rows={str(k): v for k, v in rws.items()}, summary=g, coverage=round(cov, 4))
    s3, s5 = score(lab), score(new); g3, g5 = s3["summary"], s5["summary"]; moved = tag == 8; unset = tag == 9; bridge = (tag == 10) | (tag == 11); corr = tag == 11
    corridor_purity = dict(under_piece_leaf=dict(zip(("purity", "projections"), purity(new, corr))), under_previous_label=dict(zip(("purity", "projections"), purity(old_lab, corr))), surfels=int(corr.sum()))
    kept = [l for l in log if l["kept"]]; mv = [l for l in log if not l["kept"]]
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping=("contested_resolve_R5b_corridor" if corridor else "contested_resolve_R5_bridge_cleanup"), mode=a.mode, start="R3", bridge_limit_mm=BRIDGE_MM, detach_mm=VC.DETACH_MM, pieces=len(log),
               corridor_relabel=corridor_purity,
               near_pieces_untouched=dict(pieces=len(near_untouched), surfels=int(sum(near_untouched))),
               kept=dict(pieces=len(kept), surfels=int(sum(l["surfels"] for l in kept)), bridge_filled=int(bridge.sum()), geodesic_mm=dict(median=(float(np.median([l["geodesic_mm"] for l in kept])) if kept else None), max=(max(l["geodesic_mm"] for l in kept) if kept else None))),
               moved=dict(pieces=sum(1 for l in mv if l["to"]), surfels=int(moved.sum())), unset=dict(pieces=sum(1 for l in mv if not l["to"]), surfels=int(unset.sum())),
               per_leaf={str(L): dict(kept=sum(l["surfels"] for l in kept if l["leaf"] == L), moved_or_unset=sum(l["surfels"] for l in mv if l["leaf"] == L)) for L in sorted(set(l["leaf"] for l in log))},
               largest_kept=sorted(kept, key=lambda l: -l["surfels"])[:10], largest_moved=sorted(mv, key=lambda l: -l["surfels"])[:10],
               purity=dict(bridge_fill=dict(zip(("purity", "projections"), purity(new, bridge))), moved_after=dict(zip(("purity", "projections"), purity(new, moved))), moved_before=dict(zip(("purity", "projections"), purity(lab, moved)))),
               held_out=dict(R3=s3, after=s5), reference=dict(R3=[0.5585, 0.4414], R4=[0.5142, 0.4414], photo_first=[0.5474, 0.4253], V_P=[0.546, 0.438], A_B_C_Dprime=[0.517, 0.393]),
               tag=("gate_passed" if g5["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    fl = np.zeros(len(pl["xyz"]), np.int16); fl[b] = new; ft = np.zeros(len(pl["xyz"]), np.int8); ft[b] = tag
    np.savez_compressed(OUT / "labels.npz", rows=pl["rows"], label=fl, source=ft, source_names=np.array(TAGS))
    CR.images(B, lab, new, np.where(moved | bridge, 4, np.where((new == 0), 5, np.minimum(tag3, 4))).astype(np.int8), OUT / "images")
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out", "largest_kept", "largest_moved")}, indent=1, default=float, ensure_ascii=False))
    print("largest kept", [(l["leaf"], l["surfels"], l["gap_mm"], l["geodesic_mm"], l["bridge_filled"]) for l in res["largest_kept"]]); print("largest moved", [(l["leaf"], l["surfels"], l["gap_mm"], l["to"]) for l in res["largest_moved"]])
    print("held-out R3", g3, "after", g5); print("rows after", {k: (round(v["iou"], 3), round(v["confusion"], 3)) for k, v in s5["rows"].items()})


if __name__ == "__main__":
    main()
