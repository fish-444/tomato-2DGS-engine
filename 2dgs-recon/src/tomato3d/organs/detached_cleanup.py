"""Revision R4: final clean-up of geometrically detached pieces after R3, with the existing clean-up rule A (vp_cleanup.absorb_detached, unchanged) on the
layer-aware sheet graph (graph_neighbours=True). Contract: contested_resolve/CONTRACT.md, Revision R4. May change coloured surfels (user instruction); R3 labels stay on disk.

usage: python -m tomato3d.organs.detached_cleanup
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import contested_resolve as CR
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import photo_then_geometry as PG
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import trace_reconcile as TR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vp_cleanup as VC

OUT = CR.OUT / "detached_cleanup"; STATUS = "R3-1b_unaccepted"
TAGS = tuple(PG.TAGS) + ("moved_detached", "unset_detached")


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]
    z3 = np.load(PG.OUT / "labels.npz"); lab = z3["label"][b].astype(int); tag3 = z3["source"][b]
    E = SR.edges_3mm(xyz); E = E[SR.edge_ok(xyz, S, E, LD.D["offset_mm"])]; n = len(xyz)
    G = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    new, moves = VC.absorb_detached(lab, xyz, G, graph_neighbours=True); print(f"pieces moved {len(moves)} ({time.time() - t0:.0f}s)", flush=True)
    changed = new != lab; moved = changed & (new > 0); unset = changed & (new == 0); tag = tag3.copy(); tag[moved] = 8; tag[unset] = 9
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
    s3, s4 = score(lab), score(new); g3, g4 = s3["summary"], s4["summary"]
    pm_before, _ = purity(lab, moved); pm_after, nmv = purity(new, moved); pu_before, nun = purity(lab, unset)
    per_leaf = {}
    for m in moves: per_leaf.setdefault(str(m["leaf"]), Counter())["moved" if m["to"] else "unset"] += m["surfels"]
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="contested_resolve_R4_detached_cleanup", start="R3", rule="vp_cleanup.absorb_detached on the sheet graph, graph_neighbours=True",
               constants=dict(DETACH_MM=VC.DETACH_MM, NEAR_MM=VC.NEAR_MM, edge_mm=LD.D["edge_mm"], angle_deg=LD.D["angle_deg"], offset_mm=LD.D["offset_mm"]),
               pieces=len(moves), moved_surfels=int(moved.sum()), unset_surfels=int(unset.sum()), coloured_base_changed=int((changed & (tag3 == 1)).sum()), per_leaf={L: dict(c) for L, c in per_leaf.items()},
               largest_moves=sorted(moves, key=lambda m: -m["surfels"])[:12], purity=dict(moved_before=pm_before, moved_after=pm_after, moved_projections=nmv, unset_before=pu_before, unset_projections=nun),
               held_out=dict(R3=s3, after=s4), reference=dict(R3=[0.5585, 0.4414], photo_first=[0.5474, 0.4253], V_P=[0.546, 0.438], A_B_C_Dprime=[0.517, 0.393]),
               tag=("gate_passed" if g4["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    fl = np.zeros(len(pl["xyz"]), np.int16); fl[b] = new; ft = np.zeros(len(pl["xyz"]), np.int8); ft[b] = tag
    np.savez_compressed(OUT / "labels.npz", rows=pl["rows"], label=fl, source=ft, source_names=np.array(TAGS))
    CR.images(B, lab, new, np.where(moved, 4, np.where((new == 0) & ((lab == 0) | unset), 5, np.minimum(tag3, 4))).astype(np.int8), OUT / "images")
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out", "largest_moves")}, indent=1, default=float, ensure_ascii=False))
    print("largest", [(m["leaf"], m["surfels"], m["gap_mm"], m["to"]) for m in res["largest_moves"]]); print("held-out R3", g3, "after", g4)
    print("rows after", {k: (round(v["iou"], 3), round(v["confusion"], 3)) for k, v in s4["rows"].items()})


if __name__ == "__main__":
    main()
