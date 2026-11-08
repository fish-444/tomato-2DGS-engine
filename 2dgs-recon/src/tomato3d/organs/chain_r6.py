"""Revision R6: the whole compound-leaf chain rerun from V-P with one reach limit for every leaf and per-surfel photo splitting of contests; photo evidence
from the 54 propagated V-P maps only. Steps: diffusion (uniform reach) -> photo split -> R3 distance + photo -> R5 bridge pass.
Contract: build/p1_local_diffusion_20261027_01/contested_resolve/CONTRACT.md, Revision R6.

usage: python -m tomato3d.organs.chain_r6
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import bridge_cleanup as BC
from tomato3d.organs import contested_resolve as CR
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import photo_then_geometry as PG
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import trace_reconcile as TR
from tomato3d.organs import userlabel as UL

OUT = LD.OUT / "r6_chain"; STATUS = "R3-1b_unaccepted"
U_MM = 22.72; U_SENS = (15.48, 22.72, 40.2); SPLIT = dict(share=0.5, margin=0.15)
TAGS = ("other", "base", "diffused", "photo_split", "geo_photo", "geo_only", "moved_detached", "unset_detached", "bridge_fill", "contested", "out_of_reach")


def diffuse(B, G, leaves, C, cand, U):
    reach = {L: U for L in leaves}; lab_new, src = LD.assign(leaves, C, cand, reach, LD.D["margin_mm"])      # U != 12 mm default -> used as is
    return lab_new, src


def photo_split(C, leaves, src, U, idx_c, V):
    """Per contested surfel: contest set = leaves within reach and within the margin of the best; the photo-vote leader wins if share >= 0.5 and >= 0.15 ahead."""
    out = {}
    for k, s_ in enumerate(idx_c):
        c = C[:, s_]; cb = c.min(); cs = [L for L, x in zip(leaves, c) if np.isfinite(x) and x <= U and x - cb < LD.D["margin_mm"]]; nv = V[k].sum()
        if not cs or nv == 0: continue
        sh = sorted(((V[k][L] / nv, L) for L in cs), reverse=True); s2 = sh[1][0] if len(sh) > 1 else 0.0
        if sh[0][0] >= SPLIT["share"] and sh[0][0] - s2 >= SPLIT["margin"]: out[s_] = sh[0][1]
    return out


def run_chain(B, G, Gp, leaves, C, cand, U, vis_views, vpmaps, verbose=True):
    b = B["b"]; xyz, S = B["xyz"], B["S"]; base_b = B["base"][b]; t0 = time.time(); steps = {}
    lab_new, src = diffuse(B, G, leaves, C, cand, U); lab = np.where(base_b > 0, base_b, lab_new).astype(int); tag = np.where(base_b > 0, 1, np.where(src == 2, 2, np.where(src == 3, 9, np.where(src == 4, 10, 0)))).astype(np.int8)
    steps["1_diffusion"] = lab.copy()
    idx_c = np.nonzero(tag == 9)[0]; V = PG.photo_votes(B, idx_c, vis_views, vpmaps); sp = photo_split(C, leaves, src, U, idx_c, V)
    for s_, L in sp.items(): lab[s_] = L; tag[s_] = 3
    steps["2_photo_split"] = lab.copy()
    free = np.nonzero((lab == 0) & (B["axis"][b] == 0))[0]; D = PG.candidate_distances(xyz, S, free, lab); Vf = PG.photo_votes(B, free, vis_views, vpmaps)
    for k, s_ in enumerate(free):
        L, t = PG.decide(D[k], Vf[k], PG.P3["w_dist"], PG.P3["w_photo"], PG.P3["margin"])
        if L: lab[s_] = L; tag[s_] = 4 if t == "geo_photo" else 5
    steps["3_distance_photo"] = lab.copy()
    # bridge_pass uses 8/9/10 for its own tags: park the diffusion tags 9/10 (contested / out_of_reach) out of the way first (fix 2026-10-27; earlier R6/R7 files saved them as 7/8)
    tag_p = tag.astype(np.int16); tag_p[tag == 9] = 19; tag_p[tag == 10] = 20
    new, tg, log, near, _ = BC.bridge_pass(xyz, S, lab, tag_p, False, BC.BRIDGE_MM); tg = np.where(tg == 10, 8, np.where(tg == 8, 6, np.where(tg == 9, 7, tg)))
    tg = np.where(tg == 19, 9, np.where(tg == 20, 10, tg)).astype(np.int8)
    steps["4_bridge"] = new.copy()
    if verbose: print(f"U {U}: diffused {int((tag == 2).sum())}, photo split {len(sp)}, distance+photo {int(((tag == 4) | (tag == 5)).sum())}, bridge pieces {len(log)} ({time.time() - t0:.0f}s)", flush=True)
    return steps, tg, log


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]; base_b = B["base"][b]
    G, Gp, _ = LD.sheet_graph(xyz, S); seeds, reach0, info = LD.seeds_and_reach(B, Gp); leaves, C = LD.costs(G, seeds)
    from scipy.spatial import cKDTree
    dense_all = np.vstack([LD.CC.densify(t["pts"]) for t in B["traces"] if t["leaf"] > 0]); dnear, _ = cKDTree(np.vstack([xyz[base_b > 0], dense_all])).query(xyz)
    cand = (base_b == 0) & (B["axis"][b] == 0) & (dnear <= LD.D["candidate_mm"] / 1000)
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; B["chosen"] = chosen; B["held"] = held
    from tomato3d.organs import oversegment_second as O2
    vis_views = {vid: v[0] for vid, v in O2.vote_views(B).items()}; zp = np.load(TR.VP_DIR / "propagated_maps.npz"); vpmaps = {k[1:]: zp[k] for k in zp.files if k[1:] != held and k[1:] not in chosen}
    raw, _ = UL.label_maps(UL.OUT, chosen + [held]); nm = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; um = UL.remap({v: raw[v] for v in chosen}, nm); um = {v: np.where(um[v] > 0, um[v], 0) for v in chosen}
    prj = SR.projections(pl, b, chosen); truth = UL.remap({held: raw[held]}, nm)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16)
    H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def purity(lb, mask):
        hit = tot = 0
        for vid, p in prj.items():
            ul = um[vid][p["wi"], p["ui"]]; m = B["VISb"][:, p["k"]] & p["inside"] & mask & (ul > 0); hit += int((lb[m] == ul[m]).sum()); tot += int(m.sum())
        return (round(hit / tot, 4) if tot else None), tot
    def score(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; ren = UL.render_ids(pl, f, hv, col); rws, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        return dict(rows={str(k): {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in rws.items()}, summary=g, coverage=round(cov, 4))
    s_vp = score(base_b); res_sens = {}; main_out = None
    for U in U_SENS:
        steps, tg, log = run_chain(B, G, Gp, leaves, C, cand, U, vis_views, vpmaps); prev = base_b; per_step = {}
        for name, lb in steps.items():
            added = (lb != prev) & (lb > 0); p_, n_ = purity(lb, added); sc = score(lb)
            per_step[name] = dict(added=int(added.sum()), purity_added=p_, projections=n_, held_out=[round(sc["summary"]["median_iou"], 4), round(sc["summary"]["max_confusion"], 4)], coverage=sc["coverage"],
                                  unlabelled_blade=int(((lb == 0) & (B["axis"][b] == 0)).sum()))
            prev = lb
        final = steps["4_bridge"]; added_all = (final != base_b) & (final > 0); pall, nall = purity(final, added_all); sf = score(final)
        res_sens[f"U{U:g}"] = dict(per_step=per_step, pooled_purity_all_added=pall, projections=nall, final_held_out=sf)
        if U == U_MM: main_out = (steps, tg, log, sf, pall, nall, per_step)
    steps, tg, log, sf, pall, nall, per_step = main_out; final = steps["4_bridge"]; g = sf["summary"]; gvp = s_vp["summary"]
    passed = (pall or 0) >= 0.85 and g["median_iou"] >= gvp["median_iou"] and g["max_confusion"] <= gvp["max_confusion"]
    # leaf 4 neighbourhood check (the region the user pointed at)
    near4 = cKDTree(xyz[final == 4]).query(xyz)[0] * 1000 if (final == 4).any() else np.full(len(xyz), np.inf)
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="compound_chain_R6", start="V-P", uniform_reach_mm=U_MM, photo_evidence="54 propagated V-P maps only", split=SPLIT,
               per_step=per_step, pooled_purity_all_added=pall, projections=nall, per_leaf_final=dict(Counter(int(x) for x in final[final > 0])), leaf4_surfels=dict(VP=int((base_b == 4).sum()), R6=int((final == 4).sum())),
               grey_within_10mm_of_leaf4_after=int(((final == 0) & (B["axis"][b] == 0) & (near4 <= 10)).sum()), unlabelled_blade_final=int(((final == 0) & (B["axis"][b] == 0)).sum()),
               held_out=dict(VP=s_vp, final=sf), reference=dict(V_P=[0.546, 0.438], A_B_C_Dprime=[0.517, 0.393], previous_chain_R5=[0.5136, 0.4414]),
               pass_rule="pooled purity of all added >= 0.85 AND median IoU >= V-P AND max confusion <= V-P", passed=bool(passed), beats_A_B_C_Dprime_on_both=bool(g["median_iou"] > 0.517 and g["max_confusion"] <= 0.393),
               sensitivity={k: dict(pooled_purity=v["pooled_purity_all_added"], held_out=[round(v["final_held_out"]["summary"]["median_iou"], 4), round(v["final_held_out"]["summary"]["max_confusion"], 4)],
                                    steps={s: (d["added"], d["purity_added"], d["held_out"]) for s, d in v["per_step"].items()}) for k, v in res_sens.items()},
               tag=("gate_passed" if g["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    fl = np.zeros(len(pl["xyz"]), np.int16); fl[b] = final; ft = np.zeros(len(pl["xyz"]), np.int8); ft[b] = tg                      # untagged stays 0 (other); it was written as 9 (contested) before 2026-10-27
    np.savez_compressed(OUT / "labels.npz", rows=pl["rows"], label=fl, source=ft, source_names=np.array(TAGS), **{f"step_{k}": v for k, v in steps.items()})
    CR.images(B, base_b, final, np.where((final != base_b) & (final > 0), 4, np.where(final == 0, 5, 1)).astype(np.int8), OUT / "images")
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out",)}, indent=1, default=float, ensure_ascii=False))
    print("final held-out", g, {k: (v["iou"], v["confusion"]) for k, v in sf["rows"].items()})


if __name__ == "__main__":
    main()
