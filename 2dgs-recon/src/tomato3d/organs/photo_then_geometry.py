"""Revision R3 of the contested-surfel resolution: start from the photo-first labels, then decide each remaining blade surfel by xyz distance to the
neighbouring leaves (inverse-distance share) combined with per-surfel V-P photo votes from the propagated maps only (no labelling photo, no held-out photo).
Contract: contested_resolve/CONTRACT.md, Revision R3.

usage: python -m tomato3d.organs.photo_then_geometry
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import contested_resolve as CR
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD
from tomato3d.organs import sheet_retry as SR
from tomato3d.organs import trace_reconcile as TR
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2

OUT = CR.OUT / "photo_then_geometry"; STATUS = "R3-1b_unaccepted"
P3 = dict(radius_mm=6.0, w_dist=0.5, w_photo=0.5, total_min=0.5, margin=0.15, geo_only_share=0.7, geo_only_mm=3.0, d_floor_mm=0.5)
TAGS = ("other", "base", "diffused", "absorb_single_neighbour", "absorb_evidence", "still_contested", "geo_photo", "geo_only")


def candidate_distances(xyz, S, free_idx, lab):
    """Per free surfel: {leaf: nearest same-sheet 3D distance in mm} over coloured surfels within 6 mm."""
    cidx = np.nonzero(lab > 0)[0]; tree = cKDTree(xyz[cidx]); out = []
    for s_ in free_idx:
        dmin = {}
        for j in tree.query_ball_point(xyz[s_], P3["radius_mm"] / 1000):
            t = cidx[j]
            if not CR.sheet_pair_ok(S, s_, t): continue
            d = max(np.linalg.norm(xyz[t] - xyz[s_]) * 1000, P3["d_floor_mm"]); L = int(lab[t])
            if d < dmin.get(L, 1e9): dmin[L] = d
        out.append(dmin)
    return out


def photo_votes(B, free_idx, vis_views, vpmaps, nlab=13):
    """Per free surfel: count of photos (propagated V-P maps only) where it is front-visible and the map label at its pixel is L (index 0 unused)."""
    pl, xyz = B["pl"], B["xyz"]; V = np.zeros((len(free_idx), nlab), int)
    for vid, vis in vis_views.items():
        if vid not in vpmaps: continue
        view = f"20261021_{vid}_iOS.jpg"; pv = V2.project_view(pl["xyz"][:1], pl["sc"][:1], pl["po"], view, V2.VISP["res"]); u, w, z = V2.project_points(xyz[free_idx], pv); s = UL.H2 / V2.VISP["res"]
        ui = np.round(u * s).astype(int); wi = np.round(w * s).astype(int); ok = vis[free_idx] & (z > 0) & (ui >= 0) & (wi >= 0) & (ui < UL.H2) & (wi < UL.H2)
        L = np.zeros(len(free_idx), int); L[ok] = vpmaps[vid][wi[ok], ui[ok]]; m = L > 0; np.add.at(V, (np.nonzero(m)[0], L[m]), 1)
    return V


def decide(dmin, votes, wd, wp, margin):
    """Returns (leaf, tag) with tag in geo_photo / geo_only / contested / unseen / no_candidate."""
    if not dmin: return 0, "no_candidate"
    cands = sorted(dmin); inv = np.array([1 / dmin[L] for L in cands]); sd = inv / inv.sum(); nv = votes.sum()
    if nv == 0:
        k = int(np.argmax(sd)); L = cands[k]
        return (L, "geo_only") if sd[k] >= P3["geo_only_share"] and dmin[L] <= P3["geo_only_mm"] else (0, "unseen")
    sp = np.array([votes[L] / nv for L in cands]); tot = wd * sd + wp * sp; o = np.argsort(-tot); t2 = tot[o[1]] if len(o) > 1 else 0.0
    if tot[o[0]] >= P3["total_min"] and tot[o[0]] - t2 >= margin: return cands[o[0]], "geo_photo"
    return 0, "contested"


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]
    z0 = np.load(CR.OUT / "labels.npz"); lab = z0["label"][b].astype(int); tag0 = z0["source"][b]; src2 = np.load(CR.A2 / "labels_diffused.npz")["source"][b]
    free = (lab == 0) & ((tag0 == 5) | (src2 == 4)); fidx = np.nonzero(free)[0]
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; B["chosen"] = chosen; B["held"] = held
    from tomato3d.organs import oversegment_second as O2
    vis_views = {vid: v[0] for vid, v in O2.vote_views(B).items()}; zp = np.load(TR.VP_DIR / "propagated_maps.npz")
    vpmaps = {k[1:]: zp[k] for k in zp.files if k[1:] != held and k[1:] not in chosen}
    D = candidate_distances(xyz, S, fidx, lab); V = photo_votes(B, fidx, vis_views, vpmaps); print(f"free {len(fidx)}, propagated photos used {len([v for v in vis_views if v in vpmaps])} ({time.time() - t0:.0f}s)", flush=True)
    def run(wd, wp, mg):
        new = lab.copy(); tg = tag0.copy(); cnt = Counter()
        for k, s_ in enumerate(fidx):
            L, t = decide(D[k], V[k], wd, wp, mg); cnt[t] += 1
            if L: new[s_] = L; tg[s_] = 6 if t == "geo_photo" else 7
        return new, tg, cnt
    new, tg, cnt = run(P3["w_dist"], P3["w_photo"], P3["margin"])
    raw, _ = UL.label_maps(UL.OUT, chosen + [held]); nm = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; um = UL.remap({v: raw[v] for v in chosen}, nm); um = {v: np.where(um[v] > 0, um[v], 0) for v in chosen}
    prj = SR.projections(pl, b, chosen)
    def purity(newl, mask):
        hit = tot = 0
        for vid, p in prj.items():
            ul = um[vid][p["wi"], p["ui"]]; m = B["VISb"][:, p["k"]] & p["inside"] & mask & (ul > 0); hit += int((newl[m] == ul[m]).sum()); tot += int(m.sum())
        return (round(hit / tot, 4) if tot else None), tot
    truth = UL.remap({held: raw[held]}, nm)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16); H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def score(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; ren = UL.render_ids(pl, f, hv, col); rws, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        return dict(rows={str(k): v for k, v in rws.items()}, summary=g, coverage=round(cov, 4))
    s0, s1 = score(lab), score(new); g0, g1 = s0["summary"], s1["summary"]
    pur = {t: dict(zip(("purity", "labelled_projections"), purity(new, tg == i))) for t, i in (("geo_photo", 6), ("geo_only", 7))}
    pooled, ptot = purity(new, (tg == 6) | (tg == 7)); p8, p8n = purity(lab, np.isin(tag0, (3, 4)))
    passed = (pooled or 0) >= 0.85 and g1["median_iou"] >= g0["median_iou"] and g1["max_confusion"] <= g0["max_confusion"]
    sens = {}
    for name, (wd, wp, mg) in {"w0.7/0.3": (0.7, 0.3, P3["margin"]), "w0.3/0.7": (0.3, 0.7, P3["margin"]), "margin0.10": (P3["w_dist"], P3["w_photo"], 0.10), "margin0.25": (P3["w_dist"], P3["w_photo"], 0.25)}.items():
        n2, t2, c2 = run(wd, wp, mg); g2 = score(n2)["summary"]; pp, pn = purity(n2, (t2 == 6) | (t2 == 7))
        sens[name] = dict(assigned=int(((t2 == 6) | (t2 == 7)).sum()), purity=pp, median_iou=round(g2["median_iou"], 4), max_confusion=round(g2["max_confusion"], 4))
    added = (tg == 6) | (tg == 7)
    res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping="contested_resolve_R3_photo_then_geometry", start="photo-first (§8)", params=P3, photo_evidence="54 propagated V-P maps only (labelling + held-out photos excluded)",
               free_before=int(len(fidx)), decisions=dict(cnt), assigned=int(added.sum()), free_after=int(len(fidx) - added.sum()), per_leaf_added=dict(Counter(int(x) for x in new[added])),
               purity=dict(pooled=pooled, labelled_projections=ptot, **pur), section8_purity_rescored=dict(note="§8 absorbed surfels on the same 4 photos; §8 evidence had used them (circular)", purity=p8, labelled_projections=p8n),
               held_out=dict(start=s0, after=s1), reference=dict(start_photo_first=[0.5474, 0.4253], V_P=[0.546, 0.438], A_B_C_Dprime=[0.517, 0.393]),
               passed=bool(passed), beats_A_B_C_Dprime_on_both=bool(g1["median_iou"] > 0.517 and g1["max_confusion"] <= 0.393), sensitivity=sens, tag=("gate_passed" if g1["gate_pass"] else "gate_failed_viewing_only"), seconds=round(time.time() - t0))
    json.dump(res, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    fl = np.zeros(len(pl["xyz"]), np.int16); fl[b] = new; ft = np.zeros(len(pl["xyz"]), np.int8); ft[b] = np.where(free & ~added, 5, tg)
    np.savez_compressed(OUT / "labels.npz", rows=pl["rows"], label=fl, source=ft, source_names=np.array(TAGS))
    CR.images(B, lab, new, np.where(added, 4, np.where(free & ~added, 5, tag0)).astype(np.int8), OUT / "images")
    print(json.dumps({k: v for k, v in res.items() if k not in ("held_out", "params")}, indent=1, default=float, ensure_ascii=False))
    print("held-out start", g0, "after", g1); print("rows after", {k: (round(v["iou"], 3), round(v["confusion"], 3)) for k, v in s1["rows"].items()})


if __name__ == "__main__":
    main()
