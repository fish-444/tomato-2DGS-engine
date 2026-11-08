"""Revision R2 of the contested-surfel resolution: absorb unlabelled clusters enclosed by one compound leaf, once by the photo criterion (V-P masks) and once by the
geometry criterion (neighbour majority + angular enclosure), each separately, on top of the R1 labels. Contract: contested_resolve/CONTRACT.md, Revision R2.

usage: python -m tomato3d.organs.enclosure_absorb
"""

from __future__ import annotations

import csv
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
from tomato3d.organs import vp_recluster as RC

OUT = CR.OUT / "enclosure"; R1 = CR.OUT / "geometry_first"; STATUS = "R3-1b_unaccepted"
E = dict(radius_mm=6.0, photo_share=0.5, photo_margin=0.15, geo_share=0.6, geo_cover_deg=180.0, bins=12, bin_min=3)
TAGS = ("other", "base", "diffused", "absorb_single_neighbour", "absorb_evidence", "still_contested", "absorb_geometry_only", "absorb_enclosed")


def neighbourhood(xyz, S, mem, coloured_idx, tree):
    """Coloured surfels within 6 mm of any member on the same sheet."""
    hits = set()
    for s_ in mem:
        for j in tree.query_ball_point(xyz[s_], E["radius_mm"] / 1000):
            t = coloured_idx[j]
            if CR.sheet_pair_ok(S, s_, t): hits.add(int(t))
    return np.array(sorted(hits), int)


def coverage_deg(xyz, S, mem, pts):
    c = xyz[mem].mean(0); n = S["normal"][mem].mean(0); n /= max(np.linalg.norm(n), 1e-12)
    a = np.cross(n, [1.0, 0, 0]); a = a if np.linalg.norm(a) > 1e-6 else np.cross(n, [0, 1.0, 0]); a /= np.linalg.norm(a); bvec = np.cross(n, a)
    v = xyz[pts] - c; ang = np.degrees(np.arctan2(v @ bvec, v @ a)) % 360; h = np.bincount((ang // (360 / E["bins"])).astype(int), minlength=E["bins"])
    return float((h >= E["bin_min"]).sum() * 360 / E["bins"])


def main():
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); B = LD.load("vp"); pl, b, xyz, S = B["pl"], B["b"], B["xyz"], B["S"]
    z1 = np.load(R1 / "labels.npz"); lab = z1["label"][b].astype(int); tag1 = z1["source"][b]
    z2 = np.load(CR.A2 / "labels_diffused.npz"); src2 = z2["source"][b]
    free = (lab == 0) & ((tag1 == 5) | (src2 == 4)); coloured = lab > 0
    cid, merged = CR.clusters(xyz, S, free); nC = cid.max() + 1; cidx = np.nonzero(coloured)[0]; tree = cKDTree(xyz[cidx])
    print(f"unlabelled {int(free.sum())} -> {nC} clusters ({time.time() - t0:.0f}s)", flush=True)
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; held = vc["held_out"]; B["chosen"] = chosen; B["held"] = held
    from tomato3d.organs import oversegment_second as O2
    vis_views = {vid: v[0] for vid, v in O2.vote_views(B).items()}
    raw, _ = UL.label_maps(UL.OUT, chosen + [held]); nmap_v3 = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]; seedm = UL.remap({v: raw[v] for v in chosen}, nmap_v3)
    vpmaps = {v: np.where(seedm[v] > 0, seedm[v], 0).astype(np.int16) for v in chosen}; zp = np.load(TR.VP_DIR / "propagated_maps.npz")
    for k in zp.files:
        if k[1:] != held and k[1:] not in vpmaps: vpmaps[k[1:]] = zp[k]
    vp_cnt, _, seen, _ = CR.photo_evidence(B, cid, lab, vis_views, vpmaps, {c: [] for c in range(nC)}); print(f"photo evidence ({time.time() - t0:.0f}s)", flush=True)
    newP, newG = lab.copy(), lab.copy(); rows = []
    for c in range(nC):
        mem = np.nonzero(cid == c)[0]; nb = neighbourhood(xyz, S, mem, cidx, tree); cnt = Counter(lab[nb].tolist()); cands = sorted(cnt)
        rec = dict(cluster=c, size=len(mem), candidates=";".join(f"{L}:{cnt[L]}" for L in cands), photos_seen=seen[c])
        # photo-based
        tot = sum(vp_cnt[c][L] for L in cands); sh = sorted(((vp_cnt[c][L] / tot if tot else 0.0, L) for L in cands), reverse=True)
        rec["photo_shares"] = ";".join(f"{L}:{s:.2f}" for s, L in sh)
        if sh and seen[c] and sh[0][0] >= E["photo_share"] and sh[0][0] - (sh[1][0] if len(sh) > 1 else 0) >= E["photo_margin"]: newP[mem] = sh[0][1]; rec["photo_decision"] = sh[0][1]
        else: rec["photo_decision"] = 0; rec["photo_reason"] = "unseen" if not seen[c] else ("no_candidate" if not cands else "share_or_margin")
        # geometry-based
        if cands:
            L, nL = cnt.most_common(1)[0]; share = nL / len(nb); cov = coverage_deg(xyz, S, mem, nb[lab[nb] == L]); rec.update(geo_leaf=L, geo_share=round(share, 3), geo_cover_deg=cov)
            if share >= E["geo_share"] and cov >= E["geo_cover_deg"]: newG[mem] = L; rec["geo_decision"] = L
            else: rec["geo_decision"] = 0; rec["geo_reason"] = "share" if share < E["geo_share"] else "not_enclosed"
        else: rec.update(geo_decision=0, geo_reason="no_candidate")
        rows.append(rec)
    prj = SR.projections(pl, b, chosen); truth = UL.remap({held: raw[held]}, nmap_v3)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16)
    H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; hv = AP.view_file(held); col = H2id[:, pl["views"].index(hv)]
    def score(lb):
        f = np.zeros(len(pl["xyz"]), np.int16); f[b] = lb; ren = UL.render_ids(pl, f, hv, col); rws, g = UL.score_heldout(truth, ren); cov = float(((ren > 0) & (truth > 0)).sum() / max((truth > 0).sum(), 1))
        return dict(rows={str(k): v for k, v in rws.items()}, summary=g, coverage=round(cov, 4))
    s_r1 = score(lab); res_all = {}
    for name, new in (("photo", newP), ("geometry", newG)):
        sub = OUT / name; sub.mkdir(exist_ok=True); added = (new > 0) & (lab == 0); hit = tot_ = 0
        for vid, p in prj.items():
            ul = vpmaps[vid][p["wi"], p["ui"]]; m = B["VISb"][:, p["k"]] & p["inside"] & added & (ul > 0); hit += int((new[m] == ul[m]).sum()); tot_ += int(m.sum())
        sa = score(new); g1, ga = s_r1["summary"], sa["summary"]; pur = round(hit / tot_, 4) if tot_ else None
        dec = [r for r in rows if r[f"{name[:5] if name == 'photo' else 'geo'}_decision"]]
        res = dict(input_status=STATUS, scale=LO.SCALE_TAG, grouping=f"contested_resolve_R2_enclosed_{name}", start="R1 geometry-first", rules=E, unlabelled_before=int(free.sum()), clusters=int(nC),
                   absorbed_clusters=len(dec), absorbed_surfels=int(added.sum()), unlabelled_after=int(((new == 0) & free).sum()), per_leaf_added=dict(Counter(int(x) for x in new[added])),
                   purity=dict(purity=pur, labelled_projections=tot_), held_out=dict(R1=s_r1, after=sa), reference=dict(R1=[0.5474, 0.4253], V_P=[0.546, 0.438], A_B_C_Dprime=[0.517, 0.393]),
                   passed=bool((pur or 0) >= 0.85 and ga["median_iou"] >= g1["median_iou"] and ga["max_confusion"] <= g1["max_confusion"]), tag=("gate_passed" if ga["gate_pass"] else "gate_failed_viewing_only"))
        json.dump(res, open(sub / "metrics.json", "w", encoding="utf-8"), indent=1, default=float); res_all[name] = res
        tg = tag1.copy(); tg[added] = 7; fl = np.zeros(len(pl["xyz"]), np.int16); fl[b] = new; ft = np.zeros(len(pl["xyz"]), np.int8); ft[b] = tg
        np.savez_compressed(sub / "labels.npz", rows=pl["rows"], label=fl, source=ft, source_names=np.array(TAGS))
        CR.images(B, lab, new, np.where(added, 4, np.where(free & (new == 0), 5, tag1)).astype(np.int8), sub / "images")
        print(name, json.dumps({k: res[k] for k in ("absorbed_clusters", "absorbed_surfels", "unlabelled_after", "per_leaf_added", "purity", "passed")}, default=float),
              "held-out", round(g1["median_iou"], 4), round(g1["max_confusion"], 4), "->", round(ga["median_iou"], 4), round(ga["max_confusion"], 4), {k: (round(v["iou"], 3), round(v["confusion"], 3)) for k, v in sa["rows"].items()}, flush=True)
    with open(OUT / "clusters.csv", "w", encoding="utf-8", newline="") as fh:
        keys = ["cluster", "size", "candidates", "photos_seen", "photo_shares", "photo_decision", "photo_reason", "geo_leaf", "geo_share", "geo_cover_deg", "geo_decision", "geo_reason"]
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n"); w.writeheader(); w.writerows([{k: r.get(k, "") for k in keys} for r in rows])
    both = sum(1 for r in rows if r["photo_decision"] and r["geo_decision"]); same = sum(1 for r in rows if r["photo_decision"] and r["photo_decision"] == r["geo_decision"])
    json.dump(dict(both_decide=both, same_leaf=same, photo_only=sum(1 for r in rows if r["photo_decision"] and not r["geo_decision"]), geo_only=sum(1 for r in rows if r["geo_decision"] and not r["photo_decision"]), seconds=round(time.time() - t0)),
              open(OUT / "agreement.json", "w"), indent=1); print("agreement", open(OUT / "agreement.json").read())


if __name__ == "__main__":
    main()
