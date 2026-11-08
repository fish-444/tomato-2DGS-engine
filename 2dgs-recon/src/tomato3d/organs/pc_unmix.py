"""R8_pc_unmix: rachis pieces, mixed leaflets and foreign leaflets moved to the compound leaf whose rachis spine they are attached to (geodesic distance, ratio 1.5).
Base = R8_pc + rachis absorption; photos only score the result. Contract: build/p1_pc_absorb_20261027_01/unmix/CONTRACT.md.

Attempt 2 (`tuned`, contract amendment A1): spines = non-stem surfels within 3 mm of the user trace; leaves without a trace frozen.

usage: python -m tomato3d.organs.pc_unmix [--tuned]
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import light_organs as LO

D0 = LO.ROOT / "build/p1_pc_absorb_20261027_01"; OUT = D0 / "R8_pc_unmix"
P = dict(edge_mm=3.0, angle_deg=30.0, ratio=1.5, leaflet_min=30, pure=0.9, majority=0.6, leftover_mm=15.0, accept_purity=0.6, min_proj=100)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(tuned=False):
    global OUT
    if tuned: OUT = D0 / "R8_pc_unmix_tuned"
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    Z = np.load(D0 / "kaggle_data/r8pc_bundle.npz"); X = Z["xyz"] * 1000.0; N = Z["normal"]; ax = Z["axis_kind"].astype(int); n = len(X); rows = Z["rows"]
    r8 = np.load(D0 / "R8_pc/labels.npz")["label"].astype(int); RX = np.load(D0 / "R8_pc_rachis/labels_rachis.npz"); rach = np.where(RX["target"] & (RX["label"] > 0), RX["label"], 0)
    lab0 = np.where(rach > 0, rach, np.where(r8 > 0, r8, 0)); leaves = sorted(set(lab0[lab0 > 0].tolist()))
    node = ax != 1; nid = np.nonzero(node)[0]; m = len(nid); pos = -np.ones(n, int); pos[nid] = np.arange(m); Xn = X[nid]
    sidx = Z["sheet_index"].astype(int)[nid]; nsh = Z["n_sheets"].astype(int)[nid]; bl = ax[nid] == 0
    E = cKDTree(Xn).query_pairs(P["edge_mm"], output_type="ndarray"); i, j = E[:, 0], E[:, 1]; bb = bl[i] & bl[j]
    ni, nj = N[nid][i], N[nid][j]; ang = np.degrees(np.arccos(np.clip(np.abs((ni * nj).sum(1)), 0, 1))); mn = ni + np.where(((ni * nj).sum(1) < 0)[:, None], -nj, nj); mn /= np.maximum(np.linalg.norm(mn, axis=1, keepdims=True), 1e-12)
    off = np.abs(((Xn[j] - Xn[i]) * mn).sum(1)); same = ~((nsh[i] > 1) & (nsh[j] > 1)) | (sidx[i] == sidx[j]); ok = ~bb | ((ang <= P["angle_deg"]) & (off <= float(Z["offset_mm"])) & same)
    i, j = i[ok], j[ok]; L = np.linalg.norm(Xn[i] - Xn[j], axis=1) + 1e-6; G = coo_matrix((np.r_[L, L], (np.r_[i, j], np.r_[j, i])), shape=(m, m)).tocsr()
    log(f"graph nodes {m}, edges {len(i)} ({time.time() - t0:.0f}s)")
    # spines
    tr = Z["trace_pts"] * 1000.0; tleaf = Z["trace_leaf"].astype(int); spine = {}; pieces = {}; frozen = set()
    if tuned:
        for k in leaves:
            if (tleaf == k).any(): d_, _ = cKDTree(tr[tleaf == k]).query(X[nid]); spine[k] = nid[d_ <= P["edge_mm"]]
            else: spine[k] = np.zeros(0, int); frozen.add(k)
        insp = np.zeros(n, bool)
        for k in leaves: insp[spine[k]] = True
        for k in leaves:
            rk = np.nonzero((rach == k) & ~insp)[0]
            if k in frozen or len(rk) == 0: pieces[k] = []; continue
            Ek = cKDTree(X[rk]).query_pairs(P["edge_mm"], output_type="ndarray"); cc = connected_components(coo_matrix((np.ones(len(Ek)), (Ek[:, 0], Ek[:, 1])), shape=(len(rk), len(rk))), directed=False)[1] if len(Ek) else np.arange(len(rk))
            pieces[k] = [rk[cc == c] for c in np.unique(cc)]
    for k in ([] if tuned else leaves):
        rk = np.nonzero(rach == k)[0]
        if len(rk) == 0: spine[k] = np.zeros(0, int); pieces[k] = []; continue
        Ek = cKDTree(X[rk]).query_pairs(P["edge_mm"], output_type="ndarray"); cc = connected_components(coo_matrix((np.ones(len(Ek)), (Ek[:, 0], Ek[:, 1])), shape=(len(rk), len(rk))), directed=False)[1] if len(Ek) else np.arange(len(rk))
        comps = [rk[cc == c] for c in np.unique(cc)]
        if (tleaf == k).any():
            tt = cKDTree(tr[tleaf == k]); sc = [float(tt.query(X[cmp])[0].min()) for cmp in comps]; s = int(np.argmin(sc))
        else: s = int(np.argmax([len(cmp) for cmp in comps]))
        spine[k] = comps[s]; pieces[k] = [cmp for q, cmp in enumerate(comps) if q != s]
    Dk = {k: (dijkstra(G, directed=False, indices=pos[spine[k]], min_only=True) if len(spine[k]) else np.full(m, np.inf)) for k in leaves}
    def dist(k, idx): return Dk[k][pos[idx]]
    log("spines", {k: int(len(spine[k])) for k in leaves}, "pieces", {k: len(pieces[k]) for k in leaves}, f"({time.time() - t0:.0f}s)")
    lab = lab0.copy(); moves = []; sp_all = np.zeros(n, bool)
    for k in leaves: sp_all[spine[k]] = True
    # 1. rachis pieces
    for k in leaves:
        for pc in pieces[k]:
            med = {j_: float(np.median(dist(j_, pc))) for j_ in leaves if j_ not in frozen}; best = min(med, key=med.get)
            if best != k and med[best] * P["ratio"] <= med[k]: lab[pc] = best; moves.append(dict(kind="rachis_piece", unit=f"rachis:{k}:{int(pc[0])}", surfels=int(len(pc)), old=k, new=best, d_old=round(med[k], 1), d_new=round(med[best], 1)))
    # leaflets (3D-first patches)
    PF = np.load(D0.parent / "p1_stacked_sheets_retry_20261026_01/oversegment/second_attempt/patches_filled.npz"); rp = {int(r): q for q, r in enumerate(rows)}; pid = np.full(n, -1, int); pid[[rp[int(r)] for r in PF["rows"]]] = PF["patch_filled"]
    flagged = []
    for p_ in np.unique(pid[pid >= 0]):
        mem = np.nonzero((pid == p_) & (lab > 0) & ~sp_all)[0]
        if len(mem) < P["leaflet_min"]: continue
        cnt = Counter(lab[mem].tolist()); k, c = cnt.most_common(1)[0]; share = c / len(mem)
        if k in frozen or any(l_ in frozen for l_ in cnt): continue
        med = {j_: float(np.median(dist(j_, mem))) for j_ in leaves if j_ not in frozen}; order = sorted(med, key=med.get); best = order[0]
        if share < P["pure"]:                                                                           # 2. mixed leaflet -> one label
            new = best if (np.isfinite(med[best]) and (len(order) == 1 or med[order[1]] >= P["ratio"] * med[best])) else (k if share >= P["majority"] else None)
            if new is None: flagged.append(int(p_)); continue
            ch = mem[lab[mem] != new]
            if len(ch): moves.append(dict(kind="mixed_leaflet", unit=f"patch:{int(p_)}", surfels=int(len(ch)), old=dict(Counter(lab[ch].tolist())), new=new, d_old=round(med[k], 1), d_new=round(med[new], 1))); lab[ch] = new
        else:                                                                                           # 3. foreign leaflet
            if best != k and np.isfinite(med[best]) and med[best] * P["ratio"] <= med[k]:
                moves.append(dict(kind="foreign_leaflet", unit=f"patch:{int(p_)}", surfels=int(len(mem)), old=k, new=best, d_old=round(med[k], 1), d_new=round(med[best], 1))); lab[mem] = best
    # 4. leftover blade cut off from its leaf
    inleaf = pid >= 0; nleft = 0
    for k in [l_ for l_ in leaves if l_ not in frozen]:
        dk_all = np.where(pos >= 0, Dk[k][np.maximum(pos, 0)], np.inf); cut = np.nonzero((lab == k) & (ax == 0) & ~inleaf & ~np.isfinite(dk_all))[0]
        for s_ in cut:
            fin = [j_ for j_ in leaves if j_ != k and j_ not in frozen and Dk[j_][pos[s_]] <= P["leftover_mm"]]
            if len(fin) == 1: lab[s_] = fin[0]; nleft += 1
    if nleft: moves.append(dict(kind="leftover_blade", unit="-", surfels=nleft, old="cut_off", new="nearest_finite", d_old=None, d_new=None))
    moved = lab != lab0; log(f"moved {int(moved.sum())} surfels in {len(moves)} units; mixed_unresolved {len(flagged)} ({time.time() - t0:.0f}s)")
    with open(OUT / "moves.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["kind", "unit", "surfels", "old", "new", "d_old", "d_new"]); wr.writeheader(); wr.writerows(moves)
    # scoring on the 4 labelling photos (human maps); photos are not evidence
    from tomato3d.organs import local_diffusion as LD, sheet_retry as SR, trace_reconcile as TR, userlabel as UL
    B = LD.load("vp"); pl, b = B["pl"], B["b"]; assert np.array_equal(pl["rows"], rows)
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); chosen = vc["chosen"]; raw, _ = UL.label_maps(UL.OUT, chosen); nm = json.load(open(TR.MAP_FILE, encoding="utf-8"))["map"]
    um = UL.remap(raw, nm); um = {v: np.where(um[v] > 0, um[v], 0) for v in chosen}; prj = SR.projections(pl, b, chosen)
    def purity(labb, mask):
        hit = tot = 0
        for vid, p in prj.items():
            ul = um[vid][p["wi"], p["ui"]]; mk = B["VISb"][:, p["k"]] & p["inside"] & mask & (ul > 0); hit += int((labb[mk] == ul[mk]).sum()); tot += int(mk.sum())
        return (round(hit / tot, 4) if tot else None), tot
    mv_b = moved[b]; pn, nn = purity(lab[b], mv_b); po, no = purity(lab0[b], mv_b); pall_new, _ = purity(lab[b], lab[b] > 0); pall_old, _ = purity(lab0[b], lab0[b] > 0)
    per_kind = {kind: dict(units=sum(1 for mv in moves if mv["kind"] == kind), surfels=int(sum(mv["surfels"] for mv in moves if mv["kind"] == kind)))
                for kind in ("rachis_piece", "mixed_leaflet", "foreign_leaflet", "leftover_blade")}
    ok = bool(nn >= P["min_proj"] and pn is not None and po is not None and pn >= po and pn >= P["accept_purity"])
    per_leaf = {str(k): dict(before=int((lab0 == k).sum()), after=int((lab == k).sum()), moved_in=int(((lab == k) & moved).sum()), moved_out=int(((lab0 == k) & moved).sum())) for k in leaves}
    res = dict(labels=["designated_not_adopted", "conditional_on_unvalidated_scale"] + (["tuned"] if tuned else []), frozen=sorted(frozen), params=P, spines={str(k): int(len(spine[k])) for k in leaves}, rachis_pieces={str(k): len(pieces[k]) for k in leaves},
               moved_surfels=int(moved.sum()), moved_blade=int(mv_b.sum()), moved_rachis=int((moved & (ax == 2)).sum()), units=per_kind, mixed_unresolved=len(flagged), per_leaf=per_leaf,
               purity_moved=dict(new_label=pn, old_label=po, projections=nn), purity_all_blade=dict(before=pall_old, after=pall_new), accept=ok, tag=("accepted_by_rule" if ok else "viewing_only"), seconds=round(time.time() - t0))
    np.savez_compressed(OUT / "labels_unmix.npz", rows=rows, label=lab.astype(np.int16), before=lab0.astype(np.int16), moved=moved, axis_kind=ax.astype(np.int8), spine=sp_all)
    json.dump(res, open(OUT / "results.json", "w", encoding="utf-8"), indent=1, default=float); log(json.dumps(res, default=float)[:1500])


if __name__ == "__main__":
    import sys
    main("--tuned" in sys.argv)
