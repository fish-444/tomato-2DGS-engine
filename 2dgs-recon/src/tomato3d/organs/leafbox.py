"""P1 session A — one oriented box per compound leaf (overlap allowed); overlap / outside surfels resolved by rules (same sheet, path to rachis, area density), never by the box alone.
Self-contained (numpy + scipy) so the same file runs as a Kaggle CPU kernel. No photo, SAM, Roboflow or colour evidence.
Contract: build/p1_leafbox_20261029_01/CONTRACT.md.

usage: python leafbox.py [--data dir] [--out dir]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

P = dict(edge_mm=3.0, angle_deg=30.0, pad_sheet_mm=1.5, pad_loc_mm=1.0, pct=(2.0, 98.0), bin_mm=2.0, min_rachis=10, outside_mm=10.0, anchor_mm=3.0, path_ratio=1.5, radii_mm=(4.0, 8.0, 12.0),
         density_margin=0.25, iters=4, stop_change=0.005, base_window_mm=10.0, band_mm=5.0, hide=0.20, repeats=5, mc_points=20000, poly_inside_min=0.95, u_ratio_min=0.9,
         side_share=0.05, side_dist_mm=60.0, chunk=200)
CODE = {-1: "contested", -2: "outside", -3: "stem", -4: "rachis_unpainted", 0: "not_a_surfel_label"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name, arg):
    for g in ([os.path.join(arg, name)] if arg else []) + [f"/kaggle/input/**/{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", name)]:
        hit = glob.glob(g, recursive=True)
        if hit: return hit[0]
    raise SystemExit(f"{name} not found")


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [round(c - h, 4), round(c + h, 4)]


class LeafBox:
    def __init__(self, data):
        Z = np.load(find("r8pc_bundle.npz", data)); R8 = np.load(find("labels.npz", data)); RX = np.load(find("labels_rachis.npz", data)); LT = np.load(find("per_surfel_light.npz", data))
        assert np.array_equal(Z["rows"], R8["rows"]) and np.array_equal(Z["rows"], RX["rows"]) and np.array_equal(Z["rows"], LT["ply_row"]), "row order differs"
        self.X = Z["xyz"] * 1000.0; self.N = Z["normal"]; self.ax = Z["axis_kind"].astype(int); self.n = len(self.X); self.t_sheet = float(Z["t_sheet_mm"]); self.off = float(Z["offset_mm"])
        self.r8 = R8["label"].astype(int); self.node = R8["node"].astype(bool); rx = RX["label"].astype(int); tg = RX["target"].astype(bool)
        self.rach_lab = np.where(tg & (rx > 0), rx, 0); self.area = LT["area_b_silhouette_mm2"].astype(float); self.light = LT["absorbed_ru_mm2"].astype(float)
        self.leaves = sorted(set(self.r8[self.r8 > 0].tolist())); self.nid = np.nonzero(self.node)[0]; self.m = len(self.nid); self.pos = -np.ones(self.n, int); self.pos[self.nid] = np.arange(self.m)
        Xn = self.X[self.nid]; Nn = self.N[self.nid]; sidx = Z["sheet_index"].astype(int)[self.nid]; nsh = Z["n_sheets"].astype(int)[self.nid]; nsh[self.ax[self.nid] == 2] = 1
        E = cKDTree(Xn).query_pairs(P["edge_mm"], output_type="ndarray"); i, j = E[:, 0], E[:, 1]; ni, nj = Nn[i], Nn[j]
        ang = np.degrees(np.arccos(np.clip(np.abs((ni * nj).sum(1)), 0, 1))); mn = ni + np.where(((ni * nj).sum(1) < 0)[:, None], -nj, nj); mn /= np.maximum(np.linalg.norm(mn, axis=1, keepdims=True), 1e-12)
        offv = np.abs(((Xn[j] - Xn[i]) * mn).sum(1)); same = ~((nsh[i] > 1) & (nsh[j] > 1)) | (sidx[i] == sidx[j]); ok = (ang <= P["angle_deg"]) & (offv <= self.off) & same; i, j = i[ok], j[ok]
        L = np.linalg.norm(Xn[i] - Xn[j], axis=1) + 1e-6; self.G = coo_matrix((np.r_[L, L], (np.r_[i, j], np.r_[j, i])), shape=(self.m, self.m)).tocsr(); self.comp = connected_components(self.G, directed=False)[1]
        self.stem = self.X[self.ax == 1]; self.stem_tree = cKDTree(self.stem)
        self.targets0 = np.nonzero(self.node & (self.r8 <= 0))[0]
        log(f"surfels {self.n}, nodes {self.m}, edges {len(i)}, leaves {self.leaves}, targets {len(self.targets0)}, rachis-absorbed {int((self.rach_lab > 0).sum())}")

    def own(self, lab_nodes_full, k):
        return np.nonzero((lab_nodes_full == k) | (self.rach_lab == k))[0]

    def fit_box(self, k, lab_full):
        own = self.own(lab_full, k); bl = own[self.ax[own] != 2] if False else own[self.rach_lab[own] != k]; ra = own[self.rach_lab[own] == k]; flags = []
        if len(ra) < P["min_rachis"]: ra = bl; flags.append("rachis_from_blade")
        R = self.X[ra]; c = R.mean(0); _, _, vt = np.linalg.svd(R - c, full_matrices=False); u = vt[0]; t = (R - c) @ u; perp = (R - c) - np.outer(t, u); resid = float(np.sqrt((perp ** 2).sum(1).mean()))
        pa, pb = c + u * t.min(), c + u * t.max()
        if self.stem_tree.query(pb)[0] < self.stem_tree.query(pa)[0]: u = -u; t = -t
        edges = np.arange(t.min(), t.max() + P["bin_mm"], P["bin_mm"]); idx = np.digitize(t, edges); poly = [np.median(R[idx == b], 0) for b in np.unique(idx)]; poly = np.array(poly)
        nb = self.N[bl]; wa = self.area[bl][:, None]; m0 = (nb * wa).sum(0); nb = np.where((nb @ m0 < 0)[:, None], -nb, nb); w = (nb * wa).sum(0)
        w = w - (w @ u) * u
        if np.linalg.norm(w) < 1e-9: w = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0])
        w /= np.linalg.norm(w); v = np.cross(u, w); v /= np.linalg.norm(v); w = np.cross(v, u); Rm = np.vstack([u, v, w])
        Q = (self.X[own] - c) @ Rm.T; lo, hi = np.percentile(Q, P["pct"][0], 0), np.percentile(Q, P["pct"][1], 0); mid = (lo + hi) / 2; centre = c + Rm.T @ mid; half = (hi - lo) / 2 + P["pad_sheet_mm"] + P["pad_loc_mm"]
        base, tip = poly[0], poly[-1]; nb_ = bl[np.linalg.norm(self.X[bl] - base, axis=1) <= P["base_window_mm"]]
        vext = float(np.ptp((self.X[nb_] - c) @ v)) if len(nb_) >= 2 else 0.0; sigma = float(np.sqrt(resid ** 2 + (0.5 * vext) ** 2))
        return dict(leaf=k, centre=centre, R=Rm, half=half, poly=poly, base=base, tip=tip, resid=resid, sigma=sigma, own=own, blade=bl, rachis=ra, flags=flags)

    @staticmethod
    def dist_box(B, P_):
        Q = (P_ - B["centre"]) @ B["R"].T; return np.maximum(np.abs(Q) - B["half"], 0).max(1)

    def resolve(self, seed_full, targets, iters=P["iters"], verbose=True):
        """EM: fit boxes from owners, membership of targets, rules a-d; returns final labels (full), per-target code, boxes, history."""
        lab = seed_full.copy(); lab[targets] = 0; hist = []; prev = np.zeros(len(targets), int); tpos = self.pos[targets]
        for it in range(1, iters + 1):
            boxes = {k: self.fit_box(k, lab) for k in self.leaves}
            D = np.stack([self.dist_box(boxes[k], self.X[targets]) for k in self.leaves], 1)
            own_comp = {k: set(self.comp[self.pos[boxes[k]["own"]][self.pos[boxes[k]["own"]] >= 0]].tolist()) for k in self.leaves}
            passa = np.stack([np.isin(self.comp[tpos], list(own_comp[k])) for k in self.leaves], 1)
            anch = {}
            Xn = self.X[self.nid]; tn = cKDTree(Xn)
            for k in self.leaves:
                a_ = sorted(set(sum(tn.query_ball_point(boxes[k]["poly"], P["anchor_mm"]), []))); anch[k] = a_
            DB = np.stack([dijkstra(self.G, directed=False, indices=anch[k], min_only=True)[tpos] if anch[k] else np.full(len(targets), np.inf) for k in self.leaves], 1)
            nodelab = lab[self.nid]; areaN = self.area[self.nid]; new = np.zeros(len(targets), int); code = np.zeros(len(targets), int); rule = np.full(len(targets), "", object)
            inside = D == 0; cand = np.where(inside.any(1)[:, None], inside, D <= P["outside_mm"]) & passa
            need_c = []
            for q in range(len(targets)):
                cs = np.nonzero(cand[q])[0]
                if len(cs) == 0: code[q] = -2 if not (D[q] <= P["outside_mm"]).any() else -1; rule[q] = "no_candidate_a"; continue
                if len(cs) == 1 and inside[q].sum() == 1 and inside[q, cs[0]]: new[q] = self.leaves[cs[0]]; rule[q] = "inside_one_a"; continue
                db = DB[q, cs]; o = np.argsort(db)
                if np.isfinite(db[o[0]]) and (len(cs) == 1 or db[o[1]] / max(db[o[0]], 1e-6) >= P["path_ratio"]): new[q] = self.leaves[cs[o[0]]]; rule[q] = "b_path"; continue
                need_c.append(q)
            for s0 in range(0, len(need_c), P["chunk"]):
                qs = need_c[s0:s0 + P["chunk"]]; Dg = dijkstra(self.G, directed=False, indices=tpos[qs], limit=max(P["radii_mm"]))
                for r_, q in enumerate(qs):
                    cs = np.nonzero(cand[q])[0]; ks = [self.leaves[c_] for c_ in cs]; win = None; okc = True
                    for rad in P["radii_mm"]:
                        mm = (Dg[r_] <= rad) & (nodelab > 0); tot = areaN[mm].sum(); sh = np.array([areaN[mm & (nodelab == k)].sum() for k in ks]) / tot if tot > 0 else np.zeros(len(ks))
                        o = np.argsort(-sh); top = ks[o[0]]; mg = sh[o[0]] - (sh[o[1]] if len(ks) > 1 else 0.0)
                        if tot <= 0 or mg < P["density_margin"] or (win is not None and top != win): okc = False; break
                        win = top
                    if okc and win is not None: new[q] = win; rule[q] = "c_density"
                    else: code[q] = -1; rule[q] = "d_contested"
            change = float((new != prev).mean()); prev = new.copy(); lab = seed_full.copy(); lab[targets] = np.where(new > 0, new, 0)
            hist.append(dict(iteration=it, assigned=int((new > 0).sum()), contested=int((code == -1).sum()), outside=int((code == -2).sum()), changed_share=round(change, 5),
                             by_rule={r: int((rule == r).sum()) for r in ("inside_one_a", "b_path", "c_density", "d_contested", "no_candidate_a")}))
            if verbose: log("iteration", hist[-1])
            if it > 1 and change < P["stop_change"]: break
        boxes = {k: self.fit_box(k, lab) for k in self.leaves}
        top_c = np.array(self.leaves)[np.argmin(np.where(passa, D, D + 1e6), 1)]
        return lab, new, code, boxes, hist, D, inside, top_c


def obb_inside(B, P_):
    Q = (P_ - B["centre"]) @ B["R"].T; return (np.abs(Q) <= B["half"]).all(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None); ap.add_argument("--out", default=None); a = ap.parse_args()
    out = a.out or ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "leafbox_out")); os.makedirs(out, exist_ok=True); t0 = time.time()
    M = LeafBox(a.data); seed = np.where(M.r8 > 0, M.r8, 0); seed = np.where(M.rach_lab > 0, M.rach_lab, seed)
    lab, new, code, boxes, hist, D, inside, top_c = M.resolve(seed, M.targets0)
    T = M.targets0; res = dict(individual="R3-1b", labels=["designated_not_adopted", "conditional_on_unvalidated_scale", "seed=R8_pc"], params=P, no_photo_colour_evidence=True, iterations=hist,
                               targets=int(len(T)), assigned=int((new > 0).sum()), contested=int((code == -1).sum()), outside=int((code == -2).sum()))
    # containment
    cont = {}
    for k in M.leaves:
        B = boxes[k]; ins = obb_inside(B, M.X); own = (lab == k) | (M.rach_lab == k); foreign = ins & ((lab > 0) | (M.rach_lab > 0)) & ~own
        cont[str(k)] = dict(own_inside_share=round(float(ins[own].mean()), 4), surfels_inside=int(ins.sum()), foreign_owned_inside=int(foreign.sum()), foreign_share=round(float(foreign.sum() / max(ins.sum(), 1)), 4))
    res["containment"] = cont
    # H2
    s = M.stem; sc_ = s.mean(0); _, _, vt = np.linalg.svd(s - sc_, full_matrices=False); sd = vt[0]; h2 = {}; weak = []
    for k in M.leaves:
        B = boxes[k]; poly = B["poly"]; pin = float(obb_inside(B, poly).mean()); plen = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum()) if len(poly) > 1 else 0.0; ur = float(2 * B["half"][0] / plen) if plen > 0 else None
        own = B["own"]; P_ = M.X[own]; tt = (P_ - sc_) @ sd; foot = sc_ + np.outer(tt, sd); r = (P_ - foot)[:, :2]; mdir = r.mean(0); mdir /= max(np.linalg.norm(mdir), 1e-9); opp = (r @ mdir) < 0
        side_flag = bool(opp.mean() >= P["side_share"] and opp.any() and (~opp).any() and np.linalg.norm(P_[opp].mean(0) - P_[~opp].mean(0)) > P["side_dist_mm"])
        ok = bool(pin >= P["poly_inside_min"] and ur is not None and ur >= P["u_ratio_min"])
        h2[str(k)] = dict(petiole_bases=1, polyline_inside_share=round(pin, 3), u_extent_over_rachis_length=(round(ur, 3) if ur else None), rachis_length_mm=round(plen, 1), leaflets="unavailable (no leaflet labels on the R8_pc numbering)",
                          stem_sides_flag=side_flag, opposite_share=round(float(opp.mean()), 3), pass_=ok, box_flags=B["flags"])
        if not ok: weak.append(k)
    res["H2"] = h2; res["tag"] = "leafbox_ok" if not weak else "leafbox_weak"; res["weak_leaves"] = weak
    # H3 overlap
    rng = np.random.default_rng(0); pairs = []; vol = {k: float(np.prod(2 * boxes[k]["half"])) for k in M.leaves}; ovl = {k: [] for k in M.leaves}
    allowned = np.nonzero((lab > 0) | (M.rach_lab > 0) | np.isin(np.arange(M.n), T))[0]
    for ii, ka in enumerate(M.leaves):
        for kb in M.leaves[ii + 1:]:
            A, Bb = boxes[ka], boxes[kb]; U = rng.uniform(-1, 1, (P["mc_points"], 3)) * A["half"]; pts = A["centre"] + U @ A["R"]; frac = float(obb_inside(Bb, pts).mean()); inter = frac * vol[ka]
            if inter <= 0: continue
            both = obb_inside(A, M.X[allowned]) & obb_inside(Bb, M.X[allowned]); sh = inter / min(vol[ka], vol[kb])
            pairs.append(dict(a=ka, b=kb, volume_share=round(sh, 4), surfels_in_overlap=int(both.sum()))); ovl[ka].append(dict(leaf_id=kb, volume_share=round(sh, 4), n_surfels=int(both.sum()))); ovl[kb].append(dict(leaf_id=ka, volume_share=round(sh, 4), n_surfels=int(both.sum())))
    ninside = inside.sum(1); con = code == -1
    res["H3"] = dict(pairs=sorted(pairs, key=lambda d: -d["volume_share"]), contested_in_overlap=int((con & (ninside >= 2)).sum()), contested_not_in_overlap=int((con & (ninside < 2)).sum()),
                     targets_in_overlap=int((ninside >= 2).sum()), targets_not_in_overlap=int((ninside < 2).sum()))
    # H4
    h4 = {}
    for k in M.leaves:
        own = (lab == k) | (M.rach_lab == k); ck = T[con & (top_c == k)]
        h4[str(k)] = dict(own_area_mm2=round(float(M.area[own].sum()), 1), contested_area_mm2=round(float(M.area[ck].sum()), 1), contested_area_share=round(float(M.area[ck].sum() / max(M.area[own].sum(), 1e-9)), 4),
                          own_light=round(float(M.light[own].sum()), 3), contested_light=round(float(M.light[ck].sum()), 3), contested_light_share=round(float(M.light[ck].sum() / max(M.light[own].sum(), 1e-12)), 4))
    res["H4"] = h4
    # H1 proxy: hide 20 % of the R8_pc boundary band
    bl = np.nonzero((M.r8 > 0) & (M.ax == 0))[0]; tr_ = cKDTree(M.X[bl]); dd, jj = tr_.query(M.X[bl], k=32, distance_upper_bound=P["band_mm"]); v_ = np.isfinite(dd); jj = np.where(v_, jj, 0)
    band = bl[(v_ & (M.r8[bl][jj] != M.r8[bl][:, None])).any(1)]; reps = []; K = N_ = 0
    for rep in range(P["repeats"]):
        g = np.random.default_rng(rep); hid = g.choice(band, int(round(P["hide"] * len(band))), replace=False); tg2 = np.concatenate([T, hid])
        l2, n2, c2, _, _, _, _, _ = M.resolve(seed, tg2, verbose=False); got = l2[hid]; truth = M.r8[hid]; asg = got > 0; ok = int((got[asg] == truth[asg]).sum()); na = int(asg.sum())
        reps.append(dict(repeat=rep, hidden=int(len(hid)), assigned=na, correct=ok, accuracy=(ok / na if na else None), coverage=na / len(hid))); K += ok; N_ += na; log("H1 proxy", reps[-1])
    res["H1"] = dict(human_heldout="not_available (no lite-seed held-out clicks for R3-1b)", proxy=dict(note="consistency with R8_pc labels; not_a_gate; different protocol from R8_pc's 0.833 (stage-1 truth)",
                                                                                                   band=int(len(band)), repeats=reps, pooled_accuracy=(K / N_ if N_ else None), wilson95=wilson(K, N_),
                                                                                                   pooled_coverage=N_ / sum(r["hidden"] for r in reps)))
    # interface outputs
    full = np.zeros(M.n, np.int32); full[(lab > 0)] = lab[lab > 0]; full[M.rach_lab > 0] = M.rach_lab[M.rach_lab > 0]; full[T[code == -1]] = -1; full[T[code == -2]] = -2; full[(M.ax == 1)] = -3
    full[(M.ax == 2) & (full == 0)] = -4
    leaves_json = []
    for k in M.leaves:
        B = boxes[k]; own = (full == k); ck = T[con & (top_c == k)]; a_own = float(M.area[own].sum())
        fl = list(B["flags"]) + (["stem_sides"] if h2[str(k)]["stem_sides_flag"] else []) + ([] if h2[str(k)]["pass_"] else ["H2_fail"])
        leaves_json.append(dict(leaf_id=int(k), source="seed", petiole_base_xyz=[round(float(x), 2) for x in B["base"]], petiole_base_sigma_mm=round(B["sigma"], 2), rachis_dir=[round(float(x), 5) for x in B["R"][0]],
                                rachis_polyline=[[round(float(x), 2) for x in p] for p in B["poly"]], rachis_residual_mm=round(B["resid"], 2), tip_xyz=[round(float(x), 2) for x in B["tip"]],
                                box=dict(centre=[round(float(x), 2) for x in B["centre"]], R=[[round(float(x), 6) for x in r] for r in B["R"]], half_extent_mm=[round(float(x), 2) for x in B["half"]]),
                                n_surfels=int(own.sum()), area_b_mm2_range=[round(a_own, 1), round(a_own, 1), round(a_own + float(M.area[ck].sum()), 1)], overlap_with=ovl[k], flags=fl))
    bj = dict(individual="R3-1b", scale_status="conditional_on_unvalidated_scale (board frame mm; rim fit not recorded for R3-1b)", frame="board frame, mm", labels=res["labels"], leaves=leaves_json, contested_surfels=int(con.sum()))
    json.dump(bj, open(os.path.join(out, "boxes.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "labels_leafbox.npz"), label=full, code_names=np.array([f"{k}:{v}" for k, v in CODE.items()]), r8_pc=M.r8.astype(np.int16), target=np.isin(np.arange(M.n), T), resolved_rule_note=np.array("see results.json iterations"))
    json.dump(hist, open(os.path.join(out, "iteration_log.json"), "w"), indent=1); res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, default=float); log("done", res["seconds"], "s", res["tag"], "assigned", res["assigned"], "contested", res["contested"])


if __name__ == "__main__":
    main()
