"""Auto seeds without human clicks: rachis-class surfel components -> auto stage 1 -> R8_pc -> rachis absorption -> boxes -> self-grades; human leaves used only for the
reproduction score. Runs as a Kaggle CPU kernel; imports pc_absorb.py, pc_rachis.py and leafbox.py shipped in the same dataset.
Contract: build/p1_auto_seed_20261029_01/CONTRACT.md.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

A = dict(edge_mm=2.0, min_comp=50, sens_edges=(1.5, 3.0), stage1_mm=3.0, bin_mm=2.0, gap_mm=6.0, base_mm=10.0, ray_mm=20.0, ray_hit_mm=5.0, overlap_max=0.5, iou_min=0.5, trace_mm=3.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name):
    for g in (f"/kaggle/input/**/{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", name)):
        hit = glob.glob(g, recursive=True)
        if hit: return hit[0]
    raise SystemExit(f"{name} not found")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def components(X, idx, r):
    E = cKDTree(X[idx]).query_pairs(r, output_type="ndarray"); m = len(idx)
    G = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(m, m)).tocsr() if len(E) else coo_matrix((m, m)).tocsr(); return connected_components(G, directed=False)[1]


def polyline(P, stem_tree):
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); u = vt[0]; t = (P - c) @ u
    if stem_tree.query(c + u * t.max())[0] < stem_tree.query(c + u * t.min())[0]: u = -u; t = -t
    bins = np.floor((t - t.min()) / A["bin_mm"]).astype(int); occ = np.unique(bins); poly = np.array([np.median(P[bins == b], 0) for b in occ])
    gaps = np.diff(occ) * A["bin_mm"] if len(occ) > 1 else np.zeros(0); return poly, u, float(gaps.max()) if len(gaps) else 0.0


def main():
    out = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_out"); os.makedirs(out, exist_ok=True); t0 = time.time()
    # module plumbing: pc_rachis imports tomato3d.organs.{light_organs, pc_absorb}
    for p_ in ("tomato3d", "tomato3d.organs"): sys.modules.setdefault(p_, types.ModuleType(p_))
    lo = types.ModuleType("tomato3d.organs.light_organs"); lo.ROOT = Path(out); sys.modules["tomato3d.organs.light_organs"] = lo
    PA = load_module("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load_module("tomato3d.organs.pc_rachis", find("pc_rachis.py")); LB = load_module("leafbox", find("leafbox.py"))
    Z = dict(np.load(find("r8pc_bundle.npz"))); X = Z["xyz"] * 1000.0; ax = Z["axis_kind"].astype(int); n = len(X)
    H8 = np.load(find("labels.npz"))["label"].astype(int); HX = np.load(find("labels_rachis.npz")); hx = np.where(HX["target"] & (HX["label"] > 0), HX["label"], 0)
    human = np.where(hx > 0, hx, np.where(H8 > 0, H8, 0))
    stem = np.nonzero(ax == 1)[0]; stem_tree = cKDTree(X[stem]); sc = components(X, stem, A["edge_mm"]); big = Counter(sc.tolist()).most_common()
    zext = {c: float(np.ptp(X[stem[sc == c], 2])) for c, _ in big[:10]}; stem_main = max(zext, key=zext.get)
    R = dict(status=["designated_not_adopted", "conditional_on_unvalidated_scale", "seed=auto_axis"], params=A, stem=dict(axis1=int(len(stem)), components=int(sc.max() + 1), main_vertical_extent_mm=round(zext[stem_main], 1), main_size=int((sc == stem_main).sum())))
    # 2. rachis candidates
    a2 = np.nonzero(ax == 2)[0]; cc = components(X, a2, A["edge_mm"]); cnt = Counter(cc.tolist()); keep = [c for c, k in cnt.items() if k >= A["min_comp"]]
    R["sensitivity_candidates"] = {f"{e} mm": int(sum(1 for _, k in Counter(components(X, a2, e).tolist()).items() if k >= A["min_comp"])) for e in A["sens_edges"]}
    cand = []
    for c in keep:
        mem = a2[cc == c]; poly, u, gap = polyline(X[mem], stem_tree); cand.append(dict(mem=mem, poly=poly, u=u, gap=gap, base_z=float(poly[0][2])))
    cand.sort(key=lambda d: d["base_z"])
    seedlab = np.zeros(n, int)
    for k, d in enumerate(cand, 1): seedlab[d["mem"]] = k
    R["candidates"] = dict(count=len(cand), sizes=[int(len(d["mem"])) for d in cand], axis2_total=int(len(a2)), axis2_in_candidates=int((seedlab > 0).sum()), human_leaves=8)
    log("candidates", R["candidates"], "sens", R["sensitivity_candidates"])
    # 3. auto stage 1
    bl = np.nonzero(ax == 0)[0]; st1 = np.zeros(n, int); sm = np.nonzero(seedlab > 0)[0]; ts = cKDTree(X[sm])
    for i, nb in zip(bl, ts.query_ball_point(X[bl], A["stage1_mm"])):
        if nb:
            ls = set(seedlab[sm[nb]].tolist())
            if len(ls) == 1: st1[i] = ls.pop()
    R["auto_stage1_blade"] = int((st1 > 0).sum())
    # 4. R8_pc unchanged with the auto stage 1
    Z2 = dict(Z); Z2["stage1"] = st1.astype(np.int16); M = PA.Model(Z2); M.norm_unary()
    labn = np.nonzero(M.st1n > 0)[0]; d_, jj, v_ = PA.knn(M.tree_n, M.Xn[labn], 32, PA.P["band_mm"]); jj = np.where(v_, jj, 0); band = labn[(v_ & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1)]
    cal_backup = PA.P["cal"]; PA.P["cal"] = tuple(sorted(set(M.st1n[M.st1n > 0].tolist())))[0::2]          # calibration half by leaf parity of the auto ids (no human split exists)
    auc = M.colour_auc(band) if len(band) else {k: dict(auc=0.5) for k in ("C1", "C2", "C4")}; wcol = {k: (PA.WC[k] if (k == "C3" or auc[k]["auc"] >= PA.P["auc_min"]) else 0.0) for k in PA.WC}
    cross = (M.st1n[M.ei] > 0) & (M.st1n[M.ej] > 0) & (M.st1n[M.ei] != M.st1n[M.ej]); samee = (M.st1n[M.ei] > 0) & (M.st1n[M.ei] == M.st1n[M.ej]); xa, xb = M.edge_raw["C3"][cross], M.edge_raw["C3"][samee]
    auc3 = float(np.searchsorted(np.sort(xb), xa).mean() / len(xb)) if len(xa) and len(xb) else 0.5; wcol["C3"] = PA.WC["C3"] if auc3 >= PA.P["auc_min"] else 0.0; PA.P["cal"] = cal_backup
    g, c_ = PA.SETS["B"]; r8 = M.run(g, c_, wcol, M.st1n.copy()); lab8 = np.zeros(n, int); lab8[M.nid] = r8["lab"]
    R["r8pc_auto"] = dict(colour_weights=wcol, colour_auc={**{k: v["auc"] for k, v in auc.items()}, "C3": round(auc3, 3)}, rounds=r8["counts"], blade_painted=int(((lab8 > 0) & (ax == 0)).sum()), blade_unpainted=int(((lab8 <= 0) & (ax == 0)).sum()))
    log("r8pc auto", R["r8pc_auto"])
    # 5. rachis absorption with the candidates' own polylines as traces
    tp, tl, ti = [], [], []
    for k, d in enumerate(cand, 1):
        q = d["poly"] / 1000.0; tp.append(q); tl.append(np.full(len(q), k)); ti.append(np.full(len(q), k))
    Z3 = dict(Z2); Z3["trace_pts"] = np.vstack(tp); Z3["trace_leaf"] = np.concatenate(tl); Z3["trace_id"] = np.concatenate(ti)
    rx = PR.Rx(Z3, dict(label=lab8)); rr = rx.run(); labx = lab8.copy(); labx[rx.T] = rr["lab"][:rx.nT]; rach = np.zeros(n, int); rach[rx.T] = rr["lab"][:rx.nT]
    R["rachis_auto"] = dict(targets=int(rx.nT), absorbed=int((rr["lab"][:rx.nT] > 0).sum()))
    # 6. boxes on the auto labels
    dd = os.path.join(out, "lb_data"); os.makedirs(dd, exist_ok=True); shutil.copy(find("r8pc_bundle.npz"), dd); shutil.copy(find("per_surfel_light.npz"), dd)
    node = np.zeros(n, bool); node[M.nid] = True; np.savez_compressed(os.path.join(dd, "labels.npz"), rows=Z["rows"], label=lab8.astype(np.int16), node=node)
    np.savez_compressed(os.path.join(dd, "labels_rachis.npz"), rows=Z["rows"], label=labx.astype(np.int16), target=np.isin(np.arange(n), rx.T))
    L = LB.LeafBox(dd); seed = np.where(L.r8 > 0, L.r8, 0); seed = np.where(L.rach_lab > 0, L.rach_lab, seed); fin, new, code, boxes, hist, D, inside, top_c = L.resolve(seed, L.targets0, iters=1)
    full = np.where(fin > 0, fin, 0); full = np.where(L.rach_lab > 0, L.rach_lab, full)
    vol = {k: float(np.prod(2 * boxes[k]["half"])) for k in L.leaves}; rng = np.random.default_rng(0); maxov = {k: 0.0 for k in L.leaves}
    for i_, ka in enumerate(L.leaves):
        for kb in L.leaves[i_ + 1:]:
            Aa, Bb = boxes[ka], boxes[kb]; U = rng.uniform(-1, 1, (20000, 3)) * Aa["half"]; pts = Aa["centre"] + U @ Aa["R"]; inter = float(LB.obb_inside(Bb, pts).mean()) * vol[ka]; sh = inter / min(vol[ka], vol[kb])
            maxov[ka] = max(maxov[ka], sh); maxov[kb] = max(maxov[kb], sh)
    # grades
    LT = np.load(find("per_surfel_light.npz")); area = LT["area_b_silhouette_mm2"]; light = LT["absorbed_ru_mm2"]; grades = {}; tot_area = tot_light = 0.0
    for k, d in enumerate(cand, 1):
        if k not in boxes: grades[str(k)] = dict(grade="low", reason="no labelled surfels after absorption"); continue
        B = boxes[k]; base = B["base"]; ok_a = d["gap"] <= A["gap_mm"]; db = float(stem_tree.query(base)[0]); ray = base[None] - np.outer(np.linspace(0, A["ray_mm"], 21), B["R"][0]); dr = float(stem_tree.query(ray)[0].min())
        ok_c = db <= A["base_mm"] or dr <= A["ray_hit_mm"]; ok_d = maxov[k] < A["overlap_max"]; hi = bool(ok_a and ok_c and ok_d); own = full == k
        grades[str(k)] = dict(grade="high" if hi else "low", a_rachis_gap_mm=round(d["gap"], 1), a_pass=bool(ok_a), b_leaflets="unknown", c_base_to_stem_mm=round(db, 1), c_ray_to_stem_mm=round(dr, 1), c_pass=bool(ok_c),
                              d_max_overlap=round(maxov[k], 3), d_pass=bool(ok_d), surfels=int(own.sum()), area_b_mm2=(round(float(area[own].sum()), 1) if hi else None), absorbed_light=(round(float(light[own].sum()), 3) if hi else None))
    lab_any = full > 0; R["plant_totals"] = dict(labelled_area_b_mm2=round(float(area[lab_any].sum()), 1), labelled_light=round(float(light[lab_any].sum()), 3), all_area_b_mm2=round(float(area.sum()), 1), all_light=round(float(light.sum()), 3))
    R["grades"] = grades; R["high_grade_leaves"] = [k for k, v in grades.items() if v["grade"] == "high"]
    # reproduction vs human leaves 1-8 (scoring only)
    hl = sorted(set(human[human > 0].tolist())); al = sorted(set(full[full > 0].tolist())); C = np.array([[int(((human == h) & (full == a_)).sum()) for a_ in al] for h in hl]); ri, ci = linear_sum_assignment(-C); rep = {}
    for i_, j_ in zip(ri, ci):
        h, a_ = hl[i_], al[j_]; inter = C[i_, j_]; uni = int(((human == h) | (full == a_)).sum()); iou = inter / uni if uni else 0.0
        rep[str(h)] = dict(auto_leaf=int(a_), iou=round(iou, 3), reproduced=bool(iou >= A["iou_min"]), auto_grade=grades.get(str(a_), {}).get("grade"))
    for h in hl:
        if str(h) not in rep: rep[str(h)] = dict(auto_leaf=None, iou=0.0, reproduced=False)
    tr = Z["trace_pts"] * 1000.0; tleaf = Z["trace_leaf"].astype(int); tt = {}
    for h in sorted(set(tleaf.tolist())):
        near = a2[cKDTree(tr[tleaf == h]).query(X[a2])[0] <= A["trace_mm"]]; lbs = seedlab[near]; lbs = lbs[lbs > 0]
        if not len(near): continue
        top, k_ = Counter(lbs.tolist()).most_common(1)[0] if len(lbs) else (0, 0); share = k_ / len(near); comp = seedlab == top; pur = float((human[comp] == h).sum() / max(comp.sum(), 1)) if top else 0.0
        tt[str(h)] = dict(near_axis2=int(len(near)), top_candidate=int(top), share=round(share, 3), candidate_purity=round(pur, 3), reproduced=bool(share >= 0.5))
    both = (human > 0) & (full > 0); mp = {int(al[j_]): int(hl[i_]) for i_, j_ in zip(ri, ci)}; agree = float(np.mean([mp.get(int(a_), -1) == int(h) for a_, h in zip(full[both], human[both])])) if both.any() else None
    R["reproduction"] = dict(per_human_leaf=rep, reproduced=int(sum(v["reproduced"] for v in rep.values())), of=len(hl), traces=tt, traces_reproduced=int(sum(v["reproduced"] for v in tt.values())), traces_total=len(tt),
                             agreement_after_matching=(round(agree, 4) if agree is not None else None), labelled_in_both=int(both.sum()))
    np.savez_compressed(os.path.join(out, "labels_auto.npz"), rows=Z["rows"], label=full.astype(np.int16), seed_component=seedlab.astype(np.int16), stage1=st1.astype(np.int16), human=human.astype(np.int16),
                        grade_high=np.array([int(k) for k in R["high_grade_leaves"]], np.int16))
    R["seconds"] = round(time.time() - t0); json.dump(R, open(os.path.join(out, "results.json"), "w"), indent=1, default=float); log("done", R["seconds"], "reproduced", R["reproduction"]["reproduced"], "/", len(hl), "high", R["high_grade_leaves"])


if __name__ == "__main__":
    main()
