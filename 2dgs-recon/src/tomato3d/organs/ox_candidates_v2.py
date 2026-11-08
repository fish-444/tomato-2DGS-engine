"""O/X candidates v2 (contract build/p1_ox_candidates_v2_20261029_01/CONTRACT.md): every v1 rachis candidate is grown independently into a compound-leaf box with session A's
`LeafBox.resolve` unchanged, then the same R3-1b gate (pool recall >= 6 of 8 at IoU >= 0.5). Counts at IoU >= 0.3 are a report column only. Runs as a Kaggle CPU kernel.
Inputs found under /kaggle/input: r8pc_bundle.npz, labels.npz, labels_rachis.npz, per_surfel_light.npz, leafbox.py, ox_candidates.py, pc_absorb.py, pc_rachis.py, v1 candidates.npz + candidates.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import glob
    def find(name):
        for g in (f"/kaggle/input/**/{name}", os.path.join(HERE, "**", name)):
            hit = glob.glob(g, recursive=True)
            if hit:
                return hit[0]
        raise SystemExit(f"{name} not found")
    import importlib.util
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
    out = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(HERE, "ox2_out"); os.makedirs(out, exist_ok=True); t0 = time.time()
    for p_ in ("tomato3d", "tomato3d.organs"):
        sys.modules.setdefault(p_, types.ModuleType(p_))
    lo = types.ModuleType("tomato3d.organs.light_organs"); lo.ROOT = Path(out); sys.modules["tomato3d.organs.light_organs"] = lo
    PA = load("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load("tomato3d.organs.pc_rachis", find("pc_rachis.py"))
    OX = load("ox_candidates", find("ox_candidates.py")); LB = load("leafbox", find("leafbox.py")); log = OX.log
    data = os.path.dirname(find("r8pc_bundle.npz")); L = LB.LeafBox(data); n = L.n; X = L.X; ax = L.ax; area = L.area
    Z = dict(np.load(find("r8pc_bundle.npz")))
    H8 = np.load(find("labels.npz"))["label"].astype(int); HX = np.load(find("labels_rachis.npz")); hx = np.where(HX["target"] & (HX["label"] > 0), HX["label"], 0); human = np.where(hx > 0, hx, np.where(H8 > 0, H8, 0))
    v1 = np.load(find("candidates.npz")); meta1 = json.load(open(find("candidates.json"))); assert np.array_equal(v1["rows"], Z["rows"]), "v1 pool row order differs"
    so, ro = v1["offsets"], v1["rachis_offsets"]; sets1 = [v1["surfels"][so[i]:so[i + 1]] for i in range(len(meta1))]; rach = [v1["rachis"][ro[i]:ro[i + 1]] for i in range(len(meta1))]
    # ---- grow each candidate independently (session A unchanged: one leaf, the candidate's rachis as the only seed)
    r8_keep, rx_keep, leaves_keep = L.r8.copy(), L.rach_lab.copy(), list(L.leaves); grown, info = [], []
    for i, ra in enumerate(rach):
        L.r8 = np.zeros(n, int); L.rach_lab = np.zeros(n, int); L.rach_lab[ra] = 1; L.leaves = [1]; seed = np.zeros(n, int); seed[ra] = 1
        targets = np.setdiff1d(L.nid, ra)
        try:
            lab, new, code, boxes, hist, D, inside, top_c = L.resolve(seed, targets, verbose=False)
            g = np.union1d(np.nonzero(lab == 1)[0], ra); B = boxes[1]
            info.append(dict(iterations=len(hist), last=hist[-1], half_mm=[round(float(v), 1) for v in B["half"]], centre_mm=[round(float(v), 2) for v in B["centre"]], R=np.round(B["R"], 4).tolist(), flags=B["flags"]))
        except Exception as e:                                                       # a failing candidate is reported, not hidden
            g = np.asarray(ra); info.append(dict(error=f"{type(e).__name__}: {e}"))
        grown.append(g)
        if i % 5 == 0:
            log("grown", i, len(g), f"{time.time() - t0:.0f}s")
    L.r8, L.rach_lab, L.leaves = r8_keep, rx_keep, leaves_keep
    blade_area = np.array([float(area[g[ax[g] == 0]].sum()) for g in grown]); med = float(np.median(blade_area)); keep = []; dropped = []
    for i, a in enumerate(blade_area):
        if a > 3 * med or a < 0.25 * med:
            dropped.append(dict(id=i, slot=meta1[i]["slot"], alt=meta1[i]["alt"], blade_area_mm2=round(a, 1), reason=("> 3 x median" if a > 3 * med else "< 0.25 x median")))
        else:
            keep.append(i)
    sets2 = {i: set(grown[i].tolist()) for i in keep}; s1 = [set(s.tolist()) for s in sets1]
    # ---- scoring (human leaves only here)
    per = {}; hl = sorted(set(human[human > 0].tolist()))
    for h in hl:
        S = set(np.nonzero(human == h)[0].tolist()); i1 = [OX.iou(s, S) for s in s1]; i2 = {i: OX.iou(sets2[i], S) for i in keep}; b2 = max(i2, key=i2.get) if i2 else -1
        per[str(h)] = dict(human_surfels=len(S), v1_best_iou=round(max(i1), 3), v1_n_ge_0_5=int(sum(v >= 0.5 for v in i1)), v1_n_ge_0_3=int(sum(v >= 0.3 for v in i1)),
                           v2_best_iou=round(i2[b2], 3) if b2 >= 0 else 0.0, v2_best_candidate=int(b2), v2_n_ge_0_5=int(sum(v >= 0.5 for v in i2.values())), v2_n_ge_0_3=int(sum(v >= 0.3 for v in i2.values())))
    rec5 = sum(1 for v in per.values() if v["v2_best_iou"] >= 0.5); rec3 = sum(1 for v in per.values() if v["v2_best_iou"] >= 0.3); gate = bool(rec5 >= OX.P["gate_recall"])
    R = dict(labels=["auto_candidates_box_grown", "reference_only_scoring", "conditional_on_unvalidated_scale"], v1_candidates=len(meta1), grown=len(grown), kept_after_size_check=len(keep), median_grown_blade_area_mm2=round(med, 1),
             dropped=dropped, per_human_leaf=per, pool_recall_iou_0_5=rec5, pool_recall_iou_0_3_report_only=rec3, v1_pool_recall_iou_0_5=sum(1 for v in per.values() if v["v1_best_iou"] >= 0.5),
             v1_pool_recall_iou_0_3_report_only=sum(1 for v in per.values() if v["v1_best_iou"] >= 0.3), gate_min=OX.P["gate_recall"], gate_pass=gate, ox_route="pass" if gate else "failed_gate", grow_info=info)
    log("pool recall v2", rec5, "(0.3:", rec3, ") gate", gate)
    # ---- perfect-user upper bound (0.5 = the declared choice; 0.3 = report only)
    blade = ax == 0; R["upper_bound"] = {}
    for thr in (0.5, 0.3):
        best = {h: v["v2_best_candidate"] for h, v in per.items() if v["v2_best_iou"] >= thr}; chosen = []
        for h in sorted(best, key=lambda h: -per[h]["v2_best_iou"]):
            c = best[h]
            if c not in chosen and all(OX.iou(sets2[c], sets2[o]) <= OX.P["alt_iou"] for o in chosen):
                chosen.append(c)
        order = sorted(chosen, key=lambda c: float(X[rach[c]][:, 2].min())); st1 = np.zeros(n, int); polys = {}; covered = np.zeros(n, bool)
        for k, c in enumerate(order, 1):
            g = grown[c]; m = g[ax[g] == 0]; st1[m] = np.where(st1[m] == 0, k, st1[m]); polys[k] = OX.polyline(X[rach[c]]); covered[g] = True
        res = dict(o_candidates=len(order), coverage_blade_area=round(float(area[blade & covered].sum() / area[blade].sum()), 4), report_only=(thr != 0.5))
        if len(order) >= 2:
            from scipy.optimize import linear_sum_assignment
            lab8, labx, ch = OX.run_chain(Z, st1, polys, PA, PR, ax); full = np.where(labx > 0, labx, 0); al = sorted(set(full[full > 0].tolist()))
            C = np.array([[int(((human == h) & (full == a_)).sum()) for a_ in al] for h in hl]); ri, ci = linear_sum_assignment(-C); mp = {int(al[j]): int(hl[i]) for i, j in zip(ri, ci)}; both = (human > 0) & (full > 0)
            agree = float(np.mean([mp.get(int(a_), -1) == int(h) for a_, h in zip(full[both], human[both])])) if both.any() else None
            rep = {str(hl[i]): round(C[i, j] / max(int(((human == hl[i]) | (full == al[j])).sum()), 1), 3) for i, j in zip(ri, ci)}
            res.update(chain=ch, agreement_after_matching=(round(agree, 4) if agree is not None else None), iou_per_human_leaf=rep, leaves_iou_ge_0_5=int(sum(v >= 0.5 for v in rep.values())))
        R["upper_bound"][f"iou_{thr}"] = res; log("upper bound", thr, {k: v for k, v in res.items() if k != "chain"})
    off = np.cumsum([0] + [len(g) for g in grown]); np.savez_compressed(os.path.join(out, "candidates_v2.npz"), surfels=np.concatenate(grown), offsets=off, rows=Z["rows"])
    json.dump([dict(meta1[i], grown_surfels=int(len(grown[i])), grown_blade_area_mm2=round(float(blade_area[i]), 1), kept=(i in keep), box=info[i]) for i in range(len(meta1))], open(os.path.join(out, "candidates_v2.json"), "w"))
    R["seconds"] = round(time.time() - t0); json.dump(R, open(os.path.join(out, "results_v2.json"), "w"), indent=1, default=float); log("done", R["seconds"])


if __name__ == "__main__":
    main()
