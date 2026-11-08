"""P1 fully automatic seeds with leaflet-size priors from the 씨앗 photos (no user clicks). Kaggle CPU kernel: part L (leaflet library -> priors), part X/E (R3-1b auto seeds,
R8_pc + rachis absorption + prior cap, grades, reproduction vs the human leaves). Imports pc_absorb.py, pc_rachis.py, leafbox.py from the dataset.
Contract: build/p1_autoseed_20261029_01/CONTRACT.md (+ amendments).
"""

from __future__ import annotations

import csv
import glob
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

L_ = dict(half=2, hough_frac=0.35, per_lo=5, per_hi=60, top_rows=12, row_score_min=6.0, harm_frac=0.25, tilt=0.03, band_pad=0.15, hue=(7.5, 50.0), sat=(0.12, 0.15, 0.18), val_max=0.92,
          skin_hue=12.5, skin_val=0.45, tool_hue=(95, 125), open_px=3, min_mm2=15.0, min_px=500, curled=0.45, solidity=0.60, piece_area=1.5, thr_agree=0.05, gate_n=30, gate_ratio=3.0, cap22=22.5)
X_ = dict(edge_mm=3.0, min_comp=50, merge_mm=6.0, merge_deg=25.0, split_len=1.6, split_branch=0.3, patch_mm=5.0, min_seeds=3, max_seeds=14, scale_unc=0.05, patch_min=30, leaflets=(5, 7),
          dim_share=0.8, base_mm=10.0, ray_mm=20.0, ray_hit_mm=5.0, overlap_max=0.5, iou_min=0.5, trace_mm=3.0, purity_min=0.5)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find(name, many=False):
    hits = []
    for g in (f"/kaggle/input/**/{name}", os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", name)):
        hits += glob.glob(g, recursive=True)
    if many: return sorted(set(hits))
    if not hits: raise SystemExit(f"{name} not found")
    return hits[0]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


# ---------------- part L: leaflet library ----------------
def detect_scale(gray):
    import cv2
    H, W = gray.shape; ed = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150); lines = cv2.HoughLinesP(ed, 1, np.pi / 360, 200, minLineLength=int(L_["hough_frac"] * W), maxLineGap=10)
    if lines is None: return dict(status="no_scale", reason="no long line")
    seg = lines[:, 0, :].astype(float); ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180; ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    hist = np.bincount(np.round(ang).astype(int) % 180, weights=ln, minlength=180); a0 = float(np.argmax(hist)); M = cv2.getRotationMatrix2D((W / 2, H / 2), a0, 1.0)
    rot = cv2.warpAffine(gray, M, (W, H), borderValue=255).astype(float); per = np.arange(L_["per_lo"], L_["per_hi"] + 1)
    def row_period(sig):
        s = sig - sig.mean(); F = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2; fr = np.fft.rfftfreq(len(s)); band = (fr >= 1 / L_["per_hi"]) & (fr <= 1 / L_["per_lo"])
        if not band.any(): return 0.0, None
        pw = F[band]; fb = fr[band]; score = float(pw.max() / max(np.median(pw), 1e-9)); strong = pw >= L_["harm_frac"] * pw.max()
        cand = np.nonzero(strong)[0]; k = cand[np.argmax(fb[cand])]                                  # highest strong frequency = smallest strong period (1 mm ticks)
        if 0 < k < len(pw) - 1:
            y0, y1, y2 = np.log(pw[k - 1] + 1e-9), np.log(pw[k] + 1e-9), np.log(pw[k + 1] + 1e-9); d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
            f = fb[k] + d * (fb[1] - fb[0])
        else: f = fb[k]
        return score, 1.0 / f
    sc = []; pr = []
    for y in range(0, H, 2):
        s, p = row_period(rot[y]); sc.append(s); pr.append(p)
    sc = np.array(sc); order = np.argsort(-sc)[:L_["top_rows"]]
    if np.median(sc[order]) < L_["row_score_min"]: return dict(status="no_scale", reason="no strong periodic band", angle=a0)
    rows = np.array(order) * 2; P = np.array([pr[i] for i in order]); per_px = float(np.median(P)); hl = []; hr = []
    for y in rows:
        a, b = rot[y, :W // 2], rot[y, W // 2:]; _, pa = row_period(a); _, pb = row_period(b)
        if pa and pb: hl.append(pa); hr.append(pb)
    tl = abs(np.median(hl) - np.median(hr)) / per_px if hl else 1.0; span = (rows.min(), rows.max()); pad = max(20, int(L_["band_pad"] * (span[1] - span[0] + 40)))
    return dict(status="tilted" if tl > L_["tilt"] else "ok", angle=a0, period_px=per_px, mm_per_px=1.0 / per_px, tilt_rel=float(tl), band=(int(span[0] - pad), int(span[1] + pad)), M=M.tolist())


def segment(rgb_rot, band, s_thr, mm_px):
    import cv2
    hsv = cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2HSV).astype(float); h, s, v = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255; R, G, B = [rgb_rot[..., i].astype(float) for i in range(3)]
    leaf = (h >= L_["hue"][0]) & (h <= L_["hue"][1]) & (s > s_thr) & (v < L_["val_max"]); skin = (h <= L_["skin_hue"]) & (R > G) & (G > B) & (v > L_["skin_val"]); tool = (h >= L_["tool_hue"][0]) & (h <= L_["tool_hue"][1]) & (s > 0.2)
    m = leaf & ~skin & ~tool
    if band is not None: m[max(0, band[0]):max(0, band[1]), :] = False
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((L_["open_px"], L_["open_px"]), np.uint8)); n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8); H, W = m.shape; objs = []
    min_px = (L_["min_mm2"] / mm_px ** 2) if mm_px else L_["min_px"]
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        if a < min_px or x == 0 or y == 0 or x + w >= W or y + hh >= H: continue
        ys, xs = np.nonzero(lab[y:y + hh, x:x + w] == i); P = np.c_[xs + x, ys + y].astype(float); c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); t = (P - c) @ vt[0]; q = (P - c) @ vt[1]
        Lp, Wp = float(np.ptp(t)) + 1, float(np.ptp(q)) + 1; cnt, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); hull = cv2.convexHull(max(cnt, key=cv2.contourArea)); ha = max(cv2.contourArea(hull), 1)
        objs.append(dict(cx=float(c[0]), cy=float(c[1]), area_px=int(a), L_px=Lp, W_px=Wp, solidity=float(a / ha), contour=max(cnt, key=cv2.contourArea)))
    return objs


def part_L(out):
    import cv2
    jpgs = find("IMG_*.jpg", many=True); D = os.path.join(out, "library"); os.makedirs(os.path.join(D, "overlays"), exist_ok=True); rows = []; insp = []
    for f in jpgs:
        name = os.path.basename(f)[:-4]; im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (im.shape[1] // L_["half"], im.shape[0] // L_["half"]), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY); sc = detect_scale(gray); H, W = gray.shape
        if "M" in sc: rot = cv2.warpAffine(im, np.array(sc["M"]), (W, H), borderValue=(255, 255, 255))
        else: rot = im
        mm = sc.get("mm_per_px") if sc["status"] == "ok" else None; band = sc.get("band")
        objs = {s: segment(rot, band, s, mm) for s in L_["sat"]}; base = objs[0.15]; ratios = []
        for s in (0.12, 0.18):
            if not objs[s] or not base: continue
            T = cKDTree(np.array([[o["cx"], o["cy"]] for o in objs[s]]))
            for o in base:
                d, j = T.query([o["cx"], o["cy"]])
                if d < 20: ratios.append(objs[s][j]["area_px"] / o["area_px"])
        sens = bool(ratios and np.median(np.abs(np.array(ratios) - 1)) > L_["thr_agree"]) or (not ratios and len(base) > 0)
        med_a = np.median([o["area_px"] for o in base]) if base else 0; ov = rot.copy()
        if band: cv2.rectangle(ov, (0, band[0]), (W - 1, band[1]), (255, 0, 0), 3)
        for k, o in enumerate(base):
            curled = o["area_px"] / (o["L_px"] * o["W_px"]) < L_["curled"]; piece = o["solidity"] < L_["solidity"] and o["area_px"] > L_["piece_area"] * med_a
            f_ = mm if mm else None
            rows.append(dict(photo=name, object=k, scale=sc["status"], mm_per_px=(round(mm, 5) if mm else None), length_mm=(round(o["L_px"] * f_, 2) if f_ else None), width_mm=(round(o["W_px"] * f_, 2) if f_ else None),
                             lw=round(o["L_px"] / o["W_px"], 3), area_mm2=(round(o["area_px"] * f_ * f_, 1) if f_ else None), length_unc_mm=(round(np.sqrt(2) * 0.5, 2) if f_ else None),
                             area_unc_mm2=(round(cv2.arcLength(o["contour"], True) * f_ * f_, 1) if f_ else None), petiolule_mm="not_measured", curled=bool(curled), piece_with_rachis=bool(piece), threshold_sensitive=sens))
            cv2.drawContours(ov, [o["contour"]], -1, (255, 0, 255) if piece else ((255, 160, 0) if curled else (0, 200, 0)), 3); cv2.putText(ov, str(k), (int(o["cx"]), int(o["cy"])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.imwrite(os.path.join(D, "overlays", name + ".jpg"), cv2.cvtColor(cv2.resize(ov, (1000, 1000)), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        insp.append(dict(photo=name, scale=sc["status"], reason=sc.get("reason"), angle_deg=sc.get("angle"), mm_per_px=sc.get("mm_per_px"), tilt_rel=sc.get("tilt_rel"), objects=len(base),
                         pieces=sum(r["piece_with_rachis"] for r in rows if r["photo"] == name), curled=sum(r["curled"] for r in rows if r["photo"] == name), threshold_sensitive=sens))
        log(name, sc["status"], round(sc.get("mm_per_px") or 0, 4), "objects", len(base))
    with open(os.path.join(D, "leaflet_measurements.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    use = [r for r in rows if r["scale"] == "ok" and not r["piece_with_rachis"] and not r["curled"]]; pri = dict(class_="rank_unknown", n=len(use))
    for k in ("length_mm", "width_mm", "lw", "area_mm2"):
        a = np.array([r[k] for r in use], float)
        pri[k] = dict(median=round(float(np.median(a)), 3), p5=round(float(np.percentile(a, 5)), 3), p95=round(float(np.percentile(a, 95)), 3)) if len(a) else None
    lw_all = np.array([r["lw"] for r in rows if not r["piece_with_rachis"] and not r["curled"]], float)
    pri["lw_all_including_no_scale"] = dict(n=int(len(lw_all)), median=round(float(np.median(lw_all)), 3)) if len(lw_all) else None
    ok = bool(pri["n"] >= L_["gate_n"] and pri["length_mm"] and pri["length_mm"]["p95"] <= L_["gate_ratio"] * pri["length_mm"]["median"] and pri["length_mm"]["median"] <= L_["gate_ratio"] * pri["length_mm"]["p5"])
    pri["gate"] = "priors_used" if ok else "weak_prior"; pri["library_comparison"] = dict(dissected_lw="1.578 +/- 0.259", measured_max_leaflet_length_mm="18.6 +/- 3.1", this_lw_median=(pri["lw"] or {}).get("median"),
                                                                                       this_length_p95=(pri["length_mm"] or {}).get("p95"))
    pri["photos"] = dict(total=len(insp), ok=sum(i["scale"] == "ok" for i in insp), tilted=sum(i["scale"] == "tilted" for i in insp), no_scale=sum(i["scale"] == "no_scale" for i in insp), threshold_sensitive=sum(i["threshold_sensitive"] for i in insp))
    json.dump(pri, open(os.path.join(D, "leaflet_priors.json"), "w"), indent=1, default=float); json.dump(insp, open(os.path.join(D, "inspection_auto.json"), "w"), indent=1, default=float)
    with open(os.path.join(D, "leaflet_priors.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["measure", "n", "median", "p5", "p95", "class"])
        for k in ("length_mm", "width_mm", "lw", "area_mm2"):
            if pri[k]: wr.writerow([k, pri["n"], pri[k]["median"], pri[k]["p5"], pri[k]["p95"], "rank_unknown"])
    log("priors", json.dumps({k: pri[k] for k in ("n", "length_mm", "lw", "area_mm2", "gate", "photos")}, default=float)); return pri


# ---------------- part X/E: R3-1b ----------------
def comps(X, idx, r):
    E = cKDTree(X[idx]).query_pairs(r, output_type="ndarray"); m = len(idx); w = np.linalg.norm(X[idx][E[:, 0]] - X[idx][E[:, 1]], axis=1) + 1e-6 if len(E) else np.zeros(0)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(m, m)).tocsr(); return connected_components(G, directed=False)[1], G


def axis_info(P):
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); u = vt[0]; t = (P - c) @ u; return c, u, t


def part_XE(out, pri):
    for p_ in ("tomato3d", "tomato3d.organs"): sys.modules.setdefault(p_, types.ModuleType(p_))
    lo = types.ModuleType("tomato3d.organs.light_organs"); lo.ROOT = Path(out); sys.modules["tomato3d.organs.light_organs"] = lo
    PA = load_module("tomato3d.organs.pc_absorb", find("pc_absorb.py")); PR = load_module("tomato3d.organs.pc_rachis", find("pc_rachis.py")); LB = load_module("leafbox", find("leafbox.py"))
    D = os.path.join(out, "R3-1b"); os.makedirs(D, exist_ok=True); Z = dict(np.load(find("r8pc_bundle.npz"))); X = Z["xyz"] * 1000.0; ax = Z["axis_kind"].astype(int); n = len(X); rows = Z["rows"]
    H8 = np.load(find("labels.npz"))["label"].astype(int); HX = np.load(find("labels_rachis.npz")); human = np.where(HX["target"] & (HX["label"] > 0), HX["label"], np.where(H8 > 0, H8, 0))
    stem = np.nonzero(ax == 1)[0]; stem_tree = cKDTree(X[stem]); sc_, _ = comps(X, stem, X_["edge_mm"]); ext = {c: float(np.ptp(X[stem[sc_ == c], 2])) for c in np.unique(sc_)}; smain = max(ext, key=ext.get)
    R = dict(individual="R3-1b", labels=["seed=auto_axis", "designated_not_adopted", "conditional_on_unvalidated_scale"], params=X_, axis_version="r8pc_bundle axis_kind (R3-1b classifier output)",
             stem=dict(components=int(sc_.max() + 1), main_vertical_extent_mm=round(ext[smain], 1)))
    np.savez_compressed(os.path.join(D, "axis_labels.npz"), rows=rows, axis_kind=ax.astype(np.int8))
    a2 = np.nonzero(ax == 2)[0]; cc, G2 = comps(X, a2, X_["edge_mm"]); cnt = Counter(cc.tolist()); cand = [a2[cc == c] for c, k in cnt.items() if k >= X_["min_comp"]]; R["components_raw"] = len(cand)
    # merge: nearest ends < 6 mm and directions < 25 deg
    def ends(P):
        c, u, t = axis_info(P); return c + u * t.min(), c + u * t.max(), u
    merged = True; nm = 0
    while merged:
        merged = False; info = [ends(X[m]) for m in cand]
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                ei, ej = info[i], info[j]; dmin = min(np.linalg.norm(a - b) for a in ei[:2] for b in ej[:2]); ang = np.degrees(np.arccos(min(1.0, abs(float(ei[2] @ ej[2])))))
                if dmin < X_["merge_mm"] and ang < X_["merge_deg"]: cand[i] = np.concatenate([cand[i], cand[j]]); cand.pop(j); merged = True; nm += 1; break
            if merged: break
    # split: long components with a Y-junction
    lens = [float(np.ptp(axis_info(X[m])[2])) for m in cand]; medl = float(np.median(lens)) if lens else 0.0; ns = 0; new = []
    for m, l in zip(cand, lens):
        if l <= X_["split_len"] * medl: new.append(m); continue
        lab_, G = comps(X, m, X_["edge_mm"]); d0 = dijkstra(G, directed=False, indices=0); a = int(np.argmax(np.where(np.isfinite(d0), d0, -1))); da = dijkstra(G, directed=False, indices=a, return_predecessors=True)
        dist_a, pred = da; b = int(np.argmax(np.where(np.isfinite(dist_a), dist_a, -1))); path = [b]
        while path[-1] != a and pred[path[-1]] >= 0: path.append(int(pred[path[-1]]))
        main_len = float(dist_a[b]); dpath = dijkstra(G, directed=False, indices=path, min_only=True); far = int(np.argmax(np.where(np.isfinite(dpath), dpath, -1)))
        if main_len > 0 and dpath[far] >= X_["split_branch"] * main_len:
            branch = np.isfinite(dpath) & (dpath > 0.5 * dpath[far]) & (np.linalg.norm(X[m] - X[m][far], axis=1) < np.linalg.norm(X[m] - X[m][path[len(path) // 2]], axis=1))
            if branch.sum() >= X_["min_comp"] and (~branch).sum() >= X_["min_comp"]: new += [m[~branch], m[branch]]; ns += 1; continue
        new.append(m)
    cand = sorted(new, key=lambda m: float(X[m][np.argmin(stem_tree.query(X[m])[0])][2])); K = len(cand); seedlab = np.zeros(n, int)
    for k, m in enumerate(cand, 1): seedlab[m] = k
    R["seeds"] = dict(count=K, merged=nm, split=ns, sizes=[int(len(m)) for m in cand], implausible=bool(K < X_["min_seeds"] or K > X_["max_seeds"]))
    log("seeds", R["seeds"])
    # leaflet seeds: 3D-first patches touching exactly one candidate within 5 mm
    PF = np.load(find("patches_filled.npz")); pos = {int(r): i for i, r in enumerate(rows)}; pid = np.full(n, -1, int); idxb = np.array([pos[int(r)] for r in PF["rows"]]); pid[idxb] = PF["patch_filled"]
    sm = np.nonzero(seedlab > 0)[0]; ts = cKDTree(X[sm]); touch = defaultdict(set)
    for i, nb in zip(np.nonzero(pid >= 0)[0], ts.query_ball_point(X[pid >= 0], X_["patch_mm"])):
        if nb: touch[int(pid[i])].update(seedlab[sm[nb]].tolist())
    st1 = np.zeros(n, int); pl = {}
    for p, ks in touch.items():
        if len(ks) == 1: k = ks.pop(); st1[pid == p] = k; pl[p] = k
    R["leaflet_seed_patches"] = len(pl); R["stage1_blade"] = int((st1 > 0).sum())
    json.dump(dict(individual="R3-1b", source="auto_axis", seeds=[dict(leaf_id=k, rachis_surfel_rows=[int(rows[i]) for i in m[:5000]], n_rachis_surfels=int(len(m)), leaflet_patches=sorted(p for p, kk in pl.items() if kk == k))
                                                                   for k, m in enumerate(cand, 1)], implausible=R["seeds"]["implausible"]), open(os.path.join(D, "seeds_auto.json"), "w"))
    # R8_pc unchanged with the auto stage 1
    Z2 = dict(Z); Z2["stage1"] = st1.astype(np.int16); M = PA.Model(Z2); M.norm_unary(); labn = np.nonzero(M.st1n > 0)[0]
    d_, jj, v_ = PA.knn(M.tree_n, M.Xn[labn], 32, PA.P["band_mm"]); jj = np.where(v_, jj, 0); band = labn[(v_ & (M.st1n[jj] > 0) & (M.st1n[jj] != M.st1n[labn][:, None])).any(1)]
    cb = PA.P["cal"]; PA.P["cal"] = tuple(sorted(set(M.st1n[M.st1n > 0].tolist())))[0::2]; auc = M.colour_auc(band) if len(band) else {k: dict(auc=0.5) for k in ("C1", "C2", "C4")}; PA.P["cal"] = cb
    wcol = {k: (PA.WC[k] if (k == "C3" or auc[k]["auc"] >= PA.P["auc_min"]) else 0.0) for k in PA.WC}; cr = (M.st1n[M.ei] > 0) & (M.st1n[M.ej] > 0) & (M.st1n[M.ei] != M.st1n[M.ej]); se = (M.st1n[M.ei] > 0) & (M.st1n[M.ei] == M.st1n[M.ej])
    xa, xb = M.edge_raw["C3"][cr], M.edge_raw["C3"][se]; auc3 = float(np.searchsorted(np.sort(xb), xa).mean() / len(xb)) if len(xa) and len(xb) else 0.5; wcol["C3"] = PA.WC["C3"] if auc3 >= PA.P["auc_min"] else 0.0
    g, c_ = PA.SETS["B"]; r8 = M.run(g, c_, wcol, M.st1n.copy()); lab = np.zeros(n, int); lab[M.nid] = r8["lab"]; p1 = np.zeros(n); p2 = np.zeros(n); t1 = np.zeros(n, int); t2 = np.zeros(n, int)
    p1[M.nid] = r8["p1"]; p2[M.nid] = r8["p2"]; t1[M.nid] = r8["top"][:, 0]; t2[M.nid] = r8["top"][:, 1]
    # prior cap per leaflet (3D-first patch)
    use = pri["gate"] == "priors_used"; capL = (pri["length_mm"]["p95"] * (1 + X_["scale_unc"])) if use else L_["cap22"]; capLW = pri["lw"]["p95"] if use else None; p5A = pri["area_mm2"]["p5"] if use else None
    LT = np.load(find("per_surfel_light.npz")); area = LT["area_b_silhouette_mm2"]; light = LT["absorbed_ru_mm2"]; blocked = []; small = []; reason = np.zeros(n, int)
    for p in np.unique(pid[pid >= 0]):
        mem = np.nonzero(pid == p)[0]; ls = lab[mem]; ls = ls[ls > 0]
        if not len(ls): continue
        k = Counter(ls.tolist()).most_common(1)[0][0]; mk = mem[lab[mem] == k]
        if len(mk) < 3: continue
        c, u, t = axis_info(X[mk]); q = (X[mk] - c) @ np.cross(u, np.linalg.svd(X[mk] - c, full_matrices=False)[2][2]); Lr = float(np.percentile(t, 98) - np.percentile(t, 2)); Wr = float(np.percentile(q, 98) - np.percentile(q, 2)) or 1e-6
        over = Lr > capL or (capLW is not None and Lr / Wr > capLW)
        if over:
            ab = mk[st1[mk] != k]
            if len(ab): lab[ab] = 0; reason[ab] = 1; blocked.append(dict(patch=int(p), leaf=int(k), length_mm=round(Lr, 1), lw=round(Lr / Wr, 2), reverted=int(len(ab)), reason="exceeds_leaflet_prior"))
        if p5A is not None and float(area[mk].sum()) < p5A: small.append(int(p))
    with open(os.path.join(D, "blocked_absorptions.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["patch", "leaf", "length_mm", "lw", "reverted", "reason"]); [wr.writerow([b["patch"], b["leaf"], b["length_mm"], b["lw"], b["reverted"], b["reason"]]) for b in blocked]
    R["cap"] = dict(priors=("priors_used" if use else "weak_prior (22.5 mm cap only)"), length_cap_mm=round(capL, 2), lw_cap=capLW, blocked_patches=len(blocked), reverted_surfels=int(sum(b["reverted"] for b in blocked)), small_leaflet_flags=len(small))
    # rachis absorption with the candidates' polylines as traces
    tp, tl, ti = [], [], []
    for k, m in enumerate(cand, 1):
        P_ = X[m]; c, u, t = axis_info(P_); bins = np.floor((t - t.min()) / 2.0).astype(int); poly = np.array([np.median(P_[bins == b], 0) for b in np.unique(bins)]) / 1000.0
        tp.append(poly); tl.append(np.full(len(poly), k)); ti.append(np.full(len(poly), k))
    Z3 = dict(Z2); Z3["trace_pts"] = np.vstack(tp); Z3["trace_leaf"] = np.concatenate(tl); Z3["trace_id"] = np.concatenate(ti); rx = PR.Rx(Z3, dict(label=lab)); rr = rx.run(); rach = np.zeros(n, int); rach[rx.T] = rr["lab"][:rx.nT]
    full = np.where(rach > 0, rach, lab)
    # boxes (one EM pass) for the overlap grade
    dd = os.path.join(out, "lb_data"); os.makedirs(dd, exist_ok=True); shutil.copy(find("r8pc_bundle.npz"), dd); shutil.copy(find("per_surfel_light.npz"), dd); node = np.zeros(n, bool); node[M.nid] = True
    np.savez_compressed(os.path.join(dd, "labels.npz"), rows=rows, label=lab.astype(np.int16), node=node); np.savez_compressed(os.path.join(dd, "labels_rachis.npz"), rows=rows, label=rach.astype(np.int16), target=np.isin(np.arange(n), rx.T))
    L = LB.LeafBox(dd); seed = np.where(L.r8 > 0, L.r8, 0); seed = np.where(L.rach_lab > 0, L.rach_lab, seed); fin, newb, code, boxes, hist, Dm, ins, top_c = L.resolve(seed, L.targets0, iters=1)
    vol = {k: float(np.prod(2 * boxes[k]["half"])) for k in L.leaves}; rng = np.random.default_rng(0); maxov = {k: 0.0 for k in L.leaves}
    for i_, ka in enumerate(L.leaves):
        for kb in L.leaves[i_ + 1:]:
            Aa, Bb = boxes[ka], boxes[kb]; U = rng.uniform(-1, 1, (20000, 3)) * Aa["half"]; pts = Aa["centre"] + U @ Aa["R"]; sh = float(LB.obb_inside(Bb, pts).mean()) * vol[ka] / min(vol[ka], vol[kb]); maxov[ka] = max(maxov[ka], sh); maxov[kb] = max(maxov[kb], sh)
    # grades
    contested = (lab <= 0) & node & (ax == 0); grades = []; plab = {}
    for p in np.unique(pid[pid >= 0]):
        ls = full[pid == p]; ls = ls[ls > 0]
        if len(ls) >= X_["patch_min"]: plab[int(p)] = Counter(ls.tolist()).most_common(1)[0][0]
    for k, m in enumerate(cand, 1):
        own = full == k; nleaf = sum(1 for v in plab.values() if v == k); dims = []
        for p, v in plab.items():
            if v != k: continue
            mk = np.nonzero((pid == p) & (full == k))[0]
            if len(mk) < 3: continue
            c, u, t = axis_info(X[mk]); dims.append(float(np.percentile(t, 98) - np.percentile(t, 2)))
        lo_, hi_ = (pri["length_mm"]["p5"], pri["length_mm"]["p95"] * (1 + X_["scale_unc"])) if use else (0.0, L_["cap22"]); share_in = float(np.mean([lo_ <= d <= hi_ for d in dims])) if dims else 0.0
        if k in boxes:
            base = boxes[k]["base"]; db = float(stem_tree.query(base)[0]); ray = base[None] - np.outer(np.linspace(0, X_["ray_mm"], 21), boxes[k]["R"][0]); dr = float(stem_tree.query(ray)[0].min()); ovv = maxov[k]
        else: db = dr = ovv = float("inf")
        checks = dict(one_rachis=True, leaflets_5_7=bool(X_["leaflets"][0] <= nleaf <= X_["leaflets"][1]), dims_inside=bool(share_in >= X_["dim_share"]), base_to_stem=bool(db <= X_["base_mm"] or dr <= X_["ray_hit_mm"]), overlap_low=bool(ovv < X_["overlap_max"]))
        fails = sum(not v for v in checks.values()); grade = "C" if (R["seeds"]["implausible"] or fails >= 2) else ("B" if fails == 1 else "A")
        top1 = contested & (t1 == k); top2 = contested & ((t1 == k) | (t2 == k))
        rng_a = [round(float(area[own].sum()), 1), round(float(area[own | top1].sum()), 1), round(float(area[own | top2].sum()), 1)]; rng_l = [round(float(light[own].sum()), 3), round(float(light[own | top1].sum()), 3), round(float(light[own | top2].sum()), 3)]
        grades.append(dict(leaf=k, grade=grade, **checks, leaflets=nleaf, dims_share_inside=round(share_in, 2), base_to_stem_mm=round(db, 1), ray_to_stem_mm=round(dr, 1), max_overlap=round(ovv, 3), surfels=int(own.sum()),
                           area_b_mm2_range=(rng_a if grade in "AB" else None), light_range=(rng_l if grade in "AB" else None)))
    with open(os.path.join(D, "grades.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(grades[0])); wr.writeheader(); wr.writerows(grades)
    R["grades"] = grades; R["grade_counts"] = dict(Counter(g_["grade"] for g_ in grades)); lab_any = full > 0
    R["plant_totals"] = dict(area_b_labelled=round(float(area[lab_any].sum()), 1), light_labelled=round(float(light[lab_any].sum()), 3), area_b_all=round(float(area.sum()), 1), light_all=round(float(light.sum()), 3))
    # reproduction vs human leaves (scoring only)
    hl = sorted(set(human[human > 0].tolist())); al = sorted(set(full[full > 0].tolist())); C = np.array([[int(((human == h) & (full == a_)).sum()) for a_ in al] for h in hl]); ri, ci = linear_sum_assignment(-C); rep = {}
    gmap = {g_["leaf"]: g_["grade"] for g_ in grades}
    for i_, j_ in zip(ri, ci):
        h, a_ = hl[i_], al[j_]; uni = int(((human == h) | (full == a_)).sum()); iou = C[i_, j_] / uni if uni else 0.0; aw = float(area[(human == h) & (full == a_)].sum() / max(area[human == h].sum(), 1e-9))
        rep[str(h)] = dict(auto_leaf=int(a_), iou=round(iou, 3), area_share_of_human_leaf=round(aw, 3), reproduced=bool(iou >= X_["iou_min"]), auto_grade=gmap.get(int(a_)))
    for h in hl: rep.setdefault(str(h), dict(auto_leaf=None, iou=0.0, reproduced=False))
    tr = Z["trace_pts"] * 1000.0; tleaf = Z["trace_leaf"].astype(int); tt = {}
    for h in sorted(set(tleaf.tolist())):
        near = a2[cKDTree(tr[tleaf == h]).query(X[a2])[0] <= X_["trace_mm"]]; lbs = seedlab[near]; lbs = lbs[lbs > 0]
        if not len(near): continue
        top, k_ = Counter(lbs.tolist()).most_common(1)[0] if len(lbs) else (0, 0); comp = seedlab == top; pur = float((human[comp] == h).sum() / max(comp.sum(), 1)) if top else 0.0; share = k_ / len(near)
        tt[str(h)] = dict(top_candidate=int(top), share=round(share, 3), purity=round(pur, 3), reproduced=bool(share >= 0.5 and pur >= X_["purity_min"]))
    both = (human > 0) & (full > 0); mp = {int(al[j_]): int(hl[i_]) for i_, j_ in zip(ri, ci)}; agree_w = float(area[both][np.array([mp.get(int(a_), -1) == int(h) for a_, h in zip(full[both], human[both])])].sum() / max(area[both].sum(), 1e-9)) if both.any() else None
    R["reproduction"] = dict(per_human_leaf=rep, reproduced=int(sum(v["reproduced"] for v in rep.values())), of=len(hl), traces=tt, traces_reproduced=int(sum(v["reproduced"] for v in tt.values())), traces_total=len(tt),
                             area_weighted_agreement_with_human_seeded_R8pc=(round(agree_w, 4) if agree_w is not None else None), leaves_1_and_7={k: rep.get(k) for k in ("1", "7")}, note="agreement with human-seeded R8_pc is not accuracy against truth")
    code_full = np.where(full > 0, full, np.where(reason == 1, -5, np.where(contested, -1, np.where(ax == 1, -3, np.where(ax == 2, -4, 0)))))
    np.savez_compressed(os.path.join(D, "labels_autoseed.npz"), rows=rows, label=code_full.astype(np.int16), code_names=np.array(["-1:contested", "-3:stem", "-4:rachis_unpainted", "-5:exceeds_leaflet_prior"]), seed_component=seedlab.astype(np.int16), stage1=st1.astype(np.int16), human=human.astype(np.int16))
    json.dump(R, open(os.path.join(D, "results.json"), "w"), indent=1, default=float); log("R3-1b", json.dumps(dict(seeds=R["seeds"], cap=R["cap"], grades=R["grade_counts"], reproduced=R["reproduction"]["reproduced"], traces=R["reproduction"]["traces_reproduced"]), default=float))


def main():
    out = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoseed_out"); os.makedirs(out, exist_ok=True); t0 = time.time()
    pri = part_L(out); part_XE(out, pri); log("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
