"""Top-view anchored ray matching for cross-photo leaflet identity, R3-1b test (contract build/p1_topview_ray_match_20261026_01/CONTRACT.md).

No labels on surfels: 2D masks and camera rays only. Top photos (elevation >= 75 deg) give leaflet footprints; a side-photo leaflet mask attaches to a top
instance when its rays pass through that instance's column. Surfels give only the plant height range.

usage: python -m tomato3d.organs.topview_ray --step top_photos | instances | match
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import light_organs as LO
from tomato3d.organs import userlabel as UL
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_topview_ray_match_20261026_01"
POSES_DIR = ROOT / "build/p1_autolabel_20261026_01/poses"
H2 = UL.H2
LABELS = dict(status="R3-1b_unaccepted", gate="gate_failed_viewing_only", scale=LO.SCALE_TAG)
TOP_EL = 75.0
POT_TOP = 0.066
R = dict(tol_mm=2.0, match_iou=0.3, gap_mm=5.0, n_rays=200, erode_px=2, dz_mm=1.0, s_min=0.6, margin=0.2, margin_unknown=0.3, no_anchor=0.2,
         inside_plant=0.70, plant_dilate=3, crop_margin=0.10, seed=20261026)
PASS = dict(purity=0.90, coverage=0.40, heldout_acc=0.85)


# ---------------------------------------------------------------- top photos (all individuals)
def step_top_photos():
    rows, counts = [], {}
    for f in sorted(POSES_DIR.glob("*.csv")):
        iid = f.stem; n = 0
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if not r.get("elevation_deg"):
                continue
            el = float(r["elevation_deg"]); top = el >= TOP_EL; n += top
            rows.append(dict(individual=iid, photo=r["view"], elevation_deg=el, azimuth_deg=r["azimuth_deg"], top=top, pose_flag=r["flag"]))
        counts[iid] = n
    with open(OUT / "top_photos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(counts, open(OUT / "top_photos_count.json", "w", encoding="utf-8"), indent=1); print(counts); return counts


# ---------------------------------------------------------------- camera helpers
class Cams:
    def __init__(self):
        self.po = json.load(open(LO.DEFAULTS["poses"], encoding="utf-8")); self.K = ca.scaled_K(np.array(self.po["K_undist"]), self.po["undistorted_res"], H2); self.Ki = np.linalg.inv(self.K)

    def Rt(self, view):
        v = self.po["views"][view]; return np.array(v["R_cw"]), np.array(v["t_cw"])

    def centre(self, view):
        R_, t = self.Rt(view); return -R_.T @ t

    def rays(self, view, uv):
        R_, t = self.Rt(view); p = np.c_[uv, np.ones(len(uv))] @ self.Ki.T; return -R_.T @ t, p @ R_          # origin, directions (world)

    def project(self, view, X):
        R_, t = self.Rt(view); xc = X @ R_.T + t; z = xc[..., 2:3]; uv = (xc / np.where(z > 1e-9, z, 1e-9)) @ self.K.T
        return uv[..., 0], uv[..., 1], xc[..., 2]


def plant_heights():
    from tomato3d.organs import compound_cluster as CC
    P = CC.load_r31b(); return POT_TOP, float(P["xyz"][:, 2].max())


def top_views():
    rows = [r for r in csv.DictReader(open(OUT / "top_photos.csv", encoding="utf-8")) if r["individual"] == "R3-1b" and r["top"] == "True"]
    return [r["photo"] for r in rows]


# ---------------------------------------------------------------- step 1: top instances
def plant_of(view):
    return np.asarray(Image.open(AP.FG_MASKS / (Path(view).stem + ".png")).resize((H2, H2), Image.NEAREST)) == 1


def px_per_mm_top(C, view, z_mid, plant):
    ys, xs = np.nonzero(plant); o, d = C.rays(view, np.array([[xs.mean(), ys.mean()]])); s = (z_mid - o[2]) / d[0, 2]; X = o + s * d[0]
    _, _, zc = C.project(view, X[None]); return float(C.K[0, 0] * 0.001 / zc[0])


def step_instances():
    from tomato3d.leafseg import engine as eng
    from tomato3d.leafseg import pixels as px
    from tomato3d.leafseg import rfdet
    from tomato3d.leafseg import run as r2
    from tomato3d.topview.measure2d import load_sam
    C = Cams(); lo, hi = plant_heights(); zm = (lo + hi) / 2; sam = load_sam(r2.SAMCFG); sam._load(); cfg = json.load(open(r2.T1CFG, encoding="utf-8")); key = AP._rf_key()
    d = OUT / "top_instances"; d.mkdir(parents=True, exist_ok=True); summ = {}
    for view in top_views():
        stem = Path(view).stem; rgb = np.asarray(Image.open(AP.PHOTOS / view).convert("RGB")); plant = plant_of(view); ppm = px_per_mm_top(C, view, zm, plant); log = {}
        cands = []
        pj = json.load(open(ROOT / "build/p1_recon_charuco_20261021_01/leaflets/proposals" / f"{stem}.json", encoding="utf-8"))
        cands += [dict(uid=f"H2:{x['id']}", prop="P-H2", mask=px.rle_decode(x["visible_segmentation"]).astype(bool), score=float(x.get("score") or 0)) for x in pj["instances"]]
        # P-RF
        try:
            pred = rfdet.fetch(AP.PHOTOS / view, rgb, d / "rf_cache" / f"{stem}.json", key); n = len(pred.get("predictions", [])); tiled = None
            dets = rfdet.tiled_dets(AP.PHOTOS / view, rgb, d / "rf_cache_tiled", stem, key)[0] if n >= AP.RF_CAP else rfdet.to_dets(pred, rgb.shape[:2])
            with AP.fixed_foreground(plant):
                res = eng.run_image(rgb, None, sam, None, cfg, dets=dets)
            cands += [dict(uid=f"RF:{x['id']}", prop="P-RF", mask=px.rle_decode(x["visible_segmentation"]).astype(bool), score=float(x.get("score") or 0)) for x in res["instances"]]
            log["RF"] = dict(predictions=n, boxes=len(dets))
        except Exception as e:
            log["RF"] = dict(error=AP._scrub(f"{type(e).__name__}: {e}", key))
        # P-SA on the plant crop
        ys, xs = np.nonzero(plant); g = rgb.shape[0] / H2; x0, y0, x1, y1 = xs.min() * g, ys.min() * g, (xs.max() + 1) * g, (ys.max() + 1) * g
        mx, my = R["crop_margin"] * (x1 - x0), R["crop_margin"] * (y1 - y0); x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(rgb.shape[1], x1 + mx)), int(min(rgb.shape[0], y1 + my))
        s = UL.SAMAUTO["work_px"] / max(x1 - x0, y1 - y0); crop = cv2.resize(rgb[y0:y1, x0:x1], (int((x1 - x0) * s), int((y1 - y0) * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        log["SA_crop"] = dict(x0=x0, y0=y0, x1=x1, y1=y1, scale=round(s, 4)); nsa = 0
        for i, c in enumerate(sam.automatic(crop)):
            F = np.zeros(rgb.shape[:2], np.uint8); F[y0:y1, x0:x1] = cv2.resize(c.mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            m = cv2.resize(F, (H2, H2), interpolation=cv2.INTER_NEAREST) > 0
            if not m.any() or (m & plant).sum() / m.sum() < UL.SAMAUTO["inside_plant_min"]:
                continue
            ok, sh = UL.shape_ok(m, ppm)
            if ok:
                cands.append(dict(uid=f"SA:{i}", prop="P-SA", mask=m & plant, score=float(c.score))); nsa += 1
        log["SA_kept"] = nsa
        # plant restriction + UNION
        pld = cv2.dilate(plant.astype(np.uint8), np.ones((2 * R["plant_dilate"] + 1,) * 2, np.uint8)) > 0; keep = []
        for c in cands:
            if c["mask"].sum() and (c["mask"] & pld).sum() / c["mask"].sum() >= R["inside_plant"]:
                c["mask"] = c["mask"] & pld; keep.append(c)
        final, ust = UL.union_masks(keep)
        np.savez_compressed(d / f"{stem}.npz", masks=np.array([f["mask"] for f in final]), props=np.array([f["prop"] for f in final]), scores=np.array([f["score"] for f in final]))
        summ[view] = dict(raw=dict(Counter(c["prop"] for c in cands)), after_plant=len(keep), union=ust, by_prop=dict(Counter(f["prop"] for f in final)), px_per_mm_mid=round(ppm, 3), log=log)
        print(view, json.dumps(summ[view], default=float), flush=True)
    json.dump(dict(labels=LABELS, rule=R, z_pot_top=lo, z_plant_top=hi, per_view=summ), open(d / "summary.json", "w", encoding="utf-8"), indent=1, default=float)


# ---------------------------------------------------------------- step 2: footprints, matching, heights
def footprint(C, view, m, z, rng):
    ys, xs = np.nonzero(m)
    if len(xs) > 3000:
        k = rng.choice(len(xs), 3000, replace=False); xs, ys = xs[k], ys[k]
    o, d = C.rays(view, np.c_[xs, ys].astype(float)); s = (z - o[2]) / d[:, 2]; X = o + s[:, None] * d
    return set(map(tuple, np.floor(X[:, :2] * 1000).astype(int)))


def closest(o1, d1, o2, d2):
    d1 = d1 / np.linalg.norm(d1); d2 = d2 / np.linalg.norm(d2); w = o1 - o2; a, b, c, dd, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w, d2 @ w
    den = a * c - b * b; s = (b * e - c * dd) / den; t = (a * e - b * dd) / den; p1, p2 = o1 + s * d1, o2 + t * d2
    ang = float(np.arccos(np.clip(abs(d1 @ d2), -1, 1))); return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2)), ang


def build_instances(C, tol_mm):
    lo, hi = plant_heights(); zm = (lo + hi) / 2; views = top_views(); rng = np.random.default_rng(R["seed"]); per = {}
    for v in views:
        z = np.load(OUT / "top_instances" / f"{Path(v).stem}.npz"); ms = list(z["masks"]); ppm = px_per_mm_top(C, v, zm, plant_of(v)); k = max(1, int(round(tol_mm * ppm)))
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        per[v] = [dict(view=v, mask=m, dil=cv2.dilate(m.astype(np.uint8), el) > 0, fp=footprint(C, v, m, zm, rng)) for m in ms]
    inst = []; used = set()
    if len(views) == 2:
        A, B = views; IA, IB = per[A], per[B]
        iou = np.zeros((len(IA), len(IB)))
        for i, a in enumerate(IA):
            for j, b in enumerate(IB):
                inter = len(a["fp"] & b["fp"])
                if inter:
                    iou[i, j] = inter / len(a["fp"] | b["fp"])
        for i in range(len(IA)):
            if not len(IB):
                break
            j = int(iou[i].argmax())
            if iou[i, j] >= R["match_iou"] and int(iou[:, j].argmax()) == i:
                a, b = IA[i], IB[j]; ca_ = [np.nonzero(a["mask"])[1].mean(), np.nonzero(a["mask"])[0].mean()]; cb_ = [np.nonzero(b["mask"])[1].mean(), np.nonzero(b["mask"])[0].mean()]
                oa, da = C.rays(A, np.array([ca_])); ob, db = C.rays(B, np.array([cb_])); P, gap, ang = closest(oa, da[0], ob, db[0])
                rec = dict(parts={A: a, B: b}, footprint_iou=round(float(iou[i, j]), 3), gap_mm=round(gap * 1000, 2), ray_angle_deg=round(np.degrees(ang), 2))
                if gap * 1000 <= R["gap_mm"] and ang > 1e-6:
                    sig = R["tol_mm"] / 1000 / np.sin(ang); h = float(P[2]); rec.update(height=h, sigma=sig, lo=max(lo, h - 3 * sig), hi=min(hi, h + 3 * sig), height_unknown=False)
                else:
                    rec.update(height=None, sigma=None, lo=lo, hi=hi, height_unknown=True)
                inst.append(rec); used.add((A, i)); used.add((B, j))
    for v in views:
        for i, a in enumerate(per[v]):
            if (v, i) not in used:
                inst.append(dict(parts={v: a}, height=None, sigma=None, lo=lo, hi=hi, height_unknown=True))
    for k, t in enumerate(inst):
        t["id"] = f"T{k + 1:03d}"
    return inst, (lo, hi)


# ---------------------------------------------------------------- step 3/4: side masks
def side_scores(C, view, m, inst, zr, rng):
    er = cv2.erode(m.astype(np.uint8), np.ones((2 * R["erode_px"] + 1,) * 2, np.uint8)) > 0
    ys, xs = np.nonzero(er if er.any() else m)
    if len(xs) == 0:
        return None
    k = rng.choice(len(xs), min(R["n_rays"], len(xs)), replace=False); o, d = C.rays(view, np.c_[xs[k], ys[k]].astype(float))
    zs = np.arange(zr[0], zr[1] + 1e-9, R["dz_mm"] / 1000); n = len(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (zs[None, :] - o[2]) / d[:, 2:3]
    X = o + s[..., None] * d[:, None, :]; front = s > 0
    inside = {}
    for tv in {v for t in inst for v in t["parts"]}:
        u, w, zc = C.project(tv, X); ui, wi = np.round(u).astype(int), np.round(w).astype(int); ok = (zc > 0) & (ui >= 0) & (wi >= 0) & (ui < H2) & (wi < H2) & front
        inside[tv] = (np.clip(ui, 0, H2 - 1), np.clip(wi, 0, H2 - 1), ok)
    hit = np.zeros((len(inst), n), bool); entry_s = np.full((len(inst), n), np.inf); entry_z = np.full((len(inst), n), np.nan)
    for j, t in enumerate(inst):
        inb = (zs >= t["lo"] - 1e-9) & (zs <= t["hi"] + 1e-9); h = np.ones(s.shape, bool) & inb[None, :]
        for tv, part in t["parts"].items():
            ui, wi, ok = inside[tv]; h &= ok & part["dil"][wi, ui]
        anyh = h.any(1); hit[j] = anyh
        sm = np.where(h, s, np.inf); idx = sm.argmin(1); entry_s[j] = sm[np.arange(n), idx]; entry_z[j] = np.where(anyh, zs[idx], np.nan)
    occ = np.zeros_like(hit)
    for j in range(len(inst)):
        nearer = hit & (entry_s < entry_s[j][None, :]) & (entry_z > entry_z[j][None, :]); nearer[j] = False
        occ[j] = hit[j] & nearer.any(0)
    sc = (hit & ~occ).sum(1) / n; sc_raw = hit.sum(1) / n
    return dict(score=sc, score_raw=sc_raw, occluded=occ.sum(1), entry_z=np.array([np.nanmedian(entry_z[j][hit[j]]) if hit[j].any() else np.nan for j in range(len(inst))]), n=n)


def decide(r, inst):
    sc = r["score"]; order = np.argsort(-sc); b = int(order[0]); sec = int(order[1]) if len(order) > 1 else None
    best, second = float(sc[b]), (float(sc[sec]) if sec is not None else 0.0)
    unk = inst[b]["height_unknown"] or (sec is not None and inst[sec]["height_unknown"] and second > 0); need = R["margin_unknown"] if unk else R["margin"]
    if best >= R["s_min"] and best - second >= need:
        return b, None, best, second
    if best < R["no_anchor"]:
        why = "no_top_anchor"
    elif best < R["s_min"]:
        why = "occlusion" if float(r["score_raw"][b]) >= R["s_min"] or float(r["score_raw"].max()) >= R["s_min"] else "low_score"
    else:
        why = "small_margin"
    return None, why, best, second


# ---------------------------------------------------------------- truth helpers (R3-1b user records)
def active_records(recs, vid):
    act = {}
    for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
        if r.get("removed"):
            act.pop(r["removed"], None); continue
        act[r.get("mask_id") or r["id"]] = r
    return act


def run_match(tol_mm, write=True):
    C = Cams(); inst, zr = build_instances(C, tol_mm); rng = np.random.default_rng(R["seed"])
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); train, held = vc["chosen"], vc["held_out"]
    recs = json.load(open(UL.OUT / "labels" / "leaf_labels.json", encoding="utf-8")); nmap = json.load(open(UL.OUT / "labels" / "number_map.json", encoding="utf-8"))["latest"]["map"]
    det = {v: json.load(open(UL.OUT / "detect" / f"{v}.json", encoding="utf-8")) for v in train + [held]}; by_id = {v: {m["id"]: m for m in det[v]["masks"]} for v in det}
    side, rows = [], []
    for v in train:
        for key_, r in active_records(recs, v).items():
            if r["leaf"] in ("모름", "unknown", None):
                continue
            num = nmap.get(f"{v}|{int(r['leaf'])}")
            if num is None or int(num) <= 0:
                continue
            m = UL.record_rle(r, recs, by_id[v])
            if m is None or not m.any():
                continue
            sc = side_scores(C, AP.view_file(v), m.astype(bool), inst, zr, rng)
            if sc is None:
                continue
            b, why, best, second = decide(sc, inst)
            side.append(dict(view=v, key=key_, number=int(num), area=int(m.sum()), mask=m.astype(bool), inst=b, why=why, best=best, second=second, entry_z=(float(sc["entry_z"][b]) if b is not None else None)))
            rows.append(dict(view=v, mask=key_, user_number=int(num), area_px=int(m.sum()), best_instance=inst[int(np.argmax(sc["score"]))]["id"], score=round(best, 3), second=round(second, 3),
                             margin=round(best - second, 3), assigned=(inst[b]["id"] if b is not None else ""), abstain=why or "", entry_height_mm=(round((sc["entry_z"][b] - POT_TOP) * 1000, 1) if b is not None else ""),
                             occluded_rays=int(sc["occluded"][int(np.argmax(sc["score"]))]), rays=sc["n"]))
    # numbers per instance
    att = defaultdict(list)
    for s_ in side:
        if s_["inst"] is not None:
            att[s_["inst"]].append(s_)
    numbers = {}
    for k, lst in att.items():
        c = Counter(x["number"] for x in lst); L, n = c.most_common(1)[0]
        numbers[k] = L if n / len(lst) > 0.5 else None
    # metrics (training)
    pur_num = pur_den = 0
    for k, lst in att.items():
        if numbers[k] is None:
            continue
        a = sum(x["area"] for x in lst); pur_num += sum(x["area"] for x in lst if x["number"] == numbers[k]); pur_den += a
    purity = pur_num / pur_den if pur_den else None
    cov = sum(x["area"] for x in side if x["inst"] is not None) / max(sum(x["area"] for x in side), 1)
    A_ = [x for x in side if x["inst"] is not None]; agree = n_pairs = tp = same_inst = same_num = 0
    for i in range(len(A_)):
        for j in range(i + 1, len(A_)):
            if A_[i]["view"] == A_[j]["view"]:
                continue
            si = A_[i]["inst"] == A_[j]["inst"]; sn = A_[i]["number"] == A_[j]["number"]; n_pairs += 1; agree += (si == sn); tp += si and sn; same_inst += si; same_num += sn
    nmatch = sum(1 for x in A_ if numbers.get(x["inst"]) == x["number"])
    # held-out
    raw, _ = UL.label_maps(UL.OUT, [held]); truth = UL.remap({held: raw[held]}, nmap)[held]; truth = np.where(truth > 0, truth, 0).astype(np.int16)
    hv = AP.view_file(held); hrows = []; rendered = np.zeros((H2, H2), np.int16); acc_num = acc_den = 0; hcand = []
    for m in det[held]["masks"]:
        mm = UL.unrle(m["rle"]).astype(bool); sc = side_scores(C, hv, mm, inst, zr, rng)
        if sc is None:
            continue
        b, why, best, second = decide(sc, inst); L = numbers.get(b) if b is not None else None; hcand.append((mm, L, b, why))
        hrows.append(dict(view=held, mask=m["id"], assigned=(inst[b]["id"] if b is not None else ""), number=L or "", abstain=why or ("unnumbered_instance" if b is not None and L is None else ""), score=round(best, 3), second=round(second, 3)))
    for mm, L, b, why in sorted(hcand, key=lambda t: -t[0].sum()):
        if L:
            rendered[mm] = L
    for mm, L, b, why in hcand:
        if L:
            px_ = mm & (truth > 0); acc_den += int(px_.sum()); acc_num += int((px_ & (truth == L)).sum())
    hacc = acc_num / acc_den if acc_den else None
    grows, gate = UL.score_heldout(truth, rendered)
    usable = [L for L, r_ in grows.items() if r_["iou"] >= 0.55 and r_["confusion"] <= 0.10]
    # splits
    ze = [x["entry_z"] for x in A_ if x["entry_z"] is not None and not np.isnan(x["entry_z"])]; terc = np.percentile(ze, [100 / 3, 200 / 3]) if ze else [0, 0]; tbl = {}
    for x in A_:
        if x["entry_z"] is None or np.isnan(x["entry_z"]):
            continue
        t = "low" if x["entry_z"] < terc[0] else "mid" if x["entry_z"] < terc[1] else "high"; d_ = tbl.setdefault(t, dict(masks=0, number_match=0, area=0))
        d_["masks"] += 1; d_["number_match"] += int(numbers.get(x["inst"]) == x["number"]); d_["area"] += x["area"]
    res = dict(tol_mm=tol_mm, instances=len(inst), triangulated=sum(not t["height_unknown"] for t in inst), height_unknown=sum(t["height_unknown"] for t in inst),
               matched_pairs=sum(len(t["parts"]) == 2 for t in inst), side_masks=len(side), attached=len(A_), abstain=dict(Counter(x["why"] for x in side if x["why"])),
               purity=(round(purity, 3) if purity is not None else None), coverage=round(cov, 3), attached_number_match=[nmatch, len(A_)],
               raw_agreement=(round(agree / n_pairs, 3) if n_pairs else None), pairs=n_pairs, same_instance_precision=(round(tp / same_inst, 3) if same_inst else None), same_number_recall=(round(tp / same_num, 3) if same_num else None),
               numbered_instances=sum(1 for k in numbers if numbers[k] is not None), heldout_candidates=len(hcand), heldout_attached_numbered=sum(1 for c in hcand if c[1]),
               heldout_number_accuracy=(round(hacc, 3) if hacc is not None else None), heldout_scored_px=acc_den, gate=gate, gate_rows={str(k): v for k, v in grows.items()}, usable_leaves=usable,
               by_height_tercile=tbl, tercile_edges_mm=[round((x - POT_TOP) * 1000, 1) for x in terc], per_view={v: dict(masks=sum(1 for x in side if x["view"] == v), attached=sum(1 for x in A_ if x["view"] == v)) for v in train})
    res["verdict"] = dict(passed=bool(purity is not None and hacc is not None and purity >= PASS["purity"] and cov >= PASS["coverage"] and hacc >= PASS["heldout_acc"]))
    if write:
        with open(OUT / "scores.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (OUT / "heldout_pred").mkdir(exist_ok=True); Image.fromarray(rendered.astype(np.uint8)).save(OUT / "heldout_pred" / f"{held}_pred.png")
        with open(OUT / "heldout_pred" / "candidates.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(hrows[0])); w.writeheader(); w.writerows(hrows)
        tdir = OUT / "top_instances"; json.dump([dict(id=t["id"], photos=[Path(v).stem for v in t["parts"]], height_mm=(round((t["height"] - POT_TOP) * 1000, 1) if t["height"] else None),
                                                     sigma_mm=(round(t["sigma"] * 1000, 1) if t["sigma"] else None), bounds_mm=[round((t["lo"] - POT_TOP) * 1000, 1), round((t["hi"] - POT_TOP) * 1000, 1)],
                                                     height_unknown=t["height_unknown"], gap_mm=t.get("gap_mm"), footprint_iou=t.get("footprint_iou"), ray_angle_deg=t.get("ray_angle_deg"), number=numbers.get(k)) for k, t in enumerate(inst)],
                                               open(tdir / "instances.json", "w", encoding="utf-8"), indent=1)
        json.dump(dict(labels=LABELS, assignments=[dict(view=x["view"], mask=x["key"], instance=(inst[x["inst"]]["id"] if x["inst"] is not None else None), user_number=x["number"], abstain=x["why"]) for x in side],
                       instance_numbers={inst[k]["id"]: v for k, v in numbers.items()}), open(OUT / "assignments.json", "w", encoding="utf-8"), indent=1)
        viz(C, inst, numbers, side, train, hcand, held)
    return res


PAL = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (220, 190, 255)]


def viz(C, inst, numbers, side, train, hcand, held):
    d = OUT / "viz"; d.mkdir(exist_ok=True)
    col = lambda k: PAL[k % len(PAL)]
    for v in top_views():
        img = cv2.resize(np.asarray(Image.open(AP.PHOTOS / v).convert("RGB")), (H2, H2)).copy()
        for k, t in enumerate(inst):
            if v in t["parts"]:
                cs, _ = cv2.findContours(t["parts"][v]["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cs, -1, col(k), 2)
                if cs:
                    x, y, _, _ = cv2.boundingRect(max(cs, key=cv2.contourArea)); cv2.putText(img, f"{t['id'][1:]}" + (f"/{numbers[k]}" if numbers.get(k) else ""), (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        Image.fromarray(img).save(d / f"top_{Path(v).stem}.jpg", quality=85)
    for v in train + [held]:
        img = cv2.resize(np.asarray(Image.open(AP.PHOTOS / AP.view_file(v)).convert("RGB")), (H2, H2)).copy()
        items = [(x["mask"], x["inst"], x["number"]) for x in side if x["view"] == v] if v != held else [(m, b, L) for m, L, b, _ in hcand]
        for m, b, num in items:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if b is None:
                cv2.drawContours(img, cs, -1, (150, 150, 150), 1); continue
            bad = v != held and numbers.get(b) is not None and numbers[b] != num
            cv2.drawContours(img, cs, -1, (255, 0, 0) if bad else col(b), 4 if bad else 2)
        Image.fromarray(img).save(d / f"{'heldout' if v == held else 'train'}_{v}.jpg", quality=85)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); a = ap.parse_args(argv); OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "top_photos":
        step_top_photos()
    if a.step == "instances":
        step_instances()
    if a.step == "match":
        out = {}
        for tol in (2.0, 1.0, 3.0):
            out[str(tol)] = run_match(tol, write=(tol == 2.0)); print(tol, json.dumps({k: v for k, v in out[str(tol)].items() if k not in ("gate_rows",)}, default=float), flush=True)
        json.dump(dict(labels=LABELS, rule=R, pass_rule=PASS, judged="2.0", reference=dict(prefill_raw_match="11/68", ABCDprime=dict(median_iou=0.517, max_confusion=0.393, usable=[1, 7])), runs=out),
                  open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()
