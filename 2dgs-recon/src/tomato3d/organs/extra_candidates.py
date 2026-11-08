"""Extra candidate masks for leaflets the detectors missed (contract build/p1_extra_candidates_20261026_01/CONTRACT.md).

Candidates only: no leaf numbers, no 3D projection, no label change. Proposers P1 (crop + SAM automatic), P2 (crop + M1 / Roboflow boxes drawn by SAM),
P3 (point + box prompts on missed components), pooled and filtered. Test A (R3-1b replay with withheld candidates) decides whether anything is shown.

usage: python -m tomato3d.organs.extra_candidates --step testA | testB | preview [--port 8832]
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import light_organs as LO
from tomato3d.organs import userlabel as UL

ROOT = LO.ROOT
OUT = ROOT / "build/p1_extra_candidates_20261026_01"
QUEUE = ROOT / "build/p1_all_compound_20261026_01"
H2 = UL.H2
LABELS = dict(status="R3-1b_unaccepted", gate="gate_failed_viewing_only", scale=LO.SCALE_TAG, source="auto_extra")
P = dict(open_frac_width=0.5, crop_margin=0.10, crop_long_min=1500, auto_pps=(32, 48), pred_iou=0.7, stability=0.9, m1_score=0.15, n_neg=3, box_pad=0.10,
         area_tol=(0.5, 1.5), area_pct=(5, 95), solidity=0.6, inside_plant=0.70, inside_missed=0.60, iou_existing=0.5, contain_existing=0.8, dup_iou=0.6,
         border_frac=0.20, plant_dilate=3, withhold=0.30, seed=20261026, match_iou=0.5)
PASS = dict(recall=0.50, precision=0.50)
ATTEMPT = 2                                        # contract: second attempt, tuned_on_R3-1b


# ---------------------------------------------------------------- helpers
def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def solidity(m):
    pts = np.argwhere(m)[:, ::-1].astype(np.int32)
    return float(m.sum() / max(cv2.contourArea(cv2.convexHull(pts)), 1)) if len(pts) >= 3 else 0.0


def width_mm(m, ppm):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    (_, _), (a, b), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea)); return min(a, b) / ppm


def dt_peak(m):
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); y, x = np.unravel_index(int(np.argmax(d)), d.shape); return np.array([x, y])


def border_touch(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea).reshape(-1, 2); on = (c[:, 0] <= 0) | (c[:, 1] <= 0) | (c[:, 0] >= m.shape[1] - 1) | (c[:, 1] >= m.shape[0] - 1)
    return float(on.mean())


def active_records(recs, vid):
    act = {}
    for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
        if r.get("removed"):
            act.pop(r["removed"], None); continue
        act[r.get("mask_id") or r["id"]] = r
    return act


def human_leaflets(recs, vid, by_id):
    out = []
    for r in active_records(recs, vid).values():
        m = UL.record_rle(r, recs, by_id)
        if m is not None and m.any():
            out.append(m.astype(bool))
    return out


def size_prior(items):
    """items: list of (mask, ppm). Returns area mm^2 percentiles and median width mm."""
    a = [m.sum() / ppm ** 2 for m, ppm in items]; w = [x for x in (width_mm(m, ppm) for m, ppm in items) if x]
    return dict(n=len(a), area_p5_mm2=float(np.percentile(a, P["area_pct"][0])), area_p95_mm2=float(np.percentile(a, P["area_pct"][1])), width_median_mm=float(np.median(w)))


# ---------------------------------------------------------------- models
class Models:
    def __init__(self, rf=True):
        from ultralytics import YOLO
        from tomato3d.leafseg import hybrid as hy
        from tomato3d.leafseg import run as r2
        from tomato3d.topview.measure2d import load_sam
        self.sam = load_sam(r2.SAMCFG); self.sam._load(); self.m1 = YOLO(str(hy.M1_WEIGHTS)); self.hy = hy
        self.key = AP._rf_key() if rf else None; self.gens = {}

    def auto(self, rgb, pps):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        if pps not in self.gens:
            self.gens[pps] = SAM2AutomaticMaskGenerator(self.sam.predictor.model, points_per_side=pps, points_per_batch=64, pred_iou_thresh=P["pred_iou"], stability_score_thresh=P["stability"], min_mask_region_area=50)
        return [(np.asarray(r["segmentation"], bool), float(r.get("stability_score", 0.0))) for r in self.gens[pps].generate(rgb)]

    def boxes_m1(self, rgb):
        return [(d["box"], d["score"]) for d in self.hy.m1_to_json(self.hy.m1_detect(self.m1, rgb), rgb.shape[:2])["instances"]]

    def boxes_rf(self, rgb, cache):
        from tomato3d.leafseg import rfdet
        if not self.key and not cache.exists():
            return None
        try:
            pred = rfdet.fetch(cache.with_suffix(".png"), rgb, cache, self.key)
        except Exception as e:
            raise RuntimeError(AP._scrub(f"{type(e).__name__}: {e}", self.key)) from None
        return [(d["box"], d["conf"]) for d in rfdet.to_dets(pred, rgb.shape[:2])]

    def prompt(self, pts=None, lab=None, box=None):
        masks, scores, _ = self.sam.predictor.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True)
        k = int(np.argmax(scores)); return masks[k].astype(bool), float(scores[k])


# ---------------------------------------------------------------- one view
def crop_box(plant, full):
    ys, xs = np.nonzero(plant); g = full / H2
    x0, y0, x1, y1 = xs.min() * g, ys.min() * g, (xs.max() + 1) * g, (ys.max() + 1) * g; mx, my = P["crop_margin"] * (x1 - x0), P["crop_margin"] * (y1 - y0)
    x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(full, x1 + mx)), int(min(full, y1 + my))
    s = max(1.0, P["crop_long_min"] / max(x1 - x0, y1 - y0)); return dict(x0=x0, y0=y0, x1=x1, y1=y1, scale=round(s, 4))


def back_to_grid(m, cb, full):
    """mask on the (scaled) crop -> 1536 grid."""
    w, h = cb["x1"] - cb["x0"], cb["y1"] - cb["y0"]; mm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    F = np.zeros((full, full), np.uint8); F[cb["y0"]:cb["y1"], cb["x0"]:cb["x1"]] = mm
    return cv2.resize(F, (H2, H2), interpolation=cv2.INTER_NEAREST) > 0


def run_view(M, rgb, plant, cands, ppm, prior, rf_cache, log):
    full = rgb.shape[0]; t0 = time.time(); cu = np.zeros((H2, H2), bool)
    for c in cands:
        cu |= c
    k = max(3, int(round(P["open_frac_width"] * prior["width_median_mm"] * ppm))); k += (k + 1) % 2
    missed = cv2.morphologyEx((plant & ~cu).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    missed_raw = plant & ~cu                      # attempt 2 (tuned_on_R3-1b): the inside-missed filter uses the gap before the opening
    a_lo, a_hi = prior["area_p5_mm2"] * P["area_tol"][0] * ppm ** 2, prior["area_p95_mm2"] * P["area_tol"][1] * ppm ** 2
    log.update(missed_px=int(missed.sum()), open_kernel_px=k, area_range_px=[round(a_lo), round(a_hi)])
    cb = crop_box(plant, full); crop = rgb[cb["y0"]:cb["y1"], cb["x0"]:cb["x1"]]; s = cb["scale"]
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_CUBIC) if s > 1 else crop; log["crop"] = cb
    pool = []
    def keep_pass(m):
        return filt(m, plant, missed_raw if ATTEMPT == 2 else missed, cands, a_lo, a_hi) is None
    # P1
    for pps in P["auto_pps"]:
        got = [(back_to_grid(m, cb, full), sc) for m, sc in M.auto(crop, pps)]
        add = [dict(mask=m, score=sc, proposer=f"P1_auto{pps}") for m, sc in got if m.any()]
        log[f"P1_auto{pps}_raw"] = len(add); pool += add
        if any(keep_pass(d["mask"]) for d in add):
            break
    # P2
    M.sam.predictor.set_image(crop)
    for name, boxes in (("P2_M1", M.boxes_m1(crop)), ("P2_RF", M.boxes_rf(crop, rf_cache))):
        if boxes is None:
            log[name] = "skipped: no Roboflow key in memory"; continue
        n = 0
        for b, sc in boxes:
            m, q = M.prompt(box=np.asarray(b, np.float32)); g = back_to_grid(m, cb, full)
            if g.any():
                pool.append(dict(mask=g, score=q, proposer=name)); n += 1
        log[name + "_raw"] = n
    # P3 (on the same crop image already set)
    nl, labc, st, _ = cv2.connectedComponentsWithStats(missed.astype(np.uint8), 8); peaks = [dt_peak(c) for c in cands if c.any()]; n3 = 0
    g2c = lambda p: [((p[0] * full / H2) - cb["x0"]) * s, ((p[1] * full / H2) - cb["y0"]) * s]
    for i in range(1, nl):
        comp = labc == i; a = st[i, cv2.CC_STAT_AREA]
        if not (a_lo <= a <= a_hi) or solidity(comp) < P["solidity"]:
            continue
        pk = dt_peak(comp); negs = sorted(peaks, key=lambda q: float(np.hypot(*(q - pk))))[:P["n_neg"]]
        ys, xs = np.nonzero(comp); w_, h_ = xs.max() - xs.min(), ys.max() - ys.min(); pd = P["box_pad"]
        box = np.array(g2c((xs.min() - pd * w_, ys.min() - pd * h_)) + g2c((xs.max() + pd * w_, ys.max() + pd * h_)), np.float32)
        m, q = M.prompt(np.array([g2c(pk)] + [g2c(q) for q in negs], np.float32), np.array([1] + [0] * len(negs), np.int32), box); g = back_to_grid(m, cb, full)
        if g.any():
            pool.append(dict(mask=g, score=q, proposer="P3_point_box")); n3 += 1
    log["P3_raw"] = n3
    # filters
    kept, drops = [], Counter()
    for d in pool:
        why = filt(d["mask"], plant, missed_raw if ATTEMPT == 2 else missed, cands, a_lo, a_hi); d["drop"] = why
        if why:
            drops[why] += 1; continue
        kept.append(d)
    kept.sort(key=lambda d: -d["score"]); final = []
    for d in kept:
        if any(iou(d["mask"], f["mask"]) >= P["dup_iou"] for f in final):
            drops["duplicate"] += 1; continue
        d["missed_cover"] = round(float((d["mask"] & (missed_raw if ATTEMPT == 2 else missed)).sum() / d["mask"].sum()), 3); d["area_px"] = int(d["mask"].sum()); final.append(d)
    log["_pool"] = pool
    log.update(pool=len(pool), dropped=dict(drops), kept=len(final), kept_by_proposer=dict(Counter(d["proposer"] for d in final)), seconds=round(time.time() - t0, 1))
    return final, missed


def filt(m, plant, missed, cands, a_lo, a_hi):
    a = m.sum()
    pld = cv2.dilate(plant.astype(np.uint8), np.ones((2 * P["plant_dilate"] + 1,) * 2, np.uint8)) > 0
    if (m & pld).sum() / a < P["inside_plant"]:
        return "outside_plant"
    if (m & missed).sum() / a < P["inside_missed"]:
        return "outside_missed"
    for c in cands:
        inter = (m & c).sum()
        if inter and (inter / max((m | c).sum(), 1) >= P["iou_existing"] or inter / a >= P["contain_existing"]):
            return "matches_existing"
    if not (a_lo <= a <= a_hi):
        return "size"
    if solidity(m) < P["solidity"]:
        return "solidity"
    if border_touch(m) > P["border_frac"] and border_touch(plant) <= P["border_frac"]:
        return "border"
    return None


# ---------------------------------------------------------------- Test A (R3-1b)
def test_a():
    d = OUT / "testA_R3-1b"; (d / "examples").mkdir(parents=True, exist_ok=True); M = Models()
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); train = vc["chosen"]; recs = json.load(open(UL.OUT / "labels" / "leaf_labels.json", encoding="utf-8"))
    pl = UL.load_plant(); H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; views = pl["views"]
    det = {v: json.load(open(UL.OUT / "detect" / f"{v}.json", encoding="utf-8")) for v in train}; by_id = {v: {m["id"]: m for m in det[v]["masks"]} for v in train}
    ppm = {v: UL.plant_px_per_mm(pl, AP.view_file(v), H2id[:, views.index(AP.view_file(v))])[0] for v in train}
    hum = {v: human_leaflets(recs, v, by_id[v]) for v in train}; prior = size_prior([(m, ppm[v]) for v in train for m in hum[v]])
    rng = np.random.default_rng(P["seed"]); per_view, rows = {}, []
    for v in train:
        H = hum[v]; nw = int(round(P["withhold"] * len(H))); W = sorted(rng.choice(len(H), nw, replace=False).tolist())
        cands = [UL.unrle(m["rle"]) for m in det[v]["masks"]]; removed = [i for i, c in enumerate(cands) if any(iou(c, H[j]) >= P["match_iou"] for j in W)]
        reduced = [c for i, c in enumerate(cands) if i not in removed]
        view = AP.view_file(v); rgb = np.asarray(Image.open(AP.PHOTOS / view).convert("RGB")); plant = np.asarray(Image.open(AP.FG_MASKS / (Path(view).stem + ".png")).resize((H2, H2), Image.NEAREST)) == 1
        log = {}; final, missed = run_view(M, rgb, plant, reduced, ppm[v], prior, d / "rf_cache" / f"{v}.json", log)
        pool = log.pop("_pool"); diag = []
        for j in W:
            best = max(pool, key=lambda q: iou(q["mask"], H[j]), default=None)
            diag.append(dict(best_iou=round(iou(best["mask"], H[j]), 3) if best else 0.0, best_proposer=best["proposer"] if best else None, dropped_by=(best or {}).get("drop")))
        log["withheld_diagnostic"] = diag
        rec = [any(iou(f["mask"], H[j]) >= P["match_iou"] for f in final) for j in W]; prec = [any(iou(f["mask"], h) >= P["match_iou"] for h in H) for f in final]
        byp = {}
        for f, ok in zip(final, prec):
            b = byp.setdefault(f["proposer"], [0, 0]); b[0] += int(ok); b[1] += 1
        recp = {p: sum(any(iou(f["mask"], H[j]) >= P["match_iou"] for f in final if f["proposer"] == p) for j in W) for p in byp}
        per_view[v] = dict(human_leaflets=len(H), withheld=nw, candidates_removed=len(removed), candidates_left=len(reduced), recall=[sum(rec), len(rec)], precision=[sum(prec), len(prec)],
                           precision_by_proposer=byp, recall_by_proposer=recp, log=log)
        rows += [dict(view=v, proposer=f["proposer"], score=round(f["score"], 3), area_px=f["area_px"], missed_cover=f["missed_cover"], correct=ok, best_iou_human=round(max([iou(f["mask"], h) for h in H] or [0]), 3)) for f, ok in zip(final, prec)]
        sheet(rgb, reduced, final, d / "examples" / f"{v}.jpg", withheld=[H[j] for j in W]); print(v, json.dumps(per_view[v], default=float), flush=True)
    R = (sum(p["recall"][0] for p in per_view.values()), sum(p["recall"][1] for p in per_view.values())); Pr = (sum(p["precision"][0] for p in per_view.values()), sum(p["precision"][1] for p in per_view.values()))
    byp = {}
    for p in per_view.values():
        for k, (a, b) in p["precision_by_proposer"].items():
            x = byp.setdefault(k, dict(correct=0, n=0, recall_hits=0)); x["correct"] += a; x["n"] += b; x["recall_hits"] += p["recall_by_proposer"].get(k, 0)
    tot = dict(recall=round(R[0] / R[1], 3) if R[1] else None, recall_n=R, precision=round(Pr[0] / Pr[1], 3) if Pr[1] else None, precision_n=Pr, by_proposer=byp, size_prior=prior)
    verdict = dict(passed=bool(tot["recall"] is not None and tot["precision"] is not None and tot["recall"] >= PASS["recall"] and tot["precision"] >= PASS["precision"]))
    json.dump(dict(labels=dict(LABELS, attempt=ATTEMPT, tag=("tuned_on_R3-1b" if ATTEMPT == 2 else "as_declared")), params=P, rule=PASS, per_view=per_view, totals=tot, verdict=verdict), open(d / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    if rows:
        with open(d / "proposals.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("TOTAL", json.dumps(tot, default=float), json.dumps(verdict)); return verdict


def sheet(rgb, cands, extra, path, withheld=()):
    img = cv2.resize(rgb, (H2, H2), interpolation=cv2.INTER_AREA).copy()
    def draw(m, col, t, dashed=False):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if dashed:
                c = c.reshape(-1, 2)
                for i in range(0, len(c) - 6, 12):
                    cv2.line(img, tuple(int(v) for v in c[i]), tuple(int(v) for v in c[i + 6]), col, t)
            else:
                cv2.drawContours(img, [c], -1, col, t)
    for m in cands:
        draw(m, (90, 200, 255), 1)
    for m in withheld:
        draw(m, (255, 255, 0), 2)
    for f in extra:
        draw(f["mask"], (255, 60, 200), 3, dashed=True)
    Image.fromarray(img).save(path, quality=85)


# ---------------------------------------------------------------- Test B (n_04) and outputs
def test_b(iid="n_04"):
    M = Models(); vc = json.load(open(QUEUE / iid / "views_chosen.json", encoding="utf-8")); train = [Path(v).stem for v in vc["train"]]
    di = json.load(open(QUEUE / iid / "detect_input.json", encoding="utf-8")); ppm = {Path(v).stem: x["px_per_mm"] for v, x in di["ppm"].items()}
    recs = json.load(open(OUT / "snapshots" / f"{iid}_leaf_labels_snapshot.json", encoding="utf-8"))
    det = {v: json.load(open(QUEUE / iid / "detect" / f"{v}.json", encoding="utf-8")) for v in train}; by_id = {v: {m["id"]: m for m in det[v]["masks"]} for v in train}
    prior = size_prior([(m, ppm[v]) for v in train for m in human_leaflets(recs, v, by_id[v])])
    (OUT / "extra" / iid).mkdir(parents=True, exist_ok=True); (OUT / "overlay" / iid).mkdir(parents=True, exist_ok=True); summ = {}
    for v in train:
        rgb = np.asarray(Image.open(QUEUE / iid / "undist_3024" / f"{v}.jpg").convert("RGB")); plant = np.asarray(Image.open(QUEUE / iid / "masks" / f"{v}.png").resize((H2, H2), Image.NEAREST)) == 1
        cands = [UL.unrle(m["rle"]) for m in det[v]["masks"]]; log = {}
        final, missed = run_view(M, rgb, plant, cands, ppm[v], prior, OUT / "rf_cache" / iid / f"{v}.json", log)
        json.dump(dict(view=f"{v}.jpg", individual=iid, labels=LABELS, size_prior=prior, log=log,
                       masks=[dict(id=f"X{i + 1:03d}", uid=f"auto_extra:{i + 1}", prop="auto_extra", source="auto_extra", proposer=f["proposer"], score=round(f["score"], 4), rle=UL.rle(f["mask"]),
                                   area_px=f["area_px"], missed_cover=f["missed_cover"]) for i, f in enumerate(final)]),
                  open(OUT / "extra" / iid / f"{v}.json", "w", encoding="utf-8"))
        sheet(rgb, cands, final, OUT / "overlay" / iid / f"{v}.png"); summ[v] = dict(existing=len(cands), extra=len(final), by_proposer=log.get("kept_by_proposer"), missed_px=log["missed_px"], seconds=log["seconds"])
        print(iid, v, json.dumps(summ[v]), flush=True)
    json.dump(dict(labels=LABELS, size_prior=prior, per_view=summ, held_out_excluded=vc["held_out"]), open(OUT / "extra" / iid / "summary.json", "w", encoding="utf-8"), indent=1, default=float)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); a = ap.parse_args(argv)
    if a.step == "testA":
        test_a()
    if a.step == "testB":
        test_b()


if __name__ == "__main__":
    main()
