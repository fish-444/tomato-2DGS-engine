"""Candidate masks for EVERY posed photo of one individual (contract build/p1_all_3dfirst_20261027_01, Step 1.4). Runs on Kaggle, one batch per individual.

Per photo: optional crop + upsample (plant < 10 % of the frame, from `detect_input.json`), then
  P-H2  M1 + SAM v5 (charuco_leaflets.run_view unchanged) on the crop,
  P-SA  SAM 2.1 automatic masks, grid 32 x 32, pred_iou 0.7, stability 0.9, default box NMS (nested whole / part / sub-part kept; only IoU >= 0.9 duplicates dropped),
  P-RF  drawn from cached Roboflow boxes when `rf_detections/<stem>.json` exists (the key is never present here).
Masks are mapped back to the 1536 grid of the full frame and kept if >= 70 % inside the plant mask (dilated 3 px) and inside the size range (px^2 from mm^2 through the photo's px/mm).

usage: python -m tomato3d.organs.all3d_cands_job --job JOB/<id> --out OUT/<id> --sam-config sam.json --t1 parameters.json --m1 M1.pt --size-prior size_prior.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

H2 = 1536
SA = dict(points_per_side=32, pred_iou=0.7, stability=0.9, dup_iou=0.9)
FILT = dict(inside_plant=0.70, plant_dilate=3, size_tol=(0.5, 1.5))


def rle(m):
    from tomato3d.organs import seed2d as S2
    return dict(size=[H2, H2], counts=S2.rle_encode(m))


def to_grid(mask_crop_native, crop, full):
    """bool mask at the crop's native pixel size -> 1536 grid of the full frame."""
    F = np.zeros((full, full), np.uint8); F[crop["y0"]:crop["y1"], crop["x0"]:crop["x1"]] = mask_crop_native
    return cv2.resize(F, (H2, H2), interpolation=cv2.INTER_NEAREST) > 0


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def run(job: Path, out: Path, sam_config: Path, t1: Path, m1: Path, size_prior: dict, views=None, layers="h2,sa"):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from ultralytics import YOLO
    from tomato3d.leafseg import hybrid as hy
    from tomato3d.leafseg import pixels as px
    from tomato3d.leafseg import run as r2
    from tomato3d.organs import all_detect_job as DJ
    from tomato3d.recon import charuco_leaflets as CL
    from tomato3d.topview.measure2d import load_sam
    r2.SAMCFG = Path(sam_config); r2.T1CFG = Path(t1); hy.M1_WEIGHTS = Path(m1)
    model = YOLO(str(m1)); sam = load_sam(Path(sam_config)); cfg = json.load(open(t1, encoding="utf-8"))
    gen = SAM2AutomaticMaskGenerator(sam.predictor.model, points_per_side=SA["points_per_side"], points_per_batch=64, pred_iou_thresh=SA["pred_iou"], stability_score_thresh=SA["stability"], min_mask_region_area=50)
    di = json.load(open(job / "detect_input.json", encoding="utf-8")); out.mkdir(parents=True, exist_ok=True); A = size_prior["area_mm2"]; summ = {}
    L = set(layers.split(","))
    for v in (views or di["views"]):
        res_f = out / (f"{Path(v).stem}_cands.json" if L & {"h2", "sa"} else f"{Path(v).stem}_rf.json")
        if res_f.exists() or (L == {"rf"} and not (job / "rf_detections" / f"{Path(v).stem}.json").exists()):
            continue
        t0 = time.time(); stem = Path(v).stem; info = di["photos"][v]; log = dict(crop=info.get("crop"))
        rgb = np.asarray(Image.open(job / "photos" / v).convert("RGB")); full = rgb.shape[0]
        plant1512 = np.asarray(Image.open(job / "masks" / f"{stem}.png")) == 1; plant = np.asarray(Image.fromarray(plant1512.astype(np.uint8)).resize((H2, H2), Image.NEAREST)) == 1
        pld = cv2.dilate(plant.astype(np.uint8), np.ones((2 * FILT["plant_dilate"] + 1,) * 2, np.uint8)) > 0
        c = info.get("crop") or dict(x0=0, y0=0, x1=full, y1=full, scale=1.0); crop_rgb = rgb[c["y0"]:c["y1"], c["x0"]:c["x1"]]; ch, cw = crop_rgb.shape[:2]; s = c["scale"]
        det_img = cv2.resize(crop_rgb, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_CUBIC) if s > 1 else crop_rgb
        plant_full = np.asarray(Image.fromarray(plant1512.astype(np.uint8)).resize((full, full), Image.NEAREST)) == 1; plant_crop = plant_full[c["y0"]:c["y1"], c["x0"]:c["x1"]]
        plant_det = cv2.resize(plant_crop.astype(np.uint8), (det_img.shape[1], det_img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ppm = info["px_per_mm"]; lo, hi = FILT["size_tol"][0] * A["p5"] * ppm ** 2, FILT["size_tol"][1] * A["p95"] * ppm ** 2; cands = []; seen = []
        def add(uid, prop, m, score):
            if not m.any():
                return
            if (m & pld).sum() / m.sum() < FILT["inside_plant"]:
                return
            a = m.sum()
            if not (lo <= a <= hi):
                return
            if any(iou(m, q) >= SA["dup_iou"] for q in seen):
                return
            seen.append(m); cands.append(dict(uid=uid, prop=prop, score=float(score), rle=rle(m)))
        if "h2" in L:                                                                      # P-H2 on the crop
            h2dir = out / "h2" / stem; log["H2"] = CL.run_view(v, det_img, plant_det, model, sam, cfg, h2dir); pj = json.load(open(h2dir / "proposals" / f"{stem}.json", encoding="utf-8"))
            for x in pj["instances"]:
                g = px.rle_decode(x["visible_segmentation"]).astype(np.uint8); native = cv2.resize(g, (cw, ch), interpolation=cv2.INTER_NEAREST); add(f"H2:{x['id']}", "P-H2", to_grid(native, c, full), x.get("score") or 0.0)
        if "sa" in L:                                                                      # P-SA nested
            sam.predictor.set_image(det_img); recs = gen.generate(det_img); n_sa = 0
            for i, r in enumerate(recs):
                native = cv2.resize(np.asarray(r["segmentation"], np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST); before = len(cands); add(f"SA:{i}", "P-SA", to_grid(native, c, full), r.get("stability_score", 0.0)); n_sa += len(cands) - before
            log["SA"] = dict(generated=len(recs), kept=n_sa)
        # P-RF from cached boxes (no key here)
        rf = job / "rf_detections" / f"{stem}.json"
        if "rf" in L and rf.exists():
            try:
                log["RF"] = rf_draw(job, stem, det_img, plant_det, sam, cfg, c, full, add, cw, ch)
            except (Exception, SystemExit) as e:
                log["RF"] = dict(error=f"{type(e).__name__}: {e}")
        json.dump(dict(view=v, labels=dict(status="R3-1b-style_unaccepted", gate="gate_failed_viewing_only"), size_range_px=[round(lo), round(hi)], candidates=cands, log=log, counts=dict(Counter(x["prop"] for x in cands)), seconds=round(time.time() - t0, 1)),
                  open(res_f, "w", encoding="utf-8"), default=float)
        summ[v] = dict(counts=dict(Counter(x["prop"] for x in cands)), seconds=round(time.time() - t0, 1)); print(stem, summ[v], flush=True)
    json.dump(summ, open(out / "summary.json", "w", encoding="utf-8"), indent=1)


def rf_draw(job, stem, det_img, plant_det, sam, cfg, c, full, add, cw, ch):
    from tomato3d.leafseg import engine as eng
    from tomato3d.leafseg import pixels as px
    from tomato3d.leafseg import rfdet
    from tomato3d.organs import auto_proposers as AP
    pred = rfdet.fetch(job / "photos" / f"{stem}.jpg", det_img, job / "rf_detections" / f"{stem}.json", None); n = len(pred.get("predictions", []))
    dets = rfdet.tiled_dets(job / "photos" / f"{stem}.jpg", det_img, job / "rf_detections_tiled", stem, None)[0] if n >= AP.RF_CAP else rfdet.to_dets(pred, det_img.shape[:2])
    with AP.fixed_foreground(cv2.resize(plant_det.astype(np.uint8), (H2, H2), interpolation=cv2.INTER_NEAREST) > 0):
        res = eng.run_image(det_img, None, sam, None, cfg, dets=dets)
    k = 0
    for x in res["instances"]:
        g = px.rle_decode(x["visible_segmentation"]).astype(np.uint8); native = cv2.resize(g, (cw, ch), interpolation=cv2.INTER_NEAREST); b = k; add(f"RF:{x['id']}", "P-RF", to_grid(native, c, full), x.get("score") or 0.0); k += 1
    return dict(predictions=n, boxes=len(dets), instances=len(res["instances"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("job", "out", "sam-config", "t1", "m1", "size-prior"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--views", nargs="*"); ap.add_argument("--layers", default="h2,sa"); a = ap.parse_args(argv)
    run(Path(a.job), Path(a.out), Path(a.sam_config), Path(a.t1), Path(a.m1), json.load(open(a.size_prior, encoding="utf-8")), a.views, a.layers)


if __name__ == "__main__":
    main()
