"""Gate A run of Option A (contract build/p1_leafseg_20261019_01/CONTRACT.md).

usage: python -m tomato3d.leafseg.run --out build/p1_leafseg_20261019_01 [--images tray_top_01 dwt_101 ...] [--detector roboflow]
(--detector roboflow = variant B, contract build/p1_leafseg_20261019_02/CONTRACT.md; key from ROBOFLOW_API_KEY)
Writes proposals/<image>_leafseg.json (schema leafseg-1), overlays/<image>.jpg, layers/<image>.npz, gateA.json,
parameters.json. Never writes human/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.leafseg import engine as eng
from tomato3d.leafseg import pixels as px
from tomato3d.leafseg import rfdet

TRAY = Path("data/records/tray_photos_20261019")
V5 = Path("build/p1_t1_occlusion_20261017_01/v5_segmentation_01/real_top")
OBB = Path("build/leaflets5_20260929/kout_obb/leaflet_obb_4plants.pt")
SAMCFG = Path("build/leaflets5_20260929/sam2_config_v5_4plants.json")
T1CFG = Path("build/p1_t1_occlusion_20261017_01/parameters.json")
RF_TRAIN = ("dwt_101", "dwt_102", "dwt_103")
HELD_OUT = ("dwt_104", "tray_top_07", "tray_top_08", "tray_top_09")
PSEUDO_FROM = tuple(f"tray_top_0{i}" for i in range(1, 7))   # addendum A1: development tray photos only
PSEUDO_IOU = 0.90
PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
                    [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200],
                    [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128]], np.uint8)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def images() -> dict:
    out = {f"tray_top_0{i}": TRAY / f"tray_top_0{i}.jpg" for i in range(1, 10)}
    for p in ("dwt_101", "dwt_102", "dwt_103", "dwt_104"):
        out[p] = Path("data/raw") / p / f"{p}_gen_20261005" / "top.heic"
    return out


def load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".heic":
        from tomato3d.topview.occlusion.io import decode
        return decode(path)
    return np.asarray(Image.open(path).convert("RGB"))


def v5_union(plant: str, shape) -> np.ndarray:
    d = V5 / plant; man = json.load(open(d / "masks_manifest.json", encoding="utf-8")); u = np.zeros(shape, bool)
    for m in man["masks"]:
        crop = np.array(Image.open(d / m["png"])) > 0; x0, y0 = m["offset_px"]; u[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] |= crop
    return u


def overlay(res: dict) -> np.ndarray:
    L = res["_layers"]; img = L["rgb"].astype(np.float32).copy(); lab = L["label"]; insts = res["instances"]
    grey = L["fg"] & (lab == 0); img[grey] = img[grey] * 0.35 + np.array([128, 128, 128]) * 0.65
    h, w = lab.shape; yy, xx = np.mgrid[:h, :w]; hatch = ((xx + yy) // 6) % 2 == 0
    for i, inst in enumerate(insts):
        m = lab == i + 1
        if inst["occluded"]:
            img[m & hatch] = img[m & hatch] * 0.4 + np.array([255, 140, 0]) * 0.6
        else:
            img[m] = img[m] * 0.55 + PALETTE[i % len(PALETTE)] * 0.45
    out = np.clip(img, 0, 255).astype(np.uint8)
    for i, inst in enumerate(insts):
        m = (lab == i + 1).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 140, 0) if inst["occluded"] else (255, 255, 255), 2 if inst["occluded"] else 1)
    out[L["skeleton"]] = (255, 0, 255)
    cen = {inst["id"]: inst["centroid"] for inst in insts if inst["centroid"]}
    for e in res["order_edges"]:
        a, b = cen.get(e["over"]), cen.get(e["under"])
        if a and b:
            cv2.arrowedLine(out, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 2, tipLength=0.15)
    for inst in insts:
        if inst["centroid"]:
            x, y = int(inst["centroid"][0]), int(inst["centroid"][1]); t = inst["id"][1:]
            cv2.putText(out, t, (x - 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, t, (x - 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--images", nargs="*")
    ap.add_argument("--detector", choices=("obb", "roboflow", "roboflow_tiled"), default="obb"); a = ap.parse_args(argv)
    out = Path(a.out); [(out / d).mkdir(parents=True, exist_ok=True) for d in ("proposals", "overlays", "layers")]
    from ultralytics import YOLO
    from tomato3d.topview.measure2d import load_sam
    cfg = json.load(open(T1CFG)); imgs = images()
    params = dict(contract=str(out / "CONTRACT.md"), obb=dict(path=str(OBB), sha256=sha(OBB), args=eng.DET_ARGS), sam_config=json.load(open(SAMCFG)),
                  t1_parameters=dict(path=str(T1CFG), sha256=sha(T1CFG)), grid_max_px=eng.GRID_MAX, t1_grid_px=eng.T1_GRID,
                  pixels={k: getattr(px, k) for k in ("SPEC_V", "SPEC_S", "INPAINT_R", "CLAHE_CLIP", "CLAHE_TILE", "LCN_SIGMA_AT_2000", "RF_DOWNSAMPLE",
                                                     "RF_ERODE_PX", "RF_NEG_MARGIN_PX", "PURPLE_Z", "PURPLE_SIGMA_AT_2000", "RIDGE_PCT")},
                  engine={k: getattr(eng, k) for k in ("NMS_IOU", "CONTAIN", "FG_MIN", "RACHIS_FRAC_MAX", "AREA_MIN_FRAC", "N_NEG_RACHIS", "N_NEG_PEEL",
                                                      "GREEN_FRAC", "FACE_MAJ", "FACE_BOTH")},
                  rf_train=list(RF_TRAIN), held_out=list(HELD_OUT), detector=a.detector, inputs={k: dict(path=str(v).replace("\\", "/"), sha256=sha(v)) for k, v in imgs.items()})
    json.dump(params, open(out / "parameters.json", "w"), indent=1, default=str)
    # ---- foreground RF from the v5 masks of dwt_101-103 (on the working grid)
    rng = np.random.default_rng(0); Xs, ys = [], []
    for p in RF_TRAIN:
        rgb = load_rgb(imgs[p]); H, W = rgb.shape[:2]; s = min(1.0, eng.GRID_MAX / max(H, W)); gs = (int(round(H * s)), int(round(W * s)))
        u = cv2.resize(v5_union(p, (H, W)).astype(np.uint8), (gs[1], gs[0]), interpolation=cv2.INTER_NEAREST) > 0
        F = px.pixel_features(px.preprocess(cv2.resize(rgb, (gs[1], gs[0]), interpolation=cv2.INTER_AREA))); X, y = px.training_pixels(F, u, rng); Xs.append(X); ys.append(y)
    rf = px.train_foreground(Xs, ys); print("rf0 trained", sum(len(y) for y in ys), "pixels", flush=True)
    det = YOLO(str(OBB)); sam = load_sam(SAMCFG)
    # ---- addendum A1: refine the RF with pseudo-labels from the DEVELOPMENT tray photos only
    pseudo = {}
    for name in PSEUDO_FROM:
        rgb = load_rgb(imgs[name]); H, W = rgb.shape[:2]; s = min(1.0, eng.GRID_MAX / max(H, W)); gs = (int(round(H * s)), int(round(W * s)))
        rgb_g = cv2.resize(rgb, (gs[1], gs[0]), interpolation=cv2.INTER_AREA); pp = px.preprocess(rgb_g); F = px.pixel_features(pp)
        fg = px.foreground_prob(rf, F, gs) > 0.5; rach = px.rachis_skeleton(pp, fg)
        rach_d = cv2.dilate(rach["skeleton"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        dets = eng.detect(det, rgb); sam.predictor.set_image(rgb); pos_u = np.zeros(gs, bool); boxes = np.zeros(gs, np.uint8); n_ok = 0
        for d in dets:
            x0, y0, x1, y1 = [int(round(v * s)) for v in d["box"]]; x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, gs[1] - 1), min(y1, gs[0] - 1)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes[y0:y1 + 1, x0:x1 + 1] = 1; reg = np.zeros(gs, bool); reg[y0:y1 + 1, x0:x1 + 1] = True; area = reg & fg & ~rach_d
            if area.any():
                dt = cv2.distanceTransform(area.astype(np.uint8), cv2.DIST_L2, 5); yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape); pos = np.array([[xx, yy]], np.float32)
            else:
                pos = np.array([[(x0 + x1) / 2, (y0 + y1) / 2]], np.float32)
            g, sc, q = eng._predict(sam, d["box"], pos / s, eng._sample_points(rach["skeleton"] & reg, eng.N_NEG_RACHIS, rng) / s, gs)
            if q >= PSEUDO_IOU:
                pos_u |= g; n_ok += 1
        u = pos_u.astype(np.uint8); dsk = int(px.RF_DOWNSAMPLE)
        posm = cv2.erode(u[::dsk, ::dsk], np.ones((2 * (px.RF_ERODE_PX // dsk) + 1,) * 2, np.uint8)) > 0
        negm = cv2.dilate(boxes[::dsk, ::dsk], np.ones((2 * (px.RF_NEG_MARGIN_PX // dsk) + 1,) * 2, np.uint8)) == 0
        ok = np.isfinite(F).all(-1); P, N = F[posm & ok], F[negm & ok]; k = min(len(P), len(N), 60000)
        if k:
            Xs.append(np.vstack([P[rng.choice(len(P), k, replace=False)], N[rng.choice(len(N), k, replace=False)]])); ys.append(np.r_[np.ones(k), np.zeros(k)])
        pseudo[name] = dict(detections=len(dets), pseudo_positive_masks=n_ok, pos_px=int(len(P)), neg_px=int(len(N)), used_each=int(k))
        print("pseudo", name, pseudo[name], flush=True)
    rf = px.train_foreground(Xs, ys); params["rf_pseudo_labels"] = dict(source_images=list(PSEUDO_FROM), sam_iou_min=PSEUDO_IOU, per_image=pseudo)
    json.dump(params, open(out / "parameters.json", "w"), indent=1, default=str); print("rf1 trained", sum(len(y) for y in ys), "pixels", flush=True)
    gate = json.load(open(out / "gateA.json")) if (out / "gateA.json").exists() else {}   # resume: finished images are skipped
    for name, path in imgs.items():
        if (a.images and name not in a.images) or (name in gate and (out / "proposals" / f"{name}_leafseg.json").exists()):
            continue
        t0 = time.time(); rgb = load_rgb(path); dets = comps = None
        if a.detector.startswith("roboflow"):
            key = os.environ.get("ROBOFLOW_API_KEY"); pred = rfdet.fetch(path, rgb, out / "detections" / f"{name}.json", key)
            dets = rfdet.to_dets(pred, rgb.shape[:2]); comps = rfdet.compound_boxes(pred, rgb.shape[:2])
            if a.detector == "roboflow_tiled" and name.startswith("tray_"):
                dets, tinfo = rfdet.tiled_dets(path, rgb, out / "detections_tiled", name, key); params.setdefault("tiling", {})[name] = tinfo
                json.dump(params, open(out / "parameters.json", "w"), indent=1, default=str); print("tiled", name, tinfo, flush=True)
        res = eng.run_image(rgb, det, sam, rf, cfg, dets=dets)
        if comps is not None:
            rfdet.assign_compound_boxes(res["_layers"]["label"], res["grid_scale"], res["instances"], comps); res["compound_boxes"] = comps
        ov = overlay(res); Image.fromarray(ov).save(out / "overlays" / f"{name}.jpg", quality=88)
        L = res.pop("_layers")
        np.savez_compressed(out / "layers" / f"{name}.npz", label=L["label"].astype(np.int16), fg=L["fg"], skeleton=L["skeleton"])
        doc = dict(schema="leafseg-1", image=dict(name=name, file=str(path).replace("\\", "/"), sha256=sha(path), size_hw=res["image_size"],
                                                  grid_hw=res["grid_shape"], grid_scale=res["grid_scale"],
                                                  split="held_out" if name in HELD_OUT else "development",
                                                  detector_seen="possible" if a.detector.startswith("roboflow") and name.startswith("dwt_") else "no"),
                   source="model_proposal", note="Option A proposals; RLE on the working grid (grid_hw); no accuracy claim before human review", **res)
        json.dump(doc, open(out / "proposals" / f"{name}_leafseg.json", "w"), default=float)
        insts = res["instances"]
        gate[name] = dict(split=doc["image"]["split"], detections=res["detections"], leaflets=len(insts), occluded=sum(i["occluded"] for i in insts),
                          unassigned_area_fraction=round(res["unassigned_area_fraction"], 3), order_edges=len(res["order_edges"]),
                          compounds=len({i["compound_leaf_id"] for i in insts} - {"unassigned"}), compound_unassigned=sum(i["compound_leaf_id"] == "unassigned" for i in insts),
                          faces={f: sum(i["face"] == f for i in insts) for f in ("adaxial", "abaxial", "folded", "unknown")},
                          variety={v: sum(i["variety_group"] == v for i in insts) for v in ("silvery", "green")},
                          failure_candidates=dict(merged=[i["id"] for i in insts if "merged" in i["flags"]], split_folded=res["split_folded_candidates"],
                                                  rachis_kept_as_leaf_dropped=sum(d["reason"] == "rachis_kept_as_leaf" for d in res["dropped"]),
                                                  truncated_by_image=[i["id"] for i in insts if i["truncated_by_image"]]),
                          dropped={r: sum(d["reason"] == r for d in res["dropped"]) for r in ("nms", "contained", "empty", "low_foreground", "rachis_kept_as_leaf", "too_small")},
                          seconds=round(time.time() - t0, 1))
        if comps is not None:
            gate[name].update(detector_classes={c: sum(i["detector_class"] == c for i in insts) for c in rfdet.LEAFLET_CLASSES}, compound_boxes=len(comps),
                              compound_box_unassigned=sum(i["compound_box_id"] == "unassigned" for i in insts), detector_seen=doc["image"]["detector_seen"])
        print(name, {k: v for k, v in gate[name].items() if k not in ("failure_candidates",)}, flush=True)
        json.dump(gate, open(out / "gateA.json", "w"), indent=1)


if __name__ == "__main__":
    main()
