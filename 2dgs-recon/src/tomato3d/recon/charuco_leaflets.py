"""Step 1b: leaflet instances per accepted view with the kept model H2 (M1 finds at score >= 0.15, SAM v5 draws with box +
DT-max positive + neighbour/rachis negatives, merge/clean, run-02 T1 tail `engine.finish`). Functions are imported from
tomato3d.leafseg.hybrid unchanged. Differences from the tray/dwt runs (no run-02 proposals exist for this capture):
  - foreground = the Step 1a plant mask (colour within the projected visual hull), not the run-02 RF;
  - no Roboflow v4 detections (no call): `dets = []`, so compound-box assignment is skipped;
  - size floor F = FLOOR_FRAC x plant-foreground grid area (FLOOR_FRAC = median run-04 floor / B foreground over dwt_101-104).

usage: python -m tomato3d.recon.charuco_leaflets --poses <charuco dir> --recon <recon dir> [--views a.jpg ...]
Writes <recon>/leaflets/{proposals/<view>.json, labels/<view>.png (uint16 instance id on the 1536 grid), overlays/<view>.jpg},
leaflets_summary.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.leafseg import engine as eng
from tomato3d.leafseg import hybrid as hy
from tomato3d.leafseg import pixels as px
from tomato3d.leafseg import run as r2
from tomato3d.leafseg.variant_d import _rachis_fraction

GRID = 1536
FLOOR_FRAC = 0.0139


def run_view(v: str, rgb: np.ndarray, plant: np.ndarray, model, sam, cfg, out: Path) -> dict:
    t0 = time.time(); H, W = rgb.shape[:2]; s = GRID / max(H, W); gshape = (int(round(H * s)), int(round(W * s)))
    rgb_g = cv2.resize(rgb, (gshape[1], gshape[0]), interpolation=cv2.INTER_AREA)
    fg = cv2.resize(plant.astype(np.uint8), (gshape[1], gshape[0]), interpolation=cv2.INTER_NEAREST) > 0
    rach = px.rachis_skeleton(px.preprocess(rgb_g), fg); skel = rach["skeleton"]
    rach_d = cv2.dilate(skel.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    F = max(200, int(round(FLOOR_FRAC * fg.sum()))); rng = np.random.default_rng(0)
    mp = out / "m1" / (Path(v).stem + ".json")
    if mp.exists():
        mj = json.load(open(mp))
    else:
        t1 = time.time(); mj = hy.m1_to_json(hy.m1_detect(model, rgb), (H, W)); mj["cpu_seconds"] = round(time.time() - t1, 1)
        mp.parent.mkdir(parents=True, exist_ok=True); json.dump(mj, open(mp, "w"))
    m1 = []
    for inst in mj["instances"]:
        g = cv2.resize(px.rle_decode(inst["mask"]).astype(np.float32), (gshape[1], gshape[0]), interpolation=cv2.INTER_AREA) >= 0.5
        if g.any() and fg[g].mean() >= hy.FG_SHARE:          # background plants / clutter never reach SAM
            m1.append(dict(id=inst["id"], score=inst["score"], box=inst["box"], grid=g, centre=[(inst["box"][0] + inst["box"][2]) / 2, (inst["box"][1] + inst["box"][3]) / 2]))
    sam.predictor.set_image(rgb); cands = []
    for i, inst in enumerate(m1):
        pbox = hy.padded_box(inst["box"], (H, W)); pos, neg = hy.prompts_h2(i, m1, pbox, skel, gshape, s, rng)
        r = hy.sam_pick(sam, pbox, pos, neg, gshape)
        cands.append(None if r is None or not r[0].any() else dict(m1=i, mask=r[0], score=r[1], iou_pred=r[2], source="H2_sam"))
    kept, log = hy.clean_and_merge(cands, m1, F, fg, rach_d, gshape)
    final = []
    for k, c in enumerate(sorted(kept, key=lambda c: -c["score"])):
        g = c["mask"]; ys, xs = np.nonzero(g); bb = [xs.min() / s, ys.min() / s, xs.max() / s, ys.max() / s]
        final.append(dict(id=f"H{k + 1:03d}", mask=g, score=float(c["score"]), iou_pred=c["iou_pred"], peel_rank=k, fg_mean=float(fg[g].mean()),
                          rachis_fraction=_rachis_fraction(g, rach_d, gshape),
                          det=dict(det=-1, conf=c["score"], cls="leaflet", centre=[float(xs.mean() / s), float(ys.mean() / s)], box=bb),
                          source=c.get("source", "H2_sam"), merged_from=c["merged_from"], m1_score=m1[c["m1"]]["score"]))
    res = eng.finish(final, [], rgb_g, fg, dict(skeleton=skel, ends=[], branches=[]), cfg, gshape, s, (H, W))
    for inst, c in zip(res["instances"], final):
        inst.update(source=c["source"], merged_from=c["merged_from"], m1_score=c["m1_score"], solidity=round(eng._solidity(c["mask"]), 4))
    for d in ("proposals", "labels", "overlays"):
        (out / d).mkdir(parents=True, exist_ok=True)
    Image.fromarray(r2.overlay(res)).save(out / "overlays" / (Path(v).stem + ".jpg"), quality=85)
    lab = res.pop("_layers")["label"]
    Image.fromarray(lab.astype(np.uint16)).save(out / "labels" / (Path(v).stem + ".png"))
    json.dump(dict(schema="leafseg-1", image=dict(name=v, size_hw=[H, W], grid_hw=list(gshape), grid_scale=s), source="model_proposal",
                   note="H2 on the A3 ChArUco capture (Step 1a foreground, no Roboflow detections); no accuracy claim", floor_px=F, log=log, **res),
              open(out / "proposals" / (Path(v).stem + ".json"), "w"), default=float)
    insts = res["instances"]
    return dict(m1_instances=len(mj["instances"]), m1_on_plant=len(m1), leaflets=len(insts), occluded=sum(i["occluded"] for i in insts),
                order_edges=len(res["order_edges"]), merged=sum("merged" in i["flags"] for i in insts), fallback=sum(i["source"] == "M1_fallback" for i in insts),
                unassigned_area_fraction=round(res["unassigned_area_fraction"], 3), floor_px=F, seconds=round(time.time() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--recon", required=True); ap.add_argument("--views", nargs="*")
    a = ap.parse_args(argv); src = Path(a.poses); rc = Path(a.recon); out = rc / "leaflets"; out.mkdir(parents=True, exist_ok=True)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    from ultralytics import YOLO
    from tomato3d.topview.measure2d import load_sam
    model = YOLO(str(hy.M1_WEIGHTS)); sam = load_sam(r2.SAMCFG); cfg = json.load(open(r2.T1CFG))
    sp = out / "leaflets_summary.json"; summ = json.load(open(sp)) if sp.exists() else {}
    summ.setdefault("_params", dict(m1_weights=str(hy.M1_WEIGHTS), m1_sha256=hy.sha(hy.M1_WEIGHTS), sam_config=str(r2.SAMCFG), t1_config=str(r2.T1CFG),
                                    m1_score=hy.M1_SCORE, grid=GRID, floor_frac=FLOOR_FRAC, variant="H2", undistorted_res=R0))
    for v in (a.views or po["accepted"]):
        if v in summ and (out / "labels" / (Path(v).stem + ".png")).exists():
            continue
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB"))
        plant = np.asarray(Image.open(rc / "masks" / (Path(v).stem + ".png")).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) == 1
        summ[v] = run_view(v, rgb, plant, model, sam, cfg, out); print(v, summ[v], flush=True)
        json.dump(summ, open(sp, "w"), indent=1)


if __name__ == "__main__":
    main()
