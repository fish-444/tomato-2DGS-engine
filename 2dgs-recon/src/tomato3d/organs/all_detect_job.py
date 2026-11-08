"""Detector candidates for the label views of the other individuals (CONTRACT p1_all_compound_20261026_01, Amendment 1, item 3).

Runs on Kaggle (GPU) and, for a smoke test, on the local CPU. Per view it writes RAW candidates of the three detectors on the 1536 grid:
P-H2 (M1 + SAM v5, charuco_leaflets.run_view unchanged), P-SA (SAM 2.1 automatic, userlabel.p_samauto unchanged) and P-RF (cached Roboflow v4
boxes -> SAM drawing, auto_proposers.p_rf rule with the plant mask as fixed foreground). The plant-mask restriction and the UNION are applied
afterwards on the user's PC (all_compound.step_union), so the rules are identical for every individual.
No Roboflow key is read here: only cached responses are used.

usage (inside a folder that holds tomato3d/, tools/ and the job folder):
    python -m tomato3d.organs.all_detect_job --job JOB --out OUT --sam-config sam.json --t1 parameters.json --m1 M1.pt [--ids n_04] [--views a.jpg]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

H2 = 1536


def rle(m):
    from tomato3d.organs import seed2d as S2
    return dict(size=[H2, H2], counts=S2.rle_encode(m))


def rf_masks(path: Path, rgb, stem, cache_root: Path, sam, cfg, plant_grid, log):
    from tomato3d.leafseg import engine as eng
    from tomato3d.leafseg import pixels as px
    from tomato3d.leafseg import rfdet
    from tomato3d.organs import auto_proposers as AP
    t0 = time.time(); pred = rfdet.fetch(path, rgb, cache_root / "rf_detections" / f"{stem}.json", None); n_pred = len(pred.get("predictions", [])); tiled = None
    if n_pred >= AP.RF_CAP:
        dets, tiled = rfdet.tiled_dets(path, rgb, cache_root / "rf_detections_tiled", stem, None)
    else:
        dets = rfdet.to_dets(pred, rgb.shape[:2])
    t1 = time.time()
    with AP.fixed_foreground(plant_grid):
        res = eng.run_image(rgb, None, sam, None, cfg, dets=dets)
    out = [dict(uid=f"RF:{x['id']}", prop="P-RF", mask=px.rle_decode(x["visible_segmentation"]), score=float(x.get("score") or 0.0)) for x in res["instances"]]
    log.update(rf_predictions=n_pred, rf_leaflet_boxes=len(dets), tiled=tiled, sam_seconds=round(time.time() - t1, 1), masks=len(out), dropped=dict(Counter(d["reason"] for d in res.get("dropped", []))))
    return out


def run(job: Path, out: Path, sam_config: Path, t1: Path, m1: Path, ids=None, views=None):
    from ultralytics import YOLO
    from tomato3d.leafseg import hybrid as hy
    from tomato3d.leafseg import run as r2
    from tomato3d.organs import userlabel as UL
    from tomato3d.recon import charuco_leaflets as CL
    from tomato3d.topview.measure2d import load_sam
    r2.SAMCFG = Path(sam_config); r2.T1CFG = Path(t1); hy.M1_WEIGHTS = Path(m1)
    model = YOLO(str(m1)); sam = load_sam(Path(sam_config)); cfg = json.load(open(t1, encoding="utf-8"))
    for iid in (ids or sorted(p.name for p in Path(job).iterdir() if (p / "detect_input.json").exists())):
        d = Path(job) / iid; di = json.load(open(d / "detect_input.json", encoding="utf-8")); o = Path(out) / iid; o.mkdir(parents=True, exist_ok=True)
        for v in (views or di["views"]):
            res_f = o / f"{Path(v).stem}_cands.json"
            if res_f.exists():
                continue
            t0 = time.time(); rgb = np.asarray(Image.open(d / "undist_3024" / v).convert("RGB")); stem = Path(v).stem; log = {}
            plant1512 = np.asarray(Image.open(d / "masks" / f"{stem}.png")) == 1
            plant_full = np.asarray(Image.fromarray(plant1512.astype(np.uint8)).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) == 1
            plant = np.asarray(Image.fromarray(plant1512.astype(np.uint8)).resize((H2, H2), Image.NEAREST)) == 1
            cands = []
            # P-H2
            h2dir = o / "h2"; log["H2"] = CL.run_view(v, rgb, plant_full, model, sam, cfg, h2dir)
            pj = json.load(open(h2dir / "proposals" / f"{stem}.json", encoding="utf-8"))
            from tomato3d.leafseg import pixels as px
            for x in pj["instances"]:
                m = px.rle_decode(x["visible_segmentation"])
                if m.shape == (H2, H2):
                    cands.append(dict(uid=f"H2:{x['id']}", prop="P-H2", score=float(x.get("score") or 0.0), rle=rle(m)))
            # P-SA
            salog = {}; sa = UL.p_samauto(stem, rgb, sam, plant, di["ppm"][v]["px_per_mm"], salog); log["SA"] = salog.get(stem)
            cands += [dict(uid=c["uid"], prop="P-SA", score=c["score"], rle=rle(c["mask"]), shape=c.get("shape")) for c in sa]
            # P-RF (cached boxes)
            rflog = {}
            try:
                rf = rf_masks(d / "undist_3024" / v, rgb, stem, d, sam, cfg, plant, rflog)
                cands += [dict(uid=c["uid"], prop="P-RF", score=c["score"], rle=rle(c["mask"])) for c in rf]
            except (Exception, SystemExit) as e:                                    # recorded, never fatal (a missing cached response raises SystemExit)
                rflog["error"] = f"{type(e).__name__}: {e}"
            log["RF"] = rflog
            json.dump(dict(individual=iid, view=v, candidates=cands, log=log, counts=dict(Counter(c["prop"] for c in cands)), seconds=round(time.time() - t0, 1)), open(res_f, "w", encoding="utf-8"), default=float)
            print(iid, v, dict(Counter(c["prop"] for c in cands)), round(time.time() - t0, 1), "s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("job", "out", "sam-config", "t1", "m1"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--ids", nargs="*"); ap.add_argument("--views", nargs="*"); a = ap.parse_args(argv)
    run(Path(a.job), Path(a.out), Path(a.sam_config), Path(a.t1), Path(a.m1), a.ids, a.views)


if __name__ == "__main__":
    main()
