"""R3-1 Step 1: build the mask-correction page for the 4 fresh held-out views (proposal = A2 whole-plant mask; click-add regions =
SAM v5 candidates from a point grid inside the plant box at CAND_RES). The page stores the user's masks in its db; nothing here
writes human truth.

usage: python -m tomato3d.recon.r31_maskpage --poses <charuco dir> --v2 <wholeplant run dir> --out <r3-1 run dir>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon.charuco_masks import MASK_RES

TEMPLATE = Path(__file__).with_name("r31_maskpage_template.html")
CAND_RES, GRID, IOU_MIN, DEDUP_IOU = 756, 14, 0.6, 0.7


def rle(a: np.ndarray) -> list:
    """Run lengths alternating 0-run, 1-run, 0-run, ... (a leading 0 when the array starts with 1); decoded by the page."""
    a = a.astype(np.uint8).ravel(); idx = np.nonzero(np.diff(a))[0] + 1
    runs = np.diff(np.r_[0, idx, len(a)]).tolist()
    return runs if a[0] == 0 else [0] + runs


def candidates(sam, rgb: np.ndarray, plant: np.ndarray) -> list[dict]:
    ys, xs = np.nonzero(plant); pad = int(0.15 * max(np.ptp(xs), np.ptp(ys)))
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1] - 1, xs.max() + pad); y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0] - 1, ys.max() + pad)
    sam.predictor.set_image(rgb); out = []
    for gy in np.linspace(y0, y1, GRID):
        for gx in np.linspace(x0, x1, GRID):
            masks, ious, _ = sam.predictor.predict(point_coords=np.array([[gx, gy]], np.float32), point_labels=np.array([1], np.int32), multimask_output=True)
            for m, q in zip(masks, ious):
                m = m > 0; a = int(m.sum())
                if q < IOU_MIN or a < 200 or a > 0.5 * plant.size:
                    continue
                small = cv2.resize(m.astype(np.uint8), (CAND_RES, CAND_RES), interpolation=cv2.INTER_NEAREST) > 0
                dup = False
                for c in out:
                    inter = (c["m"] & small).sum()
                    if inter / max((c["m"] | small).sum(), 1) >= DEDUP_IOU:
                        dup = True; break
                if not dup:
                    out.append(dict(m=small, area=int(small.sum()), iou=float(q)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--v2", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src, v2, out = Path(a.poses), Path(a.v2), Path(a.out)
    con = json.load(open(out / "contract.json")); po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    qc = {r["file"]: r for r in __import__("csv").DictReader(open(src / "qc.csv"))}
    from tomato3d.leafseg import run as r2
    from tomato3d.topview.measure2d import load_sam
    sam = load_sam(r2.SAMCFG); views = {}
    for h in con["heldout_detail"]:
        v = h["file"]; rgb3 = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB"))
        rgb = cv2.resize(rgb3, (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA)
        lab = np.asarray(Image.open(v2 / "masks" / (Path(v).stem + ".png"))); plant = np.isin(lab, (1, 2))
        cands = candidates(sam, rgb, plant)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=86)
        views[v] = dict(jpg=base64.b64encode(buf.getvalue()).decode(), proposal=rle(plant), ring=h["ring"], az=h["azimuth_deg"], el=h["elevation_deg"],
                        candidates=[dict(rle=rle(c["m"]), area=c["area"]) for c in sorted(cands, key=lambda c: c["area"])])
        print(v[9:18], "proposal px", int(plant.sum()), "candidates", len(cands), flush=True)
    data = dict(res=MASK_RES, cand_res=CAND_RES, views=views, note="input: fresh held-out views of R3-1; proposals are A2 masks")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (out / "review").mkdir(exist_ok=True); p = out / "review" / "mask_correction.html"; p.write_text(html, encoding="utf-8")
    print(p, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
