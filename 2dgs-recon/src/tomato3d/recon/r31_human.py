"""Assemble the user's saved masks (artifact db export, never edited) into human_masks.json and a comparison sheet vs the proposals.

usage: python -m tomato3d.recon.r31_human --poses <charuco dir> --v2 <wholeplant run dir> --out <r3-1 run dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon.charuco_masks import MASK_RES
from tomato3d.recon.r31_score import rle_decode


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--v2", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src, v2, out = Path(a.poses), Path(a.v2), Path(a.out)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]; human, rows, tiles = {}, {}, []
    for f in sorted((out / "human" / "masks").glob("*.json")):
        d = json.load(open(f, encoding="utf-8")); v = d["view"]; assert d["res"] == MASK_RES
        human[v] = dict(mask=d["mask"], band=d.get("band", [MASK_RES * MASK_RES]), res=MASK_RES, saved_at=d.get("at"))
        hm = rle_decode(d["mask"], MASK_RES * MASK_RES).reshape(MASK_RES, MASK_RES) > 0
        hb = rle_decode(human[v]["band"], MASK_RES * MASK_RES).reshape(MASK_RES, MASK_RES) > 0
        auto = np.isin(np.asarray(Image.open(v2 / "masks" / (Path(v).stem + ".png"))), (1, 2))
        rows[v] = dict(human_px=int(hm.sum()), band_px=int(hb.sum()), proposal_px=int(auto.sum()), added_px=int((hm & ~auto).sum()), removed_px=int((auto & ~hm).sum()),
                       iou_auto_vs_human=round(float((hm & auto).sum() / max((hm | auto).sum(), 1)), 4))
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR)).astype(float)
        o = rgb.copy(); o[hm & auto] = o[hm & auto] * 0.55 + np.array([255, 0, 190]) * 0.45; o[hm & ~auto] = o[hm & ~auto] * 0.2 + np.array([0, 240, 255]) * 0.8
        o[auto & ~hm] = o[auto & ~hm] * 0.2 + np.array([255, 40, 40]) * 0.8; o[hb] = o[hb] * 0.5 + np.array([255, 220, 0]) * 0.5
        ys, xs = np.nonzero(hm | auto); p = int(0.15 * max(np.ptp(xs), np.ptp(ys))) + 20
        y0, y1 = max(0, ys.min() - p), min(MASK_RES, ys.max() + p); x0, x1 = max(0, xs.min() - p), min(MASK_RES, xs.max() + p)
        t = cv2.resize(o.astype(np.uint8)[y0:y1, x0:x1], (520, 520)); cv2.putText(t, f"{v[9:18]} +{rows[v]['added_px']} -{rows[v]['removed_px']} band {rows[v]['band_px']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(t)
    json.dump(human, open(out / "human_masks.json", "w"))
    json.dump(dict(source="artifact db export, user-corrected; proposals = A2 masks", views=rows), open(out / "human_masks_summary.json", "w"), indent=1)
    sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles) - len(tiles) % 2, 2)])
    cv2.putText(sheet, "human vs proposal: pink both, cyan human only (added), red proposal only (removed), yellow user band", (8, sheet.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    (out / "sheets").mkdir(exist_ok=True); Image.fromarray(sheet).save(out / "sheets" / "human_masks.jpg", quality=88)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
