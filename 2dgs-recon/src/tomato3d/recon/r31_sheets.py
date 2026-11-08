"""R3-1 Step 2 contact sheet: 12 training views, A2 -> stage masks; added pixels cyan, removed pixels red, band yellow.

usage: python -m tomato3d.recon.r31_sheets --poses <charuco dir> --v2 <wholeplant run dir> --out <r3-1 run dir> --stage a|b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon.charuco_masks import MASK_RES

TILE = 430


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--v2", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True); a = ap.parse_args(argv); src, v2, out = Path(a.poses), Path(a.v2), Path(a.out)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]; summ = json.load(open(out / f"masks_{a.stage}_summary.json"))
    views = [v for v in summ]; pick = sorted(views, key=lambda v: -summ[v]["change_share"])[:6] + views[::max(1, len(views) // 6)][:6]
    sel = list(dict.fromkeys(pick))[:12]; tiles = []
    for v in sel:
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR))
        old = np.isin(np.asarray(Image.open(v2 / "masks" / (Path(v).stem + ".png"))), (1, 2))
        new = np.isin(np.asarray(Image.open(out / f"masks_{a.stage}" / (Path(v).stem + ".png"))), (1, 2))
        band = np.asarray(Image.open(out / f"bands_{a.stage}" / (Path(v).stem + ".png"))) > 0
        o = rgb.astype(float); o[new & old] = o[new & old] * 0.55 + np.array([255, 0, 190]) * 0.45
        o[new & ~old] = o[new & ~old] * 0.2 + np.array([0, 240, 255]) * 0.8; o[old & ~new] = o[old & ~new] * 0.2 + np.array([255, 40, 40]) * 0.8
        o[band & ~(new & ~old) & ~(old & ~new)] = o[band & ~(new & ~old) & ~(old & ~new)] * 0.6 + np.array([255, 220, 0]) * 0.4
        ys, xs = np.nonzero(new | old); p = int(0.2 * max(np.ptp(xs), np.ptp(ys))) + 20
        y0, y1 = max(0, ys.min() - p), min(MASK_RES, ys.max() + p); x0, x1 = max(0, xs.min() - p), min(MASK_RES, xs.max() + p)
        t = np.hstack([cv2.resize(rgb[y0:y1, x0:x1], (TILE, TILE)), cv2.resize(o.astype(np.uint8)[y0:y1, x0:x1], (TILE, TILE))])
        s = summ[v]; cv2.putText(t, f"{v[9:18]}  +{s.get('m2_added_px', 0) + s.get('m3_added_px', 0) + s.get('m1_added_px', 0) + s.get('m5_added_px', 0)} -{s.get('m3_removed_px', 0)} px",
                                 (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles) - len(tiles) % 2, 2)])
    cv2.putText(sheet, f"stage {a.stage}: pink = kept, cyan = added, red = removed, yellow = 2 px band", (8, sheet.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    (out / "sheets").mkdir(exist_ok=True); Image.fromarray(sheet).save(out / "sheets" / f"masks_stage_{a.stage}.jpg", quality=86); print(len(sel), "tiles")


if __name__ == "__main__":
    main()
