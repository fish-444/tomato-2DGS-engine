"""A2 contact sheets: 12 views, stem before (run 01 colour-in-hull) and after (whole-plant mask), plus an overview of all views.

usage: python -m tomato3d.recon.wholeplant_sheets --poses <charuco dir> --run01 <run 01 dir> --out <run dir>
Writes <run>/sheets/stem_before_after.jpg and <run>/sheets/overview_<n>.jpg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon.charuco_masks import MASK_RES

TILE = 440
PINK = np.array([255, 0, 190]); CYAN = np.array([0, 240, 255])


def crop(im: np.ndarray, m: np.ndarray, pad: float = 0.22) -> np.ndarray:
    ys, xs = np.nonzero(m)
    if not len(xs):
        return cv2.resize(im, (TILE, TILE))
    p = int(pad * max(np.ptp(xs), np.ptp(ys))) + 20
    y0, y1 = max(0, ys.min() - p), min(im.shape[0], ys.max() + p); x0, x1 = max(0, xs.min() - p), min(im.shape[1], xs.max() + p)
    return cv2.resize(im[y0:y1, x0:x1], (TILE, TILE))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--run01", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src, r01, out = Path(a.poses), Path(a.run01), Path(a.out); (out / "sheets").mkdir(parents=True, exist_ok=True)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    con = json.load(open(out / "contract.json")); held = set(con["heldout_new"])
    views = [v for v in po["accepted"] if (out / "masks" / (Path(v).stem + ".png")).exists()]
    summ = json.load(open(out / "masks_summary.json"))
    pick = sorted(views, key=lambda v: -summ[v].get("added_share", 0))[:6] + views[::max(1, len(views) // 6)][:6]
    seen, sel = set(), []
    for v in pick:
        if v not in seen:
            seen.add(v); sel.append(v)
    rows = []
    for v in sel[:12]:
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR))
        new = np.asarray(Image.open(out / "masks" / (Path(v).stem + ".png")))
        old = np.asarray(Image.open(r01 / "masks" / (Path(v).stem + ".png")))
        b = rgb.astype(float).copy(); b[old == 1] = b[old == 1] * 0.5 + PINK * 0.5
        c = rgb.astype(float).copy(); c[new == 1] = c[new == 1] * 0.5 + PINK * 0.5; c[new == 2] = c[new == 2] * 0.2 + CYAN * 0.8
        m = (new == 1) | (new == 2)
        t = np.hstack([crop(b.astype(np.uint8), m), crop(c.astype(np.uint8), m)])
        cv2.putText(t, f"{v[9:18]}{' HELD-OUT' if v in held else ''}  +{summ[v].get('added_share', 0):.1%}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        rows.append(t)
    sheet = np.vstack([np.hstack(rows[k:k + 2]) for k in range(0, len(rows) - len(rows) % 2, 2)])
    cv2.putText(sheet, "left: run 01 colour mask   right: whole-plant mask (cyan = added stem/petiole)", (8, sheet.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    Image.fromarray(sheet).save(out / "sheets" / "stem_before_after.jpg", quality=86)
    small = []
    for v in views:
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR)).astype(float)
        new = np.asarray(Image.open(out / "masks" / (Path(v).stem + ".png")))
        rgb[new == 1] = rgb[new == 1] * 0.5 + PINK * 0.5; rgb[new == 2] = rgb[new == 2] * 0.2 + CYAN * 0.8
        t = crop(rgb.astype(np.uint8), (new == 1) | (new == 2))
        cv2.putText(t, v[9:18] + (" H" if v in held else ""), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        small.append(cv2.resize(t, (TILE // 2, TILE // 2)))
    per = 24
    for k in range(0, len(small), per):
        t = small[k:k + per]; t += [np.zeros_like(small[0])] * ((-len(t)) % 6)
        Image.fromarray(np.vstack([np.hstack(t[j:j + 6]) for j in range(0, len(t), 6)])).save(out / "sheets" / f"overview_{k // per + 1}.jpg", quality=82)
    print(len(sel[:12]), "before/after tiles,", len(views), "views")


if __name__ == "__main__":
    main()
