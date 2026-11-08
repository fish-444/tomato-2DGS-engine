"""Step 1 contact sheets: per accepted view, three panels - plant foreground, H2 leaflet masks, projected board outline (the
pot is not modelled; it is never green, so it never enters the plant mask). Plus one overview sheet of every view.

usage: python -m tomato3d.recon.charuco_sheets --poses <charuco dir> --recon <recon dir>
Writes <recon>/sheets/<view>.jpg and <recon>/sheets/overview_<n>.jpg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import MASK_RES, project

PANEL = 420


def sheet(rgb, lab, lab_h2, R, t, K, title):
    plant = lab == 1; a = rgb.astype(np.float32)
    p1 = a.copy(); p1[plant] = p1[plant] * 0.35 + np.array([230, 40, 190]) * 0.65; p1[~plant] *= 0.55
    p2 = a * 0.45
    if lab_h2 is not None:
        L = cv2.resize(lab_h2.astype(np.int32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST); ids = np.unique(L[L > 0])
        rng = np.random.default_rng(1); cols = rng.integers(60, 255, (int(L.max()) + 1, 3))
        for i in ids:
            m = L == i; p2[m] = a[m] * 0.35 + cols[i] * 0.65
        p2 = p2.astype(np.uint8)
        for i in ids:
            cs, _ = cv2.findContours((L == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(p2, cs, -1, (255, 255, 255), 1)
    W, H = ca.SQUARES[0] * ca.SQUARE_M, ca.SQUARES[1] * ca.SQUARE_M
    p3 = rgb.copy(); grid = []
    for x in np.linspace(0, W, ca.SQUARES[0] + 1):
        grid.append(np.array([[x, 0, 0], [x, H, 0]]))
    for y in np.linspace(0, H, ca.SQUARES[1] + 1):
        grid.append(np.array([[0, y, 0], [W, y, 0]]))
    for seg in grid:
        P = np.linspace(seg[0], seg[1], 30); u, v, z = project(P, R, t, K)
        pts = np.c_[u, v][z > 0].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(p3, [pts], False, (0, 160, 255), 2)
    out = [cv2.resize(np.asarray(p, np.uint8), (PANEL, PANEL), interpolation=cv2.INTER_AREA) for p in (p1, p2, p3)]
    s = np.hstack(out); cv2.putText(s, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--recon", required=True)
    a = ap.parse_args(argv); src, rc = Path(a.poses), Path(a.recon); out = rc / "sheets"; out.mkdir(exist_ok=True)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]; K = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES)
    held = set(json.load(open(rc / "contract.json"))["heldout"]); small = []
    for v in po["accepted"]:
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR))
        lab = np.asarray(Image.open(rc / "masks" / (Path(v).stem + ".png")))
        lp = rc / "leaflets" / "labels" / (Path(v).stem + ".png"); lh = np.asarray(Image.open(lp)) if lp.exists() else None
        s = sheet(rgb, lab, lh, po["views"][v]["R_cw"], po["views"][v]["t_cw"], K, v[9:18] + (" HELD-OUT" if v in held else ""))
        Image.fromarray(s).save(out / (Path(v).stem + ".jpg"), quality=85); small.append(cv2.resize(s, (s.shape[1] // 2, s.shape[0] // 2)))
    per = 24
    for k in range(0, len(small), per):
        tiles = small[k:k + per]; tiles += [np.zeros_like(small[0])] * ((-len(tiles)) % 2)
        Image.fromarray(np.vstack([np.hstack(tiles[j:j + 2]) for j in range(0, len(tiles), 2)])).save(out / f"overview_{k // per + 1}.jpg", quality=80)
    print(len(small), "sheets")


if __name__ == "__main__":
    main()
