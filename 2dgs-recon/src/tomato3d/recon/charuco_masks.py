"""Step 1a: plant foreground for every accepted view = colour mask (Lab a* < A_MAX) restricted to the projection of a
multi-view-consistent visual hull carved with the known board poses; board plane projected and labelled.

Labels (PNG, MASK_RES px): 0 background, 1 plant, 3 board (projected board rectangle, not plant).
The hull is carved from the TRAINING views only (contract.json "heldout" excluded), so held-out views add nothing to it;
masks are written for every accepted view.
usage: python -m tomato3d.recon.charuco_masks --poses <charuco dir> --out <recon dir>   (reads <recon dir>/contract.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca

MASK_RES = 1512
A_MAX, L_MIN, OPEN_PX = -4.0, 8.0, 2           # colour rule on the undistorted image (Lab, a* in [-128, 127], L in [0, 100])
HULL_DILATE_PX = 6                            # colour mask dilation before voting (tolerates thin misses)
HULL_KEEP = 0.85                              # voxel kept if inside the dilated colour mask in >= this share of the views that see it
HULL_MIN_VIEWS = 8
COARSE_M, FINE_M, ZMAX_M = 0.004, 0.0015, 0.40
FINAL_DILATE_PX = 4                           # hull projection dilation before intersecting with the colour mask


def colour_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = ((lab[..., 1] - 128) < A_MAX) & (lab[..., 0] * 100 / 255 > L_MIN)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OPEN_PX + 1, 2 * OPEN_PX + 1))
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0


def project(P: np.ndarray, R, t, K):
    xc = P @ np.asarray(R).T + np.asarray(t); z = xc[:, 2]
    u = K[0, 0] * xc[:, 0] / z + K[0, 2]; v = K[1, 1] * xc[:, 1] / z + K[1, 2]
    return u, v, z


def carve(lo, hi, step, cams, masks, res) -> np.ndarray:
    gx, gy, gz = [np.arange(l + step / 2, h, step) for l, h in zip(lo, hi)]
    P = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    inside = np.zeros(len(P), np.int16); seen = np.zeros(len(P), np.int16)
    for (R, t, K), m in zip(cams, masks):
        u, v, z = project(P, R, t, K); ok = (z > 0) & (u >= 0) & (v >= 0) & (u < res) & (v < res)
        seen += ok; inside[ok] += m[v[ok].astype(int), u[ok].astype(int)]
    keep = (seen >= HULL_MIN_VIEWS) & (inside >= HULL_KEEP * seen)
    return P[keep]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src = Path(a.poses); out = Path(a.out); (out / "masks").mkdir(parents=True, exist_ok=True)
    po = json.load(open(src / "poses.json")); views = po["accepted"]; R0 = po["undistorted_res"]
    held = set(json.load(open(out / "contract.json"))["heldout"]); assert held <= set(views), "held-out view not accepted"
    K = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES); ud = src / f"undist_{R0}"
    rgbs, cols, dil, cams = {}, {}, [], []
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_DILATE_PX + 1,) * 2)
    for v in views:
        rgb = np.asarray(Image.open(ud / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR)); rgbs[v] = rgb
        cols[v] = colour_mask(rgb); dil.append(cv2.dilate(cols[v].astype(np.uint8), kd))
        cams.append((po["views"][v]["R_cw"], po["views"][v]["t_cw"], K))
    W, H = ca.SQUARES[0] * ca.SQUARE_M, ca.SQUARES[1] * ca.SQUARE_M
    tr = [k for k, v in enumerate(views) if v not in held]; ct, dt = [cams[k] for k in tr], [dil[k] for k in tr]
    P = carve((0, 0, 0.0), (W, H, ZMAX_M), COARSE_M, ct, dt, MASK_RES)
    assert len(P), "coarse hull empty"
    lo = np.maximum(P.min(0) - 3 * COARSE_M, [0, 0, 0]); hi = np.minimum(P.max(0) + 3 * COARSE_M, [W, H, ZMAX_M])
    Pf = carve(lo, hi, FINE_M, ct, dt, MASK_RES)
    stats, kf = {}, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FINAL_DILATE_PX + 1,) * 2)
    board = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float32)
    for v, (R, t, Kk) in zip(views, cams):
        u, w, z = project(Pf, R, t, Kk); ok = (z > 0) & (u >= 0) & (w >= 0) & (u < MASK_RES) & (w < MASK_RES)
        hp = np.zeros((MASK_RES, MASK_RES), np.uint8); hp[w[ok].astype(int), u[ok].astype(int)] = 1
        r = max(1, int(round(FINE_M * Kk[0, 0] / np.median(z[ok])))) if ok.any() else 1     # voxel footprint in pixels
        hp = cv2.dilate(hp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)); hp = cv2.dilate(hp, kf) > 0
        plant = cols[v] & hp
        bu, bv, _ = project(board, R, t, Kk); bm = np.zeros_like(hp, np.uint8)
        cv2.fillConvexPoly(bm, np.round(np.c_[bu, bv]).astype(np.int32), 1)
        lab = np.zeros((MASK_RES, MASK_RES), np.uint8); lab[bm > 0] = 3; lab[plant] = 1
        Image.fromarray(lab).save(out / "masks" / (Path(v).stem + ".png"))
        stats[v] = dict(heldout=v in held, colour_px=int(cols[v].sum()), plant_px=int(plant.sum()), removed_by_hull=round(1 - plant.sum() / max(cols[v].sum(), 1), 3))
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Pf)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    (out / "visual_hull.ply").write_bytes(hdr.encode() + Pf.astype("<f4").tobytes())
    json.dump(dict(params=dict(mask_res=MASK_RES, a_max=A_MAX, l_min=L_MIN, open_px=OPEN_PX, hull_dilate_px=HULL_DILATE_PX, hull_keep=HULL_KEEP,
                               hull_min_views=HULL_MIN_VIEWS, coarse_m=COARSE_M, fine_m=FINE_M, zmax_m=ZMAX_M, final_dilate_px=FINAL_DILATE_PX),
                   hull=dict(carved_from=len(tr), voxels=int(len(Pf)), lo=Pf.min(0).tolist(), hi=Pf.max(0).tolist()), views=stats),
              open(out / "masks_summary.json", "w"), indent=1)
    print(len(Pf), "hull voxels;", Pf.min(0).round(3), Pf.max(0).round(3))


if __name__ == "__main__":
    main()
