"""R3-1 Step 3, I2: monocular depth prior. Depth Anything V2 Small (HF `depth-anything/Depth-Anything-V2-Small-hf`, relative
inverse depth) per TRAINING view at DA_RES; per view an affine fit (scale, shift) of the model's inverse depth to 1/z of the I1
points visible in that view (RANSAC-trimmed least squares); back-projected points are added only in views with fewer than N_MIN
I1 points, only inside the Step-2 mask, subsampled on a grid, tagged mono_prior.

usage: python -m tomato3d.recon.r31_depth --poses <charuco dir> --out <r3-1 run dir> [--masks masks_a] [--points plant_points_a.ply]
Writes <out>/plant_points_mono.ply (I1 + mono_prior), depth_summary.json, depth/<view>.npz (relative depth at DA_RES).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import MASK_RES, project
from tomato3d.recon.charuco_prep import read_xyz, write_xyzrgb
from tomato3d.recon.r31_contract import M

DA_RES, GRID = 1008, 6                      # inverse-depth map size; one mono point per GRID x GRID pixels inside the mask
N_MIN, REL_MAX, MIN_FIT = M["I2"]["n_min"], M["I2"]["rel_residual_max"], 30


def fit_affine(inv_model, inv_true, iters=200, seed=0):
    """inv_true ~ s * inv_model + b, RANSAC on 2-point samples, then least squares on the inliers (|rel residual| <= REL_MAX)."""
    rng = np.random.default_rng(seed); best = None
    for _ in range(iters):
        i, j = rng.choice(len(inv_model), 2, replace=False)
        if abs(inv_model[i] - inv_model[j]) < 1e-9:
            continue
        s = (inv_true[i] - inv_true[j]) / (inv_model[i] - inv_model[j]); b = inv_true[i] - s * inv_model[i]
        res = np.abs((s * inv_model + b) - inv_true) / inv_true; inl = res <= REL_MAX
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), inl)
    if best is None or best[0] < MIN_FIT:
        return None
    inl = best[1]; A = np.c_[inv_model[inl], np.ones(inl.sum())]; s, b = np.linalg.lstsq(A, inv_true[inl], rcond=None)[0]
    res = np.abs((s * inv_model + b) - inv_true) / inv_true
    return dict(s=float(s), b=float(b), inliers=int(inl.sum()), rel_med=float(np.median(res[inl])))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--masks", default="masks_a"); ap.add_argument("--points", default="plant_points_a.ply")
    a = ap.parse_args(argv); src, out = Path(a.poses), Path(a.out); (out / "depth").mkdir(exist_ok=True)
    con = json.load(open(out / "contract.json")); train = con["train"]; po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    P1, C1 = read_xyz(out / a.points); K = ca.scaled_K(np.array(po["K_undist"]), R0, DA_RES); Km = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES)
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    added, cols, summ = [], [], {}
    # visibility = I1 observations per view from the triangulation (not "projects into the frame", which every point does)
    seen = json.load(open(out / a.points.replace("plant_points_", "sparse_").replace(".ply", "_checks.json")))["points_per_view"]
    for v in train:
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"])
        u, w, z = project(P1, R, t, K); ok = (z > 0) & (u >= 0) & (w >= 0) & (u < DA_RES) & (w < DA_RES); n1 = int(seen.get(v, 0))
        m0 = np.isin(np.asarray(Image.open(out / a.masks / (Path(v).stem + ".png"))), (1, 2))
        ui, wi = u.astype(int).clip(0, DA_RES - 1), w.astype(int).clip(0, DA_RES - 1)
        ok &= m0[wi * MASK_RES // DA_RES, ui * MASK_RES // DA_RES]
        cell = (wi // 8) * (DA_RES // 8 + 1) + ui // 8; front = np.zeros(len(z), bool); idx = np.nonzero(ok)[0]   # front-most point per 8 px cell
        if len(idx):
            o = idx[np.lexsort((z[idx], cell[idx]))]; first = np.r_[True, cell[o][1:] != cell[o][:-1]]; front[o[first]] = True
        ok &= front
        dp = out / "depth" / (Path(v).stem + ".npz")
        if dp.exists():
            inv = np.load(dp)["inv"]
        else:
            im = Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((DA_RES, DA_RES), Image.LANCZOS)
            inv = np.asarray(pipe(im)["predicted_depth"], np.float32)
            if inv.shape != (DA_RES, DA_RES):
                inv = np.asarray(Image.fromarray(inv).resize((DA_RES, DA_RES), Image.BILINEAR), np.float32)
            np.savez_compressed(dp, inv=inv)
        rec = dict(i1_points=n1, fit_points=int(ok.sum()), fit=None, mono_added=0)
        if ok.sum() >= MIN_FIT:
            fit = fit_affine(inv[w[ok].astype(int), u[ok].astype(int)], 1.0 / z[ok]); rec["fit"] = fit
            if fit and n1 < N_MIN:
                mask = np.isin(np.asarray(Image.open(out / a.masks / (Path(v).stem + ".png"))), (1, 2))
                ys, xs = np.mgrid[GRID // 2:DA_RES:GRID, GRID // 2:DA_RES:GRID]; ys, xs = ys.ravel(), xs.ravel()
                mu, mw = (xs * MASK_RES // DA_RES), (ys * MASK_RES // DA_RES); keep = mask[mw, mu]
                xs, ys = xs[keep], ys[keep]; zi = fit["s"] * inv[ys, xs] + fit["b"]; good = zi > 1e-3
                zz = 1.0 / zi[good]; xc = np.c_[(xs[good] - K[0, 2]) / K[0, 0] * zz, (ys[good] - K[1, 2]) / K[1, 1] * zz, zz]
                Xw = (xc - t) @ R; rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((DA_RES, DA_RES), Image.BILINEAR))
                added.append(Xw.astype(np.float32)); cols.append(rgb[ys[good], xs[good]].astype(np.float32) / 255); rec["mono_added"] = int(len(Xw))
        summ[v] = rec; print(v[9:18], rec, flush=True)
    Pm = np.vstack(added) if added else np.zeros((0, 3), np.float32); Cm = np.vstack(cols) if cols else np.zeros((0, 3), np.float32)
    write_xyzrgb(out / "plant_points_mono.ply", np.vstack([P1, Pm]), np.vstack([C1, Cm]))
    tag = np.r_[np.zeros(len(P1), np.uint8), np.ones(len(Pm), np.uint8)]; np.save(out / "plant_points_mono_tag.npy", tag)
    if len(Pm):
        write_xyzrgb(out / "plant_points_mono_only.ply", Pm, Cm)  # R3-1c depth prior target (tagged mono_prior)
    json.dump(dict(model="depth-anything/Depth-Anything-V2-Small-hf", da_res=DA_RES, n_min=N_MIN, rel_residual_max=REL_MAX, grid=GRID,
                   i1_points=int(len(P1)), mono_points=int(len(Pm)), views=summ), open(out / "depth_summary.json", "w"), indent=1)
    print("I1", len(P1), "+ mono_prior", len(Pm))


if __name__ == "__main__":
    main()
