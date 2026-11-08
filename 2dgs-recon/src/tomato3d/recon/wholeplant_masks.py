"""A2: whole-plant masks (leaves + stem + petioles + trusses as ONE class; no organ split) and the 2 px edge band.

Stem recovery, as declared in CONTRACT.md:
  1. ridge filter (sato) on the luminance inside the plant's dilated convex hull -> thin bright structures;
  2. seeds = ridge maxima that are not already plant and pass a colour gate (a* <= A_SEED_MAX: the brown soil and the
     achromatic board are excluded; the stem is grey-green), plus the stem base (pot centre projected from the poses);
  3. SAM v5 once per view: padded plant box, those seeds as positives, soil/board points as negatives;
  4. the added region passes the same colour gate (a* <= ADD_A_MAX, L >= ADD_L_MIN) and is kept only where it is THIN
     (morphological opening with a disc of THIN_PX removes the soil and pot blobs) and touches the existing plant mask.
     This is what makes the step add stems and not the soil surface;
  5. pass 2: a visual hull is carved from the whole-plant masks of the TRAINING views only (same rule as run 01) and every
     view's mask is intersected with its hull projection, so additions that no other view supports are dropped.

Labels (PNG, 1512 px): 0 background, 1 plant (colour-in-hull from run 01), 2 plant added by this step (stem/petioles),
3 board. Band: 0 inside/outside, 1 band (uint8 in band_<view>.png).

usage: python -m tomato3d.recon.wholeplant_masks --poses <charuco dir> --run01 <run 01 dir> --out <run dir> [--views ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import FINE_M, HULL_DILATE_PX as HULL_KEEP_DILATE, HULL_KEEP, HULL_MIN_VIEWS, MASK_RES, carve, project
from tomato3d.recon.wholeplant_contract import A2

RES = 3024                      # stem work runs at the undistorted resolution; labels are written at MASK_RES
HULL_DILATE_PX = 90             # convex hull of the plant mask, dilated, bounds the search for stems
RIDGE_SIGMAS = (3, 6, 10)       # stem/petiole half-widths at 3024 px
RIDGE_PCT = 97.0                # ridge response percentile inside the hull
A_SEED_MAX = 3.0                # Lab a* gate for seeds: brown soil and board are rejected
L_SEED = (12.0, 92.0)
N_SEED, N_NEG = 24, 6
ADD_A_MAX, ADD_L_MIN = 2.0, 15.0  # colour gate on the ADDED region: brown soil (a* ~ +7) and dark gaps are rejected
                                  # (measured on this capture: leaves a* ~ -8, stem -3..0, soil +7)
THIN_PX = 26                    # a disc of this radius; wider structures (soil, pot) are dropped
TOUCH_PX = 6                    # the added part must touch the existing plant mask
BAND_PX = A2["band_px"]


def ridge(gray: np.ndarray) -> np.ndarray:
    from skimage.filters import sato
    s = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    r = sato(s, sigmas=[x / 2 for x in RIDGE_SIGMAS], black_ridges=False).astype(np.float32)
    return cv2.resize(r, (gray.shape[1], gray.shape[0]))


def seeds(rgb: np.ndarray, plant: np.ndarray, rng) -> np.ndarray:
    ys, xs = np.nonzero(plant)
    hull = np.zeros(plant.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.c_[xs, ys]), 1)
    hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_DILATE_PX + 1,) * 2)) > 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255; A = lab[..., 1] - 128
    r = ridge(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    cand = hull & ~plant & (A <= A_SEED_MAX) & (L > L_SEED[0]) & (L < L_SEED[1])
    if not cand.any():
        return np.zeros((0, 2), np.float32)
    thr = np.percentile(r[cand], RIDGE_PCT)
    pts = np.argwhere(cand & (r >= thr))
    if not len(pts):
        return np.zeros((0, 2), np.float32)
    # spread the seeds: farthest-point sampling
    sel = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    while len(sel) < min(N_SEED, len(pts)):
        k = int(np.argmax(d)); sel.append(k); d = np.minimum(d, np.linalg.norm(pts - pts[k], axis=1))
    return pts[sel][:, ::-1].astype(np.float32)


def negatives(R, t, K, ctr, res) -> np.ndarray:
    out = []
    q = [np.array([[ctr[0], ctr[1], 0.004]])]                      # soil / pot centre at board level
    for d in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1)):
        q.append(np.array([[ctr[0] + d[0] * 0.095, ctr[1] + d[1] * 0.075, 0.0]]))   # board, outside the plant
    for p in q[:N_NEG]:
        u, w, z = project(p, R, t, K)
        if z[0] > 0 and 0 < u[0] < res and 0 < w[0] < res:
            out.append([u[0], w[0]])
    return np.array(out, np.float32).reshape(-1, 2)


def thin_parts(add: np.ndarray, plant: np.ndarray) -> np.ndarray:
    """Drop wide blobs (soil, pot) and keep thin structures that touch the plant."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * THIN_PX + 1,) * 2)
    wide = cv2.morphologyEx(add.astype(np.uint8), cv2.MORPH_OPEN, k)
    thin = (add & ~(cv2.dilate(wide, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0)).astype(np.uint8)
    near = cv2.dilate(plant.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOUCH_PX + 1,) * 2)) > 0
    n, lab = cv2.connectedComponents(thin)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        m = lab == i
        if m.sum() >= 150 and (m & near).any():
            keep[i] = True
    return keep[lab]


def band(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2); m = mask.astype(np.uint8)
    return (cv2.dilate(m, k) - cv2.erode(m, k)) > 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--run01", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--views", nargs="*")
    a = ap.parse_args(argv); src, r01, out = Path(a.poses), Path(a.run01), Path(a.out)
    for d in ("masks", "bands"):
        (out / d).mkdir(parents=True, exist_ok=True)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    K = ca.scaled_K(np.array(po["K_undist"]), R0, RES)
    held = set(json.load(open(out / "contract.json"))["heldout_new"])
    raw_dir = out / "_raw"; raw_dir.mkdir(exist_ok=True)
    hull = json.load(open(r01 / "masks_summary.json"))["hull"]; ctr = (np.array(hull["lo"]) + np.array(hull["hi"])) / 2
    from tomato3d.leafseg import run as r2
    from tomato3d.topview.measure2d import load_sam
    sam = load_sam(r2.SAMCFG)
    sp = out / "masks_summary.json"; summ = json.load(open(sp)) if sp.exists() else {}
    summ.setdefault("_params", dict(res=RES, hull_dilate_px=HULL_DILATE_PX, ridge_sigmas=RIDGE_SIGMAS, ridge_pct=RIDGE_PCT,
                                    a_seed_max=A_SEED_MAX, l_seed=L_SEED, n_seed=N_SEED, add_a_max=ADD_A_MAX, add_l_min=ADD_L_MIN,
                                    thin_px=THIN_PX, touch_px=TOUCH_PX,
                                    band_px=BAND_PX, sam_config=str(r2.SAMCFG), source_masks=str(r01 / "masks")))
    rng = np.random.default_rng(0)
    for v in (a.views or po["accepted"]):
        if v in summ:
            continue
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((RES, RES), Image.BILINEAR))
        lab0 = np.asarray(Image.open(r01 / "masks" / (Path(v).stem + ".png")).resize((RES, RES), Image.NEAREST))
        plant = lab0 == 1
        if plant.sum() < 500:
            summ[v] = dict(skipped="no plant mask"); continue
        pos = seeds(rgb, plant, rng)
        ys, xs = np.nonzero(plant); pad = 0.06 * max(np.ptp(xs), np.ptp(ys))
        box = np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], np.float32)
        dt = cv2.distanceTransform(plant.astype(np.uint8), cv2.DIST_L2, 5)
        py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
        pos = np.vstack([[[px, py]], pos]) if len(pos) else np.array([[px, py]], np.float32)
        neg = negatives(po["views"][v]["R_cw"], po["views"][v]["t_cw"], K, ctr, RES)
        sam.predictor.set_image(rgb)
        pts = np.vstack([pos, neg]) if len(neg) else pos
        lbl = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
        masks, ious, _ = sam.predictor.predict(box=box, point_coords=pts, point_labels=lbl, multimask_output=True)
        best = max(zip(masks, ious), key=lambda m: float(m[1]))
        lb = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        gate = ((lb[..., 1] - 128) <= ADD_A_MAX) & (lb[..., 0] * 100 / 255 >= ADD_L_MIN)
        add = thin_parts((best[0] > 0) & ~plant & gate, plant)
        whole = plant | add
        lab = np.zeros((RES, RES), np.uint8); lab[lab0 == 3] = 3; lab[plant] = 1; lab[add] = 2
        Image.fromarray(cv2.resize(lab, (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST)).save(raw_dir / (Path(v).stem + ".png"))
        summ[v] = dict(sam_iou=float(best[1]), seeds=int(len(pos)), plant_px=int(plant.sum()), added_px_raw=int(add.sum()),
                       added_share_raw=round(float(add.sum() / max(plant.sum(), 1)), 4))
        print(v[9:18], summ[v], flush=True)
        json.dump(summ, open(sp, "w"), indent=1)
    # ---- pass 2: visual hull from the training views' whole-plant masks, then intersect every view with its projection
    views = [v for v in po["accepted"] if (raw_dir / (Path(v).stem + ".png")).exists()]
    Km = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES)
    raw = {v: np.asarray(Image.open(raw_dir / (Path(v).stem + ".png"))) for v in views}
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_KEEP_DILATE + 1,) * 2)
    cams = [(po["views"][v]["R_cw"], po["views"][v]["t_cw"], Km) for v in views]
    dil = [cv2.dilate(((raw[v] == 1) | (raw[v] == 2)).astype(np.uint8), kd) for v in views]
    tr = [k for k, v in enumerate(views) if v not in held]
    W, H = ca.SQUARES[0] * ca.SQUARE_M, ca.SQUARES[1] * ca.SQUARE_M
    P = carve((0, 0, 0.0), (W, H, 0.40), 0.004, [cams[k] for k in tr], [dil[k] for k in tr], MASK_RES)
    assert len(P), "coarse hull empty"
    lo = np.maximum(P.min(0) - 0.012, [0, 0, 0]); hi = np.minimum(P.max(0) + 0.012, [W, H, 0.40])
    Pf = carve(lo, hi, FINE_M, [cams[k] for k in tr], [dil[k] for k in tr], MASK_RES)
    kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for v, (R, t, Kk) in zip(views, cams):
        u, w, z = project(Pf, R, t, Kk); ok = (z > 0) & (u >= 0) & (w >= 0) & (u < MASK_RES) & (w < MASK_RES)
        hp = np.zeros((MASK_RES, MASK_RES), np.uint8); hp[w[ok].astype(int), u[ok].astype(int)] = 1
        r = max(1, int(round(FINE_M * Kk[0, 0] / np.median(z[ok])))) if ok.any() else 1
        hp = cv2.dilate(hp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)); hp = cv2.dilate(hp, kf) > 0
        lab = raw[v].copy(); keep = ((lab == 1) | (lab == 2)) & hp
        lab[((lab == 1) | (lab == 2)) & ~hp] = 0
        Image.fromarray(lab).save(out / "masks" / (Path(v).stem + ".png"))
        Image.fromarray(band(keep, BAND_PX).astype(np.uint8)).save(out / "bands" / (Path(v).stem + ".png"))
        a = int((lab == 2).sum()); pl = int((lab == 1).sum())
        summ[v].update(heldout=v in held, plant_px_final=pl, added_px=a, added_share=round(a / max(pl, 1), 4),
                       hull_removed=round(float(1 - keep.sum() / max(((raw[v] == 1) | (raw[v] == 2)).sum(), 1)), 4),
                       band_share=round(float(band(keep, BAND_PX).sum() / max(keep.sum(), 1)), 4))
    summ["_hull"] = dict(carved_from=len(tr), voxels=int(len(Pf)), lo=Pf.min(0).tolist(), hi=Pf.max(0).tolist())
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Pf)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    (out / "visual_hull.ply").write_bytes(hdr.encode() + Pf.astype("<f4").tobytes())
    json.dump(summ, open(sp, "w"), indent=1)
    import statistics as st
    fin = [x for k, x in summ.items() if not k.startswith("_")]
    print("views", len(fin), "hull voxels", len(Pf), "added share median", st.median(x["added_share"] for x in fin),
          "hull removed median", st.median(x["hull_removed"] for x in fin))


if __name__ == "__main__":
    main()
