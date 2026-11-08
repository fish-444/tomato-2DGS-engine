"""R3-1 Step 2 - mask improvements on TRAINING views only (CONTRACT_R3-1.md, methods M1-M5).

Pipeline per training view (labels at 1512: 0 background, 1 plant, 2 plant added by a stem step, 3 board; soft matte 0-255):
  M2  thin-structure prompts: the A2 stem step (tomato3d.recon.wholeplant_masks) re-run on the current mask;
  M3  soil as its own class inside the projected pot-top polygon (per-capture Gaussian colour models, log-likelihood ratio);
  M1  propagation along each ring: the previous view's mask lifted through R3-base's rendered depth, used as a box + point prompt
      for SAM; union with the current mask inside the projected plant box (needs R3-base renders of the training views);
  M5  one 3D->2D round: R3-base alpha >= 0.8 where the mask is 0 and M3 says plant -> SAM positive prompts (training views only);
  M4  alpha matte: trimap (erode/dilate 2 px) + guided filter of the hard mask with the photo as guide.
Stage "a" = M2 + M3 + M4 (no R3-base renders needed); stage "b" adds M1 + M5 then redoes M4.

usage: python -m tomato3d.recon.r31_masks --poses <charuco dir> --v2 <wholeplant run dir> --out <r3-1 run dir> --stage a|b [--renders <dir>]
Writes <out>/masks_<stage>/, bands_<stage>/, soft_<stage>/, masks_<stage>_summary.json and (stage a) the 12-view contact sheet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon import wholeplant_masks as WM
from tomato3d.recon.charuco_masks import MASK_RES, project
from tomato3d.recon.r31_contract import M as METHODS

BAND_PX = 2


# ---------------------------------------------------------------- M3: soil class
def pot_polygon(c, R, t, K, res, side=METHODS["M3"]["rim_side_m"], z=METHODS["M3"]["rim_z_m"]) -> np.ndarray:
    h = side / 2; P = np.array([[c[0] - h, c[1] - h, z], [c[0] + h, c[1] - h, z], [c[0] + h, c[1] + h, z], [c[0] - h, c[1] + h, z]])
    u, w, zz = project(P, R, t, K); m = np.zeros((res, res), np.uint8)
    if (zz > 0).all():
        cv2.fillConvexPoly(m, np.round(np.c_[u, w]).astype(np.int32), 1)
    return m > 0


class Gauss:
    def __init__(self, X):
        self.mu = X.mean(0); C = np.cov(X.T) + np.eye(X.shape[1]) * 1e-3; self.iC = np.linalg.inv(C); self.logdet = np.log(np.linalg.det(C))

    def logp(self, X):
        d = X - self.mu; return -0.5 * (np.einsum("ni,ij,nj->n", d, self.iC, d) + self.logdet)


def soil_split(rgb: np.ndarray, plant: np.ndarray, poly: np.ndarray, min_soil=METHODS["M3"]["min_soil_px"]):
    """Inside the pot polygon: plant vs soil by log-likelihood ratio of two Lab Gaussians. Returns (plant_inside, soil_inside)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); X = lab.reshape(-1, 3)
    soil_px = poly & ~cv2.dilate(plant.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if soil_px.sum() < min_soil or plant.sum() < 500:
        return plant & poly, soil_px
    gs, gp = Gauss(X[soil_px.ravel()]), Gauss(X[plant.ravel()])
    idx = np.nonzero(poly.ravel())[0]; lr = gp.logp(X[idx]) - gs.logp(X[idx])
    out = np.zeros(plant.size, bool); out[idx[lr >= 0]] = True
    return out.reshape(plant.shape), soil_px


# ---------------------------------------------------------------- M4: guided filter matte
def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter, grey guide I in [0,1], input p in [0,1], box radius r."""
    k = (2 * r + 1, 2 * r + 1); box = lambda x: cv2.blur(x, k)  # noqa: E731
    mI, mp = box(I), box(p); cIp = box(I * p) - mI * mp; vI = box(I * I) - mI * mI
    a = cIp / (vI + eps); b = mp - a * mI
    return box(a) * I + box(b)


def matte(rgb: np.ndarray, hard: np.ndarray, band_px=METHODS["M4"]["band_px"], r=METHODS["M4"]["radius_px"], eps=METHODS["M4"]["eps"]):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1,) * 2); m = hard.astype(np.uint8)
    inner = cv2.erode(m, k) > 0; outer = cv2.dilate(m, k) > 0; band = outer & ~inner
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    q = guided_filter(g, hard.astype(np.float32), r, eps)
    alpha = np.where(inner, 1.0, np.where(outer, np.clip(q, 0, 1), 0.0)).astype(np.float32)
    return alpha, band


# ---------------------------------------------------------------- M1 / M5 (need R3-base renders)
def lift_mask(mask_prev, depth_prev, cam_prev, cam_next, res):
    """Back-project the previous view's masked pixels with the rendered median depth, project into the next view."""
    Rp, tp, Kp = cam_prev; Rn, tn, Kn = cam_next
    ys, xs = np.nonzero(mask_prev & np.isfinite(depth_prev) & (depth_prev > 0)); z = depth_prev[ys, xs]
    xc = np.c_[(xs - Kp[0, 2]) / Kp[0, 0] * z, (ys - Kp[1, 2]) / Kp[1, 1] * z, z]
    Xw = (xc - tp) @ Rp; u, w, zz = project(Xw, Rn, tn, Kn)
    ok = (zz > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); m = np.zeros((res, res), np.uint8)
    m[w[ok].astype(int), u[ok].astype(int)] = 1
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0


def sam_from_region(sam, rgb3, region1512, n_pts=METHODS["M1"]["prompt_points"], rng=None):
    """Box + positive points sampled in the region (at 3024) -> best SAM mask at 1512."""
    s = rgb3.shape[0] / MASK_RES; ys, xs = np.nonzero(region1512)
    if len(xs) < 50:
        return np.zeros_like(region1512)
    pad = 0.05 * max(np.ptp(xs), np.ptp(ys)) + 4
    box = np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], np.float32) * s
    dt = cv2.distanceTransform(region1512.astype(np.uint8), cv2.DIST_L2, 5); order = np.argsort(-dt.ravel())[:max(n_pts * 50, 200)]
    rng = rng or np.random.default_rng(0); pick = rng.choice(order, min(n_pts, len(order)), replace=False)
    pts = np.c_[pick % MASK_RES, pick // MASK_RES].astype(np.float32) * s
    masks, ious, _ = sam.predictor.predict(box=box, point_coords=pts, point_labels=np.ones(len(pts), np.int32), multimask_output=True)
    best = max(zip(masks, ious), key=lambda m: float(m[1]))[0] > 0
    return cv2.resize(best.astype(np.uint8), (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST) > 0


# ---------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--v2", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("a", "b"), required=True); ap.add_argument("--renders", default=None); ap.add_argument("--views", nargs="*")
    a = ap.parse_args(argv); src, v2, out = Path(a.poses), Path(a.v2), Path(a.out)
    con = json.load(open(out / "contract.json")); train = con["train"]; po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    K = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES)
    hull = json.load(open(v2 / "masks_summary.json"))["_hull"]; c = (np.array(hull["lo"]) + np.array(hull["hi"])) / 2
    md, bd, sd = out / f"masks_{a.stage}", out / f"bands_{a.stage}", out / f"soft_{a.stage}"
    for d in (md, bd, sd):
        d.mkdir(exist_ok=True)
    from tomato3d.leafseg import run as r2
    from tomato3d.topview.measure2d import load_sam
    sam = load_sam(r2.SAMCFG); rng = np.random.default_rng(0)
    base_dir = v2 / "masks" if a.stage == "a" else out / "masks_a"
    sp = out / f"masks_{a.stage}_summary.json"; summ = json.load(open(sp)) if sp.exists() else {}
    cams = {v: (np.array(po["views"][v]["R_cw"]), np.array(po["views"][v]["t_cw"]), K) for v in train}
    # ring order for M1
    qc = {r["file"]: r for r in __import__("csv").DictReader(open(src / "qc.csv"))}
    ring_of = lambda v: "top" if float(qc[v]["elevation_deg"]) > 75 else "upper" if float(qc[v]["elevation_deg"]) > 58 else "middle" if float(qc[v]["elevation_deg"]) > 45 else "low"  # noqa: E731
    order = sorted(train, key=lambda v: (ring_of(v), float(qc[v]["azimuth_deg"])))
    prev = None
    for v in (a.views or order):
        if v in summ and (md / (Path(v).stem + ".png")).exists():
            prev = v; continue
        rgb3 = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB")); rgb = cv2.resize(rgb3, (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA)
        lab0 = np.asarray(Image.open(base_dir / (Path(v).stem + ".png"))); plant0 = np.isin(lab0, (1, 2)); plant = plant0.copy(); rec = dict(base_px=int(plant0.sum()))
        R, t, _ = cams[v]
        if a.stage == "a":
            # M2: thin-structure prompts (the A2 stem step on the current mask)
            pos = WM.seeds(rgb3, cv2.resize(plant.astype(np.uint8), (WM.RES, WM.RES), interpolation=cv2.INTER_NEAREST) > 0, rng)
            ys, xs = np.nonzero(plant); pad = 0.06 * max(np.ptp(xs), np.ptp(ys)); s3 = WM.RES / MASK_RES
            box = np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], np.float32) * s3
            dt = cv2.distanceTransform(plant.astype(np.uint8), cv2.DIST_L2, 5); py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
            pos = np.vstack([[[px * s3, py * s3]], pos]) if len(pos) else np.array([[px * s3, py * s3]], np.float32)
            neg = WM.negatives(R, t, ca.scaled_K(np.array(po["K_undist"]), R0, WM.RES), c, WM.RES)
            sam.predictor.set_image(rgb3); pts = np.vstack([pos, neg]) if len(neg) else pos
            masks, ious, _ = sam.predictor.predict(box=box, point_coords=pts, point_labels=np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32), multimask_output=True)
            best = max(zip(masks, ious), key=lambda m: float(m[1]))[0] > 0
            big = cv2.resize(plant.astype(np.uint8), (WM.RES, WM.RES), interpolation=cv2.INTER_NEAREST) > 0
            lb = cv2.cvtColor(rgb3, cv2.COLOR_RGB2LAB).astype(np.float32); gate = ((lb[..., 1] - 128) <= WM.ADD_A_MAX) & (lb[..., 0] * 100 / 255 >= WM.ADD_L_MIN)
            add3 = WM.thin_parts(best & ~big & gate, big); add = cv2.resize(add3.astype(np.uint8), (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST) > 0
            plant |= add; rec["m2_added_px"] = int(add.sum())
            # M3: soil class inside the pot polygon
            poly = pot_polygon(c, R, t, K, MASK_RES); pin, soil = soil_split(rgb, plant, poly)
            removed = plant & poly & ~pin; added3 = pin & ~plant
            added3 = cv2.morphologyEx(added3.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            plant = (plant & ~removed) | added3; rec.update(m3_removed_px=int(removed.sum()), m3_added_px=int(added3.sum()), m3_soil_px=int(soil.sum()))
        else:
            rd = Path(a.renders); z = np.load(rd / "renders.npz")
            # M1: propagate from the previous view in ring order through R3-base's rendered depth
            if prev is not None and f"d_{prev}" in z:
                lifted = lift_mask(np.isin(np.asarray(Image.open(md.parent / "masks_a" / (Path(prev).stem + ".png"))), (1, 2)), z[f"d_{prev}"].astype(np.float32), cams[prev], cams[v], MASK_RES)
                sam.predictor.set_image(rgb3); m1 = sam_from_region(sam, rgb3, lifted & ~plant, rng=rng)
                poly_box = pot_polygon(c, R, t, K, MASK_RES, side=0.14, z=0.0)
                ys, xs = np.nonzero(cv2.dilate(plant.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0)
                region = np.zeros_like(plant); region[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                lb = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); gate = ((lb[..., 1] - 128) <= WM.ADD_A_MAX) & (lb[..., 0] * 100 / 255 >= WM.ADD_L_MIN)
                add = m1 & region & gate & ~plant; plant |= add; rec["m1_added_px"] = int(add.sum())
            else:
                sam.predictor.set_image(rgb3); rec["m1_added_px"] = 0
            # M5: one 3D->2D round from R3-base alpha (training views only)
            if f"a_{v}" in z:
                al = z[f"a_{v}"].astype(np.float32); miss = (al >= METHODS["M5"]["alpha_min"]) & ~plant
                poly = pot_polygon(c, R, t, K, MASK_RES); pin, _ = soil_split(rgb, plant, poly | miss)
                cand = miss & (pin | ~poly)
                m5 = sam_from_region(sam, rgb3, cand, rng=rng) if cand.sum() >= 50 else np.zeros_like(plant)
                add = m5 & cv2.dilate(cand.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~plant; plant |= add
                rec.update(m5_candidate_px=int(cand.sum()), m5_added_px=int(add.sum()))
        # M4: matte + band
        plant = cv2.morphologyEx(plant.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        alpha, band = matte(rgb, plant)
        lab = np.zeros((MASK_RES, MASK_RES), np.uint8); lab[lab0 == 3] = 3; lab[plant & (lab0 == 1)] = 1; lab[plant & (lab0 != 1)] = 2
        Image.fromarray(lab).save(md / (Path(v).stem + ".png")); Image.fromarray(band.astype(np.uint8)).save(bd / (Path(v).stem + ".png"))
        Image.fromarray((alpha * 255).round().astype(np.uint8)).save(sd / (Path(v).stem + ".png"))
        rec.update(final_px=int(plant.sum()), change_share=round(float((plant != plant0).sum() / max(plant0.sum(), 1)), 4), band_share=round(float(band.sum() / max(plant.sum(), 1)), 4))
        summ[v] = rec; prev = v; print(v[9:18], rec, flush=True); json.dump(summ, open(sp, "w"), indent=1)
    print("views", len(summ))


if __name__ == "__main__":
    main()
