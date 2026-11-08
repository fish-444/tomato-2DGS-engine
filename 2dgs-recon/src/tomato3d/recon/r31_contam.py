"""R3-1 follow-up, Part A: did the stage-b board/soil contamination reach the geometry? Read-only, no GPU.
Rules are fixed in build/p1_recon_r3-1_contam_20261022_01/CONTRACT_CONTAM.md (written before this ran).

usage: python -m tomato3d.recon.r31_contam --poses <charuco dir> --r31 <r3-1 run dir> --out <contam run dir>
Writes <out>/contam_partA.json and <out>/sheets/contam_render.jpg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import MASK_RES
from tomato3d.recon.r31_masks import pot_polygon, soil_split

RIM_Z, SIDE, C_XY = 0.066, 0.062, np.array([0.17075, 0.14525])
BELOW_MARGIN, OFF_DILATE, OFF_ZMAX = 0.005, 0.005, 0.015
STEM_R, STEM_BAND, AXIS_SLAB = 0.012, 0.025, 0.010
OP_MIN, DEPTH_TOL, AREA_MAX, RISE_MAX = 0.5, 0.008, 2000, 0.20
CLIP_D, CLIP_SHARE = 0.003, 0.05
VIEWS = ["20261021_060235403_iOS.jpg", "20261021_060439622_iOS.jpg", "20261021_060248144_iOS.jpg"]
RUNS = {"R3-base": ("kernel_out_base/out/R3-base_surfels_full.ply", "renders_base/renders.npz"),
        "R3-1b": ("kernel_out_1b/out/R3-1b_surfels_full.ply", "kernel_out_1b/out/R3-1b_views.npz"),
        "R3-1c-dens": ("kernel_out_1c/out/R3-1c-dens_surfels_full.ply", "kernel_out_1c/out/R3-1c-dens_views.npz")}


def in_square(xy, grow=0.0):
    h = SIDE / 2 + grow
    return (np.abs(xy[:, 0] - C_XY[0]) <= h) & (np.abs(xy[:, 1] - C_XY[1]) <= h)


def regions(xyz: np.ndarray, axis: np.ndarray) -> dict:
    z = xyz[:, 2]; r = np.linalg.norm(xyz[:, :2] - axis, axis=1)
    return dict(below_pot_top=z < RIM_Z - BELOW_MARGIN, off_pot_board=~in_square(xyz[:, :2], OFF_DILATE) & (z < OFF_ZMAX),
                stem_base=(r < STEM_R) & (z >= RIM_Z) & (z <= RIM_Z + STEM_BAND))


def stem_axis(xyz, op):
    s = (op >= OP_MIN) & (xyz[:, 2] >= RIM_Z) & (xyz[:, 2] <= RIM_Z + AXIS_SLAB) & in_square(xyz[:, :2])
    return np.median(xyz[s, :2], axis=0), int(s.sum())


def plane_hit(R, t, K, res, zplane):
    """Per pixel: world xy where the ray meets z = zplane, and the camera-z depth of that hit (inf if behind)."""
    ys, xs = np.mgrid[0:res, 0:res].astype(np.float64); d_cam = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs)], -1)
    Rm = np.asarray(R); C = -Rm.T @ np.asarray(t); d_w = d_cam @ Rm                     # world direction for camera-z = 1
    s = (zplane - C[2]) / np.where(np.abs(d_w[..., 2]) < 1e-12, 1e-12, d_w[..., 2]); s[s <= 0] = np.inf
    return C[:2] + d_w[..., :2] * s[..., None], s                                        # s = camera-z depth of the hit


def labels(src, r31, po, v):
    R0 = po["undistorted_res"]; K = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES); R = po["views"][v]["R_cw"]; t = po["views"][v]["t_cw"]
    rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR))
    mb = np.isin(np.asarray(Image.open(r31 / "masks_b" / (Path(v).stem + ".png"))), (1, 2))
    poly = pot_polygon(np.r_[C_XY, 0], R, t, K, MASK_RES); pin, _ = soil_split(rgb, mb, poly); soil = poly & ~pin
    body = np.zeros((MASK_RES, MASK_RES), bool)                                              # projected pot prism
    for zz in np.linspace(0, RIM_Z, 12):
        body |= pot_polygon(np.r_[C_XY, 0], R, t, K, MASK_RES, side=SIDE, z=zz)
    xy0, d0 = plane_hit(R, t, K, MASK_RES, 0.0); board = ~in_square(xy0.reshape(-1, 2), OFF_DILATE).reshape(MASK_RES, MASK_RES) & ~body & np.isfinite(d0)
    _, dr = plane_hit(R, t, K, MASK_RES, RIM_Z)
    return rgb, mb, soil, board, dr, d0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--r31", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src, r31, out = Path(a.poses), Path(a.r31), Path(a.out); po = json.load(open(src / "poses.json"))
    sf = {k: sio.read_surfels_full(r31 / p) for k, (p, _) in RUNS.items()}
    axis, n_axis = stem_axis(sf["R3-base"]["xyz"], sf["R3-base"]["opacity"])
    rep = dict(contract="CONTRACT_CONTAM.md", stem_axis_xy=axis.tolist(), stem_axis_from_surfels=n_axis, counts={}, render={}, clip={})
    for k, s in sf.items():
        xyz, op = s["xyz"], s["opacity"]; rg = regions(xyz, axis); hi = op >= OP_MIN
        rep["counts"][k] = dict(total=int(len(xyz)), total_op=int(hi.sum()),
                                **{n: dict(all=int(m.sum()), op=int((m & hi).sum()), share_all=float(m.mean()), share_op=float((m & hi).sum() / max(hi.sum(), 1))) for n, m in rg.items()})
    for k in ("R3-1b", "R3-1c-dens"):
        for n in ("below_pot_top", "off_pot_board", "stem_base"):
            b = rep["counts"]["R3-base"][n]["op"]; rep["counts"][k][n]["ratio_vs_base"] = (rep["counts"][k][n]["op"] / b) if b else None
    tiles = []
    for v in VIEWS:
        rgb, mb, soil, board, dr, d0 = labels(src, r31, po, v); rep["render"][v] = {}
        row = [cv2.resize(rgb, (420, 420))]
        for k, (_, npz) in RUNS.items():
            z = np.load(r31 / npz); al = z[f"a_{v}"].astype(np.float32); dp = z[f"d_{v}"].astype(np.float32)
            hi = al >= OP_MIN; lab = (soil | board) & mb
            on = (soil & (np.abs(dp - dr) <= DEPTH_TOL)) | (board & (np.abs(dp - d0) <= DEPTH_TOL))
            area = hi & lab & on; area2 = hi & lab
            rep["render"][v][k] = dict(declared_px=int(area.sum()), label_only_px=int(area2.sum()), soil_px=int((area & soil).sum()), board_px=int((area & board).sum()))
            o = rgb.astype(np.float32).copy(); o[hi] = o[hi] * 0.6 + np.array([255, 0, 190]) * 0.4; o[area2 & ~area] = o[area2 & ~area] * 0.3 + np.array([255, 200, 0]) * 0.7
            o[area] = o[area] * 0.2 + np.array([0, 240, 255]) * 0.8; t_ = cv2.resize(o.astype(np.uint8), (420, 420))
            cv2.putText(t_, f"{k} {area.sum()} px", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); row.append(t_)
        cv2.putText(row[0], v[9:18], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); tiles.append(np.hstack(row))
    sheet = np.vstack(tiles); cv2.putText(sheet, "pink = opacity>=0.5, cyan = on board/soil (declared), yellow = over board/soil label only", (8, sheet.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    (out / "sheets").mkdir(parents=True, exist_ok=True); Image.fromarray(sheet).save(out / "sheets" / "contam_render.jpg", quality=88)
    clipped = {k: s["xyz"][(s["opacity"] >= OP_MIN) & (s["xyz"][:, 2] >= RIM_Z - BELOW_MARGIN)] for k, s in sf.items()}
    tb = cKDTree(clipped["R3-base"])
    for k in ("R3-1b", "R3-1c-dens"):
        far = float((tb.query(clipped[k])[0] > CLIP_D).mean()); back = float((cKDTree(clipped[k]).query(clipped["R3-base"])[0] > CLIP_D).mean())
        rep["clip"][k] = dict(n=int(len(clipped[k])), n_base=int(len(clipped["R3-base"])), share_run_far_from_base=far, share_base_far_from_run=back,
                              same=bool(far <= CLIP_SHARE and back <= CLIP_SHARE))
    rise = any((rep["counts"][k][n]["ratio_vs_base"] or 0) > 1 + RISE_MAX for k in ("R3-1b", "R3-1c-dens") for n in ("below_pot_top", "off_pot_board"))
    big = {v[9:18]: {k: r["declared_px"] for k, r in rr.items() if k != "R3-base" and r["declared_px"] >= AREA_MAX} for v, rr in rep["render"].items()}
    big = {v: x for v, x in big.items() if x}
    rep["verdict"] = dict(region_rise_over_20pct=rise, views_over_2000px=big, verdict="CONTAMINATED" if (rise or big) else "CLEAN",
                          clipped_clouds_same=all(c["same"] for c in rep["clip"].values()),
                          stem_base_ratio={k: rep["counts"][k]["stem_base"]["ratio_vs_base"] for k in ("R3-1b", "R3-1c-dens")})
    json.dump(rep, open(out / "contam_partA.json", "w"), indent=1)
    print(json.dumps(dict(counts={k: {n: (c[n]["op"], round(c[n].get("ratio_vs_base") or 0, 3)) for n in ("below_pot_top", "off_pot_board", "stem_base")} | {"total_op": c["total_op"]} for k, c in rep["counts"].items()},
                          render={v[9:18]: {k: (r["declared_px"], r["label_only_px"]) for k, r in rr.items()} for v, rr in rep["render"].items()}, clip=rep["clip"], verdict=rep["verdict"]), indent=1))


if __name__ == "__main__":
    main()
