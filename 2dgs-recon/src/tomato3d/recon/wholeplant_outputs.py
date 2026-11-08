"""A5 + A6 for the accepted run: pot-rim scale check, TSDF mesh, point cloud, disc viewer, recon_meta.json.

A5 is a SCALE check on the board, not a plant trait: the pot rim is fitted in the photos (not in the plant model, which
contains no pot) by projecting a horizontal square of side s at the rim height into every training view and taking the s
that best matches the pot's silhouette; the ratio to the known rim dimension is reported.

usage: python -m tomato3d.recon.wholeplant_outputs --poses <charuco dir> --out <run dir> --k <kernel out dir> --run R2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import MASK_RES, project
from tomato3d.recon.wholeplant_contract import A5

VOX = 0.0012                 # TSDF voxel, board metres
RIM_Z_SEARCH = (0.030, 0.090)
RIM_S_SEARCH = (0.040, 0.095)


def soil_mask(rgb: np.ndarray, plant: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The soil surface (brown vermiculite) inside the projected pot region, plant removed. A colour rule alone picked the
    bright background, so the search is confined to the projection of a 3D box around the pot (centre = canopy centre)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255; A = lab[..., 1] - 128; B = lab[..., 2] - 128
    m = (B > 12) & (A > 0) & (L > 15) & region & ~plant
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def box_region(c, R, t, K, res, side=0.14, zmax=0.10) -> np.ndarray:
    h = side / 2; P = np.array([[c[0] + sx * h, c[1] + sy * h, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, zmax)])
    u, w, z = project(P, R, t, K); m = np.zeros((res, res), np.uint8)
    if (z > 0).all():
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.c_[u, w]).astype(np.int32)), 1)
    return m > 0


def rim_fit(src: Path, rc: Path, po: dict, views: list[str]) -> dict:
    """Best (z, side) of a horizontal square whose projection matches the SOIL SURFACE (the pot's open top) in the training
    views, scored by IoU so that a too-small square inside the soil does not win."""
    R0 = po["undistorted_res"]; K = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES)
    hull = json.load(open(rc / "masks_summary.json"))["_hull"]; c = (np.array(hull["lo"]) + np.array(hull["hi"])) / 2
    pots, cams = {}, {}
    for v in views:
        rgb = np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.BILINEAR))
        lab = np.asarray(Image.open(rc / "masks" / (Path(v).stem + ".png")))
        cams[v] = (po["views"][v]["R_cw"], po["views"][v]["t_cw"], K)
        pots[v] = soil_mask(rgb, np.isin(lab, (1, 2)), box_region(c, *cams[v], MASK_RES))
    best = None
    for z in np.arange(*RIM_Z_SEARCH, 0.004):
        for s in np.arange(*RIM_S_SEARCH, 0.002):
            h = s / 2; sc = []
            for v in views:
                P = np.array([[c[0] - h, c[1] - h, z], [c[0] + h, c[1] - h, z], [c[0] + h, c[1] + h, z], [c[0] - h, c[1] + h, z]])
                u, w, zz = project(P, *cams[v])
                if (zz <= 0).any():
                    continue
                q = np.zeros((MASK_RES, MASK_RES), np.uint8)
                cv2.fillConvexPoly(q, np.round(np.c_[u, w]).astype(np.int32), 1)
                q = q > 0; inter = (q & pots[v]).sum(); union = (q | pots[v]).sum()
                if union:
                    sc.append(inter / union)                   # IoU of the projected rim square and the soil surface
            if sc and len(sc) >= len(views) // 2:
                m = float(np.median(sc))
                if best is None or m > best[0]:
                    best = (m, float(z), float(s))
    return dict(score=best[0], rim_z_m=best[1], rim_side_m=best[2]) if best else {}


def tsdf(src: Path, rc: Path, po: dict, views: list[str], k: Path, run: str, out: Path) -> dict:
    """Surfel splatting into a TSDF-like occupancy volume, then marching cubes. Depth comes from the trained model's
    rendered median depth, which the kernel stored for the held-out views only, so the mesh is built from the surfels."""
    from skimage.measure import marching_cubes
    sf = sio.read_surfels_full(k / f"{run}_surfels_full.ply")
    xyz = sf["xyz"]; op = sf["opacity"]; keep = op >= 0.1
    xyz = xyz[keep]; sc = sf["scale"][keep]
    lo = xyz.min(0) - 0.006; hi = xyz.max(0) + 0.006
    dims = np.ceil((hi - lo) / VOX).astype(int) + 1
    vol = np.zeros(dims, np.float32)
    idx = np.clip(((xyz - lo) / VOX).round().astype(int), 0, dims - 1)
    r = np.clip((sc.mean(1) / VOX), 1, 4).astype(int)
    for dz in range(-2, 3):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                w = np.exp(-0.5 * (d / np.maximum(r, 1)) ** 2)
                j = np.clip(idx + [dx, dy, dz], 0, dims - 1)
                np.add.at(vol, (j[:, 0], j[:, 1], j[:, 2]), w)
    vol = cv2.GaussianBlur(vol.reshape(dims[0], -1), (3, 3), 0).reshape(dims)
    lvl = float(np.percentile(vol[vol > 0], 55))
    verts, faces, _, _ = marching_cubes(vol, level=lvl)
    V = verts * VOX + lo
    with open(out, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\ncomment board metres, z up; whole plant, no organ split\n"
                 f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(V.astype("<f4").tobytes())
        fr = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)]); fr["n"] = 3; fr["i"] = faces
        f.write(fr.tobytes())
    return dict(vertices=int(len(V)), faces=int(len(faces)), voxel_m=VOX, level=lvl, surfels_used=int(keep.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", required=True); ap.add_argument("--run", required=True); ap.add_argument("--acceptance", default="pending")
    a = ap.parse_args(argv); src, rc, kd = Path(a.poses), Path(a.out), Path(a.k)
    po = json.load(open(src / "poses.json")); con = json.load(open(rc / "contract.json"))
    views = [v for v in po["accepted"] if v not in con["heldout_new"] and (rc / "masks" / (Path(v).stem + ".png")).exists()]
    (rc / "outputs").mkdir(exist_ok=True)
    rim = rim_fit(src, rc, po, views[::4])
    ratio = rim["rim_side_m"] * 1000 / A5["pot_rim_mm"] if rim else None
    a5 = dict(known_rim_mm=A5["pot_rim_mm"], source="row 32 value (this pot was not measured by the user)",
              fitted_rim_side_m=rim.get("rim_side_m"), fitted_rim_z_m=rim.get("rim_z_m"), fit_score=rim.get("score"),
              ratio=ratio, tolerance=A5["tolerance"], flagged=(ratio is not None and abs(ratio - 1) > A5["tolerance"]),
              note="scale check on the board frame, not a plant trait")
    sf = sio.read_surfels_full(kd / f"{a.run}_surfels_full.ply")
    rec = np.zeros(len(sf["xyz"]), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for kk, c in zip("xyz", sf["xyz"].T): rec[kk] = c
    for kk, c in zip(("nx", "ny", "nz"), sf["normal"].T): rec[kk] = c
    for kk, c in zip("rgb", sf["rgb"].T.astype(np.uint8)): rec[kk] = c
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment board metres, z up; whole plant, no organ split; acceptance: "
           f"{a.acceptance}\nelement vertex {len(rec)}\n" + "".join(f"property float {x}\n" for x in ("x", "y", "z", "nx", "ny", "nz")) +
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    (rc / "outputs" / f"{a.run}_points.ply").write_bytes(hdr.encode() + rec.tobytes())
    mesh = tsdf(src, rc, po, views, kd, a.run, rc / "outputs" / f"{a.run}_mesh.ply")
    res = json.load(open(kd / "results.json"))["runs"][a.run]
    meta = dict(run=a.run, acceptance=a.acceptance, organ_split="none", frame="board metres, z up",
                poses="run 01 board poses (A1 refinement rejected: corner RMS 5.16 px > 0.8; cameras moved <= 0.38 mm / 0.011 deg)",
                contract=str(rc / "CONTRACT.md"), masks="whole-plant (leaves + stem) one class" if a.run != "R0" else "run 01 colour-in-hull",
                heldout_new=con["heldout_new"], heldout_run01=con["heldout_run01"], metrics=res, scale_check=a5, mesh=mesh,
                surfels=int(len(sf["xyz"])), note="shape only; no trait tables and no mm accuracy claims about the plant")
    json.dump(meta, open(rc / "outputs" / "recon_meta.json", "w"), indent=1)
    print(json.dumps(dict(scale=a5, mesh=mesh), indent=1))


if __name__ == "__main__":
    main()
