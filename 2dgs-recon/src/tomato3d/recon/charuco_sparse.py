"""Step 2: COLMAP (pycolmap) SIFT features -> exhaustive matching -> point triangulation with the board poses FIXED
(no pose adjustment, intrinsics fixed from the ChArUco calibration), then plant-point filtering and checks.
Held-out views (contract.json) are excluded from extraction, matching and triangulation.

COLMAP runs in an ASCII-only work folder (--work): pycolmap stops silently on the non-ASCII project path on Windows. The images
are copied there; the database and models are copied back to <recon>/sparse/.
usage: python -m tomato3d.recon.charuco_sparse --poses <charuco dir> --recon <recon dir> --work <ascii dir>
Writes <recon>/sparse/{database.db, model_in/, model/}, plant_points.ply, sparse_checks.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import MASK_RES

MIN_FG_VIEWS = 2           # a plant point lies inside the plant foreground of >= this many of its track views
CANOPY_PAD_M = 0.02        # footprint check: plant points inside the visual-hull bbox expanded by this
RMS_MAX_PX = 1.5           # check: reprojection RMS of plant points (at the undistorted resolution)
MAX_FEATURES = 16384
SIFT_THREADS = 3          # all threads at 3024 px aborted (C++ exception, RAM) while the H2 job ran beside it


def main(argv=None):
    import pycolmap
    import shutil
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--recon", required=True); ap.add_argument("--work", required=True)
    a = ap.parse_args(argv); src = Path(a.poses); rc = Path(a.recon); out_sp = rc / "sparse"; out_sp.mkdir(parents=True, exist_ok=True)
    sp = Path(a.work); assert str(sp.resolve()).isascii(), "work folder must be ASCII-only"; sp.mkdir(parents=True, exist_ok=True)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]; K = np.array(po["K_undist"])
    held = set(json.load(open(rc / "contract.json"))["heldout"]); views = [v for v in po["accepted"] if v not in held]
    img_dir = sp / "images"; img_dir.mkdir(exist_ok=True); db = sp / "database.db"
    for v in views:
        if not (img_dir / v).exists():
            shutil.copy2(src / f"undist_{R0}" / v, img_dir / v)
    if not db.exists():
        ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"; ro.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
        eo = pycolmap.FeatureExtractionOptions(); eo.sift.max_num_features = MAX_FEATURES; eo.num_threads = SIFT_THREADS; eo.max_image_size = 3024
        pycolmap.extract_features(db, img_dir, image_names=views, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)
        pycolmap.match_exhaustive(db)
    with pycolmap.Database.open(db) as d:
        ids = {im.name: im.image_id for im in d.read_all_images()}; cam = d.read_all_cameras()[0]
    mi = sp / "model_in"; mi.mkdir(exist_ok=True)
    (mi / "cameras.txt").write_text(f"{cam.camera_id} PINHOLE {R0} {R0} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
    lines = []
    for v in views:
        q = ca.rot_to_qvec(np.array(po["views"][v]["R_cw"])); t = po["views"][v]["t_cw"]
        lines += [f"{ids[v]} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} {t[0]:.12f} {t[1]:.12f} {t[2]:.12f} {cam.camera_id} {v}", ""]
    (mi / "images.txt").write_text("\n".join(lines) + "\n"); (mi / "points3D.txt").write_text("")
    rec = pycolmap.Reconstruction(str(mi)); before = {i: rec.images[i].cam_from_world().matrix().copy() for i in rec.images}
    opt = pycolmap.IncrementalPipelineOptions(); opt.fix_existing_frames = True
    opt.ba_refine_focal_length = opt.ba_refine_principal_point = opt.ba_refine_extra_params = False
    (sp / "model").mkdir(exist_ok=True)
    rec = pycolmap.triangulate_points(rec, db, img_dir, sp / "model", options=opt, refine_intrinsics=False)
    pose_change = max(float(np.abs(rec.images[i].cam_from_world().matrix() - before[i]).max()) for i in rec.images)
    rec.write_text(str(sp / "model"))
    for x in ("model_in", "model"):
        shutil.copytree(sp / x, out_sp / x, dirs_exist_ok=True)
    shutil.copy2(db, out_sp / "database.db")
    # plant filtering with the Step 1 masks
    masks = {v: np.asarray(Image.open(rc / "masks" / (Path(v).stem + ".png"))) == 1 for v in views}
    name_of = {i: rec.images[i].name for i in rec.images}; f = MASK_RES / R0
    hull = json.load(open(rc / "masks_summary.json"))["hull"]; lo = np.array(hull["lo"]) - CANOPY_PAD_M; hi = np.array(hull["hi"]) + CANOPY_PAD_M
    keep, errs, per_view, all_n = [], [], {v: 0 for v in views}, len(rec.points3D)
    for pid, p in rec.points3D.items():
        hits, tv = 0, []
        for el in p.track.elements:
            v = name_of[el.image_id]; xy = rec.images[el.image_id].points2D[el.point2D_idx].xy
            u, w = int(xy[0] * f), int(xy[1] * f)
            if 0 <= u < MASK_RES and 0 <= w < MASK_RES and masks[v][w, u]:
                hits += 1
            tv.append(v)
        if hits >= MIN_FG_VIEWS:
            keep.append((p.xyz, p.color, p.error, len(tv))); errs.append(p.error)
            for v in set(tv):
                per_view[v] += 1
    xyz = np.array([k[0] for k in keep]) if keep else np.zeros((0, 3)); col = np.array([k[1] for k in keep], np.uint8) if keep else np.zeros((0, 3), np.uint8)
    above = float((xyz[:, 2] > 0).mean()) if len(xyz) else None
    inside = float(((xyz >= lo) & (xyz <= hi)).all(1).mean()) if len(xyz) else None
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else None
    rec_ = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for k, c in zip("xyz", xyz.T): rec_[k] = c
    for k, c in zip("rgb", col.T): rec_[k] = c
    hdr = (f"ply\nformat binary_little_endian 1.0\ncomment board metres, z up; COLMAP triangulation with fixed board poses\nelement vertex {len(xyz)}\n"
           "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    (rc / "plant_points.ply").write_bytes(hdr.encode() + rec_.tobytes())
    out = dict(all_points=all_n, plant_points=len(xyz), min_fg_views=MIN_FG_VIEWS, pose_change_max=pose_change,
               plant_reproj_rms_px=rms, plant_reproj_rms_ok=(rms is not None and rms <= RMS_MAX_PX), rms_max_px=RMS_MAX_PX,
               share_above_board=above, share_inside_canopy_box=inside, canopy_box=[lo.tolist(), hi.tolist()],
               mean_track_length=float(np.mean([k[3] for k in keep])) if keep else None, points_per_view=per_view)
    json.dump(out, open(rc / "sparse_checks.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "points_per_view"}))


if __name__ == "__main__":
    main()
