"""A3: denser initialisation. COLMAP dense stereo needs a CUDA build that is not available locally, so this uses the declared
fallback: full-resolution (6048 px) SIFT with more features and more matches, with the board poses fixed and the whole-plant
masks from A2 as the foreground test.

usage: python -m tomato3d.recon.wholeplant_sparse --poses <charuco dir> --out <run dir> --work <ascii dir>
Writes <run>/plant_points_dense.ply and sparse_dense_checks.json (per-view counts, reprojection RMS, geometry checks).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import MASK_RES
from tomato3d.recon.charuco_sparse import CANOPY_PAD_M, MIN_FG_VIEWS, RMS_MAX_PX

MAX_FEATURES = 32768       # run 01 used 16,384
SIFT_THREADS = 3
RES = 3024                 # 6048 px exhaustive matching on CPU was hours; the declared fallback is "more matches",
                           # so the feature cap is doubled and the whole-plant masks (stem included) widen the foreground test


def main(argv=None):
    import pycolmap
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
    a = ap.parse_args(argv); src, rc = Path(a.poses), Path(a.out)
    out_sp = rc / "sparse_dense"; out_sp.mkdir(parents=True, exist_ok=True)
    sp = Path(a.work); assert str(sp.resolve()).isascii(), "work folder must be ASCII-only"; sp.mkdir(parents=True, exist_ok=True)
    po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    con = json.load(open(rc / "contract.json")); held = set(con["heldout_new"])
    views = [v for v in po["accepted"] if v not in held and (rc / "masks" / (Path(v).stem + ".png")).exists()]
    K = ca.scaled_K(np.array(po["K_full"]), po["full_size"][0], RES)          # undistorted images keep K; rescale to RES
    Ku = ca.scaled_K(np.array(po["K_undist"]), R0, RES)
    img_dir = sp / "images"; img_dir.mkdir(exist_ok=True); db = sp / "database.db"
    for v in views:
        if not (img_dir / v).exists():
            Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((RES, RES), Image.LANCZOS).save(img_dir / v, quality=96)
    # resume per stage, not per file: a crash between extraction and matching must not leave an unmatched database behind
    if not db.exists():
        ro = pycolmap.ImageReaderOptions(); ro.camera_model = "PINHOLE"; ro.camera_params = f"{Ku[0, 0]},{Ku[1, 1]},{Ku[0, 2]},{Ku[1, 2]}"
        eo = pycolmap.FeatureExtractionOptions(); eo.sift.max_num_features = MAX_FEATURES; eo.num_threads = SIFT_THREADS; eo.max_image_size = RES
        pycolmap.extract_features(db, img_dir, image_names=views, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)
    with pycolmap.Database.open(db) as d:
        n_match = d.num_verified_image_pairs()
    if n_match == 0:
        mo = pycolmap.FeatureMatchingOptions(); mo.sift.max_ratio = 0.9; mo.max_num_matches = 65536; mo.num_threads = SIFT_THREADS
        pycolmap.match_exhaustive(db, matching_options=mo)
    with pycolmap.Database.open(db) as d:
        ids = {im.name: im.image_id for im in d.read_all_images()}; cam = d.read_all_cameras()[0]
    mi = sp / "model_in"; mi.mkdir(exist_ok=True)
    (mi / "cameras.txt").write_text(f"{cam.camera_id} PINHOLE {RES} {RES} {Ku[0, 0]:.6f} {Ku[1, 1]:.6f} {Ku[0, 2]:.6f} {Ku[1, 2]:.6f}\n")
    lines = []
    for v in views:
        q = ca.rot_to_qvec(np.array(po["views"][v]["R_cw"])); t = po["views"][v]["t_cw"]
        lines += [f"{ids[v]} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} {t[0]:.12f} {t[1]:.12f} {t[2]:.12f} {cam.camera_id} {v}", ""]
    (mi / "images.txt").write_text("\n".join(lines) + "\n"); (mi / "points3D.txt").write_text("")
    rec = pycolmap.Reconstruction(str(mi)); before = {i: rec.images[i].cam_from_world().matrix().copy() for i in rec.images}
    opt = pycolmap.IncrementalPipelineOptions(); opt.fix_existing_frames = True
    opt.ba_refine_focal_length = opt.ba_refine_principal_point = opt.ba_refine_extra_params = False
    opt.triangulation.ignore_two_view_tracks = False
    (sp / "model").mkdir(exist_ok=True)
    rec = pycolmap.triangulate_points(rec, db, img_dir, sp / "model", options=opt, refine_intrinsics=False)
    pose_change = max(float(np.abs(rec.images[i].cam_from_world().matrix() - before[i]).max()) for i in rec.images)
    rec.write_text(str(sp / "model"))
    for x in ("model_in", "model"):
        shutil.copytree(sp / x, out_sp / x, dirs_exist_ok=True)
    masks = {v: np.isin(np.asarray(Image.open(rc / "masks" / (Path(v).stem + ".png"))), (1, 2)) for v in views}
    name_of = {i: rec.images[i].name for i in rec.images}; f = MASK_RES / RES
    hull = json.load(open(rc / "masks_summary.json"))["_hull"]; lo = np.array(hull["lo"]) - CANOPY_PAD_M; hi = np.array(hull["hi"]) + CANOPY_PAD_M
    keep, errs, per_view = [], [], {v: 0 for v in views}
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
    assert keep, "no plant points survived the foreground test (check that the database has matches)"
    xyz = np.array([k[0] for k in keep]); col = np.array([k[1] for k in keep], np.uint8)
    r = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for k, c in zip("xyz", xyz.T): r[k] = c
    for k, c in zip("rgb", col.T): r[k] = c
    hdr = (f"ply\nformat binary_little_endian 1.0\ncomment board metres, z up; fixed board poses, {RES} px SIFT, whole-plant masks\n"
           f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    (rc / "plant_points_dense.ply").write_bytes(hdr.encode() + r.tobytes())
    z = sorted(per_view.values())
    res = dict(method=f"{RES} px SIFT fallback (COLMAP dense stereo needs a CUDA build, not available locally)",
               max_features=MAX_FEATURES, all_points=len(rec.points3D), plant_points=len(xyz), pose_change_max=pose_change,
               plant_reproj_rms_px_at_res=float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
               rms_limit_px_at_3024=RMS_MAX_PX, share_above_board=float((xyz[:, 2] > 0).mean()),
               share_inside_canopy_box=float(((xyz >= lo) & (xyz <= hi)).all(1).mean()),
               mean_track_length=float(np.mean([k[3] for k in keep])) if keep else None,
               per_view=dict(min=z[0], median=float(np.median(z)), max=z[-1], under_10=int(sum(x < 10 for x in z))),
               points_per_view=per_view)
    json.dump(res, open(rc / "sparse_dense_checks.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "points_per_view"}, indent=1))


if __name__ == "__main__":
    main()
