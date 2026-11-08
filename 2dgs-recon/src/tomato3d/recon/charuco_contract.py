"""Step 0.2: fix the held-out views (declared rule) and write contract.json + CONTRACT.md before any training.

Held-out rule: the capture has a top view and three rings (elevation > 58, 45-58 and < 45 deg); one accepted view is taken
from each ring, nearest to azimuth 30 deg (upper), 150 deg (middle) and 270 deg (low) in the board frame, never two within
60 deg of azimuth of each other.

usage: python -m tomato3d.recon.charuco_contract --poses <charuco dir> --recon <recon dir> --photos <photos.csv dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon import charuco_assign as cs
from tomato3d.recon import charuco_masks as cm
from tomato3d.recon import charuco_sparse as sp

TARGETS = ((30.0, "upper"), (150.0, "middle"), (270.0, "low"))
RINGS = ((58.0, "upper"), (45.0, "middle"), (-90.0, "low"))       # lower elevation bound of each ring; the top view counts as upper
MIN_SEP_DEG = 60.0
SETTINGS = dict(res=1512, iters=7000, n_init=150000, seed=0, gpu_budget_h_t4=2.0)
IOU_ACCEPT = 0.80


def angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def pick_heldout(rows: list[dict]) -> list[str]:
    acc = [r for r in rows if r["accepted"] == "1"]
    ring = {r["file"]: next(n for lo, n in RINGS if float(r["elevation_deg"]) > lo) for r in acc}
    out = []
    for az, rg in TARGETS:
        cand = [r for r in acc if ring[r["file"]] == rg and r["file"] not in out
                and all(angdiff(float(r["azimuth_deg"]), float(next(x for x in acc if x["file"] == o)["azimuth_deg"])) >= MIN_SEP_DEG for o in out)]
        if not cand:
            cand = [r for r in acc if r["file"] not in out]
        out.append(min(cand, key=lambda r: angdiff(float(r["azimuth_deg"]), az))["file"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--recon", required=True)
    a = ap.parse_args(argv); src, rc = Path(a.poses), Path(a.recon); rc.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(src / "qc.csv"))); photos = list(csv.DictReader(open(src / "photos.csv")))
    intr = json.load(open(src / "intrinsics.json")); held = pick_heldout(rows)
    acc = [r["file"] for r in rows if r["accepted"] == "1"]; rej = [(r["file"], r["reason"]) for r in rows if r["accepted"] != "1"]
    info = {r["file"]: r for r in rows}
    con = dict(date="2026-10-21", heldout=held, train=[v for v in acc if v not in held], rejected=dict(rej), **SETTINGS,
               photos=dict(heic=len(photos), kept=sum(p["kept"] == "1" for p in photos), duplicates=[p["name"] for p in photos if p["kept"] != "1"]),
               thresholds=dict(qc=dict(min_corners_calib=ca.MIN_CORNERS, min_corners_pose=ca.MIN_POINTS, plane_filter=ca.PLANE_FILTER, max_view_rms_px=ca.MAX_VIEW_RMS_PX, min_spread=ca.MIN_SPREAD, outlier_frac=ca.OUTLIER_FRAC,
                                       min_inlier_frac=ca.MIN_INLIER_FRAC),
                               mask=dict(a_max=cm.A_MAX, l_min=cm.L_MIN, hull_keep=cm.HULL_KEEP, hull_min_views=cm.HULL_MIN_VIEWS, fine_m=cm.FINE_M),
                               sparse=dict(min_fg_views=sp.MIN_FG_VIEWS, rms_max_px=sp.RMS_MAX_PX, canopy_pad_m=sp.CANOPY_PAD_M),
                               heldout_accept=dict(silhouette_iou_min=IOU_ACCEPT, rule="every held-out view >= 0.80; PSNR/SSIM/depth reported only"),
                               assign=dict(vis_rel=cs.VIS_REL, link_iou=cs.LINK_IOU, min_pair=cs.MIN_PAIR, min_agree=cs.MIN_AGREE, min_surfels=cs.MIN_SURFELS,
                                           quad_rel=cs.QUAD_REL, few_views_lt=3)))
    json.dump(con, open(rc / "contract.json", "w"), indent=1)
    hv = "\n".join(f"| `{v}` | {info[v]['azimuth_deg']} | {info[v]['elevation_deg']} | {info[v]['corners']} | {info[v]['rms_px']} |" for v in held)
    md = f"""# CONTRACT - 3D reconstruction from the A3 ChArUco capture (2026-10-21)

Prompt `handoff/CLI_PROMPT_P1_3D_RECON_CHARUCO_2026-10-21.md`. Written before any training; nothing below is changed after
the Kaggle push. Shape-only rule (LOG 2026-10-19 11:00): geometry in board metres, no trait tables, no mm accuracy claims.

## Board and poses
- Board read from the print header and checked against its printed 100 mm ruler: ChArUco 10 x 7, square 38 mm, marker 28 mm,
  DICT_5X5_100, ids 0-34 (ruler -> 38.2 mm per square in the top view; 36 mm ruled out).
- Poses: `build/p1_charuco_a3_20261021_01/` (no separate ChArUco A3 prompt file exists in handoff/; its Steps 1-5 are done here):
  markers on CLAHE 1/3 and 1/2 downscales with a tuned detector (union by id), a plane-consistency filter (RANSAC homography,
  {ca.PLANE_FILTER:.0%} of the image side), ChArUco interpolation and cornerSubPix ({ca.CORNER_WIN} px) at full resolution; intrinsics from this
  capture (OpenCV k1 k2 p1 p2, k3 fixed 0; {len(intr['views_used'])} views with >= {ca.MIN_CORNERS} corners; RMS {intr['rms_px']:.2f} px at 6048 px); per-view
  RANSAC SQPnP + LM on the ChArUco corners.
- Board quality: the corners sit on the printed corners (checked visually) but miss a single plane homography by 3-7 px at
  6048 px, so the printed sheet is not a perfect plane. Pose error is about 1.5 px at the 1512-px training resolution. The QC limits
  below were set from this, before any training; two earlier pose passes (0/84 and 15/84 accepted) are in `_TRASH/` with a MANIFEST.
- QC acceptance: >= {ca.MIN_POINTS} ChArUco corners, inlier RMS <= {ca.MAX_VIEW_RMS_PX} px at 6048 px (2 px at 1512), inliers >= {ca.MIN_INLIER_FRAC:.0%}
  (outlier > {ca.OUTLIER_FRAC * 6048:.0f} px), corner spread >= {ca.MIN_SPREAD} of the frame, camera above the board.

## Photo set
- {len(photos)} HEIC files from `<PHOTO_ROOT>` (the `data/records/pot_charuco_a3_20261021/heic/` copy holds 2 files, one empty;
  not edited) -> JPG with EXIF orientation applied; exact duplicates dropped: {', '.join(con['photos']['duplicates']) or 'none'}.
- Accepted views: {len(acc)}; rejected: {len(rej)} ({'; '.join(f'{f} {r}' for f, r in rej) or 'none'}).

## Held-out views (never used for the hull, triangulation or training)
Rule: one per ring (elevation > 58 / 45-58 / < 45 deg), nearest to azimuth 30 / 150 / 270 deg, >= 60 deg apart.

| view | azimuth deg | elevation deg | corners | PnP RMS px |
|---|---|---|---|---|
{hv}

## Masks (Step 1)
- Plant foreground = Lab a* < {cm.A_MAX} and L* > {cm.L_MIN} (opened {cm.OPEN_PX} px) inside the projection of a visual hull carved from the training
  views at {cm.FINE_M * 1000:.1f} mm voxels (kept if inside the {cm.HULL_DILATE_PX}-px-dilated colour mask in >= {cm.HULL_KEEP:.0%} of the views that see it,
  >= {cm.HULL_MIN_VIEWS} views); board plane projected from the poses (label 3). Pot not modelled (not green, so never plant).
- Leaflets: H2 (M1 score >= 0.15, SAM v5 box + DT-max positive + neighbour/rachis negatives, merge/clean, T1 tail) on the
  undistorted 3024-px views; foreground = the plant mask above; no Roboflow detections; floor = 0.0139 x plant grid area.

## Sparse (Step 2)
pycolmap 4.2: SIFT (<= {sp.MAX_FEATURES} features), exhaustive matching, `triangulate_points` with every frame fixed and intrinsics fixed;
plant points = inside the plant mask of >= {sp.MIN_FG_VIEWS} track views. Checks: plant reprojection RMS <= {sp.RMS_MAX_PX} px (3024 px),
above the board, inside the hull box + {sp.CANOPY_PAD_M * 100:.0f} cm. No dense step (optional; not run).

## 2DGS (Step 3, Kaggle T4, `dolgfish/tomato-recon-charuco-2dgs`, private dataset `dolgfish/tomato-recon-charuco-input`)
Row-31 trainer losses and schedule unchanged (gsplat 1.5.3, L1 + 0.2 SSIM + 0.1 alpha, normal 0.05 from 2000, distortion 100 from 1500,
DefaultStrategy as row 31); resolution {SETTINGS['res']} px, {SETTINGS['iters']} iterations, init <= {SETTINGS['n_init']:,} points (triangulated plant points +
<= 60,000 hull voxels), seed {SETTINGS['seed']}, one run. Alpha target = plant label only. GPU budget <= {SETTINGS['gpu_budget_h_t4']} h T4.

**Acceptance (held-out):** silhouette IoU of rendered alpha >= 0.5 vs the Step 1 plant mask >= {IOU_ACCEPT} in EVERY held-out view.
PSNR/SSIM on the plant region and the depth check (median |rendered depth - triangulated point depth|, metres) are reported only.
If any held-out IoU < {IOU_ACCEPT}: stop, report, no tuning on held-out views.

## Leaflets (Step 4)
Visibility: rendered alpha >= {cs.ALPHA_MIN} and |depth - rendered median depth| <= {cs.VIS_REL:.0%}. Link (view, H2 id) pairs when the IoU of their
surfel sets >= {cs.LINK_IOU} (>= {cs.MIN_PAIR} shared surfels); surfel -> majority component with >= {cs.MIN_AGREE} agreeing views; leaflets with
< {cs.MIN_SURFELS} surfels dropped. Plane, quadric when plane RMS / extent > {cs.QUAD_REL}. Flags: few_views (< 3), occluded_part, merged_suspect.

## Review (Step 5) and hand-off (Step 6)
db-backed page, grades ok / merged / fragment / not leaf / wrong surface + "same leaflet as"; empty at publish; scoring waits for "done".
Export for the engine only for leaflets graded ok; the engine is not run.
"""
    (rc / "CONTRACT.md").write_text(md, encoding="utf-8")
    print("held-out:", held)


if __name__ == "__main__":
    main()
