# CONTRACT - 3D reconstruction from the A3 ChArUco capture (2026-10-21)

Prompt `handoff/CLI_PROMPT_P1_3D_RECON_CHARUCO_2026-10-21.md`. Written before any training; nothing below is changed after
the Kaggle push. Shape-only rule (LOG 2026-10-19 11:00): geometry in board metres, no trait tables, no mm accuracy claims.

## Board and poses
- Board read from the print header and checked against its printed 100 mm ruler: ChArUco 10 x 7, square 38 mm, marker 28 mm,
  DICT_5X5_100, ids 0-34 (ruler -> 38.2 mm per square in the top view; 36 mm ruled out).
- Poses: `build/p1_charuco_a3_20261021_01/` (no separate ChArUco A3 prompt file exists in handoff/; its Steps 1-5 are done here):
  markers on CLAHE 1/3 and 1/2 downscales with a tuned detector (union by id), a plane-consistency filter (RANSAC homography,
  2% of the image side), ChArUco interpolation and cornerSubPix (31 px) at full resolution; intrinsics from this
  capture (OpenCV k1 k2 p1 p2, k3 fixed 0; 56 views with >= 12 corners; RMS 5.98 px at 6048 px); per-view
  RANSAC SQPnP + LM on the ChArUco corners.
- Board quality: the corners sit on the printed corners (checked visually) but miss a single plane homography by 3-7 px at
  6048 px, so the printed sheet is not a perfect plane. Pose error is about 1.5 px at the 1512-px training resolution. The QC limits
  below were set from this, before any training; two earlier pose passes (0/84 and 15/84 accepted) are in `_TRASH/` with a MANIFEST.
- QC acceptance: >= 8 ChArUco corners, inlier RMS <= 8.0 px at 6048 px (2 px at 1512), inliers >= 80%
  (outlier > 15 px), corner spread >= 0.1 of the frame, camera above the board.

## Photo set
- 85 HEIC files from `<PHOTO_ROOT>` (the `data/records/pot_charuco_a3_20261021/heic/` copy holds 2 files, one empty;
  not edited) -> JPG with EXIF orientation applied; exact duplicates dropped: 20261021_060423912_iOS 1.heic.
- Accepted views: 65; rejected: 19 (20261021_060433835_iOS.jpg inliers<0.8; 20261021_060443572_iOS.jpg rms>8.0;inliers<0.8; 20261021_060445656_iOS.jpg rms>8.0; 20261021_060446991_iOS.jpg rms>8.0; 20261021_060452133_iOS.jpg rms>8.0; 20261021_060455325_iOS.jpg rms>8.0; 20261021_060456328_iOS.jpg rms>8.0; 20261021_060457442_iOS.jpg rms>8.0; 20261021_060458583_iOS.jpg rms>8.0; 20261021_060529992_iOS.jpg rms>8.0; 20261021_060531086_iOS.jpg points<8; 20261021_060533609_iOS.jpg points<8; 20261021_060534548_iOS.jpg points<8; 20261021_060536184_iOS.jpg points<8; 20261021_060537292_iOS.jpg spread<0.1; 20261021_060539367_iOS.jpg spread<0.1; 20261021_060546510_iOS.jpg rms>8.0; 20261021_060553242_iOS.jpg rms>8.0; 20261021_060602722_iOS.jpg rms>8.0).

## Held-out views (never used for the hull, triangulation or training)
Rule: one per ring (elevation > 58 / 45-58 / < 45 deg), nearest to azimuth 30 / 150 / 270 deg, >= 60 deg apart.

| view | azimuth deg | elevation deg | corners | PnP RMS px |
|---|---|---|---|---|
| `20261021_060326166_iOS.jpg` | 27.1 | 66.4 | 12 | 3.994 |
| `20261021_060448335_iOS.jpg` | 167.5 | 53.6 | 8 | 6.562 |
| `20261021_060605333_iOS.jpg` | 266.1 | 31.8 | 21 | 7.222 |

## Masks (Step 1)
- Plant foreground = Lab a* < -4.0 and L* > 8.0 (opened 2 px) inside the projection of a visual hull carved from the training
  views at 1.5 mm voxels (kept if inside the 6-px-dilated colour mask in >= 85% of the views that see it,
  >= 8 views); board plane projected from the poses (label 3). Pot not modelled (not green, so never plant).
- Leaflets: H2 (M1 score >= 0.15, SAM v5 box + DT-max positive + neighbour/rachis negatives, merge/clean, T1 tail) on the
  undistorted 3024-px views; foreground = the plant mask above; no Roboflow detections; floor = 0.0139 x plant grid area.

## Sparse (Step 2)
pycolmap 4.2: SIFT (<= 16384 features), exhaustive matching, `triangulate_points` with every frame fixed and intrinsics fixed;
plant points = inside the plant mask of >= 2 track views. Checks: plant reprojection RMS <= 1.5 px (3024 px),
above the board, inside the hull box + 2 cm. No dense step (optional; not run).

## 2DGS (Step 3, Kaggle T4, `dolgfish/tomato-recon-charuco-2dgs`, private dataset `dolgfish/tomato-recon-charuco-input`)
Row-31 trainer losses and schedule unchanged (gsplat 1.5.3, L1 + 0.2 SSIM + 0.1 alpha, normal 0.05 from 2000, distortion 100 from 1500,
DefaultStrategy as row 31); resolution 1512 px, 7000 iterations, init <= 150,000 points (triangulated plant points +
<= 60,000 hull voxels), seed 0, one run. Alpha target = plant label only. GPU budget <= 2.0 h T4.

**Acceptance (held-out):** silhouette IoU of rendered alpha >= 0.5 vs the Step 1 plant mask >= 0.8 in EVERY held-out view.
PSNR/SSIM on the plant region and the depth check (median |rendered depth - triangulated point depth|, metres) are reported only.
If any held-out IoU < 0.8: stop, report, no tuning on held-out views.

## Leaflets (Step 4)
Visibility: rendered alpha >= 0.5 and |depth - rendered median depth| <= 2%. Link (view, H2 id) pairs when the IoU of their
surfel sets >= 0.3 (>= 20 shared surfels); surfel -> majority component with >= 2 agreeing views; leaflets with
< 30 surfels dropped. Plane, quadric when plane RMS / extent > 0.03. Flags: few_views (< 3), occluded_part, merged_suspect.

## Review (Step 5) and hand-off (Step 6)
db-backed page, grades ok / merged / fragment / not leaf / wrong surface + "same leaflet as"; empty at publish; scoring waits for "done".
Export for the engine only for leaflets graded ok; the engine is not run.
