# STAGE_SOURCES — where each stage of `run_recon.py` comes from

Modules under `microtom_recon/tomato3d/` are copies of our project files at the same relative path (`VERSIONS.txt` has
the hashes, `EDITS.csv` the changed lines). They are imported, not rewritten.

Three of our sources are Kaggle kernel scripts that run top to bottom over our own datasets (`kernel_all_masks.py`,
`kernel_all_points.py`, `kernel_all_train.py`) and two are command-line drivers tied to our folder layout
(`r31_outputs.py`, `charuco_masks.main`, `wholeplant_masks.main`). They cannot be imported as they are, so their
driver logic was adapted into functions of `run_recon.py`. The numeric functions they call are imported unchanged.

| Stage | `run_recon.py` function | Imported unchanged | Driver logic adapted from | Differences |
|---|---|---|---|---|
| 1 photos | `stage_photos` | `charuco_photos.main` (1 edited line: JPG/PNG accepted besides HEIC) | — | the photo folder is searched under `--photos` |
| 2 poses | `stage_poses`, `check_poses` | `charuco_a3.main` (unchanged) | — | new: the four early stops and `pose_qc.json` |
| 3 masks | `stage_masks` | `charuco_masks.colour_mask/carve/project`, `wholeplant_masks.seeds/negatives/thin_parts`, `r31_masks.matte`, `wholeplant_contract.ring`, `charuco_a3.scaled_K` | `kernel_all_masks.py` (hold-out rule, second hull, matte, band, sheet); `charuco_masks.main` (colour-in-hull); `wholeplant_masks.main` pass 1 (SAM step) | no C3 U-Net and no C2 masks: the raw mask is colour-in-hull plus SAM thin-part additions; stock `sam2.1_hiera_tiny.pt` instead of our tuned checkpoint; photos are read from the undistorted 3024 px JPGs of stage 2 instead of the HEIC; `--holdout N` generalises the 2-sector rule to N sectors (N = 2 is the original rule) |
| 4 points | `stage_points` | `wholeplant_sparse.main` (unchanged) | `kernel_all_points.py` | one plant; inputs are the stage 2 folder itself |
| 5 train | `stage_train` | `charuco_prep.read_xyz/write_xyzrgb/HULL_*`, trainer `kernel_2dgs_r31.py` run as a subprocess (4 edited lines: paths) | `kernel_all_train.py` | none in the numbers: 1512 px, 7,000 iterations, 150,000 initial points, seed 0, soft matte, 2 px band, exposure on |
| 6 outputs | `stage_outputs`, `preview` | `surfels_io.read_surfels_full/read_binary_ply`, `r31_score.score_view` | `r31_outputs.py` (point-cloud record), `kernel_all_outputs.py` (meta fields) | no mesh, no viewer; `surfels.ply` and `points.ply` keep opacity ≥ 0.5; height = z range of those surfels; `preview.png` is new (orthographic point picture) |
| 7 tprox | `tprox_stage.stage_tprox`, `run_frozen` | `pc_absorb.py` (unchanged, own process, set A), `pc_rachis.main` (2 edited lines: paths), `skeleton_io.load/resample`, `surfels_io.read_surfels_full`; `seeds` and `traces_arrays` copied verbatim from `skeleton_r8pc.py` | `skeleton_r8pc.step_bundle` (generic branch, mode prox), `skeleton_tprox.download_set/step_post/provenance` | no Kaggle kernel in between; groups, sheet geometry and axis classification are read from `--tprox-inputs`, not built; no distance-route comparison, no plant tree, no figure; `stem` flag = axis_kind 1 of the attached classification; new: frame and sha256 checks, `points_tprox.ply` |

Not copied: `kernel_all_*.py`, `r31_outputs.py`, `wholeplant_outputs.py`, `charuco_viewer.py`, everything outside `tomato3d/recon` and
`tomato3d/fuse3d/surfels_io.py`. Nothing in the zip imports Ultralytics or the VTC files.

v0.2 adds `tomato3d/organs/{__init__,pc_absorb,pc_rachis,skeleton_io}.py`. Not copied in v0.2: `skeleton_r8pc.py`, `skeleton_tprox.py` (they drive Kaggle from the project PC), `all3d_first.py`, `all3d_merge.py`, `compound_cluster.py` and everything they import (group building and the axis classifier stay on the project PC).
