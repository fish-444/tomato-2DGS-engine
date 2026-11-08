# 2DGS full-surfel export + disc viewer + 3DGS/.splat — contract (declared 2026-10-18 before any number was computed)

Prompt: `handoff/CLI_PROMPT_P1_2DGS_DISC_EXPORT_VIEWER_2026-10-18.md`. Track A. Relative scale only (R1 VGGT frame;
H = R1 plant height: dwt_03 0.2875, scene_0002 0.2002); no millimetre claim for dwt_03 (pot inner diameter not
measured). Synthetic scene_0002 metrics in mm are truth-frame numbers of a synthetic scene, not real-photo performance.

## This is a re-run, not a reload
The 10-16 run (`dolgfish/tomato-dwt03-oss-2dgs`, output in `build/p1_dwt03_oss_20261015_01/`) stored the surfel normal
but reduced the quaternion to `R[:, :, 2]`, so the in-plane rotation is lost and cannot be recovered from its files.
The trainer is therefore **run again**. gsplat rasterisation / densification on the GPU is not bit-deterministic
(atomic adds), so the new surfels **will differ** from the 10-16 surfels even with identical code, data and seed.
The 10-16 surfels, the A2 consolidated cloud (`build/p1_pc_dwt03_20261016_01/`) and everything built on them stay as
they are; nothing here replaces them.

## Step 1 — kernel `dolgfish/tomato-dwt03-2dgs-full` (Kaggle T4)
- Source: `build/p1_dwt03_oss_20261015_01/kernel/run.py` (trainer sha256 b5270f5e… = that file with LF line endings).
  Changes are **export-only**, placed after the training loop: no hyper-parameter, loss, schedule, strategy, view
  list, init, seed or set order (scene_0002 then dwt_03) changes. The diff is stored next to the new script.
- Inputs: the input dataset `dolgfish/tomato-dwt03-oss-input` was deleted after the 10-16 run; it is rebuilt with the
  same builder from the same sources (dwt_03 photos sha256-checked against `teacher_dwt01_04_20261015_01/dwt_03/
  status.json`; masks, init clouds, results and `kernels/B/*_cams.json` from `build/p1_dwt03_dense_20261015_01/`), the
  file list and sha256 of the upload recorded in `inputs_manifest.json`. Kernel source `dolgfish/tomato-dwt03-dense-a`
  (scene_0002 3024-px renders) unchanged. The rebuilt dataset is deleted after the run; the kernel is kept.
- Added exports per set: `<set>_surfels_full.ply` (x y z, qw qx qy qz normalised, tu, tv = first two columns of R,
  nx ny nz = third column, s0 s1 = exp(scales[:2]), opacity = sigmoid, uchar r g b, float f_dc_0..2 = the trained
  colour parameters as stored (logits; rgb = sigmoid(f_dc); they are NOT spherical-harmonic coefficients)),
  `<set>_params.pt` (raw tensors), `<set>_cameras.json` (the 1512-px K, world-to-camera viewmat and 3024-px K/R/t per
  training view, same frame); the old-format `<set>_surfels.ply` is still written by the unchanged code.
- Added evaluation (no effect on training): per training view, full-1512-px L1 and SSIM of the final render against the
  masked target, with the trainer's own `ssim()`.

## Step 2 — acceptance of the re-run as the export source (declared now)
The 10-16 run is a single realisation, so no repeat-run spread exists. "Spread reported for the 10-16 run" is taken
as follows, per metric; every metric must pass or the task stops after the report:
1. Surfel count: within ±10 % of 10-16 (dwt_03 233,088; scene_0002 119,956).
2. Set-to-set nearest-neighbour distance (new centre → nearest old centre and back), p50 ≤ the 10-16 surfel-centre
   local spread p50 (dwt_03 0.50 % H, scene_0002 0.36 % H, from the 10-16 report); p95 reported.
3. Training L1 as logged (every 500 iterations): mean of the new run's logged values at iterations 5000–7000 inside
   the [min, max] of the 10-16 logged values at the same iterations (dwt_03 [0.0061, 0.0101], scene_0002 [0.0015,
   0.0027]).
4. Render L1 / SSIM on the training views, like-for-like from the two runs' render sheets (both are the 1512-px
   renders down-sampled ×4 and JPEG q80, compared with the masked target down-sampled the same way): per-view
   |ΔL1| median ≤ 10 % of the 10-16 median L1 and |ΔSSIM| median ≤ 0.01. The new run's full-res per-view numbers are
   reported in addition (no 10-16 counterpart exists).
5. scene_0002 vs synthetic truth with the 10-16 metric code (`tomato3d.cli.run_dwt03_oss`, unchanged; output to
   `build/p1_2dgs_full_20261018_01/rerun_eval/`, not to the 10-16 folder): fused-cloud and plant-surfel point-to-mesh
   p50 / p95 within ±25 % of 10-16 (fused 0.19 / 6.90 mm, surfels 0.34 / 2.26 mm), leaf coverage within ±3 points
   (81 %, 84 %).

## Step 3 — disc viewer (one self-contained HTML, ≤ 40 MB)
WebGL2. Each surfel = instanced quad `p ± 2·s0·tu ± 2·s1·tv`, fragment alpha `opacity · exp(-0.5 r²)` cut at r = 2;
**back-to-front sorting** by view depth (CPU counting sort on every redraw; no OIT). Camera-frame fix kept
(`p.y = -p.y; p.z = -p.z`, same centring as the A2 viewer so overlays coincide). Overlay toggles with the point clouds
already shown in the chat (taken byte-for-byte from `build/chat_dwt03_a2_20261017/dwt_03_a2_before_after.html`).
Toggles: discs / points, colour / normal colour, opacity scale, disc size multiplier. Quantisation declared: position
uint16 per axis over the A2 viewer box, tu/tv int16 (re-orthonormalised in the shader), s0/s1 float16, opacity uint8,
rgb uint8; quantisation error reported. Verified headless (Playwright + SwiftShader): renders, rotates, top view
orientation vs `dwt_03_render_sheet.jpg`.

## Step 4 — 3DGS ply + .splat (dwt_03, from the accepted re-run)
3D Gaussian per surfel: rotation q (2DGS local z = normal), scales (s0, s1, ε) with **ε = 0.01 · max(s0, s1)** per
surfel (main file) and ε = 0.001 · H (sensitivity file). Opacity unchanged (stored as logit), colour DC only:
f_dc = (rgb − 0.5) / 0.28209479 with rgb = sigmoid(trained colour); no higher SH. `.splat`: 32 B per Gaussian (float32
xyz, float32 scale×3, uint8 rgba, uint8 quaternion×4 = q·128 + 128), sorted by exp(Σ log scale) · opacity descending.
Round-trip: both files read back and compared with the source arrays; max abs error stated per field. A thin 3D
Gaussian approximates a 2D surfel; geometry measurements stay on the 2DGS surfels.

## Storage ≤ 1 GB; no file deleted (removals go to `작업물저장소/TRASH/<path>` with a MANIFEST row).
