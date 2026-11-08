# CONTRACT — leaflet curvature traits (Part A) + static MTG that follows the reconstruction (set `_04`)

Written 2026-11-05 before any other file in this folder and before any curvature number. Session: Claude Code (MTG build).
Binding texts (in `handoff/` or given in this session; sha256 in `MANIFEST.json`):
- `CLI_PROMPT_LEAF_CURVATURE_FROM_RECONSTRUCTIONS_ENGINE_C4B_AND_STATIC_2026-11-05.md` (17:30), **Part A only**.
- CLI ADDENDUM "The static MTG itself must follow the reconstruction: leaflet nodes carry the reconstructed geometry (new set `_04`)" (17:35, pasted by the user into this session; user instruction 17:23 "MTG를 복원 기준으로 맞게 설정하라고").
Labels: `leaf_curvature_from_reconstruction`, `mtg_follows_reconstruction`, `skeleton_labels_ungraded`, `conditional_on_unvalidated_scale`, `not_validation`, `split_amendment5_count_fixed_to_prior`. Curvature values are fits to ungraded reconstructions, not measurements, not validated.

## Existing bending (read first, as the prompt asks)
Engine v2 (`build/engine_v2_calib_20261104_01/CONTRACT.md`): compound-leaf rachis bend Δθ (C3, `TOTAL_BEND_DEG` normal 32.764 ± 10.602°, clipped 0–90°), θ0 by rank (C2), basal leaflet angle U(20, 50)°; **leaflets are flat diamonds** (one normal per leaflet), no leaflet-level curvature. The coplanar-vs-bending run (`build/engine_coplanar_sep_20261101_01`, R3-1b) used static test meshes T_fold / T_foldcup (not engine blocks): flat ×1.38 → fold+cup ×1.24 of disc light (fold 42 %, cup 51 % of the drop). This work adds **leaflet-surface curvature only**, measured in each leaflet's own frame (its mean normal), so rachis bend and leaflet orientation are not counted again.

## Inputs (read only)
- Split of `_03`: `build/mtg_build_v2_clicks_leaflets_20261105_01/split_<id>.npz` (per surfel: compound uid, leaflet code), `leaflets.csv`, the `_03` trees `build/mtg_build_v2_clicks_20261105_03/static/mtg_<id>_engine.plant_tree_v1.json` and specs; surfels / area (b) / normals through that folder's `run_split.surfel_data` (same sources as the split).
- Synthetic (noise floor): the same 3 Engine v2 m00 plants, scaling to 443 mm², sampling 9.73 / mm², position noise N(0, 0.124 mm) along the triangle normal, keep 0.927, seed 0; **true** leaflet membership (owner); surfel normals = the triangle normal (the measured noise figure is a position noise; no angular noise is added — stated).

## Per-leaflet fit (declared)
1. **Smoothed normals:** each surfel normal flipped to agree with the leaflet's PCA normal (oriented +z), then the mean of the leaflet's surfel normals within **r 2 mm** (same leaflet), renormalised. Never raw normals.
2. **Frame:** n̄ = area-(b)-weighted mean smoothed normal; e_s = first PCA axis of the leaflet's surfels projected ⟂ n̄, oriented from the leaflet base (split: 5 % surfels nearest the rachis) to the centroid; e_d = n̄ × e_s. Coordinates (s, d, z) about the area-weighted centroid.
3. **Quadric:** area-weighted least squares z = a0 + a1 s + a2 d + ½ κ_l s² + ½ κ_w d² + c s d (linear terms absorb residual tilt). Report κ_l, κ_w (1/mm; κ_l > 0 = tip curls toward the normal (light) side; κ_w > 0 = edges up = cupped toward the light), c, weighted R², n, length / width (p1–p99), total angle change Δα_l = κ_l · L/2 and Δα_w = κ_w · W/2 (degrees).
4. **Normal deviation:** angle between each smoothed normal and n̄ → per leaflet median and p90.
5. **Projected-area ratio:** top-view (xy) projected area ÷ 3-D area of the leaflet's α-mesh (from the split; 0.25 mm raster).
Leaflets with < 30 surfels: not fitted (listed).

## `MTG_LEAF_CURVATURE_v1.csv`
Rows by position (P1, P2, P3, T), rank class (main; side n = 0), quantity (κ_l, κ_w, c, Δα_l, Δα_w, R², normal deviation median, normal deviation p90, projected-area ratio): n, mean, SD, median, min, p1, p10, p90, p99, max; subsets all 11 and pilot only; plus the **noise-floor rows** (synthetic flat engine leaflets, same fit) — declared now: real values are read net of the noise floor (difference of medians reported). Written next to the templates in `build/traits_recon_20261101_01/` + `.sha256`; template v2.3 and the arrangement v1 file unchanged.

## Set `_04` = `_03` + leaflet geometry from each plant's own surfels
- Every split leaflet node stores: position (attachment s, petiolule offset, side) as in `_03`; frame (e_s, e_d, n̄), elevation / azimuth of e_s, spread φ, tilt, roll (as in the arrangement code); κ_l, κ_w, c, R²; length, width, area (b) (= the `_03` leaflet area, which sums to the compound area (b)).
- **Leaflet surface (rebuilt from the node values):** the leaflet's α-mesh triangulation (1 mm cells, α 2 mm: far finer than the C4B minimum of 3 × 6) taken in (s, d) of its frame, with z = the fitted quadric; then the (s, d) outline is scaled about the leaflet base by one factor k so that the curved surface area = the node's area (b) (bisection, ± 1e-6). k is reported (the α-mesh surface is about 2 × area (b), so k ≈ 0.7 is expected: area (b) is the project's silhouette-scaled convention, `conditional_on_unvalidated_scale`). The unscaled surface (k = 1) is also built and its check numbers are reported alongside (report only; the pass decision uses the area-(b) surface).
- Rachis, petiole, stem, nodes, θ0 / Δθ: unchanged from `_03`. Helper leaves of `_03` (3): unchanged, excluded from the checks, listed.

## Check (declared, per plant; thresholds fixed now)
Leaflet surfels (non-rachis, split leaves) vs the `_04` leaflet surfaces of the same plant:
1. **Area:** per compound leaf |Σ MTG leaflet surface area − area (b)| ≤ **5 %** (holds by construction unless the bisection fails; reported).
2. **Distance:** surfel → MTG surface and MTG → surfel (surface sampled at 9.73 / mm²), p50 / p95, as % of plant height H (spec `height_mm`); pass **p50 (surfel → MTG) ≤ 1.5 % H**.
3. **Normals:** the MTG surface sampled at 9.73 / mm² with position noise 0.124 mm (seed 0), smoothed r 2 mm like the surfels; per leaflet median normal deviation (MTG) − (surfels); pass: the median over the plant's leaflets within **± 5°**; p90 difference reported.
4. Overlays: per plant top and side views of the MTG leaflet surfaces over the surfels (mm, plant-centred axes; the user's figure is in engine units — the same two views are drawn).
A failing plant: per-leaf report and options; nothing loosened.

## Consumers (stated now, checked or reported)
- Engine round trip (`to_engine_structure` / `from_engine_states`) 11/11 expected on structure. The engine state holds a leaflet as base, tip, width, area (a flat diamond): curvature and outline **cannot pass** through it; reported as "import is flat; missing: per-leaflet κ_l, κ_w, c and outline (Engine C4B mesh)".
- B-model: one blade area / orientation / inclination per compound leaf; per-leaflet angles, curvature and outline are not representable; the B-model files are expected byte-identical to `_02` / `_03` (Kaggle `_02` import applies).
- Static surfel tracer: unchanged. Static MTG per-leaf light on `_04`: owned by the Engine static session (arrangement prompt Part C, "MTG per-leaf light: use the `_03` set"); this note tells it to use `_04`; not run here.

## RESULT run 1 (build_04) and amendment 1 (defects only; thresholds and H rule unchanged)
Run 1: 3 / 11 plants pass (n_04, n_05, e_04); engine round trip 11/11; B-model identical 11/11; 911 leaflets fitted, 20 not fitted (< 30 surfels); `MTG_LEAF_CURVATURE_v1.csv` written (sha256 `8cdf7ff7…`).
Two defects found before any judgement:
1. **Area (n_02, e_05, e_07 fails):** the 20 not-fitted leaflets kept their `_03` α-mesh (unscaled, ≈ 2 × area (b)) and were left out of the area sum and the distance check. Fix: a not-fitted leaflet's surface = its α-mesh, **flat in its PCA plane, scaled about its base to its area (b)** (flag `too_few_surfels_flat`); it is counted in the area and distance checks, excluded from the curvature table and the normal check.
2. **H:** the spec `height_mm` is not the plant height for several plants (n_01 16.5 mm while its stem axis spans 47 mm in z; R3-1b 37.5 mm vs 29 mm of axis plus leaves above). The pass rule stays as declared (H = spec `height_mm`); a second column with H_surfel = max z − min z of the plant's labelled surfels is reported alongside (report only), so the user can see which distance failures come from H.
Run 1 outputs kept as `*_run1.*`. The curvature fits do not change (same code and inputs); the curvature file is not rewritten if the re-run reproduces the same table (sha256 compared).

## RESULT run 2 (amendment 1) — 5 / 11 plants pass; options to the user
`mtg_geometry_check.csv` (per plant), `mtg_geometry_check_leaves.csv`, `figures/overlay_<id>.png`. Engine round trip **11/11**; B-model files byte-identical to `_03` (= `_02`) **11/11**; 911 leaflets fitted, 20 flat (< 30 surfels); outline scale k median 0.83.
| plant | area | p50 surfel→MTG % H (mm) | normal-dev diff | pass | p50 % H_surfel (report) |
|---|---|---|---|---|---|
| R3-1b | ok | 4.50 (1.68) | −6.2° | **fail** (distance, normals) | 2.66 |
| n_01 | ok | 5.89 (0.97) | −4.6° | **fail** (distance: spec H 16 mm) | 1.13 |
| n_02 | ok | 1.23 | −3.6° | pass | 0.97 |
| n_03 | ok | 0.98 | −7.1° | **fail** (normals) | 1.04 |
| n_04 | ok | 1.34 | −4.2° | pass | 1.17 |
| n_05 | ok | 1.31 | −4.8° | pass | 1.17 |
| n_06 | ok | 1.51 (0.98) | −3.8° | **fail** (distance by 0.01) | 1.09 |
| e_01 | ok | 1.65 (1.15) | −5.5° | **fail** (distance, normals) | 1.53 |
| e_04 | ok | 0.85 | −1.4° | pass | 0.95 |
| e_05 | ok | 1.29 | −4.1° | pass | 1.28 |
| e_07 | **−14 %** (e_07-L03) | 1.29 | −4.8° | **fail** (area) | 1.20 |
- e_07 area: e_07-L03 P1 (≈ 5.6 mm²) has an empty α-mesh (too few cells), so the tree keeps a param blade for it; the check counted only mesh surfaces → a check-accounting gap, not a geometry miss (distance and normals pass).
- Normals: the MTG surface is smoother than the surfels on every plant (median −1.4 … −7.1°): a single quadric per leaflet leaves RMS residual 1.2 mm (R² 0.44–0.56), and real surfel normals carry orientation noise the MTG samples do not have (the noise floor models position noise only).
- Unscaled surfaces (k = 1, area ≈ 2 × area (b), report only): p50 0.59–0.94 mm and normal diff −1.0 … −5.0°, i.e. shrinking to area (b) costs distance and normals.
- Curvature traits (`MTG_LEAF_CURVATURE_v1.csv`, sha256 `8cdf7ff764e65440a1854aa8500ee9b1caa69b96313376cb9b74c413f7370119`, re-run identical): medians P1 / P2 / P3 / T: κ_l −0.011 / −0.021 / −0.028 / −0.043 mm⁻¹ (Δα_l −5.6 / −7.9 / −11.4 / −21.9°), κ_w −0.015 / −0.037 / −0.033 / −0.025 (Δα_w −3.6 / −10.6 / −7.7 / −8.8°): tips and edges bend **away** from the light side; normal deviation median 15–18° (p90 29–37°); projected-area ratio 0.42–0.47. Noise floor (flat engine leaflets): |Δα| < 0.2° median, R² ≈ 0.01, normal deviation ≈ 0 (no angular noise modelled → the real normal-deviation values are an upper bound, not net of normal noise).
- Static MTG per-leaf light on `_04`: Engine static session (not run here).

## Amendment 2 — user decision "Fix height + area" (2026-11-05, after run 2; written before run 3)
1. **H for the distance check = H_surfel** (max z − min z of the plant's labelled surfels), replacing the spec `height_mm` (which is not the plant height). The 1.5 % threshold is unchanged; the spec-H number stays as a report column.
2. **Area check counts every leaflet.** A leaflet whose α-mesh is empty keeps a param blade; its L and W are scaled by one factor so the blade mesh area (`geometry.blade_mesh`, 12 × 7) = its area (b). It is then counted in the area sum and the distance check, and excluded from the normal check (flag `empty_alpha_mesh_param_blade`).
Nothing else changes (normal check ± 5°, area ± 5 %, quadric surfaces, scaling to area (b)). Run 2 outputs are kept as `*_run2.*`.

## RESULT run 3 (amendment 2) — 8 / 11 plants pass
Pass: n_01, n_02, n_04, n_05, n_06, e_04, e_05, e_07 (all areas within 5 %; p50 surfel → MTG 0.95–1.28 % H_surfel; normal-deviation difference −1.1 … −4.8°). Fail: **R3-1b** (p50 2.65 % H_surfel = 1.68 mm, normals −6.2°), **n_03** (normals −7.0°), **e_01** (p50 1.53 %, normals −5.8°). The cause is the same on all three: a single quadric per leaflet is smoother than the surfels (MTG normal deviation lower; unscaled surfaces narrow the gap but break area (b)). Engine round trip 11/11; B-model identical 11/11; curvature table identical to the frozen file. Thresholds not changed.

## User decision (2026-11-05): "Accept, flag 3"
`_04` is adopted for all 11 plants. R3-1b, n_03 and e_01 carry `user_accepted_despite_reconstruction_check_fail` (`build/mtg_build_v2_clicks_20261105_04/PLANT_FLAGS.json`). Static MTG per-leaf light on `_04` stays with the Engine static session.

## Outputs
This folder: `curv_lib.py`, `noise_floor.py`, `build_04.py`, `leaflet_curvature.csv`, `noise_floor.csv/json`, `mtg_geometry_check.csv` (per plant and per leaf), figures, `MANIFEST.json`, `FINAL_HASHES.json`, lock. Set `build/mtg_build_v2_clicks_20261105_04/`. `MTG_LEAF_CURVATURE_v1.csv` + `.sha256`. Korean note `handoff/MTG_FOLLOWS_RECONSTRUCTION_2026-11-05_KO.md`, LOG lines. `_01`–`_03`, templates, arrangement v1: unchanged.
