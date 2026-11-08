# CONTRACT: surfel normal noise and per-leaf light (normal smoothing test)

Written 2026-11-04, before any number of this run. Session: Engine static. Binding prompt: "CLI PROMPT — Engine static: surfel normal noise and per-leaf light (normal smoothing test)", 2026-11-04 14:05 (user decision 14:05). It is reproduced in §0.
Labels on every output: `normal_smoothing_test`, `relative_light`, `conditional_on_unvalidated_scale`, `skeleton_labels_ungraded`, `after_checkpoint_v1`. This is a sensitivity / representation test. Nothing here is measured or validated.
Input sha256 hashes: `CONTRACT_HASHES.json`, written before the Kaggle push.

## 0. Prompt (summary of the binding text)
- **Why:** the static tracer uses each surfel's own normal for the incidence cosine. Traits measured a neighbour-normal noise of 42.9° on R3-1b. Test whether noisy normals bias plant and per-leaf absorbed light low.
- **§1** Add the option `normal_mode` (raw = v1; smooth_r2/r3/r5).
- **§2** Synthetic recovery check, with a declared selection rule.
- **§3** Real plants: pilot + nine + e_01/04/05/07; normal noise before and after smoothing; disc-vs-triangle link; scenario robustness.
- **§4** MTG cross-check.
- **§5** Outputs and report. The Do-not list is in §9 below.

## 1. Option `normal_mode` (new; the frozen v1 files are untouched)
- **Code:** the v1 static package is used as byte copies, unchanged (nine bundle `static/` for §2–§4; tau bundle `static/` with `use_v1=True` for the disc-vs-triangle and scenario parts, as in the runs being repeated).
  - New code lives only in `normals.py` (this folder and the dataset). After a plant is placed, it sets the receiver normal array `q.normal` to a new array.
  - `raw` = no call, so the v1 path is untouched (regression §6).
- **`smooth_r{2,3,5}`** (r = 2, 3, 5 mm, board metres × 1000):
  - For each surfel, take a weighted PCA plane fit over the surfels of the **same label set** whose centres lie within r of it. The set includes the surfel itself.
  - Weight w = per-surfel area (b) (P1 silhouette-calibrated, mm²) × PLY opacity.
  - The normal is the eigenvector of the smallest eigenvalue of the weighted covariance about the weighted centroid.
  - Sign: the side with a non-negative dot product with the raw normal.
  - Fewer than **6** points in the neighbourhood (self included): the raw normal is kept, and these are counted.
- **What changes:** only the incidence normal of receivers (cosine in `cabinet_kernel`).
- **What stays raw:** the tracer also uses normals in two other places, and both keep the **raw** normal in this test.
  - (a) The occluder disc plane: primitive columns 9–11 plus tu/tv, from `Surfels.prims()` → `s.normal`, a different array from `q.normal`.
  - (b) The coincident-surfel probe `tracer.coincident` (it offsets along the normal). For surfel plants it only feeds the half-credit-on report line, which v1 has off; it is not recomputed for smoothed modes.
- Disc positions, sizes, d_ex 2.4 mm, the coplanar rule, lamp, field, dome, seeds 101/202 and black leaves (v1) are all unchanged.

## 2. Label sets for smoothing and per-leaf light (files hashed)
| plant | file | route / skelver |
|---|---|---|
| R3-1b | `build/engine_static_tau_20261031_01/kaggle_data/labels_rachis.npz` (`axis_kind`, `label` = R8_pc leaf) = `leaf_R8pc` of `per_surfel_D.npz` used by the MTG set | sets: R8_pc leaf k blades (axis 0, label k ≥ 1); stem (axis 1); rachis-like (axis 2); unassigned blade (axis 0, label ≤ 0) |
| n_01..n_05 | `build/p1_dz10_labels_20261101_01/labels_tprox_<id>_<skelver>.npz` | tprox; 5e803db0e8f8, 6e44394dd4d8, d267404f6ada, a40266c80d19, bcf8d6cef69a (the `_02` MTG specs) |
| n_06 | `labels_distance_n_06_64b0f9b553e9.npz` | distance (as the `_02` MTG spec) |
| e_01, e_04, e_05, e_07 | `labels_tprox_<id>_<skelver>.npz` | 26f27427b73b, 73fdd1ffeead, 524f23114149, 16690f4f8d25 |
| t2_02, t2_04, t2_05 | `labels_tprox_<id>_<skelver>.npz` (not in the MTG set; the graded snapshot) | a6e7d3823d42, 6e2e267877ca, f67c164d6f3c |

- **dz10 files:** label k+1 = `leaf_uids[k]`, status 1 = assigned; one set per leaf. All other surfels (status 0 or label 0) form one `unassigned` set, reported separately. These labels have no contested and no stem class (stated).
- **Per-leaf light:** sum over assigned surfels only (lower bound).
- **Area (b) per surfel:**
  - R3-1b and the nine: `static_surfels.csv` (nine bundle).
  - e_*: P1 area (a) × canopy r from `build/engine_static_area_e_20261103_01/area_summary.csv`, recomputed locally with the unchanged `tomato3d/plant_model` code. Check: the same recomputation on n_03 must equal `static_surfels.csv` to ≤ 1e-5 mm².

## 3. Synthetic check (§2 of the prompt)
- **Deviation, declared now:**
  - No engine run geometry (consumed.csv) is stored locally, and producing new engine plants would need GroIMP.
  - The synthetic plants are therefore the **7 engine-format MTG meshes** of `build/mtg_build_v2_clicks_20261103_02/static/` (R3-1b, n_01..n_06; `.obj`, written by the engine writers). Their leaflet triangle normals are known exactly.
  - Leaves = `Lxx__*` objects; stem / rachis objects = label 0. Sampling seed **20261104**.
- **Surfel sampling** (measured from the real plants in the same kernel before any synthetic trace; values recorded):
  - **Density:** ρ = median over the 14 real plants of (opaque surfels / Σ area (b)). Per object, N = round(ρ × area_mm²), at least 1. Positions are area-weighted uniform on the object's triangles. Receiver area = object area / N.
  - **Disc size and opacity:** (a, b, opacity) triples bootstrapped jointly from the pooled real tracer discs. tu is a random direction in the disc plane.
  - **Position noise:** an offset along the true normal, N(0, σ_off). σ_off = median over the 14 real plants of the RMS point-to-plane distance in the r = 2 mm neighbourhoods of the noise metric (same samples).
- **Normal noise:** each surfel's frame (n, tu, tv) is rotated by angle |N(0, σ)| about a random axis ⊥ n (the Traits form).
  - **Calibration:** σ is chosen per target (**43°**, 30°, 55°) on a grid 0–90° in 2.5° steps. It is the σ whose noise metric on the R3-1b mesh sample (noise draw seed 20261104) is closest to the target.
  - The metric is then reported for every synthetic plant.
  - **Noise metric** = the Traits function: median angle between a surfel normal and the unweighted PCA normal of its neighbours within 2 mm (≥ 6 points), over 20 000 random surfels, seed 0.
- **Scenes per plant and noise level** (cabinet v1, d_ex 2.4 mm, seeds 101/202):
  - **truth** = noise-free discs as occluders and receivers (one per plant).
  - **noisy** = noisy discs as occluders, with receivers in `raw`, `smooth_r2`, `smooth_r3`, `smooth_r5`.
  - **receiver_only_truth** (secondary) = noisy occluders, true receiver normals. It separates the occluder part that smoothing cannot change.
- **Metrics vs truth:**
  - plant bias % (all surfels; mean and range over the 7 plants);
  - per-leaf absolute error % (leaf absorbed / truth − 1; pooled median over all leaves of the 7 plants);
  - per-plant Spearman ρ of leaf absorbed vs truth (median over plants);
  - self-shading change (mode − truth; self = 1 − absorbed / absorbed without occluders).
- **Declared rule:**
  - A radius **recovers** if, at 43° noise, |mean plant bias| ≤ 3 % **and** the pooled per-leaf median absolute error ≤ 5 %.
  - Chosen r = the smallest radius that recovers. This is applied mechanically to the synthetic table; no other choice is made after seeing numbers.
  - If none recovers: `not_recovered`. §3 still runs for information only, and the parts that need one radius (disc-vs-triangle, scenarios, MTG cross-check) report r = 5 mm, labelled `information_only`.
  - All three radii are computed everywhere (cheap). The chosen one is the headline; the others are supplementary rows.

## 4. Real plants (§3 of the prompt)
- **Plants:** R3-1b, n_01..n_06, t2_02, t2_04, t2_05, e_01, e_04, e_05, e_07 (14), isolated at the reference spot.
- **Settings:** same as static-nine (`obs.Placed`, P1 area (b), cabinet v1, field + extension, d_ex 2.4 mm, seeds 101/202, 3 bars × 8 + 72 dome directions).
- **Per plant × mode:**
  - plant absorbed and mean relative light (absorbed / area (b));
  - self-shading;
  - per-leaf absorbed and light per area;
  - Spearman ρ of leaf ranks vs raw;
  - share of surfels that kept the raw normal (and per label set: leaves / unassigned / stem / rachis-like);
  - noise metric median and p90 before and after smoothing (same Traits function on all surfels).
- **Disc-vs-triangle:** the `k_dvt` path (tau bundle, `use_v1=True`). D receivers get the smoothed normals (R3-1b label sets above); T is unchanged (`T_triangles.csv`, 16 samples per triangle, half credit).
  - Report T / D_blades and T / D_all for raw (must reproduce 1.38 / the stored values) and for each radius, next to 1.38.
- **Scenarios:** `tau_runs.SCEN` with the v1 kernel.
  - Scenes: R3-1b alone (lamp 250/350, lateral −50/+50) and the 2×2 tray (n_04, t2_02, n_01, t2_05; baseline 300 / 0 / 60, plus lamp, lateral and spacing 90/120).
  - Smoothing is applied to every receiver in the scene.
  - Per unit (each plant, tray total), relative effect = scenario / baseline − 1 for raw and for the mode.
  - **Sign same** per scenario × unit. **Ranking same** = the order of the scenario effects within each unit is identical; Kendall τ is also given.

## 5. MTG cross-check (§4 of the prompt)
- **Plants:** the 11 plants of `_02`. Mesh = `static/mtg_<id>_engine.obj` (board mm), placed with the **same offset** as the plant's surfels (`obs.Placed` base rule).
- **Receivers:** 16 points per leaflet triangle (sample seed = trace seed), delta 1e-7, coincident half credit (as T). Occluders: every triangle of the mesh. Same cabinet and seeds.
- **Per leaf uid:** MTG leaflet absorbed vs surfel absorbed (chosen radius, and raw).
  - R3-1b: MTG uids are summed per R8_pc leaf through `attrs.pipeline_leaf` of `data/skeletons/R3-1b/2026-10-21/skeleton_v1.json`.
  - Spearman ρ over leaves present in both with area > 0.
  - Plant total ratio: MTG leaflets / surfel assigned leaves, and MTG leaflets / all surfels.
- Report only; neither side is the truth.

## 6. Regression gates (stop and report if any fails; no setting is changed to make one pass)
- **G0a:** `raw` per-surfel relative light (two-seed mean, d_ex 2.4) for R3-1b and the nine equals `build/engine_static_nine_20261028_01/kaggle_out/<id>/per_surfel_light.npz` `relative_light`, max |Δ| = 0 (byte-identical arrays).
- **G0b:** the disc-vs-triangle D v1 equals `r31b_ref.npz` (max |Δ| = 0, as in k_dvt); raw T / D_blades equals the stored `results.json` ratio to ≤ 1e-9 relative.
- **G0c:** raw scenario totals equal the stored v1 values (`engine_static_tau_20261031_01/out_k*/` scenario json) to ≤ 1e-9 relative.
- **G0d:** e_* area recomputation check (n_03) as in §2.

## 7. Compute
- Kaggle CPU only: one private dataset `dolgfish/static-normals-20261104` and two kernels.
  - **K1** (nine-bundle v1): G0a, real plants, synthetic, MTG.
  - **K2** (tau bundle): G0b, G0c, disc-vs-triangle, scenarios.
  - Runtime estimate ≈ 9 s per 77 k-surfel trace → under 1 h each.
- Local: dataset assembly (file copies, label/area extraction for e_*), figures and reports from the kernel outputs; memory ≤ 1 GB. No secrets in files or logs.

## 8. Outputs
- `synthetic_normals.csv`, `real_normals_summary.csv`, `per_leaf_light_<mode>_<plant>.csv`, `disc_vs_tri_smoothed.csv`, `scenario_signs.csv`, `mtg_vs_surfel_leaf_light.csv`.
- Figures: bias vs radius; noise histograms before / after; per-leaf raw vs smoothed scatter per plant; pilot 3D light map raw vs smoothed (front 45°, same colour scale).
- Report `handoff/ENGINE_STATIC_NORMALS_2026-11-04_KO.md`, LOG line, hashes, flow row, lock.

## 9. Do not
- Do not change v1 defaults or overwrite earlier static results.
- Do not choose the radius after seeing real-plant numbers.
- Do not change occlusion geometry.
- Do not call smoothed results measured or validated.
- Do not use validation plants.
- Do not stop a part without reporting it with options.
