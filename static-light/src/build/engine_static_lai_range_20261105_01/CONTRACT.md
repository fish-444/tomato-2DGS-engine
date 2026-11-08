# CONTRACT: per-plant LAI as a range (compound-leaf boxes) + addendum: LAI and 5 mm layer transmittance, minimum vs maximum (fill surfels)

Written 2026-11-05 (17:55), before any number of this folder. Session: **Engine static** (owner of LAI and the layer transmittance in the static Result). Binding texts:
- parent: `handoff/CLI_PROMPT_STATIC_LAI_RANGE_BOX_COMPLETENESS_2026-11-05.md` (17:20; user decision 17:18 "LAI 보정 범위를 같이 냅니다") — not started by anyone before (no folder, no LOG line);
- addendum: `handoff/CLI_ADDENDUM_LAI_AND_LAYER_TRANSMITTANCE_MIN_MAX_BOX_FILL_2026-11-05.md` (17:40; user instruction 17:35), pasted by the user into this session.
Labels: `lai_range`, `fill_surfels_assumed`, `relative_light`, `skeleton_labels_ungraded`, `conditional_on_unvalidated_scale`, `not_validation`. Reconstructed surfels, labels, area (b), the reference light model and the current results are not changed.

## 1. Inputs (hashed in `CONTRACT_HASHES.json`)
- 14 plants: opaque surfels (opacity > 0.5) of the static bundles; area (b) per surfel (P1 silhouette-calibrated; e_* from the e_* area step); leaf labels: pilot = R8_pc blades (`labels_rachis.npz`: axis_kind 0, label ≥ 1; stem = axis_kind 1 excluded), others = dz10 snapshots of the `_02` MTG set (status 1, label > 0; t2_*: graded snapshots).
- **Current LAI and layer transmittance:** `build/engine_static_v1tau_reference_20261104_01/out_k8*/<id>/plant.json` → `sensors.tau` (reference `v1tau_smoothn_r2`). **Ground area = the pot-top square 59 × 59 mm (0.003481 m²)**. Current LAI rule (`lai_pot_top`): area (b) of the plant's "leaf" surfels ÷ ground; "leaf" = the pilot's V-P cleanup leaves 1–8, **all opaque surfels for the other 13** (unassigned and stem-like surfels included). LAI_low uses this same rule (so it equals the current LAI); the leaf-labelled-only variant (Σ A_rec) is reported beside it with the difference.
- **Ranks:** main-axis rank r = `k` of the leaf in `mtg_<id>_spec.json` (`_02`; A1 node rule) for the pilot and the 10 n_* / e_* plants; t2_* (not in the MTG set): order of `attach_s_mm` along their main axis in the dz10 skeleton. A leaf attached to a lateral axis (`attach_axis_uid` of a lateral axis in the dz10 skeleton) is class **side**. Pilot R8_pc leaves: rank = smallest `k` of the MTG uids mapped to that R8_pc leaf (`attrs.pipeline_leaf`). A leaf with no rank is not filled (counted).
- **Pilot check set:** leaf-only area bounds of R3-1b (`build/leaf_only_area_r31b_20261102_01`, report `handoff/LEAF_ONLY_AREA_R31B_2026-11-02_KO.md`): leaflets 3,645 mm² (H sensitivity 2,929–4,361), + rachis 5,276, all canopy 6,031 mm².

## 2. Compound-leaf box (parent §2)
Per labelled compound leaf: PCA of its surfel centres (area-(b)-weighted). Length L along PC1 (≈ rachis direction), width W along PC2, thickness T along PC3, each = p1–p99 extent of the centres along that axis. Reported with A_rec (Σ area (b) of the leaf's surfels) and the surfel count. Unlabelled surfels: a separate share per plant.

## 3. Lower and upper bounds (parent §3–4)
- LAI_low = current rule (above).
- **Arm R:** per rank class (main r = 1, 2, …; side) the **p90** of A_rec over all leaves of that class across the 14 plants (pooled; n stated; a class with n < 5 is merged with the adjacent lower rank, repeated until n ≥ 5, stated).
- **Arm B:** A_exp = L × W × f, **f = median(A_rec / (L × W)) over the pilot's R8_pc leaves** (f and its spread reported).
- c = min(1, A_rec / A_exp). A_missing = max(0, A_exp − A_rec), **capped so that A_rec + A_missing ≤ 2 × A_rec**. Leaves with < 100 surfels or no rank: not filled (counted).
- LAI_up(arm) = LAI_low + Σ A_missing ÷ ground. Ranges [LAI_low, LAI_up(R)], [LAI_low, LAI_up(B)], ratios. Not collapsed.
- **Sensitivity (report only):** p80 / p95 for arm R; cap 1.5× / 3×.

## 4. Checks (parent §5)
- Pilot: LAI_low, LAI_up(R), LAI_up(B) against the leaf-only bounds (÷ ground) — inside / outside, report only.
- **Synthetic** (known leaf area): the 11 static MTG meshes (`_02`, engine-format, leaflet areas known) and the engine-law leaflet mesh T of R3-1b. Surfels sampled with the generator of the normals tests (density 8.525 / mm², real disc sizes and opacities, offset noise 0.394 mm, no normal noise, seed 20261105). **Removal rule (declared):** a surfel counts as reconstructed if it is seen in ≥ 2 of 8 camera views (elevation 45°, azimuth 0, 45, …, 315°): view seen = unoccluded ray to the camera direction (black discs, own disc excluded, exclusion 2.4 mm) and |n · d| ≥ 0.17. Leaf labels = true leaves; A_rec = Σ seen surfel areas (each surfel = true area / N of its object); ranks = the mesh's leaf ranks; arm R pool = the real 14-plant p90 table; f from the pilot. Report the hit rate of the true LAI inside [LAI_low, LAI_up] for both arms; no tuning.

## 5. Addendum: fill surfels and the recomputation (min / max)
- **Fill rule (declared now; never changed after results):** per compound leaf with A_missing > 0:
  1. Fit the leaf surface in its PCA frame: quadric z' = a x'² + b x'y' + c y'² + d x' + e y' + g (area-weighted least squares; R² of z'); if R² < 0.5 a plane (z' = 0). The prompt's "quadric of the curvature prompt" was not passed to this session; this quadric is declared here.
  2. Grid the box footprint (x' ∈ L extent, y' ∈ W extent) at **2 mm**; a cell is empty if no surfel of that leaf lies within **1 mm** (projected on x'y') of its centre.
  3. Fill empty cells closest-first by projected distance to the nearest surfel of the same leaf; **never beyond 6 mm**; one disc per cell at z'(x', y') with the fitted normal (gradient of the quadric), disc semi-axes = the leaf's median real disc semi-axes (the reference k 1.5 footprint rule), tu random in plane (seed 20261105); each disc adds the cell area (4 mm²) of area (b), the last one trimmed so the added area equals A_missing.
  4. Reachable cells holding less than A_missing → fill what is reachable, report the shortfall per leaf.
  5. Fill surfels carry the leaf label and the flag `fill`; optics of the current reference **`v1tau_smoothn_r2`** (rule 2 `v1tau2_smoothn_r2` is running and not adopted): τ 0.07, a 0.85. Real surfels: normals smoothed r 2 mm as the reference (smoothing over real surfels only; fill discs keep their fitted normal).
  6. Sensitivity (report only, arm R): reach 4 / 8 mm; grid 1 / 3 mm.
- **Recalculation** (same scene, seeds 101 / 202, 24 bar points / 72 dome directions, d_ex 2.4 mm, smoothing as the reference; Kaggle CPU, ≤ 2 kernels at a time, Engine v2 keeps priority): cases **min** (real surfels), **max R**, **max B** (+ the 4 sensitivity arms):
  1. LAI per plant (area ÷ the same ground; max = current-rule area + fill area);
  2. 5 mm layer canopy transmittance with the current method: horizontal one-sided sensors on a 2 mm grid over the pot square at each 5 mm layer mid-height above the substrate (z 0.060 m), cabinet kernel with leaf transmission, two-seed mean; per layer: cumulative leaf area above the pot square, transmittance; per plant profile and the extinction diagnostic;
  3. plant absorbed light (a 0.85), self-shading, the share absorbed by fill surfels.
- **Regression gate:** the min case must reproduce the current LAI and layer transmittance exactly (K8 `sensors.tau`, max |Δ| = 0); failure stops the kernel.

## 6. Outputs
Parent: `lai_range_per_plant.csv`, `compound_leaf_boxes.csv`, `lai_range_sensitivity.csv`, figures (LAI_low with range per plant; completeness by rank; box overlay for 2 plants incl. the pilot). Addendum: `lai_transmittance_min_max.csv`, `layer_transmittance_min_max.csv`, `fill_surfels_per_leaf.csv`, figures (per-plant transmittance profile with the min / max band; LAI bar with range; 2 plants incl. the pilot with reconstructed vs fill surfels in the boxes). Korean report `handoff/LAI_RANGE_BOX_COMPLETENESS_2026-11-05_KO.md` (≤ 10 lines + table, plus the addendum section ≤ 6 lines + table) with the Method sentences. LOG line, hashes, flow row, lock.

## 7. Do not
Change area (b), labels, surfels, the LAI definition, the reference light model or current results; change p90, f, the caps, the leaf-size rule, the fill rule, reach or grid after seeing results; present LAI_up / max as measured; collapse ranges into one number; skip a part on my own.

## Amendment 1 (2026-11-05 19:59 KST) — CLI ANSWER 19:55 items 1–2, written before the K15 tables / figures / report
- **Gate (item 1, accepted):** the declared gate (min case = current LAI and every 5 mm layer transmittance, max |Δ| 0) passed bit-identically in K15 run 1. My additional total-absorbed equality check (not declared) failed by ~2e-16 from summation order (mean of per-seed sums in K8 vs sum of per-surfel means); K15 run 2 sums in the K8 order. No declared value changed. Run-1 outputs kept as `out_k15a_gatefail_sumorder` / `out_k15b_gatefail_sumorder`. The check stays, labelled `extra_check_not_declared`.
- **Two maxima (item 2), named, no value change:** `LAI_up_area` = Σ max(A_rec, A_exp) ÷ ground (parent §4: area only, no placement; = LAI_up_R / LAI_up_B of `lai_range_per_plant.csv`); `LAI_max_fill` = reconstructed + placed fill surfels ÷ ground (addendum §2–3; the case used for layer transmittance, absorbed light and self-shading); `fill_shortfall` = LAI_up_area − LAI_max_fill per plant and arm (per leaf: `shortfall_mm2` in `fill_surfels_per_leaf.csv`). Transmittance band = [min, max_fill]; LAI bar shows min, max_fill and up_area. The report states in one line that the fill-placed maximum is lower because of the 6 mm reach rule, with the reach 4 / 8 mm sensitivity beside it. Reach, grid and caps unchanged.

## Amendment 2 (2026-11-05 21:20 KST) — user decision "Rerun, option b": fill-disc footprint matched to the leaf's own surfels; written before any new number
- **Why:** with the declared rule (leaf-median disc semi-axes, cell area 4 mm² credited) a fill disc's occluding footprint π a b was only 1/3.4 – 1/8.6 of the area it adds, so the max-case transmittance / self-shading did not reflect the added area (reported in LAI_RANGE_BOX_COMPLETENESS_2026-11-05_KO.md §addendum 4–5).
- **Rule (b):** per compound leaf, r_leaf = Σ π a_i b_i / Σ A_i over its reconstructed surfels (a, b = footprint-k semi-axes as the tracer's occluders; A = area (b)). Each fill disc keeps the leaf-median aspect a_med : b_med and is scaled so π a' b' = r_leaf × (its credited area): s = sqrt(r_leaf · A_cell / (π a_med b_med)), a' = s a_med, b' = s b_med. Centres, normals, tu, credited areas, grid, reach, caps, cases (R, B, R_reach4/8, R_grid1/3) unchanged.
- New files `kaggle_flat_b/fill__<id>__<case>.npz` (run-1 fill files kept); dataset `static-lai-range-b-20261105`; kernels K15c / K15d = K15 code unchanged except the input dataset. Same declared gate (min case = current LAI and every layer transmittance, max |Δ| 0) and `extra_check_not_declared`. Run-1 results kept as `out_k15a` / `out_k15b` (label `fill_footprint_median_disc`); new results `out_k15c` / `out_k15d` (label `fill_footprint_matched`).

### Amendment 2 — hold (2026-11-05 21:21 KST), before any light number: option (b) as written overshoots; run not started
- Geometry-only check (`fill_footprint_ratio.csv`, `coverage_check.json`): the reconstructed surfels' summed footprint / area (b) is 6.2–16.1 per leaf (median 9.8) because the small discs overlap many times; matching that sum makes each fill disc ≈ 2.9–4.5 mm in radius (median scale ×5.9–8.7) on the 2 mm grid.
- Top-view union coverage / area (raster 0.1 mm, case R): reconstructed surfels of the filled leaves R3-1b 1.36 / n_03 0.68; fill run 1 (median disc) 0.20 / 0.13; **option b as declared 3.37 / 1.55**; cell-sized discs (footprint = credited area, r 1.13 mm) 0.90 / 0.70.
- So (b) as declared would shade ≈ 2.3–2.5× more than the reconstructed surfels do per unit area. Not run; reported to the user with options (cell-sized discs; coverage-matched per leaf; b as declared). kaggle_flat_b is kept as a record and is not uploaded.

## Amendment 3 (2026-11-05 21:21 KST) — user decision "Cell-sized discs" (replaces the held option b), written before any new number
- Each fill disc is a circle with footprint = its credited area: a' = b' = sqrt(A_cell / π) (r = 1.128 mm for a 4 mm² cell; the trimmed last disc smaller; grid 1 / 3 mm cases use their own cell areas). Centres, normals, tu, credited areas, grid, reach, caps and cases unchanged. Top-view coverage / area of this rule (R3-1b 0.90, n_03 0.70) vs reconstructed surfels (1.36, 0.68) is reported beside the results.
- Files `kaggle_flat_c/fill__<id>__<case>.npz` (run-1 and option-b files kept); dataset `static-lai-range-c-20261105`; kernels K15c / K15d = K15 code unchanged except the input dataset; outputs `out_k15c` / `out_k15d`, label `fill_footprint_cell_sized`. Run-1 results stay as `fill_footprint_median_disc`. Same declared gate and `extra_check_not_declared`.
