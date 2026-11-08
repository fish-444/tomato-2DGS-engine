# CONTRACT — e_01, e_04, e_05, e_07: the same silhouette-corrected area (b) step as for the n_* plants

Declared 2026-11-04 (before any number of this run). Session: Engine static. Binding text:
`handoff/CLI_ANSWER_MTG_USER_DECISIONS_EAREA_DTHETA_SIDEAXIS_TRUSS_2026-11-03.md`, Part A (user "1 a").
Labels on every output: `conditional_on_unvalidated_scale`, `skeleton_labels_ungraded`, `R3-1b-style_unaccepted`. Not validation.

## 1. What "the same step" is
The area step of `build/p1_plant_model_20261026_01/CONTRACT.md` §2 and §4, Tier 0, code `tomato3d/plant_model/` **unchanged** (imported, not copied):
- Plant set: surfels with opacity ≥ 0.5 (`inputs.OPACITY_MIN`), loaded by `inputs.load_other(<id>)`.
- (a) = Σ π k² s0 s1, k = 1.5 (`areas.disc_area_m2`).
- r = `areas.silhouette_ratio` over `areas.pick_views(poses)` = **8** accepted poses evenly spaced in sorted name order, focal ≈ 3000 px,
  organ `canopy_surface` = all plant surfels (no stem class on these plants, as for n_*).
- Per surfel area_b = area_a × r_canopy (exactly what `run.static_surfels` writes into `static_surfels.csv` for n_*, which the static-nine kernel reads).

## 2. What is not run, and why it does not change the area
- The pot-rim fit (§3 of that contract) and the frame are **not** run: neither (a) nor r depends on the frame (r is a ratio of pixel areas of the
  same discs; the frame is a translation). The e_* original HEICs are not in `census_new/census.csv` (they are in `census_d1028.csv`), so
  `photo_source` finds no photo; photos enter only the rim fit, never the area. Frame for any coordinate written here: none (no coordinates written).
- Engine round trip, layers, plant_tree emission: not part of Part A.

## 3. Inputs (hashed in `CONTRACT_HASHES.json` before the run)
| item | path |
|---|---|
| surfels | `build/p1_recon_all_20261025_01/outputs/e_0k/R3-1b-style_e_0k_surfels_full.ply` |
| poses (stage 2) | `build/p1_recon_all_20261025_01/stage2_new/e_0k/poses.json` (accepted: e_01 42, e_04 42, e_05 52, e_07 41; n_* had 27–52) |
| masks (not used by the area) | `build/p1_recon_all_20261025_01/k3a/e_0k/masks/` (42 / 42 / 52 / 41 files) |
| photos (not used by the area) | census `build/p1_recon_all_20261025_01/census_new/census_d1028.csv` (per-HEIC sha256 inside; rows e_01 81, e_04 79, e_05 82, e_07 89) |
| leaf labels (v2 clicks set) | `build/p1_dz10_labels_20261101_01/labels_tprox_<id>_<skelver>.npz`, skelver e_01 26f27427b73b, e_04 73fdd1ffeead, e_05 524f23114149, e_07 16690f4f8d25 |

Photo count statement: every e_* plant has ≥ 41 accepted posed views with a mask, so the silhouette uses the same **8** views per plant as every n_*
plant (no e_* plant has fewer usable views than the minimum n_* plant, n_03 with 27).

## 4. Per-leaf area
Label file rows = the opacity ≥ 0.5 rows (checked equal before the run). label k+1 = `leaf_uids[k]`; status 1 = assigned.
Per leaf uid: area_b_mm2 = Σ area_b over its assigned surfels (**lower bound**: unassigned surfels, status 0, are not shared out), n_surfels.
A uid with no assigned surfel is written with 0 and `n_surfels` 0 (not dropped). Plant total = canopy area (b) (all plant surfels), plus the
assigned sum and the unassigned remainder.

## 5. Reproduction gate (before any e_* number is written)
The same script run on **n_03** must give canopy r and area (b) equal to `build/p1_plant_model_20261026_01/summary.csv`
(r 0.0899, area_b 4065 mm², to the rounding there) and per-surfel area_b equal to `plants/n_03/static_surfels.csv` (max |Δ| ≤ 1e-5 mm²).
If it fails: stop, report, no e_* output.

## 6. Outputs
`area_b_e_<id>.csv` (leaf_uid, area_b_mm2, n_surfels, source `recon_area_b_ungraded_labels`), `area_summary.csv`
(per plant: n_surfels, area_a, r, r_min/max, n_views, area_b total, assigned, unassigned; n_* and R3-1b rows copied from the 10-26 summary,
pilot leaves 0.0572), `per_view_ratios.json`, `HASHES.json`, LOG line, lock.

## Compute
Local CPU, one process, about 1 min per plant (the 10-26 local runtime of this exact code), memory ≪ 1 GB (60–90 k discs, 8 renders).

## Do not
Change any n_* / R3-1b / pilot value or file; edit `tomato3d/plant_model/`; tune anything after seeing a number; use validation plants.
