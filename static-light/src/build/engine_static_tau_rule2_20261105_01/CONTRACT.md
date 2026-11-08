# CONTRACT: static optics rule 2 — τ 0.07 / a 0.85 on every surfel except the stem (14 plants), reference `v1tau2_smoothn_r2`

Written 2026-11-05, before any light number of this folder (the stem table below is geometry only). **Session: Engine static** (static tracer owner). Binding prompt: "CLI PROMPT — Static optics rule v2: τ 0.07 / a 0.85 on every surfel except the stem (all 14 plants), new reference v1tau2_smoothn_r2", 2026-11-05 16:40 KST (user decision 16:38), pasted by the user into the Engine (dynamic) session and forwarded verbatim to this session by the Engine v2 session.
Labels: `light_model_v1tau2`, `smoothn_r2`, `assumed_leaf_optics`, `relative_light`, `conditional_on_unvalidated_scale`, `skeleton_labels_ungraded`. Rule-1 outputs (`engine_static_tau_20261031_01`, `engine_static_v1tau_reference_20261104_01`, tray32 v3) are not overwritten; tray32 v3 and Engine v2 kernels are not stopped.

## 0. User decision (16:38), recorded
"줄기를 제외한 모든 서펠에 이 조건 적용": every non-stem surfel (leaflets, rachis, petiole, flowers, unassigned / unlabelled) gets τ 0.07 and a 0.85; stem surfels τ 0. Sensitivity τ 0.05 / 0.10 with the same rule. This replaces the rule-1 groups (pilot: stem τ 0; nine: every axis-like surfel τ 0; e_*: nothing τ 0), which stay on record as `v1_tau` (rule 1).
- **Absorptance:** a 0.85 on every receiving surfel, stem included. Rule 1 did the same (absorbed = U·K·A·0.85 on every receiver), so nothing differs here.
- **Crossing rule unchanged:** τ of a crossing = minimum τ of its members (eps 1.529 mm); same-sheet surfels count once.
- **Normals smoothed r = 2 mm** (receiver and occluder), d_ex 2.4 mm, seeds 101 / 202, 24 bar points / 72 dome directions, P1 area (b) — as `v1tau_smoothn_r2`.

## 1. Stem definition (declared; `make_stem.py`, `stem_definition.csv`, masks in `kaggle_flat/stem__<id>.npz`)
1. Pilot (R3-1b): the existing stem class, `axis_kind` 1 (`R8_pc_rachis/labels_rachis.npz`) — the rule-1 stem.
2. Other 13: stem = opaque surfels within **d_stem = 1.5 mm** of the stem polylines and **not** carrying a compound-leaf label (dz10 snapshot status 1, label > 0). Polylines:
   - main axis: `axis.curve_mm` of `build/mtg_build_v2_clicks_20261103_02/static/mtg_<id>_spec.json` (adopted surfel fit);
   - side axes: **no `_02` spec carries side axes**, so the clicked lateral axes of each plant's dz10 skeleton are used (`fit.curve_mm`, or the clicked points where an axis has no fit);
   - t2_* (not in the MTG set): main and lateral axes of their dz10 skeletons (`p1_dz10_labels_20261101_01/t2_0x_<skelver>/skeleton_v1.json`).
   Distance = to the polyline resampled at 0.05 mm. Sensitivity d_stem 1.0 / 2.5 mm (τ 0.07 only).
3. Pilot cross-check (report only): rule 2 applied to R3-1b with its `_02` spec axis (a vertical proxy through the median leaf base; the pilot has no clicked axis).

### Stem counts (geometry; before any light number)
| plant | d_stem 1.0 mm: surfels / area (b) share | 1.5 mm (main) | 2.5 mm | polylines |
|---|---|---|---|---|
| R3-1b | 2955 / 31.89 % | 2955 / 31.89 % | 2955 / 31.89 % | mtg_R3-1b_spec.json axis.curve_mm (main, adopted surfel fit) |
| n_01 | 106 / 0.18 % | 233 / 0.36 % | 450 / 0.85 % | mtg_n_01_spec.json axis.curve_mm (main, adopted surfel fit) | n_01_5e803db0e8f8/skeleton_v1.json n_01-A02 (lateral, clicked fit) |
| n_02 | 15 / 0.07 % | 41 / 0.16 % | 112 / 0.38 % | mtg_n_02_spec.json axis.curve_mm (main, adopted surfel fit) |
| n_03 | 43 / 0.15 % | 82 / 0.27 % | 167 / 0.47 % | mtg_n_03_spec.json axis.curve_mm (main, adopted surfel fit) |
| n_04 | 11 / 0.03 % | 32 / 0.09 % | 100 / 0.28 % | mtg_n_04_spec.json axis.curve_mm (main, adopted surfel fit) |
| n_05 | 6 / 0.04 % | 14 / 0.06 % | 44 / 0.14 % | mtg_n_05_spec.json axis.curve_mm (main, adopted surfel fit) | n_05_bcf8d6cef69a/skeleton_v1.json n_05-A02 (lateral, clicked fit) |
| n_06 | 64 / 0.16 % | 159 / 0.40 % | 511 / 0.92 % | mtg_n_06_spec.json axis.curve_mm (main, adopted surfel fit) | n_06_64b0f9b553e9/skeleton_v1.json n_06-A02 (lateral, clicked fit) |
| t2_02 | 38 / 0.12 % | 102 / 0.27 % | 244 / 0.60 % | t2_02_a6e7d3823d42/skeleton_v1.json t2_02-A01 (main, clicked fit) |
| t2_04 | 196 / 0.14 % | 311 / 0.24 % | 513 / 0.44 % | t2_04_6e2e267877ca/skeleton_v1.json t2_04-A01 (main, clicked fit) | t2_04_6e2e267877ca/skeleton_v1.json t2_04-A02 (lateral, clicked fit) |
| t2_05 | 11 / 0.02 % | 28 / 0.03 % | 77 / 0.09 % | t2_05_f67c164d6f3c/skeleton_v1.json t2_05-A01 (main, clicked fit) | t2_05_f67c164d6f3c/skeleton_v1.json t2_05-A02 (lateral, clicked fit) |
| e_01 | 17 / 0.04 % | 29 / 0.06 % | 101 / 0.21 % | mtg_e_01_spec.json axis.curve_mm (main, adopted surfel fit) | e_01_26f27427b73b/skeleton_v1.json e_01-A02 (lateral, clicked fit) |
| e_04 | 84 / 0.24 % | 185 / 0.60 % | 416 / 1.40 % | mtg_e_04_spec.json axis.curve_mm (main, adopted surfel fit) | e_04_73fdd1ffeead/skeleton_v1.json e_04-A02 (lateral, clicked fit) |
| e_05 | 7 / 0.03 % | 12 / 0.07 % | 55 / 0.18 % | mtg_e_05_spec.json axis.curve_mm (main, adopted surfel fit) | e_05_524f23114149/skeleton_v1.json e_05-A02 (lateral, clicked fit) |
| e_07 | 15 / 0.07 % | 31 / 0.18 % | 73 / 0.38 % | mtg_e_07_spec.json axis.curve_mm (main, adopted surfel fit) |

**Pilot cross-check:** the stem class covers 2955 surfels = 31.9 % of area (b); the geometric rule 2 catches 34 surfels (0.25 %), IoU **0.0023** (d 1.0 / 2.5: 0.001 / 0.0056).
**Flag (reported, not acted on):** the geometric stem is 0.02–0.6 % of area (b) on the 13 plants at 1.5 mm, while the pilot's classifier stem class is 32 %; the two definitions are not comparable (the pilot axis is a proxy, and `axis_kind` 1 also holds stem-like petiole surfels). The declared values are run as given; options are listed in the report.

## 2. Runs (Kaggle CPU, ≤ 2 kernels; tray32 v3 finished, Engine v2 keeps its slots)
- **K12a / K12b (14 plants, split 5+5+4 → 7+7):** per plant, normals smoothed, arms: rule-1 τ map at 0.07 (**regression gate**), rule 2 (d 1.5) at τ 0.05 / 0.07 / 0.10, rule 2 d 1.0 and d 2.5 at τ 0.07. Plant relative light, self-shading, height thirds, per-leaf absorbed light, unassigned-light share; 5 mm-layer canopy transmittance for rule 2 d 1.5 τ 0.07; LAI unchanged (area).
- **Regression gate:** rule-2 code with the rule-1 τ map reproduces `v1tau_smoothn_r2` per surfel exactly: the two-seed kernel equals the stored K8 array (`engine_static_v1tau_reference_20261104_01/out_k8*/<id>/per_surfel.npz`, stored in float32) bit for bit in float32. Failure stops the kernel (reported).
- **K13 (after K12):** tray 60 / 120 mm self / neighbour / total shading and the 80 lamp / lateral / spacing scenario effects under rule 2 (sign and size vs rule 1 = K9 of the reference table).
- **K14 (after the reference passes):** tray32 v3 reference geometry with rule 2 (table A, flat, over trays, p 0.10 m, heights 0.30 / 0.40 / 0.50 / 0.60 m = v3 S25–S28), change vs v3 rule 1. A full v3 re-run only if the user asks.

## 3. Outputs
`stem_definition.csv`; `RESULT_TABLE_rule2.csv` with the new column `v1tau2_smoothn_r2` next to v1 / v1_tau / v1tau_smoothn_r2 and change % per item (also written into the reference table folder as a separate file, not overwriting it); tray32 reference-geometry comparison; Korean report `handoff/ENGINE_STATIC_TAU_RULE2_2026-11-05_KO.md` (≤ 10 lines + table) with a paste-ready Method sentence; LOG line, hashes, flow row, lock.

## 4. Do not
Change τ, a, d_stem, d_ex or the smoothing radius after seeing results; overwrite rule-1 outputs; stop tray32 v3 or Engine v2 kernels; call results measured or validated; skip a part on my own.

## Amendment 1 (2026-11-05 19:59 KST) — CLI ANSWER 19:55 item 4: user decision 19:53, option (c) — stem rule unified (geometric) for all 14 plants; written before any new number
- **R3-1b stem** = opaque surfels within d_stem (1.5 mm; arms 1.0 / 2.5) of its stem polyline and without a leaf label (axis_kind 0 and R8_pc label ≥ 1) — the same rule as the other 13. **Polyline source for the pilot:** main axis only = `axis.curve_mm` of `mtg_build_v2_clicks_20261103_02/static/mtg_R3-1b_spec.json` (identical to the `_04` spec axis; source `fitted_to_surfels_ungraded`, start `vertical_proxy_through_median_leaf_base`, flag `axis_is_vertical_proxy`). **The pilot has no side axes:** its skeleton (`data/skeletons/R3-1b/2026-10-21/skeleton_v1.json`) has 0 clicked axes and an empty stem_display polyline (only leaf rachis polylines). Masks already stored before this amendment in `kaggle_flat/stem__R3-1b.npz` as `stem_rule2_geometric_stem_d{1,1.5,2.5}` (d 1.0 / 1.5 / 2.5 = 17 / 34 / see stem_definition.csv surfels); the kernels map these to `stem_d*` for R3-1b. The classifier stem class is no longer used for τ in rule 2.
- **Runs (≤ 2 kernels at a time):** K16 = K12 code, R3-1b only (all arms incl. the rule-1 gate, sensors); K17 = K13 (all 80 scenario rows contain R3-1b: alone, and centre of the 60 / 120 mm trays); K18 = K14 (tray32 v3 reference geometry, table A S25–S28; both placement tables contain R3-1b copies). Other plants are not rerun on their own. Estimated wall time ≈ 2 h (K13 took 1 h 50 min).
- **Outputs:** new rows appended in a new `RESULT_TABLE_rule2_unified.csv` (RESULT_TABLE_rule2.csv kept; its pilot rows renamed in the new file as `rule2_pilot_classifier_stem`); report table and Method sentence updated in a new report section; label `stem_rule_unified_geometric`. τ, a, d_stem, d_ex and smoothing unchanged.
