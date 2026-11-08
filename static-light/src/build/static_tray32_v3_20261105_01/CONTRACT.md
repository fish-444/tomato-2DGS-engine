# CONTRACT: tray32 v3 — luminaire heights 30 / 40 / 50 / 60 cm on the adopted reference `v1tau_smoothn_r2`

Written 2026-11-05, before any light number of this folder. **Session: Engine static** (the session that ran tray32 v2; same importer and tracer, read-only byte copies). Binding prompt: "CLI PROMPT — Static tray32 v3: luminaire heights 30 / 40 / 50 / 60 cm on the adopted reference v1tau_smoothn_r2", 2026-11-05 01:15 KST (user decisions 01:11–01:14).
Labels on every output: `light_model_v1tau_smoothn_r2`, `assumed_luminaire_model`, `repeated_placement`, `relative_light`, `conditional_on_unvalidated_scale`, `skeleton_labels_ungraded`, `scenario_comparison_not_validation`, `absolute_efficiency_uncertain_up_to_1.9x`. v2 outputs (`build/static_tray32_v2_20261104_01/`) are not touched.

## 0. User decisions (01:11–01:14), recorded
1. Luminaire heights **0.30, 0.40, 0.50, 0.60 m** (floor = pot bottom z = 0 to the lowest point of the luminaire: flat panel / V vertex / outer-edge single face's lowest edge).
2. Optics = the adopted static reference **`v1tau_smoothn_r2`** for all four heights: τ 0.07, a 0.85, stems / axis-like τ 0 (e_*: τ 0.07 on every surfel, as v2), **normals smoothed with r = 2 mm for both the incidence cosine and the occluding disc orientation** (`normals.py` of `build/engine_static_normals_v2_20261104_01/`: weighted PCA over same-label-set surfels within 2 mm, weights area (b) × opacity, ≥ 6 points else raw kept, sign closest to the raw normal; disc frame tu' = tu minus its n'-component, tv' = n' × tu'; label sets as the normals test: R3-1b R8_pc leaves / stem / rachis-like / unassigned, other plants dz10 leaves / unassigned). 40 and 50 cm are re-run with smoothing.
3. Everything else unchanged from tray32 v2 (`build/static_tray32_v2_20261104_01/CONTRACT.md`): four 4 × 2 trays, gaps 0.10 m, spacing 0.06 / 0.10 / 0.14 m; plants keep their grow-frame z with **substrate surface at 0.060 m** (the plant files' value; the prompt's 70 mm is the pot rim, stated as in v2); luminaire shapes (flat 0.10 m; V 120° / V 90°, wings 0.05 m), length 0.60 m, positions (over trays; over gaps with half-output outer single faces, flat half-width end panels), equal output 4 L, luminaires only, m = 1, 384 sample points per luminaire (1536 per scene), d_ex 2.4 mm, disc footprint k = 1.5, receivers 8000 per copy (PPS by area (b)), K_e = 32, ground K_e 128, measures, closure definitions, reference scene **flat, over trays, 0.40 m, p = 0.10 m**.

## 1. Placement tables (same files as v2, verified by hash)
| table | sha256 (this folder) | sha256 (v2 CONTRACT_HASHES) |
|---|---|---|
| A | 8d1c682d3ddefef0bac6dbac087ddfd52d675e610d30a077dfad938776a98622 | 8d1c682d3ddefef0bac6dbac087ddfd52d675e610d30a077dfad938776a98622 |
| B | 6b05aaac8b4a73f29e0a7daf44a009e9ffb7bd93105e9fdcd4f3332808625199 | 6b05aaac8b4a73f29e0a7daf44a009e9ffb7bd93105e9fdcd4f3332808625199 |

## 2. Scenes and seeds
3 spacings × 3 shapes × 2 positions × 4 heights = **72 scenes per table** (S01–S72; order spacing → shape → position → height), tables A and B → 144. Emitter draws per scene use **the v2 seed for 40 / 50 cm** (`seed_k` = v2 scene number) and 100 + the v3 number for 30 / 60 cm; receiver draws are those of v2. Hence v3 with smoothing off reproduces v2 bit for bit.
Isolated-copy runs at every scene for table A; table B too if time allows (reported). Order: **table A all 72 → short report → table B**.

## 3. Clearance check (done before any light run; geometry only; `clearance.csv`, `clearance_summary.json`)
Per table × scene × copy: vertical gap = luminaire lowest point − highest surfel; 3-D distance to the nearest luminaire sample point. **Result: 0 flagged copies of 4608**; minimum vertical gap 0.1141 m (at 0.30 m height; tallest plant e_04 top at 0.186 m). No scene is marked `near_luminaire` (the column is kept in every table).

## 4. Checks (fixed rules)
- **Regression (before reading v3 numbers):** smoothing off, table A, 0.40 m, p 0.10 m, one scene per shape × position (S26 flat-trays, S30 flat-gaps, S34 V120-trays, S38 V120-gaps, S42 V90-trays, S46 V90-gaps) must reproduce the v2 table A scene JSONs (S13, S15, S17, S19, S21, S23) exactly (absorbed totals and every copy; difference reported). The local smoke run already matched exactly (8 copies).
- **Emission integral:** every height, shape and position: worst relative deviation 1.7e-07 (≤ 0.5 %, `LOCAL_C2_v3.json`), 1536 points in every scene.
- **Ray-count convergence:** on the **30 cm reference geometry (S25: flat, over trays, 0.30 m, p 0.10 m)**, smoothing on: receivers and emitter draws doubled until the efficiency changes < 0.5 %. If not converged at the declared 8000 / 32, the counts are raised for all scenes in an amendment before any scene result is read.
- C1 point-source limit carries over from v2 (same kernel, unchanged code path).

## 5. Measures (as v2, plus)
Per scene / copy / leaf / position outputs of v2 (smoothing on). **Height response:** efficiency vs height per shape × position × spacing; ratio to 0.40 m; the height at which over-gaps vs over-trays changes sign, per shape (linear interpolation between neighbouring heights; "none" if no change). **Smoothing effect (report only, 40 / 50 cm):** v3 vs v2 per scene (efficiency ratio), per copy (absorbed ratio median, p10–p90), self-shading, neighbour-shading share, position effects, and whether every paired scene effect (vs the reference scene) keeps its sign. **Energy closure residual** and the shadow-based interception vs area-(b) absorption gap per scene.

## 6. Compute
Kaggle CPU, **at most 2 kernels at a time; Engine v2 keeps priority** (if its kernels are queued, mine wait until they have started). T1 = convergence + regression + table A scenes; T2 = table A isolated copies; after the short table A report T3 / T4 = table B scenes / isolated copies. Local: geometry checks, tables, figures, report.

## 7. Outputs
`scenes.csv`, `clearance.csv`, `efficiency_by_scene.csv`, `plants_by_scene.csv`, `leaves_by_scene.csv`, `position_effect.csv`, `scenario_effects.csv`, `height_response.csv`, `smoothing_vs_v2.csv`; figures: efficiency vs height (one panel per spacing, lines = shape × position); heat map of copy absorbed light at 0.30 / 0.60 m for the reference geometry; v2 vs v3 scatter per copy at 0.40 / 0.50 m. Korean report `handoff/STATIC_TRAY32_V3_2026-11-05_KO.md` (≤ 12 lines + tables) with a paste-ready Result sentence in Korean ("파일럿 개체", no internal names) and English. LOG line, hashes, flow row, lock.

## 8. Do not
Change placement tables, layout, luminaire geometry, optics, k, scene grid or reference scene after seeing numbers; overwrite v2 outputs; drop near-luminaire scenes; call results measured or validated; use more than 2 Kaggle slots.
