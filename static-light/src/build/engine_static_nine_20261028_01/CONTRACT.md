# Static light for the nine other individuals (n_01–n_06, t2_02, t2_04, t2_05) — CONTRACT (2026-10-28, before any run)

Tags on every table: `relative_light` · `conditional_on_unvalidated_scale` · `biological_validation = not_established` · `light_model=v1 d_ex=2.4 mm` ·
`lamp_bar_position = assumed` · `optics = assumed (v1 black leaves)`.

## Pre-checks (2026-10-28)
- No `engine_static_nine_*` in `build/`, none in LOG, no `RUNNING.lock` of this job. `build/scale_check_20261028_01/` does not exist → no `length_scale_checked` is reported.
- Stage 1 (`build/p1_pc_all_20261028_01/<id>/STAGE1.json`) is `waiting_stage1` for all nine → **no R8_pc labels exist; Tier 2 cannot be reached** for any individual in this job.
- 3D-first groups exist for all nine (`build/p1_all_3dfirst_20261027_01/<id>/merge/groups.npz`, `flags.csv`). `user_3d_group_click.jsonl` has 19 records, all n_04, all action `leaf`, leaf 1, final set = 16 groups
  (90, 94, 95, 96, 97, 98, 101, 105, 106, 153, 169, 181, 199, 212, 230, 243). The prompt names 153 and 169; the file now holds the later 16-group set, which is used (last record per group wins).
- All inputs are read-only. P1 area (b) per surfel: `build/p1_plant_model_20261026_01/plants/<id>/static_surfels.csv` (silhouette ratio × area (a), per organ). It is available for all nine.

## Compute: external only (Kaggle CPU kernel, no local ray tracing, no JVM, no GroIMP)
- The static tracer is the numba code in `engine_static_20261026_02/static` (byte copy in the dataset; sha256 of the five files recorded in every output). The Kaggle route was already shown to run it on 2026-10-28 (R3-1b trace, 58 s).
- Local use: copy files into one private dataset, read small JSON results, write reports. Local memory stays ≤ 1 GB; no aggregation or tracing locally.
- One kernel, one private dataset (`dolgfish/static-nine-bundle-20261028`), not one kernel per individual: the whole run is expected to take minutes, so per-individual kernels would cost more start-up than they save.
  Individuals run sequentially in the required order; each writes `<id>/plant_light.json`, `<id>/per_surfel_light.npz` and `<id>/DONE` as soon as it finishes (checkpoint unit = individual;
  a re-push with `INDIVIDUALS=...` resumes the rest). Secrets: none are used in the kernel; nothing secret is written to logs.
- **Gate 0 (reproduction, before any individual).** The kernel first runs R3-1b through the same function as the nine and compares it with the existing local result:
  - (a) plant-level absorbed light at d 5 mm, V-P cleanup leaf surfels, cube-union (b) area, seeds 101/202 = `engine_static_20261026_01/results/run2.json` (0.0028128618 r.u.·m²);
  - (b) per-surfel `relative_light` at d 2.4 and 5 mm = the local R3-1b arrays from `engine_static_r8pc_20261028_01/per_surfel_light.npz` (reference file in the dataset).
  Rule: relative difference ≤ 1 % on (a) **and** on the plant-mean of (b). If it fails the kernel stops, writes `external_blocked` with the numbers, and nothing else runs. No setting is changed to make it match.

## Light settings (identical to the R3-1b runs; unchanged)
- Isolated plant at pot centre (280, 210) mm, grow frame, `obs.Placed` placement (stem-base proxy xy; board z used as is). No tray neighbours (B5's job).
- Surfel scene: every opaque surfel is an occluder and a receiver, two-sided, whatever its group or flag.
- Measured 46-point field 2026-10-13 (bounded, linear in z); above its top plane (z 150 mm) the model extension (U_clamped × model ratio) is used and the out-of-support area fraction is reported.
- Three user_assumed bars (x centre 300 mm, y 55/105/155 mm, z 300 mm, half-angle 60°), diffuse share 0.6. `lamp_bar_position = assumed`.
- Self-exclusion d_ex = 2.4 mm main; sensitivity 1.5 / 2 / 3 / 5 mm (5 mm = old reference). Seeds 101 / 202.
- **Optics:** the R3-1b runs used black leaves (absorptance 1). The prompt's a 0.85 / τ 0.07 / ρ 0.08 belong to light model v2, which is not part of these runs, and "identical to the R3-1b run" governs. v1 black leaves are used and stated in every report.
- Half-credit rule (±1 µm coincident surfels, as declared): main off, on reported (it only reallocates credit).
- "Ray count": the tracer is analytic (no sampled rays). Per receiver: 24 bar sample points (3 bars × 8) and 72 dome directions. Plant-level CV between the two seeds is reported for every individual,
  first for t2_02. If any CV > 5 % the kernel reruns that individual once with 16 samples per bar and 12 × 24 dome directions, and the report says so.

## Per individual
- Area (b): P1 silhouette-calibrated per-surfel area (canopy ratio and stem ratio from `static_surfels.csv`). The summed (a) is never reported as plant area. Cube-union (b) per surfel is kept as a second column (reproduction only).
  A row-set check (`surfel_row` = opaque PLY rows) stops the individual if it fails.
- `scale_status` = `rim_fit_ok` (R3-1b, n_02, n_03, n_06, t2_02, t2_05) or `rim_fit_failed` (n_01, n_04, n_05, t2_04), from `summary.csv` `rim_visual`. All areas `conditional_on_unvalidated_scale`.
- Tier 0 (every individual): plant-level absorbed light, area-weighted mean relative light (absorbed ÷ area (b)), self-shading = 1 − absorbed ÷ absorbed-with-no-occluder (U-weighted, as in the R3-1b follow-up),
  height profile in 5 mm bands (height above the plant's lowest 0.5 % surfel, primary; board z also given), out-of-support fraction, and the unassigned bucket:
  light on `axis_like` (stem/rachis-like) surfels and on blade surfels with no group (group −1). That bucket is the upper bound on what labels could still move.
- Tier 1: light per 3D-first group (top 10 by light, with their check flags from `flags.csv`), named "소엽 묶음", never "잎". n_04: clicked set → leaf 1 as a lower bound.
- Tier 2: not reached (no R8_pc). `per_surfel_light.npz` carries `group_id`, `leaf_id` (−1 if none), `label_source`, `flags` so a later re-aggregation needs no new trace.
- `label_source` per surfel: `user_3d_group_click` (n_04 clicked groups), `group_only` (other grouped surfels), `none` (no group). Never mixed silently: the tag is a column and the per-individual primary source is the highest tier available.

## Comparisons (descriptive only)
Plant-level self-shading and relative light against height and surfel count across R3-1b and the nine. No tests, no ranking of individuals.
Rim-failed individuals (n_01, n_04, n_05, t2_04): absolute heights unreliable, relative light usable.

## Schedule
All nine are expected in one kernel run. Anything not finished by Friday 2026-10-30 11:00 is `not_run`. `external_blocked` (JVM absent, quota, session limit) is written to `handoff/STATIC_NINE_STATUS_KO.md` with the exact reason; no local fallback.

## Not done here
Labels, thresholds, optics, lamp, d_ex and the half-credit rule are unchanged; no per-leaf result for an individual without R8_pc or clicked leaf numbers; no guessing of leaf numbers; frozen engine files and `build/engine_static_*` are read only.
B5 uses the planar 6+2 tray scene; which individuals it takes is not decided here (it needs the tray layout).
