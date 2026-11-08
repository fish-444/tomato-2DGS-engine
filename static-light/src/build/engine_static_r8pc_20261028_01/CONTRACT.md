# Static engine on R3-1b with the designated latest P1 labels (R8_pc + rachis) — CONTRACT (2026-10-28, before any run)

Labels on every table: `relative_light` · `conditional_on_unvalidated_scale` · `labels=R8_pc designated_not_adopted (gate failed)` ·
`light_model=v1 d_ex=2.4 mm` · `biological_validation = not_established`.

## Pre-checks
- No `engine_static_r8pc_*` exists in `build/` or in LOG.
- Locks:
  - `orchestrator/RUNNING.lock` belongs to the orchestrator (pid 16864, alive).
  - `build/engine_dynamic_20261026_01/RUNNING.lock` belongs to engine B.
  - `build/viz_r31b_light_heatmap_20261027_01/RUNNING.lock` is mine; the heatmap is waiting on S2px.
- No Java is running, and available RAM is 0.61 GB.
- `orchestrator/jobs.yaml` S2px has a command; the running orchestrator read the file before that was added.
- All 10 files in `build/p1_pc_absorb_20261027_01/LATEST_PIPELINE.json` match their sha256, including R7 `vp_leaf4/labels.npz`, R8_pc
  `labels.npz` and `R8_pc_rachis/labels_rachis.npz`.

## Case (a) — surfel scene, decided from the code
- `static/obs.py::Placed.prims()` returns `geometry.Surfels.prims()`. Every opaque surfel of `R3-1b_surfels_full.ply` (46,719) becomes a disc
  occluder and receiver, whatever its label.
- Labels enter only `obs.summarize` (leaf mask, `leaf_id`) and the aggregation. `trace_plants` and `tracer.cabinet_kernel` never read a label.
- So the geometry is identical under R7 and R8_pc. **No new geometry and no GroIMP:** one per-surfel trace (job S2px), re-aggregated under both label sets.
- **No JVM.** The static tracer is numba, not GroIMP, so GroIMP's one-JVM rule and B5's priority are not touched.

## Light configuration — copied from `engine_static_20261026_01`, Run 2 reference (R3-1b alone, pot centre (280, 210) mm)
- Measured 46-point field 2026-10-13, bounded, with the model extension above z 150 mm.
- Three user_assumed bars: x centre 300 mm, y 55/105/155 mm, z 300 mm, half-angle 60°. Dome diffuse share 0.6.
- Placement and seeds 101/202 are the same as that run.
- **Optics: black leaves (absorptance 1, τ 0, ρ 0)**, which is what that run used.
  - The prompt's a 0.85 / τ 0.07 / ρ 0.08 belongs to light model v2. Light model v2 is excluded from this run, so those optics are not applied, and this is reported.
- `relative_light` = K_shaded (absorbed per area ÷ open horizontal light at the same point).
- Exclusion distance (self-exclusion d):
  - Main d_ex = 2.4 mm (follow-up contract `engine_static_20261026_02/CONTRACT.md`: t_sheet 1.529 mm, 2 × t ≥ 2.5 → 2.4).
  - Sensitivity at 1.5 / 2 / 3 / 5 mm.
  - 5 mm reproduces the accepted run (check below).
- Half-credit rule (from the STATIC gates): a receiver lying within ±1 µm of another primitive along its normal gets ½. Reported **on and off**.
  It only reallocates credit; the trace is unchanged.

## Area
- Main: P1 silhouette-calibrated (b) per surfel, `p1_plant_model_20261026_01/plants/R3-1b/static_surfels.csv`. This is Tier 0:
  canopy ratio 0.0572 → 6,031.2 mm²; stem 0.305.
  - It does not depend on the labels, so the canopy area is the same under R7 and R8_pc.
  - Absorbed light = U_ext · K · A_sil.
- Reference: STATIC's cube-union (b) area. It produced the accepted run's 28.13 r.u.·cm² and is used here only for the reproduction check.

## Credit rules (both label sets)
- Uses `axis_kind` from `R8_pc_rachis/labels_rachis.npz`: 0 = blade 33,042, 1 = stem 2,955, 2 = rachis-like 10,722.
- **Blade:** surfels with axis_kind 0 and a leaf id credit that leaf's blade.
- **Rachis:** surfels with axis_kind 2 and a leaf id credit that leaf's rachis bucket. This covers R8_pc_rachis absorption (9,333) and the 623 axis2 surfels R8_pc
  already painted. The bucket is kept out of blade area and blade light per area. R7 labels no axis surfel, so it has no rachis bucket.
- **Unassigned:** unlabelled blade, unabsorbed axis2, and stem.
- All 46,719 surfels stay occluders in both label sets.

## Checks (fixed now)
1. **Reproduction:** at d = 5 mm with cube-union area and the V-P cleanup labels, the leaf total equals the accepted run's 28.13 r.u.·cm² within 0.5 %
   (leaf 1 and leaf 7 likewise).
2. **Same total:** R7 total = R8_pc total within 0.1 % (case (a): the same per-surfel values).
3. **Bucket closure:** per-leaf blade + rachis + unassigned = total (exact).
4. **Canopy area:** Σ A_sil over blade + axis2 = 6,031.2 mm² under both label sets.

## Compute
- Local available RAM is 0.61 GB, below the 4 GB rule for any local tracer job. The trace (5 distances × 2 seeds, same code: byte copy of
  `engine_static_20261026_02/static`) therefore runs on a **Kaggle CPU kernel**, like R8_pc did.
- The inputs are prepared locally from light numpy work only: field planes, cos m, the PLY and the placement offset.
- The kernel output feeds S2px's done marker (`engine_static_20261026_02/results/s2px_R3-1b.*`, columns unchanged plus a d 2.4 mm column). It also feeds
  `per_surfel_light.npz` here, with leaf-id columns for V-P cleanup, R7 and R8_pc and an axis_kind column.

Not done here:
- R7, R8_pc, the P1 pipeline and the plant-model parameters are not changed.
- Nothing is tuned.
- The dynamic G-init labels are not switched.
