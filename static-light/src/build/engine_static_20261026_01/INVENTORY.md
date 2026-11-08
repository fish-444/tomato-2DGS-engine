# Engine A STATIC mode — Step 0 inventory (2026-10-26, before any contract or number)

Prompt "CLI PROMPT — Engine A: STATIC mode" (user 2026-10-26). Read from the code and the build folders on 2026-10-26.

## Quantities
| quantity | status | entry point | last run folder |
|---|---|---|---|
| per-leaf **absorbed** light | **exists** (dynamic) | native `flux.csv` `absorbed_umol_s` per Leaflet: GroIMP `LightModel(300000, 1)` Monte Carlo, black optics (`photonPerJoule = 1`, depth 1 = direct only), isotropic `PointLight`s from `lamps.csv` (`tools/vtc_development_native.rgg` l.308–311) | `build/engine_maxmass_fix_20261025_01/runs/seed72` (72 d seed), `build/e7_n_uptake_adopt_20261022_01` gate runs |
| per-leaf **incident** (upper / lower face) | **partial** | `incident_diagnostics` hook (B1 collector, `tools/fsp_incident_diagnostics.py`): `incident_by_interval.csv` upper / lower / two-sided PPFD per organ; off by default | `build/h_runs_20261020_01/rp_gates` |
| per-**surfel** anything | **absent** | no path traces individual surfels | — |
| **transmittance** (plant ÷ plant removed, same point) | **partial** | the native writes an empty-scene sensor grid (`sensors.csv`, 16 fixed cells at z = 0, plant removed) but no "with plant" point sensors; `tomato3d/light/raytrace.direct_light` returns shaded and unoccluded PPFD for any receiver but is not wired to surfels; `tools/empirical_canopy.py` computes a shaded fraction for declared receivers only when direction bars are declared (the frozen field has `direction_bars: null`) | P2 synthetic demo, 2026-10-13 |
| **intercepted fraction** | **absent** as an output | derivable from absorbed ÷ lamp flux for point lamps | — |
| **LAI** | **absent** | leaf area exists per organ (`states.csv` `area_m2`) and per plant (`s1_plant_summary.csv`); no ground-area definition anywhere | — |
| **LUE** | **absent** | `gross_gC` (non-rectangular hyperbola, AMAX / ALPHA / THETA) and absorbed exist per leaflet; no ratio is formed | — |
| per-plant **attribution** (self vs neighbour shading) | **absent** | per-plant absorbed totals only; no plant-removal traces (L1 reported redistribution, not attribution) | `build/l1_lamp_spacing_20261019_01` |

## Tracer paths and external geometry
- **GroIMP XL (engine).** Can load external organ meshes only through the P1 integration route (`fsp-integration-1`: `initial_state.csv` +
  `integration_geometry.json` triangles, read at run time from `initial_shapes.csv`), with `light_only` (no growth, no sink step, gross still
  computed). That route needs a complete P1 consumer package; it cannot take raw 2DGS surfels (44–96 k primitives per plant) without one.
- **`tomato3d/light/raytrace.py`.** Loads triangle occluders and receivers; numpy, culling only by one bounding box per plant, so cost is
  rays × triangles (≈ 10¹² tests for one surfel plant). Infeasible for surfels.
- **`tools/groimp_light_scene.py` / `groimp_light_sanity.py`.** Standalone GroIMP scenes with sensors or test plates; the real XL exports
  carry zero leaf meshes.
- No `open3d` / `trimesh` / Embree in this Python; `numba 0.68` and 16 CPUs are available.
→ **First build item: an accelerated static tracer (numba, uniform grid) for triangles and surfel discs**, cross-checked against GroIMP.

## Light-source models today
| path | source model | absolute PPFD? |
|---|---|---|
| GroIMP native | isotropic point lamps from `lamps.csv` (seed/F5 runs: one synthetic 1200 µmol s⁻¹ lamp at 0.6 m) — not the cabinet's three bars | no: `synthetic_assumed_point_LED` |
| `raytrace.direct_light` | LED fixtures `I0 cos^m θ` from a LightScene JSON; `config/light_scene.json` does not exist; the P2 frozen reference is a synthetic test fixture | no |
| `empirical_light_field` | bounded bilinear-xy / linear-z interpolation of the 46 measured points (planes z = 70 and 150 mm, shelf-floor basis, grow frame, tray bounds 70–490 × 70–350 mm) | no: `meter_model_and_calibration_unknown`, `grid_origin_corner_and_axis_unconfirmed`, `lamp_pose_unrecorded` |
None gives absolute PPFD.

## The `frozen` control
`tools/tomato_fsp_acceptance.py` `frozen` = `freeze_geometry` (`freeze_at = 0`): geometry fixed, but the carbon step (demand, allocation,
reserve) still runs every interval. It is **static in geometry, not static in the prompt's sense** (no sink step). The strict static engine
run is `light_only` (`freeze_at = 0`, no allocation, gross still printed). Both need the P1 integration route.

## Inputs found
- `build/p1_plant_model_20261026_01/` **does not exist** → fallback to raw surfels as the prompt allows:
  R3-1b `build/p1_r31b_reuse_20261024_01/source/R3-1b_surfels_full.ply` (53 645 surfels) with leaf ids
  `build/p1_vp_cleanup_20261026_01/surfel_leaf_ids.csv` (46 719 opaque rows above the pot top; leaves 1–8, usable = **1 and 7**, viewing_only =
  2, 5, 6, not_in_held_out = 3, 4, 8); the nine others `build/p1_recon_all_20261025_01/outputs/<id>/R3-1b-style_<id>_surfels_full.ply`
  (t2_02, t2_04, t2_05, n_01–n_06; 43–96 k surfels; no leaf split). PLY: board metres, z up, 2DGS discs (centre, tangents tu/tv, normal,
  linear scales s0/s1 with median 0.43–0.45 mm, opacity median ≈ 1).
- Dynamic snapshot at 59.9 DAG = engine day 71 of `runs/seed72`: consumed geometry state 71 (244 organs, 38 109 vertices), flux interval 71
  (p1 absorbed 0.648 µmol s⁻¹ on 48 lit leaflets, p2 0.932), lamp L1 (0, 0, 0.6) 1200 µmol s⁻¹, plants at x = ±0.035 m.
- `microtom_synth_v2` virtual individuals: available as an extra scene (`model_generated`).
