# CONTRACT: v1_tau as the static light reference — Result table v1 vs v1_tau (Part 1)

Written 2026-11-04 (17:05), before any file or number of this folder. Session: Engine static. Binding text: "CLI ANSWER — adopt leaf transmission (`v1_tau`) as the static light reference", 2026-11-04 16:54, Part 1 (Part 2 = Engine v2, not touched here).

## 0. User decision (16:54), recorded
"투과를 한 걸로 채택": from now the static light results are reported with `light_model = v1_tau`: leaf transmittance τ = 0.07, absorptance a = 0.85 (reflection dropped), stem / axis-like τ = 0, as defined and tested in `build/engine_static_tau_20261031_01/` (R0 byte-identical at τ = 0, T1 and probe test pass, declared T2 FAIL kept on record, labelled second attempt T2b PASS).
- v1 (black leaves) stays as the earlier version, kept and not overwritten. Closed earlier results that used v1 keep their v1 label (§6).
- Labels on every reported number: `light_model_v1_tau`, `assumed_leaf_optics`, `relative_light`, `conditional_on_unvalidated_scale`.
- The optics values are assumed. The 10-15 measurement shows transmission exists but gives no value (Micro-Tom literature τ ≈ 0.015). The already-run τ = 0.05 / 0.10 columns are the optics sensitivity.
- τ and a are not changed after seeing results. a = 0.85 is part of the adopted set; the a = 1 column (+2.9 … +4.8 % vs v1) is shown beside it, so transmission and absorptance are separable.

## 1. What is collected from existing v1_tau outputs (no new run)
From `build/engine_static_tau_20261031_01/` (`plant_tau_table.csv`, `per_leaf_tau.csv`, `T2_decomposition.json`, `out_k1b/`, `out_k2/`, `out_k3/`, `tray_scenarios_tau.json`, `TABLES.md`) and the later D-vs-T runs: pilot + nine at d_ex 2.4 (plant relative light, self-shading, height thirds, per-leaf light, τ 0.05 / 0.10), R3-1b and tray (60 mm) lamp / lateral / spacing scenarios (v1 vs v1_tau relative effects), and the v1_tau columns of the disc-vs-triangle, curved-T and coplanar-separation runs (`engine_disc_vs_tri_20261101_01`, `engine_curved_T_20261101_01`, `engine_coplanar_sep_20261101_01`).
Every row carries its source file and keeps its original value (no recomputation).

## 2. What is missing and is run now (same seeds 101/202, scenes, d_ex, sampling counts as v1)
Code: the tau bundle (`static-tau-bundle-20261031` = byte copy of the v1 static package plus the v1_tau additions) with **no edits**; new code only in this folder's kernels. Kaggle CPU only; local work = file copies, small tables.
| id | content | plants |
|---|---|---|
| K5 | v1 and v1_tau (τ 0.07; a = 0.85 and 1 from the same trace) for the **14 plants** (pilot R3-1b, nine, e_01, e_04, e_05, e_07): plant relative light, self-shading, height thirds, per-leaf absorbed light, **unassigned-light share**, two-seed CV (random-noise check, 24 bar points / 72 dome directions; raised once to 16 / 12×24 if CV > 5 %, as in v1), **exclusion distance 1.5 / 2 / 3 / 5 mm**, footprint ×0.8 / ×1.2 (area (b) × scale²), diffuse share 0.5 / 0.7 (recombined from the traced parts, U recomputed), coincident-probe half-credit report, **LAI and 5 mm-layer canopy transmittance** (sensor grid 2 mm over the 59 mm pot square per 5 mm layer; extinction diagnostic) | 14 |
| K6 | **tray 2×2 (n_04, t2_02, n_01, t2_05; assumed layout) at pitch 60 and 120 mm**: all / alone / none absorbed light per plant and tray total for v1 and v1_tau → self-shading loss, neighbour-shading loss, total loss | 4 |
| K7 | **static–dynamic consistency and GroIMP closure** with the same τ: static T (engine-law leaflets of `engine_disc_vs_tri_20261101_01`) with the engine lamp (point lamp 1200 at 0.6 m), static v1 vs static v1_tau vs GroIMP T with the black shader (reproduction gate: GroIMP/static 0.986 of the earlier run) and GroIMP T with the engine's leaf shader `Phong(diffuse = ρ, transparency = τ)` (τ 0.07, ρ 0.08 so a = 0.85; stems black), 30 M rays, seeds 11/12. If the dynamic engine snapshot stays black (Engine v2 C7 not yet run), I report both and say so. | R3-1b T |
- **e_* optics sets:** e_* have no stem / axis-like classification (no 3D-first groups). τ = 0.07 on every e_* surfel (stated; it errs toward more transmission through stems). Bracket (report only): τ = 0 on the label-0 (unassigned) surfels.
- **e_* labels / area:** P1 area (b) recomputed with the unchanged `tomato3d/plant_model` code (the e_* area step; r from `engine_static_area_e_20261103_01`); leaf labels = the dz10 snapshots of the `_02` MTG set (same files as the first normals test).
- **Unassigned bucket:** R3-1b = non-leaf of the R8_pc labels (stem, rachis-like, unlabelled blade); nine = axis-like + blade without a group; e_* = label-0 / status-0 surfels (lower-bound labels).

## 3. Regression gates (stop and report; no setting is changed to pass)
- **G0a:** K5 v1 per-surfel kernel (d_ex 2.4, two-seed mean) equals the stored v1 arrays for R3-1b and the nine, max |Δ| = 0.
- **G0b:** K5 v1_tau per-surfel kernel equals the stored `out_k1b` / `out_k2` `per_surfel_light_tau.npz` arrays, max |Δ| = 0.
- **G0c:** K6 pitch 60 mm 'all' totals equal the stored `tray_scenarios_tau.json` baseline (v1 and v1_tau) within 1e-9 relative.
- **G0d:** K7 static v1 T total and the GroIMP black-shader ratio reproduce the earlier values (static T 1736.43 / GroIMP / static 0.986 within the GroIMP packet noise).

## 4. Outputs
`RESULT_TABLE_v1_vs_v1tau.csv` (item, scene or plant, v1, v1_tau a = 0.85, v1_tau a = 1, change %, label, source), per-plant and per-scenario tables from K5–K7, Korean note `handoff/ENGINE_STATIC_V1TAU_REFERENCE_2026-11-04_KO.md` (≤ 10 lines + table header that states τ = 0.07, a = 0.85 and the a = 1 column), LOG line, hashes, flow row, lock. Table header: `light_model = v1_tau (τ 0.07, a 0.85, reflection dropped, stem/axis-like τ 0); assumed_leaf_optics; v1 = black leaves a 1`.

## 5. Normal smoothing v2 integration (item 3 of the answer)
- The synthetic v2 test (`build/engine_static_normals_v2_20261104_01/`) is running on v1 (τ = 0); its declared decision stays as made.
- If a smoothing radius is adopted, the static version becomes `v1tau_smoothn_r<X>`: the §3 re-run of that prompt on the v1_tau basis, with the Result table in three columns v1 / v1_tau / v1_tau + smoothing. This is declared in an Amendment here before it starts.
- Report-only synthetic check on v1_tau: the same generator and seeds as v2 with the leaf transmittance τ = 0.07 on leaf surfels (stem / rachis τ = 0), arms raw, truth and the smooth_all arm of the chosen radius (or r5 if none is chosen, labelled information only). It does not re-make the declared decision.

## 6. Earlier results that used v1 (kept with their v1 label; listed in the note)
Divergence-sensitivity static columns (`engine_divergence_sens_20261102_01`), B-model coupling runs, MTG per-leaf light (`mtg_*`), the first normal-smoothing test (`engine_static_normals_20261104_01`), the disc-vs-triangle / curved-T / coplanar-separation v1 columns, the static-nine Tier 0/1 tables, the e_* area step (not a light result).

## 7. Do not
Overwrite v1 outputs; change τ or a after seeing results; call the optics measured; stop a part without reporting it with options; touch Part 2 (Engine v2 C7).

## Amendment 1 (2026-11-04 17:40) — smoothing decision from the v2 synthetic test
The declared synthetic decision of `build/engine_static_normals_v2_20261104_01/` is `recovered` at r = 2 mm (`smooth_all_r2`: synthetic plant bias +0.50 %, per-leaf error 0.45 %). Per the user's answer item 3 the adopted static version is `v1tau_smoothn_r2`; the plant / scenario / tray / MTG re-run on the v1_tau basis is carried out by the kernels `K8` (14 plants), `K9` (scenarios + tray) and `K10` (MTG) declared in Amendment 1 of the smoothing contract, and the Result table here gets the smoothing columns. The K5 / K6 / K7 plan of §2 is unchanged.
