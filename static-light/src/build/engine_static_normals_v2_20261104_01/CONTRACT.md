# CONTRACT: normal smoothing v2 — smooth the occluding discs too

Written 2026-11-04 (16:45), before any number of this run. Session: Engine static. Binding prompt: "CLI PROMPT — Engine static: normal smoothing v2 — smooth the occluding discs too", 2026-11-04 16:37 (user decision 16:37).
Labels on every output: `normal_smoothing_v2_occluders`, `relative_light`, `conditional_on_unvalidated_scale`, `skeleton_labels_ungraded`, `after_checkpoint_v1`. Not measured, not validated.
Follows `build/engine_static_normals_20261104_01/` (first test: receiver-only smoothing, `not_recovered`; kept unchanged). The frozen v1 tracer and all earlier results stay unchanged.

## 1. Option `normal_mode_v2` (new, default `raw` = v1)
- **`smooth_all_r{2,3,5}`:** the smoothed normal from `normals.smooth` of the first test (weighted PCA plane over same-label-set surfels within r mm, self included, weights area (b) × opacity, ≥ 6 neighbours else raw kept, sign = side closest to the raw normal, same label sets as test 1 §2).
  - It replaces the raw normal for the **receiving cosine** and for the **orientation of the occluding disc**.
  - Disc plane after smoothing: normal n' = smoothed normal; tu' = tu with its n'-component removed and renormalised; tv' = n' × tu'. Disc centre, semi-axes a, b (along tu', tv'), opacity, exclusion distance d_ex 2.4 mm are unchanged.
- **Coplanar rule:** in v1 the surfel plants only report the coincident probe (±1 µm along the **receiver normal**, half credit off by default). It uses the smoothed receiver normal in every smoothed arm, and the occluder scene of that arm. Hits are counted for every arm (`coplanar_hits.csv`).
- **Comparison arm:** `rec_r{2,3,5}` = the first test's `smooth_r*` (receiver normal smoothed; occluder discs raw), same seeds. `raw` = v1. `truth` = noise-free reference. `occ_only_r{2,3,5}` (diagnostic, not eligible for adoption) = smoothed occluder discs with the **raw (noisy) receiver normal**. `recv_true` (diagnostic) = noisy occluders, true receiver normals.
- **Code:** only `normals.py` (extended with `reframe`, `correlated_noise`) and the kernel code in this folder; the v1 static package is used as a byte copy (nine bundle) and not edited. Regression: arm `raw` equals the v1 run (same code path, no smoothing call).

## 2. Synthetic check
- **Plants:** the 7 engine-format MTG meshes of the first test (`build/mtg_build_v2_clicks_20261103_02/static/mtg_<R3-1b,n_01..n_06>_engine.obj`). New seed **20261105** (surfel sampling, size draw, noise draws).
- **Generator (unchanged from test 1):**
  - surfel density 8.5248 / mm² (median of the 14 real plants), position offset along the normal N(0, 0.3938 mm), (a, b, opacity) bootstrapped jointly from the pooled real discs (`pool.npz` = the pooled arrays of test 1);
  - normal noise = frame rotation by |N(0, σ)| about a random axis ⊥ n (the Traits form), σ calibrated on the R3-1b mesh sample on the 0–90° grid (step 2.5°) so that the Traits neighbour-normal metric (median angle to the unweighted PCA normal of neighbours within 2 mm, ≥ 6 points, 20 000 samples, seed 0) is closest to the target.
- **Levels:** target **44°** (the measured real value, 40–43° on the pilot plant) and **30°**. The 55° level is not added again.
- **Generator limit (recorded):** in test 1 the metric saturated near 48.6° at σ = 90°; a target above ≈ 48° cannot be reached by this generator.
- **Arms** (all on the same sampled surfels and seeds): `truth`, `raw`, `rec_r2/3/5`, `smooth_all_r2/3/5`, `occ_only_r2/3/5`, `recv_true`.
- **Per arm:** plant bias % vs truth (mean and range over the 7 plants), per-leaf median |error| % (pooled), per-leaf Spearman ρ vs truth (median over plants), self-shading change (mode − truth), coplanar-rule hits (count and share of receivers), median and p90 angle between the arm's normal and the true normal, share of surfels that kept the raw normal.
- **Declared rule (same thresholds as test 1):** an arm recovers if, at the 44° level (independent noise), |mean plant bias| ≤ **3 %** and the pooled per-leaf median |error| ≤ **5 %**. The chosen radius is the smallest radius among `smooth_all` arms that recovers. No other choice after seeing numbers; receiver-only and diagnostic arms are not eligible.
- **Report-only extra:** spatially correlated noise. Per surfel a 3-vector white noise field is averaged over neighbours within 3 mm (cKDTree; rescaled to unit variance), projected into the tangent plane to give the tilt direction, and its length scaled so that the Traits metric hits the same targets (own calibration on the same grid). Same arms at 44°. It does not change the decision.

## 3. If a radius recovers
- Name: `light_model = v1_smoothn_r<X>`. Phase 2 (re-run of the 14 plants, d_ex sensitivity and random-noise check, disc vs triangle flat / folded / folded+cup, all scenarios and tray scenes, static–dynamic consistency, GroIMP closure, `RESULT_TABLE_v1_vs_smoothn.csv`, MTG cross-check) is declared in an Amendment before it starts; only the synthetic decision is reported first.
## 4. If none recovers
`not_recovered_v2` with the numbers, the remaining bias and its identifiable sources (the diagnostic arms and the normal-error columns separate: normal error of the smoothed normals, occluder-only part, receiver-only part), and options for the user. Nothing is adopted; v1 stays the reported version.

## 5. Compute
Kaggle CPU, flat-named private dataset `dolgfish/static-normals-v2-20261104` (normals.py, pool.npz, mtg__*.obj, config) plus the nine bundle for the unchanged v1 static package. Two kernels split the 7 plants (3 + 4); results aggregated locally (memory ≤ 1 GB). No secrets.

## 6. Outputs
`synthetic_v2.csv`, `correlated_noise_report.csv`, `coplanar_hits.csv`, figures (bias vs radius for both arm types; the pilot 3D light map only if adopted), Korean report `handoff/ENGINE_STATIC_NORMALS_V2_2026-11-04_KO.md`, LOG line, hashes, flow row, lock. Deadline context: final freeze Sat 2026-11-07 12:00 KST.

## 7. Do not
Change thresholds, radius list or generator after seeing numbers; adopt a receiver-only arm; overwrite v1 results or the first test; call smoothed results measured or validated; use validation plants; stop a part without reporting it with options.

## Amendment 1 (2026-11-04 17:40) — synthetic decision known: `recovered`, chosen radius 2 mm; phase 2 declared before it starts
- **Synthetic result** (`RULE_v2.json`, 7 meshes, seed 20261105, 44° level): `smooth_all_r2` plant bias +0.50 %, pooled per-leaf median error 0.45 % (r3 +1.47 % / 0.75 %, r5 +1.60 % / 0.97 %); raw −22.27 % / 23.4 %; receiver-only arms do not recover (r5 −7.19 % / 6.35 %). By the declared rule the smallest recovering `smooth_all` radius is **r = 2 mm**; light model name `v1_smoothn_r2`.
- **User instruction (16:54, item 3):** the adopted static version is v1_tau + smoothed normals = `v1tau_smoothn_r2` (τ 0.07, a 0.85, stem / axis-like τ 0, smoothing r = 2 mm on receivers and occluder discs). v1 stays as the reference version.
- **Phase 2 (same seeds, scenes, ray counts and sampling as v1; code = tau bundle unchanged + `normals.py` of this folder; the smoothing hook only replaces the receiver normal and the occluder disc orientation, tu' = tu minus its n'-component, tv' = n' x tu'):**
  1. the 14 plants: plant relative light, self-shading, unassigned-light share, per-leaf absorbed light; exclusion distance 1.5 / 2 / 3 / 5 mm; random-noise check (two-seed CV); footprint x0.8 / x1.2 and diffuse 0.5 / 0.7; half-credit report; LAI and 5 mm-layer transmittance — **for v1 and v1_tau, each with r = 2 smoothing** (so v1 + smoothing and v1_tau + smoothing both exist);
  2. disc vs triangle (flat, fold, fold+cup, coplanar offsets) on the pilot plant with the smoothed D against the unchanged T meshes (T/D under v1 and v1_tau);
  3. lamp height / lateral scenarios (R3-1b alone), the tray scenarios and the tray 60 / 120 mm self / neighbour / total shading attribution;
  4. static–dynamic consistency and GroIMP closure: the smoothing changes only the surfel (D) representation, not the engine-law mesh T, so the T closure of the v1_tau reference table is unchanged by it (stated, not re-run); the half-credit effect on D is reported by item 1;
  5. MTG cross-check (`mtg_build_v2_clicks_20261103_02` meshes, 11 plants): per-leaf light of the leaflets with leaf τ 0.07 (stems / rachis 0) and with black leaves, against the surfel per-leaf light of the same model (report only);
  6. report-only synthetic check at leaf τ = 0.07 (same generator and seed, arms truth, raw, `smooth_all_r2`), per the 16:54 answer item 3.
- **Result table:** `RESULT_TABLE_v1_vs_v1tau.csv` of `build/engine_static_v1tau_reference_20261104_01/` gets a third column v1_tau + smoothing (and v1 + smoothing as a fourth); the Korean report of this folder is extended with these results.
- Nothing in the declared rule, radius list, thresholds or generator is changed.
