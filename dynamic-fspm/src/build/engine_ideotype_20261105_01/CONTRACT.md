# CONTRACT — Engine ideotype comparison under spacing × light (longer internodes / wider leaflet spacing)

Written 2026-11-05 (system clock in RUNNING.lock), **before any number of this folder**. Session: Claude Code (Engine).
Binding: `handoff/CLI_PROMPT_ENGINE_IDEOTYPE_INTERNODE_LEAFLET_SPACING_FRUIT_POOL_2026-11-05.md` (11:25) and `handoff/CLI_ADDENDUM_IDEOTYPE_USE_ENGINE_FRUIT_SINK_NO_EXTRA_POOL_2026-11-05.md` (12:20; **wins where they differ**). The user's chat answer (same day) agrees: keep the engine's S1 fruit sink, add no pool.
Input hashes: `INPUT_HASHES.json`. Labels on every output: `ideotype_scenario`, `assumed_allometry`, `relative_light`, `conditional_on_unvalidated_scale`, `scenario_comparison_not_validation`, `base_leaf_area_below_target` (Model_v2 leaf area 13.0 cm² at 59.9 DAG vs C5 target 29.3–43.6 cm²), `assumed_luminaire_model`.

## 1. Base model
* **Model_v2 = frozen Engine v2** (`build/engine_v2_calib_20261104_01/work/Model_v2.rgg`, sha256 `fa53fe89…`): C1 leaflets, C2 θ0, C3 Δθ, C4 rules v2.2, C4V (user: lowest leaf ×0.7, main internodes rank ≥ 5 ×1.25), C5 SLA ×1.15 (bound). Blocks passed: C1–C5 all verified 16/16; C6 is running when this starts (sink diagnosis, not a model change).
* **Optics: black** (as v1/Model_v2). The prompt's "C7 leaf optics" is not part of the Engine v2 work order this session executed (C1–C6) → black, stated.
* Fruit: the engine's S1 `fruit_ke2023` sink as in Model_v2 (FRUIT_MAX_DM 1.16 g, flowers per truss {1, 3, 4}); no extra pool (addendum §1).
* Multi-plant: the v2 patch is generalised to n plants (`ideo_patch.py`; controllers 0…n−1 main, then 2 side controllers per plant). **Gate G0:** one plant at the lamp axis with the generalised patch and the D-ref point lamp must give **byte-identical** states.csv/flux.csv/ledger.csv to the frozen Model_v2 C5f1.15 run of the same seed (m00). Else stop and report.

## 2. Common random numbers
* The engine already keys every draw by (run seed, organ id, key) with `java.util.Random(seed + hash(id:key))`, first output skipped. Every trait draw has its own u per organ, recorded in `random_draws.csv`; ideotypes change only the value derived from the same u (value_ideo = k × value_base, i.e. F_ideo⁻¹(u) = k·F_base⁻¹(u) for a scale family). `crn_draws.csv` = the recorded (seed, plant, organ, key, u, base value, ideotype value) for the changed traits.
* **Gate G1:** for each type, the multiplier set to 1.0 gives byte-identical outputs to BASE (one canopy, 10 cm / 40 cm, replicate 1).
* Every other draw (leaf counts, truss nodes, angles, divergence, SLA/AMAX/ALPHA factors, max-mass factors) is unchanged by construction (keys untouched). The engine's ray tracer uses one fixed seed; ray paths still differ when geometry differs (reported as light noise, not a draw).

## 3. Types (declared)
| id | trait | change on top of Model_v2 |
|---|---|---|
| BASE | — | Model_v2 |
| IN125 / IN150 | every internode (main and side, all ranks) | length cap × 1.25 / × 1.5 (Model_v2 ranges: main rank < 5 and side 2–4 mm, main rank ≥ 5 2.5–5 mm → ×1.25: 2.5–5 / 3.125–6.25 mm; ×1.5: 3–6 / 3.75–7.5 mm) |
| LS125 / LS150 | every rachis segment (distance between leaflet pairs) | length (RACHIS_M / m = 35 mm / m) × 1.25 / × 1.5 |
Secondary (only after the primary report): PE150 petiole × 1.5; COMBO = IN125 + LS125.

## 4. Allometric carbon cost (option c)
* Stretched organ: L = k·L₀, diameter d = d₀·k^β, constant density → organ maximum mass × k^(1+2β); width × k^β. **β = 0.5** (`assumed_allometry`); sensitivity β = 0 and β = 1 on IN150 and LS150 at spacing 10 cm / height 40 cm.
* Model_v2 values: internode L₀ = per-internode Beta draw, d₀ = STEM_WIDTH_M 3 mm, max mass MAX_MASS[6] 0.03 g × max-mass factor; rachis segment L₀ = 35 mm / m, d₀ = RACHIS_WIDTH_M 1 mm, max mass MAX_MASS[8] / m = 0.012 g / m × factor. Density implied (mass / (L·d²)) unchanged. The extra mass enters the organ's potential growth (logistic target × max mass) in the existing sink rule. "Ideotype cost" = extra structural carbon per plant vs BASE (paired).

## 5. Fruit
Engine S1 fruits (addendum). Response test as Engine v2 C6: paired seed-median DM change ≥ 3 %. Fruit sensitivity: FRUIT_MAX_DM × 0.8 / × 1.2 on BASE and IN150 at 10 cm / 40 cm.

## 6. Environments and canopies
* One 4 × 2 tray: 4 rows along y, 2 columns along x; spacing p = 0.06 / 0.10 / 0.14 m; plant (r = 1…4, c = 1…2) at x = (c − 1.5)p, y = (r − 2.5)p (tray32 convention); plant ids p1…p8 row-major (p1 = r1 c1, p2 = r1 c2, …).
* Heights: floor = pot bottom z = 0; substrate surface z = 0.070 m = plant base (plants.csv z_m = 0.070).
* **Luminaire:** one flat panel 0.60 m (y) × 0.10 m (x), centred over the tray (x = y = 0), horizontal, emitting downward, Lambertian (tray32 m = 1 → GroIMP AreaLight exponent 0, verified adapter in `tools/ppfd_validation.py`), bottom face z = **0.40 / 0.50 m**. Output **1200 µmol s⁻¹** (the D-ref / Engine v2 lamp output; `assumed_luminaire_model`, ratios only). No housing, no other source. = 6 environments.
* Composition: (i) uniform — 8 plants of one type; (ii) mixed — the ideotype at **p3 (row 2, column 1, a middle-row slot)** among 7 BASE plants; the matching BASE run is the uniform BASE canopy (same seed, p3 BASE).
* Seeds: replicate r = 1, 2 → run seed 20261020 + (r − 1) (the engine draws per plant id within a run, so 8 plant seeds per canopy, 16 in all); the same seeds for every type and environment.
* Clearance: daily gap (luminaire bottom − highest organ vertex); flagged if < 0.02 m, never stopped.
* Rays: the engine's light is packet-limited (Engine v2 / divsens Amendment 2). **Rule declared now:** a timing pilot (BASE, 10 cm / 40 cm, replicate 1) at 300 000 rays reports wall time and hits per plant; the ray count for all runs is then fixed once, before any comparison number, as the largest of {300 000, 600 000, 1 200 000} whose estimated total fits ≤ 12 Kaggle CPU kernel-hours; stated in Amendment 1. Same ray seed in every run.
* Run length: 72 intervals (59.9 DAG = day 71), as Engine v2.
* Order: uniform at 10 cm / 40 cm for BASE + 4 primary types → short report → other 5 environments → mixed → secondary/sensitivity. If time is short: report and give options.

## 7. Measures and outputs
Per plant per day: absorbed light (Σ leaflet absorbed / 1200), accepted and potential gross, leaflet area, added surface of rachis/petiole/internode (Σ length × width of those organs minus BASE paired), dry mass by organ class (leaf = leaflets, stem = internodes, petiole/rachis, root, fruit/flower/truss), reserve, down-regulated, height (top vertex above substrate), clearance, ideotype cost. Paired ideotype − BASE by seed and slot: median and p10–p90 over plants × seeds; trait × spacing and × height; uniform vs mixed for p3; sink response.
Files as listed in the prompt §7 (`crn_draws.csv`, `types.csv`, `environments.csv`, `plants_daily.csv`, `plants_summary.csv`, `paired_effects.csv`, `mixed_vs_uniform.csv`, `fruit_pool_check.csv` → here the S1 fruit sink check), figures, `handoff/ENGINE_IDEOTYPE_2026-11-05_KO.md`, LOG line, hashes, flow row, lock.

## Do not
Change multipliers, β, environments, seeds or the ray rule after seeing results; refit Model_v2; let an ideotype change any draw other than its own trait value; add a carbon pool; call results measured or validated; touch `build/engine_static_*` or tray32 folders; drop arms on my own.

## Amendment 0 (before any run): pilot batch P0
* One kernel, 7 jobs: G0 = BASE@G0@r1 + the frozen Engine v2 case `C5f1.15_m00` rerun in the same kernel (states/flux/ledger hashes from completion.json must be equal); timing pilot BASE@s10h40@r1 (300 000 rays); G1 = IN150x1 and LS150x1 at s10h40 r1 (their prepared Model.rgg is already byte-identical to BASE: the per-plant multiplier arrays are all 1.0; the runs check output determinism); technical runs IN150 and LS150 at s10h40 r1 (patch + validator only; their numbers are used later only if the ray rule keeps 300 000 rays, else they are re-run at the chosen count).
* G0 compares outputs, not the input hash: the seed record's text differs, which changes only the completion binding hash.

## Amendment 1 (2026-11-05 ~21:10, before any ideotype run): base changes to the frozen **Engine v2.1**
* The pilot batch P0 (Amendment 0) was never pushed (Kaggle slots, then a session restart); nothing ran. It is superseded.
* After Amendment 0 the Engine v2 work added user decisions (4 flowers, template v2.4 with 2–4 side axes and divergence 90–100°, leaflet arrangement C4L, leaflet curvature C4B, internode × 5.85 for the user's 130 mm height target) and froze **v2.1**: `build/engine_v2_calib_20261104_01/work/Model_v2_1.rgg` sha256 `0a408633872ec31f5d8bfff6e3ecfee671b12140c5a47ff56bae78fc459b41ac` (+ `v2_1_config.json`). **BASE = v2.1.** Multipliers (IN, LS, PE) act on the v2.1 values (internodes already × 5.85, main rank ≥ 5 × 1.25).
* The n-plant patch (`ideo_patch.py`) is rebuilt to carry the full v2.1 stack (C4T controllers: main + up to 4 side axes per plant; C4L, C4B organ rules unchanged). Gate G0 now compares with the frozen v2.1 confirm-run case `V21Bs5.85_m00` (point lamp, one plant): states / flux / ledger hashes equal.
* Fruit sink: v2.1's S1 fruit (FRUIT_MAX_DM 1.16 g, 4 flowers per truss). Everything else in this contract stays.

## Amendment 2 (2026-11-06 00:0x): pilot P0 result; gates; ray count fixed by the declared rule
* P0 (6/6 verified): **G0 pass** — one plant, point lamp, generalised patch: states.csv, flux.csv, ledger.csv hashes equal to the frozen v2.1 run `V21Bs5.85_m00` (attempt 1 of the G0 case hit the GroIMP start-up race, retry verified). **G1 pass** — IN150 × 1.0 at s10h40 r1: every output file hash equal to BASE.
* Timing: an 8-plant tray at 300 000 rays takes 2 700–2 940 s per run (4 runs in parallel per kernel). Full plan (uniform 60 + mixed 48 + β 8 + fruit 8 = 124 runs) ≈ 124 × 48 min / 4 ≈ **25 kernel-hours at 300 000 rays**, above the declared 12 kernel-hour budget; no candidate fits → per the declared rule the smallest candidate, **300 000 rays**, is used for every run (the pilot's s10h40 r1 runs therefore count), and the budget overrun is reported with options before the stages that do not fit.
* Order as declared: uniform canopies at s10h40 for the 5 primary types (2 replicates; BASE / IN150 / LS150 r1 from P0) → short report → other environments → mixed.

## Result U1 + P0: uniform canopies at s10h40 (2026-11-06 01:0x; 10/10 type × replicate runs verified, 8 plants each; paired ideotype − BASE per slot and replicate, n 16, day 71 = 59.9 DAG)
| type | absorbed (cum) | leaf area | dry mass | fruit DM | height | absorbed per leaf area | stem + rachis share of DM (BASE 10.2 %) |
|---|---|---|---|---|---|---|---|
| IN125 | −41 % | −45 % | −44 % | −48 % | +3 % | +9 % | 14.8 % |
| IN150 | −67 % | −71 % | −68 % | −73 % | +2 % | +16 % | 20.5 % |
| LS125 | −10 % | −12 % | −11 % | −13 % | −3 % | +3 % | 11.3 % |
| LS150 | −21 % | −24 % | −23 % | −26 % | −7 % | +5 % | 12.8 % |
(p10–p90 in `paired_effects_s10h40.csv`; per plant `plants_summary_s10h40.csv`.) Clearance min 59–95 mm (no flag). BASE per plant: leaf area 110 / 74 cm², DM 3.1 / 2.3 g (r1 / r2).
* Reading: stretching does raise light per unit leaf area (geometry works, +3 to +16 %), but the allometric cost (k^(1+2β), β 0.5) is paid early (9–14 DAG) when v2.1 is carbon-limited, so the first leaves are smaller and the loss compounds. Height barely changes because internode length is funded (fraction^(1/3) of the cap). Conditional on β, on v2.1's source limitation and on the assumed luminaire; not validated.

## Amendment 3 (2026-11-06, user decision "축소안" after the budget report; scope change only — multipliers, β, environments, seeds and rays unchanged)
* Remaining plan reduced to fit the declared 12 kernel-hours (≈ 58 runs ≈ 11.6 kernel-h at ~48 min per 8-plant run, 4 per kernel):
  1. uniform canopies in the other 5 environments (s6h40, s14h40, s6h50, s10h50, s14h50) for **BASE, IN150, LS150** only, r1 + r2 (30 runs);
  2. mixed canopies (ideotype at p3 among 7 BASE) for **IN150 and LS150** at **s6h40, s10h40, s14h40**, r1 + r2 (12 runs; matching BASE = uniform BASE);
  3. β sensitivity (b0, b1) on IN150 and LS150 at s10h40, r1 + r2 (8 runs) and fruit sensitivity (FRUIT_MAX_DM × 0.8 / × 1.2) on BASE and IN150 at s10h40, r1 + r2 (8 runs), as declared.
* Dropped from the plan: IN125 / LS125 outside s10h40, mixed at the 50 cm height and for the 125 types, secondary types PE150 / COMBO. Reported as not run.
