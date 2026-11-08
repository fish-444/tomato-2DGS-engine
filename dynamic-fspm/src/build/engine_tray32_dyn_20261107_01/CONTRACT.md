# CONTRACT — Dynamic engine on the static tray32 scenarios (frozen v2.2 A)

Written 2026-11-07, before any number of this folder. Session: Claude Code (Engine v2). User request (chat): "정적계산 시나리오대로 동적계산도 돌려줘 (광원높이 40, 60, 각도는 평면 90, 120으로)"; user choices: spacing 10 cm, over trays only; engine = frozen v2.2 A only.
Labels: `engine_v2_2_slaA`, `assumed_luminaire_model`, `groimp_arealight_broader_than_cos`, `relative_light`, `conditional_on_unvalidated_scale`, `scenario_comparison_not_validation`.

## Scenes (6)
* Layout = static tray32 (`build/static_tray32_v2_20261104_01/CONTRACT.md`): 32 plants in four 4 × 2 trays in a row, gap 0.10 m between outlines, spacing p = 0.10 m; tray i centre x_i = (i − 2.5)(2p + 0.10), plant (r, c) at x_i + (c − 1.5)p, y = (r − 2.5)p.
* Luminaires over trays: one bar (0.60 m along y) above each tray centre line (4 luminaires), shapes flat (0.10 m panel) / V120 / V90 (two 0.05 m wings), built as in the GUI canopy tool (GroIMP AreaLight half panels, exponent 0; known deviation: broader than cos¹ beyond ~18°, GUI CONTRACT "Lamp-field investigation").
* Height: static 0.40 / 0.60 m from the pot bottom to the lowest point of the luminaire, with the substrate surface at 0.060 m → **0.34 / 0.54 m above the substrate** here (the dynamic plants stand on the substrate surface; same relative geometry as static).
* Output **1200 µmol s⁻¹ per luminaire** (the engine's calibration lamp output; one per tray of 8 = the ideotype / GUI canopy convention); total 4800.
* Engine = frozen **v2.2 A** (`Model_v2_2_slaA.rgg` rules via the n-plant patch, G0′ and EQ checks byte-identical), GUI multipliers 1.0 (no stem / height change), seed 20261020 (one replicate, keyed per plant), 72 intervals (59.9 DAG). Rays per light pass **1 200 000** (= 37 500 per plant, as the 8-plant runs at 300 000).
* Plant-alone passes off (K = 0) — neighbour shading is not part of this run (cost); per-plant absorbed light, leaf area, dry mass, height, trusses, fruits per day from `daily.csv`.
* Output-only change for 32 plants: meshes / consumed exported only at state 0 and the last state (the 32-plant test of 11-06 hung at the end of the run — attributed to hashing a very large meshes.csv; growth code untouched).

## Measures
Per scene: canopy light (Σ absorbed ÷ 4800, %), per-plant absorbed, leaf area, dry mass, fruit count, height at 59.9 DAG (median, p10–p90 over 32 plants), and the same by tray position (outer trays 1, 4 vs inner 2, 3). Paired scene effects vs the reference scene (flat, 0.40 m). Compared with the static tray32 v3 scene efficiencies for the same scenes (S-numbers p 0.10 m, over trays, 40 / 60 cm), as a ranking / direction comparison only (different optics: static τ 0.07 smoothed normals vs dynamic black leaves).

## Do not
Change the engine, the lamp model or the scene list after seeing numbers; call results validated.

## Amendment 1 (2026-11-07, user "추가로 캐노피 흡수광 / 차폐율 평균도 제공") — before any number of this folder
* A second batch DT2 = the same 6 scenes with plant-alone light passes every **K = 10** days (engine days 11, 21, …, 71; 32 passes each, same lamp / rays / seed). Growth is identical with or without these passes (GUI check GT1: S10 vs S10K0, 639 / 639 daily rows equal), so DT2 reproduces DT1's plants; DT1 vs DT2 equality is checked and reported.
* Measures added: mean neighbour shading = mean over the 32 plants of 1 − absorbed in the canopy ÷ absorbed alone (update days only, same pass); index = canopy light (% of emitted 4800) ÷ mean neighbour shading (%), shown with both parts ("—" if shading < 0.5 %). Static tray32 `total_shading_mean` (self + neighbour, isolated copies) is listed beside it with that difference in definition stated.

## Results (2026-11-07 ~21:00; DT1 6/6 and DT2 6/6 runs complete to engine day 71, ~20–23 min each; the validator refuses the reduced mesh output = state coverage, expected; DT1 = DT2 growth 2343 / 2343 daily rows per scene)
* Canopy light at 59.9 DAG (% of 4800): flat 40 44.0, V120 40 38.0, V90 40 33.3, flat 60 26.8, V120 60 23.6, V90 60 20.9 — same ranking as static tray32 v3 table A (6.26 / 5.22 / 4.35 / 3.87 / 3.24 / 2.66 %); relative to flat 40: dynamic −14 / −24 / −39 / −46 / −53 %, static −17 / −30 / −38 / −48 / −58 %.
* Dry mass per plant (median): 9.23 / 9.09 / 8.90 / 8.64 / 8.42 / 8.01 g (−1 … −13 %); leaf area 297 … 282 cm²; height 233 mm and 48 fruits in every scene; outer ÷ inner trays dry mass 0.96 … 0.87.
* Mean neighbour shading at the last update (DAG 50.9; the day-71 pass falls after the run): 25.2 / 25.0 / 25.5 / 22.8 / 23.0 / 23.1 %; matched-day index (light ÷ shading at DAG 50.9): 1.58 / 1.38 / 1.19 / 1.07 / 0.93 / 0.82 (`shading_index_matched.csv`). Static total shading (self + neighbour) 49.0–50.6 % — different definition.
* Report `handoff/ENGINE_TRAY32_DYN_2026-11-07_KO.md`.

## Amendment 2 (2026-11-07, user "Yes, run over gaps") — before any over-gaps number
* Six more scenes: the same heights (40 / 60 cm from the pot bottom = 34 / 54 cm above the substrate) × shapes (flat / V120 / V90), spacing 10 cm, **over gaps** as in the static tray32 v2 contract §3: one full luminaire (1200 µmol s⁻¹) above each of the 3 gap centres (midpoint of neighbouring tray centres; V vertex line at the gap centre, left wing lights the left tray, right wing the right tray) + one single face at each outer end of the row (600 µmol s⁻¹): V shapes = one wing facing inward (same tilt), its lowest edge 0.05 m outside the outer tray outline (x₁ − p − 0.05 and x₄ + p + 0.05); flat = a 0.05 m horizontal half panel centred 0.05 m outside the outline. Total 4800 µmol s⁻¹ (= over trays).
* One batch with plant-alone passes every 10 days (growth + shading; growth is unaffected by these passes, shown in batch DT2). Same engine (frozen v2.2 A), seed, rays (1.2 M), mesh export at the ends only. Compared with the over-trays scenes and with static (gaps vs trays, per shape and height).

## Amendment 2 result (over gaps, batch DT3, 6/6 complete to day 71; `gaps_vs_trays.csv`, `tray32_dyn_gaps_vs_trays.png`)
* Canopy light at 59.9 DAG, gaps vs trays: flat 40 −28.7 %, V120 40 −16.3 %, V90 40 −11.4 %, flat 60 −2.0 %, V120 60 +4.5 %, V90 60 +5.9 % (static: −7.3 / +1.5 / +5.5 / −3.4 / +3.7 / +6.8 %). Agreement at 60 cm; at 40 cm the dynamic loss over gaps is much larger (reading, not tested: dynamic plants 233 mm tall → lamp ~11 cm above the tops at 40 cm, vs ~21 cm for the static pilot plants (tallest top 0.186 m from the floor), so a lamp above an empty gap sends more light into the gap).
* Dry mass per plant, gaps vs trays: −2.1 / −0.7 / +0.3 / −0.2 / +0.9 / +3.0 %; neighbour shading +0.4 … +1.2 points; matched index 1.09 / 1.10 / 1.01 / 1.01 / 0.96 / 0.83 (trays 1.58 / 1.38 / 1.19 / 1.07 / 0.93 / 0.82); outer ÷ inner dry mass about unchanged (0.87–0.97).

## Amendment 3 (2026-11-08, user: "gap 10 cm, pot top to lamp 30 cm, lamps over gaps"; asked: 30 cm measured from the substrate surface / pot top) — before any number
* Six scenes: lamp **30 cm above the substrate** (= 36 cm from the pot bottom in the static convention; names `G_<shape>_s30` / `K_<shape>_s30`), shapes flat / V120 / V90, spacing 10 cm, tray gap 10 cm (unchanged layout).
* Over gaps (Amendment 2 rule) is the request; the three over-trays scenes at the same height are added as the comparison (needed to say whether V helps).
* Same engine (frozen v2.2 A), seed 20261020, 1.2 M rays, plant-alone passes every 10 days, mesh export at the ends only. Batch DT4.
* Known in advance: the plants reach 233 mm above the substrate, so near the end the leaf tips are ~7 cm below the lamp; the engine does not stop growth at the lamp and light from a very close panel is sensitive to the GroIMP area-light angular excess. Static has no 36 cm scene; nearest are 30 / 40 cm from the pot bottom.

## Amendment 3 result (batch DT4, 6/6 to engine day 71; validator refusal = reduced mesh output, as before; `s30_scenes.csv`, `fig_s30_gaps_trays.png`)
* Canopy light at 59.9 DAG, 30 cm above the substrate: trays flat 48.8 / V120 42.7 / V90 37.8 %; gaps flat 35.3 / V120 33.0 / V90 31.0 %. V vs flat at the same position: trays −12.4 / −22.5 %, gaps −6.5 / −12.3 %. Gaps vs trays: flat −28 %, V120 −23 %, V90 −18 %.
* Dry mass per plant (median) 9.31 / 9.20 / 9.03 g (trays), 9.17 / 9.08 / 8.98 g (gaps); neighbour shading at DAG 51 25.9–27.1 %; index 1.70 / 1.50 / 1.31 (trays), 1.19 / 1.11 / 1.03 (gaps).
* Reading: V is not better than flat at this height over gaps or trays; lamps over gaps lose light at low height for every shape. 1 of 32 plants reaches the lamp height (300 mm) from DAG 44; the engine does not stop it (one plant, small effect expected, not tested). One seed; not validation.

## Amendment 4 (2026-11-08, user: "add V150 to see whether there is a peak between V120 and flat") — before any number
* V150 = the V120 build with the wing rotation phi 15° instead of 30° (opening 150°), applied by the test builder (`make_gui_tests.py` option VPHI=15; the GUI tool files are unchanged; the settings line in daily.csv still reads V120, the case name and gui_test.json say V150 / vphi_deg 15).
* Six scenes, same as before otherwise: over trays and over gaps × 30 / 34 / 54 cm above the substrate (34 / 54 = 40 / 60 from the pot bottom), spacing 10 cm, tray gap 10 cm, frozen v2.2 A, seed 20261020, 1.2 M rays, alone passes every 10 days, mesh export at the ends. Batch DT5.
* Read-out: canopy light at 59.9 DAG against opening angle (90 / 120 / 150 / 180 = flat) per height and position; a peak means V150 above both V120 and flat. One seed: differences under ~1 % are not separated from noise. Static has no V150.

## Amendment 4 result (batch DT5, 6/6 to engine day 71; `v150_angle.csv`, `fig_v150_angle.png`)
* Canopy light at 59.9 DAG, V90 / V120 / V150 / flat: 30 cm trays 37.8 / 42.7 / 46.6 / 48.8; 30 cm gaps 31.0 / 33.0 / 35.32 / 35.31; 34 cm trays 33.3 / 38.0 / 41.7 / 44.0; 34 cm gaps 29.5 / 31.8 / 32.8 / 31.4; 54 cm trays 20.9 / 23.6 / 25.8 / 26.8; 54 cm gaps 22.1 / 24.7 / 26.1 / 26.3 %.
* Peak (V150 > 1 % above both flat and V120) only at 34 cm over gaps (+4.6 % vs flat). Over trays light rises monotonically to flat at every height (V150 −4 … −5 %). Over gaps at 30 and 54 cm V150 = flat within 1 %. Dry mass follows light, differences ≤ 1 % between V150 and flat. One seed; not validation.
