# ADAPTERS — Engine v2 and v2.1 (build/engine_v2_calib_20261104_01)

v2.1 (frozen 2026-11-05, `work/Model_v2_1.rgg` sha256 0a408633…) = v2 + E12–E18; v2 alone = E1–E12 (copy of the v2-only file: `ADAPTERS_v2_only.md`).

Labels: `engine_v2`, `calibrated_on_mtg_building_data`, `conditional_on_unvalidated_scale`, `relative_light`, `biological_validation=not_established`.
The frozen v1 engine (DEV tools, D-ref, every v1 output) is unchanged. v2 = the prepared v1 `Model.rgg` + the patch `v2_patch.py` (cumulative levels), validated by `kaggle_kernel/v2_validate.py`.

## Engine rules added or replaced (Model_v2.rgg)
| id | block | what | source label | why an adapter |
|---|---|---|---|---|
| E1 | C1 | leaflets per compound drawn per compound (`LEAFLET_COUNT`), rank shares from rules v2.2; pairs m = n div 2 | `hand_measured` | v1 had rank-indexed PAIRS choices {1,1,2}/{1,1,3} per plant |
| E2 | C1 | n = 6 → 3 pairs with the **left leaflet of the distal pair omitted**, sink shares renormalised | `user_assumed` | the engine has no intercalary leaflet; this is the smallest stand-in with 6 blades |
| E3 | C2 | insertion = 90° − θ0, θ0 normal by rank class (`THETA0_DEG`), clipped [−5, 80]° | `hand_measured` | replaces U(60, 90) |
| E4 | C3 | rachis total bend normal 32.8 ± 10.6° clipped [0, 90]° (existing key `TOTAL_BEND_DEG`) | `skeleton_traits_ungraded` | replaces U(0, 40) |
| E5 | C4 | per-internode length cap from the scaled Beta on [2, 4] mm (`INTERNODE_MM`), exact ppf in the engine (betacf + bisection) | `user_set_range` | v1 had one cap per plant U(8, 16) mm |
| E6 | C4 | **side axes**: 2 new controllers S1, S2 per plant; attach ranks (`SIDE_ATTACH_RANK`, without replacement); created at first eligibility (main first truss present and main node r present); clock from 0; first leaf after BUD_BREAK_DELAY_DD 150 °Cd + one phyllochron; side internodes 40° above horizontal at the azimuth of the axil leaf; first side leaf azimuth = that azimuth | `hand_measured` (count, ranks, angle sparse) + `user_assumed` (BUD_BREAK delay/eligibility registry rows) + `rule_like_main_axis_user_2026-11-03` | the engine had no side-axis rule (planned axes need a pre-planned emergence day) |
| E7 | C4 | side leaf ranks written as 100·j + k | engine id convention | ranks must be unique per plant for the rank-indexed checkers and organ ids |
| E8 | C4 | truss targets per controller (N, T − N, 0): main N 5–9 (1:2:3:2:1), T 10–13; side 2, S 5–7; the existing `two_successive_trusses` leafless truss and determinate stop kept | `user_stated_2026-11-03` / `user_stated_2026-11-04` | replaces FIRST_LEAVES 9 + CADENCE {2, 1, 0}; the v1 intermediate trusses are not produced (reading declared in CONTRACT §2 C4) |
| E9 | C4 | flowers per truss categorical {1, 3, 4} (`FLOWERS_PER_TRUSS`) | `hand_measured` | replaces FLOWERS 4 |
| E10 | C4 | `axisPrefix` gives `S<j>/` ids without the axis-contract switch | — | needed for unique ids of the new controllers |
| E11 | C5 | base SLA = 0.025 · f (fitted f) | `fitted_to_reconstruction_area` | the one fitted quantity |

| E12 | C4V | lowest main leaf × 0.7; main internodes rank ≥ 5 × 1.25 | `user_stated` | user decisions 2026-11-05 |
| E13 | C4W (v2.1) | 4 flowers per truss, fixed | `user_stated` (12:15 addendum) | replaces E9 in v2.1 |
| E14 | C4T (v2.1) | template v2.4 counts as discrete normal (σ = range/3): main N 5–9, T 10–13 redrawn until T ≥ N + 2 (`MAIN_TOTAL_LEAVES_REDRAWS` recorded); `SIDE_AXES_N` 2–4 (controllers 3 → 5); side S 5–7; side axis j germinates (j − 1) phyllochrons after S1; an unused side controller has attach rank 0 and never starts | `template_v2_4` | replaces E6/E8 counts |
| E15 | C4T-d (v2.1) | divergence U(90, 100)° per leaf | `template_v2_4` | replaces 119.3–154.7° |
| E16 | C4L (v2.1) | per-leaflet position s/L, spread φ, petiolule length, tilt, roll sign from `MTG_LEAFLET_ARRANGEMENT_v1.csv`; placed in a second pass after `updateGeometry` | `reconstruction_mtg` | the v1 engine placed leaflets in fixed pairs |
| E17 | C4B (v2.1) | leaflet bend κ_l, κ_w, twist c from `MTG_LEAF_CURVATURE_v1.csv`, κ scaled by L_ref / L; area-conserving height-field bend | `reconstruction_mtg` | v1 leaflets were flat |
| E18 | height (v2.1) | every internode cap × 5.85 (user target seed-mean height 130 mm; confirm 129.9 mm) | `user_stated_target` | extrapolated beyond the s = 3 / 5 arms |

## Validator replay rules replaced (v2_validate.py; everything else is the frozen code)
| id | what the frozen validator assumed | v2 replacement |
|---|---|---|
| V1 | pair count by rank from the PAIRS choice | per rank (unique per plant) from the recorded `LEAFLET_COUNT` draws (draw reader override) |
| V2 | compound inventory 2m + 1 with both sides | per compound from the draw; n = 6 has (m, left) missing |
| V3 | internode length = plant cap × fraction^(1/3) | recorded per-internode draw × fraction^(1/3) |
| V4 | SLA base = registry 0.025 | v2 constant (0.025 · f) |
| V4b | S1 birth-frozen SLA check reads the request SLA | the validator's in-memory copy of the request carries the v2 SLA (found when all 32 C5 SLA-arm runs were refused, 2026-11-05 12:3x; the engine runs were fine) |
| V5 | flowers per truss = FLOWERS or the flowers hook | recorded `FLOWERS_PER_TRUSS` draws |
| V6 | a determinate stop forbids any later organ of the plant | only of the same controller (main / S1 / S2) |
| V7 | schedule replay with FIRST_LEAVES / CADENCE, one controller | independent v2 schedule per controller (eligibility day, clock, targets, stop) compared day by day with the states |
| V8 | (C4L) leaflet frame in `check_leaf` from the fixed pair layout | frame rebuilt from the recorded C4L draws |
| V9 | (C4B) expected outline = flat placed outline | expected outline = the same bend applied to the recorded C4B draws |
| V10 | `placed outline angle below hard floor` 3° | 1° for C4B-bent leaflets only (user decision 2026-11-05, CONTRACT Amendment 13); flat library minimum 3.62° |
