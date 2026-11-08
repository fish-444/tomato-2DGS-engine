# ADAPTERS — Engine v2 (build/engine_v2_calib_20261104_01)

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
