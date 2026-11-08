# CONTRACT AMENDMENT 2 — user answers on the irregular hand cells, ages (2026-11-01 22:16–22:21)

The source is the user's answers, relayed by the orchestrator in "CLI ANSWER — Traits session". Written before re-running the
template build. The main-stem rows (Amendment 1) are unchanged.

## Reading rules (user-confirmed)
1. Side-axis cell `k,a`: side axis k left the main compound leaf at angle a (°, from horizontal).
   - A bare integer in the 측축 column is k with the angle not recorded → `angle_not_recorded`.
2. Side-axis compound-leaf cell `n,θ`: n leaflets, base angle θ from horizontal.
3. dt02 row 19:
   - `1,5,20` = side axis 1, 5 leaflets, 20°.
   - `2,5,20` = side axis 2, 5 leaflets, 20°.
4. `4,4` (dt03 H, dt05 I, dt06 I) = 2 trusses × 4 flowers each.
5. A side axis attaches at the node of the main compound leaf in the same row.
   - A row without a main rank → `attach_node_unknown`.
6. Main-stem thickness, shifted one column; re-aligned to the thickness field:
   - dt03 H 8.1 / 5.4 / 4.3
   - dt04 E/J 4.2, 8.7, 8.3, 5.3, 4.2
   - dt06 J/K 8.1, 6, 3.7
7. dt04 row 46 (main leaf 8): left empty, not imputed.
8. dt06 has 3 main compound leaves (confirmed).

## Structural rules I apply (stated here so the user can overrule them)
- Thickness cells under the 굵기 header in dt01 and dt02 (column I) are placed as `header_aligned`.
- Side-axis compound-leaf cells in the rows below a side-axis cell belong to the most recently listed side axis of that plant
  (`assigned_by_row_order`). In dt05 and dt06 the integers in column F are written leaf ordinals.
- Every other integer in the 화방/과실 columns (dt01 G 3; dt02 H 3 ×2; dt04 G 1, H 4 ×3; dt05 H 3; dt06 I 4) is not covered by
  rules 1–8 → `pending_user`, not used.
- A row with two thickness values (dt04 row 41; dt06 row 60) keeps both → `two_values_same_row`.

## Ages
`age_dag = 60` (`user_stated`) for all reconstructed plants (R3-1b, n_*, t2_*, e_*). This replaces `age_unknown`.

## Still unanswered
- Validation membership.
  - Until answered, t2_02, t2_04 and t2_05 are excluded from template building.
  - They are not used now anyway: the template is hand-only.
- dt01–dt06 → reconstructed-ID mapping.
  - Not guessed; no per-leaf comparison until it is given.

## Labels
- `hand_measured`
- `user_confirmed_2026-11-01`
- MTG-building data, not validation.

## Amendment 2b — confirmations (2026-11-01 23:14, 23:16; user)
- 23:14 ("ㅇㅇ 맞아"): the five structural readings above are confirmed. Status changes from draft to `user_confirmed_2026-11-01`.
- 23:16 ("꽃수"): a single number in the 화방 (truss) column is the number of flowers on one truss.
  - dt01 G11 = 3 flowers; dt04 G41 = 1 flower.
  - The truss node stays unconfirmed.
- Still held (meaning not stated; unused): dt02 H19, H26 = 3; dt04 H41, H44, H45 = 4; dt05 H57 = 3; dt06 I61 = 4.
- Outputs go to new files `MTG_TEMPLATE_v0c_*` (the v0b files are kept unchanged); `hand_reinterpret_v1_2225.py` = script before 2b.

## Amendment 2c — fruit column (user, this session: "과실 열 숫자는 과실 개수")
- A number in the 과실 (fruit) column = number of fruits.
  - Applied to dt02 H19 = 3, H26 = 3; dt04 H41, H44, H45 = 4; dt05 H57 = 3 (21 fruits).
  - The node position is not confirmed (`position_not_confirmed`).
- dt06 I61 = 4 is not in the fruit column (column I = 굵기 header; dt06 is shifted). It stays held and unused.
- Outputs go to new files `MTG_TEMPLATE_v0d_*` and `hand_fruits.csv` (v0c kept; `hand_reinterpret_v2_2334.py` = script before 2c).

## Amendment 2d — dt06 I61 (user, this session: "flower yes no" → chosen option "화방 1개, 꽃 4개")
- dt06 I61 = 4 → 1 truss with 4 flowers; node not confirmed.
- No held cells remain.
- Outputs go to `MTG_TEMPLATE_v0e_*` (v0d kept; `hand_reinterpret_v3_2335.py` = script before 2d).
