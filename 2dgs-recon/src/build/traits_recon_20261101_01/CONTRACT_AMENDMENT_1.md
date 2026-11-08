# CONTRACT AMENDMENT 1 — hand-measured compound leaves (t1 dt01–dt06) and the arched rachis

Declared 2026-11-01 (user addendum 22:05), before any use of the data and before any trait number.

## Data
- Source: `measurements/hand_leaf_measurements_t1_dt01-06_2026-11-01.xlsx`.
- Copy: `inputs/` (copied unchanged), sha256 `0e1086ed700a46a9be71fdcb8fdd4c7d31d29ed146eff0c151a1094b560d62e4` (both copies).
- Role (user): MTG-building data, not validation. dt01–dt06 are separate from the day-0 validation set.

## Parse rule (fixed)
A main-stem row has an integer in column C (`복엽`, rank from the base) and `"n,θ"` text in column D.
- n = leaflet count; θ0 = base angle above horizontal (°).
- Only correction: `7,6o` → `7,60` (dt04 rank 5, Excel row 43).
- Result: 44 leaves, 6 plants (dt01 9, dt02 9, dt03 7, dt04 7, dt05 9, dt06 3).
- Every other non-empty cell (columns E–K, rows without a rank, and dt04 row 46 with rank 8 and no data) is listed in
  `hand_irregular_rows.json` and NOT interpreted until the user confirms its meaning.

## Arch representation (fixed before fitting)
θ(s) = θ0 − Δθ · g(s/L).
- Primary: arc, g = s/L. Sensitivity: quadratic, g = (s/L)².
- Arch traits and checks as in CONTRACT §2 B2.
- The primary is never switched after the fits are seen.

## Template rules
- Hand θ0 and leaflet counts are `hand_measured` (n plants 6, n leaves 44) and enter the template directly.
- Reconstruction values of the same quantity are pooled only if the per-leaf cross-check bias is ≤ 5°. Otherwise hand values take
  precedence and the bias is reported.
- Rank 1 stays its own rank.

## Model mapping
To be verified with one-leaf tests and written in `ADAPTERS.md` before any template value is used.
- Engine: insertion = 90 − θ0; `TOTAL_BEND_DEG` = Δθ (check that the bend is progressive along rachis segments).
- B-model: `LEAF_ANGLE_D` reference (vertical or horizontal) ← θ0; `Curvature_LEAF` unit and distribution ← Δθ;
  `CURVATURE_LEAFLET` = leaflet curl (C1).

## Cross-check
dt01–dt06 do not appear among the reconstructed plant ids (R3-1b, n_01–n_06, t2_02/04/05, e_01–e_07). Unless the user maps them,
the skeleton-vs-hand θ0 check is `not_possible`, and only the synthetic recovery validates the θ0 extractor.

## Not done
- Guessing the side-axis, truss, fruit or thickness cells.
- Tuning θ0 or Δθ to validation results.
- Deleting anything.
