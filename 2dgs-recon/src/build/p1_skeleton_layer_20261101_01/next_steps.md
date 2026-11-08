# next_steps (proposal only, not run; needs the user's decision)

## Proposal P-A: rachis seeds from skeleton proximity without axis labels (new contract)
Why: the frozen adapter needs axis_kind = rachis-like; the nine plants have only the S1 PCA proxy (axis share 0.014-0.041), so n_04 / t2_02 got 17 / 3 rachis seeds.
Rule sketch (to be declared in a new contract before any run): every surfel (any class) within r mm of exactly one leaf's 1 mm polyline and >= r + m mm from every other polyline -> that leaf's stage-1 seed;
blade groups as in A2 (8 / 13 mm); then frozen R8_pc (weights B) and frozen rachis absorption, unchanged.
Declared evaluation (fixed before running): on R3-1b, report-half accuracy against human leaves (A4 metric, leaves 2/4/5/7) must be >= 0.833 AND >= the distance route (0.687, CI 0.675-0.698); one run, no tuning of r / m after it.
Risk: R3-1b shows that skeleton-only seeds lose leaf 6 in both existing routes (0.02); P-A may not fix that, because the 6/7 rachises touch.
