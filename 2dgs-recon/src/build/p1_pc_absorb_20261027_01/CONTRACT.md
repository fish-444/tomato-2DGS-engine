# Contract — R8_pc: stage-1 separation, then point-cloud-only absorption (geometry + surfel colour), R3-1b, 2026-10-27

Request: pasted CLI prompt (user, 2026-10-27). Declared before any run of the method. Changes after the first result only as a dated amendment with the reason.
Labels: `R3-1b_unaccepted` · `conditional_on_unvalidated_scale` · `gate_failed_viewing_only` (until the gate below passes; even then not `user_accepted_visual` without the user's "채택").
**No SAM, no photo masks, no photo votes (V-P)** anywhere in this run (not as evidence, answer key or score; SAM IoU not reported). R7 and every earlier output unchanged; result = new label set `R8_pc`.
Note: the stage-1 labels themselves were built by the earlier chain from the V-P base (user decision: start there). The self-test measures consistency with stage 1, not truth.

## Inputs
- Stage-1 candidates: R6 `r6_chain/labels.npz` step_1 (7,023 unassigned blade), **R7 `vp_leaf4/labels.npz` step_1_diffusion (5,275 unassigned blade) — used** (the flow table's latest chain row is R7; its step 1 is "1 확산"), R7b `vp_leaf4/leaf4_only/labels.npz` step_1.
- Bundle `kaggle_data/r8pc_bundle.npz` sha256 `ec7cfbce98c8c642bd0b371585a9933b30d7e1529f08775a65ac44fa2016ed61`, built by `tomato3d/organs/pc_absorb_prep.py` from
  R7 labels sha256 `1c7df8d7…e24b4b`, surfels PLY `7131b69a…f7714f`, sheets `ef27b4e4…edbee3` (full hashes in `inputs.json`). Contents: xyz, normals (sheet normals on blade), 2DGS scales and tangents, PLY rgb (0–255), opacity,
  axis_kind, sheet index / n_sheets, stage-1 and R7 labels, densified human rachis traces, t_sheet 1.529 mm, offset limit 1.529 mm, reach U = 22.72 mm. No photo data.
- Compute: free RAM 1.81 GB < 4 GB → Kaggle CPU kernel (`kaggle_kernel/`); local runs only build the bundle and a code smoke test on a small crop (no reported number comes from it).

## Nodes and graph
- Nodes = blade surfels (axis_kind 0) ∪ **blade-like axis_kind 2**: radius √(s0·s1) ≥ p25 of stage-1 labelled blade, elongation max/min ≤ p75 of stage-1 labelled blade, normal within **20°** of the local labelled normal (8 nearest stage-1 labelled surfels, sign-free),
  > 3 mm from every traced rachis. Other axis_kind 2 stay rachis; axis_kind 1 stays unlabelled.
- Edges: pairs ≤ 3 mm, normal angle ≤ 30°, offset along the mean normal ≤ 1.529 mm, same sheet index when both lie in multi-sheet columns (blade-like axis2 count as single-sheet). Other layers are reached only through such paths (never by 3D distance across sheets).

## Features (all from the reconstruction)
Edge (pairwise): G2 normal angle / 30°; G4 offset / t_sheet; G5 |curvature difference| (surface variation of 16 nearest nodes within 5 mm); C3 |ΔL*-residual| (L* minus the opacity-weighted mean L* of nodes within 10 mm).
Unary (node × candidate leaf L, recomputed every round from the current seeds; L's local model = its ≤ 24 nearest seeds within 10 mm, opacity-weighted):
G1 geodesic distance on the graph to L's seeds / U; G3 residual to a local quadric (≥ 10 seeds; plane if 3–9) fitted to those seeds in their PCA frame / t_sheet;
G6 rachis: 0.5 × [distance to L's trace (pseudo-rachis = PCA segment from the stem-nearest seed if L has no trace) > U → 1, else d/U] + 0.5 × [node on the minority side of L's seeds near the same trace position (±15 mm, ≥ 20 seeds each side = both sides allowed)];
C1 Mahalanobis distance of (a*, b*) to L's local colour model (covariance + 2² I); C2 |L* − quadratic L* field of L's seeds in the local frame|; C4 |log((s_i + 1) / (s_L + 1))|, s = std of the L*-residual over the node's graph neighbours.
CIELAB from sRGB (D65). Normalisation (fixed rule): each edge feature ÷ its median over stage-1 same-leaf edges; G3, C1, C2, C4 ÷ their median over 2,000 random stage-1 labelled surfels (seed 0) scored against their own leaf with themselves excluded; G1, G6 already scaled.
Missing unary feature → 3.0. Candidate leaves of a node: leaves whose seeds are within geodesic U.
Colour noise level reported first (local RGB std of same-leaf neighbours within 3 mm, 0–255).
**Colour usefulness rule:** on calibration-half boundary surfels (definition below), AUC of each colour feature (own leaf vs the nearest other leaf) must be ≥ 0.60; a colour feature below 0.60 gets weight 0 in sets B and C (tests and final), stated in the report.

## Method
Random walker with priors (Grady 2005): for each leaf, solve (L_UU + γ I) x = −L_UM m + γ λ, γ = 1.0; edge weight w = exp(−E_edge), prior λ ∝ exp(−E_unary) over candidate leaves (0 elsewhere).
E_edge = g · (0.20 G2 + 0.15 G4 + 0.05 G5) + c · 0.20 C3; E_unary = g · (0.30 G1 + 0.20 G3 + 0.10 G6) + c · (0.35 C1 + 0.35 C2 + 0.10 C4) (each feature in exactly one place; inside weights as the prompt).
Accept p1 ≥ 0.6 and p1 − p2 ≥ 0.25; accepted nodes become seeds; at most 5 rounds (stop early when a round accepts nothing); count per round recorded.
Unpainted reasons: `not_connected` (no seed in its graph component, single-sheet column), `other_layer` (same, multi-sheet column), `low_colour_confidence` (contested in B/C and opacity < 0.5), `contested` (else).

## Weight sets (fixed): A geometry only g 1.0 / c 0.0; B balanced 0.7 / 0.3; C colour-heavy 0.5 / 0.5.

## Self-test (no SAM)
Boundary band = stage-1 labelled nodes within 5 mm of a node with a different stage-1 label. Hide 20 % at random (seeds 0–4), run the full absorption with them unlabelled, score the hidden ones against their stage-1 label:
accuracy among accepted, wrong-assignment rate among accepted, coverage (accepted / hidden). **Halves: calibration leaves {1, 3, 6, 8}, report leaves {2, 4, 5, 7}** (by the hidden label).
Set choice on the calibration half only: highest accuracy with coverage ≥ 0.5 (ties within 0.005 → higher coverage); if no set reaches 0.5 coverage, highest accuracy. Ablation A / B / C reported side by side.
**Pass gate (report half, chosen set):** accuracy among accepted ≥ 0.90 AND wrong-assignment rate ≤ 0.05; coverage stated (high accuracy with low coverage still passes, said so).

## Outputs
`R8_pc/labels.npz` (label, source = stage1 / absorbed_round_k / unpainted reason, p1, p2) for the chosen set (A/B/C finals kept for the ablation); one table with denominator = all 46,719 reconstructed surfels (blade painted / unpainted, axis_kind 1, axis_kind 2, blade-like re-tested) for stage 1, R7, R8_pc, counts and mm²
(union of surfel ellipses per 10 mm cell, the R3-1b silhouette scale); per-leaf areas and the uncertain share (p1 − p2 < 0.25, top leaf = that leaf); the leaf 4 / 5 boundary; agreement with R7 (informational).
8833 toggle "점군 흡수 (기하+색)" (R7 / R8_pc / differences). Report `handoff/P1_PC_ABSORB_2026-10-27_KO.md` (+ EN).
Not done: no tuning on leaf 4 or after the report-half result; no other individual before "채택".
