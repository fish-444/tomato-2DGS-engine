# CONTRACT — P1 re-cluster the V-P compound leaves with geometry + photo features (fill + split)

Declared 2026-10-26 before any number. Every output: `R3-1b_unaccepted`, `conditional_on_unvalidated_scale`,
`grouping: user_labelled_propagated_reclustered`. Local CPU, GPU 0.

## 0. Input
- **V-P of 2026-10-25, number map v3 (ChArUco-corrected)**: `build/p1_compound_userlabel_20261025_01/map_v3_charuco/user_labelled_propagated/`
  (the "recheck 3 groups + one more ring photo + V-P re-run" step has **not** run, so this is the latest V-P). Per-surfel votes are recomputed
  from its saved inputs (seed maps = V-H+fill+topo of map v3, `propagated_maps.npz`) with the unchanged vote rule and checked to reproduce the
  saved labels exactly.
- Held-out photo 060551789, truth = the user's labels mapped with map v3 (held-out rows were never corrected). **No corrected truth exists**
  → only the original truth is scored (stated, not substituted).
- Surfels, Track A blade/axis flag, poses, photo features `build/p1_compound_cluster_20261025_02/r31b/photo_features.npz` (L_resid_z, a_z, chroma_z).
- Not used: 4b leaflet instances, the trace-flank variants (§9 of the user-label contract), the held-out photo for anything but the final score.

## 1. Anchors and the internal check
- **Anchor** = V-P labelled blade surfel with winner vote share ≥ **0.8** and ≥ **3** voting views. Never relabelled by any method.
- **Blocks** = whole user-labelled leaflet masks of the four labelling photos (map v3 numbers; held-out photo excluded), lifted to the anchor
  surfels visible under them (Revision-1 splat); an anchor in several masks belongs to the first by (photo, record time). Anchors in no block
  are never hidden.
- **Folds**: blocks shuffled with seed 0 and dealt greedily into **5 folds** of ≈ 20 % of the blocked anchors each. In fold f the anchors of
  its blocks are hidden (treated as "to decide"); everything (leaf statistics, geodesics, classifier training) uses only the visible anchors
  and the other folds' hidden pairs. All choices (feature forms, classifier, λ, P_min, M_min) are made on these out-of-fold results only.
- **Internal score** per method = (correct − wrong) / hidden surfels; also correct, wrong, abstained shares.

## 2. Graph and features
- Graph: blade surfels, k = 10 nearest within **3 mm**, edge = 3-D distance. Density = neighbours within 3 mm; density R = density /
  median density of the neighbours within 10 mm.
- Candidate leaves for a surfel = leaves whose visible anchors are within geodesic **37.2 mm** (2 × 18.6 mm).
- Pair features (surfel s, candidate leaf L), A and R forms:

| feature | A | R |
|---|---|---|
| height | z above pot top | z − p10 height of L's anchors |
| inclination | angle of the sign-free normal to vertical | angle between s's normal and L's mean anchor normal |
| azimuth | θ around the pot-centre vertical | circular |θ − L's anchor θ centre| |
| density | — | density R of s |
| density continuity | — | minimum density R along the geodesic shortest path to L's nearest anchor |
| distance | — | geodesic distance to L's nearest anchor ÷ 18.6 mm |
| brightness (photo) | L_resid_z of s | |L_resid_z(s) − median over L's anchors| |
| colour (photo) | a_z, chroma_z of s | |a_z − L median|, |chroma_z − L median| |
| shape prior | — | excess beyond L's anchor range along L's axis (anchor PCA major axis, oriented away from the pot-centre vertical) ÷ 18.6 mm; lateral distance from that axis ÷ 18.6 mm |
| V-P evidence | — | V-P vote share of L at s (votes for L ÷ all votes; 0 if none) |

- **Form rule**: per feature, AUC of "candidate is the true leaf" over the out-of-fold hidden pairs, for A and for R (AUC taken as
  max(auc, 1 − auc)); keep the higher form if its AUC ≥ **0.60**, else drop the feature. Photo features additionally need invariance ≥ **0.90**
  = Spearman correlation between the feature recomputed under each photometric perturbation and the original: exposure × 0.7 and × 1.3,
  gamma 0.8 and 1.25, white balance (R × 1.1, B × 0.9) and (R × 0.9, B × 1.1); the minimum over perturbations counts. Failing → dropped.

## 3. Methods
- **M1**: classifier on (surfel, candidate) pairs → per-surfel probabilities normalised over its candidates. Logistic regression (C = 1,
  standardised) vs HistGradientBoosting (default settings, seed 0): the one with the higher internal score at P_min 0.5 / M_min 0.1 wins.
- **Abstain** (all methods): M1 max probability < **P_min** or margin to the second < **M_min** → `unassigned`.
  Grid P_min ∈ {0.5, 0.6, 0.7}, M_min ∈ {0.1, 0.2}. Surfels with no candidate stay unassigned.
- **M2** = M1 + Potts smoothing on the 3 mm graph, minimised by ICM (10 sweeps, local optimum — not an exact graph cut; stated):
  energy = Σ −log p_s(L_s) + λ Σ_edges w_ij [L_i ≠ L_j], w_ij = exp(−(angle(n_i, n_j)/30° + |ΔL_resid_z| + |Δa_z|)) × min(1, densR_i, densR_j)
  (photo terms only if those features survive §2). Anchors fixed; abstained surfels take no label and contribute no pairwise term.
  λ ∈ {0.5, 1, 2}.
- **M3** = M2 + shape constraints: a non-anchor surfel assigned to L becomes unassigned if its along-axis position lies more than 18.6 mm
  beyond L's anchor range, or its lateral distance exceeds L's anchor lateral p95 + 9.3 mm; a connected piece of L (3 mm graph, non-anchor)
  with no anchor and farther than 18.6 mm (geodesic) from L's anchors becomes unassigned.
- No method changes an anchor label. Counts reported separately: **filled** (V-P unassigned → leaf), **relabelled** (V-P leaf → other leaf),
  **un-set** (V-P leaf → unassigned).

## 4. Choice and the held-out score
Internal check chooses: classifier, then per method the (P_min, M_min[, λ]) with the highest internal score. Then the held-out photo is scored
**once** for V-P (baseline), M1, M2, M3: per-leaf IoU, confusion matrix, coverage (= share of the held-out painted pixels that get a rendered
3D number). Gate unchanged: median per-leaf IoU ≥ **0.60** and max confusion ≤ **0.10**. Pick: gate pass → higher median IoU → lower max
confusion. If none passes: the best is reported as `gate_failed_viewing_only` with per-leaf flags **usable** (held-out IoU ≥ 0.55 and
confusion ≤ 0.10) vs **viewing only**.

## 5. Outputs
`surfel_leaf_ids_v2.csv` (row, leaf_id, probability, source = anchor / filled / relabelled / kept / unassigned, method), per-leaf table (visible
area ①, mean normal / inclination, azimuth, height range, quality flag). Viewer colour sets "복엽 (재군집)" and "채움/수정/원래" + feature
maps used. Headless check before publishing.

## 6. Prohibited
Tuning on the held-out photo, editing user labels or the held-out truth, changing anchors, filling without the abstain rule, cross-leaf
growth without the pairwise cut, 4b as input, `microtom_synth_v2`, 토마토2, deleting files or db entries.

## Revision 1 (2026-10-26, declared after run 1 and before run 2)
Run 1 (kept unchanged in `run1_with_vp_share/`) made the internal check uninformative: every classifier, every (P_min, M_min, λ) scored
0.9999 for M1/M2 (0.9981 for M3), because the feature **V-P vote share** leaks the hidden labels — a hidden anchor is by definition a surfel
whose V-P share is ≥ 0.8, so the feature reproduces the V-P label exactly. Choices in run 1 were therefore ties settled by order, not by
evidence. Run 1's held-out result was seen (M1/M2 median IoU 0.566, max confusion 0.467; coverage 0.767).
Revision 1: the V-P vote-share feature is **excluded** from the feature set (form rule, classifiers, grids, M2/M3 and abstain unchanged).
Nothing else changes. The held-out photo is scored a **second** time (selection risk: two runs seen; both reported side by side; the choice
of run is not made on the held-out numbers — run 2 is the one whose internal check is valid by construction).
