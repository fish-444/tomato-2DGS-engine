# CONTRACT ADDENDUM — post-hoc acceptance (2026-10-22)

Run `build/p1_recon_charuco_20261021_01/`. Prompt "Continue the ChArUco pot reconstruction under a lowered acceptance (Steps 4–6)".
Written 2026-10-22 11:16, before any Step 4–6 work. CONTRACT.md is not changed.

## Decision
- Lowered acceptance: **user decision 2026-10-22 10:29 ("기준 내려서 복원"), made after seeing the held-out results**. It is labelled
  `post_hoc_acceptance` everywhere this run's outputs are used; Steps 4–6 outputs carry `acceptance: post_hoc`.
- A new capture on the foam-mounted board is still the clean test; it gets these criteria pre-declared (outside this task).

## Criterion (`accepted_post_hoc` if all hold on the 3 held-out views)
1. edge F-score at 2 px ≥ 0.90;
2. silhouette IoU ≥ 0.65;
3. held-out IoU ≥ training-view median IoU − 0.05.

Definitions (as in the Step 3 diagnostic of `handoff/P1_RECON_CHARUCO_EXECUTION_2026-10-22.md` §5): rendered silhouette = 2DGS alpha
≥ 0.5 from `kernel_out/out/views.npz`; reference = plant label of `masks/<view>.png` (1512 px); edge F at t px = harmonic mean of
precision (rendered pixels within t px of the reference, elliptical dilation radius t) and recall (reference pixels within t px of
the render); IoU from `kernel_out/out/done.json`. Existing Step 3 numbers only: no retraining, no re-rendering, no tuning.

## Declared Step 4 thresholds (used only if the check passes)
Quadric when plane RMS / extent > 0.03; `edge_soft` when ≥ 50 % of a leaflet's projected footprint in its supporting views lies
within 3 px of the reference/render disagreement band; link IoU ≥ 0.3; ≥ 2 agreeing views; < 3 supporting views = `few_views`.

## Check (computed 2026-10-22 11:15 from the stored Step 3 outputs)
Training-view median IoU 0.7399 → bound for check 3 = 0.6899.

| held-out view | edge F @ 2 px (≥ 0.90) | silhouette IoU (≥ 0.65) | IoU ≥ 0.6899 |
|---|---|---|---|
| 060326166 | **0.8980 — fails** | 0.6997 — passes | 0.6997 — passes |
| 060448335 | 0.9059 — passes | 0.6914 — passes | 0.6914 — passes |
| 060605333 | 0.9237 — passes | 0.7301 — passes | 0.7301 — passes |

**Result: `accepted_post_hoc` is NOT reached** (check 1 fails on 060326166 by 0.002). As this addendum requires, the run stops here;
Steps 4–6 are not run. Note: the 10-22 execution report gave this range as "0.90–0.92"; the exact values are 0.898 / 0.906 / 0.924
(the lower bound was rounded up there). The threshold is the user's to set; it is not changed in this addendum.

## Amendment 1 — user threshold (2026-10-22, after the check above)
User message after the 11:20 report: "2 px 경계 F ≥ 0.89". Criterion 1 becomes **edge F-score at 2 px ≥ 0.89**; criteria 2 and 3 and
all definitions are unchanged. This is a second post-hoc decision, made after seeing the exact values; the run stays labelled
`post_hoc_acceptance` / `acceptance: post_hoc`.

| held-out view | edge F @ 2 px (≥ 0.89) | silhouette IoU (≥ 0.65) | IoU ≥ 0.6899 |
|---|---|---|---|
| 060326166 | 0.8980 — passes | 0.6997 — passes | 0.6997 — passes |
| 060448335 | 0.9059 — passes | 0.6914 — passes | 0.6914 — passes |
| 060605333 | 0.9237 — passes | 0.7301 — passes | 0.7301 — passes |

**Result: `accepted_post_hoc`.** Steps 4–6 proceed with the thresholds declared above. Step 4 outputs go to a new subfolder
`leaflets_3d/` (the existing `leaflets/` holds the Step 1 H2 masks and is not changed), the review page to `review/`, the hand-off
files to `handoff/`.
