# CONTRACT AMENDMENT 3 — B traits from the user's skeleton lines (addendum 2026-11-01 23:40)

Written before any skeleton trait number.

## Input files
- Chosen via `<plant>/skeleton/latest.json` in `build/p1_all_3dfirst_20261027_01`. File names and sha256 are in
  `skeleton_inputs_amendment3.json`. If `latest.json` changes later, this run still uses the recorded files and the report
  says that a newer file exists.
- Plants: e_01, e_04, e_05, e_07, n_01, n_02, n_03, n_05, n_06.
- Not used:
  - n_04: clicks cancelled.
  - e_02, e_03, e_06: no 3D-first merge.
  - t2_02, t2_04, t2_05: excluded until the user confirms trays.
  - R3-1b: not in this file set.
- Labels: `skeleton_labels_ungraded`, `recon_trait`, `conditional_on_unvalidated_scale`, `after_checkpoint_v1`, `age_dag=60`, `user_stated`.
- Any template built from these values is `ungraded_draft`, not frozen.
- Per-leaf numbers are `display_only` (no grade). Plant-level aggregates are the result.

## Geometry (fixed now)
**Frame and base**
- The branch polylines are in the board frame, in mm, z up.
- Surfel xyz in `groups.npz` is in m and is multiplied by 1000.
- Leaf base = the branch end horizontally closer to the stem proxy line. The report states how often this equals the first click.

**Stem proxy**
- The plants have no stem polyline, so the stem proxy is a vertical line through the median xy of the plant's `axis_like`
  surfels. It is named `axis_vertical_proxy`.
- The §3 synthetic check used a noisy stem polyline, not this proxy. Traits that depend on the axis (B1 divergence, B2 insertion
  angle, B3 internode) therefore carry `axis_proxy_untested_s3`.
- θ0, θ_tip, Δθ, curvature, s_apex/L, tip drop and L do not need the axis.

**Leaf selection and densification**
- Branches with fewer than 2 points or with L < 25 mm are skipped.
- The addendum's `leaf_missing_or_merged` leaves are skipped: n_01 leaf 17, n_02 leaf 17 (0 surfels), e_07 leaf 5 (65 surfels).
- Densification as in CONTRACT §2: points of that branch (label = branch, status = assigned) that are `axis_like` and lie within
  3 mm of the line, in 2 mm arc bins.

**Rank and per-trait rules**
- Rank = order of base height z (lowest = rank 1); `rank_from_base_height`.
- B1: azimuth of the base chord [0, 10] mm about the vertical proxy; divergence between consecutive ranks; handedness rule as in
  §2.
- B2: θ0, θ_tip, Δθ, curvature, s_apex/L, tip drop, arc-vs-quad fits (`recon_traits.arch_traits`). Insertion angle = angle of the
  base chord to vertical.
- B3: L (arc length of the densified line); internode = difference of consecutive sorted base heights; node count = number of
  usable branches.

**C1 curl**
- C1 is `trait_unreliable` (§3), and `groups.npz` has no normals, so it is not computed.

## Template rules
**Hand vs reconstruction**
- Hand θ0 vs skeleton θ0 can only be compared at distribution level, because the dt → ID map is missing. Pooling is not allowed.
- Hand values stay the template values. The reconstruction values go into a separate column `recon_ungraded_draft`, with the
  difference to hand (by rank) reported for information.

**Output**
- `MTG_TEMPLATE_v2_ungraded_draft.csv`, not frozen. v1 is unchanged.
