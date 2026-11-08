# Contract — R8_pc_unmix: rachis pieces, blade and whole leaflets mixed into another compound leaf separated, R3-1b, 2026-10-29

Request: user "엽축, 엽신, 소엽 포함해서 다른 복엽에 섞여있는것 분리". Declared before any number. Base = the latest pipeline: R8_pc (set B) + rachis absorption (`R8_pc_rachis/labels_rachis.npz`).
Labels: `designated_not_adopted` · `conditional_on_unvalidated_scale`. No photo / SAM / V-P as evidence (photos only score the result, as before). R8_pc and R8_pc_rachis unchanged; result = new layer `R8_pc_unmix`.

## Units
- **Rachis spine** of leaf k: the 3 mm component of its rachis-absorbed surfels that lies nearest to the user's trace k (leaf 8, no trace: its largest component). Other components of leaf k's rachis = **rachis pieces**.
- **Leaflet** = an R3-1b 3D-first patch (`second_attempt/patches_filled.npz`) with ≥ 30 labelled surfels. A leaflet belongs to one compound leaf.

## Graph and distance
Graph over all labelled surfels plus unlabelled blade / rachis-like surfels: blade–blade edges = the sheet graph (≤ 3 mm, normal ≤ 30°, offset ≤ 1.529 mm, same sheet); edges touching an axis_kind 2 surfel = plain ≤ 3 mm. Edge length in mm.
d_k(x) = geodesic distance from x to leaf k's spine (multi-source Dijkstra from the spine surfels).

## Rules (one pass, in this order)
1. **Rachis pieces:** each piece of leaf k goes to the leaf j with the smallest median d_j over its surfels if d_j ≤ d_k / 1.5; otherwise stays k.
2. **Mixed leaflets:** a leaflet whose largest leaf share is < 0.9 gets one label for all its labelled surfels: the leaf with the smallest median d over its surfels if it is ≥ 1.5 × shorter than the runner-up; else its majority leaf if share ≥ 0.6; else unchanged and flagged `mixed_unresolved`.
3. **Foreign leaflets:** a leaflet labelled k (share ≥ 0.9) moves to leaf j when the median d_j over its surfels ≤ median d_k / 1.5 (it is attached to j's spine, not to k's). Its own spine surfels never move.
4. Leftover blade surfels of leaf k not inside any leaflet and with d_k = ∞ (cut off from k) take the leaf j with finite d_j ≤ 15 mm, if exactly one; else unchanged.
Spines themselves and the user's traces are never changed.

## Checks (scoring only)
- Purity on the 4 labelling photos (human maps, projections counted): moved surfels under their new label vs under their old label; < 100 projections = not measurable.
- Per leaf: surfels moved in / out, leaflets per leaf before / after; R8_pc rachis self-test is not rerun.
- Agreement with human-seeded R8_pc (before) is reported, not a target.
**Accept** (declared): moved surfels' purity under the new label ≥ their purity under the old label AND ≥ 0.6; otherwise the layer is `viewing_only`.

## Outputs
`build/p1_pc_absorb_20261027_01/R8_pc_unmix/labels_unmix.npz`, `moves.csv`, `results.json`; shared viewer set "잎 섞임 분리 (R8_pc 위)" + differences on request; report `handoff/P1_PC_UNMIX_2026-10-29_KO.md`.

## Amendment A1 — second attempt, labelled `tuned` (2026-10-29, after the first result; user "ㅇ")
Reason: attempt 1 failed (moved-surfel purity 0.069) because the spines came from the rachis absorption, which had already given leaf 6's rachis to leaf 7.
Change (only this): the **spine of leaf k = every non-stem surfel within 3 mm of the user's trace k** (the user's own 3D input). Leaves without a trace (leaf 8) are **frozen**: their surfels never move and they are never a target.
Rachis pieces = 3 mm components of leaf k's rachis-absorbed surfels outside every spine. All other rules, ratios, thresholds and the accept rule unchanged. Output `R8_pc_unmix_tuned/`.
