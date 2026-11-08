# CONTRACT p1_skeleton_assign_20261101_01 - user skeleton clicks -> compound-leaf labels (user 2026-11-01: "찍은 점을 골격이라고 생각해봐", then "진행")

Written 2026-11-01 before any number. Labels: `user_skeleton_seeds`, `no_reference` (no human leaf labels exist for these individuals, so nothing is scored), `conditional_on_unvalidated_scale`, `R3-1b-style_unaccepted`.
Light local computation only (distances from <= 100k surfels to a few hundred skeleton points, seconds, < 1 GB); runs inside the 8835 page server on request ("골격으로 배정 계산") or as `python -m tomato3d.organs.skeleton_assign --id <id>`.

## Inputs
- Skeleton = the user's alive clicks in `build/p1_all_3dfirst_20261027_01/user_base_click.jsonl` (replay of `add` / `undo`), in click order, per line; within a line a step > 20 mm from the previous click starts a new branch (the same rule the page draws). **Each branch = the skeleton (rachis trace) of one compound leaf**; a one-point branch is a base point.
- Surfels, 3D-first groups and the `axis_like` class from `<id>/merge/groups.npz` (frozen 3D-first run).

## Rule (declared now, not tuned after seeing a result)
1. Each branch is densified to points every 1 mm along its polyline.
2. Distance of every surfel to every branch = nearest densified point.
3. 3D-first groups (leaflet-sized units) are assigned as a whole: per group, the median distance of its surfels to each branch; best branch if best <= 25 mm and (second - best) >= 3 mm, else the group is `contested`.
4. Surfels not in a group (unassigned blade and `axis_like`) are assigned one by one with the same rule (25 mm, 3 mm margin).
5. Output `<id>/skeleton/labels_skeleton.npz` (label per surfel = branch number, 0 = none; status 0 unassigned / 1 assigned / 2 contested; branches as polylines) and `summary.json` (branches, clicks, surfels per branch, contested and unassigned shares). Every recomputation writes a new timestamped file; earlier files are kept.

## Limits
Distance to a hand-drawn skeleton is a simple rule: a leaflet lying closer to a neighbouring rachis than to its own goes to the neighbour; touching leaves stay `contested` when the margin is small. Nothing here is accuracy; the user checks the colours on the page. R8_pc, rachis absorption and the frozen outputs are not changed or used.

## Amendment A1 (2026-11-01 23:45, user "나머지 개체들로 진행"; before any new number)
- The skeleton is now the user's leaf lines made after the 22:27 reset: **one line ("다음 잎") = one compound leaf, never cut by a distance rule** (the 20 mm cut of the first version applies to nothing any more; clicks without a `line`, the legacy log, use the editor's vertical rule |dz| > 10 mm). Section "Rule" (densify 1 mm, group-median distance, <= 25 mm, margin >= 3 mm, contested) is unchanged.
- Run on every plant that has leaf lines except n_04 (its clicks were cancelled by the user at 23:40): e_01, e_04, e_05, e_07, n_01, n_02, n_03, n_05, n_06, t2_02, t2_04, t2_05. One run per plant; no reference exists, nothing is scored; a recomputation writes a new timestamped file. The three plants without a 3D-first merge (e_02, e_03, e_06) are not run (no groups).
- The page button "골격으로 배정 계산" uses the same code.
