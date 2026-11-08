# CONTRACT — P1: T-prox as operating route for the other individuals (distance route kept alongside)

Declared 2026-11-01 ~14:50, before any number of this run. Source: CLI prompt 13:55 "switch other individuals' operating route to T-prox (conditional)"; user decision 2026-11-01 13:48 ("yes, use T-prox for reconstructing the other individuals", conditions in the prompt are part of the decision).
Labels on every output: `route_tprox`, `tprox_unverified_until_heldout`, `conditional_on_unvalidated_scale` (+ `user_skeleton_seeds`, `frozen_code_reuse`, `legacy_split_20mm_unreviewed` where the skeleton carries that flag).
Input hashes: `inputs_hashes.md` (T-prox code `skeleton_r8pc.py`, frozen `pc_absorb.py` / `pc_rachis.py`, frozen bundle and labels, every skeleton, distance labels, 3D-first groups, axis labels).

## 1. Route (unchanged from the R3-1b pass, A5 of build/p1_skeleton_layer_20261101_01)
- Seeds: `skeleton_r8pc.seeds(..., rachis_any=True)` — rachis seeds = surfels <= 3 mm from exactly one line (no axis-label condition); blade seeds = whole 3D-first groups (axis_kind 0 part) with median distance <= 8 mm to one line and >= 13 mm to every other. ADAPT = rachis 3.0 / blade 8.0 / margin 5.0 mm. No change.
- Bundle: `skeleton_r8pc.step_bundle(pid, mode="prox")` (non-R3-1b branch, unchanged). One leaf per live skeleton line with >= 2 points (lines with 1 point cannot form a trace: listed as `line_too_short`, no T-prox label; the distance route keeps them).
- Axis labels: the existing S1 proxy + frozen Q1 files `build/p1_skeleton_layer_20261101_01/r8pc/<pid>/axis_labels.npz` (+ axis.json), copied byte-identical into `runs/<pid>/`. `axis_share_unusual` is expected for every plant and is reported, not hidden.
- R8_pc: frozen `pc_absorb.py` copied unchanged to a Kaggle CPU kernel (one per plant; dataset slug != kernel title).
  - **Weight set declared: A.** The passing R3-1b T-prox run (20261101T133038) used set A, picked by the in-kernel self-test rule. That rule's calibration/report halves are R3-1b leaf numbers (1,3,6,8 / 2,4,5,7) and mean nothing on another plant, so the operating label is set A on every plant. The in-kernel choice is recorded, not used. This is fixing the passing configuration, not tuning.
- Rachis absorption: frozen `pc_rachis.py` run locally (8 s on R3-1b) with D0 = the run folder and `R8_pc/labels.npz` := the set-A labels.
- Output label = `R8_pc_rachis/labels_rachis.npz` label (0 = contested or unpainted).

## 2. Scope and order
Eligible = `data/skeletons/<pid>/<date>/skeleton_v1.json` with >= 1 live line of >= 2 points AND `build/p1_all_3dfirst_20261027_01/<pid>/merge/groups.npz`. Skeletons have no "complete" field; every eligible skeleton is the converted legacy click log (flag `legacy_split_20mm`, not re-checked by the user) — run, and carry `legacy_split_20mm_unreviewed`.
Order: n_04, t2_02, then e_01, e_04, e_05, n_01, n_02, n_03, n_05, n_06, t2_05. At most 4 Kaggle kernels at once.
`not_run`: e_02, e_03, e_06 (no 3D-first groups), e_07 and t2_04 (no skeleton). No automatic seeds, no SAM-as-truth, no substitute route. One run per plant; a kernel that fails for an infrastructure reason (push / quota / download) may be re-pushed once with the same bundle, labelled `infra_retry`.

## 3. Outputs per plant (new folder only; nothing earlier overwritten)
- `labels_tprox_<pid>.npz`: rows, label (index into leaf_uids, 1-based; 0 none), status (1 assigned / 0 contested), leaf_uids, p1, p2, margin = p1 - p2 — the same layout as `labels_skeleton_distance.npz` (+ p1/p2/margin), so the readers of distance labels read it unchanged.
- `labels_distance_<pid>.npz`: byte-identical copy of `data/skeletons/<pid>/<date>/labels_skeleton_distance.npz`.
- `diag_<pid>.json`: lines / leaves with T-prox labels, `line_too_short` lines, seeds per leaf (rachis, blade), contested share, unlabelled share, p1 and margin quantiles (5/25/50/75/95 over assigned surfels), leaves with seed surfels < 50 (declared minimum), axis_share_unusual, agreement matrix T-prox vs distance by leaf_uid (and the share of surfels whose leaf changed among surfels assigned by both; plus assigned-by-one-only counts).
- `fig_<pid>.png`: one prepared photo (first view of `rf_inputs/`) | points coloured by T-prox leaf (top view) | points where T-prox and distance disagree.
- `plant_tree_tprox_<pid>.json`: `skeleton_tree.skeleton_to_plant_tree` with leaf areas from T-prox labels (area_b_silhouette_mm2) where `per_surfel_light.npz` exists, else no areas. Engine code untouched.
- Per-leaf light: only where per_surfel_light exists AND the plant has grade A/B. Now no plant has held-out picks -> `ungraded`, per-leaf light not reported; e_* have no light file -> `light_not_available`.

## 4. Held-out judgement (rule declared now; applied when picks exist)
For a plant with >= 10 held-out picks (never seeds): Wilson CI agreement of T-prox and of distance with the picks, side by side; grade from the better route, labelled with which route gave it. If T-prox < distance for a plant, that plant's operating route falls back to distance. A/B -> per-leaf light; C or lower -> plant-level only; no picks -> `ungraded`.

## 5. Added reporting on R3-1b (from existing results, no new run)
On run 20261101T133038 (T-prox, completed simulated skeleton), report half 2/4/5/7, front-visible human projections (A4 metric): margin = p1 - p2 of `labels_rachis.npz`; Q75 = 75th percentile of margin over assigned scored projections; report (i) wrong assigned projections, (ii) share of them with margin >= Q75, (iii) wrong rate inside the top quartile, per leaf table. Leaf 6 (~0.02 in every route) is stated as a known failure, not a route property.

## 6. Report
Korean 부록 5 in `handoff/P1_SKELETON_LAYER_2026-11-01_KO.md` (<= 10 lines + tables), LOG RESULT with hashes, flow row.

## Amendment A1 (14:05, after the first section-5 computation, before any per-plant number)
Observation: in `labels_rachis.npz`, p1/p2 are the rachis step's own values and are 0 for every label carried from R8_pc. The first section-5 output therefore had Q75 = 0, and "100 % of wrong in the top quartile" is an artefact. That output is kept as `r31b_tprox_confidence.json` (label `degenerate_margin`), not reported as a finding.
Replacement (applies to section 5 and to the per-plant margin quantiles):
- Provenance classes per surfel:
  - `seed` = R8_pc source 1 (stage1 — forced, never weighed against other leaves);
  - `absorbed` = R8_pc sources 2–6;
  - `rachis_step` = rachis-file sources 2–7 (trace_seed / rounds);
  - `none` = label 0.
- Margin = p1 − p2 of `R8_pc/labels.npz`, defined for `absorbed` only; Q75 over absorbed scored projections.
- R3-1b report (report half and per leaf), giving wrong assigned projections split by class:
  - wrong share and wrong rate per class;
  - for absorbed, the share of wrong with margin ≥ Q75, and the wrong rate in that top quartile versus below it.
- Per plant: surfel counts per class; margin quantiles over absorbed only.

## Amendment A2 (14:30, implementation defect found at the first finish, n_04; before any per-plant number was used)
Defect: `skeleton_r8pc.traces_arrays` (non-R3-1b branch, never run on Kaggle before this task) numbered trace leaves over ALL live lines (k+1, one-point lines skipped but their number consumed), whereas `seeds` / `leaf_map` number only lines with >= 2 points. For every plant with a one-point line the bundle's trace_leaf (up to 17 on n_04) disagreed with stage1 / leaf_map (1..12) -> IndexError in the local label mapping, and the traces handed to the frozen code carried wrong leaf numbers.
Fix: `traces_arrays` filters to lines with >= 2 points before numbering (same list/order as leaf_map). No seed rule, threshold or frozen-code change; R3-1b branch untouched.
Consequence: the bundles pushed before the fix (n_04, t2_02, e_01, e_04, e_05) are discarded as `numbering_defect` (kept on disk, not used, not deleted) and every plant is re-run once with the fixed bundle. This is a defect rerun, not a second attempt at a result. Their Kaggle kernels are left to finish unused.
Also: the finishing step now sets the reference to the fixed bundle's leaf_map only.

## Amendment A3 (17:25, infrastructure outage)
The 14:25–14:59 pushes of t2_02, e_01, e_04 (DNS failure for api.kaggle.com, dataset and kernel both rc 1) and the e_05 dataset (SSL EOF) never reached Kaggle; the driver then polled non-existent kernels for 2 h (status None). n_04 completed normally. Per section 2 (`infra_retry`): the same bundles (unchanged files, same stamps) are pushed once more for t2_02, e_01, e_04, e_05; remaining plants follow in order. Poll now treats a missing kernel as a failure instead of waiting forever.
