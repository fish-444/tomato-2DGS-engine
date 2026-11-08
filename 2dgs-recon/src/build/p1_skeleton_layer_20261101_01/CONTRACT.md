# CONTRACT - P1 compound-leaf skeletons as a separate layer -> propagate -> reuse at every re-capture (declared 2026-11-01, before any number)

Prompt: `CLI PROMPT — P1: compound-leaf skeletons as a separate layer → propagate to each individual → reuse the same procedure in every later growth (re-capture) session` (2026-11-01 01:05); final freeze 2026-11-07 12:00.
One instance: LOG.md checked 2026-11-01 (the last entries are the skeleton-assignment work this builds on); `RUNNING.lock` written. Builds on `handoff/P1_SKELETON_ASSIGN_2026-11-01_KO.md` (page 8835 `mode=base`, `skeleton_assign.py`).
Labels on every output: `user_skeleton_seeds`, `conditional_on_unvalidated_scale`, `after_checkpoint_v1`, `R3-1b-style_unaccepted` (R3-1b: `R3-1b_unaccepted`). Existing click logs and outputs are kept; nothing is overwritten.

## Inputs (path, sha256[:16], modification time)
| path | sha256[:16] | mtime |
|---|---|---|
| `tomato3d/organs/skeleton_assign.py` | 66295c577e2bc952 | 2026-11-01 00:45 |
| `tomato3d/organs/all3d_page.py` | 8122946b8babcb3a | 2026-11-01 00:46 |
| `tomato3d/organs/pc_absorb.py` | 372c06655d939376 | 2026-10-27 21:46 |
| `tomato3d/organs/pc_rachis.py` | b9e76f0b96a0a912 | 2026-10-29 20:32 |
| `tomato3d/organs/pc_absorb_prep.py` | dfec925e19dd08a8 | 2026-10-27 21:43 |
| `tomato3d/organs/local_diffusion.py` | f9a435d1d504c3bc | 2026-10-27 13:55 |
| `build/p1_pc_absorb_20261027_01/kaggle_data/r8pc_bundle.npz` | ec7cfbce98c8c642 | 2026-10-27 21:43 |
| `build/p1_pc_absorb_20261027_01/R8_pc/labels.npz` | e1ff792be1c4efdf | 2026-10-27 21:54 |
| `build/p1_pc_absorb_20261027_01/R8_pc_rachis/labels_rachis.npz` | ddcbac13dce1d34c | 2026-10-27 23:05 |
| `build/p1_pc_absorb_20261027_01/LATEST_PIPELINE.json` | a2f00e3aa6c41526 | 2026-10-27 23:33 |
| `build/p1_all_3dfirst_20261027_01/user_base_click.jsonl` | 7463d7e1bbeec6d8 | 2026-11-01 00:40 |
| `build/p1_skeleton_assign_20261101_01/CONTRACT.md` | b0c624f741076e2d | 2026-11-01 00:45 |
| `build/p1_compound_2dvote_20261025_01/traces_export/rachis_traces/leaf1_mvnnbr9l.json` | ea1b42b57a7090a7 | 2026-10-25 19:46 |

Frame convention of every skeleton file: board frame, metres, z up (`charuco_a3`); the click log stores millimetres (`point_mm`), the schema stores metres (`*_xyz`).

## 1. Scope of this run (declared before any number) and what is deliberately NOT run
Built in this task: schema `skeleton_v1` + I/O + validator (`tomato3d/organs/skeleton_io.py`, `SKELETON_V1.md`), conversion of the existing n_04 clicks, propagation wrapper
(`skeleton_propagate.py`), skeleton -> `plant_tree_v1` (`skeleton_tree.py`), carry-forward procedure (`skeleton_carry.py`), editor changes on page 8835, and the R3-1b simulated-click check.
Frozen and untouched: `pc_absorb.py` (R8_pc), `pc_rachis.py`, `local_diffusion.py`, `skeleton_assign.py` distance rule (reused as `distance_baseline`), the 3D-first grouping, light outputs, the engine.
No threshold is changed after a held-out result; a change = new contract + new labelled run. Heavy compute on Kaggle only; click pages and small I/O local.

**Gap found while reading the frozen code (reported, not worked around).** The frozen main route (stage 1 R7 diffusion -> R8_pc -> R8_pc_rachis) takes a bundle (`r8pc_bundle.npz`) whose arrays exist for
R3-1b only: `axis_kind` (Q1 blade / stem-like / rachis-like) and `stage1` (R7 labels grown from the V-P base, which came from human labels and photos). The nine other plants have binary masks only (no Q1 axis
labels) and no V-P base. Building those arrays would be new science, which this task may not write. Therefore:
- main route (R8_pc) on the nine: `not_runnable_inputs_missing` (axis_kind, stage1); the report says so per individual; by the prompt's own rule ("usable if the R8_pc route reaches grade A or B") per-leaf light is **not reported** for them;
- `distance_baseline` (the existing 25 mm / 3 mm rule on 3D-first groups) runs for every individual that has a skeleton, with held-out agreement and Wilson CI, as the comparison route;
- the main route is exercised on R3-1b only, in the simulated check (section 5).

## 2. Schema decisions the prompt left open
- `leaf_uid` = `<plant_id>-L<NN>` (two digits), given once in creation order and never reused; `rank_hint` = rank by base height, recomputed each date.
- `polyline_xyz` = the clicked surfel positions in click order (base -> tip); derived field `polyline_1mm_xyz` = linear resampling at 1 mm. Clicks are not pulled toward any axis candidate.
- `n_clicks` = alive (non-undone) clicks of the line; `click_log_lines` = the 1-based line numbers of the append-only click log that make up the line.
- Validator (flags, never auto-fixes): (a) base nearer to the plant axis than the tip, else `base_tip_order`; plant axis = vertical through the surfel-xy centroid unless a pot frame is given; (b) two leaves with
  bases within 3 mm -> `shared_base`; (c) a skeleton point farther than 4 mm from every surfel -> `off_surface`; (d) a step > 20 mm inside a line -> `gap_over_20mm` (warning only; the old rule that split lines is gone).
- `status` values: `active`, `new`, `lost`, `merged_into:<uid>`, plus `candidate_lost` (carry-forward only, section 6).
- Click-log format for the new editor: `data/skeletons/<plant_id>/<date>/clicks.jsonl`, actions `add`, `undo`, `new_line`, `extend`, `move_point`, `delete_point`, `mark_lost`, `merge`, `heldout_pick`; append-only, never rewritten.

## 3. Conversion of the existing n_04 clicks
`build/p1_all_3dfirst_20261027_01/user_base_click.jsonl` (n_04: 77 records, 45 alive clicks) has no `line` field (every record is line 1) and no explicit line ends. Under the new rules lines are ended by the user;
for this one legacy log the only available evidence of line ends is the old 20 mm rule, so the n_04 skeleton is created with branches split by that rule and every leaf carries
`source = user_click` plus flag `legacy_split_20mm` until the user re-checks the lines in the new editor ("다음 잎" / merge). The original log is not touched.

## 4. Propagation (section 3 of the prompt) as run here
- `distance_baseline`: `skeleton_assign.assign` logic unchanged (25 mm, 3 mm margin, 3D-first groups whole, contested otherwise), fed from skeleton_v1 polylines instead of the raw click log (identical polylines for the same lines).
- Outputs per individual and date: `labels_skeleton_distance.npz`, `summary.json` (assigned / contested / none shares, held-out agreement with Wilson 95 % CI, contested light share when the frozen per-surfel light exists),
  versioned by timestamp; `labels_skeleton_r8pc.npz` only where the main route ran.
- Held-out agreement = among held-out clicks that are not "모름", the share whose route label equals the user's `leaf_uid` (contested and none count as disagreement); grade A: >= 0.8 with n >= 8; B: 0.6-0.8; C otherwise
  (n < 8 can never be A).

## 5. R3-1b simulated check (once)
Skeleton built from the 10 human rachis traces (`simulated_from_human_traces`; leaf numbers = the trace leaf numbers, so `R3-1b-L01` ... ). Round trip: skeleton -> traces in the frozen trace format must reproduce the 403 densified
trace points of `r8pc_bundle.npz` (max deviation <= 0.01 mm, same leaf per point). Then the frozen bundle with its trace arrays replaced by the exported ones is run through the frozen `pc_absorb.py` (set B) on Kaggle CPU and the frozen `pc_rachis.py`
locally. **Plumbing pass**: label agreement with the frozen R8_pc / R8_pc_rachis labels >= 0.999 of surfels. Scoring against the human leaves (scoring only) is reported next to the frozen values (R8_pc 0.8326, rachis 0.9362);
no gate is re-judged here.

## 6. Carry-forward (`skeleton_carry.py`, procedure of section 6 of the prompt)
Registration: board/pot frame first; rigid ICP (no scaling) on the pot rim and the lowest 20 mm of plant surfels, residual reported; carrying: transform each polyline, snap each point to the nearest axis/rachis surfel within 5 mm (else keep and flag),
`source = propagated_from:<date>`; base with no surface within 8 mm -> `candidate_lost`; after the t2 skeleton: per-`leaf_uid` growth table (Delta rachis length, base height, azimuth, elevation; area and light Deltas when labels exist),
`check_identity` when the base moved > 10 mm or azimuth changed > 30 deg. Tested on synthetic transforms here; no second capture date exists yet for any plant, so it is not run on real data.

## 7. Editor (page 8835, `mode=base`) and what is deferred
Implemented only as far as it can be tested headless without disturbing a live clicking session; the page is restarted only when its click log has been quiet for 30 minutes. Items: line ended by the user, 20 mm jump = warning only,
existing lines shown with `leaf_uid`, actions (extend / move / delete point / mark lost / merge), held-out step (10 stratified surfels, >= 5 mm from any skeleton point), Korean instructions, minutes per individual.
Whatever is not finished is listed in the report as not done.

## Amendment A1 (2026-11-01, before the R3-1b check ran)
Section 5 said the swapped bundle runs through the frozen pc_absorb.py on Kaggle. The skeleton export is lossless (points kept exactly, original trace order in attrs), so the swapped bundle is first compared with the frozen r8pc_bundle.npz array for array. If trace_pts, trace_leaf and trace_id are identical, the frozen result files are by construction the output of this route (deterministic frozen code, identical input) and no rerun is made; the check then reports plumbing_pass and quotes the frozen numbers (R8_pc 0.8326, rachis 0.9362). If any array differs, the Kaggle rerun of section 5 runs with its 0.999 agreement rule.

## Note (2026-11-01 01:20): editor restart
Page 8835 restarted with the new editor (same --lan token) after the click log had been quiet since 00:40 (> 30 min, section 7 rule); headless check on a second loopback instance first (GET /api/skel, /api/heldout, refused bad actions, no file written by GET). Minutes per individual = sum of gaps <= 10 min between consecutive click / edit records (declared here).

## Amendment A2 (2026-11-01 01:50, addendum "enable the main route for the other individuals"; declared before any run). Labels add `frozen_code_reuse`.
**Axis classification, what exists.** R3-1b's `axis_kind` (0 blade / 1 stem-like / 2 rachis-like) = Track A blade/axis labelling (mask stem class + H2 leaflet labels, i.e. photo inputs; `light_organs.py`)
followed by the frozen Q1 logistic (stem-like vs rachis-like on `h_rel, vert, thick_rel, dens`; `compound_cluster.q1_and_forms(OUT2)`, held-out rachis-like share 0.76 / 1.00 / 1.00 on R3-1b). The nine plants have
binary masks only, so Track A's first step has no input; the only frozen blade/axis split that exists for them is the S1 PCA proxy of the 3D-first contract (`axis_like` in `merge/groups.npz`, l2/l1 < 0.25),
which marks **1.4-4.1 %** of surfels against R3-1b's **29.3 %**. By the addendum's own rule every one of the nine is therefore flagged **`axis_share_unusual`** (outside 0.293 +/- 50 %); the flag is reported with the
share and is not hidden. Q1 is then applied unchanged to the proxy axis points: the features need h above the pot top and r from the pot centre, which `axis_point_features` hard-codes as R3-1b constants, so each
plant's cloud is translated (base z = p1 of the lowest 2 % -> POT_TOP; xy median of those surfels -> POT_CENTRE) before the call; features are otherwise translation-invariant. No code, parameter or hash changes.
**Adapter (skeleton -> stage-1 seeds).** rachis seeds: axis_kind 2 surfels within **3 mm** of a leaf's 1 mm polyline -> that leaf; blade seeds: 3D-first groups (R3-1b: `second_attempt/patches_filled.npz` `patch_filled`;
others: `merge/groups.npz` `group`) whose median distance to one leaf's polyline <= **8 mm** and to every other leaf's polyline >= **13 mm** (8 + 5) -> that leaf, whole; everything else stage1 = 0.
Bundle = the frozen `r8pc_bundle.npz` layout with `stage1` := adapter seeds, `trace_*` := the skeleton export (leaf numbers = skeleton leaf order, `r7` := 0 placeholder, unused by `Model`); per individual
`t_sheet_mm` from its geometry, `offset_mm` = max(t_sheet, 1.0), `reach_mm` = C6.U_MM (R3-1b value). Frozen order of the code: **R8_pc (`pc_absorb.py`, weights B, p1 >= 0.6, margin >= 0.25, Kaggle CPU, one kernel per run)
then rachis absorption (`pc_rachis.py` consumes R8_pc labels)**; the addendum names the opposite order, but the frozen code supports only this one, so it is used and stated. `pc_absorb.P["cal"/"rep"]` (leaf numbers 1,3,6,8 / 2,4,5,7)
only steer its internal self-test split, not the labels.
**R3-1b gate (first, before any other individual).** Adapter seeds from the simulated R3-1b skeleton; agreement = share of surfels that the frozen R8_pc labelled (> 0) whose new label is identical (unlabelled counts as
disagreement); coverage reported beside it. **Pass >= 0.95.** Fail -> report and stop; 3 / 8 / 5 mm are not tuned.
**Per individual afterwards.** Main route labels (`labels_skeleton_r8pc.<stamp>.npz`) and the distance route side by side: assigned / contested / none, held-out agreement (Wilson 95 %) from the 10 picks, grade, contested
light share; per-leaf light only for grade A / B on the main route. Order n_04 (after the user's re-check of the 17 legacy lines), t2_02, then the rest as confirmed.
Editor: "점 옮기기" = select a leaf, press the button, click the new position (the leaf's point nearest to the old click-surfel of the previous click is not inferred: the LAST point moves, or the point index chosen in the list); logged as `move_point`.
n_04 banner for the legacy lines as the addendum words it.

## Amendment A3 (2026-11-01 02:05, after the first R3-1b gate run; before the rerun)
Attempt 1 (`r8pc/R3-1b/20261101T013915`, kept) numbered the R3-1b seeds per skeleton line (10 labels), while the frozen R8_pc labels use the pipeline leaf numbers of the traces (10 traces -> 8 leaves: 5 x 3, 7 x 2). The
label agreement was therefore 0.0895 (majority remap 0.796) - a construction error of this session, not a property of the adapter. The rerun uses the frozen numbering (`attrs.pipeline_leaf`, polylines of the same leaf
pooled; trace arrays built exactly as `pc_absorb_prep`), unchanged 3 / 8 / 13 mm. The other individuals have one line per leaf, so nothing changes for them. Attempt 1 is reported as `numbering_error`, the rerun as the gate.
Axis step result (A2): all 12 individuals with groups are `axis_share_unusual` (shares 0.014-0.041); e_02, e_03, e_06 have no `merge/groups.npz` -> main route not runnable for them.

## Amendment A4 (2026-11-01 13:30, prompt "next steps after the adapter gate failed"; label `metric_declared_after_adapter_gate`)
The adapter gate (A2/A3) measured similarity to the frozen R8_pc labels, which are themselves imperfect; it stays `failed_gate`, nothing in it is re-tuned. A second, different metric is declared now, after that result:
**R3-1b accuracy against human leaves.** Truth = the user's leaflet labels on the 4 R3-1b training photos, mapped to plant leaf numbers by the user's latest number map (the frozen "vp" numbering), projected onto the
front-visible blade surfels of each photo (the projections and visibility of the stacked-sheet retry, `oversegment.load_base`). "Report half" = human leaves **2, 4, 5, 7** (the same leaf set as R8_pc's own report split).
Accuracy = share of those labelled projections whose route label equals the human number; a surfel the route leaves contested / unlabelled counts as wrong and its share is reported separately; Wilson 95 % CI; per-leaf accuracy;
confusion among 4/5 and 6/7/8 over all leaves. Applied identically to (a) the distance route from the human-trace skeleton (25 mm, 3 mm, 3D-first groups = `patches_filled`, pipeline numbering), (b) the adapter + R8_pc run
`r8pc/R3-1b/20261101T130242` (R8_pc labels and after rachis absorption), (c) the frozen R8_pc labels and frozen R8_pc_rachis labels. R8_pc's 0.833 is its hide-and-recover self-test against its own stage-1 seeds, a different metric; it is quoted, not compared as equal.
No route is changed after this. Distance route = operating route for the other individuals (`distance_route`); held-out agreement + grade computed automatically when 10 picks exist; per-leaf light only for A / B.

## Amendment A5 (2026-11-01 13:40, prompt "check missing leaf lines -> rescore once -> two declared tests"; declared before any run). Labels add `reference_only_scoring`.
**Step 1 finding.** The 10 R3-1b traces cover leaves 1-7 (5 x 3, 7 x 2); **leaf 8 has no line**. Leaf 6 has one (trace leaf6_mvno0afa) although both skeleton routes scored it ~0.02.
**Completion rule (`sim_skeleton_completed`).** Leaf 8's surfels = blade surfels whose front-visible human label (majority over the 4 training photos' projections) is 8. Base = the one nearest the plant axis (vertical
through the pot centre, xy distance); principal direction = first PCA axis of the leaf's surfels, oriented from the base; length = span of the projections on it from the base; tip = the surfel with the largest projection;
medial points at 25 / 50 / 75 % = median xyz of the surfels whose projection lies within +/- 5 % of the length around that fraction. Line = base, m25, m50, m75, tip, with `attrs.pipeline_leaf = 8`. Scoring-side labels are
used only to place this line, as the human traces were. The completed skeleton is saved as a new version (earlier version kept); the byte-equality check of section 5 refers to the earlier version.
**Rescore (once, A4 metric).** (a) distance route and (b) adapter + R8_pc (Kaggle, same adapter 3 / 8 / 13 mm) on the completed skeleton. Old numbers stay as `incomplete_sim_skeleton`. Baseline **B** = (a) report-half accuracy.
**0.942 note.** Frozen R8_pc's stage-1 seeds were propagated from the same four photos' user labels that the scoring projects, so 0.942 is a shared-source upper reference, not a target for skeleton routes.
**T-prox (one run).** Rachis seeds = any surfel within 3 mm of exactly one leaf's line (no axis-label condition; second line > 3 mm); blade seeds = 3D-first groups <= 8 mm / >= 13 mm (as A2); then frozen R8_pc (weights B)
and frozen rachis absorption, unchanged. R8_pc itself still reads `axis_kind` for its node set (frozen code); on the nine that would be the S1 proxy. **Gate: report-half accuracy >= B + 0.05 AND contested share <= the distance route's.**
(The ">= 0.833" in next_steps.md belonged to a different metric and is replaced here, before running.)
**T-multi.** `CLI_PROMPT_P1_SKELETON_MULTICUE_ASSIGN_2026-11-01.md` is not on disk and not in LOG; its weights and margins are unknown, so T-multi is **not run** (reported as `blocked_prompt_missing`); nothing is guessed.
A passing test becomes the operating route elsewhere only after the user confirms; until then the distance route stays.

## Amendment A6 — main axis (skeleton_v1.1), declared 2026-11-01 ~18:45 before any axis number
Source: `handoff/CLI_PROMPT_P1_SKELETON_MAIN_AXIS_CLICK_PLUS_FSPM_FILL_2026-11-01.md` (pasted by the user 18:40). Outputs under `axis/`. Labels: `user_skeleton_axis`, `model_completed`, `conditional_on_unvalidated_scale`, `after_checkpoint_v1`. The photo-click addendum (`..._AXIS_FROM_POT_SURFACE_PHOTO_CLICKS_ADDENDUM_...`, on disk 18:24) is NOT part of this amendment; the schema keeps a `source` slot for it (`observed_photo_click`).
1. Schema `skeleton_v1.1` (loader accepts v1 and v1.1; file name unchanged): `axes[]` = {axis_uid `<plant>-A<NN>` never reused, type main/lateral, parent_axis_uid, attach_s_mm, points[{xyz, source}], fit{...}, segments[{s0_mm, s1_mm, provenance}], nodes[...]}; leaves gain attach_axis_uid, attach_s_mm, node_index, attach_distance_mm. Point sources: `user_click`, `pot_centre_auto`, `leaf_base_anchor`, `fspm_fill`.
2. Root: R3-1b has the rim fit (CC.POT_CENTRE, POT_TOP = 0.066 m) -> `pot_centre_auto` first point. No other plant has a rim fit on disk -> `rim_fit_unavailable`; the user's first axis click is the base on the medium.
3. Fit (`tomato3d/organs/skeleton_axis.py`):
   - anchors = root (w 1.0), user clicks (1.0), each attached-candidate leaf base moved halfway toward its nearest point on the provisional polyline (root + clicks, or a vertical line through the root when there are no clicks) (0.5); ordered by arc position on the provisional polyline;
   - curve = per-coordinate weighted smoothing spline over chord-length parameter (scipy `make_smoothing_spline`, GCV lambda) when >= 5 anchors, else the weighted polyline; sampled every 1 mm;
   - lean rule: direction change <= 15 deg per 10 mm of arc. If violated, lambda x4, at most 6 times; still violated -> flag `lean_limit_exceeded` (no other change). Self-crossing (two curve points > 10 mm apart in arc within 2 mm in space) -> flag;
   - residual per anchor reported.
4. Nodes: leaf base -> nearest curve point = attach_s; attach distance > 12 mm -> `unattached` (no node). Node index = order of attach_s among attached leaves (one node per leaf line).
5. Segment provenance: between consecutive anchors; `observed` only if both ends are user clicks <= 15 mm apart along the curve, else `model_completed`. Bottom segment root -> first node: straight line, `model_completed`, unless an observed segment covers it (no hypocotyl / cotyledon organs in the engine — declared limitation).
6. FSPM fill (engine read-only): expected internode length L(rank) and radius = reference run R, plant p1, class Internode, engine day 71 (= 59.9 DAG; no per-plant DAG is recorded, flag `dag_assumed_59.9`), `C:/fsp_build/engine_dynamic_20261026_01/runs/R/native/working/states.csv` (length_m, width_m/2). Rank of node k = k. If arc length between consecutive nodes (and root -> first node, rank 1) > 1.8 x L(next rank): insert round(gap / L) - 1 hypothesised nodes evenly spaced, flag `fspm_node_hypothesis`, leaf `unobserved`, never counted as observed. Radius per rank, `model_generated`, display / occluder only.
7. Phyllotaxis: leaf azimuth = first 10 mm of the leaf line projected on the plane normal to the local curve tangent, angle from the projected board x axis; divergence between consecutive observed nodes = (az_k+1 - az_k) mod 360, reported raw and folded to [0, 180]; circular mean / SD; compared with 137.5 deg and with B2 protractor values if present. Reported only.
8. Downstream: `skeleton_tree.skeleton_to_plant_tree` uses the fitted main axis as the main tube, nodes at attach positions with s_mm (hypothesised nodes have no leaf, flagged); internode provenance observed / model_completed in attrs; lengths only with labels. Stem display label: surfels <= 3 mm from observed segments only (`stem_display_labels.npz`), blade labels untouched. Growth carry: axis points transformed with the leaf registration, snapped to surfels within 5 mm.
9. R3-1b test (simulated): clicks = medians of frozen stem-like surfels (bundle axis_kind 1) in 10 mm height bins with >= 5 surfels, bottom -> top; leaf lines = the completed simulated skeleton (11 lines, pipeline leaves 1–8). Reported: observed vs model_completed length share, nodes per pipeline leaf (closest-attached line) vs 8, hypothesised nodes, divergence. n_04 and others after the user draws their axis.
No tuning of 1.8x, weights, 12 mm, 15 mm, 15 deg after results; engine laws and R8_pc / T-prox / distance routes unchanged.
- A6 note (written before any number): `make_smoothing_spline` / GCV is replaced by `scipy.interpolate.UnivariateSpline` (cubic, per coordinate) with per-anchor weight w = sqrt(anchor weight) / 1 mm and smoothing s = number of anchors (expected chi-square at 1 mm click noise); the lean-rule escalation multiplies s by 4 (at most 6 times). <= 3 anchors -> weighted polyline. Engine ranks beyond the last internode rank in R day 71 use the last rank's length / radius (flag `rank_beyond_engine`).
- A6 correction (19:20, found by the synthetic unit test after the first R3-1b simulated run; reported with both numbers):
  (a) segment ends are user clicks only. Leaf-base anchors are fit anchors, not segment ends. Before, a leaf anchor between two clicks turned a densely clicked span into `model_completed`, contrary to section 1 of the prompt.
  (b) the straight bottom segment starts at the root point itself, not at the spline's first sample.
  (c) the anchor arc position comes from its own spline parameter, not from a nearest-sample lookup (this made the root s = 10 mm in the first run).
  (d) the round-trip check subtracts the converter's origin (base of the main axis).
  No parameter changed.
