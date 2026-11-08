# CONTRACT - P1 all other individuals: photo preprocessing -> 3D-first leaflet segmentation -> flag only what fails (declared 2026-10-27, before any number)

Prompt: `CLI PROMPT — P1: All other individuals — photo preprocessing → 3D-first leaflet segmentation → flag only what fails (2026-10-27)`
plus the user's instruction "병렬로 진행가능하도록 서로 겹치치 않는 환경을 만들어 진행" (run in parallel, in environments that do not overlap).
Single instance: no other instance of this task exists (LOG.md and `build/` checked 2026-10-27; the R3-1b 3D-first attempts are finished and live in other folders).
Every output: `gate_failed_viewing_only` (until an individual is scored after user grouping), `R3-1b-style_unaccepted` (R3-1b: `R3-1b_unaccepted`),
`conditional_on_unvalidated_scale`. Read-only on every existing folder; ports 8830, 8833, 8834 and their click files are never touched.

## 1. Parallel design: non-overlapping cells and lanes
- **Cell** = one individual = one folder `build/p1_all_3dfirst_20261027_01/<id>/`. A worker writes only inside its own cell (`work/`, `logs/`, `state.json`, outputs) and
  reads only shared inputs read-only. No cell reads another cell. Each cell has its own `RUNNING.lock` (pid, start time) and `state.json` (done markers per step, so a
  re-run resumes and never repeats or overwrites a finished step; a finished step's files are never deleted).
- **Lanes** (independent resources, so they overlap in time but never in files):
  - Lane 0 local CPU, light: inputs, pose QC, photo preparation (HEIC -> undistorted 3024 jpg + plant mask + px/mm).
  - Lane K Kaggle (off the PC): per individual its own datasets `dolgfish/tomato-3dfirst-job-<id>` and its own kernel `dolgfish/tomato-3dfirst-cands-<id>`; kernels run one at a time
    (weekly GPU quota, same rule as the earlier batches); SAM-auto (nested) + H2 on every posed photo, one batch per individual, logged.
  - Lane R Roboflow API (network only): per-individual cache `<id>/rf/`, throttled, resumable; the key is read from `api.txt` into memory only, never printed, logged or written.
  - Lane G local CPU geometry (Steps 2-3 need only surfels, no photo candidates, so they run now in parallel with lanes K and R).
  - Lane V local CPU votes/merge/fill/flags (Steps 4-7) after the candidates of that individual exist.
- **Scheduler** `tomato3d/organs/all3d_run.py`: starts each (individual, step) as its own detached OS process (not a thread) with a per-process memory budget; before every
  start it reads free RAM; **if free RAM < 4 GB it starts nothing, logs `PAUSED_LOW_RAM` with the figure, and re-checks every 60 s** (the PC is shared with GroIMP and other
  sessions). Concurrency = min(4, floor((free RAM - 4 GB) / budget)); budgets: Lane 0 1.0 GB, Lane G 2.5 GB, Lane V 3.0 GB. It never kills another session's process.
- **Global lock** `RUNNING.lock` in the build folder (scheduler pid), removed when the scheduler ends.
- The 3D page (port **8835**) reads cell outputs only and writes only its own append-only click files under `<id>/clicks/` (the 8833 page code is copied, not edited).

## 2. Individuals and order
n_04 -> t2_02 -> n_01 -> t2_05 (tray plants first), then n_06, n_05, t2_04, n_02, then n_03 (class C: stop after Step 3 if > 50 % of surfels end up flagged, and say so),
then R3-1b once, identical pipeline, **reference-quality measurement only**: its score is reported, nothing is tuned on it. n_02 keeps the tag `heights_unusable_rim_fit`.
Lanes run all cells concurrently under the RAM rule; "order" is the priority when resources are short.

## 3. Inputs (path, modification time)
| path | mtime |
|---|---|
| `build/p1_recon_all_20261025_01/outputs` | 2026-10-26 17:14 |
| `build/p1_recon_all_20261025_01/k3a` | 2026-10-25 19:05 |
| `build/p1_recon_all_20261025_01/stage2_new` | 2026-10-25 15:14 |
| `build/p1_recon_tomato2_20261023_01/stage2_d7` | 2026-10-24 00:07 |
| `build/p1_plant_model_20261026_01/rim` | 2026-10-26 22:37 |
| `build/p1_stacked_sheets_retry_20261026_01/oversegment/second_attempt/result.json` | 2026-10-27 11:39 |
| `build/p1_all_compound_20261026_01/readiness.csv` | 2026-10-26 16:35 |
| `build/p1_stem_run2_20261026_01` | 2026-10-27 11:51 |
Code reused unchanged (imported, not copied; hashes at declaration time):
| file | sha256[:16] | mtime |
|---|---|---|
| `tomato3d/organs/sheet_retry.py` (`find_peaks`, `local_sheets`, `edges_3mm`, `edge_ok`, `patches`) | 80a1042e0202804d | 2026-10-26 22:53 |
| `tomato3d/organs/oversegment.py` (`split_patches`, `adjacency`, `B4`) | b7e763a43f6baece | 2026-10-26 23:07 |
| `tomato3d/organs/oversegment_second.py` (`consensus_merge`, `fill`, `pair_counts`) | d68198aac064cdd2 | 2026-10-27 11:38 |
| `tomato3d/organs/group_page.py` (8833 page, copied for 8835, not edited) | 9e9ff0c48de86b96 | 2026-10-26 23:10 |
| `tomato3d/organs/all_compound.py`, `all_detect_job.py`, `tomato3d/recon/kernel_all_detect.py` (photo prep, candidate job) | 1ed72610f3a791fa / 7ddb335dd496fe63 / 81cbbe5d607e81d8 | 2026-10-26 |
Functions of those files that read R3-1b paths or user labels (`load_base`, `votes`, `vote_views`, `leaflet_*`, `purity_*`) are NOT used for the other individuals; the vote step is re-implemented
generically in `tomato3d/organs/all3d_first.py` with the same rules (same thresholds, same logic, I/O generalised), and the new code's hash is recorded in each cell's `summary.json`.

## 4. Substitutes the other individuals need (declared now; each exists because an input of the R3-1b run does not exist for them)
- **S1 blade proxy.** The nine have binary C3/C2 masks only: no stem class, no Q1 axis labels. Per surfel, PCA of its 4 mm neighbourhood (eigenvalues l1 >= l2 >= l3): blade if
  >= 8 neighbours and l2 / l1 >= 0.25 (a plate has l2 ~ l1, a 3-4 mm stem/petiole tube has l2 / l1 ~ 0.1-0.15), else `axis_like` (stem, petiole, rachis, thin edge). `axis_like` surfels
  enter no sheet, patch or group, are drawn dark in the page and are not counted as flagged. R3-1b is run with the same proxy (identical pipeline); its overlap with Q1 blade is reported as a diagnostic.
- **S2 leaflet size prior** (the only R3-1b human-derived quantity used elsewhere; it is a size range, not a label): areas in mm^2 of the 135 leaflets the user labelled on the 4 R3-1b
  training views (px -> mm^2 through each photo's depth): p5 = 20.1, p95 = 182.1, median width 7.7 mm (`build/p1_extra_candidates_20261026_01/testA_R3-1b/metrics.json`); p75 and median
  are computed from the same masks and stored in each cell's `summary.json`. Converted per individual to surfel counts with its own tangent-plane density rho (median over blade surfels of
  [surfels within 4 mm in the plane and +/- 1.5 t_sheet along the normal] / (pi 4^2) per mm^2): `small_n` = p5 x rho, 4a `p75`/`med` = p75/median area x rho, merge size cap = 1.2 x p95 x rho.
  Candidate size range in a photo = [0.5 x p5, 1.5 x p95] mm^2 through that photo's px/mm.
- **S3 voting photos** = every pose-QC'd accepted photo of the individual except its `held_out` label view (kept blind for a later scoring); R3-1b: the 64-photo set of attempt 1 minus the held-out photo.
- **S4 patch visibility** in a photo = splat z-buffer visibility (res 1512, tolerance 1.5 mm, radius 1.5 sigma; the frozen rule); "patch >= 70 % visible" = >= 70 % of its surfels visible.
- **S5 Step 7 (rachis diffusion)**: none of the nine has rachis traces, so it is skipped for them; the prompt file `CLI_PROMPT_P1_LOCAL_LEAF_DIFFUSION_GREY_GAPS_2026-10-27.md` is not in `handoff/`,
  so it is also not run for R3-1b (skip reported, nothing reinterpreted).
- **S6 pose QC**: accepted views of Stage 2 whose per-view PnP rms <= 3.0 px at 3024 px (= the D7 limit 1.5 px at 1512) and whose photo decodes; dropped photos are listed in `pose_dropped_views`.

## 5. Steps (production settings; declared, not derived from R3-1b scores)
1. **Photo preprocessing.** Plant mask = `k3a/<id>/masks` (C3; C2 for t2_02). If the plant's bounding box area is < 10 % of the frame: crop the bbox + 10 % margin, upsample so the long side
   is >= 1500 px (never below the crop's own size), run the detectors on the crop and map back to the 1536 grid. Candidates for **all** posed photos: SAM 2.1 tiny v5 automatic masks with nested outputs kept
   (whole / part / sub-part: only exact duplicates, IoU >= 0.9, are dropped), H2 (M1 + SAM v5), and, as a supplement lane, Roboflow v4 boxes (cache per photo, tiled when >= 300 predictions, SAM-drawn on Kaggle).
   Kept = >= 70 % inside the plant mask (dilated 3 px) and size in the S2 range. SAM-auto and H2 are the mandatory layers; the Roboflow layer is added when its lane has finished for that individual and
   `layers_used` in `summary.json` says which layers entered the votes. No human input.
2. **Sheets.** `t_sheet` per individual = FWHM of the normal-offset peak in single-sheet columns (no second peak within 10 mm), median over those columns (second-layer test method), reported next to R3-1b's
   1.529 mm. Sheet detection and graph = `sheet_retry` with that `t_sheet` (offset limit max(t_sheet, 1.0 mm)).
3. **Over-segmentation** = the 4a rule unchanged. Report patch count, planarity distribution (l3 / l1 and l2 / l1 per patch), share of patches spanning more than one sheet index (flagged `multi_sheet_patch`).
4. **SAM votes** = the 4b rules unchanged (+1 `same` when both patches are >= 70 % inside one candidate mask; min 30 px per patch) plus **fragment repair** (one patch covered >= 30 % by each of two
   adjacent masks marks those two masks as one leaflet in that photo). A `different` vote counts only as a **veto** when: both patches >= 70 % visible; the largest leaflet-sized mask for each still separates them
   (the two best masks overlap <= 1 %); the boundary is not explained by occlusion (depth step at the shared image boundary <= 1.5 x t_sheet); the two projected patches share an image boundary (touch within 2 px).
5. **Conservative merge.** Merge two groups only if >= 3 voting views, vote-weighted mean edge score >= 0.8, no cross-pair has a veto in >= 2 views, merged size <= 1.2 x p95 leaflet size (S2).
   Average linkage (`consensus_merge`), recomputed after each merge, pair support rule as in the second attempt. Under-merge is the intended failure.
6. **Fill** = the second attempt's sheet-constrained fill (4 mm, same sheet index, normal <= 35 deg, majority of the 10 nearest, <= 5 passes) + margin rule: the best group must have >= 2 more of the
   neighbours than the second, otherwise the surfel stays unassigned and is flagged `contested`.
7. **Rachis diffusion**: only where traces exist (S5: none).

## 6. Flags and confidence
Every blade surfel ends `confident` (in a group) or flagged: `unassigned` (grey), `contested` (white), `multi_sheet_patch` (red outline), `low_vote` (dotted: group or edge with < 3 voting views),
`sam_conflict` (orange outline + photo crops: inside a merged group >= 1 pair has veto votes in some views), `oversize` (magenta: group > p95 leaflet size), `low_visibility_region` (hatched: visible in < 3 posed photos),
`pose_dropped_views` (list). Group confidence: `high` >= 5 voting views, mean score >= 0.9, no flags; `medium` passes the merge rule with no flags; `low` any flag.
Estimated user minutes per individual = (groups to assign to compound leaves + flagged items to resolve) x 3 s.

## 7. Outputs per cell (`<id>/`)
`groups.npz` (surfel -> group id, group confidence, surfel flag, axis_like), `flags.csv` (location, flag type, size, photo-crop path), `summary.json` (shares high/medium/low/flagged, group count, `t_sheet`, photos used and
dropped, `layers_used`, estimated user minutes, code hashes, resource log), `state.json`, `logs/`. One 3D page for all individuals on port **8835** (individual selector, groups by id, flags as in section 6,
traces as yellow lines where they exist, click groups (Shift for several) + leaf-number button to build compound leaves, flag side list with photo crops: accept as is / split here / merge with selected;
append-only per individual, provenance `user_3d_group_click`). Report `handoff/P1_ALL_3DFIRST_2026-10-27.md` + `_KO`, LOG entry, flow row, flow check output in the report.

## 8. Rules
No parameter change after seeing any individual's result; one labelled second attempt per individual, reason written first, reported as `tuned`. No R3-1b human label is used for any other individual
(S2 is a size range). R3-1b's score is reported, never used to tune. 8830, 8833, 8834 and their click files untouched. If an individual cannot finish (n_03), the report says why.

## Amendment A1 (2026-10-27 14:30, before any cell step ran; implementation details the prompt left open)
- SAM automatic generator: grid 32 x 32 (the grid used for small plants in the extra-candidates test), pred_iou 0.7, stability 0.9, default box NMS, min region 50 px; nested outputs kept (only IoU >= 0.9
  duplicates dropped, in arrival order H2, SAM-auto, RF).
- Candidate files: pass 1 = H2 + SAM-auto (`<stem>_cands.json`), pass 2 = Roboflow drawing (`<stem>_rf.json`) after lane R has cached boxes; both on the 1536 grid of the full frame.
- Roboflow lane R runs on the same crop + upsample image the kernel uses; the response cache holds no key; the lane is capped at 3 API streams.
- Scheduler lanes: `main` (inputs -> prep -> geometry, one step at a time per cell) and `rf` (own state file `state_rf.json`, own lock `RUNNING_rf.lock`) run side by side; the global `size_prior` step runs once before any geometry.
- Lane K driver `tomato3d/organs/all3d_kaggle.py` waits for each cell's `prep` marker and `size_prior.json`, then one kernel per individual (own datasets, own slug).
- Resource state at declaration: free RAM 0.9-1.5 GB (Claude, Chrome, OneDrive and a co2 search job; no GroIMP run seen). By the declared rule every local worker is paused; lanes K and R use almost no PC memory but depend on prep.

## Amendment A2 (2026-10-27 14:28, user instruction; overrides section 1 and A1 where they differ; before any cell step ran)
- **One local worker at a time** (was up to 4). The Kaggle lane (K) and the Roboflow lane (R) are not local workers and are not counted.
- **A local worker starts only if free RAM >= 7.0 GB** (2.5 GB worker + 4 GB reserved for the engine JVM / GroIMP runs + margin), checked by the scheduler before every local start, not derived from the worker
  count. If an engine JVM is running and free RAM is below 7 GB the scheduler waits (the same condition). The 4 GB floor of section 1 is replaced by this rule.
- **Every wait is written to `scheduler.log`**: one line per check (60 s) with the reason (`WAIT_LOW_RAM` or `WAIT_LOCAL_BUSY`), the free RAM, whether an engine JVM is present, the number of local workers and the queue.
- Lanes K and R continue; lane R's cached-box worker is the only non-gated process and stays below 0.5 GB. Lane R and K still depend on `prep` (a local step), so they start after prep.

## Amendment A3 (2026-10-27 16:15, user Addendum 2 "external compute only"; overrides section 1, A1 lanes and A2; before any cell step ran, no number seen)
Named A3 because A2 is already used above (the addendum asks for "a new Amendment A2"; the content is the addendum's). Thresholds, seeds, gates and the substitutes S1-S6 are unchanged.
- **Local worker count 0.** Locally only: copy, zip, hash, upload, download, state / LOG / status writes, small unit tests, serving the finished 8835 page; session RAM <= 1 GB; no full surfel set loaded locally;
  no java / GroIMP; nothing under `build/engine_*`. `all3d_run.py` (local scheduler) now refuses to start any step; `all3d_first.py` refuses size_prior / inputs / prep / geometry unless it runs inside a
  Kaggle kernel (`KAGGLE_KERNEL_RUN_TYPE` set). The local scheduler and the old lane-K driver (`all3d_kaggle.py`) were stopped at 16:1x (STOP / STOP_K); nothing local had run.
- **Where each step runs** (driver `tomato3d/organs/all3d_ext.py`, kernel source `tomato3d/recon/kernel_3dfirst_ext.py`, one kernel per individual and stage, own slug):
  `size_prior` = Kaggle CPU kernel `tomato-3dfirst-sizeprior` (input: private dataset of the R3-1b files the step reads); `pre` = Kaggle CPU kernel `tomato-3dfirst-pre-<id>` running `inputs`, `prep`
  (undistortion from the raw HEIC files uploaded in the individual's private dataset) and `geometry` (S1, sheets, sheet graph, 4a) in that order; `cands` = Kaggle GPU kernel `tomato-3dfirst-cands-<id>`
  (H2 + SAM-auto, unchanged `all3d_cands_job.py`) reading the `pre` kernel's output; 4c merge / fill / flags and the 8835 bundle = Kaggle CPU kernels (written next, same rule).
- **Packaging.** Code = private dataset `p1-all-code-<16 hex of this CONTRACT.md's sha256>` (tomato3d, tools, tests, config, CONTRACT.md). Each individual = private dataset `tomato-3dfirst-in-<id>`
  (surfels ply, poses, split, plant masks, views_chosen, rim file, raw HEIC of the accepted photos) laid out under `root/` with the same relative paths as on this PC, so the code runs unchanged in a mirror
  root on Kaggle (photo roots redirected to `root/heic`). Never public. Each kernel first runs `tests/test_all3d_first.py`; on failure it stops and reports. Each step keeps its done marker; a finished
  stage (local marker `<id>/ext/<stage>.done` with the kernel version) is never re-run or overwritten.
- **Downloads = results only**: `inputs.json`, `state.json`, `detect_input.json`, `geometry.json`, `geometry.npz` (per-surfel patch table), logs, and later vote tables / groups / flags / page bundle.
  Raw surfels and undistorted photos are not downloaded. One exception, declared here: the Roboflow lane needs the detector input image of each photo, so the `pre` kernel also writes it
  (`rf_inputs/<stem>.jpg`, the crop + upsample of A1, JPEG q95) and only those small images are downloaded; the API call runs locally with the key in memory (<= 3 streams), the response cache (no key) goes
  back up as a private dataset for the pass-2 kernel. This re-encodes the detector image once (q95) compared with A1's in-memory crop. No key is put in any dataset, kernel source or log (no Kaggle Secret is
  needed because no kernel calls Roboflow).
- **Quota and concurrency.** GPU only for `cands` (H2, SAM-auto). Remaining GPU weekly quota and the concurrent-session limit are read before the first GPU launch and written in `STATUS.md`; at most 1 GPU and
  2 CPU kernels at once; the first refusal or quota error stops all launching (queued, no retry loop). A GPU step that would exceed the remaining quota is put at the top of `STATUS.md` with the options
  (fewer views per individual, lower SAM-auto sampling, later individuals deferred) for the user.
- **Failures.** A failed kernel is retried once; a second failure marks the individual `failed_external` and the driver moves on. No local fallback: if the external route is blocked, `STATUS.md` says
  "needs user approval for local fallback" and the driver waits.
- **Orchestrator.** `orchestrator/jobs.yaml`: PA-g and PA-prep get `command: null` (disabled for this task: they were local workers); PA-k points to `all3d_ext.py`; PA-rf is part of that driver (no separate
  local scheduler). The orchestrator reads jobs.yaml only at start, so it is restarted once.
- **Status line** after every kernel start / finish and every individual: individual, stage, kernel version, done / running / queued / failed, remaining GPU quota, local free RAM (`STATUS.md` + `status.log`).

## Amendment A4 (2026-10-28, user chat answer "a" to the geometry question; before any merge number was seen)
- **Frozen settings kept** (choice a): the S1 blade proxy, the sheet graph, the 4a split and every threshold of sections 4-6 stay as declared. The geometry result of all nine individuals (blade proxy 96-99 %, sheet graph 1-5 patches, 25-48 % of blade surfels unassigned after 4a, 48-80 % of patches spanning more than one sheet index) is a known limitation and goes into the report; no `tuned` second attempt is declared now.
- **Steps 4b-6 are implemented** in `tomato3d/organs/all3d_merge.py` and run as Kaggle CPU kernel stage `merge` per individual (`tomato-3dfirst-merge-<id>`, inputs: the individual's `pre`, `cands`, `candsrf` kernel outputs + its input dataset). Constants are the ones of sections 5-6 (`P` in the file): 70 % inside / 30 px, fragment repair at 30 % coverage, veto conditions (both patches >= 70 % visible, overlap <= 1 %, depth step <= 1.5 t_sheet, touch <= 2 px), merge >= 3 voting views, mean >= 0.8, veto in >= 2 views blocks, size <= 1.2 x p95, fill 4 mm / 10 neighbours / 35 deg / 5 passes with margin >= 2.
- Interpretations I had to choose (declared before any result): "voting views" of a group pair = distinct photos with a vote on any of its pairs; group evidence for low_vote / high confidence = the same count over all pairs touching the group; a singleton group's score is 1.0 when it has no internal pair; flagged regions = connected components (4 mm) of unassigned or contested surfels; estimated minutes = (groups + low-confidence groups + flagged regions) x 3 s. Photo crops for `sam_conflict` are not produced in this stage (reported as such).
- The first `merge` kernel is run on n_04 alone as a dry run of the code (a crash is a code defect, not a result); the settings are not changed after seeing its numbers. The other eight follow with the same code.

## Amendment A5 (2026-10-31, user: "16개체 전부 같은 엽축배정 사이트 표시")
- The 3D-first run and the grouping page (8835) are extended from the nine to **16 individuals**: the seven plants photographed on 2026-10-28 (`e_01..e_07`, build `p1_recon_d1028_20261028_01`, results in the nine's layout under `p1_recon_all_20261025_01`). Same code, same frozen settings (S1-S6, sections 4-6), same external-compute rule (A3); order e_05, e_01, e_04, e_07, e_06, e_03, e_02 (e_02 after its training finishes).
- Differences declared: no label held-out view exists for these plants (`views_chosen.json` absent), so every pose-QC'd photo votes; no human size-prior input beyond the shared `size_prior.json`; HEIC for `e_*` is indexed from the Downloads folders only (IMG numbers collide with older individuals). Pose-QC counts are the Stage-2 accepted views (e_02 has only 20).
- `all_compound.POSES` / `E_IDS`, `all3d_first.ORDER`, `all3d_page.ORDER` and the driver's `individual_files` were extended; the 8835 page lists an individual as soon as its `merge/` output exists.
