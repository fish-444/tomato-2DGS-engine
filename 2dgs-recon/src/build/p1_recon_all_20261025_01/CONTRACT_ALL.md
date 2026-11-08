# CONTRACT_ALL — R3-1b-style whole-plant reconstruction of all 11 individuals, Kaggle only (written 2026-10-25, before any GPU work)

User 2026-10-25: reconstruct every individual in `토마토2/` (t2_01..t2_05) and `새 폴더/` (n_01..n_06, 2026-10-24 18:15–18:38, all 52 mm,
3024² square frames, no duplicates, no overlap with 토마토2) "with R3-1b", Kaggle only. Defaults taken because the user quoted but did not
answer my two questions (seeds, order); both are reversible.
Rule 0 stands: nothing but listing, EXIF, hashes and uploads of small files runs locally. Original folders are read-only.
Status label on every output: `R3-1b-style`, `unvalidated` unless a human reference exists, `conditional_on_unvalidated_scale`
(the board gives metres; the pot dimension was never measured). Shape only; `organ_split: none`; no trait tables.

## Individuals and order (Stage 2 with D7, `triage_all.json`)
Order by accepted views: t2_01 (63, two cameras), n_04 (52), n_01 (50), t2_03 (44), t2_02 (40), n_06 (36), n_05 (33), n_02 (32),
t2_05 (30), n_03 (27), t2_04 (20). Rule: an individual with fewer than 20 accepted views is reported and not reconstructed; none is
below. Only t2_01, n_04, n_01 reach the declared `reconstructable` gate (≥ 45 views and ≥ 300°); the rest are `marginal`, azimuth
coverage 274–336°, so the reconstructions of n_02 (274°) and n_03 (290°) have an unseen side of 70–86°.

## Pipeline per individual, and every deviation from R3-1b (declared)
R3-1b = masks M1–M5 (stage b) + init I1 (SIFT, fixed board poses) + I2 (mono depth where I1 is sparse) + hull init + 2DGS 1512 px,
7000 iterations, soft-mask alpha loss (M4), R3 densification, no exposure compensation, no depth loss.
| step | here | deviation |
|---|---|---|
| poses | board poses and per-camera intrinsics from Stage 2 (`charuco_a3` verbatim, D7: 6–7 pose points allowed with RMS ≤ 1.5 px at 1512) | D7 (user-approved) |
| masks | **C3 U-Net** (trained on the 4 R3-1 human whole-plant masks, 30 epochs, as `CONTRACT_MASK_METHOD.md`), applied to every accepted view, ∩ visual hull carved from the training views' own C3 masks, soft matte from R3-1's M4 (r 4, eps 1e-3, band 2 px). For t2_02 the C2 masks (SAM 2 propagation from the user's masks, 36 of 40 views) replace C3 where they exist | **D11: M1, M2, M3, M5 are not run.** M1/M5 need a base 2DGS run per individual (double the GPU); M2/M3 were tuned on brown soil and greener leaves. C3 stands in because it passed the dev gate on t2_02 (band-excl IoU 0.873, stem 0.788) without any t2_02 seed. Which of M1/M5/I1/I2 produced R3-1b's stem gain was never separated, so R3-1b's stem gain is not guaranteed here |
| init points | I1: pycolmap SIFT 3024 px, 32,768 features, fixed poses, foreground = the masks, hull init as R3-1b | **D12: I2 (mono depth) is not run** (a third model on every individual; I1 median was 565 points per view on R3, sparse views only) |
| training | `kernel_2dgs_r31.py` unchanged: 1512 px, 7000 iterations, 150k initial, seed 0, R3 densification, soft alpha loss | **D2: per-image exposure compensation ON** (cluttered, unevenly lit background; already declared in CONTRACT_TOMATO2) |
| camera t2_01 | the 55 frames at 26 mm 6048² and the 22 at 52 mm 3024² are two camera models; each view carries its own K; poses are in the one board frame | declared in CONTRACT_TOMATO2 D3 |

## Held-out and reference
Each individual holds out 2 views (one azimuth in [0°, 180°), one in [180°, 360°); low ring preferred, then upper; seed 20261023), never used
to carve the hull, build masks, seed points or train. Their scores against the C3 masks are **secondary auto scores, not accepted/not
validated**. Human references (user's masks) exist only for t2_02 (4 held-out + 6 gap views). Acceptance thresholds and the gap rule are
those of `CONTRACT_R3-1.md`, applied only where a human reference exists; elsewhere the run is `no_human_reference`.

## GPU
≤ 1.5 h T4 per individual; per-individual kernel; one GPU kernel at a time. Before starting the next, the user is asked the weekly T4
usage (Kaggle CLI cannot show it); a ceiling of 25 h/week is assumed until told otherwise. Estimated per individual: masks ≈ 0.15 h
(shared kernel for all 11), SIFT on CPU, 2DGS ≈ 0.28 h → about 0.4 h each, ≈ 4.5 h for the 11 plus ≈ 0.3 h for the shared mask kernel.

## Outputs (per individual)
`build/p1_recon_all_20261025_01/<id>/`: surfels, point cloud, mesh, pot-top plane (z from the board fit, not the R3 pot), `recon_meta.json`
(individual, source folder, session date, views before/after QC, pipeline name `R3-1b-style`, all deviations D7/D11/D12/D2, scale tag,
human-reference status, acceptance status, `organ_split: none`), a disc viewer with banner and a height colour mode (0 = pot-top plane).
Local footprint ≤ 2 GB total.

## Prohibited
Calling any of these accepted without human masks; mixing individuals, dates or camera configurations in one reconstruction; changing any
threshold after seeing results; using the held-out views to choose anything; metric claims without the scale tag; local heavy compute;
touching the originals; deleting files.

## Amendment 1 (2026-10-25 14:5x, after the K3a mask sheets, before any point cloud or training)
- **t2_03 is held back.** The fruiting plant has bright green leaves and green fruit; C3 (trained on the greyer R3-1 plants) masks only part of
  the foliage and none of the fruit, and its hull has 10,365 voxels against 19k–69k for the others. Reconstructing from that mask would
  carve the missing leaves away. t2_03 needs a different mask route (the user's seeds with C2, or a colour-agnostic seed), decided later.
- **t2_01 (two cameras) is handled last**, with its own point-cloud step (the shared SIFT kernel assumes one camera).
- Point clouds and training therefore start with: n_04, n_01, t2_02, n_06, n_05, n_02, t2_05, n_03, t2_04.
- The other nine masks were checked on contact sheets (held-out, training views); n_04, n_01 and n_03 cover leaves and stem well.

## Amendment 2 (2026-10-26, user decision "t2_03, t2_01은 제외")
- **t2_03 and t2_01 are out of this batch** (not reconstructed). Reasons on record: t2_03's masks miss the green leaves and fruit; t2_01 has two
  cameras and needs its own point-cloud step. Neither is deleted or altered; their Stage 2 pose QC and (for t2_03, t2_01) K3a mask outputs stay on disk.
- The batch is therefore the **nine** remaining individuals: n_04, n_01, n_06, n_05, n_02, n_03, t2_04, t2_05, t2_02.
- Queue order unchanged; t2_02 runs last and is the only individual with a human reference.
