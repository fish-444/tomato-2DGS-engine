# CONTRACT p1_recon_d1028_20261028_01 - R3-1b-style reconstruction of the plants photographed on 2026-10-28 (written 2026-10-28, before any kernel ran)

User 2026-10-28: "여기서 날짜 같은거만 모아서 복원" for the seven Google-Photos download folders `<DOWNLOADS>\Photos-1-001`, `(1)` ... `(6)`.
Rule 0 (user, 2026-10-27/28): heavy compute on Kaggle only; locally only listing, EXIF, hashing, copying and uploading. Originals are read-only (copied, never altered or deleted).
Labels on every output: `R3-1b-style`, `no_human_reference`, `conditional_on_unvalidated_scale`, `organ_split: none`; nothing accepted.

## Inputs (read from the files, not assumed)
583 HEIC files, all `iPhone 16`, 3024 x 3024, 52 mm equivalent, capture date 2026-10-28 15:15:18 - 16:06:40 (no other date exists in the folders, so the date alone does not split them).
Folder / IMG ranges: (6) 0093-0173, (5) 0174-0250, (4) 0251-0338, (3) 0340-0395 + 0425-0449 (0396-0424 absent), (2) 0450-0531, (1) 0533-0623, 001 0624-0714 (image numbers absent inside 0093-0714: 220, 339, 346, 396-424, 435, 532, 581, 601, 617, 651, 671, 39 in all).

## Grouping into plants (declared before any result; EXIF time gaps + contact sheet)
Consecutive shots are 2-4 s apart; a gap of 184-254 s marks a change of plant, and each plant begins with a top-down shot. Gaps of 108 s (0395 -> 0425, 30 images absent) and the 2 + n opening shots of a plant are inside one plant.
| id | IMG range | files | note |
|---|---|---|---|
| e_01 | 0093-0173 | 81 | |
| e_02 | 0174-0250 | 76 | 0174-0175 then a 21 s pause, same plant |
| e_03 | 0251-0338 | 88 | |
| e_04 | 0340-0395 + 0425-0449 | 79 | the two parts show the same plant (contact sheet) |
| e_05 | 0450-0531 | 82 | |
| e_06 | 0533-0623 | 88 | |
| e_07 | 0624-0659 | 35 | top view + sweep; fewer views |
| e_08 | 0660-0714 | 54 | side views only, a different-looking plant than 0659 (contact sheet); a 202 s gap with no image number missing |
The e_07 / e_08 split is the least certain one (visual judgement); if Stage 2 shows e_07 + e_08 are one plant they are merged only by a new amendment.

## Steps (same code and constants as the n_01..n_06 batch, `build/p1_recon_all_20261025_01/CONTRACT_ALL.md`, deviations D7 / D11 / D12 / D2 unchanged)
1. Stage 2 (Kaggle CPU, `tomato3d/recon/kernel_d1028_stage2.py`, a copy of `kernel_new_stage2.py` with the individual list from the census): board detection, per-camera calibration, per-view pose QC (D7), coverage, triage. Output reported before anything else runs.
2. Only individuals with `reconstructable` or `marginal` triage and >= 20 accepted views continue (rule of CONTRACT_ALL). Masks (C3 U-Net as before, GPU), I1 points (CPU), 2DGS training (T4, one individual per push), outputs (CPU). Each stage is adapted from the `kernel_all_*` kernel only by its individual list and input datasets; the adaptation is logged.
3. Held-out: 2 views per individual as in CONTRACT_ALL (seed 20261023); scores are secondary auto scores.

## Prohibited
Calling any output accepted; changing any threshold after seeing results; mixing individuals; local heavy compute; touching the originals; deleting files.

## Amendment 1 (2026-10-28, user: "7개체 아니야?"; before any kernel ran, the first upload was stopped before it finished and nothing was created on Kaggle)
- **Seven plants, not eight.** The user states there are seven. e_07 = IMG_0624-0714 (89 files); e_08 is dropped (merged into e_07). My split of 0659 | 0660 rested on a 202 s pause and a visibly different-looking plant in the contact sheet; the user knows the plants, so their count stands. Stage 2 will show whether the 89 views form one consistent set (azimuth coverage, pose QC); if it does not, that is reported, not silently fixed.
- Individuals and counts: e_01 81, e_02 76, e_03 88, e_04 79, e_05 82, e_06 88, e_07 89 (583 files). `census.csv` in this folder is the corrected one.
