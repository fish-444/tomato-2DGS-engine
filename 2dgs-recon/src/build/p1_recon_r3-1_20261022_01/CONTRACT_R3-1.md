# CONTRACT R3-1 — whole-plant reconstruction with better masks, initialisation and training (2026-10-22)

Prompt "R3-1" (user 2026-10-22 17:29). Written before any new mask or training. Run 01 and the v2 folder (R0-R3) are read-only.
Shape-only rule; the session never writes human truth; `organ_split: none`. GPU budget <= 2.3 h T4, one kernel at a time.

## Fresh held-out views (seed 20261023; drawn from the 58 views that were training views in R3; one per azimuth quadrant, ring sequence
drawn without replacement from (low, middle, upper, low); never used to train, to build masks from 3D, or to tune)

| view | quadrant | azimuth | elevation | ring | corners | PnP RMS px |
|---|---|---|---|---|---|---|
| `20261021_060430586_iOS.jpg` | 0-90° | 61.7° | 54.8° | middle | 17 | 5.921 |
| `20261021_060343438_iOS.jpg` | 90-180° | 176.9° | 61.5° | upper | 13 | 4.539 |
| `20261021_060603671_iOS.jpg` | 180-270° | 254.2° | 31.8° | low | 26 | 7.574 |
| `20261021_060528683_iOS.jpg` | 270-360° | 298.2° | 31.0° | low | 14 | 7.688 |

Secondary sets (already looked at): v2 held-out `20261021_060429762_iOS.jpg`, `20261021_060544649_iOS.jpg`, `20261021_060600613_iOS.jpg`, `20261021_060317552_iOS.jpg`; run 01 held-out `20261021_060326166_iOS.jpg`, `20261021_060448335_iOS.jpg`, `20261021_060605333_iOS.jpg`.
Training views: the other 61 accepted views (the 7 old held-out views are training views here).

## Reference masks (Step 1)
The 4 fresh views are scored against masks the USER corrects on a db-backed page (proposal = the A2 whole-plant mask; the user adds by
SAM point or brush, erases, and may mark a 2 px uncertainty band where hairs make the edge ambiguous). Auto-mask scores are reported too.
If the user declines, scoring is against auto masks only and the report says so.

## Mask improvements (Step 2, training views only)
- **M1** SAM 2.1 (the leaflet-tuned v5 checkpoint, image predictor) propagated view-to-view along each ring ordered by azimuth: the previous view's whole-plant mask is lifted through the R3-base model's rendered depth into the next view and used as a box + positive-point prompt; union with A2 inside the projected 3D bounding box of the plant hull dilated by 1 cm (seeds per ring 3).
- **M2** ridge candidates (sato, sigmas 3/6/10 px at 3024) inside the plant region dilated 90 px -> SAM point prompts for stems, petioles, petiolules; added region gated a* <= 2, L >= 15 and thin (26 px opening), touching the plant.
- **M3** soil as its own class: the pot-top polygon (soil plane at the fitted rim height, side 62 mm around the plant centre) is projected per view; a per-capture soil colour model (Gaussian in Lab on pixels inside the polygon and outside every current plant mask) and a plant model (Gaussian on plant pixels) classify the pixels inside the polygon by likelihood ratio; plant wins when log-ratio >= 0.
- **M4** alpha matte from a trimap: inside (mask eroded 2 px) = 1, outside (mask dilated 2 px) = 0, band = guided filter of the hard mask with the photo as guide (radius 4 px, eps 1e-3 at 1512), clipped to [0, 1]; the soft mask replaces the band exclusion in the alpha loss.
- **M5** R3-base rendered into the TRAINING views only: pixels with rendered alpha >= 0.8 that the mask misses and that pass M3's plant test become SAM positive prompts (one round; never on held-out views).
- Contact sheet on 12 training views: A2 -> M1-M5, added (cyan) and removed (red) pixels.

## Initial points (Step 3)
- **I1** pycolmap SIFT 3024 px, 32,768 features, fixed board poses, foreground = Step 2 masks (as A3 of v2).
- **I2** Depth Anything V2 Small (HF depth-anything/Depth-Anything-V2-Small-hf) per training view; per-view affine (scale, shift) fit of 1/depth to the I1 points visible in that view by RANSAC-trimmed least squares; back-projected points added only where I1 has < N points, inside the Step 2 mask, tagged mono_prior (N = 300).

## Training settings (Step 4, chosen without any held-out result)
per-image affine colour compensation (3x3 + offset, learnt), soft-mask alpha loss (M4), depth-normal consistency 0.05, depth loss (L1 on rendered median depth vs I1 points 0.05, vs I2 points 0.01), densification: R3 values and one declared alternative (grow_grad2d 1.5e-4, refine until 6000); iterations 7000; resolution 1512; 3024 only as R3-1d.

## Runs (Step 5), one change at a time
| run | contents |
|---|---|
| R3-base | R3 unchanged on the new split (baseline) |
| R3-1a | R3-base + masks M1-M4 |
| R3-1b | R3-1a + M5 (one 3D->2D round, training views only) + init I1/I2 |
| R3-1c | R3-1b + training settings (Step 4) |
| R3-1d | R3-1c at 3024 px, only if >= 0.6 h of budget remains |

Reported per run on the fresh views (human masks primary, auto secondary) and on the two old sets (secondary): raw IoU, band-excluded
IoU (2 px), edge F at 2 px, stem coverage, held-out minus training gap, depth consistency (check only).

## Acceptance (Step 6, unchanged from v2)
On the fresh views with HUMAN masks: medians band-excluded IoU >= 0.85, raw IoU >= 0.75, edge F >= 0.9,
stem coverage >= 0.6, gap >= -0.03; and no single view more than 0.05 below a median threshold. The first run in
run order that passes is accepted; otherwise the best run is kept as **best unaccepted** with the largest remaining cause. No further
tuning on these views; no threshold change after seeing results.

## Outputs (Step 7)
Surfels/discs, mesh, point cloud, `recon_meta.json` (run, acceptance, `organ_split: none`), a disc viewer with the banner
(accepted / best unaccepted); a LOG NOTE when the organ-split input may switch to R3-1.

## Amendment 1 (2026-10-22 19:40, before any R3-1 training; R3-base is training)
- Run table: **R3-1a = R3-base + M2 + M3 + M4**; **R3-1b = R3-1a + M1 + M5 + I1/I2**. M1 (view-to-view propagation) lifts a mask
  through R3-base's rendered depth, so it needs R3-base first, like M5; both go into R3-1b. R3-1c/d unchanged.
- R3-base's training-view renders come from a render-only job (the trained surfels re-rendered on the T4, about 3 min), counted in
  the GPU budget.
- Scoring on the fresh views against the HUMAN masks: raw IoU = |render ∩ human| / |render ∪ human| with the user's uncertainty band
  removed from both; band-excluded IoU additionally removes the 2 px contract band around the human mask; edge F at 2 px on the
  human mask; stem coverage = share of the A2 stem-added pixels (auto label 2) that lie inside the human mask and are covered by
  the render (alpha >= 0.5); gap = median over the 4 fresh views minus the median over the training views (auto masks, as the
  kernel reports). Auto-mask scores: the same formulas on the auto mask of the stage used for that run.
- Resolution of M1's SAM prompt: box + up to 6 positive points sampled at the lifted mask's distance-transform maxima.
