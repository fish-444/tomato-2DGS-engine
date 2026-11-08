"""R3-1, Step 0: the fresh held-out split (seed 20261023) and CONTRACT_R3-1.md, written before any new mask or training.

Rule: the pool is the 58 views that were TRAINING views in R3 (65 accepted minus the 4 held-out of R0-R3 minus the 3 of run 01).
Rings as in the v2 contract (top > 75 deg, upper > 58, middle 45-58, low < 45). One view per azimuth quadrant; the ring sequence is
drawn without replacement from (low, middle, upper, low) with the seeded generator; within a (quadrant, ring) cell uniform at random.

usage: python -m tomato3d.recon.r31_contract --poses <charuco dir> --v2 <wholeplant run dir> --out <r3-1 run dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tomato3d.recon.wholeplant_contract import A4, A5, QUADRANTS, RING_SEQUENCE, TRAIN, ring

SEED = 20261023
GPU_BUDGET_H = 2.3
RUNS = (("R3-base", "R3 unchanged on the new split (baseline)"),
        ("R3-1a", "R3-base + masks M1-M4"),
        ("R3-1b", "R3-1a + M5 (one 3D->2D round, training views only) + init I1/I2"),
        ("R3-1c", "R3-1b + training settings (Step 4)"),
        ("R3-1d", "R3-1c at 3024 px, only if >= 0.6 h of budget remains"))
# ---- declared methods and parameters (Steps 2-4)
M = dict(
    M1=dict(method="SAM 2.1 (the leaflet-tuned v5 checkpoint, image predictor) propagated view-to-view along each ring ordered by azimuth: "
                   "the previous view's whole-plant mask is lifted through the R3-base model's rendered depth into the next view and used as a "
                   "box + positive-point prompt; union with A2 inside the projected 3D bounding box of the plant hull dilated by 1 cm",
            seeds_per_ring=3, prompt_points=6),
    M2=dict(method="ridge candidates (sato, sigmas 3/6/10 px at 3024) inside the plant region dilated 90 px -> SAM point prompts for stems, "
                   "petioles, petiolules; added region gated a* <= 2, L >= 15 and thin (26 px opening), touching the plant",
            sigmas=(3, 6, 10), n_seed=24, thin_px=26, a_max=2.0, l_min=15.0),
    M3=dict(method="soil as its own class: the pot-top polygon (soil plane at the fitted rim height, side 62 mm around the plant centre) is "
                   "projected per view; a per-capture soil colour model (Gaussian in Lab on pixels inside the polygon and outside every "
                   "current plant mask) and a plant model (Gaussian on plant pixels) classify the pixels inside the polygon by likelihood "
                   "ratio; plant wins when log-ratio >= 0",
            rim_z_m=0.066, rim_side_m=0.062, min_soil_px=2000),
    M4=dict(method="alpha matte from a trimap: inside (mask eroded 2 px) = 1, outside (mask dilated 2 px) = 0, band = guided filter of the "
                   "hard mask with the photo as guide (radius 4 px, eps 1e-3 at 1512), clipped to [0, 1]; the soft mask replaces the band "
                   "exclusion in the alpha loss", band_px=2, radius_px=4, eps=1e-3),
    M5=dict(method="R3-base rendered into the TRAINING views only: pixels with rendered alpha >= 0.8 that the mask misses and that pass "
                   "M3's plant test become SAM positive prompts (one round; never on held-out views)", alpha_min=0.8),
    I1=dict(method="pycolmap SIFT 3024 px, 32,768 features, fixed board poses, foreground = Step 2 masks (as A3 of v2)"),
    I2=dict(method="Depth Anything V2 Small (HF depth-anything/Depth-Anything-V2-Small-hf) per training view; per-view affine (scale, shift) "
                   "fit of 1/depth to the I1 points visible in that view by RANSAC-trimmed least squares; back-projected points added only "
                   "where I1 has < N points, inside the Step 2 mask, tagged mono_prior", n_min=300, rel_residual_max=0.08),
    T=dict(method="per-image affine colour compensation (3x3 + offset, learnt), soft-mask alpha loss (M4), depth-normal consistency 0.05, "
                  "depth loss (L1 on rendered median depth vs I1 points 0.05, vs I2 points 0.01), densification: R3 values and one declared "
                  "alternative (grow_grad2d 1.5e-4, refine until 6000); iterations 7000; resolution 1512; 3024 only as R3-1d"),
)
BAND_PX = 2


def pick(rows: list[dict], pool: set[str]) -> list[dict]:
    rng = np.random.default_rng(SEED); cand_all = [r for r in rows if r["file"] in pool]
    seq = list(RING_SEQUENCE); rng.shuffle(seq); out, used = [], set()
    for (lo, hi), want in zip(QUADRANTS, seq):
        cand = [r for r in cand_all if lo <= float(r["azimuth_deg"]) < hi and r["file"] not in used]
        pref = [r for r in cand if ring(float(r["elevation_deg"])) == want] or cand
        if not pref:
            continue
        r = pref[int(rng.integers(len(pref)))]; used.add(r["file"]); out.append(dict(r, ring=ring(float(r["elevation_deg"])), quadrant=f"{lo}-{hi}", wanted_ring=want))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--v2", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src, v2, out = Path(a.poses), Path(a.v2), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(src / "qc.csv")) if r["accepted"] == "1"]
    c2 = json.load(open(v2 / "contract.json")); old_new, old_01 = c2["heldout_new"], c2["heldout_run01"]
    pool = {r["file"] for r in rows} - set(old_new) - set(old_01); assert len(pool) == 58, len(pool)
    held = pick(rows, pool); names = [h["file"] for h in held]
    train = [r["file"] for r in rows if r["file"] not in names]            # the 7 old held-out views are training views here
    con = dict(date="2026-10-22", seed=SEED, pool=sorted(pool), heldout_fresh=names, heldout_v2=old_new, heldout_run01=old_01, train=train,
               heldout_detail=[{k: h[k] for k in ("file", "azimuth_deg", "elevation_deg", "ring", "quadrant", "wanted_ring", "corners", "rms_px")} for h in held],
               runs=[dict(id=i, change=c) for i, c in RUNS], gpu_budget_h_t4=GPU_BUDGET_H, train_settings=TRAIN, methods=M, band_px=BAND_PX,
               acceptance=dict(**A4, guard=0.05, reference="human-corrected masks of the fresh views (primary); auto masks secondary"), scale_check=A5, organ_split="none")
    json.dump(con, open(out / "contract.json", "w"), indent=1)
    tbl = "\n".join(f"| `{h['file']}` | {h['quadrant']}° | {h['azimuth_deg']}° | {h['elevation_deg']}° | {h['ring']} | {h['corners']} | {h['rms_px']} |" for h in held)
    (out / "CONTRACT_R3-1.md").write_text(f"""# CONTRACT R3-1 — whole-plant reconstruction with better masks, initialisation and training (2026-10-22)

Prompt "R3-1" (user 2026-10-22 17:29). Written before any new mask or training. Run 01 and the v2 folder (R0-R3) are read-only.
Shape-only rule; the session never writes human truth; `organ_split: none`. GPU budget <= {GPU_BUDGET_H} h T4, one kernel at a time.

## Fresh held-out views (seed {SEED}; drawn from the 58 views that were training views in R3; one per azimuth quadrant, ring sequence
drawn without replacement from (low, middle, upper, low); never used to train, to build masks from 3D, or to tune)

| view | quadrant | azimuth | elevation | ring | corners | PnP RMS px |
|---|---|---|---|---|---|---|
{tbl}

Secondary sets (already looked at): v2 held-out {', '.join('`' + v + '`' for v in old_new)}; run 01 held-out {', '.join('`' + v + '`' for v in old_01)}.
Training views: the other {len(train)} accepted views (the 7 old held-out views are training views here).

## Reference masks (Step 1)
The 4 fresh views are scored against masks the USER corrects on a db-backed page (proposal = the A2 whole-plant mask; the user adds by
SAM point or brush, erases, and may mark a 2 px uncertainty band where hairs make the edge ambiguous). Auto-mask scores are reported too.
If the user declines, scoring is against auto masks only and the report says so.

## Mask improvements (Step 2, training views only)
- **M1** {M['M1']['method']} (seeds per ring {M['M1']['seeds_per_ring']}).
- **M2** {M['M2']['method']}.
- **M3** {M['M3']['method']}.
- **M4** {M['M4']['method']}.
- **M5** {M['M5']['method']}.
- Contact sheet on 12 training views: A2 -> M1-M5, added (cyan) and removed (red) pixels.

## Initial points (Step 3)
- **I1** {M['I1']['method']}.
- **I2** {M['I2']['method']} (N = {M['I2']['n_min']}).

## Training settings (Step 4, chosen without any held-out result)
{M['T']['method']}.

## Runs (Step 5), one change at a time
| run | contents |
|---|---|
""" + "\n".join(f"| {i} | {c} |" for i, c in RUNS) + f"""

Reported per run on the fresh views (human masks primary, auto secondary) and on the two old sets (secondary): raw IoU, band-excluded
IoU ({BAND_PX} px), edge F at 2 px, stem coverage, held-out minus training gap, depth consistency (check only).

## Acceptance (Step 6, unchanged from v2)
On the fresh views with HUMAN masks: medians band-excluded IoU >= {A4['band_excluded_iou_min']}, raw IoU >= {A4['raw_iou_min']}, edge F >= {A4['edge_f2_min']},
stem coverage >= {A4['stem_coverage_min']}, gap >= {A4['gap_min']}; and no single view more than 0.05 below a median threshold. The first run in
run order that passes is accepted; otherwise the best run is kept as **best unaccepted** with the largest remaining cause. No further
tuning on these views; no threshold change after seeing results.

## Outputs (Step 7)
Surfels/discs, mesh, point cloud, `recon_meta.json` (run, acceptance, `organ_split: none`), a disc viewer with the banner
(accepted / best unaccepted); a LOG NOTE when the organ-split input may switch to R3-1.
""", encoding="utf-8")
    print(json.dumps(dict(fresh=names, train=len(train), detail=[(h["file"][9:18], h["ring"], h["azimuth_deg"]) for h in held])))


if __name__ == "__main__":
    main()
