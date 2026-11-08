"""A0: pick the new held-out views by a seeded rule and write the contract, before any new mask or training.

Rule (seed 20261022): the 65 accepted views of run 01 are grouped into rings (top / upper > 58 deg / middle 45-58 / low < 45) and
into 4 azimuth quadrants (0-90, 90-180, 180-270, 270-360). One held-out view is drawn per quadrant; the ring sequence is drawn
without replacement from (low, middle, upper, low) so every ring except the 2-view top appears and the low ring appears twice (it is
the weakest one). Within a (quadrant, ring) cell the view is drawn uniformly at random with the seeded generator. Run 01's three
held-out views are excluded from the draw and reported as a second set.

usage: python -m tomato3d.recon.wholeplant_contract --poses <charuco dir> --run01 <run 01 dir> --out <run dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEED = 20261022
RINGS = ((75.0, "top"), (58.0, "upper"), (45.0, "middle"), (-90.0, "low"))
RING_SEQUENCE = ("low", "middle", "upper", "low")
QUADRANTS = ((0, 90), (90, 180), (180, 270), (270, 360))
# declared thresholds (A1-A5)
A1 = dict(corner_rms_max_px=0.8, cam_shift_max_m=0.002, cam_rot_max_deg=0.5, z_free=True, inplane_sigma_m=0.0005)
A2 = dict(band_px=2, stem_thin_sigma_px=(2, 4), stem_min_share=0.0)
A4 = dict(band_excluded_iou_min=0.85, raw_iou_min=0.75, edge_f2_min=0.90, gap_min=-0.03, stem_coverage_min=0.6)
A5 = dict(pot_rim_mm=59.0, tolerance=0.03)
RUNS = (("R0", "run 01 settings on the new split (baseline)"), ("R1", "R0 + A1 board refinement"),
        ("R2", "R1 + A2 whole-plant masks with stem"), ("R3", "R2 + A3 dense initialisation"),
        ("R4", "R3 + 3024 px training (only if the GPU budget allows)"))
GPU_BUDGET_H = 3.0
TRAIN = dict(res=1512, iters=7000, n_init=150000, seed=0)


def ring(el: float) -> str:
    return next(n for lo, n in RINGS if el > lo)


def pick(rows: list[dict], exclude: set[str]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    pool = [r for r in rows if r["file"] not in exclude]
    seq = list(RING_SEQUENCE); rng.shuffle(seq)
    out, used = [], set()
    for (lo, hi), want in zip(QUADRANTS, seq):
        cand = [r for r in pool if lo <= float(r["azimuth_deg"]) < hi and r["file"] not in used]
        pref = [r for r in cand if ring(float(r["elevation_deg"])) == want] or cand
        if not pref:
            continue
        r = pref[int(rng.integers(len(pref)))]; used.add(r["file"]); out.append(dict(r, ring=ring(float(r["elevation_deg"])), quadrant=f"{lo}-{hi}", wanted_ring=want))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--run01", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src, r01, out = Path(a.poses), Path(a.run01), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(src / "qc.csv")) if r["accepted"] == "1"]
    old = json.load(open(r01 / "contract.json"))["heldout"]
    held = pick(rows, set(old))
    names = [h["file"] for h in held]; train = [r["file"] for r in rows if r["file"] not in names]
    con = dict(date="2026-10-22", seed=SEED, accepted_views=len(rows), heldout_new=names, heldout_run01=old,
               heldout_detail=[{k: h[k] for k in ("file", "azimuth_deg", "elevation_deg", "ring", "quadrant", "wanted_ring", "corners", "rms_px")} for h in held],
               train=train, runs=[dict(id=i, change=c) for i, c in RUNS], gpu_budget_h_t4=GPU_BUDGET_H, train_settings=TRAIN,
               thresholds=dict(a1=A1, a2=A2, a4=A4, a5=A5), organ_split="none")
    json.dump(con, open(out / "contract.json", "w"), indent=1)
    tbl = "\n".join(f"| `{h['file']}` | {h['quadrant']}° | {h['azimuth_deg']}° | {h['elevation_deg']}° | {h['ring']} | {h['corners']} | {h['rms_px']} |" for h in held)
    (out / "CONTRACT.md").write_text(f"""# CONTRACT — whole-plant reconstruction from the A3 ChArUco capture (2026-10-22)

Prompt "Whole-plant reconstruction first: improve the ChArUco pot reconstruction before any organ detection" (user 2026-10-22 10:35
"검출전 개체 전체 복원부터 집중하도록"). Written before any new mask or training. Run 01 (`build/p1_recon_charuco_20261021_01/`) is read-only.
Shape-only rule: geometry in board metres, no trait tables, no mm accuracy claims about the plant; the A5 pot check is a scale check.
**No organ split in this task** (`organ_split: none`): the foreground is the whole plant as one class.

## Views
{len(rows)} accepted views from run 01's pose QC (`build/p1_charuco_a3_20261021_01/qc.csv`).
**New held-out set** (seed {SEED}, drawn before any new mask or training; one per azimuth quadrant, ring sequence drawn without
replacement from (low, middle, upper, low), uniform within a cell; run 01's three held-out views excluded from the draw):

| view | quadrant | azimuth | elevation | ring | corners | PnP RMS px |
|---|---|---|---|---|---|---|
{tbl}

Second set, reported separately (already looked at in run 01, so not the primary test): {', '.join('`' + v + '`' for v in old)}.
Training views: the remaining {len(train)} (run 01's held-out views are available for training here, which is noted whenever the
second set's numbers are quoted).

## Runs (each adds one change)
| run | change |
|---|---|
""" + "\n".join(f"| {i} | {c} |" for i, c in RUNS) + f"""

Base training settings (as run 01): gsplat 1.5.3 `rasterization_2dgs`, {TRAIN['res']} px, {TRAIN['iters']} iterations, seed {TRAIN['seed']},
init ≤ {TRAIN['n_init']:,} points, L1 + 0.2 SSIM + 0.1 alpha, normal 0.05 from 2000, distortion 100 from 1500, DefaultStrategy as row 31.
R4 raises the resolution to 3024 px only if the GPU budget allows. **GPU budget ≤ {GPU_BUDGET_H} h T4 in total.**

## A1 — board refinement
Bundle adjustment on board corners only (all {len(rows)} accepted views): camera poses + the 54 ChArUco corner positions, intrinsics
fixed, no plant features. Each corner's z is free; in-plane drift is regularised with sigma {A1['inplane_sigma_m'] * 1000:.1f} mm.
**Accept the refined poses if** corner reprojection RMS ≤ {A1['corner_rms_max_px']} px **and** no camera centre moves more than
{A1['cam_shift_max_m'] * 1000:.0f} mm and no camera rotates more than {A1['cam_rot_max_deg']}°. Otherwise keep run 01's poses and say why.

## A2 — whole-plant masks with stem
Foreground = leaves + stem + petioles + trusses as one class. Stem recovery: SAM v5 seeded at thin bright structures inside the
plant's convex hull (ridge filter, sigma {A2['stem_thin_sigma_px']} px) and at the stem base (pot centre projected from the poses),
unioned with run 01's colour-in-hull foreground. Reported per view: the share of pixels added. Contact sheet of 12 views, stem before
and after. **Edge band**: a {A2['band_px']} px band along every mask boundary (trimap inside / band / outside); the band is excluded
from the training mask loss and IoU is reported both with and without it. The band rule is fixed here and is not tuned.

## A3 — denser initialisation
COLMAP `patch_match_stereo` + `stereo_fusion` with the accepted poses, restricted to the A2 foreground (GPU, Kaggle); fallback:
full-resolution SIFT with more matches. Report points per view, especially the 6 low-angle views with < 10 points in run 01.

## A4 — metrics and acceptance (declared now)
Per run, on the new held-out set and on run 01's set: silhouette IoU raw; IoU excluding the {A2['band_px']} px band; edge F-score at
2 px; gap = held-out median − training-view median; stem coverage = share of A2 stem-added pixels covered by the render (alpha ≥ 0.5)
in held-out views; depth consistency vs triangulated points (check only).
**Acceptance — the first run in order R0…R4 where every clause holds on the NEW held-out set:**
band-excluded IoU ≥ {A4['band_excluded_iou_min']}; raw IoU ≥ {A4['raw_iou_min']}; edge F at 2 px ≥ {A4['edge_f2_min']};
gap ≥ {A4['gap_min']}; stem coverage ≥ {A4['stem_coverage_min']}. Thresholds apply to the median over the 4 new held-out views, and
every view must also be within 0.05 of the threshold (no single view carries the median). No tuning on held-out views; no threshold
or band change after seeing results. If no run passes: report the largest remaining cause and go to Part B.

## A5 — pot-scale check (scale only)
Fit the pot rim in the accepted model and compare with the known rim dimension {A5['pot_rim_mm']:.0f} mm (row 32 value; replaced if the
user measures this pot). Report the ratio; flag if it is off by more than {A5['tolerance']:.0%}. This checks the board scale, not plant traits.

## A6 — outputs
Surfels/discs, a TSDF mesh, a point cloud in board metres, the existing disc viewer with the banner "whole plant, no organ split",
and `recon_meta.json` (poses used, run id, metrics, acceptance, `organ_split: none`).
""", encoding="utf-8")
    print(json.dumps(dict(new_heldout=names, train=len(train), old=old), indent=1))


if __name__ == "__main__":
    main()
