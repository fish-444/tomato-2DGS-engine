"""토마토2 Stage 3 contract for one individual: 4 held-out views (R3-1 picker, seed 20261023) and 8 gap-reference training views
(one per 45-degree azimuth octant, seed 20261024), fixed before any mask exists. Pure bookkeeping on the Stage 2 qc.csv (no compute).

usage: python -m tomato3d.recon.tomato2_contract --qc build/p1_recon_tomato2_<date>_01/stage2_d7/t2_02/qc.csv --iid t2_02 --out build/p1_recon_tomato2_<date>_01/t2_02
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tomato3d.recon.r31_contract import pick
from tomato3d.recon.wholeplant_contract import ring

GAP_SEED, N_GAP = 20261024, 8


def gap_views(rows: list[dict], exclude: set[str]) -> list[dict]:
    rng = np.random.default_rng(GAP_SEED); out = []
    for k in range(N_GAP):
        lo, hi = 45 * k, 45 * (k + 1)
        cand = [r for r in rows if lo <= float(r["azimuth_deg"]) < hi and r["file"] not in exclude]
        if cand:
            r = cand[int(rng.integers(len(cand)))]; out.append(dict(r, ring=ring(float(r["elevation_deg"])), octant=f"{lo}-{hi}"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--qc", required=True); ap.add_argument("--iid", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.qc, encoding="utf-8")) if r["accepted"] == "1"]
    held = pick(rows, {r["file"] for r in rows}); names = [h["file"] for h in held]
    gap = gap_views(rows, set(names)); train = [r["file"] for r in rows if r["file"] not in names]
    keep = ("file", "group", "azimuth_deg", "elevation_deg", "ring", "corners", "rms_px")
    con = dict(individual=a.iid, seed=20261023, gap_seed=GAP_SEED, heldout=names, heldout_new=names, heldout_run01=[], train=train,
               heldout_detail=[{k: h.get(k) for k in keep + ("quadrant", "wanted_ring")} for h in held],
               gap_views=[g["file"] for g in gap], gap_detail=[{k: g.get(k) for k in keep + ("octant",)} for g in gap],
               page_views=names + [g["file"] for g in gap], organ_split="none",
               acceptance=dict(iou_no_band=0.85, iou=0.75, edge_f2=0.90, stem_coverage=0.60, gap_min=-0.03, guard=0.05,
                               reference="user-corrected masks: held-out 4 for the medians, gap views 8 for the training side (same source both sides)"))
    json.dump(con, open(out / "contract.json", "w"), indent=1)
    print(json.dumps(dict(heldout=[(h["file"], h["ring"], h["azimuth_deg"], h["elevation_deg"]) for h in held],
                          gap=[(g["file"], g["ring"], g["azimuth_deg"], g["elevation_deg"]) for g in gap], train=len(train)), indent=1))


if __name__ == "__main__":
    main()
