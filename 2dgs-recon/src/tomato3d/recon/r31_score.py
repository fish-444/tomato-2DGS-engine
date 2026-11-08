"""R3-1 scoring of one run on the fresh held-out views: human masks (primary, from the page's db export) and auto masks (secondary),
plus the declared acceptance rule (medians + per-view guard), from the kernel's <run>_views.npz alpha.

usage: python -m tomato3d.recon.r31_score --out <r3-1 run dir> --run R3-1a --k <kernel out dir> --auto masks_a [--human human_masks.json]
Writes <out>/scores_<run>.json. human_masks.json = {view: {mask: rle, band: rle, res: 1512}} exported from the artifact db (never edited).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon.charuco_masks import MASK_RES
from tomato3d.recon.wholeplant_accept import CLAUSES, GUARD
from tomato3d.recon.wholeplant_contract import A4

BAND = 2


def rle_decode(r, n):
    a = np.zeros(n, np.uint8); p = 0; v = 0
    for run in r:
        if v:
            a[p:p + run] = 1
        p += run; v ^= 1
    return a


def edge_f(gt, pr, t=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1,) * 2)
    if not pr.any() or not gt.any():
        return 0.0
    p = (pr & (cv2.dilate(gt.astype(np.uint8), k) > 0)).sum() / pr.sum(); r = (gt & (cv2.dilate(pr.astype(np.uint8), k) > 0)).sum() / gt.sum()
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def score_view(ref: np.ndarray, excl: np.ndarray, render: np.ndarray, stem: np.ndarray) -> dict:
    """ref = reference mask, excl = pixels excluded from the IoU (user band), render = alpha >= 0.5, stem = A2 stem pixels inside ref."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1,) * 2); cb = (cv2.dilate(ref.astype(np.uint8), k) - cv2.erode(ref.astype(np.uint8), k)) > 0
    ok = ~excl; raw = float(((ref & render) & ok).sum() / max(((ref | render) & ok).sum(), 1))
    ok2 = ok & ~cb; nb = float(((ref & render) & ok2).sum() / max(((ref | render) & ok2).sum(), 1))
    sc = float((stem & render).sum() / stem.sum()) if stem.sum() else None
    return dict(iou=raw, iou_no_band=nb, edge_f2=edge_f(ref, render), stem_coverage=sc, ref_px=int(ref.sum()), render_px=int(render.sum()), excluded_px=int(excl.sum()))


def judge(per_view: dict, train_median_iou_no_band: float | None, gap_iou: float | None) -> dict:
    out = {}
    for key, thr, name in CLAUSES:
        vals = {v: m[key] for v, m in per_view.items() if m[key] is not None}
        if not vals:
            out[key] = dict(name=name, threshold=thr, median=None, passes=False); continue
        med = float(np.median(list(vals.values()))); low = {v[9:18]: round(x, 4) for v, x in vals.items() if x < thr - GUARD}
        out[key] = dict(name=name, threshold=thr, median=med, views_below_floor=low, passes=bool(med >= thr and not low))
    out["gap_iou"] = dict(name="held-out minus training IoU", threshold=A4["gap_min"], median=gap_iou, passes=bool(gap_iou is not None and gap_iou >= A4["gap_min"]))
    return dict(clauses=out, accepted=all(c["passes"] for c in out.values()))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--run", required=True); ap.add_argument("--k", required=True)
    ap.add_argument("--auto", required=True); ap.add_argument("--human", default=None); ap.add_argument("--views", default=None); a = ap.parse_args(argv)
    out, kd = Path(a.out), Path(a.k); con = json.load(open(out / "contract.json")); fresh = con["heldout_fresh"]
    z = np.load(a.views or (kd / f"{a.run}_views.npz")); res = json.load(open(kd / "results.json"))["runs"][a.run]
    human = json.load(open(a.human)) if a.human and Path(a.human).exists() else None
    rep = dict(run=a.run, reference_human=bool(human), per_view_auto={}, per_view_human={})
    auto_dir = Path(a.auto) if Path(a.auto).exists() else out / a.auto
    for v in fresh:
        lab = np.asarray(Image.open(auto_dir / (Path(v).stem + ".png"))); auto = np.isin(lab, (1, 2)); stem_auto = lab == 2
        al = z[f"a_{v}"].astype(np.float32); render = al >= 0.5
        if render.shape[0] != MASK_RES:
            render = cv2.resize(render.astype(np.uint8), (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST) > 0
        rep["per_view_auto"][v] = score_view(auto, np.zeros_like(auto), render, stem_auto)
        if human and v in human:
            h = human[v]; hm = rle_decode(h["mask"], MASK_RES * MASK_RES).reshape(MASK_RES, MASK_RES) > 0
            hb = rle_decode(h.get("band", [MASK_RES * MASK_RES]), MASK_RES * MASK_RES).reshape(MASK_RES, MASK_RES) > 0
            rep["per_view_human"][v] = score_view(hm, hb, render, stem_auto & hm)
    tr = res["train"]; gap_auto = float(np.median([m["iou"] for m in rep["per_view_auto"].values()]) - tr["iou"])
    rep["auto"] = dict(judge=judge(rep["per_view_auto"], tr["iou_no_band"], gap_auto), gap_iou=gap_auto, train_medians=tr)
    if rep["per_view_human"]:
        gap_h = float(np.median([m["iou"] for m in rep["per_view_human"].values()]) - tr["iou"])
        rep["human"] = dict(judge=judge(rep["per_view_human"], tr["iou_no_band"], gap_h), gap_iou=gap_h)
    # old held-out sets (training views in the R3-1 split, already looked at): render vs the run's auto masks, secondary only
    rep["secondary_sets"] = {}
    for name, views in (("v2_heldout", con.get("heldout_v2", [])), ("run01_heldout", con.get("heldout_run01", []))):
        pv = {}
        for v in views:
            if f"a_{v}" not in z.files or not (auto_dir / (Path(v).stem + ".png")).exists():
                continue
            lab = np.asarray(Image.open(auto_dir / (Path(v).stem + ".png"))); render = z[f"a_{v}"].astype(np.float32) >= 0.5
            pv[v] = score_view(np.isin(lab, (1, 2)), np.zeros(lab.shape, bool), render, lab == 2)
        med = {k: (float(np.median([m[k] for m in pv.values() if m[k] is not None])) if pv else None) for k in ("iou", "iou_no_band", "edge_f2", "stem_coverage")}
        rep["secondary_sets"][name] = dict(note="training views in this split; vs auto masks", n=len(pv), medians=med, per_view=pv)
        print(f"   {name} (training here, vs auto): " + ", ".join(f"{k} {None if x is None else round(x, 3)}" for k, x in med.items()))
    json.dump(rep, open(out / f"scores_{a.run}.json", "w"), indent=1)
    for ref in ("human", "auto"):
        if ref in rep:
            j = rep[ref]["judge"]; print(f"{a.run} vs {ref} masks: {'ACCEPTED' if j['accepted'] else 'not accepted'}")
            for c in j["clauses"].values():
                print(f"   {c['name']:<28} median {c['median'] if c['median'] is None else round(c['median'], 3)} vs {c['threshold']} {'pass' if c['passes'] else 'FAIL'} {c.get('views_below_floor') or ''}")


if __name__ == "__main__":
    main()
