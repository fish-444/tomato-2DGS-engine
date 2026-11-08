"""Mask-method scoring against the user's masks (CONTRACT_MASK_METHOD.md): stem coverage first, band-excluded IoU, raw IoU,
edge F at 2 px, pixels added / removed, and holes inside the plant.

usage: python -m tomato3d.masks.score --cand C0 --masks <dir of <stem>.png (labels 1/2 = plant) or binary> --out <run dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.recon.r31_score import edge_f, rle_decode

T2 = Path("build/p1_recon_tomato2_20261023_01/t2_02")
R = 1512


def user_masks():
    h = json.load(open(T2 / "human_masks.json"))
    return {v: (rle_decode(d["mask"], R * R).reshape(R, R) > 0, rle_decode(d.get("band", [R * R]), R * R).reshape(R, R) > 0) for v, d in h.items()}


THIN_R = 13                                                                                  # Amendment 1: R3-1 THIN_PX 26 at 3024 -> 13 at 1512


def stem_pixels(v, hm):
    """Thin parts of the user's own mask: mask minus its opening with a disc of THIN_R, components touching the rest of the mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * THIN_R + 1,) * 2); m = hm.astype(np.uint8)
    wide = cv2.morphologyEx(m, cv2.MORPH_OPEN, k) > 0; thin = hm & ~wide
    near = cv2.dilate(wide.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    n, lab = cv2.connectedComponents(thin.astype(np.uint8)); keep = np.zeros(n, bool)
    for i in range(1, n):
        c = lab == i
        if c.sum() >= 50 and (c & near).any():
            keep[i] = True
    return keep[lab]


def holes(cand, hm):
    inv = (~cand).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    px, cnt = 0, 0
    for i in range(1, n):
        if i in border:
            continue
        m = lab == i; inside = (m & hm).sum()
        if inside:
            px += int(inside); cnt += 1
    return px, cnt


def score_view(cand, hm, hb, stem):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); cb = (cv2.dilate(hm.astype(np.uint8), k) - cv2.erode(hm.astype(np.uint8), k)) > 0
    ok = ~hb; raw = float(((cand & hm) & ok).sum() / max(((cand | hm) & ok).sum(), 1)); ok2 = ok & ~cb
    nb = float(((cand & hm) & ok2).sum() / max(((cand | hm) & ok2).sum(), 1)); hpx, hcnt = holes(cand, hm)
    return dict(stem_coverage=float((stem & cand).sum() / stem.sum()) if stem.sum() else None, iou_no_band=nb, iou=raw, edge_f2=edge_f(hm, cand),
                added_px=int((cand & ~hm).sum()), removed_px=int((hm & ~cand).sum()), user_px=int(hm.sum()), holes_px=hpx, holes_n=hcnt, stem_px=int(stem.sum()))


def load_cand(d: Path, v: str):
    p = d / f"{Path(v).stem}.png"
    if not p.exists():
        return None
    a = np.asarray(Image.open(p)); m = np.isin(a, (1, 2)) if a.max() > 1 else a > 0
    return cv2.resize(m.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST) > 0 if m.shape[0] != R else m


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--cand", required=True); ap.add_argument("--masks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev", choices=("dev", "heldout")); a = ap.parse_args(argv)
    con = json.load(open(T2 / "contract.json")); um = user_masks()
    views = [v for v in con["gap_views"] if v in um] if a.split == "dev" else [v for v in con["heldout"] if v in um]
    per = {}
    for v in views:
        c = load_cand(Path(a.masks), v); hm, hb = um[v]
        per[v] = score_view(c, hm, hb, stem_pixels(v, hm)) if c is not None else dict(missing=True)
    ok = [x for x in per.values() if not x.get("missing")]
    med = {k: (float(np.median([x[k] for x in ok if x[k] is not None])) if ok and any(x[k] is not None for x in ok) else None) for k in ("stem_coverage", "iou_no_band", "iou", "edge_f2")}
    gate = dict(stem=med["stem_coverage"] is not None and med["stem_coverage"] >= 0.60, iou_no_band=med["iou_no_band"] is not None and med["iou_no_band"] >= 0.85,
                added_lt_plant=bool(ok) and len(ok) == len(per) and all(x["added_px"] < x["user_px"] for x in ok))
    rep = dict(candidate=a.cand, split=a.split, views=per, medians=med, gate=dict(gate, passes=all(gate.values())))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); json.dump(rep, open(out / f"scores_{a.cand}_{a.split}.json", "w"), indent=1)
    print(a.cand, a.split, {k: (round(x, 3) if x is not None else None) for k, x in med.items()}, rep["gate"])
    for v, x in per.items():
        print("  ", v, {k: (round(y, 3) if isinstance(y, float) else y) for k, y in x.items()})


if __name__ == "__main__":
    main()
