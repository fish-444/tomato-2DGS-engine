"""Roboflow tomato_top_nonboard v4 boxes for the leafseg engine (contract build/p1_leafseg_20261019_02/CONTRACT.md, B1-B2).

The response is cached per image; the key is passed in memory only and never stored.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.topview.calibration import load_config

LEAFLET_CLASSES = ("leaf", "terminal_leaf", "leaf2")
COMPOUND_CLASS = "compond_leaf"
COMPOUND_COVER = 0.80
TILE, STRIDE, EDGE_PX, TILE_NMS = 1288, 644, 8, 0.5   # variant C (contract B5)


def _jpeg_bytes(path: Path, rgb: np.ndarray) -> bytes:
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return path.read_bytes()
    b = io.BytesIO(); Image.fromarray(rgb).save(b, format="JPEG", quality=95); return b.getvalue()


def fetch(path: Path, rgb: np.ndarray, cache: Path, key: str | None) -> dict:
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if not key:
        raise SystemExit(f"no cached Roboflow detections for {path} and no key")
    det = load_config()["detector"]
    url = f"{det['endpoint']}?api_key={key}&confidence={det['query_confidence_percent']}&overlap={det['query_overlap_percent']}"
    req = urllib.request.Request(url, data=base64.b64encode(_jpeg_bytes(Path(path), rgb)),
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    d.pop("inference_id", None)
    cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(d), encoding="utf-8")
    return d


def to_dets(pred: dict, image_hw, classes=LEAFLET_CLASSES) -> list[dict]:
    """Predictions of ``classes`` >= measure_min_confidence as engine detections (original pixels), most confident first."""
    cmin = load_config()["detector"]["measure_min_confidence"]
    sx, sy = image_hw[1] / pred["image"]["width"], image_hw[0] / pred["image"]["height"]
    rows = sorted((p for p in pred.get("predictions", []) if p["class"] in classes and p["confidence"] >= cmin), key=lambda p: -p["confidence"])
    out = []
    for k, p in enumerate(rows):
        x0, y0 = (p["x"] - p["width"] / 2) * sx, (p["y"] - p["height"] / 2) * sy
        x1, y1 = (p["x"] + p["width"] / 2) * sx, (p["y"] + p["height"] / 2) * sy
        out.append(dict(det=k, conf=float(p["confidence"]), cls=p["class"], obb=None, box=[x0, y0, x1, y1],
                        centre=[(x0 + x1) / 2, (y0 + y1) / 2]))
    return out


def compound_boxes(pred: dict, image_hw) -> list[dict]:
    return [dict(id=f"C{k + 1:02d}", conf=d["conf"], box=d["box"]) for k, d in enumerate(to_dets(pred, image_hw, (COMPOUND_CLASS,)))]


def assign_compound_boxes(label: np.ndarray, grid_scale: float, instances: list[dict], comps: list[dict]) -> None:
    """compound_box_id = smallest compound box holding >= 80 % of the leaflet's mask pixels (B2); in place."""
    ys, xs = np.nonzero(label)
    for i, inst in enumerate(instances):
        sel = label[ys, xs] == i + 1; px_, py_ = xs[sel] / grid_scale, ys[sel] / grid_scale; best = None
        for c in comps:
            x0, y0, x1, y1 = c["box"]
            if len(px_) and ((px_ >= x0) & (px_ <= x1) & (py_ >= y0) & (py_ <= y1)).mean() >= COMPOUND_COVER:
                a = (x1 - x0) * (y1 - y0)
                if best is None or a < best[0]:
                    best = (a, c["id"])
        inst["compound_box_id"] = best[1] if best else "unassigned"


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def tiled_dets(path: Path, rgb: np.ndarray, cache_dir: Path, name: str, key: str | None) -> tuple[list[dict], dict]:
    """Variant C (B5): 3x3 overlapping tiles -> v4 per tile -> drop boxes cut by an inner tile edge -> box NMS."""
    H, W = rgb.shape[:2]; ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)); xs = list(range(0, max(W - TILE, 0) + 1, STRIDE))
    allb, info = [], dict(tiles=0, raw=0, cut_dropped=0)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            tile = np.ascontiguousarray(rgb[y0:y0 + TILE, x0:x0 + TILE]); th, tw = tile.shape[:2]
            pred = fetch(Path(f"{name}_r{i}c{j}.png"), tile, Path(cache_dir) / f"{name}_r{i}c{j}.json", key); info["tiles"] += 1
            for d in to_dets(pred, (th, tw)):
                bx0, by0, bx1, by1 = d["box"]; info["raw"] += 1
                cut = (bx0 < EDGE_PX and x0 > 0) or (by0 < EDGE_PX and y0 > 0) or (bx1 > tw - EDGE_PX and x0 + tw < W) or (by1 > th - EDGE_PX and y0 + th < H)
                if cut:
                    info["cut_dropped"] += 1; continue
                allb.append(dict(conf=d["conf"], cls=d["cls"], box=[bx0 + x0, by0 + y0, bx1 + x0, by1 + y0], tile=f"r{i}c{j}"))
    allb.sort(key=lambda d: -d["conf"]); keep = []
    for d in allb:
        if all(_iou(d["box"], k["box"]) <= TILE_NMS for k in keep):
            keep.append(d)
    info["nms_dropped"] = len(allb) - len(keep); out = []
    for k, d in enumerate(keep):
        x0, y0, x1, y1 = d["box"]
        out.append(dict(det=k, conf=d["conf"], cls=d["cls"], obb=None, box=d["box"], centre=[(x0 + x1) / 2, (y0 + y1) / 2], tile=d["tile"]))
    info["kept"] = len(out)
    return out, info
