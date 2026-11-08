"""Immutable input/output and validation of human-owned review records."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import pillow_heif

from tomato3d.topview.instances import load_mask

PLANTS = ("dwt_101", "dwt_102", "dwt_103")
UNKNOWN = "unavailable"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    fields = columns or list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in r.items()})


def decode(path: Path) -> np.ndarray:
    pillow_heif.register_heif_opener()
    with Image.open(path) as im:
        if im.getexif().get(274, 1) != 1:
            raise ValueError(f"Unresolved EXIF orientation: {path}")
        return np.asarray(im.convert("RGB")).copy()


def load_plant(root: Path, data_root: Path, masks_root: Path, plant: str, cfg: dict) -> dict:
    folder = masks_root / plant
    head = read_json(folder / "measure2d.json")
    manifest = read_json(folder / "masks_manifest.json")
    raw_dir = data_root / "raw" / plant / f"{plant}_gen_20261005"
    captures = read_csv(raw_dir / "generator_capture.csv")
    top = next(r for r in captures if r["photo_file"] == "top.heic")
    raw = raw_dir / "top.heic"
    raw_hash = digest(raw)
    for expected in [top["sha256"], head["image_sha256"], manifest["image_sha256"]]:
        if raw_hash != expected:
            raise ValueError(f"Raw hash mismatch: {plant}")
    if head["plant_id"] != plant or manifest["plant_id"] != plant:
        raise ValueError("Plant identity mismatch")
    jpg = Path(head["image"])
    if not jpg.is_absolute():
        jpg = root / jpg
    if digest(jpg) != head["image_file_sha256"] or digest(jpg) != manifest["image_file_sha256"]:
        raise ValueError("Cached JPEG hash mismatch")
    rgb, cached = decode(raw), decode(jpg)
    h, w = rgb.shape[:2]
    if rgb.shape != cached.shape or [w, h] != manifest["image_size_px"]:
        raise ValueError("Raw/mask/JPEG grid mismatch")
    difference = np.abs(rgb.astype(np.float32) - cached)
    mean_abs, p99_abs = float(difference.mean()), float(np.percentile(difference, 99))
    if mean_abs > cfg["grid_mean_abs_max"] or p99_abs > cfg["grid_p99_abs_max"]:
        raise ValueError("Raw/JPEG pixel correspondence failed")
    scale = min(1.0, cfg["processing_max_px"] / max(w, h))
    size = (round(w * scale), round(h * scale))
    masks, kinds = {}, {}
    for e in manifest["masks"]:
        full = load_mask(e, folder)
        masks[e["instance_id"]] = cv2.resize(full.astype(np.uint8), size,
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
        kinds[e["instance_id"]] = e["kind"]
    leaves = read_csv(folder / "leaflets_2d.csv")
    compounds = read_csv(folder / "compounds_2d.csv")
    for row in leaves + compounds:
        if row["plant_id"] != plant or row["image_sha256"] != raw_hash or row["capture_id"] != head["capture_id"]:
            raise ValueError("Instance table identity/hash mismatch")
    expected_ids = {r["leaflet_id"] for r in leaves} | {r["compound_id"] for r in compounds}
    if set(masks) != expected_ids:
        raise ValueError("Mask/table instance mismatch")
    sides = []
    for r in captures:
        if r["photo_kind"] not in ("side", "level"):
            continue
        path = raw_dir / r["photo_file"]
        if digest(path) != r["sha256"]:
            raise ValueError(f"Side hash mismatch: {path}")
        sides.append(dict(file=r["photo_file"], path=str(path), sha256=r["sha256"],
                          azimuth_deg=float(r["azimuth_deg"]), kind=r["photo_kind"],
                          azimuth_basis=r.get("azimuth_basis")))
    provenance = dict(plant_id=plant, capture_id=head["capture_id"], image_sha256=raw_hash,
                      raw_path=str(raw), image_size_px=[w, h], processing_size_px=list(size),
                      processing_scale=scale, cached_jpeg_sha256=digest(jpg),
                      manifest_sha256=digest(folder / "masks_manifest.json"),
                      leaflets_table_sha256=digest(folder / "leaflets_2d.csv"),
                      compounds_table_sha256=digest(folder / "compounds_2d.csv"),
                      measure2d_sha256=digest(folder / "measure2d.json"),
                      capture_table_sha256=digest(raw_dir / "generator_capture.csv"),
                      grid_mean_abs=mean_abs, grid_p99_abs=p99_abs,
                      detector=head["detector"], segmenter=head.get("segmenter"),
                      derived_export=head.get("derived_export"), review_status="proposed", sides=sides)
    return dict(rgb_raw=rgb, rgb=cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), masks=masks,
                kinds=kinds, leaves=leaves, compounds=compounds, provenance=provenance)


def blank_review(data: dict) -> dict:
    p = data["provenance"]
    return dict(schema="occlusion-review-1", plant_id=p["plant_id"], capture_id=p["capture_id"],
                image_sha256=p["image_sha256"], image_size_px=p["image_size_px"], annotator="",
                rachises=[], apices={}, membership_fixes={}, orientation=None, side_regions=[],
                top_order=[], side_height=[], faces=[], rank_links=[], mask_reviews={})


def validate_review(review: dict, data: dict) -> dict:
    """Validate without repairing records or converting suggestions into truth."""
    p = data["provenance"]
    for key in ("plant_id", "capture_id", "image_sha256", "image_size_px"):
        if review.get(key) != p[key]:
            raise ValueError(f"Review {key} mismatch")
    if review.get("schema") != "occlusion-review-1":
        raise ValueError("Unknown review schema")
    leaf_ids = {r["leaflet_id"] for r in data["leaves"]}
    compound_ids = {r["compound_id"] for r in data["compounds"]}
    mask_reviews = review.get("mask_reviews", {})
    if not isinstance(mask_reviews, dict):
        raise ValueError("Mask reviews must be an ID-to-decision mapping")
    for iid, decision in mask_reviews.items():
        if iid not in leaf_ids | compound_ids or decision not in ("accept", "reject"):
            raise ValueError("Invalid segmentation review ID or O/X decision")
    w, h = p["image_size_px"]

    def point(v: Any, width: int = w, height: int = h) -> None:
        if not isinstance(v, list) or len(v) != 2 or not all(
                isinstance(x, (int, float)) and np.isfinite(x) for x in v):
            raise ValueError("Invalid point")
        if not (0 <= v[0] < width and 0 <= v[1] < height):
            raise ValueError("Point outside image")

    seen = set()
    for r in review.get("rachises", []):
        cid = r["compound_id"]
        if cid not in compound_ids or cid in seen or len(r["points_px"]) < 2:
            raise ValueError("Invalid/duplicate rachis")
        seen.add(cid)
        if r["axis"] not in ("M", "S1", "S2", "neighbour"):
            raise ValueError("Unknown axis tag")
        for pt in r["points_px"]:
            point(pt)
    for axis, pt in review.get("apices", {}).items():
        if axis not in ("M", "S1", "S2", "neighbour"):
            raise ValueError("Unknown apex axis")
        point(pt)
    fixes = review.get("membership_fixes", {})
    if not isinstance(fixes, dict):
        raise ValueError("Membership must map each leaflet to one compound")
    for leaf, compound in fixes.items():
        if not isinstance(compound, str):
            raise ValueError("Each leaflet must have one compound or be unassigned")
        if leaf not in leaf_ids or compound not in compound_ids | {"unassigned"}:
            raise ValueError("Unknown membership id")
    orientation = review.get("orientation")
    if orientation is not None:
        if not isinstance(orientation.get("side0_image_deg"), (int, float)) or not np.isfinite(
                orientation["side0_image_deg"]) or orientation.get("azimuth_sign") not in (-1, 1):
            raise ValueError("Orientation requires side0_image_deg and azimuth_sign +/-1")
    for key, field, ids, choices in [("top_order", "over", compound_ids, {"A", "B", "cannot_tell", ""}),
                                    ("side_height", "higher", compound_ids, {"A", "B", "cannot_tell", ""})]:
        seen = set()
        for r in review.get(key, []):
            pair = tuple(sorted((r["a"], r["b"])))
            if r["a"] == r["b"] or not set(pair) <= ids or pair in seen or r[field] not in choices:
                raise ValueError(f"Invalid/duplicate {key} pair")
            seen.add(pair)
    seen = set()
    for r in review.get("faces", []):
        if r["leaflet_id"] not in leaf_ids or r["leaflet_id"] in seen or r["face"] not in (
                "", "adaxial", "abaxial", "both_folded", "cannot_tell"):
            raise ValueError("Invalid/duplicate face label")
        seen.add(r["leaflet_id"])
    side_lookup = {s["file"]: s for s in p["sides"]}
    for r in review.get("side_regions", []):
        if orientation is None or r["file"] not in side_lookup or r["compound_id"] not in compound_ids:
            raise ValueError("Side region requires orientation, known file and compound")
        side = side_lookup[r["file"]]
        if r["image_sha256"] != side["sha256"]:
            raise ValueError("Side region image hash mismatch")
        with Image.open(side["path"]) as im:
            sw, sh = im.size
        box = r["box_px"]
        if len(box) != 4:
            raise ValueError("Invalid side rectangle")
        point(box[:2], sw, sh)
        point(box[2:], sw, sh)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Empty side rectangle")
    seen = set()
    for r in review.get("rank_links", []):
        if r["compound_id"] not in compound_ids or r["compound_id"] in seen or not r.get("source_ref"):
            raise ValueError("Rank link needs unique compound and source reference")
        if r.get("axis") not in ("M", "S1", "S2") or type(r.get("node_index_from_base")) is not int or r["node_index_from_base"] < 1:
            raise ValueError("Invalid rank/node link")
        seen.add(r["compound_id"])
    edited = any(review.get(k) for k in ("rachises", "apices", "membership_fixes", "orientation", "side_regions", "rank_links", "mask_reviews"))
    edited |= any(r.get(field) for key, field in (("faces", "face"), ("top_order", "over"), ("side_height", "higher"))
                  for r in review.get(key, []))
    if edited and not review.get("annotator", "").strip():
        raise ValueError("Human edits require annotator")
    return review


def storage_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
