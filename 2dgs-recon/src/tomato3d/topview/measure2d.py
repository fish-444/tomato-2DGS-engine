"""T1 - real top-view photo -> measured 2-D per compound and per leaflet (+ X1-X4 of
``handoff/CLAUDE_CODE_CLI_PROMPT_TOPVIEW_REAL_LINK_2026-10-12.md``).

Detector boxes (Roboflow ``tomato_top_nonboard`` v4, trained on real photos) -> SAM 2 box-prompt masks
(``tools.leaflet_segmenter``, imported read-only) -> per instance: projected area, projected length / width
(minimum-area rectangle), long-axis angle, centroid. Compounds (``compond_leaf``) and leaflets (``leaf`` /
``terminal_leaf`` / ``leaf2``) are both segmented; the masks are persisted (:mod:`tomato3d.topview.instances`);
leaflets get a compound membership and every instance an own-pot attribution. A box is not an area: every area here
comes from a mask. Every row is ``review_status = proposed`` until a person confirms it
(:mod:`tomato3d.topview.confirm_tool`); detector boxes stay suggestions (DETECTOR_POLICY).

Link (X1): a photo carries ``plant_id`` and the RAW photo's sha256 only through :mod:`tomato3d.topview.reallink`,
which refuses any file whose pixel grid cannot be shown identical to the capture's. **A row without plant_id + the
original hash carries no mm.**

Scale: per pot, from that pot's own rim (``tomato3d.state.potscale``, human-clicked rim points). Without a solved
rim scale every mm value is ``unavailable`` and only pixel values are reported - there is no image-wide scale.
With it, mask outlines are mapped through the rim homography (rim-plane mm). Leaves above the rim are nearer the
camera and magnified, so rim-plane mm OVERestimate them (factor Z_rim / Z_leaf > 1); the potscale plane note is
carried on every value and no correction is applied. The scale's 1-sigma is propagated first-order (relative sd of
``mm_per_px``; x2 for areas); segmentation error is not in it.

``vis_frac`` is ``unavailable`` on real photos: there is no amodal truth, and a hidden part cannot be measured.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tomato3d.topview import instances as inst
from tomato3d.topview.calibration import load_config

UNAVAILABLE = "unavailable"
SCHEMA = "topview-measure2d-2"
ID_COLUMNS = ("plant_id", "capture_id", "image_sha256")
SHAPE_COLUMNS = ("mask_status", "sam_score", "mask_outside_box_frac", "area_px", "length_px", "width_px",
                 "projected_aspect", "axis_deg_image", "centroid_x_px", "centroid_y_px")
MM_COLUMNS = ("scale_status", "area_mm2", "area_mm2_sd", "length_mm", "length_mm_sd", "width_mm", "width_mm_sd",
              "scale_plane", "plane_bias")
BOX_COLUMNS = ("class", "det_confidence", "box_x0_px", "box_y0_px", "box_x1_px", "box_y1_px")
CSV_COLUMNS = (("leaflet_id",) + ID_COLUMNS + BOX_COLUMNS + SHAPE_COLUMNS + MM_COLUMNS
               + ("compound_id", "membership_status", "attribution", "attribution_reason", "vis_frac",
                  "review_status"))
COMPOUND_COLUMNS = (("compound_id",) + ID_COLUMNS + BOX_COLUMNS + SHAPE_COLUMNS + MM_COLUMNS
                    + ("n_member_leaflets", "terminal_conflict", "attribution", "attribution_reason",
                       "review_status"))
PHOTO_COLUMNS = ("plant_id", "capture_id", "image_sha256", "image_path", "image_file_sha256", "link_status",
                 "link_reason", "detector_split_v4", "scale_status", "scale_reason", "mm_per_px", "mm_per_px_sd",
                 "pose_status", "depth_mm_frame_width", "depth_mm_frame_diagonal", "tilt_deg_frame_width",
                 "axis_ratio_self_check_frame_width", "n_compounds", "n_leaflets", "n_leaflets_mm",
                 "n_leaflets_inside", "n_leaflets_ambiguous", "n_leaflets_outside")


# --------------------------------------------------------------------------- detector
def detect(image_path: Path, cache: Path, cfg: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    """Roboflow serverless predictions for one photo, cached as JSON next to the run (key never stored)."""
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import base64
    import urllib.request

    key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("no cached predictions and no ROBOFLOW_API_KEY in the environment")
    det = cfg["detector"]
    url = (f"{det['endpoint']}?api_key={key}&confidence={det['query_confidence_percent']}"
           f"&overlap={det['query_overlap_percent']}")
    req = urllib.request.Request(url, data=base64.b64encode(Path(image_path).read_bytes()),
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    d.pop("inference_id", None)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d), encoding="utf-8")
    return d


def leaflet_boxes(pred: dict[str, Any], cfg: dict[str, Any], image_wh: tuple[int, int],
                  classes: list[str] | None = None) -> list[dict[str, Any]]:
    """Predictions of ``classes`` (default: the leaflet classes) >= the measurement confidence, as xyxy in
    ORIGINAL image pixels, most confident first."""
    classes = cfg["leaflet_classes"] if classes is None else classes
    iw, ih = pred["image"]["width"], pred["image"]["height"]
    sx, sy = image_wh[0] / iw, image_wh[1] / ih
    keep = []
    for p in pred.get("predictions", []):
        if p["class"] not in classes or p["confidence"] < cfg["detector"]["measure_min_confidence"]:
            continue
        keep.append(dict(cls=p["class"], conf=float(p["confidence"]),
                         box=[(p["x"] - p["width"] / 2) * sx, (p["y"] - p["height"] / 2) * sy,
                              (p["x"] + p["width"] / 2) * sx, (p["y"] + p["height"] / 2) * sy]))
    keep.sort(key=lambda b: -b["conf"])
    return keep


# --------------------------------------------------------------------------- mask -> 2-D measures
def measure_mask(mask: np.ndarray) -> dict[str, Any] | None:
    """Projected area, min-area-rect length/width and long-axis angle of the largest component (pixels)."""
    m = mask.astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    length, width = (w, h) if w >= h else (h, w)
    mo = cv2.moments(comp, binaryImage=True)       # long axis from second moments (min-area-rect angles snap)
    axis = 0.5 * math.degrees(math.atan2(2.0 * mo["mu11"], mo["mu20"] - mo["mu02"]))  # image coords, y down
    return dict(area_px=int(comp.sum()), length_px=float(length), width_px=float(width),
                projected_aspect=float(length / width) if width > 0 else None,
                axis_deg_image=round(float(axis % 180.0), 2), centroid_px=(float(cent[k][0]), float(cent[k][1])),
                contour_px=c.reshape(-1, 2).astype(np.float64), components=int(n - 1))


def measure_mm(contour_px: np.ndarray, homography: np.ndarray) -> dict[str, float]:
    """Rim-plane mm measures of one outline through the pot's homography (image px -> rim-plane mm)."""
    from tomato3d.state.potscale import apply_h

    mm = apply_h(np.asarray(homography, float), contour_px).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(mm)
    return dict(area_mm2=float(abs(cv2.contourArea(mm))), length_mm=float(max(w, h)), width_mm=float(min(w, h)))


def mm_with_sd(mm: dict[str, float], rel_sd: float) -> dict[str, float]:
    """First-order propagation of the scale's relative 1-sigma (lengths x1, areas x2)."""
    return dict(area_mm2=round(mm["area_mm2"], 3), area_mm2_sd=round(mm["area_mm2"] * 2.0 * rel_sd, 4),
                length_mm=round(mm["length_mm"], 3), length_mm_sd=round(mm["length_mm"] * rel_sd, 4),
                width_mm=round(mm["width_mm"], 3), width_mm_sd=round(mm["width_mm"] * rel_sd, 4))


# --------------------------------------------------------------------------- SAM
def load_sam(config_path: Path):
    from tools.leaflet_segmenter import load_segmenter

    seg = load_segmenter("sam2", Path(config_path))
    if not hasattr(seg, "_load"):
        raise SystemExit(f"SAM 2 unavailable: {seg.info() if hasattr(seg, 'info') else seg}")
    seg._load()                                      # verifies the checkpoint sha256
    return seg


def box_key(box: list[float]) -> str:
    """Stable key of one detector box (xyxy px, 0.01 px) shared by the local job file and the remote mask file."""
    return ",".join(f"{v:.2f}" for v in box)


def photo_boxes(pred: dict[str, Any], cfg: dict[str, Any], image_wh: tuple[int, int]) -> list[dict[str, Any]]:
    """The SAM prompts of one photo in measurement order: compound boxes, then leaflet boxes."""
    return leaflet_boxes(pred, cfg, image_wh, [cfg["compound_class"]]) + leaflet_boxes(pred, cfg, image_wh)


class _LazyMasks(dict):
    """Packed masks unpacked only when asked for (keeps memory to one full-size mask at a time)."""

    def __init__(self, packed: dict[str, np.ndarray], hw: tuple[int, int]):
        super().__init__(packed)
        self.hw = hw

    def __getitem__(self, k: str) -> np.ndarray:
        h, w = self.hw
        return np.unpackbits(super().__getitem__(k), count=h * w).reshape(h, w).astype(bool)


class _PrecomputedPredictor:
    def __init__(self, masks: dict[str, np.ndarray], scores: dict[str, float]):
        self.masks, self.scores = masks, scores

    def set_image(self, rgb: np.ndarray) -> None:
        self.shape = rgb.shape[:2]

    def predict(self, box, multimask_output):
        k = box_key([float(v) for v in box])
        if k not in self.masks:
            raise SystemExit(f"no precomputed mask for box {k} - the remote job used other boxes")
        m = self.masks[k]
        if m.shape != self.shape:
            raise SystemExit(f"precomputed mask {m.shape} != image {self.shape}")
        return m[None], np.array([self.scores[k]]), None


class PrecomputedSeg:
    """SAM masks made elsewhere (Kaggle GPU) for exactly these boxes, read back from the remote output.

    The remote job ran the same checkpoint (sha256 checked there and recorded here) with the same predictor
    settings and kept the best-scored of the 3 multimask outputs, i.e. what :func:`sam_masks` does locally."""

    precomputed = True

    def __init__(self, npz_path: Path, manifest: dict[str, Any]):
        d = np.load(npz_path)
        h, w = (int(x) for x in d["image_hw"])
        keys = [str(k) for k in d["box_keys"]]
        self.config = dict(model_name=manifest["model_name"], checkpoint_sha256=manifest["checkpoint_sha256"],
                           device=manifest["device"], torch_version=manifest["torch_version"],
                           mask_file=Path(npz_path).as_posix(),
                           mask_file_sha256=hashlib.sha256(Path(npz_path).read_bytes()).hexdigest())
        packed = d["masks_packed"]
        masks = _LazyMasks({k: packed[i] for i, k in enumerate(keys)}, (h, w))
        self.predictor = _PrecomputedPredictor(masks, {k: float(s) for k, s in zip(keys, d["best_score"])})


def sam_masks(seg, rgb: np.ndarray, boxes: list[dict[str, Any]]) -> list[tuple[np.ndarray, float]]:
    """One SAM box prompt per detector box on ONE image embedding; best-scored of the multimask outputs."""
    if getattr(seg, "precomputed", False):
        seg.predictor.set_image(rgb)
        out = []
        for b in boxes:
            masks, scores, _ = seg.predictor.predict(box=np.asarray(b["box"], np.float64), multimask_output=True)
            out.append((masks[0], float(scores[0])))
        return out
    import torch

    out = []
    with torch.inference_mode():
        seg.predictor.set_image(rgb)
        for b in boxes:
            masks, scores, _ = seg.predictor.predict(box=np.asarray(b["box"], np.float32), multimask_output=True)
            i = int(np.argmax(scores))
            out.append((np.asarray(masks[i], bool), float(scores[i])))
    return out


# --------------------------------------------------------------------------- pot scale
def _p1_value(key: str) -> Any:
    p1 = json.loads((Path(__file__).parents[1] / "state" / "p1_config.json").read_text(encoding="utf-8"))
    return p1["pot_scale"][key]["value"]


def pot_scale(image_sha256: str | None, plant_id: str | None, data_root: Path) -> Any:
    """Rim scale of ``plant_id``'s pot on the photo with RAW sha256 ``image_sha256``; no link -> unavailable."""
    from tomato3d.state import potscale as ps

    spec_path = Path(data_root) / ps.POT_SPEC_PATH.relative_to("data")
    pts_path = Path(data_root) / ps.RIM_POINTS_PATH.relative_to("data")
    if plant_id is None or image_sha256 is None:
        return ps.PotScale(UNAVAILABLE, "photo not linked to a capture record (plant_id + original sha256 needed; "
                                        "per-pot rim records are keyed by both)")
    if not spec_path.exists():
        return ps.PotScale(UNAVAILABLE, f"{ps.POT_SPEC_PATH.as_posix()} absent")
    if not pts_path.exists():
        return ps.PotScale(UNAVAILABLE, f"{ps.RIM_POINTS_PATH.as_posix()} absent - no rim points clicked "
                                        "(no rim detector exists)")
    points = ps.read_rim_points(pts_path)
    return ps.derive(points.get((plant_id, image_sha256), []), ps.spec_for(ps.read_pot_spec(spec_path), plant_id),
                     draws=int(_p1_value("monte_carlo_draws")), seed=int(_p1_value("monte_carlo_seed")),
                     accept_rel=float(_p1_value("cross_check_accept_rel")),
                     stop_rel=float(_p1_value("cross_check_stop_rel")),
                     record_ref=f"{ps.RIM_POINTS_PATH.name}#{plant_id}|{image_sha256[:12]}")


def rim_polygons(image_sha256: str, plant_id: str, data_root: Path, own: Any
                 ) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Own rim polygon (from the solved scale) and the polygons of any NEIGHBOUR pot clicked on the same photo."""
    from tomato3d.state import potscale as ps

    spec_path = Path(data_root) / ps.POT_SPEC_PATH.relative_to("data")
    pts_path = Path(data_root) / ps.RIM_POINTS_PATH.relative_to("data")
    own_poly = None
    if own.status == "solved":
        own_poly = inst.rim_polygon_px(np.asarray(own.homography), ps.spec_for(ps.read_pot_spec(spec_path),
                                                                              plant_id).outer_edge_mm)
    neighbours: dict[str, np.ndarray] = {}
    if pts_path.exists() and spec_path.exists():
        for (pid, sha) in ps.read_rim_points(pts_path):
            if sha == image_sha256 and pid != plant_id:
                sc = pot_scale(sha, pid, data_root)
                if sc.status == "solved":
                    neighbours[pid] = inst.rim_polygon_px(np.asarray(sc.homography), ps.spec_for(
                        ps.read_pot_spec(spec_path), pid).outer_edge_mm)
    return own_poly, neighbours


# --------------------------------------------------------------------------- one photo
def _shape_row(b: dict[str, Any], mask: np.ndarray, score: float) -> tuple[dict[str, Any], np.ndarray,
                                                                          dict[str, Any] | None]:
    x0, y0, x1, y1 = (int(round(v)) for v in b["box"])
    inbox = np.zeros_like(mask)
    inbox[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    total = int(mask.sum())
    outside = float((mask & ~inbox).sum() / total) if total else None
    clipped = mask & inbox                          # the detector box bounds the instance; leaks are reported
    m = measure_mask(clipped)
    row = {"class": b["cls"], "det_confidence": round(b["conf"], 4),
           "box_x0_px": round(b["box"][0], 1), "box_y0_px": round(b["box"][1], 1),
           "box_x1_px": round(b["box"][2], 1), "box_y1_px": round(b["box"][3], 1), "sam_score": round(score, 4),
           "mask_outside_box_frac": round(outside, 4) if outside is not None else UNAVAILABLE}
    if m is None:
        row.update(mask_status="empty", **{k: UNAVAILABLE for k in SHAPE_COLUMNS if k not in row})
    else:
        row.update(mask_status="sam_box_prompt", area_px=m["area_px"], length_px=round(m["length_px"], 2),
                   width_px=round(m["width_px"], 2),
                   projected_aspect=round(m["projected_aspect"], 4) if m["projected_aspect"] else UNAVAILABLE,
                   axis_deg_image=m["axis_deg_image"], centroid_x_px=round(m["centroid_px"][0], 1),
                   centroid_y_px=round(m["centroid_px"][1], 1))
    return row, clipped, m


def measure_photo(image_path: Path, out_dir: Path, *, seg: Any, link: dict[str, Any] | None = None,
                  data_root: Path | None = None, api_key: str | None = None, pose: dict[str, Any] | None = None,
                  detector_split: str = UNAVAILABLE) -> dict[str, Any]:
    """Compounds + leaflets of one photo. Without a ``link`` (reallink) nothing carries plant_id or mm."""
    from tomato3d.state.potscale import PLANE_NOTE
    from tomato3d.synth.images import imread, imwrite

    cfg = load_config()
    image_path, out_dir = Path(image_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if link is not None and file_sha != link["image_file_sha256"]:
        raise SystemExit(f"{image_path}: sha256 differs from the linked copy - relink it")
    plant_id = link["plant_id"] if link else None
    raw_sha = link["image_sha256"] if link else None
    capture_id = link["capture_id"] if link else None
    ids = dict(plant_id=plant_id or UNAVAILABLE, capture_id=capture_id or UNAVAILABLE,
               image_sha256=raw_sha or UNAVAILABLE)
    bgr = imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot decode {image_path} (HEIC: convert first, originals are never re-encoded in place)")
    h, w = bgr.shape[:2]
    pred = detect(image_path, out_dir / "detections.json", cfg, api_key)
    lboxes = leaflet_boxes(pred, cfg, (w, h))
    cboxes = leaflet_boxes(pred, cfg, (w, h), [cfg["compound_class"]])
    if [b["box"] for b in cboxes + lboxes] != [b["box"] for b in photo_boxes(pred, cfg, (w, h))]:
        raise AssertionError("prompt order drifted from photo_boxes")
    masks = sam_masks(seg, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cboxes + lboxes)
    if data_root is None:
        from tomato3d.config import get_settings

        data_root = get_settings().data_root
    scale = pot_scale(raw_sha, plant_id, data_root)
    solved = scale.status == "solved"
    rel_sd = scale.mm_per_px_sd / scale.mm_per_px if solved else None
    stem = capture_id or image_path.stem

    def mm_fields(m: dict[str, Any] | None) -> dict[str, Any]:
        base = dict(scale_status="solved" if solved else UNAVAILABLE,
                    scale_plane="rim_plane" if solved else UNAVAILABLE,
                    plane_bias="overestimates_above_rim" if solved else UNAVAILABLE)
        if solved and m is not None:
            return dict(base, **mm_with_sd(measure_mm(m["contour_px"], np.asarray(scale.homography)), rel_sd))
        return dict(base, **{k: UNAVAILABLE for k in MM_COLUMNS if k not in base})

    comp_rows, comp_objs, leaf_rows, leaf_objs, mask_store = [], [], [], [], {}
    for i, (b, (mask, score)) in enumerate(zip(cboxes, masks[:len(cboxes)]), start=1):
        cid = f"{stem}#C{i:02d}"
        row, clipped, m = _shape_row(b, mask, score)
        comp_rows.append(dict(compound_id=cid, **ids, **row, **mm_fields(m)))
        comp_objs.append(dict(id=cid, box=b["box"], mask=clipped, centroid=m["centroid_px"] if m else None))
        mask_store[cid] = clipped
    for i, (b, (mask, score)) in enumerate(zip(lboxes, masks[len(cboxes):]), start=1):
        lid = f"{stem}#L{i:03d}"
        row, clipped, m = _shape_row(b, mask, score)
        leaf_rows.append(dict(leaflet_id=lid, **ids, **row, **mm_fields(m), vis_frac=UNAVAILABLE))
        leaf_objs.append(dict(id=lid, box=b["box"], mask=clipped, centroid=m["centroid_px"] if m else None))
        mask_store[lid] = clipped

    majority = float(cfg["membership"]["box_majority_frac"]["value"])
    members = inst.membership(leaf_objs, comp_objs, majority)
    conflicts = inst.terminal_conflicts(members, {r["leaflet_id"]: r["class"] for r in leaf_rows})
    own_poly, neighbours = (rim_polygons(raw_sha, plant_id, data_root, scale) if link else (None, {}))
    attr = inst.attribute(comp_objs, leaf_objs, members, own_poly, neighbours)
    by_leaf = {m["leaflet_id"]: m for m in members}
    for r in leaf_rows:
        mem = by_leaf[r["leaflet_id"]]
        r.update(compound_id=mem["compound_id"], membership_status=mem["status"],
                 attribution=attr[r["leaflet_id"]]["attribution"],
                 attribution_reason=attr[r["leaflet_id"]]["reason"], review_status="proposed")
    for r in comp_rows:
        r.update(n_member_leaflets=sum(m["status"] == "member" and m["compound_id"] == r["compound_id"]
                                       for m in members),
                 terminal_conflict=";".join(conflicts.get(r["compound_id"], [])) or "none",
                 attribution=attr[r["compound_id"]]["attribution"],
                 attribution_reason=attr[r["compound_id"]]["reason"], review_status="proposed")

    entries = inst.save_masks(mask_store, out_dir / "masks", rel_to=out_dir)
    kind = {r["compound_id"]: "compound" for r in comp_rows} | {r["leaflet_id"]: "leaflet" for r in leaf_rows}
    for e in entries:
        e["kind"] = kind[e["instance_id"]]
    inst.write_manifest(out_dir / "masks_manifest.json",
                        dict(schema="topview-masks-1", **ids, image_path=image_path.as_posix(),
                             image_file_sha256=file_sha, image_size_px=[w, h],
                             encoding="one 8-bit PNG per instance (0/255), cropped to the mask's bounding box; "
                                      "full mask = zeros(image_size) with the crop pasted at offset_px",
                             review_status="proposed"), entries)
    for path, cols, rows in ((out_dir / "leaflets_2d.csv", CSV_COLUMNS, leaf_rows),
                             (out_dir / "compounds_2d.csv", COMPOUND_COLUMNS, comp_rows)):
        with path.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            wr.writerows(rows)
    with (out_dir / "membership.json").open("w", encoding="utf-8") as f:
        json.dump(dict(rule=cfg["membership"]["rule"], majority=majority, leaflets=members,
                       terminal_conflicts=conflicts), f, ensure_ascii=False, indent=1)

    overlay = bgr.copy()
    for o in comp_objs:
        cnts, _ = cv2.findContours(o["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 220, 255), 6)
    for r, o in zip(leaf_rows, leaf_objs):
        color = dict(inside=(0, 200, 0), outside=(0, 0, 255), ambiguous=(255, 0, 255))[r["attribution"]]
        overlay[o["mask"]] = (0.5 * overlay[o["mask"]] + 0.5 * np.array(color)).astype(np.uint8)
    if own_poly is not None:
        cv2.polylines(overlay, [np.round(own_poly).astype(np.int32)], True, (255, 255, 255), 5)
    imwrite(out_dir / "overlay.jpg", cv2.resize(overlay, (w // 3, h // 3), interpolation=cv2.INTER_AREA))

    count = lambda rows, k, v: sum(r[k] == v for r in rows)  # noqa: E731
    header = dict(
        schema=SCHEMA, image=image_path.as_posix(), image_file_sha256=file_sha, image_size_px=[w, h], **ids,
        link=link or dict(status=UNAVAILABLE, reason="no capture link: no plant_id, no original sha256, no mm"),
        detector=dict(model=cfg["detector"]["model"], v4_split=detector_split,
                      measure_min_confidence=cfg["detector"]["measure_min_confidence"],
                      measure_min_confidence_source_kind=cfg["detector"]["measure_min_confidence_source_kind"],
                      compound_threshold_note=cfg["compound_threshold_note"],
                      n_predictions=len(pred.get("predictions", [])), n_leaflet_boxes=len(lboxes),
                      n_compound_boxes=len(cboxes)),
        segmenter=dict(model=seg.config.get("model_name"), checkpoint_sha256=seg.config["checkpoint_sha256"],
                       device=seg.config.get("device", UNAVAILABLE), torch_version=seg.config.get("torch_version"),
                       mask_file=seg.config.get("mask_file"), mask_file_sha256=seg.config.get("mask_file_sha256"),
                       mask_choice=cfg["sam"]["mask_choice"],
                       clip_rule="mask clipped to its detector box; the clipped-off share is mask_outside_box_frac"),
        scale=dict(status=scale.status, reason=scale.reason, convention=scale.convention,
                   mm_per_px_centre=scale.mm_per_px, mm_per_px_sd=scale.mm_per_px_sd, source_ref=scale.source_ref,
                   plane_note=PLANE_NOTE, sd_note="scale 1-sigma propagated first-order; segmentation error excluded"),
        pose=pose or dict(status=UNAVAILABLE),
        attribution=dict(own_rim_polygon_px=own_poly.round(1).tolist() if own_poly is not None else UNAVAILABLE,
                         neighbour_pots=sorted(neighbours), rule=inst.__doc__.split("* **Attribution**")[1].strip()),
        notes=["areas come from masks, never from boxes", "every row is review_status=proposed until a person confirms",
               "vis_frac unavailable on real photos (no amodal truth)",
               "length/width = minimum-area rectangle of the projected mask, not the midrib length",
               "axis_deg_image: second-moment long-axis angle mod 180 in image coordinates (x right, y down); tip "
               "direction is not resolved from the mask"],
        counts=dict(compounds=len(comp_rows), leaflets=len(leaf_rows),
                    by_class={c: count(leaf_rows, "class", c) for c in cfg["leaflet_classes"]},
                    leaflets_with_mm=sum(r["area_mm2"] != UNAVAILABLE for r in leaf_rows),
                    empty_masks=count(leaf_rows + comp_rows, "mask_status", "empty"),
                    membership={s: count(leaf_rows, "membership_status", s) for s in inst.MEMBERSHIP},
                    attribution_leaflets={a: count(leaf_rows, "attribution", a) for a in inst.ATTRIBUTIONS},
                    attribution_compounds={a: count(comp_rows, "attribution", a) for a in inst.ATTRIBUTIONS},
                    terminal_conflicts=len(conflicts)),
    )
    (out_dir / "measure2d.json").write_text(json.dumps(header, ensure_ascii=False, indent=1), encoding="utf-8")
    return dict(header=header, rows=leaf_rows, compounds=comp_rows)


# --------------------------------------------------------------------------- all real top captures
def measure_package(package: Path, out: Path, *, sam_config: Path, project_root: Path, data_root: Path,
                    heic_python: str | None, detector_split: dict[str, str] | None = None,
                    api_key: str | None = None, only: list[str] | None = None,
                    masks_from: Path | None = None, detections_from: Path | None = None) -> list[dict[str, Any]]:
    """X1/X2: every real top capture of a P1 package, linked to its record, measured, with its camera pose.

    ``masks_from``: a remote mask output (``manifest.json`` + ``<plant>.npz``, see :func:`prepare_remote`) used
    instead of running SAM here; ``detections_from``: a folder with ``<plant>/detections.json`` to reuse."""
    from tomato3d.topview import reallink as rl

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    remote = json.loads((Path(masks_from) / "manifest.json").read_text(encoding="utf-8")) if masks_from else None
    seg = None if remote else load_sam(sam_config)
    photos = []
    for cap in rl.top_captures(package):
        if only and cap["source_plant_id"] not in only:
            continue
        row = dict(plant_id=cap["source_plant_id"], capture_id=cap["capture_id"], image_sha256=cap["image_sha256"])
        try:
            link = rl.link_top_capture(cap, project_root)
        except rl.GridRefusal as exc:
            photos.append(dict(row, link_status="refused", link_reason=str(exc)))
            continue
        scale = pot_scale(link["image_sha256"], link["plant_id"], data_root)
        exif = rl.read_focal(Path(project_root) / link["raw_path"], heic_python) if heic_python else {}
        pose = rl.photo_pose(link, scale, exif) if heic_python else dict(status=UNAVAILABLE,
                                                                         reason="no HEIC interpreter given")
        split = (detector_split or {}).get(link["plant_id"], UNAVAILABLE)
        pdir = out / link["plant_id"]
        if detections_from and not (pdir / "detections.json").exists():
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "detections.json").write_bytes((Path(detections_from) / link["plant_id"] / "detections.json")
                                                   .read_bytes())
        if remote:
            job = remote["photos"].get(link["plant_id"])
            if job is None or job["image_file_sha256"] != link["image_file_sha256"]:
                photos.append(dict(row, link_status="linked", link_reason="remote masks missing or made on another file"))
                continue
            seg = PrecomputedSeg(Path(masks_from) / job["npz"], remote)
        r = measure_photo(Path(project_root) / link["image_path"], pdir, seg=seg, link=link,
                          data_root=data_root, api_key=api_key, pose=pose, detector_split=split)
        c = r["header"]["counts"]
        b = pose.get("bases", {})
        photos.append(dict(row, image_path=link["image_path"], image_file_sha256=link["image_file_sha256"],
                           link_status="linked", link_reason=link["grid_check"], detector_split_v4=split,
                           scale_status=scale.status, scale_reason=scale.reason or "",
                           mm_per_px=scale.mm_per_px, mm_per_px_sd=scale.mm_per_px_sd, pose_status=pose["status"],
                           depth_mm_frame_width=b.get("frame_width", {}).get("pot_centre_depth_mm"),
                           depth_mm_frame_diagonal=b.get("frame_diagonal", {}).get("pot_centre_depth_mm"),
                           tilt_deg_frame_width=b.get("frame_width", {}).get("tilt_from_vertical_deg"),
                           axis_ratio_self_check_frame_width=b.get("frame_width", {}).get("axis_ratio_self_check"),
                           n_compounds=c["compounds"], n_leaflets=c["leaflets"], n_leaflets_mm=c["leaflets_with_mm"],
                           n_leaflets_inside=c["attribution_leaflets"]["inside"],
                           n_leaflets_ambiguous=c["attribution_leaflets"]["ambiguous"],
                           n_leaflets_outside=c["attribution_leaflets"]["outside"]))
        (out / link["plant_id"] / "pose.json").write_text(json.dumps(pose, indent=1), encoding="utf-8")
    with (out / "photos.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=PHOTO_COLUMNS, restval=UNAVAILABLE)
        wr.writeheader()
        wr.writerows([{k: (UNAVAILABLE if v is None else v) for k, v in p.items()} for p in photos])
    return photos


def prepare_remote(package: Path, out: Path, *, project_root: Path, api_key: str | None = None,
                   only: list[str] | None = None) -> dict[str, Any]:
    """Link every real top capture, fetch / reuse its detections, and write the SAM job for a remote GPU:
    ``out/images/<plant>_top.jpg`` (the linked copy, byte-identical) + ``out/jobs.json`` (boxes in measurement order,
    with :func:`box_key`). Detections are cached in ``out/detections/<plant>/detections.json``. No key, no record,
    no mm leaves this machine - only the photos and boxes."""
    import shutil

    from tomato3d.topview import reallink as rl

    cfg = load_config()
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    jobs, refused = {}, {}
    for cap in rl.top_captures(package):
        if only and cap["source_plant_id"] not in only:
            continue
        try:
            link = rl.link_top_capture(cap, project_root)
        except rl.GridRefusal as exc:
            refused[cap["source_plant_id"]] = str(exc)
            continue
        img = Path(project_root) / link["image_path"]
        pred = detect(img, out / "detections" / link["plant_id"] / "detections.json", cfg, api_key)
        w, h = link["size_px"]
        boxes = photo_boxes(pred, cfg, (w, h))
        shutil.copyfile(img, out / "images" / img.name)
        if rl.sha256_file(out / "images" / img.name) != link["image_file_sha256"]:
            raise SystemExit(f"copy of {img.name} differs from the linked file")
        jobs[link["plant_id"]] = dict(image=f"images/{img.name}", image_file_sha256=link["image_file_sha256"],
                                      image_size_px=[w, h], boxes=[dict(key=box_key(b["box"]), cls=b["cls"],
                                                                        box=b["box"]) for b in boxes])
    doc = dict(schema="topview-sam-job-1", detector=cfg["detector"]["model"],
               measure_min_confidence=cfg["detector"]["measure_min_confidence"],
               order="compound boxes then leaflet boxes, each most confident first (measure2d.photo_boxes)",
               mask_choice=cfg["sam"]["mask_choice"], photos=jobs, refused=refused)
    (out / "jobs.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="T1: real top photos -> per-compound / per-leaflet 2-D measures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("package", help="all real top captures of a P1 package, linked to their records")
    p.add_argument("--package", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sam-config", default=None, help="SAM 2 config JSON with checkpoint_sha256")
    p.add_argument("--heic-python", default=os.environ.get("TOMATO3D_HEIC_PYTHON"),
                   help="interpreter with pillow_heif, for the EXIF focal length (pose)")
    p.add_argument("--detector-split", default=None, help="JSON {plant_id: v4 split}")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--masks-from", default=None, help="remote SAM output folder (manifest.json + <plant>.npz)")
    p.add_argument("--detections-from", default=None, help="folder with <plant>/detections.json to reuse")
    j = sub.add_parser("prepare-remote", help="link + detect + write the SAM job for a remote GPU")
    j.add_argument("--package", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--only", nargs="*", default=None)
    q = sub.add_parser("photo", help="one unlinked photo: pixels only, never mm")
    q.add_argument("--image", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--sam-config", required=True)
    args = ap.parse_args(argv)
    from tomato3d.config import get_settings

    if args.cmd == "package":
        split = json.loads(Path(args.detector_split).read_text(encoding="utf-8"))["by_plant"] if args.detector_split \
            else None
        photos = measure_package(Path(args.package), Path(args.out),
                                 sam_config=Path(args.sam_config) if args.sam_config else None,
                                 project_root=Path.cwd(), data_root=get_settings().data_root,
                                 heic_python=args.heic_python, detector_split=split, only=args.only,
                                 masks_from=Path(args.masks_from) if args.masks_from else None,
                                 detections_from=Path(args.detections_from) if args.detections_from else None)
        print(json.dumps(dict(photos=len(photos), linked=sum(p["link_status"] == "linked" for p in photos),
                              leaflets=sum(p.get("n_leaflets", 0) for p in photos),
                              leaflets_mm=sum(p.get("n_leaflets_mm", 0) for p in photos))))
    elif args.cmd == "prepare-remote":
        d = prepare_remote(Path(args.package), Path(args.out), project_root=Path.cwd(), only=args.only)
        print(json.dumps(dict(photos=len(d["photos"]), boxes=sum(len(v["boxes"]) for v in d["photos"].values()),
                              refused=d["refused"])))
    else:
        r = measure_photo(Path(args.image), Path(args.out), seg=load_sam(Path(args.sam_config)))
        print(json.dumps(dict(counts=r["header"]["counts"], scale=r["header"]["scale"]["status"],
                              scale_reason=r["header"]["scale"]["reason"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
