"""Leaflet contour library from human-reviewed leaflet masks (read-only inputs).

Selection (user decision 2026-10-05):

* ``human_full`` - the cross-view review answered ``full`` for that mask;
* ``geometric_proxy`` - no human answer, and: boundary contact with other accepted masks of
  the same photo <= ``max_contact``, a single dominant component, solidity >= ``min_solidity``,
  and not touching the image border;
* a human ``partial`` answer always excludes the mask.

Every contour records ``source`` (``view18_reviewed`` here; ``compound_closeup`` /
``topside_capture`` for Addendum A photos), plant, image, mask hash and selection reason.
Plants in ``exclude_plants`` (the generative test set) never contribute.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from tomato3d.synth.images import imread, imwrite

from tomato3d.synth.leaflet_surface import normalize_contour

SOURCE_VIEW18 = "view18_reviewed"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def human_visibility(decision_files: Iterable[Path]) -> dict[str, str]:
    """Mask id -> latest human visibility answer (``full``/``partial``) across files."""
    answers: dict[str, tuple[str, str]] = {}
    for path in decision_files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for d in data.get("decisions", []):
            vis = d.get("visibility")
            if not isinstance(vis, dict):
                continue
            at = d.get("decided_at") or data.get("saved_at") or ""
            for mask_id, answer in vis.items():
                if mask_id not in answers or at >= answers[mask_id][0]:
                    answers[mask_id] = (at, answer)
    return {k: v[1] for k, v in answers.items()}


def _load_masks(export_dir: Path, observations: list[dict], downscale: int) -> dict[str, dict[str, Any]]:
    out = {}
    for o in observations:
        path = export_dir / o["mask_file"]
        mask = imread((path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        small = cv2.resize((mask > 0).astype(np.uint8), None, fx=1 / downscale, fy=1 / downscale,
                           interpolation=cv2.INTER_NEAREST)
        out[o["id"]] = dict(obs=o, path=path, full=mask > 0, small=small.astype(bool))
    return out


def _metrics(item: dict[str, Any], others: list[np.ndarray]) -> dict[str, float]:
    full = item["full"].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(full, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    largest = int(np.argmax(areas)) + 1 if n > 1 else 0
    dominant = float(areas.max() / max(1, areas.sum()))
    blob = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea) if contours else np.zeros((0, 1, 2), np.int32)
    area = float(cv2.contourArea(contour)) if len(contour) else 0.0
    hull = cv2.convexHull(contour) if len(contour) else contour
    solidity = area / max(1.0, float(cv2.contourArea(hull))) if len(contour) else 0.0
    h, w = full.shape
    x, y, bw, bh = cv2.boundingRect(contour) if len(contour) else (0, 0, 0, 0)
    border = x <= 2 or y <= 2 or x + bw >= w - 3 or y + bh >= h - 3
    small = item["small"].astype(np.uint8)
    ring = cv2.dilate(small, np.ones((3, 3), np.uint8)).astype(bool) & ~small.astype(bool)
    union = np.zeros_like(ring)
    for o in others:
        union |= o
    contact = float((ring & union).sum() / max(1, ring.sum()))
    return dict(dominant_fraction=dominant, solidity=solidity, touches_border=bool(border),
                contact_fraction=contact, area_px=area, contour=contour[:, 0, :])


def build_library(sources: dict[str, Path], decision_files: list[Path], *, n_points: int,
                  max_contact: float, min_solidity: float, min_dominant: float, downscale: int,
                  exclude_plants: Iterable[str] = (), use_human_visibility: bool = True,
                  exclude_border: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (selected contour records, per-mask audit rows).

    ``use_human_visibility=False`` selects on the geometric thresholds alone (human answers are recorded but neither
    force nor veto a mask) - Addendum G.1 filter D; ``exclude_border=False`` drops the off-border requirement.
    """
    vis = human_visibility(decision_files)
    excluded = set(exclude_plants)
    library, audit = [], []
    for plant_id, export_dir in sorted(sources.items()):
        export_dir = Path(export_dir)
        accepted = json.loads((export_dir / "accepted.json").read_text(encoding="utf-8"))
        obs = [o for o in accepted["observations"] if o.get("organ_class") == "leaflet"]
        by_image: dict[str, list[dict]] = {}
        for o in obs:
            by_image.setdefault(o["image_file"], []).append(o)
        for image_file, group in sorted(by_image.items()):
            masks = _load_masks(export_dir, group, downscale)
            for mask_id, item in masks.items():
                others = [m["small"] for k, m in masks.items() if k != mask_id]
                m = _metrics(item, others)
                answer = vis.get(mask_id)
                geometric = (m["contact_fraction"] <= max_contact and m["solidity"] >= min_solidity
                             and m["dominant_fraction"] >= min_dominant
                             and not (exclude_border and m["touches_border"]))
                if plant_id in excluded:
                    reason = "excluded_test_plant"
                elif not use_human_visibility:
                    reason = "geometric_only" if geometric else "proxy_failed"
                elif answer == "partial":
                    reason = "human_partial"
                elif answer == "full":
                    reason = "human_full"
                elif (m["contact_fraction"] <= max_contact and m["solidity"] >= min_solidity
                      and m["dominant_fraction"] >= min_dominant and not m["touches_border"]):
                    reason = "geometric_proxy"
                else:
                    reason = "proxy_failed"
                row = dict(mask_id=mask_id, plant_id=plant_id, image_file=image_file, human_visibility=answer,
                           selection=reason, **{k: v for k, v in m.items() if k != "contour"})
                audit.append(row)
                if reason not in ("human_full", "geometric_proxy", "geometric_only") or len(m["contour"]) < 10:
                    continue
                norm = normalize_contour(m["contour"], n_points)
                library.append(dict(
                    contour_id=f"{SOURCE_VIEW18}/{mask_id}", source=SOURCE_VIEW18, plant_id=plant_id,
                    image_file=image_file, mask_id=mask_id, mask_sha256=_sha256(item["path"]),
                    visibility_source=reason, view_group=item["obs"].get("image_provenance", {}).get("view_group"),
                    projection="perspective_photo_not_flattened", orientation_method=norm["orientation_method"],
                    aspect=norm["aspect"], widest_fraction=norm["widest_fraction"], length_px=norm["length_px"],
                    solidity=m["solidity"], contact_fraction=m["contact_fraction"],
                    points=np.round(norm["points"], 5).tolist()))
    return library, audit


def load_library(path: Path) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data["contours"]
    return [np.asarray(r["points"], float) for r in records], records


def preview_sheet(records: list[dict[str, Any]], out_path: Path, *, cell_px: int = 96, cols: int = 12) -> None:
    rows = (len(records) + cols - 1) // cols
    sheet = np.full((max(1, rows) * cell_px, cols * cell_px, 3), 255, np.uint8)
    colors = {"human_full": (40, 140, 40), "geometric_proxy": (160, 100, 20)}
    for i, r in enumerate(records):
        pts = np.asarray(r["points"])
        cy, cx = divmod(i, cols)
        scale = 0.8 * cell_px
        px = np.c_[pts[:, 0] * scale + 0.1 * cell_px, -pts[:, 1] * scale + 0.5 * cell_px]
        px += np.array([cx * cell_px, cy * cell_px])
        cv2.polylines(sheet, [px.astype(np.int32)], True, colors.get(r["visibility_source"], (0, 0, 0)), 1,
                      cv2.LINE_AA)
    imwrite((out_path), sheet)


# --------------------------------------------------------------------------- Addendum C, step C0
SOURCE_ACCEPTED_PROJECTED = "accepted_mask_projected"


def _view_geometry(pose_file: Path | None, look_at_board_mm) -> dict[str, dict[str, Any]]:
    """Image file -> camera elevation/distance from a capture.json (solved board poses only)."""
    if pose_file is None:
        return {}
    data = json.loads(Path(pose_file).read_text(encoding="utf-8"))
    K = np.asarray(data["intrinsics"]["camera_matrix"], float)
    f_px = 0.5 * (K[0, 0] + K[1, 1])
    out = {}
    for f in data["frames"]:
        if f.get("rvec_rad") is None:
            continue
        R = cv2.Rodrigues(np.asarray(f["rvec_rad"], float))[0]
        center = -R.T @ np.asarray(f["tvec_mm"], float)
        rel = center - np.asarray(look_at_board_mm, float)
        out[f["image_file"]] = dict(
            camera_elevation_deg=float(np.degrees(np.arctan2(rel[2], np.hypot(rel[0], rel[1])))),
            camera_distance_mm=float(np.linalg.norm(rel)),
            pose_reprojection_error_px=f.get("reprojection_error_px"),
            passes_pose_gate=bool(f.get("passes_pose_gate")),
            mm_per_px_at_plant_axis=float(np.linalg.norm(rel) / f_px),
            scale_method="camera distance to the pot axis / focal length (leaf depth not known; approximate)")
    return out


def unit_area_points(points: np.ndarray) -> np.ndarray:
    """Tier B form (Revision C.3): centred, principal-axis aligned, scaled to unit area."""
    p = np.asarray(points, float)
    p = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    p = p @ vt.T
    x, y = p[:, 0], p[:, 1]
    area = abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return p / np.sqrt(area) if area > 0 else p


def harvest_accepted(sources: dict[str, Path], pose_files: dict[str, Path | None], decision_files: list[Path], *,
                     n_points: int, downscale: int, look_at_board_mm) -> tuple[list[dict], list[dict]]:
    """C0: every accepted leaflet mask -> normalised projected contour (no selection, attributes kept).

    Revision C.3: pose is not required. Tier A (board pose -> viewing angle and approximate mm scale)
    and Tier B (mask only: shape and aspect, unit area) both enter the pool; nothing is dropped for a
    missing pose.

    Outlines are *projected*: photographed at an angle and curled, i.e. foreshortened and curl-shrunk
    versions of the planar shape. Selection is left to the consumer through the recorded attributes.
    """
    vis = human_visibility(decision_files)
    records, skipped = [], []
    for plant_id, export_dir in sorted(sources.items()):
        export_dir = Path(export_dir)
        geometry = _view_geometry(pose_files.get(plant_id), look_at_board_mm)
        accepted = json.loads((export_dir / "accepted.json").read_text(encoding="utf-8"))
        obs = [o for o in accepted["observations"] if o.get("organ_class") == "leaflet"]
        by_image: dict[str, list[dict]] = {}
        for o in obs:
            by_image.setdefault(o["image_file"], []).append(o)
        for image_file, group in sorted(by_image.items()):
            masks = _load_masks(export_dir, group, downscale)
            for o in group:
                item = masks.get(o["id"])
                if item is None:
                    skipped.append(dict(mask_id=o["id"], plant_id=plant_id, reason="mask_file_unreadable"))
                    continue
                m = _metrics(item, [v["small"] for k, v in masks.items() if k != o["id"]])
                if len(m["contour"]) < 10:
                    skipped.append(dict(mask_id=o["id"], plant_id=plant_id, reason="contour_too_short"))
                    continue
                norm = normalize_contour(m["contour"], n_points)
                view_code = Path(image_file).stem.rsplit("_", 1)[-1]
                geo = geometry.get(image_file)
                tier = dict(tier="A" if geo else "B", pose_available=geo is not None, scale_available=geo is not None)
                if geo:
                    mmpp = geo["mm_per_px_at_plant_axis"]
                    c = m["contour"].astype(float)
                    tier.update(contour_mm=np.round((c - c.mean(axis=0)) * mmpp, 3).tolist(),
                                area_mm2_approx=float(m["area_px"] * mmpp ** 2))
                records.append(dict(**tier,
                    contour_id=f"{SOURCE_ACCEPTED_PROJECTED}/{o['id']}", source=SOURCE_ACCEPTED_PROJECTED,
                    plant_id=plant_id, image_file=image_file, view_code=view_code, mask_id=o["id"],
                    mask_sha256=_sha256(item["path"]), mask_area_px=float(m["area_px"]),
                    view_group=o.get("image_provenance", {}).get("view_group"),
                    viewing_geometry=geometry.get(image_file, dict(status="pose_unavailable")),
                    human_visibility=vis.get(o["id"]), solidity=m["solidity"], contact_fraction=m["contact_fraction"],
                    dominant_fraction=m["dominant_fraction"], touches_border=m["touches_border"],
                    projection="perspective_photo_not_flattened", correction_status="uncorrected",
                    orientation_method=norm["orientation_method"], aspect=norm["aspect"],
                    widest_fraction=norm["widest_fraction"], length_px=norm["length_px"],
                    points=np.round(norm["points"], 5).tolist(),
                    points_unit_area=np.round(unit_area_points(norm["points"]), 5).tolist()))
    return records, skipped


def load_pool(paths: list[Path], sources: list[str] | None = None) -> tuple[list[np.ndarray], list[dict]]:
    """Pooled contour library from several files (shape library, not per-plant), optional source filter."""
    pts, recs = [], []
    for p in paths:
        for r in json.loads(Path(p).read_text(encoding="utf-8"))["contours"]:
            if sources and r["source"] not in sources:
                continue
            pts.append(np.asarray(r["points"], float))
            recs.append(r)
    return pts, recs


# --------------------------------------------------------------------------------------
# Library versions: repair of self-intersecting contours (P1-3, 2026-10-15)
# --------------------------------------------------------------------------------------

REPAIR_RASTER_PX = 2048
REPAIR_METHOD = ("even-odd raster (the triangulator's own inside rule) -> outer contour re-extracted -> shapely "
                 "simplify(preserve_topology=True) -> resampled to the library's point count -> validity re-checked")


def repair_contour(points: np.ndarray, n_points: int, tolerances=(0.0, 0.001, 0.002, 0.004, 0.008)) -> tuple[np.ndarray, dict]:
    """A valid simple outline for one normalised contour (x 0..1 base->tip). Same method as the user's v2 leaflet pool
    (topology-preserving simplification of a re-extracted outline). The smallest tolerance that yields a valid,
    simple, resampled polygon is used and reported; the frame (base at x=0, tip at x=1) is kept."""
    from shapely.geometry import Polygon

    from tomato3d.synth.leaflet_surface import resample_closed

    p = np.asarray(points, float)
    lo, hi = p.min(axis=0), p.max(axis=0)
    scale = (REPAIR_RASTER_PX - 8) / float(max(hi - lo))
    px = np.round((p - lo) * scale * 16).astype(np.int32) + 4 * 16
    mask = np.zeros((REPAIR_RASTER_PX, REPAIR_RASTER_PX), np.uint8)
    cv2.fillPoly(mask, [px], 1, lineType=cv2.LINE_8, shift=4)          # even-odd fill = the triangulator's rule
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    outline = (outline - 4) / scale + lo
    for tol in tolerances:
        poly = Polygon(outline).simplify(tol, preserve_topology=True) if tol else Polygon(outline)
        ring = np.asarray(poly.exterior.coords)[:-1]
        res = resample_closed(ring, n_points)
        if _signed_area(res) < 0:          # library convention (normalize_contour): counter-clockwise
            res = res[::-1]
        # keep the frame: the raster trims a fraction of a pixel at the tips, so restore the original length (uniform
        # scale about the base, aspect unchanged)
        s = float(np.ptp(p[:, 0])) / max(float(np.ptp(res[:, 0])), 1e-9)
        res = np.c_[(res[:, 0] - res[:, 0].min()) * s + p[:, 0].min(), res[:, 1] * s]
        if Polygon(res).is_valid:
            start = int(np.argmin(res[:, 0]))
            res = np.roll(res, -start, axis=0)
            return res, dict(tolerance=tol, valid=True, length_rescale=s)
    return np.asarray(points, float), dict(tolerance=None, valid=False)


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def repair_library(path: Path, out_path: Path, version: str) -> dict:
    """NEW library version with every invalid contour repaired; the source file is never modified. Indices are kept
    (contour_index in stored scenes keeps its meaning)."""
    from shapely.geometry import Polygon

    src = Path(path)
    data = json.loads(src.read_text(encoding="utf-8"))
    n_points = len(data["contours"][0]["points"])
    report = []
    for i, rec in enumerate(data["contours"]):
        pts = np.asarray(rec["points"], float)
        if Polygon(pts).is_valid:
            continue
        fixed, info = repair_contour(pts, n_points)
        before, after = Polygon(pts).buffer(0), Polygon(fixed)
        report.append(dict(index=i, contour_id=rec["contour_id"], plant_id=rec.get("plant_id"), repaired=info["valid"],
                           tolerance=info["tolerance"], length_rescale=info.get("length_rescale"),
                           area_before=float(abs(_signed_area(pts))), area_before_buffer0=float(before.area),
                           area_after=float(after.area),
                           length_before=float(np.ptp(pts[:, 0])), length_after=float(np.ptp(fixed[:, 0])),
                           width_before=float(np.ptp(pts[:, 1])), width_after=float(np.ptp(fixed[:, 1]))))
        if info["valid"]:
            rec["points"] = fixed.tolist()
            rec["aspect"] = float(np.ptp(fixed[:, 1]))
            rec["repair"] = dict(version=version, method=REPAIR_METHOD, tolerance=info["tolerance"],
                                 parent_sha256=_sha256(src))
    data["version"] = version
    data["parent"] = dict(path=src.as_posix(), sha256=_sha256(src))
    data["repairs"] = report
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return dict(out=out_path.as_posix(), report=report, n_points=n_points)
