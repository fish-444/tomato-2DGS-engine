"""Semi-manual leaf surface reconstruction with fixed ChArUco camera poses.

Boundary point identities are manually marked in images; they are not 3D priors.
Every variant is rebuilt only from its own visible points and contours.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

from tomato3d.io.schemas import LEAF_SURFACES_CSV_COLUMNS
from .annotations import POINT_COLUMNS, CONTOUR_COLUMNS
from .common import (StudyError, digest, file_hash, load_protocol, number, output_dir,
                     read_json, rows, variants, write_json, write_rows)


def triangulate(observations: list[tuple[dict, np.ndarray]], intrinsics: dict, p: dict) -> tuple[np.ndarray, float]:
    if len(observations) < 2:
        raise StudyError("fewer_than_two_views")
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=float)
    distortion = np.asarray(intrinsics["dist_coeffs"], dtype=float)
    design, centers = [], []
    for frame, xy in observations:
        rotation = cv2.Rodrigues(np.asarray(frame["rvec_rad"], dtype=float))[0]
        translation = np.asarray(frame["tvec_mm"], dtype=float)
        projection = np.column_stack((rotation, translation))
        u, v = cv2.undistortPoints(np.asarray(xy, dtype=float).reshape(1, 1, 2), matrix, distortion).reshape(2)
        design.extend((u * projection[2] - projection[0], v * projection[2] - projection[1]))
        centers.append(-rotation.T @ translation)
    _, _, vt = np.linalg.svd(np.asarray(design))
    if abs(vt[-1, 3]) < 1e-12:
        raise StudyError("point_at_infinity")
    point_mm = vt[-1, :3] / vt[-1, 3]
    rays = point_mm - np.asarray(centers)
    lengths = np.linalg.norm(rays, axis=1)
    if np.any(lengths < 1e-9):
        raise StudyError("point_at_camera")
    rays /= lengths[:, None]
    # Collinear rays in either direction cannot constrain depth reliably.
    angles = np.degrees(np.arccos(np.clip(np.abs(rays @ rays.T), 0, 1)))
    if angles.max() < p["min_ray_angle_deg"]:
        raise StudyError("insufficient_parallax")
    residuals = []
    for frame, xy in observations:
        rotation = cv2.Rodrigues(np.asarray(frame["rvec_rad"], dtype=float))[0]
        if (rotation @ point_mm + frame["tvec_mm"])[2] <= 0:
            raise StudyError("point_behind_camera")
        projected = project(point_mm[None], frame, intrinsics)[0]
        residuals.append(np.linalg.norm(projected - xy))
    error_px = float(np.sqrt(np.mean(np.square(residuals))))
    if not np.isfinite(point_mm).all() or max(residuals) > p["max_point_error_px"]:
        raise StudyError("reprojection_error")
    return point_mm, error_px


def project(points_mm: np.ndarray, frame: dict, intrinsics: dict) -> np.ndarray:
    return cv2.projectPoints(points_mm, np.asarray(frame["rvec_rad"], dtype=float),
                             np.asarray(frame["tvec_mm"], dtype=float),
                             np.asarray(intrinsics["camera_matrix"], dtype=float),
                             np.asarray(intrinsics["dist_coeffs"], dtype=float))[0].reshape(-1, 2)


def inside(polygon: np.ndarray, points: np.ndarray, tolerance: float = 0) -> bool:
    return all(cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, xy)), True)
               >= -tolerance for xy in points)


def reconstruct_variant(capture: dict, selection: dict, point_rows: list[dict],
                        contour_rows: list[dict], p: dict) -> tuple[list[dict], dict]:
    selected = set(selection["image_files"])
    frames = {f["image_file"]: f for f in capture["frames"] if f["image_file"] in selected}
    if len(frames) != len(selected):
        raise StudyError("Unknown selected image")
    if any(f["pose_status"] != "solved" for f in frames.values()):
        return [], {"status": "failed", "reason": "selected_pose_failed", "leaves": []}
    grouped = defaultdict(list)
    contours = defaultdict(list)
    known = {f["image_file"] for f in capture["frames"]}
    seen = set()
    for row in point_rows:
        if row["image_file"] not in known:
            raise StudyError("Annotation references unknown image")
        if row["image_file"] not in selected:
            continue
        identity = tuple(row[k] for k in ("image_file", "organ_id", "leaflet_id", "point_id"))
        if identity in seen:
            raise StudyError(f"Duplicate point observation: {identity}")
        seen.add(identity)
        x, y = number(row["x_px"], "x_px"), number(row["y_px"], "y_px")
        intr = capture["intrinsics"]
        if not 0 <= x < intr["image_width_px"] or not 0 <= y < intr["image_height_px"]:
            raise StudyError("Point is outside the calibrated image")
        grouped[(row["organ_id"], row["leaflet_id"])].append(row)
    for row in contour_rows:
        if row["image_file"] not in known:
            raise StudyError("Contour references unknown image")
        if row["image_file"] in selected:
            contours[(row["organ_id"], row["leaflet_id"], row["image_file"])].append(row)
    output, diagnostics = [], []
    for (organ, leaflet), observations in grouped.items():
        diag = {"organ_id": organ, "leaflet_id": leaflet, "status": "failed"}
        try:
            by_point = defaultdict(list)
            for row in observations:
                by_point[row["point_id"]].append(row)
            counts = {int(r["boundary_count"]) for r in observations}
            if len(counts) != 1 or next(iter(counts)) < 3:
                raise StudyError("boundary_count_inconsistent")
            boundary_count = next(iter(counts))
            boundary, xyz, point_errors = {}, [], []
            for point_id, obs in sorted(by_point.items()):
                indices = {int(r["boundary_index"]) for r in obs}
                if len(indices) != 1:
                    raise StudyError("boundary_index_inconsistent")
                index = next(iter(indices))
                if index < -1 or index >= boundary_count:
                    raise StudyError("boundary_index_out_of_range")
                point, error = triangulate([(frames[r["image_file"]], np.array([float(r["x_px"]), float(r["y_px"])])) for r in obs], capture["intrinsics"], p)
                if index >= 0:
                    if index in boundary:
                        raise StudyError("duplicate_boundary_index")
                    boundary[index] = len(xyz)
                xyz.append(point)
                point_errors.append(error)
            if set(boundary) != set(range(boundary_count)):
                raise StudyError("boundary_not_fully_observed")
            points_mm = np.asarray(xyz)
            _, _, vt = np.linalg.svd(points_mm - points_mm.mean(axis=0))
            uv = (points_mm - points_mm.mean(axis=0)) @ vt[:2].T
            polygon = uv[[boundary[i] for i in range(boundary_count)]]
            leaf_contours = []
            for name, frame in frames.items():
                cr = contours.get((organ, leaflet, name), [])
                if cr:
                    cr = sorted(cr, key=lambda r: int(r["vertex_index"]))
                    if len(cr) < 3 or [int(r["vertex_index"]) for r in cr] != list(range(len(cr))):
                        raise StudyError("invalid_contour_indices")
                    leaf_contours.append((frame, np.array([[float(r["x_px"]), float(r["y_px"])] for r in cr])))
            if len(leaf_contours) < 2:
                raise StudyError("fewer_than_two_visible_contours")
            boundary_mm = points_mm[[boundary[i] for i in range(boundary_count)]]
            for frame, poly in leaf_contours:
                if not inside(poly, project(boundary_mm, frame, capture["intrinsics"]), p["contour_tolerance_px"]):
                    raise StudyError("contour_conflicts_with_boundary:" + frame["image_file"] +
                                     "; export full visible contours only, not a polygon bridging skipped points")
            triangles = []
            for ids in Delaunay(uv).simplices:
                tri = points_mm[ids]
                samples_mm = np.vstack((tri, tri.mean(axis=0), (tri + np.roll(tri, 1, axis=0)) / 2))
                samples_uv = (samples_mm - points_mm.mean(axis=0)) @ vt[:2].T
                if not inside(polygon, samples_uv, 1e-5):
                    continue
                if all(inside(poly, project(samples_mm, frame, capture["intrinsics"]), p["contour_tolerance_px"])
                       for frame, poly in leaf_contours):
                    if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-9:
                        triangles.append(tri)
            if not triangles:
                raise StudyError("no_contour_supported_triangles")
            for i, tri in enumerate(triangles):
                row = dict(plant_id=capture["plant_id"], session_id=capture["session_id"],
                           geometry_variant_id=selection["variant_id"], geometry_source="reconstructed",
                           organ_class="leaf", organ_id=organ, leaflet_id=leaflet,
                           triangle_id=f"{organ}_{leaflet}_{i}", normal_convention="unknown")
                row.update({f"v{k}_{axis}_mm": float(tri[k, j]) for k in range(3) for j, axis in enumerate("xyz")})
                output.append(row)
            diag.update(status="reconstructed", n_points=len(xyz), n_triangles=len(triangles),
                        max_point_rms_px=max(point_errors))
        except (StudyError, QhullError, np.linalg.LinAlgError) as exc:
            diag["reason"] = str(exc)
        diagnostics.append(diag)
    return output, {"status": "reconstructed" if output else "failed", "leaves": diagnostics,
                    "n_failed_leaves": sum(d["status"] == "failed" for d in diagnostics),
                    "limitation": "Piecewise linear leaf surfaces; hidden surfaces are absent; upper/lower side not inferred."}


def reconstruct(capture_path: Path, points_path: Path | None, contours_path: Path | None,
                protocol_path: Path, out: Path, worklog_path: Path | None = None,
                annotation_sets_path: Path | None = None) -> dict:
    p, protocol_hash = load_protocol(protocol_path)
    capture = read_json(capture_path)
    if capture["protocol_sha256"] != protocol_hash:
        raise StudyError("Capture uses a different frozen protocol")
    # Source originals cannot change between capture inspection and reconstruction.
    for frame in capture["frames"]:
        if frame.get("sha256") and file_hash(Path(frame["image_path"])) != frame["sha256"]:
            raise StudyError("An original image changed after capture inspection")
    worklog = {}
    if worklog_path is not None:
        for row in rows(worklog_path, ("variant_id", "capture_seconds", "annotation_seconds", "correction_seconds", "scope", "operator")):
            if row["variant_id"] in worklog:
                raise StudyError("Duplicate worklog variant")
            if row["scope"] not in ("independent_subset", "shared_image_annotations") or not row["operator"]:
                raise StudyError("Worklog requires operator and explicit work-sharing scope")
            values = {key: number(row[key], key, missing=True) for key in ("capture_seconds", "annotation_seconds", "correction_seconds")}
            if any(v is not None and v < 0 for v in values.values()):
                raise StudyError("Work times cannot be negative")
            worklog[row["variant_id"]] = dict(values, work_scope=row["scope"], operator=row["operator"])
        if set(worklog) - {v["variant_id"] for v in variants(p)}:
            raise StudyError("Worklog contains an unplanned variant")
    selections = variants(p, capture["frames"])
    annotation_sets = {}
    if annotation_sets_path:
        packages = read_json(annotation_sets_path)["variants"]
        annotation_sets = {item["variant_id"]: item for item in packages}
        if len(annotation_sets) != len(packages) or set(annotation_sets) != {v["variant_id"] for v in selections}:
            raise StudyError("Independent annotation sets must cover every planned variant exactly once")
    elif points_path is None or contours_path is None:
        raise StudyError("Supply independent --annotation-sets or both --points and --contours")
    points = rows(points_path, POINT_COLUMNS) if points_path and not annotation_sets else []
    contours = rows(contours_path, CONTOUR_COLUMNS) if contours_path and not annotation_sets else []
    blockers = list(capture["physical_blockers"])
    if capture["data_kind"] == "observed" and not annotation_sets:
        blockers.append("independent_subset_annotation_provenance_missing")
    folder = output_dir(out)
    records = []
    for selection in selections:
        annotation_provenance = None
        if annotation_sets:
            package = annotation_sets[selection["variant_id"]]
            if package.get("source") != "independent_selected_images" or not package.get("operator"):
                raise StudyError("Each annotation set needs an operator and independent_selected_images provenance")
            if (len(package["reviewed_image_files"]) != len(selection["image_files"]) or
                    set(package["reviewed_image_files"]) != set(selection["image_files"])):
                raise StudyError("Annotation provenance includes excluded or missing images")
            point_file = (annotation_sets_path.parent / package["points"]).resolve()
            contour_file = (annotation_sets_path.parent / package["contours"]).resolve()
            points, contours = rows(point_file, POINT_COLUMNS), rows(contour_file, CONTOUR_COLUMNS)
            if any(r["image_file"] not in selection["image_files"] for r in points + contours):
                raise StudyError("Independent annotation CSV contains an excluded image")
            annotation_provenance = dict(package, points_sha256=file_hash(point_file), contours_sha256=file_hash(contour_file))
        started = perf_counter()
        surfaces, diagnostics = reconstruct_variant(capture, selection, points, contours, p)
        filename = f"{selection['variant_id']}.csv"
        write_rows(folder / filename, surfaces, LEAF_SURFACES_CSV_COLUMNS)
        record = dict(selection, geometry_file=filename, geometry_sha256=file_hash(folder / filename),
                            reconstruction_seconds=perf_counter() - started,
                            capture_seconds=None, annotation_seconds=None, correction_seconds=None,
                            work_scope="unavailable", operator="unavailable", **diagnostics)
        record.update(worklog.get(selection["variant_id"], {}))
        record["annotation_provenance"] = annotation_provenance
        records.append(record)
    result = {"protocol_sha256": protocol_hash, "data_kind": capture["data_kind"],
              "plant_id": capture["plant_id"], "session_id": capture["session_id"],
              "physical_blockers": blockers, "capture_path": str(capture_path.resolve()),
              "capture_sha256": file_hash(capture_path), "points_sha256": file_hash(points_path) if points_path else None,
              "contours_sha256": file_hash(contours_path) if contours_path else None,
              "annotation_sets_sha256": file_hash(annotation_sets_path) if annotation_sets_path else None,
              "worklog_sha256": file_hash(worklog_path) if worklog_path else None, "variants": records}
    write_json(folder / "reconstruction.json", result)
    return result
