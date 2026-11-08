"""Export traceable, unvalidated depths for manually labeled preview landmarks.

No reconstruction, calibration, correspondence inference, or deep learning runs here.
Camera-axis Z is signed distance along the optical axis; camera range is Euclidean
distance to the optical centre. Both use the existing board-to-camera pose in mm.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

from tomato3d.config import get_settings
from tools.leaf_light.common import (
    StudyError, file_hash, output_dir, read_json, rows, write_json, write_rows,
)
from tools.leaf_light.reconstruction import project


MANUAL_KINDS = {"manual", "manual_candidate", "manual_observation", "manual_candidate_unreviewed"}
POINT_COLUMNS = ("image_file", "organ_id", "leaflet_id", "point_id", "x_px", "y_px", "boundary_index")


def camera_coordinates(point_board_mm: np.ndarray, frame: dict) -> tuple[np.ndarray, float]:
    """Apply the stored board-to-camera transform without another frame conversion."""
    rvec_rad = np.asarray(frame["rvec_rad"], dtype=float)
    tvec_mm = np.asarray(frame["tvec_mm"], dtype=float)
    if rvec_rad.shape != (3,) or tvec_mm.shape != (3,) or not np.isfinite([rvec_rad, tvec_mm]).all():
        raise StudyError("Malformed stored camera pose")
    point_camera_mm = cv2.Rodrigues(rvec_rad)[0] @ point_board_mm + tvec_mm
    return point_camera_mm, float(np.linalg.norm(point_camera_mm))


def _path(value: str, relative_to: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _verified_path(value: str, expected_hash: str, relative_to: Path) -> tuple[Path, str]:
    path = _path(value, relative_to)
    actual_hash = file_hash(path)
    if not expected_hash or actual_hash != expected_hash:
        raise StudyError(f"Source hash mismatch: {path.name}")
    return path, actual_hash


def _unique(items: list[dict], key: str, description: str) -> dict:
    indexed = {item[key]: item for item in items}
    if len(indexed) != len(items):
        raise StudyError(f"Duplicate {description}")
    return indexed


def board_evidence_fields(frame: dict) -> dict:
    """Link leaf observations to measured detector evidence for this exact photo."""
    evidence = frame.get('board_evidence')
    result = dict(board_detection_status='unavailable', detected_marker_ids=None,
                  detected_corner_ids=None, existing_pose_charuco_rms_px=None)
    if evidence is None:
        return result
    if evidence.get('source_image_sha256') != frame['sha256']:
        raise StudyError('Board evidence belongs to a different photograph')
    result['board_detection_status'] = evidence['detection_status']
    for field, entries, id_field in (('detected_marker_ids', 'markers', 'marker_id'),
                                      ('detected_corner_ids', 'corners', 'corner_id')):
        ids = evidence[field]
        observed = [entry[id_field] for entry in evidence[entries]]
        if any(type(i) is not int or i < 0 for i in ids) or sorted(ids) != sorted(observed):
            raise StudyError('Board detection ID summary differs from observations')
        result[field] = (';'.join(map(str, ids)) if ids else 'none_detected'
                         if evidence['detection_status'] in ('detected', 'no_board_detected') else None)
    result['existing_pose_charuco_rms_px'] = evidence.get('existing_pose_charuco_rms_px')
    return result


def compile_records(capture: dict, preview: dict, point_rows: list[dict], provenance: dict) -> dict:
    """Pure record calculation. File/hash checks are performed by export_records."""
    if capture.get("schema_version") != "diagnostic-preview-capture-1":
        raise StudyError("Expected diagnostic-preview-capture-1")
    if preview.get("schema_version") != "manual-leaf-preview-result-1":
        raise StudyError("Expected manual-leaf-preview-result-1")
    if preview.get("geometry_source") != "manual_multiview_landmark_interpolation":
        raise StudyError("Expected explicitly manual preview geometry")
    if not capture.get("board_variant_id"):
        raise StudyError("Missing board_variant_id")
    frames = _unique(capture["frames"], "image_file", "frame")
    leaves = _unique(preview["geometry"], "leaflet_id", "leaflet")
    diagnostics = _unique(preview.get("diagnostics", []), "leaflet_id", "leaflet diagnostic")
    vertices: dict[str, np.ndarray] = {}
    for leaf_id, leaf in leaves.items():
        xyz_mm = np.asarray(leaf["vertices_mm"], dtype=float)
        if xyz_mm.ndim != 2 or xyz_mm.shape[1] != 3 or not len(xyz_mm) or not np.isfinite(xyz_mm).all():
            raise StudyError(f"Invalid preview vertices: {leaf_id}")
        if not set(leaf["source_images"]).issubset(frames):
            raise StudyError("Unknown geometry source image")
        vertices[leaf_id] = xyz_mm

    common = dict(provenance, plant_id=capture["plant_id"], session_id=capture["session_id"],
                  board_variant_id=capture["board_variant_id"], coordinate_frame="board",
                  validated_metric=False, suitable_for_ppfd=False)
    observations: list[dict] = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    identities: dict[tuple[str, str, str], int] = {}
    vertex_identities: dict[tuple[str, int], str] = {}
    seen: set[tuple[str, str, str, str]] = set()
    excluded_nonmanual = 0
    intrinsics = capture["intrinsics"]
    for candidate in point_rows:
        kind = candidate.get("observation_kind") or candidate.get("annotation_kind") or "manual_candidate"
        if kind not in MANUAL_KINDS:
            excluded_nonmanual += 1
            continue
        image_file, leaf_id = candidate["image_file"], candidate["leaflet_id"]
        if image_file not in frames:
            raise StudyError("Unknown manually labeled image")
        frame, leaf = frames[image_file], leaves.get(leaf_id)
        if leaf and (candidate["organ_id"] != leaf["organ_id"] or image_file not in leaf["source_images"]):
            raise StudyError("Manual row does not match preview source observations")
        try:
            xy_px = np.asarray([candidate["x_px"], candidate["y_px"]], dtype=float)
            vertex_index = int(candidate["boundary_index"])
        except (ValueError, TypeError) as exc:
            raise StudyError("Malformed manual landmark") from exc
        if not np.isfinite(xy_px).all() or not (0 <= xy_px[0] < intrinsics["image_width_px"] and
                                               0 <= xy_px[1] < intrinsics["image_height_px"]):
            raise StudyError("Manual landmark outside original image")
        if vertex_index < 0 or (leaf and vertex_index >= len(vertices[leaf_id])):
            raise StudyError("Invalid boundary_index")
        identity = (candidate["organ_id"], leaf_id, candidate["point_id"])
        key = (image_file, *identity)
        if key in seen:
            raise StudyError("Duplicate manual observation")
        seen.add(key)
        if identity in identities and identities[identity] != vertex_index:
            raise StudyError("Landmark index changes between views")
        vertex_key = (leaf_id, vertex_index)
        if vertex_key in vertex_identities and vertex_identities[vertex_key] != candidate["point_id"]:
            raise StudyError("Multiple point identities assigned to one vertex")
        identities[identity] = vertex_index
        vertex_identities[vertex_key] = candidate["point_id"]
        xyz_mm = vertices[leaf_id][vertex_index] if leaf else None
        record = dict(common, **board_evidence_fields(frame), image_file=image_file, image_path=frame["image_path"],
                      image_sha256=frame["sha256"], organ_id=identity[0], leaflet_id=leaf_id,
                      point_id=identity[2], boundary_index=vertex_index,
                      observation_kind="manual_candidate", review_status=(leaf or {}).get(
                          "review_status", diagnostics.get(leaf_id, {}).get("review_status", "unavailable")),
                      outline_completeness=(leaf or {}).get('outline_completeness', 'unavailable'),
                      correspondence_status=(leaf or {}).get('correspondence_status', 'unavailable'),
                      observed_x_px=float(xy_px[0]), observed_y_px=float(xy_px[1]),
                      x_board_mm=None, y_board_mm=None, z_board_mm=None,
                      camera_axis_z_mm=None, camera_range_mm=None,
                      projected_x_px=None, projected_y_px=None, reprojection_error_px=None,
                      pose_status=frame["pose_status"], passes_pose_gate=frame.get("passes_pose_gate"),
                      pose_reprojection_error_px=frame.get("reprojection_error_px"),
                      n_charuco_corners=frame.get("n_corners_detected"),
                      depth_status="geometry_unavailable")
        if xyz_mm is not None:
            record.update(x_board_mm=float(xyz_mm[0]), y_board_mm=float(xyz_mm[1]), z_board_mm=float(xyz_mm[2]))
            record["depth_status"] = "camera_pose_unavailable"
            if frame["pose_status"] == "solved" and frame.get("rvec_rad") is not None and frame.get("tvec_mm") is not None:
                camera_mm, range_mm = camera_coordinates(xyz_mm, frame)
                record.update(camera_axis_z_mm=float(camera_mm[2]), camera_range_mm=range_mm)
                record["depth_status"] = "behind_camera_unavailable" if camera_mm[2] <= 0 else "experimental_unvalidated"
                if camera_mm[2] > 0:
                    fitted_px = project(xyz_mm[None, :], frame, intrinsics)[0]
                    if not np.isfinite(fitted_px).all():
                        raise StudyError("Non-finite landmark reprojection")
                    record.update(projected_x_px=float(fitted_px[0]), projected_y_px=float(fitted_px[1]),
                                  reprojection_error_px=float(np.linalg.norm(fitted_px - xy_px)))
        observations.append(record)
        grouped[identity].append(record)

    landmarks: list[dict] = []
    limit_px = capture["quality"]["max_point_error_px"]
    for identity, entries in grouped.items():
        residuals_px = [r["reprojection_error_px"] for r in entries if r["reprojection_error_px"] is not None]
        residuals_complete = len(residuals_px) == len(entries) and len(entries) >= 2
        maximum_px = max(residuals_px) if residuals_px else None
        passes_point_gate = (maximum_px <= limit_px if limit_px is not None else None) if residuals_complete else None
        gates_pass = all(r["passes_pose_gate"] is True for r in entries)
        for record in entries:
            record.update(point_max_reprojection_error_px=maximum_px, passes_point_gate=passes_point_gate,
                          point_gate_active=limit_px is not None, passes_all_source_pose_gates=gates_pass,
                          manual_view_count=len(entries), residuals_complete=residuals_complete)
        first = entries[0]
        landmarks.append(dict(common, organ_id=identity[0], leaflet_id=identity[1], point_id=identity[2],
                              boundary_index=identities[identity],
                              x_board_mm=first["x_board_mm"], y_board_mm=first["y_board_mm"], z_board_mm=first["z_board_mm"],
                              review_status=first["review_status"], manual_view_count=len(entries),
                              source_images=";".join(r["image_file"] for r in entries),
                              point_max_reprojection_error_px=maximum_px, passes_point_gate=passes_point_gate,
                              point_gate_active=limit_px is not None, passes_all_source_pose_gates=gates_pass,
                              residuals_complete=residuals_complete,
                              measurement_status="experimental_unvalidated" if first["x_board_mm"] is not None else "unavailable"))

    frame_records: list[dict] = []
    counts = Counter(record["image_file"] for record in observations)
    for name, frame in frames.items():
        frame_records.append(dict(common, **board_evidence_fields(frame), image_file=name, image_path=frame["image_path"], image_sha256=frame["sha256"],
                                  pose_status=frame["pose_status"], passes_pose_gate=frame.get("passes_pose_gate"),
                                  pose_reprojection_error_px=frame.get("reprojection_error_px"),
                                  pose_failure_reason=frame.get("failure_reason"),
                                  n_charuco_corners=frame.get("n_corners_detected"),
                                  manual_observation_count=counts[name],
                                  labeling_status="manual_candidates" if counts[name] else "no_manual_labels",
                                  camera_depth_status="experimental_pose_available" if frame["pose_status"] == "solved" else "unavailable"))
    leaf_records: list[dict] = []
    leaf_ids = sorted(set(leaves) | set(diagnostics) | {r["leaflet_id"] for r in observations})
    for leaf_id in leaf_ids:
        leaf = leaves.get(leaf_id)
        xyz_mm = vertices.get(leaf_id)
        leaf_records.append(dict(common, leaflet_id=leaf_id, organ_id=leaf["organ_id"] if leaf else None,
                                 vertex_count=len(xyz_mm) if xyz_mm is not None else 0,
                                 min_vertex_board_height_mm=float(xyz_mm[:, 2].min()) if xyz_mm is not None else None,
                                 max_vertex_board_height_mm=float(xyz_mm[:, 2].max()) if xyz_mm is not None else None,
                                 extent_kind="existing_preview_vertices_only",
                                 source_images=";".join(leaf["source_images"]) if leaf else None,
                                 review_status=(leaf or diagnostics.get(leaf_id, {})).get("review_status", "unavailable"),
                                 outline_completeness=(leaf or {}).get('outline_completeness', 'unavailable'),
                                 correspondence_status=(leaf or {}).get('correspondence_status', 'unavailable'),
                                 measurement_status="experimental_unvalidated" if xyz_mm is not None else "unavailable"))
    return dict(observations=observations, landmarks=landmarks, frames=frame_records, leaves=leaf_records,
                excluded_nonmanual_rows=excluded_nonmanual)


def export_records(capture_path: Path, preview_path: Path, points_path: Path, out: Path) -> dict:
    """Verify source lineage and write new output only; never modify the preview."""
    capture_path, preview_path, points_path = (p.resolve() for p in (capture_path, preview_path, points_path))
    capture, preview = read_json(capture_path), read_json(preview_path)
    capture_hash = file_hash(capture_path)
    if preview.get("capture_sha256") != capture_hash:
        raise StudyError("Preview belongs to a different capture")
    calibration_path, calibration_hash = _verified_path(capture["intrinsics_path"], capture["intrinsics_sha256"], capture_path.parent)
    if read_json(calibration_path) != capture["intrinsics"]:
        raise StudyError("Embedded calibration differs from hashed calibration")
    for frame in capture["frames"]:
        _verified_path(frame["image_path"], frame["sha256"], capture_path.parent)
    board_path = get_settings().board_spec_path.resolve()
    board = read_json(board_path)
    if board["variant_id"] != capture["board_variant_id"]:
        raise StudyError("Capture board_variant_id does not match configured board")
    if capture.get('board_spec_sha256') and capture['board_spec_sha256'] != file_hash(board_path):
        raise StudyError('Configured board changed after evidence capture')
    provenance = dict(capture_path=str(capture_path), capture_sha256=capture_hash,
                      preview_path=str(preview_path), preview_sha256=file_hash(preview_path),
                      points_path=str(points_path), points_sha256=file_hash(points_path),
                      calibration_path=str(calibration_path), calibration_sha256=calibration_hash,
                      current_board_spec_path=str(board_path), current_board_spec_sha256=file_hash(board_path))
    records = compile_records(capture, preview, rows(points_path, POINT_COLUMNS), provenance)
    folder = output_dir(out)
    for name in ("observations", "landmarks", "frames", "leaves"):
        entries = records[name]
        write_rows(folder / f"{name}.csv", entries, tuple(entries[0]) if entries else ("status",))
    metadata = dict(schema_version="manual-leaf-depth-records-1", **provenance,
                    board_variant_id=capture["board_variant_id"], validated_metric=False,
                    suitable_for_ppfd=False, deep_learning_run=False,
                    counts={name: len(records[name]) for name in ("observations", "landmarks", "frames", "leaves")},
                    excluded_nonmanual_rows=records["excluded_nonmanual_rows"],
                    current_board_physical_print_verified=board.get("physical_print_verified"),
                    current_board_square_size_mm=board["square_size_mm"],
                    calibration_rms_px=capture["intrinsics"].get("rms_reprojection_error_px"),
                    quality_gates=capture["quality"],
                    unresolved_observations=preview.get("unresolved_observations", []),
                    board_evidence_provenance=capture.get('board_evidence_provenance'),
                    definitions={
                        "board_xyz_mm": "Existing estimated vertices in board frame; origin inner corner 0, +Z away from board.",
                        "camera_axis_z_mm": "Signed Z of R * point_board_mm + t; optical-axis depth, not Euclidean range. Behind-camera values are diagnostic only.",
                        "camera_range_mm": "Euclidean norm of R * point_board_mm + t; distance to optical centre, not leaf length.",
                        "observations.csv": "Manual candidate pixels only. projected_x/y are fitted locations of the same candidate, never new observed labels.",
                        "frames.csv": "All capture frames including unsolved and unlabeled ones. Unavailable detector IDs are not invented from board configuration.",
                        "leaves.csv": "Height extrema of existing vertices only; not whole-leaf bounds, length, or error bars.",
                        "metric_status": "Always unvalidated. Passing an individual residual gate does not validate scale, camera pose, identity or leaf shape.",
                        "source_hashes": "Image hashes refer to capture image_path files (working PNGs), not automatically to original phone files.",
                        "board_spec": "Current configuration checked at export; a historical board-spec hash is additionally verified when supplied by the evidence capture.",
                        "detected_ids": "Actual per-photo marker/corner detection IDs, never inferred from the configured board. none_detected denotes an empty observed set; unavailable denotes absent detector evidence. Evidence joins by photo hash, not only filename.",
                        "point_identity": "Candidate physical correspondence comes from the supplied manual CSV and remains unreviewed as indicated.",
                    })
    write_json(folder / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "preview", "points", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = export_records(args.capture, args.preview, args.points, args.out)
    print(f"Exported {result['counts']}; validated_metric=False")


if __name__ == "__main__":
    main()
