"""Record detected board evidence against existing diagnostic poses, without solving.

python -m tools.charuco_capture_evidence --capture build/.../capture.preview.json
  --out build/.../board_evidence
Existing poses and quality gates are copied unchanged; all geometry stays unvalidated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tomato3d.calib.board import _build_detector, build_charuco_board, sheet_to_board_mm
from tomato3d.config import get_settings
from tomato3d.io.schemas import BoardSpec
from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json, write_rows
from tools.leaf_light.reconstruction import project


def resolve_input(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def verify_file(value: str, sha256: str, base: Path) -> Path:
    path = resolve_input(value, base)
    if not sha256 or file_hash(path) != sha256:
        raise StudyError(f"Source hash mismatch: {path.name}")
    return path


def residuals(points_board_mm: np.ndarray | None, pixels_xy_px: np.ndarray,
              frame: dict, intrinsics: dict, matching_resolution: bool) -> tuple[list | None, list | None]:
    """Compare measured pixels with the existing pose; never re-estimate that pose."""
    if points_board_mm is None or not matching_resolution or frame.get("pose_status") != "solved":
        return None, None
    if frame.get("rvec_rad") is None or frame.get("tvec_mm") is None:
        return None, None
    rvec_rad, tvec_mm = np.asarray(frame["rvec_rad"], float), np.asarray(frame["tvec_mm"], float)
    if rvec_rad.shape != (3,) or tvec_mm.shape != (3,) or not np.isfinite([rvec_rad, tvec_mm]).all():
        raise StudyError("Malformed stored camera pose")
    camera_points_mm = points_board_mm @ cv2.Rodrigues(rvec_rad)[0].T + tvec_mm
    if np.any(camera_points_mm[:, 2] <= 0):
        return None, None
    fitted_px = project(points_board_mm, frame, intrinsics)
    if not np.isfinite(fitted_px).all():
        return None, None
    errors_px = np.linalg.norm(fitted_px - pixels_xy_px, axis=1)
    return fitted_px.tolist(), errors_px.tolist()


def _rms(values_px: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values_px)))) if values_px else None


def detect_evidence(image: np.ndarray | None, frame: dict, intrinsics: dict,
                    detector: Any, board: Any, spec: BoardSpec) -> dict:
    """Retain actual detector outputs, including unmatched IDs and failed frames."""
    evidence = dict(schema_version="charuco-frame-evidence-1", detection_status="image_unreadable",
                    reason=None, source_image_sha256=frame["sha256"], detected_marker_ids=[], detected_corner_ids=[],
                    n_markers_detected=0, n_charuco_corners_detected=0, markers=[], corners=[],
                    existing_pose_charuco_rms_px=None, existing_pose_marker_rms_px=None,
                    pose_reestimated=False, validated_metric=False)
    if image is None:
        evidence["reason"] = "Working image could not be decoded"
        return evidence
    resolution_matches = image.shape[1] == intrinsics["image_width_px"] and image.shape[0] == intrinsics["image_height_px"]
    evidence["image_width_px"], evidence["image_height_px"] = int(image.shape[1]), int(image.shape[0])
    evidence["calibration_resolution_matches"] = resolution_matches
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        charuco_pixels, charuco_ids, marker_pixels, marker_ids = detector.detectBoard(gray)
    except cv2.error as exc:
        evidence.update(detection_status="detector_error", reason=str(exc))
        return evidence
    marker_map_mm = {int(marker_id): sheet_to_board_mm(vertices, spec)
                     for marker_id, vertices in zip(board.getIds(), board.getObjPoints())}
    charuco_map_mm = sheet_to_board_mm(board.getChessboardCorners(), spec)
    marker_residuals_px: list[float] = []
    corner_residuals_px: list[float] = []
    marker_id_values = [] if marker_ids is None else np.asarray(marker_ids).reshape(-1).tolist()
    corner_id_values = [] if charuco_ids is None else np.asarray(charuco_ids).reshape(-1).tolist()
    quads = [] if marker_pixels is None else list(marker_pixels)
    corners = [] if charuco_pixels is None else list(charuco_pixels)
    if len(quads) != len(marker_id_values) or len(corners) != len(corner_id_values):
        evidence.update(detection_status="detector_error", reason="Inconsistent detector IDs and pixel arrays")
        return evidence
    for detection_index, (marker_id, quad) in enumerate(zip(marker_id_values, quads)):
        pixel_quad = np.asarray(quad, dtype=float).reshape(-1, 2)
        if pixel_quad.shape != (4, 2) or not np.isfinite(pixel_quad).all():
            raise StudyError("Malformed detected marker quadrilateral")
        marker_id = int(marker_id)
        vertices_mm = marker_map_mm.get(marker_id)
        fitted_px, errors_px = residuals(vertices_mm, pixel_quad, frame, intrinsics, resolution_matches)
        evidence["markers"].append(dict(marker_id=marker_id, detection_index=detection_index,
            pixels_xy_px=pixel_quad.tolist(), board_vertices_mm=None if vertices_mm is None else vertices_mm.tolist(),
            matched_to_configured_board=vertices_mm is not None, projected_xy_px=fitted_px, residuals_px=errors_px))
        if errors_px is not None:
            marker_residuals_px.extend(errors_px)
    for detection_index, (corner_id, corner) in enumerate(zip(corner_id_values, corners)):
        pixel_xy_px = np.asarray(corner, dtype=float).reshape(-1, 2)
        if pixel_xy_px.shape != (1, 2) or not np.isfinite(pixel_xy_px).all():
            raise StudyError("Malformed detected ChArUco corner")
        corner_id = int(corner_id)
        point_mm = charuco_map_mm[corner_id:corner_id+1] if 0 <= corner_id < len(charuco_map_mm) else None
        fitted_px, errors_px = residuals(point_mm, pixel_xy_px, frame, intrinsics, resolution_matches)
        evidence["corners"].append(dict(corner_id=corner_id, detection_index=detection_index,
            pixel_xy_px=pixel_xy_px[0].tolist(), board_xyz_mm=None if point_mm is None else point_mm[0].tolist(),
            matched_to_configured_board=point_mm is not None,
            projected_xy_px=None if fitted_px is None else fitted_px[0],
            reprojection_error_px=None if errors_px is None else errors_px[0]))
        if errors_px is not None:
            corner_residuals_px.extend(errors_px)
    evidence.update(detected_marker_ids=[int(i) for i in marker_id_values],
                    detected_corner_ids=[int(i) for i in corner_id_values],
                    n_markers_detected=len(marker_id_values), n_charuco_corners_detected=len(corner_id_values),
                    existing_pose_marker_rms_px=_rms(marker_residuals_px),
                    existing_pose_charuco_rms_px=_rms(corner_residuals_px),
                    detection_status="detected" if marker_id_values or corner_id_values else "no_board_detected")
    if not resolution_matches:
        evidence.update(detection_status="image_resolution_mismatch", reason="Actual detections retained; no residuals against mismatched calibration")
    return evidence


def _overlay(image: np.ndarray, evidence: dict, path: Path) -> None:
    canvas = image.copy()
    if evidence["markers"]:
        cv2.aruco.drawDetectedMarkers(canvas, [np.asarray(item["pixels_xy_px"], np.float32)[None]
            for item in evidence["markers"]], np.asarray(evidence["detected_marker_ids"], np.int32)[:, None])
    if evidence["corners"]:
        cv2.aruco.drawDetectedCornersCharuco(canvas,
            np.asarray([item["pixel_xy_px"] for item in evidence["corners"]], np.float32)[:, None],
            np.asarray(evidence["detected_corner_ids"], np.int32)[:, None], (255, 0, 255))
    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        raise StudyError("Could not encode detection overlay")
    encoded.tofile(path)


def export_evidence(capture_path: Path, out: Path, *, overlays: bool = False) -> dict:
    capture_path = capture_path.resolve()
    capture = read_json(capture_path)
    if capture.get("schema_version") != "diagnostic-preview-capture-1":
        raise StudyError("Expected diagnostic-preview-capture-1")
    frames = capture["frames"]
    if len({frame["image_file"] for frame in frames}) != len(frames):
        raise StudyError("Duplicate image_file")
    calibration_path = verify_file(capture["intrinsics_path"], capture["intrinsics_sha256"], capture_path.parent)
    if read_json(calibration_path) != capture["intrinsics"]:
        raise StudyError("Embedded calibration differs from hashed calibration")
    board_path = get_settings().board_spec_path.resolve()
    board_hash = file_hash(board_path)
    spec = BoardSpec.model_validate_json(board_path.read_text(encoding="utf-8"))
    if capture.get("board_variant_id") != spec.variant_id:
        raise StudyError("Capture board_variant_id does not match configured board")
    if capture.get("board_spec_sha256") not in (None, board_hash):
        raise StudyError("Recorded board-spec hash does not match configured board")
    paths = {frame["image_file"]: verify_file(frame["image_path"], frame["sha256"], capture_path.parent) for frame in frames}
    board = build_charuco_board(spec)
    detector = _build_detector(board)
    new_capture = deepcopy(capture)
    new_capture.update(validated_metric=False, board_spec_path=str(board_path), board_spec_sha256=board_hash,
                       board_evidence_provenance=dict(source_capture_path=str(capture_path),
                           source_capture_sha256=file_hash(capture_path), detector="tomato3d.calib.board._build_detector",
                           opencv_version=cv2.__version__, pose_reestimated=False, calibration_reestimated=False,
                           notice="Board-spec hash records this evidence run. Original poses and gates remain unchanged."))
    # References become absolute for the relocated output while all pose values stay byte-for-byte equivalent as JSON values.
    new_capture["intrinsics_path"] = str(calibration_path)
    folder = output_dir(out)
    frame_rows, marker_rows, corner_rows = [], [], []
    for frame in new_capture["frames"]:
        source_path = paths[frame["image_file"]]
        try:
            image = cv2.imdecode(np.fromfile(source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            image = None
        evidence = detect_evidence(image, frame, capture["intrinsics"], detector, board, spec)
        frame["image_path"] = str(source_path)
        frame["board_evidence"] = evidence
        common = dict(image_file=frame["image_file"], image_sha256=frame["sha256"],
                      board_variant_id=spec.variant_id, board_spec_sha256=board_hash,
                      calibration_sha256=capture["intrinsics_sha256"], coordinate_frame="board", validated_metric=False)
        frame_rows.append(dict(common, detection_status=evidence["detection_status"], reason=evidence["reason"],
                               existing_pose_status=frame["pose_status"], existing_passes_pose_gate=frame.get("passes_pose_gate"),
                               stored_pose_reprojection_error_px=frame.get("reprojection_error_px"),
                               n_markers_detected=evidence["n_markers_detected"],
                               n_charuco_corners_detected=evidence["n_charuco_corners_detected"],
                               detected_marker_ids=";".join(map(str, evidence["detected_marker_ids"])),
                               detected_corner_ids=";".join(map(str, evidence["detected_corner_ids"])),
                               existing_pose_charuco_rms_px=evidence["existing_pose_charuco_rms_px"],
                               existing_pose_marker_rms_px=evidence["existing_pose_marker_rms_px"]))
        for marker in evidence["markers"]:
            for index, pixel_xy_px in enumerate(marker["pixels_xy_px"]):
                xyz_mm = marker["board_vertices_mm"][index] if marker["board_vertices_mm"] is not None else [None] * 3
                fitted_px = marker["projected_xy_px"][index] if marker["projected_xy_px"] is not None else [None] * 2
                marker_rows.append(dict(common, detection_index=marker["detection_index"], marker_id=marker["marker_id"],
                    quad_vertex_index=index, x_px=pixel_xy_px[0], y_px=pixel_xy_px[1],
                    x_board_mm=xyz_mm[0], y_board_mm=xyz_mm[1], z_board_mm=xyz_mm[2],
                    projected_x_px=fitted_px[0], projected_y_px=fitted_px[1],
                    reprojection_error_px=marker["residuals_px"][index] if marker["residuals_px"] is not None else None,
                    matched_to_configured_board=marker["matched_to_configured_board"]))
        for corner in evidence["corners"]:
            xyz_mm = corner["board_xyz_mm"] if corner["board_xyz_mm"] is not None else [None] * 3
            fitted_px = corner["projected_xy_px"] if corner["projected_xy_px"] is not None else [None] * 2
            corner_rows.append(dict(common, detection_index=corner["detection_index"], corner_id=corner["corner_id"],
                x_px=corner["pixel_xy_px"][0], y_px=corner["pixel_xy_px"][1],
                x_board_mm=xyz_mm[0], y_board_mm=xyz_mm[1], z_board_mm=xyz_mm[2],
                projected_x_px=fitted_px[0], projected_y_px=fitted_px[1], reprojection_error_px=corner["reprojection_error_px"],
                matched_to_configured_board=corner["matched_to_configured_board"]))
        if overlays and image is not None:
            _overlay(image, evidence, folder / (Path(frame["image_file"]).stem + "_board_evidence.jpg"))
    write_json(folder / "capture.board_evidence.json", new_capture)
    for filename, entries in (("frames.csv", frame_rows), ("marker_observations.csv", marker_rows), ("corner_observations.csv", corner_rows)):
        write_rows(folder / filename, entries, tuple(entries[0]) if entries else ("status",))
    metadata = dict(schema_version="charuco-capture-evidence-1", board_variant_id=spec.variant_id,
        board_spec_path=str(board_path), board_spec_sha256=board_hash,
        source_capture_path=str(capture_path), source_capture_sha256=file_hash(capture_path),
        output_capture_sha256=file_hash(folder / "capture.board_evidence.json"),
        calibration_path=str(calibration_path), calibration_sha256=capture["intrinsics_sha256"],
        frame_count=len(frame_rows), marker_detection_count=len(marker_rows) // 4, charuco_corner_detection_count=len(corner_rows),
        pose_reestimated=False, calibration_reestimated=False, deep_learning_run=False, validated_metric=False,
        detector="tomato3d.calib.board._build_detector", opencv_version=cv2.__version__,
        physical_print_verified=spec.physical_print_verified,
        notice="Actual detector observations. IDs are not inferred from board configuration; unmatched IDs retain pixels with unavailable board coordinates. CSV frame rows retain empty/failed/unusable frames. Board XYZ uses core sheet_to_board_mm. Residuals use stored poses and are not a new gate or accuracy validation.")
    write_json(folder / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlays", action="store_true", help="Save actual marker/corner detection overlays")
    args = parser.parse_args()
    metadata = export_evidence(args.capture, args.out, overlays=args.overlays)
    print(f"Recorded {metadata['frame_count']} frames, {metadata['marker_detection_count']} marker observations, "
          f"{metadata['charuco_corner_detection_count']} ChArUco corners; poses unchanged")


if __name__ == "__main__":
    main()
