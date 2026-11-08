"""Strict IO and reproducible study protocol. Outputs stay under project build/."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
UNAVAILABLE = "unavailable"
# D19: shared physical unit spelling; legacy frozen inputs must be refused, not rewritten.
DLI_UNIT = "mol/m2/day"


class StudyError(ValueError):
    """An input cannot support the requested research operation."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def output_dir(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(BUILD.resolve()) or path == BUILD.resolve():
        raise StudyError("Output must be a new subdirectory of project build/.")
    if path.exists():
        raise StudyError(f"Output already exists; use a new run directory: {path}")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def rows(path: Path, required: tuple[str, ...] = ()) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise StudyError(f"{path.name}: required columns: {required}")
        result = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in result):
        raise StudyError(f"{path.name}: malformed CSV row")
    return result


def write_rows(path: Path, records: list[dict], columns: tuple | list) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for record in records:
            writer.writerow({k: UNAVAILABLE if v is None else v for k, v in record.items()})


def number(value: Any, label: str, *, missing: bool = False) -> float | None:
    if value is None or value == UNAVAILABLE or value == "":
        if missing:
            return None
        raise StudyError(f"{label}: unavailable")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyError(f"{label}: not numeric") from exc
    if not math.isfinite(result):
        raise StudyError(f"{label}: non-finite number")
    return result


def template(capture: dict) -> dict:
    """Approved research defaults, not fabricated physical measurements."""
    return {
        "schema_version": "leaf-light-study-1", "study_id": "tomato_leaf_light",
        "planned_duration_weeks": 6,
        "evaluation_week_by_slot": {"visit_1": 3, "visit_2": 4, "visit_3": 5},
        "capture_device": "ordinary_smartphone_rgb",
        "side_view_count": capture["side_view_count_pilot"],
        "top_positions": list(capture["top_view_positions_mm"]),
        "operational_top_view": capture["operational_top_view"],
        "view_counts": [41, 37, 19, 13, 10, 7, 5, 3, 2],
        "development_plants": [f"DT-{i:03d}" for i in range(1, 5)],
        "evaluation_plants": [f"DT-{i:03d}" for i in range(5, 13)],
        "session_slots": ["visit_1", "visit_2", "visit_3"],
        "sites_per_session": 9, "replicates_per_site": 3,
        "nrmse_limit_pct": 20.0, "noninferiority_margin_pp": 5.0,
        "minimum_session_success_fraction": 0.90,
        "bootstrap_iterations": 5000, "bootstrap_seed": 20260917,
        "confidence": 0.95, "minimum_evaluation_plants": 8,
        "min_pose_corners": 6, "max_pose_error_px": 1.0,
        "max_point_error_px": 2.0, "min_ray_angle_deg": 2.0,
        "contour_tolerance_px": 3.0,
        "sensor_radial_samples": 4, "sensor_angular_samples": 12,
        "minimum_calibration_readings": 3,
        "minimum_measurement_coverage_fraction": 0.90,
        "notes": "Review plant IDs before freezing. Physical LED/sensor/optical values are separate inputs."
    }


def validate_protocol(p: dict) -> None:
    if p.get("schema_version") != "leaf-light-study-1":
        raise StudyError("Unsupported study schema")
    if p["side_view_count"] != 36 or len(p["top_positions"]) != 5:
        raise StudyError("This study requires the approved 36 side + 5 top capture")
    if len(set(p["top_positions"])) != 5 or p["operational_top_view"] not in p["top_positions"]:
        raise StudyError("Invalid top positions")
    if p["view_counts"] != [41, 37, 19, 13, 10, 7, 5, 3, 2]:
        raise StudyError("Unexpected view-count plan; publish a new contract before changing it")
    development, evaluation = p["development_plants"], p["evaluation_plants"]
    if not development or not evaluation or len(set(development + evaluation)) != len(development + evaluation):
        raise StudyError("Plant IDs must be unique and development/evaluation disjoint")
    if not p["session_slots"] or len(set(p["session_slots"])) != len(p["session_slots"]):
        raise StudyError("Session slots must be unique")
    integer_fields = ("sites_per_session", "replicates_per_site", "bootstrap_iterations",
                      "minimum_evaluation_plants", "min_pose_corners", "sensor_radial_samples",
                      "sensor_angular_samples", "minimum_calibration_readings")
    for key in integer_fields:
        if type(p[key]) is not int or p[key] < 1:
            raise StudyError(f"{key}: positive integer required")
    if p["minimum_evaluation_plants"] < 2 or p["bootstrap_iterations"] < 100:
        raise StudyError("At least two plants and 100 bootstrap draws required")
    for key in ("minimum_session_success_fraction", "minimum_measurement_coverage_fraction", "confidence"):
        if not 0 < number(p[key], key) <= 1:
            raise StudyError(f"{key}: must be in (0,1]")
    if p["confidence"] >= 1:
        raise StudyError("Confidence must be below 1")
    for key in ("nrmse_limit_pct", "noninferiority_margin_pp", "max_pose_error_px",
                "max_point_error_px", "min_ray_angle_deg", "contour_tolerance_px"):
        if number(p[key], key) <= 0:
            raise StudyError(f"{key}: must be positive")


def freeze(path: Path, out: Path) -> None:
    p = read_json(path)
    validate_protocol(p)
    folder = output_dir(out)
    write_json(folder / "protocol.json", {
        "protocol": p, "sha256": digest(p),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "notice": "Local content freeze, not independent proof of preregistration time."
    })


def load_protocol(path: Path) -> tuple[dict, str]:
    envelope = read_json(path)
    p = envelope["protocol"]
    validate_protocol(p)
    if envelope.get("sha256") != digest(p):
        raise StudyError("Frozen protocol was changed")
    return p, envelope["sha256"]


def variants(p: dict, frames: list[dict] | None = None) -> list[dict]:
    sides = list(range(p["side_view_count"]))
    tops = p["top_positions"]
    if frames is not None:
        side_frames = [f for f in frames if f["view_type"] == "side"]
        top_frames = {f["top_position"]: f for f in frames if f["view_type"] == "top"}
        if len(frames) != 41 or len(side_frames) != 36 or set(top_frames) != set(tops):
            raise StudyError("Capture does not contain all 36 side and 5 top views")
    result = []
    for count in p["view_counts"]:
        n_side = 36 if count == 41 else count - 1
        step = 36 // n_side
        starts = sorted({0, step // 3, (2 * step) // 3})
        for start in starts:
            indices = sides[start::step]
            positions = tops if count == 41 else [p["operational_top_view"]]
            item = {"variant_id": f"v{count:02d}_s{start:02d}", "view_count": count,
                    "side_indices": indices, "top_positions": positions}
            if frames is not None:
                item["image_files"] = [side_frames[i]["image_file"] for i in indices] + [top_frames[t]["image_file"] for t in positions]
            result.append(item)
    return result
