"""Every artifact schema, in one module.

Rules enforced here:

* Tabular data is CSV, configuration / knowledge / metadata is JSON. No other formats.
* A value that could not be measured is serialised as the string ``"unavailable"``,
  never as ``0`` and never as an imputed number.
* CSV column order is part of the contract; it lives in the ``*_CSV_COLUMNS`` tuples.

Coordinate frames
-----------------
Two frames appear in this project and they are *not* the same:

``sheet frame``
    OpenCV's ChArUco object-point frame. Origin at the printed sheet's ``origin_corner``
    (top left), ``+X`` right along a board row, ``+Y`` down along a board column, ``+Z``
    into the board. Every ``*_position_mm`` and ``*_center_mm`` value in ``board_spec.json``
    is given in this frame, which the file declares with ``"position_frame": "sheet"``
    (e.g. on the 7x7 / 190 mm board, ``scale_block_position_mm = [13.57, 13.57]`` is the
    centre of corner cell A1).

``board frame``
    The pipeline frame, and the frame of every number in ``poses.csv``. Origin at the
    board's origin inner corner (ChArUco corner id 0), ``+X`` along the board rows,
    ``+Y`` along the board columns, ``+Z`` up and away from the board surface.

The conversion between them is a single rotation plus a translation and lives in
:mod:`tomato3d.calib.board`. Nothing else may reimplement it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

import numpy as np
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

# --------------------------------------------------------------------------------------
# Sentinels and shared vocabularies
# --------------------------------------------------------------------------------------

#: Written wherever a value could not be measured. Never substitute 0 or an estimate.
UNAVAILABLE: Final[str] = "unavailable"

SCHEMA_VERSION: Final[str] = "1.0"

Unavailable = Literal["unavailable"]
MaybeFloat = float | Unavailable

ViewType = Literal["side", "top"]
TopPosition = Literal["T0", "T1", "T2", "T3", "T4"]
PoseStatus = Literal["solved", "failed"]
AxisStatus = Literal["solved", "failed"]

POSE_SOLVED: Final[str] = "solved"
POSE_FAILED: Final[str] = "failed"


class SchemaError(ValueError):
    """Raised when a document does not satisfy its schema."""


# --------------------------------------------------------------------------------------
# CSV column contracts
# --------------------------------------------------------------------------------------

#: ``poses.csv`` column order. Fixed; downstream stages read by name but expect this order.
POSES_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "session_id",
    "image_file",
    "view_type",
    "top_position",
    "pose_status",
    "n_corners_detected",
    "reprojection_error_px",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_mm",
    "tvec_y_mm",
    "tvec_z_mm",
    "measured_angle_deg",
)

#: Columns that identify the row. Kept even on a failed pose.
POSES_CSV_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "session_id",
    "image_file",
    "view_type",
    "top_position",
    "pose_status",
)

#: Numeric columns. On a failed pose every one of these is written as ``unavailable``.
POSES_CSV_NUMERIC_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in POSES_CSV_COLUMNS if column not in POSES_CSV_IDENTITY_COLUMNS
)


# --------------------------------------------------------------------------------------
# JSON documents
# --------------------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base for configuration documents: immutable, no unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


#: Prefix of a free-text human note key in ``board_spec.json``. Notes are carried through
#: validation and serialisation unchanged and are never read by any code path.
NOTE_KEY_PREFIX: Final[str] = "_"


def _board_spec_json_schema(schema: dict[str, Any]) -> None:
    """Describe note keys the way the file actually carries them: flat, ``_``-prefixed."""
    schema.get("properties", {}).pop("human_notes", None)
    schema["patternProperties"] = {f"^{NOTE_KEY_PREFIX}": {"type": "string"}}


class BoardSpec(_StrictModel):
    """``config/board_spec.json`` - the physical reference board.

    Every ``*_position_mm`` and ``*_center_mm`` value is in the **sheet** frame
    (``position_frame``). Unknown keys are still an error; the only exception is a
    ``_``-prefixed human note, which is kept verbatim in :attr:`human_notes` and has no
    effect on behaviour.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", json_schema_extra=_board_spec_json_schema
    )

    schema_version: str
    variant_id: str
    pattern: Literal["charuco"]
    aruco_dict: str
    legacy_pattern: bool
    squares_x: int = Field(gt=1)
    squares_y: int = Field(gt=1)
    square_size_mm: float = Field(gt=0.0)
    marker_size_mm: float = Field(gt=0.0)
    board_width_mm: float = Field(gt=0.0)
    board_height_mm: float = Field(gt=0.0)
    marker_ids: tuple[int, ...]
    expected_marker_count: int = Field(gt=0)
    expected_inner_corner_count: int = Field(gt=0)
    #: Only ``top_left`` is implemented (first_inner_corner_sheet_mm assumes it) and it is what the
    #: printed board uses; other corners were accepted before but silently computed as top_left.
    origin_corner: Literal["top_left"]
    position_frame: Literal["sheet"]
    mounted_on_turntable: bool
    surface_finish: str
    print_page_size: str
    print_pdf_ref: str
    pattern_spec_ref: str
    board_center_mm: tuple[float, float]
    pot_center_offset_mm: tuple[float, float]
    scale_block_position_mm: tuple[float, float]
    scale_block_cell_code: str
    #: ``null`` until the physical block is measured. Null is inactive, never a placeholder.
    scale_block_nominal_height_mm: float | None = Field(gt=0.0)
    #: Side of the block's square footprint. ``null`` falls back to ``square_size_mm``.
    scale_block_footprint_mm: float | None = Field(default=None, gt=0.0)
    color_chart_on_board: bool
    physical_print_verified: bool
    human_notes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_note_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "human_notes" in data:
            raise ValueError(
                "human_notes is not a config key; write notes as "
                f"{NOTE_KEY_PREFIX}-prefixed top-level keys"
            )
        notes = {key: value for key, value in data.items() if key.startswith(NOTE_KEY_PREFIX)}
        for key, value in notes.items():
            if not isinstance(value, str):
                raise ValueError(f"note key {key} must be a string, got {type(value).__name__}")
        fields = {
            key: value for key, value in data.items() if not key.startswith(NOTE_KEY_PREFIX)
        }
        return {**fields, "human_notes": notes}

    @model_serializer(mode="wrap")
    def _flatten_note_keys(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        payload = handler(self)
        notes = payload.pop("human_notes", {})
        # An optional key the file never carried is not written back as null, so that a
        # dump reproduces the human's file rather than growing keys they did not write.
        if "scale_block_footprint_mm" not in self.model_fields_set:
            payload.pop("scale_block_footprint_mm", None)
        payload.update(notes)
        return payload

    @model_validator(mode="after")
    def _check_counts(self) -> BoardSpec:
        n_markers_expected = (self.squares_x * self.squares_y) // 2
        if self.expected_marker_count != n_markers_expected:
            raise ValueError(
                f"expected_marker_count {self.expected_marker_count} disagrees with "
                f"(squares_x * squares_y) // 2 = {n_markers_expected}"
            )
        n_inner_corners_expected = (self.squares_x - 1) * (self.squares_y - 1)
        if self.expected_inner_corner_count != n_inner_corners_expected:
            raise ValueError(
                f"expected_inner_corner_count {self.expected_inner_corner_count} disagrees "
                f"with (squares_x - 1) * (squares_y - 1) = {n_inner_corners_expected}"
            )
        if len(self.marker_ids) != self.expected_marker_count:
            raise ValueError(
                f"marker_ids lists {len(self.marker_ids)} ids but expected_marker_count "
                f"is {self.expected_marker_count}"
            )
        if len(set(self.marker_ids)) != len(self.marker_ids):
            raise ValueError("marker_ids contains duplicate ids")
        return self

    @model_validator(mode="after")
    def _check_geometry(self) -> BoardSpec:
        if self.marker_size_mm >= self.square_size_mm:
            raise ValueError(
                "marker_size_mm must be smaller than square_size_mm "
                f"(got {self.marker_size_mm} >= {self.square_size_mm})"
            )
        tolerance_mm = self.square_size_mm / 2.0
        expected_width_mm = self.squares_x * self.square_size_mm
        expected_height_mm = self.squares_y * self.square_size_mm
        if abs(self.board_width_mm - expected_width_mm) > tolerance_mm:
            raise ValueError(
                f"board_width_mm {self.board_width_mm} disagrees with "
                f"squares_x * square_size_mm = {expected_width_mm}"
            )
        if abs(self.board_height_mm - expected_height_mm) > tolerance_mm:
            raise ValueError(
                f"board_height_mm {self.board_height_mm} disagrees with "
                f"squares_y * square_size_mm = {expected_height_mm}"
            )
        return self

    @property
    def first_inner_corner_sheet_mm(self) -> tuple[float, float]:
        """Sheet-frame position of ChArUco corner id 0, i.e. the board-frame origin."""
        return (self.square_size_mm, self.square_size_mm)

    @property
    def board_center_sheet_mm(self) -> tuple[float, float]:
        """Sheet-frame centre of the printed board."""
        return (self.board_width_mm / 2.0, self.board_height_mm / 2.0)

    @property
    def scale_block_footprint_side_mm(self) -> float:
        """Side of the scale block footprint: the measured value, else one board square."""
        if self.scale_block_footprint_mm is not None:
            return self.scale_block_footprint_mm
        return self.square_size_mm


class CaptureConfig(_StrictModel):
    """``config/capture_config.json`` - the rig geometry a session is shot with."""

    schema_version: str
    side_camera_radius_mm: float = Field(gt=0.0)
    side_camera_height_mm: float
    side_camera_elevation_deg: float
    side_aim_point_mm: tuple[float, float, float]
    side_view_count_pilot: int = Field(gt=0)
    side_view_step_deg: float = Field(gt=0.0)
    top_camera_height_mm: float = Field(gt=0.0)
    top_view_positions_mm: dict[str, tuple[float, float, float]]
    operational_top_view: str
    settle_seconds_after_rotation: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _check_operational_top_view(self) -> CaptureConfig:
        if self.operational_top_view not in self.top_view_positions_mm:
            raise ValueError(
                f"operational_top_view {self.operational_top_view!r} is not a key of "
                f"top_view_positions_mm {sorted(self.top_view_positions_mm)}"
            )
        return self


class QualityThresholds(_StrictModel):
    """``config/quality_thresholds.json``.

    A ``null`` threshold means "not yet determined by the pilot". It is loaded but
    **inactive**: it never gates ``usable_for_training`` and no default is substituted.
    """

    schema_version: str
    min_pose_solved_ratio: float | None
    max_mean_reprojection_error_px: float | None
    max_silhouette_area_cv: float | None
    max_scale_block_height_error_mm: float | None
    min_charuco_corners_for_pose: int = Field(gt=0)

    @property
    def active_thresholds(self) -> dict[str, float]:
        """The gating thresholds the pilot has actually determined."""
        return {
            name: value
            for name, value in (
                ("min_pose_solved_ratio", self.min_pose_solved_ratio),
                ("max_mean_reprojection_error_px", self.max_mean_reprojection_error_px),
                ("max_silhouette_area_cv", self.max_silhouette_area_cv),
                (
                    "max_scale_block_height_error_mm",
                    self.max_scale_block_height_error_mm,
                ),
            )
            if value is not None
        }


class Intrinsics(_StrictModel):
    """``data/calib/intrinsics_<camera_id>.json``."""

    schema_version: str
    camera_id: str
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    camera_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    dist_coeffs: tuple[float, ...]
    rms_reprojection_error_px: float = Field(ge=0.0)
    n_calibration_images: int = Field(gt=0)
    calibrated_at: AwareDatetime
    distortion_model: Literal["opencv_pinhole"] = "opencv_pinhole"

    @property
    def camera_matrix_array(self) -> np.ndarray:
        """3x3 float64 camera matrix."""
        return np.asarray(self.camera_matrix, dtype=np.float64)

    @property
    def dist_coeffs_array(self) -> np.ndarray:
        """(N,) float64 distortion coefficients."""
        return np.asarray(self.dist_coeffs, dtype=np.float64)

    @property
    def image_shape(self) -> tuple[int, int]:
        """``(height_px, width_px)``, matching numpy array order."""
        return (self.image_height_px, self.image_width_px)


class ImageEntry(_StrictModel):
    """One row of the ``images[]`` list in ``session_manifest.json``."""

    file: str
    view_type: ViewType
    top_position: TopPosition | None = None
    nominal_angle_deg: float | None = None

    @model_validator(mode="after")
    def _check_view_consistency(self) -> ImageEntry:
        if self.view_type == "top":
            if self.top_position is None:
                raise ValueError(f"top_position is required for top view {self.file!r}")
            if self.nominal_angle_deg is not None:
                raise ValueError(
                    f"nominal_angle_deg must be null for top view {self.file!r}"
                )
        else:
            if self.top_position is not None:
                raise ValueError(f"top_position must be null for side view {self.file!r}")
            if self.nominal_angle_deg is None:
                raise ValueError(
                    f"nominal_angle_deg is required for side view {self.file!r}"
                )
        return self


class SessionManifest(_StrictModel):
    """``data/raw/<plant_id>/<session_id>/session_manifest.json``.

    ``nominal_angle_deg`` is the commanded turntable angle. It is bookkeeping only;
    geometry always comes from the measured pose in ``poses.csv``.
    """

    schema_version: str
    plant_id: str
    session_id: str
    captured_at: AwareDatetime
    camera_id: str
    #: The printed board this session was shot on. Marker ids overlap between variants,
    #: so this is the only way to know it; it must equal ``BoardSpec.variant_id``.
    board_variant_id: str
    board_spec_ref: str
    intrinsics_ref: str
    rotation_mode: str
    side_view_count: int = Field(ge=0)
    side_view_step_deg: float
    top_view_count: int = Field(ge=0)
    top_view_method: str
    top_view_positions_mm: dict[str, tuple[float, float, float]]
    operational_top_view: str
    side_camera_radius_mm: float
    side_camera_height_mm: float
    color_chart_in_frame: bool
    #: ``null`` while the block is unmeasured - never a placeholder number.
    scale_reference_block_mm: float | None = Field(gt=0.0)
    environment_log_ref: str | None
    images: list[ImageEntry]
    notes: str = ""

    @model_validator(mode="after")
    def _check_counts(self) -> SessionManifest:
        n_side = sum(1 for entry in self.images if entry.view_type == "side")
        n_top = sum(1 for entry in self.images if entry.view_type == "top")
        if n_side != self.side_view_count:
            raise ValueError(
                f"side_view_count {self.side_view_count} disagrees with "
                f"{n_side} side entries in images[]"
            )
        if n_top != self.top_view_count:
            raise ValueError(
                f"top_view_count {self.top_view_count} disagrees with "
                f"{n_top} top entries in images[]"
            )
        if self.operational_top_view not in self.top_view_positions_mm:
            raise ValueError(
                f"operational_top_view {self.operational_top_view!r} is not a key of "
                "top_view_positions_mm"
            )
        # masks are named by image stem (io.manifest.mask_filename): two images with one stem,
        # e.g. side/000.jpg and top/000.jpg, would silently overwrite each other's mask
        stems: dict[str, str] = {}
        for entry in self.images:
            stem = Path(entry.file).stem
            if stem in stems:
                raise ValueError(
                    f"images {stems[stem]!r} and {entry.file!r} share the stem {stem!r}; "
                    "their masks would overwrite each other"
                )
            stems[stem] = entry.file
        return self


class QualityRecord(BaseModel):
    """``data/derived/<plant_id>/<session_id>/quality.json``.

    Later stages append ``scale_block_height_error_mm``, ``hull_voxel_count`` and
    ``depth_scale_fit_rmse_mm``. At stage 1 those keys are **absent**, not null - which
    is why this model allows extra keys while the configuration models forbid them.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    session_id: str
    pose_solved_ratio: MaybeFloat
    mean_reprojection_error_px: MaybeFloat
    silhouette_area_cv: MaybeFloat
    n_images_total: int = Field(ge=0)
    n_images_solved: int = Field(ge=0)
    usable_for_training: bool
    failure_reasons: list[str]


# --------------------------------------------------------------------------------------
# In-memory carriers (numpy; never serialised directly)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardDetection:
    """ChArUco corners found in one image."""

    corners: np.ndarray  # (N, 2) float32, subpixel image coordinates
    ids: np.ndarray  # (N,) int32, ChArUco corner ids
    n_corners_detected: int

    @classmethod
    def empty(cls) -> BoardDetection:
        """A detection carrying no corners at all."""
        return cls(
            corners=np.zeros((0, 2), dtype=np.float32),
            ids=np.zeros((0,), dtype=np.int32),
            n_corners_detected=0,
        )


@dataclass(frozen=True)
class Pose:
    """One camera pose, board frame -> camera frame.

    ``rvec_rad`` is the Rodrigues vector in radians and ``tvec_mm`` the translation in
    millimetres. Both are ``None`` when ``pose_status == "failed"``; nothing downstream
    may substitute a value for them.
    """

    pose_status: PoseStatus
    n_corners_detected: int
    reprojection_error_px: float | None = None
    rvec_rad: np.ndarray | None = None
    tvec_mm: np.ndarray | None = None
    session_id: str | None = None
    image_file: str | None = None
    view_type: ViewType | None = None
    top_position: TopPosition | None = None
    measured_angle_deg: float | None = None
    failure_reason: str | None = None

    @property
    def is_solved(self) -> bool:
        return self.pose_status == POSE_SOLVED

    def with_identity(
        self,
        *,
        session_id: str,
        image_file: str,
        view_type: ViewType,
        top_position: TopPosition | None,
    ) -> Pose:
        """Attach the manifest identity that :func:`solve_pose` does not know about."""
        return replace(
            self,
            session_id=session_id,
            image_file=image_file,
            view_type=view_type,
            top_position=top_position,
        )

    def with_measured_angle_deg(self, angle_deg: float | None) -> Pose:
        return replace(self, measured_angle_deg=angle_deg)

    def rotation_matrix(self) -> np.ndarray:
        """3x3 board -> camera rotation. Raises when the pose failed."""
        if not self.is_solved or self.rvec_rad is None:
            raise SchemaError("rotation_matrix requested from a failed pose")
        import cv2  # local import keeps this module importable without a camera stack

        matrix, _ = cv2.Rodrigues(np.asarray(self.rvec_rad, dtype=np.float64))
        return matrix

    def camera_center_board_mm(self) -> np.ndarray:
        """Camera centre expressed in the board frame. Raises when the pose failed."""
        if not self.is_solved or self.tvec_mm is None:
            raise SchemaError("camera_center_board_mm requested from a failed pose")
        rotation = self.rotation_matrix()
        return -rotation.T @ np.asarray(self.tvec_mm, dtype=np.float64).reshape(3)

    def to_csv_row(self) -> dict[str, str]:
        """Serialise into ``POSES_CSV_COLUMNS``.

        A failed pose keeps its identity columns and writes ``unavailable`` for every
        numeric column - never ``0``.
        """
        row = {
            "session_id": _fmt_str(self.session_id),
            "image_file": _fmt_str(self.image_file),
            "view_type": _fmt_str(self.view_type),
            "top_position": _fmt_str(self.top_position),
            "pose_status": self.pose_status,
        }
        if not self.is_solved:
            row.update({column: UNAVAILABLE for column in POSES_CSV_NUMERIC_COLUMNS})
            return row
        rvec_rad = np.asarray(self.rvec_rad, dtype=np.float64).reshape(3)
        tvec_mm = np.asarray(self.tvec_mm, dtype=np.float64).reshape(3)
        row.update(
            {
                "n_corners_detected": _fmt_int(self.n_corners_detected),
                "reprojection_error_px": _fmt_float(self.reprojection_error_px),
                "rvec_x": _fmt_float(rvec_rad[0]),
                "rvec_y": _fmt_float(rvec_rad[1]),
                "rvec_z": _fmt_float(rvec_rad[2]),
                "tvec_x_mm": _fmt_float(tvec_mm[0]),
                "tvec_y_mm": _fmt_float(tvec_mm[1]),
                "tvec_z_mm": _fmt_float(tvec_mm[2]),
                "measured_angle_deg": _fmt_float(self.measured_angle_deg),
            }
        )
        return row


@dataclass(frozen=True)
class TurntableAxis:
    """Rotation axis fitted to the side-view pose sequence, in the board frame.

    Quality control and ``measured_angle_deg`` only. No downstream geometry may depend
    on it: reconstruction uses the per-frame pose directly.
    """

    status: AxisStatus
    n_poses_used: int
    origin_mm: np.ndarray | None = None
    direction: np.ndarray | None = None
    reference_direction: np.ndarray | None = None
    radius_mm: float | None = None
    rms_fit_error_mm: float | None = None
    failure_reason: str | None = None

    @property
    def is_solved(self) -> bool:
        return self.status == "solved"


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------


def _fmt_str(value: str | None) -> str:
    return UNAVAILABLE if value is None else str(value)


def _fmt_int(value: int | None) -> str:
    return UNAVAILABLE if value is None else str(int(value))


def _fmt_float(value: float | None) -> str:
    """Full-precision repr, so that reading the cell back yields the identical float."""
    if value is None:
        return UNAVAILABLE
    number = float(value)
    if not np.isfinite(number):
        return UNAVAILABLE
    return repr(number)


def parse_maybe_float(text: str) -> float | None:
    """Inverse of the float formatter: ``"unavailable"`` becomes ``None``."""
    return None if text == UNAVAILABLE else float(text)


def format_maybe_float(value: float | None) -> str:
    """Write a float cell: ``None``, NaN and infinities become ``"unavailable"``, never 0."""
    return _fmt_float(value)


# --------------------------------------------------------------------------------------
# JSON document IO
# --------------------------------------------------------------------------------------

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_json_model(path: Path, model_cls: type[ModelT]) -> ModelT:
    """Read and validate a JSON document.

    Raises:
        SchemaError: the file is missing, is not JSON, or fails validation. The message
            names every offending field.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaError(f"cannot read {model_cls.__name__} from {path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path} is not valid JSON: {exc}") from exc
    return validate_model(payload, model_cls, source=str(path))


def validate_model(
    payload: Any, model_cls: type[ModelT], *, source: str = "<memory>"
) -> ModelT:
    """Validate ``payload``, naming every offending field in the raised error."""
    from pydantic import ValidationError

    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise SchemaError(
            f"{source} is not a valid {model_cls.__name__} - {problems}"
        ) from exc


def dump_json_model(path: Path, model: BaseModel) -> None:
    """Write a validated model as UTF-8 JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for any document model, for external validators and documentation."""
    return model_cls.model_json_schema()


def now_isoformat() -> str:
    """Current local time as an ISO8601 string with an explicit offset."""
    return datetime.now().astimezone().isoformat()


# ======================================================================================
# Leaf light-use: the lighting environment, leaf geometry and PPFD readings
# ======================================================================================
#
# A third coordinate frame appears in this section.
#
# ``grow frame``
#     The growing space under the cultivation LEDs. Millimetres, right-handed, ``+Z`` up.
#     Where its origin sits and which way ``+X`` points is written by the human in
#     ``LightScene.grow_frame_description``. Fixture positions, plant placements and PPFD
#     reading positions are all in this frame. Plant geometry stays in the **board** frame
#     on disk and is moved into the grow frame only by
#     :func:`tomato3d.light.frames.board_to_grow_mm`.

#: Where an environment variable's value came from.
#:
#: ``measured``          observed on this rig (tape, plug meter, quantum sensor)
#: ``manufacturer_spec`` a datasheet value, not checked on this rig
#: ``reconstructed``     derived from the photographs by the pipeline
#: ``manual``            assigned by a person without an instrument (e.g. hand-drawn leaf)
#: ``assumed``           a placeholder chosen so a calculation can run
#: ``scenario``          a hypothetical arrangement being simulated, not the built one
#: ``fitted``            estimated from frozen calibration readings of this rig; a model
#:                       parameter, never a measurement (``source`` names the fit manifest)
#: ``unavailable``       not known; ``value`` is null and nothing is imputed
InputStatus = Literal[
    "measured",
    "manufacturer_spec",
    "reconstructed",
    "manual",
    "assumed",
    "scenario",
    "fitted",
    "unavailable",
]
INPUT_UNAVAILABLE: Final[str] = "unavailable"

#: Statuses whose value is not an observation of this rig. The run summary lists every
#: input carrying one, because every output that depends on it inherits the caveat.
NON_OBSERVED_INPUT_STATUSES: Final[tuple[str, ...]] = (
    "manufacturer_spec",
    "manual",
    "assumed",
    "scenario",
    "fitted",
)

#: The arrangement exactly as written in the scene, before any scenario override.
AS_BUILT_SCENARIO_ID: Final[str] = "as_built"
#: Reserved geometry variant: the grow space with no plant geometry at all. It is the
#: light-field baseline every other geometry variant is compared against.
NO_PLANTS_VARIANT_ID: Final[str] = "no_plants"

PhotonBand = Literal["PAR_400_700nm"]
ElectricalBoundary = Literal["wall_plug_including_driver", "driver_output_to_led"]
PostureSinceCapture = Literal["unchanged_checked", "changed", "not_checked"]
AmbientLight = Literal["excluded_dark_room", "present_not_modelled"]

#: Cell vocabularies of the light CSV tables.
GEOMETRY_SOURCES: Final[tuple[str, ...]] = ("reconstructed", "manual", "simplified", "assumed")
NORMAL_CONVENTIONS: Final[tuple[str, ...]] = ("adaxial_ccw", "unknown")
SPLIT_ROLES: Final[tuple[str, ...]] = ("calibration", "evaluation")

#: Classes a plant surface may carry. Names follow the organ ontology of the skeleton;
#: ``pot`` and ``support`` are non-organ occluders.
SURFACE_CLASSES: Final[tuple[str, ...]] = (
    "leaf",
    "leaflet",
    "petiole",
    "stem_segment",
    "truss",
    "flower_bud",
    "flower_open",
    "flower_spent",
    "fruit",
    "growing_tip",
    "pot",
    "support",
)
#: Surfaces that receive and absorb light in the budget. Every other class is an opaque
#: occluder only.
LIGHT_RECEIVER_CLASSES: Final[tuple[str, ...]] = ("leaf", "leaflet")

_HHMM_START_PATTERN: Final[str] = r"^([01]\d|2[0-3]):[0-5]\d$"
_HHMM_END_PATTERN: Final[str] = r"^(([01]\d|2[0-3]):[0-5]\d|24:00)$"
MINUTES_PER_DAY: Final[int] = 24 * 60
_MINUTES_PER_HOUR: Final[int] = 60

#: Floating-point slack when checking that absorptance + transmittance + reflectance <= 1.
_OPTICS_SUM_TOLERANCE: Final[float] = 1e-9


class ScalarInput(_StrictModel):
    """One environment variable: its value and where the value came from.

    ``value`` is null exactly when ``status`` is ``unavailable``. A measured zero is a
    value; an unknown quantity is never zero.
    """

    value: float | None
    status: InputStatus
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_value_matches_status(self) -> ScalarInput:
        if (self.value is None) != (self.status == INPUT_UNAVAILABLE):
            raise ValueError(
                "value must be null exactly when status is 'unavailable' "
                f"(value={self.value!r}, status={self.status!r})"
            )
        if self.value is not None and not np.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value!r}")
        return self

    @property
    def is_available(self) -> bool:
        return self.value is not None


class VectorInput(_StrictModel):
    """A three-component environment variable (a position), with its provenance."""

    value: tuple[float, float, float] | None
    status: InputStatus
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_value_matches_status(self) -> VectorInput:
        if (self.value is None) != (self.status == INPUT_UNAVAILABLE):
            raise ValueError(
                "value must be null exactly when status is 'unavailable' "
                f"(value={self.value!r}, status={self.status!r})"
            )
        if self.value is not None and not all(np.isfinite(v) for v in self.value):
            raise ValueError(f"value must be finite, got {self.value!r}")
        return self

    @property
    def is_available(self) -> bool:
        return self.value is not None


def _check_input_range(
    name: str,
    item: ScalarInput,
    *,
    low: float,
    high: float | None = None,
    low_inclusive: bool = True,
    high_inclusive: bool = True,
) -> None:
    """Range-check an available input; an unavailable one is left alone."""
    if item.value is None:
        return
    value = item.value
    below = value < low if low_inclusive else value <= low
    above = False
    if high is not None:
        above = value > high if high_inclusive else value >= high
    if below or above:
        low_bracket = "[" if low_inclusive else "("
        high_bracket = "]" if high_inclusive else ")"
        raise ValueError(
            f"{name} must lie in {low_bracket}{low}, {high}{high_bracket}, got {value}"
        )


class ScheduleSegment(_StrictModel):
    """One lit interval of a fixture's daily schedule.

    ``end_hhmm`` at or before ``start_hhmm`` crosses midnight. ``electrical_input_w`` is the
    power drawn during this segment; when it is absent the fixture's full-output power is
    used only if ``output_fraction`` is exactly 1 - a dimmed segment's power is never
    scaled linearly, because driver efficiency is not linear in output.
    """

    start_hhmm: str = Field(pattern=_HHMM_START_PATTERN)
    end_hhmm: str = Field(pattern=_HHMM_END_PATTERN)
    output_fraction: float = Field(ge=0.0, le=1.0)
    electrical_input_w: ScalarInput | None = None

    @staticmethod
    def _to_minutes(text: str) -> int:
        hours, minutes = text.split(":")
        return int(hours) * _MINUTES_PER_HOUR + int(minutes)

    @model_validator(mode="after")
    def _check_duration(self) -> ScheduleSegment:
        # Raw minutes: 24:00 (1440) differs from 00:00 (0), so 00:00-24:00 is a full day.
        if self._to_minutes(self.start_hhmm) == self._to_minutes(self.end_hhmm):
            raise ValueError(
                f"schedule segment {self.start_hhmm}-{self.end_hhmm} has no defined "
                "duration; write 00:00-24:00 for a lamp that never switches off"
            )
        if self.electrical_input_w is not None:
            _check_input_range("electrical_input_w", self.electrical_input_w, low=0.0)
        return self

    @property
    def start_minute(self) -> int:
        return self._to_minutes(self.start_hhmm)

    @property
    def duration_minutes(self) -> int:
        start = self._to_minutes(self.start_hhmm)
        end = self._to_minutes(self.end_hhmm)
        return end - start if end > start else end + MINUTES_PER_DAY - start

    @property
    def duration_hours(self) -> float:
        return self.duration_minutes / _MINUTES_PER_HOUR

    def day_intervals_minutes(self) -> list[tuple[int, int]]:
        """The segment as half-open ``[start, end)`` intervals within one day."""
        start = self.start_minute
        end = start + self.duration_minutes
        if end <= MINUTES_PER_DAY:
            return [(start, end)]
        return [(start, MINUTES_PER_DAY), (0, end - MINUTES_PER_DAY)]


class LightDistribution(_StrictModel):
    """Angular emission of one emitter point: ``I(theta) = I0 * cos(theta)**m``.

    ``theta`` is measured from the fixture's aim axis. ``m`` follows from the
    half-intensity half-angle (60 deg gives ``m = 1``, a Lambertian emitter). Beyond
    ``cutoff_half_angle_deg`` nothing is emitted. ``I0`` is set so that the emitted photon
    flux equals the fixture's ``photon_flux_umol_s`` - narrowing the beam concentrates the
    same photons, it does not remove any.
    """

    kind: Literal["cosine_power"]
    half_intensity_half_angle_deg: ScalarInput
    cutoff_half_angle_deg: float = Field(gt=0.0, le=90.0)

    @model_validator(mode="after")
    def _check_angle(self) -> LightDistribution:
        _check_input_range(
            "half_intensity_half_angle_deg",
            self.half_intensity_half_angle_deg,
            low=0.0,
            high=90.0,
            low_inclusive=False,
            high_inclusive=False,
        )
        return self


class LedFixture(_StrictModel):
    """One cultivation lamp.

    The emitting face is an ``emitting_length_mm`` x ``emitting_width_mm`` rectangle
    centred on ``emitting_center_grow_mm`` (grow frame), discretised into
    ``emitter_grid = [n_along_length, n_along_width]`` equal point emitters. A zero length
    or width is a line or a point.

    Orientation: ``aim_tilt_deg`` is the angle between the aim axis and straight down
    (0 = pointing down); ``aim_azimuth_deg`` is the direction the aim axis leans toward,
    measured in the grow XY plane from ``+X`` toward ``+Y``. ``long_axis_azimuth_deg`` is
    the grow-frame azimuth of the lamp's long side before tilting.
    """

    fixture_id: str = Field(min_length=1)
    model_name: str
    emitting_center_grow_mm: VectorInput
    aim_tilt_deg: ScalarInput
    aim_azimuth_deg: ScalarInput
    long_axis_azimuth_deg: ScalarInput
    emitting_length_mm: ScalarInput
    emitting_width_mm: ScalarInput
    emitter_grid: tuple[int, int]
    photon_flux_umol_s: ScalarInput
    photon_band: PhotonBand
    distribution: LightDistribution
    electrical_input_w: ScalarInput
    electrical_boundary: ElectricalBoundary
    schedule: list[ScheduleSegment]

    @model_validator(mode="after")
    def _check_fixture(self) -> LedFixture:
        if min(self.emitter_grid) < 1:
            raise ValueError(
                f"emitter_grid must be >= 1 on both axes, got {self.emitter_grid}"
            )
        _check_input_range("aim_tilt_deg", self.aim_tilt_deg, low=0.0, high=180.0)
        _check_input_range("emitting_length_mm", self.emitting_length_mm, low=0.0)
        _check_input_range("emitting_width_mm", self.emitting_width_mm, low=0.0)
        _check_input_range("photon_flux_umol_s", self.photon_flux_umol_s, low=0.0)
        _check_input_range(
            "electrical_input_w", self.electrical_input_w, low=0.0, low_inclusive=False
        )
        intervals = sorted(
            interval
            for segment in self.schedule
            for interval in segment.day_intervals_minutes()
        )
        for (_, previous_end), (next_start, _) in zip(intervals, intervals[1:]):
            if next_start < previous_end:
                raise ValueError(f"schedule segments of {self.fixture_id} overlap")
        return self


class PlantPlacement(_StrictModel):
    """Where a plant's board frame sits in the grow frame.

    The placement is the pose the capture board would have if it lay flat under the pot as
    the pot stands on the shelf: a yaw about ``+Z`` (``board_yaw_deg``, board ``+X``
    measured from grow ``+X`` toward grow ``+Y``) plus the grow-frame position of the board
    origin (ChArUco corner id 0), whose ``z`` is the height of the pot base.
    Mark the pot and board edge before moving the plant, so the yaw can be measured.
    A plant whose placement is unknown cannot be simulated,
    and it cannot be dropped either - it would stop shading its neighbours - so an
    unavailable placement is refused.
    """

    plant_id: str = Field(min_length=1)
    board_origin_grow_mm: VectorInput
    board_yaw_deg: ScalarInput
    posture_since_capture: PostureSinceCapture

    @model_validator(mode="after")
    def _check_available(self) -> PlantPlacement:
        if not (self.board_origin_grow_mm.is_available and self.board_yaw_deg.is_available):
            raise ValueError(
                f"placement of plant {self.plant_id!r} is unavailable; measure it or "
                "remove the plant's geometry from the run"
            )
        return self


class LeafOptics(_StrictModel):
    """Leaf optical properties in ``photon_band``, applied to every leaf surface.

    Transmittance is used for light passing through a leaf; absorptance turns incident
    photons into absorbed photons. Reflectance is recorded for the photon balance, but
    reflected light is not traced.
    """

    photon_band: PhotonBand
    absorptance: ScalarInput
    transmittance: ScalarInput
    reflectance: ScalarInput

    @model_validator(mode="after")
    def _check_optics(self) -> LeafOptics:
        parts = {
            "absorptance": self.absorptance,
            "transmittance": self.transmittance,
            "reflectance": self.reflectance,
        }
        for name, item in parts.items():
            _check_input_range(name, item, low=0.0, high=1.0)
        total = sum(item.value for item in parts.values() if item.value is not None)
        if total > 1.0 + _OPTICS_SUM_TOLERANCE:
            raise ValueError(
                f"absorptance + transmittance + reflectance = {total} exceeds 1"
            )
        return self


class LightSimulationSettings(_StrictModel):
    """Numerical settings of the light calculation. No physical meaning."""

    #: Each triangle is split into ``n**2`` equal sub-triangles, sampled at their centroids.
    triangle_subdivisions: int = Field(ge=1)
    #: Rays start this far off the receiving surface so it does not shade itself.
    ray_offset_mm: float = Field(gt=0.0)
    #: Upper bound on rays x triangles tested at once; trades memory for speed.
    max_ray_triangle_pairs_per_chunk: int = Field(ge=1)


class LightValidationSettings(_StrictModel):
    """How PPFD predictions are compared with quantum-sensor readings.

    ``ppfd_bin_edges_umol_m2_s`` splits evaluation readings by measured PPFD. ``null``
    means not yet decided: loaded but inactive, and no default bins are substituted.
    """

    ppfd_bin_edges_umol_m2_s: tuple[float, ...] | None

    @model_validator(mode="after")
    def _check_edges(self) -> LightValidationSettings:
        edges = self.ppfd_bin_edges_umol_m2_s
        if edges is not None:
            if len(edges) < 2:
                raise ValueError("ppfd_bin_edges_umol_m2_s needs at least two edges")
            if any(b <= a for a, b in zip(edges, edges[1:])):
                raise ValueError("ppfd_bin_edges_umol_m2_s must be strictly increasing")
        return self


class FixtureOverride(_StrictModel):
    """A scenario's change to one fixture. Only the fields written are changed."""

    fixture_id: str
    enabled: bool = True
    emitting_center_grow_mm: tuple[float, float, float] | None = None
    aim_tilt_deg: float | None = Field(default=None, ge=0.0, le=180.0)
    aim_azimuth_deg: float | None = None
    long_axis_azimuth_deg: float | None = None
    photon_flux_umol_s: float | None = Field(default=None, ge=0.0)
    electrical_input_w: float | None = Field(default=None, gt=0.0)


class PlacementOverride(_StrictModel):
    """A scenario's change to one plant's position or yaw."""

    plant_id: str
    board_origin_grow_mm: tuple[float, float, float] | None = None
    board_yaw_deg: float | None = None


class LightScenario(_StrictModel):
    """A hypothetical arrangement: the as-built scene with some fields replaced."""

    scenario_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    fixture_overrides: list[FixtureOverride] = Field(default_factory=list)
    placement_overrides: list[PlacementOverride] = Field(default_factory=list)


class LightScene(_StrictModel):
    """The lighting environment of one growing setup: every environment variable.

    Lamps, plant placements, leaf optics and the calculation settings. Each physical
    value carries its provenance. ``scenarios`` describe alternative arrangements to
    compare against the as-built one.
    """

    schema_version: str
    setup_id: str = Field(min_length=1)
    grow_frame_description: str = Field(min_length=1)
    ambient_light: AmbientLight
    reflections: Literal["not_modelled"]
    fixtures: list[LedFixture] = Field(min_length=1)
    plants: list[PlantPlacement]
    leaf_optics: LeafOptics
    simulation: LightSimulationSettings
    validation: LightValidationSettings
    scenarios: list[LightScenario] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check_references(self) -> LightScene:
        fixture_ids = [fixture.fixture_id for fixture in self.fixtures]
        plant_ids = [plant.plant_id for plant in self.plants]
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        for label, ids in (
            ("fixture_id", fixture_ids),
            ("plant_id", plant_ids),
            ("scenario_id", scenario_ids),
        ):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label}: {duplicates}")
        if AS_BUILT_SCENARIO_ID in scenario_ids:
            raise ValueError(
                f"scenario_id {AS_BUILT_SCENARIO_ID!r} is reserved for the unmodified scene"
            )
        for scenario in self.scenarios:
            for override in scenario.fixture_overrides:
                if override.fixture_id not in fixture_ids:
                    raise ValueError(
                        f"scenario {scenario.scenario_id!r} overrides unknown fixture "
                        f"{override.fixture_id!r}"
                    )
            for placement in scenario.placement_overrides:
                if placement.plant_id not in plant_ids:
                    raise ValueError(
                        f"scenario {scenario.scenario_id!r} moves unknown plant "
                        f"{placement.plant_id!r}"
                    )
        return self

    @property
    def scenario_ids(self) -> list[str]:
        """Every arrangement a run evaluates, the as-built one first."""
        return [AS_BUILT_SCENARIO_ID, *(s.scenario_id for s in self.scenarios)]


class LightInputRecord(_StrictModel):
    """One environment variable as the run summary reports it."""

    name: str
    value: str
    status: InputStatus
    source: str


class GeometryVariantRecord(_StrictModel):
    """One geometry variant a run used (e.g. every view, a reduced view subset)."""

    geometry_variant_id: str
    geometry_sources: list[str]
    plant_ids: list[str]
    n_triangles: int = Field(ge=0)


class LightRunSummary(_StrictModel):
    """``run_summary.json`` of one light run: what went in, and what the numbers rest on."""

    schema_version: str
    run_id: str
    created_at: AwareDatetime
    setup_id: str
    scene_path: str
    geometry_paths: list[str]
    readings_path: str | None
    scenario_ids: list[str]
    geometry_variants: list[GeometryVariantRecord]
    reference_variant_id: str | None
    calibration_variant_id: str
    calibration_factor_by_scenario: dict[str, MaybeFloat]
    inputs: list[LightInputRecord]
    non_observed_inputs: list[str]
    model_limits: list[str]
    warnings: list[str]


# ----- light CSV contracts ----------------------------------------------------------

#: ``leaf_surfaces.csv`` - plant surfaces as triangles, **board frame**, mm.
#: With ``normal_convention = adaxial_ccw`` the vertices run counter-clockwise seen from
#: the upper (adaxial) side, so ``(v1 - v0) x (v2 - v0)`` points out of the upper surface.
#: ``leaflet_id`` is ``unavailable`` when leaflets are not separated.
LEAF_SURFACES_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "plant_id",
    "session_id",
    "geometry_variant_id",
    "geometry_source",
    "organ_class",
    "organ_id",
    "leaflet_id",
    "triangle_id",
    "v0_x_mm",
    "v0_y_mm",
    "v0_z_mm",
    "v1_x_mm",
    "v1_y_mm",
    "v1_z_mm",
    "v2_x_mm",
    "v2_y_mm",
    "v2_z_mm",
    "normal_convention",
)

#: ``leaf_patches.csv`` - simplified leaves, one flat rectangle each, **board frame**.
#: ``normal_tilt_deg`` is the angle of the upper-surface normal from ``+Z`` (0 = flat,
#: facing up); ``normal_azimuth_deg`` is the direction that normal leans toward, which is
#: also the direction the leaf slopes down. The long side runs down that slope.
LEAF_PATCHES_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "plant_id",
    "session_id",
    "geometry_variant_id",
    "geometry_source",
    "organ_id",
    "centroid_x_mm",
    "centroid_y_mm",
    "centroid_z_mm",
    "area_mm2",
    "normal_tilt_deg",
    "normal_azimuth_deg",
    "length_to_width_ratio",
)

#: ``ppfd_readings.csv`` - human-written quantum-sensor readings, **grow frame**.
#: Only readings taken with every enabled fixture at full output are supported.
PPFD_READINGS_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "reading_id",
    "setup_id",
    "scenario_id",
    "measured_at",
    "sensor_model",
    "x_mm",
    "y_mm",
    "z_mm",
    "normal_x",
    "normal_y",
    "normal_z",
    "all_fixtures_full_output",
    "ppfd_umol_m2_s",
    "split_role",
    "region_label",
    "observer",
    "note",
)

#: ``leaf_triangle_light.csv`` - the leaf light map, one row per receiving triangle.
LEAF_TRIANGLE_LIGHT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "geometry_variant_id",
    "plant_id",
    "organ_id",
    "leaflet_id",
    "triangle_id",
    "area_mm2",
    "centroid_x_grow_mm",
    "centroid_y_grow_mm",
    "centroid_z_grow_mm",
    "ppfd_full_output_umol_m2_s",
    "ppfd_unoccluded_full_output_umol_m2_s",
    "occlusion_loss_fraction",
    "adaxial_incident_fraction",
    "dli_mol_m2_day",
)

#: ``leaf_light.csv`` - one row per leaf (``organ_id``).
LEAF_LIGHT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "geometry_variant_id",
    "plant_id",
    "organ_id",
    "n_triangles",
    "area_mm2",
    "ppfd_mean_full_output_umol_m2_s",
    "ppfd_min_full_output_umol_m2_s",
    "ppfd_max_full_output_umol_m2_s",
    "occlusion_loss_fraction",
    "adaxial_incident_fraction",
    "dli_mol_m2_day",
    "intercepted_full_output_umol_s",
    "absorbed_full_output_umol_s",
    "absorbed_mol_day",
)

#: ``plant_light.csv`` - one row per plant.
PLANT_LIGHT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "geometry_variant_id",
    "plant_id",
    "n_leaves",
    "leaf_area_mm2",
    "intercepted_full_output_umol_s",
    "absorbed_full_output_umol_s",
    "intercepted_mol_day",
    "absorbed_mol_day",
    "leaf_ppfd_cv",
)

#: ``scenario_summary.csv`` - one row per scenario x geometry variant: the calculator's
#: headline numbers.
SCENARIO_SUMMARY_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "geometry_variant_id",
    "n_plants",
    "n_leaves",
    "leaf_area_mm2",
    "emitted_mol_day",
    "intercepted_mol_day",
    "absorbed_mol_day",
    "intercepted_fraction_of_emitted",
    "absorbed_fraction_of_emitted",
    "electricity_kwh_day",
    "absorbed_umol_per_j",
    "leaf_ppfd_mean_umol_m2_s",
    "leaf_ppfd_min_umol_m2_s",
    "leaf_ppfd_cv",
    "delta_absorbed_mol_day_vs_as_built",
    "delta_electricity_kwh_day_vs_as_built",
    "delta_absorbed_umol_per_j_vs_as_built",
)

#: ``sensor_predictions.csv`` - one row per reading x geometry variant.
SENSOR_PREDICTIONS_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "reading_id",
    "scenario_id",
    "geometry_variant_id",
    "split_role",
    "region_label",
    "measured_ppfd_umol_m2_s",
    "predicted_ppfd_umol_m2_s",
    "calibrated_ppfd_umol_m2_s",
    "error_umol_m2_s",
    "calibrated_error_umol_m2_s",
)

#: ``validation.csv`` - prediction error per scenario x geometry variant x subset.
#: ``subset`` is ``all`` or a measured-PPFD bin ``[low,high)``.
VALIDATION_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "geometry_variant_id",
    "split_role",
    "subset",
    "n_used",
    "n_measured_unavailable",
    "n_prediction_unavailable",
    "mae_umol_m2_s",
    "rmse_umol_m2_s",
    "bias_umol_m2_s",
    "calibration_factor",
    "calibrated_mae_umol_m2_s",
    "calibrated_rmse_umol_m2_s",
    "calibrated_bias_umol_m2_s",
)

#: ``geometry_variant_comparison.csv`` - each geometry variant against the reference one,
#: per plant. This is where "how few photographs still predict the same light" is read.
GEOMETRY_VARIANT_COMPARISON_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "plant_id",
    "reference_variant_id",
    "candidate_variant_id",
    "reference_leaf_area_mm2",
    "candidate_leaf_area_mm2",
    "reference_absorbed_mol_day",
    "candidate_absorbed_mol_day",
    "absorbed_relative_difference",
    "n_leaves_matched",
    "n_leaves_reference_only",
    "n_leaves_candidate_only",
    "leaf_ppfd_mae_matched_umol_m2_s",
    "leaf_ppfd_bias_matched_umol_m2_s",
)
