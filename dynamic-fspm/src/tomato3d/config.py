"""Process configuration.

Environment variables are read exactly once, here. No other module calls ``os.environ``.
Every derived path is a property on :class:`Settings` so that call sites never
string-concatenate paths.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

#: Environment variable names, with their fallbacks when the variable is unset.
ENV_DATA_ROOT = "TOMATO3D_DATA_ROOT"
ENV_CONFIG_DIR = "TOMATO3D_CONFIG_DIR"
ENV_CAMERA_ID = "TOMATO3D_CAMERA_ID"
ENV_LOG_LEVEL = "TOMATO3D_LOG_LEVEL"
ENV_MAX_WORKERS = "TOMATO3D_MAX_WORKERS"
#: Synthetic generator only: JSON object ``{"<prefix recorded on another machine>": "<prefix here>"}`` used by
#: :func:`tomato3d.synth.config.input_path` so a generator config written on Windows can run on a Linux CPU box
#: without editing the config (its sha256 is pinned by run contracts). Unset = no remapping.
ENV_SYNTH_PATH_MAP = "TOMATO3D_SYNTH_PATH_MAP"

_FALLBACK_DATA_ROOT = "./data"
_FALLBACK_CONFIG_DIR = "./config"
_FALLBACK_CAMERA_ID = "cam_a"
_FALLBACK_LOG_LEVEL = "INFO"
_FALLBACK_MAX_WORKERS = "4"

#: Properties of :class:`Settings` that resolve underneath ``TOMATO3D_DATA_ROOT``.
DATA_DIR_PROPERTIES: tuple[str, ...] = (
    "raw_dir",
    "calib_dir",
    "derived_dir",
    "outputs_dir",
    "knowledge_dir",
    "tracks_dir",
    "light_runs_dir",
)

#: Properties of :class:`Settings` that resolve underneath ``TOMATO3D_CONFIG_DIR``.
CONFIG_FILE_PROPERTIES: tuple[str, ...] = (
    "board_spec_path",
    "capture_config_path",
    "quality_thresholds_path",
)


class ConfigError(ValueError):
    """Raised when the environment cannot be turned into a valid :class:`Settings`."""


@dataclass(frozen=True)
class Settings:
    """Immutable run configuration resolved from the environment."""

    data_root: Path
    config_dir: Path
    camera_id: str
    log_level: str
    max_workers: int
    #: ``(recorded prefix, local prefix)`` pairs, recorded side normalised to ``/`` without a trailing slash.
    synth_path_map: tuple[tuple[str, str], ...] = ()

    # ----- data tree -------------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Capture originals: ``raw/<plant_id>/<session_id>/``."""
        return self.data_root / "raw"

    @property
    def calib_dir(self) -> Path:
        """Camera intrinsics and calibration shots."""
        return self.data_root / "calib"

    @property
    def derived_dir(self) -> Path:
        """Pipeline outputs: ``derived/<plant_id>/<session_id>/``."""
        return self.data_root / "derived"

    @property
    def outputs_dir(self) -> Path:
        """Downstream products (diagnoses, actions)."""
        return self.data_root / "outputs"

    @property
    def knowledge_dir(self) -> Path:
        """Crop packs and other curated knowledge JSON."""
        return self.data_root / "knowledge"

    @property
    def tracks_dir(self) -> Path:
        """Cross-date organ tracks."""
        return self.data_root / "tracks"

    @property
    def light_runs_dir(self) -> Path:
        """Leaf light-use runs: ``outputs/light/<run_id>/``."""
        return self.outputs_dir / "light"

    # ----- configuration tree ----------------------------------------------------
    @property
    def board_spec_path(self) -> Path:
        return self.config_dir / "board_spec.json"

    @property
    def capture_config_path(self) -> Path:
        return self.config_dir / "capture_config.json"

    @property
    def quality_thresholds_path(self) -> Path:
        return self.config_dir / "quality_thresholds.json"

    @property
    def light_scene_path(self) -> Path:
        """The growing setup's lighting environment.

        Not in :data:`CONFIG_FILE_PROPERTIES`: the human creates this file when the lamps
        are characterised, so it may not exist yet.
        """
        return self.config_dir / "light_scene.json"

    # ----- helpers ---------------------------------------------------------------
    def intrinsics_path(self, camera_id: str | None = None) -> Path:
        """Path of the intrinsics JSON for ``camera_id`` (default: the configured camera)."""
        return self.calib_dir / f"intrinsics_{camera_id or self.camera_id}.json"


def _read_int(name: str, raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:  # pragma: no cover - exercised through ConfigError
        raise ConfigError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 1:
        raise ConfigError(f"{name} must be >= 1, got {value}")
    return value


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` for this process.

    Call ``get_settings.cache_clear()`` after mutating the environment (tests only).
    """
    load_dotenv()
    return Settings(
        data_root=Path(os.environ.get(ENV_DATA_ROOT, _FALLBACK_DATA_ROOT)).expanduser(),
        config_dir=Path(os.environ.get(ENV_CONFIG_DIR, _FALLBACK_CONFIG_DIR)).expanduser(),
        camera_id=os.environ.get(ENV_CAMERA_ID, _FALLBACK_CAMERA_ID),
        log_level=os.environ.get(ENV_LOG_LEVEL, _FALLBACK_LOG_LEVEL),
        max_workers=_read_int(
            ENV_MAX_WORKERS, os.environ.get(ENV_MAX_WORKERS, _FALLBACK_MAX_WORKERS)
        ),
        synth_path_map=_read_path_map(os.environ.get(ENV_SYNTH_PATH_MAP, "")),
    )


def _read_path_map(raw_value: str) -> tuple[tuple[str, str], ...]:
    """Parse ``TOMATO3D_SYNTH_PATH_MAP``; the recorded prefixes are normalised to ``/`` separators."""
    if not raw_value.strip():
        return ()
    try:
        data = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{ENV_SYNTH_PATH_MAP} must be a JSON object, got {raw_value!r}") from exc
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ConfigError(f"{ENV_SYNTH_PATH_MAP} must map prefix strings to prefix strings")
    return tuple((k.replace("\\", "/").rstrip("/"), v) for k, v in data.items())
