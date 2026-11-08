"""Session manifest IO and session path resolution.

A session lives in two places: its capture originals under ``<data_root>/raw`` and its
pipeline products under ``<data_root>/derived``. :class:`SessionPaths` is the only place
that knows how those two map onto each other.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from tomato3d.config import Settings
from tomato3d.io.schemas import (
    SchemaError,
    SessionManifest,
    dump_json_model,
    load_json_model,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "session_manifest.json"
IMAGES_DIRNAME = "images"
MASKS_DIRNAME = "masks"
POSES_FILENAME = "poses.csv"
QUALITY_FILENAME = "quality.json"

#: ``<plant_id>_<YYYYMMDD>_<HHMM>``
SESSION_ID_PATTERN = re.compile(r"^(?P<plant_id>.+)_(?P<date>\d{8})_(?P<time>\d{4})$")


def mask_filename(image_file: str) -> str:
    """File name of the mask for one ``images[].file`` entry: ``<image stem>.png``."""
    return f"{Path(image_file).stem}.png"


class ManifestError(SchemaError):
    """Raised when a manifest is missing, malformed, or inconsistent with its folder."""


@dataclass(frozen=True)
class SessionPaths:
    """Every path belonging to one capture session."""

    plant_id: str
    session_id: str
    raw_dir: Path
    derived_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / MANIFEST_FILENAME

    @property
    def images_dir(self) -> Path:
        return self.raw_dir / IMAGES_DIRNAME

    @property
    def masks_dir(self) -> Path:
        return self.derived_dir / MASKS_DIRNAME

    @property
    def poses_csv_path(self) -> Path:
        return self.derived_dir / POSES_FILENAME

    @property
    def quality_json_path(self) -> Path:
        return self.derived_dir / QUALITY_FILENAME

    def image_path(self, image_file: str) -> Path:
        """Resolve one ``images[].file`` entry, which is relative to the session dir."""
        return self.raw_dir / image_file

    def mask_path(self, image_file: str) -> Path:
        """Mask for one image: ``masks/<image stem>.png``."""
        return self.masks_dir / mask_filename(image_file)


def parse_session_id(session_id: str) -> tuple[str, str, str]:
    """Split ``<plant_id>_<YYYYMMDD>_<HHMM>``.

    Raises:
        ManifestError: the id does not have the required shape.
    """
    match = SESSION_ID_PATTERN.match(session_id)
    if match is None:
        raise ManifestError(
            f"session_id {session_id!r} does not match <plant_id>_<YYYYMMDD>_<HHMM>"
        )
    return match["plant_id"], match["date"], match["time"]


def read_manifest(path: Path) -> SessionManifest:
    """Read and validate a session manifest.

    Raises:
        ManifestError: the file is missing, is not JSON, or fails validation. The message
            names every offending field.
    """
    try:
        manifest = load_json_model(path, SessionManifest)
    except SchemaError as exc:
        raise ManifestError(str(exc)) from exc
    plant_id, _, _ = parse_session_id(manifest.session_id)
    if plant_id != manifest.plant_id:
        raise ManifestError(
            f"plant_id {manifest.plant_id!r} is not the prefix of session_id "
            f"{manifest.session_id!r}"
        )
    return manifest


def write_manifest(path: Path, manifest: SessionManifest) -> None:
    """Write a validated session manifest."""
    dump_json_model(path, manifest)


def resolve_session_paths(session_dir: Path, settings: Settings) -> SessionPaths:
    """Resolve every path of the session whose raw folder is ``session_dir``.

    ``session_dir`` is ``<data_root>/raw/<plant_id>/<session_id>``. The derived folder is
    always ``<data_root>/derived/<plant_id>/<session_id>``, taken from the manifest rather
    than from the folder names, so a mislabelled folder is caught instead of propagated.

    Raises:
        ManifestError: the manifest is missing or disagrees with the folder names.
    """
    session_dir = session_dir.resolve()
    manifest_path = session_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ManifestError(f"no {MANIFEST_FILENAME} in {session_dir}")
    manifest = read_manifest(manifest_path)
    if session_dir.name != manifest.session_id:
        logger.warning(
            "folder name %r differs from session_id %r; the manifest wins",
            session_dir.name,
            manifest.session_id,
        )
    return SessionPaths(
        plant_id=manifest.plant_id,
        session_id=manifest.session_id,
        raw_dir=session_dir,
        derived_dir=settings.derived_dir / manifest.plant_id / manifest.session_id,
    )
