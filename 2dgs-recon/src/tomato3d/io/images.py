"""Unicode-safe image IO.

On Windows, ``cv2.imread`` and ``cv2.imwrite`` cannot open a path containing characters
outside the ANSI code page - which includes this project's own folder name. ``imread``
returns ``None`` exactly as it does for a corrupt file, and ``imwrite`` returns ``False``
without raising, so a run silently reads nothing or writes nothing. Every image in the
pipeline therefore goes through this module, which moves bytes with numpy and leaves only
encoding and decoding to OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageIOError(OSError):
    """Raised when an image cannot be encoded or written."""


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode an image from ``path``.

    Returns:
        The decoded array, or ``None`` when the file is missing, empty, unreadable or not
        a decodable image. Callers record that; they never substitute an image.
    """
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def write_image(path: Path, image: np.ndarray) -> None:
    """Encode ``image`` in the format named by ``path``'s suffix and write it.

    Raises:
        ImageIOError: the image cannot be encoded in that format, or the bytes cannot be
            written. Never fails silently.
    """
    try:
        encoded_ok, encoded = cv2.imencode(path.suffix.lower(), image)
    except cv2.error as exc:
        raise ImageIOError(f"cannot encode {path.name}: {exc}") from exc
    if not encoded_ok:
        raise ImageIOError(f"cannot encode {path.name} as {path.suffix!r}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise ImageIOError(f"cannot write {path}: {exc}") from exc
