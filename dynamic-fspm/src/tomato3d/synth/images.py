"""Image read/write that works on non-ASCII Windows paths (cv2.imread/imwrite do not)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imwrite(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"cannot encode {path}")
    buf.tofile(str(path))


def imread(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)
