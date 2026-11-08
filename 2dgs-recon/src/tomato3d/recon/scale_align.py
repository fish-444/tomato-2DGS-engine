"""Affine fit of relative depth to metric scale.

Stage 1 fixes this import graph. Nothing here is implemented yet; every entry point
raises NotImplementedError.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401  (part of the future signatures)


def fit_depth_scale(session_dir: Path) -> None:
    """Fit the affine map from relative depth to millimetres.

    Raises:
        NotImplementedError: always. This stage is not implemented yet.
    """
    raise NotImplementedError(
        "tomato3d.recon.scale_align.fit_depth_scale is a later stage and is not implemented"
    )
