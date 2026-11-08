"""Voxel carving from silhouettes.

Stage 1 fixes this import graph. Nothing here is implemented yet; every entry point
raises NotImplementedError.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401  (part of the future signatures)


def carve_visual_hull(session_dir: Path) -> None:
    """Carve a voxel hull from the session silhouettes.

    Raises:
        NotImplementedError: always. This stage is not implemented yet.
    """
    raise NotImplementedError(
        "tomato3d.recon.visual_hull.carve_visual_hull is a later stage and is not implemented"
    )
