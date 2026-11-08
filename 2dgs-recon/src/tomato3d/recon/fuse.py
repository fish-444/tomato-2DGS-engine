"""Hull plus depth to an oriented point cloud.

Stage 1 fixes this import graph. Nothing here is implemented yet; every entry point
raises NotImplementedError.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401  (part of the future signatures)


def fuse_hull_and_depth(session_dir: Path) -> None:
    """Fuse the carved hull with scale-aligned depth.

    Raises:
        NotImplementedError: always. This stage is not implemented yet.
    """
    raise NotImplementedError(
        "tomato3d.recon.fuse.fuse_hull_and_depth is a later stage and is not implemented"
    )
