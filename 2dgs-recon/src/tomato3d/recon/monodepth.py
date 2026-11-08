"""Relative depth inference.

Stage 1 fixes this import graph. Nothing here is implemented yet; every entry point
raises NotImplementedError.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401  (part of the future signatures)


def infer_relative_depth(session_dir: Path) -> None:
    """Run monocular depth inference over the side views.

    Raises:
        NotImplementedError: always. This stage is not implemented yet.
    """
    raise NotImplementedError(
        "tomato3d.recon.monodepth.infer_relative_depth is a later stage and is not implemented"
    )
