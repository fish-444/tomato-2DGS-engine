"""Dwarf cherry tomato 3D phenotyping pipeline.

Stage 1 implements capture bookkeeping, camera intrinsic calibration, per-frame camera
pose from a rotating ChArUco board, and plant silhouette masking.

All lengths are millimetres. Angles are degrees in files and radians in code; radian
variables carry a ``_rad`` suffix.
"""

__version__ = "0.1.0"
