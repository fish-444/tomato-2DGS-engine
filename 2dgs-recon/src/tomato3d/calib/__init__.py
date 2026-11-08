"""Board construction, camera intrinsics, and per-frame pose.

Pose comes from ChArUco PnP solved independently per frame. There is deliberately no
structure-from-motion anywhere in this package: on this rig the camera and the background
are both static while the board rotates, so feature-matching pose estimation concludes
the camera never moved.
"""

from tomato3d.calib.intrinsics import CalibrationError

__all__ = ["CalibrationError"]
