#!/usr/bin/env python3
"""Step 1 - photos -> JPG, ChArUco detection, intrinsics, camera poses (poses.json, qc.csv). All options are the module's own: --help.
Code: src/tomato3d/recon/charuco_a3.py (and charuco_photos.py for the HEIC -> JPG conversion: run it first, see its --help)."""
import runpy, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    mod = "tomato3d.recon.charuco_photos" if "--photos-only" in sys.argv else "tomato3d.recon.charuco_a3"
    sys.argv = [a for a in sys.argv if a != "--photos-only"]
    runpy.run_module(mod, run_name="__main__")
