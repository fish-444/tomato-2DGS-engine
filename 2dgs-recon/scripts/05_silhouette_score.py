#!/usr/bin/env python3
"""Step 5 - silhouette scoring of held-out views (band-excluded IoU, edge F, stem coverage). Code: src/tomato3d/recon/r31_score.py."""
import runpy, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    runpy.run_module("tomato3d.recon.r31_score", run_name="__main__")
