#!/usr/bin/env python3
"""Step 3 - initial points: pycolmap SIFT with the board poses fixed (no pose estimation by SfM). Code: src/tomato3d/recon/wholeplant_sparse.py.
Needs pycolmap (pip install -r requirements-recon.txt); the options are the module's own: --poses --out --work."""
import runpy, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    try:
        runpy.run_module("tomato3d.recon.wholeplant_sparse", run_name="__main__")
    except ModuleNotFoundError as ex:
        if "pycolmap" in str(ex):
            sys.exit("Step 3 needs pycolmap, which is not in requirements.txt: pip install -r requirements-recon.txt (pycolmap==4.2.1).")
        raise
