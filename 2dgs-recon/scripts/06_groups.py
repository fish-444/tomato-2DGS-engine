#!/usr/bin/env python3
"""Step 6 - groups (blade / axis split, sheets, 4a patches, photo votes, merge). Runs on Kaggle in practice (see docs/GROUPS_AND_TPROX_KO.md and onezip/pc_side/).
--stage first : tomato3d.organs.all3d_first (--step inputs|prep|geometry)   --stage merge : tomato3d.organs.all3d_merge (needs candidate masks)
A plant must be registered first (see docs); the shipped example (R3-1b) used the older patches_filled groups instead."""
import runpy, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    m = {"first": "all3d_first", "ext": "all3d_ext", "cands": "all3d_cands_job", "page": "all3d_page"}
    k = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "first"
    if "--stage" in sys.argv:
        i = sys.argv.index("--stage"); del sys.argv[i:i + 2]
    runpy.run_module("tomato3d.organs." + m[k], run_name="__main__")
