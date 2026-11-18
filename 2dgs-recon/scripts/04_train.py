#!/usr/bin/env python3
"""Step 4 - 2DGS training (gsplat 1.5.3), R3-1b configuration. Needs a CUDA GPU (Kaggle T4 or similar).
Runs src/tomato3d/recon/kernel_2dgs_r31.py as a subprocess with the input staged in --stage (index.json, images_1512/, masks_*/, init/) and the output in --work.
The staging folder is what onezip/run_recon.py builds in its stage 5; use that script for a whole run."""
import argparse, os, subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, help="staged input folder (index.json, images_1512/, masks_<tag>/, bands_<tag>/, soft_<tag>/, init/)")
    ap.add_argument("--work", required=True, help="output folder (out/<run>_surfels_full.ply, results.json, run_log.txt)")
    a = ap.parse_args(argv)
    env = dict(os.environ, ONEZIP_WORK=str(Path(a.work).resolve()), ONEZIP_STAGE=str(Path(a.stage).resolve()), PYTHONUNBUFFERED="1")
    return subprocess.run([sys.executable, str(SRC / "tomato3d/recon/kernel_2dgs_r31.py")], env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
