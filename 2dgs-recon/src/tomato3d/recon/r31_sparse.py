"""R3-1 Step 3, I1: the A3 fallback (pycolmap SIFT 3024 px, 32,768 features, fixed board poses) re-run with the Step-2 masks.
Thin wrapper around tomato3d.recon.wholeplant_sparse: a scratch run folder with the chosen masks and the R3-1 split.

usage: python -m tomato3d.recon.r31_sparse --poses <charuco dir> --out <r3-1 run dir> --work <ascii dir> [--masks masks_a] [--tag a]
Writes <out>/plant_points_<tag>.ply and sparse_<tag>_checks.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tomato3d.recon import wholeplant_sparse as WS


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--masks", default="masks_a"); ap.add_argument("--tag", default="a")
    a = ap.parse_args(argv); out = Path(a.out); con = json.load(open(out / "contract.json"))
    tmp = out / f"_sparse_{a.tag}"; tmp.mkdir(exist_ok=True)
    json.dump(dict(heldout_new=con["heldout_fresh"]), open(tmp / "contract.json", "w"))
    if (tmp / "masks").exists():
        shutil.rmtree(tmp / "masks") if False else None
    (tmp / "masks").mkdir(exist_ok=True)
    for p in (out / a.masks).glob("*.png"):
        shutil.copy2(p, tmp / "masks" / p.name)
    v2 = Path("build/p1_recon_wholeplant_20261022_01")
    shutil.copy2(v2 / "masks_summary.json", tmp / "masks_summary.json")          # hull box for the canopy check (same capture)
    WS.main(["--poses", a.poses, "--out", str(tmp), "--work", a.work])
    shutil.copy2(tmp / "plant_points_dense.ply", out / f"plant_points_{a.tag}.ply")
    shutil.copy2(tmp / "sparse_dense_checks.json", out / f"sparse_{a.tag}_checks.json")
    print(json.dumps({k: v for k, v in json.load(open(out / f"sparse_{a.tag}_checks.json")).items() if k != "points_per_view"}, indent=1))


if __name__ == "__main__":
    main()
