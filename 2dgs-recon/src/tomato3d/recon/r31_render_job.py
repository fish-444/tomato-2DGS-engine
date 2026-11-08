"""Stage the render-only job: put a run's surfels_full.ply into the dataset and set index.json to one render_only run.

usage: python -m tomato3d.recon.r31_render_job --stage <stage dir> --ply <surfels_full.ply> --run R3-base --masks masks_R3_base
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tomato3d.recon.r31_prep import KERNEL_SRC


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--ply", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--masks", required=True); a = ap.parse_args(argv); st = Path(a.stage); d = st / "data"
    (d / "init").mkdir(exist_ok=True); dst = d / "init" / f"{a.run}_surfels_full.ply"; shutil.copy2(a.ply, dst)
    idx = json.load(open(d / "index.json")); tag = a.masks.replace("masks_", "")
    idx["runs"] = {f"{a.run}-render": dict(note=f"render-only: {a.run} surfels -> alpha/depth of every view", res=1512, masks=a.masks, fg_labels=[1, 2],
                                            band=f"bands_{tag}", init=f"init/points_{tag}.ply", init_hull=f"init/hull_{tag}.ply", render_only=f"init/{a.run}_surfels_full.ply")}
    json.dump(idx, open(d / "index.json", "w"), indent=1)
    shutil.copy2(Path(__file__).with_name("kernel_2dgs_r31.py"), st / "kernel" / "run.py")
    print("staged", list(idx["runs"]))


if __name__ == "__main__":
    main()
