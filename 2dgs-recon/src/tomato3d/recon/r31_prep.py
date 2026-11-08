"""R3-1 packaging: one Kaggle dataset version + one kernel push per run (the kernel reads index.json runs; results are merged locally).

usage: python -m tomato3d.recon.r31_prep --poses <charuco dir> --v2 <wholeplant run dir> --out <r3-1 run dir> --stage <dir>
           --run R3-base --masks <dir> --bands <dir> --init <ply> --hull <ply> [--note ...] [--soft <dir>] [--res 1512]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import MASK_RES
from tomato3d.recon.charuco_prep import HULL_COLOUR, HULL_INIT_MAX, read_xyz, write_xyzrgb

DATASET, KERNEL = "dolgfish/tomato-recon-r31-input", "dolgfish/tomato-recon-r31-2dgs"
KERNEL_SRC = Path(__file__).with_name("kernel_2dgs_wholeplant.py")


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("poses", "v2", "out", "stage", "run", "masks", "bands", "init", "hull"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--note", default=""); ap.add_argument("--soft", default=None); ap.add_argument("--res", type=int, default=1512)
    ap.add_argument("--kernel", default=str(KERNEL_SRC))
    ap.add_argument("--cfg", default="{}", help="JSON merged into the run cfg (training settings, Step 4)")
    ap.add_argument("--init-mono", default=None, help="mono_prior-only points (I2) staged as init/mono_<tag>.ply for the depth loss")
    ap.add_argument("--variants", default="{}", help='JSON {run_name: extra cfg}: declared variants sharing the same staged files')
    a = ap.parse_args(argv); src, v2, rw, st = Path(a.poses), Path(a.v2), Path(a.out), Path(a.stage)
    con = json.load(open(rw / "contract.json")); po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    held, train = con["heldout_fresh"], con["train"]; views = train + held
    d = st / "data"; d.mkdir(parents=True, exist_ok=True)
    (d / f"images_{a.res}").mkdir(exist_ok=True)
    for v in views:
        p = d / f"images_{a.res}" / v
        if not p.exists():
            Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((a.res, a.res), Image.LANCZOS).save(p, quality=95)
    tag = a.run.replace("-", "_")
    for name, srcdir in ((f"masks_{tag}", Path(a.masks)), (f"bands_{tag}", Path(a.bands))) + (((f"soft_{tag}", Path(a.soft)),) if a.soft else ()):
        (d / name).mkdir(exist_ok=True)
        for v in views:
            shutil.copy2(srcdir / (Path(v).stem + ".png"), d / name / (Path(v).stem + ".png"))
    (d / "init").mkdir(exist_ok=True)
    P1, C1 = read_xyz(Path(a.init)); write_xyzrgb(d / "init" / f"points_{tag}.ply", P1, C1)
    Ph, _ = read_xyz(Path(a.hull)); rng = np.random.default_rng(con["seed"]); Ph = Ph[rng.choice(len(Ph), min(HULL_INIT_MAX, len(Ph)), replace=False)]
    write_xyzrgb(d / "init" / f"hull_{tag}.ply", Ph, np.tile(np.array(HULL_COLOUR, np.float32), (len(Ph), 1)))
    H = float(Ph[:, 2].max() - max(Ph[:, 2].min(), 0.0))
    K = ca.scaled_K(np.array(po["K_undist"]), R0, a.res)
    cams = {str(a.res): {v: dict(K=K.tolist(), R_cw=po["views"][v]["R_cw"], t_cw=po["views"][v]["t_cw"]) for v in views}}
    cfg = dict(note=a.note or a.run, res=a.res, masks=f"masks_{tag}", fg_labels=[1, 2], band=f"bands_{tag}", init=f"init/points_{tag}.ply", init_hull=f"init/hull_{tag}.ply")
    if a.soft:
        cfg["soft"] = f"soft_{tag}"
    if a.init_mono:
        Pm, Cm = read_xyz(Path(a.init_mono)); write_xyzrgb(d / "init" / f"mono_{tag}.ply", Pm, Cm); cfg["init_mono"] = f"init/mono_{tag}.ply"
    cfg.update(json.loads(a.cfg)); runs = {a.run: cfg}
    for name, extra in json.loads(a.variants).items():
        runs[name] = {**cfg, **extra, "note": f"{a.run} declared variant: {json.dumps(extra)}"}
    idx = dict(schema="wholeplant-input-1", iters=con["train_settings"]["iters"], n_init=con["train_settings"]["n_init"], seed=con["train_settings"]["seed"],
               scene_scale_m=H, train=train, heldout_new=held, heldout_run01=[], runs=runs, cams=cams, organ_split="none", r31=True)
    json.dump(idx, open(d / "index.json", "w"), indent=1)
    json.dump(dict(title="tomato-recon-r31-input", id=DATASET, licenses=[dict(name="CC0-1.0")], isPrivate=True), open(d / "dataset-metadata.json", "w"), indent=1)
    k = st / "kernel"; k.mkdir(parents=True, exist_ok=True); shutil.copy2(a.kernel, k / "run.py")
    json.dump(dict(id=KERNEL, title="tomato-recon-r31-2dgs", code_file="run.py", language="python", kernel_type="script", is_private=True,
                   enable_gpu=True, machine_shape="NvidiaTeslaT4", enable_internet=True, dataset_sources=[DATASET], competition_sources=[], kernel_sources=[]),
              open(k / "kernel-metadata.json", "w"), indent=1)
    man = {str(p.relative_to(st)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file() and tag in str(p) or p.name == "index.json"}
    mp = rw / "inputs_manifest.json"; allm = json.load(open(mp)) if mp.exists() else {}
    allm[a.run] = dict(dataset=DATASET, kernel=KERNEL, files=man, kernel_sha256=hashlib.sha256(Path(a.kernel).read_bytes()).hexdigest())
    json.dump(allm, open(mp, "w"), indent=1)
    print(json.dumps(dict(run=a.run, train=len(train), heldout=held, H=round(H, 4), init=len(P1), hull=len(Ph))))


if __name__ == "__main__":
    main()
