"""Package the Kaggle input for the whole-plant runs (private dataset dolgfish/tomato-recon-wholeplant-input,
kernel dolgfish/tomato-recon-wholeplant-2dgs). One dataset, one kernel, all runs in sequence.

Runs (CONTRACT.md): R0 = run 01 settings on the new split; R1 is NOT packaged (A1 board refinement was rejected, and the
refinement moved cameras by at most 0.4 mm / 0.011 deg, so R1 would be R0); R2 = R0 + whole-plant masks with the band
excluded from the alpha loss; R3 = R2 + dense initialisation.

usage: python -m tomato3d.recon.wholeplant_prep --poses <charuco dir> --run01 <run 01 dir> --out <run dir> --stage <dir> [--with-r4]
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

DATASET, KERNEL = "dolgfish/tomato-recon-wholeplant-input", "dolgfish/tomato-recon-wholeplant-2dgs"
KERNEL_SRC = Path(__file__).with_name("kernel_2dgs_wholeplant.py")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--run01", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--stage", required=True); ap.add_argument("--with-r4", action="store_true")
    ap.add_argument("--only", nargs="*", help="emit only these run ids (a re-push starts with an empty /kaggle/working, so "
                                              "runs already downloaded are left out instead of being trained again)")
    a = ap.parse_args(argv); src, r01, rw, st = Path(a.poses), Path(a.run01), Path(a.out), Path(a.stage)
    con = json.load(open(rw / "contract.json")); po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    held, held01 = con["heldout_new"], con["heldout_run01"]
    views = [v for v in po["accepted"] if (rw / "masks" / (Path(v).stem + ".png")).exists()]
    train = [v for v in views if v not in held]
    d = st / "data"; d.mkdir(parents=True, exist_ok=True)
    res_list = [1512] + ([3024] if a.with_r4 else [])
    for res in res_list:
        (d / f"images_{res}").mkdir(exist_ok=True)
        for v in views:
            p = d / f"images_{res}" / v
            if not p.exists():
                Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((res, res), Image.LANCZOS).save(p, quality=95)
    for name, srcdir in (("masks_run01", r01 / "masks"), ("masks_whole", rw / "masks"), ("bands_whole", rw / "bands")):
        (d / name).mkdir(exist_ok=True)
        for v in views:
            shutil.copy2(srcdir / (Path(v).stem + ".png"), d / name / (Path(v).stem + ".png"))
    (d / "init").mkdir(exist_ok=True)
    P1, C1 = read_xyz(r01 / "plant_points.ply"); write_xyzrgb(d / "init" / "sparse_points.ply", P1, C1)
    rng = np.random.default_rng(con["seed"])
    for tag, ply in (("hull_run01", r01 / "visual_hull.ply"), ("hull_whole", rw / "visual_hull.ply")):
        if ply.exists():
            Ph, _ = read_xyz(ply); Ph = Ph[rng.choice(len(Ph), min(HULL_INIT_MAX, len(Ph)), replace=False)]
            write_xyzrgb(d / "init" / f"{tag}.ply", Ph, np.tile(np.array(HULL_COLOUR, np.float32), (len(Ph), 1)))
    dense = rw / "plant_points_dense.ply"
    if dense.exists():
        P3, C3 = read_xyz(dense); write_xyzrgb(d / "init" / "dense_points.ply", P3, C3)
    hull = json.load(open(rw / "masks_summary.json"))["_hull"]; H = float(hull["hi"][2] - max(hull["lo"][2], 0.0))
    cams = {}
    for res in res_list:
        K = ca.scaled_K(np.array(po["K_undist"]), R0, res)
        cams[str(res)] = {v: dict(K=K.tolist(), R_cw=po["views"][v]["R_cw"], t_cw=po["views"][v]["t_cw"]) for v in views}
    runs = {
        "R0": dict(note="run 01 settings on the new split (baseline)", res=1512, masks="masks_run01", fg_labels=[1],
                   band=None, init="init/sparse_points.ply", init_hull="init/hull_run01.ply"),
        "R2": dict(note="R0 + whole-plant masks with stem, 2 px band excluded from the alpha loss", res=1512, masks="masks_whole",
                   fg_labels=[1, 2], band="bands_whole", init="init/sparse_points.ply", init_hull="init/hull_whole.ply"),
    }
    if (d / "init" / "dense_points.ply").exists():
        runs["R3"] = dict(note="R2 + dense initialisation", res=1512, masks="masks_whole", fg_labels=[1, 2], band="bands_whole",
                          init="init/dense_points.ply", init_hull="init/hull_whole.ply")
    if a.with_r4 and "R3" in runs:
        runs["R4"] = dict(runs["R3"], note="R3 + 3024 px training", res=3024)
    if a.only:
        runs = {k: v for k, v in runs.items() if k in a.only}
        assert runs, f"none of {a.only} could be built"
    idx = dict(schema="wholeplant-input-1", iters=con["train_settings"]["iters"], n_init=con["train_settings"]["n_init"],
               seed=con["train_settings"]["seed"], scene_scale_m=H, train=train, heldout_new=held, heldout_run01=held01,
               runs=runs, cams=cams, organ_split="none")
    json.dump(idx, open(d / "index.json", "w"), indent=1)
    json.dump(dict(title="tomato-recon-wholeplant-input", id=DATASET, licenses=[dict(name="CC0-1.0")], isPrivate=True),
              open(d / "dataset-metadata.json", "w"), indent=1)
    k = st / "kernel"; k.mkdir(parents=True, exist_ok=True); shutil.copy2(KERNEL_SRC, k / "run.py")
    json.dump(dict(id=KERNEL, title="tomato-recon-wholeplant-2dgs", code_file="run.py", language="python", kernel_type="script",
                   is_private=True, enable_gpu=True, machine_shape="NvidiaTeslaT4", enable_internet=True,
                   dataset_sources=[DATASET], competition_sources=[], kernel_sources=[]), open(k / "kernel-metadata.json", "w"), indent=1)
    man = {str(p.relative_to(st)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(st.rglob("*")) if p.is_file()}
    json.dump(dict(dataset=DATASET, kernel=KERNEL, runs=list(runs), files=man), open(rw / "inputs_manifest.json", "w"), indent=1)
    print(json.dumps(dict(views=len(views), train=len(train), heldout=held, runs=list(runs), H=round(H, 4)), indent=1))


if __name__ == "__main__":
    main()
