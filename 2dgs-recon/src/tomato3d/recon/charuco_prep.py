"""Step 3 input packaging for the Kaggle 2DGS kernel (private dataset dolgfish/tomato-recon-charuco-input, kernel
dolgfish/tomato-recon-charuco-2dgs).

usage: python -m tomato3d.recon.charuco_prep --poses <charuco dir> --recon <recon dir> --stage <dir>
Writes <stage>/data/{index.json, images/, masks/, init/}, <stage>/data/dataset-metadata.json, <stage>/kernel/{run.py,
kernel-metadata.json} and <recon>/inputs_manifest.json (sha256 of every packaged file).
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

KERNEL_SRC = Path(__file__).with_name("kernel_2dgs_charuco.py")
DATASET, KERNEL = "dolgfish/tomato-recon-charuco-input", "dolgfish/tomato-recon-charuco-2dgs"
HULL_INIT_MAX = 60000
HULL_COLOUR = (0.30, 0.38, 0.24)


def write_xyzrgb(path: Path, P: np.ndarray, C: np.ndarray):
    rec = np.zeros(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for k, c in zip("xyz", P.T): rec[k] = c
    for k, c in zip("rgb", np.clip(C * 255, 0, 255).astype(np.uint8).T): rec[k] = c
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    path.write_bytes(hdr.encode() + rec.tobytes())


def read_xyz(path: Path) -> np.ndarray:
    b = path.read_bytes(); i = b.index(b"end_header\n") + 11; hdr = b[:i].decode()
    n = int(hdr.split("element vertex ")[1].split()[0]); has_rgb = "property uchar red" in hdr
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if has_rgb else [])
    r = np.frombuffer(b[i:], dtype=dt, count=n)
    return np.c_[r["x"], r["y"], r["z"]].astype(np.float32), (np.c_[r["r"], r["g"], r["b"]].astype(np.float32) / 255 if has_rgb else None)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--recon", required=True); ap.add_argument("--stage", required=True)
    a = ap.parse_args(argv); src, rc, st = Path(a.poses), Path(a.recon), Path(a.stage)
    con = json.load(open(rc / "contract.json")); po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    held = con["heldout"]; train = [v for v in po["accepted"] if v not in held]
    d = st / "data"; [(d / x).mkdir(parents=True, exist_ok=True) for x in ("images", "masks", "init")]
    K = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES)
    for v in train + held:
        if not (d / "images" / v).exists():
            Image.open(src / f"undist_{R0}" / v).convert("RGB").resize((MASK_RES, MASK_RES), Image.LANCZOS).save(d / "images" / v, quality=95)
        shutil.copy2(rc / "masks" / (Path(v).stem + ".png"), d / "masks" / (Path(v).stem + ".png"))
    P1, C1 = read_xyz(rc / "plant_points.ply"); write_xyzrgb(d / "init" / "plant_points.ply", P1, C1)
    Ph, _ = read_xyz(rc / "visual_hull.ply"); rng = np.random.default_rng(con["seed"])
    Ph = Ph[rng.choice(len(Ph), min(HULL_INIT_MAX, len(Ph)), replace=False)]
    write_xyzrgb(d / "init" / "hull_points.ply", Ph, np.tile(np.array(HULL_COLOUR, np.float32), (len(Ph), 1)))
    hull = json.load(open(rc / "masks_summary.json"))["hull"]; H = float(hull["hi"][2] - max(hull["lo"][2], 0.0))
    idx = dict(schema="recon-charuco-1", res=MASK_RES, iters=con["iters"], n_init=con["n_init"], seed=con["seed"], scene_scale_m=H,
               train=train, heldout=held, cams={v: dict(K=K.tolist(), R_cw=po["views"][v]["R_cw"], t_cw=po["views"][v]["t_cw"]) for v in train + held})
    json.dump(idx, open(d / "index.json", "w"), indent=1)
    json.dump(dict(title="tomato-recon-charuco-input", id=DATASET, licenses=[dict(name="CC0-1.0")], isPrivate=True), open(d / "dataset-metadata.json", "w"), indent=1)
    k = st / "kernel"; k.mkdir(parents=True, exist_ok=True); shutil.copy2(KERNEL_SRC, k / "run.py")
    json.dump(dict(id=KERNEL, title="tomato-recon-charuco-2dgs", code_file="run.py", language="python", kernel_type="script", is_private=True,
                   enable_gpu=True, machine_shape="NvidiaTeslaT4", enable_internet=True, dataset_sources=[DATASET], competition_sources=[], kernel_sources=[]),
              open(k / "kernel-metadata.json", "w"), indent=1)
    man = {str(p.relative_to(st)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(st.rglob("*")) if p.is_file()}
    json.dump(dict(dataset=DATASET, kernel=KERNEL, files=man), open(rc / "inputs_manifest.json", "w"), indent=1)
    print(len(train), "train,", len(held), "held-out; H =", round(H, 4), "m; init", len(P1), "+", len(Ph))


if __name__ == "__main__":
    main()
