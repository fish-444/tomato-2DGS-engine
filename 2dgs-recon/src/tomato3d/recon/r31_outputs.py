"""R3-1 Step 7 for the chosen run: point cloud, TSDF mesh, surfels copy, recon_meta.json, and the disc viewer input (done.json shim).

The pot-rim scale check depends only on the photos, poses and v2 masks (not on the plant model), so the v2 result is cited, not re-fitted.

usage: python -m tomato3d.recon.r31_outputs --poses <charuco dir> --v2 <wholeplant run dir> --out <r3-1 run dir> --k <kernel out dir>
       --run R3-1b --acceptance "not accepted (best unaccepted)"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.wholeplant_outputs import tsdf


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("poses", "v2", "out", "k", "run", "acceptance"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--r31", default=None, help="run dir holding contract.json and scores_<run>.json (default: --out)")
    ap.add_argument("--contamination", default=None, help="contam_partA.json to embed its verdict")
    a = ap.parse_args(argv); src, v2, rw, kd = Path(a.poses), Path(a.v2), Path(a.out), Path(a.k); r31 = Path(a.r31) if a.r31 else rw
    po = json.load(open(src / "poses.json")); con = json.load(open(r31 / "contract.json")); od = rw / "outputs"; od.mkdir(parents=True, exist_ok=True)
    sf = sio.read_surfels_full(kd / f"{a.run}_surfels_full.ply"); shutil.copy2(kd / f"{a.run}_surfels_full.ply", od / f"{a.run}_surfels_full.ply")
    rec = np.zeros(len(sf["xyz"]), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for kk, c in zip("xyz", sf["xyz"].T): rec[kk] = c
    for kk, c in zip(("nx", "ny", "nz"), sf["normal"].T): rec[kk] = c
    for kk, c in zip("rgb", sf["rgb"].T.astype(np.uint8)): rec[kk] = c
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment board metres, z up; whole plant, no organ split; acceptance: "
           f"{a.acceptance}\nelement vertex {len(rec)}\n" + "".join(f"property float {x}\n" for x in ("x", "y", "z", "nx", "ny", "nz")) +
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    (od / f"{a.run}_points.ply").write_bytes(hdr.encode() + rec.tobytes())
    mesh = tsdf(src, rw, po, con["train"], kd, a.run, od / f"{a.run}_mesh.ply")
    res = json.load(open(kd / "results.json"))["runs"][a.run]; sc = json.load(open(r31 / f"scores_{a.run}.json"))
    contam = None
    if a.contamination:
        c = json.load(open(a.contamination)); contam = dict(verdict=c["verdict"]["verdict"], contract="CONTRACT_CONTAM.md", stem_base=c["counts"][a.run]["stem_base"] if a.run in c["counts"] else None,
                                                             clipped_clouds_same=c["verdict"]["clipped_clouds_same"], pot_top_plane_z_m=0.066)
    v2meta = json.load(open(v2 / "outputs" / "recon_meta.json"))
    meta = dict(run=a.run, acceptance=a.acceptance, organ_split="none", frame="board metres, z up", poses="run 01 board poses, fixed",
                contract=str(r31 / "CONTRACT_R3-1.md"), contamination=contam, masks=con["runs"], heldout_fresh=con["heldout_fresh"],
                scores_fresh_human=sc.get("human"), scores_fresh_auto=sc.get("auto"), secondary_sets={k: v["medians"] for k, v in sc["secondary_sets"].items()},
                train_metrics=res.get("train"), mesh=mesh, surfels=int(len(sf["xyz"])),
                scale_check=dict(v2meta.get("scale_check", {}), note="cited from the v2 run: photos + poses only, independent of the plant model"),
                note="shape only; no trait tables and no mm accuracy claims about the plant")
    json.dump(meta, open(od / "recon_meta.json", "w"), indent=1)
    # viewer input: the disc viewer reads surfels_full.ply + done.json (cameras, splits)
    vi = rw / "viewer_input"; vi.mkdir(exist_ok=True); shutil.copy2(kd / f"{a.run}_surfels_full.ply", vi / "surfels_full.ply")
    R0 = po["undistorted_res"]; K = ca.scaled_K(np.array(po["K_undist"]), R0, 1512).tolist(); views = con["train"] + con["heldout_fresh"]
    json.dump(dict(cameras={v: dict(K=K, R_cw=po["views"][v]["R_cw"], t_cw=po["views"][v]["t_cw"]) for v in views},
                   views={v: dict(split="heldout" if v in con["heldout_fresh"] else "train") for v in views}), open(vi / "done.json", "w"))
    print(json.dumps(dict(mesh=mesh, surfels=len(sf["xyz"])), indent=1))


if __name__ == "__main__":
    main()
