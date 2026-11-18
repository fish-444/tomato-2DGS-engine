"""PC side of the v0.2 round trip: turn one one-zip run folder (stages 1-6 output) into the input dataset of the 3D-first group pipeline.

Nothing in tomato3d/ or in other build folders is written. The pipeline expects each plant under project-relative paths
(build/p1_recon_all_20261025_01/...); those paths exist only inside the dataset mirror (root/...) and in the kernel's temporary root.

usage: python make_inputs.py <recon_out> <dest> [--iid ozp]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

RES = 3024                      # the 3D-first pipeline was run on 3024 px photos (nine plants); the pilot's 6048 px photos are halved in the kernel
ALL = "build/p1_recon_all_20261025_01"


def scaled_K(K, src, dst):      # same pixel-centre rule as tomato3d.recon.charuco_a3.scaled_K
    s = dst / src
    return [[K[0][0] * s, 0.0, (K[0][2] + 0.5) * s - 0.5], [0.0, K[1][1] * s, (K[1][2] + 0.5) * s - 0.5], [0.0, 0.0, 1.0]]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("recon_out", type=Path); ap.add_argument("dest", type=Path); ap.add_argument("--iid", default="ozp"); a = ap.parse_args()
    src, iid = a.recon_out, a.iid; root = a.dest / "root"
    if a.dest.exists():
        shutil.rmtree(a.dest)
    po = json.load(open(src / "poses" / "poses.json", encoding="utf-8")); full = po["full_size"][0]; assert po["full_size"][0] == po["full_size"][1]
    K = scaled_K(po["K_full"], full, RES); f = RES / full
    views = {v: dict(R_cw=d["R_cw"], t_cw=d["t_cw"], center_world=d.get("center_world"), rms_px=(None if d.get("rms_px") is None else d["rms_px"] * f), accepted=d.get("accepted"),
                     K=K, dist=po["dist"], size=[RES, RES]) for v, d in po["views"].items()}
    stage2 = dict(frame=po["frame"], views=views, accepted=po["accepted"],
                  adapter=dict(source="one-zip poses/poses.json", full_size=full, photo_res=RES, K="K_full scaled with the pixel-centre rule", rms_px=f"x {f} (reported at {full} px)"))
    files = {}

    def put(rel, path=None, obj=None):
        d = root / rel; d.parent.mkdir(parents=True, exist_ok=True)
        if obj is not None:
            d.write_text(json.dumps(obj), encoding="utf-8")
        else:
            shutil.copy2(path, d)
        files[rel] = sha(d)

    put(f"{ALL}/outputs/{iid}/R3-1b-style_{iid}_surfels_full.ply", src / "surfels_full.ply")
    put(f"{ALL}/stage2_new/{iid}/poses.json", obj=stage2)
    put(f"{ALL}/k3a/{iid}/split.json", src / "split.json")
    n = 0
    for v in po["accepted"]:
        m = src / "masks" / (Path(v).stem + ".png"); assert m.exists(), m
        put(f"{ALL}/k3a/{iid}/masks/{m.name}", m); n += 1
    json.dump(dict(files=sorted(files)), open(a.dest / "MIRROR.json", "w"), indent=0)
    meta = dict(iid=iid, source_run=str(src), surfels_full_sha256=sha(src / "surfels_full.ply"), accepted_views=len(po["accepted"]), masks=n, photo_res=RES, sha256=files)
    json.dump(meta, open(a.dest / "INPUT_META.json", "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("iid", "surfels_full_sha256", "accepted_views", "masks", "photo_res")}))


if __name__ == "__main__":
    main()
