# Kaggle CPU kernel — outputs for CONTRACT_ALL.md: per individual a point cloud, a TSDF mesh, recon_meta.json and a disc viewer from the
# trained surfels. Mesh extraction runs here (Rule 0). Input dataset dolgfish/tomato-recon-all-results: <iid>/{surfels_full.ply, results.json,
# split.json, points_report.json entry} + job poses. Nothing about the plant is measured here; shape only, organ_split none.
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-image"], check=True)
import numpy as np
def find(pat):
    r = glob.glob(f"/kaggle/input/**/{pat}", recursive=True); assert r, pat; return r[0]
CODE = os.path.dirname(find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt")); RES_DIR = os.path.dirname(find("all_results_manifest.json")); JOB = os.path.dirname(find("r31_human_masks.json"))
sys.path.insert(0, CODE)
from tomato3d.fuse3d import surfels_io as sio
from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.wholeplant_outputs import tsdf
from tomato3d.recon import charuco_viewer as cv
man = json.load(open(f"{RES_DIR}/all_results_manifest.json")); W = "/kaggle/working"; T0 = time.time(); summary = {}
for iid, info in man["individuals"].items():
    od = f"{W}/{iid}"; os.makedirs(od, exist_ok=True); run = f"R3-1b-style_{iid}"
    sp = json.load(open(f"{RES_DIR}/{iid}/split.json")); pz = json.load(open(f"{JOB}/{iid}/poses.json")); res = json.load(open(f"{RES_DIR}/{iid}/results.json"))["runs"][run]
    kd = f"{RES_DIR}/{iid}"; sf = sio.read_surfels_full(f"{kd}/{run}_surfels_full.ply"); n = len(sf["xyz"])
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for kk, c in zip("xyz", sf["xyz"].T): rec[kk] = c
    for kk, c in zip(("nx", "ny", "nz"), sf["normal"].T): rec[kk] = c
    for kk, c in zip("rgb", sf["rgb"].T.astype(np.uint8)): rec[kk] = c
    hdr = (f"ply\nformat binary_little_endian 1.0\ncomment board metres, z up; whole plant, no organ split; {run}; unvalidated, conditional_on_unvalidated_scale\n"
           f"element vertex {n}\n" + "".join(f"property float {x}\n" for x in ("x", "y", "z", "nx", "ny", "nz")) + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    open(f"{od}/{run}_points.ply", "wb").write(hdr.encode() + rec.tobytes()); shutil.copy(f"{kd}/{run}_surfels_full.ply", f"{od}/{run}_surfels_full.ply")
    try:
        mesh = tsdf(None, None, None, None, Path(kd), run, Path(f"{od}/{run}_mesh.ply"))
    except Exception as ex:                                                                  # noqa: BLE001
        mesh = dict(error=str(ex)[:200])
    meta = dict(individual=iid, source_folder=info["folder"], session_date=info["date"], run=run, pipeline="R3-1b-style (CONTRACT_ALL.md)",
                deviations=["D7 pose QC: 6-7 pose points with RMS <= 1.5 px@1512", "D11 masks: C3 U-Net (C2 for t2_02 where seeded), no M1/M2/M3/M5", "D12 no I2 mono depth", "D2 exposure compensation on"],
                views_accepted=len(pz["accepted"]), train_views=len(sp["train"]), heldout=sp["heldout"], results=res, mesh=mesh, surfels=int(n), human_reference=bool(info.get("human_reference")),
                acceptance=info["acceptance"], scale_status="conditional_on_unvalidated_scale (board gives metres; pot not measured)", pot_top_plane="not fitted in this batch",
                organ_split="none", note="shape only; no trait tables and no mm accuracy claims about the plant")
    json.dump(meta, open(f"{od}/recon_meta.json", "w"), indent=1)
    vi = f"/kaggle/tmp/vi_{iid}"; os.makedirs(vi, exist_ok=True); shutil.copy(f"{kd}/{run}_surfels_full.ply", f"{vi}/surfels_full.ply")
    cams = {}
    for v in pz["accepted"]:
        pv = pz["views"][v]; cams[v] = dict(K=ca.scaled_K(np.array(pv["K"]), pv["size"][0], 1512).tolist(), R_cw=pv["R_cw"], t_cw=pv["t_cw"])
    json.dump(dict(cameras=cams, views={v: dict(split="heldout" if v in sp["heldout"] else "train") for v in pz["accepted"]}), open(f"{vi}/done.json", "w"))
    cv.main(["--k", vi, "--out", f"{od}/viewer.html", "--title", f"{iid} · {run} · {info['acceptance']}", "--note", f"{iid} · 합격 아님 · 형태만, 기관 분할 없음 · 규모 미검증"])
    summary[iid] = dict(surfels=int(n), mesh=mesh if "error" in mesh else dict(vertices=mesh["vertices"], faces=mesh["faces"])); print(iid, summary[iid], f"{time.time() - T0:.0f}s", flush=True)
json.dump(summary, open(f"{W}/outputs_summary.json", "w"), indent=1); print("done", f"{time.time() - T0:.0f}s")
