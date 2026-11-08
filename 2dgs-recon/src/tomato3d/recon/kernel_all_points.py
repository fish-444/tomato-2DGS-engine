# Kaggle CPU kernel — K3c for CONTRACT_ALL.md: init I1 (pycolmap SIFT 3024 px, 32,768 features, fixed board poses, foreground = the K3a masks)
# for every SINGLE-camera individual, one after another, resumable per individual. t2_01 (two cameras) is not handled here.
# Output per individual: <iid>/plant_points.ply, <iid>/sparse_checks.json. Nothing runs locally (Rule 0).
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif", "pycolmap"], check=True)
import cv2, numpy as np, pillow_heif
from PIL import Image
pillow_heif.register_heif_opener()
T0 = time.time()
def find(pat):
    r = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    if not r:                                                                                   # folder uploaded as a zip and not extracted
        import zipfile
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            zipfile.ZipFile(z).extractall("/kaggle/tmp/zips")
        r = glob.glob(f"/kaggle/tmp/zips/**/{pat}", recursive=True)
    assert r, pat; print("found", pat, "->", r[0], flush=True); return r[0]
CODE = os.path.dirname(find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt")); JOB = os.path.dirname(find("r31_human_masks.json"))
MASKS = os.path.dirname(find("masks_summary.json"))                                          # K3a output root: <iid>/masks, split.json
sys.path.insert(0, CODE)
from tomato3d.recon import wholeplant_sparse as ws
PHOTO = {}
for ds in glob.glob("/kaggle/input/**/census.csv", recursive=True):
    d = os.path.dirname(ds)
    for sub in os.listdir(d):
        if os.path.isdir(f"{d}/{sub}") and (sub.startswith("t2_") or sub.startswith("n_")): PHOTO[sub] = f"{d}/{sub}"
IDS = ["n_04", "n_01", "t2_02", "n_06", "n_05", "n_02", "t2_05", "n_03", "t2_04"]      # t2_03 held back: C3 misses its bright green leaves and fruit (hull 10k voxels); t2_01: two cameras
OUT = "/kaggle/working"; report = {}
for iid in IDS:
    if os.path.exists(f"{OUT}/{iid}/plant_points.ply"):
        print("skip", iid); continue
    t1 = time.time(); pz = json.load(open(f"{JOB}/{iid}/poses.json")); sp = json.load(open(f"{MASKS}/{iid}/split.json")); acc = pz["accepted"]
    v0 = pz["views"][acc[0]]; K = v0["K"]; Wd = v0["size"][0]
    assert all(pz["views"][v]["K"] == K for v in acc), f"{iid}: more than one camera"
    src = f"/kaggle/tmp/{iid}/src"; rc = f"/kaggle/tmp/{iid}/rc"; work = f"/kaggle/tmp/colmap_{iid}"
    for d in (f"{src}/undist_{Wd}", f"{rc}/masks"):
        os.makedirs(d, exist_ok=True)
    Kn, dist = np.array(K), np.array(v0["dist"])
    for v in acc:
        im = np.asarray(Image.open(f"{PHOTO[iid]}/{v[:-4]}.HEIC").convert("RGB")); assert im.shape[:2] == (Wd, Wd), (iid, v, im.shape)
        Image.fromarray(cv2.undistort(im, Kn, dist, None, Kn)).save(f"{src}/undist_{Wd}/{v}", quality=95)
        shutil.copy(f"{MASKS}/{iid}/masks/{Path(v).stem}.png", f"{rc}/masks/{Path(v).stem}.png")
    views = {v: dict(R_cw=pz["views"][v]["R_cw"], t_cw=pz["views"][v]["t_cw"]) for v in acc}
    json.dump(dict(full_size=[Wd, Wd], K_full=K, dist=v0["dist"], undistorted_res=Wd, K_undist=K, views=views, accepted=acc), open(f"{src}/poses.json", "w"))
    json.dump(dict(heldout_new=sp["heldout"]), open(f"{rc}/contract.json", "w")); json.dump(dict(_hull=sp["hull"]), open(f"{rc}/masks_summary.json", "w"))
    shutil.rmtree(work, ignore_errors=True)
    try:
        ws.main(["--poses", src, "--out", rc, "--work", work])
    except Exception as ex:                                                                  # noqa: BLE001
        report[iid] = dict(error=str(ex)[:300]); print("FAILED", iid, ex, flush=True); json.dump(report, open(f"{OUT}/points_report.json", "w"), indent=1); continue
    os.makedirs(f"{OUT}/{iid}", exist_ok=True); shutil.copy(f"{rc}/plant_points_dense.ply", f"{OUT}/{iid}/plant_points.ply"); shutil.copy(f"{rc}/sparse_dense_checks.json", f"{OUT}/{iid}/sparse_checks.json")
    ck = json.load(open(f"{rc}/sparse_dense_checks.json")); report[iid] = dict(plant_points=ck["plant_points"], rms=ck["plant_reproj_rms_px_at_res"], per_view=ck["per_view"], seconds=round(time.time() - t1))
    print(iid, report[iid], f"{time.time() - T0:.0f}s", flush=True); json.dump(report, open(f"{OUT}/points_report.json", "w"), indent=1)
    shutil.rmtree(f"/kaggle/tmp/{iid}", ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
print("done", f"{time.time() - T0:.0f}s", flush=True)
