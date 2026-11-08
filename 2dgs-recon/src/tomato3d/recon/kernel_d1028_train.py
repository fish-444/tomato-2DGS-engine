# Adapted from tomato3d/recon/kernel_all_train.py for CONTRACT p1_recon_d1028_20261028_01 (individual list and photo folder prefix only; nothing else changed).
# Kaggle T4 kernel — K3d for CONTRACT_ALL.md: ONE individual per push (edit IID). Builds the input layout that kernel_2dgs_r31.py expects
# (images_1512, masks/bands/soft from K3a, I1 points from K3c, hull init from K3a), then runs that trainer UNCHANGED (R3-1b configuration,
# deviations D2 exposure ON, D11, D12 as declared). Nothing runs locally (Rule 0).
IID = "e_05"
RUN = f"R3-1b-style_{IID}"
import glob, json, os, runpy, shutil, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif"], check=True)
import cv2, numpy as np, pillow_heif
from PIL import Image
pillow_heif.register_heif_opener()
def find(pat):
    r = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    if not r:                                                                                   # folder uploaded as a zip and not extracted
        import zipfile
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            zipfile.ZipFile(z).extractall("/kaggle/tmp/zips")
        r = glob.glob(f"/kaggle/tmp/zips/**/{pat}", recursive=True)
    assert r, pat; print("found", pat, "->", r[0], flush=True); return r[0]
CODE = os.path.dirname(find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt")); JOB = os.path.dirname(find("r31_human_masks.json"))
MASKS = os.path.dirname(find("masks_summary.json")); POINTS = os.path.dirname(find("points_report.json"))
sys.path.insert(0, CODE)
from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_prep import HULL_COLOUR, HULL_INIT_MAX, read_xyz, write_xyzrgb
PHOTO = {}
for ds in glob.glob("/kaggle/input/**/census.csv", recursive=True):
    d = os.path.dirname(ds)
    for sub in os.listdir(d):
        if os.path.isdir(f"{d}/{sub}") and sub.startswith("e_"): PHOTO[sub] = f"{d}/{sub}"
T0 = time.time(); RES = 1512; tag = IID.replace("-", "_")
pz = json.load(open(f"{JOB}/{IID}/poses.json")); sp = json.load(open(f"{MASKS}/{IID}/split.json")); acc = pz["accepted"]; held, train = sp["heldout"], sp["train"]
v0 = pz["views"][acc[0]]; K, dist, Wd = np.array(v0["K"]), np.array(v0["dist"]), v0["size"][0]
assert all(pz["views"][v]["K"] == v0["K"] for v in acc), "single camera expected"
ST = "/tmp/in/stage"; shutil.rmtree(ST, ignore_errors=True)
for d in ("images_1512", f"masks_{tag}", f"bands_{tag}", f"soft_{tag}", "init"):
    os.makedirs(f"{ST}/{d}", exist_ok=True)
for v in acc:
    im = np.asarray(Image.open(f"{PHOTO[IID]}/{v[:-4]}.HEIC").convert("RGB")); assert im.shape[:2] == (Wd, Wd)
    Image.fromarray(cv2.resize(cv2.undistort(im, K, dist, None, K), (RES, RES), interpolation=cv2.INTER_AREA)).save(f"{ST}/images_1512/{v}", quality=95)
    for src_, dst in (("masks", f"masks_{tag}"), ("bands", f"bands_{tag}"), ("soft", f"soft_{tag}")):
        shutil.copy(f"{MASKS}/{IID}/{src_}/{Path(v).stem}.png", f"{ST}/{dst}/{Path(v).stem}.png")
P1, C1 = read_xyz(Path(f"{POINTS}/{IID}/plant_points.ply")); write_xyzrgb(Path(f"{ST}/init/points_{tag}.ply"), P1, C1)
Ph, _ = read_xyz(Path(f"{MASKS}/{IID}/visual_hull.ply")); rng = np.random.default_rng(0); Ph = Ph[rng.choice(len(Ph), min(HULL_INIT_MAX, len(Ph)), replace=False)]
write_xyzrgb(Path(f"{ST}/init/hull_{tag}.ply"), Ph, np.tile(np.array(HULL_COLOUR, np.float32), (len(Ph), 1)))
H = float(Ph[:, 2].max() - max(Ph[:, 2].min(), 0.0)); Ks = ca.scaled_K(K, Wd, RES)
cams = {str(RES): {v: dict(K=Ks.tolist(), R_cw=pz["views"][v]["R_cw"], t_cw=pz["views"][v]["t_cw"]) for v in acc}}
cfg = dict(note=f"{RUN}: R3-1b configuration with D2 exposure ON, D11 C3 masks, D12 no I2", res=RES, masks=f"masks_{tag}", fg_labels=[1, 2], band=f"bands_{tag}",
           soft=f"soft_{tag}", init=f"init/points_{tag}.ply", init_hull=f"init/hull_{tag}.ply", exposure=True)
json.dump(dict(schema="wholeplant-input-1", iters=7000, n_init=150000, seed=0, scene_scale_m=H, train=train, heldout_new=held, heldout_run01=[], runs={RUN: cfg},
               cams=cams, organ_split="none", r31=True), open(f"{ST}/index.json", "w"), indent=1)
print("staged", IID, "train", len(train), "held", held, "points", len(P1), "hull", len(Ph), "H", round(H, 4), f"{time.time() - T0:.0f}s", flush=True)
runpy.run_path(f"{CODE}/tomato3d/recon/kernel_2dgs_r31.py", run_name="__main__")
