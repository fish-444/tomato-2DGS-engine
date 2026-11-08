# Kaggle GPU kernel - candidate masks for EVERY posed photo of ONE individual (CONTRACT p1_all_3dfirst_20261027_01, Step 1.4).
# IID / LAYERS are patched by the driver (tomato3d/organs/all3d_kaggle.py). Inputs: dolgfish/tomato-3dfirst-code (tomato3d, tools, t1_parameters.json, size_prior.json),
# dolgfish/tomato-3dfirst-job-<iid> (photos, masks, detect_input.json, cached Roboflow responses when the layer is requested), dolgfish/tomato-recon-code (SAM v5 + vendored sam2),
# dolgfish/tomato-mask-input (M1). No Roboflow key is present here. Output /kaggle/working/out/<iid>/<stem>_cands.json (+ _rf.json for the RF layer).
import glob, json, os, subprocess, sys, time
from pathlib import Path

IID = "n_04"
LAYERS = "h2,sa"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "hydra-core", "iopath", "skan"], check=True)
T0 = time.time()


def find(pat):
    r = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    assert r, pat
    print("found", pat, "->", r[0], flush=True)
    return r[0]


CODE = os.path.dirname(find("t1_parameters.json"))
JOB = os.path.dirname(find("detect_input.json"))
CKPT = find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt")
VENDOR = os.path.dirname(os.path.dirname(find("sam2_vendor/sam2/__init__.py")))
M1 = find("M1_s0_best.pt")
sys.path.insert(0, CODE)
os.chdir(CODE)
import torch

cfg = dict(backend="sam2", model_name="SAM 2.1 hiera tiny, leaflet-tuned v5 (4plants)", repository=VENDOR, checkpoint=CKPT,
           checkpoint_sha256="161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d", model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
           device="cuda" if torch.cuda.is_available() else "cpu", threads=4)
json.dump(cfg, open("/kaggle/working/sam2_config.json", "w"))
print("device", cfg["device"], "individual", IID, "layers", LAYERS, flush=True)
from tomato3d.organs import all3d_cands_job as J

J.run(Path(JOB), Path(f"/kaggle/working/out/{IID}"), Path("/kaggle/working/sam2_config.json"), Path(f"{CODE}/t1_parameters.json"), Path(M1),
      json.load(open(f"{CODE}/size_prior.json", encoding="utf-8")), layers=LAYERS)
print("done", round(time.time() - T0), "s", flush=True)
