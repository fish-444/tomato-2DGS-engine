# Kaggle GPU kernel - detector candidates (P-H2, P-SA, P-RF drawing) for the label views, CONTRACT p1_all_compound_20261026_01 Amendment 1.
# Inputs: dolgfish/tomato-all-detect-code (tomato3d + tools + t1_parameters.json), dolgfish/tomato-all-detect-job (per individual: undist_3024, masks,
# detect_input.json, cached Roboflow responses), dolgfish/tomato-recon-code (SAM 2.1 tiny leaflet v5 checkpoint + vendored sam2),
# dolgfish/tomato-mask-input (M1_s0_best.pt). No Roboflow key is present here. Nothing is deleted; raw candidates are written to /kaggle/working/out.
import glob, json, os, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "hydra-core", "iopath", "skan"], check=True)
T0 = time.time()
def find(pat):
    r = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    assert r, pat; print("found", pat, "->", r[0], flush=True); return r[0]
CODE = os.path.dirname(find("t1_parameters.json")); JOB = os.path.dirname(os.path.dirname(find("*/detect_input.json")))
CKPT = find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt"); VENDOR = os.path.dirname(os.path.dirname(find("sam2_vendor/sam2/__init__.py")))
M1 = find("M1_s0_best.pt")
sys.path.insert(0, CODE); os.chdir(CODE)
import torch
cfg = dict(backend="sam2", model_name="SAM 2.1 hiera tiny, leaflet-tuned v5 (4plants)", repository=VENDOR, checkpoint=CKPT,
           checkpoint_sha256="161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d", model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
           device="cuda" if torch.cuda.is_available() else "cpu", threads=4)
json.dump(cfg, open("/kaggle/working/sam2_config.json", "w"))
print("device", cfg["device"], flush=True)
from tomato3d.organs import all_detect_job as J
J.run(Path(JOB), Path("/kaggle/working/out"), Path("/kaggle/working/sam2_config.json"), Path(f"{CODE}/t1_parameters.json"), Path(M1))
print("done", round(time.time() - T0), "s", flush=True)
