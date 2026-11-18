# Kaggle kernel - one-zip v0.2 round trip, PC side: 3D-first groups for the surfels of one one-zip run (build/p1_onezip_v02_20261116_01/CONTRACT.md, section 6).
# Same calls as tomato3d/recon/kernel_3dfirst_ext.py (stages pre -> cands -> merge), in one GPU kernel; candidate layers "h2,sa" only (no Roboflow layer: user decision 2026-11-16).
# Added here, and nowhere in tomato3d/: the plant id is registered at run time, and the pilot's 6048 px HEIC photos are halved to 3024 px (the resolution this pipeline was run at).
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

IID = "ozp"
LAYERS = "h2,sa"
RES = 3024
T0 = time.time(); W = Path("/kaggle/working"); ROOT = Path("/tmp/root"); OUTD = W / "groups_out"


def info(**k):
    json.dump(dict(iid=IID, layers=LAYERS, seconds=round(time.time() - T0), **k), open(W / "run_info.json", "w"), indent=1)


def fail(msg):
    (W / "FAILED.txt").write_text(msg); info(status="failed", error=msg); print("FAILED", msg, flush=True); sys.exit(1)


def find(pat):
    g = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    if not g:
        fail(f"input not found: {pat}")
    return g[0]


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif", "ultralytics", "hydra-core", "iopath", "skan"], check=True)
CODE = Path(find("CODE_MARK.json")).parent; ROOT.mkdir(parents=True, exist_ok=True)
for x in ("tomato3d", "tools", "tests", "config"):
    if (CODE / x).exists():
        shutil.copytree(CODE / x, ROOT / x, dirs_exist_ok=True)
missing = []
for mf in glob.glob("/kaggle/input/**/MIRROR.json", recursive=True):
    base = Path(mf).parent
    for rel in json.load(open(mf))["files"]:
        src = base / "root" / rel
        if not src.exists():
            g = [p for p in glob.glob(f"{base}/**/{Path(rel).name}", recursive=True) if p.replace("\\", "/").endswith(rel)]
            if not g:
                missing.append(rel); continue
            src = Path(g[0])
        dst = ROOT / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src, dst)
if missing:
    fail(f"{len(missing)} mirrored files missing, first {missing[:5]}")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "tests/test_all3d_first.py", "tests/test_all3d_merge.py"], cwd=str(ROOT))
if r.returncode:
    fail("unit tests failed (tests/test_all3d_first.py, tests/test_all3d_merge.py)")

import cv2
import numpy as np
import pillow_heif
from PIL import Image
from tomato3d.organs import all3d_first as A
from tomato3d.organs import all_compound as AC

pillow_heif.register_heif_opener()
AC.POSES[IID] = AC.ALL / "stage2_new" / IID / "poses.json"                       # run-time registration of this plant (no source edit)
AC.PHOTO_ROOTS = [ROOT / "heic"]; (ROOT / "heic").mkdir(exist_ok=True)
pz = json.load(open(AC.POSES[IID], encoding="utf-8")); photos = {Path(p).stem.lower(): p for p in glob.glob("/kaggle/input/**/*.heic", recursive=True) + glob.glob("/kaggle/input/**/*.HEIC", recursive=True)}
nph = 0
for v in pz["accepted"]:                                                          # 6048 -> 3024, area average; written under the name the pipeline's index looks for
    stem = Path(v).stem; src = photos.get(stem.lower())
    if src is None:
        fail(f"photo of {v} not found in the attached photo dataset")
    im = np.asarray(Image.open(src).convert("RGB"))
    if im.shape[0] != im.shape[1] or im.shape[0] % RES:
        fail(f"{v}: size {im.shape}")
    if im.shape[0] != RES:
        im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
    Image.fromarray(im).save(ROOT / "heic" / f"{stem}.HEIC", format="JPEG", quality=97); nph += 1
print("photos prepared", nph, round(time.time() - T0), "s", flush=True)
if not (A.OUT / "size_prior.json").exists():
    fail("size_prior.json not in the mirror (attach the output of kernel tomato-3dfirst-sizeprior)")

try:
    for s in ("inputs", "prep", "geometry"):
        A.main(["--id", IID, "--step", s]); print("step", s, "done", round(time.time() - T0), "s", flush=True)
    import torch
    ckpt = find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt"); vendor = os.path.dirname(os.path.dirname(find("sam2_vendor/sam2/__init__.py"))); m1 = find("M1_s0_best.pt")
    sys.path.insert(0, vendor)
    cfg = dict(backend="sam2", model_name="SAM 2.1 hiera tiny, leaflet-tuned v5 (4plants)", repository=vendor, checkpoint=ckpt,
               checkpoint_sha256="161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d", model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
               device="cuda" if torch.cuda.is_available() else "cpu", threads=4)
    json.dump(cfg, open("/tmp/sam2_config.json", "w")); print("device", cfg["device"], flush=True)
    if cfg["device"] != "cuda":
        fail("no GPU")
    from tomato3d.organs import all3d_cands_job as J
    c = A.cell(IID)
    J.run(c, c / "cands", Path("/tmp/sam2_config.json"), CODE / "t1_parameters.json", Path(m1), json.load(open(A.OUT / "size_prior.json", encoding="utf-8")), layers=LAYERS)
    print("cands done", round(time.time() - T0), "s", flush=True)
    from tomato3d.organs import all3d_merge as MG
    summ = MG.run(IID); keys = ("voting_views", "layers_used", "merges", "groups", "unassigned_share", "estimated_user_minutes")
    print(json.dumps({k: summ.get(k) for k in keys}), flush=True)
except SystemExit as e:
    if e.code not in (0, None) and not (W / "FAILED.txt").exists():
        fail(f"step exited: {e}")
    raise
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"{type(e).__name__}: {e}")

n = 0
for f in c.rglob("*"):                                                            # everything of the cell except the prepared photos and mask copies
    rel = f.relative_to(c)
    if f.is_file() and rel.parts[0] not in ("photos", "masks", "rf_inputs"):
        d = OUTD / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, d); n += 1
shutil.copy(A.OUT / "size_prior.json", OUTD / "size_prior.json")
info(status="complete", exported_files=n, merge={k: summ.get(k) for k in keys}); print("done", round(time.time() - T0), "s", flush=True)
