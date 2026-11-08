# Kaggle kernel - p1_all_3dfirst_20261027_01, Amendment A3 (external compute only). STAGE / IID / LAYERS are patched by tomato3d/organs/all3d_ext.py.
# Every input dataset (and every source kernel's output) carries MIRROR.json + root/<path relative to the project root>; the files are linked into a mirror root /tmp/root so the code runs unchanged.
# First step: the unit tests (tests/test_all3d_first.py); on failure the kernel stops and reports. Output: /kaggle/working/mirror/{MIRROR.json, root/...} (this stage's results only) + run_info.json.
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

STAGE = "pre"
IID = "n_04"
LAYERS = "h2,sa"
T0 = time.time(); W = Path("/kaggle/working"); ROOT = Path("/tmp/root"); MIR = W / "mirror"


def info(**k):
    json.dump(dict(stage=STAGE, iid=IID, layers=LAYERS, seconds=round(time.time() - T0), **k), open(W / "run_info.json", "w"), indent=1)


def fail(msg):
    (W / "FAILED.txt").write_text(msg); info(status="failed", error=msg); print("FAILED", msg, flush=True); sys.exit(1)


pk = {"pre": ["pillow-heif"], "cands": ["ultralytics", "hydra-core", "iopath", "skan"]}.get(STAGE, [])
if pk:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pk], check=True)
code = glob.glob("/kaggle/input/**/CODE_MARK.json", recursive=True)
if not code:
    fail("code dataset (CODE_MARK.json) not found")
CODE = Path(code[0]).parent; ROOT.mkdir(parents=True, exist_ok=True)
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
from tomato3d.organs import all3d_first as A
from tomato3d.organs import all_compound as AC
AC.PHOTO_ROOTS = [ROOT / "heic"]


def export(paths):
    """Copy this stage's results (files or folders under ROOT, symlinks resolved) into the output mirror."""
    files = []
    for p in paths:
        p = Path(p)
        for f in ([p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]):
            rel = f.relative_to(ROOT).as_posix(); d = MIR / "root" / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, d); files.append(rel)
    MIR.mkdir(parents=True, exist_ok=True); json.dump(dict(files=sorted(set(files))), open(MIR / "MIRROR.json", "w"), indent=0); return len(files)


try:
    if STAGE == "size_prior":
        A.main(["--step", "size_prior"]); n = export([A.OUT / "size_prior.json"])
    elif STAGE == "pre":
        for s in ("inputs", "prep", "geometry"):
            A.main(["--id", IID, "--step", s])
        n_rf = A.write_rf_inputs(IID); c = A.cell(IID)
        for f in (c / "rf_inputs").glob("*.jpg"):                                      # distinct suffix so the local download can pick them by name
            if not f.name.endswith(".rfin.jpg"):
                f.rename(f.with_name(f.stem + ".rfin.jpg"))
        n = export([c]); print("rf inputs", n_rf, flush=True)
    elif STAGE == "merge":
        from tomato3d.organs import all3d_merge as MG
        summ = MG.run(IID); print(json.dumps({k: summ[k] for k in ("voting_views", "layers_used", "merges", "groups", "unassigned_share", "estimated_user_minutes")}), flush=True)
        n = export([A.cell(IID) / "merge"])
    elif STAGE == "cands":
        import torch
        def find(pat):
            g = glob.glob(f"/kaggle/input/**/{pat}", recursive=True); assert g, pat; return g[0]
        ckpt = find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt"); vendor = os.path.dirname(os.path.dirname(find("sam2_vendor/sam2/__init__.py"))); m1 = find("M1_s0_best.pt")
        sys.path.insert(0, vendor)                                                      # the vendored sam2 package (imported by all3d_cands_job before load_sam adds it)
        cfg = dict(backend="sam2", model_name="SAM 2.1 hiera tiny, leaflet-tuned v5 (4plants)", repository=vendor, checkpoint=ckpt,
                   checkpoint_sha256="161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d", model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
                   device="cuda" if torch.cuda.is_available() else "cpu", threads=4)
        json.dump(cfg, open("/tmp/sam2_config.json", "w")); print("device", cfg["device"], flush=True)
        if cfg["device"] != "cuda":
            fail("no GPU in a cands kernel")
        from tomato3d.organs import all3d_cands_job as J
        c = A.cell(IID); out = c / ("cands" if LAYERS == "h2,sa" else "cands_rf")
        J.run(c, out, Path("/tmp/sam2_config.json"), CODE / "t1_parameters.json", Path(m1), json.load(open(A.OUT / "size_prior.json", encoding="utf-8")), layers=LAYERS)
        n = export([out])
    else:
        fail(f"unknown stage {STAGE}")
except SystemExit as e:
    if e.code not in (0, None) and not (W / "FAILED.txt").exists():
        fail(f"step exited: {e}")
    raise
except Exception as e:
    import traceback; traceback.print_exc(); fail(f"{type(e).__name__}: {e}")
info(status="complete", exported_files=n); print("done", round(time.time() - T0), "s", flush=True)
