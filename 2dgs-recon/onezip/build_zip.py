"""Build microtom_recon_onezip_<version>.zip from this repository: the Kaggle / Colab package that runs the whole chain (photos -> 2DGS surfels -> optional T-Prox stage 7).
It copies the modules the package needs from ../src/tomato3d, applies the declared path edits (EDITS below, every one an exact-text replacement), adds the entry files of this folder,
scans the result for personal data and zips it. Nothing is written outside --work and --out.
usage: python onezip/build_zip.py [--out onezip/dist] [--work <temp folder>]"""
import argparse, hashlib, json, re, shutil, sys, tempfile, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent; REPO = HERE.parent; SRC = REPO / "src"; VERSION = "0.2.2"
KUSER = "dolg" + "fish"                                                                   # the owner's Kaggle name: allowed in the repository, never inside the zip
COPY = ["tomato3d/__init__.py", "tomato3d/recon/__init__.py", "tomato3d/fuse3d/__init__.py", "tomato3d/fuse3d/surfels_io.py", "tomato3d/organs/__init__.py", "tomato3d/organs/pc_absorb.py",
        "tomato3d/organs/pc_rachis.py", "tomato3d/organs/skeleton_io.py"] + [f"tomato3d/recon/{m}.py" for m in (
        "charuco_photos", "charuco_a3", "charuco_masks", "charuco_sparse", "charuco_prep", "wholeplant_contract", "wholeplant_masks", "wholeplant_sparse", "wholeplant_accept",
        "r31_contract", "r31_masks", "r31_score", "kernel_2dgs_r31")]
ENTRY = ["run_recon.py", "tprox_stage.py", "stem_masks.py", "RUN_KAGGLE.ipynb", "RUN_COLAB.ipynb", "README_KO.md", "STAGE_SOURCES.md", "VERSIONS.txt", "EDITS.csv"]
# (file under microtom_recon/, exact old text, new text)
EDITS = [
    ("tomato3d/recon/charuco_photos.py", 'if p.suffix.lower() == ".heic"), key=', 'if p.suffix.lower() in (".heic", ".heif", ".jpg", ".jpeg", ".png")), key='),
    ("tomato3d/recon/charuco_prep.py", f"(private dataset {KUSER}/tomato-recon-charuco-input, kernel", "(private dataset <kaggle_user>/tomato-recon-charuco-input, kernel"),
    ("tomato3d/recon/charuco_prep.py", f"{KUSER}/tomato-recon-charuco-2dgs).", "<kaggle_user>/tomato-recon-charuco-2dgs)."),
    ("tomato3d/recon/charuco_prep.py", f'DATASET, KERNEL = "{KUSER}/tomato-recon-charuco-input", "{KUSER}/tomato-recon-charuco-2dgs"', 'DATASET, KERNEL = "<kaggle_user>/tomato-recon-charuco-input", "<kaggle_user>/tomato-recon-charuco-2dgs"'),
    ("tomato3d/fuse3d/surfels_io.py", f"kernel {KUSER}/tomato-dwt03-2dgs-full)", "kernel <kaggle_user>/tomato-dwt03-2dgs-full)"),
    ("tomato3d/recon/kernel_2dgs_r31.py", 'WORK = "/kaggle/working"; OUT =', 'WORK = os.environ.get("ONEZIP_WORK", "/kaggle/working"); OUT ='),
    ("tomato3d/recon/kernel_2dgs_r31.py", 'for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):', 'for z in []:  # one-zip: inputs are staged by run_recon.py, no Kaggle dataset lookup'),
    ("tomato3d/recon/kernel_2dgs_r31.py", 'R = ["/kaggle/input", "/tmp/in"]', 'R = [os.environ.get("ONEZIP_STAGE", "/tmp/in")]'),
    ("tomato3d/recon/kernel_2dgs_r31.py", '{os.path.splitext(v)[0][9:18]}.jpg", quality=85)', '{os.path.splitext(v)[0]}.jpg", quality=85)'),
    ("tomato3d/organs/pc_rachis.py", "from tomato3d.organs import light_organs as LO\n", "from pathlib import Path\n"),
    ("tomato3d/organs/pc_rachis.py", 'D0 = LO.ROOT / "build/p1_pc_absorb_20261027_01"; OUT = D0 / "R8_pc_rachis"', 'D0 = Path("."); OUT = D0 / "R8_pc_rachis"      # both set by the caller (tprox_stage.run_frozen), as skeleton_r8pc.step_rachis does'),
]
SCAN = {"windows_user_path": rb"[A-Za-z]:[\\/]+Users[\\/]", "unix_home_path": rb"/(?:home|Users)/[A-Za-z0-9_.-]+/", "cloud_sync_folder": rb"(?i)one" + rb"drive", "kaggle_account": KUSER.encode(),
        "email": rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "kaggle_json": rb"(?i)kaggle\.json|\"username\"\s*:|KGAT_[A-Za-z0-9]{16,}", "token_hf": rb"hf_[A-Za-z0-9]{30,}",
        "token_github": rb"gh[pousr]_[A-Za-z0-9]{30,}|github_" + rb"pat_[A-Za-z0-9_]{30,}", "private_key": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", "korean_project_path": ("작업물" + "저장소|홈" + "팜").encode()}
EMAIL_OK = (b"noreply@anthropic.com",)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()                          # noqa: E731


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--out", default=str(HERE / "dist")); ap.add_argument("--work", default=None); a = ap.parse_args(argv)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="onezip_build_")); stage = work / "zip_root"; shutil.rmtree(stage, ignore_errors=True); pkg = stage / "microtom_recon"
    for rel in COPY:
        dst = pkg / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(SRC / rel, dst)
    for rel, old, new in EDITS:
        p = pkg / rel; s = p.read_bytes().decode("utf-8"); crlf = "\r\n" in s
        o, n = (old.replace("\n", "\r\n"), new.replace("\n", "\r\n")) if crlf else (old, new)
        if s.count(o) != 1:
            raise SystemExit(f"edit does not apply exactly once in {rel}: {old[:70]!r} ({s.count(o)} found)")
        p.write_bytes(s.replace(o, n).encode("utf-8"))
    for f in ENTRY:
        shutil.copyfile(HERE / f, stage / f)
    (stage / "board").mkdir(); shutil.copyfile(HERE / "board/charuco_a3_10x7_38mm.pdf", stage / "board/charuco_a3_10x7_38mm.pdf")
    files = sorted(p for p in stage.rglob("*") if p.is_file()); bad = []; hits = []
    for p in files:
        if p.suffix == ".py":
            try:
                compile(p.read_text(encoding="utf-8"), str(p.relative_to(stage)), "exec")
            except SyntaxError as ex:
                bad.append((str(p.relative_to(stage)), str(ex)))
        b = p.read_bytes()
        for name, rx in SCAN.items():
            for m in re.finditer(rx, b):
                if name == "email" and m.group(0) in EMAIL_OK:
                    continue
                hits.append((str(p.relative_to(stage)).replace("\\", "/"), name, b[:m.start()].count(b"\n") + 1))
    if bad or hits:
        raise SystemExit(f"not built: compile errors {bad}; personal-data hits {hits}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); zp = out / f"microtom_recon_onezip_v{VERSION}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            zi = zipfile.ZipInfo(str(p.relative_to(stage)).replace("\\", "/"), date_time=(2026, 11, 16, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16; zf.writestr(zi, p.read_bytes())
    print(json.dumps(dict(zip=str(zp), sha256=sha(zp), files=len(files), py_compile_failed=0, personal_data_hits=0), indent=1))


if __name__ == "__main__":
    main()
