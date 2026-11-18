#!/usr/bin/env python3
"""microtom_recon one-zip v0.2 — photos of one potted plant on the A3 ChArUco board -> 2DGS surfel reconstruction (+ optional stage 7: T-Prox leaf labels).

usage: python run_recon.py --photos <folder> --out <folder> [--plant-id NAME] [--mask auto|auto+stem|sam2|color]
                           [--weights DIR --pot-top-mm MM --pot-side-mm MM]   (the three are required by --mask auto+stem)
                           [--iters N] [--holdout 2] [--skip-train] [--skeleton skeleton_v1.json --tprox-inputs DIR]

Six stages, each skipped when its output already exists, each writing a line to <out>/run_log.txt:
  1 photos   HEIC/JPG -> JPG, EXIF orientation, exact duplicates dropped        (microtom_recon/tomato3d/recon/charuco_photos.py)
  2 poses    board detection, intrinsics, camera poses, pose checks             (charuco_a3.py)
  3 masks    whole-plant masks, soft matte, 2 px band, visual hull              (charuco_masks.py, wholeplant_masks.py, r31_masks.py;
                                                                                 driver adapted from kernel_all_masks.py)
  4 points   initial points, pycolmap SIFT with the board poses fixed           (wholeplant_sparse.py; driver from kernel_all_points.py)
  5 train    2DGS, gsplat 1.5.3, R3-1b configuration                            (kernel_2dgs_r31.py; staging from kernel_all_train.py)
  6 outputs  surfels.ply, points.ply, recon_meta.json, REPORT_KO.md, preview.png (surfels_io.py, r31_score.py; r31_outputs.py logic)
  7 tprox    optional: compound-leaf labels, frozen pc_absorb.py (set A) + pc_rachis.py; needs --skeleton and --tprox-inputs (tprox_stage.py)

Nothing here is validated: lengths come from the printed board and were not checked against an independent measurement,
and the masks are automatic. The stage drivers in this file are adaptations of our Kaggle kernels (see STAGE_SOURCES.md).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "microtom_recon"))

VERSION = "0.2.2"
SEED_HOLDOUT = 20261023                      # as kernel_all_masks.py
TRAIN = dict(res=1512, iters=7000, n_init=150000, seed=0)   # R3-1b configuration (kernel_all_train.py)
OPAQUE = 0.5                                  # surfels.ply / points.ply keep opacity >= this (the rule behind R3-1b's 46,719)
MIN_ACCEPTED, MIN_AZ_COVER_DEG, MAX_NOT_FOUND = 20, 270.0, 0.30
PINS = {"pillow_heif": "pillow-heif==1.8.1", "pycolmap": "pycolmap==4.2.1", "sam2": "sam2==1.1.0",
        "hydra": "hydra-core==1.3.8", "iopath": "iopath==0.1.10", "gsplat": "gsplat==1.5.3 (installed by the trainer)"}
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
IMG_EXT = (".heic", ".heif", ".jpg", ".jpeg", ".png")
MASK_LABEL = {"sam2": "mask_method_new_ungraded", "color": "fallback_low_quality"}
SCALE_LINE = "길이(mm)는 보드 기준이며 독립 실측으로 검증되지 않았습니다."


class Stop(Exception):
    """A declared early stop; the message is Korean and says what to do."""


# ------------------------------------------------------------------------------------------------ small helpers
def detect_env() -> str:
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    if os.path.isdir("/content") and ("COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or os.path.isdir("/content/sample_data")):
        return "colab"
    return "linux"


class Log:
    def __init__(self, out: Path):
        self.path = out / "run_log.txt"; self.times_path = out / "stage_times.json"
        self.times = json.load(open(self.times_path)) if self.times_path.exists() else {}

    def __call__(self, *a):
        s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

    def stage(self, n: int, name: str, done, fn):
        if done():
            self(f"stage {n} {name}: skip (output exists)"); return
        t = time.time(); self(f"stage {n} {name}: start")
        fn()
        self.times[name] = round(time.time() - t, 1); json.dump(self.times, open(self.times_path, "w"), indent=1)
        self(f"stage {n} {name}: done {self.times[name]:.0f} s")


def pip_install(args: list[str], log, env=None):
    log("$ pip install", " ".join(args))
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True, env=env)
    if r.returncode:
        log(r.stdout[-1500:], r.stderr[-1500:]); raise Stop(f"패키지 설치 실패: {' '.join(args)}\n노트북의 인터넷(Internet)이 켜져 있는지 확인하고 다시 실행하세요.")


def ensure(module: str, pin: str, log, extra: tuple[str, ...] = ()):
    """Import `module`; if it is missing, pip-install the pinned version. An already-installed version is used as it is."""
    import importlib
    try:
        importlib.import_module(module)
    except ImportError:
        pip_install([*extra, pin], log); importlib.invalidate_caches(); importlib.import_module(module)


def find_photo_dir(root: Path, work: Path, log) -> Path:
    """The folder under `root` that directly holds the most photos; code folders (with run_recon.py) are ignored.
    If none is found, zip files under `root` (not the code zip) are unpacked into the work folder and searched."""
    def best(base: Path):
        cnt = {}
        for dp, dn, fn in os.walk(base):
            if "run_recon.py" in fn:
                dn[:] = []; continue
            n = sum(f.lower().endswith(IMG_EXT) for f in fn)
            if n:
                cnt[dp] = n
        return max(cnt.items(), key=lambda kv: kv[1]) if cnt else None
    b = best(root)
    if b is None:
        for z in sorted(root.rglob("*.zip")):
            try:
                with zipfile.ZipFile(z) as zf:
                    if any(n.endswith("run_recon.py") for n in zf.namelist()):
                        continue
                    zf.extractall(work / "photos_unzipped" / z.stem)
            except zipfile.BadZipFile:
                continue
        if (work / "photos_unzipped").exists():
            b = best(work / "photos_unzipped")
    if b is None:
        raise Stop(f"사진을 찾지 못했습니다: {root}\nHEIC 또는 JPG 사진이 들어 있는 폴더(또는 그 폴더의 zip)를 올린 뒤 --photos 로 지정하세요.")
    log(f"photos: {b[1]} files in {b[0]}")
    return Path(b[0])


def write_ply(path: Path, hdr: str, rec):
    Path(path).write_bytes(hdr.encode() + rec.tobytes())


# ------------------------------------------------------------------------------------------------ stage 1
def stage_photos(a, out: Path, work: Path, log):
    ensure("pillow_heif", PINS["pillow_heif"], log)
    from tomato3d.recon import charuco_photos
    src = find_photo_dir(Path(a.photos), work, log)
    charuco_photos.main(["--src", str(src), "--out", str(out)])


def check_photos(out: Path):
    rows = [r for r in csv.DictReader(open(out / "photos.csv")) if r["kept"] == "1"]
    if not rows:
        raise Stop("사진이 한 장도 없습니다. HEIC 또는 JPG 사진 폴더를 확인하세요.")
    sizes = {(int(r["width"]), int(r["height"])) for r in rows}
    if len(sizes) > 1:
        raise Stop(f"사진 크기가 서로 다릅니다: {sorted(sizes)}\n한 개체는 같은 카메라·같은 설정(같은 해상도)으로 다시 찍으세요.")
    w, h = next(iter(sizes))
    if w != h:
        raise Stop(f"사진이 정사각형이 아닙니다 ({w}x{h}).\nv0.1 은 1:1(정사각형) 사진만 처리합니다. 카메라 비율을 1:1 로 바꿔 다시 찍으세요.")
    return rows, w


# ------------------------------------------------------------------------------------------------ stage 2
def stage_poses(out: Path, log):
    import cv2
    if not (hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoDetector")):
        pip_install(["opencv-contrib-python-headless>=4.8,<5"], log)
        raise Stop("OpenCV 에 ChArUco 검출기가 없어 새로 설치했습니다. 런타임을 다시 시작한 뒤 같은 명령을 다시 실행하세요.")
    from tomato3d.recon import charuco_a3 as ca
    pd = out / "poses"
    try:
        ca.main(["--jpg", str(out / "jpg"), "--out", str(pd), "--res", "3024", "--workers", str(min(6, os.cpu_count() or 2))])
    except Stop:
        raise
    except Exception as ex:                                                              # noqa: BLE001
        n_ok = n_all = 0
        if (pd / "detections.json").exists():
            d = json.load(open(pd / "detections.json")); n_all = len(d); n_ok = sum(x["corners"] >= ca.MIN_CORNERS for x in d)
        raise Stop(f"보드 검출·보정 단계가 실패했습니다 (보드가 잡힌 사진 {n_ok}/{n_all}장; {type(ex).__name__}: {str(ex)[:200]}).\n"
                   "인쇄한 A3 ChArUco 보드(10x7, 38 mm)가 사진마다 절반 이상 보이고 초점이 맞도록 다시 찍으세요.") from ex


def check_poses(out: Path, log) -> dict:
    """Writes pose_qc.json and stops (Korean message) when the capture cannot be reconstructed."""
    import numpy as np
    from tomato3d.recon import charuco_a3 as ca
    pd = out / "poses"; rows = list(csv.DictReader(open(pd / "qc.csv"))); intr = json.load(open(pd / "intrinsics.json"))
    acc = [r for r in rows if r["accepted"] == "1"]; n = len(rows)
    not_found = sum(int(r["points"]) < ca.MIN_POINTS for r in rows)
    az = sorted(float(r["azimuth_deg"]) for r in acc)
    gap = max([(az[(i + 1) % len(az)] - az[i]) % 360 for i in range(len(az))]) if len(az) > 1 else 360.0
    cover = 360.0 - gap; rms = [float(r["rms_px"]) for r in acc]; full = intr["size"][0]; s = TRAIN["res"] / full
    el = [float(r["elevation_deg"]) for r in acc]
    fails = []
    if not_found > MAX_NOT_FOUND * n:
        fails.append(f"보드를 찾지 못한 사진이 {not_found}/{n}장({100 * not_found / n:.0f} %)으로 30 % 를 넘습니다. "
                     "→ 보드가 잘리거나 잎에 많이 가려지지 않게, 반사 없이 밝은 곳에서 다시 찍으세요.")
    if len(acc) < MIN_ACCEPTED:
        fails.append(f"쓸 수 있는 사진이 {len(acc)}장뿐입니다(최소 {MIN_ACCEPTED}장; 전체 {n}장 중 보드 검출 {n - not_found}장). "
                     "→ 화분 둘레를 돌며 40장 이상, 보드가 절반 이상 보이게 다시 찍으세요.")
    if cover < MIN_AZ_COVER_DEG:
        fails.append(f"촬영 방향이 둘레의 {cover:.0f}° 만 덮습니다(최소 {MIN_AZ_COVER_DEG:.0f}°; 가장 큰 빈 구간 {gap:.0f}°). "
                     "→ 빠진 방향이 없도록 화분을 한 바퀴 다 돌며 다시 찍으세요.")
    qc = dict(photos=n, board_not_found=not_found, accepted=len(acc), rejected_reasons=sorted({r["reason"] for r in rows if r["accepted"] != "1" and r["reason"]}),
              calibration_rms_px_full=intr["rms_px"], calibration_views=len(intr["views_used"]), full_size_px=full,
              view_rms_px_full=dict(min=min(rms), median=float(np.median(rms)), max=max(rms)) if rms else None,
              view_rms_px_at_1512=dict(min=min(rms) * s, median=float(np.median(rms)) * s, max=max(rms) * s) if rms else None,
              azimuth_coverage_deg=cover, largest_azimuth_gap_deg=gap, elevation_deg=dict(min=min(el), max=max(el)) if el else None,
              thresholds=dict(min_accepted=MIN_ACCEPTED, min_azimuth_coverage_deg=MIN_AZ_COVER_DEG, max_board_not_found_share=MAX_NOT_FOUND),
              passed=not fails, failures=fails)
    json.dump(qc, open(out / "pose_qc.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"pose check: photos {n}, board not found {not_found}, accepted {len(acc)}, azimuth coverage {cover:.0f} deg, "
        f"calibration RMS {intr['rms_px']:.2f} px @ {full}")
    if fails:
        raise Stop("촬영 점검에서 멈췄습니다 (pose_qc.json):\n- " + "\n- ".join(fails))
    return qc


# ------------------------------------------------------------------------------------------------ stage 3
def load_sam(work: Path, dev: str, log):
    os.environ.setdefault("SAM2_BUILD_CUDA", "0")
    ensure("hydra", PINS["hydra"], log); ensure("iopath", PINS["iopath"], log)
    ensure("sam2", PINS["sam2"], log, extra=("--no-deps", "--no-build-isolation"))
    ck = work / "sam2.1_hiera_tiny.pt"
    if not ck.exists() or hashlib.sha256(ck.read_bytes()).hexdigest() != SAM_SHA256:
        import urllib.request
        log("download", SAM_URL)
        try:
            urllib.request.urlretrieve(SAM_URL, ck)
        except Exception as ex:                                                          # noqa: BLE001
            raise Stop(f"SAM 2.1 가중치를 내려받지 못했습니다 ({type(ex).__name__}). 노트북의 인터넷(Internet)을 켜고 다시 실행하세요.\n"
                       "인터넷을 쓸 수 없으면 --mask color 로 실행할 수 있습니다(품질 낮음, fallback_low_quality).") from ex
        if hashlib.sha256(ck.read_bytes()).hexdigest() != SAM_SHA256:
            raise Stop("내려받은 SAM 2.1 가중치의 해시가 기대값과 다릅니다. 다시 실행하거나 --mask color 를 쓰세요.")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(SAM_CFG, str(ck), device=dev, apply_postprocessing=False), max_hole_area=0, max_sprinkle_area=0)


def stage_masks(a, out: Path, work: Path, log):
    """Driver adapted from kernel_all_masks.py (hold-out rule, second hull, matte, band) with the C3 U-Net replaced by
    colour-in-hull (charuco_masks.main logic) and, for --mask auto/sam2, the A2 SAM step (wholeplant_masks.main, pass 1)."""
    import cv2, numpy as np
    from PIL import Image
    from tomato3d.recon import charuco_a3 as ca
    from tomato3d.recon.charuco_masks import COARSE_M, FINAL_DILATE_PX, FINE_M, HULL_DILATE_PX, MASK_RES, ZMAX_M, carve, colour_mask, project
    from tomato3d.recon.r31_masks import matte
    from tomato3d.recon.wholeplant_contract import ring
    R = MASK_RES; mode = "sam2" if a.mask in ("auto", "sam2") else a.mask
    po = json.load(open(out / "poses" / "poses.json")); acc = po["accepted"]; R0 = po["undistorted_res"]; ud = out / "poses" / f"undist_{R0}"
    K = ca.scaled_K(np.array(po["K_undist"]), R0, R); Wb, Hb = ca.SQUARES[0] * ca.SQUARE_M, ca.SQUARES[1] * ca.SQUARE_M
    ctr_b = np.array([Wb / 2, Hb / 2, 0.0])

    def az_el(c):
        d = np.array(c) - ctr_b; return float(np.degrees(np.arctan2(d[1], d[0])) % 360), float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))

    def pick_heldout(views, n, seed):
        """kernel_all_masks.pick_heldout generalised from 2 to n azimuth sectors; ring preference low, upper, middle, top."""
        r_ = np.random.default_rng(seed); o = []
        for k in range(n):
            lo, hi = 360.0 * k / n, 360.0 * (k + 1) / n
            cand = [v for v in views if lo <= az_el(po["views"][v]["center_world"])[0] < hi and v not in o]
            for want in ("low", "upper", "middle", "top"):
                c = [v for v in cand if ring(az_el(po["views"][v]["center_world"])[1]) == want]
                if c:
                    o.append(sorted(c)[int(r_.integers(len(c)))]); break
        return o

    sp_path = out / "split.json"
    if sp_path.exists():
        held = json.load(open(sp_path))["heldout"]
    else:
        held = pick_heldout(acc, a.holdout, SEED_HOLDOUT)
    if not held:
        raise Stop("보류 시점을 고르지 못했습니다. 촬영 방향이 한쪽에 몰려 있지 않은지 확인하세요.")
    train = [v for v in acc if v not in held]
    json.dump(dict(heldout=held, train=train, seed=SEED_HOLDOUT, rule="one view per azimuth sector; ring preference low, upper, middle, top"), open(sp_path, "w"), indent=1)
    json.dump(dict(heldout_new=held, train=train), open(out / "contract.json", "w"), indent=1)          # read by wholeplant_sparse
    for d in ("masks", "bands", "soft"):
        (out / d).mkdir(exist_ok=True)
    cams = {v: (po["views"][v]["R_cw"], po["views"][v]["t_cw"], K) for v in acc}; cache, cols = {}, {}
    for v in acc:
        cache[v] = cv2.resize(np.asarray(Image.open(ud / v).convert("RGB")), (R, R), interpolation=cv2.INTER_AREA); cols[v] = colour_mask(cache[v])

    def hull_from(masks, pad):
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_DILATE_PX + 1,) * 2); dil = [cv2.dilate(masks[v].astype(np.uint8), kd) for v in train]
        tc = [cams[v] for v in train]; P = carve((0, 0, 0.0), (Wb, Hb, ZMAX_M), COARSE_M, tc, dil, R)
        if not len(P):
            raise Stop("식물(초록색 부분)을 찾지 못해 형상 껍질(visual hull)이 비었습니다.\n"
                       "식물이 보드 위(보드 안쪽)에 놓여 있고, 사진 8장 이상에서 식물 전체가 보이는지 확인해 다시 찍으세요.")
        lo = np.maximum(P.min(0) - pad, [0, 0, 0]); hi = np.minimum(P.max(0) + pad, [Wb, Hb, ZMAX_M]); return carve(lo, hi, FINE_M, tc, dil, R)

    def hull_proj(Pf, v, kf):
        Rm, t, Kk = cams[v]; u, w_, z = project(Pf, Rm, t, Kk); ok = (z > 0) & (u >= 0) & (w_ >= 0) & (u < R) & (w_ < R)
        hp = np.zeros((R, R), np.uint8); hp[w_[ok].astype(int), u[ok].astype(int)] = 1
        r = max(1, int(round(FINE_M * Kk[0, 0] / np.median(z[ok])))) if ok.any() else 1
        return cv2.dilate(cv2.dilate(hp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)), kf) > 0

    # --- a + b: colour pre-mask inside its own visual hull (charuco_masks.main)
    P0 = hull_from(cols, 3 * COARSE_M); kf0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FINAL_DILATE_PX + 1,) * 2)
    lab = {}
    for v in acc:
        m = np.zeros((R, R), np.uint8); m[cols[v] & hull_proj(P0, v, kf0)] = 1; lab[v] = m
    log(f"masks: colour-in-hull done, hull voxels {len(P0)}, views {len(acc)} (train {len(train)}, held-out {len(held)})")
    stats = {v: dict(split="heldout" if v in held else "train", colour_px=int(cols[v].sum()), colour_in_hull_px=int(lab[v].sum())) for v in acc}
    del cols
    # --- c: SAM additions (wholeplant_masks.main, pass 1) with the stock SAM 2.1 tiny weights
    if mode == "sam2":
        import torch
        from tomato3d.recon import wholeplant_masks as WM
        dev = "cuda" if torch.cuda.is_available() else "cpu"; sam = load_sam(work, dev, log)
        K3 = ca.scaled_K(np.array(po["K_undist"]), R0, WM.RES); ctr = (P0.min(0) + P0.max(0)) / 2; rng = np.random.default_rng(0)
        for i, v in enumerate(acc):
            rgb = np.asarray(Image.open(ud / v).convert("RGB").resize((WM.RES, WM.RES), Image.BILINEAR))
            plant = cv2.resize(lab[v], (WM.RES, WM.RES), interpolation=cv2.INTER_NEAREST) == 1
            if plant.sum() < 500:
                stats[v]["sam"] = "skipped: no plant mask"; continue
            pos = WM.seeds(rgb, plant, rng)
            ys, xs = np.nonzero(plant); pad = 0.06 * max(np.ptp(xs), np.ptp(ys))
            box = np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], np.float32)
            dt = cv2.distanceTransform(plant.astype(np.uint8), cv2.DIST_L2, 5); py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
            pos = np.vstack([[[px, py]], pos]) if len(pos) else np.array([[px, py]], np.float32)
            neg = WM.negatives(po["views"][v]["R_cw"], po["views"][v]["t_cw"], K3, ctr, WM.RES)
            with torch.inference_mode():
                sam.set_image(rgb); pts = np.vstack([pos, neg]) if len(neg) else pos
                lbl = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
                masks, ious, _ = sam.predict(box=box, point_coords=pts, point_labels=lbl, multimask_output=True)
            best = max(zip(masks, ious), key=lambda m: float(m[1]))
            lb = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            gate = ((lb[..., 1] - 128) <= WM.ADD_A_MAX) & (lb[..., 0] * 100 / 255 >= WM.ADD_L_MIN)
            add = WM.thin_parts((best[0] > 0) & ~plant & gate, plant)
            add = cv2.resize(add.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST) > 0
            lab[v][add & (lab[v] == 0)] = 2
            stats[v].update(sam_iou=float(best[1]), seeds=int(len(pos)), added_px_raw=int(add.sum()))
            if i % 10 == 0:
                log(f"masks: SAM {i + 1}/{len(acc)} {v} added {int(add.sum())} px")
        del sam
        if dev == "cuda":
            torch.cuda.empty_cache()
    # --- d: second hull from the training views' masks, intersect, matte + band (kernel_all_masks.py)
    raw = {v: lab[v] > 0 for v in acc}
    Pf = hull_from(raw, 0.012) if mode == "sam2" else P0; kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)) if mode == "sam2" else kf0
    for v in acc:
        plant = raw[v] & hull_proj(Pf, v, kf); alpha, band = matte(cache[v], plant)
        m = lab[v].copy(); m[~plant] = 0; st = Path(v).stem
        Image.fromarray(m).save(out / "masks" / f"{st}.png"); Image.fromarray(band.astype(np.uint8)).save(out / "bands" / f"{st}.png")
        Image.fromarray((alpha * 255).astype(np.uint8)).save(out / "soft" / f"{st}.png")
        stats[v].update(plant_px=int(plant.sum()), added_px=int((m == 2).sum()), hull_removed=round(1 - plant.sum() / max(raw[v].sum(), 1), 4))
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Pf)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    write_ply(out / "hull.ply", hdr, Pf.astype("<f4"))
    tiles = []
    for v in (held + train[::max(1, len(train) // 10)])[:12]:
        o = cache[v].astype(np.float32); m = np.asarray(Image.open(out / "masks" / f"{Path(v).stem}.png")) > 0
        o[m] = o[m] * 0.5 + np.array([255, 0, 190]) * 0.5; t = cv2.resize(o.astype(np.uint8), (300, 300))
        cv2.putText(t, f"{Path(v).stem[-12:]} {'H' if v in held else 'T'}", (4, 18), 0, 0.5, (255, 255, 0), 2); tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    Image.fromarray(np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)])).save(out / "masks_sheet.jpg", quality=85)
    method = ("colour-in-hull + SAM 2.1 hiera tiny (stock weights) thin-part additions + second hull + M4 matte" if mode == "sam2"
              else "colour-in-hull + M4 matte (colour threshold only)")
    json.dump(dict(_done=True, mask_mode=mode, mask_method=method, mask_label=MASK_LABEL[mode], sam_checkpoint_sha256=SAM_SHA256 if mode == "sam2" else None,
                   _hull=dict(carved_from=len(train), voxels=int(len(Pf)), lo=Pf.min(0).tolist(), hi=Pf.max(0).tolist()),
                   plant_px_median=float(np.median([s["plant_px"] for s in stats.values()])), views=stats),
              open(out / "masks_summary.json", "w"), indent=1)
    log(f"masks: {mode} [{MASK_LABEL[mode]}], hull voxels {len(Pf)}, plant px median {np.median([s['plant_px'] for s in stats.values()]):.0f}")


# ------------------------------------------------------------------------------------------------ stage 4
def stage_points(out: Path, work: Path, log):
    """Driver from kernel_all_points.py: pycolmap SIFT 3024 px, 32,768 features, board poses fixed, foreground = the masks."""
    ensure("pycolmap", PINS["pycolmap"], log)
    from tomato3d.recon import wholeplant_sparse as ws
    cw = work / "colmap"; shutil.rmtree(cw, ignore_errors=True)
    try:
        ws.main(["--poses", str(out / "poses"), "--out", str(out), "--work", str(cw)])
    except AssertionError as ex:
        raise Stop(f"초기 점(특징점 삼각측량)을 만들지 못했습니다: {ex}\n잎 표면이 흔들리거나 흐리지 않게, 이웃한 사진끼리 많이 겹치도록 다시 찍으세요.") from ex
    shutil.copy(out / "plant_points_dense.ply", out / "plant_points.ply")
    ck = json.load(open(out / "sparse_dense_checks.json")); log(f"points: {ck['plant_points']} plant points, reprojection RMS {ck['plant_reproj_rms_px_at_res']:.2f} px @3024")
    shutil.rmtree(cw, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ stage 5
def stage_train(a, out: Path, work: Path, run: str, log):
    """Staging from kernel_all_train.py, then the trainer copy (kernel_2dgs_r31.py) as a subprocess. R3-1b configuration:
    1512 px, 7000 iterations, 150,000 initial points, seed 0, soft-mask alpha loss, 2 px band, exposure compensation on."""
    import cv2, numpy as np
    from PIL import Image
    from tomato3d.recon import charuco_a3 as ca
    from tomato3d.recon.charuco_prep import HULL_COLOUR, HULL_INIT_MAX, read_xyz, write_xyzrgb
    RES = TRAIN["res"]; tag = re.sub(r"[^A-Za-z0-9]", "_", a.plant_id)
    po = json.load(open(out / "poses" / "poses.json")); sp = json.load(open(out / "split.json")); acc = po["accepted"]; held, train = sp["heldout"], sp["train"]
    R0 = po["undistorted_res"]; ST = work / "stage"; shutil.rmtree(ST, ignore_errors=True)
    for d in ("images_1512", f"masks_{tag}", f"bands_{tag}", f"soft_{tag}", "init"):
        (ST / d).mkdir(parents=True)
    for v in acc:
        im = np.asarray(Image.open(out / "poses" / f"undist_{R0}" / v).convert("RGB"))
        Image.fromarray(cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)).save(ST / "images_1512" / v, quality=95)
        for s_, d_ in (("masks", f"masks_{tag}"), ("bands", f"bands_{tag}"), ("soft", f"soft_{tag}")):
            shutil.copy(out / s_ / f"{Path(v).stem}.png", ST / d_ / f"{Path(v).stem}.png")
    P1, C1 = read_xyz(out / "plant_points.ply"); write_xyzrgb(ST / "init" / f"points_{tag}.ply", P1, C1)
    Ph, _ = read_xyz(out / "hull.ply"); rng = np.random.default_rng(0); Ph = Ph[rng.choice(len(Ph), min(HULL_INIT_MAX, len(Ph)), replace=False)]
    write_xyzrgb(ST / "init" / f"hull_{tag}.ply", Ph, np.tile(np.array(HULL_COLOUR, np.float32), (len(Ph), 1)))
    H = float(Ph[:, 2].max() - max(Ph[:, 2].min(), 0.0)); Ks = ca.scaled_K(np.array(po["K_undist"]), R0, RES)
    cams = {str(RES): {v: dict(K=Ks.tolist(), R_cw=po["views"][v]["R_cw"], t_cw=po["views"][v]["t_cw"]) for v in acc}}
    cfg = dict(note=f"{run}: R3-1b configuration, exposure ON, automatic masks ({a.mask}), no I2", res=RES, masks=f"masks_{tag}", fg_labels=[1, 2],
               band=f"bands_{tag}", soft=f"soft_{tag}", init=f"init/points_{tag}.ply", init_hull=f"init/hull_{tag}.ply", exposure=True)
    json.dump(dict(schema="wholeplant-input-1", iters=a.iters, n_init=TRAIN["n_init"], seed=TRAIN["seed"], scene_scale_m=H, train=train, heldout_new=held,
                   heldout_run01=[], runs={run: cfg}, cams=cams, organ_split="none", r31=True), open(ST / "index.json", "w"), indent=1)
    log(f"train: staged train {len(train)} held-out {len(held)} points {len(P1)} hull {len(Ph)} H {H:.4f} m iters {a.iters}")
    tw = out / "train"; tw.mkdir(exist_ok=True)
    env = dict(os.environ, ONEZIP_WORK=str(tw), ONEZIP_STAGE=str(ST), PYTHONUNBUFFERED="1")
    r = subprocess.run([sys.executable, str(HERE / "microtom_recon" / "tomato3d" / "recon" / "kernel_2dgs_r31.py")], env=env)
    sf = tw / "out" / f"{run}_surfels_full.ply"
    if r.returncode or not sf.exists():
        raise Stop(f"2DGS 학습이 실패했습니다 (종료 코드 {r.returncode}). 자세한 내용은 {tw / 'run_log.txt'} 를 보세요.\n"
                   "GPU 메모리 부족이면 다른 노트북·세션을 닫고 같은 명령을 다시 실행하세요(끝난 단계는 건너뜁니다).")
    shutil.copy(sf, out / "surfels_full.ply"); shutil.rmtree(ST, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ stage 6
def preview(xyz, rgb, path: Path):
    """4 side views (azimuth 0/90/180/270 deg, orthographic) + 1 top view of the surfel centres, far-to-near painting."""
    import cv2, numpy as np
    S, M = 520, 30; c = (xyz.min(0) + xyz.max(0)) / 2; ext = float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2]), 1e-3)); px = (S - 2 * M) / ext
    panels = []
    for name, azd in (("az 0", 0), ("az 90", 90), ("az 180", 180), ("az 270", 270), ("top", None)):
        p = xyz - c
        if azd is None:
            u, w_, depth = p[:, 0], p[:, 1], -p[:, 2]
        else:
            t = np.radians(azd); d = np.array([np.cos(t), np.sin(t)]); r = np.array([-np.sin(t), np.cos(t)])
            u, w_, depth = p[:, :2] @ r, p[:, 2], -(p[:, :2] @ d)
        o = np.argsort(-depth); x = np.clip((S / 2 + u[o] * px).astype(int), 1, S - 2); y = np.clip((S / 2 - w_[o] * px).astype(int), 1, S - 2)
        img = np.full((S, S, 3), 255, np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                img[y + dy, x + dx] = rgb[o]
        cv2.putText(img, name, (10, 24), 0, 0.7, (0, 0, 0), 2); bar = int(round(0.02 * px))
        cv2.line(img, (10, S - 14), (10 + bar, S - 14), (0, 0, 0), 3); cv2.putText(img, "20 mm (board scale, unvalidated)", (16 + bar, S - 9), 0, 0.45, (0, 0, 0), 1)
        panels.append(img)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))


def stage_outputs(a, out: Path, run: str, env: str, log):
    import numpy as np
    from PIL import Image
    from tomato3d.fuse3d import surfels_io as sio
    from tomato3d.recon.r31_score import score_view
    full = out / "surfels_full.ply"; b = full.read_bytes(); i = b.index(b"end_header\n") + 11; raw = sio.read_binary_ply(full); sf = sio.read_surfels_full(full)
    keep = sf["opacity"] >= OPAQUE; n_all, n_op = int(len(keep)), int(keep.sum())
    lines = re.sub(r"element vertex \d+", f"element vertex {n_op}", b[:i].decode("ascii")).split(chr(10))
    lines.insert(2, f"comment opacity >= {OPAQUE} subset of surfels_full.ply; unvalidated, conditional_on_unvalidated_scale")
    hdr = chr(10).join(lines)
    write_ply(out / "surfels.ply", hdr, raw[keep])
    xyz, nrm, rgb = sf["xyz"][keep], sf["normal"][keep], sf["rgb"][keep].astype(np.uint8)
    rec = np.zeros(n_op, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for kk, c in zip("xyz", xyz.T): rec[kk] = c
    for kk, c in zip(("nx", "ny", "nz"), nrm.T): rec[kk] = c
    for kk, c in zip("rgb", rgb.T): rec[kk] = c
    ph = (f"ply\nformat binary_little_endian 1.0\ncomment board metres, z up; whole plant, no organ split; {run}; opacity >= {OPAQUE}; unvalidated, "
          f"conditional_on_unvalidated_scale\nelement vertex {n_op}\n" + "".join(f"property float {x}\n" for x in ("x", "y", "z", "nx", "ny", "nz")) +
          "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    write_ply(out / "points.ply", ph, rec)
    height_mm = float((xyz[:, 2].max() - xyz[:, 2].min()) * 1000); top_mm = float(xyz[:, 2].max() * 1000); bottom_mm = float(xyz[:, 2].min() * 1000)
    preview(xyz, rgb, out / "preview.png")
    # held-out views against the automatic masks (r31_score.score_view; band-excluded IoU)
    sp = json.load(open(out / "split.json")); z = np.load(out / "train" / "out" / f"{run}_views.npz"); pv = {}
    for v in sp["heldout"]:
        lab = np.asarray(Image.open(out / "masks" / f"{Path(v).stem}.png")); ref = np.isin(lab, (1, 2))
        pv[v] = score_view(ref, np.zeros_like(ref), z[f"a_{v}"].astype(np.float32) >= 0.5, lab == 2)
    med = {k: (float(np.median([m[k] for m in pv.values()])) if pv else None) for k in ("iou_no_band", "iou", "edge_f2")}
    res = json.load(open(out / "train" / "out" / "results.json")); rr = res["runs"][run]; ms = json.load(open(out / "masks_summary.json"))
    qc = json.load(open(out / "pose_qc.json", encoding="utf-8")); photos = list(csv.DictReader(open(out / "photos.csv")))
    ck = json.load(open(out / "sparse_dense_checks.json")); times = json.load(open(out / "stage_times.json")) if (out / "stage_times.json").exists() else {}
    vers = {}
    for m in ("torch", "gsplat", "pycolmap", "pillow_heif", "cv2", "numpy", "sam2", "skimage", "scipy"):
        try:
            import importlib.metadata as im
            vers[m] = im.version({"cv2": "opencv-python", "pillow_heif": "pillow-heif", "skimage": "scikit-image"}.get(m, m))
        except Exception:                                                                # noqa: BLE001
            vers[m] = None
    vers["gsplat"] = res["prov"].get("gsplat"); vers["torch"] = res["prov"].get("torch")
    n_file = len(photos); n_dup = sum(r["kept"] != "1" for r in photos); n_used = qc["accepted"]
    meta = dict(package=f"microtom_recon_onezip v{VERSION}", plant_id=a.plant_id, run=run, environment=env, pipeline="R3-1b-style (automatic masks)",
                validation="none: scale unvalidated, masks automatic", scale_status="conditional_on_unvalidated_scale (board gives metres; no independent measurement)",
                organ_split="none", photos=dict(files=n_file, exact_duplicates_dropped=n_dup, used=n_used, dropped=n_file - n_used,
                                                 train=len(sp["train"]), heldout=sp["heldout"]),
                pose_qc=qc, mask=dict(mode=ms["mask_mode"], method=ms["mask_method"], label=ms["mask_label"], plant_px_median=ms["plant_px_median"]),
                init_points=dict(plant_points=ck["plant_points"], reproj_rms_px_at_3024=ck["plant_reproj_rms_px_at_res"]),
                train=dict(params=res["prov"]["params"], trainer_sha256=res["prov"]["trainer_sha256"], exposure=True, res=TRAIN["res"],
                           iters_standard=a.iters == TRAIN["iters"], train_seconds=rr["train_seconds"], train_medians={k: v for k, v in rr["train"].items() if k != "per_view"}),
                surfels=dict(all=n_all, opaque=n_op, opacity_min=OPAQUE),
                height=dict(height_mm=height_mm, definition="z max - z min of the opaque surfels (no pot-top plane is fitted)", top_above_board_mm=top_mm, bottom_above_board_mm=bottom_mm),
                heldout=dict(label="auto_mask_reference_not_human", n=len(pv), medians=med, per_view=pv), versions=vers, pins=PINS, stage_seconds=times,
                note="shape only; no trait tables, no leaf separation, no mm accuracy claim")
    json.dump(meta, open(out / "recon_meta.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    f3 = lambda x: "—" if x is None else f"{x:.3f}"                                       # noqa: E731
    L = [f"# 복원 보고 — {a.plant_id}", "",
         f"microtom_recon_onezip v{VERSION} · 실행 환경 {env} · 검증되지 않은 결과입니다(자동 마스크, 규모 미검증).", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 사진 | 파일 {n_file}장, 사용 {n_used}장, 제외 {n_file - n_used}장 (완전 중복 {n_dup}장, 보드 미검출 {qc['board_not_found']}장, 나머지는 자세 점검 탈락) |",
         f"| 학습 / 보류 시점 | {len(sp['train'])} / {len(sp['heldout'])} |",
         f"| 자세 재투영 오차 | 보정 RMS {qc['calibration_rms_px_full']:.2f} px (원본 {qc['full_size_px']} px), 시점별 중앙값 {qc['view_rms_px_full']['median']:.2f} px "
         f"(1512 px 기준 {qc['view_rms_px_at_1512']['median']:.2f} px) |",
         f"| 방위 범위 | {qc['azimuth_coverage_deg']:.0f}° (가장 큰 빈 구간 {qc['largest_azimuth_gap_deg']:.0f}°) |",
         f"| 마스크 방법 | {ms['mask_mode']} — `{ms['mask_label']}` |",
         f"| 초기 점 | {ck['plant_points']:,}개 |",
         f"| 서펠 수 | 불투명(opacity ≥ {OPAQUE}) {n_op:,}개 / 전체 {n_all:,}개 |",
         f"| 개체 높이 | {height_mm:.1f} mm (불투명 서펠의 z 최대 − z 최소; 보드 면에서 {bottom_mm:.1f}–{top_mm:.1f} mm) |",
         f"| 보류 시점 IoU (띠 제외) | 중앙값 {f3(med['iou_no_band'])} (n={len(pv)}) — `auto_mask_reference_not_human` |",
         f"| 학습 반복 | {a.iters}" + ("" if a.iters == TRAIN["iters"] else f" (표준 {TRAIN['iters']} 과 다름)") + " |", "",
         SCALE_LINE, "",
         "보류 시점 IoU 는 사람이 그린 마스크가 아니라 같은 자동 마스크와 비교한 값입니다. 마스크가 틀린 곳은 이 숫자에 드러나지 않습니다.", "",
         "## 단계별 소요 시간", "", "| 단계 | 초 |", "|---|---|"]
    L += [f"| {k} | {v:.0f} |" for k, v in times.items()] + [f"| outputs | (이 보고서 작성 중) |", "",
         "## 파일", "", "- `surfels.ply` 불투명 서펠(회전·크기 포함), `surfels_full.ply` 학습기 원본 전체", "- `points.ply` 서펠 중심 + 법선 + 색",
         "- `preview.png` 옆 4방향 + 위 1장(점 그림, 표시용)", "- `recon_meta.json` 모든 수치와 버전, `pose_qc.json`, `masks_sheet.jpg`(마스크 확인용)", ""]
    (out / "REPORT_KO.md").write_text("\n".join(L), encoding="utf-8")
    log(f"outputs: surfels {n_op} opaque / {n_all}, height {height_mm:.1f} mm, held-out band-excluded IoU median {f3(med['iou_no_band'])} [auto_mask_reference_not_human]")


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    env = detect_env(); dflt = {"kaggle": "/kaggle/working/recon_out", "colab": "/content/recon_out"}.get(env, "recon_out")
    ap = argparse.ArgumentParser(description="photos on the A3 ChArUco board -> 2DGS surfel reconstruction (v0.2.2, unvalidated)")
    ap.add_argument("--photos", default=None); ap.add_argument("--out", default=dflt); ap.add_argument("--plant-id", default="plant")
    ap.add_argument("--mask", choices=("auto", "auto+stem", "unet", "sam2", "color"), default="auto"); ap.add_argument("--iters", type=int, default=TRAIN["iters"])
    ap.add_argument("--holdout", type=int, default=2); ap.add_argument("--skip-train", action="store_true"); ap.add_argument("--skeleton", default=None); ap.add_argument("--tprox-inputs", default=None)
    ap.add_argument("--weights", default=None); ap.add_argument("--pot-top-mm", type=float, default=None); ap.add_argument("--pot-side-mm", type=float, default=None)
    a = ap.parse_args(argv); a.plant_id = re.sub(r"[^A-Za-z0-9_-]", "_", a.plant_id) or "plant"
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True); log = Log(out); run = f"R3-1b-style_{a.plant_id}"
    work = {"kaggle": Path("/kaggle/tmp"), "colab": Path("/content")}.get(env, out) / f"_onezip_work_{a.plant_id}"
    if not str(work.resolve()).isascii():
        import tempfile
        work = Path(tempfile.gettempdir()) / f"_onezip_work_{a.plant_id}"
    work.mkdir(parents=True, exist_ok=True)
    log(f"microtom_recon_onezip v{VERSION} env={env} out={out} plant={a.plant_id} mask={a.mask} iters={a.iters} holdout={a.holdout} skip_train={a.skip_train}")
    try:
        import tprox_stage
        if tprox_stage.attached_recon(a, out, log):
            tprox_stage.run(a, out, log); return 0
        if not a.photos:
            raise Stop("--photos <사진 폴더> 가 필요합니다. (7단계만 실행하려면 복원이 들어 있는 --tprox-inputs 를 주세요.)")
        import stem_masks
        a.stem = a.mask == "auto+stem"
        if a.stem:                                   # weights and pot values are checked before any stage runs; no fallback to auto
            stem_masks.check_args(a, Stop); a.mask = "auto"
        if a.mask == "unet":
            raise Stop("--mask unet: v0.1 에는 U-Net 가중치가 들어 있지 않습니다(가중치 파일이 만들어진 적이 없음). --mask auto 또는 --mask color 를 쓰세요.")
        if a.holdout < 1 or a.holdout > 8:
            raise Stop("--holdout 은 1–8 사이여야 합니다(학습기가 보류 시점 점수를 계산하므로 최소 1장).")
        if not a.skip_train:
            try:
                import torch
                gpu = torch.cuda.is_available(); cap = torch.cuda.get_device_capability(0) if gpu else None; name = torch.cuda.get_device_name(0) if gpu else "none"
            except ImportError:
                gpu, cap, name = False, None, "torch not installed"
            if not gpu:
                raise Stop("GPU 가 없습니다. Kaggle: 오른쪽 Settings → Accelerator → GPU T4 x2. Colab: 런타임 → 런타임 유형 변경 → T4 GPU. 바꾼 뒤 다시 실행하세요.")
            if cap and cap[0] < 7:
                raise Stop(f"이 GPU({name})는 지원되지 않습니다. Kaggle 에서는 Accelerator 를 'GPU T4 x2' 로 바꾼 뒤 다시 실행하세요(P100 불가).")
            log(f"GPU {name}")
        log.stage(1, "photos", lambda: (out / "photos.csv").exists(), lambda: stage_photos(a, out, work, log))
        check_photos(out)
        log.stage(2, "poses", lambda: (out / "poses" / "poses.json").exists(), lambda: stage_poses(out, log))
        check_poses(out, log)
        log.stage(3, "masks", lambda: (out / "masks_summary.json").exists() and json.load(open(out / "masks_summary.json")).get("_done"),
                  lambda: stage_masks(a, out, work, log))
        log.stage(4, "points", lambda: (out / "plant_points.ply").exists(), lambda: stage_points(out, work, log))
        if a.skip_train:
            log("--skip-train: 5(학습)·6(출력) 단계는 실행하지 않았습니다."); return 0
        log.stage(5, "train", lambda: (out / "surfels_full.ply").exists(), lambda: stage_train(a, out, work, run, log))
        if a.stem:
            stem_masks.run(a, out, work, run, log, stage_points, stage_train, Stop)
        log.stage(6, "outputs", lambda: all((out / f).exists() for f in ("surfels.ply", "points.ply", "recon_meta.json", "REPORT_KO.md", "preview.png")),
                  lambda: stage_outputs(a, out, run, env, log))
        tprox_stage.run(a, out, log)
        log(f"완료. 결과 폴더: {out}  (REPORT_KO.md, preview.png, surfels.ply, points.ply, recon_meta.json)"); log(SCALE_LINE)
        return 0
    except Stop as ex:
        log("STOPPED:", str(ex)); return 2
    except Exception:                                                                    # noqa: BLE001
        import traceback
        log("ERROR:", traceback.format_exc()); return 1


if __name__ == "__main__":
    sys.exit(main())
