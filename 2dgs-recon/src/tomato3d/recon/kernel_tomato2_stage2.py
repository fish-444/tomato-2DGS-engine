# Kaggle CPU kernel — 토마토2 Stage 2: board detection, pose QC (charuco_a3 verbatim), coverage and triage per individual.
# Input dataset: dolgfish/tomato-recon-tomato2-input (t2_0x/*.HEIC, census.csv, code/charuco_a3.py). No GPU. Nothing runs locally.
import csv, glob, json, os, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif"], check=True)
import cv2, numpy as np, pillow_heif
from PIL import Image
pillow_heif.register_heif_opener()

ROOT = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/census.csv", recursive=True))))
sys.path.insert(0, os.path.join(ROOT, "code")); import charuco_a3 as ca                       # the R3 line's module, verbatim
WORK, TMP = "/kaggle/working", "/kaggle/tmp/jpg"; os.makedirs(TMP, exist_ok=True)
EXCLUDE = {"t2_04": {"IMG_8389.HEIC", "IMG_8418.HEIC", "IMG_8430.HEIC"}}                          # other date; excluded, never deleted
LOW_ELEV, BANDS = 35.0, [(0, 30), (30, 50), (50, 70), (70, 91)]
PROMPT_QC = dict(corners=12, spread=0.15, rms_1512=1.0)                                           # the prompt's figures, secondary column
ca.MIN_POINTS = 6                                                                                  # D7 (Amendment 1): 6-7 pose points allowed ...
D7_MAX_RMS_1512 = 1.5                                                                              # ... only with inlier RMS <= 1.5 px at the 1512-equivalent
T0 = time.time()
cen = {r["file"]: r for r in csv.DictReader(open(os.path.join(ROOT, "census.csv"), encoding="utf-8")) if r["individual"]}


def decode(args):
    src, dst = args
    if not os.path.exists(dst):
        Image.open(src).convert("RGB").save(dst, quality=95)
    return dst


def run_group(files, out_dir, tag):
    """charuco_a3.main's steps on one camera configuration (same frame size): detections, calibration, per-view QC, poses."""
    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        dets = list(ex.map(ca.detect, files))
    json.dump(dets, open(f"{out_dir}/detections_{tag}.json", "w"))
    W, H = dets[0]["width"], dets[0]["height"]; assert all((d["width"], d["height"]) == (W, H) for d in dets), "mixed image sizes"
    usable = [d for d in dets if d["corners"] >= ca.MIN_CORNERS]
    if len(usable) < 4:
        return dict(tag=tag, size=[W, H], n=len(dets), usable=len(usable), error="too few usable views for calibration"), dets, [], {}
    K, dist, rms0 = ca.calibrate(usable, (W, H))
    r = [ca.pnp(d, K, dist)[2] for d in usable]; keep = [d for d, x in zip(usable, r) if x <= 3 * np.median(r)]
    K, dist, rms = ca.calibrate(keep, (W, H))
    rows, poses = [], {}
    for d in dets:
        O, I = ca.pose_points(d)
        row = dict(file=d["file"], markers=d["markers"], markers_plane=d.get("markers_plane", 0), corners=d["corners"], points=len(O), spread=round(ca.spread(I, W, H), 3),
                   rms_px="", rms_px_1512="", median_px="", inlier_frac="", cam_height_m="", elevation_deg="", azimuth_deg="", accepted=0, reason="", accepted_prompt_qc=0, d7_low_points=0)
        if len(O) < ca.MIN_POINTS:
            row["reason"] = f"points<{ca.MIN_POINTS}"; rows.append(row); continue
        try:
            rv, tv, e, med, inl = ca.pnp(d, K, dist); R_cw, t_cw = ca.world_pose(rv, tv); C = -R_cw.T @ t_cw
        except cv2.error as ex:                                                                  # degenerate 6-7 point layout (D7): reject, do not crash
            row["reason"] = "pnp_failed"; rows.append(row); continue
        ctr = np.array([ca.SQUARES[0] * ca.SQUARE_M / 2, ca.SQUARES[1] * ca.SQUARE_M / 2, 0.0]); v = C - ctr
        row.update(rms_px=round(e, 3), rms_px_1512=round(e * 1512 / max(W, H), 3), median_px=round(med, 3), inlier_frac=round(float(inl.mean()), 3), cam_height_m=round(float(C[2]), 4),
                   elevation_deg=round(float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))), 1), azimuth_deg=round(float(np.degrees(np.arctan2(v[1], v[0])) % 360), 1))
        bad = [m for m, c in ((f"rms>{ca.MAX_VIEW_RMS_PX}", e > ca.MAX_VIEW_RMS_PX), (f"spread<{ca.MIN_SPREAD}", row["spread"] < ca.MIN_SPREAD),
                              (f"inliers<{ca.MIN_INLIER_FRAC}", inl.mean() < ca.MIN_INLIER_FRAC), ("camera below board", C[2] <= 0)) if c]
        if len(O) < 8 and row["rms_px_1512"] > D7_MAX_RMS_1512:
            bad.append(f"d7_points<8_rms1512>{D7_MAX_RMS_1512}")
        row["accepted"] = int(not bad); row["reason"] = ";".join(bad); row["d7_low_points"] = int(len(O) < 8 and not bad)
        row["accepted_prompt_qc"] = int(not bad and d["corners"] >= PROMPT_QC["corners"] and row["spread"] >= PROMPT_QC["spread"] and row["rms_px_1512"] <= PROMPT_QC["rms_1512"])
        rows.append(row); poses[d["file"]] = dict(R_cw=R_cw.tolist(), t_cw=t_cw.tolist(), center_world=C.tolist(), rms_px=e, accepted=row["accepted"], K=K.tolist(), dist=dist.tolist(), size=[W, H])
    with open(f"{out_dir}/qc_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ids = sorted({i for d in dets for i in d.get("mk_ids", [])}); board_ids = [int(i) for i in ca.board().getIds().ravel()]
    info = dict(tag=tag, size=[W, H], n=len(dets), usable=len(usable), calib_views=len(keep), calib_rms_px=round(rms, 3), calib_rms_first_px=round(rms0, 3), K=K.tolist(), dist=dist.tolist(),
                accepted=int(sum(r["accepted"] for r in rows)), accepted_prompt_qc=int(sum(r["accepted_prompt_qc"] for r in rows)),
                marker_ids_seen=ids, ids_outside_board=[i for i in ids if i not in board_ids], cam_height_m_median=float(np.median([r["cam_height_m"] for r in rows if r["cam_height_m"] != ""] or [0])))
    try:
        ca.qc_sheet(__import__("pathlib").Path(os.path.dirname(files[0])), dets, rows, f"{out_dir}/qc_sheet_{tag}.jpg")
    except Exception as ex:                                                                      # noqa: BLE001
        info["qc_sheet_error"] = str(ex)[:120]
    return info, dets, rows, poses


def coverage(rows):
    acc = [r for r in rows if r["accepted"]]; az = sorted(r["azimuth_deg"] for r in acc); el = [r["elevation_deg"] for r in acc]
    if not az:
        return dict(accepted=0, azimuth_coverage_deg=0.0, largest_gap_deg=360.0, elevation_min=None, elevation_max=None, rings={}, low_elevation_views=0)
    gaps = np.diff(az + [az[0] + 360]); return dict(accepted=len(acc), azimuth_coverage_deg=round(float(360 - gaps.max()), 1), largest_gap_deg=round(float(gaps.max()), 1),
                                                     elevation_min=min(el), elevation_max=max(el), rings={f"{a}-{b}": int(sum(a <= e < b for e in el)) for a, b in BANDS},
                                                     low_elevation_views=int(sum(e < LOW_ELEV for e in el)))


def triage(c):
    if c["accepted"] >= 45 and c["azimuth_coverage_deg"] >= 300:
        return "reconstructable"
    if c["accepted"] >= 30 or c["azimuth_coverage_deg"] >= 180:
        return "marginal"
    return "not_reconstructable"


report = dict(gpu="none (CPU kernel)", qc="charuco_a3 constants verbatim; prompt figures as secondary column", individuals={}); tiles = []
for iid in ("t2_01", "t2_02", "t2_03", "t2_04", "t2_05"):
    src = sorted(glob.glob(f"{ROOT}/{iid}/*.HEIC")); files = [p for p in src if os.path.basename(p) not in EXCLUDE.get(iid, set())]
    groups = defaultdict(list)
    for p in files:
        r = cen[os.path.basename(p)]; groups[f"f35_{r['FocalLengthIn35mmFilm']}_{r['decoded_w']}"].append(p)
    out_dir = f"{WORK}/{iid}"; os.makedirs(out_dir, exist_ok=True); rec = dict(files=len(src), used=len(files), excluded=sorted(EXCLUDE.get(iid, set())), groups={}, poses={})
    all_rows = []
    for tag, fl in sorted(groups.items()):
        jd = f"{TMP}/{iid}/{tag}"; os.makedirs(jd, exist_ok=True)
        with ProcessPoolExecutor(4) as ex:
            jpgs = list(ex.map(decode, [(p, f"{jd}/{os.path.basename(p)[:-5]}.jpg") for p in fl]))
        info, dets, rows, poses = run_group(jpgs, out_dir, tag); rec["groups"][tag] = info
        for r in rows:
            r["group"] = tag
        all_rows += rows; rec["poses"].update({f"{k}": dict(v, group=tag) for k, v in poses.items()})
        print(iid, tag, {k: info[k] for k in info if k not in ("K", "dist", "marker_ids_seen")}, f"{time.time() - T0:.0f}s", flush=True)
    rec["coverage_all"] = coverage(all_rows); rec["triage_all"] = triage(rec["coverage_all"])
    if len(groups) > 1:
        for tag in groups:
            c = coverage([r for r in all_rows if r["group"] == tag]); rec[f"coverage_{tag}"] = c; rec[f"triage_{tag}"] = triage(c)
    with open(f"{out_dir}/qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    json.dump(dict(frame="board metres, z up (charuco_a3)", views=rec["poses"], accepted=sorted(k for k, v in rec["poses"].items() if v["accepted"])), open(f"{out_dir}/poses.json", "w"))
    rec.pop("poses"); report["individuals"][iid] = rec
    mid = files[len(files) // 2]; im = Image.open(mid).convert("RGB"); im.thumbnail((600, 600)); t = np.asarray(im).copy()
    cv2.putText(t, f"{iid} {os.path.basename(mid)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2); tiles.append(cv2.resize(t, (600, 600)))
    json.dump(report, open(f"{WORK}/stage2_report.json", "w"), indent=1)
Image.fromarray(np.hstack(tiles)).save(f"{WORK}/appearance_sheet.jpg", quality=85)
shutil.rmtree(TMP, ignore_errors=True)                                                             # decoded intermediates: not an output
print(json.dumps({k: dict(used=v["used"], triage=v["triage_all"], cov=v["coverage_all"]) for k, v in report["individuals"].items()}, indent=1), f"total {time.time() - T0:.0f}s")
