"""A3 ChArUco capture: board definition, detection, per-view QC, intrinsics, board poses, COLMAP text export.

Board (read from the print header and checked against the printed 100 mm ruler, 2026-10-21):
ChArUco 10 x 7 squares, square 38.0 mm, marker 28.0 mm, DICT_5X5_100, ids 0-34, A3 landscape.

World frame ("board metres"): origin at the board's printed bottom-left square corner, +X along the 10-square side,
+Y along the 7-square side, +Z up out of the board (plant side).  world = A @ board_cv + b with A = diag(1, -1, -1),
b = (0, 7 * 0.038, 0), where board_cv is OpenCV's board frame (x right, y down along the print, z into the board).

usage: python -m tomato3d.recon.charuco_a3 --jpg <dir> --out <dir> [--res 3024]
Writes detections.json, qc.csv, intrinsics.json, poses.json, colmap/{cameras,images,points3D}.txt, undist_<res>/*.jpg,
qc_sheet.jpg.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SQUARES = (10, 7)
SQUARE_M, MARKER_M = 0.038, 0.028
DICT = cv2.aruco.DICT_5X5_100
DETECT_SCALES = (1 / 3, 1 / 2)  # markers are ~400 px at full res; detected on CLAHE downscales, union by id (1/2 wins)
A_WB = np.diag([1.0, -1.0, -1.0]); B_WB = np.array([0.0, SQUARES[1] * SQUARE_M, 0.0])
# detector settings (2026-10-22, before any training): the defaults found ~half the visible markers on the low ring
ARUCO = dict(adaptiveThreshWinSizeMin=3, adaptiveThreshWinSizeMax=103, adaptiveThreshWinSizeStep=10, perspectiveRemovePixelPerCell=10,
             perspectiveRemoveIgnoredMarginPerCell=0.25, maxErroneousBitsInBorderRate=0.5, errorCorrectionRate=1.0,
             polygonalApproxAccuracyRate=0.08, minOtsuStdDev=2.0)
CORNER_WIN = 31                 # ChArUco cornerSubPix window at 6048 px, scaled with the image side (soft corners: 31 cut calib RMS 6.8 -> 5.4 px)
PLANE_FILTER = 0.02             # markers kept only if consistent with one board-to-image homography (RANSAC, 2 % of the image side);
                                # the tuned detector returns a few mis-decoded ids that otherwise make the ChArUco interpolation fail
# QC thresholds (declared; see CONTRACT.md). The print is not a perfect plane: ChArUco corners sit on the printed corners
# (checked visually) yet miss a single homography by 3-7 px at 6048 px, so the limits are set at what this board delivers.
MIN_CORNERS = 12                # ChArUco corners for a view to enter the calibration
MIN_POINTS = 8                  # ChArUco corners for a view's pose (marker corners are not used: noisier than the corners)
MAX_VIEW_RMS_PX = 8.0           # per-view PnP inlier RMS at 6048 px (= 2 px at the 1512-px training resolution)
MIN_SPREAD = 0.10               # convex-hull area of the pose points / image area
OUTLIER_FRAC = 2.5e-3           # point outlier distance as a fraction of the image side (15 px at 6048)
MIN_INLIER_FRAC = 0.8           # PnP inliers / pose points


def board():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard(SQUARES, SQUARE_M, MARKER_M, d)


def _params():
    p = cv2.aruco.DetectorParameters()
    for k, v in ARUCO.items():
        setattr(p, k, v)
    return p


def detect(path: str) -> dict:
    """Markers on CLAHE downscales (union by id), plane-consistency filter, ChArUco corners interpolated and refined at full resolution."""
    b = board(); g = cv2.imread(path, cv2.IMREAD_GRAYSCALE); h, w = g.shape; det = cv2.aruco.ArucoDetector(b.getDictionary(), _params())
    found = {}
    for s in DETECT_SCALES:
        small = cv2.createCLAHE(3.0, (8, 8)).apply(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        mc, mi, _ = det.detectMarkers(small)
        for c, i in zip(mc or [], [] if mi is None else mi.ravel()):
            if int(i) < len(b.getIds()):
                found[int(i)] = c.reshape(4, 2) / s
    out = dict(file=Path(path).name, width=w, height=h, markers=len(found), corners=0, ids=[], xy=[], mk_ids=[], mk_xy=[])
    if len(found) < 2:
        return out
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01); cw = max(3, round(CORNER_WIN * max(h, w) / 6048))
    mids = sorted(found)
    if len(mids) >= 5:
        bo = b.getObjPoints(); bids = [int(i) for i in b.getIds().ravel()]
        src = np.array([np.mean(bo[bids.index(i)], 0)[:2] for i in mids], np.float32); dst = np.array([found[i].mean(0) for i in mids], np.float32)
        _, inl = cv2.findHomography(src, dst, cv2.RANSAC, PLANE_FILTER * max(h, w))
        if inl is not None:
            mids = [i for i, k in zip(mids, inl.ravel()) if k]
    out["markers_plane"] = len(mids)
    mc = [found[i].astype(np.float32).reshape(1, 4, 2) for i in mids]
    out.update(mk_ids=mids, mk_xy=[c.reshape(4, 2).tolist() for c in mc])
    cc, ci, _, _ = cv2.aruco.CharucoDetector(b).detectBoard(g, markerCorners=tuple(mc), markerIds=np.array(mids, np.int32).reshape(-1, 1))
    if ci is not None and len(ci):
        cc = cv2.cornerSubPix(g, cc.astype(np.float32).reshape(-1, 1, 2), (cw, cw), (-1, -1), crit)
        out.update(corners=int(len(ci)), ids=ci.ravel().tolist(), xy=cc.reshape(-1, 2).tolist())
    return out


def obj_points(ids) -> np.ndarray:
    return board().getChessboardCorners()[np.asarray(ids, int)].astype(np.float64)


def pose_points(d: dict):
    """ChArUco corners (board frame, metres) and their pixels. Marker corners stay in detections.json but are not used."""
    if not d["ids"]:
        return np.zeros((0, 3)), np.zeros((0, 2))
    return obj_points(d["ids"]), np.asarray(d["xy"], np.float64)


def spread(xy, w, h) -> float:
    if len(xy) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.asarray(xy, np.float32))) / (w * h))


def outlier_px(size) -> float:
    return OUTLIER_FRAC * max(size)


def calibrate(dets: list[dict], size) -> tuple[np.ndarray, np.ndarray, float]:
    """cv2.calibrateCamera on the ChArUco corners; corners off by > outlier_px after a first fit are dropped, then refit."""
    O = [obj_points(d["ids"]).astype(np.float32) for d in dets]; I = [np.asarray(d["xy"], np.float32) for d in dets]
    fl = cv2.CALIB_FIX_K3                     # k3 free gave k2 = -2.5, k3 = +5.1 with no RMS gain (over-fit)
    rms, K, dist, rvs, tvs = cv2.calibrateCamera(O, I, size, None, None, flags=fl)
    keep = [np.linalg.norm(cv2.projectPoints(o, r, t, K, dist)[0].reshape(-1, 2) - i, axis=1) <= outlier_px(size)
            for o, i, r, t in zip(O, I, rvs, tvs)]
    rms, K, dist, _, _ = cv2.calibrateCamera([o[k] for o, k in zip(O, keep)], [i[k] for i, k in zip(I, keep)], size, K, dist,
                                             flags=fl | cv2.CALIB_USE_INTRINSIC_GUESS)
    return K, dist.ravel(), float(rms)


def pnp(d: dict, K, dist):
    """RANSAC PnP (SQPnP) on ChArUco + marker corners, then LM on the inliers. Returns rv, tv, inlier RMS, inlier median, inlier mask."""
    O, I = pose_points(d); thr = outlier_px((d.get("width", 1000), d.get("height", 1000)))
    ok, rv, tv, inl = cv2.solvePnPRansac(O, I, K, dist, reprojectionError=thr, iterationsCount=500, flags=cv2.SOLVEPNP_SQPNP)
    m = np.zeros(len(O), bool)
    if ok and inl is not None:
        m[inl.ravel()] = True
    if m.sum() < 6:
        m[:] = True; ok, rv, tv = cv2.solvePnP(O, I, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    rv, tv = cv2.solvePnPRefineLM(O[m], I[m], K, dist, rv, tv)
    res = np.linalg.norm(cv2.projectPoints(O, rv, tv, K, dist)[0].reshape(-1, 2) - I, axis=1); m = res <= thr
    return rv.ravel(), tv.ravel(), float(np.sqrt((res[m] ** 2).mean())), float(np.median(res[m])), m


def world_pose(rv, tv):
    """Camera pose in the world frame: x_cam = R_cw @ x_world + t_cw."""
    R_cb = cv2.Rodrigues(rv)[0]; t_cb = tv.reshape(3)
    # x_board = A^-1 (x_world - b); A is its own inverse
    R_cw = R_cb @ A_WB; t_cw = t_cb - R_cb @ A_WB @ B_WB
    return R_cw, t_cw


def rot_to_qvec(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat(); q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def scaled_K(K, src: int, dst: int):
    s = dst / src; K2 = K.copy(); K2[0, 0] *= s; K2[1, 1] *= s
    K2[0, 2] = (K[0, 2] + 0.5) * s - 0.5; K2[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K2


def _undistort(args):
    src, dst, K, dist, res = args
    im = cv2.imread(src); und = cv2.undistort(im, K, dist, None, K)
    cv2.imwrite(dst, cv2.resize(und, (res, res), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--jpg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=3024); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in Path(a.jpg).glob("*.jpg"))
    dp = out / "detections.json"
    if dp.exists():
        dets = json.load(open(dp))
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            dets = list(ex.map(detect, files))
        json.dump(dets, open(dp, "w"))
    W, H = dets[0]["width"], dets[0]["height"]; assert all((d["width"], d["height"]) == (W, H) for d in dets), "mixed image sizes"
    usable = [d for d in dets if d["corners"] >= MIN_CORNERS]
    # calibration: all usable views, then drop views whose PnP RMS > 3 x median and recalibrate once
    K, dist, rms0 = calibrate(usable, (W, H))
    r = [pnp(d, K, dist)[2] for d in usable]; keep = [d for d, x in zip(usable, r) if x <= 3 * np.median(r)]
    K, dist, rms = calibrate(keep, (W, H))
    json.dump(dict(model="OpenCV k1 k2 p1 p2 (k3 fixed 0)", size=[W, H], K=K.tolist(), dist=dist.tolist(), rms_px=rms, rms_first_px=rms0,
                   views_used=[d["file"] for d in keep], board=dict(squares=SQUARES, square_m=SQUARE_M, marker_m=MARKER_M, dict="DICT_5X5_100")),
              open(out / "intrinsics.json", "w"), indent=1)
    rows, poses = [], {}
    for d in dets:
        O, I = pose_points(d)
        row = dict(file=d["file"], markers=d["markers"], corners=d["corners"], points=len(O), spread=round(spread(I, W, H), 3), rms_px="", median_px="",
                   inlier_frac="", cam_height_m="", elevation_deg="", azimuth_deg="", accepted=0, reason="")
        if len(O) < MIN_POINTS:
            row["reason"] = f"points<{MIN_POINTS}"; rows.append(row); continue
        rv, tv, e, med, inl = pnp(d, K, dist); R_cw, t_cw = world_pose(rv, tv); C = -R_cw.T @ t_cw
        ctr = np.array([SQUARES[0] * SQUARE_M / 2, SQUARES[1] * SQUARE_M / 2, 0.0]); v = C - ctr
        row.update(rms_px=round(e, 3), median_px=round(med, 3), inlier_frac=round(float(inl.mean()), 3), cam_height_m=round(float(C[2]), 4),
                   elevation_deg=round(float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))), 1),
                   azimuth_deg=round(float(np.degrees(np.arctan2(v[1], v[0])) % 360), 1))
        bad = [m for m, c in ((f"rms>{MAX_VIEW_RMS_PX}", e > MAX_VIEW_RMS_PX), (f"spread<{MIN_SPREAD}", row["spread"] < MIN_SPREAD),
                              (f"inliers<{MIN_INLIER_FRAC}", inl.mean() < MIN_INLIER_FRAC), ("camera below board", C[2] <= 0)) if c]
        row["accepted"] = int(not bad); row["reason"] = ";".join(bad); rows.append(row)
        poses[d["file"]] = dict(R_cw=R_cw.tolist(), t_cw=t_cw.tolist(), center_world=C.tolist(), rms_px=e, accepted=row["accepted"])
    with open(out / "qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    acc = sorted(k for k, p in poses.items() if p["accepted"])
    Kr = scaled_K(K, W, a.res)
    json.dump(dict(frame="board metres, z up (see tomato3d/recon/charuco_a3.py)", convention="x_cam = R_cw @ x_world + t_cw; pixel = K @ x_cam / z",
                   full_size=[W, H], K_full=K.tolist(), dist=dist.tolist(), undistorted_res=a.res, K_undist=Kr.tolist(), views=poses, accepted=acc),
              open(out / "poses.json", "w"), indent=1)
    # undistorted, downscaled images for COLMAP and 2DGS (pinhole K_undist)
    ud = out / f"undist_{a.res}"; ud.mkdir(exist_ok=True)
    todo = [(str(Path(a.jpg) / f), str(ud / f), K, dist, a.res) for f in acc if not (ud / f).exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(_undistort, todo))
    cm = out / "colmap"; cm.mkdir(exist_ok=True)
    (cm / "cameras.txt").write_text(f"# Camera list\n1 PINHOLE {a.res} {a.res} {Kr[0, 0]:.6f} {Kr[1, 1]:.6f} {Kr[0, 2]:.6f} {Kr[1, 2]:.6f}\n")
    lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (world = board metres, z up; poses fixed from the board)"]
    for i, f in enumerate(acc, 1):
        q = rot_to_qvec(np.array(poses[f]["R_cw"])); t = poses[f]["t_cw"]
        lines += [f"{i} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {f}", ""]
    (cm / "images.txt").write_text("\n".join(lines) + "\n"); (cm / "points3D.txt").write_text("# empty: fixed-pose model\n")
    qc_sheet(Path(a.jpg), dets, rows, out / "qc_sheet.jpg")
    print(json.dumps(dict(views=len(dets), usable=len(usable), calib_views=len(keep), calib_rms=round(rms, 3), accepted=len(acc))))


def qc_sheet(jpg: Path, dets, rows, path, tile=240, cols=10):
    tiles = []
    for d, r in zip(dets, rows):
        im = cv2.imread(str(jpg / d["file"]), cv2.IMREAD_REDUCED_COLOR_8); s = im.shape[1] / d["width"]
        for x, y in d["xy"]:
            cv2.circle(im, (int(x * s), int(y * s)), 4, (0, 255, 0) if r["accepted"] else (0, 0, 255), -1)
        im = cv2.resize(im, (tile, tile)); cv2.putText(im, f"{d['file'][9:18]} c{d['corners']} {r['rms_px']}", (4, 16), 0, 0.42, (0, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
