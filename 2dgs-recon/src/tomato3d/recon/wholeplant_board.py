"""A1: board refinement. Bundle adjustment on board corners only (no plant features): camera poses + the 54 ChArUco corner
positions, intrinsics fixed. Each corner's z is free; in-plane drift is regularised towards the printed position.

The gauge is fixed by the printed board: a 7-parameter similarity is NOT free here because the in-plane prior holds x and y near
their printed values and the z prior holds the mean height at 0, so the board frame (and the metric scale) stays run 01's.

usage: python -m tomato3d.recon.wholeplant_board --poses <charuco dir> --out <run dir>
Writes <run>/board_refined.json (poses, corner map, checks) and board_height_map.jpg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.wholeplant_contract import A1


def rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, np.float64))[0]


def residuals(x, n_cam, n_cor, obs, K, dist, P0, s_xy, s_z):
    rv = x[:n_cam * 3].reshape(-1, 3); tv = x[n_cam * 3:n_cam * 6].reshape(-1, 3)
    P = P0 + x[n_cam * 6:].reshape(-1, 3)
    ci, pi, uv = obs
    res = np.empty((len(ci), 2))
    for c in np.unique(ci):
        m = ci == c
        pr = cv2.projectPoints(P[pi[m]], rv[c], tv[c], K, dist)[0].reshape(-1, 2)
        res[m] = pr - uv[m]
    d = x[n_cam * 6:].reshape(-1, 3)
    return np.concatenate([res.ravel(), (d[:, :2] / s_xy).ravel(), d[:, 2] / s_z, [d[:, 2].mean() / s_xy]])


def sparsity(n_cam, n_cor, ci, pi):
    m = len(ci) * 2; n = n_cam * 6 + n_cor * 3
    S = lil_matrix((m + n_cor * 3 + 1, n), dtype=int)
    r = np.arange(len(ci))
    for k in range(6):
        S[2 * r, ci * 6 + k] = 1; S[2 * r + 1, ci * 6 + k] = 1
    for k in range(3):
        S[2 * r, n_cam * 6 + pi * 3 + k] = 1; S[2 * r + 1, n_cam * 6 + pi * 3 + k] = 1
    for j in range(n_cor):
        for k in range(3):
            S[m + j * 3 + k, n_cam * 6 + j * 3 + k] = 1
        S[-1, n_cam * 6 + j * 3 + 2] = 1
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv); src, out = Path(a.poses), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    po = json.load(open(src / "poses.json")); views = po["accepted"]
    K = np.array(po["K_full"]); dist = np.array(po["dist"]); dets = {d["file"]: d for d in json.load(open(src / "detections.json"))}
    P0 = ca.board().getChessboardCorners().astype(np.float64)          # 54 printed corners, board frame (z = 0)
    n_cam, n_cor = len(views), len(P0)
    rv0, tv0, ci, pi, uv = [], [], [], [], []
    for c, v in enumerate(views):
        d = dets[v]; O, I = ca.pose_points(d)
        rvec, tvec, *_ = ca.pnp(d, K, dist)
        rv0.append(rvec); tv0.append(tvec)
        ci += [c] * len(d["ids"]); pi += list(d["ids"]); uv.append(np.asarray(d["xy"], np.float64))
    ci = np.array(ci); pi = np.array(pi); uv = np.concatenate(uv)
    x0 = np.concatenate([np.array(rv0).ravel(), np.array(tv0).ravel(), np.zeros(n_cor * 3)])
    before = residuals(x0, n_cam, n_cor, (ci, pi, uv), K, dist, P0, A1["inplane_sigma_m"], 0.01)[:len(ci) * 2].reshape(-1, 2)
    rms0 = float(np.sqrt((before ** 2).sum(1).mean()))
    r = least_squares(residuals, x0, jac_sparsity=sparsity(n_cam, n_cor, ci, pi), x_scale="jac", method="trf", loss="huber",
                      f_scale=3.0, max_nfev=60, verbose=0, args=(n_cam, n_cor, (ci, pi, uv), K, dist, P0, A1["inplane_sigma_m"], 0.01))
    after = r.fun[:len(ci) * 2].reshape(-1, 2); rms1 = float(np.sqrt((after ** 2).sum(1).mean()))
    rv1 = r.x[:n_cam * 3].reshape(-1, 3); tv1 = r.x[n_cam * 3:n_cam * 6].reshape(-1, 3); dP = r.x[n_cam * 6:].reshape(-1, 3)
    shifts, rots, poses = [], [], {}
    for c, v in enumerate(views):
        R0 = rodrigues(rv0[c]); R1 = rodrigues(rv1[c])
        C0 = -R0.T @ np.asarray(tv0[c]); C1 = -R1.T @ tv1[c]
        shifts.append(float(np.linalg.norm(C1 - C0)))
        rots.append(float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1)))))
        # board frame -> world frame (z up), as in charuco_a3.world_pose
        R_cw = R1 @ ca.A_WB; t_cw = tv1[c] - R1 @ ca.A_WB @ ca.B_WB
        poses[v] = dict(R_cw=R_cw.tolist(), t_cw=t_cw.tolist(), center_world=(-R_cw.T @ t_cw).tolist(),
                        shift_m=shifts[-1], rot_deg=rots[-1])
    ok = rms1 <= A1["corner_rms_max_px"] and max(shifts) <= A1["cam_shift_max_m"] and max(rots) <= A1["cam_rot_max_deg"]
    res = dict(accepted=bool(ok), corner_rms_px=dict(before=rms0, after=rms1, limit=A1["corner_rms_max_px"]),
               camera_change=dict(shift_m_max=max(shifts), shift_m_median=float(np.median(shifts)), rot_deg_max=max(rots),
                                  rot_deg_median=float(np.median(rots)), limits=[A1["cam_shift_max_m"], A1["cam_rot_max_deg"]]),
               board_height_mm=dict(max_abs=float(np.abs(dP[:, 2]).max() * 1000), p95=float(np.percentile(np.abs(dP[:, 2]), 95) * 1000),
                                    rms=float(np.sqrt((dP[:, 2] ** 2).mean()) * 1000), note="deviation of the printed sheet from flat; describes the paper, not the plant"),
               board_inplane_mm=dict(max_abs=float(np.abs(dP[:, :2]).max() * 1000), rms=float(np.sqrt((dP[:, :2] ** 2).mean()) * 1000)),
               observations=int(len(ci)), views=len(views), corners=n_cor, poses=poses,
               corner_offsets_mm=(dP * 1000).round(4).tolist(), frame="board metres, z up (same gauge as run 01)")
    json.dump(res, open(out / "board_refined.json", "w"), indent=1)
    # height map
    g = np.full((ca.SQUARES[1] - 1, ca.SQUARES[0] - 1), np.nan)
    for j, p in enumerate(P0):
        g[int(round(p[1] / ca.SQUARE_M)) - 1, int(round(p[0] / ca.SQUARE_M)) - 1] = dP[j, 2] * 1000
    big = cv2.resize(np.nan_to_num(g), (640, 420), interpolation=cv2.INTER_CUBIC)
    v = np.abs(big).max() or 1.0
    im = cv2.applyColorMap(np.uint8(np.clip((big / v + 1) * 127.5, 0, 255)), cv2.COLORMAP_JET)
    cv2.putText(im, f"board height map, mm (max |dz| {np.abs(g[~np.isnan(g)]).max():.2f})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imwrite(str(out / "board_height_map.jpg"), im)
    print(json.dumps({k: res[k] for k in ("accepted", "corner_rms_px", "camera_change", "board_height_mm", "board_inplane_mm")}, indent=1))


if __name__ == "__main__":
    main()
