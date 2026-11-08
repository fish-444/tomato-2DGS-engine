"""Carry a skeleton_v1 forward to a later capture (contract build/p1_skeleton_layer_20261101_01/CONTRACT.md, section 6 = prompt section 6).

1. register(t2 -> t1): board/pot frame first (both reconstructions share the board frame), then rigid ICP (no scaling) on the pot rim and the lowest 20 mm of plant surfels; residual reported.
2. carry(): transform each t1 polyline, snap every point to the nearest axis / rachis surfel of t2 within 5 mm (else keep, flag `no_snap`); a base with no surface within 8 mm -> status `candidate_lost`;
   source = `propagated_from:<t1 date>`, same leaf_uid.
3. growth_table(): per leaf_uid, Delta rachis length (path), Delta base height, Delta azimuth / elevation (first 10 mm), Delta area (b) and Delta absorbed light when both dates carry labels; new / lost listed;
   `check_identity` when the base moved > 10 mm or the azimuth changed > 30 deg (shown, never auto-fixed).
No second capture date exists yet, so this module is exercised on synthetic transforms only (tests/test_skeleton_layer.py). Propagation afterwards = skeleton_propagate, unchanged parameters.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs import skeleton_tree as ST

C = dict(icp_iters=30, icp_max_pair_mm=5.0, low_band_mm=20.0, snap_mm=5.0, lost_mm=8.0, base_move_mm=10.0, az_change_deg=30.0)


# ---------------------------------------------------------------- registration
def kabsch(A, B):
    """Rigid transform (R, t) with B ~ R A + t, no scaling."""
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, cb - R @ ca


def stable_part(xyz, rim_xyz=None, z_base=None):
    """The pot rim (if given) plus the lowest 20 mm of plant surfels above z_base (default: the 2nd percentile of z)."""
    z0 = float(np.percentile(xyz[:, 2], 2)) if z_base is None else z_base; low = xyz[(xyz[:, 2] >= z0) & (xyz[:, 2] <= z0 + C["low_band_mm"] / 1000)]
    return np.vstack([low] + ([np.asarray(rim_xyz, float)] if rim_xyz is not None and len(rim_xyz) else []))


def register(xyz_t1, xyz_t2, rim_t1=None, rim_t2=None):
    """ICP of the t2 stable part onto the t1 stable part (board frame first = identity start). Returns dict(R, t, residual_mm, pairs, iterations): x_t1 ~ R x_t2 + t."""
    A = stable_part(xyz_t2, rim_t2); B = stable_part(xyz_t1, rim_t1); tree = cKDTree(B); R = np.eye(3); t = np.zeros(3); res = None
    for it in range(C["icp_iters"]):
        P = A @ R.T + t; d, j = tree.query(P); m = d <= C["icp_max_pair_mm"] / 1000
        if m.sum() < 10:
            break
        R2, t2 = kabsch(P[m], B[j[m]]); R = R2 @ R; t = R2 @ t + t2; new = float(np.sqrt((d[m] ** 2).mean()) * 1000)
        if res is not None and abs(res - new) < 1e-4:
            res = new; break
        res = new
    return dict(R=R, t=t, residual_mm=res, pairs=int(m.sum()), iterations=it + 1, scaling="none (both dates must be scale-checked before any scaling)")


# ---------------------------------------------------------------- carry
def carry(sk_t1, reg, xyz_t2, axis_like_t2, capture_date_t2, reconstruction_id_t2):
    """Transform t1 polylines into the t2 frame (x_t2 = R^T (x_t1 - t)), snap to the nearest axis/rachis surfel within 5 mm, flag the rest."""
    Rm, t = np.asarray(reg["R"]), np.asarray(reg["t"]); Rinv = Rm.T; ax = np.asarray(xyz_t2, float)[np.asarray(axis_like_t2, bool)]
    tree_ax = cKDTree(ax) if len(ax) else None; tree_all = cKDTree(np.asarray(xyz_t2, float))
    sk2 = SIO.new_skeleton(sk_t1["plant_id"], capture_date_t2, reconstruction_id_t2, frame=sk_t1["frame"]["name"], scale_status=sk_t1["scale_status"]); sk2["registration"] = dict(residual_mm=reg["residual_mm"], pairs=reg["pairs"], iterations=reg["iterations"], scaling=reg["scaling"], from_date=sk_t1["capture_date"])
    for l in sk_t1["leaves"]:
        if l["status"] not in ("active", "new", "candidate_lost"):
            continue
        P = (np.asarray(l["polyline_xyz"], float) - t) @ Rinv.T; snapped = P.copy(); flags = []
        if tree_ax is not None:
            d, j = tree_ax.query(P); ok = d <= C["snap_mm"] / 1000; snapped[ok] = ax[j[ok]]
            if (~ok).any():
                flags.append("no_snap")
        else:
            flags.append("no_snap")
        status = "candidate_lost" if tree_all.query(P[0])[0] > C["lost_mm"] / 1000 else "active"
        SIO.add_leaf(sk2, snapped, source=f"propagated_from:{sk_t1['capture_date']}", status=status, flags=flags, uid=l["leaf_uid"], attrs=dict(carried_from=l["leaf_uid"], snapped_points=int(ok.sum()) if tree_ax is not None else 0))
    carry_axes(sk_t1, sk2, Rinv, t, tree_ax, ax)
    return sk2


def carry_axes(sk_t1, sk2, Rinv, t, tree_ax, ax):
    """Amendment A6: axis points (user clicks only; derived anchors are recomputed by the fit) carried like leaf lines - same transform, snap to axis surfels within 5 mm, same axis_uid."""
    for a in sk_t1.get("axes", []):
        if a["status"] != "active":
            continue
        a2 = dict(axis_uid=a["axis_uid"], type=a["type"], parent_axis_uid=a.get("parent_axis_uid"), attach_s_mm=None, status="active", points=[], fit=None, segments=[], nodes=[], flags=["propagated_axis"])
        for p in a["points"]:
            if p["source"] not in ("user_click", "observed_photo_click", "observed_3d_click"):
                continue
            q = (np.asarray(p["xyz"], float) - t) @ Rinv.T; src = "user_click"
            if tree_ax is not None:
                d, j = tree_ax.query(q)
                if d <= C["snap_mm"] / 1000:
                    q = ax[j]
            a2["points"].append(dict(xyz=q.tolist(), source=src, carried_from=sk_t1["capture_date"]))
        sk2.setdefault("axes", []).append(a2); sk2["schema"] = SIO.SCHEMA_AXES


# ---------------------------------------------------------------- growth table
def leaf_summary(l):
    P = np.asarray(l["polyline_xyz"], float) * 1000.0; d = ST.first_direction(P); az, el = ST.az_el(d)
    return dict(base_z_mm=float(P[0, 2]), base_mm=P[0], path_mm=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0, az=az, el=el)


def growth_table(sk_t1, sk_t2, areas_t1=None, areas_t2=None, light_t1=None, light_t2=None):
    """Per leaf_uid differences t2 - t1; new and lost leaves listed; check_identity flags. areas/light: {leaf_uid: value} or None."""
    L1 = {l["leaf_uid"]: l for l in sk_t1["leaves"] if l["status"] in ("active", "new", "candidate_lost")}; L2 = {l["leaf_uid"]: l for l in sk_t2["leaves"]}
    rows = []
    for uid in sorted(set(L1) | set(L2)):
        a, b = L1.get(uid), L2.get(uid)
        if a is None:
            rows.append(dict(leaf_uid=uid, change="new")); continue
        if b is None or b["status"] in ("lost", "candidate_lost"):
            rows.append(dict(leaf_uid=uid, change="lost" if b is None or b["status"] == "lost" else "candidate_lost")); continue
        s1, s2 = leaf_summary(a), leaf_summary(b); daz = None if s1["az"] is None or s2["az"] is None else float(abs((s2["az"] - s1["az"] + 180) % 360 - 180))
        row = dict(leaf_uid=uid, change="kept", d_rachis_path_mm=round(s2["path_mm"] - s1["path_mm"], 2), d_base_height_mm=round(s2["base_z_mm"] - s1["base_z_mm"], 2), base_move_mm=round(float(np.linalg.norm(s2["base_mm"] - s1["base_mm"])), 2),
                   d_azimuth_deg=(None if daz is None else round(daz, 1)), d_elevation_deg=(None if s1["el"] is None or s2["el"] is None else round(s2["el"] - s1["el"], 1)))
        for name, A_, B_ in (("d_area_b_mm2", areas_t1, areas_t2), ("d_absorbed_light", light_t1, light_t2)):
            row[name] = round(B_[uid] - A_[uid], 4) if (A_ and B_ and uid in A_ and uid in B_) else None
        row["check_identity"] = bool(row["base_move_mm"] > C["base_move_mm"] or (daz is not None and daz > C["az_change_deg"]))
        rows.append(row)
    return rows
