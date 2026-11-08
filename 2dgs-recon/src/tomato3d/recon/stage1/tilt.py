"""E3 - elevation of each compound from a side view, and top-projected leaflet length / area -> true values.

Geometry (orthographic synthetic views): the top view gives the compound's horizontal base -> tip vector (length
``L_h``, direction ``d``); a side view whose image axis is most parallel to ``d`` gives the heights at the base, the
middle and the tip columns. Elevation (definitions: ``elevation_deg``, + = up) of the base->mid and mid->tip halves:
if they differ by more than ``single_angle_max_bend_deg`` (priors file) the compound is bent - corrected PER HALF
(alpha definition, no single angle); a half whose height cannot be read -> ``unavailable``. A compound pointing at
the camera (horizontal extent across the view < ``min_side_dx_fraction`` of ``L_h``) is unresolved in that view.

Leaflet correction: a leaflet at station s gets the elevation of its half; true area = visible top area / cos(e)
(pitch along the rachis only; blade curvature deferred, O7 -> flat assumption, stated in every row); true length
combines the along-rachis component / cos(e) with the across component.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.recon.stage1.priors import Priors
from tomato3d.recon.stage1.views import SideObs, TopObs, side_transform

COLUMN_HALF_MM = 1.0
BLADE_NOTE = "blade curvature deferred (O7): leaflet assumed flat; pitch along the rachis only"


def _pca(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    p = np.stack([xs, ys], 1).astype(float)
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    proj = (p - c) @ vt[0]
    return c, float(proj.max() - proj.min()), vt[0]


def _height_at(sv: SideObs, label: int, u_mm: float) -> tuple[float, float] | None:
    """(median height, robust sd) of the compound's side pixels in the column window: other parts of the same
    compound at other depths share the column, so the spread IS the read-out's uncertainty."""
    c0 = int(np.floor(sv.to_px(u_mm - COLUMN_HALF_MM, 0)[0]))
    c1 = int(np.ceil(sv.to_px(u_mm + COLUMN_HALF_MM, 0)[0])) + 1
    rows, _ = np.nonzero(sv.id_map[:, max(c0, 0):max(c1, 0)] == label)
    if not len(rows):
        return None
    z = (sv.camera.center_px - rows) * sv.camera.mm_per_px
    sd = float((np.percentile(z, 75) - np.percentile(z, 25)) / 1.349) if len(z) >= 4 else float(np.ptp(z))
    return float(np.median(z)), max(sd, sv.camera.mm_per_px)


def compound_geometry(top: TopObs, cid: str, base_xy: np.ndarray) -> dict | None:
    labs = [k for k, v in top.labels.items() if v["compound"] == cid]
    ys, xs = np.nonzero(np.isin(top.id_map, labs))
    if not len(xs):
        return None
    pts = top.camera.unproject_xy(np.stack([xs, ys], 1).astype(float))
    d = np.linalg.norm(pts - base_xy, axis=1)
    tip = pts[int(np.argmax(d))]
    v = tip - base_xy
    lh = float(np.linalg.norm(v))
    return dict(base_xy=base_xy, tip_xy=tip, dir=v / max(lh, 1e-9), L_h=lh)


def run(top: TopObs, sides: Sequence[SideObs], attach_rows: Sequence[Mapping], priors: Priors) -> dict:
    max_bend = float(priors.value("geometry", "single_angle_max_bend_deg"))
    min_dx = float(priors.value("geometry", "min_side_dx_fraction"))
    comp_rows, leaf_rows = [], []
    for ar in attach_rows:
        cid = ar["compound"]
        if ar.get("base_xy_mm") is None:
            comp_rows.append(dict(compound=cid, elevation_deg=None, mode=UNAVAILABLE, reason="no top-view pixels"))
            continue
        geo = compound_geometry(top, cid, np.asarray(ar["base_xy_mm"], float))
        if geo is None or geo["L_h"] < 1.0:
            comp_rows.append(dict(compound=cid, elevation_deg=None, mode=UNAVAILABLE, reason="no top-view extent"))
            continue
        best = None
        for sv in sides:
            inv = {v: k for k, v in sv.labels.items()}
            if cid not in inv:
                continue
            right = side_transform(np.array([[1.0, 0, 0], [0, 1.0, 0]]), sv.azimuth_deg)[:, 0]
            across = abs(float(geo["dir"] @ right))
            if best is None or across > best[1]:
                best = (sv, across, inv[cid])
        e1 = e2 = sd1 = sd2 = None
        reason, mode = "", UNAVAILABLE
        if best is None:
            reason = "compound not visible in any side view"
        elif best[1] < min_dx:
            reason = f"compound points at every side camera (|d.across| {best[1]:.2f} < {min_dx})"
        else:
            sv, _, lab = best

            def u_of(xy):
                return float(side_transform(np.array([[xy[0], xy[1], 0.0]]), sv.azimuth_deg)[0, 0])
            mid = geo["base_xy"] + geo["dir"] * geo["L_h"] / 2
            if ar.get("attachment_z_mm") is not None:
                zb = (ar["attachment_z_mm"], ar.get("attachment_z_sd_mm") or sv.camera.mm_per_px)
            else:
                zb = _height_at(sv, lab, u_of(geo["base_xy"]))
            zm, zt = _height_at(sv, lab, u_of(mid)), _height_at(sv, lab, u_of(geo["tip_xy"]))
            half = geo["L_h"] / 2
            if zb is None or zt is None:
                reason = "base or tip height hidden in the side view"
            else:
                if zm is not None:
                    e1 = math.degrees(math.atan2(zm[0] - zb[0], half))
                    e2 = math.degrees(math.atan2(zt[0] - zm[0], half))
                    sd1 = math.degrees(math.atan2(math.hypot(zb[1], zm[1]), half))
                    sd2 = math.degrees(math.atan2(math.hypot(zm[1], zt[1]), half))
                if e1 is not None and abs(e1 - e2) > max_bend:
                    mode = "piecewise_2seg"
                else:
                    mode = "single_angle"
                    e1 = e2 = math.degrees(math.atan2(zt[0] - zb[0], geo["L_h"]))
                    sd1 = sd2 = math.degrees(math.atan2(math.hypot(zb[1], zt[1]), geo["L_h"]))
        e_all = None if e1 is None else math.degrees(math.atan2(
            math.tan(math.radians(e1)) * geo["L_h"] / 2 + math.tan(math.radians(e2)) * geo["L_h"] / 2, geo["L_h"]))
        comp_rows.append(dict(compound=cid, elevation_deg=e_all, elevation_base_half_deg=e1, elevation_tip_half_deg=e2,
                              elevation_base_half_sd_deg=sd1 if e1 is not None else None,
                              elevation_tip_half_sd_deg=sd2 if e1 is not None else None,
                              mode=mode if e1 is not None else UNAVAILABLE, reason=reason, L_h_mm=geo["L_h"],
                              dir=geo["dir"], base_xy=geo["base_xy"], tip_xy=geo["tip_xy"],
                              view_deg=best[0].azimuth_deg if best else None))
        # leaflets of the compound
        for lab, info in top.labels.items():
            if info["compound"] != cid or info["kind"] != "leaflet":
                continue
            ys, xs = np.nonzero(top.id_map == lab)
            if not len(xs):
                continue
            pts = top.camera.unproject_xy(np.stack([xs, ys], 1).astype(float))
            c, L, ax = _pca(pts[:, 0], pts[:, 1])
            s = float((c - geo["base_xy"]) @ geo["dir"])
            side = float(geo["dir"][0] * (c - geo["base_xy"])[1] - geo["dir"][1] * (c - geo["base_xy"])[0])
            area_px = len(xs) * top.mm2_per_px()
            e = None if e1 is None else (e1 if s < geo["L_h"] / 2 else e2)
            e_sd = None if e1 is None else (sd1 if s < geo["L_h"] / 2 else sd2)
            if e is None or abs(e) >= 89.0:
                L_true = area_true = None
            else:
                ce = math.cos(math.radians(e))
                cos_phi = abs(float(ax @ geo["dir"]))
                L_true = math.hypot(L * cos_phi / ce, L * math.sqrt(max(0.0, 1 - cos_phi ** 2)))
                area_true = area_px / ce
            leaf_rows.append(dict(obs_id=info["obs_id"], label=lab, compound=cid, is_terminal=info["is_terminal"],
                                  station_mm=s, side=("L" if side > 0 else "R"), lateral_mm=abs(side),
                                  centroid_xy=c, axis_xy=ax, L_top_mm=L, area_top_mm2=area_px, elevation_deg=e,
                                  elevation_sd_deg=e_sd, cos_phi=None if e1 is None else abs(float(ax @ geo["dir"])),
                                  L_true_mm=L_true, area_corrected_mm2=area_true, blade=BLADE_NOTE))
    return dict(compounds=comp_rows, leaflets=leaf_rows)
