"""Camera intrinsics (EXIF) and camera pose from the pot rim (P1 side rim pose, 2026-10-12, Y0-Y4).

The square rim (60 mm outer edge, ``potscale``) is a known planar target in every top and side photo. With the focal
length in pixels its homography gives the camera pose relative to the rim frame (origin at the pot centre on the rim
plane, Z up). This module:

* reads the HEIC EXIF (focal length, 35 mm equivalent, EXIF pixel dimensions) without decoding pixels;
* forms ``f_px`` on two bases of the 35 mm equivalence - pre-crop WIDTH (36 mm <-> EXIF width) and DIAGONAL
  (43.27 mm <-> EXIF diagonal). The stored photo (3024 x 3024) is a crop of the EXIF frame (4032 x 3024); a crop keeps
  the pixel scale, but its offset is not recorded, so the principal point is ASSUMED at the stored image centre;
* decomposes the homography into a pose and runs the two-axis SELF-CHECK (|r1| = |r2|, r1 . r2 = 0). A pose passes
  when both residuals lie within ``k_sigma`` Monte-Carlo standard deviations of the click uncertainty; otherwise it
  is ``approximate`` and is published only with that flag;
* estimates the focal length the rim itself implies (closed form from the same two constraints, principal point
  fixed) - well conditioned only for oblique views (the side photos), ill conditioned for near-frontal top views;
* gives the VERTICAL scale on the pot axis (mm per px of a vertical step at the rim centre) for side photos, and the
  camera distance ``d`` to the rim centre (Y0: the rim-plane bias of a leaf at height h is d / (d - h));
* upright-ruler cross-check: NOT identifiable from one side photo. Scaling the ruler's position and length about the
  camera centre leaves its image unchanged (depth-length ambiguity; the vertical vanishing point is fixed by the pose),
  so two or more ticks cannot test the scale without one absolute anchor - e.g. the height of reading 0 relative to
  the rim (ruler foot on the table + the pot depth, which T6 declared "recorded only, not a scale source"), or the
  ruler's position from another view. :func:`ruler_vertical_fit` takes that anchor once it is decided; until then the
  side cross-check is ``unavailable`` and ``scale_sources`` stays 1.

2026-10-13 (rim inner plane): the rim lines may be outer or inner top edges (rim plane) or the medium junction
(medium surface, ``depth`` below the rim) - ``potscale.reference_of``. The homography is solved on the reference
plane and the pose is moved into the RIM frame (origin at the rim centre), so distances, tilt and the vertical scale
always refer to the rim. A photo with three clicked sides and the fourth recorded ``side_occluded`` is solved from
the three lines with the focal FIXED (:func:`pose_from_lines`); three lines leave no redundancy for the two-axis
self-check, so such a pose is ``approximate`` by construction and the fourth side is reported as occluded.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from tomato3d.state import potscale as ps

FULL_FRAME_WIDTH_MM = 36.0
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)


class CameraError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# EXIF and focal length
# --------------------------------------------------------------------------------------


def read_exif(path: Path) -> dict[str, Any]:
    """Focal length (mm), 35 mm equivalent (mm), EXIF pixel dimensions and lens from a HEIC/JPEG file's TIFF block."""
    buf = Path(path).read_bytes()
    starts = [i for i in (buf.find(b"MM\x00\x2a"), buf.find(b"II\x2a\x00")) if i >= 0]
    if not starts:
        raise CameraError(f"{Path(path).name}: no EXIF TIFF block")
    ex = Image.Exif()
    ex.load(b"Exif\x00\x00" + buf[min(starts):min(starts) + 131072])
    sub = ex.get_ifd(0x8769)
    out = dict(focal_mm=sub.get(0x920A), f35_mm=sub.get(0xA405), exif_width_px=sub.get(0xA002),
               exif_height_px=sub.get(0xA003), lens=sub.get(0xA434), model=ex.get(0x0110))
    if not out["f35_mm"] or not out["exif_width_px"] or not out["exif_height_px"]:
        raise CameraError(f"{Path(path).name}: EXIF lacks the 35 mm equivalent or pixel dimensions")
    return {k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "numerator") else v) for k, v in out.items()}


def focal_candidates(exif: dict[str, Any]) -> dict[str, float]:
    """``f_px`` on the two readings of the 35 mm equivalence (pixel scale of the EXIF frame; a crop keeps it)."""
    f35, w, h = float(exif["f35_mm"]), float(exif["exif_width_px"]), float(exif["exif_height_px"])
    return dict(width=f35 * w / FULL_FRAME_WIDTH_MM, diagonal=f35 * math.hypot(w, h) / FULL_FRAME_DIAGONAL_MM)


#: The stored photo is a crop of the EXIF frame and the crop offset is not recorded.
PRINCIPAL_POINT_SOURCE = "assumed_stored_image_centre (crop offset unknown)"
_CONTAINERS = {b"meta": 4, b"iprp": 0, b"ipco": 0}


def _boxes(buf: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", buf[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < hdr:
            return
        yield typ, i + hdr, i + size
        i += size


def heic_geometry(path: Path) -> tuple[tuple[int, int] | None, list[int]]:
    """(largest ``ispe`` width/height, ``irot`` values) read from the HEIC boxes, without decoding pixels."""
    buf = Path(path).read_bytes()
    sizes: list[tuple[int, int]] = []
    rots: list[int] = []

    def walk(start: int, end: int) -> None:
        for typ, body, stop in _boxes(buf, start, end):
            if typ == b"ispe":
                sizes.append(struct.unpack(">II", buf[body + 4:body + 12]))
            elif typ == b"irot":
                rots.append(buf[body] & 3)
            if typ in _CONTAINERS:
                walk(body + _CONTAINERS[typ], stop)

    walk(0, len(buf))
    return (max(sizes, key=lambda wh: wh[0] * wh[1]) if sizes else None), rots


def principal_point(capture: dict[str, str], root: Path) -> tuple[float, float]:
    """Centre of the stored photo (HEIC ``ispe``), the assumed principal point (:data:`PRINCIPAL_POINT_SOURCE`)."""
    size, _ = heic_geometry(Path(root) / capture["image_path"])
    if size is None:
        raise CameraError(f"{capture['image_path']}: no ispe box")
    return (size[0] / 2.0, size[1] / 2.0)


def intrinsics(f_px: float, pp: Sequence[float]) -> np.ndarray:
    return np.array([[f_px, 0.0, pp[0]], [0.0, f_px, pp[1]], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------------------
# Pose from the rim homography
# --------------------------------------------------------------------------------------


def _plane_to_image(h_img2plane: np.ndarray) -> np.ndarray:
    g = np.linalg.inv(h_img2plane)
    return g / np.linalg.norm(g[:, :2], axis=0).mean()


def pose_from_homography(h_img2plane: np.ndarray, k: np.ndarray) -> dict[str, Any]:
    """Pose of the camera in the rim frame. Returns R (rim -> camera), t (mm), the self-check residuals
    (``ratio - 1`` of |r1|/|r2| and the cosine between r1 and r2), the camera centre in the rim frame, its distance
    to the rim centre and the tilt of the optical axis from the rim normal."""
    m = np.linalg.inv(k) @ _plane_to_image(h_img2plane)
    n1, n2 = np.linalg.norm(m[:, 0]), np.linalg.norm(m[:, 1])
    lam = 2.0 / (n1 + n2)
    if m[2, 2] * lam < 0:  # the rim must lie in front of the camera
        lam = -lam
    r1, r2, t = m[:, 0] * lam, m[:, 1] * lam, m[:, 2] * lam
    r3 = np.cross(r1, r2)
    u, _, vt = np.linalg.svd(np.c_[r1, r2, r3])
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot = u @ np.diag([1, 1, -1]) @ vt
    centre = -rot.T @ t
    return dict(R=rot, t=t, ratio_residual=n1 / n2 - 1.0,
                orth_residual=float(r1 @ r2 / (np.linalg.norm(r1) * np.linalg.norm(r2))),
                centre=centre, distance=float(np.linalg.norm(centre)),
                tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(rot[2, 2]))))))


def focal_from_homography(h_img2plane: np.ndarray, pp: Sequence[float]) -> float | None:
    """Focal length (px) the rim implies, principal point fixed: least squares on 1/f^2 of the two constraints
    g1' w g2 = 0 and g1' w g1 = g2' w g2. None when the view is too frontal to determine it."""
    g = np.linalg.inv(h_img2plane)
    t = np.array([[1, 0, -pp[0]], [0, 1, -pp[1]], [0, 0, 1]])
    g = t @ g
    g1, g2 = g[:, 0], g[:, 1]
    a = np.array([g1[0] * g2[0] + g1[1] * g2[1], g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2])
    b = np.array([g1[2] * g2[2], g1[2] ** 2 - g2[2] ** 2])
    if float(a @ a) == 0:
        return None
    inv_f2 = -float(a @ b) / float(a @ a)
    return math.sqrt(1.0 / inv_f2) if inv_f2 > 0 else None


def project(rot: np.ndarray, t: np.ndarray, k: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    q = (rot @ np.atleast_2d(pts_mm).T).T + t
    p = (k @ q.T).T
    return p[:, :2] / p[:, 2:3]


def vertical_mm_per_px(rot: np.ndarray, t: np.ndarray, k: np.ndarray, z_mm: float = 0.0) -> float:
    """mm per px of a vertical step on the pot axis (0, 0, z) - the side-view height scale at that height."""
    a, b = project(rot, t, k, np.array([[0.0, 0.0, z_mm - 0.5], [0.0, 0.0, z_mm + 0.5]]))
    return 1.0 / float(np.linalg.norm(b - a))


def ruler_vertical_fit(rot: np.ndarray, t: np.ndarray, k: np.ndarray, ticks: Sequence[tuple[tuple[float, float], float]],
                       z0_mm: float | None = None) -> dict[str, float]:
    """Upright ruler (ASSUMED vertical) in the rim frame: unknowns X, Y (position), and ``k_ratio`` (mm in the pose
    per mm of reading), given the ANCHOR ``z0_mm`` = height of reading 0 relative to the rim plane. Without the anchor
    the problem is singular (depth-length ambiguity) and this raises. Linear in the unknowns."""
    if z0_mm is None:
        raise CameraError("upright ruler: depth-length ambiguity - one absolute anchor (height of reading 0 or the "
                          "ruler position) is needed to test the scale; none is decided")
    rows, rhs = [], []
    for (u, v), reading in ticks:
        for a in (np.array([-k[0, 0], 0.0, u - k[0, 2]]), np.array([0.0, -k[1, 1], v - k[1, 2]])):
            rows.append([a @ rot[:, 0], a @ rot[:, 1], (a @ rot[:, 2]) * reading])
            rhs.append(-(a @ t) - (a @ rot[:, 2]) * z0_mm)
    sol, res, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    x, y, kr = (float(v) for v in sol)
    return dict(x_mm=x, y_mm=y, z0_mm=z0_mm, k_ratio=kr)


def _line_through(points: np.ndarray) -> np.ndarray:
    m = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - m)
    n = vt[-1]
    return np.array([n[0], n[1], -n @ m])


def _side_samples(halves: dict[int, float], mirror: bool) -> dict[int, np.ndarray]:
    """Model points along each side on the reference plane (side 1 X=-h1, 2 Y=-h2, 3 X=+h3, 4 Y=+h4; ``mirror``
    flips Y for the other numbering direction)."""
    out = {}
    for k, h in halves.items():
        u = np.linspace(-0.8, 0.8, 7) * h
        xy = {1: np.c_[np.full(7, -h), u], 2: np.c_[u, np.full(7, -h)],
              3: np.c_[np.full(7, h), u], 4: np.c_[u, np.full(7, h)]}[k]
        if mirror:
            xy = xy * np.array([1.0, -1.0])
        out[k] = np.c_[xy, np.zeros(7)]
    return out


def pose_from_lines(lines: dict[int, np.ndarray], halves: dict[int, float], k: np.ndarray, centre_px: Sequence[float],
                    start: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Pose (R plane -> camera, t mm) from 2-4 image lines (homogeneous, unit normal) of the square's sides, focal
    FIXED: model points along each side must project onto its line. Both numbering directions and several starts
    (or the given ``start``); returns (R, t, RMS px)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    kinv = np.linalg.inv(k)

    def solve(samples: dict[int, np.ndarray], r0: np.ndarray, t0: np.ndarray):
        def residual(p):
            rot = Rotation.from_rotvec(p[:3]).as_matrix()
            res = []
            for side, pts in samples.items():
                q = (rot @ pts.T).T + p[3:]
                uv = (k @ q.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                res.append(uv @ lines[side][:2] + lines[side][2])
            return np.concatenate(res)
        return least_squares(residual, np.r_[Rotation.from_matrix(r0).as_rotvec(), t0], method="lm")

    best, best_mirror = None, False
    for mirror in (False, True):
        samples = _side_samples(halves, mirror)
        starts = []
        if start is not None:
            starts.append(start)
        else:
            ray = kinv @ np.array([centre_px[0], centre_px[1], 1.0])
            for az in (0.0, 90.0, 180.0, 270.0):
                for elev in (35.0, 55.0, 80.0):
                    a, e = np.radians(az), np.radians(elev)
                    c = 250.0 * np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)])
                    z = -c / np.linalg.norm(c)
                    up = np.array([0.0, 0.0, 1.0])
                    x = np.cross(up, z) if abs(z[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
                    x /= np.linalg.norm(x)
                    y = np.cross(z, x)
                    rot = np.vstack([x, y, z])
                    tz = (-rot @ c)[2]
                    starts.append((rot, ray / ray[2] * tz))
        for r0, t0 in starts:
            fit = solve(samples, r0, t0)
            if not np.isfinite(fit.cost) or fit.x[5] <= 0:
                continue
            if best is None or fit.cost < best.cost:
                best, best_mirror = fit, mirror
    if best is None:
        raise CameraError("no pose fits the rim lines")
    rot = Rotation.from_rotvec(best.x[:3]).as_matrix()
    # a mirrored fit only means the sides were numbered the other way round: same plane, Z still up, the rotation about
    # Z stays unknown anyway (no orientation mark). A planar target is also fitted by the camera mirrored below the
    # plane; turning the frame 180 deg about X (a proper rotation) puts the camera above it, Z up.
    if (-rot.T @ best.x[3:])[2] < 0:
        rot = rot @ np.diag([1.0, -1.0, -1.0])
    return rot, best.x[3:], float(np.sqrt(np.mean(best.fun ** 2)))


def to_rim_frame(rot: np.ndarray, t: np.ndarray, offset_mm: float) -> np.ndarray:
    """Translation of the pose expressed in the rim frame when it was solved on a plane ``offset_mm`` from the rim
    (rim point P_rim = P_plane + (0, 0, offset), Z pointing to the camera's side of the plane = up)."""
    up = 1.0 if (-rot.T @ t)[2] >= 0 else -1.0
    return t - rot @ np.array([0.0, 0.0, up * offset_mm])


def _pose_summary(rot: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    centre = -rot.T @ t
    return dict(R=rot, t=t, centre=centre, distance=float(np.linalg.norm(centre)),
                tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(rot[2, 2]))))))


# --------------------------------------------------------------------------------------
# Per-photo analysis with uncertainty
# --------------------------------------------------------------------------------------


@dataclass
class PoseResult:
    status: str  # validated | approximate | unavailable
    reason: str = ""
    f_basis: str = ""
    f_px: float | None = None
    pp: tuple[float, float] | None = None
    per_basis: dict[str, dict[str, Any]] = field(default_factory=dict)
    focal_from_rim_px: float | None = None
    focal_from_rim_sd_px: float | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    distance_mm: float | None = None
    distance_sd_mm: float | None = None
    tilt_deg: float | None = None
    vertical_mm_per_px: float | None = None
    vertical_mm_per_px_sd: float | None = None
    ruler: dict[str, float] | None = None
    cross_check: str = "none_single_source"
    scale_sources: int = 1
    reference: str = ""
    reference_plane: str = ""
    occluded_sides: tuple[int, ...] = ()


def analyse(points: Sequence[ps.RimPoint], spec: ps.PotSpec | None, *, focal: dict[str, float],
            pp: tuple[float, float], view: str, draws: int, seed: int, k_sigma: float,
            accept_rel: float, stop_rel: float) -> PoseResult:
    """Pose, self-check, focal implied by the rim, distance, and (side views) vertical scale + upright-ruler check.

    ``focal`` holds the candidate bases; a pose is ``validated`` under a basis when both self-check residuals are
    within ``k_sigma`` MC standard deviations. The basis used is the one that validates (both -> the one whose
    residual is smaller relative to its spread); neither -> ``approximate`` with the diagonal basis (CIPA convention)
    and the reason states that the crop offset / focal basis is unknown."""
    if spec is None or spec.shape != "square":
        return PoseResult("unavailable", "no square pot record for this plant")
    rim = [p for p in points if p.kind == "rim_edge"]
    if not rim:
        return PoseResult("unavailable", "no rim_edge points for this photo")
    if any(p.sigma_px is None for p in rim):
        return PoseResult("unavailable", "a rim point has no click_uncertainty_px")
    try:
        ref = ps.reference_of(points, spec, min_sides=3)
    except ps.PotRecordError as exc:
        return PoseResult("unavailable", str(exc))
    sides = {s: np.array([p.xy for p in rim if p.side == s]) for s in ref.halves}
    if len(sides) == 3:
        return _three_edge(sides, rim, ref, focal=focal, pp=pp, view=view, draws=draws, seed=seed)
    try:
        h, _, resid = ps.square_rim_homography(sides, spec.outer_edge_mm, ref.halves)
    except ps.PotRecordError as exc:
        return PoseResult("unavailable", str(exc))
    sigma = {s: max(max(p.sigma_px for p in rim if p.side == s), resid[s] or 0.0) for s in (1, 2, 3, 4)}
    rng = np.random.default_rng(seed)
    hs = []
    for _ in range(draws):
        try:
            hs.append(ps.square_rim_homography({s: v + rng.normal(0, sigma[s], v.shape) for s, v in sides.items()},
                                               spec.outer_edge_mm, ref.halves)[0])
        except ps.PotRecordError:
            continue
    if len(hs) < draws:
        return PoseResult("unavailable", "rim geometry unstable under the stated click uncertainty")
    ticks = [(p.xy, p.ruler_mm) for p in points if p.kind == "ruler_tick" and p.ruler_mm is not None]
    out = PoseResult("approximate", pp=pp, reference=ref.label, reference_plane=ref.plane)

    def on_rim(d: dict[str, Any]) -> dict[str, Any]:
        """the homography pose is on the reference plane; distances / tilt / vertical scale refer to the rim"""
        return dict(d, **_pose_summary(d["R"], to_rim_frame(d["R"], d["t"], ref.offset_mm)))

    for basis, f in focal.items():
        kk = intrinsics(f, pp)
        nominal = on_rim(pose_from_homography(h, kk))
        draws_pose = [on_rim(pose_from_homography(hk, kk)) for hk in hs]
        rr = np.array([d["ratio_residual"] for d in draws_pose])
        oo = np.array([d["orth_residual"] for d in draws_pose])
        dist = np.array([d["distance"] for d in draws_pose])
        ok = abs(nominal["ratio_residual"]) <= k_sigma * rr.std(ddof=1) and \
            abs(nominal["orth_residual"]) <= k_sigma * oo.std(ddof=1)
        entry = dict(f_px=f, pose=nominal, ratio_residual=nominal["ratio_residual"], ratio_sd=float(rr.std(ddof=1)),
                     orth_residual=nominal["orth_residual"], orth_sd=float(oo.std(ddof=1)), passes=bool(ok),
                     distance_sd=float(dist.std(ddof=1)),
                     score=abs(nominal["ratio_residual"]) / max(rr.std(ddof=1), 1e-12)
                     + abs(nominal["orth_residual"]) / max(oo.std(ddof=1), 1e-12))
        if view == "side":
            vs = np.array([vertical_mm_per_px(d["R"], d["t"], kk) for d in draws_pose])
            entry.update(vertical=vertical_mm_per_px(nominal["R"], nominal["t"], kk), vertical_sd=float(vs.std(ddof=1)))
        out.per_basis[basis] = entry
    fr = [focal_from_homography(hk, pp) for hk in hs]
    fr = [v for v in fr if v is not None and math.isfinite(v)]
    f0 = focal_from_homography(h, pp)
    if f0 is not None and len(fr) > draws // 2:
        out.focal_from_rim_px, out.focal_from_rim_sd_px = f0, float(np.std(fr, ddof=1))
    passing = [b for b, e in out.per_basis.items() if e["passes"]]
    if passing:
        out.f_basis = min(passing, key=lambda b: out.per_basis[b]["score"])
        out.status = "validated"
    else:
        out.f_basis = "diagonal" if "diagonal" in out.per_basis else next(iter(out.per_basis))
        out.reason = ("two-axis self-check fails on every focal basis (beyond "
                      f"{k_sigma:g} sigma of the click uncertainty): crop offset / focal basis unknown")
    e = out.per_basis[out.f_basis]
    out.f_px, out.R, out.t = e["f_px"], e["pose"]["R"], e["pose"]["t"]
    out.distance_mm, out.distance_sd_mm, out.tilt_deg = e["pose"]["distance"], e["distance_sd"], e["pose"]["tilt_deg"]
    if view == "side":
        vals = [b["vertical"] for b in out.per_basis.values()]
        out.vertical_mm_per_px = e["vertical"]
        # the focal basis is part of the uncertainty unless the self-check singled one out
        spread = (max(vals) - min(vals)) / (2 * math.sqrt(3)) if len(passing) != 1 else 0.0
        out.vertical_mm_per_px_sd = math.hypot(e["vertical_sd"], spread)
        if len(ticks) >= 2:
            # recorded for later; not testable without an anchor (module docstring)
            out.cross_check = "unavailable"
            out.reason = (out.reason + "; " if out.reason else "") + (
                f"{len(ticks)} upright-ruler ticks recorded but not testable from one photo (depth-length ambiguity; "
                "anchor undecided)")
    return out


def _three_edge(sides: dict[int, np.ndarray], rim: Sequence[ps.RimPoint], ref: ps.Reference, *,
                focal: dict[str, float], pp: tuple[float, float], view: str, draws: int, seed: int) -> PoseResult:
    """Three clicked sides + one recorded occluded: pose per focal basis with the focal fixed; always approximate."""
    lines = {s: _line_through(v) for s, v in sides.items()}
    resid = {s: (float(np.sqrt(np.sum((np.c_[v, np.ones(len(v))] @ lines[s]) ** 2) / (len(v) - 2)))
                 if len(v) > 2 else 0.0) for s, v in sides.items()}
    sigma = {s: max(max(p.sigma_px for p in rim if p.side == s), resid[s]) for s in sides}
    centre_px = np.vstack(list(sides.values())).mean(axis=0)
    occluded = ",".join(map(str, ref.occluded))
    out = PoseResult("approximate", pp=pp, reference=ref.label, reference_plane=ref.plane,
                     occluded_sides=ref.occluded,
                     reason=(f"three edges: side {occluded} occluded (recorded side_occluded, not a detection failure); "
                             "pose solved from three lines with the focal FIXED - no redundancy for the two-axis "
                             "self-check, so it cannot be validated"))
    rng = np.random.default_rng(seed)
    n_draws = max(draws // 4, 20)
    for basis, f in focal.items():
        kk = intrinsics(f, pp)
        try:
            rot, t, rms = pose_from_lines(lines, ref.halves, kk, centre_px)
        except CameraError as exc:
            return PoseResult("unavailable", f"three-edge pose: {exc}", occluded_sides=ref.occluded)
        nominal = _pose_summary(rot, to_rim_frame(rot, t, ref.offset_mm))
        dist, vert = [], []
        for _ in range(n_draws):
            jl = {s: _line_through(v + rng.normal(0, sigma[s], v.shape)) for s, v in sides.items()}
            try:
                r2, t2, _ = pose_from_lines(jl, ref.halves, kk, centre_px, start=(rot, t))
            except CameraError:
                continue
            d2 = _pose_summary(r2, to_rim_frame(r2, t2, ref.offset_mm))
            dist.append(d2["distance"])
            vert.append(vertical_mm_per_px(d2["R"], d2["t"], kk))
        entry = dict(f_px=f, pose=nominal, fit_rms_px=rms, passes=False, ratio_residual=None, ratio_sd=None,
                     orth_residual=None, orth_sd=None, distance_sd=float(np.std(dist, ddof=1)) if len(dist) > 2 else None)
        if view == "side":
            entry.update(vertical=vertical_mm_per_px(nominal["R"], nominal["t"], kk),
                         vertical_sd=float(np.std(vert, ddof=1)) if len(vert) > 2 else None)
        out.per_basis[basis] = entry
    out.f_basis = "diagonal" if "diagonal" in out.per_basis else next(iter(out.per_basis))
    e = out.per_basis[out.f_basis]
    out.f_px, out.R, out.t = e["f_px"], e["pose"]["R"], e["pose"]["t"]
    out.distance_mm, out.distance_sd_mm, out.tilt_deg = e["pose"]["distance"], e["distance_sd"], e["pose"]["tilt_deg"]
    if view == "side":
        out.vertical_mm_per_px, out.vertical_mm_per_px_sd = e["vertical"], e["vertical_sd"]
    return out
