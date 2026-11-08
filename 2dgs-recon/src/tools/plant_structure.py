"""Whole-plant structure from human labels and measured poses (pipeline stage S3).

    python -m tools.plant_structure --session build/new_captures_20260920/dwt_01/edit_session \\
        --backup build/dwt_01_02_3d_20260922/dwt_01/labels/seed_with_imports.backup.json \\
        [--growth <growth.json>] --out build/pipeline_<date>/dwt_01/structure

Produces, in the board frame (millimetres), from the photographs that have a solved ChArUco pose and labels:

* a visual hull: voxels whose projection falls inside the green pixels of the labelled plant region in every
  such photograph (labels bound the search; excess-green selects plant pixels; masks are dilated to absorb pose
  error). It is an occupancy envelope for orientation and grouping, not a leaf surface or a tissue volume;
* compound-leaf regions: each photo's compound boxes voted onto the hull and matched across photos;
* the main-stem axis: a 3D line fitted to the centreline rays of the main-stem polygons (>= 2 photos), or, when no
  stretch is seen by two photos, the lower trunk traced through the hull voxels under those polygons (method
  'hull_trunk'; its voxels are taken out of the compound regions before naming);
* petioles: a 3D line per compound region fitted to its sub-stem polygons (>= 2 photos), with the extent seen by
  two photos kept apart from the extent seen by one.

Everything is diagnostic (`validated_metric=false`, `suitable_for_ppfd=false`): the poses fail the 1 px gate.
Nothing here reads nominal camera angles, crops images or creates identities. Photos without a solved pose or
without labels are listed with the reason they were not used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import least_squares, linear_sum_assignment

from tools.leaf_light.common import StudyError, digest, file_hash, read_json, write_json

SCHEMA = 'plant-structure-1'
#: Excess-green (2g-r-b)/(r+g+b) above which a pixel inside the labelled region counts as plant.
EXG_MIN = 0.06
#: Minimum green channel value for a plant pixel (drops dark noise).
GREEN_MIN = 40
#: Silhouette dilation in pixels before carving, absorbing pose error (measured poses are 3-9 px off).
DILATION_PX = 30
#: Fractional margin added around the union of a photo's label boxes.
LABEL_MARGIN = 0.05
#: Coarse and fine voxel sizes (mm) and the coarse search box in the board frame (mm).
COARSE_MM, FINE_MM = 4.0, 1.0
SEARCH_MM = ((-300.0, 600.0), (-400.0, 300.0), (0.0, 300.0))
#: A compound box of another photo is matched to a reference box when their hull voxel IoU is at least this.
MIN_BOX_IOU = 0.2
#: Line fits: centreline samples every this many pixels; consensus extent needs rays within this distance (mm)
#: of the line and at least this angle (deg) to it, from at least two photos.
SAMPLE_STEP_PX = 8
CONSENSUS_MM, CONSENSUS_ANGLE_DEG = 2.0, 30.0
#: A sub-stem polygon joins the compound region holding at least this share of its hull votes.
PETIOLE_MIN_SHARE = 0.6
#: A compound region smaller than this share of the hull (or this many voxels) is a fragment: kept, never named.
MIN_REGION_SHARE, MIN_REGION_VOXELS = 0.01, 200
#: Trunk fallback when the ray fit finds no stretch seen by two photos (2026-09-28, dwt_04: photo 26 labels the lower
#: stem, photo 28 an upper piece, so the intervals never overlap). Hull voxels that project inside a main-stem polygon
#: are traced upward from the lowest slice: each slice keeps the voxels within TRUNK_RADIUS_MM of the previous slice
#: centre, and the trace stops where a slice is empty or holds more than TRUNK_WIDEN x the running median (the canopy).
TRUNK_RADIUS_MM, TRUNK_SLICE_MM, TRUNK_WIDEN, TRUNK_MIN_SLICES = 12.0, 2.0, 3.0, 5


class Cameras:
    def __init__(self, capture: dict):
        self.K = np.array(capture['intrinsics']['camera_matrix'], float)
        self.D = np.array(capture['intrinsics']['dist_coeffs'], float)
        self.size = (capture['intrinsics']['image_width_px'], capture['intrinsics']['image_height_px'])
        self.frames = {f['image_file'].lower(): f for f in capture['frames']}

    #: board poses S3 may carve with: ChArUco corners, or ArUco marker corners (tools.marker_pose_rescue, 2026-10-01;
    #: same board, same quality class). Feature/landmark poses ('feature_diagnostic', ...) stay excluded.
    BOARD_POSE_STATUSES = ('solved', 'solved_marker_corners')

    def posed(self, key):
        f = self.frames.get(key)
        return bool(f and f.get('pose_status') in self.BOARD_POSE_STATUSES and f.get('rvec_rad') is not None)

    def rt(self, key):
        f = self.frames[key]
        return np.array(f['rvec_rad'], float), np.array(f['tvec_mm'], float)

    def project(self, key, X):
        r, t = self.rt(key)
        points = np.ascontiguousarray(np.asarray(X, float).reshape(-1, 3))
        return cv2.projectPoints(points, r, t, self.K, self.D)[0].reshape(-1, 2)

    def depth(self, key, X):
        r, t = self.rt(key)
        return (np.asarray(X, float) @ cv2.Rodrigues(r)[0].T + t)[:, 2]

    def centre(self, key):
        r, t = self.rt(key)
        R = cv2.Rodrigues(r)[0]
        return -R.T @ t

    def ray(self, key, xy):
        r, t = self.rt(key)
        R = cv2.Rodrigues(r)[0]
        n = cv2.undistortPoints(np.array([[xy]], float), self.K, self.D).reshape(2)
        d = R.T @ np.array([n[0], n[1], 1.0])
        return -R.T @ t, d / np.linalg.norm(d)


def plain(o):
    """numpy scalars/arrays to JSON types."""
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, 'item') and np.ndim(x) == 0 else np.asarray(x).tolist()))


def box_xyxy(det):
    b = det['bbox']
    return b['x_min_px'], b['y_min_px'], b['x_max_px'], b['y_max_px']


def plant_mask(image: np.ndarray, boxes: list) -> np.ndarray:
    """Plant pixels: excess green inside the union of this photo's label boxes (plus a margin)."""
    h, w = image.shape[:2]
    region = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        mx, my = (x1 - x0) * LABEL_MARGIN, (y1 - y0) * LABEL_MARGIN
        region[int(max(y0 - my, 0)):int(min(y1 + my, h)), int(max(x0 - mx, 0)):int(min(x1 + mx, w))] = True
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    m = region & (exg > EXG_MIN) & (g > GREEN_MIN)
    m = ndimage.binary_opening(m, iterations=2)
    return ndimage.binary_closing(m, iterations=3)


def inside(cams, key, X, mask):
    q = cams.project(key, X)
    # points behind the camera also project to pixels; they must never count as seen
    front = (cams.depth(key, X) > 0) & np.isfinite(q).all(1)
    p = np.zeros((len(X), 2), int)
    p[front] = np.round(np.clip(q[front], -1e7, 1e7)).astype(int)  # near the camera plane projections overflow
    ok = front & (p[:, 0] >= 0) & (p[:, 0] < mask.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < mask.shape[0])
    hit = np.zeros(len(X), bool)
    hit[ok] = mask[p[ok, 1], p[ok, 0]]
    return hit


def grid(lo, hi, step):
    axes = [np.arange(lo[i] + step / 2, hi[i], step) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)


def carve(cams, views, masks):
    dil = {v: ndimage.binary_dilation(masks[v], iterations=DILATION_PX) for v in views}
    G = grid([a for a, _ in SEARCH_MM], [b for _, b in SEARCH_MM], COARSE_MM)
    keep = np.ones(len(G), bool)
    for v in views:
        keep[keep] = inside(cams, v, G[keep], dil[v])
    if not keep.any():
        raise StudyError('Visual hull is empty: the labelled silhouettes do not intersect')
    lo, hi = G[keep].min(0) - 2 * COARSE_MM, G[keep].max(0) + 2 * COARSE_MM
    F = grid(lo, hi, FINE_MM)
    keep = np.ones(len(F), bool)
    for v in views:
        keep[keep] = inside(cams, v, F[keep], dil[v])
    return F[keep]


#: A carving photo must have this share of its plant silhouette explained by the hull, else it is dropped (see build).
MIN_EXPLAINED_PCT, MIN_CARVE_VIEWS = 80.0, 2


def silhouette_explained(cams, views, masks, H) -> dict:
    """Per photo: % of its plant silhouette covered by the projected hull (6 px dilation)."""
    explained = {}
    for v in views:
        p = np.round(cams.project(v, H)).astype(int)
        cov = np.zeros_like(masks[v])
        ok = (p[:, 0] >= 0) & (p[:, 0] < cov.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < cov.shape[0])
        cov[p[ok, 1], p[ok, 0]] = True
        cov = ndimage.binary_dilation(cov, iterations=6)
        explained[v] = round(float((masks[v] & cov).sum() / max(masks[v].sum(), 1) * 100), 1)
    return explained


def main_component(H: np.ndarray, step: float = FINE_MM) -> tuple[np.ndarray, int]:
    """Keep the largest 26-connected piece of the hull (2026-09-25). With few posed photos their viewing cones also
    meet in small blobs far from the plant (dwt_03: 2,649 voxels 15-20 cm away, coloured as leaves); a plant is one
    connected body, so everything detached from the main piece is dropped and counted."""
    if not len(H):
        return H, 0
    ijk = np.round((H - H.min(0)) / step).astype(int)
    grid = np.zeros(ijk.max(0) + 1, bool)
    grid[tuple(ijk.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return H, 0
    ids = lab[tuple(ijk.T)]
    keep = ids == np.bincount(ids).argmax()
    return H[keep], int((~keep).sum())


def compound_regions(cams, views, dets, H, view_grouping: dict | None = None):
    """Vote each photo's compound boxes onto the hull, then match boxes across photos by voxel IoU."""
    per_view = {}
    for v in views:
        boxes = [box_xyxy(d) for d in dets.get(v, []) if d['class_name'] == 'compoundleaves']
        if not boxes:
            continue
        p = cams.project(v, H) + 0.5  # pixel centres to the pixel-edge convention of the labels
        area = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        owner = np.full(len(H), -1)
        best = np.full(len(H), np.inf)
        for j, (x0, y0, x1, y1) in enumerate(boxes):
            inb = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1) & (area[j] < best)
            owner[inb], best[inb] = j, area[j]
        per_view[v] = (boxes, owner)
    if not per_view:
        return np.full(len(H), -1), [], {}
    # the reference (and any new region) comes from a side photo: from above, overlapping upper leaves split into
    # several boxes (dwt_01 auto run 09-30: top view 03 had 5 boxes where side photos had 4, and the upper L03 was cut
    # into 3 regions). Top-view boxes may join a region but never start one. People never boxed compounds from above.
    from tools.leaf_area_geometry import view_geometry
    geometry = {v: view_geometry(cams, v, view_grouping) for v in per_view}
    side = [v for v in per_view if geometry[v]['group'] == 'ring']
    if not side:
        return np.full(len(H), -1), [], dict(
            reference_view=None, view_geometry=geometry,
            status='unavailable_no_measured_ring_reference',
            reference_rule='measured ring required; image numbers never determine viewing direction')
    ref = max(side, key=lambda v: len(per_view[v][0]))
    clusters = [[(ref, j)] for j in range(len(per_view[ref][0]))]
    dropped_top = []
    rep = [per_view[ref][1] == j for j in range(len(per_view[ref][0]))]  # voxel set representing each cluster
    matches = {}
    for v, (boxes, owner) in per_view.items():
        if v == ref:
            continue
        cand = [owner == j for j in range(len(boxes))]
        iou = np.array([[(a & rep[c]).sum() / max((a | rep[c]).sum(), 1) for c in range(len(clusters))] for a in cand]) \
            if clusters else np.zeros((len(boxes), 0))
        r, c = linear_sum_assignment(-iou) if iou.size else ([], [])
        taken = set()
        for j, k in zip(r, c):
            if iou[j, k] >= MIN_BOX_IOU:
                clusters[k].append((v, j)); taken.add(j); matches[(v, j)] = round(float(iou[j, k]), 3)
        for j in range(len(boxes)):
            if j not in taken:
                if v not in side:
                    dropped_top.append(dict(view=v, box_index=j, best_iou=round(float(iou[j].max()), 3) if iou.size else None))
                    continue
                clusters.append([(v, j)])
                rep.append(cand[j])
    votes = np.zeros((len(H), len(clusters)), int)
    for k, members in enumerate(clusters):
        for v, j in members:
            votes[:, k] += per_view[v][1] == j
    label = votes.argmax(1)
    top = votes.max(1)
    second = np.sort(votes, 1)[:, -2] if votes.shape[1] > 1 else np.zeros(len(H))
    label[(top == 0) | (top == second)] = -1
    info = [dict(region=k, boxes=[dict(view=v, box_index=j, iou_to_reference=matches.get((v, j))) for v, j in m],
                 photos=len({v for v, _ in m})) for k, m in enumerate(clusters)]
    return label, info, dict(reference_view=ref, unmatched_top_view_boxes=dropped_top,
                             view_geometry=geometry, status='diagnostic_grouping',
                             reference_rule='measured ring with most compound boxes; other views only join regions')


def _is_top(view_key: str) -> bool:
    """Legacy filename convention only; never used for current 3D region assignment."""
    code = Path(view_key).stem.rsplit('_', 1)[-1]
    return code.isdigit() and int(code) < 10


def centreline(cams, ring):
    m = np.zeros((cams.size[1], cams.size[0]), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(ring)).astype(np.int32)], 1)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None, None
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    t = (pts - c) @ vt[0]
    width = np.median(np.abs((pts - c) @ vt[1])) * 4  # ~full width for a uniform strip
    samples = [c + s * vt[0] for s in np.arange(t.min() + 4, t.max() - 4, SAMPLE_STEP_PX)]
    return samples, float(width)


def closest(P0, u, C, d):
    w0 = C - P0
    a, b, c_, dd, e = d @ d, d @ u, u @ u, d @ w0, u @ w0
    den = a * c_ - b * b
    s, t = (b * e - c_ * dd) / den, (a * e - b * dd) / den
    return C + s * d, P0 + t * u, t


def fit_line(rays, P0, u0):
    def unpack(x):
        return x[:3], np.array([np.cos(x[4]) * np.cos(x[3]), np.cos(x[4]) * np.sin(x[3]), np.sin(x[4])])

    def res(x):
        P, u = unpack(x)
        return np.array([np.linalg.norm(np.subtract(*closest(P, u, C, d)[:2])) for _, C, d in rays])
    x0 = np.concatenate([P0, [np.arctan2(u0[1], u0[0]), np.arcsin(np.clip(u0[2], -1, 1))]])
    fit = least_squares(res, x0, loss='soft_l1', f_scale=1.0)
    P, u = unpack(fit.x)
    return P, u, res(fit.x)


def consensus(rays, P, u):
    """Line parameter intervals per photo from rays close to and not parallel to the line."""
    iv = {}
    for v, C, d in rays:
        pr, pl, t = closest(P, u, C, d)
        ang = np.degrees(np.arccos(min(1.0, abs(d @ u))))
        if np.linalg.norm(pr - pl) <= CONSENSUS_MM and ang >= CONSENSUS_ANGLE_DEG:
            iv.setdefault(v, []).append(t)
    iv = {v: (min(ts), max(ts)) for v, ts in iv.items()}
    if not iv:
        return iv, None, None
    grid_t = np.arange(min(a for a, _ in iv.values()), max(b for _, b in iv.values()) + 0.5, 0.5)
    count = np.array([sum(a <= g <= b for a, b in iv.values()) for g in grid_t])
    two = grid_t[count >= 2]
    return iv, (float(two.min()), float(two.max())) if len(two) else None, (float(grid_t.min()), float(grid_t.max()))


def fit_stem(cams, views, dets, H):
    rays, widths = [], []
    for v in views:
        for d in dets.get(v, []):
            if d['class_name'] != 'main-stem' or not d.get('rings'):
                continue
            samples, width = centreline(cams, d['rings'][0])
            if not samples:
                continue
            for s in samples:
                rays.append((v,) + cams.ray(v, tuple(np.asarray(s) - 0.5)))
            widths.append((v, width))
    used = sorted({r[0] for r in rays})
    if len(used) < 2:
        return dict(status='unavailable', reason='main-stem polygons in fewer than two posed photos', views=used)
    c = H.mean(0)
    P, u, r = fit_line(rays, np.array([c[0], c[1], H[:, 2].min() + 20]), np.array([0, 0, 1.0]))
    if u[2] < 0:
        u = -u
    iv, two, anyv = consensus(rays, P, u)
    if two is None:
        return dict(status='unavailable', reason='no stretch of the axis is seen by two photos at a usable angle', views=used)
    a, b = P + two[0] * u, P + two[1] * u
    diam = []
    for v, w in widths:
        mid = (a + b) / 2
        diam.append(w * cams.depth(v, mid[None])[0] / cams.K[0, 0])
    return dict(status='fitted', method='ray_fit', views=used, rays=len(rays), axis=[(a + (b - a) * s).round(2).tolist() for s in np.linspace(0, 1, 12)],
                z_mm=[round(float(min(a[2], b[2])), 1), round(float(max(a[2], b[2])), 1)],
                tilt_deg=round(float(np.degrees(np.arccos(abs(u[2])))), 1),
                diameter_mm=round(float(np.median(diam)), 1) if diam else None,
                ray_line_distance_mm=dict(median=round(float(np.median(r)), 2), p90=round(float(np.percentile(r, 90)), 2)),
                residual_is_a_check=len(used) >= 3,
                residual_note=None if len(used) >= 3 else 'two photos: a line meets both ray fans by construction, so ~0 residual is not a check',
                intervals={v: [round(x, 1) for x in t] for v, t in iv.items()})


def stem_from_hull(cams, views, dets, H):
    """Trunk axis from the hull itself (fallback for fit_stem). Returns (stem dict, trunk voxel mask).

    The hull is carved by every posed photo, so the voxels under a main-stem polygon of even one photo already sit
    where all photos allow; tracing them upward from the bottom keeps the trunk column and stops at the canopy,
    where a polygon's viewing column also catches leaves. Slice centres are fitted by one straight line."""
    sel = np.zeros(len(H), bool)
    used = []
    for v in views:
        rings = [d['rings'][0] for d in dets.get(v, []) if d['class_name'] == 'main-stem' and d.get('rings')]
        if not rings:
            continue
        m = np.zeros((cams.size[1], cams.size[0]), np.uint8)
        for r in rings:
            cv2.fillPoly(m, [np.round(np.array(r)).astype(np.int32)], 1)
        sel |= inside(cams, v, H, m.astype(bool))
        used.append(v)
    trunk = np.zeros(len(H), bool)
    if not sel.any():
        return dict(status='unavailable', reason='no hull voxel lies under a main-stem polygon', views=used), trunk
    idx = np.flatnonzero(sel)
    z = H[idx, 2]
    z0 = z.min()
    c = np.median(H[idx[z < z0 + TRUNK_SLICE_MM], :2], 0)
    cen, counts, stop = [], [], 'reached the top of the selection'
    while z0 <= z.max():
        s = idx[(z >= z0) & (z < z0 + TRUNK_SLICE_MM)]
        s = s[np.hypot(*(H[s, :2] - c).T) <= TRUNK_RADIUS_MM]
        if len(s) < 3:
            stop = f'no voxels near the axis at z {z0:.1f} mm'
            break
        if len(counts) >= 3 and len(s) > TRUNK_WIDEN * np.median(counts):
            stop = f'slice widens into the canopy at z {z0:.1f} mm'
            break
        c = np.median(H[s, :2], 0)
        cen.append([c[0], c[1], z0 + TRUNK_SLICE_MM / 2])
        counts.append(len(s))
        trunk[s] = True
        z0 += TRUNK_SLICE_MM
    if len(cen) < TRUNK_MIN_SLICES:
        return dict(status='unavailable', reason=f'trunk trace too short ({len(cen)} slices; {stop})', views=used), \
            np.zeros(len(H), bool)
    cen = np.array(cen)
    A = np.column_stack([cen[:, 2], np.ones(len(cen))])
    kx, ky = np.linalg.lstsq(A, cen[:, 0], rcond=None)[0], np.linalg.lstsq(A, cen[:, 1], rcond=None)[0]
    zz = np.linspace(cen[0, 2], cen[-1, 2], 12)
    axis = np.column_stack([kx[0] * zz + kx[1], ky[0] * zz + ky[1], zz])
    res = np.hypot(cen[:, 0] - (kx[0] * cen[:, 2] + kx[1]), cen[:, 1] - (ky[0] * cen[:, 2] + ky[1]))
    return dict(status='fitted', method='hull_trunk', views=used, axis=axis.round(2).tolist(),
                z_mm=[round(float(zz[0]), 1), round(float(zz[-1]), 1)],
                tilt_deg=round(float(np.degrees(np.arctan(np.hypot(kx[0], ky[0])))), 1),
                diameter_mm=None, ray_line_distance_mm=None,
                slice_centre_distance_mm=dict(median=round(float(np.median(res)), 2), p90=round(float(np.percentile(res, 90)), 2)),
                slices=len(cen), trunk_voxels=int(trunk.sum()), trace_stop=stop,
                residual_is_a_check=False,
                note='lower trunk only, from hull voxels under the main-stem polygons (the dilated hull is wider than '
                     'the stem, so no diameter); the upper stem inside the canopy is not traced'), trunk


def fit_petioles(cams, views, dets, H, label, stem):
    if stem.get('status') != 'fitted':
        return [], 'stem axis unavailable'
    ax = np.array(stem['axis'])
    sP, sU = ax[0], (ax[-1] - ax[0]) / np.linalg.norm(ax[-1] - ax[0])
    to_stem = lambda X: float(np.linalg.norm((X - sP) - ((X - sP) @ sU) * sU))
    polys = []
    for v in views:
        for d in dets.get(v, []):
            if d['class_name'] != 'sub-stem' or not d.get('rings'):
                continue
            samples, _ = centreline(cams, d['rings'][0])
            if not samples:
                continue
            m = np.zeros((cams.size[1], cams.size[0]), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(d['rings'][0])).astype(np.int32)], 1)
            hit = inside(cams, v, H, ndimage.binary_dilation(m.astype(bool), iterations=10))
            votes = np.bincount(label[hit & (label >= 0)], minlength=label.max() + 1) if label.max() >= 0 else np.zeros(0)
            if not votes.size or votes.sum() == 0:
                continue
            k = int(votes.argmax())
            polys.append(dict(view=v, k=k, share=float(votes.max() / votes.sum()), samples=samples, annotation_id=d['annotation_id']))
    out = []
    for k in sorted({p['k'] for p in polys}):
        P_ = [p for p in polys if p['k'] == k and p['share'] >= PETIOLE_MIN_SHARE]
        if len({p['view'] for p in P_}) < 2:
            out.append(dict(region=k, status='unavailable', reason='sub-stem polygons in fewer than two posed photos',
                            views=sorted({p['view'] for p in P_})))
            continue
        rays = [(p['view'],) + cams.ray(p['view'], tuple(np.asarray(s) - 0.5)) for p in P_ for s in p['samples']]
        cent = H[label == k].mean(0)
        S0 = sP + ((cent - sP) @ sU) * sU
        P, u, r = fit_line(rays, (S0 + cent) / 2, (cent - S0) / np.linalg.norm(cent - S0))
        ts = np.linspace(-80, 80, 1601)
        A0 = P + ts[int(np.argmin([to_stem(P + x * u) for x in ts]))] * u
        iv, two, anyv = consensus(rays, A0, u)
        if not iv:
            out.append(dict(region=k, status='unavailable', reason='no usable extent rays'))
            continue
        if np.median([(a + b) / 2 for a, b in iv.values()]) < 0:
            u = -u
            iv, two, anyv = consensus(rays, A0, u)
        near, far = A0 + max(0.0, anyv[0]) * u, A0 + anyv[1] * u
        out.append(dict(region=k, status='fitted', views=sorted(iv), polygons=len(P_),
                        near_end=near.round(2).tolist(), far_end=far.round(2).tolist(),
                        near_end_to_stem_mm=round(to_stem(A0), 1),
                        length_two_view_mm=round(two[1] - max(two[0], 0.0), 1) if two else None,
                        extent_any_view_mm=round(anyv[1] - max(anyv[0], 0.0), 1),
                        elevation_deg=round(float(np.degrees(np.arcsin(abs(u[2])))), 1),
                        ray_line_distance_mm=dict(median=round(float(np.median(r)), 2), p90=round(float(np.percentile(r, 90)), 2))))
    return out, None


#: A naming decision follows its region only while the region's median height stays within this distance.
NAME_DECISION_Z_MM = 3.0


def name_regions(H, label, info, growth: dict | None):
    """Height order proposal, bottom up: L01 is the lowest compound leaf and numbers rise toward the stem top, so
    the unfolding leaf (listed last in the growth record) is the highest (user rule, 2026-09-24; it replaces the
    top-down rule of 2026-09-22). When there is exactly one region more than the record lists, the smallest is
    proposed as an unrecorded growing leaf; when an unfolding leaf has no region of its own, only the unfolded
    leaves are named."""
    regions = []
    for r in info:
        sel = label == r['region']
        if not sel.any():
            continue
        regions.append(dict(region=r['region'], voxels=int(sel.sum()), median_z_mm=round(float(np.median(H[sel, 2])), 1),
                            centroid_mm=H[sel].mean(0).round(1).tolist(), photos=r['photos'], boxes=r['boxes']))
    entries = [e for e in (growth or {}).get('compound_leaves', []) if e.get('compound_leaf_id')]
    ids = sorted(e['compound_leaf_id'] for e in entries)
    unfolding = sorted(e['compound_leaf_id'] for e in entries if e.get('is_unfolding') is True)
    mature = [i for i in ids if i not in unfolding]
    n = len(ids)
    floor = max(MIN_REGION_VOXELS, MIN_REGION_SHARE * len(H))
    for e in regions:
        e['fragment'] = e['voxels'] < floor
    fragments = [e for e in regions if e['fragment']]
    regions_all, regions = regions, [e for e in regions if not e['fragment']]
    rest = regions
    note = 'no growth record with compound leaves: regions are unnamed'
    if n and len(regions) in (n, n + 1) or (unfolding and len(regions) == len(mature)):
        if len(regions) == n + 1:
            small = min(regions, key=lambda e: e['voxels'])
            small['proposed_name'] = 'growing_leaf'
            rest = [e for e in regions if e is not small]
        names = ids if len(rest) == n else mature
        for e, name in zip(sorted(rest, key=lambda e: e['median_z_mm']), names):
            e['proposed_name'] = name
        note = 'names are a height-order proposal (L01 = lowest, numbers rise upward); confirm against the photographs'
        if len(rest) != n:
            note += f'; unfolding {", ".join(unfolding)} has no region of its own'
        top = max(rest, key=lambda e: e['median_z_mm']) if rest else None
        if unfolding and len(rest) == n and top and top['proposed_name'] not in unfolding:
            note += '; WARNING the highest region is not the recorded unfolding leaf'
    elif n:
        note = f'{len(regions)} regions for {n} recorded compound leaves: not named'
    for e in regions_all:
        e.setdefault('proposed_name', None)
    if fragments:
        note += f'; {len(fragments)} fragment region(s) under {int(floor)} voxels left unnamed'
    return regions_all, note


#: Region colours shared by the overlays and the naming page (region k -> PALETTE[k % 6]).
PALETTE = [(224, 122, 95), (61, 145, 220), (156, 110, 214), (70, 170, 120), (230, 190, 60), (200, 90, 160)]


def draw_overlay(cams, key, image, H, label, stem, petioles, out, names: dict | None = None):
    crop_pts = cams.project(key, H)
    lo, hi = crop_pts.min(0) - 150, crop_pts.max(0) + 150
    box = (int(max(lo[0], 0)), int(max(lo[1], 0)), int(min(hi[0], cams.size[0])), int(min(hi[1], cams.size[1])))
    im = Image.fromarray(image).crop(box).convert('RGBA')
    ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    pal = PALETTE
    step = max(1, len(H) // 6000)
    for (x, y), k in zip(crop_pts[::step] - box[:2], label[::step]):
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(pal[k % len(pal)] if k >= 0 else (150, 150, 150)) + (90,))
    if stem.get('status') == 'fitted':
        dr.line([tuple(p) for p in cams.project(key, np.array(stem['axis'])) - box[:2]], fill=(245, 166, 35, 255), width=10)
    for p in petioles:
        if p.get('status') == 'fitted':
            q = cams.project(key, np.array([p['near_end'], p['far_end']])) - box[:2]
            dr.line([tuple(q[0]), tuple(q[1])], fill=pal[p['region'] % len(pal)] + (255,), width=7)
    im = Image.alpha_composite(im, ov).convert('RGB')
    # the proposed name at each region's middle, plus a legend (2026-09-29: colour dots alone could not be checked)
    from PIL import ImageFont
    size = max(28, im.size[0] // 28)
    font = ImageFont.load_default(size=size)
    dr = ImageDraw.Draw(im)
    legend = []
    for k in sorted(set(label[label >= 0].tolist())):
        sel = label == k
        if sel.sum() < 50:
            continue
        name = (names or {}).get(int(k)) or f'region {k}'
        cx, cy = np.median(crop_pts[sel] - box[:2], 0)
        if 0 <= cx < im.size[0] and 0 <= cy < im.size[1]:
            tb = dr.textbbox((cx, cy), name, font=font, anchor='mm')
            dr.rectangle([tb[0] - 6, tb[1] - 4, tb[2] + 6, tb[3] + 4], fill=(255, 255, 255), outline=pal[k % len(pal)], width=4)
            dr.text((cx, cy), name, fill=(20, 20, 20), font=font, anchor='mm')
        legend.append((k, name))
    y = 12
    for k, name in legend:
        dr.rectangle([12, y, 12 + size, y + size], fill=pal[k % len(pal)])
        dr.text((20 + size, y), f'{name}  (region {k})', fill=(255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0))
        y += size + 10
    im.thumbnail((1200, 1200))
    im.save(out, quality=88)


def apply_name_decisions(regions: list, decisions: dict | None) -> list:
    """Apply a person's naming decisions from the review page (2026-09-24).

    A decision is keyed by region index and carries the height it was made at; it applies only while the region
    still sits within NAME_DECISION_Z_MM of that height, so a recomputed hull with renumbered regions never
    inherits a stale name. 'confirm' keeps the proposal, any other value replaces it."""
    applied = []
    for e in regions:
        d = (decisions or {}).get(str(e['region']))
        if not d:
            continue
        if abs(float(d.get('median_z_mm', 1e9)) - e['median_z_mm']) > NAME_DECISION_Z_MM:
            e['name_decision'] = dict(status='stale', decided=d)
            continue
        if d.get('name') != 'confirm':
            e['proposed_name'] = d.get('name') or None
        e['name_source'] = 'human'
        applied.append(e['region'])
    return applied


def reviewed_region_names(decisions: dict | None, assignment_hash: str) -> tuple[dict | None, str]:
    """A previous region number cannot carry human identity into a new partition."""
    if not decisions:
        return None, 'not_supplied'
    if decisions.get('region_assignment_hash') != assignment_hash:
        return None, 'unavailable_assignment_changed_or_unbound'
    if not isinstance(decisions.get('regions'), dict):
        raise StudyError('Hash-bound region names require a regions mapping')
    return decisions['regions'], 'bound_to_current_assignment'


def _plain(value):
    """numpy scalars/arrays -> plain Python, recursively; other values unchanged."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def region_assignment_digest(grouping, points, labels, info) -> str:
    # region info carries numpy scalars on real runs (int64 broke json); plain values hash like pure-Python input
    return digest(_plain(dict(grouping=grouping, points_mm=np.asarray(points).tolist(),
                              labels=np.asarray(labels).tolist(), regions=info)))


def build(session: Path, backup_path: Path, out: Path, growth_path: Path | None = None,
          names_path: Path | None = None, view_grouping: dict | None = None) -> dict:
    session, out = Path(session), Path(out)
    if out.exists():
        raise StudyError(f'Output exists; choose a new directory: {out}')
    capture = read_json(session / 'capture.json')
    backup = read_json(backup_path)
    dets = backup['variants']['__all__'].get('membership', {}).get('detections', {})
    cams = Cameras(capture)
    views, skipped = [], []
    for key in sorted(cams.frames):
        f = cams.frames[key]
        labelled = [d for d in dets.get(key, []) if d.get('bbox')]
        if not cams.posed(key):
            skipped.append(dict(photo=key, reason=f"no solved pose ({f.get('pose_status')})"))
        elif not labelled:
            skipped.append(dict(photo=key, reason='posed but has no labels to bound the plant region'))
        else:
            views.append(key)
    if len(views) < 2:
        raise StudyError(f'Need at least two posed, labelled photos; have {views}')
    images = {v: np.asarray(Image.open(session / cams.frames[v]['image_path']).convert('RGB')) for v in views}
    masks = {v: plant_mask(images[v], [box_xyxy(d) for d in dets[v] if d.get('bbox')]) for v in views}
    H, detached = main_component(carve(cams, views, masks))
    # photos whose silhouette the hull cannot explain disagree with the others (pose error or a bad label): drop the
    # worst one and carve again (2026-09-29; human-labelled runs explain >= 94.9 % everywhere and are unchanged)
    dropped_views = []
    explained = silhouette_explained(cams, views, masks, H)
    while len(views) > MIN_CARVE_VIEWS and min(explained.values()) < MIN_EXPLAINED_PCT:
        worst = min(explained, key=explained.get)
        dropped_views.append(dict(photo=worst, explained_pct=explained[worst]))
        views = [v for v in views if v != worst]
        H, detached = main_component(carve(cams, views, masks))
        explained = silhouette_explained(cams, views, masks, H)
    grouping = view_grouping if view_grouping is not None else capture.get('view_grouping')
    label, info, meta = compound_regions(cams, views, dets, H, grouping)
    stem = fit_stem(cams, views, dets, H)
    if stem['status'] != 'fitted':
        hull_stem, trunk = stem_from_hull(cams, views, dets, H)
        hull_stem['ray_fit_unavailable'] = stem['reason']
        if hull_stem['status'] == 'fitted':
            # trunk voxels were voted into whichever compound box covered the stem (dwt_04: L02 reached the soil);
            # they leave the regions before naming so region heights describe leaves, not the stem
            moved = label[trunk]
            hull_stem['trunk_voxels_taken_from_regions'] = {int(k): int((moved == k).sum()) for k in np.unique(moved) if k >= 0}
            label[trunk] = -1
        stem = hull_stem
    growth = read_json(growth_path) if growth_path else None
    regions, naming = name_regions(H, label, info, growth)
    assignment_hash = region_assignment_digest(grouping, H, label, info)
    decisions, names_status = reviewed_region_names(read_json(names_path) if names_path else None, assignment_hash)
    human_named = apply_name_decisions(regions, decisions)
    if human_named:
        naming += f'; names confirmed by a person for regions {human_named}'
        taken = [e['proposed_name'] for e in regions if e.get('proposed_name')]
        clash = sorted({n for n in taken if taken.count(n) > 1})
        if clash:
            naming += f'; DUPLICATE names after the decisions: {clash} - rename the proposal that was not decided'
    petioles, petiole_note = fit_petioles(cams, views, dets, H, label, stem)
    out.mkdir(parents=True)
    with open(out / 'hull_voxels.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['x_mm', 'y_mm', 'z_mm', 'compound_region'])
        w.writerows([[f'{x:.1f}', f'{y:.1f}', f'{z:.1f}', int(k)] for (x, y, z), k in zip(H, label)])
    (out / 'overlays').mkdir()
    for v in views:
        draw_overlay(cams, v, images[v], H, label, stem, petioles, out / 'overlays' / f'{Path(v).stem}.jpg',
                     {e['region']: e.get('proposed_name') for e in regions})
    report = dict(schema=SCHEMA, plant_id=backup['plant_id'], session_id=capture.get('session_id'),
                  inputs=dict(capture=str(session / 'capture.json'), capture_sha256=file_hash(session / 'capture.json'),
                              backup=str(backup_path), backup_sha256=file_hash(backup_path),
                              growth=str(growth_path) if growth_path else None,
                              name_decisions=str(names_path) if names_path else None),
                  views_used=views, views_skipped=skipped + [dict(photo=d['photo'], reason=f"hull explains only "
                  f"{d['explained_pct']} % of its silhouette (< {MIN_EXPLAINED_PCT} %): disagrees with the other photos")
                                                             for d in dropped_views],
                  view_selection=dict(min_explained_pct=MIN_EXPLAINED_PCT, min_views=MIN_CARVE_VIEWS, dropped=dropped_views,
                                      low_confidence=len(views) < 3),
                  hull=dict(voxel_mm=FINE_MM, voxels=int(len(H)), occupancy_cm3=round(len(H) * FINE_MM ** 3 / 1000, 1),
                            detached_voxels_removed=detached,
                            extent_mm={a: [round(float(H[:, i].min()), 1), round(float(H[:, i].max()), 1)] for i, a in enumerate('xyz')},
                            silhouette_explained_pct=explained, dilation_px=DILATION_PX,
                            note='occupancy envelope for orientation/grouping; not a leaf surface, area or tissue volume'),
                  compound_regions=regions, compound_region_meta=dict(meta, naming=naming, unassigned_voxels=int((label < 0).sum())),
                  view_grouping=grouping,
                  region_assignment_hash=assignment_hash, human_names_status=names_status,
                  stem=stem, petioles=petioles, petiole_note=petiole_note,
                  parameters=dict(exg_min=EXG_MIN, green_min=GREEN_MIN, dilation_px=DILATION_PX, label_margin=LABEL_MARGIN,
                                  coarse_mm=COARSE_MM, fine_mm=FINE_MM, min_box_iou=MIN_BOX_IOU, consensus_mm=CONSENSUS_MM,
                                  consensus_angle_deg=CONSENSUS_ANGLE_DEG, petiole_min_share=PETIOLE_MIN_SHARE,
                                  trunk_radius_mm=TRUNK_RADIUS_MM, trunk_slice_mm=TRUNK_SLICE_MM, trunk_widen=TRUNK_WIDEN),
                  validated_metric=False, suitable_for_ppfd=False,
                  limits='poses fail the 1 px gate; hull only from posed labelled photos; names are proposals')
    report = plain(report)
    write_json(out / 'structure.json', report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--session', required=True, type=Path)
    ap.add_argument('--backup', required=True, type=Path)
    ap.add_argument('--growth', type=Path)
    ap.add_argument('--names', type=Path, help='Hash-bound region_assignment_hash and regions mapping; see tools/OBSERVED_LEAF_REVISION.md.')
    ap.add_argument('--view-grouping', type=Path, help='Explicit board reference, elevation threshold and provenance JSON.')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    r = build(a.session, a.backup, a.out, a.growth, a.names,
              read_json(a.view_grouping) if a.view_grouping else None)
    print(json.dumps(dict(views=r['views_used'], voxels=r['hull']['voxels'], regions=len(r['compound_regions']),
                          stem=r['stem'].get('status'), petioles=[p.get('status') for p in r['petioles']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
