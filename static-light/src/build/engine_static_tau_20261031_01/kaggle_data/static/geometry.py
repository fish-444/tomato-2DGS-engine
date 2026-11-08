"""Geometry for STATIC mode (CONTRACT.md §1): surfel discs with rule-(b) area, engine snapshot triangles, receivers, placement."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numba as nb
import numpy as np

from .tracer import DISC, TRI

PLY_TYPES = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4'}


def read_ply(path: Path) -> np.ndarray:
    with open(path, 'rb') as fh:
        props, n, elem = [], 0, None
        while True:
            line = fh.readline().decode('ascii', 'replace').strip()
            if line.startswith('element'):
                elem = line.split()[1]
                if elem == 'vertex': n = int(line.split()[-1])
            elif line.startswith('property') and elem == 'vertex':
                props.append((line.split()[2], PLY_TYPES[line.split()[1]]))
            elif line == 'end_header':
                break
        dt = np.dtype(props)
        return np.frombuffer(fh.read(dt.itemsize * n), dtype=dt, count=n)


def footprint_k(opacity: np.ndarray) -> np.ndarray:
    """Half-opacity contour of a 2DGS Gaussian: alpha * exp(-r^2/2) = 0.5 -> r = sqrt(2 ln(2 alpha)) sigma (alpha > 0.5)."""
    a = np.clip(opacity.astype(float), 0.5 + 1e-9, None)
    return np.sqrt(2 * np.log(2 * a))


class Surfels:
    """Opaque surfels of one plant as discs. `scale` multiplies the footprint (sensitivity only)."""

    def __init__(self, ply: Path, keep: np.ndarray | None = None, scale: float = 1.0, labels: np.ndarray | None = None):
        d = read_ply(ply)
        opaque = d['opacity'] > 0.5
        sel = opaque if keep is None else (opaque & keep)
        self.source = str(ply); self.index = np.nonzero(sel)[0]; d = d[sel]
        self.centre = np.stack([d['x'], d['y'], d['z']], 1).astype(float)
        tu = np.stack([d['tux'], d['tuy'], d['tuz']], 1).astype(float); tv = np.stack([d['tvx'], d['tvy'], d['tvz']], 1).astype(float)
        n = np.stack([d['nx'], d['ny'], d['nz']], 1).astype(float)
        self.tu = tu / np.linalg.norm(tu, axis=1, keepdims=True); self.tv = tv / np.linalg.norm(tv, axis=1, keepdims=True)
        self.normal = n / np.linalg.norm(n, axis=1, keepdims=True)
        k = footprint_k(d['opacity']) * scale
        self.a = np.maximum(k * d['s0'].astype(float), 1e-7); self.b = np.maximum(k * d['s1'].astype(float), 1e-7)
        self.area_a = math.pi * self.a * self.b
        self.labels = None if labels is None else labels[self.index]
        self.area_b, self.cube_c = calibrate_area_b(self.centre, self.normal, self.tu, self.tv, self.a, self.b)

    def __len__(self):
        return len(self.centre)

    def prims(self, offset=np.zeros(3)) -> tuple[np.ndarray, np.ndarray]:
        P = np.zeros((len(self), 14)); P[:, 0:3] = self.centre + offset; P[:, 3:6] = self.tu; P[:, 6:9] = self.tv; P[:, 9:12] = self.normal
        P[:, 12] = self.a; P[:, 13] = self.b
        return P, np.full(len(self), DISC, np.int8)


@nb.njit(cache=True, parallel=True)
def _cube_union(cstart, order, C, N, TU, TV, A, B, res, lo, cube):
    ncube = len(cstart) - 1
    union = np.zeros(ncube); sproj = np.zeros(ncube)
    for c in nb.prange(ncube):
        s0, s1 = cstart[c], cstart[c + 1]
        r0 = order[s0]
        mx = 0.0; my = 0.0; mz = 0.0; px = 0.0; py = 0.0; pz = 0.0; wt = 0.0
        for k in range(s0, s1):
            i = order[k]; w = math.pi * A[i] * B[i]
            px += w * C[i, 0]; py += w * C[i, 1]; pz += w * C[i, 2]; wt += w
            sg = 1.0 if (N[i, 0] * N[r0, 0] + N[i, 1] * N[r0, 1] + N[i, 2] * N[r0, 2]) >= 0 else -1.0
            mx += sg * w * N[i, 0]; my += sg * w * N[i, 1]; mz += sg * w * N[i, 2]
        nm = math.sqrt(mx * mx + my * my + mz * mz)
        if nm < 1e-15:
            mx, my, mz = N[r0, 0], N[r0, 1], N[r0, 2]; nm = 1.0
        mx /= nm; my /= nm; mz /= nm
        if abs(mx) < 0.9:
            ax, ay, az = 1.0, 0.0, 0.0
        else:
            ax, ay, az = 0.0, 1.0, 0.0
        e1x = ay * mz - az * my; e1y = az * mx - ax * mz; e1z = ax * my - ay * mx
        ne = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z); e1x /= ne; e1y /= ne; e1z /= ne
        e2x = my * e1z - mz * e1y; e2y = mz * e1x - mx * e1z; e2z = mx * e1y - my * e1x
        umin = 1e30; umax = -1e30; vmin = 1e30; vmax = -1e30
        for k in range(s0, s1):
            i = order[k]
            cu = C[i, 0] * e1x + C[i, 1] * e1y + C[i, 2] * e1z; cv = C[i, 0] * e2x + C[i, 1] * e2y + C[i, 2] * e2z
            m00 = A[i] * (TU[i, 0] * e1x + TU[i, 1] * e1y + TU[i, 2] * e1z); m01 = B[i] * (TV[i, 0] * e1x + TV[i, 1] * e1y + TV[i, 2] * e1z)
            m10 = A[i] * (TU[i, 0] * e2x + TU[i, 1] * e2y + TU[i, 2] * e2z); m11 = B[i] * (TV[i, 0] * e2x + TV[i, 1] * e2y + TV[i, 2] * e2z)
            hx = math.sqrt(m00 * m00 + m01 * m01); hy = math.sqrt(m10 * m10 + m11 * m11)
            umin = min(umin, cu - hx); umax = max(umax, cu + hx); vmin = min(vmin, cv - hy); vmax = max(vmax, cv + hy)
        nu = int((umax - umin) / res) + 2; nv = int((vmax - vmin) / res) + 2
        mask = np.zeros((nu, nv), np.uint8)
        sp = 0.0
        for k in range(s0, s1):
            i = order[k]
            cu = C[i, 0] * e1x + C[i, 1] * e1y + C[i, 2] * e1z; cv = C[i, 0] * e2x + C[i, 1] * e2y + C[i, 2] * e2z
            m00 = A[i] * (TU[i, 0] * e1x + TU[i, 1] * e1y + TU[i, 2] * e1z); m01 = B[i] * (TV[i, 0] * e1x + TV[i, 1] * e1y + TV[i, 2] * e1z)
            m10 = A[i] * (TU[i, 0] * e2x + TU[i, 1] * e2y + TU[i, 2] * e2z); m11 = B[i] * (TV[i, 0] * e2x + TV[i, 1] * e2y + TV[i, 2] * e2z)
            det = m00 * m11 - m01 * m10
            if abs(det) < 1e-16:
                continue
            sp += math.pi * abs(det)
            # inverse of S = M M^T
            s00 = m00 * m00 + m01 * m01; s01 = m00 * m10 + m01 * m11; s11 = m10 * m10 + m11 * m11
            sd = s00 * s11 - s01 * s01
            q00 = s11 / sd; q01 = -s01 / sd; q11 = s00 / sd
            hx = math.sqrt(s00); hy = math.sqrt(s11)
            i0 = max(0, int((cu - hx - umin) / res)); i1 = min(nu - 1, int((cu + hx - umin) / res) + 1)
            j0 = max(0, int((cv - hy - vmin) / res)); j1 = min(nv - 1, int((cv + hy - vmin) / res) + 1)
            for ii in range(i0, i1 + 1):
                x = umin + (ii + 0.5) * res - cu
                for jj in range(j0, j1 + 1):
                    y = vmin + (jj + 0.5) * res - cv
                    if q00 * x * x + 2 * q01 * x * y + q11 * y * y <= 1.0:
                        mask[ii, jj] = 1
        # v2: keep only pixels whose point on the mean plane lies inside this cube (v1 kept footprint overhang -> boundary double count)
        px /= wt; py /= wt; pz /= wt
        u0 = px * e1x + py * e1y + pz * e1z; v0 = px * e2x + py * e2y + pz * e2z
        cnt = 0
        for ii in range(nu):
            uu = umin + (ii + 0.5) * res - u0
            for jj in range(nv):
                if mask[ii, jj] == 0:
                    continue
                vv = vmin + (jj + 0.5) * res - v0
                qx = px + uu * e1x + vv * e2x; qy = py + uu * e1y + vv * e2y; qz = pz + uu * e1z + vv * e2z
                if lo[c, 0] <= qx < lo[c, 0] + cube and lo[c, 1] <= qy < lo[c, 1] + cube and lo[c, 2] <= qz < lo[c, 2] + cube:
                    cnt += 1
        union[c] = cnt * res * res; sproj[c] = sp
    return union, sproj


def calibrate_area_b(centre, normal, tu, tv, a, b, cube: float = 0.008, res: float = 0.0002):
    """Rule (b): per 8 mm cube, union of footprints projected on the cube's mean plane / summed projected area; area_b = c * pi a b."""
    key = np.floor(centre / cube).astype(np.int64)
    ukey, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    order = np.argsort(inv, kind='stable'); counts = np.bincount(inv)
    cstart = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    union, sproj = _cube_union(cstart, order.astype(np.int64), centre, normal, tu, tv, a, b, res, (ukey * cube).astype(np.float64), float(cube))
    c = np.where(sproj > 0, union / np.maximum(sproj, 1e-30), 1.0)
    c = np.minimum(c, 1.0)
    return c[inv] * math.pi * a * b, c[inv]


class EngineSnapshot:
    """Triangles of one consumed state of an engine run (`consumed.csv`), classes and plants from `states.csv` of the same day."""

    def __init__(self, working: Path, state: int):
        verts = defaultdict(list)
        with open(working / 'consumed.csv', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if int(r['state']) == state:
                    verts[r['organ_id']].append((int(r['vertex']), float(r['x_m']), float(r['y_m']), float(r['z_m'])))
        cls, plant, area = {}, {}, {}
        with open(working / 'states.csv', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if int(r['day']) == state:
                    cls[r['organ_id']] = r['class']; plant[r['organ_id']] = r['plant_instance_id']; area[r['organ_id']] = float(r['area_m2'])
        tris, owner = [], []
        for oid, vs in verts.items():
            vs.sort(); arr = np.array([v[1:] for v in vs])
            for k in range(0, len(arr) - len(arr) % 3, 3):
                tris.append(arr[k:k + 3]); owner.append(oid)
        self.tri = np.array(tris); self.owner = np.array(owner); self.cls = cls; self.plant = plant; self.state_area = area
        self.is_leaf = np.array([cls.get(o) == 'Leaflet' for o in self.owner])
        e1 = self.tri[:, 1] - self.tri[:, 0]; e2 = self.tri[:, 2] - self.tri[:, 0]
        cr = np.cross(e1, e2); self.area = 0.5 * np.linalg.norm(cr, axis=1)
        self.normal = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-30)

    def prims(self, offset=np.zeros(3), keep: np.ndarray | None = None):
        t = self.tri if keep is None else self.tri[keep]
        P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0] + offset; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]
        return P, np.full(len(t), TRI, np.int8)

    def leaf_receivers(self, samples: int, seed: int, offset=np.zeros(3)):
        """`samples` random points per leaflet triangle (area weights area/samples). Returns R, N, weight, own (prim index), organ."""
        rng = np.random.default_rng(seed)
        idx = np.nonzero(self.is_leaf)[0]
        u = rng.random((len(idx), samples)); v = rng.random((len(idx), samples))
        flip = u + v > 1; u[flip] = 1 - u[flip]; v[flip] = 1 - v[flip]
        A = self.tri[idx, 0][:, None, :]; E1 = (self.tri[idx, 1] - self.tri[idx, 0])[:, None, :]; E2 = (self.tri[idx, 2] - self.tri[idx, 0])[:, None, :]
        R = (A + u[..., None] * E1 + v[..., None] * E2).reshape(-1, 3) + offset
        N = np.repeat(self.normal[idx], samples, axis=0); w = np.repeat(self.area[idx] / samples, samples)
        own = np.repeat(idx, samples); organ = np.repeat(self.owner[idx], samples)
        return R, N, w, own, organ
