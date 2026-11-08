"""Grid-accelerated shadow-ray tracer for STATIC mode (CONTRACT.md §1–2). Triangles and elliptical discs, black optics (any hit blocks).

Primitives are packed in P (n, 14): triangle = A, E1, E2 (cols 0–8); disc = centre, unit tu, unit tv, unit n, a, b (cols 0–13).
kind[i] = 0 triangle, 1 disc. A uniform grid (cell h metres) stores each primitive in every cell its AABB overlaps (CSR: start/items).
A shadow ray from o along unit d is blocked if any primitive other than `own` is hit with tmin < t < tmax (tmin = self-cloud exclusion δ)."""
from __future__ import annotations

import math

import numba as nb
import numpy as np

TRI, DISC = 0, 1


@nb.njit(cache=True)
def prim_aabb(P, kind):
    n = P.shape[0]
    lo = np.empty((n, 3)); hi = np.empty((n, 3))
    for i in range(n):
        if kind[i] == TRI:
            for a in range(3):
                v0 = P[i, a]; v1 = v0 + P[i, 3 + a]; v2 = v0 + P[i, 6 + a]
                lo[i, a] = min(v0, min(v1, v2)); hi[i, a] = max(v0, max(v1, v2))
        else:
            ra, rb = P[i, 12], P[i, 13]
            for a in range(3):
                e = math.sqrt((ra * P[i, 3 + a]) ** 2 + (rb * P[i, 6 + a]) ** 2)
                lo[i, a] = P[i, a] - e; hi[i, a] = P[i, a] + e
    return lo, hi


@nb.njit(cache=True)
def build_grid(lo, hi, gmin, h, dims):
    nx, ny, nz = dims[0], dims[1], dims[2]
    ncell = nx * ny * nz
    count = np.zeros(ncell + 1, np.int64)
    n = lo.shape[0]
    for i in range(n):
        i0 = max(0, min(nx - 1, int((lo[i, 0] - gmin[0]) / h))); i1 = max(0, min(nx - 1, int((hi[i, 0] - gmin[0]) / h)))
        j0 = max(0, min(ny - 1, int((lo[i, 1] - gmin[1]) / h))); j1 = max(0, min(ny - 1, int((hi[i, 1] - gmin[1]) / h)))
        k0 = max(0, min(nz - 1, int((lo[i, 2] - gmin[2]) / h))); k1 = max(0, min(nz - 1, int((hi[i, 2] - gmin[2]) / h)))
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                for kk in range(k0, k1 + 1):
                    count[(ii * ny + jj) * nz + kk + 1] += 1
    for c in range(ncell):
        count[c + 1] += count[c]
    items = np.empty(count[ncell], np.int64)
    fill = count[:ncell].copy()
    for i in range(n):
        i0 = max(0, min(nx - 1, int((lo[i, 0] - gmin[0]) / h))); i1 = max(0, min(nx - 1, int((hi[i, 0] - gmin[0]) / h)))
        j0 = max(0, min(ny - 1, int((lo[i, 1] - gmin[1]) / h))); j1 = max(0, min(ny - 1, int((hi[i, 1] - gmin[1]) / h)))
        k0 = max(0, min(nz - 1, int((lo[i, 2] - gmin[2]) / h))); k1 = max(0, min(nz - 1, int((hi[i, 2] - gmin[2]) / h)))
        for ii in range(i0, i1 + 1):
            for jj in range(j0, j1 + 1):
                for kk in range(k0, k1 + 1):
                    c = (ii * ny + jj) * nz + kk
                    items[fill[c]] = i; fill[c] += 1
    return count, items


@nb.njit(cache=True, inline='always')
def hit(P, kind, i, ox, oy, oz, dx, dy, dz, tmin, tmax):
    if kind[i] == TRI:
        ax, ay, az = P[i, 0], P[i, 1], P[i, 2]
        e1x, e1y, e1z = P[i, 3], P[i, 4], P[i, 5]
        e2x, e2y, e2z = P[i, 6], P[i, 7], P[i, 8]
        px = dy * e2z - dz * e2y; py = dz * e2x - dx * e2z; pz = dx * e2y - dy * e2x
        det = e1x * px + e1y * py + e1z * pz
        if abs(det) < 1e-20:
            return False
        inv = 1.0 / det
        tx, ty, tz = ox - ax, oy - ay, oz - az
        u = (tx * px + ty * py + tz * pz) * inv
        if u < 0.0 or u > 1.0:
            return False
        qx = ty * e1z - tz * e1y; qy = tz * e1x - tx * e1z; qz = tx * e1y - ty * e1x
        v = (dx * qx + dy * qy + dz * qz) * inv
        if v < 0.0 or u + v > 1.0:
            return False
        t = (e2x * qx + e2y * qy + e2z * qz) * inv
        return t > tmin and t < tmax
    nx, ny, nz = P[i, 9], P[i, 10], P[i, 11]
    dn = dx * nx + dy * ny + dz * nz
    if abs(dn) < 1e-15:
        return False
    t = ((P[i, 0] - ox) * nx + (P[i, 1] - oy) * ny + (P[i, 2] - oz) * nz) / dn
    if t <= tmin or t >= tmax:
        return False
    qx = ox + t * dx - P[i, 0]; qy = oy + t * dy - P[i, 1]; qz = oz + t * dz - P[i, 2]
    u = (qx * P[i, 3] + qy * P[i, 4] + qz * P[i, 5]) / P[i, 12]
    v = (qx * P[i, 6] + qy * P[i, 7] + qz * P[i, 8]) / P[i, 13]
    return u * u + v * v <= 1.0


@nb.njit(cache=True)
def occluded(ox, oy, oz, dx, dy, dz, tmin, tmax, own, P, kind, gmin, h, dims, start, items):
    if P.shape[0] == 0:
        return False
    nx, ny, nz = dims[0], dims[1], dims[2]
    o = (ox, oy, oz); d = (dx, dy, dz)
    t0 = tmin; t1 = tmax
    for a in range(3):
        lo = gmin[a]; hi = gmin[a] + dims[a] * h
        if abs(d[a]) < 1e-15:
            if o[a] < lo or o[a] > hi:
                return False
        else:
            ta = (lo - o[a]) / d[a]; tb = (hi - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            if ta > t0:
                t0 = ta
            if tb < t1:
                t1 = tb
    if t0 > t1:
        return False
    px = ox + t0 * dx; py = oy + t0 * dy; pz = oz + t0 * dz
    ix = min(nx - 1, max(0, int((px - gmin[0]) / h)))
    iy = min(ny - 1, max(0, int((py - gmin[1]) / h)))
    iz = min(nz - 1, max(0, int((pz - gmin[2]) / h)))
    sx = 1 if dx > 0 else -1; sy = 1 if dy > 0 else -1; sz = 1 if dz > 0 else -1
    big = 1e30
    if abs(dx) > 1e-15:
        nbx = gmin[0] + (ix + (1 if dx > 0 else 0)) * h; tmx = (nbx - ox) / dx; tdx = h / abs(dx)
    else:
        tmx = big; tdx = big
    if abs(dy) > 1e-15:
        nby = gmin[1] + (iy + (1 if dy > 0 else 0)) * h; tmy = (nby - oy) / dy; tdy = h / abs(dy)
    else:
        tmy = big; tdy = big
    if abs(dz) > 1e-15:
        nbz = gmin[2] + (iz + (1 if dz > 0 else 0)) * h; tmz = (nbz - oz) / dz; tdz = h / abs(dz)
    else:
        tmz = big; tdz = big
    while True:
        c = (ix * ny + iy) * nz + iz
        for k in range(start[c], start[c + 1]):
            i = items[k]
            if i == own:
                continue
            if hit(P, kind, i, ox, oy, oz, dx, dy, dz, tmin, tmax):
                return True
        if tmx < tmy and tmx < tmz:
            if tmx > t1:
                break
            ix += sx; tmx += tdx
            if ix < 0 or ix >= nx:
                break
        elif tmy < tmz:
            if tmy > t1:
                break
            iy += sy; tmy += tdy
            if iy < 0 or iy >= ny:
                break
        else:
            if tmz > t1:
                break
            iz += sz; tmz += tdz
            if iz < 0 or iz >= nz:
                break
    return False


class Scene:
    """Occluder set + grid. `h` metres per cell; an empty scene never blocks."""

    def __init__(self, P: np.ndarray, kind: np.ndarray, h: float = 0.002, pad: float = 0.002):
        self.P = np.ascontiguousarray(P, dtype=np.float64); self.kind = np.ascontiguousarray(kind, dtype=np.int8)
        if len(self.P) == 0:
            self.gmin = np.zeros(3); self.h = h; self.dims = np.ones(3, np.int64)
            self.start = np.zeros(2, np.int64); self.items = np.zeros(0, np.int64); return
        lo, hi = prim_aabb(self.P, self.kind)
        self.gmin = lo.min(axis=0) - pad; gmax = hi.max(axis=0) + pad
        self.h = h; self.dims = np.maximum(1, np.ceil((gmax - self.gmin) / h).astype(np.int64))
        self.start, self.items = build_grid(lo, hi, self.gmin, h, self.dims)


@nb.njit(cache=True, parallel=True)
def point_lamps(R, N, two_sided, own, tmin, lamps, intensity, P, kind, gmin, h, dims, start, items):
    """Isotropic point lamps: E = I |cos b| / d^2 (one-sided: max(cos b, 0)). Returns shaded and unshaded irradiance per receiver."""
    n = R.shape[0]; shaded = np.zeros(n); unshaded = np.zeros(n)
    for r in nb.prange(n):
        for j in range(lamps.shape[0]):
            vx = lamps[j, 0] - R[r, 0]; vy = lamps[j, 1] - R[r, 1]; vz = lamps[j, 2] - R[r, 2]
            dist = math.sqrt(vx * vx + vy * vy + vz * vz)
            ux, uy, uz = vx / dist, vy / dist, vz / dist
            cb = ux * N[r, 0] + uy * N[r, 1] + uz * N[r, 2]
            if two_sided:
                cb = abs(cb)
            elif cb <= 0:
                continue
            e = intensity[j] * cb / (dist * dist)
            unshaded[r] += e
            if not occluded(R[r, 0], R[r, 1], R[r, 2], ux, uy, uz, tmin, dist, own[r], P, kind, gmin, h, dims, start, items):
                shaded[r] += e
    return shaded, unshaded


@nb.njit(cache=True, parallel=True)
def cabinet_kernel(R, N, two_sided, own, tmin, bar_pts, cos_m, dome, f_diffuse, P, kind, gmin, h, dims, start, items):
    """Normalised cabinet kernel K (unit unshaded horizontal light): direct part from bar sample points (I ∝ cos^m about -z, 1/d^2) and an
    isotropic dome. Returns K_shaded, K_unshaded for the receiver normal, and Hdir (absolute model direct horizontal, for scenario ratios)."""
    n = R.shape[0]; ks = np.zeros(n); ku = np.zeros(n); hdir = np.zeros(n); parts = np.zeros((n, 4))
    dome_h = 0.0
    for j in range(dome.shape[0]):
        dome_h += dome[j, 2]
    for r in nb.prange(n):
        wsum_h = 0.0; dir_s = 0.0; dir_u = 0.0
        for k in range(bar_pts.shape[0]):
            vx = bar_pts[k, 0] - R[r, 0]; vy = bar_pts[k, 1] - R[r, 1]; vz = bar_pts[k, 2] - R[r, 2]
            dist = math.sqrt(vx * vx + vy * vy + vz * vz)
            ux, uy, uz = vx / dist, vy / dist, vz / dist
            if uz <= 0:
                continue
            w = uz ** cos_m / (dist * dist)
            wsum_h += w * uz
            cb = ux * N[r, 0] + uy * N[r, 1] + uz * N[r, 2]
            if two_sided:
                cb = abs(cb)
            elif cb <= 0:
                continue
            dir_u += w * cb
            if not occluded(R[r, 0], R[r, 1], R[r, 2], ux, uy, uz, tmin, dist, own[r], P, kind, gmin, h, dims, start, items):
                dir_s += w * cb
        hdir[r] = wsum_h
        dif_s = 0.0; dif_u = 0.0
        for j in range(dome.shape[0]):
            cb = dome[j, 0] * N[r, 0] + dome[j, 1] * N[r, 1] + dome[j, 2] * N[r, 2]
            if two_sided:
                cb = abs(cb)
            elif cb <= 0:
                continue
            dif_u += cb
            if not occluded(R[r, 0], R[r, 1], R[r, 2], dome[j, 0], dome[j, 1], dome[j, 2], tmin, 10.0, own[r], P, kind, gmin, h, dims, start, items):
                dif_s += cb
        if wsum_h > 0:
            ks[r] = (1 - f_diffuse) * dir_s / wsum_h + f_diffuse * dif_s / dome_h
            ku[r] = (1 - f_diffuse) * dir_u / wsum_h + f_diffuse * dif_u / dome_h
            parts[r, 0] = dir_s / wsum_h; parts[r, 1] = dir_u / wsum_h
        parts[r, 2] = dif_s / dome_h; parts[r, 3] = dif_u / dome_h
    return ks, ku, hdir, parts


def trace_lamps(scene: Scene, R, N, two_sided, own, tmin, lamps, intensity):
    return point_lamps(np.ascontiguousarray(R, float), np.ascontiguousarray(N, float), bool(two_sided), np.ascontiguousarray(own, np.int64), float(tmin),
                       np.ascontiguousarray(lamps, float), np.ascontiguousarray(intensity, float), scene.P, scene.kind, scene.gmin, scene.h,
                       scene.dims, scene.start, scene.items)


def trace_cabinet(scene: Scene, R, N, two_sided, own, tmin, bar_pts, cos_m, dome, f_diffuse):
    return cabinet_kernel(np.ascontiguousarray(R, float), np.ascontiguousarray(N, float), bool(two_sided), np.ascontiguousarray(own, np.int64),
                          float(tmin), np.ascontiguousarray(bar_pts, float), float(cos_m), np.ascontiguousarray(dome, float), float(f_diffuse),
                          scene.P, scene.kind, scene.gmin, scene.h, scene.dims, scene.start, scene.items)


@nb.njit(cache=True, parallel=True)
def _coincident(R, N, own, eps, P, kind, gmin, h, dims, start, items):
    n = R.shape[0]; out = np.zeros(n, np.bool_)
    for r in nb.prange(n):
        out[r] = occluded(R[r, 0] - eps * N[r, 0], R[r, 1] - eps * N[r, 1], R[r, 2] - eps * N[r, 2], N[r, 0], N[r, 1], N[r, 2], 0.0, 2 * eps,
                          own[r], P, kind, gmin, h, dims, start, items)
    return out


def coincident(scene: Scene, R, N, own, eps: float = 1e-6):
    """True where another primitive passes within +-eps of the receiver along its normal (coplanar overlap). Used for the engine's
    mirrored leaflet pairs: a GroIMP ray reaching such a point is absorbed by one surface only (GATES_DIAGNOSIS.md)."""
    return _coincident(np.ascontiguousarray(R, float), np.ascontiguousarray(N, float), np.ascontiguousarray(own, np.int64), float(eps),
                       scene.P, scene.kind, scene.gmin, scene.h, scene.dims, scene.start, scene.items)


# ---------------------------------------------------------------------------------------------------------------------------------
# v1_tau additions (engine_static_tau_20261031_01/CONTRACT.md). Nothing above this line is changed.
# A shadow ray collects every distinct primitive hit in (tmin, tmax) over the same grid cells `occluded` visits, with the same hit
# tests; hits are sorted by t and chained into crossings (consecutive gap <= eps = one sheet); T = product over crossings of the
# smallest tau among the crossing's members. tau 0 returns 0.0 at the first hit, so tau = 0 reproduces `occluded` exactly.
MAXH = 2048


@nb.njit(cache=True, inline='always')
def hit_t(P, kind, i, ox, oy, oz, dx, dy, dz, tmin, tmax):
    """Same tests as `hit`; returns t of the hit, or -1.0."""
    if kind[i] == TRI:
        ax, ay, az = P[i, 0], P[i, 1], P[i, 2]
        e1x, e1y, e1z = P[i, 3], P[i, 4], P[i, 5]
        e2x, e2y, e2z = P[i, 6], P[i, 7], P[i, 8]
        px = dy * e2z - dz * e2y; py = dz * e2x - dx * e2z; pz = dx * e2y - dy * e2x
        det = e1x * px + e1y * py + e1z * pz
        if abs(det) < 1e-20:
            return -1.0
        inv = 1.0 / det
        tx, ty, tz = ox - ax, oy - ay, oz - az
        u = (tx * px + ty * py + tz * pz) * inv
        if u < 0.0 or u > 1.0:
            return -1.0
        qx = ty * e1z - tz * e1y; qy = tz * e1x - tx * e1z; qz = tx * e1y - ty * e1x
        v = (dx * qx + dy * qy + dz * qz) * inv
        if v < 0.0 or u + v > 1.0:
            return -1.0
        t = (e2x * qx + e2y * qy + e2z * qz) * inv
        return t if (t > tmin and t < tmax) else -1.0
    nx, ny, nz = P[i, 9], P[i, 10], P[i, 11]
    dn = dx * nx + dy * ny + dz * nz
    if abs(dn) < 1e-15:
        return -1.0
    t = ((P[i, 0] - ox) * nx + (P[i, 1] - oy) * ny + (P[i, 2] - oz) * nz) / dn
    if t <= tmin or t >= tmax:
        return -1.0
    qx = ox + t * dx - P[i, 0]; qy = oy + t * dy - P[i, 1]; qz = oz + t * dz - P[i, 2]
    u = (qx * P[i, 3] + qy * P[i, 4] + qz * P[i, 5]) / P[i, 12]
    v = (qx * P[i, 6] + qy * P[i, 7] + qz * P[i, 8]) / P[i, 13]
    return t if u * u + v * v <= 1.0 else -1.0


@nb.njit(cache=True)
def transmit(ox, oy, oz, dx, dy, dz, tmin, tmax, own, P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi):
    """Transmission factor in [0, 1] along one shadow ray; -1.0 = more than len(bt) distinct hits (caller treats as blocked, counts it)."""
    if P.shape[0] == 0:
        return 1.0
    nx, ny, nz = dims[0], dims[1], dims[2]
    o = (ox, oy, oz); d = (dx, dy, dz)
    t0 = tmin; t1 = tmax
    for a in range(3):
        lo = gmin[a]; hi = gmin[a] + dims[a] * h
        if abs(d[a]) < 1e-15:
            if o[a] < lo or o[a] > hi:
                return 1.0
        else:
            ta = (lo - o[a]) / d[a]; tb = (hi - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            if ta > t0:
                t0 = ta
            if tb < t1:
                t1 = tb
    if t0 > t1:
        return 1.0
    px = ox + t0 * dx; py = oy + t0 * dy; pz = oz + t0 * dz
    ix = min(nx - 1, max(0, int((px - gmin[0]) / h)))
    iy = min(ny - 1, max(0, int((py - gmin[1]) / h)))
    iz = min(nz - 1, max(0, int((pz - gmin[2]) / h)))
    sx = 1 if dx > 0 else -1; sy = 1 if dy > 0 else -1; sz = 1 if dz > 0 else -1
    big = 1e30
    if abs(dx) > 1e-15:
        nbx = gmin[0] + (ix + (1 if dx > 0 else 0)) * h; tmx = (nbx - ox) / dx; tdx = h / abs(dx)
    else:
        tmx = big; tdx = big
    if abs(dy) > 1e-15:
        nby = gmin[1] + (iy + (1 if dy > 0 else 0)) * h; tmy = (nby - oy) / dy; tdy = h / abs(dy)
    else:
        tmy = big; tdy = big
    if abs(dz) > 1e-15:
        nbz = gmin[2] + (iz + (1 if dz > 0 else 0)) * h; tmz = (nbz - oz) / dz; tdz = h / abs(dz)
    else:
        tmz = big; tdz = big
    nh = 0; cap = bt.shape[0]
    while True:
        c = (ix * ny + iy) * nz + iz
        for k in range(start[c], start[c + 1]):
            i = items[k]
            if i == own:
                continue
            t = hit_t(P, kind, i, ox, oy, oz, dx, dy, dz, tmin, tmax)
            if t < 0.0:
                continue
            if tau_prim[i] <= 0.0:
                return 0.0
            dup = False
            for m in range(nh):
                if bi[m] == i:
                    dup = True
                    break
            if dup:
                continue
            if nh >= cap:
                return -1.0
            bt[nh] = t; bi[nh] = i; nh += 1
        if tmx < tmy and tmx < tmz:
            if tmx > t1:
                break
            ix += sx; tmx += tdx
            if ix < 0 or ix >= nx:
                break
        elif tmy < tmz:
            if tmy > t1:
                break
            iy += sy; tmy += tdy
            if iy < 0 or iy >= ny:
                break
        else:
            if tmz > t1:
                break
            iz += sz; tmz += tdz
            if iz < 0 or iz >= nz:
                break
    if nh == 0:
        return 1.0
    for m in range(1, nh):                     # insertion sort by t
        tv = bt[m]; iv = bi[m]; j = m - 1
        while j >= 0 and bt[j] > tv:
            bt[j + 1] = bt[j]; bi[j + 1] = bi[j]; j -= 1
        bt[j + 1] = tv; bi[j + 1] = iv
    T = 1.0; cur = tau_prim[bi[0]]; last = bt[0]
    for m in range(1, nh):
        if bt[m] - last <= eps:
            cur = min(cur, tau_prim[bi[m]])
        else:
            T *= cur; cur = tau_prim[bi[m]]
        last = bt[m]
    return T * cur


@nb.njit(cache=True, parallel=True)
def cabinet_kernel_tau(R, N, two_sided, own, tmin, bar_pts, cos_m, dome, f_diffuse, P, kind, tau_prim, eps, gmin, h, dims, start, items):
    """`cabinet_kernel` with `occluded` replaced by `transmit`; also returns the per-receiver overflow count."""
    n = R.shape[0]; ks = np.zeros(n); ku = np.zeros(n); hdir = np.zeros(n); parts = np.zeros((n, 4)); novf = np.zeros(n, np.int64)
    dome_h = 0.0
    for j in range(dome.shape[0]):
        dome_h += dome[j, 2]
    for r in nb.prange(n):
        bt = np.empty(MAXH); bi = np.empty(MAXH, np.int64)
        wsum_h = 0.0; dir_s = 0.0; dir_u = 0.0
        for k in range(bar_pts.shape[0]):
            vx = bar_pts[k, 0] - R[r, 0]; vy = bar_pts[k, 1] - R[r, 1]; vz = bar_pts[k, 2] - R[r, 2]
            dist = math.sqrt(vx * vx + vy * vy + vz * vz)
            ux, uy, uz = vx / dist, vy / dist, vz / dist
            if uz <= 0:
                continue
            w = uz ** cos_m / (dist * dist)
            wsum_h += w * uz
            cb = ux * N[r, 0] + uy * N[r, 1] + uz * N[r, 2]
            if two_sided:
                cb = abs(cb)
            elif cb <= 0:
                continue
            dir_u += w * cb
            T = transmit(R[r, 0], R[r, 1], R[r, 2], ux, uy, uz, tmin, dist, own[r], P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi)
            if T < 0.0:
                novf[r] += 1; T = 0.0
            if T > 0.0:
                dir_s += w * cb * T
        hdir[r] = wsum_h
        dif_s = 0.0; dif_u = 0.0
        for j in range(dome.shape[0]):
            cb = dome[j, 0] * N[r, 0] + dome[j, 1] * N[r, 1] + dome[j, 2] * N[r, 2]
            if two_sided:
                cb = abs(cb)
            elif cb <= 0:
                continue
            dif_u += cb
            T = transmit(R[r, 0], R[r, 1], R[r, 2], dome[j, 0], dome[j, 1], dome[j, 2], tmin, 10.0, own[r], P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi)
            if T < 0.0:
                novf[r] += 1; T = 0.0
            if T > 0.0:
                dif_s += cb * T
        if wsum_h > 0:
            ks[r] = (1 - f_diffuse) * dir_s / wsum_h + f_diffuse * dif_s / dome_h
            ku[r] = (1 - f_diffuse) * dir_u / wsum_h + f_diffuse * dif_u / dome_h
            parts[r, 0] = dir_s / wsum_h; parts[r, 1] = dir_u / wsum_h
        parts[r, 2] = dif_s / dome_h; parts[r, 3] = dif_u / dome_h
    return ks, ku, hdir, parts, novf


def trace_cabinet_tau(scene: Scene, R, N, two_sided, own, tmin, bar_pts, cos_m, dome, f_diffuse, tau_prim, eps):
    return cabinet_kernel_tau(np.ascontiguousarray(R, float), np.ascontiguousarray(N, float), bool(two_sided), np.ascontiguousarray(own, np.int64),
                              float(tmin), np.ascontiguousarray(bar_pts, float), float(cos_m), np.ascontiguousarray(dome, float), float(f_diffuse),
                              scene.P, scene.kind, np.ascontiguousarray(tau_prim, float), float(eps), scene.gmin, scene.h, scene.dims, scene.start, scene.items)


@nb.njit(cache=True, parallel=True)
def _transmit_rays(R, D, own, tmin, tmax, P, kind, tau_prim, eps, gmin, h, dims, start, items):
    n = R.shape[0]; out = np.zeros(n)
    for r in nb.prange(n):
        bt = np.empty(MAXH); bi = np.empty(MAXH, np.int64)
        out[r] = transmit(R[r, 0], R[r, 1], R[r, 2], D[r, 0], D[r, 1], D[r, 2], tmin, tmax, own[r], P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi)
    return out


def transmit_rays(scene: Scene, R, D, own, tmin, tmax, tau_prim, eps):
    """T along explicit rays (tests)."""
    return _transmit_rays(np.ascontiguousarray(R, float), np.ascontiguousarray(D, float), np.ascontiguousarray(own, np.int64), float(tmin), float(tmax),
                          scene.P, scene.kind, np.ascontiguousarray(tau_prim, float), float(eps), scene.gmin, scene.h, scene.dims, scene.start, scene.items)
