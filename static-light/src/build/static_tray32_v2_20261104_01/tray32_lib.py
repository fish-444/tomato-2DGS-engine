"""Layout, luminaires, emission quadratures and the area-luminaire v1_tau kernel of the 32-plant tray scenes (CONTRACT.md §1, §3, §4). New code only; the tracer
(tau bundle: Scene, transmit, MAXH) is imported by the caller and passed in through `bind_tracer`."""
from __future__ import annotations

import math

import numba as nb
import numpy as np

M_EXP = 1.0                       # cos^m, half-angle 60 deg (light_consts.json cos_m)
LUM_LEN = 0.60; DS = 0.0125; WING = 0.05; FLAT_W = 0.10; GAP = 0.10; NY = 48
SPACINGS = (0.06, 0.10, 0.14); SHAPES = ('flat', 'V120', 'V90'); POSITIONS = ('trays', 'gaps'); HEIGHTS = (0.40, 0.50)
PHI = {'V120': math.radians(30.0), 'V90': math.radians(45.0)}
MEDIUM_Z = 0.060; RHO = 0.08; A_ABS = 0.85


def scene_list():
    return [dict(sid=f'S{k + 1:02d}', p=p, shape=s, pos=ps, height=h) for k, (p, s, ps, h) in enumerate(
        (p, s, ps, h) for p in SPACINGS for s in SHAPES for ps in POSITIONS for h in HEIGHTS)]


def tray_x(p):
    return [(i - 2.5) * (2 * p + GAP) for i in range(1, 5)]


def slot_xy(p, tray, row, col):
    return tray_x(p)[tray - 1] + (col - 1.5) * p, (row - 2.5) * p


def _wing(xv, z, sign, phi):
    """Strip of a V wing rising outward from the vertex line at (xv, z); sign -1 = left wing (normal down-left), +1 = right wing."""
    return dict(xa=xv, za=z, xb=xv + sign * WING * math.cos(phi), zb=z + WING * math.sin(phi), nx=sign * math.sin(phi), nz=-math.cos(phi), width=WING)


def faces(shape, pos, p, h):
    """List of faces (strip in xz extruded along y over LUM_LEN) with flux in L. Total flux = 4 L in every option."""
    xs = tray_x(p); out = []
    centres = xs if pos == 'trays' else [(xs[i] + xs[i + 1]) / 2 for i in range(3)]
    for xc in centres:
        if shape == 'flat':
            out.append(dict(xa=xc - FLAT_W / 2, za=h, xb=xc + FLAT_W / 2, zb=h, nx=0.0, nz=-1.0, width=FLAT_W, flux=1.0, tag='lum'))
        else:
            for sg in (-1, 1):
                w = _wing(xc, h, sg, PHI[shape]); w.update(flux=0.5, tag='lum'); out.append(w)
    if pos == 'gaps':
        xl, xr = xs[0] - p - 0.05, xs[3] + p + 0.05
        if shape == 'flat':
            for xe in (xl, xr):
                out.append(dict(xa=xe - 0.025, za=h, xb=xe + 0.025, zb=h, nx=0.0, nz=-1.0, width=0.05, flux=0.5, tag='edge'))
        else:
            w = _wing(xl, h, +1, PHI[shape]); w.update(flux=0.5, tag='edge'); out.append(w)       # faces inward (+x), rises toward the trays
            w = _wing(xr, h, -1, PHI[shape]); w.update(flux=0.5, tag='edge'); out.append(w)
    return out


def emitters(fcs):
    """Sample points (S, 3), emission axes (S, 3), per-point flux phi (S,) from the faces (regular grid, DS spacing; NY points along y)."""
    P, A, F = [], [], []
    ys = (np.arange(NY) + 0.5) / NY * LUM_LEN - LUM_LEN / 2
    for f in fcs:
        nw = int(round(f['width'] / DS)); t = (np.arange(nw) + 0.5) / nw
        x = f['xa'] + t * (f['xb'] - f['xa']); z = f['za'] + t * (f['zb'] - f['za'])
        X = np.repeat(x, NY); Z = np.repeat(z, NY); Y = np.tile(ys, nw)
        P.append(np.stack([X, Y, Z], 1)); A.append(np.tile([f['nx'], 0.0, f['nz']], (len(X), 1))); F.append(np.full(len(X), f['flux'] / (nw * NY)))
    return np.concatenate(P), np.concatenate(A), np.concatenate(F)


def fib_sphere(n):
    i = np.arange(n) + 0.5; z = 1 - 2 * i / n; r = np.sqrt(1 - z * z); ph = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)


def emission_checks(fcs, m=M_EXP, n=400000):
    """C2: numerical emitted flux of every face over the sphere (front half space, cos^m); upper-half-space share (flux lost upward)."""
    D = fib_sphere(n); res = []
    for f in fcs:
        a = np.array([f['nx'], 0.0, f['nz']]); c = np.maximum(D @ a, 0.0) ** m; tot = c.sum() * 4 * math.pi / n          # = 2 pi / (m + 1)
        up = c[D[:, 2] > 0].sum() * 4 * math.pi / n
        res.append(dict(flux_declared=f['flux'], flux_numeric=f['flux'] * tot / (2 * math.pi / (m + 1)), up_fraction=up / tot))
    return res


def plane_flux(P_em, A_em, phi, m, pts, normal_up=True):
    """Deterministic unshaded irradiance (all sample points, no occluders) at horizontal one-sided points: E = sum I0 cos^m(theta_e) cos(beta) / d^2."""
    I0 = phi * (m + 1) / (2 * math.pi); E = np.zeros(len(pts))
    for k in range(0, len(P_em), 256):
        sl = slice(k, k + 256); v = P_em[sl][None, :, :] - pts[:, None, :]; d = np.linalg.norm(v, axis=2); u = v / d[..., None]
        ce = -(u * A_em[sl][None, :, :]).sum(2); cb = u[..., 2]
        E += (I0[sl][None, :] * np.where((ce > 0) & (cb > 0), ce, 0.0) ** m * np.where(cb > 0, cb, 0.0) / d ** 2).sum(1)
    return E


def outlines_flux(P_em, A_em, phi, m, p, z=MEDIUM_Z, dx=0.01):
    """Unshaded flux reaching the four tray outlines (2p x 4p) on the substrate plane."""
    tot = 0.0
    for xc in tray_x(p):
        gx = np.arange(xc - p + dx / 2, xc + p, dx); gy = np.arange(-2 * p + dx / 2, 2 * p, dx); X, Y = np.meshgrid(gx, gy)
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1); tot += plane_flux(P_em, A_em, phi, m, pts).sum() * dx * dx
    return float(tot)


_TR = {}


def bind_tracer(transmit, MAXH):
    """Compile the area-luminaire kernel against the tau tracer's `transmit` (called once by the script/test)."""
    @nb.njit(cache=False, parallel=True)
    def area_lamps_tau(R, N, own, tmin, two_sided, E_pos, E_axis, I0, m, idx, scale, tau_prim, eps, P, kind, gmin, h, dims, start, items):
        n = R.shape[0]; sh = np.zeros(n); un = np.zeros(n); novf = np.zeros(n, np.int64)
        for r in nb.prange(n):
            bt = np.empty(MAXH); bi = np.empty(MAXH, np.int64)
            for kk in range(idx.shape[1]):
                s = idx[r, kk]
                vx = E_pos[s, 0] - R[r, 0]; vy = E_pos[s, 1] - R[r, 1]; vz = E_pos[s, 2] - R[r, 2]
                dist = math.sqrt(vx * vx + vy * vy + vz * vz); ux = vx / dist; uy = vy / dist; uz = vz / dist
                ce = -(ux * E_axis[s, 0] + uy * E_axis[s, 1] + uz * E_axis[s, 2])
                if ce <= 0.0:
                    continue
                cb = ux * N[r, 0] + uy * N[r, 1] + uz * N[r, 2]
                if two_sided:
                    cb = abs(cb)
                elif cb <= 0.0:
                    continue
                e = I0[s] * ce ** m * cb / (dist * dist) * scale; un[r] += e
                T = transmit(R[r, 0], R[r, 1], R[r, 2], ux, uy, uz, tmin, dist, own[r], P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi)
                if T < 0.0:
                    novf[r] += 1; T = 0.0
                sh[r] += e * T
        return sh, un, novf
    _TR['area_lamps_tau'] = area_lamps_tau
    return area_lamps_tau


def draw_idx(n_recv, n_emit, k_e, seed):
    return np.random.default_rng(seed).integers(0, n_emit, size=(n_recv, k_e)).astype(np.int32)


def rot_z(az_deg):
    c, s = math.cos(math.radians(az_deg)), math.sin(math.radians(az_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
