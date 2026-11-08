import json, os, sys
from pathlib import Path
OUTD = Path('/kaggle/working'); OUTD.mkdir(exist_ok=True)
LIB = r'''"""Layout, luminaires, emission quadratures and the area-luminaire v1_tau kernel of the 32-plant tray scenes (CONTRACT.md §1, §3, §4). New code only; the tracer
(tau bundle: Scene, transmit, MAXH) is imported by the caller and passed in through `bind_tracer`."""
from __future__ import annotations

import math

import numba as nb
import numpy as np

M_EXP = 1.0                       # cos^m, half-angle 60 deg (light_consts.json cos_m)
LUM_LEN = 0.60; DS = 0.0125; WING = 0.05; FLAT_W = 0.10; GAP = 0.10; NY = 48
SPACINGS = (0.06, 0.10, 0.14); SHAPES = ('flat', 'V120', 'V90'); POSITIONS = ('trays', 'gaps'); HEIGHTS = (0.30, 0.40, 0.50, 0.60)   # v3: 30 / 40 / 50 / 60 cm
V2_HEIGHTS = (0.40, 0.50)
PHI = {'V120': math.radians(30.0), 'V90': math.radians(45.0)}
MEDIUM_Z = 0.060; RHO = 0.08; A_ABS = 0.85


def scene_list():
    """v3: 72 scenes (S01..S72). `seed_k` = the v2 scene number for 40 / 50 cm (same emitter draws as v2, so the smoothing-off regression is exact) and 100 + v3 number otherwise;
    `v2_sid` names the matching v2 scene."""
    v2 = [(p, s, ps, h) for p in SPACINGS for s in SHAPES for ps in POSITIONS for h in V2_HEIGHTS]; out = []
    for k, (p, s, ps, h) in enumerate((p, s, ps, h) for p in SPACINGS for s in SHAPES for ps in POSITIONS for h in HEIGHTS):
        key = (p, s, ps, h); kv2 = v2.index(key) + 1 if key in v2 else None
        out.append(dict(sid=f'S{k + 1:02d}', p=p, shape=s, pos=ps, height=h, seed_k=kv2 if kv2 else 100 + k + 1, v2_sid=f'S{kv2:02d}' if kv2 else None))
    return out


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
'''
MAIN = r'''"""32-plant tray scenes (CONTRACT.md). Modes: scene (in-scene absorbed light, closure, outlines flux), iso (isolated copies, table A), conv (C4 ray-count convergence), extra (diffuse
sensitivity + e_* bracket arm). Tau bundle tracer read-only (static/*.py unchanged); light model v1_tau (tau 0.07, a 0.85), luminaires only. One JSON per scene in OUT/<mode>/.
usage: python tray32_main.py --table A --mode scene [--scenes S01,S02] [--recv-n 8000 --ke 32] [--max-copies N --spacings 0.1 --smoke]"""
import argparse, csv, glob, hashlib, json, math, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ('tau', 'nine', 'nd', 'out', 'lib'):
    ap.add_argument('--' + k)
ap.add_argument('--table', default='A'); ap.add_argument('--mode', default='scene'); ap.add_argument('--scenes', default='')
ap.add_argument('--recv-n', type=int, default=8000); ap.add_argument('--ke', type=int, default=32); ap.add_argument('--ke-ground', type=int, default=128)
ap.add_argument('--smooth', default='on'); ap.add_argument('--nd2'); ap.add_argument('--radius', type=float, default=2.0)   # v3: v1tau_smoothn_r2 (on) / v1_tau raw normals (off = v2 regression)
ap.add_argument('--max-copies', type=int, default=32); ap.add_argument('--spacings', default=''); ap.add_argument('--placement')
a = ap.parse_args()
HERE = Path(__file__).resolve().parent
TAU_ROOT = Path(a.tau) if a.tau else Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(a.nine) if a.nine else Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
ND = Path(a.nd) if a.nd else Path(glob.glob('/kaggle/input/**/e_01__surfels.ply', recursive=True)[0]).parent
OUT = Path(a.out or '/kaggle/working'); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
ND2 = Path(a.nd2) if a.nd2 else Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT)); sys.path.insert(0, str(Path(a.lib) if a.lib else OUT)); sys.path.insert(0, str(ND2))
import numpy as np
import numba as nb
import tray32_lib as T
import normals as NM
SMOOTH = a.smooth == 'on'; SUB = 'scene' if SMOOTH else 'regress'
from static import tau_runs as TR, obs
from static.tracer import Scene, transmit, MAXH
from static.geometry import read_ply

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / f'progress_{a.mode}_{a.table}_{SUB if "SUB" in globals() else "x"}.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True); json.dump(o, open(p, 'w', encoding='utf-8'), indent=1, default=float)

KERN = T.bind_tracer(transmit, MAXH)
F = TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; EPS = CFG['eps']; DEX = CFG['dex']; TAU = CFG['tau_leaf']
E_IDS = ['e_01', 'e_04', 'e_05', 'e_07']; _orig_ply = obs.ply_path
obs.ply_path = lambda pid: (ND / f'{pid}__surfels.ply') if pid in E_IDS else _orig_ply(pid)
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}; HASH['tray32_lib.py'] = hashlib.sha256((Path(a.lib) if a.lib else OUT).joinpath('tray32_lib.py').read_bytes()).hexdigest()
SEED = 20261104; TOFF = 0 if a.table == 'A' else 1


@nb.njit(cache=False, parallel=True)
def dome_tau(R, N, own, tmin, two_sided, D, dom, tau_prim, eps, P, kind, gmin, h, dims, start, items):
    n = R.shape[0]; sh = np.zeros(n); un = np.zeros(n)
    for r in nb.prange(n):
        bt = np.empty(MAXH); bi = np.empty(MAXH, np.int64)
        for j in range(D.shape[0]):
            cb = D[j, 0] * N[r, 0] + D[j, 1] * N[r, 1] + D[j, 2] * N[r, 2]
            if two_sided:
                cb = abs(cb)
            elif cb <= 0.0:
                continue
            e = cb * dom; un[r] += e
            Tt = transmit(R[r, 0], R[r, 1], R[r, 2], D[j, 0], D[j, 1], D[j, 2], tmin, 10.0, own[r], P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi)
            if Tt < 0.0:
                Tt = 0.0
            sh[r] += e * Tt
    return sh, un


# ------------------------------------------------------------------ plants (as K5 of the v1_tau reference)
def placed_raw(pid):
    if pid in E_IDS:
        q = obs.Placed(pid, np.zeros(2)); rows = q.s.index; z = np.load(ND / f'{pid}__area_b.npz')
        if not np.array_equal(z['rows'], rows):
            raise RuntimeError(f'{pid}: area rows != opaque rows')
        q.A = z['area_b_mm2'].astype(float); q.tau = np.full(len(rows), TAU); return q
    return TR.placed(pid, np.zeros(2))


def leaf_labels(pid, q):
    rows = q.s.index
    if pid == 'R3-1b':
        leaf = np.where((q.axis == 0) & (q.r8 >= 1), q.r8, -1); return leaf, leaf < 1, leaf < 1
    z = np.load(ND / f'labels__{pid}.npz'); lr = TR._map(z['rows'], z['label'].astype(int), rows); st = TR._map(z['rows'], z['status'].astype(int), rows)
    leaf = np.where((st == 1) & (lr > 0), lr, -1); dz = leaf < 1
    return leaf, (q.axis_like | (q.group < 0)) if pid not in E_IDS else dz, dz


_BASE = {}
def base(pid):
    if pid not in _BASE:
        q = placed_raw(pid); leaf, prim, dz = leaf_labels(pid, q)
        n_, tu_, tv_ = q.normal.copy(), q.s.tu.copy(), q.s.tv.copy(); kept = None
        if SMOOTH:                                   # normal_mode_v2 smooth_all_r2: receiver normal and disc orientation (normal_smoothing_v2 CONTRACT section 1)
            sets = np.where(leaf >= 1, leaf, np.where(q.axis == 1, -10, np.where(q.axis == 2, -20, -30))) if pid == 'R3-1b' else np.where(leaf >= 1, leaf, -30)
            op = read_ply(obs.ply_path(pid))['opacity'][q.s.index].astype(float)
            n_, kept_m, _ = NM.smooth(q.s.centre * 1000.0, q.s.normal, q.A * op, sets, a.radius); tu_, tv_ = NM.reframe(n_, q.s.tu); kept = float(kept_m.mean())
        _BASE[pid] = dict(c0=q.centre.copy(), n=n_, tu=tu_, tv=tv_, kept_raw_share=kept, a=q.s.a.copy(), b=q.s.b.copy(), A=q.A.copy(), tau=np.asarray(q.tau, float).copy(), leaf=leaf, unassigned=prim, dz=dz)
        obs._CACHE.clear()
    return _BASE[pid]


def load_placement(table):
    p = Path(a.placement) if a.placement else (HERE / f'placement_{table}.csv' if (HERE / f'placement_{table}.csv').exists() else next(Path('/kaggle/input').rglob(f'placement_{table}.csv')))
    return list(csv.DictReader(open(p, encoding='utf-8')))[:a.max_copies]


PL = load_placement(a.table); NC = len(PL)
SPACINGS = [float(x) for x in a.spacings.split(',') if x] or list(T.SPACINGS)
SCENES = [s for s in T.scene_list() if (not a.scenes or s['sid'] in a.scenes.split(',')) and s['p'] in SPACINGS]


def assemble(p, only_copy=None, tau_mode='main'):
    """Global disc arrays of all copies (or one copy) at spacing p: P (n,14), tau, copy index, leaf id, unassigned flag, plant area (b) per surfel; plus per-copy ranges."""
    Ps, taus, cpy, leaf, un, A, rng_ = [], [], [], [], [], [], []; start = 0
    for ci, r in enumerate(PL):
        if only_copy is not None and ci != only_copy:
            continue
        b = base(r['plant']); x, y = T.slot_xy(p, int(r['tray']), int(r['row']), int(r['column'])); Rz = T.rot_z(float(r['azimuth_deg']))
        c = b['c0'] @ Rz.T + np.array([x, y, 0.0]); n = b['n'] @ Rz.T; tu = b['tu'] @ Rz.T; tv = b['tv'] @ Rz.T; N_ = len(c)
        P = np.zeros((N_, 14)); P[:, 0:3] = c; P[:, 3:6] = tu; P[:, 6:9] = tv; P[:, 9:12] = n; P[:, 12] = b['a']; P[:, 13] = b['b']
        tau = b['tau'].copy()
        if tau_mode == 'bracket' and r['plant'] in E_IDS:
            tau = np.where(b['dz'], 0.0, tau)
        Ps.append(P); taus.append(tau); cpy.append(np.full(N_, ci)); leaf.append(b['leaf']); un.append(b['unassigned']); A.append(b['A']); rng_.append((start, start + N_)); start += N_
    return dict(P=np.concatenate(Ps), tau=np.concatenate(taus), copy=np.concatenate(cpy), leaf=np.concatenate(leaf), unassigned=np.concatenate(un), A=np.concatenate(A), ranges=rng_)


def receivers(asm, n_r, tag):
    """n_r receivers per copy drawn with probability proportional to area (b) (systematic PPS, seeded); each stands for A_plant / n_r (m2)."""
    sel, w, cc = [], [], []
    for ci, (s0, s1) in enumerate(asm['ranges']):
        Ab = asm['A'][s0:s1]; cum = np.cumsum(Ab); tot = cum[-1]; u0 = np.random.default_rng(SEED + 7919 * ci + 13 * TOFF + tag).random()
        pos = (u0 + np.arange(n_r)) / n_r * tot; idx = np.minimum(np.searchsorted(cum, pos), len(Ab) - 1)
        sel.append(s0 + idx); w.append(np.full(n_r, tot * 1e-6 / n_r)); cc.append(np.full(n_r, ci))
    return np.concatenate(sel), np.concatenate(w), np.concatenate(cc)


def trace_receivers(scene, asm, sel, E, idx, tau_prim):
    Pm, Am, Fm = E; I0 = Fm * (T.M_EXP + 1) / (2 * math.pi); P = asm['P']; R = np.ascontiguousarray(P[sel, 0:3]); N = np.ascontiguousarray(P[sel, 9:12])
    return KERN(R, N, np.ascontiguousarray(sel, np.int64), DEX, True, Pm, Am, I0, T.M_EXP, idx, len(Fm) / idx.shape[1], tau_prim, EPS, scene.P, scene.kind, scene.gmin, scene.h, scene.dims, scene.start, scene.items)


def ground_grid(p, dx=0.025):
    xs = T.tray_x(p); gx = np.arange(xs[0] - p - 0.5 + dx / 2, xs[3] + p + 0.5, dx); gy = np.arange(-0.6 + dx / 2, 0.6, dx); X, Y = np.meshgrid(gx, gy)
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1), dx * dx


def summarize(asm, sel, w, cc, sh, un):
    """Per copy and per leaf sums (a = 1, m2-weighted irradiance sums)."""
    out = dict(copies=[])
    leaf = asm['leaf'][sel]; unas = asm['unassigned'][sel]
    for ci in range(NC):
        m = cc == ci; d = dict(absorbed_a1=float((sh[m] * w[m]).sum()), unshaded_a1=float((un[m] * w[m]).sum()), area_b_m2=float(w[m].sum() * m.sum()), unassigned_absorbed_a1=float((sh[m & unas] * w[m & unas]).sum()), leaves={})
        for k in np.unique(leaf[m]):
            if k >= 1:
                mk = m & (leaf == k); d['leaves'][int(k)] = dict(absorbed_a1=float((sh[mk] * w[mk]).sum()), unshaded_a1=float((un[mk] * w[mk]).sum()), area_b_m2=float(w[mk].sum() * mk.sum()))
        out['copies'].append(d)
    return out


def run_scene_set():
    for p in SPACINGS:
        t0 = time.time(); asm = assemble(p); occ = Scene(asm['P'], np.full(len(asm['P']), 1, np.int8), h=0.003); sel, w, cc = receivers(asm, a.recv_n, 0)
        log(f'p {p}: discs {len(asm["P"])} receivers {len(sel)} scene build {time.time() - t0:.0f} s')
        gpts, dA = ground_grid(p); gN = np.tile([0.0, 0.0, 1.0], (len(gpts), 1)); gown = np.full(len(gpts), -1, np.int64)
        for sc in [s for s in SCENES if s['p'] == p]:
            fp = OUT / SUB / f'{a.table}_{sc["sid"]}.json'
            if fp.exists():
                continue
            t1 = time.time(); fcs = T.faces(sc['shape'], sc['pos'], p, sc['height']); E = T.emitters(fcs); k = sc['seed_k']
            idx = T.draw_idx(len(sel), len(E[2]), a.ke, SEED + 100000 * TOFF + 31 * k)
            sh, un, nv = trace_receivers(occ, asm, sel, E, idx, asm['tau']); res = summarize(asm, sel, w, cc, sh, un)
            absorbed = sum(c['absorbed_a1'] for c in res['copies']); emitted = float(E[2].sum())
            chk = T.emission_checks(fcs, n=100000); up = float(sum(f['flux'] * c['up_fraction'] for f, c in zip(fcs, chk)))
            out_flux = T.outlines_flux(*E, T.M_EXP, p)
            gidx = T.draw_idx(len(gpts), len(E[2]), a.ke_ground, SEED + 100000 * TOFF + 31 * k + 5)
            I0 = E[2] * (T.M_EXP + 1) / (2 * math.pi)
            gsh, gun, _ = KERN(gpts, gN, gown, 1e-7, False, E[0], E[1], I0, T.M_EXP, gidx, len(E[2]) / gidx.shape[1], asm['tau'], EPS, occ.P, occ.kind, occ.gmin, occ.h, occ.dims, occ.start, occ.items)
            gdet = float(T.plane_flux(*E, T.M_EXP, gpts).sum() * dA)
            clos = dict(emitted=emitted, up=up, ground_unshaded_deterministic=gdet, beyond=emitted - up - gdet, floor_shaded=float(gsh.sum() * dA), floor_unshaded_traced=float(gun.sum() * dA),
                        removed_by_plants=float((gun - gsh).sum() * dA), absorbed_a085=A_ABS * absorbed, reflected=T.RHO * absorbed)
            clos['residual_fraction_of_emitted'] = (clos['removed_by_plants'] - (clos['absorbed_a085'] + clos['reflected'])) / emitted
            jdump(dict(scene=sc, table=a.table, smoothing=('smooth_all_r%g' % a.radius) if SMOOTH else 'off', receivers_per_copy=a.recv_n, ke=a.ke, emitted=emitted, emitter_points=len(E[2]), absorbed_a1=absorbed, efficiency_a085=A_ABS * absorbed / emitted,
                       efficiency_a1=absorbed / emitted, outlines_flux=out_flux, outlines_share_of_emitted=out_flux / emitted, absorbed_over_outlines_a085=A_ABS * absorbed / out_flux,
                       overflow_rays=int(nv.sum()), closure=clos, copies=res['copies'], seconds=round(time.time() - t1), engine_hash=HASH), fp)
            log(f'{a.table} {sc["sid"]} p {p} {sc["shape"]} {sc["pos"]} h {sc["height"]}: eff {A_ABS * absorbed / emitted:.4f} resid {clos["residual_fraction_of_emitted"]:+.3f} ovf {int(nv.sum())} {time.time() - t1:.0f} s')
        del occ, asm


A_ABS = T.A_ABS
if a.mode == 'scene':
    run_scene_set()
elif a.mode == 'iso':
    for p in SPACINGS:
        asm = assemble(p); sel, w, cc = receivers(asm, a.recv_n, 0); idx_rows = {}
        small = {}
        for ci in range(NC):
            s0, s1 = asm['ranges'][ci]; sub = assemble(p, only_copy=ci); small[ci] = (Scene(sub['P'], np.full(len(sub['P']), 1, np.int8), h=0.003), sub['tau'], s0)
        log(f'p {p}: isolated scenes built for {NC} copies')
        for sc in [s for s in SCENES if s['p'] == p]:
            fp = OUT / ('iso' if SMOOTH else 'iso_regress') / f'{a.table}_{sc["sid"]}.json'
            if fp.exists():
                continue
            t1 = time.time(); fcs = T.faces(sc['shape'], sc['pos'], p, sc['height']); E = T.emitters(fcs); k = sc['seed_k']
            idx = T.draw_idx(len(sel), len(E[2]), a.ke, SEED + 100000 * TOFF + 31 * k); Pm, Am, Fm = E; I0 = Fm * (T.M_EXP + 1) / (2 * math.pi); iso = []
            for ci in range(NC):
                m = cc == ci; sc_, tau_, s0 = small[ci]; loc = sel[m] - s0; R = np.ascontiguousarray(sc_.P[loc, 0:3]); N = np.ascontiguousarray(sc_.P[loc, 9:12])
                sh, un, nv = KERN(R, N, np.ascontiguousarray(loc, np.int64), DEX, True, Pm, Am, I0, T.M_EXP, np.ascontiguousarray(idx[m]), len(Fm) / idx.shape[1], tau_, EPS, sc_.P, sc_.kind, sc_.gmin, sc_.h, sc_.dims, sc_.start, sc_.items)
                leafc = asm['leaf'][sel[m]]; d = dict(isolated_a1=float((sh * w[m]).sum()), unshaded_a1=float((un * w[m]).sum()), overflow=int(nv.sum()), leaves={})
                for kk in np.unique(leafc):
                    if kk >= 1:
                        mk = leafc == kk; d['leaves'][int(kk)] = dict(isolated_a1=float((sh[mk] * w[m][mk]).sum()), unshaded_a1=float((un[mk] * w[m][mk]).sum()))
                iso.append(d)
            jdump(dict(scene=sc, table=a.table, receivers_per_copy=a.recv_n, ke=a.ke, copies=iso, seconds=round(time.time() - t1), engine_hash=HASH), fp)
            log(f'iso {a.table} {sc["sid"]} {time.time() - t1:.0f} s')
elif a.mode == 'conv':
    p = 0.10; sc = [s for s in T.scene_list() if s['p'] == p and s['shape'] == 'flat' and s['pos'] == 'trays' and s['height'] == 0.30][0]; asm = assemble(p)   # v3: C4 on the 30 cm reference geometry
    occ = Scene(asm['P'], np.full(len(asm['P']), 1, np.int8), h=0.003); fcs = T.faces(sc['shape'], sc['pos'], p, sc['height']); E = T.emitters(fcs); rows = []; prev = None
    for n_r, ke in ((8000, 32), (16000, 64), (32000, 128), (64000, 256)):
        t1 = time.time(); sel, w, cc = receivers(asm, n_r, 1); idx = T.draw_idx(len(sel), len(E[2]), ke, SEED + 777 + ke)
        sh, un, nv = trace_receivers(occ, asm, sel, E, idx, asm['tau']); eff = A_ABS * float((sh * w).sum()) / float(E[2].sum())
        rows.append(dict(receivers_per_copy=n_r, ke=ke, efficiency_a085=eff, rel_change_vs_previous=None if prev is None else eff / prev - 1, seconds=round(time.time() - t1))); prev = eff
        log(f'conv n_r {n_r} ke {ke}: eff {eff:.5f}'); jdump(dict(rows=rows, scene=sc, table=a.table), OUT / f'conv_{a.table}.json')
        if len(rows) >= 2 and abs(rows[-1]['rel_change_vs_previous']) < 0.005:
            break
elif a.mode == 'extra':
    p = 0.10; sc = [s for s in T.scene_list() if s['p'] == p and s['shape'] == 'flat' and s['pos'] == 'trays' and s['height'] == 0.40][0]; fcs = T.faces(sc['shape'], sc['pos'], p, sc['height']); E = T.emitters(fcs)
    asm = assemble(p); occ = Scene(asm['P'], np.full(len(asm['P']), 1, np.int8), h=0.003); sel, w, cc = receivers(asm, a.recv_n, 0); idx = T.draw_idx(len(sel), len(E[2]), a.ke, SEED + 100000 * TOFF + 31 * int(sc['sid'][1:]))
    sh, un, nv = trace_receivers(occ, asm, sel, E, idx, asm['tau']); dir_abs = float((sh * w).sum()); out_flux = T.outlines_flux(*E, T.M_EXP, p)
    # diffuse share 0.6: isotropic dome (72 directions), horizontal irradiance 0.6/0.4 x the mean unshaded direct horizontal irradiance over the outlines
    area_out = 4 * (2 * p) * (4 * p); E_dir_mean = out_flux / area_out; E_dif = 0.6 / 0.4 * E_dir_mean; Lrad = E_dif / math.pi
    Dh = T.fib_sphere(144); Dh = Dh[Dh[:, 2] > 0][:72] if (Dh[:, 2] > 0).sum() >= 72 else Dh[Dh[:, 2] > 0]; nd = len(Dh); dom = 2 * math.pi / nd
    R = np.ascontiguousarray(asm['P'][sel, 0:3]); N = np.ascontiguousarray(asm['P'][sel, 9:12])
    dsh, dun = dome_tau(R, N, np.ascontiguousarray(sel, np.int64), DEX, True, Dh, dom, asm['tau'], EPS, occ.P, occ.kind, occ.gmin, occ.h, occ.dims, occ.start, occ.items)
    dif_abs = Lrad * float((dsh * w).sum())
    res = dict(scene=sc, table=a.table, direct_absorbed_a1=dir_abs, diffuse_absorbed_a1=dif_abs, ratio_with_over_direct_only=(dir_abs + dif_abs) / dir_abs, direct_interception=A_ABS * dir_abs / out_flux,
               diffuse_interception=A_ABS * dif_abs / (E_dif * area_out), dome_directions=nd, diffuse_horizontal_irradiance=E_dif, outline_area_m2=area_out)
    # e_* bracket: tau 0 on unassigned e_* surfels
    asm_b = assemble(p, tau_mode='bracket'); sh_b, un_b, nv_b = trace_receivers(occ, asm_b, sel, E, idx, asm_b['tau']); res['bracket_efficiency_a085'] = A_ABS * float((sh_b * w).sum()) / float(E[2].sum()); res['main_efficiency_a085'] = A_ABS * dir_abs / float(E[2].sum())
    res['bracket_change_pct'] = 100 * (res['bracket_efficiency_a085'] / res['main_efficiency_a085'] - 1); jdump(res, OUT / f'extra_{a.table}.json'); log(f'extra: {res}')
jdump(dict(engine_hash=HASH, wall_seconds=round(time.time() - T0), mode=a.mode, table=a.table, scenes=[s['sid'] for s in SCENES]), OUT / f'SUMMARY_{a.mode}_{a.table}_{SUB}.json'); log('DONE')
'''
(OUTD / 'tray32_lib.py').write_text(LIB, encoding='utf-8')
for MODE, EXTRA in [('scene', [])]:
    sys.argv = ['tray32_main.py', '--table', 'B', '--mode', MODE, '--out', '/kaggle/working', '--lib', '/kaggle/working'] + EXTRA
    exec(compile(MAIN, 'tray32_main.py', 'exec'), {'__name__': '__main__', '__file__': '/kaggle/working/tray32_main.py'})
