"""Local code tests (CONTRACT §7 C1, C2; plus layout sanity). One plant, ~3000 receivers, a few minutes, < 1 GB. python -X utf8 test_tray32.py"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import numba as nb

HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
sys.path.insert(0, str(B / 'engine_static_tau_20261031_01/kaggle_data')); sys.path.insert(0, str(HERE))
import tray32_lib as T
from static import tau_runs as TR, obs
from static.tau_obs import build_scene
from static.tracer import Scene, trace_lamps, transmit, MAXH

out = {}
# ---------------------------------------------------------------- C2 emission integral
worst = 0.0; rows = []
for sc in T.scene_list():
    fc = T.faces(sc['shape'], sc['pos'], sc['p'], sc['height']); chk = T.emission_checks(fc, n=100000); Pm, Am, Fm = T.emitters(fc)
    tot_decl = sum(f['flux'] for f in fc); tot_num = sum(c['flux_numeric'] for c in chk); tot_pts = float(Fm.sum()); S = len(Fm)
    worst = max(worst, abs(tot_num / 4 - 1), abs(tot_pts / 4 - 1), abs(tot_decl / 4 - 1)); rows.append((sc['sid'], S, tot_decl, round(tot_num, 5), round(tot_pts, 6), round(sum(f['flux'] * c['up_fraction'] for f, c in zip(fc, chk)), 5)))
out['C2'] = dict(worst_rel_dev=worst, passed=bool(worst <= 0.005), samples_per_scene=sorted({r[1] for r in rows}), up_fraction_by_scene={r[0]: r[5] for r in rows})
print('C2', out['C2']['worst_rel_dev'], out['C2']['passed'], out['C2']['samples_per_scene'])
# ---------------------------------------------------------------- C1 point-source limit
kern = T.bind_tracer(transmit, MAXH)
TR.setup(B / 'engine_static_nine_20261028_01/kaggle_data', B / 'engine_static_tau_20261031_01/kaggle_data'); CFG = TR.CFG
q = TR.placed('R3-1b'); rng = np.random.default_rng(1); sel = np.sort(rng.choice(len(q.A), 3000, replace=False))
built = build_scene([q]); scene, tau_full, _ = built
R = q.centre[sel]; N = q.normal[sel]; own = sel.astype(np.int64)
lamp = np.array([[obs.SCENE_XY[0] + 0.01, obs.SCENE_XY[1] - 0.02, 5.0]]); Iint = np.array([7.3])
E_pos = lamp; E_axis = np.array([[0.0, 0.0, -1.0]]); idx = np.zeros((len(sel), 1), np.int32); I0 = Iint.copy()
for name, tau in (('tau0', np.zeros_like(tau_full)), ('tau0.07', tau_full)):
    sh, un, nv = kern(R, N, own, CFG['dex'], True, E_pos, E_axis, I0, 0.0, idx, 1.0, tau, CFG['eps'], scene.P, scene.kind, scene.gmin, scene.h, scene.dims, scene.start, scene.items)
    if name == 'tau0':
        sh_ref, un_ref = trace_lamps(scene, R, N, True, own, CFG['dex'], lamp, Iint)
    else:
        @nb.njit(parallel=True)
        def point_lamps_tau(R, N, two_sided, own, tmin, lamps, intensity, tau_prim, eps, P, kind, gmin, h, dims, start, items):
            n = R.shape[0]; shaded = np.zeros(n); unshaded = np.zeros(n)
            for r in nb.prange(n):
                bt = np.empty(MAXH); bi = np.empty(MAXH, np.int64)
                for j in range(lamps.shape[0]):
                    vx = lamps[j, 0] - R[r, 0]; vy = lamps[j, 1] - R[r, 1]; vz = lamps[j, 2] - R[r, 2]
                    dist = math.sqrt(vx * vx + vy * vy + vz * vz); ux, uy, uz = vx / dist, vy / dist, vz / dist
                    cb = abs(ux * N[r, 0] + uy * N[r, 1] + uz * N[r, 2]); e = intensity[j] * cb / (dist * dist); unshaded[r] += e
                    Tt = transmit(R[r, 0], R[r, 1], R[r, 2], ux, uy, uz, tmin, dist, own[r], P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi)
                    if Tt < 0.0:
                        Tt = 0.0
                    shaded[r] += e * Tt
            return shaded, unshaded
        sh_ref, un_ref = point_lamps_tau(R, N, True, own, CFG['dex'], lamp, Iint, tau, CFG['eps'], scene.P, scene.kind, scene.gmin, scene.h, scene.dims, scene.start, scene.items)
    d_sh = float(np.max(np.abs(sh - sh_ref) / np.maximum(np.abs(sh_ref), 1e-300))); d_un = float(np.max(np.abs(un - un_ref) / np.maximum(np.abs(un_ref), 1e-300)))
    out[f'C1_{name}'] = dict(max_rel_shaded=d_sh, max_rel_unshaded=d_un, passed=bool(d_sh <= 1e-9 and d_un <= 1e-9), shaded_fraction=float(sh.sum() / un.sum()), overflow=int(nv.sum()))
    print('C1', name, out[f'C1_{name}'])
# layout sanity: nearest plant centres across a gap = p + 0.10, tray size 2p x 4p, 32 slots distinct
for p in T.SPACINGS:
    xs = T.tray_x(p); out[f'layout_p{p}'] = dict(tray_centre_spacing=xs[1] - xs[0], expected=2 * p + 0.10, gap_nearest_plants=T.slot_xy(p, 2, 1, 1)[0] - T.slot_xy(p, 1, 1, 2)[0], expected_gap_plants=p + 0.10)
print({k: v for k, v in out.items() if k.startswith('layout')})
json.dump(out, open(HERE / 'LOCAL_TESTS.json', 'w'), indent=1, default=float)
