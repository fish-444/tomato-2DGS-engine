"""Local diagnostic (small): why does the floor-removed flux exceed absorbed + reflected? One plant (R3-1b or n_03) alone under a distant point lamp (nearly uniform irradiance):
ratio = floor removal (traced, tau chain) / (0.93 * sum over ALL surfels of area (b) * E * T * |cos|). Also the same with area (a) (sum of disc areas) and with the disc-union footprint."""
import math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
sys.path.insert(0, str(B / 'engine_static_tau_20261031_01/kaggle_data')); sys.path.insert(0, str(HERE))
import tray32_lib as T
from static import tau_runs as TR, obs
from static.tau_obs import build_scene
from static.tracer import transmit, MAXH
kern = T.bind_tracer(transmit, MAXH)
TR.setup(B / 'engine_static_nine_20261028_01/kaggle_data', B / 'engine_static_tau_20261031_01/kaggle_data'); CFG = TR.CFG
for pid in sys.argv[1:] or ['R3-1b']:
    q = TR.placed(pid); scene, tau, _ = build_scene([q]); n = len(q.A)
    lamp = np.array([[obs.SCENE_XY[0], obs.SCENE_XY[1], 3.0]]); E_axis = np.array([[0.0, 0.0, -1.0]]); I0 = np.array([1.0]); idx = np.zeros((n, 1), np.int32)
    R = np.ascontiguousarray(q.centre); N = np.ascontiguousarray(q.normal)
    sh, un, nv = kern(R, N, np.arange(n, dtype=np.int64), CFG['dex'], True, lamp, E_axis, I0, 0.0, idx, 1.0, tau, CFG['eps'], scene.P, scene.kind, scene.gmin, scene.h, scene.dims, scene.start, scene.items)
    Ab = q.A * 1e-6; Aa = math.pi * 1.5 ** 2 * np.ones(n)   # placeholder for (a) below
    absb = float((sh * Ab).sum()); absb_un = float((un * Ab).sum())
    dx = 0.002; c = obs.SCENE_XY; half = 0.16; gx = np.arange(c[0] - half + dx / 2, c[0] + half, dx); gy = np.arange(c[1] - half + dx / 2, c[1] + half, dx); X, Y = np.meshgrid(gx, gy)
    G = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.0)], 1); gN = np.tile([0.0, 0.0, 1.0], (len(G), 1)); gi = np.zeros((len(G), 1), np.int32)
    gsh, gun, _ = kern(G, gN, np.full(len(G), -1, np.int64), 1e-7, False, lamp, E_axis, I0, 0.0, gi, 1.0, tau, CFG['eps'], scene.P, scene.kind, scene.gmin, scene.h, scene.dims, scene.start, scene.items)
    removed = float((gun - gsh).sum() * dx * dx)
    # projected union area of the discs seen from above (shadow footprint with tau 0): fraction of ground points with T < 1 times area
    shadow_frac = float(((gun - gsh) > 1e-12).mean()); shadow_area = shadow_frac * (2 * half) ** 2
    print(pid, 'surfels', n, 'area_b m2', round(float(Ab.sum()), 5), 'shadow footprint m2 (any crossing)', round(shadow_area, 5), 'mean (1-T) over shadowed', round(float((1 - gsh[(gun - gsh) > 1e-12] / gun[(gun - gsh) > 1e-12]).mean()), 3))
    print('   absorbed(a=1) over all surfels', round(absb, 6), 'unshaded-sum', round(absb_un, 6), 'floor removed', round(removed, 6), 'removed / (0.93*absorbed_a1)', round(removed / (0.93 * absb), 3), 'removed / (E*shadow area)', round(removed / (gun.mean() * shadow_area), 3))
