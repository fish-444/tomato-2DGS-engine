"""v1_tau plant traces (engine_static_tau_20261031_01/CONTRACT.md): `obs.trace_plants` without sensors, with per-primitive tau.
Every placed plant carries `tau` (per surfel, same order as its prims). tau all zero = v1 (byte-identical, test R0)."""
from __future__ import annotations

import numpy as np

from . import light as L
from .obs import POT_SIDE, unshaded_U
from .tracer import Scene, trace_cabinet, trace_cabinet_tau


def build_scene(scene_set):
    Ps, ks, taus, owner_offset, start = [], [], [], {}, 0
    for q in scene_set:
        P, k = q.prims(); Ps.append(P); ks.append(k); taus.append(np.asarray(q.tau, float)); owner_offset[q.pid] = start; start += len(P)
    return Scene(np.concatenate(Ps), np.concatenate(ks), h=0.003), np.concatenate(taus), owner_offset


def trace_plants_tau(placed, scene_set, seed, eps, dz_mm=0.0, dy_mm=0.0, delta=0.005, field=None, tau_scale=1.0, use_v1=False, built=None):
    """Per placed plant: parts (n, 4), overflow counts, U (measured field, model extension above it). tau_scale multiplies every
    surfel's tau (T3 sensitivity: 0 / 0.05/0.07 / 0.07/0.07 / 0.10/0.07). use_v1 runs the untouched v1 kernel instead (R0)."""
    scene, tau_prim, owner_offset = built if built is not None else build_scene(scene_set)
    tp = tau_prim * tau_scale
    m = L.cos_m(); bars = L.bar_points(seed, dz_mm=dz_mm, dy_mm=dy_mm); bars_ref = L.bar_points(seed); dm = L.dome(seed)
    out = {}
    for q in placed:
        own = q.own_local + owner_offset[q.pid]
        d = delta if q.delta is None else q.delta
        if use_v1:
            _, _, _, parts = trace_cabinet(scene, q.centre, q.normal, True, own, d, bars, m, dm, L.DIFFUSE); novf = np.zeros(len(q.centre), np.int64)
        else:
            _, _, _, parts, novf = trace_cabinet_tau(scene, q.centre, q.normal, True, own, d, bars, m, dm, L.DIFFUSE, tp, eps)
        rec = dict(parts=parts, novf=novf)
        if field is not None:
            rec['U'], rec['Uext'], rec['ok'] = unshaded_U(field, q.centre, bars_ref, None if (dz_mm == 0 and dy_mm == 0) else bars, m)
        out[q.pid] = rec
    return out


def probe_ratios(scene, tau_prim, eps, above, below, bar_pts, cos_m, dome, f_diffuse):
    """Horizontal one-sided (+z) cosine probes: incident relative light just above and just below leaves, and the ratio below/above.
    `above`, `below`: (n, 3) metres, paired. Returns dict of arrays for v1 (tau = 0) and v1_tau (tau_prim)."""
    Nz = np.tile([0.0, 0.0, 1.0], (len(above), 1)); own = np.full(len(above), -1)
    res = {}
    for name, tp in (('v1', np.zeros_like(tau_prim)), ('v1_tau', tau_prim)):
        ka = trace_cabinet_tau(scene, above, Nz, False, own, 1e-7, bar_pts, cos_m, dome, f_diffuse, tp, eps)[0]
        kb = trace_cabinet_tau(scene, below, Nz, False, own, 1e-7, bar_pts, cos_m, dome, f_diffuse, tp, eps)[0]
        res[name] = dict(above=ka, below=kb, ratio=np.where(ka > 0, kb / np.maximum(ka, 1e-30), np.nan))
    return res
