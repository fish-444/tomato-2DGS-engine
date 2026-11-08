"""Shared v1_tau run helpers for the Kaggle kernels (engine_static_tau_20261031_01/CONTRACT.md). Data come from the nine-individual
bundle (surfels, P1 silhouette areas, groups, field planes) plus this bundle (R3-1b axis_kind / R8_pc labels, config)."""
from __future__ import annotations

import csv
import json
import math
import time

import numpy as np

from . import light as L
from . import obs
from .tau_obs import build_scene, trace_plants_tau

SEEDS = (101, 202)
TAU_LEVELS = {'tau0.05': 0.05 / 0.07, 'tau0.07': 1.0, 'tau0.10': 0.10 / 0.07}       # scale on the per-surfel tau (0.07 leaf, 0 stem)
CFG = {}
SCEN = {'lamp_250mm': dict(dz=-50), 'lamp_350mm': dict(dz=50), 'lateral_y_-50mm': dict(dy=-50), 'lateral_y_+50mm': dict(dy=50),
        'spacing_90mm': dict(pitch=0.090), 'spacing_120mm': dict(pitch=0.120)}
TRAY = ['n_04', 't2_02', 'n_01', 't2_05']


def setup(nine_root, tau_root):
    CFG.update(json.load(open(tau_root / 'tau_config.json', encoding='utf-8')))
    consts = json.load(open(nine_root / 'light_consts.json'))
    L.cos_m = lambda *a, **k: consts['cos_m']
    r31 = nine_root / 'R3-1b'
    obs.R31B_PLY = r31 / 'surfels.ply'; obs.R31B_IDS = r31 / 'surfel_leaf_ids.csv'
    obs.ply_path = lambda pid: obs.R31B_PLY if pid == 'R3-1b' else nine_root / pid / 'surfels.ply'
    F = obs.FieldFast.__new__(obs.FieldFast); z = np.load(nine_root / 'field_planes.npz')
    F.planes = {70.0: (z['xs_70'], z['ys_70'], z['v_70']), 150.0: (z['xs_150'], z['ys_150'], z['v_150'])}
    CFG['nine_root'] = nine_root; CFG['tau_root'] = tau_root; CFG['field'] = F
    return F


def _map(rows_src, vals, rows):
    order = np.argsort(rows_src); idx = order[np.searchsorted(rows_src[order], rows)]
    if not np.array_equal(rows_src[idx], rows):
        raise RuntimeError('row sets differ')
    return vals[idx]


def placed(pid, pos=None):
    nine, tau_root = CFG['nine_root'], CFG['tau_root']
    q = obs.Placed(pid, obs.SCENE_XY if pos is None else np.asarray(pos, float)); rows = q.s.index
    sil = {}
    with open(nine / pid / 'static_surfels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith('#')):
            sil[int(r['surfel_row'])] = float(r['area_b_mm2'])
    q.A = np.array([sil[r] for r in rows.tolist()])
    z = 1000 * q.centre[:, 2]; q.h = z - np.percentile(z, 0.5)
    q.masks = {}
    if pid == 'R3-1b':
        lr = np.load(tau_root / 'labels_rachis.npz')
        ax = _map(lr['rows'], lr['axis_kind'].astype(int), rows); r8 = _map(lr['rows'], lr['label'].astype(int), rows)
        q.tau = np.where(ax == 1, 0.0, CFG['tau_leaf']); q.axis = ax; q.r8 = r8; q.vp = q.s.labels
        q.masks.update({'vp_leaf_1': q.vp == 1, 'vp_leaf_7': q.vp == 7})
        for Lf in range(1, 9):
            q.masks[f'r8_blade_{Lf}'] = (ax == 0) & (r8 == Lf)
        q.group_key = ax * 100 + r8                     # T2 area renormalisation groups
    else:
        g = np.load(nine / pid / 'groups.npz', allow_pickle=False)
        axl = _map(g['rows'], g['axis_like'].astype(bool), rows); grp = _map(g['rows'], g['group'].astype(int), rows)
        q.tau = np.where(axl, 0.0, CFG['tau_leaf']); q.axis_like = axl; q.group = grp
        click = {}
        p = nine / 'user_3d_group_click.jsonl'
        if p.exists():
            for line in open(p, encoding='utf-8'):
                if line.strip():
                    r = json.loads(line)
                    if r.get('individual') == pid and r.get('unit') == 'group':
                        if r.get('action') == 'leaf':
                            click[int(r['unit_id'])] = int(r['leaf'])
                        else:
                            click.pop(int(r['unit_id']), None)
        if click:
            q.masks['click_leaf_1_lower_bound'] = np.isin(grp, [k for k, v in click.items() if v == 1])
        q.group_key = grp
    return q


def pm(x):
    x = [float(v) for v in x]
    m = float(np.mean(x))
    return dict(mean=m, sigma=abs(x[0] - x[1]) / 2 if len(x) > 1 else 0.0)


def metrics(q, recs, a_values=(1.0, 0.85)):
    """recs: {seed: rec}. Plant total, self-shading, height thirds, leaves; absorbed = U K A x a."""
    A = q.A; top = np.percentile(q.h, 99.5)
    thirds = {'lower': q.h < top / 3, 'middle': (q.h >= top / 3) & (q.h < 2 * top / 3), 'upper': q.h >= 2 * top / 3}
    per = []
    for sd in SEEDS:
        r = recs[sd]; K = obs.kernel(r['parts']); Ku = obs.kernel(r['parts'], shaded=False); U = r['Uext']
        ab = U * K * A; abu = U * Ku * A
        d = dict(total=ab.sum(), self=1 - ab.sum() / abu.sum(), mean_light=ab.sum() / A.sum(), ovf=int(r['novf'].sum()))
        for k, m in thirds.items():
            d[f'{k}_total'] = ab[m].sum(); d[f'{k}_mean_light'] = ab[m].sum() / A[m].sum(); d[f'{k}_share'] = ab[m].sum() / ab.sum()
        for k, m in q.masks.items():
            d[f'leaf:{k}'] = ab[m].sum(); d[f'leafmean:{k}'] = ab[m].sum() / max(A[m].sum(), 1e-30)
        per.append(d)
    out = {k: pm([p[k] for p in per]) for k in per[0] if k != 'ovf'}
    out['overflow_rays'] = sum(p['ovf'] for p in per)
    out['absorbed_by_a'] = {f'{a:g}': pm([p['total'] * a for p in per]) for a in a_values}
    return out


def kmean(recs, shaded=True):
    return np.mean([obs.kernel(recs[sd]['parts'], shaded=shaded) for sd in SEEDS], 0)


def trace(q_list, scene_set, model, built=None, dz=0.0, dy=0.0):
    """model: 'v1' (untouched v1 kernel) | 'tau0' (tau kernel, tau=0) | 'tau0.05' | 'tau0.07' | 'tau0.10'."""
    out = {}
    for sd in SEEDS:
        if model == 'v1':
            r = trace_plants_tau(q_list, scene_set, sd, CFG['eps'], dz, dy, CFG['dex'], CFG['field'], use_v1=True, built=built)
        else:
            sc = 0.0 if model == 'tau0' else TAU_LEVELS[model]
            r = trace_plants_tau(q_list, scene_set, sd, CFG['eps'], dz, dy, CFG['dex'], CFG['field'], tau_scale=sc, built=built)
        for pid, rec in r.items():
            out.setdefault(pid, {})[sd] = rec
    return out


def tray_positions(pitch):
    h = pitch / 2
    return [obs.SCENE_XY + np.array(s) * h for s in ((-1, -1), (1, -1), (-1, 1), (1, 1))]


def classify(dv1, dtau):
    same = (dv1 > 0) == (dtau > 0) and dv1 != 0
    return 'stable' if same and abs(abs(dtau) - abs(dv1)) <= 0.25 * abs(dv1) else 'optics_sensitive'


def scenarios(kind, log=print):
    """kind 'r31b' (alone, lamp scenarios) or 'tray' (60 mm baseline + all six). Relative delta = scenario/baseline - 1 (a cancels)."""
    res = {'v1': {}, 'tau0.07': {}}
    cfgs = [('baseline', {})] + [(n, s) for n, s in SCEN.items() if kind == 'tray' or 'pitch' not in s]
    cache = {}
    for name, sp in cfgs:
        pitch = sp.get('pitch', 0.060)
        if kind == 'tray':
            if pitch not in cache:
                qs = [placed(pid, pos) for pid, pos in zip(TRAY, tray_positions(pitch))]; cache[pitch] = (qs, build_scene(qs))
            qs, built = cache[pitch]
        else:
            if 'r31b' not in cache:
                q = placed('R3-1b'); cache['r31b'] = ([q], build_scene([q]))
            qs, built = cache['r31b']
        for model in ('v1', 'tau0.07'):
            t0 = time.time()
            r = trace(qs, qs, model, built, sp.get('dz', 0.0), sp.get('dy', 0.0))
            vals = {}
            for q in qs:
                m = metrics(q, r[q.pid]); vals[q.pid] = m['total']['mean']
                for k in q.masks:
                    vals[f'{q.pid}:{k}'] = m[f'leaf:{k}']['mean']
            if kind == 'tray':
                vals['tray_total'] = sum(vals[p] for p in TRAY)
            res[model][name] = vals
            log(f'scenario {kind} {name} {model} {time.time() - t0:.0f} s')
    table = {}
    for name, _ in cfgs[1:]:
        for unit in res['v1']['baseline']:
            d1 = res['v1'][name][unit] / res['v1']['baseline'][unit] - 1; dt = res['tau0.07'][name][unit] / res['tau0.07']['baseline'][unit] - 1
            table[f'{name}|{unit}'] = dict(rel_delta_v1=d1, rel_delta_tau=dt, cls=classify(d1, dt))
    return dict(raw=res, table=table, stable=sum(1 for v in table.values() if v['cls'] == 'stable'), total=len(table))
