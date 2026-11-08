"""Engine v2 measures (CONTRACT.md §3) from the downloaded kernel outputs kout/<kernel>/out/<case>/{status,summary}.json.
import analyze as A; R = A.load(); m = A.metrics(R['C1_m00']); A.block_table('C1', 'B0', R) ..."""
from __future__ import annotations

import json, math, statistics as st
from collections import defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent; DAG0 = 11.07; D71 = 71


def load(kernels=None) -> dict:
    R = {}
    for p in sorted((H / 'kout').glob('*/out/*/status.json')):
        if kernels and p.parents[2].name not in kernels: continue
        s = json.load(open(p)); sp = p.parent / 'summary.json'
        if s.get('status') == 'verified' and sp.exists():
            try:
                R[p.parent.name] = json.load(open(sp)); R[p.parent.name]['_status'] = s
            except ValueError:   # 2026-11-06: a partial download left an empty summary.json; skip it loudly (re-download fixes it)
                print('WARNING unreadable summary', sp); R.setdefault('_failed', {})[p.parent.name] = dict(status='unreadable_summary')
        else:
            R.setdefault('_failed', {})[p.parent.name] = s
    return R


def ctl(oid: str) -> str:
    q = oid.split('/'); return q[1] if len(q) > 1 and q[1][:1] == 'S' and q[1][1:].isdigit() else 'M'


def metrics(S: dict, day: int = D71) -> dict:
    d = next(r for r in S['daily'] if r['day'] == day); comp = S['compounds']; dr = defaultdict(dict)
    for x in S['draws']: dr[x['key']][x['id']] = x['value']
    main = [c for c in comp if c['ctl'] == 'M']; side = [c for c in comp if c['ctl'] != 'M']
    tri = defaultdict(list)
    for t in S['truss_init']: tri[ctl(t['organ'])].append(t['day'])
    stops = defaultdict(list)
    for e in S['events']:
        if e['event'] == 'determinate_stop': stops[ctl(e['organ'])].append(e['day'])
    last10 = [r.get('gross_gC', 0) - r.get('downregulated_gC', 0) for r in S['daily'] if day - 10 < r['day'] <= day]
    lw = [v for c in comp for v in c['lw']]
    sides = sorted({c['ctl'] for c in side}) or []
    n_side_axes = len(S.get('side_axes', []))
    m = dict(leaf_area_cm2=d['leaf_area_mm2'] / 100, dm_g=d['dm_g'], height_mm=d['height_mm'], absorbed_rel=d['absorbed_rel'],
             main_compounds=d['compounds'].get('M', 0), main_A0=d['compounds_A0'], side_compounds=sum(v for k, v in d['compounds'].items() if k != 'M'),
             side_leaves_per_axis=(sum(v for k, v in d['compounds'].items() if k != 'M') / n_side_axes) if n_side_axes else 0.0, side_axes=n_side_axes,
             trusses=sum(d['trusses'].values()), trusses_main=d['trusses'].get('M', 0), flowers=d['flowers'], fruits=d['fruits'], flowers_and_fruits=d['flowers'] + d['fruits'],
             leaflets_per_main=st.mean([c['leaflets'] for c in main]) if main else float('nan'), leaflets_per_side=st.mean([c['leaflets'] for c in side]) if side else float('nan'),
             theta0_main_mean=st.mean([c['elevation'] for c in main]) if main else float('nan'), theta0_rank1=next((c['elevation'] for c in main if c['rank'] == 1), float('nan')),
             bend_mean=st.mean([c['bend_sum'] for c in comp]) if comp else float('nan'),
             internode_draw_mean_mm=st.mean(dr['INTERNODE_MM'].values()) if dr.get('INTERNODE_MM') else float('nan'),
             internode_len_mean_mm=st.mean([i['len_mm'] for i in S['internodes']]) if S['internodes'] else float('nan'),
             max_leaflet_len_mm=max([c['max_leaflet_len_mm'] for c in comp], default=0), lw_min=min(lw, default=float('nan')), lw_max=max(lw, default=float('nan')),
             first_truss_day=min(tri['M']) if tri['M'] else float('nan'), first_truss_dag=(min(tri['M']) - DAG0) if tri['M'] else float('nan'),
             main_stop_dag=(min(stops['M']) - DAG0) if stops['M'] else float('nan'),
             side_stop_dag=[s - DAG0 for k in sorted(stops) if k != 'M' for s in stops[k]],
             side_created_dag=[a['day'] - DAG0 for a in S.get('side_axes', [])],
             accepted_gross_last10=st.mean(last10) if last10 else float('nan'),
             potential_gross_cum=sum(r.get('gross_gC', 0) for r in S['daily'] if r['day'] <= day), downregulated_cum=sum(r.get('downregulated_gC', 0) for r in S['daily'] if r['day'] <= day),
             reserve_end=d.get('reserve_end_gC', float('nan')),
             cap_first_dag=next((r['day'] - DAG0 for r in S['daily'] if r.get('downregulated_gC', 0) > 0), float('nan')),
             max_mass_ratio=S['max_mass_ratio'], n_compounds=len(comp),
             main_truss_node=dr.get('MAIN_TRUSS_NODE', {}).get('p1'), main_total=dr.get('MAIN_TOTAL_LEAVES', {}).get('p1'),
             divergence_draws=list(dr.get('DIVERGENCE_DEG', {}).values()), handedness=dr.get('HANDEDNESS', {}).get('p1'))
    return m


def q(vals, p):
    v = sorted(x for x in vals if x == x)
    if not v: return float('nan')
    k = (len(v) - 1) * p; f = math.floor(k); c = min(f + 1, len(v) - 1); return v[f] + (v[c] - v[f]) * (k - f)


SCALARS = ['leaf_area_cm2', 'dm_g', 'height_mm', 'absorbed_rel', 'main_compounds', 'main_A0', 'side_compounds', 'side_leaves_per_axis', 'trusses', 'flowers_and_fruits', 'fruits',
           'leaflets_per_main', 'leaflets_per_side', 'theta0_main_mean', 'bend_mean', 'internode_draw_mean_mm', 'internode_len_mean_mm', 'max_leaflet_len_mm', 'lw_max',
           'first_truss_dag', 'main_stop_dag', 'accepted_gross_last10', 'potential_gross_cum', 'downregulated_cum', 'reserve_end', 'cap_first_dag', 'max_mass_ratio']


def arm(R, a, day=D71):
    return {int(k.rsplit('_m', 1)[1]): metrics(v, day) for k, v in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == a}


def block_table(after, before, R, day=D71):
    A, B = arm(R, after, day), arm(R, before, day); seeds = sorted(set(A) & set(B)); rows = []
    for k in SCALARS:
        a = [A[s][k] for s in seeds]; b = [B[s][k] for s in seeds]
        pc = [(x / y - 1) * 100 for x, y in zip(a, b) if y and y == y and x == x]
        rows.append(dict(measure=k, before_median=q(b, .5), before_p05=q(b, .05), before_p95=q(b, .95), after_median=q(a, .5), after_p05=q(a, .05), after_p95=q(a, .95),
                         paired_pct_median=q(pc, .5) if pc else float('nan'), n_seeds=len(seeds)))
    return rows
