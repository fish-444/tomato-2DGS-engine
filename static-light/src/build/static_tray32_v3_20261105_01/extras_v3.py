"""v3 additions after aggregate.py: regression vs v2 (smoothing off), near_luminaire column, height_response.csv, smoothing_vs_v2.csv. python -X utf8 extras_v3.py"""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); V2 = HERE.parents[0] / 'static_tray32_v2_20261104_01'
import tray32_lib as T
TAG = 'light_model_v1tau_smoothn_r2; assumed_luminaire_model; repeated_placement; relative_light; conditional_on_unvalidated_scale; skeleton_labels_ungraded; scenario_comparison_not_validation; absolute_efficiency_uncertain_up_to_1.9x'
A_ABS = T.A_ABS; SC = {s['sid']: s for s in T.scene_list()}


def rd(p):
    return list(csv.DictReader(l for l in open(p, encoding='utf-8') if not l.startswith('#')))


def wcsv(name, rows, note=''):
    if rows:
        with open(HERE / name, 'w', encoding='utf-8', newline='') as f:
            f.write(f'# {TAG}{"; " + note if note else ""}\n'); w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def J(p):
    return json.load(open(p, encoding='utf-8'))


def find(sub, table, sid, root=HERE):
    for f in glob.glob(str(root / 'out_t*' / sub / f'{table}_{sid}.json')):
        return J(f)
    return None


r = lambda x, n=5: None if x is None else round(float(x), n)
# ---- regression (smoothing off, 40 cm) vs v2
reg = []
for sid3 in ('S26', 'S30', 'S34', 'S38', 'S42', 'S46'):
    a = find('regress', 'A', sid3); b = find('scene', 'A', SC[sid3]['v2_sid'], V2)
    if a and b:
        d = max(abs(x['absorbed_a1'] - y['absorbed_a1']) for x, y in zip(a['copies'], b['copies']))
        reg.append(dict(v3_sid=sid3, v2_sid=SC[sid3]['v2_sid'], shape=SC[sid3]['shape'], position=SC[sid3]['pos'], v3_off_absorbed_a1=a['absorbed_a1'], v2_absorbed_a1=b['absorbed_a1'],
                        total_diff=a['absorbed_a1'] - b['absorbed_a1'], max_copy_abs_diff=d, identical=bool(a['absorbed_a1'] == b['absorbed_a1'] and d == 0.0)))
json.dump(dict(rows=reg, all_identical=bool(reg and all(x['identical'] for x in reg))), open(HERE / 'regression_vs_v2.json', 'w'), indent=1)
# ---- near_luminaire column (clearance; no scene was flagged, the column is kept)
cl = rd(HERE / 'clearance.csv'); flag = {(c['table'], c['sid']) for c in cl if c['near_luminaire'] == 'True' or c['inside_luminaire_volume'] == 'True'}
flagc = {(c['table'], c['sid'], c['copy']) for c in cl if c['near_luminaire'] == 'True' or c['inside_luminaire_volume'] == 'True'}
for name, key in (('efficiency_by_scene.csv', None), ('plants_by_scene.csv', 'copy')):
    p = HERE / name
    if p.exists():
        head = [l for l in open(p, encoding='utf-8') if l.startswith('#')]; rows = rd(p)
        for x in rows:
            x['near_luminaire'] = ((x['table'], x['sid'], x[key]) in flagc) if key else ((x['table'], x['sid']) in flag)
        with open(p, 'w', encoding='utf-8', newline='') as f:
            f.writelines(head); w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# ---- height response
E = rd(HERE / 'efficiency_by_scene.csv'); hr = []
for t in sorted({e['table'] for e in E}):
    for sh in T.SHAPES:
        for pos in T.POSITIONS:
            for p in T.SPACINGS:
                v = {float(e['height_m']): float(e['efficiency_a085']) for e in E if e['table'] == t and e['shape'] == sh and e['position'] == pos and float(e['spacing_m']) == p}
                if len(v) == 4:
                    row = dict(table=t, shape=sh, position=pos, spacing_m=p)
                    for h in T.HEIGHTS:
                        row[f'eff_{int(h * 100)}cm'] = r(v[h]); row[f'ratio_to_40cm_{int(h * 100)}cm'] = r(v[h] / v[0.40])
                    hr.append(row)
    for sh in T.SHAPES:                                                       # gaps / trays ratio per height and the crossing height
        for p in T.SPACINGS:
            g = {h: next(float(e['efficiency_a085']) for e in E if e['table'] == t and e['shape'] == sh and e['position'] == 'gaps' and float(e['spacing_m']) == p and float(e['height_m']) == h) for h in T.HEIGHTS}
            tr = {h: next(float(e['efficiency_a085']) for e in E if e['table'] == t and e['shape'] == sh and e['position'] == 'trays' and float(e['spacing_m']) == p and float(e['height_m']) == h) for h in T.HEIGHTS}
            rat = [g[h] / tr[h] for h in T.HEIGHTS]; cross = 'none'
            for (h0, r0), (h1, r1) in zip(zip(T.HEIGHTS, rat), zip(T.HEIGHTS[1:], rat[1:])):
                if (r0 - 1) * (r1 - 1) < 0:
                    cross = round(h0 + (1 - r0) * (h1 - h0) / (r1 - r0), 3); break
            row = dict(table=t, shape=sh, position='gaps_over_trays', spacing_m=p)
            for h, x in zip(T.HEIGHTS, rat):
                row[f'eff_{int(h * 100)}cm'] = r(x); row[f'ratio_to_40cm_{int(h * 100)}cm'] = None
            row['sign_change_height_m'] = cross; hr.append(row)
for x in hr:
    x.setdefault('sign_change_height_m', '')
wcsv('height_response.csv', hr, 'efficiency (a 0.85 / emitted) per height and ratio to 40 cm; rows position=gaps_over_trays hold the over-gaps / over-trays ratio per height and the interpolated height where it crosses 1')
# ---- smoothing vs v2 (40 / 50 cm)
sv = []
for t in ('A', 'B'):
    refs = {}
    for ver, root, sid in (('v3', HERE, next(s for s, d in SC.items() if d['p'] == 0.10 and d['shape'] == 'flat' and d['pos'] == 'trays' and d['height'] == 0.40)), ('v2', V2, 'S13')):
        refs[ver] = find('scene', t, sid, root)
    if not refs['v3'] or not refs['v2']:
        continue
    for sid, s in SC.items():
        if not s['v2_sid']:
            continue
        a = find('scene', t, sid); b = find('scene', t, s['v2_sid'], V2); ia = find('iso', t, sid); ib = find('iso', t, s['v2_sid'], V2)
        if not (a and b):
            continue
        ca = np.array([c['absorbed_a1'] for c in a['copies']]); cb = np.array([c['absorbed_a1'] for c in b['copies']]); ua = np.array([c['unshaded_a1'] for c in a['copies']]); ub = np.array([c['unshaded_a1'] for c in b['copies']])
        row = dict(table=t, v3_sid=sid, v2_sid=s['v2_sid'], spacing_m=s['p'], shape=s['shape'], position=s['pos'], height_m=s['height'], eff_v2=r(b['efficiency_a085']), eff_v3=r(a['efficiency_a085']),
                   eff_ratio_v3_v2=r(a['efficiency_a085'] / b['efficiency_a085']), copy_ratio_median=r(np.median(ca / cb)), copy_ratio_p10=r(np.percentile(ca / cb, 10)), copy_ratio_p90=r(np.percentile(ca / cb, 90)),
                   total_shading_v2=r(1 - cb.sum() / ub.sum()), total_shading_v3=r(1 - ca.sum() / ua.sum()), closure_residual_v2=r(b['closure']['residual_fraction_of_emitted']), closure_residual_v3=r(a['closure']['residual_fraction_of_emitted']),
                   removed_over_absorbed_plus_reflected_v2=r(b['closure']['removed_by_plants'] / (b['closure']['absorbed_a085'] + b['closure']['reflected'])),
                   removed_over_absorbed_plus_reflected_v3=r(a['closure']['removed_by_plants'] / (a['closure']['absorbed_a085'] + a['closure']['reflected'])))
        if ia and ib:
            ia_ = np.array([c['isolated_a1'] for c in ia['copies']]); ib_ = np.array([c['isolated_a1'] for c in ib['copies']])
            row.update(self_shading_v2=r(1 - ib_.sum() / ub.sum()), self_shading_v3=r(1 - ia_.sum() / ua.sum()), neighbour_share_v2=r(1 - cb.sum() / ib_.sum()), neighbour_share_v3=r(1 - ca.sum() / ia_.sum()))
        e3 = a['efficiency_a085'] - refs['v3']['efficiency_a085']; e2 = b['efficiency_a085'] - refs['v2']['efficiency_a085']
        row['effect_vs_reference_v2'] = r(e2, 6); row['effect_vs_reference_v3'] = r(e3, 6); row['effect_sign_kept'] = bool(np.sign(e2) == np.sign(e3)) if abs(e2) > 0 else 'reference'
        sv.append(row)
wcsv('smoothing_vs_v2.csv', sv, 'v3 (smoothed normals) vs v2 (raw normals), same placement, receivers and emitter draws; report only')
pe3 = rd(HERE / 'position_effect.csv') if (HERE / 'position_effect.csv').exists() else []; pe2 = rd(V2 / 'position_effect.csv')
print('regression identical:', [x['identical'] for x in reg], '| near_luminaire scenes:', len(flag), '| height rows', len(hr), '| smoothing rows', len(sv))
