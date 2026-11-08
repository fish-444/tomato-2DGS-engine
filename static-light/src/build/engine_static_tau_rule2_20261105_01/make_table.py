"""RESULT_TABLE_rule2.csv: the v1 / v1_tau / v1tau_smoothn_r2 columns of the reference table plus v1tau2_smoothn_r2 (rule 2, d_stem 1.5 mm, tau 0.07) and the sensitivity arms.
Plant section from K12 (out_k12a/b); scenario section from K13 and tray32 section from K14 when present. Separate file: the reference table is not overwritten."""
import csv, glob, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; REFT = HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01/RESULT_TABLE_v1_vs_v1tau.csv'
ref = {}
for r in csv.DictReader(l for l in open(REFT, encoding='utf-8') if not l.startswith('#')):
    ref[(r['section'], r['item'], r['scene'])] = r
ARMS = ['rule1_tau0.07', 'rule2_d1.5_tau0.07', 'rule2_d1.5_tau0.05', 'rule2_d1.5_tau0.1', 'rule2_d1_tau0.07', 'rule2_d2.5_tau0.07']
COLS = ['v1', 'v1_tau_a085', 'v1tau_smooth_a085', 'k12_rule1_check', 'v1tau2_smoothn_r2', 'rule2_tau0.05', 'rule2_tau0.10', 'rule2_dstem1.0', 'rule2_dstem2.5']
pct = lambda a, b: round(100 * (a / b - 1), 3) if b else ''
rows = []
AM = 0.85   # K12 arms store the a = 1 kernel totals; absorbed = U.K.A.a with a 0.85 on every receiver (CONTRACT §0), as the reference a085 columns
ITEMS = [('mean_relative_light', lambda x, A: AM * x['total'] / A), ('absorbed_total_ru_mm2', lambda x, A: AM * x['total']), ('self_shading', lambda x, A: x['self_shading']),
         ('mean_light_lower_third', lambda x, A: AM * x['lower_mean_light']), ('mean_light_middle_third', lambda x, A: AM * x['middle_mean_light']), ('mean_light_upper_third', lambda x, A: AM * x['upper_mean_light'])]
for f in sorted(glob.glob(str(HERE / 'out_k12[ab]/*/plant.json'))):
    d = json.load(open(f, encoding='utf-8')); pid = d['plant']; A = d['area_b_mm2']
    for item, fn in ITEMS:
        r0 = ref.get(('plant', item, pid), {}); v = {a: fn(d['arms'][a], A) for a in ARMS}
        row = dict(section='plant', item=item, scene=pid, v1=r0.get('v1', ''), v1_tau_a085=r0.get('v1_tau_a085', ''), v1tau_smooth_a085=r0.get('v1tau_smooth_a085', ''),
                   k12_rule1_check=round(v['rule1_tau0.07'], 6), v1tau2_smoothn_r2=round(v['rule2_d1.5_tau0.07'], 6), **{'rule2_tau0.05': round(v['rule2_d1.5_tau0.05'], 6), 'rule2_tau0.10': round(v['rule2_d1.5_tau0.1'], 6),
                   'rule2_dstem1.0': round(v['rule2_d1_tau0.07'], 6), 'rule2_dstem2.5': round(v['rule2_d2.5_tau0.07'], 6)},
                   chg_rule2_vs_rule1_pct=pct(v['rule2_d1.5_tau0.07'], v['rule1_tau0.07']), chg_rule2_vs_v1_pct=pct(v['rule2_d1.5_tau0.07'], float(r0['v1'])) if r0.get('v1') else '',
                   stem_surfels_d1_5=d['stem_surfels']['d1.5'], label='light_model_v1tau2; assumed_leaf_optics; relative_light; stem = geometric d_stem rule (pilot: classifier stem)', source=f'K12 {Path(f).parent.parent.name}/{pid}/plant.json')
        rows.append(row)
    s = d['sensors_rule2_d1.5_tau0.07']
    rows.append(dict(section='plant', item='transmittance_at_rim_layer', scene=pid, v1tau2_smoothn_r2=round(s['transmittance_at_rim_layer'], 6) if s['transmittance_at_rim_layer'] is not None else '', label='5 mm layer; relative_light', source='K12 sensors'))
# scenario section: K13 (rule 2) vs K9 (v1tau_smoothn_r2, reference folder); scenario effect = relative change vs the scenario's base (rel_delta_tau)
K9 = json.load(open(HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01/out_k9/k9_results.json', encoding='utf-8'))
p13 = HERE / 'out_k13/k13_results.json'; SIGN = dict(flips=0, n=0, cls_changed=0)
if p13.exists():
    K13 = json.load(open(p13, encoding='utf-8'))
    for grp in ('scenarios_r31b', 'scenarios_tray'):
        for key, v in K13[grp]['table'].items():
            r1 = K9[grp]['table'].get(key, {}); a, b = r1.get('rel_delta_tau'), v['rel_delta_tau']
            SIGN['n'] += 1; SIGN['flips'] += int(a is not None and a * b < 0); SIGN['cls_changed'] += int(r1.get('cls') != v['cls'])
            rows.append(dict(section='scenario', item='rel_delta_' + grp.split('_')[1], scene=key, v1tau_smooth_a085=round(a, 6) if a is not None else '', v1tau2_smoothn_r2=round(b, 6),
                             chg_rule2_vs_rule1_pct=round(100 * (b - a), 3) if a is not None else '', label=f'scenario effect (fraction); cls rule1 {r1.get("cls")} / rule2 {v["cls"]}; chg column = difference in percentage points', source='K13 vs K9'))
# tray32 section: K14 (rule 2) vs tray32 v3 smoothing-on (out_t1/scene, table A, S25-S28)
T3 = HERE.parents[0] / 'static_tray32_v3_20261105_01/out_t1/scene'
for sid in ('S25', 'S26', 'S27', 'S28'):
    p = HERE / f'out_k14/scene/A_{sid}.json'
    if p.exists():
        n2 = json.load(open(p, encoding='utf-8')); n1 = json.load(open(T3 / f'A_{sid}.json', encoding='utf-8'))
        for item in ('efficiency_a085', 'absorbed_over_outlines_a085'):
            rows.append(dict(section='tray32', item=item, scene=f'A_{sid} h {n2["scene"]["height"]} m', v1tau_smooth_a085=round(n1[item], 6), v1tau2_smoothn_r2=round(n2[item], 6), chg_rule2_vs_rule1_pct=pct(n2[item], n1[item]),
                             label='tray32 v3 geometry; absolute_efficiency_uncertain_up_to_1.9x; relative_light', source='K14 vs tray32 v3 out_t1'))
print('scenario sign flips', SIGN)
FN = ['section', 'item', 'scene'] + COLS + ['chg_rule2_vs_rule1_pct', 'chg_rule2_vs_v1_pct', 'stem_surfels_d1_5', 'label', 'source']
with open(HERE / 'RESULT_TABLE_rule2.csv', 'w', encoding='utf-8', newline='') as f:
    f.write('# v1tau2_smoothn_r2 = rule 2 (tau 0.07 / a 0.85 on every non-stem surfel, stem tau 0, d_stem 1.5 mm) + smoothing r 2 mm; relative_light; assumed_leaf_optics; not_validation\n')
    w = csv.DictWriter(f, fieldnames=FN, restval=''); w.writeheader(); w.writerows(rows)
pl = [r for r in rows if r['item'] == 'mean_relative_light']
print(len(rows), 'rows'); [print(r['scene'], r['v1tau_smooth_a085'], r['k12_rule1_check'], r['v1tau2_smoothn_r2'], r['chg_rule2_vs_rule1_pct'], r['stem_surfels_d1_5']) for r in pl]
