"""Amendment 1 (option c): RESULT_TABLE_rule2_unified.csv = RESULT_TABLE_rule2.csv rows + a `variant` column; rows that involve R3-1b are kept as `rule2_pilot_classifier_stem`
and new rows from K16 (pilot plant), K17 (scenarios) and K18 (tray32) are added as `stem_rule_unified_geometric`. RESULT_TABLE_rule2.csv itself is not changed."""
import csv, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; REF = HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01'
rows = list(csv.DictReader(l for l in open(HERE / 'RESULT_TABLE_rule2.csv', encoding='utf-8') if not l.startswith('#'))); FN = list(rows[0])
for r in rows:
    pilot = r['section'] in ('scenario', 'tray32') or r['scene'] == 'R3-1b'
    r['variant'] = 'rule2_pilot_classifier_stem' if pilot else 'rule2 (geometric stem; unchanged by the unification)'
new = []; AM = 0.85; pct = lambda a, b: round(100 * (a / b - 1), 3) if b else ''
d = json.load(open(HERE / 'out_k16/R3-1b/plant.json', encoding='utf-8')); o = json.load(open(HERE / 'out_k12a/R3-1b/plant.json', encoding='utf-8')); A = d['area_b_mm2']
ITEMS = [('mean_relative_light', lambda x: AM * x['total'] / A), ('absorbed_total_ru_mm2', lambda x: AM * x['total']), ('self_shading', lambda x: x['self_shading']),
         ('mean_light_lower_third', lambda x: AM * x['lower_mean_light']), ('mean_light_middle_third', lambda x: AM * x['middle_mean_light']), ('mean_light_upper_third', lambda x: AM * x['upper_mean_light'])]
old = {(r['item']): r for r in rows if r['scene'] == 'R3-1b' and r['section'] == 'plant'}
for item, fn in ITEMS:
    v = {a: fn(d['arms'][a]) for a in d['arms']}; vo = fn(o['arms']['rule2_d1.5_tau0.07']); r0 = old[item]
    new.append(dict(section='plant', item=item, scene='R3-1b', v1=r0['v1'], v1_tau_a085=r0['v1_tau_a085'], v1tau_smooth_a085=r0['v1tau_smooth_a085'], k12_rule1_check=round(v['rule1_tau0.07'], 6),
                    v1tau2_smoothn_r2=round(v['rule2_d1.5_tau0.07'], 6), **{'rule2_tau0.05': round(v['rule2_d1.5_tau0.05'], 6), 'rule2_tau0.10': round(v['rule2_d1.5_tau0.1'], 6), 'rule2_dstem1.0': round(v['rule2_d1_tau0.07'], 6),
                    'rule2_dstem2.5': round(v['rule2_d2.5_tau0.07'], 6)}, chg_rule2_vs_rule1_pct=pct(v['rule2_d1.5_tau0.07'], v['rule1_tau0.07']), chg_rule2_vs_v1_pct=pct(v['rule2_d1.5_tau0.07'], float(r0['v1'])),
                    stem_surfels_d1_5=d['stem_surfels']['d1.5'], label=f'stem_rule_unified_geometric; chg vs classifier-stem rule 2 {pct(v["rule2_d1.5_tau0.07"], vo)} %', source='K16 out_k16/R3-1b/plant.json', variant='stem_rule_unified_geometric'))
new.append(dict(section='plant', item='transmittance_at_rim_layer', scene='R3-1b', v1tau2_smoothn_r2=round(d['sensors_rule2_d1.5_tau0.07']['transmittance_at_rim_layer'], 6), label='5 mm layer; stem_rule_unified_geometric', source='K16 sensors', variant='stem_rule_unified_geometric'))
SIGN = None
p17 = HERE / 'out_k17/k13_results.json'
if p17.exists():
    K9 = json.load(open(REF / 'out_k9/k9_results.json', encoding='utf-8')); K13 = json.load(open(HERE / 'out_k13/k13_results.json', encoding='utf-8')); K17 = json.load(open(p17, encoding='utf-8'))
    SIGN = dict(n=0, flips_vs_rule1=0, flips_vs_classifier=0, cls_changed_vs_rule1=0, max_abs_pp_vs_classifier=0.0)
    for grp in ('scenarios_r31b', 'scenarios_tray'):
        for key, v in K17[grp]['table'].items():
            a = K9[grp]['table'][key]['rel_delta_tau']; c = K13[grp]['table'][key]['rel_delta_tau']; b = v['rel_delta_tau']
            SIGN['n'] += 1; SIGN['flips_vs_rule1'] += int(a * b < 0); SIGN['flips_vs_classifier'] += int(c * b < 0); SIGN['cls_changed_vs_rule1'] += int(K9[grp]['table'][key]['cls'] != v['cls'])
            SIGN['max_abs_pp_vs_classifier'] = max(SIGN['max_abs_pp_vs_classifier'], abs(100 * (b - c)))
            new.append(dict(section='scenario', item='rel_delta_' + grp.split('_')[1], scene=key, v1tau_smooth_a085=round(a, 6), v1tau2_smoothn_r2=round(b, 6), chg_rule2_vs_rule1_pct=round(100 * (b - a), 3),
                            label=f'percentage points; classifier-stem rule 2 {round(c, 6)}; cls {v["cls"]}', source='K17 vs K9 / K13', variant='stem_rule_unified_geometric'))
T3 = HERE.parents[0] / 'static_tray32_v3_20261105_01/out_t1/scene'
for sid in ('S25', 'S26', 'S27', 'S28'):
    n2 = json.load(open(HERE / f'out_k18/scene/A_{sid}.json', encoding='utf-8')); n1 = json.load(open(T3 / f'A_{sid}.json', encoding='utf-8')); nc = json.load(open(HERE / f'out_k14/scene/A_{sid}.json', encoding='utf-8'))
    for item in ('efficiency_a085', 'absorbed_over_outlines_a085'):
        new.append(dict(section='tray32', item=item, scene=f'A_{sid} h {n2["scene"]["height"]} m', v1tau_smooth_a085=round(n1[item], 6), v1tau2_smoothn_r2=round(n2[item], 6), chg_rule2_vs_rule1_pct=pct(n2[item], n1[item]),
                        label=f'tray32 v3 geometry; classifier-stem rule 2 {round(nc[item], 6)} ({pct(n2[item], nc[item])} %); absolute_efficiency_uncertain_up_to_1.9x', source='K18 vs tray32 v3 / K14', variant='stem_rule_unified_geometric'))
with open(HERE / 'RESULT_TABLE_rule2_unified.csv', 'w', encoding='utf-8', newline='') as f:
    f.write('# stem_rule_unified_geometric (user decision 19:53, option c): stem = surfels within d_stem of the stem polylines without a leaf label, all 14 plants (pilot: main-axis polyline only, no side axes exist); rows `rule2_pilot_classifier_stem` = earlier values kept; relative_light; assumed_leaf_optics; not_validation\n')
    w = csv.DictWriter(f, fieldnames=FN + ['variant'], restval=''); w.writeheader(); w.writerows(rows + new)
json.dump(dict(scenario_signs=SIGN), open(HERE / 'unified_summary.json', 'w'), indent=1)
print(len(rows) + len(new), 'rows; scenario', SIGN)
