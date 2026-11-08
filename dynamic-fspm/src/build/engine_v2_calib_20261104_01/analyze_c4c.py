"""C4C checks and paired effects (CONTRACT Amendments 14-15): arms V22a / V22b vs frozen v2.1 (V21Bs5.85), 16 seeds, day 71 = 59.9 DAG.
python -X utf8 analyze_c4c.py  -> BLOCK_C4C.json, block_c4c.csv"""
import csv, json, statistics as st
import analyze as A

R = A.load(); BASE = 'V21Bs5.85'
KEYS = ['leaf_area_cm2', 'dm_g', 'height_mm', 'accepted_gross_last10', 'max_leaflet_len_mm', 'absorbed_rel', 'fruits', 'n_compounds']


def span(S):
    return [x['span_ratio'] for x in S.get('c4l_measures', []) if x.get('n_leaflets', 0) >= 5 and x.get('span_ratio') == x.get('span_ratio') and x.get('span_ratio') is not None]


out = dict(base=BASE, arms={})
rows = []
for arm in ('V22a', 'V22b'):
    cases = {int(k.rsplit('_m', 1)[1]): v for k, v in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == arm}
    if not cases: continue
    failed = {k: v for k, v in R.get('_failed', {}).items() if k.startswith(arm + '_')}
    cm = [x for S in cases.values() for x in S.get('c4c_measures', [])]
    errs = [S['case'] for S in cases.values() if 'c4c_measures_error' in S]
    lf = [x for x in cm if x['n_leaflets'] > 0]
    big = [x for x in lf if x['n_leaflets'] >= 5]
    chk = dict(verified=len(cases), failed=sorted(failed), measure_errors=errs, compounds=len(cm),
               chain_max_mm=max(x['chain_mm'] for x in cm), chain_le_40=all(x['chain_mm'] <= 40 + 1e-6 for x in cm),
               chain_median_mm=A.q([x['chain_mm'] for x in cm], .5), chain_p90_mm=A.q([x['chain_mm'] for x in cm], .9),
               maxmass_rel_range_max=max(x['maxmass_rel_range'] for x in lf), maxmass_equal=all(x['maxmass_rel_range'] <= 1e-9 for x in lf),
               area_range_over_mean_median=A.q([x['area_range_over_mean'] for x in lf], .5), area_range_over_mean_p90=A.q([x['area_range_over_mean'] for x in lf], .9))
    a, b = A.arm(R, arm), A.arm(R, BASE); seeds = sorted(set(a) & set(b)); eff = {}
    for k in KEYS:
        pc = [(a[s][k] / b[s][k] - 1) * 100 for s in seeds if b[s][k]]
        eff[k] = dict(base_median=A.q([b[s][k] for s in seeds], .5), arm_median=A.q([a[s][k] for s in seeds], .5), paired_pct_median=A.q(pc, .5),
                      paired_pct_p10=A.q(pc, .1), paired_pct_p90=A.q(pc, .9), n=len(seeds))
        rows.append(dict(arm=arm, measure=k, **{x: (round(y, 4) if isinstance(y, float) else y) for x, y in eff[k].items()}))
    eff['height_mean_mm'] = st.mean(a[s]['height_mm'] for s in seeds)
    sa = [v for S in cases.values() for v in span(S)]; sb = [v for k, S in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == BASE for v in span(S)]
    eff['span_ratio'] = dict(arm_median=A.q(sa, .5), base_median=A.q(sb, .5), n_arm=len(sa), n_base=len(sb), reference=0.85)
    out['arms'][arm] = dict(checks=chk, effects=eff)
    print(arm, json.dumps(chk, default=float)); print('  height mean %.1f mm; span %.3f (v2.1 %.3f)' % (eff['height_mean_mm'], eff['span_ratio']['arm_median'], eff['span_ratio']['base_median']))
    for k in KEYS: print('  %-22s %9.4g -> %9.4g  %+7.1f %% [%+.1f, %+.1f]' % (k, eff[k]['base_median'], eff[k]['arm_median'], eff[k]['paired_pct_median'], eff[k]['paired_pct_p10'], eff[k]['paired_pct_p90']))
json.dump(out, open('BLOCK_C4C.json', 'w'), indent=1, default=float)
with open('block_c4c.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
