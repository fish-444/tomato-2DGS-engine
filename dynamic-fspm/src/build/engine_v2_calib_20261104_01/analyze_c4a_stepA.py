"""C4A Step A (CONTRACT Amendment 17): determinism vs the earlier runs, carbon-satisfaction diagnosis, engine L by rank vs reconstruction.
python -X utf8 analyze_c4a_stepA.py -> C4A_stepA.json, C4A_stepA_diagnosis.csv, C4A_length_by_rank_base.csv"""
import csv, glob, json, statistics as st
from collections import defaultdict
from pathlib import Path
import analyze as A

H = Path(__file__).resolve().parent; DAG0 = 11.07


def load(pattern):
    out = {}
    for p in glob.glob(str(H / 'kout' / pattern / 'out' / '*' / 'summary.json')):
        s = json.load(open(p)); out[s['case']] = s
    return out


def recon_L():
    rows = [r for r in csv.DictReader(l for l in open(H.parent / 'traits_recon_20261101_01' / 'MTG_LEAFLET_ARRANGEMENT_v1.csv', encoding='utf-8') if not l.startswith('#'))]
    out = {}
    for r in rows:
        if r['subset'] == 'all_11' and r['rank_class'] == 'main' and r['quantity'] == 'rachis_L_mm' and r['position'].startswith('leaf_rank_'):
            out[int(r['position'][10:])] = dict(n=int(r['n']), median=float(r['median']), p10=float(r['p10']), p90=float(r['p90']))
    return out


def engine_L(S):
    by = defaultdict(list)
    for s in S.values():
        for c in s.get('c4c_measures', []):
            q = c['compound'].split('/')
            if len(q) == 2:  # main axis p1/Cxxxx
                by[int(q[1][1:])].append(c['chain_mm'])
            else:
                by['side'].append(c['chain_mm'])
    return by


def main():
    new = load('ev-c4ad-*'); old = {**load('ev-v22b-*'), **load('ev-v21c-*')}
    det = {}
    for k, s in new.items():
        o = old.get(k)
        det[k] = (o is not None and json.dumps(o['daily'], sort_keys=True) == json.dumps(s['daily'], sort_keys=True))
    res = dict(determinism=dict(n=len(det), identical=sum(det.values()), differing=sorted(k for k, v in det.items() if not v)))
    rows = []; R = recon_L()
    for arm, label in (('V22b', 'V22b (base)'), ('V21Bs5.85', 'v2.1')):
        S = {k: v for k, v in new.items() if k.rsplit('_m', 1)[0] == arm}
        errs = [k for k, v in S.items() if 'c4a_diag_error' in v]
        per_day = defaultdict(list); per_day_p = defaultdict(list); down = []; res_end = []
        mom = []
        for s in S.values():
            d = s.get('c4a_diag', {})
            for r in d.get('daily', []):
                dag = r['interval'] - DAG0
                if r['leaf_demand'] > 0: per_day[r['interval']].append(r['leaf_alloc'] / r['leaf_demand'])
                if r['plant_demand'] > 0: per_day_p[r['interval']].append(r['plant_alloc'] / r['plant_demand'])
            down.append(sum(r.get('downregulated_gC', 0) for r in d.get('daily', []))); res_end.append(d.get('daily', [{}])[-1].get('reserve_end_gC'))
            mom += d.get('leaflet_mass_over_max_day71', [])
        days = sorted(i for i in per_day if i - DAG0 > 20)
        med = {i: st.median(per_day[i]) for i in days}
        below = sum(1 for i in days if med[i] < 0.8)
        cls = 'source-limited' if below > 0.5 * len(days) else 'sink-limited'
        res[arm] = dict(label=label, n=len(S), diag_errors=errs, days_after_20dag=len(days), days_leaf_satisfaction_below_0_8=below, cls=cls,
                        leaf_satisfaction_median_after20=st.median(med.values()) if med else None,
                        leaf_satisfaction_p10_p90=(A.q(list(med.values()), .1), A.q(list(med.values()), .9)),
                        plant_satisfaction_median_after20=st.median(st.median(per_day_p[i]) for i in days) if days else None,
                        downregulated_cum_median=A.q(down, .5), reserve_end_median=A.q([x for x in res_end if x is not None], .5),
                        leaflet_mass_over_max_day71_median=A.q(mom, .5), share_leaflets_ge_0_95_max=sum(1 for x in mom if x >= .95) / len(mom) if mom else None)
        for i in sorted(per_day):
            rows.append(dict(arm=arm, interval=i, dag=round(i - DAG0, 2), leaf_satisfaction_median=st.median(per_day[i]), plant_satisfaction_median=st.median(per_day_p[i]) if per_day_p[i] else ''))
        EL = engine_L(S); lrows = []
        for r in sorted(R):
            e = EL.get(r, [])
            lrows.append(dict(arm=arm, rank=r, recon_n=R[r]['n'], recon_median_mm=R[r]['median'], engine_n=len(e), engine_median_mm=A.q(e, .5) if e else None,
                              ratio=(A.q(e, .5) / R[r]['median']) if e else None))
        lrows.append(dict(arm=arm, rank='side', recon_n=0, recon_median_mm=None, engine_n=len(EL.get('side', [])), engine_median_mm=A.q(EL.get('side', []), .5), ratio=None))
        res[arm]['L_by_rank'] = lrows
    json.dump(res, open(H / 'C4A_stepA.json', 'w'), indent=1, default=float)
    with open(H / 'C4A_stepA_diagnosis.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(H / 'C4A_length_by_rank_base.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(res['V22b']['L_by_rank'][0])); w.writeheader()
        for a in ('V22b', 'V21Bs5.85'): w.writerows(res[a]['L_by_rank'])
    print(json.dumps(res['determinism']))
    for a in ('V22b', 'V21Bs5.85'):
        print(a, json.dumps({k: v for k, v in res[a].items() if k != 'L_by_rank'}, default=float))
        for r in res[a]['L_by_rank']: print('   ', r['rank'], r['recon_median_mm'], r['engine_n'], None if r['engine_median_mm'] is None else round(r['engine_median_mm'], 2), None if r['ratio'] is None else round(r['ratio'], 2))


if __name__ == '__main__':
    main()
