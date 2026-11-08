"""Ideotype measures (CONTRACT §7) from kout/<kernel>/out/<case>/{status,summary}.json.
python -X utf8 analyze_ideo.py <env> [<variant>]   -> paired ideotype - BASE per plant (slot) and replicate, day 71 (59.9 DAG) and selected days.
Writes plants_summary_<env>.csv and paired_effects_<env>.csv; prints the table."""
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path

H = Path(__file__).resolve().parent; DAG0 = 11.07; D71 = 71
TYPES = ['BASE', 'IN125', 'IN150', 'LS125', 'LS150', 'PE150', 'COMBO']


def load():
    R, F = {}, {}
    for p in sorted((H / 'kout').glob('*/out/*/status.json')):
        s = json.load(open(p)); sp = p.parent / 'summary.json'
        if s.get('status') == 'verified' and sp.exists():
            R[p.parent.name] = json.load(open(sp))
        else:
            F[p.parent.name] = s.get('status')
    return R, F


def q(v, p):
    v = sorted(x for x in v if x == x)
    if not v: return float('nan')
    k = (len(v) - 1) * p; f = math.floor(k); c = min(f + 1, len(v) - 1); return v[f] + (v[c] - v[f]) * (k - f)


def plant(S, pid, day=D71):
    D = S['daily'][pid]; d = D[day]; assert d['day'] == day
    g = d['dm_by_group']; sf = d.get('surface_mm2', {})
    return dict(absorbed_rel=d['absorbed_rel'], absorbed_cum=sum(r['absorbed_rel'] for r in D[:day + 1]),
                gross_cum_gC=sum(r.get('gross_gC', 0) - r.get('downregulated_gC', 0) for r in D[:day + 1]),
                leaf_area_cm2=d['leaf_area_mm2'] / 100, dm_g=d['dm_g'], fruit_g=g.get('fruit', 0), leaf_g=g.get('leaf', 0), stem_g=g.get('stem', 0),
                petiole_rachis_g=g.get('petiole_rachis', 0), root_g=g.get('root', 0), height_mm=d['height_mm'],
                struct_surface_mm2=sum(sf.values()), fruits=d['fruits'], compounds=d['compounds'])


KEYS = ['absorbed_rel', 'absorbed_cum', 'gross_cum_gC', 'leaf_area_cm2', 'dm_g', 'fruit_g', 'leaf_g', 'stem_g', 'petiole_rachis_g', 'height_mm', 'struct_surface_mm2']


def main(env, variant=''):
    R, F = load(); suf = f'_{variant}' if variant else ''
    cases = {(t, r): R.get(f'{t}_{env}_r{r}{suf}') for t in TYPES for r in (1, 2)}
    rows, pairs = [], defaultdict(lambda: defaultdict(list))
    for (t, r), S in cases.items():
        if not S: continue
        for pid in S['daily']:
            m = plant(S, pid); m.update(type=t, rep=r, plant=pid, min_clearance_mm=min(S['clearance_mm'].values())); rows.append(m)
            B = cases.get(('BASE', r))
            if t != 'BASE' and B:
                b = plant(B, pid)
                for k in KEYS: pairs[t][k].append((m[k] / b[k] - 1) * 100 if b[k] else float('nan'))
                pairs[t]['cost_struct_g'].append((m['stem_g'] + m['petiole_rachis_g']) - (b['stem_g'] + b['petiole_rachis_g']))
    with open(H / f'plants_summary_{env}{suf}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['type', 'rep', 'plant'] + KEYS + ['fruits', 'compounds', 'min_clearance_mm']); w.writeheader()
        for m in rows: w.writerow({k: m[k] for k in w.fieldnames})
    out = []
    for t in TYPES[1:]:
        if t not in pairs: continue
        for k, v in pairs[t].items():
            out.append(dict(type=t, env=env, variant=variant, measure=k, unit='g' if k == 'cost_struct_g' else 'pct', n=len(v), median=q(v, .5), p10=q(v, .1), p90=q(v, .9)))
    with open(H / f'paired_effects_{env}{suf}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)
    print('failed:', {k: v for k, v in F.items() if env in k})
    for o in out: print(f"{o['type']:6} {o['measure']:18} n{o['n']:3} {o['median']:+9.2f} [{o['p10']:+8.2f}, {o['p90']:+8.2f}] {o['unit']}")
    return out


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
