"""Gates, decomposition and figure for the coplanar separation run (no tracing).
python make_report_sep.py -> coplanar_sep_R3-1b.csv, GATES.json, DECOMPOSITION.json, figures/decomposition_bars.png, HASHES.json"""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; KO = HERE / 'out_k1'; CUR = HERE.parent / 'engine_curved_T_20261101_01'
R = json.load(open(KO / 'results.json', encoding='utf-8')); S = R['summary']; EL = R['engine_lamp']; CS = R['coincident_share']
PREV = json.load(open(CUR / 'out_k1/results.json', encoding='utf-8'))['summary']; OM = json.load(open(HERE / 'OFFSET_META.json', encoding='utf-8'))
TAG = ' · '.join(R['labels']); VAR = ['T', 'T_off0.05', 'T_off0.1', 'T_off0.2', 'T_fold', 'T_foldcup']; LEAVES = range(1, 9)
plt.rcParams.update({'font.family': ['Malgun Gothic', 'DejaVu Sans'], 'axes.unicode_minus': False})


def m(key, k):
    return S[key][k]['mean']


def td(t, model='v1', k='total'):
    return m(f'{t}|{model}', k) / m(f'D_blades|{model}', k)


# ---------------- gates
G = dict(G1_D=R['repro'])
g1 = {}
for t in ('T', 'T_fold', 'T_foldcup'):
    a, b = m(f'{t}|v1', 'total'), PREV[f'{t}|v1']['total']['mean']; sa, sb = m(f'{t}|v1', 'self'), PREV[f'{t}|v1']['self']['mean']
    g1[t] = dict(total_rel_diff=a / b - 1, self_rel_diff=sa / sb - 1, passed=abs(a / b - 1) <= 0.005 and abs(sa / sb - 1) <= 0.005)
G['G1_T'] = g1; G['G1_pass'] = R['repro']['passed'] and all(v['passed'] for v in g1.values())
G['G2'] = {k: v for k, v in OM['variants'].items()}; G['G2_pass'] = all(v['G2_pass'] for v in OM['variants'].values())
G['G3'] = dict(coincident_share_off01_seed101=CS['T_off0.1|v1']['101'], passed=CS['T_off0.1|v1']['101'] <= 0.005)
r01 = td('T_off0.1'); G['G4'] = {f'T_off{d}': td(f'T_off{d}') / r01 - 1 for d in ('0.05', '0.2')}
G['G4_pass'] = all(abs(v) <= 0.01 for v in G['G4'].values()); G['coincident_share'] = CS

# ---------------- decomposition (delta 0.1)
def split(f):
    a, b, c, d = f('T'), f('T_off0.1'), f('T_fold'), f('T_foldcup'); tot = d - a
    return dict(flat=a, offset=b, fold=c, foldcup=d, total=tot, rule=b - a, fold_effect=c - b, cup_effect=d - c,
                rule_pct=100 * (b - a) / tot if tot else None, fold_pct=100 * (c - b) / tot if tot else None, cup_pct=100 * (d - c) / tot if tot else None)


DEC = {'cabinet_T_over_D_v1': split(lambda t: td(t)), 'cabinet_T_over_D_tau': split(lambda t: td(t, 'tau0.07')),
       'self_shading_v1': split(lambda t: m(f'{t}|v1', 'self')), 'lower_third_T_over_D_v1': split(lambda t: td(t, 'v1', 'lower_total')),
       'engine_lamp_T_over_D': split(lambda t: EL[f'{t}_static']['total']['mean'] / EL['D_blades']['total'])}
d = DEC['cabinet_T_over_D_v1']; bend = d['fold_effect'] + d['cup_effect']
if not (G['G1_pass'] and G['G2_pass']):
    DEC['sentence'] = 'stopped: reproduction or area gate failed'
elif not G['G4_pass']:
    DEC['sentence'] = 'the split could not be determined (offset-sensitive)'
else:
    DEC['sentence'] = ('Of the drop from x%.2f (flat) to x%.2f (fold+cup), %.3f (%.0f %%) came from the coplanar half-credit rule and %.3f (%.0f %%) from bending; '
                       'fold+cup x%.2f is unaffected (coplanar share %.1f %%).') % (d['flat'], d['foldcup'], -d['rule'], d['rule_pct'], -bend, d['fold_pct'] + d['cup_pct'],
                                                                                  d['foldcup'], 100 * CS['T_foldcup|v1']['101'])
DEC['G3_pass'] = G['G3']['passed']
if not G['G3']['passed'] and 'came from' in DEC['sentence']:
    cs0, cs1 = CS['T|v1']['101'], CS['T_off0.1|v1']['101']; per = d['rule'] / (cs1 - cs0); extra = per * (0 - cs1)
    DEC['rule_linear_extrapolation_to_zero_share'] = dict(rule=d['rule'] + extra, rule_pct=100 * (d['rule'] + extra) / d['total'], label='extrapolated_not_measured')
    DEC['sentence'] += (' G3 failed: the offset left %.1f %% of receivers coplanar (flat %.0f %%), so the rule share is a lower bound; '
                        'linear extrapolation to zero coplanar share gives about %.0f %% (extrapolated, not measured).') % (100 * cs1, 100 * cs0, DEC['rule_linear_extrapolation_to_zero_share']['rule_pct'])
(HERE / 'GATES.json').write_text(json.dumps(G, indent=1), encoding='utf-8'); (HERE / 'DECOMPOSITION.json').write_text(json.dumps(DEC, indent=1), encoding='utf-8')

# ---------------- table
rows = []
for model in ('v1', 'tau0.07'):
    Db = f'D_blades|{model}'
    for t in VAR:
        Tk = f'{t}|{model}'
        rows.append(dict(level='plant', unit='plant', model=model, variant=t, T_over_D=td(t, model), sigma_rel=math.hypot(S[Tk]['total']['sigma'] / m(Tk, 'total'), S[Db]['total']['sigma'] / m(Db, 'total')),
                         D_self=m(Db, 'self'), T_self=m(Tk, 'self'), coincident_share=CS[f'{t}|{model}']['101']))
        for th in ('lower', 'middle', 'upper'):
            rows.append(dict(level='third', unit=th, model=model, variant=t, T_over_D=td(t, model, f'{th}_total'), D_self=m(Db, f'{th}_self'), T_self=m(Tk, f'{th}_self')))
        dl = [m(Db, f'leaf{L}_light') for L in LEAVES]; tl = [m(Tk, f'leaf{L}_light') for L in LEAVES]; da = [m(Db, f'leaf{L}_total') for L in LEAVES]; ta = [m(Tk, f'leaf{L}_total') for L in LEAVES]
        for i, L in enumerate(LEAVES):
            rows.append(dict(level='leaf', unit=f'leaf{L}', model=model, variant=t, T_over_D=ta[i] / da[i], D_light_per_area=dl[i], T_light_per_area=tl[i], D_self=m(Db, f'leaf{L}_self'), T_self=m(Tk, f'leaf{L}_self')))
        rows.append(dict(level='rank', unit='spearman_absorbed', model=model, variant=t, T_over_D=float(spearmanr(da, ta).statistic)))
        rows.append(dict(level='rank', unit='spearman_light_per_area', model=model, variant=t, T_over_D=float(spearmanr(dl, tl).statistic)))
gr = EL.get('groimp', {})
for t in VAR:
    st = EL[f'{t}_static']; g = gr.get(t, {})
    row = dict(level='engine_lamp', unit='plant', model='black', variant=t, T_over_D=st['total']['mean'] / EL['D_blades']['total'], T_self=st['self']['mean'], D_self=EL['D_blades']['self'])
    if g.get('ok'):
        sd = list(g['seeds'].values())
        row.update(groimp_over_static=g['mean']['total']['mean'] / st['total']['mean'], groimp_packet_sigma_rel=g['packet_sigma_total'] / g['mean']['total']['mean'],
                   groimp_seeds_identical=all(x['total'] == sd[0]['total'] and x['hits'] == sd[0]['hits'] for x in sd))
    rows.append(row)
cols = []
for r in rows:
    cols += [k for k in r if k not in cols]
with open(HERE / 'coplanar_sep_R3-1b.csv', 'w', encoding='utf-8', newline='') as fh:
    fh.write(f'# {TAG}; D = R8_pc blade surfels 1-8 at d_ex 2.4 mm; coincident_share = half-credit receiver share, seed 101\n')
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

# ---------------- figure: stacked bars rule / fold / cup
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
keys = [('cabinet_T_over_D_v1', '캐비닛 T/D (v1)'), ('cabinet_T_over_D_tau', '캐비닛 T/D (v1_tau)'), ('lower_third_T_over_D_v1', '아래 1/3 T/D'), ('self_shading_v1', '자기 그늘'), ('engine_lamp_T_over_D', '엔진 램프 T/D')]
fig, axes = plt.subplots(1, len(keys), figsize=(15, 4.6), facecolor='white')
for ax, (k, lab) in zip(axes, keys):
    dd = DEC[k]; parts = [('rule', '공면 반점수 규칙', '#c44e52'), ('fold_effect', '접힘', '#55a868'), ('cup_effect', '컵', '#8172b3')]
    pos, neg = 0.0, 0.0
    for p, nm, c in parts:
        val = dd[p]; bottom = pos if val >= 0 else neg
        ax.bar(0, val, 0.5, bottom=bottom, color=c, label=nm)
        if val >= 0: pos += val
        else: neg += val
    ax.axhline(0, color='k', lw=0.6); ax.plot([-0.35, 0.35], [dd['total']] * 2, 'k--', lw=1, label='합계')
    ax.set_xticks([]); ax.set_title(f'{lab}\n평면 {dd["flat"]:.2f} → 접힘+컵 {dd["foldcup"]:.2f}', fontsize=9)
axes[0].set_ylabel('변화량 (평면 → 접힘+컵)'); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle(f'평면 T → 접힘+컵 변화의 분해 (δ = 0.1 mm)   G4 {"통과" if G["G4_pass"] else "불합격(offset_sensitive)"}', fontsize=11)
fig.text(0.01, 0.01, TAG, fontsize=6.5); fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(FIG / 'decomposition_bars.png', dpi=170); plt.close(fig)

hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*')) if p.is_file() and p.name not in ('HASHES.json', 'RUNNING.lock')}
hashes.update({f'out_k1/{p.name}': hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(KO.glob('*')) if p.is_file()})
hashes.update({f'figures/{p.name}': hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FIG.glob('*'))}); hashes['engine_hash_static'] = R['engine_hash']
(HERE / 'HASHES.json').write_text(json.dumps(hashes, indent=1), encoding='utf-8')
print(json.dumps({k: G[k] for k in ('G1_pass', 'G2_pass', 'G3', 'G4', 'G4_pass')}, indent=1))
print(json.dumps(DEC, indent=1))
for r in rows:
    if r['level'] in ('plant', 'engine_lamp') or r['level'] == 'rank':
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
