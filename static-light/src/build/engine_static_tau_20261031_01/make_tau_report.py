"""Tables and figures for v1_tau from the kernels' small JSON outputs (no tracing). python make_tau_report.py -> per_leaf_tau.csv,
plant_tau_table.csv, probe_ratios.csv, figures/*.png, TABLES.md"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
IDS = ['R3-1b', 't2_02', 'n_04', 'n_01', 't2_05', 'n_06', 'n_05', 't2_04', 'n_02', 'n_03']
TAG = 'light_model=v1_tau · assumed_leaf_optics (tau=0.07, a=0.85, reflection dropped) · relative_light · conditional_on_unvalidated_scale · after_checkpoint_v1'
plt.rcParams.update({'font.family': ['Malgun Gothic', 'DejaVu Sans'], 'axes.unicode_minus': False})


def load(pid):
    p = HERE / ('out_k1b' if pid == 'R3-1b' else 'out_k2') / pid / 'plant_tau.json'
    return json.load(open(p, encoding='utf-8')) if p.exists() else None


P = {pid: load(pid) for pid in IDS}
L = [f'# v1_tau tables\n\n{TAG}\n']
L.append('## Plant level (d_ex 2.4 mm; a = 1 unless noted; change vs v1)')
L.append('| plant | v1 mean light | τ0.07 mean light | Δ % | Δ % with a 0.85 | self-shading v1 → τ0.07 | lower third Δ % | middle Δ % | upper Δ % | τ0.05 / τ0.10 Δ % | overflow rays |')
L.append('|' + '---|' * 11)
rows = []
for pid, d in P.items():
    if not d:
        L.append(f'| {pid} | not_run |' + ' |' * 9); continue
    p = d['plant']; v, t = p['v1'], p['tau0.07']
    ch = lambda k, m='tau0.07': 100 * (p[m][k]['mean'] / p['v1'][k]['mean'] - 1)
    a085 = 100 * (t['absorbed_by_a']['0.85']['mean'] / v['absorbed_by_a']['1']['mean'] - 1)
    L.append(f"| {pid} | {v['mean_light']['mean']:.3f} | {t['mean_light']['mean']:.3f} | {ch('total'):+.1f} | {a085:+.1f} | {v['self']['mean']:.3f} → {t['self']['mean']:.3f} | "
             f"{ch('lower_total'):+.1f} | {ch('middle_total'):+.1f} | {ch('upper_total'):+.1f} | {ch('total', 'tau0.05'):+.1f} / {ch('total', 'tau0.10'):+.1f} | {t['overflow_rays']} |")
    rows.append(dict(plant=pid, d_total_pct=ch('total'), d_lower=ch('lower_total'), d_middle=ch('middle_total'), d_upper=ch('upper_total'), d_total_a085_pct=a085,
                     d_tau005=ch('total', 'tau0.05'), d_tau010=ch('total', 'tau0.10'), self_v1=v['self']['mean'], self_tau=t['self']['mean']))
with (HERE / 'plant_tau_table.csv').open('w', encoding='utf-8', newline='') as fh:
    fh.write(f'# {TAG}\n'); w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ['plant']); w.writeheader(); w.writerows(rows)
# per leaf
lrows = []
for pid, d in P.items():
    if not d:
        continue
    p = d['plant']
    for k in [k[5:] for k in p['v1'] if k.startswith('leaf:')]:
        v, t = p['v1'][f'leaf:{k}']['mean'], p['tau0.07'][f'leaf:{k}']['mean']
        lrows.append(dict(plant=pid, leaf=k, absorbed_v1=v, absorbed_tau_a1=t, absorbed_tau_a085=0.85 * t, change_pct_a1=100 * (t / v - 1),
                          mean_light_v1=p['v1'][f'leafmean:{k}']['mean'], mean_light_tau=p['tau0.07'][f'leafmean:{k}']['mean']))
with (HERE / 'per_leaf_tau.csv').open('w', encoding='utf-8', newline='') as fh:
    fh.write(f'# {TAG}\n'); w = csv.DictWriter(fh, fieldnames=list(lrows[0]) if lrows else ['plant']); w.writeheader(); w.writerows(lrows)
L.append('\n## Per leaf (R3-1b R8_pc blades, V-P leaves 1 and 7; n_04 clicked leaf 1 lower bound)')
L.append('| plant | leaf | mean light v1 | mean light τ0.07 | Δ % (a = 1) |'); L.append('|---|---|---|---|---|')
for r in lrows:
    L.append(f"| {r['plant']} | {r['leaf']} | {r['mean_light_v1']:.3f} | {r['mean_light_tau']:.3f} | {r['change_pct_a1']:+.1f} |")
# scenarios
for name, path in (('R3-1b alone, lamp', HERE / 'out_k1b/R3-1b/scenarios_tau.json'), ('tray 60 mm', HERE / 'out_k3/tray_scenarios_tau.json')):
    if not path.exists():
        L.append(f'\n## Scenarios {name}: not_run'); continue
    s = json.load(open(path, encoding='utf-8'))
    L.append(f"\n## Scenarios {name}: stable {s['stable']} / {s['total']}")
    L.append('| scenario | unit | rel Δ v1 | rel Δ τ0.07 | class |'); L.append('|---|---|---|---|---|')
    for k, v in s['table'].items():
        sc, unit = k.split('|')
        L.append(f"| {sc} | {unit} | {100 * v['rel_delta_v1']:+.1f} % | {100 * v['rel_delta_tau']:+.1f} % | {v['cls']} |")
(HERE / 'TABLES.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
with (HERE / 'probe_ratios.csv').open('w', encoding='utf-8', newline='') as fh:
    fh.write('# no measured shading-pair or sensor positions are recorded yet (MEASURE_SHADING CSV empty, VSENSOR CSV without positions, 10-15 records without xyz)\n')
    fh.write('# tool: static/tau_obs.py probe_ratios(); test passed (single sheet, near-vertical beam: below/above = 0.070 for v1_tau, 0 for v1)\n')
    fh.write('individual,probe_id,x_mm,y_mm,z_above_mm,z_below_mm,ratio_v1,ratio_tau\n')
# figures
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
if rows:
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor='white'); x = np.arange(len(rows)); w = 0.26
    for i, (k, lab, c) in enumerate((('d_lower', '아래 1/3', '#3b528b'), ('d_middle', '가운데 1/3', '#21918c'), ('d_upper', '위 1/3', '#fde725'))):
        ax.bar(x + (i - 1) * w, [r[k] for r in rows], w, label=lab, color=c, edgecolor='k', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([r['plant'] for r in rows], rotation=30); ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('흡수광 변화 % (v1_tau τ 0.07, a = 1, v1 대비)'); ax.set_title('잎 투과를 넣으면 높이 층별로 얼마나 바뀌나'); ax.legend(frameon=False); ax.grid(axis='y', alpha=0.3)
    fig.text(0.01, 0.01, TAG, fontsize=6.5); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / 'v1_vs_tau_by_height.png', dpi=170); plt.close(fig)
t2 = json.load(open(HERE / 'out_k1/tests.json', encoding='utf-8'))['T2']['results']; dec = json.load(open(HERE / 'T2_decomposition.json', encoding='utf-8'))
keys = list(t2['v1']['dup']['rel_change'])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor='white', sharey=True)
for ax, var, ttl in ((axes[0], 'dup', '서펠 두 배(지터 0.2 mm)'), (axes[1], 'thin', '서펠 절반')):
    x = np.arange(len(keys)); w = 0.28
    ax.bar(x - w, [100 * t2['v1'][var]['rel_change'][k] for k in keys], w, label='v1 변화', color='#888')
    ax.bar(x, [100 * t2['tau0.07'][var]['rel_change'][k] for k in keys], w, label='v1_tau 변화 (T2, 불합격)', color='#d95f02')
    ax.bar(x + w, [100 * dec[var][k] for k in keys], w, label='투과 몫 (T2b, 통과)', color='#1b9e77')
    for yy in (2, -2):
        ax.axhline(yy, color='r', lw=0.8, ls='--')
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=60, ha='right', fontsize=7); ax.set_title(ttl); ax.grid(axis='y', alpha=0.3)
axes[0].set_ylabel('기준 장면 대비 변화 %'); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('T2 서펠 중복·솎기 불변성 — 점선 = ±2 % 기준'); fig.text(0.01, 0.01, TAG, fontsize=6.5); fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(FIG / 'T2_bars.png', dpi=170); plt.close(fig)
print('written', len(rows), 'plants,', len(lrows), 'leaf rows')
