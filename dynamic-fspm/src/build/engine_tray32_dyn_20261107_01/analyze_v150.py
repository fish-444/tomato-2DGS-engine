"""Amendment 4: canopy light and dry mass against the V opening angle (90 / 120 / 150 / 180 = flat), per lamp height above the substrate (30 / 34 / 54 cm)
and lamp position (trays / gaps). python -X utf8 analyze_v150.py -> v150_angle.csv, fig_v150_angle.png"""
import csv, glob, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent; G = H.parent / 'engine_gui_canopy_20261106_01'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})
ANG = {'V90': 90, 'V120': 120, 'V150': 150, 'flat': 180}
HTS = {30: 's30', 34: '40', 54: '60'}            # cm above the substrate -> case suffix
HCOL = {30: '#b2182b', 34: '#08306b', 54: '#00441b'}


def measures(case):
    fs = glob.glob(str(G / 'kout' / '*' / 'out' / case / 'daily.csv'))
    if not fs: return None
    rows = [r for r in csv.reader(l for l in open(fs[0], encoding='utf-8') if not l.startswith('#'))][1:]
    can = {int(r[0]): r for r in rows if r[2] == 'CANOPY'}; last = max(can)
    if last < 71: return None
    pl = [r for r in rows if r[0] == str(last) and r[2] != 'CANOPY']
    return dict(light_pct=float(can[last][12].split('=')[1]), dm_g=st.median(float(r[4]) for r in pl), leaf_cm2=st.median(float(r[3]) for r in pl))


recs = []
for h, suf in HTS.items():
    for pos, p in (('trays', 'K'), ('gaps', 'G')):
        for sh, a in ANG.items():
            m = measures(f'{p}_{sh}_{suf}')
            if m: recs.append(dict(above_substrate_cm=h, position=pos, shape=sh, opening_deg=a, **m))
            else: print('missing', p, sh, suf)
with open(H / 'v150_angle.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(recs[0])); w.writeheader(); w.writerows(recs)
R = {(r['above_substrate_cm'], r['position'], r['shape']): r for r in recs}
for h in HTS:
    for pos in ('trays', 'gaps'):
        v = {sh: R[(h, pos, sh)]['light_pct'] for sh in ANG if (h, pos, sh) in R}
        best = max(v, key=v.get); peak = 'V150 peak (>1 % over flat and V120)' if v.get('V150', 0) > 1.01 * max(v['flat'], v['V120']) else f'best {best} (no peak beyond 1 %)'
        print(f"{h} cm {pos:5s}: " + ' / '.join(f'{sh} {x:.1f}' for sh, x in v.items()) + f" | V150 vs flat {(v.get('V150', float('nan')) / v['flat'] - 1) * 100:+.1f} % | {peak}")

fig, axs = plt.subplots(1, 3, figsize=(20, 5.8))
for h in HTS:
    for pos, ls, mk in (('trays', '-', 'o'), ('gaps', '--', 's')):
        sh = [s for s in ANG if (h, pos, s) in R]; x = [ANG[s] for s in sh]
        lab = f'배지 위 {h} cm, {"트레이 위" if pos == "trays" else "틈 위"}'
        axs[0].plot(x, [R[(h, pos, s)]['light_pct'] for s in sh], color=HCOL[h], ls=ls, marker=mk, lw=2.6, ms=8, label=lab)
        axs[1].plot(x, [(R[(h, pos, s)]['light_pct'] / R[(h, pos, 'flat')]['light_pct'] - 1) * 100 for s in sh], color=HCOL[h], ls=ls, marker=mk, lw=2.6, ms=8, label=lab)
        axs[2].plot(x, [R[(h, pos, s)]['dm_g'] for s in sh], color=HCOL[h], ls=ls, marker=mk, lw=2.6, ms=8, label=lab)
axs[1].axhline(0, color='k', lw=.8)
for ax, t in zip(axs, ('캐노피 빛 59.9 DAG (%)', '같은 높이·위치의 평판 대비 (%)', '개체 건물중 중앙값 59.9 DAG (g)')):
    ax.set_title(t, fontsize=13, fontweight='bold'); ax.set_xlabel('램프 날개 사이 각도 (°, 180 = 평판)'); ax.set_xticks([90, 120, 150, 180])
    ax.grid(alpha=.35); ax.spines[['top', 'right']].set_visible(False)
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=6, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.035))
fig.suptitle('동적 엔진 v2.2 A, tray32 (간격 10 cm, 틈 10 cm): 날개 각도 90–180° — 시나리오 비교, 검증 아님', fontsize=14)
fig.text(0.5, 0.005, 'V150 = wing rotation 15° (test build of the V120 lamp). 34 / 54 cm above the substrate = 40 / 60 cm from the pot bottom. One seed; differences under ~1 % are not separated from noise. '
         'GroIMP area lights (broader than cos¹ beyond ~18°); black leaves.', ha='center', fontsize=9.5, color='#333')
fig.tight_layout(rect=(0, 0.10, 1, 0.95)); fig.savefig(H / 'fig_v150_angle.png', dpi=120)
