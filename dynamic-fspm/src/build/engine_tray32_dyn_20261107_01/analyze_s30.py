"""Amendment 3: lamp 30 cm above the substrate, over gaps and over trays (batch DT4), next to the 34 / 54 cm scenes (40 / 60 from the pot bottom).
python -X utf8 analyze_s30.py -> s30_scenes.csv, fig_s30_gaps_trays.png"""
import csv, glob, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent; G = H.parent / 'engine_gui_canopy_20261106_01'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})
COL = {'flat': '#00441b', 'V120': '#08306b', 'V90': '#b2182b'}; LAB = {'flat': '평판', 'V120': 'V 120°', 'V90': 'V 90°'}
CASES = {30: '{p}_{s}_s30', 34: '{p}_{s}_40', 54: '{p}_{s}_60'}       # cm above the substrate -> case name


def measures(case):
    fs = glob.glob(str(G / 'kout' / '*' / 'out' / case / 'daily.csv'))
    if not fs: return None
    rows = [r for r in csv.reader(l for l in open(fs[0], encoding='utf-8') if not l.startswith('#'))][1:]
    can = {int(r[0]): r for r in rows if r[2] == 'CANOPY'}; last = max(can)
    pl = [r for r in rows if r[0] == str(last) and r[2] != 'CANOPY']
    upd = [r for d, r in sorted(can.items()) if r[10] not in ('-1', '') and int(r[0]) == int(r[10]) + 1 and r[11] not in ('', 'NaN')]
    u = upd[-1]; outer = [float(r[4]) for r in pl if (int(r[2][1:]) - 1) // 8 in (0, 3)]; inner = [float(r[4]) for r in pl if (int(r[2][1:]) - 1) // 8 in (1, 2)]
    return dict(last_day=last, light_pct=float(can[last][12].split('=')[1]), dm_g=st.median(float(r[4]) for r in pl), leaf_cm2=st.median(float(r[3]) for r in pl),
                height_mm=st.median(float(r[5]) for r in pl), max_height_mm=max(float(r[5]) for r in pl), fruits=st.median(float(r[7]) for r in pl),
                outer_over_inner_dm=st.mean(outer) / st.mean(inner), shade_pct=100 * float(u[11]), light_at_shade_pct=float(u[12].split('=')[1]),
                index=float(u[12].split('=')[1]) / (100 * float(u[11])))


recs = []
for h, pat in CASES.items():
    for pos, p in (('trays', 'K'), ('gaps', 'G')):
        for s in ('flat', 'V120', 'V90'):
            m = measures(pat.format(p=p, s=s))
            if m: recs.append(dict(above_substrate_cm=h, position=pos, shape=s, **m))
            else: print('missing', h, pos, s)
with open(H / 's30_scenes.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(recs[0])); w.writeheader(); w.writerows(recs)
R = {(r['above_substrate_cm'], r['position'], r['shape']): r for r in recs}
print('30 cm above substrate, 59.9 DAG:')
for pos in ('trays', 'gaps'):
    for s in ('flat', 'V120', 'V90'):
        r = R.get((30, pos, s))
        if r: print(f"  {pos:5s} {s:5s} light {r['light_pct']:5.1f} % | vs flat same position {(r['light_pct'] / R[(30, pos, 'flat')]['light_pct'] - 1) * 100:+5.1f} % | vs flat trays {(r['light_pct'] / R[(30, 'trays', 'flat')]['light_pct'] - 1) * 100:+5.1f} % | "
                    f"DM {r['dm_g']:.2f} g | leaf {r['leaf_cm2']:.0f} | h {r['height_mm']:.0f} (max {r['max_height_mm']:.0f}) mm | shading {r['shade_pct']:.1f} % | index {r['index']:.2f} | outer/inner DM {r['outer_over_inner_dm']:.2f}")

fig, axs = plt.subplots(1, 4, figsize=(24, 5.6))
for s in ('flat', 'V120', 'V90'):
    for pos, ls, mk in (('trays', '-', 'o'), ('gaps', '--', 's')):
        hs = [h for h in CASES if (h, pos, s) in R]
        for k, key in enumerate(('light_pct', 'dm_g', 'shade_pct', 'index')):
            axs[k].plot(hs, [R[(h, pos, s)][key] for h in hs], color=COL[s], ls=ls, marker=mk, lw=2.6, ms=8, label=f'{LAB[s]}, {"트레이 위" if pos == "trays" else "틈 위"}')
for ax, t in zip(axs, ('캐노피 빛 59.9 DAG (%)', '개체 건물중 중앙값 59.9 DAG (g)', '평균 이웃 그늘 DAG 51 (%)', '빛 ÷ 그늘 지수 DAG 51')):
    ax.set_title(t, fontsize=13, fontweight='bold'); ax.set_xlabel('배지면 위 램프 높이 (cm)'); ax.set_xticks(list(CASES)); ax.grid(alpha=.35); ax.spines[['top', 'right']].set_visible(False)
axs[0].legend(fontsize=10)
fig.suptitle('동적 엔진 v2.2 A, tray32 (간격 10 cm, 틈 10 cm): 배지면 위 30 cm 추가 — 시나리오 비교, 검증 아님', fontsize=14)
fig.text(0.5, 0.005, '30 cm above the substrate = 36 cm from the pot bottom; 34 / 54 cm = the earlier 40 / 60 cm scenes. Median plant 233 mm, so at 30 cm the lamp is ~7 cm above the median top at the end; 1 of 32 plants reaches 300 mm from DAG 44 (the engine does not stop it). '
         'One seed; GroIMP area lights (broader than cos¹ beyond ~18°); black leaves; neighbour shading = 1 - in canopy ÷ alone.', ha='center', fontsize=9.5, color='#333')
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(H / 'fig_s30_gaps_trays.png', dpi=120)
