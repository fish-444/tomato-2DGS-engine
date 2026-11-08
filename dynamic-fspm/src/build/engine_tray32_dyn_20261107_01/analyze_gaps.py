"""Over-gaps scenes (CONTRACT Amendment 2) vs over-trays, dynamic and static. python -X utf8 analyze_gaps.py -> gaps_vs_trays.csv, figure"""
import csv, glob, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent; G = H.parent / 'engine_gui_canopy_20261106_01'
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
STATIC = {('flat', 'trays', 40): 6.26, ('flat', 'trays', 60): 3.87, ('flat', 'gaps', 40): 5.80, ('flat', 'gaps', 60): 3.74,
          ('V120', 'trays', 40): 5.22, ('V120', 'trays', 60): 3.24, ('V120', 'gaps', 40): 5.30, ('V120', 'gaps', 60): 3.36,
          ('V90', 'trays', 40): 4.35, ('V90', 'trays', 60): 2.66, ('V90', 'gaps', 40): 4.59, ('V90', 'gaps', 60): 2.84}


def load(case):
    fs = glob.glob(str(G / 'kout' / '*' / 'out' / case / 'daily.csv'))
    return [r for r in csv.reader(l for l in open(fs[0], encoding='utf-8') if not l.startswith('#'))][1:] if fs else None


def measures(case):
    rows = load(case)
    if not rows: return None
    can = {int(r[0]): r for r in rows if r[2] == 'CANOPY'}; last = max(can)
    pl = [r for r in rows if r[0] == str(last) and r[2] != 'CANOPY']
    upd = [r for d, r in sorted(can.items()) if r[10] not in ('-1', '') and int(r[0]) == int(r[10]) + 1 and r[11] not in ('', 'NaN')]
    u = upd[-1] if upd else None
    outer = [float(r[4]) for r in pl if (int(r[2][1:]) - 1) // 8 in (0, 3)]; inner = [float(r[4]) for r in pl if (int(r[2][1:]) - 1) // 8 in (1, 2)]
    return dict(light_pct=float(can[last][12].split('=')[1]), dm_g=st.median(float(r[4]) for r in pl), leaf_cm2=st.median(float(r[3]) for r in pl),
                height_mm=st.median(float(r[5]) for r in pl), outer_over_inner_dm=st.mean(outer) / st.mean(inner),
                shade_pct=100 * float(u[11]) if u else float('nan'), light_at_shade_pct=float(u[12].split('=')[1]) if u else float('nan'), shade_dag=float(u[1]) if u else float('nan'))


rows = []
for h in (40, 60):
    for sh in ('flat', 'V120', 'V90'):
        t = measures(f'K_{sh}_{h}'); g = measures(f'G_{sh}_{h}')
        if not (t and g): print(sh, h, 'missing'); continue
        rec = dict(shape=sh, height_cm=h, trays_light_pct=t['light_pct'], gaps_light_pct=g['light_pct'], gaps_vs_trays_light_pct=(g['light_pct'] / t['light_pct'] - 1) * 100,
                   static_trays_pct=STATIC[(sh, 'trays', h)], static_gaps_pct=STATIC[(sh, 'gaps', h)], static_gaps_vs_trays_pct=(STATIC[(sh, 'gaps', h)] / STATIC[(sh, 'trays', h)] - 1) * 100,
                   trays_dm_g=t['dm_g'], gaps_dm_g=g['dm_g'], gaps_vs_trays_dm_pct=(g['dm_g'] / t['dm_g'] - 1) * 100, trays_leaf_cm2=t['leaf_cm2'], gaps_leaf_cm2=g['leaf_cm2'],
                   trays_outer_over_inner_dm=t['outer_over_inner_dm'], gaps_outer_over_inner_dm=g['outer_over_inner_dm'],
                   trays_shade_pct=t['shade_pct'], gaps_shade_pct=g['shade_pct'], shade_dag=g['shade_dag'],
                   trays_index=t['light_at_shade_pct'] / t['shade_pct'], gaps_index=g['light_at_shade_pct'] / g['shade_pct'], gaps_height_mm=g['height_mm'])
        rows.append(rec)
        print(f"{sh:5s} {h}: light trays {t['light_pct']:.1f} gaps {g['light_pct']:.1f} ({rec['gaps_vs_trays_light_pct']:+.1f} %) | static {rec['static_gaps_vs_trays_pct']:+.1f} % | "
              f"DM {t['dm_g']:.2f}->{g['dm_g']:.2f} g ({rec['gaps_vs_trays_dm_pct']:+.1f} %) | shading {t['shade_pct']:.1f}->{g['shade_pct']:.1f} % | index {rec['trays_index']:.2f}->{rec['gaps_index']:.2f} | outer/inner DM {t['outer_over_inner_dm']:.2f}->{g['outer_over_inner_dm']:.2f}")
if rows:
    with open(H / 'gaps_vs_trays.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4)); lab = [f"{r['shape']}\n{r['height_cm']}" for r in rows]; x = np.arange(len(rows))
    axs[0].bar(x - 0.2, [r['trays_light_pct'] for r in rows], 0.4, label='트레이 위', color='#90a4ae'); axs[0].bar(x + 0.2, [r['gaps_light_pct'] for r in rows], 0.4, label='틈 위', color='#2e7d32')
    axs[0].set_xticks(x); axs[0].set_xticklabels(lab); axs[0].set_ylabel('캐노피 빛 59.9 DAG (%)'); axs[0].legend(); axs[0].set_title('동적 캐노피 빛')
    axs[1].bar(x - 0.2, [r['gaps_vs_trays_light_pct'] for r in rows], 0.4, label='동적', color='#2e7d32'); axs[1].bar(x + 0.2, [r['static_gaps_vs_trays_pct'] for r in rows], 0.4, label='정적', color='#8d6e63')
    axs[1].axhline(0, color='k', lw=.6); axs[1].set_xticks(x); axs[1].set_xticklabels(lab); axs[1].set_ylabel('틈 위 ÷ 트레이 위 − 1 (%)'); axs[1].legend(); axs[1].set_title('위치 효과: 동적 vs 정적')
    axs[2].bar(x - 0.2, [r['trays_dm_g'] for r in rows], 0.4, label='트레이 위', color='#90a4ae'); axs[2].bar(x + 0.2, [r['gaps_dm_g'] for r in rows], 0.4, label='틈 위', color='#6d4c41')
    axs[2].set_xticks(x); axs[2].set_xticklabels(lab); axs[2].set_ylabel('개체 건물중 중앙값 (g)'); axs[2].legend(); axs[2].set_title('59.9 DAG 건물중')
    fig.suptitle('동적 엔진 v2.2 A, tray32 32개체, 간격 10 cm — 램프 위치(트레이 위 / 틈 위), 시나리오 비교, 검증 아님', fontsize=11); fig.tight_layout(); fig.savefig(H / 'tray32_dyn_gaps_vs_trays.png', dpi=120)
