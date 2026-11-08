"""Dynamic tray32 scenarios (CONTRACT.md): per scene at 59.9 DAG (engine day 71) — canopy light, per-plant absorbed, leaf area, dry mass, height,
fruits (median, p10-p90 over 32 plants), outer trays (1, 4) vs inner (2, 3); paired effects vs flat 40; static tray32 v3 table A efficiencies beside.
python -X utf8 analyze_tray32_dyn.py -> scenes_dyn.csv, plants_dyn.csv, daily_canopy_dyn.csv, figure"""
import csv, glob, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent; G = H.parent / 'engine_gui_canopy_20261106_01'
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
SCENES = [(f'D_{sh}_{h}', sh, h) for h in (40, 60) for sh in ('flat', 'V120', 'V90')]
STATIC = {('flat', 40): 6.261, ('V120', 40): 5.216, ('V90', 40): 4.353, ('flat', 60): 3.873, ('V120', 60): 3.236, ('V90', 60): 2.657}


def q(v, p): return float(np.percentile(v, p)) if v else float('nan')


def load(case):
    fs = glob.glob(str(G / 'kout' / '*' / 'out' / case / 'daily.csv'))
    if not fs: return None
    rows = [r for r in csv.reader(l for l in open(fs[0], encoding='utf-8') if not l.startswith('#'))][1:]
    return rows


scenes, plants, curves = [], [], {}
for case, sh, h in SCENES:
    rows = load(case)
    if not rows: print(case, 'no data'); continue
    last = max(int(r[0]) for r in rows)
    pl = [r for r in rows if r[0] == str(last) and r[2] != 'CANOPY']
    can = {int(r[0]): r for r in rows if r[2] == 'CANOPY'}
    light = float(can[last][12].split('=')[1]); curves[case] = [(float(r[1]), float(r[12].split('=')[1])) for d, r in sorted(can.items())]
    def col(i): return [float(r[i]) for r in pl]
    def tray(r): return int(r[2][1:]) - 1 // 1
    for r in pl:
        k = int(r[2][1:]) - 1; t = k // 8 + 1
        plants.append(dict(scene=case, shape=sh, height_cm=h, plant=r[2], tray=t, outer=t in (1, 4), leaf_area_cm2=float(r[3]), dry_mass_g=float(r[4]), height_mm=float(r[5]),
                           trusses=float(r[6]), fruits=float(r[7]), absorbed_umol_s=float(r[8])))
    P = [p for p in plants if p['scene'] == case]
    rec = dict(scene=case, shape=sh, height_cm=h, last_engine_day=last, canopy_light_pct=light, static_efficiency_pct=STATIC[(sh, h)])
    for k in ('absorbed_umol_s', 'leaf_area_cm2', 'dry_mass_g', 'height_mm', 'fruits'):
        v = [p[k] for p in P]; rec[f'{k}_median'] = q(v, 50); rec[f'{k}_p10'] = q(v, 10); rec[f'{k}_p90'] = q(v, 90)
        rec[f'{k}_outer_over_inner'] = st.mean(p[k] for p in P if p['outer']) / st.mean(p[k] for p in P if not p['outer'])
    scenes.append(rec)
# Amendment 1: shading batch (K_ cases, alone passes every 10 days) — mean neighbour shading, index, and DT1 = DT2 growth check
STATIC_SHADE = {('flat', 40): 0.50316, ('V120', 40): 0.5036, ('V90', 40): 0.50598, ('flat', 60): 0.49068, ('V120', 60): 0.49048, ('V90', 60): 0.4917}
shade_curves = {}
for s in scenes:
    sh, h = s['shape'], s['height_cm']; rk = load(f'K_{sh}_{h}'); rd = load(s['scene'])
    s['static_total_shading_mean_pct'] = 100 * STATIC_SHADE[(sh, h)]
    if not rk: s['mean_neighbour_shading_pct'] = float('nan'); s['index_light_over_shading'] = float('nan'); s['dt1_dt2_identical'] = 'no DT2 data'; continue
    key = lambda rows: {(r[0], r[2]): r[3:9] for r in rows}
    a, b = key(rd), key(rk); s['dt1_dt2_identical'] = f"{sum(1 for k in a if a[k] == b.get(k))}/{len(a)}"
    can = {int(r[0]): r for r in rk if r[2] == 'CANOPY'}; last = max(can)
    sh_last = float(can[last][11]) if can[last][11] not in ('', 'NaN') else float('nan')
    s['mean_neighbour_shading_pct'] = 100 * sh_last; s['alone_day'] = int(can[last][10])
    s['index_light_over_shading'] = s['canopy_light_pct'] / s['mean_neighbour_shading_pct'] if s['mean_neighbour_shading_pct'] >= 0.5 else float('nan')
    shade_curves[s['scene']] = [(float(r[1]), 100 * float(r[11]), float(r[12].split('=')[1])) for d, r in sorted(can.items()) if r[11] not in ('', 'NaN') and int(r[10]) == d - 1]
ref = next((s for s in scenes if s['scene'] == 'D_flat_40'), None)
for s in scenes:
    for k in ('canopy_light_pct', 'dry_mass_g_median', 'leaf_area_cm2_median'):
        s[f'{k}_vs_flat40_pct'] = (s[k] / ref[k] - 1) * 100 if ref else float('nan')
    s['static_vs_flat40_pct'] = (s['static_efficiency_pct'] / STATIC[('flat', 40)] - 1) * 100
for nm, data in (('scenes_dyn.csv', scenes), ('plants_dyn.csv', plants)):
    if data:
        with open(H / nm, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, list(data[0])); w.writeheader(); w.writerows(data)
for s in scenes:
    print(f"{s['scene']:12s} light {s['canopy_light_pct']:5.1f} %  DM {s['dry_mass_g_median']:.2f} g  leaf {s['leaf_area_cm2_median']:.0f} cm2  h {s['height_mm_median']:.0f} mm  fruits {s['fruits_median']:.0f}  "
          f"outer/inner DM {s['dry_mass_g_outer_over_inner']:.2f}  vs flat40: light {s['canopy_light_pct_vs_flat40_pct']:+.0f} %, DM {s['dry_mass_g_median_vs_flat40_pct']:+.0f} %  | static eff {s['static_efficiency_pct']:.2f} % ({s['static_vs_flat40_pct']:+.0f} %)"
          f"  | shading {s.get('mean_neighbour_shading_pct', float('nan')):.1f} % index {s.get('index_light_over_shading', float('nan')):.2f} (DT1=DT2 {s.get('dt1_dt2_identical')}; static total shading {s['static_total_shading_mean_pct']:.1f} %)")
if scenes:
    fig, axs = plt.subplots(1, 4, figsize=(21, 4.6))
    for case, sh, h in SCENES:   # 4th panel: mean neighbour shading and index on update days (shading batch)
        if case not in shade_curves or not shade_curves[case]: continue
        c = {'flat': '#2e7d32', 'V120': '#1565c0', 'V90': '#ef6c00'}[sh]; ls = '-' if h == 40 else '--'
        x, y, l = zip(*shade_curves[case]); axs[3].plot(x, y, color=c, ls=ls, marker='o', ms=3, label=f'{sh} {h} cm')
    axs[3].set_xlim(0, 60); axs[3].set_xlabel('DAG'); axs[3].set_ylabel('평균 이웃 그늘 (%)'); axs[3].set_title('평균 이웃 그늘 (10일마다 계산)'); axs[3].legend(fontsize=8)
    for case, sh, h in SCENES:
        if case not in curves: continue
        c = {'flat': '#2e7d32', 'V120': '#1565c0', 'V90': '#ef6c00'}[sh]; ls = '-' if h == 40 else '--'
        x, y = zip(*curves[case]); axs[0].plot(x, y, color=c, ls=ls, label=f'{sh} {h} cm')
    axs[0].set_xlim(0, 60); axs[0].set_xlabel('DAG'); axs[0].set_ylabel('캐노피 빛 (방출 대비 %)'); axs[0].legend(fontsize=8); axs[0].set_title('캐노피 빛 (동적)')
    lab = [f"{s['shape']}\n{s['height_cm']}" for s in scenes]; xx = np.arange(len(scenes))
    axs[1].bar(xx - 0.2, [s['dry_mass_g_median'] for s in scenes], 0.4, color='#6d4c41', label='건물중 중앙값 (g)')
    axs[1].errorbar(xx - 0.2, [s['dry_mass_g_median'] for s in scenes], yerr=[[s['dry_mass_g_median'] - s['dry_mass_g_p10'] for s in scenes], [s['dry_mass_g_p90'] - s['dry_mass_g_median'] for s in scenes]], fmt='none', ecolor='k', lw=.8)
    ax2 = axs[1].twinx(); ax2.bar(xx + 0.2, [s['leaf_area_cm2_median'] for s in scenes], 0.4, color='#66bb6a', label='잎면적 중앙값 (cm²)')
    axs[1].set_xticks(xx); axs[1].set_xticklabels(lab); axs[1].set_title('59.9 DAG 개체 건물중(갈색)·잎면적(초록), p10–p90')
    axs[2].scatter([s['static_efficiency_pct'] for s in scenes], [s['canopy_light_pct'] for s in scenes], c=['#2e7d32' if s['shape'] == 'flat' else '#1565c0' if s['shape'] == 'V120' else '#ef6c00' for s in scenes], s=60)
    for s in scenes: axs[2].annotate(f"{s['shape']} {s['height_cm']}", (s['static_efficiency_pct'], s['canopy_light_pct']), fontsize=8, xytext=(4, 3), textcoords='offset points')
    axs[2].set_xlabel('정적 효율 (%, 파일럿 개체 고정 형상, τ 0.07)'); axs[2].set_ylabel('동적 캐노피 빛 59.9 DAG (%, 검정 잎)'); axs[2].set_title('정적 vs 동적 (순위 비교만)')
    fig.suptitle('동적 엔진 v2.2 A, tray32 (32개체, 간격 10 cm, 트레이 위 램프 4개) — 시나리오 비교, 검증 아님', fontsize=11); fig.tight_layout()
    fig.savefig(H / 'tray32_dyn_scenarios.png', dpi=120)
