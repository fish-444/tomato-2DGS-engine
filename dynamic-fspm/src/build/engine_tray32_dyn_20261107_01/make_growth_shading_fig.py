"""Dynamic tray32 scenes in the layout of the engine v1/v2/v2.1 figure: leaf area, dry mass, height, trusses, fruits over DAG
(median and 5-95 % over the 32 plants) plus neighbour shading on the update days (every 10 days). Rows: lamps over trays (K_*),
lamps over gaps (G_*). User request 2026-11-07. Reads ../engine_gui_canopy_20261106_01/kout/*/out/{K,G}_*/daily.csv (no new runs).
python -X utf8 make_growth_shading_fig.py -> fig_dynamic_growth_shading.png"""
import csv, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent; G = H.parent / 'engine_gui_canopy_20261106_01'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})
SHAPE_COL = {'flat': '#00441b', 'V120': '#08306b', 'V90': '#b2182b'}       # dark green / navy / dark red
SHAPE_LAB = {'flat': '평판', 'V120': 'V 120°', 'V90': 'V 90°'}
TRAITS = [(3, '잎면적 (cm²)'), (4, '건물중 (g)'), (5, '키 (mm)'), (6, '화방 수'), (7, '열매 수')]


def series(case):
    fs = glob.glob(str(G / 'kout' / '*' / 'out' / case / 'daily.csv'))
    if not fs: return None
    rows = [r for r in csv.reader(l for l in open(fs[0], encoding='utf-8') if not l.startswith('#'))][1:]
    by_day = defaultdict(list); dag = {}; shade = defaultdict(list)
    for r in rows:
        if r[2] == 'CANOPY': continue
        d = int(r[0]); dag[d] = float(r[1]); by_day[d].append([float(r[i]) for i, _ in TRAITS])
        if r[11] not in ('', 'NaN') and r[10] not in ('', '-1') and int(r[10]) == d - 1: shade[d].append(100 * float(r[11]))
    days = sorted(d for d in by_day if dag[d] >= 0)
    x = np.array([dag[d] for d in days]); v = [np.array(by_day[d]) for d in days]
    out = {'x': x, 'traits': [np.array([np.percentile(a[:, k], q) for a in v]) for k in range(len(TRAITS)) for q in (50, 5, 95)]}
    sd = sorted(d for d in shade if dag[d] >= 0)
    out['sx'] = np.array([dag[d] for d in sd]); out['s'] = [np.array([np.percentile(shade[d], q) for d in sd]) for q in (50, 5, 95)]
    return out


fig, axs = plt.subplots(2, 6, figsize=(30, 10.5), sharex=True)
for i, (pos, pre) in enumerate((('트레이 위 램프', 'K'), ('틈 위 램프', 'G'))):
    for h in (40, 60):
        for sh in ('flat', 'V120', 'V90'):
            s = series(f'{pre}_{sh}_{h}')
            if s is None: print('missing', pre, sh, h); continue
            c = SHAPE_COL[sh]; ls = '-' if h == 40 else '--'; lab = f'{SHAPE_LAB[sh]}, {h} cm'
            for k in range(len(TRAITS)):
                m, lo, hi = s['traits'][3 * k:3 * k + 3]
                axs[i, k].plot(s['x'], m, color=c, ls=ls, lw=2.4, label=lab); band = (sh, h) == ('flat', 40)
                if band: axs[i, k].fill_between(s['x'], lo, hi, color='#9e9e9e', alpha=.30, lw=0, label='5–95 % (평판 40 cm)' if k == 0 else None)
            m, lo, hi = s['s']
            axs[i, 5].plot(s['sx'], m, color=c, ls=ls, lw=2.4, marker='o', ms=6, label=lab)
            if band: axs[i, 5].fill_between(s['sx'], lo, hi, color='#9e9e9e', alpha=.30, lw=0)
    for k, (_, t) in enumerate(TRAITS + [(None, '이웃 그늘 (%, 10일마다)')]):
        ax = axs[i, k]; ax.set_title(t if i == 0 else '', fontsize=14, fontweight='bold'); ax.grid(alpha=.35)
        ax.spines[['top', 'right']].set_visible(False); ax.set_xlim(0, 60); ax.set_ylim(bottom=0)
        if i == 1: ax.set_xlabel('DAG')
    axs[i, 0].set_ylabel(pos, fontsize=14, fontweight='bold')
axs[0, 0].legend(fontsize=10, loc='upper left')
fig.suptitle('동적 엔진 v2.2 A, tray32 (32개체, 간격 10 cm): 32개체 중앙값과 5–95 % (relative_light, 검증 아님)', fontsize=15)
fig.text(0.5, 0.005, 'Lines = median over 32 plants, grey band = 5-95 % over plants for flat 40 cm only (other scenes have similar spread). Height, trusses and fruits are the same in every scene, so those lines overlap. Neighbour shading = 1 - absorbed in canopy ÷ absorbed alone, per plant, on update days (alone pass every 10 days); '
         'not the static self + neighbour shading. Growth is identical with and without the alone passes (2343/2343 rows). One seed; GroIMP area lights.', ha='center', fontsize=10, color='#333')
fig.tight_layout(rect=(0, 0.025, 1, 0.96)); fig.savefig(H / 'fig_dynamic_growth_shading.png', dpi=110)
print('ok')
