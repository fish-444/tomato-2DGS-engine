"""Figures in the layout of the static 'canopy light ÷ mean neighbour shading' figure (3 rows × 3 lamp shapes), darker colours (user, 2026-11-07).
Static: build/static_tray32_v3_20261105_01 (read only): efficiency_a085 (efficiency_by_scene.csv) and neighbour_shading_share (plants_by_scene.csv),
mean of placement tables A and B. Dynamic: this folder (frozen v2.2 A, 32 plants, p 10 cm, 40 / 60 cm, over trays and over gaps), all three rows
on the same day (last shading update, DAG 50.9) so that row 1 = row 2 ÷ row 3.
python -X utf8 make_light_shading_figs.py -> fig_static_light_over_shading.png, fig_dynamic_light_over_shading.png"""
import csv, glob, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

H = Path(__file__).resolve().parent; S = H.parent / 'static_tray32_v3_20261105_01'; G = H.parent / 'engine_gui_canopy_20261106_01'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 11})
COL = {0.06: '#b2182b', 0.10: '#08306b', 0.14: '#00441b'}          # dark red / dark blue / dark green
SHAPES = [('flat', 'flat'), ('V120', 'V 120°'), ('V90', 'V 90°')]


def rows(p):
    return list(csv.DictReader(l for l in open(p, encoding='utf-8') if not l.startswith('#')))


def draw(data, heights, title, note, out, all_minor=False):
    """data[(shape, spacing, position)] = {height: (light_pct, shading_pct)}"""
    fig, axs = plt.subplots(3, 3, figsize=(16, 13), sharex=True)
    for j, (sh, lab) in enumerate(SHAPES):
        for (s2, sp, pos), d in sorted(data.items()):
            if s2 != sh: continue
            hs = sorted(d); light = [d[h][0] for h in hs]; shade = [d[h][1] for h in hs]
            idx = [l / s if s > 0 else float('nan') for l, s in zip(light, shade)]
            kw = dict(color=COL[sp], lw=2.6, marker='o' if pos == 'trays' else 's', ms=7, ls='-' if pos == 'trays' else '--',
                      label=f'{int(round(sp * 100))} cm, lamps over {pos}')
            axs[0, j].plot(hs, idx, **kw); axs[1, j].plot(hs, light, **kw); axs[2, j].plot(hs, shade, **kw)
        axs[0, j].set_yscale('linear' if all_minor else 'log'); axs[0, j].set_title(lab, fontsize=14, fontweight='bold')
        F = FuncFormatter(lambda v, _: f'{v:g}'); axs[0, j].yaxis.set_major_formatter(F); axs[0, j].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}' if all_minor or round(v / 10 ** __import__('math').floor(__import__('math').log10(v)), 6) in (2, 3, 5) else ''))
        axs[2, j].set_xlabel('lamp height (cm)'); axs[2, j].set_xticks(heights)
        for i in range(3): axs[i, j].grid(alpha=.35); axs[i, j].spines[['top', 'right']].set_visible(False)
    axs[0, 0].set_ylabel('canopy light ÷ mean neighbour shading'); axs[1, 0].set_ylabel('canopy light (interception efficiency, %)')
    axs[2, 0].set_ylabel('mean neighbour shading (%)'); axs[1, 0].legend(fontsize=10, loc='upper right')
    fig.suptitle(title, fontsize=13); fig.text(0.5, 0.004, note, ha='center', fontsize=9, color='#333')
    fig.tight_layout(rect=(0, 0.035, 1, 0.97)); fig.savefig(H / out, dpi=130); plt.close(fig)


# static: mean of tables A and B
eff = defaultdict(list)
for r in rows(S / 'efficiency_by_scene.csv'):
    eff[(r['shape'], round(float(r['spacing_m']), 2), r['position'], round(float(r['height_m']) * 100))].append(100 * float(r['efficiency_a085']))
scene_key = {(r['table'], r['sid']): (r['shape'], round(float(r['spacing_m']), 2), r['position'], round(float(r['height_m']) * 100)) for r in rows(S / 'efficiency_by_scene.csv')}
nb = defaultdict(list)
for r in rows(S / 'plants_by_scene.csv'):
    k = scene_key.get((r['table'], r['sid']))
    if k and r['neighbour_shading_share'] not in ('', 'nan', 'NaN'): nb[k].append(100 * float(r['neighbour_shading_share']))
sdata = defaultdict(dict)
for k, v in eff.items():
    if nb.get(k): sdata[(k[0], k[1], k[2])][k[3]] = (st.mean(v), st.mean(nb[k]))
draw(sdata, [30, 40, 50, 60], '정적 계산 (tray32 v3): 캐노피 빛 ÷ 평균 이웃 그늘',
     'Row 1 = row 2 ÷ row 3 (log scale). Mean of placement tables A and B; v1tau_smoothn_r2; lamps only; relative light; absolute efficiency uncertain up to 1.9×. '
     'The ratio is a derived index, not a physical quantity: it grows sharply when neighbour shading approaches 0.', 'fig_static_light_over_shading.png')

# dynamic: same day (last shading update) for all rows
ddata = defaultdict(dict)
for pos, pre in (('trays', 'K'), ('gaps', 'G')):
    for h in (40, 60):
        for sh, _ in SHAPES:
            fs = glob.glob(str(G / 'kout' / '*' / 'out' / f'{pre}_{sh}_{h}' / 'daily.csv'))
            if not fs: continue
            rr = [r for r in csv.reader(l for l in open(fs[0], encoding='utf-8') if not l.startswith('#'))][1:]
            upd = [r for r in rr if r[2] == 'CANOPY' and r[10] not in ('-1', '') and int(r[0]) == int(r[10]) + 1 and r[11] not in ('', 'NaN')]
            u = upd[-1]; ddata[(sh, 0.10, pos)][h] = (float(u[12].split('=')[1]), 100 * float(u[11]))
draw(ddata, [40, 60], '동적 엔진 v2.2 A (32개체, 간격 10 cm): 캐노피 빛 ÷ 평균 이웃 그늘 — DAG 51',
     'Row 1 = row 2 ÷ row 3 (linear scale), all on the last shading update (DAG 50.9). Canopy light = absorbed by leaflets ÷ emitted (4 × 1200 µmol/s); black leaves; '
     'GroIMP area lights (broader than cos¹ beyond ~18°); one seed.\nNeighbour shading = 1 - in canopy ÷ alone (definition differs from static self + neighbour). Not validation.',
     'fig_dynamic_light_over_shading.png', all_minor=True)
print('ok', len(sdata), len(ddata))
