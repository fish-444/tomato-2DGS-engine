"""v3 figures: efficiency vs height (panel per spacing), copy heat maps at 30 / 60 cm (reference geometry), v2 vs v3 per-copy scatter at 40 / 50 cm. python -X utf8 make_figures_v3.py"""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True); V2 = HERE.parents[0] / 'static_tray32_v2_20261104_01'
import tray32_lib as T
plt.rcParams.update({'font.family': ['Malgun Gothic', 'DejaVu Sans'], 'axes.unicode_minus': False})
TAG = 'light_model_v1tau_smoothn_r2; assumed_luminaire_model; repeated_placement; relative_light; conditional_on_unvalidated_scale; scenario_comparison_not_validation; absolute_efficiency_uncertain_up_to_1.9x'
def rd(p):
    return list(csv.DictReader(l for l in open(p, encoding='utf-8') if not l.startswith('#')))
E = rd(HERE / 'efficiency_by_scene.csv'); PL = rd(HERE / 'plants_by_scene.csv'); SC = {s['sid']: s for s in T.scene_list()}
COL = {'flat': '#c23b3b', 'V120': '#2a7ab9', 'V90': '#3a9e5c'}; LS = {'trays': '-', 'gaps': '--'}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, p in zip(axs, T.SPACINGS):
    for sh in T.SHAPES:
        for pos in T.POSITIONS:
            for t, mk in (('A', 'o'), ('B', 'x')):
                v = sorted((float(e['height_m']), float(e['efficiency_a085'])) for e in E if e['table'] == t and e['shape'] == sh and e['position'] == pos and float(e['spacing_m']) == p)
                if v:
                    ax.plot([100 * h for h, _ in v], [100 * y for _, y in v], LS[pos], marker=mk, ms=4, color=COL[sh], alpha=1.0 if t == 'A' else 0.45, label=f'{sh} over {pos}' if t == 'A' else None)
    ax.set_title(f'p = {int(p * 100)} cm', fontsize=10); ax.set_xlabel('luminaire height (cm)'); ax.set_xticks([30, 40, 50, 60]); ax.grid(alpha=0.25)
axs[0].set_ylabel('interception efficiency (%)'); axs[0].legend(frameon=False, fontsize=7)
fig.suptitle('Efficiency vs luminaire height (v1tau_smoothn_r2; table A solid markers, B faint ×; absolute level uncertain up to 1.9×)', fontsize=10); fig.text(0.01, 0.005, TAG, fontsize=6)
fig.tight_layout(rect=(0, 0.02, 1, 0.94)); fig.savefig(FIG / 'efficiency_vs_height.png', dpi=130); plt.close(fig)
# heat maps at 30 / 60 cm, reference geometry (flat, over trays, p 0.10), table A, common colour scale
sids = {h: next(s for s, d in SC.items() if d['p'] == 0.10 and d['shape'] == 'flat' and d['pos'] == 'trays' and d['height'] == h) for h in (0.30, 0.60)}
vals = {h: [r for r in PL if r['table'] == 'A' and r['sid'] == sids[h]] for h in sids}
if all(vals.values()):
    allv = np.concatenate([[float(r['absorbed_a085']) for r in v] for v in vals.values()]); lo, hi = np.percentile(allv, 2), np.percentile(allv, 98)
    fig, axs = plt.subplots(2, 1, figsize=(9, 7)); p = 0.10
    for ax, h in zip(axs, (0.30, 0.60)):
        for r in vals[h]:
            x, y = T.slot_xy(p, int(r['tray']), int(r['row']), int(r['column'])); ax.scatter([x], [y], c=[float(r['absorbed_a085'])], cmap='viridis', vmin=lo, vmax=hi, s=420, marker='s')
        for xc in T.tray_x(p):
            ax.add_patch(Rectangle((xc - p, -2 * p), 2 * p, 4 * p, fill=False, ec='#444', lw=0.8))
        e = next(x for x in E if x['table'] == 'A' and x['sid'] == sids[h]); ax.set_aspect('equal'); ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.23, 0.23)
        ax.set_title(f"{sids[h]}: flat, over trays, p 10 cm, {int(h * 100)} cm — efficiency {100 * float(e['efficiency_a085']):.2f} %", fontsize=9); ax.tick_params(labelsize=7)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap='viridis'), ax=axs, shrink=0.6, label='absorbed per copy (a 0.85, L units), same scale')
    fig.text(0.01, 0.005, TAG, fontsize=6); fig.savefig(FIG / 'copy_heatmap_30_60cm.png', dpi=130); plt.close(fig)
# v2 vs v3 scatter per copy (40 / 50 cm, table A)
pv2 = rd(V2 / 'plants_by_scene.csv'); d2 = {(r['table'], r['sid'], r['copy']): float(r['absorbed_a085']) for r in pv2}
fig, ax = plt.subplots(figsize=(5.6, 5.4))
for h, c in ((0.40, '#2a7ab9'), (0.50, '#c23b3b')):
    xs, ys = [], []
    for r in PL:
        s = SC[r['sid']]
        if r['table'] == 'A' and s['height'] == h and s['v2_sid'] and ('A', s['v2_sid'], r['copy']) in d2:
            xs.append(d2[('A', s['v2_sid'], r['copy'])]); ys.append(float(r['absorbed_a085']))
    if xs:
        ax.scatter(xs, ys, s=4, alpha=0.5, color=c, label=f'{int(h * 100)} cm (n={len(xs)})')
mx = ax.get_xlim()[1]; ax.plot([0, mx], [0, mx], color='#888', lw=0.8); ax.set_xlabel('absorbed per copy, v2 (raw normals)'); ax.set_ylabel('absorbed per copy, v3 (smoothed r 2 mm)')
ax.legend(frameon=False, fontsize=8); ax.set_title('v2 vs v3 per copy (table A, all 40 / 50 cm scenes)', fontsize=9); fig.text(0.01, 0.005, TAG, fontsize=5); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIG / 'v2_vs_v3_copy_scatter.png', dpi=130); plt.close(fig)
print('figures ok')
