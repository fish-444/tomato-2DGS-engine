"""Figures of the 32-plant tray scenes (CONTRACT §9). Plots only. python -X utf8 make_figures.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
import tray32_lib as T
plt.rcParams.update({'font.family': ['Malgun Gothic', 'DejaVu Sans'], 'axes.unicode_minus': False})
TAG = 'light_model_v1_tau; assumed_luminaire_model; repeated_placement; relative_light; conditional_on_unvalidated_scale; scenario_comparison_not_validation'
def rd(n):
    return list(csv.DictReader(l for l in open(HERE / n, encoding='utf-8') if not l.startswith('#')))
E = rd('efficiency_by_scene.csv'); PL = rd('plants_by_scene.csv'); PE = rd('position_effect.csv')
COL = {'flat': '#c23b3b', 'V120': '#2a7ab9', 'V90': '#3a9e5c'}

# 1. layout top views
fig, axs = plt.subplots(3, 2, figsize=(13, 9.5))
for i, p in enumerate(T.SPACINGS):
    for j, pos in enumerate(T.POSITIONS):
        ax = axs[i, j]; xs = T.tray_x(p)
        for t, xc in enumerate(xs):
            ax.add_patch(Rectangle((xc - p, -2 * p), 2 * p, 4 * p, fill=False, ec='#444', lw=1))
            for r in range(1, 5):
                for c in (1, 2):
                    x, y = T.slot_xy(p, t + 1, r, c); ax.plot(x, y, 'o', ms=3.5, color='#3a9e5c')
        for f in T.faces('V90', pos, p, 0.40):                                    # footprints of the faces (x extent) for the V shape; flat panels drawn dashed
            ax.add_patch(Rectangle((min(f['xa'], f['xb']), -T.LUM_LEN / 2), abs(f['xb'] - f['xa']), T.LUM_LEN, fc='#e0a800', alpha=0.45, ec='none'))
        for f in T.faces('flat', pos, p, 0.40):
            ax.add_patch(Rectangle((min(f['xa'], f['xb']), -T.LUM_LEN / 2), abs(f['xb'] - f['xa']), T.LUM_LEN, fill=False, ec='#c23b3b', ls='--', lw=1))
        ax.set_aspect('equal'); ax.set_xlim(xs[0] - p - 0.12, xs[3] + p + 0.12); ax.set_ylim(-0.36, 0.36); ax.set_title(f'p = {int(p * 100)} cm, luminaires over {pos}', fontsize=9)
        ax.set_xlabel('x (m)', fontsize=7); ax.set_ylabel('y (m)', fontsize=7); ax.tick_params(labelsize=7)
fig.suptitle('Layout (top view): four 4×2 trays, gap 10 cm, luminaire 60 cm (yellow = V faces, red dashed = flat panels)', fontsize=11); fig.text(0.01, 0.005, TAG, fontsize=6); fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(FIG / 'layout_top_view.png', dpi=130); plt.close(fig)

# 2. efficiency by shape x position x height, one panel per spacing
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, p in zip(axs, T.SPACINGS):
    x = 0
    for pos in T.POSITIONS:
        for h in T.HEIGHTS:
            for k, sh in enumerate(T.SHAPES):
                r = [e for e in E if float(e['spacing_m']) == p and e['shape'] == sh and e['position'] == pos and float(e['height_m']) == h]
                if r:
                    ax.bar(x + k * 0.27, float(r[0]['efficiency_a085']), 0.25, color=COL[sh], label=sh if (pos == 'trays' and h == 0.40) else None)
            ax.text(x + 0.27, -0.0045, f'{pos}\n{int(h * 100)} cm', ha='center', fontsize=7); x += 1.2
    ax.set_title(f'p = {int(p * 100)} cm', fontsize=10); ax.set_xticks([]); ax.grid(axis='y', alpha=0.25); ax.set_ylim(-0.008, 0.068)
axs[0].set_ylabel('interception efficiency (absorbed, a 0.85 / emitted)'); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle('Interception efficiency by shape × position × height (table A; area (b) basis, absolute level uncertain)', fontsize=11); fig.text(0.01, 0.005, TAG, fontsize=6); fig.tight_layout(rect=(0, 0.02, 1, 0.94)); fig.savefig(FIG / 'efficiency_by_scene.png', dpi=130); plt.close(fig)

# 3. grid heat maps of copy absorbed light: reference, best, worst (table A)
def grid_map(ax, sid, p):
    rows = [r for r in PL if r['table'] == 'A' and r['sid'] == sid]; v = np.array([float(r['absorbed_a085']) for r in rows]); lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    xs = T.tray_x(p)
    for r, val in zip(rows, v):
        x, y = T.slot_xy(p, int(r['tray']), int(r['row']), int(r['column'])); ax.scatter([x], [y], c=[val], cmap='viridis', vmin=lo, vmax=hi, s=420 * p / 0.1, marker='s')
        ax.text(x, y, r['plant'].replace('R3-1b', 'R3').replace('t2_', 't').replace('n_', 'n').replace('e_', 'e'), ha='center', va='center', fontsize=5, color='white')
    for xc in xs:
        ax.add_patch(Rectangle((xc - p, -2 * p), 2 * p, 4 * p, fill=False, ec='#444', lw=0.8))
    ax.set_aspect('equal'); ax.set_xlim(xs[0] - p - 0.03, xs[3] + p + 0.03); ax.set_ylim(-2 * p - 0.03, 2 * p + 0.03); ax.tick_params(labelsize=6)
    s = [e for e in E if e['sid'] == sid and e['table'] == 'A'][0]; ax.set_title(f"{sid}: p {s['spacing_m']}, {s['shape']}, {s['position']}, {s['height_m']} m; eff {float(s['efficiency_a085']):.4f}", fontsize=8)
effs = sorted([(float(e['efficiency_a085']), e['sid'], float(e['spacing_m'])) for e in E if e['table'] == 'A']); ref = ('S13', 0.10)
fig, axs = plt.subplots(3, 1, figsize=(9, 10))
for ax, (sid, p) in zip(axs, (ref, (effs[-1][1], effs[-1][2]), (effs[0][1], effs[0][2]))):
    grid_map(ax, sid, p)
fig.suptitle('Absorbed light per copy (a 0.85): reference, best, worst scene (table A; colour per panel)', fontsize=10); fig.text(0.01, 0.005, TAG, fontsize=6); fig.tight_layout(rect=(0, 0.02, 1, 0.96)); fig.savefig(FIG / 'copy_heatmaps_ref_best_worst.png', dpi=130); plt.close(fig)

# 4. paired plots (reference scene): per plant outer vs inner and end vs middle, absorbed per area (b)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, key, a_, b_ in ((axs[0], 'tray_class', 'outer', 'inner'), (axs[1], 'row_class', 'end', 'middle')):
    rows = [r for r in PL if r['sid'] == 'S13']; per = {}
    for r in rows:
        per.setdefault(r['plant'], {}).setdefault(r[key], []).append(float(r['absorbed_per_area_b']))
    for pl, d in per.items():
        if a_ in d and b_ in d:
            ax.plot([0, 1], [np.mean(d[b_]), np.mean(d[a_])], '-o', ms=3, lw=0.9, color='#2a7ab9')
    ax.set_xticks([0, 1]); ax.set_xticklabels([b_, a_]); ax.set_ylabel('absorbed per area (b) (a 0.85)'); ax.grid(alpha=0.25)
    ax.set_title(f'{a_} vs {b_} (one line per plant, reference scene, table A)', fontsize=9)
fig.text(0.01, 0.005, TAG, fontsize=6); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(FIG / 'position_effect_paired.png', dpi=130); plt.close(fig)

# 5. side-view sketch of the three luminaire shapes with emission directions
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, sh in zip(axs, T.SHAPES):
    for f in T.faces(sh, 'trays', 0.10, 0.40)[:2 if sh != 'flat' else 1]:
        ax.plot([f['xa'], f['xb']], [f['za'], f['zb']], '-', lw=5, color=COL[sh]); xm, zm = (f['xa'] + f['xb']) / 2, (f['za'] + f['zb']) / 2
        for ang in np.radians([-50, -25, 0, 25, 50]):
            nx, nz = f['nx'], f['nz']; c, s = np.cos(ang), np.sin(ang); dx, dz = nx * c - nz * s, nx * s + nz * c
            if dx * nx + dz * nz > 0:
                ax.annotate('', xy=(xm + 0.08 * dx, zm + 0.08 * dz), xytext=(xm, zm), arrowprops=dict(arrowstyle='->', color='#555', lw=0.8))
    ax.axhline(0.06, color='#8b5a2b', lw=1); ax.text(-0.15, 0.065, 'substrate 0.06 m', fontsize=6)
    ax.set_xlim(-0.16, 0.16); ax.set_ylim(0.0, 0.52); ax.set_aspect('equal'); ax.set_title({'flat': 'Flat panel', 'V120': 'V 120° (wings 30°)', 'V90': 'V 90° (wings 45°)'}[sh], fontsize=9); ax.tick_params(labelsize=7)
axs[0].set_ylabel('z (m)'); fig.suptitle('Luminaire shapes (side view, x–z): faces and emission directions (cos^m, half-intensity half-angle 60°)', fontsize=10); fig.text(0.01, 0.005, TAG, fontsize=6)
fig.tight_layout(rect=(0, 0.02, 1, 0.94)); fig.savefig(FIG / 'luminaire_shapes_side_view.png', dpi=130); plt.close(fig)
print('figures ok')
