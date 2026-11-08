"""Engine v2 acceptance table (v1 = B0, v2 = frozen C5f1.15), growth curves v1 vs v2, top/side render of the v2 m00 plant.
python -X utf8 make_tables_figs.py  (needs acceptance_raw.json and the downloaded kout summaries)"""
import csv, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import analyze as A

plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
v1, v2 = json.load(open('acceptance_raw.json'))
f1 = lambda x: round(x, 3) if isinstance(x, float) else x
rows = [
    ('1', 'main-stem compound leaves (main controller, A0 + successors)', 'hand 7.33 +- 2.34 (3-9)', v1['main_stem_leaves'], v2['main_stem_leaves'], 'PASS if median in [3, 9]', 'FAIL', 'FAIL',
     f"end rule 10-13 leaves makes v2 exceed the 60-DAG hand counts (addendum: report); A0 only: v1 {v1['main_A0']:.0f}, v2 {v2['main_A0']:.0f}"),
    ('2', 'largest leaflet length (mm)', 'no MTG-building measurement; v1 ref <= 22.5', v1['largest_leaflet_mm'], v2['largest_leaflet_mm'], 'report-only', '-', '-', ''),
    ('3', 'leaflet L/W max', 'no MTG-building measurement; v1 ref 1.0-2.2', v1['lw_max'], v2['lw_max'], 'report-only', '-', '-', 'library outlines'),
    ('4', 'leaflets per main compound', 'hand mean 6.18', v1['leaflets_per_main'], v2['leaflets_per_main'], 'PASS if |diff| <= 0.5', 'FAIL', 'PASS', 'v2 draws from the hand shares (consistency check)'),
    ('5', 'first flowering (Truss1 initiation, engine day)', 'no MTG-building date; literature 35-40 d after sowing', v1['first_truss_day'], v2['first_truss_day'], 'report-only', '-', '-',
     f"first truss node 9 -> 5-9: {v1['first_truss_dag']:.1f} -> {v2['first_truss_dag']:.1f} DAG"),
    ('6', 'accepted gross photosynthesis (gC/d, last 10 d)', 'no MTG-building measurement; v1 ref 0.023-0.045', v1['accepted_gross_last10'], v2['accepted_gross_last10'], 'report-only', '-', '-', 'single plant at the lamp axis'),
    ('7', 'no organ above its max mass', 'guard', v1['max_mass_ratio'], v2['max_mass_ratio'], 'PASS if <= 1 in every verified run', 'PASS', 'PASS', 'max ratio over 16 seeds'),
    ('R1', 'fruits per plant at 59.9 DAG', 'hand 3.5', v1['fruits'], v2['fruits'], 'report-only', '-', '-', ''),
    ('R2', 'side leaves per side axis at 59.9 DAG', 'hand 1.5', v1['side_leaves_per_axis'], v2['side_leaves_per_axis'], 'report-only', '-', '-', 'end counts 5-7 reached before 59.9 DAG'),
    ('R3', 'height (mm)', 'reconstruction 86.2 +- 6.1', v1['height_mm'], v2['height_mm'], 'report-only', '-', '-', ''),
    ('R4', 'trusses per plant', 'hand 1.5', v1['trusses'], v2['trusses'], 'report-only', '-', '-', 'v2: 3 per axis x 3 axes'),
    ('R5', 'main axis stop (DAG)', 'end rule', v1['main_stop_dag'], v2['main_stop_dag'], 'report-only', '-', '-', f"side axes stop at median {v2['side_stop_dag_median']:.1f} DAG"),
    ('R6', 'leaf area (cm2) / dry mass (g)', 'C5 target 29.3-43.6 cm2 leaflet-only', f"{v1['leaf_area_cm2']:.1f} / {v1['dm_g']:.2f}", f"{v2['leaf_area_cm2']:.1f} / {v2['dm_g']:.2f}",
     'report-only', '-', '-', 'base_leaf_area_below_target')]
with open('acceptance_v1_vs_v2.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['check', 'measure', 'target_source', 'v1_median_16seeds', 'v2_median_16seeds', 'rule', 'v1', 'v2', 'note'])
    for r in rows: w.writerow([r[0], r[1], r[2], f1(r[3]), f1(r[4]), *r[5:]])

ks = [f'ev-bc1-k{k}-20261104' for k in (0, 1)] + [f'ev-bc5a-k{k}-20261104' for k in (0, 1, 2)]
R = A.load(ks); days = list(range(72))


def series(arm, key):
    S = [v for k, v in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == arm]; out = []
    for d in days:
        vals = []
        for s in S:
            r = next(x for x in s['daily'] if x['day'] == d)
            vals.append({'leaf': r['leaf_area_mm2'] / 100, 'dm': r['dm_g'], 'h': r['height_mm'], 'tr': sum(r['trusses'].values()), 'fr': r['fruits']}[key])
        out.append((A.q(vals, .5), A.q(vals, .05), A.q(vals, .95)))
    return np.array(out)


fig, axs = plt.subplots(1, 5, figsize=(19, 4))
for ax, (key, lab) in zip(axs, [('leaf', '잎면적 (cm²)'), ('dm', '건물중 (g)'), ('h', '키 (mm)'), ('tr', '화방 수'), ('fr', '열매 수')]):
    for arm, col, nm in (('B0', '#888888', 'v1'), ('C5f1.15', '#2e7d32', 'v2 (동결)')):
        s = series(arm, key); x = np.array(days) - 11.07
        ax.plot(x, s[:, 0], color=col, label=nm); ax.fill_between(x, s[:, 1], s[:, 2], color=col, alpha=.15)
    ax.set_title(lab); ax.set_xlabel('DAG'); ax.set_xlim(0, 60)
axs[0].legend()
fig.suptitle('Engine v1 vs v2 (동결): 16개 시드 중앙값과 5–95 %, 단일 개체·점광원 (relative_light, 검증 아님)', fontsize=11)
fig.tight_layout(); fig.savefig('figures/growth_curves_v1_vs_v2.png', dpi=130)

p = glob.glob('kout/ev-bc5a-*/out/C5f1.15_m00/geometry_day71.npz')[0]; g = np.load(p); tri = g['tri'] * 1000; cls = g['cls']
COL = {'Leaflet': '#66bb6a', 'RachisSegment': '#3e2723', 'Petiole': '#3e2723', 'Internode': '#8d6e63', 'TrussAxis': '#c62828', 'Flower': '#fdd835', 'Fruit': '#e53935'}
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, (i, j, lab) in zip(axs, [(0, 1, '위에서 (x–y)'), (0, 2, '옆에서 (x–z)')]):
    order = np.argsort(tri[:, :, 2].mean(1)) if j == 1 else np.argsort(-tri[:, :, 1].mean(1))
    for k in COL:
        idx = [o for o in order if cls[o] == k]
        if idx:
            ax.add_collection(PolyCollection(tri[idx][:, :, [i, j]], facecolor=COL[k], edgecolor='#2e7d32' if k == 'Leaflet' else COL[k],
                                             linewidths=1.5 if k in ('RachisSegment', 'Petiole') else .2, alpha=.6 if k == 'Leaflet' else 1))
    ax.autoscale(); ax.set_aspect('equal'); ax.set_title(lab); ax.set_xlabel('x (mm)'); ax.set_ylabel(('y' if j == 1 else 'z') + ' (mm)')
fig.suptitle('Model_v2 (동결) 한 개체, 시드 m00, 59.9 DAG', fontsize=11); fig.tight_layout(); fig.savefig('figures/model_v2_m00_top_side.png', dpi=130)
print('ok')
