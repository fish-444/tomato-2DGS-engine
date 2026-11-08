"""Engine v2.1 deliverables: acceptance table v1 / v2 / v2.1 (frozen confirm arm V21Bs5.85), growth curves v1 vs v2 vs v2.1, top/side render of the v2.1 m00 plant,
per-block before/after table for the blocks added after v2 (C4W, C4T, C4L, C4B, height scale).
python -X utf8 make_tables_figs_v2_1.py   (v1 = B0, v2 = C5f1.15; values are seed medians over the verified runs)"""
import csv, glob, json, statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import analyze as A

plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
R = A.load(); days = list(range(72))


def acc(arm_name, label):
    M = A.arm(R, arm_name); med = lambda k: A.q([m[k] for m in M.values()], .5)
    return dict(model=label, n=len(M), main_stem_leaves=med('main_compounds'), main_A0=med('main_A0'), largest_leaflet_mm=med('max_leaflet_len_mm'), lw_max=med('lw_max'),
                leaflets_per_main=med('leaflets_per_main'), first_truss_day=med('first_truss_day'), first_truss_dag=med('first_truss_dag'),
                accepted_gross_last10=med('accepted_gross_last10'), max_mass_ratio=max(m['max_mass_ratio'] for m in M.values()), fruits=med('fruits'),
                side_leaves_per_axis=med('side_leaves_per_axis'), side_axes=med('side_axes'), height_mm=med('height_mm'),
                height_mean_mm=st.mean(m['height_mm'] for m in M.values()), trusses=med('trusses'), leaf_area_cm2=med('leaf_area_cm2'), dm_g=med('dm_g'),
                main_stop_dag=med('main_stop_dag'), side_stop_dag_median=A.q([s for m in M.values() for s in m['side_stop_dag']], .5))


v1_s, v2_s = json.load(open('acceptance_raw.json'))
v2_chk = acc('C5f1.15', 'v2'); bad = {k: (v2_s[k], v2_chk[k]) for k in v2_s if k != 'model' and abs(v2_s[k] - v2_chk[k]) > 1e-9}
assert not bad, ('aggregation does not reproduce the stored v2 row', bad)
v21 = acc('V21Bs5.85', 'v2.1'); json.dump([v1_s, v2_s, v21], open('acceptance_raw_v2_1.json', 'w'), indent=1)
v1, v2 = v1_s, v2_s
f1 = lambda x: round(x, 3) if isinstance(x, float) else x
pf = lambda ok: 'PASS' if ok else 'FAIL'
rows = [
    ('1', 'main-stem compound leaves', 'hand 7.33 +- 2.34 (3-9)', 'main_stem_leaves', 'PASS if median in [3, 9]', lambda m: pf(3 <= m['main_stem_leaves'] <= 9), 'template v2.4 end rule 10-13 leaves (addendum: report)'),
    ('2', 'largest leaflet length (mm)', 'v1 ref <= 22.5', 'largest_leaflet_mm', 'report-only', None, ''),
    ('3', 'leaflet L/W max', 'v1 ref 1.0-2.2', 'lw_max', 'report-only', None, 'library outlines'),
    ('4', 'leaflets per main compound', 'hand mean 6.18', 'leaflets_per_main', 'PASS if |diff| <= 0.5', lambda m: pf(abs(m['leaflets_per_main'] - 6.18) <= .5), ''),
    ('5', 'first truss initiation (DAG)', 'literature 35-40 d after sowing', 'first_truss_dag', 'report-only', None, ''),
    ('6', 'accepted gross (gC/d, last 10 d)', 'v1 ref 0.023-0.045', 'accepted_gross_last10', 'report-only', None, 'single plant, point lamp 0.60 m'),
    ('7', 'no organ above its max mass (max ratio)', 'guard', 'max_mass_ratio', 'PASS if <= 1', lambda m: pf(m['max_mass_ratio'] <= 1), 'max over seeds'),
    ('R1', 'fruits per plant', 'hand 3.5', 'fruits', 'report-only', None, 'v2.1: 4 flowers per truss (C4W)'),
    ('R2', 'side leaves per side axis', 'hand 1.5', 'side_leaves_per_axis', 'report-only', None, ''),
    ('R3', 'height (mm), median', 'reconstruction 86.2 +- 6.1; user target v2.1 mean 130', 'height_mm', 'report-only', None, 'v2.1 seed mean %.1f mm' % v21['height_mean_mm']),
    ('R4', 'trusses per plant', 'hand 1.5', 'trusses', 'report-only', None, ''),
    ('R5', 'main axis stop (DAG)', 'end rule', 'main_stop_dag', 'report-only', None, 'side stop median v1 %s / v2 %s / v2.1 %s' % tuple('-' if m['side_stop_dag_median'] is None else '%.1f' % m['side_stop_dag_median'] for m in (v1, v2, v21))),
    ('R6', 'leaf area (cm2)', 'C5 target 29.3-43.6 leaflet-only', 'leaf_area_cm2', 'report-only', None, 'base_leaf_area_below_target'),
    ('R7', 'dry mass (g)', '-', 'dm_g', 'report-only', None, '')]
with open('acceptance_v1_v2_v2_1.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['check', 'measure', 'target_source', 'v1_median', 'v2_median', 'v2_1_median', 'rule', 'v1', 'v2', 'v2_1', 'note'])
    for c, meas, src, k, rule, fn, note in rows:
        w.writerow([c, meas, src, f1(v1[k]), f1(v2[k]), f1(v21[k]), rule, *((fn(m) for m in (v1, v2, v21)) if fn else ('-', '-', '-')), note])

# per-block before/after for blocks added after v2 (paired seed medians, day 71)
CHAIN = [('C4W', 'C5f1.15', 'C4W'), ('C4T (v2.4 counts, 2-4 side axes, divergence 90-100)', 'C4W', 'C4T2'), ('C4L leaflet arrangement', 'C4T2', 'C4T2L'),
         ('C4B leaflet curvature (1 deg floor)', 'C4T2L', 'C4T2LB1'), ('height scale x5.85 (v2.1)', 'C4T2LB1', 'V21Bs5.85')]
KEYS = ['leaf_area_cm2', 'dm_g', 'height_mm', 'main_compounds', 'side_compounds', 'side_axes', 'trusses', 'fruits', 'max_leaflet_len_mm', 'accepted_gross_last10']
with open('block_before_after_v2_1.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['block', 'before_arm', 'after_arm', 'measure', 'before_median', 'after_median', 'paired_pct_median', 'n_seeds'])
    for blk, b, a in CHAIN:
        B, Aa = A.arm(R, b), A.arm(R, a); seeds = sorted(set(B) & set(Aa))
        if not seeds: w.writerow([blk, b, a, 'missing', '', '', '', 0]); continue
        for k in KEYS:
            bb = [B[s][k] for s in seeds]; aa = [Aa[s][k] for s in seeds]; pc = [(x / y - 1) * 100 for x, y in zip(aa, bb) if y]
            w.writerow([blk, b, a, k, f1(A.q(bb, .5)), f1(A.q(aa, .5)), f1(A.q(pc, .5)) if pc else '', len(seeds)])


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
    for arm, col, nm in (('B0', '#888888', 'v1'), ('C5f1.15', '#90caf9', 'v2'), ('V21Bs5.85', '#2e7d32', 'v2.1 (동결)')):
        s = series(arm, key); x = np.array(days) - 11.07
        ax.plot(x, s[:, 0], color=col, label=nm); ax.fill_between(x, s[:, 1], s[:, 2], color=col, alpha=.15)
    ax.set_title(lab); ax.set_xlabel('DAG'); ax.set_xlim(0, 60)
axs[0].legend()
fig.suptitle('Engine v1 / v2 / v2.1: 16개 시드 중앙값과 5–95 %, 단일 개체·점광원 (relative_light, 검증 아님)', fontsize=11)
fig.tight_layout(); fig.savefig('figures/growth_curves_v1_v2_v2_1.png', dpi=130)

p = glob.glob('kout/ev-v21c-*/out/V21Bs5.85_m00/geometry_day71.npz')[0]; g = np.load(p); tri = g['tri'] * 1000; cls = g['cls']
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
fig.suptitle('Model_v2.1 (동결) 한 개체, 시드 m00, 59.9 DAG', fontsize=11); fig.tight_layout(); fig.savefig('figures/model_v2_1_m00_top_side.png', dpi=130)
print(json.dumps(v21, indent=0))
