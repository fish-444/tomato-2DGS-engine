"""Engine v2.2 reference outputs (CONTRACT Amendment 19, CLI ANSWER 15:45 §3.1): both frozen arms (A = V22A_s4.6468, R = V22R_s5.0843) vs v2.1
(V21Bs5.85) and v1 (B0): acceptance table, growth curves, top / side views (v2.1 | A | R | reconstruction e_04, one scale, seed m00), MTG figure A | R.
python -X utf8 make_tables_figs_v2_2.py"""
import csv, glob, json, statistics as st, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image
import analyze as A
from make_tables_figs_v2_1 import acc  # same aggregation as the v2.1 table (it asserts the stored v2 row on import)

plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
R = A.load(); days = list(range(72)); ARMS = [('B0', 'v1', '#9e9e9e'), ('V21Bs5.85', 'v2.1', '#90caf9'), ('V22A_s4.6468', 'v2.2 A (SLA 수용)', '#2e7d32'), ('V22R_s5.0843', 'v2.2 R (SLA 비수용)', '#ef6c00')]
rows = {a: acc(a, lab) for a, lab, _ in ARMS}


def span(arm):
    return A.q([x['span_ratio'] for k, s in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == arm for x in s.get('c4l_measures', [])
                if x.get('n_leaflets', 0) >= 5 and x.get('span_ratio') is not None], .5)


f1 = lambda x: round(x, 3) if isinstance(x, float) else x
TAB = [('main-stem compound leaves', 'main_stem_leaves', 'hand 7.33 (3-9)'), ('leaflets per main compound', 'leaflets_per_main', 'hand 6.18'),
       ('largest leaflet (mm)', 'largest_leaflet_mm', '-'), ('first truss (DAG)', 'first_truss_dag', 'lit. 35-40 d after sowing'),
       ('height median (mm)', 'height_mm', 'user 130'), ('height mean (mm)', 'height_mean_mm', '-'), ('leaf area (cm2)', 'leaf_area_cm2', 'C5 29.3-43.6'),
       ('dry mass (g)', 'dm_g', '-'), ('trusses', 'trusses', 'hand 1.5'), ('fruits', 'fruits', 'hand 3.5'), ('accepted gross last 10 d (gC/d)', 'accepted_gross_last10', '-'),
       ('max organ mass ratio (guard <= 1)', 'max_mass_ratio', '<= 1')]
with open('acceptance_v2_1_v2_2.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['measure', 'reference'] + [lab for _, lab, _ in ARMS])
    for nm, k, ref in TAB: w.writerow([nm, ref] + [f1(rows[a][k]) for a, _, _ in ARMS])
    w.writerow(['span / rachis L (median)', 'recon 0.85'] + [f1(span(a)) for a, _, _ in ARMS])
    w.writerow(['SLA mean (m2/g)', 'Fan 2013 0.0044-0.0115; Ke 2023 Micro-Tom 0.0313 (36 DAS)', 0.025, 0.02875, 0.03318, 0.02875])
    w.writerow(['n seeds', '-'] + [rows[a]['n'] for a, _, _ in ARMS])


def series(arm, key):
    S = [v for k, v in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == arm]; out = []
    for d in days:
        vals = []
        for s in S:
            r = next(x for x in s['daily'] if x['day'] == d)
            vals.append({'leaf': r['leaf_area_mm2'] / 100, 'dm': r['dm_g'], 'h': r['height_mm'], 'tr': sum(r['trusses'].values()), 'fr': r['fruits']}[key])
        out.append((A.q(vals, .5), A.q(vals, .05), A.q(vals, .95)))
    return np.array(out)


fig, axs = plt.subplots(1, 5, figsize=(20, 4.2))
for ax, (key, lab) in zip(axs, [('leaf', '잎면적 (cm²)'), ('dm', '건물중 (g)'), ('h', '키 (mm)'), ('tr', '화방 수'), ('fr', '열매 수')]):
    for arm, nm, col in ARMS:
        s = series(arm, key); x = np.array(days) - 11.07
        ax.plot(x, s[:, 0], color=col, label=nm, lw=1.6); ax.fill_between(x, s[:, 1], s[:, 2], color=col, alpha=.12)
    ax.set_title(lab); ax.set_xlabel('DAG'); ax.set_xlim(0, 60)
axs[0].axhspan(29.3, 43.6, color='#2e7d32', alpha=.06, lw=0); axs[2].axhline(130, color='#555', ls=':', lw=.8); axs[0].legend(fontsize=8)
fig.suptitle('Engine v1 / v2.1 / v2.2 A·R: 16개 시드 중앙값과 5–95 %, 단일 개체·점광원 (relative_light, 검증 아님)', fontsize=11)
fig.tight_layout(); fig.savefig('figures/growth_curves_v1_v2_1_v2_2.png', dpi=120)

COL = {'Leaflet': '#66bb6a', 'RachisSegment': '#3e2723', 'Petiole': '#3e2723', 'Internode': '#8d6e63', 'TrussAxis': '#c62828', 'Flower': '#fdd835', 'Fruit': '#e53935'}


def eng(arm):
    g = np.load(glob.glob(f'kout/*/out/{arm}_m00/geometry_day71.npz')[-1]); return g['tri'] * 1000, g['cls']


def recon(path):
    V, F, C, cur = [], [], [], 'stem'
    for l in open(path):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('usemtl'): cur = l.split()[1]
        elif l.startswith('f '): F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:4]]); C.append(cur)
    V = np.array(V); stem = V[[i for f, c in zip(F, C) if c == 'stem' for i in f]]; b = stem[stem[:, 2].argmin()]
    return V[np.array(F)] - b, np.array(['Leaflet' if c.startswith('blade') else ('RachisSegment' if c == 'rachis' else 'Internode') for c in C])


panels = [('v2.1', *eng('V21Bs5.85')), ('v2.2 A (SLA 수용)', *eng('V22A_s4.6468')), ('v2.2 R (SLA 비수용)', *eng('V22R_s5.0843')),
          ('재구성 e_04 (_04 최고 키)', *recon('../mtg_build_v2_clicks_20261105_04/static/mtg_e_04_engine.obj'))]
allp = np.concatenate([p[1].reshape(-1, 3) for p in panels]); lim = np.abs(allp[:, :2]).max() * 1.05; zmax = allp[:, 2].max() * 1.05
fig, axs = plt.subplots(2, 4, figsize=(20, 10.5))
for col, (nm, tri, cls) in enumerate(panels):
    for row, (i, j, lab) in enumerate([(0, 1, '위에서 (x–y)'), (0, 2, '옆에서 (x–z)')]):
        ax = axs[row, col]; order = np.argsort(tri[:, :, 2].mean(1)) if j == 1 else np.argsort(-tri[:, :, 1].mean(1))
        for k in COL:
            idx = [o for o in order if cls[o] == k]
            if idx: ax.add_collection(PolyCollection(tri[idx][:, :, [i, j]], facecolor=COL[k], edgecolor='#2e7d32' if k == 'Leaflet' else COL[k],
                                                     linewidths=1.2 if k in ('RachisSegment', 'Petiole') else .15, alpha=.6 if k == 'Leaflet' else 1))
        ax.set_xlim(-lim, lim); (ax.set_ylim(-lim, lim) if j == 1 else ax.set_ylim(-5, zmax)); ax.set_aspect('equal')
        ax.set_title(f'{nm} — {lab}', fontsize=10); ax.set_xlabel('x (mm)'); ax.set_ylabel(('y' if j == 1 else 'z') + ' (mm)')
fig.suptitle('시드 m00, 59.9 DAG, 같은 축척 (엔진: 단일 개체·점광원; 재구성 e_04는 키 약 95 mm — 키가 비슷한 재구성 개체 없음) — 검증 아님', fontsize=11)
fig.tight_layout(); fig.savefig('figures/v2_2_reference_top_side_m00.png', dpi=105)

info = {'V22A_s4.6468': ['SLA ×1.327 (0.0332 m²/g) · 마디 사이 ×4.6468 · 복엽 길이 순위별 맞춤(엽축 ×1.5–4.0)'],
        'V22R_s5.0843': ['SLA ×1.15 (0.0288 m²/g) · 마디 사이 ×5.0843 · 복엽 길이 순위별 맞춤(엽축 ×1.6–4.7)']}
common = ['주경: 첫 화방 마디 5–9, 잎 10–13장 · 꽃 4 · 측축 2–4개 · 발산각 90–100° · 소엽 크기 동일·동시 생장']
for arm, ttl in (('V22A_s4.6468', '동결 v2.2 A (SLA 수용) — 시드 m00, 59.9 DAG'), ('V22R_s5.0843', '동결 v2.2 R (SLA 비수용) — 시드 m00, 59.9 DAG')):
    subprocess.run([sys.executable, '-X', 'utf8', 'make_mtg_figure.py', arm, '0', ttl, f'figures/_mtg_{arm}.png'] + common + info[arm], check=True, capture_output=True)
ims = [Image.open(f'figures/_mtg_{a}.png') for a in ('V22A_s4.6468', 'V22R_s5.0843')]
W = sum(i.width for i in ims); Hh = max(i.height for i in ims); canvas = Image.new('RGB', (W, Hh), 'white'); x = 0
for i in ims: canvas.paste(i, (x, 0)); x += i.width
canvas.save('figures/engine_mtg_v2_2_A_R_m00.png')
print(json.dumps({lab: {k: f1(rows[a][k]) for k in ('leaf_area_cm2', 'height_mm', 'height_mean_mm', 'dm_g', 'fruits', 'largest_leaflet_mm')} | {'span': f1(span(a))} for a, lab, _ in ARMS}, ensure_ascii=False, indent=0))
