"""C4A outputs (CONTRACT Amendment 17): L by rank (Step C arm vs reconstruction, pass rule), paired v2.1 vs candidate (C4AC<f>) and vs the B arm,
figure v2.1 | candidate | reconstructed plant (e_04, tallest of the _04 set) at the same scale, seed m00, day 71.
python -X utf8 c4a_final.py C4AC1.3271 C4AB1.3271 -> C4A_length_by_rank.csv, C4A_paired.csv, C4A_final.json, figures/c4a_v2_1_vs_candidate_m00.png"""
import csv, glob, json, statistics as st, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import analyze as A
from analyze_c4a_stepA import recon_L, engine_L

H = Path(__file__).resolve().parent; CAND, BARM = sys.argv[1], sys.argv[2]; BASE = 'V21Bs5.85'
plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
R = A.load(); RL = recon_L()
S = {k: v for k, v in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == CAND}
failed = sorted(k for k in R.get('_failed', {}) if k.startswith(CAND + '_'))
EL = engine_L(S); rows = []; npass = nclass = 0
for r in sorted(RL):
    e = EL.get(r, []); em = A.q(e, .5) if e else None; ratio = em / RL[r]['median'] if e else None
    eligible = RL[r]['n'] >= 5 and len(e) >= 5; ok = eligible and abs(ratio - 1) <= 0.20
    nclass += eligible; npass += ok
    rows.append(dict(rank=r, recon_n=RL[r]['n'], recon_median_mm=RL[r]['median'], engine_n=len(e), engine_median_mm=em, ratio=ratio, eligible=eligible, within_20pct=ok))
rows.append(dict(rank='side', recon_n=0, recon_median_mm=None, engine_n=len(EL.get('side', [])), engine_median_mm=A.q(EL.get('side', []), .5), ratio=None, eligible=False, within_20pct=None))
with open(H / f'C4A_length_by_rank_{CAND}.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
length_pass = nclass > 0 and npass / nclass >= 0.70

KEYS = ['leaf_area_cm2', 'dm_g', 'height_mm', 'absorbed_rel', 'accepted_gross_last10', 'max_leaflet_len_mm']


def organ_dm(S71):
    return S71['daily'][-1].get('dm_by_class', {})


def span(Sx):
    return [x['span_ratio'] for x in Sx.get('c4l_measures', []) if x.get('n_leaflets', 0) >= 5 and x.get('span_ratio') is not None]


paired = []; out = dict(candidate=CAND, b_arm=BARM, failed=failed, verified=len(S), length_classes=nclass, length_pass_classes=npass, length_pass=length_pass)
for ref in (BASE, BARM):
    a, b = A.arm(R, CAND), A.arm(R, ref); seeds = sorted(set(a) & set(b))
    for k in KEYS:
        pc = [(a[s][k] / b[s][k] - 1) * 100 for s in seeds if b[s][k]]
        paired.append(dict(ref=ref, measure=k, ref_median=A.q([b[s][k] for s in seeds], .5), cand_median=A.q([a[s][k] for s in seeds], .5),
                           paired_pct_median=A.q(pc, .5), p10=A.q(pc, .1), p90=A.q(pc, .9), n=len(seeds)))
    lpa = [(a[s]['absorbed_rel'] / a[s]['leaf_area_cm2']) / (b[s]['absorbed_rel'] / b[s]['leaf_area_cm2']) * 100 - 100 for s in seeds if b[s]['leaf_area_cm2'] and a[s]['leaf_area_cm2']]
    paired.append(dict(ref=ref, measure='absorbed_per_leaf_area (self-shading proxy)', ref_median='', cand_median='', paired_pct_median=A.q(lpa, .5), p10=A.q(lpa, .1), p90=A.q(lpa, .9), n=len(seeds)))
    RS = {k: v for k, v in R.items() if not k.startswith('_') and k.rsplit('_m', 1)[0] == ref}
    cls = sorted({c for v in list(S.values()) + list(RS.values()) for c in organ_dm(v)})
    for c in cls:
        ra = A.q([organ_dm(v).get(c, 0) for v in RS.values()], .5); ca = A.q([organ_dm(v).get(c, 0) for v in S.values()], .5)
        paired.append(dict(ref=ref, measure=f'dm_{c}_g', ref_median=ra, cand_median=ca, paired_pct_median=(ca / ra - 1) * 100 if ra else '', p10='', p90='', n=len(seeds)))
    paired.append(dict(ref=ref, measure='span_over_L', ref_median=A.q([x for v in RS.values() for x in span(v)], .5), cand_median=A.q([x for v in S.values() for x in span(v)], .5),
                       paired_pct_median='', p10='', p90='', n=''))
with open(H / f'C4A_paired_{CAND}.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(paired[0])); w.writeheader(); w.writerows(paired)
M = A.arm(R, CAND)
out.update(leaf_area_median=A.q([m['leaf_area_cm2'] for m in M.values()], .5), height_mean=st.mean(m['height_mm'] for m in M.values()),
           leaf_area_pass=29.3 <= A.q([m['leaf_area_cm2'] for m in M.values()], .5) <= 43.6)
json.dump(dict(out, length=rows, paired=paired), open(H / f'C4A_final_{CAND}.json', 'w'), indent=1, default=float)
print(json.dumps(out, default=float))
for r in rows: print(r)

# figure: v2.1 | candidate | reconstruction e_04, top and side, one shared scale
COL = {'Leaflet': '#66bb6a', 'RachisSegment': '#3e2723', 'Petiole': '#3e2723', 'Internode': '#8d6e63', 'TrussAxis': '#c62828', 'Flower': '#fdd835', 'Fruit': '#e53935'}


def eng(pat):
    g = np.load(glob.glob(str(H / 'kout' / pat))[0]); return g['tri'] * 1000, g['cls']


def recon(path):
    V, F, C, cur = [], [], [], 'stem'
    for l in open(path):
        if l.startswith('v '): V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('usemtl'): cur = l.split()[1]
        elif l.startswith('f '): F.append([int(x.split('/')[0]) - 1 for x in l.split()[1:4]]); C.append(cur)
    V = np.array(V); stem = V[[i for f, c in zip(F, C) if c == 'stem' for i in f]]; b = stem[stem[:, 2].argmin()]
    tri = V[np.array(F)] - b
    cls = np.array(['Leaflet' if c.startswith('blade') else ('RachisSegment' if c == 'rachis' else 'Internode') for c in C])
    return tri, cls


panels = [('v2.1 (동결)', *eng('ev-v21c-*/out/V21Bs5.85_m00/geometry_day71.npz')), (f'v2.2 ({CAND})', *eng(f'ev-*/out/{CAND}_m00/geometry_day71.npz')),
          ('재구성 e_04 (_04 세트 최고 키)', *recon(H.parent / 'mtg_build_v2_clicks_20261105_04' / 'static' / 'mtg_e_04_engine.obj'))]
allp = np.concatenate([p[1].reshape(-1, 3) for p in panels]); lim = np.abs(allp[:, :2]).max() * 1.05; zmax = allp[:, 2].max() * 1.05
fig, axs = plt.subplots(2, 3, figsize=(16, 11))
for col, (nm, tri, cls) in enumerate(panels):
    for row, (i, j, lab) in enumerate([(0, 1, '위에서 (x–y)'), (0, 2, '옆에서 (x–z)')]):
        ax = axs[row, col]; order = np.argsort(tri[:, :, 2].mean(1)) if j == 1 else np.argsort(-tri[:, :, 1].mean(1))
        for k in COL:
            idx = [o for o in order if cls[o] == k]
            if idx: ax.add_collection(PolyCollection(tri[idx][:, :, [i, j]], facecolor=COL[k], edgecolor='#2e7d32' if k == 'Leaflet' else COL[k],
                                                     linewidths=1.2 if k in ('RachisSegment', 'Petiole') else .15, alpha=.6 if k == 'Leaflet' else 1))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim) if j == 1 else ax.set_ylim(-5, zmax); ax.set_aspect('equal')
        ax.set_title(f'{nm} — {lab}', fontsize=10); ax.set_xlabel('x (mm)'); ax.set_ylabel(('y' if j == 1 else 'z') + ' (mm)')
fig.suptitle('시드 m00, 59.9 DAG, 같은 축척 (엔진: 단일 개체·점광원; 재구성은 높이가 가장 큰 _04 개체, 키가 비슷한 개체 없음) — 검증 아님', fontsize=11)
fig.tight_layout(); fig.savefig(H / 'figures' / f'c4a_v2_1_vs_{CAND}_m00.png', dpi=110)
