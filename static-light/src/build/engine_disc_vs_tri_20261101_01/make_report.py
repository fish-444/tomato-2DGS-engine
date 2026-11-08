"""Tables and figures from the kernel outputs (no tracing). python make_report.py -> disc_vs_triangle_R3-1b.csv, figures/*.png, TABLES.md, HASHES.json"""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; KO = HERE / 'out_k1'
R = json.load(open(KO / 'results.json', encoding='utf-8')); S = R['summary']; EL = R['engine_lamp']; DEC = R['decomposition']
TAG = ' · '.join(R['labels'])
plt.rcParams.update({'font.family': ['Malgun Gothic', 'DejaVu Sans'], 'axes.unicode_minus': False})
LEAVES = range(1, 9)


def v(key, k):
    return S[key][k]['mean'], S[key][k]['sigma']


def ratio(a, b):
    """ratio of means with propagated seed sigma"""
    (ma, sa), (mb, sb) = a, b
    r = ma / mb
    return r, r * math.sqrt((sa / ma) ** 2 + (sb / mb) ** 2) if ma and mb else float('nan')


rows = []
for model in ('v1', 'tau0.07'):
    Db = f'D_blades|{model}'; Da = f'D_all|{model}'
    for tname in ('T', 'T_diamond'):
        Tk = f'{tname}|{model}'
        r, s = ratio(v(Tk, 'total'), v(Db, 'total'))
        rows.append(dict(level='plant', unit='plant', model=model, T_variant=tname, D_blades=v(Db, 'total')[0], D_all=v(Da, 'total')[0], T=v(Tk, 'total')[0],
                         ratio_T_over_Dblades=r, ratio_sigma=s, D_self=v(Db, 'self')[0], T_self=v(Tk, 'self')[0], D_light_per_area=v(Db, 'mean_light')[0], T_light_per_area=v(Tk, 'mean_light')[0]))
        for th in ('lower', 'middle', 'upper'):
            r, s = ratio(v(Tk, f'{th}_total'), v(Db, f'{th}_total'))
            rows.append(dict(level='third', unit=th, model=model, T_variant=tname, D_blades=v(Db, f'{th}_total')[0], D_all=v(Da, f'{th}_total')[0], T=v(Tk, f'{th}_total')[0],
                             ratio_T_over_Dblades=r, ratio_sigma=s, D_self=v(Db, f'{th}_self')[0], T_self=v(Tk, f'{th}_self')[0], D_area=v(Db, f'{th}_area')[0], T_area=v(Tk, f'{th}_area')[0]))
        dl = [v(Db, f'leaf{L}_light')[0] for L in LEAVES]; tl = [v(Tk, f'leaf{L}_light')[0] for L in LEAVES]
        da = [v(Db, f'leaf{L}_total')[0] for L in LEAVES]; ta = [v(Tk, f'leaf{L}_total')[0] for L in LEAVES]
        rk_d = (-np.array(dl)).argsort().argsort() + 1; rk_t = (-np.array(tl)).argsort().argsort() + 1
        for i, L in enumerate(LEAVES):
            r, s = ratio(v(Tk, f'leaf{L}_total'), v(Db, f'leaf{L}_total')); d = DEC[f'leaf{L}']
            rows.append(dict(level='leaf', unit=f'leaf{L}', model=model, T_variant=tname, D_blades=da[i], T=ta[i], ratio_T_over_Dblades=r, ratio_sigma=s,
                             D_self=v(Db, f'leaf{L}_self')[0], T_self=v(Tk, f'leaf{L}_self')[0], D_light_per_area=dl[i], T_light_per_area=tl[i], rank_D=int(rk_d[i]), rank_T=int(rk_t[i]),
                             D_area=v(Db, f'leaf{L}_area')[0], T_area=v(Tk, f'leaf{L}_area')[0], D_inclination_deg=d['D_inclination_deg'], T_inclination_deg=d[f'{tname}_inclination_deg'],
                             normal_angle_T_D_deg=d[f'{tname}_angle_to_D_deg'], D_share_dev_gt15=d['D_share_area_dev_gt15'], T_share_dev_gt15=d[f'{tname}_share_area_dev_gt15']))
        rows.append(dict(level='rank', unit='spearman_light_per_area', model=model, T_variant=tname, ratio_T_over_Dblades=float(spearmanr(dl, tl).statistic)))
        rows.append(dict(level='rank', unit='spearman_absorbed', model=model, T_variant=tname, ratio_T_over_Dblades=float(spearmanr(da, ta).statistic)))
# d_ex and occluder-set sensitivity on D (blades receivers)
for k in ('v1_dex1.5', 'v1_dex5', 'v1_nostem'):
    r, s = ratio(v(f'D_blades|{k}', 'total'), v('D_blades|v1', 'total'))
    rows.append(dict(level='D_sensitivity', unit=k, model='v1', T_variant='', D_blades=v(f'D_blades|{k}', 'total')[0], ratio_T_over_Dblades=r, ratio_sigma=s, D_self=v(f'D_blades|{k}', 'self')[0]))
# engine lamp closure
gr = EL.get('groimp', {})
for tname in ('T', 'T_diamond'):
    st = EL[f'{tname}_static']; g = gr.get(tname, {})
    row = dict(level='engine_lamp', unit='plant', model='black', T_variant=tname, D_blades=EL['D_blades']['total'], D_all=EL['D_all']['total'], T=st['total']['mean'],
               ratio_T_over_Dblades=st['total']['mean'] / EL['D_blades']['total'], D_self=EL['D_blades']['self'], T_self=st['self']['mean'])
    if g.get('ok'):
        row.update(T_groimp=g['mean']['total']['mean'], T_groimp_sigma_seed=g['mean']['total']['sigma'], T_groimp_sigma_packet=g['packet_sigma_total'],
                   ratio_groimp_over_Tstatic=g['mean']['total']['mean'] / st['total']['mean'], ratio_groimp_over_Dblades=g['mean']['total']['mean'] / EL['D_blades']['total'])
    rows.append(row)
    for L in LEAVES:
        row = dict(level='engine_lamp', unit=f'leaf{L}', model='black', T_variant=tname, D_blades=EL['D_blades'][f'leaf{L}'], T=st[f'leaf{L}']['mean'],
                   ratio_T_over_Dblades=st[f'leaf{L}']['mean'] / EL['D_blades'][f'leaf{L}'])
        if g.get('ok'):
            row.update(T_groimp=g['mean'][f'leaf{L}']['mean'], ratio_groimp_over_Tstatic=g['mean'][f'leaf{L}']['mean'] / st[f'leaf{L}']['mean'])
        rows.append(row)
cols = []
for r in rows:
    cols += [k for k in r if k not in cols]
with open(HERE / 'disc_vs_triangle_R3-1b.csv', 'w', encoding='utf-8', newline='') as fh:
    fh.write(f'# {TAG}; absorbed = relative units x mm2 (a = 1); engine_lamp = umol s-1 under the 1200 point lamp at 0.6 m (black); D_blades = R8_pc blade surfels 1-8\n')
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

# ---------------- figures
FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor='white')
x = np.arange(8); wd = 0.2
ax = axes[0]
ax.bar(x - 1.5 * wd, [v('D_blades|v1', f'leaf{L}_total')[0] for L in LEAVES], wd, yerr=[v('D_blades|v1', f'leaf{L}_total')[1] for L in LEAVES], label='D 디스크', color='#4c72b0')
ax.bar(x - 0.5 * wd, [v('T|v1', f'leaf{L}_total')[0] for L in LEAVES], wd, yerr=[v('T|v1', f'leaf{L}_total')[1] for L in LEAVES], label='T 삼각형 (윤곽)', color='#dd8452')
ax.bar(x + 0.5 * wd, [v('T_diamond|v1', f'leaf{L}_total')[0] for L in LEAVES], wd, label='T 마름모', color='#f2c094')
ax.set_xticks(x); ax.set_xticklabels([f'잎 {L}' for L in LEAVES]); ax.set_ylabel('흡수 (상대 r.u.·mm², a = 1)'); ax.set_title('캐비닛 조명 (실측장 + 바 + 돔), 정적 추적기'); ax.legend(frameon=False, fontsize=8)
ax = axes[1]
ax.bar(x - wd, [EL['D_blades'][f'leaf{L}'] for L in LEAVES], wd, label='D 정적', color='#4c72b0')
ax.bar(x, [EL['T_static'][f'leaf{L}']['mean'] for L in LEAVES], wd, label='T 정적', color='#dd8452')
if gr.get('T', {}).get('ok'):
    ax.bar(x + wd, [gr['T']['mean'][f'leaf{L}']['mean'] for L in LEAVES], wd, label='T GroIMP', color='#55a868')
ax.set_xticks(x); ax.set_xticklabels([f'잎 {L}' for L in LEAVES]); ax.set_ylabel('흡수 (µmol s⁻¹)'); ax.set_title('엔진 램프 (점광원 1200, 0.6 m), 검은 잎'); ax.legend(frameon=False, fontsize=8)
fig.text(0.01, 0.01, TAG, fontsize=6.5); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / 'per_leaf_D_T_groimp.png', dpi=170); plt.close(fig)

z = np.load(KO / 'per_surfel_D.npz'); m = np.isfinite(z['dev_from_leaf_mean_deg']); X = z['xyz']
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor='white')
for ax, (i, j, lab) in zip(axes, ((0, 1, '위에서 (x, y)'), (0, 2, '옆에서 (x, z)'))):
    ax.scatter(X[~m, i] * 1000, X[~m, j] * 1000, s=0.2, c='0.85')
    sc = ax.scatter(X[m, i] * 1000, X[m, j] * 1000, s=0.4, c=np.clip(z['dev_from_leaf_mean_deg'][m], 0, 60), cmap='magma_r', vmin=0, vmax=60)
    ax.set_aspect('equal'); ax.set_title(lab); ax.set_xlabel('mm')
fig.colorbar(sc, ax=axes, shrink=0.8, label='잎 평균 법선에서 벗어난 각 (°), 회색 = 잎 아님')
fig.suptitle('D 서펠: R8_pc 잎 평균 법선에서의 편차 (말림·굽음은 T 가 표현 못 함)'); fig.text(0.01, 0.01, TAG, fontsize=6.5)
fig.savefig(FIG / 'D_surfel_normal_deviation.png', dpi=170); plt.close(fig)

hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*')) if p.is_file() and p.suffix in ('.csv', '.json', '.py', '.obj', '.ply', '.md') and p.name != 'HASHES.json'}
hashes.update({f'out_k1/{p.name}': hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(KO.glob('*')) if p.is_file()})
hashes['engine_hash_static'] = R['engine_hash']
(HERE / 'HASHES.json').write_text(json.dumps(hashes, indent=1), encoding='utf-8')
print('rows', len(rows))
for r in rows:
    if r['level'] in ('plant', 'rank', 'D_sensitivity') or (r['level'] == 'engine_lamp' and r['unit'] == 'plant'):
        print({k: (round(val, 4) if isinstance(val, float) else val) for k, val in r.items()})
