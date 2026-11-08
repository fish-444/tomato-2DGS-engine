"""Schematic MTG of one engine plant (real engine output of one seed at day 71 = 59.9 DAG): main axis + side axes, compound leaves with leaflet count,
trusses at their node (leaves below), compound line length proportional to the measured chain length (petiole + rachis).
python -X utf8 make_mtg_figure.py <arm> <seed m> <title> <out png> [info line ...]"""
import glob, json, sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
arm, m, title, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]; info = sys.argv[5:]
S = json.load(open(glob.glob(f'kout/*/out/{arm}_m{m:02d}/summary.json')[-1])); DAG0 = 11.07
dr = {(d['id'], d['key']): d['value'] for d in S['draws']}
chain = {c['compound']: c['chain_mm'] for c in S.get('c4c_measures', [])}
comp = defaultdict(list)
for c in S['compounds']: comp[c['ctl']].append(c)
for k in comp: comp[k].sort(key=lambda c: c['rank'])
tr = defaultdict(list)
for t in S['truss_init']:
    q = t['organ'].split('/'); tr[q[1] if q[1].startswith('S') else 'M'].append(t['day'] - DAG0)
att = {a['organ'].split('/')[1]: (int(a['parent'].split('/N')[1]), a['day'] - DAG0) for a in S['side_axes']}
N, T = int(dr[('p1', 'MAIN_TRUSS_NODE')]), int(dr[('p1', 'MAIN_TOTAL_LEAVES')])
STEP = 1.0; SC = 0.045   # node spacing (schematic), mm -> plot units for compound length
fig, ax = plt.subplots(figsize=(15, 11))


def axis(x0, y0, ctl, label, nodes_before):
    cs = comp[ctl]; n = len(cs); top = y0 + (n + 1.2) * STEP
    ax.plot([x0, x0], [y0, top], color='#5d4037', lw=4 if ctl == 'M' else 3, solid_capstyle='round', zorder=2)
    ax.text(x0, top + 0.35, label, ha='center', fontsize=11, fontweight='bold', color='#5d4037')
    for i, c in enumerate(cs):
        y = y0 + (i + 1) * STEP; side = -1 if i % 2 == 0 else 1; L = chain.get(c['id'], 10.0) * SC
        ax.plot([x0, x0 + side * L], [y, y + 0.35], color='#2e7d32', lw=2.6, zorder=3)
        ax.text(x0 + side * (L + 0.12), y + 0.35, f"{c['leaflets']}", color='#2e7d32', fontsize=10, fontweight='bold', ha='left' if side > 0 else 'right', va='center')
        ax.text(x0 + side * L * 0.5, y + 0.05, f"{chain.get(c['id'], float('nan')):.0f}", color='#7b8b7b', fontsize=7, ha='center', va='top')
    grp = defaultdict(list)
    for nb, d in zip(nodes_before, sorted(tr[ctl])): grp[nb].append(d)
    for nb, ds in grp.items():
        y = y0 + (nb + 0.5) * STEP
        for k in range(len(ds)): ax.plot(x0, y + 0.22 * k, marker='*', ms=14, color='#c62828', zorder=5)
        lab = f"화방 {len(ds)}개 연속 · 마디 {nb} · " if len(ds) > 1 else f'화방 마디 {nb} · '
        ax.annotate(lab + ', '.join(f'{d:.0f}' for d in ds) + ' DAG', (x0, y), (x0 + 1.15, y), color='#c62828', fontsize=9, va='center',
                    arrowprops=dict(arrowstyle='-', color='#c62828', lw=.6, ls=':'))
    return top


axis(0, 0, 'M', '주경', [N, T, T])
sides = sorted(att); xs = [-7.0, 9.0, -12.5, 14.5]
for j, sname in enumerate(sides):
    r, d = att[sname]; S_tot = int(dr[(f'p1/{sname}', 'SIDE_TOTAL_LEAVES')]); x = xs[j]; y0 = r * STEP
    ax.plot([0, x], [r * STEP, y0], color='#5d4037', lw=2.5, zorder=1)
    axis(x, y0, sname, f'측축 {sname}\n(주경 마디 {r}, {d:.0f} DAG)', [2, S_tot, S_tot])
d71 = S['daily'][-1]
ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
ax.text(0.5, 1.0, '초록 숫자 = 복엽당 소엽 수, 회색 숫자 = 복엽 길이 mm(엽병+엽축, 선 길이도 비례), ★ = 화방(마디 = 그 화방 아래 잎 수)', transform=ax.transAxes, ha='center', fontsize=9.5, color='#444')
box = [f'시드 m{m:02d}: 주경 첫 화방 마디 {N}, 주경 잎 {T}장, 측축 {len(sides)}개'] + info + [f"59.9 DAG: 잎면적 {d71['leaf_area_mm2'] / 100:.1f} cm², 키 {d71['height_mm']:.0f} mm, 건물중 {d71['dm_g']:.2f} g, 열매 {d71['fruits']}"]
ax.text(0.5, 0.01, '\n'.join(box), transform=ax.transAxes, ha='center', va='bottom', fontsize=9.5, bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#aaa'))
ax.text(0.5, -0.03, 'engine_v2 · calibrated_on_mtg_building_data · conditional_on_unvalidated_scale · 검증 아님 · 측축 위치·마디 간격은 도식, 잎·화방 순서·개수·복엽 길이는 실제 엔진 출력',
        transform=ax.transAxes, ha='center', fontsize=8, color='#777')
ax.set_aspect('equal'); ax.axis('off'); ax.autoscale(); ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo - 2.6, yhi + 0.8)
fig.tight_layout(); fig.savefig(out, dpi=120); print(out)
