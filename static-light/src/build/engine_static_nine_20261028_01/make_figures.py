"""Figures from the kernel outputs (kaggle_out/<id>/per_surfel_light.npz + plant_light.json). Plots only; no tracing or aggregation.
One individual is loaded at a time (a few columns). python make_figures.py -> figures/*.png"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
HERE = Path(__file__).resolve().parent; OUT = HERE / 'kaggle_out'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
IDS = ['R3-1b', 't2_02', 'n_04', 'n_01', 't2_05', 'n_06', 'n_05', 't2_04', 'n_02', 'n_03']
RIM = json.load(open(HERE / 'kaggle_data/rim.json', encoding='utf-8'))
FOOT = 'relative_light · conditional_on_unvalidated_scale · light model v1 d_ex=2.4 mm · 조명 바 위치·광학 가정 · biological_validation = not_established'
plt.rcParams.update({'font.family': ['Malgun Gothic', 'DejaVu Sans'], 'axes.unicode_minus': False})
norm = Normalize(0, 1)


def status(pid):
    if pid == 'R3-1b':
        return '재현 시험, 림 OK'
    return '림 OK' if RIM[pid]['rim_visual'] == 'pass' else '림 실패'


def load(pid):
    z = np.load(OUT / pid / 'per_surfel_light.npz')
    return z['x_mm'], z['y_mm'], z['h_above_base_mm'], np.minimum(z['relative_light'], 1.0)


def grid(kind, fname):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.6), facecolor='white')
    # common extent so panels are comparable in size
    ext = 105.0
    for ax, pid in zip(axes.ravel(), IDS):
        x, y, h, v = load(pid)
        o = np.argsort(h)
        cx, cy = np.median(x), np.median(y)
        if kind == 'top':
            ax.scatter(x[o] - cx, y[o] - cy, c=v[o], cmap='viridis', norm=norm, s=0.25, linewidths=0, rasterized=True)
            ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
        else:                       # 45 deg from the -y side: screen-up = h cos45 + (y-cy) sin45
            yy = (y - cy) * np.sin(np.pi / 4) + h * np.cos(np.pi / 4)
            ax.scatter(x[o] - cx, yy[o], c=v[o], cmap='viridis', norm=norm, s=0.25, linewidths=0, rasterized=True)
            ax.set_xlim(-ext, ext); ax.set_ylim(-60, 2 * ext - 60)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.plot([-ext + 8, -ext + 28], [-ext + 8 if kind == 'top' else -52] * 2, color='k', lw=2)
        ax.text(-ext + 8, (-ext + 12) if kind == 'top' else -48, '20 mm', fontsize=7)
        ax.set_title(f"{pid}  ({status(pid)})", fontsize=10, color='#b00020' if '실패' in status(pid) else 'black')
        for sp in ax.spines.values():
            sp.set_color('#999')
        del x, y, h, v
    fig.suptitle('식물별 상대 광(relative_light) — ' + ('위에서 본 지도 (가장 위 서펠이 위에)' if kind == 'top' else '앞 45° 에서 본 지도'), fontsize=13, y=0.995)
    cax = fig.add_axes([0.35, 0.065, 0.3, 0.02])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax, orientation='horizontal').set_label('relative_light (0–1, 1 초과는 1로 표시)', fontsize=9)
    fig.text(0.01, 0.01, FOOT, fontsize=7); fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.04, hspace=0.18)
    fig.savefig(FIG / fname, dpi=170, facecolor='white'); plt.close(fig)


def summary():
    P = {pid: json.load(open(OUT / pid / 'plant_light.json', encoding='utf-8')) for pid in IDS}
    fig = plt.figure(figsize=(16, 5.8), facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.28, left=0.05, right=0.985, top=0.88, bottom=0.2)
    cols = [cm.viridis(0.1) if 'R3-1b' == p else (cm.viridis(0.55) if RIM[p]['rim_visual'] == 'pass' else cm.viridis(0.85)) for p in IDS]
    for k, (key, ttl) in enumerate((('relative_light_per_area', '식물 평균 상대 광'), ('self_shading', '자기 그늘 비율'))):
        ax = fig.add_subplot(gs[0, k]); vals = [P[p]['plant_level']['2.4'][key]['mean'] for p in IDS]
        lo = [min(P[p]['plant_level'][d][key]['mean'] for d in ('1.5', '2', '3', '5')) for p in IDS]; hi = [max(P[p]['plant_level'][d][key]['mean'] for d in ('1.5', '2', '3', '5')) for p in IDS]
        ax.bar(range(len(IDS)), vals, color=cols, edgecolor='k', linewidth=0.5)
        ax.errorbar(range(len(IDS)), vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt='none', ecolor='k', capsize=2, lw=0.8)
        ax.set_xticks(range(len(IDS))); ax.set_xticklabels(IDS, rotation=45, ha='right', fontsize=8); ax.set_title(ttl + ' (d 2.4 mm, 오차 막대 = 1.5–5 mm 범위)', fontsize=10)
        ax.set_ylim(0, 0.8 if k else 0.6); ax.grid(axis='y', alpha=0.3)
    ax = fig.add_subplot(gs[0, 2])
    for pid in IDS:
        prof = P[pid]['profile_5mm_height_above_base']
        ax.plot([b['mean_relative_light'] for b in prof], [b['lo_mm'] + 2.5 for b in prof], lw=1.4, label=pid, ls='--' if pid == 'R3-1b' else '-',
                color=plt.cm.tab10(IDS.index(pid)))
    ax.set_xlabel('높이 5 mm 층의 평균 상대 광 (면적 (b) 가중)'); ax.set_ylabel('식물 맨 아래 서펠 위 높이 mm'); ax.set_title('높이에 따른 상대 광', fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False); ax.grid(alpha=0.3)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cm.viridis(c), ec='k') for c in (0.1, 0.55, 0.85)], labels=['R3-1b (재현 시험)', '림 맞춤 OK', '림 맞춤 실패 (절대 높이 신뢰 불가)'],
               loc='lower left', bbox_to_anchor=(0.05, 0.005), ncol=3, fontsize=8, frameon=False)
    fig.suptitle('열 개체 요약 — 순위·우열 판단이 아니라 묘사 (생물학 검증 없음)', fontsize=13)
    fig.text(0.01, 0.008, '', fontsize=7); fig.text(0.46, 0.012, FOOT, fontsize=6.5)
    fig.savefig(FIG / 'nine_summary.png', dpi=170, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    grid('top', 'nine_top_view.png'); grid('front', 'nine_front45.png'); summary()
    print(sorted(p.name for p in FIG.glob('*.png')))
