"""Figures (CONTRACT §4 + Amendment 1): pilot 3D light map v1_tau raw vs v1tau_smoothn_r2 (same colour scale, front 45 deg), 14-plant front-45 grid of the adopted version, and the
three-version summary (v1 / v1_tau / v1tau_smoothn_r2). Plots only. python -X utf8 make_figures.py"""
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

HERE = Path(__file__).resolve().parent; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05', 'e_01', 'e_04', 'e_05', 'e_07']
FOOT = 'relative_light; conditional_on_unvalidated_scale; assumed_leaf_optics (tau 0.07, a 0.85); v1tau_smoothn_r2; d_ex 2.4 mm; not validated'
plt.rcParams.update({'font.family': ['Malgun Gothic', 'DejaVu Sans'], 'axes.unicode_minus': False})


def find(prefix, pid, name):
    for f in glob.glob(str(HERE / f'out_{prefix}*' / pid / name)):
        return f
    return None


def front(ax, X, h, v, norm, title):
    cx, cy = np.median(X[:, 0]), np.median(X[:, 1]); yy = (X[:, 1] - cy) * np.sin(np.pi / 4) + h * np.cos(np.pi / 4); o = np.argsort(X[:, 1])[::-1]
    ax.scatter(X[o, 0] - cx, yy[o], c=v[o], cmap='viridis', norm=norm, s=0.3, linewidths=0, rasterized=True); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-105, 105); ax.set_ylim(-60, 150); ax.set_title(title, fontsize=9)
    for sp in ax.spines.values():
        sp.set_color('#999')


norm = Normalize(0, 1)
# pilot: raw vs smoothed, v1_tau a = 0.85 (light per area = U*K normalised by the open-field value at the plant top is not available; K is the unit-horizontal kernel)
z5, z8 = find('k5', 'R3-1b', 'per_surfel.npz'), find('k8', 'R3-1b', 'per_surfel.npz')
if z5 and z8:
    a = np.load(z5); b = np.load(z8); X = b['xyz_mm']; h = b['h_above_base_mm']
    va, vb = 0.85 * a['K_tau'], 0.85 * b['K_tau']; vmax = float(np.percentile(np.r_[va, vb], 99.5)); n2 = Normalize(0, vmax)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2)); front(ax[0], X, h, va, n2, 'R3-1b  v1_tau (raw normals)'); front(ax[1], X, h, vb, n2, 'R3-1b  v1tau_smoothn_r2')
    fig.colorbar(cm.ScalarMappable(norm=n2, cmap='viridis'), ax=ax, shrink=0.6, label='relative light kernel (a 0.85), same scale')
    fig.suptitle('Pilot plant: absorbed-light kernel before / after normal smoothing (front 45°)', fontsize=11); fig.text(0.01, 0.01, FOOT, fontsize=6); fig.savefig(FIG / 'pilot_light_map_v1tau_raw_vs_smooth.png', dpi=140); plt.close(fig)
    print('pilot map ok')
# 14-plant grid of the adopted version
fig, axs = plt.subplots(3, 5, figsize=(17, 11)); n = 0
for ax, pid in zip(axs.ravel(), PLANTS):
    f = find('k8', pid, 'per_surfel.npz')
    if f is None:
        ax.axis('off'); continue
    z = np.load(f); front(ax, z['xyz_mm'], z['h_above_base_mm'], np.minimum(0.85 * z['K_tau'], 1.0), norm, pid); n += 1
axs.ravel()[-1].axis('off'); fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axs, shrink=0.4, orientation='horizontal', pad=0.03, label='relative light kernel (a 0.85)')
fig.suptitle('Relative light, v1tau_smoothn_r2 (front 45°)', fontsize=13); fig.text(0.01, 0.005, FOOT, fontsize=7); fig.savefig(FIG / 'plants14_front45_v1tau_smoothn_r2.png', dpi=130); plt.close(fig); print('grid', n)
# three-version summary
rows = []
for pid in PLANTS:
    f5, f8 = find('k5', pid, 'plant.json'), find('k8', pid, 'plant.json')
    if f5 and f8:
        d5 = json.load(open(f5, encoding='utf-8')); d8 = json.load(open(f8, encoding='utf-8')); g = lambda d, m, k: d['dex']['2.4'][m][k]
        rows.append((pid, g(d5, 'v1', 'mean_light'), 0.85 * g(d5, 'tau', 'mean_light'), 0.85 * g(d8, 'tau', 'mean_light'), g(d5, 'v1', 'self_shading'), g(d5, 'tau', 'self_shading'), g(d8, 'tau', 'self_shading')))
if rows:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6)); x = np.arange(len(rows)); w = 0.27
    for k, (lab, c) in enumerate((('v1', '#6b6b6b'), ('v1_tau', '#2a7ab9'), ('v1tau_smoothn_r2', '#c23b3b'))):
        ax[0].bar(x + (k - 1) * w, [r[1 + k] for r in rows], w, color=c, label=lab); ax[1].bar(x + (k - 1) * w, [r[4 + k] for r in rows], w, color=c, label=lab)
    for a_, t in zip(ax, ('mean relative light (absorbed / area (b))', 'self-shading')):
        a_.set_xticks(x); a_.set_xticklabels([r[0] for r in rows], rotation=60, fontsize=8); a_.set_title(t, fontsize=10); a_.grid(axis='y', alpha=0.25)
    ax[0].legend(frameon=False, fontsize=8); fig.suptitle('14 plants: v1 / v1_tau / v1tau_smoothn_r2 (d_ex 2.4 mm)', fontsize=11); fig.text(0.01, 0.01, FOOT, fontsize=6)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(FIG / 'plants14_three_versions_summary.png', dpi=140); plt.close(fig); print('summary', len(rows))
