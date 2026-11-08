"""Tables and figures from out_k1 / out_k2 (CONTRACT.md §8). Local, small. python -X utf8 make_report.py"""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent; K1 = HERE / 'out_k1'; K2 = HERE / 'out_k2'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
CFG = json.load(open(HERE / 'kaggle_data/normals_config.json'))
MODES = ['raw', 'smooth_r2', 'smooth_r3', 'smooth_r5']; RAD = {'raw': 0, 'smooth_r2': 2, 'smooth_r3': 3, 'smooth_r5': 5}
TAG = 'normal_smoothing_test; relative_light; conditional_on_unvalidated_scale; skeleton_labels_ungraded; after_checkpoint_v1'
C = {'raw': '#6b6b6b', 'smooth_r2': '#2a7ab9', 'smooth_r3': '#e08a1e', 'smooth_r5': '#3a9e5c', 43.0: '#c23b3b', 30.0: '#2a7ab9', 55.0: '#7a4fb0'}


def wcsv(name, rows, note=''):
    p = HERE / name; p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(f'# {TAG}{"; " + note if note else ""}\n'); w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def r(x, n=4):
    return None if x is None else (round(x, n) if isinstance(x, float) else x)


rule = json.load(open(K1 / 'synthetic/rule.json')); MODE = rule['mode_for_single_radius_parts']
cal = json.load(open(K1 / 'synthetic/calibration.json')); syn = json.load(open(K1 / 'synthetic/rows.json'))
# ---------------- synthetic
rows = [{k: r(v) for k, v in x.items()} for x in syn['rows']]
for tg in CFG['noise_targets_deg']:
    for m in MODES + ['receiver_only_truth']:
        sel = [x for x in syn['rows'] if x['noise_target_deg'] == tg and x['mode'] == m]
        lr = [abs(x['err_pct']) for x in syn['leaf_rows'] if x['noise_target_deg'] == tg and x['mode'] == m]
        rows.append(dict(plant='ALL', noise_target_deg=tg, sigma_deg=sel[0]['sigma_deg'], noise_metric_deg=r(float(np.mean([x['noise_metric_deg'] for x in sel]))), mode=m,
                         truth_total=None, total=None, plant_bias_pct=r(float(np.mean([x['plant_bias_pct'] for x in sel]))), self_truth=r(float(np.mean([x['self_truth'] for x in sel]))),
                         self_mode=r(float(np.mean([x['self_mode'] for x in sel]))), self_change=r(float(np.mean([x['self_change'] for x in sel]))),
                         leaf_median_abs_err_pct=r(float(np.median(lr))), leaf_spearman=r(float(np.median([x['leaf_spearman'] for x in sel if x['leaf_spearman'] is not None])))))
wcsv('synthetic_normals.csv', rows, f"plant=ALL: mean plant bias, pooled per-leaf median |error|, median Spearman; rule {rule['status']} chosen {rule['chosen']}; density {cal['density_per_mm2']:.3f}/mm2, sigma_off {cal['sigma_off_mm']:.3f} mm, sigma for target {cal['sigma_for_target']}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for tg in CFG['noise_targets_deg']:
    xs = [RAD[m] for m in MODES]
    b = [[x['plant_bias_pct'] for x in syn['rows'] if x['noise_target_deg'] == tg and x['mode'] == m] for m in MODES]
    e = [np.median([abs(x['err_pct']) for x in syn['leaf_rows'] if x['noise_target_deg'] == tg and x['mode'] == m]) for m in MODES]
    ax[0].plot(xs, [np.mean(v) for v in b], 'o-', color=C[tg], label=f'{tg:g}° noise'); ax[0].fill_between(xs, [min(v) for v in b], [max(v) for v in b], color=C[tg], alpha=0.12)
    ax[1].plot(xs, e, 'o-', color=C[tg], label=f'{tg:g}° noise')
ax[0].axhspan(-3, 3, color='#999', alpha=0.15); ax[1].axhline(5, color='#999', ls='--')
ax[0].set(xlabel='smoothing radius (mm; 0 = raw)', ylabel='plant absorbed bias vs truth (%)', title='Plant bias (mean, band = range of 7)')
ax[1].set(xlabel='smoothing radius (mm; 0 = raw)', ylabel='per-leaf median |error| (%)', title='Per-leaf error (pooled median)')
for a_ in ax: a_.legend(frameon=False); a_.set_xticks([0, 2, 3, 5]); a_.grid(alpha=0.25)
fig.suptitle('Synthetic recovery (7 MTG meshes) — normal_smoothing_test', fontsize=10); fig.tight_layout(); fig.savefig(FIG / 'synthetic_bias_vs_radius.png', dpi=130); plt.close(fig)

# ---------------- real plants
REAL = CFG['real']; RR = {p: json.load(open(K1 / 'real' / f'{p}.json')) for p in REAL}
rows = []
for p in REAL:
    for m in MODES:
        d = RR[p]['modes'][m]; ks = d['kept_raw_share_by_set']
        rows.append(dict(plant=p, mode=m, absorbed=r(d['absorbed'], 3), change_vs_raw_pct=r(d.get('plant_change_vs_raw_pct', 0.0), 3), mean_relative_light=r(d['mean_relative_light'], 5),
                         self_shading=r(d['self_shading']), self_change=r(d['self_shading'] - RR[p]['modes']['raw']['self_shading']), leaf_rank_spearman_vs_raw=r(d.get('leaf_rank_spearman_vs_raw')),
                         kept_raw_share=r(d['kept_raw_share']), kept_raw_leaf=r(ks.get('leaf')), kept_raw_unassigned=r(ks.get('unassigned')), kept_raw_stem=r(ks.get('stem')),
                         kept_raw_rachis_like=r(ks.get('rachis_like')), noise_median_deg=r(d['noise_metric_median_deg'], 2), noise_p90_deg=r(d['noise_metric_p90_deg'], 2),
                         assigned_leaf_share=r(d['assigned_leaf_absorbed'] / d['absorbed']), G0a_identical=RR[p].get('G0a', {}).get('identical') if m == 'raw' else '',
                         label_source=RR[p]['label_source']))
        lrows = [dict(leaf=x['leaf'], surfels=x['surfels'], area_b_mm2=r(x['area_b_mm2'], 3), absorbed=r(x['absorbed'], 4), light_per_area=r(x['light_per_area'], 6),
                      self_shading=r(x['self_shading'])) for x in d['leaves']]
        order = sorted(range(len(lrows)), key=lambda i: -lrows[i]['absorbed'])
        for rank, i in enumerate(order): lrows[i]['rank_by_absorbed'] = rank + 1
        wcsv(f'per_leaf/per_leaf_light_{m}_{p}.csv', lrows, f'{p} {m}; assigned surfels only (lower bound); {RR[p]["label_source"]}')
wcsv('real_normals_summary.csv', rows, f'chosen mode {MODE} (rule {rule["status"]}); occlusion uses raw normals in every mode')
# histograms
fig, axs = plt.subplots(3, 5, figsize=(15, 8.5)); bins = np.arange(0, 92, 2)
for a_, p in zip(axs.ravel(), REAL):
    for m in ('raw', MODE):
        h = np.array(RR[p]['modes'][m]['noise_hist_counts'], float); a_.step(bins[:-1], h / h.sum(), where='post', color=C[m], label=f"{m} (med {RR[p]['modes'][m]['noise_metric_median_deg']:.0f}°)")
    a_.set_title(p, fontsize=9); a_.legend(fontsize=7, frameon=False); a_.set_xlim(0, 90)
axs.ravel()[-1].axis('off'); fig.supxlabel('angle to neighbour PCA normal within 2 mm (deg)'); fig.supylabel('share of sampled surfels')
fig.suptitle(f'Neighbour-normal angle before/after smoothing ({MODE})', fontsize=10); fig.tight_layout(); fig.savefig(FIG / 'noise_hist_before_after.png', dpi=120); plt.close(fig)
# per-leaf scatter
fig, axs = plt.subplots(3, 5, figsize=(15, 8.5))
for a_, p in zip(axs.ravel(), REAL):
    x = np.array([l['absorbed'] for l in RR[p]['modes']['raw']['leaves']]); y = np.array([l['absorbed'] for l in RR[p]['modes'][MODE]['leaves']])
    a_.scatter(x, y, s=14, color=C[MODE]); mx = max(x.max(), y.max()) * 1.05 if len(x) else 1; a_.plot([0, mx], [0, mx], color='#999', lw=0.8)
    a_.set_title(f"{p}  ρ={RR[p]['modes'][MODE].get('leaf_rank_spearman_vs_raw') or float('nan'):.2f}  Δ={RR[p]['modes'][MODE]['plant_change_vs_raw_pct']:+.1f}%", fontsize=8)
axs.ravel()[-1].axis('off'); fig.supxlabel('leaf absorbed, raw normals (r.u.·mm²)'); fig.supylabel(f'leaf absorbed, {MODE}')
fig.tight_layout(); fig.savefig(FIG / 'per_leaf_raw_vs_smoothed.png', dpi=120); plt.close(fig)
# pilot 3D light map
z = np.load(K1 / 'real' / 'R3-1b_persurfel.npz'); X = z['xyz_mm']; U = z['U'].astype(float)
la, lb = U * z['K_raw'], U * z[f'K_{MODE}']; vmax = np.percentile(np.r_[la, lb], 99)
fig = plt.figure(figsize=(11, 5))
for i, (v, t) in enumerate(((la, 'raw normals'), (lb, MODE))):
    a_ = fig.add_subplot(1, 2, i + 1, projection='3d'); o = np.argsort(X[:, 1])[::-1]
    sc = a_.scatter(X[o, 0], X[o, 1], X[o, 2], c=v[o], s=0.6, cmap='viridis', vmin=0, vmax=vmax, depthshade=False); a_.view_init(elev=45, azim=-90)
    a_.set_title(f'R3-1b light per area — {t}', fontsize=9); a_.set_axis_off()
fig.colorbar(sc, ax=fig.axes, shrink=0.6, label='relative light per area (U·K, same scale)'); fig.savefig(FIG / 'pilot_light_map_raw_vs_smoothed.png', dpi=130); plt.close(fig)

# ---------------- K2: disc vs triangle, scenarios
k2 = json.load(open(K2 / 'k2_results.json')); dv = k2['disc_vs_tri']
rows = [dict(mode=m, kept_raw_share=r(dv['modes'][m]['kept_raw_share']), D_blades_total=r(dv['modes'][m]['D_blades']['total']['mean'], 3), D_all_total=r(dv['modes'][m]['D_all']['total']['mean'], 3),
             T_total=r(dv['T_total'], 3), T_over_D_blades=r(dv['modes'][m]['T_over_D_blades']), T_over_D_all=r(dv['modes'][m]['T_over_D_all']), stored_T_over_D_blades=1.3804,
             chosen=m == MODE) for m in MODES]
wcsv('disc_vs_tri_smoothed.csv', rows, f"G0b {dv['G0b']}; T unchanged (engine-law leaflets, 16 receivers per triangle)")
rows, rank_rows = [], []
for kind, d in k2['scenarios'].items():
    V = d['values']; names = [n for n in V['raw'] if n != 'baseline']; units = [u for u in V['raw']['baseline']]
    for u in units:
        eff = {m: [V[m][n][u] / V[m]['baseline'][u] - 1 for n in names] for m in MODES}
        for m in MODES[1:]:
            for n, e0, e1 in zip(names, eff['raw'], eff[m]):
                rows.append(dict(scene=kind, scenario=n, unit=u, mode=m, rel_effect_raw=r(e0, 5), rel_effect_mode=r(e1, 5), sign_same=bool(np.sign(e0) == np.sign(e1)), chosen=m == MODE))
            same_rank = list(np.argsort(eff['raw'])) == list(np.argsort(eff[m]))
            rank_rows.append(dict(scene=kind, unit=u, mode=m, ranking_same=bool(same_rank), kendall_tau=r(float(kendalltau(eff['raw'], eff[m])[0])), n_scenarios=len(names), chosen=m == MODE))
wcsv('scenario_signs.csv', rows, f"relative effect = scenario/baseline - 1 (same seeds, v1 kernel); G0c r31b {k2['scenarios']['r31b']['G0c']}, tray {k2['scenarios']['tray']['G0c']}")
wcsv('scenario_ranking.csv', rank_rows, 'order of scenario effects within each unit, raw vs mode')

# ---------------- MTG
rows, lrows = [], []
for p in CFG['mtg']:
    d = json.load(open(K1 / 'mtg' / f'{p}.json'))
    for m, c in d['compare'].items():
        rows.append(dict(plant=p, mode=m, n_leaves=c['n_leaves'], leaf_rank_spearman=r(c['spearman']), mtg_leaflets_over_surfel_assigned=r(c['ratio_mtg_over_surfel_assigned']),
                         mtg_leaflets_over_surfel_all=r(c['ratio_mtg_over_surfel_all']), unmatched_mtg_leaves=len(c['unmatched_mtg']), chosen=m == MODE))
        for l in c['leaves']:
            lrows.append(dict(plant=p, mode=m, leaf=l['leaf'], mtg_absorbed=r(l['mtg'], 4), surfel_absorbed=r(l['surfel'], 4), surfel_area_b_mm2=r(l['surfel_area_b_mm2'], 3)))
wcsv('mtg_vs_surfel_leaf_light.csv', rows, 'report only; MTG leaflets = smooth reconstructions from clicks; neither is truth')
wcsv('mtg_vs_surfel_per_leaf.csv', lrows, 'report only')
print('rule', rule); print('MODE', MODE)
