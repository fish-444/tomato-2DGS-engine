"""Aggregate the synthetic v2 kernels (CONTRACT §2 rule, applied mechanically). Writes synthetic_v2.csv, correlated_noise_report.csv, coplanar_hits.csv,
RULE_v2.json, figures/synthetic_v2_bias_vs_radius.png. python -X utf8 aggregate_synth.py"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
CFG = json.load(open(HERE / 'kaggle_flat/normals_v2_config.json')); RULE = CFG['rule']
TAG = 'normal_smoothing_v2_occluders; relative_light; conditional_on_unvalidated_scale; skeleton_labels_ungraded; after_checkpoint_v1'
rows, leaf_rows, cal = [], [], {}
for f in sorted(HERE.glob('out_k3*/synth/*.json')):
    d = json.load(open(f)); rows += d['rows']; leaf_rows += d['leaf_rows']
for f in sorted(HERE.glob('out_k3*/calibration.json')):
    cal[f.parent.name] = json.load(open(f))
plants = sorted({r['plant'] for r in rows}); assert len(plants) == len(CFG['synthetic']), plants
c0 = next(iter(cal.values()))
ARMS = ['raw', 'rec_r2', 'rec_r3', 'rec_r5', 'smooth_all_r2', 'smooth_all_r3', 'smooth_all_r5', 'occ_only_r2', 'occ_only_r3', 'occ_only_r5', 'recv_true']


def agg(variant, tg, arm):
    sel = [r for r in rows if r['variant'] == variant and r['noise_target_deg'] == tg and r['arm'] == arm]
    lr = [abs(x['err_pct']) for x in leaf_rows if x['variant'] == variant and x['noise_target_deg'] == tg and x['arm'] == arm]
    rho = [x['leaf_spearman'] for x in sel if x['leaf_spearman'] is not None]
    return dict(variant=variant, noise_target_deg=tg, sigma_deg=sel[0]['sigma_deg'], noise_metric_deg=round(float(np.mean([x['noise_metric_deg'] for x in sel])), 2), arm=arm, n_plants=len(sel),
                plant_bias_pct_mean=round(float(np.mean([x['plant_bias_pct'] for x in sel])), 3), plant_bias_pct_min=round(min(x['plant_bias_pct'] for x in sel), 3),
                plant_bias_pct_max=round(max(x['plant_bias_pct'] for x in sel), 3), leaf_median_abs_err_pct=round(float(np.median(lr)), 3),
                leaf_spearman_median=round(float(np.median(rho)), 4), self_change_mean=round(float(np.mean([x['self_change'] for x in sel])), 4),
                coplanar_hits_total=int(sum(x['hits'] for x in sel)), coplanar_hits_share_mean=round(float(np.mean([x['hits_share'] for x in sel])), 6),
                normal_err_median_deg=round(float(np.mean([x['normal_err_median_deg'] for x in sel])), 2), normal_err_p90_deg=round(float(np.mean([x['normal_err_p90_deg'] for x in sel])), 2),
                kept_raw_share=round(float(np.mean([x['kept_raw_share'] for x in sel])), 4))


def wcsv(name, rr, note):
    with open(HERE / name, 'w', encoding='utf-8', newline='') as f:
        f.write(f'# {TAG}; {note}\n'); w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)


A = {}
for variant in ('white', 'corr'):
    for tg in CFG['noise_targets_deg']:
        if variant == 'corr' and tg != RULE['noise_deg']:
            continue
        for arm in ARMS:
            A[(variant, tg, arm)] = agg(variant, tg, arm)
white = [v for k, v in A.items() if k[0] == 'white']
wcsv('synthetic_v2.csv', white, f"mean over {len(plants)} meshes (seed {CFG['synthetic_seed']}); per-plant rows in synthetic_v2_per_plant.csv; independent noise; truth = noise-free discs on the same sampled surfels")
wcsv('synthetic_v2_per_plant.csv', [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows if r['variant'] != 'truth'], 'one row per plant x level x arm')
wcsv('correlated_noise_report.csv', [v for k, v in A.items() if k[0] == 'corr'], f"report only; noise field smooth over {CFG['correlated_cell_mm']} mm, tilt scale calibrated to the same Traits metric; does not change the decision")
wcsv('coplanar_hits.csv', [dict(variant=v['variant'], noise_target_deg=v['noise_target_deg'], arm=v['arm'], hits_total=v['coplanar_hits_total'], hits_share_mean=v['coplanar_hits_share_mean']) for v in A.values()]
     + [dict(variant='truth', noise_target_deg='', arm='truth', hits_total=int(sum(r['hits'] for r in rows if r['variant'] == 'truth')), hits_share_mean=round(float(np.mean([r['hits'] / r['n'] for r in rows if r['variant'] == 'truth'])), 6))],
     'coincident probe (+-1 um along the receiver normal) with the arm\'s occluder orientation; smoothed receiver normals in smoothed arms')
rule = dict(declared=RULE, per_arm={}, generator=dict(calibration=c0['sigma_white'], corr=c0['sigma_corr'], density=c0['density'], sigma_off=c0['sigma_off'],
            note='metric saturates for targets above ~48 deg (test 1: 55 deg target reached 48.6 deg at sigma 90); 55 deg not repeated'))
for arm in ARMS[1:-1]:
    a = A[('white', RULE['noise_deg'], arm)]
    rule['per_arm'][arm] = dict(mean_plant_bias_pct=a['plant_bias_pct_mean'], pooled_leaf_median_abs_err_pct=a['leaf_median_abs_err_pct'],
                                recovers=bool(abs(a['plant_bias_pct_mean']) <= RULE['plant_bias_pct'] and a['leaf_median_abs_err_pct'] <= RULE['leaf_median_abs_err_pct']),
                                eligible=arm.startswith('smooth_all'))
rec = [r for r in (2, 3, 5) if rule['per_arm'][f'smooth_all_r{r}']['recovers']]
rule['chosen_radius_mm'] = rec[0] if rec else None; rule['status'] = 'recovered' if rec else 'not_recovered_v2'
rule['light_model'] = f'v1_smoothn_r{rec[0]}' if rec else None
json.dump(rule, open(HERE / 'RULE_v2.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
col = {'rec': '#2a7ab9', 'smooth_all': '#c23b3b', 'occ_only': '#7a4fb0'}
for tg, ls in zip(CFG['noise_targets_deg'], ('-', '--')):
    xs = [0, 2, 3, 5]
    for kind in ('rec', 'smooth_all', 'occ_only'):
        b = [A[('white', tg, 'raw')]['plant_bias_pct_mean']] + [A[('white', tg, f'{kind}_r{r}')]['plant_bias_pct_mean'] for r in (2, 3, 5)]
        e = [A[('white', tg, 'raw')]['leaf_median_abs_err_pct']] + [A[('white', tg, f'{kind}_r{r}')]['leaf_median_abs_err_pct'] for r in (2, 3, 5)]
        ax[0].plot(xs, b, ls, marker='o', color=col[kind], label=f'{kind} {tg:g}°'); ax[1].plot(xs, e, ls, marker='o', color=col[kind], label=f'{kind} {tg:g}°')
for a_ in ax[:2]:
    a_.set_xticks([0, 2, 3, 5]); a_.set_xlabel('smoothing radius (mm; 0 = raw)'); a_.grid(alpha=0.25)
ax[0].axhspan(-3, 3, color='#999', alpha=0.15); ax[0].set(ylabel='plant absorbed bias vs truth (%)', title='Plant bias (mean of 7 meshes)'); ax[1].axhline(5, color='#999', ls='--'); ax[1].set(ylabel='per-leaf median |error| (%)', title='Per-leaf error (pooled median)')
ax[0].legend(frameon=False, fontsize=7)
vals = {k: A[('white', 44.0, k)]['normal_err_median_deg'] for k in ('raw', 'smooth_all_r2', 'smooth_all_r3', 'smooth_all_r5')}
ax[2].bar(range(4), list(vals.values()), color='#888'); ax[2].set_xticks(range(4)); ax[2].set_xticklabels(['raw', 'r2', 'r3', 'r5']); ax[2].set(ylabel='median angle to true normal (deg)', title='Normal error after smoothing (44°)')
fig.suptitle('Synthetic recovery v2 (occluders smoothed too) — normal_smoothing_v2_occluders', fontsize=10); fig.tight_layout(); fig.savefig(FIG / 'synthetic_v2_bias_vs_radius.png', dpi=130); plt.close(fig)
print(json.dumps(rule['per_arm'], indent=1)); print(rule['status'], rule['chosen_radius_mm'])
for k in ('raw', 'rec_r5', 'smooth_all_r2', 'smooth_all_r3', 'smooth_all_r5', 'occ_only_r5', 'recv_true'):
    a = A[('white', 44.0, k)]; print(k, a['plant_bias_pct_mean'], a['leaf_median_abs_err_pct'], a['leaf_spearman_median'], a['self_change_mean'], a['coplanar_hits_total'], a['normal_err_median_deg'])
