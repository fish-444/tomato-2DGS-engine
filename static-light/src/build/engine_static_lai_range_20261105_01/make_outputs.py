"""Addendum tables + all figures (CONTRACT §6) from K15 (out_k15a/b/<id>/plant.json), the range tables and the fill npz. Local, small.
usage: make_outputs.py [glob of plant.json]   (default out_k15[ab]/*/plant.json)"""
import csv, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
import shutil, tempfile, time
_TMP = Path(tempfile.mkdtemp())


def save(fig, name):
    t = _TMP / name; fig.savefig(t, dpi=130); plt.close(fig)
    for _ in range(20):                                   # OneDrive may refuse a write for a moment
        try:
            shutil.copyfile(t, FIG / name); return
        except OSError:
            time.sleep(3)
    raise OSError(name)
G = sys.argv[1] if len(sys.argv) > 1 else str(HERE / 'out_k15[ab]/*/plant.json')
SUF = sys.argv[2] if len(sys.argv) > 2 else ''                                    # amendment 3: '_cell' for out_k15c/d (run-1 files not overwritten)
ORDER = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07', 't2_02', 't2_04', 't2_05']
P = {json.load(open(f, encoding='utf-8'))['plant']: json.load(open(f, encoding='utf-8')) for f in glob.glob(G)}
P = {k: P[k] for k in ORDER if k in P}
LR = {r['plant']: r for r in csv.DictReader(l for l in open(HERE / 'lai_range_per_plant.csv', encoding='utf-8') if not l.startswith('#'))}
CASES = ['min', 'max_R', 'max_B', 'max_R_reach4', 'max_R_reach8', 'max_R_grid1', 'max_R_grid3']
LAB = 'lai_range; fill_surfels_assumed; relative_light; skeleton_labels_ungraded; conditional_on_unvalidated_scale; not_validation; max case is an assumption-based upper bound, not a measurement'
gate_ok = {pid: d['gate']['identical'] for pid, d in P.items()}
rows, lrows = [], []
for pid, d in P.items():
    mn = d['cases']['min']
    for c in CASES:
        x = d['cases'].get(c)
        if x is None:
            continue
        s = x['sensors']
        arm = 'B' if c == 'max_B' else 'R'; up = float(LR[pid][f'LAI_up_{arm}'])
        g = d['gate']; ext = 'pass' if g['absorbed_here'] == g['absorbed_ref'] else f"diff {g['absorbed_here'] - g['absorbed_ref']:.1e}"
        rows.append(dict(plant=pid, case=c, gate_declared_LAI_and_layers_identical=bool(g['lai_here'] == g['lai_ref'] and g['max_layer_T_diff'] == 0.0), extra_check_not_declared_absorbed_total=ext,
                         LAI=round(s['lai_pot_top'], 4), arm=arm if c != 'min' else '', LAI_up_area=round(up, 4) if c != 'min' else '', LAI_max_fill=round(s['lai_pot_top'], 4) if c != 'min' else '',
                         fill_shortfall_LAI=round(up - s['lai_pot_top'], 4) if c != 'min' else '',
                         fill_area_mm2=round(x['fill_area_mm2'], 1), fill_discs=x.get('fill_discs', 0), T_rim_layer=s['transmittance_at_rim_layer'], T_lowest_layer_in_pot=s['transmittance_lowest_layer_inside_pot'],
                         k_extinction_diagnostic=s['k_extinction_diagnostic'], absorbed_vs_min=round(x['absorbed_a1'] / mn['absorbed_a1'], 6),
                         absorbed_a1_ru_mm2=round(x['absorbed_a1'], 3), self_shading=round(x['self_shading'], 4), fill_share_of_absorbed=round(x.get('fill_share', 0.0), 4)))
        for L in s['profile']:
            lrows.append(dict(plant=pid, case=c, layer=L['layer'], z_lo_mm=L['z_mm'][0], z_hi_mm=L['z_mm'][1], transmittance=L['transmittance'], lai_above_pot_top=L['lai_above_pot_top']))
for name, rr in ((f'lai_transmittance_min_max{SUF}.csv', rows), (f'layer_transmittance_min_max{SUF}.csv', lrows)):
    with open(HERE / name, 'w', encoding='utf-8', newline='') as f:
        f.write(f'# {LAB}; reference v1tau_smoothn_r2; ground = pot-top square 59 x 59 mm; 5 mm layers; min = reconstructed surfels only (= K8 exactly); max = + fill discs (tau 0.07, a 0.85) in the compound-leaf boxes\n')
        w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)

# figure 1: LAI bar with range (planned LAI_up R/B and achieved max)
fig, ax = plt.subplots(figsize=(10, 4)); x = np.arange(len(P))
lo = [P[p]['cases']['min']['sensors']['lai_pot_top'] for p in P]; aR = [P[p]['cases']['max_R']['sensors']['lai_pot_top'] for p in P]; pR = [float(LR[p]['LAI_up_R']) for p in P]; pB = [float(LR[p]['LAI_up_B']) for p in P]
aB = [P[p]['cases']['max_B']['sensors']['lai_pot_top'] for p in P]
ax.bar(x, lo, color='#4a7ab7', label='min = current LAI')
ax.vlines(x - 0.12, lo, pR, color='#e6a5a3', lw=4, label='LAI_up_area, arm R (area only)'); ax.plot(x - 0.12, aR, 'k_', ms=10, mew=2, label='LAI_max_fill, arm R (fill placed)')
ax.vlines(x + 0.12, lo, pB, color='#bfbfbf', lw=4, label='LAI_up_area, arm B'); ax.plot(x + 0.12, aB, '_', color='#555555', ms=10, mew=2, label='LAI_max_fill, arm B')
ax.set_xticks(x, list(P), rotation=45); ax.set_ylabel('LAI (pot-top square)'); ax.legend(fontsize=8); ax.set_title('LAI range per plant (lai_range; max is an assumption-based bound)', fontsize=10)
fig.tight_layout(); save(fig, f'lai_range_bar{SUF}.png')

# figure 2: completeness by rank
B = [r for r in csv.DictReader(l for l in open(HERE / 'compound_leaf_boxes.csv', encoding='utf-8') if not l.startswith('#'))]
fig, ax = plt.subplots(figsize=(7, 4))
for cls, mk in (('main', 'o'), ('side', 's')):
    bb = [r for r in B if r['cls'] == cls and r['rank'] not in ('', 'None') and r['c_R'] not in ('', 'None')]
    ax.scatter([int(float(r['rank'])) for r in bb], [float(r['c_R']) for r in bb], marker=mk, alpha=0.6, label=f'{cls} (c = A_rec / A_exp arm R)')
ax.axhline(1, color='k', lw=0.5); ax.set_xlabel('rank (from the base)'); ax.set_ylabel('completeness c_R'); ax.legend(fontsize=8); ax.set_title('Compound-leaf completeness by rank (lai_range)', fontsize=10)
fig.tight_layout(); save(fig, f'completeness_by_rank{SUF}.png')

# figure 3: transmittance profiles with min / max band
n = len(P); nc = 5; nr = int(np.ceil(n / nc)); fig, axs = plt.subplots(nr, nc, figsize=(3 * nc, 3 * nr), sharex=True, squeeze=False)
for ax, pid in zip(axs.ravel(), P):
    get = lambda c: ([L['transmittance'] for L in P[pid]['cases'][c]['sensors']['profile']], [np.mean(L['z_mm']) for L in P[pid]['cases'][c]['sensors']['profile']])
    t0, z = get('min'); tR, _ = get('max_R'); tB, _ = get('max_B')
    ax.fill_betweenx(z, np.minimum(t0, tR), np.maximum(t0, tR), color='#f2c4c4', label='band [min, max_fill R]')
    ax.plot(t0, z, color='#4a7ab7', label='min (current)'); ax.plot(tR, z, color='#c0504d', label='max_fill R'); ax.plot(tB, z, ls='--', color='#7f7f7f', label='max_fill B'); ax.set_title(pid, fontsize=9); ax.set_xlim(0, 1)
for ax in axs.ravel()[n:]:
    ax.axis('off')
axs[0, 0].legend(fontsize=7); fig.supxlabel('5 mm layer canopy transmittance'); fig.supylabel('height above board (mm)'); fig.suptitle('Layer transmittance, min / max (lai_range; fill_surfels_assumed; relative_light)', fontsize=10)
fig.tight_layout(); save(fig, f'layer_transmittance_band{SUF}.png')

# figures 4/5: boxes over surfels (R3-1b + n_01), reconstructed vs fill (top view)
import lai_range as LRm
LV = json.load(open(HERE / 'leaves.json'))
for pid in ('R3-1b', 'n_01'):
    srows, X, A, ab, leaf, uids = LRm.load(pid); z = np.load(HERE / f'kaggle_flat/fill__{pid}__R.npz'); F = z['centre_board_m'] * 1000.0
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(X[:, 0], X[:, 1], s=0.2, c='#9fb8d9', label='reconstructed surfels')
    if len(F):
        ax.scatter(F[:, 0], F[:, 1], s=1.5, c='#c0504d', label=f'fill discs (case R, {z["area_mm2"].sum():.0f} mm2)')
    for l in [l for l in LV['leaves'] if l['plant'] == pid and 'frame' in l]:
        U = np.array(l['frame']); c = np.array(l['centre']); lo_, hi_ = np.array(l['lo']), np.array(l['hi'])
        cor = np.array([[lo_[0], lo_[1], 0], [hi_[0], lo_[1], 0], [hi_[0], hi_[1], 0], [lo_[0], hi_[1], 0], [lo_[0], lo_[1], 0]]) @ U + c
        ax.plot(cor[:, 0], cor[:, 1], lw=0.8, color='k'); ax.text(c[0], c[1], l['leaf'].split('-')[-1], fontsize=6)
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.legend(fontsize=8, markerscale=6); ax.set_title(f'{pid}: compound-leaf boxes (PC1 x PC2, top view), reconstructed vs fill (fill_surfels_assumed)', fontsize=9)
    fig.tight_layout(); save(fig, f'boxes_fill_{pid}{SUF}.png')
print('plants', list(P), 'gates', gate_ok)
for r in rows:
    if r['case'] in ('min', 'max_R', 'max_B'):
        print(r['plant'], r['case'], r['LAI'], r['LAI_up_area'], r['fill_shortfall_LAI'], r['T_rim_layer'], r['absorbed_vs_min'], r['self_shading'], r['fill_share_of_absorbed'], r['extra_check_not_declared_absorbed_total'])
