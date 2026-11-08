"""Amendment 2 (option b): rescale fill-disc semi-axes so each disc's footprint / credited area = the leaf's reconstructed-surfel footprint / area ratio. Reads kaggle_flat/fill__*.npz (kept),
writes kaggle_flat_b/fill__*.npz + k8sensors copies + fill_footprint_ratio.csv. Local, small."""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lai_range as LR
SRC = HERE / 'kaggle_flat'; DST = HERE / 'kaggle_flat_b'; DST.mkdir(exist_ok=True)
LV = json.load(open(HERE / 'leaves.json')); CASES = ['R', 'B', 'R_reach4', 'R_reach8', 'R_grid1', 'R_grid3']; rows = []
for pid in LR.PLANTS:
    srows, X, A, ab, leaf, uids = LR.load(pid); ratio = {}
    for l in [l for l in LV['leaves'] if l['plant'] == pid]:
        m = leaf == l['label']
        if m.any():
            ratio[l['label']] = (float((np.pi * ab[m, 0] * ab[m, 1]).sum() / A[m].sum()), l['disc_a_med'], l['disc_b_med'], l['leaf'])
    for case in CASES:
        z = dict(np.load(SRC / f'fill__{pid}__{case}.npz'))
        if len(z['area_mm2']):
            r = np.array([ratio[int(k)][0] for k in z['leaf_label']]); am = np.array([ratio[int(k)][1] for k in z['leaf_label']]); bm = np.array([ratio[int(k)][2] for k in z['leaf_label']])
            assert np.allclose(z['a_m'] * 1000, am) and np.allclose(z['b_m'] * 1000, bm), (pid, case)
            s = np.sqrt(r * z['area_mm2'] / (np.pi * am * bm)); z['a_m'] = z['a_m'] * s; z['b_m'] = z['b_m'] * s
            fp = np.pi * z['a_m'] * z['b_m'] * 1e6
            rows.append(dict(plant=pid, case=case, fill_discs=len(s), credited_mm2=round(float(z['area_mm2'].sum()), 1), footprint_mm2=round(float(fp.sum()), 1), footprint_over_area=round(float(fp.sum() / z['area_mm2'].sum()), 3),
                             scale_median=round(float(np.median(s)), 3), disc_radius_eq_median_mm=round(float(np.median(np.sqrt(z['a_m'] * z['b_m']))) * 1000, 3)))
        np.savez_compressed(DST / f'fill__{pid}__{case}.npz', **z)
    shutil.copyfile(SRC / f'k8sensors__{pid}.json', DST / f'k8sensors__{pid}.json')
    for lab, (rr, a_, b_, u) in ratio.items():
        rows.append(dict(plant=pid, case='leaf_ratio', leaf=u, footprint_over_area=round(rr, 3)))
json.dump({'title': 'static-lai-range-b-20261105', 'id': 'dolgfish/static-lai-range-b-20261105', 'licenses': [{'name': 'CC0-1.0'}]}, open(DST / 'dataset-metadata.json', 'w'))
keys = sorted({k for r in rows for k in r}, key=lambda k: ['plant', 'case', 'leaf', 'fill_discs', 'credited_mm2', 'footprint_mm2', 'footprint_over_area', 'scale_median', 'disc_radius_eq_median_mm'].index(k))
with open(HERE / 'fill_footprint_ratio.csv', 'w', encoding='utf-8', newline='') as f:
    f.write('# amendment 2 (option b): fill disc footprint / credited area = leaf reconstructed-surfel footprint / area (b); footprint = pi a b with footprint-k semi-axes; fill_surfels_assumed\n')
    w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)
lr = [r['footprint_over_area'] for r in rows if r['case'] == 'leaf_ratio']; rc = [r for r in rows if r['case'] == 'R']
print('leaf ratio median', np.median(lr), 'range', min(lr), max(lr)); [print(r['plant'], r['footprint_over_area'], r['scale_median'], r['disc_radius_eq_median_mm']) for r in rc]
