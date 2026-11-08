"""Stem definition of optics rule 2 (CONTRACT 짠1; geometry only, before any light number).
Pilot (R3-1b): stem = axis_kind 1 (labels_rachis.npz) = rule 1. Other 13: stem = opaque surfels within d_stem of the stem polylines and not carrying a compound-leaf label
(dz10 snapshot status 1, label > 0). Polylines: main axis = `axis.curve_mm` of build/mtg_build_v2_clicks_20261103_02/static/mtg_<id>_spec.json (adopted surfel fit; no `_02` spec has side axes),
side axes = the clicked lateral axes (`fit.curve_mm`) of the plant's dz10 skeleton; t2_* (not in the MTG set): main and lateral axes of their dz10 skeletons. Distance to a polyline =
distance to the polyline resampled at 0.05 mm (error <= 0.03 mm). d_stem = 1.0 / 1.5 / 2.5 mm. Pilot cross-check: rule 2 applied to R3-1b (its `_02` spec axis) vs axis_kind 1."""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]; B = R / 'build'
sys.path.insert(0, str(B / 'engine_static_tau_20261031_01/kaggle_data'))
from static.geometry import read_ply
S = B / 'mtg_build_v2_clicks_20261103_02/static'; DZ = B / 'p1_dz10_labels_20261101_01'; NF = B / 'engine_static_normals_20261104_01/kaggle_flat'
SKV = {'n_01': '5e803db0e8f8', 'n_02': '6e44394dd4d8', 'n_03': 'd267404f6ada', 'n_04': 'a40266c80d19', 'n_05': 'bcf8d6cef69a', 'n_06': '64b0f9b553e9', 'e_01': '26f27427b73b', 'e_04': '73fdd1ffeead',
       'e_05': '524f23114149', 'e_07': '16690f4f8d25', 't2_02': 'a6e7d3823d42', 't2_04': '6e2e267877ca', 't2_05': 'f67c164d6f3c'}
DS = (1.0, 1.5, 2.5); OUT = HERE / 'kaggle_flat'; OUT.mkdir(exist_ok=True)


def ply(pid):
    if pid == 'R3-1b':
        return B / 'engine_static_nine_20261028_01/kaggle_data/R3-1b/surfels.ply'
    if pid.startswith('e_'):
        return NF / f'{pid}__surfels.ply'
    return B / f'engine_static_nine_20261028_01/kaggle_data/{pid}/surfels.ply'


def area_b(pid, rows):
    if pid.startswith('e_'):
        z = np.load(NF / f'{pid}__area_b.npz'); assert np.array_equal(z['rows'], rows); return z['area_b_mm2']
    d = {}
    with open(B / f'engine_static_nine_20261028_01/kaggle_data/{pid}/static_surfels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith('#')):
            d[int(r['surfel_row'])] = float(r['area_b_mm2'])
    return np.array([d[int(x)] for x in rows])


def resample(poly, step=0.05):
    poly = np.asarray(poly, float); out = [poly[:1]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step))); out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.concatenate(out)


def axis_line(ax):
    """Clicked-axis polyline: fit.curve_mm if the axis was fitted, else the clicked points (board m -> mm) in click order (stated)."""
    f = ax.get('fit') or {}
    if f.get('curve_mm'):
        return f['curve_mm'], 'clicked fit'
    pts = [p['xyz'] for p in ax.get('points', []) if p.get('xyz')]
    return ([[1000 * c for c in p] for p in pts], 'clicked points (no fit)') if len(pts) >= 2 else (None, None)


def polylines(pid):
    lines, src = [], []
    if pid != 't2_02' and not pid.startswith('t2_'):
        sp = json.load(open(S / f'mtg_{pid}_spec.json', encoding='utf-8')); lines.append(sp['axis']['curve_mm']); src.append(f'mtg_{pid}_spec.json axis.curve_mm (main, adopted surfel fit)')
        if pid != 'R3-1b':
            sk = json.load(open(DZ / f'{pid}_{SKV[pid]}/skeleton_v1.json', encoding='utf-8'))
            for ax in sk.get('axes', []):
                if ax['type'] != 'main' and ax['status'] == 'active':
                    l, how = axis_line(ax)
                    if l: lines.append(l); src.append(f"{pid}_{SKV[pid]}/skeleton_v1.json {ax['axis_uid']} ({ax['type']}, {how})")
    else:
        sk = json.load(open(DZ / f'{pid}_{SKV[pid]}/skeleton_v1.json', encoding='utf-8'))
        for ax in sk['axes']:
            if ax['status'] == 'active':
                l, how = axis_line(ax)
                if l: lines.append(l); src.append(f"{pid}_{SKV[pid]}/skeleton_v1.json {ax['axis_uid']} ({ax['type']}, {how})")
    return lines, src


rows_out = []
for pid in ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05', 'e_01', 'e_04', 'e_05', 'e_07']:
    d = read_ply(ply(pid)); rows = np.nonzero(d['opacity'] > 0.5)[0]; X = np.stack([d['x'], d['y'], d['z']], 1)[rows].astype(float) * 1000.0; A = area_b(pid, rows)
    lines, src = polylines(pid); P = np.concatenate([resample(l) for l in lines]); dist = cKDTree(P).query(X)[0]
    if pid == 'R3-1b':
        z = np.load(B / 'engine_static_tau_20261031_01/kaggle_data/labels_rachis.npz'); o = np.argsort(z['rows']); idx = o[np.searchsorted(z['rows'][o], rows)]
        ax = z['axis_kind'][idx].astype(int); r8 = z['label'][idx].astype(int); leaf = (ax == 0) & (r8 >= 1); stem_rule1 = ax == 1
    else:
        z = np.load(NF / f'labels__{pid}.npz'); o = np.argsort(z['rows']); idx = o[np.searchsorted(z['rows'][o], rows)]
        leaf = (z['status'][idx] == 1) & (z['label'][idx] > 0); stem_rule1 = None
    masks = {f'stem_d{dd:g}': (dist <= dd) & ~leaf for dd in DS}
    use = {k: (stem_rule1 if pid == 'R3-1b' else v) for k, v in masks.items()}                 # pilot uses its stem class (rule 1 definition item 1)
    np.savez_compressed(OUT / f'stem__{pid}.npz', rows=rows, dist_mm=dist.astype(np.float32), leaf_labelled=leaf, **use, **({'stem_rule2_geometric_' + k: v for k, v in masks.items()} if pid == 'R3-1b' else {}))
    for dd in DS:
        m = use[f'stem_d{dd:g}']
        row = dict(plant=pid, d_stem_mm=dd, stem_surfels=int(m.sum()), surfels=len(rows), stem_count_share=round(float(m.mean()), 5), stem_area_b_share=round(float(A[m].sum() / A.sum()), 5),
                   used_definition='axis_kind 1 (pilot, rule-1 stem class)' if pid == 'R3-1b' else f'within {dd:g} mm of stem polylines, not leaf-labelled', polylines=' | '.join(src))
        if pid == 'R3-1b':
            g = masks[f'stem_d{dd:g}']; inter = (g & stem_rule1).sum(); union = (g | stem_rule1).sum()
            row.update(pilot_rule2_geometric_surfels=int(g.sum()), pilot_rule2_geometric_area_share=round(float(A[g].sum() / A.sum()), 5), pilot_iou_vs_stem_class=round(float(inter / union), 4) if union else None,
                       pilot_share_of_stem_class_caught=round(float(inter / stem_rule1.sum()), 4))
        rows_out.append(row)
    print(pid, {k: int(v.sum()) for k, v in use.items()}, 'polylines', len(lines))
keys = sorted({k for r in rows_out for k in r}, key=lambda k: list(rows_out[0]).index(k) if k in rows_out[0] else 99)
with open(HERE / 'stem_definition.csv', 'w', encoding='utf-8', newline='') as f:
    f.write('# optics rule 2 stem definition (geometry only, before any light number); area (b) = P1 silhouette-calibrated area; pilot uses its stem class, the geometric rule is a cross-check\n')
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_out)
print('stem_definition.csv', len(rows_out), 'rows')
