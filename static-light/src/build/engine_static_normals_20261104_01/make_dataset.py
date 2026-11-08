"""Assemble the private Kaggle dataset static-normals-20261104 (CONTRACT.md §2, §6-§7). File copies + small extractions; local memory < 1 GB.
usage: python -X utf8 make_dataset.py {hashes|build}"""
import csv, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; B = ROOT / 'build'
sys.path.insert(0, str(ROOT))
K = HERE / 'kaggle_data'
DZ = B / 'p1_dz10_labels_20261101_01'
MTG = B / 'mtg_build_v2_clicks_20261103_02/static'
LABELS = {'n_01': ('tprox', '5e803db0e8f8'), 'n_02': ('tprox', '6e44394dd4d8'), 'n_03': ('tprox', 'd267404f6ada'), 'n_04': ('tprox', 'a40266c80d19'),
          'n_05': ('tprox', 'bcf8d6cef69a'), 'n_06': ('distance', '64b0f9b553e9'), 'e_01': ('tprox', '26f27427b73b'), 'e_04': ('tprox', '73fdd1ffeead'),
          'e_05': ('tprox', '524f23114149'), 'e_07': ('tprox', '16690f4f8d25'), 't2_02': ('tprox', 'a6e7d3823d42'), 't2_04': ('tprox', '6e2e267877ca'),
          't2_05': ('tprox', 'f67c164d6f3c')}
NINE = ['n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05']
E = ['e_01', 'e_04', 'e_05', 'e_07']
MTG_PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07']
R31B_LABELS = B / 'engine_static_tau_20261031_01/kaggle_data/labels_rachis.npz'
R31B_SKEL = ROOT / 'data/skeletons/R3-1b/2026-10-21/skeleton_v1.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).relative_to(ROOT)).replace('\\', '/')


def inputs():
    fs = [DZ / f'labels_{r}_{p}_{v}.npz' for p, (r, v) in LABELS.items()] + [R31B_LABELS, R31B_SKEL]
    fs += [next((B / f'p1_recon_all_20261025_01/outputs/{e}').glob('*_surfels_full.ply')) for e in E]
    fs += [B / 'engine_static_area_e_20261103_01/per_view_ratios.json', B / 'engine_static_area_e_20261103_01/area_summary.csv',
           B / 'p1_plant_model_20261026_01/plants/n_03/static_surfels.csv', B / 'p1_plant_model_20261026_01/plants/n_03/plant_model.json']
    fs += [B / f'engine_static_nine_20261028_01/kaggle_out/{p}/per_surfel_light.npz' for p in ['R3-1b'] + NINE]
    fs += [B / 'engine_disc_vs_tri_20261101_01/out_k1/results.json', B / 'engine_static_tau_20261031_01/out_k1b/R3-1b/scenarios_tau.json',
           B / 'engine_static_tau_20261031_01/out_k3/tray_scenarios_tau.json']
    fs += [MTG / f'mtg_{p}_engine.obj' for p in MTG_PLANTS] + [MTG / f'mtg_{p}_spec.json' for p in MTG_PLANTS]
    fs += sorted((B / 'engine_static_nine_20261028_01/kaggle_data/static').glob('*.py')) + sorted((B / 'engine_static_tau_20261031_01/kaggle_data/static').glob('*.py'))
    fs += sorted((ROOT / 'tomato3d/plant_model').glob('*.py'))
    return fs


def e_area(pid, r):
    from tomato3d.plant_model import areas as AR
    from tomato3d.plant_model import inputs as I
    sf = I.sio.read_surfels_full(next((I.ALL_OUT / pid).glob('*_surfels_full.ply')))
    rows = np.nonzero(sf['opacity'] >= I.OPACITY_MIN)[0]
    return rows, AR.disc_area_m2(sf['scale'][rows]) * 1e6 * r


def build():
    for d in ('labels', 'ref', 'mtg'):
        (K / d).mkdir(parents=True, exist_ok=True)
    # G0d: recomputation check on n_03 (P1 code, canopy ratio at full precision)
    r03 = json.load(open(B / 'p1_plant_model_20261026_01/plants/n_03/plant_model.json', encoding='utf-8'))['organs']['canopy_surface']['silhouette_ratio']
    rows, A = e_area('n_03', r03)
    old = {}
    with open(B / 'p1_plant_model_20261026_01/plants/n_03/static_surfels.csv', encoding='utf-8') as fh:
        for r_ in csv.DictReader(l for l in fh if not l.startswith('#')):
            old[int(r_['surfel_row'])] = float(r_['area_b_mm2'])
    d = float(np.max(np.abs(np.array([old[int(x)] for x in rows]) - A)))
    g0d = dict(n03_max_abs_diff_mm2=d, passed=bool(d <= 1e-5 and set(old) == set(rows.tolist())))
    print('G0d', g0d); json.dump(g0d, open(HERE / 'G0d_area_check.json', 'w'), indent=1)
    if not g0d['passed']:
        raise SystemExit('G0d failed')
    pv = json.load(open(B / 'engine_static_area_e_20261103_01/per_view_ratios.json'))
    for e in E:
        (K / e).mkdir(exist_ok=True)
        shutil.copy2(next((B / f'p1_recon_all_20261025_01/outputs/{e}').glob('*_surfels_full.ply')), K / e / 'surfels.ply')
        rows, A = e_area(e, pv[e]['ratio'])
        np.savez_compressed(K / e / 'area_b.npz', rows=rows, area_b_mm2=A, ratio=pv[e]['ratio'])
    for p, (route, v) in LABELS.items():
        z = np.load(DZ / f'labels_{route}_{p}_{v}.npz')
        np.savez_compressed(K / 'labels' / f'{p}.npz', rows=z['rows'], label=z['label'], status=z['status'], leaf_uids=z['leaf_uids'],
                            source=np.array(f'build/p1_dz10_labels_20261101_01/labels_{route}_{p}_{v}.npz'))
    shutil.copy2(R31B_LABELS, K / 'labels' / 'R3-1b_labels_rachis.npz')
    sk = json.load(open(R31B_SKEL, encoding='utf-8'))
    m = {l['leaf_uid']: int(l['attrs']['pipeline_leaf']) for l in sk['leaves'] if l['status'] in ('active', 'new', 'candidate_lost')}
    json.dump(m, open(K / 'mtg' / 'R3-1b_uid_to_R8pc.json', 'w'), indent=1)
    for p in ['R3-1b'] + NINE:
        z = np.load(B / f'engine_static_nine_20261028_01/kaggle_out/{p}/per_surfel_light.npz')
        np.savez_compressed(K / 'ref' / f'{p}_K.npz', ply_row=z['ply_row'], relative_light=z['relative_light'], absorbed_ru_mm2=z['absorbed_ru_mm2'])
    S = json.load(open(B / 'engine_disc_vs_tri_20261101_01/out_k1/results.json'))['summary']
    json.dump({k: S[k] for k in ('T|v1', 'D_blades|v1', 'D_all|v1')}, open(K / 'ref' / 'dvt_summary_v1.json', 'w'), indent=1)
    shutil.copy2(B / 'engine_static_tau_20261031_01/out_k1b/R3-1b/scenarios_tau.json', K / 'ref' / 'r31b_scenarios_tau.json')
    shutil.copy2(B / 'engine_static_tau_20261031_01/out_k3/tray_scenarios_tau.json', K / 'ref' / 'tray_scenarios_tau.json')
    for p in MTG_PLANTS:
        shutil.copy2(MTG / f'mtg_{p}_engine.obj', K / 'mtg' / f'{p}.obj')
    json.dump(dict(labels={p: dict(route=r_, skelver=v) for p, (r_, v) in LABELS.items()}, real=['R3-1b'] + NINE + E, nine=NINE, e=E, mtg=MTG_PLANTS,
                   synthetic=['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06'], synthetic_seed=20261104, noise_targets_deg=[43.0, 30.0, 55.0],
                   sigma_grid_deg=[2.5 * i for i in range(37)], seeds=[101, 202], d_ex_mm=2.4, rule=dict(plant_bias_pct=3.0, leaf_median_abs_err_pct=5.0, noise_deg=43.0),
                   fallback_radius_mode='smooth_r5'), open(K / 'normals_config.json', 'w'), indent=1)
    json.dump({'title': 'static-normals-20261104', 'id': 'dolgfish/static-normals-20261104', 'licenses': [{'name': 'CC0-1.0'}]}, open(K / 'dataset-metadata.json', 'w'))
    h = {str(p.relative_to(K)).replace('\\', '/'): sha(p) for p in sorted(K.rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
    json.dump(h, open(HERE / 'dataset_hashes.json', 'w'), indent=1)
    print(round(sum(p.stat().st_size for p in K.rglob('*') if p.is_file()) / 1e6, 1), 'MB', len(h), 'files')


if __name__ == '__main__':
    if sys.argv[1] == 'hashes':
        json.dump({rel(p): sha(p) for p in inputs()}, open(HERE / 'CONTRACT_HASHES.json', 'w'), indent=1); print(len(inputs()), 'inputs hashed')
    else:
        build()
