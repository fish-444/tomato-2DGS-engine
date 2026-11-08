"""Assemble the private Kaggle dataset for the nine individuals (file copies + three small reference files). Local memory use: a few MB."""
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
K = HERE / 'kaggle_data'; K.mkdir(exist_ok=True)
ORDER = ['t2_02', 'n_04', 'n_01', 't2_05', 'n_06', 'n_05', 't2_04', 'n_02', 'n_03']
R = REPO / 'build'
(K / 'static').mkdir(exist_ok=True)
for f in (R / 'engine_static_20261026_02/static').glob('*.py'):
    shutil.copy2(f, K / 'static' / f.name)
shutil.copy2(R / 'engine_static_r8pc_20261028_01/kaggle_data/field_planes.npz', K / 'field_planes.npz')
shutil.copy2(R / 'engine_static_r8pc_20261028_01/kaggle_data/light_consts.json', K / 'light_consts.json')
shutil.copy2(R / 'p1_all_3dfirst_20261027_01/user_3d_group_click.jsonl', K / 'user_3d_group_click.jsonl')
# R3-1b (reproduction gate + tenth entry)
d = K / 'R3-1b'; d.mkdir(exist_ok=True)
shutil.copy2(R / 'p1_r31b_reuse_20261024_01/source/R3-1b_surfels_full.ply', d / 'surfels.ply')
shutil.copy2(R / 'p1_vp_cleanup_20261026_01/surfel_leaf_ids.csv', d / 'surfel_leaf_ids.csv')
shutil.copy2(R / 'p1_plant_model_20261026_01/plants/R3-1b/static_surfels.csv', d / 'static_surfels.csv')
z = np.load(R / 'engine_static_r8pc_20261028_01/per_surfel_light.npz')
np.savez_compressed(K / 'r31b_ref.npz', ply_row=z['ply_row'], relative_light=z['relative_light'], relative_light_d5mm=z['relative_light_d5mm'])
r2 = json.load(open(R / 'engine_static_20261026_01/results/run2.json', encoding='utf-8'))['plants']['R3-1b']['reference']
json.dump(dict(absorbed_with_extension_ru_m2=r2['absorbed_with_extension_ru_m2']['mean'], source='engine_static_20261026_01/results/run2.json R3-1b reference, d 5 mm, V-P cleanup leaves, cube-union (b)'),
          open(K / 'run2_ref.json', 'w'), indent=1)
rim = {}
with open(R / 'p1_plant_model_20261026_01/summary.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(l for l in fh if not l.startswith('#')):
        rim[r['id']] = dict(rim_status=r['rim_status'], rim_visual=r['rim_visual'], rim_ratio_vs_59=r['rim_ratio_vs_59'], frame=r['frame'], height_p995_mm=r['height_p995_mm'])
json.dump(rim, open(K / 'rim.json', 'w'), indent=1)
for pid in ORDER:
    d = K / pid; d.mkdir(exist_ok=True)
    shutil.copy2(R / f'p1_recon_all_20261025_01/outputs/{pid}/R3-1b-style_{pid}_surfels_full.ply', d / 'surfels.ply')
    shutil.copy2(R / f'p1_plant_model_20261026_01/plants/{pid}/static_surfels.csv', d / 'static_surfels.csv')
    shutil.copy2(R / f'p1_all_3dfirst_20261027_01/{pid}/merge/groups.npz', d / 'groups.npz')
    shutil.copy2(R / f'p1_all_3dfirst_20261027_01/{pid}/merge/flags.csv', d / 'flags.csv')
    shutil.copy2(R / f'p1_pc_all_20261028_01/{pid}/STAGE1.json', d / 'STAGE1.json')
json.dump(dict(individuals=ORDER, order_reason='t2_02 (human reference), n_04 (clicks), then the P1 job order'), open(K / 'run_config.json', 'w'), indent=1)
json.dump({'title': 'static-nine-bundle-20261028', 'id': 'dolgfish/static-nine-bundle-20261028', 'licenses': [{'name': 'CC0-1.0'}]}, open(K / 'dataset-metadata.json', 'w'))
h = {str(p.relative_to(K)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(K.rglob('*')) if p.is_file() and p.suffix in ('.py', '.ply', '.npz', '.json', '.csv', '.jsonl') and p.stat().st_size < 2e7}
json.dump(h, open(HERE / 'bundle_hashes.json', 'w'), indent=1)
print(sum(p.stat().st_size for p in K.rglob('*') if p.is_file()) / 1e6, 'MB,', len(h), 'files hashed')
