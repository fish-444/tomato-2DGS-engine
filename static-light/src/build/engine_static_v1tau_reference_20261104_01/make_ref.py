"""Flat reference dataset static-v1tau-ref-20261104: stored v1 / v1_tau per-surfel kernels (gate G0b), stored tray scenario values (gate G0c)."""
import hashlib, json, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]; FL = HERE / 'kaggle_ref'; FL.mkdir(exist_ok=True)
T = B / 'engine_static_tau_20261031_01'
src = {'R3-1b': T / 'out_k1b/R3-1b/per_surfel_light_tau.npz'}
for p in ['n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05']:
    src[p] = T / f'out_k2/{p}/per_surfel_light_tau.npz'
for p, f in src.items():
    z = np.load(f); np.savez_compressed(FL / f'refk__{p}.npz', ply_row=z['ply_row'], relative_light_v1=z['relative_light_v1'], relative_light_tau=z['relative_light_tau'])
shutil.copy2(T / 'out_k3/tray_scenarios_tau.json', FL / 'tray_scenarios_tau.json')
shutil.copy2(T / 'out_k1b/R3-1b/scenarios_tau.json', FL / 'r31b_scenarios_tau.json')
json.dump({'title': 'static-v1tau-ref-20261104', 'id': 'dolgfish/static-v1tau-ref-20261104', 'licenses': [{'name': 'CC0-1.0'}]}, open(FL / 'dataset-metadata.json', 'w'))
h = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FL.iterdir())}
json.dump(h, open(HERE / 'ref_dataset_hashes.json', 'w'), indent=1); print(len(h), 'files')
