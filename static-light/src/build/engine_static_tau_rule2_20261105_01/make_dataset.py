"""Flat dataset static-tau-rule2-20261105: stem masks (kaggle_flat/stem__<id>.npz, from make_stem.py) + K8 reference arrays (k8ref__<id>.npz: ply_row, K_tau float32)."""
import glob, hashlib, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REF = HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01'; FL = HERE / 'kaggle_flat'
for f in glob.glob(str(REF / 'out_k8*' / '*' / 'per_surfel.npz')):
    pid = Path(f).parent.name; z = np.load(f); np.savez_compressed(FL / f'k8ref__{pid}.npz', ply_row=z['ply_row'], K_tau=z['K_tau'])
json.dump({'title': 'static-tau-rule2-20261105', 'id': 'dolgfish/static-tau-rule2-20261105', 'licenses': [{'name': 'CC0-1.0'}]}, open(FL / 'dataset-metadata.json', 'w'))
h = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FL.iterdir())}; json.dump(h, open(HERE / 'dataset_hashes.json', 'w'), indent=1)
print(len(h), 'files', sorted(p for p in h if p.startswith('k8ref'))[:3])
