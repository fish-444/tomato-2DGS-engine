"""Flat dataset static-lai-range-20261105: fill surfels (kaggle_flat/fill__*.npz) + the K8 references for the min-case gate (k8sensors__<id>.json: sensors.tau and the tau total)."""
import glob, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; REF = HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01'; FL = HERE / 'kaggle_flat'
for f in glob.glob(str(REF / 'out_k8*' / '*' / 'plant.json')):
    d = json.load(open(f, encoding='utf-8')); pid = Path(f).parent.name
    json.dump(dict(sensors_tau=d['sensors']['tau'], absorbed_tau_total=d['dex']['2.4']['tau']['total']), open(FL / f'k8sensors__{pid}.json', 'w'))
json.dump({'title': 'static-lai-range-20261105', 'id': 'dolgfish/static-lai-range-20261105', 'licenses': [{'name': 'CC0-1.0'}]}, open(FL / 'dataset-metadata.json', 'w'))
h = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FL.iterdir())}; json.dump(h, open(HERE / 'dataset_hashes.json', 'w'), indent=1); print(len(h), 'files')
