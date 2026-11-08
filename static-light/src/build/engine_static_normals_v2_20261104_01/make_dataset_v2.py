"""Flat dataset static-normals-v2-20261104 (CONTRACT §5): normals.py, pool.npz (pooled real disc arrays of test 1), config, 7 MTG meshes."""
import hashlib, json, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; B = ROOT / 'build'
T1 = B / 'engine_static_normals_20261104_01'; FL = HERE / 'kaggle_flat'
MTG = B / 'mtg_build_v2_clicks_20261103_02/static'
cfg1 = json.load(open(T1 / 'kaggle_data/normals_config.json')); cal1 = json.load(open(T1 / 'out_k1/synthetic/calibration.json'))
pz = np.load(T1 / 'out_k1/real/pooled.npz', allow_pickle=True)['d'].item()
real = cfg1['real']
a = np.concatenate([pz[p]['a'] for p in real]); b = np.concatenate([pz[p]['b'] for p in real]); op = np.concatenate([pz[p]['op'] for p in real])
rho = float(np.median([pz[p]['density'] for p in real])); so = float(np.median([pz[p]['rms'] for p in real]))
assert abs(rho - cal1['density_per_mm2']) < 1e-9 and abs(so - cal1['sigma_off_mm']) < 1e-9, (rho, so)
np.savez_compressed(FL / 'pool.npz', a=a, b=b, op=op, density=rho, sigma_off=so)
for p in cfg1['synthetic']:
    shutil.copy2(MTG / f'mtg_{p}_engine.obj', FL / f'mtg__{p}.obj')
json.dump(dict(synthetic=cfg1['synthetic'], synthetic_seed=20261105, noise_targets_deg=[44.0, 30.0], sigma_grid_deg=[2.5 * i for i in range(37)], radii_mm=[2, 3, 5], seeds=[101, 202],
               d_ex_mm=2.4, rule=dict(plant_bias_pct=3.0, leaf_median_abs_err_pct=5.0, noise_deg=44.0), pool_source='build/engine_static_normals_20261104_01/out_k1/real/pooled.npz',
               correlated_cell_mm=3.0), open(FL / 'normals_v2_config.json', 'w'), indent=1)
json.dump({'title': 'static-normals-v2-20261104', 'id': 'dolgfish/static-normals-v2-20261104', 'licenses': [{'name': 'CC0-1.0'}]}, open(FL / 'dataset-metadata.json', 'w'))
h = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(FL.iterdir()) if p.is_file()}
json.dump(h, open(HERE / 'dataset_hashes.json', 'w'), indent=1)
json.dump({str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in
           [T1 / 'out_k1/real/pooled.npz', T1 / 'out_k1/synthetic/calibration.json', T1 / 'kaggle_data/normals_config.json'] + [MTG / f'mtg_{p}_engine.obj' for p in cfg1['synthetic']]},
          open(HERE / 'CONTRACT_HASHES.json', 'w'), indent=1)
print(len(h), 'files', round(sum(p.stat().st_size for p in FL.iterdir()) / 1e6, 1), 'MB; density', rho, 'sigma_off', so)
