"""K20 (CONTRACT amendment 4): K10 of the v1tau reference with the _04 meshes; dataset folder kaggle_c2."""
import json, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
S = B / 'mtg_build_v2_clicks_20261105_04/static'; D = HERE / 'kaggle_c2'; D.mkdir(exist_ok=True)
P = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07']
for p in P:
    shutil.copyfile(S / f'mtg_{p}_engine.obj', D / f'mtg04__{p}.obj')
json.dump({'title': 'leaflet-c2-mtg04-20261105', 'id': 'dolgfish/leaflet-c2-mtg04-20261105', 'licenses': [{'name': 'CC0-1.0'}]}, open(D / 'dataset-metadata.json', 'w'))
K0 = B / 'engine_static_v1tau_reference_20261104_01/kaggle_k10'; s = (K0 / 'k10_mtg_tau.py').read_text(encoding='utf-8')
a = "REF = Path(glob.glob('/kaggle/input/**/mtg__e_01.obj', recursive=True)[0]).parent"; assert a in s
s = s.replace(a, "REF = Path(glob.glob('/kaggle/input/**/mtg04__e_01.obj', recursive=True)[0]).parent   # K20: _04 meshes")
b = "objs = load_obj((REF if pid.startswith('e_') else ND2) / f'mtg__{pid}.obj')"; assert b in s
s = s.replace(b, "objs = load_obj(REF / f'mtg04__{pid}.obj')").replace("log('K10_DONE')", "log('K20_DONE')")
K = HERE / 'kaggle_k20'; K.mkdir(exist_ok=True)
(K / 'k20_mtg04.py').write_text('"""K20 (engine_static_leaflet_arrangement_20261105_01 CONTRACT amendment 4) = K10 with the _04 meshes."""\n' + s, encoding='utf-8')
m = json.load(open(K0 / 'kernel-metadata.json')); m['dataset_sources'] = [x for x in m['dataset_sources'] if 'v1tau-ref' not in x] + ['dolgfish/leaflet-c2-mtg04-20261105']
m.update(id='dolgfish/leaflet-c2-k20-20261105', title='leaflet-c2-k20-20261105', code_file='k20_mtg04.py'); json.dump(m, open(K / 'kernel-metadata.json', 'w')); print(m['dataset_sources'])
