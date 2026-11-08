"""Write the phase-2 kernel folders (smoothing; Amendment 1): K8a/b/c (14 plants, smooth_all_r2), K9 (scenarios + tray), K10 (MTG), K11 (synthetic tau arm). Also copies the e_* MTG meshes into kaggle_ref."""
import json, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
for p in ('e_01', 'e_04', 'e_05', 'e_07'):
    shutil.copy2(B / f'mtg_build_v2_clicks_20261103_02/static/mtg_{p}_engine.obj', HERE / 'kaggle_ref' / f'mtg__{p}.obj')
SRC = (HERE / 'kaggle_k8/k8_plants.py').read_text(encoding='utf-8')
DS = {'tau': 'dolgfish/static-tau-bundle-20261031', 'nine': 'dolgfish/static-nine-bundle-20261028', 'nd': 'dolgfish/static-normals-20261104', 'ref': 'dolgfish/static-v1tau-ref-20261104', 'v2': 'dolgfish/static-normals-v2-20261104'}
def meta(name, code, srcs):
    return {'id': f'dolgfish/{name}', 'title': name, 'code_file': code, 'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
            'dataset_sources': [DS[s] for s in srcs], 'competition_sources': [], 'kernel_sources': []}
for tag, plants in (('a', 'R3-1b,n_01,n_02,n_03,n_04'), ('b', 'n_05,n_06,t2_02,t2_04,t2_05'), ('c', 'e_01,e_04,e_05,e_07')):
    d = HERE / f'kaggle_k8{tag}'; d.mkdir(exist_ok=True); (d / 'k8_plants.py').write_text(SRC.replace('@PLANTS@', plants), encoding='utf-8')
    json.dump(meta(f'static-v1tau-k8{tag}-20261104', 'k8_plants.py', ['tau', 'nine', 'nd', 'ref', 'v2']), open(d / 'kernel-metadata.json', 'w'))
json.dump(meta('static-v1tau-k9-20261104', 'k9_scen_smooth.py', ['tau', 'nine', 'nd', 'v2']), open(HERE / 'kaggle_k9/kernel-metadata.json', 'w'))
json.dump(meta('static-v1tau-k10-20261104', 'k10_mtg_tau.py', ['tau', 'nine', 'nd', 'ref', 'v2']), open(HERE / 'kaggle_k10/kernel-metadata.json', 'w'))
json.dump(meta('static-v1tau-k11-20261104', 'k11_synth_tau.py', ['tau', 'nine', 'v2']), open(HERE / 'kaggle_k11/kernel-metadata.json', 'w'))
print('ok')
