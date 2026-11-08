"""Write the K5 kernel folders (plant split) from kaggle_k5/k5_plants.py."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; SRC = (HERE / 'kaggle_k5/k5_plants.py').read_text(encoding='utf-8')
for tag, plants in (('a', 'R3-1b,n_01,n_02,n_03,n_04'), ('b', 'n_05,n_06,t2_02,t2_04,t2_05'), ('c', 'e_01,e_04,e_05,e_07')):
    d = HERE / f'kaggle_k5{tag}'; d.mkdir(exist_ok=True)
    (d / 'k5_plants.py').write_text(SRC.replace('@PLANTS@', plants), encoding='utf-8')
    json.dump({'id': f'dolgfish/static-v1tau-k5{tag}-20261104', 'title': f'static-v1tau-k5{tag}-20261104', 'code_file': 'k5_plants.py', 'language': 'python', 'kernel_type': 'script',
               'is_private': True, 'enable_gpu': False, 'enable_internet': False,
               'dataset_sources': ['dolgfish/static-tau-bundle-20261031', 'dolgfish/static-nine-bundle-20261028', 'dolgfish/static-normals-20261104', 'dolgfish/static-v1tau-ref-20261104'],
               'competition_sources': [], 'kernel_sources': []}, open(d / 'kernel-metadata.json', 'w'))
print('ok')
