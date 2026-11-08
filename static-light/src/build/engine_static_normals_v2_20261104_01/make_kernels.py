"""Write the two kernel folders (plant split) from kaggle_synth/k3_synth_v2.py."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; SRC = (HERE / 'kaggle_synth/k3_synth_v2.py').read_text(encoding='utf-8')
for tag, plants in (('a', 'R3-1b,n_01,n_02,n_03'), ('b', 'n_04,n_05,n_06')):
    d = HERE / f'kaggle_k3{tag}'; d.mkdir(exist_ok=True)
    (d / 'k3_synth_v2.py').write_text(SRC.replace('@PLANTS@', plants), encoding='utf-8')
    json.dump({'id': f'dolgfish/static-normals-v2-k3{tag}-20261104', 'title': f'static-normals-v2-k3{tag}-20261104', 'code_file': 'k3_synth_v2.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
               'dataset_sources': ['dolgfish/static-nine-bundle-20261028', 'dolgfish/static-normals-v2-20261104'], 'competition_sources': [], 'kernel_sources': []},
              open(d / 'kernel-metadata.json', 'w'))
print('ok')
