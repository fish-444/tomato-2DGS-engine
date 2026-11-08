"""K19 (CONTRACT amendment 3): k_sep.py of engine_coplanar_sep_20261101_01 with the T list and T dataset replaced; dataset folder kaggle_c1."""
import json, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]; CS = B / 'engine_coplanar_sep_20261101_01'
D = HERE / 'kaggle_c1'; D.mkdir(exist_ok=True)
for f in (CS / 'T_flat_triangles.csv', HERE / 'T21_triangles.csv', HERE / 'T21_flat_triangles.csv', HERE / 'T21_meta.json'):
    shutil.copyfile(f, D / f.name)
json.dump({'title': 'leaflet-c1-t21-20261105', 'id': 'dolgfish/leaflet-c1-t21-20261105', 'licenses': [{'name': 'CC0-1.0'}]}, open(D / 'dataset-metadata.json', 'w'))
s = (CS / 'kaggle_kernel/k_sep.py').read_text(encoding='utf-8')
s = s.replace("DVT = Path(glob.glob('/kaggle/input/**/OFFSET_META.json', recursive=True)[0]).parent", "DVT = Path(glob.glob('/kaggle/input/**/T21_meta.json', recursive=True)[0]).parent   # K19: leaflet Part C, T21 dataset")
i = s.index("TT = {'T': (read_T('T_flat_triangles.csv'), 16)"); j = s.index('\n', i)
s = s[:i] + "TT = {'T_old': (read_T('T_flat_triangles.csv'), 16), 'T21': (read_T('T21_triangles.csv'), 16), 'T21_flat': (read_T('T21_flat_triangles.csv'), 16)}   # K19 (amendment 3)" + s[j:]
s = s.replace("RES = dict(labels=['coplanar_separation', 'representation_comparison_not_validation',", "RES = dict(labels=['leaflet_arrangement_from_reconstruction', 'engine_v21_leaflet_laws_ported', 'coplanar_separation', 'representation_comparison_not_validation',")
assert s.count('T21_meta.json') == 1 and "'T21_flat'" in s and 'leaflet_arrangement_from_reconstruction' in s
K = HERE / 'kaggle_k19'; K.mkdir(exist_ok=True)
(K / 'k19_c1.py').write_text('"""K19 (engine_static_leaflet_arrangement_20261105_01 CONTRACT amendment 3) = k_sep.py of engine_coplanar_sep_20261101_01 with the T list replaced (T_old anchor, T21, T21_flat)."""\n' + s, encoding='utf-8')
m = json.load(open(CS / 'kaggle_kernel/kernel-metadata.json'))
m['dataset_sources'] = [x for x in m['dataset_sources'] if 'coplanar' not in x and 'disc-vs-tri' not in x] + ['dolgfish/leaflet-c1-t21-20261105']
m.update(id='dolgfish/leaflet-c1-k19-20261105', title='leaflet-c1-k19-20261105', code_file='k19_c1.py'); json.dump(m, open(K / 'kernel-metadata.json', 'w'))
print(json.dumps(m))
