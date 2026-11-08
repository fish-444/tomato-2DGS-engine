"""Flat copy of kaggle_data for upload (the Kaggle CLI skips subfolders): a/b/c -> a__b__c. Kernels rebuild the tree with mirror()."""
import hashlib, json, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent; SRC = HERE / 'kaggle_data'; DST = HERE / 'kaggle_flat'
DST.mkdir(exist_ok=True)
for p in sorted(SRC.rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.name != 'dataset-metadata.json':
        rel = p.relative_to(SRC).parts; assert not any('__' in x for x in rel), rel
        shutil.copy2(p, DST / '__'.join(rel))
shutil.copy2(SRC / 'dataset-metadata.json', DST / 'dataset-metadata.json')
src_h = json.load(open(HERE / 'dataset_hashes.json'))
for rel, h in src_h.items():
    if rel == 'dataset-metadata.json':
        continue
    assert hashlib.sha256((DST / rel.replace('/', '__')).read_bytes()).hexdigest() == h, rel
print(len(list(DST.iterdir())), 'files, hashes equal to dataset_hashes.json')
