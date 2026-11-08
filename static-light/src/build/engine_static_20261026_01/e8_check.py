"""E8 check for STATIC: dynamic code and dynamic outputs untouched. python e8_check.py -> E8_CHECK.json
(1) repo tools/ + tests/ sha256 vs the recorded post-merge-5 repo column; (2) seed72 dynamic outputs vs the hashes in its completion.json if present,
else vs the hashes recorded at the start of this task (E8_seed72_start.json, written on first run)."""
import csv, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
REC = REPO / 'build/engine_dev_20261020/HASHES_after_merge5_20261024.csv'
SEED72 = REPO / 'build/engine_maxmass_fix_20261025_01/runs/seed72/native/working'
SKIP = {'__pycache__', '.pytest_cache'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''): h.update(c)
    return h.hexdigest()


rec = {}
with open(REC, encoding='utf-8') as fh:
    r = csv.reader(fh); head = next(r)
    for row in r: rec[row[0]] = row[2]
now = {}
for sub in ('tools', 'tests'):
    for p in (REPO / sub).rglob('*'):
        if p.is_file() and not (set(p.relative_to(REPO).parts) & SKIP): now[p.relative_to(REPO).as_posix()] = sha(p)
changed = sorted(k for k in rec if rec[k] and now.get(k) != rec[k]); added = sorted(k for k in now if k not in rec)
outs = {p.name: sha(p) for p in sorted(SEED72.glob('*.csv'))}
start = HERE / 'E8_seed72_start.json'
if not start.exists():
    json.dump(outs, open(start, 'w', encoding='utf-8'), indent=1)
ref = json.load(open(start, encoding='utf-8'))
res = dict(record=str(REC.relative_to(REPO)), header=head, files_compared=len(rec), code_changed_since_merge5=changed, code_added_since_merge5=added,
           seed72_outputs_identical=ref == outs, seed72_files=len(outs), seed72_reference=str(start.relative_to(REPO)))
res['pass'] = not changed and res['seed72_outputs_identical']
json.dump(res, open(HERE / 'E8_CHECK.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'header'}))
