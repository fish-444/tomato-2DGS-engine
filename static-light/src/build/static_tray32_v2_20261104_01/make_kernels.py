"""Build the Kaggle inputs and kernels of the tray32 runs: flat input dataset (placement tables) and kernel scripts that embed tray32_lib.py + tray32_main.py.
T1 = conv (C4), extra (diffuse + e_* bracket), scene (in-scene absorbed light, 36 scenes) for table A; T2 = iso (isolated copies) for table A. Table B: T3 (scene) / T4 (iso) made later."""
import json, shutil, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
LIB = (HERE / 'tray32_lib.py').read_text(encoding='utf-8'); MAIN = (HERE / 'tray32_main.py').read_text(encoding='utf-8')
assert "'''" not in LIB and "'''" not in MAIN
TEMPLATE = '''import json, os, sys
from pathlib import Path
OUTD = Path('/kaggle/working'); OUTD.mkdir(exist_ok=True)
LIB = r\'\'\'@LIB@\'\'\'
MAIN = r\'\'\'@MAIN@\'\'\'
(OUTD / 'tray32_lib.py').write_text(LIB, encoding='utf-8')
for MODE in @MODES@:
    sys.argv = ['tray32_main.py', '--table', '@TABLE@', '--mode', MODE, '--out', '/kaggle/working', '--lib', '/kaggle/working']
    exec(compile(MAIN, 'tray32_main.py', 'exec'), {'__name__': '__main__', '__file__': '/kaggle/working/tray32_main.py'})
'''
def kernel(tag, table, modes, desc):
    d = HERE / f'kaggle_{tag}'; d.mkdir(exist_ok=True)
    src = TEMPLATE.replace('@LIB@', LIB).replace('@MAIN@', MAIN).replace('@MODES@', repr(modes)).replace('@TABLE@', table)
    (d / 'k_tray32.py').write_text(src, encoding='utf-8')
    json.dump({'id': f'dolgfish/static-tray32-{tag}-20261104', 'title': f'static-tray32-{tag}-20261104', 'code_file': 'k_tray32.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
               'enable_gpu': False, 'enable_internet': False, 'dataset_sources': ['dolgfish/static-tau-bundle-20261031', 'dolgfish/static-nine-bundle-20261028', 'dolgfish/static-normals-20261104', 'dolgfish/static-tray32-input-20261104'],
               'competition_sources': [], 'kernel_sources': []}, open(d / 'kernel-metadata.json', 'w'))
inp = HERE / 'kaggle_input'; inp.mkdir(exist_ok=True)
for t in ('A', 'B'):
    shutil.copy2(HERE / f'placement_{t}.csv', inp / f'placement_{t}.csv')
json.dump({'title': 'static-tray32-input-20261104', 'id': 'dolgfish/static-tray32-input-20261104', 'licenses': [{'name': 'CC0-1.0'}]}, open(inp / 'dataset-metadata.json', 'w'))
kernel('t1', 'A', ['conv', 'extra', 'scene'], 'table A in-scene'); kernel('t2', 'A', ['iso'], 'table A isolated')
kernel('t3', 'B', ['scene'], 'table B in-scene'); kernel('t4', 'B', ['iso'], 'table B isolated')
print('ok', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in (HERE / 'kaggle_t1').iterdir()})
