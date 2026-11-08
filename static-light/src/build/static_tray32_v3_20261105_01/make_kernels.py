"""Kaggle kernels of tray32 v3 (embed tray32_lib.py + tray32_main.py). T1: conv (30 cm ref, smoothing on) + regression (smoothing off, 6 scenes at 40 cm) + table A scenes (smoothing on);
T2: table A isolated copies; T3 / T4: table B scenes / isolated copies (pushed after the table A report). Placement input = the v2 dataset (same files, same hashes)."""
import json
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
for MODE, EXTRA in @RUNS@:
    sys.argv = ['tray32_main.py', '--table', '@TABLE@', '--mode', MODE, '--out', '/kaggle/working', '--lib', '/kaggle/working'] + EXTRA
    exec(compile(MAIN, 'tray32_main.py', 'exec'), {'__name__': '__main__', '__file__': '/kaggle/working/tray32_main.py'})
'''
SRC = ['dolgfish/static-tau-bundle-20261031', 'dolgfish/static-nine-bundle-20261028', 'dolgfish/static-normals-20261104', 'dolgfish/static-normals-v2-20261104', 'dolgfish/static-tray32-input-20261104']
def kernel(tag, table, runs):
    d = HERE / f'kaggle_{tag}'; d.mkdir(exist_ok=True)
    (d / 'k_tray32v3.py').write_text(TEMPLATE.replace('@LIB@', LIB).replace('@MAIN@', MAIN).replace('@RUNS@', repr(runs)).replace('@TABLE@', table), encoding='utf-8')
    json.dump({'id': f'dolgfish/static-tray32v3-{tag}-20261105', 'title': f'static-tray32v3-{tag}-20261105', 'code_file': 'k_tray32v3.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
               'enable_gpu': False, 'enable_internet': False, 'dataset_sources': SRC, 'competition_sources': [], 'kernel_sources': []}, open(d / 'kernel-metadata.json', 'w'))
REG = ['--smooth', 'off', '--scenes', 'S26,S30,S34,S38,S42,S46']
kernel('t1', 'A', [('conv', []), ('scene', REG), ('scene', [])]); kernel('t2', 'A', [('iso', [])])
kernel('t3', 'B', [('scene', [])]); kernel('t4', 'B', [('iso', [])])
print('ok')
