"""Runs the declared C4A Step B loop (CONTRACT Amendment 17) unattended: next f from c4a_stepB.py -> prepare arm C4AB<f> (16 seeds) ->
Kaggle dataset + 4 kernels -> watch_push -> repeat until c4a_stepB.py prints STOP. One line per event on stdout."""
import json, re, subprocess, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; PY = [sys.executable, '-X', 'utf8']
ENV = dict(__import__('os').environ, PYTHONIOENCODING='utf-8')


def sh(cmd, cwd=H):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=ENV)
    return r.stdout + r.stderr


def log(*a):
    print(time.strftime('%H:%M'), *a, flush=True)


while True:
    out = sh(PY + ['c4a_stepB.py']); log(out.strip().splitlines()[-1])
    if 'STOP' in out: break
    f = float(re.search(r'NEXT f = ([0-9.]+)', out).group(1)); arm = f'C4AB{f:.4f}'
    order = [l.strip() for l in open(H / 'C4A_area_fit_order.txt') if l.strip()]
    step = len(order) + 1; batch = f'C4AB{step}'
    open(H / 'C4A_area_fit_order.txt', 'a').write(arm + '\n')
    sh(PY + ['prepare_v2.py', batch, arm]); ks = json.loads(sh(PY + ['make_kernels.py', batch, '4']).strip().splitlines()[-1].replace("'", '"'))
    log('prepared', arm, list(ks))
    slug = f'engv2-{batch.lower()}-20261104'
    for _ in range(3):
        o = sh(['kaggle', 'datasets', 'create', '-p', '.'], cwd=H / 'dataset' / batch)
        if 'being created' in o: break
        time.sleep(30)
    while 'ready' not in sh(['kaggle', 'datasets', 'status', f'dolgfish/{slug}']): time.sleep(20)
    log('dataset ready', slug)
    o = sh(PY + ['watch_push.py'] + list(ks)); log('kernels', ' | '.join(l for l in o.splitlines() if l.startswith('DONE')))
log('driver finished')
