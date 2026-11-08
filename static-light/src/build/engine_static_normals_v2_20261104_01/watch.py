"""Poll the kernels; download each output when it completes (status parsed only as KernelWorkerStatus.*). usage: watch.py k3a=slug k3b=slug ..."""
import re, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
K = dict(a.split('=') for a in sys.argv[1:]) or {'k3a': 'dolgfish/static-normals-v2-k3a-20261104', 'k3b': 'dolgfish/static-normals-v2-k3b-20261104'}
done = {}; end = time.time() + 5 * 3600
while len(done) < len(K) and time.time() < end:
    for k, slug in K.items():
        if k in done:
            continue
        r = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True)
        m = re.search(r'KernelWorkerStatus\.(\w+)', r.stdout + r.stderr); st = m.group(1) if m else 'unknown'
        print(time.strftime('%H:%M:%S'), k, st, flush=True)
        if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
            out = HERE / f'out_{k}'; out.mkdir(exist_ok=True)
            subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', str(out), '-o'], capture_output=True, text=True)
            done[k] = st; print('DOWNLOADED', k, st, flush=True)
    if len(done) < len(K):
        time.sleep(90)
print('WATCH_END', done, flush=True)
