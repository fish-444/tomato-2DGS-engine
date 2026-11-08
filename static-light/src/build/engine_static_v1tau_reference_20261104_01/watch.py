"""Poll kernels; download each output when it completes. usage: watch.py name=slug ... (status parsed only as KernelWorkerStatus.*)"""
import re, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
K = dict(a.split('=') for a in sys.argv[1:]); done = {}; end = time.time() + 6 * 3600
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
        time.sleep(120)
print('WATCH_END', done, flush=True)
