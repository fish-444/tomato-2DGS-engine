"""Push a queue of kernels as Kaggle slots free up (max 5 concurrent CPU sessions), then watch all and download outputs. usage: push_all.py folder=slug=name ... (in priority order)"""
import re, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
Q = [a.split('=') for a in sys.argv[1:]]; pushed = {}; done = {}; end = time.time() + 8 * 3600
while time.time() < end and len(done) < len(Q):
    for folder, slug, name in Q:
        if name in pushed:
            continue
        r = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(HERE / folder)], capture_output=True, text=True); txt = (r.stdout + r.stderr).strip()
        ok = 'successfully pushed' in txt; print(time.strftime('%H:%M:%S'), name, 'push', 'OK' if ok else txt.splitlines()[-1][:140], flush=True)
        if ok:
            pushed[name] = slug
        else:
            break                                   # keep the priority order: do not push a lower-priority kernel past a refused one
    for folder, slug, name in Q:
        if name in pushed and name not in done:
            r = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True)
            m = re.search(r'KernelWorkerStatus\.(\w+)', r.stdout + r.stderr); st = m.group(1) if m else 'unknown'
            if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
                out = HERE / f'out_{name}'; out.mkdir(exist_ok=True)
                subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', str(out), '-o'], capture_output=True, text=True); done[name] = st; print(time.strftime('%H:%M:%S'), 'DOWNLOADED', name, st, flush=True)
    time.sleep(150)
print('ALL_END', done, flush=True)
