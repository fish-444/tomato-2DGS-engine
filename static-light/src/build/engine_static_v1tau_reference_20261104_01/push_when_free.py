"""Push a kernel folder as soon as Kaggle accepts it (5 concurrent CPU sessions max), then watch it and download the output."""
import re, subprocess, sys, time
from pathlib import Path
folder, slug, name = sys.argv[1], sys.argv[2], sys.argv[3]
HERE = Path(__file__).resolve().parent
end = time.time() + 5 * 3600; pushed = False
while time.time() < end and not pushed:
    r = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(HERE / folder)], capture_output=True, text=True); txt = r.stdout + r.stderr
    print(time.strftime('%H:%M:%S'), 'push:', txt.strip().splitlines()[-1][:160], flush=True)
    pushed = 'successfully pushed' in txt
    if not pushed:
        time.sleep(240)
while time.time() < end:
    r = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True)
    m = re.search(r'KernelWorkerStatus\.(\w+)', r.stdout + r.stderr); st = m.group(1) if m else 'unknown'
    print(time.strftime('%H:%M:%S'), name, st, flush=True)
    if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
        out = HERE / f'out_{name}'; out.mkdir(exist_ok=True)
        subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', str(out), '-o'], capture_output=True, text=True); print('DOWNLOADED', name, st, flush=True); break
    time.sleep(120)
