"""Push the given kernels as Kaggle CPU slots free up (the 5-slot limit is shared), wait for each to finish, download its output to kout/<kernel>/.
python watch_push.py <kernel> [<kernel> ...]   one line per event; exits when all are done."""
import json, subprocess, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; pending = list(sys.argv[1:]); running = []; done = {}
def status(k):
    s = subprocess.run(['kaggle', 'kernels', 'status', f'dolgfish/{k}'], capture_output=True, text=True, encoding='utf-8', errors='replace').stdout
    for t, key in (('complete', 'COMPLETE'), ('error', 'ERROR'), ('cancel', 'CANCEL')):
        if key in s: return t
    return 'running'
while pending or running:
    for k in list(running):
        st = status(k)
        if st != 'running':
            running.remove(k); done[k] = st; o = H / 'kout' / k; o.mkdir(parents=True, exist_ok=True)
            jobs = next((v[k] for f in H.glob('KERNELS_*.json') for v in [json.load(open(f))] if k in v), [])
            for _ in range(8):  # U1 k1 (2026-11-06): first download came back partial; retry until run_log and every job's status.json are present
                subprocess.run(['kaggle', 'kernels', 'output', f'dolgfish/{k}', '-p', str(o), '-o'], capture_output=True, env=dict(__import__('os').environ, PYTHONIOENCODING='utf-8'))
                if (o / 'run_log.txt').exists() and all((o / 'out' / j / 'status.json').exists() for j in jobs): break
                time.sleep(60)
            log = o / 'run_log.txt'; tail = [l for l in log.read_text(encoding='utf-8', errors='replace').splitlines() if '=== done' in l] if log.exists() else ['no run_log']
            print(f'DONE {k} {st} {tail[-1] if tail else "no done line"}', flush=True)
    for k in list(pending):
        r = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(H / 'kernels' / k)], capture_output=True, text=True, encoding='utf-8', errors='replace')
        if 'successfully pushed' in r.stdout or 'Kernel version' in r.stdout:
            pending.remove(k); running.append(k); print(f'PUSHED {k} {time.strftime("%H:%M")}', flush=True)
        else:
            break
    time.sleep(90)
print('ALL_DONE ' + ' '.join(f'{k}:{v}' for k, v in done.items()), flush=True)
