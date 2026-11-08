"""Start K15c when one of K17 / K19 has finished, K15d when both have (keeps this session at <= 2 Kaggle kernels). Each via push_cap.py with cap 1."""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
LOGS = [(B / 'engine_static_tau_rule2_20261105_01/push_k16_k18.log', 'DOWNLOADED k17'), (B / 'engine_static_leaflet_arrangement_20261105_01/push_k19.log', 'DOWNLOADED k19')]
JOBS = [('kaggle_k15c', 'dolgfish/static-lai-k15c-20261105', 'k15c'), ('kaggle_k15d', 'dolgfish/static-lai-k15d-20261105', 'k15d')]
procs = []
while len(procs) < len(JOBS) or any(p.poll() is None for p in procs):
    free = sum(1 for p, key in LOGS if p.exists() and key in p.read_text(encoding='utf-8', errors='replace'))
    while len(procs) < min(free, len(JOBS)):
        f, slug, name = JOBS[len(procs)]
        procs.append(subprocess.Popen([sys.executable, '-X', 'utf8', str(HERE / 'push_cap.py'), '1', f'{HERE / f}={slug}={name}'], stdout=open(HERE / f'push_{name}.log', 'w'), stderr=subprocess.STDOUT))
        print(time.strftime('%H:%M:%S'), 'started', name, flush=True)
    time.sleep(60)
print('ALL_END', flush=True)
