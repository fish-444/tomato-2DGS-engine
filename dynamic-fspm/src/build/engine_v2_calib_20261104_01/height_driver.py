"""Height recalibration (CONTRACT Amendment 19) for one arm: python -X utf8 height_driver.py A|R
Median plant height at day 71 over 16 seeds -> 130 mm; stop at |median - 130| <= 2 or after 8 evaluations; plain bisection on s once bracketed.
Arm A starts from the existing evaluations s 5.85 (C4AC1.3271) and s 4.10 (C4ACs4.10); Arm R evaluates s 5.85, then 4.10, then bisects.
Writes HEIGHT_<arm>.csv (path) and prints one line per event."""
import csv, json, subprocess, sys, time
from pathlib import Path
import analyze as A

H = Path(__file__).resolve().parent; PY = [sys.executable, '-X', 'utf8']; ENV = dict(__import__('os').environ, PYTHONIOENCODING='utf-8')
ARM = sys.argv[1]; T, TOL, NMAX = 130.0, 2.0, 8
REUSE = {'A': [('C4AC1.3271', 5.85), ('C4ACs4.10', 4.10)], 'R': []}[ARM]
FIRST = {'A': [], 'R': [5.85, 4.10]}[ARM]


def sh(cmd, cwd=H):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=ENV); return r.stdout + r.stderr


def log(*a): print(time.strftime('%H:%M'), ARM, *a, flush=True)


def name(s): return f'V22{ARM}_s' + f'{s:.4f}'.rstrip('0').rstrip('.')


def evaluate(path):
    R = A.load(); out = []
    for arm, s in path:
        M = A.arm(R, arm)
        if len(M) < 16: out.append((arm, s, None, len(M))); continue
        out.append((arm, s, A.q([m['height_mm'] for m in M.values()], .5), len(M)))
    return out


path = list(REUSE)
_prev = H / f'HEIGHT_{ARM}.csv'
if _prev.exists():   # resume after an interruption: keep the evaluations already made (same order)
    for r in csv.DictReader(open(_prev, encoding='utf-8')):
        if (r['arm'], float(r['s'])) not in path and r['height_median_mm'] not in ('', 'None'): path.append((r['arm'], float(r['s'])))
while True:
    ev = evaluate(path)
    with open(H / f'HEIGHT_{ARM}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['step', 'arm', 's', 'height_median_mm', 'n'])
        for i, e in enumerate(ev, 1): w.writerow([i, *e])
    done = [e for e in ev if e[2] is not None]
    if done and abs(done[-1][2] - T) <= TOL: log('STOP within tolerance', done[-1]); break
    if len(path) >= NMAX: log('STOP 8 evaluations', min(done, key=lambda e: abs(e[2] - T))); break
    if len(path) < len(REUSE) + len(FIRST):
        s = FIRST[len(path) - len(REUSE)]
    else:
        lo = [e for e in done if e[2] < T]; hi = [e for e in done if e[2] > T]
        if not (lo and hi): log('STOP not bracketed', done); break
        a = max(lo, key=lambda e: e[1]); b = min(hi, key=lambda e: e[1]); s = round((a[1] + b[1]) / 2, 4)
    arm = name(s); batch = f'H{ARM}{len(path) + 1}'; path.append((arm, s)); log('next', arm, [(e[1], None if e[2] is None else round(e[2], 1)) for e in done])
    sh(PY + ['prepare_v2.py', batch, arm])
    for _try in range(5):   # 2026-11-06 Arm R: one empty make_kernels read crashed the driver; retry
        try:
            ks = json.loads(sh(PY + ['make_kernels.py', batch, '4']).strip().splitlines()[-1].replace("'", '"')); break
        except Exception as e:
            log('make_kernels retry', type(e).__name__); time.sleep(20)
    for _ in range(3):
        if 'being created' in sh(['kaggle', 'datasets', 'create', '-p', '.'], cwd=H / 'dataset' / batch): break
        time.sleep(30)
    while 'ready' not in sh(['kaggle', 'datasets', 'status', f'dolgfish/engv2-{batch.lower()}-20261104']): time.sleep(20)
    o = sh(PY + ['watch_push.py'] + list(ks)); log('ran', arm, ' | '.join(l[:60] for l in o.splitlines() if l.startswith('DONE')))
log('finished')
