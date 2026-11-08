"""Capped Kaggle queue: at most CAP of these kernels running at once, priority order kept; waits while any Engine v2 (ev-*) kernel is QUEUED.
usage: push_cap.py CAP abs_folder=slug=name ...   outputs to <folder>/../out_<name>"""
import re, subprocess, sys, time
from pathlib import Path
CAP = int(sys.argv[1]); Q = [a.rsplit('=', 2) for a in sys.argv[2:]]; pushed, done = {}, {}; end = time.time() + 10 * 3600


def status(slug):
    r = subprocess.run(['kaggle', 'kernels', 'status', slug], capture_output=True, text=True); m = re.search(r'KernelWorkerStatus\.(\w+)', r.stdout + r.stderr)
    return m.group(1) if m else 'unknown'


def ev_queued():
    r = subprocess.run(['kaggle', 'kernels', 'list', '--mine', '--page-size', '20'], capture_output=True, text=True)
    evs = [l.split()[0] for l in r.stdout.splitlines() if l.startswith('dolgfish/ev-')][:6]
    return any(status(s) == 'QUEUED' for s in evs)


while time.time() < end and len(done) < len(Q):
    for folder, slug, name in Q:
        if name in pushed and name not in done:
            st = status(slug)
            if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
                out = Path(folder).parent / f'out_{name}'; out.mkdir(exist_ok=True)
                for _ in range(4):
                    subprocess.run(['kaggle', 'kernels', 'output', slug, '-p', str(out), '-o'], capture_output=True, text=True)
                    if (out / 'SUMMARY.json').exists() or st != 'COMPLETE':
                        break
                    time.sleep(30)
                done[name] = st; print(time.strftime('%H:%M:%S'), 'DOWNLOADED', name, st, flush=True)
    running = len(pushed) - len(done)
    nxt = [q for q in Q if q[2] not in pushed]
    if nxt and running < CAP and not ev_queued():
        folder, slug, name = nxt[0]
        r = subprocess.run(['kaggle', 'kernels', 'push', '-p', folder], capture_output=True, text=True); txt = (r.stdout + r.stderr).strip()
        ok = 'successfully pushed' in txt; print(time.strftime('%H:%M:%S'), name, 'push', 'OK' if ok else txt.splitlines()[-1][:160], flush=True)
        if ok:
            pushed[name] = slug; continue
    time.sleep(120)
print('ALL_END', done, flush=True)
