"""One GUI-canopy check run on Kaggle CPU headless (build/engine_gui_canopy_20261106_01/CONTRACT.md §4).
python gui_case.py <case dir> <core.jar> <out dir> <evcode dir>
Runs the derived Model.rgg through the frozen bridge with the v2 / ideotype validator installed (as the ideotype runs). Keeps, whatever the
validator says (scenario settings move plants / change the lamp, which the replay may refuse): daily.csv, sensors.csv, completion.json file
hashes, the bridge status and the wall time per engine day. Working files are deleted afterwards."""
import json, shutil, sys, time, traceback
from pathlib import Path

case_src, core, out, evcode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
name = case_src.name; dst = out / name; dst.mkdir(parents=True, exist_ok=True)
status = dict(case=name, attempts=[], started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gui_test=json.load(open(case_src / 'gui_test.json')))
cfg = json.load(open(case_src / 'v2_config.json'))
sys.path.insert(0, str(evcode))
if 'n_plants' in cfg:
    import ideo_validate; status['validator'] = str(ideo_validate.install(cfg))[:400]
else:
    import v2_validate; status['validator'] = str(v2_validate.install(cfg))[:400]
from tools.vtc_growth_bridge import run

for attempt in (1, 2):
    work = Path('/tmp/runs') / f'{name}_a{attempt}_{int(time.time())}'; t0 = time.time()
    try:
        shutil.copytree(case_src, work); r = run(work / 'native', core=core, timeout_s=10 * 3600)
        rec = dict(attempt=attempt, status=r.get('status'), error=str(r.get('error'))[:2000], elapsed_s=round(time.time() - t0, 1))
    except Exception as e:
        rec = dict(attempt=attempt, status='exception', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-3000:], elapsed_s=round(time.time() - t0, 1))
    W = work / 'native' / 'working'
    rec['daily_csv'] = (W / 'daily.csv').exists()
    status['attempts'].append(rec); print(name, json.dumps({k: v for k, v in rec.items() if k != 'trace'}), flush=True)
    for fn in ('daily.csv', 'sensors.csv', 'gui_sensors.csv'):
        if (W / fn).exists(): shutil.copy2(W / fn, dst / fn)
    try:
        status['completion_files'] = json.load(open(W / 'completion.json'))['files']
    except Exception as e:
        status['completion_files'] = f'unavailable: {type(e).__name__}'
    for fn in ('groimp.log', 'execution.json'):
        if (work / 'native' / fn).exists():
            txt = (work / 'native' / fn).read_text(encoding='utf-8', errors='replace'); (dst / fn).write_text(txt[-20000:], encoding='utf-8')
    started_ok = rec['daily_csv'] or rec['status'] == 'verified'
    shutil.rmtree(work, ignore_errors=True)
    if started_ok: break
status['status'] = status['attempts'][-1]['status']; status['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(status, open(dst / 'status.json', 'w'), indent=1); print(name, status['status'], flush=True)
