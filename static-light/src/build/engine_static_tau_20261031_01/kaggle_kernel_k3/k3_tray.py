"""v1_tau kernel 3 (Kaggle CPU): tray scenario pairs (60 mm baseline vs lamp 250/350, lateral -50/+50, spacing 90/120; assumed layout
n_04, t2_02, n_01, t2_05), v1 vs tau 0.07, d_ex 2.4, seeds 101/202. stable / optics_sensitive per declared rule. Pushed only after kernel 1's gates passed."""
import glob, hashlib, json, os, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT))
from static import tau_runs as TR

T0 = time.time(); LOG = []
def log(s):
    LOG.append(f'{time.time() - T0:7.0f}s {s}'); print(LOG[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOG))
TR.setup(NINE_ROOT, TAU_ROOT)
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
res = TR.scenarios('tray', log)
res['engine_hash'] = HASH; res['layout'] = 'assumed_layout (n_04 (-,-), t2_02 (+,-), n_01 (-,+), t2_05 (+,+))'; res['wall_seconds'] = time.time() - T0
json.dump(res, open(OUT / 'tray_scenarios_tau.json', 'w'), indent=1, default=float); log('K3_DONE')
