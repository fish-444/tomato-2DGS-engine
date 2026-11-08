"""v1_tau kernel 1b (Kaggle CPU), after the labelled second attempt T2b passed (CONTRACT Amendment 1): R3-1b R0 re-check, then
v1 vs tau 0.05 / 0.07 / 0.10 runs, per-surfel file, and R3-1b lamp scenarios (v1 vs tau 0.07)."""
import glob, hashlib, json, os, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT))
import numpy as np
from static import tau_runs as TR
from static.tau_obs import build_scene

T0 = time.time(); LOG = []
def log(s):
    LOG.append(f'{time.time() - T0:7.0f}s {s}'); print(LOG[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOG))
TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
q = TR.placed('R3-1b'); built = build_scene([q])
runs = {'v1': TR.trace([q], [q], 'v1', built)['R3-1b']}
r0 = TR.trace([q], [q], 'tau0', built)['R3-1b']
same = all(np.array_equal(runs['v1'][sd]['parts'], r0[sd]['parts']) for sd in TR.SEEDS)
ref = np.load(NINE_ROOT / 'r31b_ref.npz'); diff = float(np.abs(TR.kmean(runs['v1']) - ref['relative_light']).max())
log(f'R0 byte_identical={same} max_diff_vs_local_v1={diff}')
if not (same and diff == 0.0):
    json.dump(dict(R0_failed=True), open(OUT / 'R0_failed.json', 'w')); sys.exit(0)
for model in ('tau0.07', 'tau0.05', 'tau0.10'):
    t = time.time(); runs[model] = TR.trace([q], [q], model, built)['R3-1b']; log(f'R3-1b {model} {time.time() - t:.0f} s')
plant = {m: TR.metrics(q, r) for m, r in runs.items()}
(OUT / 'R3-1b').mkdir(exist_ok=True)
json.dump(dict(plant=plant, R0_byte_identical=True, engine_hash=HASH,
               labels=['light_model=v1_tau', 'assumed_leaf_optics (tau=0.07, a=0.85, reflection dropped)', 'relative_light', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1']),
          open(OUT / 'R3-1b' / 'plant_tau.json', 'w'), indent=1, default=float)
Kv, Kt = TR.kmean(runs['v1']), TR.kmean(runs['tau0.07']); U = np.mean([runs['tau0.07'][sd]['Uext'] for sd in TR.SEEDS], 0)
np.savez_compressed(OUT / 'R3-1b' / 'per_surfel_light_tau.npz', ply_row=q.s.index, tau=q.tau, a=np.full(len(q.A), CFG['a_main']), area_b_silhouette_mm2=q.A, h_above_base_mm=q.h,
                    relative_light_v1=Kv, relative_light_tau=Kt, U_open_ru=U, absorbed_v1_ru_mm2=U * Kv * q.A, absorbed_tau_a1_ru_mm2=U * Kt * q.A,
                    absorbed_tau_a085_ru_mm2=U * Kt * q.A * CFG['a_main'], axis_kind=q.axis, leaf_id_R8pc=q.r8, leaf_id_vpcleanup=q.vp,
                    overflow_rays=np.sum([runs['tau0.07'][sd]['novf'] for sd in TR.SEEDS], 0))
log('R3-1b runs saved')
sc_r = TR.scenarios('r31b', log); sc_r['engine_hash'] = HASH
json.dump(sc_r, open(OUT / 'R3-1b' / 'scenarios_tau.json', 'w'), indent=1, default=float); log('K1B_DONE')
