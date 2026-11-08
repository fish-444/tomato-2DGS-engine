"""v1_tau kernel 2 (Kaggle CPU): the nine individuals, static job order. Per individual: R0 byte-identity (tau=0 vs v1), v1 and
tau 0.05 / 0.07 / 0.10 traces, metrics and per-surfel file. Checkpoint unit = individual (<id>/DONE); a re-push with ONLY=... resumes.
Pushed only after kernel 1's gates (R0, T1, T2, probe test) passed."""
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

ORDER = ['t2_02', 'n_04', 'n_01', 't2_05', 'n_06', 'n_05', 't2_04', 'n_02', 'n_03']
ONLY = os.environ.get('ONLY', '')
T0 = time.time(); LOG = []
def log(s):
    LOG.append(f'{time.time() - T0:7.0f}s {s}'); print(LOG[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOG))
TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
summary = dict(engine_hash=HASH, individuals={})
for pid in ([x for x in ONLY.split(',') if x] or ORDER):
    t0 = time.time(); q = TR.placed(pid); built = build_scene([q])
    runs = {'v1': TR.trace([q], [q], 'v1', built)[pid]}
    r0 = TR.trace([q], [q], 'tau0', built)[pid]
    same = all(np.array_equal(runs['v1'][sd]['parts'], r0[sd]['parts']) for sd in TR.SEEDS)
    if not same:
        summary['individuals'][pid] = dict(status='R0_failed'); log(f'{pid} R0 FAILED, skipped'); continue
    for model in ('tau0.07', 'tau0.05', 'tau0.10'):
        tt = time.time(); runs[model] = TR.trace([q], [q], model, built)[pid]; log(f'{pid} {model} {time.time() - tt:.0f} s')
    plant = {m: TR.metrics(q, r) for m, r in runs.items()}
    (OUT / pid).mkdir(exist_ok=True)
    json.dump(dict(plant=plant, R0_byte_identical=True, surfels=len(q.A), label_note='no leaf labels (Stage 1 waiting); n_04 clicked leaf-1 set = lower bound',
                   labels=['light_model=v1_tau', 'assumed_leaf_optics (tau=0.07, a=0.85, reflection dropped)', 'relative_light', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1']),
              open(OUT / pid / 'plant_tau.json', 'w'), indent=1, default=float)
    Kv, Kt = TR.kmean(runs['v1']), TR.kmean(runs['tau0.07']); U = np.mean([runs['tau0.07'][sd]['Uext'] for sd in TR.SEEDS], 0)
    np.savez_compressed(OUT / pid / 'per_surfel_light_tau.npz', ply_row=q.s.index, tau=q.tau, a=np.full(len(q.A), CFG['a_main']), area_b_silhouette_mm2=q.A, h_above_base_mm=q.h,
                        relative_light_v1=Kv, relative_light_tau=Kt, U_open_ru=U, absorbed_v1_ru_mm2=U * Kv * q.A, absorbed_tau_a1_ru_mm2=U * Kt * q.A,
                        absorbed_tau_a085_ru_mm2=U * Kt * q.A * CFG['a_main'], group_id=q.group, axis_like=q.axis_like.astype(np.int8),
                        overflow_rays=np.sum([runs['tau0.07'][sd]['novf'] for sd in TR.SEEDS], 0))
    (OUT / pid / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    summary['individuals'][pid] = dict(status='done', seconds=time.time() - t0); log(f'{pid} done {time.time() - t0:.0f} s')
    json.dump(summary, open(OUT / 'SUMMARY.json', 'w'), indent=1, default=float)
summary['wall_seconds'] = time.time() - T0
json.dump(summary, open(OUT / 'SUMMARY.json', 'w'), indent=1, default=float); log('K2_DONE')
