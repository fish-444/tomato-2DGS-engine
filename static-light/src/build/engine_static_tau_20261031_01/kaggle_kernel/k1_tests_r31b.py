"""v1_tau kernel 1 (Kaggle CPU): R0 regression + reproduction gate, T1 analytic, probe-tool test, T2 duplication invariance, then
(only if all pass) R3-1b runs (v1 vs tau 0.05/0.07/0.10) and R3-1b lamp scenarios. Outputs in /kaggle/working. No secrets used."""
import glob, hashlib, json, math, os, platform, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT))
import numpy as np, numba
from static import obs, light as L, tau_runs as TR
from static.geometry import footprint_k
from static.tracer import Scene, DISC, transmit_rays
from static.tau_obs import build_scene, probe_ratios

T0 = time.time(); LOG = []
def log(s):
    LOG.append(f'{time.time() - T0:7.0f}s {s}'); print(LOG[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOG))
F = TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
meta = dict(engine_hash=HASH, python=platform.python_version(), numpy=np.__version__, numba=numba.__version__, cpus=os.cpu_count(), eps_m=CFG['eps'], tau_leaf=CFG['tau_leaf'], dex_m=CFG['dex'])
tests = dict(meta=meta)
def save():
    json.dump(tests, open(OUT / 'tests.json', 'w'), indent=1, default=float)

# ---------------- R0: tau=0 vs untouched v1 kernel, byte-identical; reproduction vs the local v1 arrays
q = TR.placed('R3-1b'); built = build_scene([q])
t = time.time(); rv1 = TR.trace([q], [q], 'v1', built)['R3-1b']; tv1 = time.time() - t
t = time.time(); rt0 = TR.trace([q], [q], 'tau0', built)['R3-1b']; tt0 = time.time() - t
same = all(np.array_equal(rv1[sd]['parts'], rt0[sd]['parts']) for sd in TR.SEEDS)
ref = np.load(NINE_ROOT / 'r31b_ref.npz')
kv1 = TR.kmean(rv1)
rep = dict(rows_equal=bool(np.array_equal(ref['ply_row'], q.s.index)), max_abs_diff_vs_local_v1=float(np.abs(kv1 - ref['relative_light']).max()))
tests['R0'] = dict(byte_identical_tau0_vs_v1=bool(same), reproduction_vs_local=rep, seconds_v1=tv1, seconds_tau0=tt0,
                   pass_=bool(same and rep['rows_equal'] and rep['max_abs_diff_vs_local_v1'] == 0.0))
log(f"R0 {tests['R0']}"); save()

# ---------------- T1 analytic: two disc-lattice squares, vertical beam
k1 = float(footprint_k(np.ones(1))[0]); sp = 0.0004; side = 0.020; g = -side / 2 + sp / 2 + sp * np.arange(int(round(side / sp))); rad = sp / math.sqrt(2)
def square(x0, z):
    cx, cy = np.meshgrid(g + x0, g); c = np.stack([cx.ravel(), cy.ravel(), np.full(cx.size, z)], 1); n = len(c)
    P = np.zeros((n, 14)); P[:, 0:3] = c; P[:, 3:6] = [1, 0, 0]; P[:, 6:9] = [0, 1, 0]; P[:, 9:12] = [0, 0, 1]; P[:, 12] = rad; P[:, 13] = rad
    return P
eps = CFG['eps']; tau = CFG['tau_leaf']; z0 = 0.10
cases = {'gap_0': ([0.0], 1.0), 'gap_0.5eps': ([0.5 * eps], 1.0), 'gap_2eps': ([2 * eps], tau), 'gap_5mm': ([0.005], tau), 'T1e_two_layers_1mm_gap_5mm': ([0.005, 0.006], tau)}
t1 = {}
for name, (gaps, expect) in cases.items():
    lower = square(0.0, z0); uppers = [square(0.010, z0 + gp) for gp in gaps]
    Pall = np.concatenate([lower] + uppers); sc = Scene(Pall, np.full(len(Pall), DISC, np.int8), h=0.002)
    R = lower[:, 0:3]; D = np.tile([0, 0, 1.0], (len(R), 1)); own = np.arange(len(R))
    T = transmit_rays(sc, R, D, own, CFG['dex'], 1.0, np.full(len(Pall), tau), eps)
    xo = R[:, 0]; ov = (xo > 0.0005) & (xo < 0.0095) & (np.abs(R[:, 1]) < 0.0095); nov = (xo < -0.0005) & (xo > -0.0095) & (np.abs(R[:, 1]) < 0.0095)
    mo, mn = float(T[ov].mean()), float(T[nov].mean())
    t1[name] = dict(expected_overlap=expect, mean_T_overlap=mo, rel_err_overlap=abs(mo - expect) / expect, mean_T_nonoverlap=mn, rel_err_nonoverlap=abs(mn - 1.0),
                    pass_=bool(abs(mo - expect) / expect <= 0.02 and abs(mn - 1.0) <= 0.02))
tests['T1'] = dict(cases=t1, pass_=all(v['pass_'] for v in t1.values())); log(f"T1 {json.dumps({k: (round(v['mean_T_overlap'], 4), v['pass_']) for k, v in t1.items()})}"); save()

# ---------------- probe-tool test: single sheet, near-vertical single-point beam, no diffuse
sheet = square(0.0, z0); sc = Scene(sheet, np.full(len(sheet), DISC, np.int8), h=0.002)
pts = np.array([[x, y, 0.0] for x in (-0.004, 0.0, 0.004) for y in (-0.004, 0.0, 0.004)])
pr = probe_ratios(sc, np.full(len(sheet), tau), eps, pts + [0, 0, z0 + 0.002], pts + [0, 0, z0 - 0.002], np.array([[0.0, 0.0, 1000.0]]), 1.0, np.array([[0.0, 0.0, 1.0]]), 0.0)
rv, rt = float(np.nanmean(pr['v1']['ratio'])), float(np.nanmean(pr['v1_tau']['ratio']))
tests['probe_test'] = dict(ratio_v1=rv, ratio_tau=rt, expected_v1=0.0, expected_tau=tau, pass_=bool(rv == 0.0 and abs(rt - tau) / tau <= 0.02)); log(f"probe {tests['probe_test']}"); save()

# ---------------- T2 duplication invariance (R3-1b)
class Synth:
    def __init__(self, pid, P, tau_, A, masks, delta=None):
        self.pid = pid; self._P = P; self.centre = P[:, 0:3]; self.normal = P[:, 9:12]; self.tau = tau_; self.A = A; self.masks = masks
        self.own_local = np.arange(len(P)); self.delta = delta; self.h = np.zeros(len(P))
    def prims(self):
        return self._P, np.full(len(self._P), DISC, np.int8)
P0, _ = q.prims(); rng = np.random.default_rng(20261031)
def renorm(A_new, key_new, key_old, A_old):
    out = A_new.copy()
    for kk in np.unique(key_old):
        mo = key_old == kk; mn = key_new == kk
        if mn.any():
            out[mn] *= A_old[mo].sum() / A_new[mn].sum()
    return out
dupP = np.concatenate([P0, P0.copy()]); dupP[len(P0):, 0:3] += rng.normal(0, 0.0002, (len(P0), 3))
dup = Synth('dup', dupP, np.r_[q.tau, q.tau], np.r_[q.A, q.A] / 2, {k: np.r_[m, m] for k, m in q.masks.items()})
keep = rng.random(len(P0)) < 0.5
thin = Synth('thin', P0[keep], q.tau[keep], renorm(q.A[keep], q.group_key[keep], q.group_key, q.A), {k: m[keep] for k, m in q.masks.items()})
t2 = {}
for model in ('tau0.07', 'v1'):
    if model == 'v1':
        base = TR.metrics(q, rv1)
    else:
        r_base_tau = TR.trace([q], [q], 'tau0.07', built)['R3-1b']; base = TR.metrics(q, r_base_tau)
    row = {}
    for v in (dup, thin):
        tt = time.time()
        rr = TR.trace([v], [v], model)[v.pid]
        A = v.A; vals = {}
        tot = np.mean([(rr[sd]['Uext'] * obs.kernel(rr[sd]['parts']) * A).sum() for sd in TR.SEEDS])
        vals['plant_total'] = tot / base['total']['mean'] - 1
        for kname, m in v.masks.items():
            lv = np.mean([(rr[sd]['Uext'] * obs.kernel(rr[sd]['parts']) * A)[m].sum() for sd in TR.SEEDS])
            vals[kname] = lv / base[f'leaf:{kname}']['mean'] - 1
        row[v.pid] = dict(rel_change=vals, max_abs=max(abs(x) for x in vals.values()), seconds=time.time() - tt)
        log(f"T2 {model} {v.pid} max |change| {row[v.pid]['max_abs']:.4f}")
    t2[model] = row
tests['T2'] = dict(results=t2, pass_=bool(all(t2['tau0.07'][v]['max_abs'] < 0.02 for v in ('dup', 'thin'))), rule='v1_tau plant total and every leaf < 2 %; v1 reported')
tests['gates_pass'] = bool(tests['R0']['pass_'] and tests['T1']['pass_'] and tests['T2']['pass_'] and tests['probe_test']['pass_'])
log(f"GATES {tests['gates_pass']}"); save()
if not tests['gates_pass']:
    log('STOP: a gate failed; no plant result'); sys.exit(0)

# ---------------- R3-1b runs: v1 (= tau0 rv1) and tau 0.05 / 0.07 / 0.10
runs = {'v1': rv1, 'tau0.07': r_base_tau}
for model in ('tau0.05', 'tau0.10'):
    t = time.time(); runs[model] = TR.trace([q], [q], model, built)['R3-1b']; log(f'R3-1b {model} {time.time() - t:.0f} s')
plant = {m: TR.metrics(q, r) for m, r in runs.items()}
(OUT / 'R3-1b').mkdir(exist_ok=True)
json.dump(dict(plant=plant, labels=['light_model=v1_tau', 'assumed_leaf_optics (tau=0.07, a=0.85, reflection dropped)', 'relative_light', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1']),
          open(OUT / 'R3-1b' / 'plant_tau.json', 'w'), indent=1, default=float)
Kv, Kt = TR.kmean(runs['v1']), TR.kmean(runs['tau0.07']); U = np.mean([runs['tau0.07'][sd]['Uext'] for sd in TR.SEEDS], 0)
np.savez_compressed(OUT / 'R3-1b' / 'per_surfel_light_tau.npz', ply_row=q.s.index, tau=q.tau, a=np.full(len(q.A), CFG['a_main']), area_b_silhouette_mm2=q.A, h_above_base_mm=q.h,
                    relative_light_v1=Kv, relative_light_tau=Kt, U_open_ru=U, absorbed_v1_ru_mm2=U * Kv * q.A, absorbed_tau_a1_ru_mm2=U * Kt * q.A,
                    absorbed_tau_a085_ru_mm2=U * Kt * q.A * CFG['a_main'], axis_kind=q.axis, leaf_id_R8pc=q.r8, leaf_id_vpcleanup=q.vp,
                    overflow_rays=np.sum([runs['tau0.07'][sd]['novf'] for sd in TR.SEEDS], 0))
log('R3-1b runs saved')
sc_r = TR.scenarios('r31b', log); json.dump(sc_r, open(OUT / 'R3-1b' / 'scenarios_tau.json', 'w'), indent=1, default=float)
tests['wall_seconds'] = time.time() - T0; save(); log('K1_DONE')
