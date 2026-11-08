"""Synthetic check of normal smoothing v2 (engine_static_normals_v2_20261104_01/CONTRACT.md §2): smoothed normals also orient the occluding discs.
Kaggle CPU. v1 static package = byte copy in static-nine-bundle-20261028 (not edited); new code = normals.py (flat dataset static-normals-v2-20261104).
PLANTS is set per kernel (split of the 7 meshes); R3-1b is always built for the sigma calibration. Local smoke: --nine <dir> --nd <dir> --out <dir> --plants n_03"""
import argparse, glob, hashlib, json, math, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--nine'); ap.add_argument('--nd'); ap.add_argument('--out', default='/kaggle/working'); ap.add_argument('--plants', default='R3-1b,n_01,n_02,n_03')
a = ap.parse_args()
NINE = Path(a.nine) if a.nine else Path(glob.glob('/kaggle/input/**/light_consts.json', recursive=True)[0]).parent
ND = Path(a.nd) if a.nd else Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
sys.path.insert(0, str(NINE)); sys.path.insert(0, str(ND))
import numpy as np
from scipy.stats import spearmanr
from static import obs, light as L
from static.tracer import DISC, Scene, coincident
import normals as NM

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True); json.dump(o, open(p, 'w', encoding='utf-8'), indent=1, default=float)

CFG = json.load(open(ND / 'normals_v2_config.json'))
consts = json.load(open(NINE / 'light_consts.json')); L.cos_m = lambda *x, **k: consts['cos_m']
F = obs.FieldFast.__new__(obs.FieldFast); _z = np.load(NINE / 'field_planes.npz')
F.planes = {70.0: (_z['xs_70'], _z['ys_70'], _z['v_70']), 150.0: (_z['xs_150'], _z['ys_150'], _z['v_150'])}
SEEDS = tuple(CFG['seeds']); DEX = CFG['d_ex_mm'] / 1000.0; RADII = CFG['radii_mm']
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((NINE / 'static').glob('*.py'))}
HASH['normals.py'] = hashlib.sha256((ND / 'normals.py').read_bytes()).hexdigest()
PLANTS = [x for x in a.plants.split(',') if x]


def spear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(spearmanr(x, y)[0])


def load_obj(path):
    V, objs, cur = [], {}, None
    for line in open(path, encoding='utf-8'):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('o '):
            cur = line.split(None, 1)[1].strip(); objs.setdefault(cur, [])
        elif line.startswith('f '):
            idx = []
            for t in line.split()[1:]:
                k = int(t.split('/')[0]); idx.append(k - 1 if k > 0 else len(V) + k)
            for j in range(1, len(idx) - 1):
                objs[cur].append((idx[0], idx[j], idx[j + 1]))
    V = np.array(V, float)
    return {o: V[np.array(f)] for o, f in objs.items() if f}


def obj_leaf(name):
    if name == 'stem':
        return None, -10
    if name.startswith('rachis'):
        return None, -20
    if '__' in name and name.startswith('L'):
        return name.split('__')[0], 1
    return None, -30


class SynthBase:
    """Identical to the first test's generator (positions mm, area mm2)."""
    def __init__(self, pid, objs, rho, pool, sig_off, rng):
        C, Nn, Ar, lid, sets = [], [], [], [], []
        luids = sorted({obj_leaf(o)[0] for o in objs if obj_leaf(o)[0]}); self.uids = {k + 1: u for k, u in enumerate(luids)}; inv = {u: k for k, u in self.uids.items()}
        for o in sorted(objs):
            tri = objs[o]; e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]; cr = np.cross(e1, e2); ar = 0.5 * np.linalg.norm(cr, axis=1); ok = ar > 0
            tri, e1, e2, cr, ar = tri[ok], e1[ok], e2[ok], cr[ok], ar[ok]
            if not len(ar):
                continue
            Aobj = ar.sum(); N = max(1, int(round(rho * Aobj)))
            ti = rng.choice(len(ar), N, p=ar / Aobj); u = rng.random(N); v = rng.random(N); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
            nrm = cr[ti] / (2 * ar[ti, None])
            P = tri[ti, 0] + u[:, None] * e1[ti] + v[:, None] * e2[ti] + nrm * rng.normal(0, sig_off, (N, 1))
            C.append(P); Nn.append(nrm); Ar.append(np.full(N, Aobj / N))
            uid, s = obj_leaf(o)
            lid.append(np.full(N, inv[uid] if uid else -1)); sets.append(np.full(N, inv[uid] if uid else s))
        self.pid = pid; self.X = np.concatenate(C); self.n_true = np.concatenate(Nn); self.A = np.concatenate(Ar)
        self.leaf = np.concatenate(lid); self.sets = np.concatenate(sets)
        k = rng.integers(len(pool['a']), size=len(self.X)); self.a = pool['a'][k]; self.b = pool['b'][k]; self.op = pool['op'][k]
        t = rng.normal(size=(len(self.X), 3)); t -= (t * self.n_true).sum(1, keepdims=True) * self.n_true; self.tu_true = t / np.linalg.norm(t, axis=1, keepdims=True)
        self.tv_true = np.cross(self.n_true, self.tu_true)
        cm = self.X / 1000.0; z0 = np.percentile(cm[:, 2], 0.5); m = cm[:, 2] <= z0 + 0.005
        bxy = (cm[m, :2] * self.A[m, None]).sum(0) / self.A[m].sum()
        self.centre = cm + np.array([obs.SCENE_XY[0] - bxy[0], obs.SCENE_XY[1] - bxy[1], 0.0])


class SP:
    """Placed synthetic plant: occluder frame (n, tu, tv) and receiver normal set separately."""
    def __init__(self, base, n_occ, tu_occ, tv_occ, n_rec):
        self.pid = 'syn'; self.pos = obs.SCENE_XY; self.centre = base.centre; self.normal = n_rec; self.area = base.A; self.area_a = base.A
        self.leaf = np.ones(len(base.A), bool); self.leaf_id = base.leaf; self.own_local = np.arange(len(base.A)); self.delta = None; self.coincident_half = False
        P = np.zeros((len(base.A), 14)); P[:, 0:3] = base.centre; P[:, 3:6] = tu_occ; P[:, 6:9] = tv_occ; P[:, 9:12] = n_occ; P[:, 12] = base.a; P[:, 13] = base.b
        self._P = P; self._k = np.full(len(base.A), DISC, np.int8)
    def prims(self):
        return self._P, self._k


def run(base, n_occ, tu_occ, tv_occ, n_rec):
    sp = SP(base, n_occ, tu_occ, tv_occ, n_rec); ab, abu = [], []
    for sd in SEEDS:
        rec = obs.trace_plants([sp], [sp], sd, delta=DEX, sensors=False, field=F)['syn']
        K = obs.kernel(rec['parts']); Ku = obs.kernel(rec['parts'], shaded=False); U = rec['Uext']
        ab.append(U * K * base.A); abu.append(U * Ku * base.A)
    tot = float(np.mean([x.sum() for x in ab])); totu = float(np.mean([x.sum() for x in abu]))
    hits = int(coincident(Scene(sp._P, sp._k, h=0.003), sp.centre, n_rec, sp.own_local).sum())
    return dict(total=tot, self=1 - tot / totu, hits=hits, leaves={base.uids[k]: float(np.mean([x[base.leaf == k].sum() for x in ab])) for k in base.uids})


def ang_to_true(n, n_true):
    d = np.degrees(np.arccos(np.clip(np.abs((n * n_true).sum(1)), 0, 1)))
    return float(np.median(d)), float(np.percentile(d, 90))


pool = np.load(ND / 'pool.npz'); POOL = dict(a=pool['a'], b=pool['b'], op=pool['op']); rho = float(pool['density']) * float(os.environ.get('RHO_SCALE', '1')); sig_off = float(pool['sigma_off'])   # RHO_SCALE only for the local smoke test
SEED = CFG['synthetic_seed']; ALL = CFG['synthetic']; bases = {}
for i, pid in enumerate(ALL):
    if pid in PLANTS or pid == 'R3-1b':
        bases[pid] = SynthBase(pid, load_obj(ND / f'mtg__{pid}.obj'), rho, POOL, sig_off, np.random.default_rng(SEED + i))
        log(f'base {pid} surfels {len(bases[pid].A)} leaves {len(bases[pid].uids)}')

# ---- calibration on the R3-1b mesh sample: independent noise (sigma) and correlated noise (scale), same grid
cb = bases['R3-1b']; rng = np.random.default_rng(SEED); zc = rng.normal(size=len(cb.A)); axc = rng.normal(size=(len(cb.A), 3))
cal = []
for s in CFG['sigma_grid_deg']:
    n2, _, _ = NM.rotate_frames(cb.n_true, cb.tu_true, cb.tv_true, np.abs(zc) * np.radians(s), axc)
    n3, _, _ = NM.correlated_rotate(cb.n_true, cb.tu_true, cb.tv_true, cb.X, s, np.random.default_rng(SEED + 7))
    cal.append(dict(sigma_deg=s, metric_white_deg=NM.noise_metric(cb.X, n2)[0], metric_corr_deg=NM.noise_metric(cb.X, n3)[0]))
sig_white = {}; sig_corr = {}
for tg in CFG['noise_targets_deg']:
    sig_white[tg] = min(cal, key=lambda c: abs(c['metric_white_deg'] - tg))['sigma_deg']; sig_corr[tg] = min(cal, key=lambda c: abs(c['metric_corr_deg'] - tg))['sigma_deg']
jdump(dict(calibration=cal, sigma_white=sig_white, sigma_corr=sig_corr, density=rho, sigma_off=sig_off, engine_hash=HASH, seed=SEED), OUT / 'calibration.json')
log(f'calibration white {sig_white} corr {sig_corr}')

ARMS = ['raw'] + [f'rec_r{r}' for r in RADII] + [f'smooth_all_r{r}' for r in RADII] + [f'occ_only_r{r}' for r in RADII] + ['recv_true']
for pid in PLANTS:
    if (OUT / 'synth' / f'{pid}.json').exists():
        continue
    t0 = time.time(); b = bases[pid]; w = b.A * b.op; i = ALL.index(pid)
    truth = run(b, b.n_true, b.tu_true, b.tv_true, b.n_true); rows, leaf_rows = [], []
    rows.append(dict(plant=pid, variant='truth', noise_target_deg=None, arm='truth', total=truth['total'], self=truth['self'], hits=truth['hits'], n=len(b.A)))
    for variant in ('white', 'corr'):
        for j, tg in enumerate(CFG['noise_targets_deg']):
            if variant == 'corr' and tg != CFG['rule']['noise_deg']:
                continue
            r = np.random.default_rng(SEED + 1000 + 10 * i + j + (500 if variant == 'corr' else 0))
            if variant == 'white':
                z = r.normal(size=len(b.A)); ax = r.normal(size=(len(b.A), 3))
                nN, nTU, nTV = NM.rotate_frames(b.n_true, b.tu_true, b.tv_true, np.abs(z) * np.radians(sig_white[tg]), ax)
            else:
                nN, nTU, nTV = NM.correlated_rotate(b.n_true, b.tu_true, b.tv_true, b.X, sig_corr[tg], r)
            met = NM.noise_metric(b.X, nN)[0]; sm = {}
            for rad in RADII:
                Ns, kept, _ = NM.smooth(b.X, nN, w, b.sets, float(rad)); TUs, TVs = NM.reframe(Ns, nTU); sm[rad] = (Ns, TUs, TVs, kept)
            res = {'raw': run(b, nN, nTU, nTV, nN), 'recv_true': run(b, nN, nTU, nTV, b.n_true)}
            nerr = {'raw': ang_to_true(nN, b.n_true) + (1.0,)}
            for rad in RADII:
                Ns, TUs, TVs, kept = sm[rad]
                res[f'rec_r{rad}'] = run(b, nN, nTU, nTV, Ns); res[f'smooth_all_r{rad}'] = run(b, Ns, TUs, TVs, Ns); res[f'occ_only_r{rad}'] = run(b, Ns, TUs, TVs, nN)
                nerr[f'r{rad}'] = ang_to_true(Ns, b.n_true) + (float(kept.mean()),)
            for arm in ARMS:
                rr = res[arm]; errs = []
                for u, tv_ in truth['leaves'].items():
                    if tv_ > 0:
                        e = 100 * (rr['leaves'][u] / tv_ - 1); errs.append(abs(e))
                        leaf_rows.append(dict(plant=pid, variant=variant, noise_target_deg=tg, arm=arm, leaf=u, truth=tv_, value=rr['leaves'][u], err_pct=e))
                ks = sorted(truth['leaves']); key = 'raw' if arm in ('raw', 'recv_true') else 'r' + arm.split('_r')[-1]
                rows.append(dict(plant=pid, variant=variant, noise_target_deg=tg, sigma_deg=(sig_white if variant == 'white' else sig_corr)[tg], noise_metric_deg=met, arm=arm,
                                 truth_total=truth['total'], total=rr['total'], plant_bias_pct=100 * (rr['total'] / truth['total'] - 1), self_truth=truth['self'], self_arm=rr['self'],
                                 self_change=rr['self'] - truth['self'], leaf_median_abs_err_pct=float(np.median(errs)),
                                 leaf_spearman=spear([truth['leaves'][k] for k in ks], [rr['leaves'][k] for k in ks]), hits=rr['hits'], hits_share=rr['hits'] / len(b.A),
                                 normal_err_median_deg=nerr[key][0] if arm != 'recv_true' else 0.0, normal_err_p90_deg=nerr[key][1] if arm != 'recv_true' else 0.0,
                                 kept_raw_share=nerr[key][2] if arm not in ('raw', 'recv_true') else 1.0, n=len(b.A)))
            log(f'{pid} {variant} {tg}: ' + ', '.join(f"{x['arm']} {x['plant_bias_pct']:+.1f}%/{x['leaf_median_abs_err_pct']:.1f}" for x in rows[-len(ARMS):] if x['arm'] in ('raw', 'rec_r5', 'smooth_all_r2', 'smooth_all_r3', 'smooth_all_r5')))
    jdump(dict(rows=rows, leaf_rows=leaf_rows, seconds=round(time.time() - t0)), OUT / 'synth' / f'{pid}.json')
jdump(dict(engine_hash=HASH, wall_seconds=round(time.time() - T0), plants=PLANTS), OUT / 'SUMMARY.json')
log('K3_DONE')
