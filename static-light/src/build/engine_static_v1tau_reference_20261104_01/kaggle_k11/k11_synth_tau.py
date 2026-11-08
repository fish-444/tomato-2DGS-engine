"""K11 (report only; v1_tau CONTRACT section 5, 16:54 answer item 3): the synthetic normal-smoothing check of normal_smoothing_v2 repeated with leaf transmission tau = 0.07
(leaf surfels; stem / rachis tau 0): arms truth, raw, rec_r2 (receiver only), smooth_all_r2. Same generator, seed 20261105, 44 deg independent-noise level (sigma 62.5 deg from the
v2 calibration), same plants and noise draws as the v1 (tau = 0) synthetic run. It does not re-make the declared decision."""
import glob, hashlib, json, math, os, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
ND = Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT)); sys.path.insert(0, str(ND))
import numpy as np
from scipy.stats import spearmanr
import normals as NM
from static import tau_runs as TR, obs
from static.tau_obs import build_scene, trace_plants_tau
from static.tracer import DISC

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), indent=1, default=float)
F = TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; SEEDS = TR.SEEDS; EPS = CFG['eps']; DEX = 0.0024; TAU = CFG['tau_leaf']
C2 = json.load(open(ND / 'normals_v2_config.json')); SEED = C2['synthetic_seed']; ALL = C2['synthetic']; SIGMA_44 = 62.5; RADIUS = 2.0
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}; HASH['normals.py'] = hashlib.sha256((ND / 'normals.py').read_bytes()).hexdigest()


def spear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return None if (len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0])) else float(spearmanr(x, y)[0])


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
    """Identical to the v2 synthetic generator (k3_synth_v2.py)."""
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
        self.tau = np.where(self.sets > 0, TAU, 0.0)


class SP:
    def __init__(self, base, n_occ, tu_occ, tv_occ, n_rec):
        self.pid = 'syn'; self.centre = base.centre; self.normal = n_rec; self.own_local = np.arange(len(base.A)); self.delta = DEX; self.tau = base.tau
        P = np.zeros((len(base.A), 14)); P[:, 0:3] = base.centre; P[:, 3:6] = tu_occ; P[:, 6:9] = tv_occ; P[:, 9:12] = n_occ; P[:, 12] = base.a; P[:, 13] = base.b
        self._P = P; self._k = np.full(len(base.A), DISC, np.int8)
    def prims(self):
        return self._P, self._k


def run(base, n_occ, tu_occ, tv_occ, n_rec, use_v1=False):
    sp = SP(base, n_occ, tu_occ, tv_occ, n_rec); built = build_scene([sp]); ab, abu = [], []
    for sd in SEEDS:
        r = trace_plants_tau([sp], [sp], sd, EPS, 0.0, 0.0, DEX, F, tau_scale=1.0, use_v1=use_v1, built=built)['syn']
        K = obs.kernel(r['parts']); Ku = obs.kernel(r['parts'], shaded=False); ab.append(r['Uext'] * K * base.A); abu.append(r['Uext'] * Ku * base.A)
    tot = float(np.mean([x.sum() for x in ab])); totu = float(np.mean([x.sum() for x in abu]))
    return dict(total=tot, self=1 - tot / totu, leaves={base.uids[k]: float(np.mean([x[base.leaf == k].sum() for x in ab])) for k in base.uids})


pool = np.load(ND / 'pool.npz'); POOL = dict(a=pool['a'], b=pool['b'], op=pool['op']); rho = float(pool['density']); sig_off = float(pool['sigma_off'])
rows, leaf_rows = [], []
for i, pid in enumerate(ALL):
    t0 = time.time(); b = SynthBase(pid, load_obj(ND / f'mtg__{pid}.obj'), rho, POOL, sig_off, np.random.default_rng(SEED + i)); w = b.A * b.op
    truth = {m: run(b, b.n_true, b.tu_true, b.tv_true, b.n_true, use_v1=(m == 'v1')) for m in ('v1', 'tau')}
    r = np.random.default_rng(SEED + 1000 + 10 * i + 0); z = r.normal(size=len(b.A)); ax = r.normal(size=(len(b.A), 3))        # j = 0 : the 44 deg target, as in k3_synth_v2
    nN, nTU, nTV = NM.rotate_frames(b.n_true, b.tu_true, b.tv_true, np.abs(z) * np.radians(SIGMA_44), ax); met = NM.noise_metric(b.X, nN)[0]
    Ns, kept, _ = NM.smooth(b.X, nN, w, b.sets, RADIUS); TUs, TVs = NM.reframe(Ns, nTU)
    for model in ('v1', 'tau'):
        arms = {'raw': run(b, nN, nTU, nTV, nN, model == 'v1'), 'rec_r2': run(b, nN, nTU, nTV, Ns, model == 'v1'), 'smooth_all_r2': run(b, Ns, TUs, TVs, Ns, model == 'v1')}
        for arm, rr in arms.items():
            tr = truth[model]; errs = []
            for u, tv_ in tr['leaves'].items():
                if tv_ > 0:
                    e = 100 * (rr['leaves'][u] / tv_ - 1); errs.append(abs(e)); leaf_rows.append(dict(plant=pid, model=model, arm=arm, leaf=u, err_pct=e))
            ks = sorted(tr['leaves'])
            rows.append(dict(plant=pid, model=model, arm=arm, noise_metric_deg=met, plant_bias_pct=100 * (rr['total'] / tr['total'] - 1), self_change=rr['self'] - tr['self'],
                             leaf_median_abs_err_pct=float(np.median(errs)), leaf_spearman=spear([tr['leaves'][k] for k in ks], [rr['leaves'][k] for k in ks]), n=len(b.A)))
        log(f'{pid} {model}: ' + ', '.join(f"{x['arm']} {x['plant_bias_pct']:+.1f}%/{x['leaf_median_abs_err_pct']:.1f}" for x in rows[-3:]))
    jdump(dict(rows=rows, leaf_rows=leaf_rows, engine_hash=HASH), OUT / 'synth_tau.json')
jdump(dict(rows=rows, leaf_rows=leaf_rows, engine_hash=HASH, wall_seconds=round(time.time() - T0)), OUT / 'synth_tau.json'); log('K11_DONE')
