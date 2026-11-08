"""K2 of the normal smoothing test (engine_static_normals_20261104_01/CONTRACT.md §4, §6): disc-vs-triangle (k_dvt path, tau bundle use_v1=True)
and paired scenarios (tau_runs.SCEN, v1 kernel) for raw and every smoothing radius. G0b / G0c regression against the stored results.
Inputs: static-tau-bundle-20261031, static-nine-bundle-20261028, disc-vs-tri-20261101, static-normals-20261104."""
import csv, glob, hashlib, json, math, os, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
DVT = Path(glob.glob('/kaggle/input/**/T_meta.json', recursive=True)[0]).parent
ND = Path(glob.glob('/kaggle/input/**/normals_config.json', recursive=True)[0]).parent


def mirror(src):
    """The dataset is uploaded flat (a__b__c); rebuild the tree as symlinks under OUT/nd."""
    if (src / 'labels').is_dir():
        return src
    dst = OUT / 'nd'
    for p in src.iterdir():
        if p.is_file():
            t = dst.joinpath(*p.name.split('__')); t.parent.mkdir(parents=True, exist_ok=True)
            if not t.exists():
                os.symlink(p, t)
    return dst


ND = mirror(ND)
sys.path.insert(0, str(TAU_ROOT)); sys.path.insert(0, str(ND))
import numpy as np
from static import tau_runs as TR, obs
from static.geometry import read_ply
from static.tau_obs import build_scene
from static.tracer import TRI, coincident
import normals as NM

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    json.dump(o, open(p, 'w'), indent=1, default=float)
TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; SEEDS = TR.SEEDS; MODES = NM.MODES
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
HASH['normals.py'] = hashlib.sha256((ND / 'normals.py').read_bytes()).hexdigest()
RES = dict(tags=['normal_smoothing_test', 'relative_light', 'conditional_on_unvalidated_scale', 'skeleton_labels_ungraded', 'after_checkpoint_v1', 'R3-1b_unaccepted'], engine_hash=HASH)


def sets_of(q):
    """Smoothing label sets for a TR-placed plant (CONTRACT §2) and the weights area (b) x opacity."""
    rows = q.s.index; op = read_ply(obs.ply_path(q.pid))['opacity'][rows].astype(float)
    if q.pid == 'R3-1b':
        leaf = np.where((q.axis == 0) & (q.r8 >= 1), q.r8, -1)
        sets = np.where(leaf >= 1, leaf, np.where(q.axis == 1, -10, np.where(q.axis == 2, -20, -30)))
    else:
        z = np.load(ND / 'labels' / f'{q.pid}.npz'); lr = TR._map(z['rows'], z['label'].astype(int), rows); st = TR._map(z['rows'], z['status'].astype(int), rows)
        sets = np.where((st == 1) & (lr > 0), lr, -30)
    return sets, q.A * op


NCACHE = {}
def set_mode(q, mode):
    if not hasattr(q, 'raw_normal'):
        q.raw_normal = q.normal
    if mode == 'raw':
        q.normal = q.raw_normal; return 1.0
    key = (q.pid, mode)
    if key not in NCACHE:
        sets, w = sets_of(q); N, kept, _ = NM.smooth(q.s.centre * 1000.0, q.raw_normal, w, sets, NM.RADIUS[mode]); NCACHE[key] = (N, float(kept.mean()))
    q.normal = NCACHE[key][0]
    return NCACHE[key][1]


# ------------------------------------------------------------------ disc vs triangle
q = TR.placed('R3-1b'); builtD = build_scene([q])
def read_T(name):
    rows = [r for r in csv.reader(l for l in open(DVT / name, encoding='utf-8') if not l.startswith('#'))][1:]
    tri = np.array([[float(x) for x in r[6:]] for r in rows]).reshape(-1, 3, 3)
    return dict(tri=tri + q.offset, organ=np.array([r[0] for r in rows]), cls=np.array([r[1] for r in rows]), leaf=np.array([int(r[2]) for r in rows]))


class PlacedT:
    """Copy of k_dvt.PlacedT (16 receivers per leaflet triangle, sample seed = trace seed, delta 1e-7, tau not used by v1)."""
    def __init__(self, T, spt, seed):
        self.pid = 'T'; t = T['tri']
        P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0]; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]; self._P = P; self._k = np.full(len(t), TRI, np.int8)
        isl = T['cls'] == 'Leaflet'; self.tau = np.where(isl, CFG['tau_leaf'], 0.0)
        cr = np.cross(P[:, 3:6], P[:, 6:9]); ar = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / np.maximum(2 * ar[:, None], 1e-30)
        idx = np.nonzero(isl)[0]; rng = np.random.default_rng(seed)
        u = rng.random((len(idx), spt)); v = rng.random((len(idx), spt)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
        self.centre = (t[idx, 0][:, None] + u[..., None] * P[idx, None, 3:6] + v[..., None] * P[idx, None, 6:9]).reshape(-1, 3)
        self.normal = np.repeat(nrm[idx], spt, 0); self.A = np.repeat(ar[idx] / spt, spt) * 1e6; self.own_local = np.repeat(idx, spt); self.delta = 1e-7
        self.leafid = np.repeat(T['leaf'][idx], spt)
    def prims(self):
        return self._P, self._k


def D_summary(recs):
    out = {}
    for kind in ('D_all', 'D_blades'):
        per = []
        for sd in SEEDS:
            r = recs[sd]; K = obs.kernel(r['parts']); Ku = obs.kernel(r['parts'], shaded=False); U = r['Uext']; leaf = np.where(q.axis == 0, q.r8, 0)
            sel = np.ones(len(q.A), bool) if kind == 'D_all' else leaf >= 1
            ab = U * K * q.A; abu = U * Ku * q.A
            d = dict(total=ab[sel].sum(), self=1 - ab[sel].sum() / abu[sel].sum())
            for Lf in range(1, 9):
                d[f'leaf{Lf}'] = ab[leaf == Lf].sum()
            per.append(d)
        out[kind] = {k: TR.pm([p[k] for p in per]) for k in per[0]}
    return out


ref = np.load(NINE_ROOT / 'r31b_ref.npz'); refS = json.load(open(ND / 'ref' / 'dvt_summary_v1.json'))
T = read_T('T_triangles.csv'); Tper = []
for sd in SEEDS:
    o = PlacedT(T, 16, sd); built = build_scene([o])
    from static.tau_obs import trace_plants_tau
    r = trace_plants_tau([o], [o], sd, CFG['eps'], 0.0, 0.0, CFG['dex'], CFG['field'], use_v1=True, built=built)['T']
    half = coincident(built[0], o.centre, o.normal, o.own_local); parts = r['parts'].copy(); parts[half] *= 0.5
    Tper.append(float((r['Uext'] * obs.kernel(parts) * o.A).sum()))
Ttot = float(np.mean(Tper)); dvt = dict(T_total=Ttot, T_total_stored=refS['T|v1']['total']['mean'], modes={})
for mode in MODES:
    kept = set_mode(q, mode)
    recs = TR.trace([q], [q], 'v1', builtD)['R3-1b']; S = D_summary(recs)
    if mode == 'raw':
        diff = float(np.abs(TR.kmean(recs) - ref['relative_light']).max())
        st = refS['T|v1']['total']['mean'] / refS['D_blades|v1']['total']['mean']; now = Ttot / S['D_blades']['total']['mean']
        dvt['G0b'] = dict(D_max_abs_diff_vs_r31b_ref=diff, T_over_D_blades_stored=st, T_over_D_blades_now=now, rel=now / st - 1, passed=bool(diff == 0.0 and abs(now / st - 1) <= 1e-9))
        log(f'G0b {dvt["G0b"]}')
        if not dvt['G0b']['passed']:
            RES['disc_vs_tri'] = dvt; jdump(RES, OUT / 'k2_results.json'); jdump(dict(G0b_failed=dvt['G0b']), OUT / 'G0b_FAILED.json'); sys.exit(0)
    dvt['modes'][mode] = dict(kept_raw_share=kept, D_blades=S['D_blades'], D_all=S['D_all'], T_over_D_blades=Ttot / S['D_blades']['total']['mean'],
                              T_over_D_all=Ttot / S['D_all']['total']['mean'])
    log(f'dvt {mode} T/D_blades={dvt["modes"][mode]["T_over_D_blades"]:.4f} D_blades={S["D_blades"]["total"]["mean"]:.2f} kept={kept:.4f}')
set_mode(q, 'raw')
RES['disc_vs_tri'] = dvt; jdump(RES, OUT / 'k2_results.json')

# ------------------------------------------------------------------ scenarios (v1 kernel), raw + every mode
def scen(kind):
    cfgs = [('baseline', {})] + [(n, s) for n, s in TR.SCEN.items() if kind == 'tray' or 'pitch' not in s]
    cache, res = {}, {m: {} for m in MODES}
    for name, sp in cfgs:
        pitch = sp.get('pitch', 0.060)
        if kind == 'tray':
            if pitch not in cache:
                qs = [TR.placed(pid, pos) for pid, pos in zip(TR.TRAY, TR.tray_positions(pitch))]; cache[pitch] = (qs, build_scene(qs))
            qs, built = cache[pitch]
        else:
            if 'r' not in cache:
                cache['r'] = ([q], builtD)
            qs, built = cache['r']
        for mode in MODES:
            t = time.time()
            for qq in qs:
                set_mode(qq, mode)
            r = TR.trace(qs, qs, 'v1', built, sp.get('dz', 0.0), sp.get('dy', 0.0)); vals = {}
            for qq in qs:
                m = TR.metrics(qq, r[qq.pid]); vals[qq.pid] = m['total']['mean']
                for k in qq.masks:
                    vals[f'{qq.pid}:{k}'] = m[f'leaf:{k}']['mean']
            if kind == 'tray':
                vals['tray_total'] = sum(vals[p] for p in TR.TRAY)
            res[mode][name] = vals
            for qq in qs:
                set_mode(qq, 'raw')
            log(f'scenario {kind} {name} {mode} {time.time() - t:.0f} s')
    return res


refR = json.load(open(ND / 'ref' / 'r31b_scenarios_tau.json')); refT = json.load(open(ND / 'ref' / 'tray_scenarios_tau.json'))
def stored_v1(js):
    return js['raw']['v1'] if 'raw' in js else next(v['raw']['v1'] for v in js.values() if isinstance(v, dict) and 'raw' in v)
SC = {}
for kind, refjs in (('r31b', refR), ('tray', refT)):
    res = scen(kind); sv = stored_v1(refjs); g = []
    for name, vals in res['raw'].items():
        for unit, v in vals.items():
            if name in sv and unit in sv[name]:
                g.append(abs(v / sv[name][unit] - 1))
    SC[kind] = dict(values=res, G0c=dict(max_rel_diff_vs_stored=max(g) if g else None, n=len(g), passed=bool(g and max(g) <= 1e-9)))
    log(f'G0c {kind} {SC[kind]["G0c"]}')
    RES['scenarios'] = SC; jdump(RES, OUT / 'k2_results.json')
RES['seconds'] = round(time.time() - T0); jdump(RES, OUT / 'k2_results.json')
log('K2_DONE')
