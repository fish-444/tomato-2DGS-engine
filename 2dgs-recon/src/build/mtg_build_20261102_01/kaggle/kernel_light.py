"""MTG build (build/mtg_build_20261102_01, CONTRACT §5 static light consistency): our static tracer v1 (frozen settings: black leaves, d_ex 2.4 mm,
seeds 101/202, coplanar half credit, field + 3 bars + dome 0.6; static-tau-bundle-20261031 / static-nine-bundle-20261028, unchanged) on the
engine-format triangles of the MTG plants (template mean, observed plants with areas, 16 check plants x 2 internode arms). Receivers: spt points per
Leaflet triangle; Stem / Petiole triangles occlude only. Same code path as the B-model session's bm-light kernel (PlacedB). Not validation."""
import csv, glob, hashlib, json, os, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
TRI_ROOT = Path(glob.glob('/kaggle/input/**/MTGB_TRI_MANIFEST.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT))
import numpy as np
from static import tau_runs as TR, obs
from static.tau_obs import build_scene, trace_plants_tau
from static.tracer import TRI, coincident

T0 = time.time(); LOG = []


def log(s):
    LOG.append(f'{time.time() - T0:7.0f}s {s}'); print(LOG[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOG))


TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; F = CFG['field']; SEEDS = TR.SEEDS
M = json.load(open(TRI_ROOT / 'MTGB_TRI_MANIFEST.json'))
RES = dict(labels=['mtg_from_traits', 'engine_format_geometry_our_light', 'representation_comparison_not_validation', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1'],
           engine_hash={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}, settings=dict(dex=CFG['dex'], seeds=list(SEEDS), use_v1=True), plants={})


def read_T(p):
    rows = [r for r in csv.reader(l for l in open(p, encoding='utf-8') if not l.startswith('#'))][1:]
    tri = np.array([[float(x) for x in r[5:]] for r in rows]).reshape(-1, 3, 3)
    return dict(tri=tri, organ=np.array([r[0] for r in rows]), cls=np.array([r[1] for r in rows]), leaf=np.array([int(r[2]) for r in rows]), order=np.array([int(r[3]) for r in rows]))


class PlacedB:
    def __init__(self, T, spt, seed):
        self.pid = 'B'; t = T['tri']
        P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0]; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]; self._P = P; self._k = np.full(len(t), TRI, np.int8)
        isl = T['cls'] == 'Leaflet'; self.tau = np.zeros(len(t))
        cr = np.cross(P[:, 3:6], P[:, 6:9]); ar = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / np.maximum(2 * ar[:, None], 1e-30)
        idx = np.nonzero(isl & (ar > 0))[0]; rng = np.random.default_rng(seed)
        u = rng.random((len(idx), spt)); v = rng.random((len(idx), spt)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
        R = (t[idx, 0][:, None] + u[..., None] * P[idx, None, 3:6] + v[..., None] * P[idx, None, 6:9]).reshape(-1, 3)
        self.centre = R; self.normal = np.repeat(nrm[idx], spt, 0); self.A = np.repeat(ar[idx] / spt, spt) * 1e6
        self.own_local = np.repeat(idx, spt); self.delta = 1e-7; self.leafid = np.repeat(T['leaf'][idx], spt); self.order = np.repeat(T['order'][idx], spt)

    def prims(self):
        return self._P, self._k


def summarise(per_seed):
    return {k: TR.pm([d[k] for d in per_seed]) for k in per_seed[0].keys()}


files = sorted(TRI_ROOT.glob('*_triangles.csv'))
for p in files:
    pid = p.name[:-len('_triangles.csv')]; spt = 4 if not pid.startswith('P') else 2
    T = read_T(p); per = []; t1 = time.time()
    for sd in SEEDS:
        o = PlacedB(T, spt, sd); built = build_scene([o])
        r = trace_plants_tau([o], [o], sd, CFG['eps'], 0.0, 0.0, CFG['dex'], F, use_v1=True, built=built)['B']
        half = coincident(built[0], o.centre, o.normal, o.own_local); parts = r['parts'].copy(); parts[half] *= 0.5
        K = obs.kernel(parts); Ku = obs.kernel(parts, shaded=False); U = r['Uext']; ab = U * K * o.A; abu = U * Ku * o.A
        d = dict(total=ab.sum(), unshaded=abu.sum(), self=1 - ab.sum() / abu.sum(), area=o.A.sum(), mean_light=ab.sum() / o.A.sum(), coincident_share=float(half.mean()), receivers=float(len(o.A)))
        for lf in sorted(set(o.leafid.tolist())):
            m = o.leafid == lf
            d[f'leaf{lf:03d}_total'] = ab[m].sum(); d[f'leaf{lf:03d}_area'] = o.A[m].sum(); d[f'leaf{lf:03d}_light'] = ab[m].sum() / o.A[m].sum(); d[f'leaf{lf:03d}_self'] = 1 - ab[m].sum() / max(abu[m].sum(), 1e-30)
        per.append(d)
    RES['plants'][pid] = dict(summary=summarise(per), triangles=len(T['tri']), leaflet_triangles=int((T['cls'] == 'Leaflet').sum()), spt=spt, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    log(f'{pid} traced {time.time() - t1:.0f} s, receivers {per[0]["receivers"]:.0f}, mean light {per[0]["mean_light"]:.4f}, self {per[0]["self"]:.3f}')
    json.dump(RES, open(OUT / 'results.json', 'w'), indent=1, default=float)
log('done')
