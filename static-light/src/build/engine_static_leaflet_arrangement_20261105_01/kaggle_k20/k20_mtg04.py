"""K20 (engine_static_leaflet_arrangement_20261105_01 CONTRACT amendment 4) = K10 with the _04 meshes."""
"""K10 (v1tau_smoothn_r2 cross-check, report only; normal_smoothing_v2 CONTRACT Amendment 1 item 5): per-leaf light of the static MTG set `_02` (11 plants, engine-format meshes)
under the black-leaf model (v1) and under leaf transmission (v1_tau: leaflets tau 0.07, stem / rachis tau 0), same lamp scene / seeds / d_ex 2.4 mm. Receivers: 16 points per leaflet
triangle, coincident half credit (as in the first normals test). Tau bundle code unchanged. Output: per-leaf absorbed light (a = 1) per plant and model."""
import glob, hashlib, json, os, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
ND2 = Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent
ND = Path(glob.glob('/kaggle/input/**/e_01__surfels.ply', recursive=True)[0]).parent
REF = Path(glob.glob('/kaggle/input/**/mtg04__e_01.obj', recursive=True)[0]).parent   # K20: _04 meshes
sys.path.insert(0, str(TAU_ROOT))
import numpy as np
from static import tau_runs as TR, obs
from static.tau_obs import build_scene, trace_plants_tau
from static.tracer import TRI, coincident

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), indent=1, default=float)
F = TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; SEEDS = TR.SEEDS; EPS = CFG['eps']; DEX = 0.0024
_orig_ply = obs.ply_path
obs.ply_path = lambda pid: (ND / f'{pid}__surfels.ply') if pid.startswith('e_') else _orig_ply(pid)
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07']


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
    if name == 'stem' or name.startswith('rachis'):
        return None
    return name.split('__')[0] if ('__' in name and name.startswith('L')) else None


class PlacedMesh:
    def __init__(self, objs, offset, seed, spt=16):
        names, tris = [], []
        for o in sorted(objs):
            names += [o] * len(objs[o]); tris.append(objs[o])
        t = np.concatenate(tris) / 1000.0 + offset; names = np.array(names)
        P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0]; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]; self._P = P; self._k = np.full(len(t), TRI, np.int8)
        cr = np.cross(P[:, 3:6], P[:, 6:9]); ar = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / np.maximum(2 * ar[:, None], 1e-30)
        isl = np.array([obj_leaf(o) is not None for o in names]) & (ar > 0); idx = np.nonzero(isl)[0]; rng = np.random.default_rng(seed)
        self.tau = np.where(np.array([obj_leaf(o) is not None for o in names]), CFG['tau_leaf'], 0.0)
        u = rng.random((len(idx), spt)); v = rng.random((len(idx), spt)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
        self.centre = (t[idx, 0][:, None] + u[..., None] * P[idx, None, 3:6] + v[..., None] * P[idx, None, 6:9]).reshape(-1, 3)
        self.normal = np.repeat(nrm[idx], spt, 0); self.area = np.repeat(ar[idx] / spt, spt) * 1e6
        self.uid = np.repeat(np.array([obj_leaf(o) for o in names[idx]]), spt); self.own_local = np.repeat(idx, spt); self.delta = 1e-7; self.pid = 'mtg'
    def prims(self):
        return self._P, self._k


for pid in PLANTS:
    if (OUT / f'{pid}.json').exists():
        continue
    t0 = time.time(); q = obs.Placed(pid, obs.SCENE_XY); off = q.offset.copy(); obs._CACHE.clear()
    objs = load_obj(REF / f'mtg04__{pid}.obj'); res = dict(plant=pid, labels=['v1tau_smoothn_r2_crosscheck', 'mtg_cross_check_report_only', 'relative_light', 'conditional_on_unvalidated_scale'], engine_hash=HASH)
    for model in ('v1', 'tau'):
        per, tot = {}, []
        for sd in SEEDS:
            o = PlacedMesh(objs, off, sd); built = build_scene([o])
            r = trace_plants_tau([o], [o], sd, EPS, 0.0, 0.0, DEX, F, tau_scale=1.0, use_v1=(model == 'v1'), built=built)['mtg']
            half = coincident(built[0], o.centre, o.normal, o.own_local); parts = r['parts'].copy(); parts[half] *= 0.5
            ab = r['Uext'] * obs.kernel(parts) * o.area; tot.append(float(ab.sum()))
            for u in np.unique(o.uid):
                per.setdefault(str(u), []).append(float(ab[o.uid == u].sum()))
        res[model] = dict(total=float(np.mean(tot)), leaves={u: float(np.mean(v)) for u, v in per.items()})
    jdump(res, OUT / f'{pid}.json'); log(f'MTG {pid} v1 {res["v1"]["total"]:.1f} tau {res["tau"]["total"]:.1f} {time.time() - t0:.0f} s')
jdump(dict(engine_hash=HASH, wall_seconds=round(time.time() - T0)), OUT / 'SUMMARY.json'); log('K20_DONE')
