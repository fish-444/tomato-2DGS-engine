"""Per-surfel static trace for R3-1b (S2px), Kaggle CPU. Same code as engine_static_20261026_02/static (byte copy in the dataset), same light
configuration as engine_static_20261026_01 Run 2 (v1 black leaves, measured field planes, user_assumed bars, seeds 101/202), distances
2.4 (d_ex) / 1.5 / 2 / 3 / 5 mm. Labels are not used by the trace. Output: /kaggle/working/trace_out.npz (+ trace_meta.json).
Local smoke: python trace_kernel.py --root <kaggle_data dir> --out <dir> --smoke"""
import argparse, glob, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--root'); ap.add_argument('--out', default='/kaggle/working'); ap.add_argument('--smoke', action='store_true')
a = ap.parse_args()
root = Path(a.root) if a.root else Path(glob.glob('/kaggle/input/**/light_consts.json', recursive=True)[0]).parent
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(out / 'numba_cache'))
sys.path.insert(0, str(root))
import numpy as np
from static import obs, light as L
from static.tracer import Scene, coincident

consts = json.load(open(root / 'light_consts.json'))
L.cos_m = lambda *args, **kw: consts['cos_m']
obs.R31B_PLY = root / 'R3-1b_surfels_full.ply'; obs.R31B_IDS = root / 'surfel_leaf_ids.csv'
F = obs.FieldFast.__new__(obs.FieldFast); z = np.load(root / 'field_planes.npz')
F.planes = {70.0: (z['xs_70'], z['ys_70'], z['v_70']), 150.0: (z['xs_150'], z['ys_150'], z['v_150'])}

t0 = time.time()
q = obs.Placed('R3-1b', obs.SCENE_XY)
n = len(q.s); print('surfels', n, 'base_xy_mm', (1000 * obs.base_xy(q.s)).tolist(), 'load s', round(time.time() - t0, 1), flush=True)
DS = (2.4,) if a.smoke else (2.4, 1.5, 2.0, 3.0, 5.0); SEEDS = (101,) if a.smoke else (101, 202)
res = dict(ply_row=q.s.index, centre_m=q.centre, normal=q.normal, area_b_cube_m2=q.s.area_b, area_a_m2=q.s.area_a, labels_vpcleanup=q.s.labels,
           offset_m=q.offset, base_xy_m=obs.base_xy(q.s))
for d in DS:
    for sd in SEEDS:
        t = time.time()
        rec = obs.trace_plants([q], [q], sd, delta=d / 1000, sensors=False, field=F)['R3-1b']
        res[f'parts_d{d:g}_s{sd}'] = rec['parts'].astype(np.float64)
        print('d', d, 'seed', sd, 'K mean', float(obs.kernel(rec['parts']).mean()), round(time.time() - t, 1), 's', flush=True)
res['U'] = rec['U']; res['U_ext'] = rec['Uext']; res['in_support'] = rec['ok']
P, k = q.prims(); res['coincident_1um'] = coincident(Scene(P, k, h=0.003), q.centre, q.normal, np.arange(n))
np.savez_compressed(out / 'trace_out.npz', **res)
json.dump(dict(distances_mm=DS, seeds=SEEDS, surfels=n, seconds=round(time.time() - t0), coincident=int(res['coincident_1um'].sum()),
               cos_m=consts['cos_m'], diffuse=L.DIFFUSE, numpy=np.__version__, python=sys.version), open(out / 'trace_meta.json', 'w'), indent=1)
print('TRACE_DONE', round(time.time() - t0), 's', flush=True)
