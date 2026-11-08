"""K13 = K9 with optics rule 2 (tau 0.07 on every non-stem surfel, stem tau 0; engine_static_tau_rule2_20261105_01). Original header: K9 (adopted version v1tau_smoothn_r2, normal_smoothing_v2 CONTRACT Amendment 1 item 3): lamp / lateral / spacing scenarios (R3-1b alone, tray 60 mm baseline) and the tray 60 / 120 mm
self / neighbour / total shading attribution, v1 and v1_tau, with normal_mode smooth_all_r2 (receiver normal + occluder disc orientation). Tau bundle code unchanged; the smoothing hook
replaces TR.placed. Raw values are the stored ones (engine_static_tau_20261031_01 out_k1b / out_k3, tray_scenarios_tau.json) and the K6 raw run."""
import glob, hashlib, json, os, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
ND = Path(glob.glob('/kaggle/input/**/e_01__surfels.ply', recursive=True)[0]).parent
ND2 = Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent
RULE2 = Path(glob.glob('/kaggle/input/**/stem__R3-1b.npz', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT)); sys.path.insert(0, str(ND2))
import numpy as np
import normals as NM
from static.geometry import read_ply
from static import tau_runs as TR, obs
from static.tau_obs import build_scene

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), indent=1, default=float)
TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; SEEDS = TR.SEEDS; RADIUS = 2.0
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}; HASH['normals.py'] = hashlib.sha256((ND2 / 'normals.py').read_bytes()).hexdigest()
_orig_placed = TR.placed; INFO = {}


def sets_for(pid, q):
    rows = q.s.index
    if pid == 'R3-1b':
        leaf = np.where((q.axis == 0) & (q.r8 >= 1), q.r8, -1)
        return np.where(leaf >= 1, leaf, np.where(q.axis == 1, -10, np.where(q.axis == 2, -20, -30)))
    z = np.load(ND / f'labels__{pid}.npz'); lr = TR._map(z['rows'], z['label'].astype(int), rows); st = TR._map(z['rows'], z['status'].astype(int), rows)
    return np.where((st == 1) & (lr > 0), lr, -30)


def smooth_placed(pid, pos=None):
    q = _orig_placed(pid, pos); rows = q.s.index; sets = sets_for(pid, q)
    op = read_ply(obs.ply_path(pid))['opacity'][rows].astype(float); X = q.s.centre * 1000.0; raw = q.s.normal
    N, kept, _ = NM.smooth(X, raw, q.A * op, sets, RADIUS)
    P, k = q.s.prims(q.offset); P = P.copy(); P[:, 9:12] = N; tu2, tv2 = NM.reframe(N, P[:, 3:6]); P[:, 3:6] = tu2; P[:, 6:9] = tv2
    q.normal = N; q.prims = (lambda P=P, k=k: (P, k)); INFO[pid] = dict(kept_raw_share=float(kept.mean()))
    z = np.load(RULE2 / f'stem__{pid}.npz'); assert np.array_equal(z['rows'], rows), pid
    q.tau = np.where(z['stem_d1.5'].astype(bool), 0.0, CFG['tau_leaf']); INFO[pid]['stem_surfels_rule2'] = int(z['stem_d1.5'].sum())   # optics rule 2
    return q


TR.placed = smooth_placed                                   # TR.scenarios / TR.placed callers pick the smoothing hook up through the module global
res = dict(labels=['v1tau2_smoothn_r2', 'light_model_v1tau2', 'assumed_leaf_optics', 'relative_light', 'conditional_on_unvalidated_scale', 'normal_smoothing_v2_occluders'], engine_hash=HASH, radius_mm=RADIUS)
res['scenarios_r31b'] = TR.scenarios('r31b', log); jdump(res, OUT / 'k13_results.json')
res['scenarios_tray'] = TR.scenarios('tray', log); jdump(res, OUT / 'k13_results.json')
# tray attribution 60 / 120 mm (all / alone / none), v1 and v1_tau with smoothing
k6 = {}
for pitch in (0.060, 0.120):
    qs = [TR.placed(pid, pos) for pid, pos in zip(TR.TRAY, TR.tray_positions(pitch))]; built_all = build_scene(qs); key = f'{int(round(pitch * 1000))}'; k6[key] = {}
    for model in ('v1', 'tau0.07'):
        t0 = time.time(); r_all = TR.trace(qs, qs, model, built_all); vals = {}
        for q in qs:
            allt = float(np.mean([(r_all[q.pid][sd]['Uext'] * obs.kernel(r_all[q.pid][sd]['parts']) * q.A).sum() for sd in SEEDS]))
            r_q = TR.trace([q], [q], model, build_scene([q]))[q.pid]
            alone = float(np.mean([(r_q[sd]['Uext'] * obs.kernel(r_q[sd]['parts']) * q.A).sum() for sd in SEEDS]))
            none = float(np.mean([(r_q[sd]['Uext'] * obs.kernel(r_q[sd]['parts'], shaded=False) * q.A).sum() for sd in SEEDS]))
            vals[q.pid] = dict(all=allt, alone=alone, none=none, self_shading_loss=1 - alone / none, neighbour_shading_loss=1 - allt / alone, total_loss=1 - allt / none)
        tot = {kk: sum(v[kk] for v in vals.values()) for kk in ('all', 'alone', 'none')}
        vals['tray_total'] = dict(tot, self_shading_loss=1 - tot['alone'] / tot['none'], neighbour_shading_loss=1 - tot['all'] / tot['alone'], total_loss=1 - tot['all'] / tot['none'])
        k6[key][model] = vals; log(f'tray attribution pitch {key} {model} {time.time() - t0:.0f} s')
res['tray_attribution'] = k6; res['smoothing_info_last_placed'] = INFO; res['seconds'] = round(time.time() - T0); jdump(res, OUT / 'k13_results.json'); log('K13_DONE')
