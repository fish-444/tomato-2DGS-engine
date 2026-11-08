"""K5 of the v1_tau reference table (engine_static_v1tau_reference_20261104_01/CONTRACT.md §2): v1 and v1_tau for 14 plants (pilot, nine, e_*).
Tau bundle code (v1 static package + v1_tau additions) is used without edits. Per plant: gates G0a/G0b (per-surfel kernels vs stored), d_ex 2.4/1.5/2/3/5, footprint x0.8/1.2,
diffuse 0.5/0.7, half-credit report, per-leaf and bucket light, LAI and 5 mm-layer transmittance. Checkpoint unit = plant (<id>/DONE). a = 0.85 and a = 1 come from the same trace.
Local smoke: python k5_plants.py --tau <dir> --nine <dir> --nd <dir> --ref <dir> --out <dir> --plants n_03 --quick"""
import argparse, glob, hashlib, json, math, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ('tau', 'nine', 'nd', 'ref'):
    ap.add_argument('--' + k)
ap.add_argument('--out', default='/kaggle/working'); ap.add_argument('--plants', default='@PLANTS@'); ap.add_argument('--quick', action='store_true')
ap.add_argument('--tol', type=float, default=0.0)   # Kaggle: exact (0). Local smoke only: platform float differences ~2e-16
ap.add_argument('--nd2'); ap.add_argument('--radius', type=float, default=2.0)   # K8: smoothing radius (mm) of the adopted v1tau_smoothn_r2; normals.py dataset
a = ap.parse_args()
TAU_ROOT = Path(a.tau) if a.tau else Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(a.nine) if a.nine else Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
ND = Path(a.nd) if a.nd else Path(glob.glob('/kaggle/input/**/e_01__surfels.ply', recursive=True)[0]).parent
REF = Path(a.ref) if a.ref else Path(glob.glob('/kaggle/input/**/refk__R3-1b.npz', recursive=True)[0]).parent
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
ND2 = Path(a.nd2) if a.nd2 else Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT)); sys.path.insert(0, str(ND2))
import numpy as np
import normals as NM
from static.geometry import read_ply
from static import tau_runs as TR, obs, light as L
from static.tau_obs import build_scene, trace_plants_tau
from static.tracer import trace_cabinet, trace_cabinet_tau, coincident

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True); json.dump(o, open(p, 'w', encoding='utf-8'), indent=1, default=float)

F = TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; SEEDS = TR.SEEDS; A_MAIN = CFG['a_main']; EPS = CFG['eps']
E_IDS = ['e_01', 'e_04', 'e_05', 'e_07']
_orig_ply = obs.ply_path
obs.ply_path = lambda pid: (ND / f'{pid}__surfels.ply') if pid in E_IDS else _orig_ply(pid)
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
HASH['normals.py'] = hashlib.sha256((ND2 / 'normals.py').read_bytes()).hexdigest()
DS =[2.4] if a.quick else [2.4, 1.5, 2.0, 3.0, 5.0]; FOOT = [] if a.quick else [0.8, 1.2]; FDS = [0.5, 0.7]
PLANTS = [x for x in a.plants.split(',') if x]


def placed_raw(pid):
    if pid in E_IDS:
        q = obs.Placed(pid, obs.SCENE_XY); rows = q.s.index; z = np.load(ND / f'{pid}__area_b.npz')
        if not np.array_equal(z['rows'], rows):
            raise RuntimeError(f'{pid}: area rows != opaque rows')
        q.A = z['area_b_mm2'].astype(float); zz = 1000 * q.centre[:, 2]; q.h = zz - np.percentile(zz, 0.5)
        q.tau = np.full(len(rows), CFG['tau_leaf']); q.masks = {}; q.group_key = np.zeros(len(rows), int)
        return q
    return TR.placed(pid)


def apply_smooth(q, pid):
    """normal_mode_v2 = smooth_all_r<radius> (normal_smoothing_v2 CONTRACT section 1): receiver normal AND occluder disc orientation from the weighted local plane fit
    (same label sets, weights area (b) x opacity, >= 6 neighbours else raw kept). Disc centre / size / opacity / d_ex unchanged. The shared Surfels object is not touched."""
    rows = q.s.index; leaf, uids, prim, dz, note = leaf_labels(pid, q)
    sets = np.where(leaf >= 1, leaf, np.where(q.axis == 1, -10, np.where(q.axis == 2, -20, -30))) if pid == 'R3-1b' else np.where(leaf >= 1, leaf, -30)
    op = read_ply(obs.ply_path(pid))['opacity'][rows].astype(float); X = q.s.centre * 1000.0; raw = q.s.normal
    N, kept, _ = NM.smooth(X, raw, q.A * op, sets, a.radius)
    P, k = q.s.prims(q.offset); P = P.copy(); P[:, 9:12] = N; tu2, tv2 = NM.reframe(N, P[:, 3:6]); P[:, 3:6] = tu2; P[:, 6:9] = tv2
    q.raw_normal = raw; q.normal = N; q.prims = (lambda P=P, k=k: (P, k))
    q.smooth_info = dict(radius_mm=a.radius, kept_raw_share=float(kept.mean()), noise_median_before_deg=NM.noise_metric(X, raw)[0], noise_median_after_deg=NM.noise_metric(X, N)[0])
    return q


def placed(pid):
    return apply_smooth(placed_raw(pid), pid)


def leaf_labels(pid, q):
    """leaf id per surfel (-1 none), uid map, primary unassigned mask (contract section 2), dz10 unassigned mask, label note."""
    rows = q.s.index
    if pid == 'R3-1b':
        leaf = np.where((q.axis == 0) & (q.r8 >= 1), q.r8, -1)
        return leaf, {int(k): f'R8pc_{int(k)}' for k in np.unique(leaf[leaf >= 1])}, leaf < 1, leaf < 1, 'R8_pc labels (labels_rachis.npz); unassigned = non-leaf (stem, rachis-like, unlabelled blade)'
    z = np.load(ND / f'labels__{pid}.npz'); lr = TR._map(z['rows'], z['label'].astype(int), rows); st = TR._map(z['rows'], z['status'].astype(int), rows)
    leaf = np.where((st == 1) & (lr > 0), lr, -1); uids = {k + 1: str(u) for k, u in enumerate(z['leaf_uids'])}
    dz = leaf < 1
    prim = (q.axis_like | (q.group < 0)) if pid not in E_IDS else dz
    return leaf, uids, prim, dz, ('dz10 snapshot labels (status 1 = assigned, lower bound); unassigned = axis-like + blade without group' if pid not in E_IDS else 'dz10 snapshot labels; unassigned = status 0')


def trace_m(q, built, model, dex_mm):
    return {sd: trace_plants_tau([q], [q], sd, EPS, 0.0, 0.0, dex_mm / 1000.0, F, tau_scale=1.0, use_v1=(model == 'v1'), built=built)[q.pid] for sd in SEEDS}


def kU(q, recs, sd, fd=None):
    r = recs[sd]
    if fd is None:
        return obs.kernel(r['parts']), obs.kernel(r['parts'], shaded=False), r['Uext']
    _, Uext, _ = obs.unshaded_U(F, q.centre, L.bar_points(sd), None, L.cos_m(), fd)
    return obs.kernel(r['parts'], fd), obs.kernel(r['parts'], fd, shaded=False), Uext


def metr(q, recs, A, leaf, uids, prim, dz, fd=None, extra=False):
    top = np.percentile(q.h, 99.5); th = {'lower': q.h < top / 3, 'middle': (q.h >= top / 3) & (q.h < 2 * top / 3), 'upper': q.h >= 2 * top / 3}
    tot, totu, ab_s = [], [], []
    for sd in SEEDS:
        K, Ku, U = kU(q, recs, sd, fd); ab = U * K * A; tot.append(float(ab.sum())); totu.append(float((U * Ku * A).sum())); ab_s.append(ab)
    abm = np.mean(ab_s, 0); m = float(np.mean(tot)); d = dict(total=m, sigma=abs(tot[0] - tot[1]) / 2, cv=abs(tot[0] - tot[1]) / math.sqrt(2) / m, mean_light=m / A.sum(),
                                                             self_shading=1 - m / float(np.mean(totu)), overflow_rays=int(sum(recs[sd]['novf'].sum() for sd in SEEDS)))
    d['assigned_leaf_total'] = float(abm[leaf >= 1].sum())
    if extra:
        d.update({f'{k}_total': float(abm[v].sum()) for k, v in th.items()}); d.update({f'{k}_mean_light': float(abm[v].sum() / A[v].sum()) for k, v in th.items()})
        d['unassigned_share'] = float(abm[prim].sum() / m); d['unassigned_share_dz10'] = float(abm[dz].sum() / m); d['unassigned_absorbed'] = float(abm[prim].sum())
        d['leaves'] = {uids[k]: dict(absorbed=float(abm[leaf == k].sum()), area_b_mm2=float(A[leaf == k].sum()), surfels=int((leaf == k).sum())) for k in uids}
    return d


def sensors(q, built, model, A):
    scene, tau_prim, off = built; S, lay = obs.sensors_for(q.pos, q.centre[:, 2].max()); Ns = np.tile([0.0, 0.0, 1.0], (len(S), 1)); own = np.full(len(S), -1, np.int64)
    m = L.cos_m(); Ts = []
    for sd in SEEDS:
        bars = L.bar_points(sd); dm = L.dome(sd)
        if model == 'v1':
            parts = trace_cabinet(scene, S, Ns, False, own, 1e-7, bars, m, dm, L.DIFFUSE)[3]
        else:
            parts = trace_cabinet_tau(scene, S, Ns, False, own, 1e-7, bars, m, dm, L.DIFFUSE, tau_prim, EPS)[3]
        Ts.append(obs.kernel(parts))
    T = np.mean(Ts, 0); a_m2 = A * 1e-6; leaf = q.leaf; z_rel = q.centre[:, 2] - obs.MEDIUM_Z
    inpot = (np.abs(q.centre[:, 0] - q.pos[0]) <= obs.POT_SIDE / 2) & (np.abs(q.centre[:, 1] - q.pos[1]) <= obs.POT_SIDE / 2)
    prof = []
    for k in range(int(lay.max()) + 1):
        zlo, zhi = 0.005 * k, 0.005 * (k + 1); sel = leaf & (z_rel >= zlo) & (z_rel < zhi); sp = sel & inpot
        prof.append(dict(layer=k, z_mm=[1000 * zlo, 1000 * zhi], transmittance=float(T[lay == k].mean()), leaf_area_m2=float(a_m2[sel].sum()), leaf_area_above_pot_m2=float(a_m2[sp].sum())))
    lai_above = np.cumsum([p['leaf_area_above_pot_m2'] for p in prof][::-1])[::-1] / obs.POT_SIDE ** 2
    for p, la in zip(prof, lai_above):
        p['lai_above_pot_top'] = float(la - p['leaf_area_above_pot_m2'] / obs.POT_SIDE ** 2 / 2); p['below_pot_rim_walls_not_modelled'] = bool(p['z_mm'][1] <= 1000 * (obs.RIM_Z - obs.MEDIUM_Z))
    Tl = np.array([p['transmittance'] for p in prof]); X = np.array([p['lai_above_pot_top'] for p in prof]); ok = (Tl > 0) & (X > 0)
    rim = [p for p in prof if not p['below_pot_rim_walls_not_modelled']]
    return dict(lai_pot_top=float(a_m2[leaf].sum() / obs.POT_SIDE ** 2), leaf_area_fraction_above_pot_square=float(a_m2[leaf & inpot].sum() / a_m2[leaf].sum()),
                k_extinction_diagnostic=float(-(X[ok] * np.log(Tl[ok])).sum() / (X[ok] ** 2).sum()) if ok.sum() >= 2 else None,
                transmittance_at_rim_layer=rim[0]['transmittance'] if rim else None, transmittance_lowest_layer_inside_pot=prof[0]['transmittance'], profile=prof)


def run_plant(pid):
    t0 = time.time(); q = placed(pid); A = q.A; n = len(A); built = build_scene([q]); leaf, uids, prim, dz, lnote = leaf_labels(pid, q)
    res = dict(plant=pid, surfels=n, label_note=lnote, area_b_mm2=float(A.sum()), engine_hash=HASH, seeds=list(SEEDS), tau=CFG['tau_leaf'], a_main=A_MAIN, eps_mm=1000 * EPS,
               tau_set='e_*: tau 0.07 on every surfel (no stem class); R3-1b: stem tau 0; nine: axis-like tau 0' if pid in E_IDS else 'stem / axis-like tau 0, others 0.07',
               labels=['light_model_v1_tau', 'assumed_leaf_optics', 'relative_light', 'conditional_on_unvalidated_scale'], dex={}, footprint={}, diffuse={}, G={})
    base = {}
    for dex in DS:
        for model in ('v1', 'tau'):
            tt = time.time(); recs = trace_m(q, built, model, dex); base[(model, dex)] = recs
            res['dex'].setdefault(str(dex), {})[model] = metr(q, recs, A, leaf, uids, prim, dz, extra=(dex == 2.4))
            log(f'{pid} dex {dex} {model} {time.time() - tt:.0f} s')
    r0 = {m: base[(m, 2.4)] for m in ('v1', 'tau')}
    Kv = np.mean([obs.kernel(r0['v1'][sd]['parts']) for sd in SEEDS], 0); Kt = np.mean([obs.kernel(r0['tau'][sd]['parts']) for sd in SEEDS], 0)
    (OUT / pid).mkdir(exist_ok=True)
    res['G'] = dict(note='raw-mode regression gates G0a/G0b are in K5; this run is normal_mode smooth_all (compared to the K5 raw outputs offline)')
    res['normal_mode'] = 'smooth_all_r%g' % a.radius; res['smoothing'] = q.smooth_info; res['labels'] = res['labels'] + ['v1tau_smoothn_r2', 'normal_smoothing_v2_occluders']
    log(f'{pid} smoothing {q.smooth_info}')
    U = np.mean([r0['tau'][sd]['Uext'] for sd in SEEDS], 0)
    np.savez_compressed(OUT / pid / 'per_surfel.npz', ply_row=q.s.index, K_v1=Kv.astype(np.float32), K_tau=Kt.astype(np.float32), U=U.astype(np.float32),
                        area_b_mm2=A.astype(np.float32), leaf=leaf.astype(np.int16), unassigned=prim, h_above_base_mm=q.h.astype(np.float32),
                        normal_smooth=q.normal.astype(np.float32), xyz_mm=(1000 * q.centre).astype(np.float32))
    # half-credit report (coincident probe along the receiver normal; reallocation only)
    coin = coincident(built[0], q.centre, q.normal, np.arange(n))
    res['halfcredit'] = {m: dict(hits=int(coin.sum()), total_main=float(np.mean([(r0[m][sd]['Uext'] * obs.kernel(r0[m][sd]['parts']) * A).sum() for sd in SEEDS])),
                                 total_half=float(np.mean([(r0[m][sd]['Uext'] * np.where(coin, 0.5, 1.0) * obs.kernel(r0[m][sd]['parts']) * A).sum() for sd in SEEDS]))) for m in ('v1', 'tau')}
    for fd in FDS:
        res['diffuse'][str(fd)] = {m: metr(q, r0[m], A, leaf, uids, prim, dz, fd=fd) for m in ('v1', 'tau')}
    for s in FOOT:
        Pm, km = q.prims(); Pm = Pm.copy(); Pm[:, 12:14] *= s; orig = q.prims; q.prims = lambda Pm=Pm, km=km: (Pm, km); b2 = build_scene([q])
        res['footprint'][str(s)] = {}
        for model in ('v1', 'tau'):
            recs = trace_m(q, b2, model, 2.4); res['footprint'][str(s)][model] = metr(q, recs, A * s * s, leaf, uids, prim, dz)
        q.prims = orig; log(f'{pid} footprint {s}')
    res['sensors'] = {m: sensors(q, built, m, A) for m in ('v1', 'tau')}
    # tau bracket for e_*: tau 0 on label-0 (unassigned) surfels (report only)
    if pid in E_IDS:
        keep = q.tau.copy(); q.tau = np.where(dz, 0.0, keep); b3 = build_scene([q]); recs = trace_m(q, b3, 'tau', 2.4); q.tau = keep
        res['bracket_tau0_on_unassigned'] = metr(q, recs, A, leaf, uids, prim, dz)
    res['seconds'] = round(time.time() - t0); jdump(res, OUT / pid / 'plant.json'); (OUT / pid / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    log(f'{pid} done {res["seconds"]} s'); obs._CACHE.clear()
    return True


for pid in PLANTS:
    if (OUT / pid / 'DONE').exists():
        continue
    if not run_plant(pid):
        log('GATE FAILED, stopping'); break
jdump(dict(engine_hash=HASH, wall_seconds=round(time.time() - T0), plants=PLANTS), OUT / 'SUMMARY.json'); log('K5_DONE')
