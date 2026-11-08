"""K1 of the normal smoothing test (engine_static_normals_20261104_01/CONTRACT.md): G0a regression, real plants (§4), synthetic recovery (§3),
MTG cross-check (§5). v1 static package = byte copy in static-nine-bundle-20261028 (untouched); new code = normals.py (static-normals-20261104).
Kaggle CPU. Resume: parts with a DONE file are skipped. Local smoke: python k1_normals.py --nine <dir> --nd <dir> --out <dir> --parts real --only R3-1b"""
import argparse, csv, glob, hashlib, json, math, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--nine'); ap.add_argument('--nd'); ap.add_argument('--out', default='/kaggle/working')
ap.add_argument('--parts', default='real,synthetic,mtg'); ap.add_argument('--only', default='')
a = ap.parse_args()
NINE = Path(a.nine) if a.nine else Path(glob.glob('/kaggle/input/**/light_consts.json', recursive=True)[0]).parent
ND = Path(a.nd) if a.nd else Path(glob.glob('/kaggle/input/**/normals_config.json', recursive=True)[0]).parent
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))


def mirror(src):
    """The dataset is uploaded flat (a__b__c); rebuild the tree as symlinks under OUT/nd."""
    if not (src / 'normals_config.json').exists() or (src / 'labels').is_dir():
        return src
    dst = OUT / 'nd'
    for p in src.iterdir():
        if p.is_file():
            t = dst.joinpath(*p.name.split('__')); t.parent.mkdir(parents=True, exist_ok=True)
            if not t.exists():
                os.symlink(p, t)
    return dst


ND = mirror(ND)
sys.path.insert(0, str(NINE)); sys.path.insert(0, str(ND))
import numpy as np
import numba
from scipy.stats import spearmanr
from static import obs, light as L
from static.geometry import read_ply
from static.tracer import DISC, TRI
import normals as NM

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))

CFG = json.load(open(ND / 'normals_config.json'))
consts = json.load(open(NINE / 'light_consts.json')); L.cos_m = lambda *x, **k: consts['cos_m']
obs.R31B_PLY = NINE / 'R3-1b' / 'surfels.ply'; obs.R31B_IDS = NINE / 'R3-1b' / 'surfel_leaf_ids.csv'
E = set(CFG['e'])
obs.ply_path = lambda pid: obs.R31B_PLY if pid == 'R3-1b' else ((ND if pid in E else NINE) / pid / 'surfels.ply')
F = obs.FieldFast.__new__(obs.FieldFast); _z = np.load(NINE / 'field_planes.npz')
F.planes = {70.0: (_z['xs_70'], _z['ys_70'], _z['v_70']), 150.0: (_z['xs_150'], _z['ys_150'], _z['v_150'])}
SEEDS = tuple(CFG['seeds']); DEX = CFG['d_ex_mm'] / 1000.0; MODES = NM.MODES
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((NINE / 'static').glob('*.py'))}
HASH['normals.py'] = hashlib.sha256((ND / 'normals.py').read_bytes()).hexdigest()
TAGS = ['normal_smoothing_test', 'relative_light', 'conditional_on_unvalidated_scale', 'skeleton_labels_ungraded', 'after_checkpoint_v1']
parts = set(a.parts.split(','))


def jdump(obj, p):
    p.parent.mkdir(parents=True, exist_ok=True); json.dump(obj, open(p, 'w', encoding='utf-8'), indent=1, default=float)


def _map(rows_src, vals, rows):
    order = np.argsort(rows_src); idx = order[np.searchsorted(rows_src[order], rows)]
    if not np.array_equal(rows_src[idx], rows):
        raise RuntimeError('row sets differ')
    return vals[idx]


def trace(q, sd):
    rec = obs.trace_plants([q], [q], sd, delta=DEX, sensors=False, field=F)[q.pid]
    return obs.kernel(rec['parts']), obs.kernel(rec['parts'], shaded=False), rec['Uext']


def spear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(spearmanr(x, y)[0])


# ------------------------------------------------------------------ real plants
def area_of(pid, q):
    rows = q.s.index
    if pid in E:
        z = np.load(ND / pid / 'area_b.npz')
        if not np.array_equal(z['rows'], rows):
            raise RuntimeError(f'{pid}: area rows != opaque rows')
        return z['area_b_mm2'].astype(float)
    sil = {}
    with open(NINE / pid / 'static_surfels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith('#')):
            sil[int(r['surfel_row'])] = float(r['area_b_mm2'])
    if set(sil) != set(rows.tolist()):
        raise RuntimeError(f'{pid}: static_surfels rows != opaque rows')
    return np.array([sil[r] for r in rows.tolist()])


def labels_of(pid, q):
    """leaf id per surfel (-1 none), smoothing set id, uid map, category name per surfel."""
    rows = q.s.index
    if pid == 'R3-1b':
        z = np.load(ND / 'labels' / 'R3-1b_labels_rachis.npz')
        ax = _map(z['rows'], z['axis_kind'].astype(int), rows); r8 = _map(z['rows'], z['label'].astype(int), rows)
        leaf = np.where((ax == 0) & (r8 >= 1), r8, -1)
        sets = np.where(leaf >= 1, leaf, np.where(ax == 1, -10, np.where(ax == 2, -20, -30)))
        uids = {int(k): f'R8pc_{int(k)}' for k in np.unique(leaf[leaf >= 1])}
        src = 'labels_rachis.npz (axis_kind, R8_pc label)'
    else:
        z = np.load(ND / 'labels' / f'{pid}.npz')
        lr = _map(z['rows'], z['label'].astype(int), rows); st = _map(z['rows'], z['status'].astype(int), rows)
        leaf = np.where((st == 1) & (lr > 0), lr, -1)
        sets = np.where(leaf > 0, leaf, -30)
        uids = {k + 1: str(u) for k, u in enumerate(z['leaf_uids'])}
        src = str(z['source'])
    cat = np.where(sets > 0, 'leaf', np.where(sets == -10, 'stem', np.where(sets == -20, 'rachis_like', 'unassigned')))
    return leaf, sets, uids, cat, src


def run_real(pid):
    t0 = time.time()
    q = obs.Placed(pid, obs.SCENE_XY); n = len(q.s); A = area_of(pid, q)
    op = read_ply(obs.ply_path(pid))['opacity'][q.s.index].astype(float); w = A * op
    leaf, sets, uids, cat, src = labels_of(pid, q)
    X = q.s.centre * 1000.0; rawN = q.s.normal
    res = dict(plant=pid, tags=TAGS, surfels=n, label_source=src, area_b_mm2=float(A.sum()), engine_hash=HASH, seeds=list(SEEDS), d_ex_mm=CFG['d_ex_mm'],
               occlusion='raw normals (disc plane) in every mode; only receiver incidence normal changes', modes={})
    per = dict(xyz_mm=X.astype(np.float32), area_b_mm2=A.astype(np.float32), leaf=leaf.astype(np.int16), sets=sets.astype(np.int16), ply_row=q.s.index)
    leaf_ids = sorted(uids)
    pooled = dict(a=q.s.a.copy(), b=q.s.b.copy(), op=op.copy(), density=n / A.sum())
    for mode in MODES:
        if mode == 'raw':
            N = rawN; kept = np.ones(n, bool); cnt = np.zeros(n, np.int64)
        else:
            N, kept, cnt = NM.smooth(X, rawN, w, sets, NM.RADIUS[mode])
        q.normal = N
        ab_s, abu_s, K_s, U = [], [], [], None
        for sd in SEEDS:
            K, Ku, U = trace(q, sd); ab_s.append(U * K * A); abu_s.append(U * Ku * A); K_s.append(K)
        q.normal = rawN
        Km = np.mean(K_s, 0); abm = np.mean(ab_s, 0)
        tot = [float(x.sum()) for x in ab_s]; totu = [float(x.sum()) for x in abu_s]
        rowsL = []
        for k in leaf_ids:
            m = leaf == k
            rowsL.append(dict(leaf=uids[k], leaf_label=int(k), surfels=int(m.sum()), area_b_mm2=float(A[m].sum()),
                              absorbed=float(np.mean([x[m].sum() for x in ab_s])), light_per_area=float(np.mean([x[m].sum() for x in ab_s]) / max(A[m].sum(), 1e-30)),
                              self_shading=float(1 - np.mean([x[m].sum() for x in ab_s]) / max(np.mean([x[m].sum() for x in abu_s]), 1e-30))))
        med, ang, rms = NM.noise_metric(X, N, with_rms=True)
        d = dict(absorbed=float(np.mean(tot)), absorbed_seeds=tot, cv_two_seed=float(abs(tot[0] - tot[1]) / math.sqrt(2) / np.mean(tot)),
                 mean_relative_light=float(np.mean(tot) / A.sum()), self_shading=float(1 - np.mean(tot) / np.mean(totu)),
                 assigned_leaf_absorbed=float(abm[leaf >= 1].sum()), unassigned_absorbed=float(abm[leaf < 1].sum()),
                 kept_raw_share=float(kept.mean()), kept_raw_share_by_set={c: float(kept[cat == c].mean()) for c in sorted(set(cat.tolist()))},
                 neighbour_count_median=float(np.median(cnt)) if mode != 'raw' else None,
                 noise_metric_median_deg=med, noise_metric_p90_deg=float(np.percentile(ang, 90)), noise_metric_n=int(len(ang)),
                 noise_hist_counts=np.histogram(ang, bins=np.arange(0, 92, 2))[0].tolist(), leaves=rowsL)
        if mode == 'raw':
            d['noise_rms_plane_mm_median'] = float(np.median(rms)); pooled['rms'] = float(np.median(rms))
        res['modes'][mode] = d
        per[f'K_{mode}'] = Km.astype(np.float32); per[f'kept_{mode}'] = kept
        if mode == 'raw':
            per['U'] = U.astype(np.float32)
            ref = ND / 'ref' / f'{pid}_K.npz'
            if ref.exists():
                z = np.load(ref)
                eq = bool(np.array_equal(z['ply_row'], q.s.index) and np.array_equal(Km, z['relative_light']))
                res['G0a'] = dict(identical=eq, max_abs_diff=float(np.abs(Km - z['relative_light']).max()) if len(Km) == len(z['relative_light']) else None)
                log(f'G0a {pid} identical={eq} maxdiff={res["G0a"]["max_abs_diff"]}')
                if not eq:
                    jdump(dict(G0a_failed=pid, detail=res['G0a']), OUT / 'G0a_FAILED.json'); jdump(res, OUT / 'real' / f'{pid}.json'); sys.exit(0)
            else:
                res['G0a'] = dict(identical=None, note='no earlier static result (e_*)')
        log(f'{pid} {mode} absorbed={np.mean(tot):.2f} self={d["self_shading"]:.4f} kept={kept.mean():.4f} noise={med:.1f}')
    raw = {r['leaf']: r['absorbed'] for r in res['modes']['raw']['leaves'] if r['area_b_mm2'] > 0}
    for mode in MODES[1:]:
        md = {r['leaf']: r['absorbed'] for r in res['modes'][mode]['leaves'] if r['area_b_mm2'] > 0}
        ks = sorted(raw); res['modes'][mode]['leaf_rank_spearman_vs_raw'] = spear([raw[k] for k in ks], [md[k] for k in ks])
        res['modes'][mode]['plant_change_vs_raw_pct'] = 100 * (res['modes'][mode]['absorbed'] / res['modes']['raw']['absorbed'] - 1)
    res['seconds'] = round(time.time() - t0)
    jdump(res, OUT / 'real' / f'{pid}.json'); np.savez_compressed(OUT / 'real' / f'{pid}_persurfel.npz', **per)
    obs._CACHE.clear()
    return pooled


if 'real' in parts:
    pooled = {}
    plants = [x for x in a.only.split(',') if x] or CFG['real']
    for pid in plants:
        if (OUT / 'real' / f'{pid}.json').exists() and (OUT / 'real' / f'{pid}_persurfel.npz').exists():
            continue
        pooled[pid] = run_real(pid)
        old = dict(np.load(OUT / 'real' / 'pooled.npz', allow_pickle=True)['d'].item()) if (OUT / 'real' / 'pooled.npz').exists() else {}
        old[pid] = pooled[pid]; np.savez_compressed(OUT / 'real' / 'pooled.npz', d=np.array(old, dtype=object))
    log('REAL_DONE')


# ------------------------------------------------------------------ synthetic (§3)
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
    """Surfel sample of one mesh (true normals). Units: positions mm, area mm2."""
    def __init__(self, pid, objs, rho, pool, sig_off, rng):
        C, Nn, Ar, lid, sets, uidl = [], [], [], [], [], []
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
    """Placed synthetic plant for obs.trace_plants: occluder frame (n, tu, tv) and receiver normal set separately."""
    def __init__(self, base, n_occ, tu_occ, tv_occ, n_rec):
        self.pid = 'syn'; self.pos = obs.SCENE_XY; self.centre = base.centre; self.normal = n_rec; self.area = base.A; self.area_a = base.A
        self.leaf = np.ones(len(base.A), bool); self.leaf_id = base.leaf; self.own_local = np.arange(len(base.A)); self.delta = None; self.coincident_half = False
        P = np.zeros((len(base.A), 14)); P[:, 0:3] = base.centre; P[:, 3:6] = tu_occ; P[:, 6:9] = tv_occ; P[:, 9:12] = n_occ; P[:, 12] = base.a; P[:, 13] = base.b
        self._P = P; self._k = np.full(len(base.A), DISC, np.int8)
    def prims(self):
        return self._P, self._k


def syn_run(base, n_occ, tu_occ, tv_occ, n_rec):
    sp = SP(base, n_occ, tu_occ, tv_occ, n_rec); ab, abu = [], []
    for sd in SEEDS:
        K, Ku, U = trace(sp, sd); ab.append(U * K * base.A); abu.append(U * Ku * base.A)
    tot = float(np.mean([x.sum() for x in ab])); totu = float(np.mean([x.sum() for x in abu]))
    leaves = {base.uids[k]: float(np.mean([x[base.leaf == k].sum() for x in ab])) for k in base.uids}
    return dict(total=tot, self=1 - tot / totu, leaves=leaves)


if 'synthetic' in parts and not (OUT / 'synthetic' / 'DONE').exists():
    pz = np.load(OUT / 'real' / 'pooled.npz', allow_pickle=True)['d'].item()
    pool = dict(a=np.concatenate([pz[p]['a'] for p in CFG['real']]), b=np.concatenate([pz[p]['b'] for p in CFG['real']]), op=np.concatenate([pz[p]['op'] for p in CFG['real']]))
    rho = float(np.median([pz[p]['density'] for p in CFG['real']])); sig_off = float(np.median([pz[p]['rms'] for p in CFG['real']]))
    SEED = CFG['synthetic_seed']; bases = {}
    for i, pid in enumerate(CFG['synthetic']):
        bases[pid] = SynthBase(pid, load_obj(ND / 'mtg' / f'{pid}.obj'), rho, pool, sig_off, np.random.default_rng(SEED + i))
        log(f'synthetic base {pid} surfels {len(bases[pid].A)} leaves {len(bases[pid].uids)}')
    # calibration on the R3-1b mesh sample
    cb = bases['R3-1b']; rng = np.random.default_rng(SEED); zc = rng.normal(size=len(cb.A)); axc = rng.normal(size=(len(cb.A), 3))
    cal = []
    for s in CFG['sigma_grid_deg']:
        n2, _, _ = NM.rotate_frames(cb.n_true, cb.tu_true, cb.tv_true, np.abs(zc) * np.radians(s), axc)
        cal.append(dict(sigma_deg=s, metric_deg=NM.noise_metric(cb.X, n2)[0]))
    sig_for = {}
    for tg in CFG['noise_targets_deg']:
        best = min(cal, key=lambda c: abs(c['metric_deg'] - tg)); sig_for[tg] = best['sigma_deg']
    jdump(dict(density_per_mm2=rho, sigma_off_mm=sig_off, calibration=cal, sigma_for_target=sig_for, pool_size=len(pool['a'])), OUT / 'synthetic' / 'calibration.json')
    log(f'calibration rho={rho:.3f} sig_off={sig_off:.3f} sigma_for={sig_for}')
    rows, leaf_rows = [], []
    for i, pid in enumerate(CFG['synthetic']):
        b = bases[pid]; w = b.A * b.op
        truth = syn_run(b, b.n_true, b.tu_true, b.tv_true, b.n_true)
        for j, tg in enumerate(CFG['noise_targets_deg']):
            r = np.random.default_rng(SEED + 1000 + 10 * i + j); z = r.normal(size=len(b.A)); ax = r.normal(size=(len(b.A), 3))
            nN, nTU, nTV = NM.rotate_frames(b.n_true, b.tu_true, b.tv_true, np.abs(z) * np.radians(sig_for[tg]), ax)
            met = NM.noise_metric(b.X, nN)[0]
            runs = {'receiver_only_truth': syn_run(b, nN, nTU, nTV, b.n_true)}
            for mode in MODES:
                nr = nN if mode == 'raw' else NM.smooth(b.X, nN, w, b.sets, NM.RADIUS[mode])[0]
                runs[mode] = syn_run(b, nN, nTU, nTV, nr)
            for mode, rr in runs.items():
                errs = []
                for u, tv in truth['leaves'].items():
                    if tv > 0:
                        e = 100 * (rr['leaves'][u] / tv - 1); errs.append(abs(e))
                        leaf_rows.append(dict(plant=pid, noise_target_deg=tg, mode=mode, leaf=u, truth=tv, value=rr['leaves'][u], err_pct=e))
                ks = sorted(truth['leaves'])
                rows.append(dict(plant=pid, noise_target_deg=tg, sigma_deg=sig_for[tg], noise_metric_deg=met, mode=mode, truth_total=truth['total'], total=rr['total'],
                                 plant_bias_pct=100 * (rr['total'] / truth['total'] - 1), self_truth=truth['self'], self_mode=rr['self'], self_change=rr['self'] - truth['self'],
                                 leaf_median_abs_err_pct=float(np.median(errs)), leaf_spearman=spear([truth['leaves'][k] for k in ks], [rr['leaves'][k] for k in ks])))
            log(f'synthetic {pid} noise {tg}: ' + ', '.join(f"{r_['mode']} {r_['plant_bias_pct']:+.1f}%/{r_['leaf_median_abs_err_pct']:.1f}%" for r_ in rows[-len(runs):]))
        jdump(dict(rows=rows, leaf_rows=leaf_rows), OUT / 'synthetic' / 'rows.json')
    rule = dict(declared=CFG['rule'], per_mode={})
    for mode in MODES[1:]:
        sel = [r_ for r_ in rows if r_['mode'] == mode and r_['noise_target_deg'] == CFG['rule']['noise_deg']]
        lr = [abs(r_['err_pct']) for r_ in leaf_rows if r_['mode'] == mode and r_['noise_target_deg'] == CFG['rule']['noise_deg']]
        mb = float(np.mean([r_['plant_bias_pct'] for r_ in sel])); ml = float(np.median(lr))
        rule['per_mode'][mode] = dict(mean_plant_bias_pct=mb, pooled_leaf_median_abs_err_pct=ml, recovers=bool(abs(mb) <= CFG['rule']['plant_bias_pct'] and ml <= CFG['rule']['leaf_median_abs_err_pct']))
    rec = [m for m in MODES[1:] if rule['per_mode'][m]['recovers']]
    rule['chosen'] = rec[0] if rec else None; rule['status'] = 'recovered' if rec else 'not_recovered'
    rule['mode_for_single_radius_parts'] = rec[0] if rec else CFG['fallback_radius_mode']; rule['information_only'] = not rec
    jdump(rule, OUT / 'synthetic' / 'rule.json'); (OUT / 'synthetic' / 'DONE').write_text('ok'); log(f'RULE {json.dumps(rule)}')


# ------------------------------------------------------------------ MTG cross-check (§5)
class PlacedMesh:
    def __init__(self, objs, offset, seed, spt=16):
        names, tris = [], []
        for o in sorted(objs):
            names += [o] * len(objs[o]); tris.append(objs[o])
        t = np.concatenate(tris) / 1000.0 + offset; names = np.array(names)
        P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0]; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]; self._P = P; self._k = np.full(len(t), TRI, np.int8)
        cr = np.cross(P[:, 3:6], P[:, 6:9]); ar = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / np.maximum(2 * ar[:, None], 1e-30)
        isl = np.array([obj_leaf(o)[0] is not None for o in names]) & (ar > 0); idx = np.nonzero(isl)[0]; rng = np.random.default_rng(seed)
        u = rng.random((len(idx), spt)); v = rng.random((len(idx), spt)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
        self.centre = (t[idx, 0][:, None] + u[..., None] * P[idx, None, 3:6] + v[..., None] * P[idx, None, 6:9]).reshape(-1, 3)
        self.normal = np.repeat(nrm[idx], spt, 0); self.area = np.repeat(ar[idx] / spt, spt) * 1e6; self.area_a = self.area
        self.uid = np.repeat(np.array([obj_leaf(o)[0] for o in names[idx]]), spt)
        self.own_local = np.repeat(idx, spt); self.delta = 1e-7; self.coincident_half = True; self.pid = 'mtg'; self.pos = obs.SCENE_XY
        self.leaf = np.ones(len(self.centre), bool); self.leaf_id = np.zeros(len(self.centre), np.int64)
    def prims(self):
        return self._P, self._k


if 'mtg' in parts and (OUT / 'synthetic' / 'rule.json').exists():
    rule = json.load(open(OUT / 'synthetic' / 'rule.json')); MODE = rule['mode_for_single_radius_parts']
    u2r = json.load(open(ND / 'mtg' / 'R3-1b_uid_to_R8pc.json'))
    for pid in CFG['mtg']:
        if (OUT / 'mtg' / f'{pid}.json').exists():
            continue
        t0 = time.time(); q = obs.Placed(pid, obs.SCENE_XY); off = q.offset.copy(); obs._CACHE.clear()
        objs = load_obj(ND / 'mtg' / f'{pid}.obj'); per = {}
        tot = []
        for sd in SEEDS:
            o = PlacedMesh(objs, off, sd)
            rec = obs.trace_plants([o], [o], sd, delta=DEX, sensors=False, field=F)['mtg']
            ab = rec['Uext'] * obs.kernel(rec['parts']) * o.area; tot.append(float(ab.sum()))
            for u in np.unique(o.uid):
                per.setdefault(str(u), []).append(float(ab[o.uid == u].sum()))
        mtg_leaf = {u: float(np.mean(v)) for u, v in per.items()}
        if pid == 'R3-1b':
            agg = {}
            for u, v in mtg_leaf.items():
                k = u2r.get(f'R3-1b-{u}', u2r.get(u))
                agg[f'R8pc_{k}'] = agg.get(f'R8pc_{k}', 0.0) + v
            mtg_cmp = agg
        else:
            mtg_cmp = {f'{pid}-{u}': v for u, v in mtg_leaf.items()}
        rr = json.load(open(OUT / 'real' / f'{pid}.json'))
        out = dict(plant=pid, tags=TAGS + ['mtg_cross_check_report_only'], mode=MODE, information_only=rule['information_only'], mtg_total_leaflets=float(np.mean(tot)),
                   mtg_leaf_absorbed=mtg_leaf, compare={})
        for m in ('raw', MODE):
            sl = {r_['leaf']: r_ for r_ in rr['modes'][m]['leaves']}
            ks = sorted(k for k in mtg_cmp if k in sl and sl[k]['area_b_mm2'] > 0 and mtg_cmp[k] > 0)
            out['compare'][m] = dict(n_leaves=len(ks), spearman=spear([mtg_cmp[k] for k in ks], [sl[k]['absorbed'] for k in ks]),
                                     ratio_mtg_over_surfel_assigned=out['mtg_total_leaflets'] / rr['modes'][m]['assigned_leaf_absorbed'],
                                     ratio_mtg_over_surfel_all=out['mtg_total_leaflets'] / rr['modes'][m]['absorbed'],
                                     leaves=[dict(leaf=k, mtg=mtg_cmp[k], surfel=sl[k]['absorbed'], surfel_area_b_mm2=sl[k]['area_b_mm2']) for k in ks],
                                     unmatched_mtg=sorted(set(mtg_cmp) - set(ks)))
        out['seconds'] = round(time.time() - t0)
        jdump(out, OUT / 'mtg' / f'{pid}.json'); log(f'MTG {pid} total={out["mtg_total_leaflets"]:.1f} rho_raw={out["compare"]["raw"]["spearman"]} rho_{MODE}={out["compare"][MODE]["spearman"]}')
    log('MTG_DONE')

jdump(dict(engine_hash=HASH, wall_seconds=round(time.time() - T0), parts=sorted(parts), numba=numba.__version__, numpy=np.__version__), OUT / 'SUMMARY_K1.json')
log('K1_DONE')
