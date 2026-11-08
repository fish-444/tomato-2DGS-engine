"""Parent §5 synthetic check (CONTRACT §4): known-area meshes (11 static MTG `_02` meshes + engine-law T of R3-1b) sampled into surfels (normals-test generator, seed 20261105);
a surfel is 'reconstructed' if seen in >= 2 of 8 views (elevation 45 deg, azimuth 0..315 step 45): unoccluded (black discs, own excluded, tmin 2.4 mm) and |n.d| >= 0.17.
LAI_low = seen leaflet area / ground; LAI_up = + fill by arm R (real 14-plant p90 table) or arm B (pilot f), cap 2x, >= 100 seen surfels; hit = true LAI inside [low, up]. Local, small."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import numba as nb
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
sys.path.insert(0, str(B / 'engine_static_tau_20261031_01/kaggle_data'))
from static.tracer import Scene, occluded, DISC
POOL = np.load(B / 'engine_static_normals_v2_20261104_01/kaggle_flat/pool.npz'); RHO = float(POOL['density']); SIG = float(POOL['sigma_off'])
LV = json.load(open(HERE / 'leaves.json')); F = LV['f_pilot']; T90 = {k: v['p90'] for k, v in LV['armR_p90_table'].items()}; GROUND = 59.0 * 59.0
S = B / 'mtg_build_v2_clicks_20261103_02/static'


@nb.njit(parallel=True)
def seen_views(R, N, own, D, P, kind, gmin, h, dims, start, items):
    n = R.shape[0]; cnt = np.zeros(n, np.int64)
    for r in nb.prange(n):
        for j in range(D.shape[0]):
            c = abs(N[r, 0] * D[j, 0] + N[r, 1] * D[j, 1] + N[r, 2] * D[j, 2])
            if c < 0.17:
                continue
            if not occluded(R[r, 0], R[r, 1], R[r, 2], D[j, 0], D[j, 1], D[j, 2], 0.0024, 10.0, own[r], P, kind, gmin, h, dims, start, items):
                cnt[r] += 1
    return cnt


def load_obj(path):
    V, objs, cur = [], {}, None
    for line in open(path, encoding='utf-8'):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('o '):
            cur = line.split(None, 1)[1].strip(); objs.setdefault(cur, [])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            for j in range(1, len(idx) - 1):
                objs[cur].append((idx[0], idx[j], idx[j + 1]))
    V = np.array(V)
    return {o: V[np.array(f)] for o, f in objs.items() if f}


def sample(objs, leaf_of, rng):
    """objs: name -> triangles (mm); leaf_of(name) -> leaf key or None. Returns centres (mm), normals, area per surfel, leaf key, a/b discs (m)."""
    C, N, A, L = [], [], [], []
    for o in sorted(objs):
        tri = objs[o]; e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]; cr = np.cross(e1, e2); ar = 0.5 * np.linalg.norm(cr, axis=1); ok = ar > 0
        tri, e1, e2, cr, ar = tri[ok], e1[ok], e2[ok], cr[ok], ar[ok]
        if not len(ar):
            continue
        Ao = ar.sum(); n = max(1, int(round(RHO * Ao))); ti = rng.choice(len(ar), n, p=ar / Ao); u = rng.random(n); v = rng.random(n); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
        nr = cr[ti] / (2 * ar[ti, None]); C.append(tri[ti, 0] + u[:, None] * e1[ti] + v[:, None] * e2[ti] + nr * rng.normal(0, SIG, (n, 1))); N.append(nr); A.append(np.full(n, Ao / n)); L += [leaf_of(o)] * n
    C = np.concatenate(C); N = np.concatenate(N); A = np.concatenate(A); k = rng.integers(len(POOL['a']), size=len(C))
    return C, N, A, np.array(L, dtype=object), POOL['a'][k], POOL['b'][k]


def lai_range(C, A, L, seen, rank_of):
    true_area = sum(A[L == k].sum() for k in set(L) if k is not None); low_area = 0.0; addR = addB = 0.0
    for k in sorted({x for x in L if x is not None}):
        m = (L == k) & seen; low_area += A[m].sum()
        if m.sum() < 100 or rank_of(k) is None:
            continue
        X = C[m]; w = A[m]; c = (X * w[:, None]).sum(0) / w.sum(); U = np.linalg.eigh(np.cov((X - c).T, aweights=w))[1][:, ::-1]; P = (X - c) @ U
        ext = np.percentile(P, 99, axis=0) - np.percentile(P, 1, axis=0); Arec = w.sum()
        AR = T90.get(rank_of(k), None); AB = ext[0] * ext[1] * F
        if AR is not None:
            addR += min(max(0.0, AR - Arec), Arec)
        addB += min(max(0.0, AB - Arec), Arec)
    return true_area / GROUND, low_area / GROUND, (low_area + addR) / GROUND, (low_area + addB) / GROUND


els = np.radians(45.0); D = np.array([[math.cos(els) * math.cos(a), math.cos(els) * math.sin(a), math.sin(els)] for a in np.radians(np.arange(0, 360, 45))])
rows = []
cases = []
for i, pid in enumerate(['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07']):
    objs = load_obj(S / f'mtg_{pid}_engine.obj'); kk = {l['uid'].split('-')[-1]: l['k'] for l in json.load(open(S / f'mtg_{pid}_spec.json', encoding='utf-8'))['leaves']}
    cases.append((f'MTG {pid}', objs, lambda o: o.split('__')[0] if '__' in o and o.startswith('L') else None, (lambda kk_: (lambda k: f'main_{kk_[k]}' if k in kk_ and f'main_{kk_[k]}' in T90 else None))(kk), i))
# engine-law T (R3-1b crosswalk), leaf = R8_pc leaf id; rank = the pilot's real rank of that leaf
rowsT = [r for r in csv.reader(l for l in open(B / 'engine_disc_vs_tri_20261101_01/T_triangles.csv', encoding='utf-8') if not l.startswith('#'))][1:]
triT = np.array([[float(x) for x in r[6:]] for r in rowsT]).reshape(-1, 3, 3) * 1000; orgT = np.array([r[0] for r in rowsT]); clsT = np.array([r[1] for r in rowsT]); lfT = np.array([int(r[2]) for r in rowsT])
objsT = {f'{o}|{lfT[orgT == o][0]}|{clsT[orgT == o][0]}': triT[orgT == o] for o in sorted(set(orgT.tolist()))}
pr = {l['label']: l['rank'] for l in LV['leaves'] if l['plant'] == 'R3-1b'}
cases.append(('engine-law T (R3-1b)', objsT, lambda o: (int(o.split('|')[1]) if o.split('|')[2] == 'Leaflet' else None), lambda k: f'main_{pr[k]}' if pr.get(k) and f'main_{pr[k]}' in T90 else None, 11))
for name, objs, leaf_of, rank_of, i in cases:
    rng = np.random.default_rng(20261105 + i); C, N, A, L, a, b = sample(objs, leaf_of, rng)
    t = rng.normal(size=(len(C), 3)); t -= (t * N).sum(1, keepdims=True) * N; tu = t / np.linalg.norm(t, axis=1, keepdims=True); tv = np.cross(N, tu)
    Cm = C / 1000.0; P = np.zeros((len(C), 14)); P[:, 0:3] = Cm; P[:, 3:6] = tu; P[:, 6:9] = tv; P[:, 9:12] = N; P[:, 12] = a; P[:, 13] = b
    sc = Scene(P, np.full(len(C), DISC, np.int8), h=0.003); cnt = seen_views(np.ascontiguousarray(Cm), np.ascontiguousarray(N), np.arange(len(C), dtype=np.int64), D, sc.P, sc.kind, sc.gmin, sc.h, sc.dims, sc.start, sc.items)
    seen = cnt >= 2; lt, lo, uR, uB = lai_range(C, A, L, seen, rank_of)
    leafm = np.array([x is not None for x in L])
    rows.append(dict(case=name, surfels=len(C), seen_share_leaf_area=round(float(A[leafm & seen].sum() / A[leafm].sum()), 4), LAI_true=round(lt, 4), LAI_low=round(lo, 4), LAI_up_R=round(uR, 4), LAI_up_B=round(uB, 4),
                     hit_R=bool(lo <= lt <= uR), hit_B=bool(lo <= lt <= uB), up_R_over_true=round(uR / lt, 4), up_B_over_true=round(uB / lt, 4)))
    print(rows[-1], flush=True)
with open(HERE / 'synthetic_check.csv', 'w', encoding='utf-8', newline='') as f:
    f.write('# lai_range; not_validation; removal = seen in >= 2 of 8 views (elev 45 deg), |n.d| >= 0.17, black discs, own excluded; generator of the normals tests (seed 20261105); arm R = real 14-plant p90 table, arm B f = pilot median; cap 2x\n')
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('hit rate R', sum(r['hit_R'] for r in rows), '/', len(rows), 'B', sum(r['hit_B'] for r in rows), '/', len(rows))
