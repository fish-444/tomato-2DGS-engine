"""STATIC consistency gates S0–S3 (CONTRACT.md §5). python gates.py -> GATES.json, gates_s1_leaflets.csv, gates_s1b_intervals.csv"""
import csv, json, math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from static.tracer import Scene, trace_lamps, coincident, DISC, TRI
from static.geometry import calibrate_area_b, footprint_k
from static import light as L

W = REPO / 'build/engine_maxmass_fix_20261025_01/runs/seed72/native/working'
LAMPS = np.array([[0.0, 0.0, 0.6]]); POWER = 1200.0; INTENSITY = np.array([POWER / (4 * math.pi)])
PACKET = POWER / 300000; SEEDS = (101, 202); K = 16
noise = json.load(open(HERE / 'NOISE.json', encoding='utf-8'))
VR = noise['variance_ratio_observed_over_binomial']; VR_SE = math.sqrt(2 / (2 * noise['organs']))
SCALE = math.sqrt(VR) if abs(VR - 1) > 2 * VR_SE else 1.0
res = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), groimp_noise=dict(packet_umol_s=PACKET, variance_ratio=VR, ratio_se=VR_SE, sigma_scale=SCALE))


def gnoise(a):
    return SCALE * np.sqrt(np.maximum(a, 0) * PACKET)


# ---------- S0: synthetic flat leaf, triangles vs surfel discs
side = 0.020; z0 = 0.10; c0 = np.array([0.05, 0.0, z0])
tri = np.array([[[-1, -1, 0], [1, -1, 0], [1, 1, 0]], [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]], float) * side / 2 + c0
Pt = np.zeros((2, 14)); Pt[:, 0:3] = tri[:, 0]; Pt[:, 3:6] = tri[:, 1] - tri[:, 0]; Pt[:, 6:9] = tri[:, 2] - tri[:, 0]
# v2 (2026-10-26): v1 used arange(-side/2 + r_d, side/2 - r_d, sp) -> 49 centres spanning -9.7..+9.5 mm, an asymmetric lattice that left
# ~0.15 mm strips uncovered (GATES_v1.json). v2: 50 x 50 centred cells, footprint radius k*r_d = cell half-diagonal (full cover, <=0.08 mm overhang).
sp = 0.0004; g = -side / 2 + sp / 2 + sp * np.arange(int(round(side / sp))); r_d = sp / math.sqrt(2) / footprint_k(np.ones(1))[0]
cx, cy = np.meshgrid(g, g); cen = np.stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)], 1) + c0
nd = len(cen); kk = footprint_k(np.ones(nd))
Pd = np.zeros((nd, 14)); Pd[:, 0:3] = cen; Pd[:, 3:6] = [1, 0, 0]; Pd[:, 6:9] = [0, 1, 0]; Pd[:, 9:12] = [0, 0, 1]; Pd[:, 12] = kk * r_d; Pd[:, 13] = kk * r_d   # disc prims carry the footprint radius k*s, as Surfels.prims does (v1 passed s)
area_b, cub = calibrate_area_b(cen, Pd[:, 9:12], Pd[:, 3:6], Pd[:, 6:9], Pd[:, 12], Pd[:, 13])
sx, sy = np.meshgrid(np.linspace(-0.015, 0.015, 61), np.linspace(-0.015, 0.015, 61))
S = np.stack([sx.ravel(), sy.ravel(), np.full(sx.size, -0.01)], 1) + c0
Ns = np.tile([0, 0, 1.0], (len(S), 1)); own = np.full(len(S), -1)
sh_t, un = trace_lamps(Scene(Pt, np.full(2, TRI, np.int8), h=0.002), S, Ns, False, own, 1e-7, LAMPS, INTENSITY)
sh_d, _ = trace_lamps(Scene(Pd, np.full(nd, DISC, np.int8), h=0.001), S, Ns, False, own, 1e-7, LAMPS, INTENSITY)
int_t = 1 - sh_t.sum() / un.sum(); int_d = 1 - sh_d.sum() / un.sum()
res['S0'] = dict(intercepted_triangles=int_t, intercepted_discs=int_d, rel_diff=abs(int_d - int_t) / int_t, area_b_mm2=area_b.sum() * 1e6,
                 area_true_mm2=side * side * 1e6, area_rel_diff=abs(area_b.sum() - side * side) / (side * side), discs=nd, area_a_mm2=float((math.pi * (kk[0] * r_d) ** 2 * nd) * 1e6), version='v2')
res['S0']['pass'] = bool(res['S0']['rel_diff'] <= 0.02 and res['S0']['area_rel_diff'] <= 0.03)
print('S0', json.dumps(res['S0']), flush=True)

# ---------- load all consumed states once (S1, S1b, S2)
t0 = time.time()
verts = defaultdict(lambda: defaultdict(list))
with open(W / 'consumed.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        verts[int(r['state'])][r['organ_id']].append((int(r['vertex']), float(r['x_m']), float(r['y_m']), float(r['z_m'])))
cls, plant, sarea = defaultdict(dict), defaultdict(dict), defaultdict(dict)
with open(W / 'states.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        d = int(r['day']); cls[d][r['organ_id']] = r['class']; plant[d][r['organ_id']] = r['plant_instance_id']; sarea[d][r['organ_id']] = float(r['area_m2'])
flux = defaultdict(dict); fgross = defaultdict(dict)
with open(W / 'flux.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        i = int(r['interval']); flux[i][r['organ_id']] = float(r['absorbed_umol_s']); fgross[i][r['organ_id']] = float(r['gross_gC'])
hours = {int(r['interval']): float(r['light_hours']) for r in csv.DictReader(open(W.parent / 'inputs' / 'environment.csv', encoding='utf-8'))}
print('loaded', round(time.time() - t0), 's', flush=True)


def snapshot(state):
    tris, owner = [], []
    for oid, vs in verts[state].items():
        vs.sort(); arr = np.array([v[1:] for v in vs])
        for k in range(0, len(arr) - len(arr) % 3, 3): tris.append(arr[k:k + 3]); owner.append(oid)
    tri = np.array(tris); owner = np.array(owner)
    e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]; cr = np.cross(e1, e2); area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-30)
    P = np.zeros((len(tri), 14)); P[:, 0:3] = tri[:, 0]; P[:, 3:6] = e1; P[:, 6:9] = e2
    leaf = np.array([cls[state].get(o) == 'Leaflet' for o in owner])
    return tri, owner, area, nrm, P, leaf


COINC = {}


def static_absorbed(state, seed):
    tri, owner, area, nrm, P, leaf = snapshot(state)
    scene = Scene(P, np.full(len(P), TRI, np.int8), h=0.003)
    rng = np.random.default_rng(seed); idx = np.nonzero(leaf & (area > 0))[0]
    u = rng.random((len(idx), K)); v = rng.random((len(idx), K)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
    A = tri[idx, 0][:, None]; E1 = (tri[idx, 1] - tri[idx, 0])[:, None]; E2 = (tri[idx, 2] - tri[idx, 0])[:, None]
    R = (A + u[..., None] * E1 + v[..., None] * E2).reshape(-1, 3); N = np.repeat(nrm[idx], K, 0); w = np.repeat(area[idx] / K, K)
    own = np.repeat(idx, K); org = np.repeat(owner[idx], K)
    sh, _ = trace_lamps(scene, R, N, True, own, 1e-7, LAMPS, INTENSITY)
    # v2 (post-diagnosis, GATES_DIAGNOSIS.md): the engine's mirrored leaflet pairs overlap coplanar; GroIMP absorbs a ray there once, so the
    # GroIMP-equivalent static credit is 1/2 on coincident receivers. Raw (declared v1) values stay in out['__raw__'].
    half = coincident(scene, R, N, own)
    out = defaultdict(float); raw = defaultdict(float); COINC[state] = float((sh * w)[half].sum() / max((sh * w).sum(), 1e-30))
    for o, val, hf in zip(org, sh * w, half): out[o] += val * (0.5 if hf else 1.0); raw[o] += val
    leaf_area = defaultdict(float)
    for o, a in zip(owner[leaf], area[leaf]): leaf_area[o] += a
    return out, leaf_area, raw


# ---------- S1 at state 71
ST = 71
runs = [static_absorbed(ST, s) for s in SEEDS]
leaflets = sorted(set(runs[0][0]) | {o for o, v in flux[ST].items() if cls[ST].get(o) == 'Leaflet'})
rows = []
for o in leaflets:
    s1, s2 = runs[0][0].get(o, 0.0), runs[1][0].get(o, 0.0); sm = (s1 + s2) / 2; ss = abs(s1 - s2) / math.sqrt(2) / math.sqrt(2)
    dyn = flux[ST].get(o, 0.0); sg = float(gnoise(sm)); tot = math.sqrt(sg ** 2 + ss ** 2)
    rows.append(dict(organ_id=o, plant=plant[ST].get(o), static_seed1=s1, static_seed2=s2, static_mean=sm, dynamic=dyn, sigma_groimp=sg, sigma_static=ss,
                     z=(dyn - sm) / tot if tot > 0 else 0.0, area_m2=runs[0][1].get(o, 0.0)))
with (HERE / 'gates_s1_leaflets.csv').open('w', encoding='utf-8', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
s1 = {}
for p in sorted({r['plant'] for r in rows}):
    rp = [r for r in rows if r['plant'] == p]
    st_m = sum(r['static_mean'] for r in rp); dyn = sum(r['dynamic'] for r in rp)
    s_a = sum(r['static_seed1'] for r in rp); s_b = sum(r['static_seed2'] for r in rp)
    sg = float(gnoise(st_m)); ss = abs(s_a - s_b) / 2; tol = 2 * math.sqrt(sg ** 2 + ss ** 2)
    s1[p] = dict(static=st_m, static_seeds=[s_a, s_b], dynamic=dyn, diff=dyn - st_m, rel_diff=(dyn - st_m) / st_m, sigma_groimp=sg, sigma_static=ss,
                 tolerance_2sigma=tol, passed=bool(abs(dyn - st_m) <= tol))
lit = [r for r in rows if r['static_mean'] > 0 or r['dynamic'] > 0]
z = np.array([r['z'] for r in lit])
raw1 = {p: sum((runs[0][2].get(o, 0) + runs[1][2].get(o, 0)) / 2 for o in runs[0][2] if plant[ST].get(o) == p) for p in s1}
res['S1'] = dict(state=ST, dag=59.9, static_rule='v2 coincident receivers credited 1/2 (GATES_DIAGNOSIS.md)', static_raw_v1_per_plant=raw1,
                 coincident_light_share=COINC[ST], per_plant=s1, leaflets=len(lit), z_mean=float(z.mean()), z_sd=float(z.std()), share_within_2=float(np.mean(np.abs(z) <= 2)),
                 receivers_per_triangle=K, seeds=list(SEEDS))
res['S1']['pass'] = all(v['passed'] for v in s1.values())
print('S1', json.dumps({p: {k: round(v[k], 5) if isinstance(v[k], float) else v[k] for k in ('static', 'dynamic', 'rel_diff', 'tolerance_2sigma', 'passed')} for p, v in s1.items()}),
      'z mean %.3f sd %.3f within2 %.3f' % (res['S1']['z_mean'], res['S1']['z_sd'], res['S1']['share_within_2']), flush=True)

# ---------- S2 leaf area exactness at state 71
s2 = {}
for p in sorted({plant[ST][o] for o in runs[0][1]}):
    tri_area = sum(a for o, a in runs[0][1].items() if plant[ST][o] == p)
    st_area = sum(a for o, a in sarea[ST].items() if plant[ST].get(o) == p and cls[ST].get(o) == 'Leaflet')
    s2[p] = dict(triangle_area_m2=tri_area, state_area_m2=st_area, rel_diff=abs(tri_area - st_area) / st_area,
                 lai_pot_top=tri_area / 0.003481, lai_tray_cell=tri_area / 0.0036)
res['S2'] = dict(per_plant=s2, tolerance=1e-4, pass_=all(v['rel_diff'] <= 1e-4 for v in s2.values())); res['S2']['pass'] = res['S2'].pop('pass_')
print('S2', json.dumps(s2), flush=True)

# ---------- S3 LUE at state 71
rng = np.random.default_rng(7)
s3 = {}
h = hours[ST]
for p in sorted(s1):
    rp = [r for r in rows if r['plant'] == p and r['area_m2'] > 0]
    A = np.array([r['static_mean'] for r in rp]); ar = np.array([r['area_m2'] for r in rp])
    lue_static = L.gross_gC(A / ar, ar, h).sum() / L.absorbed_mol(A / ar, ar, h).sum()
    draws = []
    for _ in range(400):
        Ad = PACKET * rng.poisson(np.maximum(A, 0) / PACKET)
        m = L.absorbed_mol(Ad / ar, ar, h).sum()
        if m > 0: draws.append(L.gross_gC(Ad / ar, ar, h).sum() / m)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    dyn_g = sum(v for o, v in fgross[ST].items() if plant[ST].get(o) == p)
    dyn_a = sum(v for o, v in flux[ST].items() if plant[ST].get(o) == p)
    lue_dyn = dyn_g / (dyn_a * h * 3600 / 1e6)
    s3[p] = dict(lue_static_gC_mol=float(lue_static), band_2sigma=[float(lo), float(hi)], lue_dynamic_gC_mol=float(lue_dyn), passed=bool(lo <= lue_dyn <= hi),
                 dynamic_gross_gC=dyn_g, dynamic_absorbed_umol_s=dyn_a)
res['S3'] = dict(per_plant=s3, hours=h, pass_=all(v['passed'] for v in s3.values())); res['S3']['pass'] = res['S3'].pop('pass_')
print('S3', json.dumps(s3), flush=True)

# ---------- S1b bias over all intervals (one seed)
ib = []
for st in sorted(flux):
    if st not in verts: continue
    out, _, raw = static_absorbed(st, SEEDS[0])
    for p in ('p1', 'p2'):
        a_s = sum(v for o, v in out.items() if plant[st].get(o) == p); a_d = sum(v for o, v in flux[st].items() if plant[st].get(o) == p and cls[st].get(o) == 'Leaflet')
        ib.append(dict(interval=st, plant=p, static=a_s, dynamic=a_d, static_raw_v1=sum(v for o, v in raw.items() if plant[st].get(o) == p), coincident_light_share=COINC[st]))
with (HERE / 'gates_s1b_intervals.csv').open('w', encoding='utf-8', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(ib[0])); wr.writeheader(); wr.writerows(ib)
S_ = sum(r['static'] for r in ib); D_ = sum(r['dynamic'] for r in ib)
sig = float(gnoise(S_)) / S_ if S_ > 0 else 0
res['S1b'] = dict(intervals=len({r['interval'] for r in ib}), sum_static_raw_v1=sum(r['static_raw_v1'] for r in ib), sum_static=S_, sum_dynamic=D_, ratio=D_ / S_ if S_ else None, sigma_pooled_rel=sig,
                  pass_=bool(S_ > 0 and abs(D_ / S_ - 1) <= 2 * sig))
res['S1b']['pass'] = res['S1b'].pop('pass_')
print('S1b', json.dumps(res['S1b']), flush=True)
res['all_pass'] = all(res[k]['pass'] for k in ('S0', 'S1', 'S1b', 'S2', 'S3'))
res['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(res, open(HERE / 'GATES.json', 'w', encoding='utf-8'), indent=1, default=float)
print('ALL PASS' if res['all_pass'] else 'NOT ALL PASS', {k: res[k]['pass'] for k in ('S0', 'S1', 'S1b', 'S2', 'S3')})
