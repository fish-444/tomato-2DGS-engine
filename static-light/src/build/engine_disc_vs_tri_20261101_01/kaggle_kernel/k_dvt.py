"""Disc (D) vs engine triangle (T) representation of R3-1b, same scene, same tracer (engine_disc_vs_tri_20261101_01/CONTRACT.md + Amendment 1).
Kaggle CPU. Inputs: static-tau-bundle-20261031 (v1 code copy + tau additions, labels), static-nine-bundle-20261028 (surfels, field, refs),
disc-vs-tri-20261101 (T geometry), tomato-h-input (GroIMP 2.2.1)."""
import csv, glob, hashlib, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
DVT = Path(glob.glob('/kaggle/input/**/T_meta.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT))
import numpy as np
from static import tau_runs as TR, obs, light as L
from static.tau_obs import build_scene, trace_plants_tau
from static.tracer import Scene, TRI, trace_lamps, coincident

T0 = time.time(); LOG = []
def log(s):
    LOG.append(f'{time.time() - T0:7.0f}s {s}'); print(LOG[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOG))
TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; F = CFG['field']
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
SEEDS = TR.SEEDS; LAMP_P = 1200.0; LAMP_H = 0.6; GROIMP_RAYS = 30_000_000; GROIMP_SEEDS = (11, 12)
RES = dict(labels=['disc_vs_triangle', 'relative_light', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1', 'R3-1b_unaccepted', 'engine_law_leaflets'], engine_hash=HASH)

# ------------------------------------------------------------------ D and the reproduction gate
q = TR.placed('R3-1b'); builtD = build_scene([q])
runsD = {'v1': TR.trace([q], [q], 'v1', builtD)['R3-1b']}
ref = np.load(NINE_ROOT / 'r31b_ref.npz'); diff = float(np.abs(TR.kmean(runsD['v1']) - ref['relative_light']).max())
log(f'REPRO D v1 d_ex 2.4 vs local v1 max_diff={diff}')
RES['repro'] = dict(max_abs_diff_vs_local_v1=diff, passed=diff == 0.0)
if diff != 0.0:
    json.dump(RES, open(OUT / 'results.json', 'w'), indent=1, default=float); sys.exit(0)
runsD['tau0.07'] = TR.trace([q], [q], 'tau0.07', builtD)['R3-1b']
for dex in (0.0015, 0.005):                                              # d_ex sensitivity (D only)
    out = {}
    for sd in SEEDS:
        out[sd] = trace_plants_tau([q], [q], sd, CFG['eps'], 0.0, 0.0, dex, F, use_v1=True, built=builtD)['R3-1b']
    runsD[f'v1_dex{dex * 1000:g}'] = out
log('D runs done')


class NoStem:
    """Occluder view of D without stem surfels (axis_kind 1); receivers = all D surfels (own index remapped, stem receivers own = -1)."""
    def __init__(self, q):
        self.pid = 'R3-1b'; keep = q.axis != 1; self.keep = keep; self._q = q
        P, k = q.prims(); self._P, self._k = P[keep], k[keep]; self.tau = q.tau[keep]
        local = np.cumsum(keep) - 1; self.own_local = np.where(keep, local, -1)
        self.centre, self.normal, self.delta = q.centre, q.normal, q.delta
    def prims(self):
        return self._P, self._k


ns = NoStem(q); builtNS = build_scene([ns])
runsD['v1_nostem'] = {sd: trace_plants_tau([ns], [ns], sd, CFG['eps'], 0.0, 0.0, CFG['dex'], F, use_v1=True, built=builtNS)['R3-1b'] for sd in SEEDS}
log('D no-stem done')


# ------------------------------------------------------------------ T
def read_T(name):
    rows = [r for r in csv.reader(l for l in open(DVT / name, encoding='utf-8') if not l.startswith('#'))][1:]
    tri = np.array([[float(x) for x in r[6:]] for r in rows]).reshape(-1, 3, 3)
    return dict(tri=tri + q.offset, organ=np.array([r[0] for r in rows]), cls=np.array([r[1] for r in rows]), leaf=np.array([int(r[2]) for r in rows]))


class PlacedT:
    """T triangles at D's placement. Receivers: `spt` points per leaflet triangle (sample seed = trace seed), weights area/spt; delta 1e-7;
    coincident half credit (gate S1 v2). tau: leaflets 0.07, rachis / petiole 0."""
    def __init__(self, T, spt, seed):
        self.pid = 'T'; t = T['tri']; self.T = T
        P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0]; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]; self._P = P; self._k = np.full(len(t), TRI, np.int8)
        isl = T['cls'] == 'Leaflet'; self.tau = np.where(isl, CFG['tau_leaf'], 0.0)
        cr = np.cross(P[:, 3:6], P[:, 6:9]); ar = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / np.maximum(2 * ar[:, None], 1e-30)
        self.tri_area = ar; self.tri_normal = nrm
        idx = np.nonzero(isl)[0]; rng = np.random.default_rng(seed)
        u = rng.random((len(idx), spt)); v = rng.random((len(idx), spt)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
        R = (t[idx, 0][:, None] + u[..., None] * P[idx, None, 3:6] + v[..., None] * P[idx, None, 6:9]).reshape(-1, 3)
        self.centre = R; self.normal = np.repeat(nrm[idx], spt, 0); self.A = np.repeat(ar[idx] / spt, spt) * 1e6     # mm2 like D's q.A
        self.own_local = np.repeat(idx, spt); self.delta = 1e-7; self.leafid = np.repeat(T['leaf'][idx], spt); self.organ = np.repeat(T['organ'][idx], spt)
    def prims(self):
        return self._P, self._k


def trace_T(T, spt, model):
    recs, objs = {}, {}
    for sd in SEEDS:
        o = PlacedT(T, spt, sd); built = build_scene([o])
        if model == 'v1':
            r = trace_plants_tau([o], [o], sd, CFG['eps'], 0.0, 0.0, CFG['dex'], F, use_v1=True, built=built)['T']
        else:
            r = trace_plants_tau([o], [o], sd, CFG['eps'], 0.0, 0.0, CFG['dex'], F, tau_scale=1.0, built=built)['T']
        half = coincident(built[0], o.centre, o.normal, o.own_local); r['parts'] = r['parts'].copy(); r['parts'][half] *= 0.5; r['half'] = half
        recs[sd] = r; objs[sd] = o
    return recs, objs


z0 = np.percentile(1000 * q.centre[:, 2], 0.5); TOP = np.percentile(q.h, 99.5)


def thirds(z_mm):
    h = z_mm - z0
    return {'lower': h < TOP / 3, 'middle': (h >= TOP / 3) & (h < 2 * TOP / 3), 'upper': h >= 2 * TOP / 3}


def summary(recs, objs, kind):
    """kind 'D_all', 'D_blades' or 'T'. absorbed (a = 1) = Uext K A, relative units x mm2."""
    per = []
    for sd in SEEDS:
        r = recs[sd]; K = obs.kernel(r['parts']); Ku = obs.kernel(r['parts'], shaded=False); U = r['Uext']
        if kind == 'T':
            o = objs[sd]; A = o.A; leaf = o.leafid; zc = 1000 * o.centre[:, 2]; sel = np.ones(len(A), bool)
        else:
            A = q.A; leaf = np.where(q.axis == 0, q.r8, 0); zc = 1000 * q.centre[:, 2]; sel = np.ones(len(A), bool) if kind == 'D_all' else (leaf >= 1)
        ab = U * K * A; abu = U * Ku * A
        d = dict(total=ab[sel].sum(), unshaded=abu[sel].sum(), self=1 - ab[sel].sum() / abu[sel].sum(), area=A[sel].sum(), mean_light=ab[sel].sum() / A[sel].sum())
        for k, m in thirds(zc).items():
            m = m & sel; d[f'{k}_total'] = ab[m].sum(); d[f'{k}_area'] = A[m].sum(); d[f'{k}_self'] = 1 - ab[m].sum() / max(abu[m].sum(), 1e-30)
        for Lf in range(1, 9):
            m = leaf == Lf; d[f'leaf{Lf}_total'] = ab[m].sum(); d[f'leaf{Lf}_area'] = A[m].sum(); d[f'leaf{Lf}_light'] = ab[m].sum() / max(A[m].sum(), 1e-30)
            d[f'leaf{Lf}_self'] = 1 - ab[m].sum() / max(abu[m].sum(), 1e-30)
        per.append(d)
    return {k: TR.pm([p[k] for p in per]) for k in per[0]}


S = {}
S['D_all|v1'] = summary(runsD['v1'], None, 'D_all'); S['D_blades|v1'] = summary(runsD['v1'], None, 'D_blades')
S['D_all|tau0.07'] = summary(runsD['tau0.07'], None, 'D_all'); S['D_blades|tau0.07'] = summary(runsD['tau0.07'], None, 'D_blades')
for k in ('v1_dex1.5', 'v1_dex5', 'v1_nostem'):
    S[f'D_blades|{k}'] = summary(runsD[k], None, 'D_blades'); S[f'D_all|{k}'] = summary(runsD[k], None, 'D_all')
TT = {'T': (read_T('T_triangles.csv'), 16), 'T_diamond': (read_T('T_diamond_triangles.csv'), 64)}
Tobj = {}
for name, (T, spt) in TT.items():
    for model in ('v1', 'tau0.07'):
        t = time.time(); recs, objs = trace_T(T, spt, model); S[f'{name}|{model}'] = summary(recs, objs, 'T'); Tobj[(name, model)] = (recs, objs)
        log(f'{name} {model} receivers {len(objs[SEEDS[0]].centre)} half {float(recs[SEEDS[0]]["half"].mean()):.4f} overflow {int(sum(recs[sd]["novf"].sum() for sd in SEEDS))} {time.time() - t:.0f} s')
RES['summary'] = S

# ------------------------------------------------------------------ decomposition (b) orientation, (c) flattening
def up(n):
    return np.where(n[:, 2:3] < 0, -n, n)


dec = {}
nD = up(q.normal)
for Lf in range(1, 9):
    m = (q.axis == 0) & (q.r8 == Lf); w = q.A[m]
    mD = (nD[m] * w[:, None]).sum(0); mD /= np.linalg.norm(mD)
    dev = np.degrees(np.arccos(np.clip(np.abs(nD[m] @ mD), 0, 1)))
    row = dict(D_mean_normal=mD.tolist(), D_inclination_deg=float(np.degrees(np.arccos(abs(mD[2])))), D_share_area_dev_gt15=float(w[dev > 15].sum() / w.sum()),
               D_area_weighted_mean_dev_deg=float((dev * w).sum() / w.sum()))
    for name, (T, _) in TT.items():
        o = Tobj[(name, 'v1')][1][SEEDS[0]]; sel = (T['cls'] == 'Leaflet') & (T['leaf'] == Lf); wt = o.tri_area[sel]; nT = up(o.tri_normal[sel])
        mT = (nT * wt[:, None]).sum(0); mT /= np.linalg.norm(mT); devT = np.degrees(np.arccos(np.clip(np.abs(nT @ mT), 0, 1)))
        row[f'{name}_mean_normal'] = mT.tolist(); row[f'{name}_inclination_deg'] = float(np.degrees(np.arccos(abs(mT[2]))))
        row[f'{name}_angle_to_D_deg'] = float(np.degrees(np.arccos(np.clip(abs(mT @ mD), 0, 1)))); row[f'{name}_share_area_dev_gt15'] = float(wt[devT > 15].sum() / wt.sum())
    dec[f'leaf{Lf}'] = row
devall = np.full(len(q.A), np.nan)
for Lf in range(1, 9):
    m = (q.axis == 0) & (q.r8 == Lf); mD = np.array(dec[f'leaf{Lf}']['D_mean_normal']); devall[m] = np.degrees(np.arccos(np.clip(np.abs(nD[m] @ mD), 0, 1)))
RES['decomposition'] = dec
np.savez_compressed(OUT / 'per_surfel_D.npz', ply_row=q.s.index, xyz=q.centre - q.offset, area_b_mm2=q.A, axis_kind=q.axis, leaf_R8pc=q.r8, dev_from_leaf_mean_deg=devall,
                    K_v1=TR.kmean(runsD['v1']), K_tau=TR.kmean(runsD['tau0.07']), Uext=np.mean([runsD['v1'][sd]['Uext'] for sd in SEEDS], 0))
log('decomposition done')

# ------------------------------------------------------------------ engine-lamp closure: static D, static T, GroIMP T
lamp = np.array([[obs.SCENE_XY[0], obs.SCENE_XY[1], obs.MEDIUM_Z + LAMP_H]]); I = np.array([LAMP_P / (4 * math.pi)])
EL = {}
sceneD = builtD[0]
for sd in SEEDS[:1]:
    sh, un = trace_lamps(sceneD, q.centre, q.normal, True, q.own_local, CFG['dex'], lamp, I)
abD, abDu = sh * q.A * 1e-6, un * q.A * 1e-6; leafD = np.where(q.axis == 0, q.r8, 0)
EL['D_all'] = dict(total=float(abD.sum()), self=float(1 - abD.sum() / abDu.sum())); EL['D_blades'] = dict(total=float(abD[leafD >= 1].sum()), self=float(1 - abD[leafD >= 1].sum() / abDu[leafD >= 1].sum()))
for Lf in range(1, 9):
    EL['D_blades'][f'leaf{Lf}'] = float(abD[leafD == Lf].sum())
for name, (T, spt) in TT.items():
    vals = []
    for sd in SEEDS:
        o = PlacedT(T, spt, sd); sc = Scene(o._P, o._k, h=0.003)
        sh, un = trace_lamps(sc, o.centre, o.normal, True, o.own_local, 1e-7, lamp, I)
        half = coincident(sc, o.centre, o.normal, o.own_local); sh = sh.copy(); sh[half] *= 0.5
        ab = sh * o.A * 1e-6; d = dict(total=float(ab.sum()), self=float(1 - ab.sum() / (un * o.A * 1e-6).sum()))
        for Lf in range(1, 9):
            d[f'leaf{Lf}'] = float(ab[o.leafid == Lf].sum())
        vals.append(d)
    EL[f'{name}_static'] = {k: TR.pm([v[k] for v in vals]) for k in vals[0]}
log('engine-lamp static done')

core = sorted(glob.glob('/kaggle/input/**/core.jar', recursive=True))
GR = {}
if core:
    gdir = OUT / 'groimp_install'; shutil.copytree(os.path.dirname(os.path.realpath(core[0])), gdir, dirs_exist_ok=True)
    RES['groimp_core_sha256'] = hashlib.sha256((gdir / 'core.jar').read_bytes()).hexdigest()
    for name, (T, _) in TT.items():
        wd = OUT / f'groimp_{name}'; wd.mkdir(exist_ok=True)
        organs = sorted(set(T['organ'].tolist())); oid = {o: i for i, o in enumerate(organs)}
        with open(wd / 'mesh.csv', 'w') as fh:
            for o in organs:
                tt = T['tri'][T['organ'] == o]; fh.write(str(oid[o]) + ',' + ','.join('%.9g' % v for v in tt.reshape(-1)) + '\n')
        rgg = """// disc_vs_triangle GroIMP light on T (generated). BLACK surfaces, one MeshNode per organ, engine lamp.
import java.io.*;
import de.grogra.xl.util.FloatList;
module Surf(int sid) extends MeshNode;
module Lamp extends LightNode;
module World extends Node;
LightModel lm = new LightModel(%d, 1);
protected void init() [ Axiom ==> World [ Translate(%gf, %gf, %gf) Lamp().(setLight(new PointLight().(setPower(%gf)))) ]; ]
public void measure() throws Exception {
    println("PROBE_START");
    try {
    BufferedReader r = new BufferedReader(new FileReader("%s")); String line;
    while ((line = r.readLine()) != null) { String[] v = line.split(","); FloatList f = new FloatList();
        for (int k = 1; k < v.length; k++) f.add(Float.parseFloat(v[k]));
        final int id = Integer.parseInt(v[0]); final FloatList ff = f;
        [ w:World ==> w [Surf(id).(setPolygons(getMesh(ff).getPolygons()),setShader(BLACK))]; ] derive(); }
    r.close();
    int[] seeds = new int[]{%s};
    for (int s : seeds) { lm.setSeed(s); lm.compute();
        for (Surf x : (* Surf *)) println("RESULT\\t" + s + "\\t" + x.sid + "\\t" + lm.getAbsorbedPower(x).integrate() + "\\t" + lm.getHitCount(x)); }
    } catch (Throwable e) { println("PROBE_ERROR " + e); StringWriter sw = new StringWriter(); e.printStackTrace(new PrintWriter(sw)); println(sw.toString()); }
    println("PROBE_DONE");
}
""" % (GROIMP_RAYS, lamp[0, 0], lamp[0, 1], lamp[0, 2], LAMP_P, str(wd / 'mesh.csv'), ','.join(map(str, GROIMP_SEEDS)))
        (wd / 'Model.rgg').write_text(rgg, encoding='utf-8'); glog = wd / 'groimp.log'; t = time.time()
        env = dict(os.environ, JDK_JAVA_OPTIONS='-Djava.awt.headless=true')
        with glog.open('wb') as stream:
            p = subprocess.Popen(['java', '-Dfile.encoding=UTF-8', '-jar', str(gdir / 'core.jar'), '--headless', '-a', 'cli'], stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', cwd=str(gdir), env=env)
            def wait(pred, tmo=5400):
                end = time.time() + tmo
                while time.time() < end:
                    txt = glog.read_text(encoding='utf-8', errors='replace')
                    if pred(txt): return txt
                    if p.poll() is not None: break
                    time.sleep(1)
                raise RuntimeError('GroIMP stalled')
            try:
                wait(lambda s: '[base]>>' in s, 600); p.stdin.write(f'$cd {wd}\n$open Model.rgg\n'); p.stdin.flush()
                wait(lambda s: '[Model]>>' in s, 600); time.sleep(2); p.stdin.write('measure\n'); p.stdin.flush()
                txt = wait(lambda s: 'PROBE_DONE' in s or 'Command not found' in s)
                ok = True
            except Exception as e:
                txt = glog.read_text(encoding='utf-8', errors='replace'); ok = False; log(f'GroIMP {name} failed: {e}')
            finally:
                if p.poll() is None: p.stdin.close(); p.kill(); p.wait()
        got = re.findall(r'RESULT\t(\d+)\t(\d+)\t([0-9.eE+-]+)\t(\d+)', txt)
        leafof = {oid[o]: int(T['leaf'][T['organ'] == o][0]) for o in organs}; isl = {oid[o]: bool((T['cls'][T['organ'] == o] == 'Leaflet')[0]) for o in organs}
        per = {}
        for s, sid, a, h in got:
            d = per.setdefault(int(s), dict(total=0.0, hits=0, **{f'leaf{Lf}': 0.0 for Lf in range(1, 9)}))
            if isl[int(sid)]:
                d['total'] += float(a); d['hits'] += int(h); d[f'leaf{leafof[int(sid)]}'] += float(a)
        GR[name] = dict(ok=ok and len(per) == len(GROIMP_SEEDS), seconds=round(time.time() - t), rays=GROIMP_RAYS, seeds=per,
                        mean={k: TR.pm([per[s][k] for s in per]) for k in (per[GROIMP_SEEDS[0]] if per else {})},
                        packet_sigma_total=math.sqrt(np.mean([per[s]['total'] for s in per]) * LAMP_P / GROIMP_RAYS) if per else None)
        log(f'GroIMP {name} ok={GR[name]["ok"]} {GR[name]["seconds"]} s')
else:
    log('GroIMP core.jar not found')
EL['groimp'] = GR; RES['engine_lamp'] = EL
RES['seconds'] = round(time.time() - T0)
json.dump(RES, open(OUT / 'results.json', 'w'), indent=1, default=float)
shutil.rmtree(OUT / 'groimp_install', ignore_errors=True); shutil.rmtree(OUT / 'numba_cache', ignore_errors=True)
log('DVT_DONE')
