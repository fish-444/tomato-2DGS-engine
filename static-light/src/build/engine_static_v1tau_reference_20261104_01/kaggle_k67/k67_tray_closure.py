"""K6 + K7 of the v1_tau reference table (engine_static_v1tau_reference_20261104_01/CONTRACT.md §2).
K6: tray 2x2 (n_04, t2_02, n_01, t2_05; assumed layout) at pitch 60 / 120 mm: all / alone / none absorbed light for v1 and v1_tau -> self / neighbour / total loss. Gate G0c vs the stored tray baseline.
K7: static v1 / v1_tau vs GroIMP on the engine-law leaflet mesh T of R3-1b with the engine lamp (point 1200 at 0.6 m): GroIMP black shader (reproduction of the earlier closure) and
the engine's leaf shader Phong(diffuse = rho, transparency = tau). Tau bundle code unchanged; new code only here. No secrets."""
import csv, glob, hashlib, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

OUT = Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUT / 'numba_cache'))
TAU_ROOT = Path(glob.glob('/kaggle/input/**/tau_config.json', recursive=True)[0]).parent
NINE_ROOT = Path(glob.glob('/kaggle/input/**/run2_ref.json', recursive=True)[0]).parent
REF = Path(glob.glob('/kaggle/input/**/tray_scenarios_tau.json', recursive=True)[0]).parent
DVT = Path(glob.glob('/kaggle/input/**/T_meta.json', recursive=True)[0]).parent
sys.path.insert(0, str(TAU_ROOT))
import numpy as np
import numba as nb
from static import tau_runs as TR, obs, light as L
from static.tau_obs import build_scene, trace_plants_tau
from static.tracer import TRI, Scene, trace_lamps, coincident, transmit, MAXH

T0 = time.time(); LOGL = []
def log(s):
    LOGL.append(f'{time.time() - T0:7.0f}s {s}'); print(LOGL[-1], flush=True); (OUT / 'progress.log').write_text('\n'.join(LOGL))
def jdump(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), indent=1, default=float)
TR.setup(NINE_ROOT, TAU_ROOT); CFG = TR.CFG; SEEDS = TR.SEEDS; A_MAIN = CFG['a_main']; EPS = CFG['eps']
HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((TAU_ROOT / 'static').glob('*.py'))}
RES = dict(labels=['light_model_v1_tau', 'assumed_leaf_optics', 'relative_light', 'conditional_on_unvalidated_scale'], engine_hash=HASH)

# ------------------------------------------------------------------ K6 tray attribution
ref = json.load(open(REF / 'tray_scenarios_tau.json'))['raw']
TRAY = TR.TRAY; k6 = {}; gate = []
for pitch in (0.060, 0.120):
    qs = [TR.placed(pid, pos) for pid, pos in zip(TRAY, TR.tray_positions(pitch))]; built_all = build_scene(qs); key = f'{int(round(pitch * 1000))}'; k6[key] = {}
    for model, rkey in (('v1', 'v1'), ('tau0.07', 'tau0.07')):
        t0 = time.time(); r_all = TR.trace(qs, qs, model, built_all); vals = {}
        for q in qs:
            allt = float(np.mean([(r_all[q.pid][sd]['Uext'] * obs.kernel(r_all[q.pid][sd]['parts']) * q.A).sum() for sd in SEEDS]))
            built_q = build_scene([q]); r_q = TR.trace([q], [q], model, built_q)[q.pid]
            alone = float(np.mean([(r_q[sd]['Uext'] * obs.kernel(r_q[sd]['parts']) * q.A).sum() for sd in SEEDS]))
            none = float(np.mean([(r_q[sd]['Uext'] * obs.kernel(r_q[sd]['parts'], shaded=False) * q.A).sum() for sd in SEEDS]))
            vals[q.pid] = dict(all=allt, alone=alone, none=none, self_shading_loss=1 - alone / none, neighbour_shading_loss=1 - allt / alone, total_loss=1 - allt / none)
            if pitch == 0.060:
                gate.append(abs(allt / ref[rkey]['baseline'][q.pid] - 1))
        tot = {k: sum(v[k] for v in vals.values()) for k in ('all', 'alone', 'none')}
        vals['tray_total'] = dict(tot, self_shading_loss=1 - tot['alone'] / tot['none'], neighbour_shading_loss=1 - tot['all'] / tot['alone'], total_loss=1 - tot['all'] / tot['none'])
        k6[key][model] = vals; log(f'K6 pitch {key} {model} {time.time() - t0:.0f} s')
RES['K6_tray'] = k6; RES['G0c'] = dict(max_rel_diff_vs_stored_baseline_all=float(max(gate)), n=len(gate), passed=bool(max(gate) <= 1e-9))
log(f'G0c {RES["G0c"]}'); jdump(RES, OUT / 'k67_results.json')
if not RES['G0c']['passed']:
    jdump(dict(G0c_failed=RES['G0c']), OUT / 'G0c_FAILED.json'); sys.exit(0)


# ------------------------------------------------------------------ K7 closure on T with the engine lamp
@nb.njit(cache=True, parallel=True)
def point_lamps_tau(R, N, two_sided, own, tmin, lamps, intensity, tau_prim, eps, P, kind, gmin, h, dims, start, items):
    """`point_lamps` with `occluded` replaced by `transmit` (v1_tau); returns shaded, unshaded irradiance and overflow counts."""
    n = R.shape[0]; shaded = np.zeros(n); unshaded = np.zeros(n); novf = np.zeros(n, np.int64)
    for r in nb.prange(n):
        bt = np.empty(MAXH); bi = np.empty(MAXH, np.int64)
        for j in range(lamps.shape[0]):
            vx = lamps[j, 0] - R[r, 0]; vy = lamps[j, 1] - R[r, 1]; vz = lamps[j, 2] - R[r, 2]
            dist = math.sqrt(vx * vx + vy * vy + vz * vz); ux, uy, uz = vx / dist, vy / dist, vz / dist
            cb = ux * N[r, 0] + uy * N[r, 1] + uz * N[r, 2]
            if two_sided:
                cb = abs(cb)
            elif cb <= 0:
                continue
            e = intensity[j] * cb / (dist * dist); unshaded[r] += e
            T = transmit(R[r, 0], R[r, 1], R[r, 2], ux, uy, uz, tmin, dist, own[r], P, kind, tau_prim, eps, gmin, h, dims, start, items, bt, bi)
            if T < 0.0:
                novf[r] += 1; T = 0.0
            shaded[r] += e * T
    return shaded, unshaded, novf


q = TR.placed('R3-1b')
def read_T(name):
    rows = [r for r in csv.reader(l for l in open(DVT / name, encoding='utf-8') if not l.startswith('#'))][1:]
    tri = np.array([[float(x) for x in r[6:]] for r in rows]).reshape(-1, 3, 3)
    return dict(tri=tri + q.offset, organ=np.array([r[0] for r in rows]), cls=np.array([r[1] for r in rows]), leaf=np.array([int(r[2]) for r in rows]))


class PlacedT:
    """Copy of k_dvt.PlacedT (16 receivers per leaflet triangle, sample seed = trace seed, delta 1e-7)."""
    def __init__(self, T, spt, seed):
        t = T['tri']; P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0]; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]; self._P = P; self._k = np.full(len(t), TRI, np.int8)
        isl = T['cls'] == 'Leaflet'; self.tau = np.where(isl, CFG['tau_leaf'], 0.0)
        cr = np.cross(P[:, 3:6], P[:, 6:9]); ar = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / np.maximum(2 * ar[:, None], 1e-30)
        idx = np.nonzero(isl)[0]; rng = np.random.default_rng(seed)
        u = rng.random((len(idx), spt)); v = rng.random((len(idx), spt)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
        self.centre = (t[idx, 0][:, None] + u[..., None] * P[idx, None, 3:6] + v[..., None] * P[idx, None, 6:9]).reshape(-1, 3)
        self.normal = np.repeat(nrm[idx], spt, 0); self.A = np.repeat(ar[idx] / spt, spt) * 1e6; self.own_local = np.repeat(idx, spt); self.leafid = np.repeat(T['leaf'][idx], spt)


LAMP_P = 1200.0; LAMP_H = 0.6; lamp = np.array([[obs.SCENE_XY[0], obs.SCENE_XY[1], obs.MEDIUM_Z + LAMP_H]]); I = np.array([LAMP_P / (4 * math.pi)])
T = read_T('T_triangles.csv'); st = dict(v1=[], tau_a1=[], tau_a085=[], overflow=0, coincident_share=[])
leaf_st = {k: [] for k in ('v1', 'tau_a085')}
for sd in SEEDS:
    o = PlacedT(T, 16, sd); sc = Scene(o._P, o._k, h=0.003)
    sh, un = trace_lamps(sc, o.centre, o.normal, True, o.own_local, 1e-7, lamp, I)
    half = coincident(sc, o.centre, o.normal, o.own_local); sh = sh.copy(); sh[half] *= 0.5
    sht, _, nv = point_lamps_tau(np.ascontiguousarray(o.centre), np.ascontiguousarray(o.normal), True, np.ascontiguousarray(o.own_local, np.int64), 1e-7, lamp, I, o.tau, EPS,
                                 sc.P, sc.kind, sc.gmin, sc.h, sc.dims, sc.start, sc.items); sht = sht.copy(); sht[half] *= 0.5
    st['v1'].append(float((sh * o.A * 1e-6).sum())); st['tau_a1'].append(float((sht * o.A * 1e-6).sum())); st['tau_a085'].append(float((sht * o.A * 1e-6).sum()) * A_MAIN)
    st['overflow'] += int(nv.sum()); st['coincident_share'].append(float(half.mean()))
    leaf_st['v1'].append({int(k): float((sh * o.A * 1e-6)[o.leafid == k].sum()) for k in range(1, 9)}); leaf_st['tau_a085'].append({int(k): float((sht * o.A * 1e-6)[o.leafid == k].sum() * A_MAIN) for k in range(1, 9)})
STATIC = {k: float(np.mean(v)) for k, v in st.items() if isinstance(v, list)}
RES['K7_static'] = dict(totals=STATIC, seeds=st, per_leaf={m: {k: float(np.mean([d[k] for d in v])) for k in v[0]} for m, v in leaf_st.items()})
RES['G0d_static_v1'] = dict(total=STATIC['v1'], stored=0.6494362481297447, rel=STATIC['v1'] / 0.6494362481297447 - 1, passed=bool(abs(STATIC['v1'] / 0.6494362481297447 - 1) <= 1e-9))
log(f'static T lamp v1 {STATIC["v1"]:.6f} (stored 0.649436) tau a1 {STATIC["tau_a1"]:.6f} tau a085 {STATIC["tau_a085"]:.6f}'); jdump(RES, OUT / 'k67_results.json')

core = sorted(glob.glob('/kaggle/input/**/core.jar', recursive=True)); GR = {}
GROIMP_RAYS = 30_000_000; GROIMP_SEEDS = (11, 12); DEPTH = 10
if core:
    gdir = OUT / 'groimp_install'; shutil.copytree(os.path.dirname(os.path.realpath(core[0])), gdir, dirs_exist_ok=True)
    RES['groimp_core_sha256'] = hashlib.sha256((gdir / 'core.jar').read_bytes()).hexdigest()
    organs = sorted(set(T['organ'].tolist())); oid = {o: i for i, o in enumerate(organs)}
    isl = {oid[o]: bool((T['cls'][T['organ'] == o] == 'Leaflet')[0]) for o in organs}; leafof = {oid[o]: int(T['leaf'][T['organ'] == o][0]) for o in organs}
    SHADERS = {'black_d1': ('BLACK', 1, None), 'black_d10': ('BLACK', DEPTH, None), 'leaf_rho0.08_tau0.07': ('LEAF', DEPTH, (0.08, 0.07)), 'leaf_rho0_tau0.07': ('LEAF', DEPTH, (0.0, 0.07))}
    for name, (kind, depth, rt) in SHADERS.items():
        wd = OUT / f'groimp_{name}'; wd.mkdir(exist_ok=True)
        with open(wd / 'mesh.csv', 'w') as fh:
            for o in organs:
                tt = T['tri'][T['organ'] == o]; fh.write(str(oid[o]) + ',' + ','.join('%.9g' % v for v in tt.reshape(-1)) + '\n')
        isleaf = ','.join('true' if isl[i] else 'false' for i in range(len(organs)))
        leaf_shader = ('Phong p=new Phong();p.setDiffuse(new Graytone(%sf));p.setTransparency(new Graytone(%sf));p.setInterpolatedTransparency(false);p.setDiffuseTransparency(new Graytone(0));return p;' % rt) if rt else 'return BLACK;'
        rgg = """// v1_tau closure: engine-law leaflet mesh T of R3-1b, engine lamp. Leaflets get the shader of this run, everything else BLACK.
import java.io.*;
import de.grogra.xl.util.FloatList;
module Surf(int sid) extends MeshNode;
module Lamp extends LightNode;
module World extends Node;
LightModel lm = new LightModel(%d, %d);
boolean[] isLeaf = new boolean[]{%s};
de.grogra.imp3d.shading.Shader leafShader() { %s }
protected void init() [ Axiom ==> World [ Translate(%gf, %gf, %gf) Lamp().(setLight(new PointLight().(setPower(%gf)))) ]; ]
public void measure() throws Exception {
    println("PROBE_START");
    try {
    BufferedReader r = new BufferedReader(new FileReader("%s")); String line;
    while ((line = r.readLine()) != null) { String[] v = line.split(","); FloatList f = new FloatList();
        for (int k = 1; k < v.length; k++) f.add(Float.parseFloat(v[k]));
        final int id = Integer.parseInt(v[0]); final FloatList ff = f;
        [ w:World ==> w [Surf(id).(setPolygons(getMesh(ff).getPolygons()),setShader(isLeaf[id] ? leafShader() : BLACK))]; ] derive(); }
    r.close();
    int[] seeds = new int[]{%s};
    for (int s : seeds) { lm.setSeed(s); lm.compute();
        for (Surf x : (* Surf *)) println("RESULT\\t" + s + "\\t" + x.sid + "\\t" + lm.getAbsorbedPower(x).integrate() + "\\t" + lm.getHitCount(x)); }
    } catch (Throwable e) { println("PROBE_ERROR " + e); StringWriter sw = new StringWriter(); e.printStackTrace(new PrintWriter(sw)); println(sw.toString()); }
    println("PROBE_DONE");
}
""" % (GROIMP_RAYS, depth, isleaf, leaf_shader, lamp[0, 0], lamp[0, 1], lamp[0, 2], LAMP_P, str(wd / 'mesh.csv'), ','.join(map(str, GROIMP_SEEDS)))
        (wd / 'Model.rgg').write_text(rgg, encoding='utf-8'); glog = wd / 'groimp.log'; t = time.time(); env = dict(os.environ, JDK_JAVA_OPTIONS='-Djava.awt.headless=true'); ok = True
        with glog.open('wb') as stream:
            p = subprocess.Popen(['java', '-Dfile.encoding=UTF-8', '-jar', str(gdir / 'core.jar'), '--headless', '-a', 'cli'], stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', cwd=str(gdir), env=env)
            def wait(pred, tmo=5400):
                end = time.time() + tmo
                while time.time() < end:
                    txt = glog.read_text(encoding='utf-8', errors='replace')
                    if pred(txt):
                        return txt
                    if p.poll() is not None:
                        break
                    time.sleep(1)
                raise RuntimeError('GroIMP stalled')
            try:
                wait(lambda s: '[base]>>' in s, 600); p.stdin.write(f'$cd {wd}\n$open Model.rgg\n'); p.stdin.flush()
                wait(lambda s: '[Model]>>' in s, 600); time.sleep(2); p.stdin.write('measure\n'); p.stdin.flush()
                txt = wait(lambda s: 'PROBE_DONE' in s or 'Command not found' in s)
            except Exception as e:
                txt = glog.read_text(encoding='utf-8', errors='replace'); ok = False; log(f'GroIMP {name} failed: {e}')
            finally:
                if p.poll() is None:
                    p.stdin.close(); p.kill(); p.wait()
        got = re.findall(r'RESULT\t(\d+)\t(\d+)\t([0-9.eE+-]+)\t(\d+)', txt); per = {}
        for s, sid, a, h in got:
            d = per.setdefault(int(s), dict(total=0.0, hits=0, **{f'leaf{Lf}': 0.0 for Lf in range(1, 9)}))
            if isl[int(sid)]:
                d['total'] += float(a); d['hits'] += int(h); d[f'leaf{leafof[int(sid)]}'] += float(a)
        mean = {k: float(np.mean([per[s][k] for s in per])) for k in (per[GROIMP_SEEDS[0]] if per else {})}
        GR[name] = dict(ok=bool(ok and len(per) == len(GROIMP_SEEDS)), seconds=round(time.time() - t), rays=GROIMP_RAYS, depth=depth, shader=kind, rho_tau=rt, seeds=per, mean=mean,
                        packet_sigma_total=math.sqrt(mean['total'] * LAMP_P / GROIMP_RAYS) if mean else None, error_lines=[l for l in txt.splitlines() if 'ERROR' in l][:3])
        log(f'GroIMP {name} ok={GR[name]["ok"]} total={mean.get("total")} {GR[name]["seconds"]} s'); RES['K7_groimp'] = GR; jdump(RES, OUT / 'k67_results.json')
else:
    log('GroIMP core.jar not found')
shutil.rmtree(OUT / 'groimp_install', ignore_errors=True); shutil.rmtree(OUT / 'numba_cache', ignore_errors=True)
RES['seconds'] = round(time.time() - T0); jdump(RES, OUT / 'k67_results.json'); log('K67_DONE')
