"""Probe: does GroIMP's getMesh + LightModel absorb on small triangles? Isolated black surfaces under the engine's point lamp,
GroIMP vs analytic (static tracer, own triangles only). Explains gate S1/S1b (see GATES_DIAGNOSIS.md)."""
import json, math, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from static.geometry import EngineSnapshot
from static.tracer import Scene, trace_lamps, TRI
from tools.groimp_static_verify import cli_path
OUT = HERE / 'groimp_mesh_probe'; OUT.mkdir(exist_ok=True)
JAVA = r'C:\Program Files\Eclipse Adoptium\jre-21.0.12.101-hotspot\bin\java.exe'; CORE = r'C:\Program Files\GroIMP 2.2.1\core.jar'
W = REPO / 'build/engine_maxmass_fix_20261025_01/runs/seed72/native/working'
POWER, H, RAYS = 1200.0, 0.6, 30_000_000
pos = [(-0.15 + (j % 4) * 0.1, -0.15 + (j // 4) * 0.1) for j in range(16)]


def square(edge, side=0.02):
    n = max(1, int(round(side / edge))); g = np.linspace(-side / 2, side / 2, n + 1); t = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = (g[i], g[j]), (g[i + 1], g[j]), (g[i + 1], g[j + 1]), (g[i], g[j + 1])
            t += [[a, b, c], [a, c, d]]
    t = np.array(t); return np.concatenate([t, np.zeros(t.shape[:2] + (1,))], 2)


surfs = []
for e in (0.01, 0.002, 0.001, 0.0005, 0.00035, 0.00025):
    surfs.append(dict(kind='square', edge_mm=e * 1000, tri=square(e)))
s = EngineSnapshot(W, 71)
leaf_ids = sorted({o for o, l in zip(s.owner, s.is_leaf) if l}, key=lambda o: s.area[s.owner == o].sum())
pick = [leaf_ids[int(q * (len(leaf_ids) - 1))] for q in np.linspace(0, 1, 10)]
for o in pick:
    t = s.tri[s.owner == o].copy(); c = (t.reshape(-1, 3) * 1).mean(0); t = t - c
    e = np.stack([np.linalg.norm(t[:, 1] - t[:, 0], axis=1), np.linalg.norm(t[:, 2] - t[:, 1], axis=1), np.linalg.norm(t[:, 0] - t[:, 2], axis=1)], 1).min(1)
    surfs.append(dict(kind='leaflet', organ=o, median_min_edge_mm=float(np.median(e) * 1000), tri=t))
lines = []
for i, (sf, (x, y)) in enumerate(zip(surfs, pos)):
    t = sf['tri'] + np.array([x, y, 0.0]); sf['placed'] = t; sf['area_mm2'] = float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() * 1e6)
    lines.append(str(i) + ',' + ','.join('%.9g' % v for v in t.reshape(-1)))
(OUT / 'probe_mesh.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
RGG = """// GroIMP mesh probe (engine_static_20261026_01). Generated.
import java.io.*;
import de.grogra.xl.util.FloatList;
module Surf(int sid) extends MeshNode;
module Lamp extends LightNode;
module World extends Node;
LightModel lm = new LightModel(%d, 1);
protected void init() [ Axiom ==> World [ Translate(0, 0, %gf) Lamp().(setLight(new PointLight().(setPower(%gf)))) ]; ]
public void measure() throws Exception {
    println("PROBE_START");
    try {
    BufferedReader r = new BufferedReader(new FileReader("%s")); String line;
    while ((line = r.readLine()) != null) { String[] v = line.split(","); FloatList f = new FloatList();
        for (int k = 1; k < v.length; k++) f.add(Float.parseFloat(v[k]));
        final int id = Integer.parseInt(v[0]); final FloatList ff = f;
        [ w:World ==> w [Surf(id).(setPolygons(getMesh(ff).getPolygons()),setShader(BLACK))]; ] derive(); }
    r.close();
    lm.compute();
    for (Surf s : (* Surf *)) println("RESULT\t" + s.sid + "\t" + lm.getAbsorbedPower(s).integrate() + "\t" + lm.getHitCount(s));
    } catch (Throwable e) { println("PROBE_ERROR " + e); StringWriter sw = new StringWriter(); e.printStackTrace(new PrintWriter(sw)); println(sw.toString()); }
    println("PROBE_DONE");
}
""" % (RAYS, H, POWER, (cli_path(OUT) + "/probe_mesh.csv").replace("\\", "/"))
(OUT / 'Model.rgg').write_text(RGG, encoding='utf-8')
log = OUT / 'groimp.log'; t0 = time.time()
with log.open('wb') as stream:
    p = subprocess.Popen([JAVA, '-Dfile.encoding=UTF-8', '-jar', CORE, '--headless', '-a', 'cli'], stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    def wait(pred, tmo=3000):
        end = time.time() + tmo
        while time.time() < end:
            txt = log.read_text(encoding='utf-8', errors='replace')
            if pred(txt): return txt
            if p.poll() is not None: break
            time.sleep(0.5)
        raise RuntimeError('GroIMP stalled; see ' + str(log))
    try:
        wait(lambda s: '[base]>>' in s); p.stdin.write(f'$cd {cli_path(OUT)}\n$open Model.rgg\n'); p.stdin.flush()
        wait(lambda s: '[Model]>>' in s); time.sleep(2); p.stdin.write('measure\n'); p.stdin.flush()
        txt = wait(lambda s: 'PROBE_DONE' in s or 'Command not found' in s)
    finally:
        if p.poll() is None: p.stdin.close(); p.kill(); p.wait()
g = {int(a): (float(b), int(c)) for a, b, c in re.findall(r'RESULT\t(\d+)\t([0-9.eE+-]+)\t(\d+)', txt)}
if len(g) != len(surfs): raise SystemExit('missing results; see groimp.log')
L = np.array([[0, 0, H]]); I = np.array([POWER / (4 * math.pi)]); packet = POWER / RAYS; rows = []
for i, sf in enumerate(surfs):
    t = sf['placed']; P = np.zeros((len(t), 14)); P[:, 0:3] = t[:, 0]; P[:, 3:6] = t[:, 1] - t[:, 0]; P[:, 6:9] = t[:, 2] - t[:, 0]
    sc = Scene(P, np.full(len(t), TRI, np.int8), h=0.002); rng = np.random.default_rng(i); K = 64
    u = rng.random((len(t), K)); v = rng.random((len(t), K)); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
    R = (t[:, 0][:, None] + u[..., None] * P[:, None, 3:6] + v[..., None] * P[:, None, 6:9]).reshape(-1, 3)
    cr = np.cross(P[:, 3:6], P[:, 6:9]); ar = 0.5 * np.linalg.norm(cr, axis=1); N = cr / np.maximum(2 * ar[:, None], 1e-30)
    sh, _ = trace_lamps(sc, R, np.repeat(N, K, 0), True, np.repeat(np.arange(len(t)), K), 1e-7, L, I)
    a = float((sh * np.repeat(ar / K, K)).sum()); gv, hits = g[i]
    rows.append(dict({k: v for k, v in sf.items() if k not in ('tri', 'placed')}, triangles=len(t), analytic_umol_s=a, groimp_umol_s=gv, groimp_hits=hits,
                     ratio=gv / a, ratio_sigma=math.sqrt(a * packet) / a))
res = dict(rays=RAYS, power=POWER, lamp_z_m=H, groimp_seconds=round(time.time() - t0), rows=rows)
(HERE / 'GROIMP_MESH_PROBE.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
for r in rows: print(r['kind'], r.get('edge_mm', r.get('median_min_edge_mm')), r['triangles'], round(r['area_mm2'], 1), 'ratio %.3f +- %.3f' % (r['ratio'], r['ratio_sigma']))
