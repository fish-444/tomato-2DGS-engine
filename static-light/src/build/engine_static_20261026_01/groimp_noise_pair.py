"""GroIMP repeat-seed sample on identical geometry, for the ray-noise model of the STATIC consistency gates.
python groimp_noise_pair.py  -> runs/f5_i1_seed{12,13}/, NOISE.json
The F5 control (imported fixture, 4 plants, lamp L1 1200 umol/s at 0.6 m, 300 000 rays) is prepared with ray seeds 12 and 13 and run for ONE
interval with light_only-equivalent content: interval 0's light is computed on the imported geometry before any growth, so interval-0
absorbed per leaflet is a GroIMP repeat on identical geometry. Seed 11 = the kept 30-d E7-adoption off control (same Model.rgg except seed).
Nothing in tools/ is changed; the repo engine is used read-only; FSP_BUILD_ROOT outputs."""
import csv, json, math, os, statistics as st, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(1, str(REPO / 'build' / 'f5_runs_20261015_01' / 'scripts'))
os.environ['JDK_JAVA_OPTIONS'] = '-Djava.awt.headless=true -XX:ActiveProcessorCount=1'
os.environ['FSP_BUILD_ROOT'] = 'C:/fsp_build'
from stage23_defs import LIT_EXPANSION
from tools import microtom_hook_runs as hr
from tools import vtc_development as vd
from tools.vtc_growth_io import read_json, rows, write_json
from tools.vtc_growth_bridge import run
OUT = Path('C:/fsp_build/engine_static_20261026_01/runs'); OUT.mkdir(parents=True, exist_ok=True)
PLAN4 = REPO / 'build' / 'budbreak_plan_20261015_f4' / 'plan.csv'
SEED11 = Path('C:/fsp_build/engine_dev_20261020/gates/e7adopt/off_control/native/working/flux.csv')


def prepare_one_interval(dest: Path, ray_seed: int):
    if (dest / 'native' / 'prepared.json').exists(): return
    orig = vd.prepare

    def merged(inputs, native, *a, **k):
        inputs = Path(inputs); req = read_json(inputs / 'request.json'); req['intervals'] = 1
        env = rows(inputs / 'environment.csv')[:1]
        with (inputs / 'environment.csv').open('w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(env[0])); w.writeheader(); w.writerows(env)
        from tools.vtc_growth_io import write_json as wj; wj(inputs / 'request.json', req)
        return orig(inputs, native, *a, **k)
    vd.prepare = merged
    try: hr.f_prepare(dest, None, None, PLAN4, ray_seed, None, True, dict(LIT_EXPANSION), True)
    finally: vd.prepare = orig


res = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), seeds={})
fluxes = {11: SEED11}
for s in (12, 13):
    d = OUT / f'f5_i1_seed{s}'
    prepare_one_interval(d, s)
    if not (d / 'native' / 'working' / 'flux.csv').exists():
        t0 = time.time(); r = run(d / 'native', timeout_s=3600)
        res['seeds'][s] = dict(status=r.get('status'), elapsed_s=round(time.time() - t0))
        print('run seed', s, res['seeds'][s], flush=True)
    fluxes[s] = d / 'native' / 'working' / 'flux.csv'


def interval0(path: Path) -> dict:
    return {r['organ_id']: float(r['absorbed_umol_s']) for r in csv.DictReader(open(path, encoding='utf-8')) if r['interval'] == '0'}


A = {s: interval0(p) for s, p in fluxes.items()}
organs = sorted(k for k in A[11] if all(A[s].get(k, 0) > 0 for s in A))
P, N = 1200.0, 300000; packet = P / N
# per-organ: observed spread across 3 seeds vs the binomial packet model sd = sqrt(mean * packet)
z, rows_out = [], []
for k in organs:
    v = [A[s][k] for s in sorted(A)]; m = sum(v) / 3; sd = st.stdev(v); model = math.sqrt(m * packet)
    z.append(sd / model); rows_out.append(dict(organ_id=k, mean=m, sd=sd, model_sd=model, ratio=sd / model))
plants = sorted({k.split('|')[0].split('/')[0] for k in organs})
plant = {}
for p in plants:
    v = [sum(A[s][k] for k in organs if k.split('|')[0].split('/')[0] == p) for s in sorted(A)]
    m = sum(v) / 3; plant[p] = dict(values=v, mean=m, rel_sd=st.stdev(v) / m, model_rel_sd=math.sqrt(packet / m))
ratio2 = sum(r['sd'] ** 2 for r in rows_out) / sum(r['model_sd'] ** 2 for r in rows_out)
res.update(packet_umol_s=packet, organs=len(organs), variance_ratio_observed_over_binomial=ratio2,
           median_sd_ratio=sorted(z)[len(z) // 2], plants=plant,
           reading='variance ratio near 1 => GroIMP MC noise per organ = sqrt(absorbed x P/N); used for the snapshot gate tolerances')
with (HERE / 'NOISE_per_organ.csv').open('w', encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
write_json(HERE / 'NOISE.json', res)
print(json.dumps({k: res[k] for k in ('organs', 'variance_ratio_observed_over_binomial', 'median_sd_ratio', 'plants')}, indent=1))
