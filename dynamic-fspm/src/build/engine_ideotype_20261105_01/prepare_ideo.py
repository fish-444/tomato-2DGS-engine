"""Prepare ideotype cases locally (Python only, no JVM) and package one Kaggle dataset per batch (CONTRACT.md §1-§6).
python prepare_ideo.py <batch> <case spec> [...]
case spec = <type>@<env>@r<rep>[@<variant>]   type: BASE IN125 IN150 LS125 LS150 PE150 COMBO, or <type>x1 (multiplier forced to 1.0, gate G1)
  env: s<cm>h<cm> (s6h40 ... s14h50) or G0 (one plant at the lamp axis, D-ref point lamp, plant z 0 = Engine v2 scene)
  variant: mixed (ideotype only at p3), b0 / b1 (beta 0 / 1), fm0.8 / fm1.2 (FRUIT_MAX_DM x0.8 / x1.2), rays<N>
Every case = fresh build (Engine v2 request, n plants, plants.csv, lamps.csv) -> prepare -> ideo_patch -> prepared.json hashes re-bound -> v2_config.json."""
import argparse, copy, hashlib, json, os, shutil, sys, tarfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; V2 = REPO / 'build' / 'engine_v2_calib_20261104_01'; DEV = Path('C:/fsp_build/engine_dev_20261020')
os.environ['FSP_BUILD_ROOT'] = 'C:/fsp_build'
sys.path.insert(0, str(DEV)); sys.path.insert(1, str(REPO / 'build' / 'f5_runs_20261015_01' / 'scripts')); sys.path.insert(0, str(V2)); sys.path.insert(0, str(HERE))
from stage23_defs import LIT_EXPANSION
from tools import microtom_hook_runs as hr
from tools import microtom_parameters as mp
from tools import fsp_random_parameters as rp
from tools.fsp_engine_hooks import SCHEMA
from tools.tomato_fsp_pipeline import retention_policy
from tools.vtc_development import demo, prepare
from tools.vtc_growth_io import fresh_build, read_json, sha, write_json, write_csv
import ideo_patch as IP
BASE = Path('C:/fsp_build/engine_ideotype_20261105_01'); PREP = BASE / 'prepared'
DAYS = 72; SEED0 = 20261020; SLA_F = 1.15; LOW_LEAF = 0.7; BETA = 0.5; SUBSTRATE_Z = 0.070; LUM = dict(length=0.60, width=0.10, power=1200.0)
FLOOR = read_json(REPO / 'build' / 'reserve_cap_floor_20261018_01' / 'api_example.json')['merge_into_existing_engine_hooks']
S1 = read_json(REPO / 'build' / 's1_hooks_20261017_01' / 'api_examples.json')['examples']['fruit_ke2023']
TYPES = {'BASE': {}, 'IN125': dict(IN=1.25), 'IN150': dict(IN=1.5), 'LS125': dict(LS=1.25), 'LS150': dict(LS=1.5), 'PE150': dict(PE=1.5), 'COMBO': dict(IN=1.25, LS=1.25)}


def parse(spec):
    q = spec.split('@'); typ, env, rep = q[0], q[1], int(q[2][1:]); var = q[3] if len(q) > 3 else ''
    one = typ.endswith('x1'); base_t = typ[:-2] if one else typ
    mult = {k: (1.0 if one else v) for k, v in TYPES[base_t].items()}
    if env.startswith('T32'):   # tray32 layout (build/static_tray32_v2_20261104_01 CONTRACT): 4 trays of 4 x 2 in a row, gap 0.10 m; for the GUI timing check
        e2 = env[3:]; sp, h = int(e2[1:e2.index('h')]) / 100, int(e2[e2.index('h') + 1:]) / 100
        n = 32; pos = [((i - 2.5) * (2 * sp + 0.10) + (c - 1.5) * sp, (r - 2.5) * sp) for i in range(1, 5) for r in range(1, 5) for c in range(1, 3)]; z0 = SUBSTRATE_Z; lum = dict(LUM, z=h)
    elif env == 'G0':
        n, pos, z0, lum = 1, [(0.0, 0.0)], 0.0, None
    else:
        sp, h = int(env[1:env.index('h')]) / 100, int(env[env.index('h') + 1:]) / 100
        n = 8; pos = [((c - 1.5) * sp, (r - 2.5) * sp) for r in range(1, 5) for c in range(1, 3)]; z0 = SUBSTRATE_Z; lum = dict(LUM, z=h)
    beta = {'b0': 0.0, 'b1': 1.0}.get(var, BETA); fm = {'fm0.8': 0.8, 'fm1.2': 1.2}.get(var, 1.0)
    rays = int(var[4:]) if var.startswith('rays') else 300000
    prof = None
    for tok in var.split('+'):
        if tok.startswith('v22'):   # v22A:<s> / v22R:<s> -> Engine v2.2 arm profile with internode scalar s (Amendment 19 of the v2 calibration)
            arm_, s_ = tok[3:].split(':'); prof = IP.profile_v22(arm_, float(s_))
    var = '+'.join(x for x in var.split('+') if not x.startswith('v22'))
    plants = ([2] if var == 'mixed' else list(range(n))) if mult else []
    ideo = dict(type=typ, plants=plants, beta=beta, **mult) if mult else None
    return dict(spec=spec, type=typ, env=env, rep=rep, variant=var, n=n, pos=pos, z0=z0, lum=lum, ideo=ideo, fruit_factor=fm, rays=rays, seed=SEED0 + rep - 1, profile=prof)


def build(D, c):
    fresh_build(D); demo(D / 'inputs', intervals=DAYS, rays=c['rays'])
    req = read_json(D / 'inputs' / 'request.json')
    g = hr.g_hooks(mp.load_registry(), *hr.G_PREFERRED)['parameters']
    for k in g: g[k]['source_ref'] += ' | preferred G scenario, user decision 2026-10-15 15:10 (stagger 38 basipetal; a 0)'
    params = dict(g); params.update(LIT_EXPANSION); params['CAP_FLOOR_GC'] = dict(FLOOR['parameters']['CAP_FLOOR_GC'])
    params.update({k: dict(v) for k, v in S1['parameters'].items()})
    if c.get('profile') and c['profile'].get('order'):   # v2.2: leaflet growth order in the request hook (as prepare_v2 'order')
        params['LEAFLET_GROWTH_ORDER'] = dict(params['LEAFLET_GROWTH_ORDER']); params['LEAFLET_GROWTH_ORDER']['value'] = c['profile']['order']
        params['LEAFLET_GROWTH_ORDER']['source_ref'] += f" | user decision 2026-11-06 (Engine v2 C4C, CONTRACT Amendment 15): {c['profile']['order']}"
    if c['fruit_factor'] != 1.0:
        params['FRUIT_MAX_DM'] = dict(params['FRUIT_MAX_DM']); params['FRUIT_MAX_DM']['value'] *= c['fruit_factor']
        params['FRUIT_MAX_DM']['source_ref'] += f" | ideotype fruit sensitivity x{c['fruit_factor']} (build/engine_ideotype_20261105_01)"
    hooks = dict(schema=SCHEMA, planned_axes=False, per_axis_development=False, leaf_development=True, projected_shading=False,
                 leaflet_outline=True, reserve_cap=True, reserve_cap_floor=True, fruit_sink=True)
    hooks[rp.SWITCH] = True
    params.update(rp.registry_records(mp.load_registry()))
    params[rp.SEED_KEY] = rp.seed_record(c['seed'], f"ideotype replicate {c['rep']} (Engine v2 convention 20261020 + m), build/engine_ideotype_20261105_01")
    hooks['parameters'] = params; req['engine_hooks'] = hooks
    write_json(D / 'inputs' / 'request.json', req)
    plants = [dict(plant_instance_id=f'p{k + 1}', source_plant_id=f'synthetic_p{k + 1}', x_m=x, y_m=y, z_m=c['z0'], reserve_gC=.06, provenance='synthetic_assumed')
              for k, (x, y) in enumerate(c['pos'])]
    write_csv(D / 'inputs' / 'plants.csv', plants, list(plants[0]))
    if c['lum']:
        lp = D / 'inputs' / 'lamps.csv'; txt = lp.read_text(encoding='utf-8'); lines = txt.splitlines(); hdr = lines[0].split(','); zi = hdr.index('z_m'); ski = hdr.index('source_kind')
        cc = lines[1].split(','); assert len(lines) == 2 and abs(float(cc[zi]) - 0.6) < 1e-12, lines
        cc[zi] = repr(c['lum']['z']); cc[ski] = 'assumed_flat_luminaire_60x10cm_lambertian_centre_bottom_face'
        lp.write_text(lines[0] + '\n' + ','.join(cc) + ('\n' if txt.endswith('\n') else ''), encoding='utf-8')
    prepare(D / 'inputs', D / 'native'); write_json(D / 'retention_policy.json', retention_policy('working_last_1'))
    W = D / 'native' / 'working'; M = W / 'Model.rgg'; Pm = W / 'parametersMicroTom.rgg'; old_m, old_p = sha(M), sha(Pm)
    M.write_text(IP.apply_multi(M.read_text(encoding='utf-8'), c['n'], c['ideo'], c['lum'], profile=c.get('profile')), encoding='utf-8')
    Pm.write_text(IP.params(Pm.read_text(encoding='utf-8'), profile=c.get('profile')), encoding='utf-8')
    pj = D / 'native' / 'prepared.json'; prep = json.loads(pj.read_text(encoding='utf-8')); assert prep['model_hash'] == old_m and prep['parameter_hash'] == old_p
    prep['model_hash'] = sha(M); prep['parameter_hash'] = sha(Pm); pj.write_text(json.dumps(prep, indent=2), encoding='utf-8')
    cfg = IP.config(c['n'], c['ideo'], c['lum'], profile=c.get('profile')); cfg.update(case=c, rebind=dict(model_old=old_m, model_new=prep['model_hash']))
    write_json(D / 'v2_config.json', cfg)
    return prep['model_hash']


def name(spec):
    return spec.replace('@', '_').replace('.', 'p').replace(':', '-').replace('+', '_')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('batch'); ap.add_argument('specs', nargs='+'); ap.add_argument('--extra', nargs='*', default=[]); a = ap.parse_args()
    res = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), batch=a.batch, cases={})
    for spec in a.specs:
        c = parse(spec); D = PREP / name(spec)
        if not (D / 'v2_config.json').exists():
            if D.exists(): shutil.rmtree(D)
            build(D, c)
        res['cases'][D.name] = dict(spec=spec, model_sha=sha(D / 'native/working/Model.rgg'))
        print(spec, 'prepared', flush=True)
    members = [(PREP / cn, f'prepared/{cn}') for cn in res['cases']] + [(Path(x), f'prepared/{Path(x).name}') for x in a.extra]
    DS = HERE / 'dataset' / a.batch; DS.mkdir(parents=True, exist_ok=True)
    def tar(nm, mem):
        p = DS / f'{nm}.tar.gz'
        with tarfile.open(p, 'w:gz') as t:
            for src, arc in mem: t.add(src, arcname=arc, filter=lambda ti: None if '__pycache__' in ti.name else ti)
        return p
    runs = tar('runs', members)
    code = tar('code', [(HERE / 'kaggle_kernel' / 'ideo_case.py', 'evcode/ideo_case.py'), (HERE / 'kaggle_kernel' / 'ideo_validate.py', 'evcode/ideo_validate.py'),
                        (V2 / 'kaggle_kernel' / 'v2_validate.py', 'evcode/v2_validate.py')])
    core = Path('C:/Program Files/GroIMP 2.2.1/core.jar')
    eng = {p.name: sha(p) for p in sorted((DEV / 'tools').glob('*')) if p.is_file() and p.suffix in ('.py', '.rgg')}
    jobs = list(res['cases']) + [Path(x).name for x in a.extra]
    man = dict(schema='ideo-kaggle-input-1', batch=a.batch, cases=jobs, tarballs={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in (runs, code)},
               groimp=dict(core_sha256=sha(core)), engine_code=eng)
    write_json(DS / 'ENGV2_MANIFEST.json', man)
    slug = f'ideo-{a.batch.lower()}-20261105'; write_json(DS / 'dataset-metadata.json', dict(title=slug, id=f'dolgfish/{slug}', licenses=[dict(name='CC0-1.0')]))
    res['jobs'] = jobs; res['dataset'] = dict(slug=slug, **man['tarballs']); res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    write_json(HERE / f'PREPARE_{a.batch}.json', res); print(json.dumps({k: v for k, v in res.items() if k != 'cases'}, indent=1))


if __name__ == '__main__':
    main()
