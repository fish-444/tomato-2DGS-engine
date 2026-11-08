"""Prepare Engine v2 cases locally (Python only, no JVM) and package one Kaggle dataset per batch (CONTRACT.md §1-2, §5).
python prepare_v2.py <batch> <arm> [<arm> ...]       arms: B0 C1 C2 C2T C3 C4 C5f0.85 C5f1.15 C5 (C6 arms: C6z0.45 C6z0.75 C6<factor>x0.8/x1.2, need --f)
B0 is built exactly as the divergence-sensitivity default arm and must be byte-identical to it (reproduction gate). Other arms = B0 copy +
v2_patch (or a fresh build with changed request inputs for C6) with prepared.json model/parameter hashes re-bound (old/new recorded)."""
import argparse, hashlib, json, os, shutil, sys, tarfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; DEV = Path('C:/fsp_build/engine_dev_20261020')
os.environ['FSP_BUILD_ROOT'] = 'C:/fsp_build'
sys.path.insert(0, str(DEV)); sys.path.insert(1, str(REPO / 'build' / 'f5_runs_20261015_01' / 'scripts')); sys.path.insert(0, str(HERE))
from stage23_defs import LIT_EXPANSION
from tools import microtom_hook_runs as hr
from tools import microtom_parameters as mp
from tools import fsp_random_parameters as rp
from tools.fsp_engine_hooks import SCHEMA
from tools.tomato_fsp_pipeline import retention_policy
from tools.vtc_development import demo, prepare
from tools.vtc_growth_io import ROOT as TOOLS_ROOT, fresh_build, read_json, sha, write_json, write_csv
import v2_patch as P
assert TOOLS_ROOT == DEV, TOOLS_ROOT
BASE = Path('C:/fsp_build/engine_v2_calib_20261104_01'); PREP = BASE / 'prepared'
DIVSENS = json.load(open(HERE / 'INPUT_HASHES.json'))['divsens_prepared_A']
DAYS = 72; SEEDS = [20261020 + m for m in range(16)]
FLOOR = read_json(REPO / 'build' / 'reserve_cap_floor_20261018_01' / 'api_example.json')['merge_into_existing_engine_hooks']
S1 = read_json(REPO / 'build' / 's1_hooks_20261017_01' / 'api_examples.json')['examples']['fruit_ke2023']
C6F = {'MAXMASS': 'MAX_MASS (all kinds)', 'CAPFRAC': 'RESERVE_CAP_FRACTION', 'RESERVE0': 'initial reserve plants.csv', 'K': 'K (all kinds) + LEAFLET_K', 'FRUITMAX': 'FRUIT_MAX_DM (S1)'}


def build(D, seed, m, lamp_z=0.6, change=None, order=None):
    """divsens build() (= D-ref member request, one plant at the lamp axis), plus the optional C6 request change (factor on one quantity)."""
    fresh_build(D); demo(D / 'inputs', intervals=DAYS, rays=300000)
    req = read_json(D / 'inputs' / 'request.json')
    g = hr.g_hooks(mp.load_registry(), *hr.G_PREFERRED)['parameters']
    for k in g: g[k]['source_ref'] += ' | preferred G scenario, user decision 2026-10-15 15:10 (stagger 38 basipetal; a 0)'
    params = dict(g); params.update(LIT_EXPANSION); params['CAP_FLOOR_GC'] = dict(FLOOR['parameters']['CAP_FLOOR_GC'])
    params.update({k: dict(v) for k, v in S1['parameters'].items()})
    hooks = dict(schema=SCHEMA, planned_axes=False, per_axis_development=False, leaf_development=True, projected_shading=False,
                 leaflet_outline=True, reserve_cap=True, reserve_cap_floor=True, fruit_sink=True)
    hooks[rp.SWITCH] = True
    params.update(rp.registry_records(mp.load_registry()))
    params[rp.SEED_KEY] = rp.seed_record(seed, f'divergence sensitivity seed m {m} (H convention 20261020 + m), build/engine_divergence_sens_20261102_01')
    reserve0 = .06; note = {}
    if change:
        key, f = change; tag = f' | Engine v2 C6 sensitivity {key} x{f} (build/engine_v2_calib_20261104_01 CONTRACT §2 C6)'
        if key == 'MAXMASS':
            v = req['parameters']['MAX_MASS']['value']; req['parameters']['MAX_MASS']['value'] = [x * f for x in v]; req['parameters']['MAX_MASS']['source_ref'] += tag; note = dict(old=v)
        elif key == 'K':
            v = req['parameters']['K']['value']; req['parameters']['K']['value'] = [x * f for x in v]; req['parameters']['K']['source_ref'] += tag
            params['LEAFLET_K'] = dict(params['LEAFLET_K']); params['LEAFLET_K']['value'] *= f; params['LEAFLET_K']['source_ref'] += tag; note = dict(old=v)
        elif key == 'FRUITMAX':
            params['FRUIT_MAX_DM'] = dict(params['FRUIT_MAX_DM']); note = dict(old=params['FRUIT_MAX_DM']['value']); params['FRUIT_MAX_DM']['value'] *= f; params['FRUIT_MAX_DM']['source_ref'] += tag
        elif key == 'RESERVE0':
            reserve0 = .06 * f; note = dict(old=.06)
        elif key == 'CAPFRAC':
            from tools import fsp_reserve_cap as rcap
            import copy
            p = copy.deepcopy(rcap.DEFAULTS['RESERVE_CAP_FRACTION']); note = dict(old=p['value']); p['value'] = p['value'] * f
            p['source_ref'] = str(p.get('source_ref', '')) + tag; params['RESERVE_CAP_FRACTION'] = p
        else:
            raise SystemExit(key)
    if order:   # CONTRACT Amendment 15 (user 2026-11-06: leaflets of one compound grow together)
        params['LEAFLET_GROWTH_ORDER'] = dict(params['LEAFLET_GROWTH_ORDER']); params['LEAFLET_GROWTH_ORDER']['value'] = order
        params['LEAFLET_GROWTH_ORDER']['source_ref'] += f' | user decision 2026-11-06 (Engine v2 C4C, CONTRACT Amendment 15): {order}'
    hooks['parameters'] = params; req['engine_hooks'] = hooks
    write_json(D / 'inputs' / 'request.json', req)
    plants = [dict(plant_instance_id='p1', source_plant_id='synthetic_p1', x_m=0.0, y_m=0.0, z_m=0, reserve_gC=reserve0, provenance='synthetic_assumed')]
    write_csv(D / 'inputs' / 'plants.csv', plants, list(plants[0]))
    if lamp_z != 0.6:
        lp = D / 'inputs' / 'lamps.csv'; txt = lp.read_text(encoding='utf-8'); lines = txt.splitlines(); hdr = lines[0].split(',')
        zi = hdr.index('z_m'); out = [lines[0]]
        for ln in lines[1:]:
            c = ln.split(','); assert abs(float(c[zi]) - 0.6) < 1e-12, ln; c[zi] = repr(lamp_z); out.append(','.join(c))
        lp.write_text('\n'.join(out) + ('\n' if txt.endswith('\n') else ''), encoding='utf-8')
    prepare(D / 'inputs', D / 'native'); write_json(D / 'retention_policy.json', retention_policy('working_last_1'))
    return note


def patch_case(D, level, f=1.0, test=None, low_leaf=P.LOW_LEAF_SCALE, inter_all=1.0, flowers_fixed=0, c4l=False, c4t=False, divergence=None, c4b=False, c4c=False, rachis=None):
    W = D / 'native' / 'working'; M = W / 'Model.rgg'; Pm = W / 'parametersMicroTom.rgg'
    old_m, old_p = sha(M), sha(Pm)
    M.write_text(P.apply(M.read_text(encoding='utf-8'), level, f, test, low_leaf, inter_all, flowers_fixed, c4l, c4t, divergence, c4b), encoding='utf-8')
    Pm.write_text(P.params(Pm.read_text(encoding='utf-8'), level, f), encoding='utf-8')
    if c4c:   # CONTRACT Amendment 14: after the v2 stack
        import c4c_patch as CC
        M.write_text(CC.apply_c4c(M.read_text(encoding='utf-8')), encoding='utf-8'); Pm.write_text(CC.params(Pm.read_text(encoding='utf-8')), encoding='utf-8')
    if rachis:   # CONTRACT Amendment 17 Step C
        import c4a_patch as CA
        M.write_text(CA.apply_rachis(M.read_text(encoding='utf-8'), rachis), encoding='utf-8')
    pj = D / 'native' / 'prepared.json'; prep = json.loads(pj.read_text(encoding='utf-8'))
    assert prep['model_hash'] == old_m and prep['parameter_hash'] == old_p
    prep['model_hash'] = sha(M); prep['parameter_hash'] = sha(Pm); pj.write_text(json.dumps(prep, indent=2), encoding='utf-8')
    cfg = P.config(level, f, low_leaf, inter_all, flowers_fixed, c4l, c4t, divergence, c4b)
    if c4c: cfg.update(__import__('c4c_patch').config_extra())
    if rachis: cfg.update(__import__('c4a_patch').config_extra(rachis))
    cfg.update(test=test, rebind=dict(model_old=old_m, model_new=prep['model_hash'], parameter_old=old_p, parameter_new=prep['parameter_hash']))
    write_json(D / 'v2_config.json', cfg)
    return cfg['rebind']


def arm_spec(arm, f_fit):
    if arm == 'B0': return dict(level='B0')
    if arm in ('C1', 'C2', 'C3', 'C4', 'C4U'): return dict(level=arm)
    if arm == 'C4V': return dict(level='C4U', low_leaf=0.7)   # Amendment 4
    if arm == 'C4W': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4)   # Amendment 7
    if arm == 'C4Wm': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4)   # = C4W, rerun with the C4L measures (Amendment 8)
    if arm == 'C4L': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4l=True)   # Amendment 8 (pilot on C4W)
    if arm == 'C4T': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4t=True)   # Amendment 9
    if arm == 'C4T2': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0))   # Amendment 10 (real C4T)
    if arm == 'C4T2L': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True)   # C4L on C4T2
    if arm == 'C4T2Lr': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True)   # = C4T2L rerun for the span measure (Amendment 12)
    if arm.startswith('C4AB'):   # Amendment 17 Step B: V22b with SLA factor f (C4AB<f>) 
        return dict(level='C5', f=float(arm[4:]), low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=5.85, c4c=True, order='simultaneous')
    if arm.startswith('V22A_s') or arm.startswith('V22R_s'):   # Amendment 19: Arm A (f 1.3271, C4A_rachis_mult.json) / Arm R (f 1.15, C4A_rachis_mult_R.json), internodes x s
        A_ = arm.startswith('V22A'); mult = json.load(open(Path(__file__).resolve().parent / ('C4A_rachis_mult.json' if A_ else 'C4A_rachis_mult_R.json')))['main_rank_1_13']
        return dict(level='C5', f=1.3271 if A_ else 1.15, low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=float(arm[6:]), c4c=True, order='simultaneous', rachis=mult)
    if arm.startswith('C4ACs'):   # Amendment 18: C4AC1.3271 with internodes x s
        mult = json.load(open(Path(__file__).resolve().parent / 'C4A_rachis_mult.json'))['main_rank_1_13']
        return dict(level='C5', f=1.3271, low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=float(arm[5:]), c4c=True, order='simultaneous', rachis=mult)
    if arm.startswith('C4AC'):   # Amendment 17 Step C: C4AB<f> + per-rank rachis multipliers from C4A_rachis_mult.json (C4AC<f>)
        mult = json.load(open(Path(__file__).resolve().parent / 'C4A_rachis_mult.json'))['main_rank_1_13']
        return dict(level='C5', f=float(arm[4:]), low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=5.85, c4c=True, order='simultaneous', rachis=mult)
    if arm.startswith('V22bs'): return dict(level='C5', f=1.15, low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=float(arm[5:]), c4c=True, order='simultaneous')   # Amendment 16
    if arm == 'V22b': return dict(level='C5', f=1.15, low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=5.85, c4c=True, order='simultaneous')   # Amendment 15
    if arm == 'V22a': return dict(level='C5', f=1.15, low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=5.85, c4c=True)   # Amendment 14 (C4C on v2.1)
    if arm.startswith('V21Bs'): return dict(level='C5', f=1.15, low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=float(arm[5:]))   # Amendment 6 on C4B
    if arm == 'C4T2LB1': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True)   # Amendment 13 rerun (validator floor 1 deg)
    if arm == 'C4T2LB': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True)   # Amendment 11
    if arm == 'C4TL': return dict(level='C4U', low_leaf=0.7, flowers_fixed=4, c4t=True, c4l=True)   # Amendment 9: C4L on C4T
    if arm.startswith('V21s'): return dict(level='C5', f=1.15, low_leaf=0.7, inter_all=float(arm[4:]))   # Amendment 6
    if arm.startswith('V21C6'):   # C6 arms on the frozen v2.1 (CONTRACT: freeze of v2.1)
        sp = arm_spec('C6' + arm[5:], 1.15); sp.update(flowers_fixed=4, c4t=True, divergence=(90.0, 100.0), c4l=True, c4b=True, inter_all=5.85); return sp
    if arm == 'C2T': return dict(level='C2', test=('p1/C0003', 30.0), seeds=[0])
    if arm.startswith('C5f'): return dict(level='C5', f=float(arm[3:]), low_leaf=0.7)
    if arm == 'C5': return dict(level='C5', f=f_fit, low_leaf=0.7)
    if arm.startswith('C6z'): return dict(level='C5', f=f_fit, lamp_z=float(arm[3:]), low_leaf=0.7)
    if arm.startswith('C6'):
        key, x = arm[2:].split('x'); assert key in C6F; return dict(level='C5', f=f_fit, change=(key, float(x)), low_leaf=0.7)
    raise SystemExit(arm)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('batch'); ap.add_argument('arms', nargs='+'); ap.add_argument('--f', type=float, default=None); ap.add_argument('--m', type=int, nargs='*', default=None); a = ap.parse_args()
    res = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), batch=a.batch, arms=a.arms, cases={}, repro={})
    for arm in a.arms:
        sp = arm_spec(arm, a.f)
        for m in (sp.get('seeds') or (a.m if a.m is not None else range(16))):
            seed = SEEDS[m]; D = PREP / f'{arm}_m{m:02d}'
            if (D / 'native' / 'prepared.json').exists() and ((D / 'v2_config.json').exists() or arm == 'B0'):
                res['cases'][D.name] = dict(arm=arm, m=m, seed=seed, model_sha=sha(D / 'native/working/Model.rgg'), reused=True); continue
            if D.exists(): shutil.rmtree(D)
            if sp.get('lamp_z') or sp.get('change') or sp.get('order'):
                note = build(D, seed, m, sp.get('lamp_z', 0.6), sp.get('change'), sp.get('order'))
            else:
                B = PREP / f'B0_m{m:02d}'
                if not (B / 'native' / 'prepared.json').exists():
                    build(B, seed, m)
                    ok = sha(B / 'native/working/Model.rgg') == DIVSENS[f'A_m{m:02d}']; res['repro'][B.name] = ok
                    assert ok, f'reproduction gate failed {B.name}'
                if arm == 'B0':
                    res['cases'][B.name] = dict(arm=arm, m=m, seed=seed, model_sha=sha(B / 'native/working/Model.rgg'), repro_divsens=True); continue
                shutil.copytree(B, D); note = {}
            rb = patch_case(D, sp['level'], sp.get('f', 1.0), sp.get('test'), sp.get('low_leaf', P.LOW_LEAF_SCALE), sp.get('inter_all', 1.0), sp.get('flowers_fixed', 0), sp.get('c4l', False), sp.get('c4t', False), sp.get('divergence'), sp.get('c4b', False), sp.get('c4c', False), sp.get('rachis'))
            res['cases'][D.name] = dict(arm=arm, m=m, seed=seed, level=sp['level'], f=sp.get('f', 1.0), lamp_z=sp.get('lamp_z', 0.6), change=sp.get('change'), note=note,
                                        model_sha=sha(D / 'native/working/Model.rgg'), rebind=rb)
        print(arm, 'prepared', flush=True)
    DS = HERE / 'dataset' / a.batch; DS.mkdir(parents=True, exist_ok=True)
    def tar(name, members):
        p = DS / f'{name}.tar.gz'
        with tarfile.open(p, 'w:gz') as t:
            for src, arc in members: t.add(src, arcname=arc, filter=lambda ti: None if '__pycache__' in ti.name else ti)
        return p
    runs = tar('runs', [(PREP / c, f'prepared/{c}') for c in res['cases']])
    code = tar('code', [(HERE / 'kaggle_kernel' / 'ev_case.py', 'evcode/ev_case.py'), (HERE / 'kaggle_kernel' / 'v2_validate.py', 'evcode/v2_validate.py')])
    core = Path('C:/Program Files/GroIMP 2.2.1/core.jar')
    eng = {p.name: sha(p) for p in sorted((DEV / 'tools').glob('*')) if p.is_file() and p.suffix in ('.py', '.rgg')}
    man = dict(schema='engv2-kaggle-input-1', batch=a.batch, created=res['started'], cases=list(res['cases']), tarballs={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in (runs, code)},
               groimp=dict(core_sha256=sha(core)), engine_code=eng)
    write_json(DS / 'ENGV2_MANIFEST.json', man)
    slug = f'engv2-{a.batch.lower()}-20261104'
    write_json(DS / 'dataset-metadata.json', dict(title=slug, id=f'dolgfish/{slug}', licenses=[dict(name='CC0-1.0')]))
    res['dataset'] = dict(slug=slug, **man['tarballs']); res['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    write_json(HERE / f'PREPARE_{a.batch}.json', res); print(json.dumps({k: v for k, v in res.items() if k != 'cases'}, indent=1))


if __name__ == '__main__':
    main()
