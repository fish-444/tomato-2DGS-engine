"""Engine-session runs on Codex's engine hooks (2026-10-15): Session G leaf development cases, driven from the layer.

python -X utf8 -m tools.microtom_hook_runs g-run --out build/g_runs_20261015_01
python -X utf8 -m tools.microtom_hook_runs g-report --runs build/g_runs_20261015_01

Each case is a seed-route member built by the layer (`microtom_forecast.build_member_inputs`, registry declared values = the base member),
plus an `engine_hooks` block (schema fsp-engine-hooks-1; `leaf_development` and `projected_shading` on, `planned_axes` off so the G effect
is isolated), executed through the public V3 path (`microtom_forecast.run_member`: prepare -> run -> collect). The hook parameters come from
the registry rows Codex appended (LEAFLET_GROWTH_ORDER, LEAFLET_STAGGER_DD, LEAF_AGEING_DECAY_A, all user_assumed). A hooks-off case and its
replay give the regression baseline and the native nondeterminism of this route. Fixture/seed geometry: engineering results, not plant
predictions; nothing here edits an engine file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from tools import microtom_forecast as mf
from tools import microtom_parameters as mp
from tools.vtc_growth_io import ROOT, ContractError, read_json, require, rows, write_csv, write_json

HOOK_SCHEMA = 'fsp-engine-hooks-1'
SWITCHES = ('planned_axes', 'per_axis_development', 'leaf_development', 'projected_shading')
G_RAYS = 30_000          # the hooks test value (ENGINE_HOOKS_EXECUTION_2026-10-14: G seed-route test)
G_DAYS = 30
SHADING_DAYS = (0, 10, 18, 30)
TABLE_DAYS = (0, 10, 18, 24, 30)
TAU95_DD = 268.75572192561674  # Codex: normalized sigmoid tau95 for MID_DD 150, K 0.025
MATURE_FRACTION = 0.95
# 7 declared runs (work order): stagger set at a = 0, a set at stagger 38; plus the hooks-off baseline and its replay
G_CASES = [('off', None, None), ('off_replay', None, None),
           ('s0_a0', 0, 0.0), ('s19_a0', 19, 0.0), ('s38_a0', 38, 0.0), ('s76_a0', 76, 0.0),
           ('s38_a0.001', 38, 0.001), ('s38_a0.002', 38, 0.002), ('s38_a0.003', 38, 0.003)]
NUMERIC_TOL = 1e-12  # addendum section 3 acceptance rule (absolute, numeric columns), applied only if replay shows nondeterminism


def hook_parameter(row: dict, value) -> dict:
    return dict(value=value, unit=row['unit'], source_kind=row['source_kind'], source_ref=f'{row["key"]}: {row["source_ref"]}',
                cultivar=row['cultivar'], uncertainty=row['uncertainty'])


def g_hooks(registry: list[dict], stagger: float, decay: float) -> dict:
    by = {r['key']: r for r in registry}
    for k in ('LEAFLET_GROWTH_ORDER', 'LEAFLET_STAGGER_DD', 'LEAF_AGEING_DECAY_A'):
        require(k in by, f'registry lacks {k} (Codex hook rows)')
    require(stagger in by['LEAFLET_STAGGER_DD']['candidates'] and decay in by['LEAF_AGEING_DECAY_A']['candidates'], 'value outside the declared scenarios')
    require(decay == 0 or 1 / decay > TAU95_DD, f'a = {decay} violates 1/a > tau95 ({TAU95_DD} degC d)')
    return dict(schema=HOOK_SCHEMA, planned_axes=False, per_axis_development=False, leaf_development=True, projected_shading=True,
                parameters=dict(LEAFLET_GROWTH_ORDER=hook_parameter(by['LEAFLET_GROWTH_ORDER'], by['LEAFLET_GROWTH_ORDER']['value']),
                                LEAFLET_STAGGER_DD=hook_parameter(by['LEAFLET_STAGGER_DD'], stagger),
                                LEAF_AGEING_DECAY_A=hook_parameter(by['LEAF_AGEING_DECAY_A'], decay)))


def folder_size_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file()) / 1e6, 2)


def prune_working_checkpoints(run_dir: Path) -> list[str]:
    """CHECKPOINT_RETENTION = working_last_1: keep only the newest native/working/checkpoints/step_*; returns what was removed."""
    ck = Path(run_dir) / 'working' / 'checkpoints'
    if not ck.is_dir():
        return []
    steps = sorted((p for p in ck.iterdir() if p.is_dir() and p.name.startswith('step_')), key=lambda p: int(p.name.split('_')[1]))
    removed = []
    for p in steps[:-1]:
        shutil.rmtree(p); removed.append(p.name)
    return removed


def g_run(out: Path, cases=G_CASES, timeout_s: float = 1800) -> Path:
    out = Path(out)
    require(out.resolve().is_relative_to((ROOT / 'build').resolve()), 'runs live below build/')
    out.mkdir(parents=True, exist_ok=True)
    registry = mp.load_registry()
    scenario = mf.default_scenario('seed_route', horizons_days=[G_DAYS], members=1, seed=11, rays=G_RAYS, diagnostic=True)
    record = []
    for name, stagger, decay in cases:
        folder = out / name
        if (folder / 'case.json').exists():
            record.append(read_json(folder / 'case.json')); continue
        built = mf.build_member_inputs(folder, scenario, registry, 0, {}, issue_boundary=0)
        req = read_json(built['inputs'] / 'request.json')
        if stagger is not None:
            req['engine_hooks'] = g_hooks(registry, stagger, decay)
            (built['inputs'] / 'request.json').write_text(json.dumps(req, indent=2), encoding='utf-8')
        run_dir = folder / 'run'
        results = mf.run_member(built['inputs'], run_dir, timeout_s=timeout_s)
        removed = prune_working_checkpoints(run_dir)
        case = dict(case=name, stagger_dd=stagger, decay_a=decay, hooks=req.get('engine_hooks', 'absent (hooks off)'), results=str(results.relative_to(out)),
                    intervals=req['intervals'], rays=req['runtime']['rays'], seed=req['runtime']['seed'], pairs=req['parameters']['PAIRS']['value'],
                    first_leaves=req['parameters']['FIRST_LEAVES']['value'], cadence=req['parameters']['CADENCE']['value'],
                    pruned_checkpoints=removed, size_mb=folder_size_mb(folder),
                    execution=read_json(run_dir / 'execution.json').get('status'), elapsed_s=read_json(run_dir / 'execution.json').get('elapsed_s'))
        write_json(folder / 'case.json', case); record.append(case)
    write_json(out / 'runs.json', dict(cases=record, total_size_mb=folder_size_mb(out), route='seed_route (layer build_member_inputs, registry declared values)',
                                       statement='engineering results on generated seed-route plants, not plant predictions'))
    return out


# ---------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------

def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_case(out: Path, name: str) -> dict:
    c = read_json(Path(out) / name / 'case.json')
    res = Path(out) / c['results']
    data = dict(case=c, states=rows(res / 'states.csv'), ledger=rows(res / 'ledger.csv'))
    for opt in ('leaf_growth_clocks.csv', 'leaflet_shading.csv'):
        data[opt[:-4]] = rows(res / opt) if (res / opt).exists() else []
    data['results_dir'] = res
    return data


def table1(d: dict) -> list[dict]:
    out = []
    plants = sorted({s['plant_instance_id'] for s in d['states']})
    shade = defaultdict(list)
    for r in d['leaflet_shading']:
        v = _f(r['shading_rate'])
        if v is not None:
            shade[(int(r['day']), r['plant_instance_id'])].append(v)
    for pid in plants:
        for day in TABLE_DAYS:
            ss = [s for s in d['states'] if s['plant_instance_id'] == pid and int(s['day']) == day]
            lf = [s for s in ss if s['class'] == 'Leaflet']
            sh = shade.get((day, pid), [])
            out.append(dict(plant=pid, day=day, compounds=sum(s['class'] == 'CompoundLeaf' for s in ss), leaflets=len(lf),
                            leaf_area_cm2=round(sum(float(s['area_m2']) for s in lf) * 1e4, 3),
                            mean_leaflet_shading=(round(sum(sh) / len(sh), 4) if sh else 'unavailable') if day in SHADING_DAYS else 'not computed (checkpoint days only)',
                            n_shading_leaflets=len(sh) if day in SHADING_DAYS else ''))
    return out


def mature_fraction(d: dict, days=(10, 18, 30)) -> list[dict]:
    """Two definitions per plant/day, both reported:
    - age-mature: every leaflet of the compound has physiological age >= tau95 (268.76 degC d, the Leaflet curve the hooks validate);
    - mass-mature: every leaf tissue (leaflet, rachis, petiole) has funded mass >= 0.95 of its maximum (`mass_fraction` in leaf_growth_clocks.csv).
    """
    out = []
    tau = defaultdict(lambda: defaultdict(list)); frac = defaultdict(lambda: defaultdict(list))
    for r in d['leaf_growth_clocks']:
        if r['organ_class'] not in ('Leaflet', 'RachisSegment', 'Petiole'):
            continue
        key = (int(r['day']), r['plant_instance_id']); comp = r['organ_id'].rsplit('/', 1)[0]
        frac[key][comp].append(_f(r['mass_fraction']) or 0.0)
        if r['organ_class'] == 'Leaflet':
            tau[key][comp].append(_f(r['physiological_age_dd']) or 0.0)
    for pid in sorted({s['plant_instance_id'] for s in d['states']}):
        for day in days:
            comps = frac.get((day, pid), {})
            if not comps:
                out.append(dict(plant=pid, day=day, compounds=0, age_mature=0, mass_mature=0, age_fraction='unavailable (no compound)', mass_fraction='unavailable (no compound)'))
                continue
            am = sum(all(v >= TAU95_DD for v in tau[(day, pid)].get(c, [0.0])) for c in comps)
            mm = sum(all(v >= MATURE_FRACTION for v in fr) for fr in comps.values())
            out.append(dict(plant=pid, day=day, compounds=len(comps), age_mature=am, mass_mature=mm,
                            age_fraction=round(am / len(comps), 3), mass_fraction=round(mm / len(comps), 3)))
    return out


def carbon(d: dict) -> list[dict]:
    acc = defaultdict(lambda: dict(gross=0.0, structural=0.0, reserve_end=None))
    for r in d['ledger']:
        a = acc[r['plant_instance_id']]
        a['gross'] += float(r.get('gross_gC') or r.get('signed_exchange_gC') or 0)
        a['structural'] += float(r.get('structural_gC') or r.get('structural_delta_gC') or 0)
        a['reserve_end'] = float(r['reserve_end_gC'])
    return [dict(plant=pid, gross_gC=round(v['gross'], 5), structural_gC=round(v['structural'], 5), reserve_day30_gC=round(v['reserve_end'], 5)) for pid, v in sorted(acc.items())]


def position_of(organ_id: str) -> str:
    return organ_id.rsplit('/', 1)[-1]


def compound_curve(d: dict, compound: str) -> list[dict]:
    """Per compound age (days since birth): area per leaflet position, rachis length, shading by position (where computed)."""
    ss = [s for s in d['states'] if s['organ_id'].startswith(compound + '/') or s['organ_id'] == compound]
    birth = next((int(s['birth_day']) for s in ss if s['organ_id'] == compound), None)
    if birth is None:
        return []
    sh = {(int(r['day']), r['organ_id']): _f(r['shading_rate']) for r in d['leaflet_shading'] if r['organ_id'].startswith(compound + '/')}
    out = []
    for day in sorted({int(s['day']) for s in ss}):
        cur = [s for s in ss if int(s['day']) == day]
        row = dict(day=day, age_days=day - birth)
        for s in cur:
            if s['class'] == 'Leaflet':
                row[f'area_mm2:{position_of(s["organ_id"])}'] = round(float(s['area_m2']) * 1e6, 2)
                v = sh.get((day, s['organ_id']))
                if day in SHADING_DAYS:
                    row[f'shading:{position_of(s["organ_id"])}'] = round(v, 4) if v is not None else 'unavailable'
        row['rachis_length_mm'] = round(sum(float(s['length_m']) for s in cur if s['class'] == 'RachisSegment') * 1e3, 2)
        row['petiole_length_mm'] = round(sum(float(s['length_m']) for s in cur if s['class'] == 'Petiole') * 1e3, 2)
        out.append(row)
    return out


def clock_curve(d: dict, organ_id: str) -> list[dict]:
    return [dict(day=int(r['day']), T=round(float(r['linear_thermal_age_dd']), 2), offset=r['offset_dd'], tau=round(float(r['physiological_age_dd']), 2),
                 mass_fraction=round(float(r['mass_fraction']), 4)) for r in d['leaf_growth_clocks'] if r['organ_id'] == organ_id]


def regression(out: Path, a: str, b: str) -> dict:
    """Byte hashes per CSV, then numeric comparison (rows, non-numeric exact, numeric |delta|)."""
    ra, rb = Path(out) / read_json(Path(out) / a / 'case.json')['results'], Path(out) / read_json(Path(out) / b / 'case.json')['results']
    require(ra.is_dir() and rb.is_dir() and any(ra.glob('*.csv')), f'regression inputs missing: {ra} / {rb}')  # never report an empty comparison as equal
    res = {}
    for name in sorted(p.name for p in ra.glob('*.csv')):
        if not (rb / name).exists():
            res[name] = dict(status='missing_in_' + b); continue
        ha, hb = hashlib.sha256((ra / name).read_bytes()).hexdigest(), hashlib.sha256((rb / name).read_bytes()).hexdigest()
        if ha == hb:
            res[name] = dict(status='byte_identical'); continue
        xa, xb = list(csv.reader((ra / name).open(encoding='utf-8'))), list(csv.reader((rb / name).open(encoding='utf-8')))
        if len(xa) != len(xb) or (xa and xa[0] != xb[0]):
            res[name] = dict(status='row_or_header_difference', rows=(len(xa), len(xb))); continue
        worst, nonnum = 0.0, 0
        for ra_, rb_ in zip(xa[1:], xb[1:]):
            for u, v in zip(ra_, rb_):
                if u == v:
                    continue
                fu, fv = _f(u), _f(v)
                if fu is None or fv is None:
                    nonnum += 1
                else:
                    worst = max(worst, abs(fu - fv))
        res[name] = dict(status='numeric_differences' if not nonnum else 'nonnumeric_differences', max_abs_delta=worst, nonnumeric_cells=nonnum)
    return res


def strip_image(d_list: list[tuple[str, dict]], compound: str, ages: list[int], out_png: Path) -> Path:
    """Top view (x-y) of one compound at 4 ages, one row per case; triangles from the published meshes.csv."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, axes = plt.subplots(len(d_list), len(ages), figsize=(3.0 * len(ages), 3.0 * len(d_list)), squeeze=False)
    for i, (label, d) in enumerate(d_list):
        birth = next(int(s['birth_day']) for s in d['states'] if s['organ_id'] == compound)
        mesh = rows(d['results_dir'] / 'meshes.csv')
        bounds = []
        polys_by_age = {}
        for age in ages:
            day = birth + age
            vr = [r for r in mesh if int(r['state']) == day and r['organ_id'].startswith(compound + '/')]
            by = defaultdict(list)
            for r in vr:
                by[r['organ_id']].append((float(r['x_m']) * 1e3, float(r['y_m']) * 1e3))
            polys = []
            for oid, pts in by.items():
                polys += [pts[k:k + 3] for k in range(0, len(pts) - len(pts) % 3, 3)]
            polys_by_age[age] = polys
            for p in polys:
                bounds += p
        if bounds:
            xs, ys = [p[0] for p in bounds], [p[1] for p in bounds]
            cx, cy, half = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 2
        for j, age in enumerate(ages):
            ax = axes[i][j]
            polys = polys_by_age[age]
            if polys:
                ax.add_collection(PolyCollection(polys, facecolor='#4a8f3c', edgecolor='#2d5a25', linewidth=0.2))
                ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
            else:
                ax.text(0.5, 0.5, 'no area yet', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{label}  age {age} d', fontsize=8)
    fig.suptitle(f'{compound} top view (mm, same scale per row) - generated seed-route plant, engineering result', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------------------------------------------
# Part F — new side shoots on the hooks (bud_break), storage-capped: summarize, verify, then delete geometry
# ---------------------------------------------------------------------------------------------------------------

F_HOST = ROOT / 'build' / 'forecast_20261014_01' / 'scenarios' / 'lit_base' / 'members' / 'm00' / 'run' / 'native'  # Session A lit_base, 30 d
F_PLAN = ROOT / 'build' / 'budbreak_plan_20261014_01' / 'plan.csv'
F_LABELS = ('low', 'base', 'high')
F_MEMBERS = 4
F_SEED = 11
F_KEEP_FULL = ('base', 0)
F_CAP_BYTES = 10_000_000_000
F_TIMEOUT = 2400
F_DAYS = (0, 10, 18, 24, 30)
G_PREFERRED = (38, 0.0)  # user decision 2026-10-15 15:10: basipetal, stagger 38; a = 0 (not yet decided)
GEOMETRY_FILES = ('meshes.csv', 'consumed.csv')


def f_cases() -> list:
    return [('control', None, None)] + [(f'{lab}_m{m:02d}', lab, m) for lab in F_LABELS for m in range(F_MEMBERS)]


def new_axis_draw(member: int, seed: int = F_SEED) -> dict:
    """Session F row AXIS_LEAVES_TO_TRUSS_NEW_AXIS {2,3,4}: one seeded draw per member (the same member draws the same value in every scenario)."""
    import random
    row = next(r for r in mp.load_registry() if r['key'] == 'AXIS_LEAVES_TO_TRUSS_NEW_AXIS')
    v = random.Random(f'{seed}:{member}:AXIS_LEAVES_TO_TRUSS_NEW_AXIS').choice(row['candidates'])
    eng = mp.engine_value_for(dict(row, engine_key='FIRST_LEAVES'), v)
    return dict(value=v, unit=row['unit'], engine_first_leaves=eng,
                conversion=('engine_value_for applied as instructed; the registered unit is leaves_on_new_axis, so no U5 offset is applied '
                            '(U5 concerns the cotyledon pair; open convention question recorded in Session F)'))


def f_prepare(out: Path, label, member, plan: Path = None, ray_seed: int | None = None, rays: int | None = None,
              reserve_cap: bool = False, overrides: dict | None = None, leaflet_outline: bool = False):
    from tools.fsp_engine_hooks import AXILLARY_STAGE, SCHEMA
    from tools.fsp_engine_hooks_acceptance import fixture_axes, parameter
    from tools.tomato_fsp_pipeline import retention_policy
    from tools.vtc_development import prepare
    from tools.vtc_growth_io import fresh_build, sha
    plan = Path(plan) if plan is not None else F_PLAN
    fresh_build(out); inputs = out / 'inputs'; shutil.copytree(F_HOST / 'inputs', inputs)
    request = read_json(inputs / 'request.json')
    registry = mp.load_registry()
    host_seed = request['runtime']['seed']
    if ray_seed is not None:
        request['runtime']['seed'] = int(ray_seed)  # F4 noise floor: only the ray-tracing seed changes (lm.setSeed)
    if rays is not None:
        request['runtime']['rays'] = int(rays)  # F4 addendum: 4.8M-ray plant/compound check of the full-geometry run only
    hooks = dict(schema=SCHEMA, **{k: False for k in sorted(SWITCHES)})
    hooks.update(per_axis_development=True, leaf_development=True, planned_axes=label is not None)
    g = g_hooks(registry, *G_PREFERRED)['parameters']
    for k in g:
        g[k]['source_ref'] += ' | preferred G scenario, user decision 2026-10-15 15:10 (stagger 38 basipetal; a 0 pending)'
    hooks['parameters'] = dict(g)
    if label is not None:
        hooks['parameters'][AXILLARY_STAGE] = parameter('bud_break', 'categorical', 'Reviewed engine hooks addendum decision 2026-10-15 (bud_break)')
    # always explicit: a fresh request without the switch now defaults to library outlines (user decision 2026-10-15 21:02);
    # F3/F4 keep diamonds for comparability (F4 addendum 3)
    hooks['leaflet_outline'] = bool(leaflet_outline)
    if reserve_cap:
        hooks['reserve_cap'] = True  # F4 Stage 2/3: Codex H-R1 flat hook; defaults resolved into native/prepared.json
    for key, rec in (overrides or {}).items():
        hooks['parameters'][key] = dict(rec)  # F4 Stage 3: Codex H-R2 provenance records (LEAFLET_MID_DD, LEAFLET_K, PHYLLOCHRON_DD)
    request['engine_hooks'] = hooks
    # the acceptance adapter selects plan rows labelled base/member 0: hand it exactly this run's rows, relabelled, and record the mapping
    sel = out / 'plan_selection.csv'
    src = rows(plan)
    chosen = [dict(r, scenario_id='base', member_index='0') for r in src if label is not None and r['scenario_id'] == label and r['member_index'] == str(member)]
    with sel.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(src[0])); w.writeheader(); w.writerows(chosen)
    fixture_axes(inputs, request, sel)
    draw = new_axis_draw(member) if member is not None else None
    for a in request['axes']:
        if a.get('origin') == 'axillary':
            a['source_ref'] = f'{plan}:{label}:member{member} (rows relabelled base/0 in plan_selection.csv only for the acceptance adapter filter)'
            a['plan_sha256'] = sha(plan)
            a['parameters']['FIRST_LEAVES']['value'] = draw['engine_first_leaves']
            a['parameters']['FIRST_LEAVES']['source_ref'] = f'AXIS_LEAVES_TO_TRUSS_NEW_AXIS member draw {draw["value"]} {draw["unit"]}; {draw["conversion"]}'
    write_json(inputs / 'request.json', request)
    manifest = dict(case=out.name, scenario_id=label or 'control (no new shoots)', member_index=member, plan=str(plan), plan_sha256=sha(plan),
                    ray_seed=request['runtime']['seed'], host_ray_seed=host_seed, rays=request['runtime']['rays'],
                    selected_rows=len(chosen), emerged_rows=sum(r['emerged'] == 'True' for r in chosen), new_axis_draw=draw,
                    host=str(F_HOST), axillary_start_stage='bud_break' if label else 'disabled (control)',
                    reserve_cap=bool(reserve_cap), leaflet_outline=bool(leaflet_outline), overrides={k: v['value'] for k, v in (overrides or {}).items()},
                    leaf_development=dict(LEAFLET_GROWTH_ORDER='basipetal', LEAFLET_STAGGER_DD=G_PREFERRED[0], LEAF_AGEING_DECAY_A=G_PREFERRED[1],
                                          label='preferred G scenario (user decision 2026-10-15 15:10), a = 0 pending decision'),
                    provenance='model_generated', origin='axillary',
                    statement='new shoot emergence is generated from an assumed probability without evidence; fixture geometry - engineering result, not a plant prediction',
                    fixture_note='host p1_state_20261011_01 fixture plants arrive with 8 true leaves and no truss (Session A inconsistency), unchanged')
    write_json(out / 'fixture_manifest.json', manifest)
    prepare(inputs, out / 'native')
    policy = retention_policy('working_last_1')
    policy.update(committed_steps='hash-verified acceptance interval slices', geometry_streams='deleted after verified summaries except the kept full run (work order 2026-10-15)')
    write_json(out / 'retention_policy.json', policy)
    return out / 'native', manifest


def f_summaries(run: Path, manifest: dict) -> dict:
    """Small tables kept for every run; written, re-read and hash-recorded before any geometry is deleted."""
    nat = run / 'native' / 'working'
    req = read_json(run / 'native' / 'inputs' / 'request.json')
    planned_ids = {(a['plant_instance_id'], a['axis_id']) for a in req.get('axes', []) if a.get('origin') == 'axillary'}
    states = rows(nat / 'states.csv')
    out = run / 'summaries'; out.mkdir(exist_ok=True)
    counts = []
    for day in F_DAYS:
        for pid in sorted({s['plant_instance_id'] for s in states}):
            ss = [s for s in states if int(s['day']) == day and s['plant_instance_id'] == pid]
            new = [s for s in ss if (pid, s['axis_id']) in planned_ids]
            counts.append(dict(day=day, plant=pid, organs=len(ss), logical_axes=len({s['axis_id'] for s in ss if s['class'] != 'Plant' and s['axis_id']}),
                               new_axes_present=len({s['axis_id'] for s in new if s['class'] == 'Axis'}), compounds=sum(s['class'] == 'CompoundLeaf' for s in ss),
                               leaflets=sum(s['class'] == 'Leaflet' for s in ss), new_axis_compounds=sum(s['class'] == 'CompoundLeaf' for s in new),
                               new_axis_leaflets=sum(s['class'] == 'Leaflet' for s in new),
                               leaf_area_cm2=round(sum(float(s['area_m2']) for s in ss if s['class'] == 'Leaflet') * 1e4, 4),
                               new_axis_leaf_area_cm2=round(sum(float(s['area_m2']) for s in new if s['class'] == 'Leaflet') * 1e4, 4)))
    write_csv(out / 'organ_counts.csv', counts)
    env = rows(run / 'native' / 'inputs' / 'environment.csv')
    hours = {i: float(e['light_hours']) for i, e in enumerate(env)}
    ax = defaultdict(lambda: dict(gross_gC=0.0, allocated_gC=0.0, structural_gC=0.0, absorbed_umol=0.0))
    for r in rows(nat / 'axis_flux.csv'):
        a = ax[(r['plant_instance_id'], r['axis_id'])]
        for k in ('gross_gC', 'allocated_gC', 'structural_gC'):
            a[k] += float(r[k])
        a['absorbed_umol'] += float(r['absorbed_umol_s']) * hours[int(r['interval'])] * 3600
    write_csv(out / 'carbon_by_axis.csv', [dict(plant=p, axis=a, new_axis=(p, a) in planned_ids, **{k: round(v, 9) for k, v in d.items()}) for (p, a), d in sorted(ax.items())])
    write_csv(out / 'ledger.csv', rows(nat / 'ledger.csv'))
    cls = {s['organ_id']: s['class'] for s in states}
    leaf_light = [dict(interval=r['interval'], plant=r['plant_instance_id'], organ_id=r['organ_id'], axis_id=r['axis_id'],
                       absorbed_umol_day=round(float(r['absorbed_umol_s']) * hours[int(r['interval'])] * 3600, 6))
                  for r in rows(nat / 'flux.csv') if cls.get(r['organ_id']) == 'Leaflet']
    write_csv(out / 'leaflet_light.csv', leaf_light)
    shading = run / 'shading' / 'leaflet_shading.csv'
    if shading.exists():
        shutil.copyfile(shading, out / 'leaflet_shading.csv')
    files = {p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), rows=sum(1 for _ in p.open(encoding='utf-8')) - 1) for p in sorted(out.glob('*.csv'))}
    for name, meta in files.items():  # verification: re-read and compare
        require(sum(1 for _ in (out / name).open(encoding='utf-8')) - 1 == meta['rows'] and meta['rows'] > 0, f'summary {name} empty or unreadable')
    require('leaflet_shading.csv' in files, 'shading summary missing; geometry must not be deleted')
    s = dict(case=run.name, files=files, emerged_axes=len(planned_ids), verified=True)
    write_json(out / 'summaries_manifest.json', s)
    return s


def f_delete_geometry(run: Path) -> dict:
    """Delete working geometry + committed archive geometry, keeping every manifest and recording what was removed (with its recorded hash)."""
    nat = run / 'native' / 'working'
    require((run / 'summaries' / 'summaries_manifest.json').is_file() and read_json(run / 'summaries' / 'summaries_manifest.json')['verified'], 'summaries not verified')
    completion = read_json(nat / 'completion.json')
    removed = []
    for name in GEOMETRY_FILES:
        f = nat / name
        if f.exists():
            removed.append(dict(path=str(f.relative_to(run)), bytes=f.stat().st_size, recorded_sha256=completion['files'].get(name, 'unrecorded'))); f.unlink()
    for ck in sorted((nat / 'checkpoints').glob('step_*')):
        size = sum(p.stat().st_size for p in ck.rglob('*') if p.is_file())
        removed.append(dict(path=str(ck.relative_to(run)), bytes=size, recorded_sha256='see committed/step_NNNN/checkpoint.json')); shutil.rmtree(ck)
    for step in sorted((run / 'committed').glob('step_*')):
        meta = read_json(step / 'checkpoint.json')
        for name in GEOMETRY_FILES:
            f = step / name
            if f.exists():
                removed.append(dict(path=str(f.relative_to(run)), bytes=f.stat().st_size, recorded_sha256=meta['files'].get(name))); f.unlink()
    rec = dict(case=run.name, removed=removed, removed_bytes=sum(r['bytes'] for r in removed),
               note='geometry deleted after verified summaries (work order 2026-10-15 option 1); manifests and hashes kept; archive no longer reconstructs meshes/consumed')
    write_json(run / 'geometry_deletion_manifest.json', rec)
    return rec


def f_run(out_root: Path, cases: list | None = None, plan: Path | None = None, keep_full: tuple | str = F_KEEP_FULL) -> Path:
    """Case spec: (name, label, member[, ray_seed[, rays[, timeout]]][, {'reserve_cap': bool, 'overrides': {...}}]).
    keep_full: (label, member) of the run whose geometry is kept, or that run's case name."""
    import subprocess, sys, time
    from tools.fsp_engine_hooks_acceptance import run_case
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    ledger_path = out_root / 'runs.json'
    state = read_json(ledger_path) if ledger_path.exists() else dict(cases=[])
    done = {c['case'] for c in state['cases']}
    for case_spec in (cases if cases is not None else f_cases()):
        has_opts = isinstance(case_spec[-1], dict)
        opts = case_spec[-1] if has_opts else {}
        case_spec = case_spec[:-1] if has_opts else case_spec
        name, label, member = case_spec[:3]
        ray_seed = case_spec[3] if len(case_spec) > 3 else None
        rays = case_spec[4] if len(case_spec) > 4 else None
        timeout = case_spec[5] if len(case_spec) > 5 else F_TIMEOUT
        if name in done:
            continue
        total = folder_size_mb(out_root) * 1e6
        require(total < F_CAP_BYTES, f'storage cap reached before {name}: {total/1e9:.2f} GB')
        run = out_root / name
        t0 = time.time()
        native, manifest = f_prepare(run, label, member, plan, ray_seed, rays, opts.get('reserve_cap', False), opts.get('overrides'))
        run_case(native, timeout_s=timeout)
        audit = None
        if opts.get('reserve_cap'):
            from tools.fsp_reserve_expansion_acceptance import audit as reserve_audit
            audit = reserve_audit(native)  # writes <run>/carbon_audit.csv + audit.json from working states/flux/ledger (kept after deletion)
        t1 = time.time()
        sh = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.fsp_projected_shading', '--run-dir', str(native), '--out', str(run / 'shading'),
                             '--days', '0', '10', '18', '30'], cwd=ROOT, capture_output=True, text=True)
        require(sh.returncode == 0, f'shading failed for {name}: {sh.stderr[-800:]}')
        t2 = time.time()
        summ = f_summaries(run, manifest)
        peak = folder_size_mb(run)
        keep = name == keep_full if isinstance(keep_full, str) else (label, member) == tuple(keep_full)
        deletion = None if keep else f_delete_geometry(run)
        case = dict(case=name, scenario_id=label or 'control', member_index=member, ray_seed=manifest['ray_seed'], rays=manifest['rays'], native_status=read_json(native / 'execution.json').get('status'),
                    acceptance=read_json(run / 'acceptance_validation.json').get('status') if (run / 'acceptance_validation.json').exists() else 'missing',
                    emerged_axes=summ['emerged_axes'], new_axis_draw=manifest['new_axis_draw'], native_s=round(t1 - t0), shading_s=round(t2 - t1),
                    size_mb_before_deletion=peak, kept_full_geometry=keep, removed_mb=round(deletion['removed_bytes'] / 1e6, 1) if deletion else 0.0,
                    size_mb_final=folder_size_mb(run), running_total_mb=folder_size_mb(out_root),
                    reserve_cap=manifest['reserve_cap'], overrides=manifest['overrides'],
                    audit=None if audit is None else {k: audit[k] for k in ('status', 'intervals', 'initial_excess_intervals', 'downregulated_gC', 'max_abs_residual_gC')})
        state['cases'].append(case); write_json(ledger_path, state)
    state['total_size_mb'] = folder_size_mb(out_root); write_json(ledger_path, state)
    return out_root


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='op', required=True)
    a = sub.add_parser('g-run'); a.add_argument('--out', type=Path, required=True); a.add_argument('--timeout', type=float, default=1800)
    a = sub.add_parser('f-run'); a.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.op == 'g-run':
            print(json.dumps(read_json(g_run(a.out, timeout_s=a.timeout) / 'runs.json'), indent=2)[:4000])
        elif a.op == 'f-run':
            print(json.dumps(read_json(f_run(a.out) / 'runs.json'), indent=2)[:6000])
    except (ContractError, ValueError, KeyError, OSError) as exc:
        p.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
