"""Strict IO and independent accounting for the conditional native VTC mode."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'vtc-python-phase1-1'
HEADERS = {
    'plants.csv': 'plant_instance_id,source_plant_id,axis_id,age_days,reserve_gC,tx_mm,ty_mm,tz_mm,yaw_deg,transform_source,mass_source',
    'leaflets.csv': 'plant_instance_id,axis_id,compound_id,leaflet_id,parent_id,age_days,x_mm,y_mm,z_mm,length_mm,width_mm,dry_mass_g,potential_dm_g_day,geometry_source,mass_source,age_source',
    'environment.csv': 'interval,start_time,end_time,light_hours,temp_day_c,temp_night_c,co2_ppm,source_kind',
    'light_input.csv': 'reference_ppfd_umol_m2_s,source_kind,measured_at',
    'lamps.csv': 'lamp_id,x_mm,y_mm,z_mm,power_umol_s,source_kind',
}
COEFFICIENTS = ('amax_umol_m2_s', 'alpha_mol_mol', 'theta', 'respiration_umol_m2_s',
                'carbon_fraction_gC_gDM', 'growth_efficiency', 'molar_carbon_g_mol')


class ContractError(ValueError):
    """Input or native output cannot support a successful run."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path: Path, header: str | None = None) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), f'invalid header: {path}')
        if header:
            require(reader.fieldnames == header.split(','), f'wrong header: {path}')
        result = list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in result), f'malformed CSV: {path}')
    return result


def write_csv(path: Path, data: list[dict], headers: list[str] | None = None) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers or list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def number(value: Any, name: str, minimum: float | None = None) -> float:
    require(not isinstance(value, bool), f'boolean number: {name}')
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f'unavailable/non-numeric: {name}') from exc
    require(math.isfinite(v), f'nonfinite: {name}')
    if minimum is not None:
        require(v >= minimum, f'negative/out of range: {name}')
    return v


def aware(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    require(dt.tzinfo is not None and dt.utcoffset() is not None, 'aware timestamp required')
    return dt


BUILD_ROOT_ENV = 'FSP_BUILD_ROOT'  # F11 C2: optional output root outside the synced project folder


def build_roots() -> list[Path]:
    """Allowed output roots: the project build/ plus FSP_BUILD_ROOT when it is set (absolute, outside the project)."""
    roots = [(ROOT / 'build').resolve()]
    extra = os.environ.get(BUILD_ROOT_ENV, '').strip()
    if extra:
        root = Path(extra)
        require(root.is_absolute(), f'{BUILD_ROOT_ENV} must be an absolute path')
        root = root.resolve()
        require(not root.is_relative_to(ROOT.resolve()), f'{BUILD_ROOT_ENV} must lie outside the project folder (use build/ for that)')
        roots.append(root)
    return roots


def relative_to_root(path: Path) -> dict:
    """Record a path as root + relative path (F11 C2-3), never as a bare absolute path."""
    target = Path(path).resolve()
    for root in build_roots():
        if target.is_relative_to(root):
            return dict(root=str(root), root_kind='project_build' if root == (ROOT / 'build').resolve() else BUILD_ROOT_ENV,
                        relative=target.relative_to(root).as_posix())
    return dict(root='', root_kind='outside_declared_roots', relative=str(target))


def fresh_build(path: Path) -> Path:
    target = path.resolve()
    roots = build_roots()
    require(any(target.is_relative_to(r) and target != r for r in roots),
            'output must be below project build/' + (f' or {BUILD_ROOT_ENV}' if len(roots) > 1 else ''))
    require(not target.exists(), f'fresh output required: {target}')
    target.mkdir(parents=True)
    return target


PUBLISH_RETRY_DELAYS_S = (0.5, 1.0, 2.0, 4.0, 8.0)  # F11: bounded retries for atomic directory publication under Windows/OneDrive locks


def rename_with_retry(source: Path, target: Path) -> int:
    """Atomic directory rename with bounded, logged retries on a refused handle; returns the retry count used.
    The last PermissionError is raised unchanged so a persistent lock is still an explicit failure."""
    import time
    for i, delay in enumerate((*PUBLISH_RETRY_DELAYS_S, None)):
        try:
            source.rename(target)
            return i
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)
    return len(PUBLISH_RETRY_DELAYS_S)


def identifier(value: str) -> None:
    require(bool(value) and not any(c in value for c in '\r\n\t,"\\'), f'unsafe/empty identifier: {value!r}')


def ratio_table(path: Path) -> list[dict]:
    table = rows(path)
    require(bool(table), 'empty ratio file')
    grid = {'grid_row', 'grid_col', 'ratio_to_mean'} <= set(table[0])
    slot = {'slot_id', 'ratio_now'} <= set(table[0])
    require(grid or slot, 'ratio header must be grid or slot schema')
    keys = []
    for row in table:
        key = (row['grid_row'], row['grid_col']) if grid else row['slot_id']
        require(number(row['ratio_to_mean'] if grid else row['ratio_now'], 'ratio', 0) > 0, 'ratio must be positive')
        keys.append(key)
    require(len(set(keys)) == len(keys), 'duplicate ratio identity')
    if grid:
        require(set(keys) == {(str(r), str(c)) for r in range(1, 5) for c in range(1, 5)}, 'canonical grid must contain 16 cells')
        require(abs(sum(float(r['ratio_to_mean']) for r in table) / 16 - 1) < .005, 'ratio normalization is not 16-cell mean')
    return table


def load_package(folder: Path) -> dict:
    request = read_json(folder / 'request.json')
    for k, v in dict(schema_version=SCHEMA, dt_s=86400, frame='plant_local_mm',
                     geometry_basis='planar_rectangular_approximation', topology_scope='provided_axes',
                     completeness='supported_leaf_subset', run_status='assumption_conditioned',
                     evaluation_status='evaluated_unaccepted', biological_validation='not_established',
                     nutrient_feedback_enabled=False).items():
        require(request.get(k) == v, f'unsupported {k}: expected {v}')
    require(type(request.get('fixture')) is bool, 'fixture flag required')
    require(request.get('initialization_route') in ('synthetic_fixture', 'observation_fixture', 'observation_conditioned'), 'unsupported initialization route')
    require(request['fixture'] == request['initialization_route'].endswith('fixture'), 'fixture/route mismatch')
    require(type(request.get('freeze_geometry')) is bool, 'explicit freeze_geometry required')
    for key in ('run_id', 'scenario_id'):
        identifier(request[key])
    n = request['intervals']
    require(type(n) is int and 1 <= n <= 30, 'intervals must be 1..30 daily steps')
    start, end = aware(request['start_time']), aware(request['end_time'])
    require(end == start + timedelta(days=n), 'interval coverage mismatch')
    require(aware(request['observation_time']) == start, 'initial state must be at start_time; age is separate')
    for key in COEFFICIENTS:
        p = request['parameters'][key]
        number(p['value'], key, 0)
        require(p.get('source_kind') in ('synthetic_assumed', 'literature_other_cultivar', 'user_assumed'), f'parameter evidence: {key}')
        require(all(p.get(k) for k in ('source_ref', 'cultivar', 'domain', 'uncertainty')), f'incomplete parameter evidence: {key}')
    for key in ('theta', 'carbon_fraction_gC_gDM', 'growth_efficiency'):
        require(0 < request['parameters'][key]['value'] <= 1, f'{key} must be (0,1]')
    require(request['parameters']['molar_carbon_g_mol']['value'] == 12.011, 'unsupported molar carbon conversion')
    runtime = request['runtime']
    for key, maximum in (('rays', 100_000_000), ('depth', 20), ('seed', 2147483647)):
        require(type(runtime[key]) is int and (0 if key == 'seed' else 1) <= runtime[key] <= maximum, f'invalid runtime {key}')
    require(0 < number(runtime['balance_tolerance_gC'], 'tolerance') <= 1e-6, 'unsafe balance tolerance')
    tables = {name: rows(folder / name, header) for name, header in HEADERS.items()}
    plants, leaves = tables['plants.csv'], tables['leaflets.csv']
    require(bool(plants) and bool(leaves) and bool(tables['lamps.csv']), 'nonempty plant/leaf/lamp inventory required')
    plant_map = {p['plant_instance_id']: p for p in plants}
    require(len(plant_map) == len(plants), 'duplicate plant instance')
    for p in plants:
        for key in ('plant_instance_id', 'source_plant_id', 'axis_id'):
            identifier(p[key])
        for key in ('age_days', 'reserve_gC'):
            number(p[key], key, 0)
        for key in ('tx_mm', 'ty_mm', 'tz_mm', 'yaw_deg'):
            number(p[key], key)
        require(p['transform_source'] and p['mass_source'], 'plant transform/reserve source required')
    leaf_ids = set()
    for leaf in leaves:
        for key in ('axis_id', 'compound_id', 'leaflet_id', 'parent_id'):
            identifier(leaf[key])
        require(leaf['leaflet_id'] not in leaf_ids, 'duplicate leaflet ID')
        leaf_ids.add(leaf['leaflet_id'])
        require(leaf['plant_instance_id'] in plant_map, 'unknown plant')
        require(leaf['axis_id'] == plant_map[leaf['plant_instance_id']]['axis_id'], 'axis/plant mismatch')
        require(leaf['parent_id'] == leaf['compound_id'], 'explicit compound parent required')
        for key in ('length_mm', 'width_mm', 'dry_mass_g'):
            require(number(leaf[key], key, 0) > 0, f'positive {key} required')
        for key in ('age_days', 'potential_dm_g_day'):
            number(leaf[key], key, 0)
        require(float(leaf['age_days']) <= float(plant_map[leaf['plant_instance_id']]['age_days']), 'organ older than plant')
        for key in ('x_mm', 'y_mm', 'z_mm'):
            number(leaf[key], key)
        for key in ('geometry_source', 'mass_source', 'age_source'):
            require(bool(leaf[key]), f'explicit {key} required')
        if request['fixture']:
            require('fixture' in leaf['geometry_source'], 'synthetic provenance cannot be measured')
    require(set(p['plant_instance_id'] for p in leaves) == set(plant_map), 'plant without represented leaf inventory')
    env = tables['environment.csv']
    require(len(env) == n, 'environment coverage incomplete')
    for i, row in enumerate(env):
        require(row['interval'] == str(i), 'environment ordering')
        require(aware(row['start_time']) == start + timedelta(days=i) and aware(row['end_time']) == start + timedelta(days=i+1), 'environment timestamp mismatch')
        require(0 <= number(row['light_hours'], 'light hours') <= 24, 'photoperiod out of range')
        for key in ('temp_day_c', 'temp_night_c', 'co2_ppm'):
            number(row[key], key)
        require(bool(row['source_kind']), 'environment evidence required')
    require(len(tables['light_input.csv']) == 1, 'one reference light record required')
    light = tables['light_input.csv'][0]
    number(light['reference_ppfd_umol_m2_s'], 'reference PPFD', 0)
    require(light['source_kind'] in ('user_assumed', 'synthetic_assumed', 'measured'), 'light source_kind')
    if light['source_kind'] == 'measured':
        aware(light['measured_at'])
        require(not request['fixture'], 'fixture cannot establish measured PPFD')
    ids = set()
    for lamp in tables['lamps.csv']:
        identifier(lamp['lamp_id'])
        require(lamp['lamp_id'] not in ids, 'duplicate lamp'); ids.add(lamp['lamp_id'])
        for k in ('x_mm', 'y_mm', 'z_mm'):
            number(lamp[k], k)
        number(lamp['power_umol_s'], 'lamp power', 0)
        require(bool(lamp['source_kind']), 'lamp provenance required')
    pointer = (folder / 'light_ratio_latest.txt').read_text(encoding='utf-8-sig').strip()
    require(pointer and '\n' not in pointer and '\r' not in pointer and not Path(pointer).is_absolute(), 'one relative ratio pointer required')
    target = (folder / pointer).resolve()
    ratios = ratio_table(target)
    require(request['light_ratio_role'] == 'comparison_only', 'ratios must not double-scale photons')
    grid = request['comparison_grid']
    for k in ('spacing_mm', 'height_mm', 'origin_x_mm', 'origin_y_mm'):
        number(grid[k], k)
    require(grid['spacing_mm'] > 0 and grid['source_kind'] in ('synthetic_assumed', 'user_assumed'), 'explicit hypothetical grid required')
    require(grid['normalization_population'] == '16_grid_cells' and grid['physical_validation'] is False, 'unverified measurement grid cannot be physical validation')
    hashes = {name: sha(folder / name) for name in [*HEADERS, 'request.json', 'light_ratio_latest.txt']}
    hashes['resolved_ratio_target'] = sha(target)
    return dict(request=request, tables=tables, ratios=ratios, ratio_target=str(target), source_hashes=hashes)


def validate_outputs(package: dict, native: Path, binding: dict) -> dict:
    """Recompute mass/carbon/geometry checks; never repair native state."""
    req = package['request']; n = req['intervals']
    completion = read_json(native / 'completion.json')
    require(completion['run_id'] == req['run_id'] and completion['intervals'] == n, 'stale run/count')
    require(completion['input_hash'] == binding['input_hash'] and completion['engine_hash'] == binding['engine_hash'], 'completion binding mismatch')
    expected_files = {'geometry.csv', 'organs.csv', 'ledger.csv', 'sensors.csv', 'consumed_geometry.csv'}
    require(set(completion['files']) == expected_files, 'native file manifest mismatch')
    for name, value in completion['files'].items():
        require(sha(native / name) == value, f'native file hash mismatch: {name}')
    geometry, organs, ledger = [rows(native / f'{name}.csv') for name in ('geometry', 'organs', 'ledger')]
    leaves = {x['leaflet_id']: x for x in package['tables']['leaflets.csv']}
    plants = {x['plant_instance_id']: x for x in package['tables']['plants.csv']}
    require(len(geometry) == (n+1)*len(leaves) and len(organs) == n*len(leaves) and len(ledger) == n*len(plants), 'native inventory count')
    gmap = {(int(r['state']), r['leaflet_id']): r for r in geometry}
    omap = {(int(r['interval']), r['leaflet_id']): r for r in organs}
    lmap = {(int(r['interval']), r['plant_instance_id']): r for r in ledger}
    require(set(gmap) == {(i,l) for i in range(n+1) for l in leaves}, 'geometry identity coverage')
    require(set(omap) == {(i,l) for i in range(n) for l in leaves}, 'organ identity coverage')
    require(set(lmap) == {(i,p) for i in range(n) for p in plants}, 'ledger identity coverage')
    consumed=rows(native/'consumed_geometry.csv')
    cmap={(int(r['interval']),r['leaflet_id']):r for r in consumed}
    require(len(consumed)==len(omap) and set(cmap)==set(omap), 'consumed geometry coverage')
    par = {k:v['value'] for k,v in req['parameters'].items()}; tol=req['runtime']['balance_tolerance_gC']
    def close(a: float, b: float, label: str, atol: float = tol) -> None:
        require(math.isclose(a,b,rel_tol=1e-8,abs_tol=atol), f'{label}: {a} != {b}')
    for (i, lid), row in gmap.items():
        leaf=leaves[lid]; plant=plants[leaf['plant_instance_id']]
        require(row['plant_instance_id']==leaf['plant_instance_id'] and row['compound_id']==leaf['compound_id'], 'geometry parent identity mismatch')
        for k,v in row.items():
            if k not in ('leaflet_id','plant_instance_id','compound_id'):
                number(v,k)
        length,width,mass=map(float,(row['length_m'],row['width_m'],row['dry_mass_g']))
        require(length>0 and width>0 and mass>0,'nonpositive geometry/mass')
        close(float(row['area_m2']),length*width,'optical area')
        expected_mass=float(leaf['dry_mass_g']) if i==0 else float(gmap[i-1,lid]['dry_mass_g'])+float(omap[i-1,lid]['delta_dm_g'])
        close(mass,expected_mass,'mass continuity')
        scale=1 if req['freeze_geometry'] else math.sqrt(mass/float(leaf['dry_mass_g']))
        close(length,float(leaf['length_mm'])*.001*scale,'length mapping')
        close(width,float(leaf['width_mm'])*.001*scale,'width mapping')
        yaw=math.radians(float(plant['yaw_deg'])); x,y=float(leaf['x_mm']),float(leaf['y_mm'])
        for k,v in dict(x_m=(x*math.cos(yaw)-y*math.sin(yaw)+float(plant['tx_mm']))*.001,
                        y_m=(x*math.sin(yaw)+y*math.cos(yaw)+float(plant['ty_mm']))*.001,
                        z_m=(float(leaf['z_mm'])+float(plant['tz_mm']))*.001).items():
            close(float(row[k]),v,'transform '+k,1e-7)
        close(float(row['age_days']),float(leaf['age_days'])+i,'age continuity')
        if i<n:
            for key in ('length_m','width_m','area_m2'):
                actual=number(cmap[i,lid][key],key,0)
                require(math.isclose(actual,float(row[key]),rel_tol=2e-7,abs_tol=1e-10), 'light consumed stale geometry: '+key)
    normalized=[]
    for (i,pid), row in lmap.items():
        vals={k:number(v,k) for k,v in row.items() if k not in ('plant_instance_id','interval')}
        prev=float(plants[pid]['reserve_gC']) if i==0 else float(lmap[i-1,pid]['reserve_end_gC'])
        close(vals['reserve_start_gC'],prev,'reserve continuity')
        subset=[omap[i,l] for l,v in leaves.items() if v['plant_instance_id']==pid]
        for org in subset:
            lid=org['leaflet_id']; require(org['plant_instance_id']==pid,'organ/plant mismatch')
            v={k:number(x,k) for k,x in org.items() if k not in ('leaflet_id','plant_instance_id','interval')}
            area=float(gmap[i,lid]['area_m2']); ppfd=v['absorbed_umol_s']/area
            require(v['absorbed_umol_s']>=0,'negative photons')
            a,b,t=par['alpha_mol_mol'],par['amax_umol_m2_s'],par['theta']
            gross_rate=(a*ppfd+b-math.sqrt((a*ppfd+b)**2-4*t*a*ppfd*b))/(2*t)
            hours=float(package['tables']['environment.csv'][i]['light_hours'])
            close(v['gross_umol_CO2'],gross_rate*area*hours*3600,'VTC NRH exchange',1e-6)
            close(v['leaf_resp_gC'],par['respiration_umol_m2_s']*area*86400*par['molar_carbon_g_mol']*1e-6,'leaf respiration')
            close(v['net_gC'],v['gross_umol_CO2']*par['molar_carbon_g_mol']*1e-6-v['leaf_resp_gC'],'net exchange')
            close(v['demand_gC'],float(leaves[lid]['potential_dm_g_day'])*par['carbon_fraction_gC_gDM']/par['growth_efficiency'],'potential demand')
            require(-tol<=v['allocated_gC']<=v['demand_gC']+tol,'allocation bounds')
            close(v['growth_resp_gC'],v['allocated_gC']*(1-par['growth_efficiency']),'growth respiration')
            close(v['delta_dm_g']*par['carbon_fraction_gC_gDM'],v['allocated_gC']*par['growth_efficiency'],'structural increment')
        sums={k:sum(float(o[k]) for o in subset) for k in ('net_gC','allocated_gC','growth_resp_gC','leaf_resp_gC')}
        for k,v in sums.items(): close(vals[k],v,'per-plant '+k)
        total_demand=sum(float(o['demand_gC']) for o in subset)
        funded=min(max(0,prev+sums['net_gC']),total_demand)
        close(sums['allocated_gC'],funded,'funded allocation total')
        for org in subset:
            close(float(org['allocated_gC']),funded*float(org['demand_gC'])/total_demand if total_demand else 0,'relative sink share')
        structural=sum(float(o['delta_dm_g']) for o in subset)*par['carbon_fraction_gC_gDM']
        close(vals['structural_delta_gC'],structural,'structural total')
        residual=prev+sums['net_gC']-vals['reserve_end_gC']-structural-sums['growth_resp_gC']
        close(vals['residual_gC'],residual,'native residual')
        require(vals['unmet_deficit_gC']<=tol and abs(residual)<=tol,'unfunded respiration/deficit; conservation failed')
        require(vals['reserve_end_gC']>=-tol,'negative final reserve')
        normalized.append(dict(interval=i,plant_instance_id=pid,start_time=package['tables']['environment.csv'][i]['start_time'],
                               end_time=package['tables']['environment.csv'][i]['end_time'],reserve_start_gC=prev,
                               signed_exchange_gC=sums['net_gC'],structural_delta_gC=structural,
                               additional_respiration_gC=sums['growth_resp_gC'],reserve_end_gC=vals['reserve_end_gC'],residual_gC=residual))
    sensors=rows(native/'sensors.csv')
    require(len(sensors)==16 and {(r['grid_row'],r['grid_col']) for r in sensors}=={(str(i),str(j)) for i in range(1,5) for j in range(1,5)}, 'sensor coverage')
    for sensor in sensors:
        number(sensor['absorbed_umol_s'],'sensor flux',0)
        close(number(sensor['area_m2'],'sensor area',0),.0004,'sensor area')
    return dict(software_status='passed',run_status='assumption_conditioned',biological_validation='not_established',
                evaluation_status='evaluated_unaccepted',nutrient_feedback_enabled=False,
                normalized_ledger=normalized,geometry_hashes={str(i):digest([gmap[i,l] for l in sorted(leaves)]) for i in range(n+1)},
                full_v2_acceptance='incomplete_bounded_leaflet_subset')
