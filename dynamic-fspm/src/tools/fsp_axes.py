"""Declared-axis contracts and independent diagnostics; never a Python growth engine."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from tools.fsp_engine_hooks import planned, enabled as hook_enabled, validate as validate_hooks, validate_planned, axis_value, germination_threshold, axillary_stage
from tools.vtc_growth_io import require, number, rows, read_json, write_csv, write_json

SCHEMA = 'fsp-axes-1'
PARAMETERS = {'SUBSTEM_SINK_SCALE', 'APPLY_DETERMINATE_STOP'}
AXIS_COLUMNS = ('plant_instance_id', 'capture_id', 'axis_id', 'axis_vertex_id', 'order',
                'parent_axis_id', 'parent_node_id', 'parent_rank', 'stage', 'source_kind', 'source_ref')
FLUX_FIELDS = ('absorbed_umol_s', 'gross_gC', 'maintenance_gC', 'demand_gC', 'allocated_gC',
               'structural_gC', 'growth_resp_gC')


def enabled(request: dict) -> bool:
    return request.get('axis_contract') == SCHEMA


def validate_records(request: dict, plants: set[str]) -> list[dict]:
    validate_hooks(request)
    if not request.get('axis_contract'):
        require(not request.get('axes'), 'axis list requires versioned contract')
        return []
    require(enabled(request), 'unsupported axis contract')
    require(request.get('topology_scope') == 'provided_axes' and request.get('topology_completeness') == 'unavailable', 'axis scope/completeness must be explicit')
    records = request.get('axes', [])
    require(isinstance(records, list) and records, 'declared axes required')
    keys = [(a['plant_instance_id'], a['axis_id']) for a in records]
    require(len(set(keys)) == len(keys), 'duplicate axis IDs')
    require({p for p, _ in keys} == plants, 'axis plant coverage')
    for pid in plants:
        require((pid, 'M') in keys, 'main axis M required')
    for a in records:
        if planned(a):
            validate_planned(a, request, records)
        main = a['axis_id'] == 'M'
        require(planned(a) or main or re.fullmatch(r'S[1-9][0-9]*', a['axis_id']), 'invalid stable axis ID')
        require(planned(a) or (type(a['order']) is int and a['order'] == (0 if main else 1)), 'only declared order-1 side axes')
        require(a.get('source_kind') in ('observed', 'user_stated', 'user_assumed', 'synthetic_assumed', 'reconstructed') and a.get('source_ref'), 'axis evidence required')
        require(type(a['parent_rank']) is int, 'integer attachment rank required')
        stage=a.get('axis_state',a.get('stage'))
        require('stage' not in a or 'axis_state' not in a or a['stage']==a['axis_state'],'axis state aliases disagree')
        require(stage in (('main',) if main else ('axis_shoot', 'substem')), 'axis observation stage invalid')
        require(planned(a) or bool(a.get('axis_vertex_id')), 'axis vertex reference missing')
        if main:
            require(not a['parent_axis_id'] and not a['parent_node_id'] and a['parent_rank'] == 0, 'main attachment must be empty')
        elif not planned(a):
            require(a['parent_axis_id'] == 'M' and bool(a['parent_node_id']), 'side axis must attach to M')
            require(len(a.get('registration_leaf_ids', [])) == 2 and len(set(a['registration_leaf_ids'])) == 2, 'two registration leaves required')
        require(PARAMETERS <= set(a['parameters']) <= PARAMETERS | ({'FIRST_LEAVES', 'CADENCE'} if hook_enabled(request, 'per_axis_development') else set()), 'per-axis parameter coverage')
        for name, p in a['parameters'].items():
            require(p.get('source_kind') in ('user_assumed', 'synthetic_assumed', 'literature_other_cultivar', 'literature'), 'axis parameter is a conditional law')
            require(all(p.get(k) for k in ('unit', 'source_ref', 'cultivar', 'uncertainty')), 'axis parameter provenance incomplete')
            if name == 'SUBSTEM_SINK_SCALE':
                require(number(p['value'], name) > 0 and p['unit'] == 'dimensionless', 'positive sink scale required')
                if main: require(p['value'] == 1, 'main/root reference sink scale must be unity')
            elif name == 'FIRST_LEAVES':
                require(type(p['value']) is int and p['value'] > 0 and p['unit'] == 'count', 'per-axis FIRST_LEAVES must be positive engine rank count')
            elif name == 'CADENCE':
                v=p['value'];require(isinstance(v,list) and v and all(type(x) is int and 0<=x<=20 for x in v) and v[-1]==0 and p['unit']=='count', 'per-axis CADENCE must end in zero')
            else:
                require(type(p['value']) is bool and p['unit'] == 'boolean', 'explicit stop-program flag required')
        h = a['initialization']
        require(h.get('source_kind') in ('user_assumed', 'synthetic_assumed', 'supplement_provided') and h.get('source_ref'), 'axis hidden-state evidence required')
        for k in ('thermal_dd', 'stem_mass_gDM', 'stem_width_m'): number(h[k], k, 0)
        for k in ('produced', 'leaf_order', 'truss_count', 'axis_leaf_count'):
            require(type(h[k]) is int and h[k] >= 0, 'axis clock/count must be explicit integer')
        require(type(h['apex_stopped']) is bool, 'explicit apex state required')
        require(h['axis_leaf_count'] <= h['leaf_order'], 'axis leaf counters disagree')
        require(h['stem_mass_gDM'] > 0 or h['stem_width_m'] == 0, 'massless hidden stem cannot cast shade')
    occupied=set();forecasts=set()
    for a in records:
        if a['axis_id']=='M':continue
        key=(a['plant_instance_id'],a['parent_axis_id'],a['parent_rank'])
        require(key not in occupied,'occupied axis attachment');occupied.add(key)
        if planned(a):
            require(a['forecast_axis_id'] not in forecasts,'duplicate forecast namespace ID');forecasts.add(a['forecast_axis_id'])
    # Main controllers retain plant-index alignment used by legacy initialization.
    return [a for a in records if a['axis_id'] == 'M'] + [a for a in records if a['axis_id'] != 'M']


def select_axes(folder: Path, scenario: dict, vertices: list[dict]) -> tuple[list[dict], dict]:
    records = validate_records(scenario, set(scenario['captures_by_instance']))
    if not records:
        require(all(sum(v['class'] == 'Axis' for v in vertices if v['plant_instance_id'] == p) == 1 for p in scenario['captures_by_instance']), 'multiple source axes require fsp-axes-1; no single-axis fallback')
        return [], {}
    plans=[a for a in records if planned(a)]
    records=[a for a in records if not planned(a)]
    table = rows(folder/'axes.csv')
    selected = []
    for r in table:
        if scenario['captures_by_instance'].get(r['plant_instance_id']) != r['capture_id']:continue
        require('axis' not in r or 'axis_id' not in r or r['axis']==r['axis_id'],'producer axis aliases disagree')
        require('axis_state' not in r or 'stage' not in r or r['axis_state']==r['stage'],'producer axis-state aliases disagree')
        normalized=dict(r,axis_id=r.get('axis_id',r.get('axis')),stage=r.get('stage',r.get('axis_state')))
        if normalized['axis_id']=='M':
            for field in ('parent_axis_id','parent_node_id'):
                if normalized[field]=='unavailable':normalized[field]=''
        selected.append(normalized)
    amap = {(a['plant_instance_id'], a['axis_id']): a for a in records}
    require(len(selected) == len(amap) and len({(r['plant_instance_id'], r['axis_id']) for r in selected}) == len(selected), 'producer axis row coverage/duplicates')
    for r in selected:
        require(set(AXIS_COLUMNS) <= set(r), 'producer axes.csv columns incomplete')
        a = amap.get((r['plant_instance_id'], r['axis_id']))
        require(a is not None, 'undeclared producer axis')
        for k in ('axis_vertex_id', 'order', 'parent_axis_id', 'parent_node_id', 'parent_rank', 'stage'):
            value=a.get('axis_state',a.get('stage')) if k=='stage' else a[k]
            require(str(value) == r[k], 'producer/request axis disagreement: '+k)
        require(r['source_kind'] and r['source_ref'], 'producer axis evidence missing')
    by = {(v['plant_instance_id'], v['vertex_id']): v for v in vertices}
    vertex_axes = {(a['plant_instance_id'], a['axis_vertex_id']): a for a in records}
    require(len(vertex_axes) == len(records), 'duplicate axis vertices')
    require(set(vertex_axes) == {k for k, v in by.items() if v['class'] == 'Axis'}, 'MTG axis inventory mismatch')
    membership = {}
    for key, v in by.items():
        p, oid = key; seen = set()
        while v['class'] not in ('Axis', 'Plant'):
            require(v['vertex_id'] not in seen, 'MTG axis cycle'); seen.add(v['vertex_id'])
            ref = v['complex_id'] if v['complex_id'] not in ('', 'unavailable') else v['parent_id']
            require((p, ref) in by, 'MTG axis ancestry missing'); v = by[p, ref]
        membership[key] = vertex_axes[p, v['vertex_id']]['axis_id'] if v['class'] == 'Axis' else 'M'
        original=by[key]
        if original.get('axis') not in (None,'','unavailable'):require(original['axis']==membership[key],'MTG axis annotation disagrees with ancestry')
    for a in records:
        p = a['plant_instance_id']; av = by[p, a['axis_vertex_id']]
        if a['axis_id'] == 'M': continue
        parent = by.get((p, a['parent_node_id']))
        require(parent is not None and parent['class'] in ('Metamer', 'Attachment', 'CompoundLeaf'), 'side axis parent node missing')
        require(membership[p, parent['vertex_id']] == 'M', 'side parent must lie on M')
        require(parent['rank'] not in ('', 'unavailable') and int(parent['rank']) == a['parent_rank'], 'side attachment rank mismatch')
        main_vertex=amap[p,'M']['axis_vertex_id']
        require(av['parent_id'] in (parent['vertex_id'],main_vertex), 'MTG side-axis edge disagrees with declared attachment')
        # A4 names the leaf at the axil; older fixtures use an explicit Metamer.
        require(parent['class'] == 'CompoundLeaf' or any(v['plant_instance_id'] == p and v['class'] == 'CompoundLeaf' and v['complex_id'] == parent['vertex_id'] for v in vertices), 'attachment is not a declared leaf axil')
        for oid in a['registration_leaf_ids']:
            v = by.get((p, oid))
            require(v is not None and v['class'] == 'CompoundLeaf' and membership[p, oid] == a['axis_id'], 'registration leaf missing or on wrong axis')
    from tools.fsp_axes_segments import prepare_segments,validate_truss_rows
    if (folder/'trusses.csv').is_file():
        events=[r for r in rows(folder/'trusses.csv') if scenario['captures_by_instance'].get(r['plant_instance_id'])==r['capture_id']]
        validate_truss_rows(events,vertices,membership)
    for a in plans:
        parent=by.get((a['plant_instance_id'],a['parent_node_id']))
        require(parent is not None and parent['class'] in ('Attachment','Metamer','CompoundLeaf'), 'planned parent must be an existing leaf axil')
        require(membership[a['plant_instance_id'],a['parent_node_id']]==a['parent_axis_id'] and int(parent['rank'])==a['parent_rank'], 'planned parent axis/rank mismatch')
        require(parent['class']=='CompoundLeaf' or any(v['plant_instance_id']==a['plant_instance_id'] and v['class']=='CompoundLeaf' and v['complex_id']==a['parent_node_id'] for v in vertices), 'planned attachment lacks compound leaf')
        require(not any(v['plant_instance_id']==a['plant_instance_id'] and v['class']=='Axis' and v['parent_id']==a['parent_node_id'] for v in vertices), 'occupied planned axil')
    return prepare_segments(records, selected, vertices, membership)+plans, membership


def allocation_reference(demands: list[float], scales: list[float], budget: float) -> list[float]:
    """Independent bisection oracle used only to validate returned native allocations."""
    total = min(max(0., budget), sum(demands))
    if total == 0: return [0.] * len(demands)
    lo, hi = 0., max(1/s for s in scales)
    for _ in range(100):
        mid = (lo+hi)/2
        if sum(min(d, mid*d*s) for d, s in zip(demands, scales)) < total: lo = mid
        else: hi = mid
    return [min(d, ((lo+hi)/2)*d*s) for d, s in zip(demands, scales)]


def native_setup(request: dict, plants: list[dict]) -> tuple[str, int, list[dict]]:
    records = validate_records(request, {p['plant_instance_id'] for p in plants})
    if records:
        records = [next(a for a in records if a['plant_instance_id'] == p['plant_instance_id'] and a['axis_id'] == 'M') for p in plants] + [a for a in records if a['axis_id'] != 'M']
    owners = {p['plant_instance_id']: i for i, p in enumerate(plants)}
    def array(values: list, typ: str) -> str:
        return 'new '+typ+'[]{'+','.join(json.dumps(v) for v in values)+'}'
    declarations = [f'boolean axesEnabled={str(bool(records)).lower()};',
        'int[] currentSegments='+array([a['segment_index'] if a.get('segment_index','unavailable')!='unavailable' else -1 for a in records] if records else [1]*len(plants), 'int')+';',
        'boolean[] trussHistoryKnown='+array([a.get('truss_rule_status')=='available' for a in records] if records else [True]*len(plants), 'boolean')+';',
        'int[] controllerOwners='+array([owners[a['plant_instance_id']] for a in records] if records else list(range(len(plants))), 'int')+';',
        'String[] logicalAxisIds='+array([a['axis_id'] for a in records] if records else ['M']*len(plants), 'String')+';',
        'int[] parentRanks='+array([a['parent_rank'] for a in records] if records else [0]*len(plants), 'int')+';',
        'double[] sinkScales='+array([a['parameters']['SUBSTEM_SINK_SCALE']['value'] for a in records] if records else [1.]*len(plants), 'double')+';',
        'boolean[] axisPrograms='+array([a['parameters']['APPLY_DETERMINATE_STOP']['value'] for a in records] if records else [True]*len(plants), 'boolean')+';']
    return '\n'.join(declarations), len(records) if records else len(plants), records


def validate_axis_outputs(pkg: dict, data: dict, n: int) -> None:
    req = pkg['request']
    if not enabled(req): return
    from tools.fsp_axes_segments import validate_segments
    validate_segments(pkg,data,n)
    records = validate_records(req, {p['plant_instance_id'] for p in pkg['tables']['plants.csv']})
    amap = {(a['plant_instance_id'], a['axis_id']): a for a in records}
    state = {(int(s['day']), s['organ_id']): s for s in data['states.csv']}
    for s in state.values():
        a = amap.get((s['plant_instance_id'], s['axis_id']))
        require(a is not None and int(s['parent_rank']) == a['parent_rank'], 'native logical axis/attachment changed')
        if s['class'] != 'Plant':
            segment = state.get((int(s['day']), s['developmental_axis_id']))
            require(segment is not None and segment['class'] == 'Axis' and segment['axis_id'] == s['axis_id'], 'native segment/logical axis mismatch')
    for table, time in (('events.csv', 'day'), ('meshes.csv', 'state'), ('consumed.csv', 'state'), ('flux.csv', 'interval')):
        for row in data[table]:
            day = int(row[time]) + (1 if table == 'flux.csv' else 0)
            s = state[day, row['organ_id']]
            require(row['axis_id'] == s['axis_id'] and row['parent_rank'] == s['parent_rank'], 'axis attribution lost: '+table)
    groups = defaultdict(list)
    for row in data['flux.csv']: groups[int(row['interval']), row['plant_instance_id'], row['axis_id']].append(row)
    actual = {(int(r['interval']), r['plant_instance_id'], r['axis_id']): r for r in data['axis_flux.csv']}
    expected = {(i, p, a) for i in range(n) for p, a in amap}
    require(len(actual) == len(data['axis_flux.csv']) and set(actual) == expected, 'axis ledger coverage')
    for key, row in actual.items():
        for field in FLUX_FIELDS:
            total = sum(float(r[field]) for r in groups[key])
            require(math.isclose(float(row[field]), total, abs_tol=1e-12, rel_tol=1e-12), 'axis ledger sum mismatch: '+field)
    for a in records:
        if a['axis_id'] == 'M': continue
        if planned(a):
            oid=a['forecast_axis_id']
            births=[e for e in data['events.csv'] if e['event']=='axillary_emergence' and e['organ_id']==oid]
            from tools.fsp_extensions import boundary_time
            receiver=req.get('receiver_index',-1)
            is_receiver=receiver>=0 and pkg['tables']['plants.csv'][receiver]['plant_instance_id']==a['plant_instance_id']
            eligible=[i for i in range(n+1) if (boundary_time(req,i)-boundary_time(req,0)).total_seconds()/86400>=a['emergence_day'] and (req['freeze_at']<0 or i<=req['freeze_at']) and (not is_receiver or i<=req['receiver_day'])]
            require(len(births)==(1 if eligible else 0),'planned emergence event coverage')
            birth=int(births[0]['day']) if births else n+1
            if eligible:
                require(birth==eligible[0],'planned emergence timing')
                ids={s['organ_id'] for (day,_),s in state.items() if day==birth and s['plant_instance_id']==a['plant_instance_id'] and s['axis_id']==a['axis_id'] and s['class']=='Apex'}
                clock_rows=[r for r in data['apices.csv'] if int(r['day'])==birth and r['organ_id'] in ids]
                require(len(clock_rows)==1 and float(clock_rows[0]['plant_thermal_dd'])==0 and int(clock_rows[0]['produced_events'])==0,'planned emergence clock must start at zero')
            for i in range(n+1):
                owned=[s for (d,_),s in state.items() if d==i and s['plant_instance_id']==a['plant_instance_id'] and s['axis_id']==a['axis_id']]
                if i<birth:require(not owned,'dormant axis has organs')
                else:require(state[i,oid]['parent_id']==a['plant_instance_id']+'|'+a['parent_node_id'],'planned attachment changed')
            continue
        oid = a['plant_instance_id']+'|'+a['axis_vertex_id']
        for i in range(n+1):
            require(state[i, oid]['parent_id'] == a['plant_instance_id']+'|'+a['parent_node_id'], 'native side attachment changed')
    if req.get('light_only'): return
    from tools.fsp_extensions import interval_seconds
    from tools import fsp_stage_parameters as stage
    par={k:v['value'] for k,v in req['parameters'].items()}
    from tools import fsp_random_parameters as rp
    rpd=pkg.get('_random_draws') or rp.Draws(req,None)
    for a in records:
        pid,aid=a['plant_instance_id'],a['axis_id'];h=a['initialization']
        clock=h['thermal_dd'];produced=h['produced'];leaves=h['leaf_order'];trusses=h['truss_count'];segment_leaves=h['axis_leaf_count'];stopped=h['apex_stopped']
        initial_leaves=sum(s['class']=='CompoundLeaf' for (day,_),s in state.items() if day==0 and s['plant_instance_id']==pid and s['axis_id']==aid)
        initial_trusses=sum(s['class']=='TrussAxis' for (day,_),s in state.items() if day==0 and s['plant_instance_id']==pid and s['axis_id']==aid)
        birth=next((int(e['day']) for e in data['events.csv'] if e['event']=='axillary_emergence' and e['organ_id']==a.get('forecast_axis_id')),n+1) if planned(a) else -1
        for i in range(n):
            if planned(a) and i<birth:continue
            dt=interval_seconds(req,i)/86400;e=pkg['tables']['environment.csv'][i];hours=float(e['light_hours'])
            clock+=stage.thermal(req,e,dt,'development')
            due=math.floor((clock-germination_threshold(req,a))/(par['PHYLLOCHRON_DD']*rpd.plant(pid,'PHYLLOCHRON_FACTOR')))
            receiver=req.get('receiver_index',-1)
            receiver_frozen=receiver>=0 and pkg['tables']['plants.csv'][receiver]['plant_instance_id']==pid and i>=req['receiver_day']
            if a.get('truss_rule_status')=='available' and a['parameters']['APPLY_DETERMINATE_STOP']['value'] and (req['freeze_at']<0 or i<req['freeze_at']) and not receiver_frozen:
                while produced<due and not stopped:
                    produced+=1;target=axis_value(req,a,'FIRST_LEAVES') if trusses==0 else axis_value(req,a,'CADENCE')[min(trusses-1,len(axis_value(req,a,'CADENCE'))-1)]
                    if segment_leaves<target:
                        if stage.enabled(req,'side_axis_phytomer_cap') and planned(a) and leaves>=stage.values(req)['SIDE_AXIS_PHYTOMER_CAP']:
                            stopped=True;break
                        leaves+=1;segment_leaves+=1
                    else:trusses+=1;stopped=trusses>1 and target==0;segment_leaves=0
            ss=[s for (day,_),s in state.items() if day==i+1 and s['plant_instance_id']==pid and s['axis_id']==aid]
            require(sum(s['class']=='CompoundLeaf' for s in ss)==initial_leaves+leaves-h['leaf_order'],'independent per-axis leaf schedule failed')
            require(sum(s['class']=='TrussAxis' for s in ss)==initial_trusses+trusses-h['truss_count'],'independent per-axis truss schedule failed')


def export_registry(folder: Path, request: dict) -> None:
    if not enabled(request): return
    contract = {k: request[k] for k in ('axis_contract', 'topology_scope', 'topology_completeness', 'axes')}
    if hook_enabled(request, 'planned_axes'):
        contract['plan_provenance'] = dict(emergence_semantics='bud_break', axillary_start_stage=axillary_stage(request),
            source_kind='user_assumed', source_ref='Reviewed engine hooks addendum decision 2026-10-15',
            observed_registration='two visible leaves is a records-comparison definition only')
    write_json(folder/'axis_contract.json', contract)
    registry = []
    for a in request['axes']:
        for name, p in a['parameters'].items(): registry.append(dict(plant_instance_id=a['plant_instance_id'], axis_id=a['axis_id'], parameter=name, **p))
    write_csv(folder/'axis_parameter_registry.csv', registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(); req = read_json(args.request)
    result = validate_records(req, {a['plant_instance_id'] for a in req.get('axes', [])})
    print(json.dumps(dict(contract=SCHEMA, declared_axes=len(result), biological_validation='not_established')))


if __name__ == '__main__': main()
