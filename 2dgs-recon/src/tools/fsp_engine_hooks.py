"""Opt-in native development contracts and independent age diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tools.vtc_growth_io import require, number, read_json
from tools.fsp_development_overrides import OVERRIDES, validate_overrides, effective_parameters
from tools import fsp_reserve_cap
from tools import fsp_stage_parameters as s1
from tools import fsp_temperature

SCHEMA = 'fsp-engine-hooks-1'
SWITCHES = {'planned_axes', 'per_axis_development', 'leaf_development', 'projected_shading', 'reserve_cap', 'reserve_cap_floor', 'leaflet_outline', 'incident_diagnostics'}
SWITCHES |= s1.SWITCHES
SWITCHES |= fsp_temperature.SWITCHES
SWITCHES.add('plant_positions')
from tools import fsp_random_parameters as rp
SWITCHES.add(rp.SWITCH)
from tools import fsp_flowers_per_truss as fpt
SWITCHES.add(fpt.SWITCH)
from tools import fsp_substrate_water as water
SWITCHES.add(water.SWITCH)
from tools import fsp_nitrogen_uptake as nitrogen
SWITCHES.add(nitrogen.SWITCH)
LEAF_PARAMETERS = {'LEAFLET_GROWTH_ORDER', 'LEAFLET_STAGGER_DD', 'LEAF_AGEING_DECAY_A'}
AXILLARY_STAGE = 'AXILLARY_START_STAGE'
SOURCES = ('user_assumed', 'synthetic_assumed', 'literature_other_cultivar', 'literature', 'literature_generic', 'literature_wild_species', 'user_stated')


def enabled(request: dict, name: str) -> bool:
    return request.get('engine_hooks', {}).get(name, False) is True


def planned(axis: dict) -> bool:
    return axis.get('origin') == 'axillary' and axis.get('provenance') == 'model_generated'


def axillary_stage(request: dict) -> str:
    return request.get('engine_hooks', {}).get('parameters', {}).get(AXILLARY_STAGE, {}).get('value', 'bud_break')


def germination_threshold(request: dict, axis: dict) -> float:
    if enabled(request, 'planned_axes') and planned(axis) and axillary_stage(request) == 'bud_break':
        return 0.
    return request['parameters']['GERMINATION_DD']['value']


def provenance(parameter: dict) -> None:
    require(parameter.get('source_kind') in SOURCES, 'hook parameter source_kind invalid')
    require(all(parameter.get(k) for k in ('unit', 'source_ref', 'cultivar', 'uncertainty')), 'hook parameter provenance incomplete')


def tau95(mid_dd: float, k_per_dd: float) -> float:
    initial = 1 / (1 + math.exp(k_per_dd * mid_dd))
    sigmoid = initial + .95 * (1 - initial)
    return mid_dd + math.log(sigmoid / (1 - sigmoid)) / k_per_dd


def validate(request: dict) -> None:
    hooks = request.get('engine_hooks')
    if hooks is None:
        return
    require(isinstance(hooks, dict) and hooks.get('schema') == SCHEMA, 'unsupported engine_hooks schema')
    require(set(hooks) <= SWITCHES | {'schema', 'parameters'}, 'unknown engine hook')
    for name in SWITCHES:
        require(type(hooks.get(name, False)) is bool, 'engine hook switch must be boolean')
    params = hooks.get('parameters', {})
    require(isinstance(params, dict), 'hook parameter registry required')
    require(set(params) <= LEAF_PARAMETERS | {AXILLARY_STAGE} | set(OVERRIDES) | fsp_reserve_cap.PARAMETERS | s1.PARAMETERS | fsp_temperature.PARAMETERS | rp.PARAMETERS | fpt.PARAMETERS | water.PARAMETERS | nitrogen.PARAMETERS, 'unknown hook parameter')
    s1.validate(request)
    fsp_temperature.validate(request)
    from tools.fsp_incident_diagnostics import validate as validate_incident
    validate_incident(request)
    from tools.fsp_plant_positions import validate as validate_positions
    validate_positions(request)
    validate_overrides(request)
    fsp_reserve_cap.validate(request)
    rp.validate(request)
    fpt.validate(request)
    water.validate(request)
    nitrogen.validate(request)
    if AXILLARY_STAGE in params:
        require(enabled(request, 'planned_axes'), 'axillary stage requires planned_axes')
        stage = params[AXILLARY_STAGE]
        provenance(stage)
        require(stage['unit'] == 'categorical' and stage['source_kind'] == 'user_assumed', 'axillary stage provenance/unit')
        require(stage['value'] in ('bud_break', 'germination'), 'unknown axillary start stage')
    leaf_params = set(params) & LEAF_PARAMETERS
    if not enabled(request, 'leaf_development'):
        require(not leaf_params, 'leaf parameters require leaf_development switch')
    else:
        require(leaf_params == LEAF_PARAMETERS, 'leaf hook parameter coverage')
        for key in LEAF_PARAMETERS:
            provenance(params[key])
        order, stagger, decay = (params[k]['value'] for k in ('LEAFLET_GROWTH_ORDER', 'LEAFLET_STAGGER_DD', 'LEAF_AGEING_DECAY_A'))
        require(order in ('simultaneous', 'basipetal'), 'unknown leaflet growth order')
        require(params['LEAFLET_GROWTH_ORDER']['unit'] == 'categorical', 'growth order unit')
        require(params['LEAFLET_STAGGER_DD']['unit'] == 'degree_day', 'stagger unit')
        require(params['LEAF_AGEING_DECAY_A']['unit'] == '1/degree_day', 'age decay unit')
        number(stagger, 'leaflet stagger', 0); number(decay, 'leaf ageing decay', 0)
        par = effective_parameters(request)
        threshold = tau95(par['MID_DD']['value'][9], par['K']['value'][9])
        require(decay == 0 or 1 / decay > threshold,
                f'leaf ageing asymptote must exceed normalized leaflet tau95={threshold:.12g} degree_day')
    if enabled(request, 'planned_axes') or enabled(request, 'per_axis_development'):
        require(request.get('axis_contract') == 'fsp-axes-1', 'axis hooks require fsp-axes-1')
    if enabled(request, 'planned_axes'):
        require(not request.get('light_only'), 'planned growth incompatible with light_only')


def validate_planned(axis: dict, request: dict, records: list[dict]) -> None:
    require(enabled(request, 'planned_axes'), 'planned axis requires enabled H7')
    require(axis['axis_id'] != 'M' and re.fullmatch(r'[A-Za-z0-9_:/#.-]+', axis['axis_id']), 'invalid planned axis ID')
    forecast = axis.get('forecast_axis_id', '')
    require(isinstance(forecast, str) and forecast.startswith('F:') and re.fullmatch(r'[A-Za-z0-9_:/#.-]+', forecast), 'forecast namespace required')
    require(not axis.get('axis_vertex_id') and not axis.get('registration_leaf_ids'), 'planned axis cannot claim observed vertices')
    parent = next((a for a in records if a['plant_instance_id'] == axis['plant_instance_id'] and a['axis_id'] == axis.get('parent_axis_id')), None)
    require(parent is not None and not planned(parent), 'planned parent must be an existing declared axis')
    require(type(axis['order']) is int and axis['order'] == parent['order'] + 1, 'planned axis order mismatch')
    require(type(axis['parent_rank']) is int and axis['parent_rank'] > 0, 'positive planned parent rank required')
    require(bool(axis.get('parent_node_id')), 'planned attachment reference required')
    number(axis.get('emergence_day'), 'planned emergence elapsed day', 0)
    require(axis.get('source_kind') in ('user_assumed', 'synthetic_assumed') and axis.get('source_ref'), 'planned emergence evidence required')
    require(re.fullmatch(r'[0-9a-f]{64}', axis.get('plan_sha256', '')) is not None, 'planned source hash required')
    h = axis['initialization']
    require(all(h[k] == 0 for k in ('thermal_dd', 'stem_mass_gDM', 'stem_width_m', 'produced', 'leaf_order', 'truss_count', 'axis_leaf_count')) and h['apex_stopped'] is False, 'planned controller must start empty at zero clock')
    require(axis.get('segment_index') == 1 and axis.get('truss_rule_status') == 'available', 'new axis starts known segment 1')
    require(not axis.get('truss_history'), 'planned axis cannot have observed truss history')


def axis_value(request: dict, axis: dict, key: str):
    if enabled(request, 'per_axis_development') and key in axis['parameters']:
        return axis['parameters'][key]['value']
    return request['parameters'][key]['value']


def offset_dd(request: dict, kind: str, pair: int, pair_count: int, side: int = 0) -> float:
    if not enabled(request, 'leaf_development') or kind not in ('Leaflet', 'Petiole', 'RachisSegment'):
        return 0.
    p = request['engine_hooks']['parameters']
    if p['LEAFLET_GROWTH_ORDER']['value'] == 'simultaneous' or p['LEAFLET_STAGGER_DD']['value'] == 0:
        return 0.
    steps = pair_count if kind == 'Petiole' else (0 if kind == 'Leaflet' and side == 0 else pair_count + 1 - pair)
    require(0 <= steps <= pair_count, 'leaf organ pair index inconsistent with compound')
    return steps * p['LEAFLET_STAGGER_DD']['value']


def require_unique_leaf_order(states) -> None:
    """A7 follow-through: PAIRS[true_leaf_order - 1] assumes one leaf per node index on an axis. Two CompoundLeaf rows of one plant/axis
    with the same true_leaf_order (a U5 rank fed as a node index) are refused."""
    seen = set()
    for s in (states.values() if isinstance(states, dict) else states):
        if s.get('class') != 'CompoundLeaf':
            continue
        key = (s['plant_instance_id'], s['axis_id'], s['true_leaf_order'])
        require(key not in seen, f"duplicate true_leaf_order {s['true_leaf_order']} on axis {s['axis_id']} of {s['plant_instance_id']}: ranks are not node indices (U5); feed node_index_from_base")
        seen.add(key)


def physiological_age(thermal_dd: float, offset: float, decay: float) -> float:
    age = max(0., thermal_dd - offset)
    return age if decay == 0 else -math.expm1(-decay * age) / decay


def target_age(request: dict, state: dict, states: dict | None = None) -> float:
    age = float(state['thermal_age_dd'])
    if not enabled(request, 'leaf_development') or state['class'] not in ('Leaflet', 'Petiole', 'RachisSegment'):
        return age
    if s1.enabled(request,'process_base_temperatures'):
        return physiological_age(float(state['s1_leaf_growth_age_dd']),0.,request['engine_hooks']['parameters']['LEAF_AGEING_DECAY_A']['value'])
    if int(state['birth_day']) < 0:
        # Imported P1 records may have unavailable rachis pair indices and
        # intercalary blades. Their supplied thermal age remains the age origin;
        # do not infer a historical G1 offset from a name or an image.
        return physiological_age(age, 0., request['engine_hooks']['parameters']['LEAF_AGEING_DECAY_A']['value'])
    rank = int(state['true_leaf_order']); pairs = request['parameters']['PAIRS']['value']
    count = pairs[min(rank - 1, len(pairs) - 1)]
    if states is not None:
        require_unique_leaf_order(states)
        count=max((int(s['pair_index']) for s in states.values() if s['plant_instance_id']==state['plant_instance_id'] and s['axis_id']==state['axis_id'] and s['true_leaf_order']==state['true_leaf_order'] and s['class']=='Leaflet' and int(s['side'])!=0),default=count)
    offset = offset_dd(request, state['class'], int(state['pair_index']), count, int(state['side']))
    return physiological_age(age, offset, request['engine_hooks']['parameters']['LEAF_AGEING_DECAY_A']['value'])


def transform_native(template: str, request: dict, records: list[dict]) -> str:
    """Inject only enabled hooks; disabled requests return the exact input bytes."""
    validate(request)
    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old) == 1, 'native hook insertion point changed: ' + old[:70])
        template = template.replace(old, new)
    if enabled(request, 'per_axis_development'):
        first = ','.join(str(axis_value(request, a, 'FIRST_LEAVES')) for a in records)
        cadence = ','.join('new int[]{' + ','.join(map(str, axis_value(request, a, 'CADENCE'))) + '}' for a in records)
        replace('@AXIS_CODE@', '@AXIS_CODE@\nint[] axisFirstLeaves=new int[]{' + first + '};\nint[][] axisCadence=new int[][]{' + cadence + '};')
        replace('int target=trussCounts[controller]==0?FIRST_LEAVES:CADENCE[Math.min(trussCounts[controller]-1,CADENCE.length-1)];',
                'int[] cadence=axisCadence[controller];int target=trussCounts[controller]==0?axisFirstLeaves[controller]:cadence[Math.min(trussCounts[controller]-1,cadence.length-1)];')
    if enabled(request, 'leaf_development'):
        p = request['engine_hooks']['parameters']
        stagger = p['LEAFLET_STAGGER_DD']['value'] if p['LEAFLET_GROWTH_ORDER']['value'] == 'basipetal' else 0.
        decay = p['LEAF_AGEING_DECAY_A']['value']
        # Zero parameters use the original expression, including operation order.
        if stagger != 0 or decay != 0:
            replace('double targetMass(Organ o) {', f'''double leafTargetAge(Organ o) {{
    if(!(o.kind==7 || o.kind==8 || o.kind==9))return o.thermal;
    int m=0;for(int j=0;j<organs.size();j++){{Organ l=org(j);if(l.owner==o.owner && l.controller==o.controller && l.rank==o.rank && l.kind==9 && l.side!=0)m=Math.max(m,l.pair);}}
    int position=o.born<0?0:(o.kind==7?m:(o.kind==9 && o.side==0?0:m+1-o.pair));
    double age=Math.max(0,o.thermal-position*{float(stagger)});
    return {float(decay)}==0?age:-Math.expm1(-{float(decay)}*age)/{float(decay) if decay else 1.0};
}}
double targetMass(Organ o) {{''')
            replace('(o.thermal-MID_DD[o.kind])', '(leafTargetAge(o)-MID_DD[o.kind])')
        if stagger != 0:
            replace('void updateGeometry() {', '''double leafPairFraction(Organ o) {
    if(o.mass<=0)return 0;
    int pair=o.kind==7?1:o.pair;double mass=0,maximum=0;
    for(int j=0;j<organs.size();j++){Organ l=org(j);
        if(l.owner==o.owner && l.controller==o.controller && l.rank==o.rank && l.kind==9 && l.side!=0 && l.pair==pair){mass+=l.mass;maximum+=l.maxMass;}}
    return maximum>0?Math.cbrt(mass/maximum):0;
}
void updateGeometry() {''')
            replace('int m=pairs(o.rank);o.len=(o.kind==7?PETIOLE_M:RACHIS_M/m)*fraction;',
                    'int m=pairs(o.rank);o.len=(o.kind==7?PETIOLE_M:RACHIS_M/m)*leafPairFraction(o);')
    if enabled(request, 'planned_axes'):
        code = Path(__file__).with_name('fsp_engine_hooks_native.rgg').read_text(encoding='utf-8')
        def array(values: list, typ: str) -> str:
            return 'new ' + typ + '[]{' + ','.join(json.dumps(v) for v in values) + '}'
        code = code.replace('@PLANNED@', array([planned(a) for a in records], 'boolean'))
        code = code.replace('@EMERGENCE@', array([float(a['emergence_day']) if planned(a) else -1. for a in records], 'double'))
        code = code.replace('@PARENT_IDS@', array([a['plant_instance_id']+'|'+a['parent_node_id'] if planned(a) else '' for a in records], 'String'))
        code = code.replace('@FORECAST_IDS@', array([a['forecast_axis_id'] if planned(a) else '' for a in records], 'String'))
        replace('@AXIS_CODE@', '@AXIS_CODE@\n' + code)
        if axillary_stage(request) == 'bud_break':
            replace('if(clocks[controller]<GERMINATION_DD)return;',
                    'double germinationDD=plannedController[controller]?0:GERMINATION_DD;\n    if(clocks[controller]<germinationDD)return;')
            replace('(clocks[controller]-GERMINATION_DD)/PHYLLOCHRON_DD',
                    '(clocks[controller]-germinationDD)/PHYLLOCHRON_DD')
        replace('int p=controllerOwners[controller];\n    Organ apex=', 'int p=controllerOwners[controller];\n    if(plannedController[controller] && !controllerAwake[controller])return;\n    Organ apex=')
        replace('if(controllerOwners[c]==p)clocks[c]+=dd;', 'if(controllerOwners[c]==p && (!plannedController[c] || controllerAwake[c]))clocks[c]+=dd;')
        replace('for(int j=0;j<organs.size();j++)if(org(j).owner==p)org(j).thermal+=dd;',
                'for(int j=0;j<organs.size();j++)if(org(j).owner==p)org(j).thermal+=dd;\n            if(freezeAt<0 || step<freezeAt)emergeAxes(p,step+1);')
        replace('attachGraph();if(!observationMode)updateGeometry();', 'for(int p=0;p<plantIds.length;p++)emergeAxes(p,0);\n    attachGraph();if(!observationMode)updateGeometry();')
        replace('Organ next=add(p,lastStem[controller],-1,1,leafOrders[controller],day,prefix+"A"+count);',
                'refuseOccupiedSuccessor(controller,leafOrders[controller]);\n                Organ next=add(p,lastStem[controller],-1,1,leafOrders[controller],day,prefix+"A"+count);')
    from tools.fsp_leaflet_outline import transform_native as outline_native
    from tools.fsp_incident_diagnostics import transform_native as incident_native
    template = incident_native(fsp_temperature.transform_native(s1.transform_native(outline_native(fsp_reserve_cap.transform_native(template, request), request), request, records), request), request)
    template = fpt.transform_native(template, request)   # flowers_per_truss before random_parameters: both extend the Organ module line
    return rp.transform_native(template, request)   # last: it edits lines other hooks may have rewritten (SLA, pairs)


def export_diagnostics(source: Path, output: Path, request: dict) -> None:
    """Publish clocks/offsets and dormant axes from validated native states only."""
    from collections import defaultdict
    from tools.vtc_growth_io import rows, write_csv, write_json
    if not any(enabled(request, name) for name in SWITCHES):
        return
    states = defaultdict(dict)
    for s in rows(source / 'states.csv'):
        states[int(s['day'])][s['organ_id']] = s
    if enabled(request, 'leaf_development'):
        result = []
        for day, current in sorted(states.items()):
            require_unique_leaf_order(current)
            for s in current.values():
                if s['class'] not in ('Leaflet', 'Petiole', 'RachisSegment'):
                    continue
                imported = int(s['birth_day']) < 0
                count = max((int(t['pair_index']) for t in current.values() if t['plant_instance_id']==s['plant_instance_id'] and t['axis_id']==s['axis_id'] and t['true_leaf_order']==s['true_leaf_order'] and t['class']=='Leaflet' and int(t['side'])!=0), default=0)
                offset = 'unavailable' if imported else offset_dd(request, s['class'], int(s['pair_index']), count, int(s['side']))
                maximum = float(s['max_mass_gDM'])
                result.append(dict(day=day, plant_instance_id=s['plant_instance_id'], organ_id=s['organ_id'],
                    organ_class=s['class'], linear_thermal_age_dd=s['thermal_age_dd'], offset_dd=offset,
                    age_origin='supplied_imported_age' if imported else 'compound_birth',
                    physiological_age_dd=target_age(request,s,current), mass_fraction=float(s['mass_gDM'])/maximum if maximum else 'unavailable'))
        write_csv(output / 'leaf_growth_clocks.csv', result, ['day','plant_instance_id','organ_id','organ_class',
            'linear_thermal_age_dd','offset_dd','age_origin','physiological_age_dd','mass_fraction'])
    if request.get('axes'):
        from tools.fsp_axes_segments import axis_inventory
        apices = rows(source / 'apices.csv'); inventory = []
        for day, current in sorted(states.items()):
            inventory.extend(dict(day=day, **r) for r in axis_inventory(request, list(current.values()), [s for s in apices if int(s['day'])==day]))
        write_csv(output / 'axis_inventory.csv', inventory)
    write_json(output / 'engine_hooks.json', dict(contract=request['engine_hooks'],
        planned_axis_stage=axillary_stage(request) if enabled(request, 'planned_axes') else 'disabled',
        plan_emergence_semantics='bud_break', stage_decision_source_kind='user_assumed', stage_decision_date='2026-10-15',
        initial_primordial_mass_gDM=0., initial_primordial_area_m2=0.,
        imported_G1='Historical pair offsets unavailable; supplied thermal age is the age origin. No name-based reconstruction.',
        imported_geometry='Existing fixed-posture mass scaling retained.',
        generated_rachis='Nonzero stagger: cube root of paired funded blade mass fraction; own funded width retained.',
        maturity_tau95_dd=tau95(request['parameters']['MID_DD']['value'][9],request['parameters']['K']['value'][9]),
        biological_validation='not_established'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(); validate(read_json(args.request))
    print(json.dumps({'schema': SCHEMA, 'valid': True}))


if __name__ == '__main__':
    main()
