"""Two-layer, finite-stock replay of declared water and nutrient fluxes.

This module implements conservation, not substrate hydraulics or root kinetics.
"""
from __future__ import annotations

from copy import deepcopy
import math

from tools.leaf_light.common import StudyError, digest
from tools.nutrient_composition import formulate
from tools.nutrient_contract import (SCHEMA, Unavailable, array, evidence, number,
    obj, require_json, science_readiness, species, text, time, unavailable, vector, zeros)

PROCESS = {'irrigate': 'irrigation', 'transfer': 'layer_transport', 'drain': 'drainage',
           'reuse': 'reuse', 'evaporate': 'evaporation', 'transpire': 'transpiration',
           'uptake': 'uptake', 'retain': 'exchange', 'release': 'exchange',
           'allocate': 'plant_allocation'}
EVENT_FIELDS = {
    'irrigate': {'layer_id', 'batch_id', 'water_ml'},
    'transfer': {'from_layer_id', 'to_layer_id', 'water_ml'},
    'drain': {'layer_id', 'water_ml'}, 'reuse': {'layer_id', 'water_ml'},
    'evaporate': {'layer_id', 'water_ml'}, 'transpire': {'layer_id', 'water_ml'},
    'uptake': {'layer_id', 'nutrient_mg'}, 'retain': {'layer_id', 'nutrient_mg'},
    'release': {'layer_id', 'nutrient_mg'}, 'allocate': {'organ_id', 'nutrient_mg'},
}


def _pool(row: dict, scope: list[str], label: str) -> dict:
    row = obj(row, label)
    evidence(row.get('evidence'), label)
    return dict(water_ml=number(row.get('water_ml'), label + '.water_ml'),
                dissolved_mg=vector(row.get('dissolved_mg'), scope, label + '.dissolved_mg'))


def _add(destination: dict, amounts: dict) -> None:
    for s, amount in amounts.items():
        destination[s] += amount


def _remove(source: dict, amounts: dict) -> None:
    if any(amount > source[s] for s, amount in amounts.items()):
        raise StudyError('Nutrient overdraw: requested flux exceeds available stock')
    for s, amount in amounts.items():
        source[s] -= amount


def _aliquot(pool: dict, amount: float, scope: list[str]) -> dict:
    if amount > pool['water_ml']:
        raise StudyError('Water overdraw: requested flux exceeds available stock')
    if amount == 0:
        return zeros(scope)
    fraction = amount / pool['water_ml']
    mass = {s: pool['dissolved_mg'][s] * fraction for s in scope}
    _remove(pool['dissolved_mg'], mass)
    pool['water_ml'] -= amount
    return mass


def _receive(pool: dict, amount: float, mass: dict) -> None:
    if 'water_capacity_ml' in pool and pool['water_ml'] + amount > pool['water_capacity_ml']:
        raise StudyError('Layer water capacity exceeded; supply explicit transfer/drainage events')
    pool['water_ml'] += amount
    _add(pool['dissolved_mg'], mass)


def _totals(state: dict, scope: list[str]) -> tuple[float, dict]:
    mass = zeros(scope)
    water = math.fsum([row['water_ml'] for row in state['layers'].values()] +
                      [state['drainage']['water_ml'], state['water_lost_ml']])
    for row in state['layers'].values():
        _add(mass, row['dissolved_mg'])
        _add(mass, row['retained_mg'])
    _add(mass, state['drainage']['dissolved_mg'])
    _add(mass, state['plant']['labile_mg'])
    for row in state['plant']['organs'].values():
        _add(mass, row)
    return water, mass


def state_rows(state: dict, scope: list[str], context: dict, event_id: str, at: str) -> list[dict]:
    """Flat CSV-compatible state; zero water has undefined concentration."""
    result = []
    for layer_id, row in state['layers'].items():
        for s in scope:
            result.append(dict(**context, event_id=event_id, at=at, layer_id=layer_id,
                species=s, water_ml=row['water_ml'], dissolved_mg=row['dissolved_mg'][s],
                retained_mg=row['retained_mg'][s],
                concentration_mg_L=row['dissolved_mg'][s] * 1000 / row['water_ml']
                if row['water_ml'] > 0 else None, quantity_basis='mg_element'))
    return result


def _simulate(request: dict) -> dict:
    request = obj(request, 'scenario')
    if request.get('schema_version') != SCHEMA:
        raise StudyError('Unsupported nutrient schema_version')
    scope = species(request.get('tracked_species'))
    context = {k: text(request.get(k), k) for k in ('scenario_id', 'plant_instance_id', 'source_plant_id')}
    start, end = time(request.get('t_start'), 't_start'), time(request.get('t_end'), 't_end')
    if end <= start:
        raise StudyError('Nonpositive scenario interval')
    evidence(request.get('evidence'), 'scenario')
    if request.get('mixing_model') != 'well_mixed_each_layer':
        raise Unavailable('mixing_model_undeclared', 'Explicit within-layer mixing assumption required')
    evidence(request.get('mixing_evidence'), 'mixing model')
    mode = request.get('drainage_mode')
    if mode not in {'discard', 'retain_for_explicit_reuse'}:
        raise Unavailable('drainage_mode_undeclared', 'Declare discard or explicit reuse')
    geom = obj(request.get('geometry'), 'geometry')
    evidence(geom.get('evidence'), 'geometry')
    if geom.get('dimension_basis') != 'internal_filled':
        raise Unavailable('internal_dimensions_unconfirmed', 'Geometry is not inferred from external pot size')
    cross_mm2 = number(geom.get('width_mm'), 'width_mm', positive=True) * number(
        geom.get('depth_mm'), 'depth_mm', positive=True)
    layer_defs = array(request.get('layers'), 'layers')
    if len(layer_defs) != 2:
        raise StudyError('Exactly two ordered layers required, top then bottom')
    layers, geometry_rows = {}, []
    for i, row in enumerate(layer_defs):
        row = obj(row, 'layer')
        identity = text(row.get('layer_id'), 'layer_id')
        if identity in layers:
            raise StudyError('Duplicate layer_id')
        material = text(row.get('material'), 'material')
        if material != ('vermiculite' if i == 0 else 'perlite'):
            raise StudyError('Layer order must be upper vermiculite then lower perlite')
        volume = cross_mm2 * number(row.get('height_mm'), 'height_mm', positive=True) / 1000
        pool = _pool(row, scope, identity)
        pool['retained_mg'] = vector(row.get('retained_mg'), scope, identity + '.retained_mg')
        capacity = number(row.get('water_capacity_ml'), 'water_capacity_ml', positive=True)
        evidence(row.get('capacity_evidence'), 'water capacity')
        if capacity > volume or pool['water_ml'] > capacity:
            raise StudyError('Water/capacity exceeds declared physical bounds')
        pool['water_capacity_ml'] = capacity
        layers[identity] = pool
        geometry_rows.append(dict(layer_id=identity, material=material, geometric_volume_ml=volume,
                                  water_capacity_ml=capacity))
    plant = obj(request.get('plant_initial'), 'plant_initial')
    evidence(plant.get('evidence'), 'plant initial')
    organs = obj(plant.get('organs'), 'organs')
    plant_state = dict(labile_mg=vector(plant.get('labile_mg'), scope, 'plant labile'),
                       organs={text(k, 'organ_id'): vector(v, scope, 'organ nutrient')
                               for k, v in organs.items()})
    drainage = _pool(request.get('drainage_initial'), scope, 'drainage_initial')
    if mode == 'discard' and (drainage['water_ml'] or any(drainage['dissolved_mg'].values())):
        raise StudyError('Discarded-drainage ledger must start empty')
    declarations = obj(request.get('processes'), 'processes')
    if set(declarations) - set(PROCESS.values()):
        raise StudyError('Unknown process declaration')
    for name in sorted(set(PROCESS.values())):
        declaration = declarations.get(name)
        if declaration is None:
            raise Unavailable('process_coverage_missing', name)
        declaration = obj(declaration, name)
        if declaration.get('mode') not in {'prescribed_events', 'assumed_zero'}:
            raise Unavailable('process_not_specified', name)
        evidence(declaration.get('evidence'), name)
        if declaration['mode'] == 'assumed_zero' and declaration['evidence']['kind'] != 'assumed':
            raise StudyError('Zero-process assumption must be labelled assumed')
    state = dict(layers=layers, plant=plant_state, drainage=drainage, water_lost_ml=0.0)
    for pool in [*layers.values(), drainage]:
        if pool['water_ml'] == 0 and any(pool['dissolved_mg'].values()):
            raise Unavailable('dry_pool_chemistry_unspecified', 'initial state')
    initial = deepcopy(state)
    initial_water, initial_mass = _totals(state, scope)
    batches, formulations = {}, {}
    for recipe in array(request.get('recipes'), 'recipes'):
        result = formulate(recipe)
        if result['status'] != 'available':
            raise Unavailable(result['reason'], result['detail'])
        if result['tracked_species'] != scope:
            raise StudyError('Batch species/order mismatch')
        identity = result['batch_id']
        if identity in batches:
            raise StudyError('Duplicate batch_id')
        formulations[identity] = result
        batches[identity] = dict(water_ml=result['final_volume_L'] * 1000,
                                dissolved_mg=deepcopy(result['element_mass_mg']))
    tolerances = obj(request.get('numerics'), 'numerics')
    water_tol = number(tolerances.get('water_abs_tol_ml'), 'water_abs_tol_ml', positive=True)
    mass_tol = number(tolerances.get('mass_abs_tol_mg'), 'mass_abs_tol_mg', positive=True)
    input_water, input_mass, taken_up = 0.0, zeros(scope), zeros(scope)
    water_loss = dict(evaporate=0.0, transpire=0.0)
    records, ids, used_processes = [], set(), set()
    snapshots = state_rows(state, scope, context, 'initial', request['t_start'])
    previous = start
    for event in array(request.get('events'), 'events'):
        event = obj(event, 'event')
        identity = text(event.get('event_id'), 'event_id')
        if identity == 'initial' or identity in ids:
            raise StudyError('Duplicate/reserved event_id')
        ids.add(identity)
        at = time(event.get('at'), 'event.at')
        if at < previous or at > end:
            raise StudyError('Events must be ordered within the scenario interval')
        previous = at
        ev = evidence(event.get('evidence'), identity)
        kind = event.get('kind')
        if kind not in PROCESS:
            raise StudyError('Unsupported event kind')
        if set(event) - (EVENT_FIELDS[kind] | {'event_id', 'at', 'kind', 'evidence'}):
            raise StudyError('Unexpected event fields for kind')
        process = PROCESS[kind]
        if declarations[process]['mode'] != 'prescribed_events':
            raise StudyError('Event conflicts with zero-process declaration')
        used_processes.add(process)
        for key in ('layer_id', 'from_layer_id', 'to_layer_id'):
            if key in EVENT_FIELDS[kind] and event.get(key) not in layers:
                raise StudyError('Unknown or absent layer_id')
        amount = 0.0
        mass = zeros(scope)
        if 'water_ml' in EVENT_FIELDS[kind]:
            amount = number(event.get('water_ml'), 'event.water_ml')
        if 'nutrient_mg' in EVENT_FIELDS[kind]:
            mass = vector(event.get('nutrient_mg'), scope, 'event nutrient_mg')
        if kind == 'irrigate':
            if event.get('batch_id') not in batches:
                raise StudyError('Unknown irrigation batch_id')
            mass = _aliquot(batches[event['batch_id']], amount, scope)
            _receive(layers[event['layer_id']], amount, mass)
            input_water += amount
            _add(input_mass, mass)
        elif kind == 'transfer':
            if event['from_layer_id'] == event['to_layer_id']:
                raise StudyError('Self-transfer is invalid')
            mass = _aliquot(layers[event['from_layer_id']], amount, scope)
            _receive(layers[event['to_layer_id']], amount, mass)
        elif kind == 'drain':
            mass = _aliquot(layers[event['layer_id']], amount, scope)
            _receive(state['drainage'], amount, mass)
        elif kind == 'reuse':
            if mode != 'retain_for_explicit_reuse':
                raise StudyError('Discarded drainage cannot be reused')
            mass = _aliquot(state['drainage'], amount, scope)
            _receive(layers[event['layer_id']], amount, mass)
        elif kind in {'evaporate', 'transpire'}:
            pool = layers[event['layer_id']]
            if amount > pool['water_ml']:
                raise StudyError('Water loss exceeds available water')
            pool['water_ml'] -= amount
            state['water_lost_ml'] += amount
            water_loss[kind] += amount
        elif kind == 'uptake':
            _remove(layers[event['layer_id']]['dissolved_mg'], mass)
            _add(state['plant']['labile_mg'], mass)
            _add(taken_up, mass)
        elif kind in {'retain', 'release'}:
            pool = layers[event['layer_id']]
            src, dst = ('dissolved_mg', 'retained_mg') if kind == 'retain' else ('retained_mg', 'dissolved_mg')
            _remove(pool[src], mass)
            _add(pool[dst], mass)
        elif kind == 'allocate':
            if event.get('organ_id') not in state['plant']['organs']:
                raise StudyError('Unknown organ_id; declare its initial nutrient pool explicitly')
            _remove(state['plant']['labile_mg'], mass)
            _add(state['plant']['organs'][event['organ_id']], mass)
        # A zero-water dissolved pool needs a separate precipitation/dry-pool law.
        for pool in [*layers.values(), state['drainage']]:
            if pool['water_ml'] == 0 and any(pool['dissolved_mg'].values()):
                raise Unavailable('dry_pool_chemistry_unspecified', identity)
        water, total = _totals(state, scope)
        water_residual = water - initial_water - input_water
        residual = {s: total[s] - initial_mass[s] - input_mass[s] for s in scope}
        if abs(water_residual) > water_tol or any(abs(x) > mass_tol for x in residual.values()):
            raise StudyError('Conservation check failed')
        records.append(dict(**context, event_id=identity, at=event['at'], kind=kind,
            elapsed_seconds=(at-start).total_seconds(), water_ml=amount, nutrient_mg=mass,
            **{key: event.get(key) for key in ('layer_id', 'from_layer_id', 'to_layer_id', 'batch_id', 'organ_id')},
            evidence=deepcopy(ev), source_event_hash=digest(event),
            water_residual_ml=water_residual, nutrient_residual_mg=residual))
        snapshots.extend(state_rows(state, scope, context, identity, event['at']))
    for name, declaration in declarations.items():
        if declaration['mode'] == 'prescribed_events' and name not in used_processes:
            raise Unavailable('prescribed_process_events_missing', name)
    # Metadata may be measured, but cannot identify individual ions.
    observations = deepcopy(array(request.get('observations', []), 'observations'))
    observation_ids = set()
    for observation in observations:
        observation = obj(observation, 'observation')
        identity = text(observation.get('observation_id'), 'observation_id')
        if identity in observation_ids:
            raise StudyError('Duplicate observation_id')
        observation_ids.add(identity)
        if observation.get('plant_instance_id') != context['plant_instance_id']:
            raise StudyError('Observation plant mismatch')
        at = time(observation.get('at'), 'observation.at')
        if not start <= at <= end:
            raise StudyError('Observation outside scenario interval')
        evidence(observation.get('evidence'), 'observation')
        metric = observation.get('metric')
        units = {'ec_ms_cm': 'mS/cm', 'ph': 'dimensionless', 'pot_mass_g': 'g',
                 'supported_area_cm2': 'cm2', 'height_mm': 'mm'}
        if metric not in units or observation.get('unit') != units[metric]:
            raise StudyError('Unsupported observation metric/unit')
        number(observation.get('value'), 'observation.value')
    water, mass = _totals(state, scope)
    final_residual = {s: mass[s]-initial_mass[s]-input_mass[s] for s in scope}
    return dict(schema_version=SCHEMA, status='available', **context,
        t_start=request['t_start'], t_end=request['t_end'], tracked_species=scope,
        interpretation='conditional_prescribed_flux_replay', quantity_basis='mg_element',
        nitrogen_pool_meaning='element_N_labelled_by_input_form_not_tissue_speciation',
        input_hash=digest(request), initial_state=initial, final_state=state,
        geometry=geometry_rows, formulations=formulations, batch_remainders=batches,
        totals=dict(irrigation_water_ml=input_water, irrigation_nutrient_mg=input_mass,
                    prescribed_uptake_mg=taken_up, water_loss_ml=water_loss,
                    water_residual_ml=water-initial_water-input_water, nutrient_residual_mg=final_residual),
        events=records, layer_states=snapshots, observations=observations,
        observation_comparison=dict(status='unavailable', reason='no_predictive_observation_model',
                                    ec_to_ions='not_performed'),
        scientific_readiness=science_readiness(), process_declarations=deepcopy(declarations),
        provenance=dict(scenario_evidence=deepcopy(request['evidence']),
                        mixing_evidence=deepcopy(request['mixing_evidence']),
                        geometry_evidence=deepcopy(geom['evidence']),
                        layer_evidence={row['layer_id']: deepcopy(row['evidence']) for row in layer_defs},
                        capacity_evidence={row['layer_id']: deepcopy(row['capacity_evidence']) for row in layer_defs},
                        plant_initial_evidence=deepcopy(plant['evidence']),
                        drainage_initial_evidence=deepcopy(request['drainage_initial']['evidence'])),
        validated_nutrient_model=False, growth_prediction=None, growth_feedback_applied=False)


def simulate(request: dict) -> dict:
    """Pure, atomic scenario replay: an invalid later event never commits a prefix."""
    require_json(request)
    try:
        result = _simulate(request)
        require_json(result)
        return result
    except Unavailable as exc:
        return unavailable(exc, request)
