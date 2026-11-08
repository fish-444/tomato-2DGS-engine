"""Measured-state-conditioned growth calculator; assumptions are not calibration.

python -m tools.compound_growth probabilistic --help
"""
from __future__ import annotations

import argparse
from functools import wraps
from copy import deepcopy
from datetime import timedelta, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import expit, logsumexp
from scipy.stats import norm

from tools import compound_growth as observed
from tools.growth_observation_options import (weighted_anchor, accumulate_corrections,
    posture_offsets, adjust_posture_residual)
from tools.growth_day1 import load_day1_protocol, backfill_origin
from tools.growth_observation_evidence import comparison_reason, variance_reason, audit_state
from tools.leaf_light.common import (DLI_UNIT, StudyError, digest, file_hash, number, output_dir,
                                    read_json, rows, write_json, write_rows)
from tools.monthly_growth import ENV_FIELDS

SCHEMA = 'probabilistic-compound-1'
MODEL = 'stochastic-grammar-proposal-1'
UNITS = {
    'wait_dev_days': 'development_day', 'birth_axis_length_mm': 'mm',
    'birth_left_extent_mm': 'mm', 'birth_right_extent_mm': 'mm',
    'azimuth_rad': 'rad', 'elevation_rad': 'rad', 'D_length_mm': 'mm',
    'k_length': '1/development_day', 'left_target_mm': 'mm', 'right_target_mm': 'mm',
    'k_width': '1/development_day', 'rotation_x_rad': 'rad', 'rotation_y_rad': 'rad',
    'rotation_z_rad': 'rad', 'k_rotation': '1/development_day', 'D_area_cm2': 'cm2',
    'k_area': '1/development_day', 'birth_area_cm2': 'cm2',
    'D_segment_mm': 'mm', 'k_segment': '1/development_day',
}
SIGNED = {'azimuth_rad', 'elevation_rad', 'rotation_x_rad', 'rotation_y_rad', 'rotation_z_rad'}
FEATURES = {'axis_length_mm', 'left_extent_mm', 'right_extent_mm', 'spacing_mm',
            'azimuth_sin', 'azimuth_cos', 'order_index', *ENV_FIELDS}
ENV_UNITS=dict(zip(ENV_FIELDS,('deg C','%','umol/m2/s','h/day','mS/cm','dimensionless','ppm')))
EVIDENCE = {'measured', 'user_assumed', 'literature_informed', 'fitted'}
BETA_TRAITS = {'length': 'mm', 'left': 'mm', 'right': 'mm', 'segment': 'mm', 'area': 'cm2'}
for _trait, _unit_name in BETA_TRAITS.items():
    UNITS.update({f'{_trait}_x_max_{_unit_name}': _unit_name,
                  f'{_trait}_a_m_dev_days': 'development_day',
                  f'{_trait}_a_e_dev_days': 'development_day',
                  f'{_trait}_a_m_over_a_e': 'dimensionless'})
LIGHT_METRICS = {'dli_mol_m2_day', 'ppfd_umol_m2_s', 'photoperiod_h'}
FEATURES.add('dli_mol_m2_day')
ENV_UNITS['dli_mol_m2_day'] = DLI_UNIT


def beta_shape(alpha: float, a_m: float, a_e: float) -> float:
    """Determinate beta shape on development days, with explicit support."""
    if not 0 < _num(a_m, 'a_m') < _num(a_e, 'a_e'):
        raise StudyError('determinate_beta requires 0 < a_m < a_e')
    alpha = _num(alpha, 'alpha', 0)
    if alpha >= a_e: return 1.0
    if alpha == 0: return 0.0
    return (1 + (a_e-alpha)/(a_e-a_m)) * (alpha/a_e)**(a_e/(a_e-a_m))


def _beta_value(v: dict, trait: str, birth: float, age: float) -> float | None:
    target = v.get(f'{trait}_x_max_{BETA_TRAITS[trait]}')
    if target is None: return None
    _num(target, 'beta x_max', 0)
    end = v[f'{trait}_a_e_dev_days']
    peak = v.get(f'{trait}_a_m_dev_days')
    ratio = v.get(f'{trait}_a_m_over_a_e')
    if (peak is None) == (ratio is None):
        raise StudyError('Declare exactly one a_m or a_m_over_a_e per beta coordinate')
    if ratio is not None: peak = ratio * end
    original_birth = v.get('beta_birth_values', {}).get(trait, birth)
    if original_birth is None: return None
    reference = v.get('reference_size_targets', {}).get(f'{trait}_x_max_{BETA_TRAITS[trait]}', target)
    if reference < original_birth:
        raise StudyError('determinate_beta x_max < x_birth: ' + trait)
    anchor = v.get('beta_anchor_age', 0.0)
    change = beta_shape(age+anchor, peak, end)-beta_shape(anchor, peak, end)
    if target < original_birth:
        if not v.get('dli_binding'): raise StudyError('determinate_beta x_max < x_birth: ' + trait)
        return birth if age == 0 else None
    return birth + (target-original_birth) * change


def normalized_temperature_clock(*, t_base_c: float, t_ref_c: float, m_max: float,
                                 upper_knot_c: float, intervals: list[dict],
                                 source_kind: str, source_ref: str, applicability: str,
                                 lower_knot_c: float | None = None) -> dict:
    """Build D9 knots; no crop coefficients or extrapolation are supplied."""
    base = _num(t_base_c, 'T_base'); ref = _num(t_ref_c, 'T_ref')
    cap = _num(m_max, 'm_max'); upper = _num(upper_knot_c, 'upper knot')
    if ref <= base or cap < 1 or upper < ref:
        raise StudyError('Normalized clock needs T_base < T_ref <= upper knot and m_max >= 1')
    values = sorted({base, ref, upper, *([base+cap*(ref-base)] if base+cap*(ref-base) < upper else [])})
    if lower_knot_c is not None:
        lower = _num(lower_knot_c, 'lower knot')
        if lower >= base: raise StudyError('Declared lower knot must be below T_base')
        values.insert(0, lower)
    clock = dict(kind='piecewise_linear_response', target='development_clock', equation_version='piecewise_linear-1',
                 input_metric='temperature_c', input_unit='deg C', source_kind=source_kind, source_ref=source_ref,
                 applicability=applicability, intervals=deepcopy(intervals),
                 normalization=dict(t_base_c=base, t_ref_c=ref, m_max=cap, upper_knot_c=upper),
                 knots=[dict(value=x, multiplier=max(0., min((x-base)/(ref-base), cap))) for x in values])
    if lower_knot_c is not None: clock['normalization']['lower_knot_c'] = lower
    _clock({'clock': clock}, 30)
    return clock


def validate_calibration_clock(c: dict, calibration_reference: dict | None) -> None:
    if c.get('clock', {}).get('kind') == 'calendar': return
    clock = c['clock']; ref = calibration_reference or {}
    if clock.get('kind') != 'piecewise_linear_response' or clock.get('input_metric') != 'temperature_c':
        raise StudyError('Calendar coefficients require a normalized temperature clock')
    _evidence(ref, 'calibration temperature setpoint')
    if ref.get('kind') != 'fixed_setpoint' or ref.get('time_basis') != 'calendar_day':
        raise StudyError('Clock conversion needs frozen fixed_setpoint calendar calibration evidence; fluctuating histories are unsupported')
    temperature = _num(ref.get('temperature_c'), 'calibration temperature setpoint')
    _clock(c, 30)
    xs = [k['value'] for k in clock['knots']]; ys = [k['multiplier'] for k in clock['knots']]
    tol = c.get('computation', {}).get('numeric_tolerance', (c.get('computation', {}).get('numeric_tolerances') or {}).get('absolute'))
    if tol is None or not xs[0] <= temperature <= xs[-1] or not math.isclose(float(np.interp(temperature,xs,ys)),1,abs_tol=tol,rel_tol=0):
        raise StudyError('Calendar coefficients require m(T_ref) = 1 at the declared calibration setpoint')


def environment_audit(c: dict) -> dict:
    """Audit executable paths and frozen calibration responses, not metadata values."""
    paths: list[dict] = []
    def add(path: str, metric: str, target: str) -> None:
        paths.append(dict(path=path, metric=metric, target=target))
    clock = c.get('clock') or {}
    if clock.get('kind') == 'piecewise_linear_response': add('clock.input_metric', clock['input_metric'], 'clock')
    for i,p in enumerate(c.get('parameter_particles') or []):
        for name,f in p['families'].items():
            for j,key in enumerate(f.get('features', [])):
                metric = key.removesuffix('_missing')
                if metric in ENV_UNITS and any(row[j] != 0 for row in f['coefficients']):
                    add(f'parameter_particles[{i}].families.{name}.features[{j}]', metric, 'family')
        for aid,ap in p.get('axes', {}).items():
            for key,coef in (ap.get('stop_logit') or {}).get('coefficients', {}).items():
                if key in ENV_UNITS and coef != 0: add(f'parameter_particles[{i}].axes.{aid}.stop_logit.coefficients.{key}',key,'stop')
    light = c.get('light_response')
    if light:
        add('light_response', 'dli_mol_m2_day', 'size_target')
        for curve, binding in light.get('coefficients', {}).items():
            paths.append(dict(path=f'light_response.coefficients.{curve}', metric='dli_mol_m2_day',
                              target='size_target',growth_curve=curve,coefficient_id=binding.get('coefficient_id'),
                              role=binding.get('role'),basis_id=light.get('basis_id')))
    for i,feature in enumerate((c.get('calibration') or {}).get('applied_features', [])):
        add(f'calibration.applied_features[{i}]', feature['column'], feature['role'])
    # Older frozen packs did not store feature provenance: do not assert no coupling.
    unknown = bool(c.get('calibration') and 'applied_features' not in c['calibration'])
    if unknown: add('calibration', 'unknown', 'provenance_unavailable')
    light_used = any(x['metric'] in LIGHT_METRICS or x['metric'] not in ENV_UNITS for x in paths)
    env_used = any(x['metric'] in ENV_UNITS or x['metric'] == 'unknown' for x in paths)
    return dict(p3_light_coupling='size_target_dli' if light else 'none',
                p3_environment_paths=paths, environment_used=env_used, light_used=light_used,
                environment_audit_status='unavailable' if unknown else 'complete')


def _tissue_basis_ids(value: Any) -> set[str]:
    """Collect supplied tissue-basis identities without interpreting unknown tissue."""
    result: set[str] = set()
    if isinstance(value, dict):
        for key,item in value.items():
            if key.endswith('area_basis_id') and isinstance(item,str): result.add(item)
            if key.endswith('area_basis_ids') and isinstance(item,list):
                result.update(x for x in item if isinstance(x,str))
            result.update(_tissue_basis_ids(item))
    elif isinstance(value,list):
        for item in value: result.update(_tissue_basis_ids(item))
    return result


def _light_contract_issues(light: dict, c: dict) -> list[dict]:
    """Collect all missing/different scientific fields before numeric validation."""
    aliases=[x for x in (light.get('applicability_interval'),light.get('dli_applicability_interval'),c.get('dli_applicability_interval')) if x is not None]
    if aliases and any(x!=aliases[0] for x in aliases[1:]): raise StudyError('Conflicting DLI applicability interval declarations')
    if aliases: light['applicability_interval']=deepcopy(aliases[0])
    issues=[]
    def required(obj: dict, prefix: str, keys: tuple[str,...], expected: dict | None=None) -> None:
        if not isinstance(obj,dict): raise StudyError(prefix+': object required')
        for key in keys:
            value=obj.get(key)
            if value is None or value=='': issues.append(dict(field=prefix+key,kind='missing'))
            elif key in ('position_mm','normal_unit_xyz') and isinstance(value,list):
                issues.extend(dict(field=prefix+key+f'[{i}]',kind='missing') for i,x in enumerate(value) if x is None)
        for key,value in (expected or {}).items():
            if obj.get(key) is not None and obj[key]!=value:
                issues.append(dict(field=prefix+key,kind='different',expected=value,actual=obj[key]))
    required(light,'light_response.',('kind','unit','value','center','scale','basis_id','scope','fixed_over_horizon',
        'light_basis','source_kind','source_ref','time_basis'),dict(kind='size_target_dli',unit=DLI_UNIT,
        scope='whole_plant',fixed_over_horizon=True,light_basis='incident',time_basis='complete_local_day'))
    required(light.get('reference') or {},'reference.',('position_mm','normal_unit_xyz','frame_id','coordinate_frame','source_kind','source_ref'))
    required(light.get('local_day') or {},'local_day.',('start_at','end_at','timezone','coverage','source_kind','source_ref'),dict(coverage='complete'))
    required(light.get('applicability_interval') or {},'applicability_interval.',('min','max','unit','source_kind','source_ref','applicability_source_ref'),dict(unit=DLI_UNIT))
    coefficients=light.get('coefficients') or {}
    if not isinstance(coefficients,dict): raise StudyError('DLI coefficients object required')
    curves=set(coefficients)
    for particle in c.get('parameter_particles') or []:
        names=set(particle.get('existing_blocks',{}).values())
        for axis in particle.get('axes',{}).values():
            names.update(axis.get('segment_families',{}).values())
            names.update(axis[k] for k in ('residual_family','inter_event_family') if axis.get(k))
        curves.update(particle['families'][name].get('growth_curve','exponential') for name in names)
    roles={'exponential':'remaining_increment','determinate_beta':'absolute_potential_maximum'}
    for curve in sorted(curves):
        required(coefficients.get(curve) or {},'coefficients.'+curve+'.',
            ('coefficient_id','role','c_DLI','source_kind','source_ref','applicability_source_ref'),dict(role=roles.get(curve)))
    return issues


def _validate_reference_dli(light: dict, c: dict, origin: dict | None) -> None:
    if light.get('light_basis') != 'incident': raise StudyError('P3 reference DLI must be incident')
    if not light.get('basis_id') or light.get('scope') != 'whole_plant':
        raise StudyError('DLI response requires basis_id and scope=whole_plant')
    if light['basis_id'] in (_tissue_basis_ids(c) | _tissue_basis_ids(origin)):
        raise ScienceUnavailable('basis_collides_with_tissue_basis')
    tissue=c.get('p2_tissue_manifest')
    if tissue:
        if not isinstance(tissue,dict): raise StudyError('p2_tissue_manifest must be a hashed file reference or embedded manifest')
        if 'path' in tissue:
            path=Path(tissue['path'])
            if file_hash(path)!=tissue['sha256']: raise StudyError('P2 tissue manifest hash mismatch')
            tissue=read_json(path)
        if light['basis_id'] in _tissue_basis_ids(tissue): raise ScienceUnavailable('basis_collides_with_tissue_basis')
    ref = light.get('reference') or {}
    _evidence(ref, 'DLI reference position/plane')
    if not ref.get('frame_id') or not ref.get('coordinate_frame'):
        raise StudyError('DLI reference requires a declared frame')
    _arr(ref.get('position_mm'), (3,), 'DLI reference position_mm')
    _unit(ref.get('normal_unit_xyz'), observed.MATH_TOL, 'DLI reference plane normal')
    day = light.get('local_day') or {}
    if day.get('coverage') != 'complete': raise StudyError('DLI requires complete local-day coverage')
    _evidence(day, 'DLI local-day coverage')
    try: zone = ZoneInfo(day['timezone'])
    except (KeyError, ZoneInfoNotFoundError) as exc:
        raise StudyError('DLI local_day requires a valid IANA timezone') from exc
    start = observed.aware_time(day['start_at']).astimezone(zone)
    end = observed.aware_time(day['end_at']).astimezone(zone)
    if (start.hour,start.minute,start.second,start.microsecond) != (0,0,0,0) or (
        end.hour,end.minute,end.second,end.microsecond) != (0,0,0,0) or end.date() != start.date()+timedelta(days=1):
        raise ScienceUnavailable('dli_complete_local_day_unavailable')
    domain = light.get('applicability_interval') or {}
    _evidence(domain, 'DLI applicability interval')
    if domain.get('unit') != DLI_UNIT: raise StudyError('DLI applicability interval unit must be '+DLI_UNIT)
    lower = _num(domain.get('min'), 'DLI domain min', 0); upper = _num(domain.get('max'), 'DLI domain max', 0)
    if lower > upper: raise StudyError('DLI applicability interval is reversed')
    domain_ref=domain['applicability_source_ref']
    if isinstance(domain_ref,dict):
        if not domain_ref.get('path') or not domain_ref.get('sha256') or 'json_pointer' not in domain_ref:
            raise ScienceUnavailable('dli_applicability_source_underspecified')
        source_path=Path(domain_ref['path'])
        if file_hash(source_path)!=domain_ref['sha256']: raise StudyError('DLI applicability source hash mismatch')
        source=read_json(source_path); pointer=domain_ref['json_pointer']
        if not isinstance(pointer,str) or (pointer and not pointer.startswith('/')): raise StudyError('Invalid DLI domain JSON pointer')
        for token in pointer.split('/')[1:] if pointer else []:
            key=token.replace('~1','/').replace('~0','~')
            source=source[int(key)] if isinstance(source,list) else source[key]
        if not isinstance(source,dict) or source.get('unit')!=DLI_UNIT or source.get('min')!=lower or source.get('max')!=upper:
            raise ScienceUnavailable('dli_applicability_source_domain_mismatch')
    if not lower <= light['value'] <= upper: raise ScienceUnavailable('dli_outside_applicability_interval')
    if 'c_DLI' in light:
        raise StudyError('Shared c_DLI is unsupported; declare separate coefficient bindings per growth_curve')
    coefficients = light.get('coefficients') or {}
    roles = {'exponential':'remaining_increment','determinate_beta':'absolute_potential_maximum'}
    identities = set()
    for curve,binding in coefficients.items():
        if curve not in roles or binding.get('role') != roles[curve]:
            raise StudyError('DLI coefficient role must match its growth_curve')
        identity = binding.get('coefficient_id')
        if not isinstance(identity,str) or not identity or identity in identities:
            raise StudyError('DLI coefficient binding identity cannot be shared across growth curves')
        identities.add(identity); _evidence(binding, 'DLI '+curve+' coefficient')
        if binding['applicability_source_ref']!=domain['applicability_source_ref']:
            raise ScienceUnavailable('dli_applicability_coefficient_domain_mismatch')
        _num(binding.get('c_DLI'), 'c_DLI')
    for particle in c.get('parameter_particles') or []:
        families = set(particle.get('existing_blocks', {}).values())
        for axis in particle.get('axes', {}).values():
            families.update(axis.get('segment_families', {}).values())
            families.update(axis[k] for k in ('residual_family','inter_event_family','segment_family') if axis.get(k))
        for name in families:
            curve = particle['families'][name].get('growth_curve','exponential')
            if curve not in coefficients: raise StudyError('Missing DLI coefficient binding for growth_curve: '+curve)


def _validate_d9(c: dict, origin: dict | None = None) -> None:
    profile = c.get('environment_profile')
    if profile not in (None, 'd9'): raise StudyError('Unknown environment_profile')
    light = c.get('light_response')
    if light:
        issues=_light_contract_issues(light,c)
        if issues:
            legacy=any(r.get('actual')=='mol/m2/d' for r in issues)
            raise ScienceUnavailable('legacy_dli_unit_spelling' if legacy else 'legacy_light_response_underspecified',issues)
        _evidence(light, 'DLI response')
        _num(light['value'], 'DLI', 0); _num(light['center'], 'DLI center', 0)
        _validate_reference_dli(light, c, origin)
        if _num(light['scale'], 'DLI scale', 0) == 0: raise StudyError('DLI scale must be positive')
        scenario_dli = (c.get('environment_scenario') or {}).get('values') or {}
        if scenario_dli.get('dli_mol_m2_day') is not None and scenario_dli['dli_mol_m2_day'] != light['value']:
            raise StudyError('DLI response differs from fixed scenario DLI')
    if profile == 'd9' or light:
        for p in c.get('parameter_particles') or []:
            for name,f in p['families'].items():
                for key in f.get('features', []):
                    metric = key.removesuffix('_missing')
                    if metric in LIGHT_METRICS or (profile == 'd9' and metric in ENV_UNITS):
                        raise StudyError('D9 environment/light family feature is not a permitted entry path: '+name+'.'+key)
            for ap in p.get('axes', {}).values():
                for key in (ap.get('stop_logit') or {}).get('coefficients', {}):
                    if key in LIGHT_METRICS or (profile == 'd9' and key in ENV_UNITS):
                        raise StudyError('D9 environment/light stop feature is forbidden: '+key)
    for path in environment_audit(c)['p3_environment_paths']:
        metric, target = path['metric'], path['target']
        if profile == 'd9' and (metric == 'unknown' or metric not in {'temperature_c', 'dli_mol_m2_day'}):
            raise StudyError('D9 forbids coefficient coupling: ' + path['path'] + ' (' + metric + ')')
        if profile == 'd9' and metric == 'temperature_c' and target != 'clock':
            raise StudyError('D9 temperature enters the clock only: ' + path['path'])
        if (profile == 'd9' or light) and metric in LIGHT_METRICS and target != 'size_target':
            raise StudyError('DLI/light must enter size_target_dli only: ' + path['path'])
    if profile == 'd9' and c['clock']['kind'] not in ('calendar', 'piecewise_linear_response'):
        raise StudyError('D9 requires calendar or normalized temperature clock')
    if profile == 'd9' and c['clock']['kind'] != 'calendar':
        norm_ref = c['clock'].get('normalization')
        if not norm_ref: raise StudyError('D9 normalized temperature clock declaration required')
        expected = normalized_temperature_clock(**norm_ref, intervals=c['clock']['intervals'],
            source_kind=c['clock']['source_kind'], source_ref=c['clock']['source_ref'], applicability=c['clock']['applicability'])
        if expected['knots'] != c['clock']['knots']: raise StudyError('D9 clock knots do not match declared normalization')


def _size_response(v: dict | None, c: dict) -> dict | None:
    if v is None or not c.get('light_response'): return v
    v = deepcopy(v); light = c['light_response']
    curve = v.get('growth_curve','exponential'); binding = light['coefficients'][curve]
    factor = math.exp(binding['c_DLI']*(light['value']-light['center'])/light['scale'])
    reference = {}
    for key in list(v):
        if (curve == 'exponential' and key.startswith('D_')) or (curve == 'determinate_beta' and '_x_max_' in key):
            reference[key] = v[key]
            v[key] *= factor
            _num(v[key], 'DLI effective '+key, 0)
    v['reference_size_targets'] = reference
    v['dli_binding'] = dict(binding,growth_curve=curve,basis_id=light['basis_id'],value=light['value'],unit=light['unit'])
    v['size_target_factor'] = factor
    return v


def _check_narrowing(v: dict, state: dict, evidence: dict | None) -> None:
    for side in ('left','right'):
        widths = state.get(side+'_extent_mm'); target = v.get(side+'_target_mm')
        # Exact accepted threshold: do not silently broaden to any/all station differences.
        if widths and target is not None and target < min(widths):
            _evidence(evidence or {}, 'narrowing_assumption required: '+side+'_target_mm < min(observed stations)')


def _check_curve_availability(v: dict | None, state: dict, tr: dict, block_id: str) -> None:
    if not v: return
    unavailable = {}
    if v.get('growth_curve') == 'determinate_beta':
        births = dict(length=state.get('rachis_arc_length_mm'),area=state.get('supported_area_cm2'))
        for side in ('left','right'): births[side] = max(state.get(side+'_extent_mm') or [0.])
        for trait,birth in births.items():
            if birth is None: continue
            target = v.get(f'{trait}_x_max_{BETA_TRAITS[trait]}')
            if target is None: continue
            _beta_value(v,trait,birth,0.)  # Invalid reference targets/times still fail, not masked by DLI.
            if target < birth and v.get('dli_binding'):
                field = {'length':'rachis_arc_length_mm','area':'supported_area_cm2'}.get(trait,trait+'_extent_mm')
                unavailable[field] = 'dli_target_below_birth'
    elif v.get('dli_binding'):
        for side in ('left','right'):
            target = v.get(side+'_target_mm')
            if target is not None and any(w+(target-w)*v['size_target_factor'] < 0 for w in state.get(side+'_extent_mm') or []):
                unavailable[side+'_extent_mm'] = 'dli_negative_width_target'
    if unavailable:
        v['growth_unavailable'] = unavailable
        tr['availability'].extend(dict(component=block_id,quantity=field,reason_code=reason,
            source='light_response',coefficient_id=v['dli_binding']['coefficient_id']) for field,reason in unavailable.items())


def _pmf(spec: dict, key: str, tol: float) -> tuple[list[int], list[float]]:
    _evidence(spec, key)
    items = spec[key]; ns = [r['n'] for r in items]; ps = [_num(r['p'], key+' probability', 0) for r in items]
    if not ns or any(type(n) is not int or n < 0 for n in ns) or len(set(ns)) != len(ns):
        raise StudyError(key+': unique nonnegative integer ranks required')
    if not math.isclose(sum(ps),1,abs_tol=tol,rel_tol=0): raise StudyError(key+': probabilities must sum to one')
    return ns, ps


def _observed_count(a: dict) -> int:
    n = a.get('observed_node_count')
    if n is None: raise ScienceUnavailable('observed_node_count_unavailable')
    if not a.get('node_count_source_ref'): raise ScienceUnavailable('node_count_source_ref_missing')
    if type(n) is not int or n < 0:
        raise StudyError('Reviewed observed_node_count and node_count_source_ref required')
    return n


def _event_match(events: list[dict], evidence: dict) -> dict | None:
    kind = evidence.get('event_kind', 'leaf')
    if kind not in ('leaf','truss','terminal'): raise StudyError('Unknown event_kind')
    return next((e for e in events if e['parent_stem_id'] == evidence['parent_stem_id']
                 and e.get('event_kind','leaf') == kind and
                 (e.get('node_rank') == evidence['node_rank'] if 'node_rank' in evidence else e['event_index'] == evidence['event_index'])),None)


def _terminal_by(tr: dict, day: float, aid: str | None = None) -> bool:
    return any(e.get('event_kind') == 'terminal' and e['event_state']=='appeared' and
               e['appearance_elapsed_days'] <= day and (aid is None or e['parent_stem_id']==aid) for e in tr.get('events',[]))


def _event_likelihood(tr: dict, evidence: dict, through: float) -> bool | None:
    """Hard confirmed-event likelihood; unknown model support is not nonappearance."""
    if evidence.get('visibility') != 'confirmed': return True
    if evidence.get('event_kind')=='truss' and _truss_evidence_reason(tr,evidence): return None
    if 'node_rank' in evidence and (type(evidence['node_rank']) is not int or evidence['node_rank'] < 0):
        raise StudyError('Event node_rank must be a nonnegative integer')
    event = _event_match(tr.get('events', []), evidence)
    kind = evidence.get('kind', 'present')
    end = evidence.get('first_present_day', evidence.get('through_day', through))
    if not 0 <= _num(end, 'event evidence end') <= through: raise StudyError('Event evidence extends beyond update time')
    if kind == 'interval' and not 0 <= _num(evidence['last_absent_day'],'last_absent_day') < end:
        raise StudyError('Invalid event observation interval')
    if kind not in ('present','interval','right_censored'): raise StudyError('Unknown event observation')
    if event is None:
        aid = evidence['parent_stem_id']; n_max = tr.get('n_max_drawn', {}).get(aid)
        if n_max is None or 'node_rank' not in evidence: return None
        rule = tr.get('parameter', {}).get('axes', {}).get(aid, {}).get('truss_rule')
        if evidence.get('event_kind') == 'truss' and not rule: return None
        # A fully simulated budgeted axis has no such future event in its supported horizon.
        return kind == 'right_censored'
    time = event['appearance_elapsed_days']
    if kind == 'right_censored':
        return event['event_state'] in ('stopped','beyond_horizon') or (time is not None and time > end)
    if kind == 'present': return event['event_state'] == 'appeared' and time is not None and time <= end
    return time is not None and evidence['last_absent_day'] < time <= end


class ScienceUnavailable(StudyError):
    """A declared conditional calculation has no supported input, not malformed data."""
    def __init__(self, reason: str, details: list[dict] | None = None):
        super().__init__(reason)
        self.reason_code=reason
        self.details=details or []


def _truss_evidence_reason(tr: dict, evidence: dict) -> str | None:
    rule=tr.get('parameter',{}).get('axes',{}).get(evidence['parent_stem_id'],{}).get('truss_rule') or {}
    reason=_truss_rule_reason(rule)
    if reason: return reason
    if not evidence.get('truss_observable'): return 'truss_evidence_observable_undeclared'
    if rule['truss_observable']!=evidence['truss_observable']: return 'truss_observable_mismatch'
    return None


def _truss_rule_reason(rule: dict) -> str | None:
    if not rule.get('truss_observable'): return 'truss_observable_undeclared'
    if any(rule.get(k) in (None,'',[]) for k in ('first_truss_rank_pmf','interval_leaves','source_kind','source_ref')):
        return 'truss_rule_underspecified'
    return None


def _input_contract(function):
    @wraps(function)
    def checked(*args,**kwargs):
        try:
            return function(*args,**kwargs)
        except ScienceUnavailable as exc:
            if function.__name__=='validate_model_config':
                return dict(status='unavailable',config=_obj(args[0] if args else kwargs['config']),
                    unavailable=[dict(_reason('science',exc.reason_code),fields=exc.details)])
            if function.__name__=='update_forecast':
                previous=_load_run(args[0] if args else kwargs['previous_run'])
                return dict(status='unavailable',bank=previous,previous_run_id=previous['run_id'],
                    unavailable=[dict(reason_code=exc.reason_code,fields=exc.details)],parameter_update='unavailable')
            raise
        except StudyError:
            raise
        except (KeyError,TypeError,ValueError,IndexError,OSError) as exc:
            raise StudyError(f'{function.__name__}: invalid input: {exc}') from exc
    return checked


def _num(x: Any, name: str, low: float | None = None) -> float:
    if isinstance(x, bool):
        raise StudyError(name + ': boolean is not numeric')
    v = number(x, name)
    if low is not None and v < low:
        raise StudyError(name + ': below minimum')
    return v


def _arr(x: Any, shape: tuple, name: str) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise StudyError(name + ': invalid array') from exc
    if a.shape != shape or not np.isfinite(a).all():
        raise StudyError(name + ': wrong shape or nonfinite')
    return a


def _cov(x: Any, n: int, tol: float, name: str) -> np.ndarray:
    if n == 0 and x == []:
        return np.zeros((0,0))
    a = _arr(x, (n, n), name)
    if not np.allclose(a, a.T, atol=tol, rtol=0) or (n and np.linalg.eigvalsh(a).min() < -tol):
        raise StudyError(name + ': covariance must be symmetric positive semidefinite')
    return a


def _reason(component: str, reason: str, source: Any = None) -> dict:
    return dict(component=component, reason_code=reason, source=source)


def _obj(value: dict | str | Path) -> dict:
    return deepcopy(value) if isinstance(value, dict) else read_json(Path(value))


def _history(path: str | Path) -> tuple[list[dict], dict]:
    p = Path(path)
    manifest_path = p / 'observations.json' if p.is_dir() else p
    states, context = observed.load_history(manifest_path)
    manifest = read_json(manifest_path)
    day1=manifest.get('day1_protocol')
    if day1:
        path=(manifest_path.parent/day1['path']).resolve()
        if file_hash(path)!=day1['sha256']: raise StudyError('Day-1 source hash mismatch')
        context['day1_protocol']=load_day1_protocol(path)
        context['source_hashes'].update(context['day1_protocol']['source_hashes'])
        for state in states:
            matches=[r for r in context['day1_protocol']['records'] if r['plant_id']==state['plant_id'] and
                r['capture_session_id']==state['session_id'] and r['captured_at'] and state.get('observed_at') and
                observed.aware_time(r['captured_at'])==observed.aware_time(state['observed_at'])]
            if len(matches)>1: raise StudyError('Ambiguous Day-1 phase record')
            if matches and matches[0]['capture_phase']:
                if state.get('capture_phase') not in (None,matches[0]['capture_phase']):
                    raise StudyError('Day-1 phase conflicts with observation evidence')
                state.update(capture_phase=matches[0]['capture_phase'],day1_source_row=matches[0]['row_id'])
    table = manifest.get('event_observations')
    if table:
        source = (manifest_path.parent / table['path']).resolve()
        if file_hash(source) != table['sha256']: raise StudyError('Event observation table hash mismatch')
        events = rows(source)
        for event in events:
            if event.get('event_kind') not in ('leaf','truss','terminal') or not event.get('parent_stem_id') or not event.get('plant_id'):
                raise StudyError('Event observation kind/identity required')
            rank = _num(event['node_rank'], 'observed node_rank', 0)
            if not rank.is_integer(): raise StudyError('Observed node_rank must be integer')
            event['node_rank'] = int(rank)
            observed.aware_time(event['observed_at'])
            if event.get('visibility') not in ('confirmed','occluded','missed') or not event.get('source_ref'):
                raise StudyError('Event visibility and source_ref required')
        context['events'] = events; context['source_hashes'][str(source)] = file_hash(source)
    return states, context


def _unit(v: Any, tol: float, label: str) -> np.ndarray:
    a = _arr(v, (3,), label)
    if abs(float(np.linalg.norm(a)) - 1) > tol:
        raise StudyError(label + ': expected unit vector')
    return a


def _frames(points: np.ndarray, reference: Any, tol: float) -> list:
    frames = []
    e1 = None
    old = None
    for delta in np.diff(points, axis=0):
        length = np.linalg.norm(delta)
        if length <= tol:
            raise StudyError('Collapsed parent segment')
        e3 = delta / length
        if e1 is None:
            r = _arr(reference, (3,), 'reference_vector')
            e1 = r - np.dot(r, e3) * e3
        else:
            cross = np.cross(old, e3)
            sine, cosine = np.linalg.norm(cross), float(np.dot(old, e3))
            if sine <= tol and cosine < 0:
                raise StudyError('Antiparallel parent tangents need reviewed frames')
            if sine > tol:
                e1 = Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).apply(e1)
            e1 = e1 - np.dot(e1, e3) * e3
        if np.linalg.norm(e1) <= tol:
            raise StudyError('Degenerate local azimuth reference')
        e1 /= np.linalg.norm(e1)
        frames.append(np.column_stack((e1, np.cross(e3, e1), e3)).tolist())
        old = e3
    return frames


@_input_contract
def prepare_origin(history_dir, structure_path, origin_config) -> dict:
    """Adapt the existing immutable observation bundle; no geometry inference."""
    cfg = _obj(origin_config)
    states, context = _history(history_dir)
    t = observed.aware_time(cfg['observed_at'])
    plant = cfg['plant_id']
    selected = [s for s in states if s['plant_id'] == plant and s['observed_at']
                and observed.aware_time(s['observed_at']) == t]
    if cfg.get('session_id'):
        selected = [s for s in selected if s['session_id'] == cfg['session_id']]
    if len({s['session_id'] for s in selected}) > 1:
        raise StudyError('Origin needs one explicit session')
    blocks = sorted(deepcopy(selected), key=lambda s: s['compound_leaf_id'])
    reasons = [_reason(s['compound_leaf_id'], flag) for s in blocks for flag in s['qc_flags']]
    if not blocks:
        reasons.append(_reason('origin', 'no_observations_at_origin'))
    hashes = deepcopy(context['source_hashes'])
    axes, attachments = {}, {}
    if structure_path is not None:
        p = Path(structure_path).resolve()
        structure = read_json(p)
        hashes[str(p)] = file_hash(p)
        if structure.get('coordinate_frame') != context.get('coordinate_frame') or structure.get('frame_id') != context.get('frame_id'):
            raise StudyError('Structure/observation frame mismatch')
        if structure.get('plant_id') != plant or observed.aware_time(structure['observed_at']) != t:
            raise StudyError('Structure origin mismatch')
        tol = _num(cfg.get('numeric_tolerance'), 'numeric_tolerance', 0)
        if tol == 0:
            raise StudyError('Positive numerical tolerance required')
        table_data = {}
        for name in ('parent_axis_points', 'attachments'):
            entry = structure.get(name)
            if entry is None:
                table_data[name] = []
                continue
            f = (p.parent / entry['path']).resolve()
            h = file_hash(f)
            if entry.get('sha256') != h:
                raise StudyError('Structure table hash mismatch: ' + name)
            hashes[str(f)] = h
            table_data[name] = rows(f)
        for raw in structure.get('axes', []):
            a = deepcopy(raw)
            aid = a['parent_stem_id']
            if not aid or '/' in aid:
                raise StudyError('Parent axis IDs must be nonempty and cannot contain reserved slash')
            if aid in axes:
                raise StudyError('Duplicate parent axis')
            if a.get('apex_status') not in ('active', 'stopped', 'unknown'):
                raise StudyError('Explicit apex status required')
            rr = sorted([r for r in table_data['parent_axis_points'] if r['parent_stem_id'] == aid],
                        key=lambda r: _num(r['order'], 'point order'))
            a.update(points_mm=None, frames=None, segments=[])
            if rr:
                if len(rr) < 2 or len({r['point_id'] for r in rr}) != len(rr) or len({r['order'] for r in rr}) != len(rr):
                    raise StudyError('Invalid parent polyline IDs/order')
                if any(r.get('review_status') != 'confirmed' or not r.get('source_ref') for r in rr):
                    reasons.append(_reason(aid, 'parent_polyline_unreviewed'))
                else:
                    pts = np.array([[_num(r[k], k) for k in ('x_mm', 'y_mm', 'z_mm')] for r in rr])
                    a['points_mm'] = pts.tolist()
                    try:
                        a['frames'] = _frames(pts, a.get('reference_vector'), tol)
                    except StudyError as exc:
                        reasons.append(_reason(aid, 'local_frame_unavailable', str(exc)))
                    a['point_ids'] = [r['point_id'] for r in rr]
                    for i, d in enumerate(np.diff(pts, axis=0)):
                        length = float(np.linalg.norm(d))
                        if length <= tol:
                            raise StudyError('Collapsed parent segment')
                        a['segments'].append(dict(segment_id=f'{rr[i]["point_id"]}:{rr[i+1]["point_id"]}',
                                                  length0_mm=length, direction=(d / length).tolist(), born_dev=0.0))
                    if a.get('apex_point_id') != rr[-1]['point_id'] or a.get('apex_review') != 'confirmed':
                        a['apex_available'] = False
                        reasons.append(_reason(aid, 'apex_unavailable'))
                    else:
                        a['apex_available'] = True
            else:
                reasons.append(_reason(aid, 'parent_geometry_unavailable'))
            axes[aid] = a
        for r in table_data['attachments']:
            bid, aid = r['compound_leaf_id'], r['parent_stem_id']
            if bid in attachments:
                raise StudyError('Duplicate compound attachment')
            block = next((s for s in blocks if s['compound_leaf_id'] == bid), None)
            a = axes.get(aid)
            if block is None or a is None or block['parent_stem_id'] != aid:
                raise StudyError('Orphan or mismatched attachment')
            if not r.get('review_ref') or not a['points_mm'] or block['attachment_xyz_mm'] is None:
                reasons.append(_reason(bid, 'attachment_unavailable'))
                continue
            segid = r['segment_start_id'] + ':' + r['segment_end_id']
            ids = [s['segment_id'] for s in a['segments']]
            if segid not in ids:
                raise StudyError('Attachment segment not on parent')
            i = ids.index(segid)
            fraction = _num(r['fraction'], 'attachment fraction', 0)
            if fraction > 1:
                raise StudyError('Attachment fraction exceeds one')
            projected = np.array(a['points_mm'][i]) + fraction * (np.array(a['points_mm'][i+1]) - a['points_mm'][i])
            offset = np.array(block['attachment_xyz_mm']) - projected
            residual = float(np.linalg.norm(offset))
            limit = cfg.get('projection_tolerance_mm')
            if abs(residual - _num(r['projection_residual_mm'], 'projection residual', 0)) > tol:
                raise StudyError('Inconsistent recorded projection residual')
            if limit is None or residual > _num(limit, 'projection_tolerance_mm', 0):
                reasons.append(_reason(bid, 'projection_tolerance_unavailable_or_exceeded'))
                continue
            attachments[bid] = dict(r, segment_index=i, fraction=fraction, offset_mm=offset.tolist())
        for aid, a in axes.items():
            seen = {aid}
            parent = a.get('base_attachment')
            while parent:
                pid = parent['parent_stem_id']
                if pid in seen or pid not in axes:
                    raise StudyError('Parent axis cycle or missing parent')
                seen.add(pid)
                parent = axes[pid].get('base_attachment')
            ref=a.get('base_attachment')
            if ref and a.get('points_mm') is not None:
                pa=axes[ref['parent_stem_id']]
                if pa.get('points_mm') is None:
                    reasons.append(_reason(aid,'parent_axis_base_unavailable'))
                else:
                    i=ref.get('segment_index'); f=_num(ref.get('fraction'),'base fraction',0)
                    if type(i) is not int or not 0<=i<len(pa['segments']) or f>1 or not ref.get('review_ref'):
                        raise StudyError('Invalid/unreviewed lateral-axis base attachment')
                    off=_arr(ref.get('offset_mm'),(3,),'base offset')
                    pp=np.array(pa['points_mm'])
                    if not np.allclose(pp[i]+f*(pp[i+1]-pp[i])+off,a['points_mm'][0],atol=tol,rtol=0):
                        raise StudyError('Lateral-axis base does not preserve measured origin')
    for s in blocks:
        if s['parent_stem_id'] not in axes:
            axes[s['parent_stem_id']] = dict(parent_stem_id=s['parent_stem_id'], apex_status='unknown',
                                            points_mm=None, frames=None, segments=[])
        if s['compound_leaf_id'] not in attachments:
            reasons.append(_reason(s['compound_leaf_id'], 'material_attachment_unavailable'))
    if context.get('frame_stability')!='confirmed' or not context.get('frame_evidence'):
        for a in axes.values(): a.update(points_mm=None,frames=None,segments=[],apex_available=False)
        attachments={}
    observed._unchanged(hashes)
    origin = dict(schema_version=SCHEMA, plant_id=plant, session_id=cfg.get('session_id') or
                  (blocks[0]['session_id'] if blocks else None), observed_at=cfg['observed_at'],
                  frame_id=context.get('frame_id'), coordinate_frame=context.get('coordinate_frame'),
                  data_kind=context['data_kind'], blocks=blocks, axes=axes, attachments=attachments,
                  history=[s for s in states if s['plant_id'] == plant and s['observed_at'] and
                           observed.aware_time(s['observed_at']) <= t], source_hashes=hashes,
                  availability=reasons, origin_config=cfg)
    if context.get('events'):
        origin['event_history'] = [e for e in context['events'] if e['plant_id']==plant and
                                   observed.aware_time(e['observed_at']) <= observed.aware_time(cfg['observed_at'])]
    if context.get('day1_protocol'):
        origin=backfill_origin(origin,context['day1_protocol'],cfg.get('day1_axis_id'))
    origin['origin_id'] = cfg.get('origin_id') or digest(origin)
    return origin


def blank_model() -> dict:
    return dict(schema_version=SCHEMA, model_version=MODEL, horizon_days=30, sampler='joint_gaussian',
                topology_scope='provided_axes', leaf_retention_assumption=None, clock={'kind': 'calendar'},
                environment_scenario={'id': None, 'values': None, 'response_ref': None}, light_features=[],
                parameter_particles=None, computation=dict(seed=None, sample_count=None, output_days=[0,7,14,21,30],
                numeric_tolerances=None, max_events_per_axis=None))


def _evidence(p: dict, label: str) -> None:
    if not p.get('source_kind') or not p.get('source_ref'):
        raise ScienceUnavailable('scientific_evidence_missing', [dict(field=label,kind='missing')])
    if p.get('source_kind') not in EVIDENCE:
        raise StudyError(label + ': explicit parameter evidence required')


@_input_contract
def validate_model_config(config, origin) -> dict:
    c = _obj(config)
    if c.get('continuation_only'):
        raise StudyError('Updated trajectory bank is a bounded continuation; supply a new-origin parameter contract for a fresh forecast')
    if origin.get('schema_version')!=SCHEMA or origin.get('coordinate_frame')!='board_mm':
        raise StudyError('Expected prepared board-frame origin')
    observed.aware_time(origin.get('observed_at'))
    ids=[s['compound_leaf_id'] for s in origin['blocks']]
    if len(ids)!=len(set(ids)): raise StudyError('Duplicate measured compound identity')
    for s in origin['blocks']:
        if s.get('eligible_state'):
            traits=observed.axis_traits(s['rachis_points_mm'])
            if not s.get('trait_unavailable', {}).get('direction_unit_xyz') and not np.isclose(traits['rachis_arc_length_mm'],s['rachis_arc_length_mm']):
                raise StudyError('Origin axis summary does not match measured points')
            if s.get('geometry_status')!='observed_axis' or s.get('identity_status')!='confirmed':
                raise StudyError('Predicted/unconfirmed geometry cannot initialize measured origin')
    if c.get('schema_version') != SCHEMA or c.get('model_version') != MODEL:
        raise StudyError('Unsupported model/schema version')
    if c.get('horizon_days') != 30 or c.get('topology_scope') != 'provided_axes':
        raise StudyError('This model requires 30 days and provided_axes scope')
    missing = []
    if c.get('sampler') == 'flow_matching':
        missing.append(_reason('sampler', 'flow_matching_not_trained_or_enabled'))
    elif c.get('sampler') != 'joint_gaussian':
        raise StudyError('Unknown sampler')
    comp = c.get('computation') or {}
    # Preserve the detailed contract's plural field; the scalar spelling is an explicit compatibility alias.
    if comp.get('numeric_tolerances') is not None:
        absolute=comp['numeric_tolerances']['absolute']
        if comp.get('numeric_tolerance') is not None and comp['numeric_tolerance']!=absolute:
            raise StudyError('Conflicting numeric tolerance fields')
        comp['numeric_tolerance']=absolute
    for k in ('seed', 'sample_count', 'numeric_tolerance', 'max_events_per_axis', 'output_days'):
        if comp.get(k) is None:
            missing.append(_reason('computation', 'missing_' + k))
    for k in ('seed', 'sample_count', 'max_events_per_axis'):
        if comp.get(k) is not None and (type(comp[k]) is not int or comp[k] < (0 if k == 'seed' else 1)):
            raise StudyError(k + ': invalid integer')
    if comp.get('output_days') is not None:
        times = [_num(t, 'output_days', 0) for t in comp['output_days']]
        if not times or len(set(times)) != len(times) or max(times) > 30:
            raise StudyError('Invalid output days')
    tol = comp.get('numeric_tolerance')
    if tol is not None and _num(tol, 'numeric_tolerance', 0) == 0:
        raise StudyError('Positive numeric tolerance required')
    _clock(c, 30)
    scenario = c.get('environment_scenario') or {}
    for key, value in (scenario.get('values') or {}).items():
        if key not in ENV_UNITS:
            raise StudyError('Unknown environment field: ' + key)
        if value is not None:
            _num(value, key)
    for light in c.get('light_features', []):
        for k in ('plant_id', 'compound_leaf_id', 'observed_at', 'area_basis_id', 'metric', 'unit', 'backend', 'source_ref'):
            if not light.get(k):
                raise StudyError('External light missing ' + k)
        if light['plant_id'] != origin['plant_id']:
            raise StudyError('External light plant mismatch')
        observed.aware_time(light['observed_at'])
        if observed.aware_time(light['observed_at'])>observed.aware_time(origin['observed_at']) and light.get('state_source')!='scenario':
            raise StudyError('Future light measurement cannot condition an earlier forecast')
        _num(light['value'], 'external light value', 0)
    if scenario.get('response_ref') is not None:
        raise StudyError('response_ref alone is not executable; supply an explicit clock contract')
    clock=c['clock']
    if clock['kind']=='piecewise_linear_response':
        metric=clock['input_metric']
        if metric in ENV_UNITS and clock['input_unit']!=ENV_UNITS[metric]:
            raise StudyError('Environment response unit mismatch')
        if metric not in ENV_UNITS:
            light=next((x for x in c.get('light_features',[]) if x.get('id')==clock.get('light_feature_id')),None)
            if light is None or light['metric']!=metric or light['unit']!=clock['input_unit'] or not clock.get('fixed_over_horizon'):
                raise StudyError('Light response needs a matching feature and explicit fixed-over-horizon scenario')
            if any(r['input_value']!=light['value'] for r in clock['intervals']):
                raise StudyError('Fixed-light response differs from referenced light metric')
    if c.get('leaf_retention_assumption') != 'no_loss_over_horizon':
        missing.append(_reason('retention','no_loss_assumption_unavailable'))
    particles = c.get('parameter_particles')
    if not particles:
        missing.append(_reason('parameters', 'parameter_particles_unavailable'))
        return dict(status='unavailable', config=c, unavailable=missing)
    if tol is None:
        return dict(status='unavailable', config=c, unavailable=missing)
    try:
        _validate_d9(c, origin)
    except ScienceUnavailable as exc:
        return dict(status='unavailable',config=c,unavailable=[dict(_reason('light_response',exc.reason_code),fields=exc.details)])
    if c.get('max_leaf_age_dev_days') is not None:
        if _num(c['max_leaf_age_dev_days'], 'max_leaf_age_dev_days', 0) == 0:
            raise StudyError('Positive max_leaf_age_dev_days required')
        _evidence(c.get('leaf_age_evidence') or {}, 'leaf age limit')
    ids, weights = set(), []
    for p in particles:
        if not p.get('id') or p['id'] in ids:
            raise StudyError('Duplicate/missing parameter particle ID')
        ids.add(p['id']); weights.append(_num(p.get('weight'), 'particle weight', 0)); _evidence(p, 'particle')
        coords = p['coordinates']; n = len(coords)
        if len({x['name'] for x in coords}) != n:
            raise StudyError('Duplicate transformed coordinate')
        for x in coords:
            name = x['name']
            if name not in UNITS or x.get('unit') != UNITS[name]:
                raise StudyError('Unknown coordinate or unit: ' + name)
            if x.get('transform') not in ('identity', 'exp', 'bounded', 'elevation'):
                raise StudyError('Unknown transform')
            if x['transform'] == 'exp' and _num(x.get('reference'), 'dimensional reference', 0) == 0:
                raise StudyError('Positive dimensional reference required')
            if x['transform'] == 'bounded' and _num(x['upper'], 'upper') <= _num(x['lower'], 'lower'):
                raise StudyError('Invalid bounded transform')
            if '_x_max_' in name and x['transform'] != 'exp':
                raise StudyError('Beta x_max requires exp transform')
            if name.endswith('_a_m_over_a_e') and (x['transform'] != 'bounded' or x.get('lower') != 0 or x.get('upper') != 1):
                raise StudyError('Beta ratio requires bounded transform on (0,1)')
        _cov(p['plant_covariance'], n, tol, 'plant covariance')
        for name, f in p['families'].items():
            if f.get('growth_curve', 'exponential') not in ('exponential','determinate_beta'):
                raise StudyError('Unknown growth_curve')
            _arr(f['mean'], (n,), name + ' mean')
            feats = f.get('features', [])
            if len(set(feats)) != len(feats) or any(k not in FEATURES and not (k.endswith('_missing') and k[:-8] in FEATURES) for k in feats):
                raise StudyError('Invalid conditional feature')
            _arr(f['coefficients'], (n, len(feats)), 'feature coefficients')
            for key in feats:
                if key.endswith('_missing'): continue
                standard = f['standardization'][key]
                _num(standard['center'], 'center')
                if _num(standard['scale'], 'scale', 0) == 0:
                    raise StudyError('Positive feature scale required')
            if abs(_num(f['rho'], 'rho')) >= 1:
                raise StudyError('abs(rho) must be below one')
            _cov(f['innovation_covariance'], n, tol, 'innovation covariance')
            _cov(f['initial_covariance'], n, tol, 'initial covariance')
            for k, v in f.get('constants', {}).items():
                if k not in UNITS or k in {x['name'] for x in coords}:
                    raise StudyError('Unknown/duplicate constant coordinate: ' + k)
                if v.get('unit') != UNITS[k]:
                    raise StudyError('Constant unit mismatch: ' + k)
                _num(v['value'], k, None if k in SIGNED else 0)
                if k in ('wait_dev_days','birth_axis_length_mm') and v['value']==0:
                    raise StudyError(k + ': must be strictly positive')
        for aid, ap in p.get('axes', {}).items():
            if aid not in origin['axes']:
                raise StudyError('Parameter references unknown parent axis')
            stop = ap.get('stop_probability')
            if stop is not None and not 0 <= _num(stop, 'stop probability') <= 1:
                raise StudyError('Stop probability outside [0,1]')
            sl=ap.get('stop_logit')
            if ap.get('node_budget'):
                if stop is not None or sl is not None:
                    raise StudyError('node_budget cannot be combined with stop_probability/stop_logit')
                if origin['axes'][aid].get('base_attachment'):
                    raise StudyError('node_budget supports main axes only')
                _pmf(ap['node_budget'], 'pmf', tol)
            if ap.get('truss_rule') and not _truss_rule_reason(ap['truss_rule']):
                rule = ap['truss_rule']
                if not isinstance(rule['truss_observable'],str): raise StudyError('truss_observable must be a declared label')
                _pmf(rule, 'first_truss_rank_pmf', tol)
                if type(rule.get('interval_leaves')) is not int or rule['interval_leaves'] <= 0:
                    raise StudyError('truss interval_leaves must be a positive integer')
                if not ap.get('node_budget'): raise StudyError('Truss ranks require node_budget and observed node count')
            if sl is not None:
                if stop is not None: raise StudyError('Choose stop probability or conditional logit, not both')
                _num(sl['intercept'],'stop logit intercept')
                for k,v in sl['coefficients'].items():
                    if k not in FEATURES: raise StudyError('Unknown stop feature')
                    _num(v,'stop coefficient')
                    _num(sl['standardization'][k]['center'],'stop feature center')
                    if _num(sl['standardization'][k]['scale'],'stop feature scale',0)==0:
                        raise StudyError('Positive stop feature scale required')
            for key in ('residual_family', 'inter_event_family', 'segment_family'):
                if ap.get(key) is not None and ap[key] not in p['families']:
                    raise StudyError('Unknown axis family')
            for family in ap.get('segment_families', {}).values():
                if family not in p['families']:
                    raise StudyError('Unknown material segment family')
                if p['families'][family].get('growth_curve') == 'determinate_beta':
                    raise StudyError('Existing segments must keep exponential remaining growth')
        for bid, family in p.get('existing_blocks', {}).items():
            if family not in p['families'] or bid not in {b['compound_leaf_id'] for b in origin['blocks']}:
                raise StudyError('Unknown existing-block binding')
            if p['families'][family].get('growth_curve') == 'determinate_beta':
                raise StudyError('Existing blocks must keep exponential remaining growth')
            f = p['families'][family]
            _check_narrowing({k:v['value'] for k,v in f.get('constants',{}).items()},
                             next(b for b in origin['blocks'] if b['compound_leaf_id']==bid),f.get('narrowing_assumption'))
    if sum(weights) <= 0 or not math.isclose(sum(weights), 1, abs_tol=tol):
        raise StudyError('Parameter weights must sum to one')
    return dict(status='unavailable' if missing else 'assumption_conditioned', config=c, unavailable=missing)


def _clock(c: dict, t: float) -> float:
    clock = c.get('clock') or {}
    if clock.get('kind') == 'calendar':
        return t
    if clock.get('kind') not in ('piecewise_constant','piecewise_linear_response'):
        raise StudyError('Unsupported development clock')
    _evidence(clock, 'development clock')
    knots=None
    if clock['kind']=='piecewise_linear_response':
        if clock.get('target')!='development_clock' or clock.get('equation_version')!='piecewise_linear-1':
            raise StudyError('Unsupported response target/equation')
        if not clock.get('input_metric') or not clock.get('input_unit') or not clock.get('applicability'):
            raise StudyError('Response metric, unit and domain evidence required')
        knots=clock.get('knots',[])
        if len(knots)<2: raise StudyError('Response requires at least two knots')
        xx=[_num(r['value'],'response input') for r in knots]
        yy=[_num(r['multiplier'],'response multiplier',0) for r in knots]
        if any(b<=a for a,b in zip(xx,xx[1:])): raise StudyError('Response knots must increase')
    prev = 0.0; total = 0.0
    for interval in clock.get('intervals', []):
        start,end=(_num(interval[k],k,0) for k in ('start_day','end_day'))
        if knots is not None:
            x=_num(interval.get('input_value'),'response input')
            if not xx[0]<=x<=xx[-1]: raise StudyError('Response outside supplied domain')
            rate=float(np.interp(x,xx,yy))
        else: rate=_num(interval['multiplier'],'multiplier',0)
        if start != prev or end <= start:
            raise StudyError('Clock must be contiguous from day zero')
        total += max(0, min(t, end) - start) * rate
        prev = end
    if prev < 30:
        raise StudyError('Clock must cover 30 days')
    return total


def _calendar(c: dict, dev: float) -> float | None:
    if c['clock']['kind'] == 'calendar':
        return dev
    elapsed = 0.0
    for r in c['clock']['intervals']:
        duration=r['end_day']-r['start_day']
        rate=(_clock(c,r['end_day'])-_clock(c,r['start_day']))/duration
        end = elapsed + duration * rate
        if rate > 0 and dev <= end:
            return r['start_day'] + (dev - elapsed) / rate
        elapsed = end
    return None


def _rng(seed: int, *keys) -> np.random.Generator:
    raw = hashlib.sha256(json.dumps([seed, MODEL, *keys], sort_keys=True).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(raw, 'little')))


def _normal(rng, covariance: Any) -> np.ndarray:
    a = np.asarray(covariance, float)
    if len(a) == 0: return np.zeros(0)
    values, vectors = np.linalg.eigh(a)
    return vectors @ (np.sqrt(np.maximum(values, 0)) * rng.standard_normal(len(a)))


def _draw(p: dict, family: str, features: dict, shared: np.ndarray, residual: np.ndarray | None,
          rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    f = p['families'][family]
    vector = []
    for k in f.get('features', []):
        if k.endswith('_missing'):
            vector.append(float(features.get(k[:-8]) is None)); continue
        x = features.get(k)
        if x is None:
            if k + '_missing' not in f['features']:
                raise ScienceUnavailable('Missing conditional feature without indicator: ' + k)
            vector.append(0.0)
        else:
            st = f['standardization'][k]
            vector.append((x - st['center']) / st['scale'])
    residual = (_normal(rng, f['initial_covariance']) if residual is None else
                f['rho'] * residual + _normal(rng, f['innovation_covariance']))
    z = np.array(f['mean']) + np.array(f['coefficients']).reshape(len(p['coordinates']), len(vector)) @ vector + shared + residual
    out = {k: v['value'] for k, v in f.get('constants', {}).items()}
    for x, value in zip(p['coordinates'], z):
        tr = x['transform']
        if tr == 'exp': value = x['reference'] * math.exp(float(value))
        elif tr == 'bounded': value = x['lower'] + (x['upper'] - x['lower']) * expit(value)
        elif tr == 'elevation': value = math.pi / 2 * math.tanh(value)
        out[x['name']] = float(value)
    for k, v in out.items():
        _num(v, k, None if k in SIGNED else 0)
    if f.get('growth_curve') == 'determinate_beta':
        out['growth_curve'] = 'determinate_beta'
        for trait, unit in BETA_TRAITS.items():
            if f'{trait}_x_max_{unit}' in out:
                _beta_value(out, trait, 0., 0.)
    if f.get('narrowing_assumption') is not None:
        out['narrowing_assumption'] = deepcopy(f['narrowing_assumption'])
    return out, residual


def _features(block: dict | None, frame: Any, spacing=None, order=None, env=None) -> dict:
    f = dict(env or {}, spacing_mm=spacing, order_index=order)
    if block:
        f['axis_length_mm'] = block.get('rachis_arc_length_mm')
        for side in ('left', 'right'):
            w = block.get(side + '_extent_mm')
            f[side + '_extent_mm'] = max(w) if w else None
        if frame is not None and block.get('direction_unit_xyz') is not None:
            d = np.array(frame).T @ block['direction_unit_xyz']
            xy = math.hypot(d[0], d[1])
            if xy > observed.MATH_TOL:
                f.update(azimuth_sin=d[1]/xy, azimuth_cos=d[0]/xy)
    return f


def _h(k: float, age: float) -> float:
    return -math.expm1(-k * max(0.0, age))


def _axis_points(tr: dict, aid: str, dev: float) -> list | None:
    a = tr['axes'][aid]
    if a.get('points_mm') is None:
        return None
    base = np.array(a['points_mm'][0], float)
    if a.get('base_attachment'):
        ref = a['base_attachment']
        pp = _axis_points(tr, ref['parent_stem_id'], dev)
        if pp is None: return None
        i = int(ref['segment_index']); f = ref['fraction']
        base = np.array(pp[i]) + f * (np.array(pp[i+1]) - pp[i]) + np.array(ref['offset_mm'])
    pts = [base.tolist()]
    for s in a['segments']:
        if s['born_dev'] > dev: break
        v = s.get('growth')
        if v and v.get('growth_curve') == 'determinate_beta':
            length = _beta_value(v, 'segment', s['length0_mm'], dev-s['born_dev'])
            if length is None: return None
        elif v is None or any(k not in v for k in ('D_segment_mm', 'k_segment')):
            if dev > 0: return None
            length = s['length0_mm']
        else:
            length = s['length0_mm'] + v['D_segment_mm'] * _h(v['k_segment'], dev - s['born_dev'])
        base = base + np.array(s['direction']) * length
        pts.append(base.tolist())
    return pts


def _attachment(tr: dict, b: dict, dev: float) -> list | None:
    ref = b.get('attachment')
    if ref is None: return None
    pp = _axis_points(tr, b['state']['parent_stem_id'], dev)
    if pp is None: return None
    if 'node_index' in ref:
        return pp[ref['node_index']]
    i, f = int(ref['segment_index']), ref['fraction']
    return (np.array(pp[i]) + f * (np.array(pp[i+1])-pp[i]) + np.array(ref['offset_mm'])).tolist()


def _decode_block(tr: dict, block: dict, day: float) -> dict:
    s = deepcopy(block['state']); dev = _clock(tr['config'], day + tr.get('query_offset_days',0))
    s.update(block_id=block['block_id'], source_compound_leaf_id=block.get('source_compound_leaf_id'),
             elapsed_days=day, target_at=(observed.aware_time(tr['origin']['observed_at']) + timedelta(days=day)).isoformat(),
             frame_id=tr['origin']['frame_id'], existence='appeared', state_source=block['state_source'],
             optical_eligible=False, suitable_for_ppfd=False, eligible_state=False, validated_growth=False,
             surface_role='compound_envelope', availability=[])
    if day < block['birth_day']:
        for k in ('rachis_points_mm', 'attachment_xyz_mm', 'direction_unit_xyz', 'rachis_arc_length_mm',
                  'left_extent_mm', 'right_extent_mm', 'supported_area_cm2', 'transverse_unit_xyz'):
            s[k] = None
        s['existence'] = 'not_yet_appeared'
        return s
    if day == 0 and block['state_source'] in ('observed','weighted_observation_anchor'):
        return s
    s['state_source'] = 'predicted_existing' if block['state_source'] in ('observed','weighted_observation_anchor') else 'predicted_future'
    age = max(0, dev - block['born_dev']); v = block.get('growth') or {}
    if v.get('growth_unavailable') and age > 0:
        s['growth_unavailable'] = deepcopy(v['growth_unavailable'])
        s['availability'].extend(sorted(set(v['growth_unavailable'].values())))
    limit = tr['config'].get('max_leaf_age_dev_days')
    if limit is not None and block.get('organ_born_dev') is not None and dev-block['organ_born_dev'] > limit:
        s['availability'].append('retention_assumption_exceeded')
        s['retention_assumption_exceeded'] = True
    a = _attachment(tr, block, dev)
    s['attachment_xyz_mm'] = a
    if a is None: s['availability'].append('attachment_growth_unavailable')
    length = s.get('rachis_arc_length_mm')
    if length is not None and v.get('growth_curve') == 'determinate_beta':
        length = _beta_value(v, 'length', length, age)
        if length is None: s['availability'].append('length_growth_unavailable')
    elif length is not None and all(k in v for k in ('D_length_mm', 'k_length')):
        length += v['D_length_mm'] * _h(v['k_length'], age)
    elif age > 0:
        length = None
        s['availability'].append('length_growth_unavailable')
    s['rachis_arc_length_mm'] = length
    rotation = None
    rk = ('rotation_x_rad', 'rotation_y_rad', 'rotation_z_rad', 'k_rotation')
    if age == 0:
        rotation = np.eye(3)
    elif all(k in v for k in rk) and block.get('frame') is not None:
        q = np.array(block['frame'])
        rot = Rotation.from_rotvec(np.array([v[k] for k in rk[:3]]) * _h(v['k_rotation'], age)).as_matrix()
        rotation = q @ rot @ q.T
    if rotation is not None and s.get('direction_unit_xyz') is not None:
        s['direction_unit_xyz'] = (rotation @ s['direction_unit_xyz']).tolist()
        if s.get('transverse_unit_xyz') is not None:
            s['transverse_unit_xyz'] = (rotation @ s['transverse_unit_xyz']).tolist()
    else:
        s['direction_unit_xyz'] = None; s['transverse_unit_xyz'] = None
        s['availability'].append('direction_growth_unavailable')
    points = block['state'].get('rachis_points_mm')
    if a is not None and length is not None and points is not None and rotation is not None:
        old = np.array(points)
        s['rachis_points_mm'] = (np.array(a) + ((old-old[0]) @ rotation.T) *
                                 (length / block['state']['rachis_arc_length_mm'])).tolist()
    else:
        s['rachis_points_mm'] = None
    for side in ('left', 'right'):
        widths = block['state'].get(side + '_extent_mm')
        if age == 0: continue
        if side+'_extent_mm' in v.get('growth_unavailable',{}):
            s[side+'_extent_mm'] = None
            continue
        if widths is not None and v.get('growth_curve') == 'determinate_beta':
            values = [_beta_value(v, side, w, age) for w in widths]
            s[side + '_extent_mm'] = values if all(x is not None for x in values) else None
        elif widths is not None and side + '_target_mm' in v and 'k_width' in v:
            s[side + '_extent_mm'] = [w + (v[side + '_target_mm']-w) * v.get('size_target_factor',1) * _h(v['k_width'], age) for w in widths]
        else:
            s[side + '_extent_mm'] = None
            s['availability'].append(side + '_extent_growth_unavailable')
    area = block['state'].get('supported_area_cm2')
    if age > 0:
        if area is not None and block['state'].get('area_basis_id') and v.get('growth_curve') == 'determinate_beta':
            s['supported_area_cm2'] = _beta_value(v, 'area', area, age)
        elif area is not None and block['state'].get('area_basis_id') and all(k in v for k in ('D_area_cm2', 'k_area')):
            s['supported_area_cm2'] = area + v['D_area_cm2'] * _h(v['k_area'], age)
        else:
            s['supported_area_cm2'] = None
        s['estimated_total_area_cm2'] = None
    return s


@_input_contract
def decode_trajectory(trajectory, elapsed_days) -> list[dict]:
    """Pure analytic query: no random generator, fitting or source modification."""
    tr = _obj(trajectory)
    times = elapsed_days if isinstance(elapsed_days, (list, tuple)) else [elapsed_days]
    times = [_num(t, 'elapsed_days', 0) for t in times]
    if any(t > tr.get('available_horizon_days',30) for t in times) or len(set(times)) != len(times):
        raise StudyError('Unique query times within [0,30] required')
    if not tr['valid']: return []
    return [dict(_decode_block(tr, b, t), trajectory_id=tr['trajectory_id'])
            for t in sorted(times) for b in sorted(tr['blocks'], key=lambda b: b['block_id'])]


def _new_block(tr, aid, event, values, frame, predecessor, ap):
    length = values.get('birth_axis_length_mm')
    if length is not None and length <= 0:
        raise StudyError('Birth axis must be positive')
    a = tr['axes'][aid]
    node_index = len(a['segments'])
    attachment = dict(node_index=node_index) if a.get('apex_available') and a.get('points_mm') else None
    temp = dict(state={'parent_stem_id': aid}, attachment=attachment)
    xyz = _attachment(tr, temp, event['appearance_development_days'])
    direction = None
    if frame is not None and all(k in values for k in ('azimuth_rad', 'elevation_rad')):
        alpha, beta = values['azimuth_rad'], values['elevation_rad']
        if not -math.pi/2 <= beta <= math.pi/2:
            raise StudyError('Elevation outside signed range')
        mode = ap.get('direction_reference')
        if mode == 'predecessor':
            f = _features(predecessor, frame)
            if 'azimuth_sin' not in f:
                alpha = None
            else:
                alpha += math.atan2(f['azimuth_sin'], f['azimuth_cos'])
        elif mode != 'absolute':
            alpha = None
        if alpha is not None:
            direction = (np.array(frame) @ [math.cos(beta)*math.cos(alpha), math.cos(beta)*math.sin(alpha), math.sin(beta)]).tolist()
    points = [xyz, (np.array(xyz)+length*np.array(direction)).tolist()] if xyz is not None and length is not None and direction is not None else None
    state = dict(plant_id=tr['origin']['plant_id'], parent_stem_id=aid, compound_leaf_id=None,
                 attachment_xyz_mm=xyz, direction_unit_xyz=direction, rachis_points_mm=points,
                 rachis_arc_length_mm=length, axis_stations=[0,1] if points else [], extent_stations=[],
                 left_extent_mm=[values['birth_left_extent_mm']] if 'birth_left_extent_mm' in values else None,
                 right_extent_mm=[values['birth_right_extent_mm']] if 'birth_right_extent_mm' in values else None,
                 supported_area_cm2=values.get('birth_area_cm2') if ap.get('area_basis_id') else None,
                 area_basis_id=ap.get('area_basis_id'), estimated_total_area_cm2=None, transverse_unit_xyz=None)
    if values.get('growth_curve') == 'determinate_beta':
        births = dict(length=length, left=values.get('birth_left_extent_mm'), right=values.get('birth_right_extent_mm'),
                      segment=0., area=state['supported_area_cm2'])
        for trait, birth in births.items():
            if birth is not None: _beta_value(values, trait, birth, 0.)
    _check_curve_availability(values,state,tr,event['predicted_block_id'])
    result = dict(block_id=event['predicted_block_id'], source_compound_leaf_id=None, state=state,
                state_source='predicted_future', birth_day=event['appearance_elapsed_days'],
                born_dev=event['appearance_development_days'], growth=values, frame=frame, attachment=attachment)
    if tr['config'].get('max_leaf_age_dev_days') is not None:
        result['organ_born_dev'] = event['appearance_development_days']
    return result


def _trajectory(origin, c, index):
    seed = c['computation']['seed']; tid = f'trajectory_{index:06d}'
    particles = sorted(c['parameter_particles'], key=lambda p: p['id'])
    p = particles[int(_rng(seed, index, 'particle').choice(len(particles), p=[x['weight'] for x in particles]))]
    shared = _normal(_rng(seed,index,'plant'), p['plant_covariance'])
    tr = dict(trajectory_id=tid, parameter_particle_id=p['id'], scenario_id=(c.get('environment_scenario') or {}).get('id'),
              valid=True, status_reason=None, random_stream_id=f'PCG64/SHA256/{seed}/{index}',
              weight=1/c['computation']['sample_count'], origin=deepcopy(origin), config=deepcopy(c),
              axes=deepcopy(origin['axes']), blocks=[], events=[], availability=[], parameter=p)
    tr['n_max_drawn'] = {}
    env = (c.get('environment_scenario') or {}).get('values') or {}
    for aid, a in sorted(tr['axes'].items()):
        ap = p.get('axes', {}).get(aid, {})
        for s in a['segments']:
            fam = ap.get('segment_families', {}).get(s['segment_id'])
            s['growth'] = _draw(p,fam,env,shared,None,_rng(seed,index,aid,s['segment_id'],'segment'))[0] if fam else None
            s['growth'] = _size_response(s['growth'], c)
    for s in origin['blocks']:
        if not s['eligible_state']:
            tr['availability'].append(_reason(s['compound_leaf_id'],'ineligible_observed_state')); continue
        bid, aid = s['compound_leaf_id'], s['parent_stem_id']
        ref = origin['attachments'].get(bid); a = tr['axes'][aid]
        frame = a['frames'][int(ref['segment_index'])] if ref and a.get('frames') else None
        fam = p.get('existing_blocks', {}).get(bid)
        val = _draw(p,fam,_features(s,frame,env=env),shared,None,_rng(seed,index,bid,'existing'))[0] if fam else None
        val = _size_response(val, c)
        if val: _check_narrowing(val,s,val.get('narrowing_assumption'))
        _check_curve_availability(val,s,tr,bid)
        tr['blocks'].append(dict(block_id=bid, source_compound_leaf_id=bid, state=deepcopy(s), state_source='observed',
                                  birth_day=0.0, born_dev=0.0, growth=val, frame=frame, attachment=ref))
    for aid, a in sorted(tr['axes'].items()):
        ap = p.get('axes', {}).get(aid, {})
        apex = a['apex_status']
        if apex == 'unknown': apex = ap.get('assumed_apex_status', 'unknown')
        budget = ap.get('node_budget'); n_max = None; rank = None; first_truss = None
        if budget:
            rank = _observed_count(a)
            ns, ps = _pmf(budget, 'pmf', c['computation']['numeric_tolerance'])
            pairs = [(n, prob) for n,prob in zip(ns,ps) if n >= rank and prob > 0]
            if not pairs: raise ScienceUnavailable('node_budget_no_mass_at_or_above_observed_count')
            total = sum(prob for _,prob in pairs)
            n_max = int(_rng(seed,index,aid,'node_budget').choice([n for n,_ in pairs],p=[prob/total for _,prob in pairs]))
            tr['n_max_drawn'][aid] = n_max
            if apex == 'stopped' and n_max != rank:
                raise ScienceUnavailable('observed_stopped_apex_incompatible_with_node_budget')
            if ap.get('truss_rule'):
                truss_reason=_truss_rule_reason(ap['truss_rule'])
                if truss_reason:
                    tr['availability'].append(_reason(aid,truss_reason))
                else:
                    ns, ps = _pmf(ap['truss_rule'], 'first_truss_rank_pmf', c['computation']['numeric_tolerance'])
                    first_truss = int(_rng(seed,index,aid,'first_truss').choice(ns,p=ps))
                    a['first_truss_rank_drawn'] = first_truss
        if not (n_max is not None and n_max == rank) and (apex == 'unknown' or (not budget and ap.get('stop_probability') is None and ap.get('stop_logit') is None) or not ap.get('residual_family') or not ap.get('inter_event_family')):
            if apex != 'stopped':
                tr['availability'].append(_reason(aid, 'event_parameters_or_apex_unavailable')); continue
        current = 0.0; residual = None; predecessor = None
        spacing=None; order=None
        pred_id = ap.get('predecessor_id')
        if pred_id is not None:
            predecessor = next((b['state'] for b in tr['blocks'] if b['block_id'] == pred_id and b['state']['parent_stem_id'] == aid), None)
            if predecessor is None or origin['attachments'].get(pred_id,{}).get('adjacency_status') != 'confirmed':
                predecessor=None
                tr['availability'].append(_reason(aid,'predecessor_relation_unavailable'))
            elif predecessor is not None:
                ref=origin['attachments'][pred_id]
                if ref.get('order_status')=='confirmed' and ref.get('order_index') not in (None,''):
                    order=_num(ref['order_index'],'reviewed order')
                previous=origin['attachments'].get(ref.get('predecessor_id'))
                if previous and previous['parent_stem_id']==aid:
                    def along(r):
                        return sum(s['length0_mm'] for s in a['segments'][:int(r['segment_index'])])+r['fraction']*a['segments'][int(r['segment_index'])]['length0_mm']
                    spacing=along(ref)-along(previous)
                    if spacing<=0: raise StudyError('Confirmed predecessor order contradicts stem geometry')
        for event_index in range(c['computation']['max_events_per_axis']):
            event = dict(trajectory_id=tid,parent_stem_id=aid,predicted_block_id=f'{tid}/{aid}/new_{event_index+1}',
                         event_index=event_index+1,event_state=None,appearance_elapsed_days=None,
                         appearance_development_days=None,material_node_id=None,event_kind='leaf',node_rank=rank+1 if rank is not None else None)
            if n_max is not None and rank == n_max:
                tr['events'].append(dict(event,event_kind='terminal',node_rank=rank,event_state='appeared',
                    appearance_elapsed_days=_calendar(c,current),appearance_development_days=current,predicted_block_id=None,
                    geometry=None,optical_eligible=False,suitable_for_ppfd=False))
                break
            rng = _rng(seed,index,aid,event_index,'event')
            frame = a['frames'][-1] if a.get('frames') else None
            feature = _features(predecessor,frame,spacing=spacing,order=order,env=env)
            stop=ap.get('stop_probability')
            if apex!='stopped' and ap.get('stop_logit') is not None:
                sl=ap['stop_logit']; logit=sl['intercept']
                for k,coef in sl['coefficients'].items():
                    if feature.get(k) is None: raise ScienceUnavailable('Conditional stop feature unavailable: '+k)
                    st=sl['standardization'][k]; logit+=coef*(feature[k]-st['center'])/st['scale']
                stop=float(expit(logit))
            if apex == 'stopped' or (n_max is None and rng.random() < stop):
                event.update(event_state='stopped', predicted_block_id=None); tr['events'].append(event); break
            family = ap['residual_family'] if event_index == 0 else ap['inter_event_family']
            values, residual = _draw(p,family,feature,shared,residual,rng)
            values = _size_response(values, c)
            wait = values.get('wait_dev_days')
            if wait is None:
                tr['availability'].append(_reason(aid,'waiting_time_unavailable')); break
            if wait <= 0 or current + wait == current:
                raise StudyError('Nonpositive/underflowed event wait')
            current += wait
            day = _calendar(c,current)
            event.update(appearance_development_days=current, appearance_elapsed_days=day)
            if day is None or day > 30:
                event['event_state']='beyond_horizon'; tr['events'].append(event); break
            event.update(event_state='appeared',material_node_id=event['predicted_block_id']+'/node')
            tr['events'].append(event)
            b = _new_block(tr,aid,event,values,frame,predecessor,ap)
            tr['blocks'].append(b); predecessor = b['state']
            if rank is not None:
                rank += 1
                if first_truss is not None and rank >= first_truss and (rank-first_truss) % ap['truss_rule']['interval_leaves'] == 0:
                    tr['events'].append(dict(event,event_kind='truss',predicted_block_id=None,material_node_id=None,
                        geometry=None,optical_eligible=False,suitable_for_ppfd=False,truss_observable=ap['truss_rule']['truss_observable']))
            order=order+1 if order is not None else None
            spacing=None
            if a['segments']:
                last=a['segments'][-1]; growth=last.get('growth') or {}
                if all(k in growth for k in ('D_segment_mm','k_segment')):
                    spacing=last['length0_mm']+growth['D_segment_mm']*_h(growth['k_segment'],current-last['born_dev'])
            if a.get('points_mm') and a.get('apex_available'):
                a['segments'].append(dict(segment_id=event['material_node_id']+':terminal',length0_mm=0.0,
                    direction=a['segments'][-1]['direction'],born_dev=current,growth=values))
            if n_max is not None and rank == n_max:
                tr['events'].append(dict(event,event_kind='terminal',predicted_block_id=None,material_node_id=None,
                    geometry=None,optical_eligible=False,suitable_for_ppfd=False))
                break
        else:
            raise StudyError('Event resource limit reached; trajectory is not truncated as valid')
    tr['events'].sort(key=lambda e: (e['appearance_elapsed_days'] if e['appearance_elapsed_days'] is not None else math.inf,e['parent_stem_id'],e['event_index']))
    for evidence in origin.get('event_history', []):
        if evidence['visibility'] != 'confirmed': continue
        aid = evidence['parent_stem_id']; n = tr['n_max_drawn'].get(aid)
        if n is None: continue
        if evidence['event_kind']=='truss':
            reason=_truss_evidence_reason(tr,evidence)
            if reason:
                tr['availability'].append(_reason(aid,reason)); continue
        rank = evidence['node_rank']
        if n < rank or (evidence['event_kind']=='terminal' and n != rank):
            tr['weight'] = 0.
        if evidence['event_kind']=='truss':
            first = tr['axes'][aid].get('first_truss_rank_drawn'); rule = p['axes'][aid].get('truss_rule')
            if first is None:
                tr['availability'].append(_reason(aid,'observed_truss_rule_unavailable')); continue
            if rank < first or (rank-first) % rule['interval_leaves']: tr['weight'] = 0.
    return tr


@_input_contract
def sample_forecast(origin, config) -> dict:
    origin = _obj(origin)
    for state in origin['blocks']:
        audit_state(state)
    check = validate_model_config(config, origin); c = check['config']
    bank = dict(schema_version=SCHEMA, model_version=MODEL, origin=origin, config=c,
                status=check['status'], availability=check['unavailable'], trajectories=[],
                environment_used=c['clock']['kind']=='piecewise_linear_response' and c['clock'].get('input_metric') in ENV_FIELDS,
                light_used=c['clock']['kind']=='piecewise_linear_response' and c['clock'].get('input_metric') not in ENV_FIELDS,
                environment_coupling='explicit_response' if c['clock']['kind']=='piecewise_linear_response' and c['clock'].get('input_metric') in ENV_FIELDS else 'unavailable',
                light_coupling='fixed_external_response' if c['clock']['kind']=='piecewise_linear_response' and c['clock'].get('input_metric') not in ENV_FIELDS else 'unavailable',
                validated_growth=False, suitable_for_ppfd=False,
                uncertainty={'process':'assumption_conditioned','parameters':'supplied_particles',
                             'measurement':'unquantified','model_form':'unquantified'})
    bank.update(environment_audit(c))
    bank['environment_coupling'] = 'explicit_response' if bank['environment_used'] else 'unavailable'
    bank['light_coupling'] = ('size_target_dli' if c.get('light_response') else 'frozen_config_response') if bank['light_used'] else 'unavailable'
    bank['run_id'] = digest(dict(origin_id=origin['origin_id'],config=c))
    if check['status'] != 'unavailable':
        for i in range(c['computation']['sample_count']):
            try:
                bank['trajectories'].append(_trajectory(origin,c,i))
            except (StudyError, OverflowError, FloatingPointError) as exc:
                bank['trajectories'].append(dict(trajectory_id=f'trajectory_{i:06d}',parameter_particle_id=None,
                     scenario_id=(c.get('environment_scenario') or {}).get('id'),valid=False,status_reason=str(exc),
                     failure_kind='scientific_unavailability' if isinstance(exc,ScienceUnavailable) else 'sample_failure',
                     weight=1/c['computation']['sample_count'],random_stream_id=f'PCG64/SHA256/{c["computation"]["seed"]}/{i}',
                     events=[],blocks=[]))
        if not any(t['valid'] for t in bank['trajectories']):
            bank['status']='unavailable' if all(t['failure_kind']=='scientific_unavailability' for t in bank['trajectories']) else 'sample_failure'
        if origin.get('event_history'):
            mass = sum(t['weight'] for t in bank['trajectories'])
            if mass == 0:
                bank['status'] = 'unavailable'; bank['availability'].append(_reason('events','origin_events_incompatible_with_finite_bank'))
            else:
                for t in bank['trajectories']: t['weight'] /= mass
    observed._unchanged(origin.get('source_hashes',{}))
    return bank


def _quantile(values, weights, q):
    order = np.argsort(values); x = np.asarray(values)[order]; w = np.asarray(weights)[order]
    return float(x[min(int(np.searchsorted(np.cumsum(w)/w.sum(),q)),len(x)-1)])


@_input_contract
def summarize_forecast(bank, summary_config) -> list[dict]:
    b = _obj(bank); cfg = _obj(summary_config); out=[]; audit_decoded={}
    quantiles = cfg.get('quantiles')
    if quantiles is None or any(not 0 <= _num(q,'quantile') <= 1 for q in quantiles):
        raise StudyError('Explicit summary quantiles required')
    valid = [t for t in b['trajectories'] if t['valid'] and t['weight'] > 0]
    mass = sum(t['weight'] for t in valid)
    for day in cfg.get('output_days',b['config']['computation']['output_days']):
        decoded = {t['trajectory_id']:decode_trajectory(t,[day]) for t in valid}
        if b['config'].get('light_response'): audit_decoded[day] = decoded
        base = dict(scenario_id=(b['config'].get('environment_scenario') or {}).get('id'), horizon_days=day,
                    nominal_probability=None,available_weight=mass,reason=None if mass == 1 else 'conditional_on_valid_samples')
        # Counts are only eligible if appearance parameters were available on every supplied axis.
        terminated = any(_terminal_by(t,day) for t in valid)
        count_pairs=[]
        for t in valid:
            if terminated: continue
            if any(r['reason_code'] in ('event_parameters_or_apex_unavailable','waiting_time_unavailable') for r in t.get('availability',[])): continue
            count_pairs.append((sum(e.get('event_kind','leaf')=='leaf' and e['event_state']=='appeared' and 0<e['appearance_elapsed_days']<=day for e in t['events']),t['weight']))
        cmass=sum(w for _,w in count_pairs)
        for count in sorted({n for n,_ in count_pairs}):
            out.append(dict(base,target_kind='plant',target_id=b['origin']['plant_id'],quantity='new_block_count',
                            conditioning='valid_event_models',statistic='probability',value=sum(w for n,w in count_pairs if n==count)/cmass,
                            unit='probability',available_weight=cmass,count=count))
        if not count_pairs:
            out.append(dict(base,target_kind='plant',target_id=b['origin']['plant_id'],quantity='new_block_count',conditioning='unconditional',
                            statistic='unavailable',value=None,unit='count',available_weight=0,
                            reason='sympodial_continuation_not_modeled' if terminated else 'event_distribution_unavailable',count=None,
                            **({'post_termination_scope':'unavailable'} if terminated else {})))
        if terminated:
            out.append(dict(base,target_kind='plant',target_id=b['origin']['plant_id'],quantity='post_termination_scope',
                conditioning='valid_samples',statistic='unavailable',value=None,unit=None,count=None,
                post_termination_scope='unavailable',reason='sympodial_continuation_not_modeled'))
        for aid in b['origin']['axes']:
            budgeted = [t for t in valid if aid in t.get('n_max_drawn', {})]
            if budgeted:
                pairs = [(sum(e.get('event_kind','leaf')=='leaf' and e['parent_stem_id']==aid and e['event_state']=='appeared'
                              and 0<e['appearance_elapsed_days']<=day for e in t['events']),t['weight']) for t in budgeted]
                axis_terminated = any(_terminal_by(t,day,aid) for t in budgeted)
                for count in sorted({n for n,_ in pairs}):
                    out.append(dict(base,target_kind='axis',target_id=aid,quantity='main_axis_new_leaf_count',
                        conditioning='valid_event_models',statistic='probability',value=sum(w for n,w in pairs if n==count)/sum(w for _,w in pairs),
                        unit='probability',count=count,available_weight=sum(w for _,w in pairs),
                        post_termination_scope='unavailable' if axis_terminated else None,
                        reason='sympodial_continuation_not_modeled' if axis_terminated else None))
            truss_pairs = [(sum(e.get('event_kind')=='truss' and e['parent_stem_id']==aid and e['event_state']=='appeared'
                                and 0<e['appearance_elapsed_days']<=day for e in t['events']),t['weight']) for t in valid
                           if not _truss_rule_reason(t.get('parameter',{}).get('axes',{}).get(aid,{}).get('truss_rule') or {})]
            if truss_pairs:
                for count in sorted({n for n,_ in truss_pairs}):
                    out.append(dict(base,target_kind='axis',target_id=aid,quantity='new_truss_count',conditioning='declared_truss_rule',
                        statistic='probability',value=sum(w for n,w in truss_pairs if n==count)/sum(w for _,w in truss_pairs),
                        unit='probability',count=count,available_weight=sum(w for _,w in truss_pairs)))
            else:
                out.append(dict(base,target_kind='axis',target_id=aid,quantity='new_truss_count',conditioning='unconditional',
                    statistic='unavailable',value=None,unit='count',count=None,available_weight=0,reason='truss_rule_unavailable'))
        keys=set()
        for ss in decoded.values():
            for s in ss:
                keys.add(s['source_compound_leaf_id'] or '/'.join(s['block_id'].split('/')[1:]))
        for key in sorted(keys):
            matches=[]
            for t in valid:
                s=next((s for s in decoded[t['trajectory_id']] if (s['source_compound_leaf_id'] or '/'.join(s['block_id'].split('/')[1:]))==key and s['existence']=='appeared'),None)
                if s: matches.append((s,t['weight']))
            exists=sum(w for _,w in matches)
            out.append(dict(base,target_kind='block',target_id=key,quantity='occurrence',conditioning='valid_samples',statistic='probability',
                            value=exists/mass if mass else None,unit='probability',count=None))
            for field,unit in (('rachis_arc_length_mm','mm'),('left_extent_mm','mm'),('right_extent_mm','mm'),('supported_area_cm2','cm2')):
                pairs=[(max(s[field]) if isinstance(s.get(field),list) else s[field],w) for s,w in matches if s.get(field) is not None]
                for q in quantiles:
                    out.append(dict(base,target_kind='block',target_id=key,quantity=field,conditioning='exists_and_trait_available',
                         statistic='quantile',value=_quantile([x for x,_ in pairs],[w for _,w in pairs],q) if pairs else None,
                         unit=unit,nominal_probability=q,available_weight=sum(w for _,w in pairs),reason=None if pairs else 'trait_unavailable',count=None))
            for axis in 'xyz':
                pairs=[(s['attachment_xyz_mm']['xyz'.index(axis)],w) for s,w in matches if s.get('attachment_xyz_mm') is not None]
                for q in quantiles:
                    out.append(dict(base,target_kind='block',target_id=key,quantity=f'attachment_{axis}_mm',conditioning='exists_and_trait_available',
                         statistic='quantile',value=_quantile([x for x,_ in pairs],[w for _,w in pairs],q) if pairs else None,
                         unit='mm',nominal_probability=q,available_weight=sum(w for _,w in pairs),reason=None if pairs else 'attachment_unavailable',count=None))
            directions=[(s['direction_unit_xyz'],w) for s,w in matches if s.get('direction_unit_xyz') is not None]
            dm=sum(w for _,w in directions)
            resultant=sum((w*np.array(d) for d,w in directions),start=np.zeros(3))/dm if dm else np.zeros(3)
            strength=float(np.linalg.norm(resultant))
            out.append(dict(base,target_kind='block',target_id=key,quantity='direction_resultant_length',conditioning='exists_and_trait_available',
                            statistic='concentration',value=strength if dm else None,unit='dimensionless',available_weight=dm,
                            reason=None if dm else 'direction_unavailable',count=None))
    if b['config'].get('light_response'):
        ensemble_weight = sum(t['weight'] for t in b['trajectories'])
        for row in out:
            available = row['available_weight']
            row.update(ensemble_weight=ensemble_weight,
                available_fraction_of_ensemble=available/ensemble_weight if ensemble_weight else None,
                unavailable_weight=max(0.,ensemble_weight-available))
            reasons = {}
            if row['target_kind']=='block':
                for tr in valid:
                    state = _find_state(audit_decoded[row['horizon_days']][tr['trajectory_id']],row['target_id'])
                    reason = (state or {}).get('growth_unavailable',{}).get(row['quantity'])
                    if reason: reasons[reason] = reasons.get(reason,0.) + tr['weight']
            row['unavailable_reasons'] = [dict(reason_code=k,weight=v,
                probability_of_ensemble=v/ensemble_weight if ensemble_weight else None) for k,v in sorted(reasons.items())]
            if reasons:
                row['reason'] = 'conditional_on_available_curves' if available else '|'.join(sorted(reasons))
    _summary_support(b,out)
    for row in out:
        if row['quantity']=='new_truss_count':
            labels={(t.get('parameter',{}).get('axes',{}).get(row['target_id'],{}).get('truss_rule') or {}).get('truss_observable') for t in valid}
            labels.discard(None)
            row['truss_observable']=next(iter(labels)) if len(labels)==1 else None
            if len(labels)>1: row.update(value=None,reason='truss_observable_mixture',statistic='unavailable')
    return out


def _summary_support(bank: dict, records: list[dict]) -> None:
    """Masses use the original bank weights; computational support is independent of presence."""
    total=sum(t['weight'] for t in bank['trajectories'])
    cache={}
    for row in records:
        existence=computable=available=not_present=unknown_existence=0.; reasons={}
        day=row['horizon_days']; field=row['quantity']
        for tr in bank['trajectories']:
            weight=tr['weight']; present=False; computed=False; reason=None; existence_known=tr['valid']
            if not tr['valid']:
                reason=tr.get('status_reason','invalid_trajectory')
            elif row['target_kind']=='block':
                cache_key=(tr['trajectory_id'],day)
                if cache_key not in cache: cache[cache_key]=decode_trajectory(tr,[day])
                state=_find_state(cache[cache_key],row['target_id'])
                present=bool(state and state['existence']=='appeared')
                if state is None and any(x['reason_code'] in ('waiting_time_unavailable','event_parameters_or_apex_unavailable') for x in tr.get('availability',[])):
                    existence_known=False
                block=next((x for x in tr['blocks'] if _find_state([dict(x,state_source=x['state_source'])],row['target_id'])),None)
                if field=='occurrence': computed=not any(x['reason_code'] in ('waiting_time_unavailable','event_parameters_or_apex_unavailable') for x in tr.get('availability',[]))
                elif block is None:
                    # Biological nonappearance in a complete event model is not a failed size model.
                    computed=not any(x['reason_code'] in ('waiting_time_unavailable','event_parameters_or_apex_unavailable') for x in tr.get('availability',[]))
                else:
                    query=state if present else _decode_block(tr,block,min(tr.get('available_horizon_days',30),block['birth_day']+1e-8))
                    reason=(block.get('growth') or {}).get('growth_unavailable',{}).get(field)
                    if present: reason=(state or {}).get('growth_unavailable',{}).get(field)
                    key='direction_unit_xyz' if field=='direction_resultant_length' else field
                    computed=reason is None and _scalar(query,key) is not None
            else:
                aid=row['target_id'] if row['target_kind']=='axis' else None
                present=not (field in ('new_block_count','post_termination_scope') and _terminal_by(tr,day,aid))
                computed=not any(x['reason_code'] in ('waiting_time_unavailable','event_parameters_or_apex_unavailable') and
                    (aid is None or x['component']==aid) for x in tr.get('availability',[]))
                if field=='new_truss_count':
                    computed=computed and _truss_rule_reason(tr.get('parameter',{}).get('axes',{}).get(aid,{}).get('truss_rule') or {}) is None
                    if not computed: reason='truss_observable_or_rule_unavailable'
            if present: existence+=weight
            elif existence_known: not_present+=weight
            else: unknown_existence+=weight
            if computed: computable+=weight
            if present and computed: available+=weight
            if not computed:
                reason=reason or 'trait_or_parameter_unavailable'
                reasons[reason]=reasons.get(reason,0.)+weight
        # Occurrence probability is evaluated for absent as well as present states;
        # its available_mass describes presence+trait support, not Bernoulli trials.
        row.update(ensemble_weight=total,existence_mass=existence,computable_mass=computable,available_mass=available,
            not_present_mass=not_present,existence_unknown_mass=unknown_existence,computational_unavailable_mass=max(0.,total-computable),
            support_denominator='original_bank_weight_sum',available_fraction_of_ensemble=available/total if total else None,
            computational_reasons=[dict(reason_code=k,weight=v) for k,v in sorted(reasons.items())])


def _find_state(states, key):
    return next((s for s in states if s.get('block_id') == key or s.get('source_compound_leaf_id') == key
                 or '/'.join(s.get('block_id','').split('/')[1:]) == key),None)


def _scalar(state, field):
    if field.startswith('attachment_') and field.endswith('_mm'):
        v=state.get('attachment_xyz_mm'); axis=field[len('attachment_'):-3]
        return v['xyz'.index(axis)] if v is not None and axis in 'xyz' and len(axis)==1 else None
    v=state.get(field)
    return max(v) if isinstance(v,list) and v else v


def _discrepancy(pred, target, fields):
    if pred is None or pred.get('existence')!='appeared': return None
    residual=[]
    for field in fields:
        semantic_field = 'direction_unit_xyz' if field.startswith('direction_tangent_') else field
        if semantic_field in ('direction_unit_xyz', 'supported_area_cm2') and comparison_reason(pred, target, semantic_field):
            return None
        if field in ('direction_tangent_1_rad','direction_tangent_2_rad'):
            a,b=pred.get('direction_unit_xyz'),target.get('direction_unit_xyz')
            if a is None or b is None: return None
            a,b=np.array(a),np.array(b); cosine=float(np.clip(a@b,-1,1))
            if cosine <= -1+observed.MATH_TOL: return None
            # Deterministic observed tangent chart; not a change to measured geometry.
            reference=np.eye(3)[int(np.argmin(abs(b)))]; u=reference-(reference@b)*b; u/=np.linalg.norm(u)
            v=np.cross(b,u); delta=a-cosine*b; sine=np.linalg.norm(delta)
            tangent=delta*(math.acos(cosine)/sine) if sine>observed.MATH_TOL else np.zeros(3)
            residual.append(float(tangent@(u if field.endswith('1_rad') else v)))
        else:
            x,y=_scalar(pred,field),_scalar(target,field)
            if x is None or y is None: return None
            if field=='supported_area_cm2' and pred.get('area_basis_id')!=target.get('area_basis_id'): return None
            residual.append(float(x-y))
    return np.array(residual)


def _remaining(block, dev, state):
    b=deepcopy(block); v=deepcopy(b.get('growth') or {}); age=max(0,dev-b['born_dev'])
    if v.get('growth_curve') == 'determinate_beta':
        old = b['state']
        v.setdefault('beta_birth_values', dict(length=old.get('rachis_arc_length_mm'),
            left=(old.get('left_extent_mm') or [None])[0], right=(old.get('right_extent_mm') or [None])[0],
            area=old.get('supported_area_cm2')))
        v['beta_anchor_age'] = v.get('beta_anchor_age',0.) + age
    for delta,rate in (('D_length_mm','k_length'),('D_area_cm2','k_area')):
        if delta in v and rate in v: v[delta]*=math.exp(-v[rate]*age)
    for side in ('left','right'):
        k=side+'_target_mm'; old=b['state'].get(side+'_extent_mm'); new=state.get(side+'_extent_mm')
        if k in v and 'k_width' in v and old is not None and new is not None:
            # A scalar target can represent a profile update only when residual targets agree.
            if len(old)!=len(new): raise StudyError('Extent station count changed during re-anchor')
            targets=[n+(v[k]-o)*math.exp(-v['k_width']*age) for o,n in zip(old,new)]
            if min(targets)<0: raise StudyError('Negative rebased width target')
            if not np.allclose(targets,targets[0],atol=observed.MATH_TOL,rtol=0):
                v.pop(k)  # Keep state; this scalar decoder cannot resolve changing profile targets.
            else: v[k]=targets[0]
    if 'k_rotation' in v:
        for k in ('rotation_x_rad','rotation_y_rad','rotation_z_rad'):
            if k in v: v[k]*=math.exp(-v['k_rotation']*age)
    b.update(state=deepcopy(state),growth=v,born_dev=dev,birth_day=0.0)
    return b


def _ess_from_log_weights(log_weights: list[float]) -> float:
    normalizer = float(logsumexp(log_weights)) if log_weights else -math.inf
    if not math.isfinite(normalizer): return 0.
    weights = np.exp(np.array(log_weights)-normalizer)
    return float(1/np.sum(weights*weights))


def _conditioning_diagnostics(prior: list[float], geometry: list[float], joint: list[float],
                              unavailable: list[str], model_present: bool, rejected: list[str]) -> dict:
    initial = _ess_from_log_weights(prior)
    geometry_ok = model_present and not any(x!='event_particle_unavailable' and not str(x).startswith('truss_') for x in unavailable)
    joint_ok = geometry_ok and not unavailable
    after_geometry = _ess_from_log_weights(geometry) if geometry_ok else None
    after_events = _ess_from_log_weights(joint) if joint_ok else None
    return dict(conditioning_order=['geometry','events'],prior_ess=initial,
        event_stage='hard_admissibility',n_particles_rejected_by_events=len(set(rejected)),
        after_geometry_ess=after_geometry,after_geometry_then_events_ess=after_events,
        geometry_ess_loss=initial-after_geometry if after_geometry is not None else None,
        event_stage_ess_loss=after_geometry-after_events if after_events is not None else None,
        geometry_stage_status='evaluated' if geometry_ok else 'unavailable',
        event_stage_status='evaluated' if joint_ok else 'unavailable',
        event_rejected_trajectory_ids=rejected,
        interpretation='ordered_conditional_diagnostics_not_causal_decomposition; signed ESS loss can be negative',
        zero_support_convention='ESS=0 when all log weights are -inf; no replacement distribution',
        cross_organ_measurement_correlation='not_modeled',
        observation_error_structure='supplied_within_organ_covariance; conditional_independence_across_organs')


@_input_contract
def update_forecast(previous_run, observation_dir, associations, observation_model) -> dict:
    """Finite-bank likelihood update, with immutable previous forecasts and re-anchoring."""
    previous=_load_run(previous_run)
    if observation_dir is None:
        return dict(status='no_new_observations',previous_run_id=previous['run_id'],bank=previous)
    states,context=_history(observation_dir)
    association=_obj(associations); model=_obj(observation_model) if observation_model is not None else None
    association['observation_data_kind'] = context['data_kind']
    if association.get('plant_id') not in (None, previous['origin']['plant_id']):
        raise StudyError('Association plant differs from forecast plant')
    now=observed.aware_time(association['observed_at']); before=observed.aware_time(previous['origin']['observed_at'])
    dt=(now-before).total_seconds()/86400
    if dt<=0 or dt>30: raise StudyError('Update must follow origin within forecast horizon')
    targets={s['compound_leaf_id']:s for s in states if s['plant_id']==previous['origin']['plant_id'] and
             s['observed_at'] and observed.aware_time(s['observed_at'])==now}
    matching_events = [e for e in context.get('events', []) if e['plant_id']==previous['origin']['plant_id'] and
                       before < observed.aware_time(e['observed_at']) <= now]
    if not targets and not association.get('records') and not matching_events and not (model or {}).get('events'):
        return dict(status='no_new_observations',previous_run_id=previous['run_id'],bank=previous,
                    reason='no_observations_for_forecast_plant',source_hashes=context['source_hashes'])
    phases={s['capture_phase'] for s in targets.values() if s.get('capture_phase')}
    if len(phases)>1: raise StudyError('Conflicting Day-1 capture phases')
    if phases:
        phase=next(iter(phases))
        if association.get('capture_phase') not in (None,phase): raise StudyError('Association phase conflicts with Day-1 record')
        association['capture_phase']=phase
    if any(not s['eligible_state'] for s in targets.values()): raise StudyError('Update has ineligible measured blocks')
    if context['frame_id']!=previous['origin']['frame_id'] or context['coordinate_frame']!=previous['origin']['coordinate_frame']:
        raise StudyError('Update frame changed')
    entries=association.get('records',[]); seen=set()
    for r in entries:
        if r['status'] not in ('confirmed','ambiguous','missed','occluded','unmatched'):
            raise StudyError('Unknown association status')
        if r['status'] not in ('missed','occluded'):
            if r['observed_id'] not in targets or r['observed_id'] in seen:
                raise StudyError('Duplicate or unknown associated observation')
            seen.add(r['observed_id'])
        if not r.get('source_ref'): raise StudyError('Association evidence required')
        if r['status'] == 'confirmed':
            metadata = targets[r['observed_id']].get('observation_evidence', {})
            identity = metadata.get('identity', {})
            scope = r.get('identity_scope', identity.get('scope'))
            reason = None
            if scope not in (None, 'cross_date_same_organ'):
                reason = 'temporal_identity_unconfirmed'
            if identity.get('change') in ('merged', 'split', 'canopy_placement'):
                reason = 'organ_lineage_mapping_unsupported'
            # Existing manually reviewed same-ID associations remain supported.
            # A renamed measured organ additionally needs explicit old/new lineage.
            old_id = next((b['source_compound_leaf_id'] for t in previous['trajectories']
                for b in t['blocks'] if b.get('source_compound_leaf_id') and
                r['predicted_id'] in (b['block_id'], b['source_compound_leaf_id'])), None)
            if old_id is not None and r['observed_id'] != old_id:
                lineage = r.get('lineage') or {}
                if not (lineage.get('kind') == 'same_organ_rename' and lineage.get('source_ref') and
                        lineage.get('previous_id') == old_id and lineage.get('observed_id') == r['observed_id'] and
                        lineage.get('plant_id') == previous['origin']['plant_id']):
                    reason = 'organ_lineage_evidence_missing'
            if reason:
                return dict(status='unavailable', reason=reason, bank=previous,
                            previous_run_id=previous['run_id'], source_hashes=context['source_hashes'],
                            associations=entries, parameter_update='unavailable')
    confirmed_ids=[r['predicted_id'] for r in entries if r['status']=='confirmed']
    if len(confirmed_ids)!=len(set(confirmed_ids)):
        raise StudyError('Two observed blocks cannot share one confirmed predicted identity')
    for bid in sorted(set(targets)-seen):
        entries.append(dict(observed_id=bid,status='unmatched',source_ref='observation_bundle'))
    fields=[]; covariance=None
    if model is not None:
        model.setdefault('events', [])
        for event in context.get('events', []):
            event_time = observed.aware_time(event['observed_at'])
            if event['plant_id'] == previous['origin']['plant_id'] and before < event_time <= now:
                model['events'].append(dict(event,kind='present',through_day=(event_time-before).total_seconds()/86400))
        _evidence(model,'observation model'); fields=model['fields']; n=len(fields)
        covariance=_cov(model['covariance'],n,_num(model['numeric_tolerance'],'numeric_tolerance',0),'observation covariance')
        if n and np.linalg.eigvalsh(covariance).min()<=0: raise StudyError('Observation covariance must be positive definite')
        if len(set(fields))!=n: raise StudyError('Duplicate likelihood field')
        units={**{k:v for k,v in [('rachis_arc_length_mm','mm'),('left_extent_mm','mm'),('right_extent_mm','mm'),('supported_area_cm2','cm2')]},
               **{f'attachment_{a}_mm':'mm' for a in 'xyz'},'direction_tangent_1_rad':'rad','direction_tangent_2_rad':'rad'}
        if any(k not in units or model.get('units',{}).get(k)!=units[k] for k in fields):
            raise StudyError('Observation field/unit mismatch')
    offsets,posture_issues=posture_offsets(model,association)
    unavailable=list(posture_issues); logs=[]; discrepancies=[]; candidate_posteriors=[]
    trait_diagnostics=[]
    covariance_reason = variance_reason(model, data_kind=context['data_kind']) if model else None
    if covariance_reason:
        unavailable.append(covariance_reason)
    prior_logs=[]; geometry_logs=[]; event_rejected=[]
    for tr in previous['trajectories']:
        logw=math.log(tr['weight']) if tr['valid'] and tr['weight']>0 else -math.inf
        prior_logs.append(logw)
        if not tr['valid'] or tr['weight'] == 0:
            geometry_logs.append(logw); logs.append(logw); continue
        decoded=decode_trajectory(tr,[dt]) if tr['valid'] else []
        matched_ids = [_find_state(decoded, r['predicted_id']) for r in entries if r['status']=='confirmed']
        canonical = [s['block_id'] for s in matched_ids if s is not None]
        if len(set(canonical)) != len(canonical):
            raise StudyError('Duplicate confirmed identity through predicted aliases')
        for r in entries:
            if r['status'] in ('missed','occluded','unmatched'): continue
            candidates=([dict(predicted_id=r['predicted_id'],probability=1)] if r['status']=='confirmed' else r.get('candidates',[]))
            if not candidates or any(c.get('probability') is None for c in candidates):
                unavailable.append('association_probabilities_unavailable'); continue
            probs=[_num(c['probability'],'association probability',0) for c in candidates]
            if not math.isclose(sum(probs),1,abs_tol=observed.MATH_TOL): raise StudyError('Association probabilities must sum to one')
            ll=[]
            for candidate,prob in zip(candidates,probs):
                pred=_find_state(decoded,candidate['predicted_id']); target=targets[r['observed_id']]
                if pred is not None and pred['parent_stem_id']!=target['parent_stem_id']:
                    raise StudyError('Association changes parent axis')
                active_fields = []
                for field in fields:
                    semantic_field = 'direction_unit_xyz' if field.startswith('direction_tangent_') else field
                    reason = comparison_reason(pred or {}, target, semantic_field) if semantic_field in ('direction_unit_xyz', 'supported_area_cm2') else None
                    if reason:
                        trait_diagnostics.append(dict(trajectory_id=tr['trajectory_id'], observed_id=r['observed_id'],
                            quantity=field, reason_code=reason))
                    else:
                        active_fields.append(field)
                indexes = [fields.index(f) for f in active_fields]
                active_covariance = covariance[np.ix_(indexes,indexes)] if model else None
                diff=_discrepancy(pred,target,active_fields) if model and not covariance_reason else None
                diff=adjust_posture_residual(diff,active_fields,offsets)
                discrepancies.append(dict(trajectory_id=tr['trajectory_id'],observed_id=r['observed_id'],predicted_id=candidate['predicted_id'],
                                           fields=active_fields,residual=None if diff is None else diff.tolist()))
                if model is None: continue
                if fields and not active_fields:
                    unavailable.append('likelihood_trait_unavailable'); continue
                if diff is None:
                    likelihood=-math.inf if pred is None or pred.get('existence')!='appeared' else None
                    if likelihood is None: unavailable.append('likelihood_trait_unavailable'); continue
                else:
                    likelihood=-0.5*(len(active_fields)*math.log(2*math.pi)+np.linalg.slogdet(active_covariance)[1]+float(diff@np.linalg.solve(active_covariance,diff))) if active_fields else 0.0
                ll.append((math.log(prob) if prob>0 else -math.inf)+likelihood)
            if model and len(ll)==len(candidates):
                total=float(logsumexp(ll)); logw+=total
                if math.isfinite(total): candidate_posteriors.append(dict(trajectory_id=tr['trajectory_id'],observed_id=r['observed_id'],
                    candidates=[dict(c,posterior_probability=float(math.exp(x-total))) for c,x in zip(candidates,ll)]))
        geometry_logs.append(logw)
        rejected = False
        if model:
            for evidence in model.get('events',[]):
                if evidence.get('visibility')!='confirmed': continue
                ok = _event_likelihood(tr,evidence,dt)
                if ok is None:
                    unavailable.append((_truss_evidence_reason(tr,evidence) or 'event_particle_unavailable') if evidence.get('event_kind')=='truss' else 'event_particle_unavailable')
                    continue
                if not ok: logw=-math.inf; rejected=True
        if rejected: event_rejected.append(tr['trajectory_id'])
        logs.append(logw)
    posterior=deepcopy(previous)
    normalizer=float(logsumexp(logs)) if logs else -math.inf
    status='reanchored_without_parameter_update' if model is None or unavailable else 'reweighted'
    diagnostics = _conditioning_diagnostics(prior_logs,geometry_logs,logs,unavailable,model is not None,event_rejected)
    diagnostics['event_rejected_prior_mass']=sum(t['weight'] for t in previous['trajectories'] if t['trajectory_id'] in event_rejected)
    diagnostics['rejection_count_definition']='unique valid positive-prior-weight trajectories rejected by confirmed Boolean event evidence'
    diagnostics['final_weights_updated'] = model is not None and not unavailable and math.isfinite(normalizer)
    if model is not None and not math.isfinite(normalizer):
        return dict(status='update_incompatible',reason='all_zero_or_nonfinite_likelihood',previous_run_id=previous['run_id'],
                    associations=entries,discrepancies=discrepancies,source_hashes=context['source_hashes'],
                    conditioning_diagnostics=diagnostics)
    if model is not None and not unavailable:
        for tr,lw in zip(posterior['trajectories'],logs): tr['weight']=float(math.exp(lw-normalizer))
    neworigin=deepcopy(previous['origin']); neworigin['observed_at']=association['observed_at']
    neworigin['blocks']=sorted(targets.values(),key=lambda s:s['compound_leaf_id'])
    neworigin['history']+=list(targets.values())
    if context.get('day1_protocol'): neworigin['day1_records']=deepcopy(context['day1_protocol']['records'])
    if context.get('events'):
        neworigin.setdefault('event_history', []).extend(e for e in context['events'] if e['plant_id']==previous['origin']['plant_id'] and
            before < observed.aware_time(e['observed_at']) <= now)
    neworigin['session_id']=next(iter(targets.values()))['session_id'] if targets else None
    neworigin['origin_config'].update(observed_at=association['observed_at'],session_id=neworigin['session_id'])
    # Compound observations do not constitute a new measured parent stem.
    neworigin['attachments']={}
    for axis in neworigin['axes'].values():
        axis.update(points_mm=None,frames=None,segments=[],apex_available=False,state_source='unavailable_at_new_origin')
    neworigin['availability'].append(_reason('parent_geometry','new_parent_measurement_unavailable'))
    neworigin['source_hashes'].update(context['source_hashes']); neworigin['origin_id']=digest(neworigin)
    # Remaining original-horizon trajectories are continuations, not fabricated new 30-day runs.
    corrections=[]
    update_id=digest(dict(previous=previous['run_id'],association=association,hashes=context['source_hashes']))
    for tr in posterior['trajectories']:
        if not tr['valid']: continue
        dev=_clock(tr['config'],dt+tr.get('query_offset_days',0)); decoded=decode_trajectory(tr,[dt]); rebased=[]
        confirmed={r['predicted_id']:r['observed_id'] for r in entries if r['status']=='confirmed'}
        matched=set()
        for b in tr['blocks']:
            if b['birth_day']>dt:
                b['birth_day']-=dt; rebased.append(b); continue
            pred=_find_state(decoded,b['block_id'])
            key=b['source_compound_leaf_id'] or '/'.join(b['block_id'].split('/')[1:])
            obsid=confirmed.get(b['block_id'],confirmed.get(key))
            target=targets.get(obsid) if obsid else pred
            if target is None: continue
            if obsid:
                target,records=weighted_anchor(pred,target,association)
                corrections.extend(dict(r,trajectory_id=tr['trajectory_id'],organ_id=obsid,update_id=update_id) for r in records)
            try:
                rb=_remaining(b,dev,target)
            except StudyError as exc:
                tr.update(valid=False,status_reason=str(exc)); break
            rb['state_source']='observed' if obsid else 'inferred_preexisting'
            if obsid and target.get('reanchor_status'): rb['state_source']='weighted_observation_anchor'
            if obsid:
                matched.add(obsid); rb.update(block_id=obsid,source_compound_leaf_id=obsid)
                old_anchor=_attachment(tr,b,dev)
                if rb.get('attachment') and old_anchor is not None and target.get('attachment_xyz_mm') is not None and 'offset_mm' in rb['attachment']:
                    rb['attachment']['offset_mm']=(np.array(rb['attachment']['offset_mm'])+np.array(target['attachment_xyz_mm'])-old_anchor).tolist()
                elif rb.get('attachment') and old_anchor is not None and target.get('attachment_xyz_mm') is not None and 'node_index' in rb['attachment']:
                    # Retain node attachment with an observed offset instead of shifting the stem.
                    i=rb['attachment']['node_index']-1
                    rb['attachment']=dict(segment_index=i,fraction=1.0,offset_mm=(np.array(target['attachment_xyz_mm'])-old_anchor).tolist())
            rebased.append(rb)
        for obsid,target in targets.items():
            if obsid not in matched:
                # Explicitly unmatched measurements supersede same-ID latent copies but do not score as a match.
                rebased=[r for r in rebased if r['block_id']!=obsid]
                rebased.append(dict(block_id=obsid,source_compound_leaf_id=obsid,state=deepcopy(target),state_source='observed',
                                    birth_day=0.0,born_dev=dev,growth=None,frame=None,attachment=None))
        tr.update(blocks=rebased,origin=deepcopy(neworigin),query_offset_days=tr.get('query_offset_days',0)+dt,
                  available_horizon_days=tr.get('available_horizon_days',30)-dt)
        for event in tr['events']:
            if event['appearance_elapsed_days'] is not None: event['appearance_elapsed_days']-=dt
        tr['config']['computation']['output_days']=[d for d in tr['config']['computation']['output_days'] if d<=tr['available_horizon_days']]
    posterior.update(origin=neworigin,run_id=digest(dict(previous=previous['run_id'],association=association,model=model,hashes=context['source_hashes'])),
                     available_horizon_days=min((t.get('available_horizon_days',30) for t in posterior['trajectories'] if t['valid']),default=0))
    posterior['config']['computation']['output_days']=[d for d in posterior['config']['computation']['output_days'] if d<=posterior['available_horizon_days']]
    posterior['config']['continuation_only']=True
    cumulative=accumulate_corrections(previous.get('reanchor_cumulative',[]),corrections,update_id)
    posterior['reanchor_cumulative']=cumulative
    weights=[t['weight'] for t in posterior['trajectories']]
    result=dict(status=status,previous_run_id=previous['run_id'],previous_digest=digest(previous),bank=posterior,
                associations=entries,discrepancies=discrepancies,candidate_posteriors=candidate_posteriors,
                unavailable=sorted(set(unavailable)),effective_sample_size=1/sum(w*w for w in weights) if weights else 0,
                conditioning_diagnostics=diagnostics,
                trait_diagnostics=trait_diagnostics,
                posture_diagnostics=dict(offsets=offsets,unavailable=posture_issues,state_effect='none'),
                reanchor_corrections=corrections,reanchor_cumulative=cumulative,
                source_hashes=context['source_hashes'],parameter_update='unavailable' if status!='reweighted' else 'finite_bank',
                continuation_only=True,next_origin=neworigin,
                parameter_particle_weights={p['id']:sum(t['weight'] for t in posterior['trajectories'] if t.get('parameter_particle_id')==p['id'])
                                           for p in posterior['config'].get('parameter_particles') or []})
    observed._unchanged(neworigin['source_hashes'])
    return result


def _load_run(value):
    if isinstance(value,dict): return deepcopy(value)
    p=Path(value)
    folder=p if p.is_dir() else p.parent
    manifest=folder/'manifest.json'
    if manifest.exists():
        for name,h in read_json(manifest).get('output_hashes',{}).items():
            f=(folder/name).resolve()
            if not f.is_relative_to(folder.resolve()) or file_hash(f)!=h:
                raise StudyError('Frozen run output hash mismatch: '+name)
    return read_json(p/'trajectory_parameters.json' if p.is_dir() else p)


def _proper_score(values,weights,target):
    x=np.asarray(values,float); y=np.asarray(target,float); w=np.asarray(weights,float); w/=w.sum()
    if x.ndim==1:
        return float(w@abs(x-y)-0.5*np.sum(w[:,None]*w[None,:]*abs(x[:,None]-x[None,:])))
    return float(w@np.linalg.norm(x-y,axis=1)-0.5*np.sum(w[:,None]*w[None,:]*np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)))


@_input_contract
def evaluate_forecasts(run_paths, target_dir, evaluation_plan) -> dict:
    plan=_obj(evaluation_plan); states,context=_history(target_dir)
    train,test=plan.get('train_plant_ids',[]),plan.get('test_plant_ids',[])
    if not train or not test or len(set(train+test))!=len(train+test) or not plan.get('split_provenance'):
        raise StudyError('Explicit disjoint plant splits and provenance required')
    quantiles=plan.get('interval_quantiles')
    if not quantiles or len(quantiles)!=2 or not 0<quantiles[0]<quantiles[1]<1:
        raise StudyError('Explicit central interval quantiles required')
    records=[]; failures=[]; baselines=[]; run_audits=[]
    for path in run_paths:
        bank=_load_run(path); plant=bank['origin']['plant_id']; origin=observed.aware_time(bank['origin']['observed_at'])
        run_audits.append(dict(run_id=bank['run_id'],**environment_audit(bank['config'])))
        if plant not in test: raise StudyError('Run is not a held-out plant')
        fitted=any(p['source_kind']=='fitted' for p in bank['config'].get('parameter_particles') or [])
        if fitted and set(bank['config'].get('fitting_plant_ids',[]))-set(train): raise StudyError('Fitting split leaks into test')
        if fitted and not bank['config'].get('fitting_plant_ids'): raise StudyError('Missing fitting split provenance')
        for request in plan.get('targets',[]):
            if request['plant_id']!=plant: continue
            if request.get('forecast_origin')!=bank['origin']['observed_at']: continue
            if 'target_at' not in request and request.get('observed_at'):
                request = dict(request,target_at=request['observed_at'])
            target_time=observed.aware_time(request['target_at']); day=(target_time-origin).total_seconds()/86400
            if day<=0 or day>bank.get('available_horizon_days',30): raise StudyError('Invalid evaluation chronology/horizon')
            base=dict(run_id=bank['run_id'],plant_id=plant,horizon_days=day,target_id=request.get('compound_leaf_id'),target_at=request['target_at'])
            valid=[t for t in bank['trajectories'] if t['valid'] and t['weight']>0]
            if not valid:
                failures.append(dict(base,reason='no_valid_samples')); continue
            observed_target=next((s for s in states if s['plant_id']==plant and s['compound_leaf_id']==request.get('compound_leaf_id') and s['observed_at'] and observed.aware_time(s['observed_at'])==target_time),None)
            if request.get('kind','geometry')=='count':
                if any(_terminal_by(t,day) for t in valid) and request.get('scope') != 'main_axis':
                    failures.append(dict(base,reason='sympodial_continuation_not_modeled')); continue
                if request.get('visibility')!='confirmed' or request.get('count') is None:
                    failures.append(dict(base,reason='count_visibility_or_target_unavailable')); continue
                if type(request['count']) is not int or request['count']<0:
                    raise StudyError('Observed event count must be a nonnegative integer')
                pairs=[]
                for t in valid:
                    if any(r['reason_code'] in ('event_parameters_or_apex_unavailable','waiting_time_unavailable') for r in t.get('availability',[])): continue
                    pairs.append((sum(e.get('event_kind','leaf')=='leaf' and e['event_state']=='appeared' and 0<e['appearance_elapsed_days']<=day
                        and (request.get('scope')!='main_axis' or e['parent_stem_id']==request['parent_stem_id']) for e in t['events']),t['weight']))
                if pairs:
                    x,w=zip(*pairs); p=sum(ww for xx,ww in pairs if xx>0)/sum(w)
                    records.extend([dict(base,quantity='new_block_count',metric='crps',value=_proper_score(x,w,request['count']),unit='count',available_weight=sum(w)),
                                    dict(base,quantity='any_new_block',metric='brier',value=(p-float(request['count']>0))**2,unit='dimensionless',available_weight=sum(w))])
                else: failures.append(dict(base,reason='event_distribution_unavailable'))
                continue
            if request.get('kind')=='event':
                if request.get('visibility')!='confirmed': failures.append(dict(base,reason='event_visibility_unavailable')); continue
                if request.get('first_present_day') is not None:
                    left=_num(request['last_absent_day'],'last_absent_day',0); right=_num(request['first_present_day'],'first_present_day',0)
                    if not left<right<=day: raise StudyError('Event interval must end by its evaluation target time')
                elif request.get('through_day') is not None and not 0<=_num(request['through_day'],'through_day')<=day:
                    raise StudyError('Censoring time exceeds evaluation target')
                success=0.0; mass=0.0
                for tr in valid:
                    evidence = dict(request,kind='interval' if request.get('first_present_day') is not None else
                                    ('right_censored' if request.get('through_day') is not None else 'present'))
                    ok = _event_likelihood(tr,evidence,day)
                    if ok is None: continue
                    mass+=tr['weight']
                    if ok: success+=tr['weight']
                records.append(dict(base,quantity='appearance_time',metric='interval_log_loss',value=-math.log(success/mass) if success>0 else None,
                                    unit='dimensionless',available_weight=mass,reason='infinite_loss_zero_probability' if mass>0 and success==0 else ('event_distribution_unavailable' if mass==0 else None)))
                continue
            if observed_target is None or not observed_target['eligible_state']:
                failures.append(dict(base,reason='target_unavailable')); continue
            if request.get('association_status')!='confirmed':
                failures.append(dict(base,reason='association_unconfirmed')); continue
            predicted=[]
            for tr in valid:
                s=_find_state(decode_trajectory(tr,[day]),request.get('predicted_id',request['compound_leaf_id']))
                if s and s['existence']=='appeared': predicted.append((s,tr['weight']))
            exists=sum(w for _,w in predicted); mass=sum(t['weight'] for t in valid)
            records.append(dict(base,quantity='occurrence',metric='brier',value=(exists/mass-1)**2,unit='dimensionless',available_weight=mass))
            for field in ('rachis_arc_length_mm','left_extent_mm','right_extent_mm','supported_area_cm2'):
                y=_scalar(observed_target,field)
                pairs=[(_scalar(s,field),w) for s,w in predicted if _scalar(s,field) is not None and (field!='supported_area_cm2' or not comparison_reason(s,observed_target,field))]
                if y is None or not pairs:
                    failures.append(dict(base,reason=field+'_unavailable')); continue
                x,w=zip(*pairs); lower,upper=(_quantile(x,w,q) for q in quantiles); med=_quantile(x,w,0.5)
                for metric,value in (('crps',_proper_score(x,w,y)),('absolute_error',abs(med-y)),('interval_coverage',float(lower<=y<=upper)),('interval_width',upper-lower)):
                    records.append(dict(base,quantity=field,metric=metric,value=value,unit='dimensionless' if metric=='interval_coverage' else ('cm2' if field=='supported_area_cm2' else 'mm'),available_weight=sum(w)))
            for field,unit in (('attachment_xyz_mm','mm'),('direction_unit_xyz','rad')):
                pairs=[(s[field],w) for s,w in predicted if s.get(field) is not None and
                       (field!='direction_unit_xyz' or not comparison_reason(s,observed_target,field))]; y=observed_target.get(field)
                if not pairs or y is None: failures.append(dict(base,reason=field+'_unavailable')); continue
                x,w=zip(*pairs); medoid=min(range(len(x)),key=lambda i:sum(ww*np.linalg.norm(np.array(x[i])-xx) for xx,ww in pairs))
                error=math.acos(float(np.clip(np.dot(x[medoid],y),-1,1))) if unit=='rad' else float(np.linalg.norm(np.array(x[medoid])-y))
                records.append(dict(base,quantity=field,metric='representative_error',value=error,unit=unit,available_weight=sum(w)))
                scale=plan.get('energy_scales',{}).get(field)
                if scale is not None:
                    scale=_num(scale,'energy scale',0)
                    if scale==0: raise StudyError('Energy scale must be positive')
                    records.append(dict(base,quantity=field,metric='energy_loss',value=_proper_score(np.array(x)/scale,w,np.array(y)/scale),unit='dimensionless',available_weight=sum(w)))
            history=[s for s in bank['origin']['history'] if s['compound_leaf_id']==request['compound_leaf_id']]
            for name in ('persistence','recent_change'):
                try:
                    f=observed.forecast(history,[day],None,dict(model=name,max_horizon_days=30,provenance=plan['split_provenance']))['predictions'][0]
                    baselines.append(dict(base,model=name,status=f['status'],reason=f['reason'],axis_absolute_error_mm=abs(f['state']['rachis_arc_length_mm']-observed_target['rachis_arc_length_mm']) if f['state'] else None))
                except StudyError as exc: baselines.append(dict(base,model=name,status='unavailable',reason=str(exc)))
    observed._unchanged(context['source_hashes'])
    return dict(status='evaluated_unaccepted',metrics=records,failures=failures,baselines=baselines,plan=plan,
                source_hashes=context['source_hashes'],data_kind=context['data_kind'],validated_growth=False,
                run_environment_audits=run_audits)


def _csv(path, records, columns):
    # Shared writer uses the legacy unavailable string; this version explicitly requires empty cells.
    write_rows(path,[{k:('' if r.get(k) is None else r[k]) for k in columns} for r in records],columns)


@_input_contract
def export_run(result, output_path) -> Path:
    folder=output_dir(Path(output_path)); result=deepcopy(result)
    bank=result.get('bank',result)
    if result.get('status')=='unavailable' and 'trajectories' not in bank:
        write_json(folder/'unavailable.json',result)
    if 'trajectories' in bank:
        bank.update(environment_audit(bank['config']))
        for trajectory in bank['trajectories']:
            for event in trajectory.get('events', []): event.setdefault('event_kind', 'leaf')
        write_json(folder/'trajectory_parameters.json',bank)
        write_json(folder/'origin.json',bank['origin']); write_json(folder/'model.json',bank['config'])
        _csv(folder/'trajectories.csv',[dict(t,run_id=bank['run_id'],n_max_drawn=json.dumps(t['n_max_drawn'],sort_keys=True) if t.get('n_max_drawn') else None) for t in bank['trajectories']],
             ('run_id','trajectory_id','parameter_particle_id','scenario_id','weight','valid','status_reason','random_stream_id','n_max_drawn'))
        _csv(folder/'events.csv',[e for t in bank['trajectories'] for e in t.get('events',[])],
             ('trajectory_id','parent_stem_id','predicted_block_id','event_index','event_kind','node_rank','event_state','appearance_elapsed_days','appearance_development_days','material_node_id','truss_observable','geometry','optical_eligible','suitable_for_ppfd'))
        flat=[]
        for t in bank['trajectories']:
            for s in decode_trajectory(t,bank['config']['computation']['output_days'] or []):
                flat.append(dict(s,axis_length_mm=s.get('rachis_arc_length_mm'),
                    growth_unavailable=json.dumps(s['growth_unavailable'],sort_keys=True) if s.get('growth_unavailable') else None,
                    left_extent_mm=max(s['left_extent_mm']) if s.get('left_extent_mm') else None,
                    right_extent_mm=max(s['right_extent_mm']) if s.get('right_extent_mm') else None,
                    **dict(zip(('attachment_x_mm','attachment_y_mm','attachment_z_mm'),s.get('attachment_xyz_mm') or [None]*3)),
                    **dict(zip(('direction_x','direction_y','direction_z'),s.get('direction_unit_xyz') or [None]*3))))
        _csv(folder/'block_states.csv',flat,('trajectory_id','block_id','source_compound_leaf_id','parent_stem_id','elapsed_days',
             'target_at','existence','state_source','frame_id','attachment_x_mm','attachment_y_mm','attachment_z_mm',
             'direction_x','direction_y','direction_z','axis_length_mm','left_extent_mm','right_extent_mm',
             'supported_area_cm2','area_basis_id','optical_eligible','retention_assumption_exceeded','growth_unavailable'))
        summaries=result.get('summaries',[])
        _csv(folder/'summary.csv',[dict(r,computational_reasons=json.dumps(r.get('computational_reasons',[]),sort_keys=True),unavailable_reasons=json.dumps(r['unavailable_reasons'],sort_keys=True)
             if r.get('unavailable_reasons') else None) for r in summaries],('scenario_id','horizon_days','target_kind','target_id','quantity','conditioning',
             'statistic','value','unit','nominal_probability','available_weight','reason','count','post_termination_scope',
             'ensemble_weight','available_fraction_of_ensemble','unavailable_weight','unavailable_reasons',
             'existence_mass','computable_mass','available_mass','not_present_mass','existence_unknown_mass','computational_unavailable_mass','support_denominator','computational_reasons','truss_observable'))
    if 'previous_run_id' in result:
        write_json(folder/'update.json',{k:v for k,v in result.items() if k!='bank'})
        _csv(folder/'associations.csv',[dict(r,candidates=json.dumps(r.get('candidates')) if r.get('candidates') else None) for r in result.get('associations',[])],
             ('observed_id','predicted_id','status','source_ref','candidates'))
    if 'metrics' in result:
        write_json(folder/'evaluation.json',result)
        _csv(folder/'evaluation.csv',result['metrics'],('run_id','plant_id','horizon_days','target_id','target_at','quantity','metric','value','unit','available_weight','reason'))
        for key in ('failures','baselines'):
            rec=result[key]; cols=sorted({k for r in rec for k in r}) or ['reason']
            _csv(folder/(key+'.csv'),rec,cols)
    hashes=bank.get('origin',{}).get('source_hashes',result.get('source_hashes',{}))
    observed._unchanged(hashes)
    audit = environment_audit(bank.get('config', {}))
    if result.get('run_environment_audits'):
        audits = result['run_environment_audits']
        audit.update(p3_light_coupling='size_target_dli' if any(a['p3_light_coupling']=='size_target_dli' for a in audits) else 'none',
            p3_environment_paths=[dict(x,run_id=a['run_id']) for a in audits for x in a['p3_environment_paths']],
            environment_used=any(a['environment_used'] for a in audits),light_used=any(a['light_used'] for a in audits),
            environment_audit_status='unavailable' if any(a['environment_audit_status']=='unavailable' for a in audits) else 'complete')
    write_json(folder/'manifest.json',dict(schema_version=SCHEMA,run_id=bank.get('run_id'),created_at=datetime.now(timezone.utc).isoformat(),
        status=result.get('status'),source_hashes=hashes,code_hashes={str(p.resolve()):file_hash(p) for p in
            (Path(__file__),Path(observed.__file__),Path(__file__).with_name('growth_observation_evidence.py'),
             Path(__file__).with_name('growth_observation_options.py'))},
        config_hash=digest(bank.get('config')),output_hashes={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()},
        python=platform.python_version(),numpy=np.__version__,random_generator='PCG64 with SHA256 stable keys',command=sys.argv,
        validated_growth=False,suitable_for_ppfd=False,cross_organ_measurement_correlation='not_modeled',**audit))
    return folder


def templates(out):
    folder=output_dir(Path(out)); observed.templates(folder/'observations')
    columns={'parent_axis_points':('parent_stem_id','point_id','order','x_mm','y_mm','z_mm','source_ref','review_status'),
             'attachments':('compound_leaf_id','parent_stem_id','segment_start_id','segment_end_id','fraction','material_node_id',
                            'predecessor_id','adjacency_status','projection_residual_mm','review_ref')}
    table={}
    for name,cols in columns.items():
        p=folder/(name+'.csv'); _csv(p,[],cols); table[name]=dict(path=p.name,sha256=file_hash(p))
    write_json(folder/'structure.json',dict(schema_version=SCHEMA,plant_id=None,observed_at=None,coordinate_frame='board_mm',frame_id=None,
                axes=[],**table))
    write_json(folder/'origin.json',dict(plant_id=None,session_id=None,observed_at=None,numeric_tolerance=None,projection_tolerance_mm=None))
    write_json(folder/'model.json',blank_model())
    write_json(folder/'d8_d10_parameters.json',dict(
        environment_profile=None, growth_curve=None,
        beta_coordinates={trait:dict(x_max=None,unit=unit,a_m_dev_days=None,a_e_dev_days=None,a_m_over_a_e=None)
                          for trait,unit in BETA_TRAITS.items()},
        temperature_clock=dict(t_base_c=None,t_ref_c=None,m_max=None,lower_knot_c=None,upper_knot_c=None,source_kind=None,source_ref=None),
        calibration_clock_reference=dict(kind=None,time_basis=None,temperature_c=None,source_kind=None,source_ref=None),
        light_response=dict(kind=None,value=None,unit=DLI_UNIT,basis_id=None,scope=None,fixed_over_horizon=None,
            center=None,scale=None,light_basis=None,time_basis='complete_local_day',source_kind=None,source_ref=None,
            reference=dict(position_mm=None,normal_unit_xyz=None,frame_id=None,coordinate_frame=None,source_kind=None,source_ref=None),
            local_day=dict(start_at=None,end_at=None,timezone=None,coverage=None,source_kind=None,source_ref=None),
            applicability_interval=dict(min=None,max=None,unit=DLI_UNIT,source_kind=None,source_ref=None,applicability_source_ref=None),
            coefficients={curve:dict(coefficient_id=None,role=role,c_DLI=None,source_kind=None,source_ref=None,applicability_source_ref=None)
                          for curve,role in [('exponential','remaining_increment'),('determinate_beta','absolute_potential_maximum')]}),
        narrowing_assumption=dict(source_kind=None,source_ref=None),
        axis=dict(observed_node_count=None,node_count_source_ref=None,
                  node_budget=dict(pmf=None,source_kind=None,source_ref=None),
                  truss_rule=dict(first_truss_rank_pmf=None,interval_leaves=None,truss_observable=None,source_kind=None,source_ref=None)),
        max_leaf_age_dev_days=None,leaf_age_evidence=dict(source_kind=None,source_ref=None)))
    _csv(folder/'event_observations.csv',[],('plant_id','parent_stem_id','event_kind','node_rank','observed_at','visibility','source_ref','truss_observable'))
    write_json(folder/'observation_options.json',dict(
        associations=dict(reanchor_weight=1,capture_phase=None),
        reanchor_covariance=dict(mode='diagonal_variance',cross_covariance=None,traits={
            'rachis_arc_length_mm':dict(variance_state=None,variance_measurement=None,unit='mm^2',
                identifiable=None,source_kind=None,source_ref=None)}),
        posture_offset=dict(enabled=False,reference_phase=None,observed_at=None,time_reference='declared_capture_phase',
            source_kind=None,source_ref=None,offsets={}),
        day1_protocol=dict(path=None,sha256=None),day1_axis_id=None))
    write_json(folder/'summary_config.json',dict(quantiles=None,output_days=[0,7,14,21,30]))
    return folder


def synthetic_fixture(out: Path) -> tuple[dict,dict]:
    """Small explicit software fixture; never a crop parameter recommendation."""
    folder=output_dir(out); inp=folder/'observations'; inp.mkdir()
    timestamp='2026-10-03T00:00:00+00:00'
    base=dict(plant_id='synthetic_plant',compound_leaf_id='L1',session_id='s0')
    points=[[0.2,0,5],[5,1,6],[10,2,7]]
    d=np.array(points[-1])-points[0]; transverse=np.cross([0,0,1],d); transverse/=np.linalg.norm(transverse)
    record=dict(base,observed_at=timestamp,parent_stem_id='stem',identity_status='confirmed',review_source='synthetic_fixture',
                geometry_status='observed_axis',development_state='unfolding',first_observed_at=None,supported_area_cm2=2,
                area_basis_id='synthetic_patch',estimated_total_area_cm2=None,
                **dict(zip(('transverse_x','transverse_y','transverse_z'),transverse)))
    rr={'observations':[record], 'axes':[dict(base,station=s,**dict(zip(('x_mm','y_mm','z_mm'),point))) for s,point in zip([0,.5,1],points)],
        'extents':[dict(base,station=.5,left_mm=2,right_mm=3)]}
    tables={}
    for name,cols in (('observations',observed.OBS_COLUMNS),('axes',observed.AXIS_COLUMNS),('extents',observed.EXTENT_COLUMNS)):
        p=inp/(name+'.csv'); _csv(p,rr[name],cols); tables[name]=dict(path=p.name,sha256=file_hash(p))
    write_json(inp/'observations.json',dict(schema='compound-observations-1',coordinate_frame='board_mm',frame_id='synthetic_board',
         frame_stability='confirmed',frame_evidence='synthetic_fixture',data_kind='synthetic',reference_status='synthetic_truth',tables=tables))
    pr=[dict(parent_stem_id='stem',point_id=f'p{i}',order=i,x_mm=x,y_mm=y,z_mm=z,source_ref='synthetic_fixture',review_status='confirmed')
        for i,(x,y,z) in enumerate([[0,0,0],[0,0,10],[1,0,20]])]
    ar=[dict(compound_leaf_id='L1',parent_stem_id='stem',segment_start_id='p0',segment_end_id='p1',fraction=.5,
             material_node_id='n1',predecessor_id='',adjacency_status='confirmed',projection_residual_mm=.2,review_ref='synthetic_fixture')]
    table={}
    for name,recs in (('parent_axis_points',pr),('attachments',ar)):
        p=folder/(name+'.csv'); _csv(p,recs,tuple(recs[0])); table[name]=dict(path=p.name,sha256=file_hash(p))
    write_json(folder/'structure.json',dict(plant_id='synthetic_plant',observed_at=timestamp,frame_id='synthetic_board',coordinate_frame='board_mm',
        axes=[dict(parent_stem_id='stem',apex_status='active',apex_point_id='p2',apex_review='confirmed',reference_vector=[1,0,0])],**table))
    oc=dict(plant_id='synthetic_plant',session_id='s0',observed_at=timestamp,numeric_tolerance=1e-9,projection_tolerance_mm=.3)
    write_json(folder/'origin.json',oc)
    origin=prepare_origin(inp,folder/'structure.json',oc)
    constants=dict(birth_axis_length_mm=2,birth_left_extent_mm=.5,birth_right_extent_mm=.5,elevation_rad=.2,
                  k_length=.1,left_target_mm=5,right_target_mm=6,k_width=.1,rotation_x_rad=0,rotation_y_rad=0,
                  rotation_z_rad=.1,k_rotation=.1,D_area_cm2=1,k_area=.1,D_segment_mm=3,k_segment=.1)
    def family(wait):
        return dict(mean=[math.log(wait),0,math.log(5)],features=[],coefficients=[[],[],[]],standardization={},rho=.3,
                    innovation_covariance=np.diag([.03,.04,.02]).tolist(),initial_covariance=np.diag([.03,.04,.02]).tolist(),
                    constants={k:dict(value=v,unit=UNITS[k]) for k,v in constants.items()})
    particle=dict(id='synthetic_parameters',weight=1,source_kind='user_assumed',source_ref='SYNTHETIC SOFTWARE FIXTURE ONLY',
         coordinates=[dict(name='wait_dev_days',unit='development_day',transform='exp',reference=1),
                      dict(name='azimuth_rad',unit='rad',transform='identity'),
                      dict(name='D_length_mm',unit='mm',transform='exp',reference=1)],plant_covariance=np.diag([.01,.01,.01]).tolist(),
         families={'residual':family(1),'interval':family(7),'existing':family(7),'segment':family(7)},
         axes={'stem':dict(stop_probability=.05,residual_family='residual',inter_event_family='interval',direction_reference='absolute',
                          predecessor_id='L1',segment_families={'p0:p1':'segment','p1:p2':'segment'})},existing_blocks={'L1':'existing'})
    model=blank_model(); model.update(leaf_retention_assumption='no_loss_over_horizon',parameter_particles=[particle],
        environment_scenario=dict(id='synthetic_fixed_scenario',values=None,response_ref=None),
        computation=dict(seed=20261003,sample_count=16,output_days=[0,7,14,21,30],numeric_tolerance=1e-9,max_events_per_axis=100))
    write_json(folder/'model.json',model); write_json(folder/'summary_config.json',dict(quantiles=[.1,.5,.9]))
    return origin,model


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name in ('templates','demo','audit','forecast','decode','summarize','update','evaluate'):
        p=sub.add_parser(name); p.add_argument('--out',type=Path,required=True)
        if name in ('audit','forecast'):
            p.add_argument('--history',type=Path,required=True); p.add_argument('--structure',type=Path)
            p.add_argument('--origin',type=Path,required=True); p.add_argument('--model',type=Path,required=True)
        if name in ('decode','summarize','update'): p.add_argument('--run',type=Path,required=True)
        if name=='decode': p.add_argument('--days',nargs='+',type=float,required=True)
        if name in ('summarize','forecast'): p.add_argument('--summary-config',type=Path)
        if name=='update':
            p.add_argument('--observations',type=Path,required=True); p.add_argument('--associations',type=Path,required=True)
            p.add_argument('--observation-model',type=Path)
        if name=='evaluate':
            p.add_argument('--runs',type=Path,nargs='+',required=True); p.add_argument('--targets',type=Path,required=True); p.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='templates': templates(args.out)
        elif args.command=='demo':
            folder=output_dir(args.out); origin,c=synthetic_fixture(folder/'inputs'); bank=sample_forecast(origin,c)
            bank['summaries']=summarize_forecast(bank,{'quantiles':[.1,.5,.9]}); export_run(bank,folder/'results')
        elif args.command in ('audit','forecast'):
            origin=prepare_origin(args.history,args.structure,args.origin); c=read_json(args.model)
            if args.command=='audit':
                folder=output_dir(args.out); write_json(folder/'audit.json',dict(origin=origin,model=validate_model_config(c,origin)))
            else:
                result=sample_forecast(origin,c)
                if args.summary_config: result['summaries']=summarize_forecast(result,read_json(args.summary_config))
                export_run(result,args.out)
        elif args.command=='decode':
            bank=_load_run(args.run); bank['config']['computation']['output_days']=args.days; export_run(bank,args.out)
        elif args.command=='summarize':
            bank=_load_run(args.run)
            if args.summary_config is None: raise StudyError('--summary-config is required')
            bank['summaries']=summarize_forecast(bank,read_json(args.summary_config)); export_run(bank,args.out)
        elif args.command=='update':
            export_run(update_forecast(args.run,args.observations,args.associations,args.observation_model),args.out)
        elif args.command=='evaluate': export_run(evaluate_forecasts(args.runs,args.targets,args.plan),args.out)
    except (StudyError,KeyError,ValueError,OSError) as exc:
        parser.exit(2,f'Growth input/calculation error: {exc}\n')
    print(json.dumps(dict(status='written',out=str(args.out),validated_growth=False)))


if __name__=='__main__': main()
