"""Default-off S2-T period temperature responses under explicit scenario assumptions."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.vtc_growth_io import number, require

SWITCHES = {'development_temperature_curve', 'photosynthesis_temperature', 'maintenance_temperature_periods'}
UNITS = {'DEV_TOPT_C': 'degC', 'DEV_TMAX_C': 'degC', 'DEV_TEMPERATURE_FORM': 'categorical',
         'DEV_CURVE_BUD_STAGGER': 'boolean', 'DEV_CURVE_ORGAN_GROWTH': 'boolean',
         'FRUIT_TOPT_C': 'degC', 'FRUIT_TMAX_C': 'degC',
         'PHOTO_TMIN_C': 'degC', 'PHOTO_TOPT_C': 'degC', 'PHOTO_TMAX_C': 'degC', 'PHOTO_TREF_C': 'degC'}
PARAMETERS = set(UNITS)
SOURCES = {'literature_verified_fulltext', 'literature_secondary', 'literature_abstract_only',
           'user_assumed', 'derived_low_confidence'}
PROCESSES = {'development', 'bud_delay', 'stagger', 'organ_growth', 'fruit', 'germination'}


def enabled(req: dict, key: str) -> bool:
    hooks = req.get('engine_hooks')
    return isinstance(hooks, dict) and hooks.get(key, False) is True


def active(req: dict) -> bool:
    return any(enabled(req, key) for key in SWITCHES)


def values(req: dict) -> dict:
    return {key: rec['value'] for key, rec in (req.get('engine_hooks') or {}).get('parameters', {}).items() if key in PARAMETERS}


def validate(req: dict) -> None:
    hooks = req.get('engine_hooks') or {}
    require(isinstance(hooks, dict), 'S2-T engine_hooks must be an object')
    for switch in SWITCHES:
        require(type(hooks.get(switch, False)) is bool, 'S2-T switch must be boolean')
    records = hooks.get('parameters', {})
    require(isinstance(records, dict), 'S2-T parameter records required')
    required, allowed = set(), set()
    if enabled(req, 'development_temperature_curve'):
        require(enabled(req, 'process_base_temperatures'), 'S2-T development requires process_base_temperatures')
        required.update(('DEV_TOPT_C', 'DEV_TMAX_C'))
        allowed.update(('DEV_TEMPERATURE_FORM', 'DEV_CURVE_BUD_STAGGER', 'DEV_CURVE_ORGAN_GROWTH'))
        if enabled(req, 'fruit_sink') or records.get('DEV_CURVE_ORGAN_GROWTH', {}).get('value', False):
            required.update(('FRUIT_TOPT_C', 'FRUIT_TMAX_C'))
    if enabled(req, 'photosynthesis_temperature'):
        required.update(('PHOTO_TMIN_C', 'PHOTO_TOPT_C', 'PHOTO_TMAX_C', 'PHOTO_TREF_C'))
        require(not req.get('native_boundary'), 'S2-T photosynthesis native_boundary is not implemented')
    if enabled(req, 'maintenance_temperature_periods'):
        require(enabled(req, 'organ_maintenance_q10') or enabled(req, 'maint_eq7'),
                'S2-T maintenance periods requires S1 maintenance')
    require(required <= set(records), 'missing S2-T parameters: ' + ', '.join(sorted(required - set(records))))
    require((set(records) & PARAMETERS) <= required | allowed, 'S2-T parameter supplied for disabled process')
    if not active(req):
        return
    require(not req.get('light_only'), 'S2-T growth hooks incompatible with light_only')
    for key in set(records) & PARAMETERS:
        rec = records[key]
        require(isinstance(rec, dict) and rec.get('unit') == UNITS[key] and rec.get('source_kind') in SOURCES
                and bool(rec.get('source_ref')), 'S2-T provenance/unit invalid: ' + key)
        if key == 'DEV_TEMPERATURE_FORM':
            require(rec.get('value') in ('bilinear', 'beta'), 'S2-T form must be bilinear or beta')
        elif UNITS[key] == 'boolean':
            require(type(rec.get('value')) is bool and rec['source_kind'] == 'user_assumed',
                    'S2-T process extension must be an explicit user_assumed boolean')
        else:
            number(rec.get('value'), key)
    v = values(req)
    if enabled(req, 'development_temperature_curve'):
        # Existing S1 records remain replayable with their old provenance when S2-T is off.
        for key in ('TB_DEVELOPMENT_C', 'TB_ORGAN_GROWTH_C', 'TB_GERMINATION_C'):
            rec = records.get(key, {})
            require(rec.get('source_kind') in SOURCES and bool(rec.get('source_ref')), 'S2-T base provenance invalid: ' + key)
        base = number(records['TB_DEVELOPMENT_C']['value'], 'TB_DEVELOPMENT_C')
        growth = number(records['TB_ORGAN_GROWTH_C']['value'], 'TB_ORGAN_GROWTH_C')
        _cardinals(base, v['DEV_TOPT_C'], v['DEV_TMAX_C'])
        if 'FRUIT_TOPT_C' in v:
            _cardinals(growth, v['FRUIT_TOPT_C'], v['FRUIT_TMAX_C'])
    if enabled(req, 'photosynthesis_temperature'):
        _cardinals(v['PHOTO_TMIN_C'], v['PHOTO_TOPT_C'], v['PHOTO_TMAX_C'])
        require(v['PHOTO_TMIN_C'] < v['PHOTO_TREF_C'] < v['PHOTO_TMAX_C'], 'S2-T photo reference must be inside the response domain')


def _cardinals(base: float, optimum: float, maximum: float) -> None:
    require(all(math.isfinite(float(x)) for x in (base, optimum, maximum)) and base < optimum < maximum,
            'S2-T cardinal temperatures must satisfy base < optimum < maximum')


def rate(temp: float, tb: float, topt: float, tmax: float, form: str = 'bilinear') -> float:
    """Peak-scaled Yan-Hunt Eq4 or the declared triangular/bilinear thermal rate."""
    _cardinals(tb, topt, tmax)
    require(math.isfinite(temp), 'S2-T temperature must be finite')
    require(form in ('bilinear', 'beta'), 'S2-T unknown response form')
    if temp <= tb or temp >= tmax:
        return 0.
    if form == 'bilinear':
        return temp - tb if temp <= topt else (topt - tb) * (tmax - temp) / (tmax - topt)
    return ((tmax - temp) / (tmax - topt)) * ((temp - tb) / (topt - tb)) ** ((topt - tb) / (tmax - topt)) * (topt - tb)


def _legacy(req: dict, env: dict, dt: float, base: float) -> float:
    h = float(env['light_hours'])
    return max(0., (float(env['temp_day_c']) * h + float(env['temp_night_c']) * (24 * dt - h)) / (24 * dt) - base) * dt


def thermal(req: dict, env: dict, dt: float, process: str = 'organ_growth') -> float:
    require(process in PROCESSES, 'S2-T unknown thermal process')
    h, td, tn = (float(env[key]) for key in ('light_hours', 'temp_day_c', 'temp_night_c'))
    require(math.isfinite(dt) and dt > 0 and 0 <= h <= 24 * dt and math.isfinite(td) and math.isfinite(tn), 'S2-T invalid period environment')
    base_name = 'development' if process in ('development', 'bud_delay', 'stagger') else 'organ_growth' if process == 'fruit' else process
    records = (req.get('engine_hooks') or {}).get('parameters', {})
    base = records['TB_' + base_name.upper() + '_C']['value'] if enabled(req, 'process_base_temperatures') else req['parameters']['BASE_C']['value']
    if not enabled(req, 'development_temperature_curve') or process == 'germination':
        return _legacy(req, env, dt, base)
    v = values(req)
    curved = process == 'development' or process in ('bud_delay', 'stagger') and v.get('DEV_CURVE_BUD_STAGGER', False)
    fruit_curve = process == 'fruit' and enabled(req, 'fruit_sink') or process == 'organ_growth' and v.get('DEV_CURVE_ORGAN_GROWTH', False)
    curved = curved or fruit_curve
    opt = v.get('FRUIT_TOPT_C', 0.) if fruit_curve else v['DEV_TOPT_C']
    maximum = v.get('FRUIT_TMAX_C', 0.) if fruit_curve else v['DEV_TMAX_C']
    form = v.get('DEV_TEMPERATURE_FORM', 'bilinear')
    # Preserve the exact legacy arithmetic in the required constant sub-optimal gate.
    if td == tn and (not curved or form == 'bilinear' and td <= opt):
        return _legacy(req, env, dt, base)
    day = rate(td, base, opt, maximum, form) if curved else max(0., td - base)
    night = rate(tn, base, opt, maximum, form) if curved else max(0., tn - base)
    return (day * h + night * (24 * dt - h)) / 24


def photo_factor(req: dict, temp: float) -> float:
    if not enabled(req, 'photosynthesis_temperature'):
        return 1.
    v = values(req)
    if temp == v['PHOTO_TREF_C']:
        return 1.
    args = (v['PHOTO_TMIN_C'], v['PHOTO_TOPT_C'], v['PHOTO_TMAX_C'], 'beta')
    return rate(temp, *args) / rate(v['PHOTO_TREF_C'], *args)


def advance_fruit(flower_age: float, fruit_age: float, development: float, fruit: float,
                  onset: float, *, already_fruit: bool = False, eligible: bool = True) -> tuple[float, float]:
    """User-approved remaining aggregate-interval fraction, not a sub-day event order."""
    fraction = 1. if already_fruit else max(0., min(1., (flower_age + development - onset) / development)) if development > 0 else float(flower_age >= onset)
    return flower_age + development, fruit_age + fruit * fraction if eligible else fruit_age


def metadata(req: dict) -> dict:
    return dict(switches={key: enabled(req, key) for key in sorted(SWITCHES)},
                parameters={key: rec for key, rec in (req.get('engine_hooks') or {}).get('parameters', {}).items() if key in PARAMETERS},
                process_mapping=dict(development='leaf/truss appearance and pre-anthesis flowering',
                                     bud_delay='linear development base unless DEV_CURVE_BUD_STAGGER',
                                     stagger='separate linear development base unless DEV_CURVE_BUD_STAGGER',
                                     organ_growth='linear growth base unless DEV_CURVE_ORGAN_GROWTH',
                                     fruit='fruit-only response after declared onset when fruit_sink enabled',
                                     germination='unchanged S1 mean-temperature linear clock'),
                maintenance_temperature='per-period Q10 only with maintenance_temperature_periods; Eq7 already per-period',
                photo_accounting='Vreezen2022 Eq1 NRH core precedes separate Rd; native gross core uses no Rd; organ maintenance charged once; Merlice transfer remains assumed',
                period_discretization='day/night instantaneous rates integrated first; user-approved remaining aggregate interval fraction at flowering crossing; no sub-day ordering inferred',
                fruit_crossing='clamp((flower_before+development_increment-onset)/development_increment,0,1); already Fruit gets full increment; newborn boundary age remains zero',
                leaf_temperature='assumed equal to air temperature',
                omitted_processes=['acclimation', 'heat injury', 'cumulative damage', 'recovery', 'alpha_photorespiration'],
                biological_validation='not_established')


def transform_native(template: str, req: dict) -> str:
    if not active(req):
        return template
    validate(req)
    v = values(req)
    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old) == 1, 'S2-T insertion point changed: ' + old[:100])
        template = template.replace(old, new)
    code = Path(__file__).with_name('fsp_temperature_native.rgg').read_text(encoding='utf-8')
    development = enabled(req, 'development_temperature_curve')
    if not development:
        a, b = code.index('// S2_DEVELOPMENT_BEGIN'), code.index('// S2_DEVELOPMENT_END')
        code = code[:a] + code[b + len('// S2_DEVELOPMENT_END'):]
    constants = dict(v)
    records = (req.get('engine_hooks') or {}).get('parameters', {})
    for key in PARAMETERS:
        constants.setdefault(key, 'bilinear' if key == 'DEV_TEMPERATURE_FORM' else False if UNITS[key] == 'boolean' else 0.)
    for key in ('TB_DEVELOPMENT_C', 'TB_ORGAN_GROWTH_C', 'TB_GERMINATION_C'):
        constants[key] = records.get(key, {}).get('value', 0.)
    constants['PHOTO_ENABLED'] = enabled(req, 'photosynthesis_temperature')
    constants['FRUIT_ENABLED'] = enabled(req, 'fruit_sink')
    for key, value in constants.items():
        code = code.replace('@S2_' + key + '@', json.dumps(value))
    require('@S2_' not in code, 'unresolved S2-T native constant')
    replace('@AXIS_CODE@', '@AXIS_CODE@\n' + code)
    if enabled(req, 'photosynthesis_temperature'):
        replace('calculatePS_Thornley(o.photons/o.area,AMAX,ALPHA,THETA)',
                'calculatePS_Thornley(o.photons/o.area,AMAX*s2PhotoFactor(tday[step]),ALPHA,THETA)')
    if enabled(req, 'maintenance_temperature_periods'):
        replace('if(s1Eq7)q=(hours[step]', 'if(true)q=(hours[step]')
    if development:
        replace('ArrayList s1History=new ArrayList();', 'ArrayList s1History=new ArrayList(); double s2Stagger=0;')
        replace('if(step==0 && o.s1Initial){o.s1Dev=o.thermal;o.s1LeafAge=o.thermal;}',
                'if(step==0 && o.s1Initial){o.s1Dev=o.thermal;o.s1LeafAge=o.thermal;o.s2Stagger=o.thermal;}')
        replace('double dd=s1DD(s1GrowthBase),devDD=s1DD(s1DevBase),germDD=s1DD(s1GermBase);',
                'double dd=s2Thermal(step,"organ_growth"),devDD=s2Thermal(step,"development"),germDD=s1DD(s1GermBase);')
        replace('s1AdvanceOrgan(org(j),dd,devDD);', 's2AdvanceOrgan(org(j),dd,devDD,s2Thermal(step,"stagger"),s2FruitEnabled?s2Thermal(step,"fruit"):0);')
        replace('total+=Math.max(0,(tday[i]*hours[i]+tnight[i]*(24*d-hours[i]))/(24*d)-(s1Process?s1GrowthBase:BASE_C))*d;',
                'total+=s2Thermal(i,"organ_growth");')
        replace('s1_sla_m2_g,s1_fruit_age_dd");', 's1_sla_m2_g,s1_fruit_age_dd,s2_stagger_age_dd");')
        replace('+","+o.s1FruitAge);', '+","+o.s1FruitAge+","+o.s2Stagger);')
        if enabled(req, 'fruit_sink'):
            replace('if(o.thermal>=onset){', 'if(o.s1Dev>=onset){')
            replace('o.s1FruitAge=o.thermal-onset;', '// S2-T fruit age is accumulated with the approved crossing fraction.')
    return template
