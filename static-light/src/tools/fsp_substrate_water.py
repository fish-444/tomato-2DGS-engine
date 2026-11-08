"""Engine hook `substrate_water` (E6 / W1; handoff/CLI_PROMPT_ENGINE_ENV_CO2_WATER_N_HOOKS_2026-10-18.md rev 2 § E6; contract
build/e6_water_hooks_20261020_01/CONTRACT.md). One water pool per pot: W_end = W_start + irrigation - drainage - transpiration - soil evaporation,
drainage = excess above field capacity at the interval start, transpiration = radiation term (per-leaflet absorbed light) + VPD term (leaf area,
Tetens es(T) per period day/night from environment.csv temperatures and humidity.csv), soil evaporation = declared fraction x potential of the
pot surface. Stress factor f_w(theta_available) in [0, 1] (1 above THETA_CRIT, linear to 0 at THETA_WP) with separate sensitivities:
photosynthesis AMAX x f_photo (weaker) and Leaflet expansion demand x f_expansion (stronger). Default off: the native template is returned byte
for byte and the new inputs are ignored. On: irrigation.csv + humidity.csv are required, frozen and hashed; working/water_balance.csv exports theta,
fluxes and every factor per plant per interval; `audit` closes the water balance to 1e-12. No default parameter values: every record is explicit."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from tools.vtc_growth_io import require, number, rows, read_json, write_csv, write_json, sha

SWITCH = 'substrate_water'
UNITS = {'W_SUBSTRATE_VOLUME_ML': 'mL', 'W_POT_SURFACE_M2': 'm2', 'W_THETA_FC': 'm3/m3', 'W_THETA_WP': 'm3/m3', 'W_THETA_CRIT': 'm3/m3',
         'W_THETA_INITIAL': 'm3/m3', 'W_RAD_COEF_ML_PER_MOL': 'mL/mol', 'W_VPD_COEF_ML_PER_M2_KPA_H': 'mL/(m2 kPa h)',
         'W_NIGHT_STOMATAL_FACTOR': 'dimensionless', 'W_SOIL_EVAP_FRACTION': 'dimensionless', 'W_PHOTO_SENSITIVITY': 'dimensionless',
         'W_EXPANSION_SENSITIVITY': 'dimensionless'}
PARAMETERS = set(UNITS)
SOURCES = ('literature_verified_fulltext', 'literature_secondary', 'literature_abstract_only', 'user_assumed', 'derived_low_confidence', 'user_measured')
INPUTS = {'irrigation.csv': 'interval,plant_instance_id,irrigation_ml,source_kind', 'humidity.csv': 'interval,rh_day_pct,rh_night_pct,source_kind'}
OUTPUT = 'water_balance.csv'
OUTPUTS = (OUTPUT,)
FIELDS = ['interval', 'plant_instance_id', 'w_start_ml', 'irrigation_ml', 'drainage_ml', 'w_available_ml', 'theta_start', 'theta_available', 'vpd_day_kpa',
          'vpd_night_kpa', 'absorbed_mol', 'leaf_area_m2', 'transpiration_potential_ml', 'soil_evaporation_potential_ml', 'water_limited', 'transpiration_ml',
          'soil_evaporation_ml', 'w_end_ml', 'theta_end', 'f_w', 'f_photo', 'f_expansion']
NATIVE = Path(__file__).with_name('fsp_substrate_water_native.rgg')
RELATIVE_TOLERANCE = 1e-9      # recomputation of the exported fluxes (Math.exp vs math.exp, summation order)
CLOSURE_TOLERANCE = 1e-12      # relative water closure per plant-interval


def enabled(req: dict) -> bool:
    return req.get('engine_hooks', {}).get(SWITCH, False) is True


active = enabled


def _records(req: dict) -> dict:
    return req.get('engine_hooks', {}).get('parameters', {})


def values(req: dict) -> dict:
    return {k: float(_records(req)[k]['value']) for k in PARAMETERS}


def validate(req: dict) -> None:
    params = _records(req)
    present = set(params) & PARAMETERS
    if not enabled(req):
        require(not present, 'substrate_water records require the substrate_water switch')
        return
    require(present == PARAMETERS, f'substrate_water needs every record (no defaults): missing {sorted(PARAMETERS - present)}')
    require(not req.get('light_only'), 'substrate_water requires growth, incompatible with light_only')
    require(not req.get('schedule_intervals'), 'substrate_water: uniform dt_s intervals only (schedule_intervals not implemented)')
    for key in PARAMETERS:
        rec = params[key]
        require(isinstance(rec, dict) and all(rec.get(k) for k in ('unit', 'source_ref', 'cultivar', 'uncertainty')), f'{key}: provenance incomplete')
        require(rec.get('source_kind') in SOURCES, f'{key}: source_kind not accepted')
        require(rec['unit'] == UNITS[key], f'{key}: unit must be {UNITS[key]}')
        require(type(rec.get('value')) in (int, float) and not isinstance(rec['value'], bool), f'{key}: numeric value required')
    v = values(req)
    for key in ('W_SUBSTRATE_VOLUME_ML', 'W_POT_SURFACE_M2'):
        require(v[key] > 0, f'{key} must be positive')
    for key in ('W_RAD_COEF_ML_PER_MOL', 'W_VPD_COEF_ML_PER_M2_KPA_H'):
        require(v[key] >= 0, f'{key} must be >= 0')
    for key in ('W_NIGHT_STOMATAL_FACTOR', 'W_SOIL_EVAP_FRACTION', 'W_PHOTO_SENSITIVITY', 'W_EXPANSION_SENSITIVITY'):
        require(0 <= v[key] <= 1, f'{key} must lie in [0, 1]')
    require(0 <= v['W_THETA_WP'] < v['W_THETA_CRIT'] <= v['W_THETA_FC'] <= 1, 'substrate_water: 0 <= THETA_WP < THETA_CRIT <= THETA_FC <= 1 required')
    require(v['W_THETA_WP'] <= v['W_THETA_INITIAL'] <= v['W_THETA_FC'], 'substrate_water: THETA_WP <= THETA_INITIAL <= THETA_FC required')
    require(v['W_EXPANSION_SENSITIVITY'] >= v['W_PHOTO_SENSITIVITY'], 'substrate_water: leaf expansion sensitivity must be >= photosynthesis sensitivity (spec: stronger)')


# ---- pot geometry helpers (data/records/pots/pot_spec.csv; the run owner records the result as parameters)
def pot_row(pot_spec: Path, pot_type: str = 'square_cell') -> dict:
    row = next((r for r in rows(pot_spec) if r['pot_type'] == pot_type), None)
    require(row is not None, f'pot type {pot_type} not in {pot_spec}')
    require(row['shape'] == 'square' and row['wall_taper'] == 'none', 'substrate volume helper: square cell with straight walls only')
    return row


def substrate_volume_ml(pot_spec: Path, pot_type: str = 'square_cell') -> float:
    """inner edge^2 x (depth - medium surface depth below the rim), mL; assumes full filling and straight walls (derived_low_confidence)."""
    r = pot_row(pot_spec, pot_type)
    inner = float(r['pot_rim_inner_edge_mm']) / 1000.; depth = (float(r['pot_depth_mm']) - float(r['medium_surface_depth_below_rim_mm'])) / 1000.
    require(inner > 0 and depth > 0, 'pot dimensions must be positive')
    return inner * inner * depth * 1e6


def pot_surface_m2(pot_spec: Path, pot_type: str = 'square_cell') -> float:
    inner = float(pot_row(pot_spec, pot_type)['pot_rim_inner_edge_mm']) / 1000.
    return inner * inner


# ---- inputs
def input_tables(folder: Path, req: dict) -> dict:
    """Validated irrigation + humidity tables when the switch is on (absent or incomplete -> refuse); {} when off (files ignored)."""
    if not enabled(req):
        return {}
    n = req['intervals']
    for name in INPUTS:
        require((Path(folder) / name).is_file(), f'substrate_water on but inputs/{name} missing')
    tables = {name: rows(Path(folder) / name, header) for name, header in INPUTS.items()}
    plants = [p['plant_instance_id'] for p in rows(Path(folder) / 'plants.csv')]
    irr = tables['irrigation.csv']
    keys = [(r['interval'], r['plant_instance_id']) for r in irr]
    require(len(set(keys)) == len(keys), 'irrigation.csv: duplicate (interval, plant) row')
    require(set(keys) == {(str(i), p) for i in range(n) for p in plants}, 'irrigation.csv must hold exactly one row per interval and plant (explicit 0 mL required)')
    for r in irr:
        number(r['irrigation_ml'], 'irrigation_ml', 0); require(r['source_kind'], 'irrigation evidence missing')
    hum = tables['humidity.csv']
    require([r['interval'] for r in hum] == [str(i) for i in range(n)], 'humidity.csv must hold one row per interval in order')
    for r in hum:
        for k in ('rh_day_pct', 'rh_night_pct'):
            require(0 <= number(r[k], k, 0) <= 100, f'{k} must lie in [0, 100]')
        require(r['source_kind'], 'humidity evidence missing')
    return tables


def irrigation_matrix(tables: dict, plants: list[str], n: int) -> list[list[float]]:
    m = {(r['interval'], r['plant_instance_id']): float(r['irrigation_ml']) for r in tables['irrigation.csv']}
    return [[m[str(i), p] for i in range(n)] for p in plants]


# ---- equations (mirrors the native evaluation order)
def es_kpa(t_c: float) -> float:
    """Tetens saturation vapour pressure (FAO-56 eq. 11), kPa."""
    return 0.6108 * math.exp(17.27 * t_c / (t_c + 237.3))


def vpd_kpa(t_c: float, rh_pct: float) -> float:
    return es_kpa(t_c) * (1 - rh_pct / 100.0)


def stress(theta: float, wp: float, crit: float) -> float:
    if theta >= crit: return 1.
    if theta <= wp: return 0.
    return (theta - wp) / (crit - wp)


def step(v: dict, w_start: float, irrigation: float, t_day: float, t_night: float, rh_day: float, rh_night: float, light_h: float, dt_days: float,
         absorbed_umol_s: float, leaf_area_m2: float) -> dict:
    """One plant-interval of the pool model; the same arithmetic order as the native fragment."""
    V = v['W_SUBSTRATE_VOLUME_ML']
    w1 = w_start + irrigation; d = max(0., w1 - v['W_THETA_FC'] * V); avail = w1 - d; theta = avail / V
    fw = stress(theta, v['W_THETA_WP'], v['W_THETA_CRIT'])
    photo = 1 - v['W_PHOTO_SENSITIVITY'] * (1 - fw); expand = 1 - v['W_EXPANSION_SENSITIVITY'] * (1 - fw)
    h = light_h; nh = 24 * dt_days - h
    vd = vpd_kpa(t_day, rh_day); vn = vpd_kpa(t_night, rh_night)
    mol = absorbed_umol_s * h * 3600 / 1e6
    tp = v['W_RAD_COEF_ML_PER_MOL'] * mol + v['W_VPD_COEF_ML_PER_M2_KPA_H'] * leaf_area_m2 * (vd * h + v['W_NIGHT_STOMATAL_FACTOR'] * vn * nh)
    ep = v['W_SOIL_EVAP_FRACTION'] * v['W_VPD_COEF_ML_PER_M2_KPA_H'] * v['W_POT_SURFACE_M2'] * (vd * h + vn * nh)
    x = max(0., avail - v['W_THETA_WP'] * V)
    scale = x / (tp + ep) if (tp + ep > x and tp + ep > 0) else 1.
    t = scale * tp; e = scale * ep; w_end = avail - t - e
    return dict(w_start_ml=w_start, irrigation_ml=irrigation, drainage_ml=d, w_available_ml=avail, theta_start=w_start / V, theta_available=avail / V,
                vpd_day_kpa=vd, vpd_night_kpa=vn, absorbed_mol=mol, leaf_area_m2=leaf_area_m2, transpiration_potential_ml=tp, soil_evaporation_potential_ml=ep,
                water_limited=scale < 1, transpiration_ml=t, soil_evaporation_ml=e, w_end_ml=w_end, theta_end=w_end / V, f_w=fw, f_photo=photo, f_expansion=expand)


def reference(req: dict, tables: dict, absorbed: dict, leaf_area: dict) -> dict:
    """Pool trajectory for every plant from the inputs; absorbed[(interval, pid)] umol/s, leaf_area[(interval, pid)] m2 at the interval start."""
    v = values(req); n = req['intervals']; dt = req['dt_s'] / 86400
    plants = [p['plant_instance_id'] for p in tables['plants.csv']]
    irr = irrigation_matrix(tables, plants, n); hum = tables['humidity.csv']; env = tables['environment.csv']
    out = {}
    for pi, pid in enumerate(plants):
        w = v['W_THETA_INITIAL'] * v['W_SUBSTRATE_VOLUME_ML']
        for i in range(n):
            e = env[i]; h = hum[i]
            row = step(v, w, irr[pi][i], float(e['temp_day_c']), float(e['temp_night_c']), float(h['rh_day_pct']), float(h['rh_night_pct']), float(e['light_hours']), dt,
                       absorbed.get((i, pid), 0.), leaf_area.get((i, pid), 0.))
            out[i, pid] = row; w = row['w_end_ml']
    return out


def metadata(req: dict) -> dict:
    return dict(switch=True, parameters={k: _records(req)[k] for k in sorted(PARAMETERS)}, inputs=list(INPUTS), output='working/' + OUTPUT,
                pool='one water pool per pot (plant instance); every plant starts at THETA_INITIAL x volume',
                balance='W_end = W_start + irrigation - drainage - transpiration - soil_evaporation; irrigation at the interval start, drainage = excess above field capacity, losses limited to the water above the wilting point (proportional scaling, water_limited flag)',
                transpiration='c_R x absorbed photons (mol, per-leaflet absorbed light of the interval) + c_V x leaf area x (VPD_day x h + f_night x VPD_night x (24 dt - h)); per period, never on the daily mean',
                soil_evaporation='f_soil x c_V x pot surface x (VPD_day x h + VPD_night x (24 dt - h))',
                vpd='es(T) = 0.6108 exp(17.27 T / (T + 237.3)) kPa (Tetens, FAO-56 eq. 11, literature_secondary); VPD = es(T)(1 - RH/100) with day/night T and RH',
                stress='f_w(theta_available) = 1 above THETA_CRIT, linear to 0 at THETA_WP; f_photo = 1 - s_P (1 - f_w) on AMAX; f_expansion = 1 - s_E (1 - f_w) on Leaflet demand',
                effects='AMAX x f_photo in the leaflet NRH gross; Leaflet (blade) demand x f_expansion; unfunded carbon stays in the reserve',
                neutral='theta_available >= THETA_CRIT every interval -> every factor 1 -> growth tables bit-identical to the hook-off run',
                assumptions=['leaf temperature = air temperature', 'constant RH within each period', 'no capillary rise, no runoff beyond drainage',
                             'coefficients user_assumed unless the record says otherwise', 'no osmotic / salinity effect (EC recorded elsewhere)'],
                biological_validation='not_established')


def transform_native(template: str, req: dict, tables: dict) -> str:
    """Inject the pool fragment, the factor uses and the output file; disabled requests return the exact input bytes."""
    validate(req)
    if not enabled(req):
        return template
    v = values(req); n = req['intervals']; plants = [p['plant_instance_id'] for p in tables['plants.csv']]
    irr = irrigation_matrix(tables, plants, n); hum = tables['humidity.csv']
    arr = lambda xs: 'new double[]{' + ','.join(repr(float(x)) for x in xs) + '}'
    code = NATIVE.read_text(encoding='utf-8')
    for key, token in (('W_SUBSTRATE_VOLUME_ML', 'VOLUME'), ('W_POT_SURFACE_M2', 'POT_AREA'), ('W_THETA_FC', 'THETA_FC'), ('W_THETA_WP', 'THETA_WP'), ('W_THETA_CRIT', 'THETA_CRIT'),
                       ('W_THETA_INITIAL', 'THETA_INITIAL'), ('W_RAD_COEF_ML_PER_MOL', 'RAD_COEF'), ('W_VPD_COEF_ML_PER_M2_KPA_H', 'VPD_COEF'), ('W_NIGHT_STOMATAL_FACTOR', 'NIGHT_FACTOR'),
                       ('W_SOIL_EVAP_FRACTION', 'EVAP_FRACTION'), ('W_PHOTO_SENSITIVITY', 'PHOTO_SENS'), ('W_EXPANSION_SENSITIVITY', 'EXPAND_SENS')):
        code = code.replace(f'@W_{token}@', repr(v[key]))
    code = code.replace('@W_IRRIGATION@', 'new double[][]{' + ','.join(arr(r) for r in irr) + '}')
    code = code.replace('@W_RH_DAY@', arr(float(h['rh_day_pct']) for h in hum)).replace('@W_RH_NIGHT@', arr(float(h['rh_night_pct']) for h in hum))
    code = code.replace('@W_HEADER@', ','.join(FIELDS))
    require('@W_' not in code, 'unresolved substrate_water token')

    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old) == 1, f'substrate_water insertion point changed ({template.count(old)} != 1): ' + old[:70])
        template = template.replace(old, new)

    replace('@AXIS_CODE@', '@AXIS_CODE@\n' + code)
    replace('edges.println("day,parent_id,child_id");', 'edges.println("day,parent_id,child_id");\n    wInit();')
    replace('exportMeshes(consumed,step);', 'exportMeshes(consumed,step);wBegin();')                       # before the light computation of the interval
    replace('calculatePS_Thornley(o.photons/o.area,AMAX', 'calculatePS_Thornley(o.photons/o.area,wPhoto[o.owner]*AMAX')   # prefix: compatible with rp / S2-T edits
    replace('o.demand=Math.max(0,targetMass(o)-o.mass)*CARBON_FRACTION/EFFICIENCY;', 'o.demand=Math.max(0,targetMass(o)-o.mass)*CARBON_FRACTION/EFFICIENCY*(o.kind==9?wExpand[p]:1);')
    replace('            exportAxisFlux(p);\n        }\n        attachGraph();', '            wEnd(p);exportAxisFlux(p);\n        }\n        attachGraph();')
    replace('names=boundaryFiles(names,boundary);', 'wBalance.flush();names=wFiles(names);\n    names=boundaryFiles(names,boundary);')
    replace('names=boundaryFiles(names,n);', 'names=wFiles(names);\n    names=boundaryFiles(names,n);')
    replace('edges.close();', 'edges.close();wBalance.close();')
    return template


# ---- independent checkers
class Factors:
    """f_photo / f_expansion per (interval, plant) from working/water_balance.csv; 1.0 for every plant when the hook is off."""

    def __init__(self, req: dict, data: dict | None):
        self.on = enabled(req); self.values = {}
        if self.on:
            require(data is not None and OUTPUT in data, 'substrate_water on but ' + OUTPUT + ' missing')
            for r in data[OUTPUT]:
                self.values[int(r['interval']), r['plant_instance_id']] = (float(r['f_photo']), float(r['f_expansion']))

    def _get(self, pid: str, interval: int) -> tuple[float, float]:
        if not self.on: return (1., 1.)
        v = self.values.get((interval, pid)); require(v is not None, f'water factor missing: {pid} interval {interval}'); return v

    def photo(self, pid: str, interval: int) -> float:
        return self._get(pid, interval)[0]

    def expansion(self, pid: str, interval: int) -> float:
        return self._get(pid, interval)[1]


def _close(a: float, b: float, label: str, rel: float = RELATIVE_TOLERANCE) -> None:
    require(math.isclose(a, b, rel_tol=rel, abs_tol=1e-15), f'substrate_water {label}: {a} != {b}')


def validate_outputs(pkg: dict, data: dict, n: int) -> dict:
    """Recompute every water_balance.csv row from the frozen inputs, the plant's absorbed light and interval-start leaf area; closure 1e-12."""
    req = pkg['request']
    if not enabled(req):
        require(OUTPUT not in data, 'disabled substrate_water output present')
        return {}
    require(OUTPUT in data, 'substrate_water output missing')
    tables = pkg['tables']; plants = [p['plant_instance_id'] for p in tables['plants.csv']]
    out = data[OUTPUT]; keys = [(int(r['interval']), r['plant_instance_id']) for r in out]
    require(len(set(keys)) == len(keys) and set(keys) == {(i, p) for i in range(n) for p in plants}, 'water_balance.csv coverage')
    leaflets = {(int(s['day']), s['organ_id']): float(s['area_m2']) for s in data['states.csv'] if s['class'] == 'Leaflet'}
    owner = {s['organ_id']: s['plant_instance_id'] for s in data['states.csv']}
    absorbed = {}; area = {}
    for f in data['flux.csv']:
        k = (int(f['interval']), f['plant_instance_id']); absorbed[k] = absorbed.get(k, 0.) + float(f['absorbed_umol_s'])
    for (day, oid), a in leaflets.items():
        if day < n:
            k = (day, owner[oid]); area[k] = area.get(k, 0.) + a
    ref = reference(req, tables, absorbed, area)
    worst = 0.; limited = 0
    for r in out:
        i, pid = int(r['interval']), r['plant_instance_id']; e = ref[i, pid]
        for k, val in e.items():
            if k == 'water_limited': require((r[k] == 'true') == val, f'substrate_water water_limited flag {pid} {i}')
            else: _close(float(r[k]), val, f'{k} {pid} interval {i}')
        residual = float(r['w_start_ml']) + float(r['irrigation_ml']) - float(r['drainage_ml']) - float(r['transpiration_ml']) - float(r['soil_evaporation_ml']) - float(r['w_end_ml'])
        scale = max(abs(float(r[k])) for k in ('w_start_ml', 'irrigation_ml', 'drainage_ml', 'transpiration_ml', 'soil_evaporation_ml', 'w_end_ml'))
        require(abs(residual) <= CLOSURE_TOLERANCE * scale if scale else residual == 0, f'water closure failed {pid} interval {i}: {residual}')
        worst = max(worst, abs(residual) / scale if scale else 0.); limited += r['water_limited'] == 'true'
        require(0 <= float(r['f_w']) <= 1 and float(r['f_photo']) <= 1 and float(r['f_expansion']) <= 1, 'water factor bounds')
    return dict(rows=len(out), max_relative_closure_residual=worst, water_limited_intervals=limited)


def audit(native: Path) -> dict:
    """Water closure per plant-interval next to carbon_audit.csv (native.parent): water_audit.csv + water_audit.json."""
    from tools.vtc_development import verify_prepared
    prepared, pkg = verify_prepared(native); req = pkg['request']
    require(read_json(native / 'execution.json')['status'] == 'verified', 'native validation must precede the water audit')
    require(enabled(req), 'substrate_water audit on a run without the hook')
    records = []; worst = 0.
    for r in rows(native / 'working' / OUTPUT):
        residual = float(r['w_start_ml']) + float(r['irrigation_ml']) - float(r['drainage_ml']) - float(r['transpiration_ml']) - float(r['soil_evaporation_ml']) - float(r['w_end_ml'])
        scale = max(abs(float(r[k])) for k in ('w_start_ml', 'irrigation_ml', 'drainage_ml', 'transpiration_ml', 'soil_evaporation_ml', 'w_end_ml'))
        rel = abs(residual) / scale if scale else 0.
        require(rel <= CLOSURE_TOLERANCE, f'water closure failed {r["plant_instance_id"]} interval {r["interval"]}: {residual}')
        worst = max(worst, rel)
        records.append(dict(interval=int(r['interval']), plant_instance_id=r['plant_instance_id'], w_start_ml=r['w_start_ml'], irrigation_ml=r['irrigation_ml'],
                            drainage_ml=r['drainage_ml'], transpiration_ml=r['transpiration_ml'], soil_evaporation_ml=r['soil_evaporation_ml'], w_end_ml=r['w_end_ml'],
                            theta_end=r['theta_end'], f_w=r['f_w'], water_limited=r['water_limited'], residual_ml=residual, relative_residual=rel))
    write_csv(native.parent / 'water_audit.csv', records)
    result = dict(status='verified', substrate_water=True, intervals=len(records), max_relative_residual=worst, closure_tolerance=CLOSURE_TOLERANCE,
                  water_limited_intervals=sum(r['water_limited'] == 'true' for r in records), min_theta_end=min(float(r['theta_end']) for r in records),
                  min_f_w=min(float(r['f_w']) for r in records), binding=prepared['binding'], model_hash=prepared['model_hash'],
                  audit_sha256=sha(native.parent / 'water_audit.csv'), metadata=metadata(req))
    write_json(native.parent / 'water_audit.json', result)
    return result
