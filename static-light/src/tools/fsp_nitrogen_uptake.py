"""Engine hook `nitrogen_uptake` (E7 / S3-N). Base spec handoff/CLI_PROMPT_ENGINE_ENV_CO2_WATER_N_HOOKS_2026-10-18.md rev 2 § E7 and
build/e7_nitrogen_hooks_20261020_01/CONTRACT.md; adoption addendum build/e7_n_uptake_adopt_20261022_01/CONTRACT.md (2026-10-22).

Uptake capacity uses the Cardenas-Navarro et al. 1999 paired day/night Michaelis-Menten fits (references/tomato_n_uptake_20261021,
rows NU011/NU012 day and NU017/NU018 night, tier P3, literature_secondary). They are 5-minute 15NO3 influx capacities converted to a day
(flux_kind capacity_from_short_influx), never measured daily net uptake, and a day IMAX is never paired with a night KM:
    U_cap = root_DM x dt x [ f_day x IMAX_day x C/(KM_day + C) + f_night x IMAX_night x C/(KM_night + C) ]
with f_day = light_hours / (24 dt) from environment.csv. Root dry mass is the engine's own Root organ mass of that interval; there is no
root-mass parameter. Capacity is far above demand for this system, so uptake is capped by N status (feedback form (a) of the addendum):
    N_demand = max(0, (1 + luxury) x N_critical(Tei curve at the interval-start shoot mass) - plant N),  U = min(N_demand, U_cap, available)
NNI = plant N after the uptake / N_critical; it drives the growth factors only and never re-enters the uptake equation (no double limiting).
With substrate_water (E6) on, C is the root-zone concentration after irrigation and drainage (c_source root_zone) and the substrate pool is
tracked; with E6 off, C is the fertigation supply concentration (c_source supply) and the solution is an unlimited reservoir. Form (b),
CN99's internal-nitrate feedback (NU029, 923.06 mg nitrate-N per L plant water), is documented but not implemented: the engine has no plant
water or plant nitrate pool, and that threshold is not interchangeable with NNI. Default off: template byte-identical, inputs ignored.
EC is recorded only. No default parameter values; no value here may be tuned to reach an NNI target."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from tools.vtc_growth_io import require, number, rows, read_json, write_csv, write_json, sha
from tools import fsp_substrate_water as water

SWITCH = 'nitrogen_uptake'
UNITS = {'N_SUBSTRATE_INITIAL_MG': 'mg', 'N_PLANT_INITIAL_NNI': 'dimensionless', 'N_CURVE_A_PCT': 'percent', 'N_CURVE_B': 'dimensionless',
         'N_CURVE_MIN_MASS_T_HA': 't/ha', 'N_PLANT_DENSITY_PER_M2': 'plants/m2',
         'N_IMAX_DAY_MG_PER_G_ROOT_DAY': 'mg/(g d)', 'N_KM_DAY_MG_PER_L': 'mg/L',
         'N_IMAX_NIGHT_MG_PER_G_ROOT_DAY': 'mg/(g d)', 'N_KM_NIGHT_MG_PER_L': 'mg/L',
         'N_KINETICS_STATE': 'categorical', 'N_FLUX_KIND': 'categorical', 'N_LUXURY_FRACTION': 'dimensionless',
         'N_PHOTO_SENSITIVITY': 'dimensionless', 'N_EXPANSION_SENSITIVITY': 'dimensionless', 'N_ROOT_SHIFT': 'dimensionless', 'N_SUPPLY_FORM': 'categorical'}
PARAMETERS = set(UNITS)
LABELS = {'N_KINETICS_STATE', 'N_FLUX_KIND', 'N_SUPPLY_FORM'}
NUMERIC = PARAMETERS - LABELS
KINETIC_KEYS = ('N_IMAX_DAY_MG_PER_G_ROOT_DAY', 'N_KM_DAY_MG_PER_L', 'N_IMAX_NIGHT_MG_PER_G_ROOT_DAY', 'N_KM_NIGHT_MG_PER_L')
# Declared state pairs: the only kinetic values this hook accepts, each tied to its N_UPTAKE_VALUES.csv rows (no free tuning).
STATES = {
    'CN99_highN_phase3': dict(
        values=(154.0453842, 59.613792, 129.811465296, 78.15906),
        rows='day NU011/NU012 (56.275 umol g-1 whole-plant DM h-1, 4.256 mM), night NU017/NU018 (47.422, 5.58)',
        fit_domain_mM='day 2.335-20, night 2.087-20', note='Cardenas-Navarro 1999 Table 1 p629 phase III, Rondello 19 DAS, 20 C, 3 mM NO3 pretreatment'),
    'CN99_highN_phase1_lower_bound': dict(
        values=(18.039255120000004, 1.610805, 18.039255120000004, 1.610805),
        rows='NU007/NU008 (6.59 umol g-1 whole-plant DM h-1, 0.115 mM) applied to both light states',
        fit_domain_mM='0.01-0.051 (the sensitivity uses it far outside this domain, as a declared lower bound only)',
        note='phase I of the same end-of-light state; the dark phase-I pair NU013/NU014 exists and is deliberately not used'),
}
FLUX_KINDS = ('capacity_from_short_influx',)
FORMS = ('all_NO3', 'all_NH4', 'mixed_unspecified')
LUXURY_RANGE = (0.0, 0.30)   # declared range of the addendum; the value inside it is an assumption, never fitted
SOURCES = water.SOURCES
INPUTS = {'fertigation.csv': 'interval,plant_instance_id,n_mg_per_l,ec_ms_cm,source_kind'}
OUTPUT = 'nitrogen_balance.csv'
OUTPUTS = (OUTPUT,)
FIELDS = ['interval', 'plant_instance_id', 'n_substrate_start_mg', 'irrigation_ml', 'n_in_mg', 'c_mixed_mg_l', 'drainage_ml', 'n_drain_mg', 'n_available_mg',
          'water_available_ml', 'c_mg_l', 'c_source', 'root_dm_g', 'shoot_dm_g', 'plant_dm_g', 'shoot_t_ha', 'n_critical_pct', 'n_critical_mg', 'n_target_mg',
          'n_plant_start_mg', 'nni', 'n_demand_mg', 'u_cap_mg', 'uptake_mg', 'limit', 'f_day', 'imax_day', 'km_day', 'imax_night', 'km_night',
          'n_substrate_end_mg', 'n_plant_end_mg', 'f_n_photo', 'f_n_expansion', 'f_n_root', 'ec_ms_cm']
UPTAKE_TABLE = 'n_uptake.csv'
UPTAKE_FIELDS = ['interval', 'plant_instance_id', 'u_cap_mg', 'uptake_mg', 'n_demand_mg', 'limit', 'root_dm_g', 'c_mg_l', 'c_source', 'nni',
                 'n_plant_start_mg', 'n_plant_end_mg', 'f_day']
LIMITS = ('demand', 'capacity', 'supply')
UNLIMITED = 1e30             # stands for "no solution pool" when E6 is off; mirrored in the native fragment
NATIVE = Path(__file__).with_name('fsp_nitrogen_uptake_native.rgg')
RELATIVE_TOLERANCE = 1e-9
CLOSURE_TOLERANCE = 1e-12
RUN_CLOSURE_TOLERANCE = 1e-9


def enabled(req: dict) -> bool:
    return req.get('engine_hooks', {}).get(SWITCH, False) is True


active = enabled


def _records(req: dict) -> dict:
    return req.get('engine_hooks', {}).get('parameters', {})


def values(req: dict) -> dict:
    return {k: float(_records(req)[k]['value']) for k in NUMERIC}


def labels(req: dict) -> dict:
    return {k: _records(req)[k]['value'] for k in LABELS}


def has_water(req: dict) -> bool:
    return water.enabled(req)


def c_source(req: dict) -> str:
    return 'root_zone' if has_water(req) else 'supply'


def validate(req: dict) -> None:
    params = _records(req)
    present = set(params) & PARAMETERS
    if not enabled(req):
        require(not present, 'nitrogen_uptake records require the nitrogen_uptake switch')
        return
    require(present == PARAMETERS, f'nitrogen_uptake needs every record (no defaults): missing {sorted(PARAMETERS - present)}')
    require(not req.get('light_only'), 'nitrogen_uptake requires growth, incompatible with light_only')
    require(not req.get('schedule_intervals'), 'nitrogen_uptake: uniform dt_s intervals only (schedule_intervals not implemented)')
    for key in PARAMETERS:
        rec = params[key]
        require(isinstance(rec, dict) and all(rec.get(k) for k in ('unit', 'source_ref', 'cultivar', 'uncertainty')), f'{key}: provenance incomplete')
        require(rec.get('source_kind') in SOURCES, f'{key}: source_kind not accepted')
        require(rec['unit'] == UNITS[key], f'{key}: unit must be {UNITS[key]}')
        if key in NUMERIC:
            require(type(rec.get('value')) in (int, float) and not isinstance(rec['value'], bool), f'{key}: numeric value required')
    lab = labels(req)
    require(lab['N_SUPPLY_FORM'] in FORMS, f'N_SUPPLY_FORM must be one of {FORMS}')
    require(lab['N_FLUX_KIND'] in FLUX_KINDS, f'N_FLUX_KIND must be one of {FLUX_KINDS} (the adopted values are short-influx capacities)')
    require(lab['N_KINETICS_STATE'] in STATES, f'N_KINETICS_STATE must be one of {sorted(STATES)}')
    v = values(req)
    declared = STATES[lab['N_KINETICS_STATE']]['values']
    got = tuple(v[k] for k in KINETIC_KEYS)
    require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=0.) for a, b in zip(got, declared)),
            f'kinetic values must equal the declared {lab["N_KINETICS_STATE"]} pair {declared} (no tuning): got {got}')
    require(v['N_IMAX_DAY_MG_PER_G_ROOT_DAY'] >= 0 and v['N_IMAX_NIGHT_MG_PER_G_ROOT_DAY'] >= 0, 'IMAX must be >= 0')
    require(v['N_KM_DAY_MG_PER_L'] > 0 and v['N_KM_NIGHT_MG_PER_L'] > 0, 'KM must be > 0')
    require(LUXURY_RANGE[0] <= v['N_LUXURY_FRACTION'] <= LUXURY_RANGE[1], f'N_LUXURY_FRACTION must lie in {LUXURY_RANGE} (declared range)')
    require(v['N_SUBSTRATE_INITIAL_MG'] >= 0, 'N_SUBSTRATE_INITIAL_MG must be >= 0')
    require(0 < v['N_PLANT_INITIAL_NNI'] <= 2, 'N_PLANT_INITIAL_NNI must lie in (0, 2]')
    require(v['N_CURVE_A_PCT'] > 0 and 0 <= v['N_CURVE_B'] < 1 and v['N_CURVE_MIN_MASS_T_HA'] > 0, 'critical dilution curve: A > 0, 0 <= B < 1, W_min > 0')
    require(v['N_PLANT_DENSITY_PER_M2'] > 0, 'plant density must be > 0')
    for key in ('N_PHOTO_SENSITIVITY', 'N_EXPANSION_SENSITIVITY'):
        require(0 <= v[key] <= 1, f'{key} must lie in [0, 1]')
    require(0 <= v['N_ROOT_SHIFT'] <= 5, 'N_ROOT_SHIFT must lie in [0, 5]')


def input_tables(folder: Path, req: dict) -> dict:
    """Validated fertigation table when the switch is on (absent or incomplete -> refuse); {} when off."""
    if not enabled(req):
        return {}
    n = req['intervals']; name = 'fertigation.csv'
    require((Path(folder) / name).is_file(), f'nitrogen_uptake on but inputs/{name} missing')
    table = rows(Path(folder) / name, INPUTS[name])
    plants = [p['plant_instance_id'] for p in rows(Path(folder) / 'plants.csv')]
    keys = [(r['interval'], r['plant_instance_id']) for r in table]
    require(len(set(keys)) == len(keys), 'fertigation.csv: duplicate (interval, plant) row')
    require(set(keys) == {(str(i), p) for i in range(n) for p in plants}, 'fertigation.csv must hold exactly one row per interval and plant')
    for r in table:
        number(r['n_mg_per_l'], 'n_mg_per_l', 0)
        if r['ec_ms_cm'] != 'unavailable': number(r['ec_ms_cm'], 'ec_ms_cm', 0)
        require(r['source_kind'], 'fertigation evidence missing')
    return {name: table}


def concentration_matrix(tables: dict, plants: list[str], n: int, column: str = 'n_mg_per_l') -> list[list]:
    m = {(r['interval'], r['plant_instance_id']): r[column] for r in tables['fertigation.csv']}
    return [[m[str(i), p] for i in range(n)] for p in plants]


# ---- equations (mirror the native evaluation order exactly)
def critical_pct(shoot_g: float, v: dict) -> tuple[float, float]:
    wt = shoot_g * v['N_PLANT_DENSITY_PER_M2'] * 0.01
    return wt, v['N_CURVE_A_PCT'] * math.pow(max(wt, v['N_CURVE_MIN_MASS_T_HA']), -v['N_CURVE_B'])


def capacity(v: dict, root: float, c: float, f_day: float, dt_days: float) -> float:
    """Paired day/night Michaelis-Menten capacity over the interval; states are never mixed."""
    day = v['N_IMAX_DAY_MG_PER_G_ROOT_DAY'] * c / (v['N_KM_DAY_MG_PER_L'] + c)
    night = v['N_IMAX_NIGHT_MG_PER_G_ROOT_DAY'] * c / (v['N_KM_NIGHT_MG_PER_L'] + c)
    return root * dt_days * (f_day * day + (1 - f_day) * night)


def step(v: dict, n_sub: float, n_plant: float | None, w: dict | None, c_in: float, root: float, plant: float,
         dt_days: float, light_hours: float, water_on: bool) -> dict:
    """One plant-interval. `w` is the E6 row of the same interval when water_on, else None (then C = supply, pool untracked)."""
    shoot = plant - root
    wt, pct = critical_pct(shoot, v); ncrit = pct / 100 * plant * 1000
    if n_plant is None: n_plant = v['N_PLANT_INITIAL_NNI'] * ncrit
    f_day = light_hours / (24 * dt_days) if dt_days > 0 else 0.
    if water_on:
        n_in = w['irrigation_ml'] / 1000 * c_in; wmix = w['w_start_ml'] + w['irrigation_ml']
        cmix = (n_sub + n_in) / wmix * 1000 if wmix > 0 else 0.
        n_drain = w['drainage_ml'] / 1000 * cmix; n_avail = n_sub + n_in - n_drain
        c = n_avail / w['w_available_ml'] * 1000 if w['w_available_ml'] > 0 else 0.
        irrigation = w['irrigation_ml']; drainage = w['drainage_ml']; w_avail = w['w_available_ml']; sub_start = n_sub
    else:
        n_in = cmix = n_drain = n_avail = 0.; c = c_in
        irrigation = drainage = w_avail = 0.; sub_start = 0.
    u_cap = capacity(v, root, c, f_day, dt_days)
    target = (1 + v['N_LUXURY_FRACTION']) * ncrit
    demand = max(0., target - n_plant)
    avail = n_avail if water_on else UNLIMITED
    uptake = min(demand, min(u_cap, avail))
    n_end = target if uptake >= demand else n_plant + uptake
    nni = n_end / ncrit if ncrit > 0 else 1.
    limit = 'demand' if (demand <= u_cap and demand <= avail) else ('capacity' if u_cap <= avail else 'supply')
    f_n = max(0., min(1., 1 - v['N_PHOTO_SENSITIVITY'] * (1 - nni))); f_e = max(0., min(1., 1 - v['N_EXPANSION_SENSITIVITY'] * (1 - nni)))
    f_r = max(0., 1 + v['N_ROOT_SHIFT'] * (1 - nni)) if nni < 1 else 1.
    return dict(n_substrate_start_mg=sub_start, irrigation_ml=irrigation, n_in_mg=n_in, c_mixed_mg_l=cmix, drainage_ml=drainage, n_drain_mg=n_drain,
                n_available_mg=n_avail, water_available_ml=w_avail, c_mg_l=c, c_source='root_zone' if water_on else 'supply',
                root_dm_g=root, shoot_dm_g=shoot, plant_dm_g=plant, shoot_t_ha=wt, n_critical_pct=pct, n_critical_mg=ncrit, n_target_mg=target,
                n_plant_start_mg=n_plant, nni=nni, n_demand_mg=demand, u_cap_mg=u_cap, uptake_mg=uptake, limit=limit, f_day=f_day,
                imax_day=v['N_IMAX_DAY_MG_PER_G_ROOT_DAY'], km_day=v['N_KM_DAY_MG_PER_L'],
                imax_night=v['N_IMAX_NIGHT_MG_PER_G_ROOT_DAY'], km_night=v['N_KM_NIGHT_MG_PER_L'],
                n_substrate_end_mg=(n_avail - uptake) if water_on else 0., n_plant_end_mg=n_end,
                f_n_photo=f_n, f_n_expansion=f_e, f_n_root=f_r)


def reference(req: dict, tables: dict, water_rows: dict | None, masses: dict) -> dict:
    """Pool trajectories; water_rows[(i, pid)] = E6 row (floats) or None when E6 is off; masses[(i, pid)] = (root_g, plant_g) at the interval start."""
    v = values(req); n = req['intervals']; dt = req['dt_s'] / 86400; on = has_water(req)
    plants = [p['plant_instance_id'] for p in tables['plants.csv']]
    conc = concentration_matrix(tables, plants, n); ec = concentration_matrix(tables, plants, n, 'ec_ms_cm')
    env = tables['environment.csv']
    out = {}
    for pi, pid in enumerate(plants):
        n_sub = v['N_SUBSTRATE_INITIAL_MG']; n_plant = None
        for i in range(n):
            root, plant = masses[i, pid]
            w = water_rows[i, pid] if on else None
            row = step(v, n_sub, n_plant, w, float(conc[pi][i]), root, plant, dt, float(env[i]['light_hours']), on)
            row['ec_ms_cm'] = ec[pi][i]
            out[i, pid] = row
            if on: n_sub = row['n_substrate_end_mg']
            n_plant = row['n_plant_end_mg']
    return out


def metadata(req: dict) -> dict:
    lab = labels(req); v = values(req); state = STATES[lab['N_KINETICS_STATE']]
    return dict(switch=True, parameters={k: _records(req)[k] for k in sorted(PARAMETERS)}, inputs=list(INPUTS),
                outputs=['working/' + OUTPUT, UPTAKE_TABLE + ' (audit)', 'n_uptake_summary.json (audit)'],
                requires='substrate_water optional: on -> c_source root_zone with the pot pool; off -> c_source supply, unlimited reservoir',
                c_source=c_source(req), contract='build/e7_n_uptake_adopt_20261022_01/CONTRACT.md',
                pools='substrate mineral N (mg per pot, only when substrate_water is on) and one whole-plant N pool (mg); no per-organ N',
                kinetics=dict(state=lab['N_KINETICS_STATE'], flux_kind=lab['N_FLUX_KIND'], rows=state['rows'], fit_domain_mM=state['fit_domain_mM'],
                              note=state['note'], source='Cardenas-Navarro et al. 1999 doi:10.1093/jxb/50.334.625, converted in references/tomato_n_uptake_20261021/CALCULATIONS.json',
                              equation='U_cap = root_DM x dt x [f_day x IMAX_day x C/(KM_day + C) + f_night x IMAX_night x C/(KM_night + C)]',
                              day_night='f_day = light_hours / (24 dt) from environment.csv; the same C in both states; a day IMAX is never paired with a night KM',
                              caveat='5-minute 15NO3 influx capacities converted to a day; not measured daily net uptake; no long-term acclimation to 16-21 mM was tested'),
                root_mass='the engine Root organ dry mass of that interval; no root-mass parameter exists',
                demand=f'max(0, (1 + {v["N_LUXURY_FRACTION"]}) x N_critical - plant N); N_critical = %N_c/100 x plant DM x 1000 with %N_c = A max(W, W_min)^-B, W = shoot DM x density x 0.01',
                luxury_fraction=dict(value=v['N_LUXURY_FRACTION'], declared_range=list(LUXURY_RANGE), status='assumption until tissue-N data exist'),
                uptake='U = min(N_demand, U_cap, available); available only when substrate_water is on',
                nni='plant N after the interval uptake / N_critical at the interval-start mass; demand met -> 1 + luxury',
                feedback_form='(a) demand cap (implemented); (b) CN99 internal-nitrate threshold NU029 923.06 mg nitrate-N per L plant water documented, not implemented (no plant water / plant nitrate pool)',
                no_double_limiting='NNI drives the growth factors only; the uptake equation contains no NNI term',
                effects='AMAX x f_N(NNI); Leaflet demand x f_E(NNI); Root demand x f_R(NNI); no SLA effect; EC recorded only',
                n_form=lab['N_SUPPLY_FORM'] + ' (declared assumption; no form effect in this item)',
                assumptions=['organic additives ignored', 'N never derived from EC/TDS',
                             'demand curve is field processing tomato (Tei et al. 2002, literature_secondary) applied to juvenile Micro-Tom through a tray density and a W_min plateau',
                             'for the only Micro-Tom plant-N measurement (Machado 2023 ~37 DAS: 0.774 g plant DM, 11.6 mg N) this demand gives NNI 0.39, so NNI gates test model self-consistency, not agreement with measured tissue N',
                             'feed concentration is not the concentration at the absorbing root surface',
                             'carbon reserve excluded from DM'],
                excluded=['Albornoz 2018 (unit quarantine)', 'NH4 kinetics (no verified table)', 'Q10 temperature response (abstract only)'],
                biological_validation='not_established')


def transform_native(template: str, req: dict, tables: dict) -> str:
    """Inject the N fragment (after substrate_water when that is on); disabled requests return the exact input bytes."""
    validate(req)
    if not enabled(req):
        return template
    v = values(req); lab = labels(req); on = has_water(req)
    n = req['intervals']; plants = [p['plant_instance_id'] for p in tables['plants.csv']]
    conc = concentration_matrix(tables, plants, n); ec = concentration_matrix(tables, plants, n, 'ec_ms_cm')
    arr = lambda xs: 'new double[]{' + ','.join(repr(float(x)) for x in xs) + '}'
    sarr = lambda xs: 'new String[]{' + ','.join('"' + str(x).replace('"', '') + '"' for x in xs) + '}'
    code = NATIVE.read_text(encoding='utf-8')
    for key, token in (('N_SUBSTRATE_INITIAL_MG', 'SUB_INIT'), ('N_PLANT_INITIAL_NNI', 'NNI_INIT'), ('N_CURVE_A_PCT', 'A'), ('N_CURVE_B', 'B'),
                       ('N_CURVE_MIN_MASS_T_HA', 'MIN'), ('N_PLANT_DENSITY_PER_M2', 'DENSITY'), ('N_IMAX_DAY_MG_PER_G_ROOT_DAY', 'IMAX_DAY'),
                       ('N_KM_DAY_MG_PER_L', 'KM_DAY'), ('N_IMAX_NIGHT_MG_PER_G_ROOT_DAY', 'IMAX_NIGHT'), ('N_KM_NIGHT_MG_PER_L', 'KM_NIGHT'),
                       ('N_LUXURY_FRACTION', 'LUXURY'), ('N_PHOTO_SENSITIVITY', 'PHOTO_SENS'), ('N_EXPANSION_SENSITIVITY', 'EXPAND_SENS'),
                       ('N_ROOT_SHIFT', 'ROOT_SHIFT')):
        code = code.replace(f'@N_{token}@', repr(v[key]))
    code = code.replace('@N_HAS_WATER@', 'true' if on else 'false').replace('@N_C_SOURCE@', '"%s"' % c_source(req)).replace('@N_UNLIMITED@', repr(UNLIMITED))
    for token, expr in (('W_START', 'wStart[p]'), ('IRR', 'wIrr[p]'), ('DRAIN', 'wDrain[p]'), ('AVAIL_W', 'wAvail[p]')):
        code = code.replace(f'@N_{token}@', expr if on else '0')
    code = code.replace('@N_CONC@', 'new double[][]{' + ','.join(arr(r) for r in conc) + '}').replace('@N_EC@', 'new String[][]{' + ','.join(sarr(r) for r in ec) + '}')
    code = code.replace('@N_HEADER@', ','.join(FIELDS))
    require('@N_' not in code, 'unresolved nitrogen_uptake token')

    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old) == 1, f'nitrogen_uptake insertion point changed ({template.count(old)} != 1): ' + old[:70])
        template = template.replace(old, new)

    replace('@AXIS_CODE@', '@AXIS_CODE@\n' + code)
    if on:
        # after the substrate_water edits of the same lines (E6 runs first in the chain)
        replace('wInit();', 'wInit();nInit();')
        replace('exportMeshes(consumed,step);wBegin();', 'exportMeshes(consumed,step);wBegin();nBegin();')
        replace('calculatePS_Thornley(o.photons/o.area,wPhoto[o.owner]*AMAX', 'calculatePS_Thornley(o.photons/o.area,nPhoto[o.owner]*wPhoto[o.owner]*AMAX')
        replace('*CARBON_FRACTION/EFFICIENCY*(o.kind==9?wExpand[p]:1);', '*CARBON_FRACTION/EFFICIENCY*(o.kind==9?wExpand[p]*nExpand[p]:(o.kind==5?nRoot[p]:1));')
        replace('wBalance.flush();names=wFiles(names);', 'wBalance.flush();nBalance.flush();names=nFiles(wFiles(names));')
        replace('names=wFiles(names);\n    names=boundaryFiles(names,n);', 'names=nFiles(wFiles(names));\n    names=boundaryFiles(names,n);')
        replace('edges.close();wBalance.close();', 'edges.close();wBalance.close();nBalance.close();')
    else:
        replace('edges.println("day,parent_id,child_id");', 'edges.println("day,parent_id,child_id");\n    nInit();')
        replace('exportMeshes(consumed,step);', 'exportMeshes(consumed,step);nBegin();')
        replace('calculatePS_Thornley(o.photons/o.area,AMAX', 'calculatePS_Thornley(o.photons/o.area,nPhoto[o.owner]*AMAX')
        replace('o.demand=Math.max(0,targetMass(o)-o.mass)*CARBON_FRACTION/EFFICIENCY;',
                'o.demand=Math.max(0,targetMass(o)-o.mass)*CARBON_FRACTION/EFFICIENCY*(o.kind==9?nExpand[p]:(o.kind==5?nRoot[p]:1));')
        replace('names=boundaryFiles(names,boundary);', 'nBalance.flush();names=nFiles(names);\n    names=boundaryFiles(names,boundary);')
        replace('names=boundaryFiles(names,n);', 'names=nFiles(names);\n    names=boundaryFiles(names,n);')
        replace('edges.close();', 'edges.close();nBalance.close();')
    return template


# ---- independent checkers
class Factors:
    """f_n_photo / f_n_expansion / f_n_root per (interval, plant) from working/nitrogen_balance.csv; 1.0 when the hook is off."""

    def __init__(self, req: dict, data: dict | None):
        self.on = enabled(req); self.values = {}
        if self.on:
            require(data is not None and OUTPUT in data, 'nitrogen_uptake on but ' + OUTPUT + ' missing')
            for r in data[OUTPUT]:
                self.values[int(r['interval']), r['plant_instance_id']] = (float(r['f_n_photo']), float(r['f_n_expansion']), float(r['f_n_root']))

    def _get(self, pid: str, interval: int) -> tuple[float, float, float]:
        if not self.on: return (1., 1., 1.)
        v = self.values.get((interval, pid)); require(v is not None, f'nitrogen factor missing: {pid} interval {interval}'); return v

    def photo(self, pid: str, interval: int) -> float: return self._get(pid, interval)[0]

    def expansion(self, pid: str, interval: int) -> float: return self._get(pid, interval)[1]

    def root(self, pid: str, interval: int) -> float: return self._get(pid, interval)[2]


def _close(a: float, b: float, label: str) -> None:
    require(math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-15), f'nitrogen_uptake {label}: {a} != {b}')


def _closure(r: dict, water_on: bool) -> tuple[float, float]:
    """(substrate, plant) relative closure residuals of one row; the substrate term is 0 when the pool is untracked."""
    plant = float(r['n_plant_start_mg']) + float(r['uptake_mg']) - float(r['n_plant_end_mg'])
    scale_p = max(abs(float(r[k])) for k in ('n_plant_start_mg', 'uptake_mg', 'n_plant_end_mg'))
    rel_p = abs(plant) / scale_p if scale_p else abs(plant)
    if not water_on:
        return 0., rel_p
    sub = float(r['n_substrate_start_mg']) + float(r['n_in_mg']) - float(r['n_drain_mg']) - float(r['uptake_mg']) - float(r['n_substrate_end_mg'])
    scale_s = max(abs(float(r[k])) for k in ('n_substrate_start_mg', 'n_in_mg', 'n_drain_mg', 'uptake_mg', 'n_substrate_end_mg'))
    return (abs(sub) / scale_s if scale_s else abs(sub)), rel_p


def validate_outputs(pkg: dict, data: dict, n: int) -> dict:
    """Recompute every nitrogen_balance.csv row from the frozen inputs, the run's water balance (E6 on) and the interval-start masses."""
    req = pkg['request']
    if not enabled(req):
        require(OUTPUT not in data, 'disabled nitrogen_uptake output present')
        return {}
    require(OUTPUT in data, 'nitrogen_uptake output missing')
    on = has_water(req)
    require(not on or water.OUTPUT in data, 'nitrogen_uptake with substrate_water needs the water balance')
    tables = pkg['tables']; plants = [p['plant_instance_id'] for p in tables['plants.csv']]
    out = data[OUTPUT]; keys = [(int(r['interval']), r['plant_instance_id']) for r in out]
    require(len(set(keys)) == len(keys) and set(keys) == {(i, p) for i in range(n) for p in plants}, 'nitrogen_balance.csv coverage')
    wrows = None
    if on:
        wrows = {(int(r['interval']), r['plant_instance_id']): {k: float(r[k]) for k in ('w_start_ml', 'irrigation_ml', 'drainage_ml', 'w_available_ml')}
                 for r in data[water.OUTPUT]}
    masses = {}
    for s in data['states.csv']:
        k = (int(s['day']), s['plant_instance_id'])
        if k[0] < n:
            root, plant = masses.get(k, (0., 0.)); m = float(s['mass_gDM'])
            masses[k] = (root + (m if s['class'] == 'Root' else 0.), plant + m)
    ref = reference(req, tables, wrows, masses)
    worst = 0.; deficit = 0; limits = {}
    for r in out:
        i, pid = int(r['interval']), r['plant_instance_id']; e = ref[i, pid]
        for k, val in e.items():
            if k in ('ec_ms_cm', 'c_source', 'limit'): require(r[k] == str(val), f'nitrogen_uptake {k} {pid} {i}: {r[k]} != {val}')
            else: _close(float(r[k]), val, f'{k} {pid} interval {i}')
        s, p = _closure(r, on)
        require(s <= CLOSURE_TOLERANCE and p <= CLOSURE_TOLERANCE, f'nitrogen closure failed {pid} interval {i}: substrate {s} plant {p}')
        worst = max(worst, s, p); deficit += float(r['nni']) < 1; limits[r['limit']] = limits.get(r['limit'], 0) + 1
        require(r['limit'] in LIMITS, f'unknown limit label {r["limit"]}')
        require(float(r['nni']) >= 0 and all(0 <= float(r[k]) <= 1 for k in ('f_n_photo', 'f_n_expansion')) and float(r['f_n_root']) >= 0, 'nitrogen factor bounds')
        require(float(r['uptake_mg']) <= float(r['u_cap_mg']) + 1e-12 and float(r['uptake_mg']) <= float(r['n_demand_mg']) + 1e-12, 'uptake exceeds capacity or demand')
    return dict(rows=len(out), max_relative_closure_residual=worst, deficit_intervals=deficit, limits=limits, c_source=c_source(req))


def _run_balances(records: list[dict], water_on: bool) -> dict:
    """Run-level balances per plant: sum uptake == plant N gain, and (E6 on) solution N in - drained - taken up == pool change."""
    out = {}
    for pid in sorted({r['plant_instance_id'] for r in records}):
        rs = sorted((r for r in records if r['plant_instance_id'] == pid), key=lambda r: int(r['interval']))
        up = sum(float(r['uptake_mg']) for r in rs)
        gain = float(rs[-1]['n_plant_end_mg']) - float(rs[0]['n_plant_start_mg'])
        rel_plant = abs(up - gain) / max(abs(up), abs(gain), 1e-30)
        rec = dict(uptake_total_mg=up, plant_n_gain_mg=gain, plant_relative_residual=rel_plant)
        if water_on:
            n_in = sum(float(r['n_in_mg']) for r in rs); drained = sum(float(r['n_drain_mg']) for r in rs)
            pool = float(rs[-1]['n_substrate_end_mg']) - float(rs[0]['n_substrate_start_mg'])
            rel_sol = abs(n_in - drained - up - pool) / max(abs(n_in), abs(drained), abs(up), abs(pool), 1e-30)
            rec.update(n_in_total_mg=n_in, n_drained_total_mg=drained, substrate_pool_change_mg=pool, solution_relative_residual=rel_sol)
        out[pid] = rec
    return out


def audit(native: Path) -> dict:
    """N closure per plant-interval, the daily n_uptake table and the per-run summary, next to carbon_audit.csv (native.parent)."""
    from tools.vtc_development import verify_prepared
    prepared, pkg = verify_prepared(native); req = pkg['request']
    require(read_json(native / 'execution.json')['status'] == 'verified', 'native validation must precede the nitrogen audit')
    require(enabled(req), 'nitrogen_uptake audit on a run without the hook')
    on = has_water(req)
    table = rows(native / 'working' / OUTPUT)
    records = []; uptake_rows = []; worst = 0.
    for r in table:
        s, p = _closure(r, on)
        require(s <= CLOSURE_TOLERANCE and p <= CLOSURE_TOLERANCE, f'nitrogen closure failed {r["plant_instance_id"]} interval {r["interval"]}')
        worst = max(worst, s, p)
        records.append(dict(interval=int(r['interval']), plant_instance_id=r['plant_instance_id'], n_substrate_start_mg=r['n_substrate_start_mg'],
                            n_in_mg=r['n_in_mg'], n_drain_mg=r['n_drain_mg'], u_cap_mg=r['u_cap_mg'], n_demand_mg=r['n_demand_mg'], uptake_mg=r['uptake_mg'],
                            limit=r['limit'], n_substrate_end_mg=r['n_substrate_end_mg'], n_plant_start_mg=r['n_plant_start_mg'],
                            n_plant_end_mg=r['n_plant_end_mg'], nni=r['nni'], substrate_relative_residual=s, plant_relative_residual=p))
        uptake_rows.append({k: r[k] for k in UPTAKE_FIELDS})
    write_csv(native.parent / 'nitrogen_audit.csv', records)
    write_csv(native.parent / UPTAKE_TABLE, uptake_rows, UPTAKE_FIELDS)
    balances = _run_balances(table, on)
    for pid, b in balances.items():
        require(b['plant_relative_residual'] <= RUN_CLOSURE_TOLERANCE, f'run plant N balance failed for {pid}: {b}')
        if on: require(b['solution_relative_residual'] <= RUN_CLOSURE_TOLERANCE, f'run solution N balance failed for {pid}: {b}')
    limits = {}
    for r in table: limits[r['limit']] = limits.get(r['limit'], 0) + 1
    per_plant = {}
    for pid in sorted({r['plant_instance_id'] for r in table}):
        rs = sorted((r for r in table if r['plant_instance_id'] == pid), key=lambda r: int(r['interval']))
        nni = [float(r['nni']) for r in rs]
        per_plant[pid] = dict(intervals=len(rs), limits={k: sum(1 for r in rs if r['limit'] == k) for k in LIMITS},
                              nni_min=min(nni), nni_max=max(nni), nni_last=nni[-1],
                              uptake_total_mg=sum(float(r['uptake_mg']) for r in rs), demand_total_mg=sum(float(r['n_demand_mg']) for r in rs),
                              capacity_total_mg=sum(float(r['u_cap_mg']) for r in rs),
                              root_dm_first_g=float(rs[0]['root_dm_g']), root_dm_last_g=float(rs[-1]['root_dm_g']),
                              c_min_mg_l=min(float(r['c_mg_l']) for r in rs), c_max_mg_l=max(float(r['c_mg_l']) for r in rs),
                              balances=balances[pid])
    summary = dict(status='verified', nitrogen_uptake=True, intervals=len(table), c_source=c_source(req), substrate_water=on,
                   kinetics=metadata(req)['kinetics'], luxury_fraction=values(req)['N_LUXURY_FRACTION'], limits=limits,
                   max_relative_residual=worst, closure_tolerance=CLOSURE_TOLERANCE, run_closure_tolerance=RUN_CLOSURE_TOLERANCE,
                   nni_min=min(float(r['nni']) for r in table), nni_max=max(float(r['nni']) for r in table),
                   deficit_intervals=sum(float(r['nni']) < 1 for r in table), per_plant=per_plant,
                   uptake_table=UPTAKE_TABLE, uptake_table_sha256=sha(native.parent / UPTAKE_TABLE),
                   binding_limit_note='demand = N status caps uptake (capacity above demand); capacity = kinetics cap it; supply = the pot pool is exhausted',
                   biological_validation='not_established')
    write_json(native.parent / 'n_uptake_summary.json', summary)
    result = dict(summary, audit_sha256=sha(native.parent / 'nitrogen_audit.csv'), binding=prepared['binding'], model_hash=prepared['model_hash'],
                  metadata=metadata(req))
    write_json(native.parent / 'nitrogen_audit.json', result)
    return result
