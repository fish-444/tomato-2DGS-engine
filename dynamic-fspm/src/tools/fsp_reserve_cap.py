"""Opt-in reserve feedback: explicit potential-assimilate suppression, no carbon clipping."""
from __future__ import annotations

from copy import deepcopy
import math
from tools.vtc_growth_io import require, number

FRACTION_PARAMETERS = {'RESERVE_CAP_FRACTION', 'RESERVE_DOWNREG_START'}
PARAMETERS = FRACTION_PARAMETERS | {'CAP_FLOOR_GC'}
DEFAULTS = {
    'RESERVE_CAP_FRACTION': dict(value=.20, unit='fraction_total_plant_dry_mass',
        source_kind='literature_other_cultivar', source_ref='Butturini 2026 / Zepeda 2022; handoff/F3_LITERATURE_COMPARISON_2026-10-15.md',
        cultivar='other_cultivar_not_Micro_Tom_calibrated', uncertainty='borrowed scenario; not biologically validated'),
    'RESERVE_DOWNREG_START': dict(value=.50, unit='fraction_of_reserve_cap',
        source_kind='user_assumed', source_ref='CLI_PROMPT_CODEX_HOOKS_RESERVE_CAP_EXPANSION_2026-10-15.md',
        cultivar='Micro_Tom_unfitted_hypothesis', uncertainty='assumed linear feedback scenario'),
}
INITIAL_POLICY = 'preserve_initial_excess_until_consumed'


def active(request: dict) -> bool:
    return request.get('engine_hooks', {}).get('reserve_cap', False) is True


def floor_active(request: dict) -> bool:
    return request.get('engine_hooks', {}).get('reserve_cap_floor', False) is True


def floor_value(request: dict) -> float | None:
    """No scientific default; a disabled hook has no floor."""
    if not floor_active(request):
        return None
    return float(request['engine_hooks']['parameters']['CAP_FLOOR_GC']['value'])


def cap_terms(request: dict, mass_gDM: float, reserve_gC: float) -> dict:
    """Boundary diagnostics in carbon units; equality is not floor binding."""
    fraction = parameters(request)['RESERVE_CAP_FRACTION']['value']
    cf = request['parameters']['CARBON_FRACTION']['value']
    term = fraction * (cf * mass_gDM + reserve_gC)
    floor = floor_value(request)
    return dict(fractional_cap_gC=term, cap_floor_gC=floor,
                cap_gC=term if floor is None else max(term, floor),
                cap_floor_binding=floor is not None and floor > term)


def parameters(request: dict) -> dict:
    result = deepcopy(DEFAULTS)
    for key, value in request.get('engine_hooks', {}).get('parameters', {}).items():
        if key in PARAMETERS:
            result[key] = deepcopy(value)
    return result


def validate(request: dict) -> None:
    params = request.get('engine_hooks', {}).get('parameters', {})
    if floor_active(request):
        require(active(request), 'reserve_cap_floor requires reserve_cap')
        require('CAP_FLOOR_GC' in params, 'reserve_cap_floor requires explicit CAP_FLOOR_GC')
        p = params['CAP_FLOOR_GC']
        require(isinstance(p, dict) and p.get('unit') == 'gC', 'CAP_FLOOR_GC unit must be gC')
        require(p.get('source_kind') == 'user_assumed' and isinstance(p.get('source_ref'), str)
                and p['source_ref'].strip(), 'CAP_FLOOR_GC requires user_assumed provenance')
        require(type(p.get('value')) in (float, int), 'CAP_FLOOR_GC must be numeric')
        require(number(p['value'], 'CAP_FLOOR_GC', 0) > 0, 'CAP_FLOOR_GC must be positive')
    else:
        require('CAP_FLOOR_GC' not in params, 'CAP_FLOOR_GC requires reserve_cap_floor')
    if not active(request):
        require(not PARAMETERS.intersection(params), 'reserve parameters require reserve_cap')
        return
    require(not request.get('light_only'), 'reserve cap requires growth, incompatible with light_only')
    for key, p in parameters(request).items():
        if key == 'CAP_FLOOR_GC':
            continue
        require(isinstance(p, dict) and p.get('source_kind') in
                ('user_assumed', 'synthetic_assumed', 'literature_other_cultivar', 'literature', 'derived_low_confidence'),
                'reserve parameter source_kind invalid')
        require(all(p.get(k) for k in ('source_ref', 'cultivar', 'uncertainty')), 'reserve parameter provenance incomplete')
        require(p.get('unit') == DEFAULTS[key]['unit'], 'reserve parameter unit: ' + key)
        value = p.get('value')
        require(type(value) in (float, int), 'numeric reserve fraction required')
        number(value, key, 0)
        require(value < 1 and (key != 'RESERVE_CAP_FRACTION' or value > 0), 'reserve fraction out of range')


def accepted_gross(organ_mass_gDM: float, reserve_gC: float, potential_gC: float,
                   maintenance_gC: float, demand_gC: float, carbon_fraction: float,
                   efficiency: float, fraction: float, downreg_start: float,
                   cap_floor_gC: float | None = None) -> float:
    """Reference only; the native engine performs the actual causal update.

    A cap of f*(M+R/CF) gDM corresponds to f*(CF*M+R) gC.
    End-state inequality R_end <= f/(1-f)*CF*M_end yields the maximum
    accepted gross when demand is fully funded. Below that limit the ordinary
    allocator is unchanged. An initial excess is retained, with zero inflow.
    """
    cap_gC = fraction * (carbon_fraction * organ_mass_gDM + reserve_gC)
    if cap_floor_gC is not None:
        cap_gC = max(cap_gC, cap_floor_gC)
    if cap_gC <= 0 or reserve_gC >= cap_gC:
        factor = 0.
    elif reserve_gC <= downreg_start * cap_gC:
        factor = 1.
    else:
        factor = (cap_gC - reserve_gC) / ((1 - downreg_start) * cap_gC)
    capacity_gC = maintenance_gC + demand_gC + fraction / (1 - fraction) * (
        carbon_fraction * organ_mass_gDM + efficiency * demand_gC) - reserve_gC
    if cap_floor_gC is not None:
        capacity_gC = max(capacity_gC, maintenance_gC + demand_gC + cap_floor_gC - reserve_gC)
    return min(potential_gC * factor, max(0., capacity_gC))


def transform_native(template: str, request: dict) -> str:
    if not active(request):
        return template
    validate(request)
    p = parameters(request)
    fraction, onset = (float(p[k]['value']) for k in ('RESERVE_CAP_FRACTION', 'RESERVE_DOWNREG_START'))
    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old) == 1, 'reserve hook insertion point changed: ' + old[:65])
        template = template.replace(old, new)
    replace('deficit_gC,residual_gC");', 'deficit_gC,residual_gC,downregulated_gC");')
    old = 'double start=reserves[p],budget=start+gross-maintenance,funded=Math.min(Math.max(0,budget),demand),structural=0,gr=0;'
    replace(old, f'''// Gross remains potential NRH in flux/ledger; suppressed input is debited once.
            double plantOrganMass=0;for(int j=0;j<organs.size();j++)if(org(j).owner==p)plantOrganMass+=org(j).mass;
            double start=reserves[p],reserveCap={fraction}*(CARBON_FRACTION*plantOrganMass+start);
            double feedback=(reserveCap<=0 || start>=reserveCap)?0:(start<={onset}*reserveCap?1:(reserveCap-start)/((1-{onset})*reserveCap));
            double grossCapacity=maintenance+demand+{fraction}/(1-{fraction})*(CARBON_FRACTION*plantOrganMass+EFFICIENCY*demand)-start;
            double acceptedGross=Math.min(gross*feedback,Math.max(0,grossCapacity)),downregulated=gross-acceptedGross;
            double budget=start+acceptedGross-maintenance,funded=Math.min(Math.max(0,budget),demand),structural=0,gr=0;''')
    replace('(budget-reserves[p]-structural-gr));', '(budget-reserves[p]-structural-gr)+","+downregulated);')
    # This branch is unreachable because light_only+cap is refused at input validation.
    replace('+reserves[p]+",0,0");exportAxisFlux(p);continue;', '+reserves[p]+",0,0,0");exportAxisFlux(p);continue;')
    if floor_active(request):
        floor = floor_value(request)
        replace(f'reserveCap={fraction}*(CARBON_FRACTION*plantOrganMass+start);',
                f'reserveCap=Math.max({fraction}*(CARBON_FRACTION*plantOrganMass+start),{floor});')
        replace('double acceptedGross=Math.min(gross*feedback,Math.max(0,grossCapacity)),downregulated=gross-acceptedGross;',
                f'''double floorCapacity=maintenance+demand+{floor}-start;
            if(floorCapacity>grossCapacity)grossCapacity=floorCapacity;
            double acceptedGross=Math.min(gross*feedback,Math.max(0,grossCapacity)),downregulated=gross-acceptedGross;''')
    return template


def validate_interval(request: dict, previous_mass_gDM: float, end_mass_gDM: float,
                      ledger: dict, demand_gC: float) -> float:
    """Independent trial feedback/allocation check; never calls accepted_gross."""
    if not active(request):
        require('downregulated_gC' not in ledger, 'disabled reserve cap changed ledger schema')
        return 0.
    require('downregulated_gC' in ledger, 'reserve ledger missing downregulated_gC')
    p = parameters(request); f = p['RESERVE_CAP_FRACTION']['value']; onset = p['RESERVE_DOWNREG_START']['value']
    cf = request['parameters']['CARBON_FRACTION']['value']; efficiency = request['parameters']['EFFICIENCY']['value']
    start, gross, maintenance, end, removed = (
        number(ledger[k], k, 0) for k in ('reserve_start_gC', 'gross_gC', 'maintenance_gC', 'reserve_end_gC', 'downregulated_gC'))
    cap = f * (cf * previous_mass_gDM + start)
    floor = floor_value(request)
    if floor is not None:
        cap = max(cap, floor)
    ratio = start / cap if cap > 0 else 1.
    factor = min(1., max(0., (1 - ratio) / (1 - onset)))
    accepted = gross * factor
    trial_budget = start + accepted - maintenance
    trial_funded = min(max(0., trial_budget), demand_gC)
    trial_mass = previous_mass_gDM + trial_funded * efficiency / cf
    trial_reserve = max(0., trial_budget - trial_funded)
    trial_limit = f / (1 - f) * cf * trial_mass
    if floor is not None:
        trial_limit = max(trial_limit, floor)
    excess = max(0., trial_reserve - trial_limit)
    expected_removed = gross - max(0., accepted - excess)
    require(math.isclose(removed, expected_removed, abs_tol=1e-12, rel_tol=1e-12), 'reserve downregulation mismatch')
    require(removed <= gross + 1e-12, 'downregulation exceeds potential gross')
    end_cap = f * (cf * end_mass_gDM + end)
    if floor is not None:
        end_cap = max(end_cap, floor)
    if start > cap:
        require(math.isclose(removed, gross, abs_tol=1e-12, rel_tol=1e-12), 'initial excess must suppress all new gross')
        require(end <= start + 1e-12, 'initial excess increased')
    else:
        require(end <= end_cap + 1e-12, 'reserve exceeds end-state cap')
    return removed


def metadata(request: dict) -> dict:
    result = dict(parameters=parameters(request), initial_excess_policy=INITIAL_POLICY,
        initial_excess_source_kind='user_assumed', initial_excess_source_ref='User clarification 2026-10-15',
        plant_dry_mass_definition='sum organ mass_gDM + reserve_gC/CARBON_FRACTION',
        gross_gC_definition='potential NRH assimilation, before reserve feedback',
        effective_gross_gC_definition='gross_gC - downregulated_gC',
        feedback_time='interval start', hard_cap_time='interval end, after funded growth',
        discrete_cap='accepted gross <= maintenance + demand + f/(1-f)*(CF*organ_mass + efficiency*demand) - reserve_start; lower bound 0',
        carbon_closure='reserve_start + gross - downregulated - maintenance - structural - growth_resp - reserve_end',
        initial_excess_exception='Zero accepted gross while above cap; excess decreases only through existing maintenance and allocation. No carbon deletion.')
    if floor_active(request):
        result.update(reserve_cap_floor=True,
            cap_in_gC='max(f*(CF*organ_mass_gDM + reserve_gC), CAP_FLOOR_GC)',
            discrete_cap='accepted gross <= max(maintenance + demand + f/(1-f)*(CF*organ_mass + efficiency*demand) - reserve_start, maintenance + demand + CAP_FLOOR_GC - reserve_start); lower bound 0',
            floor_binding_definition='CAP_FLOOR_GC > fractional cap (strict; equality does not bind)',
            floor_diagnostics='carbon_audit.csv: interval-start cap_floor_binding and both terms; interval-end equivalents also exported')
    return result
