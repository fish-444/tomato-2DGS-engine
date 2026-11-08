"""Prepare and run small 30-day reserve/expansion fixtures or a declared frozen host."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

from tools.vtc_growth_io import ROOT, require, fresh_build, rows, read_json, write_json, write_csv, sha
from tools.fsp_engine_hooks import SCHEMA
from tools.fsp_engine_hooks_acceptance import parameter, leaf_parameters, storage_report
from tools import fsp_reserve_cap as cap


def prepare_case(out: Path, case: str, host: Path | None = None) -> Path:
    from tools.vtc_development import demo, prepare
    fresh_build(out)
    if host:
        shutil.copytree(host/'inputs', out/'inputs')
    else:
        demo(out/'inputs', intervals=30, rays=300000)
    inputs = out/'inputs'; req = read_json(inputs/'request.json')
    require(req['fixture'] is True, 'acceptance adapter requires a declared fixture')
    hooks = req.setdefault('engine_hooks', dict(schema=SCHEMA))
    if case == 'reserve':
        hooks['reserve_cap'] = True
    elif case == 'development':
        hooks.update(leaf_development=True)
        hooks.setdefault('parameters', {}).update(leaf_parameters(38., .001))
        hooks['parameters'].update(
            LEAFLET_MID_DD=parameter(250., 'degree_day', 'F4 expansion sensitivity; not fitted'),
            LEAFLET_K=parameter(.014, '1/degree_day', 'F4 expansion sensitivity; not fitted'),
            PHYLLOCHRON_DD=parameter(77., 'degree_day', 'F4 phyllochron sensitivity; no initial-state retiming'))
    elif case == 'off':
        hooks['reserve_cap'] = False
        require(not (cap.PARAMETERS | {'LEAFLET_MID_DD','LEAFLET_K','PHYLLOCHRON_DD'}).intersection(hooks.get('parameters',{})),
                'off baseline cannot contain new parameters')
    else:
        raise ValueError(case)
    write_json(inputs/'request.json', req)
    prepare(inputs, out/'native')
    write_json(out/'fixture_manifest.json', dict(case=case,
        source=str(host.resolve()) if host else 'vtc_development.demo explicit synthetic seed fixture',
        initial_reserve_changed=False, initial_excess_policy=cap.INITIAL_POLICY,
        biological_validation='not_established', days=req['intervals']*req['dt_s']/86400))
    from tools.tomato_fsp_pipeline import retention_policy
    write_json(out/'retention_policy.json', retention_policy('working_last_1'))
    return out/'native'


def audit(native: Path) -> dict:
    from tools.vtc_development import verify_prepared
    prepared, pkg = verify_prepared(native)
    req = pkg['request']; working = native/'working'
    require(read_json(native/'execution.json')['status'] == 'verified', 'native validation must precede audit')
    masses = defaultdict(float)
    for row in rows(working/'states.csv'):
        masses[int(row['day']), row['plant_instance_id']] += float(row['mass_gDM'])
    demands = defaultdict(float);maintenance_by_kind=defaultdict(float);structure_by_kind=defaultdict(float)
    state_kinds={(int(s['day']),s['organ_id']):s['class'] for s in rows(working/'states.csv')}
    for row in rows(working/'flux.csv'):
        demands[int(row['interval']), row['plant_instance_id']] += float(row['demand_gC'])
        kind=state_kinds[int(row['interval'])+1,row['organ_id']]
        maintenance_by_kind[int(row['interval']),row['plant_instance_id'],kind]+=float(row['maintenance_gC'])
        structure_by_kind[int(row['interval']),row['plant_instance_id'],kind]+=float(row['structural_gC'])
    records = []; total_removed = 0.; max_residual = 0.; max_relative_residual = 0.
    from tools import fsp_temperature as temperature
    cf = req['parameters']['CARBON_FRACTION']['value']
    fraction = cap.parameters(req)['RESERVE_CAP_FRACTION']['value']
    for row in rows(working/'ledger.csv'):
        interval = int(row['interval']); pid = row['plant_instance_id']
        previous_mass, end_mass = masses[interval,pid], masses[interval+1,pid]
        removed = cap.validate_interval(req, previous_mass, end_mass, row, demands[interval,pid])
        start, end = float(row['reserve_start_gC']), float(row['reserve_end_gC'])
        residual = start + float(row['gross_gC']) - removed - sum(float(row[k]) for k in (
            'maintenance_gC','structural_gC','growth_resp_gC')) - end
        require(abs(residual) <= req['runtime']['balance_tolerance_gC'], 'audited carbon closure failed')
        from tools import fsp_stage_parameters as stage
        if stage.active(req) or cap.floor_active(req) or temperature.active(req):
            scale=max(abs(start),abs(end),abs(float(row['gross_gC'])),abs(float(row['maintenance_gC'])),abs(float(row['structural_gC'])),abs(float(row['growth_resp_gC'])))
            require(abs(residual)<=1e-12*scale if scale else residual==0,'hook relative carbon closure failed')
            max_relative_residual=max(max_relative_residual,abs(residual)/scale if scale else 0.)
            import math
            for column,by_kind in (('maintenance_gC',maintenance_by_kind),('structural_gC',structure_by_kind)):
                total=sum(value for (i,p,_),value in by_kind.items() if i==interval and p==pid)
                require(math.isclose(total,float(row[column]),rel_tol=1e-12,abs_tol=0.),'S1 organ-kind sum '+column)
        total_removed += removed; max_residual = max(max_residual, abs(residual))
        start_terms = cap.cap_terms(req, previous_mass, start)
        end_terms = cap.cap_terms(req, end_mass, end)
        records.append(dict(interval=interval, plant_instance_id=pid, organ_mass_start_gDM=previous_mass,
            organ_mass_end_gDM=end_mass, reserve_start_gC=start, reserve_end_gC=end,
            cap_start_gC=start_terms['cap_gC'], cap_end_gC=end_terms['cap_gC'],
            initial_excess_active=cap.active(req) and start>start_terms['cap_gC'],
            potential_gross_gC=row['gross_gC'], effective_gross_gC=float(row['gross_gC'])-removed,
            downregulated_gC=removed, residual_gC=residual))
        if cap.floor_active(req):
            records[-1].update(cap_floor_binding=start_terms['cap_floor_binding'],
                fractional_cap_start_gC=start_terms['fractional_cap_gC'], cap_floor_gC=start_terms['cap_floor_gC'],
                cap_floor_binding_end=end_terms['cap_floor_binding'], fractional_cap_end_gC=end_terms['fractional_cap_gC'])
    write_csv(native.parent/'carbon_audit.csv', records)
    result = dict(status='verified', reserve_cap=cap.active(req), intervals=len(records),
        initial_excess_intervals=sum(r['initial_excess_active'] for r in records),
        downregulated_gC=total_removed, max_abs_residual_gC=max_residual,
        binding=prepared['binding'], model_hash=prepared['model_hash'], parameter_hash=prepared['parameter_hash'],
        audit_sha256=sha(native.parent/'carbon_audit.csv'), storage=storage_report(native.parent))
    if cap.floor_active(req):
        result.update(reserve_cap_floor=True, cap_floor_gC=cap.floor_value(req),
            cap_floor_binding_intervals=sum(r['cap_floor_binding'] for r in records),
            cap_floor_binding_end_intervals=sum(r['cap_floor_binding_end'] for r in records),
            min_fractional_cap_gC_by_plant={pid:min(min(r['fractional_cap_start_gC'],r['fractional_cap_end_gC'])
                for r in records if r['plant_instance_id']==pid) for pid in sorted({r['plant_instance_id'] for r in records})})
    if temperature.active(req):
        result.update(temperature_response=temperature.metadata(req),
            physiology_temperature_response=req['physiology_temperature_response'],
            carbon_relative_tolerance=1e-12,max_relative_carbon_residual=max_relative_residual)
    write_json(native.parent/'audit.json', result)
    if stage.active(req) or temperature.active(req):
        write_csv(native.parent/'carbon_audit_by_kind.csv',[dict(interval=i,plant_instance_id=p,organ_kind=k,maintenance_gC=value,structural_gC=structure_by_kind[i,p,k]) for (i,p,k),value in sorted(maintenance_by_kind.items())])
        stage.export_diagnostics(working,native.parent,pkg)
    return result


def prepare_mature_case(out: Path, host: Path, day: int) -> Path:
    """Import an actual synthetic native snapshot without changing its reserves."""
    from tools.vtc_development_observation import fixture_from_native
    from tools.vtc_development import prepare
    from tools.tomato_fsp_pipeline import retention_policy
    fresh_build(out)
    inputs = fixture_from_native(host, out/'inputs', day=day, intervals=30)
    req = read_json(inputs/'request.json')
    req['engine_hooks'] = dict(schema=SCHEMA, reserve_cap=True)
    write_json(inputs/'request.json', req)
    prepare(inputs, out/'native')
    write_json(out/'retention_policy.json', retention_policy('working_last_1'))
    write_json(out/'fixture_manifest.json', dict(case='mature_reserve', source=str(host.resolve()),
        snapshot_day=day, initial_reserve_changed=False, biological_validation='not_established', days=30))
    return out/'native'


def run_case(native: Path, timeout_s: float = 2400) -> dict:
    require(shutil.disk_usage(ROOT).free >= 10_000_000_000, 'stop: less than 10 GB free')
    req = read_json(native/'inputs/request.json')
    if req.get('integration_contract'):
        from tools.fsp_engine_hooks_acceptance import run_case as run_integrated
        run_integrated(native, timeout_s)
    else:
        from tools.vtc_growth_bridge import run
        def check_space(boundary: int) -> None:
            require(shutil.disk_usage(ROOT).free >= 10_000_000_000, 'stop: less than 10 GB free')
        run(native, timeout_s=timeout_s, on_interval=check_space, fail_closed_interval_hook=True)
    return audit(native)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case', choices=('reserve','development','off'), required=True); p.add_argument('--host', type=Path)
    p = sub.add_parser('run'); p.add_argument('--native', type=Path, required=True); p.add_argument('--timeout', type=float, default=2400)
    p = sub.add_parser('prepare-mature'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--host', type=Path, required=True); p.add_argument('--day', type=int, default=45)
    p = sub.add_parser('audit'); p.add_argument('--native', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        print(prepare_case(args.out, args.case, args.host))
    elif args.operation == 'prepare-mature':
        print(prepare_mature_case(args.out, args.host, args.day))
    else:
        print(json.dumps(run_case(args.native,args.timeout) if args.operation=='run' else audit(args.native), indent=2))


if __name__ == '__main__':
    main()
