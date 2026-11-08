"""Reproducible native hook acceptance using a frozen Session F host/plan."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import csv
import math
import re
import hashlib
import time

from tools.fsp_engine_hooks import SCHEMA, SWITCHES, planned, AXILLARY_STAGE, axillary_stage
from tools.vtc_growth_io import ROOT, read_json, write_json, rows, write_csv, sha, require, fresh_build

# Numeric measures only. Identity, categorical, integer clock/index and unknown
# fields remain exact even if a particular value looks like a number.
NUMERIC_COLUMNS = {
    'absorbed_umol_s','gross_gC','maintenance_gC','target_gDM','demand_gC','allocated_gC','structural_gC',
    'growth_resp_gC','reserve_start_gC','reserve_end_gC','deficit_gC','residual_gC','mass_before_gDM',
    'available_gC','area_m2','thermal_age_dd','mass_gDM','max_mass_gDM','x_m','y_m','z_m',
    'tip_x_m','tip_y_m','tip_z_m','length_m','width_m','rachis_bend_alpha_deg','azimuth_deg',
    'elevation_deg','plant_thermal_dd'}


def parameter(value, unit: str, source: str) -> dict:
    return dict(value=value, unit=unit, source_kind='user_assumed', source_ref=source,
                cultivar='Micro-Tom_unfitted_hypothesis', uncertainty='not biologically validated')


def leaf_parameters(stagger: float, decay: float) -> dict:
    return dict(LEAFLET_GROWTH_ORDER=parameter('basipetal', 'categorical', 'G1 user growth order'),
                LEAFLET_STAGGER_DD=parameter(stagger, 'degree_day', 'G3 declared scenario'),
                LEAF_AGEING_DECAY_A=parameter(decay, '1/degree_day', 'G7 user equation'))


def fixture_axes(inputs: Path, request: dict, plan: Path) -> None:
    """Explicit fixture-only adapter; original P1 and host run remain immutable."""
    require(request['fixture'] is True, 'acceptance adapter is fixture-only')
    from tools.microtom_parameters import load_registry, engine_value_for
    new_axis_prior = next(r for r in load_registry() if r['key']=='AXIS_LEAVES_TO_TRUSS_NEW_AXIS')
    new_axis_first = engine_value_for(dict(new_axis_prior, engine_key='FIRST_LEAVES'), new_axis_prior['value'])
    states = rows(inputs / 'initial_state.csv'); controls = read_json(inputs / 'initial_controls.json')
    axes = []
    for pid, control in controls.items():
        axis = next(s for s in states if s['plant_instance_id'] == pid and s['class'] == 'Axis')
        axes.append(dict(plant_instance_id=pid, axis_id='M', order=0, parent_axis_id='',
                         parent_node_id='', parent_rank=0, stage='main', axis_vertex_id=axis['organ_id'].split('|', 1)[1],
                         source_kind='synthetic_assumed', source_ref='unchanged Session A synthetic host; explicit single-axis adapter',
                         parameters=dict(SUBSTEM_SINK_SCALE=parameter(1., 'dimensionless', 'legacy unity allocation'),
                                         APPLY_DETERMINATE_STOP=parameter(True, 'boolean', 'existing shared stop rule')),
                         initialization=dict(source_kind='synthetic_assumed', source_ref='frozen host initial_controls.json',
                            stem_mass_gDM=0., stem_width_m=0., apex_stopped=False,
                            **{k: control[k] for k in ('thermal_dd', 'produced', 'leaf_order', 'truss_count', 'axis_leaf_count')}),
                         segment_index=1, truss_rule_status='available', truss_history=[]))
        require(control['truss_count'] == 0, 'fixture adapter requires no initial truss history')
        controls[pid] = dict(control, axes={'M': deepcopy(control)})
        for s in states:
            if s['plant_instance_id'] == pid:
                s.update(axis_id='M', developmental_axis_id=axis['organ_id'] if s['class'] != 'Plant' else '',
                         parent_rank=0, segment_index=1 if s['class'] not in ('Plant', 'Root') else 'unavailable')
    for r in rows(plan):
        if r['scenario_id'] != 'base' or r['member_index'] != '0' or r['emerged'] != 'True':
            continue
        pid = r['plant_instance_id']; rank = int(r['parent_rank'])
        leaf = next(s for s in states if s['organ_id'] == r['subtending_leaf_id'])
        parent = next(s for s in states if s['organ_id'] == leaf['parent_id'])
        # A4 permits the observed compound vertex itself to name an axil when
        # no separate Metamer/Attachment vertex was exported by P1.
        if parent['class'] != 'Attachment':
            parent = leaf
        require(int(parent['true_leaf_order']) == rank, 'plan parent rank mismatch')
        axes.append(dict(plant_instance_id=pid, axis_id=r['new_axis_id'], order=int(r['order']),
                         parent_axis_id=r['parent_axis'], parent_node_id=parent['organ_id'].split('|', 1)[1],
                         parent_rank=rank, stage='axis_shoot', origin='axillary', provenance='model_generated',
                         forecast_axis_id=r['forecast_axis_id'], emergence_day=float(r['emergence_day']),
                         emergence_semantics='bud_break',
                         source_kind='user_assumed', source_ref=str(plan.resolve()) + ':base:member0', plan_sha256=sha(plan),
                         parameters=dict(SUBSTEM_SINK_SCALE=parameter(1., 'dimensionless', 'legacy unity allocation'),
                                         APPLY_DETERMINATE_STOP=parameter(True, 'boolean', 'shared development rules'),
                                         FIRST_LEAVES=parameter(new_axis_first, 'count', 'Session F registered new-axis base candidate; existing engine_value_for U5 conversion'),
                                         CADENCE=deepcopy(request['parameters']['CADENCE'])),
                         initialization=dict(source_kind='synthetic_assumed', source_ref='H7 zero local clock; stage selected by AXILLARY_START_STAGE',
                             thermal_dd=0., stem_mass_gDM=0., stem_width_m=0., produced=0, leaf_order=0,
                             truss_count=0, axis_leaf_count=0, apex_stopped=False),
                         segment_index=1, truss_rule_status='available', truss_history=[]))
        axes[-1]['parameters']['CADENCE']['unit'] = 'count'
    request.update(axis_contract='fsp-axes-1', topology_scope='provided_axes', topology_completeness='unavailable', axes=axes)
    write_csv(inputs / 'initial_state.csv', states); write_json(inputs / 'initial_controls.json', controls)
    initialization = read_json(inputs / 'initialization.json')
    initialization['files'] = {f: sha(inputs / f) for f in initialization['files']}
    write_json(inputs / 'initialization.json', initialization)


def prepare_case(host: Path, out: Path, case: str, plan: Path | None = None, stage: str = 'bud_break') -> Path:
    from tools.vtc_development import prepare
    fresh_build(out); inputs = out / 'inputs'; shutil.copytree(host / 'inputs', inputs)
    request = read_json(inputs / 'request.json')
    request['engine_hooks'] = dict(schema=SCHEMA, **{k: False for k in sorted(SWITCHES)})
    if case == 'h7':
        require(plan is not None, 'Session F plan required')
        request['engine_hooks'].update(planned_axes=True, per_axis_development=True)
        request['engine_hooks']['parameters'] = {AXILLARY_STAGE: parameter(stage, 'categorical', 'Reviewed engine hooks addendum decision 2026-10-15')}
        fixture_axes(inputs, request, plan)
    elif case in ('g_zero', 'g_on'):
        request['engine_hooks'].update(leaf_development=True,
            parameters=leaf_parameters(0. if case == 'g_zero' else 38., 0. if case == 'g_zero' else .002))
    write_json(inputs / 'request.json', request)
    write_json(out / 'fixture_manifest.json', dict(source_native_inputs=str(host / 'inputs'),
        source_hashes={p.name: sha(p) for p in (host / 'inputs').iterdir() if p.is_file()},
        note='Original p1_state_20261011_01 is inconsistent with Session A scenarios: 8 true leaves and no truss; built under FIRST_LEAVES 9. Original files unchanged.',
        case=case, new_axis_clock='zero at emergence; no fabricated initial leaves',
        axillary_start_stage=axillary_stage(request) if case == 'h7' else 'disabled', emergence_semantics='bud_break'))
    prepare(inputs, out / 'native')
    from tools.tomato_fsp_pipeline import retention_policy
    policy = retention_policy('working_last_1')
    policy.update(committed_steps='hash-verified acceptance interval slices; full semantic trajectory validation at completion',
                  geometry_streams='final cumulative streams and newest checkpoint retained for regression/shading')
    write_json(out / 'retention_policy.json', policy)
    return out / 'native'


def prepare_replay(host: Path, out: Path) -> Path:
    """Replay frozen input bytes, including JSON key order and provenance."""
    from tools.vtc_development import prepare, verify_prepared
    original, _ = verify_prepared(host)
    fresh_build(out)
    shutil.copytree(host/'inputs',out/'inputs')
    prepare(out/'inputs',out/'native')
    replay = read_json(out/'native/prepared.json')
    require(original['frozen_hashes']==replay['frozen_hashes'], 'replay frozen input bytes changed')
    require(original['model_hash']==replay['model_hash'], 'replay native model changed')
    shutil.copyfile(host.parent/'retention_policy.json',out/'retention_policy.json')
    write_json(out/'replay_manifest.json',dict(source=str(host.resolve()), source_prepared_sha256=sha(host/'prepared.json'),
        frozen_inputs_byte_identical=True, native_model_byte_identical=True))
    return out/'native'


def storage_report(folder: Path) -> dict:
    files = [p for p in folder.rglob('*') if p.is_file()]
    result = dict(folder=str(folder.resolve()), bytes=sum(p.stat().st_size for p in files), files=len(files),
                  working_checkpoints=len(list((folder/'native/working/checkpoints').glob('step_*'))))
    write_json(folder/'storage.json', result)
    return result


def commit_acceptance_interval(native: Path, boundary: int) -> dict:
    """Commit exact CSV interval slices, with hash lineage, before pruning.

    These are acceptance archives, not the pipeline's P2/P3/P4 publication.
    Prefix bytes are verified at each boundary; independent biological/numeric
    contracts are checked by the native runner on the complete trajectory.
    """
    from tools.vtc_growth_io import rename_with_retry
    folder = native.parent.resolve()
    require(folder.is_relative_to((ROOT/'build').resolve()), 'acceptance retention must stay inside project build')
    checkpoints = native/'working/checkpoints'
    require(checkpoints.resolve().is_relative_to(folder), 'checkpoint path escapes acceptance run')
    checkpoint = checkpoints/f'step_{boundary}'
    require(checkpoint.resolve().parent == checkpoints.resolve(), 'checkpoint link escapes run')
    completion = read_json(checkpoint/'completion.json')
    binding = read_json(native/'prepared.json')['binding']
    require(completion['intervals'] == boundary and completion.get('checkpoint') is True, 'wrong checkpoint boundary')
    require(all(completion[k] == v for k,v in binding.items()), 'checkpoint binding mismatch')
    root = folder/'committed'; root.mkdir(exist_ok=True)
    previous = root/f'step_{boundary-1:04d}'/'checkpoint.json'
    require(boundary == 1 or previous.is_file(), 'missing acceptance predecessor')
    target = root/f'step_{boundary:04d}'
    require(not target.exists(), 'acceptance boundary already committed')
    attempt = root/f'.step_{boundary:04d}_partial'; attempt.mkdir()
    selected = {}; sizes = {}; started = time.perf_counter()
    prior = read_json(previous) if boundary>1 else None
    for name, expected in completion['files'].items():
        require(Path(name).name == name and name.endswith('.csv'), 'unsafe checkpoint file name')
        hasher = hashlib.sha256()
        with (checkpoint/name).open('rb') as source, (attempt/name).open('wb') as output:
            header = source.readline(); hasher.update(header); output.write(header)
            key = header.split(b',',1)[0].decode('ascii')
            cumulative = key in ('day','state','interval')
            # Verify the immutable predecessor prefix, then copy only appended
            # bytes. No reparsing of millions of old mesh rows at every step.
            old_size = prior['native_file_sizes'][name] if prior and cumulative else len(header)
            remaining = old_size-len(header)
            require(remaining>=0, 'checkpoint header changed')
            while remaining:
                chunk=source.read(min(1024*1024,remaining))
                require(bool(chunk), 'checkpoint prefix truncated')
                hasher.update(chunk); remaining-=len(chunk)
            if prior and cumulative:
                require(hasher.hexdigest()==prior['native_completion']['files'][name], 'checkpoint rewrote predecessor prefix: '+name)
            for chunk in iter(lambda:source.read(1024*1024),b''):
                hasher.update(chunk);output.write(chunk)
            sizes[name]=source.tell()
        require(hasher.hexdigest() == expected, 'checkpoint CSV hash mismatch: '+name)
        selected[name] = sha(attempt/name)
    receipt = dict(schema='engine-hooks-acceptance-commit-1', status='hash_verified_interval_archive', boundary=boundary,
        semantic_validation='pending_complete_trajectory', files=selected, native_file_sizes=sizes,
        native_completion=completion, native_completion_sha256=sha(checkpoint/'completion.json'),
        previous_sha256=sha(previous) if boundary>1 else sha(native/'prepared.json'), elapsed_s=time.perf_counter()-started)
    write_json(attempt/'checkpoint.json',receipt)
    rename_with_retry(attempt,target)
    return receipt


def verify_acceptance_archive(native: Path, source: Path | None = None) -> None:
    """Reconstruct each cumulative file's hash from committed slices, no copy."""
    complete = read_json((source or native/'working')/'completion.json')
    root = native.parent/'committed'
    previous = sha(native/'prepared.json')
    manifests = []
    for k in range(1,complete['intervals']+1):
        step = root/f'step_{k:04d}'; meta = read_json(step/'checkpoint.json')
        require(meta['previous_sha256'] == previous, 'acceptance lineage changed')
        require(all(sha(step/name)==h for name,h in meta['files'].items()), 'committed interval changed')
        previous=sha(step/'checkpoint.json'); manifests.append(step)
    for name, expected in complete['files'].items():
        if name not in read_json(manifests[-1]/'checkpoint.json')['files']:
            continue  # Final-only sensor table is covered by native validation.
        hasher = hashlib.sha256()
        for i,step in enumerate(manifests):
            with (step/name).open('rb') as stream:
                header=stream.readline()
                key=header.split(b',',1)[0]
                if key not in (b'day',b'state',b'interval'):
                    if i==len(manifests)-1: hasher.update(header); hasher.update(stream.read())
                    continue
                if i==0: hasher.update(header)
                for chunk in iter(lambda: stream.read(1024*1024),b''): hasher.update(chunk)
        require(hasher.hexdigest()==expected, 'committed slices do not reproduce final '+name)


def run_case(native: Path, timeout_s: float = 1800) -> dict:
    from tools.vtc_growth_bridge import run
    from tools.tomato_fsp_pipeline import prune_working_checkpoints
    policy = read_json(native.parent/'retention_policy.json')
    require(policy['value']=='working_last_1', 'acceptance requires working_last_1')
    def on_interval(k: int) -> None:
        commit_acceptance_interval(native,k)
        prune_working_checkpoints(native.parent,1,k)
        write_json(native.parent/'acceptance_progress.json',dict(committed_boundary=k, status='running'))
    try:
        result = run(native, timeout_s=timeout_s, on_interval=on_interval, fail_closed_interval_hook=True)
        verify_acceptance_archive(native)
        write_json(native.parent/'acceptance_validation.json',dict(status='verified', native_execution_sha256=sha(native/'execution.json'),
            semantics='independent complete native trajectory validation passed', committed_archive='reproduces native cumulative CSV hashes'))
        write_json(native.parent/'acceptance_progress.json',dict(committed_boundary=read_json(native/'inputs/request.json')['intervals'],status='verified'))
        return result
    finally:
        storage_report(native.parent)


def summarize(folder: Path) -> dict:
    from tools.fsp_axes_segments import axis_inventory
    req = read_json(folder / 'inputs/request.json'); source = folder / 'working'
    states = rows(source / 'states.csv'); flux = rows(source / 'flux.csv'); events = rows(source / 'events.csv')
    counts = []
    for day in (0, 10, 18, 30):
        ss = [s for s in states if int(s['day']) == day]
        counts.append(dict(day=day, organs=len(ss), axes=sum(s['class'] == 'Axis' for s in ss),
                           compounds=sum(s['class'] == 'CompoundLeaf' for s in ss), leaflets=sum(s['class'] == 'Leaflet' for s in ss)))
    result = dict(counts=counts, axillary_emergences=sum(e['event'] == 'axillary_emergence' for e in events))
    if req.get('axes'):
        ids = {a['axis_id'] for a in req['axes'] if planned(a)}
        result['new_axes'] = {key: sum(float(r[key]) for r in flux if r['axis_id'] in ids)
                              for key in ('absorbed_umol_s', 'gross_gC', 'allocated_gC', 'structural_gC')}
        environment=rows(folder/'inputs/environment.csv')
        result['new_axes']['absorbed_umol']=sum(float(r['absorbed_umol_s'])*float(environment[int(r['interval'])]['light_hours'])*3600 for r in flux if r['axis_id'] in ids)
        result['new_axes']['absorbed_umol_s_note']='Sum of interval rate samples, not an integrated photon amount; use absorbed_umol.'
        inventory = []
        apices = rows(source / 'apices.csv')
        for day in range(req['intervals'] + 1):
            inventory.extend(dict(day=day, **r) for r in axis_inventory(req,
                [s for s in states if int(s['day']) == day], [s for s in apices if int(s['day']) == day]))
        write_csv(folder.parent / 'axis_inventory.csv', inventory)
    write_csv(folder.parent / 'organ_counts.csv', counts)
    write_json(folder.parent / 'summary.json', result)
    return result


def regression(host: Path, native: Path) -> dict:
    """Retain byte failures; do not quietly replace the specified criterion."""
    old_source = host / ('partial' if (host/'partial/completion.json').is_file() else 'working')
    new_source = native / ('working' if (native / 'working/completion.json').exists() else 'partial')
    old_completion = old_source / 'completion.json'
    if not old_completion.exists():
        old_completion = host / 'results/completion.json'
    old = read_json(old_completion); new = read_json(new_source / 'completion.json')
    result = {}
    for name, expected in old['files'].items():
        actual = sha(new_source / name)
        require(actual == new['files'][name], 'new regression file hash mismatch')
        record = dict(byte_equal=actual == expected, reference_sha256=expected, actual_sha256=actual,
                      reference_method='retained_file' if (old_source / name).exists() else 'issued_verified_manifest')
        if (old_source / name).exists():
            require(sha(old_source / name) == expected, 'reference file hash mismatch')
        if not record['byte_equal'] and (old_source / name).exists():
            differences = {}; nonnumeric = 0; count_old = count_new = 0
            with (old_source / name).open(newline='',encoding='utf-8-sig') as a, (new_source / name).open(newline='',encoding='utf-8-sig') as b:
                from itertools import zip_longest
                old_reader, new_reader = csv.DictReader(a), csv.DictReader(b)
                record['headers_equal'] = old_reader.fieldnames == new_reader.fieldnames
                for x, y in zip_longest(old_reader,new_reader):
                    count_old += x is not None; count_new += y is not None
                    if x is None or y is None:
                        nonnumeric += 1; continue
                    for key in set(x) | set(y):
                        if x.get(key) == y.get(key): continue
                        if key not in NUMERIC_COLUMNS:
                            nonnumeric += 1; continue
                        try:
                            delta = abs(float(x[key]) - float(y[key]))
                            require(math.isfinite(delta), 'nonfinite regression difference')
                            differences[key] = max(differences.get(key,0.), delta)
                        except (ValueError, TypeError, KeyError): nonnumeric += 1
            record.update(max_abs_numeric_difference=differences, nonnumeric_differences=nonnumeric, rows_old=count_old, rows_new=count_new)
        result[name] = record
    def normalized(path: Path) -> str:
        return re.sub(r'(inputHash|engineHash)="[0-9a-f]+"', r'\1="BINDING"', path.read_text(encoding='utf-8'))
    report = dict(all_csv_bytes_equal=all(r['byte_equal'] for r in result.values()),
        model_equal_except_provenance_hashes=normalized(host/'working/Model.rgg')==normalized(native/'working/Model.rgg'),
        files=result, acceptance='strict byte criterion retained; any unequal file is a failure')
    write_json(native.parent / 'byte_regression.json', report)
    return report


def regression_decision(baseline_pair: dict, candidate: dict) -> dict:
    """Apply the addendum's conditional rule, never infer tolerance beforehand."""
    def changed(report):
        return {name:r for name,r in report['files'].items() if not r['byte_equal']}
    def numeric_only(records, tolerance):
        return all(r.get('reference_method')=='retained_file' and r.get('headers_equal') is True
            and r.get('rows_old')==r.get('rows_new') and r.get('nonnumeric_differences')==0
            and bool(r.get('max_abs_numeric_difference'))
            and all(math.isfinite(v) and v<=tolerance for v in r['max_abs_numeric_difference'].values()) for r in records.values())
    baseline = changed(baseline_pair); actual = changed(candidate)
    stochastic = {'flux.csv','ledger.csv','proposals.csv','sensors.csv'}
    observed_roundoff = bool(baseline) and set(baseline)<=stochastic and numeric_only(baseline,1e-15)
    models_equal = baseline_pair['model_equal_except_provenance_hashes'] and candidate['model_equal_except_provenance_hashes']
    if observed_roundoff:
        accepted = models_equal and set(actual)<=set(baseline) and numeric_only(actual,1e-12)
        rule='absolute_numeric_delta_le_1e-12; exact headers, row counts and nonnumeric fields; changed files restricted to baseline evidence'
    else:
        accepted = models_equal and not baseline and not actual
        rule='strict_bytes; tolerance not authorized by baseline replay evidence'
    return dict(accepted=accepted, rule=rule, baseline_roundoff_observed=observed_roundoff,
                baseline_changed_files=sorted(baseline), candidate_changed_files=sorted(actual))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--host', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--plan', type=Path)
    p.add_argument('--case', choices=('off', 'h7', 'g_zero', 'g_on'), required=True)
    p.add_argument('--stage', choices=('bud_break','germination'), default='bud_break')
    p = sub.add_parser('run'); p.add_argument('--native', type=Path, required=True); p.add_argument('--timeout', type=float, default=1800)
    p = sub.add_parser('replay'); p.add_argument('--host', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--timeout', type=float, default=1800)
    p = sub.add_parser('summarize'); p.add_argument('--native', type=Path, required=True)
    p = sub.add_parser('regression'); p.add_argument('--host', type=Path, required=True); p.add_argument('--native', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'prepare':
        print(prepare_case(args.host, args.out, args.case, args.plan, args.stage))
    elif args.operation == 'run':
        print(run_case(args.native,args.timeout))
    elif args.operation == 'replay':
        print(run_case(prepare_replay(args.host,args.out),args.timeout))
    elif args.operation == 'summarize':
        print(summarize(args.native))
    else:
        print(regression(args.host,args.native))


if __name__ == '__main__':
    main()
