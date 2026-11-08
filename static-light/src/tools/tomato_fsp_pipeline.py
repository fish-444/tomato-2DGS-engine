"""P1 consumer -> one native P2/P4/P3 trajectory -> atomic interval checkpoints."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tools.vtc_growth_io import ROOT,require,ContractError,sha,digest,read_json,write_json,rows,write_csv,fresh_build,aware
from tools.tomato_fsp_io import SCHEMA,P1_FILES,audit_p1_package,selected_package,make_native_inputs
from tools.vtc_development import prepare as native_prepare,verify_prepared
from tools.vtc_development_io import validate_outputs
from tools.vtc_growth_bridge import run as native_run

# F11 C1: checkpoint retention is a declared parameter (tools/microtom_parameters.csv), recorded per run.
RETENTION_KEY = 'CHECKPOINT_RETENTION'
RETENTION_VALUES = {'working_last_1': 1, 'working_last_2': 2}


def retention_policy(override: str | None = None) -> dict:
    from tools.microtom_parameters import load_registry, registry_hash
    row = next((r for r in load_registry() if r['key'] == RETENTION_KEY), None)
    require(row is not None, f'{RETENTION_KEY} missing from tools/microtom_parameters.csv')
    value = override or row['value']
    require(value in RETENTION_VALUES, f'unknown {RETENTION_KEY}: {value!r} (allowed {sorted(RETENTION_VALUES)})')
    return dict(key=RETENTION_KEY, value=value, keep_working_checkpoints=RETENTION_VALUES[value],
                source_kind='user_stated', source_ref=row['source_ref'] if override is None else f'run override of registry value {row["value"]}',
                registry_value=row['value'], registry_sha256=registry_hash(), committed_steps='all',
                effective_retention='retain_pending_move_review',
                geometry_streams='retained; FILE_REMOVAL_RULE requires a separate reference/age review before moving')


def checkpoint_number(path: Path) -> int:
    return int(path.name.split('_')[1])


def working_checkpoints(folder: Path) -> dict[int, Path]:
    return {checkpoint_number(c.parent): c.parent for c in (folder / 'native/working/checkpoints').glob('step_*/completion.json')}


MOVE_REVIEW_CHECKS = ('active_claims', 'flow_rows', 'tests', 'kept_runs', 'base_requests', 'running_processes', 'link_targets')
MOVE_MIN_AGE_S = 600  # FILE_REMOVAL_RULE_2026-10-17, not a scientific parameter.


def _check_move_audit_path(path: Path) -> None:
    require(path == path.resolve() and not path.is_symlink() and
            (not path.exists() or (path.is_file() and path.stat().st_nlink == 1)), 'unsafe linked/non-file move audit path')


def prune_working_checkpoints(folder: Path, keep: int, upto: int, *, move_review: Path | None = None) -> list[str]:
    """Move reviewed, committed checkpoints to storage/TRASH; retain everything by default.

    A current caller-authored review must attest each reference check with evidence,
    and bind the complete source inventory plus committed receipt by SHA-256. This
    helper cannot infer absence of active jobs or external references from filenames.
    Runtime three-argument calls therefore never remove files. No overwrite/delete.
    """
    require(isinstance(keep, int) and keep >= 1, 'keep must be positive')
    if move_review is None:
        return []
    project = ROOT.resolve(); storage = project.parent.parent
    folder = folder.absolute()
    require(folder == folder.resolve() and folder.is_relative_to(project/'build'), 'retention run escapes project build or uses a link')
    review_sha = sha(move_review); review = read_json(move_review)
    require(sha(move_review) == review_sha, 'move review changed during read')
    require(review.get('run_path') == str(folder) and bool(review.get('reviewer')), 'move review belongs to another run or lacks reviewer')
    reviewed = aware(review['reviewed_at']).timestamp()
    require(0 <= time.time()-reviewed < MOVE_MIN_AGE_S, 'move review is stale')
    require(review.get('schema') == 'checkpoint-move-review-1', 'unknown move review')
    for check in MOVE_REVIEW_CHECKS:
        item = review.get('checks', {}).get(check, {})
        require(item.get('clear') is True and bool(item.get('evidence')), 'missing reference review: '+check)
    moved = []
    for k, path in sorted(working_checkpoints(folder).items()):
        receipt = folder/'committed'/f'step_{k:04d}'/'checkpoint.json'
        entry = review.get('checkpoints', {}).get(path.name)
        if k > upto-keep or not receipt.is_file() or entry is None:
            continue
        require(path == path.resolve() and path.parent == folder/'native/working/checkpoints', 'checkpoint escapes run or uses a link')
        files = sorted(path.rglob('*'))
        require(all(p == p.resolve() and not p.is_symlink() and (not p.is_file() or p.stat().st_nlink == 1) for p in files), 'checkpoint contains a link')
        require(all(time.time()-p.stat().st_mtime >= MOVE_MIN_AGE_S for p in [path, *files]), 'checkpoint modified within 10 minutes')
        inventory = {str(p.relative_to(path)): dict(sha256=sha(p), size_bytes=p.stat().st_size) for p in files if p.is_file()}
        require(inventory == entry.get('files'), 'reviewed checkpoint inventory changed')
        require(sha(receipt) == entry.get('committed_sha256'), 'reviewed committed receipt changed')
        meta = read_json(receipt)
        if meta.get('schema') == 'engine-hooks-acceptance-commit-1':
            from tools.fsp_engine_hooks_acceptance import verify_acceptance_archive
            require(meta['native_completion_sha256'] == sha(path/'completion.json'), 'checkpoint completion changed')
            verify_acceptance_archive(folder/'native', source=path)
        else:
            for boundary in range(1, k+1):
                verify_committed(folder, boundary, path if boundary == k else None)
        trash = storage/'TRASH'; target = trash/path.relative_to(storage)
        require(trash == trash.resolve() and target == target.resolve() and target.is_relative_to(trash), 'TRASH path escapes storage or uses a link')
        require(not target.exists(), 'TRASH destination already exists')
        total = sum(v['size_bytes'] for v in inventory.values())
        reason = f'reviewed superseded working checkpoint; {total} bytes; committed slices verified; review={move_review.resolve()}; files={json.dumps(inventory, sort_keys=True)}'
        record = dict(moved_at=datetime.now(timezone.utc).isoformat(), source_path=str(path), trash_path=str(target), reason=reason)
        journal = folder/f'retention_move_{uuid.uuid4().hex}.json'
        write_json(journal, dict(status='planned', **record, review_sha256=review_sha))
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = trash/'MANIFEST.csv'
        note = folder/f'MOVED_{datetime.now().strftime("%Y%m%d")}.txt'
        for audit_path in (manifest, note): _check_move_audit_path(audit_path)
        fields = ['moved_at', 'source_path', 'trash_path', 'reason']
        if manifest.exists():
            with manifest.open(encoding='utf-8-sig', newline='') as stream:
                require(next(csv.reader(stream), None) == fields, 'unexpected TRASH manifest columns')
        # Repeat bounds/inventory immediately before the same-volume rename.
        require(path == path.resolve() and target == target.resolve() and not target.exists(), 'move path changed')
        require(all(time.time()-p.stat().st_mtime >= MOVE_MIN_AGE_S for p in [path, *path.rglob('*')]), 'checkpoint modified before move')
        require(inventory == {str(p.relative_to(path)): dict(sha256=sha(p), size_bytes=p.stat().st_size) for p in sorted(path.rglob('*')) if p.is_file()}, 'checkpoint changed before move')
        require(sha(move_review) == review_sha and 0 <= time.time()-reviewed < MOVE_MIN_AGE_S, 'move review changed or expired before move')
        try:
            path.rename(target)
        except OSError as exc:
            write_json(journal, dict(status='move_failed_source_retained', **record, error=str(exc)))
            raise
        # The pre-move journal remains recoverable if a subsequent audit write fails.
        try:
            require(inventory == {str(p.relative_to(target)): dict(sha256=sha(p), size_bytes=p.stat().st_size) for p in sorted(target.rglob('*')) if p.is_file()}, 'moved checkpoint hash mismatch')
            for audit_path in (manifest, note): _check_move_audit_path(audit_path)
            with manifest.open('a', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                if stream.tell() == 0: writer.writeheader()
                writer.writerow(record)
            with note.open('a', encoding='utf-8') as stream:
                stream.write(f'{record["moved_at"]} {path} -> {target}; {total} bytes; journal={journal.name}\n')
            write_json(journal, dict(status='moved', **record, review_sha256=review_sha))
        except Exception as exc:
            write_json(journal, dict(status='moved_audit_incomplete', **record, error=str(exc)))
            raise
        moved.append(path.name)
    return moved


def verify_committed(folder: Path, boundary: int, native_checkpoint: Path | None = None) -> dict:
    """Re-verify an already committed step: file hashes, lineage chain, scenario/contract binding and, when the
    working checkpoint still exists (possibly compacted), its completion.json binding. Never recomputes anything."""
    p = read_json(folder / 'prepared.json'); target = folder / 'committed' / f'step_{boundary:04d}'
    meta = read_json(target / 'checkpoint.json'); require(meta['status'] == 'committed_once' and meta['lineage']['boundary'] == boundary, 'committed step mismatch')
    if native_checkpoint is not None and (native_checkpoint / 'completion.json').is_file():
        require(meta['lineage']['native_completion_sha256'] == sha(native_checkpoint / 'completion.json'), 'conflicting/stale checkpoint')
    require(meta['lineage']['scenario_sha256'] == p['scenario_sha256'] and meta['lineage']['contract_sha256'] == p['contract_sha256'], 'committed step belongs to another run')
    previous = folder / 'committed' / f'step_{boundary - 1:04d}' / 'checkpoint.json'
    expected = sha(previous) if boundary > 1 else digest(dict(initialization=sha(folder / 'initialization.json'), prepared=sha(folder / 'prepared.json')))
    require(meta['lineage']['previous_checkpoint_sha256'] == expected, 'committed lineage break')
    for name, h in meta['files'].items(): require(sha(target / name) == h, 'committed output changed')
    return dict(status='already_committed', boundary=boundary, checkpoint_sha256=sha(target / 'checkpoint.json'), verified_from='committed_prefix_only')


def publish_attempt(attempt: Path, target: Path) -> int:
    """Atomic rename with bounded retries (tools.vtc_growth_io.rename_with_retry); returns the retries needed."""
    from tools.vtc_growth_io import rename_with_retry
    return rename_with_retry(attempt, target)


def recover_partial(attempt: Path, target: Path, lineage: dict) -> dict | None:
    """A .step_NNNN_partial left by a refused rename is republished only if its own manifest verifies."""
    manifest = attempt / 'checkpoint.json'
    if not manifest.is_file(): return None
    old = read_json(manifest)
    if old.get('lineage') != lineage: return None
    for name, h in old['files'].items():
        if not (attempt / name).is_file() or sha(attempt / name) != h: return None
    publish_attempt(attempt, target)
    return dict(status='committed_once', boundary=lineage['boundary'], checkpoint_sha256=sha(target / 'checkpoint.json'), recovered_partial_publication=True)


def compact_working(folder: Path, policy: dict) -> dict:
    """Keep geometry and checkpoints pending a separate FILE_REMOVAL_RULE review."""
    working = folder / 'native/working'
    record = dict(applied=False, policy=policy['value'], effective_retention='retain_pending_move_review',
                  removed_checkpoints=[], removed_files={},
                  retained_checkpoints=[p.name for p in sorted(working_checkpoints(folder).values(), key=checkpoint_number)],
                  compacted_checkpoint_geometry={},
                  reproducible_from='committed/step_*/geometry_next.csv (state k) and consumed_geometry.json (state k-1)',
                  note='No automatic moves/deletions: recent, active or referenced files must be retained; reviewed moves require a manifest and MOVED note')
    write_json(working / 'compaction.json', record); return record


def inventory(out: Path) -> dict:
    packages=sorted((ROOT/'build').glob('p1_state_*/p1_manifest.json'))
    found=[]
    for p in packages:
        try:found.append(dict(path=str(p.parent),**audit_p1_package(p.parent)))
        except (ValueError,KeyError,OSError) as exc:found.append(dict(path=str(p.parent),status='unavailable',reason=str(exc)))
    result=dict(schema_version=SCHEMA,p1_packages=found,p1_producer_report=(ROOT/'handoff/P1_EXECUTION_2026-10-11.md').is_file(),
        actual_p1_reconstruction='consumer_does_not_run_tracking',real_readiness='requires_supported_metric_P1_package_and_explicit_hidden_state',
        native_v3_evidence=str(ROOT/'build/vtc_dev_v3_acceptance_01/acceptance.json'),nutrient_growth_feedback='unavailable_missing_scientific_contract',
        raw_photo_directories={str(p.relative_to(ROOT)):p.is_dir() for p in [ROOT/'data/raw',ROOT/'real_eval']},
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tomato3d/state/schema.py',ROOT/'tools/vtc_development_native.rgg']})
    out=fresh_build(out);write_json(out/'inventory.json',result);return result


def prepare_run(p1_package: Path,scenario: Path,output_dir: Path) -> dict:
    from tools.fsp_extensions import normalize_scenario,hook_settings
    settings=normalize_scenario(read_json(scenario))
    if settings.get('observation_route'):
        from tools.fsp_observation_intake import write_intake
        readiness=write_intake(p1_package,settings['observation_route'],settings,output_dir)
        if settings['observation_route']=='observation_only':return readiness
        require(readiness['simulation_status']=='ready_conditionally','observation route unavailable: '+str(readiness['missing_requirements']))
        out=output_dir
    else:out=None
    selected=selected_package(p1_package,settings);out=out or fresh_build(output_dir)
    frozen=out/'p1';frozen.mkdir()
    for name,h in selected['audit']['source_hashes'].items():
        target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p1_package/name,target)
        require(sha(target)==h,'producer source changed during freeze')
    shutil.copyfile(scenario,out/'scenario.json');require(sha(scenario)==sha(out/'scenario.json'),'scenario changed during freeze')
    if settings.get('extensions'):write_json(out/'normalized_scenario.json',settings)
    # Re-read the copied package. No subsequent access to mutable producer inputs.
    selected=selected_package(frozen,settings);native_inputs=make_native_inputs(selected,settings,out/'native_inputs')
    native_prepare(native_inputs,out/'native')
    mappings=[
        ('captures.csv','capture_time','start_time','aware timestamp; exact selected boundary'),
        ('mtg_vertices.csv','vertex_id/parent_id/complex_id','organ_id/parent_id','instance|vertex namespace; same-scale parent else complex'),
        ('mtg_vertices.csv','base_xyz_mm/tip_xyz_mm','local x_m/tip_x_m','divide 1000 once; no placement in local state'),
        ('optical_surface_manifest.csv','surface_file/sha256/external_leaflet_id','initial_shapes.csv','actual triangles retained; no envelope reconstruction'),
        ('areas.csv','one_sided_lamina_area_mm2','initial area_m2','mesh-derived one-sided area; projected/completed/green kept separate'),
        ('canopy_placement.json','rotation/translation_mm','native placement matrix','R*x+t once, mm->m translation once'),
        ('scenario.json','organ_initialization/plant_initialization','mass/age/reserves/roots/apex','separate conditional assumptions; not P1 measurements'),
        ('canopy_placement.json','app_ratio','comparison metadata only','never multiplied into ray light'),
    ]
    write_csv(out/'producer_consumer_map.csv',[dict(producer=f,source_field=a,native_field=b,conversion=c) for f,a,b,c in mappings])
    write_csv(out/'surface_identity_map.csv',selected['canopy']['identity_map'])
    write_json(out/'canopy_state_t.json',selected['canopy'])
    contract=dict(schema_version=SCHEMA,producer_schema=selected['audit']['schema_version'],producer_schema_sha256=selected['audit']['producer_schema_sha256'],
        scientific_contract=str(ROOT/'handoff/VTC_DEVELOPMENT_SCIENCE_AND_IO_CONTRACT_V3_2026-10-11.md'),scientific_contract_sha256=sha(ROOT/'handoff/VTC_DEVELOPMENT_SCIENCE_AND_IO_CONTRACT_V3_2026-10-11.md'),
        conditional_extension='imported meshes retain initial posture and scale isotropically by funded mass; local structure remains local; generated organs use V3 rules',
        coupling='frozen_geometry_per_declared_interval; new tissue receives light next interval',dt_s=settings.get('dt_s',86400),refinement='half-day allowed only for numerical fixtures with explicitly interval-integrated light_hours',incident_ppfd='unavailable_backend_not_exported',optics='black_absorbing; no tissue transmission or scattering',
        temperature_physiology=False,co2_physiology=False,nutrient_feedback_enabled=False,biological_validation='not_established')
    if settings.get('extensions'):
        contract.update(extension_schema='fsp-extensions-1',extensions=settings['extensions'],incident_ppfd='independent first-hit received tally; true total incidence available only for black leaves',
            optics='explicit Phong grey-band Lambertian reflection/straight transmission; mono-wavelength W/photon conversion',
            terminal_photon_closure='unavailable_no_escape_or_cutoff_instrumentation',schedule='actual piecewise_constant' if settings['extensions'].get('schedule') else 'legacy_duration_only')
    if settings.get('native_boundary'):
        contract.update(native_boundary=settings['native_boundary'],incident_ppfd='unavailable_no_incident_mapping',optics='incoming absorbed photon flux; external transport once',reference_sensor_grid='unavailable_external_authority',nrh_basis=settings['native_boundary']['nrh_basis'])
    contract['parameter_activity']=hook_settings(settings['parameters'])['activity']
    if settings.get('axis_contract'):
        from tools.fsp_axes import export_registry
        contract.update(axis_contract=settings['axis_contract'],topology_scope='provided_axes',topology_completeness='unavailable',carbon_pool='one_per_plant_shared_across_declared_axes')
        export_registry(out,settings)
    write_json(out/'contract.json',contract)
    init=read_json(native_inputs/'initialization.json');write_json(out/'initialization.json',init)
    write_json(out/'stage_0_contract_report.json',dict(status='passed_conditional',audit=selected['audit'],real_execution=not settings['fixture'],missing_real_biology='independent validation and measured hidden-state inputs'))
    all_instances={r['plant_instance_id'] for r in read_json(frozen/'canopy_placement.json')['instances']}
    write_json(out/'stage_1_canopy_report.json',dict(status='passed',instance_count=len(selected['placement']),surfaces=len(selected['surfaces']),triangles=sum(len(s['triangles_mm']) for s in selected['surfaces']),roundtrip_max_error_mm=selected['canopy']['roundtrip_max_error_mm'],placement_applied='once_in_native_mesh_coordinates',app_ratio_applied=False,omitted_package_instances=sorted(all_instances-set(selected['placement'])),unknown_neighbors='not_reconstructed; selected scenario scope only'))
    result=dict(schema_version=SCHEMA,status='prepared',source_p1_path=str(p1_package.resolve()),p1_hashes=selected['audit']['source_hashes'],
        scenario_sha256=sha(out/'scenario.json'),contract_sha256=sha(out/'contract.json'),native_prepared_sha256=sha(out/'native/prepared.json'),
        code_hashes={p.name:sha(p) for p in [Path(__file__),ROOT/'tools/tomato_fsp_io.py',ROOT/'tools/tomato_fsp_canopy.py']})
    write_json(out/'prepared.json',result);return result


def verify_run(folder: Path) -> tuple[dict,dict,dict]:
    p=read_json(folder/'prepared.json');require(p['schema_version']==SCHEMA,'pipeline schema mismatch')
    require(sha(folder/'scenario.json')==p['scenario_sha256'],'scenario changed')
    require(sha(folder/'contract.json')==p['contract_sha256'],'contract changed')
    require(sha(folder/'native/prepared.json')==p['native_prepared_sha256'],'native prepared binding changed')
    for name,h in p['p1_hashes'].items():require(sha(folder/'p1'/name)==h,'P1 frozen source changed')
    np,pkg=verify_prepared(folder/'native');return p,np,pkg


def nutrient_prefix(scenario: dict,pkg: dict,boundary: int) -> dict:
    """Replay prescribed ledger from its immutable initial state; never debit twice."""
    from tools.nutrient_balance import simulate
    requests=scenario.get('nutrient_scenarios')
    if requests is None:
        require(not scenario.get('nutrient_ledger_required',False),'required nutrient ledger missing')
        return dict(status='disabled',nutrient_feedback_enabled=False)
    plants={p['plant_instance_id']:p for p in pkg['tables']['plants.csv']};require(set(requests)==set(plants),'nutrient plant membership mismatch')
    from tools.fsp_extensions import boundary_time
    end=boundary_time(pkg['request'],boundary);outputs={}
    for pid,source in requests.items():
        require(source['plant_instance_id']==pid and source['source_plant_id']==plants[pid]['source_plant_id'],'nutrient identity mismatch')
        require(aware(source['t_start'])==aware(pkg['request']['start_time']) and aware(source['t_end'])>=end,'nutrient interval mismatch')
        request=deepcopy(source);request['t_end']=end.isoformat();request['events']=[e for e in source['events'] if aware(e['at'])<end]
        request['observations']=[e for e in source.get('observations',[]) if aware(e['at'])<=end]
        result=simulate(request);require(result['status']=='available','required nutrient ledger unavailable')
        require(abs(result['totals']['water_residual_ml'])<1e-8 and all(abs(x)<1e-8 for x in result['totals']['nutrient_residual_mg'].values()),'nutrient conservation failed')
        outputs[pid]=result
    return dict(status='synchronized_prescribed_diagnostic',nutrient_feedback_enabled=False,plants=outputs)


def commit_step(folder: Path,checkpoint: Path) -> dict:
    p,np,pkg=verify_run(folder);check=validate_outputs(pkg,checkpoint,np['binding'],checkpoint=True)
    boundary=read_json(checkpoint/'completion.json')['intervals'];settings=read_json(folder/'scenario.json')
    root=folder/'committed';root.mkdir(exist_ok=True);target=root/f'step_{boundary:04d}'
    previous=root/f'step_{boundary-1:04d}'/'checkpoint.json'
    require(boundary==1 or previous.is_file(),'missing previous committed interval')
    prev_hash=sha(previous) if boundary>1 else digest(dict(initialization=sha(folder/'initialization.json'),prepared=sha(folder/'prepared.json')))
    lineage=dict(boundary=boundary,previous_checkpoint_sha256=prev_hash,native_completion_sha256=sha(checkpoint/'completion.json'),scenario_sha256=p['scenario_sha256'],contract_sha256=p['contract_sha256'])
    if target.exists():
        old=read_json(target/'checkpoint.json');require(old['lineage']==lineage,'conflicting/stale checkpoint')
        for name,h in old['files'].items():require(sha(target/name)==h,'committed output changed')
        return dict(status='already_committed',boundary=boundary,checkpoint_sha256=sha(target/'checkpoint.json'))
    nutrients=nutrient_prefix(settings,pkg,boundary) # Failure precedes publication.
    attempt=root/f'.step_{boundary:04d}_partial'
    if attempt.exists():
        recovered=recover_partial(attempt,target,lineage)
        if recovered:return recovered
        require(False,'incomplete publication requires inspection')
    attempt.mkdir()
    i=boundary-1;allstates=rows(checkpoint/'states.csv');current=[s for s in allstates if int(s['day'])==i];future=[s for s in allstates if int(s['day'])==boundary]
    flux=[f for f in rows(checkpoint/'flux.csv') if int(f['interval'])==i];ledger=[f for f in rows(checkpoint/'ledger.csv') if int(f['interval'])==i]
    consumed=[m for m in rows(checkpoint/'consumed.csv') if int(m['state'])==i];mesh=[m for m in rows(checkpoint/'meshes.csv') if int(m['state'])==boundary]
    topology=[{k:s.get(k,'unavailable') for k in ('organ_id','parent_id','axis_id','class','segment_index')} for s in current]
    from tools.fsp_extensions import interval_seconds,boundary_time,photon_factor
    duration=interval_seconds(pkg['request'],i);delta=timedelta(seconds=duration);time_start=boundary_time(pkg['request'],i)
    binding=dict(run_id=pkg['request']['run_id'],step_id=i,interval_start=time_start.isoformat(),interval_end=(time_start+delta).isoformat(),duration_s=duration,timezone=str(time_start.tzinfo),
        geometry_hash=digest(consumed),topology_hash=digest(topology),placement_hash=digest(pkg['integration_geometry']['placement']),
        lamps_hash=digest(pkg['tables']['lamps.csv']),optics_hash=digest({'optics':'black','surface_basis':'one_sided_full_outline'}),
        environment_hash=digest(pkg['tables']['environment.csv'][i]),engine_hash=np['binding']['engine_hash'],runtime=pkg['request']['runtime'])
    execution=read_json(folder/'native/execution.json')
    binding.update(backend_core_sha256=execution['core_sha256'],java_version=execution['java_version'])
    ext=pkg['request'].get('extensions')
    if ext:
        binding.update(scene_objects_hash=digest(ext.get('objects',[])),optics_hash=digest(ext['materials']),spectral_basis_hash=digest(dict(monochromatic_nm=ext['wavelength_nm'])),
            light_schedule_hash=digest(pkg['request'].get('schedule_intervals',[None]*pkg['request']['intervals'])[i]))
    if pkg['request'].get('axis_contract'):binding['axes_hash']=digest(pkg['request']['axes'])
    if pkg['request'].get('native_boundary'):
        incoming=read_json(checkpoint/f'boundary_{i}_response.json')
        binding.update(transport_authority='external_once',incoming_light_sha256=sha(checkpoint/f'boundary_{i}_response.json'),configuration_sha256=pkg['request']['native_boundary']['configuration_sha256'])
        from tools.tomato_fsp_native_boundary import files as boundary_files
        for name in boundary_files(boundary):shutil.copyfile(checkpoint/name,attempt/name)
    cache_key=digest(binding);write_json(attempt/'consumed_geometry.json',dict(**binding,cache_key=cache_key,native_checkpoint_sha256=sha(checkpoint/'completion.json'),vertices=consumed))
    smap={s['organ_id']:s for s in current};leaves=[];plant_light=[]
    for f in flux:
        s=smap.get(f['organ_id'])
        if not s or s['class']!='Leaflet':continue
        area=float(s['area_m2']);absorbed=float(f['absorbed_umol_s'])
        leaves.append(dict(**{k:v for k,v in binding.items() if k!='runtime'},plant_instance_id=s['plant_instance_id'],organ_id=s['organ_id'],surface_id=s['organ_id'],one_sided_area_m2=area,
            incident_ppfd='unavailable',incident_reason='not_exported_by_selected_native_backend',absorbed_umol_s=absorbed,absorbed_ppfd_umol_m2_s=absorbed/area if area else 'unavailable',face_aggregation='both_geometric_faces_single_lamina_area',optical_readiness='conditional_black_optics'))
    for plant in pkg['tables']['plants.csv']:
        pid=plant['plant_instance_id'];plant_light.append(dict(step_id=i,plant_instance_id=pid,absorbed_umol_s=sum(r['absorbed_umol_s'] for r in leaves if r['plant_instance_id']==pid),geometry_hash=binding['geometry_hash']))
    if ext:
        factor=photon_factor(ext['wavelength_nm']);optical=[r for r in rows(checkpoint/'optical_flux.csv') if int(r['interval'])==i];by={r['surface_id']:r for r in optical}
        black=ext['materials'][ext['leaf_material']]['absorptance']==1;lit_seconds=float(pkg['tables']['environment.csv'][i]['light_hours'])*3600
        for leaf in leaves:
            native=by.get(leaf['organ_id']);received=float(native['received_first_hit_W'])*factor if native else 0.;area=leaf['one_sided_area_m2']
            leaf.update(received_first_hit_umol_s=received,received_first_hit_ppfd_umol_m2_s=received/area if area else 'unavailable',
                incident_ppfd=received/area if black and area else 'unavailable',incident_reason='direct native received estimator; black tissue terminates first hit' if black else 'backend suppresses repeated hits per object; total incidence unavailable',
                absorbed_dli_mol_m2=leaf['absorbed_ppfd_umol_m2_s']*lit_seconds/1e6 if area else 'unavailable',
                optical_readiness='conditional_explicit_material',radiometric_absorbed_W=float(native['absorbed_W']) if native else 0.,monochromatic_wavelength_nm=ext['wavelength_nm'])
        write_csv(attempt/'optical_flux.csv',optical,['interval','surface_id','role','received_first_hit_W','absorbed_W','reflected_W','transmitted_W','hit_count'])
        budget_rows=[]
        for r in rows(checkpoint/'scene_photon_budget.csv'):
            if int(r['interval'])!=i:continue
            converted={k+'_umol_s':float(r[k+'_W'])*factor if r[k+'_W']!='unavailable' else 'unavailable' for k in ('emitted','leaf_absorbed','nonleaf_absorbed','escape','termination','estimator_residual')}
            budget_rows.append(dict(**binding,**r,**converted,photon_conversion_umol_per_J=factor))
        write_csv(attempt/'scene_photon_budget.csv',budget_rows)
    if pkg['request'].get('native_boundary'):
        for leaf in leaves:leaf.update(optical_readiness='conditional_external_absorbed',incident_reason='incident_to_absorbed_mapping_unreviewed',transport_authority='external_once')
    write_csv(attempt/'leaf_light.csv',leaves);write_csv(attempt/'plant_light.csv',plant_light)
    write_json(attempt/'canopy_light.json',dict(**binding,leaf_absorbed_umol_s=sum(r['absorbed_umol_s'] for r in leaves),accounting='leaf and nonleaf absorption; terminal closure unavailable' if ext else 'leaf absorption only; nonleaf absorption and escaping photons not exported',tissue_transmission='external_response_source_contract' if pkg['request'].get('native_boundary') else 'explicit_straight_through_material' if ext else 'zero_under_black_hypothesis',gap_transport='external_once_actual_native_geometry' if pkg['request'].get('native_boundary') else 'native_geometry'))
    proposals=[r for r in rows(checkpoint/'proposals.csv') if int(r['interval'])==i]
    write_json(attempt/'resource_proposal.json',dict(**binding,status='provisional_native_output',organs=proposals,state_t_sha256=digest(current),geometry_t_unchanged=True))
    write_json(attempt/'candidate_state_next.json',dict(state_time=(time_start+delta).isoformat(),local_frame='plant_local_m',organs=future,geometry_next_hash=digest(mesh),status='validated_candidate'))
    write_csv(attempt/'state.csv',future);write_csv(attempt/'geometry_next.csv',mesh);write_csv(attempt/'carbon_ledger.csv',ledger);write_csv(attempt/'organ_allocations.csv',flux)
    if pkg['request'].get('axis_contract'):
        from tools.fsp_axes import export_registry
        export_registry(attempt,pkg['request'])
        write_csv(attempt/'axis_flux.csv',[r for r in rows(checkpoint/'axis_flux.csv') if int(r['interval'])==i])
        from tools.fsp_axes_segments import axis_inventory
        write_csv(attempt/'axis_inventory.csv',axis_inventory(pkg['request'],future,[r for r in rows(checkpoint/'apices.csv') if int(r['day'])==boundary]))
        write_csv(attempt/'organ_inventory.csv',[{k:s[k] for k in ('organ_id','plant_instance_id','class','axis_id','parent_rank','segment_index','provenance')} for s in future])
    ev=[e for e in rows(checkpoint/'events.csv') if int(e['day'])==boundary];write_csv(attempt/'development_events.csv',ev,list(rows(checkpoint/'events.csv')[0]) if rows(checkpoint/'events.csv') else ['day','organ_id','parent_id','event','rule','structural_gC'])
    write_json(attempt/'nutrient_ledger.json',nutrients)
    checkpoint_manifest=dict(schema_version=SCHEMA,status='committed_once',state_time=(time_start+delta).isoformat(),lineage=lineage,cache_key=cache_key,
        validation=check,files={q.name:sha(q) for q in attempt.iterdir()},native_advance_count=1,nutrient_feedback_enabled=False)
    write_json(attempt/'checkpoint.json',checkpoint_manifest);retries=publish_attempt(attempt,target)
    return dict(status='committed_once',boundary=boundary,checkpoint_sha256=sha(target/'checkpoint.json'),publication_retries=retries)


def collect(folder: Path,viewer: bool=False) -> dict:
    _,_,pkg=verify_run(folder);ex=read_json(folder/'native/execution.json')
    require(ex['prepared_sha256']==sha(folder/'native/prepared.json') and ex['log_sha256']==sha(folder/'native/groimp.log'),'native execution binding mismatch')
    checkpoints=working_checkpoints(folder)
    results=[]
    try:
        for boundary in range(1,pkg['request']['intervals']+1):
            if (folder/'committed'/f'step_{boundary:04d}'/'checkpoint.json').is_file():results.append(verify_committed(folder,boundary,checkpoints.get(boundary)))
            elif boundary in checkpoints:results.append(commit_step(folder,checkpoints[boundary]))
            else:break
    except Exception as exc:
        write_json(folder/'failed_attempt.json',dict(status='failed',error=str(exc),last_valid_boundary=results[-1]['boundary'] if results else 0,interval_hook_error=ex.get('interval_hook_error')));raise
    complete=ex['status']=='verified' and len(results)==pkg['request']['intervals']
    result=dict(schema_version=SCHEMA,status='complete_conditional_trajectory' if complete else 'failed_with_valid_prefix',last_valid_boundary=results[-1]['boundary'] if results else 0,
        native_execution=ex,checkpoints=results,independent_biological_validation='not_established',nutrient_feedback_enabled=False)
    write_json(folder/'collection.json',result);require(complete,'native run incomplete; only explicitly validated prefix retained')
    from tools.vtc_development import collect as collect_native
    if not (folder/'native/results').exists():collect_native(folder/'native',viewer=viewer,copy_geometry=False)
    for stage,name in [(2,'light'),(3,'resources'),(4,'growth'),(5,'feedback'),(6,'extensions')]:
        ext=pkg['request'].get('extensions')
        limitations={'light':'first-hit received plus absorption/reflection/transmission; total incidence for nonblack leaves and terminal photon closure unavailable' if ext else 'black optics; incident PPFD unavailable; dedicated convergence evidence required',
            'resources':'fixed NRH; temperature/CO2 physiology inactive',
            'growth':'imported posture held; V3 generated direction; no phototropism',
            'feedback':'explicit piecewise schedule; frozen geometry within each interval; biological subdaily response unvalidated' if pkg['request'].get('schedule_intervals') else 'daily default; numerical fixture refinement only',
            'extensions':'prescribed plant-level nutrient accounting only; no assimilation or nutrient growth law'}
        write_json(folder/f'stage_{stage}_{name}_report.json',dict(status='not_executed_light_only' if pkg['request'].get('light_only') and stage>2 else 'executed_conditional',intervals=len(results),evidence='committed/step_*/',
            limitations=limitations[name]))
    return result


def advance_interval(folder: Path,checkpoint: Path) -> dict:
    """Validate/publish one native-produced interval; does not recalculate growth."""
    return commit_step(folder,checkpoint)


def run(folder: Path,timeout: float=1200,retention: str | None=None,viewer: bool=False) -> dict:
    """Native run with incremental publication: every completed interval is committed while the engine runs,
    working checkpoints are retained pending separate move review, and the run ends with daily_summary.csv,
    RUN_SUMMARY.md and run_manifest.json (F11 C1/C4). viewer=True opts in to native/results/viewer.html."""
    verify_run(folder);policy=retention_policy(retention);write_json(folder/'retention_policy.json',policy)
    def on_interval(k: int) -> None:
        commit_step(folder,folder/'native/working/checkpoints'/f'step_{k}')
        prune_working_checkpoints(folder,policy['keep_working_checkpoints'],k)
    try:native_run(folder/'native',timeout_s=timeout,on_interval=on_interval)
    except Exception:
        try:collect(folder,viewer=viewer)
        except Exception:pass
        raise
    result=collect(folder,viewer=viewer)
    result['compaction']=compact_working(folder,policy)
    from tools.fsp_run_summary import write_summary
    result['run_manifest']=write_summary(folder,retention=policy)
    return result


def replay(folder: Path,out: Path,timeout: float=1200) -> dict:
    verify_run(folder);prepare_run(folder/'p1',folder/'scenario.json',out);result=run(out,timeout)
    existing=sorted((folder/'committed').glob('step_*/state.csv'))
    for path in existing:
        a={s['organ_id']:s for s in rows(path)};b={s['organ_id']:s for s in rows(out/'committed'/path.parent.name/'state.csv')};require(set(a)==set(b),'replay identity mismatch')
        for oid in a:
            for k in ('mass_gDM','thermal_age_dd','length_m','area_m2'):require(abs(float(a[oid][k])-float(b[oid][k]))<1e-9,'replay state mismatch')
    write_json(out/'replay_verification.json',dict(status='passed',source=str(folder),verified_existing_checkpoints=len(existing),numeric_tolerance=1e-9,resume_policy='fresh replay validates prefix; no in-place native reset'))
    return result


def compare_observation(forecast: Path,p1: Path,out: Path) -> dict:
    """Read-only, exact time/identity/area comparison; no assimilation or future leak."""
    audit=audit_p1_package(p1);require(audit['status']=='audited','later producer unavailable')
    meta=read_json(forecast/'checkpoint.json')
    for name,h in meta['files'].items():require(sha(forecast/name)==h,'forecast changed')
    states={r['organ_id']:r for r in rows(forecast/'state.csv')};at=aware(meta['state_time']);captures={(r['plant_instance_id'],r['capture_id']):r for r in rows(p1/'captures.csv')}
    ids={(r['plant_instance_id'],r['capture_id'],r['organ_id']):r for r in rows(p1/'identity_manifest.csv')};comparisons=[]
    for a in rows(p1/'areas.csv'):
        c=captures[a['plant_instance_id'],a['capture_id']]
        try:aligned=aware(c['capture_time'])==at
        except ValueError:continue
        if not aligned:continue
        require(a['area_basis']=='full_outline','observation area basis differs')
        key=a['plant_instance_id']+'|'+a['leaflet_id'];identity=ids.get((a['plant_instance_id'],a['capture_id'],a['leaflet_id']))
        valid=identity and identity['correspondence_status']=='confirmed' and a['one_sided_lamina_area_mm2__status']!='unavailable' and key in states
        comparisons.append(dict(organ_id=key,capture_id=a['capture_id'],forecast_time=meta['state_time'],status='comparable' if valid else 'unavailable',
            reason='' if valid else 'identity_or_surface_area_unavailable',forecast_area_mm2=float(states[key]['area_m2'])*1e6 if key in states else 'unavailable',
            observation_area_mm2=a['one_sided_lamina_area_mm2'],difference_mm2=float(a['one_sided_lamina_area_mm2'])-float(states[key]['area_m2'])*1e6 if valid else 'unavailable',evidence=c['evidence']))
    out=fresh_build(out)
    write_csv(out/'comparison.csv',comparisons,['organ_id','capture_id','forecast_time','status','reason','forecast_area_mm2','observation_area_mm2','difference_mm2','evidence'])
    report=dict(status='compared' if comparisons else 'unavailable',reason='' if comparisons else 'no_exact_time_observation',forecast_checkpoint_sha256=sha(forecast/'checkpoint.json'),observation_hashes=audit['source_hashes'],assimilation_applied=False,forecast_unchanged=True)
    write_json(out/'comparison.json',report);return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    q=sub.add_parser('inventory');q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('prepare');q.add_argument('--p1',type=Path,required=True);q.add_argument('--scenario',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    for op in ('run','collect','replay','summarize'):
        q=sub.add_parser(op);q.add_argument('--run-dir',type=Path,required=True)
        if op in ('run','replay'):q.add_argument('--timeout',type=float,default=1200)
        if op=='replay':q.add_argument('--out',type=Path,required=True)
        if op in ('run','collect'):q.add_argument('--viewer',action='store_true',help='opt in to native/results/viewer.html (F11 C1-5)')
        if op=='run':q.add_argument('--retention',choices=sorted(RETENTION_VALUES),help='override CHECKPOINT_RETENTION for this run')
    q=sub.add_parser('compare-observation');q.add_argument('--forecast',type=Path,required=True);q.add_argument('--p1',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.op=='inventory':r=inventory(a.out)
        elif a.op=='prepare':r=prepare_run(a.p1,a.scenario,a.out)
        elif a.op=='run':r=run(a.run_dir,a.timeout,retention=a.retention,viewer=a.viewer)
        elif a.op=='collect':r=collect(a.run_dir,viewer=a.viewer)
        elif a.op=='summarize':
            from tools.fsp_run_summary import write_summary
            r=write_summary(a.run_dir)
        elif a.op=='replay':r=replay(a.run_dir,a.out,a.timeout)
        else:r=compare_observation(a.forecast,a.p1,a.out)
        print(json.dumps(r,ensure_ascii=False,indent=2))
    except (ValueError,KeyError,OSError,TimeoutError) as e:p.exit(1,f'{type(e).__name__}: {e}\n')


if __name__=='__main__':main()
