"""Default-off native interval handshake; no empirical or nutrient response law."""
from __future__ import annotations
import argparse
import math
import re
import time
from pathlib import Path
import numpy as np
from tools.vtc_growth_io import ROOT, require, read_json, write_json, write_csv, rows, sha, digest, number

SCHEMA = 'fsp-native-boundary-1'


def hash_value(value: str) -> None:
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'boundary_sha256_required')


def validate_options(req: dict, state: list, geometry: dict) -> None:
    shape = req.get('imported_shape_api')
    if shape is not None:
        require(shape.get('schema_version') == 'fsp-imported-anchor-1', 'shape_api_schema')
        require(shape.get('enabled') is False, 'developmental_shape_law_unavailable')
        by = {s['organ_id']: s for s in state}; seen = set()
        for a in shape.get('anchors', []):
            oid = a['organ_id']; require(oid in geometry['triangles_mm'] and oid not in seen, 'anchor_identity_invalid'); seen.add(oid)
            require(a['parent_id'] == by[oid]['parent_id'] and a['parent_id'] in by, 'anchor_parent_identity_mismatch')
            require(a['frame'] == 'plant_local_m', 'anchor_frame_mismatch')
            point = np.asarray(a['anchor_m'], float)
            require(point.shape == (3,) and np.isfinite(point).all(), 'anchor_coordinates_invalid')
            base = np.array([float(by[oid][k]) for k in ('x_m','y_m','z_m')])
            require(np.allclose(point, base, rtol=0, atol=1e-12), 'anchor_not_declared_organ_base')
            require(a['reference_geometry_sha256'] == digest(geometry['triangles_mm'][oid]), 'anchor_geometry_changed')
            require(a.get('source_kind') and a.get('source_ref'), 'anchor_provenance_missing')
    resource = req.get('resource_hook')
    if resource is not None:
        require(resource.get('schema_version') == 'fsp-native-resource-hook-1', 'resource_hook_schema')
        require(resource.get('enabled') is False, 'native_nutrient_causal_law_unavailable')
        require(resource.get('quantity_basis') == 'EC_equivalent_not_element_resolved', 'resource_quantity_basis')
    b = req.get('native_boundary')
    if b is None: return
    require(set(b) == {'schema_version','enabled','basis','unit','area_basis','transport_authority','nrh_basis','configuration_sha256','timeout_s','source_kind','source_ref'}, 'boundary_unknown_or_missing_fields')
    require(b.get('schema_version') == SCHEMA and b.get('enabled') is True, 'boundary_schema_or_enable')
    require(req.get('integration_contract') == 'fsp-integration-1', 'boundary_requires_imported_integration')
    require(not req.get('extensions'), 'boundary_extension_transport_not_supported')
    require(b.get('basis') == 'absorbed_photon_flux', 'incident_to_absorbed_mapping_unreviewed')
    require(b.get('unit') == 'umol/s' and b.get('area_basis') == 'one_sided_lamina_full_outline', 'boundary_unit_or_area_basis')
    require(b.get('transport_authority') == 'external_once', 'boundary_single_authority_required')
    if not req.get('light_only'):
        require(req.get('fixture') is True and b.get('nrh_basis') == 'fixture_black_absorbed_equivalence', 'nrh_light_basis_unreviewed')
    else:
        require(b.get('nrh_basis') == 'disabled_diagnostic', 'diagnostic_nrh_must_be_disabled')
    hash_value(b.get('configuration_sha256'))
    require(type(b.get('timeout_s')) is int and 1 <= b['timeout_s'] <= 300, 'boundary_timeout_range')
    require(b.get('source_kind') in ('fixture_truth','derived') and b.get('source_ref'), 'boundary_source_required')


def native_code(req: dict) -> str:
    b = req.get('native_boundary')
    from tools.fsp_extensions import boundary_time
    code = (ROOT/'tools/tomato_fsp_boundary_native.rgg').read_text(encoding='utf-8')
    mapping = dict(ENABLED=str(bool(b)).lower(), TIMEOUT=str((b or {}).get('timeout_s', 1)*1000),
                   CONFIG=(b or {}).get('configuration_sha256',''),
                   STARTS=','.join('"'+boundary_time(req,i).isoformat()+'"' for i in range(req['intervals'])),
                   ENDS=','.join('"'+boundary_time(req,i+1).isoformat()+'"' for i in range(req['intervals'])))
    for k,v in mapping.items(): code = code.replace('@B_'+k+'@',v)
    return code


def files(n: int) -> set[str]:
    names = {'boundary_audit.csv'}
    for i in range(n):
        names.update(f'boundary_{i}_{suffix}' for suffix in ('geometry.csv','leaves.csv','request.json','response.json','light.csv','ready.csv'))
    return names


def read_request(working: Path, interval: int) -> tuple[dict, list]:
    p = working/f'boundary_{interval}_request.json'; r = read_json(p)
    require(r['schema_version'] == SCHEMA and r['interval'] == interval, 'boundary_request_identity')
    require(r['coordinate_frame'] == 'canopy_m', 'boundary_frame_mismatch')
    for suffix,key in (('geometry.csv','geometry_sha256'),('leaves.csv','leaves_sha256')):
        require(sha(working/f'boundary_{interval}_{suffix}') == r[key], 'boundary_consumed_geometry_changed')
    for k in ('configuration_sha256','input_hash','engine_hash','predecessor_sha256'): hash_value(r[k])
    if interval:
        require(sha(working/f'checkpoints/step_{interval}/completion.json') == r['predecessor_sha256'], 'boundary_predecessor_changed')
    else: require(r['predecessor_sha256'] == r['input_hash'], 'boundary_initial_predecessor')
    return r, rows(working/f'boundary_{interval}_leaves.csv')


def submit_light(working: Path, interval: int, response: dict, now_ms: int | None = None) -> dict:
    """Validate every ID/value and publish ready last; never mutate a native state."""
    request, leaves = read_request(working, interval)
    prefix = working/f'boundary_{interval}_'
    def target(suffix: str) -> Path: return Path(str(prefix)+suffix)
    require(response.get('schema_version') == 'fsp-native-light-response-1', 'boundary_response_schema')
    require(response.get('request_sha256') == sha(target('request.json')), 'boundary_stale_request')
    require(response.get('configuration_sha256') == request['configuration_sha256'], 'boundary_configuration_changed')
    require(response.get('basis') == 'absorbed_photon_flux' and response.get('unit') == 'umol/s', 'boundary_flux_basis_or_unit')
    require(response.get('transport_count') == 1 and 'shading_multiplier' not in response, 'boundary_double_shading')
    require(response.get('source_kind') in ('derived','fixture_truth') and response.get('source_ref'), 'boundary_response_source')
    source_files = response.get('source_files', {}); code_files = response.get('code_files', {})
    require(source_files and code_files, 'boundary_source_and_code_hashes_required')
    for group in (source_files,code_files):
        for name,h in group.items():
            hash_value(h); require(sha(Path(name)) == h, 'boundary_source_or_code_changed')
    values = response.get('leaves', []); by = {r['organ_id']:r for r in values}
    expected = {r['organ_id']:r for r in leaves}
    require(len(by) == len(values) and set(by) == set(expected), 'boundary_leaf_coverage')
    for oid,r in by.items():
        require(r['plant_instance_id'] == expected[oid]['plant_instance_id'], 'boundary_leaf_owner')
        require(math.isclose(number(r['area_m2'],'area',0),float(expected[oid]['area_m2']),abs_tol=1e-12,rel_tol=1e-10), 'boundary_leaf_area')
        number(r['absorbed_umol_s'],'absorbed light',0)
    # Identical retries do not publish twice, even after expiration; validate immutable result files.
    if target('ready.csv').exists():
        require(read_json(target('response.json')) == response, 'boundary_conflicting_retry')
        ready = rows(target('ready.csv'))[0]
        require(sha(target('light.csv')) == ready['light_sha256'] and sha(target('response.json')) == ready['response_sha256'], 'boundary_published_response_changed')
        return dict(status='already_submitted',request_sha256=response['request_sha256'])
    now_ms = int(time.time()*1000) if now_ms is None else now_ms
    require(type(response.get('expires_epoch_ms')) is int and now_ms < response['expires_epoch_ms'] <= request['deadline_epoch_ms'], 'boundary_response_expired_or_deadline')
    lock = target('partial'); require(not lock.exists(), 'boundary_partial_publication'); lock.mkdir()
    write_json(lock/'response.json', response)
    write_csv(lock/'light.csv', [by[r['organ_id']] for r in leaves], ['organ_id','plant_instance_id','area_m2','absorbed_umol_s'])
    write_csv(lock/'ready.csv',[dict(request_sha256=response['request_sha256'],response_sha256=sha(lock/'response.json'),
        light_sha256=sha(lock/'light.csv'),expires_epoch_ms=response['expires_epoch_ms'])])
    for name in ('response.json','light.csv','ready.csv'): (lock/name).rename(target(name))
    lock.rmdir()
    return dict(status='submitted',request_sha256=response['request_sha256'])


def validate_output(pkg: dict, native: Path, data: dict, n: int, binding: dict) -> None:
    if not pkg['request'].get('native_boundary'): return
    audits = data['boundary_audit.csv']; require(len(audits) == n, 'boundary_audit_coverage')
    for i,a in enumerate(audits):
        r = read_json(native/f'boundary_{i}_request.json'); response = read_json(native/f'boundary_{i}_response.json')
        from tools.fsp_extensions import boundary_time
        require(r['schema_version'] == SCHEMA and r['coordinate_frame'] == 'canopy_m' and r['run_id'] == pkg['request']['run_id'] and r['interval'] == i, 'boundary_output_identity')
        require(r['interval_start'] == boundary_time(pkg['request'],i).isoformat() and r['interval_end'] == boundary_time(pkg['request'],i+1).isoformat(), 'boundary_output_time')
        require(r['engine_hash'] == binding['engine_hash'], 'boundary_output_engine')
        require(a['transport_authority'] == 'external_once' and a['native_growth_authority'] == 'native_once', 'boundary_output_authority')
        require(a['interval'] == str(i) and a['request_sha256'] == sha(native/f'boundary_{i}_request.json'), 'boundary_audit_request')
        require(a['response_sha256'] == sha(native/f'boundary_{i}_response.json') and a['light_sha256'] == sha(native/f'boundary_{i}_light.csv'), 'boundary_audit_response')
        require(r['input_hash'] == digest(pkg['source_hashes']) and r['configuration_sha256'] == pkg['request']['native_boundary']['configuration_sha256'], 'boundary_output_input_binding')
        require(response['request_sha256'] == a['request_sha256'], 'boundary_output_response_request')
        require(response['configuration_sha256'] == r['configuration_sha256'] and response['basis'] == 'absorbed_photon_flux' and response['unit'] == 'umol/s' and response['transport_count'] == 1 and 'shading_multiplier' not in response, 'boundary_output_response_semantics')
        require(r['geometry_sha256'] == sha(native/f'boundary_{i}_geometry.csv') and r['leaves_sha256'] == sha(native/f'boundary_{i}_leaves.csv'), 'boundary_output_geometry_binding')
        require(rows(native/f'boundary_{i}_geometry.csv') == [v for v in data['consumed.csv'] if int(v['state']) == i], 'boundary_not_actual_consumed_geometry')
        light = {x['organ_id']:x for x in rows(native/f'boundary_{i}_light.csv')}
        inventory = {x['organ_id']:x for x in rows(native/f'boundary_{i}_leaves.csv')}
        require(set(light) == set(inventory), 'boundary_output_leaf_coverage')
        declared = {x['organ_id']:x for x in response['leaves']}
        require(len(declared) == len(response['leaves']) and set(declared) == set(light), 'boundary_output_response_coverage')
        for oid,v in light.items():
            require(v['plant_instance_id'] == inventory[oid]['plant_instance_id'] == declared[oid]['plant_instance_id'], 'boundary_output_leaf_owner')
            require(float(v['area_m2']) == float(inventory[oid]['area_m2']) == float(declared[oid]['area_m2']) and float(v['absorbed_umol_s']) == float(declared[oid]['absorbed_umol_s']), 'boundary_output_response_values')
        for f in data['flux.csv']:
            if int(f['interval']) != i: continue
            expected = float(light[f['organ_id']]['absorbed_umol_s']) if f['organ_id'] in light else 0.
            require(math.isclose(float(f['absorbed_umol_s']),expected,rel_tol=1e-12,abs_tol=1e-12), 'boundary_flux_not_used_once')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--working',type=Path,required=True);p.add_argument('--interval',type=int,required=True);p.add_argument('--response',type=Path,required=True)
    a=p.parse_args();print(submit_light(a.working,a.interval,read_json(a.response)))


if __name__ == '__main__': main()
