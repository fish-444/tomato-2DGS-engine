"""Frozen, configuration-bound horizontal incident PPFD; no extrapolation or fitting."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from tools.leaf_light.common import (StudyError, digest, file_hash, output_dir, read_json,
                                    rows, write_json, write_rows)

SCHEMA = 'empirical-light-field-1'
COLUMNS = ('sample_id', 'plane_id', 'x_mm', 'y_mm', 'z_mm', 'ppfd_umol_m2_s')


def finite(value: object) -> float:
    if isinstance(value, bool):
        raise StudyError('boolean_is_not_numeric')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StudyError('finite_number_required') from exc
    if not math.isfinite(result):
        raise StudyError('finite_number_required')
    return result


def code_hashes() -> dict:
    root = Path(__file__).resolve().parent
    paths = [root/name for name in ('empirical_light_field.py', 'empirical_canopy.py',
        'reference_dli_empirical.py', 'reference_dli_consumer.py', 'reference_dli.py', 'leaf_light/common.py')]
    paths.append(root.parent/'tomato3d/light/raytrace.py')
    return {str(p): file_hash(p) for p in paths}


def verify_hashes(hashes: dict) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise StudyError('hashes_missing')
    for name, expected in hashes.items():
        if file_hash(Path(name)) != expected:
            raise StudyError('source_or_code_hash_mismatch:'+name)


def validate_config(config: dict) -> None:
    for key in ('board', 'tray', 'lamp', 'environment', 'canopy_grid', 'empirical_contract'):
        if not isinstance(config.get(key), dict):
            raise StudyError('configuration_missing:'+key)
    contract = config['empirical_contract']
    if (contract.get('coordinate_frame') != 'grow' or
            contract.get('height_basis') != 'shelf_floor' or
            contract.get('source_kind') not in ('user_assumed', 'fixture_truth')):
        raise StudyError('explicit_coordinate_contract_required')
    bounds = [finite(v) for v in contract.get('tray_bounds_xy_mm', [])]
    if len(bounds) != 4 or not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise StudyError('invalid_tray_bounds')
    if contract.get('coordinate_variant') not in ('as_recorded', 'reverse_x', 'reverse_y', 'reverse_xy'):
        raise StudyError('unsupported_coordinate_variant')
    if contract.get('plants_above_150_mm') is not False:
        raise StudyError('above_canopy_observation_confirmation_required')
    if not contract.get('conditional_reasons') or not contract.get('source_refs'):
        raise StudyError('explicit_assumptions_and_provenance_required')


def sample_records(path: Path, config: dict) -> list[dict]:
    source = rows(path, ('plane_id', 'plane_height_cm', 'grid_row', 'grid_col',
                         'x_cm', 'y_cm', 'ppfd_umol_m2_s'))
    result = []
    for r in source:
        q = finite(r['ppfd_umol_m2_s'])
        if q < 0:
            raise StudyError('negative_ppfd')
        result.append(dict(sample_id=f"{r['plane_id']}:{r['grid_row']}:{r['grid_col']}",
            plane_id=r['plane_id'], x_mm=10*finite(r['x_cm']), y_mm=10*finite(r['y_cm']),
            z_mm=10*finite(r['plane_height_cm']), ppfd_umol_m2_s=q))
    if len({r['sample_id'] for r in result}) != len(result):
        raise StudyError('duplicate_sample_id')
    variant = config['empirical_contract']['coordinate_variant']
    for z_mm in {r['z_mm'] for r in result}:
        plane = [r for r in result if r['z_mm'] == z_mm]
        for axis in ('x', 'y'):
            if axis in variant.removeprefix('reverse_') and variant != 'as_recorded':
                key = axis+'_mm'
                total = min(r[key] for r in plane)+max(r[key] for r in plane)
                for r in plane:
                    r[key] = total-r[key]
    grids(result)
    return result


def grids(samples: list[dict]) -> dict:
    planes = {}
    if {r['z_mm'] for r in samples} != {70., 150.}:
        raise StudyError('requires_70_and_150_mm_planes')
    for z_mm in (70., 150.):
        plane = [r for r in samples if r['z_mm'] == z_mm]
        xs = sorted({r['x_mm'] for r in plane}); ys = sorted({r['y_mm'] for r in plane})
        cells = {(r['x_mm'], r['y_mm']): r for r in plane}
        if len(xs) < 2 or len(ys) < 2 or len(cells) != len(plane) or len(cells) != len(xs)*len(ys):
            raise StudyError('incomplete_or_duplicate_rectangular_grid')
        planes[z_mm] = (xs, ys, cells)
    return planes


def bracket(axis: list[float], value: float) -> list[tuple[float, float]]:
    if value < axis[0] or value > axis[-1]:
        raise StudyError('outside_sample_centre_support')
    if value in axis:
        return [(value, 1.)]
    hi = next(i for i, v in enumerate(axis) if v > value)
    a, b = axis[hi-1:hi+1]; t = (value-a)/(b-a)
    return [(a, 1-t), (b, t)]


def evaluate(field: dict, position_mm: list[float]) -> dict:
    if len(position_mm) != 3:
        raise StudyError('position_requires_xyz')
    x_mm, y_mm, z_mm = map(finite, position_mm)
    lower_x, upper_x, lower_y, upper_y = field['configuration']['empirical_contract']['tray_bounds_xy_mm']
    if not (lower_x <= x_mm <= upper_x and lower_y <= y_mm <= upper_y):
        raise StudyError('outside_tray_block')
    weights = []
    for z, wz in bracket([70., 150.], z_mm):
        xs, ys, cells = grids(field['samples'])[z]
        for x, wx in bracket(xs, x_mm):
            for y, wy in bracket(ys, y_mm):
                weights.append(dict(sample_id=cells[x, y]['sample_id'], weight=wx*wy*wz,
                                    ppfd_umol_m2_s=cells[x, y]['ppfd_umol_m2_s']))
    return dict(position_mm=[x_mm, y_mm, z_mm],
        ppfd_umol_m2_s=sum(w['weight']*w['ppfd_umol_m2_s'] for w in weights),
        source_kind='derived', method='empirical_light_field', interpolation_weights=weights)


def freeze(grid_path: Path, configuration_path: Path, out: Path) -> dict:
    configuration_path = configuration_path.resolve(); grid_path = grid_path.resolve()
    config = read_json(configuration_path); validate_config(config)
    samples = sample_records(grid_path, config)
    sources = {str(p): file_hash(p) for p in (grid_path, configuration_path)}
    for ref in config['empirical_contract']['source_refs']:
        p = Path(ref['path']).resolve()
        if file_hash(p) != ref['sha256']:
            raise StudyError('configuration_evidence_hash_mismatch')
        sources[str(p)] = ref['sha256']
    folder = output_dir(out)
    write_rows(folder/'samples.csv', samples, COLUMNS)
    record = dict(schema_version=SCHEMA, method='empirical_light_field', source_kind='derived',
        quantity='horizontal_incident', unit='umol/m2/s', coordinate_frame='grow',
        height_basis='shelf_floor', interpolation='bilinear_xy_then_linear_z_no_extrapolation',
        configuration=config, configuration_sha256=digest(config), samples=samples,
        inputs=dict(grid=str(grid_path), configuration=str(configuration_path)), source_hashes=sources,
        code_hashes=code_hashes(), artifact_hashes={'samples.csv': file_hash(folder/'samples.csv')},
        status='conditional', physical_accuracy='unvalidated',
        uncertainty='unavailable_sensor_calibration_and_spatial_reconstruction_error',
        conditional_reasons=config['empirical_contract']['conditional_reasons'])
    write_json(folder/'field.json', record)
    return record


def load_field(path: Path, current_configuration: Path) -> dict:
    field = read_json(path)
    if (field.get('schema_version') != SCHEMA or field.get('source_kind') != 'derived' or
            field.get('method') != 'empirical_light_field'):
        raise StudyError('invalid_empirical_field_contract')
    for key, expected in dict(quantity='horizontal_incident', unit='umol/m2/s', coordinate_frame='grow',
            height_basis='shelf_floor', interpolation='bilinear_xy_then_linear_z_no_extrapolation',
            status='conditional', physical_accuracy='unvalidated').items():
        if field.get(key) != expected:
            raise StudyError('invalid_empirical_field_contract:'+key)
    validate_config(field['configuration'])
    if digest(field['configuration']) != field['configuration_sha256'] or digest(
            read_json(current_configuration)) != field['configuration_sha256']:
        raise StudyError('configuration_changed_remeasurement_required')
    verify_hashes(field['source_hashes']); verify_hashes(field['code_hashes'])
    if set(field['code_hashes']) != set(code_hashes()):
        raise StudyError('incomplete_code_hash_coverage')
    for key in ('grid', 'configuration'):
        if field['inputs'][key] not in field['source_hashes']:
            raise StudyError('unbound_field_input')
    if read_json(Path(field['inputs']['configuration'])) != field['configuration']:
        raise StudyError('frozen_configuration_source_mismatch')
    if field['samples'] != sample_records(Path(field['inputs']['grid']), field['configuration']):
        raise StudyError('derived_samples_do_not_match_raw_grid')
    if set(field['artifact_hashes']) != {'samples.csv'} or file_hash(path.parent/'samples.csv') != field['artifact_hashes']['samples.csv']:
        raise StudyError('sample_artifact_hash_mismatch')
    return field


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_cmd = sub.add_parser('freeze')
    for name in ('grid', 'configuration', 'out'):
        freeze_cmd.add_argument('--'+name, type=Path, required=True)
    query_cmd = sub.add_parser('query')
    for name in ('field', 'configuration'):
        query_cmd.add_argument('--'+name, type=Path, required=True)
    query_cmd.add_argument('--position-mm', type=float, nargs=3, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'freeze':
            result = freeze(args.grid, args.configuration, args.out)
            print(dict(status=result['status'], samples=len(result['samples']), out=str(args.out)))
        else:
            print(evaluate(load_field(args.field, args.configuration), args.position_mm))
        return 0
    except (StudyError, OSError, KeyError, ValueError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
