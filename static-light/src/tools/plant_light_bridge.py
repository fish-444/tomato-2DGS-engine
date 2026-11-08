"""Connect one explicit plant snapshot to the existing PPFD study workflow.

python -m tools.plant_light_bridge --help
Preparation never runs inference, reconstruction or GroIMP. Execution is explicit.
Positions, hulls and axes remain display-only; only reviewed leaf tables enter transport.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import json
import math
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote

import cv2
import numpy as np

from tomato3d.light.geometry import load_geometry, place_in_grow_frame, triangle_areas_mm2
from tomato3d.light.scene import load_scene
from tools import ppfd_validation as pv
from tools.groimp_density import surface_gate
from tools.groimp_structure_export import load_snapshot, make_graph
from tools.leaf_light.common import StudyError, digest, file_hash, number, output_dir, read_json, rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'plant-light-bridge-1'
STUDY_FILES = ('scene', 'lamp_conditions', 'rig_surfaces', 'sensors', 'points', 'readings', 'spectrum')
SURFACE_COLUMNS = ('surface_id', 'plant_id', 'session_id', 'capture_date', 'geometry_variant_id',
                   'geometry_sha256', 'organ_id', 'leaflet_id', 'triangle_id', 'geometry_source', 'area_mm2',
                   'optical_status', 'source_photos')
LINK_COLUMNS = ('plant_id', 'session_id', 'capture_date', 'geometry_sha256', 'case_id', 'surface_id',
                'triangle_id', 'organ_id', 'leaflet_id', 'area_mm2', 'received_both_faces_umol_s',
                'absorbed_umol_s', 'received_both_faces_per_one_sided_area_umol_m2_s',
                'absorbed_per_one_sided_area_umol_m2_s', 'status')
RUN_COLUMNS = ('bridge_run_id', 'scene_sha256', 'solver_jobs_sha256')


def identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise StudyError(f'{label}: expected a safe explicit identifier')
    return value


def resolve(base: Path, value: str) -> Path:
    return (base / value).resolve()


def capture_date(objects: list[dict]) -> str | None:
    """Only explicit capture dates; session names and file times are not dates."""
    values = set()
    for obj in objects:
        for key in ('capture_date', 'captured_at'):
            value = obj.get(key)
            if value in (None, '', 'unavailable'):
                continue
            try:
                values.add(date.fromisoformat(value).isoformat() if key == 'capture_date'
                           else pv.aware_time(value, key).date().isoformat())
            except (ValueError, TypeError) as exc:
                raise StudyError(f'Invalid explicit {key}: {value}') from exc
    if len(values) > 1:
        raise StudyError('Conflicting capture dates')
    return next(iter(values), None)


def measured_views(capture: dict, grouping: dict | None) -> list[dict]:
    """Reuse the elevation formula from regroup_traits; never its hardcoded centre."""
    centre = threshold = None
    if grouping:
        centre = np.asarray(grouping.get('centre_board_mm'), dtype=float)
        threshold = number(grouping.get('top_elevation_gt_deg'), 'top_elevation_gt_deg')
        if centre.shape != (3,) or not np.isfinite(centre).all() or not -90 <= threshold <= 90:
            raise StudyError('Invalid measured-elevation grouping settings')
        if not grouping.get('source'):
            raise StudyError('View grouping needs a source for its centre and threshold')
    result, seen = [], set()
    for frame in capture.get('frames', []):
        photo = frame['image_file']
        if photo in seen:
            raise StudyError('Duplicate capture photo')
        seen.add(photo)
        elevation = None
        if centre is not None and frame.get('rvec_rad') is not None and frame.get('tvec_mm') is not None:
            rvec = np.asarray(frame['rvec_rad'], dtype=float)
            tvec = np.asarray(frame['tvec_mm'], dtype=float)
            if rvec.shape != (3,) or tvec.shape != (3,) or not np.isfinite([rvec, tvec]).all():
                raise StudyError('Nonfinite or malformed camera pose')
            rotation = cv2.Rodrigues(rvec)[0]
            v = -rotation.T @ tvec - centre
            if np.linalg.norm(v) == 0:
                raise StudyError('Camera and grouping centre coincide')
            elevation = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        result.append(dict(photo=photo, elevation_deg=elevation,
                           group=None if elevation is None else ('top' if elevation > threshold else 'ring'),
                           passes_pose_gate=frame.get('passes_pose_gate'),
                           reprojection_error_px=frame.get('reprojection_error_px')))
    return result


def audit(run: Path, plant: str, session: str, study_path: Path, case_id: str | None = None,
          metadata_path: Path | None = None) -> dict:
    """Read-only join and optical audit. Integrity conflicts fail before any output."""
    identifier(plant, 'plant'); identifier(session, 'session')
    run, study_path = run.resolve(), study_path.resolve()
    snap = load_snapshot(run, plant)
    if snap['hierarchy']['session_id'] != session:
        raise StudyError('Requested session differs from plant snapshot')
    sources = dict(snap['source_hashes'])
    media_hashes: dict[str, str] = {}
    missing: list[dict] = []
    blockers: list[str] = []

    def need(key: str, reason: str, blocks: str = 'plant light preparation') -> None:
        missing.append(dict(entity='plant_bridge', input=key, status='unavailable', reason=reason, blocks=blocks))

    def track(path: Path, expected: str | None = None, required: bool = True) -> bool:
        path = path.resolve()
        if not path.is_file():
            if required:
                raise StudyError(f'Missing referenced input: {path}')
            need(str(path), 'referenced file missing', 'source inspection')
            return False
        actual = file_hash(path)
        if expected and actual != expected:
            raise StudyError(f'Input hash mismatch: {path}')
        if str(path) in sources and sources[str(path)] != actual:
            raise StudyError(f'Input changed during audit: {path}')
        sources[str(path)] = actual
        return True

    metadata = read_json(metadata_path) if metadata_path else {}
    if metadata_path:
        track(metadata_path)
        if metadata.get('plant_id') != plant or metadata.get('session_id') != session:
            raise StudyError('Source metadata plant/session mismatch')
        if metadata.get('capture_date') and not metadata.get('source_hashes'):
            raise StudyError('Capture date mapping needs source_hashes provenance')
        for path, expected in metadata.get('source_hashes', {}).items():
            if not expected:
                raise StudyError('Empty provenance hash')
            track(resolve(metadata_path.parent, path), expected)

    cap = snap['backup'].get('capture', {})
    cap_base = Path(snap['hierarchy']['inputs']['backup']).parent
    date_evidence = []
    # Structure poses can differ from annotation capture poses; retain both snapshots.
    sinputs = snap['structure'].get('inputs', {})
    for key in ('capture', 'backup', 'growth', 'name_decisions'):
        if sinputs.get(key):
            path = resolve(Path(snap['hierarchy']['inputs']['structure']), sinputs[key])
            track(path, sinputs.get(key + '_sha256'))
            if key == 'capture':
                cap = read_json(path); cap_base = path.parent
            if key == 'growth':
                growth = read_json(path)
                if growth.get('plant_id') != plant or growth.get('session_id', session) != session:
                    raise StudyError('Referenced growth metadata plant/session mismatch')
                if growth.get('capture_date'):
                    # The structure explicitly names this record as its capture input.
                    date_evidence.append(dict(source=str(path), capture_date=growth['capture_date']))
    if cap.get('plant_id', plant) != plant or cap.get('session_id') != session:
        raise StudyError('Pose capture plant/session mismatch')
    for key in ('intrinsics', 'board_spec'):
        if cap.get(key + '_path'):
            track(resolve(cap_base, cap[key + '_path']), cap.get(key + '_sha256'), required=False)
    for image in cap.get('frames', []):
        for path_key, hash_key in (('image_path', 'sha256'), ('source_path', 'source_sha256')):
            if not image.get(path_key):
                continue
            path = resolve(cap_base, image[path_key])
            if not path.is_file():
                need(str(path), 'Source image unavailable', 'source image inspection')
                continue
            expected = image.get(hash_key)
            actual = file_hash(path)
            if expected and expected != actual:
                raise StudyError(f'Source image hash mismatch: {path}')
            media_hashes[str(path)] = actual
    frame = cap.get('coordinate_frame')
    if not frame or not frame.startswith('board'):
        need('coordinate_frame', 'Explicit board frame missing'); blockers.append('board_frame_unavailable')
    stamp = capture_date([cap, snap['backup'].get('capture', {}), snap['hierarchy'], metadata, *date_evidence])
    if stamp is None:
        need('capture_date', 'No explicit capture date; filenames/session IDs are not interpreted', 'dated joins and plant light preparation')
        blockers.append('capture_date_unavailable')
    graph = make_graph(snap)
    trait_path = run / plant / 'traits.json'
    traits = None
    if trait_path.is_file():
        track(trait_path)
        traits = read_json(trait_path)
        if traits.get('plant_id', plant) != plant or traits.get('session_id', session) not in (None, session):
            raise StudyError('Traits plant/session mismatch')
        if capture_date([traits]) not in (None, stamp):
            raise StudyError('Trait date conflicts with capture date')
    views = measured_views(cap, metadata.get('view_grouping'))
    if not metadata.get('view_grouping'):
        need('view_grouping', 'No source-backed centre/threshold supplied; code-prefix groups not reused', 'top/ring trait comparisons')
    motion = metadata.get('movement') or cap.get('movement')
    if motion:
        groups = motion.get('groups', [])
        ids = [g['group_id'] for g in groups]
        if len(ids) != len(set(ids)):
            raise StudyError('Duplicate movement group')
        included = [g for g in groups if g.get('included_in_optical_scene')]
        if any(g.get('alignment_status') not in ('reference', 'accepted') for g in included):
            blockers.append('unaligned_movement_group')
        # This adapter never estimates or applies registration transforms.
        if len(included) > 1 and not motion.get('registered_geometry_sha256'):
            blockers.append('registered_geometry_provenance_unavailable')
    else:
        need('movement', 'Movement review unavailable; no grouping or registration inferred', 'physical geometry validation')

    study = pv.load_study(study_path); track(study_path)
    for key in STUDY_FILES:
        if study.get(key):
            track(resolve(study_path.parent, study[key]), required=False)
    ids = [identifier(c.get('case_id'), 'case_id') for c in study.get('cases', [])]
    if len(ids) != len(set(ids)) or any(i + '__empty_twin' in ids for i in ids):
        raise StudyError('Duplicate or reserved empty-twin case ID')
    candidates = [c for c in study.get('cases', []) if c.get('stage') == 'plant'
                  and (c.get('case_id') == case_id if case_id else
                       c.get('plant_id') == plant and c.get('session_id') == session)]
    if len(candidates) > 1:
        raise StudyError('Ambiguous plant cases; supply --case')
    case = candidates[0] if candidates else None
    if case and (case.get('plant_id') != plant or case.get('session_id') != session):
        raise StudyError('Light case plant/session mismatch')
    if case and stamp is not None and case.get('geometry_snapshot_date') != stamp:
        raise StudyError('Light case capture date mismatch')
    paudit = pv.validate_inputs(study_path)
    solver = study.get('solver') or {}
    for key in ('rays', 'max_depth', 'max_local_native_jobs'):
        if solver.get(key) is not None and (type(solver[key]) is not int or solver[key] < 1):
            raise StudyError(f'solver.{key} must be a positive integer')
    if solver.get('seed') is not None and type(solver['seed']) is not int:
        raise StudyError('solver.seed must be an integer')
    for key, id_key in (('sensors', 'sensor_id'), ('lamp_conditions', 'fixture_id'), ('rig_surfaces', 'surface_id')):
        path = resolve(study_path.parent, study[key]) if study.get(key) else None
        if path and path.is_file():
            values = [r[id_key] for r in rows(path)]
            if len(values) != len(set(values)):
                raise StudyError(f'Duplicate {key} records')
    missing.extend(paudit['missing_physical_inputs'])
    blockers.extend(paudit['issues'])
    surface_records, meshes, review = [], [], {}
    if not case:
        need('plant_case', 'No explicit matching plant/session case in optical study')
        blockers.append('plant_case_unavailable')
    else:
        for key in ('geometry', 'geometry_variant_id', 'surface_review'):
            if not case.get(key):
                need(key, 'Not supplied in selected plant case'); blockers.append(key + '_unavailable')
        for key in ('geometry', 'surface_review'):
            if case.get(key):
                track(resolve(study_path.parent, case[key]), required=False)
        geometry_path = resolve(study_path.parent, case['geometry']) if case.get('geometry') else None
        review_path = resolve(study_path.parent, case['surface_review']) if case.get('surface_review') else None
        if review_path and review_path.is_file():
            review = read_json(review_path)
            for path, expected in review.get('source_hashes', {}).items():
                track(resolve(review_path.parent, path), expected)
        if geometry_path and geometry_path.is_file() and case.get('geometry_variant_id'):
            variants = load_geometry([geometry_path])
            surfaces = variants.get(case['geometry_variant_id'])
            if surfaces is None:
                blockers.append('no_leaf_surfaces')
            else:
                if any(p != plant or s != session for p, s in zip(surfaces.plant_ids, surfaces.session_ids)):
                    raise StudyError('Selected geometry contains a foreign plant/session')
                areas = triangle_areas_mm2(surfaces.vertices_board_mm)
                by_identity = {}
                for node in graph['nodes']:
                    meta = node['metadata']
                    if node['module'] == 'Leaflet' and not node['proposed']:
                        key = (meta.get('organ_id'), meta.get('leaflet_id'))
                        by_identity.setdefault(key, []).append(meta)
                for i in range(surfaces.n_triangles):
                    identity = by_identity.get((surfaces.organ_ids[i], surfaces.leaflet_ids[i]), [])
                    photos = sorted({p for meta in identity for p in meta.get('photos', [])})
                    surface_records.append(dict(surface_id=f't{i:07d}', plant_id=plant, session_id=session,
                        capture_date=stamp, geometry_variant_id=surfaces.geometry_variant_id,
                        geometry_sha256=file_hash(geometry_path), organ_id=surfaces.organ_ids[i],
                        leaflet_id=surfaces.leaflet_ids[i], triangle_id=surfaces.triangle_ids[i],
                        geometry_source=surfaces.geometry_sources[i], area_mm2=float(areas[i]),
                        optical_status='pending_gate', source_photos=json.dumps(photos)))
                meshes = surfaces.vertices_board_mm.tolist()
            if all(case.get(k) for k in ('geometry', 'surface_review', 'geometry_variant_id')):
                gate, _ = surface_gate(dict(case, snapshot_date=case.get('geometry_snapshot_date'),
                    instances=[dict(source_plant_id=plant, source_session_id=session)]), study_path.parent)
                if any('mismatch' in b for b in gate):
                    raise StudyError('Surface review integrity conflict: ' + ', '.join(gate))
                blockers.extend(gate)
                if motion and motion.get('registered_geometry_sha256') not in (None, file_hash(geometry_path)):
                    raise StudyError('Registered geometry hash mismatch')
        else:
            blockers.append('no_leaf_surfaces')
    if not surface_records:
        blockers.append('no_leaf_surfaces')
        need('leaf_surfaces', 'Positions, bbox extents and hulls are not optical leaf surfaces')
    for key, cols in (('points', pv.POINT_COLUMNS), ('readings', pv.READING_COLUMNS)):
        path = resolve(study_path.parent, study[key]) if study.get(key) else None
        if not path or not path.is_file():
            continue
        records = rows(path, cols)
        if key == 'points':
            for r in records:
                identifier(r['point_id'], 'point_id')
        for r in records:
            if case and r['case_id'] == case['case_id']:
                if r.get('plant_id') != plant or r.get('stage') != 'plant':
                    raise StudyError(f'{key}: plant/stage mismatch')
                if key == 'readings' and (r.get('session_id') != session or
                                          stamp is None or r.get('geometry_snapshot_date') != stamp):
                    raise StudyError('Reading session/date mismatch or unavailable')
    if study.get('scene') and resolve(study_path.parent, study['scene']).is_file():
        scene = load_scene(resolve(study_path.parent, study['scene']))
        if not any(p.plant_id == plant for p in scene.plants):
            blockers.append('plant_placement_unavailable')
    else:
        blockers.append('light_scene_unavailable')
    completeness = review.get('geometry_completeness', 'unknown')
    if completeness == 'unknown':
        need('geometry_completeness', 'Not declared by surface review', 'whole-plant light interpretation')
    for r in surface_records:
        r['optical_status'] = 'blocked' if blockers else 'eligible_diagnostic'
    return dict(schema=SCHEMA, plant_id=plant, session_id=session, capture_date=stamp, run_id=run.name,
                run=str(run), study=str(study_path), case=case, source_hashes=sources, media_hashes=media_hashes,
                capture_date_evidence=date_evidence,
                blockers=sorted(set(blockers)), missing_inputs=missing, source_coordinate_frame=frame,
                length_unit='mm', quality=cap.get('quality'), pose_provenance=sinputs,
                mask_review=snap['status'].get('stages', {}).get('S2'),
                geometry_counts=dict(Counter(g['kind'] for g in graph['geometry'])),
                optical_triangle_count=len(surface_records),
                surface_area_mm2=sum(r['area_mm2'] for r in surface_records) if surface_records else None,
                geometry_completeness=completeness, surface_review=review,
                source_geometry_counts=dict(Counter(r['geometry_source'] for r in surface_records)),
                view_grouping=metadata.get('view_grouping'), views=views, movement=motion,
                structure_historical_hash_available=False, validated_metric=False, suitable_for_ppfd=False,
                excluded_plant_cases=[c['case_id'] for c in study.get('cases', []) if c.get('stage') == 'plant' and c != case],
                graph=graph, surfaces=surface_records, vertices_board_mm=meshes, diagnostic_traits=traits)


def freeze_inputs(audit_result: dict, folder: Path) -> tuple[Path, dict]:
    """Copy consumed inputs without basename collisions; rewrite only the derived study."""
    target = folder / 'inputs'; target.mkdir()
    copies = {}
    for i, (source, expected) in enumerate(sorted(audit_result['source_hashes'].items())):
        src = Path(source)
        dest = target / str(i) / src.name; dest.parent.mkdir()
        shutil.copyfile(src, dest)
        if file_hash(dest) != expected or file_hash(src) != expected:
            raise StudyError(f'Source changed while freezing: {source}')
        copies[source] = str(dest)
    study_path = Path(audit_result['study'])
    study = read_json(Path(copies[str(study_path)]))
    # Retain empty-rig development/validation cases; exclude unrelated plant cases.
    case = audit_result['case']
    study['cases'] = [c for c in study.get('cases', []) if c.get('stage') != 'plant' or c == case]
    kept = {c['case_id'] for c in study['cases']}
    for key in STUDY_FILES:
        if study.get(key):
            source = str(resolve(study_path.parent, study[key]))
            study[key] = copies.get(source, str(target / ('missing_' + key)))
    for c in study['cases']:
        for key in ('geometry', 'surface_review'):
            if c.get(key):
                source = str(resolve(study_path.parent, c[key]))
                c[key] = copies.get(source, str(target / ('missing_' + key)))
    for key, cols in (('points', pv.POINT_COLUMNS), ('readings', pv.READING_COLUMNS)):
        if study.get(key) and Path(study[key]).is_file():
            records = [r for r in rows(Path(study[key]), cols) if r['case_id'] in kept]
            path = target / f'scoped_{key}.csv'; write_rows(path, records, cols)
            study[key] = str(path)
    path = target / 'scoped_study.json'; write_json(path, study)
    return path, copies


def constant_light_seconds(scene_path: Path | None) -> float | None:
    """Only equal full-output schedules can scale an aggregate full-output flux."""
    if scene_path is None or not scene_path.is_file():
        return None
    scene = load_scene(scene_path)
    schedules = [[(s.start_hhmm, s.end_hhmm, s.output_fraction) for s in f.schedule] for f in scene.fixtures]
    if not schedules or not schedules[0] or any(s != schedules[0] for s in schedules):
        return None
    if any(s.output_fraction != 1 for s in scene.fixtures[0].schedule):
        return None
    return sum(s.duration_minutes * 60 for s in scene.fixtures[0].schedule)


def aggregate_light(records: list[dict], seconds: float | None, completeness: str) -> dict:
    """Sum flux, then divide by covered area; triangle count never weights PPFD."""
    if not records:
        return dict(status='unavailable', covered_area_mm2=None, received_both_faces_umol_s=None,
                    absorbed_umol_s=None, received_both_faces_area_weighted_umol_m2_s=None,
                    received_both_faces_dli_mol_m2_d=None, absorbed_mmol_d=None,
                    photosynthesis=None, whole_plant_absorbed_umol_s=None)
    values = [(number(r['area_mm2'], 'area_mm2'), number(r['received_both_faces_umol_s'], 'received'),
               number(r['absorbed_umol_s'], 'absorbed')) for r in records]
    if any(a <= 0 or r < 0 or b < 0 for a, r, b in values):
        raise StudyError('Invalid area or photon flux')
    area = sum(a for a, _, _ in values)
    received = sum(r for _, r, _ in values); absorbed = sum(b for _, _, b in values)
    mean = received / (area / pv.MM_PER_M ** 2)
    return dict(status='diagnostic_declared_inputs', covered_area_mm2=area,
                received_both_faces_umol_s=received, absorbed_umol_s=absorbed,
                received_both_faces_area_weighted_umol_m2_s=mean,
                received_both_faces_dli_mol_m2_d=None if seconds is None else mean * seconds / 1e6,
                absorbed_mmol_d=None if seconds is None else absorbed * seconds / 1000,
                schedule_basis='identical full-output daily intervals' if seconds is not None else 'unavailable',
                photosynthesis=None,
                whole_plant_absorbed_umol_s=absorbed if completeness == 'whole_plant_reviewed' else None)


def result_tables(audit_result: dict, light_dir: Path, out: Path) -> tuple[dict, list[dict]]:
    case = audit_result['case']
    leaf_path = light_dir / 'leaf_triangle_light.csv'
    leaves = rows(leaf_path) if leaf_path.is_file() else []
    selected = [r for r in leaves if case and r['case_id'] == case['case_id']]
    inventory = {r['surface_id']: r for r in audit_result['surfaces']}
    jobs_path = light_dir / 'prepared/jobs.json'
    provenance = dict(bridge_run_id=out.name,
                      scene_sha256=read_json(jobs_path)['scene_sha256'] if jobs_path.is_file() else None,
                      solver_jobs_sha256=file_hash(jobs_path) if jobs_path.is_file() else None)
    linked, seen = [], set()
    for row in selected:
        sid = row['surface_id']; original = inventory.get(sid)
        if original is None or sid in seen:
            raise StudyError('Unknown or duplicate solver surface ID')
        seen.add(sid)
        for k in ('plant_id', 'session_id', 'organ_id', 'leaflet_id', 'triangle_id'):
            if row[k] != original[k]:
                raise StudyError(f'Solver surface identity mismatch: {k}')
        if not math.isclose(number(row['area_mm2'], 'area'), original['area_mm2'], rel_tol=1e-9):
            raise StudyError('Solver surface area mismatch')
        linked.append({**{k: row[k] for k in LINK_COLUMNS if k in row},
                       'capture_date': audit_result['capture_date'], 'geometry_sha256': original['geometry_sha256'],
                       'status': 'diagnostic_declared_inputs', **provenance})
    if selected and seen != set(inventory):
        raise StudyError('Incomplete solver leaf results')
    write_rows(out / 'surface_light_links.csv', linked, LINK_COLUMNS + RUN_COLUMNS)
    sensors_path = light_dir / 'sensor_predictions.csv'
    sensor_rows, seen_points = [], set()
    if sensors_path.is_file():
        for row in rows(sensors_path, pv.PREDICTION_COLUMNS):
            if not case or row['case_id'] != case['case_id']:
                continue
            if row['point_id'] in seen_points:
                raise StudyError('Duplicate solver sensor point')
            seen_points.add(row['point_id'])
            sensor_rows.append(dict(row, plant_id=audit_result['plant_id'], session_id=audit_result['session_id'],
                                    capture_date=audit_result['capture_date'],
                                    geometry_sha256=next(iter(inventory.values()))['geometry_sha256'] if inventory else None,
                                    **provenance))
    write_rows(out / 'sensor_light_links.csv', sensor_rows,
               ('plant_id', 'session_id', 'capture_date', 'geometry_sha256') + pv.PREDICTION_COLUMNS + RUN_COLUMNS)
    study = read_json(out / 'inputs/scoped_study.json')
    seconds = constant_light_seconds(Path(study['scene']) if study.get('scene') else None)
    summary = aggregate_light(linked, seconds, audit_result['geometry_completeness'])
    summary.update(plant_id=audit_result['plant_id'], session_id=audit_result['session_id'],
                   capture_date=audit_result['capture_date'], case_id=case['case_id'] if case else None,
                   data_kind=case.get('data_kind', 'observed_geometry') if case else 'observed_geometry',
                   geometry_completeness=audit_result['geometry_completeness'], validated_metric=False,
                   legacy_intercepted_mmol_d_basis='projected-area proxy; never absorbed photons')
    write_json(out / 'light_summary.json', summary)
    # Persist numerical plant summaries in CSV as well as viewer metadata.
    write_rows(out / 'plant_light.csv', [summary], tuple(summary))
    return summary, linked


def report(folder: Path, audit_result: dict, status: str, summary: dict, linked: list[dict]) -> None:
    study = read_json(folder / 'inputs/scoped_study.json')
    scene_path = Path(study['scene']) if study.get('scene') else None
    grow_vertices = None
    if audit_result['surfaces'] and scene_path and scene_path.is_file():
        case = next(c for c in study['cases'] if c['case_id'] == audit_result['case']['case_id'])
        scene = load_scene(scene_path)
        placements = {p.plant_id: p for p in scene.plants}
        if audit_result['plant_id'] in placements:
            mesh = load_geometry([Path(case['geometry'])])[case['geometry_variant_id']]
            grow_vertices = place_in_grow_frame(mesh, placements).tolist()
    def href(path: Path) -> str:
        return quote(os.path.relpath(path, folder).replace('\\', '/'), safe='/.:')
    run = Path(audit_result['run'])
    existing = run / 'site.html'
    if not existing.is_file():
        existing = run / audit_result['plant_id'] / 'hierarchy/viewer.html'
    predictions = folder / 'light/sensor_predictions.csv'
    points = rows(Path(study['points'])) if study.get('points') and Path(study['points']).is_file() else []
    data = dict(audit=audit_result, status=status, summary=summary, linked=linked,
                vertices_grow_mm=grow_vertices, points=points,
                sensors=rows(predictions) if predictions.is_file() else [],
                source_viewer=href(existing) if existing.is_file() else None,
                ppfd_report='light/report.html' if (folder / 'light/report.html').is_file() else None)
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (folder / 'report.html').write_text(template.replace('__DATA__', payload), encoding='utf-8')


def connect(run: Path, plant: str, session: str, study: Path, out: Path, *, case_id: str | None = None,
            metadata: Path | None = None, prepare: bool = True, execute: bool = False,
            java: str = 'java', core: Path = pv.CORE) -> dict:
    result = audit(run, plant, session, study, case_id, metadata)
    if execute and not prepare:
        raise StudyError('Execution requires preparation')
    folder = output_dir(out)
    frozen, copies = freeze_inputs(result, folder)
    write_json(folder / 'input_audit.json', result)
    write_json(folder / 'structure_snapshot.json', result['graph'])
    write_rows(folder / 'surfaces.csv', result['surfaces'], SURFACE_COLUMNS)
    write_rows(folder / 'views.csv', result['views'], ('photo', 'elevation_deg', 'group', 'passes_pose_gate', 'reprojection_error_px'))
    status = 'blocked' if result['blockers'] else 'audited'
    steps = {}
    if prepare and not result['blockers']:
        solver = read_json(frozen).get('solver') or {}
        if execute:
            # Existing execute() does not enforce the study's job budget itself.
            limit = solver.get('max_local_native_jobs')
            if type(limit) is not int or limit < 1:
                raise StudyError('Set a positive max_local_native_jobs before execution')
        pv.run(frozen, folder / 'light', execute=False)
        jobs_path = folder / 'light/prepared/jobs.json'
        if jobs_path.is_file():
            jobs = read_json(jobs_path)
            selected = next(c for c in jobs['cases'] if c['case_id'] == result['case']['case_id'])
            status = 'prepared' if selected['status'] == 'prepared' else 'blocked'
            result['blockers'].extend(selected['blockers'])
            if execute and status == 'prepared':
                count = sum(len(c['jobs']) for c in jobs['cases'] if c['status'] == 'prepared')
                if count > limit:
                    raise StudyError(f'Native job count {count} exceeds declared limit {limit}')
                pv.density_execute(folder / 'light/prepared', folder / 'light/execution', core, java)
                collected = pv.collect(folder / 'light/prepared', folder / 'light/execution', folder / 'light')
                status = 'native_completed_physical_validation_pending' if not collected['failures'] else 'native_failed_or_partial'
                points = pv._csv(frozen.parent, read_json(frozen).get('points'), pv.POINT_COLUMNS)
                comparison = None
                if collected['predictions']:
                    readings = Path(read_json(frozen)['readings']) if read_json(frozen).get('readings') else None
                    if readings and readings.is_file():
                        comparison = pv.compare_ppfd(folder / 'light/sensor_predictions.csv', readings,
                                                     folder / 'light/split_manifest.json', read_json(frozen))
                        write_json(folder / 'light/comparison.json', comparison)
                (folder / 'light/report.html').write_text(pv.report_html(collected['predictions'], comparison,
                    read_json(folder / 'light/readiness.json'), points, 'Plant light: diagnostic result'), encoding='utf-8')
        else:
            status = 'blocked'
            steps = read_json(folder / 'light/run_manifest.json')['steps']
            result['blockers'].append(steps.get('prepare', 'preparation_unavailable'))
    write_rows(folder / 'missing_inputs.csv', result['missing_inputs'], ('entity', 'input', 'status', 'reason', 'blocks'))
    summary, linked = result_tables(result, folder / 'light', folder)
    report(folder, result, status, summary, linked)
    changed = [p for p, h in {**result['source_hashes'], **result['media_hashes']}.items()
               if not Path(p).is_file() or file_hash(Path(p)) != h]
    if changed:
        raise StudyError('Source changed during connection: ' + ', '.join(changed))
    code_paths = [Path(__file__), Path(__file__).with_suffix('.html'), Path(pv.__file__),
                  ROOT / 'tools/groimp_density.py', ROOT / 'tools/groimp_structure_export.py',
                  ROOT / 'tools/groimp_bridge.py', ROOT / 'tools/leaf_light/common.py',
                  ROOT / 'tools/groimp_light_sanity.py', ROOT / 'tomato3d/light/geometry.py',
                  ROOT / 'tomato3d/light/frames.py', ROOT / 'tomato3d/light/scene.py',
                  ROOT / 'tomato3d/light/raytrace.py', ROOT / 'tomato3d/io/schemas.py']
    manifest = dict(schema=SCHEMA, plant_id=plant, session_id=session, capture_date=result['capture_date'],
        run=str(run.resolve()), case_id=result['case']['case_id'] if result['case'] else None,
        data_kind=(result['case'] or {}).get('data_kind', 'observed_geometry'),
        status=status, blockers=sorted(set(result['blockers'])), source_hashes=result['source_hashes'],
        media_hashes=result['media_hashes'],
        frozen_inputs=copies, code_hashes={str(p): file_hash(p) for p in code_paths},
        artifact_hashes={str(p.relative_to(folder)).replace('\\', '/'): file_hash(p) for p in folder.rglob('*') if p.is_file()},
        validated_metric=False, suitable_for_ppfd=False, source_files_unchanged=True)
    write_json(folder / 'manifest.json', manifest)
    return manifest


def stale(folder: Path) -> dict:
    folder = folder.resolve()
    manifest = read_json(folder / 'manifest.json')
    if manifest.get('schema') != SCHEMA:
        raise StudyError('Not a plant light bridge result')
    changed = []
    for key in ('source_hashes', 'media_hashes', 'code_hashes', 'artifact_hashes'):
        for name, expected in manifest.get(key, {}).items():
            path = folder / name if key == 'artifact_hashes' else Path(name)
            if key == 'artifact_hashes' and not path.resolve().is_relative_to(folder):
                raise StudyError('Unsafe artifact path')
            if not path.is_file() or file_hash(path) != expected:
                changed.append(dict(kind=key, path=str(path)))
    return dict(stale=bool(changed), changed=changed, status=manifest['status'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('audit', 'prepare'):
        cmd = actions.add_parser(name)
        for key in ('run', 'study', 'out'):
            cmd.add_argument('--' + key, type=Path, required=True)
        for key in ('plant', 'session'):
            cmd.add_argument('--' + key, required=True)
        cmd.add_argument('--case'); cmd.add_argument('--metadata', type=Path)
        if name == 'prepare':
            cmd.add_argument('--execute', action='store_true')
            cmd.add_argument('--java', default='java'); cmd.add_argument('--core', type=Path, default=pv.CORE)
    check = actions.add_parser('stale'); check.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'stale':
            value = stale(args.result)
            print(json.dumps(value, ensure_ascii=True)); return int(value['stale'])
        value = connect(args.run, args.plant, args.session, args.study, args.out, case_id=args.case,
                        metadata=args.metadata, prepare=args.action == 'prepare',
                        execute=getattr(args, 'execute', False), java=getattr(args, 'java', 'java'),
                        core=getattr(args, 'core', pv.CORE))
        print(json.dumps({k: value[k] for k in ('status', 'plant_id', 'session_id', 'blockers')}, ensure_ascii=True))
        return int(value['status'] in ('blocked', 'native_failed_or_partial'))
    except (StudyError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'plant-light bridge: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
