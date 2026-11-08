"""Rebuild diagnostic photo/3D/depth artifacts from an explicit editor backup.

python -m tools.rebuild_leaf_edit_session --session build/.../edit_session01
  --backup build/.../backup.json --context build/.../manual_candidates.review05.json
  --out build/.../edited_run01

The current backup is authoritative for edits/deletions. The explicit context
keeps noneditable partial/2D observations current. Nothing is promoted to metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from html import escape
import json
from pathlib import Path
import re
import shutil

import numpy as np

from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json, write_rows
from tools.leaf_depth_records import board_evidence_fields, camera_coordinates, export_records
from tools.manual_leaf_preview import build, diagnostic_point, project, board_reference
from tools.prepare_leaf_edit_session import basename, verify_path
from tools.editor_point_review import write_point_review
from tools import reconstruction_request as reference_tool
from tools.leaflet_extent import METHOD as EXTENT_METHOD, ExtentStatus, feasible_extent, point_relation


def compact_capture(capture: dict) -> dict:
    """Mirror only the annotator's documented JSON normalization, not geometry."""
    doc = deepcopy(capture)
    if not isinstance(doc.get('intrinsics'), dict) or not isinstance(doc.get('frames'), list):
        raise StudyError('Backup capture is missing camera evidence')
    doc.setdefault('plant_id', '')
    doc.setdefault('session_id', '')
    doc['intrinsics'].setdefault('camera_id', '')
    for frame in doc['frames']:
        frame.setdefault('rvec_rad', None)
        frame.setdefault('tvec_mm', None)
        frame.setdefault('view_type', 'top' if frame.get('observed_view_group') == 'near_top' else '')
        frame.setdefault('top_position', '')
    doc.setdefault('variants', [])
    for variant in doc['variants']:
        variant.setdefault('view_count', None)
        variant.setdefault('image_files', [])
    return doc


def checked_inputs(session: Path, backup_path: Path, context_path: Path) -> tuple[dict, dict, dict, dict]:
    report = read_json(session / 'session_report.json')
    for name, field in (('capture.json', 'capture_sha256'), ('seed.backup.json', 'seed_sha256'),
                        ('annotations.source.json', 'annotations_sha256')):
        if file_hash(session / name) != report[field]:
            raise StudyError(f'Prepared file hash mismatch: {name}')
    capture, seed = read_json(session / 'capture.json'), read_json(session / 'seed.backup.json')
    backup, context = read_json(backup_path), read_json(context_path)
    if (capture.get('schema_version') != 'diagnostic-preview-capture-1' or
            context.get('schema_version') != 'manual-leaf-preview-1'):
        raise StudyError('Expected diagnostic capture and explicit manual context')
    if backup.get('plant_id') != seed['plant_id'] or backup.get('session_id') != seed['session_id']:
        raise StudyError('Backup plant/session identity mismatch')
    if compact_capture(backup.get('capture') or {}) != compact_capture(capture):
        raise StudyError('Backup capture differs from the prepared cameras/quality/evidence')
    intrinsics_path = verify_path(capture['intrinsics_path'], capture['intrinsics_sha256'], session)
    if read_json(intrinsics_path) != capture['intrinsics']:
        raise StudyError('Embedded intrinsics differ from source')
    verify_path(capture['board_spec_path'], capture['board_spec_sha256'], session)
    board_reference(capture)
    capture['intrinsics_path'] = str(intrinsics_path)
    capture['board_spec_path'] = str((session / capture['board_spec_path']).resolve())
    for frame in capture['frames']:
        frame['image_path'] = str(verify_path(frame['image_path'], frame['sha256'], session))
        board_evidence_fields(frame)
    return capture, seed, backup, context


def current_state(backup: dict) -> dict:
    variants = backup.get('variants', {})
    if '__all__' not in variants:
        raise StudyError('The prepared all-photo variant is missing')
    if any(key != '__all__' and (value.get('leaflets') or value.get('boxes')) for key, value in variants.items()):
        raise StudyError('Independent subset variants need a separate rebuild; never pool their labels')
    return variants['__all__']


# Review drafts (unassigned automatic masks, landmark candidates) are evidence for the
# editor, never labels: they are counted in the report and left out of the annotations.
EDITOR_METADATA_SKIP = ('pts', 'contours', 'original_candidate_observations', 'landmark_candidates')


def draft_counts(state: dict) -> dict:
    drafts = Counter(d.get('state') for d in (state.get('auto_drafts') or {}).values())
    landmarks = Counter(c.get('state') for leaf in state.get('leaflets', {}).values()
                        for views in (leaf.get('landmark_candidates') or {}).values() for c in views)
    accepted_from = Counter(entry.get('source') for leaf in state.get('leaflets', {}).values()
                            for views in (leaf.get('point_provenance') or {}).values() for entry in views.values())
    return dict(auto_mask_drafts_by_state=dict(drafts), landmark_candidates_by_state=dict(landmarks),
                accepted_point_observations_by_source=dict(accepted_from))


def semantic_name(leaf: dict, point_id: str) -> str | None:
    """Point meaning is per physical point ID, identical in every photo."""
    names = leaf.get('point_semantic_names') or {}
    if not isinstance(names, dict):
        raise StudyError('Malformed point semantic names')
    value = names.get(point_id)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise StudyError('Malformed point semantic name')
    return value


def convert(backup: dict, seed: dict, context: dict, capture: dict) -> tuple[dict, list[dict], dict]:
    """Convert explicit physical boundary points; never use polygon indices as matches."""
    state, initial = current_state(backup), current_state(seed)
    frames = {basename(f['image_file']).lower(): f for f in capture['frames']}
    if len(frames) != len(capture['frames']):
        raise StudyError('Duplicate photo basename')
    original_size = [capture['intrinsics']['image_width_px'], capture['intrinsics']['image_height_px']]
    context_physical = {}
    for candidate in context.get('physical_point_candidates', []):
        identity = candidate['organ_id'] + '|' + candidate['leaflet_id']
        for observation in candidate['observations']:
            for point in observation['points']:
                context_physical[(identity, point['point_id'], basename(observation['image_file']).lower())] = (
                    np.asarray(point['xy_px']) * np.asarray(original_size) / observation['label_image_size_px']).tolist()
    context_by_key = {}
    for kind in ('leaflets', 'two_d_only_candidates'):
        for leaf in context.get(kind, []):
            key = leaf['organ_id'] + '|' + leaf['leaflet_id']
            if key in context_by_key:
                raise StudyError('Duplicate context identity')
            context_by_key[key] = (kind, deepcopy(leaf))
    managed = set(initial.get('leaflets', {}))
    active = state.get('leaflets', {})
    annotation = deepcopy(context)
    for kind in ('leaflets', 'two_d_only_candidates'):
        annotation[kind] = [deepcopy(leaf) for key, (category, leaf) in context_by_key.items()
                            if category == kind and key not in managed and key not in active]
    used_ids = {leaf['leaflet_id'] for kind in ('leaflets', 'two_d_only_candidates') for leaf in annotation[kind]}
    rows, leaf_reports, mapping = [], [], {}
    id_counts = Counter(leaf.get('leaflet_id') for leaf in active.values())
    for key, leaf in active.items():
        organ, leaf_id = leaf.get('organ_id'), leaf.get('leaflet_id')
        if not isinstance(organ, str) or not isinstance(leaf_id, str) or key != organ + '|' + leaf_id:
            raise StudyError('Malformed editor identity')
        if not organ or not leaf_id:
            raise StudyError('Empty editor identity')
        viewer_id = leaf_id
        if viewer_id == 'unavailable' or id_counts[viewer_id] > 1 or viewer_id in used_ids or not re.fullmatch(r'[A-Za-z0-9_.:-]+', viewer_id):
            viewer_id = 'edit_' + hashlib.sha256(key.encode()).hexdigest()[:16]
        if viewer_id in used_ids:
            raise StudyError('Output identity collision')
        used_ids.add(viewer_id)
        mapping[key] = viewer_id
        n = leaf.get('boundary_count', 0)
        if type(n) is not int or n < 0:
            raise StudyError('Invalid boundary count')
        pts = leaf.get('pts', {})
        row_start = len(rows)
        for pid, observed in pts.items():
            if not re.fullmatch(r'(?:b|i)\d+', pid):
                raise StudyError('Unknown physical point kind')
            if pid.startswith('b') and int(pid[1:]) >= n:
                raise StudyError('Boundary point exceeds boundary_count')
            for image_key, xy in observed.items():
                if image_key not in frames:
                    raise StudyError('Point belongs to an unknown photo')
                if (not isinstance(xy, list) or len(xy) != 2 or
                        any(type(value) not in (int, float) for value in xy)):
                    raise StudyError('Malformed pixel coordinate')
                pixel = np.asarray(xy, dtype=float)
                if not np.isfinite(pixel).all() or np.any(pixel < 0) or np.any(pixel >= original_size):
                    raise StudyError('Point outside original image')
                rows.append(dict(plant_id=backup.get('plant_id'), session_id=backup.get('session_id'),
                                 variant_id='__all__', organ_id=organ, leaflet_id=leaf_id, viewer_leaflet_id=viewer_id,
                                 point_id=pid, image_file=frames[image_key]['image_file'],
                                 x_px=float(pixel[0]), y_px=float(pixel[1]),
                                 point_kind='boundary' if pid.startswith('b') else 'interior',
                                 point_semantic_name=semantic_name(leaf, pid),
                                 observation_kind='image_observation',
                                 view_visibility=view_status(leaf, image_key)[0],
                                 view_identity_review=view_status(leaf, image_key)[1],
                                 context_depth_restriction=None))
        complete = [image_key for image_key, f in frames.items() if n >= 3 and
                    f['pose_status'] == 'solved' and all(image_key in pts.get(f'b{i}', {}) for i in range(n))]
        status = dict(editor_identity=key, viewer_leaflet_id=viewer_id, complete_solved_views=len(complete),
                      mesh_status='pending_boundary_correspondence', validated_metric=False)
        old = initial.get('leaflets', {}).get(key)
        boundary_pts = lambda value: {pid: obs for pid, obs in value.get('pts', {}).items() if pid.startswith('b')}
        unchanged = (old is not None and boundary_pts(leaf) == boundary_pts(old) and
                     all(leaf.get(field) == old.get(field) for field in ('boundary_count', 'contours')))
        if unchanged and key in context_by_key:
            kind, current = deepcopy(context_by_key[key])
            current['leaflet_id'] = viewer_id
            current['editor_metadata'] = {k: deepcopy(v) for k, v in leaf.items()
                                          if k not in EDITOR_METADATA_SKIP}
            if 'identity_history' in leaf:
                current['identity_history'] = deepcopy(leaf['identity_history'])
            if leaf.get('outline_completeness') == 'partial_observed_patch':
                current['outline_completeness'] = 'partial_observed_patch'
            current['review_status'] = 'candidate_unreviewed'
            current['validated_metric'] = False
            annotation[kind].append(current)
            status['mesh_status'] = 'unchanged_editor_label_uses_latest_explicit_context'
            status['context_kind'] = kind
            if kind == 'two_d_only_candidates':
                for row in rows[row_start:]:
                    matched_point = context_physical.get((key, row['point_id'], basename(row['image_file']).lower()))
                    if matched_point != [row['x_px'], row['y_px']]:
                        row['context_depth_restriction'] = 'context_physical_correspondence_unresolved'
            else:
                context_points = {}
                for entry in current['observations']:
                    for i, xy in enumerate(entry['boundary_xy_px']):
                        # Match seed creation's operation order: reassociation can
                        # change the last float bit and falsely invalidate a point.
                        context_points.setdefault(f'b{i}', {})[basename(entry['image_file']).lower()] = (
                            np.asarray(xy) * np.asarray(original_size) / entry['label_image_size_px']).tolist()
                if context_points != boundary_pts(leaf):
                    for row in rows[row_start:]:
                        if row['point_kind'] == 'boundary':
                            row['context_depth_restriction'] = 'unchanged_seed_points_superseded_by_context'
        elif len(complete) >= 2:
            item = {k: deepcopy(v) for k, v in leaf.items() if k not in EDITOR_METADATA_SKIP}
            item.update(leaflet_id=viewer_id, original_editor_identity=key,
                        description=leaf.get('description') or f'Edited candidate {organ}/{leaf_id}',
                        review_status='candidate_unreviewed', correspondence_status='candidate_unreviewed',
                        validated_metric=False,
                        observations=[dict(image_file=frames[k]['image_file'], label_image_size_px=original_size,
                                           boundary_xy_px=[pts[f'b{i}'][k] for i in range(n)]) for k in complete])
            annotation['leaflets'].append(item)
            status['mesh_status'] = 'submitted_for_diagnostic_reconstruction'
        elif len(rows) > row_start and key in context_by_key and context_by_key[key][0] == 'two_d_only_candidates':
            # A separately observed region can remain visible without filling the
            # missing physical boundary point or using polygon vertices as matches.
            current = deepcopy(context_by_key[key][1])
            current['leaflet_id'] = viewer_id
            current['editor_has_incomplete_physical_points'] = True
            annotation['two_d_only_candidates'].append(current)
            status['context_kind'] = 'two_d_only_candidates'
        leaf_reports.append(status)
    annotation.update(human_review='pending', annotation_source='Explicit editor backup plus versioned noneditable context; no learned model',
                      notice='Edited points remain unreviewed and metric-unvalidated. Complete boundary views form coarse patches. Sparse/interior point depths are separately exported; no missing boundary is filled.')
    report = dict(leaf_status=leaf_reports, identity_mapping=mapping,
                  deleted_seed_identities=sorted(managed - set(active)),
                  preserved_context_ids=[leaf['leaflet_id'] for key, (_, leaf) in context_by_key.items() if key not in managed and key not in active],
                  explicit_contours_preserved_in_backup=True, boxes_preserved_in_backup=True,
                  no_missing_point_imputation=True)
    return annotation, rows, report


def point_depths(rows: list[dict], capture: dict) -> list[dict]:
    """Triangulate each physical point independently, including sparse/interior points."""
    frames = {f['image_file']: f for f in capture['frames']}
    grouped = {}
    for row in rows:
        grouped.setdefault((row['organ_id'], row['leaflet_id'], row['point_id']), []).append(row)
    result = []
    for entries in grouped.values():
        usable = [(frames[r['image_file']], np.array([r['x_px'], r['y_px']])) for r in entries
                  if frames[r['image_file']]['pose_status'] == 'solved']
        xyz, diag, failure = None, None, 'fewer_than_two_solved_views'
        restriction = next((r.get('context_depth_restriction') for r in entries if r.get('context_depth_restriction')), None)
        if restriction:
            failure = restriction
        elif len(usable) >= 2:
            try:
                xyz, diag = diagnostic_point(usable, capture['intrinsics'], capture['quality'])
                failure = None
            except (ValueError, np.linalg.LinAlgError) as exc:
                failure = str(exc)
        for entry in entries:
            frame = frames[entry['image_file']]
            item = dict(entry, **board_evidence_fields(frame), image_sha256=frame['sha256'],
                        board_variant_id=capture['board_variant_id'], coordinate_frame='board',
                        x_board_mm=None, y_board_mm=None, z_board_mm=None,
                        camera_axis_z_mm=None, camera_range_mm=None, reprojection_error_px=None,
                        solved_view_count=len(usable), source_view_count=len(entries),
                        point_max_error_px=diag['max_error_px'] if diag else None,
                        passes_point_gate=(diag['passes_point_gate'] if diag and
                                           capture['quality']['max_point_error_px'] is not None else None),
                        point_gate_active=capture['quality']['max_point_error_px'] is not None,
                        passes_pose_gate=frame.get('passes_pose_gate'),
                        depth_status=failure, validated_metric=False, suitable_for_ppfd=False,
                        all_contributing_views_identity_confirmed=all(
                            r['view_identity_review'] == 'confirmed' for r in entries),
                        reconstruction_scope='independent_physical_point_all_available_solved_views')
            if xyz is not None:
                item.update(x_board_mm=float(xyz[0]), y_board_mm=float(xyz[1]), z_board_mm=float(xyz[2]),
                            depth_status='camera_pose_unavailable')
                if frame['pose_status'] == 'solved':
                    camera_mm, distance_mm = camera_coordinates(xyz, frame)
                    fitted = project(xyz[None], frame, capture['intrinsics'])[0]
                    item.update(camera_axis_z_mm=float(camera_mm[2]), camera_range_mm=distance_mm,
                                reprojection_error_px=float(np.linalg.norm(fitted - [entry['x_px'], entry['y_px']])),
                                depth_status='experimental_unvalidated')
            result.append(item)
    return result


VISIBILITY = ('full', 'partial', 'unknown')
IDENTITY_REVIEW = ('confirmed', 'unconfirmed')


def view_status(leaf: dict, image_key: str) -> tuple[str, str]:
    """Missing per-view metadata is conservative: unknown and unconfirmed."""
    entry = (leaf.get('view_status') or {}).get(image_key) or {}
    visibility, review = entry.get('visibility', 'unknown'), entry.get('identity_review', 'unconfirmed')
    if visibility not in VISIBILITY or review not in IDENTITY_REVIEW:
        raise StudyError('Unknown leaflet visibility or identity review value')
    return visibility, review


def leaflet_bbox_rows(backup: dict, capture: dict, mapping: dict) -> list[dict]:
    """One row per saved leaflet bbox; ineligible boxes stay listed with reasons."""
    frames = {basename(f['image_file']).lower(): f for f in capture['frames']}
    size = (capture['intrinsics']['image_width_px'], capture['intrinsics']['image_height_px'])
    rows = []
    for key, leaf in current_state(backup).get('leaflets', {}).items():
        boxes = leaf.get('bboxes') or {}
        if not isinstance(boxes, dict):
            raise StudyError('Malformed leaflet bbox collection')
        for image_key, box in sorted(boxes.items()):
            if image_key not in frames:
                raise StudyError('Leaflet bbox belongs to an unknown photo')
            if not isinstance(box, dict):
                raise StudyError('Malformed leaflet bbox')
            frame = frames[image_key]
            visibility, review = view_status(leaf, image_key)
            coords = [box.get(k) for k in ('x_min_px', 'y_min_px', 'x_max_px', 'y_max_px')]
            reasons = []
            if (any(type(v) not in (int, float) or not np.isfinite(v) for v in coords) or
                    coords[0] < 0 or coords[1] < 0 or coords[2] - coords[0] < 1 or coords[3] - coords[1] < 1):
                reasons.append('invalid_bbox_geometry')
            elif (box.get('image_width_px'), box.get('image_height_px')) != size:
                # Preview-resolution coordinates are never reinterpreted as original pixels.
                reasons.append('bbox_image_size_differs_from_calibrated_original')
            elif coords[2] > size[0] or coords[3] > size[1]:
                reasons.append('invalid_bbox_geometry')
            if frame.get('pose_status') != 'solved' or frame.get('rvec_rad') is None or frame.get('tvec_mm') is None:
                reasons.append('camera_pose_unavailable')
            if visibility != 'full':
                reasons.append('visibility_' + visibility + '_not_a_full_silhouette')
            if review != 'confirmed':
                reasons.append('identity_link_unconfirmed')
            rows.append(dict(plant_id=backup.get('plant_id'), session_id=backup.get('session_id'), variant_id='__all__',
                             organ_id=leaf['organ_id'], leaflet_id=leaf['leaflet_id'], editor_identity=key,
                             viewer_leaflet_id=mapping.get(key), image_file=frame['image_file'],
                             image_sha256=frame['sha256'], annotation_id=box.get('annotation_id'),
                             x_min_px=coords[0], y_min_px=coords[1], x_max_px=coords[2], y_max_px=coords[3],
                             image_width_px=box.get('image_width_px'), image_height_px=box.get('image_height_px'),
                             coordinate_convention='original_pixel_edges', visibility=visibility, identity_review=review,
                             pose_status=frame.get('pose_status'), passes_pose_gate=frame.get('passes_pose_gate'),
                             **board_evidence_fields(frame),
                             region_source=box.get('region_source', 'manual_bbox'),
                             proposal_id=box.get('proposal_id'),
                             has_accepted_region=image_key in (leaf.get('regions') or {}),
                             region_model=((leaf.get('regions') or {}).get(image_key) or {}).get('model', {}).get('model_name'),
                             used_for_extent=not reasons, exclusion_reason=';'.join(reasons) or None,
                             observation_kind='image_observation', validated_metric=False))
    return rows


def leaflet_extents(bbox_rows: list[dict], depths: list[dict], capture: dict, leaf_status: list[dict]) -> list[dict]:
    """Per-leaflet bbox-derived and point-derived extents, kept separate."""
    frames = {f['image_file']: f for f in capture['frames']}
    quality_pass = [bool(f.get('passes_pose_gate')) for f in capture['frames']]
    result = []
    for status in leaf_status:
        key = status['editor_identity']
        boxes = [r for r in bbox_rows if r['editor_identity'] == key]
        eligible = [r for r in boxes if r['used_for_extent']]
        views = [(frames[r['image_file']], r) for r in eligible]
        extent = feasible_extent(views, capture['intrinsics'])
        solved = {}
        for row in depths:
            if row['viewer_leaflet_id'] == status['viewer_leaflet_id'] and row['z_board_mm'] is not None:
                solved[row['point_id']] = (row['x_board_mm'], row['y_board_mm'], row['z_board_mm'])
        points = np.asarray(list(solved.values()), dtype=float).reshape(-1, 3)
        relations = {pid: point_relation(np.asarray(xyz), extent) for pid, xyz in solved.items()}
        outside = sorted(pid for pid, rel in relations.items() if rel['inside_feasible_region'] is False)
        observed_points = {r['point_id'] for r in depths if r['viewer_leaflet_id'] == status['viewer_leaflet_id']}
        geometry = ('partial_surface_available' if status.get('surface_available') else
                    'point_depth_available' if solved else
                    'bbox_only' if boxes and not observed_points else
                    'image_points_without_depth' if observed_points else 'no_saved_observation')
        result.append(dict(
            editor_identity=key, viewer_leaflet_id=status['viewer_leaflet_id'],
            organ_id=key.split('|')[0], leaflet_id=key.split('|', 1)[1],
            geometry_status=geometry, surface_available=bool(status.get('surface_available')),
            bbox_view_count=len(boxes), bbox_views=[r['image_file'] for r in boxes],
            eligible_bbox_view_count=len(eligible), eligible_bbox_views=[r['image_file'] for r in eligible],
            excluded_bbox_views={r['image_file']: r['exclusion_reason'] for r in boxes if not r['used_for_extent']},
            cuboid_status=extent['status'], cuboid_detail=extent['detail'],
            cuboid_available=extent['status'] == ExtentStatus.OK,
            cuboid_min_mm=extent['cuboid_min_mm'], cuboid_max_mm=extent['cuboid_max_mm'],
            cuboid_size_mm=extent['cuboid_size_mm'], cuboid_volume_mm3=extent['cuboid_volume_mm3'],
            cuboid_local_frame=('translation only: local = board - cuboid_min_mm; board axes; mm'
                                if extent['status'] == ExtentStatus.OK else None),
            cuboid_origin_board_mm=extent['cuboid_min_mm'],
            feasible_volume_mm3=extent['feasible_volume_mm3'],
            feasible_fraction_of_cuboid=extent['feasible_fraction_of_cuboid'],
            inscribed_radius_mm=extent['inscribed_radius_mm'],
            feasible_vertices_mm=extent['feasible_vertices_mm'], feasible_edges=extent['feasible_edges'],
            max_projected_outside_bbox_px=extent['max_projected_outside_bbox_px'],
            halfspaces=extent['halfspaces'],
            eligible_views_pass_pose_gate=[bool(frames[r['image_file']].get('passes_pose_gate')) for r in eligible],
            any_accepted_pose=any(quality_pass),
            point_extent_status=('unavailable' if not solved else 'single_point_no_extent' if len(solved) == 1
                                 else 'sparse_points_only_not_leaf_boundary'),
            point_extent_min_mm=points.min(axis=0).tolist() if len(solved) >= 2 else None,
            point_extent_max_mm=points.max(axis=0).tolist() if len(solved) >= 2 else None,
            triangulated_point_ids=sorted(solved), image_point_ids=sorted(observed_points),
            points_outside_feasible_region=outside,
            point_relations=relations,
            validated_metric=False, suitable_for_ppfd=False, used_as_surface=False, used_as_occluder=False,
            estimate_kind='bbox_viewing_region_intersection'))
    return result


EXTENT_COLUMNS = ('editor_identity', 'viewer_leaflet_id', 'organ_id', 'leaflet_id', 'geometry_status',
                  'surface_available', 'bbox_view_count', 'eligible_bbox_view_count', 'cuboid_status',
                  'cuboid_available', 'cuboid_detail', 'cuboid_min_x_mm', 'cuboid_min_y_mm', 'cuboid_min_z_mm',
                  'cuboid_max_x_mm', 'cuboid_max_y_mm', 'cuboid_max_z_mm', 'cuboid_volume_mm3',
                  'feasible_volume_mm3', 'feasible_fraction_of_cuboid', 'max_projected_outside_bbox_px',
                  'point_extent_status', 'point_extent_min_x_mm', 'point_extent_min_y_mm', 'point_extent_min_z_mm',
                  'point_extent_max_x_mm', 'point_extent_max_y_mm', 'point_extent_max_z_mm',
                  'triangulated_point_count', 'points_outside_feasible_region', 'any_accepted_pose',
                  'surface_kinds', 'attached_surface_status', 'attached_surface_faces',
                  'validated_metric', 'suitable_for_ppfd')


def extent_table(extents: list[dict]) -> list[dict]:
    def axis(values, i):
        return None if values is None else values[i]
    rows = []
    for item in extents:
        row = {k: item.get(k) for k in EXTENT_COLUMNS if k in item}
        for prefix, source in (('cuboid_min', 'cuboid_min_mm'), ('cuboid_max', 'cuboid_max_mm'),
                               ('point_extent_min', 'point_extent_min_mm'), ('point_extent_max', 'point_extent_max_mm')):
            for i, name in enumerate('xyz'):
                row[f'{prefix}_{name}_mm'] = axis(item[source], i)
        row['triangulated_point_count'] = len(item['triangulated_point_ids'])
        row['points_outside_feasible_region'] = ';'.join(item['points_outside_feasible_region']) or 'none'
        row['surface_kinds'] = ';'.join(item.get('surface_kinds') or []) or 'none'
        rows.append({k: row.get(k) for k in EXTENT_COLUMNS})
    return rows


def relate_points(depths: list[dict], bbox_rows: list[dict], extents: list[dict]) -> None:
    """Add cuboid relations and 2D bbox containment; never move a point."""
    by_leaf = {e['viewer_leaflet_id']: e for e in extents}
    boxes = {(r['viewer_leaflet_id'], r['image_file']): r for r in bbox_rows}
    for row in depths:
        extent = by_leaf.get(row['viewer_leaflet_id'], {})
        relation = (extent.get('point_relations') or {}).get(row['point_id']) or point_relation(None, {})
        box = boxes.get((row['viewer_leaflet_id'], row['image_file']))
        row.update(estimate_kind='triangulated_estimate' if row['z_board_mm'] is not None else 'image_observation_only',
                   cuboid_status=extent.get('cuboid_status', ExtentStatus.NONE), **relation,
                   observation_inside_leaflet_bbox=(None if box is None else
                       bool(box['x_min_px'] <= row['x_px'] + 0.5 <= box['x_max_px'] and
                            box['y_min_px'] <= row['y_px'] + 0.5 <= box['y_max_px'])))



# Partial surfaces are produced downstream (tools.sam_supported_leaf_patches / visible_leaf_patch)
# from a previous result's point depths. Attaching one here links it to this run's leaflets and
# states plainly whether it still matches the current correspondences.
SURFACE_CURRENT = 'current_for_this_point_snapshot'
SURFACE_STALE = 'stale_from_earlier_point_snapshot'


def attach_surfaces(patches: Path | None, depth_sha256: str, depths: list[dict]) -> dict:
    """Link an existing partial-surface run to this rebuild without recomputing geometry."""
    base = dict(status='not_attached', patch_dir=None, surfaces=[], unavailable=[], model=None,
                expected_depth_sha256=depth_sha256, patch_depth_sha256=None,
                note='Interpolated between existing triangulated landmarks and clipped to unreviewed masks. '
                     'Not measured curvature, not a complete leaf, never a PPFD occluder.')
    if patches is None:
        return base
    patches = patches.resolve()
    report = read_json(patches / 'report.json')
    if not isinstance(report.get('records'), list) or not report.get('depth_sha256'):
        raise StudyError('Expected a partial-surface run report with records and depth_sha256')
    viewer = {}
    for row in depths:
        viewer.setdefault((row['organ_id'], row['leaflet_id']), row['viewer_leaflet_id'])
    fresh = report['depth_sha256'] == depth_sha256
    surfaces, unavailable = [], []
    for index, record in enumerate(report['records']):
        identity = (record.get('organ_id'), record.get('leaflet_id'))
        entry = dict(organ_id=identity[0], leaflet_id=identity[1], viewer_leaflet_id=viewer.get(identity),
                     identity_known_in_this_run=identity in viewer)
        if record.get('status') != 'diagnostic_partial_patch':
            unavailable.append(dict(entry, reason=record.get('reason', 'unavailable')))
            continue
        folder = patches / (record.get('url', f'leaf_{index}/index.html')).split('/')[0]
        patch = read_json(folder / 'patch.json')
        surfaces.append(dict(entry, surface_index=len(surfaces), patch_dir=folder.name,
                             geometry_source=patch.get('geometry_source', 'landmark_interpolation_clipped_to_mask'),
                             vertices_mm=patch['vertices_mm'], faces=patch['faces'],
                             physical_landmarks=patch.get('physical_landmarks', []),
                             source_images=patch.get('source_images', []),
                             retained_faces=patch.get('retained_faces'), proposed_faces=patch.get('proposed_faces'),
                             interpolated_patch_area_mm2=patch.get('interpolated_patch_area_mm2'),
                             all_source_pose_gates_passed=patch.get('all_source_pose_gates_passed'),
                             mask_source='unreviewed_segmentation_draft', complete_leaf=False,
                             measured_curvature=False, measured_leaf_area=False,
                             geometry_kind='interpolated_surface', validated_metric=False, suitable_for_ppfd=False,
                             status=SURFACE_CURRENT if fresh else SURFACE_STALE))
    return dict(base, status=SURFACE_CURRENT if fresh else SURFACE_STALE, patch_dir=str(patches),
                patch_report_sha256=file_hash(patches / 'report.json'), patch_depth_sha256=report['depth_sha256'],
                model=report.get('model'), inference_run_elsewhere=bool(report.get('inference_run')),
                surfaces=surfaces, unavailable=unavailable,
                stale_note=None if fresh else ('The attached surfaces were interpolated from different point depths. '
                                               'They are shown as stale and must be rebuilt from the current snapshot.'))



# ---------------------------------------------------------------------------------------------
# Hybrid branch: measured ChArUco poses stay the baseline. Side cameras inferred from reconstructed
# landmarks (tools.side_pose_candidate) are attached separately, never merged into capture.json, and
# only recomputed into their own table. An inferred camera never becomes a board measurement.
# ---------------------------------------------------------------------------------------------
HYBRID_POSE_SOURCE = 'leaflet_landmark_pnp'
BASELINE_POSE_SOURCE = 'charuco_board_pnp'


def load_side_poses(side_poses: Path | None, depth_sha256: str, capture: dict) -> dict:
    """Attach side-pose candidates and decide freshness from the inputs they were solved against."""
    base = dict(status='not_attached', directory=None, candidates=[], included_images=[],
                note='Inferred side cameras are diagnostic: they inherit the reference landmarks uncertainty '
                     'and never replace or average a measured ChArUco pose.')
    if side_poses is None:
        return base
    side_poses = side_poses.resolve()
    files = sorted(side_poses.glob('*.json')) if side_poses.is_dir() else [side_poses]
    known = {f['image_file'] for f in capture['frames']}
    candidates = []
    for path in files:
        candidate = read_json(path)
        if candidate.get('schema_version') != 'side-pose-candidate-1':
            continue
        stale = []
        if candidate.get('reference_depth_csv_sha256') != depth_sha256:
            stale.append('reference_points_changed')
        if candidate.get('intrinsics_sha256') != capture.get('intrinsics_sha256'):
            stale.append('calibration_changed')
        if candidate.get('image_file') not in known:
            stale.append('image_not_in_this_capture')
        measured = next((f for f in capture['frames'] if f['image_file'] == candidate.get('image_file')), None)
        if measured is not None and measured.get('pose_status') == 'solved':
            stale.append('measured_charuco_pose_exists_and_is_never_replaced')
        candidate = dict(candidate, freshness='current' if not stale else 'stale', stale_reasons=stale,
                         source_file=str(path))
        candidate['used_in_hybrid'] = (not stale and candidate.get('calculation_status') == 'solved'
                                       and candidate.get('diagnostic_inclusion') in
                                       ('eligible', 'included_by_explicit_override'))
        candidates.append(candidate)
    return dict(base, status='attached' if candidates else 'no_candidates_found', directory=str(side_poses),
                candidates=candidates, included_images=[c['image_file'] for c in candidates if c['used_in_hybrid']])


def hybrid_capture(capture: dict, attached: dict) -> dict:
    """A separate capture whose inferred frames are explicitly labelled, for the hybrid table only."""
    doc = deepcopy(capture)
    doc['contains_inferred_poses'] = True
    doc['schema_version'] = 'diagnostic-preview-capture-1'
    for frame in doc['frames']:
        frame['pose_source'] = BASELINE_POSE_SOURCE if frame['pose_status'] == 'solved' else None
    for candidate in attached['candidates']:
        if not candidate['used_in_hybrid']:
            continue
        frame = next(f for f in doc['frames'] if f['image_file'] == candidate['image_file'])
        frame.update(pose_status='solved', rvec_rad=candidate['rvec_rad'], tvec_mm=candidate['tvec_mm'],
                     pose_source=HYBRID_POSE_SOURCE, passes_pose_gate=False,
                     pose_candidate_id=candidate['candidate_id'], pose_is_measured_board=False,
                     failure_reason='inferred from reconstructed landmarks; not a board measurement')
    return doc


def hybrid_depths(rows: list[dict], capture: dict, attached: dict, baseline: list[dict]) -> tuple[list[dict], dict]:
    """Recompute point depths with baseline + inferred cameras, keeping the baseline values alongside."""
    if not attached['included_images']:
        return [], dict(points_added=0, points_changed=0, max_shift_mm=None, images_used=[])
    doc = hybrid_capture(capture, attached)
    sources = {f['image_file']: f.get('pose_source') for f in doc['frames']}
    recomputed = point_depths(rows, doc)
    previous = {(r['organ_id'], r['leaflet_id'], r['point_id'], r['image_file']): r for r in baseline}
    added = changed = 0
    shifts = []
    for row in recomputed:
        before = previous.get((row['organ_id'], row['leaflet_id'], row['point_id'], row['image_file']), {})
        baseline_xyz = [before.get(f'{axis}_board_mm') for axis in 'xyz']
        hybrid_xyz = [row.get(f'{axis}_board_mm') for axis in 'xyz']
        shift = (None if None in baseline_xyz or None in hybrid_xyz else
                 float(np.linalg.norm(np.asarray(hybrid_xyz, float) - np.asarray(baseline_xyz, float))))
        if shift is not None:
            shifts.append(shift)
        row.update(pose_source_for_this_observation=sources.get(row['image_file']),
                   baseline_x_board_mm=baseline_xyz[0], baseline_y_board_mm=baseline_xyz[1],
                   baseline_z_board_mm=baseline_xyz[2], baseline_depth_status=before.get('depth_status'),
                   shift_from_baseline_mm=shift,
                   uses_inferred_camera=any(sources.get(r['image_file']) == HYBRID_POSE_SOURCE for r in rows
                                            if (r['organ_id'], r['leaflet_id'], r['point_id']) ==
                                            (row['organ_id'], row['leaflet_id'], row['point_id'])),
                   branch='hybrid', validated_metric=False, suitable_for_ppfd=False)
    seen_added, seen_changed = set(), set()
    for row in recomputed:
        key = (row['organ_id'], row['leaflet_id'], row['point_id'])
        if row.get('z_board_mm') is not None and row.get('baseline_z_board_mm') is None and key not in seen_added:
            seen_added.add(key); added += 1
        elif row.get('shift_from_baseline_mm') and row['shift_from_baseline_mm'] > 1e-9 and key not in seen_changed:
            seen_changed.add(key); changed += 1
    return recomputed, dict(points_added=added, points_changed=changed,
                            max_shift_mm=max(shifts) if shifts else None,
                            images_used=attached['included_images'])



# ---------------------------------------------------------------------------------------------
# JSON reference layer. A request manifest names the human growth record, the annotation snapshot
# and the capture/calibration registry; it is checked before anything is calculated. Referencing
# JSON supplies counts, identities and provenance only: it never creates depth or validates a pose.
# ---------------------------------------------------------------------------------------------

def reviewed_leaflets(backup: dict, bbox_rows: list[dict], rows: list[dict]) -> list[dict]:
    """Reviewed label identities with their parent compound leaf, for the growth-count review."""
    confirmed_keys = {r['editor_identity'] for r in bbox_rows if r.get('identity_review') == 'confirmed'}
    confirmed_keys |= {r['organ_id'] + '|' + r['leaflet_id'] for r in rows
                       if r.get('identity_review') == 'confirmed'}
    observed = Counter(r['organ_id'] + '|' + r['leaflet_id'] for r in rows)
    result = []
    for key, leaf in current_state(backup).get('leaflets', {}).items():
        result.append(dict(organ_id=leaf.get('organ_id'), leaflet_id=leaf.get('leaflet_id'), leaf_key=key,
                           identity_review='confirmed' if key in confirmed_keys else 'unconfirmed',
                           # Absent on backups written before compound-leaf membership existed.
                           membership_state=(leaf.get('membership_review') or {}).get('state'),
                           saved_observations=observed.get(key, 0)))
    return result


def imported_label_summary(backup: dict, capture: dict) -> dict:
    """Per-photo counts of imported 2D labels. Observations only: they add no depth or identity."""
    state = current_state(backup)
    detections = (state.get('membership') or {}).get('detections') or {}
    confirmed = sum(1 for leaf in (state.get('leaflets') or {}).values()
                    if any(s.get('identity_review') == 'confirmed' for s in (leaf.get('view_status') or {}).values()))
    photos = []
    for frame in capture['frames']:
        found = detections.get(basename(frame['image_file']).lower(), [])
        photos.append(dict(image_file=frame['image_file'], view_code=frame.get('view_code'),
                           pose_status=frame.get('pose_status'), passes_pose_gate=frame.get('passes_pose_gate'),
                           n_markers_detected=frame.get('n_markers_detected'),
                           by_class=dict(Counter(d.get('class_name') for d in found)),
                           adopted_leaflets=sum(1 for d in found if d.get('adopted_into')),
                           state='2d_labels_only' if found else 'no_imported_labels'))
    return dict(total=sum(sum(p['by_class'].values()) for p in photos),
                adopted_total=sum(p['adopted_leaflets'] for p in photos), confirmed_identities=confirmed,
                photos=photos, note='Imported labels are 2D observations; rectangles are not leaf outlines.')


def checked_request(request: Path | None, capture: dict) -> dict | None:
    """Resolve and check the references before anything is calculated or written."""
    if request is None:
        return None
    loaded = reference_tool.require_ready(reference_tool.load_request(request))
    wanted = (loaded['request'].get('plant_id'), loaded['request'].get('session_id'))
    if wanted != (capture.get('plant_id'), capture.get('session_id')):
        raise StudyError(f'Reconstruction request is for {wanted}, not this session '
                         f"({capture.get('plant_id')}, {capture.get('session_id')})")
    return loaded


def attach_references(loaded: dict | None, capture: dict, rows: list[dict], backup: dict,
                      bbox_rows: list[dict]) -> dict | None:
    """Review counts and routes against the already-checked references."""
    if loaded is None:
        return None
    return reference_tool.reference_review(loaded, capture, rows, reviewed_leaflets(backup, bbox_rows, rows),
                                           current_state(backup).get('membership'))


def rebuild(session: Path, backup_path: Path, context_path: Path, out: Path,
            patches: Path | None = None, side_poses: Path | None = None,
            request: Path | None = None) -> dict:
    session, backup_path, context_path = (p.resolve() for p in (session, backup_path, context_path))
    capture, seed, backup, context = checked_inputs(session, backup_path, context_path)
    # References are resolved and refused first: no calculation runs on missing, inconsistent or stale inputs.
    loaded_request = checked_request(request, capture)
    annotation, rows, report = convert(backup, seed, context, capture)
    depths = point_depths(rows, capture)
    bbox_rows = leaflet_bbox_rows(backup, capture, report['identity_mapping'])
    input_hashes = {str(p): file_hash(p) for p in (backup_path, context_path, session / 'seed.backup.json')}
    folder = output_dir(out)
    shutil.copyfile(backup_path, folder / 'editor.backup.source.json')
    shutil.copyfile(context_path, folder / 'annotations.context.source.json')
    write_json(folder / 'capture.json', capture)
    write_json(folder / 'annotations.edited.json', annotation)
    preview = build(folder / 'capture.json', folder / 'annotations.edited.json', folder / 'preview')
    exported = export_records(folder / 'capture.json', folder / 'preview/preview.json',
                              folder / 'preview/points.candidate.csv', folder / 'depth')
    surfaces = {g['leaflet_id'] for g in read_json(folder / 'preview/preview.json')['geometry']}
    for status in report['leaf_status']:
        status['surface_available'] = status['viewer_leaflet_id'] in surfaces
        status['computed_point_count'] = len({r['point_id'] for r in depths
            if r['viewer_leaflet_id'] == status['viewer_leaflet_id'] and r['z_board_mm'] is not None})
    extents = leaflet_extents(bbox_rows, depths, capture, report['leaf_status'])
    relate_points(depths, bbox_rows, extents)
    for status, extent in zip(report['leaf_status'], extents):
        status.update(geometry_status=extent['geometry_status'], bbox_view_count=extent['bbox_view_count'],
                      cuboid_status=extent['cuboid_status'], cuboid_available=extent['cuboid_available'])
    write_rows(folder / 'editor_point_depths.csv', depths, tuple(depths[0]) if depths else ('status',))
    depth_sha256 = file_hash(folder / 'editor_point_depths.csv')
    side = load_side_poses(side_poses, depth_sha256, capture)
    hybrid_rows, hybrid_summary = hybrid_depths(rows, capture, side, depths)
    if side['status'] != 'not_attached':
        write_json(folder / 'side_pose_candidates.json', dict(
            schema_version='attached-side-poses-1', baseline_pose_source=BASELINE_POSE_SOURCE,
            hybrid_pose_source=HYBRID_POSE_SOURCE, reference_depth_csv_sha256=depth_sha256, **side))
    if hybrid_rows:
        write_rows(folder / 'hybrid_point_depths.csv', hybrid_rows, tuple(hybrid_rows[0]))
        write_json(folder / 'hybrid_capture.json', hybrid_capture(capture, side))
    attached = attach_surfaces(patches, depth_sha256, depths)
    surface_by_leaflet = {s['viewer_leaflet_id']: s for s in attached['surfaces'] if s['viewer_leaflet_id']}
    for status, extent in zip(report['leaf_status'], extents):
        surface = surface_by_leaflet.get(extent['viewer_leaflet_id'])
        kinds = (['preview_boundary_patch'] if extent['surface_available'] else []) + \
                ([('sam_clipped_partial_patch' if attached['status'] == SURFACE_CURRENT else 'stale_partial_patch')] if surface else [])
        extent.update(attached_surface_status=surface['status'] if surface else 'none',
                      attached_surface_faces=surface['retained_faces'] if surface else None,
                      surface_kinds=kinds)
        status.update(attached_surface_status=extent['attached_surface_status'], surface_kinds=kinds)
    write_json(folder / 'leaflet_surfaces.json', attached)
    write_rows(folder / 'leaflet_bbox_observations.csv', bbox_rows, tuple(bbox_rows[0]) if bbox_rows else ('status',))
    write_rows(folder / 'leaflet_extents.csv', extent_table(extents), EXTENT_COLUMNS)
    calibration = dict(intrinsics_path=capture['intrinsics_path'], intrinsics_sha256=capture['intrinsics_sha256'],
                       board_variant_id=capture['board_variant_id'], board_spec_sha256=capture.get('board_spec_sha256'),
                       quality_policy=capture['quality'],
                       accepted_pose_count=sum(bool(f.get('passes_pose_gate')) for f in capture['frames']),
                       solved_raw_pose_count=sum(f.get('pose_status') == 'solved' for f in capture['frames']),
                       photo_sha256={f['image_file']: f['sha256'] for f in capture['frames']})
    write_json(folder / 'leaflet_extents.json', dict(
        schema_version='leaflet-bbox-extent-1', method=EXTENT_METHOD, calibration=calibration,
        leaflets=extents, validated_metric=False, suitable_for_ppfd=False, used_as_surface=False,
        notice=('Exploratory extents from unvalidated raw poses. A cuboid is a spatial reference only: it does not '
                'define identity, thickness, surface or occlusion. Points outside it are preserved and reported.')))
    # Saved regions are review overlays only; never send their vertices to triangulation.
    accepted_regions = []
    for key, leaf in current_state(backup).get('leaflets', {}).items():
        for image_key, region in (leaf.get('regions') or {}).items():
            box = next((r for r in bbox_rows if r['editor_identity'] == key and
                        basename(r['image_file']).lower() == image_key), None)
            if box is not None:
                accepted_regions.append(dict(editor_identity=key, organ_id=leaf['organ_id'],
                    leaflet_id=leaf['leaflet_id'], image_file=box['image_file'],
                    image_sha256=box['image_sha256'], **region))
    references = attach_references(loaded_request, capture, rows, backup, bbox_rows)
    if references is not None:
        write_json(folder / 'reference_review.json', references)
        table = reference_tool.observation_table(references['observations'])
        write_rows(folder / 'route_observations.csv', table, reference_tool.OBSERVATION_COLUMNS)
    write_json(folder / 'accepted_regions.json', dict(regions=accepted_regions,
               coordinate_convention='original OpenCV pixel centres', used_for_triangulation=False))
    imported = imported_label_summary(backup, capture)
    write_point_review(capture, depths, folder, leaflets=extents, bbox_rows=bbox_rows,
                       accepted_regions=accepted_regions, surfaces=attached,
                       hybrid=dict(summary=hybrid_summary, side=side, rows=hybrid_rows),
                       references=references, imported=imported)
    report.update(imported_labels={k: v for k, v in imported.items() if k != 'photos'},
                  imported_labels_by_photo=imported['photos'])
    report.update(schema_version='rebuilt-local-leaf-edit-session-2',
                  backup_path=str(backup_path), backup_sha256=file_hash(backup_path),
                  context_path=str(context_path), context_sha256=file_hash(context_path),
                  prepared_capture_sha256=file_hash(session / 'capture.json'),
                  seed_sha256=file_hash(session / 'seed.backup.json'), calibration=calibration,
                  editor_observations=len(rows), modeled_leaflets=preview['modeled_leaflets'],
                  leaflet_bbox_observations=len(bbox_rows),
                  leaflet_bbox_used_for_extent=sum(r['used_for_extent'] for r in bbox_rows),
                  cuboids_available=sum(e['cuboid_available'] for e in extents),
                  side_pose_status=side['status'],
                  side_pose_candidates=len(side['candidates']),
                  side_poses_used_in_hybrid=len(side['included_images']),
                  side_pose_by_status=dict(Counter(c.get('calculation_status') for c in side['candidates'])),
                  side_pose_by_inclusion=dict(Counter(c.get('diagnostic_inclusion') for c in side['candidates'])),
                  side_pose_stale=sum(c['freshness'] == 'stale' for c in side['candidates']),
                  hybrid=hybrid_summary,
                  hybrid_note=('Hybrid point depths are written separately and use cameras inferred from '
                               'reconstructed landmarks. Baseline ChArUco results are unchanged, additional points '
                               'are coverage rather than accuracy, and nothing here is metric or PPFD validated.'),
                  **(reference_tool.summary(references) if references is not None else
                     dict(reference_status='no_request_attached')),
                  attached_surface_status=attached['status'], attached_surface_count=len(attached['surfaces']),
                  attached_surface_dir=attached['patch_dir'],
                  attached_surface_note='Attached partial surfaces are interpolated between existing landmarks and clipped to unreviewed masks; they are not measured curvature and are never exported as PPFD occluders.',
                  ai_proposals_by_state=dict(Counter(p.get('state') for p in (current_state(backup).get('proposals') or {}).values())),
                  **draft_counts(current_state(backup)),
                  draft_note='Unassigned automatic masks and landmark candidates are review drafts: they are never labels, never triangulated, and only accepted point observations reach the reconstruction.',
                  ai_proposal_note='Unaccepted AI proposals are never used. Accepted regions are 2D outlines: their vertices are not physical correspondences and never enter triangulation.',
                  mesh_depth_counts=exported['counts'], validated_metric=False, suitable_for_ppfd=False,
                  deep_learning_run=False,
                  input_files_changed=input_hashes != {p: file_hash(Path(p)) for p in input_hashes},
                  point_depth_note='Independent point CSV includes incomplete-view/interior labels. Mesh-depth tables use each patch source view set and may differ when more sparse observations exist. Neither is validated.',
                  extent_note='Leaflet bbox cuboids use only full-visibility, identity-confirmed boxes on solved raw poses. Point-derived extents are listed separately; sparse points do not establish a leaflet boundary.')
    write_json(folder / 'rebuild_report.json', report)
    write_summary(folder, report)
    return report


def write_summary(folder: Path, report: dict) -> None:
    labels = {'pending_boundary_correspondence': '대응점·윤곽 추가 필요',
              'submitted_for_diagnostic_reconstruction': '수정한 점으로 시험 복원',
              'unchanged_editor_label_uses_latest_explicit_context': '최근 검토 라벨 반영'}
    def status_label(row: dict) -> str:
        count = row.get('computed_point_count', 0)
        if row.get('surface_available') is False:
            return (f'대응점 {count}개 깊이 계산 · 잎 면 없음 · 미검증' if count else '사진 라벨 유지 · 깊이 미산출')
        if row.get('context_kind') == 'two_d_only_candidates':
            return '사진 영역 라벨 · 개별 대응점 깊이는 사진·CSV에서 확인'
        return labels.get(row['mesh_status'], row['mesh_status'])
    listing = ''.join('<tr><td>' + escape(r['editor_identity']) + '</td><td>' + str(r['complete_solved_views']) +
                      '</td><td>' + escape(status_label(r)) + '</td><td>' +
                      escape(GEOMETRY_LABELS.get(r.get('geometry_status'), str(r.get('geometry_status')))) +
                      '</td><td>' + str(r.get('bbox_view_count', 0)) + '</td><td>' +
                      escape(CUBOID_LABELS.get(r.get('cuboid_status'), str(r.get('cuboid_status')))) + '</td><td>' +
                      escape(' · '.join(SURFACE_LABELS.get(k, k) for k in (r.get('surface_kinds') or [])) or '없음') + '</td></tr>'
                      for r in report['leaf_status'])
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>수정 라벨의 3D·깊이 결과</title>'
        '<style>body{font:17px sans-serif;max-width:1000px;margin:40px auto;padding:20px;background:#10231a;color:#e4f0e8}a{color:#91e1b2}td,th{padding:10px;border:1px solid #567}table{border-collapse:collapse}</style>'
        '<h1>수정 라벨의 3D·깊이 결과</h1><p>사람 검토 전 후보 · 깊이 미검증 · 전체 식물 복원 미완료</p>'
        '<p><a href="preview/index.html">3D 모델과 라벨링 사진 보기</a></p>'
        '<p><a href="labels/index.html">대응점 3D 위치와 라벨링 사진 보기 · 깊이가 없는 점 포함</a></p>'
        '<p><a href="editor_point_depths.csv">모든 편집 대응점의 깊이 CSV</a> · <a href="rebuild_report.json">처리 기록</a> · '
        '<a href="editor.backup.source.json">입력 작업 백업</a></p>'
        '<p><a href="leaflet_surfaces.json">연결된 부분 면(보간·미검증) JSON</a></p>'
        '<p><a href="leaflet_extents.csv">소엽별 가능 공간(직육면체) CSV</a> · <a href="leaflet_extents.json">계산 방법·보정 출처 JSON</a> · '
        '<a href="leaflet_bbox_observations.csv">소엽 박스 관찰 CSV</a></p>'
        '<p>주맥·일부 사진의 점도 사진과 CSV에 보존합니다. 대응 사진이 부족한 점은 깊이를 계산하지 않습니다. '
        '완전한 윤곽 사진으로 만든 면과 전체 대응점으로 따로 계산한 깊이는 사용 사진에 따라 다를 수 있습니다.</p>'
        '<p>부분 면은 기존 대응점 사이를 보간해 분할 윤곽 안쪽만 남긴 결과입니다. 측정된 곡률도, 완전한 잎도, PPFD 입력도 아닙니다. '
        '대응점을 고치면 오래된 면으로 표시하며 다시 계산해야 합니다.</p>'
        '<p>소엽 직육면체는 ‘전체 보임’과 ‘연결 확인’으로 표시한 박스 2장 이상에서만 계산합니다. 잎 표면·두께·그늘 계산에 쓰지 않으며, '
        '밖에 있는 점도 옮기거나 지우지 않고 불일치로 표시합니다.</p>'
        '<table><tr><th>편집 라벨</th><th>윤곽이 완성된 포즈 사진</th><th>처리 상태</th><th>형상 상태</th><th>소엽 박스 사진</th><th>직육면체</th><th>부분 면</th></tr>'
        + listing + '</table>', encoding='utf-8')


SURFACE_LABELS = {'preview_boundary_patch': '윤곽 점 보간 면', 'sam_clipped_partial_patch': '분할 윤곽으로 자른 보간 면',
                  'stale_partial_patch': '오래된 면 · 다시 계산 필요'}
GEOMETRY_LABELS = {'partial_surface_available': '부분 면 있음', 'point_depth_available': '점 깊이 있음',
                   'bbox_only': '박스만 있음', 'image_points_without_depth': '사진 점만 · 깊이 없음',
                   'no_saved_observation': '저장된 관찰 없음'}
CUBOID_LABELS = {ExtentStatus.OK: '계산됨 · 미검증', ExtentStatus.NONE: '사용 가능한 박스 없음',
                 ExtentStatus.SINGLE: '사진 1장 · 깊이 범위 없음', ExtentStatus.UNBOUNDED: '범위가 닫히지 않음',
                 ExtentStatus.INCOMPATIBLE: '박스끼리 맞지 않음', ExtentStatus.DEGENERATE: '부피 없음 · 표시 안 함',
                 ExtentStatus.FAILED: '계산 실패'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'backup', 'context', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--patches', type=Path, help='Existing partial-surface run directory (report.json + leaf_*/patch.json)')
    parser.add_argument('--side-poses', type=Path,
                        help='Directory or JSON of tools.side_pose_candidate results to attach as a hybrid branch')
    parser.add_argument('--request', type=Path,
                        help='Reconstruction request JSON (tools.reconstruction_request) naming the growth record, '
                             'annotation snapshot and capture/calibration references to consult')
    args = parser.parse_args()
    report = rebuild(args.session, args.backup, args.context, args.out, args.patches, args.side_poses, args.request)
    print(json.dumps({key: report[key] for key in ('editor_observations', 'modeled_leaflets', 'validated_metric')}))


if __name__ == '__main__':
    main()
