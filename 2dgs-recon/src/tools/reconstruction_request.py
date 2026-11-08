"""JSON references for the three-route 3D calculator.

This module is a reference layer only. It resolves and checks the JSON a reconstruction
run consults, reviews biological counts against reviewed labels, and groups saved
observations by physical point across the three review routes:

  global           every associated photo
  vertical         top 01/03 -> oblique 2X -> side 1X
  circumferential  adjacent side views 11 -> 12 -> ... -> 18 -> 11, and the oblique ring

The routes are navigation over one set of identities and observations, not three
independent measurements: the same saved observation reached through several routes is
counted once. Nothing here solves a pose, creates depth, validates a camera or edits a
human record. Growth counts are a review reference and never become geometry.

  python -m tools.reconstruction_request write --out build/requests/dwt_01.json \\
      --plant dwt_01 --session dwt_01_20260920 --growth <growth.json> \\
      --prepared-session <session dir> --annotation-snapshot <editor backup.json> \\
      --annotation-context <annotations.json> --output <result dir>
  python -m tools.reconstruction_request check --request build/requests/dwt_01.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json

REQUEST_SCHEMA = 'reconstruction-request-1'
#: References whose content decides geometry. A changed or unrecorded hash on these blocks a run.
GEOMETRY_REFERENCES = ('annotation_snapshot', 'annotation_context', 'capture', 'calibration')
#: The growth record is consulted for counts only; it can never change a coordinate.
REVIEW_REFERENCES = ('growth_record',)
ROUTES = ('global', 'vertical', 'circumferential')
RECONSTRUCTION_MODES = ('baseline_charuco', 'hybrid_diagnostic')

BLOCKING_FINDINGS = ('reference_file_missing', 'reference_not_readable_json', 'reference_hash_not_recorded',
                     'reference_hash_changed_since_snapshot', 'plant_identity_mismatch',
                     'session_identity_mismatch', 'unknown_request_schema', 'reference_missing_from_request')
FINDING_TEXT = {
    'reference_file_missing': 'the referenced file does not exist at the path resolved from this request',
    'reference_not_readable_json': 'the referenced file is not readable JSON',
    'reference_hash_not_recorded': 'no sha256 was recorded for this reference, so staleness cannot be decided',
    'reference_hash_changed_since_snapshot': 'the file changed since the request was written; refresh the '
                                             'snapshot explicitly or invalidate the request',
    'plant_identity_mismatch': 'the referenced file belongs to a different plant',
    'session_identity_mismatch': 'the referenced file belongs to a different session',
    'capture_date_mismatch': 'the growth record carries a different capture date',
    'unknown_request_schema': 'unsupported request schema version',
    'reference_missing_from_request': 'a required reference is absent from the request',
    'growth_leaflet_count_not_recorded': 'the growth record leaves this leaflet count unentered (null)',
    'more_reviewed_leaflets_than_recorded': 'more distinct reviewed leaflet identities than the recorded count',
    'fewer_reviewed_leaflets_than_recorded': 'fewer reviewed leaflet identities than the recorded count: '
                                             'incomplete coverage or a count to review',
    'parent_not_in_growth_record': 'labels name a parent compound leaf the growth record does not list',
    'compound_leaf_entry_count_differs': 'compound_leaves entries differ in number from compound_leaf_count',
    'contradictory_coordinates_for_one_observation': 'the same physical point has different coordinates saved for '
                                                     'one photograph; both are kept for review',
    'leaflet_membership_unresolved': 'open membership candidates name this compound leaf; they are not counted and '
                                     'no exact total can be claimed until they are decided in the editor',
    'leaflet_membership_not_reviewed': 'leaflets carry this parent in organ_id without a reviewed membership decision',
}
#: Editor membership candidates still awaiting a human decision.
OPEN_MEMBERSHIP_STATES = ('proposed', 'ambiguous', 'kept_unresolved')


# ----------------------------------------------------------------------------------------------
# The request manifest
# ----------------------------------------------------------------------------------------------

def _relative(path: Path, base: Path) -> str:
    """Store a path relative to the manifest when possible; an absolute path stays absolute."""
    path, base = Path(path).resolve(), Path(base).resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        try:  # One level up covers a manifest kept beside, not inside, its inputs.
            return '../' + path.relative_to(base.parent).as_posix()
        except ValueError:
            return str(path)


def _reference(path: Path | None, base: Path, role: str) -> dict | None:
    if path is None:
        return None
    path = Path(path)
    entry = dict(path=_relative(path, base), role=role, sha256=None)
    if path.is_file():
        entry['sha256'] = file_hash(path)
    elif path.is_dir():
        entry['kind'] = 'directory'
    return entry


def write_request(out: Path, plant_id: str, session_id: str, growth_record: Path, prepared_session: Path,
                  annotation_snapshot: Path, annotation_context: Path, output: Path,
                  capture: Path | None = None, calibration: Path | None = None, board_spec: Path | None = None,
                  observation_csv: list[Path] | None = None, quality_settings: dict | None = None,
                  reconstruction_mode: str = 'baseline_charuco', reference_geometry: Path | None = None,
                  capture_date: str | None = None) -> dict:
    """Write a reconstruction request whose paths resolve against its own location."""
    if reconstruction_mode not in RECONSTRUCTION_MODES:
        raise StudyError(f'Unknown reconstruction mode: {reconstruction_mode}')
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = out.parent
    prepared_session = Path(prepared_session)
    capture = Path(capture) if capture else prepared_session / 'capture.json'
    calibration = Path(calibration) if calibration else prepared_session / 'intrinsics.json'
    request = dict(
        schema_version=REQUEST_SCHEMA,
        created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        plant_id=plant_id, session_id=session_id, capture_date=capture_date,
        reconstruction_mode=reconstruction_mode,
        path_resolution='every path below is resolved relative to the directory holding this file',
        references=dict(
            growth_record=_reference(Path(growth_record), base, 'biological count and parent review only; no geometry'),
            prepared_session=_reference(prepared_session, base, 'prepared editor session directory'),
            annotation_snapshot=_reference(Path(annotation_snapshot), base, 'immutable editor backup snapshot'),
            annotation_context=_reference(Path(annotation_context), base, 'annotation/association context JSON'),
            capture=_reference(capture, base, 'image registry, poses and their sources'),
            calibration=_reference(calibration, base, 'camera intrinsics'),
            board_spec=_reference(Path(board_spec) if board_spec else None, base, 'board variant definition'),
            observation_csv=[_reference(Path(p), base, 'exported observation table')
                             for p in (observation_csv or [])]),
        quality_settings=quality_settings or dict(source='capture.quality'),
        reference_geometry=_reference(Path(reference_geometry) if reference_geometry else None, base,
                                      'frozen reference landmarks for the diagnostic hybrid branch'),
        output=dict(path=_relative(Path(output), base)),
        validated_metric=False, suitable_for_ppfd=False,
        notice=('References only. The routes share one set of observations; consulting this file creates no depth, '
                'validates no pose and never edits a human growth record.'))
    write_json(out, request)
    return request


def load_request(path: Path) -> dict:
    """Resolve and check a request. Findings are reported; nothing referenced is modified."""
    path = Path(path).resolve()
    if not path.is_file():
        raise StudyError(f'Reconstruction request not found: {path}')
    try:
        request = read_json(path)
    except (ValueError, UnicodeError) as error:
        raise StudyError(f'Reconstruction request is not readable JSON: {error}') from error
    base = path.parent
    findings, resolved, documents = [], {}, {}

    def note(code: str, reference: str, **extra) -> None:
        findings.append(dict(code=code, reference=reference, detail=FINDING_TEXT[code], **extra))

    if request.get('schema_version') != REQUEST_SCHEMA:
        note('unknown_request_schema', 'request', found=request.get('schema_version'), expected=REQUEST_SCHEMA)
        return dict(request=request, request_path=str(path), base=str(base), resolved=resolved,
                    documents=documents, findings=findings, status='blocked')
    references = request.get('references') or {}
    for name in GEOMETRY_REFERENCES + REVIEW_REFERENCES:
        if not references.get(name):
            note('reference_missing_from_request', name)
    for name, entry in references.items():
        entries = entry if isinstance(entry, list) else [entry]
        for item in [e for e in entries if e]:
            target = (base / item['path']).resolve()
            resolved.setdefault(name, []).append(str(target))
            if item.get('kind') == 'directory':
                if not target.is_dir():
                    note('reference_file_missing', name, path=str(target))
                continue
            if not target.is_file():
                note('reference_file_missing', name, path=str(target))
                continue
            if item.get('sha256') is None:
                note('reference_hash_not_recorded', name, path=str(target))
            elif file_hash(target) != item['sha256']:
                note('reference_hash_changed_since_snapshot', name, path=str(target),
                     recorded=item['sha256'], current=file_hash(target))
            if target.suffix.lower() == '.json':
                try:
                    documents[name] = read_json(target)
                except (ValueError, UnicodeError):
                    note('reference_not_readable_json', name, path=str(target))
    capture = documents.get('capture')
    if isinstance(capture, dict):
        if capture.get('plant_id') != request.get('plant_id'):
            note('plant_identity_mismatch', 'capture', found=capture.get('plant_id'), expected=request.get('plant_id'))
        if capture.get('session_id') != request.get('session_id'):
            note('session_identity_mismatch', 'capture', found=capture.get('session_id'),
                 expected=request.get('session_id'))
    growth = documents.get('growth_record')
    if isinstance(growth, dict):
        if growth.get('plant_id') != request.get('plant_id'):
            note('plant_identity_mismatch', 'growth_record', found=growth.get('plant_id'),
                 expected=request.get('plant_id'))
        if request.get('capture_date') and growth.get('capture_date') != request.get('capture_date'):
            note('capture_date_mismatch', 'growth_record', found=growth.get('capture_date'),
                 expected=request.get('capture_date'))
    blocking = [f for f in findings if f['code'] in BLOCKING_FINDINGS]
    return dict(request=request, request_path=str(path), base=str(base), resolved=resolved, documents=documents,
                findings=findings, blocking_findings=blocking, status='blocked' if blocking else 'ready')


def require_ready(loaded: dict) -> dict:
    """Refuse to calculate on missing, unreadable, inconsistent or stale references."""
    if loaded['status'] != 'ready':
        detail = '; '.join(f"{f['reference']}: {f['code']}" for f in loaded['blocking_findings'])
        raise StudyError(f'Reconstruction request is not usable: {detail}')
    return loaded


# ----------------------------------------------------------------------------------------------
# Growth record review: findings only
# ----------------------------------------------------------------------------------------------

def review_counts(growth: dict, leaflets: list[dict], membership: dict | None = None) -> dict:
    """Compare recorded leaflet counts with the distinct reviewed identities labelled for that parent.

    ``leaflets`` are reviewed label records with ``organ_id`` (the parent compound leaf),
    ``leaflet_id`` and, when present, ``identity_review``/``identity_confirmed`` and the editor's
    ``membership_state``. Unconfirmed identities are listed separately and never merged, deleted or
    invented to satisfy a count. ``membership`` is the editor's compound-leaf membership block: its
    open candidates are listed per parent and never counted.
    """
    recorded = growth.get('compound_leaves') or []
    by_parent: dict[str, dict[str, set]] = defaultdict(lambda: dict(confirmed=set(), unconfirmed=set(),
                                                                    member=set(), unreviewed=set()))
    for leaf in leaflets:
        parent, identity = leaf.get('organ_id'), leaf.get('leaflet_id')
        if identity in (None, '', 'unavailable'):
            continue
        state = leaf.get('identity_review', 'confirmed' if leaf.get('identity_confirmed') else 'unconfirmed')
        by_parent[parent]['confirmed' if state == 'confirmed' else 'unconfirmed'].add(identity)
        by_parent[parent]['member' if leaf.get('membership_state') == 'confirmed' else 'unreviewed'].add(identity)
    open_candidates = [c for c in ((membership or {}).get('candidates') or {}).values()
                       if c.get('state') in OPEN_MEMBERSHIP_STATES]
    unresolved: dict[str, set] = defaultdict(set)
    for candidate in open_candidates:
        for entry in candidate.get('candidate_parents') or []:
            unresolved[entry.get('parent_id')].add(candidate.get('leaf_key'))
    findings, entries = [], []
    listed = set()
    for entry in recorded:
        parent = entry.get('compound_leaf_id')
        listed.add(parent)
        seen = by_parent.get(parent, dict(confirmed=set(), unconfirmed=set(), member=set(), unreviewed=set()))
        count = entry.get('leaflet_count')
        reviewed = sorted(seen['confirmed'])
        record = dict(compound_leaf_id=parent, recorded_leaflet_count=count,
                      is_unfolding=entry.get('is_unfolding'),
                      reviewed_leaflet_ids=reviewed, reviewed_leaflet_count=len(reviewed),
                      unconfirmed_leaflet_ids=sorted(seen['unconfirmed']),
                      membership_confirmed_ids=sorted(seen['member']),
                      membership_unreviewed_ids=sorted(seen['unreviewed']),
                      unresolved_candidate_leaflets=sorted(str(k) for k in unresolved.get(parent, ())),
                      exact_total_claimed=not unresolved.get(parent), finding=None)
        if membership is not None:
            if unresolved.get(parent):
                findings.append(dict(code='leaflet_membership_unresolved', reference='labels', compound_leaf_id=parent,
                                     unresolved=len(unresolved[parent]),
                                     detail=FINDING_TEXT['leaflet_membership_unresolved']))
            if seen['unreviewed']:
                findings.append(dict(code='leaflet_membership_not_reviewed', reference='labels', compound_leaf_id=parent,
                                     unreviewed=len(seen['unreviewed']),
                                     detail=FINDING_TEXT['leaflet_membership_not_reviewed']))
        if count is None:
            record['finding'] = 'growth_leaflet_count_not_recorded'
        elif len(reviewed) > count:
            record['finding'] = 'more_reviewed_leaflets_than_recorded'
        elif len(reviewed) < count:
            record['finding'] = 'fewer_reviewed_leaflets_than_recorded'
        if record['finding']:
            findings.append(dict(code=record['finding'], reference='growth_record', compound_leaf_id=parent,
                                 recorded=count, reviewed=len(reviewed), detail=FINDING_TEXT[record['finding']]))
        entries.append(record)
    for parent in sorted(p for p in by_parent if p not in listed):
        findings.append(dict(code='parent_not_in_growth_record', reference='labels', compound_leaf_id=parent,
                             reviewed=len(by_parent[parent]['confirmed']),
                             detail=FINDING_TEXT['parent_not_in_growth_record']))
    if growth.get('compound_leaf_count') is not None and len(recorded) != growth['compound_leaf_count']:
        findings.append(dict(code='compound_leaf_entry_count_differs', reference='growth_record',
                             recorded=growth['compound_leaf_count'], entries=len(recorded),
                             detail=FINDING_TEXT['compound_leaf_entry_count_differs']))
    return dict(schema_version='growth-count-review-1', plant_id=growth.get('plant_id'),
                capture_date=growth.get('capture_date'), compound_leaves=entries, findings=findings,
                growth_shoot_count=growth.get('growth_shoot_count'), truss_count=growth.get('truss_count'),
                membership_schema=(membership or {}).get('schema_version'),
                open_membership_candidates=len(open_candidates),
                labels_changed=False, geometry_changed=False,
                notice=('Counts are a biological review reference. A mismatch is a finding to review against the '
                        'photographs; it never merges, deletes or creates a label and never supplies geometry.'))


# ----------------------------------------------------------------------------------------------
# Routes over one set of observations
# ----------------------------------------------------------------------------------------------

def route_membership(capture: dict) -> list[dict]:
    """Which review routes reach each photograph, from its capture label only."""
    frames = capture.get('frames') or []
    result = []
    for frame in frames:
        code = str(frame.get('view_code') or '')
        group = frame.get('observed_view_group')
        routes, position, ring = ['global'], None, None
        if group == 'top' or code in ('01', '03'):
            routes.append('vertical')
            position = 'vertical_start_top'
        elif len(code) == 2 and code[0] in '12':
            routes.append('vertical')
            position = 'vertical_oblique' if code[0] == '2' else 'vertical_side'
            routes.append('circumferential')
            ring = 'oblique_ring' if code[0] == '2' else 'side_ring'
        result.append(dict(image_file=frame.get('image_file'), view_code=code or None, view_group=group,
                           routes=routes, vertical_index=code[1] if position in ('vertical_oblique', 'vertical_side')
                           else None, vertical_position=position, ring=ring,
                           pose_status=frame.get('pose_status'), passes_pose_gate=frame.get('passes_pose_gate')))
    return result


def route_images(capture: dict, route: str = 'global') -> list[str]:
    """Photographs reachable on one route, in that route's navigation order."""
    if route not in ROUTES:
        raise StudyError(f'Unknown review route: {route}')
    membership = [m for m in route_membership(capture) if route in m['routes']]
    if route == 'vertical':
        order = {'vertical_start_top': 0, 'vertical_oblique': 1, 'vertical_side': 2}
        membership.sort(key=lambda m: (m['vertical_index'] or '', order[m['vertical_position']]))
    elif route == 'circumferential':
        # The side ring is the route; the oblique ring follows it where it exists.
        membership.sort(key=lambda m: (m['ring'] != 'side_ring', m['view_code'] or ''))
    return [m['image_file'] for m in membership]


def physical_point_key(row: dict) -> tuple:
    """Scoped key. Local names such as b1/i1 repeat across leaflets and never join on point_id alone."""
    return (row.get('plant_id'), row.get('session_id'), row.get('organ_id'), row.get('leaflet_id'),
            row.get('point_id'))


def collect_observations(rows: list[dict], capture: dict, route: str = 'global',
                         leaflet: str | None = None) -> dict:
    """Group saved observations by physical point for one route, counting each observation once."""
    allowed = set(route_images(capture, route))
    by_image = {f['image_file']: f for f in capture.get('frames') or []}
    by_route = {name: set(route_images(capture, name)) for name in ROUTES}
    reachable = {name: [r for r in ROUTES if name in by_route[r]] for name in allowed}
    points: dict[tuple, dict] = {}
    conflicts, collapsed = [], 0
    for row in rows:
        if row.get('image_file') not in allowed:
            continue
        if leaflet is not None and row.get('leaflet_id') != leaflet and row.get('viewer_leaflet_id') != leaflet:
            continue
        key = physical_point_key(row)
        point = points.setdefault(key, dict(
            plant_id=key[0], session_id=key[1], organ_id=key[2], leaflet_id=key[3], point_id=key[4],
            viewer_leaflet_id=row.get('viewer_leaflet_id'), point_kind=row.get('point_kind'),
            point_semantic_name=row.get('point_semantic_name'), observations=[], routes_reached=set()))
        frame = by_image.get(row['image_file'], {})
        coordinates = (float(row['x_px']), float(row['y_px']))
        existing = next((o for o in point['observations'] if o['image_file'] == row['image_file']), None)
        if existing is not None:
            # One saved observation reached through several routes stays one observation.
            collapsed += 1
            if (existing['x_px'], existing['y_px']) != coordinates:
                existing['contradictory_coordinates'].append(list(coordinates))
                existing.update(eligible=False, ineligible_reason='contradictory_coordinates_for_one_observation')
                conflicts.append(dict(code='contradictory_coordinates_for_one_observation', reference='observations',
                                      physical_point=list(key), image_file=row['image_file'],
                                      kept=[existing['x_px'], existing['y_px']], also_saved=list(coordinates),
                                      detail=FINDING_TEXT['contradictory_coordinates_for_one_observation']))
            continue
        posed = frame.get('pose_status') == 'solved'
        point['observations'].append(dict(
            image_file=row['image_file'], x_px=coordinates[0], y_px=coordinates[1],
            view_code=frame.get('view_code'), routes_reached=reachable.get(row['image_file'], ['global']),
            pose_status=frame.get('pose_status'), passes_pose_gate=frame.get('passes_pose_gate'),
            view_visibility=row.get('view_visibility'), identity_review=row.get('identity_review'),
            eligible=posed, ineligible_reason=None if posed else 'unposed_no_triangulation_ray',
            contradictory_coordinates=[]))
        point['routes_reached'].update(reachable.get(row['image_file'], ['global']))
    collected = []
    for key in sorted(points, key=lambda k: tuple(str(v) for v in k)):
        point = points[key]
        eligible = [o for o in point['observations'] if o['eligible']]
        point.update(physical_point_key='|'.join(str(v) for v in key),
                     routes_reached=sorted(point['routes_reached']),
                     unique_observation_count=len(point['observations']),
                     eligible_observation_count=len(eligible),
                     triangulation_status='two_or_more_eligible_views' if len(eligible) >= 2 else
                     'fewer_than_two_solved_views',
                     ineligible_reasons=sorted({o['ineligible_reason'] for o in point['observations']
                                                if o['ineligible_reason']}))
        collected.append(point)
    return dict(schema_version='route-observations-1', route=route, leaflet=leaflet,
                images=sorted(allowed), points=collected,
                unique_physical_points=len(collected),
                unique_observations=sum(p['unique_observation_count'] for p in collected),
                duplicate_observations_collapsed=collapsed,
                points_with_two_or_more_eligible_views=sum(
                    p['triangulation_status'] == 'two_or_more_eligible_views' for p in collected),
                conflicts=conflicts,
                notice=('Routes are navigation over one set of observations. Reaching an observation through more '
                        'routes does not increase its weight, and an unposed photograph supports identity review '
                        'but supplies no triangulation ray.'))


OBSERVATION_COLUMNS = ('physical_point_key', 'plant_id', 'session_id', 'organ_id', 'leaflet_id', 'viewer_leaflet_id',
                       'point_id', 'point_kind', 'point_semantic_name', 'image_file', 'view_code', 'x_px', 'y_px',
                       'routes_reached', 'pose_status', 'passes_pose_gate', 'view_visibility', 'identity_review',
                       'eligible', 'ineligible_reason', 'contradictory_coordinates', 'validated_metric')


def observation_table(collected: dict) -> list[dict]:
    """One row per unique physical-point observation, whichever routes reached it."""
    table = []
    for point in collected['points']:
        for observation in point['observations']:
            table.append(dict(
                physical_point_key=point['physical_point_key'], plant_id=point['plant_id'],
                session_id=point['session_id'], organ_id=point['organ_id'], leaflet_id=point['leaflet_id'],
                viewer_leaflet_id=point['viewer_leaflet_id'], point_id=point['point_id'],
                point_kind=point['point_kind'], point_semantic_name=point['point_semantic_name'],
                image_file=observation['image_file'], view_code=observation['view_code'],
                x_px=observation['x_px'], y_px=observation['y_px'],
                routes_reached=';'.join(observation['routes_reached']), pose_status=observation['pose_status'],
                passes_pose_gate=observation['passes_pose_gate'], view_visibility=observation['view_visibility'],
                identity_review=observation['identity_review'], eligible=observation['eligible'],
                ineligible_reason=observation['ineligible_reason'],
                contradictory_coordinates=';'.join(str(c) for c in observation['contradictory_coordinates']),
                validated_metric=False))
    return table


def reference_review(loaded: dict, capture: dict, rows: list[dict], leaflets: list[dict],
                     membership: dict | None = None) -> dict:
    """The whole reference layer for one run: provenance, count review and route collection."""
    growth = loaded['documents'].get('growth_record') or {}
    counts = review_counts(growth, leaflets, membership) if growth else dict(
        schema_version='growth-count-review-1', compound_leaves=[], findings=[],
        notice='No growth record was referenced; no biological count review was possible.')
    routes = {route: collect_observations(rows, capture, route) for route in ROUTES}
    request = loaded['request']
    return dict(schema_version='reconstruction-reference-review-1',
                request_path=loaded['request_path'], request_sha256=file_hash(Path(loaded['request_path'])),
                plant_id=request.get('plant_id'), session_id=request.get('session_id'),
                reconstruction_mode=request.get('reconstruction_mode'),
                status=loaded['status'], findings=loaded['findings'] + counts['findings'] +
                routes['global']['conflicts'],
                references={name: dict(paths=paths, sha256=[file_hash(Path(p)) if Path(p).is_file() else None
                                                            for p in paths])
                            for name, paths in loaded['resolved'].items()},
                growth_review=counts,
                routes={route: {k: v for k, v in data.items() if k != 'points'} for route, data in routes.items()},
                observations=routes['global'],
                validated_metric=False, suitable_for_ppfd=False,
                notice=('Reference layer only. Route selection changes navigation, not geometry: the same '
                        'observations and poses reconstruct identically on every route.'))


def summary(review: dict) -> dict:
    """Compact counts for the rebuild report."""
    return dict(reference_status=review['status'],
                reference_findings=[dict(code=f['code'], reference=f.get('reference')) for f in review['findings']],
                reference_finding_counts=dict(Counter(f['code'] for f in review['findings'])),
                growth_review={e['compound_leaf_id']: dict(recorded=e['recorded_leaflet_count'],
                                                           reviewed=e['reviewed_leaflet_count'],
                                                           unresolved=len(e.get('unresolved_candidate_leaflets', [])),
                                                           finding=e['finding'])
                               for e in review['growth_review'].get('compound_leaves', [])},
                route_summary={route: dict(images=len(data['images']),
                                           unique_physical_points=data['unique_physical_points'],
                                           unique_observations=data['unique_observations'],
                                           duplicate_observations_collapsed=data['duplicate_observations_collapsed'],
                                           points_with_two_or_more_eligible_views=data[
                                               'points_with_two_or_more_eligible_views'])
                               for route, data in review['routes'].items()},
                reference_note=('JSON references supply counts, identities and provenance only. They create no '
                                'depth, validate no pose, and never edit a human growth record.'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    write = sub.add_parser('write', help='Write a reconstruction request manifest')
    write.add_argument('--out', type=Path, required=True)
    for key in ('plant', 'session'):
        write.add_argument('--' + key, required=True)
    write.add_argument('--capture-date')
    for key in ('growth', 'prepared-session', 'annotation-snapshot', 'annotation-context', 'output'):
        write.add_argument('--' + key, type=Path, required=True)
    for key in ('capture', 'calibration', 'board-spec', 'reference-geometry'):
        write.add_argument('--' + key, type=Path)
    write.add_argument('--observation-csv', type=Path, nargs='*')
    write.add_argument('--mode', choices=RECONSTRUCTION_MODES, default='baseline_charuco')
    check = sub.add_parser('check', help='Resolve and check a request without calculating anything')
    check.add_argument('--request', type=Path, required=True)
    check.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'write':
        request = write_request(args.out, args.plant, args.session, args.growth, args.prepared_session,
                                args.annotation_snapshot, args.annotation_context, args.output,
                                args.capture, args.calibration, args.board_spec, args.observation_csv,
                                reconstruction_mode=args.mode, reference_geometry=args.reference_geometry,
                                capture_date=args.capture_date)
        print(json.dumps(dict(request=str(Path(args.out).resolve()), plant_id=request['plant_id'])))
    else:
        loaded = load_request(args.request)
        report = dict(status=loaded['status'], findings=loaded['findings'], resolved=loaded['resolved'])
        if args.out:
            write_json(output_dir(args.out) / 'request_check.json', report)
        print(json.dumps(dict(status=loaded['status'],
                              findings=[f['code'] for f in loaded['findings']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
