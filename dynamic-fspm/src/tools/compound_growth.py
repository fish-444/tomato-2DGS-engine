"""Observed CompoundLeaf states and explicitly bounded kinematic forecasts.

python -m tools.compound_growth --help
No biological coefficients, source records, or optical receivers are invented.
CSV contains observations; JSON contains schemas, settings and provenance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from tools.leaf_light.common import StudyError, digest, file_hash, number, output_dir, read_json, rows, write_json, write_rows
from tools.growth_observation_evidence import load_evidence, attach_evidence, audit_state, comparison_reason

OBS_COLUMNS = ('plant_id', 'compound_leaf_id', 'session_id', 'observed_at', 'parent_stem_id',
               'identity_status', 'review_source', 'geometry_status', 'development_state',
               'first_observed_at', 'supported_area_cm2', 'area_basis_id', 'estimated_total_area_cm2',
               'transverse_x', 'transverse_y', 'transverse_z')
AXIS_COLUMNS = ('plant_id', 'compound_leaf_id', 'session_id', 'station', 'x_mm', 'y_mm', 'z_mm')
EXTENT_COLUMNS = ('plant_id', 'compound_leaf_id', 'session_id', 'station', 'left_mm', 'right_mm')
IDENTITY = ('plant_id', 'compound_leaf_id')
SESSION = (*IDENTITY, 'session_id')
MATH_TOL = 1e-9  # Numerical precision only, not a biological acceptance threshold.


def available(value: Any) -> bool:
    return value not in (None, '', 'unavailable')


def aware_time(value: str) -> datetime:
    try:
        t = datetime.fromisoformat(value)
        if t.tzinfo is None or t.utcoffset() is None: raise ValueError()
        return t.astimezone(timezone.utc)
    except (ValueError, TypeError) as exc:
        raise StudyError('Timestamp must be ISO8601 with timezone') from exc


def vector(value: Any, label: str) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all(): raise StudyError('Invalid '+label)
    return v


def axis_traits(points_mm: list | np.ndarray | None) -> dict:
    empty = dict(attachment_xyz_mm=None, tip_xyz_mm=None, rachis_arc_length_mm=None,
                 direction_unit_xyz=None, azimuth_deg=None, elevation_deg=None)
    if points_mm is None or len(points_mm) == 0: return empty
    xyz = np.asarray(points_mm, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise StudyError('Axis needs at least two finite 3D points')
    lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if np.any(lengths <= MATH_TOL): raise StudyError('Collapsed axis segment')
    d = xyz[-1]-xyz[0]; norm = np.linalg.norm(d)
    result = dict(attachment_xyz_mm=xyz[0].tolist(), tip_xyz_mm=xyz[-1].tolist(),
                  rachis_arc_length_mm=float(lengths.sum()), direction_unit_xyz=None, azimuth_deg=None, elevation_deg=None)
    if norm <= MATH_TOL: return result
    d /= norm; xy = np.hypot(d[0], d[1])
    result.update(direction_unit_xyz=d.tolist(), azimuth_deg=None if xy <= MATH_TOL else float(np.degrees(np.arctan2(d[1],d[0]))),
                  elevation_deg=float(np.degrees(np.arctan2(d[2],xy))))
    return result


def _stations(records: list[dict], fields: tuple[str, ...], endpoints: bool) -> tuple[list, list]:
    records = sorted(records, key=lambda r: number(r['station'], 'station'))
    stations = [number(r['station'], 'station') for r in records]
    if len(set(stations)) != len(stations) or any(s < 0 or s > 1 for s in stations):
        raise StudyError('Duplicate or invalid stations')
    if stations and endpoints and (stations[0] != 0 or stations[-1] != 1):
        raise StudyError('Axis stations must include attachment 0 and tip 1')
    values = [[number(r[f], f) for f in fields] for r in records]
    return stations, values


def extract_compound_state(record: dict, axis_rows: list[dict], extent_rows: list[dict], context: dict) -> dict:
    for key in SESSION:
        if not available(record.get(key)): raise StudyError('Missing '+key)
    state = {k: record.get(k) if available(record.get(k)) else None for k in OBS_COLUMNS if not k.startswith('transverse_')}
    flags = []
    if state['observed_at'] not in (None, 'unavailable'): aware_time(state['observed_at'])
    else: state['observed_at'] = None; flags.append('capture_time_unavailable')
    if state['first_observed_at'] not in (None, 'unavailable'):
        first = aware_time(state['first_observed_at'])
        if state['observed_at'] and first > aware_time(state['observed_at']): raise StudyError('First observed after current observation')
    else: state['first_observed_at'] = None
    if state['identity_status'] != 'confirmed' or not state['review_source']:
        flags.append('temporal_identity_unconfirmed')
    if not state['parent_stem_id']: flags.append('parent_attachment_unconfirmed')
    stations, points = _stations(axis_rows, ('x_mm','y_mm','z_mm'), True)
    es, extents = _stations(extent_rows, ('left_mm','right_mm'), False)
    if any(x < 0 for pair in extents for x in pair): raise StudyError('Negative extent')
    state.update(axis_stations=stations, rachis_points_mm=points or None,
                 extent_stations=es, left_extent_mm=[p[0] for p in extents] or None,
                 right_extent_mm=[p[1] for p in extents] or None, **axis_traits(points))
    if state['direction_unit_xyz'] is None: flags.append('whole_rachis_direction_unavailable')
    if state['geometry_status'] != 'observed_axis': flags.append('whole_rachis_not_observed')
    transverse = [number(record.get('transverse_'+a), 'transverse_'+a, missing=True) for a in 'xyz']
    state['transverse_unit_xyz'] = None
    if any(t is not None for t in transverse):
        if any(t is None for t in transverse): raise StudyError('Incomplete transverse reference')
        y = vector(transverse, 'transverse reference')
        if not np.isclose(np.linalg.norm(y), 1, atol=MATH_TOL, rtol=0): raise StudyError('Transverse reference must be unit length')
        if state['direction_unit_xyz'] is None or abs(np.dot(y,state['direction_unit_xyz'])) > MATH_TOL:
            raise StudyError('Transverse reference must be orthogonal to observed axis direction')
        state['transverse_unit_xyz'] = y.tolist()
    if extents and state['transverse_unit_xyz'] is None:
        raise StudyError('Left/right extents require an observed transverse reference')
    for key in ('supported_area_cm2', 'estimated_total_area_cm2'):
        value = number(record.get(key), key, missing=True)
        if value is not None and value < 0: raise StudyError('Negative tissue area')
        state[key] = value
    if any(state[k] is not None for k in ('supported_area_cm2','estimated_total_area_cm2')) and not state['area_basis_id']:
        raise StudyError('Area requires explicit coverage/basis identity')
    for key in ('frame_id','coordinate_frame','frame_stability','frame_evidence'):
        state[key] = context.get(key)
    if context.get('coordinate_frame') != 'board_mm': raise StudyError('Expected existing board-frame millimetres')
    if context.get('frame_stability') != 'confirmed' or not available(context.get('frame_evidence')) or not available(context.get('frame_id')):
        flags.append('stable_frame_unconfirmed')
    state.update(qc_flags=flags, eligible_state=not flags, provenance=context.get('source_hashes', {}),
                 data_kind=context.get('data_kind','unverified'), reference_status=context.get('reference_status','unverified'),
                 uncertainty={'status':'uncalibrated'}, validated_growth=False, suitable_for_ppfd=False)
    if record.get('observation_evidence'):
        state['observation_evidence'] = deepcopy(record['observation_evidence'])
        phases = {c.get('capture_phase') for k,c in state['observation_evidence'].items()
                  if k in ('area','direction') and isinstance(c,dict) and c.get('capture_phase')}
        if len(phases)>1:
            raise StudyError('Conflicting observation evidence capture phases')
        if phases:
            state['capture_phase'] = next(iter(phases))
    if record.get('source_row'):
        state['source_row'] = deepcopy(record['source_row'])
    audit_state(state)
    return state


def load_history(path: Path) -> tuple[list[dict], dict]:
    manifest = read_json(path)
    if manifest.get('schema') != 'compound-observations-1': raise StudyError('Unknown observation schema')
    if manifest.get('data_kind') not in ('synthetic','observed'):
        raise StudyError('Explicit synthetic or observed data_kind required')
    hashes = {str(path.resolve()):file_hash(path)}; tables = {}
    for name, columns in (('observations',OBS_COLUMNS), ('axes',AXIS_COLUMNS), ('extents',EXTENT_COLUMNS)):
        entry = manifest['tables'][name]; p = (path.resolve().parent/entry['path']).resolve()
        h = file_hash(p)
        if h != entry['sha256']: raise StudyError('Input hash mismatch: '+name)
        hashes[str(p)] = h; tables[name] = rows(p, columns)
    evidence = load_evidence(manifest, path.resolve().parent, hashes)
    attach_evidence(tables['observations'], evidence)
    for i, row in enumerate(tables['observations'], 2):
        row['source_row'] = dict(table_sha256=manifest['tables']['observations']['sha256'], csv_line=i)
    context = {**manifest, 'source_hashes':hashes}
    axes, extents = defaultdict(list), defaultdict(list)
    for r in tables['axes']: axes[tuple(r[k] for k in SESSION)].append(r)
    for r in tables['extents']: extents[tuple(r[k] for k in SESSION)].append(r)
    keys = [tuple(r[k] for k in SESSION) for r in tables['observations']]
    if len(keys) != len(set(keys)): raise StudyError('Duplicate observation session key')
    if (set(axes)|set(extents))-set(keys): raise StudyError('Orphan axis or extent rows')
    states = [extract_compound_state(r, axes[k], extents[k], context) for r,k in zip(tables['observations'],keys)]
    # Equal timestamps in different sessions are ambiguous for a physical compound.
    times = set()
    for s in states:
        if s['observed_at'] is None: continue
        k = (*[s[n] for n in IDENTITY], aware_time(s['observed_at']))
        if k in times: raise StudyError('Ambiguous same-time compound observations')
        times.add(k)
    _unchanged(hashes)
    return states, context


def _unchanged(hashes: dict) -> None:
    for p,h in hashes.items():
        if file_hash(Path(p)) != h: raise StudyError('Source changed during operation: '+p)


def validate_history(history: list[dict]) -> list[dict]:
    if not history: raise StudyError('No observations')
    if len({tuple(s[k] for k in IDENTITY) for s in history}) != 1: raise StudyError('Mixed compound identity')
    if any(not s.get('eligible_state') for s in history): raise StudyError('History contains unavailable or unconfirmed states')
    if len({(s['frame_id'], s['coordinate_frame'], s['parent_stem_id']) for s in history}) != 1:
        raise StudyError('Frame or parent changed; resolve observation provenance first')
    ordered = sorted(history, key=lambda s: aware_time(s['observed_at']))
    if len({aware_time(s['observed_at']) for s in ordered}) != len(ordered): raise StudyError('Duplicate observation timestamp')
    for s in ordered:
        if s.get('trait_unavailable', {}).get('direction_unit_xyz'):
            continue  # Scalar observations remain usable; no oriented path is inferred.
        derived = axis_traits(s['rachis_points_mm'])
        if not np.isclose(derived['rachis_arc_length_mm'], s['rachis_arc_length_mm']): raise StudyError('Inconsistent derived axis length')
    return ordered


def _basis(state: dict) -> np.ndarray | None:
    if state['transverse_unit_xyz'] is None: return None
    x, y = np.array(state['direction_unit_xyz']), np.array(state['transverse_unit_xyz'])
    return np.column_stack((x, y, np.cross(x,y)))


def _direction_rotation(a: np.ndarray, b: np.ndarray) -> Rotation:
    cross = np.cross(a,b); sine = np.linalg.norm(cross); cosine = float(np.clip(np.dot(a,b),-1,1))
    if sine <= MATH_TOL:
        if cosine < 0: raise StudyError('Antipodal directions need a transverse orientation reference')
        return Rotation.identity()
    return Rotation.from_rotvec(cross/sine*np.arctan2(sine,cosine))


def forecast_persistence(history: list[dict], horizon_days: float) -> dict:
    if number(horizon_days,'horizon_days') <= 0: raise StudyError('Positive horizon required')
    last = validate_history(history)[-1]
    result = deepcopy(last)
    result.update(geometry_status='forecast_axis', parent_motion='held_constant', model='persistence')
    return result


def forecast_recent_change(history: list[dict], horizon_days: float) -> dict:
    if number(horizon_days,'horizon_days') <= 0: raise StudyError('Positive horizon required')
    ordered = validate_history(history)
    if len(ordered) < 2: raise StudyError('Recent change requires two distinct timestamps')
    a,b = ordered[-2:]
    direction_reason = comparison_reason(a, b, 'direction_unit_xyz')
    if direction_reason:
        result = deepcopy(b)
        result.update(geometry_status='forecast_axis', model='recent_change', parent_motion='unavailable')
        for field in ('rachis_points_mm', 'direction_unit_xyz', 'transverse_unit_xyz', 'azimuth_deg', 'elevation_deg'):
            result[field] = None
        result.setdefault('trait_unavailable', {})['direction_unit_xyz'] = direction_reason
        result['qc_flags'].append(direction_reason)
        # Reuse the existing scalar recent-change equations without fabricating
        # an oriented path. Missing direction does not turn size into persistence.
        dt = (aware_time(b['observed_at'])-aware_time(a['observed_at'])).total_seconds()/86400
        for field in ('rachis_arc_length_mm','left_extent_mm','right_extent_mm',
                      'supported_area_cm2','estimated_total_area_cm2'):
            reason = comparison_reason(a,b,field) if field.endswith('area_cm2') else None
            x,y = a.get(field),b.get(field)
            if x is None or y is None or reason or np.shape(x)!=np.shape(y) or (
                    field.endswith('_extent_mm') and a['extent_stations']!=b['extent_stations']):
                result[field] = None
                result.setdefault('trait_unavailable', {})[field] = reason or 'recent_change_correspondence_unavailable'
                continue
            value = np.asarray(y)+horizon_days/dt*(np.asarray(y)-np.asarray(x))
            if np.any(value<0): raise StudyError('Negative predicted size; outside model domain')
            result[field] = value.tolist()
        result['attachment_xyz_mm'] = None
        return result
    if a['axis_stations'] != b['axis_stations']: raise StudyError('Incompatible normalized axis samples')
    dt = (aware_time(b['observed_at'])-aware_time(a['observed_at'])).total_seconds()/86400
    ratio = horizon_days/dt
    pa,pb = np.array(a['rachis_points_mm']),np.array(b['rachis_points_mm'])
    aa,ab = pa[0],pb[0]
    qa,qb = _basis(a),_basis(b)
    if (qa is None) != (qb is None): raise StudyError('Transverse reference availability changed')
    rotation = Rotation.from_matrix(qb@qa.T) if qa is not None else _direction_rotation(np.array(a['direction_unit_xyz']),np.array(b['direction_unit_xyz']))
    if np.isclose(rotation.magnitude(), np.pi, atol=MATH_TOL):
        raise StudyError('Half-turn temporal rotation is ambiguous')
    future_rotation = Rotation.from_rotvec(rotation.as_rotvec()*ratio)
    # Back-transport the previous shape into the latest orientation before local change.
    previous_local = rotation.apply(pa-aa)
    current_local = pb-ab
    future_local = current_local + ratio*(current_local-previous_local)
    length = b['rachis_arc_length_mm'] + ratio*(b['rachis_arc_length_mm']-a['rachis_arc_length_mm'])
    if length <= MATH_TOL or np.dot(future_local[-1],current_local[-1]) <= 0:
        raise StudyError('Extrapolated axis collapses or reverses; outside model domain')
    raw_length = axis_traits(future_local)['rachis_arc_length_mm']
    local = future_rotation.apply(future_local*(length/raw_length))
    anchor = ab+ratio*(ab-aa)
    result = deepcopy(b)
    result.update(rachis_points_mm=(local+anchor).tolist(), geometry_status='forecast_axis',
                  parent_motion='recent_attachment_velocity', model='recent_change', **axis_traits(local+anchor))
    if qb is not None:
        y = future_rotation.apply(np.array(b['transverse_unit_xyz']))
        x = np.array(result['direction_unit_xyz']); y -= np.dot(y,x)*x
        if np.linalg.norm(y) <= MATH_TOL: raise StudyError('Future transverse frame collapses')
        result['transverse_unit_xyz'] = (y/np.linalg.norm(y)).tolist()
    for field in ('left_extent_mm','right_extent_mm'):
        if a[field] is None or b[field] is None or a['extent_stations'] != b['extent_stations']:
            result[field] = None; result['qc_flags'].append('extent_rate_unavailable'); continue
        future = np.array(b[field])+ratio*(np.array(b[field])-np.array(a[field]))
        if np.any(future < 0): raise StudyError('Negative predicted extent; outside model domain')
        result[field] = future.tolist()
    for field in ('supported_area_cm2','estimated_total_area_cm2'):
        reason = comparison_reason(a, b, field)
        if a[field] is None or b[field] is None or reason or not a['area_basis_id']:
            result[field] = None; result['qc_flags'].append(reason or 'area_rate_or_coverage_unavailable'); continue
        value = b[field]+ratio*(b[field]-a[field])
        if value < 0: raise StudyError('Negative predicted tissue area; outside model domain')
        result[field] = value
    return result


def forecast(history: list[dict], horizons_days: list[float], environment_scenario: dict | None,
             model_parameters: dict) -> dict:
    ordered = validate_history(history); last = ordered[-1]
    maximum = number(model_parameters.get('max_horizon_days'), 'max_horizon_days')
    if maximum <= 0: raise StudyError('Positive explicit maximum horizon required')
    horizons = [number(h, 'horizon_days') for h in horizons_days]
    if not horizons or len(set(horizons)) != len(horizons) or any(h <= 0 or h > maximum for h in horizons):
        raise StudyError('Horizons must be unique, positive and within the configured maximum')
    model = model_parameters.get('model')
    if model not in ('persistence','recent_change','reviewed_growth'):
        raise StudyError('Unknown model')
    if not available(model_parameters.get('provenance')): raise StudyError('Model settings need provenance')
    predictions = []
    for h in sorted(horizons):
        envelope = dict(plant_id=last['plant_id'], compound_leaf_id=last['compound_leaf_id'],
                        origin_session_id=last['session_id'], forecast_origin=last['observed_at'],
                        predicted_at=(aware_time(last['observed_at'])+timedelta(days=h)).isoformat(),
                        horizon_days=h, model=model)
        try:
            if model == 'reviewed_growth': raise StudyError('Reviewed biological transition specification and fitted parameters unavailable')
            state = (forecast_persistence if model == 'persistence' else forecast_recent_change)(ordered,h)
            state['eligible_state'] = False  # A prediction must never become an observation.
            state['qc_flags'] = sorted(set(state['qc_flags']))
            envelope.update(status='diagnostic_forecast', state=state, reason=None)
        except StudyError as exc:
            envelope.update(status='unavailable', state=None, reason=str(exc))
        predictions.append(envelope)
    return dict(schema='compound-forecast-1', observed_state=last, predictions=predictions,
                model_parameters=model_parameters, history_digest=digest(ordered),
                environment_scenario=environment_scenario, environment_used=False,
                light_conditioned_forecast='unavailable', uncertainty={'status':'uncalibrated'},
                validated_growth=False, suitable_for_ppfd=False)


def evaluate_compound_forecasts(states: list[dict], plan: dict) -> dict:
    train, test = plan.get('train_plant_ids',[]),plan.get('test_plant_ids',[])
    if not train or not test or set(train)&set(test) or len(train)!=len(set(train)) or len(test)!=len(set(test)):
        raise StudyError('Explicit disjoint training/test plant groups required')
    if not available(plan.get('split_provenance')): raise StudyError('Frozen split provenance required')
    grouped = defaultdict(list)
    for s in states: grouped[tuple(s[k] for k in IDENTITY)].append(s)
    metrics = []; seen = set()
    for request in plan['forecasts']:
        key = tuple(request[k] for k in IDENTITY)
        origin,target = aware_time(request['forecast_origin']),aware_time(request['target_at'])
        if key[0] not in test or target <= origin: raise StudyError('Invalid test forecast origin/target')
        if (key,origin,target) in seen: raise StudyError('Duplicate evaluation request')
        seen.add((key,origin,target))
        candidates = grouped.get(key,[])
        timed = [s for s in candidates if s.get('observed_at')]
        history = [s for s in timed if aware_time(s['observed_at']) <= origin and s.get('eligible_state')]
        truth = [s for s in timed if aware_time(s['observed_at']) == target]
        for model in ('persistence','recent_change','reviewed_growth'):
            row = dict(plant_id=key[0],compound_leaf_id=key[1],forecast_origin=request['forecast_origin'],
                       target_at=request['target_at'],horizon_days=(target-origin).total_seconds()/86400,model=model,
                       status='unavailable',reason=None,tip_error_mm=None,direction_error_deg=None,
                       axis_error_mm=None,length_error_mm=None,extent_error_mm=None,area_error_cm2=None)
            try:
                if not history or max(aware_time(s['observed_at']) for s in history) != origin: raise StudyError('Eligible origin observation unavailable')
                if len(truth) != 1 or not truth[0].get('eligible_state'): raise StudyError('Independent eligible target unavailable or ambiguous')
                if truth[0].get('reference_status') not in ('independent_observation','synthetic_truth'):
                    raise StudyError('Target is not an independent observation or explicit synthetic truth')
                result=forecast(history,[row['horizon_days']],None,{**plan['model_parameters'],'model':model})['predictions'][0]
                if result['state'] is None: raise StudyError(result['reason'])
                pred,actual=result['state'],truth[0]
                if (pred['frame_id'],pred['parent_stem_id']) != (actual['frame_id'],actual['parent_stem_id']): raise StudyError('Target frame or parent mismatch')
                row.update(status='evaluated',tip_error_mm=float(np.linalg.norm(np.array(pred['tip_xyz_mm'])-actual['tip_xyz_mm'])),
                           direction_error_deg=float(np.degrees(np.arccos(np.clip(np.dot(pred['direction_unit_xyz'],actual['direction_unit_xyz']),-1,1)))),
                           length_error_mm=abs(pred['rachis_arc_length_mm']-actual['rachis_arc_length_mm']))
                if pred['axis_stations']==actual['axis_stations']:
                    row['axis_error_mm']=float(np.linalg.norm(np.array(pred['rachis_points_mm'])-actual['rachis_points_mm'],axis=1).mean())
                if pred['extent_stations']==actual['extent_stations'] and all(s[f] is not None for s in (pred,actual) for f in ('left_extent_mm','right_extent_mm')):
                    row['extent_error_mm']=float(np.abs(np.array([pred['left_extent_mm'],pred['right_extent_mm']])-np.array([actual['left_extent_mm'],actual['right_extent_mm']])).mean())
                if not comparison_reason(pred, actual, 'supported_area_cm2') and all(s['supported_area_cm2'] is not None for s in (pred,actual)):
                    row['area_error_cm2']=abs(pred['supported_area_cm2']-actual['supported_area_cm2'])
            except StudyError as exc: row['reason']=str(exc)
            metrics.append(row)
    summaries=[]
    for model,h in sorted({(r['model'],r['horizon_days']) for r in metrics}):
        rr=[r for r in metrics if r['model']==model and r['horizon_days']==h]
        good=[r for r in rr if r['status']=='evaluated']
        summary=dict(model=model,horizon_days=h,requested=len(rr),evaluated=len(good),coverage=len(good)/len(rr),
                     plants=len({r['plant_id'] for r in good}),compounds=len({tuple(r[k] for k in IDENTITY) for r in good}))
        for field in ('tip_error_mm','direction_error_deg','axis_error_mm','length_error_mm','extent_error_mm','area_error_cm2'):
            values=[r[field] for r in good if r[field] is not None]
            summary['mean_'+field]=float(np.mean(values)) if values else None
        summaries.append(summary)
    return dict(schema='compound-evaluation-1',metrics=metrics,summary=summaries,split=plan,
                uncertainty='uncalibrated; no independent confidence interval claimed',validated_growth=False)


def audit_legacy_structure(path: Path) -> dict:
    """Read existing proposals without renaming a petiole segment to a whole rachis."""
    structure = read_json(path); hashes={str(path.resolve()):file_hash(path)}
    capture_path=(path.resolve().parent/structure['inputs']['capture']).resolve(); capture=read_json(capture_path)
    h=file_hash(capture_path)
    if h!=structure['inputs']['capture_sha256']: raise StudyError('Legacy capture hash mismatch')
    hashes[str(capture_path.resolve())]=h
    observed_at=capture.get('captured_at')
    if available(observed_at): aware_time(observed_at)
    else: observed_at=None
    candidates=[]
    regions={r['region']:r for r in structure.get('compound_regions',[])}
    for p in structure.get('petioles',[]):
        region=regions.get(p['region'],{})
        reasons=['temporal_identity_unconfirmed','whole_rachis_unavailable','attachment_unconfirmed',
                 'extent_unavailable','tissue_area_unavailable','stable_frame_unconfirmed']
        if not observed_at: reasons.append('capture_time_unavailable')
        points=[p['near_end'],p['far_end']] if p['status']=='fitted' else None
        traits=axis_traits(points)
        candidates.append(dict(plant_id=structure['plant_id'],session_id=structure['session_id'],
                               region_id=p['region'],proposed_name=region.get('proposed_name'),observed_at=observed_at,
                               observed_petiole_points_mm=points,petiole_signed_elevation_deg=traits['elevation_deg'],
                               petiole_azimuth_deg=traits['azimuth_deg'],compound_leaf_id=None,eligible_state=False,
                               blockers=reasons,source_photos=p.get('views',[])))
    _unchanged(hashes)
    return dict(schema='compound-legacy-audit-1',candidates=candidates,source_hashes=hashes,
                capture_quality=capture.get('quality'),board_evidence=capture.get('board_variant_evidence'),
                status='partial_observation_only',validated_growth=False,suitable_for_ppfd=False)


def export_run(out: Path, states: list[dict], context: dict, result: dict | None = None) -> Path:
    _unchanged(context.get('source_hashes',{}))
    folder=output_dir(out)
    state_rows=[]; axis_rows=[]; extent_rows=[]
    for i,s in enumerate(states):
        key=dict(zip(SESSION,(s[k] for k in SESSION)))
        row={**key,'observed_at':s['observed_at'],'state_index':i,'eligible_state':s['eligible_state'],
             'azimuth_deg':s['azimuth_deg'],'elevation_deg':s['elevation_deg'],'rachis_arc_length_mm':s['rachis_arc_length_mm'],
             'supported_area_cm2':s['supported_area_cm2'],'estimated_total_area_cm2':s['estimated_total_area_cm2'],
             'qc_flags':'|'.join(s['qc_flags'])}
        state_rows.append(row)
        for station,point in zip(s['axis_stations'],s['rachis_points_mm'] or []):
            axis_rows.append({**key,'station':station,**dict(zip(('x_mm','y_mm','z_mm'),point))})
        for station,left,right in zip(s['extent_stations'],s['left_extent_mm'] or [],s['right_extent_mm'] or []):
            extent_rows.append({**key,'station':station,'left_mm':left,'right_mm':right})
    columns=(*SESSION,'observed_at','state_index','eligible_state','azimuth_deg','elevation_deg','rachis_arc_length_mm',
             'supported_area_cm2','estimated_total_area_cm2','qc_flags')
    write_rows(folder/'states.csv',state_rows,columns);write_rows(folder/'axes.csv',axis_rows,AXIS_COLUMNS)
    write_rows(folder/'extents.csv',extent_rows,EXTENT_COLUMNS)
    write_json(folder/'state_metadata.json',dict(schema='compound-states-1',states=states,context=context))
    if result is not None:
        write_json(folder/'result.json',result)
        if 'predictions' in result:
            flat=[]; curves=[]; widths=[]
            for p in result['predictions']:
                s=p['state'] or {}
                flat.append({**{k:p[k] for k in ('plant_id','compound_leaf_id','forecast_origin','predicted_at','horizon_days','model','status','reason')},
                             **{k:s.get(k) for k in ('azimuth_deg','elevation_deg','rachis_arc_length_mm','supported_area_cm2','estimated_total_area_cm2')}})
                for station,point in zip(s.get('axis_stations',[]),s.get('rachis_points_mm') or []):
                    curves.append(dict(plant_id=p['plant_id'],compound_leaf_id=p['compound_leaf_id'],horizon_days=p['horizon_days'],
                                       station=station,**dict(zip(('x_mm','y_mm','z_mm'),point))))
                for station,left,right in zip(s.get('extent_stations',[]),s.get('left_extent_mm') or [],s.get('right_extent_mm') or []):
                    widths.append(dict(plant_id=p['plant_id'],compound_leaf_id=p['compound_leaf_id'],horizon_days=p['horizon_days'],
                                       station=station,left_mm=left,right_mm=right))
            write_rows(folder/'forecasts.csv',flat,tuple(flat[0]) if flat else ())
            write_rows(folder/'forecast_axes.csv',curves,('plant_id','compound_leaf_id','horizon_days','station','x_mm','y_mm','z_mm'))
            write_rows(folder/'forecast_extents.csv',widths,('plant_id','compound_leaf_id','horizon_days','station','left_mm','right_mm'))
            template=Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
            (folder/'viewer.html').write_text(template.replace('__COMPOUND_JSON__',json.dumps(result,ensure_ascii=True,allow_nan=False).replace('<','\\u003c')),encoding='utf-8')
        if 'metrics' in result:
            write_rows(folder/'metrics.csv',result['metrics'],tuple(result['metrics'][0]) if result['metrics'] else ())
    write_json(folder/'manifest.json',dict(schema='compound-run-1',source_hashes=context.get('source_hashes',{}),
               code_hashes={str(p.resolve()):file_hash(p) for p in (Path(__file__),Path(__file__).with_name('growth_observation_evidence.py'))},
               output_hashes={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()},validated_growth=False,
               suitable_for_ppfd=False,model_gap='Reviewed biological growth law and calibrated prediction intervals unavailable'))
    return folder


def templates(out: Path) -> None:
    folder=output_dir(out)
    tables={}
    for name,columns in (('observations',OBS_COLUMNS),('axes',AXIS_COLUMNS),('extents',EXTENT_COLUMNS)):
        p=folder/(name+'.csv');write_rows(p,[],columns);tables[name]=dict(path=p.name,sha256=file_hash(p))
    write_json(folder/'observations.json',dict(schema='compound-observations-1',coordinate_frame='board_mm',frame_id=None,
               frame_stability='unverified',frame_evidence=None,data_kind='observed',reference_status='unverified',tables=tables))
    write_json(folder/'forecast_settings.json',dict(model='recent_change',max_horizon_days=None,provenance=None))


def main() -> None:
    # Extend the existing entry point without changing legacy forecast semantics.
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'probabilistic':
        from tools.probabilistic_compound_growth import main as probabilistic_main
        probabilistic_main(sys.argv[2:])
        return
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('probabilistic', help='Joint 30-day growth calculator; use probabilistic --help')
    for name in ('templates','audit-legacy','extract','forecast','evaluate','prepare-light'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
        if name in ('extract','forecast','evaluate'): p.add_argument('--input',type=Path,required=True)
        if name=='audit-legacy':p.add_argument('--structure',type=Path,required=True)
        if name=='forecast':
            p.add_argument('--plant',required=True);p.add_argument('--compound',required=True)
            p.add_argument('--origin',required=True);p.add_argument('--horizons',nargs='+',type=float,required=True)
            p.add_argument('--settings',type=Path,required=True)
        if name=='evaluate':p.add_argument('--plan',type=Path,required=True)
        if name=='prepare-light':
            p.add_argument('--run',type=Path,required=True);p.add_argument('--plant',required=True);p.add_argument('--session',required=True)
            p.add_argument('--study',type=Path,required=True);p.add_argument('--case');p.add_argument('--metadata',type=Path)
    a=parser.parse_args()
    if a.command=='templates':templates(a.out)
    elif a.command=='audit-legacy':
        result=audit_legacy_structure(a.structure);folder=output_dir(a.out);write_json(folder/'audit.json',result)
    elif a.command=='prepare-light':
        from tools.plant_light_bridge import connect
        connect(a.run,a.plant,a.session,a.study,a.out,case_id=a.case,metadata=a.metadata,execute=False)
    else:
        states,context=load_history(a.input);result=None
        if a.command=='forecast':
            origin=aware_time(a.origin)
            history=[s for s in states if (s['plant_id'],s['compound_leaf_id'])==(a.plant,a.compound) and s['observed_at'] and aware_time(s['observed_at'])<=origin]
            if not history or max(aware_time(s['observed_at']) for s in history)!=origin: raise StudyError('Explicit forecast origin unavailable')
            context['source_hashes'][str(a.settings.resolve())]=file_hash(a.settings)
            result=forecast(history,a.horizons,None,read_json(a.settings))
        elif a.command=='evaluate':
            context['source_hashes'][str(a.plan.resolve())]=file_hash(a.plan)
            result=evaluate_compound_forecasts(states,read_json(a.plan))
        export_run(a.out,states,context,result)
    print({'status':'written','out':str(a.out),'validated_growth':False})


if __name__=='__main__':main()
