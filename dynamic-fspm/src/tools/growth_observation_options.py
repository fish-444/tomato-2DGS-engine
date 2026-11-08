"""Opt-in observation corrections; no fitted covariance or new growth equations."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from tools.leaf_light.common import StudyError
from tools.growth_observation_evidence import comparison_reason, variance_reason, state_trait_reason, AREA_FIELDS

TRAIT_UNITS = {'rachis_arc_length_mm':'mm','left_extent_mm':'mm','right_extent_mm':'mm',
               'supported_area_cm2':'cm2','attachment_xyz_mm':'mm','direction_unit_xyz':'rad'}


def _finite(value: Any) -> bool:
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _weight(options: dict, field: str) -> tuple[float | None,str | None]:
    covariance=options.get('reanchor_covariance')
    if covariance is None:
        value=options.get('reanchor_weight',1)
        if not _finite(value) or not 0<=value<=1: raise StudyError('reanchor_weight must be in [0,1]')
        return float(value),None
    if not isinstance(covariance,dict): raise StudyError('reanchor_covariance must be an object')
    if 'reanchor_weight' in options: raise StudyError('Choose manual reanchor_weight or covariance mode, not both')
    if covariance.get('mode')!='diagonal_variance': return None,'reanchor_covariance_mode_unavailable'
    if covariance.get('cross_covariance') is not None: return None,'reanchor_cross_covariance_unsupported'
    record=covariance.get('traits',{}).get(field) or {}
    if not isinstance(record,dict): raise StudyError('reanchor variance record must be an object')
    if not record.get('source_ref') or record.get('source_kind') not in ('measured','fitted','literature_informed','user_assumed'):
        return None,'reanchor_variance_evidence_missing'
    reason = variance_reason(record, data_kind=options.get('observation_data_kind', 'synthetic'))
    if reason:
        return None, reason
    if record.get('unit')!=TRAIT_UNITS[field]+'^2' or record.get('identifiable') is not True:
        return None,'reanchor_variance_not_identifiable_or_unit_mismatch'
    a,b=record.get('variance_state'),record.get('variance_measurement')
    if not _finite(a) or not _finite(b) or min(a,b)<0: return None,'reanchor_variance_invalid'
    if a+b==0: return None,'reanchor_zero_variance_denominator'
    return a/(a+b),None


def weighted_anchor(pred: dict | None, target: dict, options: dict) -> tuple[dict,list[dict]]:
    """Use the existing axis dilation/translation representation; unsupported rotations stay explicit."""
    manual=options.get('reanchor_covariance') is None
    weight,_=_weight(options,'rachis_arc_length_mm') if manual else (None,None)
    exact=manual and weight==1
    result=deepcopy(target if exact else (pred or target))
    blocked = {field: comparison_reason(pred or {}, target, field)
               for field in (*AREA_FIELDS, 'direction_unit_xyz')}
    audit=[]
    for field,unit in TRAIT_UNITS.items():
        omega,reason=_weight(options,field)
        x=(pred or {}).get(field); y=target.get(field)
        row=dict(quantity=field,unit=unit,omega=omega,displacement=None,applied_displacement=None,reason_code=reason)
        if blocked.get(field):
            row['reason_code'] = blocked[field]
            result[field] = deepcopy(x) if pred and not state_trait_reason(pred,field) else None
            result.setdefault('trait_update_unavailable', {})[field] = blocked[field]
        elif x is None or y is None:
            row['reason_code']=reason or 'reanchor_trait_unavailable'
            if not exact: result[field]=None
        elif field=='supported_area_cm2' and pred.get('area_basis_id')!=target.get('area_basis_id'):
            row['reason_code']='reanchor_area_basis_mismatch'
        elif np.shape(x)!=np.shape(y): row['reason_code']='reanchor_correspondence_unavailable'
        elif field.endswith('_extent_mm') and pred.get('extent_stations')!=target.get('extent_stations'):
            row['reason_code']='reanchor_correspondence_unavailable'
        elif field=='direction_unit_xyz':
            delta=math.acos(float(np.clip(np.array(x)@y,-1,1)))
            row['displacement']=delta
            if not reason and (omega in (0,1) or delta<=1e-12):
                if not exact: result[field]=deepcopy(y if omega==1 else x)
                row['applied_displacement']=omega*delta
            elif not reason: row['reason_code']='reanchor_direction_correction_unsupported'
        else:
            delta=np.array(y)-np.array(x)
            row['displacement']=delta.tolist()
            if not reason:
                correction=omega*delta
                row['applied_displacement']=correction.tolist()
                if not exact: result[field]=(np.array(x)+correction).tolist()
        audit.append(row)
    for field in AREA_FIELDS:
        if blocked.get(field):
            result[field] = deepcopy(pred.get(field)) if pred and not state_trait_reason(pred,field) else None
            result.setdefault('trait_update_unavailable', {})[field] = blocked[field]
    for field, reason in blocked.items():
        if reason:
            result['reanchor_status'] = 'weighted_observation_anchor'
            result.setdefault('retained_prediction_traits', []).append(field)
            result.get('trait_unavailable', {}).pop(field, None)
    if blocked.get('direction_unit_xyz'):
        if (pred or {}).get('observation_evidence', {}).get('direction'):
            result.setdefault('observation_evidence', {})['direction'] = deepcopy(pred['observation_evidence']['direction'])
        elif result.get('observation_evidence'):
            result['observation_evidence'].pop('direction', None)
    if blocked.get('supported_area_cm2'):
        result['area_basis_id'] = (pred or {}).get('area_basis_id')
        # Preserve the provenance of the retained state, not the rejected value.
        if (pred or {}).get('observation_evidence', {}).get('area'):
            result.setdefault('observation_evidence', {})['area'] = deepcopy(pred['observation_evidence']['area'])
        elif result.get('observation_evidence'):
            result['observation_evidence'].pop('area', None)
    if blocked.get('direction_unit_xyz'):
        result['reanchor_status'] = 'weighted_observation_anchor'
        for field in ('direction_unit_xyz', 'transverse_unit_xyz', 'azimuth_deg', 'elevation_deg'):
            result[field] = deepcopy((pred or {}).get(field)) if pred and not state_trait_reason(pred,'direction_unit_xyz') else None
        points = (pred or {}).get('rachis_points_mm') if pred and not state_trait_reason(pred,'direction_unit_xyz') else None
        length = (pred or {}).get('rachis_arc_length_mm')
        if points is not None and length and result.get('attachment_xyz_mm') is not None and result.get('rachis_arc_length_mm') is not None:
            points = np.asarray(points)
            result['rachis_points_mm'] = (np.asarray(result['attachment_xyz_mm']) +
                (points-points[0])*result['rachis_arc_length_mm']/length).tolist()
        else:
            result['rachis_points_mm'] = None
    if exact: return result,audit  # Preserve the delivered measured-shape path byte-for-byte.
    # Scalar correction uses the existing anchored dilation. A new rotation/shape
    # interpolation law is intentionally not inferred from two different shapes.
    points=(pred or {}).get('rachis_points_mm'); length=result.get('rachis_arc_length_mm')
    anchor=result.get('attachment_xyz_mm'); old_length=(pred or {}).get('rachis_arc_length_mm')
    if points is not None and length is not None and anchor is not None and old_length:
        points=np.asarray(points); result['rachis_points_mm']=(np.asarray(anchor)+(points-points[0])*length/old_length).tolist()
        result['direction_unit_xyz']=deepcopy(pred.get('direction_unit_xyz'))
        result['transverse_unit_xyz']=deepcopy(pred.get('transverse_unit_xyz'))
        direction=next(r for r in audit if r['quantity']=='direction_unit_xyz')
        if direction['applied_displacement'] not in (None,0):
            direction.update(applied_displacement=None,reason_code='reanchor_direction_correction_unsupported')
    else:
        result['rachis_points_mm']=None
    result['reanchor_status']='weighted_observation_anchor'
    return result,audit


def accumulate_corrections(prior: list[dict], current: list[dict], update_id: str) -> list[dict]:
    """One trajectory/organ/trait/station/unit ledger; no cross-unit scalar sums."""
    result=deepcopy(prior)
    for item in current:
        correction=item.get('applied_displacement')
        if correction is None: continue
        values=correction if isinstance(correction,list) else [correction]
        for station,value in enumerate(values):
            key={k:item[k] for k in ('trajectory_id','organ_id','quantity','unit')}
            key['station_index']=station if isinstance(correction,list) else None
            row=next((r for r in result if all(r.get(k)==v for k,v in key.items())),None)
            if row is None:
                row=dict(key,cumulative_absolute_displacement=0.,update_ids=[]); result.append(row)
            if update_id not in row['update_ids']:
                row['cumulative_absolute_displacement']+=abs(value); row['update_ids'].append(update_id)
    return result


def posture_offsets(model: dict | None, association: dict) -> tuple[dict,list[str]]:
    spec=(model or {}).get('posture_offset')
    if spec is None: return {},[]
    if not isinstance(spec,dict): raise StudyError('posture_offset must be an object')
    if spec.get('enabled') is False: return {},[]
    if spec.get('enabled') is not True: return {},['posture_enablement_undeclared']
    required=('reference_phase','source_kind','source_ref','observed_at','time_reference','offsets')
    if any(spec.get(k) is None for k in required): return {},['posture_evidence_missing']
    if spec['source_kind'] not in ('measured','fitted','literature_informed','user_assumed') or not spec['source_ref']:
        return {},['posture_evidence_missing']
    if spec['observed_at']!=association.get('observed_at') or spec['time_reference']!='declared_capture_phase':
        return {},['posture_time_reference_mismatch']
    phase=association.get('capture_phase')
    records=spec['offsets']
    if not isinstance(records,dict): raise StudyError('posture offsets must be phase-keyed records')
    reference=records.get(spec['reference_phase'])
    if not phase or reference is None or phase not in records: return {},['posture_phase_unavailable']
    allowed={'direction_tangent_1_rad','direction_tangent_2_rad'}
    for label,record in records.items():
        if not isinstance(record,dict): raise StudyError('posture phase offset must be an object')
        if record.get('unit')!='rad' or not record.get('source_ref'): return {},['posture_offset_evidence_missing']
        values=record.get('values',{})
        if not values or not set(values)<=allowed or any(not _finite(x) for x in values.values()):
            return {},['posture_angle_definition_unsupported']
        if label==spec['reference_phase'] and any(x!=0 for x in values.values()):
            return {},['posture_reference_offset_nonzero']
    if any(set(r['values'])!=set(reference['values']) for r in records.values()):
        return {},['posture_reference_coordinates_incomplete']
    values=records[phase]['values']
    if not set(values)<=set(model['fields']): return {},['posture_likelihood_field_missing']
    return values,[]


def adjust_posture_residual(residual: np.ndarray | None, fields: list[str], offsets: dict) -> np.ndarray | None:
    if residual is None or not offsets: return residual
    result=residual.copy()
    for i,field in enumerate(fields):
        if field in offsets and offsets[field]!=0:
            angle=result[i]+offsets[field]  # residual is state minus observation.
            result[i]=math.atan2(math.sin(angle),math.cos(angle))
    return result
