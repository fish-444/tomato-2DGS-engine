"""P3 observation semantics and continuity, not an area estimator or validation claim.

Optional manifest metadata is versioned independently of the existing CSVs.
No conversion/comparability mapping is inferred from matching units or labels.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from tools.leaf_light.common import StudyError, file_hash

SCHEMA = 'p3-observation-evidence-1'
AREA_FIELDS = ('supported_area_cm2', 'estimated_total_area_cm2')
CONTINUITY = ('quantity', 'unit', 'scope', 'method', 'method_version',
              'calibration_hash', 'pose_hash', 'partition_hash', 'tissue_support_id',
              'visibility_basis', 'capture_phase', 'canopy_placement_id')
NON_VARIANCE = {'synthetic_mae', 'mask_confidence', 'calibration_rms',
                'generated_sample_spread', 'reconstruction_sample_spread'}


def load_evidence(manifest: dict, base: Path, hashes: dict) -> dict:
    """Verify declared sources. Missing scientific declarations are checked per trait."""
    evidence = manifest.get('observation_evidence')
    if evidence is None:
        return {}
    if not isinstance(evidence, dict) or evidence.get('schema') != SCHEMA:
        raise StudyError('Unknown observation evidence schema')
    if not isinstance(evidence.get('records', []), list):
        raise StudyError('Observation evidence records must be a list')
    if not isinstance(evidence.get('sources', {}), dict):
        raise StudyError('Observation evidence sources must be an object')
    for ref in evidence.get('sources', {}).values():
        if not isinstance(ref, dict) or not ref.get('path') or not ref.get('sha256'):
            raise StudyError('Malformed observation evidence source')
        path = (base / ref['path']).resolve()
        if file_hash(path) != ref['sha256']:
            raise StudyError('Observation evidence source hash mismatch')
        hashes[str(path)] = ref['sha256']
    return deepcopy(evidence)


def attach_evidence(records: list[dict], evidence: dict, *, calibration: bool = False) -> None:
    """Exact plant/organ/row joins; never distribute aggregate evidence among organs."""
    def key(row: dict, meta: bool = False) -> tuple:
        organ = 'organ_id' if calibration or meta else 'compound_leaf_id'
        return (row.get('plant_id'), row.get(organ),
                row.get('record_id' if calibration else 'session_id'))

    by_key = {key(r): r for r in records}
    seen, source_rows = set(), {}
    sources = evidence.get('sources', {})
    for entry in evidence.get('records', []):
        if not isinstance(entry, dict):
            raise StudyError('Malformed observation evidence record')
        k = key(entry, True)
        if k in seen or k not in by_key or not all(k):
            raise StudyError('Duplicate, orphan or wrong-plant observation evidence')
        seen.add(k)
        r = by_key[k]
        for trait in ('area', 'direction', 'anchor_area', 'anchor_direction'):
            claim = entry.get(trait)
            if claim is None:
                continue
            if not isinstance(claim, dict):
                raise StudyError('Observation trait evidence must be an object')
            ref = claim.get('source_ref')
            source = sources.get(ref) if isinstance(ref, str) else None
            claim = deepcopy(claim)
            claim['source_verified'] = source is not None
            if source and claim.get('source_row_id'):
                token = (source['sha256'], str(claim['source_row_id']), trait.removeprefix('anchor_'))
                # The same anchor may recur within a calibration series, but never
                # under another organ/plant or as two current observations.
                previous = source_rows.get(token)
                owner = k[:2]
                if previous and (previous[0] != owner or not trait.startswith('anchor_') or not previous[1]):
                    raise StudyError('Source measurement row reused across observations or plants')
                source_rows[token] = (owner, trait.startswith('anchor_'))
            entry = dict(entry, **{trait: claim})
        r['observation_evidence'] = deepcopy(entry)


def trait_reason(claim: dict | None, trait: str, data_kind: str, *, angle: bool = False) -> str | None:
    if claim is None:
        # Existing measured whole-axis geometry has a complete legacy contract.
        # Legacy area only identified a diagnostic basis, not real tissue support.
        return 'area_evidence_missing' if trait == 'area' and data_kind != 'synthetic' else None
    if not isinstance(claim, dict):
        raise StudyError('Observation trait evidence must be an object')
    quantity = 'one_sided_physical_tissue' if trait == 'area' else ('directed_whole_rachis_angle' if angle else 'directed_whole_rachis')
    if claim.get('quantity') != quantity:
        return trait + '_quantity_unsupported'
    if claim.get('scope') != 'compound_leaf':
        return trait + '_scope_unsupported'
    if claim.get('unit') != ('cm2' if trait == 'area' else 'rad' if angle else 'unit_vector'):
        return trait + '_unit_mismatch'
    if claim.get('identity_scope') != 'cross_date_same_organ' or not claim.get('identity_source_ref'):
        return 'temporal_identity_unconfirmed'
    if trait == 'area' and claim.get('tissue_evidence_status') not in (
            ('reviewed', 'synthetic_assumption') if data_kind == 'synthetic' else ('reviewed',)):
        return 'area_tissue_evidence_unavailable'
    required = (*CONTINUITY, 'source_ref', 'source_row_id', 'observed_at')
    if any(claim.get(k) in (None, '', 'unavailable') for k in required):
        return trait + '_provenance_incomplete'
    if data_kind != 'synthetic' and not claim.get('source_verified'):
        return trait + '_source_evidence_unavailable'
    return None


def state_trait_reason(state: dict, field: str) -> str | None:
    trait = 'area' if field in AREA_FIELDS else 'direction'
    return (state.get('trait_unavailable') or {}).get(field) or trait_reason(
        state.get('observation_evidence', {}).get(trait), trait, state.get('data_kind', 'unverified'))


def comparison_reason(previous: dict, current: dict, field: str) -> str | None:
    trait = 'area' if field in AREA_FIELDS else 'direction'
    # A newly generated block has no earlier measured method to compare. Its
    # first observation may anchor it, but cannot establish a measured increment.
    if previous.get('state_source') == 'predicted_future' and not previous.get('source_compound_leaf_id'):
        reason = state_trait_reason(current, field)
        if reason:
            return reason
        return ('reanchor_area_basis_mismatch' if trait == 'area' and
                previous.get('area_basis_id') != current.get('area_basis_id') else None)
    for s in (previous, current):
        reason = state_trait_reason(s, field)
        if reason:
            return reason
    a = previous.get('observation_evidence', {}).get(trait)
    b = current.get('observation_evidence', {}).get(trait)
    if a is None and b is None:
        return ('reanchor_area_basis_mismatch' if trait == 'area' and
                previous.get('area_basis_id') != current.get('area_basis_id') else None)
    if a is None or b is None:
        return trait + '_comparison_evidence_missing'
    if any(a.get(k) != b.get(k) for k in CONTINUITY):
        return trait + '_method_or_support_changed'
    if trait == 'area' and previous.get('area_basis_id') != current.get('area_basis_id'):
        return 'reanchor_area_basis_mismatch'
    return None


def audit_state(state: dict) -> None:
    """Retain source measurements, suppress only unsupported quantities."""
    evidence = state.get('observation_evidence', {})
    for trait in ('area', 'direction'):
        fields = AREA_FIELDS if trait == 'area' else ('direction_unit_xyz',)
        if not any(state.get(f) is not None for f in fields):
            continue
        reason = trait_reason(evidence.get(trait), trait, state.get('data_kind', 'unverified'))
        claim = evidence.get(trait)
        if not reason and claim and claim.get('observed_at') != state.get('observed_at'):
            reason = trait + '_capture_time_mismatch'
        if reason:
            for field in fields:
                state.setdefault('raw_observation_traits', {})[field] = deepcopy(state.get(field))
                state.setdefault('trait_unavailable', {})[field] = reason
                state[field] = None
            if trait == 'direction':
                # A normal axis cannot supply the positioned rachis path. Retain
                # independent scalar size observations and original raw geometry.
                for field in ('rachis_points_mm', 'transverse_unit_xyz', 'azimuth_deg', 'elevation_deg'):
                    state.setdefault('raw_observation_traits', {})[field] = deepcopy(state.get(field))
                    state[field] = None


def calibration_reasons(row: dict, target: dict, context: dict) -> list[str]:
    meta = row.get('observation_evidence', {})
    identity = meta.get('identity', {})
    identity_reasons = []
    if meta.get('measurement_role', row.get('measurement_role')) in ('repeatability', 'single_capture'):
        identity_reasons.append('repeatability_or_single_capture_is_not_growth')
    if row.get('identity_scope', identity.get('scope')) not in (None, 'cross_date_same_organ'):
        identity_reasons.append('temporal_identity_unconfirmed')
    if identity.get('change') in ('merged', 'split', 'canopy_placement'):
        identity_reasons.append('organ_lineage_mapping_unsupported')
    trait = ('area' if target.get('unit') == 'cm2' else
             'direction' if target.get('family') == 'azimuth_rotation' else None)
    if trait is None:
        return identity_reasons
    current, anchor = meta.get(trait), meta.get('anchor_' + trait)
    reasons = identity_reasons
    for claim in (current, anchor):
        reason = trait_reason(claim, trait, context['data_kind'], angle=trait == 'direction')
        if reason:
            reasons.append(reason)
    if current and current.get('capture_phase') != row.get('capture_phase'):
        reasons.append(trait + '_capture_phase_mismatch')
    if current or anchor:
        if not current or not anchor:
            reasons.append(trait + '_comparison_evidence_missing')
        else:
            if any(current.get(k) != anchor.get(k) for k in CONTINUITY):
                reasons.append(trait + '_method_or_support_changed')
            if current.get('observed_at') != row.get('observed_at') or anchor.get('observed_at') != row.get('origin_at'):
                reasons.append(trait + '_capture_time_mismatch')
            if current.get('source_ref') == anchor.get('source_ref') and current.get('source_row_id') == anchor.get('source_row_id'):
                reasons.append('same_capture_is_not_growth')
            if trait == 'direction' and current.get('angle_reference') != target.get('angle_reference'):
                reasons.append('angle_reference_mismatch')
    return list(dict.fromkeys(reasons))


def variance_reason(record: dict, *, data_kind: str = 'synthetic') -> str | None:
    """Reject substitute diagnostics; keep explicitly assumed legacy policy."""
    kind = record.get('evidence_kind')
    if kind is not None and kind not in ('trait_measurement_variance', 'user_assumed_variance'):
        return 'reanchor_variance_quantity_unsupported'
    if data_kind != 'synthetic' and record.get('source_kind') in ('measured', 'fitted'):
        if kind != 'trait_measurement_variance' or not record.get('repeat_measurement_source_ref'):
            return 'reanchor_variance_evidence_missing'
    return None
