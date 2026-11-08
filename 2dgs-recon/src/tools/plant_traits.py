"""S8: whole-plant traits from one pipeline run, plus the calibration / light summary over many plants (2026-10-01).

    python -X utf8 -m tools.plant_traits --run build/pipeline_<ts>            # every plant of a run + summary
    python -X utf8 -m tools.plant_traits --summary build/plants_<ts>/traits     # summary again (new measurements)

The user's goal (2026-10-01): whole-plant leaf area and PPFD / DLI for 24 plants with as little human work as
possible; calibration is non-destructive. So nothing here waits for a person:

* leaf area uses EVERY generated leaflet mask that was not rejected (accepted by the person, accepted by the learned
  review model, or still unreviewed). Masks of one photo are merged into one union so duplicate boxes never count
  twice; each pixel gets the depth of the nearest mask over it (20th depth percentile of the S3 hull voxels that project
  inside that mask), area_mm2 = sum over pixels of (Z / f)^2. The share of that area a person or the review model
  accepted is reported, so a later review shows how much it moved the number;
* canopy area for light interception: the top-photo leaf area (same method) and the hull footprint on the board
  plane (cm2 of occupied x-y cells; an envelope, larger than the leaves);
* size: hull volume, hull height above its lowest voxel, stem diameter (tools.stem_thickness, from S3).

Summary (all plants together): if measurements/leaf_area.csv exists (plant_id, leaf_area_cm2: the person's
non-destructive measurement), estimated leaf area = k x image proxy with k fitted through the origin, leave-one-out
error when >= 3 plants are measured. If measurements/light.csv exists (plant_id or *, ppfd_umol_m2_s, photoperiod_h),
DLI = PPFD x h x 3600 / 1e6 and intercepted light = DLI x canopy area. Missing files are reported as the only human
tasks; calibration diagnostics do not establish metric validation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import math
from pathlib import Path

import numpy as np

from tools.leaf_light.common import StudyError, digest, file_hash, read_json, write_json
from tools.plant_structure import Cameras

ROOT = Path(__file__).resolve().parents[1]
FRONT_PERCENTILE = 20
MIN_HULL_POINTS = 5
MEASUREMENTS = 'measurements'
LEAF_AREA_FIELDS = ['plant_id', 'leaf_area_cm2', 'method', 'date']
LIGHT_FIELDS = ['plant_id', 'ppfd_umol_m2_s', 'photoperiod_h', 'date']


def view_group(code: str) -> str:
    return 'top' if code.startswith('0') else 'side' if code.startswith('2') else 'middle'


def batch_masks(batch_dir: Path) -> dict:
    """{image_file lower: [(crop_xyxy, bool mask, accepted)]} for every ready, not rejected leaflet mask."""
    from tools.leaflet_mask_batch import Batch
    b = Batch(Path(batch_dir))
    if b.manifest.get('organ_class', 'leaflet') != 'leaflet':
        return {}
    out = {}
    for i in b.manifest['ids']:
        it = b.item(i)
        if it['generation'] != 'ready' or it['review'] == 'rejected':
            continue
        m, meta = b.mask(it)
        out.setdefault(it['image_file'].lower(), []).append((meta['crop_px'], m, it['review'] in ('accepted', 'accepted_by_user')))
    return out


def photo_area(cams: Cameras, key: str, hull: np.ndarray, masks: list, *, area_policy: dict | None = None) -> dict:
    """Union leaf area (cm2) of one photo, all masks and accepted-only, with per-mask hull depth."""
    if area_policy is not None:
        from tools.leaf_area_geometry import IntegrationPolicy, photo_area_v2
        corrected = photo_area_v2(cams, key, hull, masks, IntegrationPolicy.parse(area_policy),
                                  FRONT_PERCENTILE, MIN_HULL_POINTS)
        # Historical fields retain their old meaning; the new estimator is explicit.
        if len(hull):
            legacy = photo_area(cams, key, hull, masks)
        else:
            legacy = dict(masks=len(masks), masks_used=0, masks_without_depth=len(masks),
                          union_px=0, leaf_area_cm2=None, accepted_leaf_area_cm2=None,
                          legacy_status='empty_hull_unavailable')
        return {**legacy, 'area_v2': corrected}
    W, H = cams.size
    px = np.round(cams.project(key, hull)).astype(int)
    depth = cams.depth(key, hull)
    ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H) & (depth > 0)
    f = float(cams.K[0, 0])
    z_all = np.full((H, W), np.inf, np.float32)
    z_acc = np.full((H, W), np.inf, np.float32)
    used = no_depth = 0
    for (x0, y0, x1, y1), m, accepted in masks:
        sel = ok & (px[:, 0] >= x0) & (px[:, 0] < x1) & (px[:, 1] >= y0) & (px[:, 1] < y1)
        idx = np.flatnonzero(sel)
        idx = idx[m[px[idx, 1] - y0, px[idx, 0] - x0]]
        if idx.size < MIN_HULL_POINTS:
            no_depth += 1
            continue
        z = np.float32(np.percentile(depth[idx], FRONT_PERCENTILE))
        for zi, take in ((z_all, True), (z_acc, accepted)):
            if take:
                view = zi[y0:y1, x0:x1]
                view[m] = np.minimum(view[m], z)
        used += 1

    def area(zi):
        v = zi[np.isfinite(zi)].astype(np.float64)
        return float(((v / f) ** 2).sum()) / 100.0, int(v.size)
    a_all, n_all = area(z_all)
    a_acc, _ = area(z_acc)
    return dict(masks=len(masks), masks_used=used, masks_without_depth=no_depth, union_px=n_all,
                leaf_area_cm2=round(a_all, 2), accepted_leaf_area_cm2=round(a_acc, 2))


def plant_traits(plant_id: str, session: Path, structure_dir: Path, batch_dir: Path | None, stem_thickness: Path | None = None,
                 *, area_policy: dict | None = None, view_grouping: dict | None = None,
                 capture_style: str | None = None) -> dict:
    capture = read_json(Path(session) / 'capture.json')
    cams = Cameras(capture)
    st = read_json(Path(structure_dir) / 'structure.json')
    hull = np.loadtxt(Path(structure_dir) / 'hull_voxels.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2), ndmin=2)
    vox = float(st['hull'].get('voxel_mm', 1.0))
    masks = batch_masks(batch_dir) if batch_dir and (Path(batch_dir) / 'manifest.json').exists() else {}
    rows = []
    for key in sorted(k for k in cams.frames if cams.posed(k)):
        if not masks.get(key):
            continue
        code = cams.frames[key].get('view_code') or key[-6:-4]
        rows.append(dict(photo=code, image_file=key, group=view_group(code), pose_status=cams.frames[key].get('pose_status'),
                         **photo_area(cams, key, hull, masks[key], area_policy=area_policy)))
        if area_policy is not None:
            from tools.leaf_area_geometry import view_geometry
            rows[-1]['view_geometry'] = view_geometry(cams, key, view_grouping)

    def mean(group, field='leaf_area_cm2'):
        v = [r[field] for r in rows if r['group'] == group]
        return (round(float(np.mean(v)), 2) if v else None), len(v)
    side, n_side = mean('side')
    side_acc, _ = mean('side', 'accepted_leaf_area_cm2')
    top, n_top = mean('top')
    # occupied x-y cells (floor, not round: voxel centres sit on .5 and round-half-even would merge neighbours)
    footprint = len(np.unique(np.floor(hull[:, :2] / vox).astype(np.int64), axis=0)) * vox * vox / 100.0 if len(hull) else None
    stem = None
    if stem_thickness and Path(stem_thickness).exists():
        ms = read_json(stem_thickness).get('main_stem', {})
        stem = ms.get('diameter_mm') if ms.get('status') == 'measured' else None
    n_masks = sum(len(v) for v in masks.values())
    n_acc = sum(a for v in masks.values() for *_, a in v)
    result = dict(schema='plant-traits-1', plant_id=plant_id,
                session_id=capture.get('session_id'), capture_date=record_date(capture) or None,
                leaf_area_proxy_cm2=side, leaf_area_proxy_photos=n_side, leaf_area_proxy_view='side ring (mean of posed photos)',
                accepted_share_pct=round(100 * side_acc / side, 1) if side and side_acc is not None else None,
                top_leaf_area_cm2=top, top_photos=n_top, hull_footprint_cm2=round(footprint, 1) if footprint else None,
                hull_cm3=st['hull']['occupancy_cm3'], hull_height_mm=round(float(np.ptp(hull[:, 2])) + vox, 1) if len(hull) else None,
                hull_views=len(st['views_used']), stem_diameter_mm=stem, masks=n_masks, masks_accepted=n_acc,
                photos=rows, validated_metric=False,
                note='image proxies; calibration estimates leaf area but does not validate dimensions')
    if area_policy is not None:
        from tools.leaf_area_geometry import METHOD, calibration_identity
        if not capture_style:
            raise StudyError('Explicit capture_style required for versioned calibration')
        from tools.leaf_area_geometry import view_geometry
        excluded = {r['photo'].lower(): r['reason'] for r in st.get('views_skipped', [])}
        by_image = {r['image_file']: r for r in rows}
        for key, frame in cams.frames.items():
            reason = ('pose_unavailable' if not cams.posed(key) else
                      'excluded_by_structure' if key in excluded else
                      'no_masks' if not masks.get(key) else None)
            if reason is None:
                continue
            if key not in by_image:
                code = frame.get('view_code') or key[-6:-4]
                row = dict(photo=code, image_file=key, group=view_group(code),
                           pose_status=frame.get('pose_status'),
                           view_geometry=view_geometry(cams, key, view_grouping))
                rows.append(row)
                by_image[key] = row
            row = by_image[key]
            row['area_v2'] = dict(status=reason, projected_area_cm2=None,
                                  accepted_projected_area_cm2=None)
            row['hull_exclusion_reason'] = excluded.get(key)
        def corrected_mean(group: str, field: str) -> tuple[float | None, int]:
            selected = [r for r in rows if (r['view_geometry']['group'] == group if view_grouping else r['group'] == group)]
            values = [r['area_v2'][field] for r in selected]
            # Do not silently drop unsupported selected photos.
            return (float(np.mean(values)) if values and all(v is not None for v in values) else None), len(values)
        proxy_group = 'ring' if view_grouping else 'side'
        new_side, count = corrected_mean(proxy_group, 'projected_area_cm2')
        new_accepted, _ = corrected_mean(proxy_group, 'accepted_projected_area_cm2')
        new_top, top_count = corrected_mean('top', 'projected_area_cm2')
        result.update(schema='plant-traits-2', area_method=METHOD,
                      integration_policy_hash=digest(area_policy), calibration_hash=calibration_identity(cams.K, cams.D, cams.size),
                      capture_style=capture_style, grouping_policy_hash=digest(view_grouping) if view_grouping else 'legacy_code_prefix',
                      legacy_leaf_area_proxy_cm2=side, legacy_top_leaf_area_cm2=top,
                      leaf_area_proxy_cm2=new_side, leaf_area_proxy_photos=count,
                      leaf_area_proxy_view='measured elevation ring' if view_grouping else 'legacy code-prefix side ring',
                      top_leaf_area_cm2=new_top, top_photos=top_count, legacy_top_photos=n_top,
                      accepted_share_pct=100*new_accepted/new_side if new_side and new_accepted is not None else None)
        result['aggregation_audit'] = aggregation_audit(rows, view_grouping is not None)
        selected = result['aggregation_audit']['footprint_measured' if view_grouping is not None else 'footprint_code']
        result['aggregation_status'] = selected['status']
        ratio_ok = (selected['status'] == 'complete_selected_projection' and
                    new_top is not None and new_side is not None and new_side > 0)
        result['top_to_ring_diagnostic'] = dict(
            value=new_top/new_side if ratio_ok else None, unit='1',
            status='projected_area_ratio_only' if ratio_ok else 'unavailable',
            reason=None if ratio_ok else 'incomplete_grouping_or_projection_or_nonpositive_ring',
            mean_inclination_deg=None, leaf_angle_distribution=None,
            botanical_face=None, occlusion_correction=None)
    result['area_provenance'] = trait_area_provenance(
        result, capture, st, session, structure_dir, batch_dir, masks, area_policy, view_grouping)
    return result


def aggregation_audit(rows: list[dict], measured_enabled: bool) -> dict:
    """A 2x2 diagnostic comparison with explicit membership and missingness.

    Legacy cells retain historical per-photo rounding. All four cells use the
    same frame inventory; excluded/unsupported frames remain unavailable.
    """
    result = {}
    for integrator in ('legacy', 'footprint'):
        for grouping in ('code', 'measured'):
            if grouping == 'measured' and not measured_enabled:
                result[f'{integrator}_{grouping}'] = dict(status='policy_unavailable', groups={})
                continue
            groups = {}
            unknown = []
            for row in rows:
                group = row['view_geometry']['group'] if grouping == 'measured' else row['group']
                group = 'ring' if group == 'side' else group
                if group == 'unavailable':
                    unknown.append(row['image_file'])
                    continue
                entry = groups.setdefault(group, dict(images=[], values_cm2=[], missing_images=[]))
                value = (row.get('leaf_area_cm2') if integrator == 'legacy' else
                         row['area_v2'].get('projected_area_cm2'))
                if integrator == 'legacy' and (row.get('masks_without_depth', 0) or
                        row['area_v2']['status'] in ('pose_unavailable', 'excluded_by_structure', 'no_masks')):
                    value = None
                entry['images'].append(row['image_file'])
                entry['values_cm2'].append(value)
                if value is None:
                    entry['missing_images'].append(row['image_file'])
            for entry in groups.values():
                entry['mean_cm2'] = (float(np.mean(entry['values_cm2'])) if not entry['missing_images'] else None)
            incomplete = bool(unknown or any(e['missing_images'] for e in groups.values()))
            result[f'{integrator}_{grouping}'] = dict(
                status='no_frames' if not rows else 'incomplete' if incomplete else 'complete_selected_projection',
                groups=groups, unknown_group_images=unknown, frame_count=len(rows),
                validated_metric=False, grouping=grouping, integrator=integrator)
    return result


def trait_area_provenance(trait: dict, capture: dict, structure: dict, session: Path,
                          structure_dir: Path, batch_dir: Path | None, masks: dict,
                          area_policy: dict | None, view_grouping: dict | None) -> dict:
    """Additive diagnostic lineage, never a physical area acceptance certificate."""
    from tools.leaf_area_geometry import LEGACY
    sources = {}
    paths = dict(capture=Path(session)/'capture.json', structure=Path(structure_dir)/'structure.json',
                 hull=Path(structure_dir)/'hull_voxels.csv')
    if batch_dir:
        paths['batch_manifest'] = Path(batch_dir)/'manifest.json'
    for name, path in paths.items():
        sources[name] = dict(path=str(path.resolve()), sha256=file_hash(path) if path.is_file() else None)
    frame_rows = []
    computed = {r['image_file']: r for r in trait['photos']}
    for frame in capture['frames']:
        key = frame['image_file'].lower()
        row = computed.get(key, {})
        frame_rows.append(dict(image_file=key, pose_status=frame.get('pose_status'),
            passes_pose_gate=frame.get('passes_pose_gate'), reprojection_error_px=frame.get('reprojection_error_px'),
            captured_at=frame.get('captured_at'), capture_phase=frame.get('capture_phase'),
            mask_input_digest=digest([dict(crop_px=list(crop), shape=list(mask.shape),
                packed_bits=np.packbits(mask).tolist(), accepted=bool(accepted))
                for crop, mask, accepted in masks.get(key, [])]),
            view_geometry=row.get('view_geometry'), area_v2=row.get('area_v2'),
            legacy_supported_union_px=row.get('union_px'),
            status=row.get('area_v2', {}).get('status', 'legacy_diagnostic' if row else 'not_computed')))
    intr = capture.get('intrinsics', {})
    matrix = intr.get('camera_matrix')
    focal_ratio = float(matrix[1][1]/matrix[0][0]) if matrix and matrix[0][0] else None
    return dict(schema='plant-traits-area-provenance-1', sources=sources,
        data_kind=capture.get('data_kind', 'undeclared'),
        area_method=trait.get('area_method', LEGACY), integration_policy=area_policy,
        view_grouping=view_grouping,
        measured_grouping_status=('enabled' if area_policy is not None else 'not_enabled_without_area_policy')
            if view_grouping is not None else 'policy_unavailable',
        capture_style=trait.get('capture_style'), captured_at=capture.get('captured_at'),
        capture_phase=capture.get('capture_phase'), pose_source='sources.capture:frames',
        calibration_source='sources.capture:intrinsics', physical_scale_status='not_validated_by_traits',
        structure_inputs=structure.get('inputs'), views_used=structure.get('views_used'),
        views_skipped=structure.get('views_skipped'), view_selection=structure.get('view_selection'),
        region_assignment_hash=structure.get('region_assignment_hash'),
        human_names_status=structure.get('human_names_status'),
        identity_status='whole_view_union_not_leaflet_correspondence',
        partition_status='hull_regions_not_tissue_partition',
        depth_method=dict(name='per_mask_hull_percentile', percentile=FRONT_PERCENTILE,
                          minimum_points=MIN_HULL_POINTS, dilation_px=structure['hull'].get('dilation_px')),
        focal_diagnostic=dict(fy_over_fx=focal_ratio,
            interpretation='legacy_fx_squared_over_fx_fy_for_equal_depth_no_distortion_only'),
        quantities=dict(mask_union='pixel_count', projected_area='cm2_projected_proxy',
            hull_footprint='cm2_occupancy_envelope', hull_volume='cm3_occupancy_envelope',
            estimated_total='cm2_calibration_diagnostic_only'),
        frames=frame_rows, aggregation_status=trait.get('aggregation_status', 'legacy_completeness_not_established'),
        aggregation_audit=trait.get('aggregation_audit'),
        occlusion_status='unknown_no_hidden_tissue_correction', uncertainty_status='not_quantified',
        true_area_cm2=None, true_area_interval_cm2=None, area_basis_id=None,
        biological_face_status='unavailable', normal_axis=None,
        normal_sign_convention=None, validated_metric=False, suitable_for_ppfd=False)


# ---------------------------------------------------------------- summary over plants

def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return [{k.strip(): (v or '').strip() for k, v in r.items() if k} for r in csv.DictReader(fh)]


def record_date(record: dict) -> str:
    values = {str(record[k]) for k in ('date', 'capture_date') if record.get(k)}
    if len(values) > 1:
        raise StudyError('Conflicting date and capture_date')
    value = next(iter(values), '')
    if value:
        date.fromisoformat(value)
    return value


def observation_key(record: dict) -> tuple[str, str, str]:
    return record['plant_id'], record_date(record), record.get('session_id') or ''


def match_record(trait: dict, records: list[dict], traits: list[dict], wildcard: bool = False) -> tuple[dict | None, str]:
    """Exact dates; optional session constraint. Never reuse an undated value across dates."""
    plant, day, session = observation_key(trait)
    for target in ([plant, '*'] if wildcard else [plant]):
        candidates = [r for r in records if r['plant_id'] == target and record_date(r) == day
                      and (not r.get('session_id') or r['session_id'] == session)]
        if len(candidates) > 1:
            raise StudyError(f'Ambiguous measurement records for {plant}/{day}/{session}')
        if candidates:
            r = candidates[0]
            # A date-only measurement cannot choose between multiple sessions of one plant.
            peers = [t for t in traits if t['plant_id'] == plant and record_date(t) == day]
            if not r.get('session_id') and len(peers) > 1:
                raise StudyError(f'Measurement needs session_id for {plant}/{day}')
            if not day and len([t for t in traits if t['plant_id'] == plant]) > 1:
                raise StudyError(f'Undated measurement cannot join repeated plant {plant}')
            return r, 'matched' if day else 'legacy_undated_single_observation'
    return None, 'no_matching_date_or_session'


def area_calibration_contract(trait: dict) -> dict:
    """Identify compatible proxy definitions; physical reference rows stay unchanged."""
    from tools.leaf_area_geometry import LEGACY, METHOD
    method = trait.get('area_method', LEGACY)
    if method not in (LEGACY, METHOD):
        raise StudyError('Unknown area estimator version')
    fields = ('calibration_hash', 'capture_style', 'grouping_policy_hash', 'integration_policy_hash')
    if method == METHOD and any(not trait.get(k) for k in fields):
        raise StudyError('Versioned proxy needs full calibration/capture contract')
    return dict(area_method=method, mask_selection_hash=trait.get('mask_selection_hash', 'all_ready_not_rejected'),
                **{k: trait.get(k, 'legacy_unspecified') for k in fields})


def calibrate(traits: list[dict], measured: dict) -> dict:
    """Through-origin diagnostic fit; all dates of each held-out plant stay together."""
    if len({observation_key(t) for t in traits}) != len(traits):
        raise StudyError('Duplicate calibration observations')
    contracts = {digest(area_calibration_contract(t)) for t in traits}
    if len(contracts) > 1:
        raise StudyError('Incompatible area estimator/calibration/capture styles; fit separate groups')
    contract = area_calibration_contract(traits[0]) if traits else None
    pairs = []
    plants = []
    for t in traits:
        key = observation_key(t)
        y = measured.get(key)
        if y is None and not key[1] and not key[2]:
            y = measured.get(t['plant_id'])  # Single undated legacy API.
        x = t.get('leaf_area_proxy_cm2')
        if y is None or x is None:
            continue
        if not math.isfinite(x) or x <= 0 or not math.isfinite(y) or y <= 0:
            raise StudyError('Calibration areas must be finite and positive')
        pairs.append((x, y)); plants.append(t['plant_id'])
    if not pairs:
        return dict(status='no_measurements', plants=0, area_contract=contract)
    x, y = np.array(pairs, float).T
    k = float((x * y).sum() / (x * x).sum())
    res = dict(status='fitted', plants=len(set(plants)), observations=len(pairs), k=round(k, 4), model='leaf_area_cm2 = k x side-mean proxy (through origin)',
               validation_status='diagnostic_only_no_acceptance_gate', area_contract=contract)
    if len(set(plants)) >= 3:
        errs = []
        plant_array = np.asarray(plants)
        for plant in sorted(set(plants)):
            keep = plant_array != plant
            ki = (x[keep] * y[keep]).sum() / (x[keep] * x[keep]).sum()
            errs.append(float(np.mean(abs(ki * x[~keep] - y[~keep]) / y[~keep])))
        res.update(loo_mean_abs_rel_error_pct=round(100 * float(np.mean(errs)), 1), loo_max_abs_rel_error_pct=round(100 * float(np.max(errs)), 1))
        res['loo_unit'] = 'plant; all dates held out together; errors averaged per plant'
    return res


def summarize(traits: list[dict], out: Path, measurements: Path | None = None) -> dict:
    """traits.csv + TRAITS.md over all plants; measurements/ (person-made) adds calibration and light."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(measurements) if measurements else ROOT / MEASUREMENTS
    la_file, light_file = mdir / 'leaf_area.csv', mdir / 'light.csv'
    keys = [observation_key(t) for t in traits]
    if len(set(keys)) != len(keys):
        raise StudyError('Duplicate plant/date/session traits; specify distinct sessions')
    area_records = [r for r in _read_csv(la_file) if r.get('leaf_area_cm2')] if la_file.exists() else []
    light_records = [r for r in _read_csv(light_file) if r.get('ppfd_umol_m2_s')] if light_file.exists() else []
    measured, light, joins = {}, {}, {}
    for t in traits:
        key = observation_key(t)
        area, astatus = match_record(t, area_records, traits)
        illumination, lstatus = match_record(t, light_records, traits, wildcard=True)
        if area is not None:
            measured[key] = float(area['leaf_area_cm2'])
        if illumination is not None:
            light[key] = illumination
        joins[key] = dict(area_join_status=astatus, light_join_status=lstatus)
    cal = calibrate(traits, measured)
    rows = []
    provenance = {}
    for t in traits:
        key = observation_key(t)
        r = {k: t.get(k) for k in ('plant_id', 'leaf_area_proxy_cm2', 'leaf_area_proxy_photos', 'accepted_share_pct', 'top_leaf_area_cm2',
                                   'hull_footprint_cm2', 'hull_cm3', 'hull_height_mm', 'stem_diameter_mm', 'masks', 'masks_accepted')}
        r.update(capture_date=key[1] or None, session_id=key[2] or None, **joins[key])
        r.update(area_method=area_calibration_contract(t)['area_method'],
                 area_contract_hash=digest(area_calibration_contract(t)))
        r['measured_leaf_area_cm2'] = measured.get(key)
        r['estimated_leaf_area_cm2'] = round(cal['k'] * t['leaf_area_proxy_cm2'], 1) if cal['status'] == 'fitted' and t.get('leaf_area_proxy_cm2') else None
        evidence = dict(schema='plant-traits-summary-provenance-1', observation_key=list(key),
            trait_digest=digest(t), area_provenance=t.get('area_provenance'),
            area_provenance_status='present' if t.get('area_provenance') else 'legacy_provenance_unavailable',
            area_contract=area_calibration_contract(t), calibration=cal,
            measurement_source=dict(path=str(la_file.resolve()), sha256=file_hash(la_file) if la_file.is_file() else None),
            measurement_join=joins[key]['area_join_status'],
            estimated_area_status='diagnostic_only_no_acceptance_gate' if r['estimated_leaf_area_cm2'] is not None else 'unavailable',
            true_area_interval_cm2=None, area_basis_id=None, validated_metric=False)
        evidence_id = digest(evidence)
        provenance[evidence_id] = evidence
        r.update(area_provenance_id=evidence_id, area_provenance_file='traits_area_provenance.json',
                 aggregation_status=t.get('aggregation_status', 'legacy_completeness_not_established'),
                 estimated_area_status=evidence['estimated_area_status'])
        lt = light.get(key)
        if lt:
            ppfd = float(lt['ppfd_umol_m2_s'])
            h = float(lt['photoperiod_h']) if lt.get('photoperiod_h') else None
            if not math.isfinite(ppfd) or ppfd < 0 or (h is not None and (not math.isfinite(h) or not 0 <= h <= 24)):
                raise StudyError('Invalid PPFD or photoperiod')
            dli = ppfd * h * 3600 / 1e6 if h is not None else None
            has_top = t.get('top_leaf_area_cm2') is not None
            canopy = t.get('top_leaf_area_cm2') if has_top else t.get('hull_footprint_cm2')
            r.update(ppfd_umol_m2_s=ppfd, photoperiod_h=h, dli_mol_m2_d=round(dli, 2) if dli is not None else None,
                     canopy_area_cm2_for_light=canopy, canopy_source='top photo leaves' if has_top else 'hull footprint',
                     intercepted_mmol_d=round(dli * canopy / 1e4 * 1000, 1) if canopy is not None and dli is not None else None)
        r.update(light_quantity_status='projected_area_interception_proxy_not_absorption',
                 absorbed_mmol_d=None, photosynthesis_umol_co2_s=None, validated_metric=False)
        rows.append(r)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'traits.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    write_json(out / 'traits_area_provenance.json', dict(schema='plant-traits-provenance-index-1', records=provenance))
    validated = False  # An error statistic alone is not an accuracy acceptance gate.
    doc = dict(schema='plant-traits-summary-1', plants=len({t['plant_id'] for t in traits}), observations=len(rows), calibration=cal, measurements_dir=str(mdir),
               leaf_area_file=la_file.exists(), light_file=light_file.exists(), rows=rows, validated_metric=validated)
    write_json(out / 'traits_summary.json', doc)
    todo = []
    if not measured:
        todo.append(f'잎 면적 보정: 4~6개체를 잎을 떼지 않고 재서(소엽마다 길이×폭, 계수 k 는 보드 위에 편 소엽 몇 장으로) '
                    f'`{la_file}` 에 plant_id, date, leaf_area_cm2로 적고 같은 날짜의 여러 세션은 session_id로 구분하세요.')
    elif 'loo_mean_abs_rel_error_pct' not in cal:
        todo.append(f'잎 면적 보정 개체가 {cal["plants"]}개입니다. 3개체 이상이면 오차(한 개체씩 빼고 맞춘 오차)가 계산됩니다.')
    if not light:
        todo.append(f'광량: 식물 위치에서 PPFD(µmol m⁻² s⁻¹)와 일장(h)을 `{light_file}` 에 plant_id(전체 같으면 *), date, ppfd_umol_m2_s, photoperiod_h로 적으세요. 필요한 경우 session_id를 추가하세요.')
    lines = ['# 개체별 형질', '', f'개체 {doc["plants"]}개, 관찰 {len(rows)}건. 잎 면적은 이미지 대리값이며 보정한 값도 검증 전에는 추정치입니다.', '',
             '| 개체/날짜/세션 | 대리 잎 면적 cm² | 수락 비율 % | 위 사진 잎 cm² | 외곽 바닥 cm² | 외곽 cm³ | 높이 mm | 줄기 mm | 추정 잎 면적 cm² | DLI | 투영면적 기반 광 포획 근사 mmol/d |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| ' + ' | '.join('—' if v is None else str(v) for v in (
            '/'.join(str(r[k] or '미상') for k in ('plant_id','capture_date','session_id')), r['leaf_area_proxy_cm2'], r['accepted_share_pct'], r['top_leaf_area_cm2'], r['hull_footprint_cm2'], r['hull_cm3'],
            r['hull_height_mm'], r['stem_diameter_mm'], r['estimated_leaf_area_cm2'], r.get('dli_mol_m2_d'), r.get('intercepted_mmol_d'))) + ' |')
    lines += ['', '## 보정', '', f"- 상태: {cal['status']}, 개체 {cal['plants']}" + (f", k = {cal['k']}" if 'k' in cal else '') +
              (f", 한 개체의 모든 날짜를 빼고 맞춘 평균 오차 {cal['loo_mean_abs_rel_error_pct']}% (최대 {cal['loo_max_abs_rel_error_pct']}%)" if 'loo_mean_abs_rel_error_pct' in cal else ''),
              '', '오차 산출은 정확도 검증 통과가 아닙니다. 광 포획 근사값은 흡수광·광합성량이 아닙니다.']
    if todo:
        lines += ['', '## 사람이 할 일 (형질에 필요한 것만)', ''] + [f'{i}. {t}' for i, t in enumerate(todo, 1)]
    (out / 'TRAITS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    doc['human_tasks'] = todo
    return doc


def from_run(run: Path, config: Path | None = None) -> list[dict]:
    """traits.json of every plant folder of a plant_pipeline run (S8 writes them)."""
    return [read_json(p) for p in sorted(Path(run).glob('*/traits.json'))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', type=Path, help='a plant_pipeline run folder; summarises the traits.json of its plants')
    ap.add_argument('--traits', type=Path, nargs='*', default=[], help='extra traits.json files (e.g. from several auto runs)')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--measurements', type=Path)
    a = ap.parse_args()
    traits = (from_run(a.run) if a.run else []) + [read_json(p) for p in a.traits]
    if not traits:
        raise StudyError('No traits.json found')
    doc = summarize(traits, a.out or a.run, a.measurements)
    print('\n'.join((Path(a.out or a.run) / 'TRAITS.md').read_text(encoding='utf-8').splitlines()[:6 + len(traits)]))
    print({k: doc[k] for k in ('calibration', 'leaf_area_file', 'light_file', 'validated_metric')})


if __name__ == '__main__':
    main()
