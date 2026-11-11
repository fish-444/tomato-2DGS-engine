"""Light-distribution validation: empty rig -> held-out empty rig -> opaque occluder -> reviewed leaves -> plant PPFD.

    python -X utf8 -m tools.ppfd_validation templates --out build/ppfd_inputs_<ts>
    python -X utf8 -m tools.ppfd_validation audit     --study S/study.json --out build/<new>
    python -X utf8 -m tools.ppfd_validation run       --study S/study.json --out build/ppfd_validation_<ts> [--execute]
    python -X utf8 -m tools.ppfd_validation compare   --predictions P.csv --readings R.csv --split split.json --out build/<new>
    python -X utf8 -m tools.ppfd_validation convergence --prepared build/<run>/prepared --case C --out build/<new>
    python -X utf8 -m tools.ppfd_validation synthetic --out build/ppfd_validation_synthetic_<ts>
    python -X utf8 -m tools.ppfd_validation stale     --run build/ppfd_validation_<ts>

Spec: handoff/CLAUDE_PPFD_VALIDATION_CODING_PROMPT_2026-10-02.md (Codex). Reuses, never rewrites:
tools.groimp_density (mesh_node, sensor_triangles, surface_gate, execute, parse_log, verify_prepared),
tools.groimp_light_sanity.run_groimp, tomato3d.light (LightScene, build_emitters, fixture_axes, direct_light).

Directional LED adapter (measured natively 2026-10-02, build/ppfd_led_probe*): GroIMP 2.2.1 CPU LightModel
* PointLight: isotropic, matches the analytic point source.
* PhysicalLight (GPUFlux tabulated distribution): NOT usable on the CPU path - even a uniform table sends rays only
  along a few discrete directions. Never used here.
* AreaLight on a Parallelogram: radiance ~ cos^e, i.e. intensity I(theta) ~ cos^(e+1) theta about the emitting
  normal, total emitted flux = setPower; the Parallelogram emits along local +y, spans local x (centred, width) and
  local +z (from 0 to length). A LightScene fixture with cos^m (m >= 1) and a 90 deg cutoff is therefore represented
  exactly as one emitting rectangle with exponent m - 1. A cutoff below 90 deg, m < 1 or a zero-area emitter is
  refused (unsupported), never replaced by a surrogate.
Units: grow frame mm in every file; metres only inside generated RGG (MM_PER_M applied once in `_m`). Numeric GroIMP
power is photon flux in umol s-1 (achromatic convention); no watt-to-photon conversion is ever applied to it.
Everything stays diagnostic: validated_metric=false, suitable_for_ppfd=false; only a human review may change that.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tomato3d.io.schemas import PPFD_READINGS_CSV_COLUMNS, LightSimulationSettings, ScalarInput, SchemaError
from tomato3d.light.frames import fixture_axes
from tomato3d.light.raytrace import Occluders, Receivers, direct_light
from tomato3d.light.scene import build_emitters, cosine_exponent, input_records, load_scene
from tools.groimp_density import execute as density_execute
from tools.groimp_density import mesh_node, parse_log, sensor_triangles, surface_gate, verify_prepared
from tools.groimp_light_sanity import CORE, run_groimp
from tools.leaf_light.common import (StudyError, digest, file_hash, number, output_dir, read_json, rows, write_json,
                                    write_rows)

STUDY_SCHEMA = 'ppfd-validation-study-1'
READINGS_SCHEMA = 'ppfd-readings-2'
STAGES = ('empty_rig', 'occluder', 'plant')
ROLES = ('calibration', 'validation')
MM_PER_M = 1000.0
UNAVAILABLE = 'unavailable'
#: exact SI constants (2019 definitions) for the optional spectral path
PLANCK_J_S, LIGHT_M_S, AVOGADRO = 6.62607015e-34, 299_792_458.0, 6.02214076e23
PAR_NM = (400.0, 700.0)
ADAPTER_KIND = 'groimp_arealight_cosine_power'
ADAPTER_EVIDENCE = ('native probes 2026-10-02: AreaLight exponent 0/0.5/4 matched cos^1/cos^1.5/cos^5 point-source '
                    'analytics within Monte-Carlo noise (build/ppfd_led_probe2_area*); PhysicalLight failed (uniform '
                    'table -> discrete directions); see module docstring')
FROZEN_FILES = ('study.json',)

# ---------------------------------------------------------------- CSV contracts (versioned; old readers keep theirs)
LAMP_CONDITION_COLUMNS = ('fixture_id', 'output_setting', 'output_setting_unit', 'is_full_output', 'spectrum_file',
                          'spectrum_value_kind', 'stabilization_min', 'status', 'source', 'note')
RIG_SURFACE_COLUMNS = ('surface_id', 'kind', 'role', 'center_x_mm', 'center_y_mm', 'center_z_mm', 'normal_x', 'normal_y',
                       'normal_z', 'width_axis_x', 'width_axis_y', 'width_axis_z', 'width_mm', 'height_mm',
                       'reflectance', 'transmittance', 'geometry_status', 'geometry_source', 'optics_status',
                       'optics_source', 'note')
SENSOR_COLUMNS = ('sensor_id', 'sensor_model', 'serial_number', 'calibration_reference', 'calibration_date',
                  'sensing_shape', 'sensing_width_mm', 'sensing_height_mm', 'sensing_diameter_mm', 'angular_response',
                  'dimensions_status', 'dimensions_source', 'probe_body_note')
POINT_COLUMNS = ('point_id', 'stage', 'case_id', 'plant_id', 'sensor_id', 'x_mm', 'y_mm', 'z_mm', 'normal_x', 'normal_y',
                 'normal_z', 'height_band', 'split_role', 'position_status', 'position_source', 'accessible', 'note')
READING_COLUMNS = ('schema_version', 'study_id', 'run_id', 'stage', 'case_id', 'plant_id', 'session_id',
                   'geometry_snapshot_date', 'sensor_id', 'point_id', 'repeat_id', 'observed_at', 'split_role',
                   'ppfd_umol_m2_s', 'all_fixtures_full_output', 'ambient_condition', 'led_stabilized_min',
                   'sensor_moved_leaves', 'disturbed', 'observer', 'note')
SPECTRUM_COLUMNS = ('fixture_id', 'wavelength_low_nm', 'wavelength_high_nm', 'value', 'value_kind', 'quantity', 'unit')
PREDICTION_COLUMNS = ('stage', 'case_id', 'point_id', 'sensor_id', 'height_band', 'split_role', 'job',
                      'predicted_ppfd_umol_m2_s', 'simulated_ppfd_declared_inputs_umol_m2_s',
                      'python_direct_ppfd_declared_inputs_umol_m2_s', 'empty_twin_simulated_ppfd_umol_m2_s',
                      'plant_contribution_umol_m2_s', 'occlusion_class', 'quantity', 'backend', 'lamp_model_sha256',
                      'absolute_status', 'status', 'reason')
AMBIENT = ('excluded_dark', 'subtracted', 'present_unsubtracted')
YES_NO = ('yes', 'no', 'unknown')
#: rig surface roles; display_only objects exist for viewing and never enter an optical scene
SURFACE_ROLES = ('rig', 'physical_occluder', 'display_only')
SUPPORTED_SHAPES = ('rectangle', 'disc')


def _m(mm: float) -> float:
    """The single mm -> m conversion used for generated RGG."""
    return float(mm) / MM_PER_M


def _f(x: float) -> str:
    return repr(float(x))


def _vec(row: dict, prefix: str, unit: str = '') -> np.ndarray:
    return np.array([number(row[f'{prefix}{a}{unit}'], f'{prefix}{a}{unit}') for a in 'xyz'], float)


def unit_normal(row: dict, prefix: str = 'normal_') -> np.ndarray:
    n = _vec(row, prefix)
    if not np.isclose(np.linalg.norm(n), 1.0, atol=1e-6):
        raise StudyError(f'{prefix}* must be a unit vector (got |n|={np.linalg.norm(n):.6g}); it is not normalised silently')
    return n


def aware_time(text: str, label: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise StudyError(f'{label}: not an ISO-8601 time') from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise StudyError(f'{label}: observation time must carry a timezone offset')
    return stamp


# ---------------------------------------------------------------- templates
def templates(out: Path) -> dict:
    """Header-only human-input contracts. Nothing physical is written; null means not yet decided."""
    folder = output_dir(out)
    for name, cols in (('lamp_conditions.csv', LAMP_CONDITION_COLUMNS), ('rig_surfaces.csv', RIG_SURFACE_COLUMNS),
                       ('sensors.csv', SENSOR_COLUMNS), ('measurement_points.csv', POINT_COLUMNS),
                       ('ppfd_readings.csv', READING_COLUMNS), ('spectrum.csv', SPECTRUM_COLUMNS)):
        write_rows(folder / name, [], cols)
    write_json(folder / 'study.json', dict(
        schema_version=STUDY_SCHEMA, study_id=None, timezone=None, scene=None,
        lamp_conditions='lamp_conditions.csv', rig_surfaces='rig_surfaces.csv', sensors='sensors.csv',
        points='measurement_points.csv', readings='ppfd_readings.csv', spectrum=None, cases=[],
        acquisition_proposal=dict(note='configurable pilot proposal, not a sample-size requirement or measured table',
                                  empty_rig_heights=3, empty_rig_grid=[3, 3], plant_height_bands=3,
                                  plant_points_per_band=3, sensor_replacement_repeats=3),
        fit=dict(enabled_parameters=[], bounds=dict(photon_flux_scale=None, half_intensity_half_angle_deg=None),
                 half_angle_grid_step_deg=None, min_points_above_parameters=1, profile_chi2_95=3.841458820694124),
        solver=dict(backend='groimp_cpu_lightmodel', rays=None, max_depth=None, seed=None,
                    convergence_rays=[], convergence_seeds=[], max_local_native_jobs=None,
                    numerical_convergence_rel_tol=None),
        python=dict(emitter_grid=None, sensor_samples=None, disc_segments=None,
                    shadow_class_tolerance=None),
        sensitivity=dict(lamp_height_mm=None, half_angle_deg=None, flux_rel=None, sensor_tilt_deg=None,
                         point_offset_mm=None),
        acceptance=dict(nrmse_max=None, abs_bias_max_umol_m2_s=None),
        inference=dict(min_independent_units_for_ci=None, bootstrap_resamples=None, bootstrap_seed=None)))
    (folder / 'README.txt').write_text(
        'Header-only templates; no readings, geometry or lamp values are supplied by the software.\n'
        'scene = tomato3d LightScene JSON made by a person (config/ belongs to the human).\n'
        'Every length is mm in the grow frame; normals are unit vectors (not normalised for you).\n'
        'cases: {case_id, stage: empty_rig|occluder|plant, session_id, geometry_snapshot_date,\n'
        '        occluder_surface_ids (occluder), plant_id + geometry + geometry_variant_id + surface_review (plant)}.\n'
        'Readings: one row per sensor placement; repeats share point_id and split_role; observed_at with offset.\n'
        'Null numbers in study.json are undecided: the dependent step reports unavailable, it never uses a default.\n',
        encoding='utf-8')
    (folder / '작성안내.txt').write_text(
        '빈 칸 양식만 있습니다. 측정하지 않은 값은 비워 두세요 (0 이 아님).\n'
        '1) scene: 조명 위치·기울기·크기·광량(PPF, umol/s)·배광각 - 사람이 LightScene JSON 으로 작성\n'
        '2) lamp_conditions.csv: 출력 설정(디머 %), 전체 출력 여부, 안정화 시간, 스펙트럼 파일\n'
        '3) rig_surfaces.csv: 벽·바닥·반사판·차광판(occluder_plate) 위치(mm), 법선, 크기, 반사율/투과율과 출처\n'
        '4) sensors.csv: 광량계 모델·시리얼·교정 기준·감지면 모양(rectangle/disc)과 실제 치수\n'
        '5) measurement_points.csv: 측정점 좌표(mm, 재배 좌표계)·센서 방향·높이대·calibration/validation 구분\n'
        '6) ppfd_readings.csv: 같은 점의 반복 측정은 같은 point_id, 같은 split_role. 시간은 +09:00 처럼 시간대 포함\n'
        '권장 배치(제안일 뿐): 빈 랙 3높이 x 3x3, 식물 3높이대 x 3점 x 센서 재설치 3회.\n', encoding='utf-8')
    return dict(status='empty_templates_created', out=str(folder))


# ---------------------------------------------------------------- study loading and audit
def load_study(study_path: Path) -> dict:
    study = read_json(study_path)
    if study.get('schema_version') != STUDY_SCHEMA:
        raise StudyError(f'unsupported study schema {study.get("schema_version")!r}; expected {STUDY_SCHEMA}')
    return study


def _path(base: Path, rel: str | None) -> Path | None:
    return None if not rel else (base / rel)


def _csv(base: Path, rel: str | None, cols: tuple) -> list[dict]:
    p = _path(base, rel)
    return rows(p, cols) if p is not None and p.is_file() else []


def sensing_mesh(point: dict, sensor: dict, disc_segments: int | None) -> tuple[np.ndarray, float]:
    """Triangles (grow mm, front face = point normal) of the actual sensing area, and that area in mm2."""
    shape = sensor.get('sensing_shape')
    normal = unit_normal(point)
    if shape == 'rectangle':
        w = number(sensor['sensing_width_mm'], 'sensing_width_mm'); h = number(sensor['sensing_height_mm'], 'sensing_height_mm')
        if w <= 0 or h <= 0:
            raise StudyError('sensing rectangle dimensions must be positive')
        rec = dict(x_mm=point['x_mm'], y_mm=point['y_mm'], z_mm=point['z_mm'], normal_x=normal[0], normal_y=normal[1],
                   normal_z=normal[2], width_mm=w, height_mm=h)
        return sensor_triangles(rec), w * h
    if shape == 'disc':
        if not disc_segments or disc_segments < 8:
            raise StudyError('python.disc_segments (>= 8) is required for a disc sensor; not defaulted')
        d = number(sensor['sensing_diameter_mm'], 'sensing_diameter_mm')
        if d <= 0:
            raise StudyError('sensing diameter must be positive')
        r = d / 2
        # area-matched polygon: N/2 R^2 sin(2 pi/N) = pi r^2
        big = r * math.sqrt(2 * math.pi / (disc_segments * math.sin(2 * math.pi / disc_segments)))
        t = np.cross(normal, [0, 0, 1] if abs(normal[2]) < .9 else [0, 1, 0]); t /= np.linalg.norm(t)
        b = np.cross(normal, t)
        c = _vec(point, '', '_mm')
        ang = 2 * math.pi * np.arange(disc_segments + 1) / disc_segments
        ring = c + big * (np.cos(ang)[:, None] * t + np.sin(ang)[:, None] * b)
        return np.array([[c, ring[k], ring[k + 1]] for k in range(disc_segments)]), math.pi * r * r
    raise StudyError(f'unsupported sensing shape {shape!r}; supported: {SUPPORTED_SHAPES} '
                     '(a new shape needs a measured-shape adapter)')


def footprint_samples(point: dict, sensor: dict, n: int) -> np.ndarray:
    """Points (grow mm) evenly covering the sensing area, for the python comparator."""
    normal = unit_normal(point)
    t = np.cross(normal, [0, 0, 1] if abs(normal[2]) < .9 else [0, 1, 0]); t /= np.linalg.norm(t)
    b = np.cross(normal, t)
    c = _vec(point, '', '_mm')
    g = (np.arange(n) + 0.5) / n - 0.5
    u, v = np.meshgrid(g, g)
    if sensor['sensing_shape'] == 'rectangle':
        w = float(sensor['sensing_width_mm']); h = float(sensor['sensing_height_mm'])
        uu, vv = u.ravel() * w, v.ravel() * h
    else:
        d = float(sensor['sensing_diameter_mm'])
        uu, vv = u.ravel() * d, v.ravel() * d
        keep = uu ** 2 + vv ** 2 <= (d / 2) ** 2
        uu, vv = uu[keep], vv[keep]
    return c + uu[:, None] * t + vv[:, None] * b


def _status_missing(row: dict, key: str) -> bool:
    return row.get(key, '') in ('', UNAVAILABLE)


def validate_inputs(study_path: Path) -> dict:
    """Read-only audit of every input; lists each missing/non-measured physical input explicitly."""
    study = load_study(study_path)
    base = study_path.parent
    missing: list[dict] = []
    issues: list[str] = []
    files = {}

    def need(entity: str, name: str, status: str, reason: str, blocks: str) -> None:
        missing.append(dict(entity=entity, input=name, status=status, reason=reason, blocks=blocks))

    for key in ('scene', 'lamp_conditions', 'rig_surfaces', 'sensors', 'points', 'readings', 'spectrum'):
        p = _path(base, study.get(key))
        files[key] = dict(path=str(p) if p else None, exists=bool(p and p.is_file()),
                          sha256=file_hash(p) if p and p.is_file() else None)
    for key in ('study_id', 'timezone'):
        if not study.get(key):
            need('study', key, UNAVAILABLE, 'not set in study.json', 'all')
    # lamp
    scene = None
    if not files['scene']['exists']:
        need('lamp', 'LightScene (fixture centre, aim, size, photon flux, distribution, plant placement, leaf optics)',
             UNAVAILABLE, 'no scene file; config/light_scene.json does not exist', 'absolute PPFD, every stage')
    else:
        try:
            scene = load_scene(Path(files['scene']['path']))
        except SchemaError as exc:
            issues.append(f'scene invalid: {exc}')
        if scene is not None:
            for rec in input_records(scene):
                if rec.status != 'measured':
                    need('scene', rec.name, rec.status, rec.source, 'absolute PPFD' if 'fixtures' in rec.name else
                         'plant stage' if 'leaf_optics' in rec.name or 'plants' in rec.name else 'context')
            for fx in scene.fixtures:
                ok, why = adapter_support(fx)
                if not ok:
                    need('lamp_adapter', fx.fixture_id, 'unsupported', why, 'native GroIMP prediction')
    lamp_rows = _csv(base, study.get('lamp_conditions'), LAMP_CONDITION_COLUMNS)
    if scene is not None:
        for fx in scene.fixtures:
            row = next((r for r in lamp_rows if r['fixture_id'] == fx.fixture_id), None)
            if row is None:
                need('lamp_conditions', f'{fx.fixture_id}: output setting / full output / stabilization', UNAVAILABLE,
                     'no lamp_conditions row', 'comparison with readings')
            else:
                for k in ('output_setting', 'is_full_output', 'stabilization_min', 'spectrum_file'):
                    if _status_missing(row, k):
                        need('lamp_conditions', f'{fx.fixture_id}.{k}', UNAVAILABLE, 'blank', 'comparison with readings'
                             if k != 'spectrum_file' else 'spectral path only (photon-flux path unaffected)')
    # rig surfaces
    rig = _csv(base, study.get('rig_surfaces'), RIG_SURFACE_COLUMNS)
    if not rig:
        need('rig', 'walls/floor/reflective surfaces and occluder plate geometry + optics', UNAVAILABLE,
             'rig_surfaces.csv empty: reflections in the real rig are unaccounted', 'empty rig, occluder stage')
    for r in rig:
        if r['role'] not in SURFACE_ROLES:
            issues.append(f'rig surface {r["surface_id"]}: role must be one of {SURFACE_ROLES}')
        if r['role'] == 'display_only':
            continue
        for k in ('geometry_status', 'optics_status'):
            if r[k] != 'measured':
                need('rig', f'{r["surface_id"]}.{k}', r[k] or UNAVAILABLE, r.get(k.replace('status', 'source')) or 'blank',
                     'absolute PPFD')
        try:
            rig_surface_triangles(r)
        except StudyError as exc:
            issues.append(f'rig surface {r["surface_id"]}: {exc}')
    # sensors
    sensors = {r['sensor_id']: r for r in _csv(base, study.get('sensors'), SENSOR_COLUMNS)}
    if not sensors:
        need('sensor', 'quantum sensor model, calibration reference, sensing shape and dimensions', UNAVAILABLE,
             'sensors.csv empty: no sensor is available', 'every comparison with readings')
    for s in sensors.values():
        if s['sensing_shape'] not in SUPPORTED_SHAPES:
            issues.append(f'sensor {s["sensor_id"]}: unsupported sensing shape {s["sensing_shape"]!r}')
        if s['angular_response'] not in ('cosine_corrected',):
            need('sensor', f'{s["sensor_id"]}.angular_response', s['angular_response'] or UNAVAILABLE,
                 'only a planar cosine-corrected receiver is modelled', 'comparison with readings')
        if s['dimensions_status'] != 'measured':
            need('sensor', f'{s["sensor_id"]}.sensing dimensions', s['dimensions_status'] or UNAVAILABLE,
                 s['dimensions_source'] or 'blank', 'absolute PPFD')
        for k in ('calibration_reference', 'sensor_model'):
            if _status_missing(s, k):
                need('sensor', f'{s["sensor_id"]}.{k}', UNAVAILABLE, 'blank', 'comparison with readings')
    # points and readings
    points = _csv(base, study.get('points'), POINT_COLUMNS)
    readings = _csv(base, study.get('readings'), READING_COLUMNS)
    point_audit = audit_points(points, sensors, study)
    reading_audit = audit_readings(readings, points, study)
    issues += point_audit['issues'] + reading_audit['issues']
    if not points:
        need('points', 'measurement point positions/normals', UNAVAILABLE, 'measurement_points.csv empty', 'all stages')
    if not reading_audit['n_rows']:
        need('readings', 'independent PPFD readings', UNAVAILABLE, 'ppfd_readings.csv empty (no sensor available)',
             'calibration and every independent accuracy metric')
    for p in points:
        if p['position_status'] != 'measured':
            need('points', f'{p["case_id"]}:{p["point_id"]} position', p['position_status'] or UNAVAILABLE,
                 p['position_source'] or 'blank', 'absolute PPFD at that point')
    cases = study.get('cases') or []
    if not cases:
        need('study', 'cases', UNAVAILABLE, 'no empty_rig/occluder/plant case declared', 'all stages')
    for c in cases:
        if c.get('stage') not in STAGES:
            issues.append(f'case {c.get("case_id")}: unknown stage {c.get("stage")!r}')
        if c.get('stage') == 'plant':
            for k in ('plant_id', 'geometry', 'geometry_variant_id', 'surface_review'):
                if not c.get(k):
                    need('plant', f'{c.get("case_id")}.{k}', UNAVAILABLE, 'reviewed leaf triangles not supplied',
                         'plant stage')
        if c.get('stage') == 'occluder' and not c.get('occluder_surface_ids'):
            need('occluder', f'{c.get("case_id")}.occluder_surface_ids', UNAVAILABLE, 'no measured plate', 'occluder stage')
    solver = study.get('solver') or {}
    for k in ('rays', 'max_depth', 'seed'):
        if solver.get(k) is None:
            need('solver', k, UNAVAILABLE, 'undecided numerical setting (not defaulted)', 'native execution')
    status = 'invalid' if issues else ('templates_only_no_measurements' if not reading_audit['n_rows'] else 'audited')
    return dict(schema='ppfd-validation-audit-1', status=status, study=str(study_path.resolve()),
                study_sha256=file_hash(study_path), files=files, issues=issues, missing_physical_inputs=missing,
                points=point_audit, readings=reading_audit, validated_metric=False, suitable_for_ppfd=False)


def audit_points(points: list[dict], sensors: dict, study: dict) -> dict:
    issues, seen = [], set()
    for p in points:
        key = (p['case_id'], p['point_id'])
        if key in seen:
            issues.append(f'duplicate point {key}')
        seen.add(key)
        if p['stage'] not in STAGES:
            issues.append(f'point {key}: unknown stage {p["stage"]!r}')
        if p['split_role'] not in ROLES:
            issues.append(f'point {key}: split_role must be one of {ROLES}')
        if p['sensor_id'] and sensors and p['sensor_id'] not in sensors:
            issues.append(f'point {key}: unknown sensor {p["sensor_id"]!r}')
        try:
            unit_normal(p)
            _vec(p, '', '_mm')
        except StudyError as exc:
            issues.append(f'point {key}: {exc}')
    return dict(n_points=len(points), issues=issues,
                by_role={r: sum(p['split_role'] == r for p in points) for r in ROLES})


def audit_readings(readings: list[dict], points: list[dict], study: dict) -> dict:
    issues, seen = [], set()
    point_role = {(p['case_id'], p['point_id']): p['split_role'] for p in points}
    roles_by_point = defaultdict(set)
    for r in readings:
        key = (r['case_id'], r['point_id'], r['repeat_id'])
        if r['schema_version'] != READINGS_SCHEMA:
            issues.append(f'reading {key}: schema_version must be {READINGS_SCHEMA}')
        if key in seen:
            issues.append(f'duplicated repeat {key}')
        seen.add(key)
        if r['split_role'] not in ROLES:
            issues.append(f'reading {key}: unknown split_role')
        roles_by_point[key[:2]].add(r['split_role'])
        if key[:2] not in point_role:
            issues.append(f'reading {key}: point not declared in measurement_points.csv')
        elif point_role[key[:2]] != r['split_role']:
            issues.append(f'reading {key}: split_role differs from its point ({point_role[key[:2]]})')
        try:
            aware_time(r['observed_at'], f'reading {key} observed_at')
        except StudyError as exc:
            issues.append(str(exc))
        if r['ambient_condition'] not in AMBIENT:
            issues.append(f'reading {key}: ambient_condition must be one of {AMBIENT}')
        for k in ('sensor_moved_leaves', 'disturbed', 'all_fixtures_full_output'):
            if r[k] not in YES_NO:
                issues.append(f'reading {key}: {k} must be one of {YES_NO}')
        try:
            v = number(r['ppfd_umol_m2_s'], 'ppfd', missing=True)
            if v is not None and v < 0:
                issues.append(f'reading {key}: negative PPFD')
        except StudyError as exc:
            issues.append(f'reading {key}: {exc}')
        if study.get('study_id') and r['study_id'] != study['study_id']:
            issues.append(f'reading {key}: study_id mismatch')
    for key, roles in roles_by_point.items():
        if len(roles) > 1:
            issues.append(f'point {key}: repeats split across calibration and validation')
    return dict(n_rows=len(readings), issues=issues,
                by_role={r: sum(x['split_role'] == r for x in readings) for r in ROLES})


def to_tomato3d_readings(readings: list[dict], points: list[dict], setup_id: str) -> list[dict]:
    """Adapter to the existing PPFD_READINGS_CSV_COLUMNS reader (validation -> evaluation; extra fields in note)."""
    where = {(p['case_id'], p['point_id']): p for p in points}
    out = []
    for r in readings:
        p = where[(r['case_id'], r['point_id'])]
        out.append(dict(reading_id=f'{r["case_id"]}:{r["point_id"]}:{r["repeat_id"]}', setup_id=setup_id,
                        scenario_id='as_built', measured_at=r['observed_at'], sensor_model=r['sensor_id'],
                        x_mm=p['x_mm'], y_mm=p['y_mm'], z_mm=p['z_mm'], normal_x=p['normal_x'], normal_y=p['normal_y'],
                        normal_z=p['normal_z'], all_fixtures_full_output=r['all_fixtures_full_output'],
                        ppfd_umol_m2_s=r['ppfd_umol_m2_s'] or UNAVAILABLE,
                        split_role='evaluation' if r['split_role'] == 'validation' else 'calibration',
                        region_label=p['height_band'], observer=r['observer'],
                        note=f'stage={r["stage"]};session={r["session_id"]};{r["note"]}'))
    assert all(set(o) == set(PPFD_READINGS_CSV_COLUMNS) for o in out)
    return out


# ---------------------------------------------------------------- spectral path (only when spectra are supplied)
def spectral_to_photon_umol(spectrum: list[dict]) -> float:
    """Integrate energy bins over 400-700 nm into photons: n = E lambda / (h c N_A), per bin, never one W->PPFD factor.

    value_kind 'per_nm' (spectral density) is integrated over the clipped bin; 'bin_integrated' must lie inside the band.
    quantity 'energy' in W (or W m-2); 'photon' rows are summed as given (umol s-1 or umol m-2 s-1).
    """
    total = 0.0
    for r in spectrum:
        lo, hi = number(r['wavelength_low_nm'], 'low'), number(r['wavelength_high_nm'], 'high')
        if hi <= lo:
            raise StudyError('spectral bin with non-positive width')
        clip_lo, clip_hi = max(lo, PAR_NM[0]), min(hi, PAR_NM[1])
        if clip_hi <= clip_lo:
            continue
        value = number(r['value'], 'spectral value')
        if r['value_kind'] == 'per_nm':
            width_nm, centre_nm = clip_hi - clip_lo, (clip_lo + clip_hi) / 2
        elif r['value_kind'] == 'bin_integrated':
            if lo < PAR_NM[0] or hi > PAR_NM[1]:
                raise StudyError('a bin-integrated value straddling the PAR edge cannot be split')
            width_nm, centre_nm = 1.0, (lo + hi) / 2
        else:
            raise StudyError(f'value_kind must be per_nm or bin_integrated, got {r["value_kind"]!r}')
        if r['quantity'] == 'energy':
            photons_mol = value * width_nm * centre_nm * 1e-9 / (PLANCK_J_S * LIGHT_M_S * AVOGADRO)
            total += photons_mol * 1e6
        elif r['quantity'] == 'photon':
            total += value * width_nm
        else:
            raise StudyError('quantity must be energy or photon; electrical watts are not optical power')
    return total


# ---------------------------------------------------------------- lamp adapter
def adapter_support(fx) -> tuple[bool, str]:
    em, missing = build_emitters(fx)
    if em is None:
        return False, 'missing inputs: ' + ', '.join(missing)
    if abs(fx.distribution.cutoff_half_angle_deg - 90.0) > 1e-9:
        return False, 'cutoff below 90 deg is not representable by GroIMP AreaLight (no verified cut-off light)'
    if em.cosine_exponent < 1 - 1e-9:
        return False, f'cos exponent {em.cosine_exponent:.4g} < 1 needs a negative AreaLight exponent (unverified)'
    if fx.emitting_length_mm.value <= 0 or fx.emitting_width_mm.value <= 0:
        return False, 'zero-area emitter: AreaLight needs a rectangle; no point surrogate is substituted'
    return True, 'supported: one AreaLight rectangle, exponent = m - 1'


def arealight_node(fx) -> tuple[str, dict]:
    ok, why = adapter_support(fx)
    if not ok:
        raise StudyError(f'fixture {fx.fixture_id}: {why}')
    aim, long_axis, width_axis = fixture_axes(math.radians(fx.aim_tilt_deg.value), math.radians(fx.aim_azimuth_deg.value),
                                              math.radians(fx.long_axis_azimuth_deg.value))
    length_m, width_m = _m(fx.emitting_length_mm.value), _m(fx.emitting_width_mm.value)
    centre_m = np.array([_m(v) for v in fx.emitting_center_grow_mm.value])
    rot = np.column_stack([width_axis, aim, long_axis])  # local x -> width, local +y (emitting) -> aim, local z -> long
    origin = centre_m - long_axis * length_m / 2  # Parallelogram spans local z from 0 to length
    mat = [rot[0, 0], rot[0, 1], rot[0, 2], origin[0], rot[1, 0], rot[1, 1], rot[1, 2], origin[1],
           rot[2, 0], rot[2, 1], rot[2, 2], origin[2], 0.0, 0.0, 0.0, 1.0]
    m = cosine_exponent(math.radians(fx.distribution.half_intensity_half_angle_deg.value))
    flux = fx.photon_flux_umol_s.value
    node = (f'[ Null().(setTransform(new javax.vecmath.Matrix4d({", ".join(_f(v) for v in mat)}))) '
            f'Parallelogram({_f(length_m)}f, {_f(width_m)}f).(setLight(new de.grogra.imp3d.objects.AreaLight()'
            f'.(setPower({_f(flux)}f), setExponent({_f(m - 1)}f)))) ]')
    return node, dict(fixture_id=fx.fixture_id, kind=ADAPTER_KIND, cos_exponent_m=m, arealight_exponent=m - 1,
                      photon_flux_umol_s=flux, flux_status=fx.photon_flux_umol_s.status,
                      length_m=length_m, width_m=width_m, centre_grow_m=centre_m.tolist(), aim=aim.tolist(),
                      long_axis=long_axis.tolist(), width_axis=width_axis.tolist())


def lamp_adapter(scene_path: Path, scene=None) -> dict:
    """groimp_density-compatible lamp_adapter.json content; review_state stays pending for a person to accept."""
    scene = scene or load_scene(scene_path)
    nodes, fixtures, unsupported = [], [], []
    for fx in scene.fixtures:
        ok, why = adapter_support(fx)
        if not ok:
            unsupported.append(dict(fixture_id=fx.fixture_id, reason=why))
            continue
        node, rec = arealight_node(fx)
        nodes.append(node); fixtures.append(rec)
    return dict(scene_sha256=file_hash(scene_path), review_state='pending', reviewer=None, coordinate_frame='grow_m',
                power_convention='photon_umol_s', empty_scene_validation=None, header_rgg='',
                initialization_rgg=' '.join(nodes), absolute_led_ppfd_valid=False, adapter_kind=ADAPTER_KIND,
                fixtures=fixtures, unsupported_fixtures=unsupported, evidence=ADAPTER_EVIDENCE,
                note='AreaLight power is read as photon flux umol s-1 (GroIMP labels it W); no conversion applied. '
                     'Accept only after reviewing the empty-scene check against independent readings.')


# ---------------------------------------------------------------- rig surfaces / occluder
def rig_surface_triangles(r: dict) -> np.ndarray:
    n = unit_normal(r)
    u = _vec(r, 'width_axis_')
    if not np.isclose(np.linalg.norm(u), 1, atol=1e-6) or abs(float(u @ n)) > 1e-6:
        raise StudyError('width_axis must be a unit vector perpendicular to the normal')
    v = np.cross(n, u)
    c = _vec(r, 'center_', '_mm')
    w, h = number(r['width_mm'], 'width_mm'), number(r['height_mm'], 'height_mm')
    if w <= 0 or h <= 0:
        raise StudyError('surface dimensions must be positive')
    k = [c + sx * w / 2 * u + sy * h / 2 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.array([[k[0], k[1], k[2]], [k[0], k[2], k[3]]])


def surface_optics(r: dict) -> tuple[float, float]:
    rho = number(r['reflectance'], 'reflectance'); tau = number(r['transmittance'], 'transmittance')
    if min(rho, tau) < 0 or rho + tau > 1 + 1e-9:
        raise StudyError(f'surface {r["surface_id"]}: R and T must be >= 0 with R+T <= 1')
    return rho, tau


# ---------------------------------------------------------------- python comparator (direct light only)
def python_states(scene, grid: tuple[int, int] | None, overrides: dict | None = None) -> tuple:
    states = []
    for fx in scene.fixtures:
        upd = {}
        if grid:
            upd['emitter_grid'] = tuple(grid)
        fx2 = fx.model_copy(update=upd)
        for name, value in (overrides or {}).items():
            fx2 = _override(fx2, name, value)
        em, _ = build_emitters(fx2)
        states.append(SimpleNamespace(fixture_id=fx.fixture_id, emitters=em))
    return tuple(states)


def _override(fx, name: str, value: float):
    src = 'sensitivity/fit override'
    if name == 'flux_scale':
        return fx.model_copy(update=dict(photon_flux_umol_s=ScalarInput(value=fx.photon_flux_umol_s.value * value,
                                                                         status='scenario', source=src)))
    if name == 'half_angle_deg':
        dist = fx.distribution.model_copy(update=dict(half_intensity_half_angle_deg=ScalarInput(value=value, status='scenario', source=src)))
        return fx.model_copy(update=dict(distribution=dist))
    if name == 'lamp_dz_mm':
        c = list(fx.emitting_center_grow_mm.value); c[2] += value
        return fx.model_copy(update=dict(emitting_center_grow_mm=type(fx.emitting_center_grow_mm)(value=tuple(c), status='scenario', source=src)))
    raise StudyError(f'unknown override {name}')


def python_point_ppfd(states, point: dict, sensor: dict, n: int, occluders: Occluders | None = None,
                      transmittance: float | None = None) -> float:
    pts = footprint_samples(point, sensor, n)
    normal = unit_normal(point)
    rec = Receivers(points_grow_mm=pts, unit_normals=np.tile(normal, (len(pts), 1)), two_sided=False,
                    own_triangle=np.full(len(pts), -1))
    settings = LightSimulationSettings(triangle_subdivisions=1, ray_offset_mm=0.01, max_ray_triangle_pairs_per_chunk=2_000_000)
    res = direct_light(rec, states, occluders or Occluders.none(), transmittance, settings)
    return float(res.ppfd_umol_m2_s.sum(0).mean())


def opaque_occluders(triangles: list[np.ndarray]) -> Occluders:
    if not triangles:
        return Occluders.none()
    v = np.concatenate(triangles)
    return Occluders(vertices_grow_mm=v, is_leaf=np.zeros(len(v), bool), surface_code=np.arange(len(v)),
                     group=np.zeros(len(v), np.int64))


# ---------------------------------------------------------------- empty-rig calibration
def aggregate_points(readings: list[dict], role: str | None) -> tuple[dict, list[dict]]:
    """Mean of valid repeats per (case, point); every exclusion kept with its reason."""
    groups, excluded = defaultdict(list), []
    for r in readings:
        if role and r['split_role'] != role:
            continue
        key = (r['case_id'], r['point_id'])
        reason = None
        if r['disturbed'] == 'yes':
            reason = 'disturbed'
        elif r['all_fixtures_full_output'] != 'yes':
            reason = 'not_all_fixtures_full_output'
        elif r['ambient_condition'] == 'present_unsubtracted':
            reason = 'ambient_present_unsubtracted'
        elif r['sensor_moved_leaves'] == 'yes':
            reason = 'sensor_moved_leaves'
        value = number(r['ppfd_umol_m2_s'], 'ppfd', missing=True)
        if reason is None and value is None:
            reason = 'reading_unavailable'
        if reason:
            excluded.append(dict(case_id=key[0], point_id=key[1], repeat_id=r['repeat_id'], reason=reason))
            continue
        groups[key].append(value)
    return {k: dict(n=len(v), mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None)
            for k, v in groups.items()}, excluded


def fit_empty_rig(calibration_csv: Path, scene_path: Path, fit_settings: dict, points_csv: Path, sensors_csv: Path,
                  python_settings: dict, rig_csv: Path | None = None) -> dict:
    """Fit only explicitly enabled parameters (photon_flux_scale, half_intensity_half_angle_deg) on calibration points.

    Fixture geometry stays fixed. A reflective rig is refused for this direct-light backend. Insufficient points or a
    profile interval that reaches a bound give status insufficient_calibration; nothing is promoted to measured.
    """
    scene = load_scene(scene_path)
    enabled = list(fit_settings.get('enabled_parameters') or [])
    unknown = set(enabled) - {'photon_flux_scale', 'half_intensity_half_angle_deg'}
    if unknown:
        raise StudyError(f'unsupported fit parameters {sorted(unknown)}')
    readings = rows(calibration_csv, READING_COLUMNS)
    points = {(p['case_id'], p['point_id']): p for p in rows(points_csv, POINT_COLUMNS)}
    sensors = {s['sensor_id']: s for s in rows(sensors_csv, SENSOR_COLUMNS)}
    cal = [r for r in readings if r['split_role'] == 'calibration' and r['stage'] == 'empty_rig']
    agg, excluded = aggregate_points(cal, 'calibration')
    base = dict(schema='ppfd-lamp-fit-1', scene_sha256=file_hash(scene_path), calibration_source=str(calibration_csv),
                calibration_source_sha256=file_hash(calibration_csv), enabled_parameters=enabled,
                calibration_dataset_sha256=digest(sorted([[k[0], k[1], v['n'], v['mean']] for k, v in agg.items()])),
                calibration_points=[f'{k[0]}:{k[1]}' for k in sorted(agg)], excluded_repeats=excluded,
                backend='python direct light (tomato3d.light.raytrace), no reflections', fitted_parameters={},
                provenance='fitted (not measured)')
    rig_rows = [r for r in rows(rig_csv, RIG_SURFACE_COLUMNS) if r['role'] == 'rig'] if rig_csv and rig_csv.is_file() else []
    optics = [(number(r['reflectance'], 'reflectance', missing=True), number(r['transmittance'], 'transmittance', missing=True))
              for r in rig_rows]
    if any(a is None or b is None for a, b in optics):
        return dict(base, status='blocked_rig_optics_unavailable')
    if any(a > 0 or b > 0 for a, b in optics):
        return dict(base, status='blocked_reflective_rig_not_in_fit_backend')
    if not enabled:
        return dict(base, status='no_parameters_enabled')
    n_points, p = len(agg), len(enabled)
    if n_points < p + int(fit_settings.get('min_points_above_parameters') or 0) or n_points == 0:
        return dict(base, status='insufficient_calibration', reason=f'{n_points} calibration points for {p} parameters')
    grid = python_settings.get('emitter_grid'); n_samp = python_settings.get('sensor_samples')
    if not grid or not n_samp:
        raise StudyError('python.emitter_grid and python.sensor_samples are required (not defaulted)')
    keys = sorted(agg)
    y = np.array([agg[k]['mean'] for k in keys])

    def predict(half: float | None) -> np.ndarray:
        ov = {} if half is None else {'half_angle_deg': half}
        states = python_states(scene, grid, ov)
        return np.array([python_point_ppfd(states, points[k], sensors[points[k]['sensor_id']], n_samp) for k in keys])

    def solve(pred: np.ndarray) -> tuple[float, float]:
        if 'photon_flux_scale' in enabled:
            denom = float(pred @ pred)
            if denom <= 0:
                raise StudyError('all calibration predictions are zero; flux not identifiable')
            s = float(pred @ y / denom)
            lo, hi = fit_settings['bounds'].get('photon_flux_scale') or (None, None)
            if lo is None or hi is None:
                raise StudyError('bounds.photon_flux_scale is required')
            s = min(max(s, lo), hi)
        else:
            s = 1.0
        return s, float(((s * pred - y) ** 2).sum())

    result = dict(base)
    if 'half_intensity_half_angle_deg' in enabled:
        lo, hi = fit_settings['bounds'].get('half_intensity_half_angle_deg') or (None, None)
        step = fit_settings.get('half_angle_grid_step_deg')
        if lo is None or hi is None or not step:
            raise StudyError('bounds.half_intensity_half_angle_deg and half_angle_grid_step_deg are required')
        halves = np.arange(lo, hi + step / 2, step)
        prof = [(h, *solve(predict(float(h)))) for h in halves]
        best = min(prof, key=lambda t: t[2])
        dof = max(n_points - p, 1)
        sigma2 = best[2] / dof
        limit = best[2] + float(fit_settings.get('profile_chi2_95') or 3.841458820694124) * sigma2
        inside = [h for h, _, sse in prof if sse <= limit]
        interval = [float(min(inside)), float(max(inside))]
        at_bound = interval[0] <= lo + 1e-9 or interval[1] >= hi - 1e-9
        half, scale, sse = float(best[0]), best[1], best[2]
        result['profile'] = [dict(half_angle_deg=float(h), flux_scale=s, sse=e) for h, s, e in prof]
        result['fitted_parameters']['half_intensity_half_angle_deg'] = dict(
            value=half, status='fitted', bounds=[lo, hi], profile_interval_95=interval, reaches_bound=at_bound)
        pred = predict(half)
    else:
        half = None
        pred = predict(None)
        scale, sse = solve(pred)
        at_bound = False
    if 'photon_flux_scale' in enabled:
        b = fit_settings['bounds']['photon_flux_scale']
        result['fitted_parameters']['photon_flux_scale'] = dict(value=scale, status='fitted', bounds=list(b),
                                                                reaches_bound=scale <= b[0] or scale >= b[1])
        at_bound = at_bound or scale <= b[0] or scale >= b[1]
    resid = scale * pred - y
    result['residuals'] = [dict(case_id=k[0], point_id=k[1], n_repeats=agg[k]['n'], measured_mean=agg[k]['mean'],
                                fitted_prediction=float(scale * pv), residual=float(r)) for k, pv, r in zip(keys, pred, resid)]
    result['calibration_rmse_umol_m2_s'] = float(np.sqrt(np.mean(resid ** 2)))
    result['status'] = 'insufficient_calibration' if at_bound else 'fitted'
    if at_bound:
        result['reason'] = 'a parameter or its 95 % profile interval reaches its bound: not identified by these points'
    result['frozen_at'] = datetime.now().astimezone().isoformat()
    return result


def fitted_scene(scene_path: Path, fit: dict, out_path: Path) -> Path:
    """Scene copy with fitted values marked status 'fitted' and the fit manifest hash as source."""
    data = read_json(scene_path)
    src = f'fitted: lamp_fit sha256 {digest(fit)}'
    for fx in data['fixtures']:
        if 'photon_flux_scale' in fit['fitted_parameters']:
            fx['photon_flux_umol_s'] = dict(value=fx['photon_flux_umol_s']['value'] * fit['fitted_parameters']['photon_flux_scale']['value'],
                                            status='fitted', source=src)
        if 'half_intensity_half_angle_deg' in fit['fitted_parameters']:
            fx['distribution']['half_intensity_half_angle_deg'] = dict(
                value=fit['fitted_parameters']['half_intensity_half_angle_deg']['value'], status='fitted', source=src)
    write_json(out_path, data)
    load_scene(out_path)
    return out_path


# ---------------------------------------------------------------- RGG generation
RGG_HEADER = """// generated by tools.ppfd_validation; do not edit. Grow frame, metres, +Z up.
import de.grogra.xl.util.FloatList;
import de.grogra.imp3d.shading.SideSwitchShader;
module LeafSurface(String surfaceId) extends MeshNode;
module Probe(String sensorId) extends MeshNode;
module RigSurface(String surfaceId) extends MeshNode;
module Lamp extends LightNode;
LightModel lm = new LightModel(@RAYS@, @DEPTH@);
"""
PHONG = """static Phong @NAME@ = new Phong();
static {
@NAME@.setDiffuse(new Graytone(@R@));
@NAME@.setDiffuseTransparency(new Graytone(@T@));
@NAME@.setSpecular(new Graytone(0));
@NAME@.setTransparency(new Graytone(0));
@NAME@.setAmbient(new Graytone(0));
@NAME@.setEmissive(new Graytone(0));
@NAME@.setInterpolatedTransparency(false);
}
"""
MEASURE = """
public void measure() {
    lm.setSeed(@SEED@L);
    lm.compute();
    for (LeafSurface r : (* LeafSurface *)) {
        println("LEAF\\t" + r.surfaceId + "\\t" + lm.getReceivedPower(r).integrate() + "\\t" + lm.getAbsorbedPower(r).integrate());
    }
    for (Probe s : (* Probe *)) {
        println("PROBE\\t" + s.sensorId + "\\t" + lm.getAbsorbedPower(s).integrate());
    }
    for (RigSurface s : (* RigSurface *)) {
        println("RIG\\t" + s.surfaceId + "\\t" + lm.getAbsorbedPower(s).integrate());
    }
    println("SANITY_DONE");
}
"""
PROBE_SHADER = 'new SideSwitchShader(BLACK, new RGBAShader(0,0,0,0))'


def make_job_rgg(lamp_nodes: str, rays: int, depth: int, seed: int, rig: list[dict] = (), leaves=(),
                 leaf_optics: tuple[float, float, float] | None = None, probes=()) -> str:
    """One GroIMP job: lamps, rig/occluder surfaces, optional leaf triangles, one-sided black probes."""
    head = [RGG_HEADER.replace('@RAYS@', str(int(rays))).replace('@DEPTH@', str(int(depth)))]
    nodes = [lamp_nodes]
    for k, r in enumerate(rig):
        rho, tau = surface_optics(r)
        if rho == 0 and tau == 0:
            shader = 'BLACK'
        else:
            shader = f'rig{k}'
            head.append(PHONG.replace('@NAME@', shader).replace('@R@', _f(rho)).replace('@T@', _f(tau)))
        for i, tri in enumerate(rig_surface_triangles(r)):
            nodes.append(mesh_node('RigSurface', f'{r["surface_id"]}:{i}', tri, shader))
    if leaves:
        if leaf_optics is None:
            raise StudyError('leaf optics are required for leaf triangles')
        a, rho, tau = leaf_optics
        if min(a, rho, tau) < 0 or not np.isclose(a + rho + tau, 1, atol=1e-6):
            raise StudyError('native leaf optics need explicit A+R+T = 1')
        head.append(PHONG.replace('@NAME@', 'leafOptics').replace('@R@', _f(rho)).replace('@T@', _f(tau)))
        for meta, tri in leaves:
            nodes.append(mesh_node('LeafSurface', meta['surface_id'], tri, 'leafOptics'))
    for key, tri_list in probes:
        for i, tri in enumerate(tri_list):
            nodes.append(mesh_node('Probe', f'{key}:{i}', tri, PROBE_SHADER))
    body = '\n'.join(nodes)
    return ''.join(head) + '\nprotected void init()\n[\nAxiom ==> ' + body + ';\n]\n' + MEASURE.replace('@SEED@', str(int(seed)))


def parse_rig(path: Path) -> dict:
    text = path.read_text(encoding='utf-8', errors='replace')
    out = {}
    for k, v in re.findall(r'RIG\t([^\s]+)\t([^\s]+)', text):
        out[k] = float(v)
    return out


def probe_values(probes: dict) -> dict:
    """Sum the per-triangle probe outputs '<key>:<i>' into one photon rate per probe key."""
    out = defaultdict(float)
    for k, v in probes.items():
        out[k.rsplit(':', 1)[0]] += v
    return dict(out)


# ---------------------------------------------------------------- prepare / collect
def _leaf_triangles(case: dict, base: Path, scene) -> tuple[list, list[str], dict]:
    """Reviewed leaf triangles through the existing groimp_density surface gate; blockers are never bypassed."""
    from tomato3d.light.geometry import load_geometry, place_in_grow_frame, triangle_areas_mm2
    gate_case = dict(geometry=case['geometry'], surface_review=case['surface_review'],
                     geometry_variant_id=case['geometry_variant_id'], snapshot_date=case['geometry_snapshot_date'],
                     data_kind=case.get('data_kind', 'observed_geometry'),
                     instances=[dict(plant_id=case['plant_id'], source_plant_id=case['plant_id'],
                                     source_session_id=case['session_id'])])
    blockers, evidence = surface_gate(gate_case, base)
    review = evidence.get('review') or {}
    completeness = review.get('geometry_completeness', 'unknown')
    evidence['geometry_completeness'] = completeness
    if completeness != 'whole_plant_reviewed':
        evidence['limitation'] = ('partial or unknown geometry: an isolated, limited test only; missing organs on light '
                                  'paths are not explained (completeness is never inferred from triangle count)')
    if blockers:
        return [], blockers, evidence
    placements = {p.plant_id: p for p in scene.plants}
    if case['plant_id'] not in placements:
        return [], ['no_plant_placement_in_scene'], evidence
    surfaces = load_geometry([base / case['geometry']])[case['geometry_variant_id']]
    grow = place_in_grow_frame(surfaces, placements)
    areas = triangle_areas_mm2(grow)
    leaves = []
    for i in range(surfaces.n_triangles):
        meta = dict(surface_id=f't{i:07d}', plant_id=surfaces.plant_ids[i], session_id=surfaces.session_ids[i],
                    organ_id=surfaces.organ_ids[i], leaflet_id=surfaces.leaflet_ids[i],
                    triangle_id=surfaces.triangle_ids[i], area_mm2=float(areas[i]))
        leaves.append((meta, grow[i]))
    return leaves, [], evidence


def absolute_blockers(scene, case: dict, rig: list[dict], sensors: dict, points: list[dict], lamp_rows: list[dict],
                      leaves: bool) -> list[str]:
    out = []
    for fx in scene.fixtures:
        ok, why = adapter_support(fx)
        if not ok:
            out.append(f'{fx.fixture_id}: {why}')
        for name in ('emitting_center_grow_mm', 'aim_tilt_deg', 'aim_azimuth_deg', 'long_axis_azimuth_deg',
                     'emitting_length_mm', 'emitting_width_mm', 'photon_flux_umol_s'):
            st = getattr(fx, name).status
            if st not in ('measured', 'fitted'):
                out.append(f'{fx.fixture_id}.{name} is {st}')
        st = fx.distribution.half_intensity_half_angle_deg.status
        if st not in ('measured', 'fitted'):
            out.append(f'{fx.fixture_id}.distribution.half_intensity_half_angle_deg is {st}')
        row = next((r for r in lamp_rows if r['fixture_id'] == fx.fixture_id), None)
        if row is None or row['is_full_output'] != 'yes':
            out.append(f'{fx.fixture_id}: full-output state not confirmed in lamp_conditions.csv')
    for r in rig:
        if r['role'] not in SURFACE_ROLES:
            out.append(f'rig surface {r["surface_id"]}: role must be one of {SURFACE_ROLES}')
        if r['role'] == 'display_only':
            continue
        for k in ('geometry_status', 'optics_status'):
            if r[k] != 'measured':
                out.append(f'surface {r["surface_id"]}.{k} is {r[k] or UNAVAILABLE}')
    for p in points:
        if p['position_status'] != 'measured':
            out.append(f'point {p["point_id"]} position is {p["position_status"] or UNAVAILABLE}')
        s = sensors.get(p['sensor_id'])
        if s is None or s['dimensions_status'] != 'measured':
            out.append(f'sensor {p["sensor_id"]} dimensions not measured')
    if leaves:
        for name in ('absorptance', 'transmittance', 'reflectance'):
            st = getattr(scene.leaf_optics, name).status
            if st != 'measured':
                out.append(f'leaf_optics.{name} is {st}')
        for pl in scene.plants:
            if pl.plant_id == case.get('plant_id'):
                for k in ('board_origin_grow_mm', 'board_yaw_deg'):
                    if getattr(pl, k).status != 'measured':
                        out.append(f'{pl.plant_id}.{k} is {getattr(pl, k).status}')
    return sorted(set(out))


def occlusion_class(states, point: dict, sensor: dict, plates: list[np.ndarray], n: int, tol: float | None) -> tuple[str, float | None]:
    if not plates:
        return 'no_occluder', None
    free = python_point_ppfd(states, point, sensor, n)
    shaded = python_point_ppfd(states, point, sensor, n, opaque_occluders(plates))
    if free <= 0:
        return 'outside_beam', None
    frac = shaded / free
    if tol is None:
        return 'unclassified_tolerance_unset', frac
    return ('illuminated' if frac >= 1 - tol else 'shadow_interior' if frac <= tol else 'shadow_edge'), frac


def prepare(study_path: Path, out: Path, scene_override: Path | None = None) -> dict:
    """Native jobs per case (groimp_density.execute-compatible jobs.json) + python comparator values."""
    study = load_study(study_path)
    base = study_path.parent
    scene_path = scene_override or _path(base, study.get('scene'))
    if scene_path is None or not scene_path.is_file():
        raise StudyError('no LightScene: native jobs cannot be prepared (absolute and relative maps unavailable)')
    scene = load_scene(scene_path)
    solver = study.get('solver') or {}
    for k in ('rays', 'max_depth', 'seed'):
        if not isinstance(solver.get(k), int):
            raise StudyError(f'solver.{k} must be an explicit integer')
    py = study.get('python') or {}
    if not py.get('emitter_grid') or not py.get('sensor_samples'):
        raise StudyError('python.emitter_grid and python.sensor_samples must be explicit')
    adapter = lamp_adapter(scene_path, scene)
    if adapter['unsupported_fixtures']:
        raise StudyError(f'unsupported fixtures for native prediction: {adapter["unsupported_fixtures"]}')
    rig_all = {r['surface_id']: r for r in _csv(base, study.get('rig_surfaces'), RIG_SURFACE_COLUMNS)}
    sensors = {s['sensor_id']: s for s in _csv(base, study.get('sensors'), SENSOR_COLUMNS)}
    points = _csv(base, study.get('points'), POINT_COLUMNS)
    lamp_rows = _csv(base, study.get('lamp_conditions'), LAMP_CONDITION_COLUMNS)
    states = python_states(scene, py['emitter_grid'])
    folder = output_dir(out)
    write_json(folder / 'lamp_adapter.json', adapter)
    shutil.copyfile(scene_path, folder / 'scene.json')
    manifest = dict(schema_version='ppfd-validation-jobs-1', status='prepared', cases=[], study=str(study_path.resolve()),
                    study_sha256=file_hash(study_path), scene_sha256=file_hash(scene_path),
                    lamp_model_sha256=file_hash(folder / 'lamp_adapter.json'), solver=solver, python=py,
                    validated_metric=False, suitable_for_ppfd=False)
    rig_shared = [r for r in rig_all.values() if r['role'] == 'rig']
    reflective = any(surface_optics(r) != (0.0, 0.0) for r in rig_shared)
    cases = list(study.get('cases') or [])
    for c in cases:
        if c.get('stage') == 'plant':  # same positions without the plant: the plant geometry's contribution
            cases.append(dict(c, case_id=c['case_id'] + '__empty_twin', stage='plant_empty_twin', twin_of=c['case_id']))
    for c in cases:
        cid = c['case_id']
        src_case = c.get('twin_of', cid)
        pts = [p for p in points if p['case_id'] == src_case]
        item = dict(case_id=cid, stage=c['stage'], status='prepared', blockers=[], jobs=[], model_hashes={},
                    n_points=len(pts))
        manifest['cases'].append(item)
        if not pts:
            item.update(status='blocked', blockers=['no_measurement_points'])
            continue
        plates = [rig_surface_triangles(rig_all[s]) for s in c.get('occluder_surface_ids') or [] if s in rig_all]
        missing_plates = [s for s in c.get('occluder_surface_ids') or [] if s not in rig_all]
        if missing_plates:
            item['blockers'].append(f'occluder surfaces not declared: {missing_plates}')
        not_physical = [s for s in c.get('occluder_surface_ids') or [] if s in rig_all and rig_all[s]['role'] != 'physical_occluder']
        if not_physical:
            item['blockers'].append(f'occluder ids are not physical_occluder surfaces (display/rig): {not_physical}')
        leaves, leaf_evidence = [], {}
        if c['stage'] == 'plant':
            leaves, gate, leaf_evidence = _leaf_triangles(c, base, scene)
            item['blockers'] += gate
            item['surface_evidence'] = {k: v for k, v in leaf_evidence.items() if k != 'review'}
        if item['blockers']:
            item['status'] = 'blocked'
            continue
        rig = rig_shared + [rig_all[s] for s in c.get('occluder_surface_ids') or []]
        lo = scene.leaf_optics
        optics = None
        if leaves:
            vals = [lo.absorptance.value, lo.reflectance.value, lo.transmittance.value]
            if any(v is None for v in vals):
                item.update(status='blocked', blockers=['leaf_optics_unavailable'])
                continue
            optics = tuple(vals)
        meshes = {}
        for p in pts:
            s = sensors.get(p['sensor_id'])
            if s is None:
                item['blockers'].append(f'point {p["point_id"]}: sensor {p["sensor_id"]!r} not declared')
                continue
            meshes[p['point_id']] = sensing_mesh(p, s, py.get('disc_segments'))
        if item['blockers']:
            item['status'] = 'blocked'
            continue
        # python comparator and occlusion classes (direct light; leaves not included: different transport model)
        py_rows = []
        for p in pts:
            s = sensors[p['sensor_id']]
            occ = opaque_occluders(plates)
            val = python_point_ppfd(states, p, s, py['sensor_samples'], occ)
            cls, frac = occlusion_class(states, p, s, plates, py['sensor_samples'], py.get('shadow_class_tolerance'))
            py_rows.append(dict(point_id=p['point_id'], python_direct_ppfd=val, occlusion_class=cls, unshaded_fraction=frac))
        # batch probes only when they provably cannot shade one another and nothing reflects
        independent = not leaves and not reflective
        if independent:
            for p in pts:
                others = [meshes[q['point_id']][0] for q in pts if q is not p]
                s = sensors[p['sensor_id']]
                a = python_point_ppfd(states, p, s, py['sensor_samples'], opaque_occluders(plates))
                b = python_point_ppfd(states, p, s, py['sensor_samples'], opaque_occluders(plates + others))
                if not np.isclose(a, b, rtol=1e-9, atol=0):
                    independent = False
                    break
        groups = [pts] if independent else [[p] for p in pts]
        cf = folder / cid
        cf.mkdir()
        specs = []
        if leaves:
            job = 'leaves'
            (cf / job).mkdir()
            (cf / job / 'Model.rgg').write_text(make_job_rgg(adapter['initialization_rgg'], solver['rays'], solver['max_depth'],
                                                             solver['seed'], rig, leaves, optics), encoding='utf-8')
            item['jobs'].append(job)
            specs.append(dict(job=job, probes=[]))
            write_rows(cf / 'triangles.csv', [m for m, _ in leaves], ('surface_id', 'plant_id', 'session_id', 'organ_id',
                                                                 'leaflet_id', 'triangle_id', 'area_mm2'))
        for g, group in enumerate(groups):
            job = 'probes' if independent else f'probe_{group[0]["point_id"]}'
            (cf / job).mkdir()
            probes = [(p['point_id'], meshes[p['point_id']][0]) for p in group]
            (cf / job / 'Model.rgg').write_text(make_job_rgg(adapter['initialization_rgg'], solver['rays'], solver['max_depth'],
                                                             solver['seed'], rig, leaves, optics, probes), encoding='utf-8')
            item['jobs'].append(job)
            specs.append(dict(job=job, probes=[dict(point_id=p['point_id'], area_mm2=meshes[p['point_id']][1]) for p in group]))
        item['model_hashes'] = {j: file_hash(cf / j / 'Model.rgg') for j in item['jobs']}
        item['absolute_blockers'] = absolute_blockers(scene, c, rig, sensors, pts, lamp_rows, bool(leaves))
        item['batched_probes'] = independent
        item['n_leaf_triangles'] = len(leaves)
        write_json(cf / 'job_specs.json', dict(case=c, specs=specs, python=py_rows, rig_surface_ids=[r['surface_id'] for r in rig],
                                               points=pts, leaf_optics=optics))
    manifest['status'] = 'prepared' if all(c['status'] == 'prepared' for c in manifest['cases']) else 'blocked_or_partial'
    manifest['artifact_hashes'] = {str(p.relative_to(folder)).replace('\\', '/'): file_hash(p)
                                   for p in folder.rglob('*') if p.is_file()}
    write_json(folder / 'jobs.json', manifest)
    return manifest


def collect(prepared: Path, execution: Path, out: Path | None = None) -> dict:
    """Prediction rows per case/point from completed native jobs; failed or blocked cases stay visible."""
    jobs = read_json(prepared / 'jobs.json')
    verify_prepared(prepared, jobs)
    run = read_json(execution / 'execution.json')
    if file_hash(prepared / 'jobs.json') != run['jobs_sha256']:
        raise StudyError('prepared manifest changed since execution (stale)')
    done = {(r['case_id'], r['job']): r for r in run['runs'] if r.get('job')}
    predictions, leaf_rows, failures = [], [], []
    twin_values = {}
    ordered = sorted(jobs['cases'], key=lambda c: 0 if c['stage'] == 'plant_empty_twin' else 1)
    for case in ordered:
        cid = case['case_id']
        if case['status'] != 'prepared':
            failures.append(dict(case_id=cid, reason=';'.join(case['blockers'])))
            continue
        spec = read_json(prepared / cid / 'job_specs.json')
        pyv = {r['point_id']: r for r in spec['python']}
        pts = {p['point_id']: p for p in spec['points']}
        for js in spec['specs']:
            rec = done.get((cid, js['job']))
            ok = rec is not None and rec['status'] == 'completed'
            if not ok:
                failures.append(dict(case_id=cid, job=js['job'], reason=(rec or {}).get('reason', 'not executed')))
            log = execution / cid / js['job'] / 'groimp.log'
            if ok and file_hash(log) != rec.get('log_sha256'):
                raise StudyError(f'execution log changed: {log}')
            leaves, probes = parse_log(log) if ok else ({}, {})
            if js['job'] == 'leaves' and ok:
                tri = {t['surface_id']: t for t in rows(prepared / cid / 'triangles.csv')}
                if set(leaves) != set(tri):
                    raise StudyError('missing/unexpected leaf outputs')
                for sid, (received, absorbed) in leaves.items():
                    area_m2 = float(tri[sid]['area_mm2']) / MM_PER_M ** 2
                    leaf_rows.append(dict(case_id=cid, **tri[sid],
                                          received_both_faces_umol_s=received, absorbed_umol_s=absorbed,
                                          received_both_faces_per_one_sided_area_umol_m2_s=received / area_m2,
                                          absorbed_per_one_sided_area_umol_m2_s=absorbed / area_m2))
            vals = probe_values(probes)
            for pr in js['probes']:
                p = pts[pr['point_id']]
                sim = vals.get(pr['point_id'])
                if ok and sim is None:
                    raise StudyError(f'probe {pr["point_id"]} missing in {log}')
                sim_ppfd = sim / (pr['area_mm2'] / MM_PER_M ** 2) if sim is not None else None
                if case['stage'] == 'plant_empty_twin':
                    twin_values[(spec['case']['twin_of'], pr['point_id'])] = sim_ppfd
                absolute = sim_ppfd if (sim_ppfd is not None and not case['absolute_blockers']) else None
                twin = twin_values.get((cid, pr['point_id']))
                predictions.append(dict(
                    stage=case['stage'], case_id=cid, point_id=pr['point_id'], sensor_id=p['sensor_id'],
                    height_band=p['height_band'], split_role=p['split_role'], job=js['job'],
                    predicted_ppfd_umol_m2_s=absolute, simulated_ppfd_declared_inputs_umol_m2_s=sim_ppfd,
                    python_direct_ppfd_declared_inputs_umol_m2_s=pyv[pr['point_id']]['python_direct_ppfd'],
                    empty_twin_simulated_ppfd_umol_m2_s=twin,
                    plant_contribution_umol_m2_s=(sim_ppfd - twin) if sim_ppfd is not None and twin is not None else None,
                    occlusion_class=pyv[pr['point_id']]['occlusion_class'],
                    quantity='front-face incident PPFD on the declared sensing area (one-sided)',
                    backend=f'GroIMP CPU LightModel rays={jobs["solver"]["rays"]} depth={jobs["solver"]["max_depth"]} '
                            f'seed={jobs["solver"]["seed"]}', lamp_model_sha256=jobs['lamp_model_sha256'],
                    absolute_status='available' if absolute is not None else 'unavailable',
                    status='computed' if ok else 'failed',
                    reason='; '.join(case['absolute_blockers'])[:2000] if absolute is None else ''))
    result = dict(predictions=predictions, leaf_rows=leaf_rows, failures=failures)
    if out is not None:
        folder = out if out.exists() else output_dir(out)
        write_rows(folder / 'sensor_predictions.csv', predictions, PREDICTION_COLUMNS)
        if leaf_rows:
            write_rows(folder / 'leaf_triangle_light.csv', leaf_rows, tuple(leaf_rows[0]))
        write_json(folder / 'collect.json', dict(failures=failures, n_predictions=len(predictions),
                                                 n_leaf_rows=len(leaf_rows), validated_metric=False))
    return result


# ---------------------------------------------------------------- comparison
def _metrics(errors: list[float], measured: list[float]) -> dict:
    if not errors:
        return dict(n_points=0, mae=None, rmse=None, bias=None, mean_measured=None, nrmse=None)
    e = np.array(errors)
    mean = float(np.mean(measured))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return dict(n_points=len(errors), mae=float(np.abs(e).mean()), rmse=rmse, bias=float(e.mean()), mean_measured=mean,
                nrmse=rmse / mean if mean > 0 else None)


def make_split_manifest(points: list[dict], lamp_fit: dict | None) -> dict:
    roles = {f'{p["case_id"]}:{p["point_id"]}': p['split_role'] for p in points}
    body = dict(schema='ppfd-split-1', roles=roles,
                frozen_lamp_fit_sha256=digest(lamp_fit) if lamp_fit else None,
                calibration_dataset_sha256=(lamp_fit or {}).get('calibration_dataset_sha256'))
    return dict(body, split_sha256=digest(body))


def compare_ppfd(predictions_csv: Path, readings_csv: Path, split_manifest: Path, settings: dict | None = None) -> dict:
    """Independent validation only: calibration rows never enter; points weigh equally whatever their repeat count."""
    settings = settings or {}
    split = read_json(split_manifest)
    body = {k: v for k, v in split.items() if k != 'split_sha256'}
    if digest(body) != split.get('split_sha256'):
        raise StudyError('split manifest hash mismatch (edited after freezing)')
    preds = rows(predictions_csv, ('case_id', 'point_id', 'predicted_ppfd_umol_m2_s'))
    readings = rows(readings_csv, READING_COLUMNS)
    for r in readings:
        key = f'{r["case_id"]}:{r["point_id"]}'
        if key not in split['roles']:
            raise StudyError(f'reading {key} is not in the frozen split')
        if split['roles'][key] != r['split_role']:
            raise StudyError(f'reading {key}: split_role differs from the frozen split')
    pmap = {}
    for p in preds:
        key = (p['case_id'], p['point_id'])
        if key in pmap:
            raise StudyError(f'duplicate prediction {key}')
        pmap[key] = p
    agg, excluded = aggregate_points(readings, 'validation')
    n_cal_rows = sum(r['split_role'] == 'calibration' for r in readings)
    per_point, unavailable = [], []
    unit_of = {}
    for r in readings:
        unit_of[(r['case_id'], r['point_id'])] = (r['stage'], r['plant_id'] or '-', r['session_id'] or '-')
    for key, a in sorted(agg.items()):
        p = pmap.get(key)
        pred = number(p['predicted_ppfd_umol_m2_s'], 'pred', missing=True) if p else None
        if pred is None:
            unavailable.append(dict(case_id=key[0], point_id=key[1], reason='no absolute prediction' if p else 'no prediction row'))
            continue
        per_point.append(dict(case_id=key[0], point_id=key[1], stage=unit_of[key][0], height_band=(p or {}).get('height_band', ''),
                              unit=f'{unit_of[key][1]}|{unit_of[key][2]}', n_repeats=a['n'], measured_mean=a['mean'],
                              repeat_sd=a['sd'], repeat_cv=(a['sd'] / a['mean']) if a['sd'] is not None and a['mean'] > 0 else None,
                              predicted=pred, error=pred - a['mean']))
    groups = []
    keys = [('all', lambda r: True)]
    for st in sorted({r['stage'] for r in per_point}):
        keys.append((f'stage={st}', lambda r, st=st: r['stage'] == st))
        for band in sorted({r['height_band'] for r in per_point if r['stage'] == st}):
            keys.append((f'stage={st};height={band}', lambda r, st=st, band=band: r['stage'] == st and r['height_band'] == band))
    acc = settings.get('acceptance') or {}
    inf = settings.get('inference') or {}
    for name, sel in keys:
        sub = [r for r in per_point if sel(r)]
        m = _metrics([r['error'] for r in sub], [r['measured_mean'] for r in sub])
        units = sorted({r['unit'] for r in sub})
        m.update(subset=name, n_independent_units=len(units), n_repeat_rows=sum(r['n_repeats'] for r in sub))
        m['ci95_rmse'] = _cluster_ci(sub, inf)
        if acc.get('nrmse_max') is None and acc.get('abs_bias_max_umol_m2_s') is None:
            m['acceptance'] = 'no_threshold_set'
        elif m['nrmse'] is None:
            m['acceptance'] = 'unavailable'
        else:
            okn = acc.get('nrmse_max') is None or m['nrmse'] <= acc['nrmse_max']
            okb = acc.get('abs_bias_max_umol_m2_s') is None or abs(m['bias']) <= acc['abs_bias_max_umol_m2_s']
            m['acceptance'] = 'meets_predeclared_threshold' if okn and okb else 'fails_predeclared_threshold'
        groups.append(m)
    return dict(status='evaluated' if per_point else 'unavailable', per_point=per_point, metrics=groups,
                excluded_repeats=excluded, unavailable_points=unavailable, n_calibration_rows_excluded=n_cal_rows,
                n_validation_points=len(agg), split_sha256=split['split_sha256'],
                note='error = predicted - measured; primary metrics on point means; repeats are not independent units')


def _cluster_ci(sub: list[dict], inf: dict) -> object:
    need = inf.get('min_independent_units_for_ci')
    units = sorted({r['unit'] for r in sub})
    if need is None:
        return 'unavailable: min_independent_units_for_ci not declared'
    if len(units) < need:
        return f'unavailable: {len(units)} independent units < {need}'
    n_boot, seed = inf.get('bootstrap_resamples'), inf.get('bootstrap_seed')
    if not n_boot or seed is None:
        return 'unavailable: bootstrap_resamples/bootstrap_seed not declared'
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in sub:
        by[r['unit']].append(r['error'])
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(units), len(units), replace=True)
        e = np.concatenate([by[units[i]] for i in pick])
        stats.append(float(np.sqrt(np.mean(e ** 2))))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


# ---------------------------------------------------------------- sensitivity (python direct model)
def sensitivity(study: dict, base: Path, scene_path: Path) -> list[dict]:
    cfg = study.get('sensitivity') or {}
    py = study.get('python') or {}
    scene = load_scene(scene_path)
    sensors = {s['sensor_id']: s for s in _csv(base, study.get('sensors'), SENSOR_COLUMNS)}
    points = _csv(base, study.get('points'), POINT_COLUMNS)
    rig = {r['surface_id']: r for r in _csv(base, study.get('rig_surfaces'), RIG_SURFACE_COLUMNS)}
    out = []
    if not py.get('emitter_grid') or not py.get('sensor_samples'):
        return [dict(parameter='all', status='unavailable: python settings undecided')]
    base_states = python_states(scene, py['emitter_grid'])
    case_plates = {c['case_id']: [rig_surface_triangles(rig[s]) for s in c.get('occluder_surface_ids') or []
                                  if s in rig and rig[s]['role'] == 'physical_occluder']
                   for c in study.get('cases') or []}
    for p in points:
        s = sensors.get(p['sensor_id'])
        if s is None:
            continue
        occ = opaque_occluders(case_plates.get(p['case_id'], []))
        ref = python_point_ppfd(base_states, p, s, py['sensor_samples'], occ)
        trials = []
        for d in cfg.get('lamp_height_mm') or []:
            trials.append(('lamp_height_mm', d, python_states(scene, py['emitter_grid'], {'lamp_dz_mm': d}), p))
        for d in cfg.get('half_angle_deg') or []:
            fx_half = scene.fixtures[0].distribution.half_intensity_half_angle_deg.value
            if fx_half is not None:
                trials.append(('half_angle_deg', d, python_states(scene, py['emitter_grid'], {'half_angle_deg': fx_half + d}), p))
        for d in cfg.get('flux_rel') or []:
            trials.append(('flux_rel', d, python_states(scene, py['emitter_grid'], {'flux_scale': 1 + d}), p))
        for d in cfg.get('point_offset_mm') or []:
            q = dict(p, z_mm=str(float(p['z_mm']) + d))
            trials.append(('point_dz_mm', d, base_states, q))
        for d in cfg.get('sensor_tilt_deg') or []:
            n = unit_normal(p); t = np.cross(n, [0, 0, 1] if abs(n[2]) < .9 else [0, 1, 0]); t /= np.linalg.norm(t)
            a = math.radians(d); n2 = n * math.cos(a) + t * math.sin(a)
            q = dict(p, normal_x=str(n2[0]), normal_y=str(n2[1]), normal_z=str(n2[2]))
            trials.append(('sensor_tilt_deg', d, base_states, q))
        for name, d, st, q in trials:
            v = python_point_ppfd(st, q, s, py['sensor_samples'], occ)
            out.append(dict(case_id=p['case_id'], point_id=p['point_id'], parameter=name, delta=d, reference_ppfd=ref,
                            perturbed_ppfd=v, relative_change=(v - ref) / ref if ref > 0 else None,
                            backend='python direct (no reflections)', status='computed'))
    if not out:
        out.append(dict(parameter='all', status='unavailable: no sensitivity deltas configured or no points'))
    return out


# ---------------------------------------------------------------- convergence (bounded native runs)
def convergence(prepared: Path, case_id: str, out: Path, rays_list: list[int], seeds: list[int], max_jobs: int | None,
                java: str = 'java', core: Path = CORE) -> dict:
    jobs = read_json(prepared / 'jobs.json'); verify_prepared(prepared, jobs)
    case = next((c for c in jobs['cases'] if c['case_id'] == case_id), None)
    if case is None or case['status'] != 'prepared':
        raise StudyError(f'case {case_id} not prepared')
    spec = read_json(prepared / case_id / 'job_specs.json')
    probe_jobs = [js for js in spec['specs'] if js['probes']]
    n = len(probe_jobs) * len(rays_list) * len(seeds)
    if max_jobs is None:
        raise StudyError('solver.max_local_native_jobs must be declared before a local sweep')
    if n > max_jobs:
        raise StudyError(f'{n} native jobs exceed max_local_native_jobs={max_jobs}; prepare a remote job instead')
    folder = output_dir(out)
    records = []
    for js in probe_jobs:
        text = (prepared / case_id / js['job'] / 'Model.rgg').read_text(encoding='utf-8')
        for rays in rays_list:
            for seed in seeds:
                t = re.sub(r'new LightModel\(\d+, (\d+)\)', f'new LightModel({int(rays)}, \\1)', text)
                t = re.sub(r'lm\.setSeed\(-?\d+L\)', f'lm.setSeed({int(seed)}L)', t)
                d = folder / f'{js["job"]}_r{rays}_s{seed}'
                d.mkdir()
                (d / 'Model.rgg').write_text(t, encoding='utf-8')
                started = time.monotonic()
                try:
                    run_groimp(d, core, java)
                except (StudyError, OSError):
                    pass
                try:
                    _, probes = parse_log(d / 'groimp.log')
                    vals = probe_values(probes)
                    for pr in js['probes']:
                        records.append(dict(job=js['job'], rays=rays, seed=seed, point_id=pr['point_id'],
                                            ppfd=vals[pr['point_id']] / (pr['area_mm2'] / MM_PER_M ** 2),
                                            seconds=round(time.monotonic() - started, 1), status='completed'))
                except (StudyError, OSError, KeyError) as exc:
                    records.append(dict(job=js['job'], rays=rays, seed=seed, point_id=None, ppfd=None,
                                        seconds=round(time.monotonic() - started, 1), status=f'failed: {exc}'))
    summary = []
    for pid in sorted({r['point_id'] for r in records if r['point_id']}):
        for rays in rays_list:
            vals = [r['ppfd'] for r in records if r['point_id'] == pid and r['rays'] == rays and r['ppfd'] is not None]
            distinct = len(set(vals)) > 1
            summary.append(dict(point_id=pid, rays=rays, n_seeds=len(vals), mean=float(np.mean(vals)) if vals else None,
                                seed_cv=float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) > 0 else None,
                                seeds_give_distinct_results=distinct))
    write_rows(folder / 'convergence_runs.csv', records, ('job', 'rays', 'seed', 'point_id', 'ppfd', 'seconds', 'status'))
    write_rows(folder / 'convergence_summary.csv', summary, ('point_id', 'rays', 'n_seeds', 'mean', 'seed_cv',
                                                             'seeds_give_distinct_results'))
    result = dict(case_id=case_id, n_jobs=n, summary=summary,
                  independent_streams=('verified: different seeds give different results'
                                       if any(s['seeds_give_distinct_results'] for s in summary) else
                                       'not verified: seeds produced identical results; repeated runs are not MC samples'),
                  note='numerical convergence only; never an experimental accuracy criterion')
    write_json(folder / 'convergence.json', result)
    return result


# ---------------------------------------------------------------- report
def _color(v: float | None, lo: float, hi: float) -> str:
    if v is None:
        return '#9aa0a6'
    f = 0 if hi <= lo else min(max((v - lo) / (hi - lo), 0), 1)
    return f'rgb({int(40 + 215 * f)},{int(90 + 120 * (1 - abs(f - .5) * 2))},{int(255 * (1 - f))})'


def report_html(predictions: list[dict], comparison: dict | None, readiness: dict, points: list[dict], title: str) -> str:
    """Point maps with one shared colour scale (no interpolation), predicted-vs-measured and residual plots."""
    where = {(p['case_id'], p['point_id']): p for p in points}
    meas = {(r['case_id'], r['point_id']): r['measured_mean'] for r in (comparison or {}).get('per_point', [])}
    vals = [v for v in [r['simulated_ppfd_declared_inputs_umol_m2_s'] for r in predictions] + list(meas.values())
            if isinstance(v, (int, float))]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    parts = [f'<!doctype html><meta charset="utf-8"><title>{escape(title)}</title>',
             '<style>body{font:14px system-ui;margin:16px;max-width:1100px}svg{background:#fafafa;border:1px solid #ddd}'
             'table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px;font-size:12px}</style>',
             f'<h1>{escape(title)}</h1>',
             '<p>Diagnostic. validated_metric=false, suitable_for_ppfd=false. Maps show sampled points only '
             '(no interpolation; maps add no evaluation samples). Shared colour scale '
             f'{lo:.4g} - {hi:.4g} umol m-2 s-1 for simulated (declared inputs) and measured values.</p>',
             f'<h2>Readiness</h2><pre>{escape(json.dumps(readiness, indent=1)[:6000])}</pre>']
    for cid in sorted({r['case_id'] for r in predictions}):
        sub = [r for r in predictions if r['case_id'] == cid]
        xy = [(float(where[(cid.split('__')[0], r['point_id'])]['x_mm']), float(where[(cid.split('__')[0], r['point_id'])]['y_mm']))
              for r in sub]
        if not xy:
            continue
        x0, y0 = min(a for a, _ in xy) - 20, min(b for _, b in xy) - 20
        w, h = max(max(a for a, _ in xy) - x0 + 20, 1), max(max(b for _, b in xy) - y0 + 20, 1)
        dots = []
        for (x, y), r in zip(xy, sub):
            sim = r['simulated_ppfd_declared_inputs_umol_m2_s']
            m = meas.get((cid, r['point_id']))
            rr = max(w, h) / 60
            dots.append(f'<circle cx="{x}" cy="{y}" r="{rr}" fill="{_color(sim, lo, hi)}"><title>{escape(r["point_id"])} '
                        f'sim {sim}</title></circle>')
            if m is not None:
                dots.append(f'<rect x="{x + rr}" y="{y - rr}" width="{rr * 2}" height="{rr * 2}" fill="{_color(m, lo, hi)}">'
                            f'<title>{escape(r["point_id"])} measured {m:.4g}</title></rect>')
        parts.append(f'<h2>{escape(cid)}</h2><p>circle = simulated, square = measured mean (top view, grow mm; '
                     'points at different heights overlap - see table)</p>'
                     f'<svg width="420" height="320" viewBox="{x0} {y0} {w} {h}">{"".join(dots)}</svg>')
    if comparison and comparison.get('per_point'):
        pp = comparison['per_point']
        mx = max(max(r['measured_mean'], r['predicted']) for r in pp) or 1
        pts = ''.join(f'<circle cx="{40 + 300 * r["measured_mean"] / mx}" cy="{340 - 300 * r["predicted"] / mx}" r="3">'
                      f'<title>{escape(r["case_id"])}:{escape(r["point_id"])}</title></circle>' for r in pp)
        parts.append('<h2>Predicted vs measured (validation points)</h2><svg width="360" height="360">'
                     '<line x1="40" y1="340" x2="340" y2="40" stroke="#999"/>' + pts + '</svg>')
        em = max(abs(r['error']) for r in pp) or 1
        res = ''.join(f'<circle cx="{40 + 300 * r["measured_mean"] / mx}" cy="{190 - 140 * r["error"] / em}" r="3"/>' for r in pp)
        parts.append('<h2>Residual (predicted - measured) vs measured</h2><svg width="360" height="360">'
                     '<line x1="40" y1="190" x2="340" y2="190" stroke="#999"/>' + res + '</svg>')
        parts.append('<h2>Metrics</h2><table><tr>' + ''.join(f'<th>{k}</th>' for k in comparison['metrics'][0]) + '</tr>' +
                     ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in m.values()) + '</tr>'
                             for m in comparison['metrics']) + '</table>')
    parts.append('<h2>Predictions</h2><table><tr>' + ''.join(f'<th>{c}</th>' for c in PREDICTION_COLUMNS[:15]) + '</tr>' +
                 ''.join('<tr>' + ''.join(f'<td>{escape(str(r.get(c)))}</td>' for c in PREDICTION_COLUMNS[:15]) + '</tr>'
                         for r in predictions) + '</table>')
    return ''.join(parts)


# ---------------------------------------------------------------- readiness
def readiness(audit: dict, fit: dict | None, comparison: dict | None, prepared: dict | None) -> dict:
    by_stage = {m['subset'].split('=')[1]: m for m in (comparison or {}).get('metrics', [])
                if m['subset'].startswith('stage=') and ';' not in m['subset']}
    stages = {}
    for st in STAGES:
        m = by_stage.get(st)
        stages[st] = dict(independent_validation='evaluated' if m and m['n_points'] else 'unavailable',
                          acceptance=(m or {}).get('acceptance', 'unavailable'),
                          n_points=(m or {}).get('n_points', 0))
    stages['empty_rig']['calibration'] = (fit or {}).get('status', 'unavailable: no calibration readings')
    partial = [c for c in (prepared or {}).get('cases', []) if c.get('stage') == 'plant'
               and (c.get('surface_evidence') or {}).get('geometry_completeness') != 'whole_plant_reviewed']
    evidence = dict(
        lamp=stages['empty_rig']['acceptance'] == 'meets_predeclared_threshold',
        receiver_and_occlusion=stages['occluder']['acceptance'] == 'meets_predeclared_threshold',
        plant_geometry=stages['plant']['acceptance'] == 'meets_predeclared_threshold' and not partial)
    return dict(schema='ppfd-readiness-1', stages=stages, independent_evidence=evidence,
                canopy_comparison_ready=all(evidence.values()),
                blocking_reasons=[k for k, v in evidence.items() if not v] + (['partial_plant_geometry'] if partial else []),
                missing_physical_inputs=len(audit.get('missing_physical_inputs', [])),
                validated_metric=False, suitable_for_ppfd=False,
                note='Flags are never promoted by software; a person decides after reviewing independent evidence.')


# ---------------------------------------------------------------- whole run
def run(study_path: Path, out: Path, execute: bool = False, java: str = 'java', core: Path = CORE) -> dict:
    """One immutable run folder: audit, split, lamp fit, adapter, prepared jobs, (native), predictions, report."""
    study = load_study(study_path)
    base = study_path.parent
    folder = output_dir(out)
    inputs = folder / 'inputs'; inputs.mkdir()
    sources = {}
    for key in ('scene', 'lamp_conditions', 'rig_surfaces', 'sensors', 'points', 'readings', 'spectrum'):
        p = _path(base, study.get(key))
        if p is not None and p.is_file():
            shutil.copyfile(p, inputs / p.name)
            sources[str(p.resolve())] = file_hash(p)
    shutil.copyfile(study_path, inputs / 'study.json')
    sources[str(study_path.resolve())] = file_hash(study_path)
    audit = validate_inputs(study_path)
    write_json(folder / 'input_audit.json', audit)
    write_rows(folder / 'missing_physical_inputs.csv', audit['missing_physical_inputs'],
               ('entity', 'input', 'status', 'reason', 'blocks'))
    points = _csv(base, study.get('points'), POINT_COLUMNS)
    readings_path = _path(base, study.get('readings'))
    scene_path = _path(base, study.get('scene'))
    fit = None
    if (audit['status'] != 'invalid' and readings_path and readings_path.is_file() and audit['readings']['by_role']['calibration']
            and scene_path and scene_path.is_file()):
        fit = fit_empty_rig(readings_path, scene_path, study.get('fit') or {}, _path(base, study['points']),
                            _path(base, study['sensors']), study.get('python') or {}, _path(base, study.get('rig_surfaces')))
        write_json(folder / 'lamp_fit_manifest.json', fit)
        if fit.get('status') == 'fitted':
            scene_path = fitted_scene(scene_path, fit, folder / 'scene_fitted_frozen.json')
    split = make_split_manifest(points, fit)
    write_json(folder / 'split_manifest.json', split)
    prepared = None
    predictions, comparison = [], None
    steps = dict(audit=audit['status'], fit=(fit or {}).get('status', 'unavailable: no calibration readings'))
    if audit['status'] == 'invalid':
        steps['prepare'] = 'skipped: input audit invalid'
    elif not (scene_path and scene_path.is_file()):
        steps['prepare'] = 'unavailable: no LightScene'
    else:
        try:
            prepared = prepare(study_path, folder / 'prepared', scene_override=scene_path)
            steps['prepare'] = prepared['status']
        except StudyError as exc:
            steps['prepare'] = f'blocked: {exc}'
    if prepared is not None:
        write_rows(folder / 'sensitivity.csv', sensitivity(study, base, scene_path),
                   ('case_id', 'point_id', 'parameter', 'delta', 'reference_ppfd', 'perturbed_ppfd', 'relative_change',
                    'backend', 'status'))
        if execute:
            density_execute(folder / 'prepared', folder / 'execution', core, java)
            res = collect(folder / 'prepared', folder / 'execution', folder)
            predictions = res['predictions']
            steps['execute'] = f'{len(predictions)} predictions, {len(res["failures"])} failures/blocked'
        else:
            steps['execute'] = 'not requested (generated projects are not completed simulations)'
    if predictions and readings_path and readings_path.is_file() and audit['readings']['by_role']['validation']:
        comparison = compare_ppfd(folder / 'sensor_predictions.csv', readings_path, folder / 'split_manifest.json', study)
        write_json(folder / 'comparison.json', comparison)
        write_rows(folder / 'per_point_comparison.csv', comparison['per_point'],
                   ('case_id', 'point_id', 'stage', 'height_band', 'unit', 'n_repeats', 'measured_mean', 'repeat_sd',
                    'repeat_cv', 'predicted', 'error'))
        write_rows(folder / 'stage_height_metrics.csv', comparison['metrics'],
                   ('subset', 'n_points', 'n_repeat_rows', 'n_independent_units', 'mae', 'rmse', 'bias', 'mean_measured',
                    'nrmse', 'ci95_rmse', 'acceptance'))
    else:
        steps['compare'] = 'unavailable: no independent validation readings with absolute predictions'
    ready = readiness(audit, fit, comparison, prepared)
    write_json(folder / 'readiness.json', ready)
    (folder / 'report.html').write_text(report_html(predictions, comparison, ready, points, 'PPFD validation run'),
                                        encoding='utf-8')
    manifest = dict(schema='ppfd-validation-run-1', created_at=datetime.now().astimezone().isoformat(),
                    steps=steps, source_hashes=sources, code_sha256={m: file_hash(Path(__file__).parent / f'{m}.py')
                                                                     for m in ('ppfd_validation', 'groimp_density', 'groimp_light_sanity')},
                    validated_metric=False, suitable_for_ppfd=False)
    write_json(folder / 'run_manifest.json', manifest)
    return manifest


def stale(run_dir: Path) -> dict:
    """A changed source or tool marks the run stale; recomputation must write a new run folder."""
    m = read_json(run_dir / 'run_manifest.json')
    changed = [p for p, h in m['source_hashes'].items() if not Path(p).is_file() or file_hash(Path(p)) != h]
    code = [k for k, h in m['code_sha256'].items() if file_hash(Path(__file__).parent / f'{k}.py') != h]
    return dict(run=str(run_dir), stale=bool(changed or code), changed_sources=changed, changed_code=code)


# ---------------------------------------------------------------- synthetic native checks
def _analytic_sensor(lamp_m: np.ndarray, flux: float, m: float | None, aim: np.ndarray, centre_m, normal, size_m: float,
                     n: int = 60) -> float:
    """Finite-area mean irradiance of a point source: isotropic (m None) or cos^m about aim (cutoff 90)."""
    normal = np.asarray(normal, float)
    t = np.cross(normal, [0, 0, 1] if abs(normal[2]) < .9 else [0, 1, 0]); t /= np.linalg.norm(t)
    b = np.cross(normal, t)
    g = ((np.arange(n) + .5) / n - .5) * size_m
    u, v = np.meshgrid(g, g)
    pts = np.asarray(centre_m)[None] + u.reshape(-1, 1) * t + v.reshape(-1, 1) * b
    d = lamp_m[None] - pts
    r = np.linalg.norm(d, axis=1); dirn = d / r[:, None]
    if m is None:
        inten = np.full(len(r), flux / (4 * math.pi))
    else:
        ce = -(dirn @ aim)
        inten = np.where(ce > 0, flux * (m + 1) / (2 * math.pi) * np.clip(ce, 0, None) ** m, 0.0)
    return float((inten * np.clip(dirn @ normal, 0, None) / r ** 2).mean())


def synthetic(out: Path, java: str = 'java', core: Path = CORE, rays: int = 10_000_000) -> dict:
    """Bounded native GroIMP checks against analytic / python references. Software checks, not rig accuracy."""
    folder = output_dir(out)
    checks, rows_out = {}, []
    size = 0.03

    def run_job(name: str, lamp: str, probes: list, rig=(), seed: int = 1, n_rays: int = rays) -> tuple[dict, dict]:
        d = folder / name
        d.mkdir()
        (d / 'Model.rgg').write_text(make_job_rgg(lamp, n_rays, 1, seed, list(rig), (), None, probes), encoding='utf-8')
        try:
            run_groimp(d, core, java)
        except StudyError:
            pass
        _, pr = parse_log(d / 'groimp.log')
        return probe_values(pr), parse_rig(d / 'groimp.log')

    def plate(pid, c, n):
        rec = dict(x_mm=c[0] * 1000, y_mm=c[1] * 1000, z_mm=c[2] * 1000, normal_x=n[0], normal_y=n[1], normal_z=n[2],
                   width_mm=size * 1000, height_mm=size * 1000)
        return (pid, sensor_triangles(rec))

    def sigma(expected_rate: float, total: float, n_rays: int) -> float:
        hits = n_rays * expected_rate / total
        return 1 / math.sqrt(hits) if hits > 0 else float('inf')

    def record(check, pid, analytic, got_rate, total, n_rays=rays):
        an_rate = analytic * size * size
        sg = sigma(an_rate, total, n_rays) if an_rate > 0 else None
        rel = (got_rate - an_rate) / an_rate if an_rate > 0 else None
        ok = (abs(got_rate) < 1e-12) if an_rate == 0 else abs(rel) <= 3 * sg + 0.005
        rows_out.append(dict(check=check, probe=pid, analytic_ppfd=analytic, groimp_ppfd=got_rate / size ** 2,
                             relative_error=rel, mc_sigma=sg, within_3sigma_plus_0_5pct=ok))
        return ok

    flux, H = 1000.0, 0.5
    lamp_pt = f'[ Translate(0, 0, {H}) Lamp().(setLight(new PointLight().(setPower({flux}f)))) ]'
    # 1 inverse square + cosine response (front, tilted 60, edge-on, reversed) with an isotropic point source
    # tilted: normal leans 60 deg from +z toward the lamp; edge-on: the sensing plane contains the lamp
    s60 = (-math.sin(math.radians(60)), 0.0, math.cos(math.radians(60)))
    # far sensor sits off-axis (never in the near sensor's shadow) and faces the lamp
    far = np.array([0.2, 0.0, -0.5]); far_n = tuple((np.array([0, 0, H]) - far) / np.linalg.norm(np.array([0, 0, H]) - far))
    geo = [('front_d', (0, 0, 0), (0, 0, 1)), ('front_2d', tuple(far), far_n), ('tilt60', (0.3, 0, 0), s60),
           ('edge_on', (-0.3, 0, 0), (0, 1, 0)), ('reversed', (0, 0.3, 0), (0, 0, -1))]
    vals, _ = run_job('point_cosine_inverse_square', lamp_pt, [plate(*g) for g in geo])
    ok = [record('point_source_cosine_and_inverse_square', pid,
                 _analytic_sensor(np.array([0, 0, H]), flux, None, None, c, n, size), vals.get(pid, 0.0), flux)
          for pid, c, n in geo]
    ratio = vals['front_d'] / vals['front_2d'] if vals.get('front_2d') else None
    an_ratio = (_analytic_sensor(np.array([0, 0, H]), flux, None, None, (0, 0, 0), (0, 0, 1), size) /
                _analytic_sensor(np.array([0, 0, H]), flux, None, None, tuple(far), far_n, size))
    checks['point_source_cosine_and_inverse_square'] = dict(passed=all(ok), inverse_square_ratio=ratio,
                                                            analytic_ratio=an_ratio)
    # 2 directional adapter: tilted bar AreaLight vs python finite-rectangle cos^m (fine emitter grid)
    from tomato3d.io.schemas import LedFixture
    def si(v):
        return dict(value=v, status='assumed', source='synthetic software check')
    fx = LedFixture.model_validate(dict(
        fixture_id='synthetic_bar', model_name='synthetic', emitting_center_grow_mm=dict(value=(0.0, 0.0, 500.0), status='assumed', source='synthetic'),
        aim_tilt_deg=si(25.0), aim_azimuth_deg=si(20.0), long_axis_azimuth_deg=si(90.0), emitting_length_mm=si(300.0),
        emitting_width_mm=si(40.0), emitter_grid=(60, 8), photon_flux_umol_s=si(flux), photon_band='PAR_400_700nm',
        distribution=dict(kind='cosine_power', half_intensity_half_angle_deg=si(45.0), cutoff_half_angle_deg=90.0),
        electrical_input_w=si(10.0), electrical_boundary='wall_plug_including_driver',
        schedule=[dict(start_hhmm='00:00', end_hhmm='24:00', output_fraction=1.0)]))
    node, rec = arealight_node(fx)
    states = (SimpleNamespace(fixture_id=fx.fixture_id, emitters=build_emitters(fx)[0]),)
    bar_pts = [('b_m3', (-0.3, 0, 0), (0, 0, 1)), ('b_0', (0, 0, 0), (0, 0, 1)), ('b_p2', (0.2, 0.05, 0), (0, 0, 1)),
               ('b_p4', (0.4, 0.1, 0), (0, 0, 1)), ('b_y3', (0.1, 0.3, 0), (0, 0, 1)), ('b_wall', (0.6, 0, 0.3), (-1, 0, 0))]
    vals, _ = run_job('arealight_tilted_bar', node, [plate(*g) for g in bar_pts])
    ok = []
    sensor = dict(sensing_shape='rectangle', sensing_width_mm=size * 1000, sensing_height_mm=size * 1000)
    for pid, c, n in bar_pts:
        pt = dict(x_mm=c[0] * 1000, y_mm=c[1] * 1000, z_mm=c[2] * 1000, normal_x=n[0], normal_y=n[1], normal_z=n[2])
        an = python_point_ppfd(states, pt, sensor, 30)
        ok.append(record('directional_adapter_vs_python', pid, an, vals.get(pid, 0.0), flux))
    checks['directional_adapter_vs_python'] = dict(passed=all(ok), adapter=rec)
    # 3 flux conservation: closed black box around the bar: absorbed = emitted
    box = []
    for k, (c, n, u) in enumerate([((0, 0, 0.0), (0, 0, 1), (1, 0, 0)), ((0, 0, 1.0), (0, 0, -1), (1, 0, 0)),
                                   ((1, 0, .5), (-1, 0, 0), (0, 1, 0)), ((-1, 0, .5), (1, 0, 0), (0, 1, 0)),
                                   ((0, 1, .5), (0, -1, 0), (1, 0, 0)), ((0, -1, .5), (0, 1, 0), (1, 0, 0))]):
        w_mm, h_mm = (2000, 2000) if k < 2 else (2000, 1000)  # side walls: width along u, height along z
        box.append(dict(surface_id=f'box{k}', kind='wall', role='rig', center_x_mm=c[0] * 1000, center_y_mm=c[1] * 1000,
                        center_z_mm=c[2] * 1000, normal_x=n[0], normal_y=n[1], normal_z=n[2], width_axis_x=u[0],
                        width_axis_y=u[1], width_axis_z=u[2], width_mm=w_mm, height_mm=h_mm, reflectance=0,
                        transmittance=0))
    _, rig = run_job('flux_conservation_box', node, [], box)
    absorbed = sum(rig.values())
    checks['flux_conservation'] = dict(passed=abs(absorbed / flux - 1) <= 0.005, absorbed_umol_s=absorbed, emitted_umol_s=flux,
                                       relative_error=absorbed / flux - 1)
    # 4 opaque occluder: illuminated, shadow interior, shadow edge (point source)
    occ = dict(surface_id='plate', kind='occluder_plate', role='physical_occluder', center_x_mm=0, center_y_mm=0,
               center_z_mm=250, normal_x=0, normal_y=0, normal_z=1, width_axis_x=1, width_axis_y=0, width_axis_z=0,
               width_mm=100, height_mm=100, reflectance=0, transmittance=0)
    # shadow of a 100 mm plate at half height spans |x| <= 100 mm on the floor; edge sensor straddles x=100 mm
    occ_pts = [('lit', (0.3, 0, 0), (0, 0, 1)), ('interior', (0.0, 0, 0), (0, 0, 1)), ('edge', (0.1, 0, 0), (0, 0, 1))]
    vals, _ = run_job('occluder', lamp_pt, [plate(*g) for g in occ_pts], [occ])
    iso = (SimpleNamespace(fixture_id='p', emitters=SimpleNamespace(
        positions_grow_mm=np.array([[0, 0, H * 1000]]), aim=np.array([0, 0, -1.0]), axial_intensity_umol_s_sr=flux / (4 * math.pi),
        cosine_exponent=0.0, cos_cutoff=-1.0)),)
    plates = [rig_surface_triangles({k: str(v) for k, v in occ.items()})]
    ok, classes = [], {}
    for pid, c, n in occ_pts:
        pt = dict(x_mm=c[0] * 1000, y_mm=c[1] * 1000, z_mm=c[2] * 1000, normal_x=n[0], normal_y=n[1], normal_z=n[2])
        an = python_point_ppfd(iso, pt, sensor, 60, opaque_occluders(plates))
        classes[pid] = occlusion_class(iso, pt, sensor, plates, 60, 0.02)
        ok.append(record('opaque_occluder', pid, an, vals.get(pid, 0.0), flux))
    checks['opaque_occluder'] = dict(passed=all(ok) and classes['lit'][0] == 'illuminated' and
                                     classes['interior'][0] == 'shadow_interior' and classes['edge'][0] == 'shadow_edge',
                                     classes={k: list(v) for k, v in classes.items()})
    # 5 random streams: two seeds, same scene
    v1, _ = run_job('seed_a', lamp_pt, [plate('s', (0.2, 0, 0), (0, 0, 1))], seed=11)
    v2, _ = run_job('seed_b', lamp_pt, [plate('s', (0.2, 0, 0), (0, 0, 1))], seed=12)
    v3, _ = run_job('seed_a_repeat', lamp_pt, [plate('s', (0.2, 0, 0), (0, 0, 1))], seed=11)
    checks['random_streams'] = dict(seed_11=v1.get('s'), seed_12=v2.get('s'), seed_11_repeat=v3.get('s'),
                                    distinct_seeds_differ=v1.get('s') != v2.get('s'),
                                    same_seed_reproduces=bool(np.isclose(v1.get('s'), v3.get('s'), rtol=1e-12, atol=0)))
    checks['random_streams']['passed'] = checks['random_streams']['distinct_seeds_differ'] and checks['random_streams']['same_seed_reproduces']
    write_rows(folder / 'synthetic_checks.csv', rows_out, ('check', 'probe', 'analytic_ppfd', 'groimp_ppfd', 'relative_error',
                                                           'mc_sigma', 'within_3sigma_plus_0_5pct'))
    result = dict(schema='ppfd-synthetic-checks-1', data_kind='synthetic software checks (not rig accuracy)', rays=rays,
                  checks=checks, passed=all(c.get('passed', True) for c in checks.values()),
                  groimp_core_sha256=file_hash(core) if core.is_file() else None,
                  validated_metric=False, suitable_for_ppfd=False)
    write_json(folder / 'synthetic_checks.json', result)
    return result


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('templates'); t.add_argument('--out', type=Path, required=True)
    a = sub.add_parser('audit'); a.add_argument('--study', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('--study', type=Path, required=True); r.add_argument('--out', type=Path, required=True)
    r.add_argument('--execute', action='store_true'); r.add_argument('--java', default='java'); r.add_argument('--core', type=Path, default=CORE)
    c = sub.add_parser('compare')
    for k in ('predictions', 'readings', 'split', 'out'):
        c.add_argument(f'--{k}', type=Path, required=True)
    c.add_argument('--study', type=Path)
    v = sub.add_parser('convergence'); v.add_argument('--prepared', type=Path, required=True); v.add_argument('--case', required=True)
    v.add_argument('--out', type=Path, required=True); v.add_argument('--study', type=Path, required=True)
    v.add_argument('--java', default='java'); v.add_argument('--core', type=Path, default=CORE)
    s = sub.add_parser('synthetic'); s.add_argument('--out', type=Path, required=True); s.add_argument('--java', default='java')
    s.add_argument('--core', type=Path, default=CORE); s.add_argument('--rays', type=int, default=10_000_000)
    st = sub.add_parser('stale'); st.add_argument('--run', type=Path, required=True)
    x = ap.parse_args()
    if x.cmd == 'templates':
        res = templates(x.out)
    elif x.cmd == 'audit':
        res = validate_inputs(x.study)
        folder = output_dir(x.out)
        write_json(folder / 'input_audit.json', res)
        write_rows(folder / 'missing_physical_inputs.csv', res['missing_physical_inputs'], ('entity', 'input', 'status', 'reason', 'blocks'))
        res = dict(status=res['status'], n_missing=len(res['missing_physical_inputs']), issues=res['issues'])
    elif x.cmd == 'run':
        res = run(x.study, x.out, x.execute, x.java, x.core)
    elif x.cmd == 'compare':
        res = compare_ppfd(x.predictions, x.readings, x.split, load_study(x.study) if x.study else None)
        folder = output_dir(x.out)
        write_json(folder / 'comparison.json', res)
        res = dict(status=res['status'], metrics=res['metrics'])
    elif x.cmd == 'convergence':
        sv = (load_study(x.study).get('solver') or {})
        res = convergence(x.prepared, x.case, x.out, sv.get('convergence_rays') or [], sv.get('convergence_seeds') or [],
                          sv.get('max_local_native_jobs'), x.java, x.core)
    elif x.cmd == 'synthetic':
        res = synthetic(x.out, x.java, x.core, x.rays)
    else:
        res = stale(x.run)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str)[:6000])


if __name__ == '__main__':
    main()
