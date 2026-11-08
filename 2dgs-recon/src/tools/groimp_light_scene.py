"""Phase 2 of the XL light-simulation plan: a real tomato snapshot in a GroIMP light scene, sensor planes only.

    python -X utf8 -m tools.groimp_light_scene --handoff build/xl_handoff_<run> --plant dwt_01 --out build/<new> \\
        [--scene config/light_scene.json] [--rays 20000000]

Plan: handoff/CLAUDE_XL_LIGHT_SIMULATION_ADOPTION_2026-09-29.md; answers CLAUDE_XL_LIGHT_REVIEW_ANSWERS_2026-09-29.md;
v2 (same day) closes the five gaps in Codex's CLAUDE_XL_LIGHT_PHASE2_PREPARATION_2026-09-29.md:

1. scene input: `--scene` takes the project's LightScene JSON (tomato3d.io.schemas.LightScene): fixture emitters
   (tomato3d.light.scene.build_emitters) and the plant's placement (board origin + yaw) are mapped from the grow frame
   into the export's board frame. Without `--scene` a synthetic lamp is used and every scene value is labelled assumed.
2. display geometry is not optical: the exports' PositionMarker (sphere) and AxisSegment (line) exist only to be seen.
   The optical copy turns them into plain graph nodes (IDs and relations kept); only LeafSurface meshes would stay
   optical. A third run keeps the display geometry, and the optical plant scene must equal the empty scene.
3. sensor quantity: a sensor is one horizontal GroIMP Parallelogram facing the lamp (orientation and anchor measured
   on 2026-09-29: after RL(-90) it lies flat and extends +y from its anchor, matching the analytic top-face value to
   0.3 %), so absorbed rate / area is the incident PPFD on that face. It is relative while the lamp is assumed.
4. outputs: sensor_predictions.csv and validation.csv in the tomato3d/light column contracts plus traceability columns;
   nothing measured is invented (measured values, errors and scores stay `unavailable`). light_results.csv is kept.
5. reproducibility: every consumed export file is checked against the XL handoff's artifact hashes; backend, version,
   ray count and the random-stream behaviour are recorded; results must be complete and finite, zero denominators give
   `unavailable`, never a number.
Everything stays diagnostic: validated_metric=false, suitable_for_ppfd=false.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from tools.groimp_light_sanity import CORE, run_groimp
from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json

UNAVAILABLE = 'unavailable'
SYNTH_FLUX = 1000.0          # umol s-1 of the synthetic lamp
LAMP_ABOVE_TOP_M = 0.25
SENSOR_M, SENSOR_PITCH_M, GRID = 0.03, 0.05, 5
MODULES = ('Plant', 'MainStem', 'CompoundLeaf', 'SubStem', 'Leaflet', 'GrowthShoot')
DISPLAY_MODULES = {'PositionMarker': 'Sphere(0.001)', 'AxisSegment': 'Line'}
EXPORT_FILES = ('Model.rgg', 'coordinates.csv', 'organs.csv', 'export.json')
SENSOR_PREDICTION_COLUMNS = ('reading_id', 'scenario_id', 'geometry_variant_id', 'split_role', 'region_label',
                             'measured_ppfd_umol_m2_s', 'predicted_ppfd_umol_m2_s', 'calibrated_ppfd_umol_m2_s',
                             'error_umol_m2_s', 'calibrated_error_umol_m2_s')
VALIDATION_COLUMNS = ('scenario_id', 'geometry_variant_id', 'split_role', 'subset', 'n_used', 'n_measured_unavailable',
                      'n_prediction_unavailable', 'mae_umol_m2_s', 'rmse_umol_m2_s', 'bias_umol_m2_s', 'calibration_factor',
                      'calibrated_mae_umol_m2_s', 'calibrated_rmse_umol_m2_s', 'calibrated_bias_umol_m2_s')
TRACE_COLUMNS = ('plant_id', 'session_id', 'sensor_id', 'sensor_center_board_m', 'predicted_relative_to_empty',
                 'predicted_incident_umol_m2_s_assumed_lamp', 'python_incident_umol_m2_s', 'backend', 'lamp_status',
                 'geometry_eligibility', 'calculation_status')

HEADER = """import de.grogra.ray.physics.Spectrum;
// ---- light scene added by tools.groimp_light_scene (Phase 2 v2); the plant graph below is the export ----
module Sensor(int sid) extends Parallelogram({size}f, {size}f);
module Lamp extends LightNode;
LightModel lm = new LightModel({rays}, 1);
"""
MEASURE = """
public void measure()
{{
    lm.compute();
    for (Sensor r : (* Sensor *)) {{
        Spectrum s = lm.getAbsorbedPower(r);
        println("SENSOR\\t" + r.sid + "\\t" + s.integrate());
    }}
{counts}
    println("SANITY_DONE");
}}
"""


# ---------------- inputs

def verify_export(handoff: Path, plant: str) -> tuple[Path, dict]:
    """The export folder and the SHA-256 of every consumed file, each checked against the handoff's artifact hashes."""
    export = Path(handoff) / plant / 'export'
    record = read_json(Path(handoff) / 'handoff.json')['plants'].get(plant)
    if record is None:
        raise StudyError(f'{plant} is not in {handoff}/handoff.json')
    expected = {Path(k).as_posix(): v for k, v in record.get('artifact_hashes', {}).items()}
    hashes = {}
    for name in EXPORT_FILES:
        path = export / name
        if not path.is_file():
            raise StudyError(f'missing export file {path}')
        want = expected.get(f'{plant}/export/{name}')
        got = file_hash(path)
        if want is None:
            raise StudyError(f'{name} has no recorded hash in handoff.json; refusing an unverified input')
        if got != want:
            raise StudyError(f'{name} changed since the XL handoff was written (hash differs)')
        hashes[name] = got
    return export, hashes


def plant_extent(export: Path) -> tuple[np.ndarray, float]:
    xyz = []
    with open(export / 'coordinates.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            xyz.append([float(r['x_m']), float(r['y_m']), float(r['z_m'])])
    xyz = np.array(xyz)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise StudyError('The export has no finite coordinates to place a synthetic lamp over')
    return xyz[:, :2].mean(0), float(xyz[:, 2].max())


#: inputs whose status decides whether absolute PPFD may be published (all consumed by the lamp placement)
PLACEMENT_INPUTS = ('board_origin_grow_mm', 'board_yaw_deg')
FIXTURE_INPUTS = ('emitting_center_grow_mm', 'aim_tilt_deg', 'aim_azimuth_deg', 'long_axis_azimuth_deg', 'emitting_length_mm',
                  'emitting_width_mm', 'photon_flux_umol_s', 'distribution.half_intensity_half_angle_deg')
#: acceptance of the empty scene against the python comparator: |relative error| <= 3 sigma + 0.5 % (per sensor)
PY_TOLERANCE_SIGMAS, PY_TOLERANCE_ABS = 3.0, 0.005


def within_tolerance(rel_err: float, sigma: float) -> bool:
    """The stated criterion exactly: |e| <= 3*sigma + 0.005 (Codex review F3: the old form allowed 3*sigma + 0.015)."""
    return abs(rel_err) <= PY_TOLERANCE_SIGMAS * sigma + PY_TOLERANCE_ABS + 1e-12  # 1e-12: float rounding at the boundary


def lamps_from_scene(scene_path: Path, plant: str) -> tuple[list[tuple[np.ndarray, float]], dict]:
    """Point lamps (board frame m, flux umol s-1) from a LightScene, and a description of what was used."""
    from tomato3d.light.scene import build_emitters, load_scene
    scene = load_scene(Path(scene_path))
    placement = next((p for p in scene.plants if p.plant_id == plant), None)
    if placement is None:
        raise StudyError(f'{scene_path} has no placement for {plant}; a plant cannot be put under the lamps without it')
    origin = np.array(placement.board_origin_grow_mm.value, float)
    yaw = math.radians(placement.board_yaw_deg.value)
    c, s = math.cos(yaw), math.sin(yaw)
    grow_to_board = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # inverse of the board->grow yaw
    lamps, used, statuses, blockers = [], [], {}, []
    # every input that changes where the light is or how much there is (Codex review F1, 09-29): the plant placement,
    # and per fixture everything build_emitters consumes
    statuses['placement'] = {k: getattr(placement, k).status for k in PLACEMENT_INPUTS}
    for fx in scene.fixtures:
        em, missing = build_emitters(fx)
        flux = fx.photon_flux_umol_s.value
        if em is None or flux is None:
            used.append(dict(fixture_id=fx.fixture_id, skipped=True, missing=list(missing) or ['photon_flux_umol_s']))
            blockers.append(f'fixture {fx.fixture_id} skipped (missing {", ".join(missing) or "photon_flux_umol_s"}): '
                            'its light is absent from the scene')
            continue
        per = flux / len(em.positions_grow_mm)
        for p in em.positions_grow_mm:
            lamps.append((grow_to_board @ (np.asarray(p) - origin) / 1000.0, per))
        isotropic = em.cosine_exponent == 0 and em.cos_cutoff <= -1 + 1e-9
        used.append(dict(fixture_id=fx.fixture_id, emitters=len(em.positions_grow_mm), flux_umol_s=flux,
                         cosine_exponent=em.cosine_exponent, cos_cutoff=em.cos_cutoff,
                         distribution='isotropic (matches PointLight)' if isotropic else
                         'ASSUMED SURROGATE: fixture is directional (cos^m, cutoff); GroIMP PointLight and the python '
                         'comparator are isotropic, so both omit the real distribution'))
        if not isotropic:  # Codex review F2: a note is not enough, the absolute column must stay unavailable
            blockers.append(f'fixture {fx.fixture_id}: directional LED distribution replaced by an isotropic point surrogate')
        statuses[fx.fixture_id] = {k: _input(fx, k).status for k in FIXTURE_INPUTS}
    if not lamps:
        raise StudyError('no fixture of the scene has complete emitter inputs')
    blockers += [f'{owner}.{k} is {v}' for owner, st in statuses.items() for k, v in st.items() if v != 'measured']
    return lamps, dict(scene=str(scene_path), scene_sha256=file_hash(Path(scene_path)), fixtures=used,
                       placement=dict(board_origin_grow_mm=origin.tolist(), board_yaw_deg=placement.board_yaw_deg.value),
                       input_statuses=statuses, absolute_blockers=blockers,
                       lamp_status='measured' if not blockers else 'assumed (see absolute_blockers)')


def _input(fx, dotted: str):
    obj = fx
    for part in dotted.split('.'):
        obj = getattr(obj, part)
    return obj


def sensors(center: np.ndarray) -> list[tuple[int, float, float]]:
    off = (np.arange(GRID) - (GRID - 1) / 2) * SENSOR_PITCH_M
    return [(i * GRID + j, float(center[0] + dx), float(center[1] + dy)) for i, dx in enumerate(off) for j, dy in enumerate(off)]


# ---------------- scenes

def optical_copy(base: str) -> tuple[str, dict]:
    """The export with display-only geometry turned into plain nodes: IDs and the graph stay, spheres/lines go."""
    text, eligibility = base, {}
    for module, shape in DISPLAY_MODULES.items():
        decl = f'module {module}(String organKey, String metadata) extends {shape};'
        if decl not in text:
            raise StudyError(f'unexpected export: declaration of {module} not found')
        text = text.replace(decl, f'module {module}(String organKey, String metadata) extends Node;  // display only, not optical')
        eligibility[module] = dict(instances=text.count(f'{module}("'), optical=False)
    text = re.sub(r'\.\(setShader\([A-Z_]+\)\)', '', text)
    text = re.sub(r'\.\(setAxis\([-0-9.eEf, ]+\)\)', '', text)
    if 'setShader(' in text or 'setAxis(' in text:
        raise StudyError('a display call is left in the optical copy; refusing to guess its target')
    eligibility['LeafSurface'] = dict(instances=text.count('LeafSurface("'), optical=True)
    return text, eligibility


def scene_rgg(base: str | None, lamps, grid, rays: int) -> str:
    lamp_nodes = ' '.join(f'[ Translate({p[0]:.5f}f, {p[1]:.5f}f, {p[2]:.5f}f) Lamp().(setLight(new PointLight().(setPower({f:.6g}f)))) ]'
                          for p, f in lamps)
    # after RL(-90) a Parallelogram lies flat and extends +y from its anchor: anchor at y - size/2 centres it
    sens = ' '.join(f'[ Translate({x:.5f}f, {y - SENSOR_M / 2:.5f}f, 0) RL(-90) Sensor({i}).(setShader(BLACK)) ]' for i, x, y in grid)
    header = HEADER.format(size=SENSOR_M, rays=rays)
    if base is None:
        return header + f'protected void init()\n[\n    Axiom ==> {lamp_nodes} {sens};\n]\n' + MEASURE.format(counts='')
    if base.count('Axiom ==>') != 1:
        raise StudyError('Unexpected export: expected exactly one "Axiom ==>" production')
    body = base.replace('Axiom ==>', f'Axiom ==> {lamp_nodes} {sens}', 1)
    lines = body.split('\n')
    last = max((k for k, line in enumerate(lines) if line.startswith('import ')), default=-1)
    body = '\n'.join(lines[:last + 1] + [header] + lines[last + 1:])
    counts = '\n'.join(f'    println("COUNT {m}=" + count((* {m} *)));' for m in MODULES)
    return body + MEASURE.format(counts=counts)


def run_scene(folder: Path, rgg: str, core: Path) -> dict:
    folder.mkdir()
    (folder / 'Model.rgg').write_text(rgg, encoding='utf-8')
    t0 = time.time()
    try:
        run_groimp(folder, core)
    except StudyError:
        pass  # run_groimp parses Phase-1 lines; the log is parsed here
    text = (folder / 'groimp.log').read_text(encoding='utf-8', errors='replace')
    if 'SANITY_DONE' not in text:
        raise StudyError(f'measure() did not run; see {folder / "groimp.log"}')
    values = {int(i): float(v) for i, v in re.findall(r'SENSOR\t(\d+)\t([^\s]+)', text)}
    return dict(sensors=values, counts={k: int(v) for k, v in re.findall(r'COUNT (\w+)=(\d+)', text)},
                seconds=round(time.time() - t0, 1), model_sha256=file_hash(folder / 'Model.rgg'))


def python_sensors(lamps, grid, n: int = 120) -> dict:
    """Incident photon rate on each flat sensor from tomato3d.light.raytrace.direct_light (isotropic point lamps)."""
    from types import SimpleNamespace
    from tomato3d.io.schemas import LightSimulationSettings
    from tomato3d.light.raytrace import Occluders, Receivers, direct_light
    from tomato3d.light.scene import Emitters
    settings = LightSimulationSettings(triangle_subdivisions=1, ray_offset_mm=0.01, max_ray_triangle_pairs_per_chunk=1_000_000)
    states = [SimpleNamespace(fixture_id=f'lamp{k}', emitters=Emitters(
        fixture_id=f'lamp{k}', positions_grow_mm=np.array([p]) * 1000, aim=np.array([0.0, 0.0, -1.0]),
        axial_intensity_umol_s_sr=f / (4 * math.pi), cosine_exponent=0.0, cos_cutoff=-1.0)) for k, (p, f) in enumerate(lamps)]
    g = ((np.arange(n) + 0.5) / n - 0.5) * SENSOR_M
    out = {}
    for i, x, y in grid:
        X, Y = np.meshgrid(x + g, y + g)
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1) * 1000
        rec = Receivers(points_grow_mm=pts, unit_normals=np.tile([0.0, 0.0, 1.0], (len(pts), 1)), two_sided=False,
                        own_triangle=np.full(len(pts), -1))
        ppfd = direct_light(rec, tuple(states), Occluders.none(), None, settings).ppfd_umol_m2_s.sum(0)
        out[i] = float(ppfd.mean() * SENSOR_M ** 2)
    return out


# ---------------- run

def build(handoff: Path, plant: str, out: Path, rays: int = 20_000_000, core: Path = CORE, scene: Path | None = None) -> dict:
    export, hashes = verify_export(handoff, plant)
    manifest = read_json(export / 'export.json')
    out = output_dir(Path(out))
    center, top = plant_extent(export)
    if scene:
        lamps, scene_used = lamps_from_scene(scene, plant)
    else:
        lamps = [(np.array([center[0], center[1], top + LAMP_ABOVE_TOP_M]), SYNTH_FLUX)]
        scene_used = dict(scene=None, lamp_status='assumed (synthetic: one isotropic point lamp '
                                                   f'{LAMP_ABOVE_TOP_M} m above the highest exported coordinate)',
                          flux_umol_s=SYNTH_FLUX, lamp_board_m=lamps[0][0].round(4).tolist())
    write_json(out / 'scene_used.json', scene_used)
    grid = sensors(center)
    base = (export / 'Model.rgg').read_text(encoding='utf-8')
    optical, eligibility = optical_copy(base)
    runs = {name: run_scene(out / name, scene_rgg(b, lamps, grid, rays), core)
            for name, b in (('empty', None), ('plant_optical', optical), ('plant_display', base))}
    py = python_sensors(lamps, grid)
    area = SENSOR_M ** 2

    def finite(v):
        return v is not None and math.isfinite(v)
    problems = [f'{name}: sensor {i} missing or not finite' for name, r in runs.items() for i, _, _ in grid
                if not finite(r['sensors'].get(i))]
    if problems:
        raise StudyError('incomplete sensor results: ' + '; '.join(problems[:5]))
    scenario = f'phase2v2_{plant}_{"scene" if scene else "synthetic"}'
    variant = 'xl_static_optical'
    sp_rows, lr_rows, rel_opt, rel_disp, gro_vs_py = [], [], [], [], []
    hits_empty = {i: rays * runs['empty']['sensors'][i] / sum(f for _, f in lamps) for i, _, _ in grid}
    for i, x, y in grid:
        e, o, d = (runs[k]['sensors'][i] for k in ('empty', 'plant_optical', 'plant_display'))
        r_opt = o / e if e > 0 else None
        r_disp = d / e if e > 0 else None
        rel_opt.append(r_opt); rel_disp.append(r_disp)
        gro_vs_py.append(e / py[i] - 1 if py[i] > 0 else None)
        absolute = scene_used['lamp_status'] == 'measured' and not scene_used.get('absolute_blockers')
        sp_rows.append({
            'reading_id': f'synthetic_sensor_{i:02d}', 'scenario_id': scenario, 'geometry_variant_id': variant,
            'split_role': 'unassigned', 'region_label': 'board_plane_grid', 'measured_ppfd_umol_m2_s': UNAVAILABLE,
            'predicted_ppfd_umol_m2_s': f'{o / area:.6g}' if absolute else UNAVAILABLE,
            'calibrated_ppfd_umol_m2_s': UNAVAILABLE, 'error_umol_m2_s': UNAVAILABLE, 'calibrated_error_umol_m2_s': UNAVAILABLE,
            'plant_id': plant, 'session_id': manifest.get('session_id'), 'sensor_id': i,
            'sensor_center_board_m': f'{x:.4f};{y:.4f};0', 'predicted_relative_to_empty': f'{r_opt:.6g}' if r_opt is not None else UNAVAILABLE,
            'predicted_incident_umol_m2_s_assumed_lamp': f'{o / area:.6g}', 'python_incident_umol_m2_s': f'{py[i] / area:.6g}',
            'backend': 'GroIMP 2.2.1 LightModel CPU', 'lamp_status': scene_used['lamp_status'],
            'geometry_eligibility': 'display geometry excluded; optical surfaces: ' + str(eligibility['LeafSurface']['instances']),
            'calculation_status': 'computed' if e > 0 else 'unavailable_zero_empty_scene'})
        for q, v, unit in (('incident_photon_rate_empty', e, 'umol s-1'), ('incident_photon_rate_plant_optical', o, 'umol s-1'),
                           ('incident_photon_rate_plant_display', d, 'umol s-1'), ('incident_photon_rate_python', py[i], 'umol s-1'),
                           ('relative_incident_optical', r_opt, '1'), ('relative_incident_display', r_disp, '1')):
            lr_rows.append(dict(scenario_id=scenario, plant_id=plant, session_id=manifest.get('session_id'), organ_key=f'sensor/{i}',
                                geometry_source='horizontal_sensor_plate', quantity=q, unit=unit,
                                value=f'{v:.6g}' if v is not None else UNAVAILABLE, surface_or_sensor_plane=f'board z=0 ({x:.3f}, {y:.3f}) m',
                                calculation_status='computed' if v is not None else 'unavailable', backend='groimp' if 'python' not in q else 'python'))
    with open(export / 'organs.csv', encoding='utf-8') as fh:
        organs = list(csv.DictReader(fh))
    for o in organs:
        if o.get('module') in MODULES:
            lr_rows.append(dict(scenario_id=scenario, plant_id=plant, session_id=manifest.get('session_id'), organ_key=o['key'],
                                geometry_source='diagnostic_marker_or_axis (not optical)', quantity='incident_photon_rate', unit='umol s-1',
                                value=UNAVAILABLE, surface_or_sensor_plane='', calculation_status='not_a_surface', backend=''))
    write_csv(out / 'sensor_predictions.csv', SENSOR_PREDICTION_COLUMNS + TRACE_COLUMNS, sp_rows)
    write_csv(out / 'validation.csv', VALIDATION_COLUMNS + ('note',), [dict(
        scenario_id=scenario, geometry_variant_id=variant, split_role='all', subset='all', n_used=0,
        n_measured_unavailable=len(grid), n_prediction_unavailable=sum(1 for r in sp_rows if r['predicted_ppfd_umol_m2_s'] == UNAVAILABLE),
        **{k: UNAVAILABLE for k in VALIDATION_COLUMNS[7:]}, note='no PPFD readings supplied; no score is computed')])
    write_csv(out / 'light_results.csv', ('scenario_id', 'plant_id', 'session_id', 'organ_key', 'geometry_source', 'quantity', 'unit',
                                          'value', 'surface_or_sensor_plane', 'calculation_status', 'backend'), lr_rows)
    expected_counts = {m: sum(1 for o in organs if o.get('module') == m) for m in MODULES}
    sigma = {i: 1 / math.sqrt(h) if h > 0 else float('inf') for i, h in hits_empty.items()}
    opt_dev = [abs(r - 1) / sigma[i] for (i, _, _), r in zip(grid, rel_opt) if r is not None]
    py_ok = [within_tolerance(v, sigma[i]) for (i, _, _), v in zip(grid, gro_vs_py) if v is not None]
    checks = dict(
        inputs_match_handoff_hashes=True,
        organ_counts_match_export=all(runs[k]['counts'].get(m) == expected_counts[m] for k in ('plant_optical', 'plant_display') for m in MODULES),
        every_sensor_finite=True,
        display_only_plant_equals_empty_within_3_sigma=bool(opt_dev) and max(opt_dev) <= 3.0,
        empty_scene_matches_python_within_3_sigma_plus_0_5pct=bool(py_ok) and all(py_ok))
    stat = lambda v: dict(min=round(float(min(v)), 5), max=round(float(max(v)), 5)) if v else None  # noqa: E731
    doc = dict(schema='groimp-light-scene-2', phase=2, status='passed' if all(checks.values()) else 'failed', checks=checks,
               plant_id=plant, session_id=manifest.get('session_id'), consumed_inputs=hashes, handoff=str(handoff),
               scene=scene_used, geometry_eligibility=eligibility,
               sensors=dict(ids=[i for i, _, _ in grid], shape='GroIMP Parallelogram, flat, facing +z, RL(-90), centred',
                            size_m=SENSOR_M, pitch_m=SENSOR_PITCH_M, plane='board z=0', quantity='incident photon rate on the top face'),
               relative_incident_optical=stat([r for r in rel_opt if r is not None]),
               relative_incident_display=stat([r for r in rel_disp if r is not None]),
               groimp_empty_vs_python=stat([v for v in gro_vs_py if v is not None]),
               backend=dict(name='GroIMP LightModel (CPU Monte-Carlo raytracer)', version='2.2.1', core_sha256=file_hash(core),
                            rays=rays, depth=1, random_stream='fixed inside LightModel; no seed is exposed by this API, repeated '
                                                              'compute() calls give identical numbers (measured 2026-09-29)'),
               runs={k: {kk: vv for kk, vv in v.items() if kk != 'sensors'} for k, v in runs.items()},
               frame='board frame, metres, +Z up (as the export)', validated_metric=False, suitable_for_ppfd=False,
               note='software check: without reviewed leaf surfaces no organ receives or casts light in the optical scene')
    write_json(out / 'manifest.json', doc)
    return doc


def write_csv(path: Path, fields, rows) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--handoff', required=True, type=Path)
    ap.add_argument('--plant', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--scene', type=Path, help='LightScene JSON (default: synthetic assumed lamp)')
    ap.add_argument('--rays', type=int, default=20_000_000)
    a = ap.parse_args()
    d = build(a.handoff, a.plant, a.out, a.rays, scene=a.scene)
    print(json.dumps({k: d[k] for k in ('status', 'checks', 'relative_incident_optical', 'relative_incident_display',
                                        'groimp_empty_vs_python', 'geometry_eligibility')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
