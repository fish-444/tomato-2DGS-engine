"""Reviewed-triangle GroIMP density jobs, result maps and independent PPFD audit.

python -m tools.groimp_density --help
The existing LightScene remains the lighting contract. A reviewed GroIMP lamp
adapter must name and hash that scene; directional LEDs are never replaced by
isotropic lamps implicitly. Generated projects are not completed simulations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path
import re
import time

import numpy as np

from tomato3d.io.schemas import LEAF_SURFACES_CSV_COLUMNS
from tomato3d.light.geometry import load_geometry, place_in_grow_frame, triangle_areas_mm2
from tomato3d.light.scene import resolve_scenario
from tools.canopy_light import identifier, materialize_case
from tools.groimp_light_sanity import CORE, run_groimp
from tools.groimp_structure_export import float_literal, java_string
from tools.leaf_light.common import (StudyError, digest, file_hash, number, output_dir,
                                    read_json, rows, write_json, write_rows)

SENSORS = ('sensor_id','x_mm','y_mm','z_mm','normal_x','normal_y','normal_z',
           'width_mm','height_mm','height_group')
TRIANGLES = ('surface_id','plant_id','source_plant_id','session_id','organ_id','leaflet_id',
             'triangle_id','height_group','area_mm2','x_mm','y_mm','z_mm')
MEASUREMENTS = ('reading_id','case_id','sensor_id','date','split_role','ppfd_umol_m2_s',
                'scene_sha256','sensor_spec_sha256','sensor_model','calibration_reference',
                'all_fixtures_full_output')


def positive(value: object, name: str) -> float:
    result = number(value, name)
    if result <= 0:
        raise StudyError(f'{name} must be positive')
    return result


def sensor_triangles(record: dict) -> np.ndarray:
    centre = np.array([number(record[f'{a}_mm'], a) for a in 'xyz'])
    normal = np.array([number(record[f'normal_{a}'], a) for a in 'xyz'])
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-6):
        raise StudyError('Sensor normal must be a unit vector')
    width_mm, height_mm = positive(record['width_mm'],'width_mm'), positive(record['height_mm'],'height_mm')
    tangent = np.cross(normal, [0,0,1] if abs(normal[2]) < .9 else [0,1,0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    corners = [centre+sx*width_mm/2*tangent+sy*height_mm/2*bitangent for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    return np.array([[corners[0],corners[1],corners[2]],[corners[0],corners[2],corners[3]]])


def validate_study(plan: dict) -> None:
    if plan.get('schema_version') != 'groimp-density-flowering-1':
        raise StudyError('Unsupported study schema')
    positive(plan.get('plot_width_mm'), 'plot_width_mm')
    positive(plan.get('plot_depth_mm'), 'plot_depth_mm')
    if plan.get('co2_recorded') is not False:
        raise StudyError('This track explicitly has no recorded CO2')
    if plan.get('groimp_optical_transport') != 'diffuse_reflection_and_transmission':
        raise StudyError('Explicit GroIMP transport setting required; do not inherit the Python no-reflection flag')
    if not plan.get('cases'):
        raise StudyError('Supply explicit density cases')
    names = [identifier(c['case_id'],'case_id') for c in plan['cases']]
    if len(names) != len(set(names)):
        raise StudyError('Duplicate case ID')
    density_counts={}
    for c in plan['cases']:
        if 'scenarios' in c or 'scene' in c:
            raise StudyError('Density cases cannot override fixed lighting')
        identifier(c['density_id'],'density_id')
        date.fromisoformat(c['snapshot_date'])
        number(c['height_split_board_mm'],'height_split_board_mm')
        if not c.get('instances'):
            raise StudyError('No instances in density case')
        if c.get('data_kind') not in ('observed_geometry','hypothetical_layout','synthetic'):
            raise StudyError('Explicit data_kind is required')
        if c['data_kind']=='observed_geometry' and any(i['plant_id']!=i['source_plant_id'] for i in c['instances']):
            raise StudyError('Copied template instances cannot be observed plants')
        if c['density_id'] in density_counts and density_counts[c['density_id']]!=len(c['instances']):
            raise StudyError('One density_id cannot have different densities in a fixed footprint')
        density_counts[c['density_id']]=len(c['instances'])
        ids = [identifier(i['plant_id'],'plant_id') for i in c['instances']]
        if len(ids) != len(set(ids)):
            raise StudyError('Duplicate instance ID')
        origins = []
        for i in c['instances']:
            xyz = i['placement']['board_origin_grow_mm']['value']
            if len(xyz) != 3 or not all(np.isfinite(xyz)):
                raise StudyError('Finite grow-frame origins required')
            if not 0 <= xyz[0] <= plan['plot_width_mm'] or not 0 <= xyz[1] <= plan['plot_depth_mm']:
                raise StudyError('Plant origin outside fixed plot footprint')
            origins.append(tuple(xyz))
        if len(origins) != len(set(origins)):
            raise StudyError('Plant origins overlap')
    for k in ('rays','max_depth'):
        if not isinstance(plan.get(k), int) or plan[k] < 1:
            raise StudyError(f'{k} must be a positive integer')
    if not isinstance(plan.get('seed'), int):
        raise StudyError('Explicit random seed required')


def templates(out: Path) -> dict:
    """Empty contracts only: physical values are not guessed from photographs."""
    folder=output_dir(out)
    write_rows(folder/'leaf_surfaces.csv',[],LEAF_SURFACES_CSV_COLUMNS)
    write_rows(folder/'sensors.csv',[],SENSORS)
    write_rows(folder/'ppfd_readings.csv',[],MEASUREMENTS)
    write_json(folder/'surface_review.json',dict(geometry_sha256=None,geometry_variant_id=None,
        snapshot_date=None,surface_kind='leaf_triangles',review_state='pending',reviewer=None,
        reviewed_at=None,validated_metric=False,suitable_for_ppfd=False))
    write_json(folder/'lamp_adapter.json',dict(scene_sha256=None,review_state='pending',reviewer=None,
        coordinate_frame='grow_m',power_convention='photon_umol_s',empty_scene_validation=None,
        header_rgg='',initialization_rgg='',absolute_led_ppfd_valid=False))
    write_json(folder/'study.json',dict(schema_version='groimp-density-flowering-1',
        co2_recorded=False,planned_observation_weeks=6,plot_width_mm=None,plot_depth_mm=None,
        rays=2000000,max_depth=6,seed=20260930,scene='scene.json',sensors='sensors.csv',
        groimp_optical_transport='diffuse_reflection_and_transmission',
        lamp_adapter='lamp_adapter.json',cases=[]))
    (folder/'README.txt').write_text('Fill measured inputs before prepare. No guessed plot, lamp or plant values.\n'
        'scene.json uses the existing tomato3d.io.schemas.LightScene contract; config belongs to the human.\n'
        'Each case: case_id, density_id, snapshot_date, geometry, geometry_variant_id, surface_review,\n'
        'height_split_board_mm, data_kind, instances[{plant_id,source_plant_id,source_session_id,placement}].\n'
        'placement uses the existing PlantPlacement contract. Actual observations retain physical IDs; template copies are hypothetical.\n'
        'Lamp adapter is reviewed XL declarations plus graph nodes in metres, tied to the scene hash.\n'
        'Do not copy the isotropic synthetic test into a real directional LED experiment.\n'
        'Sensor rectangle is its actual sensing area; all runs assume all fixtures at full output.\n'
        'Numeric solver defaults above are exploratory, not convergence guarantees.\n',encoding='utf-8')
    return dict(status='empty_templates_created',out=str(folder))


def surface_gate(case: dict, base: Path) -> tuple[list[str], dict]:
    geometry = base/case['geometry']
    review_file = base/case['surface_review']
    blockers, evidence = [], {}
    if not geometry.is_file():
        return ['missing_leaf_triangle_file'], evidence
    evidence['geometry_sha256'] = file_hash(geometry)
    if not review_file.is_file():
        return ['missing_surface_review'], evidence
    review = read_json(review_file)
    evidence.update(review_path=str(review_file.resolve()), review_sha256=file_hash(review_file), review=review)
    if review.get('geometry_sha256') != evidence['geometry_sha256']:
        blockers.append('surface_review_hash_mismatch')
    if review.get('geometry_variant_id') != case['geometry_variant_id']:
        blockers.append('surface_variant_mismatch')
    if review.get('snapshot_date') != case['snapshot_date']:
        blockers.append('surface_snapshot_date_mismatch')
    if review.get('surface_kind') != 'leaf_triangles' or review.get('review_state') != 'accepted':
        blockers.append('not_accepted_leaf_triangles')
    if not review.get('reviewer') or not review.get('reviewed_at'):
        blockers.append('missing_review_provenance')
    geometry_rows = rows(geometry, LEAF_SURFACES_CSV_COLUMNS)
    selected = [r for r in geometry_rows if r['geometry_variant_id'] == case['geometry_variant_id']]
    if not selected:
        blockers.append('no_leaf_surfaces')
    else:
        surfaces = load_geometry([geometry])[case['geometry_variant_id']]
        for item in case['instances']:
            subset = [i for i,p in enumerate(surfaces.plant_ids) if p == item['source_plant_id']]
            if not subset or not any(surfaces.is_receiver[subset]):
                blockers.append(f"no_leaf_surfaces:{item['source_plant_id']}")
            if any(surfaces.session_ids[i] != item['source_session_id'] for i in subset):
                blockers.append(f"session_mismatch:{item['source_plant_id']}")
        if any(not x for x in surfaces.normal_known):
            blockers.append('unknown_surface_normal_convention')
        if not all(surfaces.is_receiver):
            blockers.append('non_leaf_geometry_requires_separate_optical_adapter')
        if case.get('data_kind') != 'synthetic' and any(s in ('simplified','assumed') for s in surfaces.geometry_sources):
            blockers.append('assumed_or_simplified_source_not_observed_leaf_surface')
    return sorted(set(blockers)), evidence


def lamp_adapter(plan: dict, base: Path, scene_path: Path) -> tuple[dict | None,list[str]]:
    spec_path = base/plan['lamp_adapter']
    if not spec_path.is_file():
        return None, ['missing_reviewed_groimp_lamp_adapter']
    spec = read_json(spec_path)
    blockers = []
    if spec.get('scene_sha256') != file_hash(scene_path):
        blockers.append('lamp_adapter_scene_hash_mismatch')
    if spec.get('review_state') != 'accepted' or not spec.get('reviewer'):
        blockers.append('lamp_adapter_not_reviewed')
    if spec.get('coordinate_frame') != 'grow_m' or spec.get('power_convention') != 'photon_umol_s':
        blockers.append('lamp_adapter_frame_or_power_convention_mismatch')
    if not spec.get('empty_scene_validation'):
        blockers.append('lamp_adapter_missing_empty_scene_validation')
    if not isinstance(spec.get('initialization_rgg'), str) or not spec['initialization_rgg'].strip():
        blockers.append('lamp_adapter_empty')
    spec = dict(spec, path=str(spec_path.resolve()), sha256=file_hash(spec_path))
    return spec, blockers


def mesh_node(kind: str, key: str, vertices_mm: np.ndarray, shader: str) -> str:
    coords = ','.join(float_literal(float(c)/1000) for c in vertices_mm.reshape(-1))
    return f'[{kind}({java_string(key)}).(setPolygons(getMesh(new FloatList(new float[]{{{coords}}})).getPolygons()),setShader({shader}))]'


def make_rgg(plan: dict, triangles: list[tuple[dict,np.ndarray]], adapter: dict,
             optics: dict, sensor: dict | None = None) -> str:
    """Native optics: diffuse reflection/transmission; alpha is the residual."""
    rho, tau, alpha = [number(optics[k]['value'], k) for k in ('reflectance','transmittance','absorptance')]
    if min(rho,tau,alpha) < 0 or not np.isclose(rho+tau+alpha,1,atol=1e-6):
        raise StudyError('GroIMP leaf optics require R+T+A=1 with explicit values')
    header = '\n'.join([
        'import de.grogra.xl.util.FloatList;',
        'import de.grogra.imp3d.shading.SideSwitchShader;',
        'module LeafSurface(String surfaceId) extends MeshNode;',
        'module Probe(String sensorId) extends MeshNode;',
        f'LightModel lm = new LightModel({plan["rays"]}, {plan["max_depth"]});',
        'static Phong leafOptics = new Phong();',
        'static {',
        f'leafOptics.setDiffuse(new Graytone({float_literal(rho)}));',
        f'leafOptics.setDiffuseTransparency(new Graytone({float_literal(tau)}));',
        'leafOptics.setSpecular(new Graytone(0));',
        'leafOptics.setTransparency(new Graytone(0));',
        'leafOptics.setAmbient(new Graytone(0));',
        'leafOptics.setEmissive(new Graytone(0));',
        'leafOptics.setInterpolatedTransparency(false);',
        '}', adapter.get('header_rgg','')])
    nodes = [adapter['initialization_rgg']]
    for meta, vertices in triangles:
        nodes.append(mesh_node('LeafSurface', meta['surface_id'], vertices, 'leafOptics'))
    if sensor:
        # One probe per run: sensor plates cannot shade one another.
        for i, v in enumerate(sensor_triangles(sensor)):
            nodes.append(mesh_node('Probe', f"{sensor['sensor_id']}:{i}", v,
                'new SideSwitchShader(BLACK, new RGBAShader(0,0,0,0))'))
    body = '\n'.join(nodes)
    measure = '''
public void measure() {
    lm.setSeed(SEED);
    lm.compute();
    for (LeafSurface r : (* LeafSurface *)) {
        println("LEAF\\t" + r.surfaceId + "\\t" + lm.getReceivedPower(r).integrate() + "\\t" + lm.getAbsorbedPower(r).integrate());
    }
    for (Probe s : (* Probe *)) {
        println("PROBE\\t" + s.sensorId + "\\t" + lm.getAbsorbedPower(s).integrate());
    }
    println("SANITY_DONE");
}
'''.replace('SEED',str(plan['seed'])+'L')
    return header+'\nprotected void init()\n[\nAxiom ==> '+body+';\n]\n'+measure


def prepare(plan_path: Path, out: Path) -> dict:
    plan = read_json(plan_path); validate_study(plan)
    base = plan_path.parent
    scene_path = base/plan['scene']
    from tomato3d.light.scene import load_scene
    scene = load_scene(scene_path)
    adapter, lamp_blockers = lamp_adapter(plan,base,scene_path)
    sensor_path = base/plan['sensors']
    sensors = rows(sensor_path,SENSORS)
    if len({s['sensor_id'] for s in sensors}) != len(sensors):
        raise StudyError('Duplicate sensor ID')
    for s in sensors:
        identifier(s['sensor_id'],'sensor_id'); sensor_triangles(s)
    folder = output_dir(out)
    write_json(folder/'study.json',plan)
    write_json(folder/'scene.json',read_json(scene_path))
    write_rows(folder/'sensors.csv',sensors,SENSORS)
    if adapter:
        write_json(folder/'lamp_adapter.json',adapter)
    report = dict(schema_version='groimp-density-jobs-1', status='prepared', cases=[],
        source_plan=str(plan_path.resolve()), source_hashes={str(p.resolve()):file_hash(p) for p in (plan_path,scene_path,sensor_path)},
        scene_sha256=file_hash(scene_path), sensor_spec_sha256=file_hash(sensor_path),
        fixed_plot_area_m2=plan['plot_width_mm']*plan['plot_depth_mm']/1e6,
        source_python_reflections=scene.reflections, groimp_optical_transport=plan['groimp_optical_transport'],
        validated_metric=False, suitable_for_ppfd=False, photosynthesis='unavailable',
        limits=['Generated GroIMP projects require execution; outputs remain diagnostic until independent sensor validation.',
                'Static snapshots apply only to their recorded dates. Template copies do not simulate morphological adaptation.',
                'Leaf received flux includes both faces; sensor PPFD is front-face absorbed flux / sensor area.',
                'Diffuse leaf transmission/reflection and finite path depth are explicit optical assumptions.'])
    for c in plan['cases']:
        blockers, evidence = surface_gate(c,base)
        blockers += lamp_blockers
        item = dict(case_id=c['case_id'], density_id=c['density_id'], snapshot_date=c['snapshot_date'],
            n_plants=len(c['instances']), plants_per_m2=len(c['instances'])/report['fixed_plot_area_m2'],
            status='blocked' if blockers else 'prepared', blockers=blockers, surface_evidence=evidence)
        report['cases'].append(item)
        if blockers:
            continue
        cf = folder/c['case_id']; cf.mkdir()
        case = dict(c,scene=str(scene_path.resolve()),geometry=str((base/c['geometry']).resolve()),
                    plot_area_m2=report['fixed_plot_area_m2'],data_kind=c.get('data_kind','hypothetical_layout'))
        # The source scene is identical for all cases; no case-specific lamp/schedule overrides.
        if 'scenarios' in c or 'scene' in c:
            raise StudyError('Density cases cannot override fixed lighting')
        model, geometry, provenance = materialize_case(case,base,cf)
        resolved = resolve_scenario(model,'as_built')
        surfaces = load_geometry([geometry])[c['geometry_variant_id']]
        grow = place_in_grow_frame(surfaces,resolved.placements)
        areas = triangle_areas_mm2(grow)
        instances = {i['plant_id']:i for i in c['instances']}
        triangles = []
        for i in range(surfaces.n_triangles):
            pid = surfaces.plant_ids[i]
            centre = grow[i].mean(axis=0)
            meta = dict(surface_id=f't{i:07d}', plant_id=pid, source_plant_id=instances[pid]['source_plant_id'],
                session_id=surfaces.session_ids[i],organ_id=surfaces.organ_ids[i],leaflet_id=surfaces.leaflet_ids[i],
                triangle_id=surfaces.triangle_ids[i],area_mm2=float(areas[i]),
                height_group='lower' if surfaces.vertices_board_mm[i,:,2].mean() < c['height_split_board_mm'] else 'upper',
                **{f'{a}_mm':float(v) for a,v in zip('xyz',centre)})
            triangles.append((meta,grow[i]))
        write_rows(cf/'triangles.csv',[m for m,v in triangles],TRIANGLES)
        write_json(cf/'triangle_vertices.json',{m['surface_id']:v.tolist() for m,v in triangles})
        optic = model.leaf_optics.model_dump(mode='json')
        main = cf/'leaves'; main.mkdir()
        (main/'Model.rgg').write_text(make_rgg(plan,triangles,adapter,optic),encoding='utf-8')
        jobs = ['leaves']
        for s in sensors:
            name = 'sensor_'+s['sensor_id']; dest=cf/name; dest.mkdir()
            (dest/'Model.rgg').write_text(make_rgg(plan,triangles,adapter,optic,s),encoding='utf-8')
            jobs.append(name)
        write_json(cf/'provenance.json',provenance)
        item.update(jobs=jobs, n_triangles=len(triangles),
            common_full_output_hours_per_day=(resolved.fixtures[0].full_output_hours_per_day
                if resolved.fixtures and len({f.full_output_hours_per_day for f in resolved.fixtures})==1 else None),
            lighting_state='all_fixtures_full_output',
            model_hashes={j:file_hash(cf/j/'Model.rgg') for j in jobs})
    report['status'] = 'prepared' if all(c['status']=='prepared' for c in report['cases']) else 'blocked_or_partial'
    report['artifact_hashes']={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file()}
    write_json(folder/'jobs.json',report)
    return report


def parse_log(path: Path) -> tuple[dict,dict]:
    text = path.read_text(encoding='utf-8',errors='replace')
    if 'SANITY_DONE' not in text:
        raise StudyError('Incomplete GroIMP run')
    leaves, probes = {}, {}
    for key, incoming, absorbed in re.findall(r'LEAF\t([^\s]+)\t([^\s]+)\t([^\s]+)',text):
        if key in leaves:
            raise StudyError('Duplicate leaf output')
        vals = (number(incoming,'incoming flux'),number(absorbed,'absorbed flux'))
        if min(vals) < 0 or vals[1] > vals[0]+1e-7:
            raise StudyError('Invalid received/absorbed flux')
        leaves[key] = vals
    for key,value in re.findall(r'PROBE\t([^\s]+)\t([^\s]+)',text):
        if key in probes or number(value,'probe flux') < 0:
            raise StudyError('Duplicate/negative probe output')
        probes[key] = float(value)
    return leaves,probes


def execute(prepared: Path, out: Path, core: Path, java: str) -> dict:
    """Fresh execution directory; never overwrite a prepared or completed run."""
    import shutil
    jobs = read_json(prepared/'jobs.json')
    verify_prepared(prepared,jobs)
    folder = output_dir(out)
    records = []
    for case in jobs['cases']:
        if case['status'] != 'prepared':
            records.append(dict(case_id=case['case_id'],job=None,status='blocked',reason=';'.join(case['blockers'])))
            continue
        for name in case['jobs']:
            src = prepared/case['case_id']/name/'Model.rgg'
            if file_hash(src) != case['model_hashes'][name]:
                raise StudyError('Prepared Model.rgg hash changed')
            target = folder/case['case_id']/name; target.mkdir(parents=True)
            shutil.copyfile(src,target/'Model.rgg')
            started = time.monotonic()
            try:
                _, version = run_groimp(target,core,java)
                parse_log(target/'groimp.log')
                record = dict(case_id=case['case_id'],job=name,status='completed',version=version,reason=None)
            except (StudyError,OSError) as exc:
                record = dict(case_id=case['case_id'],job=name,status='failed',reason=str(exc))
            record['elapsed_seconds']=time.monotonic()-started
            record['log_sha256']=file_hash(target/'groimp.log') if (target/'groimp.log').is_file() else None
            records.append(record)
            write_json(folder/'execution.json',dict(prepared=str(prepared.resolve()),jobs_sha256=file_hash(prepared/'jobs.json'),runs=records))
    result=dict(prepared=str(prepared.resolve()),jobs_sha256=file_hash(prepared/'jobs.json'),runs=records)
    write_json(folder/'execution.json',result)
    return result


def verify_prepared(folder: Path, manifest: dict) -> None:
    if not manifest.get('artifact_hashes'):
        raise StudyError('Prepared artifact hashes are missing')
    for rel, expected in manifest['artifact_hashes'].items():
        path=(folder/rel).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or file_hash(path)!=expected:
            raise StudyError(f'Prepared artifact changed/missing: {rel}')


def collect(execution: Path, out: Path, measurements: Path | None = None) -> dict:
    run = read_json(execution/'execution.json'); prepared=Path(run['prepared'])
    if file_hash(prepared/'jobs.json') != run['jobs_sha256']:
        raise StudyError('Prepared manifest changed since execution')
    jobs = read_json(prepared/'jobs.json'); verify_prepared(prepared,jobs)
    sensors=rows(prepared/'sensors.csv',SENSORS)
    for r in run['runs']:
        if r['status']=='completed':
            path=execution/r['case_id']/r['job']/'groimp.log'
            if not path.is_file() or file_hash(path)!=r.get('log_sha256'):
                raise StudyError('Execution log changed/missing')
    folder = output_dir(out)
    leaf_rows, sensor_rows, summary = [], [], []
    for case in jobs['cases']:
        cid=case['case_id']; cf=prepared/cid
        matching=[r for r in run['runs'] if r['case_id']==cid]
        completed={r['job'] for r in matching if r['status']=='completed'}
        triangles=rows(cf/'triangles.csv',TRIANGLES) if (cf/'triangles.csv').is_file() else []
        fluxes={}
        if 'leaves' in completed:
            fluxes,_=parse_log(execution/cid/'leaves/groimp.log')
            if set(fluxes) != {t['surface_id'] for t in triangles}:
                raise StudyError('Missing/unexpected leaf outputs')
        for t in triangles:
            inc,absorb=fluxes.get(t['surface_id'],(None,None)); area=float(t['area_mm2'])/1e6
            leaf_rows.append(dict(case_id=cid,density_id=case['density_id'],snapshot_date=case['snapshot_date'],**t,
                incident_both_faces_umol_m2_s=inc/area if inc is not None else None,
                absorbed_umol_m2_s=absorb/area if absorb is not None else None))
        for s in sensors:
            val=None; name='sensor_'+s['sensor_id']
            if name in completed:
                _,probes=parse_log(execution/cid/name/'groimp.log')
                expected={s['sensor_id']+':0',s['sensor_id']+':1'}
                if set(probes)!=expected:
                    raise StudyError('Missing/unexpected sensor faces')
                val=sum(probes.values())/(float(s['width_mm'])*float(s['height_mm'])/1e6)
            sensor_rows.append(dict(case_id=cid,sensor_id=s['sensor_id'],height_group=s['height_group'],
                snapshot_date=case['snapshot_date'],predicted_ppfd_umol_m2_s=val,
                scene_sha256=jobs['scene_sha256'],sensor_spec_sha256=jobs['sensor_spec_sha256']))
        part=[r for r in leaf_rows if r['case_id']==cid]
        means={}
        for band in ('lower','upper'):
            group=[r for r in part if r['height_group']==band]
            vals=[r['incident_both_faces_umol_m2_s'] for r in group]
            means[band+'_incident_both_faces_umol_m2_s']=float(np.average(vals,weights=[float(r['area_mm2']) for r in group])) if vals and all(v is not None for v in vals) else None
        plant_means=[]
        for pid in sorted({r['plant_id'] for r in part}):
            group=[r for r in part if r['plant_id']==pid]
            vals=[r['incident_both_faces_umol_m2_s'] for r in group]
            if vals and all(v is not None for v in vals):
                plant_means.append(float(np.average(vals,weights=[float(r['area_mm2']) for r in group])))
        summary.append(dict(case_id=cid,density_id=case['density_id'],snapshot_date=case['snapshot_date'],
            plants_per_m2=case['plants_per_m2'],n_plants=case['n_plants'],**means,
            plant_mean_cv=float(np.std(plant_means)/np.mean(plant_means)) if len(plant_means)==case['n_plants'] and np.mean(plant_means)>0 else None,
            status='simulated_diagnostic' if fluxes else 'unavailable', flowering_result=None, density_optimum=None))
    write_rows(folder/'leaf_triangle_light.csv',leaf_rows,('case_id','density_id','snapshot_date')+TRIANGLES+('incident_both_faces_umol_m2_s','absorbed_umol_m2_s'))
    leaf_groups=defaultdict(list)
    for r in leaf_rows:
        leaf_groups[(r['case_id'],r['plant_id'],r['session_id'],r['organ_id'],r['leaflet_id'])].append(r)
    leaf_summary=[]
    for key,group in leaf_groups.items():
        weights=[float(r['area_mm2']) for r in group]
        vals=[r['incident_both_faces_umol_m2_s'] for r in group]
        absorbed=[r['absorbed_umol_m2_s'] for r in group]
        leaf_summary.append(dict(zip(('case_id','plant_id','session_id','organ_id','leaflet_id'),key),
            n_triangles=len(group),area_mm2=sum(weights),
            incident_both_faces_umol_m2_s=float(np.average(vals,weights=weights)) if all(v is not None for v in vals) else None,
            absorbed_umol_s=float(np.dot(absorbed,weights)/1e6) if all(v is not None for v in absorbed) else None))
    write_rows(folder/'leaf_light.csv',leaf_summary,('case_id','plant_id','session_id','organ_id','leaflet_id','n_triangles','area_mm2','incident_both_faces_umol_m2_s','absorbed_umol_s'))
    write_rows(folder/'sensor_predictions.csv',sensor_rows,('case_id','sensor_id','height_group','snapshot_date','predicted_ppfd_umol_m2_s','scene_sha256','sensor_spec_sha256'))
    write_rows(folder/'density_comparison.csv',summary,('case_id','density_id','snapshot_date','plants_per_m2','n_plants','lower_incident_both_faces_umol_m2_s','upper_incident_both_faces_umol_m2_s','plant_mean_cv','status','flowering_result','density_optimum'))
    validation=validate_ppfd(sensor_rows,rows(measurements,MEASUREMENTS)) if measurements else {'status':'unavailable','reason':'no_independent_measurements'}
    write_json(folder/'ppfd_validation.json',validation)
    write_rows(folder/'ppfd_validation.csv',validation.get('metrics',[]),
        ('height_group','n_readings','mae','rmse','bias','nrmse'))
    write_rows(folder/'ppfd_errors.csv',validation.get('errors',[]),
        ('reading_id','height_group','case_id','sensor_id','measured','error'))
    write_rows(folder/'ppfd_repeats.csv',validation.get('repeat_measurements',[]),
        ('case_id','sensor_id','n_repeats','measured_mean','measured_sd'))
    # A top projection of actual triangles, not a constructed box or surface.
    sections=[]
    for case in jobs['cases']:
        cid=case['case_id']; path=prepared/cid/'triangle_vertices.json'
        if not path.is_file():
            sections.append(f'<h2>{escape(cid)}</h2><p>Unavailable: {escape(str(case["blockers"]))}</p>');continue
        verts=read_json(path); group=[r for r in leaf_rows if r['case_id']==cid]
        xy=np.concatenate([np.array(v)[:,:2] for v in verts.values()])
        lo=xy.min(0); span=np.maximum(xy.max(0)-lo,1)
        maximum=max((r['incident_both_faces_umol_m2_s'] or 0 for r in group),default=0)
        polys=[]
        for r in group:
            v=np.array(verts[r['surface_id']]); points=' '.join(f'{p[0]},{p[1]}' for p in v)
            value=r['incident_both_faces_umol_m2_s']; fraction=value/maximum if maximum and value is not None else 0
            color=f'rgb({int(255*fraction)},70,{int(255*(1-fraction))})' if value is not None else '#aaa'
            polys.append(f'<polygon points="{points}" fill="{color}" stroke="#444" stroke-width="0.15"><title>{escape(str(r))}</title></polygon>')
        sections.append(f'<h2>{escape(cid)}</h2><p>Per-case scale 0–{maximum:.3g}; incident on both faces. Hover for IDs.</p><svg width="600" height="400" viewBox="{lo[0]} {lo[1]} {span[0]} {span[1]}">'+''.join(polys)+'</svg>')
    (folder/'light_maps.html').write_text('<!doctype html><meta charset="utf-8"><title>Diagnostic density light maps</title><h1>Diagnostic triangle maps</h1><p>Top projection; not independently validated PPFD. No flowering or optimum inferred.</p>'+''.join(sections),encoding='utf-8')
    report=dict(status='collected',n_triangle_rows=len(leaf_rows),n_sensor_rows=len(sensor_rows),
        validated_metric=False,suitable_for_ppfd=False,ppfd_validation=validation,
        source_hashes={str(execution/'execution.json'):file_hash(execution/'execution.json')},limitations=jobs['limits'])
    if measurements:
        report['source_hashes'][str(measurements.resolve())]=file_hash(measurements)
    write_json(folder/'results.json',report)
    return report


def validate_ppfd(predictions: list[dict], measured: list[dict]) -> dict:
    lookup={(r['case_id'],r['sensor_id']):r for r in predictions}
    if len(lookup)!=len(predictions):
        raise StudyError('Duplicate prediction IDs')
    seen=set(); differences=[]; missing=[]
    for r in measured:
        if r['reading_id'] in seen:
            raise StudyError('Duplicate reading_id')
        seen.add(r['reading_id'])
        if r['split_role'] not in ('calibration','validation'):
            raise StudyError('Unknown measurement role')
        p=lookup.get((r['case_id'],r['sensor_id']))
        if not p:
            raise StudyError('Measurement has no planned sensor/case')
        for field in ('scene_sha256','sensor_spec_sha256'):
            if r[field]!=p[field]:
                raise StudyError(f'Measurement {field} mismatch')
        if r['date']!=p['snapshot_date']:
            raise StudyError('Measurement date differs from geometry snapshot')
        if not r['sensor_model'] or not r['calibration_reference']:
            raise StudyError('Sensor provenance is required')
        if r.get('all_fixtures_full_output') != 'true':
            raise StudyError('Measurement must match the all-fixtures-full-output simulation')
        val=number(r['ppfd_umol_m2_s'],'measured PPFD',missing=True)
        pred=number(p['predicted_ppfd_umol_m2_s'],'predicted PPFD',missing=True)
        if val is not None and val<0:
            raise StudyError('Negative measured PPFD')
        if r['split_role']=='validation':
            if val is None or pred is None:
                missing.append(r['reading_id'])
            else:
                differences.append(dict(reading_id=r['reading_id'],height_group=p['height_group'],case_id=p['case_id'],sensor_id=p['sensor_id'],measured=val,error=pred-val))
    groups=[]
    for band in ['all']+sorted({r['height_group'] for r in differences}):
        group=[r for r in differences if band=='all' or r['height_group']==band]
        err=np.array([r['error'] for r in group]); mean=np.mean([r['measured'] for r in group]) if group else 0
        groups.append(dict(height_group=band,n_readings=len(group),
            mae=float(np.abs(err).mean()) if group else None,rmse=float(np.sqrt(np.mean(err**2))) if group else None,
            bias=float(err.mean()) if group else None,nrmse=float(np.sqrt(np.mean(err**2))/mean) if group and mean>0 else None))
    repeated=[]
    for key in sorted({(r['case_id'],r['sensor_id']) for r in differences}):
        values=[r['measured'] for r in differences if (r['case_id'],r['sensor_id'])==key]
        repeated.append(dict(case_id=key[0],sensor_id=key[1],n_repeats=len(values),
            measured_mean=float(np.mean(values)),measured_sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    return dict(status='evaluated' if differences else 'unavailable',metrics=groups,repeat_measurements=repeated,
        n_planned_sensor_cases=len(predictions),n_covered_sensor_cases=len(repeated),
        missing_validation_reading_ids=missing,n_calibration_excluded=sum(r['split_role']=='calibration' for r in measured),
        errors=differences,independent_plot_inference='unavailable',
        note='Raw errors only; this command never fits LED parameters to evaluation readings.')


def readiness_xl(handoff: Path, out: Path) -> dict:
    """Inspect the pinned existing XL bundle without converting display geometry."""
    from tools.groimp_light_scene import verify_export
    manifest=read_json(handoff/'handoff.json'); records=[]
    for pid in manifest['plants']:
        try:
            export,hashes=verify_export(handoff,pid)
            report=read_json(export/'export.json')
            count=int(report.get('geometry_counts',{}).get('leaf_mesh',0))
            records.append(dict(plant_id=pid,session_id=report['session_id'],leaf_surface_count=count,
                status='requires_surface_and_light_review' if count else 'blocked_no_leaf_surfaces',
                export_hashes=hashes,validated_metric=False,suitable_for_ppfd=False))
        except (StudyError,OSError) as exc:
            records.append(dict(plant_id=pid,status='blocked_export_integrity',reason=str(exc)))
    folder=output_dir(out)
    result=dict(source_handoff=str(handoff.resolve()),source_sha256=file_hash(handoff/'handoff.json'),
        plants=records,co2_recorded=False,primary_endpoint='first_open_flower',
        required=['accepted real leaf triangle surfaces','reviewed directional LED adapter and empty-scene validation',
                  'independent oriented sensor readings','dated plot/plant growth and first-open-flower visits'])
    write_json(folder/'readiness.json',result)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--templates',action='store_true')
    action.add_argument('--readiness-xl',type=Path,metavar='XL_HANDOFF_DIR')
    action.add_argument('--prepare',type=Path,metavar='STUDY_JSON')
    action.add_argument('--execute',type=Path,metavar='PREPARED_DIR')
    action.add_argument('--collect',type=Path,metavar='EXECUTION_DIR')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--core',type=Path,default=CORE);p.add_argument('--java',default='java')
    p.add_argument('--measurements',type=Path)
    a=p.parse_args()
    if a.templates: result=templates(a.out)
    elif a.readiness_xl: result=readiness_xl(a.readiness_xl,a.out)
    elif a.prepare: result=prepare(a.prepare,a.out)
    elif a.execute: result=execute(a.execute,a.out,a.core,a.java)
    else: result=collect(a.collect,a.out,a.measurements)
    print(result)


if __name__=='__main__':
    main()
