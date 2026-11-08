"""Session D native experiments; immutable inputs, one changed factor, no growth."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from tomato3d.light.experiments import paired_statistics, convergence, signature
from tomato3d.light.geometry import triangle_areas_mm2
from tomato3d.light.scene import load_scene
from tomato3d.light.shape_controls import CurvatureInput, flat_geometry, ellipse_same_area_aspect, normalize_outline
from tools.leaf_light.common import StudyError, digest, file_hash, read_json, write_json, write_rows, output_dir
from tools.ppfd_validation import lamp_adapter, make_job_rgg
from tools.groimp_density import execute as execute_jobs, parse_log, verify_prepared

SCHEMA = 'p2-light-experiment-1'
ENSEMBLES = ('ray_noise','outline_selection','structural_member','paired_joint')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise StudyError(reason)


def ref(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=file_hash(path))


def source(record: dict, sources: dict) -> Path:
    p = Path(record['path']).resolve()
    require(file_hash(p)==record['sha256'], 'source hash mismatch: '+str(p))
    sources[str(p)] = record['sha256']
    return p


def code_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = ('tools/p2_light_experiments.py','tomato3d/light/experiments.py',
        'tomato3d/light/shape_controls.py','tomato3d/light/geometry.py','tomato3d/light/scene.py',
        'tomato3d/light/frames.py','tomato3d/io/schemas.py','tools/ppfd_validation.py',
        'tools/groimp_density.py','tools/groimp_light_sanity.py','tools/leaf_light/common.py')
    return {str(root/p):file_hash(root/p) for p in paths}


def materialize(snapshot: dict, rep: dict, shape: str, segments: int, spacing: float) -> tuple[list, dict]:
    """Draw once from supplied local meshes; reuse those exact draws across scenarios."""
    require(snapshot['frame']=='grow_mm' and snapshot['area_basis']=='full_outline', 'geometry frame/area contract')
    require(snapshot['source_kind'] in ('fixture_truth','reviewed_optical_surface'), 'physical surface provenance required')
    CurvatureInput(**snapshot['curvature']).validate()
    plants = snapshot['plant_origins_grow_mm']
    rotation_rng = np.random.default_rng(rep['rotation_seed'])
    rotation_mode = snapshot.get('rotation_mode', 'seeded_yaw')
    require(rotation_mode in ('seeded_yaw', 'preserve_snapshot'), 'unknown rotation mode')
    rotations = {p:(float(rotation_rng.uniform(0,360)) if rotation_mode=='seeded_yaw' else 0.)
                 for p in sorted(plants)}
    outline_rng = np.random.default_rng(rep['outline_seed'])
    output = []; choices = {}; seen = set()
    for leaf in sorted(snapshot['leaves'], key=lambda s:(s['plant_id'],s['leaf_id'],s['leaflet_id'])):
        key = '|'.join(leaf[k] for k in ('plant_id','leaf_id','leaflet_id'))
        require(key not in seen, 'duplicate leaflet namespace'); seen.add(key)
        pool = leaf['outline_pool']
        require(bool(pool) and len({x['id'] for x in pool})==len(pool), 'outline pool missing/duplicated')
        choice = pool[int(outline_rng.integers(len(pool)))]; choices[key]=choice['id']
        local = normalize_outline(np.asarray(choice['triangles_mm']),leaf['target_area_mm2'],leaf['target_length_width_ratio'])
        area = float(triangle_areas_mm2(local).sum())
        if shape=='ellipse':
            local = ellipse_same_area_aspect(local, segments)
        require(shape in ('outline','ellipse'), 'unknown shape treatment')
        r = np.asarray(leaf['local_rotation'],dtype=float)
        require(r.shape==(3,3) and np.allclose(r.T@r,np.eye(3),atol=1e-10,rtol=1e-10)
                and np.isclose(np.linalg.det(r),1), 'proper local rotation required')
        center = np.asarray(leaf['center_grow_mm'],dtype=float)
        origin = np.asarray(plants[leaf['plant_id']],dtype=float)
        require(center.shape==origin.shape==(3,) and np.isfinite(center).all() and np.isfinite(origin).all(), 'invalid placement')
        angle = np.deg2rad(rotations[leaf['plant_id']]); c,s=np.cos(angle),np.sin(angle)
        rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        # Spacing changes plant origins only; shape, local leaf placement and orientation stay paired.
        placed = (local@r.T+center-origin)@rz.T+origin*np.array([spacing,spacing,1.])
        output.append(dict(plant_id=leaf['plant_id'],leaf_id=leaf['leaf_id'],leaflet_id=leaf['leaflet_id'],
            rank=leaf['rank'],position=leaf['position'],entity_id=key,area_m2=area/1e6,triangles_mm=placed.tolist()))
    require(bool(output), 'empty geometry')
    return output, dict(ray_seed=rep['ray_seed'],rotation_seed=rep['rotation_seed'],
        rotation_mode=rotation_mode,rotations_deg=rotations,outline_seed=rep['outline_seed'],outline_choices=choices,
        member_seed=rep['member_seed'],snapshot_sha256=rep['snapshot_ref']['sha256'])


def validate_design(plan: dict) -> None:
    require(plan['schema_version']==SCHEMA, 'unsupported plan')
    require(plan['scope'] in ('fixture','real'), 'declare fixture/real scope')
    require(plan['factor'] in ('shape','lamp_height_delta_mm','spacing_scale'), 'unsupported factor (fertilizer has no light geometry law)')
    scenarios = plan['scenarios']; baseline = plan['baseline']
    require(len(scenarios)>=1 and len({s['id'] for s in scenarios})==len(scenarios), 'scenario inventory')
    require(baseline in {s['id'] for s in scenarios}, 'baseline absent')
    for s in scenarios:
        require(set(s)=={'id','value'}, 'only one scenario factor is allowed')
        if plan['factor']=='shape': require(s['value'] in ('outline','ellipse'), 'unknown shape')
        else: require(np.isfinite(s['value']) and (plan['factor']!='spacing_scale' or s['value']>0), 'invalid treatment')
    counts=plan['ray_counts']
    require(counts==sorted(set(counts)) and all(type(n) is int and n>0 for n in counts), 'ordered positive unique ray counts')
    require(type(plan['max_depth']) is int and plan['max_depth']>0, 'positive max depth')
    require(plan['target_cv'] is None or 0<plan['target_cv']<1, 'invalid CV threshold')
    require(plan['threshold_source_kind']=='user_assumed', 'CV criterion is an assumption')
    require(len({e['kind'] for e in plan['ensembles']})==len(plan['ensembles']), 'duplicate ensemble')
    for e in plan['ensembles']:
        require(e['kind'] in ENSEMBLES and len(e['replicates'])>=2, 'ensemble requires >=2 replicates')
        reps=e['replicates']
        require(len({r['rep_id'] for r in reps})==len(reps), 'duplicate repeat')
        for r in reps:
            for k in ('ray_seed','rotation_seed','outline_seed','member_seed'):
                require(type(r[k]) is int and 0<=r[k]<=2147483647, 'explicit 32-bit seeds required')
        changing={'ray_noise':'ray_seed','outline_selection':'outline_seed','structural_member':'member_seed'}.get(e['kind'])
        if changing:
            fixed=['ray_seed','rotation_seed','outline_seed','member_seed','snapshot_ref']
            fixed.remove(changing)
            if changing=='member_seed': fixed.remove('snapshot_ref')
            require(all(len({signature(r[k]) for r in reps})==1 for k in fixed), 'uncertainty sources mixed')
            require(len({r[changing] for r in reps})==len(reps), 'varying seeds must be unique')
    jobs=sum(len(e['replicates']) for e in plan['ensembles'])*len(counts)*len(scenarios)
    rays=sum(len(e['replicates']) for e in plan['ensembles'])*sum(counts)*len(scenarios)
    require(jobs<=plan['limits']['max_jobs'] and rays<=plan['limits']['max_total_rays'], 'declared native budget exceeded')


def prepare(plan_path: Path, out: Path) -> dict:
    started=time.monotonic(); plan=read_json(plan_path); validate_design(plan)
    sources={str(plan_path.resolve()):file_hash(plan_path)}
    scene_path=source(plan['scene_ref'],sources); scene=load_scene(scene_path)
    rig=read_json(source(plan['rig_ref'],sources))
    for record in plan.get('evidence_refs',{}).values(): source(record,sources)
    if plan['scope']=='real':
        required={'tray_registration','optical_scene_review','outline_pool_review','mt_aspect_review'}
        require(required<=set(plan.get('evidence_refs',{})), 'real D3 needs tray/optics/outline/aspect reviews')
        for key in required:
            review=read_json(Path(plan['evidence_refs'][key]['path']))
            require(review.get('review_state')=='accepted' and review.get('source_kind') in ('measured','reviewed'), 'unreviewed real input: '+key)
    optics=tuple(getattr(scene.leaf_optics,k).value for k in ('absorptance','reflectance','transmittance'))
    require(all(v is not None for v in optics) and np.isclose(sum(optics),1.), 'complete native optics required')
    jobs=[]; cases=[]; models={}; sources.update(code_hashes())
    folder=output_dir(out)
    for e in plan['ensembles']:
        for rep in e['replicates']:
            snapshot=read_json(source(rep['snapshot_ref'],sources))
            require(plan['scope']!='real' or snapshot['source_kind']=='reviewed_optical_surface', 'real scene cannot use fixture geometry')
            for scenario in plan['scenarios']:
                shape=scenario['value'] if plan['factor']=='shape' else 'outline'
                spacing=scenario['value'] if plan['factor']=='spacing_scale' else 1.
                leaves,controls=materialize(snapshot,rep,shape,plan['ellipse_segments'],spacing)
                require(sum(len(x['triangles_mm']) for x in leaves)<=plan['limits']['max_triangles'], 'triangle budget exceeded')
                copied=scene.model_copy(deep=True)
                if plan['factor']=='lamp_height_delta_mm':
                    fixtures=[]
                    for fx in copied.fixtures:
                        v=fx.emitting_center_grow_mm.value
                        require(v is not None,'lamp location unavailable')
                        center=fx.emitting_center_grow_mm.model_copy(update=dict(
                            value=(v[0],v[1],v[2]+scenario['value']),status='scenario',source='Session D height treatment'))
                        fixtures.append(fx.model_copy(update=dict(emitting_center_grow_mm=center)))
                    copied=copied.model_copy(update=dict(fixtures=fixtures))
                adapter=lamp_adapter(scene_path,copied)
                require(not adapter['unsupported_fixtures'],'unsupported native lamp')
                for rays in plan['ray_counts']:
                    cid=f'case_{len(cases):05d}'; dest=folder/cid/'transport'; dest.mkdir(parents=True)
                    triangles=[]; mapping=[]
                    for leaf in leaves:
                        for triangle in leaf['triangles_mm']:
                            tid=f't{len(triangles):07d}'
                            triangles.append((dict(surface_id=tid),np.asarray(triangle)))
                            mapping.append(dict(triangle_id=tid,**{k:v for k,v in leaf.items() if k!='triangles_mm'}))
                    model=make_job_rgg(adapter['initialization_rgg'],rays,plan['max_depth'],rep['ray_seed'],rig,triangles,optics)
                    (dest/'Model.rgg').write_text(model,encoding='utf-8')
                    h=file_hash(dest/'Model.rgg'); models[f'{cid}/transport/Model.rgg']=h
                    cases.append(dict(case_id=cid,status='prepared',blockers=[],jobs=['transport'],model_hashes={'transport':h}))
                    jobs.append(dict(case_id=cid,ensemble=e['kind'],rep_id=rep['rep_id'],scenario=scenario['id'],
                        rays=rays,controls=controls,geometry_hash=digest(leaves),triangles=mapping))
    write_json(folder/'jobs.json',dict(schema_version='groimp-density-jobs-1',cases=cases,artifact_hashes=models))
    manifest=dict(schema_version=SCHEMA,plan=plan,source_hashes=sources,jobs=jobs,
        jobs_sha256=file_hash(folder/'jobs.json'),prepare_elapsed_s=time.monotonic()-started,
        limitations=['Native total emitted rays, not hit count per leaflet.',
            'Equal seed is shared RNG initialization; geometry-dependent branching may decorrelate paths.',
            'Absorbed grey photon-equivalent transport, not spectral/Micro-Tom validation.',
            'No shape replacement in production; curvature remains disabled.'])
    write_json(folder/'experiment.json',manifest)
    return manifest


def fresh(prepared: Path) -> dict:
    m=read_json(prepared/'experiment.json')
    for path,h in m['source_hashes'].items(): require(file_hash(Path(path))==h,'stale source/code: '+path)
    require(file_hash(prepared/'jobs.json')==m['jobs_sha256'],'changed jobs')
    verify_prepared(prepared,read_json(prepared/'jobs.json'))
    return m


def execute(prepared: Path, out: Path, core: Path, java: str) -> dict:
    fresh(prepared)
    result=execute_jobs(prepared,out,core,java)
    write_json(out/'runtime.json',dict(core=str(core.resolve()),core_sha256=file_hash(core),java=java,
        experiment_sha256=file_hash(prepared/'experiment.json')))
    return result


def collect(prepared: Path, execution: Path, out: Path) -> dict:
    m=fresh(prepared); run=read_json(execution/'execution.json')
    require(run['jobs_sha256']==m['jobs_sha256'] and Path(run['prepared']).resolve()==prepared.resolve(),'execution lineage mismatch')
    runtime=read_json(execution/'runtime.json')
    require(runtime['experiment_sha256']==file_hash(prepared/'experiment.json'),'runtime experiment mismatch')
    require(file_hash(Path(runtime['core']))==runtime['core_sha256'],'native core changed')
    done={(r['case_id'],r['job']):r for r in run['runs']}
    require(len(done)==len(run['runs']),'duplicate execution record')
    records=[]; timing=[]
    for job in m['jobs']:
        rec=done.get((job['case_id'],'transport'),{}); vals={}
        if rec.get('status')=='completed':
            log=execution/job['case_id']/'transport/groimp.log'
            require(file_hash(log)==rec['log_sha256'],'changed log')
            vals,probes=parse_log(log)
            require(not probes and set(vals)=={t['triangle_id'] for t in job['triangles']},'native triangle inventory mismatch')
        timing.append(dict(case_id=job['case_id'],ensemble=job['ensemble'],rays=job['rays'],
            scenario=job['scenario'],rep_id=job['rep_id'],elapsed_s=rec.get('elapsed_seconds'),status=rec.get('status','not_executed')))
        for level,keys in (('leaflet',('plant_id','leaf_id','leaflet_id')),('leaf',('plant_id','leaf_id')),('plant',('plant_id',))):
            groups=defaultdict(list)
            for t in job['triangles']: groups['|'.join(t[k] for k in keys)].append(t)
            for entity,group in groups.items():
                flux=sum(vals[t['triangle_id']][1] for t in group) if vals else None
                # A leaflet area is stored once per leaflet, never once per triangle.
                area=sum({t['entity_id']:t['area_m2'] for t in group}.values())
                records.append({k:job[k] for k in ('ensemble','rep_id','scenario','rays','controls','geometry_hash')} |
                    dict(level=level,entity_id=entity,area_m2=area,absorbed_umol_s=flux,
                         absorbed_umol_m2_s=flux/area if flux is not None else None,
                         rank=group[0]['rank'] if level!='plant' else None,
                         position=';'.join(sorted({str(t['position']) for t in group})) if level!='plant' else None))
    plan=m['plan']
    deltas,statistics=paired_statistics(records,plan['baseline'],plan['bootstrap_seed'],plan['bootstrap_draws'])
    cv,minimum=convergence(records,plan['baseline'],plan['target_cv'])
    folder=output_dir(out)
    uncertainty=[]
    for kind in ENSEMBLES[:3]:
        grouped=defaultdict(list)
        for r in records:
            if r['ensemble']==kind and r['scenario']==plan['baseline']:
                grouped[(r['rays'],r['level'],r['entity_id'])].append(r['absorbed_umol_s'])
        for key,values in grouped.items():
            valid=len(values)>1 and all(v is not None for v in values)
            uncertainty.append(dict(ensemble=kind,rays=key[0],level=key[1],entity_id=key[2],
                n_repeats=len(values),mean_umol_s=float(np.mean(values)) if valid else None,
                sd_umol_s=float(np.std(values,ddof=1)) if valid else None,status='estimated' if valid else 'unavailable'))
    tables=dict(flux=records,paired_deltas=deltas,paired_statistics=statistics,convergence=cv,timing=timing,uncertainty=uncertainty)
    for name,table in tables.items():
        simple=[{k:(signature(v) if isinstance(v,dict) else v) for k,v in r.items()} for r in table]
        write_rows(folder/(name+'.csv'),simple,list(simple[0]) if simple else ['status'])
    report=dict(schema_version=SCHEMA,scope=plan['scope'],status='complete' if all(r['absorbed_umol_s'] is not None for r in records) else 'incomplete',
        convergence=minimum,total_native_elapsed_s=sum(r['elapsed_s'] or 0 for r in timing),
        jobs=len(timing),uncertainty_sources_separate=True,
        unavailable_uncertainty_sources=[k for k in ENSEMBLES[:3] if k not in {e['kind'] for e in plan['ensembles']}],
        shape_model_changed=False,
        physical_accuracy_validated=False,curvature=CurvatureInput().record(),
        source_hashes=m['source_hashes'],execution_ref=ref(execution/'execution.json'),
        experiment_ref=ref(prepared/'experiment.json'),runtime_ref=ref(execution/'runtime.json'),
        artifact_hashes={p.name:file_hash(p) for p in folder.iterdir()})
    write_json(folder/'report.json',report)
    return report


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('prepare','execute','collect'):
        q=sub.add_parser(cmd);q.add_argument('--input',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
        if cmd=='execute': q.add_argument('--core',type=Path,required=True);q.add_argument('--java',required=True)
        if cmd=='collect': q.add_argument('--execution',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='prepare': r=prepare(a.input,a.out)
        elif a.command=='execute': r=execute(a.input,a.out,a.core,a.java)
        else: r=collect(a.input,a.execution,a.out)
        print({k:r[k] for k in ('status','jobs','convergence') if k in r and k!='jobs'});return 0
    except (ValueError,StudyError,KeyError,OSError) as exc: p.exit(2,str(exc)+'\n')


if __name__=='__main__': raise SystemExit(main())
