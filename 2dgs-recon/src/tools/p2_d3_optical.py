"""Native optical continuation for an accepted D3 addendum2 geometry package."""
from __future__ import annotations
import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import re
import numpy as np
from tomato3d.light.geometry import triangle_areas_mm2
from tomato3d.light.experiments import paired_statistics, convergence
from tomato3d.synth.leaflet_surface import surface_points
from tomato3d.state.fixture_truth import _orient_adaxial
from tools.p2_d3_quality import Settings, constrained
from tools.p2_d3_meshes import export_vertices, ellipse, SCOPE
from tools.p2_d3_gated import HELD, table,mesh_hash
from tools.ppfd_validation import RGG_HEADER, MEASURE
from tools.groimp_light_sanity import run_groimp
from tools.groimp_density import parse_log
from tools.leaf_light.common import read_json,write_json,file_hash
from tools.p2_light_experiments import ref
from tools.p2_d3_verify import archive_inputs, verify, input_path, read_rows


def model(geometry: Path, lamps: list[dict], rays: int, seed: int) -> str:
    """Same native transport as existing adapter; split large Java string constants."""
    if rays <= 0: raise ValueError('positive ray count required')
    rows = list(csv.reader(geometry.open(encoding='utf-8')))
    ids = [r[0] for r in rows]
    if len(ids) != len(set(ids)): raise ValueError('duplicate native identity')
    chunks = []; starts = [0]
    for row in rows:
        numbers = np.asarray(row[1:],float)
        if len(numbers)%9 or not np.isfinite(numbers).all(): raise ValueError('finite complete triangles required')
        value = ','.join(format(x,'.9g') for x in numbers)
        chunks.extend(value[i:i+50000] for i in range(0,len(value),50000)); starts.append(len(chunks))
    declarations = 'String[] surfaceIds={'+','.join(json.dumps(x) for x in ids)+'};\n'
    declarations += 'String[] coordinateChunks={'+','.join(json.dumps(x) for x in chunks)+'};\n'
    declarations += 'int[] starts={'+','.join(str(x) for x in starts)+'};\n'
    nodes = []
    for lamp in lamps:
        x,y,z,p = [float(lamp[k]) for k in ('x_m','y_m','z_m','power_umol_s')]
        if not np.isfinite([x,y,z,p]).all() or p <= 0: raise ValueError('invalid point lamp')
        nodes.append(f'[ Translate({x},{y},{z}) Lamp().(setLight(new PointLight().(setPower({p}f)))) ]')
    body = '''
FloatList surfaceVertices(int index) {
    StringBuilder joined=new StringBuilder();
    for(int j=starts[index];j<starts[index+1];j++)joined.append(coordinateChunks[j]);
    String[] parts=joined.toString().split(",");
    float[] coords=new float[parts.length];
    for(int k=0;k<parts.length;k++)coords[k]=Float.parseFloat(parts[k]);
    return new FloatList(coords);
}
protected void init()
[
    Axiom ==> @LAMPS@
      for(int j=0;j<surfaceIds.length;j++)
        ([LeafSurface(surfaceIds[j]).(setPolygons(getMesh(surfaceVertices(j)).getPolygons()),setShader(BLACK))]);
]
'''.replace('@LAMPS@',' '.join(nodes))
    measure = MEASURE.replace('@SEED@',str(seed)).replace('    lm.compute();',
        '    long traceStarted=System.nanoTime();\n    lm.compute();\n    println("TRANSPORT_SECONDS\\t" + ((System.nanoTime()-traceStarted)/1.0e9));')
    return RGG_HEADER.replace('@RAYS@',str(rays)).replace('@DEPTH@','3')+declarations+body+measure


def enforce_budget(jobs: list[tuple], inventory: list[dict], limits: dict) -> dict:
    for key in ('max_jobs','max_total_rays','max_mesh_triangles'):
        if type(limits.get(key)) is not int or limits[key]<=0: raise ValueError('explicit positive native budget required')
    total = sum(rays for _,rays,_ in jobs)
    if len(jobs)>limits['max_jobs'] or total>limits['max_total_rays']: raise ValueError('declared native budget exceeded')
    maximum = 0
    for item in inventory:
        with Path(item['path']).open(encoding='utf-8') as stream:
            triangles=sum((len(row)-1)//9 for row in csv.reader(stream))
        maximum=max(maximum,triangles)
    if maximum>limits['max_mesh_triangles']: raise ValueError('declared native triangle budget exceeded')
    return dict(planned_jobs=len(jobs),total_requested_rays=total,maximum_mesh_triangles=maximum,limits=limits)


def accepted(results: Path) -> list[dict]:
    verify(results)
    report = read_json(results/'preparation_report.json')
    if report['status'] != 'mesh_preflight_passed' or report['failed_checks']:
        raise ValueError('D3 mesh gate not passed; no native job authorized by this package')
    inventory = read_json(results/'mesh_inventory.json')
    if any(not r['eligible_for_native'] or r['surface_count'] != 129 for r in inventory):
        raise ValueError('ineligible or partial canopy')
    return inventory


def storage_budget(run_dir: Path, limits: dict, reserve_bytes: int = 0) -> dict:
    """Refuse the next job before its bounded disk reservation exceeds limits."""
    for key in ('max_run_bytes','min_free_disk_bytes'):
        if type(limits.get(key)) is not int or limits[key] <= 0:
            raise ValueError('explicit positive storage budget required')
    if reserve_bytes < 0: raise ValueError('negative disk reservation')
    used=sum(p.stat().st_size for p in run_dir.rglob('*') if p.is_file())
    free=shutil.disk_usage(run_dir).free
    if used+reserve_bytes>limits['max_run_bytes'] or free-reserve_bytes<limits['min_free_disk_bytes']:
        raise ValueError('storage budget exceeded; existing evidence retained, no next native job')
    return dict(run_logical_bytes=used,free_disk_bytes=free,reserve_bytes=reserve_bytes)


def area_from_verified_fine(record: dict,contour: np.ndarray,coarse_scale: float,fine_quality: dict) -> float:
    """Rebuild the recorded triangulation and require exact accepted mesh identity.

    Rechecking all boundary predicates is redundant once the rebuilt fine mesh
    hash matches. Then change only uniform scale for the convergence diagnostic.
    """
    import triangle
    c=np.asarray(contour,float)
    result=triangle.triangulate(dict(vertices=c,segments=np.column_stack((np.arange(len(c)),np.roll(np.arange(len(c)),-1)))),fine_quality['options'])
    uv=np.asarray(result['vertices']);faces=np.asarray(result['triangles'],int)
    matrix=np.asarray(record['canopy_transform'])
    params=dict(record['parameters']);params['length_mm']*=float(fine_quality['scale'])
    vertices=surface_points(uv,params);oriented=_orient_adaxial(vertices,faces,np.asarray(params['up']))
    tri=export_vertices(vertices,matrix)[oriented]
    if mesh_hash(tri)!=fine_quality['mesh_sha256']:
        raise ValueError('fine triangulation reconstruction does not match accepted mesh hash')
    params['length_mm']=record['parameters']['length_mm']*coarse_scale
    vertices=surface_points(uv,params);oriented=_orient_adaxial(vertices,faces,np.asarray(params['up']))
    return float(triangle_areas_mm2(export_vertices(vertices,matrix)[oriented]).sum())


def fixed_scale_area(results: Path, recovery: Path) -> list[dict]:
    """Refinement diagnostic uses the coarse scale, never two renormalized areas."""
    settings = Settings(**read_json(results/'settings.json')['numerical'])
    records = {r['surface_id']:r for r in read_json(recovery/'recovered_parameters.json')['leaflets']}
    contours = {r['id']:np.asarray(r['contour']) for r in read_json(results/'normalized_pool.json')['leaflets']}
    draws = {(r['pool_size'],r['outline_seed'],r['surface_id']):r['outline_id'] for r in csv.DictReader((results/'outline_draws.csv').open(encoding='utf-8-sig'))}
    quality = list(csv.DictReader((results/'mesh_quality.csv').open(encoding='utf-8-sig')))
    groups = {}
    for q in quality:
        groups.setdefault(tuple(q[k] for k in ('pool_size','outline_seed','treatment','surface_id')),[]).append(q)
    output = []
    for key,group in groups.items():
        group.sort(key=lambda r:int(r['grid_n']))
        for coarse,fine in zip(group,group[1:]):
            r = records[key[3]]
            if key[3] in HELD:
                delta = 0.; fine_area = float(coarse['area_mm2'])
            else:
                c = np.asarray(r['contour']) if key[2]=='source_regenerated' else contours[draws[(key[0],key[1],key[3])]]
                if key[2]=='ellipse': c = ellipse(c,96)
                fine_area = area_from_verified_fine(r,c,float(coarse['scale']),fine)
                delta = fine_area/float(coarse['area_mm2'])-1
            output.append(dict(scope=SCOPE,pool_size=int(key[0]),outline_seed=int(key[1]),treatment=key[2],surface_id=key[3],
                coarse_grid=int(coarse['grid_n']),fine_grid=int(fine['grid_n']),coarse_scale=float(coarse['scale']),
                coarse_area_mm2=float(coarse['area_mm2']),fine_area_at_coarse_scale_mm2=fine_area,
                relative_area_change=delta,below_0_1pct=abs(delta)<.001))
    return output


def aggregate(values: dict, metadata: list[dict], areas_m2: dict, controls: dict,
              scenario: str, rays: int, seed: int, geometry_hash: str) -> list[dict]:
    if set(values) != {r['native_id'] for r in metadata}: raise ValueError('native surface inventory mismatch')
    output = []
    for level,keys in [('leaflet',('plant_id','leaf_id','leaflet_id')),('leaf',('plant_id','leaf_id')),('plant',('plant_id',))]:
        groups = {}
        for r in metadata: groups.setdefault('|'.join(r[k] for k in keys),[]).append(r)
        for key,group in groups.items():
            absorbed = sum(values[r['native_id']][1] for r in group)
            area = sum(areas_m2[r['native_id']] for r in group)
            output.append(dict(scope=SCOPE,ensemble='ray_noise',rep_id=str(seed),seed=seed,rays=rays,scenario=scenario,
                level=level,entity_id=key,plant_id=group[0]['plant_id'],rank=group[0]['rank'] if level!='plant' else None,
                position=';'.join(sorted({r['position'] for r in group})),controls=controls,geometry_hash=geometry_hash,
                absorbed_umol_s=absorbed,area_m2=area,absorbed_umol_m2_s=absorbed/area,
                held_count=sum(r['surface_id'] in HELD for r in group),
                outline_statistics_included=level!='leaflet' or group[0]['surface_id'] not in HELD))
    return output


def job_name(item: dict, rays: int, seed: int) -> tuple[str,str]:
    name=f"p{item['pool_size']}_o{item['outline_seed']}_g{item['grid_n']}_{item['treatment']}"
    return name,f'{name}_r{rays}_s{seed}'


def checked_prefix(jobs: list[tuple], times: list[dict]) -> None:
    expected=[job_name(*job)[1] for job in jobs]
    cases=[r['case'] for r in times]
    if cases!=expected[:len(cases)] or len(cases)>len(expected) or len(set(cases))!=len(cases):
        raise ValueError('resume evidence is not the exact complete job prefix')
    for (_,rays,seed),row in zip(jobs,times):
        if int(row['rays'])!=rays or int(row['seed'])!=seed:
            raise ValueError('resume rays/seed mismatch')


def import_prefix(previous: Path, root: Path, jobs: list[tuple], metadata: list[dict],
                  lamps: list[dict], limits: dict) -> tuple[list[dict],list[dict]]:
    """Rebuild expected models and sums from raw evidence; never trust cached flux."""
    previous=previous.resolve()
    old=read_json(previous/'optical'/'execution_settings.json')
    new=read_json(root/'execution_settings.json')
    for key in ('scope','ray_counts','seeds','lamps','java_executable','optics'):
        if old[key]!=new[key]: raise ValueError('resume optical settings differ: '+key)
    archive=read_json(previous/'archived_inputs.json')['by_sha256']
    old_inventory=read_json(previous/'mesh_inventory.json')
    old_by_name={job_name(i,1,1)[0]:i['sha256'] for i in old_inventory}
    for item,_,_ in jobs:
        if old_by_name.get(job_name(item,1,1)[0])!=item['sha256']:
            raise ValueError('resume geometry hash mismatch')
    for source in old['sources'].values():
        frozen=input_path(previous,source,archive)
        if file_hash(frozen)!=source['sha256']:raise ValueError('resume frozen source hash mismatch')
        if Path(source['path']).name!='p2_d3_optical.py':
            current=[s for s in new['sources'].values() if Path(s['path']).resolve()==Path(source['path']).resolve()]
            if len(current)!=1 or current[0]['sha256']!=source['sha256']:
                raise ValueError('resume transport dependency changed')
        else:
            # The controller may gain recovery logic. Geometry serialization and
            # native model generation/aggregation must remain AST-identical.
            def bodies(text):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body
                        if isinstance(n,ast.FunctionDef) and n.name in ('model','aggregate')}
            if bodies(frozen.read_text(encoding='utf-8'))!=bodies(Path(__file__).read_text(encoding='utf-8')):
                raise ValueError('resume model/aggregation algorithm changed')
    times=read_rows(previous/'optical'/'timing.csv');checked_prefix(jobs,times)
    records=[];typed_times=[];raw=root/'native_evidence';raw.mkdir(exist_ok=True)
    raw_bytes=sum((previous/'optical'/'native_evidence'/(r['case']+suffix)).stat().st_size
                  for r in times for suffix in ('.rgg.gz','.log'))
    storage_budget(root.parent.parent,limits,raw_bytes)
    cache_key=None;template='';template_seed=None;areas={}
    for (item,rays,seed),row in zip(jobs,times):
        name,case=job_name(item,rays,seed);geometry=Path(item['path'])
        key=(item['sha256'],rays)
        if key!=cache_key:
            if file_hash(geometry)!=item['sha256']:raise ValueError('resume current mesh changed')
            template=model(geometry,lamps,rays,seed);template_seed=seed;cache_key=key
            areas={r[0]:float(triangle_areas_mm2(np.asarray(r[1:],float).reshape(-1,3,3)*1000).sum())/1e6
                   for r in csv.reader(geometry.open(encoding='utf-8'))}
        text=template.replace(f'lm.setSeed({template_seed}L)',f'lm.setSeed({seed}L)')
        # Match Path.write_text's platform newline translation byte-for-byte.
        expected=hashlib.sha256(text.replace('\n',os.linesep).encode('utf-8')).hexdigest()
        old_raw=previous/'optical'/'native_evidence'
        with gzip.open(old_raw/(case+'.rgg.gz'),'rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if expected!=row['model_sha256'] or actual!=expected:
            raise ValueError('resume exact native model hash mismatch')
        values,probes=parse_log(old_raw/(case+'.log'))
        log=(old_raw/(case+'.log')).read_text(encoding='utf-8',errors='replace')
        traces=re.findall(r'TRANSPORT_SECONDS\s+([0-9.eE+-]+)',log)
        if probes or 'SANITY_DONE' not in log or len(traces)!=1 or float(traces[0])!=float(row['transport_elapsed_s']):
            raise ValueError('resume incomplete native log')
        controls=dict(ray_seed=seed,rotation='fixed_source',member='fixed_source',outline_seed=item['outline_seed'],pool_size=item['pool_size'])
        part=aggregate(values,metadata,areas,controls,name,rays,seed,item['sha256'])
        for r in part:r.update(treatment=item['treatment'],pool_size=item['pool_size'],outline_seed=item['outline_seed'],grid_n=item['grid_n'])
        records.extend(part)
        typed_times.append(dict(row,rays=rays,seed=seed,elapsed_s=float(row['elapsed_s']),
                                transport_elapsed_s=float(row['transport_elapsed_s']),model_bytes=int(row['model_bytes'])))
        for suffix in ('.rgg.gz','.log'):
            source=old_raw/(case+suffix);dest=raw/(case+suffix);shutil.copyfile(source,dest)
            if file_hash(dest)!=file_hash(source):raise ValueError('resume evidence copy mismatch')
        print(f'Verified native prefix {len(typed_times)}/{len(times)}: {case}',flush=True)
    provenance=root/'resume_inputs';provenance.mkdir(exist_ok=True);refs={}
    for p in [previous/'manifest.json',previous/'optical'/'execution_settings.json',previous/'optical'/'timing.csv']:
        dest=provenance/p.name;shutil.copyfile(p,dest)
        if file_hash(dest)!=file_hash(p):raise ValueError('resume provenance copy mismatch')
        refs[str(p)]=dict(ref(p),archived_path=dest.relative_to(root.parent).as_posix())
    write_json(root/'resume_verification.json',dict(status='passed',previous_results=str(previous),
        reused_native_jobs=len(times),checks='exact regenerated native model, original mesh/dependency hashes, complete raw logs; flux independently rebuilt',
        sources=refs))
    return records,typed_times


def run(results: Path, recovery: Path, canopy: Path, scenario: Path, truth: Path,
        core: Path, java: str, ray_counts: list[int], seeds: list[int],limits: dict,
        resume_from: Path | None = None) -> dict:
    inventory = accepted(results)
    storage_budget(results.parent,limits)
    if len(seeds)<4 or len(set(seeds))!=len(seeds) or any(not 0<=s<=2147483647 for s in seeds) or not ray_counts or ray_counts!=sorted(set(ray_counts)) or any(n<=0 for n in ray_counts):
        raise ValueError('>=4 unique ray seeds and positive counts required')
    root = results/'optical'; root.mkdir()  # Never resume by silently overwriting evidence.
    scratch = results.parent/'trash'/'native'; scratch.mkdir()
    start = time.monotonic(); area_table = fixed_scale_area(results,recovery)
    table(root/'fixed_scale_area.csv',area_table)
    if not area_table or not all(r['below_0_1pct'] for r in area_table):
        write_json(root/'report.json',dict(status='area_refinement_failed',native_runs=0,scope=SCOPE))
        raise ValueError('double grid again; fixed-scale area convergence not established')
    config = read_json(scenario); source = {r['surface_id']:r for r in read_json(canopy)['surfaces']}
    settings = read_json(results/'settings.json')
    expected = next((r for r in settings['source_refs'].values() if Path(r['path']).resolve()==canopy.resolve()),None)
    if expected is None or ref(canopy)['sha256'] != expected['sha256']:
        raise ValueError('canopy differs from validated package')
    lamps = config['lamps']
    if len(lamps)!=1 or [lamps[0][k] for k in ('x_m','y_m','z_m','power_umol_s')] != [0,.1,.7,1200]:
        raise ValueError('contract requires unchanged conditional point lamp')
    truth_rows = list(csv.DictReader(truth.open(encoding='utf-8-sig')))
    lookup = {(r['plant_instance_id'],r['capture_id'],r['vertex_id']):r for r in truth_rows}
    metadata = []
    for r in csv.DictReader((results/'identity.csv').open(encoding='utf-8-sig')):
        s = source[r['surface_id']]; plant = s['plant_instance_id']; capture = config['captures_by_instance'][plant]
        row = lookup[(s['source_plant_id'],capture,s['producer_vertex_id'])]
        compound = row['complex_id']; cr = lookup[(s['source_plant_id'],capture,compound)]
        metadata.append(dict(native_id=r['native_id'],surface_id=r['surface_id'],plant_id=plant,leaf_id=compound,
            leaflet_id=s['producer_vertex_id'],rank=cr['rank'],position=row['side']))
    table(root/'identity.csv',metadata)
    write_json(root/'execution_settings.json',dict(scope=SCOPE,ray_counts=ray_counts,seeds=seeds,lamps=lamps,java_executable=java,
        optics='black; direct-only conditional case; no reflectors',
        sources={str(p):ref(p) for p in [Path(__file__),core,scenario,truth,Path('tools/ppfd_validation.py'),
            Path('tools/groimp_light_sanity.py'),Path('tools/groimp_density.py'),Path('tomato3d/light/experiments.py')]}))
    # Freeze exact source bytes before lengthy transport work, so later shared
    # workspace edits cannot invalidate or replace this run's implementation.
    storage_budget(results.parent,limits,sum(Path(item['path']).stat().st_size for item in inventory))
    archive_inputs(results)
    fine_grid = max(r['grid_n'] for r in inventory); base_rays = min(ray_counts)
    jobs = []; records = []; times = []
    for item in inventory:
        counts = ray_counts if item['treatment']=='source_regenerated' and item['grid_n']==fine_grid else [base_rays]
        for rays in counts:
            for seed in seeds: jobs.append((item,rays,seed))
    write_json(root/'resource_budget.json',enforce_budget(jobs,inventory,limits))
    reused_jobs=0
    if resume_from is not None:
        records,times=import_prefix(resume_from,root,jobs,metadata,lamps,limits);reused_jobs=len(times)
        table(root/'flux.csv',records);table(root/'timing.csv',times)
        write_json(root/'progress.json',dict(status='resumed',completed_jobs=len(times),planned_jobs=len(jobs),
            reused_native_jobs=reused_jobs,**storage_budget(results.parent,limits)))
    work = scratch/'current'; work.mkdir()
    for item,rays,seed in jobs[reused_jobs:]:
        name,case=job_name(item,rays,seed);geometry = Path(item['path'])
        disk=storage_budget(results.parent,limits,4*geometry.stat().st_size)
        write_json(scratch/'storage_status.json',dict(case=case,**disk))
        write_json(scratch/'current_job.json',dict(case=case,geometry_sha256=item['sha256'],rays=rays,seed=seed))
        text = model(geometry,lamps,rays,seed); (work/'Model.rgg').write_text(text,encoding='utf-8')
        print('Running '+case,flush=True); began = time.monotonic()
        try:
            _,version = run_groimp(work,core,java,timeout_s=600)
        except Exception as exc:
            write_json(root/'failure.json',dict(status='native_job_failed',case=case,completed_jobs=len(times),
                error_type=type(exc).__name__,message=str(exc),completed_evidence_preserved=True))
            write_json(root/'progress.json',dict(status='native_job_failed',completed_jobs=len(times),planned_jobs=len(jobs),case=case))
            raise
        values,probes = parse_log(work/'groimp.log')
        if probes: raise ValueError('unexpected probes')
        trace = re.findall(r'TRANSPORT_SECONDS\s+([0-9.eE+-]+)',(work/'groimp.log').read_text(encoding='utf-8',errors='replace'))
        if len(trace)!=1: raise ValueError('native compute timing missing/duplicated')
        raw = root/'native_evidence'; raw.mkdir(exist_ok=True)
        # Preserve exact model bytes compressed; reuse only the accepted scratch
        # input on the next fresh JVM. A failure leaves current_job + model/log.
        with (work/'Model.rgg').open('rb') as stream, gzip.GzipFile(filename=str(raw/(case+'.rgg.gz')),mode='wb',mtime=0) as packed:
            shutil.copyfileobj(stream,packed)
        shutil.copyfile(work/'groimp.log',raw/(case+'.log'))
        areas = {r[0]:float(triangle_areas_mm2(np.asarray(r[1:],float).reshape(-1,3,3)*1000).sum())/1e6 for r in csv.reader(geometry.open(encoding='utf-8'))}
        controls = dict(ray_seed=seed,rotation='fixed_source',member='fixed_source',outline_seed=item['outline_seed'],pool_size=item['pool_size'])
        part = aggregate(values,metadata,areas,controls,name,rays,seed,item['sha256'])
        for r in part: r.update(treatment=item['treatment'],pool_size=item['pool_size'],outline_seed=item['outline_seed'],grid_n=item['grid_n'])
        records.extend(part)
        times.append(dict(case=case,rays=rays,seed=seed,elapsed_s=time.monotonic()-began,version=version,
            transport_elapsed_s=float(trace[0]),model_sha256=file_hash(work/'Model.rgg'),model_bytes=(work/'Model.rgg').stat().st_size))
        table(root/'flux.csv',records); table(root/'timing.csv',times)
        write_json(root/'progress.json',dict(status='running',completed_jobs=len(times),planned_jobs=len(jobs),
            last_case=case,**storage_budget(results.parent,limits)))
    noise = []; minima = {}
    for name in sorted({r['scenario'] for r in records}):
        rows,minimum = convergence(records,name,.05)
        noise.extend(dict(scenario=name,**r) for r in rows); minima[name] = minimum
    table(root/'ray_noise.csv',noise); write_json(root/'D2_minimum_tested.json',minima)
    noise_lookup = {(r['scenario'],r['rays'],r['level'],r['entity_id']):r for r in noise}
    grid_rows = []
    for item in inventory:
        if item['treatment']=='stored_source' or item['grid_n']==fine_grid: continue
        coarse_name = f"p{item['pool_size']}_o{item['outline_seed']}_g{item['grid_n']}_{item['treatment']}"
        fine_name = f"p{item['pool_size']}_o{item['outline_seed']}_g{fine_grid}_{item['treatment']}"
        for plant in {r['plant_id'] for r in metadata}:
            a = noise_lookup[(coarse_name,base_rays,'plant',plant)]; b = noise_lookup[(fine_name,base_rays,'plant',plant)]
            delta = b['mean_umol_s']-a['mean_umol_s']; threshold = b['sd_umol_s']
            grid_rows.append(dict(scope=SCOPE,coarse=coarse_name,fine=fine_name,plant_id=plant,delta_umol_s=delta,
                ray_noise_sd_umol_s=threshold,below_noise_sd=abs(delta)<threshold))
    table(root/'grid_light_convergence.csv',grid_rows)
    summaries = []; deltas = []
    pools = sorted({r['pool_size'] for r in inventory if r['pool_size']})
    for pool in pools:
        outline_seeds = sorted({r['outline_seed'] for r in inventory if r['pool_size']==pool})
        for outline_seed in outline_seeds:
            selected = [dict(r) for r in records if r['rays']==base_rays and r['grid_n']==fine_grid and
                ((r['pool_size']==pool and r['outline_seed']==outline_seed) or r['treatment']=='source_regenerated') and r['outline_statistics_included']]
            for r in selected:
                r['scenario'] = r['treatment']; r['controls'] = dict(r['controls'],pool_size=pool,outline_seed=outline_seed)
            for baseline in ('source_regenerated','ellipse'):
                d,s = paired_statistics(selected,baseline,20261014,1000)
                deltas.extend(dict(pool_size=pool,outline_seed=outline_seed,baseline=baseline,**r) for r in d)
                summaries.extend(dict(pool_size=pool,outline_seed=outline_seed,baseline=baseline,**r) for r in s)
        # Independent outline variability: only the same fixed ray seed is used
        # across distinct outline draws. It is never pooled with ray-seed noise.
        outline_records = []
        for outline_seed in outline_seeds:
            selected = [dict(r) for r in records if r['rays']==base_rays and r['seed']==seeds[0] and r['grid_n']==fine_grid and
                ((r['pool_size']==pool and r['outline_seed']==outline_seed) or r['treatment']=='source_regenerated') and r['outline_statistics_included']]
            for r in selected:
                r.update(ensemble='outline_draw',rep_id=str(outline_seed),scenario=r['treatment'],
                    controls=dict(r['controls'],pool_size=pool,outline_seed=outline_seed))
            outline_records.extend(selected)
        for baseline in ('source_regenerated','ellipse'):
            d,s = paired_statistics(outline_records,baseline,20261014,1000)
            deltas.extend(dict(pool_size=pool,outline_seed='multiple',baseline=baseline,**r) for r in d)
            summaries.extend(dict(pool_size=pool,outline_seed='multiple',baseline=baseline,**r) for r in s)
    # Quantitative leaflet conclusions require BOTH compared scenarios to pass
    # the actual per-geometry repeat CV gate, at every included outline draw.
    for row in summaries:
        candidates = [r for r in records if r['level']==row['level'] and r['entity_id']==row['entity_id'] and
            r['rays']==base_rays and r['grid_n']==fine_grid and r['treatment'] in (row['scenario'],row['baseline']) and
            (r['pool_size']==row['pool_size'] or r['treatment']=='source_regenerated') and
            (row['outline_seed']=='multiple' or r['outline_seed'] in (0,row['outline_seed']))]
        cvs = [noise_lookup[(r['scenario'],base_rays,r['level'],r['entity_id'])]['cv'] for r in candidates]
        row['precision_status'] = 'CV_within_provisional5pct' if cvs and all(cv is not None and cv<=.05 for cv in cvs) else 'exploratory_CV_gate_not_met'
    reference_records = [dict(r) for r in records if r['level']=='plant' and r['rays']==base_rays and
        (r['treatment']=='stored_source' or (r['treatment']=='source_regenerated' and r['grid_n']==fine_grid))]
    for r in reference_records:
        r['scenario']=r['treatment'];r['controls']=dict(r['controls'],outline_seed=0,pool_size=0)
    tr_delta,tr_stats = paired_statistics(reference_records,'stored_source',20261014,1000)
    table(root/'triangulation_effect.csv',tr_delta);table(root/'triangulation_effect_statistics.csv',tr_stats)
    table(root/'paired_deltas.csv',deltas); table(root/'paired_statistics.csv',summaries)
    report = dict(scope=SCOPE,status='conditional_optical_results' if all(r['below_noise_sd'] for r in grid_rows) else 'grid_light_not_converged',
        native_runs=len(times),chosen_grid_n=fine_grid if all(r['below_noise_sd'] for r in grid_rows) else None,
        reused_native_jobs=reused_jobs,new_native_jobs=len(times)-reused_jobs,
        new_native_elapsed_s=sum(r['elapsed_s'] for r in times[reused_jobs:]),
        total_elapsed_s=time.monotonic()-start,native_elapsed_s=sum(r['elapsed_s'] for r in times),
        native_transport_elapsed_s=sum(r['transport_elapsed_s'] for r in times),
        outline_uncertainty='separate fixed-ray-seed outline_draw ensemble; conditional on source geometry/member/rotation',
        leaflet_precision='use per-entity ray_noise CV<=5%; otherwise exploratory',physical_validation=False)
    write_json(root/'report.json',report)
    write_json(root/'progress.json',dict(status=report['status'],completed_jobs=len(times),planned_jobs=len(jobs),
        **storage_budget(results.parent,limits)))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('results','recovery','canopy','scenario','truth','core'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--java',required=True); p.add_argument('--ray-counts',type=int,nargs='+',required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--limits',type=Path,required=True)
    p.add_argument('--resume-from',type=Path)
    a = p.parse_args(); print(json.dumps(run(a.results,a.recovery,a.canopy,a.scenario,a.truth,a.core,a.java,a.ray_counts,a.seeds,read_json(a.limits),a.resume_from)))


if __name__=='__main__': main()
