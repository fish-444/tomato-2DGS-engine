"""D3 addendum2 preflight with immutable held geometry and fail-closed gates."""
from __future__ import annotations

import argparse
import ast
import shutil
import csv
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from shapely import Polygon
from shapely.validation import explain_validity

from tomato3d.light.geometry import triangle_areas_mm2
from tomato3d.synth.leaflet_surface import normalize_contour, surface_points, triangulate_contour
from tomato3d.state.fixture_truth import _orient_adaxial
from tools.p2_d3_meshes import export_vertices, ellipse, SCOPE
from tools.p2_d3_recovery import reproduction
from tools.p2_d3_quality import Settings, mapped, intersections
from tools.p2_light_experiments import ref
from tools.leaf_light.common import BUILD, read_json, write_json, write_rows, file_hash


def quality_cache_compatible(old_text: str,new_text: str, *, ignore_solver_exhaustion: bool = True) -> bool:
    """Allow only solver exhaustion or monotone confirmation of old positives.

    The exact filter can only remove old positives. It is never called on a
    previously passed pair. Every other AST node must remain identical.
    """
    trees=[]
    for text in (old_text,new_text):
        tree=ast.parse(text)
        filters=0
        expected=ast.dump(ast.parse('_exact_noncoplanar_intersection(t[i], t[j], tol)',mode='eval').body)
        for function in tree.body:
            if not isinstance(function,ast.FunctionDef): continue
            if function.name=='mapped' and ignore_solver_exhaustion:
                loop=next(n for n in function.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
                loop.orelse=[]
            if function.name=='intersections':
                for node in ast.walk(function):
                    if isinstance(node,ast.If) and isinstance(node.test,ast.BoolOp) and isinstance(node.test.op,ast.And):
                        values=node.test.values
                        if len(values)==2 and isinstance(values[0],ast.Call) and isinstance(values[0].func,ast.Name) and values[0].func.id=='_pair_intersects' and ast.dump(values[1])==expected:
                            node.test=values[0];filters+=1
        if filters==1:
            tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='_exact_noncoplanar_intersection')]
        trees.append(ast.dump(tree,include_attributes=False))
    return trees[0]==trees[1]

HELD = frozenset({
    'fx_om0|fx_om0/C0007/P3R', 'fx_om0#2|fx_om0/C0007/P3R',
    'fx_sp0|fx_sp0/C0001/P1R', 'fx_sp0|fx_sp0/C0005/P1R', 'fx_sp0|fx_sp0/C0005/P2L'})
HELD_REASON = 'held_at_source (source contour self-intersects)'


def table(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    write_rows(path, [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in r.items()} for r in rows], columns)


def mesh_hash(triangles_mm: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(triangles_mm, dtype='<f8').tobytes()).hexdigest()


def held_or_none(record: dict, source: dict) -> tuple[np.ndarray, dict] | None:
    sid = record['surface_id']; invalid = not Polygon(record['contour']).is_valid
    if sid in HELD:
        if not invalid:
            raise ValueError('held identity no longer has the approved defect')
        t = np.asarray(source[sid]['triangles_m'], float)*1000
        return t, dict(status='held_at_source', reason=HELD_REASON, outline_treatment=HELD_REASON,
            triangles=len(t), area_mm2=float(triangle_areas_mm2(t).sum()), area_relative_error=0., scale=1.,
            mesh_sha256=mesh_hash(t), leaflet_outline_statistics_included=False,
            geometry_gate='explicit source5 exception; stored geometry not repaired')
    if invalid:
        raise ValueError('new invalid source contour outside approved source5: '+sid)
    return None


def geometry_csv(path: Path, records: list[dict], meshes: list[np.ndarray]) -> None:
    if len(records) != len(meshes):
        raise ValueError('partial canopy cannot be exported')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for i, tri in enumerate(meshes):
            writer.writerow([f's{i:04d}']+[format(x, '.17g') for x in (tri/1000).ravel()])


def evaluate_record(payload: tuple) -> tuple[np.ndarray | None,dict]:
    r,contour,grid,settings,source_held,treatment,reference_quality = payload
    try:
        held = held_or_none(r,source_held)
        if held is not None:
            tri,quality = held
            quality.update({k:reference_quality[k] for k in ('self_intersection_status','self_intersections') if k in reference_quality})
        else:
            tri,quality = mapped(r,contour,grid,settings)
            quality.update(outline_treatment=treatment,leaflet_outline_statistics_included=True,mesh_sha256=mesh_hash(tri))
        return tri,quality
    except ValueError as exc:
        return None,dict(status='unavailable',reason=str(exc))


def prepare(recovery: Path, pool_path: Path, canopy: Path, out: Path,
            seeds: list[int], grids: list[int], settings: Settings,workers: int=2,reuse_results: Path | None=None) -> dict:
    began = time.monotonic(); settings.validate()
    root = out.resolve()
    if not root.is_relative_to(BUILD.resolve()) or root == BUILD.resolve():
        raise ValueError('output must be under build')
    results, trash = root/'results', root/'trash'
    results.mkdir(parents=True, exist_ok=True); trash.mkdir(parents=True, exist_ok=True)
    if any(results.iterdir()):
        raise ValueError('nonempty results: use a fresh run')
    if len(set(seeds)) != len(seeds) or not seeds or not grids or any(g < 2 for g in grids):
        raise ValueError('unique seeds and valid grids required')
    if workers not in (1,2,3,4): raise ValueError('workers must be 1..4')
    if any(b != 2*a for a,b in zip(grids, grids[1:])):
        raise ValueError('grid refinement must double grid_n')
    old = read_json(recovery/'recovery_report.json')
    for item in old['sources'].values():
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('recovery source hash mismatch: '+item['path'])
    records = read_json(recovery/'recovered_parameters.json')['leaflets']
    original = read_json(canopy)
    source = {s['surface_id']:s for s in original['surfaces']}
    if len(records) != len(source) or {r['surface_id'] for r in records} != set(source) or len(records) != 129:
        raise ValueError('complete source129 required')
    if {r['surface_id'] for r in records if not Polygon(r['contour']).is_valid} != HELD:
        raise ValueError('source defects differ from explicit addendum list')
    sources = {str(p):ref(p) for p in [canopy,pool_path,recovery/'recovered_parameters.json',
        recovery/'recovery_report.json',recovery/'reproduction.csv',Path(__file__),Path('tools/p2_d3_quality.py'),
        Path('tools/p2_d3_meshes.py'),Path('tools/p2_d3_recovery.py'),Path('tomato3d/synth/leaflet_surface.py'),
        Path('tomato3d/state/fixture_truth.py'),Path('tomato3d/state/schema.py')]}
    for key,item in old['sources'].items(): sources.setdefault(key,item)
    import triangle
    sources['triangle_native'] = ref(Path(triangle.core.__file__))
    cached={};cached_rows={};reused=0
    if reuse_results is not None:
        from tools.p2_d3_verify import verify as independent_verify,input_path
        independent_verify(reuse_results)
        cache_manifest=read_json(reuse_results/'manifest.json')
        archive=read_json(reuse_results/'archived_inputs.json')['by_sha256']
        old_settings=read_json(reuse_results/'settings.json')
        if old_settings['numerical']!=asdict(settings):raise ValueError('cache numerical settings differ')
        old_quality=next(r for r in cache_manifest['sources'].values() if Path(r['path']).name=='p2_d3_quality.py')
        old_code=input_path(reuse_results,old_quality,archive).read_text(encoding='utf-8')
        new_code=Path('tools/p2_d3_quality.py').read_text(encoding='utf-8')
        if not quality_cache_compatible(old_code,new_code):
            raise ValueError('cache geometry algorithm differs outside post-exhaustion acceptance')
        cache_area_tolerance=settings.area_rtol if quality_cache_compatible(old_code,new_code,ignore_solver_exhaustion=False) else settings.solver_rtol
        for path in (canopy,pool_path,recovery/'recovered_parameters.json'):
            if file_hash(path) not in {r['sha256'] for r in cache_manifest['sources'].values()}:
                raise ValueError('cache input differs')
        for row in csv.DictReader((reuse_results/'mesh_quality.csv').open(encoding='utf-8-sig')):
            key=tuple(row[k] for k in ('pool_size','outline_seed','grid_n','treatment'))
            cached_rows.setdefault(key,[]).append(row)
        for item in read_json(reuse_results/'mesh_inventory.json'):
            key=tuple(str(item[k]) for k in ('pool_size','outline_seed','grid_n','treatment'))
            rows=cached_rows.get(key,[])
            if item['eligible_for_native'] and len(rows)==len(records) and all(r['status']=='held_at_source' or
                (r['status']=='supported' and float(r['area_relative_error'])<=cache_area_tolerance) for r in rows):
                cached[key]=(item,input_path(reuse_results,item,archive))
        sources['cache_manifest']=ref(reuse_results/'manifest.json')
        sources['cache_archives']=ref(reuse_results/'archived_inputs.json')
        sources['cache_verifier']=ref(Path('tools/p2_d3_verify.py'))
        for name in ('report.json','fixed_scale_area.csv','failure_provenance.json','implementation_at_area_check.py'):
            cache_path=reuse_results/'optical'/name
            if cache_path.exists():sources['cache_optical_'+name]=ref(cache_path)
    write_json(results/'settings.json', dict(scope=SCOPE,numerical=asdict(settings),grids=grids,outline_seeds=seeds,workers=workers,
        normalization_points=96,held_surface_ids=sorted(HELD),reuse_results=str(reuse_results) if reuse_results else None,
        cache_policy='verified hashes and identical inputs/numerics; exact confirmation only filters old positives; all other geometry AST identical; changed solver-exhaustion branch permits strict-pass only; reuse frozen paths without working copies',
        angle_provenance='numerical engineering choice: original acute corners prevent universal20deg; common3deg floor and q20 target',
        dependency=dict(triangle=triangle.__version__,python=platform.python_version(),
            install='python -m pip install --only-binary=:all: --no-deps --target <run>/trash/runtime triangle==20250106',
            api='https://rufat.be/triangle/API.html',quality='https://www.cs.cmu.edu/~quake/triangle.quality.html'),
        tolerances=dict(step0_vertex_atol_mm=.01,step0_area_rtol=1e-6,grid_area_rtol=.001,provisional_cv_target=.05),
        statistics='held5 excluded only from leaflet outline statistics; included in all transport totals',
        source_refs=sources))
    step0 = []
    for r in records:
        uv,faces = triangulate_contour(np.asarray(r['contour']),r['grid_n'])
        vertices = surface_points(uv,r['parameters'])
        faces = _orient_adaxial(vertices,faces,np.asarray(r['parameters']['up']))
        t = export_vertices(vertices,np.asarray(r['canopy_transform']))[faces]
        step0.append(dict(scope=SCOPE,surface_id=r['surface_id'],**reproduction(np.asarray(source[r['surface_id']]['triangles_m'])*1000,t)))
    table(results/'step0_reproduction.csv',step0)
    (results/'raw_generator_vs_export_quantisation.csv').write_bytes((recovery/'reproduction.csv').read_bytes())
    if any(r['status'] != 'reproduced' for r in step0):
        raise ValueError('full P1 export reproduction gate failed')
    table(results/'identity.csv',[dict(native_id=f's{i:04d}',surface_id=r['surface_id'],
        plant_instance_id=r['plant_instance_id'],leaflet_id=r['leaflet_id'],held=r['surface_id'] in HELD) for i,r in enumerate(records)])
    pool = read_json(pool_path)
    if pool['meta']['version'] != 2 or len(pool['leaflets']) != 28:
        raise ValueError('v2 pool28 required')
    normalized = []
    for p in pool['leaflets']:
        norm = normalize_contour(np.asarray(p['contour_mm']),96)
        c = np.asarray(norm['points'])
        if not Polygon(c).is_valid:
            raise ValueError('normalized pool invalid; no repair or redraw: '+p['id'])
        normalized.append(dict(id=p['id'],contour=c))
    write_json(results/'normalized_pool.json',dict(scope=SCOPE,leaflets=[dict(id=p['id'],contour=p['contour'].tolist()) for p in normalized]))
    choices = []; batches = []
    for grid in grids:
        batches.append((0,0,grid,'source_regenerated',None))
    for size in (28,25):
        candidates = [p for p in normalized if size == 28 or p['id'] not in {'R5C2','R7C3','R7C4'}]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            draws = [candidates[int(rng.integers(len(candidates)))] for _ in records]
            choices.extend(dict(scope=SCOPE,pool_size=size,outline_seed=seed,surface_id=r['surface_id'],
                outline_id=draw['id'],applied=r['surface_id'] not in HELD) for r,draw in zip(records,draws))
            for grid in grids:
                for treatment in ('pool_outline','ellipse'):
                    batches.append((size,seed,grid,treatment,draws))
    table(results/'outline_draws.csv', choices)
    # Stored source is a reference, not a repaired treatment. Its checks are fully reported.
    audit = []; inventory = []; reference = []; meshes = []
    for i,r in enumerate(records):
        t = np.asarray(source[r['surface_id']]['triangles_m'])*1000; meshes.append(t)
        a = triangle_areas_mm2(t)
        q = intersections(t,settings) if np.all(a>0) else dict(self_intersection_status='unavailable_degenerate_mesh')
        reference.append(dict(scope=SCOPE,surface_id=r['surface_id'],treatment='stored_source',
            held=r['surface_id'] in HELD,triangles=len(t),area_mm2=float(a.sum()),
            min_triangle_area_mm2=float(a.min()),mesh_sha256=mesh_hash(t),**q))
        if i%20 == 0: print(f'Reference 3D check {i+1}/{len(records)}',flush=True)
    table(results/'stored_source_quality.csv',reference)
    path = trash/'stored_source.csv'; geometry_csv(path,records,meshes)
    inventory.append(dict(treatment='stored_source',role='reference_only',pool_size=0,outline_seed=0,grid_n=18,
        path=str(path),sha256=file_hash(path),surface_count=len(meshes),eligible_for_native=True))
    stopped = None
    for size,seed,grid,treatment,draws in batches:
        tag = f'pool{size}_seed{seed}_grid{grid}_{treatment}'; print('Checking '+tag,flush=True)
        meshes = []; batch_audit = []
        key=(str(size),str(seed),str(grid),treatment)
        if key in cached:
            item,cached_path=cached[key];path=cached_path
            if file_hash(path)!=item['sha256']:raise ValueError('cached mesh hash mismatch')
            rows=cached_rows[key]
            if [r['surface_id'] for r in rows]!=[r['surface_id'] for r in records]:raise ValueError('cached identity order mismatch')
            audit.extend(dict(r,geometry_reused_from=str(reuse_results),cache_source_quality_sha256=old_quality['sha256']) for r in rows)
            inventory.append(dict(item,path=str(path),geometry_reused_from=str(reuse_results)))
            table(results/'mesh_quality.csv',audit);write_json(results/'mesh_inventory.json',inventory)
            reused+=1;print('  Reused complete passed batch with verified hashes and AST equivalence',flush=True)
            continue
        payloads = []
        for i,r in enumerate(records):
            contour = np.asarray(r['contour']) if draws is None else draws[i]['contour']
            if treatment == 'ellipse': contour = ellipse(contour,96)
            source_held = {r['surface_id']:source[r['surface_id']]} if r['surface_id'] in HELD else {}
            payloads.append((r,contour,grid,settings,source_held,treatment,reference[i]))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for i,(t,q) in enumerate(executor.map(evaluate_record,payloads,chunksize=1)):
                row = dict(scope=SCOPE,pool_size=size,outline_seed=seed,grid_n=grid,treatment=treatment,surface_id=records[i]['surface_id'],**q)
                if t is not None: meshes.append(t)
                batch_audit.append(row)
                if i%20 == 0: print(f'  {i+1}/{len(records)}',flush=True)
        audit.extend(batch_audit)
        failed = [r for r in batch_audit if r['status']=='unavailable']
        if len(meshes) == len(records):
            path = trash/(tag+'.csv'); geometry_csv(path,records,meshes)
            inventory.append(dict(treatment=treatment,pool_size=size,outline_seed=seed,grid_n=grid,path=str(path),
                sha256=file_hash(path),surface_count=len(meshes),eligible_for_native=not failed))
        table(results/'mesh_quality.csv',audit)
        write_json(results/'mesh_inventory.json',inventory)
        if failed:
            stopped = tag
            break  # No selected contour redraw, no further treatments or native on a failed gate.
    failed = [r for r in audit if r['status']=='unavailable']
    table(results/'rejected_meshes.csv', failed)
    table(results/'held_source.csv',[dict(scope=SCOPE,surface_id=r['surface_id'],outline_treatment=HELD_REASON,
        source_contour_reason=explain_validity(Polygon(r['contour'])),source_ref=r['source_ref']) for r in records if r['surface_id'] in HELD])
    report = dict(scope=SCOPE,status='mesh_quality_blocked' if failed else 'mesh_preflight_passed',
        step0_reproduced=len(step0),source_leaflets=len(records),held_source_count=len(HELD),
        replaced_mesh_checks=sum(r['status']!='held_at_source' for r in audit),failed_checks=len(failed),stopped_batch=stopped,
        planned_batches=len(batches),completed_batches=len(inventory)-1,elapsed_s=time.monotonic()-began,native_runs=0,
        reused_complete_batches=reused,
        minimum_angle_deg=settings.minimum_angle_deg,refinement_target_deg=settings.refinement_angle_deg,
        area_matching='one uniform length scale; width_scale and curvature angles unchanged',
        grid_convergence='unavailable: geometry gate' if failed else 'pending actual optical refinement',
        D2='unavailable: no native run',D3='unavailable: no native run',real_optical_validation=False,
        source_refs=sources)
    write_json(results/'preparation_report.json',report)
    artifacts = {p.name:file_hash(p) for p in results.iterdir() if p.is_file()}
    write_json(results/'manifest.json',dict(scope=SCOPE,artifacts=artifacts,sources=sources,meshes=inventory))
    verify(results)
    print(json.dumps({k:report[k] for k in ('status','replaced_mesh_checks','failed_checks','elapsed_s')}),flush=True)
    return report


def verify(results: Path) -> dict:
    manifest = read_json(results/'manifest.json')
    for rel,h in manifest['artifacts'].items():
        p = (results/rel).resolve()
        if not p.is_relative_to(results.resolve()) or file_hash(p) != h:
            raise ValueError('result hash mismatch: '+rel)
    for item in manifest['sources'].values():
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('source hash mismatch: '+item['path'])
    for item in manifest['meshes']:
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('mesh hash mismatch')
    report = read_json(results/'preparation_report.json')
    records = list(csv.DictReader((results/'mesh_quality.csv').open(encoding='utf-8-sig')))
    failed = [r for r in records if r['status']=='unavailable']
    if len(failed) != report['failed_checks'] or (failed and report['native_runs'] != 0):
        raise ValueError('gate summary mismatch')
    original = {r['surface_id']:r['mesh_sha256'] for r in csv.DictReader((results/'stored_source_quality.csv').open(encoding='utf-8-sig'))}
    for r in records:
        if r['surface_id'] in HELD and r['mesh_sha256'] != original[r['surface_id']]:
            raise ValueError('held source mesh changed')
    result = dict(status='passed',artifacts_verified=len(manifest['artifacts']),meshes_verified=len(manifest['meshes']),
        unchanged_held_surface_ids=sorted(HELD),verification='package integrity and gate accounting; not optical validation')
    write_json(results/'verification.json',result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    for name in ('recovery','pool','canopy','out'): prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--seeds',type=int,nargs='+',required=True)
    prep.add_argument('--grids',type=int,nargs='+',required=True)
    prep.add_argument('--settings',type=Path)
    prep.add_argument('--workers',type=int,default=2)
    prep.add_argument('--reuse-results',type=Path)
    check = sub.add_parser('verify'); check.add_argument('--results',type=Path,required=True)
    args = p.parse_args()
    if args.command == 'verify':
        print(verify(args.results))
    else:
        settings = Settings(**read_json(args.settings)) if args.settings else Settings()
        prepare(args.recovery,args.pool,args.canopy,args.out,args.seeds,args.grids,settings,args.workers,args.reuse_results)


if __name__ == '__main__': main()
