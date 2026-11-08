"""Curvature-preserving D3 mesh preparation under the full-export reference contract."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import csv
import numpy as np
from shapely import Polygon
from tomato3d.synth.leaflet_surface import normalize_contour,triangulate_contour,surface_points
from tomato3d.state.fixture_truth import _orient_adaxial
from tomato3d.state.schema import fmt
from tomato3d.light.geometry import triangle_areas_mm2
from tools.p2_d3_recovery import reproduction
from tools.p2_light_experiments import ref
from tools.leaf_light.common import read_json,write_json,write_rows,output_dir,file_hash

SCOPE='conditional_fixture_geometry'


def export_vertices(vertices: np.ndarray,matrix: np.ndarray) -> np.ndarray:
    q=np.array([[float(fmt(v,4)) for v in point] for point in vertices])
    return q@matrix[:3,:3].T+matrix[:3,3]


def ellipse(contour: np.ndarray,n: int) -> np.ndarray:
    span=np.ptp(contour,axis=0);theta=np.arange(n)*2*np.pi/n
    return np.column_stack(((1-np.cos(theta))/2,np.sin(theta)*span[1]/span[0]/2))


def mapped(record: dict,contour: np.ndarray,grid: int,scale: float=1.) -> tuple[np.ndarray,dict]:
    contour=np.asarray(contour,dtype=float)
    if not np.isfinite(contour).all() or contour[:,0].min()<0 or contour[:,0].max()>1:
        raise ValueError('contour outside normalized domain')
    polygon=Polygon(contour)
    if not polygon.is_valid or polygon.area<=0:raise ValueError('self_intersecting_or_empty_contour')
    params=deepcopy(record['parameters']);params['length_mm']*=scale
    uv,faces=triangulate_contour(contour,grid);vertices=surface_points(uv,params)
    # Exact source orientation/export convention, not fitted normals.
    faces=_orient_adaxial(vertices,faces,np.asarray(params['up']))
    placed=export_vertices(vertices,np.asarray(record['canopy_transform']));tri=placed[faces]
    areas=triangle_areas_mm2(tri)
    up=np.asarray(record['canopy_transform'])[:3,:3]@np.asarray(params['up'])
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    quality=dict(triangles=len(tri),min_triangle_area_mm2=float(areas.min()),
        degenerate_triangles=int((areas<1e-6).sum()),normal_flips=int((normals@up<=0).sum()),
        area_mm2=float(areas.sum()),scale=scale,
        # Generator centroid-only Delaunay can bridge concave boundaries; report exact support mismatch.
        uv_outside_area=float(sum(Polygon(uv[f]).difference(polygon).area for f in faces)),
        curvature_source='fixture_generator (user_assumed roll/bend)')
    return tri,quality


def match_area(record: dict,contour: np.ndarray,grid: int) -> tuple[np.ndarray,dict]:
    scale=1.;target=record['source_area_mm2']
    for iteration in range(12):
        tri,quality=mapped(record,contour,grid,scale)
        error=abs(quality['area_mm2']/target-1)
        if error<=.00001:
            quality.update(area_relative_error=error,iterations=iteration+1)
            return tri,quality
        scale*=np.sqrt(target/quality['area_mm2'])
    raise ValueError('area matching failed')


def prepare(recovery: Path,pool_path: Path,canopy: Path,out: Path,seeds: list[int]) -> dict:
    if out.exists() and {p.name for p in out.iterdir()}=={'results','trash'} and not any(p.is_file() for p in out.rglob('*')):
        root=out.resolve();results=root/'results';trash=root/'trash'
    else:
        root=output_dir(out);results=root/'results';results.mkdir();trash=root/'trash';trash.mkdir()
    old=read_json(recovery/'recovery_report.json')
    # Historical recovery source file may change later: all existing listed hashes must still match.
    for item in old['sources'].values():
        if file_hash(Path(item['path']))!=item['sha256']:raise ValueError('recovery source changed: '+item['path'])
    records=read_json(recovery/'recovered_parameters.json')['leaflets']
    source={s['surface_id']:s for s in read_json(canopy)['surfaces']}
    rows=[]
    for r in records:
        uv,faces=triangulate_contour(np.asarray(r['contour']),r['grid_n'])
        vertices=surface_points(uv,r['parameters'])
        faces=_orient_adaxial(vertices,faces,np.asarray(r['parameters']['up']))
        tri=export_vertices(vertices,np.asarray(r['canopy_transform']))[faces]
        rows.append(dict(scope=SCOPE,surface_id=r['surface_id'],**reproduction(np.asarray(source[r['surface_id']]['triangles_m'])*1000,tri)))
    write_rows(results/'step0_reproduction.csv',rows,list(rows[0]))
    (results/'raw_generator_vs_export_quantisation.csv').write_bytes((recovery/'reproduction.csv').read_bytes())
    passing=sum(x['status']=='reproduced' for x in rows)
    write_json(results/'step0.json',dict(scope=SCOPE,reproduced=passing,total=len(rows),reference='full_P1_export_path',area_rtol=1e-6,vertex_atol_mm=.01))
    if passing!=len(records):raise ValueError('incomplete recovered canopy; cannot silently drop surfaces')
    pool=read_json(pool_path)
    if pool['meta']['version']!=2 or len(pool['leaflets'])!=28:raise ValueError('v2 pool28 required')
    normalized=[];pool_audit=[]
    for item in pool['leaflets']:
        # Preserve the supplied polygon-area metadata; normalization is scale independent.
        norm=normalize_contour(np.asarray(item['contour_mm']),96)
        contour=np.asarray(norm['points']);normalized.append(dict(id=item['id'],contour=contour))
        pool_audit.append(dict(scope=SCOPE,outline_id=item['id'],polygon_area_mm2=item['polygon_area_mm2'],
            mask_area_mm2=item['area_mm2'],normalization_method=norm['orientation_method'],parameter_width_over_length=float(np.ptp(contour[:,1])/np.ptp(contour[:,0])),
            valid=Polygon(contour).is_valid,u_min=float(contour[:,0].min()),u_max=float(contour[:,0].max())))
    write_rows(results/'pool_normalization.csv',pool_audit,list(pool_audit[0]))
    audit=[];choices=[];paths=[]
    for size in (28,25):
        candidates=[p for p in normalized if size==28 or p['id'] not in {'R5C2','R7C3','R7C4'}]
        for seed in seeds:
            rng=np.random.default_rng(seed);draws=[candidates[int(rng.integers(len(candidates)))] for r in records]
            for r,draw in zip(records,draws):choices.append(dict(scope=SCOPE,pool_size=size,outline_seed=seed,surface_id=r['surface_id'],outline_id=draw['id']))
            for grid in (18,36):
                for treatment in ('source_regenerated','pool_outline','ellipse'):
                    meshes=[]
                    for r,draw in zip(records,draws):
                        contour=np.asarray(r['contour']) if treatment=='source_regenerated' else draw['contour']
                        if treatment=='ellipse':contour=ellipse(contour,96)
                        try:
                            if treatment=='source_regenerated':tri,q=mapped(r,contour,grid);q['area_relative_error']=abs(q['area_mm2']/r['source_area_mm2']-1)
                            else:tri,q=match_area(r,contour,grid)
                            status='supported' if q['degenerate_triangles']==0 and q['normal_flips']==0 and q['area_relative_error']<=.005 else 'unavailable'
                            audit.append(dict(scope=SCOPE,pool_size=size,outline_seed=seed,grid_n=grid,treatment=treatment,surface_id=r['surface_id'],status=status,reason='' if status=='supported' else 'mesh_quality_or_area_gate',**q))
                            meshes.append((r['surface_id'],tri))
                        except ValueError as exc:
                            audit.append(dict(scope=SCOPE,pool_size=size,outline_seed=seed,grid_n=grid,treatment=treatment,surface_id=r['surface_id'],status='unavailable',reason=str(exc)))
                    path=trash/f'pool{size}_seed{seed}_grid{grid}_{treatment}.csv'
                    with path.open('w',encoding='utf-8',newline='') as f:
                        w=csv.writer(f)
                        for i,(sid,tri) in enumerate(meshes):w.writerow([f's{i:04d}']+[format(v,'.17g') for v in (tri/1000).ravel()])
                    paths.append(dict(pool_size=size,outline_seed=seed,grid_n=grid,treatment=treatment,path=str(path.resolve()),sha256=file_hash(path),surface_count=len(meshes)))
    fields=list(dict.fromkeys(k for r in audit for k in r));write_rows(results/'mesh_quality.csv',audit,fields)
    write_rows(results/'outline_draws.csv',choices,list(choices[0]))
    write_json(results/'mesh_inventory.json',paths)
    failed=[x for x in audit if x['status']!='supported']
    report=dict(scope=SCOPE,step0='passed',step0_count=passing,status='mesh_quality_blocked' if failed else 'meshes_prepared',
        mesh_checks=len(audit),failed_checks=len(failed),native_runs=0,
        sources={str(p):ref(p) for p in (pool_path,canopy,recovery/'recovered_parameters.json',Path(__file__),Path('tomato3d/synth/leaflet_surface.py'),Path('tomato3d/state/schema.py'))},
        self_intersection_check='normalized contours checked; complete3D pairwise checks pending transport readiness',
        grid_convergence='pending; do not infer from area-normalized meshes',uniform_scaling='length scaled; dimensionless width_scale fixed, physical width scales equally')
    write_json(results/'preparation_report.json',report);print({k:report[k] for k in ('status','step0_count','mesh_checks','failed_checks')});return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recovery','pool','canopy','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args()
    prepare(a.recovery,a.pool,a.canopy,a.out,a.seeds)

if __name__=='__main__':main()
