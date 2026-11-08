"""Recover original fixture generator parameters and enforce D3 reproduction gate."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tomato3d.state import fixture_truth as ft
from tomato3d.synth.contours import load_library
from tomato3d.synth.leaflet_surface import surface_points,triangulate_contour
from tomato3d.light.geometry import triangle_areas_mm2
from tools.leaf_light.common import read_json,write_json,write_rows,output_dir
from tools.p2_light_experiments import ref


def reproduction(actual: np.ndarray, regenerated: np.ndarray) -> dict:
    a=np.asarray(actual,dtype=float);b=np.asarray(regenerated,dtype=float)
    av=np.unique(a.reshape(-1,3),axis=0);bv=np.unique(b.reshape(-1,3),axis=0)
    distance=max(float(cKDTree(av).query(bv)[0].max()),float(cKDTree(bv).query(av)[0].max()))
    area=float(triangle_areas_mm2(a).sum());other=float(triangle_areas_mm2(b).sum())
    error=abs(other-area)/area
    return dict(max_vertex_set_distance_mm=distance,source_area_mm2=area,regenerated_area_mm2=other,
                area_relative_error=error,status='reproduced' if distance<=.01 and error<=1e-6 else 'unavailable',
                reason='' if distance<=.01 and error<=1e-6 else 'source_reproduction_tolerance_failed')


def run(p1: Path,canopy: Path,scenario: Path,out: Path) -> dict:
    root=output_dir(out);results=root/'results';results.mkdir();(root/'trash').mkdir()
    manifest=read_json(p1/'p1_manifest.json');request=manifest['request'];config=read_json(scenario)
    sources={str(p):ref(p) for p in (p1/'p1_manifest.json',p1/'canopy_placement.json',canopy,scenario,Path(__file__),
        Path(ft.__file__),Path('tomato3d/synth/leaflet_surface.py'),Path('tomato3d/synth/contours.py'))}
    def load(path: str,pid: str):
        scene=Path(path)
        for p in (scene/'plant_graph.json',scene.parent.parent/'leaflet_contours.json',scene.parent.parent/'generator_config.json'):
            sources[str(p)]=ref(p)
        return ft.load_plant(scene,pid)
    donor=load(request['donor_scene'],'d3_donor')
    plants={}
    for item in request['source_plants']:
        pid=item['source_plant_id'];plant=load(item['scene_dir'],pid)
        capture=next(config['captures_by_instance'][s['plant_instance_id']] for s in read_json(canopy)['surfaces'] if s['source_plant_id']==pid)
        for step in request['sequences'][pid]:
            for serial in step.get('remove',[]):plant=ft.remove_compound(plant,serial)
            for serial,yaw in step.get('add',[]):plant=ft.add_compound_from(plant,donor,serial,yaw)
            if step.get('side_shoots'):raise ValueError('side-shoot replay not implemented; cannot omit')
            if step['capture_id']==capture:break
        else:raise ValueError('capture missing from source sequence')
        plants[pid]=plant
    placements={x['plant_instance_id']:np.array(x['transform']['matrix_4x4']) for x in read_json(p1/'canopy_placement.json')['instances'] if x.get('placement_status')=='placed'}
    rows=[];recovered=[];cache={}
    for source in read_json(canopy)['surfaces']:
        pid=source['source_plant_id'];plant=plants[pid];lid=source['producer_vertex_id'];instance=source['plant_instance_id']
        record=dict(scope='conditional_fixture_geometry',surface_id=source['surface_id'],plant_instance_id=instance,leaflet_id=lid)
        try:
            comp=next(c for c in plant.compounds if lid.startswith(plant.compound_vertex_id(c)+'/'))
            leaf=next(l for l in comp.leaflets if plant.compound_vertex_id(comp)+'/'+l['tag']==lid)
            scene=Path(comp.source_ref.split('#')[0]);cell=scene.parent.parent
            if str(cell) not in cache:
                contours,_=load_library(cell/'leaflet_contours.json');cfg=read_json(cell/'generator_config.json')
                cache[str(cell)]=(contours,int(cfg['constants']['leaflet_mesh_grid_n']['value']))
            contours,grid=cache[str(cell)];contour=contours[int(leaf['contour_index'])]
            uv,faces=triangulate_contour(contour,grid);vertices=surface_points(uv,leaf)
            mat=placements[instance];placed=vertices@mat[:3,:3].T+mat[:3,3]
            regen=placed[faces];actual=np.asarray(source['triangles_m'])*1000
            audit=reproduction(actual,regen);record.update(audit)
            recovered.append(dict(**record,curvature_source='fixture_generator (user_assumed roll/bend)',
                parameters=leaf,contour=contour.tolist(),grid_n=grid,canopy_transform=mat.tolist(),source_ref=comp.source_ref))
        except (KeyError,StopIteration,ValueError) as exc:
            record.update(status='unavailable',reason=type(exc).__name__+': '+str(exc),max_vertex_set_distance_mm=None,
                source_area_mm2=None,regenerated_area_mm2=None,area_relative_error=None)
        rows.append(record)
    columns=['scope','surface_id','plant_instance_id','leaflet_id','status','reason','max_vertex_set_distance_mm','source_area_mm2','regenerated_area_mm2','area_relative_error']
    write_rows(results/'reproduction.csv',rows,columns)
    write_json(results/'recovered_parameters.json',dict(scope='conditional_fixture_geometry',leaflets=recovered))
    passing=sum(x['status']=='reproduced' for x in rows);fraction=passing/len(rows)
    report=dict(scope='conditional_fixture_geometry',step='0',status='passed' if fraction>=.8 else 'stopped_below_80_percent',
        total=len(rows),reproduced=passing,reproduction_fraction=fraction,vertex_atol_mm=.01,area_rtol=1e-6,
        distance_method='symmetric nearest-vertex maximum; source generator triangulation',sources=sources,
        native_runs=0,source_modified=False,curve_fitting=False)
    write_json(results/'recovery_report.json',report);print({k:report[k] for k in ('status','total','reproduced','reproduction_fraction')});return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('p1','canopy','scenario','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.p1,a.canopy,a.scenario,a.out)

if __name__=='__main__':main()
