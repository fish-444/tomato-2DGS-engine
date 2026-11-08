"""Rigid P1-local to native canopy transforms; no light or physiology here."""
from __future__ import annotations
import math
import numpy as np
from tools.vtc_growth_io import require, digest


def rigid(rotation: list, translation_mm: list) -> tuple[np.ndarray,np.ndarray]:
    r=np.asarray(rotation,dtype=float);t=np.asarray(translation_mm,dtype=float)
    require(r.shape==(3,3) and t.shape==(3,) and np.isfinite(r).all() and np.isfinite(t).all(),'invalid placement shape/value')
    require(np.allclose(r.T@r,np.eye(3),atol=1e-9,rtol=0) and abs(np.linalg.det(r)-1)<1e-9,'placement must be proper rigid rotation; no scale/reflection')
    return r,t


def triangle_area(vertices: list) -> float:
    v=np.asarray(vertices,dtype=float)
    require(v.shape==(3,3) and np.isfinite(v).all(),'triangle must have three finite 3D vertices')
    return float(np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[0]))/2)


def assemble_canopy(local_states: list[dict], placement: dict) -> dict:
    """Normalized local surfaces use mm. This is the sole placement conversion."""
    require(set(placement)=={s['plant_instance_id'] for s in local_states},'placement membership mismatch')
    surfaces=[];identity=[];seen=set();max_error=0.;areas={}
    for s in local_states:
        pid=s['plant_instance_id'];p=placement[pid];r,t=rigid(p['rotation'],p['translation_mm'])
        require(p.get('provenance'),'placement provenance required')
        sid=s['surface_id'];require(sid not in seen,'duplicate runtime surface ID');seen.add(sid)
        local=np.asarray(s['triangles_mm'],dtype=float);require(local.ndim==3 and local.shape[1:]==(3,3),'triangle array shape')
        world=(local@r.T+t)*.001;back=(world*1000-t)@r
        max_error=max(max_error,float(np.max(np.abs(back-local))))
        before=sum(triangle_area(v) for v in local)*1e-6;after=sum(triangle_area(v) for v in world)
        require(math.isclose(before,after,rel_tol=1e-10,abs_tol=1e-14),'area changed under rigid placement')
        normals=[]
        for a,b in zip(local,world):
            u=np.cross(a[1]-a[0],a[2]-a[0]);v=np.cross(b[1]-b[0],b[2]-b[0])
            require(np.linalg.norm(u)>0,'degenerate optical triangle')
            require(np.allclose(r@(u/np.linalg.norm(u)),v/np.linalg.norm(v),atol=1e-9),'normal/winding changed')
            normals.append((v/np.linalg.norm(v)).tolist())
        areas[sid]=after
        surfaces.append(dict(**s,triangles_m=world.tolist(),normals_canopy=normals,area_m2=after))
        identity.append({k:s[k] for k in ('surface_id','plant_instance_id','source_plant_id','organ_id','producer_vertex_id')})
    return dict(frame='canopy_m',length_unit='m',surfaces=surfaces,identity_map=identity,placement_hash=digest(placement),
        geometry_hash=digest([{k:s[k] for k in ('surface_id','triangles_m')} for s in surfaces]),
        roundtrip_max_error_mm=max_error,one_sided_areas_m2=areas)


def split_by_instance(scene_outputs: list[dict],identity_map: list[dict]) -> dict:
    lookup={s['surface_id']:s for s in identity_map};require(len(lookup)==len(identity_map),'duplicate identity map')
    out={s['plant_instance_id']:[] for s in identity_map};seen=set()
    for row in scene_outputs:
        sid=row['surface_id'];require(sid in lookup and sid not in seen,'unknown/duplicate surface result');seen.add(sid)
        owner=lookup[sid];require(row.get('plant_instance_id',owner['plant_instance_id'])==owner['plant_instance_id'],'cross-instance result')
        out[owner['plant_instance_id']].append(dict(row,**{k:owner[k] for k in ('plant_instance_id','source_plant_id','organ_id')}))
    require(seen==set(lookup),'missing surface results')
    return out
