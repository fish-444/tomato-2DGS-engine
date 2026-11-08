"""Hash-pinned v2 outlines for native leaflets without stored shapes."""
from __future__ import annotations
import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from shapely import Polygon, union_all
from tools.vtc_growth_io import ROOT, require, read_json, write_json, write_csv, sha, digest

LIBRARY = ROOT/'build/leaflet_library_v2_20261015/leaflet_contours_v2.json'
LIBRARY_SHA256 = '479f117f40610768a42cfff53fb1a8992f4f79e1d7ffff33e93a876e187ae0a4'
FROZEN = 'leaflet_outline_library.json'
GRID = 2
EXCLUDED = {'view18_reviewed/dwt_02_a218'}  # User decision 2026-10-15; never repair or redraw.


def active(request: dict) -> bool:
    return request.get('engine_hooks', {}).get('leaflet_outline', False) is True


def source(folder: Path) -> Path:
    path = folder/FROZEN if (folder/FROZEN).is_file() else LIBRARY
    require(path.is_file() and sha(path)==LIBRARY_SHA256, 'leaflet outline v2 source hash mismatch')
    return path


def library(path: Path) -> list[dict]:
    require(sha(path)==LIBRARY_SHA256, 'leaflet outline v2 source hash mismatch')
    return compile_text(path.read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=1)
def compile_text(text: str) -> list[dict]:
    try:
        import triangle
    except ImportError:
        sys.path.insert(0, str(ROOT/'build/leaflet_outline_runtime_20261015'))
        import triangle
    require(triangle.__version__=='20250106', 'pinned Triangle20250106 required')
    from tools.p2_d3_quality import constrained, Settings
    data=json.loads(text)
    require(data['version']=='v2_20261015' and len(data['contours'])==195, 'v2 contour coverage')
    result=[]
    for item in data['contours']:
        if item['contour_id'] in EXCLUDED:continue
        contour=np.asarray(item['points'],float)
        uv,faces,quality=constrained(contour,GRID,Settings())
        require(quality['angle_bound_passed'] and quality['refinement_bound_passed'], 'outline triangulation angle/refinement failed: '+item['contour_id'])
        require(quality['outside_uv2'] <= 1e-12 and quality['missing_uv2'] <= 1e-12, 'outline boundary coverage failed')
        tri=uv[faces];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];signed=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        # Match the native diamond winding: length cross width points opposite up.
        faces=faces.copy(); faces[signed>0]=faces[signed>0][:,[0,2,1]]
        area=float(Polygon(contour).area); unit=uv/math.sqrt(area)
        triangles=unit[faces]
        total=sum(Polygon(t).area for t in triangles)
        require(abs(total-1)<=1e-9, 'outline unit area mismatch')
        require(abs(sum(Polygon(t).area for t in triangles)-union_all([Polygon(t) for t in triangles]).area)<=1e-12,
                'outline overlapping triangles')
        result.append(dict(id=item['contour_id'], triangles=triangles.tolist(),
            length=float(np.ptp(unit[:,0])), width=float(np.ptp(unit[:,1])),
            unit_area=total, quality=quality))
    require(len({r['id'] for r in result})==194, 'outline candidate coverage/duplicate IDs')
    return result


def draw(seed: int, organ_id: str, count: int=194) -> int:
    """Exact Java String.hashCode + java.util.Random.nextInt; no shared draw stream."""
    h=0; encoded=organ_id.encode('utf-16-be')
    for i in range(0,len(encoded),2):h=(31*h+int.from_bytes(encoded[i:i+2],'big')) & 0xffffffff
    if h>=0x80000000:h-=0x100000000
    value=((seed+h)^0x5DEECE66D)&((1<<48)-1)
    while True:
        value=(value*0x5DEECE66D+11)&((1<<48)-1); bits=value>>17; result=bits%count
        if count & (count-1)==0:return (count*bits)>>31
        if bits-result+count-1 < (1<<31):return result


def emit(path: Path, working: Path) -> dict:
    compiled=library(path)
    write_csv(working/'leaflet_outline_uv.csv', [dict(outline=i,coordinate=j,value=value)
        for i,r in enumerate(compiled) for j,value in enumerate(np.asarray(r['triangles']).ravel())],
        ['outline','coordinate','value'])
    metadata=dict(library_sha256=LIBRARY_SHA256, library_version='v2_20261015', count=len(compiled),
        assignment='uniform over194 candidates; no terminal/lateral labels present',
        candidate_ids_sha256=digest([r['id'] for r in compiled]),
        exclusions=[dict(id=r['contour_id'],contour_record_sha256=digest(r),reason='boundary angle0.1149604334deg cannot satisfy3deg hard floor',
                        source_kind='user_assumed',source_ref='User explicit exclusion decision2026-10-15')
                    for r in read_json(path)['contours'] if r['contour_id'] in EXCLUDED],
        seed='java.util.Random((long)runtime.seed + (long)organ_id.hashCode()).nextInt(194)',
        grid_n=GRID, minimum_angle_deg=3., refinement_angle_deg=20., triangle_version='20250106',
        mathematical_area_rtol=1e-9, transport_area='existing engine tolerance, measured separately',
        contours=[{k:v for k,v in r.items() if k!='triangles'}|{'triangles':len(r['triangles'])} for r in compiled])
    write_json(working.parent/'leaflet_outline_manifest.json',metadata)
    return metadata


def transform_native(template: str, request: dict) -> str:
    if not active(request):return template
    compiled=library(Path(request.get('_outline_library_path',LIBRARY)))
    code=Path(__file__).with_name('fsp_leaflet_outline_native.rgg').read_text(encoding='utf-8')
    for key,values in [('IDS',[r['id'] for r in compiled]),('SIZES',[len(r['triangles'])*6 for r in compiled]),
                       ('LENGTHS',[r['length'] for r in compiled]),('WIDTHS',[r['width'] for r in compiled])]:
        code=code.replace('@'+key+'@',','.join(json.dumps(v) for v in values))
    def replace(old: str,new: str) -> None:
        nonlocal template
        require(template.count(old)==1,'outline insertion point changed: '+old[:70]);template=template.replace(old,new)
    replace('float[] shape=null;', 'float[] shape=null; int outline=-1; String outlineKey="";')
    replace('@AXIS_CODE@','@AXIS_CODE@\n'+code)
    replace('organs.add(o);','organs.add(o);if(kind==9)outlineIndex(o);')
    replace('o.wid=o.len/LEAF_LW;', 'o.wid=o.len/LEAF_LW;int oi=outlineIndex(o);o.len=Math.sqrt(o.area)*outlineLengths[oi];o.wid=Math.sqrt(o.area)*outlineWidths[oi];')
    replace('double az=Math.toRadians(o.az),el=Math.toRadians(o.elev);\n    float x=',
            'if(o.kind==9)return outlineVertices(o);\n    double az=Math.toRadians(o.az),el=Math.toRadians(o.elev);\n    float x=')
    replace('attachGraph();if(!observationMode)updateGeometry();', 'loadOutlineLibrary();attachGraph();if(!observationMode)updateGeometry();')
    replace('provenance,parent_rank,developmental_axis_id,segment_index");','provenance,parent_rank,developmental_axis_id,segment_index,shape_source");')
    # Both mesh streams keep exactly the same fields plus an opt-in final source column.
    template=template.replace('state,organ_id,vertex,x_m,y_m,z_m,axis_id,parent_rank,segment_index',
                              'state,organ_id,vertex,x_m,y_m,z_m,axis_id,parent_rank,segment_index,shape_source')
    replace('+segmentLabel(o));\n        if(o.kind==2)', '+segmentLabel(o)+","+shapeSource(o));\n        if(o.kind==2)')
    replace('+segmentLabel(org(s.oi)));', '+segmentLabel(org(s.oi))+","+shapeSource(org(s.oi)));')
    return template


def counts(request: dict, states: dict, stored: dict) -> dict:
    result=dict(stored)
    if active(request):
        compiled=library(Path(request['_outline_library_path']))
        for oid,s in states.items():
            if s['class']=='Leaflet' and oid not in stored:result[oid]=len(compiled[draw(request['runtime']['seed'],oid,len(compiled))]['triangles'])*3
    return result


def validate_sources(pkg: dict, data: dict) -> None:
    request=pkg['request']
    if not active(request):return
    compiled=library(Path(request['_outline_library_path']));stored=pkg.get('integration_shape_counts',{})
    sources={}
    for s in data['states.csv']:
        oid=s['organ_id']
        expected='stored' if oid in stored else 'library_v2:'+compiled[draw(request['runtime']['seed'],oid,len(compiled))]['id'] if s['class']=='Leaflet' else 'diamond'
        require(s.get('shape_source')==expected,'outline state source/assignment mismatch')
        sources[int(s['day']),oid]=expected
    for name in ('meshes.csv','consumed.csv'):
        for row in data[name]:require(row.get('shape_source')==sources.get((int(row['state']),row['organ_id'])), 'outline mesh source mismatch')


def check_leaf(request: dict, state: dict, states: dict, coordinates: np.ndarray, placement: dict | None=None, draws=None) -> dict:
    """Double area and actual float32 placement are distinct acceptance checks."""
    compiled=library(Path(request['_outline_library_path']))
    record=compiled[draw(request['runtime']['seed'],state['organ_id'],len(compiled))]
    area=float(state['area_m2']);require(area>0,'zero-area leaflet emitted a surface')
    az,el=np.radians([float(state['azimuth_deg']),float(state['elevation_deg'])])
    parent=states[state['parent_id']];pa,pe=np.radians([float(parent['azimuth_deg']),float(parent['elevation_deg'])])
    from tools.fsp_engine_hooks import require_unique_leaf_order
    require_unique_leaf_order(states)  # PAIRS[true_leaf_order-1] needs one leaf per node index (A7)
    pairs=draws.pairs(state['plant_instance_id'],False) if draws is not None else request['parameters']['PAIRS']['value'];rank=int(state['true_leaf_order']);m=pairs[min(rank-1,len(pairs)-1)]
    basal=request['parameters']['BASAL_LEAFLET_ANGLE_DEG']['value']
    if draws is not None:
        from tools.fsp_random_parameters import compound_of
        basal=draws.compound(compound_of(state['organ_id']),'BASAL_LEAFLET_ANGLE_DEG',basal)
    beta=math.radians(int(state['side'])*basal*(m+1-int(state['pair_index']))/m)
    u=np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
    v=np.array([-math.sin(beta)*math.cos(pe)*math.cos(pa)-math.cos(beta)*math.sin(pa),
        -math.sin(beta)*math.cos(pe)*math.sin(pa)+math.cos(beta)*math.cos(pa),-math.sin(beta)*math.sin(pe)])
    uv=np.asarray(record['triangles']);base=np.array([float(state[k]) for k in ('x_m','y_m','z_m')])
    expected=base+math.sqrt(area)*(uv[:,:,0,None]*u+uv[:,:,1,None]*v)
    precise=float(np.linalg.norm(np.cross(expected[:,1]-expected[:,0],expected[:,2]-expected[:,0]),axis=1).sum()/2)
    require(abs(precise/area-1)<=1e-9,'double outline area mismatch')
    # Native converts to float before (and after) the rigid canopy transform.
    quantized=expected.astype(np.float32).astype(float)
    if placement:
        quantized=(quantized@np.asarray(placement['rotation']).T+np.asarray(placement['translation_mm'])*.001).astype(np.float32).astype(float)
    actual=np.asarray(coordinates).reshape(-1,3,3)
    require(actual.shape==expected.shape and np.allclose(actual,quantized,atol=8e-8,rtol=0), 'native outline placement mismatch')
    normals=np.cross(actual[:,1]-actual[:,0],actual[:,2]-actual[:,0]);normal=-np.cross(u,v)
    if placement:normal=np.asarray(placement['rotation'])@normal
    require(np.all(normals@normal>0),'outline triangle degeneracy/orientation flip')
    actual_area=float(np.linalg.norm(normals,axis=1).sum()/2)
    require(math.isclose(actual_area,area,rel_tol=3e-7,abs_tol=1e-8),'actual native outline area mismatch')
    angles=[]
    for i in range(3):
        a=actual[:,(i+1)%3]-actual[:,i];b=actual[:,(i+2)%3]-actual[:,i]
        angles.extend(np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))))
    minimum_angle=float(min(angles));require(minimum_angle>=3.,'placed outline angle below hard floor')
    # Project actual triangles into the ideal plane: any nonlocal 3D intersection
    # would require projected interior overlap. Legal shared edges remain legal.
    plane_u=u;plane_v=v
    if placement:plane_u=np.asarray(placement['rotation'])@u;plane_v=np.asarray(placement['rotation'])@v
    flat=np.stack([actual@plane_u,actual@plane_v],axis=-1)
    polygons=[Polygon(t) for t in flat]
    overlap=sum(p.area for p in polygons)-union_all(polygons).area
    require(overlap<=max(area*1e-9,1e-18),'placed outline triangle self-intersection')
    return dict(organ_id=state['organ_id'],shape_source=state['shape_source'],area_m2=area,
        double_area_relative_error=abs(precise/area-1),native_area_relative_error=abs(actual_area/area-1),
        native_area_absolute_error_m2=abs(actual_area-area),minimum_native_angle_deg=minimum_angle,
        projected_overlap_m2=max(0.,overlap),triangles=len(actual))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library',type=Path,default=LIBRARY)
    args=parser.parse_args();compiled=library(args.library)
    print(json.dumps(dict(count=len(compiled),min_triangles=min(len(r['triangles']) for r in compiled),
                         max_triangles=max(len(r['triangles']) for r in compiled))))


if __name__=='__main__':main()
