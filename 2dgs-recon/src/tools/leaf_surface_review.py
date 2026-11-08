"""Immutable partial-surface jobs for the existing vein review web.

python -m tools.leaf_surface_review --snapshot-from build/<old> --out build/<new>
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from tools.leaf_preprocess_lab import Lab, assert_build, read_json, write_json, sha
from tools.vein_correspondence import Workflow, prepare
from tools.vein_surface_bridge import run as build_surface, read_rows
from tools.leaf_light.annotations import POINT_COLUMNS, CONTOUR_COLUMNS
from tools.leaf_light.common import StudyError


def snapshot(source, out):
    """Freeze old reviewed pixels/cameras, refresh only explicitly audited context dependencies."""
    source=assert_build(source); out=assert_build(out)
    old=Lab(source); context=read_json(source/'landmarks/context.json')
    for p,h in old.doc['sources'].items():
        if sha(Path(p))!=h: raise ValueError('Changed dataset source: '+p)
    capture=Path(context['capture_path'])
    if sha(capture)!=context['sources'][str(capture)]: raise ValueError('Original camera capture changed')
    if out.exists(): raise ValueError('Snapshot output already exists')
    out.mkdir(parents=True)
    shutil.copy2(source/'dataset.json',out/'dataset.json')
    shutil.copytree(source/'samples',out/'samples')
    prepare(out)
    fresh=read_json(out/'landmarks/context.json')
    if fresh['capture']!=context['capture']:
        raise ValueError('New context would change original cameras; snapshot is not ready')
    # Old reviews remain archived, never implicitly re-attested under new hashes.
    archive=out/'prior_review_archive'; archive.mkdir()
    for name in ('context.json','reviews','current','proposals'):
        src=source/'landmarks'/name
        if src.is_dir(): shutil.copytree(src,archive/name)
        elif src.exists(): shutil.copy2(src,archive/name)
    record=dict(source=str(source),dataset_sha256=sha(out/'dataset.json'),
        original_capture_sha256=sha(capture),source_context_sha256=sha(source/'landmarks/context.json'),
        context_sha256=sha(out/'landmarks/context.json'),
        context_source_changes=[p for p,h in fresh['sources'].items() if context['sources'].get(p)!=h],
        review_policy='Old histories archived read-only; re-review required under new snapshot',
        rescued_poses_imported=False,validated_metric=False,suitable_for_ppfd=False)
    Workflow(Lab(out)).verify();write_json(out/'snapshot.json',record)
    return record


def write_csv(path, data, columns):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(data)


def run_surface(w, leaflet_id, mode, reviewed_by, selected_images=None):
    w.verify()
    if not isinstance(leaflet_id,str) or not leaflet_id: raise ValueError('Leaflet ID required')
    if not isinstance(reviewed_by,str) or not reviewed_by.strip(): raise ValueError('Reviewer name required')
    exported=w.export(mode)
    export=w.lab.root/exported['path']
    out=w.root/'surfaces'/str(time.time_ns());out.mkdir(parents=True,exist_ok=False)
    points=[p for p in exported['points'] if p['leaflet_id']==leaflet_id]
    record=dict(schema='reviewed-partial-surface-1',leaflet_id=leaflet_id,mode=mode,reviewed_by=reviewed_by,
        export=exported['path'],context_sha256=w.context_hash,status='unavailable',reason='',
        points=points,triangles=[],projections=[],heldout_projections=[],
        source_hashes={str(export/'result.json'):sha(export/'result.json')},
        code_hashes={str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name('vein_surface_bridge.py')):sha(Path(__file__).with_name('vein_surface_bridge.py'))},
        validated_metric=False,suitable_for_ppfd=False)
    try:
        if not points:
            record['reviewed_by']=None
            raise ValueError('no_human_confirmed_points')
        eligible=[p for p in points if p['xyz_mm'] is not None]
        record['status']='points_only' if eligible else 'unavailable'
        boundary=[p for p in eligible if p['kind']=='boundary']
        if len(boundary)<3: raise ValueError('fewer_than_three_triangulated_boundary_points')
        order=sorted(p['boundary_index'] for p in boundary)
        if order!=list(range(len(boundary))): raise ValueError('boundary_order_incomplete_or_failed_points')
        xyz=np.array([p['xyz_mm'] for p in boundary])
        if np.linalg.matrix_rank(xyz-xyz.mean(axis=0))<2: raise ValueError('collinear_boundary_points')
        from tools.vein_boundary import check_boundary_polygon
        check_boundary_polygon(np.array([p['xyz_mm'] for p in sorted(boundary,key=lambda p:p['boundary_index'])]))
        if not any(p['kind']=='interior' for p in eligible): raise ValueError('no_triangulated_interior_point')
        # The existing bridge verifies export freshness and consumes the same triangulated points.
        rows=[r for r in read_rows(export/'points.boundary.csv',POINT_COLUMNS) if r['leaflet_id']==leaflet_id]
        image_names=sorted({r['image_file'] for r in rows})
        if selected_images is not None:
            raise ValueError('Use separately saved two-view point reviews for a held-out third view')
        contours=[]; observations={}
        for r in w.latest_reviews():
            if r['identity']!='same' or r['leaflet_id']!=leaflet_id or not w.review_is_current(r): continue
            for sid in (r['a'],r['b']):
                obs=w.lab.samples[sid]
                if obs['image_file'] in image_names: observations[obs['image_file']]=sid
        for name,sid in observations.items():
            _,mask=w.lab.load(sid)
            cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # Choose one connected visible component containing all reviewed boundary observations.
            observed=np.array([[float(r['x_px']),float(r['y_px'])] for r in rows if r['image_file']==name])
            offset=np.array(w.lab.samples[sid]['crop_xyxy_px'][:2])
            candidates=[c for c in cs if len(c)>=3 and all(cv2.pointPolygonTest(c.astype(np.float32),tuple(p-offset),False)>=0 for p in observed)]
            if len(candidates)!=1: raise ValueError('boundary_crosses_disconnected_visible_components:'+name)
            for i,xy in enumerate(candidates[0].reshape(-1,2)+offset):
                contours.append(dict(image_file=name,organ_id=rows[0]['organ_id'],leaflet_id=leaflet_id,
                    vertex_index=i,x_px=int(xy[0]),y_px=int(xy[1])))
        write_csv(out/'boundary.csv',rows,POINT_COLUMNS);write_csv(out/'contours.csv',contours,CONTOUR_COLUMNS)
        result=build_surface(w,export,out/'boundary.csv',out/'contours.csv',out/'mesh',reviewed_by,leaflet_id=leaflet_id)
        record['source_hashes'].update({str(out/n):sha(out/n) for n in ('boundary.csv','contours.csv')})
        record['bridge']=result
        record['source_hashes'].update({str(out/'mesh'/n):sha(out/'mesh'/n) for n in ('manifest.json','leaf_surfaces.csv','points.csv')})
        triangles=read_rows(out/'mesh/leaf_surfaces.csv',['triangle_id'])
        record['triangles']=[dict(id=t['triangle_id'],xyz_mm=[[float(t[f'v{i}_{a}_mm']) for a in 'xyz'] for i in range(3)],source='interpolated') for t in triangles]
        record['status']='interpolated_partial_surface' if triangles else 'points_only'
        if not triangles: record['reason']='no_visible_mask_supported_triangles'
    except (ValueError,StudyError) as exc:
        record['reason']=str(exc)
    # Project into all independently confirmed same-leaflet observations, including unused views.
    member_sids=set()
    for sid,ids in w.reviewed_owners().items():
        if leaflet_id in ids:member_sids.add(sid)
    used={r['image_file'] for p in points for r in p.get('reprojections',[])}
    for sid in sorted(member_sids):
        obs=w.lab.samples[sid];name=obs['image_file'];key=name.lower()
        if not w.cams.posed(key): continue
        view=dict(sample_id=sid,image_file=name,used_for_fit=name in used,points=[],triangles=[])
        for p in points:
            if p['xyz_mm'] is None: continue
            xy=w.cams.project(key,[p['xyz_mm']])[0]
            if not np.isfinite(xy).all(): continue
            measured=next((r['observed_xy'] for r in p.get('reprojections',[]) if r['image_file']==name),None)
            view['points'].append(dict(point_id=p['point_id'],kind=p.get('kind','interior'),xy=xy.tolist(),observed_xy=measured))
        for t in record['triangles']:
            xy=w.cams.project(key,t['xyz_mm'])
            if np.isfinite(xy).all():view['triangles'].append(dict(id=t['id'],xy=xy.tolist()))
        record['projections'].append(view)
    record['path']=str(out.relative_to(w.lab.root));write_json(out/'result.json',record)
    return record


def surface_results(w):
    # A changed input prevents recomputation, but saved history remains viewable as stale.
    freshness_error=None
    try:
        exports={r['path']:r['stale'] for r in w.status()['exports']}
    except (ValueError,FileNotFoundError) as exc:
        exports={};freshness_error=str(exc)
    records=[]
    for path in sorted((w.root/'surfaces').glob('*/result.json')):
        r=read_json(path)
        r['stale']=exports.get(str(Path(r['export'])/'result.json'),True) or r['context_sha256']!=w.context_hash
        r['stale'] |= any(not Path(p).exists() or sha(Path(p))!=h for p,h in {**r['source_hashes'],**r['code_hashes']}.items())
        r['freshness_error']=freshness_error
        records.append(r)
    return dict(results=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot-from',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(snapshot(a.snapshot_from,a.out)))


if __name__=='__main__': main()
