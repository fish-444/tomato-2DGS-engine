"""Attach reviewed vein points to a separately reviewed visible-boundary set.

python -m tools.vein_surface_bridge --help
No boundary is inferred from the vein graph. All faces are interpolated proposals.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

from tools.leaf_preprocess_lab import Lab, assert_build, read_json, write_json, sha
from tools.vein_correspondence import Workflow
from tools.leaf_light.reconstruction import reconstruct_variant
from tools.leaf_light.annotations import POINT_COLUMNS, CONTOUR_COLUMNS
from tools.visible_leaf_patch import inside_mask
from tomato3d.io.schemas import LEAF_SURFACES_CSV_COLUMNS


def read_rows(path: Path, required: list) -> list:
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not set(required)<=set(reader.fieldnames or []): raise ValueError('CSV columns missing: '+str(path))
        return list(reader)


def merge_points(boundaries: list, interior: list) -> list:
    """An explicit existing leaf ID and boundary count are required for every interior row."""
    groups={}
    for r in boundaries:
        if int(r['boundary_index'])<0: raise ValueError('Boundary input must contain boundary points only')
        key=r['leaflet_id']; value=(r['organ_id'],int(r['boundary_count']))
        if value[1]<3 or (key in groups and groups[key]!=value): raise ValueError('Ambiguous boundary identity/count')
        groups[key]=value
    merged=[{k:r[k] for k in POINT_COLUMNS} for r in boundaries]
    identities={(r['leaflet_id'],r['point_id']) for r in boundaries}
    for r in interior:
        if r['leaflet_id'] not in groups: raise ValueError('No reviewed boundary for leaflet '+r['leaflet_id'])
        if (r['leaflet_id'],r['point_id']) in identities: raise ValueError('Boundary/interior physical point ID collision')
        organ,count=groups[r['leaflet_id']]
        merged.append({**{k:r[k] for k in POINT_COLUMNS},'organ_id':organ,'boundary_count':count,'boundary_index':-1})
    return merged


def run(workflow: Workflow, export: Path, boundaries: Path, contours: Path, out: Path, reviewed_by: str, leaflet_id: str | None = None) -> dict:
    if not reviewed_by.strip(): raise ValueError('An explicit boundary/contour reviewer is required')
    w=workflow; w.verify(); export=export.resolve()
    key=str((export/'result.json').relative_to(w.lab.root))
    entries=[r for r in w.status()['exports'] if r['path']==key]
    if len(entries)!=1 or entries[0]['stale']: raise ValueError('Export is absent or stale')
    record=read_json(export/'result.json'); mode=record['mode']
    good={(p['leaflet_id'],p['point_id']) for p in record['points'] if p['xyz_mm'] is not None}
    interior=[r for r in read_rows(export/'points.interior.csv',POINT_COLUMNS) if (r['leaflet_id'],r['point_id']) in good and (leaflet_id is None or r['leaflet_id']==leaflet_id)]
    if not interior: raise ValueError('No eligible triangulated vein points')
    boundary_rows=read_rows(boundaries,POINT_COLUMNS); contour_rows=read_rows(contours,CONTOUR_COLUMNS)
    merged=merge_points(boundary_rows,interior)
    frames={f['image_file']:f for f in w.context['capture']['frames']}
    names=sorted({r['image_file'] for r in merged})
    if any(n not in frames or frames[n]['pose_status']!='solved' or (mode=='strict' and frames[n].get('passes_pose_gate') is not True) for n in names):
        raise ValueError('Boundary set contains an ineligible camera pose')
    policy=dict(w.context['capture']['quality'])
    if policy['max_point_error_px'] is None: policy['max_point_error_px']=float('inf')
    if policy['min_ray_angle_deg'] is None: policy['min_ray_angle_deg']=0
    variant=dict(variant_id='reviewed_vein_partial',image_files=names)
    surfaces,diagnostics=reconstruct_variant(w.context['capture'],variant,merged,contour_rows,policy)
    # Require full projected face footprints to stay in visible accepted masks; holes remain holes.
    leaf_masks={}
    for ref in record['source_reviews']:
        r=read_json(Path(ref['path']))
        if r['identity']!='same': continue
        for sid in (r['a'],r['b']):
            obs=w.lab.samples[sid]; _,mask=w.lab.load(sid)
            intr=w.context['capture']['intrinsics']; full=np.zeros((intr['image_height_px'],intr['image_width_px']),bool)
            x0,y0,x1,y1=obs['crop_xyxy_px']; full[y0:y1,x0:x1]=mask
            leaf_masks[(r['leaflet_id'],obs['image_file'])]=full
    kept=[]; removed=[]
    for row in surfaces:
        tri=np.array([[row[f'v{i}_{a}_mm'] for a in 'xyz'] for i in range(3)])
        evidence=[(frames[n],m) for (leaf,n),m in leaf_masks.items() if leaf==row['leaflet_id'] and n in names]
        if len(evidence)>=2 and all(inside_mask(tri,f,w.context['capture']['intrinsics'],m) for f,m in evidence): kept.append(row)
        else: removed.append(dict(triangle_id=row['triangle_id'],reason='visible_mask_footprint_or_coverage'))
    out=assert_build(out); out.mkdir(parents=True,exist_ok=False)
    for name,data,columns in [('points.csv',merged,POINT_COLUMNS),('leaf_surfaces.csv',kept,LEAF_SURFACES_CSV_COLUMNS)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(data)
    result=dict(status='interpolated_partial_surface' if kept else 'unavailable',faces=len(kept),removed=removed,
        diagnostics=diagnostics,mode=mode,reviewed_by=reviewed_by,context_sha256=w.context_hash,
        source_hashes={str(p):sha(p) for p in [export/'result.json',export/'points.interior.csv',boundaries,contours]},
        source_export=key,surface_semantics='Interpolation between confirmed points; not observed tissue between veins',
        validated_metric=False,suitable_for_ppfd=False)
    write_json(out/'manifest.json',result)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','export','boundaries','contours','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reviewed-by',required=True)
    a=p.parse_args()
    import json
    print(json.dumps(run(Workflow(Lab(a.run)),a.export,a.boundaries,a.contours,a.out,a.reviewed_by)))


if __name__=='__main__':main()
