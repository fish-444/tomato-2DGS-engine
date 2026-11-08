"""Explicit optional native FSP capabilities; no new biological response defaults."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import numpy as np
from tools.vtc_growth_io import ROOT,require,number,aware,read_json,write_json,write_csv,rows,digest
from tools.tomato_fsp_canopy import rigid,triangle_area

SCHEMA='fsp-extensions-1'
HOOKS={'CURVE_RATIO','BENT_FRACTION','AREA_CORRECTION'}
OUTPUTS=('optical_flux.csv','scene_photon_budget.csv')


def photon_factor(wavelength_nm: float) -> float:
    """umol photons per joule for an explicitly monochromatic source (SI exact constants)."""
    require(400<=number(wavelength_nm,'wavelength_nm')<=700,'declared PAR wavelength must be 400..700 nm')
    return wavelength_nm*1e-9/(6.62607015e-34*299792458*6.02214076e23)*1e6


def interval_seconds(req: dict,index: int) -> float:
    schedule=req.get('schedule_intervals')
    return schedule[index]['duration_s'] if schedule else req['dt_s']


def boundary_time(req: dict,index: int):
    return aware(req['start_time'])+timedelta(seconds=sum(interval_seconds(req,i) for i in range(index)))


def piecewise_schedule(schedule: dict,lamps: list) -> list[dict]:
    require(set(schedule)=={'start','end','lights','environment','interpolation','max_step_s'},'schedule fields')
    require(schedule['interpolation']=='piecewise_constant','unsupported environment interpolation')
    start,end=aware(schedule['start']),aware(schedule['end']);require(end>start,'empty schedule')
    maxstep=number(schedule['max_step_s'],'max_step_s',0);require(maxstep>0,'positive maximum step')
    ids={l['lamp_id'] for l in lamps};require(set(schedule['lights'])==ids,'schedule lamp coverage')
    groups=[*schedule['lights'].values(),schedule['environment']];boundaries={start,end}
    for group in groups:
        require(bool(group),'schedule gap');previous=start
        for row in group:
            a,z=aware(row['start']),aware(row['end']);require(a==previous and a<z<=end,'schedule gap/overlap/order/range')
            require(row.get('source_kind'),'schedule evidence required');previous=z;boundaries.update((a,z))
        require(previous==end,'schedule incomplete')
    for group in schedule['lights'].values():
        for row in group:require(0<=number(row['fraction'],'dimming')<=1,'dimming out of range')
    for row in schedule['environment']:
        for key in ('temp_c','co2_ppm'):number(row[key],key)
    cuts=sorted(boundaries);segments=[]
    for a,z in zip(cuts,cuts[1:]):
        while a<z:
            stop=min(z,a+timedelta(seconds=maxstep));seconds=(stop-a).total_seconds()
            env=next(r for r in schedule['environment'] if aware(r['start'])<=a<aware(r['end']))
            fractions={lid:next(r['fraction'] for r in group if aware(r['start'])<=a<aware(r['end'])) for lid,group in schedule['lights'].items()}
            segments.append(dict(start=a.isoformat(),end=stop.isoformat(),duration_s=seconds,source_fractions=fractions,temp_c=env['temp_c'],co2_ppm=env['co2_ppm'],source_kind=env['source_kind']))
            a=stop
    require(len(segments)<=365,'schedule exceeds current 365-boundary limit');return segments


def normalize_scenario(scenario: dict) -> dict:
    out=deepcopy(scenario);ext=out.get('extensions')
    if not ext:return out
    require(ext.get('schema_version')==SCHEMA,'unknown extension schema')
    if ext.get('schedule'):
        intervals=piecewise_schedule(ext['schedule'],out['lamps']);require(aware(out['start_time'])==aware(intervals[0]['start']),'schedule/start mismatch')
        out['intervals']=len(intervals)
        out['environment']=[dict(light_hours=r['duration_s']/3600,temp_day_c=r['temp_c'],temp_night_c=r['temp_c'],co2_ppm=r['co2_ppm'],source_kind=r['source_kind']) for r in intervals]
    return out


def hook_settings(parameters: dict) -> dict:
    result=dict(curves={},fraction=None,flat=None,bent=None,area=None,activity={})
    if 'CURVE_RATIO' in parameters:
        entry=parameters['CURVE_RATIO'];weights=entry['value'];require(isinstance(weights,list) and weights,'CURVE_RATIO needs explicit nonempty vector')
        by={str(len(weights)):weights,**entry.get('by_segment_count',{})}
        require(all(number(x,'bend weight',0)>=0 for x in weights) and sum(weights)>0,'invalid default CURVE_RATIO')
        for key,vector in by.items():
            require(key.isdigit() and 1<=int(key)<=20 and isinstance(vector,list) and len(vector)==int(key),'invalid declared segment count')
            require(all(number(x,'bend weight',0)>=0 for x in vector) and sum(vector)>0,'invalid declared bend vector')
        for count in set(parameters['PAIRS']['value']):
            require(str(count) in by,'CURVE_RATIO lacks exact segment count; no interpolation')
            vector=by[str(count)];require(len(vector)==count and all(number(x,'bend weight',0)>=0 for x in vector) and sum(vector)>0,'invalid/zero-sum CURVE_RATIO')
            result['curves'][str(count)]=[x/sum(vector) for x in vector]
        result['activity']['CURVE_RATIO']='active_generated_only'
    if 'BENT_FRACTION' in parameters:
        e=parameters['BENT_FRACTION'];value=e['value']
        if value is None:result['activity']['BENT_FRACTION']='inactive_unavailable_fraction'
        else:
            require(0<=number(value,'BENT_FRACTION')<=1,'BENT_FRACTION out of range')
            template=e.get('templates',{});require(set(template)=={'flat_total_bend_deg','bent_total_bend_deg'},'explicit posture templates required')
            result.update(fraction=value,flat=number(template['flat_total_bend_deg'],'flat amplitude'),bent=number(template['bent_total_bend_deg'],'bent amplitude'))
            result['activity']['BENT_FRACTION']='active_generated_identity_mixture_not_observation_classifier'
    if 'AREA_CORRECTION' in parameters:
        e=parameters['AREA_CORRECTION'];require(e.get('area_basis')=='summed_one_sided_full_outline_lamina_m2','AREA_CORRECTION basis required')
        require(number(e['value'],'AREA_CORRECTION',0)>0,'positive reporting factor required');result['area']=e;result['activity']['AREA_CORRECTION']='reporting_only'
    return result


def stable_draw(organ_id: str,seed: int) -> float:
    return int.from_bytes(hashlib.sha256((organ_id+'|'+str(seed)).encode('utf8')).digest(),'big')%1000000000/1000000000


def rigid_about_anchor(triangles_m: list,anchor_m: list,rotation: list) -> list:
    """Explicit anchored rigid geometry primitive, not a developmental deformation law."""
    r,pivot=rigid(rotation,anchor_m);tri=np.asarray(triangles_m,float)
    require(tri.ndim==3 and tri.shape[1:]==(3,3) and np.isfinite(tri).all(),'invalid anchored triangles')
    return ((tri-pivot)@r.T+pivot).tolist()


def validate_extensions(req: dict,tables: dict) -> dict:
    ext=req.get('extensions');hooks=hook_settings(req['parameters'])
    if not ext:return hooks
    require(ext.get('schema_version')==SCHEMA,'unknown extension schema')
    require(set(ext)<={'schema_version','wavelength_nm','materials','leaf_material','nonleaf_material','objects','max_depth','schedule','imported_deformation','nutrient_coupling'},'unknown extension fields')
    photon_factor(ext['wavelength_nm']);require(type(ext['max_depth']) is int and 1<=ext['max_depth']<=64,'invalid transport depth')
    materials=ext['materials'];require(materials,'materials empty')
    for key,m in materials.items():
        require(m.get('scattering')=='lambertian_reflection_straight_transmission','unsupported scattering law')
        require(all(m.get(x) for x in ('source_kind','source_ref','uncertainty','domain')),'optical provenance missing')
        vals=[number(m[k],k,0) for k in ('absorptance','reflectance','transmittance')];require(abs(sum(vals)-1)<1e-12,'optical fractions must sum to one')
    require(ext['leaf_material'] in materials and ext['nonleaf_material'] in materials,'material binding missing')
    if not req.get('light_only'):
        m=materials[ext['leaf_material']];require(m['absorptance']==1,'nonblack P4 blocked: reviewed NRH light-basis decision required')
    require(not ext.get('imported_deformation'),'imported developmental deformation unavailable: reviewed age/anchor/mass-area contract required')
    require(not ext.get('nutrient_coupling'),'native nutrient coupling unavailable: executable science contract required')
    seen=set();world_triangles=set()
    for obj in ext.get('objects',[]):
        oid=obj['object_id'];require(oid not in seen and oid.startswith('scene:') and all(c not in oid for c in ',\n\r'),'unique scene: object ID required');seen.add(oid)
        require(obj['role'] in ('pot','shelf','post','non_target_plant','optical_fixture'),'unsupported object role')
        require(obj['material'] in materials and obj.get('source_kind') and obj.get('source_ref'),'object provenance/material required')
        require(obj['frame']=='object_local_m' and obj.get('dynamic',False) is False,'objects require static local metre geometry')
        mat=np.asarray(obj['transform'],float);require(mat.shape==(4,4) and np.allclose(mat[3],[0,0,0,1]),'object transform');rigid(mat[:3,:3],mat[:3,3])
        tri=np.asarray(obj['triangles_m'],float);require(tri.ndim==3 and tri.shape[1:]==(3,3) and np.isfinite(tri).all(),'invalid object triangles')
        require(all(triangle_area(t)>1e-14 for t in tri),'degenerate object triangle')
        keys=[tuple(sorted(tuple(v) for v in t)) for t in tri];require(len(set(keys))==len(keys),'duplicate object triangles')
        placed=tri@mat[:3,:3].T+mat[:3,3]
        keys={tuple(sorted(tuple(round(float(v),12) for v in point) for point in t)) for t in placed}
        require(not world_triangles.intersection(keys),'duplicate placed triangles across scene objects');world_triangles.update(keys)
    if ext.get('schedule'):
        expected=piecewise_schedule(ext['schedule'],tables['lamps.csv']);require(expected==req.get('schedule_intervals'),'compiled schedule mismatch')
        require(len(expected)==req['intervals'] and aware(expected[0]['start'])==aware(req['start_time']),'schedule timing mismatch')
    return hooks


def native_code(req: dict,tables: dict,working: Path) -> tuple[str,str]:
    """Emit bounded helpers into the same XL model; original backend jars are untouched."""
    hooks=validate_extensions(req,tables);ext=req.get('extensions');n=req['intervals'];lamps=tables['lamps.csv']
    def array(values,kind='double'):return 'new '+kind+'[]{'+','.join(json.dumps(v) for v in values)+'}'
    durations=[interval_seconds(req,i)/86400 for i in range(n)];elapsed=[0.]
    for d in durations:elapsed.append(elapsed[-1]+d)
    factor=photon_factor(ext['wavelength_nm']) if ext else 1.
    powers=[[float(l['power_umol_s'])*(req['schedule_intervals'][i]['source_fractions'][l['lamp_id']] if req.get('schedule_intervals') else 1)/factor for l in lamps] for i in range(n)]
    material_list=list(ext['materials']) if ext else [];objects=ext.get('objects',[]) if ext else []
    metadata=[];coords=[]
    for i,o in enumerate(objects):
        mat=np.asarray(o['transform']);v=(np.asarray(o['triangles_m'])@mat[:3,:3].T+mat[:3,3]).reshape(-1)
        metadata.append(dict(index=i,object_id=o['object_id'],role=o['role'],material_index=material_list.index(o['material']),coordinates=len(v)))
        coords.extend(dict(object_index=i,coordinate=j,value=float(x)) for j,x in enumerate(v))
    if ext:write_csv(working/'extension_scene.csv',coords,['object_index','coordinate','value'])
    code=(ROOT/'tools/fsp_extensions_native.rgg').read_text(encoding='utf8')
    mapping=dict(ENABLED=str(bool(ext)).lower(),DT=array(durations),ELAPSED=array(elapsed),FACTOR=factor,POWERS='new double[][]{'+','.join(array(p) for p in powers)+'}',
        OBJECT_IDS=array([o['object_id'] for o in metadata],'String'),OBJECT_ROLES=array([o['role'] for o in metadata],'String'),OBJECT_MATERIALS=array([o['material_index'] for o in metadata],'int'),OBJECT_SIZES=array([o['coordinates'] for o in metadata],'int'),
        MAT_R=array([ext['materials'][m]['reflectance'] for m in material_list]),MAT_T=array([ext['materials'][m]['transmittance'] for m in material_list]),
        LEAF_MAT=material_list.index(ext['leaf_material']) if ext else 0,STEM_MAT=material_list.index(ext['nonleaf_material']) if ext else 0,
        CURVES='new double[][]{'+','.join(array(hooks['curves'][str(i)]) if str(i) in hooks['curves'] else 'null' for i in range(21))+'}',
        FRACTION=hooks['fraction'] if hooks['fraction'] is not None else -1,FLAT=hooks['flat'] or 0,BENT=hooks['bent'] or 0)
    for k,v in mapping.items():code=code.replace('@'+k+'@',str(v))
    return code,str(ext['max_depth'] if ext else 1)


def validate_scene_partition(req: dict,geometry: dict) -> None:
    """Reject exact duplicate one-sided leaf/object triangles after their one placement."""
    ext=req.get('extensions')
    if not ext:return
    def keys(triangles):return {tuple(sorted(tuple(round(float(v),12) for v in point) for point in t)) for t in triangles}
    seen=set()
    identities={r['native_organ_id']:r['plant_instance_id'] for r in geometry['identity_map']}
    for oid,triangles in geometry['triangles_mm'].items():
        placement=geometry['placement'][identities[oid]];r,t=rigid(placement['rotation'],placement['translation_mm'])
        current=keys((np.asarray(triangles)@r.T+t)*.001)
        require(not seen.intersection(current),'duplicate placed one-sided leaflet triangles');seen.update(current)
    for obj in ext.get('objects',[]):
        m=np.asarray(obj['transform']);current=keys(np.asarray(obj['triangles_m'])@m[:3,:3].T+m[:3,3])
        require(not seen.intersection(current),'scene object duplicates an organ/other object surface');seen.update(current)


def reporting_area(folder: Path,pkg: dict) -> None:
    hooks=hook_settings(pkg['request']['parameters']);entry=hooks['area'];write_json(folder/'parameter_activity.json',hooks['activity'])
    by={}
    for s in rows(folder/'states.csv'):by.setdefault(s['day'],{})[s['organ_id']]=s
    assignments=[]
    if 'BENT_FRACTION' in hooks['activity']:
        for day,ss in by.items():
            for oid,s in ss.items():
                if s['class']!='CompoundLeaf':continue
                generated=s['provenance']=='model_generated';draw=stable_draw(oid,pkg['request']['runtime']['seed']) if generated else None
                amplitude=(hooks['bent'] if draw<hooks['fraction'] else hooks['flat']) if generated and hooks['fraction'] is not None else 'unavailable_or_imported_unchanged'
                assignments.append(dict(day=day,organ_id=oid,stable_draw=draw,selected_amplitude_deg=amplitude,origin=s['provenance'],observation_label_overwritten=False))
        write_csv(folder/'posture_assignments.csv',assignments,['day','organ_id','stable_draw','selected_amplitude_deg','origin','observation_label_overwritten'])
    if not entry:return
    result=[]
    for day,ss in by.items():
        sums={oid:0. for oid,s in ss.items() if s['class']=='CompoundLeaf'}
        for s in ss.values():
            if s['class']!='Leaflet':continue
            parent=s['parent_id']
            while parent and parent not in sums:parent=ss[parent]['parent_id']
            if parent:sums[parent]+=float(s['area_m2'])
        for oid,area in sums.items():result.append(dict(day=day,organ_id=oid,basis='summed_one_sided_full_outline_lamina_m2',uncorrected_area_m2=area,correction_factor=entry['value'],corrected_reporting_area_m2=area*entry['value'],source_ref=entry['source_ref'],uncertainty=entry['uncertainty'],status='conditional_reporting_only',omitted_leaflet_area='unavailable_no_optical_surfaces_created'))
    write_csv(folder/'area_correction.csv',result,['day','organ_id','basis','uncorrected_area_m2','correction_factor','corrected_reporting_area_m2','source_ref','uncertainty','status','omitted_leaflet_area'])


def validate_native_optics(pkg: dict,data: dict,n: int) -> None:
    if not pkg['request'].get('extensions'):return
    req=pkg['request'];factor=photon_factor(req['extensions']['wavelength_nm']);budget=data['scene_photon_budget.csv'];opt=data['optical_flux.csv']
    require(len(budget)==n,'photon budget interval coverage')
    for i,b in enumerate(budget):
        require(int(b['interval'])==i and b['closure_status']=='unavailable_terminal_paths_not_instrumented','fabricated scene closure')
        require(all(b[k]=='unavailable' for k in ('escape_W','termination_W','estimator_residual_W')),'unmeasured terminal categories must remain unavailable')
        fraction=req.get('schedule_intervals',[{}]*n)[i].get('source_fractions',{})
        emitted=sum(float(l['power_umol_s'])*fraction.get(l['lamp_id'],1) for l in pkg['tables']['lamps.csv'])/factor
        require(math.isclose(float(b['emitted_W']),emitted,rel_tol=3e-7,abs_tol=1e-10),'emitted source accounting mismatch')
        group=[r for r in opt if int(r['interval'])==i];require(len({r['surface_id'] for r in group})==len(group),'duplicate optical surface')
        expected={r['organ_id'] for r in data['meshes.csv'] if int(r['state'])==i}|{o['object_id'] for o in req['extensions'].get('objects',[])}
        require({r['surface_id'] for r in group}==expected,'optical scene inventory incomplete')
        for row in group:
            # Phong uses float32 spectra. Preserve its signed raw subtraction tally,
            # but permit only a relative floating-point roundoff bound near zero.
            roundoff=32*np.finfo(np.float32).eps*max(float(row['received_first_hit_W']),float(row['reflected_W'])+float(row['transmitted_W']))
            for k in ('received_first_hit_W','absorbed_W','reflected_W','transmitted_W'):require(number(row[k],k)>=-roundoff,'negative optical tally exceeds float32 roundoff bound')
        leaves=sum(float(r['absorbed_W']) for r in group if r['role']=='leaf')
        objects=sum(float(r['absorbed_W']) for r in group if r['role']!='leaf')
        require(math.isclose(leaves,float(b['leaf_absorbed_W']),abs_tol=1e-9) and math.isclose(objects,float(b['nonleaf_absorbed_W']),abs_tol=1e-9),'scene absorption categories mismatch')
        native_flux={r['organ_id']:r for r in data['flux.csv'] if int(r['interval'])==i}
        for r in group:
            if r['role']=='leaf':require(math.isclose(float(native_flux[r['surface_id']]['absorbed_umol_s']),max(0,float(r['absorbed_W']))*factor,rel_tol=3e-7,abs_tol=1e-9),'radiometric/photon conversion mismatch')


def validate_hook_geometry(pkg: dict,data: dict) -> None:
    settings=hook_settings(pkg['request']['parameters'])
    if not (settings['curves'] or settings['fraction'] is not None):return
    par=pkg['request']['parameters'];by={}
    for s in data['states.csv']:by.setdefault(s['day'],{})[s['organ_id']]=s
    imported={s['organ_id'] for s in pkg['tables'].get('initial_state.csv',[])}
    from tools.fsp_engine_hooks import require_unique_leaf_order
    for ss in by.values():
        require_unique_leaf_order(ss)  # PAIRS[true_leaf_order-1] needs one leaf per node index (A7)
        for oid,s in ss.items():
            if s['class']!='RachisSegment' or oid in imported:continue
            c=ss[s['parent_id']]
            while c['class']!='CompoundLeaf':c=ss[c['parent_id']]
            count=par['PAIRS']['value'][min(int(c['true_leaf_order'])-1,len(par['PAIRS']['value'])-1)]
            amplitude=par['TOTAL_BEND_DEG']['value']
            if settings['fraction'] is not None:amplitude=settings['bent'] if stable_draw(c['organ_id'],pkg['request']['runtime']['seed'])<settings['fraction'] else settings['flat']
            idx=int(s['pair_index']);weight=settings['curves'].get(str(count),[(1 if k==0 else 2)/(2*count-1) for k in range(count)])[idx-1]
            require(math.isclose(float(s['rachis_bend_alpha_deg']),amplitude*weight,abs_tol=1e-10),'native hook amplitude/weight/sign mismatch')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenario',type=Path,required=True);a=p.parse_args()
    result=normalize_scenario(read_json(a.scenario));print(json.dumps(dict(schema=SCHEMA,intervals=result['intervals'],hooks=hook_settings(result['parameters'])),indent=2))


if __name__=='__main__':main()
