"""Read-only P1 consumer and explicit conditional native-state boundary."""
from __future__ import annotations
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from tools.vtc_growth_io import ROOT, require, rows, read_json, write_json, write_csv, sha, digest, number, aware, identifier
from tools.tomato_fsp_canopy import rigid, triangle_area, assemble_canopy

SCHEMA='fsp-integration-1'
P1_FILES=('captures.csv','identity_manifest.csv','mtg_vertices.csv','areas.csv','optical_surface_manifest.csv','canopy_placement.json','p1_manifest.json')
NATIVE_FILES=['initial_state.csv','initial_controls.json','integration_geometry.json','initialization.json']


def inside(folder: Path,name: str) -> Path:
    p=(folder/name).resolve();require(p.is_relative_to(folder.resolve()),'package path escape');return p


def audit_p1_package(package_path: Path) -> dict:
    """Call the actual producer validators, then check consumer readiness separately."""
    from tomato3d.state import schema as s
    missing=[n for n in P1_FILES if not (package_path/n).is_file()]
    if missing:return dict(status='unavailable',fixture_ready=False,real_ready=False,missing=missing,reason='producer_package_not_delivered')
    manifest=read_json(package_path/'p1_manifest.json')
    require(manifest.get('schema',manifest.get('schema_version'))==s.P1_SCHEMA_VERSION,'unsupported P1 schema')
    tables={n:rows(package_path/n) for n in P1_FILES if n.endswith('.csv')}
    s.check_captures(tables['captures.csv']);s.check_identity_rows(tables['identity_manifest.csv'])
    s.check_mtg_rows(tables['mtg_vertices.csv']);s.check_optical_rows(tables['optical_surface_manifest.csv'])
    from tools.tomato_fsp_intake import plane_evidence
    planes,plane_hashes=plane_evidence(package_path,tables['captures.csv'],tables['areas.csv'],manifest.get('schema_revision',0))
    hashes={n:sha(package_path/n) for n in P1_FILES};hashes.update(plane_hashes);gaps=[];surface_cache={}
    for m in tables['optical_surface_manifest.csv']:
        if m['optical_readiness']=='unavailable':gaps.append('optical_surface_unavailable:'+m['external_leaflet_id']);continue
        p=inside(package_path,m['surface_file']);require(p.is_file() and sha(p)==m['surface_sha256'],'surface hash mismatch')
        if m['surface_file'] not in surface_cache:surface_cache[m['surface_file']]=rows(p,','.join(s.OPTICAL_TRIANGLES_COLUMNS))
        rr=[r for r in surface_cache[m['surface_file']] if r['external_leaflet_id']==m['external_leaflet_id']];require(len(rr)==int(m['n_triangles']),'triangle count mismatch')
        require(len({r['triangle_id'] for r in rr})==len(rr),'duplicate triangle IDs')
        total=0.
        for r in rr:
            require(r['external_leaflet_id']==m['external_leaflet_id'] and r['frame']=='plant_local_mm','surface identity/frame mismatch')
            total+=triangle_area([[number(r[f'v{i}_{k}_mm'],'vertex') for k in 'xyz'] for i in range(3)])
        require(math.isclose(total,number(m['one_sided_area_mm2'],'area',0),rel_tol=3e-4,abs_tol=.01),'surface/area manifest mismatch')
        require(m['area_convention']=='one_sided_lamina_full_outline','unsupported one-sided area convention')
        hashes[m['surface_file']]=sha(p)
    for c in tables['captures.csv']:
        if c['capture_time']=='unavailable':gaps.append('capture_time_unavailable:'+c['capture_id'])
        else:
            try:aware(c['capture_time'])
            except ValueError:gaps.append('aware_capture_time_required:'+c['capture_id'])
        if c['scale_status']!='solved':gaps.append('scale_unavailable:'+c['capture_id'])
        if c['evidence']=='real_photo':
            photo=Path(c['image_path']);photo=photo if photo.is_absolute() else ROOT/photo
            if not photo.is_file():gaps.append('source_photo_unavailable:'+c['capture_id'])
            elif sha(photo)!=c['image_sha256']:gaps.append('source_photo_hash_mismatch:'+c['capture_id'])
    for name in ('truth/truth_mtg_vertices.csv','truth/truth_areas.csv','truth/compound_serials.csv'):
        if (package_path/name).is_file():hashes[name]=sha(package_path/name)
    if (package_path/'axes.csv').is_file():hashes['axes.csv']=sha(package_path/'axes.csv')
    if (package_path/'trusses.csv').is_file():hashes['trusses.csv']=sha(package_path/'trusses.csv')
    if (package_path/'completion_manifest.json').is_file():
        completion=read_json(package_path/'completion_manifest.json');hashes['completion_manifest.json']=sha(package_path/'completion_manifest.json')
        for name,h in completion.get('files',{}).items():
            require(sha(inside(package_path,name))==h,'completion source hash mismatch');hashes[name]=h
    for name,h in hashes.items():
        if name in manifest.get('files',{}):require(manifest['files'][name]==h,'producer manifest file hash mismatch: '+name)
    evidence={r['evidence'] for r in tables['captures.csv']}
    fixture=evidence=={'synthetic_fixture'}
    return dict(status='audited',fixture_ready=fixture and not gaps,real_ready=not fixture and not gaps,
        fixture=fixture,missing=gaps,physiological_initialization='separate_scenario_required',
        schema_version=s.P1_SCHEMA_VERSION,schema_revision=manifest.get('schema_revision','unavailable'),scale_plane_evidence=planes,producer_schema_sha256=sha(Path(s.__file__)),source_hashes=hashes,
        producer_manifest=manifest,table_rows={k:len(v) for k,v in tables.items()})


def selected_package(folder: Path,scenario: dict) -> dict:
    audit=audit_p1_package(folder);require(audit['status']=='audited','P1 package unavailable')
    require(scenario.get('schema_version')==SCHEMA and scenario.get('nutrient_feedback_enabled') is False,'unsupported scenario/scientific nutrient feedback')
    allowed={'schema_version','fixture','captures_by_instance','start_time','intervals','rays','seed','parameters','organ_initialization','plant_initialization','initialization_provenance','environment','lamps','allow_conditional_surfaces','nutrient_feedback_enabled','nutrient_ledger_required','nutrient_scenarios','freeze_geometry','light_only','state_source','dt_s'}
    allowed.update(('extensions','observation_route','completion_manifest','simulation_identity_map'))
    allowed.update(('engine_hooks','axis_contract','axes','topology_scope','topology_completeness','native_boundary','imported_shape_api','resource_hook','plant_positions'))
    require(set(scenario)<=allowed,'unsupported scenario fields (schedules/response laws must be explicitly implemented)')
    for key in ('freeze_geometry','light_only'):require(type(scenario.get(key,False)) is bool,'diagnostic flag must be boolean')
    require(audit['fixture']==scenario['fixture'],'fixture/real mode mismatch')
    captures=rows(folder/'captures.csv');chosen=scenario['captures_by_instance']
    capmap={(c['plant_instance_id'],c['capture_id']):c for c in captures};require(chosen and len(capmap)==len(captures),'duplicate selected capture')
    for pid,cid in chosen.items():
        require((pid,cid) in capmap,'capture identity mismatch')
        c=capmap[pid,cid];require(c['scale_status']=='solved' and c['scale_source'] not in ('','unavailable'),'metric scale required')
        if scenario.get('state_source','main')=='main':require(c['frame']=='plant_local_mm','selected metric capture requires declared plant-local frame; no implicit image/board conversion')
        require(aware(c['capture_time'])==aware(scenario['start_time']),'initial observations must align with forecast start')
    require(scenario.get('state_source','main') in ('main','fixture_truth'),'unsupported P1 state source')
    if scenario.get('state_source')=='fixture_truth':
        require(scenario['fixture'] is True,'truth cannot initialize real mode')
        vertices=[];truth=rows(folder/'truth/truth_mtg_vertices.csv')
        for pid,cid in chosen.items():
            source=capmap[pid,cid]['source_plant_id']
            vertices.extend(dict(r,plant_instance_id=pid) for r in truth if r['plant_instance_id']==source and r['capture_id']==cid)
    else:vertices=[r for r in rows(folder/'mtg_vertices.csv') if chosen.get(r['plant_instance_id'])==r['capture_id']]
    vmap={(r['plant_instance_id'],r['vertex_id']):r for r in vertices};require(len(vmap)==len(vertices),'duplicate selected MTG vertex')
    identities=[r for r in rows(folder/'identity_manifest.csv') if chosen.get(r['plant_instance_id'])==r['capture_id']]
    imap={(r['plant_instance_id'],r['organ_id']):r for r in identities}
    for r in vertices:
        require(r['frame']=='plant_local_mm','no implicit board/image to plant transform')
        if r['class'] in ('CompoundLeaf','Leaflet') and scenario.get('state_source','main')=='main':
            mapped=scenario.get('simulation_identity_map',{}).get(r['plant_instance_id']+'|'+r['vertex_id'],r['vertex_id'])
            identity=imap.get((r['plant_instance_id'],mapped))
            require(identity is not None and identity['correspondence_status']=='confirmed' and identity['track_state'] not in ('ambiguous','removal_confirmed'),'unconfirmed/removed P1 identity')
    surfaces=[];surface_rows={}
    for m in rows(folder/'optical_surface_manifest.csv'):
        pid=m['plant_instance_id']
        if chosen.get(pid)!=m['capture_id']:continue
        if scenario.get('state_source')=='fixture_truth' and m['geometry_source']!='synthetic_truth':continue
        require(m['optical_readiness'] in ('supported','conditional_completion'),'required optical surface unavailable')
        require((pid,m['vertex_id']) in vmap and vmap[pid,m['vertex_id']]['class']=='Leaflet','optical surface is not a located leaflet')
        if m['optical_readiness']=='conditional_completion':require(scenario.get('allow_conditional_surfaces') is True,'conditional surfaces not selected')
        if m['surface_file'] not in surface_rows:
            grouped=defaultdict(list)
            for row in rows(inside(folder,m['surface_file'])):grouped[row['external_leaflet_id']].append(row)
            surface_rows[m['surface_file']]=grouped
        triangles=[[[float(r[f'v{i}_{k}_mm']) for k in 'xyz'] for i in range(3)] for r in surface_rows[m['surface_file']][m['external_leaflet_id']]]
        surfaces.append(dict(plant_instance_id=pid,source_plant_id=m['source_plant_id'],organ_id=pid+'|'+m['vertex_id'],
            producer_vertex_id=m['vertex_id'],surface_id=m['external_leaflet_id'],triangles_mm=triangles,provenance=m['evidence'],geometry_source=m['geometry_source'],face_convention=m['face_convention'],surface_file=m['surface_file']))
    require({(s['plant_instance_id'],s['producer_vertex_id']) for s in surfaces}=={k for k,v in vmap.items() if v['class']=='Leaflet'},'missing/duplicate leaflet optical coverage')
    require(len({s['surface_id'] for s in surfaces})==len(surfaces),'duplicate surface manifest')
    completion=None
    if scenario.get('observation_route')=='conditional_completion':
        from tools.tomato_fsp_intake import validate_completion
        completion=validate_completion(folder,scenario,vertices,surfaces)
    if scenario.get('observation_route')=='observed_metric':
        require(all(x['geometry_source'] not in ('generated_candidate','synthetic_truth') for x in surfaces),'synthetic/completed geometry is not observed metric')
    placement=read_placement(folder/'canopy_placement.json',set(chosen))
    canopy=assemble_canopy(surfaces,placement)
    from tools.fsp_axes import select_axes
    axes,membership=select_axes(folder,scenario,vertices)
    return dict(completion=completion,audit=audit,vertices=vertices,identities=identities,surfaces=surfaces,placement=placement,canopy=canopy,captures={pid:capmap[pid,cid] for pid,cid in chosen.items()},axes=axes,axis_membership=membership)


def read_placement(path: Path,instances: set) -> dict:
    doc=read_json(path)
    require(doc.get('schema')=='p1_state_v1' and doc.get('frame')=='canopy_mm','unsupported producer placement schema/frame')
    out={}
    for row in doc['instances']:
        pid=row['plant_instance_id']
        if pid not in instances:continue
        require(pid not in out,'duplicate instance placement')
        require(row['placement_status']=='placed' and row['in_optical_scene'] is True,'plant placement unavailable/omitted')
        transform=row['transform'];require(transform['from_frame']=='plant_local_mm' and transform['to_frame']=='canopy_mm','placement frame mismatch')
        matrix=np.asarray(transform['matrix_4x4'],dtype=float);require(matrix.shape==(4,4) and np.allclose(matrix[3],[0,0,0,1]),'homogeneous transform invalid')
        r,t=rigid(matrix[:3,:3],matrix[:3,3]);require(np.allclose(t,transform['translation_mm']),'placement translation disagrees with matrix')
        out[pid]=dict(rotation=r.tolist(),translation_mm=t.tolist(),provenance=transform['translation_source']+';'+transform['yaw_source'])
    require(set(out)==instances,'placement unavailable');return out


def validate_native_inputs(folder: Path,req: dict,plants: list) -> list:
    init=read_json(folder/'initialization.json');require(init['schema_version']==SCHEMA,'initialization schema')
    for f,h in init['files'].items():require(sha(folder/f)==h,'initial state/surface changed')
    require(set(init['files'])==set(NATIVE_FILES)-{'initialization.json'},'initialization file coverage')
    ss=rows(folder/'initial_state.csv');by={s['organ_id']:s for s in ss};pids={p['plant_instance_id'] for p in plants}
    require(len(by)==len(ss)>0,'duplicate initial state');require({s['plant_instance_id'] for s in ss}==pids,'initial plant mismatch')
    for s in ss:
        require(s['day']=='0' and int(s['birth_day'])<=0,'initial birth/time invalid')
        require(s['parent_id'] in by or s['class']=='Plant','initial parent missing')
        for key in ('mass_gDM','max_mass_gDM','thermal_age_dd','length_m','width_m','area_m2'):number(s[key],key,0)
        require(s['provenance'] in ('observation_fixture','conditional_hidden_state','observed'),'initial provenance')
    require(set(read_json(folder/'initial_controls.json'))==pids,'initial controls mismatch')
    from tools.fsp_axes import enabled,validate_records
    if enabled(req):
        records=validate_records(req,pids);controls=read_json(folder/'initial_controls.json')
        from tools.fsp_engine_hooks import planned
        for a in records:
            if planned(a):
                parent=by.get(a['plant_instance_id']+'|'+a['parent_node_id'])
                require(parent is not None and parent['class'] in ('Attachment','CompoundLeaf'), 'planned parent must be an existing leaf axil')
                require(parent['plant_instance_id']==a['plant_instance_id'] and parent['axis_id']==a['parent_axis_id'] and int(parent['true_leaf_order'])==a['parent_rank'], 'planned native attachment axis/rank mismatch')
                require(not any(s['class']=='Axis' and s['parent_id']==parent['organ_id'] for s in ss), 'occupied planned native axil')
                require(a['forecast_axis_id'] not in by,'planned axis ID already exists')
                continue
            c=controls[a['plant_instance_id']]['axes'][a['axis_id']]
            for key in ('active_axis','active_apex','last_stem'):
                require(c[key] in by and by[c[key]]['plant_instance_id']==a['plant_instance_id'],'axis control parent missing/cross-plant')
            require(by[c['active_axis']]['axis_id']==a['axis_id'],'axis control mismatch')
            if a['axis_id']!='M':require(by[c['active_axis']]['parent_id']==a['plant_instance_id']+'|'+a['parent_node_id'],'side parent node missing')
    geo=read_json(folder/'integration_geometry.json');require(set(geo['placement'])==pids,'initial placement mismatch')
    require(set(geo['triangles_mm'])=={s['organ_id'] for s in ss if s['class']=='Leaflet'},'initial triangle coverage')
    for oid,triangles in geo['triangles_mm'].items():
        area=sum(triangle_area(t) for t in triangles)*1e-6
        require(math.isclose(area,float(by[oid]['area_m2']),rel_tol=1e-8),'initial mesh area mismatch')
        require(float(by[oid]['mass_gDM'])>0 and area>0,'initial optical mass required')
    return NATIVE_FILES


def native_extension(pkg: dict,folder: Path) -> tuple[str,str]:
    """Hash-bound native initial shape and placement; no Python growth update."""
    state=pkg['tables']['initial_state.csv'];pending=list(state);ordered=[];seen=set()
    while pending:
        ready=[s for s in pending if not s['parent_id'] or s['parent_id'] in seen];require(ready,'initial topology cycle')
        for s in ready:ordered.append(s);seen.add(s['organ_id']);pending.remove(s)
    lookup={s['organ_id']:s for s in state};geo=pkg['integration_geometry'];lines=[];shape_rows=[]
    for i,s in enumerate(ordered):
        lines.append('{ Organ o=org('+str(i)+');o.imported=true;')
        for field,column in [('initialMass','mass_gDM'),('initialLength','length_m'),('initialWidth','width_m'),('initialArea','area_m2'),('initialAz','azimuth_deg'),('initialElev','elevation_deg')]:lines.append(f'o.{field}={float(s[column])};')
        parent=lookup.get(s['parent_id']);base=[float(s[k]) for k in ('x_m','y_m','z_m')]
        ref=[float(parent['tip_'+k]) if parent else 0. for k in ('x_m','y_m','z_m')]
        for k,v in zip(('offsetX','offsetY','offsetZ'),np.asarray(base)-ref):lines.append(f'o.{k}={v};')
        if s['organ_id'] in geo['triangles_mm']:
            v=np.asarray(geo['triangles_mm'][s['organ_id']])*.001-np.asarray(base)
            lines.append('o.shape=new float['+str(v.size)+'];')
            for j,x in enumerate(v.reshape(-1)):shape_rows.append(dict(organ_index=i,coordinate=j,value=float(x)))
        lines.append('}')
    # Prepared working auxiliary input avoids JVM method-size limits for real meshes.
    write_csv(folder.parent/'working/initial_shapes.csv',shape_rows,['organ_index','coordinate','value'])
    lines.append('{ BufferedReader shapeInput=new BufferedReader(new FileReader("initial_shapes.csv"));shapeInput.readLine();String line;while((line=shapeInput.readLine())!=null){String[] values=line.split(",");org(Integer.parseInt(values[0])).shape[Integer.parseInt(values[1])]=Float.parseFloat(values[2]);}shapeInput.close();}')
    p=[]
    for plant in pkg['tables']['plants.csv']:
        placement=geo['placement'][plant['plant_instance_id']];r,t=rigid(placement['rotation'],placement['translation_mm'])
        p.append('new double[]{'+','.join(str(float(x)) for x in [*r.reshape(-1),*(t*.001)])+'}')
    return '\n'.join(lines),'new double[][]{'+','.join(p)+'}'


def validate_integration_geometry(pkg: dict,data: dict,n: int) -> None:
    initial={s['organ_id']:s for s in pkg['tables']['initial_state.csv']};meshes=defaultdict(list)
    for r in data['meshes.csv']:meshes[int(r['state']),r['organ_id']].append(r)
    states={(int(s['day']),s['organ_id']):s for s in data['states.csv']};geo=pkg['integration_geometry']
    for (day,oid),s in states.items():
        if oid not in geo['triangles_mm']:continue
        original=initial[oid];mass=float(original['mass_gDM']);scale=math.sqrt(float(s['mass_gDM'])/mass)
        req=pkg['request']
        frozen=req['freeze_at']>=0 and day>req['freeze_at']
        if frozen:scale=math.sqrt(float(states[req['freeze_at'],oid]['mass_gDM'])/mass)
        original_base=np.array([float(original[k]) for k in ('x_m','y_m','z_m')]);base=np.array([float(s[k]) for k in ('x_m','y_m','z_m')])
        expected=(np.asarray(geo['triangles_mm'][oid])*.001-original_base)*scale+base
        p=geo['placement'][s['plant_instance_id']];r,t=rigid(p['rotation'],p['translation_mm']);expected=expected@r.T+t*.001
        native=np.array([[float(row[k]) for k in ('x_m','y_m','z_m')] for row in sorted(meshes[day,oid],key=lambda r:int(r['vertex']))]).reshape(-1,3,3)
        require(native.shape==expected.shape and np.allclose(native,expected,atol=8e-8,rtol=1e-6),'native did not consume placed/scaled P1 geometry')
    if not pkg['request'].get('light_only'):
        f={(r['interval'],r['organ_id']):r for r in data['flux.csv']};proposals=data['proposals.csv']
        ledger={(r['interval'],r['plant_instance_id']):r for r in data['ledger.csv']}
        require(len(proposals)==len(f),'proposal coverage')
        seen=set()
        for p in proposals:
            k=p['interval'],p['organ_id'];require(k not in seen and k in f,'duplicate/unknown proposal');seen.add(k)
            require(p['provisional']=='true','proposal must remain provisional')
            l=ledger[p['interval'],p['plant_instance_id']]
            require(math.isclose(float(p['available_gC']),float(l['reserve_start_gC'])+float(l['gross_gC'])-float(l.get('downregulated_gC',0))-float(l['maintenance_gC']),abs_tol=1e-10),'proposal budget mismatch')
            for key in ('demand_gC','allocated_gC'):require(math.isclose(float(p[key]),float(f[k][key]),abs_tol=1e-10),'allocation differs from proposal')
            old=states.get((int(p['interval']),p['organ_id']))
            require(math.isclose(float(p['mass_before_gDM']),float(old['mass_gDM']) if old else 0,abs_tol=1e-10),'proposal is not based on uncommitted starting mass')


def validate_light_only(pkg: dict,data: dict,n: int) -> dict:
    initial={s['organ_id']:s for s in pkg['tables']['initial_state.csv']}
    for s in data['states.csv']:
        require(s['organ_id'] in initial,'birth during frozen light stage')
        require(all(s[k]==v or (k not in ('organ_id','class','provenance','parent_id','axis_id','developmental_axis_id','plant_instance_id','source_plant_id') and math.isclose(float(s[k]),float(v),abs_tol=1e-10)) for k,v in initial[s['organ_id']].items() if k!='day'),'state advanced in light-only stage')
    require(len(data['states.csv'])==(n+1)*len(initial) and not data['events.csv'],'light-only state/event coverage')
    for r in data['ledger.csv']:require(float(r['reserve_start_gC'])==float(r['reserve_end_gC']) and all(float(r[k])==0 for k in ('gross_gC','maintenance_gC','structural_gC','growth_resp_gC','deficit_gC')),'light-only carbon committed')
    mesh={(r['state'],r['organ_id'],r['vertex']):(r['x_m'],r['y_m'],r['z_m']) for r in data['meshes.csv']}
    consumed={(r['state'],r['organ_id'],r['vertex']):(r['x_m'],r['y_m'],r['z_m']) for r in data['consumed.csv']}
    require(consumed=={k:v for k,v in mesh.items() if int(k[0])<n},'stale light-only geometry')
    for r in data['flux.csv']:number(r['absorbed_umol_s'],'absorbed light',0)
    validate_integration_geometry(pkg,data,n)
    return dict(software_status='verified_native_light_only',intervals=n,new_organs=0,determinate_stops=0,max_carbon_residual_gC=0,biological_validation='not_established')


def make_native_inputs(selected: dict,scenario: dict,out: Path) -> Path:
    """Convert supported P1 structure, using only separately supplied hidden state."""
    from tools.vtc_development import demo,KINDS
    from datetime import timedelta
    out=demo(out,scenario['intervals'],scenario['rays'],seed=scenario['seed'])
    req=read_json(out/'request.json');req.update(run_id=out.parent.name,initialization_route='p1_conditioned',integration_contract=SCHEMA,
        fixture=scenario['fixture'],start_time=scenario['start_time'],dt_s=scenario.get('dt_s',86400),light_only=scenario.get('light_only',False),
        freeze_at=0 if scenario.get('freeze_geometry') or scenario.get('light_only') else -1,
        imported_shape_policy='fixed_initial_posture_isotropic_mass_scaling',initial_geometry_frame='plant_local_mm',native_mesh_frame='canopy_m')
    req['parameters']=scenario['parameters']
    for key in ('engine_hooks','native_boundary','imported_shape_api','resource_hook','plant_positions'):
        if key in scenario:req[key]=scenario[key]
    if selected.get('axes'):
        for key in ('axis_contract','axes','topology_scope','topology_completeness'):req[key]=scenario[key]
        req['axes']=selected['axes']
    if scenario.get('extensions'):
        from tools.fsp_extensions import piecewise_schedule
        req['extensions']=scenario['extensions']
        if req['extensions'].get('schedule'):req['schedule_intervals']=piecewise_schedule(req['extensions']['schedule'],scenario['lamps'])
    write_json(out/'request.json',req)
    require(scenario.get('initialization_provenance'),'hidden-state provenance required')
    vertices=selected['vertices'];vmap={(r['plant_instance_id'],r['vertex_id']):r for r in vertices}
    surfaces={s['organ_id']:s for s in selected['surfaces']};state=[];plants=[];controls={};idmap=[]
    hidden=scenario['organ_initialization'];plant_hidden=scenario['plant_initialization']
    require(set(plant_hidden)==set(selected['captures']),'plant physiological initialization missing')
    for pid,cap in selected['captures'].items():
        pv=[v for v in vertices if v['plant_instance_id']==pid];ph=plant_hidden[pid]
        for key in ('reserve_gC','root_mass_gDM','stem_mass_gDM','thermal_dd','produced','leaf_order','truss_count','axis_leaf_count'):number(ph[key],key,0)
        plants.append(dict(plant_instance_id=pid,source_plant_id=cap['source_plant_id'],x_m=0,y_m=0,z_m=0,reserve_gC=ph['reserve_gC'],provenance=scenario['initialization_provenance']))
        roots=[v for v in pv if v['class']=='Plant'];axes=[v for v in pv if v['class']=='Axis']
        from tools.fsp_engine_hooks import planned
        axis_records=[a for a in selected.get('axes',[]) if a['plant_instance_id']==pid and not planned(a)]
        require(len(roots)==1 and (axis_records or len(axes)==1),'explicit source axes required')
        declared={a['axis_id']:a for a in axis_records}
        main_vertex=declared['M']['axis_vertex_id'] if declared else axes[0]['vertex_id']
        root_id=pid+'|'+roots[0]['vertex_id'];axis_id=pid+'|'+main_vertex
        for v in pv:
            oid=pid+'|'+v['vertex_id'];kind={'Metamer':'Attachment','Truss':'TrussAxis'}.get(v['class'],v['class']);require(kind in KINDS,'unsupported P1 organ class')
            h=hidden.get(oid);require(h is not None,'missing organ physiology assumption: '+oid)
            for k in ('mass_gDM','max_mass_gDM','thermal_age_dd','true_leaf_order'):number(h[k],k,0)
            require(h.get('provenance') and h.get('units')=={'mass':'gDM','age':'degree_day'},'organ hidden-state units/provenance')
            parent=v['parent_id'] if v['parent_id'] not in ('','unavailable') else v['complex_id']
            if declared and kind=='Axis':
                ar=declared[selected['axis_membership'][pid,v['vertex_id']]]
                if ar['axis_id']!='M':parent=ar['parent_node_id']
            parent_id='' if kind=='Plant' else pid+'|'+parent
            require(not parent_id or (pid,parent) in vmap,'unsupported MTG parent/complex reference')
            if kind=='Plant':base=tip=np.zeros(3)
            elif kind=='TrussAxis' and (v['base_xyz_mm__status']=='unavailable' or v['tip_xyz_mm__status']=='unavailable'):
                require(h.get('geometry_source_kind') in ('user_assumed','supplement_provided') and h.get('geometry_source_ref'),'unlocated truss requires explicit geometry supplement')
                points=[]
                for field in ('base_xyz_m','tip_xyz_m'):
                    values=h.get(field);require(isinstance(values,list) and len(values)==3,'truss supplement requires three coordinates')
                    points.append(np.asarray([number(x,field) for x in values],dtype=float))
                base,tip=points
            else:
                require(v['base_xyz_mm__status']!='unavailable' and v['tip_xyz_mm__status']!='unavailable','located organ points required')
                base=np.asarray(json.loads(v['base_xyz_mm']),dtype=float)*.001;tip=np.asarray(json.loads(v['tip_xyz_mm']),dtype=float)*.001
            vector=tip-base;length=float(np.linalg.norm(vector));width=number(h.get('width_m',0),'explicit non-leaf width',0);area=0.
            if kind in ('Plant','CompoundLeaf','Attachment'):tip=base;vector=tip-base;length=width=0.
            if kind=='Leaflet':
                require(oid in surfaces,'leaflet shape absent');area=sum(triangle_area(t) for t in surfaces[oid]['triangles_mm'])*1e-6
                width=number(v['width_mm'],'leaflet width',0)*.001
            if kind=='Axis':width=0.
            az=math.degrees(math.atan2(vector[1],vector[0]))%360 if length else 0.
            el=math.degrees(math.asin(vector[2]/length)) if length else 0.
            side={'L':-1,'R':1,'left':-1,'right':1}.get(v['side'],0)
            row=dict(day=0,plant_instance_id=pid,source_plant_id=cap['source_plant_id'],organ_id=oid,parent_id=parent_id,axis_id=axis_id if kind!='Plant' else '',
                **{'class':kind},true_leaf_order=int(h['true_leaf_order']),birth_day=-1,emergence_day=0,thermal_age_dd=h['thermal_age_dd'],mass_gDM=h['mass_gDM'],max_mass_gDM=h['max_mass_gDM'],
                x_m=base[0],y_m=base[1],z_m=base[2],tip_x_m=tip[0],tip_y_m=tip[1],tip_z_m=tip[2],length_m=length,width_m=width,area_m2=area,
                rachis_bend_alpha_deg=0.,azimuth_deg=az,elevation_deg=el,pair_index=int(v['pair_index']) if v['pair_index'] not in ('','unavailable') else 0,side=side,
                provenance='observation_fixture' if scenario['fixture'] else 'conditional_hidden_state')
            if declared:
                from tools.fsp_axes_segments import segment
                logical=selected['axis_membership'][pid,v['vertex_id']];ar=declared[logical]
                row.update(axis_id=logical,developmental_axis_id=pid+'|'+ar['axis_vertex_id'] if kind!='Plant' else '',parent_rank=ar['parent_rank'])
                row['segment_index']=segment(v)
            state.append(row);idmap.append(dict(plant_instance_id=pid,producer_vertex_id=v['vertex_id'],native_organ_id=oid,source_plant_id=cap['source_plant_id']))
        def extra(suffix: str,kind: str,parent: str,mass: float,maxmass: float,logical: str='M') -> dict:
            proto={k:0 for k in state[-1]};proto.update(day=0,plant_instance_id=pid,source_plant_id=cap['source_plant_id'],organ_id=pid+'/'+suffix,parent_id=parent,axis_id=axis_id,
                **{'class':kind},true_leaf_order=0,birth_day=-1,emergence_day=0,thermal_age_dd=ph['thermal_dd'],mass_gDM=mass,max_mass_gDM=maxmass,provenance='conditional_hidden_state')
            if declared:proto.update(axis_id=logical,developmental_axis_id=pid+'|'+declared[logical]['axis_vertex_id'],parent_rank=declared[logical]['parent_rank'])
            if declared:proto['segment_index']=declared[logical]['segment_index'] if kind!='Root' else 'unavailable'
            state.append(proto);return proto
        extra('Root','Root',root_id,ph['root_mass_gDM'],max(ph['root_mass_gDM'],req['parameters']['MAX_MASS']['value'][5]))
        if declared:
            axis_controls={}
            for logical,ar in declared.items():
                ah=ar['initialization'];aid=pid+'|'+ar['axis_vertex_id'];axis=next(s for s in state if s['organ_id']==aid)
                stem=extra(logical+'/InitialStem','Internode',aid,ah['stem_mass_gDM'],ah['stem_mass_gDM']*1.4,logical)
                for k in ('x_m','y_m','z_m','tip_x_m','tip_y_m','tip_z_m','length_m','azimuth_deg','elevation_deg'):stem[k]=axis[k]
                stem['width_m']=ah['stem_width_m'];stem['true_leaf_order']=ah['leaf_order'];stem['thermal_age_dd']=ah['thermal_dd']
                apex=extra(logical+'/InitialApex','Apex',aid,0,0,logical);apex['thermal_age_dd']=ah['thermal_dd']
                axis_controls[logical]=dict(active_apex=apex['organ_id'],active_axis=aid,last_stem=stem['organ_id'],plant_root=root_id,
                    **{k:ah[k] for k in ('thermal_dd','produced','leaf_order','truss_count','axis_leaf_count')},stopped_apices=[apex['organ_id']] if ah['apex_stopped'] else [],provenance=ah['source_kind']+':'+ah['source_ref'])
            controls[pid]=dict(axis_controls['M'],axes=axis_controls)
            continue
        axis=next(s for s in state if s['organ_id']==axis_id)
        stem=extra('InitialStem','Internode',root_id,ph['stem_mass_gDM'],ph['stem_mass_gDM']*1.4)
        for k in ('x_m','y_m','z_m','tip_x_m','tip_y_m','tip_z_m','length_m','azimuth_deg','elevation_deg'):stem[k]=axis[k]
        stem['width_m']=ph['stem_width_m'];stem['true_leaf_order']=ph['leaf_order']
        apex=extra('InitialApex','Apex',axis_id,0,0)
        controls[pid]=dict(active_apex=apex['organ_id'],active_axis=axis_id,last_stem=stem['organ_id'],plant_root=root_id,
            **{k:ph[k] for k in ('thermal_dd','produced','leaf_order','truss_count','axis_leaf_count')},stopped_apices=[],provenance=scenario['initialization_provenance'])
    require(set(hidden)=={r['native_organ_id'] for r in idmap},'unknown initialization organ IDs')
    by_state={s['organ_id']:s for s in state}
    for s in state:
        if s['class']=='RachisSegment':
            parent=by_state[s['parent_id']]
            a=np.array([parent['tip_'+k]-parent[k] for k in ('x_m','y_m','z_m')]);b=np.array([s['tip_'+k]-s[k] for k in ('x_m','y_m','z_m')])
            if np.linalg.norm(a)*np.linalg.norm(b)>0:
                angle=math.degrees(math.acos(float(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
                s['rachis_bend_alpha_deg']=angle if s['elevation_deg']<=parent['elevation_deg'] else -angle
    write_csv(out/'initial_state.csv',state);write_csv(out/'plants.csv',plants);write_json(out/'initial_controls.json',controls)
    geo=dict(scale_plane_evidence=selected['audit']['scale_plane_evidence'],completion=selected.get('completion'),placement=selected['placement'],triangles_mm={k:s['triangles_mm'] for k,s in surfaces.items()},identity_map=idmap)
    write_json(out/'integration_geometry.json',geo)
    write_json(out/'initialization.json',dict(schema_version=SCHEMA,source_package_hash=digest(selected['audit']['source_hashes']),provenance=scenario['initialization_provenance'],
        assumptions=scenario['organ_initialization'],plant_assumptions=plant_hidden,files={f:sha(out/f) for f in NATIVE_FILES if f!='initialization.json'}))
    env=scenario['environment'];require(len(env)==scenario['intervals'],'interval environment coverage')
    from tools.fsp_extensions import boundary_time
    write_csv(out/'environment.csv',[dict(interval=i,start_time=boundary_time(req,i).isoformat(),**e) for i,e in enumerate(env)])
    write_csv(out/'lamps.csv',scenario['lamps'])
    return out
