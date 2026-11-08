"""Read-only Stage-2 boundary with an explicit conditional physiology supplement.

No photograph supplies mass, age, depth, reserve or native apex state implicitly.
Fixture export exercises the agreed Stage-2 columns; it is not a real producer.
"""
from __future__ import annotations
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from tools.vtc_growth_io import require, rows, read_json, write_json, write_csv, sha, number, identifier

FILES=['initial_state.csv','initial_controls.json','mtg_vertices.csv','mtg_edges.csv','structure_state.csv','observation_manifest.json']
SUPPORTED={'Plant':1,'Axis':2,'CompoundLeaf':3,'Attachment':3,'Internode':4,'Petiole':4,'RachisSegment':4,'Leaflet':4,'Apex':4}
#: Engine -> MTG rank export (MTG Amendment A7, U5). The engine's true_leaf_order is a per-axis NODE index; the MTG
#: rank is derived only through tomato3d.state.ranks.u5_rank_of_node (main stem: nodes 1 and 2 share rank 1; side
#: axes: one rank per node because the A4 15.5 tie rule is still open - not decided here).
RANK_BASIS_MAIN='A7_u5_rank_of_node(true_leaf_order,shared_first=True):main_stem'
RANK_BASIS_SIDE='A7_u5_rank_of_node(true_leaf_order,shared_first=False):side_axis_tie_rule_A4_15.5_open'
#: Revision of the P1 MTG rank/node-index convention the observation package declares (p1_state_v1 revision 9 =
#: node_index_from_base beside every CompoundLeaf rank). Packages without it are pre-A7 and are read as before.
NODE_INDEX_REVISION=9


def main_axis(axis_id: str) -> bool:
    """Engine main stem: native axis id '<plant>/A0' or the logical axis 'M' (fsp-axes)."""
    return axis_id=='M' or axis_id.endswith('/A0')


def mtg_rank(state: dict) -> tuple[int,str]:
    """(MTG U5 rank, basis) of an engine CompoundLeaf from its node index; never a +/-1 shortcut."""
    from tomato3d.state.ranks import u5_rank_of_node
    main=main_axis(state['axis_id'])
    return u5_rank_of_node(int(state['true_leaf_order']),shared_first=main),(RANK_BASIS_MAIN if main else RANK_BASIS_SIDE)


def fixture_from_native(run_dir: Path, out: Path, day: int=30, intervals: int=5) -> Path:
    from tools.vtc_development import demo, verify_prepared
    prepared,pkg=verify_prepared(run_dir)
    require(read_json(run_dir/'execution.json')['status']=='verified','verified native fixture source required')
    out=demo(out,intervals,100000,scenario='sufficient')
    req=read_json(out/'request.json');req['initialization_route']='stage2_observation_fixture';req['parameters']=pkg['request']['parameters'];req['observation_biological_age_days']=day
    write_json(out/'request.json',req)
    state=[dict(s) for s in rows(run_dir/'working/states.csv') if int(s['day'])==day]
    require(bool(state),'requested source snapshot absent')
    for s in state:
        s['day']='0';s['birth_day']=str(int(s['birth_day'])-day)
        s['emergence_day']='0' if int(s['emergence_day'])>=0 else '-1';s['provenance']='observation_fixture'
    write_csv(out/'initial_state.csv',state)
    apex_rows=[s for s in rows(run_dir/'working/apices.csv') if int(s['day'])==day]
    controls={};plants=pkg['tables']['plants.csv'];ledger=rows(run_dir/'working/ledger.csv')
    for p in plants:
        pid=p['plant_instance_id'];ss=[s for s in state if s['plant_instance_id']==pid];by_id={s['organ_id']:s for s in ss}
        active=next((a for a in apex_rows if a['organ_id'] in by_id and a['stopped']=='false'),None)
        if active is None:active=[a for a in apex_rows if a['organ_id'] in by_id][-1]
        axis=by_id[active['organ_id']]['axis_id'];stems=[s for s in ss if s['axis_id']==axis and s['class']=='Internode']
        last=max(stems,key=lambda s:int(s['true_leaf_order']))['organ_id'] if stems else axis
        controls[pid]=dict(active_apex=active['organ_id'],active_axis=axis,last_stem=last,plant_root=next(s['organ_id'] for s in ss if s['class']=='Plant'),
            thermal_dd=float(active['plant_thermal_dd']),produced=int(active['produced_events']),leaf_order=max(int(s['true_leaf_order']) for s in ss),
            truss_count=sum(s['class']=='TrussAxis' for s in ss),axis_leaf_count=sum(s['class']=='CompoundLeaf' and s['axis_id']==axis for s in ss),
            stopped_apices=[a['organ_id'] for a in apex_rows if a['organ_id'] in by_id and a['stopped']=='true'],
            provenance='conditional_state_from_synthetic_native_fixture')
        p['reserve_gC']=next(r['reserve_end_gC'] for r in ledger if r['plant_instance_id']==pid and int(r['interval'])==day-1)
    write_csv(out/'plants.csv',plants);write_json(out/'initial_controls.json',controls)
    vertices=[];edges=[];struct=[];all_states={s['organ_id']:s for s in state}
    for s in state:
        kind=s['class']
        if kind not in SUPPORTED:continue
        oid=s['organ_id'];parent=s['parent_id']
        # Reserved root/reproductive classes stay in the explicit physiology supplement.
        scale=SUPPORTED[kind];ancestor=parent;within='';complex_id='unavailable'
        while ancestor:
            a=all_states[ancestor];a_scale=SUPPORTED.get(a['class'])
            if a_scale==scale and not within:within=ancestor
            if a_scale==scale-1:complex_id=ancestor;break
            ancestor=a['parent_id']
        if kind=='Internode':
            complex_id=next((a['organ_id'] for a in state if a['class']=='Attachment' and a['parent_id']==oid),'unavailable')
        v=dict(frame='plant_local_mm',snapshot_capture_id='synthetic_day30',vertex_id=oid,scale=scale,
               **{'class':'Metamer' if kind=='Attachment' else kind},complex_id=complex_id,parent_id=within,
               edge_type='<' if kind in ('Internode','RachisSegment') or (kind=='Leaflet' and int(s['side'])==0) else '+',
               base_xyz_mm=json.dumps([float(s[k])*1000 for k in ('x_m','y_m','z_m')]),tip_xyz_mm=json.dumps([float(s[k])*1000 for k in ('tip_x_m','tip_y_m','tip_z_m')]),
               length_mm=float(s['length_m'])*1000,width_mm=float(s['width_m'])*1000,area_mm2=float(s['area_m2'])*1e6,
               azimuth_deg=float(s['azimuth_deg']),elevation_deg=float(s['elevation_deg']),normal_xyz='unavailable',
               rank=mtg_rank(s)[0] if kind=='CompoundLeaf' else 'unavailable',
               node_index_from_base=int(s['true_leaf_order']) if kind=='CompoundLeaf' else 'unavailable',order=0 if main_axis(s['axis_id']) else 1)
        for key in ('base_xyz_mm','tip_xyz_mm','length_mm','width_mm','area_mm2','azimuth_deg','elevation_deg','normal_xyz','rank','node_index_from_base','order'):
            v[key+'__status']='unavailable' if v[key]=='unavailable' else 'model_estimated'
        vertices.append(v)
        if within:edges.append(dict(snapshot_capture_id='synthetic_day30',parent_id=within,child_id=oid,edge_type=v['edge_type']))
        struct.append(dict(plant_id=s['source_plant_id'],capture_id='synthetic_day30',vertex_id=oid,part=kind,upstream_track_state='active',link_status='confirmed',time_since_lights_on_min='unavailable',photoperiod_phase='unavailable',frame='plant_local_mm',scale_status='solved',evidence_ref='synthetic_native_fixture:'+oid))
    write_csv(out/'mtg_vertices.csv',vertices);write_csv(out/'mtg_edges.csv',edges);write_csv(out/'structure_state.csv',struct)
    manifest=dict(fixture=True,producer='exact_column_fixture_not_phynotype2',source_manifest_sha256=sha(run_dir/'prepared.json'),
        rotation_status='solved',transform_source='explicit_fixture_identity_plant_local_to_canopy',scale_source='native_m_to_mm_fixture',
        biology_status='not_established',supplement_status='conditional_assumptions',schema_revision=NODE_INDEX_REVISION,
        rank_convention='U5_A7: rank = u5_rank_of_node(node_index_from_base); main stem shared first rank, side axes one rank per node (A4 15.5 open)',files={f:sha(out/f) for f in FILES if f!='observation_manifest.json'})
    write_json(out/'observation_manifest.json',manifest)
    return out


def validate_initial(folder: Path,req: dict,plants: list[dict]) -> list[str]:
    manifest=read_json(folder/'observation_manifest.json')
    require(manifest['fixture']==req['fixture'],'observation fixture status mismatch')
    require(manifest.get('rotation_status')=='solved' and manifest.get('transform_source') and manifest.get('scale_source'),'missing frame/scale evidence')
    require(set(manifest['files'])==set(FILES)-{'observation_manifest.json'},'observation file coverage')
    for name,value in manifest['files'].items():require(sha(folder/name)==value,'observation source changed')
    state=rows(folder/'initial_state.csv');smap={s['organ_id']:s for s in state};require(len(smap)==len(state)>0,'duplicate/empty initial state')
    pmap={p['plant_instance_id']:p for p in plants}
    for s in state:
        for k in ('organ_id','plant_instance_id','source_plant_id'):identifier(s[k])
        require(s['plant_instance_id'] in pmap,'unknown initial plant')
        require(s['source_plant_id']==pmap[s['plant_instance_id']]['source_plant_id'],'source identity mismatch')
        require(s['parent_id'] in smap or s['class']=='Plant','initial orphan')
        require(s['axis_id'] in smap or s['class']=='Plant','initial axis missing')
        for k in ('mass_gDM','max_mass_gDM','thermal_age_dd','length_m','width_m','area_m2'):number(s[k],k,0)
        require(s['day']=='0' and int(s['birth_day'])<=0,'observation age policy')
        require(s['provenance']=='observation_fixture' if req['fixture'] else s['provenance'] in ('observed','conditional_hidden_state'),'initial provenance mismatch')
    controls=read_json(folder/'initial_controls.json');require(set(controls)==set(pmap),'initial plant controls missing')
    for pid,c in controls.items():
        for key in ('active_apex','active_axis','last_stem','plant_root'):
            require(c[key] in smap and smap[c[key]]['plant_instance_id']==pid,'initial control identity mismatch')
        for key in ('thermal_dd','produced','leaf_order','truss_count','axis_leaf_count'):number(c[key],key,0)
        require(c.get('provenance'),'hidden developmental-state source required')
    vertices=rows(folder/'mtg_vertices.csv');vmap={v['vertex_id']:v for v in vertices}
    require(len(vmap)==len(vertices),'duplicate Stage-2 vertex')
    require(set(vmap)=={oid for oid,s in smap.items() if s['class'] in SUPPORTED},'Stage-2/supplement identity mismatch')
    struct=rows(folder/'structure_state.csv');require({s['vertex_id'] for s in struct}==set(vmap),'Stage-2 state identity coverage')
    require(len(struct)==len(vmap),'duplicate Stage-2 structure row')
    for s in struct:
        require(s['frame']=='plant_local_mm' and s['scale_status']=='solved','Stage-2 frame/scale unavailable')
        require(s['link_status']=='confirmed','Stage-2 identity not confirmed')
    edge_rows=rows(folder/'mtg_edges.csv')
    actual={(e['parent_id'],e['child_id'],e['edge_type']) for e in edge_rows}
    expected={(v['parent_id'],v['vertex_id'],v['edge_type']) for v in vertices if v['parent_id']}
    require(len(actual)==len(edge_rows) and actual==expected,'Stage-2 edge coverage mismatch')
    for v in vertices:
        require(v['frame']=='plant_local_mm','board frame needs an explicit transform; no relabelling')
        s=smap[v['vertex_id']]
        require(v['class']==('Metamer' if s['class']=='Attachment' else s['class']) and int(v['scale'])==SUPPORTED[s['class']],'Stage-2 class/scale mismatch')
        if v['parent_id']:require(v['parent_id'] in vmap and vmap[v['parent_id']]['scale']==v['scale'],'Stage-2 parent scale mismatch')
        for field,keys,factor in [('base_xyz_mm',('x_m','y_m','z_m'),1000),('tip_xyz_mm',('tip_x_m','tip_y_m','tip_z_m'),1000)]:
            require(v[field+'__status'] in ('observed','model_estimated'),'metric geometry unavailable')
            val=json.loads(v[field]);require(len(val)==3,'3D geometry required')
            for x,k in zip(val,keys):require(math.isclose(number(x,k),float(s[k])*factor,abs_tol=1e-8),'Stage-2 metre/mm mismatch')
        for field,k,factor in [('length_mm','length_m',1000),('width_mm','width_m',1000),('area_mm2','area_m2',1e6)]:
            require(v[field+'__status'] in ('observed','model_estimated'),'unsupported geometry status')
            require(math.isclose(number(v[field],field),float(s[k])*factor,abs_tol=1e-8),'Stage-2 metric mismatch')
        if int(manifest.get('schema_revision',0))>=NODE_INDEX_REVISION:check_rank_node_cells(v,s)
    return FILES


def check_rank_node_cells(v: dict,s: dict) -> None:
    """Revision >= 9 observation vertex: node_index_from_base is the engine node index and rank its U5 rank
    (tomato3d.state.schema.check_rank_node; main stem shared first rank, side axes one rank per node)."""
    from tomato3d.state.schema import check_rank_node, P1SchemaError
    require('node_index_from_base' in v and 'node_index_from_base__status' in v,'revision-9 observation lacks node_index_from_base')
    for key in ('rank','node_index_from_base'):
        require((v[key]=='unavailable')==(v[key+'__status']=='unavailable'),f'{key} and its status disagree')
    if s['class']!='CompoundLeaf':
        require(v['rank']=='unavailable' and v['node_index_from_base']=='unavailable','rank/node index on a non-leaf vertex');return
    require(v['node_index_from_base']==str(int(s['true_leaf_order'])),'node_index_from_base differs from the engine node index')
    try:check_rank_node(v['rank'],v['node_index_from_base'],main_axis(s['axis_id']),'observation mtg_vertices.csv '+v['vertex_id'])
    except P1SchemaError as e:require(False,str(e))


def native_initialization(pkg: dict,folder: Path) -> str:
    from tools.vtc_development_io import KINDS
    state=pkg['tables']['initial_state.csv'];ordered=[];pending=list(state);known=set()
    while pending:
        ready=[s for s in pending if not s['parent_id'] or s['parent_id'] in known]
        require(bool(ready),'initial topology cycle')
        for s in ready:ordered.append(s);known.add(s['organ_id']);pending.remove(s)
    indices={s['organ_id']:i for i,s in enumerate(ordered)};pmap={p['plant_instance_id']:i for i,p in enumerate(pkg['tables']['plants.csv'])}
    lines=[]
    fields=dict(thermal='thermal_age_dd',mass='mass_gDM',maxMass='max_mass_gDM',x='x_m',y='y_m',z='z_m',tx='tip_x_m',ty='tip_y_m',tz='tip_z_m',len='length_m',wid='width_m',area='area_m2',az='azimuth_deg',elev='elevation_deg',alpha='rachis_bend_alpha_deg')
    for i,s in enumerate(ordered):
        p=pmap[s['plant_instance_id']];parent=indices.get(s['parent_id'],-1);axis=indices.get(s.get('developmental_axis_id',s['axis_id']),-1)
        lines.append('{ Organ o=add('+','.join(map(str,[p,parent,axis,KINDS.index(s['class']),int(s['true_leaf_order']),int(s['birth_day'])]))+',"initial'+str(i)+'");')
        lines.append('o.id='+json.dumps(s['organ_id'])+';o.origin='+json.dumps(s['provenance'])+';')
        for field,column in fields.items():lines.append(f'o.{field}={number(s[column],column)};')
        from tools.fsp_axes_segments import segment
        value=segment(s)
        lines.append(f'o.segment={value if value!="unavailable" else -1};')
        for field,column in [('pair','pair_index'),('side','side'),('emerged','emergence_day')]:lines.append(f'o.{field}={int(s[column])};')
        lines.append('}')
    controls=read_json(folder/'initial_controls.json')
    for pid,c in controls.items():
        p=pmap[pid]
        for native,key in [('activeApex','active_apex'),('activeAxis','active_axis'),('lastStem','last_stem'),('plantRoots','plant_root')]:lines.append(f'{native}[{p}]={indices[c[key]]};')
        for native,key in [('clocks','thermal_dd'),('produced','produced'),('leafOrders','leaf_order'),('trussCounts','truss_count'),('axisLeafCounts','axis_leaf_count')]:lines.append(f'{native}[{p}]={c[key]};')
        for oid in c['stopped_apices']:lines.append(f'org({indices[oid]}).stopped=true;')
    from tools.fsp_axes import enabled,native_setup
    if enabled(pkg['request']):
        _,_,records=native_setup(pkg['request'],pkg['tables']['plants.csv'])
        controller={(a['plant_instance_id'],a['axis_id']):i for i,a in enumerate(records)}
        for s in ordered:lines.append(f'org({indices[s["organ_id"]]}).controller={controller[s["plant_instance_id"],s["axis_id"]]};')
        from tools.fsp_engine_hooks import planned
        for i,a in enumerate(records):
            if planned(a):continue
            c=controls[a['plant_instance_id']]['axes'][a['axis_id']]
            for native,key in [('activeApex','active_apex'),('activeAxis','active_axis'),('lastStem','last_stem')]:lines.append(f'{native}[{i}]={indices[c[key]]};')
            for native,key in [('clocks','thermal_dd'),('produced','produced'),('leafOrders','leaf_order'),('trussCounts','truss_count'),('axisLeafCounts','axis_leaf_count')]:lines.append(f'{native}[{i}]={c[key]};')
            for oid in c['stopped_apices']:lines.append(f'org({indices[oid]}).stopped=true;')
    return '\n'.join(lines)


def derived_tables(folder: Path,pkg: dict) -> None:
    """Geometric/accounting summaries of returned states, never state updates."""
    data=rows(folder/'states.csv');by_day=defaultdict(dict)
    for s in data:by_day[int(s['day'])][s['organ_id']]=s
    compounds=[];leaflets=[];plant_metrics=[]
    mesh_top=defaultdict(list)
    for r in rows(folder/'meshes.csv'):mesh_top[int(r['state']),r['organ_id']].append(float(r['z_m']))
    max_pairs=max(pkg['request']['parameters']['PAIRS']['value'])
    def vec(s: dict) -> list[float]:return [float(s['tip_'+k])-float(s[k]) for k in ('x_m','y_m','z_m')]
    def angle(a: list,b: list) -> float|str:
        scale=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
        return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(a,b))/scale)))) if scale else 'unavailable'
    for day,ss in by_day.items():
        descendants=defaultdict(list)
        for s in ss.values():
            ancestor=s['parent_id']
            while ancestor:
                descendants[ancestor].append(s);ancestor=ss[ancestor]['parent_id']
        for oid,s in ss.items():
            if s['class']=='CompoundLeaf':
                children=descendants[oid];pet=next(x for x in children if x['class']=='Petiole');rach=sorted([x for x in children if x['class']=='RachisSegment'],key=lambda x:int(x['pair_index']))
                terminal=next(x for x in children if x['class']=='Leaflet' and x['side']=='0')
                stem=ss[ss[s['parent_id']]['parent_id']];alphas=[float(x['rachis_bend_alpha_deg']) for x in rach]
                bvec=[float(terminal['tip_'+k])-float(pet['tip_'+k]) for k in ('x_m','y_m','z_m')]
                blade_az=(math.degrees(math.atan2(bvec[1],bvec[0]))+360)%360 if math.hypot(*bvec[:2])>0 else 'unavailable'
                geometry_known=all(float(x['length_m'])>0 for x in [pet,*rach])
                rank,basis=mtg_rank(s)
                record=dict(day=day,organ_id=oid,true_leaf_order=s['true_leaf_order'],node_index_from_base=s['true_leaf_order'],mtg_rank=rank,rank_basis=basis,
                    compound_insertion_angle_deg=angle(vec(stem),vec(pet)),n_bend_nodes=len(rach),rachis_curvature_deg=sum(alphas) if geometry_known else 'unavailable',theta_deg=abs(sum(alphas[:-1])) if len(rach)>1 and geometry_known else 'unavailable',
                    petiole_base_azimuth_deg=pet['azimuth_deg'] if float(pet['length_m'])>0 else 'unavailable',petiole_base_azimuth_status='model_estimated' if float(pet['length_m'])>0 else 'unavailable',
                    blade_azimuth_deg=blade_az,blade_azimuth_status='model_estimated' if blade_az!='unavailable' else 'unavailable',blade_azimuth_basis='terminal_tip',
                    leaflet_count=sum(x['class']=='Leaflet' for x in children),provenance=s['provenance'])
                for i in range(max_pairs):record[f'rachis_bend_alpha_{i+1}_deg']=alphas[i] if i<len(alphas) else 'unavailable'
                compounds.append(record)
            if s['class']=='Leaflet':
                leaflets.append(dict(day=day,organ_id=oid,leaflet_insertion_angle_deg=angle(vec(ss[s['parent_id']]),vec(s)),
                    area_basis='full_outline',surface_role='individual_leaflet_polygon',outline_source='conditional_diamond',provenance=s['provenance']))
        for p in pkg['tables']['plants.csv']:
            ps=[s for s in ss.values() if s['plant_instance_id']==p['plant_instance_id']]
            from tools.fsp_extensions import boundary_time
            elapsed=(boundary_time(pkg['request'],day)-boundary_time(pkg['request'],0)).total_seconds()/86400
            plant_metrics.append(dict(day=day,days_after_sowing=elapsed if pkg['request']['initialization_route']=='seed' else 'unavailable',plant_instance_id=p['plant_instance_id'],
                stem_length_mm=sum(float(s['length_m'])*1000 for s in ps if s['class']=='Internode'),
                height_top_leaf_mm=max([z for s in ps if s['class']=='Leaflet' for z in mesh_top[day,s['organ_id']]],default=float(p['z_m']))*1000-float(p['z_m'])*1000,
                structural_mass_gDM=sum(float(s['mass_gDM']) for s in ps),root_mass_gDM=sum(float(s['mass_gDM']) for s in ps if s['class']=='Root'),
                compounds=sum(s['class']=='CompoundLeaf' for s in ps),trusses=sum(s['class']=='TrussAxis' for s in ps)))
    if compounds:write_csv(folder/'compound_traits.csv',compounds)
    if leaflets:write_csv(folder/'leaflet_traits.csv',leaflets)
    write_csv(folder/'plant_metrics.csv',plant_metrics)
    sensors=rows(folder/'sensors.csv');total=sum(float(s['absorbed_umol_s'])/float(s['area_m2']) for s in sensors)
    measured={(r.get('grid_row'),r.get('grid_col')):r.get('ratio_to_mean') for r in pkg['ratios']}
    comparison=[]
    for s in sensors:
        value=float(s['absorbed_umol_s'])/float(s['area_m2'])
        comparison.append(dict(grid_row=s['grid_row'],grid_col=s['grid_col'],simulated_ppfd_umol_m2_s=value,simulated_ratio=value/(total/16) if total else 'unavailable',
            phone_ratio=measured.get((s['grid_row'],s['grid_col']),'unavailable'),physical_validation=False,measurement_grid='hypothetical'))
    write_csv(folder/'light_validation.csv',comparison)
