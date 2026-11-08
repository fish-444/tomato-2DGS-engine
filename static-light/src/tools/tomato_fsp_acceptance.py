"""Consumer boundary fixtures made by actual Claude P1 exporter functions."""
from __future__ import annotations
import argparse
import math
from datetime import timedelta
from pathlib import Path
from dataclasses import replace
from tools.vtc_growth_io import ROOT,fresh_build,rows,write_csv,write_json,read_json,sha,aware,require
from tools.vtc_development import parameters
from tools.tomato_fsp_io import audit_p1_package,SCHEMA


def producer_fixture(out: Path,at: str='2026-10-11T06:00:00+09:00') -> Path:
    """Call P1 truth/area/optical/placement APIs. This is not a tracking implementation."""
    from tomato3d.state import fixture_truth as ft,schema as s,optical,canopy
    from tomato3d.state.topraster import OrthoCamera
    from tomato3d.io.tables import write_table
    out=fresh_build(out);opt_dir=out/'optical_surfaces';opt_dir.mkdir()
    source=ROOT/'build/synthetic_dwarf_tomato_20261008_07/O-mature/scenes/scene_0014'
    plant=ft.load_plant(source,'fixture_a');other=ft.load_plant(source,'fixture_b')
    # Entire source plants, including their real exporter meshes, not native diamonds.
    camera=OrthoCamera(image_size_px=512,mm_per_px=.6);render=ft.render_capture(plant,0,camera);render_b=ft.render_capture(other,0,camera)
    captures=[];vertices=[];areas=[];identities=[];surfaces=[];requests=[]
    for pid,p,r,yaw,slot in [('fixture_a#1',plant,render,0,'T1R1L'),('fixture_a#2',plant,render,65,'T1R1R'),('fixture_b#1',other,render_b,140,'T1R2L')]:
        cid=pid.replace('#','_')+'_t0';cap={k:'unavailable' for k in s.CAPTURES_COLUMNS}
        cap.update(capture_id=cid,plant_instance_id=pid,source_plant_id=p.source_plant_id,evidence='synthetic_fixture',view='top',capture_time=at,capture_time__status='synthetic_truth',
            time_since_lights_on_min='60',time_since_lights_on_min__status='synthetic_truth',photoperiod_phase='am',frame='plant_local_mm',scale_source='synthetic_ortho_camera_declared',scale_status='solved',mm_per_px='.6',rotation_status='solved',session_id='consumer_boundary_fixture',note='Actual P1 truth exporter; not observed reconstruction')
        captures.append(cap)
        ranks={p.compound_vertex_id(c):i+1 for i,c in enumerate(sorted(p.compounds,key=lambda c:float(c.attachment_mm[2])))}
        vv=ft.truth_mtg(p,pid,cid,ranks);vertices.extend(vv);areas.extend(ft.truth_area_rows(p,pid,cid,r,camera))
        surfaces.extend(optical.export_truth(opt_dir,Path('optical_surfaces'),p,pid,cid))
        for v in vv:
            row={k:'unavailable' for k in s.IDENTITY_COLUMNS};row.update(plant_instance_id=pid,source_plant_id=p.source_plant_id,capture_id=cid,organ_id=v['vertex_id'],organ_class=v['class'],parent_id=v['parent_id'],track_state='first_observed',correspondence_status='confirmed',gap_captures='0',evidence='synthetic_fixture')
            identities.append(row)
        requests.append(canopy.InstanceRequest(pid,slot,yaw,'micro_tom','fixture_declared','fixture_declared'))
    slots=canopy.load_slots(ROOT/canopy.SLOT_TABLE);cfg=ft.load_p1_config()
    # Exact available slot names are read from the producer's human-record adapter.
    eligible=[k for k,v in slots.items() if v['variety']=='dwarf']
    requests=[replace(r,slot_id=eligible[i]) for i,r in enumerate(requests)]
    placement=canopy.placement_document(canopy.place(requests,slots,cfg),requests,cfg,sha(ROOT/canopy.SLOT_TABLE),'synthetic_fixture')
    for name,table,cols in [('captures.csv',captures,s.CAPTURES_COLUMNS),('identity_manifest.csv',identities,s.IDENTITY_COLUMNS),('mtg_vertices.csv',vertices,s.MTG_VERTICES_COLUMNS),('areas.csv',areas,s.AREAS_COLUMNS),('optical_surface_manifest.csv',surfaces,s.OPTICAL_MANIFEST_COLUMNS)]:write_table(out/name,table,cols)
    write_json(out/'canopy_placement.json',placement)
    write_json(out/'p1_manifest.json',dict(schema=s.P1_SCHEMA_VERSION,evidence='synthetic_fixture',run_status='fixture_only',
        producer='Claude P1 functions invoked by consumer boundary acceptance; not full P1 CLI or real reconstruction',
        producer_apis=['fixture_truth.load_plant','fixture_truth.truth_mtg','fixture_truth.truth_area_rows','optical.export_truth','canopy.placement_document'],
        source=ft.fixture_manifest_entry(plant,camera),code_hashes={Path(m.__file__).name:sha(Path(m.__file__)) for m in (ft,s,optical,canopy)}))
    audit=audit_p1_package(out);write_json(out/'consumer_fixture_audit.json',audit);return out


def scenario_fixture(package: Path,out: Path,intervals: int=3,rays: int=100000,seed: int=5000,state_source: str='main',
                     *,plant_reserve_gC: float | None=None,reserve_source_kind: str | None=None,
                     reserve_source_ref: str | None=None) -> Path:
    """Explicit synthetic physiology assumptions, separate from unchanged P1 package."""
    if plant_reserve_gC is None:
        require(reserve_source_kind is None and reserve_source_ref is None,'reserve provenance without reserve override')
    else:
        require(not isinstance(plant_reserve_gC,bool) and isinstance(plant_reserve_gC,(int,float))
                and math.isfinite(plant_reserve_gC) and plant_reserve_gC>=0,'plant_reserve_gC must be finite and nonnegative')
        require(reserve_source_kind in ('fixture_declared','user_assumed'),'reserve_source_kind must identify a fixture assumption')
        require(isinstance(reserve_source_ref,str) and bool(reserve_source_ref.strip()),'reserve_source_ref is required')
    out=fresh_build(out);caps=rows(package/'captures.csv');vv=rows(package/'mtg_vertices.csv');par=parameters();hidden={};plants={}
    if state_source=='fixture_truth':
        instances=[p['plant_instance_id'] for p in read_json(package/'canopy_placement.json')['instances'] if p['in_optical_scene']]
        caps=[max((c for c in caps if c['plant_instance_id']==pid and c['capture_time']!='unavailable'),key=lambda c:aware(c['capture_time'])) for pid in instances]
        truth=rows(package/'truth/truth_mtg_vertices.csv');vv=[]
        for c in caps:vv.extend(dict(v,plant_instance_id=c['plant_instance_id']) for v in truth if v['plant_instance_id']==c['source_plant_id'] and v['capture_id']==c['capture_id'])
    from tools.microtom_axes import rank_basis as package_rank_basis, RANK_BASIS_PRE_A7
    basis=package_rank_basis(package)  # A7 follow-through: a producer rank is a node index only in pre-revision-9 packages
    def node_index(v: dict) -> int:
        """Engine true_leaf_order of a CompoundLeaf = its node index on its axis. Revision 9: node_index_from_base (refused when unavailable); pre-A7: the rank."""
        if basis==RANK_BASIS_PRE_A7:return int(v['rank'])
        n=v.get('node_index_from_base','unavailable');require(n not in ('','unavailable',None),v['vertex_id']+': rank without node_index_from_base in a revision-9 package; never guessed');return int(n)
    for c in caps:
        pid=c['plant_instance_id'];ps=[v for v in vv if v['plant_instance_id']==pid];by={v['vertex_id']:v for v in ps}
        compounds=[v for v in ps if v['class']=='CompoundLeaf'];leaf_order=max(node_index(v) for v in compounds)
        for v in ps:
            oid=pid+'|'+v['vertex_id'];kind=v['class'];ancestor=v
            while ancestor['class'] not in ('CompoundLeaf','Axis','Plant'):
                ancestor=by[ancestor['complex_id']]
            rank=node_index(ancestor) if ancestor['class']=='CompoundLeaf' else 0
            mass=float(v['area_mm2'])*1e-6/.025 if kind=='Leaflet' else .006 if kind in ('Petiole','RachisSegment') else 0.
            hidden[oid]=dict(mass_gDM=mass,max_mass_gDM=mass*1.5,thermal_age_dd=300.,true_leaf_order=rank,width_m=.001 if kind in ('Petiole','RachisSegment') else 0,
                units=dict(mass='gDM',age='degree_day'),provenance='explicit_consumer_fixture_assumption; leaflet initial mass from assumed SLA, not measured')
        plants[pid]=dict(reserve_gC=.2,root_mass_gDM=.04,stem_mass_gDM=.06,stem_width_m=.003,thermal_dd=206.3+37.9*(leaf_order+.5),produced=leaf_order,leaf_order=leaf_order,truss_count=0,axis_leaf_count=leaf_order)
        if plant_reserve_gC is not None:
            plants[pid].update(reserve_gC=float(plant_reserve_gC),reserve_source_kind=reserve_source_kind,
                               reserve_source_ref=reserve_source_ref.strip(),reserve_unit='gC')
    start=caps[0]['capture_time'];env=[dict(light_hours=14,temp_day_c=26,temp_night_c=22,co2_ppm=400,source_kind='user_assumed_fixture') for _ in range(intervals)]
    nutrient={}
    from tools.fsp_nutrients_demo import synthetic_scenario
    for c in caps:
        pid=c['plant_instance_id'];q=synthetic_scenario();q.update(plant_instance_id=pid,source_plant_id=c['source_plant_id'],t_start=start,t_end=(aware(start)+timedelta(days=intervals)).isoformat())
        for i,e in enumerate(q['events']):e['at']=(aware(start)+timedelta(hours=i+1)).isoformat()
        # Name the explicit allocation target after an existing native tissue.
        target=next(k for k,v in hidden.items() if k.startswith(pid+'|') and v['mass_gDM']>0)
        q['plant_initial']['organs']={target:q['plant_initial']['organs']['leaf']}
        for e in q['events']:
            if e['kind']=='allocate':e['organ_id']=target
        nutrient[pid]=q
    scenario=dict(schema_version=SCHEMA,fixture=True,state_source=state_source,captures_by_instance={c['plant_instance_id']:c['capture_id'] for c in caps},start_time=start,
        intervals=intervals,rays=rays,seed=seed,parameters=par,organ_initialization=hidden,plant_initialization=plants,
        initialization_provenance='explicit synthetic boundary assumptions; no measured mass, age, reserve or uptake; rank_basis='+basis,environment=env,
        lamps=[dict(lamp_id='L1',x_m=0,y_m=.1,z_m=.7,power_umol_s=1200,source_kind='explicit_hypothetical_point_LED')],
        allow_conditional_surfaces=False,nutrient_feedback_enabled=False,nutrient_ledger_required=True,nutrient_scenarios=nutrient)
    write_json(out/'scenario.json',scenario);return out/'scenario.json'


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--scenario-p1',type=Path);p.add_argument('--intervals',type=int,default=3);p.add_argument('--rays',type=int,default=100000);p.add_argument('--state-source',choices=['main','fixture_truth'],default='main')
    p.add_argument('--plant-reserve-gC',type=float,help='Explicit per-plant fixture reserve (gC); default remains 0.2')
    p.add_argument('--reserve-source-kind',choices=['fixture_declared','user_assumed'])
    p.add_argument('--reserve-source-ref',help='Required provenance for a reserve override');a=p.parse_args()
    if not a.scenario_p1 and any(v is not None for v in (a.plant_reserve_gC,a.reserve_source_kind,a.reserve_source_ref)):
        p.error('reserve arguments require --scenario-p1')
    print(scenario_fixture(a.scenario_p1,a.out,a.intervals,a.rays,state_source=a.state_source,
          plant_reserve_gC=a.plant_reserve_gC,reserve_source_kind=a.reserve_source_kind,reserve_source_ref=a.reserve_source_ref)
          if a.scenario_p1 else producer_fixture(a.out))


def controlled_light(out: Path) -> None:
    """Explicitly hypothetical optical stress fixtures, never producer observations."""
    import shutil
    import numpy as np
    from tools.tomato_fsp_pipeline import prepare_run,run
    out=fresh_build(out)
    source=ROOT/'build/fsp_integration_20261011_p1_fixture_01'
    base=read_json(ROOT/'build/fsp_integration_20261011_light_02/isolated/scenario.json')
    full=read_json(ROOT/'build/fsp_integration_20261011_light_01/full/scenario.json')
    for name in ('stacked','tilted'):
        package=out/(name+'_p1');shutil.copytree(source,package)
        doc=read_json(package/'canopy_placement.json');by={r['plant_instance_id']:r for r in doc['instances']}
        if name=='stacked':
            receiver=by['fixture_a#2']['transform'];tr=by['fixture_a#1']['transform']
            matrix=np.array(receiver['matrix_4x4']);matrix[:3,3]+=np.array([-30.,18.,120.])
            tr.update(matrix_4x4=matrix.tolist(),translation_mm=matrix[:3,3].tolist(),yaw_deg=receiver['yaw_deg'],yaw_source='conditional_stacked_occluder',translation_source='conditional optical stress test; raised source along lamp direction')
        else:
            tr=by['fixture_a#2']['transform'];matrix=np.array(tr['matrix_4x4']);rx=np.array([[1,0,0],[0,0,-1],[0,1,0]])
            matrix[:3,:3]=matrix[:3,:3]@rx
            tr.update(matrix_4x4=matrix.tolist(),yaw_source='conditional rigid 90-degree local-X tilt after yaw; matrix authoritative')
        doc['evidence']='synthetic_fixture';doc['fixture_note']='Consumer-derived placement stress test; not measured shelf geometry'
        write_json(package/'canopy_placement.json',doc)
        for seed in (5000,5100):
            scenario=dict(full if name=='stacked' else base,seed=seed)
            scenario_path=out/f'{name}_{seed}_scenario.json';write_json(scenario_path,scenario)
            folder=out/f'{name}_{seed}';prepare_run(package,scenario_path,folder)
            print(name,seed,'prepared',flush=True);print(run(folder,300)['status'],flush=True)


def render_native_delivery(out: Path) -> Path:
    """Plot the returned native meshes, not reconstructed or illustrative shapes."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from collections import defaultdict
    out=fresh_build(out);source=ROOT/'build/fsp_integration_20261011_producer_run_02/native/results'
    states=rows(source/'states.csv');owners={s['organ_id']:s['plant_instance_id'] for s in states}
    last=max(int(s['day']) for s in states);mesh=defaultdict(list)
    for row in rows(source/'meshes.csv'):
        day=int(row['state'])
        if day in (0,last):mesh[day,owners[row['organ_id']]].append([float(row[k])*1000 for k in ('x_m','y_m','z_m')])
    fig=plt.figure(figsize=(13,6.5),facecolor='#f7f8fa');colors=['#26867d','#48a88e','#6d8ecd','#cb9146'];ids=sorted(set(owners.values()))
    coords=np.array([v for vv in mesh.values() for v in vv]);lo=coords.min(axis=0);hi=coords.max(axis=0);span=hi-lo
    for column,day in enumerate((0,last),1):
        ax=fig.add_subplot(1,2,column,projection='3d');ax.set_facecolor('#f7f8fa')
        for pid,color in zip(ids,colors):
            triangles=np.array(mesh[day,pid]).reshape(-1,3,3)
            ax.add_collection3d(Poly3DCollection(triangles,facecolors=color,edgecolors='none',alpha=1))
            ax.scatter([],[],[],c=color,label=pid,s=20)
        ss=[s for s in states if int(s['day'])==day]
        ax.set_title(f'Day {day} | {len(ss)} organs | {sum(float(s["mass_gDM"]) for s in ss):.4f} gDM',fontsize=12)
        ax.set(xlim=(lo[0]-10,hi[0]+10),ylim=(lo[1]-10,hi[1]+10),zlim=(min(0,lo[2]),hi[2]+10),xlabel='Canopy X (mm)',ylabel='Canopy Y (mm)',zlabel='Z (mm)')
        ax.set_box_aspect(np.maximum(span,30));ax.view_init(elev=27,azim=-65)
        if column==1:ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Delivered P1 fixture truth → native light / carbon / growth → updated canopy',fontsize=14,y=.97)
    fig.text(.5,.035,'Conditional fixture • actual GroIMP mesh vertices • one placement transform • unvalidated biology',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.01,right=.99,top=.9,bottom=.10,wspace=.04);path=out/'native_canopy.png';fig.savefig(path,dpi=180);plt.close(fig)
    write_json(out/'figure_provenance.json',dict(source=str(source.relative_to(ROOT)),mesh_sha256=sha(source/'meshes.csv'),state_sha256=sha(source/'states.csv'),plot_sha256=sha(path),render='native triangles without geometric simplification'))
    return path


def summarize_delivery(out: Path) -> dict:
    """Verify recorded artifacts and summarize actual runs without rerunning physics."""
    import shutil
    import xml.etree.ElementTree as ET
    from collections import Counter
    from tools.vtc_growth_io import require
    out=fresh_build(out);b=ROOT/'build';prefix='fsp_integration_20261011_'
    paths={
        'producer':prefix+'producer_run_02','dynamic':prefix+'final_02',
        'frozen':prefix+'controls_01/frozen','seed':prefix+'controls_01/seed',
        'no_nutrient':prefix+'controls_01/no_nutrient','halfday':prefix+'refinement_01/run',
        'light':prefix+'light_01/full','light_seed':prefix+'light_02/full_seed',
        'lowray':prefix+'light_02/full_lowray','dark':prefix+'light_02/dark',
        'isolated':prefix+'light_02/isolated','yaw':prefix+'orientation_01/run',
        **{f'{name}_{seed}':prefix+f'optical_stress_01/{name}_{seed}' for name in ('stacked','tilted') for seed in (5000,5100)},
    }
    runs={};state_tables={};plant_light={}
    for name,relative in paths.items():
        folder=b/relative;prepared=read_json(folder/'prepared.json');ex=read_json(folder/'native/execution.json')
        require(ex['status']=='verified' and ex['exit_code']==0,'native execution failed: '+name)
        require(ex['prepared_sha256']==sha(folder/'native/prepared.json') and ex['log_sha256']==sha(folder/'native/groimp.log'),'execution binding changed')
        require(prepared['scenario_sha256']==sha(folder/'scenario.json') and prepared['contract_sha256']==sha(folder/'contract.json'),'scenario/contract changed')
        for file,h in prepared['p1_hashes'].items():require(sha(folder/'p1'/file)==h,'P1 frozen input changed')
        np=read_json(folder/'native/prepared.json')
        require(np['model_hash']==sha(folder/'native/working/Model.rgg'),'native model changed')
        for file,h in np['frozen_hashes'].items():
            if (folder/'native/inputs'/file).is_file():require(sha(folder/'native/inputs'/file)==h,'native input changed')
        checkpoints=sorted((folder/'committed').glob('step_*/checkpoint.json'));initial=rows(folder/'native/inputs/initial_state.csv')
        state_tables[name]=[];plant_light[name]=[];carbon=[];light=[];cache=[];previous=None
        for i,path in enumerate(checkpoints,1):
            cp=read_json(path);require(cp['lineage']['boundary']==i and cp['native_advance_count']==1,'invalid advance count')
            if previous:require(cp['lineage']['previous_checkpoint_sha256']==sha(previous),'broken checkpoint chain')
            for file,h in cp['files'].items():require(sha(path.parent/file)==h,'published checkpoint changed')
            state_tables[name].append(rows(path.parent/'state.csv'));carbon.extend(rows(path.parent/'carbon_ledger.csv'))
            plant_light[name].append({r['plant_instance_id']:float(r['absorbed_umol_s']) for r in rows(path.parent/'plant_light.csv')})
            light.append(sum(plant_light[name][-1].values()));cache.append(cp['cache_key']);previous=path
        request=read_json(folder/'native/inputs/request.json');require(len(checkpoints)==request['intervals'],'missing boundary')
        final=state_tables[name][-1];residual=max(abs(float(r['residual_gC'])) for r in carbon)
        require(residual<1e-9,'carbon nonclosure')
        runs[name]=dict(path=str(folder.relative_to(ROOT)),native_command=ex['command'],exit_code=ex['exit_code'],native_elapsed_s=ex['elapsed_s'],
            model_sha256=np['model_hash'],input_hash=np['binding']['input_hash'],engine_hash=np['binding']['engine_hash'],
            prepared_sha256=sha(folder/'prepared.json'),execution_sha256=sha(folder/'native/execution.json'),producer_schema_sha256=read_json(folder/'stage_0_contract_report.json').get('producer_schema_sha256','see frozen P1 audit'),
            intervals=len(checkpoints),dt_s=request['dt_s'],instances=sorted({s['plant_instance_id'] for s in final}),initial_organs=len(initial),final_organs=len(final),
            initial_leaflets=sum(s['class']=='Leaflet' for s in initial),final_mass_gDM=sum(float(s['mass_gDM']) for s in final),
            max_carbon_residual_gC=residual,total_gross_gC=sum(float(r['gross_gC']) for r in carbon),absorbed_leaf_umol_s=light,
            leaf_light_by_instance=plant_light[name],checkpoint_hashes=[sha(p) for p in checkpoints],cache_keys=cache,
            light_only=request.get('light_only',False),rays=request['runtime']['rays'],seed=request['runtime']['seed'])
        if request.get('light_only'):
            a={s['organ_id']:s for s in initial};z={s['organ_id']:s for s in final};require(set(a)==set(z),'light-only birth')
            for oid in a:
                for key in ('mass_gDM','thermal_age_dd','area_m2','length_m'):require(abs(float(a[oid][key])-float(z[oid][key]))<1e-12,'light-only state changed')
    # Nutrient bookkeeping is deliberately uncoupled from native physiology.
    for aa,bb in zip(state_tables['dynamic'],state_tables['no_nutrient']):
        a={s['organ_id']:s for s in aa};z={s['organ_id']:s for s in bb};require(set(a)==set(z),'disabled-feedback identities')
        for oid in a:
            for k in ('mass_gDM','thermal_age_dd','length_m','area_m2'):require(abs(float(a[oid][k])-float(z[oid][k]))<1e-12,'disabled-feedback state differs')
    dynamic=runs['dynamic']['absorbed_leaf_umol_s'];frozen=runs['frozen']['absorbed_leaf_umol_s'];seed=runs['seed']['absorbed_leaf_umol_s']
    noise=max(abs(a-bb) for a,bb in zip(dynamic,seed));effect=abs(dynamic[-1]-frozen[-1]);require(effect>noise,'relight difference not resolved')
    require(all(v==0 for v in runs['dark']['absorbed_leaf_umol_s']),'nonzero no-source light')
    pid='fixture_a#2';isolated=plant_light['isolated'][0][pid]
    stress_noise=max(abs(plant_light[f'{name}_5000'][0][pid]-plant_light[f'{name}_5100'][0][pid]) for name in ('stacked','tilted'))
    stress_noise=max(stress_noise,abs(plant_light['light'][0][pid]-plant_light['light_seed'][0][pid]))
    shading=isolated-plant_light['stacked_5000'][0][pid];tilt=abs(isolated-plant_light['tilted_5000'][0][pid])
    require(shading>stress_noise and tilt>stress_noise,'controlled optical effect not resolved')
    refinement=abs(runs['halfday']['final_mass_gDM']/runs['dynamic']['final_mass_gDM']-1)
    real=read_json(b/(prefix+'real_audit_01/audit.json'));failure=read_json(b/(prefix+'failure_01/verified_failure.json'))
    require(failure['last_valid_boundary']==1 and not (b/(prefix+'failure_01/run/committed/step_0002')).exists(),'failed ledger published boundary')
    replay=read_json(b/(prefix+'final_02/replay_verification.json'));require(replay['status']=='passed','replay failed')
    comparison=read_json(b/(prefix+'comparison_01/comparison.json'));require(comparison['forecast_unchanged'] and not comparison['assimilation_applied'],'comparison mutated forecast')
    junit=b/(prefix+'regression_final.xml');root=ET.parse(junit).getroot();suites=list(root) if root.tag=='testsuites' else [root]
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')};require(tests['failures']==tests['errors']==tests['skipped']==0,'regression incomplete')
    snapshot=out/'source';snapshot.mkdir()
    files=[*ROOT.glob('tools/tomato_fsp_*.py'),*ROOT.glob('tools/vtc_development*'),ROOT/'tools/vtc_growth_bridge.py',ROOT/'tools/TOMATO_FSP_PIPELINE.md',ROOT/'tools/tests/test_tomato_fsp_pipeline.py',ROOT/'handoff/FOUR_PROJECT_INTEGRATION_CONTRACT_2026-10-11.md']
    hashes={}
    for path in files:
        if not path.is_file():continue
        dst=snapshot/path.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst);hashes[str(path.relative_to(ROOT))]=sha(dst)
    shutil.copyfile(junit,out/'regression.xml')
    result=dict(status='conditional_connected_workflow_verified',biological_validation='not_established',nutrient_growth_feedback=False,
        stage_results={str(i):status for i,status in enumerate(['actual_P1_truth_route_consumed_and_real_audited','rigid_canopy_verified','native_light_controls_verified_with_declared_optical_limits','carbon_proposals_and_closure_verified','funded_native_growth_verified','three_interval_relight_and_replay_verified','prescribed_nutrient_ledger_and_readonly_comparison_verified','delivered_conditional'])},
        runs=runs,controls=dict(relight_effect_umol_s=effect,max_seed_difference_umol_s=noise,
            controlled_shading_effect_umol_s=shading,controlled_tilt_effect_umol_s=tilt,stress_max_seed_difference_umol_s=stress_noise,
            yaw_effect_umol_s=abs(isolated-plant_light['yaw'][0][pid]),yaw_resolved=False,
            lowray_relative_total_difference=abs(runs['lowray']['absorbed_leaf_umol_s'][0]/runs['light']['absorbed_leaf_umol_s'][0]-1),
            highray_seed_relative_total_difference=abs(runs['light_seed']['absorbed_leaf_umol_s'][0]/runs['light']['absorbed_leaf_umol_s'][0]-1),
            halfday_relative_final_mass_difference=refinement,refinement_interpretation='same 3-day horizon and averaged climate; not a real diurnal response',disabled_nutrient_equivalence=True),
        real_readiness=dict(ready=real['real_ready'],captures=real['table_rows']['captures.csv'],missing_counts=dict(Counter(x.split(':')[0] for x in real['missing'])),audit_path=prefix+'real_audit_01/audit.json'),
        expected_failure=failure,replay=replay,comparison=comparison,final_regression=tests,earlier_broad_regression_passed=288,source_hashes=hashes,
        v3_long_horizon='build/vtc_dev_v3_delivery_sufficient',v3_acceptance='build/vtc_dev_v3_acceptance_01/acceptance.json',
        limits=['Fixture truth is separate from P1 observation reconstruction','Black hypothetical optics: no tissue transmission; no exported total nonleaf/escape energy closure','Incident PPFD unavailable; leaf absorption is exported','Two seeds quantify tested-scene variation, not a confidence interval','Imported posture is held; new organs use conditional V3 laws','No phototropism, nutrient growth, fruit/yield or real-plant prediction validation','Frozen native model hashes identify each run; final source snapshot includes later compatibility fixes'])
    write_json(out/'acceptance.json',result);return result


if __name__=='__main__':main()
