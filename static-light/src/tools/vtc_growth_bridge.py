"""Freeze, launch and validate a local GroIMP/VTC conditional growth run."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from tools.groimp_light_sanity import CORE
from tools.groimp_static_verify import cli_path
from tools.vtc_growth_fork import DEFAULT_SOURCE, create_fork, make_model
from tools.vtc_growth_io import (ROOT, SCHEMA, HEADERS, ContractError, digest, fresh_build,
    load_package, read_json, require, rows, sha, validate_outputs, write_csv, write_json)


def prepare(package_dir: Path, fork: Path, out: Path) -> dict:
    package=load_package(package_dir)
    require(package['request'].get('mode','phase1_leaflet_subset_v1')=='phase1_leaflet_subset_v1','wrong engine mode; no subset fallback')
    manifest=read_json(fork/'source_manifest.json')
    for name,value in manifest['modified_hashes'].items():
        require(sha(fork/name)==value,'fork template modified since manifest')
    out=fresh_build(out)
    frozen=out/'inputs'; frozen.mkdir()
    for name in [*HEADERS,'request.json']:
        shutil.copyfile(package_dir/name,frozen/name)
        require(sha(frozen/name)==package['source_hashes'][name], 'source changed during freeze')
    shutil.copyfile(package['ratio_target'],frozen/'ratios.csv')
    require(sha(frozen/'ratios.csv')==package['source_hashes']['resolved_ratio_target'],'ratio changed during freeze')
    (frozen/'light_ratio_latest.txt').write_text('ratios.csv\n',encoding='utf-8')
    # Preserve original pointer bytes separately; no later reads through the human pointer.
    shutil.copyfile(package_dir/'light_ratio_latest.txt',out/'original_ratio_pointer.txt')
    frozen_package=load_package(frozen)
    binding=dict(input_hash=digest(frozen_package['source_hashes']),engine_hash=sha(fork/'phase1_template.rgg'))
    working=out/'working'; working.mkdir()
    model=make_model(frozen_package,(fork/'phase1_template.rgg').read_text(encoding='utf-8'),binding)
    (working/'Model.rgg').write_text(model,encoding='utf-8')
    shutil.copyfile(fork/'source_manifest.json',out/'fork_manifest.json')
    result=dict(schema_version=SCHEMA,status='prepared',run_id=package['request']['run_id'],
                binding=binding,source_hashes=package['source_hashes'],resolved_ratio_target=package['ratio_target'],
                frozen_hashes=frozen_package['source_hashes'],model_hash=sha(working/'Model.rgg'),
                bridge_hashes={p.name:sha(p) for p in (Path(__file__),ROOT/'tools/vtc_growth_io.py',ROOT/'tools/vtc_growth_fork.py')})
    write_json(out/'prepared.json',result)
    return result


def verify_prepared(folder: Path) -> tuple[dict,dict]:
    prepared=read_json(folder/'prepared.json')
    if prepared.get('mode') == 'vtc_development_v3':
        from tools.vtc_development import verify_prepared as verify_development
        return verify_development(folder)
    require(prepared.get('mode', 'phase1_leaflet_subset_v1') == 'phase1_leaflet_subset_v1', 'unknown engine mode')
    package=load_package(folder/'inputs')
    require(package['request'].get('mode','phase1_leaflet_subset_v1')=='phase1_leaflet_subset_v1','wrong engine mode; no subset fallback')
    require(package['source_hashes']==prepared['frozen_hashes'],'frozen input changed')
    require(sha(folder/'working/Model.rgg')==prepared['model_hash'],'native model changed')
    require(digest(package['source_hashes'])==prepared['binding']['input_hash'],'input binding changed')
    return prepared,package


def run(folder: Path, java: str='java', core: Path=CORE, timeout_s: float=180, on_interval=None, fail_closed_interval_hook: bool=False) -> dict:
    """on_interval(k) is called once per completed native interval k (after the engine has written
    working/checkpoints/step_k and printed VTCDEV_INTERVAL k). It is a progress hook for incremental
    publication (F11 C1); it never changes what the engine does. A hook exception is recorded in
    execution.json and disables the hook for the rest of the run; the run itself continues."""
    prepared,package=verify_prepared(folder)
    development=prepared.get('mode')=='vtc_development_v3'
    if development:
        from tools.vtc_development_io import validate_outputs as check_outputs, OUTPUTS
        output_names=(*OUTPUTS,'completion.json')
        if package['request'].get('integration_contract'):output_names=(*output_names,'proposals.csv')
        if package['request'].get('extensions'):output_names=(*output_names,'optical_flux.csv','scene_photon_budget.csv')
        if package['request'].get('axis_contract'):output_names=(*output_names,'axis_flux.csv')
        from tools import fsp_incident_diagnostics as incident
        if incident.active(package['request']):output_names=(*output_names,*incident.OUTPUTS)
    else:
        check_outputs=validate_outputs
        output_names=('completion.json','geometry.csv','organs.csv','ledger.csv','sensors.csv','consumed_geometry.csv')
    marker='VTCDEV' if development else 'PHASE1'
    require(not (folder/'execution.json').exists(),'execution already attempted; use fresh deterministic replay')
    working=folder/'working'
    require(not any((working/n).exists() for n in output_names),'stale native outputs')
    require(core.is_file(),'GroIMP core.jar missing')
    java_path=shutil.which(java); require(java_path is not None,'Java runtime missing')
    require(timeout_s>0,'positive timeout required')
    command=[java_path,'-Dfile.encoding=UTF-8','-jar',str(core.resolve()),'--headless','-a','cli']
    version=subprocess.run([java_path,'-version'],capture_output=True,text=True,encoding='utf-8',timeout=15).stderr
    log=folder/'groimp.log'; started=time.monotonic(); execution=dict(status='running',command=command,cwd=str(working.resolve()),
        java_version=version,core_sha256=sha(core),prepared_sha256=sha(folder/'prepared.json'),exit_code=None)
    write_json(folder/'execution.json',execution)
    process=None
    try:
        with log.open('wb') as stream:
            process=subprocess.Popen(command,cwd=working,stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            seen_intervals=0;hook_active=on_interval is not None
            def notify_intervals(content: str) -> None:
                nonlocal seen_intervals,hook_active
                if not hook_active: return
                for k in [int(m) for m in re.findall(r'VTCDEV_INTERVAL (\d+)',content)]:
                    if k<=seen_intervals: continue
                    try: on_interval(k)
                    except Exception as exc:  # the native run continues; collection reports the failure
                        if fail_closed_interval_hook:
                            raise
                        execution['interval_hook_error']=f'{type(exc).__name__}: {exc}';hook_active=False;return
                    seen_intervals=k
            def wait_for(token: str, *, offset: int=0, allow_compile_errors: bool=False) -> str:
                while time.monotonic()-started<timeout_s:
                    content=log.read_text(encoding='utf-8',errors='replace')
                    notify_intervals(content)
                    if token in content[offset:]: return content
                    if process.poll() is not None:
                        # The hook may have taken seconds; re-read once so a token written meanwhile is not missed.
                        content=log.read_text(encoding='utf-8',errors='replace');notify_intervals(content)
                        if token in content[offset:]: return content
                        raise ContractError(f'engine exited before {token}; inspect groimp.log')
                    boundary_failure=re.search(r'IllegalStateException: (native_boundary_[a-z_]+)',content)
                    if boundary_failure:raise ContractError(boundary_failure.group(1))
                    if not allow_compile_errors and any(message in content[offset:] for message in ('Syntax error','Compilation failed','Command not found','MethodTooLargeException','could not be opened.')):
                        raise ContractError('native compile/command failure; inspect groimp.log')
                    time.sleep(.1)
                raise TimeoutError('GroIMP timeout')
            wait_for('[base]>>')
            # Existing repository short-path helper handles GroIMP CLI whitespace restrictions.
            process.stdin.write(f'$cd {cli_path(working)}\n$open Model.rgg\n'); process.stdin.flush()
            ready_offset=0
            if development and 'incident_diagnostics' in prepared:
                # The observer jar belongs only to this project. Initial RGG
                # resolution may fail until addFile installs its frozen classes.
                wait_for('[Model',allow_compile_errors=True)
                metadata=prepared['incident_diagnostics'];helper_name=metadata['helper_jar']
                helper=working/helper_name
                require(helper.is_file() and helper.resolve().is_relative_to(working.resolve()), 'incident helper missing or outside working folder')
                frozen_helpers=prepared.get('auxiliary_hashes',{});declared_helpers=metadata.get('auxiliary_hashes',{})
                require(helper_name in frozen_helpers and helper_name in declared_helpers and
                        frozen_helpers[helper_name]==declared_helpers[helper_name] and sha(helper)==frozen_helpers[helper_name],
                        'selected incident helper is not bound to frozen auxiliary hashes')
                ready_offset=len(log.read_text(encoding='utf-8',errors='replace'))
                process.stdin.write(f'addFile {cli_path(helper)}\ncompile\n'); process.stdin.flush()
            # GroIMP may display [Model~R] instead of [Model]; the unique init
            # acknowledgement is the authority, not the renderer prompt spelling.
            wait_for(marker+'_READY '+package['request']['run_id'],offset=ready_offset)
            process.stdin.write('simulate\n'); process.stdin.flush()
            wait_for(marker+'_DONE '+package['request']['run_id'],offset=ready_offset)
            process.wait(timeout=max(.1,timeout_s-(time.monotonic()-started)))
            execution['exit_code']=process.returncode
            require(process.returncode==0,'nonzero native exit')
        content=log.read_text(encoding='utf-8',errors='replace')
        execution['groimp_version']=(re.findall(r'V (\d+\.\d+\.\d+)',content) or ['unavailable'])[0]
        require(execution['groimp_version']!='unavailable','runtime version not verified')
        check_outputs(package,working,prepared['binding'])
        execution['status']='verified'
    except Exception as exc:
        execution['status']='failed'; execution['error']=f'{type(exc).__name__}: {exc}'
        if process is not None and process.poll() is None: process.kill(); process.wait()
        if process is not None: execution['exit_code']=process.returncode
        partial=folder/'partial'; partial.mkdir(exist_ok=True)
        for name in output_names:
            if (working/name).exists(): shutil.move(str(working/name),str(partial/name))
        raise
    finally:
        if process is not None and process.stdin is not None: process.stdin.close()
        execution['elapsed_s']=time.monotonic()-started
        if log.exists(): execution['log_sha256']=sha(log)
        write_json(folder/'execution.json',execution)
    return execution


def collect(folder: Path) -> dict:
    prepared,package=verify_prepared(folder); execution=read_json(folder/'execution.json')
    require(execution['status']=='verified' and execution['exit_code']==0,'verified successful execution required')
    require(execution['prepared_sha256']==sha(folder/'prepared.json'),'execution/prepared binding mismatch')
    require(execution['log_sha256']==sha(folder/'groimp.log'),'engine log changed')
    require(not (folder/'results').exists(),'results already published')
    verified=validate_outputs(package,folder/'working',prepared['binding'])
    stage=folder/'publication_partial'; require(not stage.exists(),'partial publication exists; inspect before retry')
    stage.mkdir()
    for name in ('geometry.csv','organs.csv','ledger.csv','sensors.csv','consumed_geometry.csv','completion.json'):
        shutil.copyfile(folder/'working'/name,stage/name)
    write_csv(stage/'normalized_ledger.csv',verified.pop('normalized_ledger'))
    ids=[dict(plant_instance_id=l['plant_instance_id'],source_plant_id=next(p['source_plant_id'] for p in package['tables']['plants.csv'] if p['plant_instance_id']==l['plant_instance_id']),axis_id=l['axis_id'],compound_id=l['compound_id'],leaflet_id=l['leaflet_id'],parent_id=l['parent_id'],vtc_node_id=i,geometry_source=l['geometry_source'],mass_source=l['mass_source'],age_source=l['age_source']) for i,l in enumerate(package['tables']['leaflets.csv'])]
    write_csv(stage/'identity_mapping.csv',ids)
    env=package['tables']['environment.csv']
    write_csv(stage/'interval_binding.csv',[dict(interval=i,geometry_hash=verified['geometry_hashes'][str(i)],next_geometry_hash=verified['geometry_hashes'][str(i+1)],environment_hash=digest(e),start_time=e['start_time'],end_time=e['end_time']) for i,e in enumerate(env)])
    sensors=rows(stage/'sensors.csv'); total=sum(float(s['absorbed_umol_s'])/float(s['area_m2']) for s in sensors)
    ratio_map={(r.get('grid_row'),r.get('grid_col')):r.get('ratio_to_mean') for r in package['ratios']}
    comparison=[]
    for s in sensors:
        flux=float(s['absorbed_umol_s'])/float(s['area_m2']); ratio=flux/(total/16) if total>0 else None
        measured=ratio_map.get((s['grid_row'],s['grid_col']))
        comparison.append(dict(grid_row=s['grid_row'],grid_col=s['grid_col'],simulated_ppfd_umol_m2_s=flux,
          simulated_ratio=ratio,phone_ratio=measured,physical_validation=False,source_kind='assumption_conditioned'))
    write_csv(stage/'light_validation.csv',comparison)
    verified['ratio_rmse']= (sum((float(c['phone_ratio'])-c['simulated_ratio'])**2 for c in comparison)/16)**.5 if all(c['phone_ratio'] is not None and c['simulated_ratio'] is not None for c in comparison) else None
    geometry=rows(stage/'geometry.csv')
    # Self-contained canvas viewer; every frame is an exported native state.
    payload=json.dumps(geometry,ensure_ascii=True).replace('</','<\\/')
    html='''<!doctype html><meta charset="utf-8"><title>Conditional native VTC trajectory</title>
<style>body{font:16px system-ui;background:#10231d;color:#eaf6ef;margin:30px}canvas{background:#1d392e;border-radius:12px}label{display:block;margin:20px 0}</style>
<h1>GroIMP 반환 소엽 상태</h1><p>합성 조건부 시험 · 생물학적 검증 없음 · 제공된 소엽만 표시</p>
<label>일수 <input id="step" type="range" min="0" max="MAX" value="0"><span id="day"></span></label>
<canvas id="view" width="950" height="620"></canvas><pre id="info"></pre>
<script>const rows=DATA;const slider=document.getElementById('step');function draw(){let k=+slider.value,ctx=document.getElementById('view').getContext('2d');ctx.clearRect(0,0,950,620);document.getElementById('day').textContent=k;let rs=rows.filter(r=>+r.state===k);rs.sort((a,b)=>+a.z_m-+b.z_m);for(let r of rs){let x=180+1000*(+r.x_m)+400*(+r.y_m),y=500-1000*(+r.z_m)+300*(+r.y_m);ctx.fillStyle=r.plant_instance_id===rs[0].plant_instance_id?'#6ceca4':'#70b8ff';ctx.fillRect(x,y,1000*(+r.width_m),350*(+r.length_m));ctx.fillStyle='#fff';ctx.fillText(r.leaflet_id,x,y-5)}document.getElementById('info').textContent=rs.map(r=>r.leaflet_id+' | DM '+(+r.dry_mass_g).toFixed(5)+' g | area '+(+r.area_m2*1e4).toFixed(2)+' cm²').join('\\n')}slider.oninput=draw;draw();</script>'''.replace('MAX',str(package['request']['intervals'])).replace('DATA',payload)
    (stage/'viewer.html').write_text(html,encoding='utf-8')
    verified.update(run_id=package['request']['run_id'],scenario_id=package['request']['scenario_id'],source_hashes=package['source_hashes'],
        initialization_route=package['request']['initialization_route'],completeness='supported_leaf_subset',
        parameters=package['request']['parameters'],artifact_hashes={p.name:sha(p) for p in stage.iterdir() if p.is_file()})
    write_json(stage/'manifest.json',verified)
    stage.rename(folder/'results')
    return verified


def demo_inputs(out: Path, scenario: str='sufficient', rays: int=300000) -> Path:
    out=fresh_build(out); start=datetime.fromisoformat('2026-10-11T06:00:00+09:00'); n=3
    parameters={k:dict(value=v,source_kind='synthetic_assumed',source_ref='explicit arithmetic fixture; not measured',cultivar='none_synthetic',domain='software test',uncertainty='unvalidated') for k,v in dict(amax_umol_m2_s=13.57,alpha_mol_mol=.048,theta=.92,respiration_umol_m2_s=0.,carbon_fraction_gC_gDM=.45,growth_efficiency=.75,molar_carbon_g_mol=12.011).items()}
    for k in ('amax_umol_m2_s','alpha_mol_mol','theta'):
        parameters[k].update(source_kind='literature_other_cultivar',source_ref='VTC parametersMerlice.rgg upper NRH; no empirical-fit revalidation',cultivar='Merlice')
    if scenario=='dark_deficit': parameters['respiration_umol_m2_s']['value']=1.
    request=dict(schema_version=SCHEMA,run_id='demo_'+scenario,scenario_id=scenario,initialization_route='observation_fixture' if scenario=='observation' else 'synthetic_fixture',
        fixture=True,observation_time=start.isoformat(),start_time=start.isoformat(),end_time=(start+timedelta(days=n)).isoformat(),dt_s=86400,intervals=n,
        frame='plant_local_mm',geometry_basis='planar_rectangular_approximation',topology_scope='provided_axes',completeness='supported_leaf_subset',
        run_status='assumption_conditioned',evaluation_status='evaluated_unaccepted',biological_validation='not_established',nutrient_feedback_enabled=False,
        freeze_geometry=scenario=='frozen',parameters=parameters,runtime=dict(rays=rays,depth=1,seed=5000,balance_tolerance_gC=1e-9),light_ratio_role='comparison_only',
        comparison_grid=dict(spacing_mm=100.,height_mm=0.,origin_x_mm=-150.,origin_y_mm=-150.,source_kind='synthetic_assumed',normalization_population='16_grid_cells',physical_validation=False))
    write_json(out/'request.json',request)
    plants=[dict(plant_instance_id=p,source_plant_id='synthetic_'+p,axis_id=p+'/A0',age_days=30,reserve_gC=0,
        tx_mm=0,ty_mm=0,tz_mm=0,yaw_deg=0,transform_source='fixture_identity_transform',mass_source='fixture_explicit_zero_reserve') for p in ('p1','p2')]
    # Cross-plant shadow: upper p1 blade grows above p2 lower blade, with no shared carbon pool.
    leaves=[]
    for p,j,x,z,length,width in [('p1',1,-25,180,50,50),('p1',2,110,100,80,80),('p2',1,-40,60,80,80)]:
        c=p+f'/C{j:04d}'
        leaves.append(dict(plant_instance_id=p,axis_id=p+'/A0',compound_id=c,leaflet_id=c+'/T',parent_id=c,age_days=10,
            x_mm=x,y_mm=-length/2,z_mm=z,length_mm=length,width_mm=width,dry_mass_g=.02,potential_dm_g_day=.08,
            geometry_source='observation_like_fixture' if scenario=='observation' else 'synthetic_fixture',mass_source='fixture_assumed',age_source='fixture_assumed'))
    env=[dict(interval=i,start_time=(start+timedelta(days=i)).isoformat(),end_time=(start+timedelta(days=i+1)).isoformat(),light_hours=14,temp_day_c=26,temp_night_c=22,co2_ppm=400,source_kind='synthetic_assumed_constant_coefficients_no_temperature_response') for i in range(n)]
    power=0 if scenario.startswith('dark') else 20 if scenario=='limited' else 1200
    tables={'plants.csv':plants,'leaflets.csv':leaves,'environment.csv':env,'light_input.csv':[dict(reference_ppfd_umol_m2_s=300,source_kind='synthetic_assumed',measured_at='')],
        'lamps.csv':[dict(lamp_id='L1',x_mm=0,y_mm=0,z_mm=600,power_umol_s=power,source_kind='synthetic_assumed_point_LED_approximation')]}
    for name,data in tables.items(): write_csv(out/name,data,HEADERS[name].split(','))
    write_csv(out/'ratios.csv',[dict(grid_row=r,grid_col=c,ratio_to_mean=1) for r in range(1,5) for c in range(1,5)])
    (out/'light_ratio_latest.txt').write_text('ratios.csv\n',encoding='utf-8')
    return out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='op',required=True)
    p=sub.add_parser('audit'); p.add_argument('--input',type=Path,required=True)
    p=sub.add_parser('fork'); p.add_argument('--source',type=Path,default=DEFAULT_SOURCE); p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('prepare'); p.add_argument('--input',type=Path,required=True); p.add_argument('--fork',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('run'); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--java',default='java'); p.add_argument('--core',type=Path,default=CORE); p.add_argument('--timeout',type=float,default=180)
    p=sub.add_parser('collect'); p.add_argument('--run-dir',type=Path,required=True)
    p=sub.add_parser('demo'); p.add_argument('--out',type=Path,required=True); p.add_argument('--scenario',choices=('sufficient','limited','dark','dark_deficit','frozen','observation'),default='sufficient'); p.add_argument('--rays',type=int,default=300000)
    args=parser.parse_args()
    try:
        if args.op=='audit': result=dict(status='valid',source_hashes=load_package(args.input)['source_hashes'])
        elif args.op=='fork': result=dict(fork=str(create_fork(args.source,args.out)))
        elif args.op=='prepare': result=prepare(args.input,args.fork,args.out)
        elif args.op=='run': result=run(args.run_dir,args.java,args.core,args.timeout)
        elif args.op=='collect': result=collect(args.run_dir)
        else: result=dict(inputs=str(demo_inputs(args.out,args.scenario,args.rays)))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ContractError,OSError,ValueError,KeyError) as exc:
        parser.exit(1,f'{type(exc).__name__}: {exc}\n')


if __name__=='__main__': main()
