"""Read-only P1 intake: observed, explicitly completed, or observation-only."""
from __future__ import annotations
import argparse
from pathlib import Path
from tools.vtc_growth_io import require,read_json,write_json,write_csv,rows,fresh_build,number,sha
from tools.tomato_fsp_io import audit_p1_package,selected_package,inside

ROUTES=('observed_metric','conditional_completion','observation_only','fixture_truth')


def metric_points(values: list,unit: str,frame: str) -> list:
    """Boundary primitive for explicit producer frames, never an image-depth estimator."""
    import numpy as np
    require((unit,frame) in (('mm','plant_local_mm'),('m','plant_local_m')),'unsupported units/frame; pixels cannot become metric')
    a=np.asarray(values,float);require(a.shape[-1]==3 and np.isfinite(a).all(),'finite 3D points required')
    return (a*(.001 if unit=='mm' else 1.)).tolist()


def inspect_route(package: Path,route: str,scenario: dict|None=None) -> dict:
    require(route in ROUTES,'unknown observation route');audit=audit_p1_package(package)
    require(audit['status']=='audited','P1 intake files unavailable')
    reasons=[];selected=None
    if route=='observation_only':reasons=list(audit['missing'])+['simulation_not_requested','hidden_state_supplement_required']
    elif scenario is None:reasons=['explicit_scenario_and_hidden_state_supplement_required']
    else:
        s=dict(scenario,observation_route=route)
        try:
            if route=='fixture_truth':require(s['fixture'] is True and s.get('state_source')=='fixture_truth','truth is fixture-only and must be explicit')
            else:require(s.get('state_source','main')=='main','no automatic truth fallback')
            if route=='conditional_completion':
                require(s.get('allow_conditional_surfaces') is True,'conditional completion must be selected')
                require(s.get('completion_manifest','completion_manifest.json')=='completion_manifest.json','completion manifest must use the hash-bound package entrypoint')
                manifest_path=inside(package,s.get('completion_manifest','completion_manifest.json'))
                require(manifest_path.is_file(),'versioned producer completion manifest unavailable')
                m=read_json(manifest_path);require(m.get('schema_version')=='fsp-p1-completion-1','unsupported completion manifest')
                require(all(m.get(k) for k in ('producer_version','observation_links','scale_assumptions','depth_assumptions','posture_assumptions','simulation_identity_map','files')),'completion assumptions/mapping incomplete')
                for f,h in m['files'].items():require(sha(inside(package,f))==h,'completion file changed')
                require(s.get('simulation_identity_map')==m['simulation_identity_map'],'explicit completion ID selection mismatch')
            selected=selected_package(package,s)
            if route=='observed_metric':require(all(x['geometry_source']!='generated_candidate' for x in selected['surfaces']),'generated completion is not observed metric')
            hidden=s.get('organ_initialization',{});plants=s.get('plant_initialization',{})
            require(bool(s.get('initialization_provenance')),'hidden state provenance missing')
            for v in selected['vertices']:
                key=v['plant_instance_id']+'|'+v['vertex_id'];h=hidden.get(key)
                require(h is not None,'hidden state missing: '+key)
                require(h.get('units')=={'mass':'gDM','age':'degree_day'} and h.get('provenance'),'hidden state units/source required')
                for field in ('mass_gDM','max_mass_gDM','thermal_age_dd','true_leaf_order'):number(h[field],field,0)
            for pid in selected['captures']:
                require(pid in plants,'root/reserve/apex hidden state missing: '+pid)
                for key in ('reserve_gC','root_mass_gDM','stem_mass_gDM','thermal_dd','produced','leaf_order','truss_count','axis_leaf_count'):number(plants[pid][key],key,0)
        except (ValueError,KeyError,OSError) as e:reasons.append(str(e))
    ready=selected is not None and not reasons
    return dict(schema_version='fsp-observation-intake-1',route=route,intake_status='accepted',simulation_status='ready_conditionally' if ready else 'unavailable',
        engineering_status='intake_verified',real_input_readiness='selected_inputs_ready' if ready and not audit['fixture'] else 'not_real_initialized',biological_validation='not_established',
        fixture=audit['fixture'],fixture_truth_substitution=route=='fixture_truth',source_hashes=audit['source_hashes'],producer_schema_sha256=audit['producer_schema_sha256'],
        producer_schema=audit['schema_version'],producer_revision=audit.get('schema_revision'),scale_plane_evidence=audit.get('scale_plane_evidence'),completion=selected.get('completion') if selected else None,missing_requirements=reasons,physical_units='actual p1_state_v1 located geometry uses mm; alternate m producer schema not available',
        hidden_state_source='separate scenario supplement; never measured from shoot images',audit=audit)


def write_intake(package: Path,route: str,scenario: dict|None,out: Path,existing: bool=False) -> dict:
    result=inspect_route(package,route,scenario)
    if not existing:out=fresh_build(out)
    evidence=[]
    for file in ('captures.csv','mtg_vertices.csv','areas.csv'):
        for row in rows(package/file):
            for key,value in row.items():
                if key.endswith('__status'):evidence.append(dict(table=file,plant_instance_id=row['plant_instance_id'],capture_id=row['capture_id'],organ_id=row.get('vertex_id',row.get('leaflet_id','')),field=key[:-8],value=row.get(key[:-8],'unavailable'),status=value,evidence=row.get('evidence','unavailable')))
    write_json(out/'observation_readiness.json',result);write_csv(out/'observation_field_evidence.csv',evidence,['table','plant_instance_id','capture_id','organ_id','field','value','status','evidence'])
    if scenario:write_json(out/'hidden_state_supplement.json',dict(schema_version='fsp-hidden-state-1',source='explicit scenario',provenance=scenario.get('initialization_provenance'),organs=scenario.get('organ_initialization'),plants=scenario.get('plant_initialization'),axes=scenario.get('axes')))
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--p1',type=Path,required=True);p.add_argument('--route',choices=ROUTES,required=True);p.add_argument('--scenario',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=write_intake(a.p1,a.route,read_json(a.scenario) if a.scenario else None,a.out);print(r['simulation_status'],r['missing_requirements'])


if __name__=='__main__':main()
