"""Independent consumer of D3 mesh CSVs and optional native optical evidence."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
from pathlib import Path
import math
import shutil
import numpy as np
from tools.leaf_light.common import read_json,write_json,file_hash
from tools.groimp_density import parse_log


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))


def close(actual: float,expected: float,atol: float,rtol: float,label: str) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual-expected)>atol+rtol*abs(expected):
        raise ValueError('numeric mismatch: '+label)


def mesh_rows(path: Path) -> dict[str,np.ndarray]:
    result = {}
    with path.open(encoding='utf-8',newline='') as stream:
        for row in csv.reader(stream):
            if row[0] in result: raise ValueError('duplicate mesh ID')
            values = np.asarray(row[1:],float)
            if not len(values) or len(values)%9 or not np.isfinite(values).all(): raise ValueError('malformed geometry')
            result[row[0]]=values.reshape(-1,3,3)
    return result


def input_path(results: Path,item: dict,archive: dict) -> Path:
    """Resolve a frozen copy by the original hash; never change recorded hashes."""
    relative=archive.get(item['sha256'])
    if relative is None: return Path(item['path'])
    path=(results/relative).resolve()
    if not path.is_relative_to(results.resolve()): raise ValueError('archive path escaped results')
    return path


def archive_inputs(results: Path) -> dict:
    """Keep final evidence usable if this run's trash is later removed by its owner."""
    manifest=read_json(results/'manifest.json')
    records=list(manifest['sources'].values())+read_json(results/'mesh_inventory.json')
    optical=results/'optical'
    if (optical/'execution_settings.json').exists():
        records.extend(read_json(optical/'execution_settings.json')['sources'].values())
    old_mapping=read_json(results/'archived_inputs.json')['by_sha256'] if (results/'archived_inputs.json').exists() else {}
    target=results/'frozen_inputs';target.mkdir(exist_ok=True);mapping={}
    for item in records:
        source=input_path(results,item,old_mapping);digest=item['sha256']
        if file_hash(source)!=digest: raise ValueError('cannot archive changed input: '+str(source))
        name=digest[:16]+'_'+Path(item['path']).name;dest=target/name
        if not dest.exists():shutil.copyfile(source,dest)
        if file_hash(dest)!=digest:raise ValueError('archive copy mismatch')
        mapping[digest]=dest.relative_to(results).as_posix()
    write_json(results/'archived_inputs.json',dict(schema='d3-frozen-inputs-1',by_sha256=mapping,
        policy='Original manifests and hashes unchanged. Every copy was verified against its original recorded hash.'))
    return mapping


def verify(results: Path) -> dict:
    manifest = read_json(results/'manifest.json')
    archive=read_json(results/'archived_inputs.json')['by_sha256'] if (results/'archived_inputs.json').exists() else {}
    for relative,digest in manifest['artifacts'].items():
        path = (results/relative).resolve()
        if not path.is_relative_to(results.resolve()) or file_hash(path)!=digest: raise ValueError('artifact hash mismatch')
    for item in manifest['sources'].values():
        if file_hash(input_path(results,item,archive))!=item['sha256']: raise ValueError('source hash mismatch')
    inventory = read_json(results/'mesh_inventory.json')
    identities = read_rows(results/'identity.csv'); sid = {r['native_id']:r['surface_id'] for r in identities}
    held = {r['native_id'] for r in identities if r['held']=='True'}
    if len(held)!=5 or len(sid)!=129: raise ValueError('source inventory incomplete')
    stored_item = next(r for r in inventory if r['treatment']=='stored_source')
    stored = mesh_rows(input_path(results,stored_item,archive))
    quality = read_rows(results/'mesh_quality.csv')
    lookup = {(r['pool_size'],r['outline_seed'],r['grid_n'],r['treatment'],r['surface_id']):r for r in quality}
    mesh_areas = {}; held_checks=0; area_checks=0
    for item in inventory:
        path=input_path(results,item,archive)
        if file_hash(path)!=item['sha256']: raise ValueError('mesh hash mismatch')
        mesh=mesh_rows(path)
        if set(mesh)!=set(sid): raise ValueError('partial mesh inventory')
        name=f"p{item['pool_size']}_o{item['outline_seed']}_g{item['grid_n']}_{item['treatment']}"
        mesh_areas[name]={}
        for native_id,t in mesh.items():
            if native_id in held:
                if not np.array_equal(t,stored[native_id]): raise ValueError('held geometry changed in actual CSV')
                held_checks+=1
            area=float(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()/2)
            mesh_areas[name][native_id]=area
            if item['treatment']!='stored_source':
                q=lookup[(str(item['pool_size']),str(item['outline_seed']),str(item['grid_n']),item['treatment'],sid[native_id])]
                close(area*1e6,float(q['area_mm2']),1e-8,1e-10,'area_mm2');area_checks+=1
    preflight=read_json(results/'preparation_report.json')
    failed=[r for r in quality if r['status']=='unavailable']
    if len(failed)!=preflight['failed_checks']: raise ValueError('failure accounting mismatch')
    result=dict(status='passed',scope='conditional_fixture_geometry',mesh_csvs=len(inventory),held_csv_equalities=held_checks,
        independently_recomputed_areas=area_checks,area_atol_mm2=1e-8,area_rtol=1e-10,
        native_runs=0,optical_validation='unavailable',physical_validation=False)
    optical=results/'optical'
    if optical.exists():
        if failed or preflight['status']!='mesh_preflight_passed': raise ValueError('native evidence after failed geometry gate')
        report=read_json(optical/'report.json')
        if report['status']=='area_refinement_failed' and report['native_runs']==0:
            if any((optical/name).exists() for name in ('native_evidence','flux.csv','timing.csv')):
                raise ValueError('native artifacts contradict area-gate stop')
            area_rows=read_rows(optical/'fixed_scale_area.csv')
            if not any(r['below_0_1pct']=='False' for r in area_rows):raise ValueError('area-gate stop lacks failing evidence')
            result['optical_validation']='unavailable: fixed-scale area refinement required; no native execution'
            write_json(results/'independent_verification.json',result)
            return result
        for item in read_json(optical/'execution_settings.json')['sources'].values():
            if file_hash(input_path(results,item,archive))!=item['sha256']:raise ValueError('optical source hash mismatch')
        metadata=read_rows(optical/'identity.csv');timing=read_rows(optical/'timing.csv');flux=read_rows(optical/'flux.csv')
        groups={}
        for level,keys in [('leaflet',('plant_id','leaf_id','leaflet_id')),('leaf',('plant_id','leaf_id')),('plant',('plant_id',))]:
            for r in metadata: groups.setdefault((level,'|'.join(r[k] for k in keys)),[]).append(r['native_id'])
        native={};count=0
        for r in timing:
            with gzip.open(optical/'native_evidence'/(r['case']+'.rgg.gz'),'rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=r['model_sha256']: raise ValueError('native model hash mismatch')
            vals,probes=parse_log(optical/'native_evidence'/(r['case']+'.log'))
            if probes or set(vals)!=set(sid): raise ValueError('native log identity mismatch')
            native[r['case']]=vals
        for r in flux:
            case=f"{r['scenario']}_r{r['rays']}_s{r['seed']}";ids=groups[(r['level'],r['entity_id'])]
            expected=math.fsum(native[case][n][1] for n in ids)
            area=math.fsum(mesh_areas[r['scenario']][n] for n in ids)
            close(float(r['absorbed_umol_s']),expected,1e-10,1e-10,'native absorbed umol/s')
            close(float(r['area_m2']),area,1e-14,1e-10,'aggregation area m2')
            close(float(r['absorbed_umol_m2_s']),expected/area,1e-8,1e-10,'flux divided by supported area')
            if int(r['held_count'])!=len(held.intersection(ids)): raise ValueError('held count mismatch')
            count+=1
        if len(native)!=report['native_runs'] or len(flux)!=len(native)*len(groups): raise ValueError('missing optical records')
        noises=read_rows(optical/'ray_noise.csv')
        for row in noises:
            values=[float(r['absorbed_umol_s']) for r in flux if all(r[k]==row[k] for k in ('scenario','rays','level','entity_id'))]
            mean=math.fsum(values)/len(values)
            if len(values)>1:
                sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1))
                close(float(row['sd_umol_s']),sd,1e-10,1e-10,'sample SD')
                if mean>0: close(float(row['cv']),sd/mean,1e-10,1e-10,'CV')
        result.update(native_runs=len(native),native_aggregate_checks=count,independent_cv_checks=len(noises),
            flux_atol_umol_s=1e-10,flux_rtol=1e-10,optical_validation='native log/flux/area/CV accounting passed')
    write_json(results/'independent_verification.json',result)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--archive-inputs',action='store_true');args=p.parse_args()
    if args.archive_inputs:archive_inputs(args.results)
    print(verify(args.results))


if __name__=='__main__':main()
