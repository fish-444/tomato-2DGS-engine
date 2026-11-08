"""Verify a generated static model using the installed GroIMP CLI, without GUI automation."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from tools.groimp_structure_export import MODULES
from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json

NS = '{http://grogra.de/xmlpersistence}'


def cli_path(path: Path) -> str:
    value = str(path.resolve())
    if os.name == 'nt':
        buf = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetShortPathNameW(value, buf, len(buf)):
            raise StudyError('Cannot resolve a GroIMP CLI-safe directory')
        value = buf.value
    if any(c.isspace() for c in value):
        raise StudyError('GroIMP CLI requires a directory without spaces (Windows short path unavailable)')
    return value


def archive_state(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read('graph.xml'))
    organs, visuals = [], []

    def walk(n, parent=None, position=None):
        fields = {d.get(NS+'name'): d.get(NS+'value') for d in n if d.tag in (NS+'data', NS+'object')}
        kind = n.get('type', '').split('$')[-1]
        if kind in MODULES:
            # GroIMP wraps strings requiring escaping as quoted Java-compatible literals.
            def decode(value):
                return json.loads(value) if value and value.startswith('"') else value
            fields = {k: decode(v) for k,v in fields.items()}
            organs.append(dict(key=fields['organKey'], module=kind, parent=parent,
                original_id=fields['originalId'], relation=fields['relationState'],
                identity=fields['identityState'], metadata=json.loads(fields['metadata'])))
            parent = fields['organKey']
        if n.get('type') == 'de.grogra.turtle.Translate':
            position = [float(fields.get('translate'+axis, 0)) for axis in 'XYZ']
        if kind in ('PositionMarker','AxisSegment','LeafSurface'):
            item = dict(kind=kind, key=fields['organKey'], position=position)
            if kind == 'AxisSegment':
                item['axis'] = [float(x) for x in fields['axis'].split()]
            if kind == 'LeafSurface':
                item['mesh_xml'] = ET.tostring(next(x for x in n if x.get(NS+'name')=='polygons'), encoding='unicode')
                meta=fields['metadata']
                if meta.startswith('"'):
                    meta=json.loads(meta)
                item['triangle_id']=json.loads(meta)['triangle_id']
            visuals.append(item)
        for child in n:
            if child.tag == NS+'node':
                walk(child, parent, position)
    for n in root:
        if n.get('root') == 'MainGraph':
            walk(n)
    return dict(organs=organs, visuals=visuals)


def validate_state(state: dict, graph: dict, mesh_report=None) -> dict:
    actual = {n['key']:n for n in state['organs']}
    if len(actual) != len(state['organs']) or set(actual) != {n['key'] for n in graph['nodes']}:
        raise StudyError('GroIMP organ identities/counts do not match the snapshot')
    for n in graph['nodes']:
        expected_meta = dict(n['metadata'], proposed_relations=[e for e in graph['relations']
                             if e['child']==n['key'] and e['status']=='proposed'])
        expected = {k:n[k] for k in ('key','module','parent','original_id','relation','identity')}
        expected['metadata'] = expected_meta
        if actual[n['key']] != expected:
            raise StudyError('GroIMP metadata/relationship mismatch: '+n['key'])
    expected_visuals = []
    for g in graph['geometry']:
        if g['hidden_by_default']:
            continue
        v = g['vertices_m']
        if g['kind']=='position':
            expected_visuals.append(dict(kind='PositionMarker', key=g['organ_key'], position=v[0]))
        elif g['kind']=='axis':
            for a,b in zip(v,v[1:]):
                expected_visuals.append(dict(kind='AxisSegment',key=g['organ_key'],position=a,axis=np.subtract(b,a).tolist()))
        else:
            expected_visuals.append(dict(kind='LeafSurface',key=g['organ_key'],vertices=v,triangle_id=g['triangle_id']))
    # Match in stable per-organ order, independent of graph traversal order.
    sortkey = lambda x:(x['key'],x['kind'],x.get('triangle_id',''))
    actual_visuals=sorted(state['visuals'],key=sortkey)
    expected_visuals=sorted(expected_visuals,key=sortkey)
    if len(actual_visuals)!=len(expected_visuals):
        raise StudyError('Visible geometry count mismatch')
    error=0.0
    for a,e in zip(actual_visuals,expected_visuals):
        if (a['key'],a['kind'])!=(e['key'],e['kind']):
            raise StudyError('Geometry-to-organ mismatch')
        for field in ('position','axis'):
            if field in e:
                error=max(error,float(np.max(np.abs(np.subtract(a[field],e[field]))))*1000)
        if e['kind']=='LeafSurface':
            # GroIMP serializes PolygonMesh as Java binary. Read actual vertices through its public getter.
            identity=(a['key'],a['triangle_id'])
            if mesh_report is None or identity not in mesh_report:
                raise StudyError('Saved mesh has no runtime vertex report')
            xyz=np.array(mesh_report[identity])
            expected=np.array(e['vertices']).ravel()
            if xyz.shape!=expected.shape:
                raise StudyError('Saved mesh vertex count differs')
            error=max(error,float(np.max(np.abs(xyz-expected)))*1000)
    if error>0.001:
        raise StudyError(f'Coordinate transport exceeds 0.001 mm: {error}')
    return dict(organ_count=len(actual), visible_geometry_count=len(actual_visuals), max_coordinate_error_mm=error)


def verify(project: Path, core: Path, out: Path, java='java') -> dict:
    project=project.resolve()
    manifest=read_json(project/'export.json')
    if file_hash(project/'Model.rgg')!=manifest['model_sha256']:
        raise StudyError('Model.rgg changed since export')
    if file_hash(project/'structure_snapshot.json')!=manifest.get('snapshot_sha256'):
        raise StudyError('Structure snapshot changed or has no recorded hash')
    graph=read_json(project/'structure_snapshot.json')
    out=output_dir(out)
    shutil.copy2(project/'Model.rgg',out/'Model.rgg')
    prefix='$cd '+cli_path(out)+'\n'
    expected=Counter(n['module'] for n in graph['nodes'])

    def run(name,commands,reports):
        # CLI model methods are asynchronous. Wait for each report/save before the next command.
        log_path=out/(name+'.log')
        with log_path.open('wb') as stream:
            process=subprocess.Popen([java,'-Dfile.encoding=UTF-8','-jar',str(core.resolve()),'--headless','-a','cli'],
                stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            deadline=time.monotonic()+60
            def wait_for(predicate):
                while time.monotonic()<deadline:
                    if predicate(log_path.read_text(encoding='utf-8',errors='replace')):
                        return
                    if process.poll() is not None:
                        break
                    time.sleep(0.05)
                raise StudyError(f'GroIMP timed out; inspect {name}.log')
            try:
                wait_for(lambda log:'[base]>>' in log)
                seen_reports=0
                for command in (prefix+commands).splitlines():
                    process.stdin.write(command+'\n'); process.stdin.flush()
                    if command.startswith('$open'):
                        wait_for(lambda log:'[Model]>>' in log or '[saved]>>' in log)
                        time.sleep(0.5)
                    elif command=='report':
                        seen_reports+=1
                        wait_for(lambda log:len(re.findall(r'MESHES=\d+',log))>=seen_reports)
                    elif command=='compile':
                        time.sleep(2)
                    elif command.startswith('saveas '):
                        target=out/command.split(' ',1)[1]
                        wait_for(lambda log:target.exists() and zipfile.is_zipfile(target))
                    else:
                        time.sleep(0.1)
                process.stdin.close()
                process.wait(timeout=max(1,deadline-time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
        log=log_path.read_text(encoding='utf-8',errors='replace')
        if process.returncode or 'Command not found' in log:
            raise StudyError(f'GroIMP command failed; inspect {name}.log')
        for kind in MODULES:
            values=[int(v) for v in re.findall(r'COUNT '+kind+r'=(\d+)',log)]
            if values!=[expected[kind]]*reports:
                raise StudyError(f'GroIMP report missing or inconsistent: {name}/{kind}: {values}')
        meshes={}
        for line in log.rsplit('COUNT Plant=',1)[-1].splitlines():
            if line.startswith('MESH_VERTICES\t'):
                _,key,metadata,vertices=line.split('\t')
                identity=(key,json.loads(metadata)['triangle_id'])
                if identity in meshes:
                    raise StudyError('Duplicate mesh triangle identity in runtime')
                meshes[identity]=json.loads(vertices)
        return meshes

    try:
        meshes=run('open_reset_save','$open Model.rgg\nreport\ncompile\nreport\nsaveas saved.gsz\n',2)
        first=archive_state(out/'saved.gsz')
        check=validate_state(first,graph,meshes)
        meshes=run('reopen','$open saved.gsz\nreport\nsaveas reopened.gsz\n',1)
        second=archive_state(out/'reopened.gsz')
        validate_state(second,graph,meshes)
        if first!=second:
            raise StudyError('Saved/reopened graph differs')
        result=dict(status='passed',groimp_open_verified=True,reinitialization_verified=True,
            save_reopen_verified=True,gui_visual_verified=False,**check)
    except (ValueError,OSError,subprocess.SubprocessError) as exc:
        result=dict(status='failed',error=str(exc),gui_visual_verified=False)
    result.update(project=str(project),model_sha256=manifest['model_sha256'],core_sha256=file_hash(core),
                  validated_metric=False,suitable_for_ppfd=False)
    write_json(out/'verification.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',required=True,type=Path)
    ap.add_argument('--core',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--java',default='java')
    a=ap.parse_args()
    result=verify(a.project,a.core,a.out,a.java)
    print(json.dumps(result,ensure_ascii=True))
    return int(result['status']!='passed')


if __name__=='__main__':
    raise SystemExit(main())
