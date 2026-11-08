"""Export existing pipeline snapshots to standalone static GroIMP Model.rgg projects.

python -m tools.groimp_structure_export --run build/pipeline_TIMESTAMP --out build/xl_export
No inference, reconstruction, growth rules, light calculation or running API is required.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from tools.groimp_bridge import mm_to_m
from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json, write_rows

MODULES = ('Plant', 'MainStem', 'CompoundLeaf', 'SubStem', 'Leaflet', 'GrowthShoot', 'Group')
SCHEMA = 'static-xl-structure-1'


def text_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def java_string(value: str) -> str:
    # JSON escaping is compatible with Java string literals for these values.
    return json.dumps(str(value), ensure_ascii=True)


def scoped(plant: str, session: str, kind: str, original: str) -> str:
    return '/'.join(quote(str(x), safe='') for x in (plant, session, kind, original))


def points_m(value: object) -> list:
    import numpy as np
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise StudyError('Geometry must contain finite XYZ triples in board millimetres')
    return mm_to_m(a).tolist()


def load_snapshot(run: Path, plant: str) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', plant):
        raise StudyError('Invalid plant identifier')
    run = Path(run).resolve()
    path = run / plant / 'hierarchy' / 'hierarchy.json'
    h = read_json(path)
    if h.get('schema') != 'plant-hierarchy-1' or h.get('plant_id') != plant:
        raise StudyError(f'Wrong hierarchy schema or plant: {path}')
    session = h.get('session_id')
    if not session:
        raise StudyError('Hierarchy has no session_id')
    inputs = h['inputs']
    backup_path = Path(inputs['backup'])
    if not backup_path.is_absolute():
        raise StudyError('Hierarchy backup reference must be an absolute recorded path')
    actual = file_hash(backup_path)
    if not inputs.get('backup_sha256') or actual != inputs['backup_sha256']:
        raise StudyError(f'Annotation backup changed; recheck snapshot: {backup_path}')
    backup = read_json(backup_path)
    cap = backup.get('capture', {})
    if backup.get('plant_id') != plant or cap.get('session_id') != session:
        raise StudyError('Hierarchy and annotation backup plant/session disagree')
    structure_path = Path(inputs['structure']) / 'structure.json'
    structure = read_json(structure_path)
    status_path = run / plant / 'status.json'
    status = read_json(status_path)
    if structure.get('plant_id', plant) != plant or status.get('plant') != plant:
        raise StudyError('Structure/status plant mismatch')
    if structure.get('session_id', session) != session:
        raise StudyError('Structure session mismatch')
    sources = {str(p.resolve()): file_hash(p) for p in (path, backup_path, structure_path, status_path)}
    return dict(hierarchy=h, backup=backup, structure=structure, status=status, source_hashes=sources,
                backup_hash_verified=True, structure_historical_hash_available=False)


def make_graph(snapshot: dict) -> dict:
    h, backup = snapshot['hierarchy'], snapshot['backup']
    plant, session = h['plant_id'], h['session_id']
    state = backup['variants']['__all__']
    nodes, edges, geometry = [], [], []
    by_id = {}

    def node(kind: str, source: str, parent: str | None, relation: str, *, identity='unconfirmed',
             meta=None, proposed=False) -> str:
        key = scoped(plant, session, kind, source)
        if key in by_id:
            raise StudyError(f'Duplicate organ key: {key}')
        record = dict(key=key, module=kind, original_id=source, parent=parent, relation=relation,
                      identity=identity, proposed=proposed, metadata=meta or {})
        nodes.append(record); by_id[key] = record
        if parent:
            edges.append(dict(parent=parent, child=key, status=relation))
        return key

    def visual(key: str, kind: str, xyz: list, source: str) -> None:
        geometry.append(dict(organ_key=key, kind=kind, vertices_m=points_m(xyz),
                             source=source, display_only=kind != 'leaf_mesh',
                             hidden_by_default=by_id[key]['proposed']))

    root = node('Plant', plant, None, 'root', identity='plant', meta=dict(
        session_id=session, growth_reference=h.get('growth_reference'),
        validated_metric=False, suitable_for_ppfd=False,
        source_stage='S7 diagnostic snapshot; not S6 evidence', stage_status=snapshot['status']['stages']))
    unassigned = node('Group', 'unassigned', root, 'organizational')
    proposals = node('Group', 'proposals', root, 'organizational', proposed=True)
    observations = node('Group', 'photo_observations', root, 'organizational')
    stem = h.get('stem', {})
    if stem:
        key = node('MainStem', 'structure.stem', root, 'structural_estimate',
                   identity='fitted' if stem.get('status') == 'fitted' else 'unavailable', meta=stem)
        if stem.get('status') == 'fitted' and stem.get('axis'):
            visual(key, 'axis', stem['axis'], 'S3 fitted stem axis')

    membership = state.get('membership', {})
    parents = {}
    # Only explicitly registered or reviewed parents become biological parent IDs.
    reviewed_parent_ids = {v['membership_review']['parent_id'] for v in state.get('leaflets', {}).values()
                          if v.get('membership_review', {}).get('state') == 'confirmed'
                          and v['membership_review'].get('parent_id')}
    for pid in sorted(set(membership.get('parents', {})) | reviewed_parent_ids):
        parents[pid] = node('CompoundLeaf', pid, root, 'registered', identity='registered',
                            meta=dict(registration=membership.get('parents', {}).get(pid),
                                      source='annotation membership; no spatial region assumed'))

    regions = {}
    leaf_regions = {}
    for compound in h.get('compound_leaves', []):
        region = str(compound['region'])
        meta = {k: v for k, v in compound.items() if k not in ('leaflets', 'petiole')}
        regions[region] = node('CompoundLeaf', f'region:{region}', proposals, 'region_proposal',
                               identity='region_not_confirmed_parent', meta=meta, proposed=True)
        petiole = compound.get('petiole', {})
        if petiole.get('status') == 'fitted':
            key = node('SubStem', f'region:{region}:axis', proposals, 'unconfirmed_parent',
                       identity='fitted', meta=petiole, proposed=True)
            edges.append(dict(parent=regions[region], child=key, status='proposed'))
            visual(key, 'axis', [petiole['near_end'], petiole['far_end']], 'S3 fitted sub_stem axis')
        for leaf in compound.get('leaflets', []):
            if leaf['leaf_key'] in leaf_regions:
                raise StudyError('Leaflet appears in multiple hierarchy regions')
            leaf_regions[leaf['leaf_key']] = leaf
    for leaf in h.get('unassigned_leaflets', []):
        if leaf['leaf_key'] in leaf_regions:
            raise StudyError('Duplicate hierarchy leaflet')
        leaf_regions[leaf['leaf_key']] = leaf

    for leaf_key in sorted(set(state.get('leaflets', {})) | set(leaf_regions)):
        leaf = state.get('leaflets', {}).get(leaf_key, {})
        region = leaf_regions.get(leaf_key, {})
        review = leaf.get('membership_review', {})
        parent = parents.get(review.get('parent_id')) if review.get('state') == 'confirmed' else None
        statuses = leaf.get('view_status', {})
        confirmed = sum(s.get('identity_review') == 'confirmed' for s in statuses.values()) >= 2
        meta = dict(organ_id=leaf.get('organ_id'), leaflet_id=leaf.get('leaflet_id'),
                    leaflet_uid=leaf.get('leaflet_uid'), view_status=statuses,
                    photos=sorted(set(leaf.get('bboxes', {})) | set(region.get('photos', []))),
                    membership_review=review, cross_view_review=leaf.get('cross_view_review'),
                    hierarchy_region=region, geometry_source='S7 diagnostic region' if region else 'unavailable')
        key = node('Leaflet', leaf_key, parent or unassigned, 'confirmed' if parent else 'unassigned',
                   identity='confirmed' if confirmed else 'unconfirmed', meta=meta)
        if region.get('status') == 'region' and region.get('centroid_mm') is not None:
            visual(key, 'position', [region['centroid_mm']], 'S7 mask/hull intersection centre; not leaf surface')
        proposed_parent = regions.get(str(region.get('compound_region')))
        if proposed_parent:
            edges.append(dict(parent=proposed_parent, child=key, status='proposed'))

    for p in h.get('proposed_leaflets', []):
        key = node('Leaflet', 'candidate:' + str(p['key']), proposals, 'identity_proposal',
                   meta=p, proposed=True)
        if p.get('status') == 'region' and p.get('centroid_mm') is not None:
            visual(key, 'position', [p['centroid_mm']], 'S7 automatic candidate centre')
        if str(p.get('compound_region')) in regions:
            edges.append(dict(parent=regions[str(p['compound_region'])], child=key, status='proposed'))

    # Growth-shoot photo observations have no cross-view physical identity or 3D pose.
    for photo, detections in sorted(membership.get('detections', {}).items()):
        for d in detections:
            if d.get('class_name') not in ('growth_shoot', 'growth-shoot'):
                continue
            aid = d.get('annotation_id') or d.get('object_id')
            if not aid:
                raise StudyError('Growth-shoot observation has no annotation ID')
            node('GrowthShoot', f'{photo}:{aid}', observations, 'photo_observation', identity='unlinked_observation',
                 meta=dict(photo=photo, annotation_id=aid, bbox=d.get('bbox'), source=d.get('source'),
                           geometry_source='unavailable', note='not a unique physical shoot count'))
    return dict(schema=SCHEMA, plant_id=plant, session_id=session, root=root, nodes=nodes, relations=edges,
                geometry=geometry, validated_metric=False, suitable_for_ppfd=False)


def attach_meshes(graph: dict, geometry_path: Path, variant: str) -> None:
    from tomato3d.light.geometry import load_geometry
    surfaces = load_geometry([geometry_path])
    if variant not in surfaces:
        raise StudyError(f'Unknown geometry variant: {variant}')
    mesh = surfaces[variant]
    lookup = {}
    for n in graph['nodes']:
        if n['module'] == 'Leaflet' and not n['proposed']:
            meta = n['metadata']
            identity = (meta.get('organ_id'), meta.get('leaflet_id'))
            if identity in lookup:
                raise StudyError(f'Ambiguous surface identity: {identity}')
            lookup[identity] = n
    matched = 0
    for i in range(mesh.n_triangles):
        if mesh.plant_ids[i] != graph['plant_id'] or mesh.session_ids[i] != graph['session_id']:
            continue
        if not mesh.is_receiver[i]:
            raise StudyError('Optional geometry must contain leaf receiver surfaces only')
        n = lookup.get((mesh.organ_ids[i], mesh.leaflet_ids[i]))
        if n is None:
            raise StudyError(f'Leaf surface has no existing organ identity: {mesh.organ_ids[i]}/{mesh.leaflet_ids[i]}')
        graph['geometry'].append(dict(organ_key=n['key'], kind='leaf_mesh', vertices_m=points_m(mesh.vertices_board_mm[i]),
            source=mesh.geometry_sources[i], triangle_id=mesh.triangle_ids[i], normal_known=bool(mesh.normal_known[i]),
            review_status='not_supplied_by_geometry_table', display_only=False, hidden_by_default=False))
        matched += 1
    if not matched:
        raise StudyError('No optional geometry matches this plant and session')
    graph['geometry'] = [g for g in graph['geometry'] if g['kind'] != 'position' or
                        not any(t['kind'] == 'leaf_mesh' and t['organ_key'] == g['organ_key'] for t in graph['geometry'])]


def float_literal(value: float) -> str:
    if not math.isfinite(value):
        raise StudyError('Nonfinite RGG coordinate')
    return f'{value:.12g}f'


def render_rgg(graph: dict) -> str:
    nodes = {n['key']: n for n in graph['nodes']}
    children = {key: [] for key in nodes}
    visuals = {key: [] for key in nodes}
    for n in nodes.values():
        if n['parent']:
            children[n['parent']].append(n['key'])
    for g in graph['geometry']:
        visuals[g['organ_key']].append(g)
    lines = ['// Static diagnostic snapshot. No radiation or growth functions.',
             '// Board frame unchanged; all geometry in metres. Display markers are not organs.',
             'import de.grogra.graph.impl.Node;', 'import de.grogra.xl.util.FloatList;', '']
    for kind in MODULES:
        lines.append(f'module {kind}(String organKey, String originalId, String relationState, String identityState, String metadata) extends Node;')
    lines += ['module PositionMarker(String organKey, String metadata) extends Sphere(0.001);',
              'module AxisSegment(String organKey, String metadata) extends Line;',
              'module LeafSurface(String organKey, String metadata) extends MeshNode;',
              'boolean showCandidates = false;', '']

    def visual(g: dict, pad: str) -> list[str]:
        key = java_string(g['organ_key'])
        info = java_string(text_json({k: v for k, v in g.items() if k != 'vertices_m'}))
        v = g['vertices_m']
        if g['kind'] == 'position':
            xyz = ','.join(float_literal(c) for c in v[0])
            return [f'{pad}[ Translate({xyz}) PositionMarker({key}, {info}).(setShader(YELLOW)) ]']
        if g['kind'] == 'axis':
            result = []
            for a, b in zip(v, v[1:]):
                # Line setters are verified against the installed GroIMP class during integration.
                xyz = ','.join(float_literal(c) for c in a)
                delta = ','.join(float_literal(y-x) for x, y in zip(a,b))
                result.append(f'{pad}[ Translate({xyz}) AxisSegment({key}, {info}).(setAxis({delta})) ]')
            return result
        coords = ','.join(float_literal(c) for p in v for c in p)
        return [f'{pad}[ LeafSurface({key}, {info}).(setPolygons(getMesh(new FloatList(new float[]{{{coords}}})).getPolygons()), setShader(GREEN)) ]']

    def branch(key: str, depth: int) -> list[str]:
        n, pad = nodes[key], '    ' * depth
        meta = dict(n['metadata'], proposed_relations=[e for e in graph['relations'] if e['child'] == key and e['status'] == 'proposed'])
        args = ', '.join(java_string(v) for v in (key, n['original_id'], n['relation'], n['identity'], text_json(meta)))
        result = [f'{pad}{n["module"]}({args})']
        for g in visuals[key]:
            if g['hidden_by_default']:
                result.append(pad + 'if (showCandidates) (')
                result.extend(visual(g, pad+'    ')); result.append(pad+')')
            else:
                result.extend(visual(g,pad))
        for child in children[key]:
            result.append(pad+'['); result.extend(branch(child,depth+1)); result.append(pad+']')
        return result
    lines += ['protected void init()', '[', '    Axiom ==>'] + branch(graph['root'],1) + ['    ;', ']', '']
    lines += ['public void report()', '{']
    for kind in MODULES:
        lines.append(f'    println("COUNT {kind}=" + count((* {kind} *)));')
        lines.append(f'    for ({kind} n : (* {kind} *)) println("NODE\\t" + n.organKey + "\\t" + n.relationState + "\\t" + n.identityState + "\\t" + n.metadata);')
    if any(g['kind']=='leaf_mesh' for g in graph['geometry']):
        lines.append('    for (LeafSurface n : (* LeafSurface *)) println("MESH_VERTICES\\t" + n.organKey + "\\t" + n.metadata + "\\t" + java.util.Arrays.toString(((de.grogra.imp3d.objects.PolygonMesh)n.getPolygons()).getVertexData()));')
    lines += ['    println("MARKERS=" + count((* PositionMarker *)));',
              '    println("AXIS_SEGMENTS=" + count((* AxisSegment *)));',
              '    println("MESHES=" + count((* LeafSurface *)));', '}',
              '// Compile/Reset reconstructs the same snapshot; no growth button is defined.']
    return '\n'.join(lines)+'\n'


def export_plant(run: Path, plant: str, out: Path, geometry: Path | None = None, variant: str | None = None) -> dict:
    if bool(geometry) != bool(variant):
        raise StudyError('--geometry and --variant must be supplied together')
    snapshot = load_snapshot(run, plant)
    graph = make_graph(snapshot)
    if geometry:
        snapshot['source_hashes'][str(geometry.resolve())] = file_hash(geometry)
        attach_meshes(graph, geometry, variant)
    model = render_rgg(graph)
    # Recheck after all reads, before creating a successful output.
    for p, digest in snapshot['source_hashes'].items():
        if file_hash(Path(p)) != digest:
            raise StudyError(f'Input changed during export: {p}')
    out = output_dir(out)
    write_json(out/'structure_snapshot.json', graph)
    (out/'Model.rgg').write_text(model,encoding='utf-8')
    write_rows(out/'organs.csv', [dict(key=n['key'], module=n['module'], original_id=n['original_id'],
        parent=n['parent'], relation=n['relation'], identity=n['identity'], proposed=n['proposed'],
        metadata_json=text_json(n['metadata'])) for n in graph['nodes']],
        ('key','module','original_id','parent','relation','identity','proposed','metadata_json'))
    write_rows(out/'relationships.csv', graph['relations'], ('parent','child','status'))
    coords = [dict(organ_key=g['organ_key'], geometry_index=i, kind=g['kind'], vertex_index=j,
                  x_m=v[0], y_m=v[1], z_m=v[2], source=g['source'], display_only=g['display_only'])
              for i,g in enumerate(graph['geometry']) for j,v in enumerate(g['vertices_m'])]
    write_rows(out/'coordinates.csv', coords, ('organ_key','geometry_index','kind','vertex_index','x_m','y_m','z_m','source','display_only'))
    manifest = dict(schema=SCHEMA, plant_id=plant, session_id=graph['session_id'], status='generated_not_runtime_verified',
        source_hashes=snapshot['source_hashes'], backup_hash_verified=True,
        historical_structure_hash_available=False, coordinate_frame='board', units='metres', source_units='millimetres',
        scale_factor=0.001, module_counts=dict(Counter(n['module'] for n in graph['nodes'])),
        geometry_counts=dict(Counter(g['kind'] for g in graph['geometry'])),
        source_stage='S7 diagnostic snapshot; not S6 fusion', source_stages=snapshot['status']['stages'],
        validated_metric=False, suitable_for_ppfd=False, groimp_open_verified=False,
        save_reopen_verified=False, gui_visual_verified=False,
        unavailable=[dict(key=n['key'], reason='No existing 3D geometry; metadata only') for n in graph['nodes']
                     if n['module'] not in ('Plant','Group') and not any(g['organ_key']==n['key'] for g in graph['geometry'])],
        exporter_sha256=file_hash(Path(__file__)), model_sha256=file_hash(out/'Model.rgg'),
        snapshot_sha256=file_hash(out/'structure_snapshot.json'))
    write_json(out/'export.json',manifest)
    (out/'열기안내.md').write_text(
        '# 토마토 정적 XL 구조\n\nGroIMP에서 File → Open으로 이 폴더의 `Model.rgg`를 여세요. '
        '파일을 저장/컴파일하면 같은 스냅샷으로 초기화됩니다.\n\n'
        '- 기관 트리: Panels → Object Inspector에서 기관을 선택하고 Attribute Editor의 ID·metadata를 확인합니다.\n'
        '- 노란 점: 소엽 위치 표시입니다. 점 크기는 실제 잎 크기가 아닙니다.\n'
        '- 축 선: 기존 추정 줄기/잎자루 축입니다.\n'
        '- 녹색 면: 선택 입력으로 전달된 삼각형이며, 출처와 검토 상태를 확인해야 합니다.\n'
        '- proposals 그룹: 후보 관계/형상입니다. 형상은 기본 숨김입니다. '
        '코드의 showCandidates를 true로 바꾸고 저장하면 표시됩니다.\n'
        '- unassigned 그룹: 복엽 소속 미확정 소엽입니다.\n'
        '- photo_observations: 동일성 연결 전 생장점 관측이며 실제 생장점 개수가 아닙니다.\n\n'
        '`report` 버튼은 기관 목록과 개수를 출력합니다. File → Save As로 .gsz 파일을 저장하고 다시 열 수 있습니다.\n\n'
        '보드 원점과 축을 유지하며 좌표 단위는 m입니다(+Z는 보드에서 위쪽). '
        '1 mm = 0.001 m 변환을 한 번 적용했습니다.\n\n'
        '현재 결과는 진단용입니다. S7 보기 생성은 S6 성공·잎 표면 완성·PPFD 사용 가능을 뜻하지 않습니다. '
        '실제 열기/저장 검증 상태는 export.json과 별도 검증 기록을 확인하세요.\n',encoding='utf-8')
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--plant', action='append')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--geometry', type=Path)
    ap.add_argument('--variant')
    a = ap.parse_args(argv)
    if bool(a.geometry) != bool(a.variant):
        ap.error('--geometry and --variant must be supplied together')
    plants = a.plant or sorted(p.parent.parent.name for p in a.run.glob('*/hierarchy/hierarchy.json'))
    if not plants or len(set(plants)) != len(plants):
        ap.error('No plants found or duplicate --plant values')
    out = output_dir(a.out)
    results = {}
    for plant in plants:
        try:
            m = export_plant(a.run,plant,out/plant,a.geometry,a.variant)
            results[plant] = dict(status=m['status'], module_counts=m['module_counts'], geometry_counts=m['geometry_counts'])
        except (OSError, ValueError, KeyError) as e:
            results[plant] = dict(status='failed', error=str(e))
    write_json(out/'summary.json',results)
    print(text_json(results))
    return int(any(r['status']=='failed' for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
