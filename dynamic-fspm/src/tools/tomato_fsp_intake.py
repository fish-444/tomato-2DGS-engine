"""Producer-owned plane evidence and explicit conditional-completion binding."""
from __future__ import annotations
from pathlib import Path
from tools.vtc_growth_io import require, rows, read_json, sha, number


def plane_evidence(folder: Path, captures: list, areas: list, revision: int) -> tuple[dict, dict]:
    from tomato3d.state import schema as s
    from tools.tomato_fsp_io import inside
    planes = {c['capture_id']: s.reference_plane_of(c) for c in captures}
    s.check_areas(areas, {c['capture_id']: c['scale_status'] for c in captures}, planes)
    evidence = {c['plant_instance_id']+'|'+c['capture_id']: {
        k: c.get(k, 'unavailable') for k in ('scale_reference', 'scale_reference_plane',
        'scale_reference_plane_offset_mm', 'scale_source', 'scale_status', 'scale_plane', 'frame')}
        for c in captures}
    hashes = {}
    p = inside(folder, 'frame_transforms.csv')
    require(revision < 7 or p.is_file(), 'revision7_frame_transforms_missing')
    if p.is_file():
        frames = rows(p)
        legacy = ('plant_instance_id','capture_id','center_x_px','center_y_px','mm_per_px','rotation_deg','scale_status','rotation_status','method')
        if revision < 6 and frames and set(frames[0]) == set(legacy):
            # Historical producer schema; preserve absent plane/uncertainty as unavailable.
            s.check_columns(frames,legacy,'frame_transforms.csv legacy')
            for r in frames:
                if r['scale_status'] == 'solved':require(number(r['mm_per_px'],'legacy mm/px',0)>0,'legacy_frame_scale_invalid')
                if r['rotation_status'] == 'solved':number(r['rotation_deg'],'legacy rotation')
        else:
            if revision >= 7:s.check_columns(frames,s.FRAME_TRANSFORMS_COLUMNS,'frame_transforms.csv revision7')
            s.check_frame_transforms(frames)
        by = {(c['plant_instance_id'], c['capture_id']): c for c in captures}
        seen = set()
        for r in frames:
            key = r['plant_instance_id'], r['capture_id']
            require(key in by and key not in seen, 'frame_capture_identity_mismatch'); seen.add(key)
            c = by[key]
            if revision >= 7 and r['scale_status'] == 'solved':
                require(r['scale_reference_plane'] == c['scale_reference_plane'] and
                        r['scale_reference_plane_offset_mm'] == c['scale_reference_plane_offset_mm'],
                        'frame_capture_plane_disagreement')
        # Producer emits planar transforms for applicable views only; side poses are separate evidence.
        hashes['frame_transforms.csv'] = sha(p)
    return evidence, hashes


def validate_completion(folder: Path, scenario: dict, selected_vertices: list, surfaces: list) -> dict:
    from tools.tomato_fsp_io import inside
    path = inside(folder, 'completion_manifest.json')
    require(path.is_file(), 'versioned_producer_completion_manifest_unavailable')
    m = read_json(path)
    require(m.get('schema_version') == 'fsp-p1-completion-1', 'unsupported_completion_manifest')
    require(m.get('producer_version'), 'completion_producer_version_required')
    for key in ('scale_assumptions', 'depth_assumptions', 'posture_assumptions'):
        require(isinstance(m.get(key), list) and m[key], 'completion_assumptions_missing:'+key)
        for entry in m[key]:
            require(isinstance(entry, dict) and entry.get('source_kind') and entry.get('source_ref'),
                    'completion_assumption_provenance_missing:'+key)
    require(m.get('simulation_identity_map') == scenario.get('simulation_identity_map'), 'completion_identity_selection_mismatch')
    files = m.get('files', {})
    needed = {'captures.csv', 'identity_manifest.csv', 'mtg_vertices.csv', 'areas.csv',
              'optical_surface_manifest.csv', 'canopy_placement.json'}
    needed.update(s['surface_file'] for s in surfaces)
    require(needed <= set(files), 'completion_geometry_file_coverage')
    for name, h in files.items():
        require(name != 'completion_manifest.json' and sha(inside(folder, name)) == h,
                'completion_file_changed:'+name)
    identities = {(r['plant_instance_id'], r['capture_id'], r['organ_id']) for r in rows(folder/'identity_manifest.csv')}
    links = m.get('observation_links')
    require(isinstance(links, list) and links, 'completion_observation_links_required')
    expected = {v['plant_instance_id']+'|'+v['vertex_id'] for v in selected_vertices if v['class'] in ('CompoundLeaf', 'Leaflet')}
    actual = set()
    for link in links:
        key = link['simulation_organ_id']
        require(key in expected and key not in actual, 'completion_link_identity_coverage')
        source = (link['plant_instance_id'], link['capture_id'], link['observed_organ_id'])
        require(source in identities and scenario['captures_by_instance'].get(source[0]) == source[1], 'completion_observation_identity_missing')
        require(m['simulation_identity_map'].get(key) == source[2], 'completion_observation_mapping_disagreement')
        require(key.startswith(source[0]+'|') and link.get('source_ref'), 'completion_cross_plant_or_unsourced_link')
        actual.add(key)
    require(actual == expected and set(m['simulation_identity_map']) == expected, 'completion_link_identity_coverage')
    return dict(schema_version=m['schema_version'], manifest_sha256=sha(path),
                geometry_status='conditional_completion', observation_links=links,
                assumptions={k:m[k] for k in ('scale_assumptions','depth_assumptions','posture_assumptions')})
