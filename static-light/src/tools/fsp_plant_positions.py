"""Opt-in effective canopy placement; measured records and local organs stay immutable."""
from __future__ import annotations

import csv
import hashlib
import io
import math
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np

from tools.tomato_fsp_canopy import rigid
from tools.vtc_growth_io import ROOT, require

SWITCH = 'plant_positions'
FROZEN = 'plant_positions_pot_spec.csv'


def active(request: dict) -> bool:
    hooks = request.get('engine_hooks')
    return isinstance(hooks, dict) and hooks.get(SWITCH, False) is True


def _finite(value, label: str) -> float:
    require(type(value) in (int, float), 'plant_positions numeric value required: ' + label)
    try:
        result = float(value)
    except OverflowError:
        require(False, 'plant_positions nonfinite value: ' + label)
    require(math.isfinite(result), 'plant_positions nonfinite value: ' + label)
    return result


def validate(request: dict) -> None:
    hooks = request.get('engine_hooks')
    require(hooks is None or isinstance(hooks, dict), 'plant_positions engine_hooks must be an object')
    require(hooks is None or type(hooks.get(SWITCH, False)) is bool, 'plant_positions switch must be boolean')
    if not active(request):
        require(SWITCH not in request, 'plant_positions payload requires enabled switch')
        return
    require(request.get('integration_contract') == 'fsp-integration-1'
            or request.get('schema_version') == 'fsp-integration-1',
            'plant_positions requires imported fsp-integration-1 host')
    payload = request.get(SWITCH)
    require(isinstance(payload, dict) and bool(payload), 'plant_positions requires a nonempty object')
    if 'layout_scale' in payload:
        require(set(payload) == {'layout_scale'}, 'plant_positions scale and mapping cannot be mixed')
        require(_finite(payload['layout_scale'], 'layout_scale') > 0, 'plant_positions layout_scale must be positive')
    else:
        for pid, point in payload.items():
            require(isinstance(pid, str) and bool(pid), 'plant_positions plant ID required')
            require(isinstance(point, dict) and set(point) == {'x_m', 'y_m'},
                    'plant_positions point requires exactly x_m and y_m')
            for key in ('x_m', 'y_m'):
                _finite(point[key], pid + '.' + key)


def _pot_spec(folder: Path) -> tuple[Path, str, dict]:
    frozen = folder / FROZEN
    path = frozen if frozen.exists() else ROOT / 'data/records/pots/pot_spec.csv'
    require(path.is_file(), 'plant_positions measured pot specification missing')
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
            'plant_positions pot specification header invalid')
    records = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in records),
            'plant_positions pot specification CSV invalid')
    selected = [row for row in records if row.get('pot_type') == 'square_cell']
    require(len(selected) == 1, 'plant_positions requires one measured square_cell pot specification')
    row = selected[0]
    require(row.get('shape') == 'square' and row.get('source_kind') == 'measured',
            'plant_positions requires measured square pot footprint')
    require(row.get('pot_rim_outer_edge_mm') and float(row['pot_rim_outer_edge_mm']) == 60.,
            'plant_positions requires measured 60 mm outer pot edge')
    return path.resolve(), hashlib.sha256(content).hexdigest(), row


def _axes(placement: dict) -> tuple[np.ndarray, np.ndarray]:
    r, t = rigid(placement['rotation'], placement['translation_mm'])
    require(np.allclose(r[2], [0., 0., 1.], atol=1e-12, rtol=0)
            and np.allclose(r[:2, 2], [0., 0.], atol=1e-12, rtol=0),
            'plant_positions pot collision requires planar yaw orientation')
    yaw = math.atan2(r[1, 0], r[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s], [-s, c]]), t


def _overlap(a: dict, b: dict, half_edge_mm: float) -> bool:
    """Square SAT in millimetres; only floating-point roundoff permits touching."""
    axes_a, ta = _axes(a)
    axes_b, tb = _axes(b)
    delta = tb[:2] - ta[:2]
    for normal in (*axes_a, *axes_b):
        distance = abs(float(delta @ normal))
        radius = half_edge_mm * sum(abs(float(axis @ normal)) for axis in (*axes_a, *axes_b))
        roundoff = 8 * math.ulp(max(1., distance, radius))
        if distance >= radius - roundoff:
            return False
    return True


def _move_proxies(request: dict, deltas: dict, ids: set[str]) -> list[dict]:
    ext = request.get('extensions')
    if not ext:
        return []
    moved = []
    objects = ext.get('objects', [])
    require(isinstance(objects, list), 'plant_positions extension objects must be a list')
    for obj in objects:
        require(isinstance(obj, dict), 'plant_positions object must be a mapping')
        owner = obj.get('plant_instance_id')
        if 'plant_instance_id' in obj:
            require(isinstance(owner, str) and owner in ids, 'plant_positions unknown object plant association')
        else:
            require(obj.get('role') != 'pot', 'plant_positions pot requires explicit plant_instance_id')
            continue
        require(obj.get('frame') == 'object_local_m', 'plant_positions proxy frame must be object_local_m')
        matrix = np.asarray(obj.get('transform'), dtype=float)
        require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
                and np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-12, rtol=0),
                'plant_positions proxy transform invalid')
        rigid(matrix[:3, :3], matrix[:3, 3])
        dx, dy = deltas[owner]['dx_m'], deltas[owner]['dy_m']
        if dx != 0 or dy != 0:
            # Do not round-trip untouched rotation, z or homogeneous entries.
            obj['transform'][0][3] += dx
            obj['transform'][1][3] += dy
            require(math.isfinite(obj['transform'][0][3]) and math.isfinite(obj['transform'][1][3]),
                    'plant_positions proxy translation overflow')
            moved.append(dict(object_id=obj['object_id'], plant_instance_id=owner, dx_m=dx, dy_m=dy))
    return moved


def resolve_package(pkg: dict, folder: Path) -> dict | None:
    """Resolve once from raw package values, publishing only after every gate passes."""
    validate(pkg['request'])
    if not active(pkg['request']):
        return None
    require('plant_positions' not in pkg, 'plant_positions package was already resolved')
    geometry = deepcopy(pkg.get('integration_geometry'))
    request = deepcopy(pkg['request'])
    require(isinstance(geometry, dict) and isinstance(geometry.get('placement'), dict),
            'plant_positions imported placement missing')
    placement = geometry['placement']
    plant_ids = [row['plant_instance_id'] for row in pkg['tables']['plants.csv']]
    require(plant_ids and len(set(plant_ids)) == len(plant_ids) and set(placement) == set(plant_ids),
            'plant_positions placement/plant membership mismatch')
    ids = set(plant_ids)
    for point in placement.values():
        _axes(point)
    before = {pid: dict(x_m=p['translation_mm'][0] / 1000., y_m=p['translation_mm'][1] / 1000.)
              for pid, p in placement.items()}
    centroid = {key: math.fsum(point[key] / len(before) for point in before.values()) for key in ('x_m', 'y_m')}
    payload = request[SWITCH]
    after = deepcopy(before)
    scale = payload.get('layout_scale')
    if scale is not None:
        if scale != 1:
            for pid in after:
                for key in ('x_m', 'y_m'):
                    after[pid][key] = centroid[key] + scale * (before[pid][key] - centroid[key])
    else:
        require(set(payload) <= ids, 'plant_positions unknown plant ID')
        for pid, point in payload.items():
            after[pid] = {key: _finite(point[key], pid + '.' + key) for key in ('x_m', 'y_m')}
    deltas = {}
    for pid, point in after.items():
        for axis, key in enumerate(('x_m', 'y_m')):
            value = _finite(point[key], pid + '.' + key)
            if value != before[pid][key]:
                mm = value * 1000.
                require(math.isfinite(mm), 'plant_positions translation overflow')
                placement[pid]['translation_mm'][axis] = mm
        deltas[pid] = dict(dx_m=point['x_m'] - before[pid]['x_m'], dy_m=point['y_m'] - before[pid]['y_m'])
    pot_path, pot_sha, pot = _pot_spec(Path(folder))
    for a, b in combinations(plant_ids, 2):
        require(not _overlap(placement[a], placement[b], float(pot['pot_rim_outer_edge_mm']) / 2),
                'plant_positions overlapping measured pot footprints: ' + a + ', ' + b)
    moved = _move_proxies(request, deltas, ids)
    metadata = dict(pot_spec_path=str(pot_path), pot_spec_sha256=pot_sha,
                    pot_type=pot['pot_type'], pot_outer_edge_mm=60., pot_source_kind=pot['source_kind'],
                    mode='layout_scale' if scale is not None else 'mapping', layout_scale=scale,
                    before=before, after=after, deltas=deltas, centroid=centroid, moved_proxies=moved,
                    frame='canopy_m', raw_inputs_modified=False,
                    collision_method='yaw_oriented_square_SAT_touching_allowed_8_ulp_roundoff')
    pkg['integration_geometry'] = geometry
    pkg['request'] = request
    pkg['source_hashes']['resolved_plant_positions_pot_spec'] = pot_sha
    if (Path(folder) / FROZEN).exists():
        pkg['source_hashes'][FROZEN] = pot_sha
    pkg[SWITCH] = metadata
    return metadata
