"""Exact polygon projection diagnostic from point lamps; no transport rays."""
from __future__ import annotations

import argparse
import csv
import time
from itertools import groupby
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from tools.vtc_growth_io import require, rows, write_csv, write_json, sha

MIN_CAMERA_DEPTH_M = 1e-9
DEPTH_TIE_TOLERANCE_PER_M = 1e-9
MIN_PROJECTED_AREA_M2 = 1e-20


def clip_halfplane(vertices: list, coefficients: np.ndarray) -> list:
    """Clip a convex polygon where a*x+b*y+c >= 0 (no polygon repair)."""
    if len(vertices) == 0:
        return []
    result = []
    previous = np.asarray(vertices[-1]); old = float(coefficients[:2] @ previous + coefficients[2])
    for point in vertices:
        point = np.asarray(point); value = float(coefficients[:2] @ point + coefficients[2])
        if (old >= 0) != (value >= 0):
            result.append((previous + (point - previous) * old / (old - value)).tolist())
        if value >= 0:
            result.append(point.tolist())
        previous, old = point, value
    return result


def project_scene(triangles: list, lamp: np.ndarray) -> tuple[list, list, list]:
    """Central projection to a unit-distance image plane aimed at scene centroid.

    Inverse camera depth is affine over each projected triangle. Its difference
    gives the exact half-plane in which one triangle occludes another.
    """
    points = np.asarray([v for _, triangle in triangles for v in triangle])
    forward = points.mean(axis=0) - lamp
    require(np.linalg.norm(forward) > 0, 'lamp at scene centroid')
    forward /= np.linalg.norm(forward)
    reference = np.eye(3)[int(np.argmin(np.abs(forward)))]
    horizontal = np.cross(forward, reference); horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(forward, horizontal)
    camera = np.asarray([horizontal, vertical, forward]).T
    polygons, depths, owners = [], [], []
    for oid, triangle in triangles:
        xyz = (np.asarray(triangle) - lamp) @ camera
        if np.all(xyz[:, 2] <= 0):
            continue
        require(np.all(xyz[:, 2] > MIN_CAMERA_DEPTH_M), 'projection unavailable: triangle crosses lamp image plane')
        xy = xyz[:, :2] / xyz[:, 2, None]
        polygon = Polygon(xy)
        if polygon.area <= MIN_PROJECTED_AREA_M2:
            continue
        require(polygon.is_valid, 'invalid native projected triangle')
        inverse_depth = np.linalg.solve(np.column_stack((xy, np.ones(3))), 1 / xyz[:, 2])
        polygons.append(polygon); depths.append(inverse_depth); owners.append(oid)
    return polygons, depths, owners


def shading(triangles: list, lamp: list, compounds: dict, leaf_ids: list, progress=None) -> list[dict]:
    if not triangles:
        return [dict(organ_id=oid, own_projected_area_m2=0., visible_projected_area_m2=0.,
                     shading_rate='unavailable', self_shading_rate='unavailable',
                     other_shading_rate='unavailable', overlap_shading_rate='unavailable') for oid in leaf_ids]
    polygons, depths, owners = project_scene(triangles, np.asarray(lamp, dtype=float))
    tree = STRtree(polygons); owned = defaultdict(list)
    polygon_array = np.asarray(polygons, dtype=object)
    depth_array = np.asarray(depths)
    owner_array = np.asarray(owners)
    corners = np.asarray([list(p.exterior.coords)[:3] for p in polygons])
    homogeneous = np.concatenate((corners,np.ones((len(corners),3,1))),axis=2) if len(corners) else np.empty((0,3,3))
    for index, oid in enumerate(owners):
        owned[oid].append(index)
    result = []
    for oid in leaf_ids:
        started = time.perf_counter()
        projected = unary_union([polygons[i] for i in owned[oid]])
        own = projected.area
        visible, self_visible, other_visible = [], [], []
        has_self = has_other = False
        for i in owned[oid]:
            receiver = polygons[i]; self_shadows, other_shadows = [], []
            candidates = tree.query(receiver, predicate='intersects')
            # Filter own facets and entirely rearward triangles in a vectorized
            # batch before constructing any intersections. An affine inverse
            # depth reaches its extrema at triangle corners, so this is exact.
            candidates = candidates[owner_array[candidates] != oid]
            differences = depth_array[candidates] - depth_array[i]
            differences[:,2] -= DEPTH_TIE_TOLERANCE_PER_M
            signs = np.sum(homogeneous[candidates] * differences[:,None,:],axis=2)
            front = np.max(signs,axis=1)>0
            candidates, differences, signs = candidates[front], differences[front], signs[front]
            complete = np.min(signs, axis=1) >= 0
            # GEOS batch operations preserve every original triangle and the
            # same depth test; this is not mesh simplification or rasterization.
            shadows = shapely.intersection(receiver, polygon_array[candidates[complete]])
            for j, shadow in zip(candidates[complete], shadows):
                if shadow.area > 0:
                    same = compounds.get(oid) is not None and compounds.get(oid) == compounds.get(owners[int(j)])
                    (self_shadows if same else other_shadows).append(shadow)
            for raw_index, difference, values in zip(candidates[~complete],differences[~complete],signs[~complete]):
                j = int(raw_index)
                # Restrict to the portion that is strictly nearer the lamp.
                if np.min(values)>=0:
                    shadow = receiver.intersection(polygons[j])
                else:
                    clipped = clip_halfplane(corners[j], difference)
                    if len(clipped) < 3:
                        continue
                    shadow = receiver.intersection(Polygon(clipped))
                if shadow.area <= 0:
                    continue
                same = compounds.get(oid) is not None and compounds.get(oid) == compounds.get(owners[j])
                (self_shadows if same else other_shadows).append(shadow)
            if not self_shadows and not other_shadows:
                self_visible.append(receiver); other_visible.append(receiver); visible.append(receiver)
                continue
            has_self |= bool(self_shadows); has_other |= bool(other_shadows)
            a = unary_union(self_shadows) if self_shadows else None
            b = unary_union(other_shadows) if other_shadows else None
            sv = receiver.difference(a) if a is not None else receiver
            ov = receiver.difference(b) if b is not None else receiver
            self_visible.append(sv); other_visible.append(ov)
            visible.append(sv if b is None else ov if a is None else receiver.difference(a.union(b)))
        area = unary_union(visible).area if has_self or has_other else own
        total = 1 - area / own if own else 'unavailable'
        self_rate = (1 - unary_union(self_visible).area / own if has_self else 0.) if own else 'unavailable'
        other_rate = (1 - unary_union(other_visible).area / own if has_other else 0.) if own else 'unavailable'
        result.append(dict(organ_id=oid, own_projected_area_m2=own, visible_projected_area_m2=area,
                           shading_rate=total, self_shading_rate=self_rate, other_shading_rate=other_rate,
                           overlap_shading_rate=self_rate + other_rate - total if own else 'unavailable'))
        if progress is not None:
            progress(dict(organ_id=oid, completed_leaflets=len(result), total_leaflets=len(leaf_ids), elapsed_s=time.perf_counter()-started))
    return result


def export_shading(source: Path, output: Path, pkg: dict, days: list[int] | None = None) -> None:
    """One row per leaflet/state/lamp, including massless (unavailable) leaflets."""
    states = defaultdict(dict)
    for row in rows(source / 'states.csv'):
        states[int(row['day'])][row['organ_id']] = row
    if days is None and pkg.get('request', {}).get('engine_hooks', {}).get('planned_axes') is True:
        days = sorted(set(states) & {0, 10, 18, 30})
    selected = set(states) if days is None else set(days)
    require(bool(selected) and selected <= set(states), 'requested shading states unavailable')
    def mesh_frames():
        with (source/'meshes.csv').open(encoding='utf-8-sig',newline='') as stream:
            previous=-1
            for day, group in groupby(csv.DictReader(stream),key=lambda r:int(r['state'])):
                require(day>previous,'native mesh states not ordered');previous=day
                frame=defaultdict(list)
                for row in group:frame[row['organ_id']].append(row)
                yield day,frame
    frames=iter(mesh_frames());mesh_day,mesh=next(frames,(None,{}))
    result = []; timings = []
    for day, current in sorted(states.items()):
        if day not in selected:
            if mesh_day == day: mesh_day,mesh=next(frames,(None,{}))
            continue
        started = time.perf_counter()
        compounds = {}
        for oid, row in current.items():
            parent = row
            while parent['class'] != 'CompoundLeaf' and parent['parent_id']:
                parent = current[parent['parent_id']]
            compounds[oid] = parent['organ_id'] if parent['class'] == 'CompoundLeaf' else None
        triangles = []
        require(mesh_day is None or mesh_day>=day,'mesh/state mismatch')
        for oid, vertices in (mesh if mesh_day==day else {}).items():
            values = [[float(r[k]) for k in ('x_m', 'y_m', 'z_m')] for r in sorted(vertices, key=lambda r: int(r['vertex']))]
            require(len(values) % 3 == 0, 'native triangle vertex count')
            triangles.extend((oid, values[i:i+3]) for i in range(0, len(values), 3))
        leaf_ids = [oid for oid, row in current.items() if row['class'] == 'Leaflet']
        for lamp in pkg['tables']['lamps.csv']:
            def progress(record):
                write_json(output/'shading_progress.json',dict(day=day, lamp_id=lamp['lamp_id'], status='running', **record))
            for row in shading(triangles, [float(lamp[k]) for k in ('x_m', 'y_m', 'z_m')], compounds, leaf_ids, progress):
                result.append(dict(day=day, lamp_id=lamp['lamp_id'], plant_instance_id=current[row['organ_id']]['plant_instance_id'], compound_id=compounds[row['organ_id']], **row))
        timings.append(dict(day=day, elapsed_s=time.perf_counter()-started, triangles=len(triangles), leaflets=len(leaf_ids), lamps=len(pkg['tables']['lamps.csv'])))
        write_csv(output/'shading_timing.csv',timings)
        day_rows = [r for r in result if r['day']==day]
        if day_rows: write_csv(output/f'leaflet_shading_day_{day}.csv',day_rows)
        if mesh_day==day:mesh_day,mesh=next(frames,(None,{}))
        write_json(output/'shading_progress.json',dict(completed_day=day,rows=len(result),status='running'))
    require(mesh_day is None,'extra mesh states')
    write_csv(output / 'leaflet_shading.csv', result, ['day','lamp_id','plant_instance_id','compound_id','organ_id',
        'own_projected_area_m2','visible_projected_area_m2','shading_rate','self_shading_rate','other_shading_rate','overlap_shading_rate'])
    write_json(output / 'leaflet_shading_manifest.json', dict(
        projection='central_point_lamp_unit_distance_image_plane_aimed_at_scene_centroid',
        area_unit='m2_on_unit_distance_image_plane; ratio is dimensionless',
        occluders='all native above-ground mesh organs; opaque geometric silhouettes',
        split='same compound vs other organs; total=self+other-overlap; folds counted once',
        zero_area='unavailable', transport_runs_added=0, optical_absorption_estimate=False,
        selected_days=sorted(selected), geometry_decimated=False, state_timings=timings,
        numerical_tolerances=dict(camera_depth_m=MIN_CAMERA_DEPTH_M, inverse_depth_tie_per_m=DEPTH_TIE_TOLERANCE_PER_M, projected_area_m2=MIN_PROJECTED_AREA_M2),
        source_hashes={n: sha(source / n) for n in ('states.csv', 'meshes.csv')}, rows=len(result)))
    write_json(output/'shading_progress.json',dict(completed_day=max(selected),rows=len(result),status='complete'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--days', type=int, nargs='+', help='exact state days; dense H7 acceptance uses 0 10 18 30')
    args = parser.parse_args()
    from tools.vtc_development import verify_prepared
    from tools.vtc_growth_io import fresh_build
    _, pkg = verify_prepared(args.run_dir)
    export_shading(args.run_dir / 'working', fresh_build(args.out), pkg, args.days)


if __name__ == '__main__':
    main()
