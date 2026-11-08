"""Clip a landmark-interpolated diagnostic leaf patch to observed image regions.

python -m tools.visible_leaf_patch --capture build/.../capture.json
  --depths build/.../editor_point_depths.csv --annotations build/.../annotations.edited.json
  --leaflet B1 --out build/.../visible_B1

This is a piecewise-linear partial surface, never a measured whole-leaf shape.
Independent polygon indices are never interpreted as corresponding 3D points.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError
from PIL import Image, ImageDraw

from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, rows, write_json
from tools.manual_leaf_preview import annotation_polygon, project
from tools.leaf_depth_records import board_evidence_fields
from tools.export_leaflet_training import region_mask


def subdivide(triangle: np.ndarray, levels: int) -> list[np.ndarray]:
    faces = [triangle]
    for _ in range(levels):
        children = []
        for a, b, c in faces:
            ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
            children.extend((np.array([a, ab, ca]), np.array([ab, b, bc]),
                             np.array([ca, bc, c]), np.array([ab, bc, ca])))
        faces = children
    return faces


def inside_mask(triangle_mm: np.ndarray, frame: dict, intrinsics: dict, mask: np.ndarray) -> bool:
    # Sample curved projected edges before rasterizing the complete face footprint.
    samples = np.array([a*(1-t)+b*t for a, b in zip(triangle_mm, np.roll(triangle_mm, -1, axis=0))
                        for t in (0., .25, .5, .75)])
    rotation = cv2.Rodrigues(np.asarray(frame['rvec_rad'], dtype=float))[0]
    if np.any((samples @ rotation.T + np.asarray(frame['tvec_mm']))[:, 2] <= 0):
        return False
    projected = project(samples, frame, intrinsics)
    if not np.isfinite(projected).all():
        return False
    height, width = mask.shape
    if (projected < 0).any() or (projected[:, 0] >= width).any() or (projected[:, 1] >= height).any():
        return False
    low = np.floor(projected.min(axis=0)).astype(int)
    high = np.ceil(projected.max(axis=0)).astype(int)
    if high[0] >= width or high[1] >= height:
        return False
    footprint = np.zeros((high[1]-low[1]+1, high[0]-low[0]+1), np.uint8)
    cv2.fillPoly(footprint, [np.rint((projected-low)*256).astype(np.int32)], 1, shift=8)
    region = mask[low[1]:high[1]+1, low[0]:high[0]+1]
    return bool(footprint.any() and not np.any((footprint > 0) & (region == 0)))


def make_patch(capture: dict, depth_rows: list[dict], region: dict, levels: int, margin_px: int) -> dict:
    if not 0 <= levels <= 6 or not 0 <= margin_px <= 20:
        raise StudyError('Subdivision or pixel-margin value outside supported range')
    frames = {f['image_file']: f for f in capture['frames']}
    observed_regions = region.get('observations', [region])
    if len({r['image_file'] for r in observed_regions}) != len(observed_regions):
        raise StudyError('Duplicate region photograph')
    masks = []
    intr = capture['intrinsics']
    for observation in observed_regions:
        frame = frames.get(observation['image_file'])
        if frame is None or frame['pose_status'] != 'solved':
            continue
        polygon = annotation_polygon(observation, intr)
        mask = np.zeros((intr['image_height_px'], intr['image_width_px']), np.uint8)
        if 'rings' in observation:
            for ring in observation['rings']:
                points = np.asarray(ring['points_px'], dtype=float)
                if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or
                        not np.isfinite(points).all() or (points < 0).any() or
                        (points[:, 0] >= mask.shape[1]).any() or (points[:, 1] >= mask.shape[0]).any()):
                    raise StudyError('Invalid original-pixel mask rings')
            mask = region_mask(observation['rings'], mask.shape[1], mask.shape[0])
        else:
            cv2.fillPoly(mask, [np.rint(polygon*256).astype(np.int32)], 1, shift=8)
        if margin_px:
            mask = cv2.erode(mask, np.ones((2*margin_px+1, 2*margin_px+1), np.uint8))
        masks.append((frame, mask))
    if len(masks) < 2:
        raise StudyError('At least two independently observed regions with solved poses are required')
    grouped = {}
    for row in depth_rows:
        if (row['organ_id'], row['leaflet_id']) != (region['organ_id'], region['leaflet_id']):
            continue
        if row.get('coordinate_frame') != 'board':
            raise StudyError('Expected board-coordinate point depths')
        frame = frames.get(row['image_file'])
        if frame is None or row.get('image_sha256') != frame.get('sha256'):
            raise StudyError('Depth observation source image mismatch')
        try:
            xyz = np.array([float(row[k]) for k in ('x_board_mm', 'y_board_mm', 'z_board_mm')])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(xyz).all():
            continue
        old = grouped.get(row['point_id'])
        if old is not None and not np.array_equal(old, xyz):
            raise StudyError('Conflicting positions for a physical point')
        grouped[row['point_id']] = xyz
    if len(grouped) < 3:
        raise StudyError('At least three finite physical landmarks are required; no missing point is filled')
    xyz_mm = np.array(list(grouped.values()))
    centered = xyz_mm-xyz_mm.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise StudyError('Physical points are collinear')
    _, _, vt = np.linalg.svd(centered)
    uv = centered @ vt[:2].T
    try:
        seed_faces = Delaunay(uv).simplices
    except QhullError as exc:
        raise StudyError('Cannot parameterize the physical landmark patch') from exc
    accepted, proposed = [], 0
    for face in seed_faces:
        for tri in subdivide(xyz_mm[face], levels):
            proposed += 1
            if all(inside_mask(tri, f, intr, mask) for f, mask in masks):
                accepted.append(tri)
    if not accepted:
        raise StudyError('No interpolated surface fragment lies inside all observed regions')
    vertices, lookup, faces = [], {}, []
    for triangle in accepted:
        face = []
        for v in triangle:
            key = tuple(float(x) for x in v)
            if key not in lookup:
                lookup[key] = len(vertices)
                vertices.append(list(key))
            face.append(lookup[key])
        faces.append(face)
    area_mm2 = sum(float(np.linalg.norm(np.cross(t[1]-t[0], t[2]-t[0]))/2) for t in accepted)
    return dict(schema_version='visible-landmark-patch-1', organ_id=region['organ_id'], leaflet_id=region['leaflet_id'],
                coordinate_frame='board', vertices_mm=vertices, faces=faces,
                physical_landmarks=[dict(point_id=k, xyz_mm=v.tolist()) for k, v in grouped.items()],
                source_images=[f['image_file'] for f, _ in masks], proposed_faces=proposed,
                retained_faces=len(faces), interpolated_patch_area_mm2=area_mm2,
                subdivision_levels=levels, mask_margin_px=margin_px,
                all_source_pose_gates_passed=all(f.get('passes_pose_gate') is True for f, _ in masks),
                geometry_source='piecewise_linear_landmark_interpolation_clipped_to_observed_regions',
                interpolation_vertices_are_observations=False, independent_polygon_vertices_are_correspondences=False,
                complete_leaf=False, measured_leaf_area=False, measured_curvature=False,
                validated_metric=False, suitable_for_ppfd=False)


def build_patch(capture_path: Path, depth_path: Path, annotations_path: Path, leaflet: str, out: Path,
                levels: int = 4, margin_px: int = 1) -> dict:
    capture, annotations = read_json(capture_path), read_json(annotations_path)
    candidates = [r for r in annotations['two_d_only_candidates'] if r['leaflet_id'] == leaflet]
    if len(candidates) != 1:
        raise StudyError('Choose exactly one independently observed leaf region')
    depth_rows = rows(depth_path)
    patch = make_patch(capture, depth_rows, candidates[0], levels, margin_px)
    patch['source_hashes'] = {str(p.resolve()): file_hash(p) for p in (capture_path, depth_path, annotations_path)}
    source_paths = {}
    for frame in capture['frames']:
        if frame['image_file'] not in patch['source_images']:
            continue
        path = Path(frame['image_path'])
        if not path.is_absolute():
            path = capture_path.parent/path
        if file_hash(path) != frame['sha256']:
            raise StudyError('Original photograph hash mismatch')
        board_evidence_fields(frame)
        source_paths[frame['image_file']] = path
    folder = output_dir(out)
    patch['photo_reviews'] = []
    photos_html = []
    region = candidates[0]
    for index, observation in enumerate(region.get('observations', [region])):
        if observation['image_file'] not in patch['source_images']:
            continue
        frame = next(f for f in capture['frames'] if f['image_file'] == observation['image_file'])
        path = source_paths[frame['image_file']]
        with Image.open(path) as source:
            photo = source.convert('RGB')
        overlay = Image.new('RGBA', photo.size)
        drawing = ImageDraw.Draw(overlay)
        projected = project(np.array(patch['vertices_mm']), frame, capture['intrinsics'])
        for face in patch['faces']:
            drawing.polygon([tuple(projected[i]) for i in face], fill=(55, 220, 110, 125))
        polygon = annotation_polygon(observation, capture['intrinsics'])
        for ring in observation.get('rings', [dict(points_px=polygon.tolist())]):
            points = [tuple(p) for p in ring['points_px']]
            drawing.line(points+[points[0]], fill=(255,80,225,255), width=3)
        observed_xy = []
        for row in depth_rows:
            if (row['image_file'],row['organ_id'],row['leaflet_id']) != (frame['image_file'],region['organ_id'],leaflet):
                continue
            x_px,y_px = float(row['x_px']),float(row['y_px'])
            observed_xy.append([x_px,y_px])
            drawing.ellipse((x_px-6,y_px-6,x_px+6,y_px+6), fill=(255,225,55,255))
            drawing.text((x_px+8,y_px-12),row['point_id'],fill=(255,225,55,255),stroke_width=1,stroke_fill=(0,0,0,255))
        merged = Image.alpha_composite(photo.convert('RGBA'),overlay).convert('RGB')
        bounds = np.vstack([polygon, np.asarray(observed_xy).reshape(-1,2)])
        low = np.maximum(np.floor(bounds.min(axis=0)).astype(int)-40,0)
        high = np.minimum(np.ceil(bounds.max(axis=0)).astype(int)+40,photo.size)
        crop = (int(low[0]),int(low[1]),int(high[0]),int(high[1]))
        name = f'photo_{index}.jpg'
        merged.crop(crop).save(folder/name,quality=95,subsampling=0)
        evidence = board_evidence_fields(frame)
        patch['photo_reviews'].append(dict(image_file=frame['image_file'],source_sha256=frame['sha256'],crop_xyxy_px=crop,overlay_file=name,**evidence))
        markers = str(evidence.get('detected_marker_ids') or 'unavailable')
        photos_html.append(f'<figure><figcaption>{escape(frame["image_file"])}<br>검출 마커 ID: {escape(markers)}</figcaption><a href="{name}"><img src="{name}" alt="{escape(leaflet)} 부분 면 재투영과 실제 관찰점"></a></figure>')
    write_json(folder/'patch.json', patch)
    with (folder/'visible_patch_mm.ply').open('w', encoding='ascii') as stream:
        stream.write('ply\nformat ascii 1.0\ncomment unvalidated partial interpolated surface; board coordinates mm\n')
        stream.write(f"element vertex {len(patch['vertices_mm'])}\nproperty float x\nproperty float y\nproperty float z\n")
        stream.write(f"element face {len(patch['faces'])}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in patch['vertices_mm']:
            stream.write(' '.join(map(str, v))+'\n')
        for face in patch['faces']:
            stream.write('3 '+' '.join(map(str, face))+'\n')
    payload = json.dumps(patch, ensure_ascii=True).replace('<', '\\u003c')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    (folder/'index.html').write_text(template.replace('__PATCH__', payload).replace('__PHOTOS__',''.join(photos_html)), encoding='utf-8')
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'depths', 'annotations', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--leaflet', required=True)
    parser.add_argument('--subdivision-levels', type=int, default=4)
    parser.add_argument('--mask-margin-px', type=int, default=1)
    a = parser.parse_args()
    result = build_patch(a.capture, a.depths, a.annotations, a.leaflet, a.out, a.subdivision_levels, a.mask_margin_px)
    print(json.dumps({k: result[k] for k in ('leaflet_id', 'retained_faces', 'validated_metric')}))


if __name__ == '__main__':
    main()
