"""Create an explicitly unvalidated 3D preview from cross-view manual landmarks.

python -m tools.manual_leaf_preview --capture build/.../capture.preview.json
  --annotations build/.../manual_candidates.json --out build/.../preview
This never changes calibration, pose gates, training records or light-pipeline inputs.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay, QhullError

from tomato3d.config import get_settings
from tomato3d.io.schemas import BoardSpec
from tomato3d.calib.board import build_charuco_board, sheet_to_board_mm, board_rectangle_board_mm
from tools.leaf_light.common import StudyError, file_hash, output_dir, read_json, write_json, write_rows
from tools.leaf_light.reconstruction import project, triangulate, inside
from tools.leaf_depth_records import compile_records, board_evidence_fields


def diagnostic_point(observations: list, intrinsics: dict, quality: dict) -> tuple[np.ndarray, dict]:
    """Retain a finite estimate but report the original residual gate separately."""
    if len({f['image_file'] for f, _ in observations}) != len(observations):
        raise StudyError('duplicate_view_for_landmark')
    for frame, xy in observations:
        if frame.get('pose_status') != 'solved' or frame.get('rvec_rad') is None:
            raise StudyError('no_measured_pose')
        if not np.isfinite(xy).all():
            raise StudyError('nonfinite_annotation')
        if not (0 <= xy[0] < intrinsics['image_width_px'] and 0 <= xy[1] < intrinsics['image_height_px']):
            raise StudyError('annotation_outside_original_image')
    policy = dict(quality, max_point_error_px=float('inf'))
    point_mm, rms_px = triangulate(observations, intrinsics, policy)
    residuals = [float(np.linalg.norm(project(point_mm[None], f, intrinsics)[0] - xy)) for f, xy in observations]
    limit = quality['max_point_error_px']
    return point_mm, dict(rms_px=rms_px, max_error_px=max(residuals), residuals_px=residuals,
        passes_point_gate=limit is None or max(residuals) <= limit,
        passes_all_pose_gates=all(f.get('passes_pose_gate') is True for f, _ in observations),
        validated_metric=False)


def supported_faces(xyz_mm: np.ndarray, observations: list, intrinsics: dict,
                    contour_tolerance_px: float) -> tuple[list, list]:
    """Piecewise linear patches; every triangle is checked in each annotated view."""
    if np.linalg.matrix_rank(xyz_mm - xyz_mm.mean(axis=0)) < 2:
        raise StudyError('degenerate_leaf_patch')
    _, _, vt = np.linalg.svd(xyz_mm - xyz_mm.mean(axis=0))
    uv = (xyz_mm - xyz_mm.mean(axis=0)) @ vt[:2].T
    triangles, support = [], []
    for ids in Delaunay(uv).simplices:
        samples_uv = np.vstack((uv[ids], uv[ids].mean(axis=0), (uv[ids] + np.roll(uv[ids], 1, axis=0)) / 2))
        if not inside(uv, samples_uv, 1e-6):
            continue
        samples_mm = np.vstack((xyz_mm[ids], xyz_mm[ids].mean(axis=0), (xyz_mm[ids] + np.roll(xyz_mm[ids], 1, axis=0)) / 2))
        triangles.append(ids.tolist())
        support.append(all(inside(poly, project(samples_mm, frame, intrinsics), contour_tolerance_px)
                           for frame, poly in observations))
    return triangles, support


def annotation_polygon(entry: dict, intrinsics: dict) -> np.ndarray:
    """Map visible 2D outlines to original pixels without making correspondences."""
    label_size = np.asarray(entry['label_image_size_px'], dtype=float)
    polygon = np.asarray(entry['boundary_xy_px'], dtype=float)
    if label_size.shape != (2,) or not np.isfinite(label_size).all() or np.any(label_size <= 0):
        raise StudyError('Invalid annotation image size')
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise StudyError('Invalid annotation polygon')
    if np.any(polygon < 0) or np.any(polygon >= label_size):
        raise StudyError('annotation_outside_original_image')
    return polygon * (np.array([intrinsics['image_width_px'], intrinsics['image_height_px']]) / label_size)


def board_reference(capture: dict) -> dict:
    """Keep the displayed board in the core pipeline's single coordinate frame."""
    path = get_settings().board_spec_path
    spec = BoardSpec.model_validate_json(path.read_text(encoding='utf-8'))
    if capture['board_variant_id'] != spec.variant_id:
        raise StudyError('Board variant does not match configured board')
    if capture.get('board_spec_sha256') and capture['board_spec_sha256'] != file_hash(path):
        raise StudyError('Configured board changed after evidence capture')
    board = build_charuco_board(spec)
    corners_mm = sheet_to_board_mm(board.getChessboardCorners(), spec)
    return dict(board_variant_id=spec.variant_id, board_spec_sha256=file_hash(path),
        square_size_mm=spec.square_size_mm, physical_print_verified=spec.physical_print_verified,
        rectangle_mm=board_rectangle_board_mm(spec).tolist(),
        inner_corners_mm=corners_mm.tolist(),
        marker_reference=[dict(marker_id=int(i), vertices_mm=sheet_to_board_mm(p, spec).tolist())
                          for i, p in zip(board.getIds(), board.getObjPoints())],
        notice='Configured board reference, not per-photo detected marker IDs. Estimated mm, not validated measurements.')


def build(capture_path: Path, annotations_path: Path, out: Path) -> dict:
    capture, annotation = read_json(capture_path), read_json(annotations_path)
    if capture.get('schema_version') != 'diagnostic-preview-capture-1':
        raise StudyError('Use an explicitly diagnostic capture, not an operational capture')
    if annotation.get('schema_version') != 'manual-leaf-preview-1':
        raise StudyError('Unknown annotation schema')
    frames = {f['image_file']: f for f in capture['frames']}
    if len(frames) != len(capture['frames']):
        raise StudyError('Duplicate capture frame')
    for frame in frames.values():
        if file_hash(Path(frame['image_path'])) != frame['sha256']:
            raise StudyError('Source photograph changed')
    board = board_reference(capture)
    folder = output_dir(out)
    intr = capture['intrinsics']; quality = capture['quality']
    geometry, diagnostics, points_csv, contours_csv, fit_rows = [], [], [], [], []
    partial_contours_csv = []
    overlays = defaultdict(list); ids_seen = set()
    visible_fragments = []; fragment_overlays = defaultdict(list); fragment_rows = []
    for leaf in annotation.get('two_d_only_candidates', []):
        entries = leaf.get('observations', [leaf])
        names = [entry['image_file'] for entry in entries]
        if (leaf['leaflet_id'] in ids_seen or not names or len(names) != len(set(names))
                or any(name not in frames for name in names)):
            raise StudyError('Duplicate leaflet ID or unknown fragment image')
        ids_seen.add(leaf['leaflet_id'])
        item = dict(leaflet_id=leaf['leaflet_id'], organ_id=leaf['organ_id'], label=leaf['description'],
                    source_images=names, review_status=leaf['review_status'],
                    geometry_status='2d_only', depth_status='unavailable',
                    surface_extent='visible_fragment_only', cross_view_identity='unconfirmed')
        visible_fragments.append(item)
        for entry in entries:
            polygon = annotation_polygon(entry, intr)
            fragment_overlays[entry['image_file']].append((leaf['leaflet_id'], polygon))
            for index, xy in enumerate(polygon):
                fragment_rows.append(dict(image_file=entry['image_file'], organ_id=leaf['organ_id'],
                    leaflet_id=leaf['leaflet_id'], vertex_index=index, x_px=float(xy[0]), y_px=float(xy[1]),
                    geometry_status='2d_only', depth_status='unavailable', correspondence_status='not_a_landmark',
                    surface_extent='visible_fragment_only', review_status=leaf['review_status']))
    for leaf in annotation['leaflets']:
        leaflet_id = leaf['leaflet_id']
        if leaflet_id in ids_seen:
            raise StudyError('Duplicate leaflet ID')
        ids_seen.add(leaflet_id)
        views = []
        for entry in leaf['observations']:
            if entry['image_file'] not in frames:
                raise StudyError('Unknown source image')
            frame = frames[entry['image_file']]
            polygon = annotation_polygon(entry, intr)
            views.append((frame, polygon))
            for index, xy in enumerate(polygon):
                common = dict(image_file=frame['image_file'], organ_id=leaf['organ_id'], leaflet_id=leaflet_id,
                              x_px=float(xy[0]), y_px=float(xy[1]))
                points_csv.append(dict(common, point_id=f'b{index}', boundary_index=index, boundary_count=len(polygon)))
                contour_rows = partial_contours_csv if leaf.get('outline_completeness') == 'partial_observed_patch' else contours_csv
                contour_rows.append(dict(common, vertex_index=index))
        diag = dict(leaflet_id=leaflet_id, status='unavailable', validated_metric=False,
                    review_status=leaf['review_status'])
        try:
            counts = {len(poly) for _, poly in views}
            if len(counts) != 1 or min(counts) < 3:
                raise StudyError('inconsistent_boundary_landmarks')
            xyz, records = [], []
            for i in range(next(iter(counts))):
                point, record = diagnostic_point([(f, poly[i]) for f, poly in views], intr, quality)
                xyz.append(point); records.append(record)
                fit_rows.append(dict(leaflet_id=leaflet_id, point_id=f'b{i}', x_board_mm=point[0],
                    y_board_mm=point[1], z_board_mm=point[2], rms_px=record['rms_px'],
                    max_error_px=record['max_error_px'], passes_point_gate=record['passes_point_gate'],
                    passes_pose_gates=record['passes_all_pose_gates'], validated_metric=False))
            xyz = np.asarray(xyz)
            faces, support = supported_faces(xyz, views, intr, quality['contour_tolerance_px'])
            if not faces:
                raise StudyError('no_non_degenerate_leaf_patch')
            item = dict(leaflet_id=leaflet_id, organ_id=leaf['organ_id'], label=leaf['description'],
                vertices_mm=xyz.tolist(), faces=faces, contour_supported=support,
                review_status=leaf['review_status'], validated_metric=False,
                review_notes=leaf.get('review_notes', []),
                correspondence_status=leaf.get('correspondence_status', 'candidate_unreviewed'),
                outline_completeness=leaf.get('outline_completeness', 'coarse_candidate_boundary'),
                surface_interpretation=leaf.get('surface_interpretation', 'Coarse unvalidated patch; not complete leaf curvature.'),
                boundary_landmark_names=leaf.get('boundary_landmark_names', []),
                point_identity_revision_note=leaf.get('point_identity_revision_note'),
                source_images=[f['image_file'] for f, _ in views],
                max_landmark_error_px=max(r['max_error_px'] for r in records))
            geometry.append(item)
            diag.update(status='experimental_patch', max_landmark_error_px=item['max_landmark_error_px'],
                landmarks=len(xyz), triangles=len(faces), contour_supported_triangles=sum(support),
                points_passing_residual_gate=sum(r['passes_point_gate'] for r in records))
            for frame, poly in views:
                overlays[frame['image_file']].append((leaflet_id, poly, project(xyz, frame, intr)))
        except (ValueError, np.linalg.LinAlgError, QhullError) as exc:
            diag['reason'] = str(exc)
        diagnostics.append(diag)
    annotated_count = len(set(overlays) | set(fragment_overlays))
    photos = []
    for name, frame in frames.items():
        items = overlays.get(name, [])
        focus_points = []
        with Image.open(frames[name]['image_path']) as source:
            full_image = source.convert('RGB')
            im = full_image.copy(); im.thumbnail((1200, 1200))
        factor = im.width / intr['image_width_px']; draw = ImageDraw.Draw(im)
        if frame['pose_status'] == 'solved':
            for item in geometry:
                if name in item['source_images']:
                    continue
                fitted = project(np.asarray(item['vertices_mm']), frame, intr) * factor
                focus_points.extend(fitted.tolist())
                draw.line([tuple(xy) for xy in np.vstack((fitted, fitted[0]))], fill='#53a6ed', width=2)
                centroid = fitted.mean(axis=0)
                draw.text(tuple(centroid), item['leaflet_id']+' projected', fill='#53a6ed', font_size=14)
        for leaflet_id, observed, fitted in items:
            focus_points.extend((observed * factor).tolist())
            draw.line([tuple(xy * factor) for xy in np.vstack((observed, observed[0]))], fill='#ffe35a', width=2)
            for i, (a, b) in enumerate(zip(observed * factor, fitted * factor)):
                draw.line([tuple(a), tuple(b)], fill='#ff657b', width=2)
                draw.ellipse((a[0]-3, a[1]-3, a[0]+3, a[1]+3), fill='#ffe35a')
                draw.ellipse((b[0]-2, b[1]-2, b[0]+2, b[1]+2), fill='#33e5ff')
                draw.text((a[0]+4, a[1]+3), str(i), fill='#ffffaa', font_size=12)
            centroid = observed.mean(axis=0) * factor
            draw.text((centroid[0], centroid[1]), leaflet_id, fill='#ffffaa', font_size=18,
                      stroke_width=1, stroke_fill='#10231a')
        for leaflet_id, polygon in fragment_overlays.get(name, []):
            visible = polygon * factor
            focus_points.extend(visible.tolist())
            draw.line([tuple(xy) for xy in np.vstack((visible, visible[0]))], fill='#ff95db', width=2)
            anchor = visible[0]
            draw.text((anchor[0], anchor[1]-18), leaflet_id, fill='#ffb5e7', font_size=15,
                      stroke_width=1, stroke_fill='#10231a')
        im.save(folder / name.replace('.png', '_review.jpg'), quality=92)
        evidence_fields = board_evidence_fields(frame)
        board_data_url = None
        if frame.get('board_evidence'):
            board_image = im.copy(); board_draw = ImageDraw.Draw(board_image)
            for marker in frame['board_evidence']['markers']:
                xy = np.asarray(marker['pixels_xy_px']) * factor
                board_draw.line([tuple(p) for p in np.vstack((xy,xy[0]))],fill='#55ff8b',width=2)
                board_draw.text(tuple(xy.mean(axis=0)),f"M{marker['marker_id']}",fill='#55ff8b',font_size=18,
                                stroke_width=2,stroke_fill='#10231a')
            for corner in frame['board_evidence']['corners']:
                a = np.asarray(corner['pixel_xy_px']) * factor
                board_draw.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3),fill='#ffb44c')
                board_draw.text((a[0]+4,a[1]+4),f"C{corner['corner_id']}",fill='#ffb44c',font_size=15,
                                stroke_width=1,stroke_fill='#10231a')
            board_image.save(folder / name.replace('.png','_board_labels.jpg'),quality=92)
            board_buf = BytesIO(); board_image.save(board_buf,format='JPEG',quality=88)
            board_data_url = 'data:image/jpeg;base64,'+base64.b64encode(board_buf.getvalue()).decode()
        buf = BytesIO(); im.save(buf, format='JPEG', quality=88)
        focus = im
        if focus_points:
            pts = np.asarray(focus_points)
            pts = pts[(pts[:,0]>=0)&(pts[:,0]<im.width)&(pts[:,1]>=0)&(pts[:,1]<im.height)]
            if len(pts):
                margin_px = .08 * max(im.size)
                low = np.maximum(0, pts.min(axis=0)-margin_px).astype(int)
                high = np.minimum(im.size, pts.max(axis=0)+margin_px).astype(int)
                focus = im.crop((*low,*high))
        closeup = BytesIO(); focus.save(closeup,format='JPEG',quality=92)
        focus.save(folder / name.replace('.png','_labels_detail.jpg'),quality=92)
        leaf_details = {}
        for leaflet_id, polygon, fitted in (items + [(ident, poly, None) for ident, poly
                                                       in fragment_overlays.get(name, [])]):
            margin_px = max(60., float(np.ptp(polygon, axis=0).max()) * .35)
            low = np.maximum(0, polygon.min(axis=0)-margin_px).astype(int)
            high = np.minimum(full_image.size, polygon.max(axis=0)+margin_px).astype(int)
            detail = full_image.crop((*low, *high))
            detail_scale = min(600 / detail.width, 600 / detail.height)
            detail = detail.resize((round(detail.width*detail_scale), round(detail.height*detail_scale)))
            detail_draw = ImageDraw.Draw(detail)
            xy = (polygon-low)*detail_scale
            color = '#ff95db' if fitted is None else '#ffe35a'
            detail_draw.line([tuple(p) for p in np.vstack((xy, xy[0]))], fill=color, width=2)
            if fitted is not None:
                for i, (a, b) in enumerate(zip(xy, (fitted-low)*detail_scale)):
                    detail_draw.line([tuple(a),tuple(b)],fill='#ff657b',width=2)
                    detail_draw.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3),fill=color)
                    detail_draw.ellipse((b[0]-2,b[1]-2,b[0]+2,b[1]+2),fill='#33e5ff')
                    detail_draw.text((a[0]+4,a[1]+3),str(i),fill=color,font_size=15,stroke_width=1,stroke_fill='#10231a')
            detail_draw.text((8,8),leaflet_id,fill=color,font_size=18,stroke_width=1,stroke_fill='#10231a')
            detail_buf = BytesIO(); detail.save(detail_buf, format='JPEG', quality=94)
            leaf_details[leaflet_id] = 'data:image/jpeg;base64,'+base64.b64encode(detail_buf.getvalue()).decode()
        photos.append(dict(name=name, annotation_kind='manual_candidates_and_projections' if items or fragment_overlays.get(name) else
            'projections_only' if frame['pose_status']=='solved' else 'no_pose_no_labels',
            leaf_details=leaf_details,
            board_data_url=board_data_url, board_evidence=evidence_fields,
            pose_status=frame['pose_status'], passes_pose_gate=frame.get('passes_pose_gate'),
            detail_data_url='data:image/jpeg;base64,'+base64.b64encode(closeup.getvalue()).decode(),
            data_url='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()))
    summary = dict(schema_version='manual-leaf-preview-result-1', status='experimental_partial_model',
        validated_metric=False, suitable_for_ppfd=False, deep_learning_run=False,
        available_photos=len(frames), computed_poses=sum(f['pose_status']=='solved' for f in frames.values()),
        accepted_poses=sum(f['passes_pose_gate'] for f in frames.values()),
        annotated_photos=annotated_count, modeled_leaflets=len(geometry), diagnostics=diagnostics,
        two_d_only_candidates=visible_fragments, board_reference=board,
        unresolved_observations=annotation.get('unresolved_observations', []),
        coordinate_frame='board; origin inner corner 0; +Z away from board; estimated mm',
        capture_sha256=file_hash(capture_path), annotations_sha256=file_hash(annotations_path),
        geometry_source='manual_multiview_landmark_interpolation',
        notice='Partial low-resolution leaf patches, not the whole plant. Assistant labels require human review. Failed residuals are visible. Hidden leaves, petioles, stems and curvature are not invented.')
    preview = dict(summary,geometry=geometry)
    depth_records = compile_records(dict(capture,plant_id=capture.get('plant_id'),session_id=capture.get('session_id')),
                                    preview, points_csv, dict(capture_sha256=file_hash(capture_path)))
    write_json(folder/'preview.json',preview)
    from tools.leaf_light.annotations import POINT_COLUMNS, CONTOUR_COLUMNS
    write_rows(folder/'points.candidate.csv',points_csv,POINT_COLUMNS)
    write_rows(folder/'contours.candidate.csv',contours_csv,CONTOUR_COLUMNS)
    write_rows(folder/'partial_patch_boundaries.candidate.csv',partial_contours_csv,CONTOUR_COLUMNS)
    depth_columns = tuple(depth_records['observations'][0]) if depth_records['observations'] else ('status',)
    write_rows(folder/'leaf_depth_observations.csv',depth_records['observations'],depth_columns)
    write_rows(folder/'visible_fragments.candidate.csv',fragment_rows,('image_file','organ_id','leaflet_id',
        'vertex_index','x_px','y_px','geometry_status','depth_status','correspondence_status',
        'surface_extent','review_status'))
    write_rows(folder/'landmark_fit.csv',fit_rows,('leaflet_id','point_id','x_board_mm','y_board_mm','z_board_mm',
        'rms_px','max_error_px','passes_point_gate','passes_pose_gates','validated_metric'))
    vertices = [v for item in geometry for v in item['vertices_mm']]
    faces = []; offset = 0
    for item in geometry:
        faces.extend([[i+offset for i in face] for face in item['faces']]); offset += len(item['vertices_mm'])
    with (folder/'leaf_preview_mm.ply').open('w',encoding='ascii') as stream:
        stream.write('ply\nformat ascii 1.0\ncomment UNVALIDATED PARTIAL PREVIEW; estimated mm; not for PPFD\n')
        stream.write(f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for xyz in vertices: stream.write(' '.join(map(str,xyz))+'\n')
        for face in faces: stream.write('3 '+' '.join(map(str,face))+'\n')
    payload = json.dumps(dict(summary=summary,geometry=geometry,photos=photos,
                             depth_observations=depth_records['observations']),ensure_ascii=False).replace('<','\\u003c')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    (folder/'index.html').write_text(template.replace('__PAYLOAD__',payload),encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture','annotations','out'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    result = build(args.capture,args.annotations,args.out)
    print(json.dumps({k:result[k] for k in ('status','modeled_leaflets','annotated_photos','validated_metric')}))


if __name__ == '__main__': main()
