"""Read-only photo navigation; board evidence never establishes leaf identity."""
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from tools.leaf_preprocess_lab import data_url, sha


def order_key(name):
    return tuple(int(p) if p.isdigit() else p for p in re.split(r'(\d+)', name.lower()))


def board_evidence(frame, image_hash):
    evidence = frame.get('board_evidence') or {}
    if not image_hash or evidence.get('source_image_sha256') != image_hash:
        return [], []
    ids = sorted(set(evidence.get('detected_marker_ids') or []))
    markers = [m for m in evidence.get('markers', []) if m.get('marker_id') in ids]
    return ids, markers


def view_guide(workflow, sid):
    workflow.verify()
    samples = workflow.lab.samples
    reference = samples[sid]
    frame_a = workflow.cams.frames.get(reference['image_file'].lower(), {})
    ref_ids, _ = board_evidence(frame_a, reference.get('source_image_sha256'))
    photos = {}
    for key, obs in samples.items():
        if (obs.get('plant_id'), obs.get('session_id')) != (reference.get('plant_id'), reference.get('session_id')):
            continue
        photos.setdefault(obs['image_file'], []).append(key)
    # Include unavailable capture views explicitly, without manufacturing annotations.
    names = set(photos) | {f['image_file'] for f in workflow.context['capture']['frames']}
    ordered = sorted(names, key=order_key)
    rows = []
    for name in ordered:
        keys = photos.get(name, [])
        frame = workflow.cams.frames.get(name.lower(), {})
        image_hash = samples[keys[0]].get('source_image_sha256') if keys else frame.get('sha256')
        ids, _ = board_evidence(frame, image_hash)
        angle = None
        ka, kb = reference['image_file'].lower(), name.lower()
        if workflow.cams.posed(ka) and workflow.cams.posed(kb):
            a, b = workflow.cams.centre(ka), workflow.cams.centre(kb)
            if np.isfinite(a).all() and np.isfinite(b).all() and np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                angle = float(np.degrees(np.arccos(np.clip(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b)), -1, 1))))
        rows.append(dict(image_file=name, sample_ids=keys, view_code=frame.get('view_code'),
            view_group=frame.get('observed_view_group') or (samples[keys[0]].get('view_group') if keys else None),
            marker_ids=ids, shared_marker_ids=sorted(set(ref_ids) & set(ids)),
            board_origin_view_angle_deg=angle, pose_solved=workflow.cams.posed(name.lower()),
            poses_pass_gate=frame_a.get('passes_pose_gate') is True and frame.get('passes_pose_gate') is True,
            is_reference=name == reference['image_file']))
    return dict(photos=rows, reference_image=reference['image_file'],
        angle_semantics='Angle between measured camera-centre directions from board origin; not commanded turntable rotation',
        order_semantics='Capture filename order only; no angle inferred from view number',
        validated_metric=False, suitable_for_ppfd=False)


def photo_preview(workflow, sid):
    workflow.verify()
    obs = workflow.lab.samples[sid]
    path = Path(obs['source_image'])
    if sha(path) != obs['source_image_sha256']:
        raise ValueError('Source photograph changed')
    with Image.open(path) as im:
        rgb = np.array(im.convert('RGB'))
    height, width = rgb.shape[:2]
    scale = min(1.0, 900 / max(width, height))
    rgb = cv2.resize(rgb, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
    def points(values):
        return np.rint(np.asarray(values)*scale).astype(np.int32)
    for key, other in workflow.lab.samples.items():
        if (other['image_file'],other.get('plant_id'),other.get('session_id')) != (obs['image_file'],obs.get('plant_id'),obs.get('session_id')):
            continue
        x0,y0,x1,y1 = points(other['crop_xyxy_px'])
        color = (255,220,50) if key == sid else (150,100,210)
        cv2.rectangle(rgb,(x0,y0),(x1,y1),color,3 if key == sid else 1)
        cv2.putText(rgb,str(other['source_annotation_id']),(x0,max(12,y0)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    ids, markers = board_evidence(workflow.cams.frames.get(obs['image_file'].lower(),{}), obs['source_image_sha256'])
    for marker in markers:
        xy = np.asarray(marker.get('pixels_xy_px',[]),float)
        if xy.shape != (4,2) or not np.isfinite(xy).all(): continue
        polygon = points(xy)
        cv2.polylines(rgb,[polygon],True,(30,230,240),2)
        cv2.putText(rgb,'M'+str(marker['marker_id']),tuple(polygon[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,230,240),1,cv2.LINE_AA)
    return dict(image=data_url(rgb),sample_id=sid,image_file=obs['image_file'],marker_ids=ids,
                scale=scale,original_size_px=[width,height],boxes_are='display_crop_regions_not_leaf_surfaces')
