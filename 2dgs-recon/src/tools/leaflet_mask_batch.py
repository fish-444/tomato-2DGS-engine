"""Persistent, individually reviewed masks for existing photo annotation boxes.

python -m tools.leaflet_mask_batch prepare --plant dwt_01 --labels ... --session ... --out ...
python -m tools.leaflet_mask_batch run --batch ... --segmenter-config ... [--pilot]
python -m tools.leaflet_mask_batch status --batch ...

No identity, landmark, 3D or training operation is performed. Source inputs are read-only.
The raster is authoritative: contour simplification must not delete small regions or holes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from tools import leaflet_segmenter as seg
from tools.leaf_light import common

DEFAULT_SENSITIVITY = 20
BASELINE_SENSITIVITY = 50
THRESHOLD_SCALE = 0.08
SCHEMA = 'leaflet-mask-batch-1'
#: mapping_report schema written by tools.plant_auto for model-drafted labels in original pixels.
AUTO_LABELS_SCHEMA = 'plant-auto-labels-1'
REPLACE_RETRY_DELAYS_SECONDS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
MUTABLE = ('generation', 'review', 'prompts', 'cache_key', 'selected_candidate', 'sensitivity',
           'manual_rings', 'changed', 'error', 'generation_seconds', 'manual_raster', 'reject_reason', 'visibility')
#: Why a person rejected an observation (user's four reasons, 2026-09-27). 'occluded' is not a rejection: an occluded
#: leaflet is accepted with visibility 'partial' (its visible part is still a correct mask).
REJECT_REASONS = ('duplicate_box', 'box_misses_leaf', 'false_positive')
VISIBILITY = ('full', 'partial')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    common.write_json(temporary, value)
    # OneDrive/antivirus can briefly open the destination without delete sharing.
    # Keep the old file intact and retry the atomic replace, never truncate it.
    for delay in REPLACE_RETRY_DELAYS_SECONDS:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(temporary, path)


def threshold(sensitivity: float) -> float:
    if not np.isfinite(sensitivity) or not 0 <= sensitivity <= 100:
        raise common.StudyError('Sensitivity must be 0..100')
    return (BASELINE_SENSITIVITY - sensitivity) * THRESHOLD_SCALE


def pack_mask(mask: np.ndarray) -> dict:
    """Lossless foreground intervals, including single pixels and holes."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask.ravel(), False].astype(np.int8)))
    return dict(shape=list(mask.shape), intervals=edges.reshape(-1, 2).tolist())


def unpack_mask(data: dict, shape: tuple) -> np.ndarray:
    if data['shape'] != list(shape):
        raise common.StudyError('Edited raster shape mismatch')
    result = np.zeros(int(np.prod(shape)), bool)
    previous = 0
    for start, stop in data['intervals']:
        if type(start) is not int or type(stop) is not int or not previous <= start < stop <= len(result):
            raise common.StudyError('Invalid edited raster intervals')
        result[start:stop] = True
        previous = stop
    return result.reshape(shape)


def brush_mask(mask: np.ndarray, crop: list, strokes: list) -> np.ndarray:
    """Rasterize original-pixel strokes; never smooth or remove other components."""
    if not isinstance(strokes, list) or not 1 <= len(strokes) <= 500:
        raise common.StudyError('Expected 1..500 brush strokes')
    result = mask.astype(np.uint8).copy()
    total = 0
    for stroke in strokes:
        if not isinstance(stroke, dict):
            raise common.StudyError('Invalid brush stroke')
        mode, radius, points = stroke.get('mode'), stroke.get('radius_px'), stroke.get('points_px')
        if mode not in ('paint', 'erase') or not isinstance(radius, (float, int)) or not np.isfinite(radius) or not 1 <= radius <= 200:
            raise common.StudyError('Invalid brush mode/radius')
        if not isinstance(points, list):
            raise common.StudyError('Invalid brush points')
        xy = np.asarray(points, dtype=float)
        total += len(xy)
        if xy.ndim != 2 or xy.shape[1] != 2 or not 1 <= len(xy) or total > 50000 or not np.isfinite(xy).all():
            raise common.StudyError('Invalid brush points')
        if np.any(xy < np.asarray(crop[:2])) or np.any(xy > np.asarray(crop[2:])-1):
            raise common.StudyError('Brush points must be inside the original-pixel crop')
        local = np.floor(xy-np.asarray(crop[:2])+.5).astype(np.int32)
        value, rad = int(mode == 'paint'), max(1, int(np.floor(radius+.5)))
        for a, b in zip(local, local[1:]):
            cv2.line(result, tuple(a), tuple(b), value, rad*2+1, cv2.LINE_8)
        for point in (local[0], local[-1]):
            cv2.circle(result, tuple(point), rad, value, -1, cv2.LINE_8)
    return result.astype(bool)


def organ_class(value: str) -> str:
    key = str(value).lower().replace('_', '').replace('-', '').replace(' ', '')
    if key in ('compoundleaf', 'compoundleaves', 'compondleaf', 'compondleaves'):
        return 'compoundleaves'
    return key


def prepare(plant: str, labels: Path, session: Path, out: Path,
            selected_class: str = 'leaflet') -> dict:
    """Reuse verified original-pixel labels; fail each invalid observation explicitly."""
    # Local SAM mask generation is authorised per plant by the user: dwt_01/02 (2026-09-22), dwt_03/04 (2026-09-22, later).
    # Photo-only auto mode (user, 2026-09-28) authorises any plant whose labels come from tools.plant_auto: the labels
    # are model drafts already in original pixels, marked as such, and every mask they yield stays marked 'auto'.
    labels_report = common.read_json(labels.resolve()/'mapping_report.json')
    auto = labels_report.get('schema_version') == AUTO_LABELS_SCHEMA
    if plant not in ('dwt_01', 'dwt_02', 'dwt_03', 'dwt_04') and not auto:
        raise common.StudyError('This batch is limited to the user-authorised plants dwt_01 to dwt_04 or photo-only auto labels')
    selected_class = organ_class(selected_class)
    if selected_class not in ('leaflet', 'compoundleaves'):
        raise common.StudyError('Choose leaflet or compoundleaves')
    initial_sensitivity = BASELINE_SENSITIVITY if selected_class == 'compoundleaves' else DEFAULT_SENSITIVITY
    labels, session = labels.resolve(), session.resolve()
    coco_path, report_path = labels/'original_pixel_labels.coco.json', labels/'mapping_report.json'
    coco, mapping = common.read_json(coco_path), common.read_json(report_path)
    capture = common.read_json(session/'capture.json')
    if mapping['plant_id'] != plant or Path(mapping['session']).resolve() != session:
        raise common.StudyError('Mapping belongs to another plant/session')
    verified_inputs = common.read_json(labels/'input_hashes.json')
    changed_inputs = [name for name, sha in verified_inputs.items()
                      if not Path(name).is_file() or common.file_hash(Path(name)) != sha]
    # A stale mapping is never silently reused for generation.
    frames = {Path(f['image_file']).name: f for f in capture['frames']}
    checks = {p['prepared_frame']: p for p in mapping['photos']}
    photos = {p['id']: p for p in coco['images']}
    categories = {c['id']: c['name'] for c in coco['categories']}
    out = common.output_dir(out)
    (out/'items').mkdir(); (out/'cache').mkdir(); (out/'exports').mkdir()
    ids, pilot, seen_groups = [], [], set()
    photo_checks = {}
    for image_id, image in photos.items():
        name, error = image['file_name'], None
        frame, check = frames.get(name), checks.get(name)
        try:
            if changed_inputs:
                raise common.StudyError('Mapping inputs changed: ' + ', '.join(changed_inputs))
            # auto labels are drawn on the original photo itself: there is no export image to align
            if not frame or not check or not check.get('mapped') or not (auto or check.get('alignment', {}).get('accepted')):
                raise common.StudyError('Photo transform is not verified')
            path = (session/frame['image_path']).resolve()
            if not path.is_relative_to(session) or common.file_hash(path) != frame['sha256']:
                raise common.StudyError('Source image hash/path differs')
            with Image.open(path) as picture:
                if picture.size != (image['width'], image['height']) or picture.getexif().get(274, 1) != 1:
                    raise common.StudyError('Original image size/orientation differs')
            transform = image.get('extra', {}).get('transform', {})
            if transform.get('type') not in (('uncropped_stretch', 'original_pixels') if auto else ('uncropped_stretch',))                     or transform.get('original_size_px') != [image['width'], image['height']]:
                raise common.StudyError('Missing verified original-pixel transform')
        except (OSError, ValueError, KeyError) as exc:
            error = str(exc)
        photo_checks[image_id] = dict(frame=frame, error=error)
        group = image.get('extra', {}).get('view_group')
        if group in ('middle', 'side') and group not in seen_groups:
            seen_groups.add(group); pilot.append(name)
    for annotation in coco['annotations']:
        if organ_class(categories[annotation['category_id']]) != selected_class:
            continue
        image = photos[annotation['image_id']]
        checked = photo_checks[annotation['image_id']]
        identity = f"{plant}_a{annotation['id']}"
        if identity in ids:
            raise common.StudyError('Duplicate source annotation ID')
        ids.append(identity)
        x, y, w, h = annotation['bbox']
        box = [x, y, x+w, y+h]
        error = checked['error']
        try:
            seg.checked_prompts({'box_px': box}, image['width'], image['height'])
        except ValueError as exc:
            error = str(exc)
        frame = checked['frame'] or {}
        item = dict(id=identity, plant_id=plant, organ_class=selected_class,
                    source_class_name=categories[annotation['category_id']], source_annotation_id=annotation['id'],
                    source_annotation=annotation, image_file=image['file_name'],
                    image_path=frame.get('image_path'), image_sha256=frame.get('sha256'),
                    image_size_px=[image['width'], image['height']], source_bbox_xywh_px=annotation['bbox'],
                    transform=image.get('extra', {}).get('transform'), image_provenance=image.get('extra'),
                    view_group=image.get('extra', {}).get('view_group'), mapping_error=error,
                    generation='failed' if error else 'pending', review='unreviewed', revision=0,
                    prompts=dict(box_px=box, positive_px=[], negative_px=[]), cache_key=None,
                    selected_candidate=0, sensitivity=initial_sensitivity, manual_rings=None,
                    changed=False, error=error, generation_seconds=0., review_seconds=0.,
                    edit_seconds=0., history=[], undo=[], created_at=now())
        atomic_json(out/'items'/f'{identity}.json', item)
    manifest = dict(schema_version=SCHEMA, plant_id=plant, session=str(session), ids=ids, organ_class=selected_class,
                    source_coco=str(coco_path), source_coco_sha256=common.file_hash(coco_path),
                    mapping_report_sha256=common.file_hash(report_path),
                    input_hashes={**verified_inputs, str(coco_path): common.file_hash(coco_path),
                                  str(report_path): common.file_hash(report_path)},
                    pilot_photos=pilot, photo_count=len(photos), created_at=now(),
                    coordinates='original image pixels; boxes use pixel edges',
                    defaults=dict(sensitivity=initial_sensitivity, baseline=BASELINE_SENSITIVITY),
                    identity_confirmed=False, suitable_for_ppfd=False,
                    label_source='auto' if auto else 'human')
    atomic_json(out/'manifest.json', manifest)
    return manifest


def raster_rings(mask: np.ndarray, crop: list[int]) -> list[dict]:
    rings, _ = seg.mask_rings(mask, None, [], [], keep_all=True)
    for ring in rings:
        ring['points_px'] = (np.asarray(ring['points_px']) + crop[:2]).tolist()
    return rings


def render_rings(rings: list[dict], shape: tuple[int, int], crop: list[int]) -> np.ndarray:
    """Even-odd fill preserves holes; manual edits are applied as a raster delta."""
    result = np.zeros(shape, np.uint8)
    if len(rings) > 4096:
        raise common.StudyError('Too many contours')
    for ring in rings:
        points = np.asarray(ring['points_px'], dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or not 3 <= len(points) <= 20000 or not np.isfinite(points).all():
            raise common.StudyError('Invalid contour')
        local = points - crop[:2]
        if (local < 0).any() or (local[:, 0] >= shape[1]).any() or (local[:, 1] >= shape[0]).any():
            raise common.StudyError('Contour outside the crop')
        one = np.zeros(shape, np.uint8)
        cv2.fillPoly(one, [np.rint(local).astype(np.int32)], 1)
        result ^= one
    return result.astype(bool)


def risk_signals(masks: np.ndarray, selected: int, crop: list[int], box: list[float]) -> dict:
    mask = masks[selected]
    area = int(mask.sum())
    x0, y0 = crop[:2]
    xx0, yy0 = max(0, int(np.floor(box[0]-x0))), max(0, int(np.floor(box[1]-y0)))
    xx1, yy1 = min(mask.shape[1], int(np.ceil(box[2]-x0))), min(mask.shape[0], int(np.ceil(box[3]-y0)))
    inside = int(mask[yy0:yy1, xx0:xx1].sum())
    differences = [1 - float((mask & other).sum())/max(1, int((mask | other).sum())) for other in masks]
    components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = int((hierarchy[0, :, 3] >= 0).sum()) if hierarchy is not None else 0
    disagreement, outside = max(differences, default=0.), 1-inside/max(1, area)
    return dict(area_px=area, candidate_disagreement=disagreement, outside_bbox_fraction=outside,
                components=components, holes=holes, empty=area == 0,
                crop_edge_touch=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
                priority=disagreement + outside + min(max(components-1, 0), 5)/5 + (2 if area == 0 else 0))


class Batch:
    def __init__(self, root: Path, segmenter=None, segment_lock=None):
        self.root = root.resolve()
        if not self.root.is_relative_to(common.BUILD.resolve()) or self.root == common.BUILD.resolve():
            raise common.StudyError('Batch must be under build/')
        self.manifest = common.read_json(self.root/'manifest.json')
        if self.manifest.get('schema_version') != SCHEMA:
            raise common.StudyError('Unsupported batch schema')
        self.session = Path(self.manifest['session'])
        self.segmenter = segmenter
        self.segment_lock = segment_lock or threading.Lock()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.job = dict(running=False, current=None, error=None)

    @contextmanager
    def writer(self):
        """A process-wide lease also protects review/export against another CLI."""
        with (self.root/'writer.lock').open('a+b') as lease:
            lease.seek(0, os.SEEK_END)
            if lease.tell() == 0:
                lease.write(b'0'); lease.flush()
            lease.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise common.StudyError('다른 작업에서 이 마스크 목록을 처리 중입니다. 완료 후 다시 시도하세요.') from exc
            try:
                yield
            finally:
                lease.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lease, fcntl.LOCK_UN)

    def item(self, identity: str) -> dict:
        if identity not in self.manifest['ids']:
            raise common.StudyError('Unknown observation')
        with self.lock:
            result = common.read_json(self.root/'items'/f'{identity}.json')
            result.setdefault('manual_raster', None)
            result.setdefault('reject_reason', None)
            result.setdefault('visibility', None)
            return result

    def save(self, item: dict) -> None:
        atomic_json(self.root/'items'/f"{item['id']}.json", item)

    def model_signature(self) -> dict:
        if self.segmenter is None or not self.segmenter.info().get('available'):
            raise common.StudyError('No segmentation model configured')
        info = self.segmenter.info()
        return dict(config=getattr(self.segmenter, 'config', None),
                    model={k: info.get(k) for k in ('backend', 'model_name', 'model_config', 'checkpoint_sha256', 'is_mock')},
                    crop_margin=seg.CROP_MARGIN_FRACTION, minimum_margin_px=seg.MIN_CROP_MARGIN_PX,
                    pipeline_version=SCHEMA, postprocessing=False)

    def _validate_source(self, item: dict) -> Path:
        if item['mapping_error']:
            raise common.StudyError(item['mapping_error'])
        for path in (self.manifest['source_coco'],):
            if common.file_hash(Path(path)) != self.manifest['source_coco_sha256']:
                raise common.StudyError('Source labels changed; prepare a new batch')
        path = (self.session/item['image_path']).resolve()
        if not path.is_relative_to(self.session) or common.file_hash(path) != item['image_sha256']:
            raise common.StudyError('Source image changed; prepare a new batch')
        return path

    def _cache(self, item: dict, prompts: dict) -> tuple[str, dict, float]:
        path = self._validate_source(item)
        width, height = item['image_size_px']
        box, positive, negative = seg.checked_prompts(prompts, width, height)
        if box != item['prompts']['box_px']:
            raise common.StudyError('Original bbox is immutable')
        signature = self.model_signature()
        key = common.digest(dict(image_sha256=item['image_sha256'], prompts=prompts, model=signature))
        folder = self.root/'cache'/key
        if (folder/'result.json').exists():
            meta = common.read_json(folder/'result.json')
            if common.file_hash(folder/'logits.npz') == meta['logits_sha256']:
                return key, meta, 0.
            raise common.StudyError('Cached scores changed')
        started = time.perf_counter()
        crop = list(seg.prompt_crop(box, positive, negative, width, height))
        x0, y0, x1, y1 = crop
        with Image.open(path) as picture:
            if picture.size != (width, height) or picture.getexif().get(274, 1) != 1:
                raise common.StudyError('Source dimensions/orientation changed')
            rgb = np.array(picture.convert('RGB').crop(tuple(crop)))
        local_box = [box[0]-x0, box[1]-y0, box[2]-x0, box[3]-y0]
        local = lambda points: [[x-x0, y-y0] for x, y in points]
        with self.segment_lock:
            candidates = self.segmenter.predict_logits(rgb, local_box, local(positive), local(negative))
        if not candidates:
            raise common.StudyError('No mask candidates returned')
        logits = np.stack([c.logits for c in candidates]).astype(np.float32)
        scores = [c.score for c in candidates]
        if logits.shape != (len(candidates), y1-y0, x1-x0) or not np.isfinite(logits).all():
            raise common.StudyError('Invalid crop-resolution logits')
        if not all(s is None or np.isfinite(s) for s in scores):
            raise common.StudyError('Invalid model score')
        selected = max(range(len(scores)), key=lambda i: scores[i] if scores[i] is not None else -float('inf'))
        sensitivity = self.manifest['defaults']['sensitivity']
        signals = risk_signals(logits > threshold(sensitivity), selected, crop, box)
        folder.mkdir(exist_ok=True)
        temp = folder/'logits.tmp.npz'
        np.savez_compressed(temp, logits=logits)
        os.replace(temp, folder/'logits.npz')
        elapsed = time.perf_counter()-started
        meta = dict(crop_px=crop, scores=scores, default_candidate=selected, model=self.segmenter.info(),
                    model_signature=signature, prompts=prompts, image_sha256=item['image_sha256'],
                    logits_sha256=common.file_hash(folder/'logits.npz'), risk=signals,
                    generation_seconds=elapsed, created_at=now(), sensitivity=sensitivity)
        atomic_json(folder/'result.json', meta)
        return key, meta, elapsed

    def _generate(self, identity: str) -> None:
        item = self.item(identity)
        # Resume reuses unchanged results, but a changed model requires fresh review.
        if item['generation'] == 'ready':
            meta = common.read_json(self.root/'cache'/item['cache_key']/'result.json')
            try:
                self._validate_source(item)
                if meta['model_signature'] == self.model_signature():
                    return
                previous = {k: deepcopy(item[k]) for k in MUTABLE}
                item['undo'].append(previous)
                item['history'].append(dict(action='model_changed', at=now()))
                item.update(review='unreviewed', manual_rings=None, manual_raster=None, changed=True,
                            sensitivity=self.manifest['defaults']['sensitivity'])
            except Exception as exc:
                item.update(generation='failed', review='unreviewed', error=str(exc), revision=item['revision']+1)
                self.save(item)
                return
        with self.lock:
            item['generation'] = 'running'; item['revision'] += 1; self.save(item)
        started = time.perf_counter()
        try:
            key, result, seconds = self._cache(item, item['prompts'])
            item.update(cache_key=key, selected_candidate=result['default_candidate'], generation='ready', error=None,
                        generation_seconds=item['generation_seconds']+seconds)
        except Exception as exc:
            item.update(generation='failed', error=f'{type(exc).__name__}: {exc}',
                        generation_seconds=item['generation_seconds']+time.perf_counter()-started)
        with self.lock:
            item['revision'] += 1
            item['history'].append(dict(action='generation', at=now(), state=item['generation'], error=item['error']))
            self.save(item)

    def run(self, photos: list[str] | None = None, retry_failed: bool = False,
            identity: str | None = None) -> dict:
        if identity is not None:
            self.item(identity)  # Validate before creating a writer or starting work.
        with self.writer():
            stop_file = self.root/'stop_requested.json'
            if stop_file.exists():
                stop_file.unlink()
            for identity in ([identity] if identity is not None else self.manifest['ids']):
                if self.stop_event.is_set() or stop_file.exists():
                    break
                item = self.item(identity)
                if photos is not None and item['image_file'] not in photos:
                    continue
                if item['mapping_error'] or (item['generation'] == 'failed' and not retry_failed):
                    continue
                self.job['current'] = identity
                self._generate(identity)
            self.refresh_overlaps()
        return self.status()

    def refresh_overlaps(self) -> None:
        """Rank same-photo overlap for review; never merge or discard masks."""
        groups = {}
        for identity in self.manifest['ids']:
            item = self.item(identity)
            if item['generation'] == 'ready':
                groups.setdefault(item['image_file'], []).append(item)
        for items in groups.values():
            masks = [self.mask(i) for i in items]
            maxima = [0.] * len(items)
            for a in range(len(items)):
                ma, ca = masks[a][0], masks[a][1]['crop_px']
                for b in range(a+1, len(items)):
                    mb, cb = masks[b][0], masks[b][1]['crop_px']
                    x0, y0, x1, y1 = max(ca[0], cb[0]), max(ca[1], cb[1]), min(ca[2], cb[2]), min(ca[3], cb[3])
                    if x1 <= x0 or y1 <= y0:
                        continue
                    overlap = int((ma[y0-ca[1]:y1-ca[1], x0-ca[0]:x1-ca[0]] & mb[y0-cb[1]:y1-cb[1], x0-cb[0]:x1-cb[0]]).sum())
                    fraction = overlap/max(1, min(int(ma.sum()), int(mb.sum())))
                    maxima[a] = max(maxima[a], fraction); maxima[b] = max(maxima[b], fraction)
            for item, (_, meta), fraction in zip(items, masks, maxima):
                risk = meta['risk']
                base = risk['priority'] - risk.get('neighbor_overlap_max', 0.)
                risk.update(neighbor_overlap_max=fraction, priority=base+fraction,
                            overlap_context='same-photo generated masks; review signal, not identity')
                atomic_json(self.root/'cache'/item['cache_key']/'result.json', meta)

    def start(self, pilot: bool = False, retry_failed: bool = False, identity: str | None = None) -> dict:
        with self.lock:
            if self.job['running']:
                raise common.StudyError('A batch job is already running')
            if identity is not None:
                self.item(identity)
            self.model_signature()
            self.stop_event.clear()
            self.job = dict(running=True, current=None, error=None, started_at=now())
            def work():
                try:
                    self.run(self.manifest['pilot_photos'] if pilot else None, retry_failed, identity)
                except Exception as exc:
                    self.job['error'] = f'{type(exc).__name__}: {exc}'
                finally:
                    self.job.update(running=False, current=None, finished_at=now())
                    atomic_json(self.root/'summary.json', self.status())
            self.worker = threading.Thread(target=work, daemon=True)
            self.worker.start()
            return deepcopy(self.job)

    def stop(self) -> dict:
        self.stop_event.set()
        atomic_json(self.root/'stop_requested.json', dict(requested_at=now()))
        return dict(stopping=self.status()['job']['running'], note='Stops after the current bbox; completed drafts are retained')

    def status(self) -> dict:
        items = [self.item(i) for i in self.manifest['ids']]
        reviewed = [i for i in items if i['review'] in ('accepted', 'rejected', 'deferred')]
        accepted = [i for i in items if i['review'] == 'accepted']
        job = deepcopy(self.job)
        if not job['running']:
            try:
                with self.writer():
                    pass
            except common.StudyError:
                job.update(running=True, external_process=True,
                           current=next((i['id'] for i in items if i['generation'] == 'running'), None))
        return dict(plant_id=self.manifest['plant_id'], total=len(items), photos=self.manifest['photo_count'],
                    organ_class=self.manifest.get('organ_class', 'leaflet'),
                    generation=dict(Counter(i['generation'] for i in items)), review=dict(Counter(i['review'] for i in items)),
                    generation_seconds=sum(i['generation_seconds'] for i in items),
                    review_seconds=sum(i['review_seconds'] for i in items) if reviewed else None,
                    edit_seconds=sum(i['edit_seconds'] for i in items) if reviewed else None,
                    unchanged_acceptance_fraction=sum(not i['changed'] for i in accepted)/len(accepted) if accepted else None,
                    unchanged_acceptance_denominator='currently accepted observations', job=job)

    def queue(self) -> list[dict]:
        result = []
        for identity in self.manifest['ids']:
            item = self.item(identity)
            risk = {}
            if item['cache_key']:
                risk = common.read_json(self.root/'cache'/item['cache_key']/'result.json')['risk']
            result.append({**{k: item[k] for k in ('id', 'image_file', 'view_group', 'generation', 'review', 'revision', 'error')},
                           'organ_class': self.manifest.get('organ_class', 'leaflet'), 'risk': risk})
        return sorted(result, key=lambda x: (-float(x['risk'].get('priority', 0)), x['id']))

    def mask(self, item: dict, candidate: int | None = None, sensitivity: float | None = None) -> tuple[np.ndarray, dict]:
        if item['generation'] != 'ready' or not item['cache_key']:
            raise common.StudyError('No generated mask for this observation')
        folder = self.root/'cache'/item['cache_key']
        meta = common.read_json(folder/'result.json')
        candidate = item['selected_candidate'] if candidate is None else candidate
        sensitivity = item['sensitivity'] if sensitivity is None else sensitivity
        if not isinstance(candidate, int) or not 0 <= candidate < len(meta['scores']):
            raise common.StudyError('Invalid mask candidate')
        with np.load(folder/'logits.npz', allow_pickle=False) as data:
            base = data['logits'][candidate] > threshold(float(sensitivity))
        if item['manual_rings'] is not None and candidate == item['selected_candidate'] and sensitivity == item['sensitivity']:
            original = render_rings(raster_rings(base, meta['crop_px']), base.shape, meta['crop_px'])
            changed = render_rings(item['manual_rings'], base.shape, meta['crop_px'])
            base = (base & ~(original & ~changed)) | (changed & ~original)
        if item.get('manual_raster') is not None and candidate == item['selected_candidate'] and sensitivity == item['sensitivity']:
            base = unpack_mask(item['manual_raster'], base.shape)
        return base, meta

    def detail(self, identity: str) -> dict:
        item = self.item(identity)
        detail = {k: v for k, v in item.items() if k not in ('undo',)}
        detail['organ_class'] = self.manifest.get('organ_class', 'leaflet')
        if item['generation'] == 'ready':
            mask, meta = self.mask(item)
            detail.update(result=meta, rings=item['manual_rings'] if item['manual_rings'] is not None else raster_rings(mask, meta['crop_px']),
                          area_px=int(mask.sum()))
        return detail

    def preview_logits(self, identity: str) -> bytes:
        """Read-only float32 previews: 4 uint32 LE words (N,H,W,revision), then logits.

        Browser preview uses these exact saved scores; it never infers or writes state.
        """
        item = self.item(identity)
        if item['generation'] != 'ready' or not item['cache_key']:
            raise common.StudyError('Generate this observation before previewing')
        folder = self.root/'cache'/item['cache_key']
        meta = common.read_json(folder/'result.json')
        if common.file_hash(folder/'logits.npz') != meta['logits_sha256']:
            raise common.StudyError('Cached scores changed')
        with np.load(folder/'logits.npz', allow_pickle=False) as archive:
            scores = np.asarray(archive['logits'], dtype='<f4')
        x0, y0, x1, y1 = meta['crop_px']
        if scores.shape != (len(meta['scores']), y1-y0, x1-x0) or not np.isfinite(scores).all():
            raise common.StudyError('Invalid cached score dimensions')
        header = np.asarray([*scores.shape, item['revision']], dtype='<u4')
        return header.tobytes() + scores.tobytes(order='C')

    def review(self, identity: str, request: dict) -> dict:
        with self.lock, self.writer():
            if self.job['running']:
                raise common.StudyError('Stop generation before reviewing masks')
            item = self.item(identity)
            if request.get('revision') != item['revision']:
                raise common.StudyError('Observation changed; reload before editing')
            if item['generation'] == 'running':
                raise common.StudyError('Observation is still being generated')
            action = request.get('action')
            previous = {k: deepcopy(item[k]) for k in MUTABLE}
            item['review_source'] = 'human'
            seconds = float(request.get('active_seconds', 0))
            if not np.isfinite(seconds) or not 0 <= seconds <= 1800:
                raise common.StudyError('Invalid active review time')
            if action == 'undo':
                if not item['undo']:
                    raise common.StudyError('Nothing to undo')
                restored = item['undo'].pop()
                restored.setdefault('manual_raster', None)
                item.update(restored)
            elif action in ('accept', 'defer', 'reject'):
                if action == 'accept':
                    self._validate_source(item)
                    mask, _ = self.mask(item)
                    if not mask.any():
                        raise common.StudyError('Empty masks cannot be accepted')
                item['review'] = {'accept': 'accepted', 'defer': 'deferred', 'reject': 'rejected'}[action]
                reason, visibility = request.get('reason'), request.get('visibility')
                if reason is not None and (action != 'reject' or reason not in REJECT_REASONS):
                    raise common.StudyError('A rejection reason is one of ' + ', '.join(REJECT_REASONS))
                if visibility is not None and (action != 'accept' or visibility not in VISIBILITY):
                    raise common.StudyError('Visibility is recorded only on accept: full or partial')
                item['reject_reason'] = reason if action == 'reject' else None
                item['visibility'] = visibility if action == 'accept' else None
            elif action == 'select':
                candidate = request.get('candidate', item['selected_candidate'])
                sensitivity = request.get('sensitivity', item['sensitivity'])
                self.mask(item, candidate, sensitivity)
                item.update(selected_candidate=candidate, sensitivity=sensitivity, manual_rings=None, manual_raster=None, changed=True, review='edited')
            elif action == 'contour':
                mask, meta = self.mask(item)
                rings = request['rings']
                changed = render_rings(rings, mask.shape, meta['crop_px'])
                original = render_rings(item['manual_rings'] if item['manual_rings'] is not None else raster_rings(mask, meta['crop_px']), mask.shape, meta['crop_px'])
                raster = (mask & ~(original & ~changed)) | (changed & ~original)
                item.update(manual_rings=deepcopy(rings), manual_raster=pack_mask(raster), changed=True, review='edited')
            elif action == 'brush':
                mask, meta = self.mask(item)
                raster = brush_mask(mask, meta['crop_px'], request.get('strokes'))
                item.update(manual_raster=pack_mask(raster), manual_rings=raster_rings(raster, meta['crop_px']), changed=True, review='edited')
            elif action == 'refine':
                prompts = dict(box_px=item['prompts']['box_px'], positive_px=request.get('positive_px', []), negative_px=request.get('negative_px', []))
                key, meta, elapsed = self._cache(item, prompts)
                item.update(prompts=prompts, cache_key=key, generation='ready', error=None,
                            selected_candidate=meta['default_candidate'], manual_rings=None, manual_raster=None, changed=True, review='edited',
                            generation_seconds=item['generation_seconds']+elapsed)
            else:
                raise common.StudyError('Unknown mask review action')
            if action != 'undo':
                item['undo'].append(previous)
            item['revision'] += 1
            item['review_seconds'] += seconds
            if action in ('select', 'contour', 'refine', 'brush'):
                item['edit_seconds'] += seconds
            item['history'].append(dict(action=action, at=now(), active_seconds=seconds, revision=item['revision']))
            if action == 'brush':
                item['history'][-1]['strokes'] = deepcopy(request['strokes'])
            self.save(item)
            return self.detail(identity)

    def auto_accept(self, only: set | None = None, source: str = 'auto') -> dict:
        """Photo-only auto mode: accept every generated, still unreviewed, non-empty mask as review_source='auto'.
        With `only` (2026-09-29): accept just those items, e.g. the ones the learned review model
        (tools.decision_learning) is confident about, marked review_source=`source`; allowed on any batch because the
        rest stays for a person. A person's later review replaces it and sets review_source='human'."""
        if only is None and self.manifest.get('label_source') != 'auto':
            raise common.StudyError('Auto acceptance is only for batches prepared from photo-only auto labels')
        counts = Counter()
        with self.lock, self.writer():
            if self.job['running']:
                raise common.StudyError('Stop generation before auto acceptance')
            for identity in self.manifest['ids']:
                if only is not None and identity not in only:
                    continue
                item = self.item(identity)
                if item['generation'] != 'ready' or item['review'] != 'unreviewed':
                    counts['skipped'] += 1
                    continue
                mask, _ = self.mask(item)
                if not mask.any():
                    counts['empty'] += 1
                    continue
                item.update(review='accepted', review_source=source)
                item['history'].append(dict(action='auto_accept', at=now(), active_seconds=0., revision=item['revision']))
                self.save(item)
                counts['accepted'] += 1
        return dict(counts)

    def png(self, identity: str) -> bytes:
        mask, _ = self.mask(self.item(identity))
        buffer = BytesIO()
        Image.fromarray(mask.astype(np.uint8)*255).save(buffer, format='PNG')
        return buffer.getvalue()

    def export(self) -> dict:
        with self.lock, self.writer():
            if self.job['running']:
                raise common.StudyError('Stop generation before exporting')
            destination = self.root/'exports'/('accepted_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:6])
            destination.mkdir()
            records = []
            for identity in self.manifest['ids']:
                item = self.item(identity)
                if item['review'] != 'accepted':
                    continue
                self._validate_source(item)
                mask, meta = self.mask(item)
                width, height = item['image_size_px']
                full = np.zeros((height, width), np.uint8)
                x0, y0, x1, y1 = meta['crop_px']
                full[y0:y1, x0:x1] = mask.astype(np.uint8)*255
                Image.fromarray(full).save(destination/f'{identity}.png')
                ys, xs = np.where(full)
                record = {k: deepcopy(item[k]) for k in ('id', 'plant_id', 'source_annotation_id', 'source_annotation', 'image_file',
                          'image_sha256', 'image_size_px', 'source_bbox_xywh_px', 'transform', 'image_provenance', 'prompts',
                          'sensitivity', 'selected_candidate', 'cache_key', 'history', 'changed', 'visibility')}
                record.update(organ_class=self.manifest.get('organ_class', 'leaflet'),
                              mask_file=f'{identity}.png', mask_bbox_xyxy_px=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                              mask_area_px=int(mask.sum()), model=meta['model'], source_logits_sha256=meta['logits_sha256'],
                              review='accepted_by_user', identity_confirmed=False, depth_confirmed=False,
                              rings=raster_rings(mask, meta['crop_px']), raster_is_authoritative=True)
                records.append(record)
            kind = self.manifest.get('organ_class', 'leaflet')
            result = dict(schema_version='accepted-leaflet-masks-1' if kind == 'leaflet' else 'accepted-compound-masks-1',
                          organ_class=kind, observations=records, count=len(records),
                          source_manifest_sha256=common.file_hash(self.root/'manifest.json'), created_at=now(),
                          notice='Accepted 2D visible regions only; no cross-view identity or depth confirmation')
            atomic_json(destination/'accepted.json', result)
            return dict(count=len(records), path=str(destination), manifest=str(destination/'accepted.json'))

    def audit(self) -> dict:
        """Account for every observation and verify preserved inputs/cached scores."""
        items = [self.item(identity) for identity in self.manifest['ids']]
        changed_inputs, bad_caches = [], []
        inputs = dict(self.manifest['input_hashes'])
        for item in items:
            if item['image_path']:
                inputs[str(self.session/item['image_path'])] = item['image_sha256']
        for name, sha in inputs.items():
            path = Path(name)
            if not path.is_file() or common.file_hash(path) != sha:
                changed_inputs.append(name)
        records = []
        for item in items:
            risk = {}
            if item['cache_key']:
                cache = self.root/'cache'/item['cache_key']
                try:
                    meta = common.read_json(cache/'result.json')
                    if common.file_hash(cache/'logits.npz') != meta['logits_sha256']:
                        bad_caches.append(item['id'])
                    risk = meta['risk']
                except (OSError, ValueError):
                    bad_caches.append(item['id'])
            record = {k: item[k] for k in ('id', 'plant_id', 'image_file', 'view_group', 'source_annotation_id',
                      'generation', 'review', 'revision', 'generation_seconds', 'review_seconds', 'edit_seconds',
                      'selected_candidate', 'sensitivity', 'error')}
            record.update(default_mask_area_px=risk.get('area_px'), initial_components=risk.get('components'),
                          initial_holes=risk.get('holes'), initial_outside_bbox_fraction=risk.get('outside_bbox_fraction'),
                          initial_candidate_disagreement=risk.get('candidate_disagreement'),
                          initial_neighbor_overlap_fraction=risk.get('neighbor_overlap_max'))
            records.append(record)
        common.write_rows(self.root/'processing.csv', records, list(records[0]) if records else ['id'])
        report = dict(schema_version='leaflet-mask-batch-audit-1', checked_at=now(),
                      status=self.status(), observed_item_count=len(items), checked_input_files=len(inputs),
                      changed_inputs=changed_inputs, invalid_cache_items=bad_caches,
                      all_observations_accounted_for=len(items) == len(self.manifest['ids']),
                      generation_complete=all(i['generation'] in ('ready', 'failed') for i in items),
                      human_review_complete=all(i['review'] in ('accepted', 'rejected') for i in items),
                      timing_definition='Sum of per-item generation wall seconds, including failed attempts; not total project elapsed time',
                      quality_notice='No measured mask accuracy; risk signals are only review priorities')
        atomic_json(self.root/'audit.json', report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--plant', required=True)
    p.add_argument('--organ-class', choices=['leaflet', 'compoundleaves'], default='leaflet')
    for name in ('labels', 'session', 'out'):
        p.add_argument('--'+name, required=True, type=Path)
    for name in ('run', 'status', 'export', 'audit', 'auto-accept'):
        p = sub.add_parser(name); p.add_argument('--batch', required=True, type=Path)
        if name == 'run':
            p.add_argument('--segmenter-config', required=True, type=Path)
            p.add_argument('--pilot', action='store_true'); p.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.plant, args.labels, args.session, args.out, args.organ_class)
        print(json.dumps(dict(items=len(result['ids']), pilot_photos=result['pilot_photos'])))
        return
    batch = Batch(args.batch, seg.load_segmenter('sam2', args.segmenter_config) if args.command == 'run' else None)
    if args.command == 'run':
        batch.start(args.pilot, args.retry_failed)
        while batch.worker.is_alive():
            batch.worker.join(10)
            print(json.dumps(batch.status(), ensure_ascii=False), flush=True)
        result = batch.status()
        atomic_json(batch.root/'summary.json', result)
    else:
        result = batch.export() if args.command == 'export' else batch.audit() if args.command == 'audit'             else batch.auto_accept() if args.command == 'auto-accept' else batch.status()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
