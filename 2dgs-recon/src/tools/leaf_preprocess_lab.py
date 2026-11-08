"""Independent, local RGB preprocessing comparison; no learned inference or geometry edits.

Prepare: python -m tools.leaf_preprocess_lab prepare --accepted PATH --session PATH --out BUILD
Serve:   python -m tools.leaf_preprocess_lab serve --run BUILD --port 8832
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import secrets
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import cv2
import numpy as np
from PIL import Image, ImageDraw

DEFAULTS = dict(ev=0.0, gamma=1.0, clahe=0.0, denoise=0.0, feature='combined', width=2.0,
                protocol='legacy', smoothing='none', tone='none', channel='green')
LIMITS = dict(ev=(-2, 2), gamma=(0.5, 2), clahe=(0, 3), denoise=(0, 30), width=(1, 5))
PRESETS = [
    dict(id='original', label='원본', settings=DEFAULTS),
    dict(id='lift', label='어두운 잎 밝히기', settings=dict(DEFAULTS, ev=0.5, gamma=0.85)),
    dict(id='lower', label='밝은 잎 낮추기', settings=dict(DEFAULTS, ev=-0.5)),
    dict(id='local', label='국소 대비', settings=dict(DEFAULTS, clahe=1.5)),
    dict(id='gentle', label='밝기 + 약한 대비', settings=dict(DEFAULTS, ev=0.25, gamma=0.9, clahe=1.0)),
    dict(id='clean', label='약한 노이즈 완화', settings=dict(DEFAULTS, clahe=1.0, denoise=12.0)),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def settings(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('Unknown settings')
    result = dict(DEFAULTS, **value)
    for k, (lo, hi) in LIMITS.items():
        if isinstance(result[k], bool):
            raise ValueError(f'Invalid {k}')
        result[k] = float(result[k])
        if not math.isfinite(result[k]) or not lo <= result[k] <= hi:
            raise ValueError(f'{k} must be in [{lo}, {hi}]')
    if result['feature'] not in ('brightness', 'ratio', 'combined'):
        raise ValueError('Unknown feature mode')
    from tools.preprocess_study import recipe
    recipe(result['smoothing'], result['tone'])
    if result['protocol'] not in ('legacy', 'study') or result['channel'] not in ('luminance','red','green','blue'):
        raise ValueError('Unknown protocol or map')
    if result['protocol'] == 'study' and any(result[k] != DEFAULTS[k] for k in LIMITS):
        raise ValueError('Study mode fixes EV and ridge width; use smoothing and tone selectors')
    return result


def linear_rgb(rgb):
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def encode_rgb(x):
    x = np.clip(x, 0, 1)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)


def colour_ratios(linear):
    total = linear.sum(axis=2)
    valid = total > 1e-6
    result = np.zeros_like(linear)
    np.divide(linear, total[..., None], out=result, where=valid[..., None])
    return result, valid


def full_channel_fraction(rgb, channel):
    """Scalar channel/(R+G+B) from stored RGB, no mask or tone adjustment."""
    if channel not in ('red', 'green', 'blue'):
        raise ValueError('Unknown RGB channel')
    index = ('red', 'green', 'blue').index(channel)
    values = np.asarray(rgb, dtype=np.float32)
    total = values.sum(axis=2)
    valid = total > 0
    ratio = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(values[..., index], total, out=ratio, where=valid)
    gray = np.rint(np.nan_to_num(ratio) * 255).astype(np.uint8)
    display = np.repeat(gray[..., None], 3, axis=2)
    display[~valid] = [255, 0, 255]
    return ratio, display


def full_green_fraction(rgb):
    """Preserve the existing green-fraction entry point."""
    return full_channel_fraction(rgb, 'green')


def apply_reviewed_mask(display, mask):
    """Full-resolution RGBA: preserve all values and holes; hide outside mask."""
    if display.shape[:2] != mask.shape or mask.dtype != bool:
        raise ValueError('Reviewed mask must match original image dimensions')
    return np.dstack((display, mask.astype(np.uint8) * 255))


def enhance(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """Photometric edits only: identical dimensions/pixel grid, never warp or sharpen."""
    cfg = settings(cfg)
    if cfg['protocol'] == 'study':
        from tools.preprocess_study import process_rgb
        return process_rgb(rgb, cfg['smoothing'], cfg['tone'])
    if all(cfg[k] == DEFAULTS[k] for k in ('ev', 'gamma', 'clahe', 'denoise')):
        return rgb.copy()
    result = encode_rgb(np.clip(linear_rgb(rgb) * (2 ** cfg['ev']), 0, 1) ** cfg['gamma'])
    if cfg['denoise']:
        result = cv2.bilateralFilter(result, 5, cfg['denoise'], 2)
    if cfg['clahe']:
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=cfg['clahe'], tileGridSize=(8, 8)).apply(lab[..., 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result


def ridge(channel: np.ndarray, widths: list[float]) -> np.ndarray:
    """Generic bright/dark Hessian line evidence, not an anatomical classifier."""
    output = np.zeros(channel.shape, np.float32)
    for sigma in widths:
        smooth = cv2.GaussianBlur(channel.astype(np.float32), (0, 0), sigma)
        xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3) * sigma ** 2
        yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3) * sigma ** 2
        xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3) * sigma ** 2
        delta = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
        a, b = (xx + yy + delta) / 2, (xx + yy - delta) / 2
        small, large = np.minimum(np.abs(a), np.abs(b)), np.maximum(np.abs(a), np.abs(b))
        response = large * np.exp(-0.5 * (small / (large + 1e-8) / 0.5) ** 2)
        output = np.maximum(output, response)
    return output


def normalize_evidence(evidence, valid):
    values = evidence[valid]
    scale = float(np.percentile(values, 99)) if len(values) else 0.0
    return np.clip(evidence / scale, 0, 1) if scale > 1e-8 else np.zeros_like(evidence)


def analyze(rgb, mask, cfg):
    cfg = settings(cfg)
    result = enhance(rgb, cfg)
    original_linear, processed_linear = linear_rgb(rgb), linear_rgb(result)
    ratios, ratio_valid = colour_ratios(original_linear)
    # Original clipping/darkness cannot be repaired by brightening. Do not create confidence there.
    orig_y = original_linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    processed_y = processed_linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    reliability = (orig_y > 0.008) & (rgb.max(axis=2) < 250) & ratio_valid
    if cfg['feature'] != 'ratio':
        reliability &= (processed_y > 0.008) & (result.max(axis=2) < 250)
    widths = [max(0.7, cfg['width'] * 0.5), cfg['width'], cfg['width'] * 1.5]
    # Context RGB is retained for derivatives; only score safely inside the mask.
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    support_radius = int(math.ceil(max(widths) * 3)) + 1
    bad = (~reliability).astype(np.uint8)
    contaminated = cv2.dilate(bad, np.ones((support_radius * 2 + 1,) * 2, np.uint8)) > 0
    valid = mask & (distance > support_radius) & ~contaminated
    brightness = normalize_evidence(ridge(processed_y, widths), valid)
    chroma = normalize_evidence(np.maximum.reduce([ridge(ratios[..., k], widths) for k in range(3)]), valid)
    evidence = {'brightness': brightness, 'ratio': chroma, 'combined': 0.7 * brightness + 0.3 * chroma}[cfg['feature']]
    evidence = np.where(valid, evidence, 0).astype(np.float32)
    overlay = result.copy()
    alpha = evidence[..., None] * 0.8
    overlay = np.rint(overlay * (1 - alpha) + np.array([255, 72, 24]) * alpha).astype(np.uint8)
    # Neutral rather than painted green: colour ratios are a diagnostic display only.
    ratio_display = encode_rgb(ratios)
    ratio_display[~(mask & reliability)] = 32
    vals = processed_y[mask]
    local = processed_y - cv2.GaussianBlur(processed_y, (0, 0), 5)
    stats = dict(mask_pixels=int(mask.sum()),
                 source_clipped_percent=float(100 * np.mean(rgb[mask].max(axis=1) >= 250)),
                 processed_clipped_percent=float(100 * np.mean(result[mask].max(axis=1) >= 250)),
                 source_dark_percent=float(100 * np.mean(orig_y[mask] <= 0.008)),
                 processed_dark_percent=float(100 * np.mean(vals <= 0.008)),
                 usable_line_region_percent=float(100 * valid.sum() / mask.sum()),
                 local_contrast_std=float(np.std(local[mask])))
    return dict(processed=result, overlay=overlay, ratio=ratio_display, evidence=evidence,
                valid=valid, stats=stats, settings=cfg)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def data_url(array):
    return 'data:image/png;base64,' + base64.b64encode(png(array)).decode('ascii')


def assert_build(path):
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1] / 'build'
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Output must be a new subdirectory of this project build/')
    return path


def prepare(accepted_paths, sessions, out, per_group=2, all_observations=False):
    """Read immutable accepted exports; deterministic pilot sampling, no quality selection."""
    if len(accepted_paths) != len(sessions) or not 1 <= per_group <= 10:
        raise ValueError('Pair each accepted export with its session; per-group must be 1..10')
    out = assert_build(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'dataset.json').exists() or (out / 'samples').exists():
        raise ValueError('Pilot already exists; use a fresh output directory')
    candidates, exclusions, sources = [], [], {}
    for accepted, session in zip(accepted_paths, sessions):
        accepted, session = Path(accepted).resolve(), Path(session).resolve()
        export = read_json(accepted)
        if export.get('schema_version') != 'accepted-leaflet-masks-1':
            raise ValueError('Only reviewed leaflet exports are supported')
        capture_path = session / 'capture.json'
        capture = read_json(capture_path)
        frames = {f['image_file']: f for f in capture['frames']}
        sources.update({str(accepted): sha(accepted), str(capture_path): sha(capture_path)})
        groups = {}
        for obs in export['observations']:
            if obs.get('review') != 'accepted_by_user':
                continue
            group = obs.get('image_provenance', {}).get('view_group', 'unknown')
            if group not in ('middle', 'side'):
                exclusions.append(dict(id=obs['id'], reason='outside_middle_side_pilot'))
                continue
            groups.setdefault(group, []).append(obs)
        for group, records in sorted(groups.items()):
            records.sort(key=lambda r: (r['mask_area_px'], r['id']))
            # Representative area quantiles, distinct photos when possible; no claim of representativeness.
            selected, photos = [], set()
            for fraction in ([] if all_observations else np.linspace(0.4, 0.8, per_group)):
                target = int((len(records) - 1) * fraction)
                ordered = sorted(range(len(records)), key=lambda i: abs(i - target))
                record = next((records[i] for i in ordered if records[i]['image_file'] not in photos), None)
                if record is None:
                    break
                selected.append(record); photos.add(record['image_file'])
            if all_observations:
                selected = sorted(records, key=lambda r: (r['image_file'], str(r['id'])))
            for obs in selected:
                frame = frames[obs['image_file']]
                image_path = (session / frame['image_path']).resolve()
                mask_path = (accepted.parent / obs['mask_file']).resolve()
                if not mask_path.is_relative_to(accepted.parent):
                    raise ValueError('Mask path escapes accepted export')
                image_hash = sha(image_path)
                if image_hash != obs['image_sha256'] or image_hash != frame['sha256']:
                    raise ValueError(f'Source photo changed: {obs["image_file"]}')
                with Image.open(image_path) as im:
                    if im.getexif().get(274, 1) != 1:
                        raise ValueError('Use existing orientation-normalized session photos')
                    rgb = np.array(im.convert('RGB'))
                with Image.open(mask_path) as im:
                    full_mask = np.array(im.convert('L')) > 0
                if full_mask.shape != rgb.shape[:2] or list(rgb.shape[1::-1]) != obs['image_size_px'] or not full_mask.any():
                    raise ValueError('Image/mask geometry mismatch or empty mask')
                yy, xx = np.where(full_mask)
                pad = 40
                crop = [max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad),
                        min(rgb.shape[1], int(xx.max()) + pad + 1), min(rgb.shape[0], int(yy.max()) + pad + 1)]
                x0, y0, x1, y1 = crop
                sid = f'sample_{len(candidates):03d}'
                folder = out / 'samples' / sid
                folder.mkdir(parents=True)
                Image.fromarray(rgb[y0:y1, x0:x1]).save(folder / 'original.png')
                Image.fromarray(full_mask[y0:y1, x0:x1].astype(np.uint8) * 255).save(folder / 'mask.png')
                mask_hash = sha(mask_path)
                sources.update({str(image_path): image_hash, str(mask_path): mask_hash})
                candidates.append(dict(id=sid, plant_id=obs['plant_id'], session_id=capture.get('session_id'),
                    source_annotation_id=obs['source_annotation_id'], observation_id=obs['id'],
                    image_file=obs['image_file'], view_group=group, visibility=obs.get('visibility'),
                    source_image=str(image_path), source_image_sha256=image_hash, source_mask=str(mask_path),
                    source_mask_sha256=mask_hash, accepted_export=str(accepted), accepted_export_sha256=sources[str(accepted)],
                    original_size_px=obs['image_size_px'], crop_xyxy_px=crop,
                    transform=dict(scale_x=1, scale_y=1, offset_x_px=x0, offset_y_px=y0,
                                   formula='x_original=x_crop+offset_x_px; y_original=y_crop+offset_y_px'),
                    original_crop_sha256=sha(folder / 'original.png'), mask_crop_sha256=sha(folder / 'mask.png'),
                    pose_status=frame.get('pose_status'), passes_pose_gate=frame.get('passes_pose_gate'),
                    physical_identity_confirmed=False))
    if not candidates:
        raise ValueError('No usable accepted pilot observations')
    for path, digest in sources.items():
        if sha(Path(path)) != digest:
            raise ValueError('Source changed during preparation')
    manifest = dict(schema='rgb-preprocess-lab-1', samples=candidates, sources=sources, exclusions=exclusions,
        selection=('All human-accepted middle/side observations; not unique physical leaflets' if all_observations else
                   '2D mask-area quantiles, distinct photos per plant/view group; not an accuracy or identity sample'),
        colour_assumption='Prepared 8-bit RGB treated as sRGB for diagnostic processing; original phone tone mapping unknown',
        inference_run=False, training_run=False, validated_metric=False, suitable_for_ppfd=False)
    write_json(out / 'dataset.json', manifest)
    return manifest


class Lab:
    def __init__(self, root, experiments=None):
        self.root = assert_build(root)
        self.dataset_path = self.root / 'dataset.json'
        self.dataset_hash = sha(self.dataset_path)
        self.doc = read_json(self.dataset_path)
        self.samples = {r['id']: r for r in self.doc['samples']}
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.study_cache = {}
        self.experiments = [assert_build(p) for p in (experiments or [])]

    def model_images(self, sid, cfg, rgb):
        """Read completed predictions only. Viewing never triggers model inference."""
        record = self.samples[sid]
        condition = cfg['smoothing']+'__'+cfg['tone']
        stem = Path(record['image_file']).stem
        images, details = {}, []
        for name, suffix in ((condition,''), ('none__none','_baseline')):
            for root in self.experiments:
                manifest_path = root/'manifest.json'
                path = root/name/stem/'predictions.json'
                if not manifest_path.exists() or not path.exists():
                    continue
                manifest = read_json(manifest_path)
                relative = Path(record['source_image']).relative_to(Path(__file__).resolve().parents[1]).as_posix()
                source_hashes = [h for p,h in manifest['source_hashes'].items()
                                 if p.replace('\\','/').endswith('/'+relative)]
                if not source_hashes or any(h != record['source_image_sha256'] for h in source_hashes):
                    continue
                pd = read_json(path); image = Image.fromarray(rgb.copy()); draw = ImageDraw.Draw(image)
                x0,y0,x1,y1 = record['crop_xyxy_px']
                from tools.organ_prelabel import COLOURS, CLASSES
                for c,b,s in pd['predictions']:
                    draw.rectangle([b[0]-x0,b[1]-y0,b[2]-x0,b[3]-y0],outline=COLOURS[c],width=2)
                    if x0<=b[0]<x1 and y0<=b[1]<y1:
                        draw.text((b[0]-x0,b[1]-y0),f'{CLASSES[c]} {s:.2f}',fill=COLOURS[c])
                images['model'+suffix] = np.asarray(image)
                details.append(dict(run=str(root),condition=name,inference_seconds=pd['seconds']))
                meta = path.parent/f"{record['observation_id']}_prediction.json"
                maskpath = path.parent/f"{record['observation_id']}_predicted.png"
                if meta.exists() and maskpath.exists():
                    m=read_json(meta); a,b,c,d=m['crop_xyxy_px']
                    with Image.open(maskpath) as im:
                        pred=np.array(im)>0
                    canvas=np.zeros(rgb.shape[:2],bool)
                    left,top,right,bottom=max(a,x0),max(b,y0),min(c,x1),min(d,y1)
                    if right>left and bottom>top:
                        canvas[top-y0:bottom-y0,left-x0:right-x0]=pred[top-b:bottom-b,left-a:right-a]
                    overlay=rgb.copy(); overlay[canvas]=np.rint(.6*overlay[canvas]+.4*np.array([0,220,220])).astype(np.uint8)
                    images['modelmask'+suffix]=overlay
                break
        return images, details

    def evaluate(self, sid, cfg):
        cfg = settings(cfg)
        rgb, mask = self.load(sid)
        if cfg['protocol'] == 'legacy':
            return analyze(rgb, mask, cfg)
        from tools.preprocess_study import study_analysis, process_rgb
        record = self.samples[sid]
        source = Path(record['source_image'])
        if sha(source) != record['source_image_sha256']:
            raise ValueError('Source photograph changed')
        # Full-frame CLAHE and filtering before cropping: no crop-dependent tone or boundary artifacts.
        key = (record['source_image_sha256'], cfg['smoothing'], cfg['tone'])
        with self.lock:
            if key not in self.study_cache:
                with Image.open(source) as im:
                    whole = np.array(im.convert('RGB'))
                processed = process_rgb(whole, cfg['smoothing'], cfg['tone'])
                self.study_cache.clear()  # bounded one-frame cache
                self.study_cache[key] = processed
            x0,y0,x1,y1 = record['crop_xyxy_px']
            processed = self.study_cache[key][y0:y1,x0:x1].copy()
        result = study_analysis(rgb, mask, cfg['smoothing'], cfg['tone'], processed, cfg['channel'])
        result['settings'] = cfg
        return result

    def load(self, sid):
        if sid not in self.samples:
            raise ValueError('Unknown observation')
        r = self.samples[sid]
        folder = self.root / 'samples' / sid
        if sha(folder / 'original.png') != r['original_crop_sha256'] or sha(folder / 'mask.png') != r['mask_crop_sha256']:
            raise ValueError('Pilot crop changed; prepare a fresh pilot')
        with Image.open(folder / 'original.png') as im:
            rgb = np.array(im.convert('RGB'))
        with Image.open(folder / 'mask.png') as im:
            mask = np.array(im) > 0
        return rgb, mask

    def render(self, sid, cfg):
        rgb, mask = self.load(sid)
        result = self.evaluate(sid, cfg)
        outline = rgb.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline, contours, -1, (0, 210, 210), 1)
        images = dict(original=rgb, mask=outline, processed=result['processed'], overlay=result['overlay'], ratio=result['ratio'])
        for key in ('scalar_before', 'difference'):
            if key in result:
                images[key] = result[key]
        models,details=self.model_images(sid,result['settings'],rgb)
        images.update(models)
        return dict(settings=result['settings'], stats=result['stats'],model_results=details,
                    images={k: data_url(v) for k, v in images.items()})

    def save(self, sid, cfg, verdict, note):
        cfg = settings(cfg)
        if verdict not in ('easier', 'similar', 'harder', 'unreviewed') or not isinstance(note, str) or len(note) > 2000:
            raise ValueError('Invalid review')
        result = self.evaluate(sid, cfg)
        with self.lock:
            if sha(self.dataset_path) != self.dataset_hash:
                raise ValueError('Pilot manifest changed; restart with a fresh snapshot')
            r = self.samples.get(sid)
            if r is None:
                raise ValueError('Unknown observation')
            for path, key in (('source_image', 'source_image_sha256'), ('source_mask', 'source_mask_sha256'),
                              ('accepted_export', 'accepted_export_sha256')):
                if sha(Path(r[path])) != r[key]:
                    raise ValueError('Original photo, reviewed mask or accepted export changed; save refused')
            rgb, mask = self.load(sid)
            folder = self.root / 'selections' / (datetime.now().strftime('%Y%m%d_%H%M%S_') + secrets.token_hex(4))
            folder.mkdir(parents=True)
            Image.fromarray(result['processed']).save(folder / 'processed_crop.png')
            Image.fromarray(result['overlay']).save(folder / 'line_candidates.png')
            Image.fromarray(result['valid'].astype(np.uint8) * 255).save(folder / 'line_evidence_valid.png')
            np.save(folder / 'line_evidence.npy', result['evidence'], allow_pickle=False)
            if 'scalar_array' in result:
                np.save(folder / 'scalar_map.npy', result['scalar_array'], allow_pickle=False)
                Image.fromarray(result['scalar_common'].astype(np.uint8)*255).save(folder / 'scalar_common.png')
                Image.fromarray(result['difference']).save(folder / 'difference_x4.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(folder / 'unchanged_mask.png')
            record = dict(schema='rgb-preprocess-selection-1', observation=r, settings=cfg,
                review=dict(verdict=verdict, note=note), diagnostics=result['stats'],
                saved_at=datetime.now().astimezone().isoformat(), dataset_sha256=self.dataset_hash,
                versions=dict(opencv=cv2.__version__, numpy=np.__version__),
                tool_sha256=sha(Path(__file__)), evidence_display_normalization='per-image 99th percentile, not a confidence',
                feature_recipe='source linear RGB ratios + processed luminance Hessian; combined weights 0.3/0.7',
                geometrical_transform='crop translation only; no resampling/warp',
                new_depth_information=False, physical_point_identity_confirmed=False,
                inference_run=False, training_run=False, validated_metric=False, suitable_for_ppfd=False,
                artifact_hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
            if cfg['protocol'] == 'study':
                from tools.preprocess_study import recipe
                record['feature_recipe'] = recipe(cfg['smoothing'],cfg['tone'])
                record['scalar_stats'] = result['scalar_stats']
            write_json(folder / 'selection.json', record)
            return dict(path=str(folder), relative_path=str(folder.relative_to(self.root)), record=record)

    def compare(self):
        rows = []
        for sid in self.samples:
            rgb, mask = self.load(sid)
            for preset in PRESETS:
                result = analyze(rgb, mask, preset['settings'])
                rows.append(dict(sample_id=sid, plant_id=self.samples[sid]['plant_id'],
                    image_file=self.samples[sid]['image_file'], preset=preset['id'], **result['stats']))
        path = self.root / 'preset_diagnostics.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return rows

    def midrib_preview(self, sid, cfg, anchors, options):
        from tools.midrib_review import propose, parse_options
        cfg = settings(cfg)
        options = parse_options(options)
        rgb, mask = self.load(sid)
        processed = self.evaluate(sid, cfg)['processed'] if options['channel'] == 'processed_l' else rgb
        r = propose(rgb, mask, processed, anchors, self.samples[sid]['crop_xyxy_px'], options)
        valid = r.pop('valid'); evidence = r.pop('evidence')
        outline = rgb.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline, contours, -1, (0,210,210), 1)
        overlay = rgb.copy()
        overlay[mask & ~valid] = np.rint(.6*overlay[mask & ~valid]+.4*np.array([110,70,160])).astype(np.uint8)
        alpha = evidence[...,None]*.7
        overlay = np.rint(overlay*(1-alpha)+np.array([255,120,0])*alpha).astype(np.uint8)
        r['settings'] = cfg
        r['images'] = dict(original=data_url(outline), evidence=data_url(overlay))
        return r

    def save_midrib(self, sid, cfg, anchors, options, review, note):
        if review not in ('candidate','confirmed_2d','rejected') or not isinstance(note,str) or len(note)>2000:
            raise ValueError('Invalid midrib review')
        result = self.midrib_preview(sid,cfg,anchors,options)
        if result['status'] != 'proposed':
            raise ValueError('A valid visible path is required before saving')
        with self.lock:
            if sha(self.dataset_path) != self.dataset_hash:
                raise ValueError('Pilot manifest changed; save refused')
            observation = self.samples[sid]
            for path, key in (('source_image','source_image_sha256'),('source_mask','source_mask_sha256'),
                              ('accepted_export','accepted_export_sha256')):
                if sha(Path(observation[path])) != observation[key]:
                    raise ValueError('Source photo, mask or export changed; save refused')
            folder = self.root/'midrib_reviews'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+secrets.token_hex(4))
            folder.mkdir(parents=True)
            for name,value in result.pop('images').items():
                (folder/(name+'.png')).write_bytes(base64.b64decode(value.split(',',1)[1]))
            rgb, _ = self.load(sid)
            annotated = Image.fromarray(rgb); draw = ImageDraw.Draw(annotated)
            x0,y0,_,_ = observation['crop_xyxy_px']
            draw.line([(x-x0,y-y0) for x,y in result['path_original_px']],fill=(255,70,70),width=2)
            for x,y in anchors:
                draw.ellipse((x-x0-3,y-y0-3,x-x0+3,y-y0+3),outline=(255,230,90),width=2)
            annotated.save(folder/'path_overlay.png')
            with (folder/'path.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f); writer.writerow(['point_index','x_original_px','y_original_px'])
                writer.writerows((i,*p) for i,p in enumerate(result['path_original_px']))
            record = dict(schema='midrib-review-1', observation=observation, proposal=result,
                review=dict(state=review,note=note), saved_at=datetime.now().astimezone().isoformat(),
                coordinates='zero-based original-image pixel centres, x right, y down',
                dataset_sha256=self.dataset_hash, tool_sha256=sha(Path(__file__)),
                path_tool_sha256=sha(Path(__file__).with_name('midrib_review.py')),
                shared_algorithm_sha256=sha(Path(__file__).with_name('midrib_flow.py')),
                versions=dict(opencv=cv2.__version__,numpy=np.__version__),
                endpoint_semantics='User orders visible base -> optional anchors -> visible tip; no hidden completion',
                physical_point_identity_confirmed=False, anatomical_accuracy='unavailable',
                new_depth_information=False,validated_metric=False,suitable_for_ppfd=False,
                inference_run=False,training_run=False,
                artifact_hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
            write_json(folder/'review.json',record)
            return dict(relative_path=str(folder.relative_to(self.root)),record=record)


def handler(lab):
    vein_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200, mime='application/json; charset=utf-8'):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if not isinstance(value, bytes) else value
            self.send_response(status); self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
                self.reply(dict(error='Use the local 127.0.0.1 address'), 403); return
            route = urlsplit(self.path).path
            if route == '/':
                page = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
                from tools.preprocess_study import SMOOTHING, TONES
                payload = dict(samples=lab.doc['samples'], presets=PRESETS, defaults=DEFAULTS, token=lab.token,
                               smoothing=list(SMOOTHING), tones=list(TONES))
                page = page.replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
                self.reply(page.encode(), mime='text/html; charset=utf-8')
            elif route == '/app.js':
                self.reply(Path(__file__).with_suffix('.js').read_bytes(), mime='text/javascript; charset=utf-8')
            elif route == '/vein.js':
                self.reply(Path(__file__).with_name('vein_review.js').read_bytes(), mime='text/javascript; charset=utf-8')
            elif route == '/surface.js':
                self.reply(Path(__file__).with_name('leaf_surface_review.js').read_bytes(), mime='text/javascript; charset=utf-8')
            elif route == '/veins/' or route.startswith('/veins/'):
                name = route.removeprefix('/veins/') or 'index.html'
                path = lab.root / 'veins' / name
                allowed = name == 'index.html' or (name.endswith('.png') and name[:-4] in lab.samples)
                if not allowed or not path.is_file():
                    self.reply(dict(error='No vein artifact'), 404); return
                self.reply(path.read_bytes(), mime='image/png' if name.endswith('.png') else 'text/html; charset=utf-8')
            elif route.startswith('/source/'):
                record = lab.samples.get(route.removeprefix('/source/'))
                if record is None:
                    self.reply(dict(error='Unknown observation'), 404); return
                try:
                    path = Path(record['source_image'])
                    if sha(path) != record['source_image_sha256']:
                        raise ValueError('Source photograph changed')
                    with Image.open(path) as im:
                        rgb = np.array(im.convert('RGB'))
                    query = parse_qs(urlsplit(self.path).query)
                    mode = query.get('mode', ['original'])[0]
                    smoothing = query.get('smoothing', ['none'])[0]
                    tone = query.get('tone', ['none'])[0]
                    if mode in ('red', 'green', 'blue'):
                        from tools.preprocess_study import scalar_result, display_scalar
                        values, rgb = full_channel_fraction(rgb, mode)
                        if smoothing != 'none':
                            a,valid,_,_ = scalar_result(values,np.ones(values.shape,bool),smoothing)
                            rgb = display_scalar(a,valid)
                    elif mode == 'processed':
                        from tools.preprocess_study import process_rgb
                        rgb = process_rgb(rgb,smoothing,tone)
                    elif mode != 'original':
                        raise ValueError('Unknown full-frame display mode')
                    mask_mode = query.get('mask', ['none'])[0]
                    if mask_mode == 'reviewed':
                        mask_path = Path(record['source_mask'])
                        if sha(mask_path) != record['source_mask_sha256']:
                            raise ValueError('Reviewed mask changed')
                        with Image.open(mask_path) as im:
                            mask = np.array(im.convert('L')) > 0
                        rgb = apply_reviewed_mask(rgb, mask)
                    elif mask_mode != 'none':
                        raise ValueError('Unknown mask mode')
                    self.reply(png(rgb), mime='image/png')
                except (ValueError, OSError) as exc:
                    self.reply(dict(error=str(exc)), 400)
            else:
                self.reply(dict(error='Not found'), 404)

        def do_POST(self):
            route = urlsplit(self.path).path
            try:
                origin = self.headers.get('Origin')
                allowed = f'http://127.0.0.1:{self.server.server_port}'
                if (self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}' or
                    self.headers.get('X-Lab-Token') != lab.token or (origin is not None and origin != allowed)):
                    self.reply(dict(error='Local session token required'), 403); return
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= (128000 if route.startswith('/api/vein/') else 16000):
                    self.reply(dict(error='Invalid request size'), 413); return
                body = json.loads(self.rfile.read(length))
                if route.startswith('/api/vein/'):
                    from tools.vein_correspondence import Workflow
                    with vein_lock:
                        if not hasattr(lab,'vein_workflow'): lab.vein_workflow=Workflow(lab)
                        w=lab.vein_workflow
                        if route=='/api/vein/pair': result=w.pair(body['a'],body['b'],body.get('graph_settings',{}))
                        elif route=='/api/vein/candidates': result=w.candidates(body['a'],body.get('graph_settings',{}),body.get('image_file'))
                        elif route=='/api/vein/view-guide':
                            from tools.vein_view_guide import view_guide
                            result=view_guide(w,body['a'])
                        elif route=='/api/vein/photo-preview':
                            from tools.vein_view_guide import photo_preview
                            result=photo_preview(w,body['a'])
                        elif route=='/api/vein/epiline': result=dict(points=w.epiline(body['a'],body['b'],body['xy']))
                        elif route=='/api/vein/save': result=w.save(body)
                        elif route=='/api/vein/export': result=w.export(body.get('mode','strict'))
                        elif route=='/api/vein/status': result=w.status()
                        elif route=='/api/vein/surface':
                            from tools.leaf_surface_review import run_surface
                            result=run_surface(w,body['leaflet_id'],body.get('mode','diagnostic'),body.get('reviewed_by',''))
                        elif route=='/api/vein/surface-results':
                            from tools.leaf_surface_review import surface_results
                            result=surface_results(w)
                        else: self.reply(dict(error='Not found'),404); return
                    self.reply(result); return
                sid, cfg = body['sample_id'], settings(body['settings'])
                if route == '/api/render':
                    self.reply(lab.render(sid, cfg))
                elif route == '/api/save':
                    self.reply(lab.save(sid, cfg, body.get('verdict', 'unreviewed'), body.get('note', '')))
                elif route == '/api/midrib':
                    self.reply(lab.midrib_preview(sid,cfg,body.get('anchors',[]),body.get('options',{})))
                elif route == '/api/midrib/save':
                    self.reply(lab.save_midrib(sid,cfg,body.get('anchors',[]),body.get('options',{}),
                                              body.get('review','candidate'),body.get('note','')))
                else:
                    self.reply(dict(error='Not found'), 404)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.reply(dict(error=str(exc)), 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--accepted', type=Path, action='append', required=True)
    p.add_argument('--session', type=Path, action='append', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-group', type=int, default=2)
    p.add_argument('--all-observations', action='store_true', help='All accepted middle/side labels, including multiple labels per photo')
    p = sub.add_parser('compare'); p.add_argument('--run', type=Path, required=True)
    p = sub.add_parser('serve'); p.add_argument('--run', type=Path, required=True); p.add_argument('--port', type=int, default=8832)
    p.add_argument('--experiment',type=Path,action='append',default=[],help='Read derived model comparison results')
    args = parser.parse_args()
    if args.command == 'prepare':
        doc = prepare(args.accepted, args.session, args.out, args.per_group, args.all_observations)
        print(json.dumps(dict(samples=len(doc['samples']), out=str(args.out))))
    elif args.command == 'compare':
        print(json.dumps(dict(comparisons=len(Lab(args.run).compare()))))
    else:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Lab(args.run,args.experiment)))
        print(f'http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
