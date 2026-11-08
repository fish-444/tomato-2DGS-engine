"""User-requested leaflet mask proposals for the local editor; never auto-accepted.

A segmenter turns the user's prompts (a bbox and/or include/exclude clicks, all in
original-image pixels) into candidate masks. SAM is a *prompted* segmentation model,
not a trained tomato organ detector: every result is a proposal for human review.

Backends
  unavailable  default. Reports why; manual labeling stays fully usable.
  sam2         loaded lazily from a JSON config (see SAM2_CONFIG_EXAMPLE; relative
               paths are resolved from the project root). The
               checkpoint hash is verified before loading. Not executed by tests.
  mock         deterministic ellipse/disc masks for workflow tests. NOT a model;
               its provenance says so and the editor shows it.

Two entrypoints: ``propose`` (one prompt -> alternative masks of ONE target) and
``propose_automatic`` (whole photograph -> many class-agnostic masks). Automatic masks
are unclassified: one can be a lobe, a whole compound leaf, several leaflets or
background. Nothing here detects "a tomato leaflet"; a human assigns identity.

Coordinates: clicks use OpenCV pixel centres (same as saved points); boxes use
continuous pixel edges (same as saved bboxes). Returned rings are OpenCV pixel
centres of boundary pixels; bbox_px = [min x, min y, max x + 1, max y + 1].
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

from tools.leaf_light.common import ROOT, StudyError, file_hash, read_json

CONTOUR_TOLERANCE_PX = 1.0     # approxPolyDP tolerance for editable rings
CROP_MARGIN_FRACTION = 0.5     # context added around the prompt extent, per side
MIN_CROP_MARGIN_PX = 128
MAX_CLICKS = 64
AUTOMATIC_MAX_MASKS = 60
AUTOMATIC_MIN_AREA_PX = 400          # keeps genuinely small leaflets, drops speckle
AUTOMATIC_MAX_AREA_FRACTION = 0.6    # a mask covering most of the photo is background, not a leaflet

SAM2_CONFIG_EXAMPLE = {
    'backend': 'sam2',
    'model_name': 'SAM 2.1 hiera tiny',
    'repository': 'build/sam_trial_20260919/vendor/sam2',
    'runtime': 'build/sam_trial_20260919/runtime',
    'checkpoint': 'build/sam_trial_20260919/checkpoints/sam2.1_hiera_tiny.pt',
    'checkpoint_sha256': '<sha256 of the checkpoint file>',
    'model_config': 'configs/sam2.1/sam2.1_hiera_t.yaml',
    'device': 'cpu',
    'threads': 4,
}


@dataclass
class MaskCandidate:
    mask: np.ndarray            # bool, crop-sized
    score: float | None         # model's own confidence, not measured accuracy
    logits: np.ndarray | None = None  # optional full crop scores; never thresholded twice


class UnavailableSegmenter:
    def __init__(self, reason: str):
        self.reason = reason

    def info(self) -> dict:
        return dict(available=False, backend='unavailable', reason=self.reason, is_mock=False,
                    automatic=False, class_agnostic=False)

    def predict(self, rgb, box, positive, negative) -> list[MaskCandidate]:
        raise StudyError('segmenter_unavailable: ' + self.reason)

    def automatic(self, rgb) -> list[MaskCandidate]:
        raise StudyError('segmenter_unavailable: ' + self.reason)


class MockSegmenter:
    """Deterministic geometric masks for tests and UI walkthroughs; not a model."""

    def info(self) -> dict:
        return dict(available=True, backend='mock', model_name='mock-ellipse-not-a-model', model_version='mock-1',
                    is_mock=True, automatic=True, class_agnostic=True,
                    reason='Deterministic test geometry, not segmentation of image content')

    def predict(self, rgb, box, positive, negative) -> list[MaskCandidate]:
        height, width = rgb.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
            rx, ry = max((box[2] - box[0]) / 2, 1), max((box[3] - box[1]) / 2, 1)
        else:
            cx, cy = np.mean(positive, axis=0)
            rx = ry = 40.0
        candidates = []
        for shrink, score in ((1.0, 0.9), (0.7, 0.6)):
            mask = ((xx - cx) / (rx * shrink)) ** 2 + ((yy - cy) / (ry * shrink)) ** 2 <= 1
            for x, y in negative:
                mask &= (xx - x) ** 2 + (yy - y) ** 2 > 15 ** 2
            candidates.append(MaskCandidate(mask, score))
        return candidates

    def automatic(self, rgb) -> list[MaskCandidate]:
        """Deterministic overlapping discs of different sizes; not image content."""
        height, width = rgb.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        plan = [(0.35, 0.40, 0.20, 0.80), (0.48, 0.52, 0.16, 0.70), (0.70, 0.35, 0.05, 0.55)]
        candidates = []
        for fx, fy, fr, score in plan:
            radius = max(6.0, fr * min(height, width))
            mask = (xx - fx * width) ** 2 + (yy - fy * height) ** 2 <= radius ** 2
            if mask.any():
                candidates.append(MaskCandidate(mask, score))
        return candidates


class Sam2Segmenter:
    """SAM 2 image predictor with a verified checkpoint. Loaded on first request."""

    def __init__(self, config: dict, base: Path):
        required = ('repository', 'checkpoint', 'checkpoint_sha256', 'model_config')
        if config.get('backend') != 'sam2' or any(not config.get(k) for k in required):
            raise StudyError('SAM 2 config needs backend=sam2 and ' + ', '.join(required))
        self.config, self.base, self.predictor, self.loaded = dict(config), base, None, {}
        self.generator = None

    def _path(self, key: str) -> Path:
        path = Path(self.config[key])
        return (path if path.is_absolute() else self.base / path).resolve()

    def info(self) -> dict:
        checkpoint = self._path('checkpoint')
        ready = checkpoint.is_file() and self._path('repository').is_dir()
        return dict(available=ready, backend='sam2', model_name=self.config.get('model_name', 'SAM 2'),
                    model_config=self.config['model_config'], checkpoint_sha256=self.config['checkpoint_sha256'],
                    is_mock=False, automatic=True, class_agnostic=True,
                    loaded=self.predictor is not None, **self.loaded,
                    reason=None if ready else 'checkpoint or repository not found; nothing is downloaded automatically')

    def _load(self) -> None:
        if self.predictor is not None:
            return
        checkpoint = self._path('checkpoint')
        if not checkpoint.is_file():
            raise StudyError('segmenter_unavailable: checkpoint not found')
        if file_hash(checkpoint) != self.config['checkpoint_sha256']:
            raise StudyError('segmenter_unavailable: checkpoint hash differs from config')
        for key in ('runtime', 'repository'):
            if self.config.get(key):
                sys.path.insert(0, str(self._path(key)))
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        torch.set_num_threads(int(self.config.get('threads', 4)))
        device = self.config.get('device', 'cpu')
        model = build_sam2(self.config['model_config'], str(checkpoint), device=device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
        self.loaded = dict(torch_version=torch.__version__, device=device)

    def _input(self, rgb):
        """Optional photometric preprocessing of the SAM input (config 'input_smoothing', a tools.preprocess_study
        recipe on Lab L; part of the config, so part of the cache key). Masks stay on the original pixel grid."""
        method = self.config.get('input_smoothing', 'none')
        if method == 'none':
            return rgb
        from tools.preprocess_study import process_rgb
        return process_rgb(np.ascontiguousarray(rgb, dtype=np.uint8), method)

    def predict(self, rgb, box, positive, negative) -> list[MaskCandidate]:
        self._load()
        import torch
        points = np.asarray(list(positive) + list(negative), dtype=np.float32) if (positive or negative) else None
        labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32) if points is not None else None
        with torch.inference_mode():
            self.predictor.set_image(self._input(rgb))
            masks, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=labels,
                box=None if box is None else np.asarray(box, dtype=np.float32), multimask_output=True)
        return [MaskCandidate(np.asarray(m, dtype=bool), float(s)) for m, s in zip(masks, scores)]

    def predict_logits(self, rgb, box, positive, negative) -> list[MaskCandidate]:
        """Keep complete crop-resolution scores for reversible threshold review."""
        self._load()
        import torch
        points = np.asarray(list(positive) + list(negative), dtype=np.float32) if (positive or negative) else None
        labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int32) if points is not None else None
        with torch.inference_mode():
            self.predictor.set_image(self._input(rgb))
            logits, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=labels,
                box=None if box is None else np.asarray(box, dtype=np.float32),
                multimask_output=True, return_logits=True)
        return [MaskCandidate(np.asarray(m) > 0, float(s), np.asarray(m, dtype=np.float32))
                for m, s in zip(logits, scores)]


def automatic_masks_from_sam2(self, rgb) -> list[MaskCandidate]:
    """SAM 2 automatic mask generation, invoked only by an explicit request."""
    self._load()
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    if self.generator is None:
        settings = dict(points_per_side=int(self.config.get('points_per_side', 24)),
                        points_per_batch=int(self.config.get('points_per_batch', 1)),
                        pred_iou_thresh=float(self.config.get('pred_iou_thresh', 0.7)),
                        stability_score_thresh=float(self.config.get('stability_score_thresh', 0.9)),
                        min_mask_region_area=int(self.config.get('min_mask_region_area', AUTOMATIC_MIN_AREA_PX)))
        self.generator = SAM2AutomaticMaskGenerator(self.predictor.model, **settings)
        self.loaded['automatic_settings'] = settings
    records = self.generator.generate(rgb)
    return [MaskCandidate(np.asarray(r['segmentation'], dtype=bool), float(r.get('predicted_iou', 0.0)))
            for r in records]


Sam2Segmenter.automatic = automatic_masks_from_sam2


def load_segmenter(backend: str | None, config_path: Path | None = None):
    if backend in (None, 'unavailable'):
        return UnavailableSegmenter('No segmentation model configured; start the server with --segmenter-config')
    if backend == 'mock':
        return MockSegmenter()
    if backend == 'sam2':
        if config_path is None or not config_path.is_file():
            return UnavailableSegmenter('SAM 2 config file not found')
        return Sam2Segmenter(read_json(config_path), ROOT)  # relative paths are project-root relative
    raise StudyError('Unknown segmenter backend')


def checked_prompts(prompts: dict, width: int, height: int) -> tuple[list | None, list, list]:
    box = prompts.get('box_px')
    positive, negative = prompts.get('positive_px') or [], prompts.get('negative_px') or []
    if len(positive) + len(negative) > MAX_CLICKS:
        raise StudyError('Too many prompt clicks')
    for x, y in [*positive, *negative]:
        if not (np.isfinite([x, y]).all() and 0 <= x <= width - 1 and 0 <= y <= height - 1):
            raise StudyError('Prompt click outside the original image')
    if box is not None:
        box = [float(v) for v in box]
        if len(box) != 4 or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise StudyError('Prompt box outside the original image or empty')
    if box is None and not positive:
        raise StudyError('A prompt box or at least one include click is required')
    return box, [list(map(float, p)) for p in positive], [list(map(float, p)) for p in negative]


def prompt_crop(box, positive, negative, width: int, height: int) -> tuple[int, int, int, int]:
    xs = [p[0] for p in positive + negative] + ([box[0], box[2]] if box else [])
    ys = [p[1] for p in positive + negative] + ([box[1], box[3]] if box else [])
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    margin = max(MIN_CROP_MARGIN_PX, CROP_MARGIN_FRACTION * span)
    return (int(max(0, np.floor(min(xs) - margin))), int(max(0, np.floor(min(ys) - margin))),
            int(min(width, np.ceil(max(xs) + margin + 1))), int(min(height, np.ceil(max(ys) + margin + 1))))


def mask_rings(mask: np.ndarray, box, positive, negative, keep_all: bool = False) -> tuple[list[dict], int]:
    """Keep the components tied to this prompt (or all of them); holes stay child rings.

    ``keep_all`` is for automatic masks, where the mask itself is the candidate and
    there is no prompt to select a component with.
    """
    mask = mask.astype(np.uint8)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    if keep_all:
        keep[1:] = True
    for x, y in positive:
        keep[labels[int(round(y)), int(round(x))]] = True
    if box is not None and not positive:
        x0, y0, x1, y1 = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3])))
        inside = labels[y0:y1, x0:x1]
        ids, pixels = np.unique(inside[inside > 0], return_counts=True)
        if len(ids):
            keep[ids[np.argmax(pixels)]] = True  # the dominant component inside the prompt box
    keep[0] = False
    for x, y in negative:
        keep[labels[int(round(y)), int(round(x))]] = False
    selected = keep[labels]
    discarded = int(count - 1 - keep[1:].sum())
    contours, hierarchy = cv2.findContours(selected.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    rings = []
    for index, contour in enumerate(contours):
        polygon = cv2.approxPolyDP(contour, CONTOUR_TOLERANCE_PX, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        rings.append(dict(points_px=polygon.astype(float).tolist(), is_hole=bool(hierarchy[0, index, 3] >= 0)))
    return rings, discarded


def bbox_from_rings(rings: list[dict]) -> list[float]:
    points = np.asarray([p for ring in rings if not ring['is_hole'] for p in ring['points_px']], dtype=float)
    return [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max() + 1), float(points[:, 1].max() + 1)]


def propose_automatic(segmenter, image_path: Path, image_sha256: str, image_file: str | None = None) -> dict:
    """Class-agnostic masks for one whole photograph, as UNASSIGNED drafts.

    Overlapping candidates are kept separate (they may be different leaflets) and small
    components are kept (real small leaves). Masks covering most of the frame are dropped
    as background. No candidate carries a leaflet identity, class or landmark.
    """
    info = segmenter.info()
    if not info.get('available'):
        raise StudyError('segmenter_unavailable: ' + str(info.get('reason')))
    if not info.get('automatic'):
        raise StudyError('automatic_masks_unavailable: this backend has no automatic mask generation')
    if file_hash(image_path) != image_sha256:
        raise StudyError('Source photograph changed')
    with Image.open(image_path) as image:
        if image.getexif().get(274, 1) != 1:
            raise StudyError('Normalize image orientation before segmentation')
        rgb = np.array(image.convert('RGB'))
    height, width = rgb.shape[:2]
    frame_area = float(height * width)
    output, dropped = [], dict(too_small=0, too_large=0, no_outer_ring=0)
    for index, candidate in enumerate(segmenter.automatic(rgb)):
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != (height, width):
            raise StudyError('Segmenter mask dimensions disagree with the photograph')
        area = int(mask.sum())
        if area < AUTOMATIC_MIN_AREA_PX:
            dropped['too_small'] += 1
            continue
        if area > AUTOMATIC_MAX_AREA_FRACTION * frame_area:
            dropped['too_large'] += 1
            continue
        rings, _ = mask_rings(mask, None, [], [], keep_all=True)
        if not any(not r['is_hole'] for r in rings):
            dropped['no_outer_ring'] += 1
            continue
        output.append(dict(candidate_index=index, score=candidate.score, rings=rings,
                           bbox_px=bbox_from_rings(rings), mask_area_px=area, discarded_components=0,
                           assigned_class=None, assigned_leaflet=None))
    output.sort(key=lambda c: c['mask_area_px'], reverse=True)
    truncated = max(0, len(output) - AUTOMATIC_MAX_MASKS)
    return dict(candidates=output[:AUTOMATIC_MAX_MASKS], image_size_px=[width, height],
                image_file=image_file, image_sha256=image_sha256, mode='automatic',
                model=segmenter.info(), dropped=dict(dropped, truncated=truncated),
                filters=dict(min_area_px=AUTOMATIC_MIN_AREA_PX, max_area_fraction=AUTOMATIC_MAX_AREA_FRACTION,
                             max_masks=AUTOMATIC_MAX_MASKS, overlapping_masks_kept_separate=True),
                coordinate_convention='rings: pixel centres; bbox_px: pixel edges',
                human_review='pending', candidate_class='unclassified',
                notice=('Class-agnostic mask candidates for review. A candidate may be a lobe, a whole compound '
                        'leaf, several leaflets or background. Scores are model confidence, not botanical accuracy.'))


def propose(segmenter, image_path: Path, image_sha256: str, prompts: dict) -> dict:
    """Return candidate proposals in original-image pixels for one prompt set."""
    if file_hash(image_path) != image_sha256:
        raise StudyError('Source photograph changed')
    with Image.open(image_path) as image:
        if image.getexif().get(274, 1) != 1:
            raise StudyError('Normalize image orientation before segmentation')
        rgb = np.array(image.convert('RGB'))
    height, width = rgb.shape[:2]
    box, positive, negative = checked_prompts(prompts, width, height)
    x0, y0, x1, y1 = prompt_crop(box, positive, negative, width, height)
    shift = np.array([x0, y0], dtype=float)
    local_box = None if box is None else [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
    local = lambda pts: [list(np.asarray(p) - shift) for p in pts]
    candidates = segmenter.predict(rgb[y0:y1, x0:x1], local_box, local(positive), local(negative))
    output = []
    for index, candidate in enumerate(candidates):
        if candidate.mask.shape != (y1 - y0, x1 - x0):
            raise StudyError('Segmenter mask dimensions disagree with the crop')
        rings, discarded = mask_rings(candidate.mask, local_box, local(positive), local(negative))
        if not any(not r['is_hole'] for r in rings):
            continue
        for ring in rings:
            ring['points_px'] = (np.asarray(ring['points_px']) + shift).tolist()
        output.append(dict(candidate_index=index, score=candidate.score, rings=rings, bbox_px=bbox_from_rings(rings),
                           mask_area_px=int(candidate.mask.sum()), discarded_components=discarded))
    return dict(candidates=output, crop_px=[x0, y0, x1, y1], image_size_px=[width, height],
                prompts=dict(box_px=box, positive_px=positive, negative_px=negative),
                model=segmenter.info(), coordinate_convention='rings: pixel centres; bbox_px/box_px: pixel edges',
                human_review='pending', notice='Prompted segmentation proposal; requires review. Score is model confidence, not accuracy.')


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write-sam2-config', type=Path, help='Write an example SAM 2 config JSON to this path and exit')
    args = parser.parse_args()
    if args.write_sam2_config:
        if args.write_sam2_config.exists():
            raise SystemExit('Config already exists; not overwritten')
        args.write_sam2_config.write_text(json.dumps(SAM2_CONFIG_EXAMPLE, indent=2) + '\n', encoding='utf-8')
        print(args.write_sam2_config)


if __name__ == '__main__':
    main()
