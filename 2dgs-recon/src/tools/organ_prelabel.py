"""Draft organ labels for photos nobody has labelled yet (top views, new captures): detector ensemble + flip TTA +
weighted box fusion, optionally a leaflet outline from the leaflet-tuned SAM (2026-09-25).

    python -X utf8 -m tools.organ_prelabel --models build/organ_detector_20260925/kout_v3/final_all_s*.pt \\
        --images build/new_captures_20260920/dwt_01/edit_session/images/dwt_01_20260920_01.png ... \\
        --out build/prelabel_<date> [--sam-config build/leaflet_pieces_20260925/sam2_config_tuned.json]

Output: prelabels.coco.json (original pixels, category names as in Roboflow's unified classes, score per box, source
'model draft') and one overlay JPEG per photo. These are DRAFTS for a person to correct in Roboflow; they are never
imported as human labels by this pipeline.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.leaf_light.common import StudyError, file_hash, read_json, write_json

CLASSES = ['compound_leaf', 'leaflet', 'sub_stem', 'main_stem', 'growth_shoot']
COLOURS = [(230, 126, 34), (46, 204, 113), (155, 89, 182), (241, 196, 15), (231, 76, 60)]
IMGSZ, CONF_MEMBER, IOU_FUSE, GAMMA = 1280, 0.01, 0.55, 0.4
#: Boxes below this fused score are not written (the dead-tree lesson: keep recall high, a person rejects extras).
CONF_OUT = 0.25


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def plant_window(rgb, pad=0.25):
    """Square window around the green plant pixels. The detector learnt from side/middle photos where the plant fills
    much of the frame; in a top view it is a small patch, and running on the whole frame yields many duplicate boxes."""
    f = rgb.astype(np.float32)
    exg = (2 * f[..., 1] - f[..., 0] - f[..., 2]) / (f.sum(-1) + 1e-6)
    ys, xs = np.nonzero((exg > 0.06) & (f[..., 1] > 40))
    H, W = rgb.shape[:2]
    if len(xs) < 500:
        return 0, 0, W, H
    x0, x1, y0, y1 = np.percentile(xs, 0.5), np.percentile(xs, 99.5), np.percentile(ys, 0.5), np.percentile(ys, 99.5)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return X0, Y0, int(min(W, X0 + side)), int(min(H, Y0 + side))


def predict(model, rgb, view):
    a = rgb[:, ::-1] if view == 'h' else rgb[::-1] if view == 'v' else rgb
    r = model.predict(np.ascontiguousarray(a[:, :, ::-1]), imgsz=IMGSZ, conf=CONF_MEMBER, iou=0.6, max_det=600, verbose=False)[0]
    H, W = rgb.shape[:2]
    xy = r.boxes.xyxy.cpu().numpy(); s = r.boxes.conf.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int)
    if view == 'h':
        xy[:, [0, 2]] = W - xy[:, [2, 0]]
    if view == 'v':
        xy[:, [1, 3]] = H - xy[:, [3, 1]]
    return [(int(ci), [float(x) for x in bi], float(si)) for bi, si, ci in zip(xy, s, c)]


#: Classes whose outline matters downstream (S3 fits stem/petiole axes to polygon centrelines, 2026-09-28): their
#: polygon comes from the first detector's own instance mask on the unflipped window, matched to the fused box.
MASK_CLASSES = ('main_stem', 'sub_stem')
MASK_MIN_IOU = 0.5


def instance_polygons(model, rgb, fused):
    """{fused index: polygon (N x 2, window pixels)} for fused boxes of MASK_CLASSES, from the detector's mask whose box
    overlaps the fused box by at least MASK_MIN_IOU. Boxes without such a mask keep no polygon (never a guessed one)."""
    r = model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), imgsz=IMGSZ, conf=CONF_MEMBER, iou=0.6, max_det=600,
                      retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return {}
    xy = r.boxes.xyxy.cpu().numpy(); c = r.boxes.cls.cpu().numpy().astype(int); polys = r.masks.xy
    out = {}
    for i, (ci, b, _) in enumerate(fused):
        if CLASSES[ci] not in MASK_CLASSES:
            continue
        cand = [(iou(b, list(xy[j])), j) for j in range(len(xy)) if c[j] == ci and len(polys[j]) >= 3]
        if cand:
            best, j = max(cand)
            if best >= MASK_MIN_IOU:
                out[i] = np.asarray(polys[j], float)
    return out


def wbf(members, n):
    out = []
    for c in range(len(CLASSES)):
        boxes = sorted([(b, s, m) for m, ps in enumerate(members) for ci, b, s in ps if ci == c], key=lambda t: -t[1])
        clusters = []
        for b, s, m in boxes:
            for cl in clusters:
                if iou(cl['box'], b) > IOU_FUSE:
                    cl['items'].append((b, s, m)); w = np.array([x[1] for x in cl['items']])
                    cl['box'] = list((np.array([x[0] for x in cl['items']]) * w[:, None]).sum(0) / w.sum()); break
            else:
                clusters.append(dict(box=list(b), items=[(b, s, m)]))
        for cl in clusters:
            sc = np.mean([x[1] for x in cl['items']]) * (len({x[2] for x in cl['items']}) / n) ** GAMMA
            out.append((c, cl['box'], float(sc)))
    return out


#: Per-class scale plan from the v4 comparison (2026-09-29, leave-one-plant-out on full photos): the whole-photo model
#: wins for compound leaves, leaflets and main stems; 2x2 tiles of the plant window double growth-shoot AP (0.115 ->
#: 0.228); petioles are best with both fused (0.439 -> 0.464). Tiles hurt large organs (compound 0.33), so only these
#: classes are taken from tiles. The tile geometry must match training (organ_detector_dataset TILE_OVERLAP).
TILE_ONLY_CLASSES = ('growth_shoot',)
TILE_FUSE_CLASSES = ('sub_stem',)
TILE_OVERLAP = 0.2


def tiles_of(window) -> list[tuple[int, int, int]]:
    X0, Y0, X1, Y1 = window
    t = int(round((X1 - X0) * (1 + TILE_OVERLAP) / 2))
    return [(ox, oy, t) for ox in (X0, X1 - t) for oy in (Y0, Y1 - t)]


def merge_same_class(boxes, thr=IOU_FUSE):
    """Score-weighted fusion of one class's boxes from overlapping tiles / scales (one merged box per cluster)."""
    out, cl = [], []
    for b, s, poly in sorted(boxes, key=lambda t: -t[1]):
        for k in cl:
            if iou(k['box'], b) > thr:
                k['items'].append((b, s)); w = np.array([x[1] for x in k['items']])
                k['box'] = list((np.array([x[0] for x in k['items']]) * w[:, None]).sum(0) / w.sum())
                break
        else:
            cl.append(dict(box=list(b), items=[(b, s)], poly=poly))
    return [(k['box'], max(x[1] for x in k['items']), k['poly']) for k in cl]


def tile_predictions(nets, rgb, window, views, stem_masks: bool):
    """Class-wise boxes (full-photo pixels) from the tile models on the 2x2 tiles of the plant window."""
    by_class = {}
    for ox, oy, t in tiles_of(window):
        sub = np.ascontiguousarray(rgb[oy:oy + t, ox:ox + t])
        fused = [f for f in wbf([predict(m, sub, v) for m in nets for v in views], len(nets) * len(views)) if f[2] >= CONF_OUT]
        polys = instance_polygons(nets[0], sub, fused) if stem_masks else {}
        for i, (c, b, s) in enumerate(fused):
            poly = polys.get(i)
            by_class.setdefault(c, []).append(([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy], s, None if poly is None else poly + [ox, oy]))
    return {c: merge_same_class(v) for c, v in by_class.items()}


def run(models: list[Path], images: list[Path], out: Path, sam_config: Path | None = None, views=('orig', 'h', 'v'),
        crop: bool = True, stem_masks: bool = True, tile_models: list[Path] | None = None, smoothing: str = 'none') -> dict:
    """smoothing: a tools.preprocess_study recipe (Lab L only) applied to the DETECTOR input only; SAM and the saved
    drafts use the original pixels. Codex pilot 09-30 (dwt_01, 4 photos, held-out models): mean3/gauss5 cut compound
    false positives 10 -> 5 and misses 3 -> 0; masks unchanged."""
    from ultralytics import YOLO
    from tools.preprocess_study import process_rgb
    out = Path(out)
    if out.exists():
        raise StudyError(f'Output exists: {out}')
    out.mkdir(parents=True)
    nets = [YOLO(str(m)) for m in models]
    tile_nets = [YOLO(str(m)) for m in (tile_models or [])]
    seg = None
    if sam_config:
        from tools.leaflet_segmenter import load_segmenter
        seg = load_segmenter('sam2', sam_config)
    coco = dict(images=[], annotations=[], categories=[dict(id=i + 1, name=c) for i, c in enumerate(CLASSES)],
                info=dict(source='model draft (tools.organ_prelabel)', models=[dict(path=str(m), sha256=file_hash(m)) for m in models],
                          views=list(views), conf_out=CONF_OUT, plant_window=crop, gamma=GAMMA, iou_fuse=IOU_FUSE,
                          tile_models=[dict(path=str(m), sha256=file_hash(m)) for m in (tile_models or [])],
                          scale_plan=dict(tiles_only=list(TILE_ONLY_CLASSES), fused=list(TILE_FUSE_CLASSES)) if tile_models else None,
                          detector_smoothing=smoothing,
                          note='drafts for human correction; not human labels'))
    for n, img_path in enumerate(images, 1):
        rgb = np.asarray(Image.open(img_path).convert('RGB'))
        det_rgb = process_rgb(rgb, smoothing) if smoothing != 'none' else rgb  # detector input only
        H, W = rgb.shape[:2]
        X0, Y0, X1, Y1 = plant_window(rgb) if crop else (0, 0, W, H)
        sub = np.ascontiguousarray(det_rgb[Y0:Y1, X0:X1])
        fused = wbf([predict(m, sub, v) for m in nets for v in views], len(nets) * len(views))
        fused = [f for f in fused if f[2] >= CONF_OUT]
        polys = instance_polygons(nets[0], sub, fused) if stem_masks else {}
        polys = {id(fused[i]): p + [X0, Y0] for i, p in polys.items()}
        fused = [(c, [b[0] + X0, b[1] + Y0, b[2] + X0, b[3] + Y0], s, polys.get(id(f))) for f in fused for c, b, s in [f]]
        if tile_nets:  # per-class scale plan (see TILE_ONLY_CLASSES / TILE_FUSE_CLASSES)
            tiled = tile_predictions(tile_nets, det_rgb, plant_window(rgb), views, stem_masks)
            keep = [f for f in fused if CLASSES[f[0]] not in TILE_ONLY_CLASSES + TILE_FUSE_CLASSES]
            for name in TILE_ONLY_CLASSES + TILE_FUSE_CLASSES:
                c = CLASSES.index(name)
                boxes = tiled.get(c, [])
                if name in TILE_FUSE_CLASSES:
                    boxes = merge_same_class(boxes + [(b, s, p) for cc, b, s, p in fused if cc == c])
                keep += [(c, b, s, p) for b, s, p in boxes if s >= CONF_OUT]
            fused = keep
        coco['images'].append(dict(id=n, file_name=Path(img_path).name, width=W, height=H, path=str(img_path)))
        im = Image.fromarray(rgb).convert('RGB'); dr = ImageDraw.Draw(im)
        for c, b, s, poly in sorted(fused, key=lambda t: t[0]):
            ann = dict(id=len(coco['annotations']) + 1, image_id=n, category_id=c + 1, bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]],
                       area=(b[2] - b[0]) * (b[3] - b[1]), iscrowd=0, score=round(s, 4))
            if poly is not None:
                ann['segmentation'] = [np.clip(poly, 0, [W, H]).ravel().tolist()]
                ann['segmentation_source'] = 'detector instance mask'
                dr.line([tuple(p) for p in poly] + [tuple(poly[0])], fill=COLOURS[c], width=4)
            if seg is not None and CLASSES[c] == 'leaflet':
                # crop around the box as the editor does: the image encoder runs on the crop, not the whole photo
                side = max(256, 1.6 * max(b[2] - b[0], b[3] - b[1]))
                x0, y0 = int(max(0, (b[0] + b[2] - side) / 2)), int(max(0, (b[1] + b[3] - side) / 2))
                x1, y1 = int(min(W, x0 + side)), int(min(H, y0 + side))
                cands = seg.predict(np.ascontiguousarray(rgb[y0:y1, x0:x1]), [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0], [], [])
                if cands:
                    m = max(cands, key=lambda k: k.score).mask
                    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cs:
                        poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 2.0, True).reshape(-1, 2) + [x0, y0]
                        if len(poly) >= 3:
                            ann['segmentation'] = [poly.astype(float).ravel().tolist()]
                            dr.line([tuple(p) for p in poly] + [tuple(poly[0])], fill=COLOURS[c], width=4)
            coco['annotations'].append(ann)
            dr.rectangle(b, outline=COLOURS[c], width=5)
            dr.text((b[0] + 4, b[1] + 2), f'{CLASSES[c]} {s:.2f}', fill=COLOURS[c])
        im.thumbnail((1400, 1400)); im.save(out / f'{Path(img_path).stem}_prelabel.jpg', quality=85)
    write_json(out / 'prelabels.coco.json', coco)
    counts = {}
    for a in coco['annotations']:
        counts[CLASSES[a['category_id'] - 1]] = counts.get(CLASSES[a['category_id'] - 1], 0) + 1
    return dict(out=str(out), images=len(images), boxes=len(coco['annotations']), per_class=counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sam-config', type=Path)
    ap.add_argument('--views', default='orig,h,v')
    ap.add_argument('--no-crop', action='store_true', help='run on the whole frame instead of the plant window')
    ap.add_argument('--tile-models', nargs='*', default=[], help='2x2-tile models for the small classes (v4 scale plan)')
    a = ap.parse_args()
    models = [Path(p) for g in a.models for p in sorted(glob.glob(g))]
    images = [Path(p) for g in a.images for p in sorted(glob.glob(g))]
    if not models or not images:
        raise StudyError('No models or images matched')
    tiles = [Path(p) for g in a.tile_models for p in sorted(glob.glob(g))]
    print(json.dumps(run(models, images, a.out, a.sam_config, tuple(a.views.split(',')), not a.no_crop, tile_models=tiles or None),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
