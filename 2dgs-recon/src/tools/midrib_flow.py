"""SG input -> leaflet mask -> mask on the original photo -> Gaussian midrib path (user flow, 2026-09-30).

    python -X utf8 -m tools.midrib_flow --plant dwt_01 --views 11 18 24 25 --out build/midrib_flow_<date>
        [--sam-config build/leaflets5_20260929/sam2_config_v5_heldout_dwt_01.json] [--smoothing sg5]

Per human-reviewed leaflet observation (its box is the prompt, its accepted mask is the reference):

1. segmentation input: the full photo smoothed with a tools.preprocess_study recipe (default sg5 = separable
   Savitzky-Golay window 5, degree 2, Lab L only) -> the same crop rule as the editor -> SAM (held-out tuned model,
   highest-score candidate). The same crop is also run on the original pixels, so SG vs original is compared.
2. the mask goes back to the original photo: no warp or resize happened, so it is the crop offset only; its IoU with
   the person's accepted mask is recorded.
3. midrib on the ORIGINAL pixels (not the smoothed ones): bright-ridge response of the Lab L channel from Gaussian
   second derivatives at sigma 1-3 px (Hessian, bright line = strongly negative curvature across it), computed on the
   crop with real context (the background is not zeroed), then kept only inside the mask eroded by the largest filter
   radius so the leaf edge does not count as a line.
4. path: the two ends of the mask along its long axis (PCA) are joined by the cheapest path through
   cost = 1 / (ridge + eps) inside the eroded mask (Dijkstra on the 8-neighbour grid). Reported per leaflet: the
   polyline in original pixels, mean ridge response along it vs along the straight chord, and length / chord.

Everything is a PROPOSAL for a person: a Hessian line can be a vein, a fold or a shadow edge, and the ends of the long
axis are not a verified base and tip. No anatomical accuracy or 3D is claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from tools import leaflet_segmenter as seg
from tools.leaf_light.common import StudyError, file_hash, read_json, write_json
from tools.preprocess_study import process_rgb, recipe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name('plant_pipeline_plants.json')
DEFAULT_SAM = 'build/leaflets5_20260929/sam2_config_v5_heldout_{plant}.json'
SIGMAS = (1.0, 2.0, 3.0)
EDGE_MARGIN_PX = 7          # ~ 2.5 x the largest sigma: the lamina edge must not read as a line
PATH_EPS = 0.05
MAX_PATH_SIDE = 700         # crops longer than this are halved for the path search (the ridge map stays full size)


def bright_ridge(gray: np.ndarray, sigmas=SIGMAS) -> np.ndarray:
    """Max over scales of sigma^2 * max(0, -lambda_min) of the Hessian: high on thin bright lines."""
    g = gray.astype(np.float32)
    out = np.zeros_like(g)
    for s in sigmas:
        xx = ndimage.gaussian_filter(g, s, order=(0, 2))
        yy = ndimage.gaussian_filter(g, s, order=(2, 0))
        xy = ndimage.gaussian_filter(g, s, order=(1, 1))
        tmp = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
        lam_min = (xx + yy - tmp) / 2
        out = np.maximum(out, (s ** 2) * np.maximum(0.0, -lam_min))
    return out


def long_axis_ends(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    t = (pts - c) @ vt[0]
    a, b = pts[t.argmin()], pts[t.argmax()]
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def cheapest_path(cost: np.ndarray, allowed: np.ndarray, a, b) -> list[tuple[int, int]] | None:
    """Dijkstra over the allowed pixels (8-neighbours, step cost = mean pixel cost x step length)."""
    H, W = cost.shape
    idx = -np.ones((H, W), np.int64)
    ys, xs = np.nonzero(allowed)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), H - max(0, dy)
        x0, x1 = max(0, -dx), W - max(0, dx)
        src = idx[y0:y1, x0:x1]
        dst = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        ok = (src >= 0) & (dst >= 0)
        w = (cost[y0:y1, x0:x1][ok] + cost[y0 + dy:y1 + dy, x0 + dx:x1 + dx][ok]) / 2 * np.hypot(dy, dx)
        rows += [src[ok], dst[ok]]; cols += [dst[ok], src[ok]]; vals += [w, w]
    n = len(ys)
    graph = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    ia, ib = idx[a[1], a[0]], idx[b[1], b[0]]
    if ia < 0 or ib < 0:
        return None
    _, pred = dijkstra(graph, indices=ia, return_predecessors=True)
    if pred[ib] < 0 and ia != ib:
        return None
    path, k = [], ib
    while k != ia:
        path.append((int(xs[k]), int(ys[k])))
        k = pred[k]
    path.append((int(xs[ia]), int(ys[ia])))
    return path[::-1]


def nearest_inside(allowed: np.ndarray, p):
    ys, xs = np.nonzero(allowed)
    k = np.argmin((xs - p[0]) ** 2 + (ys - p[1]) ** 2)
    return int(xs[k]), int(ys[k])


def midrib(rgb_crop: np.ndarray, mask: np.ndarray) -> dict:
    L = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2LAB)[..., 0]
    r = bright_ridge(L)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * EDGE_MARGIN_PX + 1,) * 2, np.uint8)).astype(bool)
    if inner.sum() < 30:
        return dict(status='unavailable', reason='mask too thin after removing the edge band')
    scale = float(np.percentile(r[inner], 99)) or 1.0
    rn = np.clip(r / scale, 0, 1)
    a, b = (nearest_inside(inner, p) for p in long_axis_ends(mask))
    f = 1
    if max(mask.shape) > MAX_PATH_SIDE:
        f = 2
    cost = 1.0 / (rn[::f, ::f] + PATH_EPS)
    path = cheapest_path(cost, inner[::f, ::f], (a[0] // f, a[1] // f), (b[0] // f, b[1] // f))
    if path is None:
        return dict(status='unavailable', reason='the two ends are not connected inside the eroded mask')
    path = np.array(path) * f
    along = rn[path[:, 1], path[:, 0]]
    n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1])))
    chord = np.column_stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]).round().astype(int)
    chord_ok = inner[chord[:, 1], chord[:, 0]]
    on_chord = rn[chord[chord_ok, 1], chord[chord_ok, 0]]
    length = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    chord_len = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    return dict(status='proposed', ends_px=[list(a), list(b)], path_px=path.tolist(), ridge_norm_scale=scale,
                mean_ridge_on_path=round(float(along.mean()), 3), mean_ridge_on_chord=round(float(on_chord.mean()), 3) if len(on_chord) else None,
                path_over_chord=round(length / max(chord_len, 1.0), 3), length_px=round(length, 1), ridge=rn)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def run(plant: str, views: list[str], out: Path, sam_config: Path | None = None, smoothing: str = 'sg5',
        limit: int | None = None) -> dict:
    out = Path(out)
    if out.exists():
        raise StudyError(f'Output exists: {out}')
    recipe(smoothing)
    pc = read_json(CONFIG)['plants'][plant]
    session = ROOT / pc['capture'] / 'edit_session'
    frames = {f['image_file']: f for f in read_json(session / 'capture.json')['frames']}
    exp = sorted((ROOT / pc['batch'] / 'exports').glob('accepted_*'))[-1]
    acc = read_json(exp / 'accepted.json')
    sam_config = Path(sam_config or DEFAULT_SAM.format(plant=plant))
    sam = seg.load_segmenter('sam2', ROOT / sam_config if not sam_config.is_absolute() else sam_config)
    (out / 'overlays').mkdir(parents=True)
    rows, per_obs = [], []
    obs = [o for o in acc['observations'] if o.get('organ_class', 'leaflet') == 'leaflet' and o['review'] == 'accepted_by_user'
           and Path(o['image_file']).stem[-2:] in views]
    obs.sort(key=lambda o: (o['image_file'], o['id']))
    if limit:
        obs = obs[:limit]
    cur, rgb, rgb_s = None, None, None
    for o in obs:
        if o['image_file'] != cur:
            cur = o['image_file']
            rgb = np.asarray(Image.open(session / frames[cur]['image_path']).convert('RGB'))
            rgb_s = process_rgb(rgb, smoothing)  # full frame first, then the same crop (Codex protocol)
        H, W = rgb.shape[:2]
        box = [float(v) for v in o['prompts']['box_px']]
        x0, y0, x1, y1 = seg.prompt_crop(box, [], [], W, H)
        rel = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
        human = np.asarray(Image.open(exp / o['mask_file'])) > 0
        res = {}
        for name, src in (('sg', rgb_s), ('original', rgb)):
            cands = sam.predict(np.ascontiguousarray(src[y0:y1, x0:x1]), rel, [], [])
            m = max(cands, key=lambda c: c.score)
            full = np.zeros((H, W), bool)
            full[y0:y1, x0:x1] = m.mask          # back on the original photo: crop offset only
            res[name] = dict(mask=full, score=float(m.score), iou_vs_person=round(iou(full, human), 4))
        crop_mask = res['sg']['mask'][y0:y1, x0:x1]
        mr = midrib(rgb[y0:y1, x0:x1], crop_mask) if crop_mask.any() else dict(status='unavailable', reason='empty SG mask')
        path_full = [[x + x0, y + y0] for x, y in mr.get('path_px', [])]
        row = dict(observation=o['id'], photo=Path(cur).stem[-2:], sg_iou_vs_person=res['sg']['iou_vs_person'],
                   original_iou_vs_person=res['original']['iou_vs_person'], sg_score=round(res['sg']['score'], 4),
                   midrib_status=mr['status'], mean_ridge_on_path=mr.get('mean_ridge_on_path'),
                   mean_ridge_on_chord=mr.get('mean_ridge_on_chord'), path_over_chord=mr.get('path_over_chord'),
                   length_px=mr.get('length_px'))
        rows.append(row)
        per_obs.append(dict(row, crop_xyxy=[x0, y0, x1, y1], midrib_path_original_px=path_full,
                            ends_original_px=[[x + x0, y + y0] for x, y in mr.get('ends_px', [])], reason=mr.get('reason')))
        _overlay(out / 'overlays' / f"{o['id']}.jpg", rgb[y0:y1, x0:x1], human[y0:y1, x0:x1], crop_mask, mr)
    with open(out / 'midrib_summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r['midrib_status'] == 'proposed']
    d = np.array([r['sg_iou_vs_person'] - r['original_iou_vs_person'] for r in rows])
    summary = dict(schema='midrib-flow-1', plant=plant, views=views, smoothing=recipe(smoothing), sam_config=str(sam_config),
                   sam_checkpoint_sha256=read_json(ROOT / sam_config if not sam_config.is_absolute() else sam_config)['checkpoint_sha256'],
                   accepted_export=str(exp.relative_to(ROOT)), accepted_sha256=file_hash(exp / 'accepted.json'),
                   observations=len(rows), masks=dict(sg_mean_iou=round(float(np.mean([r['sg_iou_vs_person'] for r in rows])), 4),
                                                       original_mean_iou=round(float(np.mean([r['original_iou_vs_person'] for r in rows])), 4),
                                                       sg_minus_original_mean=round(float(d.mean()), 4),
                                                       sg_better=int((d > 0.005).sum()), sg_worse=int((d < -0.005).sum())),
                   midrib=dict(proposed=len(ok), unavailable=len(rows) - len(ok),
                               median_ridge_path=round(float(np.median([r['mean_ridge_on_path'] for r in ok])), 3) if ok else None,
                               median_ridge_chord=round(float(np.median([r['mean_ridge_on_chord'] for r in ok if r['mean_ridge_on_chord'] is not None])), 3) if ok else None,
                               median_path_over_chord=round(float(np.median([r['path_over_chord'] for r in ok])), 3) if ok else None),
                   ridge=dict(channel='Lab L of the ORIGINAL pixels', sigmas_px=list(SIGMAS), polarity='bright line',
                              edge_margin_px=EDGE_MARGIN_PX, path='Dijkstra between long-axis ends, cost 1/(ridge+0.05)'),
                   validated_metric=False, note='midrib paths are proposals: no independent midrib reference exists yet')
    write_json(out / 'summary.json', summary)
    write_json(out / 'midribs.json', dict(schema='midrib-flow-1', coordinates='original image pixels (x right, y down)', leaflets=per_obs))
    _sheet(out)
    return summary


def _overlay(path: Path, crop: np.ndarray, human: np.ndarray, sgm: np.ndarray, mr: dict) -> None:
    both_masks = human | sgm
    if both_masks.any():  # zoom to the leaflet (+24 px) so a small leaflet is readable
        ys, xs = np.nonzero(both_masks)
        y0, y1 = max(0, ys.min() - 24), min(crop.shape[0], ys.max() + 25)
        x0, x1 = max(0, xs.min() - 24), min(crop.shape[1], xs.max() + 25)
        crop, human, sgm = crop[y0:y1, x0:x1], human[y0:y1, x0:x1], sgm[y0:y1, x0:x1]
        mr = dict(mr)
        if 'ridge' in mr:
            mr['ridge'] = mr['ridge'][y0:y1, x0:x1] * (sgm > 0)
        if mr.get('path_px'):
            mr['path_px'] = [[x - x0, y - y0] for x, y in mr['path_px']]
            mr['ends_px'] = [[x - x0, y - y0] for x, y in mr['ends_px']]
    left = Image.fromarray(np.ascontiguousarray(crop)).convert('RGB')
    dr = ImageDraw.Draw(left)
    for m, colour in ((human, (255, 255, 255)), (sgm, (60, 200, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if len(c) > 2:
                dr.line([tuple(p) for p in c.reshape(-1, 2)] + [tuple(c.reshape(-1, 2)[0])], fill=colour, width=2)
    if mr.get('path_px'):
        dr.line([tuple(p) for p in mr['path_px']], fill=(255, 60, 60), width=3)
        for p in mr['ends_px']:
            dr.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=(255, 220, 0), width=2)
    right = Image.fromarray((np.clip(mr['ridge'], 0, 1) * 255).astype(np.uint8)).convert('RGB') if 'ridge' in mr else \
        Image.new('RGB', left.size, 'black')
    both = Image.new('RGB', (left.width * 2 + 6, left.height), 'black')
    both.paste(left, (0, 0)); both.paste(right, (left.width + 6, 0))
    if both.width < 600:
        both = both.resize((600, int(both.height * 600 / both.width)))
    both.thumbnail((1000, 1000))
    both.save(path, quality=88)


def _sheet(out: Path) -> None:
    items = sorted((out / 'overlays').glob('*.jpg'))[:40]
    if not items:
        return
    tiles = [Image.open(p) for p in items]
    for t in tiles:
        t.thumbnail((480, 480))
    cols = 4
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new('RGB', (cols * w, ((len(tiles) + cols - 1) // cols) * h), 'black')
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * w, (k // cols) * h))
    sheet.save(out / 'contact_sheet.jpg', quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--plant', required=True)
    ap.add_argument('--views', nargs='+', required=True, help='two-digit view codes, e.g. 11 18 24 25')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--sam-config', type=Path)
    ap.add_argument('--smoothing', default='sg5')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    s = run(a.plant, a.views, a.out, a.sam_config, a.smoothing, a.limit)
    print(json.dumps({k: s[k] for k in ('observations', 'masks', 'midrib')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
