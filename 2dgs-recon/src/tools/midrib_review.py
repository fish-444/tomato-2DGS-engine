"""Classical 2D path proposals inside fixed reviewed masks; no model inference."""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy import ndimage
from tools.midrib_flow import bright_ridge, cheapest_path

OPTIONS = dict(sigma_px=2.0, channel='original_l', polarity='bright')
MAX_ANCHORS = 12
MAX_GRAPH_PIXELS = 500_000
PATH_EPSILON = 0.05


def parse_options(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - set(OPTIONS):
        raise ValueError('Unknown midrib options')
    cfg = dict(OPTIONS, **value)
    if isinstance(cfg['sigma_px'], bool):
        raise ValueError('Invalid sigma')
    cfg['sigma_px'] = float(cfg['sigma_px'])
    if not math.isfinite(cfg['sigma_px']) or not 0.5 <= cfg['sigma_px'] <= 4:
        raise ValueError('Sigma must be 0.5..4 original pixels')
    if cfg['channel'] not in ('original_l', 'processed_l', 'red', 'green', 'blue'):
        raise ValueError('Unknown midrib channel')
    if cfg['polarity'] not in ('bright', 'dark'):
        raise ValueError('Unknown ridge polarity')
    return cfg


def propose(rgb: np.ndarray, mask: np.ndarray, processed: np.ndarray,
            anchors_original_px: list, crop_xyxy_px: list, options: dict) -> dict:
    cfg = parse_options(options)
    if mask.shape != rgb.shape[:2] or processed.shape != rgb.shape or not mask.any():
        raise ValueError('Nonempty reviewed mask and matching images required')
    if not isinstance(anchors_original_px, list) or len(anchors_original_px) > MAX_ANCHORS:
        raise ValueError('Use at most 12 ordered anchors')
    x0, y0, x1, y1 = crop_xyxy_px
    if (y1-y0, x1-x0) != mask.shape:
        raise ValueError('Crop dimensions disagree')
    points = []
    for p in anchors_original_px:
        if not isinstance(p, list) or len(p) != 2 or any(isinstance(v, bool) for v in p):
            raise ValueError('Each anchor must be an original-pixel x,y pair')
        a = np.asarray(p, dtype=float)
        if not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
            raise ValueError('Anchors must be finite integer pixel centres')
        x, y = int(a[0])-x0, int(a[1])-y0
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            raise ValueError('Anchor is outside this original-image crop')
        points.append((x, y))
    if len(set(points)) != len(points):
        raise ValueError('Repeated anchors are not allowed')

    source = processed if cfg['channel'] == 'processed_l' else rgb
    valid_input = (rgb.max(axis=2) < 250) & (rgb.sum(axis=2) > 0)
    if cfg['channel'].endswith('_l'):
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)/255
        valid_input &= (source.max(axis=2) < 250) & (source.sum(axis=2) > 0)
    else:
        values = rgb.astype(np.float32)
        gray = np.divide(values[..., ('red','green','blue').index(cfg['channel'])],
                         values.sum(axis=2), out=np.zeros(mask.shape, np.float32),
                         where=values.sum(axis=2) > 0)
    # Full scipy Gaussian support, including crop edge and mask holes. Never filter a zero-filled leaf.
    radius_px = math.ceil(4*cfg['sigma_px'])
    allowed = ndimage.binary_erosion(mask & valid_input,
                                    structure=np.ones((2*radius_px+1,)*2, bool), border_value=0)
    # Finite sampled derivative kernels have small DC residuals; a constant field must not become a ridge.
    centered = gray-float(gray.mean())
    response = bright_ridge(centered if cfg['polarity']=='bright' else -centered, [cfg['sigma_px']])
    scale = float(np.percentile(response[allowed], 99)) if allowed.any() else 0.0
    evidence = np.clip(response/scale, 0, 1) if scale > 1e-8 else np.zeros(mask.shape, np.float32)
    evidence[~allowed] = 0
    result = dict(options=cfg, anchors_original_px=anchors_original_px, path_original_px=[],
                  status='needs_anchors', reason='기부와 끝을 잎 내부에 순서대로 찍어 주세요.',
                  support_radius_px=radius_px, usable_percent=float(100*allowed.sum()/mask.sum()),
                  ridge_normalization_scale=scale, gray_mean_removed=float(gray.mean()), valid=allowed, evidence=evidence,
                  path_length_px=None, mean_ridge_on_path=None)
    if not allowed.any():
        result.update(status='unavailable', reason='경계·구멍을 제외하면 분석 영역이 없습니다. 선 굵기를 줄여 보세요.')
        return result
    for i, (x,y) in enumerate(points):
        if not allowed[y,x]:
            result.update(status='unavailable', reason=f'{i+1}번 점이 경계·구멍·밝기 한계 영역에 있습니다. 안쪽으로 다시 찍어 주세요.')
            return result
    if len(points) < 2:
        return result
    if scale <= 1e-8:
        result.update(status='unavailable', reason='선 구조의 밝기 근거가 부족합니다.')
        return result
    labels, _ = ndimage.label(allowed, structure=np.ones((3,3)))
    component = labels[points[0][1],points[0][0]]
    if any(labels[y,x] != component for x,y in points):
        result.update(status='unavailable', reason='점들이 가림·구멍으로 분리되어 있습니다. 보이는 연속 구간만 지정해 주세요.')
        return result
    connected = labels == component
    if connected.sum() > MAX_GRAPH_PIXELS:
        raise ValueError('Path region exceeds the bounded CPU pilot size; use a smaller observation')
    path = []
    for a,b in zip(points, points[1:]):
        segment = cheapest_path(1/(evidence+PATH_EPSILON), connected, a, b)
        if segment is None:
            result.update(status='unavailable', reason='두 점 사이의 경로를 찾지 못했습니다.')
            return result
        # Do not silently cross even a single forbidden diagonal corner.
        if any(x!=u and y!=v and (not connected[y,u] or not connected[v,x])
               for (x,y),(u,v) in zip(segment,segment[1:])):
            result.update(status='unavailable', reason='경로가 제외 영역의 모서리를 가로지릅니다. 점 위치를 조정해 주세요.')
            return result
        path.extend(segment if not path else segment[1:])
    a = np.asarray(path)
    result.update(status='proposed', reason='2D 주맥 후보입니다. 원본에서 주름·그림자와 구분해 확인하세요.',
                  path_original_px=[[int(x+x0),int(y+y0)] for x,y in path],
                  path_length_px=float(np.linalg.norm(np.diff(a,axis=0),axis=1).sum()),
                  mean_ridge_on_path=float(evidence[a[:,1],a[:,0]].mean()))
    return result
