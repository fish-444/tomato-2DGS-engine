"""Reproducible full-frame photometric preprocessing and scalar-map diagnostics."""
from __future__ import annotations

import hashlib
import json
import math

import cv2
import numpy as np
from scipy.signal import savgol_filter

SMOOTHING = {
    'none': dict(radius=0),
    'mean3': dict(radius=1, kernel=3),
    'gauss5': dict(radius=2, kernel=5, sigma=0.8),
    'gauss9': dict(radius=4, kernel=9, sigma=1.2),
    'sg5': dict(radius=2, window=5, degree=2),
    'sg7': dict(radius=3, window=7, degree=2),
    'bilateral12': dict(radius=2, diameter=5, strength=12, sigma_space=2),
}
TONES = ('none', 'gamma085', 'gamma115', 'clahe15')
RIDGE_WIDTHS = (1., 2., 3.)
RIDGE_RADIUS = 13  # Gaussian truncated at 4 sigma, then 3x3 derivatives.


def recipe(smoothing='none', tone='none'):
    if smoothing not in SMOOTHING or tone not in TONES:
        raise ValueError('Unknown study condition')
    return dict(schema='preprocess-study-1', smoothing=smoothing,
                smoothing_parameters=SMOOTHING[smoothing], tone=tone, ev=0,
                rgb_smoothing='Lab L only, a/b retained',
                scalar_order='original scalar map then smooth',
                ratio_basis='stored RGB', gamma_basis='linear sRGB',
                border='reflect101 / SG mirror', ridge_widths=list(RIDGE_WIDTHS),
                ridge_gaussian_radius='ceil(4*sigma)', ridge_derivative_kernel=3)


def smooth(values, method, scale=1.):
    """Float32 values; preserve SG overshoot. Strength 12 is in 0..255 units."""
    spec = recipe(method)['smoothing_parameters']
    a = np.asarray(values, np.float32)
    if method == 'none':
        return a.copy()
    if method == 'mean3':
        return cv2.blur(a, (3, 3), borderType=cv2.BORDER_REFLECT_101)
    if method.startswith('gauss'):
        return cv2.GaussianBlur(a, (spec['kernel'],)*2, spec['sigma'], borderType=cv2.BORDER_REFLECT_101)
    if method.startswith('sg'):
        return savgol_filter(savgol_filter(a, spec['window'], 2, axis=1, mode='mirror'),
                             spec['window'], 2, axis=0, mode='mirror').astype(np.float32)
    return cv2.bilateralFilter(a, 5, 12 * scale / 255, 2, borderType=cv2.BORDER_REFLECT_101)


def process_rgb(rgb, smoothing='none', tone='none'):
    recipe(smoothing, tone)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    if smoothing == 'none' and tone == 'none':
        return rgb.copy()
    result = rgb.copy()
    if smoothing != 'none':
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[..., 0] = np.rint(np.clip(smooth(lab[..., 0], smoothing, 255), 0, 255)).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if tone.startswith('gamma'):
        from tools.leaf_preprocess_lab import linear_rgb, encode_rgb
        result = encode_rgb(linear_rgb(result) ** (0.85 if tone == 'gamma085' else 1.15))
    elif tone == 'clahe15':
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[..., 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result


def scalar_maps(rgb):
    from tools.leaf_preprocess_lab import linear_rgb, full_channel_fraction
    maps = {'luminance': linear_rgb(rgb) @ np.array([.2126, .7152, .0722], np.float32)}
    maps.update({c: full_channel_fraction(rgb, c)[0] for c in ('red', 'green', 'blue')})
    return maps


def interior(valid, radius):
    return cv2.erode(valid.astype(np.uint8), np.ones((2*radius+1,)*2, np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def scalar_result(values, mask, method, common_radius=4, ridge=False):
    """Filter original context, never black-filled masks. Invalid footprints are excluded."""
    if values.shape != mask.shape or mask.dtype != bool:
        raise ValueError('Scalar/mask geometry mismatch')
    finite = np.isfinite(values)
    radius = SMOOTHING[method]['radius']
    extra = RIDGE_RADIUS if ridge else 0
    valid = interior(mask & finite, radius + extra)
    common = interior(mask & finite, max(common_radius, radius) + extra)
    filtered = smooth(np.where(finite, values, 0), method)
    # Numerics outside valid support are explicitly unavailable, including SG NaN neighbours.
    filtered[~valid] = np.nan
    vals = filtered[common]
    stats = dict(mask_pixels=int(mask.sum()), valid_pixels=int(valid.sum()), common_pixels=int(common.sum()),
                 common_percent=float(100*common.sum()/mask.sum()) if mask.any() else 0.,
                 nonfinite_source_pixels=int((mask & ~finite).sum()),
                 boundary_hole_or_invalid_excluded=int((mask & ~common).sum()),
                 minimum=float(vals.min()) if vals.size else None,
                 maximum=float(vals.max()) if vals.size else None,
                 out_of_range_count=int(((vals<0)|(vals>1)).sum()),
                 out_of_range_percent=float(100*np.mean((vals<0)|(vals>1))) if vals.size else None)
    return filtered, valid, common, stats


def display_scalar(values, valid=None):
    finite = np.isfinite(values)
    gray = np.rint(np.clip(np.nan_to_num(values), 0, 1)*255).astype(np.uint8)
    display = np.repeat(gray[..., None], 3, axis=2)
    display[~finite] = [255, 0, 255]
    if valid is not None:
        display[~valid] = [40, 40, 40]
    return display


def fixed_ridge(values):
    from tools.leaf_preprocess_lab import ridge
    # The legacy ridge uses OpenCV's implicit kernel. Explicitly fixed kernels here
    # make the support radius auditable across float32 and package versions.
    output = np.zeros(values.shape, np.float32)
    for sigma in RIDGE_WIDTHS:
        radius = math.ceil(4*sigma)
        a = cv2.GaussianBlur(values.astype(np.float32), (2*radius+1,)*2, sigma)
        xx = cv2.Sobel(a, cv2.CV_32F, 2, 0, ksize=3)*sigma**2
        yy = cv2.Sobel(a, cv2.CV_32F, 0, 2, ksize=3)*sigma**2
        xy = cv2.Sobel(a, cv2.CV_32F, 1, 1, ksize=3)*sigma**2
        delta = np.sqrt((xx-yy)**2+4*xy**2)
        p, q = (xx+yy+delta)/2, (xx+yy-delta)/2
        small, large = np.minimum(abs(p),abs(q)), np.maximum(abs(p),abs(q))
        output = np.maximum(output, large*np.exp(-.5*(small/(large+1e-8)/.5)**2))
    return output


def cache_key(rgb, condition, provenance):
    """Recipe is part of identity even when two recipes happen to yield equal pixels."""
    pixel_hash = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    payload = dict(shape=list(rgb.shape), dtype=str(rgb.dtype), processed_pixel_sha256=pixel_hash,
                   recipe=condition, provenance=provenance)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def study_analysis(rgb, mask, smoothing, tone, processed=None, channel='green'):
    from tools.leaf_preprocess_lab import analyze, DEFAULTS, normalize_evidence
    processed = process_rgb(rgb, smoothing, tone) if processed is None else processed
    result = analyze(rgb, mask, DEFAULTS)
    maps = scalar_maps(rgb)
    if channel not in maps:
        raise ValueError('Unknown map')
    raw = maps[channel]
    array, valid, common, stats = scalar_result(raw, mask, smoothing)
    # Ridge sees original context, not the NaNs written to the exported analysis array.
    finite = np.isfinite(raw)
    values = smooth(np.where(finite, raw, 0), smoothing)
    ridge_valid = interior(mask & finite, 4+RIDGE_RADIUS)
    evidence = normalize_evidence(fixed_ridge(values), ridge_valid)
    evidence = np.where(ridge_valid, evidence, 0).astype(np.float32)
    alpha = evidence[..., None]*.8
    overlay = np.rint(processed*(1-alpha)+np.array([255,72,24])*alpha).astype(np.uint8)
    result.update(processed=processed, overlay=overlay, evidence=evidence, valid=ridge_valid,
                  ratio=display_scalar(array, common), scalar_before=display_scalar(raw, common),
                  difference=np.clip(abs(processed.astype(np.int16)-rgb.astype(np.int16))*4, 0, 255).astype(np.uint8),
                  scalar_array=array, scalar_common=common, scalar_stats=stats,
                  recipe=recipe(smoothing,tone))
    result['stats']['usable_line_region_percent'] = float(100*ridge_valid.sum()/mask.sum()) if mask.any() else 0.
    result['stats']['scalar_common_percent'] = stats['common_percent']
    result['stats']['scalar_overshoot_percent'] = stats['out_of_range_percent']
    from tools.leaf_preprocess_lab import linear_rgb
    y = linear_rgb(processed) @ np.array([.2126,.7152,.0722], np.float32)
    result['stats']['processed_clipped_percent'] = float(100*np.mean(processed[mask].max(axis=1)>=250))
    result['stats']['processed_dark_percent'] = float(100*np.mean(y[mask]<=.008))
    result['stats']['local_contrast_std'] = float(np.std((y-cv2.GaussianBlur(y,(0,0),5))[mask]))
    return result
