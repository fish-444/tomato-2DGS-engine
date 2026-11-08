"""Relative photometric features, conservative fold gates and boundary votes.

All geometry is on the documented processing grid. Texture/continuity detectors
are engineering proxies; semantic face names are provisional hypotheses.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import KMeans


def zscore(a: np.ndarray, mask: np.ndarray, eps: float) -> np.ndarray:
    values = a[mask]
    if not values.size:
        return np.zeros_like(a, dtype=np.float32)
    return (a - float(values.mean())) / max(float(values.std()), eps)


def maps(rgb: np.ndarray, plant: np.ndarray, cfg: dict, shadow_channel: bool = False) -> dict:
    eps = cfg["epsilon"]
    f = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2Lab)
    luminance = lab[:, :, 0] / 100
    sat = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[:, :, 1]
    if shadow_channel:
        n = cfg["alternative"]["shadow_window_px"]
        luminance = luminance - cv2.blur(luminance, (n, n))
    smoothed = cv2.GaussianBlur(luminance, (0, 0), 1)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3) / 8
    gradient = np.hypot(gx, gy)
    xx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3) / 4
    yy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3) / 4
    xy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3) / 4
    disc = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
    ridge = np.maximum(np.abs(xx + yy + disc), np.abs(xx + yy - disc)) / 2
    n = cfg["window_px"]
    window = (n, n)
    energy = np.sqrt(np.maximum(0, cv2.blur(gradient ** 2, window)))
    ridge_ratio = cv2.blur(ridge, window) / (energy + eps)
    fine = np.abs(luminance - cv2.GaussianBlur(luminance, (0, 0), 1))
    fine = cv2.blur(fine, window)
    hair = fine / (float(np.median(fine[plant])) + eps) if plant.any() else fine * 0
    lap = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)
    sharp = np.maximum(0, cv2.blur(lap ** 2, window) - cv2.blur(lap, window) ** 2)
    local_mean = cv2.blur(luminance, window)
    local_sd = np.sqrt(np.maximum(0, cv2.blur(luminance ** 2, window) - local_mean ** 2))
    # Structure-tensor orientation: axial direction; modulo pi is deliberate.
    tx, ty, txy = cv2.blur(gx * gx, window), cv2.blur(gy * gy, window), cv2.blur(gx * gy, window)
    angle = 0.5 * np.arctan2(2 * txy, tx - ty) + np.pi / 2
    coherence = np.sqrt((tx - ty) ** 2 + 4 * txy ** 2) / (tx + ty + eps)
    return dict(luminance=luminance, gradient=gradient, ridge=ridge,
                crease_z=zscore(ridge, plant, eps), angle=angle, coherence=coherence,
                sharp=sharp, local_sd=local_sd, gx=gx, gy=gy,
                texture=np.stack([ridge_ratio, hair, zscore(sat, plant, eps),
                                  zscore(luminance, plant, eps)], axis=-1))


def face_clusters(feature: dict, plant: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    texture = feature["texture"]
    stride = cfg["sample_stride_px"]
    grid = np.zeros(plant.shape, bool)
    grid[::stride, ::stride] = True
    samples = texture[plant & grid]
    if len(samples) < 2 or float(np.std(samples, axis=0).max()) <= cfg["epsilon"]:
        return np.full(plant.shape, -1, np.int8), dict(status="unavailable", reason="degenerate_texture")
    mean, sd = samples.mean(0), np.maximum(samples.std(0), cfg["epsilon"])
    km = KMeans(n_clusters=2, random_state=cfg["seed"], n_init=5).fit((samples - mean) / sd)
    scores = km.cluster_centers_ @ np.array(cfg["face_semantic_weights"])
    if np.isclose(scores[0], scores[1], atol=cfg["epsilon"]):
        return np.full(plant.shape, -1, np.int8), dict(status="unavailable", reason="cluster_semantics_tied")
    abaxial = int(np.argmax(scores))
    labels = np.full(plant.shape, -1, np.int8)
    labels[plant] = (km.predict((texture[plant] - mean) / sd) == abaxial).astype(np.int8)
    return labels, dict(status="provisional", semantic_basis="ridge+hair-saturation+L",
                        semantic_scores=scores.tolist(), centers=km.cluster_centers_.tolist(),
                        n_samples=len(samples))


def edge(mask: np.ndarray) -> np.ndarray:
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, independent of optional OpenCV ximgproc wheels."""
    out = np.zeros(mask.shape, bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = np.pad(mask[y0:y1, x0:x1], 1).copy()
    while True:
        changed = False
        for phase in (0, 1):
            p = [m[:-2, 1:-1], m[:-2, 2:], m[1:-1, 2:], m[2:, 2:],
                 m[2:, 1:-1], m[2:, :-2], m[1:-1, :-2], m[:-2, :-2]]
            neighbours = sum(v.astype(np.uint8) for v in p)
            transitions = sum((~p[i] & p[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            remove = m[1:-1, 1:-1] & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            if phase == 0:
                remove &= ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                remove &= ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            if remove.any():
                m[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            break
    out[y0:y1, x0:x1] = m[1:-1, 1:-1]
    return out


def contact(a: np.ndarray, b: np.ndarray, radius_px: int) -> np.ndarray:
    roi = pair_roi(a, b, radius_px)
    if roi is None:
        return np.zeros(a.shape, bool)
    ys, xs = roi
    aa, bb = a[ys, xs], b[ys, xs]
    k = np.ones((radius_px * 2 + 1,) * 2, np.uint8)
    da = cv2.dilate(aa.astype(np.uint8), k).astype(bool)
    db = cv2.dilate(bb.astype(np.uint8), k).astype(bool)
    out = np.zeros(a.shape, bool)
    out[ys, xs] = skeleton(da & db & (edge(aa) | edge(bb)))
    return out


def pair_roi(a: np.ndarray, b: np.ndarray, radius_px: int) -> tuple | None:
    """Crop to the union bounding box plus padding; never truncate an outline."""
    ay, ax = np.flatnonzero(a.any(1)), np.flatnonzero(a.any(0))
    by, bx = np.flatnonzero(b.any(1)), np.flatnonzero(b.any(0))
    if not len(ax) or not len(bx):
        return None
    if ax[-1] + radius_px < bx[0] or bx[-1] + radius_px < ax[0] or ay[-1] + radius_px < by[0] or by[-1] + radius_px < ay[0]:
        return None
    pad = radius_px + 2
    return (slice(max(0, min(ay[0], by[0]) - pad), min(a.shape[0], max(ay[-1], by[-1]) + pad + 1)),
            slice(max(0, min(ax[0], bx[0]) - pad), min(a.shape[1], max(ax[-1], bx[-1]) + pad + 1)))


def _components(mask: np.ndarray, minimum: int = 1) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= minimum]


def _junction_map(a: np.ndarray, b: np.ndarray, cfg: dict) -> np.ndarray:
    lines = skeleton(edge(a) | edge(b))
    padded = np.pad(lines, 1)
    h, w = lines.shape
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    ring = [padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in offsets]
    branches = sum((~ring[i] & ring[(i + 1) % 8]).astype(np.uint8) for i in range(8))
    junction = lines & (branches >= 3)
    r = cfg["junction_probe_px"]
    return cv2.dilate(junction.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)).astype(bool)


def _junctions(a: np.ndarray, b: np.ndarray, boundary: np.ndarray, cfg: dict) -> float:
    return float(_junction_map(a, b, cfg)[boundary].mean()) if boundary.any() else 1.0


def _axial_average(angle: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    w = weights[mask]
    if not w.size or float(w.sum()) <= 1e-8:
        return 0.0, 0.0
    c = float(np.sum(np.cos(2 * angle[mask]) * w) / w.sum())
    s = float(np.sum(np.sin(2 * angle[mask]) * w) / w.sum())
    return math.atan2(s, c) / 2, math.hypot(c, s)


def fold_gate(evidence: dict, cfg: dict) -> tuple[bool, list[str]]:
    failures = []
    if not evidence["opposite_faces"]:
        failures.append("opposite_faces_missing")
    if evidence["boundary_px"] < cfg["min_boundary_px"] or evidence["crease_fraction"] < cfg["crease_fraction_min"]:
        failures.append("shared_crease_missing")
    if evidence["continuity"] < cfg["continuity_min"] or evidence["t_junction_fraction"] > cfg["t_junction_fraction_max"]:
        failures.append("outline_interrupted_or_T_junction")
    if evidence["midvein_agreement"] < cfg["midvein_agreement_min"]:
        failures.append("midvein_direction_inconsistent")
    if not evidence.get("membership_compatible", True):
        failures.append("membership_not_confirmed_compatible")
    return not failures, failures


def fold_evidence(a: np.ndarray, b: np.ndarray, feature: dict, labels: np.ndarray, cfg: dict,
                  membership_compatible: bool = True) -> dict:
    roi = pair_roi(a, b, cfg["contact_radius_px"])
    if roi is not None:
        a, b, labels = a[roi], b[roi], labels[roi]
        feature = {k: v[roi] for k, v in feature.items()}
    boundary = contact(a, b, cfg["contact_radius_px"])
    boundary_px = int(boundary.sum())
    fa = float((labels[a] == 1).mean()) if a.any() else 0.5
    fb = float((labels[b] == 1).mean()) if b.any() else 0.5
    opposite = ((fa >= cfg["opposite_face_fraction_min"] and fb <= 1 - cfg["opposite_face_fraction_min"]) or
                (fb >= cfg["opposite_face_fraction_min"] and fa <= 1 - cfg["opposite_face_fraction_min"]))
    aa, ca = _axial_average(feature["angle"], feature["coherence"], a)
    ab, cb = _axial_average(feature["angle"], feature["coherence"], b)
    agreement = abs(math.cos(aa - ab)) if ca > cfg["epsilon"] and cb > cfg["epsilon"] else 0.0
    tj = _junctions(a, b, boundary, cfg) if boundary_px else 1.0
    union = a | b
    pieces = _components(union)
    # Multiple disconnected silhouettes cannot establish continuous outline.
    continuity = (1.0 - tj) if len(pieces) == 1 else 0.0
    near_ridge = cv2.dilate((feature["crease_z"] >= cfg["crease_z_min"]).astype(np.uint8),
                            np.ones((3, 3), np.uint8)).astype(bool)
    return dict(boundary_px=boundary_px, crease_fraction=float(near_ridge[boundary].mean()) if boundary_px else 0,
                continuity=continuity, t_junction_fraction=tj, midvein_agreement=agreement,
                opposite_faces=bool(opposite), membership_compatible=membership_compatible,
                face_fraction_a=fa, face_fraction_b=fb)


def unify(masks: dict[str, np.ndarray], members: dict[str, str | None], feature: dict,
          labels: np.ndarray, cfg: dict) -> tuple[dict, list[dict], list[dict]]:
    """Merge compatible instances only; all three gates apply to the accumulated union."""
    current = {k: dict(mask=v.copy(), source_ids=[k], compound_id=members.get(k), folded=False,
                       crease_length_px=0) for k, v in masks.items()}
    evidence_rows = []
    for a, b in combinations(sorted(masks), 2):
        if a not in current or b not in current:
            continue
        ma, mb = current[a]["mask"], current[b]["mask"]
        if not contact(ma, mb, cfg["contact_radius_px"]).any():
            continue
        compatible = members.get(a) is not None and members.get(a) == members.get(b)
        ev = fold_evidence(ma, mb, feature, labels, cfg, compatible)
        yes, failures = fold_gate(ev, cfg)
        evidence_rows.append(dict(a=a, b=b, kind="cross_instance", merged=yes, failures=failures, **ev))
        if yes:
            current[a]["mask"] |= mb
            current[a]["source_ids"] += current[b]["source_ids"]
            current[a]["folded"] = True
            current[a]["crease_length_px"] += ev["boundary_px"]
            del current[b]
    for iid, item in current.items():
        m = item["mask"]
        valid = labels[m] >= 0
        fraction = float((labels[m] == 1).mean()) if valid.any() else None
        if fraction is not None and cfg["fold_min_face_fraction"] <= fraction <= 1 - cfg["fold_min_face_fraction"]:
            # Remove small texture islands; largest face regions are fold candidates.
            pa = _components(m & (labels == 0), cfg["min_boundary_px"])
            pb = _components(m & (labels == 1), cfg["min_boundary_px"])
            if pa and pb:
                a, b = max(pa, key=np.sum), max(pb, key=np.sum)
                ev = fold_evidence(a, b, feature, labels, cfg)
                yes, failures = fold_gate(ev, cfg)
                evidence_rows.append(dict(a=iid, b=iid, kind="within_instance", merged=yes, failures=failures, **ev))
                if yes:
                    item["folded"] = True
                    item["crease_length_px"] += ev["boundary_px"]
        item["face"] = ("both_folded" if item["folded"] else "unavailable" if fraction is None
                        else "abaxial" if fraction >= 0.5 else "adaxial")
        item["abaxial_fraction"] = fraction
    rows = [dict(leaflet_id=k, compound_id=v["compound_id"] or "unavailable", face=v["face"],
                 face_status="provisional" if v["face"] != "unavailable" else "unavailable",
                 abaxial_fraction=v["abaxial_fraction"], merged_from=v["source_ids"],
                 crease_length_px=v["crease_length_px"], visible_area_px=int(v["mask"].sum()),
                 area_underestimate_flag=v["folded"], review_status="proposed") for k, v in current.items()]
    return current, rows, evidence_rows


def _segments(boundary: np.ndarray, length: int) -> list[np.ndarray]:
    """Walk the skeleton graph, never bridge spatially disconnected boundaries."""
    points = set(map(tuple, np.argwhere(boundary).tolist()))
    result = []
    while points:
        start = min(points)
        stack, ordered = [start], []
        while stack:
            pt = stack.pop()
            if pt not in points:
                continue
            points.remove(pt)
            ordered.append(pt)
            y, x = pt
            neighbours = sorted((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                                if (dy or dx) and (y + dy, x + dx) in points)
            stack.extend(reversed(neighbours))
        result.extend(np.array(ordered[i:i + length]) for i in range(0, len(ordered), length))
    return result


def _sample(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(arr, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def boundary_features(a: np.ndarray, b: np.ndarray, feature: dict, cfg: dict) -> list[dict]:
    roi = pair_roi(a, b, max(cfg["contact_radius_px"], cfg["profile_half_px"]))
    if roi is None:
        return []
    rows = _boundary_features(a[roi], b[roi], {k: v[roi] for k, v in feature.items()}, cfg)
    for row in rows:
        row["x_px"] += roi[1].start
        row["y_px"] += roi[0].start
    return rows


def _boundary_features(a: np.ndarray, b: np.ndarray, feature: dict, cfg: dict) -> list[dict]:
    boundary = contact(a, b, cfg["contact_radius_px"])
    if not boundary.any():
        return []
    eps = cfg["epsilon"]
    # Direction toward A based on signed distance difference, independent of IDs.
    field = (cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5) -
             cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5) -
             cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 5) +
             cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5))
    nx = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3)
    ny = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.hypot(nx, ny)
    nx, ny = nx / (norm + eps), ny / (norm + eps)
    half = cfg["profile_half_px"]
    offsets = np.arange(-half, half + 1, dtype=np.float32)
    rows = []
    ea, eb = edge(a).astype(np.float32), edge(b).astype(np.float32)
    junction_map = _junction_map(a, b, cfg)
    for number, pts in enumerate(_segments(boundary, cfg["segment_len_px"])):
        yy, xx = pts[:, 0], pts[:, 1]
        px, py = nx[yy, xx], ny[yy, xx]
        valid_normal = np.hypot(px, py) > eps
        if not valid_normal.any():
            continue
        yy, xx, px, py = yy[valid_normal], xx[valid_normal], px[valid_normal], py[valid_normal]
        sx, sy = xx[:, None] + px[:, None] * offsets, yy[:, None] + py[:, None] * offsets
        lum = _sample(feature["luminance"], sx, sy)
        grad = _sample(feature["gradient"], sx, sy)
        sharp = _sample(feature["sharp"], sx, sy)
        # Positive offset is A, negative offset B. Exclude centre/anti-alias fringe.
        qa, qb = offsets >= 2, offsets <= -2
        sd = np.maximum(lum.std(1), eps)
        shadow = float(np.mean((lum[:, qa].mean(1) - lum[:, qb].mean(1)) / sd))
        drop_a = np.mean((lum[:, -1] - lum[:, half + 1]) / sd)
        drop_b = np.mean((lum[:, 0] - lum[:, half - 1]) / sd)
        sharp_ratio = float(np.mean(np.log((sharp[:, qa].mean(1) + eps) / (sharp[:, qb].mean(1) + eps))))
        gradient_ratio = float(np.mean(grad[:, half] / (0.5 * (grad[:, qa].mean(1) + grad[:, qb].mean(1)) + eps)))
        # Tangent continuation on both ends of a small normal-offset strip.
        r = cfg["junction_probe_px"]
        tangent_x, tangent_y = -py, px
        def continuity(e: np.ndarray) -> float:
            values = []
            for direction in (-1, 1):
                sx2 = xx[:, None] + direction * r * tangent_x[:, None] + px[:, None] * np.arange(-2, 3)
                sy2 = yy[:, None] + direction * r * tangent_y[:, None] + py[:, None] * np.arange(-2, 3)
                values.append(_sample(e, sx2, sy2).max(1))
            return float(np.minimum(*values).mean())
        cont_a, cont_b = continuity(ea), continuity(eb)
        segment_mask = np.zeros(a.shape, bool)
        segment_mask[yy, xx] = True
        tj = float(junction_map[segment_mask].mean())
        crease = float((feature["crease_z"][yy, xx] >= cfg["crease_z_min"]).mean())
        raw_score = cfg["rule_weights"]["shadow"] * np.tanh(shadow) + cfg["rule_weights"]["continuity"] * (cont_a - cont_b)
        rows.append(dict(segment=number, boundary_px=len(xx), x_px=float(xx.mean()), y_px=float(yy.mean()),
                         shadow=shadow, drop_asymmetry=float(drop_b - drop_a), gradient_ratio=gradient_ratio,
                         sharpness_log_ratio=sharp_ratio, continuity_a=cont_a, continuity_b=cont_b,
                         continuity_difference=cont_a - cont_b, t_junction_fraction=tj, crease_fraction=crease,
                         score=float(raw_score), crease_excluded=False))
    return rows


def pool(rows: list[dict], cfg: dict) -> dict:
    usable = [r for r in rows if not r.get("crease_excluded", False)]
    n = sum(r["boundary_px"] for r in usable)
    if not n:
        return dict(boundary_px=0, score=0.0, confidence=0.0, decision="undecided", conflict_fraction=0.0,
                    reason="no_eligible_boundary")
    score = sum(r["score"] * r["boundary_px"] for r in usable) / n
    positive = sum(r["boundary_px"] for r in usable if r["score"] >= cfg["margin_min"])
    negative = sum(r["boundary_px"] for r in usable if r["score"] <= -cfg["margin_min"])
    conflict = min(positive, negative) / max(positive + negative, 1)
    reason = ("insufficient_boundary" if n < cfg["min_boundary_px"] else
              "conflicting_votes" if conflict > cfg["conflict_fraction_max"] else
              "low_margin" if abs(score) < cfg["margin_min"] else "relative_evidence")
    decision = "undecided" if reason != "relative_evidence" else "A" if score > 0 else "B"
    return dict(boundary_px=n, score=float(score), confidence=float(abs(score) * (1 - conflict)),
                decision=decision, conflict_fraction=float(conflict), reason=reason)


def graph_order(compounds: list[str], pairs: list[dict]) -> dict:
    edges = {}
    for p in pairs:
        if p["decision"] in ("A", "B"):
            a, b = (p["a"], p["b"]) if p["decision"] == "A" else (p["b"], p["a"])
            edges[a, b] = p["confidence"]
    adjacency = {c: sorted(b for a, b in edges if a == c) for c in compounds}
    cycles, seen = [], set()
    def walk(start: str, path: list[str]) -> None:
        for nxt in adjacency[path[-1]]:
            if nxt == start:
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    cycle_edges = [(path[i], path[(i + 1) % len(path)]) for i in range(len(path))]
                    weak = min(cycle_edges, key=lambda e: (edges[e], e))
                    cycles.append(dict(nodes=path.copy(), weakest_edge=list(weak), confidence=edges[weak]))
            elif nxt not in path and nxt >= start:
                walk(start, path + [nxt])
    for node in sorted(compounds):
        walk(node, [node])
    order = []
    indegree = {n: sum(b == n for a, b in edges) for n in compounds}
    queue = sorted(n for n, degree in indegree.items() if degree == 0)
    while queue:
        n = queue.pop(0)
        order.append(n)
        for dest in adjacency[n]:
            indegree[dest] -= 1
            if indegree[dest] == 0:
                queue.append(dest)
                queue.sort()
    return dict(edges=[dict(over=a, under=b, confidence=c) for (a, b), c in edges.items()], cycles=cycles,
                topological_order=order if not cycles else None, status="cyclic" if cycles else "acyclic")


def alternative_rgb(rgb: np.ndarray, cfg: dict) -> np.ndarray:
    a = cfg["alternative"]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(a["clahe_clip"], (a["clahe_tile"],) * 2).apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    image += a["unsharp_amount"] * (image - cv2.GaussianBlur(image, (0, 0), a["unsharp_sigma_px"]))
    return np.clip(image, 0, 255).astype(np.uint8)
