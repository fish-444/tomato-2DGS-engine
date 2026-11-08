"""E1 - height order of leaflets from the top view's occlusion boundaries, supported by side-view heights.

Occlusion cue (visible masks only, no amodal shape) - the HULL-CUT rule: when B lies on A, A's visible region is
cut by B, so A's convex hull spans the cut and contains B's pixels. ``D(A|B)`` = area of B inside A's convex hull;
``D(A|B) > D(B|A) + HULL_MARGIN_MM2`` -> B is above A. (A local-convexity cue was tried first and is INVERTED on these
plants: the occluded leaflet often shows only a thin, convex-looking sliver; 0.33 correct on 10 synthetic scenes vs
0.84 for the hull cut. Leaflet outlines are not perfectly convex, so the rule is a cue, not a proof.)

Side views: each leaflet's top-view centroid gives its column in a side image; the median height of its compound's
side-mask pixels in that column window is the leaflet's ``z_side_mm`` (a compound-level proxy: other parts of the same
compound at that column are mixed in). Side heights decide undecided pairs that differ by more than
``SIDE_MIN_DZ_MM``; a pair where the occlusion cue and the side heights disagree by more than
``SIDE_CONFLICT_DZ_MM`` is a contradiction.

Contradictions and cycles leave the leaflets involved ``height_order = unavailable`` with the reason. Output: one row
per leaflet - ``height_rank`` (1 = top-most among ranked leaflets) and ``height_order_source`` (occlusion | side_view |
both). These thresholds are method constants of this module (not plant priors).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import cv2
import numpy as np

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.recon.stage1.views import SideObs, TopObs, side_transform

HULL_MARGIN_MM2 = 0.4
MIN_BOUNDARY_PX = 6
SIDE_COLUMN_HALF_MM = 1.5
SIDE_MIN_DZ_MM = 2.0
SIDE_CONFLICT_DZ_MM = 6.0


def adjacency(id_map: np.ndarray, keep: set[int]) -> dict[tuple[int, int], np.ndarray]:
    """Boundary pixels of label a that touch label b (4-neighbourhood), for labels in ``keep``."""
    out: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    h, w = id_map.shape
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        y0, y1, x0, x1 = max(0, -dy), h - max(0, dy), max(0, -dx), w - max(0, dx)
        a = id_map[y0:y1, x0:x1]
        b = id_map[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        diff = (a != b) & (a > 0) & (b > 0)
        ys, xs = np.nonzero(diff)
        for la, lb, y, x in zip(a[ys, xs], b[ys, xs], ys + y0, xs + x0):
            if la in keep and lb in keep:
                out[(int(la), int(lb))].append(np.array([y, x]))
    return {k: np.unique(np.array(v), axis=0) for k, v in out.items()}


def side_heights(top: TopObs, sides: Sequence[SideObs]) -> dict[int, float]:
    """Leaflet label -> median side-view height (mm) of its compound's pixels in the leaflet's column window."""
    out: dict[int, float] = {}
    for lab, info in top.labels.items():
        if info["kind"] != "leaflet":
            continue
        ys, xs = np.nonzero(top.id_map == lab)
        if not len(xs):
            continue
        xy = top.camera.unproject_xy(np.stack([xs, ys], 1).astype(float)).mean(axis=0)
        zs = []
        for sv in sides:
            inv = {v: k for k, v in sv.labels.items()}
            if info["compound"] not in inv:
                continue
            u = float(side_transform(np.array([[xy[0], xy[1], 0.0]]), sv.azimuth_deg)[0, 0])
            c0, _ = sv.to_px(u - SIDE_COLUMN_HALF_MM, 0)
            c1, _ = sv.to_px(u + SIDE_COLUMN_HALF_MM, 0)
            cols = slice(max(int(np.floor(c0)), 0), int(np.ceil(c1)) + 1)
            rows, _ = np.nonzero(sv.id_map[:, cols] == inv[info["compound"]])
            if len(rows):
                zs.append(sv.to_mm(0, float(np.median(rows)))[1])
        if zs:
            out[lab] = float(np.median(zs))
    return out


def _sccs(nodes: Sequence[int], edges: dict[int, set[int]]) -> list[list[int]]:
    index, low, on, stack, out, counter = {}, {}, set(), [], [], [0]

    def visit(v: int) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in edges.get(v, ()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    for v in nodes:
        if v not in index:
            visit(v)
    return out


def hull_masks(id_map: np.ndarray, labels: set[int]) -> dict[int, np.ndarray]:
    out = {}
    for lab in labels:
        m = (id_map == lab).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        h = np.zeros_like(m)
        if cs:
            cv2.fillPoly(h, [cv2.convexHull(np.vstack(cs))], 1)
        out[lab] = h.astype(bool)
    return out


def run(top: TopObs, sides: Sequence[SideObs]) -> dict:
    leaf = {k for k, v in top.labels.items() if v["kind"] == "leaflet"}
    adj = adjacency(top.id_map, leaf)
    hulls = hull_masks(top.id_map, leaf)
    margin_px = HULL_MARGIN_MM2 / top.mm2_per_px()
    zside = side_heights(top, sides)
    relations = []
    above: dict[int, set[int]] = defaultdict(set)
    occl_involved, side_involved, bad = set(), set(), {}
    done = set()
    for (a, b), pa in adj.items():
        if (b, a) in done or (a, b) in done:
            continue
        done.add((a, b))
        pb = adj.get((b, a))
        if pb is None or len(pa) + len(pb) < MIN_BOUNDARY_PX:
            continue
        fa = float(((hulls[a]) & (top.id_map == b)).sum())   # B inside A's hull: A cut by B
        fb = float(((hulls[b]) & (top.id_map == a)).sum())
        za, zb = zside.get(a), zside.get(b)
        rel = None
        if fa > fb + margin_px:
            rel, src = (b, a), "occlusion"
        elif fb > fa + margin_px:
            rel, src = (a, b), "occlusion"
        elif za is not None and zb is not None and abs(za - zb) > SIDE_MIN_DZ_MM:
            rel, src = ((a, b) if za > zb else (b, a)), "side_view"
        if rel is None:
            relations.append(dict(a=a, b=b, source="undecided", cut_a_px=fa, cut_b_px=fb, boundary_px=len(pa) + len(pb)))
            continue
        hi, lo = rel
        if src == "occlusion" and zside.get(hi) is not None and zside.get(lo) is not None and \
                zside[lo] - zside[hi] > SIDE_CONFLICT_DZ_MM:
            reason = (f"occlusion says {top.labels[hi]['obs_id']} above {top.labels[lo]['obs_id']}, side view says "
                      f"{zside[lo] - zside[hi]:.1f} mm below")
            bad[hi] = bad[lo] = reason
            relations.append(dict(a=hi, b=lo, source="contradiction", cut_a_px=fa, cut_b_px=fb, boundary_px=len(pa) + len(pb)))
            continue
        above[hi].add(lo)
        (occl_involved if src == "occlusion" else side_involved).update((hi, lo))
        relations.append(dict(a=hi, b=lo, source=src, cut_a_px=fa, cut_b_px=fb, boundary_px=len(pa) + len(pb)))
    for comp in _sccs(sorted(leaf), above):
        if len(comp) > 1:
            names = ", ".join(sorted(top.labels[c]["obs_id"] for c in comp))
            for c in comp:
                bad.setdefault(c, f"cycle in the occlusion graph among {names}")
    # topological order (top-most first) among valid leaflets; ties -> higher side z first, else unranked
    valid = [l for l in leaf if l not in bad and (l in occl_involved or l in side_involved or l in zside)]
    indeg = {l: 0 for l in valid}
    for hi, los in above.items():
        if hi in indeg:
            for lo in los:
                if lo in indeg:
                    indeg[lo] += 1
    order, ready = [], [l for l in valid if indeg[l] == 0]
    while ready:
        ready.sort(key=lambda l: -zside.get(l, -1e9))
        cur = ready.pop(0)
        order.append(cur)
        for lo in above.get(cur, ()):
            if lo in indeg:
                indeg[lo] -= 1
                if indeg[lo] == 0:
                    ready.append(lo)
    rank = {l: i + 1 for i, l in enumerate(order)}
    rows = []
    for lab in sorted(leaf):
        info = top.labels[lab]
        o, sv = lab in occl_involved, lab in side_involved or lab in zside
        if lab in bad:
            hr, src, reason = UNAVAILABLE, UNAVAILABLE, bad[lab]
        elif lab in rank:
            hr, src, reason = str(rank[lab]), ("both" if o and sv else "occlusion" if o else "side_view"), ""
        else:
            hr, src, reason = UNAVAILABLE, UNAVAILABLE, "no occlusion relation and no side height"
        rows.append(dict(obs_id=info["obs_id"], compound=info["compound"], label=lab, height_rank=hr,
                         height_order_source=src, z_side_mm=zside.get(lab), reason=reason))
    return dict(rows=rows, relations=relations, above={k: sorted(v) for k, v in above.items()}, z_side=zside)
