"""E2 - attach compounds to the stem from the side views; axis and rank come from the human registry.

Ground truth of topology is the human registry (side-shoot / truss registries, A6): ``axis``, ``parent_rank`` and the
compound's rank are taken from it unchanged. The side view adds only a ``reconstructed`` auxiliary value: the height
where the compound meets the stem. Method: the stem's column is fitted (``u = a + b * height``) to the visible stem
pixels of a side view; the compound's pixels within ``STEM_NEAR_MM`` of that line are its attachment candidates and
their lowest-median height is the node height. A compound whose junction with the stem is hidden in every side view
gets ``attachment = unavailable`` and is LISTED (never left floating in the structure).

Azimuth-order check: the top-view azimuth of each compound's base is compared between consecutive registry ranks
with a phyllotaxis prior (file ``priors``; generic golden angle, NOT a Micro-Tom measurement) - only flagged, never
used to move a leaf.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.recon.stage1.priors import Priors
from tomato3d.recon.stage1.views import SideObs, TopObs

STEM_NEAR_MM = 1.2
MIN_PIXELS = 3


def _stem_line(sv: SideObs) -> tuple[float, float] | None:
    inv = {v: k for k, v in sv.labels.items()}
    if "stem" not in inv:
        return None
    rows, cols = np.nonzero(sv.id_map == inv["stem"])
    if len(rows) < 10:
        return None
    up = np.array([sv.to_mm(0, r)[1] for r in rows])
    u = np.array([sv.to_mm(c, 0)[0] for c in cols])
    b, a = np.polyfit(up, u, 1)
    return float(a), float(b)


def attachment_heights(sides: Sequence[SideObs], compounds: Sequence[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cid in compounds:
        best = None
        for sv in sides:
            line = _stem_line(sv)
            inv = {v: k for k, v in sv.labels.items()}
            if line is None or cid not in inv:
                continue
            rows, cols = np.nonzero(sv.id_map == inv[cid])
            if not len(rows):
                continue
            cc, rr = sv.camera.center_px, sv.camera.mm_per_px
            u = (cols - cc) * rr
            up = (cc - rows) * rr
            near = np.abs(u - (line[0] + line[1] * up)) < STEM_NEAR_MM
            if near.sum() < MIN_PIXELS:
                continue
            z = np.sort(up[near])
            low_part = z[: max(MIN_PIXELS, len(z) // 3)]
            low = float(np.median(low_part))
            # spread of the candidate heights (robust sd): the measurement's own uncertainty, used by the E2 samples
            sd = float((np.percentile(z, 75) - np.percentile(z, 25)) / 1.349) if len(z) >= 4 else float(np.ptp(z))
            cand = dict(z_mm=low, z_sd_mm=max(sd, sv.camera.mm_per_px), n_px=int(near.sum()),
                        azimuth_view_deg=sv.azimuth_deg)
            if best is None or cand["n_px"] > best["n_px"]:
                best = cand
        out[cid] = best or dict(z_mm=None, n_px=0, azimuth_view_deg=None,
                                reason="compound-stem junction hidden in every side view")
    return out


def base_azimuths(top: TopObs) -> dict[str, dict]:
    """Top-view base of each compound: its rachis pixel nearest the stem (else its nearest pixel)."""
    out = {}
    stem_px = top.camera.project(np.array([*top.stem_xy_mm, 0.0]))
    comps = sorted({v["compound"] for v in top.labels.values() if v["compound"]})
    for cid in comps:
        labs_r = [k for k, v in top.labels.items() if v["compound"] == cid and v["kind"] == "rachis"]
        labs_a = [k for k, v in top.labels.items() if v["compound"] == cid]
        for labs, src in ((labs_r, "rachis"), (labs_a, "any_part")):
            ys, xs = np.nonzero(np.isin(top.id_map, labs)) if labs else (np.array([]), np.array([]))
            if len(xs):
                d = np.hypot(xs - stem_px[0], ys - stem_px[1])
                i = int(np.argmin(d))
                xy = top.camera.unproject_xy(np.array([xs[i], ys[i]], float))
                v = xy - top.stem_xy_mm
                out[cid] = dict(base_xy_mm=xy, azimuth_deg=float(np.degrees(np.arctan2(v[1], v[0]))), base_source=src)
                break
    return out


def run(top: TopObs, sides: Sequence[SideObs], registry: Mapping[str, Mapping[str, object]], priors: Priors) -> dict:
    """``registry``: compound id -> {rank, axis, parent_rank[, node_index]} (the human record; synthetic test: the
    generator's). ``rank`` is the U5 rank (A7: the first two main-stem leaves share rank 1), so ORDER along the axis
    comes from ``node_index`` when given and falls back to (rank, id)."""
    compounds = sorted(registry)

    def order_key(r):
        return (registry[r["compound"]].get("node_index", r["rank"]), r["compound"])
    heights = attachment_heights(sides, compounds)
    az = base_azimuths(top)
    rows = []
    for cid in compounds:
        h, reg = heights[cid], registry[cid]
        rows.append(dict(
            compound=cid, rank=reg["rank"], rank_source="registry", axis=reg["axis"], axis_source="registry",
            parent_rank=reg.get("parent_rank", UNAVAILABLE), attachment_z_mm=h["z_mm"],
            attachment_z_sd_mm=h.get("z_sd_mm"),
            attachment="reconstructed" if h["z_mm"] is not None else UNAVAILABLE,
            attachment_source_kind="reconstructed" if h["z_mm"] is not None else UNAVAILABLE,
            reason=h.get("reason", ""), base_azimuth_deg=az.get(cid, {}).get("azimuth_deg"),
            base_xy_mm=az.get(cid, {}).get("base_xy_mm")))
    # registry-rank order vs detected heights (inversions are reported, the registry wins)
    det = sorted((r for r in rows if r["attachment_z_mm"] is not None), key=order_key)
    inversions = [(a["compound"], b["compound"]) for a, b in zip(det, det[1:]) if b["attachment_z_mm"] < a["attachment_z_mm"]]
    # phyllotaxis flag (prior only)
    div, tol = float(priors.value("phyllotaxis", "divergence_deg")), float(priors.value("phyllotaxis", "tolerance_deg"))
    by_rank = sorted((r for r in rows if r["base_azimuth_deg"] is not None and r["axis"] == "M"), key=order_key)
    flags = []
    for a, b in zip(by_rank, by_rank[1:]):
        d = (b["base_azimuth_deg"] - a["base_azimuth_deg"]) % 360.0
        dev = min(abs(d - div), abs((360 - d) - div))
        if dev > tol:
            flags.append(dict(lower=a["compound"], upper=b["compound"], divergence_deg=d, deviation_deg=dev))
    return dict(rows=rows, rank_height_inversions=inversions, phyllotaxis_flags=flags,
                phyllotaxis_prior=priors.source("phyllotaxis", "divergence_deg"))
