"""Instance masks on a real top photo: persistence, compound -> leaflet membership, own-pot attribution (X3/X4 of
``handoff/CLAUDE_CODE_CLI_PROMPT_TOPVIEW_REAL_LINK_2026-10-12.md``).

Every result here is a PROPOSAL: detector boxes are suggestions (DETECTOR_POLICY), SAM masks are machine output,
and nothing becomes an observation until a person confirms it (:mod:`tomato3d.topview.confirm_tool`).

* **Masks** are stored losslessly, one PNG per instance cropped to the mask's own bounding box (0 / 255, 8 bit),
  plus a manifest with offset, full image size, pixel count and the PNG's sha256 - :func:`load_mask` rebuilds the
  full-size mask and refuses a PNG whose hash changed.
* **Membership** (leaflet -> compound) comes from the detector boxes: a compound is a candidate when its box holds
  the majority of the leaflet's box. One candidate -> member. Several -> the one holding the majority of the
  leaflet's MASK, if exactly one does; otherwise ``ambiguous`` with the candidates listed. None -> ``unassigned``.
* **Attribution** (X3) to the photo's own pot, from that pot's rim polygon (rim record): a compound is ``inside``
  when its mask reaches into the own rim polygon (a compound leaf starts at the stem, inside the pot); a leaflet is
  ``inside`` when its centroid lies in the own rim polygon or it is a member of an ``inside`` compound.
  ``outside`` needs positive evidence - the detection sits in a NEIGHBOUR pot's rim polygon (rim points clicked for
  another plant on the same photo). Everything else stays ``ambiguous`` and is never assigned silently. Known limit:
  a neighbour's compound overhanging the own rim polygon reads ``inside``; only the human check catches it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

UNAVAILABLE = "unavailable"
ATTRIBUTIONS = ("inside", "outside", "ambiguous")
MEMBERSHIP = ("member", "ambiguous", "unassigned")


# --------------------------------------------------------------------------- persistence
def save_masks(masks: dict[str, np.ndarray], folder: Path, rel_to: Path | None = None) -> list[dict[str, Any]]:
    """Write one cropped PNG per non-empty mask; return manifest entries (empty masks get ``png = null``)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = Path(rel_to) if rel_to is not None else folder
    out = []
    for iid, m in masks.items():
        m = np.asarray(m, bool)
        h, w = m.shape
        entry = dict(instance_id=iid, image_size_px=[w, h], area_px=int(m.sum()))
        if not m.any():
            out.append(dict(entry, png=None, sha256=None, offset_px=None, crop_size_px=None))
            continue
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        ok, buf = cv2.imencode(".png", (m[y0:y1, x0:x1] * 255).astype(np.uint8))
        if not ok:
            raise RuntimeError(f"PNG encode failed for {iid}")
        name = iid.replace("/", "_").replace("#", "_") + ".png"
        (folder / name).write_bytes(buf.tobytes())
        out.append(dict(entry, png=Path(os.path.relpath(folder / name, base)).as_posix(),
                        sha256=hashlib.sha256(buf.tobytes()).hexdigest(), offset_px=[x0, y0],
                        crop_size_px=[x1 - x0, y1 - y0]))
    return out


def load_mask(entry: dict[str, Any], base: Path) -> np.ndarray:
    """Full-size boolean mask of one manifest entry; the PNG's sha256 must match."""
    w, h = entry["image_size_px"]
    full = np.zeros((h, w), bool)
    if entry["png"] is None:
        return full
    data = (Path(base) / entry["png"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['instance_id']}: mask PNG changed since the manifest (sha256)")
    crop = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) > 0
    x0, y0 = entry["offset_px"]
    if list(crop.shape[::-1]) != list(entry["crop_size_px"]):
        raise ValueError(f"{entry['instance_id']}: crop size {crop.shape[::-1]} != manifest {entry['crop_size_px']}")
    full[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    if int(full.sum()) != entry["area_px"]:
        raise ValueError(f"{entry['instance_id']}: pixel count {int(full.sum())} != manifest {entry['area_px']}")
    return full


def write_manifest(path: Path, header: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(dict(header, masks=entries), ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- membership
def _box_area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_inside_frac(inner: list[float], outer: list[float]) -> float:
    """Share of ``inner``'s box area that lies inside ``outer`` (xyxy)."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = _box_area(inner)
    return ix * iy / a if a > 0 else 0.0


def membership(leaflets: list[dict[str, Any]], compounds: list[dict[str, Any]], majority: float) -> list[dict[str, Any]]:
    """Per leaflet: status member / ambiguous / unassigned, compound id, candidates and their fractions.

    ``leaflets`` / ``compounds``: dicts with ``id``, ``box`` (xyxy px) and ``mask`` (full-size bool or None)."""
    out = []
    for lf in leaflets:
        fr = {c["id"]: box_inside_frac(lf["box"], c["box"]) for c in compounds}
        cands = [cid for cid, f in fr.items() if f > majority]
        mask_fr: dict[str, float] = {}
        lm = lf.get("mask")
        if lm is not None and lm.any():
            n = float(lm.sum())
            for c in compounds:
                if c["id"] in cands and c.get("mask") is not None:
                    mask_fr[c["id"]] = float((lm & c["mask"]).sum() / n)
        if len(cands) == 1:
            res = dict(status="member", compound_id=cands[0], basis="box_majority")
        elif len(cands) > 1:
            by_mask = [cid for cid in cands if mask_fr.get(cid, 0.0) > majority]
            res = (dict(status="member", compound_id=by_mask[0], basis="box_majority_then_mask_majority")
                   if len(by_mask) == 1 else dict(status="ambiguous", compound_id=UNAVAILABLE,
                                                  basis="several compound boxes hold the leaflet; mask does not decide"))
        else:
            res = dict(status="unassigned", compound_id=UNAVAILABLE, basis="no compound box holds the leaflet's majority")
        res.update(leaflet_id=lf["id"], candidates=cands,
                   box_inside_frac={k: round(v, 4) for k, v in fr.items() if v > 0},
                   mask_inside_frac={k: round(v, 4) for k, v in mask_fr.items()})
        out.append(res)
    return out


def terminal_conflicts(members: list[dict[str, Any]], leaflet_class: dict[str, str]) -> dict[str, list[str]]:
    """Compounds that received more than one ``terminal_leaf`` (a compound has one terminal leaflet)."""
    per: dict[str, list[str]] = {}
    for m in members:
        if m["status"] == "member" and leaflet_class.get(m["leaflet_id"]) == "terminal_leaf":
            per.setdefault(m["compound_id"], []).append(m["leaflet_id"])
    return {k: v for k, v in per.items() if len(v) > 1}


# --------------------------------------------------------------------------- attribution
def rim_polygon_px(h_img_to_mm: np.ndarray, edge_mm: float) -> np.ndarray:
    """The rim square's corners (across the flats) in image px, from the image -> rim-plane homography."""
    half = edge_mm / 2.0
    corners = np.array([[-half, -half, 1.0], [half, -half, 1.0], [half, half, 1.0], [-half, half, 1.0]])
    p = corners @ np.linalg.inv(np.asarray(h_img_to_mm, float)).T
    return p[:, :2] / p[:, 2:3]


def polygon_mask(poly_px: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    m = np.zeros((size_wh[1], size_wh[0]), np.uint8)
    cv2.fillPoly(m, [np.round(poly_px).astype(np.int32)], 1)
    return m.astype(bool)


def _in_poly(xy: tuple[float, float], poly_px: np.ndarray) -> bool:
    return cv2.pointPolygonTest(poly_px.astype(np.float32), (float(xy[0]), float(xy[1])), False) >= 0


def attribute(compounds: list[dict[str, Any]], leaflets: list[dict[str, Any]], members: list[dict[str, Any]],
              own_poly: np.ndarray | None, neighbour_polys: dict[str, np.ndarray]) -> dict[str, dict[str, str]]:
    """``{instance_id: {attribution, reason}}`` for compounds then leaflets. No own rim polygon -> all ambiguous."""
    out: dict[str, dict[str, str]] = {}
    if own_poly is None:
        for x in compounds + leaflets:
            out[x["id"]] = dict(attribution="ambiguous", reason="no solved rim polygon for this photo's own pot")
        return out
    for c in compounds:
        m = c.get("mask")
        size = (m.shape[1], m.shape[0]) if m is not None else None
        reach = bool(m is not None and m.any() and (m & polygon_mask(own_poly, size)).any())
        nb = [p for p, poly in neighbour_polys.items() if c.get("centroid") and _in_poly(c["centroid"], poly)]
        if reach:
            out[c["id"]] = dict(attribution="inside", reason="compound mask reaches into the own rim polygon")
        elif nb:
            out[c["id"]] = dict(attribution="outside", reason=f"centroid in the rim polygon of {nb[0]}")
        else:
            out[c["id"]] = dict(attribution="ambiguous",
                                reason="compound mask does not reach the own rim polygon and no neighbour rim record "
                                       "places it elsewhere")
    by_leaf = {m["leaflet_id"]: m for m in members}
    for lf in leaflets:
        cen = lf.get("centroid")
        mem = by_leaf.get(lf["id"], {})
        parent = out.get(mem.get("compound_id", ""), {}).get("attribution") if mem.get("status") == "member" else None
        nb = [p for p, poly in neighbour_polys.items() if cen and _in_poly(cen, poly)]
        if cen and _in_poly(cen, own_poly):
            out[lf["id"]] = dict(attribution="inside", reason="centroid in the own rim polygon")
        elif parent == "inside":
            out[lf["id"]] = dict(attribution="inside", reason=f"member of compound {mem['compound_id']} (inside)")
        elif nb:
            out[lf["id"]] = dict(attribution="outside", reason=f"centroid in the rim polygon of {nb[0]}")
        elif parent == "outside":
            out[lf["id"]] = dict(attribution="outside", reason=f"member of compound {mem['compound_id']} (outside)")
        else:
            why = {"ambiguous": "its compound is ambiguous", None: "no compound membership"}.get(parent, "")
            if mem.get("status") == "ambiguous":
                why = "compound membership ambiguous"
            out[lf["id"]] = dict(attribution="ambiguous",
                                 reason=f"centroid outside the own rim polygon; {why}; no neighbour rim record")
    return out
