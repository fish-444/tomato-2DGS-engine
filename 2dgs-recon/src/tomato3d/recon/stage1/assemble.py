"""E5 - assemble the initial structure (MTG ids + 3-D pose) and turn it into a point cloud, per completion sample.

Structure per compound: base = top-view base (xy) + E2 attachment height (or E3 base height); rachis = top-view
direction with the E3 elevation (one angle, or two halves for a bent compound). Elements:

* ``observed`` leaflets: their visible top-view pixels lifted onto the rachis-pitched plane at their station (the
  observed outline, not a template) - kind (1) visible geometry;
* ``reconstructed_prior`` leaflets (E4 slots and cut completions): a leaflet from the user's 28-leaflet shape pool,
  meshed with the GENERATOR's surface mapper (:func:`tomato3d.synth.leaflet_surface.surface_points`, flat: bend,
  roll and twist 0 - blade curvature deferred, O7), placed at the slot, heading away from the rachis.

Point clouds are area-weighted samples (``POINT_DENSITY_PER_MM2``); every point carries its element index, so source
kinds stay separable down to the point. Ids are the MTG / track ids (``<plant>/C0001``, ``<compound>/l<k>`` for an
observed leaflet mask); no new id system.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tomato3d.recon.stage1.priors import Priors, mesh_for
from tomato3d.recon.stage1.views import TopObs
from tomato3d.state import geometry as g
from tomato3d.synth.leaflet_surface import surface_points

POINT_DENSITY_PER_MM2 = 1.0


def rachis_z(comp: Mapping, z_base: float, station: float) -> float:
    e1, e2 = comp["elevation_base_half_deg"], comp["elevation_tip_half_deg"]
    half = comp["L_h_mm"] / 2
    if station <= half:
        return z_base + station * np.tan(np.radians(e1))
    return z_base + half * np.tan(np.radians(e1)) + (station - half) * np.tan(np.radians(e2))


def _sample_mesh(verts: np.ndarray, faces: np.ndarray, rng: np.random.Generator, density: float) -> np.ndarray:
    areas = g.triangle_areas(verts, faces)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        return np.zeros((0, 3))
    n = max(1, int(round(areas.sum() * density)))
    idx = rng.choice(len(faces), n, p=areas / areas.sum())
    a, b = rng.random((2, n))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    t = verts[faces[idx]]
    return t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0])


def prior_leaflet_mesh(priors: Priors, shape_id: str, length_mm: float, lw: float, base_xy: np.ndarray,
                       heading_xy: np.ndarray, z: float, pitch_deg: float,
                       along: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Flat pool leaflet centred at ``base_xy``; pitched by ``pitch_deg`` in proportion to its heading's component
    ``along`` the rachis (the compound's elevation; no own tilt of the leaflet - O7)."""
    shape = next(s for s in priors.shapes if s.shape_id == shape_id)
    uv, faces = mesh_for(shape)
    aspect = float(np.ptp(shape.contour[:, 1]))
    h = np.array([heading_xy[0], heading_xy[1], np.tan(np.radians(pitch_deg)) * along])
    h = h / np.linalg.norm(h)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, h)
    leaf = dict(length_mm=length_mm, width_scale=(1.0 / lw) / max(aspect, 1e-6), roll_deg=0.0, twist_deg=0.0,
                bend_deg=0.0, heading=h, left=left / np.linalg.norm(left), up=up,
                base_mm=np.array([base_xy[0], base_xy[1], z]) - h * length_mm / 2)
    return surface_points(uv, leaf), faces


def build(top: TopObs, attach_rows: Sequence[Mapping], tilt_out: Mapping, completion: Mapping, priors: Priors,
          sample: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 7919 * sample)
    comp_geo = {c["compound"]: c for c in tilt_out["compounds"]}
    att = {r["compound"]: r for r in attach_rows}
    leaf_rows = {r["obs_id"]: r for r in tilt_out["leaflets"]}
    elements, clouds, meshes = [], [], {}
    for cid, comp in comp_geo.items():
        if comp.get("elevation_base_half_deg") is None:
            continue
        z0 = att[cid]["attachment_z_mm"]
        if z0 is None:
            continue
        base, d = np.asarray(comp["base_xy"]), np.asarray(comp["dir"])
        for el in completion["elements"]:
            if el["compound"] != cid or (el["sample"] not in (None, sample)):
                continue
            if el["source_kind"] == "observed":
                r = leaf_rows[el["element"]]
                ys, xs = np.nonzero(top.id_map == r["label"])
                xy = top.camera.unproject_xy(np.stack([xs, ys], 1).astype(float))
                s = (xy - base) @ d
                z = np.array([rachis_z(comp, z0, float(v)) for v in np.clip(s, 0, None)])
                pts = np.c_[xy, z]
                keep = rng.random(len(pts)) < min(1.0, POINT_DENSITY_PER_MM2 * top.mm2_per_px())
                pts = pts[keep]
                elements.append(dict(element=el["element"], compound=cid, source_kind="observed", n_points=len(pts)))
            else:
                if el["kind"] == "cut_completion" or el.get("shape_id") is None:
                    # completion of a cut leaflet carries area only; its hidden part has no located outline
                    elements.append(dict(element=el["element"], compound=cid, source_kind="reconstructed_prior",
                                         n_points=0, note="area only (hidden part of a cut leaflet not located)"))
                    continue
                xy = np.asarray(el["xy"], float)
                if not (np.isfinite(xy).all() and np.isfinite(el["length_mm"]) and el["length_mm"] > 0):
                    elements.append(dict(element=el["element"], compound=cid, source_kind="reconstructed_prior",
                                         n_points=0, note="degenerate (non-finite or zero length) - skipped, not meshed"))
                    continue
                s = float((xy - base) @ d)
                away = xy - (base + max(s, 0.0) * d)
                heading = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else d
                e_local = comp["elevation_base_half_deg"] if s <= comp["L_h_mm"] / 2 else comp["elevation_tip_half_deg"]
                v, f = prior_leaflet_mesh(priors, el["shape_id"], el["length_mm"], el["length_over_width"],
                                          xy, heading, rachis_z(comp, z0, max(s, 0.0)), e_local, float(heading @ d))
                meshes[el["element"]] = (v, f)
                pts = _sample_mesh(v, f, rng, POINT_DENSITY_PER_MM2)
                elements.append(dict(element=el["element"], compound=cid, source_kind="reconstructed_prior",
                                     n_points=len(pts)))
            clouds.append(np.c_[pts, np.full(len(pts), len(elements) - 1)])
    cloud = np.vstack(clouds) if clouds else np.zeros((0, 4))
    return dict(sample=sample, elements=elements, points=cloud, meshes=meshes)
