"""Direct PPFD on receivers, with shading by plant surfaces.

For every receiver point and every emitter point::

    E = I0 * cos(theta)**m * cos(beta) / d**2        umol m-2 s-1

``theta`` is measured from the fixture's aim axis, ``beta`` between the ray and the
receiver normal, ``d`` the distance. The segment from the receiver to the emitter is
tested against every surface triangle:

* crossing an opaque surface (anything that is not a leaf) stops the ray;
* crossing a leaf multiplies it by the leaf transmittance, once per leaf crossed;
* with the transmittance unavailable, a ray that crosses a leaf is unavailable (NaN),
  while a ray with a clear path stays known.

Only direct light is traced: nothing reflected by walls, shelves or leaves is added, and
light transmitted through a leaf keeps its direction.

Crossings are counted per surface (plant, organ, leaflet), not per triangle, so a ray
that passes exactly through the shared edge of two triangles of one leaf is attenuated
once, not twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from tomato3d.io.schemas import LightSimulationSettings
from tomato3d.light.scene import FixtureState
from tomato3d.light.units import MM2_PER_M2

DIRECT_ANGULAR_PROTOCOL = 'direct-angular-1'


def _emit_direction_chunks(callback: Callable[[Mapping[str, Any]], None], limit: int,
                           fixture_id: str, contributions: np.ndarray,
                           directions: np.ndarray, reachable: np.ndarray) -> None:
    """Borrowed read-only diagnostic snapshots, bounded to limit paths per call.

    Copies isolate solver arrays even if a caller deliberately resets write flags.
    No buffer may be retained without copying: reuse after callback return is allowed.
    Paths are receiver-major, all zeros and unknowns included; exceptions propagate.
    """
    receivers, emitters = contributions.shape
    total = receivers * emitters
    for start in range(0, total, limit):
        stop = min(start + limit, total)
        index = np.arange(start, stop, dtype=np.int64)
        q = contributions.reshape(-1)[start:stop].copy()
        u = directions.reshape(-1, 3)[start:stop].copy()
        valid = np.isfinite(q)
        direction_valid = reachable.reshape(-1)[start:stop].copy()
        if np.isinf(q).any() or np.any(q[valid] < 0) or not np.isfinite(u).all():
            raise ValueError('invalid_actual_direction_contribution')
        event = dict(protocol=DIRECT_ANGULAR_PROTOCOL, kind='paths', fixture_id=fixture_id,
            receiver_count=receivers, emitter_count=emitters, expected_path_count=total,
            path_start=start, path_stop=stop, receiver_index=index // emitters, emitter_index=index % emitters,
            incident_ppfd_umol_m2_s=q, arrival_unit_grow_xyz=u, ppfd_valid=valid, direction_valid=direction_valid,
            missing_reason=tuple(None if ok else 'leaf_transmittance_unavailable' for ok in valid),
            direction_missing_reason=tuple(None if ok else 'coincident_receiver_emitter' for ok in direction_valid))
        for value in event.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        callback(MappingProxyType(event))


@dataclass(frozen=True)
class Receivers:
    """Points that receive light, grow frame.

    ``two_sided`` receivers (leaves) take light on either face; one-sided receivers
    (quantum sensors) only on the face their normal points out of.
    """

    points_grow_mm: np.ndarray  # (S, 3)
    unit_normals: np.ndarray  # (S, 3)
    two_sided: bool
    #: Index into the occluder triangles of the surface a point lies on, ``-1`` for none.
    own_triangle: np.ndarray  # (S,) int


@dataclass(frozen=True)
class Occluders:
    """Every surface triangle that can block light, grow frame."""

    vertices_grow_mm: np.ndarray  # (T, 3, 3)
    is_leaf: np.ndarray  # (T,) bool
    #: Identifies the surface (plant, organ, leaflet) a triangle belongs to.
    surface_code: np.ndarray  # (T,) int
    #: Culling group, one per plant: rays are tested against a plant's triangles only when
    #: they cross the plant's bounding box.
    group: np.ndarray  # (T,) int

    @classmethod
    def none(cls) -> Occluders:
        return cls(
            vertices_grow_mm=np.zeros((0, 3, 3)),
            is_leaf=np.zeros((0,), dtype=bool),
            surface_code=np.zeros((0,), dtype=np.int64),
            group=np.zeros((0,), dtype=np.int64),
        )


@dataclass(frozen=True)
class DirectLight:
    """Direct PPFD at every receiver, split by fixture, all fixtures at full output.

    NaN marks an unavailable value: the fixture's inputs are incomplete, or the ray
    crosses a leaf whose transmittance is unavailable.
    """

    fixture_ids: tuple[str, ...]
    ppfd_umol_m2_s: np.ndarray  # (F, S), with shading
    unoccluded_ppfd_umol_m2_s: np.ndarray  # (F, S), as if no surface existed
    upper_side_ppfd_umol_m2_s: np.ndarray  # (F, S), the shaded part arriving on the normal side


def segment_hits_box(
    origins_mm: np.ndarray, segments_mm: np.ndarray, box_min_mm: np.ndarray, box_max_mm: np.ndarray
) -> np.ndarray:
    """``(R,)`` bool: does ``origin + t * segment``, ``t`` in ``[0, 1]``, meet the box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t_first = (box_min_mm - origins_mm) / segments_mm
        t_second = (box_max_mm - origins_mm) / segments_mm
    flat = segments_mm == 0.0
    inside = (origins_mm >= box_min_mm) & (origins_mm <= box_max_mm)
    t_low = np.where(flat, np.where(inside, -np.inf, np.inf), np.minimum(t_first, t_second))
    t_high = np.where(flat, np.where(inside, np.inf, -np.inf), np.maximum(t_first, t_second))
    enter = t_low.max(axis=1)
    leave = t_high.min(axis=1)
    return (enter <= leave) & (leave >= 0.0) & (enter <= 1.0)


def segment_triangle_hits(
    origins_mm: np.ndarray, segments_mm: np.ndarray, vertices_mm: np.ndarray
) -> np.ndarray:
    """``(R, T)`` bool: segment ``r`` crosses triangle ``t`` strictly between its ends.

    Moller-Trumbore with the segment vector as the ray direction, so ``0 < t < 1``.
    """
    edge1 = vertices_mm[:, 1] - vertices_mm[:, 0]  # (T, 3)
    edge2 = vertices_mm[:, 2] - vertices_mm[:, 0]
    p = np.cross(segments_mm[:, None, :], edge2[None, :, :])  # (R, T, 3)
    det = np.einsum("tk,rtk->rt", edge1, p)
    s = origins_mm[:, None, :] - vertices_mm[None, :, 0]  # (R, T, 3)
    q = np.cross(s, edge1[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_det = 1.0 / det
        u = np.einsum("rtk,rtk->rt", s, p) * inv_det
        v = np.einsum("rk,rtk->rt", segments_mm, q) * inv_det
        t = np.einsum("tk,rtk->rt", edge2, q) * inv_det
    return (det != 0.0) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0) & (t < 1.0)


def count_crossings(
    origins_mm: np.ndarray,
    segments_mm: np.ndarray,
    own_triangle: np.ndarray,
    occluders: Occluders,
    max_pairs_per_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Leaves crossed and opaque surfaces met by each segment.

    Returns:
        ``(leaf_crossings, blocked)``: ``(R,)`` int number of distinct leaf surfaces
        crossed, and ``(R,)`` bool whether any opaque surface is crossed.
    """
    n_rays = origins_mm.shape[0]
    blocked = np.zeros(n_rays, dtype=bool)
    if n_rays == 0 or occluders.vertices_grow_mm.shape[0] == 0:
        return np.zeros(n_rays, dtype=np.int64), blocked

    n_codes = int(occluders.surface_code.max()) + 1
    leaf_pair_keys: list[np.ndarray] = []
    for group in np.unique(occluders.group):
        triangles = np.flatnonzero(occluders.group == group)
        group_vertices = occluders.vertices_grow_mm[triangles].reshape(-1, 3)
        in_box = segment_hits_box(
            origins_mm, segments_mm, group_vertices.min(axis=0), group_vertices.max(axis=0)
        )
        rays = np.flatnonzero(in_box)
        if rays.size == 0:
            continue
        triangle_chunk = max(1, min(triangles.size, max_pairs_per_chunk))
        ray_chunk = max(1, max_pairs_per_chunk // triangle_chunk)
        for ray_start in range(0, rays.size, ray_chunk):
            ray_ids = rays[ray_start : ray_start + ray_chunk]
            for triangle_start in range(0, triangles.size, triangle_chunk):
                triangle_ids = triangles[triangle_start : triangle_start + triangle_chunk]
                hits = segment_triangle_hits(
                    origins_mm[ray_ids],
                    segments_mm[ray_ids],
                    occluders.vertices_grow_mm[triangle_ids],
                )
                hits &= own_triangle[ray_ids][:, None] != triangle_ids[None, :]
                leaf = occluders.is_leaf[triangle_ids]
                blocked[ray_ids] |= (hits & ~leaf[None, :]).any(axis=1)
                hit_rows, hit_cols = np.nonzero(hits & leaf[None, :])
                if hit_rows.size:
                    codes = occluders.surface_code[triangle_ids[hit_cols]]
                    leaf_pair_keys.append(ray_ids[hit_rows].astype(np.int64) * n_codes + codes)

    leaf_crossings = np.zeros(n_rays, dtype=np.int64)
    if leaf_pair_keys:
        unique_keys = np.unique(np.concatenate(leaf_pair_keys))
        np.add.at(leaf_crossings, unique_keys // n_codes, 1)
    return leaf_crossings, blocked


def direct_light(
    receivers: Receivers,
    fixtures: tuple[FixtureState, ...],
    occluders: Occluders,
    leaf_transmittance: float | None,
    settings: LightSimulationSettings,
    *,
    diagnostic_callback: Callable[[Mapping[str, Any]], None] | None = None,
    diagnostic_max_paths_per_chunk: int | None = None,
) -> DirectLight:
    """Direct PPFD at every receiver from every fixture at full output.

    Optional synchronous diagnostics use DIRECT_ANGULAR_PROTOCOL. Each receiver-major
    chunk reports actual attenuated/cosine-weighted PPFD before area weighting and
    grow-frame unit vectors pointing receiver -> emitter (opposite photon travel).
    Stable emitter indices refer to the supplied fixture emitter array. Counts and
    contiguous half-open ranges include dark/unknown paths. Unknown fixtures emit a
    single fixture_unavailable event, never zero paths as a substitute for ignorance.
    Buffers/mappings are read-only and borrowed until callback return; copy to retain.
    Exceptions abort the call. Enabling diagnostics requires an explicit positive
    integer chunk budget; disabled calls allocate no diagnostic arrays. Existing
    transport/pair budgets and the DirectLight return are unchanged.
    """
    if diagnostic_callback is not None:
        if not callable(diagnostic_callback):
            raise ValueError('diagnostic_callback_must_be_callable')
        if type(diagnostic_max_paths_per_chunk) is not int or diagnostic_max_paths_per_chunk < 1:
            raise ValueError('positive_diagnostic_chunk_budget_required')
        ids = [state.fixture_id for state in fixtures]
        if len(set(ids)) != len(ids) or any(fid in ('all', '__all__') for fid in ids):
            raise ValueError('unique_real_diagnostic_fixture_ids_required')
        if any(state.emitters is not None and not len(state.emitters.positions_grow_mm) for state in fixtures):
            raise ValueError('empty_emitter_inventory_is_not_unavailable')
    n_fixtures = len(fixtures)
    n_receivers = receivers.points_grow_mm.shape[0]
    shaded = np.zeros((n_fixtures, n_receivers))
    unoccluded = np.zeros((n_fixtures, n_receivers))
    upper = np.zeros((n_fixtures, n_receivers))
    transmittance = np.nan if leaf_transmittance is None else leaf_transmittance

    for f, state in enumerate(fixtures):
        emitters = state.emitters
        if emitters is None:
            shaded[f] = unoccluded[f] = upper[f] = np.nan
            if diagnostic_callback is not None:
                diagnostic_callback(MappingProxyType(dict(protocol=DIRECT_ANGULAR_PROTOCOL,
                    kind='fixture_unavailable', fixture_id=state.fixture_id, receiver_count=n_receivers,
                    emitter_count=None, expected_path_count=None,
                    missing_reason='fixture_inputs_unavailable:' + ','.join(state.missing_inputs))))
            continue
        to_emitter = emitters.positions_grow_mm[None, :, :] - receivers.points_grow_mm[:, None, :]
        d2 = np.einsum("sek,sek->se", to_emitter, to_emitter)  # (S, E)
        reachable = d2 > 0.0
        safe_d2 = np.where(reachable, d2, 1.0)
        direction = to_emitter / np.sqrt(safe_d2)[:, :, None]

        cos_emit = -np.einsum("sek,k->se", direction, emitters.aim)
        in_beam = reachable & (cos_emit > 0.0) & (cos_emit >= emitters.cos_cutoff)
        intensity = np.where(
            in_beam,
            emitters.axial_intensity_umol_s_sr
            * np.power(np.clip(cos_emit, 0.0, None), emitters.cosine_exponent),
            0.0,
        )
        cos_incidence = np.einsum("sek,sk->se", direction, receivers.unit_normals)
        facing = np.abs(cos_incidence) if receivers.two_sided else np.clip(cos_incidence, 0.0, None)
        irradiance = intensity * facing / safe_d2 * MM2_PER_M2  # (S, E)

        attenuation = np.ones_like(irradiance)
        lit_s, lit_e = np.nonzero(irradiance > 0.0)
        if lit_s.size and occluders.vertices_grow_mm.shape[0]:
            ray_direction = direction[lit_s, lit_e]
            origins = receivers.points_grow_mm[lit_s] + settings.ray_offset_mm * ray_direction
            ends = emitters.positions_grow_mm[lit_e] - settings.ray_offset_mm * ray_direction
            leaf_crossings, blocked = count_crossings(
                origins,
                ends - origins,
                receivers.own_triangle[lit_s],
                occluders,
                settings.max_ray_triangle_pairs_per_chunk,
            )
            with np.errstate(invalid="ignore"):
                through_leaves = np.where(
                    leaf_crossings > 0, np.power(transmittance, leaf_crossings), 1.0
                )
            attenuation[lit_s, lit_e] = np.where(blocked, 0.0, through_leaves)

        shaded_per_emitter = irradiance * attenuation
        unoccluded[f] = irradiance.sum(axis=1)
        shaded[f] = shaded_per_emitter.sum(axis=1)
        upper[f] = np.where(cos_incidence > 0.0, shaded_per_emitter, 0.0).sum(axis=1)
        if diagnostic_callback is not None:
            _emit_direction_chunks(diagnostic_callback, diagnostic_max_paths_per_chunk,
                                   state.fixture_id, shaded_per_emitter, direction, reachable)

    return DirectLight(
        fixture_ids=tuple(state.fixture_id for state in fixtures),
        ppfd_umol_m2_s=shaded,
        unoccluded_ppfd_umol_m2_s=unoccluded,
        upper_side_ppfd_umol_m2_s=upper,
    )
