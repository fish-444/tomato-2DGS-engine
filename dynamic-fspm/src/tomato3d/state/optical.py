"""P1.7 - located leaflet optical surfaces for P2, with an explicit readiness per leaflet.

Conventions handed to P2 (also written into every manifest row):

* frame ``plant_local_mm``; P2 places each instance with ``canopy_placement.json``;
* ``face_convention = adaxial_ccw``: triangle vertices run counter-clockwise seen from the upper
  (adaxial) side, so ``(v1 - v0) x (v2 - v0)`` points out of the adaxial face;
* ``area_convention = one_sided_lamina_full_outline``: the sum of triangle areas is the one-sided
  lamina area on the full original outline (MTG 13); incident/absorbed light per leaflet is
  computed on that area, both faces exposed;
* ``external_leaflet_id = <plant_instance_id>|<vertex_id>`` is stable across captures and splits
  back to the instance and the MTG vertex.

Readiness:

* ``supported`` - located surface from truth (fixtures) or, in future, from a supported 3-D
  reconstruction. Real photos: board-free 3-D (Q2) is not implemented, so never today.
* ``conditional_completion`` - generator candidates constrained by an observed 2-D mask. Flagged,
  kept in their own files, never mixed with observed geometry. Two adapters: :func:`complete_compound`
  (pre-built pool compounds, rotate + scale only) and :func:`generate_candidates` (``build_compound``
  numpy backend conditioned on the observation, P1 fix A4).
* ``unavailable`` - with the reason. Never a bbox, hull, compound envelope or area+angle tuple.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.io.tables import write_table
from tomato3d.state import fixture_truth as ft
from tomato3d.state import geometry as g
from tomato3d.state import schema as s
from tomato3d.state.topraster import OrthoCamera, render_top, rotate_z

FACE_CONVENTION = ft.FACE_CONVENTION
AREA_CONVENTION = ft.AREA_CONVENTION
Q2_UNAVAILABLE = ("board-free 3-D reconstruction (Q2) is not implemented; a top view alone gives no located "
                  "leaflet surface")


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("#", "~").replace("|", "__")


def write_surfaces(out_dir: Path, rel_root: Path, plant_instance_id: str, capture_id: str,
                   surfaces: Mapping[str, tuple[np.ndarray, np.ndarray]], tag: str) -> tuple[str, str, dict[str, int]]:
    """One CSV of triangles per (instance, capture, tag). Returns (relative path, sha256, n per leaflet)."""
    rows = []
    counts: dict[str, int] = {}
    for vid, (verts, faces) in sorted(surfaces.items()):
        ext = s.external_leaflet_id(plant_instance_id, vid)
        v = np.asarray(verts, float)
        for k, f in enumerate(np.asarray(faces, int)):
            a, b, c = v[f[0]], v[f[1]], v[f[2]]
            rows.append(dict(external_leaflet_id=ext, triangle_id=str(k), frame="plant_local_mm",
                             v0_x_mm=s.fmt(a[0], 4), v0_y_mm=s.fmt(a[1], 4), v0_z_mm=s.fmt(a[2], 4),
                             v1_x_mm=s.fmt(b[0], 4), v1_y_mm=s.fmt(b[1], 4), v1_z_mm=s.fmt(b[2], 4),
                             v2_x_mm=s.fmt(c[0], 4), v2_y_mm=s.fmt(c[1], 4), v2_z_mm=s.fmt(c[2], 4)))
        counts[vid] = len(faces)
    name = f"{_safe(plant_instance_id)}__{_safe(capture_id)}__{tag}.csv"
    path = out_dir / name
    write_table(path, rows, s.OPTICAL_TRIANGLES_COLUMNS)
    return (rel_root / name).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest(), counts


def manifest_row(plant_instance_id: str, capture_id: str, vertex_id: str, readiness: str, reason: str,
                 geometry_source: str, evidence: str, surface_file: str = UNAVAILABLE,
                 surface_sha256: str = UNAVAILABLE, n_triangles: int | None = None,
                 area_mm2: float | None = None) -> dict[str, str]:
    return dict(plant_instance_id=plant_instance_id, source_plant_id=s.source_of_instance(plant_instance_id),
                capture_id=capture_id, external_leaflet_id=s.external_leaflet_id(plant_instance_id, vertex_id),
                vertex_id=vertex_id, compound_id=vertex_id.rsplit("/", 1)[0] if vertex_id.count("/") >= 2 else vertex_id,
                frame="plant_local_mm", surface_file=surface_file, surface_sha256=surface_sha256,
                n_triangles=UNAVAILABLE if n_triangles is None else str(n_triangles),
                one_sided_area_mm2=s.fmt(area_mm2, 3), face_convention=FACE_CONVENTION if readiness != "unavailable"
                else UNAVAILABLE, area_convention=AREA_CONVENTION if readiness != "unavailable" else UNAVAILABLE,
                geometry_source=geometry_source, optical_readiness=readiness, readiness_reason=reason,
                evidence=evidence)


def export_truth(out_dir: Path, rel_root: Path, plant: ft.FixturePlant, plant_instance_id: str,
                 capture_id: str) -> list[dict[str, str]]:
    """Fixture: truth leaflet meshes as ``supported`` surfaces (evidence synthetic_fixture)."""
    surfaces = ft.truth_surfaces(plant)
    rel, sha, counts = write_surfaces(out_dir, rel_root, plant_instance_id, capture_id, surfaces, "truth")
    return [manifest_row(plant_instance_id, capture_id, vid, "supported",
                         "synthetic fixture truth mesh (never photo reconstruction)", "synthetic_truth",
                         "synthetic_fixture", rel, sha, counts[vid], float(g.triangle_areas(*surfaces[vid]).sum()))
            for vid in sorted(surfaces)]


def unavailable_rows(plant_instance_id: str, capture_id: str, vertex_ids: Sequence[str], reason: str,
                     evidence: str) -> list[dict[str, str]]:
    return [manifest_row(plant_instance_id, capture_id, vid, "unavailable", reason, "none", evidence)
            for vid in vertex_ids]


# --------------------------------------------------------------------------------------
# Conditional completion from a generator pool, constrained by the observed 2-D mask
# --------------------------------------------------------------------------------------


@dataclass
class Candidate:
    candidate_id: str
    donor_ref: str
    yaw_deg: float
    scale: float
    coverage: float
    background_violation: float
    meshes: dict[str, tuple[np.ndarray, np.ndarray]]


def _distal(mask: np.ndarray, camera: OrthoCamera) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    xy = camera.unproject_xy(np.stack([xs, ys], axis=1).astype(float))
    r = np.hypot(xy[:, 0], xy[:, 1])
    k = int(np.argmax(r))
    return float(np.degrees(np.arctan2(xy[k, 1], xy[k, 0]))), float(r[k])


def complete_compound(observed_mask: np.ndarray, plant_visible: np.ndarray, capture_yaw_deg: float,
                      camera: OrthoCamera, pool: Sequence[tuple[str, ft.FixtureCompound]],
                      coverage_tolerance: float, max_kept: int, compound_vertex_id: str) -> tuple[list[Candidate], dict[str, Any]]:
    """Pool-donor adapter (L-Py's ``build_compound`` is not available in the project env).

    Each donor compound (generator output from other fixture scenes, fixed seeds) is rotated about
    the stem so its projected distal end has the observed azimuth and scaled uniformly so the
    distal radius matches; it is kept only if its amodal footprint covers ``coverage_tolerance`` of
    the observed visible pixels. Ranked by background violation; a ranking is not a probability.
    """
    target = _distal(observed_mask, camera)
    report: dict[str, Any] = dict(n_pool=len(pool), n_generated=0, n_kept=0, rejected_low_coverage=0,
                                  adapter="pool_donor_rotate_scale", prior_status="user_assumed_unvalidated")
    if target is None:
        report["candidate_status"] = "unavailable_empty_observation"
        return [], report
    az_obs, r_obs = target
    obs_px = int(observed_mask.sum())
    kept: list[Candidate] = []
    for k, (ref, comp) in enumerate(pool):
        meshes = {tag: (v - comp.attachment_mm * np.array([1.0, 1.0, 0.0]), f) for tag, (v, f) in comp.meshes.items()}
        allv = np.vstack([v for v, _ in meshes.values()])
        r = np.hypot(allv[:, 0], allv[:, 1])
        far = int(np.argmax(r))
        if r[far] < 1e-6:
            continue
        az_d = float(np.degrees(np.arctan2(allv[far, 1], allv[far, 0])))
        scale = r_obs / float(r[far])
        yaw = (az_obs - capture_yaw_deg) - az_d
        placed = {tag: (rotate_z(v * scale, yaw), f) for tag, (v, f) in meshes.items()}
        shown = {tag: (rotate_z(v, capture_yaw_deg), f) for tag, (v, f) in placed.items()}
        rend = render_top({f"x/{t}": m for t, m in shown.items()}, camera)
        amodal = np.zeros_like(observed_mask)
        for m in rend.amodal.values():
            amodal |= m
        report["n_generated"] += 1
        coverage = float((amodal & observed_mask).sum()) / obs_px
        if coverage < coverage_tolerance:
            report["rejected_low_coverage"] += 1
            continue
        violation = float((amodal & ~plant_visible).sum()) / max(int(amodal.sum()), 1)
        kept.append(Candidate(f"{compound_vertex_id}#cand{k:03d}", ref, yaw, scale, coverage, violation,
                              {f"{compound_vertex_id}/{t}": m for t, m in placed.items()}))
    kept.sort(key=lambda c: (c.background_violation, c.candidate_id))
    kept = kept[:max_kept]
    report["n_kept"] = len(kept)
    report["candidate_status"] = "kept" if kept else "rejected"
    return kept, report


@dataclass(frozen=True)
class ObservedCompound:
    """Top-view geometry of one observed compound in the plant frame (fixture: declared ortho camera)."""

    azimuth_deg: float
    distal_radius_mm: float
    left_reach_mm: float
    right_reach_mm: float


def observed_geometry(mask: np.ndarray, capture_yaw_deg: float, camera: OrthoCamera) -> ObservedCompound | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    xy = camera.unproject_xy(np.stack([xs, ys], axis=1).astype(float))
    xy = rotate_z(np.c_[xy, np.zeros(len(xy))], -capture_yaw_deg)[:, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    k = int(np.argmax(r))
    az = float(np.degrees(np.arctan2(xy[k, 1], xy[k, 0])))
    left = np.array([-np.sin(np.radians(az)), np.cos(np.radians(az))])
    lateral = xy @ left
    return ObservedCompound(az % 360.0, float(r[k]), float(max(lateral.max(), 0.0)), float(max(-lateral.min(), 0.0)))


def terminal_pins(terminal_mask: np.ndarray | None, observed: ObservedCompound, capture_yaw_deg: float,
                  camera: OrthoCamera, gen_cfg: Mapping[str, Any], contours: Sequence[np.ndarray]
                  ) -> tuple[dict[str, Any], dict[str, str]]:
    """``fixed=`` pins from an observed terminal leaflet, each kept only inside the declared generator range.

    Returns (pins, notes). ``contour_index.T`` = the pool contour nearest the observed outline after
    ``normalize_contour`` (mirror allowed); ``terminal_rel_length`` = observed terminal length / observed
    distal radius; ``width_scale.T`` = observed aspect / chosen contour aspect. A top view foreshortens the
    terminal leaflet, so these are conditioning choices, not measurements.
    """
    import cv2

    from tomato3d.synth.config import range_of
    from tomato3d.synth.leaflet_surface import normalize_contour

    notes: dict[str, str] = {}
    if terminal_mask is None or not terminal_mask.any():
        return {}, {"terminal": "no observed terminal leaflet; nothing pinned"}
    found, _ = cv2.findContours(terminal_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(found, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(outline) < 8:
        return {}, {"terminal": "observed terminal outline too small to normalise; nothing pinned"}
    n_pts = len(contours[0])
    norm = normalize_contour(outline, n_pts)
    obs_pts = norm["points"]
    dists = []
    for c in contours:
        c = np.asarray(c, float)
        d = min(float(np.linalg.norm(obs_pts - c, axis=1).mean()),
                float(np.linalg.norm(obs_pts * [1.0, -1.0] - c, axis=1).mean()))
        dists.append(d)
    best = int(np.argmin(dists))
    pins: dict[str, Any] = {"contour_index.T": best}
    notes["contour_index.T"] = f"nearest normalised pool outline (mean point distance {dists[best]:.4f})"
    length_mm = float(norm["length_px"]) * camera.mm_per_px
    rel = length_mm / observed.distal_radius_mm if observed.distal_radius_mm > 0 else None
    lo, hi = range_of(gen_cfg, "terminal_rel_length")
    if rel is not None and lo <= rel <= hi:
        pins["terminal_rel_length"] = rel
        notes["terminal_rel_length"] = f"observed {rel:.3f} within [{lo}, {hi}]"
    else:
        notes["terminal_rel_length"] = f"observed {rel} outside [{lo}, {hi}]; not pinned (generator draws it)"
    pool_aspect = float(np.ptp(np.asarray(contours[best], float)[:, 1]))
    width = float(norm["aspect"]) / pool_aspect if pool_aspect > 0 else None
    wlo, whi = range_of(gen_cfg, "leaflet_width_scale")
    if width is not None and wlo <= width <= whi:
        pins["width_scale.T"] = width
        notes["width_scale.T"] = f"observed {width:.3f} within [{wlo}, {whi}]"
    else:
        notes["width_scale.T"] = f"observed {width} outside [{wlo}, {whi}]; not pinned"
    return pins, notes


def generate_candidates(observed_mask: np.ndarray, plant_visible: np.ndarray, capture_yaw_deg: float,
                        camera: OrthoCamera, gen_cfg: Mapping[str, Any], contours: Sequence[np.ndarray],
                        n_samples: int, extent_slack_rel: float, coverage_tolerance: float, max_kept: int,
                        compound_vertex_id: str, terminal_mask: np.ndarray | None = None,
                        refine_top: int = 0, refine_yaw_deg: Sequence[float] = (0.0,),
                        refine_scale: Sequence[float] = (1.0,)) -> tuple[list[Candidate], dict[str, Any]]:
    """Generator candidates (``tomato3d.synth.compound_leaf.build_compound``, numpy backend) built from
    the observation (phynotype2 6.3): azimuth and distal radius from the observed mask, lateral extents
    from the observed reach widened by ``extent_slack_rel``, elevation and every other interior value drawn
    from the declared generator ranges (``user_assumed``), the terminal leaflet pinned through ``fixed=``
    when its observed value lies inside the generator range. Each sample is then rotated about the stem
    and scaled uniformly so its distal end meets the observed one (rotation + scale fit; per-pair sizes
    come from the generator's own extent contract, not from a post-hoc rescale). The ``refine_top`` best
    samples are then fitted locally over the declared grids ``refine_yaw_deg`` x ``refine_scale`` (rigid
    rotation about the stem and uniform scale only). Kept only if its amodal footprint covers
    ``coverage_tolerance`` of the observed pixels; background violation is reported, never hidden.
    Seeds are fixed (replayable)."""
    from tomato3d.synth.compound_leaf import CompoundRejected, build_compound
    from tomato3d.synth.config import constant, make_rng, range_of
    from tomato3d.synth.leaflet_surface import surface_points, triangulate_contour

    report: dict[str, Any] = dict(adapter="generator_numpy_conditioned", n_samples=n_samples, n_generated=0,
                                  n_contract_rejected=0, rejected_low_coverage=0, n_kept=0,
                                  prior_status="user_assumed_unvalidated")
    obs = observed_geometry(observed_mask, capture_yaw_deg, camera)
    if obs is None:
        report["candidate_status"] = "unavailable_empty_observation"
        return [], report
    pins, notes = terminal_pins(terminal_mask, obs, capture_yaw_deg, camera, gen_cfg, contours)
    report["pins"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pins.items()}
    report["pin_notes"] = notes
    grid_n = int(constant(gen_cfg, "leaflet_mesh_grid_n"))
    elev_lo, elev_hi = range_of(gen_cfg, "leaf_elevation_deg")
    obs_px = int(observed_mask.sum())
    tri_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    samples: list[tuple[int, str, dict[str, tuple[np.ndarray, np.ndarray]], float, float]] = []

    def score(meshes: dict[str, tuple[np.ndarray, np.ndarray]], yaw: float, scale: float) -> tuple[float, float]:
        shown = {f"x/{tag}": (rotate_z(rotate_z(v * scale, yaw), capture_yaw_deg), f) for tag, (v, f) in meshes.items()}
        rend = render_top(shown, camera)
        amodal = np.zeros_like(observed_mask)
        for m in rend.amodal.values():
            amodal |= m
        return (float((amodal & observed_mask).sum()) / obs_px,
                float((amodal & ~plant_visible).sum()) / max(int(amodal.sum()), 1))

    for k in range(n_samples):
        rng = make_rng(k, "p1_completion", compound_vertex_id)
        elev = float(rng.uniform(elev_lo, elev_hi))
        e, a = np.radians(elev), np.radians(obs.azimuth_deg)
        compound = dict(compound_id=compound_vertex_id, attachment_mm=[0.0, 0.0, 0.0],
                        direction_unit=[float(np.cos(e) * np.cos(a)), float(np.cos(e) * np.sin(a)), float(np.sin(e))],
                        rachis_arc_length_mm=obs.distal_radius_mm / float(np.cos(e)),
                        left_extent_mm=max(obs.left_reach_mm, 1.0) * (1.0 + extent_slack_rel),
                        right_extent_mm=max(obs.right_reach_mm, 1.0) * (1.0 + extent_slack_rel))
        try:
            interior, _ = build_compound(dict(gen_cfg), compound, list(contours), k, "p1_candidate", fixed=pins or None,
                                         backend="numpy")
        except CompoundRejected:
            report["n_contract_rejected"] += 1
            continue
        report["n_generated"] += 1
        meshes = {}
        for leaf in interior["leaflets"]:
            ci = int(leaf["contour_index"])
            if ci not in tri_cache:
                tri_cache[ci] = triangulate_contour(np.asarray(contours[ci], float), grid_n)
            uv, faces = tri_cache[ci]
            meshes[leaf["tag"]] = (surface_points(uv, leaf), faces)
        allv = np.vstack([v for v, _ in meshes.values()])
        r = np.hypot(allv[:, 0], allv[:, 1])
        far = int(np.argmax(r))
        if r[far] < 1e-6:
            continue
        scale = obs.distal_radius_mm / float(r[far])
        yaw = obs.azimuth_deg - float(np.degrees(np.arctan2(allv[far, 1], allv[far, 0])))
        samples.append((k, f"build_compound(numpy, seed={k}, attempt={interior['attempt']}, elevation={elev:.1f})",
                        meshes, yaw, scale))
    fits = []
    for k, ref, meshes, yaw, scale in samples:
        cov, vio = score(meshes, yaw, scale)
        fits.append([cov, vio, k, ref, meshes, yaw, scale, "distal_alignment"])
    fits.sort(key=lambda f: -f[0])
    for fit in fits[:refine_top]:
        best = fit[:2] + fit[5:7]
        for dyaw in refine_yaw_deg:
            for sc in refine_scale:
                cov, vio = score(fit[4], fit[5] + dyaw, fit[6] * sc)
                if cov > best[0] + 1e-12:
                    best = [cov, vio, fit[5] + dyaw, fit[6] * sc]
        if best[0] > fit[0]:
            fit[0], fit[1], fit[5], fit[6], fit[7] = best[0], best[1], best[2], best[3], "distal_alignment+local_fit"
    report["best_coverage"] = round(max((f[0] for f in fits), default=0.0), 4)
    kept: list[Candidate] = []
    for cov, vio, k, ref, meshes, yaw, scale, how in fits:
        if cov < coverage_tolerance:
            report["rejected_low_coverage"] += 1
            continue
        placed = {f"{compound_vertex_id}/{tag}": (rotate_z(v * scale, yaw), f) for tag, (v, f) in meshes.items()}
        kept.append(Candidate(f"{compound_vertex_id}#gen{k:03d}", f"{ref}; fit={how}", yaw, scale, cov, vio, placed))
    kept.sort(key=lambda c: (c.background_violation, c.candidate_id))
    kept = kept[:max_kept]
    report["n_kept"] = len(kept)
    report["candidate_status"] = "kept" if kept else "rejected"
    return kept, report


def export_candidates(out_dir: Path, rel_root: Path, plant_instance_id: str, capture_id: str,
                      candidates: Sequence[Candidate]) -> list[dict[str, str]]:
    rows = []
    for cand in candidates:
        rel, sha, counts = write_surfaces(out_dir, rel_root, plant_instance_id, capture_id, cand.meshes,
                                          f"cand_{_safe(cand.candidate_id)}")
        for vid, (verts, faces) in sorted(cand.meshes.items()):
            rows.append(manifest_row(
                plant_instance_id, capture_id, vid, "conditional_completion",
                f"generated candidate {cand.candidate_id} from {cand.donor_ref}: yaw {cand.yaw_deg:.2f} deg, "
                f"scale {cand.scale:.3f}, coverage {cand.coverage:.3f}, background violation "
                f"{cand.background_violation:.3f}; user_assumed priors, not an observation",
                "generated_candidate", "synthetic_fixture", rel, sha, counts[vid],
                float(g.triangle_areas(verts, faces).sum())))
    return rows
