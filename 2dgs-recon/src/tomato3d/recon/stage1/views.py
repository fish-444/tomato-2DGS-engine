"""Synthetic observations for the stage-1 test (E8-1): top + side ID maps from a generator plant, truth kept apart.

Reuses the project's renderer (:func:`tomato3d.state.topraster.render_top`, orthographic, painter's order on the
mean triangle height) and the generator's truth loader (:func:`tomato3d.state.fixture_truth.load_plant`). Side views
are the same orthographic rasteriser after a rigid rotation (camera horizontal, looking at the plant from azimuth 0
or 90 deg); stems and rachises are drawn as thin square tubes so that a side view shows where a compound meets the
stem. Real side photos are perspective - this is a SYNTHETIC test only.

What the reconstruction may see (:class:`TopObs`, :class:`SideObs`): visible ID maps; per leaflet mask only its
compound membership and a terminal flag (a detector class exists for the terminal leaflet); leaflet tags, amodal
masks, depths and 3-D geometry stay in :class:`Truth` for scoring. Cross-view identity (which side mask belongs to
which compound) is GIVEN in this synthetic test - a real product must match it (reported as an assumption).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tomato3d.state import fixture_truth as ft
from tomato3d.state import geometry as g
from tomato3d.state.topraster import OrthoCamera, render_top

SIDE_AZIMUTHS = (0.0, 90.0)
STEM_TUBE_MM = 2.0
RACHIS_TUBE_MM = 0.8


def tube_mesh(points: np.ndarray, width_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Square tube along a polyline (4 long faces per segment); visible from any horizontal or vertical view."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    verts, faces = [], []
    h = width_mm / 2.0
    for i in range(len(p) - 1):
        d = p[i + 1] - p[i]
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d /= n
        a = np.cross(d, [0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.cross(d, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        corners = [a * h + b * h, -a * h + b * h, -a * h - b * h, a * h - b * h]
        base = len(verts)
        for c in corners:
            verts.append(p[i] + c)
        for c in corners:
            verts.append(p[i + 1] + c)
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append([base + k, base + k2, base + 4 + k2])
            faces.append([base + k, base + 4 + k2, base + 4 + k])
    return np.asarray(verts, float), np.asarray(faces, np.int32)


def side_transform(points: np.ndarray, azimuth_deg: float) -> np.ndarray:
    """Plant mm -> a frame whose +Z points to a horizontal side camera at ``azimuth_deg`` (0: camera at -Y looking
    +Y; 90: camera at +X looking -X) and whose image up is plant +Z. Returns (u_mm, up_mm, toward_camera_mm)."""
    p = np.asarray(points, float)
    a = np.radians(azimuth_deg)
    toward = np.array([np.sin(a), -np.cos(a), 0.0])      # unit vector from the plant toward the camera
    right = np.array([np.cos(a), np.sin(a), 0.0])         # image right
    return np.stack([p @ right, p[..., 2], p @ toward], axis=-1)


@dataclass
class TopObs:
    camera: OrthoCamera
    id_map: np.ndarray                 # int32 label image (0 background)
    labels: dict[int, dict]            # label -> {obs_id, compound, kind: leaflet|rachis|stem, is_terminal}
    stem_xy_mm: np.ndarray             # observed stem position in the top view (pot centre), plant mm

    def mask(self, label: int) -> np.ndarray:
        return self.id_map == label

    def mm2_per_px(self) -> float:
        return self.camera.mm_per_px ** 2


@dataclass
class SideObs:
    azimuth_deg: float
    camera: OrthoCamera
    id_map: np.ndarray
    labels: dict[int, str]             # label -> compound id | "stem"

    def to_px(self, u_mm: float, up_mm: float) -> tuple[float, float]:
        return (self.camera.center_px + u_mm / self.camera.mm_per_px, self.camera.center_px - up_mm / self.camera.mm_per_px)

    def to_mm(self, col: float, row: float) -> tuple[float, float]:
        return ((col - self.camera.center_px) * self.camera.mm_per_px, (self.camera.center_px - row) * self.camera.mm_per_px)


@dataclass
class Truth:
    plant: ft.FixturePlant
    leaflet_tag: dict[str, str]                  # obs_id -> "<cid>/<tag>"
    leaflet_area_mm2: dict[str, float]           # "<cid>/<tag>" -> one-sided mesh area
    compound_area_mm2: dict[str, float]
    compound_rank: dict[str, int]                # registry rank, U5 (A7): generator node index -> max(1, i - 1) on M
    compound_axis: dict[str, str]
    attachment_z_mm: dict[str, float]
    elevation_deg: dict[str, float]              # base -> terminal tip
    segment_elevation_deg: dict[str, tuple[float, float]]
    visible_fraction: dict[str, float]           # "<cid>/<tag>" -> visible / amodal (top)
    above_pairs: dict[tuple[str, str], int] = field(default_factory=dict)  # (A, B) -> px where A covers B (top)
    compound_node_index: dict[str, int] = field(default_factory=dict)      # generator rank_from_bottom = node index (ordering)

    def registry(self) -> dict[str, dict]:
        """Human-record stand-in for the recon steps: U5 rank + node index (ordering) + axis."""
        return {c: dict(rank=self.compound_rank[c], node_index=self.compound_node_index.get(c, self.compound_rank[c]),
                        axis=self.compound_axis[c]) for c in self.compound_rank}


def compound_ids(plant: ft.FixturePlant) -> dict[int, str]:
    return {c.serial: plant.compound_vertex_id(c) for c in plant.compounds}


def build(plant: ft.FixturePlant, mm_per_px: float = 0.2, seed: int = 0) -> tuple[TopObs, list[SideObs], Truth]:
    rng = np.random.default_rng(seed)
    leaf_meshes = plant.leaflet_meshes()
    extra: dict[str, tuple[np.ndarray, np.ndarray]] = {"stem": tube_mesh(plant.stem_points_mm, STEM_TUBE_MM)}
    for comp in plant.compounds:
        cid = plant.compound_vertex_id(comp)
        tube = tube_mesh(comp.rachis_points_mm, RACHIS_TUBE_MM)
        if len(tube[0]):
            extra[f"{cid}/R"] = tube
    allv = np.vstack([v for v, _ in list(leaf_meshes.values()) + list(extra.values()) if len(v)])
    reach = float(np.max(np.abs(allv[:, :3]))) + 10.0
    size = int(np.ceil(2 * reach / mm_per_px))
    cam = OrthoCamera(size, mm_per_px)

    # ---- top view: leaflets + rachis tubes (stem seen end-on)
    top_meshes = {**leaf_meshes, **{k: v for k, v in extra.items() if k != "stem"}}
    tr = render_top(top_meshes, cam)
    # anonymise leaflet ids: per compound, numbered by a random permutation (no tag leaks to the reconstruction)
    labels: dict[int, dict] = {}
    leaflet_tag: dict[str, str] = {}
    by_comp: dict[str, list[str]] = {}
    for lid in tr.ids:
        if lid.endswith("/R"):
            continue
        by_comp.setdefault(lid.rsplit("/", 1)[0], []).append(lid)
    anon: dict[str, str] = {}
    for cid, lids in by_comp.items():
        for k, lid in enumerate(rng.permutation(lids), start=1):
            anon[lid] = f"{cid}/l{k:02d}"
            leaflet_tag[anon[lid]] = lid
    id_map = tr.visible.astype(np.int32)
    for k, lid in enumerate(tr.ids, start=1):
        if lid.endswith("/R"):
            labels[k] = dict(obs_id=lid, compound=lid[:-2], kind="rachis", is_terminal=False)
        else:
            labels[k] = dict(obs_id=anon[lid], compound=lid.rsplit("/", 1)[0], kind="leaflet",
                             is_terminal=lid.endswith("/T"))
    top = TopObs(cam, id_map, labels, np.asarray(plant.stem_points_mm[0][:2], float))

    # ---- side views: compound-level labels (leaflets + rachis of a compound share its id) + stem
    sides = []
    for az in SIDE_AZIMUTHS:
        meshes = {}
        for lid, (v, f) in {**leaf_meshes, **extra}.items():
            meshes[lid] = (side_transform(v, az), f)
        sr = render_top(meshes, cam)
        side_map = np.zeros_like(sr.visible, np.int32)
        slabels: dict[int, str] = {}
        comp_label: dict[str, int] = {}
        for k, lid in enumerate(sr.ids, start=1):
            owner = "stem" if lid == "stem" else (lid[:-2] if lid.endswith("/R") else lid.rsplit("/", 1)[0])
            if owner not in comp_label:
                comp_label[owner] = len(comp_label) + 1
                slabels[comp_label[owner]] = owner
            side_map[sr.visible == k] = comp_label[owner]
        sides.append(SideObs(az, cam, side_map, slabels))

    # ---- truth (scoring only)
    leaflet_area = {lid: float(g.triangle_areas(*m).sum()) for lid, m in leaf_meshes.items()}
    comp_area: dict[str, float] = {}
    for lid, a in leaflet_area.items():
        comp_area[lid.rsplit("/", 1)[0]] = comp_area.get(lid.rsplit("/", 1)[0], 0.0) + a
    from tomato3d.state import ranks as rk
    rank, axis, attach, elev, seg, node = {}, {}, {}, {}, {}, {}
    for comp in plant.compounds:
        cid = plant.compound_vertex_id(comp)
        node[cid] = comp.gen_rank_from_bottom
        rank[cid] = rk.rank1010_to_u5(comp.gen_rank_from_bottom, shared_first=(comp.axis == "M"))
        axis[cid] = comp.axis
        attach[cid] = float(comp.rachis_points_mm[0][2])
        pts = ft.compound_points(plant, comp)
        tip = pts.terminal_tip if pts.terminal_tip is not None else pts.rachis_end
        elev[cid] = float(g.elevation_deg(tip - pts.p0))
        path = [pts.p0, *pts.nodes, tip]
        mid = g.path_midpoint(path)
        seg[cid] = (float(g.elevation_deg(mid - pts.p0)), float(g.elevation_deg(tip - mid)))
    vis_frac = {}
    for k, lid in enumerate(tr.ids, start=1):
        if lid.endswith("/R"):
            continue
        am = tr.amodal[lid].sum()
        vis_frac[lid] = float((tr.visible == k).sum() / am) if am else 0.0
    above: dict[tuple[str, str], int] = {}
    leaf_ids = [lid for lid in tr.ids if not lid.endswith("/R")]
    for i, a in enumerate(leaf_ids):
        for b in leaf_ids[i + 1:]:
            both = tr.amodal[a] & tr.amodal[b]
            if not both.any():
                continue
            ka, kb = tr.ids.index(a) + 1, tr.ids.index(b) + 1
            na, nb = int((both & (tr.visible == ka)).sum()), int((both & (tr.visible == kb)).sum())
            if na:
                above[(a, b)] = na
            if nb:
                above[(b, a)] = nb
    truth = Truth(plant, leaflet_tag, leaflet_area, comp_area, rank, axis, attach, elev, seg, vis_frac, above, node)
    return top, sides, truth
