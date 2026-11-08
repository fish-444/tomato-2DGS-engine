"""Orthographic top-view rasteriser for synthetic fixtures (plant-local mm -> image px).

Why a separate rasteriser: :mod:`tomato3d.synth.raster` needs a board-posed camera, and the
tracking fixtures need controlled pot rotations, removals and additions without re-running the
generator (which needs L-Py). This one draws leaflet triangles only (stem, rachis and petiolules
are not drawn - stated in every fixture manifest), looking straight down ``-Z``:

    u_px = cx_px + x_mm / mm_per_px
    v_px = cy_px - y_mm / mm_per_px        (image rows grow downward, plant +Y points up)

Occlusion is resolved per pixel by painter's order on triangle mean height (higher drawn last).
This is exact for non-intersecting leaflets and approximate where two triangles interpenetrate;
the approximation is recorded as ``occlusion_rule`` in the fixture manifest.

Fixture only: the scale is the declared synthetic camera, never a measured scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

OCCLUSION_RULE = "painter_order_by_triangle_mean_z"


@dataclass(frozen=True)
class OrthoCamera:
    """Declared synthetic top camera. ``mm_per_px`` is exact by construction."""

    image_size_px: int
    mm_per_px: float

    @property
    def center_px(self) -> float:
        return self.image_size_px / 2.0

    def project(self, xyz_mm: np.ndarray) -> np.ndarray:
        p = np.asarray(xyz_mm, float)
        u = self.center_px + p[..., 0] / self.mm_per_px
        v = self.center_px - p[..., 1] / self.mm_per_px
        return np.stack([u, v], axis=-1)

    def unproject_xy(self, uv_px: np.ndarray) -> np.ndarray:
        q = np.asarray(uv_px, float)
        return np.stack(
            [(q[..., 0] - self.center_px) * self.mm_per_px, (self.center_px - q[..., 1]) * self.mm_per_px],
            axis=-1,
        )


@dataclass
class TopRender:
    """Rasterised top view. ``visible`` holds ``ids[k-1]`` at value ``k`` (0 = background)."""

    visible: np.ndarray
    ids: list[str]
    amodal: dict[str, np.ndarray]

    def visible_mask(self, leaflet_id: str) -> np.ndarray:
        return self.visible == (self.ids.index(leaflet_id) + 1)


def render_top(
    meshes: Mapping[str, tuple[np.ndarray, np.ndarray]],
    camera: OrthoCamera,
) -> TopRender:
    """Rasterise ``{leaflet_id: (vertices_mm (N,3), faces (M,3))}`` seen from +Z."""
    ids = sorted(meshes)
    size = camera.image_size_px
    visible = np.zeros((size, size), np.uint16)
    amodal: dict[str, np.ndarray] = {}
    tris: list[tuple[float, int, np.ndarray]] = []
    shift = 4  # fixed-point bits for cv2.fillPoly sub-pixel vertices
    scale = float(1 << shift)
    for k, lid in enumerate(ids, start=1):
        verts, faces = meshes[lid]
        uv = camera.project(np.asarray(verts, float))
        z = np.asarray(verts, float)[:, 2]
        mask = np.zeros((size, size), np.uint8)
        polys = np.round(uv[np.asarray(faces)] * scale).astype(np.int32)
        cv2.fillPoly(mask, list(polys), 1, lineType=cv2.LINE_8, shift=shift)
        amodal[lid] = mask.astype(bool)
        zmean = z[np.asarray(faces)].mean(axis=1)
        for poly, zm in zip(polys, zmean):
            tris.append((float(zm), k, poly))
    tris.sort(key=lambda item: item[0])
    for _, k, poly in tris:
        cv2.fillPoly(visible, [poly], int(k), lineType=cv2.LINE_8, shift=shift)
    return TopRender(visible=visible, ids=ids, amodal=amodal)


def rotate_z(points_mm: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate about +Z (counter-clockwise seen from above), as the MTG azimuth sense (A2 12.1)."""
    a = np.radians(yaw_deg)
    r = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(points_mm, float) @ r.T
