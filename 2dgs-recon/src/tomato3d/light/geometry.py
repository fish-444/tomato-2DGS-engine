"""Plant surfaces for the light calculation.

Two input tables, both in the **board** frame (:mod:`tomato3d.io.schemas`):

``leaf_surfaces.csv``
    Triangles. Whatever produces a mesh writes this: multi-view reconstruction, a
    reduced-view reconstruction, a hand-corrected mesh.
``leaf_patches.csv``
    One flat rectangle per leaf, from its centroid, area and upper-surface normal angles.
    This is the simplified geometry a person can measure with a ruler and a protractor
    before automatic reconstruction exists.

Rows with the same ``geometry_variant_id`` form one variant, whichever table they come
from. A variant is one complete description of the plants' surfaces, so "41 views" and
"top + 3 sides" are two variants of the same plants.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomato3d.io.schemas import (
    GEOMETRY_SOURCES,
    LEAF_PATCHES_CSV_COLUMNS,
    LEAF_SURFACES_CSV_COLUMNS,
    LIGHT_RECEIVER_CLASSES,
    NO_PLANTS_VARIANT_ID,
    NORMAL_CONVENTIONS,
    SURFACE_CLASSES,
    UNAVAILABLE,
    PlantPlacement,
)
from tomato3d.io.tables import read_table, read_table_columns
from tomato3d.light.frames import board_to_grow_mm
from tomato3d.light.scene import LightInputError

_KNOWN_NORMAL = "adaxial_ccw"
_PATCH_TRIANGLE_IDS = ("patch_0", "patch_1")
_PATCH_SURFACE_CLASS = "leaf"


@dataclass(frozen=True)
class SurfaceSet:
    """Every surface triangle of one geometry variant, board frame.

    The per-triangle tuples and arrays are index-aligned.
    """

    geometry_variant_id: str
    plant_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    geometry_sources: tuple[str, ...]
    surface_classes: tuple[str, ...]
    organ_ids: tuple[str, ...]
    leaflet_ids: tuple[str, ...]
    triangle_ids: tuple[str, ...]
    vertices_board_mm: np.ndarray  # (T, 3, 3)
    #: ``True`` where the vertex order marks the upper (adaxial) side.
    normal_known: np.ndarray  # (T,) bool

    @property
    def n_triangles(self) -> int:
        return len(self.plant_ids)

    @property
    def is_receiver(self) -> np.ndarray:
        """``(T,)`` bool: leaf surfaces, which receive and absorb light."""
        return np.array([c in LIGHT_RECEIVER_CLASSES for c in self.surface_classes], dtype=bool)

    @property
    def plant_id_set(self) -> list[str]:
        return sorted(set(self.plant_ids))


def empty_surface_set(geometry_variant_id: str = NO_PLANTS_VARIANT_ID) -> SurfaceSet:
    """A variant with no surfaces: the empty growing space."""
    return SurfaceSet(
        geometry_variant_id=geometry_variant_id,
        plant_ids=(),
        session_ids=(),
        geometry_sources=(),
        surface_classes=(),
        organ_ids=(),
        leaflet_ids=(),
        triangle_ids=(),
        vertices_board_mm=np.zeros((0, 3, 3)),
        normal_known=np.zeros((0,), dtype=bool),
    )


# --------------------------------------------------------------------------------------
# Triangle measures
# --------------------------------------------------------------------------------------


def triangle_areas_mm2(vertices_mm: np.ndarray) -> np.ndarray:
    """``(T,)`` areas of ``(T, 3, 3)`` triangles."""
    cross = np.cross(vertices_mm[:, 1] - vertices_mm[:, 0], vertices_mm[:, 2] - vertices_mm[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def triangle_unit_normals(vertices_mm: np.ndarray) -> np.ndarray:
    """``(T, 3)`` unit normals ``(v1 - v0) x (v2 - v0)``; the upper side for ``adaxial_ccw``."""
    cross = np.cross(vertices_mm[:, 1] - vertices_mm[:, 0], vertices_mm[:, 2] - vertices_mm[:, 0])
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    return cross / norms


def subdivision_barycentrics(subdivisions: int) -> np.ndarray:
    """``(n**2, 2)`` coordinates ``(a, b)`` of equal sub-triangle centroids.

    A point is ``v0 + a (v1 - v0) + b (v2 - v0)``. Each sub-triangle has area
    ``1 / n**2`` of the parent, so the samples are equally weighted.
    """
    n = subdivisions
    upward = [((i + 1.0 / 3.0) / n, (j + 1.0 / 3.0) / n) for i in range(n) for j in range(n - i)]
    downward = [
        ((i + 2.0 / 3.0) / n, (j + 2.0 / 3.0) / n) for i in range(n) for j in range(n - 1 - i)
    ]
    return np.array(upward + downward, dtype=np.float64)


def sample_triangles(
    vertices_mm: np.ndarray, subdivisions: int
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-area sample points on every triangle.

    Returns:
        ``(points, triangle_index)``: ``(T * n**2, 3)`` points and the ``(T * n**2,)``
        index of the triangle each lies on.
    """
    coords = subdivision_barycentrics(subdivisions)
    edge1 = vertices_mm[:, 1] - vertices_mm[:, 0]
    edge2 = vertices_mm[:, 2] - vertices_mm[:, 0]
    points = (
        vertices_mm[:, None, 0]
        + coords[None, :, 0:1] * edge1[:, None, :]
        + coords[None, :, 1:2] * edge2[:, None, :]
    )
    n_triangles = vertices_mm.shape[0]
    triangle_index = np.repeat(np.arange(n_triangles), coords.shape[0])
    return points.reshape(-1, 3), triangle_index


# --------------------------------------------------------------------------------------
# Simplified leaves
# --------------------------------------------------------------------------------------


def patch_vertices_mm(
    centroid_mm: Sequence[float],
    area_mm2: float,
    normal_tilt_rad: float,
    normal_azimuth_rad: float,
    length_to_width_ratio: float,
) -> np.ndarray:
    """``(2, 3, 3)`` triangles of a flat rectangular leaf, counter-clockwise from above.

    The upper-surface normal is ``(sin t cos a, sin t sin a, cos t)``. The long side runs
    down the slope, ``(cos t cos a, cos t sin a, -sin t)``; the short side is horizontal.
    """
    tilt, azimuth = normal_tilt_rad, normal_azimuth_rad
    long_axis = np.array(
        [math.cos(tilt) * math.cos(azimuth), math.cos(tilt) * math.sin(azimuth), -math.sin(tilt)]
    )
    width_axis = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    length_mm = math.sqrt(area_mm2 * length_to_width_ratio)
    width_mm = area_mm2 / length_mm
    center = np.asarray(centroid_mm, dtype=np.float64)
    half_long = 0.5 * length_mm * long_axis
    half_wide = 0.5 * width_mm * width_axis
    corners = [
        center - half_long - half_wide,
        center + half_long - half_wide,
        center + half_long + half_wide,
        center - half_long + half_wide,
    ]
    return np.array([[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]])


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


@dataclass
class _VariantBuilder:
    plant_ids: list[str]
    session_ids: list[str]
    geometry_sources: list[str]
    surface_classes: list[str]
    organ_ids: list[str]
    leaflet_ids: list[str]
    triangle_ids: list[str]
    vertices: list[np.ndarray]
    normal_known: list[bool]

    @classmethod
    def new(cls) -> _VariantBuilder:
        return cls([], [], [], [], [], [], [], [], [])

    def add(
        self,
        *,
        plant_id: str,
        session_id: str,
        geometry_source: str,
        surface_class: str,
        organ_id: str,
        leaflet_id: str,
        triangle_id: str,
        vertices_mm: np.ndarray,
        normal_known: bool,
    ) -> None:
        self.plant_ids.append(plant_id)
        self.session_ids.append(session_id)
        self.geometry_sources.append(geometry_source)
        self.surface_classes.append(surface_class)
        self.organ_ids.append(organ_id)
        self.leaflet_ids.append(leaflet_id)
        self.triangle_ids.append(triangle_id)
        self.vertices.append(vertices_mm)
        self.normal_known.append(normal_known)


class _RowReader:
    """Cell accessors that name the file, row and column in every error."""

    def __init__(self, path: Path, index: int, row: dict[str, str]) -> None:
        self._where = f"{path.name} row {index + 2}"
        self._row = row

    def text(self, column: str) -> str:
        value = self._row[column].strip()
        if not value or value == UNAVAILABLE:
            raise LightInputError(f"{self._where}: {column} is required, got {value!r}")
        return value

    def optional_text(self, column: str) -> str:
        value = self._row[column].strip()
        return value if value else UNAVAILABLE

    def choice(self, column: str, choices: Iterable[str]) -> str:
        value = self._row[column].strip()
        allowed = tuple(choices)
        if value not in allowed:
            raise LightInputError(
                f"{self._where}: {column} must be one of {allowed}, got {value!r}"
            )
        return value

    def number(self, column: str) -> float:
        value = self._row[column].strip()
        if value == UNAVAILABLE:
            raise LightInputError(
                f"{self._where}: {column} is unavailable; a surface cannot be placed without it"
            )
        try:
            number = float(value)
        except ValueError as exc:
            raise LightInputError(f"{self._where}: {column} is not a number: {value!r}") from exc
        if not math.isfinite(number):
            raise LightInputError(f"{self._where}: {column} must be finite, got {value!r}")
        return number


def _read_surfaces(
    path: Path, builders: dict[str, _VariantBuilder], organ_origin: dict[tuple[str, ...], str]
) -> None:
    for index, row in enumerate(read_table(path, LEAF_SURFACES_CSV_COLUMNS)):
        cells = _RowReader(path, index, row)
        variant_id = cells.text("geometry_variant_id")
        plant_id = cells.text("plant_id")
        organ_id = cells.text("organ_id")
        vertices = np.array(
            [
                [cells.number(f"v{k}_{axis}_mm") for axis in ("x", "y", "z")]
                for k in range(3)
            ]
        )
        if triangle_areas_mm2(vertices[None])[0] <= 0.0:
            raise LightInputError(f"{path.name} row {index + 2}: degenerate triangle (zero area)")
        _claim_organ(organ_origin, (variant_id, plant_id, organ_id), "surfaces", path)
        builders.setdefault(variant_id, _VariantBuilder.new()).add(
            plant_id=plant_id,
            session_id=cells.text("session_id"),
            geometry_source=cells.choice("geometry_source", GEOMETRY_SOURCES),
            surface_class=cells.choice("organ_class", SURFACE_CLASSES),
            organ_id=organ_id,
            leaflet_id=cells.optional_text("leaflet_id"),
            triangle_id=cells.text("triangle_id"),
            vertices_mm=vertices,
            normal_known=cells.choice("normal_convention", NORMAL_CONVENTIONS) == _KNOWN_NORMAL,
        )


def _read_patches(
    path: Path, builders: dict[str, _VariantBuilder], organ_origin: dict[tuple[str, ...], str]
) -> None:
    for index, row in enumerate(read_table(path, LEAF_PATCHES_CSV_COLUMNS)):
        cells = _RowReader(path, index, row)
        variant_id = cells.text("geometry_variant_id")
        plant_id = cells.text("plant_id")
        organ_id = cells.text("organ_id")
        area_mm2 = cells.number("area_mm2")
        ratio = cells.number("length_to_width_ratio")
        tilt_deg = cells.number("normal_tilt_deg")
        if area_mm2 <= 0.0 or ratio <= 0.0:
            raise LightInputError(
                f"{path.name} row {index + 2}: area_mm2 and length_to_width_ratio must be > 0"
            )
        if not 0.0 <= tilt_deg <= 180.0:
            raise LightInputError(
                f"{path.name} row {index + 2}: normal_tilt_deg must lie in [0, 180]"
            )
        _claim_organ(organ_origin, (variant_id, plant_id, organ_id), "patches", path)
        triangles = patch_vertices_mm(
            [cells.number(f"centroid_{axis}_mm") for axis in ("x", "y", "z")],
            area_mm2,
            math.radians(tilt_deg),
            math.radians(cells.number("normal_azimuth_deg")),
            ratio,
        )
        builder = builders.setdefault(variant_id, _VariantBuilder.new())
        for triangle_id, vertices in zip(_PATCH_TRIANGLE_IDS, triangles):
            builder.add(
                plant_id=plant_id,
                session_id=cells.text("session_id"),
                geometry_source=cells.choice("geometry_source", GEOMETRY_SOURCES),
                surface_class=_PATCH_SURFACE_CLASS,
                organ_id=organ_id,
                leaflet_id=UNAVAILABLE,
                triangle_id=triangle_id,
                vertices_mm=vertices,
                normal_known=True,
            )


def _claim_organ(
    organ_origin: dict[tuple[str, ...], str], key: tuple[str, ...], kind: str, path: Path
) -> None:
    """An organ comes from one table only; two descriptions would double its area."""
    previous = organ_origin.setdefault(key, kind)
    if previous != kind:
        variant_id, plant_id, organ_id = key
        raise LightInputError(
            f"{path.name}: organ {organ_id!r} of plant {plant_id!r} in variant "
            f"{variant_id!r} is described both as {previous} and as {kind}"
        )


def _finish(variant_id: str, builder: _VariantBuilder) -> SurfaceSet:
    key_counts = Counter(
        zip(builder.plant_ids, builder.organ_ids, builder.leaflet_ids, builder.triangle_ids)
    )
    duplicates = sorted(key for key, count in key_counts.items() if count > 1)
    if duplicates:
        raise LightInputError(
            f"variant {variant_id!r} repeats (plant_id, organ_id, leaflet_id, triangle_id): "
            f"{duplicates[:5]}"
        )
    sessions_by_plant: dict[str, set[str]] = defaultdict(set)
    for plant_id, session_id in zip(builder.plant_ids, builder.session_ids):
        sessions_by_plant[plant_id].add(session_id)
    mixed = {plant: sorted(ids) for plant, ids in sessions_by_plant.items() if len(ids) > 1}
    if mixed:
        raise LightInputError(
            f"variant {variant_id!r} mixes capture sessions within one plant: {mixed}"
        )
    return SurfaceSet(
        geometry_variant_id=variant_id,
        plant_ids=tuple(builder.plant_ids),
        session_ids=tuple(builder.session_ids),
        geometry_sources=tuple(builder.geometry_sources),
        surface_classes=tuple(builder.surface_classes),
        organ_ids=tuple(builder.organ_ids),
        leaflet_ids=tuple(builder.leaflet_ids),
        triangle_ids=tuple(builder.triangle_ids),
        vertices_board_mm=np.array(builder.vertices, dtype=np.float64).reshape(-1, 3, 3),
        normal_known=np.array(builder.normal_known, dtype=bool),
    )


def load_geometry(paths: Sequence[Path]) -> dict[str, SurfaceSet]:
    """Read leaf surface and leaf patch tables into geometry variants.

    The table kind is recognised by its header.

    Raises:
        LightInputError: an unknown header, a reserved or repeated id, an unavailable
            vertex, a degenerate triangle, or one plant from two sessions in a variant.
    """
    builders: dict[str, _VariantBuilder] = {}
    organ_origin: dict[tuple[str, ...], str] = {}
    for path in paths:
        header = read_table_columns(path)
        if header == LEAF_SURFACES_CSV_COLUMNS:
            _read_surfaces(path, builders, organ_origin)
        elif header == LEAF_PATCHES_CSV_COLUMNS:
            _read_patches(path, builders, organ_origin)
        else:
            raise LightInputError(
                f"{path.name} is neither leaf_surfaces.csv nor leaf_patches.csv "
                f"(header {header})"
            )
    if NO_PLANTS_VARIANT_ID in builders:
        raise LightInputError(
            f"geometry_variant_id {NO_PLANTS_VARIANT_ID!r} is reserved for the empty space"
        )
    return {variant_id: _finish(variant_id, b) for variant_id, b in sorted(builders.items())}


def place_in_grow_frame(
    surfaces: SurfaceSet, placements: dict[str, PlantPlacement]
) -> np.ndarray:
    """``(T, 3, 3)`` grow-frame vertices of every triangle.

    Raises:
        LightInputError: a plant in the geometry has no placement in the scene.
    """
    vertices_grow = np.empty_like(surfaces.vertices_board_mm)
    plant_ids = np.array(surfaces.plant_ids, dtype=object)
    for plant_id in surfaces.plant_id_set:
        placement = placements.get(plant_id)
        if placement is None:
            raise LightInputError(
                f"variant {surfaces.geometry_variant_id!r} has geometry for plant "
                f"{plant_id!r}, which the scene does not place"
            )
        selected = plant_ids == plant_id
        origin = placement.board_origin_grow_mm.value
        yaw_deg = placement.board_yaw_deg.value
        if origin is None or yaw_deg is None:  # pragma: no cover - refused by the schema
            raise LightInputError(f"placement of {plant_id!r} is unavailable")
        vertices_grow[selected] = board_to_grow_mm(
            surfaces.vertices_board_mm[selected], np.asarray(origin), math.radians(yaw_deg)
        )
    return vertices_grow
