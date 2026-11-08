"""Micro-Tom priors for stage-1 reconstruction, from a parameter FILE (never hard-coded; E4).

``priors_mt_20261014.json`` beside this module names every value's source. The leaflet SHAPE pool is the user's 28
dissected Micro-Tom leaflets of 2026-10-14 (contours in mm) - "shape: use this as the reference" - ONE position-free
set (rows / columns carry no meaning, confirmed by the user), drawn uniformly. No Merlice or other non-MT value
enters here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomato3d.synth.leaflet_surface import normalize_contour, triangulate_contour

HERE = Path(__file__).resolve().parent
PRIORS_PATH = HERE / "priors_mt_20261014.json"
PROJECT_ROOT = HERE.parents[2]
SHAPE_POINTS = 64


@dataclass(frozen=True)
class Shape:
    shape_id: str
    length_mm: float
    width_mm: float
    area_mm2: float
    contour: np.ndarray       # normalised leaflet frame (x 0..1 base->tip, y in length units)
    fill: float               # area / (length * width)


@dataclass
class Priors:
    raw: dict
    shapes: list[Shape]
    path: Path

    def range(self, name: str) -> tuple[float, float]:
        r = self.raw["ratios"][name]["range"]
        return float(r[0]), float(r[1])

    def value(self, section: str, name: str):
        return self.raw[section][name]["value"]

    def source(self, section: str, name: str) -> str:
        e = self.raw[section][name]
        return f"{e['source_kind']}: {e['source_ref']}"

    def draw(self, name: str, rng: np.random.Generator) -> float:
        lo, hi = self.range(name)
        return float(rng.uniform(lo, hi)) if hi > lo else lo


def load(path: Path = PRIORS_PATH) -> Priors:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pool_path = PROJECT_ROOT / raw["shape_pool"]["path"]
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    shapes = []
    for leaf in data["leaflets"]:
        c = np.asarray(leaf["contour_mm"], float)
        norm = normalize_contour(c, SHAPE_POINTS)
        L, W = float(leaf["length_mm"]), float(leaf["width_mm"])
        a = float(leaf.get("polygon_area_mm2") or leaf["area_mm2"])
        shapes.append(Shape(leaf["id"], L, W, a, norm["points"], a / (L * W)))
    return Priors(raw, shapes, Path(path))


def mesh_for(shape: Shape, grid_n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    return triangulate_contour(shape.contour, grid_n)
