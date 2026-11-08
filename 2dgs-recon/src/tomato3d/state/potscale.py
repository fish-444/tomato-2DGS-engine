"""Per-pot rim scale (P1 T6, 2026-10-12): pot records, rim-plane homography, uncertainty, ruler cross-check.

Measured reference (user, 2026-10-12, caliper, ``source_kind = measured``): the pots are **square cells,
60 mm outer top edge across the flats**, 70 mm deep. The value enters through the human record
``data/records/pots/pot_spec.csv`` (human-owned, read-only for agents; template ``build/pot_records_template/``),
never through code or config.

Rules implemented (each has a test):

* **Field.** ``pot_rim_outer_edge_mm`` (square cell, one side, top rim, across the flats) and ``pot_depth_mm``
  (recorded only, never a scale source) beside the older round-pot ``pot_rim_inner_diameter_mm``. A record carries
  one rim convention or the other, never both; every mm value names its ``scale_convention``.
* **Per pot, never image-wide.** Handheld top views are perspective projections, so the scale is derived per pot
  from that pot's own rim in that image: points clicked on the four flat rim edges
  (``data/records/pots/rim_points.csv``, keyed by plant and the photo's sha256) -> one line per side ->
  their four intersections = the corners of the across-flats square (the chamfers are cut off, the lines are not)
  -> the homography from the image to the reference plane (``plant_local_mm``, origin at the pot centre, axes along
  the rim sides; the rotation about Z stays unknown - no orientation mark exists). ``mm_per_px`` is the local scale
  at the pot centre (``sqrt|det J|``); the full homography is kept in ``frame_transforms.csv``.
* **Plane.** Leaves sit above the reference plane, nearer the camera, so they are magnified in the photo and a
  rim-derived scale OVERESTIMATES their sizes (factor Z_ref / Z_leaf > 1; corrected 2026-10-12 - the T6 prompt said
  "underestimates", which is the wrong direction). That is the existing ``plant_local_mm`` convention (definitions
  section 2: a plane projection, never called "true size"); no correction factor is applied.
* **Uncertainty.** Every rim point carries the annotator's stated click uncertainty in px; per side the larger of
  that and the line-fit residual RMS is used, and propagated to ``mm_per_px`` (and to every mm value) by a seeded
  Monte Carlo. No stated uncertainty -> no mm value.
* **Cross-check.** Where the photo has a second source - ticks of the ruler lying on the rim, clicked with their
  readings - the ruler distance measured through the rim homography is compared with its reading (STAGE1 rule as
  given in T6): within 2 % -> accepted; more than 5 % -> stop (scale unavailable, reported); in between ->
  flagged and reported. Only the pot rim -> ``scale_sources = 1``, no cross-check. Sources are never averaged.

Revision 2026-10-13 (inner rim line and its plane; prompt ``CLAUDE_CODE_CLI_PROMPT_P1_RIM_INNER_PLANE_2026-10-13``):

* **New measurements** (user, caliper, ``measured``): inner top edge 59 mm across the flats, medium surface 10 mm
  below the rim, straight walls (no taper). Pot-spec columns ``pot_rim_inner_edge_mm``,
  ``medium_surface_depth_below_rim_mm``, ``wall_taper``, each with its own ``*_source_kind``. Old 10-column records
  still parse (the new values are then unavailable).
* **Which line, which plane.** Every rim point names its line in the new ``edge`` column: ``outer_top`` (outer top
  edge, rim plane, outer value), ``inner_top`` (inner top edge, rim plane, inner value) or ``medium_junction``
  (where the medium meets the inner wall: inner value - straight walls only - on the plane ``depth`` below the rim).
  Records without the column were collected under the "outer flat edges" instruction and are read as ``outer_top``.
  A rim side whose inner wall faces away from the camera (the FRONT side of a side photo) shows NO medium junction:
  the sharp rim/soil line there is the ``inner_top`` edge hiding the medium, i.e. still on the rim plane.
* **The plane travels with the value.** A solved scale names ``reference`` (the lines), ``reference_plane``
  (``pot_rim`` | ``medium_surface``) and its offset from the rim (0 or ``-depth``). One homography needs all four
  lines on one plane; outer and inner top edges may be mixed on the rim plane (per-side half widths).
* **Occluded side.** ``side_occluded`` rows (side number only) state that a side is hidden (e.g. the back rim behind
  leaves); it is reported as occluded, not as a detection failure. Three lines alone do not fix a homography.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.state import schema as s

POT_SPEC_PATH = Path("data/records/pots/pot_spec.csv")
RIM_POINTS_PATH = Path("data/records/pots/rim_points.csv")
#: Revision-1 pot-spec header (2026-10-12), still accepted (inner edge / medium depth / taper then unavailable).
POT_SPEC_COLUMNS_V1 = ("pot_type", "shape", "plant_ids", "pot_rim_outer_edge_mm", "pot_rim_inner_diameter_mm",
                       "pot_depth_mm", "source_kind", "method", "measured_date", "note")
#: Current header (2026-10-13): ``source_kind`` is the outer edge's; each new value carries its own.
POT_SPEC_COLUMNS = POT_SPEC_COLUMNS_V1[:7] + (
    "pot_rim_inner_edge_mm", "pot_rim_inner_edge_source_kind",
    "medium_surface_depth_below_rim_mm", "medium_surface_depth_source_kind",
    "wall_taper", "wall_taper_source_kind") + POT_SPEC_COLUMNS_V1[7:]
#: Revision-1 rim-points header, still accepted: every rim_edge row then means the OUTER top edge (the instruction
#: under which those records were collected).
RIM_POINTS_COLUMNS_V1 = ("plant_id", "image_sha256", "kind", "side", "x_px", "y_px", "click_uncertainty_px",
                         "ruler_reading_mm", "annotator", "note")
RIM_POINTS_COLUMNS = RIM_POINTS_COLUMNS_V1[:4] + ("edge",) + RIM_POINTS_COLUMNS_V1[4:]
SHAPES = ("square", "round")
POINT_KINDS = ("rim_edge", "ruler_tick", "side_occluded")
#: the lines a rim point may lie on (module docstring)
EDGES = ("outer_top", "inner_top", "medium_junction")
WALL_TAPERS = ("none", UNAVAILABLE)
#: physical planes a scale may refer to, and the name of the overestimate each implies for anything above it
REFERENCE_PLANES = ("pot_rim", "medium_surface")
PLANE_BIAS = {"pot_rim": "overestimates_above_rim", "medium_surface": "overestimates_above_medium_surface"}
SQUARE_CONVENTION = "pot_rim_outer_edge_square"
#: rim plane, outer and/or inner top edges with their own measured half widths
RIM_EDGES_CONVENTION = "pot_rim_edges_square"
#: medium surface: the medium / inner-wall junction, inner value, ``depth`` below the rim
MEDIUM_CONVENTION = "pot_medium_junction_square"
ROUND_CONVENTION = "pot_rim_inner_diameter_round"
#: Rim-plane caveat carried on every rim-derived value (definitions section 2; no correction factor).
PLANE_NOTE = ("rim-plane projection (plant_local_mm): leaves above the rim are nearer the camera and magnified, so "
              "this scale overestimates their sizes; not a true size; no correction applied")
MEDIUM_PLANE_NOTE = ("medium-surface projection: the reference lies on the medium surface below the rim; everything "
                     "above it (rim, leaves) is nearer the camera and magnified, so this scale overestimates their "
                     "sizes - by more than a rim-plane scale; not a true size; no correction applied")


def plane_note(plane: str) -> str:
    return MEDIUM_PLANE_NOTE if plane == "medium_surface" else PLANE_NOTE


def plane_bias_factor(camera_distance_mm: float, height_above_rim_mm: float, plane_offset_mm: float = 0.0) -> float:
    """Length overestimate of a feature ``height_above_rim_mm`` above the rim by a scale referred to the plane
    ``plane_offset_mm`` from the rim (0 rim, -depth medium surface), camera ``camera_distance_mm`` from the rim
    centre along the view axis: (d - offset) / (d - h). Areas: its square. Illustration only - never applied (leaf
    heights are not measured)."""
    d, h = float(camera_distance_mm), float(height_above_rim_mm)
    if not d > h:
        raise ValueError("the feature must lie in front of the camera (height < camera distance)")
    return (d - float(plane_offset_mm)) / (d - h)


class PotRecordError(ValueError):
    """A pot record breaks its own rules; reported, never repaired (CONVENTIONS section 4)."""


@dataclass(frozen=True)
class PotSpec:
    pot_type: str
    shape: str
    plant_ids: tuple[str, ...]  # exact ids or fnmatch patterns ('dwt_1*')
    outer_edge_mm: float | None
    inner_diameter_mm: float | None
    depth_mm: float | None
    source_kind: str
    ref: str
    inner_edge_mm: float | None = None
    inner_edge_source: str = UNAVAILABLE
    medium_depth_mm: float | None = None
    medium_depth_source: str = UNAVAILABLE
    wall_taper: str = UNAVAILABLE
    wall_taper_source: str = UNAVAILABLE

    @property
    def convention(self) -> str:
        return SQUARE_CONVENTION if self.shape == "square" else ROUND_CONVENTION

    def applies_to(self, plant_id: str) -> bool:
        return any(fnmatch(plant_id, p) for p in self.plant_ids)


@dataclass(frozen=True)
class RimPoint:
    kind: str
    side: int | None
    xy: tuple[float, float]
    sigma_px: float | None
    ruler_mm: float | None
    edge: str = "outer_top"
    edge_recorded: bool = True  # False: legacy record without the edge column (read as outer_top)


@dataclass(frozen=True)
class EdgeLine:
    """Where one rim line lies: its half width from the pot centre (mm) and its plane."""

    edge: str
    half_mm: float
    plane: str
    offset_mm: float  # plane height relative to the rim plane (0 or -depth)


@dataclass(frozen=True)
class Reference:
    """The lines of one photo's four (or three) sides, all on one plane."""

    halves: dict[int, float]
    edges: dict[int, str]
    plane: str
    offset_mm: float
    convention: str
    occluded: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        names = {"outer_top": f"outer_top_edge", "inner_top": "inner_top_edge", "medium_junction": "medium_junction"}
        kinds = sorted(set(self.edges.values()))
        if len(kinds) == 1:
            text = f"{names[kinds[0]]}_{2 * next(iter(self.halves.values())):g}mm"
        else:
            text = "+".join(f"s{k}:{names[e]}_{2 * self.halves[k]:g}mm" for k, e in sorted(self.edges.items()))
        return text + (f";occluded:s{','.join(map(str, self.occluded))}" if self.occluded else "")


@dataclass
class PotScale:
    """Result for one pot in one image. ``status`` solved | unavailable; ``reason`` says why when unavailable."""

    status: str
    reason: str = ""
    convention: str = UNAVAILABLE
    mm_per_px: float | None = None
    mm_per_px_sd: float | None = None
    anisotropy: float | None = None
    center_px: tuple[float, float] | None = None
    homography: list[list[float]] | None = None  # image px (x right, y down) -> reference-plane mm
    side_sigma_px: dict[int, float] = field(default_factory=dict)
    side_residual_px: dict[int, float | None] = field(default_factory=dict)
    scale_sources: int = 0
    cross_check: str = UNAVAILABLE
    ruler_disagreement: float | None = None
    source_ref: str = UNAVAILABLE
    reference: str = UNAVAILABLE
    reference_plane: str = UNAVAILABLE
    reference_plane_offset_mm: float | None = None
    occluded_sides: tuple[int, ...] = ()


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


def _num(text: str, what: str, where: str, positive: bool = True) -> float | None:
    text = (text or "").strip()
    if text in ("", UNAVAILABLE):
        return None
    try:
        value = float(text)
    except ValueError:
        raise PotRecordError(f"{where}: {what} {text!r} is not a number") from None
    if not math.isfinite(value) or (positive and value <= 0):
        raise PotRecordError(f"{where}: {what} {text!r} must be a positive number")
    return value


def _read(path: Path, columns: Sequence[str], *legacy: Sequence[str]) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        if header not in (tuple(columns), *(tuple(c) for c in legacy)):
            raise PotRecordError(f"{Path(path).name}: header {reader.fieldnames} != {tuple(columns)}")
        return list(reader), header


def _source(text: str, what: str, where: str, value_given: bool) -> str:
    text = (text or "").strip() or UNAVAILABLE
    if value_given and (text not in s.VALUE_STATUSES or text in (UNAVAILABLE, "synthetic_truth")):
        raise PotRecordError(f"{where}: {what} {text!r} (measured | user_stated)")
    if not value_given and text != UNAVAILABLE:
        raise PotRecordError(f"{where}: {what} {text!r} without its value")
    return text


def read_pot_spec(path: Path) -> list[PotSpec]:
    """Pot types. Square: outer edge required, round diameter blank; round: the reverse; never both."""
    out = []
    rows, header = _read(path, POT_SPEC_COLUMNS, POT_SPEC_COLUMNS_V1)
    for i, r in enumerate(rows, start=2):
        where = f"{Path(path).name} line {i}"
        if r["shape"] not in SHAPES:
            raise PotRecordError(f"{where}: shape {r['shape']!r} not in {SHAPES}")
        outer = _num(r["pot_rim_outer_edge_mm"], "pot_rim_outer_edge_mm", where)
        inner = _num(r["pot_rim_inner_diameter_mm"], "pot_rim_inner_diameter_mm", where)
        if outer is not None and inner is not None:
            raise PotRecordError(f"{where}: a pot carries one rim convention, never both "
                                 "(pot_rim_outer_edge_mm for square cells, pot_rim_inner_diameter_mm for round pots)")
        if r["shape"] == "square" and outer is None:
            raise PotRecordError(f"{where}: a square cell needs pot_rim_outer_edge_mm (one side, across the flats)")
        if r["shape"] == "round" and inner is None:
            raise PotRecordError(f"{where}: a round pot needs pot_rim_inner_diameter_mm")
        if r["source_kind"] not in s.VALUE_STATUSES or r["source_kind"] in ("unavailable", "synthetic_truth"):
            raise PotRecordError(f"{where}: source_kind {r['source_kind']!r}")
        ids = tuple(x.strip() for x in r["plant_ids"].split(";") if x.strip())
        if not ids:
            raise PotRecordError(f"{where}: plant_ids empty")
        extra = {}
        if header == POT_SPEC_COLUMNS:
            inner_edge = _num(r["pot_rim_inner_edge_mm"], "pot_rim_inner_edge_mm", where)
            depth = _num(r["medium_surface_depth_below_rim_mm"], "medium_surface_depth_below_rim_mm", where)
            taper = (r["wall_taper"] or "").strip() or UNAVAILABLE
            if taper not in WALL_TAPERS:
                raise PotRecordError(f"{where}: wall_taper {taper!r} not in {WALL_TAPERS} (a tapered wall needs its "
                                     "own geometry, not implemented)")
            if inner_edge is not None and (r["shape"] != "square" or outer is None or inner_edge >= outer):
                raise PotRecordError(f"{where}: pot_rim_inner_edge_mm is a square cell's inner top edge, smaller than "
                                     "the outer edge")
            extra = dict(
                inner_edge_mm=inner_edge,
                inner_edge_source=_source(r["pot_rim_inner_edge_source_kind"], "pot_rim_inner_edge_source_kind",
                                          where, inner_edge is not None),
                medium_depth_mm=depth,
                medium_depth_source=_source(r["medium_surface_depth_source_kind"],
                                            "medium_surface_depth_source_kind", where, depth is not None),
                wall_taper=taper,
                wall_taper_source=_source(r["wall_taper_source_kind"], "wall_taper_source_kind", where,
                                          taper != UNAVAILABLE))
        out.append(PotSpec(r["pot_type"], r["shape"], ids, outer, inner,
                           _num(r["pot_depth_mm"], "pot_depth_mm", where), r["source_kind"],
                           f"{Path(path).name}#line{i}", **extra))
    return out


def spec_for(specs: Sequence[PotSpec], plant_id: str) -> PotSpec | None:
    hits = [p for p in specs if p.applies_to(plant_id)]
    if len(hits) > 1:
        raise PotRecordError(f"{plant_id}: matched by more than one pot type {[p.pot_type for p in hits]}")
    return hits[0] if hits else None


def read_rim_points(path: Path) -> dict[tuple[str, str], list[RimPoint]]:
    """Rim / ruler points per (plant_id, image_sha256). Pixel grid = the decoded full-resolution photo with its
    EXIF orientation applied (x right, y down)."""
    out: dict[tuple[str, str], list[RimPoint]] = {}
    rows, header = _read(path, RIM_POINTS_COLUMNS, RIM_POINTS_COLUMNS_V1)
    legacy = header == RIM_POINTS_COLUMNS_V1
    for i, r in enumerate(rows, start=2):
        where = f"{Path(path).name} line {i}"
        s.check_identifier(r["plant_id"], "plant_id")
        if r["kind"] not in POINT_KINDS:
            raise PotRecordError(f"{where}: kind {r['kind']!r} not in {POINT_KINDS}")
        key = (r["plant_id"], r["image_sha256"].strip())
        if r["kind"] == "side_occluded":
            if r["side"].strip() not in ("1", "2", "3", "4"):
                raise PotRecordError(f"{where}: side_occluded needs its side 1-4")
            out.setdefault(key, []).append(
                RimPoint("side_occluded", int(r["side"]), (math.nan, math.nan), None, None, UNAVAILABLE, not legacy))
            continue
        edge = "outer_top" if legacy else (r["edge"] or "").strip()
        if r["kind"] == "rim_edge" and edge not in EDGES:
            raise PotRecordError(f"{where}: edge {edge!r} not in {EDGES} (which rim line was clicked)")
        x = _num(r["x_px"], "x_px", where, positive=False)
        y = _num(r["y_px"], "y_px", where, positive=False)
        if x is None or y is None:
            raise PotRecordError(f"{where}: x_px/y_px required")
        sigma = _num(r["click_uncertainty_px"], "click_uncertainty_px", where)
        side = None
        ruler = None
        if r["kind"] == "rim_edge":
            if r["side"].strip() not in ("1", "2", "3", "4"):
                raise PotRecordError(f"{where}: rim_edge side must be 1-4, consecutive around the rim")
            side = int(r["side"])
        else:
            ruler = _num(r["ruler_reading_mm"], "ruler_reading_mm", where, positive=False)
            if ruler is None:
                raise PotRecordError(f"{where}: ruler_tick needs ruler_reading_mm")
        out.setdefault(key, []).append(
            RimPoint(r["kind"], side, (x, y), sigma, ruler, edge if r["kind"] == "rim_edge" else UNAVAILABLE,
                     not legacy))
    for key, pts in out.items():
        occluded = {p.side for p in pts if p.kind == "side_occluded"}
        clicked = {p.side for p in pts if p.kind == "rim_edge"}
        if occluded & clicked:
            raise PotRecordError(f"{key[0]} photo {key[1][:12]}: side(s) {sorted(occluded & clicked)} both occluded "
                                 "and clicked")
    return out


def rim_point_files(folder: Path) -> list[Path]:
    """``rim_points.csv`` (top photos) and any ``rim_points_*.csv`` beside it (e.g. ``rim_points_side.csv``)."""
    folder = Path(folder)
    return sorted(p for p in folder.glob("rim_points*.csv") if p.is_file()) if folder.exists() else []


def read_rim_points_all(paths: Sequence[Path]) -> dict[tuple[str, str], list[RimPoint]]:
    """Merge several rim-point records; one photo (plant, image_sha256) may appear in ONE file only."""
    out: dict[tuple[str, str], list[RimPoint]] = {}
    where: dict[tuple[str, str], str] = {}
    for path in paths:
        for key, pts in read_rim_points(path).items():
            if key in out:
                raise PotRecordError(f"{key[0]} photo {key[1][:12]} appears in both {where[key]} and {Path(path).name}")
            out[key], where[key] = pts, Path(path).name
    return out


def write_templates(folder: Path) -> list[Path]:
    """Headers + clearly marked example rows + README, for the human to copy to data/records/pots/."""
    folder.mkdir(parents=True, exist_ok=True)
    spec = folder / "pot_spec.csv"
    with spec.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(POT_SPEC_COLUMNS)
        w.writerow(["square_cell", "square", "dwt_1*", "60", "", "70", "measured",
                    "59", "measured", "10", "measured", "none", "measured", "caliper", "2026-10-13",
                    "user caliper 2026-10-12/13: outer top edge 60 mm and inner top edge 59 mm across the flats; "
                    "medium surface 10 mm below the rim; straight walls"])
    pts = folder / "rim_points.csv"
    with pts.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(RIM_POINTS_COLUMNS)
        w.writerow(["dwt_101", "<sha256 of the raw top photo, as in captures.csv>", "rim_edge", "1", "outer_top",
                    "1200.5", "1650.0", "3", "", "initials", "EXAMPLE ROW - delete; a point on rim side 1 (flat part)"])
        w.writerow(["dwt_101", "<same sha256>", "ruler_tick", "", "", "1700.0", "1900.0", "3", "50", "initials",
                    "EXAMPLE ROW - delete; the 5 cm tick of the ruler lying on the rim"])
        w.writerow(["dwt_101", "<sha256 of a side photo>", "side_occluded", "4", "", "", "", "", "", "initials",
                    "EXAMPLE ROW - delete; side 4 (back) hidden by leaves"])
    readme = folder / "README.md"
    readme.write_text(
        "# Pot records (human-written; P1 per-pot rim scale)\n\n"
        "Copy the files to `data/records/pots/` and replace the example rows. Older files without the new columns "
        "still parse (their rim points then mean the OUTER top edge; inner edge and medium depth stay unavailable).\n\n"
        "## pot_spec.csv - one row per pot type\n"
        "| column | meaning |\n|---|---|\n"
        "| pot_type | a name |\n| shape | `square` (cell) or `round` |\n"
        "| plant_ids | `;`-separated plant ids or patterns (`dwt_1*`); a plant may match one row only |\n"
        "| pot_rim_outer_edge_mm | square cells: ONE side of the top rim, OUTER edge, across the flats (not the "
        "diagonal) |\n"
        "| pot_rim_inner_diameter_mm | round pots only; leave blank for square cells (never both) |\n"
        "| pot_depth_mm | recorded only; not a scale source |\n"
        "| source_kind | of the outer edge: `measured` (caliper) or `user_stated` (label/nominal) |\n"
        "| pot_rim_inner_edge_mm (+ _source_kind) | square cells: ONE side of the top rim, INNER edge, across the "
        "flats |\n"
        "| medium_surface_depth_below_rim_mm (+ medium_surface_depth_source_kind) | how far the medium surface lies "
        "below the rim |\n"
        "| wall_taper (+ _source_kind) | `none` (straight walls: the inner size is the same at the medium surface) "
        "or blank |\n"
        "| method, measured_date, note | free text / YYYY-MM-DD |\n\n"
        "## rim_points.csv - points clicked on each photo\n"
        "Pixel coordinates on the full-resolution photo with its EXIF rotation applied (x to the right, y down).\n"
        "- `rim_edge`: at least 2 (better 3+) points on each FLAT rim side; number the sides 1-4 in order around the "
        "rim (any start). Do not click on the cut corners. `edge` says WHICH line: `outer_top` (outer top edge), "
        "`inner_top` (inner top edge) or `medium_junction` (where the medium meets the inner wall, 10 mm lower). "
        "One side = one line. On a side photo the sharp rim/soil line of the FRONT side is `inner_top` (the front "
        "wall hides the medium junction there).\n"
        "- `side_occluded`: a side you cannot see (e.g. the back rim behind leaves) - side number only.\n"
        "- `ruler_tick` (optional cross-check): 2+ ticks of the ruler lying on the rim with their reading in mm.\n"
        "- `click_uncertainty_px`: how far off your click may be, in px (required; no value -> no mm value).\n"
        "- `image_sha256`: the photo's sha256 as listed in `captures.csv` of a P1 package.\n\n"
        "## rim_points_side*.csv - the same columns, for the SIDE photos\n"
        "`ruler_tick` here means ticks of the UPRIGHT ruler: they are recorded, but one photo cannot test a vertical "
        "scale with them until an anchor is decided (depth-length ambiguity). A photo may appear in one "
        "rim_points*.csv file only.\n\n"
        "Every mm value names the plane of its reference (`pot_rim` or `medium_surface`). Leaves sit above that "
        "plane, nearer the camera, so they look magnified: every such size OVERestimates them (no correction is "
        "applied); a medium-surface reference overestimates more than a rim reference.\n", encoding="utf-8")
    return [spec, pts, readme]


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


def edge_line(spec: PotSpec, edge: str) -> EdgeLine:
    """Half width and plane of one rim line from the pot record; PotRecordError says which value is missing."""
    if edge == "outer_top":
        return EdgeLine(edge, spec.outer_edge_mm / 2.0, "pot_rim", 0.0)
    if spec.inner_edge_mm is None:
        raise PotRecordError(f"edge {edge}: pot_rim_inner_edge_mm is not recorded for pot type {spec.pot_type}")
    if edge == "inner_top":
        return EdgeLine(edge, spec.inner_edge_mm / 2.0, "pot_rim", 0.0)
    if edge == "medium_junction":
        if spec.medium_depth_mm is None:
            raise PotRecordError(f"edge {edge}: medium_surface_depth_below_rim_mm is not recorded for pot type "
                                 f"{spec.pot_type}")
        if spec.wall_taper != "none":
            raise PotRecordError(f"edge {edge}: the inner size at the medium surface equals the inner top edge only "
                                 f"for straight walls; wall_taper is {spec.wall_taper!r}")
        return EdgeLine(edge, spec.inner_edge_mm / 2.0, "medium_surface", -spec.medium_depth_mm)
    raise PotRecordError(f"edge {edge!r} not in {EDGES}")


def reference_of(points: Sequence[RimPoint], spec: PotSpec, min_sides: int = 4) -> Reference:
    """The reference lines of one photo: one line per side, all on one plane, sides 1-4 (or three + one occluded
    when ``min_sides`` is 3)."""
    rim = [p for p in points if p.kind == "rim_edge"]
    occluded = tuple(sorted({p.side for p in points if p.kind == "side_occluded"}))
    edges: dict[int, str] = {}
    for p in rim:
        if edges.setdefault(p.side, p.edge) != p.edge:
            raise PotRecordError(f"rim side {p.side}: points on two different lines ({edges[p.side]}, {p.edge}); "
                                 "one side = one line")
    missing = [k for k in (1, 2, 3, 4) if k not in edges and k not in occluded]
    if missing:
        raise PotRecordError(f"rim side {', '.join(map(str, missing))}: no points (click them, or record "
                             "side_occluded)")
    if len(edges) < min_sides:
        raise PotRecordError(f"{len(edges)} rim sides clicked, side(s) {list(occluded)} occluded: a homography needs "
                             f"all four lines")
    lines = {k: edge_line(spec, e) for k, e in edges.items()}
    planes = {ln.plane for ln in lines.values()}
    if len(planes) != 1:
        raise PotRecordError(f"rim lines on different planes ({sorted(planes)}): one homography needs one plane")
    plane = planes.pop()
    kinds = set(edges.values())
    convention = (MEDIUM_CONVENTION if plane == "medium_surface"
                  else SQUARE_CONVENTION if kinds == {"outer_top"} else RIM_EDGES_CONVENTION)
    return Reference({k: ln.half_mm for k, ln in lines.items()}, edges, plane,
                     next(iter(lines.values())).offset_mm, convention, occluded)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Total-least-squares line as homogeneous (a, b, c) with a^2+b^2=1; residual RMS (None for 2 points)."""
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean)
    normal = vt[-1]
    line = np.array([normal[0], normal[1], -normal @ mean])
    resid = points @ normal - normal @ mean
    rms = float(np.sqrt(np.mean(resid ** 2) * len(points) / (len(points) - 2))) if len(points) > 2 else None
    return line, rms


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-12:
        raise PotRecordError("two consecutive rim sides are parallel; sides must be numbered 1-4 around the rim")
    return p[:2] / p[2]


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(a, float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.c_[np.atleast_2d(xy), np.ones(len(np.atleast_2d(xy)))] @ h.T
    return p[:, :2] / p[:, 2:3]


def _jacobian(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x, y = xy
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    u = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
    v = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
    return np.array([[(h[0, 0] - u * h[2, 0]) / w, (h[0, 1] - u * h[2, 1]) / w],
                     [(h[1, 0] - v * h[2, 0]) / w, (h[1, 1] - v * h[2, 1]) / w]])


def square_rim_homography(sides: Mapping[int, np.ndarray], edge_mm: float,
                          halves: Mapping[int, float] | None = None
                          ) -> tuple[np.ndarray, np.ndarray, dict[int, float | None]]:
    """Lines through the points of sides 1-4 -> corners -> homography image px -> reference plane mm.

    Plane frame: origin at the pot centre, X/Y along the rim sides, right-handed seen from above (image y points
    down, so the image->plane map must reverse orientation). Side 1 is the line X = -h1, side 2 Y = -h2, side 3
    X = +h3, side 4 Y = +h4, with ``halves`` the per-side half widths (default ``edge_mm / 2`` on every side).
    Returns (H, centre_px, residual RMS per side)."""
    lines, resid = {}, {}
    for k in (1, 2, 3, 4):
        pts = sides.get(k)
        if pts is None or len(pts) < 2:
            raise PotRecordError(f"rim side {k}: at least 2 points needed")
        lines[k], resid[k] = _fit_line(np.asarray(pts, float))
    corners = np.array([_intersect(lines[1], lines[2]), _intersect(lines[2], lines[3]),
                        _intersect(lines[3], lines[4]), _intersect(lines[4], lines[1])])
    # convex, consecutive quadrilateral
    def _cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    cross = [_cross2(corners[(i + 1) % 4] - corners[i], corners[(i + 2) % 4] - corners[(i + 1) % 4])
             for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        raise PotRecordError("rim sides do not form a convex quadrilateral; number them 1-4 in order around the rim")
    hv = {k: (halves[k] if halves else edge_mm / 2.0) for k in (1, 2, 3, 4)}
    target = np.array([[-hv[1], -hv[2]], [hv[3], -hv[2]], [hv[3], hv[4]], [-hv[1], hv[4]]])
    h = _homography(corners, target)
    centre = apply_h(np.linalg.inv(h), np.zeros((1, 2)))[0]
    if np.linalg.det(_jacobian(h, centre)) > 0:  # image (y down) -> plane (Y up) must flip orientation
        flipped = np.array([[-hv[1], hv[2]], [hv[3], hv[2]], [hv[3], -hv[4]], [-hv[1], -hv[4]]])
        h = _homography(corners, flipped)
        centre = apply_h(np.linalg.inv(h), np.zeros((1, 2)))[0]
    return h, centre, resid


def _scale_at(h: np.ndarray, centre: np.ndarray) -> tuple[float, float]:
    sv = np.linalg.svd(_jacobian(h, centre), compute_uv=False)
    return float(math.sqrt(abs(np.linalg.det(_jacobian(h, centre))))), float(sv[0] / sv[1])


def derive(points: Sequence[RimPoint], spec: PotSpec | None, *, draws: int, seed: int, accept_rel: float,
           stop_rel: float, record_ref: str) -> PotScale:
    """Per-pot scale for one photo on the plane of its reference lines. Every missing input -> ``unavailable``
    with its reason."""
    if spec is None:
        return PotScale("unavailable", f"no pot type in {POT_SPEC_PATH.as_posix()} for this plant")
    if spec.shape != "square":
        return PotScale("unavailable", "round-pot rim derivation not implemented (no round pots in use)",
                        convention=spec.convention)
    rim = [p for p in points if p.kind == "rim_edge"]
    if not rim:
        return PotScale("unavailable", f"no rim_edge points in {RIM_POINTS_PATH.as_posix()} for this photo",
                        convention=spec.convention)
    if any(p.sigma_px is None for p in rim):
        return PotScale("unavailable", "a rim point has no click_uncertainty_px; no mm value without uncertainty",
                        convention=spec.convention)
    try:
        ref = reference_of(points, spec)
    except PotRecordError as exc:
        occ = tuple(sorted({p.side for p in points if p.kind == "side_occluded"}))
        return PotScale("unavailable", str(exc), convention=spec.convention, occluded_sides=occ)
    sides = {k: np.array([p.xy for p in rim if p.side == k]) for k in (1, 2, 3, 4)}
    try:
        h, centre, resid = square_rim_homography(sides, spec.outer_edge_mm, ref.halves)
    except PotRecordError as exc:
        return PotScale("unavailable", str(exc), convention=ref.convention)
    sigma = {k: max(max(p.sigma_px for p in rim if p.side == k), resid[k] or 0.0) for k in (1, 2, 3, 4)}
    scale, aniso = _scale_at(h, centre)
    rng = np.random.default_rng(seed)
    ticks = [p for p in points if p.kind == "ruler_tick"]
    pair = None
    if len(ticks) >= 2:
        a, b = min(ticks, key=lambda p: p.ruler_mm), max(ticks, key=lambda p: p.ruler_mm)
        if b.ruler_mm > a.ruler_mm and a.sigma_px is not None and b.sigma_px is not None:
            pair = (a, b)
    if pair and ref.plane != "pot_rim":
        pair = None  # the ruler lies on the rim, not on the reference plane: not comparable through this homography
    samples, ruler_samples = [], []
    for _ in range(draws):
        jittered = {k: v + rng.normal(0.0, sigma[k], v.shape) for k, v in sides.items()}
        try:
            hk, ck, _ = square_rim_homography(jittered, spec.outer_edge_mm, ref.halves)
        except PotRecordError:
            continue
        samples.append(_scale_at(hk, ck)[0])
        if pair:
            pa = np.array(pair[0].xy) + rng.normal(0.0, pair[0].sigma_px, 2)
            pb = np.array(pair[1].xy) + rng.normal(0.0, pair[1].sigma_px, 2)
            ruler_samples.append(float(np.linalg.norm(apply_h(hk, pa[None])[0] - apply_h(hk, pb[None])[0])))
    if len(samples) < draws:
        return PotScale("unavailable", f"rim geometry unstable under the stated click uncertainty "
                                       f"({draws - len(samples)} of {draws} draws degenerate)",
                        convention=ref.convention)
    legacy = any(not p.edge_recorded for p in rim)
    out = PotScale("solved", convention=ref.convention, mm_per_px=scale, mm_per_px_sd=float(np.std(samples, ddof=1)),
                   anisotropy=aniso, center_px=(float(centre[0]), float(centre[1])), homography=h.tolist(),
                   side_sigma_px=sigma, side_residual_px=resid, scale_sources=1, cross_check="none_single_source",
                   source_ref=f"{spec.ref};{record_ref}",
                   reference=ref.label + (";edge_from_legacy_record_instruction" if legacy else ""),
                   reference_plane=ref.plane, reference_plane_offset_mm=ref.offset_mm, occluded_sides=ref.occluded)
    if pair:
        measured = float(np.linalg.norm(apply_h(h, np.array([pair[0].xy]))[0] - apply_h(h, np.array([pair[1].xy]))[0]))
        reading = pair[1].ruler_mm - pair[0].ruler_mm
        out.ruler_disagreement = measured / reading - 1.0
        out.scale_sources = 2
        if abs(out.ruler_disagreement) <= accept_rel:
            out.cross_check = "accepted_within_2pct"
        elif abs(out.ruler_disagreement) > stop_rel:
            out.cross_check = "stopped_over_5pct"
            out.status = "unavailable"
            out.reason = (f"rim vs ruler disagree by {out.ruler_disagreement:+.2%} (> {stop_rel:.0%}): STOP, "
                          "report; sources are never averaged")
        else:
            out.cross_check = "flagged_2_to_5pct"
    return out


#: ``scale_plane`` (measurement plane) of each reference plane for a horizontal-plane scale
SCALE_PLANE_OF = {"pot_rim": "rim_plane", "medium_surface": "medium_surface_plane"}


def capture_fields(result: PotScale) -> dict[str, str]:
    """The captures.csv / frame_transforms.csv cells of one result (unavailable cells stay unavailable)."""
    solved = result.status == "solved"
    return dict(
        scale_status="solved" if solved else UNAVAILABLE,
        scale_source=(f"pot_rim:{result.source_ref}" if solved else UNAVAILABLE),
        mm_per_px=s.fmt(result.mm_per_px, 8) if solved else UNAVAILABLE,
        mm_per_px_sd=s.fmt(result.mm_per_px_sd, 8) if solved else UNAVAILABLE,
        scale_convention=result.convention if solved else UNAVAILABLE,
        scale_sources=str(result.scale_sources),
        scale_cross_check=result.cross_check,
        scale_plane=SCALE_PLANE_OF[result.reference_plane] if solved else UNAVAILABLE,
        scale_reference=result.reference if solved else UNAVAILABLE,
        scale_reference_plane=result.reference_plane if solved else UNAVAILABLE,
        scale_reference_plane_offset_mm=s.fmt(result.reference_plane_offset_mm, 3) if solved else UNAVAILABLE,
        rim_homography_px_to_mm=json.dumps([[round(v, 12) for v in row] for row in result.homography])
        if solved else UNAVAILABLE)
