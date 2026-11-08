"""Compound-leaf interior: L-Py grammar + PlantGL turtle, constrained by the P3 block.

Contract P3 -> L-Py
-------------------
* The whole compound axis (rachis + terminal leaflet midrib) equals the P3
  ``rachis_arc_length_mm`` within ``contract_length_tolerance_rel``.
* Every leaflet outline, laid flat in the compound plane (before bend/roll, the widest
  case), reaches at most ``(1 + contract_extent_tolerance_rel)`` times the P3 left/right
  extent from the rachis line.

A sample that breaks the contract is **rejected and redrawn** with the next attempt seed,
never clipped. Rejections are returned so the caller can log them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from tomato3d.synth.config import constant, make_rng, sample_range

LPY_PATH = Path(__file__).with_name("compound_leaf.lpy")

#: Interior parameters a caller may pin (P1 occlusion completion conditioned on an observed terminal
#: leaflet, phynotype2 section 6.4). Applied AFTER the normal draw, so the RNG stream - and every
#: other parameter - is unchanged; ``fixed=None`` is bit-identical to the unpinned generator.
FIXED_WHITELIST = ("terminal_rel_length", "contour_index.T", "width_scale.T")
BACKENDS = ("lpy", "numpy")


class CompoundRejected(RuntimeError):
    """No interior sample satisfied the P3 contract within the attempt budget."""

    def __init__(self, compound_id: str, rejections: list[dict[str, Any]]):
        super().__init__(f"{compound_id}: {len(rejections)} interior samples rejected")
        self.compound_id = compound_id
        self.rejections = rejections


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def initial_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heading = P3 direction; left = z x heading (P3 transverse convention); up = heading x left."""
    heading = _unit(np.asarray(direction, float))
    left = np.cross([0.0, 0.0, 1.0], heading)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0, 0.0])
    left = _unit(left)
    up = _unit(np.cross(heading, left))
    return heading, left, up


def sample_interior(cfg: dict[str, Any], compound: dict[str, Any], n_contours: int,
                    seed: int, attempt: int, fixed: dict[str, Any] | None = None) -> dict[str, Any]:
    """Draw every interior parameter for one compound leaf; then apply ``fixed`` (whitelist only)."""
    rng = make_rng(seed, "compound", compound["compound_id"], attempt)
    length = compound["rachis_arc_length_mm"]
    p = {name: sample_range(rng, cfg, name) for name in (
        "leaflet_pairs", "first_pair_fraction", "last_pair_fraction", "terminal_rel_length", "pair_taper",
        "left_right_asymmetry", "intercalary_count", "intercalary_rel_size", "petiolule_angle_deg",
        "petiolule_rel_length", "rachis_droop_deg")}
    bud = length < constant(cfg, "min_compound_rachis_mm")
    pairs_cap = max(1, int(length // constant(cfg, "mm_per_leaflet_pair_min")))
    p["leaflet_pairs"] = 0 if bud else min(int(p["leaflet_pairs"]), pairs_cap)
    p["intercalary_count"] = 0 if bud or p["leaflet_pairs"] < 2 else int(p["intercalary_count"])
    p["bud"] = bool(bud)
    p["asymmetry_sign"] = float(rng.choice([-1.0, 1.0]))
    tags: list[str] = (["T"] + [f"P{i + 1}{s}" for i in range(p["leaflet_pairs"]) for s in "LR"]
                       + [f"I{i + 1}" for i in range(p["intercalary_count"])])
    p["intercalary_gaps"] = sorted(int(g) for g in rng.choice(max(1, p["leaflet_pairs"] - 1),
                                                              size=p["intercalary_count"], replace=True)) \
        if p["intercalary_count"] else []
    p["leaflets"] = {tag: dict(
        bend_deg=sample_range(rng, cfg, "leaflet_bend_deg"),
        roll_deg=sample_range(rng, cfg, "leaflet_roll_deg"),
        twist_deg=sample_range(rng, cfg, "leaflet_twist_deg"),
        width_scale=sample_range(rng, cfg, "leaflet_width_scale"),
        angle_jitter_deg=float(rng.normal(0.0, 5.0)),
        contour_index=int(rng.integers(0, n_contours))) for tag in tags}
    if fixed:
        _apply_fixed(p, fixed, n_contours)
    return p


def _apply_fixed(p: dict[str, Any], fixed: dict[str, Any], n_contours: int) -> None:
    unknown = sorted(set(fixed) - set(FIXED_WHITELIST))
    if unknown:
        raise ValueError(f"fixed= keys outside the whitelist {FIXED_WHITELIST}: {unknown}")
    if "terminal_rel_length" in fixed:
        p["terminal_rel_length"] = float(fixed["terminal_rel_length"])
    if "contour_index.T" in fixed:
        index = int(fixed["contour_index.T"])
        if not 0 <= index < n_contours:
            raise ValueError(f"contour_index.T {index} outside the contour pool (0..{n_contours - 1})")
        p["leaflets"]["T"]["contour_index"] = index
    if "width_scale.T" in fixed:
        p["leaflets"]["T"]["width_scale"] = float(fixed["width_scale.T"])


def _events(p: dict[str, Any], length_mm: float, segments: int) -> tuple[list[tuple], float, float]:
    """Lay leaflets along the rachis; return (EVENTS, segment length, terminal length)."""
    if p["bud"]:
        return [], 0.0, length_mm
    terminal = p["terminal_rel_length"] * length_mm
    petiolule = p["petiolule_rel_length"] * length_mm
    rachis = length_mm - terminal  # petiolules are lateral, not part of the axis
    seg = rachis / segments
    n = p["leaflet_pairs"]
    fractions = ([p["first_pair_fraction"]] if n == 1 else
                 list(np.linspace(p["first_pair_fraction"], p["last_pair_fraction"], n)))
    events = []
    for i, f in enumerate(fractions):
        size = terminal * (p["pair_taper"] + (1 - p["pair_taper"]) * (i / max(1, n - 1)))
        for side in "LR":
            sign = 1.0 if side == "L" else -1.0
            size_side = size * (1 + sign * p["asymmetry_sign"] * p["left_right_asymmetry"] / 2)
            angle = p["petiolule_angle_deg"] + p["leaflets"][f"P{i + 1}{side}"]["angle_jitter_deg"]
            k = min(segments - 1, int(f * segments))
            events.append((k, f"P{i + 1}{side}", side, size_side, petiolule, angle))
    for j, gap in enumerate(p["intercalary_gaps"]):
        f = 0.5 * (fractions[gap] + fractions[min(gap + 1, n - 1)])
        side = "L" if j % 2 == 0 else "R"
        k = min(segments - 1, int(f * segments))
        events.append((k, f"I{j + 1}", side, p["intercalary_rel_size"] * terminal, 0.0,
                       p["petiolule_angle_deg"]))
    return events, seg, terminal


def derive(events: list[tuple], seg_mm: float, droop_deg: float, segments: int, terminal_mm: float):
    """Run the L-Py grammar; return the derived lstring."""
    from openalea import lpy  # lazy: tomato_synth env only

    variables = dict(SEGMENTS=int(segments), SEG_MM=float(seg_mm), DROOP_DEG=float(droop_deg),
                     EVENTS=list(events), TERMINAL_MM=float(terminal_mm))
    # L-Py's C++ file loader cannot open non-ASCII paths (this project lives under a Korean
    # folder name), so the grammar text is passed in directly.
    system = lpy.Lsystem()
    system.setCode(LPY_PATH.read_text(encoding="utf-8"), variables)
    return system.derive()


def derive_modules(events: list[tuple], seg_mm: float, droop_deg: float, segments: int,
                   terminal_mm: float) -> list[tuple]:
    """The same derivation as ``compound_leaf.lpy`` without L-Py (``backend="numpy"``).

    The grammar is A(k) -> I(SEG_MM, DROOP_DEG) [LP(event) for events with index k] then A(k+1),
    and T(TERMINAL_MM) after the last internode; ``derivation length: SEGMENTS``. Equivalence with
    the L-Py derivation is checked against stored L-Py/PlantGL interiors (tests/test_synth_numpy_backend.py).
    """
    out: list[tuple] = []
    for k in range(int(segments)):
        out.append(("I", float(seg_mm), float(droop_deg)))
        for index, tag, side, length_mm, petiolule_mm, angle_deg in events:
            if index == k:
                out.append(("LP", tag, side, length_mm, petiolule_mm, angle_deg))
    out.append(("T", float(terminal_mm)))
    return out


class _Turtle:
    """PlantGL turtle subset used by :func:`interpret`: f, down, left, right, push, pop (degrees).

    Frame: heading H, up U, left L = U x H. down(a) pitches H toward -U; left(a) turns H toward +L.
    """

    def __init__(self, position: np.ndarray, heading: np.ndarray, up: np.ndarray):
        self.p = np.asarray(position, float)
        self.h = _unit(np.asarray(heading, float))
        self.u = _unit(np.asarray(up, float))
        self.stack: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    @property
    def left_axis(self) -> np.ndarray:
        return np.cross(self.u, self.h)

    def f(self, length: float) -> None:
        self.p = self.p + self.h * float(length)

    def down(self, angle_deg: float) -> None:
        a = math.radians(angle_deg)
        h, u = self.h, self.u
        self.h = _unit(h * math.cos(a) - u * math.sin(a))
        self.u = _unit(u * math.cos(a) + h * math.sin(a))

    def left(self, angle_deg: float) -> None:
        a = math.radians(angle_deg)
        self.h = _unit(self.h * math.cos(a) + self.left_axis * math.sin(a))

    def right(self, angle_deg: float) -> None:
        self.left(-angle_deg)

    def push(self) -> None:
        self.stack.append((self.p.copy(), self.h.copy(), self.u.copy()))

    def pop(self) -> None:
        self.p, self.h, self.u = self.stack.pop()


def interpret_numpy(modules: list[tuple], compound: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """:func:`interpret` on :func:`derive_modules` output, with :class:`_Turtle` instead of PlantGL."""
    heading, _left, up = initial_frame(np.array(compound["direction_unit"]))
    turtle = _Turtle(np.array(compound["attachment_mm"], float), heading, up)
    rachis = [np.array(compound["attachment_mm"], float)]
    leaflets = []

    def add(tag: str, kind: str, side: str, length_mm: float, rachis_station_mm: float,
            petiolule_mm: float = 0.0) -> None:
        leaflets.append(dict(tag=tag, kind=kind, side=side, length_mm=float(length_mm), petiolule_mm=float(petiolule_mm),
                             base_mm=turtle.p.tolist(), heading=turtle.h.tolist(), up=turtle.u.tolist(),
                             left=turtle.left_axis.tolist(), rachis_station_mm=float(rachis_station_mm),
                             **p["leaflets"][tag]))

    station = 0.0
    for module in modules:
        name = module[0]
        if name == "I":
            turtle.down(float(module[2]))
            turtle.f(float(module[1]))
            station += float(module[1])
            rachis.append(turtle.p.copy())
        elif name == "LP":
            tag, side, length_mm, petiolule_mm, angle = module[1:]
            turtle.push()
            (turtle.left if side == "L" else turtle.right)(float(angle))
            turtle.f(float(petiolule_mm))
            add(tag, "intercalary" if tag.startswith("I") else "pair", side, length_mm, station, float(petiolule_mm))
            turtle.pop()
        elif name == "T":
            add("T", "terminal", "C", float(module[1]), station)
    return dict(rachis_points_mm=[r.tolist() for r in rachis], leaflets=leaflets)


def _v(x) -> np.ndarray:
    return np.array([x[0], x[1], x[2]], float)


def interpret(lstring, compound: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """Walk the lstring with a PlantGL turtle; return rachis points and leaflet frames."""
    import openalea.plantgl.all as pgl  # lazy

    heading, left, up = initial_frame(np.array(compound["direction_unit"]))
    turtle = pgl.PglTurtle()
    turtle.move(pgl.Vector3(*compound["attachment_mm"]))
    turtle.setHead(pgl.Vector3(*heading), pgl.Vector3(*up))
    rachis = [np.array(compound["attachment_mm"], float)]
    leaflets = []

    def frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (_v(turtle.getPosition()), _v(turtle.getHeading()), _v(turtle.getUp()), _v(turtle.getLeft()))

    def add(tag: str, kind: str, side: str, length_mm: float, rachis_station_mm: float,
            petiolule_mm: float = 0.0) -> None:
        pos, h, u, lft = frame()
        leaflets.append(dict(tag=tag, kind=kind, side=side, length_mm=float(length_mm), petiolule_mm=float(petiolule_mm),
                             base_mm=pos.tolist(), heading=h.tolist(), up=u.tolist(), left=lft.tolist(),
                             rachis_station_mm=float(rachis_station_mm), **p["leaflets"][tag]))

    station = 0.0
    for module in lstring:
        name = module.name
        if name == "I":
            turtle.down(float(module[1]))
            turtle.f(float(module[0]))
            station += float(module[0])
            rachis.append(_v(turtle.getPosition()))
        elif name == "LP":
            tag, side, length_mm, petiolule_mm, angle = module[0], module[1], module[2], module[3], module[4]
            turtle.push()
            # PlantGL: left() turns the heading towards +left; right() towards -left.
            (turtle.left if side == "L" else turtle.right)(float(angle))
            turtle.f(float(petiolule_mm))
            add(tag, "intercalary" if tag.startswith("I") else "pair", side, length_mm, station, float(petiolule_mm))
            turtle.pop()
        elif name == "T":
            add("T", "terminal", "C", float(module[0]), station)
    return dict(rachis_points_mm=[r.tolist() for r in rachis], leaflets=leaflets)


def flat_outline(leaflet: dict[str, Any], contour: np.ndarray) -> np.ndarray:
    """Leaflet outline in 3D laid flat in its (heading, left) plane - the widest case."""
    h, lft = np.array(leaflet["heading"]), np.array(leaflet["left"])
    c = np.asarray(contour, float)
    scale = leaflet["length_mm"]
    return (np.array(leaflet["base_mm"])[None] + scale * c[:, :1] * h[None]
            + scale * leaflet["width_scale"] * c[:, 1:2] * lft[None])


def check_contract(compound: dict[str, Any], interior: dict[str, Any], contours: list[np.ndarray],
                   cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Return None when the interior satisfies the P3 contract, else a rejection record."""
    rachis = np.array(interior["rachis_points_mm"])
    terminal = next(l for l in interior["leaflets"] if l["tag"] == "T")
    axis_mm = float(np.linalg.norm(np.diff(rachis, axis=0), axis=1).sum()) + terminal["length_mm"]
    p3_len = compound["rachis_arc_length_mm"]
    tol_len = constant(cfg, "contract_length_tolerance_rel")
    if abs(axis_mm - p3_len) > tol_len * p3_len:
        return dict(reason="rachis_length", value_mm=axis_mm, p3_mm=p3_len)
    _, left0, _ = initial_frame(np.array(compound["direction_unit"]))
    attach = np.array(compound["attachment_mm"])
    reach_left = reach_right = 0.0
    for leaf in interior["leaflets"]:
        pts = flat_outline(leaf, contours[leaf["contour_index"]])
        lateral = (pts - attach[None]) @ left0
        reach_left = max(reach_left, float(lateral.max()))
        reach_right = max(reach_right, float(-lateral.min()))
    tol = 1.0 + constant(cfg, "contract_extent_tolerance_rel")
    if reach_left > tol * compound["left_extent_mm"]:
        return dict(reason="left_extent", value_mm=reach_left, p3_mm=compound["left_extent_mm"])
    if reach_right > tol * compound["right_extent_mm"]:
        return dict(reason="right_extent", value_mm=reach_right, p3_mm=compound["right_extent_mm"])
    return None


def build_compound(cfg: dict[str, Any], compound: dict[str, Any], contours: list[np.ndarray],
                   seed: int, plant_id: str, *, fixed: dict[str, Any] | None = None,
                   backend: str = "lpy") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sample, derive and check one compound leaf; resample on contract violation.

    ``backend="lpy"`` (default) derives with L-Py and interprets with PlantGL (tomato_synth env);
    ``backend="numpy"`` uses :func:`derive_modules` + :func:`interpret_numpy` (any env). ``fixed``
    pins whitelisted interior parameters after the draw (:data:`FIXED_WHITELIST`).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} not in {BACKENDS}")
    segments = int(constant(cfg, "rachis_segments"))
    rejections = []
    for attempt in range(int(constant(cfg, "max_compound_attempts"))):
        p = sample_interior(cfg, compound, len(contours), seed, attempt, fixed)
        events, seg, terminal = _events(p, compound["rachis_arc_length_mm"], segments)
        args = ([], 0.0, 0.0, 1, terminal) if p["bud"] else (
            events, seg, p["rachis_droop_deg"] / segments, segments, terminal)
        if backend == "lpy":
            interior = interpret(derive(*args), compound, p)
        else:
            interior = interpret_numpy(derive_modules(*args), compound, p)
        problem = check_contract(compound, interior, contours, cfg)
        if problem is None:
            for leaf in interior["leaflets"]:
                leaf["leaflet_id"] = f"{plant_id}/{compound['compound_id']}/{leaf['tag']}"
            interior.update(attempt=attempt, params={k: v for k, v in p.items() if k != "leaflets"})
            return interior, rejections
        rejections.append(dict(compound_id=compound["compound_id"], attempt=attempt, **problem))
    raise CompoundRejected(compound["compound_id"], rejections)
