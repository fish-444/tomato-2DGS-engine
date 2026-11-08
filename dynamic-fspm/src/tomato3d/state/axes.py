"""Axes and side shoots (MTG Amendment A4): human registry, axes.csv, cluster classification.

User decisions of 2026-10-12 (``user_stated``):

* a side shoot first appears with **two leaves together**; it is registered as an ``axis_shoot``
  (order-1 axis), ``first_observed`` = that capture, emergence interval = (previous capture, this capture];
* it becomes a ``substem`` when its first two leaves are fully expanded; its apex bundle is then shown as
  ``shoot′`` (S1), ``shoot″`` (S2)...; the main apex bundle stays ``shoot``. Primes are display-only;
* ranks are counted on every axis by the same unfolding rule and never renumbered; a tie of the first two
  leaves of a side axis -> the basal leaf is rank 1 (proposal in A4, **confirmed by the user 2026-10-12**, A5);
* side-shoot numbering = registration order, immutable, same capture -> lower node first (proposal in A4,
  **confirmed 2026-10-12**, A5); a sympodial continuation above a truss is main stem (M), never a substem;
  a truss bears no leaves, so the two-leaf rule already excludes it (trusses: :mod:`tomato3d.state.trusses`);
* ``topology_scope = provided_axes``: unobserved side shoots are never generated.

Amendment A6 (approved by the user 2026-10-14; decisions U2 / U6, "곁순에도 동일한 법칙"):

* every axis ENDS at its first sympodial branch (a Y with a truss in the crotch); the leafy stem continuing
  above is a NEW axis of order + 1 with ``origin = post_first_truss``; ordinary axillary shoots are
  ``origin = axillary``. The A4/A5 rule "the continuation above a truss stays the same axis" is withdrawn for
  every axis (``segment_index`` deprecated). Both arms of a Y with leaves -> two registered axes;
* spelling: dotted path ``S1.1`` (order 2, parent_axis S1, parent_rank = leaf rank on S1), ``S1.1.1`` (order 3);
  no order cap in code. Numbering = registration order among the children of the same parent axis;
* registry columns ``order``, ``parent_axis``, ``origin``; an A4 file (no such columns) still parses as
  order 1 / parent M / origin axillary (under A4 a registered axis could only be an axillary shoot).

The registry ``data/records/side_shoots/side_shoot_registry.csv`` is human-owned and read-only; it is the only
source of ``parent_rank`` from top views. A template is shipped under ``build/`` (never into data/records/).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from tomato3d.io.schemas import UNAVAILABLE
from tomato3d.state import schema as s

REGISTRY_PATH = Path("data/records/side_shoots/side_shoot_registry.csv")
#: Prompt S3 columns + ``state_date`` (the date the row's ``axis_state`` was recorded), needed to place the
#: axis_shoot -> substem change at a capture. One row per (plant, axis, state); rows are a state history.
REGISTRY_COLUMNS_V1 = ("plant_id", "axis", "parent_rank", "first_observed", "axis_state", "state_date", "note")
#: A6 (2026-10-14): + ``order``, ``parent_axis`` (both must agree with the dotted axis path) and ``origin``.
REGISTRY_COLUMNS = ("plant_id", "axis", "order", "parent_axis", "parent_rank", "origin", "first_observed",
                    "axis_state", "state_date", "note")
REGISTRY_ORIGINS = ("axillary", "post_first_truss")
REGISTRY_STATES = ("axis_shoot", "substem")


class RegistryError(ValueError):
    """The registry breaks its own rules; reported, never repaired (CONVENTIONS section 4)."""


@dataclass(frozen=True)
class RegistryEntry:
    plant_id: str
    axis: str
    parent_rank: int | None
    first_observed: date
    states: tuple[tuple[date, str], ...]  # (state_date, axis_state), chronological
    row_refs: tuple[str, ...]
    origin: str = "axillary"  # A6: axillary | post_first_truss (A4 files: axillary)

    @property
    def order(self) -> int:
        return s.axis_order(self.axis)

    @property
    def parent_axis(self) -> str:
        return s.axis_parent(self.axis) or "M"


def _date(text: str, where: str) -> date:
    try:
        return date.fromisoformat(text.strip())
    except ValueError as exc:
        raise RegistryError(f"{where}: {text!r} is not an ISO date (YYYY-MM-DD)") from exc


def read_registry(path: Path) -> list[RegistryEntry]:
    """Read and validate the registry. Every violated rule raises; nothing is filled in."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        if header not in (REGISTRY_COLUMNS, REGISTRY_COLUMNS_V1):
            raise RegistryError(f"{path.name}: header {reader.fieldnames} != {REGISTRY_COLUMNS}")
        rows = list(reader)
    legacy = header == REGISTRY_COLUMNS_V1
    s.check_no_primes(rows, path.name)
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = {}
    for i, r in enumerate(rows, start=2):
        where = f"{path.name} line {i}"
        if not s.AXIS_RE.match(r["axis"]) or r["axis"] == "M":
            raise RegistryError(f"{where}: axis {r['axis']!r} must be S<k> or a dotted path S<k>.<j>... (the main "
                                "stem M is not registered)")
        if legacy:
            if s.axis_order(r["axis"]) != 1:
                raise RegistryError(f"{where}: axis {r['axis']!r} is order {s.axis_order(r['axis'])}; an A4 file "
                                    f"(no order / parent_axis / origin columns) holds order-1 axes only - use {REGISTRY_COLUMNS}")
            r = dict(r, order="1", parent_axis="M", origin="axillary")
            rows[i - 2] = r
        else:
            if r["order"].strip() != str(s.axis_order(r["axis"])):
                raise RegistryError(f"{where}: order {r['order']!r} != {s.axis_order(r['axis'])}, the depth of "
                                    f"{r['axis']} (S1 -> 1, S1.1 -> 2, ...)")
            if r["parent_axis"].strip() != s.axis_parent(r["axis"]):
                raise RegistryError(f"{where}: parent_axis {r['parent_axis']!r} != {s.axis_parent(r['axis'])} "
                                    f"(the path of {r['axis']} names its parent)")
            if r["origin"].strip() not in REGISTRY_ORIGINS:
                raise RegistryError(f"{where}: origin {r['origin']!r} not in {REGISTRY_ORIGINS}")
        if r["axis_state"] not in REGISTRY_STATES:
            raise RegistryError(f"{where}: axis_state {r['axis_state']!r} not in {REGISTRY_STATES}")
        s.check_identifier(r["plant_id"], "plant_id")
        grouped.setdefault((r["plant_id"], r["axis"]), []).append((i, r))
    out = []
    for (plant, axis), items in sorted(grouped.items()):
        first = {_date(r["first_observed"], f"{path.name} line {i}") for i, r in items}
        if len(first) != 1:
            raise RegistryError(f"{plant} {axis}: first_observed differs between rows {sorted(first)}")
        pr = {r["parent_rank"].strip() for _, r in items}
        if len(pr) != 1:
            raise RegistryError(f"{plant} {axis}: parent_rank differs between rows {sorted(pr)}")
        pr_text = pr.pop()
        parent_rank = None if pr_text in ("", UNAVAILABLE) else int(pr_text)
        origins = {r["origin"].strip() for _, r in items}
        if len(origins) != 1:
            raise RegistryError(f"{plant} {axis}: origin differs between rows {sorted(origins)}")
        states = sorted((_date(r["state_date"] or r["first_observed"], f"{path.name} line {i}"), r["axis_state"])
                        for i, r in items)
        # a shoot first recorded already expanded (registered late) may start as substem
        seen_substem = False
        for d, st in states:
            if d < next(iter(first)):
                raise RegistryError(f"{plant} {axis}: state date {d} before first_observed")
            if st == "axis_shoot" and seen_substem:
                raise RegistryError(f"{plant} {axis}: substem -> axis_shoot is not a defined change")
            seen_substem |= st == "substem"
        out.append(RegistryEntry(plant, axis, parent_rank, next(iter(first)), tuple(states),
                                 tuple(f"{path.name}#line{i}" for i, _ in items), origins.pop()))
    _check_parents(out)
    _check_numbering(out)
    return out


def _check_parents(entries: Sequence[RegistryEntry]) -> None:
    """A6: an order >= 2 axis needs its parent axis registered, first observed no later than itself."""
    have = {(e.plant_id, e.axis): e for e in entries}
    for e in entries:
        if e.order < 2:
            continue
        parent = have.get((e.plant_id, e.parent_axis))
        if parent is None:
            raise RegistryError(f"{e.plant_id} {e.axis}: parent axis {e.parent_axis} is not registered")
        if e.first_observed < parent.first_observed:
            raise RegistryError(f"{e.plant_id} {e.axis}: first_observed {e.first_observed} before its parent "
                                f"{e.parent_axis} ({parent.first_observed})")


def _check_numbering(entries: Sequence[RegistryEntry]) -> None:
    """Numbering basis (user_stated, A5 16.1; per parent axis since A6): the children of one parent axis are
    numbered in registration order; same date -> lower node first. A registry that numbers otherwise is reported,
    not renumbered."""
    by_parent: dict[tuple[str, str], list[RegistryEntry]] = {}
    for e in entries:
        by_parent.setdefault((e.plant_id, e.parent_axis), []).append(e)
    for (plant, parent), es in by_parent.items():
        ordered = sorted(es, key=lambda e: (e.first_observed, e.parent_rank if e.parent_rank is not None else 10 ** 6,
                                            int(e.axis.rsplit(".", 1)[-1].lstrip("S"))))
        prefix = "S" if parent == "M" else f"{parent}."
        expected = [f"{prefix}{k}" for k in range(1, len(ordered) + 1)]
        if [e.axis for e in ordered] != expected:
            raise RegistryError(f"{plant}: axes on {parent} {[e.axis for e in ordered]} do not follow registration "
                                f"order {expected} ({s.NUMBERING_BASIS})")


def write_template(folder: Path) -> Path:
    """Header + one clearly marked example row, for the human to copy to data/records/side_shoots/."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "side_shoot_registry.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(REGISTRY_COLUMNS)
        w.writerow(["dwt_101", "S1", "1", "M", "2", "axillary", "2026-10-12", "axis_shoot", "2026-10-12",
                    "EXAMPLE ROW - delete before use; two leaves seen together in the axil of main-stem leaf rank 2"])
        w.writerow(["dwt_101", "S2", "1", "M", "7", "post_first_truss", "2026-10-14", "substem", "2026-10-14",
                    "EXAMPLE ROW - delete before use; the leafy stem continuing above the first sympodial branch "
                    "(Y with the first truss in the crotch) at main-stem node rank 7"])
        w.writerow(["dwt_101", "S1.1", "2", "S1", "3", "post_first_truss", "2026-10-14", "axis_shoot", "2026-10-14",
                    "EXAMPLE ROW - delete before use; continuation above the first truss of S1, at S1 rank 3"])
    (folder / "README.md").write_text(
        "# Side-shoot registry (human-written; MTG Amendment A4)\n\n"
        "Copy `side_shoot_registry.csv` to `data/records/side_shoots/side_shoot_registry.csv` and delete the example row.\n"
        "One row per side shoot and recorded state; add a new row (same first_observed, parent_rank) when it becomes a substem.\n\n"
        "| column | meaning |\n|---|---|\n"
        "| plant_id | e.g. dwt_101 |\n"
        "| axis | S1, S2, ... on the main stem; S1.1, S1.2 ... on S1 (order 2); S1.1.1 (order 3). Children of one "
        "parent are numbered in registration order (same day: lower node first). Never use prime characters. |\n"
        "| order | 1 for S<k>, 2 for S<k>.<j>, ... (must match the path) |\n"
        "| parent_axis | M for S<k>; S1 for S1.1; ... (must match the path) |\n"
        "| parent_rank | rank, ON THE PARENT AXIS, of the node in whose axil the axis sits; blank if unknown |\n"
        "| origin | axillary (ordinary side shoot) or post_first_truss (leafy stem continuing above the parent "
        "axis's first sympodial branch: a Y with the truss in the crotch) |\n"
        "| first_observed | date (YYYY-MM-DD) the two leaves were first seen together |\n"
        "| axis_state | axis_shoot (just appeared) or substem (first two leaves fully expanded) |\n"
        "| state_date | date this axis_state was observed |\n"
        "| note | free text |\n\n"
        "Display labels 1, 1′, 1″ are produced by the software; files store axis codes only.\n\n"
        "**Axis end rule (A6, 2026-10-14, every axis):** an axis ends at its first sympodial branch (Y with the "
        "truss in the crotch). The leafy stem above it is a NEW axis of the next order (`origin = "
        "post_first_truss`); if both arms of the Y carry leaves, register both. Ranks restart at 1 on every axis. "
        "Counting (U5): cotyledons are not counted (rank 0 = cotyledon node); the first two true leaves share rank 1; "
        "pruning scars count as nodes. Trusses go in the separate truss registry "
        "(`data/records/trusses/truss_registry.csv`).\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# axes.csv rows
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureClock:
    capture_id: str
    when: datetime | None  # None = capture time unavailable


def _capture_on_or_after(clock: Sequence[CaptureClock], day: date) -> int | None:
    for i, c in enumerate(clock):
        if c.when is not None and c.when.date() >= day:
            return i
    return None


def axes_rows(plant_instance_id: str, plant_id: str, clock: Sequence[CaptureClock],
              entries: Sequence[RegistryEntry], parent_node_of_rank: Mapping[int, str], evidence: str,
              registry_status: str, registration_leaves: Mapping[str, Sequence[str]] | None = None,
              parent_node_index_of: Mapping[str, int] | None = None) -> list[dict[str, str]]:
    """One row per (capture, axis): M always; each registered S-axis from the capture of its first observation.

    Registry date -> capture: the first capture of the plant on or after that date. A registry date before the
    first capture gives ``first_observed_capture_id = unavailable`` and an emergence interval (unavailable, first
    capture]. ``registry_status`` is ``user_stated`` for the human registry and ``synthetic_truth`` for a fixture.
    ``parent_node_index_of`` (axis -> node index of the parent node, revision 9) comes from the fixture truth or a
    record; without it the node index is DERIVED from ``parent_rank`` where A7 makes it exact (parent M: rank r >= 2
    -> node r + 1; other parents: node = rank) and stays unavailable for a main-stem parent rank 1."""
    from tomato3d.state import ranks as rk
    rows: list[dict[str, str]] = []
    mine = [e for e in entries if e.plant_id == plant_id]
    for ci, cap in enumerate(clock):
        rows.append(_row(plant_instance_id, cap.capture_id, "M", f"{plant_id}/A0", "0", "", UNAVAILABLE,
                         s.MAIN_AXIS_PARENT_RANK, "derived", "main", UNAVAILABLE, UNAVAILABLE, UNAVAILABLE,
                         UNAVAILABLE, UNAVAILABLE, UNAVAILABLE, "definition:main_axis", "derived",
                         "MTG A4: the main stem incl. sympodial continuation", evidence))
        for e in mine:
            first_i = _capture_on_or_after(clock, e.first_observed)
            if cap.when is None or cap.when.date() < e.first_observed:
                continue
            state = "axis_shoot"
            since_day = e.first_observed
            for d, st in e.states:
                if d <= cap.when.date():
                    state, since_day = st, d
            since_i = _capture_on_or_after(clock, since_day)
            exact_first = first_i is not None and clock[first_i].when is not None and \
                clock[first_i].when.date() == e.first_observed
            first_id = clock[first_i].capture_id if exact_first else UNAVAILABLE
            if exact_first:
                start = clock[first_i - 1].capture_id if first_i and first_i > 0 else UNAVAILABLE
                end = clock[first_i].capture_id
            else:
                start, end = UNAVAILABLE, (clock[first_i].capture_id if first_i is not None else UNAVAILABLE)
            pr = UNAVAILABLE if e.parent_rank is None else str(e.parent_rank)
            if parent_node_index_of and e.axis in parent_node_index_of:
                pn, pn_status = str(parent_node_index_of[e.axis]), registry_status
            elif e.parent_rank is not None and rk.node_index_from_rank(e.parent_rank, e.parent_axis == "M") is not None:
                pn, pn_status = str(rk.node_index_from_rank(e.parent_rank, e.parent_axis == "M")), "derived"
            else:
                pn, pn_status = UNAVAILABLE, UNAVAILABLE
            leaves = (registration_leaves or {}).get(e.axis, [])
            # parent node ids are known for the main stem only (its rank -> node map); higher orders stay unavailable
            on_main = e.parent_axis == "M"
            rows.append(_row(
                plant_instance_id, cap.capture_id, e.axis, f"{plant_id}/{e.axis}", str(e.order), e.parent_axis,
                parent_node_of_rank.get(e.parent_rank, UNAVAILABLE) if e.parent_rank is not None and on_main
                else UNAVAILABLE,
                pr, UNAVAILABLE if pr == UNAVAILABLE else registry_status, state,
                clock[since_i].capture_id if since_i is not None else UNAVAILABLE, first_id, e.first_observed.isoformat(),
                start, end, ";".join(leaves) if leaves else UNAVAILABLE, ";".join(e.row_refs),
                registry_status if registry_status != "synthetic_truth" else "fixture_declared",
                "side_shoot_registry " + ";".join(e.row_refs), evidence, e.origin, pn, pn_status))
    return rows


def _row(pid: str, capture: str, axis: str, vertex: str, order: str, parent_axis: str, parent_node: str,
         parent_rank: str, parent_rank_status: str, state: str, since: str, first_capture: str, first_date: str,
         start: str, end: str, leaves: str, reg_ref: str, source_kind: str, source_ref: str,
         evidence: str, origin: str = UNAVAILABLE, parent_node_index: str = UNAVAILABLE,
         parent_node_index_status: str = UNAVAILABLE) -> dict[str, str]:
    return dict(
        plant_instance_id=pid, capture_id=capture, axis_id=axis, axis_vertex_id=vertex, order=order,
        parent_axis_id=parent_axis or UNAVAILABLE, parent_node_id=parent_node, parent_rank=parent_rank, stage=state,
        source_kind=source_kind, source_ref=source_ref, axis=axis, axis_state=state,
        parent_rank__status=parent_rank_status if parent_rank != UNAVAILABLE else UNAVAILABLE,
        axis_state_since_capture_id=since, first_observed_capture_id=first_capture, first_observed_date=first_date,
        emergence_interval_start=start, emergence_interval_end=end, registration_leaf_ids=leaves,
        registration_ref=reg_ref,
        numbering_basis=s.NUMBERING_BASIS if axis != "M" else UNAVAILABLE,
        tie_rule=s.TIE_RULE if axis != "M" else UNAVAILABLE,
        # MTG A5 columns, filled by tomato3d.state.trusses.annotate_axes (unavailable without a truss registry)
        segment_index=UNAVAILABLE, segment_index__status=UNAVAILABLE, truss_count=UNAVAILABLE, evidence=evidence,
        # A6: origin from the registry; node / leaf counts need a human count record (none exists) -> unavailable
        origin=origin, node_count_incl_scars=UNAVAILABLE, node_count_incl_scars__status=UNAVAILABLE,
        leaf_present_count=UNAVAILABLE, leaf_present_count__status=UNAVAILABLE,
        # revision 9 (A7): node index of the parent node (fixture truth / record / derived where exact)
        parent_node_index=parent_node_index, parent_node_index__status=parent_node_index_status)


# --------------------------------------------------------------------------------------
# S4 distinction
# --------------------------------------------------------------------------------------

CLASSIFICATION_COLUMNS = ("plant_instance_id", "capture_id", "obs_id", "obs_class", "axis_hint", "axis_hint_source",
                          "axis_state", "cluster_class", "rule")

#: detector / label class -> (obs_class, axis hint). ``sub_stem`` says "a side axis" but not which one.
DETECTOR_AXIS = {"main_stem": "M", "sub_stem": UNAVAILABLE, "growth_shoot": UNAVAILABLE}


def classify(observations: Sequence[Mapping[str, str]], state_of_axis: Mapping[str, str]) -> list[dict[str, str]]:
    """Each observed cluster -> main apex ``shoot``, side ``axis_shoot`` leaves, ``substem`` leaves, main leaves or
    ``truss``, from the explicit axis hint (human / synthetic GT / detector class) and the registry state.
    Nothing is inferred from position: without an axis hint the class is ``*_axis_unavailable``."""
    out = []
    for o in observations:
        axis = o.get("axis_hint", UNAVAILABLE)
        state = "main" if axis == "M" else state_of_axis.get(axis, UNAVAILABLE)
        cls = o["obs_class"]
        if cls == "truss":
            cluster, rule = "truss", "truss bears no leaves: never registers a side shoot (A4)"
        elif cls == "growth_shoot":
            if axis == "M":
                cluster, rule = "shoot", "main apex bundle"
            elif state == "substem":
                cluster, rule = "side_shoot_apex", "apex bundle of a substem (displayed shoot′, shoot″ ...)"
            elif state == "axis_shoot":
                cluster, rule = "axis_shoot", "the bundle is the young side shoot itself"
            else:
                cluster, rule = "growth_shoot_axis_unavailable", "no axis evidence"
        elif cls in ("compound_leaf", "terminal_leaflet", "leaflet"):
            if axis == "M":
                cluster, rule = "main_leaves", "axis hint M"
            elif state in ("axis_shoot", "substem"):
                cluster, rule = f"{state}_leaves", f"axis hint {axis} + registry state {state}"
            elif axis != UNAVAILABLE:
                cluster, rule = "side_leaves_unregistered", f"axis hint {axis} has no registry entry"
            else:
                cluster, rule = "leaves_axis_unavailable", "no axis evidence"
        elif cls == "main_stem":
            cluster, rule = "main_stem", "detector/label class main_stem -> M"
        elif cls == "sub_stem":
            cluster, rule = "side_stem", "detector/label class sub_stem -> some S-axis (index from registry/human)"
        else:
            cluster, rule = "other", cls
        out.append(dict(plant_instance_id=o["plant_instance_id"], capture_id=o["capture_id"], obs_id=o["obs_id"],
                        obs_class=cls, axis_hint=axis, axis_hint_source=o.get("axis_hint_source", UNAVAILABLE),
                        axis_state=state, cluster_class=cluster, rule=rule))
    return out


def state_of_axes(rows: Sequence[Mapping[str, str]], capture_id: str) -> dict[str, str]:
    return {r["axis"]: r["axis_state"] for r in rows if r["capture_id"] == capture_id}
