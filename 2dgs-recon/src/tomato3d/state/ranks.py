"""Leaf rank under the 2026-10-14 record (U5; MTG Amendment A7, 2026-10-17) - the ONE helper every producer uses.

Rule (binding, `handoff/MTG_CONVENTION_2026-10-10.md` section 18):

* origin = the cotyledon node (rank 0); cotyledons are never counted;
* nodes are counted upward from the cotyledon node, pruning scars included (a scar is a node whose leaf is gone);
* the first two true leaves above the cotyledons share rank 1; node ``i`` (1-based from the cotyledon node) carries
  rank ``max(1, i - 1)``; so ``true_leaves = ranks + 1`` (:func:`tomato3d.state.schema.true_leaves_from_ranks`);
* when the cotyledon node is not visible / not recorded, rank is ``unavailable`` for the whole axis with the reason
  :data:`COTYLEDON_NOT_VISIBLE` - a rank is never inferred from the lowest visible leaf;
* side axes: ranks restart at 1 at the first leaf of the axis (A4 15.3). Whether a side axis's first two leaves also
  share rank 1 is NOT decided (A4 tie rule stays ``proposed``), so side axes keep one rank per node (``shared_first``
  False) until the user decides.

The generator (``tomato3d/synth/structure.py``) has no cotyledon node and no scars: its ``rank_from_bottom`` is a node
index from the stem base. Under :data:`GENERATOR_RANK_BASIS` (the lowest generated compound is the first true leaf)
its U5 rank is :func:`u5_rank_of_node` of that index - a derivation, recorded wherever it is applied.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

COTYLEDON_NOT_VISIBLE = "cotyledon_node_not_visible"
INSERTED_BELOW = "new node below an already counted node; U5 counts from the cotyledon node, so the earlier count was incomplete"
#: assumption used when generator node indices are converted (status to write beside any such rank)
GENERATOR_RANK_BASIS = "generator_assumed:no_cotyledon_node_no_scars;lowest_compound_is_first_true_leaf"
RANK1_NODE_AMBIGUOUS = "rank 1 on a shared-first axis is node 1 or node 2; the node index needs the record or the truth"
SCAR = None  # a node whose leaf is gone (pruning scar) in a node list


def u5_rank_of_node(node_index: int, shared_first: bool = True) -> int:
    """Rank of the node ``node_index`` (1 = the first node above the cotyledon node).

    ``shared_first`` True (main axis, U5): nodes 1 and 2 -> rank 1, node i -> i - 1. False (side axes, A4 tie rule
    undecided): node i -> i.
    """
    if node_index < 1:
        raise ValueError(f"node index {node_index} < 1 (0 is the cotyledon node, never ranked)")
    return max(1, node_index - 1) if shared_first else node_index


@dataclass(frozen=True)
class RankAssignment:
    ranks: dict[str, int]
    unresolved: dict[str, str] = field(default_factory=dict)
    node_index: dict[str, int] = field(default_factory=dict)
    reason: str | None = None


def u5_ranks(nodes: Sequence[str | None], *, cotyledon_visible: bool = True, shared_first: bool = True) -> RankAssignment:
    """Ranks for an ordered node list ``nodes`` (bottom first, starting at the first node ABOVE the cotyledon node).

    Entries are leaf ids or :data:`SCAR` for a node whose leaf is gone. With ``cotyledon_visible`` False nothing is
    ranked (reason :data:`COTYLEDON_NOT_VISIBLE`).
    """
    if not cotyledon_visible:
        return RankAssignment({}, {k: COTYLEDON_NOT_VISIBLE for k in nodes if k is not None}, {}, COTYLEDON_NOT_VISIBLE)
    ranks, index = {}, {}
    for i, k in enumerate(nodes, start=1):
        if k is not None:
            ranks[k] = u5_rank_of_node(i, shared_first); index[k] = i
    return RankAssignment(ranks, {}, index)


def u5_ranks_over_captures(heights_by_capture: Sequence[Mapping[str, float]], *, shared_first: bool = True,
                           cotyledon_visible: bool = True) -> list[RankAssignment]:
    """One axis over successive captures: a leaf that disappears leaves a scar (its node still counts); a leaf that
    appears above every counted node takes the next node; a leaf appearing BELOW a counted node is unresolved
    (:data:`INSERTED_BELOW`) - U5 has no case for it other than recounting from the cotyledon node."""
    out: list[RankAssignment] = []
    nodes: list[str | None] = []          # node list so far (ids, SCAR for lost leaves)
    height_of: dict[str, float] = {}
    for heights in heights_by_capture:
        if not cotyledon_visible:
            out.append(u5_ranks(list(heights), cotyledon_visible=False)); continue
        unresolved: dict[str, str] = {}
        known = [k for k in nodes if k is not None]
        top = max((height_of[k] for k in known), default=float("-inf"))
        for k in sorted((k for k in heights if k not in height_of), key=lambda k: (heights[k], k)):
            if heights[k] > top:
                nodes.append(k); height_of[k] = heights[k]; top = heights[k]
            else:
                unresolved[k] = INSERTED_BELOW
        present = u5_ranks(nodes, shared_first=shared_first)
        ranks = {k: r for k, r in present.ranks.items() if k in heights}
        idx = {k: i for k, i in present.node_index.items() if k in heights}
        out.append(RankAssignment(ranks, unresolved, idx))
    return out


def node_index_from_rank(rank: int, shared_first: bool = True) -> int | None:
    """Exact inverse of :func:`u5_rank_of_node` where it exists: on a shared-first axis rank r >= 2 is node r + 1
    (A7, by definition); rank 1 is node 1 OR node 2 -> ``None`` (:data:`RANK1_NODE_AMBIGUOUS`, never guessed).
    On an axis without the shared first rank (side axes while the tie rule is open) node = rank."""
    rank = int(rank)
    if rank < 1:
        raise ValueError(f"rank {rank} < 1")
    if not shared_first:
        return rank
    return None if rank == 1 else rank + 1


def rank_node_consistent(rank: int, node_index: int, shared_first: bool = True) -> bool:
    """True when ``rank`` is the U5 rank of ``node_index`` on that kind of axis."""
    return u5_rank_of_node(int(node_index), shared_first) == int(rank)


def rank1010_to_u5(rank_from_bottom: int, shared_first: bool = True) -> int:
    """Generator / 10-10 'lowest present = 1' index -> U5 rank under :data:`GENERATOR_RANK_BASIS`."""
    return u5_rank_of_node(int(rank_from_bottom), shared_first)


def conversion_record(ids_to_old: Mapping[str, int], shared_first: bool = True) -> list[dict[str, object]]:
    """Per-vertex old -> new record for reports (never a silent rewrite)."""
    return [dict(vertex_id=k, rank_1010=int(v), rank_u5=rank1010_to_u5(int(v), shared_first), basis=GENERATOR_RANK_BASIS)
            for k, v in sorted(ids_to_old.items())]


def true_leaf_count(ranks: Iterable[int]) -> int:
    """Leaf count from a set of ranks present (U5): the highest rank + 1 when rank 1 is present twice or more is
    implied by the convention; callers that have a rank COUNT use :func:`schema.true_leaves_from_ranks` instead."""
    from tomato3d.state import schema as s
    rs = list(ranks)
    return s.true_leaves_from_ranks(max(rs)) if rs else 0
