"""Schema, vocabularies and validators for the P1 current-state package.

One module owns every column tuple, status word and frame token of
``build/p1_state_<YYYYMMDD>_<nn>/`` so that producers (fixture exporter, ingest, tracker,
updater, side geometry, optical export, canopy placement) and the P2 dry-read share one
definition.

Sources of the vocabularies (nothing here is invented without a note):

* value status - MTG convention section 6 (``measured``, ``photo_estimate``, ``reconstructed``,
  ``generated_candidate``, ``assumed``, ``unavailable``) + A1 (``user_stated``, ``app_ratio``,
  ``derived``) + Stage-2 spec section 4 (``observed``, ``model_estimated``). **P1 extension:**
  ``synthetic_truth`` = a value read from a synthetic scene's own mesh; legal only on rows whose
  ``evidence`` is ``synthetic_fixture`` (validator-enforced), so it can never pass as a photo value.
* frame tokens - Stage-2 spec section 3 (``board_mm``, ``plant_local_mm``, ``image_px``). **P1
  extension:** ``canopy_mm`` = the shared canopy frame produced only by
  :mod:`tomato3d.state.canopy` from ``plant_local_mm`` through a declared rigid transform. No
  conversion between ``board_mm`` and ``plant_local_mm`` or ``image_px`` exists or may be added.
* track states / events - phynotype2 spec sections 5.4 and 5.5; the P1 prompt's words (split,
  merge, occluded, disappeared_unconfirmed, removed_confirmed, new) are mapped onto them in
  :data:`PROMPT_TRACK_WORDS`.
* area kinds - Stage-2 spec section 4 + MTG section 13 (``area_basis = full_outline``; green-only
  only as ``area_green_only_*``).

Blade curl / rugosity (MTG 8.3) and flat/bent theta thresholds (MTG 8.4) are on hold: no column
for them exists here and :func:`check_no_on_hold_fields` rejects any table that adds one.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from tomato3d.io.schemas import UNAVAILABLE

#: Version of the P1 package layout. A breaking change bumps it; replay and consumers compare it.
P1_SCHEMA_VERSION: Final[str] = "p1_state_v1"
#: Additive revision within ``p1_state_v1`` (recorded in every manifest). Revision 2 (P1 fix A1/A2,
#: 2026-10-12): ``captures.csv`` gains ``capture_tz`` + ``capture_tz_source``; ``side_geometry.csv``
#: gains ``selected_for_mtg`` + ``view_selection_rule``. Validators still accept revision-1 headers.
P1_SCHEMA_REVISION: Final[int] = 9
#: Revision 9 (MTG Amendment A7 follow-through, 2026-10-17): the node index is stored beside every rank so no
#: consumer reconstructs it from U5 ranks (the first two main-stem leaves share rank 1). ``mtg_vertices.csv``
#: gains ``node_index_from_base`` (CompoundLeaf: 1 = first node above the cotyledon node, scars counted),
#: ``parent_node_index`` (side Axis vertices, beside ``parent_rank``) and ``above_node_index`` (Truss vertices,
#: beside ``parent_rank`` = registry above_rank), each with ``__status``; ``axes.csv`` gains ``parent_node_index``
#: (+ status); ``trusses.csv`` gains ``above_node_index`` (+ status). Values come from the fixture truth or a human
#: record; otherwise they are DERIVED from the rank where the A7 mapping is exact (main stem: rank r >= 2 is node
#: r + 1; side axes: node = rank while the tie rule is open) and stay ``unavailable`` for a main-stem rank 1
#: (node 1 or 2). ``COUNT_CONVENTION`` unchanged. Revision-8 headers are still accepted.
#: Revision 8 (MTG Amendment A6, APPROVED by the user 2026-10-14; decisions U2/U5/U6): an axis ENDS at its first
#: sympodial branch (Y with a truss in the crotch); the leafy stem continuing above it is a NEW axis of order + 1,
#: spelled as a dotted path (``S1`` -> ``S1.1`` -> ``S1.1.1``), on every axis; no order cap in code. ``axes.csv``
#: gains ``origin`` (axillary | post_first_truss), ``node_count_incl_scars`` and ``leaf_present_count`` (+ status).
#: ``segment_index`` is DEPRECATED (one truss per axis -> always 1 under the new rule); kept, never removed.
#: Revision 7 (P1 rim inner plane, 2026-10-13): ``captures.csv`` gains ``scale_reference``, ``scale_reference_plane``
#: (``pot_rim`` | ``medium_surface`` | ``synthetic_camera``) and ``scale_reference_plane_offset_mm`` (plane height
#: relative to the rim: 0 or -depth); ``frame_transforms.csv`` gains the two plane columns. A solved mm scale whose
#: reference plane is unstated is REFUSED (current header: the columns; revision-5/6 header: ``scale_plane``;
#: earlier headers carry no plane, so a real-photo mm value there is refused). New conventions
#: ``pot_rim_edges_square`` (rim plane, outer/inner top edges) and ``pot_medium_junction_square`` (medium surface).
#: Revision 6 (P1 side rim pose, 2026-10-12): new table ``camera_poses.csv``; scale convention
#: ``pot_rim_pose_vertical_axis`` with plane ``pot_axis_vertical`` (side photos, validated pose only).
#: Revision 3 (P1 side shoots, MTG Amendment A4, 2026-10-12): ``axes.csv``; ``axis`` on
#: ``mtg_vertices.csv`` / ``identity_manifest.csv``; MTG attributes ``parent_rank`` (Axis vertices) and
#: ``leaf_unfolded`` (CompoundLeaf vertices); observations gain ``axis_hint`` + ``axis_hint_source``;
#: obs classes ``main_stem``, ``sub_stem``, ``growth_shoot``, ``truss``. Earlier headers are still accepted.
#: Revision 4 (MTG Amendment A5, 2026-10-12): MTG attribute ``segment_index`` (+ status) on every row that
#: carries ``axis``; ``identity_manifest.csv`` gains ``segment_index``; ``axes.csv`` gains ``segment_index``,
#: ``segment_index__status``, ``truss_count``; vertex class ``Truss`` (scale 3); new table ``trusses.csv``.
#: Revision-3 (and earlier) headers are still accepted.
#: Revision 5 (P1 T6 pot-rim scale, 2026-10-12): ``captures.csv`` gains ``mm_per_px_sd``, ``scale_convention``,
#: ``scale_sources``, ``scale_cross_check``, ``scale_plane``; ``frame_transforms.csv`` gains ``mm_per_px_sd``,
#: ``scale_convention``, ``rim_homography_px_to_mm``; ``areas.csv`` gains ``visible_projected_area_mm2_sd`` and
#: ``completed_projected_area_mm2_sd``. A real-photo mm value needs its convention, plane and uncertainty.


class P1SchemaError(ValueError):
    """A P1 table or document violates the contract. The message names the rule."""


# --------------------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------------------

VALUE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "photo_estimate",
        "reconstructed",
        "generated_candidate",
        "assumed",
        UNAVAILABLE,
        "user_stated",
        "app_ratio",
        "derived",
        "observed",
        "model_estimated",
        "synthetic_truth",
    }
)
#: Statuses that may never appear outside a synthetic fixture.
FIXTURE_ONLY_STATUSES: Final[frozenset[str]] = frozenset({"synthetic_truth"})

FRAME_TOKENS: Final[tuple[str, ...]] = ("board_mm", "plant_local_mm", "image_px", "canopy_mm")
MM_FRAMES: Final[frozenset[str]] = frozenset({"board_mm", "plant_local_mm", "canopy_mm"})

EVIDENCE_KINDS: Final[tuple[str, ...]] = ("real_photo", "synthetic_fixture")
OBSERVATION_PROVENANCE: Final[tuple[str, ...]] = ("human", "detector_proposal", "synthetic_gt")
#: ``unvalidated_on_boardfree_top``: the C6 judge ``full_final_all.pt`` (sha256 609fd7ae...). The P1
#: prompt calls it "synthetic-trained"; on disk it is organ detector v4, trained on reviewed masks
#: of the REAL board-era plants dwt_01-04 (AP50 0.28 is its score ON synthetic scenes). It was never
#: validated on board-free top views, so its output is a proposal, never an observation.
DETECTOR_STATUSES: Final[tuple[str, ...]] = (
    "unavailable",
    "unvalidated_on_boardfree_top",
    "not_applicable",
)
REVIEW_STATES: Final[tuple[str, ...]] = ("accepted", "edited", "rejected", "unreviewed")
OBS_CLASSES: Final[tuple[str, ...]] = (
    "compound_leaf",
    "terminal_leaflet",
    "leaflet",
    "stem",
    "other",
    "main_stem",
    "sub_stem",
    "growth_shoot",
    "truss",
)

# ----- axes and side shoots (MTG Amendment A4) ------------------------------------------
#: Stored axis code: ``M`` main stem (cotyledon node up to its first sympodial branch, A6), ``S<k>`` k-th axis
#: registered on M, and dotted paths for higher orders: ``S1.1`` = first axis registered on S1 (order 2),
#: ``S1.1.1`` (order 3) ... (A5 notation, 2026-10-12; order >= 2 allowed since A6, 2026-10-14; no cap in code).
#: Prime characters never appear in a file; display labels (1, 1′, 1″) are made by :func:`display_label`.
AXIS_RE: Final[re.Pattern[str]] = re.compile(r"^(M|S[1-9][0-9]*(\.[1-9][0-9]*)*)$")
#: A dotted (order >= 2) axis path. Accepted since A6; kept as a name for callers.
ORDER2_AXIS_RE: Final[re.Pattern[str]] = re.compile(r"^S[1-9][0-9]*(\.[1-9][0-9]*)+$")
ORDER2_RESERVED: Final[str] = ("dotted axis paths: S1.1 = first axis registered on S1 (order 2, parent_axis S1, "
                               "parent_rank = rank on S1); S1.1.1 order 3 (MTG A6)")
AXIS_STATES: Final[tuple[str, ...]] = ("main", "axis_shoot", "substem")
#: A6 ``origin`` of a registered axis: an ordinary axillary shoot, or the leafy stem continuing above the first
#: sympodial branch (Y + truss in the crotch) of its parent axis.
AXIS_ORIGINS: Final[tuple[str, ...]] = ("axillary", "post_first_truss", UNAVAILABLE)


def axis_order(code: str) -> int:
    """``M`` -> 0, ``S3`` -> 1, ``S1.2`` -> 2, ``S1.2.1`` -> 3 (A6: no cap)."""
    if not AXIS_RE.match(code):
        raise P1SchemaError(f"axis {code!r} is not M, S<k> or a dotted path S<k>.<j>...")
    return 0 if code == "M" else code.count(".") + 1


def axis_parent(code: str) -> str | None:
    """Parent axis of a code: ``S3`` -> ``M``, ``S1.2`` -> ``S1``; ``M`` -> None."""
    order = axis_order(code)
    if order == 0:
        return None
    return "M" if order == 1 else code.rsplit(".", 1)[0]


# ----- counting convention (U5, 2026-10-14) ----------------------------------------------
#: Rank 0 = the cotyledon node; cotyledons are never counted. The FIRST TWO true leaves above the cotyledons
#: share rank 1, so true leaves = ranks + 1. Pruning scars count as nodes. Every count field states this.
COUNT_CONVENTION: Final[str] = "scars_included;first_two_leaves_one_rank"
RANK_TO_TRUE_LEAF_OFFSET: Final[int] = 1


def true_leaves_from_ranks(ranks: int) -> int:
    """True-leaf count for literature comparison (``leaves_to_first_truss_true``) from a rank count (U5)."""
    if ranks < 0:
        raise P1SchemaError(f"rank count {ranks} < 0")
    return ranks + RANK_TO_TRUE_LEAF_OFFSET


#: MTG attribute definitions changed by A6 (U2): ``stem_length_mm`` of the main stem runs from the cotyledon
#: node to the node of its first truss (the axis end); plant height is ``height_top_leaf_mm`` only.
STEM_LENGTH_DEFINITION: Final[str] = "main_stem: cotyledon node -> node of the first truss (A6, U2 2026-10-14)"
#: ``segment_index`` (A5) is deprecated by A6: an axis ends at its first truss, so it is always 1 under the new
#: rule; a value > 1 only arises from a registry written under A5 (leaves above a truss on the same axis).
SEGMENT_INDEX_DEPRECATED: Final[str] = "deprecated_A6:axis_ends_at_first_truss"
#: MTG A5: a truss is a scale-3 vertex on its bearing axis, beside CompoundLeaf; it bears no leaves.
TRUSS_CLASS: Final[str] = "Truss"
AXIS_HINT_SOURCES: Final[tuple[str, ...]] = ("human", "synthetic_gt", "detector_class", UNAVAILABLE)
#: Characters that render as primes; never stored (A4).
PRIME_CHARACTERS: Final[str] = "\u2032\u2033\u2034\u2057\u02b9\u02ba"
#: A4 items, flagged in every axes row. Revision <= 3 packages wrote the ``proposed:`` values; the user
#: confirmed both on 2026-10-12 (MTG A5 16.1), so new packages write ``user_stated:``.
NUMBERING_BASIS_PROPOSED: Final[str] = "proposed:registration_order_immutable;same_capture_lower_node_first"
TIE_RULE_PROPOSED: Final[str] = "proposed:basal_leaf_of_side_axis_is_rank_1"
NUMBERING_BASIS: Final[str] = "user_stated:registration_order_immutable;same_capture_lower_node_first"
TIE_RULE: Final[str] = "user_stated:basal_leaf_of_side_axis_is_rank_1"
#: fsp-axes-1 (handoff/P4_SIDESHOOT_SCIENCE_AND_IO_CONTRACT_2026-10-12.md section 3) uses parent_rank 0
#: for the main axis; it is a sentinel, not a leaf rank.
MAIN_AXIS_PARENT_RANK: Final[str] = "0"
PLANT_ATTRIBUTION: Final[tuple[str, ...]] = ("this_plant", "neighbor", "uncertain")
MEMBERSHIP_STATUSES: Final[tuple[str, ...]] = ("confirmed", "proposed", "unassigned")

#: organ id of an observation that cannot be given an id (a null threshold decides its fate);
#: never minted as a track, always ``ambiguous``.
UNASSIGNED_ORGAN: Final[str] = "unassigned"
CORRESPONDENCE_STATUSES: Final[tuple[str, ...]] = ("proposed", "confirmed", "rejected")
TRACK_STATES: Final[tuple[str, ...]] = (
    "continued",
    "unobserved",
    "reappeared",
    "first_observed",
    "removal_confirmed",
    "ambiguous",
)
UNOBSERVED_REASONS: Final[tuple[str, ...]] = ("occluded", "not_detected", "out_of_frame", "unknown")
EVENT_TYPES: Final[tuple[str, ...]] = (
    "fragment_suspect",
    "merge_suspect",
    "track_merge",
    "rename_same_organ",
    "removal_confirmed",
    "axis_assigned",
)
#: P1 prompt wording -> phynotype2 contract (state, reason or event). Documentation + tests.
PROMPT_TRACK_WORDS: Final[dict[str, str]] = {
    "new": "state=first_observed",
    "occluded": "state=unobserved,unobserved_reason=occluded",
    "disappeared_unconfirmed": "state=unobserved,unobserved_reason=not_detected|unknown",
    "removed_confirmed": "state=removal_confirmed",
    "split": "event=fragment_suspect",
    "merge": "event=merge_suspect",
}

OPTICAL_READINESS: Final[tuple[str, ...]] = ("supported", "conditional_completion", "unavailable")
RUN_STATUSES: Final[tuple[str, ...]] = ("executed", "fixture_only", "unavailable", "failed")
AREA_BASIS_MAIN: Final[str] = "full_outline"
RACHIS_BEND_TYPES: Final[tuple[str, ...]] = ("flat", "bent", "intermediate", "unknown")
#: ``rachis_bend_type`` is a human label only (MTG 8.4 on hold).
RACHIS_BEND_TYPE_STATUSES: Final[frozenset[str]] = frozenset({"user_stated", UNAVAILABLE})
VIEWS: Final[tuple[str, ...]] = ("top", "side", "level", "oblique")
PHOTOPERIOD_PHASES: Final[tuple[str, ...]] = ("am", "pm", UNAVAILABLE)
SCALE_STATUSES: Final[tuple[str, ...]] = ("solved", UNAVAILABLE)
#: P1 T6: which convention produced a mm value. Square cells: ``pot_rim_outer_edge_mm`` (one side, outer top edge,
#: across the flats); round pots: ``pot_rim_inner_diameter_mm`` (never both on one pot record).
SCALE_CONVENTIONS: Final[tuple[str, ...]] = (
    "pot_rim_outer_edge_square", "pot_rim_edges_square", "pot_medium_junction_square",
    "pot_rim_inner_diameter_round", "pot_rim_pose_vertical_axis", "synthetic_ortho_camera_declared", UNAVAILABLE)
REAL_SCALE_CONVENTIONS: Final[tuple[str, ...]] = (
    "pot_rim_outer_edge_square", "pot_rim_edges_square", "pot_medium_junction_square",
    "pot_rim_inner_diameter_round", "pot_rim_pose_vertical_axis")
#: STAGE1 rule as stated in T6: within 2 % accept, over 5 % stop; between is flagged and reported.
SCALE_CROSS_CHECKS: Final[tuple[str, ...]] = (
    "accepted_within_2pct", "flagged_2_to_5pct", "stopped_over_5pct", "none_single_source", UNAVAILABLE)
#: ``pot_axis_vertical`` (P1 side rim pose, revision 6): a side photo's mm per px of a VERTICAL step on the pot axis
#: at the rim centre, from the camera pose the rim gives (``tomato3d.state.camera``); only with a validated pose.
SCALE_PLANES: Final[tuple[str, ...]] = (
    "rim_plane", "medium_surface_plane", "pot_axis_vertical", "synthetic_camera", UNAVAILABLE)
#: plane each real-photo convention must name
CONVENTION_PLANE: Final[dict[str, str]] = {
    "pot_rim_outer_edge_square": "rim_plane", "pot_rim_edges_square": "rim_plane",
    "pot_medium_junction_square": "medium_surface_plane", "pot_rim_inner_diameter_round": "rim_plane",
    "pot_rim_pose_vertical_axis": "pot_axis_vertical"}
#: Revision 7: the PHYSICAL plane the reference lies on (``scale_reference_plane``) + its offset from the rim.
SCALE_REFERENCE_PLANES: Final[tuple[str, ...]] = ("pot_rim", "medium_surface", "synthetic_camera", UNAVAILABLE)
CONVENTION_REFERENCE_PLANE: Final[dict[str, str]] = {
    "pot_rim_outer_edge_square": "pot_rim", "pot_rim_edges_square": "pot_rim",
    "pot_medium_junction_square": "medium_surface", "pot_rim_inner_diameter_round": "pot_rim",
    "pot_rim_pose_vertical_axis": "pot_rim", "synthetic_ortho_camera_declared": "synthetic_camera"}
POSE_STATUSES: Final[tuple[str, ...]] = ("validated", "approximate", UNAVAILABLE)

# --------------------------------------------------------------------------------------
# Column contracts (CSV). Order is part of the contract.
# --------------------------------------------------------------------------------------

#: ``capture_tz`` = IANA zone name (e.g. ``Asia/Seoul``) or a fixed offset ``UTC+HH:MM``;
#: ``capture_tz_source`` in :data:`CAPTURE_TZ_SOURCES`. A real photo's time is timezone-aware only when
#: a timezone was stated; otherwise it stays naive (the device clock carries no offset) - never guessed.
CAPTURE_TZ_SOURCES: Final[tuple[str, ...]] = ("user_stated", "assumed", UNAVAILABLE)

CAPTURES_COLUMNS: Final[tuple[str, ...]] = (
    "capture_id",
    "plant_instance_id",
    "source_plant_id",
    "evidence",
    "view",
    "capture_time",
    "capture_time__status",
    "capture_tz",
    "capture_tz_source",
    "time_since_lights_on_min",
    "time_since_lights_on_min__status",
    "photoperiod_phase",
    "frame",
    "scale_source",
    "scale_status",
    "mm_per_px",
    "mm_per_px_sd",
    "scale_convention",
    "scale_sources",
    "scale_cross_check",
    "scale_plane",
    "scale_reference",
    "scale_reference_plane",
    "scale_reference_plane_offset_mm",
    "rotation_status",
    "image_path",
    "image_sha256",
    "session_id",
    "note",
)

_REV5_CAPTURE_ADDED: Final[tuple[str, ...]] = (
    "mm_per_px_sd", "scale_convention", "scale_sources", "scale_cross_check", "scale_plane")
_REV7_CAPTURE_ADDED: Final[tuple[str, ...]] = (
    "scale_reference", "scale_reference_plane", "scale_reference_plane_offset_mm")
#: Revision 5-6 header of ``captures.csv`` (2026-10-12), still accepted; its ``scale_plane`` states the plane.
CAPTURES_COLUMNS_REV6: Final[tuple[str, ...]] = tuple(c for c in CAPTURES_COLUMNS if c not in _REV7_CAPTURE_ADDED)
#: Revision 2-4 header of ``captures.csv`` (before T6), still accepted.
CAPTURES_COLUMNS_REV4: Final[tuple[str, ...]] = tuple(c for c in CAPTURES_COLUMNS_REV6 if c not in _REV5_CAPTURE_ADDED)
#: Revision-1 header of ``captures.csv`` (packages built before 2026-10-12), still accepted.
CAPTURES_COLUMNS_REV1: Final[tuple[str, ...]] = tuple(
    c for c in CAPTURES_COLUMNS_REV4 if c not in ("capture_tz", "capture_tz_source"))

OBSERVATIONS_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "obs_id",
    "obs_class",
    "axis_hint",
    "axis_hint_source",
    "provenance",
    "source_id",
    "source_score",
    "detector_status",
    "review_state",
    "mask_ref",
    "mask_quality",
    "area_px2",
    "bbox_xyxy_px",
    "centroid_xy_px",
    "touches_other_mask",
    "touches_image_border",
    "plant_attribution",
    "parent_obs_id",
    "membership_status",
    "frame",
    "evidence",
)

#: Fixture-only join of observation ids to the scene's truth vertex ids. The tracker never
#: reads this file; only evaluation does.
TRUTH_JOIN_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "obs_id",
    "truth_vertex_id",
    "evidence",
)

IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "source_plant_id",
    "capture_id",
    "organ_id",
    "organ_class",
    "axis",
    "segment_index",
    "parent_id",
    "obs_id",
    "track_state",
    "unobserved_reason",
    "correspondence_status",
    "match_cost",
    "match_margin",
    "gap_captures",
    "emergence_interval_start",
    "emergence_interval_end",
    "events",
    "evidence",
)

EVENTS_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "plant_instance_id",
    "capture_id",
    "event_type",
    "track_ids",
    "obs_ids",
    "decision_ref",
)

MATCH_CANDIDATES_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "track_id",
    "obs_id",
    "d_azimuth_deg",
    "d_radial_px",
    "shape_compound",
    "mask_iou_prev",
    "dt_h",
    "same_phase",
    "gap_captures",
    "cost",
)

#: MTG 12.3: one column per lateral-pair node. The CSV needs a fixed header, so P1 allocates
#: nodes 1..MAX_BEND_NODES; a compound with more nodes is a hard error, never a truncation.
#: (Generator ``leaflet_pairs`` is 1-3; user-stated real compounds carry 11-13 leaflets.)
MAX_BEND_NODES: Final[int] = 8
ALPHA_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"rachis_bend_alpha_{i}_deg" for i in range(1, MAX_BEND_NODES + 1)
)

#: MTG section 9 + 12.7 + A1.2 columns, each value with ``__status``. ``base_xyz_mm`` /
#: ``tip_xyz_mm`` / ``normal_xyz`` hold a JSON triple, as the V3 Stage-2 boundary reads them.
MTG_VALUE_ATTRIBUTES: Final[tuple[str, ...]] = (
    "base_xyz_mm",
    "tip_xyz_mm",
    "length_mm",
    "width_mm",
    "area_mm2",
    "azimuth_deg",
    "elevation_deg",
    "normal_xyz",
    "rank",
    "order",
    "parent_rank",
    "leaf_unfolded",
    "segment_index",
    "compound_insertion_angle_deg",
    "leaflet_insertion_angle_deg",
    *ALPHA_COLUMNS,
    "n_bend_nodes",
    "rachis_curvature_deg",
    "rachis_bend_theta_deg",
    "chord_to_path_ratio",
    "rachis_bend_type",
    "petiole_base_azimuth_deg",
    "blade_azimuth_deg",
    "blade_azimuth_basis",
    "variety",
    "slot_id",
    "light_ratio",
    "height_top_leaf_mm",
    "stem_length_mm",
    # revision 9 (A7): node index beside every rank
    "node_index_from_base",
    "parent_node_index",
    "above_node_index",
)
MTG_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "plant_id",
    "capture_id",
    "snapshot_capture_id",
    "vertex_id",
    "scale",
    "class",
    "axis",
    "complex_id",
    "parent_id",
    "edge_type",
    "display_name",
    "pair_index",
    "side",
    "node_index",
    "observed",
    "candidate_id",
    "frame",
    "evidence",
)
MTG_VERTICES_COLUMNS: Final[tuple[str, ...]] = MTG_IDENTITY_COLUMNS + tuple(
    column for attr in MTG_VALUE_ATTRIBUTES for column in (attr, f"{attr}__status")
)
#: Revision <= 2 headers (packages built before 2026-10-12 side-shoot work), still accepted.
_REV3_MTG_ADDED: Final[frozenset[str]] = frozenset(
    {"axis", "parent_rank", "parent_rank__status", "leaf_unfolded", "leaf_unfolded__status"})
#: Revision 3 headers (before MTG A5, 2026-10-12), still accepted.
_REV4_MTG_ADDED: Final[frozenset[str]] = frozenset({"segment_index", "segment_index__status"})
#: Revision 9 (A7 node index) additions; revision 4-8 headers are still accepted.
_REV9_MTG_ADDED: Final[frozenset[str]] = frozenset({"node_index_from_base", "node_index_from_base__status",
                                                    "parent_node_index", "parent_node_index__status",
                                                    "above_node_index", "above_node_index__status"})
MTG_VERTICES_COLUMNS_REV8: Final[tuple[str, ...]] = tuple(c for c in MTG_VERTICES_COLUMNS if c not in _REV9_MTG_ADDED)
MTG_VERTICES_COLUMNS_REV3: Final[tuple[str, ...]] = tuple(c for c in MTG_VERTICES_COLUMNS_REV8 if c not in _REV4_MTG_ADDED)
MTG_VERTICES_COLUMNS_REV2: Final[tuple[str, ...]] = tuple(
    c for c in MTG_VERTICES_COLUMNS_REV3 if c not in _REV3_MTG_ADDED)
IDENTITY_COLUMNS_REV3: Final[tuple[str, ...]] = tuple(c for c in IDENTITY_COLUMNS if c != "segment_index")
IDENTITY_COLUMNS_REV2: Final[tuple[str, ...]] = tuple(c for c in IDENTITY_COLUMNS_REV3 if c != "axis")
OBSERVATIONS_COLUMNS_REV2: Final[tuple[str, ...]] = tuple(
    c for c in OBSERVATIONS_COLUMNS if c not in ("axis_hint", "axis_hint_source"))

#: ``axes.csv`` - one row per (plant instance, capture, axis). The first block is the fsp-axes-1
#: producer exchange of handoff/P4_SIDESHOOT_SCIENCE_AND_IO_CONTRACT_2026-10-12.md section 3
#: (``axis_id`` = A4 ``axis``, ``stage`` = A4 ``axis_state``; the validator enforces equality).
AXES_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "axis_id",
    "axis_vertex_id",
    "order",
    "parent_axis_id",
    "parent_node_id",
    "parent_rank",
    "stage",
    "source_kind",
    "source_ref",
    "segment_index",
    "segment_index__status",
    "truss_count",
    "axis",
    "axis_state",
    "parent_rank__status",
    "axis_state_since_capture_id",
    "first_observed_capture_id",
    "first_observed_date",
    "emergence_interval_start",
    "emergence_interval_end",
    "registration_leaf_ids",
    "registration_ref",
    "numbering_basis",
    "tie_rule",
    "evidence",
    "origin",
    "node_count_incl_scars",
    "node_count_incl_scars__status",
    "leaf_present_count",
    "leaf_present_count__status",
    # revision 9 (A7): node index of the parent node beside parent_rank
    "parent_node_index",
    "parent_node_index__status",
)

_REV9_AXES_ADDED: Final[tuple[str, ...]] = ("parent_node_index", "parent_node_index__status")
#: Revision 8 ``axes.csv`` header (A6, before the node index), still accepted.
AXES_COLUMNS_REV8: Final[tuple[str, ...]] = tuple(c for c in AXES_COLUMNS if c not in _REV9_AXES_ADDED)

_REV8_AXES_ADDED: Final[tuple[str, ...]] = (
    "origin", "node_count_incl_scars", "node_count_incl_scars__status", "leaf_present_count",
    "leaf_present_count__status")
#: Revision 4-7 ``axes.csv`` header (before A6), still accepted.
AXES_COLUMNS_REV7: Final[tuple[str, ...]] = tuple(c for c in AXES_COLUMNS_REV8 if c not in _REV8_AXES_ADDED)
AXES_COLUMNS_REV3: Final[tuple[str, ...]] = tuple(
    c for c in AXES_COLUMNS_REV7 if c not in ("segment_index", "segment_index__status", "truss_count"))

#: ``trusses.csv`` (MTG A5) - one row per (plant instance, capture, truss) from the capture of the truss's
#: first record on. ``parent_rank`` = the registry's ``above_rank``: rank, on the truss's own (bearing) axis, of
#: the leaf immediately below the truss. ``segment_index`` = the segment the truss closes (1-based).
TRUSSES_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "truss_vertex_id",
    "axis_id",
    "axis_vertex_id",
    "parent_rank",
    "parent_rank__status",
    "segment_index",
    "segment_index__status",
    "first_observed_date",
    "first_observed_capture_id",
    "emergence_interval_start",
    "emergence_interval_end",
    "source_kind",
    "source_ref",
    "registry_ref",
    "evidence",
    # revision 9 (A7): node index of the bearing node beside parent_rank (= above_rank)
    "above_node_index",
    "above_node_index__status",
)
#: Revision 5-8 ``trusses.csv`` header, still accepted.
TRUSSES_COLUMNS_REV8: Final[tuple[str, ...]] = tuple(c for c in TRUSSES_COLUMNS if c not in ("above_node_index", "above_node_index__status"))

MTG_EDGES_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "snapshot_capture_id",
    "parent_id",
    "child_id",
    "edge_type",
)

#: Stage-2 spec section 6 ``structure_state.csv`` + ``plant_instance_id``.
STRUCTURE_STATE_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "plant_id",
    "capture_id",
    "vertex_id",
    "part",
    "upstream_track_state",
    "link_status",
    "time_since_lights_on_min",
    "photoperiod_phase",
    "frame",
    "scale_status",
    "evidence_ref",
)

AREAS_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "compound_id",
    "leaflet_id",
    "part",
    "obs_id",
    "evidence",
    "area_basis",
    "visible_projected_area_px2",
    "visible_projected_area_px2__status",
    "visible_projected_area_mm2",
    "visible_projected_area_mm2__status",
    "visible_projected_area_mm2_sd",
    "scale_source",
    "completed_projected_area_px2",
    "completed_projected_area_px2__status",
    "completed_projected_area_mm2",
    "completed_projected_area_mm2__status",
    "completed_projected_area_mm2_sd",
    "completion_method",
    "completion_coverage",
    "completion_interval_low_px2",
    "completion_interval_high_px2",
    "one_sided_lamina_area_mm2",
    "one_sided_lamina_area_mm2__status",
    "lamina_geometric_evidence",
    "area_green_only_mm2",
    "area_green_only_mm2__status",
)

#: Revision <= 4 ``areas.csv`` header (before T6), still accepted.
AREAS_COLUMNS_REV4: Final[tuple[str, ...]] = tuple(
    c for c in AREAS_COLUMNS if c not in ("visible_projected_area_mm2_sd", "completed_projected_area_mm2_sd"))

OPTICAL_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "source_plant_id",
    "capture_id",
    "external_leaflet_id",
    "vertex_id",
    "compound_id",
    "frame",
    "surface_file",
    "surface_sha256",
    "n_triangles",
    "one_sided_area_mm2",
    "face_convention",
    "area_convention",
    "geometry_source",
    "optical_readiness",
    "readiness_reason",
    "evidence",
)

#: Optical surface triangles; the column set mirrors ``tomato3d.io.schemas.LEAF_SURFACES_CSV_COLUMNS``
#: with the frame named explicitly instead of implied.
OPTICAL_TRIANGLES_COLUMNS: Final[tuple[str, ...]] = (
    "external_leaflet_id",
    "triangle_id",
    "frame",
    "v0_x_mm",
    "v0_y_mm",
    "v0_z_mm",
    "v1_x_mm",
    "v1_y_mm",
    "v1_z_mm",
    "v2_x_mm",
    "v2_y_mm",
    "v2_z_mm",
)

#: ``direction_class`` = the camera-relative terminal-leaflet class of that compound in that view
#: (cl_left | cl_right | cl_toward | cl_away | unclear | unavailable). Views where it is cl_left /
#: cl_right show the compound roughly side-on; P1 joins side values only from such a view.
SIDE_GEOMETRY_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "view_id",
    "compound_id",
    "points_source",
    "direction_class",
    "selected_for_mtg",
    "view_selection_rule",
    "evidence",
    "compound_insertion_angle_deg",
    "compound_insertion_angle_deg__status",
    "n_bend_nodes",
    *(c for a in ALPHA_COLUMNS for c in (a, f"{a}__status")),
    "rachis_curvature_deg",
    "rachis_curvature_deg__status",
    "rachis_bend_theta_deg",
    "rachis_bend_theta_deg__status",
    "chord_to_path_ratio",
    "chord_to_path_ratio__status",
    "blade_azimuth_quadrant",
    "blade_azimuth_quadrant__status",
    "frame",
)

#: Every ``unavailable`` cell of a snapshot with its blocking reason (Stage-2 criterion 11).
UNRESOLVED_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "vertex_id",
    "attr",
    "reason",
)

CHANGE_LOG_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "from_capture_id",
    "to_capture_id",
    "vertex_id",
    "attr",
    "old_value",
    "old_status",
    "new_value",
    "new_status",
    "rule_applied",
)

#: Stage-2 rule 6 diurnal guard: a change between two captures is written only for equal
#: photoperiod phases with both times-since-lights-on present.
COMPARISONS_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "from_capture_id",
    "to_capture_id",
    "vertex_id",
    "quantity",
    "comparison_status",
    "from_value",
    "to_value",
    "delta",
)

FRAME_TRANSFORMS_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "center_x_px",
    "center_y_px",
    "mm_per_px",
    "mm_per_px_sd",
    "scale_convention",
    "scale_reference_plane",
    "scale_reference_plane_offset_mm",
    "rim_homography_px_to_mm",
    "rotation_deg",
    "scale_status",
    "rotation_status",
    "method",
)
FRAME_TRANSFORMS_COLUMNS_REV6: Final[tuple[str, ...]] = tuple(
    c for c in FRAME_TRANSFORMS_COLUMNS if c not in ("scale_reference_plane", "scale_reference_plane_offset_mm"))

#: Each item a real-data audit could not produce, with the exact missing input.
#: ``camera_poses.csv`` (revision 6): camera pose per real top/side capture from the pot rim (Q2 INPUT, not Q2).
#: ``pose_status``: validated (two-axis self-check within k sigma of the click uncertainty) | approximate (published
#: only with this flag and its reason) | unavailable. ``camera_distance_mm`` = d to the rim centre (Y0: the rim-plane
#: bias of a leaf at height h is d / (d - h)). Matrices / vectors are JSON; rotation maps rim frame -> camera.
CAMERA_POSES_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "view",
    "image_sha256",
    "pose_status",
    "reason",
    "f_px",
    "f_basis",
    "f_px_width_basis",
    "f_px_diagonal_basis",
    "principal_point_px",
    "principal_point_source",
    "focal_from_rim_px",
    "focal_from_rim_sd_px",
    "self_check_ratio_residual",
    "self_check_ratio_sd",
    "self_check_orth_residual",
    "self_check_orth_sd",
    "rotation_rim_to_camera",
    "translation_mm",
    "camera_centre_rim_mm",
    "camera_distance_mm",
    "camera_distance_sd_mm",
    "tilt_deg",
    "source_ref",
    "evidence",
    "occluded_sides",
)
#: Revision 6 header of ``camera_poses.csv``, still accepted.
CAMERA_POSES_COLUMNS_REV6: Final[tuple[str, ...]] = CAMERA_POSES_COLUMNS[:-1]

AUDIT_GAPS_COLUMNS: Final[tuple[str, ...]] = (
    "plant_instance_id",
    "capture_id",
    "segment",
    "item",
    "run_status",
    "missing_input",
)

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

_INSTANCE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.\-]+(#[0-9]+)?$")
_COMPOUND_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.\-#]+/C[0-9]{4}$")
_UNSAFE_ID_CHARS: Final[str] = "\r\n\t,\"\\"


def is_unavailable(value: object) -> bool:
    return isinstance(value, str) and value == UNAVAILABLE


def fmt(value: float | int | None, digits: int = 6) -> str:
    """Format a number cell; ``None`` / NaN / inf become ``unavailable`` (never 0)."""
    if value is None:
        return UNAVAILABLE
    if isinstance(value, bool):
        raise P1SchemaError("a boolean is not a numeric cell")
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return UNAVAILABLE
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def fmt_triple(values: Sequence[float] | None, digits: int = 4) -> str:
    """JSON triple cell (``[x, y, z]``), as the V3 Stage-2 boundary reads ``base_xyz_mm``."""
    if values is None:
        return UNAVAILABLE
    if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
        raise P1SchemaError(f"not a finite 3-vector: {values!r}")
    return json.dumps([round(float(v), digits) for v in values])


def check_identifier(value: str, what: str) -> None:
    if not value or any(char in value for char in _UNSAFE_ID_CHARS):
        raise P1SchemaError(f"unsafe or empty {what}: {value!r}")


def instance_id(source_plant_id: str, index: int) -> str:
    """Plant-instance id. The first instance of a source plant keeps the plain id."""
    check_identifier(source_plant_id, "source_plant_id")
    if "#" in source_plant_id or "/" in source_plant_id:
        raise P1SchemaError(f"source_plant_id may not contain '#' or '/': {source_plant_id!r}")
    if index < 1:
        raise P1SchemaError("instance index starts at 1")
    return source_plant_id if index == 1 else f"{source_plant_id}#{index}"


def source_of_instance(plant_instance_id: str) -> str:
    return plant_instance_id.split("#", 1)[0]


def compound_id(plant_id: str, serial: int) -> str:
    """``<plant_id>/C<4-digit serial>`` - the phynotype2 track id = MTG CompoundLeaf vertex id."""
    if not 1 <= serial <= 9999:
        raise P1SchemaError(f"compound serial out of range: {serial}")
    return f"{plant_id}/C{serial:04d}"


def external_leaflet_id(plant_instance_id: str, vertex_id: str) -> str:
    """Canopy-wide leaflet id: ``<plant_instance_id>|<vertex_id>``; split with :func:`split_external_id`."""
    check_identifier(plant_instance_id, "plant_instance_id")
    check_identifier(vertex_id, "vertex_id")
    if "|" in plant_instance_id or "|" in vertex_id:
        raise P1SchemaError("'|' is reserved as the external-id separator")
    return f"{plant_instance_id}|{vertex_id}"


def split_external_id(external_id: str) -> tuple[str, str]:
    instance, sep, vertex = external_id.partition("|")
    if not sep or not instance or not vertex:
        raise P1SchemaError(f"not an external leaflet id: {external_id!r}")
    return instance, vertex


# --------------------------------------------------------------------------------------
# Validators. Each raises P1SchemaError naming the rule; none repairs anything.
# --------------------------------------------------------------------------------------

#: Column-name fragments of quantities that are on hold (MTG 8.3, 8.4) and may not be exported.
_ON_HOLD_FRAGMENTS: Final[tuple[str, ...]] = (
    "curl",
    "rugosity",
    "flat_threshold",
    "bent_threshold",
    "theta_threshold",
    "bend_type_threshold",
    "flat_bent",
)


def check_columns(rows: Sequence[Mapping[str, str]], columns: Sequence[str], table: str) -> None:
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise P1SchemaError(
                f"{table} row {index}: columns differ from the contract "
                f"(missing={sorted(expected - set(row))}, unexpected={sorted(set(row) - expected)})"
            )


def check_no_on_hold_fields(columns: Iterable[str], table: str) -> None:
    """Blade curl and flat/bent theta thresholds are on hold: no column may carry them."""
    for column in columns:
        low = column.lower()
        if any(fragment in low for fragment in _ON_HOLD_FRAGMENTS):
            raise P1SchemaError(f"{table}: column {column!r} is an on-hold quantity (MTG 8.3/8.4)")


def check_insertion_angle_prefix(columns: Iterable[str], table: str) -> None:
    """MTG 12.2: a flat export must class-prefix the two homonymous insertion angles."""
    for column in columns:
        if column in ("insertion_angle_deg", "insertion_angle_deg__status"):
            raise P1SchemaError(
                f"{table}: bare {column!r}; use compound_insertion_angle_deg / "
                "leaflet_insertion_angle_deg (MTG 12.2)"
            )


def check_status_cells(rows: Sequence[Mapping[str, str]], table: str) -> None:
    """Every ``__status`` cell is in the vocabulary; ``unavailable`` value <-> ``unavailable`` status;
    ``synthetic_truth`` only on synthetic-fixture rows."""
    for index, row in enumerate(rows):
        evidence = row.get("evidence", "")
        for column, status in row.items():
            if not column.endswith("__status"):
                continue
            if status not in VALUE_STATUSES:
                raise P1SchemaError(f"{table} row {index}: {column}={status!r} not in the vocabulary")
            if status in FIXTURE_ONLY_STATUSES and evidence != "synthetic_fixture":
                raise P1SchemaError(
                    f"{table} row {index}: {column}={status!r} on a row whose evidence is "
                    f"{evidence!r} - synthetic values can never pass as photo values"
                )
            value_column = column[: -len("__status")]
            if value_column not in row:
                continue
            value = row[value_column]
            if (status == UNAVAILABLE) != (value == UNAVAILABLE):
                raise P1SchemaError(
                    f"{table} row {index}: {value_column}={value!r} with status {status!r} "
                    "(an unavailable value and an unavailable status go together)"
                )
            if value == "":
                raise P1SchemaError(f"{table} row {index}: empty {value_column}; write 'unavailable'")


def check_frame_cells(rows: Sequence[Mapping[str, str]], table: str) -> None:
    """Every row names its frame; mm values only in an mm frame; px values only in ``image_px``.

    Area tables are exempt from the px rule (they carry both kinds on purpose) but not from the
    scale rule, see :func:`check_areas`.
    """
    for index, row in enumerate(rows):
        frame = row.get("frame")
        if frame not in FRAME_TOKENS:
            raise P1SchemaError(f"{table} row {index}: frame {frame!r} not in {FRAME_TOKENS}")
        for column, value in row.items():
            if value == UNAVAILABLE or column.endswith("__status"):
                continue
            is_mm = column.endswith(("_mm", "_mm2", "xyz_mm"))
            if is_mm and frame not in MM_FRAMES:
                raise P1SchemaError(
                    f"{table} row {index}: {column} holds a millimetre value in frame {frame!r}"
                )


def _header_for(rows: Sequence[Mapping[str, str]], current: Sequence[str], *legacy: Sequence[str]) -> Sequence[str]:
    """The current header, or an earlier revision's when a table was written before it (read-only use)."""
    if rows:
        for old in legacy:
            if set(rows[0]) == set(old):
                return old
    return current


def check_no_primes(rows: Sequence[Mapping[str, str]], table: str) -> None:
    """A4: prime characters are display-only; no stored cell may contain one."""
    for index, row in enumerate(rows):
        for column, value in row.items():
            if any(ch in value for ch in PRIME_CHARACTERS):
                raise P1SchemaError(f"{table} row {index}: {column}={value!r} stores a prime character (A4)")


def check_axis_code(value: str, where: str) -> None:
    if value != UNAVAILABLE and not AXIS_RE.match(value):
        raise P1SchemaError(f"{where}: axis {value!r} is not M, S<k> or a dotted path S<k>.<j>... (A4/A6)")


def _positive_int(value: str, what: str, where: str) -> int | None:
    """``unavailable`` -> None; otherwise an integer >= 1 (MTG A5 segment_index), else an error."""
    if value == UNAVAILABLE:
        return None
    try:
        number = int(value)
    except ValueError:
        raise P1SchemaError(f"{where}: {what} {value!r} is not an integer") from None
    if number < 1 or str(number) != value:
        raise P1SchemaError(f"{where}: {what} {value!r} must be an integer >= 1 (A5: 1-based)")
    return number


def display_label(rank: int, side_index: int) -> str:
    """Display-only label (A4): rank on M -> '1'; on S1 -> '1′'; on S2 -> '1″'; never written to a file."""
    primes = {0: "", 1: "\u2032", 2: "\u2033", 3: "\u2034"}
    return f"{rank}{primes.get(side_index, '(' + str(side_index) + ')')}"


def check_mtg_rows(rows: Sequence[Mapping[str, str]], table: str = "mtg_vertices.csv") -> None:
    header = _header_for(rows, MTG_VERTICES_COLUMNS, MTG_VERTICES_COLUMNS_REV8, MTG_VERTICES_COLUMNS_REV3,
                         MTG_VERTICES_COLUMNS_REV2)
    check_columns(rows, header, table)
    check_no_primes(rows, table)
    if header != MTG_VERTICES_COLUMNS_REV2:
        _check_axis_cells(rows, table)
    if header in (MTG_VERTICES_COLUMNS, MTG_VERTICES_COLUMNS_REV8):
        check_segments(rows, table)
    if header == MTG_VERTICES_COLUMNS:
        _check_node_index_cells(rows, table)
    check_status_cells(rows, table)
    check_frame_cells(rows, table)
    for index, row in enumerate(rows):
        check_identifier(row["vertex_id"], "vertex_id")
        if row["evidence"] not in EVIDENCE_KINDS:
            raise P1SchemaError(f"{table} row {index}: evidence {row['evidence']!r}")
        if row["candidate_id"] != UNAVAILABLE:
            raise P1SchemaError(
                f"{table} row {index}: candidate {row['candidate_id']!r} in the main snapshot "
                "(generated candidates live in a separate sub-MTG)"
            )
        bend_type = row["rachis_bend_type"]
        if bend_type != UNAVAILABLE:
            if bend_type not in RACHIS_BEND_TYPES:
                raise P1SchemaError(f"{table} row {index}: rachis_bend_type {bend_type!r}")
            if row["rachis_bend_type__status"] not in RACHIS_BEND_TYPE_STATUSES:
                raise P1SchemaError(
                    f"{table} row {index}: rachis_bend_type must be a human label "
                    f"(status {row['rachis_bend_type__status']!r}); thresholds are on hold"
                )
        if row["class"] == "CompoundLeaf" and row["frame"] == "image_px":
            for attr in ("compound_insertion_angle_deg", "elevation_deg", "base_xyz_mm"):
                if row[attr] != UNAVAILABLE:
                    raise P1SchemaError(
                        f"{table} row {index}: {attr} from a top image ({row['frame']}) - "
                        "a top view alone leaves it unavailable"
                    )


def _check_axis_cells(rows: Sequence[Mapping[str, str]], table: str) -> None:
    for index, row in enumerate(rows):
        where = f"{table} row {index}"
        check_axis_code(row["axis"], where)
        if row["order"] != UNAVAILABLE and not (row["order"].isdigit() and str(int(row["order"])) == row["order"]):
            raise P1SchemaError(f"{where}: order {row['order']!r} is not an integer >= 0 (A6: no order cap)")
        if row["class"] == "Axis" and row["axis"] != UNAVAILABLE and row["order"] != UNAVAILABLE:
            expected = str(axis_order(row["axis"]))
            if row["order"] != expected:
                raise P1SchemaError(f"{where}: axis {row['axis']} with order {row['order']}")
        if row["leaf_unfolded"] not in (UNAVAILABLE, "true", "false"):
            raise P1SchemaError(f"{where}: leaf_unfolded {row['leaf_unfolded']!r}")
        if row["class"] == "CompoundLeaf" and row["rank"] != UNAVAILABLE and row["leaf_unfolded"] != "true":
            raise P1SchemaError(
                f"{where}: rank {row['rank']} on a leaf not recorded as unfolded (A4: a leaf gets its rank when "
                "it has unfolded and is separately visible)")


def check_rank_node(rank: str, node: str, shared_first: bool, where: str, what: str = "rank") -> None:
    """A7 / revision 9: a node index must be the node whose U5 rank is ``rank`` (main stem: nodes 1 and 2 -> rank 1,
    node i -> i - 1; side axes while the tie rule is open: node = rank). A node index without its rank is refused."""
    from tomato3d.state import ranks as rk
    if node == UNAVAILABLE:
        return
    if rank == UNAVAILABLE:
        raise P1SchemaError(f"{where}: node index {node!r} without a {what} (revision 9: unavailable whenever the rank is)")
    n = _rank_value(node, "node index", where); r = _rank_value(rank, what, where)
    if n < 1:
        raise P1SchemaError(f"{where}: node index {n} < 1 (node 0 is the cotyledon node)")
    if not rk.rank_node_consistent(r, n, shared_first):
        raise P1SchemaError(f"{where}: node index {n} is not the node of {what} {r} ({'shared first rank' if shared_first else 'one rank per node'}, A7)")


def _check_node_index_cells(rows: Sequence[Mapping[str, str]], table: str) -> None:
    for index, row in enumerate(rows):
        where = f"{table} row {index}"
        for attr in ("node_index_from_base", "parent_node_index", "above_node_index"):
            if (row[attr] == UNAVAILABLE) != (row[f"{attr}__status"] == UNAVAILABLE):
                raise P1SchemaError(f"{where}: {attr} and its status disagree")
        if row["class"] == "CompoundLeaf":
            check_rank_node(row["rank"], row["node_index_from_base"], row["axis"] == "M", where)
            if row["parent_node_index"] != UNAVAILABLE or row["above_node_index"] != UNAVAILABLE:
                raise P1SchemaError(f"{where}: a leaf carries node_index_from_base only")
        elif row["class"] == "Axis":
            if row["node_index_from_base"] != UNAVAILABLE or row["above_node_index"] != UNAVAILABLE:
                raise P1SchemaError(f"{where}: an axis carries parent_node_index only")
            if row["parent_node_index"] != UNAVAILABLE:
                parent = axis_parent(row["axis"]) if row["axis"] not in (UNAVAILABLE, "M") else None
                check_rank_node(row["parent_rank"], row["parent_node_index"], parent == "M", where, "parent_rank")
        elif row["class"] == TRUSS_CLASS:
            if row["node_index_from_base"] != UNAVAILABLE or row["parent_node_index"] != UNAVAILABLE:
                raise P1SchemaError(f"{where}: a truss carries above_node_index only")
            check_rank_node(row["parent_rank"], row["above_node_index"], row["axis"] == "M", where, "above_rank (parent_rank)")
        elif row["node_index_from_base"] != UNAVAILABLE or row["parent_node_index"] != UNAVAILABLE or row["above_node_index"] != UNAVAILABLE:
            raise P1SchemaError(f"{where}: node index on a {row['class']} vertex")


def _rank_value(value: str, what: str, where: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise P1SchemaError(f"{where}: {what} {value!r} is not an integer") from None


def check_segments(rows: Sequence[Mapping[str, str]], table: str = "mtg_vertices.csv") -> None:
    """MTG A5 on one table of snapshots: ``segment_index`` is 1-based and counts the trusses below a node.

    Per (plant instance, snapshot, axis), nodes are ordered along the axis by position: a CompoundLeaf of rank r
    sits at (r, 0), a Truss whose ``parent_rank`` (= registry above_rank) is a sits at (a, 1), directly above
    leaf a. Every node with a segment_index must have ``1 + (trusses below it)``: the index starts at 1, never
    decreases along the axis, and the node after a truss is one higher than the truss (the truss closes its
    segment). A Truss carries no leaf rank and is a scale-3 vertex; a leaflet repeats its compound's index; an
    Axis row carries its apical segment = 1 + its trusses."""
    groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    by_vertex: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for index, row in enumerate(rows):
        where = f"{table} row {index}"
        _positive_int(row["segment_index"], "segment_index", where)
        if (row["segment_index"] == UNAVAILABLE) != (row["segment_index__status"] == UNAVAILABLE):
            raise P1SchemaError(f"{where}: segment_index and its status disagree")
        if row["class"] == TRUSS_CLASS:
            if row["rank"] != UNAVAILABLE:
                raise P1SchemaError(f"{where}: a truss carries no leaf rank (A5)")
            if row["scale"] != "3":
                raise P1SchemaError(f"{where}: a truss is a scale-3 vertex beside CompoundLeaf (A5)")
            if row["axis"] == UNAVAILABLE:
                raise P1SchemaError(f"{where}: a truss is recorded on a stated axis (A5)")
            if row["leaf_unfolded"] != UNAVAILABLE:
                raise P1SchemaError(f"{where}: a truss bears no leaves (A5)")
        if row["segment_index"] != UNAVAILABLE and row["axis"] == UNAVAILABLE:
            raise P1SchemaError(f"{where}: segment_index without an axis (A5: counted per axis)")
        key = (row["plant_instance_id"], row["snapshot_capture_id"], row["axis"])
        groups.setdefault(key, []).append(row)
        by_vertex[(row["plant_instance_id"], row["snapshot_capture_id"], row["vertex_id"])] = row
    for (pid, snap, axis), members in groups.items():
        if axis == UNAVAILABLE:
            continue
        trusses = [r for r in members if r["class"] == TRUSS_CLASS]
        placed = [_rank_value(r["parent_rank"], "truss parent_rank", f"{table} {r['vertex_id']}")
                  for r in trusses if r["parent_rank"] != UNAVAILABLE]
        any_unplaced = len(placed) != len(trusses)
        for r in members:
            where = f"{table} {r['vertex_id']} @ {snap}"
            seg = r["segment_index"]
            if seg == UNAVAILABLE:
                continue
            if r["class"] == "Axis":
                if int(seg) != 1 + len(trusses):
                    raise P1SchemaError(f"{where}: axis apical segment_index {seg} != 1 + {len(trusses)} trusses")
                continue
            if r["class"] == "CompoundLeaf":
                if r["rank"] == UNAVAILABLE:
                    raise P1SchemaError(f"{where}: segment_index on a leaf without rank (A5 derives it from ranks)")
                pos = (_rank_value(r["rank"], "rank", where), 0)
            elif r["class"] == TRUSS_CLASS:
                if r["parent_rank"] == UNAVAILABLE:
                    raise P1SchemaError(f"{where}: segment_index on a truss without a position")
                pos = (_rank_value(r["parent_rank"], "truss parent_rank", where), 1)
            else:
                parent = by_vertex.get((pid, snap, r["complex_id"]))
                if parent is None or parent["segment_index"] != seg:
                    raise P1SchemaError(f"{where}: segment_index {seg} differs from its complex {r['complex_id']}")
                continue
            if any_unplaced:
                raise P1SchemaError(f"{where}: segment_index while a truss on axis {axis} has no position")
            below = sum(1 for a in placed if (a, 1) < pos)
            if int(seg) != 1 + below:
                raise P1SchemaError(
                    f"{where}: segment_index {seg} but {below} truss(es) below it on {axis} (A5: 1 + trusses below; "
                    "starts at 1, increments at each truss, never decreases)")
        if len(set(placed)) != len(placed):
            raise P1SchemaError(f"{pid} {snap} {axis}: two trusses above the same leaf rank")


def check_trusses(rows: Sequence[Mapping[str, str]]) -> None:
    """``trusses.csv`` (MTG A5)."""
    table = "trusses.csv"
    header = _header_for(rows, TRUSSES_COLUMNS, TRUSSES_COLUMNS_REV8)
    check_columns(rows, header, table)
    check_no_primes(rows, table)
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        where = f"{table} row {index}"
        key = (row["plant_instance_id"], row["capture_id"], row["truss_vertex_id"])
        if key in seen:
            raise P1SchemaError(f"{where}: duplicate truss row {key}")
        seen.add(key)
        check_identifier(row["truss_vertex_id"], "truss_vertex_id")
        check_axis_code(row["axis_id"], where)
        if row["axis_id"] == UNAVAILABLE:
            raise P1SchemaError(f"{where}: a truss is recorded on a stated axis")
        _positive_int(row["segment_index"], "segment_index", where)
        for attr in ("parent_rank", "segment_index"):
            if (row[attr] == UNAVAILABLE) != (row[f"{attr}__status"] == UNAVAILABLE):
                raise P1SchemaError(f"{where}: {attr} and its status disagree")
            if row[f"{attr}__status"] not in VALUE_STATUSES:
                raise P1SchemaError(f"{where}: {attr}__status {row[f'{attr}__status']!r}")
            if row[f"{attr}__status"] in FIXTURE_ONLY_STATUSES and row["evidence"] != "synthetic_fixture":
                raise P1SchemaError(f"{where}: synthetic {attr} on a real row")
        if row["registry_ref"] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{where}: a truss needs its registry record (A5: trusses are human-recorded)")
        if header == TRUSSES_COLUMNS:
            if (row["above_node_index"] == UNAVAILABLE) != (row["above_node_index__status"] == UNAVAILABLE):
                raise P1SchemaError(f"{where}: above_node_index and its status disagree")
            if row["above_node_index__status"] not in VALUE_STATUSES:
                raise P1SchemaError(f"{where}: above_node_index__status {row['above_node_index__status']!r}")
            check_rank_node(row["parent_rank"], row["above_node_index"], row["axis_id"] == "M", where, "above_rank (parent_rank)")


def check_axes(rows: Sequence[Mapping[str, str]]) -> None:
    """``axes.csv``: A4 + fsp-axes-1 (+ A5 segment columns). An order-1 axis needs a registration record."""
    table = "axes.csv"
    header = _header_for(rows, AXES_COLUMNS, AXES_COLUMNS_REV8, AXES_COLUMNS_REV7, AXES_COLUMNS_REV3)
    check_columns(rows, header, table)
    check_no_primes(rows, table)
    if header == AXES_COLUMNS:
        for index, row in enumerate(rows):
            where = f"{table} row {index}"
            if (row["parent_node_index"] == UNAVAILABLE) != (row["parent_node_index__status"] == UNAVAILABLE):
                raise P1SchemaError(f"{where}: parent_node_index and its status disagree")
            if row["parent_node_index__status"] not in VALUE_STATUSES:
                raise P1SchemaError(f"{where}: parent_node_index__status {row['parent_node_index__status']!r}")
            if row["axis"] == "M" and row["parent_node_index"] != UNAVAILABLE:
                raise P1SchemaError(f"{where}: the main axis has no parent node")
            if row["axis"] != "M":
                check_rank_node(row["parent_rank"], row["parent_node_index"], row["parent_axis_id"] == "M", where, "parent_rank")
    if header in (AXES_COLUMNS, AXES_COLUMNS_REV8):
        for index, row in enumerate(rows):
            where = f"{table} row {index}"
            if row["origin"] not in AXIS_ORIGINS:
                raise P1SchemaError(f"{where}: origin {row['origin']!r} not in {AXIS_ORIGINS}")
            if row["axis"] == "M" and row["origin"] != UNAVAILABLE:
                raise P1SchemaError(f"{where}: the main axis has no origin")
            counts = {}
            for col in ("node_count_incl_scars", "leaf_present_count"):
                if (row[col] == UNAVAILABLE) != (row[f"{col}__status"] == UNAVAILABLE):
                    raise P1SchemaError(f"{where}: {col} and its status disagree")
                if row[col] != UNAVAILABLE:
                    counts[col] = _rank_value(row[col], col, where)
                    if counts[col] < 0:
                        raise P1SchemaError(f"{where}: {col} < 0")
            if len(counts) == 2 and counts["leaf_present_count"] > counts["node_count_incl_scars"]:
                raise P1SchemaError(f"{where}: more leaves present than nodes formed (scars count as nodes, U5)")
    if header in (AXES_COLUMNS, AXES_COLUMNS_REV8, AXES_COLUMNS_REV7):
        for index, row in enumerate(rows):
            where = f"{table} row {index}"
            seg = _positive_int(row["segment_index"], "segment_index", where)
            if (row["segment_index"] == UNAVAILABLE) != (row["segment_index__status"] == UNAVAILABLE):
                raise P1SchemaError(f"{where}: segment_index and its status disagree")
            if row["truss_count"] != UNAVAILABLE:
                count = _rank_value(row["truss_count"], "truss_count", where)
                if count < 0 or (seg is not None and seg != count + 1):
                    raise P1SchemaError(f"{where}: apical segment_index {seg} != truss_count {count} + 1 (A5)")
            elif seg is not None:
                raise P1SchemaError(f"{where}: segment_index without truss_count")
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        where = f"{table} row {index}"
        key = (row["plant_instance_id"], row["capture_id"], row["axis_id"])
        if key in seen:
            raise P1SchemaError(f"{where}: duplicate axis row {key}")
        seen.add(key)
        if row["axis_id"] != row["axis"] or row["stage"] != row["axis_state"]:
            raise P1SchemaError(f"{where}: axis_id/axis or stage/axis_state disagree")
        check_axis_code(row["axis"], where)
        if row["axis"] == UNAVAILABLE:
            raise P1SchemaError(f"{where}: axis {row['axis']!r}")
        if row["axis_state"] not in AXIS_STATES:
            raise P1SchemaError(f"{where}: axis_state {row['axis_state']!r}")
        if row["order"] != str(axis_order(row["axis"])):
            raise P1SchemaError(f"{where}: order {row['order']!r} != {axis_order(row['axis'])}, the depth of axis "
                                f"{row['axis']} ({ORDER2_RESERVED})")
        if row["axis"] == "M":
            if row["axis_state"] != "main" or row["parent_rank"] != MAIN_AXIS_PARENT_RANK:
                raise P1SchemaError(f"{where}: the main axis is order 0, state main, parent_rank 0 (fsp-axes-1)")
            continue
        if row["axis_state"] == "main":
            raise P1SchemaError(f"{where}: a registered axis is axis_shoot|substem")
        if row["registration_ref"] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{where}: axis {row['axis']} without a registration record (A4)")
        if row["parent_axis_id"] != axis_parent(row["axis"]):
            raise P1SchemaError(f"{where}: axis {row['axis']} attaches to {axis_parent(row['axis'])}, not "
                                f"{row['parent_axis_id']!r} ({ORDER2_RESERVED})")
        if (row["parent_rank"] == UNAVAILABLE) != (row["parent_rank__status"] == UNAVAILABLE):
            raise P1SchemaError(f"{where}: parent_rank and its status disagree")
        if row["parent_rank__status"] not in VALUE_STATUSES:
            raise P1SchemaError(f"{where}: parent_rank__status {row['parent_rank__status']!r}")
        if row["parent_rank__status"] in FIXTURE_ONLY_STATUSES and row["evidence"] != "synthetic_fixture":
            raise P1SchemaError(f"{where}: synthetic parent_rank on a real row")
        leaves = [x for x in row["registration_leaf_ids"].split(";") if x and x != UNAVAILABLE]
        if leaves and len(leaves) != 2:
            raise P1SchemaError(f"{where}: registration needs the two leaves that appeared together (A4)")


def check_axes_coverage(axes: Sequence[Mapping[str, str]], *tables: Sequence[Mapping[str, str]]) -> None:
    """Every S-axis used by an MTG or identity row has a registration row in ``axes.csv``."""
    registered = {(r["plant_instance_id"], r["axis"]) for r in axes}
    for table in tables:
        for row in table:
            axis = row.get("axis", UNAVAILABLE)
            if axis.startswith("S") and (row["plant_instance_id"], axis) not in registered:
                raise P1SchemaError(f"axis {axis} of {row['plant_instance_id']} used without an axes.csv registration")


def check_candidate_rows(rows: Sequence[Mapping[str, str]], table: str) -> None:
    """Candidate sub-MTG: every row has a candidate id and every value is generated_candidate."""
    check_columns(rows, _header_for(rows, MTG_VERTICES_COLUMNS, MTG_VERTICES_COLUMNS_REV8, MTG_VERTICES_COLUMNS_REV3,
                                    MTG_VERTICES_COLUMNS_REV2), table)
    for index, row in enumerate(rows):
        if row["candidate_id"] == UNAVAILABLE:
            raise P1SchemaError(f"{table} row {index}: candidate row without candidate_id")
        for column, status in row.items():
            if column.endswith("__status") and status not in ("generated_candidate", UNAVAILABLE):
                raise P1SchemaError(
                    f"{table} row {index}: {column}={status!r} in a candidate sub-MTG"
                )


def check_areas(rows: Sequence[Mapping[str, str]], scale_by_capture: Mapping[str, str],
                plane_by_capture: Mapping[str, str] | None = None) -> None:
    """Area rules: full_outline basis, mm2 only with a scale, kinds never merged. With ``plane_by_capture``
    (capture -> stated reference plane, revision 7), an mm2 value whose capture states no plane is refused."""
    table = "areas.csv"
    header = _header_for(rows, AREAS_COLUMNS, AREAS_COLUMNS_REV4)
    check_columns(rows, header, table)
    check_status_cells(rows, table)
    for index, row in enumerate(rows):
        if header == AREAS_COLUMNS:
            for column in ("visible_projected_area_mm2", "completed_projected_area_mm2"):
                sd = row[f"{column}_sd"]
                if sd != UNAVAILABLE:
                    _non_negative(sd, f"{column}_sd", f"{table} row {index}")
                if row["evidence"] == "real_photo" and row[column] != UNAVAILABLE and sd == UNAVAILABLE:
                    raise P1SchemaError(f"{table} row {index}: real-photo {column} without its uncertainty (T6)")
        if row["area_basis"] != AREA_BASIS_MAIN:
            raise P1SchemaError(
                f"{table} row {index}: area_basis {row['area_basis']!r}; the main columns are "
                f"{AREA_BASIS_MAIN} (MTG 13); green-only goes to area_green_only_mm2"
            )
        scale_status = scale_by_capture.get(row["capture_id"], UNAVAILABLE)
        for column in (
            "visible_projected_area_mm2",
            "completed_projected_area_mm2",
        ):
            if row[column] != UNAVAILABLE and (
                scale_status != "solved" or row["scale_source"] == UNAVAILABLE
            ):
                raise P1SchemaError(
                    f"{table} row {index}: {column} without a scale source "
                    f"(capture {row['capture_id']} scale_status={scale_status})"
                )
            if row[column] != UNAVAILABLE and plane_by_capture is not None and \
                    plane_by_capture.get(row["capture_id"], UNAVAILABLE) in ("", UNAVAILABLE):
                raise P1SchemaError(
                    f"{table} row {index}: {column} whose capture {row['capture_id']} states no reference plane "
                    "(a millimetre value without its plane is refused, revision 7)")
        if row["one_sided_lamina_area_mm2"] != UNAVAILABLE and row[
            "lamina_geometric_evidence"
        ] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{table} row {index}: lamina surface without geometric evidence")
        if row["completed_projected_area_px2"] != UNAVAILABLE and row["completion_method"] in (
            "",
            UNAVAILABLE,
        ):
            raise P1SchemaError(f"{table} row {index}: completed area without a method")


def parse_capture_tz(name: str):
    """IANA zone or ``UTC+HH:MM`` / ``UTC-HH:MM`` -> tzinfo. Anything else raises (no default zone)."""
    import re as _re
    from datetime import timedelta, timezone
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    if not name or name == UNAVAILABLE:
        raise P1SchemaError("capture timezone is unavailable; it is never assumed")
    m = _re.fullmatch(r"UTC([+-])(\d{2}):(\d{2})", name)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours, minutes = int(m.group(2)), int(m.group(3))
        if hours > 14 or minutes >= 60:
            raise P1SchemaError(f"invalid fixed offset {name!r}")
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    if "/" not in name:
        raise P1SchemaError(f"capture timezone {name!r} is not an IANA name (Area/City) or UTC+HH:MM")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise P1SchemaError(f"unknown capture timezone {name!r}") from exc


def localize_capture_time(naive_iso: str, tz_name: str) -> str:
    """Attach a stated timezone to a naive device-clock time. Ambiguous or non-existent local times
    (DST transitions) raise - the offset would be a guess."""
    from datetime import datetime, timezone

    naive = datetime.fromisoformat(naive_iso)
    if naive.tzinfo is not None:
        raise P1SchemaError(f"{naive_iso} already carries an offset")
    tz = parse_capture_tz(tz_name)
    early, late = naive.replace(tzinfo=tz, fold=0), naive.replace(tzinfo=tz, fold=1)
    if early.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) != naive:
        raise P1SchemaError(f"{naive_iso} does not exist in {tz_name} (DST gap)")
    if early.utcoffset() != late.utcoffset():
        raise P1SchemaError(f"{naive_iso} is ambiguous in {tz_name} (DST fold)")
    return early.isoformat()


def _check_capture_tz(row: Mapping[str, str], index: int, table: str) -> None:
    from datetime import datetime

    tz_name, source = row["capture_tz"], row["capture_tz_source"]
    if source not in CAPTURE_TZ_SOURCES:
        raise P1SchemaError(f"{table} row {index}: capture_tz_source {source!r}")
    if (tz_name == UNAVAILABLE) != (source == UNAVAILABLE):
        raise P1SchemaError(f"{table} row {index}: capture_tz and capture_tz_source disagree")
    if row["capture_time"] == UNAVAILABLE:
        return
    stamp = datetime.fromisoformat(row["capture_time"])
    if tz_name == UNAVAILABLE:
        if stamp.tzinfo is not None and row["evidence"] == "real_photo":
            raise P1SchemaError(
                f"{table} row {index}: real capture time carries an offset but no capture_tz was stated "
                "(no implicit offset)")
        return
    if stamp.tzinfo is None:
        raise P1SchemaError(f"{table} row {index}: capture_tz {tz_name} stated but capture_time is naive")
    expected = datetime.fromisoformat(localize_capture_time(stamp.replace(tzinfo=None).isoformat(), tz_name))
    if stamp.utcoffset() != expected.utcoffset():
        raise P1SchemaError(f"{table} row {index}: capture_time offset disagrees with capture_tz {tz_name}")


def check_captures(rows: Sequence[Mapping[str, str]]) -> None:
    table = "captures.csv"
    header = _header_for(rows, CAPTURES_COLUMNS, CAPTURES_COLUMNS_REV6, CAPTURES_COLUMNS_REV4, CAPTURES_COLUMNS_REV1)
    revision1 = header == CAPTURES_COLUMNS_REV1
    check_columns(rows, header, table)
    check_status_cells(rows, table)
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        check_identifier(row["capture_id"], "capture_id")
        key = (row["plant_instance_id"], row["capture_id"])
        if key in seen:
            raise P1SchemaError(f"{table}: duplicate capture {key}")
        seen.add(key)
        if row["evidence"] not in EVIDENCE_KINDS:
            raise P1SchemaError(f"{table} row {index}: evidence {row['evidence']!r}")
        if row["frame"] not in FRAME_TOKENS:
            raise P1SchemaError(f"{table} row {index}: frame {row['frame']!r}")
        if row["view"] not in VIEWS:
            raise P1SchemaError(f"{table} row {index}: view {row['view']!r}")
        if row["photoperiod_phase"] not in PHOTOPERIOD_PHASES:
            raise P1SchemaError(f"{table} row {index}: photoperiod_phase {row['photoperiod_phase']!r}")
        if row["scale_status"] not in SCALE_STATUSES:
            raise P1SchemaError(f"{table} row {index}: scale_status {row['scale_status']!r}")
        if row["scale_status"] == "solved" and row["scale_source"] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{table} row {index}: solved scale without a scale_source")
        if (row["mm_per_px"] != UNAVAILABLE) != (row["scale_status"] == "solved"):
            raise P1SchemaError(f"{table} row {index}: mm_per_px vs scale_status disagree")
        if row["evidence"] == "real_photo" and row["image_sha256"] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{table} row {index}: real photo without its sha256")
        if row["source_plant_id"] != source_of_instance(row["plant_instance_id"]):
            raise P1SchemaError(f"{table} row {index}: plant_instance_id / source_plant_id disagree")
        if not revision1:
            _check_capture_tz(row, index, table)
        if header in (CAPTURES_COLUMNS, CAPTURES_COLUMNS_REV6):
            _check_capture_scale(row, index, table)
        if header == CAPTURES_COLUMNS:
            _check_reference_plane(row, f"{table} row {index}")
        elif header in (CAPTURES_COLUMNS_REV4, CAPTURES_COLUMNS_REV1) and row["scale_status"] == "solved" \
                and row["evidence"] == "real_photo":
            raise P1SchemaError(f"{table} row {index}: real-photo mm scale in a header without any plane column - a "
                                "millimetre value without its reference plane is refused (revision 7)")


def _check_reference_plane(row: Mapping[str, str], where: str) -> None:
    """Revision 7: a solved scale names its reference and the plane that reference lies on, with the plane's offset
    from the rim; the plane must match the convention."""
    plane, offset = row["scale_reference_plane"], row["scale_reference_plane_offset_mm"]
    if plane not in SCALE_REFERENCE_PLANES:
        raise P1SchemaError(f"{where}: scale_reference_plane {plane!r} not in {SCALE_REFERENCE_PLANES}")
    if row["scale_status"] != "solved":
        if plane != UNAVAILABLE or offset != UNAVAILABLE:
            raise P1SchemaError(f"{where}: reference plane on an unsolved scale")
        return
    if plane == UNAVAILABLE or row["scale_reference"] in ("", UNAVAILABLE):
        raise P1SchemaError(f"{where}: solved mm scale without scale_reference / scale_reference_plane - a millimetre "
                            "value whose reference plane is unstated is refused (it is wrong by the plane distance)")
    expected = CONVENTION_REFERENCE_PLANE.get(row["scale_convention"])
    if expected is not None and plane != expected:
        raise P1SchemaError(f"{where}: convention {row['scale_convention']} refers to plane {expected}, not {plane}")
    if plane == "synthetic_camera":
        if offset != UNAVAILABLE:
            raise P1SchemaError(f"{where}: a declared synthetic camera has no rim offset")
        return
    try:
        value = float(offset)
    except ValueError:
        raise P1SchemaError(f"{where}: scale_reference_plane_offset_mm {offset!r} is not a number") from None
    if plane == "pot_rim" and value != 0.0:
        raise P1SchemaError(f"{where}: the pot_rim plane has offset 0, not {offset}")
    if plane == "medium_surface" and not (math.isfinite(value) and value < 0):
        raise P1SchemaError(f"{where}: the medium surface lies below the rim (negative offset), not {offset}")


_LEGACY_PLANE: Final[dict[str, str]] = {"rim_plane": "pot_rim", "pot_axis_vertical": "pot_rim",
                                          "synthetic_camera": "synthetic_camera"}


def reference_plane_of(capture: Mapping[str, str]) -> str:
    """The stated reference plane of a capture's scale: the revision-7 column, else the revision-5/6 ``scale_plane``,
    else (a pre-T6 fixture) the declared synthetic camera; a real photo without any plane column -> unavailable."""
    if "scale_reference_plane" in capture:
        return capture["scale_reference_plane"]
    if capture.get("scale_plane", UNAVAILABLE) in _LEGACY_PLANE:
        return _LEGACY_PLANE[capture["scale_plane"]]
    if capture.get("evidence") == "synthetic_fixture" and capture.get("scale_status") == "solved":
        return "synthetic_camera"
    return UNAVAILABLE


def check_frame_transforms(rows: Sequence[Mapping[str, str]]) -> None:
    """Revision 7: a solved frame (mm per px / homography to a plane) names that plane; older headers accepted."""
    table = "frame_transforms.csv"
    header = _header_for(rows, FRAME_TRANSFORMS_COLUMNS, FRAME_TRANSFORMS_COLUMNS_REV6)
    check_columns(rows, header, table)
    if header != FRAME_TRANSFORMS_COLUMNS:
        return
    for index, row in enumerate(rows):
        where = f"{table} row {index}"
        if row["scale_status"] == "solved" and row["scale_reference_plane"] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{where}: solved scale / homography without its reference plane (revision 7)")
        if row["scale_reference_plane"] not in SCALE_REFERENCE_PLANES:
            raise P1SchemaError(f"{where}: scale_reference_plane {row['scale_reference_plane']!r}")


def _non_negative(value: str, what: str, where: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise P1SchemaError(f"{where}: {what} {value!r} is not a number") from None
    if not math.isfinite(number) or number < 0:
        raise P1SchemaError(f"{where}: {what} {value!r} must be >= 0")
    return number


def _check_capture_scale(row: Mapping[str, str], index: int, table: str) -> None:
    """T6: every mm scale names its convention; a real photo's scale is per pot (its own rim), on the rim plane,
    with an uncertainty; a cross-check that stopped leaves the scale unavailable."""
    where = f"{table} row {index}"
    for column, vocab in (("scale_convention", SCALE_CONVENTIONS), ("scale_cross_check", SCALE_CROSS_CHECKS),
                          ("scale_plane", SCALE_PLANES)):
        if row[column] not in vocab:
            raise P1SchemaError(f"{where}: {column} {row[column]!r}")
    if row["scale_sources"] != UNAVAILABLE:
        try:
            sources = int(row["scale_sources"])
        except ValueError:
            raise P1SchemaError(f"{where}: scale_sources {row['scale_sources']!r}") from None
        if sources < 0:
            raise P1SchemaError(f"{where}: scale_sources {sources}")
        if row["scale_cross_check"] == "none_single_source" and sources != 1:
            raise P1SchemaError(f"{where}: none_single_source with scale_sources {sources}")
        if row["scale_cross_check"] in ("accepted_within_2pct", "flagged_2_to_5pct", "stopped_over_5pct") \
                and sources < 2:
            raise P1SchemaError(f"{where}: a cross-check needs two scale sources")
    if row["mm_per_px_sd"] != UNAVAILABLE:
        _non_negative(row["mm_per_px_sd"], "mm_per_px_sd", where)
    if row["scale_cross_check"] == "stopped_over_5pct" and row["scale_status"] == "solved":
        raise P1SchemaError(f"{where}: scale sources disagree by more than 5 % - the scale may not be solved (T6)")
    if row["scale_status"] != "solved" or row["evidence"] != "real_photo":
        return
    if row["scale_convention"] not in REAL_SCALE_CONVENTIONS:
        raise P1SchemaError(f"{where}: a real-photo mm scale must name its pot-rim convention (T6), got "
                            f"{row['scale_convention']!r}")
    if row["mm_per_px_sd"] == UNAVAILABLE:
        raise P1SchemaError(f"{where}: real-photo scale without its uncertainty (T6: never an mm value without it)")
    if row["scale_plane"] != CONVENTION_PLANE[row["scale_convention"]]:
        raise P1SchemaError(f"{where}: convention {row['scale_convention']} is a "
                            f"{CONVENTION_PLANE[row['scale_convention']]} scale, not {row['scale_plane']!r} "
                            "(a pot-rim scale is a rim-plane scale; plant_local_mm caveat)")
    if not row["scale_source"].startswith("pot_rim:"):
        raise P1SchemaError(f"{where}: real-photo scale_source {row['scale_source']!r} is not a per-pot rim "
                            "derivation (an image-wide scale is rejected, T6)")


def check_camera_poses(rows: Sequence[Mapping[str, str]],
                       captures: Sequence[Mapping[str, str]] | None = None) -> None:
    """``camera_poses.csv``: an approximate pose is published only with its flag and reason; a validated pose carries
    its self-check numbers; a side capture's vertical scale requires a validated pose."""
    table = "camera_poses.csv"
    check_columns(rows, _header_for(rows, CAMERA_POSES_COLUMNS, CAMERA_POSES_COLUMNS_REV6), table)
    status_of: dict[tuple[str, str], str] = {}
    for index, row in enumerate(rows):
        where = f"{table} row {index}"
        if row["pose_status"] not in POSE_STATUSES:
            raise P1SchemaError(f"{where}: pose_status {row['pose_status']!r}")
        if row["pose_status"] != "validated" and row["reason"] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{where}: a {row['pose_status']} pose needs its reason")
        three_edge = row.get("occluded_sides", UNAVAILABLE) not in ("", UNAVAILABLE)
        if three_edge and row["pose_status"] == "validated":
            raise P1SchemaError(f"{where}: a pose from three edges (side {row['occluded_sides']} occluded) has no "
                                "redundancy for the self-check and cannot be validated")
        if row["pose_status"] != UNAVAILABLE:
            required = ("f_px", "camera_distance_mm", "camera_distance_sd_mm") + (() if three_edge else (
                "self_check_ratio_residual", "self_check_ratio_sd", "self_check_orth_residual", "self_check_orth_sd"))
            for col in required:
                if row[col] == UNAVAILABLE:
                    raise P1SchemaError(f"{where}: {row['pose_status']} pose without {col}")
        if row["pose_status"] == "validated":
            for res, sd in (("self_check_ratio_residual", "self_check_ratio_sd"),
                            ("self_check_orth_residual", "self_check_orth_sd")):
                if not math.isfinite(float(row[res])) or float(row[sd]) <= 0:
                    raise P1SchemaError(f"{where}: validated pose with an invalid {res}/{sd}")
        status_of[(row["plant_instance_id"], row["capture_id"])] = row["pose_status"]
    for c in captures or ():
        if c.get("scale_convention") == "pot_rim_pose_vertical_axis" and c.get("scale_status") == "solved" and \
                status_of.get((c["plant_instance_id"], c["capture_id"])) != "validated":
            raise P1SchemaError(f"captures.csv {c['capture_id']}: a pose-derived vertical scale needs a validated pose")


def check_observations(rows: Sequence[Mapping[str, str]], evidence: str) -> None:
    table = "observations.csv"
    header = _header_for(rows, OBSERVATIONS_COLUMNS, OBSERVATIONS_COLUMNS_REV2)
    check_columns(rows, header, table)
    check_no_primes(rows, table)
    for index, row in enumerate(rows):
        if header == OBSERVATIONS_COLUMNS:
            check_axis_code(row["axis_hint"], f"{table} row {index}")
            if row["axis_hint_source"] not in AXIS_HINT_SOURCES:
                raise P1SchemaError(f"{table} row {index}: axis_hint_source {row['axis_hint_source']!r}")
            if (row["axis_hint"] == UNAVAILABLE) != (row["axis_hint_source"] == UNAVAILABLE):
                raise P1SchemaError(f"{table} row {index}: axis_hint and its source disagree")
        if row["provenance"] not in OBSERVATION_PROVENANCE:
            raise P1SchemaError(f"{table} row {index}: provenance {row['provenance']!r}")
        if row["provenance"] == "synthetic_gt" and row["evidence"] != "synthetic_fixture":
            raise P1SchemaError(f"{table} row {index}: synthetic_gt on a non-fixture row")
        if row["evidence"] != evidence:
            raise P1SchemaError(f"{table} row {index}: evidence {row['evidence']!r} != {evidence!r}")
        if row["provenance"] == "detector_proposal" and row["review_state"] != "unreviewed":
            raise P1SchemaError(
                f"{table} row {index}: a detector proposal is never accepted without review"
            )
        if row["detector_status"] not in DETECTOR_STATUSES:
            raise P1SchemaError(f"{table} row {index}: detector_status {row['detector_status']!r}")
        if row["review_state"] not in REVIEW_STATES:
            raise P1SchemaError(f"{table} row {index}: review_state {row['review_state']!r}")
        if row["obs_class"] not in OBS_CLASSES:
            raise P1SchemaError(f"{table} row {index}: obs_class {row['obs_class']!r}")
        if row["frame"] != "image_px":
            raise P1SchemaError(f"{table} row {index}: observations are in image_px")


def check_identity_rows(rows: Sequence[Mapping[str, str]]) -> None:
    table = "identity_manifest.csv"
    header = _header_for(rows, IDENTITY_COLUMNS, IDENTITY_COLUMNS_REV3, IDENTITY_COLUMNS_REV2)
    check_columns(rows, header, table)
    check_no_primes(rows, table)
    for index, row in enumerate(rows):
        if header != IDENTITY_COLUMNS_REV2:
            check_axis_code(row["axis"], f"{table} row {index}")
        if header == IDENTITY_COLUMNS:
            _positive_int(row["segment_index"], "segment_index", f"{table} row {index}")
            if row["segment_index"] != UNAVAILABLE and row["axis"] == UNAVAILABLE:
                raise P1SchemaError(f"{table} row {index}: segment_index without an axis (A5)")
        if row["track_state"] not in TRACK_STATES:
            raise P1SchemaError(f"{table} row {index}: track_state {row['track_state']!r}")
        if row["correspondence_status"] not in CORRESPONDENCE_STATUSES:
            raise P1SchemaError(f"{table} row {index}: correspondence {row['correspondence_status']!r}")
        if row["track_state"] == "unobserved" and row["unobserved_reason"] not in UNOBSERVED_REASONS:
            raise P1SchemaError(f"{table} row {index}: unobserved without a reason")
        if row["track_state"] == "removal_confirmed" and row["events"] in ("", UNAVAILABLE):
            raise P1SchemaError(
                f"{table} row {index}: removal_confirmed needs a record/decision event; "
                "absence in an image is never removal"
            )
        if row["organ_id"] == UNASSIGNED_ORGAN:
            if row["track_state"] != "ambiguous":
                raise P1SchemaError(f"{table} row {index}: an unassigned observation must be ambiguous")
            continue
        if row["organ_class"] == "compound_leaf" and not _COMPOUND_RE.match(row["organ_id"]):
            raise P1SchemaError(f"{table} row {index}: compound id {row['organ_id']!r}")


def check_optical_rows(rows: Sequence[Mapping[str, str]]) -> None:
    table = "optical_surface_manifest.csv"
    check_columns(rows, OPTICAL_MANIFEST_COLUMNS, table)
    for index, row in enumerate(rows):
        readiness = row["optical_readiness"]
        if readiness not in OPTICAL_READINESS:
            raise P1SchemaError(f"{table} row {index}: optical_readiness {readiness!r}")
        if row["readiness_reason"] in ("", UNAVAILABLE):
            raise P1SchemaError(f"{table} row {index}: readiness without a reason")
        if readiness == "unavailable":
            if row["surface_file"] != UNAVAILABLE:
                raise P1SchemaError(f"{table} row {index}: unavailable surface with a file")
            continue
        if row["frame"] != "plant_local_mm":
            raise P1SchemaError(f"{table} row {index}: optical surfaces are in plant_local_mm")
        if row["geometry_source"] in ("bbox", "hull", "compound_envelope", "area_angle_tuple"):
            raise P1SchemaError(
                f"{table} row {index}: {row['geometry_source']} is not certified optical tissue"
            )
        if readiness == "supported" and row["geometry_source"] == "generated_candidate":
            raise P1SchemaError(
                f"{table} row {index}: generated geometry can only be conditional_completion"
            )
        if readiness == "conditional_completion" and row["geometry_source"] != "generated_candidate":
            raise P1SchemaError(f"{table} row {index}: conditional completion must be generated")
        if row["evidence"] == "real_photo" and row["geometry_source"] == "synthetic_truth":
            raise P1SchemaError(f"{table} row {index}: synthetic truth labelled as real")
        split_external_id(row["external_leaflet_id"])


def check_table_header(columns: Sequence[str], table: str) -> None:
    """Rules that apply to any exported header."""
    check_no_on_hold_fields(columns, table)
    check_insertion_angle_prefix(columns, table)
    for column in columns:
        if column.startswith("area_") and "green" in column and not column.startswith(
            "area_green_only_"
        ):
            raise P1SchemaError(f"{table}: green-only area must be named area_green_only_*")


@dataclass(frozen=True)
class SegmentStatus:
    """``run_status`` of one P1 segment inside ``p1_manifest.json``."""

    segment: str
    status: str
    reason: str
    outputs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise P1SchemaError(f"segment {self.segment}: run_status {self.status!r}")
        if self.status != "executed" and not self.reason:
            raise P1SchemaError(f"segment {self.segment}: {self.status} needs a reason")

    def as_dict(self) -> dict[str, object]:
        return {
            "segment": self.segment,
            "run_status": self.status,
            "reason": self.reason,
            "outputs": list(self.outputs),
        }


def blank_mtg_row() -> dict[str, str]:
    """A vertex row with every value ``unavailable``; producers fill what they can support."""
    row = {column: UNAVAILABLE for column in MTG_VERTICES_COLUMNS}
    return row


def set_value(row: dict[str, str], attr: str, value: str, status: str) -> None:
    """Set ``attr`` and its ``__status`` together."""
    if attr not in MTG_VALUE_ATTRIBUTES and f"{attr}__status" not in row:
        raise P1SchemaError(f"unknown attribute {attr!r}")
    if status not in VALUE_STATUSES:
        raise P1SchemaError(f"status {status!r} not in the vocabulary")
    if (value == UNAVAILABLE) != (status == UNAVAILABLE):
        raise P1SchemaError(f"{attr}: value {value!r} with status {status!r}")
    row[attr] = value
    row[f"{attr}__status"] = status
