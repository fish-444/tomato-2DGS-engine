# plant_tree_v1 — one plant-tree format for reconstruction, synthetic plants and engine states (2026-10-22)

Code: `tomato3d/plant_tree/` (`schema.validate`, `obj.to_obj` / `obj.from_obj_v1`, `summary.summary` / `summary.compare`,
`engine.to_engine_structure` / `engine.from_engine_states`). One JSON document per plant. Validation is plain Python
(`schema.validate` raises `ValidationError` with the path of the first problem). The organ-split session's
`organ_tree.json` and the synthetic generator v2 both produce this format; the engine's observation route reads it
through `to_engine_structure`.

## 1. Header
| key | values |
|---|---|
| `schema` | `"plant_tree_v1"` |
| `source` | `synthetic` / `reconstruction` / `engine` |
| `input_status` | free text, e.g. `R3_unaccepted`, `accepted`, `v1_synthetic`, `engine_state` |
| `units` | `mm` or `board_m` (declared; all geometry in that unit) |
| `frame` | free text, e.g. `pot top centre, z up`, `board metres, z up`, `plant_local_mm, z up`. **z is always up.** |
| `synthetic` | bool; a `synthetic` source must carry `true` |
| `provenance` | `{field: tag}` default provenance per value: `user_measured` / `literature` / `assumption` / `reconstructed` / `derived_from_mesh` / `engine_state` / `unavailable`; any organ may override with its own `provenance` dict |
| `plant` | `{id, flags, attrs}` (`attrs.organ_split`, `attrs.capture_id` … free) |
| `params`, `notes` | generator parameters (with per-key provenance) / free notes |

## 2. Nodes of the tree
Ids are unique in the document. Lists are ordered; axes list parents before children.

| level | list | required fields |
|---|---|---|
| plant | `plant` | `id` |
| axis | `axes[]` | `id` (`M` = main; `S1…` side; `A1…` sympodial), `kind` (`main` / `side` / `sympodial`), `parent_axis`, `parent_node`, `tube` (polyline + radius profile, §3), `flags` |
| node | `nodes[]` | `id`, `axis`, `index` (1 = first node above the cotyledon node), `rank` (U5: main stem nodes 1 and 2 share rank 1, node i → i−1; side axes one rank per node), `s_mm` (arc length along the axis), `position`, `leaves[]`, `trusses[]`, `flags` |
| leaf | `leaves[]` | `id` (`L01`, `S1_L01`), `node`, `axis`, `rank`, `petiole` (tube or null), `rachis` (tube: polyline from the node to the tip), `attachments[]` = `{s_mm, leaflets[]}` (arc length along the rachis), `length_mm`, `azimuth_deg`, `elevation_deg`, `age_d`, `expansion` (0–1), `flags` |
| leaflet | `leaflets[]` | `id` (`L01__P1L`, `L01__T`, `L01__I1`), `leaf`, `type` (P/T/I), `side` (L/R/C), `pair` (0 = terminal), `attachment_s_mm`, `base`, `frame` `{d, n, b}` (local frame: along the midrib, normal, across), `length_mm`, `width_mm`, `area_mm2`, `normal`, `surface` (§3), `outline` (optional normalised polygon), `petiolule` (tube or null), `age_d`, `expansion`, `flags` |
| truss | `trusses[]` | `id` (`T1`), exactly one of `node` / `axis`, `peduncle` (tube), `flowers[]` = `{id, position, mesh?, pedicel?}`, `flower_count`, `stage`, `flags` |
| cotyledon | `cotyledons[]` | `id`, `surface` (kept for the v1 round trip; no engine kind) |

Flags (per organ, any level): `few_views`, `ambiguous`, `unattached`, `gap`, `edge_soft`, `occluded_part`, `merged_suspect`,
`stub`, `intercalary_as_extra_leaflet`, `lateral_shoot_mapped_to_sympodial_axis`, `dropped_no_engine_kind`,
`clipped_to_measured_max`, `requires_declaration`.

## 3. Geometry
- **tube** `{kind: "tube", polyline: [[x,y,z]…], radius: [r…] (one per point), segs}` — regenerated with the v1 tube maths
  (same vertices as v1); or `{kind: "mesh", vertices, faces, centreline, radius, segs}` — stored verbatim (used by
  `from_obj_v1`), `centreline` / `radius` derived from the rings.
- **surface** of a leaflet: `param_blade` `{base, d, n, L, W, cup, bend, nu, nv, outline?}` — regenerated with the v1 blade
  maths; with `outline` the half-widths follow the outline polygon (option B); `grid` `{nu, nv, vertices}` — a stored
  (nu+1)×(nv+1) blade grid; `outline_polygon` `{points: [[u,v]…], tag}` — normalised outline (u along the midrib 0→1,
  v across in units of the length); `mesh` `{vertices, faces}`; `ref` `{ref}` (a surface stored elsewhere, e.g. a
  reconstruction's leaflet mesh).
- Descriptors (`length_mm`, `width_mm`, `azimuth_deg`, `curvature`) are computed by `summary()` from the geometry with ONE
  definition for every source: leaf azimuth/elevation from the first part of the rachis polyline (3 points), curvature =
  turning angle per mm of rachis, leaflet length = chord base→tip of the grid, width = widest row.

## 4. Engine mapping (`engine.py`; the engine `kinds` array, read-only)
| plant_tree_v1 | engine kind | engine id (plant `p1`) | synth OBJ name |
|---|---|---|---|
| plant | Plant | `p1/Plant` | — |
| axis main | Axis, Internode, Apex | `p1/A0`, `p1/I<node>`, `p1/A0/apex` | `stem` |
| axis sympodial / side | Axis | `p1/A<k>` (+ `p1/A<k>/I<node>`, `/N<node>`) | `side_stem` |
| node | Attachment | `p1/N<node>` | — |
| leaf | CompoundLeaf, Petiole, RachisSegment | `p1/C<nnnn>`, `…/P`, `…/R<i>` | `rachis_L01` |
| leaflet | Leaflet | `…/P<i>L`, `…/P<i>R`, `…/T` | `L01__P1L`, `L01__T` |
| truss | TrussAxis, Flower | `p1/Truss<n>`, `…/F<i>` | `T1_peduncle`, `T1_flower<i>` |
| (root, mass only) | Root | `p1/Root` (flag `requires_declaration`) | — |

Declared fallbacks (gaps the engine has no organ for): intercalary leaflet **I → extra Leaflet on the nearest rachis
segment**, id `…/I<k>L|R`, flag `intercalary_as_extra_leaflet`; Micro-Tom **lateral shoot S<k> → sympodial Axis A<k>** on
its node's internode, flag `lateral_shoot_mapped_to_sympodial_axis`; **cotyledons / hypocotyl → dropped**, listed in
`dropped.json` (flag `dropped_no_engine_kind`). `to_engine_structure` writes the Stage-2 inputs the observation route
validates (`initial_state.csv`, `initial_controls.json`, `mtg_vertices.csv` revision 9, `mtg_edges.csv`,
`structure_state.csv`, `observation_manifest.json`, `plants.csv`); hidden variables (mass, age, reserves, root) are
**0.0 placeholders** declared in the manifest, with `initial_state_template.csv` carrying them EMPTY and flagged
`requires_declaration`. `from_engine_states(states.csv, day)` reads one snapshot back (leaflets become `param_blade`
with cup = bend = 0; the engine stores no outline). Engine-default mismatches are reported in `mapping_report.json`
(FIRST_LEAVES 9 vs 6–8; PAIRS [1,1,2] vs 3 pairs + intercalary; LEAF_LW 2.0 vs 1.0–2.2; flat diamonds; INTERNODE_CAP 12 mm
vs 6–9; sympodial A1–A3 + 4 trusses vs two successive trusses then laterals) — the engine session decides any change.

## 5. Converters
`to_obj(tree, plant_prefix=None)`, `from_obj_v1(obj_text)`, `summary(tree)`, `compare(tree_a, tree_b)` (node-count error,
leaf matching by rank + azimuth (Hungarian, cost = 90·|Δrank| + Δazimuth, +60 across axes, cut 120), leaflet-membership
agreement = Σmin/Σmax of matched leaflet counts with unmatched leaves in the denominator, 1-D Wasserstein distances of
the per-leaf descriptors; azimuth compared as sorted gaps). Backward-compatible extensions bump a minor field
`schema_minor`; a new `schema` string means a new version.

`human_confirmed_shape` (added 2026-10-22): an outline taken from a photo mask that a human answered "one whole leaflet" (library `outlines_human_v1`); a top projection, shape only, never a size.
