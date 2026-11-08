# skeleton_v1 — persistent compound-leaf identity (2026-11-01)

One polyline per compound leaf, base → tip, from the user's clicks. Stored apart from surfel labels; labels are always re-derived from it (never edited by hand).
Contract: `build/p1_skeleton_layer_20261101_01/CONTRACT.md`. Code: `skeleton_io.py` (schema, validator, conversions, held-out agreement), `skeleton_store.py` (re-derivation from the append-only logs),
`skeleton_propagate.py` (skeleton → labels), `skeleton_tree.py` (skeleton → `plant_tree_v1`), `skeleton_carry.py` (next capture date).

## File
`data/skeletons/<plant_id>/<capture_date>/skeleton_v1.json` (+ `clicks.jsonl`, the editor's append-only log; `skeleton_v1.<stamp>.json` = earlier versions, kept).

| field | meaning |
|---|---|
| `schema` | `skeleton_v1` |
| `plant_id`, `capture_date` | e.g. `n_04`, `2026-10-24` |
| `reconstruction_id` | build folder of the surfels + sha256[:16] of the PLY |
| `frame` | `{name: board_m_zup, transform_to_pot: null or 4x4}`; all coordinates in metres, board frame, z up |
| `scale_status` | `conditional_on_unvalidated_scale` until the pot is measured |
| `labels` | `user_skeleton_seeds`, `conditional_on_unvalidated_scale`, `after_checkpoint_v1` |
| `leaves[]` | see below |
| `stem_display` | `{display_only: true, polyline_xyz: []}` — schematic only, never a measurement |
| `heldout[]` | `{surfel, leaf_uid or null (= 모름), point_xyz, at}` — never used as seeds |
| `editor_version`, `created`, `edited` | provenance |
| `registration` | carry-forward only: ICP residual, pairs, iterations, `scaling: none`, `from_date` |
| `minutes` | clicking time (sum of gaps ≤ 10 min between records) |

### leaf
| field | meaning |
|---|---|
| `leaf_uid` | `<plant_id>-L<NN>`, given in creation order, never reused, persistent across dates |
| `rank_hint` | rank by base height among live leaves (recomputed each date; 0 for lost / merged) |
| `base_xyz` | first polyline point |
| `polyline_xyz[]` | as clicked, base → tip (metres) |
| `polyline_1mm_xyz[]` | derived: linear resampling at 1 mm |
| `source` | `user_click` · `propagated_from:<date>` · `propagated_edited` · `simulated_from_human_traces` |
| `status` | `active` · `new` · `lost` · `merged_into:<uid>` · `candidate_lost` (carry-forward: base found no surface within 8 mm) |
| `n_clicks`, `click_log_lines` | alive clicks; 1-based line numbers in the base-click log |
| `flags` | validator output (below) + `legacy_split_20mm` (n_04 lines cut by the old rule, to be re-checked) + `no_snap` (carry) + `check_identity` |
| `attrs` | free: `trace_index`, `trace_key`, `pipeline_leaf`, `flipped_from_trace` (simulated R3-1b), `carried_from`, `snapped_points` |

## Validator (`skeleton_io.validate`) — flags, never fixes
- `base_tip_order`: base farther from the plant axis (vertical through the pot centre, else the surfel-xy centroid) than the tip.
- `shared_base`: two live leaves with bases < 3 mm apart.
- `off_surface`: a polyline point > 4 mm from every surfel.
- `gap_over_20mm`: a step > 20 mm inside one line (warning; the old rule that split lines there is gone — the user ends lines).

## Example (abridged)
```json
{"schema": "skeleton_v1", "plant_id": "n_04", "capture_date": "2026-10-24", "reconstruction_id": "build/p1_recon_all_20261025_01/outputs/n_04#…",
 "leaves": [{"leaf_uid": "n_04-L01", "rank_hint": 3, "base_xyz": [0.2248, 0.1217, 0.1194], "polyline_xyz": [[…]], "source": "user_click", "status": "active",
             "n_clicks": 1, "click_log_lines": [1], "flags": ["legacy_split_20mm"]}],
 "heldout": [{"surfel": 1219, "leaf_uid": "n_04-L03", "at": "…"}]}
```

## Re-derivation (`skeleton_store.rebuild`)
base clicks (`user_base_click.jsonl`, `add`/`undo`, `line`) → one leaf per user-ended line (legacy records without `line`: cut by 20 mm, flagged) → editor log replayed in order
(`extend`, `move_point`, `delete_point`, `mark_lost`, `merge`, `heldout_pick`, `end_line`, `start`, `done`) → validator → saved as a new version. Logs are never rewritten.


## skeleton_v1.1 — axes (Amendment A6, 2026-11-01)
- `schema` becomes `skeleton_v1.1` as soon as an axis exists. v1 files without axes are unchanged, and `load` accepts both.
- `axes[]`: `axis_uid` (`<plant>-A<NN>`, never reused), `type` (main / lateral), `parent_axis_uid`, `attach_s_mm`, `status`, `points[{xyz, source}]` with source `user_click` / `pot_centre_auto` / `leaf_base_anchor` / `fspm_fill` / `observed_photo_click` / `observed_3d_click`, plus the derived `fit`, `segments[{s0_mm, s1_mm, provenance}]` and `nodes[]`.
- Derived on every rebuild (`skeleton_axis.fit_all`): curve, segments (observed = consecutive user clicks <= 15 mm apart along the curve), nodes, `fspm_node_hypothesis` nodes, and the leaf fields `attach_axis_uid` / `attach_s_mm` / `node_index` / `attach_distance_mm` (> 12 mm = unattached).
- Editor records: `axis_start {type, parent_axis_uid}`, `axis_add {axis_uid, point_mm, source}`, `axis_move_point {axis_uid, index, point_mm}`, `axis_delete_point {axis_uid, index}`.
