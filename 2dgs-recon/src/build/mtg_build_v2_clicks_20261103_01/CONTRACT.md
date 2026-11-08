# CONTRACT — MTG static set v2 (click skeleton), packaged for the B-model and Engine v2 sessions

Written 2026-11-03 22:48 before any output. User choice "골격 MTG 새로 묶기" (after asking for MTG modelling based on the clicked skeleton). Packaging only — **no new geometry, no fitting**.

* Source of every observed-plant MTG: `build/mtg_fit_surfels_20261103_01/fitted_clicks/mtg_<id>_spec_fitted.json` (CONTRACT G there: rachis from the clicked line; stem + nodes = adopted surfel fit U). Read only; sha256 in `MANIFEST.json`.
* Template-mean MTG, dynamic rule file, random-parameter additions, internode setting: copied **byte-identical** from `build/mtg_build_20261102_01/` (unchanged by this version; sha256 checked equal after copying).
* Files are regenerated with the original build's writers (`writers.write_engine`, `writers.spec_to_bmodel`, Level-2 polylines as in `fit_plant.write_fitted`) under the **original names** (`static/mtg_<id>_spec.json`, `_engine.plant_tree_v1.json`, `_engine/`, `_engine.obj`, `_bmodel.csv` + `.rows.json`), so a reader can point at this folder instead of the old one.
* Checks (declared): each regenerated B-model CSV must be byte-identical to the one already imported on Kaggle (`mtgb-bm-clicks-20261103`, 7/7 exact) → that round trip carries over; any difference → re-run on Kaggle. Engine round trip re-run locally (11/11 required, same tolerance as before).
* Labels: `mtg_from_traits`, `rachis_from_clicks`, `stem_fitted_to_surfels_user_accepted`, `skeleton_labels_ungraded`, `conditional_on_unvalidated_scale`, `after_checkpoint_v1`, `v2_ungraded_draft_not_frozen` (template). Observed-plant φ / θ0 / Δθ / L are plant geometry, not traits.
* Not changed / not included: template v1 / v2 draft, Traits files, skeletons, labels, the old build. e_01 / e_04 / e_05 / e_07: engine only (no blade area → no B-model), as in v1 of the set.
