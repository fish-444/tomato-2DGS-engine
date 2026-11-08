# R3-1b as surfel discs (D) vs engine triangles (T): same scene, same tracer — CONTRACT (2026-11-01 14:20, before any number)

Labels on every output: `disc_vs_triangle` · `relative_light` · `conditional_on_unvalidated_scale` · `after_checkpoint_v1` · `R3-1b_unaccepted` · `engine_law_leaflets`.
This is a representation comparison. It is not validated against any measurement and is never presented as validation.
Source: the addendum to `CLI_PROMPT_ENGINE_TO_POINTS_AND_ANALYSIS_BY_SYNTHESIS_2026-10-31.md` Part A (2026-11-01 13:30).

## 0. G-init decision (user, 2026-11-01, in chat)
- **What exists:**
  - The R3-1b skeleton `data/skeletons/R3-1b/2026-10-21/skeleton_v1.json` has 11 lines covering R8_pc leaves 1–8. Sources: `simulated_from_human_traces`, plus leaf 8 `sim_skeleton_completed`.
  - The crosswalk `organs.skeleton_tree.skeleton_to_plant_tree` → `plant_tree.engine.to_engine_structure` emits CompoundLeaf, Petiole, Attachment, Internode, Axis, Apex and Root. It emits **no Leaflet or RachisSegment rows**; the tree has `leaflets=unavailable`.
  - So no engine leaflet with imported posture exists. Engine contract §3 says G-init is attempted only on the user's decision.
- **User decision:** "엔진 법칙 소엽" (engine-law leaflets).
  - T = the observed rachis lines (imported posture) carrying leaflets made by the engine's own geometry laws.
  - Each leaflet set is scaled to the leaf's area (b).
  - Label: `engine_law_leaflets`. This is the G-init decision for this comparison only; it is not a general G-init.
- **Not changed:** the crosswalk, the engine, d_ex, the half-credit rule and the lamp. The generator lives in this folder.

## 1. Geometry (same board frame, same placement)
- **D (discs):** the frozen static scene of R3-1b, identical to `engine_static_tau_20261031_01` k1b `v1`:
  - surfel discs, area rule (b) (P1 silhouette, `static_surfels.csv`), d_ex 2.4 mm;
  - black leaves (v1); labels R8_pc (`labels_rachis.npz`);
  - blade of leaf L = axis_kind 0 & label L.
- **T (triangles):**
  - **Crosswalk:**
    - Run unchanged on the R3-1b skeleton: `skeleton_to_plant_tree` → `to_engine_structure`.
    - Output kept in `crosswalk/`. Its validators must pass.
    - One CompoundLeaf per skeleton line (11 lines; leaf 5 has 3 lines and leaf 7 has 2). Lines are not joined: the crosswalk does not join them.
  - **Leaflets by engine law,** from `tools/vtc_development_native.rgg` with the parameter values of the seed72 run's Model.rgg (hash recorded):
    - `PAIRS = [1, 1, 2]` by leaf rank; `pairWeight` with `MIDDLE_RAMP 0.35`;
    - petiole : rachis = `PETIOLE_M 0.02 : RACHIS_M 0.035`, segments equal;
    - leaflet base at the tip of its parent (pair 1 on the petiole, pair i on rachis segment i−1, terminal on the last segment);
    - leaflet direction = parent direction turned by β = side·35°·(m+1−pair)/m;
    - full-outline diamond, len = √(2·A·LEAF_LW), wid = len / LEAF_LW, LEAF_LW 2.0, two triangles (the engine's `vertices()` formula);
    - rachis and petiole segments: engine slender diamonds, width `RACHIS_WIDTH_M` 1 mm (occluders only).
  - **Imported posture:**
    - Engine proportions are scaled to the observed path length of each line.
    - Parent direction = chord of that parent's interval on the observed polyline (the crosswalk's rachis polyline = the skeleton clicks).
  - **Rank** (PAIRS index): the crosswalk node index (rank by base height of the skeleton; an ordering, not a node count, as the crosswalk labels it). So lines with rank 1–2 get one pair, the others two.
  - **Area:**
    - leaf L's blade area (b) in D is split over its lines in proportion to path length;
    - leaflet area = line area × the engine share (pair weight / total).
  - **Hidden organs:**
    - No stem: the crosswalk main axis is a zero-length placeholder and the stem is display-only.
    - No trusses or flowers (none observed).
    - Labelled `not_in_T`. Their effect is bounded by the D variant without stem surfels (§3e).
  - **Receivers:** leaflet triangles (64 points per triangle, area weights, two seeds), delta 1e-7 m. The coincident-receiver half credit is as in gate S1 v2 (`obs.PlacedModel`, ±1 µm).
  - **Occluders:** all T triangles.
  - **Files:**
    - written in the Part A exporter's convention (board metres, OBJ + PLY with leaf / leaflet ids);
    - the Part A plant_tree path cannot carry engine diamonds, because it regenerates leaflets as `param_blade`;
    - also `T_triangles.csv` (organ, class, leaf, line, 9 coordinates).
- **T_flat check:** area per leaf in D (area (b), blades) vs in T (Σ leaflet triangle area). This should match by construction. Any mismatch is reported, not corrected.
- **Placement:** both at the frozen static position `SCENE_XY` (0.280, 0.210) m with D's offset (`obs.Placed.offset`); T is moved by the same offset.

## 2. Runs (Kaggle CPU; static tracer = the v1 code copy in `dolgfish/static-tau-bundle-20261031`, unchanged)
- **Reproduction check, first:** D v1 at d_ex 2.4 must equal the local v1 R3-1b array (`r31b_ref.npz` relative_light, max |Δ| = 0). If not: stop.
- **Main, for D and T, cabinet kernel:**
  - measured field and its extension, the user_assumed bars, the dome;
  - seeds 101/202, black, 24 bar samples, 72 dome directions.
- **Second pass `v1_tau` (τ 0.07), reported separately:**
  - D uses the existing rule (axis_kind 1 → τ 0, else 0.07);
  - T: leaflets 0.07, rachis and petiole 0;
  - ε = 1.529 mm; T applies the same half credit.
- **Engine-lamp closure** (gate S1 lamp): isotropic point lamp 1200 µmol s⁻¹ at 0.6 m above the medium surface over the pot centre.
  - D and T static (point_lamps), seeds 101/202;
  - T in GroIMP 2.2.1 LightModel: the same T triangles as MeshNodes, BLACK shader, 30 M rays, ray seeds 11 and 12, `lm.setSeed`;
  - this is the dynamic engine's light model on the day-0 triangles: D (static) ↔ T (static) ↔ T (GroIMP).
- **Per-leaf metrics:**
  - absorbed (relative, a = 1), light per area, rank (R8_pc blade leaves 1–8);
  - plant absorbed for D in two receiver sets: all surfels (frozen definition) and blades 1–8 (the receivers T represents); T is leaflets;
  - self-shading = 1 − absorbed / unshaded;
  - height thirds on D's height reference (h = z − p0.5 of D, top = p99.5 of D), applied to both.
- **Seed noise:** σ = |s1 − s2| / 2. GroIMP noise: packet model σ = √(A·P/N), and the two-seed spread.

## 3. Report (declared, descriptive; no gate, no tuning)
1. **Table:** D vs T vs T-GroIMP — plant absorbed (ratio T/D), self-shading, thirds, per-leaf ratio and rank correlation (Spearman, 8 leaves) with seed noise.
2. **Decomposition:**
   - (a) area per leaf D vs T;
   - (b) orientation: the area-weighted mean normal per leaf (normals flipped to +z before averaging), D vs T angle, and each one's inclination;
   - (c) shape flattening: per leaf, the share of D blade area whose normal is more than 15° from the leaf's mean normal (|cos|);
   - (d) d_ex sensitivity on D (1.5 / 2.4 / 5 mm) vs the D–T gap;
   - (e) occluder set: D without stem surfels (axis_kind 1) as occluders, a bound on T having no stem.
3. **One sentence:** "At day 0, the triangle representation of R3-1b absorbs X % (range) of the disc representation; leaf ranks agree with ρ = Y."
4. **Scope:** R3-1b only (the only plant with a crosswalk skeleton). The others get the same comparison once their skeleton → `plant_tree_v1` exists.

## Outputs
- `disc_vs_triangle_R3-1b.csv` (plant / third / leaf rows);
- figures: per-leaf bars D vs T vs GroIMP; surfels coloured by deviation from the leaf mean normal;
- a Korean section appended to `handoff/ENGINE_TO_POINTS_2026-10-31_KO.md` (≤ 8 lines + table);
- LOG line, hashes (`HASHES.json`).

## Do not
- Change the crosswalk, the engine, d_ex, the half-credit rule or the lamp to make D and T agree.
- Tune any T parameter after seeing a number.
- Present this as validation.

## Amendment 1 (2026-11-01 14:55, after building T geometry, before any light number)
- **The engine "as run" draws leaflets from the outline library, not as diamonds.**
  - The reference run behind the static↔dynamic consistency and D-ref (seed72 Model.rgg, sha in `T_meta.json`) has the leaflet_outline hook on: `vertices()` → `outlineVertices()`, a library_v2 outline (194 outlines, unit area) scaled by √A in the leaflet frame.
  - The outline is chosen by `java.util.Random(seed + id.hashCode()).nextInt(194)`, with engine seed 5000 and organ id = the crosswalk id `p1/C000n/P<i>L|R|T`. This is reproduced exactly in Python.
- **So T (main)** = library outlines (5,628 leaflet triangles, 51 leaflets).
- **T_diamond (secondary, reported beside it)** = the base engine diamond of `tools/vtc_development_native.rgg` (2 triangles per leaflet).
- **Both** carry the same rachis lines, ranks, areas and engine laws otherwise. T_flat check: max |rel. area diff| = 9e-16 (T) and 2e-15 (T_diamond).
- **Receivers:** points per triangle so that every leaflet gets at least 128 points: 64 per triangle for T_diamond, 16 per triangle for T (about 110 triangles per leaflet). Sample seed = trace seed (101 / 202).
- **GroIMP:** run for T and T_diamond, 30 M rays, ray seeds 11 and 12.
- **Not changed:** anything else in §1–3.
