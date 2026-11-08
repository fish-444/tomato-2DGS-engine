# Separate the coplanar half-credit effect from the bending effect (R3-1b) — CONTRACT (2026-11-01 18:40, before any number)

Labels: `coplanar_separation` · `representation_comparison_not_validation` · `conditional_on_unvalidated_scale` · `after_checkpoint_v1` · `R3-1b_unaccepted` · `engine_law_leaflets`.
Source: the pasted prompt "separate the coplanar half-credit effect from the bending effect" (2026-11-01 18:35; user decision 18:29).
Hashes of the crosswalk, engine, tracer code copy, `tau_config.json` (d_ex 2.4 mm), and the curved-T folder (flat / fold / fold+cup meshes, kernel, outputs) are in `CONTRACT_HASHES.json`, written before this file. The half-credit rule (`static.tracer.coincident`, ±1 µm) and the lamps are in the hashed code.
If not finished by 2026-11-05: `not_done`, and the curved-T limitation sentence stands.

## 1. Variants (one run each)
- **Unchanged inputs, copied byte for byte from `build/engine_curved_T_20261101_01/`:** `T_flat`, `T_fold` and `T_foldcup`.
- **`T_flat_offset_δ`, δ = 0.05, 0.1 (primary) and 0.2 mm:**
  - Every leaflet of `T_flat` is translated rigidly along its own unit normal: the normal of its first triangle, flipped to +z.
  - Each leaflet's whole triangle set moves by the same vector. There is no rotation and no bending, so area and orientation are unchanged.
  - Rachis and petiole triangles are not moved.
- **Sign rule (declared):**
  - Within each skeleton line (one compound leaf), leaflets are numbered in file order: P1L, P1R, P2L, P2R, …, T.
  - Leaflet k gets +δ for even k and −δ for odd k.
  - So the two leaflets of one node pair always get opposite signs, and consecutive nodes alternate.
- **Receivers:** as in the curved-T run: 16 points per triangle for the flat and offset variants, 8 for fold and fold+cup. Sample seed = trace seed (101 / 202).
- **D:** the frozen static R3-1b scene, d_ex 2.4 mm. The same tracer, lamps, seeds and half-credit rule for every T.

## 2. Gates (method-of-separation statements only)
- **G1 reproduction:**
  - D v1 = local v1 (max |Δ| = 0, overflow 0);
  - T_flat, T_fold and T_foldcup equal the curved-T run's cabinet v1 totals and self-shading within 0.5 % relative. The meshes, seeds and code are the same, so equality is expected.
  - Otherwise stop.
- **G2 area:** per-leaf leaflet area of every offset variant equals T_flat within 1e-12 relative.
- **G3 coplanarity broken:** the share of coincident (half-credit) receivers in `T_flat_offset_0.1` is ≤ 0.5 % (seed 101).
- **G4 offset insensitivity:** cabinet v1 T/D for δ = 0.05 and 0.2 each within ±1 % of δ = 0.1, i.e. |T/D(δ) / T/D(0.1) − 1| ≤ 0.01. Otherwise `offset_sensitive`, and the decomposition is reported as unreliable, with its numbers shown.

## 3. Report
- **Metrics,** as in the curved-T run:
  - cabinet T/D (v1, v1_tau), self-shading, thirds, per-leaf T/D, Spearman ρ (absorbed rank, light-per-area rank);
  - engine-lamp static T/D, GroIMP / static (all T variants; packet noise; seed identity stated);
  - coincident receiver share per variant.
- **Decomposition at δ = 0.1 mm:** flat → fold+cup = rule (flat → offset) + fold (offset → fold) + cup (fold → fold+cup).
  - Shown in absolute T/D units and as % of the total.
  - Done for cabinet T/D (v1 and v1_tau), self-shading and lower-third T/D.
- **Sentence:** "Of the drop from ×1.38 (flat) to ×1.24 (fold+cup), X (Y %) came from the coplanar half-credit rule and Z (W %) from bending; fold+cup ×1.24 is unaffected (coplanar share ≤ 0.2 %)." If G4 fails: "the split could not be determined (offset-sensitive)".
- **Outputs:**
  - `coplanar_sep_R3-1b.csv`, `DECOMPOSITION.json`, `GATES.json`, `figures/decomposition_bars.png`;
  - `HASHES.json`, LOG line, flow row;
  - a Korean section in `handoff/ENGINE_TO_POINTS_2026-10-31_KO.md`.

## Do not
- Change the half-credit rule, tracer, d_ex, crosswalk, engine laws, lamp or frozen outputs.
- Bend or rotate leaflets in the offset variants.
- Choose δ after seeing results.
- Present T/D → 1 as validation.
- Run more than once per variant.
