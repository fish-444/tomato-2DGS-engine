# CONTRACT — Traits from the 3D reconstructions (build/traits_recon_20261101_01)

Written 2026-11-01 before any trait number was computed. Prompt: `handoff/CLI_PROMPT_TRAITS_FROM_RECONSTRUCTIONS_TO_MTG_2026-11-01.md`
(sha256 in `input_hashes.json`). Amendment 1 = the user's addendum of 22:05 (`CONTRACT_AMENDMENT_1.md`). Nothing below is changed after
results are seen; a change means a new contract (`tuned` label).

Labels on every output: `recon_trait`, `conditional_on_unvalidated_scale`, `after_checkpoint_v1`.

## 0. Scope of this run
- Now: A1–A3 for every reconstructed plant that has an engine-static surfel file, and §3 synthetic recovery for every extractor
  (A1–A3, B1–B3 including the Amendment-1 arch traits, C1, C3). B and C on real plants wait for the user's skeleton/axis confirmation
  on 8835 and for grades — not run here.
- Plants: R3-1b, n_01–n_06, t2_02, t2_04, t2_05 (input `build/engine_static_nine_20261028_01/kaggle_data/<id>/static_surfels.csv`
  plus `kaggle_out/<id>/per_surfel_light.npz`). e_01–e_07: no area (b) or light file → `not_run`. t2_01, t2_03: excluded by the user.
- Validation plants (day-0 series, validation trays 1·2) are never used. Whether any of the ten plants above belongs to them is
  asked to the user; until answered, outputs carry `validation_status_unconfirmed`.
- Sowing dates: none found in the repository for these plants → `age_unknown`; no age pooling.
- Compute: one Kaggle CPU kernel. Local work is limited to parsing the hand workbook (44 rows) and writing tables.

## 1. Inputs and point sets (real plants)
- Normals as stored (two-sided, oriented +z). Weight = `area_b_mm2` (fallback `area_a_mm2` only if area_b is missing; counted).
- Height h: `h_above_base_mm` from per_surfel_light when the row count matches and |z_csv − z_npz| ≤ 0.01 mm for all rows;
  otherwise h = z − min(z) and flag `h_from_z_min`.
- Primary set P: organ ≠ `stem` (R3-1b has a stem class; the nine others are all `canopy_surface`).
- Secondary set S: P minus `axis_like == 1` (proxy, share 0.014–0.041 on the nine vs 0.29 on R3-1b) → comparability caveat.

## 2. Trait definitions and fixed thresholds
**A1 LAD.** Inclination θ = arccos(|n_z|) in degrees (leaf inclination from horizontal = normal zenith).
- Histogram 0–90° in 5° bins, area-weighted. Weighted mean and SD.
- Ellipsoidal (Campbell) fit: maximum weighted likelihood on χ ∈ 400 log-spaced values in [0.05, 20], with
  g(θ) = 2χ³ sinθ / (Λ (cos²θ + χ² sin²θ)²), Λ = χ + 1.774 (χ + 1.182)^−0.733.
- Beta fit: method of moments on t = θ/90 clipped to [0.001, 0.999].
- By height thirds: equal-height thirds of [p1, p99] of h over P.

**A2 size.**
- Height = p99.5 of h over P ∪ stem; also max.
- Canopy projected area: xy occupancy, cell 1 mm (primary), 2 mm (sensitivity).
- Widths: p0.5–p99.5 extent in x and in y of the plant frame; PCA major/minor extent.
- Area (b) total = Σ area_b over P.
- Profile: Σ area_b per 5 mm layer of h.

**A3 azimuth.** Surfels with θ ≥ 10° only. φ = atan2(n_y, n_x).
- Rose: 12 bins of 30°.
- Surfel-level area-weighted mean resultant length R̄.
- Random-orientation test: 10 mm 3D voxels; each voxel contributes the unit vector of its area-weighted mean azimuth. Rayleigh test on
  voxels, α = 0.01 → `non_random` / `random_not_rejected`.
- The frame is the board or fallback frame, so the mean direction is reported but not interpreted against light.

**B (synthetic now; real later).** Skeleton = polyline base → tip.
- Densification: points of the leaf's rachis within 3 mm of the polyline, binned every 2 mm of arc length (bin mean); the polyline
  itself where a bin is empty.
- B1 divergence: leaf base direction = chord s ∈ [0, 10] mm projected onto the plane ⟂ to the local stem axis (stem polyline
  direction at the nearest point).
  - Azimuths by rank; δ_k = (ψ_{k+1} − ψ_k) mod 360. If the circular mean of δ > 180°, use 360 − δ and handedness `cw`,
    else `ccw`.
  - Report the circular mean ± circular SD; compare with 137.5° and with protractor data when given.
- B2 insertion angle = angle between the chord [0, 10] mm and the local stem axis. Arch traits (Amendment 1):
  - θ0 = elevation above horizontal of chord [0, 10] mm; θ_tip = elevation of chord [L − 10, L].
  - Total bend Δθ = θ0 − θ_tip; mean curvature Δθ / L (°/mm).
  - s_apex / L = arc position of max z; tip drop = z(L) − z(0).
  - Arc vs quadratic: θ(s) = θ0 − Δθ g(s/L), g = s/L (arc, primary) or (s/L)² (quadratic, sensitivity), drawn in the vertical
    plane through the base along the base → tip horizontal direction.
    - (i) Direct model from the measured (θ0, Δθ, L): max deviation and tip error.
    - (ii) Least-squares fit of (θ0, Δθ): RMS residual.
    - "Better" = lower fitted RMS, reported per leaf. The primary stays arc whatever the result.
- B3: leaf length L (arc length); internode = difference of consecutive leaf-base positions projected on the stem polyline (by rank);
  node count = number of main-axis leaves.

**C1 curl.** φ/ψ with the estimator of `build/engine_curved_T_20261101_01/make_T_curved.py::phi_psi`, re-implemented as a pure
function with identical arithmetic (normals → +z, area weights, mean normal, principal in-plane axis, side medians, weighted slope
sign, |corr| < 0.1 → `sign_undetermined`).

**C3.** Area-weighted mean θ per leaf vs rank.

**C2** (per-leaf area, L/W, leaflet count/size/insertion): needs leaf/leaflet segmentation of real plants → `not_run` now.
The area (b) estimator belongs to the static pipeline and cannot be re-derived from synthetic points → its §3 status is `not_tested`.

## 3. Synthetic recovery
- Generator: `tomato3d/synth/microtom_v2.py`; geometry to points: `engine_points.py` unchanged (`meshes`, `sample`; density 4/mm²,
  blades two-sided). Its recon-only imports (`charuco_a3`, `r31_score`) are stubbed because `sample` does not use them.
- Grid: default `params_for()` plus one-factor changes. Each configuration uses seeds 101–104, giving 44 plants:
  - phyllo {130, 145}
  - (elev_low, elev_high) {(60, 30), (80, 55)}
  - bend {0.1, 0.5}
  - cup {0.1, 0.45}
  - droop {0.15, 0.6}
- Point noise:
  - Each point moves by U(−0.75, +0.75) mm along its normal (sheet ~1.5 mm).
  - Each normal is rotated by |N(0, σ_n)| about a random perpendicular axis. σ_n (primary) = the R3-1b estimate: median angle between
    a surfel normal and the PCA normal of its P-neighbours within 2 mm (≥ 6 neighbours), computed in the same kernel. σ_n = 15° is
    reported as a stress level.
  - Normals are then oriented +z.
- Skeleton noise:
  - True rachis polyline resampled at 20 mm arc spacing (matching `legacy_split_20mm`) plus both ends.
  - Each node gets N(0, 1 mm) 3D noise.
  - The skeleton is then densified with the leaf's noisy rachis points as in §2.
  - Stem axis = main-axis tube polyline with the same node noise.
- Truth = the same definition applied to exact geometry:
  - mesh triangles (centroids, normals, areas) for A1/A3/C1/C3;
  - exact polylines for B;
  - max z of the mesh for height;
  - 0.25 mm rasterised mesh footprint for projected area;
  - min–max vertex extents for widths.
- Truth for A1/A3 = blade (+ cotyledon) triangles only. Extraction uses all non-stem points (primary, as in the reconstructions, so
  rachis contamination is part of the bias) and blades-only (secondary).
- Gate per trait: |mean bias| ≤ 5° for angles; ≤ 10 % mean relative bias for lengths and areas; |bias R̄| ≤ 0.05 for A3 R̄. Divergence
  bias is the circular difference. Evaluated at the primary σ_n; failures → `trait_unreliable`.
  - Relative bias = Σ(extracted − truth) / Σ|truth| over all synthetic items (ratio of means, stable when single truths are near 0,
    e.g. tip drop).
  - Not gated (reported only): ellipsoidal χ, beta parameters, s_apex/L, profile, node count (equal by construction), Rayleigh
    decision agreement.
  - Mean curvature Δθ/L (°/mm) is gated as a relative quantity (same 10 %).
  - Leaves with true rachis length < 25 mm are skipped for B (two 10 mm chords need L ≥ 20 mm).
  - Synthetic B/C use main-axis leaves only.
  (Clarification added before any number was computed, 2026-11-01.)
- Hand comparison:
  - Hand θ0 (Amendment 1) vs skeleton θ0 per leaf where the same plants were reconstructed.
  - If dt01–dt06 were not reconstructed, this is stated and only the synthetic check applies.

## 4. Provenance in the mapping table
- `recon_measured`: n ≥ 3 plants and passed §3.
- `recon_sparse`: passed §3 but n < 3.
- `trait_unreliable`: failed §3.
- `hand_measured`: Amendment 1.
- `published`: literature values.

Template: MTG_TEMPLATE_v1 once ≥ 3 plants have B traits (not in this run).

## 5. Do not
- Use validation plants.
- Use traits that fail §3.
- Tune thresholds after values are seen.
- Present published or sparse values as measured.
- Edit skeletons, labels, engines or frozen outputs.
- Run heavy local compute.
- Delete anything.
