# Engine A STATIC mode — contract (2026-10-26, written before any number; INVENTORY.md is Step 0)

STATIC = one frozen geometry, no growth, no sink step, no random draw except the declared ray/sample seeds. Code lives in
`build/engine_static_20261026_01/static/` and imports `tools/` and `tomato3d/` read-only; **no engine file, parameter or dynamic output is
changed** (E8-control hash check at the end). P1 files are read only. Nothing is fitted to measured shading data. No absolute PPFD is claimed.

## 1. Geometry
**Frames and units.** Metres. Observed plants: P1 board frame (z up) → placed into the cabinet grow frame (shelf-floor height basis, the
measured field's frame) by a declared translation; engine snapshot: engine world frame (lamp scene) for the gates, and placed in the cabinet
for the model-vs-real comparison. Every output row carries `frame` and `scale_label`.

**Surfels → two-sided discs.** Each 2DGS surfel is an ellipse in its tangent plane: centre, tangents tu/tv, normal n, semi-axes `k·s0, k·s1`.
- Occlusion footprint: `k = sqrt(2 ln(2α))` for opacity α > 0.5 — the contour inside which the surfel is more than half opaque; surfels with
  α ≤ 0.5 do not occlude and do not receive (declared; footprint ×0.8 / ×1.2 sensitivity reported on R3-1b).
- Receiving area, rule (b) silhouette-calibrated (fallback because `build/p1_plant_model_20261026_01/` is absent): surfels are binned into 8 mm
  cubes; in each cube the footprints are projected onto the cube's area-weighted mean plane and rasterised at 0.2 mm; the union area divided by
  the summed projected area gives `c_cube`; each surfel's one-sided area = `c_cube · π k² s0 s1`. Rule (a) = the plain sum `Σ π k² s0 s1`.
  The (b)/(a) ratio is recorded per plant (and per leaf for R3-1b, where `leaves.csv area_m2` is also shown).
- Plant surface = opaque surfels above the pot top. R3-1b: leaf ids from `surfel_leaf_ids.csv` (receivers labelled leaf 1…8; −1/axis =
  non-leaf "plant surface"); per-leaf results **only for usable leaves 1 and 7**; all other leaves pooled as `other_leaves`; viewing_only leaves
  never get a per-leaf row. The other nine plants have no leaf split: all opaque surfels above the pot top form one canopy surface
  (`canopy_surface`, stems included; their "LAI" is labelled `canopy_surface_area_index`).
- Self-cloud exclusion: the surfel cloud of one leaf is a few mm thick, so a shadow ray ignores hits closer than **δ = 5 mm** to its origin
  (declared; δ = 3 / 8 mm sensitivity reported on R3-1b). Engine meshes are thin: δ = 0 there (origin offset 1e-7 m along the ray).

**Engine snapshot.** `consumed.csv` state 71 of `build/engine_maxmass_fix_20261025_01/runs/seed72` (the geometry GroIMP traced for interval 71 =
engine day 71 = 59.9 DAG): consecutive vertex triples are triangles; Leaflets receive, every meshed organ occludes.

**Scale check.** Pot surfels (below the pot top) with horizontal normals (|n_z| < 0.5) are the walls; their xy, rotated to principal axes,
give the square's side `L` (2nd–98th percentile extent). `r = L / 59.5 mm` (rim spans the 59 mm inner and 60 mm outer edge). Label
`scale_checked_pot_rim (r)` only if `|r − 1| ≤ 0.03` and the fitted outline is square (side ratio within 0.95–1.05); otherwise
`conditional_on_unvalidated_scale`. No geometry is rescaled by r. Pot top `z_top` = 99th percentile of wall-surfel z; it is mapped to the
measured rim plane (grow z = 70 mm); the medium surface is `z_top − 10 mm`.

## 2. Optics and light
Optics = the engine's current setting: **black** (absorptance 1, τ = ρ = 0) for leaves and plant surface; absorbed = incident.
Three declared source routes:
1. **`engine_lamp`** (gates S1–S3 only): the snapshot run's `lamps.csv` — isotropic point lamp 1200 µmol s⁻¹ at (0, 0, 0.6) m,
   `E = P/(4π) · |cos β| / d²` with shadowing; absolute in the engine's own synthetic units (as in the dynamic run).
2. **`measured_field`** (observed and model plants in the cabinet): unshaded horizontal light `U(p) = F(p) / F_max` where `F` is the bounded
   `tools.empirical_light_field` interpolation of the 46 measured points and `F_max = 367.416` (the canopy-plane maximum), so values are
   **relative units (r.u.)**. Support: x 122.5–437.5, y 105–315, z 70–150 mm (grow, shelf floor); a receiver outside support is
   `outside_measured_support` (unavailable on this route) and its area fraction is reported.
   Shading directions: three bars along x, centred at x = 300 mm, y = 55 / 105 / 155 mm (centre bar on the canopy-plane peak row), emitting
   surface at z = 300 mm, 500 mm long — **user_assumed** (the bar pose is unrecorded); 8 jittered samples per bar, `cos^m` about −z with
   half-intensity half-angle 60° (user_assumed), plus a uniform diffuse dome (6 × 12 jittered directions). Diffuse fraction **0.6**
   (enclosure white, reflectance unavailable; 0.5 / 0.7 spread reported). The kernel is normalised to unit unshaded horizontal light, so
   `incident(p, n) = U(p) · K(p, n)` with K the two-sided cosine-weighted visible fraction.
3. **`bar_model_ratio`** (scenarios only): for a changed lamp height / lateral position, `U_s(p) = U(p) · M_s(p) / M_ref(p)` with `M` the same
   bar + dome kernel's unshaded horizontal value under each pose — the measured field carries the reference, the model only the change.

## 3. Quantities
- **Absorbed per receiver element** = incident × area; per leaf / plant = area-weighted sum (flux first, then divide by area).
- **Transmittance** at a point = light with the plant ÷ light with the plant removed (second trace, same kernel, same seed); never depends on U.
- **Height layers**: 5 mm bins of height above the medium surface. Per layer: leaf area, absorbed, and mean transmittance of horizontal point
  sensors on a 2 mm grid over the pot square at the layer's mid height. Extinction `k` from `ln T = −k · LAI_above` (least squares through the
  origin) — diagnostic only.
- **LAI** = one-sided leaf area ÷ ground area, both definitions: pot top (0.059² = 0.003481 m²) and tray cell (0.060² m² touching, or the
  scene's pitch²).
- **LUE** = gross ÷ absorbed PAR (gC mol⁻¹), per leaf and per plant. Gross per element uses the engine law unchanged
  `NRH(q) = (αq + Amax − sqrt((αq + Amax)² − 4θαqAmax)) / (2θ)` with q the absorbed PPFD of that element, × area × hours × 3600 × 12.011e-6
  (AMAX 13.57, ALPHA 0.048, THETA 0.92, hours 14 — registry values, `literature_other_cultivar`). On the `measured_field` route q needs an
  absolute scale: q = r.u. × F_max (meter scale) → LUE labelled `conditional_on_meter_scale`; on `engine_lamp` it is the engine's own scale.
- **Intercepted fraction** (cabinet) = plant absorbed ÷ unshaded horizontal light × pot-top area (both in r.u.).
- **Per-plant attribution** (tray scenes): three traces per plant — all plants, the plant alone, no plant (transmittance reference).
  `self_shading = 1 − alone/none`, `neighbour_shading = (alone − all)/none` (incident-weighted over the plant's receivers).

## 4. Noise and resolution
Static noise: every run is done with two seeds (receiver sample points on triangles, bar/dome jitter). `σ_static` = |run1 − run2| / √2.
GroIMP noise (snapshot gates): packet model `σ = sqrt(A · P/N)` per organ (A absorbed, P = 1200 µmol s⁻¹, N = 300 000 rays), **validated** on
a GroIMP three-seed sample on identical geometry (F5 control interval 0, seeds 11/12/13; `NOISE.json`); if the observed/model variance ratio
differs from 1, σ is scaled by its square root. A difference is **resolved** only if |Δ| > 2·sqrt(σ₁² + σ₂²); otherwise "not resolved".

## 5. Consistency gates (must pass before any other number is reported)
On the 59.9 DAG snapshot with `engine_lamp`, against the dynamic run's own interval-71 values:
- **S1 absorbed**: per plant, |static − dynamic| ≤ 2·sqrt(σ_GroIMP² + σ_static²); per leaflet the z-scores are reported (mean, sd, share within
  ±2). **S1b** (bias): over all 72 intervals' consumed geometry, Σ dynamic / Σ static within 1 ± 2·σ_pooled.
- **S2 LAI**: Σ leaflet triangle area == Σ `area_m2` of day-71 leaflets, relative 1e-4 (exact up to the area definition).
- **S3 LUE**: dynamic Σgross/Σabsorbed of interval 71 inside the 2σ band of static LUE computed with per-leaflet absorbed redrawn under the
  GroIMP packet noise (400 draws).
A surfel-path unit gate **S0** (synthetic): one flat square leaf as two triangles and as a dense surfel disc set must give the same
interception (≤ 2 %) and the same calibrated area (≤ 3 %).

## 6. Runs
1. Gates S0–S3. 2. S-obs single: each of the 10 observed plants alone at the cabinet position (280, 210) mm, reference lamp. 3. Model vs real:
R3-1b, n_04 and the snapshot plant p1 at the same position and scene. 4. Trays: n_04, t2_02, n_01, t2_05 (own surfels each, no copies) on a
2 × 2 layout centred at (280, 210), pitch 60 mm (touching) and 120 mm (sparse). 5. Scenarios on the touching tray, one factor at a time, same
seeds: lamp height 250 / 300 / 350 mm; lamp lateral position y-offset −50 / 0 / +50 mm; spacing 60 / 90 / 120 mm.

## 7. Static validation (V-S), declared now
Shading ratio = light just below a leaf ÷ just above it, same xy. Protocol + empty CSV in `handoff/MEASURE_SHADING_PROTOCOL_2026-10-26.md`.
Model side at the same points (±5 mm xy and z sampled, 200 draws, plus ray noise). Gate: median |Δratio| ≤ 0.10 **and** Spearman ≥ 0.6 across
points; leaf-specific points only for usable leaves, otherwise by height layer. The comparison script runs only when the CSV exists.
