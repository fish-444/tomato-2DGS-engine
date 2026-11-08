# STATIC gates — v1 failure, diagnosis, v2 (2026-10-26)

## v1 (as declared in CONTRACT.md) — FAILED
`GATES_v1.json`, `gates_v1.log`: S0 fail (disc vs triangle interception 4.6 % > 2 %), S1 fail for p1 (static 0.782 vs dynamic 0.648
µmol/s, tolerance 0.112), S1b fail (57 intervals: Σdynamic / Σstatic = 0.909, σ 0.008), S3 fail for p1. S2 passed (area exact to 1e-8).

## Diagnosis
1. **Tracer is exact.** 2000 random leaf receivers at state 71: grid tracer and brute-force all-triangle occlusion disagree on 0.
2. **Lamp normalisation is right.** The engine's own empty-scene sensor grid (`sensors.csv`, 16 × 4 cm²) = 1.066 × analytic
   I cos/d² (σ ≈ 0.05).
3. **Not a small-triangle effect in GroIMP.** Pooling 16 states, the deficit looked size-dependent (small leaflets low), so a GroIMP
   probe was run (`groimp_mesh_probe.py`, `GROIMP_MESH_PROBE.json`, 30 M rays): isolated black 20 mm squares meshed at edge 10 → 0.25 mm
   and ten real state-71 leaflets (10–312 mm²) all match analytic within noise (ratios 0.97–1.02 squares, 0.93–1.12 leaflets ± 2σ).
   Hypothesis refuted.
4. **Cause: coplanar overlap of the engine's mirrored leaflet pairs.** At state 71, 19.5 % of the engine's leaf area lies within 1 µm
   of the mirror leaflet's surface (pair 2: 47 % of its area overlaps P2L↔P2R; pair 1: 16 %; terminal: 0 %). No L/R mesh is an exact
   duplicate (0/44 at states 30, 50, 71); the overlap is partial. 22.3 % of the static lit light falls on such receivers. GroIMP absorbs a
   ray reaching an overlap on one surface only; the static receiver test credited both. Expected static excess ≈ ½ × 22 % ≈ 11 %, the
   observed bias.
5. **S0 test construction.** v1 disc lattice `arange(-9.7, 9.7, 0.4)` mm had 49 asymmetric centres (−9.7…+9.5) and passed the
   Gaussian scale s instead of the footprint radius k·s, so it did not tile the square. **Area rule (b)** also double-counted: each 8 mm
   cube's union kept footprint pixels overhanging into neighbouring cubes (+2.5 % on the synthetic square; worse for larger real surfels).

## v2 rules (introduced after v1 was seen; geometry-derived, no fitted parameter)
- Gate static only: a receiver lying within ±1 µm (along its normal) of another primitive gets credit ½ (`tracer.coincident`).
  Raw v1 numbers are kept in `GATES.json` (`static_raw_v1_per_plant`, `sum_static_raw_v1`). Observed plants (surfels) are unaffected.
- S0 lattice: 50 × 50 centred cells, disc radius = k·s = cell half-diagonal.
- Area rule (b) v2: per cube, only raster pixels whose point on the cube's mean plane lies inside the cube are counted (exact partition).
  Synthetic squares: flat 404 mm² (true 400, the residual is the real scalloped overhang), 30° tilted 402 mm².

## v2 result — ALL PASS (`GATES.json`)
| gate | result |
|---|---|
| S0 | interception discs 0.4517 vs triangles 0.4509 (0.18 %); area_b 404 vs 400 mm² (1.0 %) |
| S1 (59.9 DAG, state 71) | p1 static 0.701 vs dynamic 0.648 µmol/s (−7.6 %, tol ±0.106); p2 0.924 vs 0.932 (+0.9 %, tol ±0.122); leaflet z mean −0.04, sd 0.88, 97 % within 2 |
| S1b (57 intervals) | Σdynamic/Σstatic = 1.008, σ 0.009 (v1 raw: 0.909) |
| S2 | triangle area = state area to 1.4e-8 |
| S3 | LUE p1 static 0.482, band [0.405, 0.459], dynamic 0.459; p2 0.479, band [0.414, 0.465], dynamic 0.446 (gC mol⁻¹) |

S3 note: the band comes from Poisson-resampling the static per-leaflet absorption; resampling lowers LUE (a concave leaf law on
packet-quantised small leaflets), so the static point value lies above its own band. The declared test (dynamic inside band) is kept.

## Finding for the DYNAMIC engine (not fixed here — dynamic outputs are frozen in this task)
Mirrored leaflet pairs (pairs 1 and 2 of library_v2 shapes) are placed coplanar and partly on top of each other. In every dynamic run
using these shapes, the overlapping area is counted twice as leaf area but receives light once, so pair-2 leaflets get roughly
half the light on their overlap. Whole-plant absorbed light per declared leaf area is about 10 % lower than the geometry implies (share grows from 0 to
22 % of lit light between intervals 15 and 71). Proposed to the DYNAMIC owner via handoff/LOG.md.
