# ADAPTERS — trait → model parameter conventions (Amendment 1 §2)

Status: our engine side is **read from code**. The one-leaf engine run is `pending` (Kaggle CPU; no local JVM per STATE_NOW §2.4).
The B-model side (`LEAF_ANGLE_D`, `Curvature_LEAF` in `organs.rgg`) belongs to the B-model session
(`build/bmodel_mtg_20261101_01`) and is not verified here.

## Our engine (`tools/vtc_development_native.rgg`, V3 native; read-only)

| Code | What it does |
|---|---|
| `:206` | Compound leaf (kind 4): `o.elev = 90 − COMPOUND_INSERTION_DEG`. `elev` = elevation above horizontal (`:199` tip = len·cos e …, sin e for z). So **insertion is from vertical**, and θ0 = 90 − insertion. Hand mean θ0 26° → insertion 64°. |
| `:194` | Children start from the parent's `elev` (cumulative). |
| `:207–209` | Petiole (kind 7, `PETIOLE_M`) has no bend, so it runs straight at θ0. The rachis is m = `pairs(rank)` straight segments (kind 8) of `RACHIS_M/m`, each with `o.elev −= segmentBend(o, m)`. |
| `tools/fsp_extensions_native.rgg:19–24` | `segmentBend` gives weights (pair 1 → 1/(2m−1), others → 2/(2m−1)), which sum to 1. Total bend = `TOTAL_BEND_DEG` = Δθ. The last segment points at θ0 − Δθ = θ_tip. |

Consequences:
- The bend is **progressive** (a piecewise-straight arc), not a single kink, when m ≥ 2.
- **With m = 1 (rank 1 under `PAIRS` [1,1,…]) the whole rachis is one segment at θ0 − Δθ, which is effectively a kink at the
  petiole tip.**
- The first rachis segment points at θ0 − Δθ/(2m−1). An exact midpoint arc would give θ0 − Δθ/(2m). The difference is small for
  m ≥ 2.
- Measured θ0 (first 10 mm of the skeleton) mainly samples the petiole, which matches `COMPOUND_INSERTION_DEG` directly.

## Leaflet count vs `PAIRS`
Measured counts:
- rank 1: 7 leaflets ×4, 5 ×2;
- overall: 7 → 57 %, 5 → 39 %, 6 → 5 %.

The engine's `PAIRS` choices ([1,1,2] / [1,1,3] pairs by rank, i.e. 3–7 leaflets with 3 at ranks 1–2) give fewer leaflets on
low ranks. This is noted only; nothing is changed (engine edits are outside this session).

## Pending
- One-leaf engine test on Kaggle. Check:
  - insertion 64°, bend 30°, m = 1 / 2 / 3;
  - segment elevations equal the formula above;
  - tip position matches the model leaf of `recon_traits.arch_curve`.
- B-model conventions: owned by the B-model session.
