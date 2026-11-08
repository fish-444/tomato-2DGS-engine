# Engine static — leaf transmission in the shadow-ray tracer (variant `v1_tau`) — CONTRACT (2026-10-31, before any number)

Labels on every output: `light_model=v1_tau` · `assumed_leaf_optics (tau=0.07, a=0.85, reflection dropped)` · `relative_light` · `conditional_on_unvalidated_scale` · `after_checkpoint_v1`.
Adoption is the user's decision. Nothing here replaces v1, and v1's frozen outputs are not written to.

## Pre-checks
- No `engine_static_tau_*` exists in `build/` or in LOG.
- There is no RUNNING.lock in `build/` (the stale engine B lock was released at 22:53 by another session, per LOG).
- This folder's `RUNNING.lock` was written now.
- **Compute is external only** (Kaggle CPU). Locally: file copies, small JSON and CSV, reports.

## Code
- `./static/` is a byte copy of `engine_static_20261026_02/static` (the v1 code used by every static run) plus **additions only**. The v1 functions are unchanged:
  - `tracer.hit_t`: the same tests as `hit`, returning t.
  - `tracer.transmit`: one shadow ray.
  - `tracer.cabinet_kernel_tau` / `trace_cabinet_tau`, a copy of `cabinet_kernel` with `occluded` replaced by `transmit`.
  - `tau_obs.trace_plants_tau`, a copy of `obs.trace_plants` without sensors.
- **Why τ = 0 equals v1 exactly:**
  - v1's `occluded` is any-hit over the grid cells the ray visits, with no cell-bound t limit.
  - `transmit` visits the same cells with the same hit tests and collects every hit.
  - With τ = 0 it returns 0.0 at the first hit and 1.0 when there is none.
  - The kernel adds `w·cb·T` only when T > 0, so T = 1 gives the identical float sum.

## Rule (declared)
1. **Crossings.** Hits of distinct primitives in (tmin, tmax) are sorted by t. Consecutive hits ≤ ε apart are chained into one crossing (one physical sheet).
   - ε = t_sheet = **1.529 mm**: median single-sheet thickness, `build/p1_thin_shell_20261026_01/metrics/R3-1b.json` → `thickness.median_T_mm`.
   - `second_layer_test.json`, the file the prompt names, has no thickness field. Its sheet-gap verdict (real second layers 2.5–10 mm away) is consistent with this value.
2. **Attenuation.** T = Π over crossings of τ_crossing, where τ_crossing = the minimum τ of its member primitives.
   - **R3-1b:** τ = 0.07 for leaf surfels (axis_kind 0 blade and 2 rachis/petiole-like, labelled or not). τ = 0 for stem (axis_kind 1, `R8_pc_rachis/labels_rachis.npz`).
   - **The nine individuals:** P1 has no stem class, so stem cannot be told from rachis. τ = 0 for `axis_like` surfels (3D-first `groups.npz`) and τ = 0.07 for all others. This errs toward v1 on axis-like surfels; stated.
   - No pot or board is in any scene.
   - Light at a receiver = Σ lamp/dome contribution × |cos| × T.
3. **Absorption.** absorbed = incident × a.
   - Main run: a = 0.85 (= 1 − τ − ρ, with ρ = 0.08 dropped; reflection is not traced).
   - Also reported: a = 1, which isolates the effect of τ.
4. **Unchanged from v1:**
   - geometry, area (b) (P1 silhouette per surfel) and placement;
   - the measured field and its model extension, and the user_assumed bars;
   - seeds 101/202, 24 bar samples and 72 dome directions;
   - d_ex = 2.4 mm (ray tmin, the receiver's own sheet);
   - the half-credit rule as declared (main off);
   - labels and credit buckets as in the R7/R8_pc job.
5. **Overflow.** A ray that collects more than 2,048 distinct hits is treated as blocked (T = 0) and counted. The count is reported.

## Tests (before any plant result); fixed pass rules
- **R0 regression:** τ = 0 and a = 1 must reproduce v1 **byte-identically** per surfel: `np.array_equal` on the four kernel parts for every scene traced (R3-1b and all nine, both seeds, d_ex 2.4).
  - Separately, R3-1b at d_ex 2.4 must equal the local v1 arrays (`engine_static_r8pc_20261028_01/per_surfel_light.npz`). If not: stop.
- **T1 analytic:**
  - Two parallel 20 × 20 mm disc-lattice squares (as gate S0 v2), the upper one offset 10 mm sideways, gaps 0, 0.5ε, 2ε and 5 mm. A vertical beam is traced from the lower square's disc centres with d_ex 2.4 mm.
  - Expected:
    - overlapped part: T = τ when the gap is ≥ 2ε, and T = 1 when the gap is ≤ ε (same sheet; the hits fall inside the receiver's own exclusion);
    - non-overlapped part: T = 1.
  - Extra case T1e: the upper square as two layers 1.0 mm apart (≤ ε) at gap 5 mm, expected τ (one crossing, not τ²).
  - Receivers within 0.5 mm of the overlap edge are excluded.
  - Pass: each case's mean T within 2 % (relative) of expected.
- **T2 duplication invariance (R3-1b):**
  - Variants: every surfel duplicated (copy jittered N(0, 0.2 mm) per axis); and 50 % thinned (fixed random seed 20261031).
  - Areas are renormalised so each label group keeps its silhouette area: duplicates get A/2; kept surfels get A × (group area / kept group area).
  - Pass: plant total absorbed and per-leaf absorbed (V-P cleanup leaves 1 and 7; R8_pc blade leaves 1–8) change < 2 % against the base, for `v1_tau` (τ 0.07). v1 is reported alongside.
- **T3 τ sensitivity:** τ ∈ {0, 0.05, 0.07, 0.10} on R3-1b and the nine: plant total and per height third (thirds of each plant's height range p0.5–p99.5). Reported only; τ is not fitted.
- **T4 GroIMP:** answer only, from code (no engine change). The extension path shader `organShader` → `Phong.setTransparency(Graytone(materialT))` in `tools/fsp_extensions_native.rgg` (lines 28–31). Answer: **yes**. The default seed runs use `BLACK`, so the static↔dynamic consistency gate stays defined at τ = 0.
- **Failure rule:** if R0, T1 or T2 fails, `v1_tau` is not used; report and stop. One labelled second attempt is allowed, with its reason written down.
- **Probe tool test:** the probe tool returns the below/above ratio for horizontal one-sided cosine probes. Under a near-vertical single-point beam with no diffuse, a probe just under a single disc sheet gives τ (v1: 0) within 2 %.

## Runs (only if the tests pass)
1. R3-1b, then the nine (static job order). For each: plant-level relative light, self-shading and the three height thirds under v1 vs `v1_tau` (a = 0.85 and a = 1). Per leaf for R3-1b (R8_pc blades, V-P leaves 1 and 7) and for n_04's clicked leaf 1 (lower bound). The other eight have no leaf labels (Stage 1 still waiting).
2. **Probe values:** no shading-pair or sensor positions are recorded (the shading CSV is empty, the sensor CSV has no positions, the 10-15 under-leaf records have no xyz). Only the probe tool and its test are delivered.
3. **Scenarios:** the tray 60 mm (n_04, t2_02, n_01, t2_05, assumed layout) baseline vs lamp 250/350, lateral ±50 and spacing 90/120; R3-1b alone vs the lamp scenarios. v1 and `v1_tau` both at d_ex 2.4. Relative Δ = scenario/baseline − 1 per plant and tray total, and for R3-1b per plant and leaves 1 and 7.
   - **stable:** same sign, and |Δ_tau| within ±25 % of |Δ_v1|.
   - **optics_sensitive:** anything else.

## Amendment 1 — labelled second attempt T2b (2026-10-31, user: "Yes, second attempt", after T2 failed)
- **Original T2 stays on record as FAILED:** dup max 5.72 %, thin max 9.12 % (kernel `static-tau-k1-20261031` v1, `out_k1/tests.json`).
- **Reason:** v1 itself, without transmission, moves −6.4 % / +9.6 % under the same duplication and thinning. Surfel density changes the black-occluder coverage. So the declared T2 measured a v1 geometry property, which no transmission rule can remove. It did not isolate what it was meant to test: whether transmission on overlapping surfels depends on surfel multiplicity.
- **T2b criterion (fixed now, one attempt):** transmission-attributable change = (1 + Δ_v1_tau) / (1 + Δ_v1) − 1, for the plant total and every leaf (V-P leaves 1 and 7, R8_pc blades 1–8), for both the duplicated and the thinned variant. Pass if every value is < 2 %.
- T2b is evaluated on the same deterministic traces as T2 (`T2_decomposition.json`), with no new run.
  - Result: dup ≤ 0.73 %, thin ≤ 0.41 % → **T2b PASS**.
- **v1's density sensitivity (−6 % / +8 %) is reported as a v1 geometry uncertainty.** It is not fixed here; v1 geometry is unchanged by rule.
- **Runs** then proceed as declared. Three Kaggle kernels:
  - k1b: R3-1b v1 / τ runs and lamp scenarios;
  - k2: the nine individuals, each with its own R0 byte check;
  - k3: tray scenarios.
