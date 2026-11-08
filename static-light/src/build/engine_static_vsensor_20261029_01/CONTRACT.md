# Static light vs. PAR sensor readings, tray 1 (e_01–e_06, four faces) — CONTRACT (2026-10-29, before any model computation)

Labels: `validation_comparison` · `lamp_assumed` (until a measured lamp position is given) · `relative_light` · `conditional_on_unvalidated_scale` · `biological_validation = not_established`.

## Pre-checks
- No `engine_static_vsensor_*` existed in `build/` and none appears in LOG. The only other lock belongs to engine B (`build/engine_dynamic_20261026_01`). `RUNNING.lock` was written here.
- `Vsensor20261029110933.xlsx` was copied unchanged from Downloads to `handoff/` (sha256 identical) and is read from there.
  - Sheet1 is A1:D6, with no header and the cells stored as text. Its values equal the prompt's table.
- Rows are e_01…e_06; columns are yellow, green, red, blue (user-confirmed order). The unit is `unit_unconfirmed`.
- `ppfd_unshaded_ref = 282` (user statement). Its position, height, point vs mean, and plants present or not are all unknown.

## Data-only facts (no model)
- **Above the reference:** r_meas = ppfd / 282 exceeds 1.0 for **three** readings, not four as the prompt states: e_03 blue 1.077, e_04 yellow 1.052, e_04 red 1.123. They are not clipped.
- **`low_signal` (< 10):** six readings. e_01 yellow 7.19, e_01 red 2.73, e_01 blue 2.52, e_02 red 6.47, e_04 blue 3.37, e_05 blue 5.76.

## Step 2 — reconstruction availability (read-only)
- `build/p1_recon_d1028_20261028_01/` holds only Stage 2 (board poses and QC) for e_01–e_07.
- **No surfel scene, no mesh and no scale exist for e_01–e_06.** Without geometry, Step 3 (model values) cannot run for any of them.
- e_07 is not touched.

| ind | recon_quality (stage 2 triage) | accepted views | azimuth coverage | largest gap | scale_status |
|---|---|---|---|---|---|
| e_01 | marginal | 42 | 333° | 26.7° | no reconstruction (no scale) |
| e_02 | marginal | 20 | 288° | **71.9°** | no reconstruction (no scale) |
| e_03 | marginal | 31 | 329° | 31.0° | no reconstruction (no scale) |
| e_04 | marginal | 42 | 321° | 38.7° | no reconstruction (no scale) |
| e_05 | reconstructable | 52 | 340° | 19.9° | no reconstruction (no scale) |
| e_06 | marginal | 38 | 328° | 32.1° | no reconstruction (no scale) |

- e_02's gap lies roughly between board azimuths 154° and 217°. This is about the centroid of the accepted cameras, which gives 63° there; the report's 71.9° uses its own centre.
- Faces facing into that gap will be marked `in_azimuth_gap` once each face's direction is known. Until then no face can be marked.

## Step 3 (when unblocked) — fixed now
- **Scene:**
  - tray-1 scene with e_01–e_06 at their real cells;
  - lamp as in the static runs (`lamp_assumed`) unless a measured position is given (`lamp_measured`);
  - light settings identical to the nine-individual run: v1 black leaves, d_ex 2.4 mm plus 1.5/2/3/5, seeds 101/202, analytic tracer with 24 bar samples and 72 dome directions.
- **Sensor:** a 10 × 10 mm flat one-sided receiver unless the head area is given, at the stated position and normal, reporting incident relative light. If a receiver cannot be placed, the label is `proxy_receiver`.
- **Compute:** Kaggle CPU only.

## Criteria — declared before any computation, unchanged after
- **C4 (primary, no free parameter):** r_model = light at the sensor with plants ÷ light at the same position and orientation with all plants removed, compared with r_meas = ppfd / 282.
  - Pass: median |r_model − r_meas| ≤ 0.10 **and** Spearman ≥ 0.6 over the 24 positions.
  - r_meas > 1 cases are reported separately.
  - If the 282 position is given, r is also reported relative to the model value there.
- **C1:** Spearman ρ, model vs measured over 24. Pass ≥ 0.6. Within-individual ρ and Kendall τ are descriptive.
- **C2:** per individual, each face ÷ that individual's brightest face. Pass: median |Δ log₁₀| ≤ 0.30.
- **C3:** ratio of individual means; report only.
- **Factor k:** one global multiplicative k, for C1–C3 only.
  - Split fit: fit on e_01/e_03/e_05 and test on e_02/e_04/e_06, then the reverse; both are reported.
  - k is never fitted on all 24.
- **Metrics are reported with and without `low_signal`.** `in_azimuth_gap` faces are reported separately, not dropped.
- **Diagnostics are descriptive:** factor > 3 disagreements and what lies between sensor and lamp; the e_04 blue face; sensitivity to d_ex and to ±20 mm sensor height.
- **Not allowed:** choosing lamp, d_ex, optics, ray count, scale or labels to improve any criterion. Any such change is a new labelled run, tested only on new measurements.
