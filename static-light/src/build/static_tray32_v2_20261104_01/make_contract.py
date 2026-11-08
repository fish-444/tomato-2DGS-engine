"""Write CONTRACT.md (declared before any number) and CONTRACT_HASHES.json for the 32-plant tray scenes v2."""
import csv, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; B = ROOT / 'build'
PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05', 'e_01', 'e_04', 'e_05', 'e_07']


def table_md(name):
    rows = list(csv.DictReader(open(HERE / f'placement_{name}.csv', encoding='utf-8')))
    out = ['| copy | plant | copy index | tray | row | column | tray class | row class | azimuth (deg) |', '|---|---|---|---|---|---|---|---|---|']
    out += [f"| {r['copy_id']} | {r['plant']} | {r['copy_index']} | {r['tray']} | {r['row']} | {r['column']} | {r['tray_class']} | {r['row_class']} | {r['azimuth_deg']} |" for r in rows]
    return '\n'.join(out)


meta = json.load(open(HERE / 'placement_meta.json'))
txt = f"""# CONTRACT: 32 plants in four 4×2 trays — spacing × luminaire shape × luminaire position × height (v2)

Written 2026-11-04 (18:35), before any number of this folder. **Session: Engine static** (the owner of the static tracer; the tracer is used read-only as byte copies). Binding prompt: "CLI PROMPT — Static: 32 plants in four 4×2 trays — spacing × luminaire shape (flat / V 120° / V 90°) × luminaire position (over trays / over gaps) × height (v2, replaces the 17:28 version)", 2026-11-04 17:36, with the over-gaps layout amended 18:07 (file `handoff/CLI_PROMPT_STATIC_TRAY32_V2_FOUR_TRAYS_FLAT_V_LAMPS_2026-11-04.md`).
The 17:28 version (`CLI_PROMPT_STATIC_TRAY32_LAYOUT_LAMP_SCENARIOS_2026-11-04.md`) was **never started**: no `build/*tray32*` folder and no LOG line existed before this one, so there is nothing to stop.
Labels on every output: `light_model_v1_tau` (τ 0.07, a 0.85, reflection dropped, stem / axis-like τ 0), `assumed_luminaire_model`, `repeated_placement`, `relative_light`, `conditional_on_unvalidated_scale`, `skeleton_labels_ungraded`, `scenario_comparison_not_validation`. Surfel normals are raw (no smoothing), as the label `light_model_v1_tau` says; a smoothed-normal arm can be added later on request.

## 0. User decisions (17:25–17:38; over-gaps amended 18:07), recorded
32 plants = the 14 reconstructed plants placed repeatedly (no model-generated plants). Outputs: (1) **interception efficiency** = light absorbed by all plants ÷ light emitted by the luminaires; (2) **per-plant and per-leaf absorbed light**, self-shading and neighbour-shading shares. Instantaneous (one fixed geometry per scene).

## 1. Layout (declared; metres, grow frame: floor = pot bottom z = 0)
- Tray = 4 × 2 plants: 4 rows along the tray's long axis **y**, 2 columns across **x**. Plant spacing **p** = **0.06 / 0.10 / 0.14 m** (centre to centre, both directions). Tray outline = half a spacing beyond the outer plant centres → **2p (x) × 4p (y)**.
- **Four trays in one row** along x, long axes parallel; **gap between outlines = 0.10 m** in every scenario. Tray i = 1..4 has its centre at x_i = (i − 2.5)·(2p + 0.10), y = 0. Plant (row r = 1..4, column c = 1..2) of tray i sits at x = x_i + (c − 1.5)·p, y = (r − 2.5)·p.
- **Heights:** pot rim z = 0.070 m, **substrate surface z = 0.060 m** (the grow frame of every static run: rim 70 mm − 10 mm; the plant files use 60 mm, not 70 mm, so their value is used and stated). Plants keep their reconstructed z (board z taken as grow z, as in all earlier static runs); each plant is rotated about the vertical axis through its stem-base proxy (`obs.base_xy`) and moved so that this point sits on its slot.
- Plants: R3-1b (pilot), n_01–n_06, t2_02, t2_04, t2_05, e_01, e_04, e_05, e_07 (14). Optics per surfel as in the v1_tau reference: τ = 0.07 on leaf surfels; τ = 0 on stem / axis-like (R3-1b: `axis_kind` 1; nine: `axis_like`). **e_\\*: no stem class → τ = 0.07 on every surfel (main)**; bracket arm (report only, reference scene, table A): τ = 0 on the unassigned (label-0) e_* surfels.
- Leaf labels: R3-1b = R8_pc (`labels_rachis.npz`); the other 13 = the dz10 snapshots of the `_02` MTG set (status 1 = assigned, lower bound; labels ungraded). Unassigned-light share = the primary definition of the v1_tau reference table.

## 2. Placement tables (declared, seed {meta['seed']}; `make_placement.py`, accepted at attempt {meta['attempts']})
Two tables **A** and **B**; in each every plant appears 2 or 3 times (14 × 2 = 28 + 4 plants a third time, chosen by the seed: A {', '.join(meta['third_copy_A'])}; B {', '.join(meta['third_copy_B'])}); each copy has a random azimuth (uniform 0–360°, seeded). Position classes: tray outer (1, 4) / inner (2, 3); row end (1, 4) / middle (2, 3). Constraints (checked by the generator): across A and B every plant has ≥ 1 copy in an outer and ≥ 1 in an inner tray, ≥ 1 in an end row and ≥ 1 in a middle row; in one table no plant has two copies in neighbouring slots (8-neighbourhood) of the same tray.
### Table A
{table_md('A')}
### Table B
{table_md('B')}

## 3. Luminaires (declared)
- **Length 0.60 m along y**, centred on the trays in y (y from −0.30 to +0.30 m). **Output per luminaire = 1 L** (unit flux; all efficiencies are ratios). Same output in every shape.
- **Flat:** one horizontal panel 0.10 m (x) wide, emitting downward (normal −z). **V 120° / V 90°:** shape ∨, vertex line at the bottom, two wings each **0.05 m wide along the wing**, inner angle 120° / 90° between the wings. The wing rises outward at φ = 30° (120°) / 45° (90°) above horizontal; its emitting (lower, outer) face has the normal (±sin φ, 0, −cos φ), i.e. tilted φ from vertical, pointing down and outward. Output split equally between the two wings (0.5 L each).
- **Emission:** the existing bar model, cos^m about the face normal with half-intensity half-angle 60° → **m = 1.0** (`light_consts.json`); a face emits only into the half space in front of it. Each face is sampled by a regular grid with spacing 0.0125 m (48 points along y; 4 across a 0.05 m wing; 8 across the 0.10 m flat panel) → **384 sample points per luminaire in every shape**, equal flux φ0 = 1/384 L each; point intensity I0 = φ0 (m + 1)/(2π).
- **Position (scenario):**
  - **Over trays:** one luminaire centred above each tray centre line (x = x_i), 4 luminaires = 4 L (1536 sample points).
  - **Over gaps:** one full luminaire above each of the 3 gap centres, **plus one single face at each outer edge** of the row, its lowest edge 0.05 m outside the outer tray outline (x = x_1 − p − 0.05 and x_4 + p + 0.05) and facing inward: a single V wing (0.05 m wide, same tilt as that V shape, normal down and inward, rising toward the trays) with **0.5 L**. 3 + 2 × ½ = 4 L (3 × 384 + 2 × 192 = 1536 sample points).
  - Flat shape, over gaps (my extension, as the prompt states; the user may change it): the outer-edge pieces are half-width flat panels (0.05 m, horizontal, 0.5 L) centred 0.05 m outside the outline.
  - Over-gaps wings of the V shapes at the gap centres: the left wing of a gap luminaire rises toward −x and lights the left tray, the right wing the right tray (vertex line at the gap centre).
- **Height:** z of the lowest point (flat panel / V vertex or lowest edge of an edge face) = **0.40 m or 0.50 m** above the floor.
- Source = luminaires only (no diffuse dome, no measured field) in the main runs. Luminaire housings do not occlude (not modelled).

## 4. Light model (v1_tau tracer, unchanged byte copies; new code only in this folder)
- Occluders: every surfel disc of every plant in the scene; tracer = `tau bundle` (`static-tau-bundle-20261031`): crossings chained at eps 1.529 mm, T = ∏ τ_crossing, overflow > 2048 hits = blocked (counted), exclusion distance d_ex 2.4 mm for the receiver's own sheet, transmission rule as in the tau contract.
- Receivers: surfel centres, two-sided |cos|; per copy **n_r = 8000** receivers drawn with probability ∝ area (b) (systematic PPS, seeded); each receiver stands for A_plant/n_r of area (P1 silhouette area (b)). Per receiver **K_e = 32** emitter points drawn uniformly at random from the scene's sample points (seeded per scene) and weighted by S/K_e; absorbed = a · E · T · cos · area, E = I0 · cos^m(θ_emission) / d², a = 0.85 (and a = 1 reported).
- Isolated copies: the same receivers and the same emitter draws, occluders = that copy only.
- **Absorbed power** per scene = Σ over copies; **interception efficiency η = absorbed ÷ emitted (4 L)**.
- Light reaching the tray outlines = unshaded horizontal-plane irradiance on the substrate plane (z = 0.060 m) integrated over the four outlines (quadrature 0.01 m); reported share = that ÷ emitted, and plant-absorbed ÷ that.
- **Energy closure per scene:** emitted = up (emission into the upper half space of each face, analytic quadrature) + beyond (flux that the unshaded floor region misses; region = row outline ± 0.50 m in x and y ±0.60 m) + floor_shaded (floor plane z = 0 in the region, traced) + removed_by_plants (unshaded − shaded floor flux in the region). Compared with absorbed + reflected (ρ = 0.08, dropped by the model) the **residual** (fraction of emitted) is reported; a closed chain model gives 0 (single sheet: removed = 1 − τ = a + ρ), so the residual diagnoses the area (b) versus sheet-footprint mismatch.
- Diffuse sensitivity (table A, spacing 0.10 m): the earlier uniform diffuse share 0.6 added as an isotropic dome (72 directions, traced) with horizontal irradiance 0.6/0.4 × the mean unshaded direct horizontal irradiance over the tray outlines; reported as absorbed(direct + diffuse) ÷ absorbed(direct) and the diffuse part's own interception per horizontal-irradiance area.

## 5. Scenes
3 spacings × 3 shapes × 2 positions × 2 heights = **36 scenes per table**, tables A and B → 72. Order: **table A all 36 → report → table B**. Isolated-copy runs (neighbour shading): table A all 36 scenes; table B if time allows (stated in the report). **Reference scene: flat, over trays, 0.40 m, p = 0.10 m**, same table. Scenario effects = paired differences vs the reference scene of the same table.

## 6. Measures (all per scene)
Interception efficiency; share reaching the tray outlines; plant-absorbed ÷ outline light; per copy: absorbed light, absorbed per leaf area (b), self-shading (1 − absorbed/absorbed without occluders, same receivers, unshaded), neighbour-shading share (1 − in-scene absorbed ÷ isolated absorbed), unassigned-light share; per leaf: absorbed light and rank (indicative; labels ungraded); position effects (outer vs inner tray, end vs middle row, paired on the same plant across copies and tables); uniformity among the 32 copies (median, 10–90 % range, Gini).

## 7. Checks before reporting (fixed pass rules)
- **C1 point-source limit:** one plant (R3-1b) alone, one point emitter straight above (5 m), m = 0, τ = 0 and τ = 0.07: per-surfel shaded irradiance equals the static tracer's `trace_lamps` / the v1_tau lamp kernel of the closure test (K67 `point_lamps_tau`), max relative difference ≤ 1e-9.
- **C2 emission integral:** numerical integration of every face's emitted flux over the sphere (cos^m, front half space) = the declared output; total within ± 0.5 % for every shape, position, height.
- **C3 energy closure** per scene (residual reported, not a pass/fail).
- **C4 ray-count convergence** on the reference scene: n_r and K_e doubled until the efficiency changes < 0.5 %; if not converged at the declared counts the declared counts are raised for all scenes before any scene result is read (declared change, written to an amendment).

## 8. Compute
Kaggle CPU only (heavy). **At most 2 kernels at a time, started only after the Engine v2 test kernel has started** (checked in the kernel list before every push). The account limit of 5 CPU sessions is shared with the v1_tau reference jobs of this session; tray32 kernels wait for a free slot. Local use: code tests on one or two plants (minutes, < 1 GB), tables, figures, reports. No secrets in files or logs.

## 9. Outputs
`placement_A.csv`, `placement_B.csv`, `luminaires.csv`, `scenes.csv`, `efficiency_by_scene.csv`, `plants_by_scene.csv`, `leaves_by_scene.csv`, `position_effect.csv`, `scenario_effects.csv`; figures (layout top view with luminaires per position option and spacing; efficiency by shape × position × height, one panel per spacing; grid heat map of copy absorbed light for the reference and the best / worst scene; outer vs inner and end vs middle paired plots; side-view sketch of the three luminaire shapes with emission directions); Korean report `handoff/STATIC_TRAY32_V2_2026-11-04_KO.md` (≤ 12 lines + tables, paste-ready Result sentence); LOG line, hashes, flow row, lock.

## 10. Do not
Add model-generated plants; change layout, luminaire geometry, scene grid, reference scene or placement tables after seeing numbers; use the measured 46-point field; call results measured or validated; use more than 2 Kaggle slots or start before the Engine v2 test kernel has started; drop scenes on my own (time limits and options are reported).
"""
(HERE / 'CONTRACT.md').write_text(txt, encoding='utf-8')
files = [B / 'engine_static_tau_20261031_01/kaggle_data/tau_config.json', B / 'engine_static_nine_20261028_01/kaggle_data/light_consts.json'] + sorted((B / 'engine_static_tau_20261031_01/kaggle_data/static').glob('*.py'))
files += [B / 'engine_static_tau_20261031_01/kaggle_data/labels_rachis.npz'] + [B / f'engine_static_nine_20261028_01/kaggle_data/{p}/surfels.ply' for p in ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05']]
files += [B / f'engine_static_nine_20261028_01/kaggle_data/{p}/static_surfels.csv' for p in ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05']]
files += [B / f'engine_static_nine_20261028_01/kaggle_data/{p}/groups.npz' for p in ['n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05']]
files += [B / f'engine_static_normals_20261104_01/kaggle_flat/{n}' for n in sorted(p.name for p in (B / 'engine_static_normals_20261104_01/kaggle_flat').iterdir() if p.is_file() and (p.name.startswith('e_') or p.name.startswith('labels__')))]
files += [HERE / 'placement_A.csv', HERE / 'placement_B.csv', HERE / 'placement_meta.json', HERE / 'make_placement.py']
h = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
json.dump(h, open(HERE / 'CONTRACT_HASHES.json', 'w'), indent=1); print(len(h), 'inputs hashed; contract', len(txt), 'chars')
