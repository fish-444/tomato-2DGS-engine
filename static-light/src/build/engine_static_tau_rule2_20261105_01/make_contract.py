"""CONTRACT.md + CONTRACT_HASHES.json of optics rule 2 (written before any light number; the stem table is geometry only)."""
import csv, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; B = ROOT / 'build'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = list(csv.DictReader(l for l in open(HERE / 'stem_definition.csv', encoding='utf-8') if not l.startswith('#')))
tab = ['| plant | d_stem 1.0 mm: surfels / area (b) share | 1.5 mm (main) | 2.5 mm | polylines |', '|---|---|---|---|---|']
for pid in dict.fromkeys(r['plant'] for r in rows):
    rr = {r['d_stem_mm']: r for r in rows if r['plant'] == pid}
    cell = lambda d: f"{rr[d]['stem_surfels']} / {100 * float(rr[d]['stem_area_b_share']):.2f} %"
    tab.append(f"| {pid} | {cell('1.0')} | {cell('1.5')} | {cell('2.5')} | {rr['1.5']['polylines']} |")
p = {r['d_stem_mm']: r for r in rows if r['plant'] == 'R3-1b'}
txt = f"""# CONTRACT: static optics rule 2 — τ 0.07 / a 0.85 on every surfel except the stem (14 plants), reference `v1tau2_smoothn_r2`

Written 2026-11-05, before any light number of this folder (the stem table below is geometry only). **Session: Engine static** (static tracer owner). Binding prompt: "CLI PROMPT — Static optics rule v2: τ 0.07 / a 0.85 on every surfel except the stem (all 14 plants), new reference v1tau2_smoothn_r2", 2026-11-05 16:40 KST (user decision 16:38), pasted by the user into the Engine (dynamic) session and forwarded verbatim to this session by the Engine v2 session.
Labels: `light_model_v1tau2`, `smoothn_r2`, `assumed_leaf_optics`, `relative_light`, `conditional_on_unvalidated_scale`, `skeleton_labels_ungraded`. Rule-1 outputs (`engine_static_tau_20261031_01`, `engine_static_v1tau_reference_20261104_01`, tray32 v3) are not overwritten; tray32 v3 and Engine v2 kernels are not stopped.

## 0. User decision (16:38), recorded
"줄기를 제외한 모든 서펠에 이 조건 적용": every non-stem surfel (leaflets, rachis, petiole, flowers, unassigned / unlabelled) gets τ 0.07 and a 0.85; stem surfels τ 0. Sensitivity τ 0.05 / 0.10 with the same rule. This replaces the rule-1 groups (pilot: stem τ 0; nine: every axis-like surfel τ 0; e_*: nothing τ 0), which stay on record as `v1_tau` (rule 1).
- **Absorptance:** a 0.85 on every receiving surfel, stem included. Rule 1 did the same (absorbed = U·K·A·0.85 on every receiver), so nothing differs here.
- **Crossing rule unchanged:** τ of a crossing = minimum τ of its members (eps 1.529 mm); same-sheet surfels count once.
- **Normals smoothed r = 2 mm** (receiver and occluder), d_ex 2.4 mm, seeds 101 / 202, 24 bar points / 72 dome directions, P1 area (b) — as `v1tau_smoothn_r2`.

## 1. Stem definition (declared; `make_stem.py`, `stem_definition.csv`, masks in `kaggle_flat/stem__<id>.npz`)
1. Pilot (R3-1b): the existing stem class, `axis_kind` 1 (`R8_pc_rachis/labels_rachis.npz`) — the rule-1 stem.
2. Other 13: stem = opaque surfels within **d_stem = 1.5 mm** of the stem polylines and **not** carrying a compound-leaf label (dz10 snapshot status 1, label > 0). Polylines:
   - main axis: `axis.curve_mm` of `build/mtg_build_v2_clicks_20261103_02/static/mtg_<id>_spec.json` (adopted surfel fit);
   - side axes: **no `_02` spec carries side axes**, so the clicked lateral axes of each plant's dz10 skeleton are used (`fit.curve_mm`, or the clicked points where an axis has no fit);
   - t2_* (not in the MTG set): main and lateral axes of their dz10 skeletons (`p1_dz10_labels_20261101_01/t2_0x_<skelver>/skeleton_v1.json`).
   Distance = to the polyline resampled at 0.05 mm. Sensitivity d_stem 1.0 / 2.5 mm (τ 0.07 only).
3. Pilot cross-check (report only): rule 2 applied to R3-1b with its `_02` spec axis (a vertical proxy through the median leaf base; the pilot has no clicked axis).

### Stem counts (geometry; before any light number)
{chr(10).join(tab)}

**Pilot cross-check:** the stem class covers {int(p['1.5']['stem_surfels'])} surfels = {100 * float(p['1.5']['stem_area_b_share']):.1f} % of area (b); the geometric rule 2 catches {p['1.5']['pilot_rule2_geometric_surfels']} surfels ({100 * float(p['1.5']['pilot_rule2_geometric_area_share']):.2f} %), IoU **{p['1.5']['pilot_iou_vs_stem_class']}** (d 1.0 / 2.5: {p['1.0']['pilot_iou_vs_stem_class']} / {p['2.5']['pilot_iou_vs_stem_class']}).
**Flag (reported, not acted on):** the geometric stem is 0.02–0.6 % of area (b) on the 13 plants at 1.5 mm, while the pilot's classifier stem class is 32 %; the two definitions are not comparable (the pilot axis is a proxy, and `axis_kind` 1 also holds stem-like petiole surfels). The declared values are run as given; options are listed in the report.

## 2. Runs (Kaggle CPU, ≤ 2 kernels; tray32 v3 finished, Engine v2 keeps its slots)
- **K12a / K12b (14 plants, split 5+5+4 → 7+7):** per plant, normals smoothed, arms: rule-1 τ map at 0.07 (**regression gate**), rule 2 (d 1.5) at τ 0.05 / 0.07 / 0.10, rule 2 d 1.0 and d 2.5 at τ 0.07. Plant relative light, self-shading, height thirds, per-leaf absorbed light, unassigned-light share; 5 mm-layer canopy transmittance for rule 2 d 1.5 τ 0.07; LAI unchanged (area).
- **Regression gate:** rule-2 code with the rule-1 τ map reproduces `v1tau_smoothn_r2` per surfel exactly: the two-seed kernel equals the stored K8 array (`engine_static_v1tau_reference_20261104_01/out_k8*/<id>/per_surfel.npz`, stored in float32) bit for bit in float32. Failure stops the kernel (reported).
- **K13 (after K12):** tray 60 / 120 mm self / neighbour / total shading and the 80 lamp / lateral / spacing scenario effects under rule 2 (sign and size vs rule 1 = K9 of the reference table).
- **K14 (after the reference passes):** tray32 v3 reference geometry with rule 2 (table A, flat, over trays, p 0.10 m, heights 0.30 / 0.40 / 0.50 / 0.60 m = v3 S25–S28), change vs v3 rule 1. A full v3 re-run only if the user asks.

## 3. Outputs
`stem_definition.csv`; `RESULT_TABLE_rule2.csv` with the new column `v1tau2_smoothn_r2` next to v1 / v1_tau / v1tau_smoothn_r2 and change % per item (also written into the reference table folder as a separate file, not overwriting it); tray32 reference-geometry comparison; Korean report `handoff/ENGINE_STATIC_TAU_RULE2_2026-11-05_KO.md` (≤ 10 lines + table) with a paste-ready Method sentence; LOG line, hashes, flow row, lock.

## 4. Do not
Change τ, a, d_stem, d_ex or the smoothing radius after seeing results; overwrite rule-1 outputs; stop tray32 v3 or Engine v2 kernels; call results measured or validated; skip a part on my own.
"""
(HERE / 'CONTRACT.md').write_text(txt, encoding='utf-8')
files = [HERE / 'make_stem.py', HERE / 'stem_definition.csv'] + sorted((HERE / 'kaggle_flat').glob('stem__*.npz'))
files += [B / f'mtg_build_v2_clicks_20261103_02/static/mtg_{p_}_spec.json' for p_ in ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07']]
files += [B / 'engine_static_tau_20261031_01/kaggle_data/labels_rachis.npz', B / 'engine_static_normals_v2_20261104_01/kaggle_flat/normals.py']
h = {str(f.relative_to(ROOT)).replace('\\', '/'): sha(f) for f in files}
json.dump(h, open(HERE / 'CONTRACT_HASHES.json', 'w'), indent=1); print('contract written,', len(h), 'hashes')
