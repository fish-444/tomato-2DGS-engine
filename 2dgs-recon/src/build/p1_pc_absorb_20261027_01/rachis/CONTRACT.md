# Contract — R8_pc_rachis: rachis-like (axis_kind 2) surfels around each compound leaf absorbed into that leaf, 2026-10-27

Request: user message "각 복엽 주위의 엽축 서펠도 흡수" (right after the "rachis-like ?" question about the light-grey surfels in the R8_pc viewer). Declared before any number of this job.
Labels: `R3-1b_unaccepted` · `conditional_on_unvalidated_scale` · `gate_failed_viewing_only`. No photo, SAM or V-P anywhere (point cloud + the user's own rachis traces only). R8_pc, R7 and every earlier output unchanged; result = new set `R8_pc_rachis`.
Assumption (stated, not asked): base = **R8_pc labels (set B)**, the latest viewer layer the request followed, not R7. The result is a separate layer, so switching the base later is a rerun, not a change.

## What is absorbed
- Targets: axis_kind 2 surfels not painted in R8_pc (10,722 − 623 blade-like already painted = 10,099). A target taken into a leaf is labelled as that leaf's **rachis part** (`axis2_absorbed`), not as blade; leaf blade areas do not change, rachis area is reported separately.
- Not absorbed: axis_kind 1 (main stem, 2,955), blade surfels (their R8_pc labels stay fixed).
- Seeds (labels fixed): R8_pc-labelled surfels within 3 mm of a target (nodes of the graph) and, for the 3D-surroundings term, all R8_pc-labelled surfels within 8 mm; plus **trace seeds** = targets within 3 mm of a human rachis trace, labelled with the nearest trace's leaf (traces exist for leaves 1–7; leaf 8 has none).

## Method (all from the reconstruction + the user's traces)
- Graph nodes = targets ∪ R8_pc-labelled surfels within 3 mm of a target. Edges = pairs ≤ 3 mm (no normal or sheet test: rachis is tubular), weight exp(−d / 2 mm); blade–blade pairs omitted.
- Prior per target and candidate leaf L (L has an R8_pc-labelled surfel within 8 mm or a trace point within 10 mm), E_L = 0.5 (1 − nb3d_L) + 0.3 tr_L + 0.2 dir_L:
  nb3d_L = distance-weighted (1 / max(d, 0.5 mm)) share of the R8_pc-labelled surfels within 8 mm that belong to L (rule 14: reconstructed-3D surroundings);
  tr_L = min(distance to L's trace, 10 mm) / 10; dir_L = 1 − |cos| between the surfel's longer-axis tangent (tu if scale0 ≥ scale1 else tv) and L's trace tangent, 0.5 when the trace is farther than 10 mm;
  a leaf without trace gets tr = dir = 0.5. Prior λ_L = exp(−3 E_L) normalised over candidates.
- Random walker with priors (as R8_pc): γ = 1.0; accept p1 ≥ 0.6 and p1 − p2 ≥ 0.25; accepted targets become seeds; at most 5 rounds.
- Unpainted reasons: `not_connected` (component without any seed), `no_candidate` (no candidate leaf), `contested` (else).

## Self-test (leave-one-trace-out; no weights tuned on it, no halves needed)
For each leaf with a trace (1–7): remove that leaf's trace from the seeds, from tr / dir (→ 0.5) and un-seed its trace-seeded targets; run the full absorption; score those hidden targets against the leaf the trace belongs to
(accuracy among accepted, wrong-assignment rate, coverage). Pooled and per leaf (leaves 4 and 5 reported explicitly). Other leaves' traces stay in place, so a wrong neighbour assignment is possible and counts.
**Pass gate (pooled):** accuracy among accepted ≥ 0.90 and wrong rate ≤ 0.05; coverage stated. Same gate as R8_pc.
Not independent of the user's traces by construction (they define the truth and the trace seeds) — only the non-trace evidence is tested; stated in the report.

## Outputs
`R8_pc_rachis/labels_rachis.npz` (label, source tag, p1, p2, top1, top2, R8_pc label), `results.json`; table with denominator = all 46,719 reconstructed surfels (axis2 painted / unpainted, axis1, blade) for R8_pc and R8_pc_rachis, counts and mm² (union of surfel ellipses per 10 mm cell);
per-leaf rachis counts and mm²; reasons for the unpainted rest; the 8833 read-only toggle "엽축 흡수 (R8_pc 위)" (R8_pc / after absorption / differences). Report `handoff/P1_PC_RACHIS_2026-10-27_KO.md` (+ EN).
Not done: no change to R8_pc / R7; no other individual before "채택"; artifact page only if the user asks (it is at its size limit).
