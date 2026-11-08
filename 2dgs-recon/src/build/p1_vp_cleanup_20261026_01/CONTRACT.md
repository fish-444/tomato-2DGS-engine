# CONTRACT — P1 V-P clean-up: fill gaps inside one colour, absorb detached pieces into the nearby cluster

Declared 2026-10-26 before any number. Human instruction (verbatim, two viewer screenshots): "이런식으로 띄어진데 점이 안칠해진곳을 색칠,
2. 이런식으로 멀리 떨어진 곳은 같은 복엽으로 묶지말고, 가까운 군집 밀도에 흡수". Every output: `R3-1b_unaccepted`,
`conditional_on_unvalidated_scale`, `grouping: user_labelled_propagated_cleaned`. Local CPU, GPU 0.

Base = 10-25 V-P, map v3 (`build/p1_compound_userlabel_20261025_01/map_v3_charuco/user_labelled_propagated/`) — the best held-out result so far.
Graph = blade surfels, k = 10 within 3 mm (as the re-cluster contract). Scales from the leaflet prior only: leaflet length 18.6 mm, width ≈ 11 mm
(18.6 / 1.7). Nothing is fitted on R3-1b.

## A. Detached pieces → nearby cluster ("멀리 떨어진 곳 … 가까운 군집 밀도에 흡수")
For each leaf L: connected components of its surfels on the graph; the **main body** = its largest component. A component is **detached** if its
minimum 3-D distance to the main body is > **11 mm** (one leaflet width: farther than that no blade of the same leaf connects it). A detached
component goes to the label with the **most labelled surfels within 5 mm** of it (other leaves only — the "nearby cluster density"); if there is
none, it becomes unassigned. Applied once, to every leaf, against the base labels.

## B. Gaps inside one colour ("띄어진데 점이 안칠해진곳을 색칠")
After A: connected components of the **unassigned** blade surfels. A component is filled with label L when (i) ≥ **80 %** of its labelled graph
neighbours carry L and (ii) its extent (max pairwise 3-D distance) ≤ **18.6 mm** (at most one leaflet; larger empty areas are not gaps). Otherwise it
stays unassigned. Applied once.

## Score
Held-out photo 060551789 (original truth, map v3), once, for: base V-P, A only, A + B. Gate unchanged (median per-leaf IoU ≥ 0.60, max confusion ≤ 0.10);
usable = IoU ≥ 0.55 and confusion ≤ 0.10. This is one more method scored on the same held-out photo (selection risk stated in the report).
Counts reported: pieces moved (per leaf from → to, surfels), surfels un-set, gaps filled (per leaf).

## Prohibited
Editing labels or traces, changing the gate, tuning on the held-out photo, growth beyond A/B, `microtom_synth_v2`, deleting files or db entries.

## C. Fill the rest (human instruction 2026-10-26, verbatim: "이 방식대로, 아직 안채워진 곳도 진행"; declared before any number)
After A + B, the remaining unassigned blade surfels are filled from the surrounding clusters, outside → in, on the same 3 mm graph:
in each round, an unassigned surfel with labelled graph neighbours takes the majority label among them if that label holds ≥ **60 %** of them;
otherwise it waits (ties never resolve by force). Rounds repeat until nothing changes, but a surfel is filled only if its geodesic distance to the
nearest surfel labelled before C is ≤ **37.2 mm** (two leaflet lengths); farther surfels, and components with no labelled neighbour, stay unassigned.
Scored once more on the held-out photo as state **A + B + C** (selection risk stated); the viewer set "복엽 정리" then shows A + B + C.

## D. Geometry-only completion (human instruction 2026-10-26, verbatim: "이런부분 정답 sam이 아니어도 클러스터링 진행, 다른 부분도"; declared before any number)
Finding behind "노란색안에 다른색이 섞이는 이유": the 96 purple/red surfels inside the yellow (label 3) body were V-P *unassigned* (propagated photos
voted 3 : 1 : 6 ≈ 5.0 : 3.4 : 2.5 per surfel, share < 0.6); 28 purple surfels absorbed next to them by A then seeded C's majority fill. Rule D decides
at cluster level and needs no SAM/propagated label:
1. **Islands**: for each leaf, connected components (3 mm graph) other than its main body whose outside neighbours carry one other label for
   ≥ **70 %** of their boundary take that label (applies to A's absorbed pieces too; no distance test).
2. **Remaining unassigned** (after A, B, C, including those beyond C's 37.2 mm limit): connected components of unassigned blade surfels; each joins the
   leaf holding ≥ **50 %** of its boundary edges on the 3 mm graph, provided the component's height range overlaps that leaf's p5–p95 height range;
   otherwise it becomes its own cluster **U<n>** ("geometry-only, no label"; never a leaf number, rendered as no leaf in the score). Components with
   < 20 surfels and no qualifying leaf stay unassigned.
Order A → B → C → D. State **A + B + C + D** scored once on the held-out photo (selection risk stated); U clusters are reported, not counted as leaves.

### D revised (human instruction 2026-10-26: "새 계약으로 선언해 큰 덩어리 이동을 막으면, 보라 섬 같은 작은 섞임만 정리만 진행 (이번 진행 전 버전에서"; declared before running)
Same as D, applied to the A + B + C state, with one addition: an island is moved only if its extent (max pairwise 3-D distance) ≤ **18.6 mm** (one leaflet)
**and** it has < **300** surfels (a leaflet ≈ 18.6 × 11 mm at the observed 1.5 surfels/mm² ≈ 300). Larger enclosed components are left as they are and listed
("large enclosed lobe — needs a human decision"). The unassigned-component rule (D.2) is unchanged. State **A + B + C + D′** replaces D as the viewer's
"복엽 정리"; D's result is kept in `scores.json` under `held_out.ABCD_unbounded` for comparison. Scored once (selection risk stated).
