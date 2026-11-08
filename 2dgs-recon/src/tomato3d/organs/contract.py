"""Declared constants of the organ-split contract (build/p1_organ_split_20261022_01/CONTRACT.md).

Every number here is fixed in CONTRACT.md; changes are contract revisions with a reason. Revision 4 (2026-10-22 19:0x):
the module was rebuilt after the session's written files were reverted (checkpoint rewind at 18:51); the data files of
revisions 1-3 (bench, features, labeller, TRAIN results) survived and the definitions below reproduce the contract text.
Features are recomputed with the definitions in features.py and the labeller is retrained on the TRAIN half only.
"""

from __future__ import annotations

from pathlib import Path

TAG = dict(input="R3_unaccepted", purpose="dev")
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "build" / "p1_organ_split_20261022_01"
SYNTH = ROOT / "build" / "p1_synth_microtom_20261022_01"
TOPO_RUN = ROOT / "build" / "p1_organ_topology_20261024_01"   # 2026-10-24 topology contract (CONTRACT_ORGAN_TOPOLOGY.md)
R3 = dict(run_dir=ROOT / "build" / "p1_recon_wholeplant_20261022_01",
          surfels=ROOT / "build" / "p1_recon_wholeplant_20261022_01" / "kernel_out" / "R3" / "surfels_full.ply",
          poses=ROOT / "build" / "p1_charuco_a3_20261021_01" / "poses.json",
          masks=ROOT / "build" / "p1_recon_wholeplant_20261022_01" / "masks",
          h2_labels=ROOT / "build" / "p1_recon_charuco_20261021_01" / "leaflets" / "labels",
          acceptance="not accepted (band-excluded IoU 0.788 < 0.80 on 060317552); cause: a mask miss against brown soil")
BENCH = dict(train=["ind_007", "ind_016", "ind_001", "ind_005", "ind_000_v4", "ind_003_v6"],
             test=["ind_008", "ind_032", "ind_012", "ind_002", "ind_008_v5", "ind_004_v6"])
POT_CENTRE_M = (0.1707, 0.1452, 0.066)        # OBJ origin = pot top centre; root of every skeleton (synthetic and R3)
# Step 2 - axis vs blade
FEAT_KS = (24, 96)
FEATURES = ["linearity_k24", "planarity_k24", "sphericity_k24", "thickness_mm_k24", "radius_k_mm_k24", "anisotropy_k24",
            "linearity_k96", "planarity_k96", "sphericity_k96", "thickness_mm_k96", "radius_k_mm_k96", "anisotropy_k96",
            "density_1p5mm", "density_3mm"]
LABEL_SMOOTH_MM = 1.5
LABEL_THR = 0.6                                 # axis when p_axis >= LABEL_THR (TRAIN sweep rev 3: 0.6 -> precision 0.68 / recall 0.90)
LABEL_AXIS = ("stem", "rachis", "petiolule", "side_stem", "peduncle", "pedicel")
LABEL_BLADE = ("blade", "cotyledon")
R3_FUSION = dict(stem_share_axis=0.40, blade_share_blade=0.60, vis_res=504, vis_tol_m=0.0015)
# Step 3 - skeleton
SKEL = dict(knn=10, max_edge_mm=3.0, bin_mm=2.0, root_link_mm=10.0, prune_spur_mm=3.0, bridge_mm=12.0, bridge_turn_mm=8.0,
            node_merge_mm=4.0, heaviest_tie=0.20, stub_mm=6.0,
            side_axis="a branch at a main-stem node is a side axis when >= 2 of its sub-branches are leaf-like "
                      "(each >= 10 mm long and carrying >= 3 blade clusters)", leaflike_mm=10.0, leaflike_clusters=3)
# Step 5 - leaflets
LEAF = dict(cluster_radius_mm=1.5, normal_cut_deg=40.0, min_cluster_pts=40, graph_knn=10, graph_cap_mm=4.0,
            ambiguous_margin=0.20, euclid_fallback_mm=8.0, intercalary_ratio=0.35, pair_window_mm=2.0, truss_apex_share=0.30)

# 2026-10-24 topology contract (build/p1_organ_topology_20261024_01/CONTRACT_ORGAN_TOPOLOGY.md, sections 4.1-4.4). DEV-tuned, frozen before TEST.
TOPO = dict(
    # 4.3 sympodial main stem: continuation by direction, measured on ring-fitted centres (circle fit of each vertex's members in the
    # plane normal to the local tangent; RANSAC, tolerance circle_tol_mm, >= circle_min_inliers), over dir_window_mm of path behind
    # the branch point and head_mm of the candidate child ahead. Subtree weight is not a criterion.
    dir_window_mm=15.0, dir_min_span_mm=4.0, dir_drift_max_deg=12.0, dir_sparse_mm=15.0, dir_min_pts=30, head_mm=6.0, stem_turn_max_deg=20.0, stem_turn_hard_deg=35.0, offset_growth_mm=0.8, gap_angle_max_deg=12.0, stem_global_tilt_max_deg=25.0, global_span_mm=15.0, stem_turn_tie_deg=8.0, circle_tol_mm=0.4, circle_min_inliers=8, ring_r_ratio=1.6, circle_r_mm=(0.2, 3.5),
    # junction split: members of a main-path vertex farther than radius + off_axis_margin_mm from the fitted stem line are branch bases
    off_axis_margin_mm=0.8, split_min_pts=3,
    # tube follower (4.3 on points): step, search radius before a ring exists, membership band, min points per step, gap reach
    tube_step_mm=3.0, tube_search_mm=4.0, tube_member_tol_mm=0.8, member_clean_mm=1.0, tube_min_pts=4, tube_gap_max_mm=25.0, stem_turn_sparse_deg=40.0, ring_accept_ratio=1.28, ring_grow_ratio=1.25, clean_ring_cov=0.9, ring_coverage_min=0.5, sparse_coverage_min=0.34,
    # 4.4 lateral-vs-rachis: organs on a branch's own path -> compound leaf (>= 2 clusters and >= leaflike_mm, or >= 3 clusters) or
    # truss (>= truss_min_mm of axis without a blade cluster); a lateral carries >= 2 organs, at least one a compound leaf
    truss_min_mm=6.0, petiolule_max_mm=4.0, lateral_min_organs=2,
    # 4.2 bridging (declared; stage 1 passed 12/12 on DEV baseline, so the old rule stays active until a plant fails there)
    bridge_long_mm=25.0, bridge_angle_deg=30.0, bridge_radius_ratio=2.0, bridge_short_mm=12.0,
    # 4.1 base (declared; stage 2 passed 12/12 on DEV baseline)
    base_height_mm=10.0, pot_radius_mm=35.0,
)
# Step 6 - pass rules (synthetic TEST, pooled)
PASS = dict(node_exact_min=0.70, node_mae_max=1.0, leaf_membership_min=0.85, leaflet_f1_min=0.70)

REVISION_4 = """
## Revision 4 (2026-10-22 19:0x) — code rebuilt after a checkpoint revert
`tomato3d/organs/` and `tests/test_organs*.py` were reverted on disk at 18:51 (the fork local_9aacbd0b was interrupted at
18:50). The bench, features, labeller and TRAIN results of revisions 1-3 survived. The module is rebuilt to this contract:
features recomputed with the written definitions (PCA eigenvalues l1 >= l2 >= l3 of the k-neighbourhood covariance:
linearity (l1-l2)/l1, planarity (l2-l3)/l1, sphericity l3/l1, anisotropy (l1-l3)/l1, thickness 2*sqrt(l3), radius = distance to
the k-th neighbour, densities = neighbour counts within 1.5 / 3 mm), labeller retrained on the TRAIN half only, all thresholds
as above. Geodesic distances for leaflet assignment run from every axis point of a leaf's own subtree (petiolules included).
"""

REVISION_5 = """
## Revision 5 (2026-10-22 19:3x, TRAIN only, before any TEST result) — gaps on thinned axes
Diagnosis on TRAIN with truth labels (membership 0.67, 22 % of blade clusters unattached): a rachis cut by a 4-10 mm gap was
bridged to the NEAREST reached point, which is often the stem 3 mm to the side or a neighbouring rachis; so one leaf became
2-3 'leaves' at the stem (ind_001 L06 -> L06+L07+L08), a distal rachis half hung from another leaf, and whole leaves whose
base gap was bridged into the trunk lost all their leaflets (ind_007 L06: 3,480 of 4,496 points unattached; 21 of 46
unattached clusters sit next to main-stem points). Three declared changes:
1. bridging prefers the collinear continuation: score = distance + 8 mm x (1 - |cos| of the angle between the bridge and the
   component's principal direction), still within 12 mm (bridge_turn_mm = 8.0; components < 5 points take the nearest point);
2. the Euclidean fallback for clusters without a geodesic path is measured from the cluster's NEAREST POINT to the nearest
   axis point of any leaf (8 mm; the first form used the cluster centroid, which is > 8 mm from the rachis for most leaflets);
3. a branch that carries no blade cluster is never a compound leaf (stub, or truss_candidate near the apex); a leaf needs >= 1.
"""

REVISION_6 = """
## Revision 6 (2026-10-22 19:5x, TRAIN only, before any TEST result) — the stem never ends in a leaf
With the fallback of revision 5 membership rose to 0.80 (truth labels), and the remaining whole-leaf losses had one cause:
in every TRAIN plant the main-stem descent reaches a vertex whose only child is the top leaf's rachis (the stem apex above
it has no axis points after thinning) and runs out along that rachis; its leaflets become unattached and its node is lost
(ind_003_v6 L05: all 179 rachis points on the main path). Turning angles cannot decide it: smoothed over 10 mm, stem
continuations turn up to 51° (p90) while the entered rachis turns only 16-31°. What decides it is what hangs from the segment:
a stem apex carries trusses or nothing, a rachis carries leaflets. Rule: the main stem ends at the last node that carries a
compound leaf or a side axis; if the segment beyond it carries blade clusters, the path is cut there and the segment becomes
a branch of that node (then classified as usual; flag `apex_cut`), at most 3 rounds. A segment without clusters stays the apex.
"""
