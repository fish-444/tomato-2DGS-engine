"""Shared pipeline (synthetic bench and R3): labels -> skeleton -> main stem -> nodes -> leaves -> clusters -> assignment."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import contract as C
from tomato3d.organs import leaves as LV
from tomato3d.organs import skeleton as SK
from tomato3d.organs import tracking as TR


def run(xyz: np.ndarray, normal: np.ndarray, is_axis: np.ndarray, root, is_blade: np.ndarray | None = None) -> dict:
    xyz = np.asarray(xyz, np.float64); is_axis = np.asarray(is_axis, bool)
    is_blade = ~is_axis if is_blade is None else np.asarray(is_blade, bool)
    axis_idx = np.nonzero(is_axis)[0]
    # 2026-10-24 (CONTRACT_ORGAN_TOPOLOGY 4.3): the main stem is followed as a tube on the axis points; the skeleton is built with
    # that tube as its trunk (branch vertices on the off-tube points only; branch bases split per child)
    sk, split = TR.build_tracked(xyz[axis_idx], root); SK.nodes(sk)
    cl = LV.blade_clusters(xyz, normal, is_blade)
    # cluster -> nearest skeleton vertex (through the nearest axis point)
    cl_vertex = {}
    if len(axis_idx):
        at = cKDTree(xyz[axis_idx])
        for c in range(cl.max() + 1):
            cen = xyz[cl == c].mean(0); _, j = at.query(cen); v = int(sk.vert_of[j])
            cl_vertex[c] = v if v >= 0 else 0
    leaves, side_axes, trusses, stubs = LV.structure(sk, cl_vertex)
    # revision 6: the stem never ends in a leaf. If the segment beyond the last leaf-carrying node carries blade clusters,
    # cut the main path at that node and classify the segment as a branch there (apex_cut). At most 3 rounds.
    apex_cuts = 0
    # 2026-10-24: the tube follower ends the stem by ring evidence (contract 4.3); revision 6's apex_cut stays only for a path without it
    for _ in range(0 if getattr(sk, 'tube', None) is not None else 3):
        carrying = [nd for nd in sk.nodes if any(lf["node"] == nd["id"] for lf in leaves) or any(sa["node"] == nd["id"] for sa in side_axes)]
        if not carrying:
            break
        vlast = max(carrying[-1]["vertices"], key=sk.main_path.index); i = sk.main_path.index(vlast); term = sk.main_path[i + 1:]
        if not term:
            break
        inv = {}
        for c, v in cl_vertex.items():
            inv.setdefault(v, []).append(c)
        if not any(v in inv for v in sk.subtree(term[0])):
            break
        sk.main_path = sk.main_path[:i + 1]; SK.nodes(sk); apex_cuts += 1
        leaves, side_axes, trusses, stubs = LV.structure(sk, cl_vertex)
        for lf in leaves:
            if lf["root"] == term[0]:
                lf["flags"].append("apex_cut")
    leaf_of, cflags = LV.assign(xyz, cl, sk, leaves, side_axes, axis_idx)
    labels = LV.leaflet_labels(xyz, cl, sk, leaves, side_axes, leaf_of)
    ncl = cl.max() + 1
    counts = dict(axis_points=int(is_axis.sum()), blade_points=int(is_blade.sum()), skeleton_vertices=int(len(sk.V)),
                  unreached_axis_points=int(len(sk.unreached)), bridges=len(sk.bridges), gap_vertices=int(sk.gap.sum()),
                  nodes=len(sk.nodes), leaf_nodes=len({lf["node"] for lf in leaves} | {sa["node"] for sa in side_axes}),
                  leaves=len(leaves), side_axes=len(side_axes), side_leaves=sum(len(sa["leaves"]) for sa in side_axes),
                  truss_candidates=len(trusses), stubs=len(stubs), blade_clusters=int(ncl),
                  ambiguous=sum("ambiguous" in f for f in cflags.values()), unattached=sum(l == "unattached" for l in leaf_of.values()),
                  euclid_attach=sum("euclid_attach" in f for f in cflags.values()),
                  main_stem_len_rel=float(sk.s_mm[sk.main_path[-1]]) / 1000, root_far=bool(sk.root_far), apex_cuts=apex_cuts,
                  tube=split)
    return dict(sk=sk, cl=cl, cl_vertex=cl_vertex, leaves=leaves, side_axes=side_axes, trusses=trusses, stubs=stubs,
                leaf_of_cluster=leaf_of, cluster_flags=cflags, leaflet_labels=labels, counts=counts, axis_idx=axis_idx, is_axis=is_axis, is_blade=is_blade)
