"""Steps 4-5: branches at nodes -> compound leaves / side axes / truss candidates / stubs; blade clusters; geodesic
assignment of clusters to leaves; P/T/I labels along the rachis."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import contract as C
from tomato3d.organs.skeleton import Skeleton
from tomato3d.organs import tracking as TR

S, L = C.SKEL, C.LEAF


# ---------------------------------------------------------------- blade clusters
def blade_clusters(xyz: np.ndarray, normal: np.ndarray, is_blade: np.ndarray) -> np.ndarray:
    """Cluster id per point (-1 = none): radius graph on blade points, edges cut over the normal-angle limit, small parts dropped."""
    cl = np.full(len(xyz), -1, int); idx = np.nonzero(is_blade)[0]
    if len(idx) < L["min_cluster_pts"]:
        return cl
    P = xyz[idx]; N = normal[idx]; N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    pairs = cKDTree(P).query_pairs(L["cluster_radius_mm"] / 1000, output_type="ndarray")
    cos = np.abs((N[pairs[:, 0]] * N[pairs[:, 1]]).sum(1)); keep = cos >= np.cos(np.radians(L["normal_cut_deg"]))
    pairs = pairs[keep]
    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    nc, lab = connected_components(G, directed=False)
    vals, cnt = np.unique(lab, return_counts=True); big = {v: k for k, v in enumerate(vals[cnt >= L["min_cluster_pts"]])}
    cl[idx] = [big.get(l, -1) for l in lab]
    return cl


# ---------------------------------------------------------------- branches at nodes
def _ancestors(sk: Skeleton, v: int, stop: int) -> list:
    out = []; p = sk.parent[v]
    while p >= 0 and p != stop:
        out.append(int(p)); p = sk.parent[p]
    return out


def _descend(sk: Skeleton, v: int) -> list:
    """Heaviest-subtree descent inside a branch (its 'main path')."""
    path = [v]
    while sk.children.get(v):
        v = max(sk.children[v], key=lambda c: sk.subtree_size[c]); path.append(v)
    return path


def structure(sk: Skeleton, cl_vertex: dict, z_up=np.array([0, 0, 1.0])):
    """cl_vertex: cluster id -> skeleton vertex it hangs from (nearest axis vertex). Returns leaves, side_axes, trusses, stubs."""
    leaves, side_axes, trusses, stubs = [], [], [], []
    s_total = float(sk.s_mm[sk.main_path[-1]]); inv = {}
    for c, v in cl_vertex.items():
        inv.setdefault(v, []).append(c)

    def clusters_in(vs):
        return sorted(c for v in vs for c in inv.get(v, []))

    def sub_branches(path):
        onp = set(path); return [c for v in path for c in sk.children.get(v, []) if c not in onp]

    def leaflike(br):
        p = _descend(sk, br); return sk.length(p) * 1000 >= S["leaflike_mm"] and len(clusters_in(sk.subtree(br))) >= S["leaflike_clusters"]

    # 2026-10-24 CONTRACT_ORGAN_TOPOLOGY 4.4: a branch is a lateral when its own path (continuation descent) carries >= 2 organs of
    # the kind a stem carries - compound leaves (>= 2 clusters and >= leaflike_mm, or >= 3 clusters) or trusses (>= truss_min_mm of
    # axis without a blade cluster) - at least one of them a compound leaf. A rachis carries leaflets (single clusters on short
    # petiolules) and no compound leaf. A branch with no cluster: truss candidate when >= truss_min_mm (flag off_apex below the top
    # truss_apex_share of the stem), else stub.
    T = C.TOPO

    def compound_leaflike(b):
        n = len(clusters_in(sk.subtree(b))); return (n >= 2 and sk.length(_descend(sk, b)) * 1000 >= S["leaflike_mm"]) or n >= 3

    def trusslike(b):
        return not clusters_in(sk.subtree(b)) and sk.length(_descend(sk, b)) * 1000 >= T["truss_min_mm"]

    for ni, node in enumerate(sk.nodes):
        for br in node["branches"]:
            path = _descend(sk, br); sub = sk.subtree(br); cls = clusters_in(sub)
            if not cls:
                if sk.length(path) * 1000 >= T["truss_min_mm"]:
                    fl = ["truss_candidate"] + ([] if node["s_mm"] >= (1 - L["truss_apex_share"]) * s_total else ["off_apex"])
                    trusses.append(dict(root=br, node=ni, path=path, vertices=sub, flags=fl)); continue
                stubs.append(dict(root=br, node=ni, path=path, vertices=sub, clusters=cls, flags=["stub"])); continue
            tpath_full = TR.descend_tracked(sk, br); onp_full = set(tpath_full)
            # terminal segment of the branch's own path: after the last path vertex carrying a cluster (directly or in a sub-branch).
            # Without clusters and >= truss_min_mm it is a truss ending this axis (4.4); the axis path then stops where the truss begins.
            has_org = [i for i, v in enumerate(tpath_full) if inv.get(v) or any(clusters_in(sk.subtree(c)) for c in sk.children.get(v, []) if c not in onp_full)]
            i_last = max(has_org) if has_org else -1
            term = tpath_full[i_last + 1:]
            term_truss = bool(term) and sk.length(tpath_full[max(i_last, 0):]) * 1000 >= T["truss_min_mm"] and i_last >= 0
            tpath = tpath_full[:i_last + 1] if term_truss else tpath_full
            subs_t = sub_branches(tpath)
            leafy = [b for b in subs_t if compound_leaflike(b)]
            # trusses anywhere in the branch's subtree (nested under a glued leaf as well): a compound leaf never carries a truss
            onp = set(tpath); trussy = []
            for v in sub:
                if v in onp or v in (term[0] if term_truss and term else -1,):
                    continue
                if trusslike(v) and not any(a in set(trussy) for a in _ancestors(sk, v, br)):
                    trussy.append(v)
            # leaflet attachments directly on the branch's own path: >= 2 of them mean the path itself runs along a rachis
            leaflet_subs = [b for b in subs_t if b not in leafy and b not in trussy and clusters_in(sk.subtree(b))]
            direct = [c for v in tpath for c in inv.get(v, [])]
            path_leaf_root = None
            if len(leaflet_subs) + len(direct) >= 2:
                path_leaf_root = min((sk.parent[b] for b in leaflet_subs), key=lambda v: sk.s_mm[v]) if leaflet_subs else tpath[0]
            n_leaf = len(leafy) + (1 if path_leaf_root is not None else 0)
            if n_leaf >= 1 and n_leaf + len(trussy) + int(term_truss) >= T["lateral_min_organs"]:
                sa = dict(root=br, node=ni, path=tpath, vertices=sub, flags=[], leaves=[], trusses=[])
                for b in sorted(leafy, key=lambda b: sk.s_mm[b]):
                    p2 = _descend(sk, b); sa["leaves"].append(_leaf(sk, b, ni, p2, sub_branches(p2), clusters_in(sk.subtree(b)), z_up, side=True))
                if path_leaf_root is not None:
                    p2 = _descend(sk, path_leaf_root)
                    sa["leaves"].append(_leaf(sk, path_leaf_root, ni, p2, sub_branches(p2), clusters_in(sk.subtree(path_leaf_root)), z_up, side=True))
                    sa["leaves"][-1]["flags"].append("path_leaf")
                sa["leaves"].sort(key=lambda lf: lf["s_mm"])
                for b in trussy:
                    sa["trusses"].append(dict(root=b, node=ni, path=_descend(sk, b), vertices=sk.subtree(b), flags=["truss_candidate", "on_lateral"]))
                if term_truss:
                    sa["trusses"].append(dict(root=term[0], node=ni, path=term, vertices=sk.subtree(term[0]), flags=["truss_candidate", "on_lateral", "terminal"]))
                sa["vertices"] = [v for v in sub if v not in set(sa["trusses"][-1]["vertices"])] if term_truss else sub
                side_axes.append(sa); continue
            if len(cls) < 2 and sk.length(path) * 1000 < S["leaflike_mm"]:       # rev 5/6b: a leaf needs >= 2 clusters or a rachis >= 10 mm
                stubs.append(dict(root=br, node=ni, path=path, vertices=sub, clusters=cls, flags=["stub"])); continue
            leaves.append(_leaf(sk, br, ni, path, sub_branches(path), cls, z_up))
    # ids in node order, lower insertion first within a node
    leaves.sort(key=lambda lf: (lf["node"], lf["s_mm"]))
    for k, lf in enumerate(leaves):
        lf["id"] = f"L{k + 1:02d}"
    for j, sa in enumerate(side_axes):
        sa["id"] = f"S{j + 1}"
        for k, lf in enumerate(sa["leaves"]):
            lf["id"] = f"S{j + 1}_L{k + 1:02d}"
    for sa in side_axes:                                   # trusses that terminate a lateral are trusses of that axis
        for t in sa["trusses"]:
            t["axis"] = sa["id"]; trusses.append(t)
    for j, t in enumerate(trusses):
        t["id"] = f"T{j + 1}"
    return leaves, side_axes, trusses, stubs


def _leaf(sk, br, ni, path, subs, cls, z_up, side=False):
    P = sk.V[path]; d0 = P[min(3, len(P) - 1)] - P[0] if len(P) > 1 else np.array([1e-9, 0, 0])
    az = float(np.degrees(np.arctan2(d0[1], d0[0])) % 360); el = float(np.degrees(np.arcsin(np.clip(d0[2] / max(np.linalg.norm(d0), 1e-12), -1, 1))))
    turn = 0.0
    for a, b, c in zip(P, P[1:], P[2:]):
        u, w = b - a, c - b; turn += float(np.arccos(np.clip(np.dot(u, w) / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-12), -1, 1)))
    length = sk.length(path)
    return dict(root=br, node=ni, path=path, vertices=sk.subtree(br), clusters=cls, s_mm=float(sk.s_mm[br]), side=side,
                azimuth_deg=az, elevation_deg=el, curvature_rel=float(turn / max(length, 1e-9) / 1000), rachis_len_rel=length,
                attachments=[int(b) for b in subs], n_attach=len(subs), flags=(["gap"] if sk.gap[path].any() else []))


# ---------------------------------------------------------------- geodesic assignment
def combined_graph(xyz: np.ndarray):
    n = len(xyz); k = min(L["graph_knn"] + 1, n); d, idx = cKDTree(xyz).query(xyz, k=k)
    rows, cols, w = [], [], []
    for j in range(1, k):
        m = d[:, j] <= L["graph_cap_mm"] / 1000; rows.append(np.nonzero(m)[0]); cols.append(idx[m, j]); w.append(d[m, j])
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    return coo_matrix((np.r_[w, w], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()


def assign(xyz, cl, sk: Skeleton, leaves, side_axes, axis_idx):
    """Cluster -> leaf id by geodesic distance from the leaf's own axis points; `ambiguous` / `unattached` / euclid fallback."""
    G = combined_graph(xyz); all_leaves = leaves + [lf for sa in side_axes for lf in sa["leaves"]]
    ncl = cl.max() + 1
    if not all_leaves:                                      # no compound leaf found: every cluster stays unattached
        return {c: "unattached" for c in range(ncl)}, {c: [] for c in range(ncl)}
    D = np.full((len(all_leaves), ncl), np.inf)
    cl_pts = [np.nonzero(cl == c)[0] for c in range(ncl)]
    for li, lf in enumerate(all_leaves):
        src = np.concatenate([axis_idx[sk.members[v]] for v in lf["vertices"] if len(sk.members[v])]) if lf["vertices"] else np.array([], int)
        if not len(src):
            continue
        dist = dijkstra(G, directed=False, indices=src, min_only=True)
        for c in range(ncl):
            D[li, c] = np.median(dist[cl_pts[c]]) if len(cl_pts[c]) else np.inf
    leaf_of, flags = {}, {}
    vertex_leaf = {v: lf["id"] for lf in all_leaves for v in lf["vertices"]}
    # revision 5 (precise form): the fallback measures from the cluster's NEAREST POINT to the nearest axis point of any leaf
    la = [(axis_idx[p], vertex_leaf[v]) for v in sorted(vertex_leaf) for p in sk.members[v]]
    la_idx = np.array([p for p, _ in la], int); la_leaf = np.array([l for _, l in la]); vt = cKDTree(xyz[la_idx]) if len(la_idx) else None
    for c in range(ncl):
        col = D[:, c]; order = np.argsort(col); f = []
        if np.isfinite(col[order[0]]):
            best = all_leaves[order[0]]["id"]
            if len(order) > 1 and np.isfinite(col[order[1]]) and col[order[1]] <= (1 + L["ambiguous_margin"]) * col[order[0]]:
                f.append("ambiguous")
        else:
            best = "unattached"
            if vt is not None and len(cl_pts[c]):
                dd, jj = vt.query(xyz[cl_pts[c]]); k = int(np.argmin(dd))
                if dd[k] <= L["euclid_fallback_mm"] / 1000:
                    best = str(la_leaf[jj[k]]); f.append("euclid_attach")
        leaf_of[c] = best; flags[c] = f
    return leaf_of, flags


def leaflet_labels(xyz, cl, sk: Skeleton, leaves, side_axes, leaf_of):
    """Within each leaf: clusters ordered by arc position along the rachis; pairs P1L/P1R..., T terminal, I intercalary."""
    out = {}
    for lf in leaves + [x for sa in side_axes for x in sa["leaves"]]:
        cs = [c for c, l in leaf_of.items() if l == lf["id"]]
        if not cs:
            continue
        P = sk.V[lf["path"]]; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] if len(P) > 1 else np.zeros(1)
        rows = []
        for c in cs:
            cen = xyz[cl == c].mean(0); j = int(np.argmin(np.linalg.norm(P - cen, axis=1)))
            d = P[min(j + 1, len(P) - 1)] - P[max(j - 1, 0)]; side = np.cross(d, cen - P[j])[2]
            rows.append((c, float(s[j]), "L" if side >= 0 else "R", int((cl == c).sum())))
        rows.sort(key=lambda r: r[1]); sizes = np.array([r[3] for r in rows]); med = np.median(sizes)
        term = max(rows, key=lambda r: r[1]); out[term[0]] = f"{lf['id']}__T"
        rest = [r for r in rows if r[0] != term[0]]; pair = 0; last_s = -1e9
        for r in rest:
            if r[3] < L["intercalary_ratio"] * med:
                out[r[0]] = f"{lf['id']}__I"; continue
            if r[1] - last_s > L["pair_window_mm"] / 1000:
                pair += 1; last_s = r[1]
            out[r[0]] = f"{lf['id']}__P{pair}{r[2]}"
    return out
