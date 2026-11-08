"""Step 3: geodesic level-set skeleton on the axis points, main stem by heaviest-subtree descent, nodes.

Gaps stay gaps: an axis component the 3 mm graph cannot reach is linked by ONE bridging edge (<= bridge_mm) to the nearest
reached point and flagged `gap`; farther components stay unreachable and are reported. No points are invented.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import contract as C


class Skeleton:
    """V: vertex positions (0 = root); parent[v]; children; members[v] = axis point indices; bin[v]; gap[v]; s_mm[v] = arc length."""

    def __init__(self, V, parent, members, bins, gap, unreached, root_far):
        self.V, self.parent, self.members, self.bin, self.gap = V, parent, members, bins, gap
        self.unreached, self.root_far = unreached, root_far
        self.children = defaultdict(list)
        for v, p in enumerate(parent):
            if p >= 0:
                self.children[p].append(v)
        self.s_mm = np.zeros(len(V))
        for v in sorted(range(1, len(V)), key=lambda v: bins[v]):
            self.s_mm[v] = self.s_mm[parent[v]] + np.linalg.norm(V[v] - V[parent[v]]) * 1000
        self.main_path = None

    def subtree(self, v):
        out, stack = [], [v]
        while stack:
            u = stack.pop(); out.append(u); stack.extend(self.children.get(u, []))
        return out

    def path_to_root(self, v):
        p = [v]
        while self.parent[p[-1]] >= 0:
            p.append(self.parent[p[-1]])
        return p[::-1]

    def length(self, path):
        return float(sum(np.linalg.norm(self.V[b] - self.V[a]) for a, b in zip(path, path[1:])))

    def longest_path_from(self, v):
        """Root-to-tip: from v, the tip with the largest arc length; returns the vertex list v..tip."""
        tips = [u for u in self.subtree(v) if not self.children.get(u)]
        tip = max(tips, key=lambda u: self.s_mm[u]) if tips else v
        p = self.path_to_root(tip); return p[p.index(v):]


def axis_graph(A: np.ndarray, root: np.ndarray, S=C.SKEL):
    """kNN graph (<= max_edge) + root node (index n) + bridging edges for unreachable components."""
    n = len(A); tree = cKDTree(A); k = min(S["knn"] + 1, n); d, idx = tree.query(A, k=k)
    rows, cols, w = [], [], []
    for j in range(1, k):
        m = d[:, j] <= S["max_edge_mm"] / 1000
        rows.append(np.nonzero(m)[0]); cols.append(idx[m, j]); w.append(d[m, j])
    rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    dr = np.linalg.norm(A - root, axis=1); near = np.nonzero(dr <= S["root_link_mm"] / 1000)[0]; root_far = False
    if not len(near):
        near = np.array([int(np.argmin(dr))]); root_far = True
    rows = np.r_[rows, np.full(len(near), n)]; cols = np.r_[cols, near]; w = np.r_[w, dr[near]]
    G = coo_matrix((np.r_[w, w], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n + 1, n + 1)).tocsr()
    gap = np.zeros(n, bool); bridges = []
    for _ in range(50):
        dist = dijkstra(G, directed=False, indices=n)
        reached = np.isfinite(dist[:n])
        if reached.all():
            break
        ncomp, lab = connected_components(G[:n, :n], directed=False)
        rt = cKDTree(A[reached]); ridx = np.nonzero(reached)[0]; added = False
        for c in np.unique(lab[~reached]):
            mem = np.nonzero((lab == c) & ~reached)[0]
            pick = bridge_pick(A, mem, rt, ridx, S)
            if pick is not None:
                a, bidx, d_ab = pick
                G = G + coo_matrix(([d_ab, d_ab], ([a, bidx], [bidx, a])), shape=G.shape).tocsr()
                gap[a] = gap[bidx] = True; bridges.append((int(a), int(bidx), float(d_ab))); added = True
        if not added:
            break
    dist = dijkstra(G, directed=False, indices=n)
    unreached = np.nonzero(~np.isfinite(dist[:n]))[0]
    return G, dist, gap, bridges, unreached, root_far


def bridge_pick(A, mem, rt, ridx, S):
    """Revision 5: the bridge goes to the reached point that CONTINUES the component's axis, not simply the nearest one.
    score = distance + bridge_turn_mm * (1 - |cos(angle between the bridge and the component's principal direction)|);
    a component with < 5 points has no direction and takes the nearest point. Returns (a, b, distance) or None."""
    lim = S["bridge_mm"] / 1000
    nb = rt.query_ball_point(A[mem], lim)
    cand = [(i, j) for i, js in zip(mem, nb) for j in js]
    if not cand:
        return None
    ai = np.array([c[0] for c in cand]); bj = ridx[np.array([c[1] for c in cand])]
    vec = A[bj] - A[ai]; dist = np.linalg.norm(vec, axis=1)
    if len(mem) >= 5:
        P = A[mem] - A[mem].mean(0); axis = np.linalg.svd(P, full_matrices=False)[2][0]
        cosang = np.abs(vec @ axis) / np.maximum(dist, 1e-12)
        score = dist + S["bridge_turn_mm"] / 1000 * (1 - cosang)
    else:
        score = dist
    k = int(np.argmin(score))
    return int(ai[k]), int(bj[k]), float(dist[k])


def build(A: np.ndarray, root, S=C.SKEL) -> Skeleton:
    root = np.asarray(root, float); n = len(A)
    G, dist, gap_pt, bridges, unreached, root_far = axis_graph(A, root, S)
    ok = np.isfinite(dist[:n]); bins = np.full(n, -1, int); bins[ok] = np.floor(dist[:n][ok] * 1000 / S["bin_mm"]).astype(int)
    # vertices = connected components per bin
    vert_of = np.full(n, -1, int); V = [root]; members = [np.array([], int)]; vbin = [-1]; vgap = [False]
    Gn = G[:n, :n]
    for b in range(0, bins[ok].max() + 1):
        pts = np.nonzero(bins == b)[0]
        if not len(pts):
            continue
        sub = Gn[pts][:, pts]; nc, lab = connected_components(sub, directed=False)
        for c in range(nc):
            mem = pts[lab == c]; vert_of[mem] = len(V); V.append(A[mem].mean(0)); members.append(mem); vbin.append(b); vgap.append(bool(gap_pt[mem].any()))
    V = np.array(V); parent = np.full(len(V), -1, int)
    # parent = the lower-bin vertex with the most graph edges into this vertex (fallback: nearest lower-bin vertex within reach)
    Gcoo = Gn.tocoo()
    for v in range(1, len(V)):
        mem = members[v]; mset = np.zeros(n, bool); mset[mem] = True
        e = (mset[Gcoo.row]) & (vert_of[Gcoo.col] >= 0) & (bins[Gcoo.col] >= 0) & (bins[Gcoo.col] < vbin[v])
        cand = vert_of[Gcoo.col[e]]
        if len(cand):
            vals, cnt = np.unique(cand, return_counts=True); parent[v] = int(vals[np.argmax(cnt)])
        elif vbin[v] == 0:
            parent[v] = 0
        else:
            lower = [u for u in range(1, len(V)) if vbin[u] < vbin[v]]
            parent[v] = min(lower, key=lambda u: np.linalg.norm(V[u] - V[v])) if lower else 0
    sk = Skeleton(V, parent, members, np.array(vbin), np.array(vgap), unreached, root_far)
    sk.bridges = bridges; sk.vert_of = vert_of
    sk.A = np.asarray(A, np.float64)                      # 2026-10-24: axis point coordinates for the tracked main stem / junction split
    prune(sk, S["prune_spur_mm"])
    return sk


def prune(sk: Skeleton, spur_mm: float):
    """Drop tip vertices whose branch from the nearest branching ancestor is shorter than spur_mm (members go to the parent)."""
    changed = True
    while changed:
        changed = False
        for v in [u for u in range(1, len(sk.V)) if not sk.children.get(u)]:
            p = [v]
            while sk.parent[p[-1]] > 0 and len(sk.children[sk.parent[p[-1]]]) == 1:
                p.append(sk.parent[p[-1]])
            anc = sk.parent[p[-1]]
            if anc >= 0 and len(sk.children.get(anc, [])) > 1 and sk.length([anc] + p[::-1]) * 1000 < spur_mm:
                for u in p:
                    sk.members[anc] = np.r_[sk.members[anc], sk.members[u]]; sk.vert_of[sk.members[u]] = anc
                    sk.members[u] = np.array([], int); sk.parent[u] = -2
                    sk.children[sk.parent[u]] if False else None
                for u in p:
                    for q in list(sk.children):
                        if u in sk.children[q]:
                            sk.children[q].remove(u)
                    sk.children.pop(u, None)
                changed = True
                break


def main_stem(sk: Skeleton, tie=C.SKEL["heaviest_tie"]) -> list:
    """Heaviest-subtree descent from the root; when the two heaviest children are within `tie`, the one that turns less."""
    size = np.array([len(m) for m in sk.members], float); sub = size.copy()
    for v in sorted(range(len(sk.V)), key=lambda v: -sk.bin[v]):
        if sk.parent[v] >= 0:
            sub[sk.parent[v]] += sub[v]
    path, v = [0], 0
    while sk.children.get(v):
        ch = sorted(sk.children[v], key=lambda c: -sub[c]); best = ch[0]
        if len(ch) > 1 and sub[ch[1]] >= (1 - tie) * sub[ch[0]] and len(path) >= 2:
            d0 = sk.V[v] - sk.V[path[-2]]
            best = max(ch[:2], key=lambda c: np.dot(d0, sk.V[c] - sk.V[v]) / max(np.linalg.norm(d0) * np.linalg.norm(sk.V[c] - sk.V[v]), 1e-12))
        path.append(best); v = best
    sk.main_path = path; sk.subtree_size = sub
    return path


def nodes(sk: Skeleton, merge_mm=C.SKEL["node_merge_mm"]) -> list[dict]:
    """Branch points on the main stem, merged within merge_mm of arc length. Each node: s_mm, vertices, branches (off-path children)."""
    onpath = set(sk.main_path); raw = []
    for v in sk.main_path:
        br = [c for c in sk.children.get(v, []) if c not in onpath]
        if br:
            raw.append(dict(s_mm=float(sk.s_mm[v]), vertices=[v], branches=br))
    out = []
    for r in raw:
        if out and r["s_mm"] - out[-1]["s_mm"] <= merge_mm:
            out[-1]["vertices"] += r["vertices"]; out[-1]["branches"] += r["branches"]; out[-1]["s_end"] = r["s_mm"]
        else:
            out.append(dict(r, s_end=r["s_mm"]))
    for i, nd in enumerate(out):
        nd["id"] = i; nd["pos"] = sk.V[nd["vertices"][0]].tolist()
    sk.nodes = out
    return out
