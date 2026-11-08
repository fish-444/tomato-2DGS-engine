"""2026-10-24 topology contract, stage 3 (CONTRACT_ORGAN_TOPOLOGY.md 4.3 + junction split): the main stem tracked as a tube.

- ring(): RANSAC circle fit of a vertex's members in the plane normal to the local tangent -> stem centre and radius
  (DEV diagnostic 2026-10-24: centre error vs the truth stem line p50 0.11 mm against 0.30 mm for the vertex centroid, 0.14 vs
  0.69 mm at junction vertices).
- main_stem_tracked(): continuation by direction over dir_window_mm behind / head_mm ahead, children beyond stem_turn_max_deg are
  branches, ties by radius then verticality; subtree weight is NOT a criterion (sympodial rule).
- split_junctions(): members of main-path vertices off the fitted stem line become branch-base vertices, so a short side stem (or a
  rachis base) that the 2 mm geodesic bins merged into the stem appears as ONE branch at the node.
- descend_tracked(): the same continuation rule for a lateral's own path (vertex centroids; rings are too sparse on thin axes).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from tomato3d.organs import contract as C
from tomato3d.organs.skeleton import Skeleton

T = C.TOPO
UP = np.array([0, 0, 1.0])


def circle_fit(P2: np.ndarray, tol_m: float, r_range_m, rng: np.random.Generator, iters: int = 60):
    """RANSAC circle in 2D (3-point samples, inliers within tol), refined by an algebraic least-squares fit on the inliers.
    Returns (centre, radius, inlier_mask) or None when fewer than 6 points or no circle in the radius range."""
    n = len(P2)
    if n < 6:
        return None
    best = (0, None)
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False); a, b, c = P2[i]
        M = np.array([b - a, c - a]) * 2
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        cen = np.linalg.solve(M, np.array([b @ b - a @ a, c @ c - a @ a])); r = np.linalg.norm(a - cen)
        if not (r_range_m[0] <= r <= r_range_m[1]):
            continue
        inl = np.abs(np.linalg.norm(P2 - cen, axis=1) - r) < tol_m
        if inl.sum() > best[0]:
            best = (int(inl.sum()), (cen, r, inl))
    if best[1] is None:
        return None
    cen, r, inl = best[1]; Q = P2[inl]
    if len(Q) >= 4:
        sol = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
        cen2 = sol[:2]; r2 = np.sqrt(max(sol[2] + cen2 @ cen2, 1e-12))
        if r_range_m[0] <= r2 <= r_range_m[1]:
            cen, r = cen2, r2; inl = np.abs(np.linalg.norm(P2 - cen, axis=1) - r) < tol_m
    return cen, r, inl


def _frame(t: np.ndarray):
    t = t / max(np.linalg.norm(t), 1e-12)
    a = np.cross(t, [1.0, 0, 0]) if abs(t[0]) < 0.9 else np.cross(t, [0, 1.0, 0]); a /= np.linalg.norm(a)
    return t, a, np.cross(t, a)


def ring(sk: Skeleton, v: int, t: np.ndarray, rng, T_=T, r_expect: float | None = None) -> dict:
    """Ring fit of vertex v's members in the plane normal to t -> dict(centre, r (m), inliers, ok). With r_expect (m) the radius is
    searched within x/ ring_r_ratio of it, so a junction vertex fits the stem ring and not a branch's smaller ring."""
    M = sk.A[sk.members[v]]
    if len(M) < 6:
        return dict(centre=sk.V[v].copy(), r=np.nan, inliers=np.ones(len(M), bool), ok=False)
    t, a, b = _frame(t); rel = M - sk.V[v]; P2 = np.c_[rel @ a, rel @ b]
    rr = (T_["circle_r_mm"][0] / 1000, T_["circle_r_mm"][1] / 1000)
    if r_expect is not None and np.isfinite(r_expect):
        rr = (max(rr[0], r_expect / T_["ring_r_ratio"]), min(rr[1], r_expect * T_["ring_r_ratio"]))
    res = circle_fit(P2, T_["circle_tol_mm"] / 1000, rr, rng)
    if res is None or res[2].sum() < T_["circle_min_inliers"]:
        return dict(centre=sk.V[v].copy(), r=np.nan, inliers=np.ones(len(M), bool), ok=False)
    cen, r, inl = res
    return dict(centre=sk.V[v] + cen[0] * a + cen[1] * b, r=float(r), inliers=inl, ok=True)


def line_dir(P: list, fallback: np.ndarray) -> np.ndarray:
    """Unit direction of the least-squares line through the points, oriented from the first to the last point."""
    P = np.asarray(P, float)
    if len(P) < 2:
        d = np.asarray(fallback, float)
    elif len(P) == 2:
        d = P[1] - P[0]
    else:
        d = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
        if d @ (P[-1] - P[0]) < 0:
            d = -d
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else UP.copy()


def head_chain(sk: Skeleton, c: int, length_mm: float, min_pts: int = 0, max_mm: float | None = None) -> list:
    """c and its descendants along the chain of largest-member children while within length_mm of arc (the child's 'head');
    when the head holds fewer than min_pts points (sparse base, thinned axes) the chain extends up to max_mm."""
    out, v = [c], c; n = len(sk.members[c])
    while sk.children.get(v):
        v = max(sk.children[v], key=lambda u: len(sk.members[u])); d = sk.s_mm[v] - sk.s_mm[c]
        if d > length_mm and not (n < min_pts and max_mm is not None and d <= max_mm):
            break
        out.append(v); n += len(sk.members[v])
    return out


def points_dir(sk: Skeleton, vs: list, origin: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """PCA direction of the member points of vs, oriented from origin towards their centroid (sparse fallback for rings)."""
    P = np.concatenate([sk.A[sk.members[u]] for u in vs if len(sk.members[u])]) if vs else np.zeros((0, 3))
    if len(P) < 6:
        return line_dir([], fallback)
    d = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
    if d @ (P.mean(0) - origin) < 0:
        d = -d
    return d


def angle_deg(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1, 1))))


def compute_subtree_size(sk: Skeleton):
    size = np.array([len(m) for m in sk.members], float); sub = size.copy()
    for v in sorted(range(len(sk.V)), key=lambda v: -sk.s_mm[v]):
        if sk.parent[v] >= 0:
            sub[sk.parent[v]] += sub[v]
    sk.subtree_size = sub
    return sub


def circle_fit_constrained(P2: np.ndarray, tol_m: float, r_range_m, offset_max_m: float, rng: np.random.Generator, iters: int = 80):
    """RANSAC circle whose centre lies within offset_max of the origin (= the extrapolated stem axis): direction continuity is part of
    the fit, so a branch's ring (off-axis) never wins. Returns (centre, radius, inliers) or None."""
    n = len(P2)
    if n < 6:
        return None
    best = (0, None)
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False); a, b, c = P2[i]
        M = np.array([b - a, c - a]) * 2
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        cen = np.linalg.solve(M, np.array([b @ b - a @ a, c @ c - a @ a])); r = np.linalg.norm(a - cen)
        if not (r_range_m[0] <= r <= r_range_m[1]) or np.linalg.norm(cen) > offset_max_m:
            continue
        inl = np.abs(np.linalg.norm(P2 - cen, axis=1) - r) < tol_m
        if inl.sum() > best[0]:
            best = (int(inl.sum()), (cen, r, inl))
    if best[1] is None:
        return None
    cen, r, inl = best[1]; Q = P2[inl]
    if len(Q) >= 4:
        sol = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
        cen2 = sol[:2]; r2 = np.sqrt(max(sol[2] + cen2 @ cen2, 1e-12))
        if r_range_m[0] <= r2 <= r_range_m[1] and np.linalg.norm(cen2) <= offset_max_m:
            cen, r = cen2, r2; inl = np.abs(np.linalg.norm(P2 - cen, axis=1) - r) < tol_m
    return cen, r, inl


def coverage(P2: np.ndarray, bins: int = 12) -> float:
    """Share of angular bins around the origin occupied by the points (a full ring -> 1.0; a branch crossing -> a short arc)."""
    if len(P2) == 0:
        return 0.0
    ang = np.arctan2(P2[:, 1], P2[:, 0]); k = np.floor((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    return float(len(np.unique(k)) / bins)


def follow_tube(A: np.ndarray, root: np.ndarray, T_=T, trace: list | None = None) -> dict:
    """Contract 4.3 on points: from the root the stem is followed as a tube. Each step takes the axis points within tube_step_mm
    ahead along the current direction and within the search radius and fits the ring (RANSAC; radius within ring_r_ratio of the
    arriving radius; centre within the stem_turn_max_deg cone of the arriving direction - direction continuity is part of the fit, so
    a branch's ring cannot be chosen). The centre moves to the ring centre; the direction is the line through the recent ring centres
    (PCA of the on-tube points when fewer than 3 rings; UP until the path spans dir_min_span_mm). When no acceptable ring exists at
    this reach (too few points, or only branch points ahead) the reach grows up to tube_gap_max_mm along the same direction - gaps
    are bridged by continuation and flagged gap. The stem ENDS when nothing within the cone continues it within tube_gap_max_mm.
    Subtree weight is never consulted. Returns centres, radii, on-tube mask, step index per on-tube point, gap flags, stop reason."""
    rng = np.random.default_rng(20261024); tree = cKDTree(A); n = len(A)
    step = T_["tube_step_mm"] / 1000; r_search0 = T_["tube_search_mm"] / 1000; tol_m = T_["tube_member_tol_mm"] / 1000
    c = np.asarray(root, float); d = UP.copy(); centres = [c.copy()]; dirs = [d.copy()]; radii = [np.nan]; gaps = [False]; ok_ring = [False]
    on = np.zeros(n, bool); step_of = np.full(n, -1, int); stop = "no continuation within the cone and the gap reach"; cov_hist = [0.0]
    tan_lim = np.tan(np.radians(T_["stem_turn_max_deg"])); tan_sparse = np.tan(np.radians(T_["stem_turn_sparse_deg"]))
    for k in range(1, 10000):
        # arriving radius: the last clean rings only (full coverage); junction slabs give inflated circles that must not widen the window
        clean = [radii[i] for i in range(len(radii)) if np.isfinite(radii[i]) and cov_hist[i] >= T_["clean_ring_cov"]][-4:]
        r_in = float(np.median(clean)) if clean else np.nan
        # anchor for the turn test: the mean of the ring centres behind (within dir_window_mm), not the last centre alone - a single
        # ring centre can be off by ~1 mm at a junction, which would read as a 20 deg turn over one 3 mm step
        behind = [i for i in range(1, len(centres)) if np.linalg.norm(centres[-1] - centres[i]) <= T_["dir_window_mm"] / 1000 and ok_ring[i]]
        anchor = np.mean([centres[i] for i in behind], axis=0) if len(behind) >= 2 else c
        ringed_all = [centres[i] for i in range(1, len(centres)) if ok_ring[i]]
        g_axis = line_dir(ringed_all, UP) if len(ringed_all) >= 4 and np.linalg.norm(ringed_all[-1] - ringed_all[0]) >= T_["global_span_mm"] / 1000 else None
        c_g0 = np.mean(ringed_all[-8:], axis=0) if g_axis is not None else c      # the global-tilt test runs from the recent part of the stem
        r_search = (r_in * T_["ring_r_ratio"] + T_["off_axis_margin_mm"] / 1000) if np.isfinite(r_in) else r_search0
        rr_accept = (T_["circle_r_mm"][0] / 1000, T_["circle_r_mm"][1] / 1000)
        if np.isfinite(r_in):                                   # radius continuity: a peduncle (0.7) or rachis (0.5) does not continue a 1.0+ stem
            rr_accept = (max(rr_accept[0], r_in / T_["ring_accept_ratio"]), min(rr_accept[1], r_in * T_["ring_grow_ratio"]))
        accepted = None; reach = step
        idx_all = np.array(tree.query_ball_point(c + d * (T_["tube_gap_max_mm"] / 2000), T_["tube_gap_max_mm"] / 2000 + r_search), int)
        if len(idx_all):
            rel = A[idx_all] - c; along_all = rel @ d; perp_all = np.linalg.norm(rel - np.outer(along_all, d), axis=1)
            base_ok = (along_all > 0.25 * step) & (perp_all <= r_search) & ~on[idx_all]
            # scan successive slabs of one step along the axis (half-step stride) up to the gap reach: branch bases just ahead of a
            # stem gap fill the first slab, the stem resumes in a later one
            for slab0 in np.arange(0.25 * step, T_["tube_gap_max_mm"] / 1000 - step + 1e-9, step / 2):
                m = base_ok & (along_all >= slab0) & (along_all < slab0 + step)
                if m.sum() < T_["tube_min_pts"]:
                    continue
                reach = slab0 + step
                sel = idx_all[m]; al = along_all[m]; s_mid = float(np.median(al)); c_mid = c + d * s_mid
                t, a, b = _frame(d); relm = A[sel] - c_mid; P2 = np.c_[relm @ a, relm @ b]
                res = circle_fit_constrained(P2, T_["circle_tol_mm"] / 1000, rr_accept, tan_lim * s_mid + T_["circle_tol_mm"] / 1000, rng)
                if trace is not None:
                    trace.append(dict(step=k, reach_mm=reach * 1000, n=int(len(sel)), s_mid_mm=s_mid * 1000, r_in_mm=r_in * 1000 if np.isfinite(r_in) else None,
                                      ring=None if res is None else dict(r_mm=res[1] * 1000, inl=int(res[2].sum()), off_mm=float(np.linalg.norm(res[0])) * 1000, cov=coverage(P2[res[2]] - res[0])),
                                      cov_all=coverage(P2 - P2.mean(0)), off_all_mm=float(np.linalg.norm(P2.mean(0))) * 1000, perp_med_mm=float(np.median(np.linalg.norm(P2 - P2.mean(0), axis=1))) * 1000))
                def note(why):
                    if trace is not None:
                        trace[-1]["reject"] = why
                if res is None or res[2].sum() < T_["circle_min_inliers"]:
                    note("no ring in radius window / cone")
                elif coverage(P2[res[2]] - res[0]) < T_["ring_coverage_min"]:
                    note("ring coverage")
                if res is not None and res[2].sum() >= T_["circle_min_inliers"] and coverage(P2[res[2]] - res[0]) >= T_["ring_coverage_min"]:
                    cen, r, inl = res; dist = np.abs(np.linalg.norm(P2 - cen, axis=1) - r); c_try = c_mid + cen[0] * a + cen[1] * b
                    # the step itself must stay within the cone measured from the last centre (a peduncle or lateral slab is skipped, not followed)
                    # across a gap the extrapolation is long, so the cone is wide; the ring that continues the stem lies within
                    # gap_angle_max_deg of the axis, a lateral or peduncle base (2-3 mm off-axis) does not
                    lim = T_["stem_turn_max_deg"] if slab0 <= step else T_["gap_angle_max_deg"]
                    # turn = lateral offset of the candidate centre from the fitted stem line (through the anchor along d), over the
                    # distance from the last centre: a lateral's first slab (33 deg) offsets ~2 mm per 3 mm, the stem < 1 mm
                    v_try = c_try - anchor; off_line = np.linalg.norm(v_try - (v_try @ d) * d); turn = np.degrees(np.arctan2(off_line, max(s_mid, 1e-6)))
                    if k > 1 and turn > lim:
                        # a ring centre pulled ~1 mm towards a branch base reads as a 20 deg turn over one step, a lateral at 33 deg
                        # reads the same; they differ in the NEXT slab: a shifted stem keeps its offset, a lateral's offset grows
                        if turn > T_["stem_turn_hard_deg"] or slab0 > step:
                            note(f"turn {turn:.0f} > {lim}"); continue
                        nxt = base_ok & (along_all >= slab0 + step) & (along_all < slab0 + 2 * step)
                        if nxt.sum() < T_["tube_min_pts"]:
                            note(f"turn {turn:.0f} > {lim}, no next slab"); continue
                        rel_n = A[idx_all[nxt]] - anchor; off_n = np.linalg.norm(rel_n - np.outer(rel_n @ d, d), axis=1)
                        if np.median(off_n) > off_line + T_["offset_growth_mm"] / 1000:
                            note(f"turn {turn:.0f} > {lim}, offset grows {np.median(off_n) * 1000:.1f} > {off_line * 1000:.1f}"); continue
                    # continuity with the WHOLE stem so far: once the path spans global_span_mm, a slab whose direction from the
                    # anchor departs more than stem_global_tilt_max_deg from the stem's global axis is a lateral or peduncle, not the stem
                    if g_axis is not None and angle_deg(c_try - c_g0, g_axis) > T_["stem_global_tilt_max_deg"]:
                        note(f"global tilt {angle_deg(c_try - c_g0, g_axis):.0f}"); continue
                    if slab0 > step:                       # across a gap: one slab of continuation evidence beyond the candidate
                        la = base_ok & (along_all >= slab0 + step) & (along_all < slab0 + 2 * step) & (np.linalg.norm(A[idx_all] - c_try - np.outer((A[idx_all] - c_try) @ d, d), axis=1) <= r_search)
                        if la.sum() < T_["tube_min_pts"]:
                            note("no continuation evidence"); continue
                    accepted = dict(c_new=c_try, r=float(r), members=sel[dist <= tol_m], ring=True, cov=coverage(P2[inl] - cen)); break
                if len(sel) < T_["circle_min_inliers"] + 2:              # sparse: a thin ring of points around the extrapolated axis
                    off = P2.mean(0); perp_med = float(np.median(np.linalg.norm(P2 - off, axis=1)))
                    r_ok = (not np.isfinite(r_in)) or (r_in / T_["ring_accept_ratio"] <= perp_med <= r_in * T_["ring_accept_ratio"])
                    if np.linalg.norm(off) <= tan_sparse * s_mid + tol_m and r_ok and coverage(P2 - off) >= T_["sparse_coverage_min"]:
                        accepted = dict(c_new=c_mid + off[0] * a + off[1] * b, r=np.nan, members=sel, ring=False, cov=0.0); break
        if accepted is None:
            break
        c_new, members, ring_ok = accepted["c_new"], accepted["members"], accepted["ring"]; gap = reach > 1.5 * step + 1e-9
        cs = centres + [c_new]; oks = ok_ring + [ring_ok]
        recent = [i for i in range(1, len(cs)) if np.linalg.norm(cs[-1] - cs[i]) <= T_["dir_window_mm"] / 1000 and oks[i]]
        span = np.linalg.norm(cs[recent[-1]] - cs[recent[0]]) if len(recent) >= 2 else 0.0
        total_span = np.linalg.norm(c_new - centres[1]) if len(centres) > 1 else 0.0
        if total_span < T_["dir_min_span_mm"] / 1000:
            d_new = UP.copy()
        elif len(recent) >= 3 and span >= T_["dir_min_span_mm"] / 1000:
            d_new = line_dir([cs[i] for i in recent], d)
            # drift guard: ring centres at junctions are pulled towards branch bases; when the local line departs from the stem's
            # global axis by more than dir_drift_max_deg the global axis is used (the stem is straight at the scale of 15 mm)
            if g_axis is not None and angle_deg(d_new, g_axis) > T_["dir_drift_max_deg"]:
                d_new = g_axis.copy()
        elif g_axis is not None:
            d_new = g_axis.copy()                                   # too few rings nearby (after a gap): the stem's global axis
        else:
            win = on.copy(); win[members] = True; pts = A[win]; pts = pts[np.linalg.norm(pts - c_new, axis=1) <= T_["dir_sparse_mm"] / 1000]
            span_pts = float(((pts - c_new) @ d).max() - ((pts - c_new) @ d).min()) if len(pts) else 0.0
            if len(pts) >= T_["dir_min_pts"] and span_pts >= 2 * T_["dir_min_span_mm"] / 1000:     # a single ring has no axis: PCA needs length
                d_new = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][0]
                if d_new @ d < 0:
                    d_new = -d_new
            else:
                d_new = line_dir([c, c_new], d) if np.linalg.norm(c_new - c) >= T_["dir_min_span_mm"] / 1000 else d.copy()
        centres.append(c_new); dirs.append(d_new); radii.append(accepted["r"]); gaps.append(gap); ok_ring.append(ring_ok); cov_hist.append(accepted["cov"])
        on[members] = True; step_of[members] = len(centres) - 1; c, d = c_new, d_new
    return dict(centres=np.array(centres), dirs=np.array(dirs), radii=np.array(radii), gaps=np.array(gaps), ring_ok=np.array(ok_ring), on=on, step_of=step_of, stop=stop, steps=len(centres) - 1)


def build_on_tube(A: np.ndarray, root, tube: dict, S=C.SKEL, T_=T):
    """Geodesic level-set skeleton with the tube as the trunk. The axis graph (kNN <= 3 mm + root link + bridging, skeleton.axis_graph)
    is kept; geodesic distance is measured from the tube by a multi-source Dijkstra (a virtual node linked to every on-tube point with
    weight = that point's arc length along the tube), so a branch's distance = where it leaves the stem + its own length. Vertices:
    one per tube step (members = the step's on-tube points, parent = the previous step) and, for the off-tube points, the connected
    components per 2 mm bin; a branch vertex's parent is the lower-bin vertex it has most edges to, or the tube step its points touch
    most. Spurs < prune_spur_mm are pruned as before. A branch base with several children is split so each child keeps the base
    points nearest to it (a leaf and a lateral from one axil become separate branches)."""
    from tomato3d.organs.skeleton import Skeleton, axis_graph, prune
    A = np.asarray(A, np.float64); n = len(A); root = np.asarray(root, float)
    G, dist0, gap_pt, bridges, unreached, root_far = axis_graph(A, root, S)
    on = tube["on"].copy(); step_of = tube["step_of"].copy(); nsteps = tube["steps"]
    # membership cleanup (declared): an on-tube point whose neighbours within member_clean_mm are mostly off-tube belongs to the
    # branch base that surrounds it (a side stem or rachis base hugging the stem ring), not to the stem; two passes
    if on.any() and (~on).any():
        tree_all = cKDTree(A)
        for _ in range(2):
            idx_on = np.nonzero(on)[0]; nb = tree_all.query_ball_point(A[idx_on], T_["member_clean_mm"] / 1000)
            flip = np.array([(~on[np.array(b)]).sum() > on[np.array(b)].sum() - 1 for b in nb]) if len(idx_on) else np.zeros(0, bool)
            if not flip.any():
                break
            on[idx_on[flip]] = False; step_of[idx_on[flip]] = -1
    tube["on"], tube["step_of"] = on, step_of
    s_step = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(tube["centres"], axis=0), axis=1))]          # arc length per tube step (m)
    src = np.nonzero(on)[0]
    if not len(src):
        sk = Skeleton(np.array([root]), np.array([-1]), [np.array([], int)], np.array([-1]), np.array([False]), np.arange(n), root_far)
        sk.A = A; sk.bridges = bridges; sk.vert_of = np.full(n, -1, int); sk.main_path = [0]; sk.centre = {}; sk.radius = {}; sk.tangent = {}; compute_subtree_size(sk)
        return sk, dict(tube_steps=0, tube_points=0, stop=tube["stop"], gap_steps=0, rings_ok=0, base_splits=0, reparented=0)
    Gs = G[:n, :n].tocoo()                                             # drop the pot-root links; the tube is the trunk now
    # every off-tube point within max_edge_mm of the tube gets an edge to its nearest on-tube point: a branch base's kNN edges all stay
    # inside its own dense points, so without this the base would reach the stem only through a stray bridging edge
    off = np.nonzero(~on)[0]; ont = cKDTree(A[src]); dn, jn = ont.query(A[off]); near = dn <= S["max_edge_mm"] / 1000
    er, ec, ew = off[near], src[jn[near]], dn[near]
    Gs = coo_matrix((np.r_[Gs.data, ew, ew], (np.r_[Gs.row, er, ec], np.r_[Gs.col, ec, er])), shape=(n, n)).tocsr(); Gs.sum_duplicates()
    nearest_step = np.full(n, -1, int); nearest_step[off[near]] = step_of[src[jn[near]]]
    sup = n; w = s_step[step_of[src]] + 1e-6
    Gm = coo_matrix((np.r_[Gs.tocoo().data, w, w], (np.r_[Gs.tocoo().row, np.full(len(src), sup), src], np.r_[Gs.tocoo().col, src, np.full(len(src), sup)])), shape=(n + 1, n + 1)).tocsr()
    dist = dijkstra(Gm, directed=False, indices=sup)[:n]
    ok = np.isfinite(dist) & ~on; unreached = np.nonzero(~np.isfinite(dist))[0]
    # vertices: root (0), tube steps (1..nsteps), then branch vertices
    V = [root] + [tube["centres"][k] for k in range(1, nsteps + 1)]; members = [np.array([], int)] + [np.nonzero(step_of == k)[0] for k in range(1, nsteps + 1)]
    vbin = [-1] + list(range(nsteps)); vgap = [False] + [bool(tube["gaps"][k]) for k in range(1, nsteps + 1)]
    parent = [-1] + [k - 1 for k in range(1, nsteps + 1)]; vert_of = np.full(n, -1, int)
    for k in range(1, nsteps + 1):
        vert_of[members[k]] = k
    bins = np.full(n, -1, int); bins[ok] = np.floor(dist[ok] * 1000 / S["bin_mm"]).astype(int)
    Gcoo = Gs.tocoo(); first_branch = len(V)
    for b in range(0, int(bins[ok].max()) + 1 if ok.any() else 0):
        pts = np.nonzero(bins == b)[0]
        if not len(pts):
            continue
        sub = Gs[pts][:, pts]; nc, lab = connected_components(sub, directed=False)
        for c in range(nc):
            mem = pts[lab == c]; vert_of[mem] = len(V); V.append(A[mem].mean(0)); members.append(mem); vbin.append(b + nsteps); vgap.append(bool(gap_pt[mem].any())); parent.append(-1)
    V = np.array(V); parent = np.array(parent); vbin = np.array(vbin); vgap = np.array(vgap)
    for v in range(first_branch, len(V)):
        mem = members[v]; mset = np.zeros(n, bool); mset[mem] = True
        e = mset[Gcoo.row] & (vert_of[Gcoo.col] >= 0) & (vert_of[Gcoo.col] != v)
        cand = vert_of[Gcoo.col[e]]
        tube_c = cand[(cand <= nsteps) & (cand >= 1)]; lower = cand[(cand > nsteps) & (vbin[cand] < vbin[v])]
        touch = nearest_step[mem]; touch = touch[touch >= 1]
        if len(lower):
            vals, cnt = np.unique(lower, return_counts=True); parent[v] = int(vals[np.argmax(cnt)])
        elif len(touch):                                                       # the base touches the tube: the step its nearest on-tube points belong to
            vals, cnt = np.unique(touch, return_counts=True); parent[v] = int(vals[np.argmax(cnt)])
        elif len(tube_c):
            vals, cnt = np.unique(tube_c, return_counts=True); parent[v] = int(vals[np.argmax(cnt)])
        else:                                                                  # no edge to a lower vertex: nearest tube step by distance
            dd = np.linalg.norm(V[1:nsteps + 1][:, None, :] - A[mem][None], axis=2).min(1) if nsteps else np.array([0.0])
            parent[v] = int(np.argmin(dd)) + 1 if nsteps else 0
    sk = Skeleton(V, parent, members, vbin, vgap, unreached, root_far)
    sk.A = A; sk.bridges = bridges; sk.vert_of = vert_of
    prune(sk, S["prune_spur_mm"])
    sk.main_path = list(range(0, nsteps + 1))
    sk.centre = {k: tube["centres"][k] for k in range(1, nsteps + 1)}; sk.radius = {k: tube["radii"][k] for k in range(1, nsteps + 1) if np.isfinite(tube["radii"][k])}
    sk.tangent = {k: tube["dirs"][k] for k in range(1, nsteps + 1)}
    splits = split_bases(sk, T_)
    compute_subtree_size(sk)
    rec = dict(tube_steps=nsteps, tube_points=int(on.sum()), stop=tube["stop"], gap_steps=int(tube["gaps"].sum()), rings_ok=int(tube["ring_ok"].sum()),
               base_splits=splits, branch_vertices=int(len(V) - first_branch), unreached=int(len(unreached)))
    return sk, rec


def split_bases(sk: Skeleton, T_=T) -> int:
    """A branch-base vertex (child of a tube step) with >= 2 children is split by nearest child; returns the number of new vertices."""
    A = sk.A; split = 0
    for u in list(sk.main_path[1:]):
        for b in list(sk.children.get(u, [])):
            ch = list(sk.children.get(b, []))
            if len(ch) < 2 or len(sk.members[b]) < 2 * T_["split_min_pts"]:
                continue
            trees = [cKDTree(A[sk.members[c]]) if len(sk.members[c]) else None for c in ch]
            D = np.stack([t.query(A[sk.members[b]])[0] if t is not None else np.full(len(sk.members[b]), np.inf) for t in trees], 1); lab = D.argmin(1)
            groups = [sk.members[b][lab == i] for i in range(len(ch))]
            big = [i for i, g in enumerate(groups) if len(g) >= T_["split_min_pts"]]
            if len(big) < 2:
                continue
            first = big[0]; rest = big[1:]
            keep = np.concatenate([groups[i] for i in range(len(ch)) if i == first or i not in big]); sk.members[b] = keep; sk.V[b] = A[keep].mean(0)
            for i in rest:
                c, g = ch[i], groups[i]; w = len(sk.V)
                sk.V = np.vstack([sk.V, A[g].mean(0)]); sk.parent = np.append(sk.parent, u); sk.members.append(g); sk.bin = np.append(sk.bin, sk.bin[b])
                sk.gap = np.append(sk.gap, sk.gap[b]); sk.vert_of[g] = w; sk.children[u].append(w); sk.children[b].remove(c); sk.children[w] = [c]; sk.parent[c] = w
                sk.s_mm = np.append(sk.s_mm, sk.s_mm[u] + np.linalg.norm(sk.V[w] - sk.V[u]) * 1000); split += 1
    return split


def build_tracked(A: np.ndarray, root, S=C.SKEL, T_=T):
    """Stage 3 entry: follow the stem tube on the axis points, then build the skeleton with the tube as its trunk."""
    A = np.asarray(A, np.float64); tube = follow_tube(A, np.asarray(root, float), T_)
    sk, rec = build_on_tube(A, root, tube, S, T_); sk.tube = tube
    return sk, rec


def descend_tracked(sk: Skeleton, v: int, d0: np.ndarray | None = None, T_=T) -> list:
    """Continuation descent inside a branch (a lateral's own axis), vertex centroids, within stem_turn_max_deg; else the path ends."""
    path = [v]
    d_prev = d0 if d0 is not None else (sk.V[v] - sk.V[sk.parent[v]] if sk.parent[v] >= 0 else UP)
    while sk.children.get(v):
        recent = [u for u in path if sk.s_mm[v] - sk.s_mm[u] <= T_["dir_window_mm"]]
        d_in = line_dir([sk.V[u] for u in recent], d_prev); cands = []
        for c in sk.children[v]:
            head = head_chain(sk, c, T_["head_mm"]); d_c = line_dir([sk.V[v]] + [sk.V[u] for u in head], sk.V[c] - sk.V[v])
            cands.append((angle_deg(d_in, d_c), -len(sk.members[c]), int(c), d_c))
        cands.sort(key=lambda x: x[:3]); best = cands[0]
        if best[0] > T_["stem_turn_max_deg"]:
            break
        path.append(best[2]); d_prev = best[3]; v = best[2]
    return path
