"""Base angle θ0 and rachis bend Δθ from surfels near the petiole base.

Rules fixed by build/traits_base_angle_surfel_20261101_01/CONTRACT.md (§1, §4). Units mm, z up.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.traits import recon_traits as rt

LINK_MM = 2.0
THIN_R_MM, THIN_MAX_MM = 3.0, 3.0
REACH_MM, MAX_PATH_MM, MAX_TRY = 5.0, 25.0, 10
LINE_R_MM, BIN_MM = 3.0, 2.0
END_MM = 10.0
STEM_R_MM, SNAP_MAX_MM, SHARE_MM = 6.0, 3.0, 1.0


class StemLine:
    """Fitted stem axis (v4, build/traits_base_angle_v4_20261102_01/CONTRACT.md §0): point p0, unit direction d with d_z > 0."""

    def __init__(self, p0, d):
        self.p0 = np.asarray(p0, float); d = np.asarray(d, float); d = d / np.linalg.norm(d); self.d = d if d[2] > 0 else -d

    def xy_at(self, z):
        t = (np.asarray(z, float) - self.p0[2]) / self.d[2]
        return self.p0[:2] + np.multiply.outer(t, self.d[:2])


def cxy(c, p):
    """Stem xy for point(s) p: the fixed 2-vector c (vertical stand-in) or a StemLine at each point's own height."""
    return c.xy_at(np.asarray(p, float)[..., 2]) if isinstance(c, StemLine) else c


def hdist(p, c):
    p = np.asarray(p, float); q = cxy(c, p); return np.hypot(p[..., 0] - q[..., 0], p[..., 1] - q[..., 1])


def base_first(line, c):
    line = np.asarray(line, float)
    return line[::-1].copy() if hdist(line[-1], c) < hdist(line[0], c) else line.copy()


def width_across(xyz, tree, i, d):
    nb = tree.query_ball_point(xyz[i], THIN_R_MM)
    if len(nb) < 3:
        return 0.0
    q = xyz[nb] - xyz[i]; perp = q - np.outer(q @ d, d)
    return 2.0 * float(np.percentile(np.linalg.norm(perp, axis=1), 90))


def grow_petiole(xyz, tree, start, c, allowed, thin_max=THIN_MAX_MM):
    """Greedy walk on the surfel graph from the line's base-side end toward the stand-in through thin structure."""
    d0, i0 = tree.query(start)
    if d0 > LINK_MM:
        return [], int(hdist(start, c) <= REACH_MM), 'no_surfel_at_start'
    path = [int(i0)]; visited = {int(i0)}; length = 0.0
    while True:
        cur = path[-1]; hc = float(hdist(xyz[cur], c))
        if hc <= REACH_MM:
            return path, 1, 'reached'
        if length >= MAX_PATH_MM:
            return path, 0, 'max_length'
        nb = [j for j in tree.query_ball_point(xyz[cur], LINK_MM) if j not in visited and allowed[j]]
        if not nb:
            return path, 0, 'no_neighbour'
        nb = np.array(nb); prog = hc - hdist(xyz[nb], c); order = np.argsort(-prog)
        step = None
        for j in nb[order][:MAX_TRY]:
            if hc - hdist(xyz[j], c) <= 0:
                break
            v = xyz[j] - xyz[cur]; nv = np.linalg.norm(v)
            if nv < 1e-9:
                v = np.r_[cxy(c, xyz[cur]) - xyz[cur, :2], 0.0]; nv = np.linalg.norm(v)
            if width_across(xyz, tree, j, v / max(nv, 1e-12)) <= thin_max:
                step = int(j); break
        if step is None:
            return path, 0, 'no_thin_progress'
        length += float(np.linalg.norm(xyz[step] - xyz[cur])); path.append(step); visited.add(step)


def centreline(xyz, tree, path, line):
    """Path (stem end first) + surfels within 3 mm of the line; 2 mm bin medians; 3-bin moving median."""
    P = xyz[path[::-1]] if len(path) else np.zeros((0, 3))
    sp = rt.arclen(P) if len(P) > 1 else np.zeros(len(P)); Lp = float(sp[-1]) if len(P) else 0.0
    S = rt.arclen(line); g = np.arange(0, S[-1] + 1e-9, 1.0); G = rt.at_s(line, g)
    lists = tree.query_ball_point(G, LINE_R_MM)
    near = np.unique(np.concatenate([np.asarray(x, int) for x in lists] + [np.zeros(0, int)]))
    s_line = g[cKDTree(G).query(xyz[near])[1]] if len(near) else np.zeros(0)
    s_all = np.r_[sp, Lp + s_line]; X = np.vstack([P, xyz[near]]) if len(near) else P
    if len(X) < 2:
        return np.zeros(0), np.zeros((0, 3))
    b = np.floor(s_all / BIN_MM).astype(int); keys = np.unique(b)
    M = np.array([np.median(X[b == k], axis=0) for k in keys]); sm = (keys + 0.5) * BIN_MM
    if len(M) >= 3:
        M = np.vstack([M[0], np.array([np.median(M[i - 1:i + 2], axis=0) for i in range(1, len(M) - 1)]), M[-1]])
    return sm, M


def angles(M):
    """θ(s) along a centreline (stem end first): θ0, θ_tip, Δθ, tip drop, s_apex/L, L and the θ(s) samples."""
    M = np.asarray(M, float)
    if len(M) < 2:
        return None
    d = np.diff(M, axis=0); seg = np.linalg.norm(d, axis=1); s = np.r_[0, np.cumsum(seg)]; L = float(s[-1]); mid = (s[:-1] + s[1:]) / 2
    th = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    a, b = mid <= END_MM, mid >= L - END_MM
    th0 = float(th[a].mean()) if a.any() else float(th[0]); tt = float(th[b].mean()) if b.any() else float(th[-1])
    return dict(theta0_deg=th0, theta_tip_deg=tt, dtheta_deg=th0 - tt, tip_drop_mm=float(M[-1, 2] - M[0, 2]),
                s_apex_rel=float(s[int(np.argmax(M[:, 2]))] / L) if L > 0 else float('nan'), L_mm=L, s_mid=mid, theta_s=th)


def truth_angles(poly):
    poly = np.asarray(poly, float); S = rt.arclen(poly); return angles(rt.at_s(poly, np.arange(0, S[-1] + 1e-9, 0.5)))


def leaf_base(xyz, tree, line, c, allowed):
    line = base_first(line, c)
    path, reached, why = grow_petiole(xyz, tree, line[0], c, allowed)
    sm, M = centreline(xyz, tree, path, line)
    a = angles(M)
    if a is None:
        return dict(base_reached=reached, stop=why, path=path, ok=False)
    fit = rt.arch_traits(M) if len(M) >= 3 and rt.arclen(M)[-1] >= 2 * rt.CHORD_MM else None
    return dict(base_reached=reached, stop=why, path=path, ok=True, centre=M, line=line, path_len_mm=float(rt.arclen(xyz[path])[-1]) if len(path) > 1 else 0.0,
                better=fit['better'] if fit else None, arc_rms=fit['arc']['fit_rms_mm'] if fit else None, quad_rms=fit['quad']['fit_rms_mm'] if fit else None, **a)


# ------------------------------------------------------------------ DIV_INTERNODE re-check (§4)
def attachments(xyz, c, leaves, stem_mask):
    """A_k = petiole path end (base_reached) snapped to the nearest stem-region surfel; flags per §4.2."""
    reg = np.where(stem_mask & (hdist(xyz, c) <= STEM_R_MM))[0]; T = cKDTree(xyz[reg]) if len(reg) else None
    out = {}
    for k, r in leaves.items():
        if not r.get('ok') or not r['base_reached'] or not len(r['path']):
            continue
        end = xyz[r['path'][-1]]
        if T is None:
            out[k] = dict(A=end, snap_mm=float('nan'), flags=['outside_stem_region']); continue
        dd, j = T.query(end); A = xyz[reg[j]]; fl = ['snap_gt_3mm'] if dd > SNAP_MAX_MM else []
        P = xyz[r['path'][::-1]]; S = rt.arclen(P) if len(P) > 1 else np.zeros(1); q = rt.at_s(P, [min(END_MM, S[-1])])[0] if len(P) > 1 else P[0]
        az = float(np.degrees(np.arctan2(q[1] - A[1], q[0] - A[0])) % 360)
        out[k] = dict(A=A, snap_mm=float(dd), flags=fl, azimuth_deg=az)
    ks = list(out)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if np.linalg.norm(out[ks[i]]['A'] - out[ks[j]]['A']) <= SHARE_MM:
                out[ks[i]]['flags'].append('shared_A'); out[ks[j]]['flags'].append('shared_A')
    return out


def fold180(d):
    d = np.asarray(d, float) % 360; return np.where(d > 180, 360 - d, d)


def div_internode(att):
    ok = sorted([(v['A'][2], k, v) for k, v in att.items() if not v['flags'] and 'azimuth_deg' in v], key=lambda t: t[0])
    if len(ok) < 2:
        return dict(n=len(ok), divergence_folded=[], internode_mm=[])
    az = [v['azimuth_deg'] for _, _, v in ok]; z = [h for h, _, _ in ok]
    return dict(n=len(ok), order=[k for _, k, _ in ok], divergence_folded=fold180(np.diff(az)).tolist(), internode_mm=np.diff(z).tolist())


# ------------------------------------------------------------------ v2 (build/traits_base_angle_surfel_v2_20261102_01/CONTRACT.md): only the start changes
START_R_MM, BRIDGE_STEP_MM, MIN_WINDOW_MM = 10.0, 0.5, 4.0


def thin_start(xyz, tree, P, c, allowed, thin_max=THIN_MAX_MM):
    """Nearest surfel to P within 10 mm that is thin across the direction to the stand-in, allowed, and nearer the stand-in than P."""
    hp = float(hdist(P, c)); cand = np.array(tree.query_ball_point(P, START_R_MM), int)
    if not len(cand):
        return None
    cand = cand[allowed[cand] & (hdist(xyz[cand], c) < hp)]
    for j in cand[np.argsort(np.linalg.norm(xyz[cand] - P, axis=1))]:
        d = np.r_[cxy(c, xyz[j]) - xyz[j, :2], 0.0]; nd = np.linalg.norm(d)
        if nd < 1e-9:
            return int(j)
        if width_across(xyz, tree, int(j), d / nd) <= thin_max:
            return int(j)
    return None


def leaf_base_v2(xyz, tree, line, c, allowed, thin_max=THIN_MAX_MM):
    line = base_first(line, c); P = line[0]
    j0 = thin_start(xyz, tree, P, c, allowed, thin_max)
    if j0 is None:
        return dict(base_reached=0, stop='no_thin_start', path=[], ok=False, start=None, line=line)
    path, reached, why = grow_petiole(xyz, tree, xyz[j0], c, allowed, thin_max)
    Pp = xyz[path[::-1]] if len(path) else xyz[[j0]]
    sp = rt.arclen(Pp) if len(Pp) > 1 else np.zeros(1); Lp = float(sp[-1])
    br = xyz[j0] + np.outer(np.arange(0, 1 + 1e-9, BRIDGE_STEP_MM / max(np.linalg.norm(P - xyz[j0]), BRIDGE_STEP_MM)), P - xyz[j0])
    Lb = float(np.linalg.norm(P - xyz[j0])); sb = Lp + np.linspace(0, Lb, len(br))
    S = rt.arclen(line); g = np.arange(0, S[-1] + 1e-9, 1.0); G = rt.at_s(line, g)
    near = np.unique(np.concatenate([np.asarray(x, int) for x in tree.query_ball_point(G, LINE_R_MM)] + [np.zeros(0, int)]))
    sl = Lp + Lb + (g[cKDTree(G).query(xyz[near])[1]] if len(near) else np.zeros(0))
    X = np.vstack([Pp, br, xyz[near]]); s_all = np.r_[sp, sb, sl]
    b = np.floor(s_all / BIN_MM).astype(int); keys = np.unique(b)
    M = np.array([np.median(X[b == k], axis=0) for k in keys])
    if len(M) >= 3:
        M = np.vstack([M[0], np.array([np.median(M[i - 1:i + 2], axis=0) for i in range(1, len(M) - 1)]), M[-1]])
    sbin = (keys + 0.5) * BIN_MM
    a = angles(M)
    if a is None:
        return dict(base_reached=0, stop='too_short', path=path, ok=False, start=j0, line=line)
    # map tangent midpoints to the bin s scale: centreline arc length from the stem end, bridge range in the same arc length
    d = np.diff(M, axis=0); s_c = np.r_[0, np.cumsum(np.linalg.norm(d, axis=1))]
    in_bridge_bin = (sbin >= Lp) & (sbin <= Lp + Lb)
    bs = s_c[in_bridge_bin] if in_bridge_bin.any() else np.array([])
    b_lo, b_hi = (float(bs.min()), float(bs.max())) if len(bs) else (np.inf, -np.inf)
    mid = a['s_mid']; win = (mid <= END_MM) & ~((mid >= b_lo) & (mid <= b_hi))
    nonbridge_len = min(END_MM, b_lo if np.isfinite(b_lo) else END_MM) + max(0.0, END_MM - (b_hi if np.isfinite(b_hi) else END_MM))
    if not np.isfinite(b_lo):
        nonbridge_len = END_MM
    stop = why
    if reached and (nonbridge_len < MIN_WINDOW_MM or not win.any()):
        reached, stop = 0, 'window_lt_4mm'
    if win.any():
        a['theta0_deg'] = float(a['theta_s'][win].mean()); a['dtheta_deg'] = a['theta0_deg'] - a['theta_tip_deg']
    fit = rt.arch_traits(M) if len(M) >= 3 and rt.arclen(M)[-1] >= 2 * rt.CHORD_MM else None
    return dict(base_reached=reached, stop=stop, path=path, ok=True, start=j0, bridge=br, bridge_len_mm=Lb, centre=M, line=line,
                path_len_mm=Lp, window_nonbridge_mm=nonbridge_len, theta0_window=(0.0, END_MM),
                better=fit['better'] if fit else None, arc_rms=fit['arc']['fit_rms_mm'] if fit else None, quad_rms=fit['quad']['fit_rms_mm'] if fit else None, **a)


# ------------------------------------------------------------------ v3 (build/traits_base_angle_surfel_v3_20261102_01/CONTRACT.md §4): width where a failed leaf stopped
def width_at_stop(xyz, tree, r, c, allowed):
    """Widths the rule saw at the step where it stopped: all start candidates (no_thin_start) or the tried next steps (no_thin_progress)."""
    if r['stop'] == 'no_thin_start':
        P = r['line'][0]; hp = float(hdist(P, c)); cand = np.array(tree.query_ball_point(P, START_R_MM), int)
        cand = cand[allowed[cand] & (hdist(xyz[cand], c) < hp)] if len(cand) else cand
        if not len(cand):
            return dict(min_width_mm=float('nan'), nearest_width_mm=float('nan'), n_tested=0, note='no_candidate')
        cand = cand[np.argsort(np.linalg.norm(xyz[cand] - P, axis=1))]; w = []
        for j in cand:
            d = np.r_[cxy(c, xyz[j]) - xyz[j, :2], 0.0]; nd = np.linalg.norm(d)
            w.append(0.0 if nd < 1e-9 else width_across(xyz, tree, int(j), d / nd))
        return dict(min_width_mm=float(min(w)), nearest_width_mm=float(w[0]), n_tested=len(w), note='')
    if r['stop'] == 'no_thin_progress':
        path = r['path']; cur = path[-1]; hc = float(hdist(xyz[cur], c)); seen = set(path)
        nb = np.array([j for j in tree.query_ball_point(xyz[cur], LINK_MM) if j not in seen and allowed[j]], int)
        w = []
        if len(nb):
            for j in nb[np.argsort(-(hc - hdist(xyz[nb], c)))][:MAX_TRY]:
                if hc - hdist(xyz[j], c) <= 0:
                    break
                v = xyz[j] - xyz[cur]; nv = np.linalg.norm(v)
                if nv < 1e-9:
                    v = np.r_[cxy(c, xyz[cur]) - xyz[cur, :2], 0.0]; nv = np.linalg.norm(v)
                w.append(width_across(xyz, tree, int(j), v / max(nv, 1e-12)))
        if not w:
            return dict(min_width_mm=float('nan'), nearest_width_mm=float('nan'), n_tested=0, note='no_candidate')
        return dict(min_width_mm=float(min(w)), nearest_width_mm=float(w[0]), n_tested=len(w), note='')
    return dict(min_width_mm=float('nan'), nearest_width_mm=float('nan'), n_tested=0, note='not_measured')
