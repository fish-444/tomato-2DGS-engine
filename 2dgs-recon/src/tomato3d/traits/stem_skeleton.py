"""Stem line per plant from the base-side ends of the skeleton leaf lines (build/traits_stem_from_skeleton_20261102_01/CONTRACT.md §2, §5, §6).
Units mm, z up. Lines are (n, 3) click polylines in clicked order."""
from __future__ import annotations

import numpy as np

from tomato3d.traits.base_angle import StemLine

LOW_SHARE, AMBIG_MM, MAX_ITER = 0.30, 5.0, 20
HUBER_MM, IRLS_ITER, IRLS_TOL = 3.0, 100, 1e-6
TILT_MAX_DEG, INLIER_MM, RESID_MAX_MM, MIN_LINES = 15.0, 8.0, 8.0, 5
CHORD_MM, MIN_L_MM = 10.0, 25.0


def heights(z):
    return float(np.percentile(z, 0.5)), float(np.percentile(z, 99.5))


def tilt_deg(d):
    d = np.asarray(d, float); return float(np.degrees(np.arccos(min(1.0, abs(d[2]) / np.linalg.norm(d)))))


def hres(P, line):
    """Horizontal distance of points P to the line evaluated at their own heights."""
    P = np.atleast_2d(P); return np.linalg.norm(P[:, :2] - line.xy_at(P[:, 2]), axis=1)


def huber_line(E):
    """IRLS Huber (scale HUBER_MM) on horizontal residuals, x(z), y(z) linear, slope limited to TILT_MAX_DEG."""
    E = np.asarray(E, float); zb = E[:, 2].mean(); zz = E[:, 2] - zb; w = np.ones(len(E)); bmax = np.tan(np.radians(TILT_MAX_DEG))
    a = b = None; at_limit = False
    for _ in range(IRLS_ITER):
        W = w.sum(); zw = (w * zz).sum() / W; den = (w * (zz - zw) ** 2).sum()
        bn = np.array([(w * (zz - zw) * E[:, k]).sum() / den if den > 1e-12 else 0.0 for k in (0, 1)])
        nb = np.linalg.norm(bn); at_limit = nb > bmax
        if at_limit:
            bn = bn * (bmax / nb)
        an = np.array([(w * (E[:, k] - bn[k] * zz)).sum() / W for k in (0, 1)])
        done = a is not None and max(np.abs(an - a).max(), np.abs(bn - b).max() * 100) < IRLS_TOL
        a, b = an, bn
        r = np.linalg.norm(E[:, :2] - (a + np.outer(zz, b)), axis=1); w = np.where(r <= HUBER_MM, 1.0, HUBER_MM / np.maximum(r, 1e-12))
        if done:
            break
    d = np.array([b[0], b[1], 1.0]); d /= np.linalg.norm(d)
    return StemLine(np.array([a[0], a[1], zb]), d), bool(at_limit)


def vertical(xy, z0=0.0):
    return StemLine(np.array([xy[0], xy[1], z0]), np.array([0.0, 0.0, 1.0]))


def fit_from_lines(lines, z_b, z_t):
    """lines: dict id -> (n,3). Returns dict(status, line, tilt_deg, flags, per-line table, iterations)."""
    ids = list(lines); ends = {k: (np.asarray(lines[k], float)[0], np.asarray(lines[k], float)[-1]) for k in ids}
    if not ids:
        return dict(status='stem_unknown', line=None, tilt_deg=None, flags=['no_lines'], rows=[], iterations=0, resid_median_mm=None, n_fit=0)
    allE = np.array([e for k in ids for e in ends[k]]); low = allE[allE[:, 2] <= z_b + LOW_SHARE * (z_t - z_b)]
    if len(low) < 3:
        low = allE[np.argsort(allE[:, 2])[:max(3, int(np.ceil(LOW_SHARE * len(allE))))]]
    line = vertical(np.median(low[:, :2], axis=0), float(np.median(low[:, 2]))); flags, prev, at_limit, it = [], None, False, 0
    for it in range(1, MAX_ITER + 1):
        d0 = np.array([hres(ends[k][0], line)[0] for k in ids]); d1 = np.array([hres(ends[k][1], line)[0] for k in ids])
        first = d0 <= d1; amb = np.abs(d0 - d1) < AMBIG_MM; state = (tuple(first), tuple(amb))
        if state == prev:
            break
        prev = state; use = [k for k, a in zip(ids, amb) if not a]
        if len(use) < 2:
            break
        E = np.array([ends[k][0] if f else ends[k][1] for k, f, a in zip(ids, first, amb) if not a]); line, at_limit = huber_line(E)
    else:
        flags.append('not_converged')
    d0 = np.array([hres(ends[k][0], line)[0] for k in ids]); d1 = np.array([hres(ends[k][1], line)[0] for k in ids])
    first = d0 <= d1; amb = np.abs(d0 - d1) < AMBIG_MM
    rows = []
    for k, f, a, x0, x1 in zip(ids, first, amb, d0, d1):
        B = ends[k][0] if f else ends[k][1]; r = float(min(x0, x1))
        rows.append(dict(leaf=k, base_end='first_click' if f else 'last_click', d_first_mm=float(x0), d_last_mm=float(x1), direction_ambiguous=bool(a),
                         base_x=float(B[0]), base_y=float(B[1]), base_z=float(B[2]), resid_mm=r, used_in_fit=bool(not a), inlier=bool((not a) and r <= INLIER_MM)))
    used = [r for r in rows if r['used_in_fit']]; med = float(np.median([r['resid_mm'] for r in used])) if used else None
    if at_limit:
        flags.append('tilt_at_limit')
    status = 'stem_unknown' if (len(used) < MIN_LINES or med is None or med > RESID_MAX_MM) else 'fitted'
    if len(used) < MIN_LINES:
        flags.append('lt_5_usable_lines')
    elif med is not None and med > RESID_MAX_MM:
        flags.append('resid_median_gt_8mm')
    return dict(status=status, line=line, tilt_deg=tilt_deg(line.d), flags=flags, rows=rows, iterations=it, resid_median_mm=med, n_fit=len(used))


def ls_line(P):
    """Plain least-squares x(z), y(z) through points (no tilt limit), e.g. the user-clicked axis."""
    P = np.asarray(P, float); zb = P[:, 2].mean(); zz = P[:, 2] - zb; den = (zz ** 2).sum()
    b = np.array([(zz * (P[:, k] - P[:, k].mean())).sum() / den for k in (0, 1)]); a = P[:, :2].mean(0)
    d = np.array([b[0], b[1], 1.0]); return StemLine(np.array([a[0], a[1], zb]), d / np.linalg.norm(d))


def project(line, p):
    t = float((np.asarray(p, float) - line.p0) @ line.d); return t, line.p0 + t * line.d


def azimuth(line, p):
    """Azimuth (deg) of p - A around the line, in the plane perpendicular to it (reference = x axis projected)."""
    t, A = project(line, p); v = np.asarray(p, float) - A; d = line.d
    e1 = np.array([1.0, 0, 0]) - d[0] * d; e1 /= np.linalg.norm(e1); e2 = np.cross(d, e1)
    return float(np.degrees(np.arctan2(v @ e2, v @ e1)) % 360), t, A


def fold(d):
    d = np.asarray(d, float) % 360; return np.where(d > 180, 360 - d, d)


def arclen(P):
    P = np.asarray(P, float); return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def at_s(P, s):
    P = np.asarray(P, float); S = arclen(P); return np.array([np.interp(s, S, P[:, k]) for k in range(3)])


def div_internode(line, bases):
    """bases: list of (leaf, xyz) base-side ends. Returns per-leaf table (ordered by t) and plant summary."""
    rows = []
    for k, B in bases:
        psi, t, A = azimuth(line, B); rows.append(dict(leaf=k, t_mm=t, psi_deg=psi, A=A))
    rows.sort(key=lambda r: r['t_mm'])
    psi = np.array([r['psi_deg'] for r in rows]); t = np.array([r['t_mm'] for r in rows])
    dv = fold(np.diff(psi)); it = np.diff(t)
    for i, r in enumerate(rows):
        r['divergence_to_next_deg'] = float(dv[i]) if i < len(dv) else None; r['internode_to_next_mm'] = float(it[i]) if i < len(it) else None
    S = dict(n_leaves=len(rows), divergence_mean_deg=float(dv.mean()) if len(dv) else None, divergence_sd_deg=float(dv.std(ddof=1)) if len(dv) > 1 else None,
             internode_mean_mm=float(it.mean()) if len(it) else None, internode_median_mm=float(np.median(it)) if len(it) else None,
             internode_sd_mm=float(it.std(ddof=1)) if len(it) > 1 else None)
    return rows, S


def chord_elevation(A, P_base_first):
    """Elevation (deg) of the vector from A to the point CHORD_MM along the clicked line from its base-side end."""
    P = np.asarray(P_base_first, float); L = arclen(P)[-1]; q = at_s(P, min(CHORD_MM, L)); v = q - np.asarray(A, float)
    return float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))), bool(L < CHORD_MM)


# ---- v6: the user's clicked main axis as the stem (build/traits_clicked_axis_20261102_01/CONTRACT.md §2-§5)
class AxisCurve:
    """x(z), y(z) polynomial (deg 1 or 2) through clicked axis points, sampled densely for projections."""

    def __init__(self, pts, deg=1, pad=40.0, step=0.1):
        P = np.asarray(pts, float); P = P[np.argsort(P[:, 2])]; self.pts, self.deg = P, deg
        self.cx, self.cy = np.polyfit(P[:, 2], P[:, 0], deg), np.polyfit(P[:, 2], P[:, 1], deg)
        self.z_lo, self.z_hi = float(P[0, 2]), float(P[-1, 2])
        z = np.arange(self.z_lo - pad, self.z_hi + pad + 1e-9, step); self.S = np.c_[self.xy_at(z), z]; self.s = arclen(self.S)

    def xy_at(self, z):
        z = np.asarray(z, float); return np.stack([np.polyval(self.cx, z), np.polyval(self.cy, z)], axis=-1)

    def resid_rms(self):
        return float(np.sqrt(np.mean(np.sum((self.pts[:, :2] - self.xy_at(self.pts[:, 2])) ** 2, axis=1))))

    def tilt_deg(self):
        return float(np.degrees(np.arctan(np.hypot(self.cx[-2], self.cy[-2])))) if self.deg == 1 else None

    def nearest(self, p):
        p = np.asarray(p, float); i = int(np.argmin(np.linalg.norm(self.S - p, axis=1))); i0, i1 = max(i - 1, 0), min(i + 1, len(self.S) - 1)
        t = self.S[i1] - self.S[i0]; return self.S[i], float(self.s[i]), t / np.linalg.norm(t), float(np.linalg.norm(self.S[i] - p))


def wrap180(d):
    d = (np.asarray(d, float) + 180.0) % 360.0 - 180.0; return np.where(d == -180.0, 180.0, d)


def circ_stats(deg):
    u = np.exp(1j * np.radians(np.asarray(deg, float)))
    if len(u) == 0:
        return dict(n=0, R=None, circ_mean_deg=None, circ_sd_deg=None)
    m = u.mean(); R = float(abs(m))
    return dict(n=int(len(u)), R=R, circ_mean_deg=float(np.degrees(np.angle(m))), circ_sd_deg=float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12))))))


def axis_attach(curve, lines, r_lo=1.5, r_hi=15.0, range_mm=10.0, tie_mm=2.0, flip=(), order_by='z', user_order_wrong=()):
    """Per-leaf base end, A_k, s_k, phi_k, r_k, flags; leaf order; pairs with signed d_phi and internode (CONTRACT §3-§5).
    A pair is used only if neither leaf is flagged (flagged leaves keep their place in the order)."""
    rows = []
    for k, L in lines.items():
        L = np.asarray(L, float); A0, s0, t0, d0 = curve.nearest(L[0]); A1, s1, t1, d1 = curve.nearest(L[-1])
        first = (d0 <= d1) != (k in flip)
        B, A, s, t = (L[0], A0, s0, t0) if first else (L[-1], A1, s1, t1)
        v = B - A; e1 = np.array([1.0, 0, 0]) - t[0] * t; e1 /= np.linalg.norm(e1); e2 = np.cross(t, e1); r = float(np.linalg.norm(v))
        fl = []
        if abs(d0 - d1) < AMBIG_MM: fl.append('direction_ambiguous')
        if B[2] < curve.z_lo - range_mm or B[2] > curve.z_hi + range_mm: fl.append('outside_axis_range')
        if r < r_lo: fl.append('r_lt_lo')
        if r > r_hi: fl.append('r_gt_hi')
        if k in user_order_wrong: fl.append('user_order_wrong')
        rows.append(dict(leaf=k, base_end='first_click' if first else 'last_click', user_flipped=k in flip, d_first_mm=d0, d_last_mm=d1,
                         base_x=B[0], base_y=B[1], base_z=B[2], A_x=A[0], A_y=A[1], A_z=A[2], s_mm=s, r_mm=r,
                         phi_deg=float(np.degrees(np.arctan2(v @ e2, v @ e1)) % 360), clicked_L_mm=float(arclen(L)[-1]), flags=fl))
    rows.sort(key=lambda r: r['A_z'] if order_by == 'z' else r['s_mm'])
    for a, b in zip(rows[:-1], rows[1:]):
        if abs(b['A_z'] - a['A_z']) < tie_mm:
            for r in (a, b):
                if 'tie' not in r['flags']: r['flags'].append('tie')
    for i, r in enumerate(rows, start=1): r['order'] = i
    pairs = [dict(leaf_a=a['leaf'], leaf_b=b['leaf'], order_a=a['order'], used=not a['flags'] and not b['flags'],
                  dropped_because=';'.join(sorted(set(a['flags'] + b['flags']))), d_phi_deg=float(wrap180(b['phi_deg'] - a['phi_deg'])),
                  internode_mm=float(b['s_mm'] - a['s_mm'])) for a, b in zip(rows[:-1], rows[1:])]
    return rows, pairs


def spiral_summary(pairs):
    """Signed d_phi stats, handedness sign = sign of sin(circular mean), sign-corrected stats, internode over the used pairs."""
    d = np.array([p['d_phi_deg'] for p in pairs if p['used']]); C = circ_stats(d)
    sign = 1 if C['circ_mean_deg'] is None or np.sin(np.radians(C['circ_mean_deg'])) >= 0 else -1
    dc = wrap180(sign * d); Cc = circ_stats(dc); it = np.array([p['internode_mm'] for p in pairs if p['used']])
    return dict(n_pairs=int(len(d)), R=C['R'], circ_mean_deg=C['circ_mean_deg'], circ_sd_deg=C['circ_sd_deg'], handedness_sign=sign,
                corrected_circ_mean_deg=Cc['circ_mean_deg'], corrected_circ_mean_0_360=None if Cc['circ_mean_deg'] is None else Cc['circ_mean_deg'] % 360,
                shows_spiral=bool(C['R'] is not None and C['R'] >= 0.5 and len(d) >= 5), d_phi_sorted=sorted(np.round(d, 1).tolist()),
                d_phi_corrected=np.round(dc, 1).tolist(), internode_n=int(len(it)), internode_mean_mm=float(it.mean()) if len(it) else None,
                internode_sd_mm=float(it.std(ddof=1)) if len(it) > 1 else None)
