"""Stem and leaf fit of observed static MTGs to surfels (CONTRACT §2-3). Pure numpy/scipy; all lengths mm, board frame (z up).
The leaf model is the old build's engine writer leaf (writers.leaf_polyline + add_leaf leaflet rule), re-evaluated here for speed."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OLD = ROOT / "build/mtg_build_20261102_01"
for p in (str(ROOT), str(OLD)):
    if p not in sys.path:
        sys.path.insert(0, p)
import writers as W  # noqa: E402  (old build, read only)
from tools import bmodel_mtg_export as BX  # noqa: E402

HUBER = 2.0; STEM_TUBE = 10.0; THIN_W = 4.0; THIN_R = 4.0; BAND = 10.0; R_MIN = 15.0
W_AXIS = 0.3; W_CLICK = 0.5; W_COV = 0.5; COV_R = 5.0; FEW = 30; FEW_R = 15.0
GATE_MED = 3.0; GATE_IN = 5.0; GATE_SHARE = 0.90; SMOOTH_MAX = 45.0


def huber(d, delta=HUBER):
    d = np.abs(d); return np.where(d <= delta, d * d / (2 * delta), d - delta / 2)


# ------------------------------------------------------------------ polyline helpers
def resample(P, step=1.0):
    P = np.asarray(P, float); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return P[:1].copy(), np.zeros(1)
    t = np.arange(0, s[-1] + 1e-9, step)
    if t[-1] < s[-1]:
        t = np.append(t, s[-1])
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1), t


def point_seg_dist(Q, P):
    """distance of points Q (n,3) to polyline P (m,3)."""
    A = P[:-1]; B = P[1:]; AB = B - A; L2 = np.maximum((AB * AB).sum(1), 1e-12)
    t = np.clip(((Q[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    C = A[None] + t[..., None] * AB[None]
    return np.sqrt(((Q[:, None, :] - C) ** 2).sum(2)).min(1)


# ------------------------------------------------------------------ stem
def local_width(X, idx, tree):
    out = np.full(len(idx), np.inf)
    for j, i in enumerate(idx):
        nb = tree.query_ball_point(X[i], THIN_R)
        if len(nb) < 8:
            continue
        ev = np.linalg.eigvalsh(np.cov(X[nb].T)); out[j] = 4 * math.sqrt(max(ev[1], 0.0))
    return out


def fit_stem(C, X, leaf_mask, tree=None):
    """C: start curve (mm). X: all surfels (mm). leaf_mask: assigned to a leaf. Returns dict(curve, start, flags, report...)."""
    C1, s1 = resample(C, 1.0); S = s1[-1]
    tC = cKDTree(C1); d, j = tC.query(X, distance_upper_bound=STEM_TUBE + 1)
    near = np.nonzero(np.isfinite(d) & (d <= STEM_TUBE))[0]
    tree = tree or cKDTree(X)
    thin = local_width(X, near, tree) <= THIN_W
    cand = near[(~leaf_mask[near]) | thin]; s_c = s1[j[cand]]
    nb = int(math.floor(S / BAND)) + 1; band_n = np.bincount(np.minimum((s_c // BAND).astype(int), nb - 1), minlength=nb)
    rep = dict(start_length_mm=float(S), candidates=int(len(cand)), band_counts=band_n.tolist())
    sc = np.arange(0, S + 1e-9, BAND)
    if sc[-1] < S:
        sc = np.append(sc, S)
    if len(cand) < 100 or len(sc) < 2:
        rep.update(flags=["stem_kept_clicked"], kept_bands=nb, curve=C1, shift_max=0.0, shift_median=0.0); return rep
    band_of = np.minimum((sc // BAND).astype(int), nb - 1); free = band_n[band_of] >= 20; free[0] = False
    Y = X[cand]
    def curve(p):
        D = np.zeros((len(sc), 3)); D[free] = p.reshape(-1, 3)
        f = CubicSpline(sc, D, axis=0) if len(sc) >= 3 else (lambda t: np.stack([np.interp(t, sc, D[:, k]) for k in range(3)], 1))
        return C1 + f(s1)
    def cost(p):
        P = curve(p); dd = cKDTree(P).query(Y)[0]; da = np.linalg.norm(P - C1, axis=1)
        k = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=1)
        return float(huber(dd).mean() + W_AXIS * da.mean() + 100 * (np.maximum(0, k - 1 / R_MIN) ** 2).sum())
    if not free.any():
        rep.update(flags=["stem_kept_clicked"], kept_bands=nb, curve=C1, shift_max=0.0, shift_median=0.0); return rep
    r = minimize(cost, np.zeros(3 * free.sum()), method="L-BFGS-B", options=dict(maxiter=300))
    P = curve(r.x); sh = np.linalg.norm(P - C1, axis=1)
    kept = int((band_n < 20).sum())
    rep.update(curve=P, flags=(["stem_kept_clicked_band"] if kept else []), kept_bands=kept, shift_max=float(sh.max()), shift_median=float(np.median(sh)), cost=float(r.fun), iters=int(r.nit))
    top = P[s1 >= S - 10]; dtop = tree.query(top)[0]; axis_dir = (P[-1] - P[0]) / max(np.linalg.norm(P[-1] - P[0]), 1e-9)
    tdir = (top[-1] - top[0]) / max(np.linalg.norm(top[-1] - top[0]), 1e-9)
    rep.update(top_dist_to_surfel_median=float(np.median(dtop)), top_angle_deg=float(math.degrees(math.acos(np.clip(tdir @ axis_dir, -1, 1)))))
    rep["top_curls"] = rep["top_angle_deg"] > 60
    if rep["top_curls"]:
        rep["flags"].append("top_curls")
    return rep


def point_at_s(P, s):
    Q, t = resample(P, 0.25); return np.array([np.interp(s, t, Q[:, k]) for k in range(3)])


# ------------------------------------------------------------------ leaf model
def leaflet_points(pl, m, n_inter, area):
    """5 points per blade as in writers.add_leaf (base, centre, tip, two edge midpoints)."""
    pl = np.asarray(pl, float); units = m * 2 * W.SHARES["P"] + W.SHARES["T"] + n_inter * W.SHARES["I"]; a_unit = area / units; up = np.array([0, 0, 1.0])
    out = []
    def blade(s_idx, side, share):
        p = pl[s_idx]; t = pl[min(s_idx + 1, len(pl) - 1)] - pl[max(s_idx - 1, 0) if s_idx == len(pl) - 1 else s_idx]
        t = t / (np.linalg.norm(t) or 1); sd = np.cross(t, up); sd = sd / (np.linalg.norm(sd) or 1)
        a = math.radians(W.PAIR_ANGLE_DEG); d = t if side == 0 else (t * math.cos(a) + sd * side * math.sin(a)); d = d / np.linalg.norm(d)
        n = up - d * float(d @ up); n = n / (np.linalg.norm(n) or 1); b = np.cross(n, d)
        Lf = math.sqrt(share * a_unit * W.LW / W.kappa()); Wf = Lf / W.LW
        c = p + d * Lf / 2; out.extend([p, c, p + d * Lf, c + b * Wf / 2, c - b * Wf / 2])
    for j in range(1, m + 1):
        blade(j, 1, W.SHARES["P"]); blade(j, -1, W.SHARES["P"])
    for q in range(n_inter):
        blade(1, 1 if q % 2 == 0 else -1, W.SHARES["I"])
    blade(len(pl) - 1, 0, W.SHARES["T"])
    return np.asarray(out)


def pieces_polyline(base, dirs, L, m):
    """Level 2: petiole (PET_SHARE L) then m segments, each with its own (az, el) in degrees."""
    pts = [np.asarray(base, float)]; lens = [W.PET_SHARE * L] + [(1 - W.PET_SHARE) * L / max(m, 1)] * m
    for (az, el), ln in zip(dirs, lens):
        a, e = math.radians(az), math.radians(el); pts.append(pts[-1] + ln * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]))
    return np.asarray(pts)


class LeafFit:
    def __init__(self, base, leaflets, area, set_pts, click, any_tree):
        self.base = np.asarray(base, float); self.m, self.n_inter = W.pairs_of(leaflets); self.area = area
        self.set = set_pts; self.tset = cKDTree(set_pts); self.click = resample(click, 1.0)[0] if len(click) > 1 else np.asarray(click, float); self.any = any_tree

    def model(self, pl):
        """(cost/gate points = rachis every 2 mm + leaflet centres, coverage points = rachis + 5 points per blade) - CONTRACT A2."""
        R = resample(pl, 2.0)[0]; B = leaflet_points(pl, self.m, self.n_inter, self.area)
        return np.vstack([R, B[1::5]]), np.vstack([R, B])

    def cost_pl(self, pl):
        M, Mc = self.model(pl); d = self.tset.query(M)[0]; dc = point_seg_dist(self.click, np.asarray(pl, float)).mean() if len(self.click) else 0.0
        cov = float((cKDTree(Mc).query(self.set, distance_upper_bound=COV_R)[0] <= COV_R).mean())
        return float(huber(d).mean() + W_CLICK * dc + W_COV * (1 - cov))

    def pl1(self, x):
        az, th, dt, L = x; return np.asarray(W.leaf_polyline(self.base, az, th, dt, L, self.m)[0], float)

    def fit1(self, starts, bounds):
        best = None
        for x0 in starts:
            x0 = np.clip(np.asarray(x0, float), [b[0] for b in bounds], [b[1] for b in bounds])
            r = minimize(lambda x: self.cost_pl(self.pl1(x)), x0, method="Powell", bounds=bounds, options=dict(xtol=1e-2, ftol=1e-4, maxfev=3000))
            if best is None or r.fun < best.fun:
                best = r
        return best

    def fit2(self, x1, az_bounds):
        """Level 2: [az0, el0, az1, el1, ..., L]; start = Level-1 piece directions."""
        az, th, dt, L = x1; pl = self.pl1(x1); dirs = []
        for a, b in zip(pl[:-1], pl[1:]):
            v = b - a; dirs += [math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(np.clip(v[2] / np.linalg.norm(v), -1, 1)))]
        x0 = np.array(dirs + [L]); n = len(dirs) // 2
        def unwrap(a):
            return az + ((a - az + 180) % 360 - 180)
        x0[0:2 * n:2] = [unwrap(a) for a in x0[0:2 * n:2]]
        bnds = [(az_bounds[0], az_bounds[1]) if i % 2 == 0 else (-90.0, 90.0) for i in range(2 * n)] + [(x1[3] * 0.999, x1[3] * 1.001)]
        bnds[-1] = (L - 1e-6, L + 1e-6)                          # CONTRACT A2(c): L held at the Level-1 value
        def pen(x):
            V = []
            for i in range(n):
                a, e = math.radians(x[2 * i]), math.radians(x[2 * i + 1]); V.append([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
            V = np.asarray(V); ang = np.degrees(np.arccos(np.clip((V[1:] * V[:-1]).sum(1), -1, 1)))
            return 100 * (np.maximum(0, ang - SMOOTH_MAX) ** 2).sum()
        f = lambda x: self.cost_pl(pieces_polyline(self.base, [(x[2 * i], x[2 * i + 1]) for i in range(n)], x[-1], self.m)) + pen(x)
        x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])
        r = minimize(f, x0, method="Powell", bounds=bnds, options=dict(xtol=1e-2, ftol=1e-4, maxfev=4000))
        return r, pieces_polyline(self.base, [(r.x[2 * i], r.x[2 * i + 1]) for i in range(n)], r.x[-1], self.m)

    def gate(self, pl):
        M, _ = self.model(pl); d = self.any.query(M)[0]
        return dict(median_dist_mm=float(np.median(d)), share_within_5mm=float((d <= GATE_IN).mean()), passed=bool(np.median(d) <= GATE_MED and (d <= GATE_IN).mean() >= GATE_SHARE))


def az_el_first(pl, length=10.0):
    d = BX.first_direction(np.asarray(pl, float), length)
    return BX.az_el(d) if d is not None else (None, None)


# ------------------------------------------------------------------ mode B (CONTRACT B): theta0 and dtheta fixed at the spec values
def fitB1(F, th, dt, bounds_phi, bounds_L, x0):
    """Level 1 of mode B: (phi, L) with theta0 / dtheta fixed; Powell, one start."""
    f = lambda x: F.cost_pl(F.pl1((x[0], th, dt, x[1])))
    lo = [bounds_phi[0], bounds_L[0]]; hi = [bounds_phi[1], bounds_L[1]]; x0 = np.clip(np.asarray(x0, float), lo, hi)
    return minimize(f, x0, method="Powell", bounds=[bounds_phi, bounds_L], options=dict(xtol=1e-2, ftol=1e-4, maxfev=2000))


def fitB2(F, phi, th, dt, L, bounds_phi):
    """Level 2 of mode B: per-piece azimuth only; elevations = the template theta0 / dtheta rule (writers.leaf_polyline); L fixed."""
    pl = F.pl1((phi, th, dt, L)); v = np.diff(pl, axis=0); el = [math.degrees(math.asin(np.clip(x[2] / np.linalg.norm(x), -1, 1))) for x in v]; n = len(el)

    def pen(a):
        V = np.array([[math.cos(math.radians(ai)) * math.cos(math.radians(ei)), math.sin(math.radians(ai)) * math.cos(math.radians(ei)), math.sin(math.radians(ei))] for ai, ei in zip(a, el)])
        ang = np.degrees(np.arccos(np.clip((V[1:] * V[:-1]).sum(1), -1, 1))); return 100 * (np.maximum(0, ang - SMOOTH_MAX) ** 2).sum()
    f = lambda a: F.cost_pl(pieces_polyline(F.base, list(zip(a, el)), L, F.m)) + pen(a)
    r = minimize(f, np.full(n, phi), method="Powell", bounds=[bounds_phi] * n, options=dict(xtol=1e-2, ftol=1e-4, maxfev=3000))
    return r, pieces_polyline(F.base, list(zip(r.x, el)), L, F.m)
