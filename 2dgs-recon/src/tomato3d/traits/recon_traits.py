"""Trait extractors for reconstructed (or synthetic) plants. Thresholds are fixed by build/traits_recon_20261101_01/CONTRACT.md.

Conventions: mm, z up, normals oriented +z, weights = one-sided area (mm^2). Angles in degrees.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

LAD_BINS = np.arange(0, 95, 5.0)
CHI_GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
ROSE_BINS = np.arange(0, 361, 30.0)
A3_MIN_INCL = 10.0
A3_VOXEL = 10.0
RAYLEIGH_ALPHA = 0.01
CELL_PRIMARY, CELL_SENS = 1.0, 2.0
PROFILE_DZ = 5.0
CHORD_MM = 10.0
DENSIFY_R, DENSIFY_BIN = 3.0, 2.0


def plus_z(n):
    n = np.asarray(n, float)
    return np.where(n[:, 2:3] < 0, -n, n)


def inclination(n):
    return np.degrees(np.arccos(np.clip(np.abs(np.asarray(n)[:, 2]), 0, 1)))


def wmean_sd(x, w):
    m = float(np.average(x, weights=w)); return m, float(math.sqrt(max(np.average((x - m) ** 2, weights=w), 0.0)))


# ------------------------------------------------------------------ A1
def campbell_pdf(theta_deg, chi):
    t = np.radians(theta_deg); lam = chi + 1.774 * (chi + 1.182) ** -0.733
    return 2 * chi ** 3 * np.sin(t) / (lam * (np.cos(t) ** 2 + chi ** 2 * np.sin(t) ** 2) ** 2)


def lad(theta, w):
    theta, w = np.asarray(theta, float), np.asarray(w, float)
    hist, _ = np.histogram(theta, LAD_BINS, weights=w); hist = hist / max(w.sum(), 1e-12)
    m, sd = wmean_sd(theta, w)
    tc = np.clip(theta, 0.05, 89.95)
    ll = [float((w * np.log(np.maximum(campbell_pdf(tc, c), 1e-300))).sum()) for c in CHI_GRID]
    chi = float(CHI_GRID[int(np.argmax(ll))])
    t = np.clip(theta / 90.0, 0.001, 0.999); mu, s = wmean_sd(t, w); v = s * s
    k = mu * (1 - mu) / v - 1 if v > 0 else float('nan')
    return dict(mean_deg=m, sd_deg=sd, hist_5deg=hist.round(5).tolist(), ellipsoidal_chi=chi, beta_nu=mu * k, beta_omega=(1 - mu) * k,
                weight_sum=float(w.sum()), n=int(len(theta)))


def lad_by_thirds(theta, w, h):
    lo, hi = np.percentile(h, [1, 99]); edges = np.linspace(lo, hi, 4); out = []
    for i in range(3):
        m = (h >= edges[i]) & (h <= edges[i + 1]) if i == 2 else (h >= edges[i]) & (h < edges[i + 1])
        out.append(dict(layer=i + 1, h_from_mm=float(edges[i]), h_to_mm=float(edges[i + 1]), **(lad(theta[m], w[m]) if m.sum() > 10 and w[m].sum() > 0 else dict(n=int(m.sum())))))
    return out


# ------------------------------------------------------------------ A2
def occupancy_area(xy, cell):
    return float(len(np.unique(np.floor(np.asarray(xy) / cell).astype(np.int64), axis=0)) * cell * cell)


def size_traits(xyz, w, h, h_all):
    q = np.asarray(xyz)[:, :2]; c = q - q.mean(0); ev, evec = np.linalg.eigh(np.cov(c.T)); pr = c @ evec[:, ::-1]
    ext = lambda v: float(np.ptp(np.percentile(v, [0.5, 99.5])))
    top = math.ceil(max(h.max(), 1e-9) / PROFILE_DZ); prof, _ = np.histogram(h, np.arange(0, (top + 1) * PROFILE_DZ, PROFILE_DZ), weights=w)
    return dict(height_p995_mm=float(np.percentile(h_all, 99.5)), height_max_mm=float(np.max(h_all)),
                proj_area_1mm_mm2=occupancy_area(q, CELL_PRIMARY), proj_area_2mm_mm2=occupancy_area(q, CELL_SENS),
                width_x_mm=ext(q[:, 0]), width_y_mm=ext(q[:, 1]), width_major_mm=ext(pr[:, 0]), width_minor_mm=ext(pr[:, 1]),
                area_total_mm2=float(np.sum(w)), profile_5mm_mm2=prof.round(3).tolist())


# ------------------------------------------------------------------ A3
def azimuth_traits(xyz, n, w):
    n = plus_z(n); th = inclination(n); m = th >= A3_MIN_INCL
    x, n, w = np.asarray(xyz)[m], n[m], np.asarray(w)[m]
    phi = np.degrees(np.arctan2(n[:, 1], n[:, 0])) % 360
    rose, _ = np.histogram(phi, ROSE_BINS, weights=w); rose = rose / max(w.sum(), 1e-12)
    u = np.c_[np.cos(np.radians(phi)), np.sin(np.radians(phi))]
    R = np.average(u, axis=0, weights=w); rbar = float(np.linalg.norm(R)); mdir = float(np.degrees(np.arctan2(R[1], R[0])) % 360)
    key = np.floor(x / A3_VOXEL).astype(np.int64); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    V = np.zeros((inv.max() + 1, 2)); np.add.at(V, inv, u * w[:, None]); nv = np.linalg.norm(V, axis=1); V = V[nv > 0] / nv[nv > 0, None]
    nvox = len(V); Rv = float(np.linalg.norm(V.sum(0)) / max(nvox, 1)); Z = nvox * Rv * Rv
    p = float(math.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * nvox) - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * nvox ** 2))) if nvox else float('nan')
    p = min(max(p, 0.0), 1.0)
    return dict(rose_30deg=rose.round(5).tolist(), rbar_surfel=rbar, mean_dir_deg=mdir, n_voxels=nvox, rbar_voxel=Rv, rayleigh_p=p,
                decision='non_random' if p < RAYLEIGH_ALPHA else 'random_not_rejected', n_used=int(m.sum()))


# ------------------------------------------------------------------ C1 (same arithmetic as make_T_curved.phi_psi)
def wslope_corr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w); cxy = np.average((x - mx) * (y - my), weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return cxy / vx, cxy / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def phi_psi(x, n, w):
    n = plus_z(n); x = np.asarray(x, float); w = np.asarray(w, float)
    mh = (n * w[:, None]).sum(0); mh /= np.linalg.norm(mh)
    c0 = np.average(x, axis=0, weights=w); xp = (x - c0) - np.outer((x - c0) @ mh, mh)
    cov = (xp * w[:, None]).T @ xp / w.sum(); ev, evec = np.linalg.eigh(cov); es = evec[:, -1]; es -= (es @ mh) * mh; es /= np.linalg.norm(es); ec = np.cross(mh, es)
    s, c = (x - c0) @ es, (x - c0) @ ec
    ac = np.degrees(np.arctan2(n @ ec, n @ mh)); as_ = np.degrees(np.arctan2(n @ es, n @ mh))
    if (c > 0).sum() == 0 or (c < 0).sum() == 0:
        return dict(phi_deg=float('nan'), psi_deg=float('nan'), flags=['one_sided'])
    phi = (float(np.median(np.abs(ac[c > 0]))) + float(np.median(np.abs(ac[c < 0])))) / 2; psi = float(np.median(np.abs(as_)))
    sl_c, r_c = wslope_corr(c, n @ ec, w); sl_s, r_s = wslope_corr(s, n @ es, w); flags = []
    if abs(r_c) < 0.1: flags.append('phi_sign_undetermined')
    if abs(r_s) < 0.1: flags.append('psi_sign_undetermined')
    return dict(phi_deg=phi, psi_deg=psi, s_phi=(1 if sl_c < 0 else -1) if abs(r_c) >= 0.1 else 1, s_psi=(1 if sl_s < 0 else -1) if abs(r_s) >= 0.1 else 1, flags=flags)


# ------------------------------------------------------------------ B: skeleton lines
def arclen(P):
    P = np.asarray(P, float); return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def at_s(P, s):
    P = np.asarray(P, float); S = arclen(P); return np.c_[[np.interp(s, S, P[:, k]) for k in range(3)]].T


def densify(poly, pts):
    """Skeleton densified by nearby rachis points: bins of DENSIFY_BIN mm arc length, bin mean of points within DENSIFY_R mm."""
    poly = np.asarray(poly, float); S = arclen(poly); L = S[-1]
    s_grid = np.arange(0, L + 1e-9, DENSIFY_BIN / 4); G = at_s(poly, s_grid)
    out_s = np.arange(DENSIFY_BIN / 2, L, DENSIFY_BIN); base = at_s(poly, out_s)
    if pts is not None and len(pts):
        d, i = cKDTree(G).query(pts); m = d <= DENSIFY_R; sp = s_grid[i[m]]; P = pts[m]
        b = np.floor(sp / DENSIFY_BIN).astype(int)
        for k in range(len(out_s)):
            sel = b == k
            if sel.any(): base[k] = P[sel].mean(0)
    return np.vstack([poly[0], base, poly[-1]])


def chord_elev(P, s0, s1):
    a, b = at_s(P, [s0, s1]); d = b - a; return float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))), d / max(np.linalg.norm(d), 1e-12)


def arch_curve(theta0, dtheta, L, kind, n=200):
    s = np.linspace(0, L, n); g = s / L if kind == 'arc' else (s / L) ** 2; th = np.radians(theta0 - dtheta * g)
    ds = np.diff(s, prepend=0); return s, np.c_[np.cumsum(np.cos(th) * ds), np.cumsum(np.sin(th) * ds)]


def to_plane(P):
    P = np.asarray(P, float); h = P[-1, :2] - P[0, :2]; nh = np.linalg.norm(h); h = h / nh if nh > 1e-9 else np.array([1.0, 0])
    Q = P - P[0]; return np.c_[Q[:, :2] @ h, Q[:, 2]]


def arch_traits(P):
    P = np.asarray(P, float); S = arclen(P); L = float(S[-1]); c = min(CHORD_MM, L / 2)
    th0, _ = chord_elev(P, 0, c); thtip, _ = chord_elev(P, L - c, L); dth = th0 - thtip
    Q = to_plane(P); s_obs = S
    res = {}
    for kind in ('arc', 'quad'):
        s, C = arch_curve(th0, dth, L, kind); M = np.c_[np.interp(s_obs, s, C[:, 0]), np.interp(s_obs, s, C[:, 1])]
        dev = np.linalg.norm(M - Q, axis=1)
        def f(p, kind=kind):
            s2, C2 = arch_curve(p[0], p[1], L, kind); M2 = np.c_[np.interp(s_obs, s2, C2[:, 0]), np.interp(s_obs, s2, C2[:, 1])]
            return float(np.sqrt(np.mean(np.sum((M2 - Q) ** 2, axis=1))))
        r = minimize(f, [th0, dth], method='Nelder-Mead', options=dict(xatol=0.01, fatol=1e-4, maxiter=400))
        res[kind] = dict(direct_max_dev_mm=float(dev.max()), direct_tip_err_mm=float(dev[-1]), fit_rms_mm=float(r.fun), fit_theta0_deg=float(r.x[0]), fit_dtheta_deg=float(r.x[1]))
    return dict(L_mm=L, theta0_deg=th0, theta_tip_deg=thtip, dtheta_deg=dth, curv_deg_per_mm=dth / L if L > 0 else float('nan'),
                s_apex_rel=float(S[int(np.argmax(P[:, 2]))] / L) if L > 0 else float('nan'), tip_drop_mm=float(P[-1, 2] - P[0, 2]),
                arc=res['arc'], quad=res['quad'], better='arc' if res['arc']['fit_rms_mm'] <= res['quad']['fit_rms_mm'] else 'quad')


def stem_dir_at(stem, p):
    stem = np.asarray(stem, float); S = arclen(stem); G = at_s(stem, np.arange(0, S[-1] + 1e-9, 0.5)); i = int(cKDTree(G).query(p)[1])
    a, b = G[max(i - 10, 0)], G[min(i + 10, len(G) - 1)]; d = b - a; return d / max(np.linalg.norm(d), 1e-12), i * 0.5


def leaf_base_traits(P, stem):
    c = min(CHORD_MM, arclen(P)[-1] / 2); _, d = chord_elev(P, 0, c); ax, s_on = stem_dir_at(stem, P[0])
    ins = float(np.degrees(np.arccos(np.clip(abs(d @ ax), 0, 1)))) if (d @ ax) >= 0 else float(180 - np.degrees(np.arccos(np.clip(abs(d @ ax), 0, 1))))
    hd = d - (d @ ax) * ax; ref = np.array([1.0, 0, 0]) - ax[0] * ax; ref /= np.linalg.norm(ref); ref2 = np.cross(ax, ref)
    psi = float(np.degrees(np.arctan2(hd @ ref2, hd @ ref)) % 360)
    return dict(insertion_deg=ins, base_azimuth_deg=psi, s_on_stem_mm=float(s_on))


def divergence(psis):
    d = np.diff(np.asarray(psis, float)) % 360
    if len(d) == 0:
        return dict(n=0)
    u = np.exp(1j * np.radians(d)); m = float(np.degrees(np.angle(u.mean())) % 360); hand = 'ccw'
    if m > 180:
        d = 360 - d; hand = 'cw'; u = np.exp(1j * np.radians(d)); m = float(np.degrees(np.angle(u.mean())) % 360)
    R = float(abs(u.mean())); csd = float(np.degrees(math.sqrt(-2 * math.log(max(R, 1e-12)))))
    return dict(n=int(len(d)), circ_mean_deg=m, circ_sd_deg=csd, handedness=hand, divergences_deg=d.round(2).tolist())


def internodes(s_on_stem):
    s = np.asarray(s_on_stem, float); return np.diff(s).tolist()


def neighbour_normal_noise(xyz, n, r=2.0, kmin=6, max_pts=20000, seed=0):
    """Median angle between a surfel normal and the PCA normal of its neighbours within r mm (>= kmin neighbours)."""
    xyz = np.asarray(xyz, float); n = plus_z(n); rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), min(max_pts, len(xyz)), replace=False); T = cKDTree(xyz); ang = []
    for i in idx:
        nb = T.query_ball_point(xyz[i], r)
        if len(nb) < kmin: continue
        Q = xyz[nb] - xyz[nb].mean(0); _, _, vt = np.linalg.svd(Q, full_matrices=False); pn = vt[-1]
        ang.append(np.degrees(np.arccos(min(1.0, abs(float(pn @ n[i]))))))
    return float(np.median(ang)) if ang else float('nan'), len(ang)
