"""Leaflet split of one compound leaf from its labelled surfels (CONTRACT 'Split rule'). Values below are the declared ones."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from sklearn.mixture import GaussianMixture

UP = np.array([0.0, 0.0, 1.0])
STEP = 0.25; EXT = 60.0; EXT_DIR = 2.0
R_RACHIS = 1.0
BIN = 1.0; FWHM = 2.0; GAP_FRAC = 0.02; T_FALLBACK = 0.8
MIN_SIDE = 30; DBIC = 10.0; M_SET = (1, 2, 3)
BASE_FRAC = 0.05; NORMAL_R = 2.0; CELL = 1.0; ALPHA = 2.0
MIN_SURFELS = 100


def resample(curve, step=STEP):
    P = np.asarray(curve, float); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); keep = np.r_[True, seg > 1e-9]; P = P[keep]
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; L = float(cum[-1])
    s = np.arange(0, L + 1e-9, step); s = np.r_[s, L] if s[-1] < L - 1e-9 else s
    Q = np.stack([np.interp(s, cum, P[:, i]) for i in range(3)], 1)
    return Q, s, L


def extended(curve, ext=EXT):
    Q, s, L = resample(curve)
    j = int(np.searchsorted(s, max(L - EXT_DIR, 0.0))); d = Q[-1] - Q[min(j, len(Q) - 2)]; d = d / (np.linalg.norm(d) or 1)
    se = np.arange(STEP, max(ext, STEP) + 1e-9, STEP); Qe = Q[-1] + d[None] * se[:, None]
    return np.vstack([Q, Qe]), np.r_[s, L + se], L


def point_at(Q, s, si):
    return np.stack([np.interp(si, s, Q[:, i]) for i in range(3)], -1)


def tangents(Q):
    T = np.gradient(Q, axis=0); return T / np.maximum(np.linalg.norm(T, axis=1)[:, None], 1e-12)


def frame(X, curve, ext=EXT):
    """s, r, side, tangent at foot for every surfel."""
    Q, s, L = extended(curve, ext); T = tangents(Q); tr = cKDTree(Q); _, j = tr.query(X)
    # refine on the segment j-1..j+1
    best_s = s[j].copy(); best_p = Q[j].copy()
    for dj in (-1, 0):
        a = np.clip(j + dj, 0, len(Q) - 2); A = Q[a]; B = Q[a + 1]; AB = B - A; t = np.clip(((X - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
        P = A + t[:, None] * AB; better = np.linalg.norm(X - P, axis=1) < np.linalg.norm(X - best_p, axis=1)
        best_p[better] = P[better]; best_s[better] = (s[a] + t * (s[a + 1] - s[a]))[better]
    off = X - best_p; r = np.linalg.norm(off, axis=1)
    B = np.cross(T, UP); nb = np.linalg.norm(B, axis=1)
    for i in range(len(B)):           # degenerate (vertical tangent) -> previous valid sample
        if nb[i] < 0.05:
            B[i] = B[i - 1] if i > 0 else np.array([1.0, 0, 0]); nb[i] = np.linalg.norm(B[i])
    B = B / nb[:, None]
    side = np.sign((off * B[j]).sum(1)); side[side == 0] = 1
    return dict(s=best_s, r=r, side=side.astype(int), T=T[j], Q=Q, sQ=s, L=L)


def smooth_hist(sv, lo, hi):
    edges = np.arange(math.floor(lo), math.ceil(hi) + BIN + 1e-9, BIN); h, _ = np.histogram(sv, edges)
    sig = FWHM / 2.3548 / BIN; k = np.arange(-int(4 * sig) - 1, int(4 * sig) + 2); g = np.exp(-0.5 * (k / sig) ** 2); g /= g.sum()
    return np.convolve(h, g, mode="same"), edges


def terminal_s(sv, L):
    if len(sv) == 0:
        return T_FALLBACK * L, False
    hs, edges = smooth_hist(sv, min(0.0, sv.min()), sv.max()); low = hs <= GAP_FRAC * hs.max()
    gaps = []; i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            gaps.append((edges[i], edges[j + 1])); i = j + 1
        else:
            i += 1
    smax = float(sv.max()); cand = [g for g in gaps if 0.5 * L < g[0] <= smax and g[1] <= smax + BIN]
    cand = [g for g in cand if np.any(sv >= g[1])]      # something must lie beyond the gap
    if cand:
        return float(cand[-1][0]), True
    return T_FALLBACK * L, False


def gmm_side(sv, prior):
    if len(sv) < MIN_SIDE:
        return dict(m_data=0, m_chosen=0, bic={}, labels=np.zeros(len(sv), int) - 1, means=[])
    x = sv.reshape(-1, 1); fits = {}
    for m in M_SET:
        if len(sv) < m * 5:
            continue
        gm = GaussianMixture(m, covariance_type="full", random_state=0, n_init=5).fit(x); fits[m] = (gm.bic(x), gm)
    bmin = min(b for b, _ in fits.values()); m_data = min(fits, key=lambda m: fits[m][0])
    sup = [m for m in fits if fits[m][0] <= bmin + DBIC]
    m_ch = min(sup, key=lambda m: (abs(m - prior), fits[m][0]))
    gm = fits[m_ch][1]; lab = gm.predict(x); order = np.argsort(gm.means_.ravel()); remap = {int(o): i for i, o in enumerate(order)}
    lab = np.array([remap[int(v)] for v in lab])
    return dict(m_data=int(m_data), m_chosen=int(m_ch), bic={int(m): float(b) for m, (b, _) in fits.items()}, labels=lab, means=[float(sv[lab == i].mean()) if np.any(lab == i) else float("nan") for i in range(m_ch)])


def prior_per_side(n):
    n = int(n); return (n - 1) / 2.0


def align_pairs(sL, sR):
    """order-preserving matching of left / right leaflet mean s; returns list of (iL or None, iR or None)."""
    if len(sL) == len(sR):
        return list(zip(range(len(sL)), range(len(sR))))
    long_, short, flip = (sL, sR, False) if len(sL) > len(sR) else (sR, sL, True)
    best = None
    for keep in itertools.combinations(range(len(long_)), len(short)):
        c = sum(abs(long_[a] - short[b]) for a, b in zip(keep, range(len(short))))
        if best is None or c < best[0]:
            best = (c, keep)
    keep = best[1]; out = []; it = iter(range(len(short)))
    for a in range(len(long_)):
        b = next(it) if a in keep else None
        out.append((b, a) if flip else (a, b))
    return out


def leaflet_geometry(X, A, N, curve_info):
    """per-leaflet descriptors and alpha mesh (CONTRACT step 6)."""
    c = (X * A[:, None]).sum(0) / A.sum(); r = curve_info["r"]; s = curve_info["s"]
    nb = max(1, int(math.ceil(BASE_FRAC * len(X)))); near = np.argsort(r)[:nb]; base = X[near].mean(0); base_s = float(np.median(s[near]))
    Y = X - X.mean(0); w, v = np.linalg.eigh(np.cov(Y.T)); ax1 = v[:, 2]; ax2 = v[:, 1]; nrm = v[:, 0]
    if np.dot(ax1, c - base) < 0:
        ax1 = -ax1
    if nrm[2] < 0:
        nrm = -nrm
    ax2 = np.cross(nrm, ax1)
    p1 = Y @ ax1; p2 = Y @ ax2; length = float(np.percentile(p1, 99) - np.percentile(p1, 1)); width = float(np.percentile(p2, 99) - np.percentile(p2, 1))
    # normals
    if N is not None:
        Nn = N / np.maximum(np.linalg.norm(N, axis=1)[:, None], 1e-12); Nn = Nn * np.sign(Nn @ nrm + 1e-12)[:, None]
        tr = cKDTree(X); nb_l = tr.query_ball_point(X, NORMAL_R); Ns = np.array([Nn[i].mean(0) for i in nb_l]); Ns /= np.maximum(np.linalg.norm(Ns, axis=1)[:, None], 1e-12)
        n_mean = (Ns * A[:, None]).sum(0); n_mean /= np.linalg.norm(n_mean) or 1
    else:
        n_mean = nrm
    elev = float(np.degrees(np.arcsin(np.clip(ax1[2], -1, 1)))); az = float(np.degrees(np.arctan2(ax1[1], ax1[0])) % 360)
    # alpha mesh on 1 mm cells
    u = Y @ ax1; vv = Y @ ax2; key = np.floor(np.c_[u, vv] / CELL).astype(np.int64); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    k = inv.max() + 1; V = np.zeros((k, 3)); cnt = np.bincount(inv, minlength=k).astype(float)
    for i in range(3):
        V[:, i] = np.bincount(inv, weights=X[:, i], minlength=k) / cnt
    F = np.zeros((0, 3), int)
    if k >= 3:
        uv = np.c_[(V - X.mean(0)) @ ax1, (V - X.mean(0)) @ ax2]
        try:
            tri = Delaunay(uv).simplices
            a = uv[tri[:, 0]]; b = uv[tri[:, 1]]; cc = uv[tri[:, 2]]
            la = np.linalg.norm(b - cc, axis=1); lb = np.linalg.norm(a - cc, axis=1); lc = np.linalg.norm(a - b, axis=1)
            ar = 0.5 * np.abs((b - a)[:, 0] * (cc - a)[:, 1] - (b - a)[:, 1] * (cc - a)[:, 0])
            R = la * lb * lc / np.maximum(4 * ar, 1e-12); F = tri[R <= ALPHA]
        except Exception:
            F = np.zeros((0, 3), int)
    mesh_area = float(sum(0.5 * np.linalg.norm(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])) for f in F)) if len(F) else 0.0
    return dict(area_own=float(A.sum()), centroid=c.tolist(), s_mean=float((s * A).sum() / A.sum()), base=base.tolist(), base_s=base_s, base_r=float(np.median(r[near])),
                d=ax1.tolist(), b=ax2.tolist(), pca_n=nrm.tolist(), normal=n_mean.tolist(), length_mm=length, width_mm=width, elev_deg=elev, az_deg=az,
                mesh=dict(vertices=V.tolist(), faces=F.tolist()), mesh_area=mesh_area, n=int(len(X)))


def split_leaf_a3(X, A, N, curve, n_prior, ext=EXT):   # amendment 3 (withheld)
    """X (n,3) mm, A (n,) mm2, N (n,3) or None, curve (k,3) mm, n_prior leaflets. Returns leaflets + summary."""
    fr = frame(X, curve, ext); s = fr["s"]; r = fr["r"]; side = fr["side"]; L = fr["L"]
    rach = (r <= R_RACHIS) & (s <= L); non = ~rach          # amendment 3: the extension is not rachis
    sT = L; found = bool(np.any(non & (s > L)))           # amendment 3: terminal = s > L (clicks end at the end-leaflet base)
    lab = np.full(len(X), -1, int)      # leaflet index
    leaflets = []
    term = non & (s > L); lat = non & (s <= L)
    pri = prior_per_side(n_prior); sides = {}
    for sd, nm in ((1, "L"), (-1, "R")):
        idx = np.nonzero(lat & (side == sd))[0]; g = gmm_side(s[idx], pri); g["idx"] = idx; sides[nm] = g
    pairs = align_pairs(sides["L"]["means"], sides["R"]["means"])
    for k, (iL, iR) in enumerate(pairs, 1):
        for nm, ii in (("L", iL), ("R", iR)):
            if ii is None:
                continue
            g = sides[nm]; sel = g["idx"][g["labels"] == ii]
            if len(sel) == 0:
                continue
            leaflets.append(dict(type="P", pair=k, side=nm, idx=sel))
    ti = np.nonzero(term)[0]
    if len(ti):
        leaflets.append(dict(type="T", pair=0, side="C", idx=ti))
    # amendment 2: non-rachis surfels without a leaflet (m = 0 side) -> leaflet with the nearest mean s; none at all -> terminal
    used = np.zeros(len(X), bool)
    for lt in leaflets:
        used[lt["idx"]] = True
    orphan = np.nonzero(non & ~used)[0]
    if len(orphan):
        if not leaflets:
            leaflets.append(dict(type="T", pair=0, side="C", idx=orphan))
        else:
            ms = np.array([s[lt["idx"]].mean() for lt in leaflets]); near = np.argmin(np.abs(s[orphan][:, None] - ms[None]), axis=1)
            for q in np.unique(near):
                leaflets[q]["idx"] = np.sort(np.r_[leaflets[q]["idx"], orphan[near == q]])
    n_orphan = int(len(orphan))
    for q, lt in enumerate(leaflets):
        lab[lt["idx"]] = q
        sub = dict(r=r[lt["idx"]], s=s[lt["idx"]])
        lt.update(leaflet_geometry(X[lt["idx"]], A[lt["idx"]], None if N is None else N[lt["idx"]], sub))
    missing = [f"P{k}{'R' if iR is None else 'L'}" for k, (iL, iR) in enumerate(pairs, 1) if iL is None or iR is None]
    summ = dict(L=L, s_T=sT, terminal_gap_found=bool(found), terminal_present=bool(len(ti)), rachis_area=float(A[rach].sum()), rachis_share=float(A[rach].sum() / A.sum()),
                total_area=float(A.sum()), leaflet_area=float(sum(l["area_own"] for l in leaflets)),
                m_data_L=sides["L"]["m_data"], m_chosen_L=sides["L"]["m_chosen"], m_data_R=sides["R"]["m_data"], m_chosen_R=sides["R"]["m_chosen"],
                bic_L=sides["L"]["bic"], bic_R=sides["R"]["bic"], n_prior=int(n_prior), prior_per_side=pri, n_leaflets=len(leaflets), missing=missing, n_surfels=int(len(X)), n_orphan_reassigned=n_orphan)
    summ["conservation_err"] = summ["leaflet_area"] + summ["rachis_area"] - summ["total_area"]
    return dict(leaflets=leaflets, summary=summ, labels=lab, frame=fr)


# ------------------------------------------------------------------ amendment 4: clicks reach the end-leaflet tip
def gmm_side_tip(sv, prior):
    """m lateral components + 1 end-leaflet component (largest mean s). labels: 0..m-1 laterals, m = end-leaflet half."""
    if len(sv) < MIN_SIDE:
        return dict(m_data=0, m_chosen=0, bic={}, labels=np.zeros(len(sv), int), means=[])
    x = sv.reshape(-1, 1); fits = {}
    for m in M_SET:
        if len(sv) < (m + 1) * 5:
            continue
        gm = GaussianMixture(m + 1, covariance_type="full", random_state=0, n_init=5).fit(x); fits[m] = (gm.bic(x), gm)
    if not fits:
        return dict(m_data=0, m_chosen=0, bic={}, labels=np.zeros(len(sv), int), means=[])
    bmin = min(b for b, _ in fits.values()); m_data = min(fits, key=lambda m: fits[m][0])
    sup = [m for m in fits if fits[m][0] <= bmin + DBIC]
    m_ch = min(sup, key=lambda m: (abs(m - prior), fits[m][0]))
    gm = fits[m_ch][1]; lab = gm.predict(x); order = np.argsort(gm.means_.ravel()); remap = {int(o): i for i, o in enumerate(order)}
    lab = np.array([remap[int(v)] for v in lab])
    return dict(m_data=int(m_data), m_chosen=int(m_ch), bic={int(m): float(b) for m, (b, _) in fits.items()}, labels=lab,
                means=[float(sv[lab == i].mean()) if np.any(lab == i) else float("nan") for i in range(m_ch)])


def split_leaf(X, A, N, curve, n_prior, ext=EXT):
    """amendment 4. X (n,3) mm, A (n,) mm2, N (n,3) or None, curve (k,3) mm (node -> near the end-leaflet tip), n_prior leaflets."""
    fr = frame(X, curve, ext); s = fr["s"]; r = fr["r"]; side = fr["side"]; L = fr["L"]
    band = r <= R_RACHIS; non = ~band
    pri = prior_per_side(n_prior); sides = {}; term_parts = []
    # amendment 5: m per side fixed to the prior; n = 6 -> 3 on the side where BIC(3+1) - BIC(2+1) is lower
    idx_s = {nm: np.nonzero(non & (side == sd))[0] for sd, nm in ((1, "L"), (-1, "R"))}
    fixed = {nm: int(math.floor(pri)) for nm in idx_s}
    if pri != math.floor(pri):
        dl = {nm: (lambda bb: bb.get(3, np.inf) - bb.get(2, np.inf) if (3 in bb and 2 in bb) else np.inf)(bic_all(s[idx_s[nm]])) for nm in idx_s}
        fixed["L" if dl["L"] <= dl["R"] else "R"] += 1
    for nm in ("L", "R"):
        idx = idx_s[nm]; g = gmm_side_fixed(s[idx], fixed[nm]); g["idx"] = idx; sides[nm] = g
        term_parts.append(idx[g["labels"] == g["m_chosen"]])
    ti = np.concatenate(term_parts) if term_parts else np.zeros(0, int)
    sT = float(min(s[p].min() for p in term_parts if len(p))) if len(ti) else L
    band_t = np.nonzero(band & (s >= sT))[0]; ti = np.sort(np.r_[ti, band_t]); rach = band & (s < sT)
    lab = np.full(len(X), -1, int); leaflets = []
    pairs = align_pairs(sides["L"]["means"], sides["R"]["means"])
    for k, (iL, iR) in enumerate(pairs, 1):
        for nm, ii in (("L", iL), ("R", iR)):
            if ii is None:
                continue
            g = sides[nm]; sel = g["idx"][g["labels"] == ii]
            if len(sel):
                leaflets.append(dict(type="P", pair=k, side=nm, idx=sel))
    if len(ti):
        leaflets.append(dict(type="T", pair=0, side="C", idx=ti))
    used = np.zeros(len(X), bool)
    for lt in leaflets:
        used[lt["idx"]] = True
    orphan = np.nonzero(~rach & ~used)[0]
    if len(orphan):
        if not leaflets:
            leaflets.append(dict(type="T", pair=0, side="C", idx=orphan))
        else:
            ms = np.array([s[lt["idx"]].mean() for lt in leaflets]); near = np.argmin(np.abs(s[orphan][:, None] - ms[None]), axis=1)
            for q in np.unique(near):
                leaflets[q]["idx"] = np.sort(np.r_[leaflets[q]["idx"], orphan[near == q]])
    for q, lt in enumerate(leaflets):
        lab[lt["idx"]] = q
        lt.update(leaflet_geometry(X[lt["idx"]], A[lt["idx"]], None if N is None else N[lt["idx"]], dict(r=r[lt["idx"]], s=s[lt["idx"]])))
    missing = [f"P{k}{'R' if iR is None else 'L'}" for k, (iL, iR) in enumerate(pairs, 1) if iL is None or iR is None]
    summ = dict(L=L, s_T=sT, terminal_gap_found=False, terminal_present=bool(len(ti)), rachis_area=float(A[rach].sum()), rachis_share=float(A[rach].sum() / A.sum()),
                total_area=float(A.sum()), leaflet_area=float(sum(l["area_own"] for l in leaflets)),
                m_data_L=sides["L"]["m_data"], m_chosen_L=sides["L"]["m_chosen"], m_data_R=sides["R"]["m_data"], m_chosen_R=sides["R"]["m_chosen"],
                bic_L=sides["L"]["bic"], bic_R=sides["R"]["bic"], n_prior=int(n_prior), prior_per_side=pri, n_leaflets=len(leaflets), missing=missing, n_surfels=int(len(X)), n_orphan_reassigned=int(len(orphan)))
    summ["conservation_err"] = summ["leaflet_area"] + summ["rachis_area"] - summ["total_area"]
    return dict(leaflets=leaflets, summary=summ, labels=lab, frame=fr)


# ------------------------------------------------------------------ amendment 5 helpers
def bic_all(sv):
    x = sv.reshape(-1, 1); out = {}
    for m in M_SET:
        if len(sv) >= max(MIN_SIDE, (m + 1) * 5):
            out[m] = float(GaussianMixture(m + 1, covariance_type="full", random_state=0, n_init=5).fit(x).bic(x))
    return out


def gmm_side_fixed(sv, m):
    """m laterals fixed (amendment 5) + 1 end-leaflet component; data-only m reported from the BICs."""
    bb = bic_all(sv); m_data = min(bb, key=bb.get) if bb else 0
    if m <= 0 or len(sv) < max(MIN_SIDE, (m + 1) * 5):
        return dict(m_data=int(m_data), m_chosen=0, bic=bb, labels=np.zeros(len(sv), int), means=[])
    x = sv.reshape(-1, 1); gm = GaussianMixture(m + 1, covariance_type="full", random_state=0, n_init=5).fit(x)
    lab = gm.predict(x); order = np.argsort(gm.means_.ravel()); remap = {int(o): i for i, o in enumerate(order)}; lab = np.array([remap[int(v)] for v in lab])
    return dict(m_data=int(m_data), m_chosen=int(m), bic=bb, labels=lab, means=[float(sv[lab == i].mean()) if np.any(lab == i) else float("nan") for i in range(m)])
