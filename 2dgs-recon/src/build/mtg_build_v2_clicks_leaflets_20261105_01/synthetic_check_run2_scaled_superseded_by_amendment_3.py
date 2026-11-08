"""Synthetic gate (CONTRACT 'Checks'): Engine v2 compound leaves -> surfels -> split -> count / area error. python -X utf8 synthetic_check.py"""
import csv, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "build/mtg_build_20261102_01"))
import split_lib as SL  # noqa: E402
from template_rows import Template  # noqa: E402

DENS = 9.73; SIG = 0.124; KEEP = 0.927; MIN_LEAF = 20.0; REAL_AREA = 443.0  # amendment 2: scale each leaf to the real median compound area
GEO = ROOT / "build/engine_v2_calib_20261104_01/kout"; ARMS = ["ev-bc5a-k1-20261104/out/C5f1.15_m00", "ev-bc5a-k0-20261104/out/C5f0.85_m00", "ev-bc4v-k0-20261104/out/C4V_m00"]  # amendment 1
TPL = Template(); rng = np.random.default_rng(0)


def ends(V):
    c = V.mean(0); w, v = np.linalg.eigh(np.cov((V - c).T)) if len(V) > 2 else (None, None); a = v[:, -1] if v is not None else (V[-1] - V[0]); a = a / (np.linalg.norm(a) or 1)
    p = (V - c) @ a; return V[np.argmin(p)], V[np.argmax(p)]


def chain(P, Rs, ref):
    """petiole + rachis segments -> polyline; ref = a point far along the leaf (to orient the petiole)."""
    a, b = ends(P); first_next = Rs[0] if Rs else None
    tgt = ends(first_next)[0] if first_next is not None else ref
    if np.linalg.norm(a - tgt) < np.linalg.norm(b - tgt):
        a, b = b, a
    pl = [a, b]
    for R in Rs:
        x, y = ends(R)
        if np.linalg.norm(y - pl[-1]) < np.linalg.norm(x - pl[-1]):
            x, y = y, x
        pl.append(y)
    return np.array(pl)


def sample(tris, owners):
    X, A, O, Nn = [], [], [], []
    for t, o in zip(tris, owners):
        e1 = t[1] - t[0]; e2 = t[2] - t[0]; cr = np.cross(e1, e2); ar = 0.5 * np.linalg.norm(cr)
        if ar <= 0:
            continue
        n = int(np.floor(ar * DENS + rng.random()))
        if n == 0:
            continue
        u = rng.random(n); v = rng.random(n); f = u + v > 1; u[f] = 1 - u[f]; v[f] = 1 - v[f]
        nrm = cr / np.linalg.norm(cr); p = t[0] + u[:, None] * e1 + v[:, None] * e2 + nrm[None] * rng.normal(0, SIG, n)[:, None]
        X.append(p); A.append(np.full(n, ar / n)); O += [o] * n; Nn.append(np.repeat(nrm[None], n, 0))
    X = np.vstack(X); A = np.concatenate(A); O = np.array(O); Nn = np.vstack(Nn)
    keep = rng.random(len(X)) < KEEP
    return X[keep], A[keep], O[keep], Nn[keep]


rows = []
for arm in ARMS:
    z = np.load(GEO / arm / "geometry_day71.npz", allow_pickle=True); tri = np.asarray(z["tri"], float) * 1000.0; own = np.asarray(z["owner"]).astype(str); cls = np.asarray(z["cls"]).astype(str)
    by = defaultdict(list)
    for i, o in enumerate(own):
        m = re.match(r"^(.*?/C\d+)/(.+)$", o)
        if m and cls[i] in ("Leaflet", "Petiole", "RachisSegment"):
            by[m.group(1)].append(i)
    for cid in sorted(by):
        ii = np.array(by[cid]); lf = ii[cls[ii] == "Leaflet"]
        la = sum(0.5 * np.linalg.norm(np.cross(tri[i, 1] - tri[i, 0], tri[i, 2] - tri[i, 0])) for i in lf)
        if la < MIN_LEAF:
            continue
        P = [tri[i] for i in ii if own[i] == f"{cid}/P"]; Rk = sorted({own[i] for i in ii if re.match(rf"^{re.escape(cid)}/R\d+$", own[i])}, key=lambda s: int(s.rsplit("R", 1)[1]))
        if not P:
            continue
        Rs = [np.vstack([tri[i] for i in ii if own[i] == r]) for r in Rk]
        ref = tri[lf].reshape(-1, 3).mean(0); curve = chain(np.vstack(P), Rs, ref)
        k_sc = float(np.sqrt(REAL_AREA / la)); o0 = curve[0].copy(); curve = o0 + (curve - o0) * k_sc; T_s = o0 + (tri[ii] - o0) * k_sc   # amendment 2
        X, A, O, Nn = sample(T_s, own[ii])
        true = sorted({o for o in O if cls[ii][own[ii] == o][0] == "Leaflet"}); n_true_all = len({own[i] for i in lf})
        rank = int(cid.rsplit("C", 1)[1]); prior = TPL.leaflets_mode(rank)
        res = SL.split_leaf(X, A, Nn, curve, prior); S = res["summary"]; lab = res["labels"]
        errs = []
        for o in true:
            mt = O == o; ta = A[mt].sum()
            if not np.any(lab[mt] >= 0):
                errs.append(-1.0); continue
            ov = np.bincount(lab[mt][lab[mt] >= 0], weights=A[mt][lab[mt] >= 0]); q = int(np.argmax(ov)); errs.append((res["leaflets"][q]["area_own"] - ta) / ta)
        rows.append(dict(arm=arm, leaf=cid, scale=round(k_sc, 3), leaf_area_unscaled=round(la, 2), rank=rank, prior=prior, n_true=n_true_all, n_true_sampled=len(true), n_pred=S["n_leaflets"], count_ok=S["n_leaflets"] == n_true_all,
                         mL=S["m_chosen_L"], mR=S["m_chosen_R"], mL_data=S["m_data_L"], mR_data=S["m_data_R"], terminal_gap=S["terminal_gap_found"], rachis_share=round(S["rachis_share"], 4),
                         leaf_area=round(S["total_area"], 2), median_abs_area_err=float(np.median(np.abs(errs))), errs=";".join(f"{e:.3f}" for e in errs), cons=S["conservation_err"], orphans=S["n_orphan_reassigned"]))
        print(rows[-1]["arm"], cid, prior, n_true_all, S["n_leaflets"], round(rows[-1]["median_abs_area_err"], 3), flush=True)
all_err = np.concatenate([np.abs(np.array([float(x) for x in r["errs"].split(";")])) for r in rows])
cnt = float(np.mean([r["count_ok"] for r in rows])); med = float(np.median(all_err))
R = dict(leaves=len(rows), count_exact_share=cnt, median_abs_area_err=med, passed=bool(cnt >= 0.80 and med <= 0.15), gate="count exact >= 0.80 and median |area err| <= 0.15",
         by_true_n={str(n): dict(leaves=sum(r["n_true"] == n for r in rows), count_ok=sum(r["count_ok"] for r in rows if r["n_true"] == n)) for n in sorted({r["n_true"] for r in rows})},
         terminal_gap_share=float(np.mean([r["terminal_gap"] for r in rows])), max_conservation_err=float(max(abs(r["cons"]) for r in rows)), arms=ARMS)
with open(HERE / "synthetic_check.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(R, open(HERE / "synthetic_check.json", "w", encoding="utf-8"), indent=1); print(json.dumps(R, indent=1))
