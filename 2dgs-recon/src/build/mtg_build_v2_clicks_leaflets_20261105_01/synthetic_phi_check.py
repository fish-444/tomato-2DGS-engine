"""Report (not a gate): spread angle phi from the split vs the true leaflet's phi on the synthetic leaves (amendment 5 setup).
phi = angle between leaflet long axis (PCA, base -> centroid) and the rachis tangent at the base, in the compound-leaf plane, same code for both."""
import csv, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = open(HERE / "synthetic_check.py", encoding="utf-8").read().split("rows = []")[0]
exec(src)


def phi_of(Xl, Al, fr_s, fr_r, Q, sQ, nc):
    g = SL.leaflet_geometry(Xl, Al, None, dict(r=fr_r, s=fr_s)); T = SL.tangents(Q); j = int(np.clip(np.searchsorted(sQ, g["base_s"]), 0, len(Q) - 1)); t = T[j]; d = np.asarray(g["d"])
    tp = t - nc * (t @ nc); dp = d - nc * (d @ nc); tp /= np.linalg.norm(tp) or 1; dp /= np.linalg.norm(dp) or 1
    return float(np.degrees(np.arccos(np.clip(tp @ dp, -1, 1)))), g


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
        curve = chain(np.vstack(P), Rs, tri[lf].reshape(-1, 3).mean(0))
        Tv = np.vstack([tri[i] for i in ii if own[i] == f"{cid}/T"]) if any(own[i] == f"{cid}/T" for i in ii) else None
        if Tv is not None:
            curve = np.vstack([curve, Tv[np.argmax(np.linalg.norm(Tv - curve[-1], axis=1))]])
        k_sc = float(np.sqrt(REAL_AREA / la)); o0 = curve[0].copy(); curve = o0 + (curve - o0) * k_sc; T_s = o0 + (tri[ii] - o0) * k_sc
        X, A, O, Nn = sample(T_s, own[ii]); prior = TPL.leaflets_mode(int(cid.rsplit("C", 1)[1]))
        res = SL.split_leaf(X, A, Nn, curve, prior); lab = res["labels"]; fr = res["frame"]
        Y = X - X.mean(0); w, v = np.linalg.eigh(np.cov(Y.T)); nc = v[:, 0] * (1 if v[2, 0] >= 0 else -1)
        for o in sorted({o for o in O if o.rsplit("/", 1)[1] not in ("P",) and not re.match(r"^R\d+$", o.rsplit("/", 1)[1])}):
            mt = O == o
            if mt.sum() < 30 or not np.any(lab[mt] >= 0):
                continue
            q = int(np.argmax(np.bincount(lab[mt][lab[mt] >= 0], weights=A[mt][lab[mt] >= 0]))); mp = lab == q
            pt, _ = phi_of(X[mt], A[mt], fr["s"][mt], fr["r"][mt], fr["Q"], fr["sQ"], nc); pp, _ = phi_of(X[mp], A[mp], fr["s"][mp], fr["r"][mp], fr["Q"], fr["sQ"], nc)
            rows.append(dict(arm=arm, leaf=cid, leaflet=o.rsplit("/", 1)[1], phi_true=pt, phi_split=pp, diff=pp - pt))
d = np.array([r["diff"] for r in rows]); pt = np.array([r["phi_true"] for r in rows]); pp = np.array([r["phi_split"] for r in rows])
R = dict(leaflets=len(rows), phi_true_median=float(np.median(pt)), phi_split_median=float(np.median(pp)), diff_median=float(np.median(d)), abs_diff_median=float(np.median(np.abs(d))), abs_diff_p90=float(np.percentile(np.abs(d), 90)),
         note="report only (not a gate); engine leaflets; same phi code for truth and split")
with open(HERE / "synthetic_phi_check.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(R, open(HERE / "synthetic_phi_check.json", "w", encoding="utf-8"), indent=1); print(json.dumps(R, indent=1))
