"""Addendum fill surfels (CONTRACT §5): per compound leaf with A_missing > 0, quadric (or plane if R2 < 0.5) in the leaf's PCA frame; grid g over the box footprint; empty cell = no leaf
surfel within 1 mm (projected); fill closest-first within reach; one disc per cell on the surface with the fitted normal, leaf-median disc semi-axes, area g^2 (last trimmed).
Cases: R, B (g 2 mm, reach 6 mm) and sensitivity R_reach4, R_reach8, R_grid1, R_grid3. Writes kaggle_flat/fill__<plant>__<case>.npz and fill_surfels_per_leaf.csv. Local, small."""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lai_range as LR                                                          # reuses the loaders (recomputes the range tables; deterministic)
OUT = HERE / 'kaggle_flat'; OUT.mkdir(exist_ok=True)
CASES = {'R': ('R', 2.0, 6.0), 'B': ('B', 2.0, 6.0), 'R_reach4': ('R', 2.0, 4.0), 'R_reach8': ('R', 2.0, 8.0), 'R_grid1': ('R', 1.0, 6.0), 'R_grid3': ('R', 3.0, 6.0)}
LV = json.load(open(HERE / 'leaves.json')); rows = []


def quad_fit(P, w):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]; M = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]; sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(M * sw[:, None], z * sw, rcond=None); res = z - M @ coef; zm = (w * z).sum() / w.sum()
    r2 = 1 - (w * res ** 2).sum() / max((w * (z - zm) ** 2).sum(), 1e-30)
    return coef, float(r2)


for pid in LR.PLANTS:
    srows, X, A, ab, leaf, uids = LR.load(pid); inv = {u: k for k, u in uids.items()}
    for case, (arm, g, reach) in CASES.items():
        C, N, TU, TV, AA, BB, AR, LB = [], [], [], [], [], [], [], []; rng = np.random.default_rng(20261105)
        for l in [l for l in LV['leaves'] if l['plant'] == pid]:
            need = l[f'A_missing_{arm}']
            if need <= 0 or 'frame' not in l:
                continue
            m = leaf == l['label']; U = np.array(l['frame']); c = np.array(l['centre']); lo = np.array(l['lo']); hi = np.array(l['hi'])
            P = (X[m] - c) @ U.T; coef, r2 = quad_fit(P, A[m]); kind = 'quadric'
            if r2 < 0.5:
                coef = np.zeros(6); kind = 'plane'
            gx = np.arange(lo[0] + g / 2, hi[0], g); gy = np.arange(lo[1] + g / 2, hi[1], g); GX, GY = np.meshgrid(gx, gy); cells = np.c_[GX.ravel(), GY.ravel()]
            tr = cKDTree(P[:, :2]); dist = tr.query(cells)[0]; empty = dist > 1.0; cand = np.nonzero(empty & (dist <= reach))[0]; cand = cand[np.argsort(dist[cand], kind='stable')]
            k_need = need / (g * g); n_full = int(np.floor(k_need)); last = need - n_full * g * g; pick = cand[:n_full + (1 if last > 1e-9 else 0)]
            areas = np.full(len(pick), g * g)
            if len(pick) == n_full + 1 and last > 1e-9:
                areas[-1] = last
            filled = float(areas.sum()) if len(pick) else 0.0
            if len(pick):
                xx, yy = cells[pick, 0], cells[pick, 1]; a_, b_, c_, d_, e_, f_ = coef; zz = a_ * xx * xx + b_ * xx * yy + c_ * yy * yy + d_ * xx + e_ * yy + f_
                dzx = 2 * a_ * xx + b_ * yy + d_; dzy = b_ * xx + 2 * c_ * yy + e_; nl = np.c_[-dzx, -dzy, np.ones_like(xx)]; nl /= np.linalg.norm(nl, axis=1, keepdims=True)
                Pw = np.c_[xx, yy, zz] @ U + c; nw = nl @ U
                t = rng.normal(size=(len(pick), 3)); t -= (t * nw).sum(1, keepdims=True) * nw; t /= np.linalg.norm(t, axis=1, keepdims=True)
                C.append(Pw / 1000.0); N.append(nw); TU.append(t); TV.append(np.cross(nw, t)); AA.append(np.full(len(pick), l['disc_a_med'] / 1000.0)); BB.append(np.full(len(pick), l['disc_b_med'] / 1000.0))
                AR.append(areas); LB.append(np.full(len(pick), l['label']))
            rows.append(dict(plant=pid, case=case, leaf=l['leaf'], fit=kind, fit_r2=round(r2, 4), A_rec_mm2=round(l['A_rec'], 2), A_missing_mm2=round(need, 2), filled_mm2=round(filled, 2),
                             shortfall_mm2=round(need - filled, 2), empty_reachable_cells=int(len(cand)), cells_used=int(len(pick))))
        cat = lambda v, w: np.concatenate(v) if v else np.zeros((0, w)) if w else np.zeros(0)
        np.savez_compressed(OUT / f'fill__{pid}__{case}.npz', centre_board_m=cat(C, 3), normal=cat(N, 3), tu=cat(TU, 3), tv=cat(TV, 3), a_m=cat(AA, 0), b_m=cat(BB, 0), area_mm2=cat(AR, 0), leaf_label=cat(LB, 0).astype(int) if LB else np.zeros(0, int))
    print(pid, {cs: round(sum(r['filled_mm2'] for r in rows if r['plant'] == pid and r['case'] == cs), 1) for cs in CASES}, 'shortfall R', round(sum(r['shortfall_mm2'] for r in rows if r['plant'] == pid and r['case'] == 'R'), 1), flush=True)
with open(HERE / 'fill_surfels_per_leaf.csv', 'w', encoding='utf-8', newline='') as f:
    f.write('# fill_surfels_assumed; lai_range; quadric in the leaf PCA frame (plane if R2 < 0.5); grid / reach per case; empty = no leaf surfel within 1 mm projected; cell area per disc, last trimmed\n')
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
