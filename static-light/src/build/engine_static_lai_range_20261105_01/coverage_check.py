"""Geometry-only check before the option-b rerun: top-view union coverage / area for (1) the leaf's reconstructed surfels, (2) fill discs run 1 (median disc), (3) option b as declared
(sum-footprint matched), (4) cell-sized discs (footprint = credited area). Raster 0.1 mm, discs projected as circles of radius sqrt(a b) (top view). Plants R3-1b, n_03; case R."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lai_range as LR
RES = 0.1


def cover(xy, r):
    lo = xy.min(0) - r.max() - RES; hi = xy.max(0) + r.max() + RES; nx = int((hi[0] - lo[0]) / RES) + 1; ny = int((hi[1] - lo[1]) / RES) + 1; g = np.zeros((ny, nx), bool)
    for (x, y), rr in zip(xy, r):
        i0, i1 = int((x - rr - lo[0]) / RES), int((x + rr - lo[0]) / RES) + 1; j0, j1 = int((y - rr - lo[1]) / RES), int((y + rr - lo[1]) / RES) + 1
        X, Y = np.meshgrid(lo[0] + RES * np.arange(i0, i1), lo[1] + RES * np.arange(j0, j1)); g[j0:j1, i0:i1] |= (X - x) ** 2 + (Y - y) ** 2 <= rr * rr
    return g.sum() * RES * RES


out = {}
for pid in ('R3-1b', 'n_03'):
    srows, X, A, ab, leaf, uids = LR.load(pid); z1 = np.load(HERE / f'kaggle_flat/fill__{pid}__R.npz'); zb = np.load(HERE / f'kaggle_flat_b/fill__{pid}__R.npz')
    F = z1['centre_board_m'][:, :2] * 1000; Acr = z1['area_mm2'].sum(); rows = {}
    m = np.isin(leaf, np.unique(z1['leaf_label']))
    rows['reconstructed_surfels_of_filled_leaves'] = cover(X[m, :2], np.sqrt(ab[m, 0] * ab[m, 1])) / A[m].sum()
    rows['fill_run1_median_disc'] = cover(F, np.sqrt(z1['a_m'] * z1['b_m']) * 1000) / Acr
    rows['fill_option_b_declared'] = cover(F, np.sqrt(zb['a_m'] * zb['b_m']) * 1000) / Acr
    rows['fill_cell_sized_disc'] = cover(F, np.sqrt(z1['area_mm2'] / np.pi)) / Acr
    out[pid] = {k: round(float(v), 3) for k, v in rows.items()}; print(pid, out[pid], flush=True)
json.dump(out, open(HERE / 'coverage_check.json', 'w'), indent=1)
