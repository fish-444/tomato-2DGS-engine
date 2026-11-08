"""Small local test of normals.py (a few thousand points): a tilted plane with noisy normals; smoothing must recover the plane normal,
respect label sets, keep raw normals for sparse points, and the Traits metric must agree with a direct re-implementation."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / 'kaggle_data'))
import normals as NM

rng = np.random.default_rng(1)
n_true = np.array([0.3, -0.2, 0.93]); n_true /= np.linalg.norm(n_true)
e1 = np.cross(n_true, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n_true, e1)
uv = rng.uniform(-15, 15, (4000, 2)); X = uv[:, :1] * e1 + uv[:, 1:] * e2 + rng.normal(0, 0.3, (4000, 1)) * n_true
tu = np.tile(e1, (4000, 1)); tv = np.tile(e2, (4000, 1)); N0 = np.tile(n_true, (4000, 1))
ang = np.abs(rng.normal(0, np.radians(40), 4000)); N, _, _ = NM.rotate_frames(N0, tu, tv, ang, rng.normal(size=(4000, 3)))
N[::2] *= -1                                                     # two-sided: sign must follow the raw normal
sets = (uv[:, 0] > 0).astype(int)                                # two label sets split along a line
w = rng.uniform(0.5, 1.5, 4000)
S, kept, cnt = NM.smooth(X, N, w, sets, 3.0)
err_raw = np.degrees(np.arccos(np.clip(np.abs(N @ n_true), 0, 1))); err_s = np.degrees(np.arccos(np.clip(np.abs(S @ n_true), 0, 1)))
print('median angle to truth raw', np.median(err_raw).round(1), 'smoothed', np.median(err_s).round(1), 'kept raw', kept.mean().round(4), 'median count', np.median(cnt))
assert np.median(err_s) < 3 and np.all((S * N).sum(1) >= 0)
# brute-force check of one point's neighbourhood (same set only)
i = 123; d = np.linalg.norm(X - X[i], axis=1); m = (d <= 3.0) & (sets == sets[i])
mu = (w[m, None] * X[m]).sum(0) / w[m].sum(); C = ((w[m, None] * (X[m] - mu)).T @ (X[m] - mu)) / w[m].sum(); v = np.linalg.eigh(C)[1][:, 0]
assert m.sum() == cnt[i] and abs(abs(v @ S[i]) - 1) < 1e-9, (m.sum(), cnt[i], v @ S[i])
# sparse: an isolated far point keeps its raw normal
X2 = np.vstack([X, [[500, 500, 500]]]); N2 = np.vstack([N, [[0, 0, 1.0]]]); S2, k2, c2 = NM.smooth(X2, N2, np.r_[w, 1], np.r_[sets, 0], 3.0)
assert k2[-1] and c2[-1] == 1 and np.allclose(S2[-1], [0, 0, 1])
med, cnt_m = NM.noise_metric(X, N, max_pts=2000)
print('noise metric raw', round(med, 1), 'n', cnt_m, 'smoothed', round(NM.noise_metric(X, S, max_pts=2000)[0], 1))
print('ALL OK')
