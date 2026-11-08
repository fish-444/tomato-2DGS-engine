"""Small local run (R3-1b only, <= 3 min): blade-only (R8_pc leaves) absorbed light at d_ex 1.5 / 2 / 2.4 / 3 / 5 mm for v1 and v1_tau, so the D-vs-T d_ex band can be given
for both models (K5 stores plant totals only at those d_ex). Same code path and settings as K5 (tau bundle, seeds 101/202, a = 1 totals); local float differences ~1e-16."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
sys.path.insert(0, str(B / 'engine_static_tau_20261031_01/kaggle_data'))
from static import tau_runs as TR, obs
from static.tau_obs import build_scene, trace_plants_tau
F = TR.setup(B / 'engine_static_nine_20261028_01/kaggle_data', B / 'engine_static_tau_20261031_01/kaggle_data'); CFG = TR.CFG
q = TR.placed('R3-1b'); built = build_scene([q]); blade = (q.axis == 0) & (q.r8 >= 1); out = {}
for dex in (1.5, 2.0, 2.4, 3.0, 5.0):
    for model in ('v1', 'tau'):
        tot, totb = [], []
        for sd in TR.SEEDS:
            r = trace_plants_tau([q], [q], sd, CFG['eps'], 0.0, 0.0, dex / 1000, F, tau_scale=1.0, use_v1=(model == 'v1'), built=built)['R3-1b']
            ab = r['Uext'] * obs.kernel(r['parts']) * q.A; tot.append(float(ab.sum())); totb.append(float(ab[blade].sum()))
        out[f'{dex}|{model}'] = dict(total=float(np.mean(tot)), blades=float(np.mean(totb)))
        print(dex, model, out[f'{dex}|{model}'], flush=True)
json.dump(out, open(HERE / 'R3-1b_dex_blades_local.json', 'w'), indent=1)
