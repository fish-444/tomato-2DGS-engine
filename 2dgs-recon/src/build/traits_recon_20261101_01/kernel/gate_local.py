"""Gate step of run_traits.py re-run locally on the kernel CSVs; only fix: LEN key proj1 -> proj (proj_true column; extracted = proj1_ext renamed below)."""
import json, numpy as np, pandas as pd
from pathlib import Path
OUT=Path('kaggle_out/out'); LABELS=['recon_trait','conditional_on_unvalidated_scale','after_checkpoint_v1','validation_status_unconfirmed','age_unknown']
rows_plant=pd.read_csv(OUT/'synthetic_recovery_plant.csv').to_dict('records'); rows_leaf=pd.read_csv(OUT/'synthetic_recovery_leaf.csv').to_dict('records')
sigma_n=json.load(open(OUT/'real_A_traits.json'))['R3-1b']['neighbour_normal_noise_deg']
def say(*a): print(*a)
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
T0=0
import time
# ------------------------------------------------------------------ gate (CONTRACT section 3)
dp, dl = pd.DataFrame(rows_plant), pd.DataFrame(rows_leaf)
ANG = dict(A1_mean=dp, A1_sd=dp, A1_third1=dp, A1_third2=dp, A1_third3=dp, theta0_deg=dl, theta_tip_deg=dl, dtheta_deg=dl, insertion_deg=dl, phi=dl, psi=dl, C3_incl=dl)
dp = dp.rename(columns={'proj1_ext': 'proj_ext'})
LEN = dict(height=dp, proj=dp, width_x=dp, width_y=dp, L_mm=dl, tip_drop_mm=dl, internode_mm=dl, curv_deg_per_mm=dl)
gate = {}
for noise in ('primary', 'stress15'):
    g = {}
    for k, df in ANG.items():
        x = df[df.noise == noise][[k + '_ext', k + '_true']].dropna(); b = x[k + '_ext'] - x[k + '_true']
        g[k] = dict(kind='angle_deg', n=len(x), bias=float(b.mean()), sd=float(b.std()), passed=bool(abs(b.mean()) <= 5.0))
    for k, df in LEN.items():
        x = df[df.noise == noise][[k + '_ext', k + '_true']].dropna(); rb = float((x[k + '_ext'] - x[k + '_true']).sum() / x[k + '_true'].abs().sum())
        g[k] = dict(kind='relative', n=len(x), rel_bias=rb, passed=bool(abs(rb) <= 0.10))
    x = dp[dp.noise == noise]; d = ((x.div_ext - x.div_true + 180) % 360) - 180
    g['divergence'] = dict(kind='angle_deg_circular', n=int(d.notna().sum()), bias=float(d.mean()), sd=float(d.std()), passed=bool(abs(d.mean()) <= 5.0),
                           handedness_agree=float((x.hand_ext == x.hand_true).mean()))
    b = x.A3_rbar_ext - x.A3_rbar_true; g['A3_rbar'] = dict(kind='rbar', n=len(x), bias=float(b.mean()), sd=float(b.std()), passed=bool(abs(b.mean()) <= 0.05),
                                                          decision_agree=float((x.A3_dec_ext == x.A3_dec_true).mean()))
    y = dl[(dl.noise == noise) & dl.leaf.ne('internode')]
    g['arch_better'] = dict(ext_arc_share=float((y.better_ext == 'arc').mean()), true_arc_share=float((y.better_true == 'arc').mean()),
                            agree=float((y.better_ext == y.better_true).mean()))
    gate[noise] = g
json.dump(clean(dict(labels=LABELS, sigma_n_primary=sigma_n, gate=gate)), open(OUT / 'gate.json', 'w'), indent=1)
say('DONE', round(time.time() - T0, 1))
