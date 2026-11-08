"""C4A Step C (CONTRACT Amendment 17): per main-rank rachis multiplier from the Step B result arm.
m_c = (T_c / f_c - P) / R; T_c = reconstruction median L (rank c), f_c = median over the arm's rank-c main compounds of chain / (P + R), P, R = C4C bases.
python -X utf8 c4a_stepC.py <B arm, e.g. C4AB1.2481>  -> C4A_rachis_mult.json (main_rank_1_13)"""
import json, sys
from collections import defaultdict
import analyze as A
from analyze_c4a_stepA import recon_L

P, RB = 40 * 20 / 55, 40 * 35 / 55
arm = sys.argv[1]; R = A.load(); RL = recon_L()
fr = defaultdict(list)
for k, s in R.items():
    if k.startswith('_') or k.rsplit('_m', 1)[0] != arm: continue
    for c in s.get('c4c_measures', []):
        q = c['compound'].split('/')
        if len(q) == 2: fr[int(q[1][1:])].append(c['chain_mm'] / (P + RB))
mult, rows = [], []
for r in range(1, 14):
    f = A.q(fr[r], .5) if fr[r] else None; T = RL[r]['median']
    m = (T / f - P) / RB if f else 1.0
    note = '' if f else 'no engine compound: 1.0'
    if f and m < 0: note = 'not reachable (target below petiole): 0'; m = 0.0
    mult.append(m); rows.append(dict(rank=r, recon_median_mm=T, engine_n=len(fr[r]), engine_fraction_median=f, multiplier=m, note=note))
    print(r, T, len(fr[r]), None if f is None else round(f, 4), round(m, 4), note)
json.dump(dict(source_arm=arm, P_mm=P, R_mm=RB, main_rank_1_13=mult, rows=rows), open(sys.argv[2] if len(sys.argv) > 2 else 'C4A_rachis_mult.json', 'w'), indent=1)
