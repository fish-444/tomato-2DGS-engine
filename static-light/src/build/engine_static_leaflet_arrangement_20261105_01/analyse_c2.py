"""C-2 table: per-leaf absorbed light (a = 1, relative units x mm2) of the _04 MTG meshes (K20) vs the _02 meshes (K10) vs the reconstructed surfels (K8, v1tau_smoothn_r2), per leaf uid."""
import csv, glob, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REF = HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01'
FLAG = json.load(open(HERE.parents[0] / 'mtg_build_v2_clicks_20261105_04/PLANT_FLAGS.json', encoding='utf-8'))
P = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07']; rows = []; summ = []
for pid in P:
    a = json.load(open(HERE / f'out_k20/{pid}.json')); b = json.load(open(REF / f'out_k10/{pid}.json')); k = json.load(open(next(REF.glob(f'out_k8*/{pid}/plant.json'))))
    S = {(u if pid == 'R3-1b' else u.split('-')[-1]): v for u, v in k['dex']['2.4']['tau']['leaves'].items()}
    fl = FLAG['flags'].get(pid, []); flag = ';'.join(fl)
    if pid == 'R3-1b':                                                           # pilot: surfel leaves = R8_pc labels; MTG leaves = skeleton uids -> summed per R8_pc leaf (crosswalk lines)
        mp = {l['leaf_uid'].split('-')[-1]: f"R8pc_{l['leaf_R8pc']}" for l in json.load(open(HERE.parents[0] / 'engine_disc_vs_tri_20261101_01/T_meta.json'))['lines']}
        for d_ in (a, b):
            for mdl in ('v1', 'tau'):
                agg = {}
                for L, v in d_[mdl]['leaves'].items():
                    agg[mp[L]] = agg.get(mp[L], 0.0) + v
                d_[mdl]['leaves'] = agg
        flag += ';pilot_mtg_leaves_summed_per_R8pc (L05/L08/L09 duplicate L03/L10 per MTG build note)'
    xs, ys, zs = [], [], []
    for L in sorted(set(a['tau']['leaves']) | set(S)):
        m4 = a['tau']['leaves'].get(L); m2 = b['tau']['leaves'].get(L); s = S.get(L, {}).get('absorbed')
        rows.append(dict(plant=pid, leaf=L, mtg04_tau=m4, mtg02_tau=m2, surfel_tau=s, mtg04_over_surfel=round(m4 / s, 4) if m4 and s else '', mtg02_over_surfel=round(m2 / s, 4) if m2 and s else '',
                         mtg04_v1=a['v1']['leaves'].get(L), flag=flag))
        if m4 and s and m2:
            xs.append(s); ys.append(m4); zs.append(m2)
    xs, ys, zs = map(np.array, (xs, ys, zs))
    summ.append(dict(plant=pid, leaves_matched=len(xs), total_mtg04=round(a['tau']['total'], 1), total_mtg02=round(b['tau']['total'], 1), surfel_leaf_sum=round(float(sum(v['absorbed'] for v in S.values())), 1),
                     ratio04_sum=round(float(ys.sum() / xs.sum()), 4), ratio02_sum=round(float(zs.sum() / xs.sum()), 4), r_leaf04=round(float(np.corrcoef(xs, ys)[0, 1]), 3), r_leaf02=round(float(np.corrcoef(xs, zs)[0, 1]), 3),
                     median_abs_log_ratio04=round(float(np.median(np.abs(np.log(ys / xs)))), 3), median_abs_log_ratio02=round(float(np.median(np.abs(np.log(zs / xs)))), 3), flag=flag))
for name, rr in (('c2_per_leaf.csv', rows), ('c2_summary.csv', summ)):
    with open(HERE / name, 'w', encoding='utf-8', newline='') as f:
        f.write('# leaflet_arrangement_from_reconstruction; relative_light; a = 1; tau 0.07 leaflets; MTG _04 (K20) vs _02 (K10) vs reconstructed surfels v1tau_smoothn_r2 (K8); report only, not_validation\n')
        w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
for s in summ:
    print(s)
