"""RESULT_TABLE_v1_vs_v1tau.csv and the per-section tables from the kernel outputs (CONTRACT.md §4 + Amendment 1). Local, small. python -X utf8 build_table.py
Columns: section,item,scene,v1,v1_tau_a085,v1_tau_a1,v1_smooth,v1tau_smooth_a085,v1tau_smooth_a1,chg_v1tau_pct,chg_v1tau_a1_pct,chg_smoothn_pct,label,source
v1 = black leaves (a 1); v1_tau = tau 0.07 (a 0.85 and a 1 from the same trace); *_smooth = normal_mode smooth_all_r2 (receiver + occluder discs). chg_* are relative to v1 (for ratios, differences are in the same unit)."""
import csv, glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; B = HERE.parents[0]; A085 = 0.85
HEAD = 'light_model = v1_tau (tau 0.07, a 0.85, reflection dropped, stem/axis-like tau 0); assumed_leaf_optics; relative_light; conditional_on_unvalidated_scale; v1 = black leaves a 1; smooth = smooth_all_r2 (v1tau_smoothn_r2)'
PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05', 'e_01', 'e_04', 'e_05', 'e_07']
COLS = ['section', 'item', 'scene', 'v1', 'v1_tau_a085', 'v1_tau_a1', 'v1_smooth', 'v1tau_smooth_a085', 'v1tau_smooth_a1', 'chg_v1tau_pct', 'chg_v1tau_a1_pct', 'chg_smoothn_pct', 'label', 'source']
rows = []


def J(p):
    p = Path(p)
    return json.load(open(p, encoding='utf-8')) if p.exists() else None


def plant_json(prefix, pid):
    for f in glob.glob(str(HERE / f'out_{prefix}*' / pid / 'plant.json')):
        return json.load(open(f, encoding='utf-8'))
    return None


def r(x, n=6):
    return None if x is None else round(float(x), n)


def add(section, item, scene, v1=None, t1=None, s_v1=None, s_t1=None, a_dep=True, kind='ratio', label='', source=''):
    """t1 / s_t1 are a = 1 values of v1_tau; a_dep multiplies the a085 column by 0.85. kind 'ratio' -> change in % ; 'diff' -> absolute difference."""
    t85 = None if t1 is None else (t1 * A085 if a_dep else t1); s85 = None if s_t1 is None else (s_t1 * A085 if a_dep else s_t1)
    def chg(x):
        if x is None or v1 in (None, 0):
            return None
        return 100 * (x / v1 - 1) if kind == 'ratio' else x - v1
    rows.append(dict(section=section, item=item, scene=scene, v1=r(v1), v1_tau_a085=r(t85), v1_tau_a1=r(t1), v1_smooth=r(s_v1), v1tau_smooth_a085=r(s85), v1tau_smooth_a1=r(s_t1),
                     chg_v1tau_pct=r(chg(t85), 3), chg_v1tau_a1_pct=r(chg(t1), 3), chg_smoothn_pct=r(chg(s85), 3), label=label, source=source))


def get(d, *path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


# ---------------------------------------------------------------- plants (K5 raw, K8 smooth)
P = {pid: (plant_json('k5', pid), plant_json('k8', pid)) for pid in PLANTS}
for pid, (k5, k8) in P.items():
    if k5 is None:
        continue
    src = 'K5 (out_k5*/<id>/plant.json)' + (' + K8' if k8 else '')
    def vals(*path):
        return (get(k5, *path[:-1], 'v1', path[-1]) if False else None)
    def g(model, key, d='2.4'):
        return get(k5, 'dex', d, model, key), get(k8, 'dex', d, model, key)
    mv1, ms1 = g('v1', 'mean_light'); mt, mst = g('tau', 'mean_light')
    add('plant', 'mean_relative_light', pid, mv1, mt, ms1, mst, True, label='d_ex 2.4 mm; absorbed / area (b)', source=src)
    av1, as1 = g('v1', 'total'); at, ast = g('tau', 'total')
    add('plant', 'absorbed_total_ru_mm2', pid, av1, at, as1, ast, True, source=src)
    sv1, ss1 = g('v1', 'self_shading'); st, sst = g('tau', 'self_shading')
    add('plant', 'self_shading', pid, sv1, st, ss1, sst, False, 'diff', source=src)
    for th in ('lower', 'middle', 'upper'):
        x1, y1 = g('v1', f'{th}_mean_light'); x2, y2 = g('tau', f'{th}_mean_light')
        add('plant', f'mean_light_{th}_third', pid, x1, x2, y1, y2, True, source=src)
    uv1, us1 = g('v1', 'unassigned_share'); ut, ust = g('tau', 'unassigned_share')
    add('plant', 'unassigned_light_share', pid, uv1, ut, us1, ust, False, 'diff', label=(k5 or {}).get('label_note', ''), source=src)
    dv1, ds1 = g('v1', 'unassigned_share_dz10'); dt, dst = g('tau', 'unassigned_share_dz10')
    add('plant', 'unassigned_light_share_dz10_labels', pid, dv1, dt, ds1, dst, False, 'diff', label='label-0 / status-0 surfels of the dz10 snapshots (lower-bound labels)', source=src)
    cv1, cs1 = g('v1', 'cv'); ct, cst = g('tau', 'cv')
    add('plant', 'two_seed_CV_plant_absorbed', pid, cv1, ct, cs1, cst, False, 'diff', label='random-noise check (24 bar points / 72 dome directions, seeds 101/202)', source=src)
    ov = get(k5, 'dex', '2.4', 'tau', 'overflow_rays'); add('plant', 'overflow_rays_v1tau', pid, None, ov, None, get(k8, 'dex', '2.4', 'tau', 'overflow_rays'), False, 'diff', source=src)
    for d in ('1.5', '2.0', '3.0', '5.0'):
        x1 = get(k5, 'dex', d, 'v1', 'total'); x2 = get(k5, 'dex', d, 'tau', 'total'); y1 = get(k8, 'dex', d, 'v1', 'total'); y2 = get(k8, 'dex', d, 'tau', 'total')
        if x1 is None:
            continue
        add('d_ex', f'absorbed_total_d_ex_{d}_over_2.4', pid, x1 / av1, x2 / at, (y1 / as1) if y1 and as1 else None, (y2 / ast) if y2 and ast else None, False, 'ratio', label='ratio to the same model at d_ex 2.4 mm (the Result reports D at 1.5 / 5 mm)', source=src)
    for s in ('0.8', '1.2'):
        x1 = get(k5, 'footprint', s, 'v1', 'total'); x2 = get(k5, 'footprint', s, 'tau', 'total'); y1 = get(k8, 'footprint', s, 'v1', 'total'); y2 = get(k8, 'footprint', s, 'tau', 'total')
        if x1 is None:
            continue
        add('footprint', f'absorbed_total_footprint_x{s}_over_x1', pid, x1 / av1, x2 / at, (y1 / as1) if y1 and as1 else None, (y2 / ast) if y2 and ast else None, False, 'ratio', label='disc axes x scale, area (b) x scale^2', source=src)
        ss = float(s) ** 2
        add('footprint', f'absorbed_per_area_b_footprint_x{s}_over_x1', pid, x1 / av1 / ss, x2 / at / ss, (y1 / as1 / ss) if y1 and as1 else None, (y2 / ast / ss) if y2 and ast else None, False, 'ratio',
            label='occlusion-only effect: the footprint ratio divided by scale^2 (the area (b) of the receivers is scaled by scale^2 in this run; the Result footprint figures used a recalculated area, so compare this row, not the one above)', source=src)
    for fd in ('0.5', '0.7'):
        x1 = get(k5, 'diffuse', fd, 'v1', 'total'); x2 = get(k5, 'diffuse', fd, 'tau', 'total'); y1 = get(k8, 'diffuse', fd, 'v1', 'total'); y2 = get(k8, 'diffuse', fd, 'tau', 'total')
        if x1 is None:
            continue
        add('diffuse', f'absorbed_total_diffuse_{fd}_over_0.6', pid, x1 / av1, x2 / at, (y1 / as1) if y1 and as1 else None, (y2 / ast) if y2 and ast else None, False, 'ratio', source=src)
    h = get(k5, 'halfcredit'); h8 = get(k8, 'halfcredit')
    if h:
        add('half_credit', 'half_credit_on_over_off', pid, h['v1']['total_half'] / h['v1']['total_main'], h['tau']['total_half'] / h['tau']['total_main'],
            (h8['v1']['total_half'] / h8['v1']['total_main']) if h8 else None, (h8['tau']['total_half'] / h8['tau']['total_main']) if h8 else None, False, 'ratio', label=f"coincident receivers raw {h['v1']['hits']}" + (f", smoothed {h8['v1']['hits']}" if h8 else ''), source=src)
    for key in ('lai_pot_top', 'k_extinction_diagnostic', 'transmittance_at_rim_layer', 'transmittance_lowest_layer_inside_pot', 'leaf_area_fraction_above_pot_square'):
        add('canopy', key, pid, get(k5, 'sensors', 'v1', key), get(k5, 'sensors', 'tau', key), get(k8, 'sensors', 'v1', key), get(k8, 'sensors', 'tau', key), False, 'ratio',
            label='sensor grid 2 mm over the 59 mm pot square, 5 mm layers; area (b) (P1)', source=src)
    sm = get(k8, 'smoothing')
    if sm:
        add('smoothing', 'raw_normal_kept_share', pid, None, None, None, sm['kept_raw_share'], False, 'diff', source='K8')
        add('smoothing', 'neighbour_normal_angle_median_deg', pid, sm['noise_median_before_deg'], None, None, sm['noise_median_after_deg'], False, 'diff', label='before (raw) / after (smoothed)', source='K8')
    for L_, d1 in (get(k5, 'dex', '2.4', 'v1', 'leaves') or {}).items():
        t_ = get(k5, 'dex', '2.4', 'tau', 'leaves', L_); s1_ = get(k8, 'dex', '2.4', 'v1', 'leaves', L_); s2_ = get(k8, 'dex', '2.4', 'tau', 'leaves', L_)
        if d1['area_b_mm2'] > 0:
            add('per_leaf', 'leaf_light_per_area', f'{pid}|{L_}', d1['absorbed'] / d1['area_b_mm2'], t_['absorbed'] / t_['area_b_mm2'], (s1_['absorbed'] / s1_['area_b_mm2']) if s1_ else None,
                (s2_['absorbed'] / s2_['area_b_mm2']) if s2_ else None, True, 'ratio', label='assigned surfels only (lower bound)', source=src)

# ---------------------------------------------------------------- existing scenarios (stored) and K9 smoothing
tau_dir = B / 'engine_static_tau_20261031_01'
for kind, js in (('R3-1b alone', J(tau_dir / 'out_k1b/R3-1b/scenarios_tau.json')), ('tray 60 mm', J(tau_dir / 'out_k3/tray_scenarios_tau.json'))):
    if js is None:
        continue
    k9 = J(next(iter(glob.glob(str(HERE / 'out_k9*' / 'k9_results.json'))), 'none'))
    s9 = get(k9, 'scenarios_r31b' if kind.startswith('R3') else 'scenarios_tray', 'table') or {}
    for key, v in js['table'].items():
        sv = s9.get(key)
        add('scenario', 'relative_effect_vs_baseline', f'{kind}|{key}', v['rel_delta_v1'], v['rel_delta_tau'], sv['rel_delta_v1'] if sv else None, sv['rel_delta_tau'] if sv else None, False, 'diff',
            label=f"class {v['cls']}" + (f" / smooth {sv['cls']}" if sv else ''), source='engine_static_tau_20261031_01 (stored) + K9')
# tray attribution (K6 raw, K9 smoothing)
k67 = J(next(iter(glob.glob(str(HERE / 'out_k67*' / 'k67_results.json'))), 'none')); k9 = J(next(iter(glob.glob(str(HERE / 'out_k9*' / 'k9_results.json'))), 'none'))
if k67:
    for pitch, d in k67['K6_tray'].items():
        for unit in d['v1']:
            for key in ('self_shading_loss', 'neighbour_shading_loss', 'total_loss'):
                s = get(k9, 'tray_attribution', pitch)
                add('tray_attribution', key, f'tray {pitch} mm|{unit}', d['v1'][unit][key], d['tau0.07'][unit][key], get(s, 'v1', unit, key), get(s, 'tau0.07', unit, key), False, 'diff', source='K6 (K67) + K9')

# ---------------------------------------------------------------- disc vs triangle (existing summaries) with smoothed D from K8
dvt = J(B / 'engine_disc_vs_tri_20261101_01/out_k1/results.json'); cur = J(B / 'engine_curved_T_20261101_01/out_k1/results.json'); cop = J(B / 'engine_coplanar_sep_20261101_01/out_k1/results.json')
k5r, k8r = P['R3-1b']
def T_over_D(S, T_key, D_key, model):
    return S[f'{T_key}|{model}']['total']['mean'] / S[f'{D_key}|{model}']['total']['mean']
if cop and cur and dvt:
    D8 = {m: get(k8r, 'dex', '2.4', m, 'assigned_leaf_total') for m in ('v1', 'tau')} if k8r else {}
    arms = [('T_flat', cop, 'T'), ('T_diamond', dvt, 'T_diamond'), ('T_flat_offset_0.05', cop, 'T_off0.05'), ('T_flat_offset_0.1', cop, 'T_off0.1'), ('T_flat_offset_0.2', cop, 'T_off0.2'), ('T_fold', cop, 'T_fold'), ('T_foldcup', cop, 'T_foldcup')]
    for name, js, key in arms:
        S = js['summary']; tv1 = S[f'{key}|v1']['total']['mean']; tt = S[f'{key}|tau0.07']['total']['mean']
        add('disc_vs_tri', 'T_over_D_blades', name, T_over_D(S, key, 'D_blades', 'v1'), T_over_D(S, key, 'D_blades', 'tau0.07'), (tv1 / D8['v1']) if D8.get('v1') else None, (tt / D8['tau']) if D8.get('tau') else None, False, 'ratio',
            label='T unchanged; smoothed D (K8, R3-1b blades) in the smooth columns', source='engine_*_20261101_01/out_k1/results.json + K8')
        add('disc_vs_tri', 'T_self_shading', name, S[f'{key}|v1']['self']['mean'], S[f'{key}|tau0.07']['self']['mean'], None, None, False, 'diff', source='engine_*_20261101_01/out_k1/results.json')
    loc = J(HERE / 'R3-1b_dex_blades_local.json')
    if loc:
        T0v = cop['summary']['T|v1']['total']['mean']; T0t = cop['summary']['T|tau0.07']['total']['mean']
        for d in ('1.5', '2.0', '3.0', '5.0'):
            add('disc_vs_tri', 'T_over_D_blades_at_d_ex', f'D d_ex {d} mm', T0v / loc[f'{d}|v1']['blades'], T0t / loc[f'{d}|tau']['blades'], (T0v / get(k8r, 'dex', d, 'v1', 'assigned_leaf_total')) if get(k8r, 'dex', d, 'v1', 'assigned_leaf_total') else None,
                (T0t / get(k8r, 'dex', d, 'tau', 'assigned_leaf_total')) if get(k8r, 'dex', d, 'tau', 'assigned_leaf_total') else None, False, 'ratio', label='flat engine-law T over D blades at that d_ex (R3-1b local small run for raw)', source='R3-1b_dex_blades_local.json + K8')
if k67 and k67.get('K7_static'):
    s = k67['K7_static']['totals']; g = k67.get('K7_groimp', {})
    add('closure', 'static_T_engine_lamp_total', 'R3-1b T', s['v1'], s['tau_a1'], None, None, True, 'ratio', label='engine lamp 1200 at 0.6 m; coincident half credit; leaflets only', source='K7 (K67)')
    for name, v in g.items():
        if v.get('ok') and v.get('mean'):
            ref_s = s['v1'] if name.startswith('black') else (s['tau_a085'] if 'rho0.08' in name else s['tau_a1'] * 0.93)
            add('closure', 'GroIMP_over_static', f'GroIMP {name}', None, v['mean']['total'] / ref_s, None, None, False, 'ratio', label=f"depth {v['depth']}; packet sigma {v.get('packet_sigma_total')}; static reference = v1 (black) / v1_tau a 0.85 (rho 0.08) / v1_tau a 0.93 (rho 0)", source='K7 (K67)')

# ---------------------------------------------------------------- MTG cross-check (K10 meshes with tau / black leaves vs surfel per-leaf light), report only
from scipy.stats import spearmanr
u2r = J(B / 'engine_static_normals_20261104_01/kaggle_flat/mtg__R3-1b_uid_to_R8pc.json') or {}
for pid in ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 'e_01', 'e_04', 'e_05', 'e_07']:
    mt = J(HERE / 'out_k10' / f'{pid}.json'); k5, k8 = P[pid]
    if not (mt and k5):
        continue
    def per_leaf(mdl):
        d = {}
        for u, v in mt[mdl]['leaves'].items():
            key = f"R8pc_{u2r.get(f'R3-1b-{u}', u2r.get(u))}" if pid == 'R3-1b' else f'{pid}-{u}'
            d[key] = d.get(key, 0.0) + v
        return d
    def corr(mdl, plant):
        sl = get(plant, 'dex', '2.4', 'v1' if mdl == 'v1' else 'tau', 'leaves') if plant else None
        if not sl:
            return None, None
        mp = per_leaf(mdl); ks = [k for k in mp if k in sl and sl[k]['area_b_mm2'] > 0 and mp[k] > 0]
        rho = float(spearmanr([mp[k] for k in ks], [sl[k]['absorbed'] for k in ks])[0]) if len(ks) >= 3 else None
        return rho, mt[mdl]['total'] / sum(v['absorbed'] for v in sl.values())
    r1, t1 = corr('v1', k5); r2, t2 = corr('tau', k5); r3, t3 = corr('v1', k8); r4, t4 = corr('tau', k8)
    add('mtg_cross_check', 'leaf_rank_spearman_mtg_vs_surfel', pid, r1, r2, r3, r4, False, 'diff', label='report only; MTG leaflets (smooth reconstructions from clicks) vs surfel per-leaf light of the same model; v1_tau columns = a 1 values', source='K10 + K5/K8')
    add('mtg_cross_check', 'total_ratio_mtg_leaflets_over_surfel_assigned', pid, t1, t2, t3, t4, False, 'diff', source='K10 + K5/K8')
# ---------------------------------------------------------------- synthetic check at tau 0.07 (K11, report only)
sy = J(HERE / 'out_k11' / 'synth_tau.json')
if sy:
    for arm in ('raw', 'rec_r2', 'smooth_all_r2'):
        def agg(m):
            s = [x for x in sy['rows'] if x['model'] == m and x['arm'] == arm]; l = [abs(x['err_pct']) for x in sy['leaf_rows'] if x['model'] == m and x['arm'] == arm]
            return float(np.mean([x['plant_bias_pct'] for x in s])), float(np.median(l)), float(np.mean([x['self_change'] for x in s]))
        b0, l0, s0 = agg('v1'); b1, l1, s1 = agg('tau')
        add('synthetic_normals', 'plant_bias_pct_vs_truth', arm, b0, b1, None, None, False, 'diff', label='7 MTG meshes, 44 deg noise, seed 20261105; v1 (tau 0) vs v1_tau (tau 0.07); report only', source='K11 (out_k11/synth_tau.json)')
        add('synthetic_normals', 'leaf_median_abs_err_pct', arm, l0, l1, None, None, False, 'diff', source='K11')
        add('synthetic_normals', 'self_shading_change', arm, s0, s1, None, None, False, 'diff', source='K11')
wr = csv.DictWriter(open(HERE / 'RESULT_TABLE_v1_vs_v1tau.csv', 'w', encoding='utf-8', newline=''), fieldnames=COLS)
wr.writeheader(); wr.writerows(rows)
(HERE / 'RESULT_TABLE_header.txt').write_text(HEAD + '\n', encoding='utf-8')
print(len(rows), 'rows; plants with K5', sum(1 for p in P.values() if p[0]), 'K8', sum(1 for p in P.values() if p[1]))
