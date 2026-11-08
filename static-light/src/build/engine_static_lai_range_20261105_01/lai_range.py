"""Parent prompt §2–4 (CONTRACT.md): compound-leaf boxes, ranks, arm R (p90 per rank class) and arm B (L x W x f), LAI range per plant, sensitivity. Local, CPU, small.
Also writes leaves.npz (per plant: per-leaf box frame, A_rec, A_missing per arm) used by the fill step. python -X utf8 lai_range.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
sys.path.insert(0, str(B / 'engine_static_tau_20261031_01/kaggle_data'))
from static.geometry import read_ply, footprint_k
NF = B / 'engine_static_normals_20261104_01/kaggle_flat'; S = B / 'mtg_build_v2_clicks_20261103_02/static'; DZ = B / 'p1_dz10_labels_20261101_01'; REF = B / 'engine_static_v1tau_reference_20261104_01'
PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05', 'e_01', 'e_04', 'e_05', 'e_07']
SKV = {'n_01': '5e803db0e8f8', 'n_02': '6e44394dd4d8', 'n_03': 'd267404f6ada', 'n_04': 'a40266c80d19', 'n_05': 'bcf8d6cef69a', 'n_06': '64b0f9b553e9', 'e_01': '26f27427b73b', 'e_04': '73fdd1ffeead',
       'e_05': '524f23114149', 'e_07': '16690f4f8d25', 't2_02': 'a6e7d3823d42', 't2_04': '6e2e267877ca', 't2_05': 'f67c164d6f3c'}
GROUND = 59.0 * 59.0                         # mm2, pot-top square (obs.POT_SIDE)
MIN_SURFELS = 100


def ply_path(pid):
    return B / 'engine_static_nine_20261028_01/kaggle_data/R3-1b/surfels.ply' if pid == 'R3-1b' else (NF / f'{pid}__surfels.ply' if pid.startswith('e_') else B / f'engine_static_nine_20261028_01/kaggle_data/{pid}/surfels.ply')


def area_b(pid, rows):
    if pid.startswith('e_'):
        z = np.load(NF / f'{pid}__area_b.npz'); assert np.array_equal(z['rows'], rows); return z['area_b_mm2'].astype(float)
    d = {}
    with open(B / f'engine_static_nine_20261028_01/kaggle_data/{pid}/static_surfels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith('#')):
            d[int(r['surfel_row'])] = float(r['area_b_mm2'])
    return np.array([d[int(x)] for x in rows])


def amap(src_rows, vals, rows):
    o = np.argsort(src_rows); i = o[np.searchsorted(src_rows[o], rows)]; assert np.array_equal(src_rows[i], rows); return vals[i]


def load(pid):
    d = read_ply(ply_path(pid)); rows = np.nonzero(d['opacity'] > 0.5)[0]; d = d[rows]
    X = np.stack([d['x'], d['y'], d['z']], 1).astype(float) * 1000.0; A = area_b(pid, rows); k = footprint_k(d['opacity']); ab = np.stack([k * d['s0'], k * d['s1']], 1).astype(float) * 1000.0
    if pid == 'R3-1b':
        z = np.load(B / 'engine_static_tau_20261031_01/kaggle_data/labels_rachis.npz'); ax = amap(z['rows'], z['axis_kind'].astype(int), rows); r8 = amap(z['rows'], z['label'].astype(int), rows)
        leaf = np.where((ax == 0) & (r8 >= 1), r8, -1); uids = {int(k_): f'R8pc_{int(k_)}' for k_ in np.unique(leaf[leaf >= 1])}
    else:
        z = np.load(NF / f'labels__{pid}.npz'); lr = amap(z['rows'], z['label'].astype(int), rows); st = amap(z['rows'], z['status'].astype(int), rows)
        leaf = np.where((st == 1) & (lr > 0), lr, -1); uids = {i + 1: str(u) for i, u in enumerate(z['leaf_uids'])}
    return rows, X, A, ab, leaf, uids


def ranks(pid, uids):
    """uid -> (class, rank): class 'main' with rank r, or 'side'; None if no rank."""
    out = {}
    if pid == 'R3-1b':
        sp = json.load(open(S / 'mtg_R3-1b_spec.json', encoding='utf-8')); kk = {l['uid']: l['k'] for l in sp['leaves']}
        sk = json.load(open(B.parent / 'data/skeletons/R3-1b/2026-10-21/skeleton_v1.json', encoding='utf-8'))
        by = {}
        for l in sk['leaves']:
            if l['status'] in ('active', 'new', 'candidate_lost') and l['leaf_uid'] in kk:
                by.setdefault(int(l['attrs']['pipeline_leaf']), []).append(kk[l['leaf_uid']])
        for lab, u in uids.items():
            out[u] = ('main', int(min(by[lab]))) if lab in by else None
        return out
    sk = json.load(open(DZ / f'{pid}_{SKV[pid]}/skeleton_v1.json', encoding='utf-8')); axt = {a['axis_uid']: a['type'] for a in sk.get('axes', [])}
    att = {l['leaf_uid']: (l.get('attach_axis_uid'), l.get('attach_s_mm')) for l in sk['leaves']}
    if (S / f'mtg_{pid}_spec.json').exists():
        kk = {l['uid']: l['k'] for l in json.load(open(S / f'mtg_{pid}_spec.json', encoding='utf-8'))['leaves']}
    else:                                                                       # t2_*: order of attach_s_mm along the main axis
        mains = sorted((s, u) for u, (a, s) in att.items() if s is not None and axt.get(a, 'main') == 'main'); kk = {u: i + 1 for i, (s, u) in enumerate(mains)}
    for u in uids.values():
        a, s = att.get(u, (None, None))
        if a is not None and axt.get(a) not in (None, 'main'):
            out[u] = ('side', 0)
        else:
            out[u] = ('main', int(kk[u])) if u in kk else None
    return out


def box(Xl, Al):
    c = (Xl * Al[:, None]).sum(0) / Al.sum(); C = np.cov((Xl - c).T, aweights=Al); w, U = np.linalg.eigh(C); U = U[:, ::-1]
    P = (Xl - c) @ U; lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    return c, U, lo, hi, hi - lo


leaves, plants = [], {}
for pid in PLANTS:
    rows, X, A, ab, leaf, uids = load(pid); rk = ranks(pid, uids); k8 = json.load(open(next(REF.glob(f'out_k8*/{pid}/plant.json')), encoding='utf-8'))
    plants[pid] = dict(lai_current=k8['sensors']['tau']['lai_pot_top'], area_total=float(A.sum()), area_leaf_labelled=float(A[leaf >= 1].sum()), unassigned_share=float(A[leaf < 1].sum() / A.sum()), n_surfels=len(A))
    for lab, u in uids.items():
        m = leaf == lab; n = int(m.sum())
        rec = dict(plant=pid, leaf=u, label=lab, surfels=n, A_rec=float(A[m].sum()), cls=None, rank=None)
        if rk.get(u):
            rec['cls'], rec['rank'] = rk[u]
        if n >= 3:
            c, U, lo, hi, ext = box(X[m], A[m]); rec.update(L=float(ext[0]), W=float(ext[1]), T=float(ext[2]), centre=c.tolist(), frame=U.T.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                                                              disc_a_med=float(np.median(ab[m, 0])), disc_b_med=float(np.median(ab[m, 1])))
        leaves.append(rec)
# ---- arm R: p90 per class (merge sparse ranks downward), arm B: f from the pilot
def class_key(l):
    return ('side', 0) if l['cls'] == 'side' else ('main', l['rank'])
eligible = [l for l in leaves if l['cls'] and l['surfels'] >= MIN_SURFELS]
def pool_table(q):
    groups = {}
    for l in eligible:
        groups.setdefault(class_key(l), []).append(l['A_rec'])
    mains = sorted(k for k in groups if k[0] == 'main'); merged = {}; acc, keys = [], []
    for k in mains:                                                                 # merge with the adjacent lower rank until n >= 5
        acc += groups[k]; keys.append(k[1])
        if len(acc) >= 5:
            for r in keys:
                merged[('main', r)] = (float(np.percentile(acc, q)), len(acc), f'{keys[0]}-{keys[-1]}' if len(keys) > 1 else str(keys[0]))
            acc, keys = [], []
    if keys:                                                                         # leftover top ranks: merge into the last class
        last = max(v for v in merged if v[0] == 'main') if merged else None
        prev = [l['A_rec'] for l in eligible if l['cls'] == 'main' and (l['rank'] in keys or (last and merged[last][2] and l['rank'] in range(int(merged[last][2].split('-')[0]), int(merged[last][2].split('-')[-1]) + 1)))]
        span = (merged[last][2].split('-')[0] if last else str(keys[0])) + f'-{keys[-1]}'
        for r in keys + ([int(x) for x in range(int(merged[last][2].split('-')[0]), int(merged[last][2].split('-')[-1]) + 1)] if last else []):
            merged[('main', r)] = (float(np.percentile(prev, q)), len(prev), span)
    if ('side', 0) in groups:
        merged[('side', 0)] = (float(np.percentile(groups[('side', 0)], q)), len(groups[('side', 0)]), 'side')
    return merged
pil = [l for l in leaves if l['plant'] == 'R3-1b' and 'L' in l]; fvals = np.array([l['A_rec'] / (l['L'] * l['W']) for l in pil]); F = float(np.median(fvals))


def fill(l, arm, q=90, cap=2.0, table=None):
    if not l['cls'] or l['surfels'] < MIN_SURFELS or 'L' not in l:
        return None
    Aexp = table[class_key(l)][0] if arm == 'R' else l['L'] * l['W'] * F
    return dict(A_exp=Aexp, c=min(1.0, l['A_rec'] / Aexp) if Aexp > 0 else 1.0, A_missing=min(max(0.0, Aexp - l['A_rec']), (cap - 1) * l['A_rec']))


T90 = pool_table(90); tabs = {q: pool_table(q) for q in (80, 90, 95)}
box_rows = []
for l in leaves:
    fr, fb = fill(l, 'R', table=T90), fill(l, 'B')
    l['A_missing_R'] = fr['A_missing'] if fr else 0.0; l['A_missing_B'] = fb['A_missing'] if fb else 0.0
    box_rows.append(dict(plant=l['plant'], leaf=l['leaf'], cls=l['cls'], rank=l['rank'], surfels=l['surfels'], L_mm=round(l.get('L', np.nan), 2), W_mm=round(l.get('W', np.nan), 2), T_mm=round(l.get('T', np.nan), 2),
                         A_rec_mm2=round(l['A_rec'], 2), A_exp_R=round(fr['A_exp'], 2) if fr else None, A_exp_B=round(fb['A_exp'], 2) if fb else None, c_R=round(fr['c'], 4) if fr else None, c_B=round(fb['c'], 4) if fb else None,
                         A_missing_R=round(l['A_missing_R'], 2), A_missing_B=round(l['A_missing_B'], 2), filled=bool(fr is not None),
                         cap_binding_R=bool(fr and fr['A_missing'] < fr['A_exp'] - l['A_rec'] - 1e-9), cap_binding_B=bool(fb and fb['A_missing'] < fb['A_exp'] - l['A_rec'] - 1e-9)))
def lai_up(pid, arm, q=90, cap=2.0):
    t = tabs[q] if arm == 'R' else None; add = 0.0
    for l in leaves:
        if l['plant'] == pid:
            f_ = fill(l, arm, q, cap, t); add += f_['A_missing'] if f_ else 0.0
    return plants[pid]['lai_current'] + add / GROUND
per = []
for pid, p in plants.items():
    lv = [l for l in leaves if l['plant'] == pid]; nf = sum(1 for l in lv if fill(l, 'R', table=T90))
    lo = p['lai_current']; uR, uB = lai_up(pid, 'R'), lai_up(pid, 'B')
    per.append(dict(plant=pid, ground_area_mm2=GROUND, n_leaves=len(lv), n_filled=nf, n_not_filled=len(lv) - nf, unassigned_area_share=round(p['unassigned_share'], 4), LAI_low=round(lo, 4),
                    LAI_leaf_labelled_only=round(p['area_leaf_labelled'] / GROUND, 4), LAI_up_R=round(uR, 4), LAI_up_B=round(uB, 4), ratio_R=round(uR / lo, 4), ratio_B=round(uB / lo, 4)))
sens = []
for pid in plants:
    lo = plants[pid]['lai_current']
    sens.append(dict(plant=pid, LAI_low=round(lo, 4), **{f'LAI_up_R_p{q}': round(lai_up(pid, 'R', q), 4) for q in (80, 90, 95)}, **{f'LAI_up_R_cap{c}': round(lai_up(pid, 'R', 90, c), 4) for c in (1.5, 2.0, 3.0)},
                     **{f'LAI_up_B_cap{c}': round(lai_up(pid, 'B', 90, c), 4) for c in (1.5, 2.0, 3.0)}))
TAG = 'lai_range; skeleton_labels_ungraded; conditional_on_unvalidated_scale; not_validation'
def wcsv(name, rows, note):
    with open(HERE / name, 'w', encoding='utf-8', newline='') as f:
        f.write(f'# {TAG}; {note}\n'); w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
wcsv('lai_range_per_plant.csv', per, 'ground = pot-top square 59 x 59 mm; LAI_low = current rule (pilot V-P leaves, others all opaque surfels) = K8 sensors.tau lai_pot_top; LAI_up = LAI_low + fill area / ground')
wcsv('compound_leaf_boxes.csv', box_rows, f'box = p1-p99 extents along area-weighted PCA axes; arm R p90 per rank class (pooled 14 plants); arm B f = {F:.4f} (pilot median); cap 2 x A_rec; not filled: < 100 surfels or no rank')
wcsv('lai_range_sensitivity.csv', sens, 'arm R p80 / p90 / p95; caps 1.5 / 2 / 3 (report only)')
json.dump(dict(f_pilot=F, f_pilot_values=fvals.tolist(), f_pilot_p10=float(np.percentile(fvals, 10)), f_pilot_p90=float(np.percentile(fvals, 90)),
               armR_p90_table={f'{k[0]}_{k[1]}': dict(p90=v[0], n=v[1], pooled_ranks=v[2]) for k, v in sorted(T90.items(), key=lambda kv: (kv[0][0], kv[0][1]))},
               leaves=[{k: v for k, v in l.items()} for l in leaves], plants=plants), open(HERE / 'leaves.json', 'w'), indent=1, default=float)
for r in per:
    print(r['plant'], 'low', r['LAI_low'], 'labelled-only', r['LAI_leaf_labelled_only'], 'up R', r['LAI_up_R'], 'up B', r['LAI_up_B'], 'n', r['n_leaves'], 'filled', r['n_filled'])
print('f pilot', round(F, 4), 'p10-p90', np.percentile(fvals, [10, 90]).round(3)); print({f'{k[0]}{k[1]}': (round(v[0]), v[1], v[2]) for k, v in sorted(T90.items())})
