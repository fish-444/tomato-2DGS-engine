"""Build the K12 plant kernel of optics rule 2 from the K8 kernel of the v1_tau reference (same placement, smoothing hook, metrics), replacing run_plant with the rule-2 arms."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; REF = HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01'
s = (REF / 'kaggle_k8/k8_plants.py').read_text(encoding='utf-8')
s = s.replace('"""K5 of the v1_tau reference table', '"""K12 (optics rule 2, engine_static_tau_rule2_20261105_01/CONTRACT.md), built from K8 of the v1_tau reference. Original header: K5 of the v1_tau reference table', 1)
s = s.replace("REF = Path(a.ref) if a.ref else Path(glob.glob('/kaggle/input/**/refk__R3-1b.npz', recursive=True)[0]).parent",
              "REF = Path(a.ref) if a.ref else Path(glob.glob('/kaggle/input/**/stem__R3-1b.npz', recursive=True)[0]).parent   # rule-2 dataset: stem masks + K8 reference arrays")
s = s.replace("def trace_m(q, built, model, dex_mm):\n    return {sd: trace_plants_tau([q], [q], sd, EPS, 0.0, 0.0, dex_mm / 1000.0, F, tau_scale=1.0, use_v1=(model == 'v1'), built=built)[q.pid] for sd in SEEDS}",
              "def trace_m(q, built, model, dex_mm, scale=1.0):\n    return {sd: trace_plants_tau([q], [q], sd, EPS, 0.0, 0.0, dex_mm / 1000.0, F, tau_scale=scale, use_v1=(model == 'v1'), built=built)[q.pid] for sd in SEEDS}")
i = s.index('def run_plant(pid):'); j = s.index('for pid in PLANTS:')
NEW = '''def tau_rule2(pid, q, d):
    """Rule 2: tau 0.07 on every surfel except the stem (tau 0). Pilot stem = axis_kind 1; others = within d mm of the stem polylines and not leaf-labelled (make_stem.py)."""
    z = np.load(REF / f'stem__{pid}.npz'); rows = q.s.index
    if not np.array_equal(z['rows'], rows):
        raise RuntimeError(f'{pid}: stem rows != opaque rows')
    stem = z[f'stem_d{d:g}'].astype(bool)
    return np.where(stem, 0.0, CFG['tau_leaf']), int(stem.sum())


def run_plant(pid):
    t0 = time.time(); q = placed(pid); A = q.A; n = len(A); leaf, uids, prim, dz, lnote = leaf_labels(pid, q); tau1 = np.asarray(q.tau, float).copy()
    res = dict(plant=pid, surfels=n, label_note=lnote, area_b_mm2=float(A.sum()), engine_hash=HASH, seeds=list(SEEDS), a_main=A_MAIN, eps_mm=1000 * EPS, smoothing=q.smooth_info,
               labels=['light_model_v1tau2', 'smoothn_r2', 'assumed_leaf_optics', 'relative_light', 'conditional_on_unvalidated_scale', 'skeleton_labels_ungraded'], arms={})
    # regression gate: rule-2 code with the rule-1 tau map reproduces v1tau_smoothn_r2 (K8, stored float32) bit for bit in float32
    built = build_scene([q]); recs = trace_m(q, built, 'tau', 2.4, 1.0)
    Kt = np.mean([obs.kernel(recs[sd]['parts']) for sd in SEEDS], 0); zr = np.load(REF / f'k8ref__{pid}.npz')
    same = bool(np.array_equal(zr['ply_row'], q.s.index) and np.array_equal(Kt.astype(np.float32), zr['K_tau']))
    res['gate'] = dict(identical_float32=same, max_abs_diff=float(np.abs(Kt.astype(np.float32) - zr['K_tau']).max()))
    res['arms']['rule1_tau0.07'] = metr(q, recs, A, leaf, uids, prim, dz, extra=True); log(f'{pid} gate {res["gate"]}')
    if not same:
        (OUT / pid).mkdir(exist_ok=True); jdump(res, OUT / pid / 'plant.json'); jdump(dict(gate_failed=pid, gate=res['gate']), OUT / 'GATE_FAILED.json'); return False
    for d in (1.5, 1.0, 2.5):
        q.tau, ns = tau_rule2(pid, q, d); b2 = build_scene([q]); res.setdefault('stem_surfels', {})[f'd{d:g}'] = ns
        for lev in ((0.05, 0.07, 0.10) if d == 1.5 else (0.07,)):
            tt = time.time(); recs = trace_m(q, b2, 'tau', 2.4, lev / CFG['tau_leaf']); res['arms'][f'rule2_d{d:g}_tau{lev:g}'] = metr(q, recs, A, leaf, uids, prim, dz, extra=True)
            log(f'{pid} rule2 d{d:g} tau{lev:g} {time.time() - tt:.0f} s')
            if d == 1.5 and lev == 0.07:
                K2 = np.mean([obs.kernel(recs[sd]['parts']) for sd in SEEDS], 0)
        if d == 1.5:
            res['sensors_rule2_d1.5_tau0.07'] = sensors(q, b2, 'tau', A)
    q.tau = tau1
    (OUT / pid).mkdir(exist_ok=True); np.savez_compressed(OUT / pid / 'per_surfel.npz', ply_row=q.s.index, K_rule1=Kt.astype(np.float32), K_rule2=K2.astype(np.float32), area_b_mm2=A.astype(np.float32))
    res['seconds'] = round(time.time() - t0); jdump(res, OUT / pid / 'plant.json'); (OUT / pid / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    log(f'{pid} done {res["seconds"]} s'); obs._CACHE.clear()
    return True


'''
s = s[:i] + NEW + s[j:]
s = s.replace("log('K5_DONE')", "log('K12_DONE')")
(HERE / 'kaggle_k12').mkdir(exist_ok=True); (HERE / 'kaggle_k12/k12_rule2.py').write_text(s, encoding='utf-8')
SRC = ['dolgfish/static-tau-bundle-20261031', 'dolgfish/static-nine-bundle-20261028', 'dolgfish/static-normals-20261104', 'dolgfish/static-normals-v2-20261104', 'dolgfish/static-tau-rule2-20261105']
for tag, plants in (('a', 'R3-1b,n_01,n_02,n_03,n_04,n_05,n_06'), ('b', 't2_02,t2_04,t2_05,e_01,e_04,e_05,e_07')):
    d = HERE / f'kaggle_k12{tag}'; d.mkdir(exist_ok=True); (d / 'k12_rule2.py').write_text(s.replace('@PLANTS@', plants), encoding='utf-8')
    json.dump({'id': f'dolgfish/static-rule2-k12{tag}-20261105', 'title': f'static-rule2-k12{tag}-20261105', 'code_file': 'k12_rule2.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
               'enable_gpu': False, 'enable_internet': False, 'dataset_sources': SRC, 'competition_sources': [], 'kernel_sources': []}, open(d / 'kernel-metadata.json', 'w'))
print('ok')
