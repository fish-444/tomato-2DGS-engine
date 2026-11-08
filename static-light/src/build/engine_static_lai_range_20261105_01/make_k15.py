"""K15 (addendum §3): min / max LAI and 5 mm layer transmittance with fill surfels, built from K8 of the v1_tau reference (same placement, smoothing hook, sensors).
Gate: min case reproduces K8 sensors.tau exactly. Max cases: fill discs added as occluders (tau 0.07) and receivers (area = cell area)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; REF = HERE.parents[0] / 'engine_static_v1tau_reference_20261104_01'
s = (REF / 'kaggle_k8/k8_plants.py').read_text(encoding='utf-8')
s = s.replace("REF = Path(a.ref) if a.ref else Path(glob.glob('/kaggle/input/**/refk__R3-1b.npz', recursive=True)[0]).parent",
              "REF = Path(a.ref) if a.ref else Path(glob.glob('/kaggle/input/**/k8sensors__R3-1b.json', recursive=True)[0]).parent   # LAI-range dataset: fill surfels + K8 sensors / totals")
i = s.index('def run_plant(pid):'); j = s.index('for pid in PLANTS:')
NEW = '''CASES = ['R', 'B', 'R_reach4', 'R_reach8', 'R_grid1', 'R_grid3']


class Aug:
    """Placed plant (smoothed) + fill discs: occluders and receivers. Fill centres given in board metres, moved by the plant's placement offset."""
    def __init__(self, q, z, pid):
        self.pid = pid; self.pos = q.pos; nf = len(z['area_mm2']); self.nf = nf
        Pq, kq = q.prims(); Pf = np.zeros((nf, 14)); Pf[:, 0:3] = z['centre_board_m'] + q.offset; Pf[:, 3:6] = z['tu']; Pf[:, 6:9] = z['tv']; Pf[:, 9:12] = z['normal']; Pf[:, 12] = z['a_m']; Pf[:, 13] = z['b_m']
        self._P = np.concatenate([Pq, Pf]); self._k = np.concatenate([kq, np.full(nf, kq[0], np.int8)])
        self.centre = np.concatenate([q.centre, Pf[:, 0:3]]); self.normal = np.concatenate([q.normal, z['normal']]); self.tau = np.concatenate([np.asarray(q.tau, float), np.full(nf, CFG['tau_leaf'])])
        self.A = np.concatenate([q.A, z['area_mm2']]); self.leaf = np.concatenate([q.leaf, np.ones(nf, bool)]); self.own_local = np.arange(len(self.A)); self.delta = None
        self.fill = np.r_[np.zeros(len(q.A), bool), np.ones(nf, bool)]
    def prims(self):
        return self._P, self._k


def totals(obj, recs, A, fill=None):
    ab, abu = [], []
    for sd in SEEDS:
        K, Ku, U = kU(obj, recs, sd); ab.append(U * K * A); abu.append(U * Ku * A)
    abm = np.mean(ab, 0); t = float(np.mean([float(x.sum()) for x in ab]))   # same summation order as K8 (mean of per-seed sums)
    return dict(absorbed_a1=t, self_shading=1 - t / float(np.mean([float(x.sum()) for x in abu])), fill_share=float(abm[fill].sum() / abm.sum()) if fill is not None else 0.0)


def run_plant(pid):
    t0 = time.time(); q = placed(pid); A = q.A; ref = json.load(open(REF / f'k8sensors__{pid}.json'))
    res = dict(plant=pid, labels=['lai_range', 'fill_surfels_assumed', 'relative_light', 'skeleton_labels_ungraded', 'conditional_on_unvalidated_scale', 'not_validation'], reference='v1tau_smoothn_r2', smoothing=q.smooth_info, cases={})
    built = build_scene([q]); recs = trace_m(q, built, 'tau', 2.4); sen = sensors(q, built, 'tau', A); tot = totals(q, recs, A)
    same = bool(json.dumps(sen, sort_keys=True) == json.dumps(ref['sensors_tau'], sort_keys=True) and abs(tot['absorbed_a1'] - ref['absorbed_tau_total']) <= a.tol * abs(ref['absorbed_tau_total']))   # tol 0 on Kaggle = exact
    res['gate'] = dict(identical=same, lai_here=sen['lai_pot_top'], lai_ref=ref['sensors_tau']['lai_pot_top'], absorbed_here=tot['absorbed_a1'], absorbed_ref=ref['absorbed_tau_total'],
                       max_layer_T_diff=max(abs(x['transmittance'] - y['transmittance']) for x, y in zip(sen['profile'], ref['sensors_tau']['profile'])))
    res['cases']['min'] = dict(sensors=sen, **tot, fill_area_mm2=0.0); log(f'{pid} gate {res["gate"]["identical"]}')
    if not same:
        (OUT / pid).mkdir(exist_ok=True); jdump(res, OUT / pid / 'plant.json'); jdump(dict(gate_failed=pid, gate=res['gate']), OUT / 'GATE_FAILED.json'); return False
    for case in CASES:
        z = np.load(REF / f'fill__{pid}__{case}.npz')
        if len(z['area_mm2']) == 0:
            res['cases'][f'max_{case}'] = dict(sensors=sen, **tot, fill_area_mm2=0.0, note='no fill surfels'); continue
        tt = time.time(); g = Aug(q, z, pid); bg = build_scene([g]); rg = trace_m(g, bg, 'tau', 2.4)
        res['cases'][f'max_{case}'] = dict(sensors=sensors(g, bg, 'tau', g.A), **totals(g, rg, g.A, g.fill), fill_area_mm2=float(z['area_mm2'].sum()), fill_discs=int(len(z['area_mm2'])))
        log(f'{pid} max_{case} {time.time() - tt:.0f} s')
    res['seconds'] = round(time.time() - t0); (OUT / pid).mkdir(exist_ok=True); jdump(res, OUT / pid / 'plant.json'); (OUT / pid / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    log(f'{pid} done {res["seconds"]} s'); obs._CACHE.clear()
    return True


'''
s = s[:i] + NEW + s[j:]
s = s.replace("log('K5_DONE')", "log('K15_DONE')")
SRC = ['dolgfish/static-tau-bundle-20261031', 'dolgfish/static-nine-bundle-20261028', 'dolgfish/static-normals-20261104', 'dolgfish/static-normals-v2-20261104', 'dolgfish/static-lai-range-20261105']
(HERE / 'kaggle_k15').mkdir(exist_ok=True); (HERE / 'kaggle_k15/k15_lai.py').write_text(s, encoding='utf-8')
for tag, plants in (('a', 'R3-1b,n_01,n_02,n_03,n_04,n_05,n_06'), ('b', 't2_02,t2_04,t2_05,e_01,e_04,e_05,e_07')):
    d = HERE / f'kaggle_k15{tag}'; d.mkdir(exist_ok=True); (d / 'k15_lai.py').write_text(s.replace('@PLANTS@', plants), encoding='utf-8')
    json.dump({'id': f'dolgfish/static-lai-k15{tag}-20261105', 'title': f'static-lai-k15{tag}-20261105', 'code_file': 'k15_lai.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
               'enable_gpu': False, 'enable_internet': False, 'dataset_sources': SRC, 'competition_sources': [], 'kernel_sources': []}, open(d / 'kernel-metadata.json', 'w'))
print('ok')
