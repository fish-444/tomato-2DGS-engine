"""STATIC runs 2–5 (CONTRACT.md §6). python runs.py <run> -> results/<run>.json"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from static import obs, light as L
OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
SEEDS = (101, 202)
FIELD = obs.FieldFast()


def strip(d):
    return {k: v for k, v in d.items() if k != 'profile'}


def single(pid, seed, footprint=1.0, delta=0.005, dz=0.0, dy=0.0, f_list=(L.DIFFUSE,)):
    q = obs.Placed(pid, obs.SCENE_XY, footprint)
    rec = obs.trace_plants([q], [q], seed, dz_mm=dz, dy_mm=dy, delta=delta, field=FIELD)[pid]
    return {f'{f:.1f}': obs.summarize(rec, pid, f_d=f) for f in f_list}


def noise_pair(a, b):
    """Two-seed mean and sigma of the mean (|a-b|/2) for every float leaf of two summaries."""
    if isinstance(a, dict):
        return {k: noise_pair(a[k], b[k]) for k in a if k in b}
    if isinstance(a, list):
        return [noise_pair(x, y) for x, y in zip(a, b)]
    if isinstance(a, float) and isinstance(b, float):
        return dict(mean=(a + b) / 2, sigma=abs(a - b) / 2)
    return a


def run2():
    res = dict(run='S-obs single', position_mm=(obs.SCENE_XY * 1000).tolist(), scale=obs.SCALE_LABEL, seeds=SEEDS, plants={})
    for pid in obs.OBSERVED:
        t = time.time()
        base = [single(pid, s, f_list=(0.5, 0.6, 0.7)) for s in SEEDS]
        sens = dict(footprint_0p8=single(pid, SEEDS[0], footprint=0.8)['0.6'], footprint_1p2=single(pid, SEEDS[0], footprint=1.2)['0.6'],
                    delta_3mm=single(pid, SEEDS[0], delta=0.003)['0.6'], delta_8mm=single(pid, SEEDS[0], delta=0.008)['0.6'],
                    diffuse_0p5=base[0]['0.5'], diffuse_0p7=base[0]['0.7'])
        res['plants'][pid] = dict(reference=noise_pair(base[0]['0.6'], base[1]['0.6']), sensitivity={k: strip(v) for k, v in sens.items()},
                                  seconds=round(time.time() - t))
        print(pid, round(time.time() - t), 's', flush=True)
        json.dump(res, open(OUT / 'run2.json', 'w', encoding='utf-8'), indent=1, default=float)
    obs._CACHE.clear()
    return res


def run3():
    """Model vs real at the same age: R3-1b, n_04 and the 59.9 DAG engine plant p1 (seed72 state 71), each alone at the reference spot."""
    from static.geometry import EngineSnapshot
    snap = EngineSnapshot(REPO / 'build/engine_maxmass_fix_20261025_01/runs/seed72/native/working', 71)
    res = dict(run='model vs real', engine_state=71, engine_dag=59.9, engine_base_m=[-0.035, 0.0, 0.0], position_mm=(obs.SCENE_XY * 1000).tolist(),
               note='model receivers: 8 per leaflet triangle; coincident mirrored-pair receivers credited 1/2 (GATES_DIAGNOSIS.md)', plants={})
    for pid in ('R3-1b', 'n_04', 'model_p1'):
        out = []
        for sd in SEEDS:
            q = obs.PlacedModel(snap, 'p1', np.array([-0.035, 0.0, 0.0]), obs.SCENE_XY) if pid == 'model_p1' else obs.Placed(pid, obs.SCENE_XY)
            rec = obs.trace_plants([q], [q], sd, field=FIELD)[q.pid]
            out.append(obs.summarize(rec, q.pid))
        res['plants'][pid] = noise_pair(out[0], out[1]); print(pid, flush=True)
    json.dump(res, open(OUT / 'run3.json', 'w', encoding='utf-8'), indent=1, default=float)


TRAY = ['n_04', 't2_02', 'n_01', 't2_05']            # assumed_layout order: (-,-), (+,-), (-,+), (+,+) around the reference spot


def tray_positions(pitch):
    h = pitch / 2
    return [obs.SCENE_XY + np.array(d) * h for d in ((-1, -1), (1, -1), (-1, 1), (1, 1))]


def tray(pitch, seed, dz=0.0, dy=0.0, alone=True):
    qs = [obs.Placed(pid, pos) for pid, pos in zip(TRAY, tray_positions(pitch))]
    allr = obs.trace_plants(qs, qs, seed, dz_mm=dz, dy_mm=dy, field=FIELD, sensors=False)
    out = {}
    for q in qs:
        d = dict(all=strip(obs.summarize(allr[q.pid], q.pid, pitch=pitch)))
        if alone:
            d['alone'] = strip(obs.summarize(obs.trace_plants([q], [q], seed, dz_mm=dz, dy_mm=dy, field=FIELD, sensors=False)[q.pid], q.pid, pitch=pitch))
            d['none'] = strip(obs.summarize(obs.trace_plants([q], [], seed, dz_mm=dz, dy_mm=dy, field=FIELD, sensors=False)[q.pid], q.pid, pitch=pitch))
            A = {k: d[k]['absorbed_with_extension_ru_m2'] for k in ('all', 'alone', 'none')}
            d['attribution'] = dict(self_shading_loss=1 - A['alone'] / A['none'], neighbour_shading_loss=1 - A['all'] / A['alone'],
                                    total_loss=1 - A['all'] / A['none'])
        out[q.pid] = d
    return out


def run4():
    res = dict(run='trays', layout='assumed_layout', order=TRAY, corners=['(-,-)', '(+,-)', '(-,+)', '(+,+)'], pitches_mm=[60, 120], seeds=SEEDS, trays={})
    for pitch in (0.060, 0.120):
        r = [tray(pitch, sd) for sd in SEEDS]
        res['trays'][f'{int(round(pitch * 1000))}'] = noise_pair(r[0], r[1]); print('tray', pitch, flush=True)
        json.dump(res, open(OUT / 'run4.json', 'w', encoding='utf-8'), indent=1, default=float)


def delta_rows(ref, alt, keys):
    """Paired Delta (same seeds) with sigma from the two-seed spread of each side; resolved if |D| > 2 sqrt(s1^2 + s2^2)."""
    out = {}
    for k in keys:
        a = [ref[i].get(k) for i in range(2)]; b = [alt[i].get(k) for i in range(2)]
        if any(v is None for v in a + b):
            continue
        ma, mb = sum(a) / 2, sum(b) / 2; sa, sb = abs(a[0] - a[1]) / 2, abs(b[0] - b[1]) / 2; sig = (sa ** 2 + sb ** 2) ** 0.5
        out[k] = dict(reference=ma, scenario=mb, delta=mb - ma, rel_delta=(mb - ma) / ma if ma else None, sigma=sig, resolved=bool(abs(mb - ma) > 2 * sig))
    return out


KEYS = ['absorbed_with_extension_ru_m2', 'absorbed_in_support_ru_m2', 'mean_incident_with_extension_ru', 'shaded_fraction_leaf', 'area_fraction_outside_support',
        'lue_gC_mol_conditional_on_meter_scale']
SCEN = {'lamp_250mm': dict(dz=-50), 'lamp_350mm': dict(dz=50), 'lateral_y_-50mm': dict(dy=-50), 'lateral_y_+50mm': dict(dy=50),
        'spacing_90mm': dict(pitch=0.090), 'spacing_120mm': dict(pitch=0.120)}


def run5():
    res = dict(run='paired scenarios', route='bar_model_ratio on the measured field (U_s = U * M_s / M_ref); out of support: model extension',
               reference='lamp 300 mm, y offset 0, tray 60 mm', tray_layout='assumed_layout', seeds=SEEDS, tray={}, r31b_single={})
    ref = [tray(0.060, sd, alone=False) for sd in SEEDS]
    for name, sp in SCEN.items():
        alt = [tray(sp.get('pitch', 0.060), sd, dz=sp.get('dz', 0.0), dy=sp.get('dy', 0.0), alone=False) for sd in SEEDS]
        res['tray'][name] = {pid: delta_rows([ref[i][pid]['all'] for i in range(2)], [alt[i][pid]['all'] for i in range(2)], KEYS) for pid in TRAY}
        print(name, flush=True)
        json.dump(res, open(OUT / 'run5.json', 'w', encoding='utf-8'), indent=1, default=float)

    def r31(sd, dz=0.0, dy=0.0):
        q = obs.Placed('R3-1b', obs.SCENE_XY)
        return obs.summarize(obs.trace_plants([q], [q], sd, dz_mm=dz, dy_mm=dy, field=FIELD, sensors=False)['R3-1b'], 'R3-1b')
    base = [r31(sd) for sd in SEEDS]
    for name, sp in SCEN.items():
        if 'pitch' in sp:
            continue      # spacing needs a tray; R3-1b is not a tray plant
        alt = [r31(sd, sp.get('dz', 0.0), sp.get('dy', 0.0)) for sd in SEEDS]
        d = dict(plant=delta_rows(base, alt, KEYS))
        for leaf in ('leaf_1', 'leaf_7'):
            d[leaf] = delta_rows([b['per_leaf'][leaf] for b in base], [a['per_leaf'][leaf] for a in alt],
                                 ['absorbed_with_extension_ru_m2', 'mean_incident_ru', 'shaded_fraction', 'lue_gC_mol_conditional_on_meter_scale'])
        res['r31b_single'][name] = d
    json.dump(res, open(OUT / 'run5.json', 'w', encoding='utf-8'), indent=1, default=float)


if __name__ == '__main__':
    {'run2': run2, 'run3': run3, 'run4': run4, 'run5': run5}[sys.argv[1]]()
