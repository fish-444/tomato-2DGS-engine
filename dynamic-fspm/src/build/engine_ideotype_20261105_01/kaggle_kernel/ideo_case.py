"""One ideotype engine run on Kaggle CPU (build/engine_ideotype_20261105_01/CONTRACT.md).
python ideo_case.py <prepared case dir> <core.jar> <out dir> <evcode dir>
v2_config.json present -> ideotype validator (= Engine v2 validator + LS/PE replay); absent (frozen Engine v2 reference case) -> v2 validator from its own
config. Writes <out>/<case>/status.json (with completion.json output hashes for gates G0/G1) + summary.json (per plant per day), geometry_day71.npz
for replicate-1 cases at s10h40 and G0. Working files are deleted afterwards."""
import csv, json, math, os, shutil, sys, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np

case_src, core, out, evcode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
name = case_src.name; dst = out / name; dst.mkdir(parents=True, exist_ok=True); LAMP = 1200.0
status = dict(case=name, attempts=[], started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
cfg = json.load(open(case_src / 'v2_config.json'))
sys.path.insert(0, str(evcode))
if 'n_plants' in cfg:
    import ideo_validate; status['validator'] = ideo_validate.install(cfg)
else:
    import v2_validate; status['validator'] = v2_validate.install(cfg)
from tools.vtc_growth_bridge import run

ok = False
for attempt in (1, 2):
    work = Path('/tmp/runs') / f'{name}_a{attempt}_{int(time.time())}'; t0 = time.time()
    try:
        shutil.copytree(case_src, work); r = run(work / 'native', core=core, timeout_s=10 * 3600)
        rec = dict(attempt=attempt, status=r.get('status'), error=r.get('error'), elapsed_s=round(time.time() - t0))
    except Exception as e:
        rec = dict(attempt=attempt, status='exception', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-4000:], elapsed_s=round(time.time() - t0))
    status['attempts'].append(rec); print(name, json.dumps({k: v for k, v in rec.items() if k != 'trace'}), flush=True)
    json.dump(status, open(dst / 'status.json', 'w'), indent=1)
    if rec['status'] == 'verified':
        ok = True; break
    if attempt == 1:
        try:
            W = work / 'native' / 'working'; keep = dst / 'failed_attempt1'; keep.mkdir(exist_ok=True)
            for fn in ('events.csv', 'random_draws.csv', 'ledger.csv'):
                if (W / fn).exists(): shutil.copy2(W / fn, keep / fn)
            for fn in ('execution.json', 'groimp.log'):
                if (work / 'native' / fn).exists(): shutil.copy2(work / 'native' / fn, keep / fn)
        except Exception as e:
            status['keep_error'] = str(e)
    shutil.rmtree(work, ignore_errors=True)
status['status'] = 'verified' if ok else 'failed'
if not ok:
    json.dump(status, open(dst / 'status.json', 'w'), indent=1); sys.exit(0)

W = work / 'native' / 'working'
try:
    import pandas as pd
    status['completion_files'] = json.load(open(W / 'completion.json'))['files']
    case = cfg.get('case', {}); lum = cfg.get('luminaire'); z0 = float(case.get('z0', 0.0))
    st = pd.read_csv(W / 'states.csv'); fx = pd.read_csv(W / 'flux.csv', usecols=['interval', 'plant_instance_id', 'organ_id', 'absorbed_umol_s'])
    led = pd.read_csv(W / 'ledger.csv'); ev = pd.read_csv(W / 'events.csv'); draws = pd.read_csv(W / 'random_draws.csv')
    cls = dict(zip(st['organ_id'], st['class'])); fx['cls'] = fx['organ_id'].map(cls)
    ab = fx[fx['cls'] == 'Leaflet'].groupby(['interval', 'plant_instance_id'])['absorbed_umol_s'].sum()
    zmax = defaultdict(float)
    for ch in pd.read_csv(W / 'meshes.csv', usecols=['state', 'organ_id', 'z_m'], chunksize=2_000_000):
        ch['pid'] = ch['organ_id'].str.split('/').str[0]
        for (s, p), v in ch.groupby(['state', 'pid'])['z_m'].max().items(): zmax[(int(s), p)] = max(zmax[(int(s), p)], float(v))
    GROUP = {'Leaflet': 'leaf', 'Internode': 'stem', 'Petiole': 'petiole_rachis', 'RachisSegment': 'petiole_rachis', 'Root': 'root', 'Fruit': 'fruit', 'Flower': 'flower', 'TrussAxis': 'truss'}
    L = led.set_index(['interval', 'plant_instance_id']); daily = defaultdict(list)
    for (d, p), sd in st.groupby(['day', 'plant_instance_id']):
        d = int(d)
        row = dict(day=d, leaf_area_mm2=float(sd.loc[sd['class'] == 'Leaflet', 'area_m2'].sum() * 1e6), dm_g=float(sd['mass_gDM'].sum()),
                   dm_by_group={k: float(v) for k, v in sd.assign(g=sd['class'].map(GROUP).fillna('other')).groupby('g')['mass_gDM'].sum().items()},
                   surface_mm2={k: float((g['length_m'] * g['width_m']).sum() * 1e6) for k, g in sd[sd['class'].isin(['Internode', 'Petiole', 'RachisSegment'])].groupby('class')},
                   height_mm=(zmax.get((d, p), z0) - z0) * 1000, top_z_mm=zmax.get((d, p), 0.0) * 1000,
                   compounds=int((sd['class'] == 'CompoundLeaf').sum()), trusses=int((sd['class'] == 'TrussAxis').sum()), fruits=int((sd['class'] == 'Fruit').sum()),
                   absorbed_rel=float(ab.get((d - 1, p), 0.0) / LAMP) if d > 0 else 0.0)
        if d > 0 and (d - 1, p) in L.index:
            r = L.loc[(d - 1, p)]; row.update(gross_gC=float(r['gross_gC']), downregulated_gC=float(r['downregulated_gC']), reserve_end_gC=float(r['reserve_end_gC']),
                                              structural_gC=float(r['structural_gC']), maintenance_gC=float(r['maintenance_gC']))
        daily[p].append(row)
    clearance = {}
    if lum:
        for d in sorted({k[0] for k in zmax}):
            top = max(v for (s, _), v in zmax.items() if s == d); clearance[d] = (lum['z'] - top) * 1000
    S = dict(case=name, cfg=dict(case=case, ideo=cfg.get('ideo'), luminaire=lum), daily=dict(daily), clearance_mm=clearance,
             draws=[dict(id=r.id, key=r.key, u=float(r.u), value=float(r.value)) for r in draws.itertuples() if r.key in ('INTERNODE_MM', 'MAIN_TRUSS_NODE', 'MAIN_TOTAL_LEAVES', 'SIDE_TOTAL_LEAVES', 'SIDE_ATTACH_RANK', 'SLA_FACTOR')],
             max_mass_ratio=float((st['mass_gDM'] / st['max_mass_gDM'].where(st['max_mass_gDM'] > 0)).max()),
             stops=[dict(day=int(r.day), organ=r.organ_id) for r in ev[ev['event'] == 'determinate_stop'].itertuples()])
    json.dump(S, open(dst / 'summary.json', 'w'))
    if ('_s10h40_r1' in name) or name.endswith('_G0_r1'):
        d71 = int(st['day'].max()); tri, own = [], []
        for ch in pd.read_csv(W / 'meshes.csv', usecols=['state', 'organ_id', 'vertex', 'x_m', 'y_m', 'z_m'], chunksize=2_000_000):
            ch = ch[ch['state'] == d71]
            for oid, g in ch.sort_values(['organ_id', 'vertex']).groupby('organ_id', sort=False):
                a = g[['x_m', 'y_m', 'z_m']].to_numpy(float); k = len(a) - len(a) % 3
                if k: tri.append(a[:k].reshape(-1, 3, 3)); own += [oid] * (k // 3)
        np.savez_compressed(dst / 'geometry_day71.npz', tri=np.concatenate(tri).astype(np.float32), owner=np.array(own), cls=np.array([cls.get(o, '') for o in own]))
    status['summary'] = 'written'
except Exception as e:
    status['summary'] = f'error {type(e).__name__}: {e}'; status['summary_trace'] = traceback.format_exc()[-4000:]
shutil.rmtree(work, ignore_errors=True)
status['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(status, open(dst / 'status.json', 'w'), indent=1); print(name, status['status'], status.get('summary'), flush=True)
