"""One Engine v2 run on Kaggle CPU (build/engine_v2_calib_20261104_01/CONTRACT.md).
python ev_case.py <prepared case dir> <core.jar> <out dir> <evcode dir>
v2_config.json present (C1+) -> v2 validator installed before the bridge run; B0 -> frozen validator. One retry from a fresh copy.
Writes <out>/<case>/status.json + summary.json (daily per-plant series, compound table day 71, leaflet stats, events, draws, ledger) and, for m00
and C2T, geometry_day71.npz (all organ triangles). Working files are deleted afterwards."""
import csv, json, math, os, shutil, sys, time, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np

case_src, core, out, evcode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
name = case_src.name; dst = out / name; dst.mkdir(parents=True, exist_ok=True)
LAMP = 1200.0
status = dict(case=name, attempts=[], started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
cfgp = case_src / 'v2_config.json'; cfg = json.load(open(cfgp)) if cfgp.exists() else None
sys.path.insert(0, str(evcode))
if cfg is not None:
    import v2_validate
    status['v2_validator'] = v2_validate.install(cfg)
from tools.vtc_growth_bridge import run

ok = False
for attempt in (1, 2):
    work = Path('/tmp/runs') / f'{name}_a{attempt}_{int(time.time())}'; t0 = time.time()
    try:
        shutil.copytree(case_src, work)
        r = run(work / 'native', core=core, timeout_s=8 * 3600)
        rec = dict(attempt=attempt, status=r.get('status'), error=r.get('error'), elapsed_s=round(time.time() - t0))
    except Exception as e:
        rec = dict(attempt=attempt, status='exception', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-4000:], elapsed_s=round(time.time() - t0))
    status['attempts'].append(rec); print(name, json.dumps({k: v for k, v in rec.items() if k != 'trace'}), flush=True)
    json.dump(status, open(dst / 'status.json', 'w'), indent=1)
    if rec['status'] == 'verified':
        ok = True; break
    if attempt == 1:
        try:   # keep the failing run's small files for diagnosis (no geometry)
            W = work / 'native' / 'working'; keep = dst / 'failed_attempt1'; keep.mkdir(exist_ok=True)
            for fn in ('events.csv', 'random_draws.csv', 'ledger.csv', 'apices.csv'):
                if (W / fn).exists(): shutil.copy2(W / fn, keep / fn)
            for fn in ('execution.json', 'groimp.log'):
                if (work / 'native' / fn).exists(): shutil.copy2(work / 'native' / fn, keep / fn)
            if (W / 'states.csv').exists() and (W / 'states.csv').stat().st_size < 400_000_000: shutil.copy2(W / 'states.csv', keep / 'states.csv')
        except Exception as e:
            status['keep_error'] = str(e)
    shutil.rmtree(work, ignore_errors=True)
status['status'] = 'verified' if ok else 'failed'
if not ok:
    json.dump(status, open(dst / 'status.json', 'w'), indent=1); sys.exit(0)

W = work / 'native' / 'working'
try:
    import pandas as pd
    if cfg is not None and '_schedule' in v2_validate.CFG: status['v2_schedule'] = v2_validate.CFG['_schedule']
    st = pd.read_csv(W / 'states.csv')
    fx = pd.read_csv(W / 'flux.csv', usecols=['interval', 'plant_instance_id', 'organ_id', 'absorbed_umol_s'])
    led = pd.read_csv(W / 'ledger.csv'); ev = pd.read_csv(W / 'events.csv'); draws = pd.read_csv(W / 'random_draws.csv')
    ctl = lambda oid: (oid.split('/')[1] if len(oid.split('/')) > 1 and oid.split('/')[1][:1] == 'S' and oid.split('/')[1][1:].isdigit() else 'M')
    st['ctl'] = st['organ_id'].map(ctl)
    cls = dict(zip(st['organ_id'], st['class'])); fx['cls'] = fx['organ_id'].map(cls)
    absorbed = fx[fx['cls'] == 'Leaflet'].groupby('interval')['absorbed_umol_s'].sum()
    zmax = defaultdict(float); zmin = {}
    for ch in pd.read_csv(W / 'meshes.csv', usecols=['state', 'organ_id', 'z_m'], chunksize=2_000_000):
        g = ch.groupby('state')['z_m']
        for s, v in g.max().items(): zmax[int(s)] = max(zmax[int(s)], float(v))
    daily = []
    for d, sd in st.groupby('day'):
        lf = sd[sd['class'] == 'Leaflet']; cl = sd[sd['class'] == 'CompoundLeaf']; tr = sd[sd['class'] == 'TrussAxis']
        row = dict(day=int(d), leaf_area_mm2=float(lf['area_m2'].sum() * 1e6), dm_g=float(sd['mass_gDM'].sum()), height_mm=zmax.get(int(d), 0.0) * 1000,
                   dm_by_class={k: float(v) for k, v in sd.groupby('class')['mass_gDM'].sum().items()},
                   compounds={k: int(v) for k, v in cl.groupby('ctl').size().items()}, compounds_A0=int((cl['axis_id'] == 'p1/A0').sum()),
                   trusses={k: int(v) for k, v in tr.groupby('ctl').size().items()}, flowers=int((sd['class'] == 'Flower').sum()), fruits=int((sd['class'] == 'Fruit').sum()),
                   leaflets=int(len(lf)), absorbed_rel=float(absorbed.get(int(d) - 1, 0.0) / LAMP) if d > 0 else 0.0)
        daily.append(row)
    L = led.set_index('interval')
    for row in daily:
        i = row['day'] - 1
        if i in L.index:
            r = L.loc[i]; row.update(gross_gC=float(r['gross_gC']), downregulated_gC=float(r['downregulated_gC']), reserve_end_gC=float(r['reserve_end_gC']),
                                     reserve_start_gC=float(r['reserve_start_gC']), maintenance_gC=float(r['maintenance_gC']), structural_gC=float(r['structural_gC']))
    s71 = st[st['day'] == st['day'].max()]; by = {r.organ_id: r for r in s71.itertuples()}
    kids = defaultdict(list)
    for r in s71.itertuples(): kids[r.parent_id].append(r)
    comp = []
    for c in s71[s71['class'] == 'CompoundLeaf'].itertuples():
        desc = []; todo = list(kids[c.organ_id])
        while todo:
            x = todo.pop(); desc.append(x); todo.extend(kids[x.organ_id])
        bl =[x for x in desc if cls.get(x.organ_id) == 'Leaflet']; ra = [x for x in desc if cls.get(x.organ_id) == 'RachisSegment']; pe = [x for x in desc if cls.get(x.organ_id) == 'Petiole']
        pel = None
        if pe:
            p = pe[0]; dx, dy, dz = p.tip_x_m - p.x_m, p.tip_y_m - p.y_m, p.tip_z_m - p.z_m; hh = math.hypot(dx, dy)
            pel = math.degrees(math.atan2(dz, hh)) if (hh > 0 or dz != 0) else None
        grown = [x for x in bl if x.area_m2 > 0]
        comp.append(dict(id=c.organ_id, ctl=ctl(c.organ_id), axis=c.axis_id, rank=int(c.true_leaf_order), birth_day=int(c.birth_day), azimuth=float(c.azimuth_deg),
                         elevation=float(c.elevation_deg), petiole_elev=pel, leaflets=len(bl), rachis=len(ra), bend_sum=float(sum(x.rachis_bend_alpha_deg for x in ra)),
                         area_mm2=float(sum(x.area_m2 for x in bl) * 1e6), max_leaflet_len_mm=float(max([x.length_m for x in grown], default=0) * 1000),
                         lw=[float(x.length_m / x.width_m) for x in grown if x.width_m > 0]))
    ints = [dict(id=r.organ_id, ctl=ctl(r.organ_id), len_mm=float(r.length_m * 1000), mass=float(r.mass_gDM), max_mass=float(r.max_mass_gDM)) for r in s71[s71['class'] == 'Internode'].itertuples()]
    evs = ev[ev['event'].isin(['determinate_stop', 'axis_terminated', 'sympodial_continuation', 'anthesis_fruit_set'])].copy()
    tr_init = ev[(ev['event'] == 'initiation') & ev['organ_id'].str.contains('/Truss') & ~ev['organ_id'].str.contains('/F')]
    side_init = ev[(ev['event'] == 'initiation') & ev['organ_id'].str.match(r'^[^/]+/S[0-9]+/A0$')]
    S = dict(case=name, cfg=dict(level=cfg['level'], sla=cfg['sla_m2_g'], test=cfg.get('test')) if cfg else dict(level='B0'), daily=daily, compounds=comp, internodes=ints,
             events=[dict(day=int(r.day), organ=r.organ_id, event=r.event, rule=r.rule) for r in evs.itertuples()],
             truss_init=[dict(day=int(r.day), organ=r.organ_id) for r in tr_init.itertuples()],
             side_axes=[dict(day=int(r.day), organ=r.organ_id, parent=r.parent_id) for r in side_init.itertuples()],
             draws=[dict(id=r.id, key=r.key, u=float(r.u), value=float(r.value)) for r in draws.itertuples()],
             max_mass_ratio=float((st['mass_gDM'] / st['max_mass_gDM'].where(st['max_mass_gDM'] > 0)).max()))
    # C4L measures (CONTRACT Amendment 8), day 71, compound leaves with >= 5 leaflets: measured spread angle per lateral leaflet
    # (angle between its long axis and the tangent of the nearest chain organ) and within-leaf top-view overlap (raster 0.05 mm).
    try:
        d71 = int(st['day'].max()); s71b = st[st['day'] == d71]; rows = {r.organ_id: r for r in s71b.itertuples()}; cls71 = dict(zip(s71b['organ_id'], s71b['class']))
        def comp_of(oid):
            c = rows[oid].parent_id
            while cls71.get(c) != 'CompoundLeaf': c = rows[c].parent_id
            return c
        chains = defaultdict(list); leafl = defaultdict(list)
        for oid, k in cls71.items():
            if k in ('Petiole', 'RachisSegment'): chains[comp_of(oid)].append(rows[oid])
            if k == 'Leaflet' and rows[oid].area_m2 > 0: leafl[comp_of(oid)].append(rows[oid])
        want = {r.organ_id for v in leafl.values() for r in v}; tris = defaultdict(list)
        parts = [ch[(ch['state'] == d71) & ch['organ_id'].isin(want)] for ch in pd.read_csv(W / 'meshes.csv', usecols=['state', 'organ_id', 'vertex', 'x_m', 'y_m', 'z_m'], chunksize=2_000_000)]
        allm = pd.concat(parts) if parts else pd.DataFrame(columns=['organ_id', 'vertex', 'x_m', 'y_m', 'z_m'])
        for oid, g in allm.sort_values(['organ_id', 'vertex']).groupby('organ_id', sort=False):
            a = g[['x_m', 'y_m', 'z_m']].to_numpy(float); k = len(a) - len(a) % 3; tris[oid].append(a[:k].reshape(-1, 3, 3))
        def unit(az, el):
            az, el = math.radians(az), math.radians(el); return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        def seg_dist(p, a, b):
            d = b - a; L2 = float(d @ d); t = 0.0 if L2 == 0 else max(0.0, min(1.0, float((p - a) @ d) / L2)); return float(np.linalg.norm(a + t * d - p))
        RES = 5e-5; c4lm = []
        for c, ls in leafl.items():
            n_all = sum(1 for oid, k in cls71.items() if k == 'Leaflet' and comp_of(oid) == c)
            if n_all < 5 or not chains[c]: continue
            phis = []
            for r in ls:
                if int(r.side) == 0: continue
                p = np.array([r.x_m, r.y_m, r.z_m])
                q = min(chains[c], key=lambda o: seg_dist(p, np.array([o.x_m, o.y_m, o.z_m]), np.array([o.tip_x_m, o.tip_y_m, o.tip_z_m])))
                cosang = float(np.clip(unit(r.azimuth_deg, r.elevation_deg) @ unit(q.azimuth_deg, q.elevation_deg), -1, 1))
                phis.append(dict(pos=f'P{min(int(r.pair_index), 3)}', side=int(r.side), phi=math.degrees(math.acos(cosang))))
            # span (MTG_LEAFLET_SPAN_v1 definition): max - min lateral offset of the leaflet vertices from the chain polyline (node -> end-leaflet base),
            # measured in the compound plane (PCA of leaflet vertices + chain points) perpendicular to the local chain tangent; ÷ chain length L_r
            span_ratio = None
            try:
                chs = sorted(chains[c], key=lambda o: (0, 0) if cls71.get(o.organ_id) == 'Petiole' else (1, int(o.pair_index)))
                poly = np.array([[chs[0].x_m, chs[0].y_m, chs[0].z_m]] + [[o.tip_x_m, o.tip_y_m, o.tip_z_m] for o in chs])
                Lr = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
                V = np.concatenate([np.concatenate(tris[r.organ_id]).reshape(-1, 3) for r in ls if tris.get(r.organ_id)])
                allpts = np.vstack([V, poly]); w_, E_ = np.linalg.eigh(np.cov((allpts - allpts.mean(0)).T)); nrm = E_[:, 0]
                offs = []
                for p in V:
                    best = None
                    for a, b in zip(poly[:-1], poly[1:]):
                        d = b - a; L2 = float(d @ d)
                        if L2 == 0: continue
                        t = max(0.0, min(1.0, float((p - a) @ d) / L2)); q = a + t * d; dist = float(np.linalg.norm(p - q))
                        if best is None or dist < best[0]: best = (dist, q, d / math.sqrt(L2))
                    if best is None: continue
                    lat = np.cross(nrm, best[2]); ln = np.linalg.norm(lat)
                    if ln == 0: continue
                    offs.append(float((p - best[1]) @ (lat / ln)))
                if offs and Lr > 0: span_ratio = (max(offs) - min(offs)) / Lr
            except Exception:
                span_ratio = None
            T = {r.organ_id: np.concatenate(tris[r.organ_id])[:, :, :2] for r in ls if tris.get(r.organ_id)}
            if len(T) < 2: continue
            allp = np.concatenate(list(T.values())).reshape(-1, 2); lo = allp.min(0); hi = allp.max(0); nx, ny = (np.ceil((hi - lo) / RES).astype(int) + 1)
            if nx * ny > 4_000_000: continue
            gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + .5) * RES, lo[1] + (np.arange(ny) + .5) * RES, indexing='ij'); count = np.zeros((nx, ny), np.int16); own = 0
            for oid, tr in T.items():
                m = np.zeros((nx, ny), bool)
                for a, b, cc in tr:
                    x0, y0 = np.floor((np.minimum(np.minimum(a, b), cc) - lo) / RES).astype(int); x1, y1 = np.ceil((np.maximum(np.maximum(a, b), cc) - lo) / RES).astype(int) + 1
                    X = gx[x0:x1, y0:y1]; Y = gy[x0:x1, y0:y1]
                    d = (b[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - b[0]) * (a[1] - cc[1])
                    if abs(d) < 1e-18: continue
                    l1 = ((b[1] - cc[1]) * (X - cc[0]) + (cc[0] - b[0]) * (Y - cc[1])) / d; l2 = ((cc[1] - a[1]) * (X - cc[0]) + (a[0] - cc[0]) * (Y - cc[1])) / d
                    m[x0:x1, y0:y1] |= (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
                own += int(m.sum()); count += m
            union = int((count > 0).sum())
            c4lm.append(dict(compound=c, n_leaflets=n_all, overlap=(own - union) / own if own else None, phis=phis, span_ratio=span_ratio))
        S['c4l_measures'] = c4lm
        # C4B measures (CONTRACT Amendment 11), per leaflet of compounds with >= 5 leaflets: normal deviation from the leaflet mean normal
        # (exact triangle normals, area-weighted; and surfels sampled 20 / mm2 with position noise 0.124 mm, PCA normals within r = 2 mm and
        # within r scaled by L / L_ref), top-view projected-area ratio (raster).
        LREF = {'P1': 17.0926, 'P2': 14.5765, 'P3': 14.3016, 'T': 19.4931}

        def wq(x, w, q):
            o = np.argsort(x); x = x[o]; c = np.cumsum(w[o]); return float(x[np.searchsorted(c, q * c[-1])])

        def proj_area(t2):
            lo2 = t2.reshape(-1, 2).min(0); hi2 = t2.reshape(-1, 2).max(0); res = max(float((hi2 - lo2).max()) / 200, 1e-6)
            nx2, ny2 = (np.ceil((hi2 - lo2) / res).astype(int) + 1)
            gx2, gy2 = np.meshgrid(lo2[0] + (np.arange(nx2) + .5) * res, lo2[1] + (np.arange(ny2) + .5) * res, indexing='ij'); m = np.zeros((nx2, ny2), bool)
            for a, b, cc in t2:
                d = (b[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - b[0]) * (a[1] - cc[1])
                if abs(d) < 1e-24: continue
                l1 = ((b[1] - cc[1]) * (gx2 - cc[0]) + (cc[0] - b[0]) * (gy2 - cc[1])) / d; l2 = ((cc[1] - a[1]) * (gx2 - cc[0]) + (a[0] - cc[0]) * (gy2 - cc[1])) / d
                m |= (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
            return float(m.sum()) * res * res

        cm = []
        for c, ls in leafl.items():
            n_all = sum(1 for oid, k in cls71.items() if k == 'Leaflet' and comp_of(oid) == c)
            if n_all < 5: continue
            for r in ls:
                if not tris.get(r.organ_id): continue
                T = np.concatenate(tris[r.organ_id]) * 1000.0
                cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); a = 0.5 * np.linalg.norm(cr, axis=1); A = float(a.sum())
                if A <= 0: continue
                nrm = cr / np.maximum(2 * a[:, None], 1e-30); mean = (nrm * a[:, None]).sum(0); mean = mean / np.linalg.norm(mean)
                dev = np.degrees(np.arccos(np.clip(np.abs(nrm @ mean), 0, 1)))
                pos = 'T' if int(r.side) == 0 else f'P{min(int(r.pair_index), 3)}'; L = float(r.length_m) * 1000
                rec = dict(pos=pos, area_mm2=A, length_mm=L, ndev_exact_med=wq(dev, a, .5), ndev_exact_p90=wq(dev, a, .9), proj_ratio=proj_area(T[:, :, :2]) / A)
                rng = np.random.default_rng(__import__("zlib").crc32(r.organ_id.encode())); npt = max(30, int(20 * A))
                ti = rng.choice(len(T), size=npt, p=a / A); r1 = np.sqrt(rng.random(npt)); r2 = rng.random(npt)
                P = (1 - r1)[:, None] * T[ti, 0] + (r1 * (1 - r2))[:, None] * T[ti, 1] + (r1 * r2)[:, None] * T[ti, 2] + rng.normal(0, 0.124, (npt, 3))
                D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
                for tag, rad in (('r2', 2.0), ('rscaled', 2.0 * L / LREF[pos] if L > 0 else 2.0)):
                    ns = []
                    for i in range(npt):
                        nb = P[D[i] <= rad]
                        if len(nb) < 3: continue
                        w_, V_ = np.linalg.eigh(np.cov((nb - nb.mean(0)).T)); ns.append(V_[:, 0])
                    if len(ns) < 3: continue
                    ns = np.array(ns); mn = ns[0].copy()
                    for _ in range(3):
                        sg = np.sign(ns @ mn); sg[sg == 0] = 1; mn = (ns * sg[:, None]).mean(0); mn /= np.linalg.norm(mn)
                    dv = np.degrees(np.arccos(np.clip(np.abs(ns @ mn), 0, 1)))
                    rec[f'ndev_{tag}_med'] = float(np.median(dv)); rec[f'ndev_{tag}_p90'] = float(np.percentile(dv, 90)); rec[f'radius_{tag}_mm'] = rad
                cm.append(rec)
        S['c4b_measures'] = cm
    except Exception as e:
        S['c4l_measures_error'] = f'{type(e).__name__}: {e}'
    # C4C measures (CONTRACT Amendment 14), day 71, every compound: chain length (petiole + rachis), leaflet max-mass relative range, leaflet area range / mean
    try:
        d71 = int(st['day'].max()); s71 = st[st['day'] == d71]; par = dict(zip(s71['organ_id'], s71['parent_id'])); kl = dict(zip(s71['organ_id'], s71['class']))
        def comp_c(oid):
            c = par[oid]
            while kl.get(c) != 'CompoundLeaf': c = par[c]
            return c
        cc = defaultdict(lambda: dict(chain_m=0.0, mm=[], area=[]))
        for r in s71.itertuples():
            if r.organ_id not in kl or kl[r.organ_id] not in ('Petiole', 'RachisSegment', 'Leaflet'): continue
            c = cc[comp_c(r.organ_id)]
            if kl[r.organ_id] == 'Leaflet': c['mm'].append(float(r.max_mass_gDM)); c['area'].append(float(r.area_m2))
            else: c['chain_m'] += float(r.length_m)
        S['c4c_measures'] = [dict(compound=k, chain_mm=v['chain_m'] * 1000, n_leaflets=len(v['mm']),
                                  maxmass_rel_range=(max(v['mm']) - min(v['mm'])) / max(v['mm']) if v['mm'] and max(v['mm']) > 0 else 0.0,
                                  area_range_over_mean=(max(v['area']) - min(v['area'])) / (sum(v['area']) / len(v['area'])) if v['area'] and sum(v['area']) > 0 else 0.0)
                             for k, v in cc.items()]
    except Exception as e:
        S['c4c_measures_error'] = f'{type(e).__name__}: {e}'
    # C4A Step A diagnosis (CONTRACT Amendment 17): per day leaflet / plant satisfaction (allocated / demand), ledger reserve and down-regulation, day-71 leaflet mass / max
    try:
        fd = pd.read_csv(W / 'flux.csv', usecols=['interval', 'organ_id', 'demand_gC', 'allocated_gC'])
        fd['cls'] = fd['organ_id'].map(cls)
        gl = fd[fd['cls'] == 'Leaflet'].groupby('interval')[['demand_gC', 'allocated_gC']].sum(); ga = fd.groupby('interval')[['demand_gC', 'allocated_gC']].sum()
        lg = led.set_index('interval')
        diag = []
        for i in ga.index:
            r = dict(interval=int(i), leaf_demand=float(gl.loc[i, 'demand_gC']) if i in gl.index else 0.0, leaf_alloc=float(gl.loc[i, 'allocated_gC']) if i in gl.index else 0.0,
                     plant_demand=float(ga.loc[i, 'demand_gC']), plant_alloc=float(ga.loc[i, 'allocated_gC']))
            if i in lg.index:
                for k in ('reserve_start_gC', 'reserve_end_gC', 'gross_gC', 'deficit_gC', 'downregulated_gC'):
                    if k in lg.columns: r[k] = float(lg.loc[i, k])
            diag.append(r)
        d71 = int(st['day'].max()); lf71 = st[(st['day'] == d71) & (st['class'] == 'Leaflet') & (st['max_mass_gDM'] > 0)]
        S['c4a_diag'] = dict(daily=diag, leaflet_mass_over_max_day71=[float(x) for x in (lf71['mass_gDM'] / lf71['max_mass_gDM'])])
    except Exception as e:
        S['c4a_diag_error'] = f'{type(e).__name__}: {e}'
    json.dump(S, open(dst / 'summary.json', 'w'))
    if name.endswith('_m00') or name.startswith('C2T'):
        d71 = int(st['day'].max()); tri, own = [], []
        for ch in pd.read_csv(W / 'meshes.csv', usecols=['state', 'organ_id', 'vertex', 'x_m', 'y_m', 'z_m'], chunksize=2_000_000):
            ch = ch[ch['state'] == d71]
            for oid, g in ch.sort_values(['organ_id', 'vertex']).groupby('organ_id', sort=False):
                a = g[['x_m', 'y_m', 'z_m']].to_numpy(float); k = len(a) - len(a) % 3
                if k: tri.append(a[:k].reshape(-1, 3, 3)); own += [oid] * (k // 3)
        np.savez_compressed(dst / 'geometry_day71.npz', tri=np.concatenate(tri), owner=np.array(own), cls=np.array([cls.get(o, '') for o in own]))
    status['summary'] = 'written'
except Exception as e:
    status['summary'] = f'error {type(e).__name__}: {e}'; status['summary_trace'] = traceback.format_exc()[-4000:]
shutil.rmtree(work, ignore_errors=True)
status['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(status, open(dst / 'status.json', 'w'), indent=1); print(name, status['status'], status.get('summary'), flush=True)
