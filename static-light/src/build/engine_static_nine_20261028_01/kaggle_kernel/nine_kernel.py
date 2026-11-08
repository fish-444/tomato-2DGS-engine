"""Static light for the nine individuals (+ R3-1b reproduction gate), Kaggle CPU. CONTRACT.md in build/engine_static_nine_20261028_01.
Same static package as engine_static_20261026_02/static (byte copy in the dataset), v1 light model (black leaves), measured field planes,
user_assumed bars, seeds 101/202, d_ex 2.4 mm + 1.5/2/3/5 mm. Output per individual: <id>/plant_light.json, <id>/per_surfel_light.npz, <id>/DONE.
No secrets are read or written. Local smoke: python nine_kernel.py --root <bundle> --out <dir> --only R3-1b"""
import argparse, csv, glob, hashlib, json, math, os, platform, sys, time
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--root'); ap.add_argument('--out', default='/kaggle/working'); ap.add_argument('--only', default='')
a = ap.parse_args()
root = Path(a.root) if a.root else Path(glob.glob('/kaggle/input/**/light_consts.json', recursive=True)[0]).parent
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('NUMBA_CACHE_DIR', str(out / 'numba_cache'))
sys.path.insert(0, str(root))
import numpy as np
import numba
from static import obs, light as L
from static.tracer import Scene, coincident

T_START = time.time()
cfg = json.load(open(root / 'run_config.json', encoding='utf-8'))
consts = json.load(open(root / 'light_consts.json'))
L.cos_m = lambda *args, **kw: consts['cos_m']
R31 = root / 'R3-1b'
obs.R31B_PLY = R31 / 'surfels.ply'; obs.R31B_IDS = R31 / 'surfel_leaf_ids.csv'
obs.ply_path = lambda pid: obs.R31B_PLY if pid == 'R3-1b' else root / pid / 'surfels.ply'
F = obs.FieldFast.__new__(obs.FieldFast); _z = np.load(root / 'field_planes.npz')
F.planes = {70.0: (_z['xs_70'], _z['ys_70'], _z['v_70']), 150.0: (_z['xs_150'], _z['ys_150'], _z['v_150'])}
DS = [2.4, 1.5, 2.0, 3.0, 5.0]; DEX = 2.4; SEEDS = (101, 202); FD = L.DIFFUSE
ENGINE_HASH = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / 'static').glob('*.py'))}
FLAGNAMES = {1: 'unassigned', 2: 'contested', 4: 'multi_sheet_patch', 8: 'low_vote', 16: 'sam_conflict', 32: 'oversize', 64: 'low_visibility_region'}
ORIG_BARS, ORIG_DOME = L.bar_points, L.dome


def set_density(high):
    if high:
        L.bar_points = lambda seed, samples_per_bar=16, dz_mm=0.0, dy_mm=0.0: ORIG_BARS(seed, samples_per_bar, dz_mm, dy_mm)
        L.dome = lambda seed, n_mu=12, n_phi=24: ORIG_DOME(seed, n_mu, n_phi)
    else:
        L.bar_points, L.dome = ORIG_BARS, ORIG_DOME


def read_sil(pid):
    d = {}
    with open(root / pid / 'static_surfels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith('#')):
            d[int(r['surfel_row'])] = (float(r['area_b_mm2']), r['organ'])
    return d


def read_flags(pid):
    fl = defaultdict(list)
    with open(root / pid / 'flags.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['group'] != '':
                fl[int(r['group'])].append(r['flag'])
    return fl


def clicks(pid):
    state = {}
    p = root / 'user_3d_group_click.jsonl'
    if not p.exists():
        return {}
    for line in open(p, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('individual') != pid or r.get('unit') != 'group':
            continue
        if r.get('action') == 'leaf':
            state[int(r['unit_id'])] = int(r['leaf'])
        else:
            state.pop(int(r['unit_id']), None)            # any other action clears the group (none exist in the file at build time)
    return state


def trace_all(q, pid, high=False):
    set_density(high)
    res = {}
    for d in DS:
        for sd in SEEDS:
            rec = obs.trace_plants([q], [q], sd, delta=d / 1000, sensors=False, field=F)[pid]
            res[(d, sd)] = dict(K=obs.kernel(rec['parts']), Ku=obs.kernel(rec['parts'], shaded=False), U=rec['Uext'], ok=rec['ok'])
    set_density(False)
    return res


def pm(x):
    return dict(mean=float(np.mean(x)), sigma=float(abs(x[0] - x[1]) / 2), cv=float(abs(x[0] - x[1]) / math.sqrt(2) / max(abs(np.mean(x)), 1e-30)))


def totals(res, A, half=None, mask=None):
    """Per distance: absorbed (U K A), no-occluder absorbed, over the surfels in `mask`, two seeds."""
    o = {}
    m = np.ones(len(A), bool) if mask is None else mask
    for d in DS:
        ab, abu, k_ = [], [], []
        for sd in SEEDS:
            r = res[(d, sd)]; K = r['K'] if half is None else np.where(half, 0.5 * r['K'], r['K'])
            ab.append(float((r['U'] * K * A)[m].sum())); abu.append(float((r['U'] * r['Ku'] * A)[m].sum())); k_.append(float((K * A)[m].sum() / A[m].sum()))
        o[d] = dict(absorbed=pm(ab), absorbed_no_occluder=pm(abu), self_shading=pm([1 - x / y for x, y in zip(ab, abu)]), mean_K=pm(k_),
                    relative_light_per_area=pm([x / A[m].sum() for x in ab]))
    return o


def run_individual(pid):
    t0 = time.time()
    q = obs.Placed(pid, obs.SCENE_XY); n = len(q.s); rows = q.s.index
    sil = read_sil(pid)
    if set(sil) != set(rows.tolist()):
        return dict(error=f'P1 surfel_row set != opaque PLY rows ({len(sil)} vs {n})')
    A = np.array([sil[r][0] for r in rows.tolist()]); organ = np.array([sil[r][1] for r in rows.tolist()]); Acube = 1e6 * q.s.area_b
    P, k = q.prims(); coin = coincident(Scene(P, k, h=0.003), q.centre, q.normal, np.arange(n))
    res = trace_all(q, pid); high = False
    cv = totals(res, A)[DEX]['absorbed']['cv']
    if cv > 0.05 and pid != 'R3-1b':
        res = trace_all(q, pid, True); high = True; cv = totals(res, A)[DEX]['absorbed']['cv']
    z_mm = 1000 * q.centre[:, 2]; base = float(np.percentile(z_mm, 0.5)); h = z_mm - base
    tag = dict(individual=pid, relative_light=True, scale='conditional_on_unvalidated_scale', biological_validation='not_established', light_model='v1 d_ex=2.4 mm',
               lamp_bar_position='assumed', optics='assumed (v1 black leaves)')
    pl = dict(tag, surfels=n, platform=dict(system='Kaggle CPU kernel', python=platform.python_version(), numpy=np.__version__, numba=numba.__version__, cpus=os.cpu_count()),
              engine_hash=ENGINE_HASH, seeds=list(SEEDS), samples=dict(bars=3 * (16 if high else 8), dome_dirs=(12 * 24 if high else 72), analytic_tracer_no_sampled_rays=True, raised_once=high),
              distances_mm=DS, d_ex_mm=DEX)
    pl['plant_area_b_silhouette_mm2'] = float(A.sum()); pl['area_b_note'] = 'P1 silhouette-calibrated (b), conditional_on_unvalidated_scale; summed (a) not reported as area'
    pl['area_b_cube_union_mm2'] = float(Acube.sum())
    pl['height_above_base_mm'] = dict(p995=float(np.percentile(h, 99.5)), max=float(h.max()), base_z_board_mm=base)
    pl['plant_level'] = {f'{d:g}': v for d, v in totals(res, A).items()}
    pl['plant_level_half_credit_on'] = {f'{d:g}': v for d, v in totals(res, A, half=coin).items()}
    pl['coincident_1um_surfels'] = int(coin.sum())
    pl['cv_two_seed_plant_absorbed_dex'] = cv
    r0 = [res[(DEX, sd)] for sd in SEEDS]
    K = np.mean([r['K'] for r in r0], 0); U = np.mean([r['U'] for r in r0], 0); ab_s = [r['U'] * r['K'] * A for r in r0]; ab = np.mean(ab_s, 0)
    abu = np.mean([r['U'] * r['Ku'] * A for r in r0], 0); ok = r0[0]['ok']
    pl['out_of_field_support_area_fraction'] = float(A[~ok].sum() / A.sum()); pl['out_of_field_support_note'] = 'U above the measured planes = model extension (U_clamped x model ratio)'
    # height profile, 5 mm bands (height above the plant base; board z also given)
    def bands(v):
        out_ = []
        for lo in range(int(math.floor(v.min() / 5)) * 5, int(math.ceil(v.max() / 5)) * 5, 5):
            m = (v >= lo) & (v < lo + 5)
            if m.any():
                out_.append(dict(lo_mm=lo, surfels=int(m.sum()), area_b_mm2=float(A[m].sum()), absorbed=float(ab[m].sum()), mean_relative_light=float(ab[m].sum() / A[m].sum()),
                                 mean_K=float((K[m] * A[m]).sum() / A[m].sum()), self_shading=float(1 - ab[m].sum() / abu[m].sum())))
        return out_
    pl['profile_5mm_height_above_base'] = bands(h); pl['profile_5mm_board_z'] = bands(z_mm)
    # buckets
    if pid == 'R3-1b':
        lab = q.s.labels
        bk = dict(leaf_vp_cleanup=(lab >= 1) & (lab <= 8), non_leaf_vp_cleanup=~((lab >= 1) & (lab <= 8)))
        pl['bucket_note'] = 'R3-1b: V-P cleanup labels; non_leaf = stem/axis + unassigned (not comparable one to one with the nine)'
        pl['label_source'] = 'vp_cleanup_reference (reproduction gate)'
        grp = np.full(n, -1); axis = lab == -1; leaf_id = np.full(n, -1); flags = np.zeros(n, np.int16); lsrc = np.zeros(n, np.int8)
    else:
        g = np.load(root / pid / 'groups.npz', allow_pickle=False); gr = g['rows']
        order = np.argsort(gr); idx = order[np.searchsorted(gr[order], rows)]
        if not np.array_equal(gr[idx], rows):
            return dict(error='groups rows != opaque PLY rows')
        grp = g['group'][idx].astype(int); axis = g['axis_like'][idx].astype(bool); flags = g['flags'][idx].astype(np.int16)
        click = clicks(pid); leaf_id = np.array([click.get(int(x), -1) if x >= 0 else -1 for x in grp])
        lsrc = np.where(leaf_id >= 0, 2, np.where(grp >= 0, 1, 0)).astype(np.int8)
        bk = dict(stem_or_rachis_like_axis=axis, blade_in_group=(~axis) & (grp >= 0), blade_no_group=(~axis) & (grp < 0))
        pl['label_source'] = 'user_3d_group_click' if click else 'group_only'
        pl['bucket_note'] = 'unassigned bucket = axis_like + blade without a group; upper bound of what labels could still move'
    tot = float(ab.sum())
    pl['buckets'] = {k_: dict(surfels=int(m.sum()), area_b_mm2=float(A[m].sum()), absorbed=float(ab[m].sum()), share_of_absorbed=float(ab[m].sum() / tot)) for k_, m in bk.items()}
    ua = (bk['non_leaf_vp_cleanup'] if pid == 'R3-1b' else (bk['stem_or_rachis_like_axis'] | bk['blade_no_group']))
    pl['unassigned_light_share'] = float(ab[ua].sum() / tot)
    pl['p1_organ_split'] = {o: dict(surfels=int((organ == o).sum()), share_of_absorbed=float(ab[organ == o].sum() / tot)) for o in sorted(set(organ.tolist()))}
    # Tier 1
    pl['tier_reached'] = 0 if pid == 'R3-1b' else 1
    if pid != 'R3-1b':
        fl = read_flags(pid); gids = np.unique(grp[grp >= 0]); rowsT = []
        for gid in gids:
            m = grp == gid
            rowsT.append(dict(group=int(gid), surfels=int(m.sum()), area_b_mm2=float(A[m].sum()), absorbed=float(ab[m].sum()), light_per_area=float(ab[m].sum() / A[m].sum()),
                              share_of_plant=float(ab[m].sum() / tot), flags=sorted(set(fl.get(int(gid), []))), z_above_base_mm=float(np.median(h[m]))))
        rowsT.sort(key=lambda r_: -r_['absorbed'])
        pl['groups'] = dict(count=len(rowsT), flagged_groups=int(sum(1 for r_ in rowsT if r_['flags'])), flagged_share_of_absorbed=float(sum(r_['absorbed'] for r_ in rowsT if r_['flags']) / tot),
                            top10=rowsT[:10], top10_share_of_plant=float(sum(r_['absorbed'] for r_ in rowsT[:10]) / tot), name='소엽 묶음 (leaflet-level groups, not leaves)')
        if click:
            m = leaf_id == 1
            pl['user_click_leaf1_lower_bound'] = dict(groups=sorted(k_ for k_, v in click.items() if v == 1), surfels=int(m.sum()), area_b_mm2=float(A[m].sum()), absorbed=float(ab[m].sum()),
                                                      share_of_plant=float(ab[m].sum() / tot), light_per_area=float(ab[m].sum() / A[m].sum()),
                                                      note='clicked groups only; everything else stays unassigned; lower bound for leaf 1')
    cols = dict(surfel=np.arange(n), ply_row=rows, group_id=grp, leaf_id=leaf_id, label_source=lsrc, flags=flags, axis_like=axis.astype(np.int8), p1_stem_surface=(organ == 'stem_surface').astype(np.int8),
                absorbed_ru_mm2=ab, relative_light=K, U_open_ru=U, in_field_support=ok.astype(np.int8), area_b_silhouette_mm2=A, area_b_cube_mm2=Acube, coincident_1um=coin.astype(np.int8),
                x_mm=1000 * q.centre[:, 0], y_mm=1000 * q.centre[:, 1], z_mm=z_mm, h_above_base_mm=h)
    for d in DS:
        cols[f'relative_light_d{d:g}mm'] = np.mean([res[(d, sd)]['K'] for sd in SEEDS], 0)
    (out / pid).mkdir(exist_ok=True)
    np.savez_compressed(out / pid / 'per_surfel_light.npz', **cols, label_source_names=np.array(['none', 'group_only', 'user_3d_group_click']), tags=np.array(json.dumps(tag)))
    pl['seconds'] = round(time.time() - t0)
    json.dump(pl, open(out / pid / 'plant_light.json', 'w', encoding='utf-8'), indent=1, default=float)
    (out / pid / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
    return dict(ok=True, res=res, A=A, Acube=Acube, q=q, n=n, plant_level=pl['plant_level'])


# ---------------- gate 0: R3-1b reproduction
print('engine hash', ENGINE_HASH, flush=True)
g = run_individual('R3-1b')
if 'error' in g:
    json.dump(dict(external_blocked=g['error']), open(out / 'external_blocked.json', 'w')); print('BLOCKED', g); sys.exit(0)
ref2 = json.load(open(root / 'run2_ref.json')); ref = np.load(root / 'r31b_ref.npz')
lab = g['q'].s.labels; vp = (lab >= 1) & (lab <= 8)
mine5 = float(np.mean([(g['res'][(5.0, sd)]['U'] * g['res'][(5.0, sd)]['K'] * g['Acube'])[vp].sum() for sd in SEEDS]) * 1e-6)
chk = dict(plant_total_d5_cube_vpcleanup=dict(here=mine5, local=ref2['absorbed_with_extension_ru_m2']))
chk['plant_total_d5_cube_vpcleanup']['rel_diff'] = mine5 / ref2['absorbed_with_extension_ru_m2'] - 1
rows_ = g['q'].s.index; assert np.array_equal(ref['ply_row'], rows_)
for key, d in (('relative_light', 2.4), ('relative_light_d5mm', 5.0)):
    mine = np.mean([g['res'][(d, sd)]['K'] for sd in SEEDS], 0)
    chk[f'mean_{key}'] = dict(here=float(mine.mean()), local=float(ref[key].mean()), rel_diff=float(mine.mean() / ref[key].mean() - 1), max_abs_surfel_diff=float(np.abs(mine - ref[key]).max()))
chk['pass'] = all(abs(v['rel_diff']) <= 0.01 for v in chk.values() if isinstance(v, dict))
json.dump(chk, open(out / 'R3-1b' / 'reproduction_gate.json', 'w'), indent=1)
print('GATE0', json.dumps(chk), flush=True)
if not chk['pass']:
    json.dump(dict(external_blocked='R3-1b reproduction failed (>1 %)', gate=chk), open(out / 'external_blocked.json', 'w'), indent=1); print('BLOCKED'); sys.exit(0)
del g
summary = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), engine_hash=ENGINE_HASH, gate0=chk, individuals={})
only = [x for x in a.only.split(',') if x] or cfg['individuals']
for pid in only:
    if (out / pid / 'DONE').exists():
        continue
    r = run_individual(pid)
    if 'error' in r:
        summary['individuals'][pid] = dict(status='error', reason=r['error']); print(pid, 'ERROR', r['error'], flush=True); continue
    summary['individuals'][pid] = dict(status='done'); print(pid, 'done', r['n'], 'surfels', round(time.time() - T_START), 's elapsed', flush=True)
    json.dump(summary, open(out / 'SUMMARY.json', 'w'), indent=1, default=float)
summary['wall_seconds'] = round(time.time() - T_START)
json.dump(summary, open(out / 'SUMMARY.json', 'w'), indent=1, default=float)
print('NINE_DONE', summary['wall_seconds'], flush=True)
