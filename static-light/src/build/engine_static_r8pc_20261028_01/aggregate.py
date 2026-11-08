"""Re-aggregate the per-surfel static trace (case (a)) under V-P cleanup (reference), R7 and R8_pc(+rachis) labels. CONTRACT.md.
python aggregate.py <trace_out.npz> -> per_surfel_light.npz, per_leaf.csv, buckets.csv, CHECKS.json, TABLES.md; also fills S2px
(engine_static_20261026_02/results/s2px_R3-1b.{npz,csv,_meta.json}, accepted-run semantics: d 5 mm, cube-union area, V-P cleanup labels)."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
TAGS = 'relative_light · conditional_on_unvalidated_scale · labels=R8_pc designated_not_adopted (gate failed) · light_model=v1 d_ex=2.4 mm · biological_validation = not_established'
DS = [1.5, 2.0, 2.4, 3.0, 5.0]; DEX = 2.4; SEEDS = (101, 202); FD = 0.6; MEDIUM_Z = 0.060
T = np.load(sys.argv[1]); rows = T['ply_row']; n = len(rows)


def by_rows(path, key):
    z = np.load(path); m = dict(zip(z['rows'].tolist(), z[key].tolist()))
    return np.array([m[r] for r in rows.tolist()])


R7 = by_rows(REPO / 'build/p1_local_diffusion_20261027_01/vp_leaf4/labels.npz', 'label')
RCH = REPO / 'build/p1_pc_absorb_20261027_01/R8_pc_rachis/labels_rachis.npz'
R8 = by_rows(RCH, 'label'); AX = by_rows(RCH, 'axis_kind'); R8SRC = by_rows(RCH, 'source')
VP = T['labels_vpcleanup']
sil = {}
with open(REPO / 'build/p1_plant_model_20261026_01/plants/R3-1b/static_surfels.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(l for l in fh if not l.startswith('#')):
        sil[int(r['surfel_row'])] = (float(r['area_b_mm2']), r['organ'])
A_SIL = np.array([sil[r][0] for r in rows.tolist()]); ORGAN = np.array([sil[r][1] for r in rows.tolist()])
A_CUBE = 1e6 * T['area_b_cube_m2']; U = T['U_ext']; COIN = T['coincident_1um']


def K(d, half):
    k = np.mean([(1 - FD) * T[f'parts_d{d:g}_s{s}'][:, 0] + FD * T[f'parts_d{d:g}_s{s}'][:, 2] for s in SEEDS], 0)
    return np.where(COIN, 0.5 * k, k) if half else k


def ksig(d):
    ks = [(1 - FD) * T[f'parts_d{d:g}_s{s}'][:, 0] + FD * T[f'parts_d{d:g}_s{s}'][:, 2] for s in SEEDS]
    return np.abs(ks[0] - ks[1]) / 2


def credit(labels, name):
    """blade:<L>, rachis:<L>, unassigned. Blade = axis_kind 0 with a leaf id; rachis = axis_kind 2 with a leaf id (R7 has none)."""
    c = np.full(n, 'unassigned', object)
    lab = labels.astype(int)
    blade = (AX == 0) & (lab >= 1) & (lab <= 8); rach = (AX == 2) & (lab >= 1) & (lab <= 8)
    for i in np.nonzero(blade)[0]: c[i] = f'blade:{lab[i]}'
    for i in np.nonzero(rach)[0]: c[i] = f'rachis:{lab[i]}'
    return c


CR = {'R7': credit(R7, 'R7'), 'R8_pc': credit(R8, 'R8_pc')}


def table(cr, d, half, area=A_SIL):
    ab = U * K(d, half) * area; tot = float(ab.sum()); out = {}
    for L in range(1, 9):
        b = cr == f'blade:{L}'; r = cr == f'rachis:{L}'
        out[L] = dict(blade_area_mm2=float(area[b].sum()), blade_absorbed=float(ab[b].sum()),
                      light_per_area=float(ab[b].sum() / area[b].sum()) if area[b].sum() > 0 else None, rachis_absorbed=float(ab[r].sum()))
    rank = sorted(range(1, 9), key=lambda L: -out[L]['blade_absorbed'])
    for i, L in enumerate(rank): out[L]['rank'] = i + 1
    bl = sum(out[L]['blade_absorbed'] for L in out); ra = sum(out[L]['rachis_absorbed'] for L in out); un = float(ab[cr == 'unassigned'].sum())
    return out, dict(total=tot, leaves=bl, rachis=ra, unassigned=un, share_leaves=bl / tot, share_rachis=ra / tot, share_unassigned=un / tot,
                     closure_abs=abs(bl + ra + un - tot))


res = {s: {h: {d: table(CR[s], d, h) for d in DS} for h in (False, True)} for s in CR}
# ---- checks
chk = {}
r2 = json.load(open(REPO / 'build/engine_static_20261026_01/results/run2.json', encoding='utf-8'))['plants']['R3-1b']['reference']
vpleaf = (VP >= 1) & (VP <= 8); ab5 = U * K(5.0, False) * A_CUBE
chk['reproduction_d5_cube_vpcleanup'] = dict(
    plant=dict(here=float(ab5[vpleaf].sum()) * 1e-6, engine=r2['absorbed_with_extension_ru_m2']['mean']),
    leaf_1=dict(here=float(ab5[VP == 1].sum()) * 1e-6, engine=r2['per_leaf']['leaf_1']['absorbed_with_extension_ru_m2']['mean']),
    leaf_7=dict(here=float(ab5[VP == 7].sum()) * 1e-6, engine=r2['per_leaf']['leaf_7']['absorbed_with_extension_ru_m2']['mean']))
for v in chk['reproduction_d5_cube_vpcleanup'].values(): v['rel_diff'] = v['here'] / v['engine'] - 1
chk['reproduction_pass'] = all(abs(v['rel_diff']) <= 0.005 for v in chk['reproduction_d5_cube_vpcleanup'].values())
chk['same_total'] = {f'd{d:g}_half{h}': dict(R7=res['R7'][h][d][1]['total'], R8_pc=res['R8_pc'][h][d][1]['total'],
                                             rel_diff=res['R8_pc'][h][d][1]['total'] / res['R7'][h][d][1]['total'] - 1) for d in DS for h in (False, True)}
chk['same_total_pass'] = all(abs(v['rel_diff']) <= 0.001 for v in chk['same_total'].values())
chk['closure_max_abs'] = max(res[s][h][d][1]['closure_abs'] for s in res for h in res[s] for d in DS)
canopy = (AX == 0) | (AX == 2)
chk['canopy_area_sil_mm2'] = dict(here=float(A_SIL[canopy].sum()), p1_summary=6031.0, canopy_surface_rows=int((ORGAN == 'canopy_surface').sum()),
                                  blade_plus_axis2_rows=int(canopy.sum()), same_rows=bool(np.array_equal(ORGAN == 'canopy_surface', canopy)))
chk['canopy_area_same_both_labels'] = True     # A_SIL does not depend on the label set
chk['unassigned_blade'] = dict(R7=int(((AX == 0) & (R7 == 0)).sum()), R8_pc=int(((AX == 0) & (R8 == 0)).sum()))
chk['rachis_surfels'] = dict(R7=int((CR['R7'] != 'unassigned').sum() - ((AX == 0) & (R7 > 0)).sum()), R8_pc=int(((AX == 2) & (R8 > 0)).sum()),
                             R8_pc_unabsorbed_axis2=int(((AX == 2) & (R8 == 0)).sum()), stem=int((AX == 1).sum()))
chk['coincident_1um_surfels'] = int(COIN.sum())
chk['pass'] = chk['reproduction_pass'] and chk['same_total_pass'] and chk['closure_max_abs'] < 1e-6 and abs(chk['canopy_area_sil_mm2']['here'] - 6031.17) < 0.5
json.dump(chk, open(HERE / 'CHECKS.json', 'w', encoding='utf-8'), indent=1, default=float)
# ---- tables
L_ = []
P = L_.append
P(f'# Static R3-1b, R7 vs R8_pc — tables\n\n{TAGS}\n\nArea = P1 silhouette-calibrated (b) per surfel (Tier 0). Absorbed = relative light × area (r.u.·mm²). Half-credit off unless noted.\n')
for h in (False, True):
    P(f'## Per leaf at d_ex 2.4 mm, half-credit {"ON" if h else "OFF"}')
    P('| leaf | blade area R7 | R8_pc | Δ | absorbed R7 | R8_pc | Δ | range R8_pc (1.5–5 mm) | light/area R7 | R8_pc | Δ | range R8_pc | rachis R8_pc | rank R7 → R8_pc |')
    P('|' + '---|' * 14)
    for L in range(1, 9):
        a, b = res['R7'][h][DEX][0][L], res['R8_pc'][h][DEX][0][L]
        rng_ab = [res['R8_pc'][h][d][0][L]['blade_absorbed'] for d in DS]; rng_lp = [res['R8_pc'][h][d][0][L]['light_per_area'] for d in DS]
        P(f"| {L}{' (보기 전용)' if L in (2, 5, 6) else ''} | {a['blade_area_mm2']:.1f} | {b['blade_area_mm2']:.1f} | {b['blade_area_mm2'] - a['blade_area_mm2']:+.1f} | "
          f"{a['blade_absorbed']:.2f} | {b['blade_absorbed']:.2f} | {b['blade_absorbed'] - a['blade_absorbed']:+.2f} | {min(rng_ab):.2f}–{max(rng_ab):.2f} | "
          f"{a['light_per_area']:.3f} | {b['light_per_area']:.3f} | {b['light_per_area'] - a['light_per_area']:+.3f} | {min(rng_lp):.3f}–{max(rng_lp):.3f} | "
          f"{b['rachis_absorbed']:.2f} | {a['rank']} → {b['rank']} |")
    P('')
P('## Buckets (share of total absorbed light)')
P('| labels | d mm | half | total r.u.·mm² | leaves (blade) | rachis | unassigned |'); P('|---|---|---|---|---|---|---|')
for s in ('R7', 'R8_pc'):
    for h in (False, True):
        for d in DS:
            b = res[s][h][d][1]
            P(f"| {s} | {d:g} | {'on' if h else 'off'} | {b['total']:.2f} | {b['share_leaves']:.3f} | {b['share_rachis']:.3f} | {b['share_unassigned']:.3f} |")
P('\n## Dynamic G-init view: per-leaf blade area (silhouette (b), mm²) it would receive')
P('| leaf | V-P cleanup (P1 plant model input) | R7 | R8_pc | R8_pc − R7 |'); P('|---|---|---|---|---|')
for L in range(1, 9):
    vp = float(A_SIL[(VP == L) & (AX == 0)].sum()); a = res['R7'][False][DEX][0][L]['blade_area_mm2']; b = res['R8_pc'][False][DEX][0][L]['blade_area_mm2']
    P(f'| {L} | {vp:.1f} | {a:.1f} | {b:.1f} | {b - a:+.1f} |')
(HERE / 'TABLES.md').write_text('\n'.join(L_) + '\n', encoding='utf-8')
with (HERE / 'per_leaf.csv').open('w', encoding='utf-8', newline='') as fh:
    fh.write(f'# {TAGS}\n'); w = csv.writer(fh)
    w.writerow(['labels', 'd_mm', 'half_credit', 'leaf', 'blade_area_mm2', 'blade_absorbed_ru_mm2', 'light_per_area', 'rachis_absorbed_ru_mm2', 'rank'])
    for s in res:
        for h in res[s]:
            for d in DS:
                for L, v in res[s][h][d][0].items():
                    w.writerow([s, d, h, L, v['blade_area_mm2'], v['blade_absorbed'], v['light_per_area'], v['rachis_absorbed'], v['rank']])
with (HERE / 'buckets.csv').open('w', encoding='utf-8', newline='') as fh:
    fh.write(f'# {TAGS}\n'); w = csv.writer(fh); w.writerow(['labels', 'd_mm', 'half_credit', 'total', 'leaves', 'rachis', 'unassigned', 'share_leaves', 'share_rachis', 'share_unassigned'])
    for s in res:
        for h in res[s]:
            for d in DS:
                b = res[s][h][d][1]; w.writerow([s, d, h, b['total'], b['leaves'], b['rachis'], b['unassigned'], b['share_leaves'], b['share_rachis'], b['share_unassigned']])
# ---- per-surfel file (heatmap schema + label columns); main relative_light = d_ex 2.4 mm, half-credit off
c = T['centre_m']
cols = dict(surfel=np.arange(n), ply_row=rows, x_mm=1000 * c[:, 0], y_mm=1000 * c[:, 1], z_mm=1000 * c[:, 2], z_above_substrate_mm=1000 * (c[:, 2] - MEDIUM_Z),
            area_b_mm2=A_SIL, area_b_cube_mm2=A_CUBE, area_a_mm2=1e6 * T['area_a_m2'], U_open_ru=U, in_field_support=T['in_support'].astype(int),
            relative_light=K(DEX, False), relative_light_seed_sigma=ksig(DEX), absorbed_ru_mm2=U * K(DEX, False) * A_SIL, coincident_1um=COIN.astype(int),
            axis_kind=AX, leaf_id=R8, leaf_id_R8pc=R8, leaf_id_R7=R7, leaf_id_vpcleanup=VP, R8pc_source=R8SRC,
            credit_R8pc=np.array([str(x) for x in CR['R8_pc']]), credit_R7=np.array([str(x) for x in CR['R7']]))
for d in DS:
    cols[f'relative_light_d{d:g}mm'] = K(d, False)
np.savez_compressed(HERE / 'per_surfel_light.npz', **cols, tags=np.array(TAGS))
# ---- S2px done marker for the heatmap (accepted-run semantics)
S = REPO / 'build/engine_static_20261026_02/results'; S.mkdir(exist_ok=True)
s2 = dict(surfel=np.arange(n), ply_row=rows, leaf_id=VP, is_leaf=vpleaf.astype(int), x_mm=cols['x_mm'], y_mm=cols['y_mm'], z_mm=cols['z_mm'],
          z_above_substrate_mm=cols['z_above_substrate_mm'], nx=T['normal'][:, 0], ny=T['normal'][:, 1], nz=T['normal'][:, 2], area_b_mm2=A_CUBE, area_a_mm2=cols['area_a_mm2'],
          U_open_ru=U, in_field_support=cols['in_field_support'], relative_light=K(5.0, False), relative_light_seed_sigma=ksig(5.0),
          relative_light_no_occluder=np.mean([(1 - FD) * T[f'parts_d5_s{s}'][:, 1] + FD * T[f'parts_d5_s{s}'][:, 3] for s in SEEDS], 0), absorbed_ru_mm2=ab5)
for d in (1.5, 2.0, 2.4, 3.0):
    s2[f'relative_light_d{d:g}mm'] = K(d, False)
np.savez_compressed(S / 's2px_R3-1b.npz', **s2)
meta = dict(job='S2px', computed_by='Kaggle CPU kernel dolgfish/static-r8pc-trace-r31b-20261028 (local RAM 0.6 GB), aggregated by build/engine_static_r8pc_20261028_01/aggregate.py',
            run_id='engine_static_20261026_01 / Run 2 reference (R3-1b alone at (280, 210) mm)', light_model='STATIC v1: black leaves; self-exclusion d = 5 mm (main column); gate rules v2',
            labels='build/p1_vp_cleanup_20261026_01/surfel_leaf_ids.csv', seeds=SEEDS,
            checks=dict(plant_total=dict(surfels=chk['reproduction_d5_cube_vpcleanup']['plant']['here'], engine=chk['reproduction_d5_cube_vpcleanup']['plant']['engine'], rel_diff=chk['reproduction_d5_cube_vpcleanup']['plant']['rel_diff']),
                        leaf_1=dict(rel_diff=chk['reproduction_d5_cube_vpcleanup']['leaf_1']['rel_diff']), leaf_7=dict(rel_diff=chk['reproduction_d5_cube_vpcleanup']['leaf_7']['rel_diff']), **{'pass': chk['reproduction_pass']}))
json.dump(meta, open(S / 's2px_R3-1b_meta.json', 'w', encoding='utf-8'), indent=1, default=float)
names = list(s2)
with (S / 's2px_R3-1b.tmp').open('w', encoding='utf-8') as fh:
    fh.write(','.join(names) + '\n')
    for i in range(n):
        fh.write(','.join(('%d' % s2[k][i]) if np.asarray(s2[k]).dtype.kind in 'iub' else ('%.6g' % s2[k][i]) for k in names) + '\n')
(S / 's2px_R3-1b.tmp').replace(S / 's2px_R3-1b.csv')
print(json.dumps({k: chk[k] for k in ('reproduction_pass', 'same_total_pass', 'closure_max_abs', 'canopy_area_sil_mm2', 'unassigned_blade', 'rachis_surfels', 'coincident_1um_surfels', 'pass')}, default=float))
