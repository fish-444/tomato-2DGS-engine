"""Tables of the 32-plant tray scenes from the kernel outputs (out_t*/scene, out_t*/iso, conv, extra). python -X utf8 aggregate.py [A|B|AB]
Writes scenes.csv, luminaires.csv, efficiency_by_scene.csv, plants_by_scene.csv, leaves_by_scene.csv, position_effect.csv, scenario_effects.csv (+ convergence / extra / closure notes)."""
import csv, glob, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tray32_lib as T
TAG = 'light_model_v1_tau; assumed_luminaire_model; repeated_placement; relative_light; conditional_on_unvalidated_scale; skeleton_labels_ungraded; scenario_comparison_not_validation'
A_ABS = T.A_ABS
REF = dict(p=0.10, shape='flat', pos='trays', height=0.40)


def J(p):
    return json.load(open(p, encoding='utf-8'))


def load(table):
    sc, iso = {}, {}
    for f in glob.glob(str(HERE / 'out_t*' / 'scene' / f'{table}_S*.json')):
        d = J(f); sc[d['scene']['sid']] = d
    for f in glob.glob(str(HERE / 'out_t*' / 'iso' / f'{table}_S*.json')):
        d = J(f); iso[d['scene']['sid']] = d
    return sc, iso


def wcsv(name, rows, note=''):
    if not rows:
        return
    with open(HERE / name, 'w', encoding='utf-8', newline='') as f:
        f.write(f'# {TAG}{"; " + note if note else ""}\n'); w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def r(x, n=5):
    return None if x is None else round(float(x), n)


tables = [t for t in (sys.argv[1] if len(sys.argv) > 1 else 'AB') if t in 'AB']
PL = {t: list(csv.DictReader(open(HERE / f'placement_{t}.csv', encoding='utf-8'))) for t in tables}
SC = T.scene_list(); LU = []
for s in SC:
    for f in T.faces(s['shape'], s['pos'], s['p'], s['height']):
        LU.append(dict(sid=s['sid'], p=s['p'], shape=s['shape'], position=s['pos'], height=s['height'], tag=f['tag'], xa=r(f['xa'], 4), za=r(f['za'], 4), xb=r(f['xb'], 4), zb=r(f['zb'], 4), nx=r(f['nx'], 4), nz=r(f['nz'], 4), flux_L=f['flux'], length_m=T.LUM_LEN))
wcsv('luminaires.csv', LU, 'faces (strips in xz extruded along y); flux in L (unit luminaire output); 384 sample points per luminaire')
wcsv('scenes.csv', [dict(sid=s['sid'], spacing_m=s['p'], shape=s['shape'], position=s['pos'], height_m=s['height'], emitted_L=4.0, emitter_points=1536, reference=bool(all(s[k] == v for k, v in REF.items()))) for s in SC], 'reference scene = flat, over trays, 0.40 m, p 0.10 m')
eff, pls, lvs, scen, posrows = [], [], [], [], []
DATA = {}
for t in tables:
    sc, iso = load(t); DATA[t] = (sc, iso)
    if not sc:
        continue
    ref_sid = [s['sid'] for s in SC if all(s[k] == v for k, v in REF.items())][0]; refd = sc.get(ref_sid)
    for sid, d in sorted(sc.items()):
        s = d['scene']; cps = d['copies']; ab = np.array([c['absorbed_a1'] * A_ABS for c in cps]); ar = np.array([c['area_b_m2'] for c in cps]); un = np.array([c['unshaded_a1'] * A_ABS for c in cps]); perarea = ab / ar
        isod = iso.get(sid); c = d['closure']
        row = dict(table=t, sid=sid, spacing_m=s['p'], shape=s['shape'], position=s['pos'], height_m=s['height'], efficiency_a085=r(d['efficiency_a085']), efficiency_a1=r(d['efficiency_a1']), outlines_share_of_emitted=r(d['outlines_share_of_emitted']),
                   absorbed_over_outlines_light=r(d['absorbed_over_outlines_a085']), absorbed_total_a085=r(ab.sum()), copies_absorbed_median=r(np.median(ab)), copies_absorbed_p10=r(np.percentile(ab, 10)), copies_absorbed_p90=r(np.percentile(ab, 90)),
                   copies_absorbed_gini=r(T.gini(ab)), per_area_median=r(np.median(perarea)), per_area_p10=r(np.percentile(perarea, 10)), per_area_p90=r(np.percentile(perarea, 90)), per_area_gini=r(T.gini(perarea)),
                   total_shading_mean=r(1 - ab.sum() / un.sum()), closure_up=r(c['up'] / c['emitted']), closure_beyond=r(c['beyond'] / c['emitted']), closure_floor_shaded=r(c['floor_shaded'] / c['emitted']), closure_removed_by_plants=r(c['removed_by_plants'] / c['emitted']),
                   closure_absorbed_plus_reflected=r((c['absorbed_a085'] + c['reflected']) / c['emitted']), closure_residual=r(c['residual_fraction_of_emitted']), overflow_rays=d['overflow_rays'], receivers_per_copy=d['receivers_per_copy'], ke=d['ke'])
        if isod:
            isoab = np.array([x['isolated_a1'] * A_ABS for x in isod['copies']]); row['self_shading_mean'] = r(1 - isoab.sum() / un.sum()); row['neighbour_shading_share_mean'] = r(1 - ab.sum() / isoab.sum())
        if refd is not None:
            row['efficiency_diff_vs_reference'] = r(d['efficiency_a085'] - refd['efficiency_a085']); row['efficiency_ratio_vs_reference'] = r(d['efficiency_a085'] / refd['efficiency_a085'])
        eff.append(row)
        for i, (cp, pl) in enumerate(zip(cps, PL[t])):
            a_ = cp['absorbed_a1'] * A_ABS; u_ = cp['unshaded_a1'] * A_ABS; io = isod['copies'][i] if isod else None
            pls.append(dict(table=t, sid=sid, copy=pl['copy_id'], plant=pl['plant'], tray=pl['tray'], row=pl['row'], column=pl['column'], tray_class=pl['tray_class'], row_class=pl['row_class'], absorbed_a085=r(a_, 7), area_b_m2=r(cp['area_b_m2'], 7),
                            absorbed_per_area_b=r(a_ / cp['area_b_m2'], 6), total_shading=r(1 - a_ / u_), self_shading=r(1 - io['isolated_a1'] * A_ABS / u_) if io else None, neighbour_shading_share=r(1 - cp['absorbed_a1'] / io['isolated_a1']) if io else None,
                            unassigned_light_share=r(cp['unassigned_absorbed_a1'] / cp['absorbed_a1'])))
            ranked = sorted(cp['leaves'].items(), key=lambda kv: -kv[1]['absorbed_a1'])
            for rank, (k, lv) in enumerate(ranked):
                lvs.append(dict(table=t, sid=sid, copy=pl['copy_id'], plant=pl['plant'], leaf=k, absorbed_a085=r(lv['absorbed_a1'] * A_ABS, 8), area_b_m2=r(lv['area_b_m2'], 8), absorbed_per_area_b=r(lv['absorbed_a1'] * A_ABS / lv['area_b_m2'], 6) if lv['area_b_m2'] > 0 else None,
                                rank_in_copy=rank + 1, isolated_a085=r(io['leaves'][k]['isolated_a1'] * A_ABS, 8) if io and k in io['leaves'] else None))
        if refd is not None and sid != ref_sid:
            ref_ab = np.array([c_['absorbed_a1'] * A_ABS for c_ in refd['copies']])
            scen.append(dict(table=t, sid=sid, spacing_m=s['p'], shape=s['shape'], position=s['pos'], height_m=s['height'], efficiency_a085=r(d['efficiency_a085']), reference_efficiency=r(refd['efficiency_a085']), diff=r(d['efficiency_a085'] - refd['efficiency_a085']),
                             ratio=r(d['efficiency_a085'] / refd['efficiency_a085']), per_copy_ratio_median=r(np.median(ab / ref_ab)), per_copy_ratio_p10=r(np.percentile(ab / ref_ab, 10)), per_copy_ratio_p90=r(np.percentile(ab / ref_ab, 90))))
# position effects: per plant, mean absorbed per area over copies of each class; paired over the plants (pooled over the available tables)
if pls:
    import collections
    for sid in sorted({p['sid'] for p in pls}):
        rows = [p for p in pls if p['sid'] == sid]
        for cls_key, a_name, b_name in (('tray_class', 'outer', 'inner'), ('row_class', 'end', 'middle')):
            per = collections.defaultdict(lambda: collections.defaultdict(list))
            for p in rows:
                per[p['plant']][p[cls_key]].append(p['absorbed_per_area_b'])
            diffs = {pl: np.mean(v[a_name]) / np.mean(v[b_name]) - 1 for pl, v in per.items() if v[a_name] and v[b_name]}
            if diffs:
                x = np.array(list(diffs.values())); s = next(sc_ for sc_ in SC if sc_['sid'] == sid)
                posrows.append(dict(sid=sid, spacing_m=s['p'], shape=s['shape'], position=s['pos'], height_m=s['height'], contrast=f'{a_name}/{b_name}', tables=''.join(tables), n_plants=len(x), median_rel_diff=r(np.median(x)), p10=r(np.percentile(x, 10)), p90=r(np.percentile(x, 90)),
                                    mean_rel_diff=r(x.mean()), share_plants_positive=r((x > 0).mean(), 3)))
wcsv('efficiency_by_scene.csv', eff, 'efficiency = absorbed (a 0.85) / emitted (4 L); closure terms as fractions of emitted; self / neighbour shading only where isolated runs exist')
wcsv('plants_by_scene.csv', pls, 'one row per copy and scene; absorbed_per_area_b = absorbed (a 0.85) per m2 of area (b); neighbour_shading_share = 1 - in-scene / isolated (same receivers, same emitter draws)')
wcsv('leaves_by_scene.csv', lvs, 'leaf-level indicative (labels ungraded; assigned surfels only, lower bound)')
wcsv('position_effect.csv', posrows, 'paired per plant: mean absorbed per area (b) of the copies in one class over the other class; median / p10 / p90 over plants')
wcsv('scenario_effects.csv', scen, 'paired differences vs the reference scene (flat, over trays, 0.40 m, p 0.10 m) of the same table')
for t in tables:
    for nm in (f'conv_{t}.json', f'extra_{t}.json'):
        for f in glob.glob(str(HERE / 'out_t*' / nm)):
            (HERE / nm).write_text(Path(f).read_text(encoding='utf-8'), encoding='utf-8')
print('tables', tables, 'scenes', {t: len(DATA[t][0]) for t in DATA}, 'iso', {t: len(DATA[t][1]) for t in DATA}, 'rows', len(eff), len(pls), len(lvs))
