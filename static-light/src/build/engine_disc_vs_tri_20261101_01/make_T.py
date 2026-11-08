"""T geometry for the disc-vs-triangle comparison (CONTRACT.md §1): the unchanged crosswalk (skeleton -> plant_tree_v1 -> engine Stage-2
structure) plus engine-law leaflets on the observed rachis lines. Local, light. python make_T.py -> crosswalk/, T_triangles.csv, T.obj, T.ply, T_meta.json"""
import csv, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs.skeleton_tree import skeleton_to_plant_tree, engine_round_trip

SKEL = ROOT / 'data/skeletons/R3-1b/2026-10-21/skeleton_v1.json'
NINE = ROOT / 'build/engine_static_nine_20261028_01/kaggle_data'
LABELS = ROOT / 'build/engine_static_tau_20261031_01/kaggle_data/labels_rachis.npz'
MODEL = ROOT / 'build/engine_maxmass_fix_20261025_01/runs/seed72/native/working/Model.rgg'
TAGS = ['disc_vs_triangle', 'relative_light', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1', 'R3-1b_unaccepted', 'engine_law_leaflets']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def engine_params():
    """Parameter values and leaflet laws of the reference engine run (seed72 Model.rgg: leaflet_outline hook on, library_v2)."""
    txt = MODEL.read_text(encoding='utf-8'); out = {}
    for k in ('PETIOLE_M', 'RACHIS_M', 'RACHIS_WIDTH_M', 'LEAF_LW', 'BASAL_LEAFLET_ANGLE_DEG', 'MIDDLE_RAMP'):
        out[k] = float(re.search(rf'double {k} = ([0-9.eE+-]+);', txt).group(1))
    out['PAIRS'] = [int(v) for v in re.search(r'int\[\] PAIRS = new int\[\]\{([0-9,]+)\}', txt).group(1).split(',')]
    out['ENGINE_SEED'] = int(re.search(r'int n=\d+,seed=(\d+)', txt).group(1))
    for law in ('int pairs(int rank) { return PAIRS[Math.min(rank-1,PAIRS.length-1)]; }',
                'double pairWeight(int i,int m) { return m==1?1:1+MIDDLE_RAMP*(1-Math.abs(2.0*(i-1)/(m-1)-1)); }',
                'if(o.kind==9)return outlineVertices(o);',
                'o.outline=new java.util.Random((long)seed+(long)o.id.hashCode()).nextInt(outlineIds.length);',
                'double az=Math.toRadians(o.az),el=Math.toRadians(o.elev),scale=Math.sqrt(o.area);',
                'result[j]=(float)(o.x+scale*(uv[i]*ux+uv[i+1]*vx));',
                'double beta=Math.toRadians(o.side*BASAL_LEAFLET_ANGLE_DEG*(m+1-o.pair)/m);'):
        assert law in txt, 'engine law not found verbatim: ' + law
    sizes = [int(x) for x in re.search(r'outlineSizes=new int\[\]\{([^}]*)\}', txt).group(1).split(',')]
    uv = [np.zeros(n) for n in sizes]
    with open(MODEL.parent / 'leaflet_outline_uv.csv', encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            a, b, c = line.strip().split(','); uv[int(a)][int(b)] = float(c)
    out['_uv'] = uv; out['OUTLINES'] = len(uv)
    return out


def java_hash(s):
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= 1 << 31 else h


def java_next_int(seed, bound):
    """java.util.Random(seed).nextInt(bound), exactly."""
    st = [(seed ^ 0x5DEECE66D) & ((1 << 48) - 1)]
    def nxt():
        st[0] = (st[0] * 0x5DEECE66D + 0xB) & ((1 << 48) - 1); return st[0] >> 17
    if bound & (bound - 1) == 0:
        return (bound * nxt()) >> 31
    while True:
        bits = nxt(); val = bits % bound
        if ((bits - val + (bound - 1)) & 0xFFFFFFFF) < (1 << 31):
            return val


def d_blade_areas():
    """Blade area (b) per R8_pc leaf over the kept surfels of the frozen static scene (static_surfels.csv rows)."""
    rows, A = [], []
    with open(NINE / 'R3-1b/static_surfels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith('#')):
            rows.append(int(r['surfel_row'])); A.append(float(r['area_b_mm2']))
    rows = np.array(rows); A = np.array(A); z = np.load(LABELS)
    pos = {r: i for i, r in enumerate(z['rows'].tolist())}; idx = np.array([pos[r] for r in rows.tolist()])
    ax = z['axis_kind'][idx].astype(int); lab = z['label'][idx].astype(int)
    return {L: float(A[(ax == 0) & (lab == L)].sum()) for L in range(1, 9)}, float(A.sum())


def interval_point(P, cum, s):
    k = int(np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(P) - 2)); seg = cum[k + 1] - cum[k]
    t = 0.0 if seg <= 0 else (s - cum[k]) / seg
    return P[k] + t * (P[k + 1] - P[k])


def diamond(x, u, v):
    return [np.array([x, x + u / 2 + v, x + u]), np.array([x, x + u, x + u / 2 - v])]


def az_el(d):
    d = d / np.linalg.norm(d)
    return math.atan2(d[1], d[0]), math.asin(max(-1.0, min(1.0, d[2])))


def leaf_triangles(P, rank, area, E, tag, oid, variant):
    """Engine-law compound leaf on the observed polyline P (mm). Returns list of (organ, class, pair, side, tri 3x3 mm)."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]); Lp = cum[-1]
    m = E['PAIRS'][min(rank - 1, len(E['PAIRS']) - 1)]
    w = (lambda i: 1.0 if m == 1 else 1 + E['MIDDLE_RAMP'] * (1 - abs(2.0 * (i - 1) / (m - 1) - 1)))
    total = 1 + sum(2 * w(i) for i in range(1, m + 1))
    s_pet = Lp * E['PETIOLE_M'] / (E['PETIOLE_M'] + E['RACHIS_M']); step = (Lp - s_pet) / m
    marks = [0.0, s_pet] + [s_pet + i * step for i in range(1, m + 1)]           # petiole [0,s_pet], R_k [marks[k], marks[k+1]]
    pts = [interval_point(P, cum, s) for s in marks]
    out = []
    names = ['P'] + [f'R{k}' for k in range(1, m + 1)]
    for k, nm in enumerate(names):                                              # slender diamonds along each chord, width RACHIS_WIDTH_M
        a, b = pts[k], pts[k + 1]; u = b - a
        if np.linalg.norm(u) <= 0:
            continue
        az, _ = az_el(u); wd = 1000 * E['RACHIS_WIDTH_M']; v = np.array([-wd * math.sin(az) / 2, wd * math.cos(az) / 2, 0.0])
        for t in diamond(a, u, v):
            out.append((f'{tag}/{nm}', 'Petiole' if nm == 'P' else 'RachisSegment', 0, 0, t))
    leaflets = [(i, s, w(i) / total, k) for i in range(1, m + 1) for s in (-1, 1) for k in [i - 1]] + [(m + 1, 0, 1 / total, m)]
    for pair, side, share, parent in leaflets:                                   # parent interval index: 0 petiole, k = R_k
        base = pts[parent + 1]; pa, pe = az_el(pts[parent + 1] - pts[parent])
        A = area * share; ln = math.sqrt(2 * A * E['LEAF_LW']); wd = ln / E['LEAF_LW']
        beta = math.radians(side * E['BASAL_LEAFLET_ANGLE_DEG'] * (m + 1 - pair) / m)
        dx = math.cos(beta) * math.cos(pe) * math.cos(pa) - math.sin(beta) * math.sin(pa)
        dy = math.cos(beta) * math.cos(pe) * math.sin(pa) + math.sin(beta) * math.cos(pa)
        dz = math.cos(beta) * math.sin(pe)
        az = math.atan2(dy, dx); el = math.asin(max(-1.0, min(1.0, dz)))
        u = ln * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        v = wd / 2 * np.array([-math.sin(beta) * math.cos(pe) * math.cos(pa) - math.cos(beta) * math.sin(pa),
                               -math.sin(beta) * math.cos(pe) * math.sin(pa) + math.cos(beta) * math.cos(pa), -math.sin(beta) * math.sin(pe)])
        nm = 'T' if side == 0 else f'P{pair}{"L" if side < 0 else "R"}'
        if variant == 'diamond':                                                 # base engine (tools/vtc_development_native.rgg): full-outline diamond
            for t in diamond(base, u, v):
                out.append((f'{tag}/{nm}', 'Leaflet', pair, side, t))
        else:                                                                    # reference run (seed72): library_v2 outline, outlineVertices()
            k = java_next_int(E['ENGINE_SEED'] + java_hash(f'{oid}/{nm}'), E['OUTLINES']); uvk = E['_uv'][k]; sc = math.sqrt(A)
            uh = u / ln; vh = v / (wd / 2)
            pts3 = base + sc * (uvk[0::2, None] * uh + uvk[1::2, None] * vh)
            for t in pts3.reshape(-1, 3, 3):
                out.append((f'{tag}/{nm}', 'Leaflet', pair, side, t))
    return out, m


def write_files(tris, stem, full):
    with open(HERE / f'{stem}_triangles.csv', 'w', encoding='utf-8', newline='') as fh:
        fh.write('# ' + ' · '.join(TAGS) + f'; variant {stem}; board frame metres\n'); w = csv.writer(fh)
        w.writerow(['organ', 'class', 'leaf_R8pc', 'line', 'pair', 'side', 'x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
        for r in tris:
            w.writerow([r['organ'], r['cls'], r['leaf'], r['line'], r['pair'], r['side']] + [f'{v / 1000:.9f}' for v in r['tri'].reshape(-1)])
    files = [f'{stem}_triangles.csv']
    if full:
        with open(HERE / f'{stem}.obj', 'w', encoding='utf-8') as fh:
            fh.write('# ' + ' · '.join(TAGS) + f'; variant {stem}; board frame metres; one group per organ\n'); k = 1; cur = None
            for r in tris:
                if r['organ'] != cur:
                    cur = r['organ']; fh.write(f'g {cur.replace("/", "__")}\n')
                for p in r['tri']:
                    fh.write('v %.9f %.9f %.9f\n' % tuple(p / 1000))
                fh.write(f'f {k} {k + 1} {k + 2}\n'); k += 3
        V = np.array([r['tri'] for r in tris]) / 1000; cr = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]); nrm = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-30)
        cen = V.mean(1)
        with open(HERE / f'{stem}.ply', 'wb') as fh:
            fh.write((f'ply\nformat binary_little_endian 1.0\ncomment {" · ".join(TAGS)}; variant {stem}; triangle centroids, board m\nelement vertex {len(cen)}\n'
                      'property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty int leaf\nproperty char leaflet\nend_header\n').encode())
            a = np.empty(len(cen), np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('leaf', '<i4'), ('leaflet', 'i1')]))
            for i, c in enumerate('xyz'):
                a[c] = cen[:, i]; a['n' + c] = nrm[:, i]
            a['leaf'] = [r['leaf'] for r in tris]; a['leaflet'] = [r['cls'] == 'Leaflet' for r in tris]; fh.write(a.tobytes())
        files += [f'{stem}.obj', f'{stem}.ply']
    return files


def main():
    E = engine_params(); sk = SIO.load(SKEL); areas_leaf, d_total = d_blade_areas()
    live = [l for l in sk['leaves'] if l['status'] in ('active', 'new', 'candidate_lost')]
    plen = {l['leaf_uid']: SIO.path_length(np.asarray(l['polyline_xyz'], float)) * 1000 for l in live}
    pl_leaf = {l['leaf_uid']: int(l['attrs']['pipeline_leaf']) for l in live}
    line_area = {}
    for u in plen:
        L = pl_leaf[u]; tot = sum(plen[v] for v in plen if pl_leaf[v] == L)
        line_area[u] = dict(area_b_mm2=areas_leaf[L] * plen[u] / tot, n_surfels=0)
    tree = skeleton_to_plant_tree(sk, line_area, area_source='D blade area (b) of the R8_pc leaf split over its skeleton lines by path length (CONTRACT §1)')
    rt = engine_round_trip(tree, HERE / 'crosswalk', 'R3-1b')
    assert rt.get('round_trip_pass') is True, rt
    (HERE / 'crosswalk' / 'plant_tree_v1.json').write_text(json.dumps(tree, indent=1), encoding='utf-8')
    from tomato3d.plant_tree import engine as PE
    rep = PE.to_engine_structure(tree, HERE / 'crosswalk', plant_instance_id='p1', source_plant_id='reconstruction_R3-1b', fixture=False, capture_id='skeleton_R3-1b')
    nodes = {n['id']: n for n in tree['nodes']}; meta_files = {}; flat_all = {}; lines = []; counts = {}
    for variant, stem in (('outline', 'T'), ('diamond', 'T_diamond')):
        tris = []
        for lf in tree['leaves']:
            uid = lf['attrs']['leaf_uid']; rank = nodes[lf['node']]['index']; P = np.asarray(lf['rachis']['polyline'], float); oid = rep['id_map'][lf['id']]
            t, m = leaf_triangles(P, rank, line_area[uid]['area_b_mm2'], E, f'R3-1b/{uid}', oid, variant)
            for organ, cls, pair, side, tri in t:
                tris.append(dict(organ=organ, cls=cls, leaf=pl_leaf[uid], line=uid, pair=pair, side=side, tri=tri))
            if variant == 'outline':
                lines.append(dict(leaf_uid=uid, leaf_R8pc=pl_leaf[uid], engine_organ_id=oid, rank=rank, pairs=m, path_mm=plen[uid], area_b_mm2=line_area[uid]['area_b_mm2']))
        meta_files.update({f: None for f in write_files(tris, stem, True)})
        V = np.array([r['tri'] for r in tris]); area = 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1)
        leafcol = np.array([r['leaf'] for r in tris]); isl = np.array([r['cls'] == 'Leaflet' for r in tris])
        t_leaf = {L: float(area[(leafcol == L) & isl].sum()) for L in range(1, 9)}
        flat_all[stem] = [dict(leaf=L, D_area_b_mm2=areas_leaf[L], T_area_mm2=t_leaf[L], rel_diff=(t_leaf[L] / areas_leaf[L] - 1) if areas_leaf[L] else None) for L in range(1, 9)]
        counts[stem] = dict(triangles=len(tris), leaflet_triangles=int(isl.sum()), leaflets=len({r['organ'] for r in tris if r['cls'] == 'Leaflet'}))
    E2 = {k: v for k, v in E.items() if not k.startswith('_')}
    meta = dict(labels=TAGS, variants={'T': 'main: reference engine run seed72 (leaflet_outline hook on, library_v2 outline per leaflet, outlineIndex = java.util.Random(seed 5000 + id.hashCode()))',
                                       'T_diamond': 'secondary: base engine tools/vtc_development_native.rgg full-outline diamond (len = sqrt(2 A LEAF_LW))'},
                engine_params=E2, engine_model_rgg=str(MODEL.relative_to(ROOT)), engine_model_sha256=sha(MODEL), outline_csv_sha256=sha(MODEL.parent / 'leaflet_outline_uv.csv'),
                skeleton=str(SKEL.relative_to(ROOT)), skeleton_sha256=sha(SKEL), crosswalk_round_trip=rt, lines=lines, counts=counts,
                D_blade_area_b_mm2=areas_leaf, D_all_surfel_area_b_mm2=d_total, T_flat_check=flat_all,
                max_abs_rel_area_diff={k: max(abs(f['rel_diff']) for f in v if f['rel_diff'] is not None) for k, v in flat_all.items()},
                not_in_T=['stem (crosswalk main axis = zero-length placeholder; display only)', 'trusses / flowers (none observed)', 'D surfels outside the blades (rachis, stem, unassigned) as receivers'],
                files={p: sha(HERE / p) for p in meta_files})
    (HERE / 'T_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(dict(counts=counts, max_abs_rel_area_diff=meta['max_abs_rel_area_diff'])))
    for l in lines:
        print(l['leaf_uid'], l['leaf_R8pc'], l['engine_organ_id'], 'rank', l['rank'], 'pairs', l['pairs'], 'path %.1f' % l['path_mm'], 'area %.0f' % l['area_b_mm2'])


if __name__ == '__main__':
    main()
