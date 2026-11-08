"""Offset variants of flat T (CONTRACT.md §1): each leaflet translated rigidly along its +z-flipped normal by +-delta, sign alternating by leaflet
order within each skeleton line. Local, light. python make_offsets.py -> T_flat_offset_<d>_triangles.csv, inputs copied, OFFSET_META.json"""
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; CUR = HERE.parent / 'engine_curved_T_20261101_01'
TAGS = ['coplanar_separation', 'representation_comparison_not_validation', 'conditional_on_unvalidated_scale', 'after_checkpoint_v1', 'R3-1b_unaccepted', 'engine_law_leaflets']
DELTAS = (0.05, 0.1, 0.2)                                          # mm; 0.1 primary


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    lines = open(p, encoding='utf-8').read().splitlines()
    hdr = lines[1]; rows = list(csv.reader(lines[2:]))
    return hdr, rows


def leaflet_area(rows):
    out = {}
    for r in rows:
        if r[1] != 'Leaflet':
            continue
        t = np.array([float(x) for x in r[6:]]).reshape(3, 3)
        out[r[0]] = out.get(r[0], 0.0) + 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
    return out


def main():
    for f in ('T_flat_triangles.csv', 'T_fold_triangles.csv', 'T_foldcup_triangles.csv'):
        shutil.copyfile(CUR / f, HERE / f)
        assert sha(CUR / f) == sha(HERE / f)
    hdr, rows = read(HERE / 'T_flat_triangles.csv')
    order, normal = {}, {}
    for r in rows:                                                  # leaflet order within each line, file order
        if r[1] == 'Leaflet' and r[0] not in order:
            order[r[0]] = sum(1 for o in order if o.rsplit('/', 1)[0] == r[0].rsplit('/', 1)[0])
            t = np.array([float(x) for x in r[6:]]).reshape(3, 3); n = np.cross(t[1] - t[0], t[2] - t[0]); n /= np.linalg.norm(n)
            normal[r[0]] = -n if n[2] < 0 else n
    base_area = leaflet_area(rows); meta = dict(labels=TAGS, deltas_mm=DELTAS, primary_mm=0.1, leaflets=len(order),
                                                sign_rule='+delta for even, -delta for odd leaflet index within each skeleton line (file order P1L,P1R,...,T)', variants={}, files={})
    # coplanarity check of flat leaflets (all triangles of a leaflet in one plane)
    flat_dev = 0.0
    for r in rows:
        if r[1] == 'Leaflet':
            t = np.array([float(x) for x in r[6:]]).reshape(3, 3); n = np.cross(t[1] - t[0], t[2] - t[0])
            if np.linalg.norm(n) > 0:
                flat_dev = max(flat_dev, 1 - abs((n / np.linalg.norm(n)) @ normal[r[0]]))
    meta["flat_leaflet_planarity_max_1_minus_cos"] = float(flat_dev)
    for d in DELTAS:
        name = f'T_flat_offset_{d:g}'; out = []
        for r in rows:
            if r[1] == 'Leaflet':
                s = 1.0 if order[r[0]] % 2 == 0 else -1.0; sh = s * d / 1000 * normal[r[0]]
                t = np.array([float(x) for x in r[6:]]).reshape(3, 3) + sh
                out.append(r[:6] + [repr(float(v)) for v in t.reshape(-1)])
            else:
                out.append(r)
        with open(HERE / f'{name}_triangles.csv', 'w', encoding='utf-8', newline='') as fh:
            fh.write('# ' + ' · '.join(TAGS) + f'; variant {name} (flat T leaflets translated +-{d:g} mm along their normals); board frame metres\n')
            fh.write(hdr + '\n'); csv.writer(fh).writerows(out)
        _, back = read(HERE / f'{name}_triangles.csv'); a = leaflet_area(back)
        rel = max(abs(a[k] / base_area[k] - 1) for k in base_area)
        meta['variants'][name] = dict(max_abs_rel_leaflet_area_diff=float(rel), G2_pass=bool(rel <= 1e-12), note='offset coordinates written at full float precision')
        meta['files'][f'{name}_triangles.csv'] = sha(HERE / f'{name}_triangles.csv')
    for f in ('T_flat_triangles.csv', 'T_fold_triangles.csv', 'T_foldcup_triangles.csv'):
        meta['files'][f] = sha(HERE / f)
    (HERE / 'OFFSET_META.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ('leaflets', 'flat_leaflet_planarity_max_1_minus_cos', 'variants')}, indent=1))


if __name__ == '__main__':
    main()
