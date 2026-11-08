"""Check (iii), geometry only: within-compound-leaf top-view overlap = 1 - union / sum of the leaflets' projected areas (raster 0.1 mm), per compound leaf; median reported.
Old T (engine_disc_vs_tri_20261101_01/T_triangles.csv) vs T21 vs T21_flat."""
import csv, json, sys
from pathlib import Path
import numpy as np
from matplotlib.path import Path as MP
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
SRC = {'T_old': B / 'engine_disc_vs_tri_20261101_01/T_triangles.csv', 'T_old_flat(coplanar_sep)': B / 'engine_coplanar_sep_20261101_01/T_flat_triangles.csv', 'T21': HERE / 'T21_triangles.csv', 'T21_flat': HERE / 'T21_flat_triangles.csv'}
RES = 0.1


def read(p):
    rows = [r for r in csv.reader(l for l in open(p, encoding='utf-8') if not l.startswith('#'))][1:]
    return [(r[0], r[1], r[3], np.array([float(x) for x in r[6:]]).reshape(3, 3) * 1000) for r in rows]


out = {}
for name, p in SRC.items():
    T = [t for t in read(p) if t[1] == 'Leaflet']; per = []
    for line in sorted({t[2] for t in T}):
        tl = [t for t in T if t[2] == line]; xy = np.concatenate([t[3][:, :2] for t in tl]); lo = xy.min(0) - RES; hi = xy.max(0) + RES
        gx = np.arange(lo[0], hi[0], RES); gy = np.arange(lo[1], hi[1], RES); G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        cnt = np.zeros(len(G), np.int16)
        for organ in sorted({t[0] for t in tl}):
            m = np.zeros(len(G), bool)
            for t in tl:
                if t[0] != organ:
                    continue
                tri = t[3][:, :2]; bb = (G[:, 0] >= tri[:, 0].min()) & (G[:, 0] <= tri[:, 0].max()) & (G[:, 1] >= tri[:, 1].min()) & (G[:, 1] <= tri[:, 1].max())
                if bb.any():
                    idx = np.nonzero(bb)[0]; m[idx[MP(tri).contains_points(G[idx])]] = True
            cnt += m
        s = cnt.sum(); per.append(dict(line=line, overlap=float(1 - (cnt > 0).sum() / s) if s else None))
    v = [x['overlap'] for x in per if x['overlap'] is not None]
    out[name] = dict(median=float(np.median(v)), per_line=per); print(name, 'median overlap', round(float(np.median(v)), 4), 'range', round(min(v), 3), round(max(v), 3), flush=True)
json.dump(out, open(HERE / 'overlap_check.json', 'w'), indent=1)
