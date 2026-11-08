"""Clearance check (CONTRACT §2 of the v3 prompt, report only, before any light run): per table x scene x copy, vertical gap = luminaire lowest point - highest surfel,
and the 3-D distance from the copy's surfels (top 5 cm) to the nearest luminaire sample point (0.0125 m grid, so the distance is an upper bound by <= 9 mm).
Flags: gap < 0.02 m -> near_luminaire; any surfel above the lowest luminaire point -> inside_luminaire_volume. Nothing is moved or dropped. python -X utf8 clearance.py"""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
sys.path.insert(0, str(B / 'engine_static_tau_20261031_01/kaggle_data')); sys.path.insert(0, str(HERE))
import tray32_lib as T
from static import tau_runs as TR, obs
from scipy.spatial import cKDTree
TR.setup(B / 'engine_static_nine_20261028_01/kaggle_data', B / 'engine_static_tau_20261031_01/kaggle_data')
ND = B / 'engine_static_normals_20261104_01/kaggle_flat'; E_IDS = ['e_01', 'e_04', 'e_05', 'e_07']; _o = obs.ply_path
obs.ply_path = lambda pid: (ND / f'{pid}__surfels.ply') if pid in E_IDS else _o(pid)
top = {}
for pid in ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05'] + E_IDS:
    q = obs.Placed(pid, np.zeros(2)); c = q.centre; zmax = c[:, 2].max(); top[pid] = (zmax, c[c[:, 2] > zmax - 0.05].copy()); obs._CACHE.clear()
rows = []; flags = 0
for t in 'AB':
    PL = list(csv.DictReader(open(HERE / f'placement_{t}.csv', encoding='utf-8')))
    for sc in T.scene_list():
        Pm, _, _ = T.emitters(T.faces(sc['shape'], sc['pos'], sc['p'], sc['height'])); tree = cKDTree(Pm)
        for r in PL:
            zmax, pts = top[r['plant']]; x, y = T.slot_xy(sc['p'], int(r['tray']), int(r['row']), int(r['column'])); Rz = T.rot_z(float(r['azimuth_deg']))
            P = pts @ Rz.T + np.array([x, y, 0.0]); d = float(tree.query(P)[0].min()); gap = sc['height'] - zmax
            near = gap < 0.02; inside = gap < 0.0; flags += near or inside
            rows.append(dict(table=t, sid=sc['sid'], spacing_m=sc['p'], shape=sc['shape'], position=sc['pos'], height_m=sc['height'], copy=r['copy_id'], plant=r['plant'], highest_surfel_z_m=round(zmax, 4),
                             vertical_gap_m=round(gap, 4), min_3d_distance_to_luminaire_points_m=round(d, 4), near_luminaire=near, inside_luminaire_volume=inside))
with open(HERE / 'clearance.csv', 'w', encoding='utf-8', newline='') as f:
    f.write('# vertical gap = luminaire lowest point (height) - highest surfel z (grow frame, floor = pot bottom); 3-D distance to the nearest luminaire sample point (upper bound by <= 9 mm); flags: gap < 0.02 m, surfel above the lowest luminaire point\n')
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
g = np.array([r['vertical_gap_m'] for r in rows]); d3 = np.array([r['min_3d_distance_to_luminaire_points_m'] for r in rows])
summ = dict(rows=len(rows), flagged=int(flags), min_vertical_gap_m=float(g.min()), min_vertical_gap_by_height={str(h): float(min(r['vertical_gap_m'] for r in rows if r['height_m'] == h)) for h in T.HEIGHTS},
            min_3d_distance_by_height={str(h): float(min(r['min_3d_distance_to_luminaire_points_m'] for r in rows if r['height_m'] == h)) for h in T.HEIGHTS}, tallest=max(((k, v[0]) for k, v in top.items()), key=lambda kv: kv[1]))
json.dump(summ, open(HERE / 'clearance_summary.json', 'w'), indent=1); print(summ)
