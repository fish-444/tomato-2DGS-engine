"""V-S: measured leaf shading ratios (handoff/MEASURE_SHADING_PROTOCOL_2026-10-26.md) vs the STATIC model. Runs only when the CSV has rows.
python compare_shading.py [--csv path] -> results/VS_COMPARISON.json (gate: median |model - measured| <= 0.10 and Spearman >= 0.6).
No model parameter is read from or adjusted to the measurements."""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
CSV = REPO / 'handoff/MEASURE_SHADING_2026-10-26.csv'
DRAWS, JITTER, GAP, DISC = 200, 0.005, 0.001, 16
SEEDS = (101, 202)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else None


def model_ratio(scene_plants, target, row, rng):
    """Median K_below / K_above over jittered draws: column at (x+dx, y+dy); leaf surfel (of leaf_id if given) nearest to z+dz within the column;
    horizontal sensors GAP below and above it, full scene as occluders. Returns (median, p2.5, p97.5, accepted draws)."""
    from static import obs, light as L
    from static.tracer import Scene, trace_cabinet
    q = target; lid = row['leaf_id'].strip()
    sel = q.leaf & ((q.leaf_id == int(lid)) if lid.isdigit() else np.ones(len(q.centre), bool))
    C = q.centre[sel]
    x0 = q.pos[0] + float(row['x_mm']) / 1000; y0 = q.pos[1] + float(row['y_mm']) / 1000; z0 = obs.RIM_Z + float(row['z_mm']) / 1000
    rad = (float(row['diffuser_mm']) / 2000) if row.get('diffuser_mm', '').strip() else 0.005
    # the meter reads the mean over its diffuser: DISC points on a sunflower disc of the diffuser radius, sensors 1 mm beyond the leaf's
    # surfel extent inside that disc (reconstructed leaves have thickness), below and above
    k = np.arange(DISC) + 0.5; rr = rad * np.sqrt(k / DISC); th = k * math.pi * (3 - math.sqrt(5)); disc = np.stack([rr * np.cos(th), rr * np.sin(th)], 1)
    pts, ok = [], 0
    for _ in range(DRAWS):
        dx, dy, dz = rng.uniform(-JITTER, JITTER, 3)
        m = (np.hypot(C[:, 0] - x0 - dx, C[:, 1] - y0 - dy) <= rad) & (np.abs(C[:, 2] - z0 - dz) <= JITTER)
        if not m.any():
            continue
        zb, za = C[m, 2].min() - GAP, C[m, 2].max() + GAP
        xy = disc + [x0 + dx, y0 + dy]
        pts.append(np.concatenate([np.column_stack([xy, np.full(DISC, zb)]), np.column_stack([xy, np.full(DISC, za)])])); ok += 1
    if ok == 0:
        return None
    P = np.concatenate(pts); Ns = np.tile([0, 0, 1.0], (len(P), 1))
    Ps, ks = zip(*[p.prims() for p in scene_plants]); scene = Scene(np.concatenate(Ps), np.concatenate(ks), h=0.003)
    ratios = []
    for sd in SEEDS:
        _, _, _, parts = trace_cabinet(scene, P, Ns, False, np.full(len(P), -1), 1e-7, L.bar_points(sd), L.cos_m(), L.dome(sd), L.DIFFUSE)
        K = obs.kernel(parts).reshape(ok, 2, DISC).mean(2); ratios.append(K[:, 0] / np.maximum(K[:, 1], 1e-9))
    r = np.concatenate(ratios)
    return float(np.median(r)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', default=str(CSV)); ap.add_argument('--out', default=str(HERE / 'results/VS_COMPARISON.json'))
    a = ap.parse_args(); path = Path(a.csv)
    if not path.exists():
        print('V-S not evaluated: no CSV at', path); return
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    if not rows:
        print('V-S not evaluated: CSV has no data rows'); return
    from static import obs
    from runs import TRAY, tray_positions
    out, dropped = [], []
    rng = np.random.default_rng(20261026)
    for r in rows:
        if r['leaf_moved'].strip().lower() == 'yes' or float(r['par_above'] or 0) <= 0 or r['par_below'].strip() == '':
            dropped.append(dict(r, reason='leaf_moved or missing/zero reading')); continue
        if r['leaf_id'].strip() in ('2', '5', '6'):
            dropped.append(dict(r, reason='viewing_only leaf')); continue
        centre = np.array([float(r['pot_centre_x_mm']), float(r['pot_centre_y_mm'])]) / 1000
        if r['config'] == 'single':
            target = obs.Placed(r['plant_id'], centre); scene = [target]
        elif r['config'] == 'tray60':
            # measured pot centres are not in this row for the neighbours: the declared assumed_layout around the target's own position is used
            off = centre - tray_positions(0.060)[TRAY.index(r['plant_id'])]
            scene = [obs.Placed(pid, pos + off) for pid, pos in zip(TRAY, tray_positions(0.060))]; target = scene[TRAY.index(r['plant_id'])]
        else:
            dropped.append(dict(r, reason='config not modelled')); continue
        if r['point_kind'] == 'leaf' and r['plant_id'] != 'R3-1b':
            dropped.append(dict(r, reason='leaf points need R3-1b leaf ids')); continue
        mr = model_ratio(scene, target, r, rng)
        if mr is None:
            dropped.append(dict(r, reason='no model surface within 2 mm of the jittered column')); continue
        meas = float(r['par_below']) / float(r['par_above'])
        out.append(dict(plant=r['plant_id'], config=r['config'], point_id=r['point_id'], point_kind=r['point_kind'], leaf_id=r['leaf_id'],
                        measured=meas, model=mr[0], model_p2_5=mr[1], model_p97_5=mr[2], draws=mr[3], abs_diff=abs(mr[0] - meas)))
    res = dict(csv=str(path), rows=len(rows), compared=len(out), dropped=dropped, points=out, scale='conditional_on_unvalidated_scale')
    if out:
        med = float(np.median([p['abs_diff'] for p in out])); rho = spearman(np.array([p['measured'] for p in out]), np.array([p['model'] for p in out]))
        res.update(median_abs_diff=med, spearman=rho, gate=dict(median_abs_diff_max=0.10, spearman_min=0.6),
                   passed=bool(med <= 0.10 and rho is not None and rho >= 0.6))
    Path(a.out).parent.mkdir(exist_ok=True); json.dump(res, open(a.out, 'w', encoding='utf-8'), indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('rows', 'compared', 'median_abs_diff', 'spearman', 'passed')}))


if __name__ == '__main__':
    main()
