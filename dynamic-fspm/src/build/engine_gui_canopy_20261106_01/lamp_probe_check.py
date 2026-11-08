"""Lamp probe GP1 (investigation of the lamp-field FAIL, 2026-11-07): GroIMP 2 x 2 cm black sensors on two lines (x -0.30..0.30 at y 0;
y -0.40..0.40 at x 0; z 0) vs analytic references, 1e8 rays.
  P_point: the prepared isotropic PointLight 1200 at (0, 0, 0.6) -> E = P/(4 pi) cos(theta_r)/d^2 (tests the sensor's angular response);
  P_flat2 / P_flat1: flat bar lamp at 0.40 m as two half panels / one panel -> static tray32 cos^1 bar model;
  P_V90: V90 lamp -> same model with the wings.
python -X utf8 lamp_probe_check.py -> lamp_probe_check.csv + printed ratio profiles"""
import csv, glob, math
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent; P = 1200.0; A = 0.0004


def bar(shape, h, x, y, n_long=480, n_across=40, m=1.0):
    phi = {'flat': 0.0, 'V90': math.radians(45)}[shape]; ys = (np.arange(n_long) + 0.5) / n_long * 0.6 - 0.3; E = 0.0
    for k in (-1, 1):
        us = (np.arange(n_across // 2) + 0.5) / (n_across // 2) * 0.05
        for u in us:
            ex, ez = k * u * math.cos(phi), u * math.sin(phi) + h; n = np.array([k * math.sin(phi), 0, -math.cos(phi)])
            d = np.stack([x - ex + 0 * ys, y - ys, -ez + 0 * ys], 1); r = np.linalg.norm(d, axis=1); u_ = d / r[:, None]
            ce = np.clip(u_ @ n, 0, None); cr = np.clip(-u_[:, 2], 0, None)
            E += (P / (n_long * n_across) * (m + 1) / (2 * math.pi) * ce ** m * cr / r ** 2).sum()
    return E * A


def point(x, y, z0=0.6):
    d2 = x * x + y * y + z0 * z0; return P / (4 * math.pi) * (z0 / math.sqrt(d2)) / d2 * A


rows = []
for case, ref in (('P_point', lambda x, y: point(x, y)), ('P_flat2', lambda x, y: bar('flat', 0.40, x, y)), ('P_flat1', lambda x, y: bar('flat', 0.40, x, y)),
                  ('P_V90', lambda x, y: bar('V90', 0.40, x, y)), ('Q_e0', lambda x, y: bar('flat', 0.40, x, y)),
                  ('Q_e1', lambda x, y: bar('flat', 0.40, x, y, m=2.0)), ('Q_e0001', lambda x, y: bar('flat', 0.40, x, y, m=1.001))):
    fs = glob.glob(str(H / 'kout' / '*' / 'out' / case / 'gui_sensors.csv'))
    if not fs: print(case, 'no data'); continue
    for r in csv.DictReader(open(fs[0])):
        x, y, g = float(r['x_m']), float(r['y_m']), float(r['absorbed_umol_s']); e = ref(x, y)
        rows.append(dict(case=case, x=x, y=y, groimp=g, reference=e, ratio=g / e if e else None, angle_deg=math.degrees(math.atan2(math.hypot(x, y), 0.6 if case == 'P_point' else 0.4))))
with open(H / 'lamp_probe_check.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
for case in ('P_point', 'P_flat2', 'P_flat1', 'P_V90', 'Q_e0', 'Q_e1', 'Q_e0001'):
    rr = [r for r in rows if r['case'] == case]
    if not rr: continue
    xs = sorted([r for r in rr if r['y'] == 0.0], key=lambda r: r['x']); ys = sorted([r for r in rr if r['x'] == 0.0 and r['y'] != 0.0], key=lambda r: r['y'])
    print(case, 'along x:', ' '.join(f"{r['x']:+.3f}:{r['ratio']:.3f}" for r in xs[::2]))
    print(' ' * len(case), 'along y:', ' '.join(f"{r['y']:+.2f}:{r['ratio']:.3f}" for r in ys[::2]))
