"""Lamp probe GP3 (2026-11-07): GroIMP AreaLight angular law and panel-shape dependence.
R_s_e0_h10 / R_s_e1_h10: 1 x 1 cm emitter at 0.30 m (names say h10; the settings floor is 30 cm), exponent 0 / 1 -> I(theta)/I(0) from each sensor
  (E d^2 / cos(theta_r)) vs cos^(e+1);  R_long_e0: 1 cm (along y) x 10 cm (across x) at 0.40 m;  R_wide_e0: 60 cm x 1 cm at 0.40 m -> vs cos^1 rectangle.
python -X utf8 lamp_probe_gp3.py"""
import csv, glob, math
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent; P = 1200.0; A = 0.0004


def load(case):
    fs = glob.glob(str(H / 'kout' / '*' / 'out' / case / 'gui_sensors.csv'))
    return [(float(r['x_m']), float(r['y_m']), float(r['absorbed_umol_s'])) for r in csv.DictReader(open(fs[0]))] if fs else []


def rect(L, W, h, x, y, m=1.0, nl=200, nw=40):
    ys = (np.arange(nl) + 0.5) / nl * L - L / 2; xs = (np.arange(nw) + 0.5) / nw * W - W / 2
    ex, ey = np.meshgrid(xs, ys); ex = ex.ravel(); ey = ey.ravel()
    d = np.stack([x - ex, y - ey, -h + 0 * ex], 1); r = np.linalg.norm(d, axis=1); u = d / r[:, None]
    ce = np.clip(-u[:, 2], 0, None); return float((P / len(ex) * (m + 1) / (2 * math.pi) * ce ** m * ce / r ** 2).sum() * A)


for case, e in (('R_s_e0_h10', 0), ('R_s_e1_h10', 1)):
    d = load(case)
    if not d: print(case, 'no data'); continue
    h = 0.30; out = []
    for x, y, g in d:
        r2 = x * x + y * y + h * h; ct = h / math.sqrt(r2); I = g / A * r2 / ct; out.append((math.degrees(math.acos(ct)), I))
    I0 = np.mean([I for t, I in out if t < 3]); out.sort()
    print(case, 'theta: measured I/I0 vs cos^%d' % (e + 1))
    print('   ', ' '.join(f'{t:4.1f}:{I / I0:.3f}/{math.cos(math.radians(t)) ** (e + 1):.3f}' for t, I in out[::3]))
for case, L, W in (('R_long_e0', 0.01, 0.10), ('R_wide_e0', 0.60, 0.01)):
    d = load(case)
    if not d: print(case, 'no data'); continue
    xs = sorted([(x, g / rect(L, W, 0.40, x, y)) for x, y, g in d if y == 0.0])
    ys = sorted([(y, g / rect(L, W, 0.40, x, y)) for x, y, g in d if x == 0.0 and y != 0.0])
    print(case, 'along x:', ' '.join(f'{x:+.3f}:{q:.3f}' for x, q in xs[::2]))
    print(' ' * len(case), 'along y:', ' '.join(f'{y:+.2f}:{q:.3f}' for y, q in ys[::2]))
