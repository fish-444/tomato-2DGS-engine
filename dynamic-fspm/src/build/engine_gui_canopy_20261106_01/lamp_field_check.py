"""Lamp checks (CONTRACT.md §1, §4) for the GUI luminaire built from GroIMP AreaLights.
(a) emitted total: each half panel has AreaLight power P/2 with exponent 0 (intensity ∝ cos^(0+1)); GroIMP's AreaLight power is the total emitted
    power, and the analytic integral of the declared tray32 emitter (I0 = phi0 (m+1)/(2 pi), cos^m over the front half space) is reported per shape.
(b) field: the static tray32 v2 bar model (build/static_tray32_v2_20261104_01/CONTRACT.md: 0.60 m bar; flat 0.10 m panel or two 0.05 m wings at
    phi 30 / 45 deg; 0.0125 m grid = 384 points; cos^m, m = 1) evaluated at the engine's 16 empty-scene sensors (4 x 4, x, y in {-0.15, -0.05, 0.05,
    0.15} m, z = 0, 0.02 x 0.02 m, facing up, black) and compared with GroIMP sensors.csv (absorbed umol/s). Pass: every sensor within 5 %.
python -X utf8 lamp_field_check.py <kout dir> -> lamp_field_check.csv, prints the table."""
import csv, glob, json, math, sys
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent; P = 1200.0; m = 1.0; L = 0.60; SP = 0.0125


def emitters(shape, tilt_deg, h, x0):
    phi = {'flat': 0.0, 'V120': math.radians(30), 'V90': math.radians(45)}[shape]; a = math.radians(tilt_deg)
    ys = (np.arange(48) + 0.5) * SP - L / 2; pts, nrm = [], []
    for k in (-1, 1):
        us = (np.arange(4) + 0.5) * SP            # across the half (0.05 m) from the centre line outward
        for u in us:
            x, z = k * u * math.cos(phi), u * math.sin(phi)   # wing rises outward by phi
            n = (k * math.sin(phi), -math.cos(phi))
            xr, zr = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)   # rotY(a): x' = x cos + z sin, z' = -x sin + z cos
            nr = (n[0] * math.cos(a) + n[1] * math.sin(a), -n[0] * math.sin(a) + n[1] * math.cos(a))
            for y in ys: pts.append((xr, y, zr)); nrm.append((nr[0], 0.0, nr[1]))
    pts, nrm = np.array(pts), np.array(nrm)
    # lowest point of the luminaire at height h above z = 0 (the 1-plant base has plantZ = 0); corners as in the model
    zmin = min(0.0, min(-(k * 0.05 * math.cos(phi)) * math.sin(a) + 0.05 * math.sin(phi) * math.cos(a) for k in (-1, 1)))
    pts[:, 2] += h - zmin; pts[:, 0] += x0
    return pts, nrm


def field(shape, tilt, h, x0, sensors):
    pts, nrm = emitters(shape, tilt, h, x0); phi0 = P / len(pts); I0 = phi0 * (m + 1) / (2 * math.pi); out = []
    for (sx, sy) in sensors:
        d = np.array([sx, sy, 0.0]) - pts; r = np.linalg.norm(d, axis=1); u = d / r[:, None]
        ce = np.clip((u * nrm).sum(1), 0, None); cr = np.clip(-u[:, 2], 0, None)
        out.append(float((I0 * ce ** m * cr / r ** 2).sum() * 0.0004))
    return out


def main():
    kout = Path(sys.argv[1]); rows = []
    sens = [(-0.15 + (j % 4) * 0.1, -0.15 + (j // 4) * 0.1) for j in range(16)]
    for st in sorted(kout.glob('*/out/LF*/status.json')):
        s = json.load(open(st)); g = s['gui_test']['settings']; d = st.parent
        sf = d / 'gui_sensors.csv' if (d / 'gui_sensors.csv').exists() else d / 'sensors.csv'
        if not sf.exists(): rows.append(dict(test=d.name, error='no sensors file')); continue
        meas = {(int(r['grid_row']), int(r['grid_col'])): float(r['absorbed_umol_s']) for r in csv.DictReader(open(sf))}
        shape = g['LAMP_SHAPE'].strip('"'); tilt = float(g['LAMP_TILT_DEG']); h = float(g['LAMP_HEIGHT_CM']) / 100
        ref = field(shape, tilt, h, 0.0, sens)
        for j, (sx, sy) in enumerate(sens):
            mv = meas.get((j // 4 + 1, j % 4 + 1)); rows.append(dict(test=d.name, shape=shape, tilt_deg=tilt, x=sx, y=sy, groimp=mv, static_ref=ref[j], rel_diff=(mv / ref[j] - 1) if mv and ref[j] else None))
    with open(H / 'lamp_field_check.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    for t in sorted({r['test'] for r in rows}):
        rr = [r for r in rows if r['test'] == t and r.get('rel_diff') is not None]
        if rr: print(t, 'max |diff| %.2f %%' % (100 * max(abs(r['rel_diff']) for r in rr)), 'median %.2f %%' % (100 * float(np.median([r['rel_diff'] for r in rr]))), 'PASS' if max(abs(r['rel_diff']) for r in rr) <= .05 else 'FAIL')
    # (a) emitted total of the declared emitter: integral of I0 cos^m over the front half space = phi0 per point -> P
    print('declared emitter total = P x (m+1)/(2 pi) x 2 pi/(m+1) = P (exact); GroIMP AreaLight power per half = P/2 -> total', P)


if __name__ == '__main__':
    main()
