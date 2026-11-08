"""GUI height fit (CONTRACT Amendment 4): median height of the 8 plants at engine day 71 per grid case, linear interpolation to 150 mm.
python -X utf8 gui_height_fit.py [prefix ...]  (default HA_ HR_ and confirm cases CA_ CR_)"""
import csv, glob, statistics as st, sys
from pathlib import Path

H = Path(__file__).resolve().parent; T = 150.0


def day71(case):
    fs = glob.glob(str(H / 'kout' / '*' / 'out' / case / 'daily.csv'))
    if not fs: return None
    rows = [r for r in csv.reader(l for l in open(fs[0], encoding='utf-8') if not l.startswith('#'))][1:]
    h = [float(r[5]) for r in rows if r[0] == '71' and r[2] != 'CANOPY']
    la = [float(r[3]) for r in rows if r[0] == '71' and r[2] != 'CANOPY']
    return (st.median(h), min(h), max(h), st.median(la)) if h else None


for arm in ('A', 'R'):
    pts = []
    for f in sorted(glob.glob(str(H / 'kout' / '*' / 'out' / f'[HC]{arm}_*'))):
        case = Path(f).name; m = float(case.split('_', 1)[1].replace('p', '.')); v = day71(case)
        if v: pts.append((m, *v)); print(arm, case, 'mult %.4f' % m, 'height median %.1f (min %.1f max %.1f) leaf area median %.1f cm2' % v)
    pts.sort()
    lo = [p for p in pts if p[1] < T]; hi = [p for p in pts if p[1] > T]
    if lo and hi:
        a = max(lo, key=lambda p: p[0]); b = min(hi, key=lambda p: p[0])
        m = a[0] + (T - a[1]) * (b[0] - a[0]) / (b[1] - a[1]); print(arm, 'interpolated multiplier for 150 mm: %.4f' % m)
    else:
        print(arm, 'not bracketed')
