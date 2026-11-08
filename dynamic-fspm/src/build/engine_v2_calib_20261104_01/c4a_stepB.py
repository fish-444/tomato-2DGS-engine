"""C4A Step B search (CONTRACT Amendment 17): evaluations of f_SLA (arms C4AB<f>), median plant leaflet area at day 71 over 16 seeds.
Rule: x1 = 1.15 (36.5 / A0)^(1/2.60); then log-space secant on the last two points until the target 36.5 is bracketed
(one point with only one evaluation: slope 2.60); once bracketed, bisection in log space on the tightest bracket.
Stop when |A - 36.5| <= 1 cm2 or after 8 evaluations. python -X utf8 c4a_stepB.py -> prints path + next value; writes C4A_area_fit.csv."""
import csv, math, statistics as st
import analyze as A

T, TOL, NMAX, SLOPE = 36.5, 1.0, 8, 2.60
R = A.load()
ev = []
for k in sorted({k.rsplit('_m', 1)[0] for k in R if k.startswith('C4AB')}, key=lambda a: float(a[4:])):
    M = A.arm(R, k)
    if len(M) < 16: print('incomplete', k, len(M)); continue
    ev.append(dict(arm=k, f=float(k[4:]), n=len(M), leaf_area_median=A.q([m['leaf_area_cm2'] for m in M.values()], .5),
                   leaf_area_p05=A.q([m['leaf_area_cm2'] for m in M.values()], .05), leaf_area_p95=A.q([m['leaf_area_cm2'] for m in M.values()], .95),
                   height_mean=st.mean(m['height_mm'] for m in M.values()), dm_median=A.q([m['dm_g'] for m in M.values()], .5)))
order = []  # evaluation order = the order the arms were run (recorded in C4A_area_fit_order.txt, one arm per line)
try:
    order = [l.strip() for l in open('C4A_area_fit_order.txt') if l.strip()]
except FileNotFoundError:
    pass
M0 = A.arm(R, 'V22b')   # step 0 = the base (V22b is C4AB1.15 by construction; Amendment 17 A0)
base = dict(arm='V22b', f=1.15, n=len(M0), leaf_area_median=A.q([m['leaf_area_cm2'] for m in M0.values()], .5), leaf_area_p05=A.q([m['leaf_area_cm2'] for m in M0.values()], .05),
            leaf_area_p95=A.q([m['leaf_area_cm2'] for m in M0.values()], .95), height_mean=st.mean(m['height_mm'] for m in M0.values()), dm_median=A.q([m['dm_g'] for m in M0.values()], .5))
by = {e['arm']: e for e in ev}; path = [base] + [by[a] for a in order if a in by]
with open('C4A_area_fit.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, ['step', 'arm', 'f', 'n', 'leaf_area_median', 'leaf_area_p05', 'leaf_area_p95', 'height_mean', 'dm_median', 'within_tol'])
    w.writeheader()
    for i, e in enumerate(path): w.writerow(dict(step=i, within_tol=abs(e['leaf_area_median'] - T) <= TOL, **e))
for i, e in enumerate(path): print(i, e['arm'], round(e['leaf_area_median'], 2), 'height', round(e['height_mean'], 1))
if not path: raise SystemExit('no evaluation yet')
last = path[-1]
if abs(last['leaf_area_median'] - T) <= TOL: print('STOP: within tolerance at f =', last['f']); raise SystemExit
if len(path) - 1 >= NMAX: print('STOP: 8 evaluations, closest', min(path, key=lambda e: abs(e['leaf_area_median'] - T))['f']); raise SystemExit
lo = [e for e in path if e['leaf_area_median'] < T]; hi = [e for e in path if e['leaf_area_median'] > T]
if lo and hi:
    a = max(lo, key=lambda e: e['f']); b = min(hi, key=lambda e: e['f'])
    nxt = math.exp((math.log(a['f']) + math.log(b['f'])) / 2); how = f"bisection between {a['f']} and {b['f']}"
elif len(path) == 1:
    nxt = last['f'] * (T / last['leaf_area_median']) ** (1 / SLOPE); how = 'slope 2.60 from one point'
else:
    p, q = path[-2], path[-1]
    s = (math.log(q['leaf_area_median']) - math.log(p['leaf_area_median'])) / (math.log(q['f']) - math.log(p['f']))
    nxt = math.exp(math.log(q['f']) + (math.log(T) - math.log(q['leaf_area_median'])) / s); how = f'log secant (slope {s:.3f})'
print(f'NEXT f = {nxt:.4f} ({how})')
