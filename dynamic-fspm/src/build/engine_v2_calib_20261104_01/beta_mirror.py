"""Python mirror of the Java Beta ppf in v2_patch.py (NR betacf + bisection, embedded ln B(a,b)); checked against scipy."""
import math
A, B = 5.024584763293008, 0.9754152367069917
LNB = math.lgamma(A) + math.lgamma(B) - math.lgamma(A + B)

def betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1; c = 1.0; d = 1 - qab * x / qap
    if abs(d) < 1e-300: d = 1e-300
    d = 1 / d; h = d
    for m in range(1, 301):
        m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1e-300 if abs(d) < 1e-300 else d; c = 1 + aa / c; c = 1e-300 if abs(c) < 1e-300 else c; d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1e-300 if abs(d) < 1e-300 else d; c = 1 + aa / c; c = 1e-300 if abs(c) < 1e-300 else c; d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 3e-16: break
    return h

def ibeta(x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(A * math.log(x) + B * math.log(1 - x) - LNB)
    return bt * betacf(A, B, x) / A if x < (A + 1) / (A + B + 2) else 1 - bt * betacf(B, A, 1 - x) / B

def ppf_mm(u):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ibeta(mid) < u: lo = mid
        else: hi = mid
    return 2.0 + 2.0 * 0.5 * (lo + hi)

if __name__ == '__main__':
    import numpy as np
    from scipy.stats import beta
    us = np.concatenate([np.random.default_rng(0).random(3000), np.random.default_rng(1).random(1000) ** 8, [1e-9, 0.5, 0.999999]])
    err = max(abs(ppf_mm(u) - (2 + 2 * beta.ppf(u, A, B))) for u in us); print('LNB', repr(LNB), 'max err mm', err, 'median', ppf_mm(0.5))
