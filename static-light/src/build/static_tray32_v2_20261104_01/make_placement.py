"""Placement tables A and B of the 32-plant tray scenes (CONTRACT.md section 2, seed 20261104). Declared before any light run; not changed afterwards.
Slots: tray t = 1..4 (x order), row r = 1..4 along y, column c = 1..2 across x. Position classes: tray outer (1, 4) / inner (2, 3); row end (1, 4) / middle (2, 3).
Each table: every plant 2 copies + 4 plants (chosen by the seed) a 3rd copy = 32 copies; random azimuth per copy. Constraints across A and B together: every plant has >= 1 copy in an outer
and >= 1 in an inner tray, >= 1 in an end row and >= 1 in a middle row; in one table no plant has two copies in neighbouring slots (8-neighbourhood) of the same tray.
Rejection sampling with numpy default_rng(seed 20261104); the first accepted pair is used. python -X utf8 make_placement.py"""
import csv, itertools, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; SEED = 20261104
PLANTS = ['R3-1b', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 'n_06', 't2_02', 't2_04', 't2_05', 'e_01', 'e_04', 'e_05', 'e_07']
SLOTS = [(t, r, c) for t in range(1, 5) for r in range(1, 5) for c in range(1, 3)]


def neighbours(a, b):
    return a[0] == b[0] and a != b and abs(a[1] - b[1]) <= 1 and abs(a[2] - b[2]) <= 1


def table(rng):
    for _ in range(100000):
        extra = rng.choice(len(PLANTS), 4, replace=False); order = [p for i, p in enumerate(PLANTS) for _ in range(3 if i in extra else 2)]
        perm = rng.permutation(32); slot_of = [SLOTS[i] for i in perm]; ok = True
        by = {}
        for p, s in zip(order, slot_of):
            by.setdefault(p, []).append(s)
        for p, ss in by.items():
            if any(neighbours(a, b) for a, b in itertools.combinations(ss, 2)):
                ok = False; break
        if ok:
            return [dict(plant=p, slot=s) for p, s in zip(order, slot_of)], sorted(PLANTS[i] for i in extra)
    raise RuntimeError('no table found')


def classes(rows):
    out = {}
    for x in rows:
        t, r, c = x['slot']; d = out.setdefault(x['plant'], dict(outer=0, inner=0, end=0, middle=0))
        d['outer' if t in (1, 4) else 'inner'] += 1; d['end' if r in (1, 4) else 'middle'] += 1
    return out


rng = np.random.default_rng(SEED)
for attempt in range(100000):
    A, xa = table(rng); B, xb = table(rng); ca, cb = classes(A), classes(B)
    if all(ca[p][k] + cb[p][k] >= 1 for p in PLANTS for k in ('outer', 'inner', 'end', 'middle')):
        break
else:
    raise RuntimeError('no A/B pair found')
az = rng.uniform(0, 360, size=64)
meta = dict(seed=SEED, attempts=attempt + 1, third_copy_A=xa, third_copy_B=xb)
for name, tab, off in (('A', A, 0), ('B', B, 32)):
    cnt = {}
    with open(HERE / f'placement_{name}.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['copy_id', 'plant', 'copy_index', 'tray', 'row', 'column', 'tray_class', 'row_class', 'azimuth_deg'])
        for i, x in enumerate(sorted(tab, key=lambda x: x['slot'])):
            p = x['plant']; cnt[p] = cnt.get(p, 0) + 1; t, r, c = x['slot']
            w.writerow([f'{name}{i + 1:02d}', p, cnt[p], t, r, c, 'outer' if t in (1, 4) else 'inner', 'end' if r in (1, 4) else 'middle', f'{az[off + i]:.3f}'])
json.dump(meta, open(HERE / 'placement_meta.json', 'w'), indent=1)
print(meta)
