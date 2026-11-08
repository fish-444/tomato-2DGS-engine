"""Diagnostic only (no score): why A covers < 1 % — size of union-find components per share."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep as P0
from tomato3d.organs import userlabel as UL

Z = np.load(P0.RUN / 'kaggle_data/r8pc_bundle.npz'); pl = UL.load_plant(); b = np.nonzero(pl['blade'])[0]; W = np.load(P0.SR.FIRST / 'contributions.npz')['W']; vis_all = W >= P0.SR.P['w_min']; st1 = Z['stage1'].astype(int)
views = [f.split('_')[1] for f in pl['views']]; prj = P0.SR.projections(pl, b, views); sets = []
for vid in views:
    p = prj[vid]; vis = vis_all[b, p['k']] & p['inside']; j = json.load(open(P0.S2.PROPOSALS / f'{P0.AP.view_file(vid)[:-4]}.json', encoding='utf-8'))
    for inst in j['instances']:
        m = P0.px.rle_decode(inst['visible_segmentation']); mem = b[vis & m[p['wi'], p['ui']]]
        if len(mem) >= 5: sets.append(np.unique(mem))
sz = np.array([len(s) for s in sets]); print('set size pct', np.percentile(sz, [5, 25, 50, 75, 95]).round(), 'sets with seeds of >1 leaf', round(float(np.mean([len(set(st1[s][st1[s] > 0].tolist())) > 1 for s in sets])), 3))
inv = {}
for si, s in enumerate(sets):
    for r in s.tolist(): inv.setdefault(r, []).append(si)
cand = set()
for lst in inv.values():
    for a in range(len(lst)):
        for c in range(a + 1, len(lst)): cand.add((lst[a], lst[c]))
for share in (0.4, 0.5, 0.6):
    uf = P0.UF(len(sets))
    for a, c in cand:
        if len(np.intersect1d(sets[a], sets[c], assume_unique=True)) / max(len(sets[a]), len(sets[c])) >= share: uf.u(a, c)
    roots = Counter(uf.f(i) for i in range(len(sets))); print(share, 'components', len(roots), 'largest', [n for _, n in roots.most_common(4)])
