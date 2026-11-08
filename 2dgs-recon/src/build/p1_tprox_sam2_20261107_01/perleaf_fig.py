"""Per-leaf accuracy of T0 and the chosen settings (already scored arms only) + figure T0 | A | A+B (top view, changed surfels)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; D = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(D))
import score as S
from tomato3d.organs import oversegment as OS, skeleton_r31b_eval as EV
Bs = OS.load_base(); si, hl = EV.human_projections(Bs); sc = json.load(open(D / 'score.json')); K = D / 'kout1'
tags = {a: v['tag'] for a, v in sc['arms'].items()}; per = {}
for a, t in tags.items():
    g = S.per_proj(np.load(K / f'labels_{t}.npz'), Bs, si); per[a] = {str(L): round(float((g[hl == L] == L).mean()), 3) for L in range(1, 9)}
json.dump(per, open(D / 'per_leaf.json', 'w'), indent=1); print(json.dumps(per))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
Z = np.load(K / 'labels_T0.npz'); X = np.load(ROOT / 'build/p1_skeleton_layer_20261101_01/r8pc/R3-1b/20261101T133038/kaggle_data/r8pc_bundle.npz')['xyz'] * 1000; L0 = Z['label']
cm = plt.get_cmap('tab10'); fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
for a_, arm in zip(ax, ('T0', 'A', 'AB')):
    L = np.load(K / f"labels_{tags[arm]}.npz")['label']; m = L > 0
    a_.scatter(X[~m, 0], X[~m, 1], s=0.3, c='0.85'); a_.scatter(X[m, 0], X[m, 1], s=0.5, c=[cm(int(k) % 10) for k in L[m]])
    if arm != 'T0':
        ch = L != L0; a_.scatter(X[ch, 0], X[ch, 1], s=2, facecolors='none', edgecolors='k', linewidths=0.3)
    a_.set_aspect('equal'); a_.set_title(f"{arm} ({tags[arm]})" + ('' if arm == 'T0' else f' — changed {int((L != L0).sum())} (black)')); a_.set_xlabel('mm')
fig.tight_layout(); fig.savefig(D / 'fig_R3-1b_T0_A_AB.png', dpi=110); print('fig')
