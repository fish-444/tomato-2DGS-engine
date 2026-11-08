"""Amendment A3: grade T0 and A of the dz10 versions with the user's held-out picks (skeleton_io.heldout_agreement), pooled McNemar.
usage (project root): python -X utf8 build/p1_tprox_sam2_20261107_01/grade_picks.py build/p1_tprox_sam2_20261107_01/kout_dz10"""
from math import comb
from pathlib import Path
import csv, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; D = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from tomato3d.organs import skeleton_dz10_labels as DZ, skeleton_io as SIO
DZD = ROOT / 'build/p1_dz10_labels_20261101_01'; PIDS = ('e_01', 'e_04', 'e_05', 'e_07', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 't2_02', 't2_04', 't2_05')


def main(kout):
    kout = Path(kout); rows = []; disc = dict(t0_only=0, a_only=0); pool = dict(T0=[0, 0], A=[0, 0]); per = {}
    for pid in PIDS:
        G = json.load(open(sorted(DZD.glob(f'grade_{pid}_*.json'))[0], encoding='utf-8')); sk, v = DZ.live(pid); assert v == G['skelver'], pid; picks = sk.get('heldout', [])
        run = [d for d in (DZD / f'{pid}_{v}' / 'runs' / pid).iterdir() if d.is_dir() and d.name[:8].isdigit()][0]; lm = {int(k): u for k, u in json.load(open(run / 'bundle_meta.json', encoding='utf-8'))['leaf_map'].items()}
        res = {}; hits = {}
        for arm in ('T0', 'A'):
            lab = np.load(kout / f'labels_{arm}_{pid}.npz')['label'].astype(int)
            uid_of = lambda s, lab=lab: (lm.get(int(lab[int(s)])) if 0 <= int(s) < len(lab) and lab[int(s)] > 0 else None)  # noqa: E731
            res[arm] = SIO.heldout_agreement(picks, uid_of); hits[arm] = [uid_of(p['surfel']) == p['leaf_uid'] for p in picks if p.get('leaf_uid')]
            pool[arm][0] += res[arm]['agree']; pool[arm][1] += res[arm]['n']
        stored = G['routes']['tprox']; ok = (res['T0']['agree'], res['T0']['n']) == (stored['agree'], stored['n'])
        for a, b in zip(hits['T0'], hits['A']):
            disc['t0_only'] += int(a and not b); disc['a_only'] += int(b and not a)
        per[pid] = dict(T0=res['T0'], A=res['A'], T0_reproduces_stored_grade=ok)
        rows.append([pid, res['T0']['agree'], res['T0']['n'], res['T0']['grade'], res['A']['agree'], res['A']['n'], res['A']['grade'], ok])
        print(pid, 'T0', res['T0']['agree'], '/', res['T0']['n'], res['T0']['grade'], '| A', res['A']['agree'], '/', res['A']['n'], res['A']['grade'], '| stored T0 reproduced', ok)
    b, c = disc['t0_only'], disc['a_only']; m = b + c; p = min(1.0, 2 * sum(comb(m, k) for k in range(0, min(b, c) + 1)) / 2 ** m) if m else 1.0
    out = dict(per_plant=per, pooled={k: dict(agree=v[0], n=v[1], rate=round(v[0] / v[1], 4), wilson95=[round(x, 4) for x in SIO.wilson(*v)]) for k, v in pool.items()}, discordant=disc, mcnemar_exact_p=round(p, 4),
               a_better_on_picks=bool(pool['A'][0] / pool['A'][1] > pool['T0'][0] / pool['T0'][1] and p < 0.05))
    json.dump(out, open(D / 'heldout_T0_vs_A.json', 'w', encoding='utf-8'), indent=1, default=float)
    with open(D / 'heldout_T0_vs_A.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['plant', 'T0_agree', 'T0_n', 'T0_grade', 'A_agree', 'A_n', 'A_grade', 'T0_reproduces_stored']); w.writerows(rows)
    print(json.dumps({k: out[k] for k in ('pooled', 'discordant', 'mcnemar_exact_p', 'a_better_on_picks')}))


if __name__ == '__main__':
    main(sys.argv[1])
