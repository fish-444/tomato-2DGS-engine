"""Amendment 1 (option c): K16 / K17 / K18 = K12 (R3-1b only) / K13 / K14 with the pilot's stem mask switched to the geometric rule.
Only change: every stem-mask load maps stem_d* -> stem_rule2_geometric_stem_d* for R3-1b (masks already in stem__R3-1b.npz). Same dataset, same code otherwise."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
GEO = "(lambda _z, _p: {_k: (_z['stem_rule2_geometric_' + _k] if _p == 'R3-1b' and _k.startswith('stem_d') else _z[_k]) for _k in _z.files})(np.load(@ROOT@ / f'stem__{pid}.npz'), pid)"
JOBS = [('kaggle_k12a/k12_rule2.py', 'kaggle_k16', 'k16_rule2_pilot_geo.py', 'static-rule2-k16-20261105', 'REF'),
        ('kaggle_k13/k13_scen_rule2.py', 'kaggle_k17', 'k17_scen_rule2_geo.py', 'static-rule2-k17-20261105', 'RULE2'),
        ('kaggle_k14/k14_tray32_rule2.py', 'kaggle_k18', 'k18_tray32_rule2_geo.py', 'static-rule2-k18-20261105', 'RULE2')]
for src, d, code, kid, root in JOBS:
    s = (HERE / src).read_text(encoding='utf-8'); old = f"np.load({root} / f'stem__{{pid}}.npz')"; n = s.count(old); assert n >= 1, (src, n)
    s = s.replace(old, GEO.replace('@ROOT@', root))
    if 'k12' in src:
        s = re.sub(r"ap\.add_argument\('--plants', default='[^']*'\)", "ap.add_argument('--plants', default='R3-1b')", s); assert "default='R3-1b'" in s
        s = s.replace("log('K12_DONE')", "log('K16_DONE')")
    s = '# stem_rule_unified_geometric (amendment 1, option c): pilot stem = geometric d_stem rule\n' + s
    (HERE / d).mkdir(exist_ok=True); (HERE / d / code).write_text(s, encoding='utf-8')
    m = json.load(open(HERE / src.split('/')[0] / 'kernel-metadata.json')); m.update(id=f'dolgfish/{kid}', title=kid, code_file=code); json.dump(m, open(HERE / d / 'kernel-metadata.json', 'w'))
    print(d, 'replaced', n)
