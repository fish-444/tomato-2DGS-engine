"""K13 (scenarios + tray attribution under rule 2, from K9 of the v1_tau reference) and K14 (tray32 v3 reference geometry under rule 2, from the v3 main), with only the tau map replaced."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]
SRC = ['dolgfish/static-tau-bundle-20261031', 'dolgfish/static-nine-bundle-20261028', 'dolgfish/static-normals-20261104', 'dolgfish/static-normals-v2-20261104', 'dolgfish/static-tau-rule2-20261105']
def meta(name, code, extra=()):
    return {'id': f'dolgfish/{name}', 'title': name, 'code_file': code, 'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
            'dataset_sources': SRC + list(extra), 'competition_sources': [], 'kernel_sources': []}
# ---- K13
s = (B / 'engine_static_v1tau_reference_20261104_01/kaggle_k9/k9_scen_smooth.py').read_text(encoding='utf-8')
s = s.replace('"""K9 (adopted version', '"""K13 = K9 with optics rule 2 (tau 0.07 on every non-stem surfel, stem tau 0; engine_static_tau_rule2_20261105_01). Original header: K9 (adopted version', 1)
s = s.replace("ND2 = Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent",
              "ND2 = Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent\nRULE2 = Path(glob.glob('/kaggle/input/**/stem__R3-1b.npz', recursive=True)[0]).parent")
s = s.replace("    q.normal = N; q.prims = (lambda P=P, k=k: (P, k)); INFO[pid] = dict(kept_raw_share=float(kept.mean()))",
              "    q.normal = N; q.prims = (lambda P=P, k=k: (P, k)); INFO[pid] = dict(kept_raw_share=float(kept.mean()))\n"
              "    z = np.load(RULE2 / f'stem__{pid}.npz'); assert np.array_equal(z['rows'], rows), pid\n"
              "    q.tau = np.where(z['stem_d1.5'].astype(bool), 0.0, CFG['tau_leaf']); INFO[pid]['stem_surfels_rule2'] = int(z['stem_d1.5'].sum())   # optics rule 2")
s = s.replace("res = dict(labels=['v1tau_smoothn_r2', 'light_model_v1_tau',", "res = dict(labels=['v1tau2_smoothn_r2', 'light_model_v1tau2',")
s = s.replace("jdump(res, OUT / 'k9_results.json')", "jdump(res, OUT / 'k13_results.json')").replace("log('K9_DONE')", "log('K13_DONE')")
assert 'RULE2' in s and 'stem_d1.5' in s
(HERE / 'kaggle_k13').mkdir(exist_ok=True); (HERE / 'kaggle_k13/k13_scen_rule2.py').write_text(s, encoding='utf-8'); json.dump(meta('static-rule2-k13-20261105', 'k13_scen_rule2.py'), open(HERE / 'kaggle_k13/kernel-metadata.json', 'w'))
# ---- K14 (tray32 v3 main + lib, tau map rule 2 in base())
V3 = B / 'static_tray32_v3_20261105_01'; LIB = (V3 / 'tray32_lib.py').read_text(encoding='utf-8'); MAIN = (V3 / 'tray32_main.py').read_text(encoding='utf-8')
old = "        _BASE[pid] = dict(c0=q.centre.copy(), n=n_, tu=tu_, tv=tv_, kept_raw_share=kept,"
assert MAIN.count(old) == 1
MAIN = MAIN.replace(old, "        if RULE2 is not None:                        # optics rule 2: tau 0.07 on every non-stem surfel, stem tau 0 (engine_static_tau_rule2_20261105_01)\n"
                         "            zs = np.load(RULE2 / f'stem__{pid}.npz'); assert np.array_equal(zs['rows'], q.s.index), pid; q.tau = np.where(zs['stem_d1.5'].astype(bool), 0.0, TAU)\n" + old)
MAIN = MAIN.replace("SMOOTH = a.smooth == 'on'; SUB = 'scene' if SMOOTH else 'regress'",
                    "SMOOTH = a.smooth == 'on'; SUB = 'scene' if SMOOTH else 'regress'\n_r2 = glob.glob('/kaggle/input/**/stem__R3-1b.npz', recursive=True); RULE2 = Path(_r2[0]).parent if _r2 else None")
assert "'''" not in LIB and "'''" not in MAIN
TEMPLATE = '''import sys
from pathlib import Path
OUTD = Path('/kaggle/working'); OUTD.mkdir(exist_ok=True)
LIB = r\'\'\'@LIB@\'\'\'
MAIN = r\'\'\'@MAIN@\'\'\'
(OUTD / 'tray32_lib.py').write_text(LIB, encoding='utf-8')
sys.argv = ['tray32_main.py', '--table', 'A', '--mode', 'scene', '--scenes', 'S25,S26,S27,S28', '--out', '/kaggle/working', '--lib', '/kaggle/working']
exec(compile(MAIN, 'tray32_main.py', 'exec'), {'__name__': '__main__', '__file__': '/kaggle/working/tray32_main.py'})
'''
(HERE / 'kaggle_k14').mkdir(exist_ok=True); (HERE / 'kaggle_k14/k14_tray32_rule2.py').write_text(TEMPLATE.replace('@LIB@', LIB).replace('@MAIN@', MAIN), encoding='utf-8')
json.dump(meta('static-rule2-k14-20261105', 'k14_tray32_rule2.py', ['dolgfish/static-tray32-input-20261104']), open(HERE / 'kaggle_k14/kernel-metadata.json', 'w'))
print('ok')
