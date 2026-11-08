"""Make tray32_main.py v3 from the v2 copy: smoothing switch (v1tau_smoothn_r2), stable seed keys, conv at 30 cm, output dirs per smoothing state. Prints each replacement."""
from pathlib import Path
p = Path(__file__).resolve().parent / 'tray32_main.py'; s = p.read_text(encoding='utf-8')
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old)); s = s.replace(old, new); print('ok:', old[:70])
rep("ap.add_argument('--max-copies', type=int, default=32);",
    "ap.add_argument('--smooth', default='on'); ap.add_argument('--nd2'); ap.add_argument('--radius', type=float, default=2.0)   # v3: v1tau_smoothn_r2 (on) / v1_tau raw normals (off = v2 regression)\nap.add_argument('--max-copies', type=int, default=32);")
rep("sys.path.insert(0, str(TAU_ROOT)); sys.path.insert(0, str(Path(a.lib) if a.lib else OUT))",
    "ND2 = Path(a.nd2) if a.nd2 else Path(glob.glob('/kaggle/input/**/normals_v2_config.json', recursive=True)[0]).parent\nsys.path.insert(0, str(TAU_ROOT)); sys.path.insert(0, str(Path(a.lib) if a.lib else OUT)); sys.path.insert(0, str(ND2))")
rep("import tray32_lib as T", "import tray32_lib as T\nimport normals as NM\nSMOOTH = a.smooth == 'on'; SUB = 'scene' if SMOOTH else 'regress'")
rep("_BASE[pid] = dict(c0=q.centre.copy(), n=q.normal.copy(), tu=q.s.tu.copy(), tv=q.s.tv.copy(),",
    "n_, tu_, tv_ = q.normal.copy(), q.s.tu.copy(), q.s.tv.copy(); kept = None\n"
    "        if SMOOTH:                                   # normal_mode_v2 smooth_all_r2: receiver normal and disc orientation (normal_smoothing_v2 CONTRACT section 1)\n"
    "            sets = np.where(leaf >= 1, leaf, np.where(q.axis == 1, -10, np.where(q.axis == 2, -20, -30))) if pid == 'R3-1b' else np.where(leaf >= 1, leaf, -30)\n"
    "            op = read_ply(obs.ply_path(pid))['opacity'][q.s.index].astype(float)\n"
    "            n_, kept_m, _ = NM.smooth(q.s.centre * 1000.0, q.s.normal, q.A * op, sets, a.radius); tu_, tv_ = NM.reframe(n_, q.s.tu); kept = float(kept_m.mean())\n"
    "        _BASE[pid] = dict(c0=q.centre.copy(), n=n_, tu=tu_, tv=tv_, kept_raw_share=kept,")
rep("fp = OUT / 'scene' / f'{a.table}_{sc[\"sid\"]}.json'", "fp = OUT / SUB / f'{a.table}_{sc[\"sid\"]}.json'")
rep("E = T.emitters(fcs); k = int(sc['sid'][1:])", "E = T.emitters(fcs); k = sc['seed_k']", count=2)
rep("fp = OUT / 'iso' / f'{a.table}_{sc[\"sid\"]}.json'", "fp = OUT / ('iso' if SMOOTH else 'iso_regress') / f'{a.table}_{sc[\"sid\"]}.json'")
rep("sc = [s for s in T.scene_list() if s['p'] == p and s['shape'] == 'flat' and s['pos'] == 'trays' and s['height'] == 0.40][0]; asm = assemble(p)",
    "sc = [s for s in T.scene_list() if s['p'] == p and s['shape'] == 'flat' and s['pos'] == 'trays' and s['height'] == 0.30][0]; asm = assemble(p)   # v3: C4 on the 30 cm reference geometry")
s = s.replace("jdump(dict(scene=sc, table=a.table, receivers_per_copy=a.recv_n, ke=a.ke, emitted=emitted,", "jdump(dict(scene=sc, table=a.table, smoothing=('smooth_all_r%g' % a.radius) if SMOOTH else 'off', receivers_per_copy=a.recv_n, ke=a.ke, emitted=emitted,")
s = s.replace("OUT / f'progress_{a.mode}_{a.table}.log'", "OUT / f'progress_{a.mode}_{a.table}_{SUB if \"SUB\" in globals() else \"x\"}.log'")
s = s.replace("OUT / f'SUMMARY_{a.mode}_{a.table}.json'", "OUT / f'SUMMARY_{a.mode}_{a.table}_{SUB}.json'")
p.write_text(s, encoding='utf-8'); print('patched')
