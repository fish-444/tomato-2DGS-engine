"""MTG build (build/mtg_build_20261102_01, CONTRACT §5): B-model import round trip on Kaggle CPU. Work copy = the B-model session's geo7
work_model.tar.bin (hash in MANIFEST; measured conventions A4). Environment items (ADAPTERS.md E-*): -Duser.home=<run dir>, -Xverify:none,
output folders created, single plant (A-layout 1/1/0/0, placement rot 0). Export_Geometry_Day0 runs every scenario CSV in the run folder
and writes <name>.csv.obj + <name>.csv_leaves.csv. No light, no growth."""
import glob, hashlib, json, os, re, shutil, subprocess, tarfile, time
W = '/kaggle/working/out'; S = '/tmp/bm'
os.makedirs(f'{W}/logs', exist_ok=True); os.makedirs(S, exist_ok=True)
LOG = open(f'{W}/run_log.txt', 'a', encoding='utf-8', buffering=1)


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')


def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()


say('=== start', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
man = sorted(glob.glob('/kaggle/input/**/MTGB_IMPORT_MANIFEST.json', recursive=True)); INP = os.path.dirname(man[0]); M = json.load(open(man[0]))
for n, h in M['inputs'].items():
    assert sha(f'{INP}/{n}') == h, f'{n} hash mismatch'
core = os.path.realpath(sorted(glob.glob('/kaggle/input/**/core.jar', recursive=True))[0]); assert sha(core) == M['core_sha256']
subprocess.run(['cp', '-a', os.path.dirname(core), f'{S}/groimp'], check=True); core = f'{S}/groimp/core.jar'
with tarfile.open(f'{INP}/scenarios.tar.bin') as t:
    t.extractall(f'{S}/scn')
OUT_DIRS = ['output/rendering_output', 'output/sensor_output', 'output/dynamic_output/crop', 'output/dynamic_output/plant', 'output/dynamic_output/organ',
            'output/static_output/crop', 'output/static_output/plant', 'output/static_output/organ']


def tree(dst, scenarios, switches):
    x = f'{S}/x_{os.path.basename(dst)}'
    with tarfile.open(f'{INP}/work_model.tar.bin') as t:
        t.extractall(x)
    shutil.copytree(f'{x}/model', dst)
    for k in ('static_input', 'dynamic_input'):
        for n in os.listdir(f'{dst}/input/{k}'):
            os.remove(f'{dst}/input/{k}/{n}')
    for src, name in scenarios:
        shutil.copy2(src, f'{dst}/input/static_input/{name}')
    os.makedirs(f'{dst}/input/placement'); open(f'{dst}/input/placement/placement.csv', 'w').write('plant_number,x_m,y_m,rot_deg\n1,0,0,0\n')
    p = f'{dst}/param/parameters.rgg'; s = open(p, 'rb').read()
    for k, v in switches.items():
        s2 = re.sub(rb'(static\s+\w+\s+' + k.encode() + rb'\s*=\s*)[^;]+;', lambda m: m.group(1) + v.encode() + b';', s, count=1)
        assert s2 != s, k; s = s2
    open(p, 'wb').write(s)
    for d in OUT_DIRS:
        os.makedirs(f'{dst}/{d}', exist_ok=True)
    return dst


def cli(tag, cwd, calls=(), done=None, wait_open=300, wait_call=6000):
    env = dict(os.environ); env['JDK_JAVA_OPTIONS'] = f'-Djava.awt.headless=true -XX:ActiveProcessorCount=1 -Duser.home={cwd} -Xverify:none'
    log = f'{W}/logs/{tag}.log'; f = open(log, 'wb')
    p = subprocess.Popen(['java', '-Dfile.encoding=UTF-8', '-jar', core, '--headless', '-a', 'cli'], stdin=subprocess.PIPE, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    rd = lambda: open(log, encoding='utf-8', errors='replace').read()
    r = dict(tag=tag); t1 = time.time()
    try:
        end = time.time() + 120
        while '[base]>>' not in rd():
            assert time.time() < end and p.poll() is None, 'no cli prompt'
            time.sleep(0.5)
        off = len(rd()); p.stdin.write(f'$cd {cwd}\n$open project.gs\n'); p.stdin.flush(); end = time.time() + wait_open
        while True:
            t = rd()[off:]
            if 'Welcome #' in t or 'not found' in t or 'SEVERE' in t:
                break
            assert time.time() < end and p.poll() is None, 'open timeout'
            time.sleep(0.5)
        time.sleep(5); t = rd()[off:]
        r.update(compiled='not found' not in t, init_ok='Welcome #' in t and 'SEVERE' not in t, open_s=round(time.time() - t1, 1), errors=re.findall(r'Caused by: [^\n]*', t)[:3])
        for c in calls:
            ts = time.time(); off = len(rd()); p.stdin.write(c + '\n'); p.stdin.flush(); end = time.time() + wait_call
            while not done(rd()[off:]):
                t = rd()[off:]
                assert 'SEVERE' not in t and 'Exception' not in t, 'error: ' + t[-1500:]
                assert time.time() < end and p.poll() is None, f'{c} timeout'
                time.sleep(1)
            r[f'{c}_s'] = round(time.time() - ts, 1)
        r['status'] = 'ok'
    except Exception as e:
        r['status'] = 'failed'; r['error'] = f'{type(e).__name__}: {e}'[:2500]
    if p.poll() is None:
        p.kill(); p.wait()
    f.close()
    say('CASE', tag, json.dumps({k: v for k, v in r.items() if k != 'error'})[:700], r.get('error', '')[:600])
    return r


R = dict(cases={}, manifest=M)
single = dict(isStatic='true', NR_PLANTS='1', NR_ROWS='1', NR_BORDER_ROWS='0', NR_BORDER_PLANTS_IN_ROW='0')
names = sorted(n for n in os.listdir(f'{S}/scn') if n.endswith('.csv'))
# batches of <= 40 scenarios per JVM (the published end-of-run code indexes the file list; each batch is a fresh process)
for bi in range(0, len(names), 40):
    batch = names[bi:bi + 40]; g = tree(f'{S}/geo{bi}', [(f'{S}/scn/{n}', n) for n in batch], single)
    R['cases'][f'GEO{bi}'] = cli(f'GEO{bi}', g, ['Export_Geometry_Day0'], lambda t: 'BM_GEO_ALL_DONE' in t)
    R['cases'][f'GEO{bi}']['geo_done_n'] = len(re.findall(r'BM_GEO_DONE ', open(f'{W}/logs/GEO{bi}.log', encoding='utf-8', errors='replace').read()))
    if os.path.isdir(f'{g}/output/rendering_output'):
        os.makedirs(f'{W}/geometry', exist_ok=True)
        for fn in os.listdir(f'{g}/output/rendering_output'):
            shutil.copy2(f'{g}/output/rendering_output/{fn}', f'{W}/geometry/{fn}')
    json.dump(R, open(f'{W}/summary.json', 'w'), indent=1)
say('=== done', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
