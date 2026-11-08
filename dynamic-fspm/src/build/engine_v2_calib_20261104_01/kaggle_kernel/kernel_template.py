"""Engine v2 calibration on Kaggle CPU (build/engine_v2_calib_20261104_01/CONTRACT.md). Internet off, no secrets.
Inputs: dolgfish/engv2-<batch>-20261104 (ENGV2_MANIFEST.json, runs.tar.gz, code.tar.gz), dolgfish/tomato-dreftray-input-20261031 (repo.tar.gz = frozen DEV
tools, triangle wheel), dolgfish/tomato-h-input (GroIMP 2.2.1). Jobs = case names, 4 at a time, one JVM per run. Outputs /kaggle/working/out/<case>/."""
import glob, hashlib, json, os, shutil, subprocess, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
JOBS = @JOBS@
PARALLEL = 4
WORK = '/kaggle/working'; SCR = '/tmp/ev'; OUT = f'{WORK}/out'
os.makedirs(SCR, exist_ok=True); os.makedirs(OUT, exist_ok=True)
LOG = open(f'{WORK}/run_log.txt', 'a', encoding='utf-8', buffering=1)
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n')
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''): h.update(c)
    return h.hexdigest()
say('=== start', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), len(JOBS), 'jobs')
mdv = sorted(glob.glob('/kaggle/input/**/ENGV2_MANIFEST.json', recursive=True))[0]; MD = json.load(open(mdv)); IDV = os.path.dirname(mdv)
mdt = sorted(glob.glob('/kaggle/input/**/DREFTRAY_MANIFEST.json', recursive=True))[0]; MT = json.load(open(mdt)); IDT = os.path.dirname(mdt)
def unpack(folder, man, name):
    tgz = f'{folder}/{name}.tar.gz'
    if os.path.exists(tgz):
        assert sha256(tgz) == man['tarballs'][f'{name}.tar.gz']['sha256'], f'{name} hash mismatch'
        with tarfile.open(tgz) as t: t.extractall(SCR, filter='data')
    else:
        shutil.copytree(f'{folder}/{name}', SCR, dirs_exist_ok=True)
unpack(IDT, MT, 'repo'); unpack(IDV, MD, 'runs'); unpack(IDV, MD, 'code')
repo = f'{SCR}/repo'; dvcode = f'{SCR}/evcode'
for n, h in MD['engine_code'].items():
    p = f'{repo}/tools/{n}'
    assert os.path.exists(p) and sha256(p) == h, f'engine code differs: {n}'
core = os.path.realpath(sorted(glob.glob('/kaggle/input/**/core.jar', recursive=True))[0])
assert sha256(core) == MD['groimp']['core_sha256'], 'core.jar differs'
shutil.copytree(os.path.dirname(core), f'{SCR}/groimp', dirs_exist_ok=True); core = f'{SCR}/groimp/core.jar'
pyext = f'{SCR}/pyext'; os.makedirs(pyext, exist_ok=True)
for whl in sorted(set(glob.glob('/kaggle/input/**/triangle-*.whl', recursive=True)))[:1]:
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', pyext, whl], capture_output=True, text=True); say('pip', os.path.basename(whl), r.returncode)
say(subprocess.run(['java', '-version'], capture_output=True, text=True).stderr.strip(), '| cpus', os.cpu_count())
rows = []


def job(i, case):
    slot = i % PARALLEL
    env = dict(os.environ, PYTHONPATH=f'{repo}:{pyext}', JDK_JAVA_OPTIONS='-Djava.awt.headless=true -XX:ActiveProcessorCount=1 -Xmx6g', PYTHONUTF8='1',
               PYTHONUNBUFFERED='1', FSP_BUILD_ROOT='/tmp', NUMBA_NUM_THREADS='1', NUMBA_CACHE_DIR=f'/tmp/nb{slot}')
    t0 = time.time()
    with open(f'{WORK}/log_{case}.txt', 'w', encoding='utf-8') as lf:
        p = subprocess.run([sys.executable, '-X', 'utf8', f'{dvcode}/ev_case.py', f'{SCR}/prepared/{case}', core, OUT, dvcode], cwd=repo, env=env,
                           stdout=lf, stderr=subprocess.STDOUT, text=True)
    stp = f'{OUT}/{case}/status.json'
    st = json.load(open(stp)) if os.path.exists(stp) else dict(status='no_status')
    row = dict(case=case, status=st.get('status'), attempts=len(st.get('attempts', [])), summary=st.get('summary'), wall_s=round(time.time() - t0))
    say('JOB', json.dumps(row)); return row


with ThreadPoolExecutor(PARALLEL) as ex:
    for row in ex.map(lambda a: job(*a), list(enumerate(JOBS))):
        rows.append(row); json.dump(rows, open(f'{WORK}/kernel_jobs.json', 'w'), indent=1)
for f in glob.glob(f'{WORK}/log_*.txt'):
    if os.path.getsize(f) > 2_000_000:
        with open(f, 'rb') as fh: data = fh.read()
        with open(f, 'wb') as fh: fh.write(data[:500_000] + b'\n...[truncated]...\n' + data[-500_000:])
say('=== done', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), sum(r['status'] == 'verified' for r in rows), '/', len(rows), 'verified')
