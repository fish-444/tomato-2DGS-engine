"""Build GUI-canopy check cases + Kaggle dataset + kernels (CONTRACT.md §4).
python make_gui_tests.py <batch> <n_kernels> <test> [<test> ...]
test = <name>:<base>:<KEY=VAL,...>   base = G1 (1 plant, point lamp, v2.2A) | T8 (8 plants, v2.2A); keys = settings without SET_, plus RAYS=<n> (test only:
LightModel ray count, e.g. for the lamp-field check). Cases: C:/fsp_build/engine_gui_canopy_20261106_01/prepared/<name>."""
import hashlib, json, shutil, subprocess, sys, tarfile
from pathlib import Path

H = Path(__file__).resolve().parent; REPO = H.parents[1]; IDEO = REPO / 'build' / 'engine_ideotype_20261105_01'; V2 = REPO / 'build' / 'engine_v2_calib_20261104_01'
BASES = dict(G1=Path('C:/fsp_build/engine_ideotype_20261105_01/prepared/BASE_G0_r1_v22A-4p6468'),
             T8=Path('C:/fsp_build/engine_ideotype_20261105_01/prepared/BASE_s10h40_r1_v22A-4p6468'),
             T32=Path('C:/fsp_build/engine_ideotype_20261105_01/prepared/BASE_T32s10h40_r1_v22A-4p6468'),
             T8R=Path('C:/fsp_build/engine_ideotype_20261105_01/prepared/BASE_s10h40_r1_v22R-5p0843'))
PREP = Path('C:/fsp_build/engine_gui_canopy_20261106_01/prepared')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(test):
    nm, base, kvs = test.split(':', 2); kv = dict(x.split('=', 1) for x in kvs.split(',') if x)
    rays = kv.pop('RAYS', None); kv_probe = kv.pop('PROBE', None); exp_ = kv.pop('EXP', None); panel = kv.pop('PANEL', None); meshes_ = kv.pop('MESHES', None); vphi = kv.pop('VPHI', None); B = BASES[base]; D = PREP / nm
    if D.exists(): shutil.rmtree(D)
    shutil.copytree(B, D)
    tmp = PREP / f'_gui_{nm}'
    if tmp.exists(): shutil.rmtree(tmp)
    r = subprocess.run([sys.executable, '-X', 'utf8', str(H / 'make_model_gui_canopy.py'), str(B), str(tmp)] + [f'{k}={v}' for k, v in kv.items()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    M = D / 'native' / 'working' / 'Model.rgg'; t = (tmp / 'Model.rgg').read_bytes()
    if rays:
        old = b'LightModel lm=new LightModel(300000,1);'; assert t.count(old) == 1; t = t.replace(old, f'LightModel lm=new LightModel({int(rays)},1);'.encode())
    if panel:   # test only: size of the flat1 panel, PANEL=<length_m>x<width_m> (angular-law probe with a near-point emitter)
        Lp, Wp = (float(v) for v in panel.split('x'))
        for a, b in ((b'Parallelogram(0.6f,0.1f)', f'Parallelogram({Lp}f,{Wp}f)'.encode()), (b'double L=0.60,w=0.05', f'double L={Lp},w={Wp / 2}'.encode())):
            assert t.count(a) == 1, (a, t.count(a)); t = t.replace(a, b)
    if meshes_ == 'ends':   # scenario runs (build/engine_tray32_dyn_20261107_01): mesh files only at state 0 and the last state (32-plant runs hung at the end)
        s_ = t.decode('utf-8')
        for a, b in (('    exportMeshes(meshes,day);', '    if(day==0 || day>=SET_DAYS+11)exportMeshes(meshes,day);'),
                     ('        exportMeshes(consumed,step);', '        if(step==0 || step>=SET_DAYS+10)exportMeshes(consumed,step);')):
            assert s_.count(a) == 1, (a, s_.count(a)); s_ = s_.replace(a, b)
        t = s_.encode('utf-8')
    if vphi:   # scenario only (tray32 dyn Amendment 4): wing rotation phi of the V120 build replaced (V150 = phi 15 deg; opening = 180 - 2 phi); the settings line still says V120
        old = b'SET_LAMP_SHAPE.equals("V120")?Math.toRadians(30)'; assert t.count(old) == 1, t.count(old)
        t = t.replace(old, f'SET_LAMP_SHAPE.equals("V120")?Math.toRadians({float(vphi)})'.encode())
    if exp_:   # test only: AreaLight exponent of the GUI lamp nodes (probe of GroIMP's exponent-0 behaviour)
        n0 = t.count(b'setExponent(0.0f)'); assert n0 >= 3, n0
        t = t.replace(b'setExponent(0.0f)', f'setExponent({float(exp_)}f)'.encode())
    if kv_probe:   # test only: lamp probe sensor lines (25 along x at y 0, x -0.30..0.30 step 0.025; 17 along y at x 0, y -0.40..0.40 step 0.05)
        s = t.decode('utf-8'); NLc = '\r\n'
        reps = [('    [ w:World ==> w for(int j=0;j<16;j++)' + NLc + '        ([Translate((float)(GRID_X+(j%4)*GRID_SPACING),(float)(GRID_Y+(j/4)*GRID_SPACING),(float)GRID_Z) RL(-90)',
                 '    [ w:World ==> w for(int j=0;j<42;j++)' + NLc + '        ([Translate((float)gpX(j),(float)gpY(j),(float)GRID_Z) RL(-90)'),
                ('sensors.println("grid_row,grid_col,absorbed_umol_s,area_m2");' + NLc + '    for(Sensor s:(* Sensor *))sensors.println((s.sid/4+1)+","+(s.sid%4+1)+","',
                 'sensors.println("sid,x_m,y_m,absorbed_umol_s,area_m2");' + NLc + '    for(Sensor s:(* Sensor *))sensors.println(s.sid+","+gpX(s.sid)+","+gpY(s.sid)+","')]
        for a, b in reps:
            assert s.count(a) == 1, ('probe point', s.count(a), a[:60]); s = s.replace(a, b)
        s = s.rstrip('\r\n') + NLc + 'double gpX(int j) { return j<25?-0.30+0.025*j:0.0; }' + NLc + 'double gpY(int j) { return j<25?0.0:-0.40+0.05*(j-25); }' + NLc
        t = s.encode('utf-8')
    pj = D / 'native' / 'prepared.json'; prep = json.loads(pj.read_text(encoding='utf-8')); assert prep['model_hash'] == sha(M)
    M.write_bytes(t); prep['model_hash'] = sha(M); pj.write_text(json.dumps(prep, indent=2), encoding='utf-8')
    man = json.load(open(tmp / 'gui_canopy_manifest.json')); shutil.rmtree(tmp)
    json.dump(dict(test=nm, base=base, base_dir=str(B), settings=man['settings'], rays=int(rays) if rays else 300000, vphi_deg=float(vphi) if vphi else None, model_sha256=sha(M)), open(D / 'gui_test.json', 'w'), indent=1)
    return nm


def main():
    batch, nk, tests = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    names = [build(t) for t in tests]
    DS = H / 'dataset' / batch; DS.mkdir(parents=True, exist_ok=True)

    def tar(nm, mem):
        p = DS / f'{nm}.tar.gz'
        with tarfile.open(p, 'w:gz') as tf:
            for src, arc in mem: tf.add(src, arcname=arc, filter=lambda ti: None if '__pycache__' in ti.name else ti)
        return p
    runs = tar('runs', [(PREP / n, f'prepared/{n}') for n in names])
    code = tar('code', [(H / 'kaggle_kernel' / 'gui_case.py', 'evcode/gui_case.py'), (IDEO / 'kaggle_kernel' / 'ideo_validate.py', 'evcode/ideo_validate.py'),
                        (V2 / 'kaggle_kernel' / 'v2_validate.py', 'evcode/v2_validate.py')])
    core = Path('C:/Program Files/GroIMP 2.2.1/core.jar'); DEV = Path('C:/fsp_build/engine_dev_20261020')
    man = dict(schema='ideo-kaggle-input-1', batch=batch, cases=names, tarballs={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in (runs, code)},
               groimp=dict(core_sha256=sha(core)), engine_code={p.name: sha(p) for p in sorted((DEV / 'tools').glob('*')) if p.is_file() and p.suffix in ('.py', '.rgg')})
    (DS / 'ENGV2_MANIFEST.json').write_text(json.dumps(man, indent=1)); slug = f'guic-{batch.lower()}-20261106'
    (DS / 'dataset-metadata.json').write_text(json.dumps(dict(title=slug, id=f'dolgfish/{slug}', licenses=[dict(name='CC0-1.0')])))
    tpl = (IDEO / 'kaggle_kernel' / 'kernel_template.py').read_text(encoding='utf-8')
    assert tpl.count('ideo_case.py') == 1; tpl = tpl.replace('ideo_case.py', 'gui_case.py')
    meta = {}
    for k in range(nk):
        kn = f'gc-{batch.lower()}-k{k}-20261106'; js = names[k::nk]; d = H / 'kernels' / kn; d.mkdir(parents=True, exist_ok=True)
        (d / 'kernel.py').write_text(tpl.replace('@JOBS@', json.dumps(js)), encoding='utf-8')
        (d / 'kernel-metadata.json').write_text(json.dumps(dict(id=f'dolgfish/{kn}', title=kn, code_file='kernel.py', language='python', kernel_type='script', is_private=True,
            enable_gpu=False, enable_tpu=False, enable_internet=False, dataset_sources=[f'dolgfish/{slug}', 'dolgfish/tomato-dreftray-input-20261031', 'dolgfish/tomato-h-input'],
            competition_sources=[], kernel_sources=[]), indent=1))
        meta[kn] = js
    (H / f'KERNELS_{batch}.json').write_text(json.dumps(meta, indent=1)); print(json.dumps({k: v for k, v in meta.items()}))


if __name__ == '__main__':
    main()
