"""python make_kernels.py <batch> <n_kernels>: kernel folders for the jobs in PREPARE_<batch>.json, split round-robin."""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent; batch, nk = sys.argv[1], int(sys.argv[2])
jobs = json.load(open(H / f'PREPARE_{batch}.json'))['jobs']; slug = f'ideo-{batch.lower()}-20261105'
tpl = (H / 'kaggle_kernel' / 'kernel_template.py').read_text(encoding='utf-8'); meta = {}
for k in range(nk):
    name = f'id-{batch.lower()}-k{k}-20261105'; js = jobs[k::nk]; d = H / 'kernels' / name; d.mkdir(parents=True, exist_ok=True)
    (d / 'kernel.py').write_text(tpl.replace('@JOBS@', json.dumps(js)), encoding='utf-8')
    (d / 'kernel-metadata.json').write_text(json.dumps(dict(id=f'dolgfish/{name}', title=name, code_file='kernel.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=False, enable_tpu=False, enable_internet=False, dataset_sources=[f'dolgfish/{slug}', 'dolgfish/tomato-dreftray-input-20261031', 'dolgfish/tomato-h-input'],
        competition_sources=[], kernel_sources=[]), indent=1))
    meta[name] = js
(H / f'KERNELS_{batch}.json').write_text(json.dumps(meta, indent=1)); print({k: len(v) for k, v in meta.items()})
