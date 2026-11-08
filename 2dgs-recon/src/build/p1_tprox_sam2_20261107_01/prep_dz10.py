"""Amendment A2 (local, light: file reading only): per plant pack the frozen T-prox bundle + stored labels, opaque surfels (xyz, scale) for occlusion, poses, vote views and the H2 layer of the 3D-first candidates.
usage (project root): python -X utf8 build/p1_tprox_sam2_20261107_01/prep_others.py"""
from pathlib import Path
import hashlib, json, shutil, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; D = Path(__file__).resolve().parent; OUT = D / 'kaggle_dz10'; sys.path.insert(0, str(ROOT))
from tomato3d.organs import all_compound as AC
DZ = ROOT / 'build/p1_dz10_labels_20261101_01'; ALL3 = ROOT / 'build/p1_all_3dfirst_20261027_01'
PIDS = ('e_01', 'e_04', 'e_05', 'e_07', 'n_01', 'n_02', 'n_03', 'n_04', 'n_05', 't2_02', 't2_04', 't2_05')


def main():
    OUT.mkdir(exist_ok=True); meta = {}
    for pid in PIDS:
        G = json.load(open(sorted(DZ.glob(f'grade_{pid}_*.json'))[0], encoding='utf-8')); st = [d for d in sorted((DZ / f'{pid}_{G["skelver"]}' / 'runs' / pid).iterdir()) if d.is_dir() and d.name[:8].isdigit()]; assert len(st) == 1, (pid, st); run = st[0]; o = OUT / pid; o.mkdir(exist_ok=True)
        shutil.copy2(run / 'kaggle_data/r8pc_bundle.npz', o / 'r8pc_bundle.npz'); shutil.copy2(run / 'R8_pc/labels.npz', o / 't0_r8pc.npz'); shutil.copy2(run / 'R8_pc_rachis/labels_rachis.npz', o / 't0_rachis.npz')
        pl = AC.load_individual(pid); Z = np.load(o / 'r8pc_bundle.npz'); pos = {int(r): i for i, r in enumerate(pl['rows'])}; brow = np.array([pos.get(int(r), -1) for r in Z['rows']]); assert (brow >= 0).all(), pid
        np.savez_compressed(o / 'surfels.npz', xyz=pl['xyz'], sc=pl['sc'], bundle_index=brow)
        man = json.load(open(ALL3 / pid / 'inputs.json', encoding='utf-8')); held = man.get('label_heldout_excluded_from_votes'); views = [v for v in man['views'] if v != held and v in pl['po']['views']]
        h2 = {}
        for v in views:
            f = ALL3 / pid / 'cands' / f'{Path(v).stem}_cands.json'
            if f.exists(): h2[v] = [c['rle'] for c in json.load(open(f, encoding='utf-8'))['candidates'] if c.get('prop') == 'P-H2']
        json.dump(dict(po=pl['po'], views=views, held_excluded=held, h2=h2), open(o / 'inputs.json', 'w'))
        bm = json.load(open(run / 'bundle_meta.json', encoding='utf-8'))
        meta[pid] = dict(run=run.name, views=len(views), photos_with_h2=len(h2), h2_masks=sum(len(x) for x in h2.values()), leaf_map=bm['leaf_map'], bundle_sha16=hashlib.sha256((o / 'r8pc_bundle.npz').read_bytes()).hexdigest()[:16])
        print(pid, {k: meta[pid][k] for k in ('run', 'views', 'photos_with_h2', 'h2_masks')})
    shutil.copy2(D / 'kaggle_data/pc_absorb.py', OUT / 'pc_absorb.py'); shutil.copy2(D / 'kaggle_data/pc_rachis.py', OUT / 'pc_rachis.py')
    (D / 'prep_dz10_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    (OUT / 'dataset-metadata.json').write_text(json.dumps({'title': 'p1 tprox sam2 dz10 20261108', 'id': 'dolgfish/p1-tprox-sam2-dz10-20261108', 'licenses': [{'name': 'CC0-1.0'}]}), encoding='utf-8')


if __name__ == '__main__':
    main()
