"""CONTRACT.md sections 1-2 (local, light): project the H2 leaflet masks (A) and the V-P compound maps (B) onto the R3-1b blade surfels with the frozen projection + W >= 0.1 visibility.
Writes kaggle_data/: groups_s{40,50,60}.npz (group id per bundle row, -1 none), vp_votes.npz (per bundle row, photos voting for each leaf), the frozen bundle and code copies.
No human label is read here.
usage (project root): python -X utf8 build/p1_tprox_sam2_20261107_01/prep.py"""
from pathlib import Path
import hashlib, json, shutil, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; D = Path(__file__).resolve().parent; OUT = D / 'kaggle_data'; sys.path.insert(0, str(ROOT))
from tomato3d.leafseg import pixels as px
from tomato3d.organs import auto_proposers as AP, seed2d as S2, sheet_retry as SR, trace_reconcile as TR, userlabel as UL
RUN = ROOT / 'build/p1_skeleton_layer_20261101_01/r8pc/R3-1b/20261101T133038'; CODE = ROOT / 'build/p1_auto_compound_v3_20261031_01/kaggle_data'
SHARES = (0.40, 0.50, 0.60); MIN_SURFELS = 5


class UF:
    def __init__(self, n): self.p = list(range(n))
    def f(self, a):
        while self.p[a] != a: self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def main():
    OUT.mkdir(parents=True, exist_ok=True); Z = np.load(RUN / 'kaggle_data/r8pc_bundle.npz'); pl = UL.load_plant(); assert np.array_equal(Z['rows'], pl['rows'])
    n = len(Z['rows']); b = np.nonzero(pl['blade'])[0]; W = np.load(SR.FIRST / 'contributions.npz')['W']; vis_all = W >= SR.P['w_min']; st1 = Z['stage1'].astype(int)
    vc = json.load(open(UL.OUT / 'views_chosen.json', encoding='utf-8')); views = [f.split('_')[1] for f in pl['views']]
    prj = SR.projections(pl, b, views)
    # ---- A: H2 leaflet masks -> member sets (bundle row indices)
    sets = []; per_photo = {}
    for vid in views:
        p = prj[vid]; vis = vis_all[b, p['k']] & p['inside']; j = json.load(open(S2.PROPOSALS / f'{AP.view_file(vid)[:-4]}.json', encoding='utf-8')); k = 0
        for inst in j['instances']:
            m = px.rle_decode(inst['visible_segmentation'])
            if not m.any(): continue
            mem = b[vis & m[p['wi'], p['ui']]]
            if len(mem) >= MIN_SURFELS: sets.append(np.unique(mem)); k += 1
        per_photo[vid] = k
    print('A sets', len(sets))
    inv = {}
    for si, s in enumerate(sets):
        for r in s.tolist(): inv.setdefault(r, []).append(si)
    stats = {}
    for share in SHARES:
        uf = UF(len(sets)); cand = set()
        for lst in inv.values():
            for a in range(len(lst)):
                for c in range(a + 1, len(lst)): cand.add((lst[a], lst[c]))
        for a, c in cand:
            inter = len(np.intersect1d(sets[a], sets[c], assume_unique=True))
            if inter / max(len(sets[a]), len(sets[c])) >= share: uf.u(a, c)                 # A1: each group shares >= share of its own surfels
        gid = -np.ones(n, int); conflict = np.zeros(n, bool)
        for r, lst in inv.items():
            roots = {uf.f(x) for x in lst}
            if len(roots) == 1: gid[r] = roots.pop()
            else: conflict[r] = True
        mixed = 0
        for g in np.unique(gid[gid >= 0]):
            m = gid == g; ls = set(st1[m & (st1 > 0)].tolist())
            if len(ls) > 1: gid[m] = -1; mixed += 1
        ug, gid2 = np.unique(gid, return_inverse=True); gid = np.where(gid >= 0, gid2 - (1 if ug[0] == -1 else 0), -1)
        tag = f's{int(share * 100)}'; np.savez_compressed(OUT / f'groups_{tag}.npz', rows=Z['rows'], gid=gid.astype(np.int32), conflict=conflict)
        stats[tag] = dict(groups=int(gid.max() + 1), grouped_blade=int((gid >= 0).sum()), blade=int(len(b)), coverage_blade=round(float((gid >= 0).sum() / len(b)), 4), group_conflict_surfels=int(conflict.sum()), group_mixed_seeds=mixed,
                          groups_with_seed=int(len(np.unique(gid[(gid >= 0) & (st1 > 0)]))))
        print(tag, stats[tag])
    # ---- B: V-P compound maps -> per-row votes (54 photos; the 4 labelled photos are not V-P maps)
    zp = np.load(TR.VP_DIR / 'propagated_maps.npz'); vp = {k[1:]: zp[k] for k in zp.files}; assert not set(vp) & set(vc['chosen']) and vc['held_out'] not in vp
    pv = SR.projections(pl, b, sorted(vp)); votes = np.zeros((n, 9), np.int16)
    for vid in sorted(vp):
        p = pv[vid]; vis = vis_all[b, p['k']] & p['inside']; lab = vp[vid][p['wi'], p['ui']].astype(int); ok = vis & (lab > 0) & (lab < 9)
        np.add.at(votes, (b[ok], lab[ok]), 1)
    np.savez_compressed(OUT / 'vp_votes.npz', rows=Z['rows'], votes=votes); print('B voted rows', int((votes.sum(1) > 0).sum()))
    shutil.copy2(RUN / 'kaggle_data/r8pc_bundle.npz', OUT / 'r8pc_bundle.npz'); shutil.copy2(RUN / 'kaggle_kernel/pc_absorb.py', OUT / 'pc_absorb.py'); shutil.copy2(CODE / 'pc_rachis.py', OUT / 'pc_rachis.py')
    sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    meta = dict(A_sets=len(sets), A_masks_per_photo=per_photo, A=stats, B_photos=len(vp), B_rows_voted=int((votes.sum(1) > 0).sum()), hashes={f.name: sh(f)[:16] for f in sorted(OUT.glob('*')) if f.is_file()})
    (D / 'prep_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    (OUT / 'dataset-metadata.json').write_text(json.dumps({'title': 'p1 tprox sam2 inputs 20261107', 'id': 'dolgfish/p1-tprox-sam2-inputs-20261107', 'licenses': [{'name': 'CC0-1.0'}]}), encoding='utf-8')


if __name__ == '__main__':
    main()
