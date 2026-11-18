"""PC side of the v0.2 round trip, last step: axis classification (frozen Q1, skeleton_r8pc.step_axis) on the groups of one one-zip run,
then the third dataset for stage 7. tomato3d/ is imported, never written; its output folders are pointed at this build folder.

usage: python make_tprox_inputs.py <groups_out (kernel output)> <recon_out (the one-zip run)> <dest> [--iid ozp] [--skeleton skeleton_v1.json]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("groups_out", type=Path); ap.add_argument("recon_out", type=Path); ap.add_argument("dest", type=Path)
    ap.add_argument("--iid", default="ozp"); ap.add_argument("--skeleton", type=Path, default=None); a = ap.parse_args()
    from tomato3d.organs import skeleton_propagate as SP
    from tomato3d.organs import skeleton_r8pc as R8
    all3 = HERE / "all3"; cell = all3 / a.iid; (cell / "merge").mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.groups_out / "merge" / "groups.npz", cell / "merge" / "groups.npz")
    SP.ALL3 = all3; R8.RUNS = HERE / "axis"                                               # redirect: nothing is written into the other build folders
    axis = R8.step_axis(a.iid)
    full = a.recon_out / "surfels_full.ply"; g = np.load(cell / "merge" / "groups.npz"); ax = np.load(R8.RUNS / a.iid / "axis_labels.npz")
    from tomato3d.fuse3d import surfels_io as sio
    sf = sio.read_surfels_full(full); rows = np.nonzero(sf["opacity"] >= 0.5)[0]
    assert np.array_equal(g["rows"], rows) and np.array_equal(ax["rows"], rows) and np.allclose(g["xyz"], sf["xyz"][rows], atol=1e-6), "groups do not index this run's opaque surfels"
    if a.dest.exists():
        shutil.rmtree(a.dest)
    a.dest.mkdir(parents=True)
    shutil.copy2(cell / "merge" / "groups.npz", a.dest / "groups.npz"); shutil.copy2(a.groups_out / "work" / "geometry.npz", a.dest / "geometry.npz")
    shutil.copy2(a.groups_out / "geometry.json", a.dest / "geometry.json"); shutil.copy2(R8.RUNS / a.iid / "axis_labels.npz", a.dest / "axis_labels.npz"); shutil.copy2(full, a.dest / "surfels_full.ply")
    if a.skeleton:
        shutil.copy2(a.skeleton, a.dest / "skeleton_v1.json")
    summ = json.load(open(a.groups_out / "merge" / "summary.json", encoding="utf-8")) if (a.groups_out / "merge" / "summary.json").exists() else {}
    layers = summ.get("layers_used"); gj = json.load(open(a.dest / "geometry.json", encoding="utf-8"))
    meta = dict(made_for="microtom_recon_onezip v0.2 stage 7 (--tprox-inputs)", surfels_full_sha256=sha(full), surfels_opaque=int(len(rows)),
                labels=["groups_sam_candidates_only", "R3-1b-style_unaccepted", "gate_failed_viewing_only", "conditional_on_unvalidated_scale"],
                groups=dict(n_groups=int(len(np.unique(g["group"][g["group"] >= 0]))), grouped_share=round(float((g["group"] >= 0).mean()), 4), candidate_layers=layers,
                            merge_summary={k: summ.get(k) for k in ("voting_views", "merges", "groups", "unassigned_share")}, t_sheet_mm=gj.get("t_sheet_mm")),
                axis=dict(source=axis["axis_source"], axis_share=axis["axis_share"], axis_share_unusual=axis["axis_share_unusual"], stem_like=axis["stem_like"], rachis_like=axis["rachis_like"], q1=axis["q1"]),
                adapter="pilot photos 6048 -> 3024 px (area average), K scaled, pose rms halved; plant id registered at run time; no Roboflow layer",
                sha256={f: sha(a.dest / f) for f in ("groups.npz", "geometry.npz", "geometry.json", "axis_labels.npz")},
                skeleton_sha256=(sha(a.dest / "skeleton_v1.json") if a.skeleton else None))
    json.dump(meta, open(a.dest / "TPROX_INPUTS_META.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: meta[k] for k in ("surfels_opaque", "groups", "axis")}, indent=1, default=float))


if __name__ == "__main__":
    main()
