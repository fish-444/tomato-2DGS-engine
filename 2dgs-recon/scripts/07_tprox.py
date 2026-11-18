#!/usr/bin/env python3
"""Step 7 - T-Prox compound-leaf labels on a frozen bundle: frozen pc_absorb.py (weight set A, geometry only) then the frozen rachis step pc_rachis.py.
Default input is the shipped R3-1b bundle; CPU only, a few minutes, about 2 GB of memory (it ran on a Kaggle CPU kernel in 3 min).
  python scripts/07_tprox.py                                  # run on example/r3-1b/frozen_run
  python scripts/07_tprox.py --rebuild-seeds                  # first rebuild the stage-1 seeds from the skeleton + groups + axis labels and check they equal the bundle's
Output (in --out): R8_pc/labels.npz, R8_pc_rachis/labels_rachis.npz (label per surfel; source; p1, p2), results.json, run_info.json.
Parameters are inside the frozen files and are not options: set A, gamma 1.0, p1 >= 0.6, margin >= 0.25, 5 rounds, edge 3 mm; rachis seed 3 mm, blade seed 8 mm + 5 mm margin."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; REPO = HERE.parent; SRC = REPO / "src"; EX = REPO / "example/r3-1b"
sys.path.insert(0, str(SRC))
SET = "A"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rebuild_seeds(bundle_path, skeleton, groups):
    """The seed rule of the R3-1b branch of skeleton_r8pc.step_bundle (mode prox): lines with the same recorded leaf number form one leaf; stage 1 and the densified traces."""
    import numpy as np
    from tomato3d.organs import compound_cluster as CC, skeleton_io as SIO, skeleton_r8pc as R8
    Z = dict(np.load(bundle_path)); pf = np.load(groups); pos = {r: i for i, r in enumerate(Z["rows"])}
    group = np.full(len(Z["xyz"]), -1); group[[pos[r] for r in pf["rows"]]] = pf["patch_filled"]
    sk = SIO.load(skeleton); back = [t for t in SIO.to_traces(sk) if t["leaf"] > 0 and len(t["pts"]) >= 2]; nums = sorted({t["leaf"] for t in back}); assert nums == list(range(1, len(nums) + 1)), nums
    st, info = R8.seeds(Z["xyz"], Z["axis_kind"], group, [[t["pts"] for t in back if t["leaf"] == L] for L in nums], rachis_any=True)
    pts = np.vstack([CC.densify(np.asarray(t["pts"], float)) for t in back])
    ok = bool(np.array_equal(st.astype(Z["stage1"].dtype), Z["stage1"]) and np.allclose(pts, Z["trace_pts"]))
    print("rebuilt seeds:", info, "| equal to the bundle's stage1 and traces:", ok)
    if not ok:
        raise SystemExit("the rebuilt seeds differ from the shipped bundle")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(EX / "frozen_run/kaggle_data/r8pc_bundle.npz")); ap.add_argument("--out", default=str(REPO / "results/r3-1b_tprox"))
    ap.add_argument("--rebuild-seeds", action="store_true"); ap.add_argument("--skeleton", default=str(EX / "skeleton/skeleton_v1.json")); ap.add_argument("--groups", default=str(EX / "groups/patches_filled.npz"))
    a = ap.parse_args(argv); out = Path(a.out).resolve(); got = out / "kaggle_out"; got.mkdir(parents=True, exist_ok=True); (out / "kaggle_data").mkdir(exist_ok=True)
    shutil.copyfile(a.bundle, out / "kaggle_data/r8pc_bundle.npz")
    if a.rebuild_seeds:
        rebuild_seeds(a.bundle, a.skeleton, a.groups)
    pa = SRC / "tomato3d/organs/pc_absorb.py"; print("pc_absorb.py (frozen) sha256", sha(pa)[:16], flush=True)
    with open(out / "pc_absorb.log", "w", encoding="utf-8") as lf:
        r = subprocess.run([sys.executable, str(pa), "--bundle", str(out / "kaggle_data/r8pc_bundle.npz"), "--out", str(got)], stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"pc_absorb.py exit {r.returncode}; see {out / 'pc_absorb.log'}")
    res = json.load(open(got / "results.json")); chosen = res["choice"]["chosen"]
    f = got / ("R8_pc" if chosen == SET else f"set_{SET}") / "labels.npz"
    import numpy as np
    assert str(np.load(f)["set_name"]) == SET, f
    (out / "R8_pc").mkdir(exist_ok=True); shutil.copyfile(f, out / "R8_pc/labels.npz"); shutil.copyfile(got / "results.json", out / "R8_pc/results.json")
    from tomato3d.organs import pc_rachis as PR
    PR.D0 = out; PR.OUT = out / "R8_pc_rachis"; PR.main()
    json.dump(dict(used_set=SET, in_kernel_choice=chosen, in_kernel_gate=res["choice"]["gate_pass"], pc_absorb_sha256=sha(pa), bundle_sha256=sha(a.bundle)), open(out / "run_info.json", "w"), indent=1)
    print("done:", out / "R8_pc_rachis/labels_rachis.npz")


if __name__ == "__main__":
    main()
