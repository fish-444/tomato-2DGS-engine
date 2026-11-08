"""R8_pc input bundle (user prompt 2026-10-27 "stage-1 separation, then point-cloud-only absorption"): every array the point-cloud absorption needs, as one npz.
No photo, SAM or V-P map is written into the bundle; the stage-1 labels are the R7 chain's step 1 (diffusion) output, as the user fixed.

usage: python -m tomato3d.organs.pc_absorb_prep
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import chain_r6 as C6
from tomato3d.organs import light_organs as LO
from tomato3d.organs import local_diffusion as LD

OUT = LO.ROOT / "build/p1_pc_absorb_20261027_01"; R7DIR = LD.OUT / "vp_leaf4"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""): h.update(blk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "kaggle_data").mkdir(exist_ok=True)
    B = LD.load("vp"); pl, b, S = B["pl"], B["b"], B["S"]; n = len(pl["xyz"])
    sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); rows = pl["rows"]
    z = np.load(R7DIR / "labels.npz"); stage1 = np.zeros(n, np.int16); stage1[b] = z["step_1_diffusion"]; r7 = z["label"].astype(np.int16)
    N = pl["nrm"].astype(np.float64).copy(); N[b] = S["normal"]; sidx = np.zeros(n, np.int16); nsh = np.ones(n, np.int16); sidx[b] = S["index"]; nsh[b] = S["n_sheets"]
    tr_pts, tr_leaf, tr_id = [], [], []
    for k, t in enumerate(B["traces"]):
        if t["leaf"] <= 0: continue
        d = LD.CC.densify(t["pts"]); tr_pts.append(d); tr_leaf.append(np.full(len(d), t["leaf"])); tr_id.append(np.full(len(d), k))
    bundle = OUT / "kaggle_data" / "r8pc_bundle.npz"
    np.savez_compressed(bundle, xyz=pl["xyz"].astype(np.float64), normal=N, scale=pl["sc"][:, :2], tu=pl["tu"], tv=pl["tv"], rgb=sf["rgb"][rows].astype(np.float32), opacity=sf["opacity"][rows].astype(np.float32),
                        axis_kind=B["axis"].astype(np.int8), sheet_index=sidx, n_sheets=nsh, stage1=stage1, r7=r7, trace_pts=np.vstack(tr_pts), trace_leaf=np.concatenate(tr_leaf), trace_id=np.concatenate(tr_id),
                        t_sheet_mm=np.float64(LD.T_SHEET_MM), offset_mm=np.float64(LD.D["offset_mm"]), reach_mm=np.float64(C6.U_MM), rows=rows)
    srcs = {"R7 labels (stage 1 = step_1_diffusion)": R7DIR / "labels.npz", "surfels PLY": LO.DEFAULTS["surfels"], "sheets": LO.ROOT / "build/p1_stacked_sheets_retry_20261026_01/sheets/sheets.npz"}
    info = dict(bundle=str(bundle.relative_to(LO.ROOT)), bundle_sha256=sha(bundle), sources={k: dict(path=str(v.relative_to(LO.ROOT)), sha256=sha(v)) for k, v in srcs.items()},
                n=n, blade=int((B["axis"] == 0).sum()), axis1=int((B["axis"] == 1).sum()), axis2=int((B["axis"] == 2).sum()), stage1_unassigned_blade=int(((stage1 == 0) & (B["axis"] == 0)).sum()),
                stage1_labelled=int((stage1 > 0).sum()), traces=len(tr_pts), t_sheet_mm=float(LD.T_SHEET_MM), offset_mm=float(LD.D["offset_mm"]), reach_mm=float(C6.U_MM))
    json.dump(info, open(OUT / "inputs.json", "w", encoding="utf-8"), indent=1); print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
