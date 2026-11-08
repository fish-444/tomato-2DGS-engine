"""2DGS surfels of the A3 ChArUco run in the existing disc viewer (tomato3d/fuse3d/disc_viewer_template.html, unchanged).
Photo cameras are offered as view presets; no overlay cloud.

usage: python -m tomato3d.recon.charuco_viewer --k <kernel out dir> --out <html> [--title ...] [--note ...]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from tomato3d.fuse3d import surfels_io as sio

TEMPLATE = Path(__file__).resolve().parents[1] / "fuse3d" / "disc_viewer_template.html"


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--k", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="화분 ChArUco 2DGS 원판"); ap.add_argument("--note", default="")
    a = ap.parse_args(argv); kd = Path(a.k)
    sf = sio.read_surfels_full(kd / "surfels_full.ply"); n = len(sf["xyz"]); done = json.load(open(kd / "done.json"))
    lo = sf["xyz"].min(0); rng = np.maximum(sf["xyz"].max(0) - lo, 1e-9); ctr = (lo + rng / 2).tolist()
    H = float(rng[2])
    cams = {}
    for v, c in done["cameras"].items():
        V = np.eye(4); V[:3, :3] = np.array(c["R_cw"]); V[:3, 3] = np.array(c["t_cw"]); cams[v[9:18]] = dict(K=c["K"], viewmat=V.tolist())
    held = [v for v, m in done["views"].items() if m["split"] == "heldout"]
    data = dict(ctr=ctr, res=1512, note=a.note, cam_view=held[0][9:18] if held else None, overlay=dict(clouds=[], lo=lo.tolist(), rng=rng.tolist()), cams=cams,
                discs=dict(n=n, n_plant=n, H=H, lo=lo.tolist(), rng=rng.tolist(), pos=_b64(np.round((sf["xyz"] - lo) / rng * 65535).astype(np.uint16)),
                           tu=_b64(np.round(np.clip(sf["tu"], -1, 1) * 32767).astype(np.int16)), tv=_b64(np.round(np.clip(sf["tv"], -1, 1) * 32767).astype(np.int16)),
                           s=_b64(sf["scale"].astype(np.float16)), op=_b64(np.round(np.clip(sf["opacity"], 0, 1) * 255).astype(np.uint8)),
                           col=_b64(sf["rgb"].astype(np.uint8)), cls=_b64(np.ones(n, np.uint8))))
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    html = html.replace("dwt_03 2DGS 원판 (재실행 10-18)", a.title).replace("<title>dwt_03 2DGS 원판</title>", f"<title>{a.title}</title>").replace("A2 대화 뷰어와 같은 자료", "이 런에는 없음")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(html, encoding="utf-8")
    print(a.out, Path(a.out).stat().st_size, "bytes,", n, "surfels")


if __name__ == "__main__":
    main()
