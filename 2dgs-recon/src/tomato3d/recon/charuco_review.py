"""Steps 4-5 page: 3D leaflet proposals (surfels coloured per leaflet), click -> its H2 source masks in each view, and a
grade form whose answers go to the artifact's db collection `grades` (empty at publish; proposals stay in the page).

usage: python -m tomato3d.recon.charuco_review --poses <charuco dir> --recon <recon dir> --k <kernel out dir> --out <html>
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.fuse3d import surfels_io as sio

TEMPLATE = Path(__file__).with_name("charuco_review_template.html")
MAX_UNASSIGNED, MAX_ASSIGNED, CROP_PX, MAX_CROPS = 40000, 140000, 240, 8


def palette(n: int) -> list[str]:
    out = []
    for i in range(n):
        h = (i * 0.618034) % 1.0; l = 0.50 if i % 2 else 0.42; s = 0.62 if i % 3 else 0.75
        r, g, b = colorsys.hls_to_rgb(h, l, s); out.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return out


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def crop(rgb: np.ndarray, lab: np.ndarray, label: int, colour: str):
    s = rgb.shape[0] / lab.shape[0]; m = lab == label
    if not m.any():
        return None, 0
    ys, xs = np.nonzero(m); pad = max(12, int(0.25 * max(np.ptp(xs), np.ptp(ys))))
    y0, y1 = max(0, ys.min() - pad), min(lab.shape[0], ys.max() + pad + 1); x0, x1 = max(0, xs.min() - pad), min(lab.shape[1], xs.max() + pad + 1)
    im = rgb[int(y0 * s):int(y1 * s), int(x0 * s):int(x1 * s)].copy()
    mm = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = tuple(int(colour[k:k + 2], 16) for k in (1, 3, 5)); im = np.ascontiguousarray(im)
    cv2.drawContours(im, cs, -1, (255, 255, 255), max(3, im.shape[0] // 90)); cv2.drawContours(im, cs, -1, c, max(2, im.shape[0] // 140))
    pi = Image.fromarray(im); pi.thumbnail((CROP_PX, CROP_PX)); buf = io.BytesIO(); pi.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode(), int(m.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", required=True); ap.add_argument("--recon", required=True); ap.add_argument("--k", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args(argv)
    src, rc, kd = Path(a.poses), Path(a.recon), Path(a.k); po = json.load(open(src / "poses.json")); R0 = po["undistorted_res"]
    doc = json.load(open(rc / "leaflets_3d.json")); L = doc["leaflets"]; asg = np.load(rc / "surfel_assignment.npy")
    sf = sio.read_surfels_full(kd / "surfels_full.ply"); xyz = sf["xyz"]
    rng = np.random.default_rng(0); ia = np.nonzero(asg >= 0)[0]; iu = np.nonzero(asg < 0)[0]
    ia = ia[rng.choice(len(ia), min(MAX_ASSIGNED, len(ia)), replace=False)] if len(ia) else ia
    iu = iu[rng.choice(len(iu), min(MAX_UNASSIGNED, len(iu)), replace=False)] if len(iu) else iu
    idx = np.r_[ia, iu]; P = xyz[idx]; lo = P.min(0); span = np.maximum(P.max(0) - lo, 1e-9)
    cols = palette(len(L)); leaflets = []
    for l, c in zip(L, cols):
        leaflets.append(dict(id=l["id"], color=c, flags=l["flags"], supporting_views=l["supporting_views"], n_surfels=l["n_surfels"], surface=l["surface"],
                             normal_spread_deg=l["normal_spread_deg"], sources=l["sources"]))
    crops, cache = {}, {}
    for l, c in zip(L, cols):
        rows = []
        for s_ in l["sources"]:
            v = s_["view"]
            if v not in cache:
                cache.clear()
                cache[v] = (np.asarray(Image.open(src / f"undist_{R0}" / v).convert("RGB")),
                            np.asarray(Image.open(rc / "leaflets" / "labels" / (Path(v).stem + ".png"))).astype(np.int32))
            jpg, area = crop(*cache[v], s_["h2_label"], c)
            if jpg:
                rows.append(dict(view=v, h2_id=s_["h2_id"], jpg=jpg, area=area))
        rows.sort(key=lambda r: -r["area"]); crops[l["id"]] = [dict(view=r["view"], h2_id=r["h2_id"], jpg=r["jpg"]) for r in rows[:MAX_CROPS]]
    ctr = ((xyz[asg >= 0].min(0) + xyz[asg >= 0].max(0)) / 2) if (asg >= 0).any() else P.mean(0)
    ext = float(np.linalg.norm(np.percentile(xyz[asg >= 0] if (asg >= 0).any() else P, 99, 0) - np.percentile(xyz[asg >= 0] if (asg >= 0).any() else P, 1, 0)))
    data = dict(ctr=ctr.tolist(), rad=1.6 * ext, pt_size=max(0.0008, ext / 250), n_views=len(json.load(open(kd / "done.json"))["views"]), counts=doc["counts"],
                leaflets=leaflets, crops=crops,
                pts=dict(lo=lo.tolist(), rng=span.tolist(), pos=b64(np.round((P - lo) / span * 65535).astype(np.uint16)),
                         lid=b64(asg[idx].astype(np.int16))))
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(html, encoding="utf-8")
    print(a.out, Path(a.out).stat().st_size, "bytes;", len(L), "leaflets")


if __name__ == "__main__":
    main()
