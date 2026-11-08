"""Step 4: surfel -> leaflet by multi-view voting, cross-view linking by voting overlap, per-leaflet geometry records.

1. Every surfel centre is projected into every TRAINING view; it is visible there when the rendered alpha >= 0.5 and its
   depth is within VIS_REL of the rendered median depth (the row-31 C2 visibility test). The H2 label there is read.
2. (view, leaflet id) pairs are linked when the IoU of their surfel sets, over surfels visible in both views, is
   >= LINK_IOU (no A3 v2 association exists for this capture). Connected components = 3D leaflet candidates.
3. A surfel takes the component voted by most of its views, with >= MIN_AGREE views agreeing; otherwise `unassigned`.
4. Per leaflet: plane fit (PCA); quadric in plane coordinates when plane RMS / extent > QUAD_REL; mean normal and angular
   spread (surfel normals flipped to the plane normal's side, plane normal oriented towards the mean camera); outline =
   concave hull of the surfels projected on the fitted plane; supporting views; flags few_views (< 3 views),
   occluded_part (H2 `occluded` in any supporting view), merged_suspect (two ids of one view in the component).
Geometry records for review and the engine's observation route - not trait claims.

usage: python -m tomato3d.recon.charuco_assign --recon <recon dir> --k <kernel out dir>
Writes <recon>/leaflets_3d.json, surfel_assignment.npy (int32 leaflet index per surfel, -1 unassigned).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.fuse3d import surfels_io as sio

VIS_REL, ALPHA_MIN = 0.02, 0.5
LINK_IOU, MIN_PAIR = 0.3, 20          # IoU of surfel sets; at least MIN_PAIR surfels visible in both views
MIN_AGREE = 2
MIN_SURFELS = 30
QUAD_REL = 0.03


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def visibility_labels(xyz, cams, views, npz, labels, res):
    """lab[v][i] = H2 leaflet id (0 = none) of surfel i in view v when visible there, else -1."""
    out = {}
    for v in views:
        c = cams[v]; K = np.array(c["K"]); R = np.array(c["R_cw"]); t = np.array(c["t_cw"])
        xc = xyz @ R.T + t; z = xc[:, 2]; u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res)
        d = npz[f"d_{v}"]; a = npz[f"a_{v}"].astype(np.float32); lab = np.full(len(xyz), -1, np.int32)
        uu, ww = u[ok], w[ok]; vis = (a[ww, uu] >= ALPHA_MIN) & (np.abs(z[ok] - d[ww, uu]) <= VIS_REL * np.maximum(d[ww, uu], 1e-9))
        L = labels[v]; s = L.shape[0] / res
        idx = np.nonzero(ok)[0][vis]; lab[idx] = L[np.minimum((w[idx] * s).astype(int), L.shape[0] - 1), np.minimum((u[idx] * s).astype(int), L.shape[1] - 1)]
        out[v] = lab
    return out


def link(vl: dict, views):
    dsu = DSU(); sets = {}
    for v in views:
        for i in np.unique(vl[v][vl[v] > 0]):
            sets[(v, int(i))] = vl[v] == i; dsu.find((v, int(i)))
    pairs = []
    for a_i, va in enumerate(views):
        for vb in views[a_i + 1:]:
            both = (vl[va] >= 0) & (vl[vb] >= 0)
            if both.sum() < MIN_PAIR:
                continue
            la, lb = vl[va][both], vl[vb][both]; m = (la > 0) & (lb > 0)
            for (ia, ib), n in Counter(zip(la[m].tolist(), lb[m].tolist())).items():
                if n < MIN_PAIR:
                    continue
                na, nb = int((la == ia).sum()), int((lb == ib).sum()); iou = n / (na + nb - n)
                if iou >= LINK_IOU:
                    dsu.union((va, ia), (vb, ib)); pairs.append(((va, ia), (vb, ib), round(iou, 3)))
    comps = defaultdict(list)
    for k in sets:
        comps[dsu.find(k)].append(k)
    return list(comps.values()), pairs


def concave_outline(P2):
    from shapely import concave_hull
    from shapely.geometry import MultiPoint
    g = concave_hull(MultiPoint(P2), ratio=0.3)
    if g.geom_type != "Polygon":
        g = MultiPoint(P2).convex_hull
    return np.asarray(g.exterior.coords) if g.geom_type == "Polygon" else np.zeros((0, 2))


def fit_surface(P, N, cam_dir):
    c = P.mean(0); U, S, Vt = np.linalg.svd(P - c, full_matrices=False); n = Vt[2]
    if n @ cam_dir < 0:
        n = -n
    e1, e2 = Vt[0], np.cross(n, Vt[0]); X = np.c_[(P - c) @ e1, (P - c) @ e2]; h = (P - c) @ n
    extent = float(np.ptp(X[:, 0]) + 1e-12); rms_plane = float(np.sqrt(np.mean(h ** 2)))
    rec = dict(centroid=c.tolist(), normal=n.tolist(), axes=[e1.tolist(), e2.tolist()], extent_m=[float(np.ptp(X[:, 0])), float(np.ptp(X[:, 1]))],
               plane_rms_m=rms_plane, plane_rms_rel=rms_plane / extent, surface="plane")
    if rms_plane / extent > QUAD_REL and len(P) >= 12:
        A = np.c_[X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2, X[:, 0], X[:, 1], np.ones(len(X))]
        coef, *_ = np.linalg.lstsq(A, h, rcond=None); rq = float(np.sqrt(np.mean((A @ coef - h) ** 2)))
        rec.update(surface="quadric", quadric=coef.tolist(), quadric_rms_m=rq, quadric_rms_rel=rq / extent)
    Nf = np.where((N @ n)[:, None] < 0, -N, N); mn = Nf.mean(0); mn = mn / np.linalg.norm(mn)
    rec.update(mean_normal=mn.tolist(), normal_spread_deg=float(np.degrees(np.sqrt(np.mean(np.arccos(np.clip(Nf @ mn, -1, 1)) ** 2)))),
               outline_plane=concave_outline(X).round(5).tolist())
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--recon", required=True); ap.add_argument("--k", required=True)
    a = ap.parse_args(argv); rc, kd = Path(a.recon), Path(a.k)
    done = json.load(open(kd / "done.json")); cams = done["cameras"]; npz = np.load(kd / "views.npz")
    train = [v for v, m in done["views"].items() if m["split"] == "train"]
    sf = sio.read_surfels_full(kd / "surfels_full.ply"); xyz, nrm = sf["xyz"].astype(np.float64), sf["normal"].astype(np.float64)
    res = npz[f"d_{train[0]}"].shape[0]
    labels = {v: np.asarray(Image.open(rc / "leaflets" / "labels" / (Path(v).stem + ".png"))).astype(np.int32) for v in train}
    props = {v: json.load(open(rc / "leaflets" / "proposals" / (Path(v).stem + ".json"))) for v in train}
    vl = visibility_labels(xyz, cams, train, npz, labels, res)
    comps, pairs = link(vl, train)
    comps = [c for c in comps if len({v for v, _ in c}) >= 1]
    cid = {k: j for j, c in enumerate(comps) for k in c}
    votes = np.full((len(xyz), len(train)), -1, np.int32)
    for j, v in enumerate(train):
        m = vl[v] > 0; votes[m, j] = [cid[(v, int(i))] for i in vl[v][m]]
    assign = np.full(len(xyz), -1, np.int32); n_agree = np.zeros(len(xyz), np.int32)
    for i in np.nonzero((votes >= 0).sum(1) >= MIN_AGREE)[0]:
        c, n = Counter(votes[i][votes[i] >= 0].tolist()).most_common(1)[0]
        if n >= MIN_AGREE:
            assign[i] = c; n_agree[i] = n
    cam_c = np.mean([-np.array(cams[v]["R_cw"]).T @ np.array(cams[v]["t_cw"]) for v in train], 0)
    leaflets, remap = [], np.full(len(comps), -1, np.int32)
    for j, comp in enumerate(comps):
        sel = np.nonzero(assign == j)[0]
        if len(sel) < MIN_SURFELS:
            continue
        vs = sorted({v for v, _ in comp}); per_view = Counter(v for v, _ in comp)
        inst_flags = {}
        for v, i in comp:
            ins = props[v]["instances"][i - 1] if i - 1 < len(props[v]["instances"]) else {}
            inst_flags[(v, i)] = bool(ins.get("occluded", False))
        P = xyz[sel]; dirc = cam_c - P.mean(0); dirc /= np.linalg.norm(dirc)
        rec = fit_surface(P, nrm[sel], dirc)
        k = len(leaflets); remap[j] = k
        rec.update(id=f"L{k + 1:03d}", n_surfels=int(len(sel)), supporting_views=len(vs), views=vs,
                   sources=[dict(view=v, h2_label=int(i), h2_id=props[v]["instances"][i - 1]["id"] if i - 1 < len(props[v]["instances"]) else None)
                            for v, i in sorted(comp)],
                   flags=[f for f, c in (("few_views", len(vs) < 3), ("occluded_part", any(inst_flags.values())),
                                         ("merged_suspect", any(n > 1 for n in per_view.values()))) if c])
        leaflets.append(rec)
    final = np.where(assign >= 0, remap[np.maximum(assign, 0)], -1).astype(np.int32)
    np.save(rc / "surfel_assignment.npy", final)
    plant = int(len(xyz)); asg = int((final >= 0).sum())
    out = dict(schema="leaflets3d-1", frame="board metres, z up", note="geometry records for review and the engine observation route; not trait claims",
               params=dict(vis_rel=VIS_REL, alpha_min=ALPHA_MIN, link_iou=LINK_IOU, min_pair=MIN_PAIR, min_agree=MIN_AGREE, min_surfels=MIN_SURFELS, quad_rel=QUAD_REL),
               counts=dict(surfels=plant, assigned=asg, unassigned=plant - asg, components=len(comps), leaflets=len(leaflets), links=len(pairs),
                           flags={f: sum(f in l["flags"] for l in leaflets) for f in ("few_views", "occluded_part", "merged_suspect")},
                           quadric=sum(l["surface"] == "quadric" for l in leaflets)),
               leaflets=leaflets)
    json.dump(out, open(rc / "leaflets_3d.json", "w"), indent=1)
    print(json.dumps(out["counts"]))


if __name__ == "__main__":
    main()
