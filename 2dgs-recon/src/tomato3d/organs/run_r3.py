"""Step 7: the frozen pipeline once on R3 (input: R3_unaccepted, purpose: dev).

Labels per surfel: multi-view fusion as declared (R3_FUSION) - a surfel is `axis` when >= 40 % of the training views that see
it put it on a stem-added pixel (A2 mask label 2), `blade` when >= 60 % put it on an H2 leaflet pixel, else the labeller
decides (p_axis >= LABEL_THR after smoothing). Visibility = point-splat z-buffer at 504 px (tolerance 1.5 mm) over the 61
training views of the whole-plant run. H2 ids voted over >= 2 training views cross-check the blade clusters (`h2_disagree`).

usage: python -m tomato3d.organs.run_r3 [--plant-prefix P]
Writes <run>/r3/{organ_tree.json, organs.obj, labels.npz, r3_summary.json}.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import contract as C
from tomato3d.organs import features as FE
from tomato3d.organs import labeller as LB
from tomato3d.organs import pipeline as PL
from tomato3d.recon import charuco_a3 as ca

OPACITY_MIN = 0.5
MASK_RES, H2_RES = 1512, 1536


def visibility_and_votes(xyz, po, views, res, tol):
    """Per view: visible surfels (z-buffer) and their A2 mask label / H2 id. Returns n_seen, n_stem, n_h2, h2_votes (point -> {view: id})."""
    R0 = po["undistorted_res"]; K = ca.scaled_K(np.array(po["K_undist"]), R0, res)
    n = len(xyz); n_seen = np.zeros(n, int); n_stem = np.zeros(n, int); n_h2 = np.zeros(n, int); votes = defaultdict(dict)
    for v in views:
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"])
        xc = xyz @ R.T + t; z = xc[:, 2]; u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); idx = np.nonzero(ok)[0]
        zb = np.full((res, res), np.inf); order = idx[np.argsort(-z[idx])]          # far first, near overwrites
        zb[w[order], u[order]] = z[order]
        vis = idx[z[idx] <= zb[w[idx], u[idx]] + tol]
        lab = np.asarray(Image.open(C.R3["masks"] / (Path(v).stem + ".png"))); h2 = np.asarray(Image.open(C.R3["h2_labels"] / (Path(v).stem + ".png"))).astype(int)
        su, sw = (u[vis] * MASK_RES // res), (w[vis] * MASK_RES // res); hu, hw = (u[vis] * H2_RES // res), (w[vis] * H2_RES // res)
        n_seen[vis] += 1; n_stem[vis] += (lab[sw, su] == 2); ids = h2[hw, hu]; n_h2[vis] += ids > 0
        for p, i in zip(vis[ids > 0], ids[ids > 0]):
            votes[int(p)][v] = int(i)
    return n_seen, n_stem, n_h2, votes


def h2_disagree(cl, votes, views):
    """Cluster flag: in >= 2 views the cluster's visible points split over two H2 ids with the minority >= 30 %."""
    out = {}
    for c in range(cl.max() + 1):
        pts = np.nonzero(cl == c)[0]; split = 0
        for v in views:
            ids = [votes[int(p)][v] for p in pts if int(p) in votes and v in votes[int(p)]]
            if len(ids) >= 10:
                cnt = Counter(ids).most_common(2)
                if len(cnt) > 1 and cnt[1][1] / len(ids) >= 0.30:
                    split += 1
        out[c] = split >= 2
    return out


def write_obj(path, xyz, r, prefix=""):
    sk = r["sk"]; pre = f"{prefix}__" if prefix else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# organ split on R3 (input: R3_unaccepted, purpose: dev); board metres, z up; points only, no mesh\n")
        n = 0
        def obj(name, P, poly=None):
            nonlocal n
            f.write(f"o {pre}{name}\n")
            for p in P:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            if poly is not None and len(poly) > 1:
                base = n + len(P)
                for q in poly:
                    f.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
                f.write("l " + " ".join(str(base + k + 1) for k in range(len(poly))) + "\n"); n += len(poly)
            n += len(P)
        main = np.concatenate([r["axis_idx"][sk.members[v]] for v in sk.main_path if len(sk.members[v])]) if sk.main_path else []
        obj("stem", xyz[main], sk.V[sk.main_path])
        for lf in r["leaves"] + [x for sa in r["side_axes"] for x in sa["leaves"]]:
            ax = np.concatenate([r["axis_idx"][sk.members[v]] for v in lf["vertices"] if len(sk.members[v])]) if lf["vertices"] else np.array([], int)
            obj(f"rachis_{lf['id']}", xyz[ax], sk.V[lf["path"]])
        for c, name in r["leaflet_labels"].items():
            obj(name, xyz[r["cl"] == c])
        for sa in r["side_axes"]:
            ax = np.concatenate([r["axis_idx"][sk.members[v]] for v in sa["path"] if len(sk.members[v])])
            obj(f"side_stem_{sa['id']}", xyz[ax], sk.V[sa["path"]])
        for t in r["trusses"]:
            ax = np.concatenate([r["axis_idx"][sk.members[v]] for v in t["vertices"] if len(sk.members[v])])
            obj(f"{t['id']}_peduncle", xyz[ax], sk.V[t["path"]])
        un = [c for c, l in r["leaf_of_cluster"].items() if l == "unattached"]
        if un:
            obj("unattached_blades", xyz[np.isin(r["cl"], un)])


def organ_tree(r, xyz, extra_flags):
    sk = r["sk"]
    def leaf_rec(lf):
        cls = [c for c, l in r["leaf_of_cluster"].items() if l == lf["id"]]
        return dict(id=lf["id"], node=lf["node"], rachis_polyline_m=sk.V[lf["path"]].round(5).tolist(), rachis_point_ids=[int(p) for v in lf["vertices"] for p in r["axis_idx"][sk.members[v]]],
                    attachments_m=sk.V[lf["attachments"]].round(5).tolist() if lf["attachments"] else [], n_attach=lf["n_attach"],
                    azimuth_deg=round(lf["azimuth_deg"], 1), elevation_deg=round(lf["elevation_deg"], 1), curvature_rel=round(lf["curvature_rel"], 4),
                    rachis_len_rel=round(lf["rachis_len_rel"] / max(r["counts"]["main_stem_len_rel"], 1e-9), 4),
                    leaflets=[dict(cluster=int(c), label=r["leaflet_labels"].get(c, f"{lf['id']}__?"), n_points=int((r["cl"] == c).sum()),
                                   point_ids=np.nonzero(r["cl"] == c)[0].tolist(), flags=sorted(set(r["cluster_flags"].get(c, []) + extra_flags.get(c, []))))
                              for c in cls],
                    flags=lf["flags"])
    return dict(schema="organ_tree_dev_1 (own minimal form; plant_tree_v1 was not yet declared in LOG when Step 7 ran)",
                **C.TAG, source="reconstruction", input_status="R3_unaccepted", units="board metres", frame="board frame, z up",
                note="geometry records for review; not measurements, not a validated structure",
                r3_acceptance=C.R3["acceptance"], root_m=list(C.POT_CENTRE_M), counts=r["counts"],
                main_stem=dict(polyline_m=sk.V[sk.main_path].round(5).tolist(), point_ids=[int(p) for v in sk.main_path for p in r["axis_idx"][sk.members[v]]],
                               gap_vertices=[int(v) for v in sk.main_path if sk.gap[v]]),
                nodes=[dict(id=nd["id"], rank=k + 1, s_rel=round(nd["s_mm"] / max(r["counts"]["main_stem_len_rel"] * 1000, 1e-9), 4), pos_m=[round(x, 5) for x in nd["pos"]],
                            leaves=[lf["id"] for lf in r["leaves"] if lf["node"] == nd["id"]], side_axes=[sa["id"] for sa in r["side_axes"] if sa["node"] == nd["id"]])
                       for k, nd in enumerate([n for n in sk.nodes if any(l["node"] == n["id"] for l in r["leaves"]) or any(s["node"] == n["id"] for s in r["side_axes"])])],
                leaves=[leaf_rec(lf) for lf in r["leaves"]],
                side_axes=[dict(id=sa["id"], node=sa["node"], polyline_m=sk.V[sa["path"]].round(5).tolist(), leaves=[leaf_rec(lf) for lf in sa["leaves"]]) for sa in r["side_axes"]],
                trusses=[dict(id=t["id"], node=t["node"], polyline_m=sk.V[t["path"]].round(5).tolist(), flags=t["flags"]) for t in r["trusses"]],
                stubs=[dict(node=s["node"], polyline_m=sk.V[s["path"]].round(5).tolist(), flags=s["flags"]) for s in r["stubs"]],
                unattached_clusters=[dict(cluster=int(c), n_points=int((r["cl"] == c).sum()), point_ids=np.nonzero(r["cl"] == c)[0].tolist()) for c, l in r["leaf_of_cluster"].items() if l == "unattached"],
                unreached_axis_points=[int(r["axis_idx"][i]) for i in sk.unreached])


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--plant-prefix", default=""); a = ap.parse_args(argv)
    out = C.RUN / "r3"; out.mkdir(exist_ok=True)
    sf = sio.read_surfels_full(C.R3["surfels"]); keep = sf["opacity"] >= OPACITY_MIN
    xyz = sf["xyz"][keep].astype(np.float64); nrm = sf["normal"][keep].astype(np.float64)
    po = json.load(open(C.R3["poses"])); views = json.load(open(C.R3["run_dir"] / "contract.json"))["train"]
    F, _ = FE.compute(xyz); p_axis = LB.Labeller.load().p_axis(F)
    n_seen, n_stem, n_h2, votes = visibility_and_votes(xyz, po, views, C.R3_FUSION["vis_res"], C.R3_FUSION["vis_tol_m"])
    seen = np.maximum(n_seen, 1); stem_share = n_stem / seen; h2_share = n_h2 / seen
    lab = np.where(stem_share >= C.R3_FUSION["stem_share_axis"], 1, np.where(h2_share >= C.R3_FUSION["blade_share_blade"], 0, -1))
    src = np.where(lab >= 0, "mask", "labeller"); lab = np.where(lab >= 0, lab, (p_axis >= C.LABEL_THR).astype(int))
    is_axis = FE.smooth(xyz, lab == 1)
    r = PL.run(xyz, nrm, is_axis, C.POT_CENTRE_M)
    dis = h2_disagree(r["cl"], votes, views); extra = {c: ["h2_disagree"] for c, d in dis.items() if d}
    tree = organ_tree(r, xyz, extra); json.dump(tree, open(out / "organ_tree.json", "w"), indent=1)
    write_obj(out / "organs.obj", xyz, r, a.plant_prefix)
    np.savez_compressed(out / "labels.npz", xyz=xyz.astype(np.float32), normal=nrm.astype(np.float32), is_axis=is_axis, p_axis=p_axis.astype(np.float32),
                        label_source=src, n_seen=n_seen, stem_share=stem_share.astype(np.float32), h2_share=h2_share.astype(np.float32), cl=r["cl"],
                        main_path_vertex=np.array(r["sk"].main_path), vert_of_axis=r["sk"].vert_of, axis_idx=r["axis_idx"])
    summ = dict(**C.TAG, surfels_total=int(len(sf["xyz"])), surfels_used=int(keep.sum()), label_source=dict(Counter(src.tolist())),
                axis_share=float(is_axis.mean()), n_seen_median=float(np.median(n_seen)), counts=r["counts"], h2_disagree_clusters=int(sum(dis.values())),
                flags=dict(Counter(f for fs in r["cluster_flags"].values() for f in fs)), leaf_flags=dict(Counter(f for lf in r["leaves"] for f in lf["flags"])))
    json.dump(summ, open(out / "r3_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
