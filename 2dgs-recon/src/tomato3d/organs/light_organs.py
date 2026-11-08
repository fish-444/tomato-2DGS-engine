"""Track A: light-relevant organ attributes from a reconstruction - leaflets and compound leaves only.

Contract: build/p1_light_organs_20261024_01/CONTRACT_LIGHT_ORGANS.md. The grouping is the frozen organ-split pipeline
(tomato3d.organs.pipeline.run, contract revisions 4-6) called unchanged; this module only labels the input surfels
(the run_r3 fusion rule with the run's own mask directories), computes the per-leaflet / per-leaf attributes and
writes the outputs. Axis membership, node index / rank / attachment and trusses are written as `unavailable`.

usage: python -m tomato3d.organs.light_organs --surfels <surfels_full.ply> --out <dir>
           [--masks DIR] [--h2 DIR] [--poses poses.json] [--views contract.json] [--labeller labeller.json]
           [--input-status R3-1b_unaccepted] [--pot-top 0.066] [--root x y z] [--repeat]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import contract as C
from tomato3d.organs import features as FE
from tomato3d.organs import labeller as LB
from tomato3d.organs import pipeline as PL
from tomato3d.plant_tree.schema import new_tree, validate
from tomato3d.recon import charuco_a3 as ca

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = dict(surfels=ROOT / "build/p1_recon_r3-1_contam_20261022_01/outputs/R3-1b_surfels_full.ply",
                masks=ROOT / "build/p1_recon_r3-1_20261022_01/masks_b", h2=ROOT / "build/p1_recon_charuco_20261021_01/leaflets/labels",
                poses=ROOT / "build/p1_charuco_a3_20261021_01/poses.json", views=ROOT / "build/p1_recon_r3-1_20261022_01/contract.json",
                labeller=C.RUN / "labeller.json", out=ROOT / "build/p1_light_organs_20261024_01")
OPACITY_MIN = 0.5
POT_TOP_Z = 0.066
MASK_RES, H2_RES = 1512, 1536
SCALE_TAG = "conditional_on_unvalidated_scale"
# declared floors (CONTRACT_LIGHT_ORGANS.md section 3)
FLOORS = dict(min_surfels=80, min_views=3, max_planarity=0.35, ellipse_k=1.5, raster_mm=0.25, min_azimuth_incl_deg=5.0)
# declared review limits (section 7)
REVIEW_LIMITS = dict(leaflet_disagreement_max=0.25, leaf_disagreement_max=0.35)
UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------- helpers
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def f6(x) -> str:
    return "%.6f" % float(x)


def visibility_and_votes(xyz, po, views, mask_dir, h2_dir, res, tol):
    """run_r3's rule with explicit mask / H2 directories. Returns n_seen, n_stem, n_h2, votes (point -> {view: h2 id})."""
    R0 = po["undistorted_res"]; K = ca.scaled_K(np.array(po["K_undist"]), R0, res)
    n = len(xyz); n_seen = np.zeros(n, int); n_stem = np.zeros(n, int); n_h2 = np.zeros(n, int); votes = defaultdict(dict)
    for v in views:
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"])
        xc = xyz @ R.T + t; z = xc[:, 2]; u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); idx = np.nonzero(ok)[0]
        zb = np.full((res, res), np.inf); order = idx[np.argsort(-z[idx], kind="stable")]
        zb[w[order], u[order]] = z[order]
        vis = idx[z[idx] <= zb[w[idx], u[idx]] + tol]
        lab = np.asarray(Image.open(Path(mask_dir) / (Path(v).stem + ".png"))); h2 = np.asarray(Image.open(Path(h2_dir) / (Path(v).stem + ".png"))).astype(int)
        su, sw = (u[vis] * MASK_RES // res), (w[vis] * MASK_RES // res); hu, hw = (u[vis] * H2_RES // res), (w[vis] * H2_RES // res)
        n_seen[vis] += 1; n_stem[vis] += (lab[sw, su] == 2); ids = h2[hw, hu]; n_h2[vis] += ids > 0
        for p, i in zip(vis[ids > 0], ids[ids > 0]):
            votes[int(p)][v] = int(i)
    return n_seen, n_stem, n_h2, votes


def h2_disagree(cl, votes, views):
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


def pca_frame(P):
    """Centre, eigenvalues ascending, axes e1 (major), e2, n (normal oriented n_z >= 0)."""
    c = P.mean(0); Q = P - c; w, v = np.linalg.eigh(Q.T @ Q / max(len(P) - 1, 1))
    n = v[:, 0]; e1 = v[:, 2]; e2 = v[:, 1]
    if n[2] < 0:
        n = -n
    if e1[2] < 0 or (abs(e1[2]) < 1e-12 and e1[0] < 0):
        e1 = -e1
    e2 = np.cross(n, e1)
    return c, w, e1, e2, n


def union_area(P, tu, tv, s, c, e1, e2, k=FLOORS["ellipse_k"], px=FLOORS["raster_mm"] / 1000):
    """Area of the union of the projected surfel ellipses (semi-axes k*s0 along tu, k*s1 along tv) on the (e1, e2) plane,
    rasterised at px; returns area_m2, length_m (extent along e1), width_m (extent along e2), covered pixel count."""
    A = np.c_[(tu @ e1), (tu @ e2)] * (k * s[:, 0])[:, None]; B = np.c_[(tv @ e1), (tv @ e2)] * (k * s[:, 1])[:, None]
    X = np.c_[(P - c) @ e1, (P - c) @ e2]
    r = np.sqrt((A ** 2).sum(1) + (B ** 2).sum(1)); lo = (X - r[:, None]).min(0) - px; hi = (X + r[:, None]).max(0) + px
    W = int(np.ceil((hi[0] - lo[0]) / px)) + 1; Hh = int(np.ceil((hi[1] - lo[1]) / px)) + 1
    cov = np.zeros((Hh, W), bool)
    for i in range(len(P)):
        det = A[i, 0] * B[i, 1] - A[i, 1] * B[i, 0]
        i0 = int((X[i, 0] - r[i] - lo[0]) / px); i1 = int((X[i, 0] + r[i] - lo[0]) / px) + 1
        j0 = int((X[i, 1] - r[i] - lo[1]) / px); j1 = int((X[i, 1] + r[i] - lo[1]) / px) + 1
        if abs(det) < 1e-16:
            cov[min(max(int((X[i, 1] - lo[1]) / px), 0), Hh - 1), min(max(int((X[i, 0] - lo[0]) / px), 0), W - 1)] = True; continue
        gx = lo[0] + (np.arange(i0, i1) + 0.5) * px - X[i, 0]; gy = lo[1] + (np.arange(j0, j1) + 0.5) * px - X[i, 1]
        GX, GY = np.meshgrid(gx, gy)
        uu = (B[i, 1] * GX - B[i, 0] * GY) / det; vv = (-A[i, 1] * GX + A[i, 0] * GY) / det
        cov[j0:j1, i0:i1] |= (uu * uu + vv * vv) <= 1.0
    jj, ii = np.nonzero(cov); n_px = int(cov.sum())
    length = float((ii.max() - ii.min() + 1) * px) if n_px else 0.0; width = float((jj.max() - jj.min() + 1) * px) if n_px else 0.0
    return float(n_px) * px * px, length, width, n_px


def hull_area(X2):
    try:
        return float(ConvexHull(X2).volume)
    except Exception:
        return 0.0


def deg_azimuth(vx, vy):
    return float(np.degrees(np.arctan2(vy, vx)) % 360)


def parse_label(lbl: str):
    """'L01__P2R' -> (type, side, pair)."""
    tail = lbl.rsplit("__", 1)[-1]
    if tail.startswith("P") and len(tail) >= 3:
        return "P", tail[-1], int(tail[1:-1])
    if tail.startswith("I"):
        return "I", "C", 0
    return "T", "C", 0


# ---------------------------------------------------------------- the computation
def compute(a) -> dict:
    sf = sio.read_surfels_full(a.surfels); n_all = len(sf["xyz"])
    opaque = sf["opacity"] >= OPACITY_MIN; above = sf["xyz"][:, 2] > a.pot_top; keep = opaque & above
    rows = np.nonzero(keep)[0]
    xyz = sf["xyz"][keep].astype(np.float64); nrm = sf["normal"][keep].astype(np.float64)
    tu, tv, sc = sf["tu"][keep], sf["tv"][keep], sf["scale"][keep]
    H = float(xyz[:, 2].max() - a.pot_top)
    # Step 2 labels: fusion rule of run_r3 with the declared directories
    po = json.load(open(a.poses)); views = json.load(open(a.views))["train"]
    F, _ = FE.compute(xyz); p_axis = LB.Labeller.load(a.labeller).p_axis(F)
    n_seen, n_stem, n_h2, votes = visibility_and_votes(xyz, po, views, a.masks, a.h2, C.R3_FUSION["vis_res"], C.R3_FUSION["vis_tol_m"])
    seen = np.maximum(n_seen, 1); stem_share = n_stem / seen; h2_share = n_h2 / seen
    lab = np.where(stem_share >= C.R3_FUSION["stem_share_axis"], 1, np.where(h2_share >= C.R3_FUSION["blade_share_blade"], 0, -1))
    src = np.where(lab >= 0, "mask", "labeller"); lab = np.where(lab >= 0, lab, (p_axis >= C.LABEL_THR).astype(int))
    is_axis = FE.smooth(xyz, lab == 1)
    # Steps 3-5: frozen pipeline
    r = PL.run(xyz, nrm, is_axis, np.asarray(a.root, float))
    cl = r["cl"]; ncl = int(cl.max() + 1); dis = h2_disagree(cl, votes, views)
    all_leaves = r["leaves"] + [lf for sa in r["side_axes"] for lf in sa["leaves"]]
    leaf_by_internal = {lf["id"]: lf for lf in all_leaves}
    # leaflet attributes
    lfs = []
    for c in range(ncl):
        m = cl == c; P = xyz[m]; cen, w, e1, e2, nn = pca_frame(P)
        area, length, width, n_px = union_area(P, tu[m], tv[m], sc[m], cen, e1, e2)
        X2 = np.c_[(P - cen) @ e1, (P - cen) @ e2]
        incl = float(np.degrees(np.arccos(np.clip(nn[2], -1, 1))))
        planarity = float(w[0] / max(w[1], 1e-18))
        flags = []
        if m.sum() < FLOORS["min_surfels"]: flags.append("low_surfel_count")
        sup = float(np.median(n_seen[m]))
        if sup < FLOORS["min_views"]: flags.append("few_views")
        if planarity > FLOORS["max_planarity"]: flags.append("non_planar")
        flags += sorted(set(r["cluster_flags"].get(c, [])))
        if dis.get(c): flags.append("h2_disagree")
        owner = r["leaf_of_cluster"].get(c, "unattached")
        lfs.append(dict(cluster=c, rows=rows[m].tolist(), n_surfels=int(m.sum()), centroid=cen, normal=nn, e1=e1, e2=e2, inclination_deg=incl,
                        normal_elevation_deg=90.0 - incl, normal_azimuth_deg=(deg_azimuth(nn[0], nn[1]) if incl >= FLOORS["min_azimuth_incl_deg"] else None),
                        area_m2=area, area_hull_m2=hull_area(X2), area_sum_ellipse_m2=float((np.pi * FLOORS["ellipse_k"] ** 2 * sc[m, 0] * sc[m, 1]).sum()),
                        length_m=length, width_m=width, lw_ratio=(length / width if width > 0 else None), n_px=n_px,
                        support_views=sup, support_min=int(n_seen[m].min()), planarity=planarity, flags=flags,
                        uncertain=any(f in flags for f in ("low_surfel_count", "few_views", "non_planar")),
                        internal_leaf=(None if owner == "unattached" else owner), label=r["leaflet_labels"].get(c)))
    # ids: by centroid z, x, y
    order = sorted(range(len(lfs)), key=lambda i: (round(lfs[i]["centroid"][2], 9), round(lfs[i]["centroid"][0], 9), round(lfs[i]["centroid"][1], 9)))
    for k, i in enumerate(order):
        lfs[i]["leaflet_id"] = f"LF{k + 1:03d}"
    lfs = [lfs[i] for i in order]
    # compound leaves
    leaves = []
    for lf in all_leaves:
        mem = [x for x in lfs if x["internal_leaf"] == lf["id"]]
        if not mem:
            continue
        A = np.array([x["area_m2"] for x in mem]); Cc = np.array([x["centroid"] for x in mem]); wts = A / A.sum() if A.sum() > 0 else np.full(len(A), 1 / len(A))
        cen = (Cc * wts[:, None]).sum(0); incl_w = float((np.array([x["inclination_deg"] for x in mem]) * wts).sum())
        incl_p = None
        if len(mem) >= 3:
            _, w, _, _, nn = pca_frame(Cc)
            if w[1] > 1e-12 and w[0] / w[1] < 0.9:          # not collinear
                incl_p = float(np.degrees(np.arccos(np.clip(nn[2], -1, 1))))
        base = r["sk"].V[lf["path"][0]]
        ext = 0.0
        if len(mem) > 1:
            d = Cc[:, None, :2] - Cc[None, :, :2]; ext = float(np.sqrt((d ** 2).sum(-1)).max())
        leaves.append(dict(internal_id=lf["id"], members=mem, n_leaflets=len(mem), area_m2=float(A.sum()), inclination_area_weighted_deg=incl_w,
                           inclination_plane_deg=incl_p, centroid=cen, azimuth_deg=deg_azimuth(cen[0] - base[0], cen[1] - base[1]), horizontal_extent_m=ext,
                           support_min=int(min(x["support_min"] for x in mem)), support_median=float(np.median([x["support_views"] for x in mem])),
                           uncertain=any(x["uncertain"] or "ambiguous" in x["flags"] for x in mem), path=r["sk"].V[lf["path"]], leaf_flags=list(lf["flags"]),
                           grouping_source="auto"))
    leaves.sort(key=lambda L: (round(L["centroid"][2], 9), round(L["centroid"][0], 9)))
    for k, L in enumerate(leaves):
        L["leaf_id"] = f"CL{k + 1:02d}"
        for x in L["members"]:
            x["leaf_id"] = L["leaf_id"]
    for x in lfs:
        x.setdefault("leaf_id", "unassigned")
    # coverage buckets (acceptance A1)
    assigned = np.isin(cl, [x["cluster"] for x in lfs if x["leaf_id"] != "unassigned"])
    unassigned = np.isin(cl, [x["cluster"] for x in lfs if x["leaf_id"] == "unassigned"])
    buckets = dict(leaflet_assigned=int(assigned.sum()), leaflet_unassigned=int(unassigned.sum()),
                   blade_unclustered=int(((cl < 0) & ~is_axis).sum()), axis=int(((cl < 0) & is_axis).sum()))
    cov_ok = sum(buckets.values()) == int(keep.sum()) and not (assigned & unassigned).any()
    bucket = np.where(assigned, 1, np.where(unassigned, 2, np.where(is_axis, 4, 3))).astype(np.uint8)
    return dict(n_all=n_all, n_opaque=int(opaque.sum()), n_used=int(keep.sum()), n_opaque_below=int((opaque & ~above).sum()), H=H, rows=rows, xyz=xyz,
                is_axis=is_axis, cl=cl, bucket=bucket, n_seen=n_seen, label_source=dict(Counter(src.tolist())), leaflets=lfs, leaves=leaves,
                buckets=buckets, coverage_ok=bool(cov_ok), counts=r["counts"], n_views=len(views), stubs=len(r["stubs"]), trusses=len(r["trusses"]))


# ---------------------------------------------------------------- tables
LEAFLET_UNITS = ("# units: *_m metres (board frame, z up), *_m2 one-sided lamina area, *_deg degrees, *_rel_H fraction of plant height H; "
                 "all metric values conditional_on_unvalidated_scale; input_status R3-1b_unaccepted; shape only, not a measurement")
LEAFLET_COLS = ["leaflet_id", "external_leaflet_id", "leaf_id", "grouping_source", "input_status", "scale_tag", "centroid_x_m", "centroid_y_m", "centroid_z_m",
                "centroid_z_rel_H", "area_m2", "area_rel_H2", "area_hull_m2", "area_sum_ellipse_m2", "length_m", "length_rel_H", "width_m", "lw_ratio",
                "normal_x", "normal_y", "normal_z", "inclination_deg", "normal_elevation_deg", "normal_azimuth_deg", "n_surfels", "support_views_median",
                "support_views_min", "planarity", "uncertain", "flags", "leaflet_type", "leaflet_side", "leaflet_pair", "axis_membership", "node_index", "node_rank"]
LEAF_UNITS = LEAFLET_UNITS
LEAF_COLS = ["leaf_id", "internal_id", "grouping_source", "input_status", "scale_tag", "n_leaflets", "leaflet_ids", "area_m2", "area_rel_H2",
             "inclination_area_weighted_deg", "inclination_plane_deg", "centroid_x_m", "centroid_y_m", "centroid_z_m", "centroid_z_rel_H", "azimuth_deg",
             "horizontal_extent_m", "horizontal_extent_rel_H", "rachis", "n_primary_pairs", "n_intercalary", "has_secondary_leaflets", "n_bend_nodes",
             "compound_leaf_insertion_angle_deg", "support_views_min", "support_views_median", "uncertain", "flags",
             "axis_membership", "axis_kind", "node_index", "node_rank", "attachment"]


def mtg_derived(members):
    """MTG_CONVENTION 2026-10-10 section 6: attributes derived from the child leaflets, never typed in.
    n_primary_pairs = distinct lateral pair indices (P labels); n_intercalary = I labels; has_secondary_leaflets = the
    pipeline has no secondary level, so False means 'none detected', not 'none present'."""
    pairs = set(); inter = 0
    for x in members:
        t, _, pr = parse_label(x["label"]) if x["label"] else ("T", "C", 0)
        if t == "P": pairs.add(pr)
        elif t == "I": inter += 1
    return dict(n_primary_pairs=len(pairs), n_intercalary=inter, has_secondary_leaflets="false_none_detected")


def _csv(units, cols, rows_):
    buf = io.StringIO(newline=""); buf.write(units + "\n"); w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n"); w.writeheader()
    for row in rows_:
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in cols})
    return buf.getvalue()


def tables(R, status):
    H = R["H"]; H2 = H * H
    lrows = []
    for x in R["leaflets"]:
        t, sd, pr = parse_label(x["label"]) if x["label"] else (UNAVAILABLE, UNAVAILABLE, UNAVAILABLE)
        if x["leaf_id"] == "unassigned":
            t, sd, pr = UNAVAILABLE, UNAVAILABLE, UNAVAILABLE
        c = x["centroid"]; nn = x["normal"]
        lrows.append(dict(leaflet_id=x["leaflet_id"], external_leaflet_id=f"{status}|{x['leaflet_id']}", leaf_id=x["leaf_id"],
                          grouping_source=("auto" if x["leaf_id"] != "unassigned" else "none"), input_status=status, scale_tag=SCALE_TAG,
                          centroid_x_m=f6(c[0]), centroid_y_m=f6(c[1]), centroid_z_m=f6(c[2]), centroid_z_rel_H=f6((c[2] - POT_TOP_Z_OF[0]) / H),
                          area_m2="%.8f" % x["area_m2"], area_rel_H2=f6(x["area_m2"] / H2), area_hull_m2="%.8f" % x["area_hull_m2"],
                          area_sum_ellipse_m2="%.8f" % x["area_sum_ellipse_m2"], length_m=f6(x["length_m"]), length_rel_H=f6(x["length_m"] / H),
                          width_m=f6(x["width_m"]), lw_ratio=(f6(x["lw_ratio"]) if x["lw_ratio"] else UNAVAILABLE),
                          normal_x=f6(nn[0]), normal_y=f6(nn[1]), normal_z=f6(nn[2]), inclination_deg="%.2f" % x["inclination_deg"],
                          normal_elevation_deg="%.2f" % x["normal_elevation_deg"],
                          normal_azimuth_deg=("%.2f" % x["normal_azimuth_deg"] if x["normal_azimuth_deg"] is not None else UNAVAILABLE),
                          n_surfels=x["n_surfels"], support_views_median="%.1f" % x["support_views"], support_views_min=x["support_min"],
                          planarity="%.4f" % x["planarity"], uncertain=str(x["uncertain"]).lower(), flags=";".join(x["flags"]),
                          leaflet_type=t, leaflet_side=sd, leaflet_pair=pr, axis_membership=UNAVAILABLE, node_index=UNAVAILABLE, node_rank=UNAVAILABLE))
    crows = []
    for L in R["leaves"]:
        c = L["centroid"]
        crows.append(dict(leaf_id=L["leaf_id"], internal_id=L["internal_id"], grouping_source=L["grouping_source"], input_status=status, scale_tag=SCALE_TAG,
                          n_leaflets=L["n_leaflets"], leaflet_ids=";".join(x["leaflet_id"] for x in L["members"]), area_m2="%.8f" % L["area_m2"],
                          area_rel_H2=f6(L["area_m2"] / H2), inclination_area_weighted_deg="%.2f" % L["inclination_area_weighted_deg"],
                          inclination_plane_deg=("%.2f" % L["inclination_plane_deg"] if L["inclination_plane_deg"] is not None else UNAVAILABLE),
                          centroid_x_m=f6(c[0]), centroid_y_m=f6(c[1]), centroid_z_m=f6(c[2]), centroid_z_rel_H=f6((c[2] - POT_TOP_Z_OF[0]) / H),
                          azimuth_deg="%.2f" % L["azimuth_deg"], horizontal_extent_m=f6(L["horizontal_extent_m"]), horizontal_extent_rel_H=f6(L["horizontal_extent_m"] / H),
                          rachis=UNAVAILABLE, **mtg_derived(L["members"]), n_bend_nodes=UNAVAILABLE, compound_leaf_insertion_angle_deg=UNAVAILABLE,
                          support_views_min=L["support_min"], support_views_median="%.1f" % L["support_median"],
                          uncertain=str(L["uncertain"]).lower(), flags=";".join(L["leaf_flags"]), axis_membership=UNAVAILABLE, axis_kind=UNAVAILABLE,
                          node_index=UNAVAILABLE, node_rank=UNAVAILABLE, attachment=UNAVAILABLE))
    return _csv(LEAFLET_UNITS, LEAFLET_COLS, lrows), _csv(LEAF_UNITS, LEAF_COLS, crows)


POT_TOP_Z_OF = [POT_TOP_Z]      # set from the CLI so the tables use the declared plane


# ---------------------------------------------------------------- plant_tree_v1
def plant_tree(R, a, status, input_sha):
    tree = new_tree("reconstruction", "board_m", "board metres, z up", status, plant_id=status, synthetic=False)
    tree["provenance"].update(axes=UNAVAILABLE, nodes=UNAVAILABLE, rank=UNAVAILABLE, attachment=UNAVAILABLE, trusses=UNAVAILABLE, rachis=UNAVAILABLE,
                              leaflet_geometry="reconstructed", area_mm2="reconstructed", normal="reconstructed")
    tree["plant"].update(flags=["requires_declaration"], attrs=dict(organ_split="leaflets_and_compound_leaves_only", input_surfels=str(a.surfels), input_sha256=input_sha,
                                                                  scale=SCALE_TAG, plant_height_H_m=R["H"], pot_top_z_m=a.pot_top, stem_topology=UNAVAILABLE))
    root = [float(x) for x in a.root]
    tree["axes"] = [dict(id="M", kind="main", parent_axis=None, parent_node=None, flags=["requires_declaration"], provenance=dict(tube=UNAVAILABLE),
                         note="placeholder required by the validator; the main axis was NOT resolved in this run",
                         tube=dict(kind="tube", polyline=[root, [root[0], root[1], root[2] + R["H"]]], radius=[0.0, 0.0], segs=3))]
    tree["nodes"] = [dict(id="N_unavailable", axis="M", index=1, rank=None, s_mm=0.0, position=root, leaves=[L["leaf_id"] for L in R["leaves"]], trusses=[],
                          flags=["requires_declaration"], provenance=dict(index=UNAVAILABLE, rank=UNAVAILABLE, position=UNAVAILABLE),
                          note="placeholder: node index / rank / attachment unavailable in this run")]
    tree["leaves"], tree["leaflets"] = [], []
    for L in R["leaves"]:
        pl = [[float(v) for v in p] for p in L["path"]]
        if len(pl) < 2:
            pl = [pl[0], [pl[0][0], pl[0][1], pl[0][2] + 1e-4]]
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array(pl), axis=0), axis=1))] * 1000
        att = []
        for x in L["members"]:
            j = int(np.argmin(np.linalg.norm(np.array(pl) - x["centroid"], axis=1))); att.append((float(s[j]), x["leaflet_id"]))
            t, sd, pr = parse_label(x["label"]) if x["label"] else ("T", "C", 0)
            fl = [f for f in ("few_views", "ambiguous") if f in x["flags"]]
            tree["leaflets"].append(dict(id=x["leaflet_id"], leaf=L["leaf_id"], type=t, side=sd, pair=pr, attachment_s_mm=float(s[j]), base=[float(v) for v in x["centroid"]],
                                         frame=dict(d=x["e1"].tolist(), n=x["normal"].tolist(), b=x["e2"].tolist()), length_mm=x["length_m"] * 1000, width_mm=x["width_m"] * 1000,
                                         area_mm2=x["area_m2"] * 1e6, normal=x["normal"].tolist(), surface=dict(kind="ref", ref=f"leaflet_surfels.json#{x['leaflet_id']}"),
                                         petiolule=None, age_d=None, expansion=None, flags=fl,
                                         quality=dict(n_surfels=x["n_surfels"], support_views_median=x["support_views"], planarity=x["planarity"], flags=x["flags"],
                                                      uncertain=x["uncertain"], scale=SCALE_TAG, inclination_deg=x["inclination_deg"])))
        groups = defaultdict(list)
        for sm, lid in att:
            groups[round(sm, 3)].append(lid)
        tree["leaves"].append(dict(id=L["leaf_id"], node="N_unavailable", axis="M", rank=None, petiole=None,
                                   rachis=dict(kind="tube", polyline=pl, radius=[0.0] * len(pl), segs=3), attachments=[dict(s_mm=k, leaflets=v) for k, v in sorted(groups.items())],
                                   length_mm=float(s[-1]), azimuth_deg=L["azimuth_deg"], elevation_deg=None, age_d=None, expansion=None, flags=["requires_declaration"],
                                   provenance=dict(rachis="reconstructed", node=UNAVAILABLE, rank=UNAVAILABLE),
                                   note="rachis tube = skeleton segment the grouping used (not a resolved rachis); node placeholder",
                                   attrs=dict(internal_id=L["internal_id"], grouping_source=L["grouping_source"], area_mm2=L["area_m2"] * 1e6, **mtg_derived(L["members"]),
                                              n_bend_nodes=UNAVAILABLE, compound_leaf_insertion_angle_deg=UNAVAILABLE, leaflet_insertion_angle_deg=UNAVAILABLE,
                                              inclination_area_weighted_deg=L["inclination_area_weighted_deg"], inclination_plane_deg=L["inclination_plane_deg"],
                                              horizontal_extent_mm=L["horizontal_extent_m"] * 1000, uncertain=L["uncertain"], scale=SCALE_TAG)))
    tree["trusses"] = []; tree["cotyledons"] = []
    tree["unassigned_leaflets"] = [dict(id=x["leaflet_id"], base=[float(v) for v in x["centroid"]], area_mm2=x["area_m2"] * 1e6, normal=x["normal"].tolist(),
                                        flags=["unattached"], quality=dict(flags=x["flags"], n_surfels=x["n_surfels"]), surface=dict(kind="ref", ref=f"leaflet_surfels.json#{x['leaflet_id']}"))
                                   for x in R["leaflets"] if x["leaf_id"] == "unassigned"]
    tree["notes"] = ("leaf and leaflet levels only; axis / node / rank / attachment / trusses unavailable (placeholders flagged requires_declaration); "
                     f"every metric value {SCALE_TAG}; {status}; not a measurement, not a validated structure")
    validate(tree)
    return tree


# ---------------------------------------------------------------- distributions for microtom_synth_v2
def _dist(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return dict(n=0)
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0, min=float(v.min()), max=float(v.max()),
                p05=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p95=float(q[4]))


def distributions(R, status, input_sha):
    lf = [x for x in R["leaflets"]]; asg = [x for x in lf if x["leaf_id"] != "unassigned"]
    per_leaf = [L["n_leaflets"] for L in R["leaves"]]
    lengths = [x["length_m"] * 1000 for x in lf]; largest = max(lengths) if lengths else None
    ratios = [x["lw_ratio"] for x in lf if x["lw_ratio"]]
    d = dict(schema="light_organs_distributions_v1", for_generator="microtom_synth_v2", source="reconstruction", input_status=status, input_sha256=input_sha,
             scale=SCALE_TAG, plant_height_H_m=R["H"], note="read with microtom_v2 --params (keys outside DEF are ignored by the generator); every mm value is "
             "conditional on the unvalidated board scale; distributions carry n and spread; nothing here is a measurement",
             params={k: v for k, v in dict(lf_len=largest, lw_ratio=(float(np.mean(ratios)) if ratios else None),
                                           max_pairs=(int(max(1, round((np.median(per_leaf) - 1) / 2))) if per_leaf else None), visible=len(R["leaves"])).items() if v is not None},
             params_provenance=dict(lf_len="reconstructed:" + SCALE_TAG, lw_ratio="reconstructed:" + SCALE_TAG, max_pairs="reconstructed (median leaflets per leaf - 1) / 2",
                                    visible="reconstructed compound-leaf count (grouping unvalidated)"),
             distributions=dict(leaflets_per_compound_leaf=_dist(per_leaf), leaflet_area_mm2=_dist([x["area_m2"] * 1e6 for x in lf]),
                                leaflet_area_rel_H2=_dist([x["area_m2"] / R["H"] ** 2 for x in lf]), leaflet_length_mm=_dist(lengths),
                                leaflet_lw_ratio=_dist(ratios), leaflet_inclination_deg=_dist([x["inclination_deg"] for x in lf]),
                                leaflet_normal_elevation_deg=_dist([x["normal_elevation_deg"] for x in lf]),
                                compound_leaf_area_mm2=_dist([L["area_m2"] * 1e6 for L in R["leaves"]]),
                                compound_leaf_inclination_area_weighted_deg=_dist([L["inclination_area_weighted_deg"] for L in R["leaves"]]),
                                compound_leaf_inclination_plane_deg=_dist([L["inclination_plane_deg"] for L in R["leaves"]]),
                                compound_leaf_horizontal_extent_mm=_dist([L["horizontal_extent_m"] * 1000 for L in R["leaves"]])),
             counts=dict(leaflets=len(lf), leaflets_assigned=len(asg), leaflets_unassigned=len(lf) - len(asg), compound_leaves=len(R["leaves"])))
    return d


# ---------------------------------------------------------------- plausibility (report only)
def plausibility(R):
    lf = R["leaflets"]; lengths = sorted([x["length_m"] * 1000 for x in lf], reverse=True); ratios = [x["lw_ratio"] for x in lf if x["lw_ratio"]]
    per_leaf = [L["n_leaflets"] for L in R["leaves"]]
    return dict(scale=SCALE_TAG,
                check1_largest_leaflet_length_mm=dict(reference="18.6 +- 3.1 (14.2-22.5) user dissection", largest=lengths[0] if lengths else None,
                                                      top5=lengths[:5], within_14p2_22p5=(14.2 <= lengths[0] <= 22.5) if lengths else None),
                check2_lw_ratio=dict(reference="1.0-2.2, mean ~1.5", **_dist(ratios), share_in_1p0_2p2=(float(np.mean([(1.0 <= r <= 2.2) for r in ratios])) if ratios else None)),
                check3_leaflets_per_compound_leaf=dict(reference="7 (dissection) / 5 (Carvalho 2011)", **_dist(per_leaf), histogram=dict(Counter(per_leaf))),
                check4_compound_leaves_per_plant=dict(reference="6-8 at ~60 days (user)", found=len(R["leaves"]), unassigned_leaflets=sum(x["leaf_id"] == "unassigned" for x in lf)),
                check5_human_topview_masks=dict(status=UNAVAILABLE, reason="no human leaflet masks exist on this plant's 2026-10-21 photos (libraries cover tray_top_01-09 and dwt_101-104)"))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v, type=Path)
    ap.add_argument("--input-status", default="R3-1b_unaccepted"); ap.add_argument("--pot-top", type=float, default=POT_TOP_Z)
    ap.add_argument("--root", type=float, nargs=3, default=list(C.POT_CENTRE_M)); ap.add_argument("--repeat", action="store_true", help="run twice, compare table hashes")
    a = ap.parse_args(argv); status = a.input_status; POT_TOP_Z_OF[0] = a.pot_top
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); input_sha = sha256_file(a.surfels)
    R = compute(a); lcsv, ccsv = tables(R, status)
    repro = None
    if a.repeat:
        R2 = compute(a); l2, c2 = tables(R2, status)
        repro = dict(leaflets_csv_identical=hashlib.sha256(l2.encode()).hexdigest() == hashlib.sha256(lcsv.encode()).hexdigest(),
                     compound_leaves_csv_identical=hashlib.sha256(c2.encode()).hexdigest() == hashlib.sha256(ccsv.encode()).hexdigest())
    (out / "leaflets.csv").write_text(lcsv, encoding="utf-8"); (out / "compound_leaves.csv").write_text(ccsv, encoding="utf-8")
    json.dump({x["leaflet_id"]: dict(leaf_id=x["leaf_id"], surfel_rows=x["rows"]) for x in R["leaflets"]} | {"_meta": dict(input_status=status, input_sha256=input_sha, note="surfel_rows = row index into the input PLY")},
              open(out / "leaflet_surfels.json", "w", encoding="utf-8"))
    tree = plant_tree(R, a, status, input_sha); json.dump(tree, open(out / "plant_tree.json", "w", encoding="utf-8"), indent=1)
    dist = distributions(R, status, input_sha); json.dump(dist, open(out / "distributions_microtom_synth_v2.json", "w", encoding="utf-8"), indent=1)
    np.savez_compressed(out / "surfel_labels.npz", rows=R["rows"], bucket=R["bucket"], cl=R["cl"], is_axis=R["is_axis"], n_seen=R["n_seen"],
                        leaflet_id=np.array([next((x["leaflet_id"] for x in R["leaflets"] if x["cluster"] == c), "") for c in range(R["cl"].max() + 1)]),
                        leaf_id=np.array([next((x["leaf_id"] for x in R["leaflets"] if x["cluster"] == c), "") for c in range(R["cl"].max() + 1)]))
    pl = plausibility(R)
    summ = dict(input_status=status, scale=SCALE_TAG, input=str(a.surfels), input_sha256=input_sha, pot_top_z_m=a.pot_top, root_m=list(a.root), plant_height_H_m=R["H"],
                surfels=dict(total=R["n_all"], opaque=R["n_opaque"], opaque_below_pot_top=R["n_opaque_below"], used=R["n_used"]), label_source=R["label_source"],
                pipeline_counts=R["counts"], training_views=R["n_views"], buckets=R["buckets"], coverage_ok=R["coverage_ok"],
                leaflets=len(R["leaflets"]), leaflets_unassigned=sum(x["leaf_id"] == "unassigned" for x in R["leaflets"]), leaflets_uncertain=sum(x["uncertain"] for x in R["leaflets"]),
                compound_leaves=len(R["leaves"]), skeleton_stubs_not_reported=R["stubs"], truss_candidates_not_reported=R["trusses"],
                flags=dict(Counter(f for x in R["leaflets"] for f in x["flags"])), reproducibility=repro, plant_tree_valid=True, plausibility=pl,
                unavailable=dict(axis_membership="not resolved (node rules fail the synthetic gate)", main_vs_lateral="not resolved", node_index_rank_attachment="not resolved",
                                 flowers_fruit_truss="out of scope", rachis="unavailable (skeleton segment kept as geometry only)"),
                floors=FLOORS, review_limits=REVIEW_LIMITS)
    json.dump(summ, open(out / "run_summary.json", "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ("surfels", "buckets", "coverage_ok", "leaflets", "leaflets_unassigned", "compound_leaves", "reproducibility", "plant_height_H_m")}, default=float))


if __name__ == "__main__":
    main()
