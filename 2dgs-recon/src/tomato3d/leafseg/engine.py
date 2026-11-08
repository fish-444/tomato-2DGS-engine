"""Option A per image (contract §3.4–3.10): OBB v5 boxes -> structured prompts -> tuned SAM 2.1 tiny v5 with sequential
peeling -> mask-IoU NMS + containment -> leaf-only rule -> T1 relative-evidence order -> occlusion flags, compound IDs,
faces, variety tag, failure candidates. Masks live on a working grid (max side GRID_MAX px); SAM runs at full size."""

from __future__ import annotations

import cv2
import numpy as np

from tomato3d.leafseg import pixels as px
from tomato3d.topview.occlusion import features as t1

GRID_MAX = 1536
T1_GRID = 1008
DET_ARGS = dict(imgsz=1280, conf=0.25, iou=0.7, max_det=300, device="cpu", augment=False, half=False)
NMS_IOU, CONTAIN, FG_MIN, RACHIS_FRAC_MAX, AREA_MIN_FRAC = 0.55, 0.85, 0.5, 0.40, 0.0002
N_NEG_RACHIS, N_NEG_PEEL = 4, 3
GREEN_FRAC = 0.6
FACE_MAJ, FACE_BOTH = 0.7, 0.3


def _solidity(m: np.ndarray) -> float:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return 0.0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); comp = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cs)); ha = cv2.contourArea(hull)
    return float(comp.sum() / ha) if ha > 0 else 0.0


def _to_grid(m_full: np.ndarray, gshape) -> np.ndarray:
    return cv2.resize(m_full.astype(np.float32), (gshape[1], gshape[0]), interpolation=cv2.INTER_AREA) >= 0.5


def _sample_points(mask: np.ndarray, n: int, rng) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((0, 2))
    idx = rng.choice(len(xs), min(n, len(xs)), replace=False)
    return np.c_[xs[idx], ys[idx]].astype(np.float32)


def detect(det, rgb: np.ndarray) -> list[dict]:
    p = det.predict(np.ascontiguousarray(rgb[:, :, ::-1]), verbose=False, **DET_ARGS)[0]
    if p.obb is None or not len(p.obb):
        return []
    poly = p.obb.xyxyxyxy.cpu().numpy(); conf = p.obb.conf.cpu().numpy(); out = []
    for k in range(len(poly)):
        x0, y0 = poly[k].min(0); x1, y1 = poly[k].max(0)
        out.append(dict(det=k, conf=float(conf[k]), obb=poly[k].tolist(), box=[float(x0), float(y0), float(x1), float(y1)],
                        centre=[float((x0 + x1) / 2), float((y0 + y1) / 2)]))
    return out


def _predict(sam, box_full, pos_full, neg_full, gshape):
    pts = np.vstack([pos_full, neg_full]) if len(neg_full) else pos_full
    lab = np.r_[np.ones(len(pos_full)), np.zeros(len(neg_full))].astype(np.int32)
    masks, ious, _ = sam.predictor.predict(box=np.asarray(box_full, np.float32), point_coords=pts.astype(np.float32), point_labels=lab,
                                           multimask_output=True)
    best = None
    for m, q in zip(masks, ious):
        g = _to_grid(m > 0, gshape); sc = float(q) * _solidity(g)
        if best is None or sc > best[1]:
            best = (g, sc, float(q))
    return best


def run_image(rgb: np.ndarray, det, sam, rf, cfg: dict, seed: int = 0, dets: list[dict] | None = None) -> dict:
    rng = np.random.default_rng(seed); H, W = rgb.shape[:2]; s = min(1.0, GRID_MAX / max(H, W))
    gshape = (int(round(H * s)), int(round(W * s))); rgb_g = cv2.resize(rgb, (gshape[1], gshape[0]), interpolation=cv2.INTER_AREA)
    pp = px.preprocess(rgb_g); F = px.pixel_features(pp); fg_p = px.foreground_prob(rf, F, gshape); fg = fg_p > 0.5
    rach = px.rachis_skeleton(pp, fg); rach_d = cv2.dilate(rach["skeleton"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    dets = detect(det, rgb) if dets is None else dets; sam.predictor.set_image(rgb)
    # ---- pass 1: structured prompts per box
    cands = []
    for d in dets:
        x0, y0, x1, y1 = [int(round(v * s)) for v in d["box"]]; x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, gshape[1] - 1), min(y1, gshape[0] - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        reg = np.zeros(gshape, bool); reg[y0:y1 + 1, x0:x1 + 1] = True
        area = reg & fg & ~rach_d
        if area.any():
            dt = cv2.distanceTransform(area.astype(np.uint8), cv2.DIST_L2, 5); yy, xx = np.unravel_index(int(np.argmax(dt)), dt.shape); pos = np.array([[xx, yy]], np.float32)
        else:
            pos = np.array([[(x0 + x1) / 2, (y0 + y1) / 2]], np.float32)
        neg = _sample_points(rach["skeleton"] & reg, N_NEG_RACHIS, rng)
        g, sc, q = _predict(sam, d["box"], pos / s, neg / s, gshape)
        cands.append(dict(det=d, mask=g, score=sc, iou_pred=q, pos=pos, neg_rachis=neg, reg=(x0, y0, x1, y1)))
    # ---- pass 2: sequential peeling (accepted masks become negative prompts of later, overlapping boxes)
    cands.sort(key=lambda c: -c["score"]); accepted = []
    for rank, c in enumerate(cands):
        x0, y0, x1, y1 = c["reg"]; reg = np.zeros(gshape, bool); reg[y0:y1 + 1, x0:x1 + 1] = True
        peel = np.zeros(gshape, bool)
        for a in accepted:
            peel |= a["mask"] & reg
        if peel.any():
            negp = _sample_points(peel & ~cv2.dilate(c["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool), N_NEG_PEEL, rng)
            if len(negp):
                g, sc, q = _predict(sam, c["det"]["box"], c["pos"] / s, np.vstack([c["neg_rachis"], negp]) / s if len(c["neg_rachis"]) else negp / s, gshape)
                c.update(mask=g, score=sc, iou_pred=q)
        c["peel_rank"] = rank; accepted.append(c)
    # ---- partition: mask-IoU NMS + containment (never box-based)
    accepted.sort(key=lambda c: -c["score"]); kept, dropped = [], []
    for c in accepted:
        m = c["mask"]; a = m.sum(); why = None
        if a == 0:
            why = "empty"
        for k in kept:
            inter = (m & k["mask"]).sum()
            if inter / max((m | k["mask"]).sum(), 1) > NMS_IOU:
                why = "nms"; break
            if inter / max(a, 1) > CONTAIN:
                why = "contained"; break
        (dropped if why else kept).append(c)
        if why:
            c["drop"] = why
    # ---- leaf-only rule
    final = []; img_area = gshape[0] * gshape[1]
    for c in kept:
        m = c["mask"]; c["fg_mean"] = float(fg_p[m].mean()) if m.any() else 0.0
        med = t1.skeleton(cv2.resize(m.astype(np.uint8), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) > 0)
        med = cv2.resize(med.astype(np.uint8), (gshape[1], gshape[0]), interpolation=cv2.INTER_NEAREST) > 0
        med = cv2.dilate(med.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        c["rachis_fraction"] = float((med & rach_d).sum() / max(med.sum(), 1))
        if c["fg_mean"] < FG_MIN:
            c["drop"] = "low_foreground"
        elif c["rachis_fraction"] > RACHIS_FRAC_MAX:
            c["drop"] = "rachis_kept_as_leaf"
        elif m.sum() < AREA_MIN_FRAC * img_area:
            c["drop"] = "too_small"
        (dropped if c.get("drop") else final).append(c)
    for i, c in enumerate(sorted(final, key=lambda c: (c["det"]["centre"][1], c["det"]["centre"][0]))):
        c["id"] = f"L{i + 1:03d}"
    final.sort(key=lambda c: c["id"])
    res = finish(final, dets, rgb_g, fg, rach, cfg, gshape, s, (H, W))
    res.update(detections=len(dets), candidates=len(cands),
               dropped=[dict(det=c["det"]["det"], reason=c["drop"], score=c["score"], rachis_fraction=c.get("rachis_fraction"), fg_mean=c.get("fg_mean")) for c in dropped])
    return res


def finish(final: list[dict], dets: list[dict], rgb_g: np.ndarray, fg: np.ndarray, rach: dict, cfg: dict, gshape, s: float,
           image_hw, rank_key=None) -> dict:
    """Recut -> T1 order -> occlusion flags, compound ids, faces, variety, flags for a given mask set (`final` entries need
    id, mask, score, det{det, conf, centre}, fg_mean, rachis_fraction, iou_pred, peel_rank). `rank_key(i)` orders the first
    recut (default: score; must return a tuple); variant D ranks recovered masks below the base masks."""
    H, W = image_hw; rank_key = rank_key or (lambda i: (-final[i]["score"],))
    rach_d = cv2.dilate(rach["skeleton"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # ---- recut by score -> disjoint labels; T1 order on the processing grid
    def recut(owner_of):
        lab = np.zeros(gshape, np.int32)
        order = sorted(range(len(final)), key=owner_of)
        for i in order:
            lab[final[i]["mask"] & (lab == 0)] = i + 1
        return lab
    lab = recut(rank_key)
    t = min(1.0, T1_GRID / max(gshape)); tshape = (int(round(gshape[0] * t)), int(round(gshape[1] * t)))
    rgb_t = cv2.resize(rgb_g, (tshape[1], tshape[0]), interpolation=cv2.INTER_AREA)
    lab_t = cv2.resize(lab.astype(np.int32), (tshape[1], tshape[0]), interpolation=cv2.INTER_NEAREST)
    plant_t = lab_t > 0; feat = t1.maps(rgb_t, plant_t, cfg)
    faces, face_info = t1.face_clusters(feat, plant_t, cfg)
    pairs, edges = [], {}
    n = len(final)
    for i in range(n):
        ai = lab_t == i + 1
        if not ai.any():
            continue
        for j in range(i + 1, n):
            bj = lab_t == j + 1
            if not bj.any() or not t1.contact(ai, bj, cfg["contact_radius_px"]).sum() >= cfg["min_boundary_px"]:
                continue
            rows = t1.boundary_features(ai, bj, feat, cfg); pl = t1.pool(rows, cfg)
            fe = t1.fold_evidence(ai, bj, feat, faces, cfg); fold_ok, fold_fail = t1.fold_gate(fe, cfg)
            pr = dict(a=final[i]["id"], b=final[j]["id"], **pl, fold_gate=fold_ok)
            pairs.append(pr)
            if pl["decision"] in ("A", "B"):
                over, under = (i, j) if pl["decision"] == "A" else (j, i)
                edges[(over, under)] = pl["confidence"]
    # cycles: drop the lowest-margin edge until acyclic
    while True:
        g = t1.graph_order([str(i) for i in range(n)], [dict(a=str(a), b=str(b), decision="A", confidence=c) for (a, b), c in edges.items()])
        if g["status"] == "acyclic":
            break
        a, b = g["cycles"][0]["weakest_edge"]; edges.pop((int(a), int(b)), None)
    # second recut: overlap pixels go to the instance ranked over, else the higher score
    for i, c in enumerate(final):
        c["occluders"] = [final[o]["id"] for (o, u) in edges if u == i]
        c["occluded"] = bool(c["occluders"])
    rank = {i: 0 for i in range(n)}
    for (o, u) in edges:
        rank[o] += 1
    lab = recut(lambda i: (-rank[i],) + rank_key(i))
    # ---- compound ids (rachis components), faces, variety, flags
    ncomp, comp = cv2.connectedComponents(rach["skeleton"].astype(np.uint8), connectivity=8)
    rg = rgb_g.astype(np.float32); ssum = rg.sum(-1) + 1e-6; r_, g_, b_ = rg[..., 0] / ssum, rg[..., 1] / ssum, rg[..., 2] / ssum
    exgr = 3 * g_ - 2.4 * r_ - b_
    centres = np.array([d["centre"] for d in dets]) * s if dets else np.zeros((0, 2))
    areas = [int((lab == i + 1).sum()) for i in range(n)]; med_area = float(np.median(areas)) if areas else 0.0
    insts = []
    for i, c in enumerate(final):
        vis = lab == i + 1; full = c["mask"]
        touch = comp[cv2.dilate(full.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0]; touch = touch[touch > 0]
        cid = f"K{int(np.bincount(touch).argmax()):03d}" if len(touch) else "unassigned"
        ft = cv2.resize(full.astype(np.uint8), (tshape[1], tshape[0]), interpolation=cv2.INTER_NEAREST) > 0
        fl = faces[ft]; fl = fl[fl >= 0]
        if face_info.get("status") != "provisional" or not len(fl):
            face = "unknown"
        else:
            fa = float(fl.mean()); face = "abaxial" if fa >= FACE_MAJ else "adaxial" if fa <= 1 - FACE_MAJ else "folded" if min(fa, 1 - fa) >= FACE_BOTH else "unknown"
        inside = [k for k, (cx, cy) in enumerate(centres) if k != c["det"]["det"] and 0 <= int(cy) < gshape[0] and 0 <= int(cx) < gshape[1] and full[int(cy), int(cx)]]
        flags = []
        if len(inside) >= 2 or (med_area and areas[i] > 3 * med_area):
            flags.append("merged")
        if full[0].any() or full[-1].any() or full[:, 0].any() or full[:, -1].any():
            flags.append("truncated_by_image")
        ys, xs = np.nonzero(vis)
        insts.append(dict(id=c["id"], category="leaflet", visible_segmentation=px.rle_encode(vis), proposal_segmentation=px.rle_encode(full),
                          amodal_segmentation=None, occluded=c["occluded"], occluded_fraction=None, truncated_by_image="truncated_by_image" in flags,
                          occluders=c["occluders"], face=face, compound_leaf_id=cid, plant_id="unassigned",
                          variety_group="green" if (exgr[full] > 0).mean() >= GREEN_FRAC else "silvery",
                          edge_uncertainty_px=None, source="model_proposal", score=c["score"], sam_iou_pred=c["iou_pred"], peel_rank=c["peel_rank"],
                          detector_conf=c["det"]["conf"], detector_class=c["det"].get("cls", "leaflet_obb"), fg_mean=c["fg_mean"], rachis_fraction=c["rachis_fraction"], flags=flags,
                          centroid=[float(xs.mean()), float(ys.mean())] if len(xs) else None, visible_px=int(vis.sum()), proposal_px=int(full.sum())))
    pscore = {}
    for p_ in pairs:
        pscore[(p_["a"], p_["b"])] = p_["score"]; pscore[(p_["b"], p_["a"])] = -p_["score"]
    order_edges = [dict(over=final[o]["id"], under=final[u]["id"], p=float(0.5 + 0.5 * np.tanh(pscore[(final[o]["id"], final[u]["id"])])),
                        margin=float(cf), source="t1_rule") for (o, u), cf in edges.items()]
    for inst in insts:
        inst["order_edges"] = [e for e in order_edges if inst["id"] in (e["over"], e["under"])]
    split_folded = [dict(a=p_["a"], b=p_["b"]) for p_ in pairs if p_["fold_gate"]]
    fg_area = int(fg.sum()); covered = int((fg & (lab > 0)).sum())
    return dict(grid_shape=list(gshape), grid_scale=s, image_size=[H, W],
                instances=insts, order_edges=order_edges, pairs=pairs, split_folded_candidates=split_folded,
                face_clusters=face_info, rachis=dict(skeleton_px=int(rach["skeleton"].sum()), ends=len(rach["ends"]), branches=len(rach["branches"]),
                                                     components=int(ncomp - 1)),
                unassigned_area_fraction=float(1 - covered / max(fg_area, 1)), foreground_px=fg_area,
                _layers=dict(label=lab, fg=fg, skeleton=rach["skeleton"], rgb=rgb_g))
