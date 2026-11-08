"""Pose-aware label pre-fill and automatic gap masks (contract build/p1_autolabel_20261026_01/CONTRACT.md).

Steps: poses (reconstruction pose vs board-only PnP on the stored corners), visibility (per-pixel visible-surfel index map + facing class),
pre-fill (lift user labels with the frozen V-H vote, project, assign a number + confidence per mask), gap masks (SAM 2.1 on uncovered plant
pixels), and the declared replay on R3-1b. Read-only on every input folder; writes only under build/p1_autolabel_20261026_01/.

usage: python -m tomato3d.organs.autolabel --step poses|visibility|replay
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.organs import all_compound as AC
from tomato3d.organs import auto_proposers as AP
from tomato3d.organs import light_organs as LO
from tomato3d.organs import userlabel as UL
from tomato3d.organs import vote2d as V2
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_autolabel_20261026_01"
LABELS = dict(status="R3-1b_unaccepted", gate="gate_failed_viewing_only", scale=LO.SCALE_TAG)
H2 = UL.H2
RES = V2.VISP["res"]
POSE = dict(max_rot_deg=3.0, max_centre_mm=10.0, min_corners=8)
FACING = dict(front=0.5, grazing=0.15)
CONF = dict(high_cover=0.7, med_cover=0.5, vote_share=0.8, front_share=0.5, nonedge_share=0.5, upper_share=0.5, ring_px=3)
GAP = dict(open_mm=5.5, area_pct=(5, 95), solidity=0.6, pad=0.10, inside_dilate_px=3, inside_min=0.95, iou_min=0.5, crop_frac=1 / 3, crop_pad=0.15, crop_long=1024, n_neg=3)
PASS = dict(high_acc=0.90, coverage=0.50, gap_inside=0.70)
R3_SEED = "060313903"


# ---------------------------------------------------------------- step 1: poses
def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2; return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def board_only_pose(det, K, dist):
    O, I = ca.pose_points(det)
    if len(O) < POSE["min_corners"]:
        return None
    ok, rv, tv = cv2.solvePnP(O, I, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    return ca.world_pose(rv.ravel(), tv.ravel()) if ok else None


def marker_direction_deg(det, centre_xy):
    if not det["ids"]:
        return None
    Pw = (ca.A_WB @ ca.obj_points(det["ids"]).T).T + ca.B_WB
    bc = ca.A_WB @ np.array([ca.SQUARES[0] * ca.SQUARE_M / 2, ca.SQUARES[1] * ca.SQUARE_M / 2, 0.0]) + ca.B_WB; c = Pw[:, :2].mean(0) - bc[:2]
    return float(np.degrees(np.arctan2(c[1], c[0])) % 360)


def individual_inputs(iid):
    """(views dict v -> dict(R_cw, t_cw, K, dist)), detections list, centre_xy, z_ref."""
    if iid == "R3-1b":
        po = json.load(open(LO.DEFAULTS["poses"], encoding="utf-8")); dets = json.load(open(LO.DEFAULTS["poses"].parent / "detections.json", encoding="utf-8"))
        K, dist = np.array(po["K_full"]), np.array(po["dist"])
        views = {v: dict(R_cw=np.array(d["R_cw"]), t_cw=np.array(d["t_cw"]), K=K, dist=dist) for v, d in po["views"].items() if d.get("accepted", 1)}
        return views, dets, np.array(V2.CENTRE, float), V2.POT_TOP, "pot_centre_vertical"
    pz = json.load(open(AC.POSES[iid], encoding="utf-8")); dets = json.load(open(AC.POSES[iid].parent / "detections_f35_52_3024.json", encoding="utf-8"))
    views = {v: dict(R_cw=np.array(pz["views"][v]["R_cw"]), t_cw=np.array(pz["views"][v]["t_cw"]), K=np.array(pz["views"][v]["K"]), dist=np.array(pz["views"][v]["dist"])) for v in pz["accepted"]}
    from tomato3d.fuse3d import surfels_io as sio
    sf = sio.read_surfels_full(AC.ALL / "outputs" / iid / f"R3-1b-style_{iid}_surfels_full.ply"); x = sf["xyz"][sf["opacity"] >= LO.OPACITY_MIN]
    return views, dets, x[:, :2].mean(0), float(np.percentile(x[:, 2], 2)), "plant_centroid_vertical"


def step_poses(ids):
    (OUT / "poses").mkdir(parents=True, exist_ok=True); summ = {}
    for iid in ids:
        views, dets, cxy, zref, axis = individual_inputs(iid); by = {d["file"]: d for d in dets}; rows = []
        for v in sorted(set(views) | set(by)):
            d = by.get(v); r = dict(individual=iid, view=v, axis=axis, corners=(d["corners"] if d else 0), markers=(d.get("markers") if d else 0), marker_ids=" ".join(map(str, (d.get("mk_ids") or d.get("ids") or []) if d else [])))
            rp = views.get(v); bp = board_only_pose(d, rp["K"] if rp else views[next(iter(views))]["K"], rp["dist"] if rp else views[next(iter(views))]["dist"]) if d else None
            if rp is None and bp is None:
                continue
            Rr, tr = (rp["R_cw"], rp["t_cw"]) if rp else bp
            c = -Rr.T @ tr; dxy = c[:2] - cxy; az = float(np.degrees(np.arctan2(dxy[1], dxy[0])) % 360); el = float(np.degrees(np.arctan2(c[2] - zref, np.hypot(*dxy))))
            r.update(azimuth_deg=round(az, 2), elevation_deg=round(el, 2), cam_x=round(float(c[0]), 4), cam_y=round(float(c[1]), 4), cam_z=round(float(c[2]), 4))
            if rp is not None and bp is not None:
                cb = -bp[0].T @ bp[1]; r.update(rot_diff_deg=round(rot_deg(Rr, bp[0]), 3), centre_diff_mm=round(float(np.linalg.norm(c - cb) * 1000), 2))
                r["flag"] = "pose_disagree" if (r["rot_diff_deg"] > POSE["max_rot_deg"] or r["centre_diff_mm"] > POSE["max_centre_mm"]) else "ok"
            elif rp is None:
                r["flag"] = "pose_board_only"
            else:
                r["flag"] = "no_board_pose"
            md = marker_direction_deg(d, cxy) if d else None
            r["marker_centroid_dir_deg"] = None if md is None else round(md, 1); r["marker_vs_azimuth_deg"] = None if md is None else round(AC.az_diff(md, az), 1)
            rows.append(r)
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        with open(OUT / "poses" / f"{iid}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        rd = [r["rot_diff_deg"] for r in rows if "rot_diff_deg" in r]; cd = [r["centre_diff_mm"] for r in rows if "centre_diff_mm" in r]
        mv = [r["marker_vs_azimuth_deg"] for r in rows if r.get("marker_vs_azimuth_deg") is not None]
        summ[iid] = dict(photos=len(rows), flags=dict(Counter(r["flag"] for r in rows)), rot_diff_deg_median=round(float(np.median(rd)), 3) if rd else None, rot_diff_deg_max=round(float(max(rd)), 3) if rd else None,
                         centre_diff_mm_median=round(float(np.median(cd)), 2) if cd else None, centre_diff_mm_max=round(float(max(cd)), 2) if cd else None,
                         marker_dir_within_90deg_share=round(float(np.mean(np.array(mv) <= 90)), 3) if mv else None)
        print(iid, json.dumps(summ[iid]), flush=True)
    json.dump(dict(summary=summ, rule=POSE, labels=LABELS, note="board-only pose = iterative PnP on the same stored corners: a consistency check, not an independent pose"), open(OUT / "poses" / "summary.json", "w", encoding="utf-8"), indent=1)
    return summ


# ---------------------------------------------------------------- step 2: visibility (R3-1b)
def r31b_plant():
    pl = UL.load_plant(); z = np.load(UL.VOTE2D / "stage_a_h2id.npz"); assert list(z["views"]) == pl["views"]; return pl, z["H2id"]


def visibility_map(pl, view):
    """Per-pixel visible-surfel index (1512, -1 = none), and per-surfel facing class / upper-side flag for visible surfels."""
    pv = V2.project_view(pl["xyz"], pl["sc"], pl["po"], view, RES); idx = np.nonzero(pv["vis"])[0]; img = np.full((RES, RES), -1, np.int32)
    r = np.minimum(V2.VISP["radius_sigma"] * pl["sc"][idx].max(1) * pv["K"][0, 0] / pv["z"][idx], V2.VISP["max_r_px"])
    order = np.argsort(-pv["z"][idx]); u, w = pv["u"][idx], pv["w"][idx]
    for k in order:
        cv2.circle(img, (int(u[k]), int(w[k])), int(max(1, round(r[k]))), int(idx[k]), -1)
    cam = -pv["R"].T @ pv["t"]; d = cam[None, :] - pl["xyz"]; d /= np.linalg.norm(d, axis=1, keepdims=True)
    nn = pl["nrm"] / np.maximum(np.linalg.norm(pl["nrm"], axis=1, keepdims=True), 1e-12); up = nn * np.where(nn[:, 2:3] < 0, -1, 1)
    cos = (up * d).sum(1); a = np.abs(cos); facing = np.where(a >= FACING["front"], 2, np.where(a >= FACING["grazing"], 1, 0)).astype(np.int8)
    return img, facing, (cos > 0), pv


def step_visibility():
    pl, H2id = r31b_plant(); vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); d = OUT / "visibility" / "R3-1b"; d.mkdir(parents=True, exist_ok=True)
    for vid in vc["chosen"] + [vc["held_out"]]:
        view = AP.view_file(vid); img, facing, upper, pv = visibility_map(pl, view); vis = img >= 0
        np.savez_compressed(d / f"{vid}.npz", index=img, facing=facing, upper=upper, labels=json.dumps(LABELS))
        s = img[vis]; print(vid, "visible px", int(vis.sum()), "front/grazing/edge", np.bincount(facing[s], minlength=3).tolist(), "upper", round(float(upper[s].mean()), 3), flush=True)


# ---------------------------------------------------------------- step 3: pre-fill
def to_grid(a):
    """1512 per-pixel array -> 1536 grid (nearest)."""
    ii = np.arange(H2) * RES // H2; return a[np.ix_(ii, ii)]


def predicted_layers(lab, share, facing, upper, idx1536):
    ok = idx1536 >= 0; j = np.where(ok, idx1536, 0)
    pred = np.where(ok, lab[j], 0); sh = np.where(ok, share[j], 0.0); fc = np.where(ok, facing[j], -1); up = np.where(ok, upper[j], False)
    return pred, sh, fc, up


def assign(mask, pred, sh, fc, up, cap=None):
    m = mask; n = int(m.sum())
    if n == 0:
        return dict(number=None, confidence="low", reason="empty")
    vals = pred[m]; c = Counter(vals[vals > 0].tolist())
    if not c:
        return dict(number=None, confidence="low", reason="no_projection", cover=0.0)
    L, k = c.most_common(1)[0]; cover = k / n; contrib = m & (pred == L)
    vs = float(sh[contrib].mean()); front = float((fc[contrib] == 2).mean()); nonedge = float((fc[contrib] >= 1).mean()); upper = float(up[contrib].mean())
    ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CONF["ring_px"] + 1,) * 2)) > 0
    other = bool(((pred > 0) & (pred != L) & ring).any())
    conf = "low"
    if not other and vs >= CONF["vote_share"] and upper >= CONF["upper_share"]:
        if cover >= CONF["high_cover"] and front >= CONF["front_share"]:
            conf = "high"
        elif cover >= CONF["med_cover"] and nonedge >= CONF["nonedge_share"]:
            conf = "medium"
    if cap == "medium" and conf == "high":
        conf = "medium"
    return dict(number=(int(L) if conf != "low" else None), candidate_number=int(L), confidence=conf, cover=round(cover, 3), vote_share=round(vs, 3), front=round(front, 3), nonedge=round(nonedge, 3),
                upper=round(upper, 3), other_leaf_within_3px=other)


def view_dir(pl, view):
    R = np.array(pl["po"]["views"][view]["R_cw"]); return R[2]          # camera optical axis in world


def next_view(pl, labelled, pool):
    best = None
    for v in pool:
        a = min(np.degrees(np.arccos(np.clip(view_dir(pl, AP.view_file(v)) @ view_dir(pl, AP.view_file(u)), -1, 1))) for u in labelled)
        if best is None or a < best[0]:
            best = (a, v)
    return best


# ---------------------------------------------------------------- truth (R3-1b human records)
def active_records(recs, vid):
    act = {}
    for r in sorted((r for r in recs.values() if r["view"] == vid), key=lambda r: r["at"]):
        if r.get("removed"):
            act.pop(r["removed"], None); continue
        act[r.get("mask_id") or r["id"]] = r
    return act


def truth_number(r, vid, nmap):
    if r["leaf"] in ("모름", "unknown", None):
        return 0
    t = nmap.get(f"{vid}|{int(r['leaf'])}"); return int(t) if t is not None and int(t) > 0 else 0


def truth_image(recs, vid, by_id, nmap):
    img = np.zeros((H2, H2), np.int16); any_ = np.zeros((H2, H2), bool)
    for r in active_records(recs, vid).values():
        m = UL.record_rle(r, recs, by_id)
        if m is None:
            continue
        any_ |= m; t = truth_number(r, vid, nmap)
        if t > 0:
            img[m] = t
    return img, any_


# ---------------------------------------------------------------- step 4: gap masks
def gap_components(plant, cands_union, ppm, area_lo, area_hi, extra=None):
    gap = plant & ~cands_union
    k = max(3, int(round(GAP["open_mm"] * ppm))); k += (k + 1) % 2
    gap = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    if extra is not None:
        gap = gap | (extra & plant & ~cands_union)
    n, lab, st, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), 8); comps = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if not (area_lo <= a <= area_hi):
            continue
        m = lab == i; hull = cv2.convexHull(np.argwhere(m)[:, ::-1].astype(np.int32)); sol = a / max(cv2.contourArea(hull), 1)
        if sol >= GAP["solidity"]:
            comps.append(m)
    return comps, int(gap.sum())


def dt_peak(m):
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); y, x = np.unravel_index(int(np.argmax(d)), d.shape); return x, y


def sam_gap_masks(sam, rgb, plant, comps, cand_masks, log):
    """SAM 2.1 per gap component (1536 grid in, full-res prompts); crop + upsample when the plant is small."""
    Hf = rgb.shape[0]; g2f = Hf / H2; ys, xs = np.nonzero(plant); out = []
    bb = [xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else [0, 0, H2 - 1, H2 - 1]; side = max(bb[2] - bb[0], bb[3] - bb[1]) / H2
    crop = None
    if side < GAP["crop_frac"]:
        pad = GAP["crop_pad"] * max(bb[2] - bb[0], bb[3] - bb[1]); x0, y0 = max(0, int((bb[0] - pad) * g2f)), max(0, int((bb[1] - pad) * g2f)); x1, y1 = min(Hf, int((bb[2] + pad) * g2f)), min(Hf, int((bb[3] + pad) * g2f))
        s = GAP["crop_long"] / max(x1 - x0, y1 - y0); img = cv2.resize(rgb[y0:y1, x0:x1], (int((x1 - x0) * s), int((y1 - y0) * s)), interpolation=cv2.INTER_CUBIC); crop = dict(x0=x0, y0=y0, x1=x1, y1=y1, scale=round(s, 4))
    else:
        img = rgb; s = 1.0; x0 = y0 = 0
    sam.predictor.set_image(img); log["crop"] = crop
    cc = [(m, np.array(dt_peak(m))) for m in cand_masks if m.any()]
    pl_d = cv2.dilate(plant.astype(np.uint8), np.ones((2 * GAP["inside_dilate_px"] + 1,) * 2, np.uint8)) > 0; kept, dropped = 0, 0
    for comp in comps:
        px_, py_ = dt_peak(comp); ys_, xs_ = np.nonzero(comp)
        negs = sorted(cc, key=lambda t: np.hypot(*(t[1] - [px_, py_])))[:GAP["n_neg"]]
        f = lambda x, y: [(x * g2f - x0) * s, (y * g2f - y0) * s]
        pts = np.array([f(px_, py_)] + [f(*t[1]) for t in negs], np.float32); lab = np.array([1] + [0] * len(negs), np.int32)
        w_, h_ = xs_.max() - xs_.min(), ys_.max() - ys_.min(); p = GAP["pad"]
        box = np.array(f(xs_.min() - p * w_, ys_.min() - p * h_) + f(xs_.max() + p * w_, ys_.max() + p * h_), np.float32)
        masks, scores, _ = sam.predictor.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True)
        best = None
        for mm, sc in zip(masks, scores):
            full = np.zeros((Hf, Hf), np.uint8); mm_ = cv2.resize(mm.astype(np.uint8), (int(round(mm.shape[1] / s)), int(round(mm.shape[0] / s))), interpolation=cv2.INTER_NEAREST)
            hh, ww = min(mm_.shape[0], Hf - y0), min(mm_.shape[1], Hf - x0); full[y0:y0 + hh, x0:x0 + ww] = mm_[:hh, :ww]
            g = cv2.resize(full, (H2, H2), interpolation=cv2.INTER_NEAREST) > 0
            if not g.any():
                continue
            inside = (g & pl_d).sum() / g.sum(); iou = (g & comp).sum() / max((g | comp).sum(), 1)
            if inside >= GAP["inside_min"] and iou >= GAP["iou_min"] and (best is None or sc > best[1]):
                best = (g, float(sc), round(float(iou), 3))
        if best is None:
            dropped += 1; continue
        kept += 1; out.append(dict(mask=best[0], score=best[1], iou_comp=best[2], prompt=dict(pos=[int(px_), int(py_)], neg=[t[1].tolist() for t in negs])))
    log.update(components=len(comps), created=kept, dropped=dropped); return out


# ---------------------------------------------------------------- step 6: replay on R3-1b
def step_replay(sam_on=True):
    pl, H2id = r31b_plant(); views = pl["views"]; vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); train = vc["chosen"]; held = vc["held_out"]
    recs = json.load(open(UL.OUT / "labels" / "leaf_labels.json", encoding="utf-8")); nmap = json.load(open(UL.OUT / "labels" / "number_map.json", encoding="utf-8"))["latest"]["map"]
    nmap3 = json.load(open(UL.OUT / "labels" / "number_map_v3_charuco.json", encoding="utf-8"))["map"]
    det = {v: json.load(open(UL.OUT / "detect" / f"{v}.json", encoding="utf-8")) for v in train}; by_id = {v: {m["id"]: m for m in det[v]["masks"]} for v in train}
    truth = {v: truth_image(recs, v, by_id[v], nmap) for v in train}
    vdir = OUT / "visibility" / "R3-1b"; d = OUT / "replay_R3-1b"; (d / "examples").mkdir(parents=True, exist_ok=True); (OUT / "gapmasks" / "R3-1b").mkdir(parents=True, exist_ok=True); (OUT / "prefill" / "R3-1b").mkdir(parents=True, exist_ok=True)
    sam = None
    if sam_on:
        from tomato3d.leafseg import run as r2
        from tomato3d.topview.measure2d import load_sam
        sam = load_sam(r2.SAMCFG)
    labelled = [R3_SEED]; order = []; per_view = {}; mask_rows = []; gap_rows = []
    while len(labelled) < len(train):
        ang, v = next_view(pl, labelled, [u for u in train if u not in labelled]); order.append(dict(view=v, angle_to_labelled_deg=round(float(ang), 1), labelled_before=list(labelled)))
        maps = {u: truth[u][0] for u in labelled}; lab, cnt = UL.vote_views(pl, maps, labelled, None, H2id)
        tot = cnt.sum(1); share = np.where(tot > 0, cnt.max(1) / np.maximum(tot, 1), 0.0)
        z = np.load(vdir / f"{v}.npz"); pred, sh, fc, up = predicted_layers(lab, share, z["facing"], z["upper"], to_grid(z["index"]))
        view = AP.view_file(v); plant = np.asarray(Image.open(AP.FG_MASKS / (Path(view).stem + ".png")).resize((H2, H2), Image.NEAREST)) == 1
        cmask = [UL.unrle(m["rle"]) for m in det[v]["masks"]]; cu = np.zeros((H2, H2), bool)
        for m in cmask:
            cu |= m
        # prefill on detector masks
        act = active_records(recs, v); props = []
        for m, mm in zip(det[v]["masks"], cmask):
            a = assign(mm, pred, sh, fc, up); r = act.get(m["id"]); t = truth_number(r, v, nmap) if r else None; t3 = int(nmap3.get(f"{v}|{int(r['leaf'])}", 0)) if (r and r["leaf"] not in ("모름", "unknown", None)) else None
            props.append(dict(mask_id=m["id"], prop=m["prop"], kind="detector", **a, truth=t, truth_v3=t3, area_px=int(mm.sum())))
        # gap masks: leaflet area range from accepted views' human records (mm^2 -> px via ppm)
        areas = []
        for u in labelled:
            ppm_u, _ = UL.plant_px_per_mm(pl, AP.view_file(u), H2id[:, views.index(AP.view_file(u))])
            for r in active_records(recs, u).values():
                mk = UL.record_rle(r, recs, by_id[u])
                if mk is not None and r["leaf"] not in ("모름",):
                    areas.append(mk.sum() / ppm_u ** 2)
        ppm, _ = UL.plant_px_per_mm(pl, view, H2id[:, views.index(view)]); lo, hi = np.percentile(areas, GAP["area_pct"]) * ppm ** 2
        extra = (pred > 0) & (fc >= 1)
        comps, gap_px = gap_components(plant, cu, ppm, lo, hi, extra=extra); glog = dict(gap_px_before=gap_px)
        gms = sam_gap_masks(sam, np.asarray(Image.open(AP.PHOTOS / view).convert("RGB")), plant, comps, cmask, glog) if sam is not None else []
        tv, anyh = truth[v]; after = gap_px
        for i, g in enumerate(gms):
            a = assign(g["mask"], pred, sh, fc, up, cap="medium"); vals = tv[g["mask"]]; c = Counter(vals[vals > 0].tolist()); tg = c.most_common(1)[0][0] if c else 0
            props.append(dict(mask_id=f"G{i + 1:03d}", prop="auto_gap_mask", kind="gap", **a, truth=(tg or None), area_px=int(g["mask"].sum()), iou_comp=g["iou_comp"]))
            gap_rows.append(dict(view=v, id=f"G{i + 1:03d}", area_px=int(g["mask"].sum()), inside_human=round(float((g["mask"] & anyh).sum() / g["mask"].sum()), 3), iou_comp=g["iou_comp"]))
        gu = np.zeros((H2, H2), bool)
        for g in gms:
            gu |= g["mask"]
        glog["gap_px_after"] = int((plant & ~cu & ~gu).sum()); glog["gap_px_before_raw"] = int((plant & ~cu).sum())
        # recovered human leaflets that had no detector candidate
        sam_pieces = [UL.record_rle(r, recs, by_id[v]) for r in act.values() if r.get("source") == "user_point_sam"]
        rec_ = [any((p & g["mask"]).sum() / max((p | g["mask"]).sum(), 1) >= 0.5 for g in gms) for p in sam_pieces if p is not None]
        np.savez_compressed(OUT / "gapmasks" / "R3-1b" / f"{v}.npz", masks=np.array([g["mask"] for g in gms]) if gms else np.zeros((0, H2, H2), bool))
        json.dump(dict(view=v, labels=LABELS, prompts=[g["prompt"] for g in gms], scores=[g["score"] for g in gms], log=glog), open(OUT / "gapmasks" / "R3-1b" / f"{v}.json", "w", encoding="utf-8"), default=float)
        json.dump(dict(view=v, labels=LABELS, proposals=props), open(OUT / "prefill" / "R3-1b" / f"{v}.json", "w", encoding="utf-8"), default=float)
        # metrics for this view
        hp = (tv > 0); covered = np.zeros((H2, H2), bool)
        for p, mm in zip(props, cmask + [g["mask"] for g in gms]):
            if p["confidence"] in ("high", "medium"):
                covered |= mm
        def acc(conf, kind=None):
            rows = [p for p in props if p["confidence"] == conf and p.get("truth") and (kind is None or p["kind"] == kind)]
            return (sum(p["number"] == p["truth"] for p in rows), len(rows))
        per_view[v] = dict(high=acc("high"), medium=acc("medium"), high_detector=acc("high", "detector"), coverage=round(float((covered & hp).sum() / max(hp.sum(), 1)), 3),
                           proposals=dict(Counter(p["confidence"] for p in props)), gap=glog, gap_recovered=(sum(rec_), len(rec_)), lifted_surfels=int((lab > 0).sum()))
        mask_rows += [dict(view=v, **{k: p.get(k) for k in ("mask_id", "kind", "prop", "number", "candidate_number", "confidence", "truth", "truth_v3", "cover", "vote_share", "front", "upper", "other_leaf_within_3px", "area_px")}) for p in props]
        example(v, view, props, cmask + [g["mask"] for g in gms], d / "examples" / f"{v}.jpg")
        print(v, json.dumps(per_view[v], default=float), flush=True)
        labelled.append(v)
    # totals and verdict
    S = lambda key: (sum(per_view[v][key][0] for v in per_view), sum(per_view[v][key][1] for v in per_view))
    hc, hn = S("high"); mc, mn = S("medium"); cov_num = cov_den = 0
    for v in per_view:
        tv = truth[v][0]; hp = tv > 0; cov_den += int(hp.sum()); cov_num += per_view[v]["coverage"] * int(hp.sum())
    gin = [g["inside_human"] * g["area_px"] for g in gap_rows]; gpx = sum(g["area_px"] for g in gap_rows)
    tot = dict(high_accuracy=(round(hc / hn, 3) if hn else None), high_n=hn, medium_accuracy=(round(mc / mn, 3) if mn else None), medium_n=mn, coverage=round(cov_num / max(cov_den, 1), 3),
               gap_masks=len(gap_rows), gap_inside_human=(round(sum(gin) / gpx, 3) if gpx else None), gap_recovered=(sum(per_view[v]["gap_recovered"][0] for v in per_view), sum(per_view[v]["gap_recovered"][1] for v in per_view)))
    verdict = dict(prefill_pass=bool(tot["high_accuracy"] is not None and tot["high_accuracy"] >= PASS["high_acc"] and tot["coverage"] >= PASS["coverage"]),
                   gap_pass=bool(tot["gap_inside_human"] is not None and tot["gap_inside_human"] >= PASS["gap_inside"]))
    json.dump(dict(labels=LABELS, rule=PASS, conf_rule=CONF, gap_rule=GAP, seed=R3_SEED, order=order, per_view=per_view, totals=tot, verdict=verdict, held_out_untouched=held), open(d / "metrics.json", "w", encoding="utf-8"), indent=1, default=float)
    with open(d / "masks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mask_rows[0])); w.writeheader(); w.writerows(mask_rows)
    with open(d / "gapmasks.csv", "w", newline="", encoding="utf-8") as f:
        if gap_rows:
            w = csv.DictWriter(f, fieldnames=list(gap_rows[0])); w.writeheader(); w.writerows(gap_rows)
    print("TOTAL", json.dumps(tot), json.dumps(verdict)); return tot, verdict


PALC = {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25), 4: (0, 130, 200), 5: (245, 130, 48), 6: (145, 30, 180), 7: (70, 240, 240), 8: (240, 50, 230), 9: (210, 245, 60), 10: (250, 190, 190), 11: (0, 128, 128), 12: (220, 190, 255)}


def example(vid, view, props, masks, path):
    """Failure examples: wrong high/medium in red outline, right ones in their colour, gap masks dashed white."""
    img = np.asarray(Image.open(AP.PHOTOS / view).convert("RGB").resize((H2, H2))).copy()
    for p, m in zip(props, masks):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if p["confidence"] == "low":
            col, t = (160, 160, 160), 1
        elif p.get("truth") and p["number"] != p["truth"]:
            col, t = (255, 0, 0), 4
        else:
            col, t = PALC.get(p["number"], (255, 255, 255)), 2
        cv2.drawContours(img, cs, -1, col, t)
        if p["kind"] == "gap":
            cv2.drawContours(img, cs, -1, (255, 255, 255), 1)
        if p["confidence"] != "low" and cs:
            x, y, _, _ = cv2.boundingRect(max(cs, key=cv2.contourArea)); cv2.putText(img, f"{p['number']}{'H' if p['confidence'] == 'high' else 'M'}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    Image.fromarray(img).resize((1024, 1024)).save(path, quality=85)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--ids", nargs="*", default=["R3-1b"] + AC.LABEL_IDS + ["t2_04", "n_03"]); ap.add_argument("--no-sam", action="store_true")
    a = ap.parse_args(argv); OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "poses":
        step_poses(a.ids)
    if a.step == "visibility":
        step_visibility()
    if a.step == "replay":
        step_replay(not a.no_sam)


if __name__ == "__main__":
    main()
