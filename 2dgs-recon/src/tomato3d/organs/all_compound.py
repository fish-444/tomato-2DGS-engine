"""P1 - frozen R3-1b compound-leaf pipeline on the other individuals (contract build/p1_all_compound_20261026_01/CONTRACT.md).

Step 0: readiness table (this module, `--step readiness`). Later steps are added after the readiness result decides what can run.
Every output: R3-1b_unaccepted, R3-1b-style, conditional_on_unvalidated_scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tomato3d.fuse3d import surfels_io as sio
from tomato3d.organs import light_organs as LO
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
ALL = ROOT / "build/p1_recon_all_20261025_01"
OUT = ROOT / "build/p1_all_compound_20261026_01"
IDS = ["n_04", "t2_02", "n_01", "t2_05", "t2_04", "n_06", "n_05", "n_02", "n_03"]
R31B_BAND_EXCL = 0.901
CUT = dict(auto_min=R31B_BAND_EXCL - 0.05, a_points=300, b_points=100)
LABELS = dict(status="R3-1b_unaccepted", pipeline="R3-1b-style", scale="conditional_on_unvalidated_scale")
QUEUE_STATE = {"n_06": "COMPLETE", "n_05": "TIMEOUT (kernel past 45 min; output present)", "n_02": "COMPLETE", "t2_05": "ERROR on first push (cause unlogged); re-run COMPLETE",
               "n_03": "COMPLETE", "t2_04": "COMPLETE", "t2_02": "COMPLETE (first try hung 5 h, cancelled)", "n_04": "COMPLETE", "n_01": "COMPLETE"}
GRAPH_K, GRAPH_MM = 10, 3.0


def connectivity(xyz: np.ndarray) -> dict:
    """Largest connected component share on the k = 10 within 3 mm graph (the cleanup/re-cluster graph)."""
    d, i = cKDTree(xyz).query(xyz, k=GRAPH_K + 1)
    r = np.repeat(np.arange(len(xyz)), GRAPH_K); c = i[:, 1:].ravel(); m = d[:, 1:].ravel() <= GRAPH_MM / 1000
    A = coo_matrix((np.ones(m.sum()), (r[m], c[m])), shape=(len(xyz),) * 2)
    n, lab = connected_components(A, directed=False); cnt = np.bincount(lab)
    return dict(components=int(n), largest_share=float(cnt.max() / len(xyz)), largest=int(cnt.max()), n=int(len(xyz)), over_100=int((cnt >= 100).sum()))


def last_loss(log: Path):
    its, loss = None, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r" (\d+) loss ([0-9.]+) l1 ([0-9.]+)", line)
        if m and int(m.group(1)) > 1:
            its, loss = int(m.group(1)), float(m.group(2))
    return its, loss


def readiness():
    triage = {r[0]: r for r in json.load(open(ALL / "triage_all.json", encoding="utf-8"))}
    pts = json.load(open(ALL / "pts_all/points_report.json", encoding="utf-8")); rows = []
    for i in IDS:
        meta = json.load(open(ALL / "outputs" / i / "recon_meta.json", encoding="utf-8")); res = meta["results"]; t = triage[i]
        sf = sio.read_surfels_full(ALL / "outputs" / i / f"R3-1b-style_{i}_surfels_full.ply")
        xyz = sf["xyz"].astype(np.float64); keep = sf["opacity"] >= LO.OPACITY_MIN if "opacity" in sf else np.ones(len(xyz), bool)
        conn = connectivity(xyz[keep])
        its, loss = last_loss(ALL / i / "kernel_out" / "run_log.txt")
        ho = res["heldout_new"]; per = ho["per_view"]; med_nb = float(np.median([v["iou_no_band"] for v in per.values()]))
        pv = pts[i]["per_view"]
        mask = "C2 (SAM 2 from user masks) where seeded, else C3" if i == "t2_02" else "C3 U-Net"
        rows.append(dict(individual=i, recon_status=QUEUE_STATE[i], iterations=its, last_loss=loss, init_points=res["init_points"], plant_surfels=int(len(xyz)), plant_surfels_opacity=int(keep.sum()),
                         views_accepted=meta["views_accepted"], train_views=meta["train_views"], azimuth_coverage_deg=t[4], azimuth_gap_deg=t[5], low_ring_views=t[6], triage_class=t[8],
                         mask_source=mask, heldout=" ".join(meta["heldout"]), auto_iou_no_band_heldout=round(ho["iou_no_band"], 3), auto_iou_no_band_median_views=round(med_nb, 3),
                         auto_edge_f2=round(ho["edge_f2"], 3), train_iou_no_band=round(res["train"]["iou_no_band"], 3), i1_points_per_view_median=pv["median"], i1_views_under_10=pv["under_10"],
                         conn_largest_share=round(conn["largest_share"], 3), conn_components=conn["components"], conn_components_ge100=conn["over_100"], human_reference=meta["human_reference"]))
    # R3-1b with the same rule (all surfels) for comparison
    sf = sio.read_surfels_full(LO.DEFAULTS["surfels"]); ref = connectivity(sf["xyz"].astype(np.float64)[sf["opacity"] >= LO.OPACITY_MIN])
    ta = np.load(LO.DEFAULTS["out"] / "surfel_labels.npz"); blade = connectivity(sf["xyz"][ta["rows"]][~ta["is_axis"].astype(bool)].astype(np.float64))
    its31, loss31 = None, None
    for r in rows:
        r["class"] = classify(r)
    return rows, dict(r31b_all=ref, r31b_blade=blade)


def classify(r) -> str:
    if r["i1_points_per_view_median"] < CUT["b_points"]:
        return "C"
    a1 = r["triage_class"] == "reconstructable" or r["i1_points_per_view_median"] >= CUT["a_points"]
    return "A" if a1 and r["auto_iou_no_band_median_views"] >= CUT["auto_min"] else "B"


# ---------------------------------------------------------------- Amendment 1: views for labelling
LABEL_IDS = ["n_04", "t2_02", "n_01", "t2_05", "n_06", "n_05", "n_02"]
POSES = {i: ALL / "stage2_new" / i / "poses.json" for i in ("n_01", "n_02", "n_03", "n_04", "n_05")}
POSES["n_06"] = ALL / "stage2_new_n06" / "n_06" / "poses.json"
POSES.update({i: ROOT / "build/p1_recon_tomato2_20261023_01/stage2_d7" / i / "poses.json" for i in ("t2_01", "t2_02", "t2_03", "t2_04", "t2_05")})
E_IDS = ["e_01", "e_02", "e_03", "e_04", "e_05", "e_06", "e_07"]                  # 2026-10-28 batch (build/p1_recon_d1028_20261028_01), same layout as the nine
POSES.update({i: ALL / "stage2_new" / i / "poses.json" for i in E_IDS})
E_PHOTO_ROOT = Path("<DOWNLOADS>")                                  # HEIC of the e_* batch; IMG numbers collide with the older individuals, so they are indexed per batch
VIEWS_RULE = dict(n_train=4, azimuth_min_deg=60.0, heldout_azimuth_min_deg=45.0, top_horiz_mm=50.0, vis_res=1512, vis_tol_m=0.0015, radius_sigma=1.5, max_r_px=14)
RINGS = ((75.0, "top"), (58.0, "upper"), (45.0, "middle"), (-90.0, "low"))


def ring_of(el):
    return next(n for lo, n in RINGS if el > lo)


def load_individual(iid):
    """Surfels (opacity >= 0.5) and the R3-1b-format poses dict of one individual."""
    sf = sio.read_surfels_full(ALL / "outputs" / iid / f"R3-1b-style_{iid}_surfels_full.ply"); keep = np.nonzero(sf["opacity"] >= LO.OPACITY_MIN)[0]
    pz = json.load(open(POSES[iid], encoding="utf-8")); acc = pz["accepted"]; v0 = pz["views"][acc[0]]; Wd = v0["size"][0]
    assert all(pz["views"][v]["K"] == v0["K"] for v in acc), f"{iid}: more than one camera"
    po = dict(full_size=[Wd, Wd], K_full=v0["K"], dist=v0["dist"], undistorted_res=Wd, K_undist=v0["K"], accepted=acc,
              views={v: dict(R_cw=pz["views"][v]["R_cw"], t_cw=pz["views"][v]["t_cw"]) for v in acc})
    sp = json.load(open(ALL / "k3a" / iid / "split.json", encoding="utf-8"))
    return dict(iid=iid, rows=keep, xyz=sf["xyz"][keep].astype(np.float64), sc=sf["scale"][keep].astype(np.float64), po=po, split=sp)


def camera_geometry(po, view, centre_xy, z_ref):
    R = np.array(po["views"][view]["R_cw"]); t = np.array(po["views"][view]["t_cw"]); c = -R.T @ t
    d = c[:2] - centre_xy; horiz = float(np.hypot(*d)); az = float(np.degrees(np.arctan2(d[1], d[0])) % 360)
    el = float(np.degrees(np.arctan2(c[2] - z_ref, max(horiz, 1e-9)))); return az, horiz * 1000, el


def az_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def visibility(pl, views):
    from tomato3d.organs import seed2d as S2
    V = S2.votes_per_view(pl["xyz"], pl["po"], views, VIEWS_RULE["vis_res"], VIEWS_RULE["vis_tol_m"], sc=pl["sc"], mode="splat", label_maps={v: np.zeros((S2.H2_GRID, S2.H2_GRID), np.int32) for v in views})
    return V >= 0


def choose_label_views(iid):
    pl = load_individual(iid); tr, ho = pl["split"]["train"], pl["split"]["heldout"]; allv = tr + ho
    vis = visibility(pl, allv); n = len(pl["xyz"]); xy = pl["xyz"][:, :2].mean(0); zref = float(np.median(pl["xyz"][:, 2]))
    geo = {v: camera_geometry(pl["po"], v, xy, zref) for v in allv}
    top = {v: geo[v][1] < VIEWS_RULE["top_horiz_mm"] for v in allv}; rg = {v: ring_of(geo[v][2]) for v in allv}
    def sep(v, chosen, deg):
        return all(top[v] or top[c] or az_diff(geo[v][0], geo[c][0]) >= deg for c in chosen)
    covered = np.zeros(n, bool); chosen, steps = [], []
    low = [v for v in tr if rg[v] == "low"]
    for k in range(VIEWS_RULE["n_train"]):
        pool = [v for v in tr if v not in chosen and sep(v, chosen, VIEWS_RULE["azimuth_min_deg"])]
        if k == 0 and low:
            pool = [v for v in pool if rg[v] == "low"]
        if not pool:
            steps.append(dict(step=k + 1, note="no view satisfies the azimuth rule")); break
        gain, v = max((int((vis[:, allv.index(v)] & ~covered).sum()), v) for v in pool)
        chosen.append(v); covered |= vis[:, allv.index(v)]
        steps.append(dict(step=k + 1, view=v, gain=gain, gain_share=round(gain / n, 4), ring=rg[v], azimuth=round(geo[v][0], 1), elevation=round(geo[v][2], 1), horiz_mm=round(geo[v][1], 1), forced_low=bool(k == 0 and low)))
    hc = [(int(vis[:, allv.index(v)].sum()), v) for v in ho if sep(v, chosen, VIEWS_RULE["heldout_azimuth_min_deg"])]
    held = max(hc)[1] if hc else max((int(vis[:, allv.index(v)].sum()), v) for v in ho)[1]
    cnt = vis[:, [allv.index(v) for v in chosen]].sum(1)
    rep = dict(individual=iid, labels=LABELS, rule=VIEWS_RULE, train=chosen, held_out=held, held_out_rule_met=bool(hc), steps=steps, low_ring_available=bool(low), plant_surfels=int(n),
               seen_ge1_share=round(float((cnt >= 1).mean()), 4), seen_ge2_share=round(float((cnt >= 2).mean()), 4),
               heldout_visible_share=round(float(vis[:, allv.index(held)].mean()), 4), heldout_new_share=round(float((vis[:, allv.index(held)] & ~covered).sum() / n), 4),
               geometry={v: dict(azimuth=round(geo[v][0], 1), horiz_mm=round(geo[v][1], 1), elevation=round(geo[v][2], 1), ring=rg[v]) for v in chosen + [held]}, centre_xy=[round(float(x), 4) for x in xy])
    d = OUT / iid; d.mkdir(parents=True, exist_ok=True); json.dump(rep, open(d / "views_chosen.json", "w", encoding="utf-8"), indent=1); return rep


PHOTO_ROOTS = [ROOT / "새 폴더", ROOT / "토마토2"]


def heic_index():
    idx = {}
    for r in PHOTO_ROOTS:
        for f in r.rglob("*.HEIC"):
            idx.setdefault(f.name, []).append(f)
    return idx


def view_list(iid):
    r = json.load(open(OUT / iid / "views_chosen.json", encoding="utf-8")); return r["train"] + [r["held_out"]], r


def step_photos(ids):
    """Undistorted 3024 frames of the label views (K3c recipe), R3-1b-format poses.json, plant-mask copies and px/mm per view."""
    import cv2
    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener(); idx = heic_index()
    for iid in ids:
        views, rep = view_list(iid); pl = load_individual(iid); d = OUT / iid; (d / "undist_3024").mkdir(parents=True, exist_ok=True); (d / "masks").mkdir(exist_ok=True)
        po = pl["po"]; Kn, dist = np.array(po["K_undist"]), np.array(po["dist"]); json.dump(po, open(d / "poses.json", "w", encoding="utf-8"))
        vis = visibility(pl, views); from tomato3d.organs import seed2d as S2
        K = ca.scaled_K(Kn, po["undistorted_res"], S2.H2_GRID); ppm = {}
        for k, v in enumerate(views):
            cands = idx.get(v[:-4] + ".HEIC", []); assert len(cands) >= 1, (iid, v)
            f = cands[0]; im = np.asarray(Image.open(f).convert("RGB")); assert im.shape[:2] == (po["undistorted_res"],) * 2, (iid, v, im.shape)
            Image.fromarray(cv2.undistort(im, Kn, dist, None, Kn)).save(d / "undist_3024" / v, quality=95)
            Image.open(ALL / "k3a" / iid / "masks" / (Path(v).stem + ".png")).save(d / "masks" / (Path(v).stem + ".png"))
            R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"]); z = (pl["xyz"] @ R.T + t)[:, 2]; zz = z[vis[:, k]]
            ppm[v] = dict(px_per_mm=float(K[0, 0] * 0.001 / zz.mean()), plant_depth_m=float(zz.mean()), visible=int(vis[:, k].sum()), src=str(f.name))
        json.dump(dict(views=views, train=rep["train"], held_out=rep["held_out"], ppm=ppm, labels=LABELS), open(d / "detect_input.json", "w", encoding="utf-8"), indent=1); print(iid, "photos", len(views), flush=True)


def step_roboflow(ids):
    """Roboflow v4 boxes by API for the label views only (key in memory, never written); responses cached for the Kaggle SAM drawing."""
    from PIL import Image
    from tomato3d.leafseg import rfdet
    from tomato3d.organs import auto_proposers as AP
    key = AP._rf_key(); assert key, "no Roboflow key"
    for iid in ids:
        views, _ = view_list(iid); d = OUT / iid; log = {}
        for v in views:
            rgb = np.asarray(Image.open(d / "undist_3024" / v).convert("RGB")); stem = Path(v).stem
            try:
                pred = rfdet.fetch(d / "undist_3024" / v, rgb, d / "rf_detections" / f"{stem}.json", key); n = len(pred.get("predictions", [])); tiled = None
                if n >= AP.RF_CAP:
                    _, tiled = rfdet.tiled_dets(d / "undist_3024" / v, rgb, d / "rf_detections_tiled", stem, key)
                log[v] = dict(predictions=n, tiled=tiled)
            except Exception as e:
                log[v] = dict(error=AP._scrub(f"{type(e).__name__}: {e}", key))
            print(iid, v, log[v], flush=True)
        json.dump(log, open(d / "roboflow_log.json", "w", encoding="utf-8"), indent=1, default=float)


PLANT_INSIDE_MIN = 0.80          # Amendment 1 item 3: candidates with < 80 % of their area inside the plant mask are dropped, the rest clipped


def step_union(ids, cand_root: Path):
    """UNION of the raw candidates of one view: plant-mask restriction -> dedup (userlabel.union_masks) -> detect/<vid>.json for the labelling page."""
    from PIL import Image
    from tomato3d.organs import userlabel as UL
    for iid in ids:
        views, rep = view_list(iid); d = OUT / iid; (d / "detect").mkdir(exist_ok=True); summ = {}
        for v in views:
            stem = Path(v).stem; doc = json.load(open(cand_root / iid / f"{stem}_cands.json", encoding="utf-8"))
            plant = np.asarray(Image.open(d / "masks" / f"{stem}.png").resize((UL.H2, UL.H2), Image.NEAREST)) == 1; allm = []; drop = {"outside_plant": 0, "empty": 0}
            for c in doc["candidates"]:
                m = UL.unrle(c["rle"]).astype(bool)
                if m.sum() == 0:
                    drop["empty"] += 1; continue
                if (m & plant).sum() / m.sum() < PLANT_INSIDE_MIN:
                    drop["outside_plant"] += 1; continue
                allm.append(dict(uid=c["uid"], prop=c["prop"], score=c["score"], mask=m & plant, shape=c.get("shape")))
            final, ust = UL.union_masks(allm)
            json.dump(dict(view=v, individual=iid, labels=LABELS, union=UL.UNION, plant_inside_min=PLANT_INSIDE_MIN,
                           masks=[dict(id=f"U{i + 1:03d}", uid=m["uid"], prop=m["prop"], score=round(m["score"], 4), rle=UL.rle(m["mask"]), area_px=int(m["mask"].sum()), shape=m.get("shape")) for i, m in enumerate(final)],
                           merge_suspect=[dict(uid=m["uid"], prop=m["prop"], parts=m["parts"], rle=UL.rle(m["mask"])) for m in allm if m.get("merge_suspect")]),
                      open(d / "detect" / f"{stem}.json", "w", encoding="utf-8"))
            from collections import Counter
            summ[v] = dict(raw=doc["counts"], dropped_plant_restriction=drop, union=ust, by_prop_in_union=dict(Counter(m["prop"] for m in final)), held_out=(v == rep["held_out"]))
            print(iid, v, json.dumps(summ[v]), flush=True)
        json.dump(summ, open(d / "detect_summary.json", "w", encoding="utf-8"), indent=1, default=float)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", default="readiness"); ap.add_argument("--ids", nargs="*", default=LABEL_IDS); ap.add_argument("--cands", default=str(OUT / "kernel_out" / "out")); a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "union":
        step_union(a.ids, Path(a.cands))
    if a.step == "photos":
        step_photos(a.ids)
    if a.step == "roboflow":
        step_roboflow(a.ids)
    if a.step == "views":
        for i in a.ids:
            r = choose_label_views(i); print(i, r["train"], r["held_out"], [s.get("ring") for s in r["steps"]], r["seen_ge1_share"], r["seen_ge2_share"], r["heldout_visible_share"], r["held_out_rule_met"], flush=True)
    if a.step == "readiness":
        rows, ref = readiness()
        with open(OUT / "readiness.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        json.dump(dict(rows=rows, r31b_connectivity=ref, cutoffs=CUT, labels=LABELS), open(OUT / "readiness.json", "w", encoding="utf-8"), indent=1, default=float)
        for r in rows:
            print(r["individual"], r["class"], r["triage_class"], "pts/view", r["i1_points_per_view_median"], "auto", r["auto_iou_no_band_median_views"], "conn", r["conn_largest_share"], "comp>=100", r["conn_components_ge100"], "loss", r["last_loss"], r["iterations"])
        print("R3-1b", json.dumps(ref))


if __name__ == "__main__":
    main()
