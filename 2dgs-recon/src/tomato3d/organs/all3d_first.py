"""P1 all other individuals: photo preprocessing -> 3D-first leaflet segmentation -> flag only what fails.
Contract: build/p1_all_3dfirst_20261027_01/CONTRACT.md. One cell per individual, every worker writes only inside its own cell.

Steps (each idempotent, marked in <cell>/state.json; a finished step is never redone or overwritten):
  size_prior   (global, once)  S2 size range from the R3-1b human leaflets
  inputs       manifest, pose QC (S6)
  prep         every posed photo: undistorted 3024 jpg, plant mask copy, px/mm, crop box (Step 1.3), packed visibility
  geometry     blade proxy (S1), local sheets, t_sheet, sheet graph patches, 4a over-segmentation (Steps 2-3)

usage: python -m tomato3d.organs.all3d_first --id n_04 --step geometry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import all_compound as AC
from tomato3d.organs import light_organs as LO
from tomato3d.organs import sheet_retry as SR

ROOT = LO.ROOT
OUT = ROOT / "build/p1_all_3dfirst_20261027_01"
LABELS = dict(status="R3-1b-style_unaccepted", gate="gate_failed_viewing_only", scale=LO.SCALE_TAG)
ORDER = ["n_04", "t2_02", "n_01", "t2_05", "n_06", "n_05", "t2_04", "n_02", "n_03", "e_05", "e_01", "e_04", "e_07", "e_06", "e_03", "e_02"]       # e_* = 2026-10-28 batch (user 2026-10-31: all 16 individuals in the same grouping page)
RIM_NOTE = {"n_02": "heights_unusable_rim_fit"}
POSE_QC_RMS_PX_3024 = 3.0                      # S6: D7 limit 1.5 px at 1512
PROXY = dict(radius_mm=4.0, min_neighbours=8, l2_over_l1=0.25)       # S1
CROP = dict(frame_share=0.10, margin=0.10, long_px=1500)             # Step 1.3
SAMPLE = dict(n=3000, seed=20261027)


# ---------------------------------------------------------------- cell state
def cell(iid):
    d = OUT / iid; (d / "logs").mkdir(parents=True, exist_ok=True); (d / "work").mkdir(exist_ok=True); return d


def _sf(step):
    return "state_rf.json" if step == "rf" else "state.json"          # the Roboflow lane has its own state file (its own writer)


def state(iid, step="inputs"):
    f = cell(iid) / _sf(step); return json.load(open(f, encoding="utf-8")) if f.exists() else {}


def mark(iid, step, info=None):
    s = state(iid, step); s[step] = dict(done=time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {}))
    tmp = cell(iid) / (_sf(step) + ".tmp"); json.dump(s, open(tmp, "w", encoding="utf-8"), indent=1, default=float); tmp.replace(cell(iid) / _sf(step))


def log(iid, msg):
    line = time.strftime("%H:%M:%S ") + msg
    with open(cell(iid) / "logs" / "worker.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- global: size prior (S2)
def step_size_prior():
    f = OUT / "size_prior.json"
    if f.exists():
        return json.load(open(f, encoding="utf-8"))
    from tomato3d.organs import auto_proposers as AP
    from tomato3d.organs import extra_candidates as EC
    from tomato3d.organs import userlabel as UL
    pl = UL.load_plant(); H2id = np.load(UL.VOTE2D / "stage_a_h2id.npz")["H2id"]; views = pl["views"]
    vc = json.load(open(UL.OUT / "views_chosen.json", encoding="utf-8")); train = vc["chosen"]
    recs = json.load(open(UL.OUT / "labels" / "leaf_labels.json", encoding="utf-8"))
    det = {v: json.load(open(UL.OUT / "detect" / f"{v}.json", encoding="utf-8")) for v in train}; by_id = {v: {m["id"]: m for m in det[v]["masks"]} for v in train}
    areas, widths = [], []
    for v in train:
        ppm = UL.plant_px_per_mm(pl, AP.view_file(v), H2id[:, views.index(AP.view_file(v))])[0]
        for m in EC.human_leaflets(recs, v, by_id[v]):
            areas.append(m.sum() / ppm ** 2); w = EC.width_mm(m, ppm)
            if w:
                widths.append(w)
    a = np.array(areas); res = dict(source="R3-1b user leaflet masks on the 4 training views (size range only, no labels)", n=int(len(a)),
                                    area_mm2={k: float(np.percentile(a, p)) for k, p in (("p5", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95))}, width_median_mm=float(np.median(widths)), labels=LABELS)
    json.dump(res, open(f, "w", encoding="utf-8"), indent=1); return res


# ---------------------------------------------------------------- inputs (S6)
def step_inputs(iid):
    pz = json.load(open(AC.POSES[iid], encoding="utf-8")); acc = pz["accepted"]; drop = []; keep = []
    for v in acc:
        r = pz["views"][v].get("rms_px")
        if r is not None and float(r) > POSE_QC_RMS_PX_3024:
            drop.append(dict(view=v, reason=f"rms {float(r):.2f} px > {POSE_QC_RMS_PX_3024}")); continue
        keep.append(v)
    ply = AC.ALL / "outputs" / iid / f"R3-1b-style_{iid}_surfels_full.ply"; sp = json.load(open(AC.ALL / "k3a" / iid / "split.json", encoding="utf-8"))
    vc = json.load(open(AC.OUT / iid / "views_chosen.json", encoding="utf-8")) if (AC.OUT / iid / "views_chosen.json").exists() else {}
    held = vc.get("held_out"); rim = ROOT / "build/p1_plant_model_20261026_01/rim" / f"{iid}.json"
    man = dict(individual=iid, labels=LABELS, tag=RIM_NOTE.get(iid), surfels=dict(path=str(ply), sha256_16=sha(ply), mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(ply)))),
               poses=dict(path=str(AC.POSES[iid]), mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(AC.POSES[iid])))), accepted=len(acc), pose_qc_kept=len(keep), pose_dropped_views=drop,
               recon_heldout=sp["heldout"], label_heldout_excluded_from_votes=held, rim=(str(rim) if rim.exists() else None), code=dict(sheet_retry=sha(ROOT / "tomato3d/organs/sheet_retry.py"),
               oversegment=sha(ROOT / "tomato3d/organs/oversegment.py"), second=sha(ROOT / "tomato3d/organs/oversegment_second.py"), this=sha(Path(__file__))), views=keep)
    json.dump(man, open(cell(iid) / "inputs.json", "w", encoding="utf-8"), indent=1); mark(iid, "inputs", dict(kept=len(keep), dropped=len(drop))); return man


# ---------------------------------------------------------------- prep (Step 1.1-1.3)
def step_prep(iid):
    import cv2
    import pillow_heif
    from PIL import Image
    from tomato3d.organs import seed2d as S2
    from tomato3d.recon import charuco_a3 as ca
    pillow_heif.register_heif_opener(); man = json.load(open(cell(iid) / "inputs.json", encoding="utf-8")); views = man["views"]; pl = AC.load_individual(iid); po = pl["po"]
    d = cell(iid); (d / "photos").mkdir(exist_ok=True); (d / "masks").mkdir(exist_ok=True); Kn, dist = np.array(po["K_undist"]), np.array(po["dist"]); idx = AC.heic_index(); json.dump(po, open(d / "poses.json", "w", encoding="utf-8"))
    xy = pl["xyz"]; K = ca.scaled_K(Kn, po["undistorted_res"], S2.H2_GRID); info = {}; V = np.zeros((len(xy), len(views)), bool)
    for k0 in range(0, len(views), 8):                                         # visibility in blocks of 8 photos (bounded memory)
        blk = views[k0:k0 + 8]; V[:, k0:k0 + len(blk)] = AC.visibility(pl, blk)
    for k, v in enumerate(views):
        out = d / "photos" / v
        if not out.exists():
            f = idx.get(v[:-4] + ".HEIC", [None])[0]
            if f is None:
                info[v] = dict(error="HEIC not found"); log(iid, f"{v}: HEIC not found"); continue
            im = np.asarray(Image.open(f).convert("RGB"))
            if im.shape[:2] != (po["undistorted_res"],) * 2:
                info[v] = dict(error=f"size {im.shape}"); continue
            Image.fromarray(cv2.undistort(im, Kn, dist, None, Kn)).save(out, quality=95)
        mpath = AC.ALL / "k3a" / iid / "masks" / (Path(v).stem + ".png")
        if not (d / "masks" / mpath.name).exists():
            Image.open(mpath).save(d / "masks" / mpath.name)
        plant = np.asarray(Image.open(mpath).resize((S2.H2_GRID, S2.H2_GRID), Image.NEAREST)) == 1
        R = np.array(po["views"][v]["R_cw"]); t = np.array(po["views"][v]["t_cw"]); z = (pl["xyz"] @ R.T + t)[:, 2]; zz = z[V[:, k]]
        ys, xs = np.nonzero(plant); share = float(plant.mean()); g = po["undistorted_res"] / S2.H2_GRID
        bbox = [float(xs.min() * g), float(ys.min() * g), float((xs.max() + 1) * g), float((ys.max() + 1) * g)] if len(xs) else None; crop = None
        if bbox and share < CROP["frame_share"]:
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]; mx, my = CROP["margin"] * w, CROP["margin"] * h; x0, y0 = max(0, int(bbox[0] - mx)), max(0, int(bbox[1] - my))
            x1, y1 = min(po["undistorted_res"], int(bbox[2] + mx)), min(po["undistorted_res"], int(bbox[3] + my)); crop = dict(x0=x0, y0=y0, x1=x1, y1=y1, scale=round(max(1.0, CROP["long_px"] / max(x1 - x0, y1 - y0)), 4))
        info[v] = dict(px_per_mm=(float(K[0, 0] * 0.001 / zz.mean()) if len(zz) else None), plant_depth_m=(float(zz.mean()) if len(zz) else None), visible_surfels=int(V[:, k].sum()), plant_frame_share=round(share, 4), crop=crop)
    np.savez_compressed(d / "work" / "visibility_plant.npz", rows=pl["rows"], views=np.array(views), vis=np.packbits(V, axis=0), n=len(xy))
    json.dump(dict(labels=LABELS, views=views, photos=info, crops_needed=sum(1 for x in info.values() if x.get("crop"))), open(d / "detect_input.json", "w", encoding="utf-8"), indent=1)
    mark(iid, "prep", dict(photos=len(views), crops=sum(1 for x in info.values() if x.get("crop")), errors=sum(1 for x in info.values() if "error" in x)))


# ---------------------------------------------------------------- geometry (S1, Steps 2-3)
def blade_proxy(xyz):
    """S1: blade if >= 8 neighbours within 4 mm and l2 / l1 >= 0.25 (PCA of the neighbourhood), else axis_like. Also returns l3/l1."""
    tree = cKDTree(xyz); nb = tree.query_ball_point(xyz, PROXY["radius_mm"] / 1000); n = len(xyz); blade = np.zeros(n, bool); r21 = np.zeros(n, np.float32); r31 = np.zeros(n, np.float32); few = np.zeros(n, bool)
    for i in range(n):
        j = nb[i]
        if len(j) < PROXY["min_neighbours"]:
            few[i] = True; continue
        w = np.linalg.eigvalsh(np.cov((xyz[j] - xyz[i]).T)); w = np.maximum(w[::-1], 1e-18); r21[i] = w[1] / w[0]; r31[i] = w[2] / w[0]; blade[i] = r21[i] >= PROXY["l2_over_l1"]
    return blade, r21, r31, few


def sampled_sheet_stats(xyz, S, seed=SAMPLE["seed"], n=SAMPLE["n"]):
    """t_sheet inputs: thin_shell.fwhm of the normal-offset histogram of sampled single-sheet columns (the median is taken by the caller)."""
    from tomato3d.recon import thin_shell as TS
    rng = np.random.default_rng(seed); s = rng.choice(len(xyz), min(n, len(xyz)), replace=False); tree = cKDTree(xyz); rad = np.hypot(SR.P["lateral_mm"], SR.P["range_mm"]) / 1000; fw = []
    for i in s:
        if S["n_sheets"][i] != 1:
            continue
        nb = np.array(tree.query_ball_point(xyz[i], rad)); d = xyz[nb] - xyz[i]; v = S["normal"][i]; a = d @ v; lat = np.linalg.norm(d - np.outer(a, v), axis=1); m = (lat <= SR.P["lateral_mm"] / 1000) & (np.abs(a) <= SR.P["range_mm"] / 1000)
        off = a[m] * 1000; f = TS.fwhm(off)
        if np.isfinite(f):
            fw.append(f)
    return fw


def step_geometry(iid):
    from tomato3d.organs import oversegment as OS
    t0 = time.time(); pl = AC.load_individual(iid); sp = step_size_prior(); xyz_all = pl["xyz"]; log(iid, f"surfels {len(xyz_all)}")
    blade, r21, r31, few = blade_proxy(xyz_all); b = np.nonzero(blade)[0]; xyz = xyz_all[b]; log(iid, f"blade proxy {len(b)} of {len(xyz_all)} (axis_like {int((~blade & ~few).sum())}, isolated {int(few.sum())}) {time.time() - t0:.0f}s")
    S = SR.local_sheets(xyz); S.pop("columns", None); share_multi = float((S["n_sheets"] > 1).mean()); log(iid, f"sheets: multi-sheet share {share_multi:.3f} {time.time() - t0:.0f}s")
    fw = sampled_sheet_stats(xyz, S); t_sheet = float(np.median(fw)) if fw else SR.T_SHEET_MM; off_lim = max(t_sheet, 1.0)
    # density: surfels per mm^2 of one sheet = neighbours in the 4 mm plane and +-1.5 t_sheet along the normal / (pi 4^2)
    rng = np.random.default_rng(SAMPLE["seed"]); s = rng.choice(len(xyz), min(SAMPLE["n"], len(xyz)), replace=False); tree = cKDTree(xyz); c = []
    for i in s:
        nb = np.array(tree.query_ball_point(xyz[i], 0.004)); d = xyz[nb] - xyz[i]; a = np.abs(d @ S["normal"][i]) * 1000; c.append(int((a <= 1.5 * t_sheet).sum()))
    rho = float(np.median(c) / (np.pi * 4.0 ** 2)); A = sp["area_mm2"]
    small_n = max(5, int(round(A["p5"] * rho))); p75 = int(round(A["p75"] * rho)); med = int(round(A["median"] * rho)); cap = int(round(1.2 * A["p95"] * rho))
    E = SR.edges_3mm(xyz); patch, st = SR.patches(xyz, S, E, off_lim, small_n); log(iid, f"patches {st} {time.time() - t0:.0f}s")
    new, st4 = OS.split_patches(xyz, S, patch, E, p75, med); sizes = np.bincount(new[new >= 0]) if (new >= 0).any() else np.zeros(1, int)
    # planarity per patch and multi-sheet spanning
    plan, multi_patch = [], []
    for g in range(len(sizes)):
        idx = np.nonzero(new == g)[0]
        if len(idx) < 6:
            continue
        w = np.maximum(np.linalg.eigvalsh(np.cov((xyz[idx] - xyz[idx].mean(0)).T))[::-1], 1e-18); plan.append((w[2] / w[0], w[1] / w[0]))
        ms = S["n_sheets"][idx] > 1; multi_patch.append(bool(ms.any() and len(np.unique(S["index"][idx][ms])) > 1))
    plan = np.array(plan) if plan else np.zeros((0, 2))
    summ = dict(labels=LABELS, tag=RIM_NOTE.get(iid), surfels=int(len(xyz_all)), blade_proxy=int(len(b)), axis_like=int((~blade & ~few).sum()), isolated=int(few.sum()), multi_sheet_surfel_share=round(share_multi, 4),
                t_sheet_mm=round(t_sheet, 3), t_sheet_r31b_mm=1.529, t_sheet_columns=int(len(fw)), offset_limit_mm=round(off_lim, 3), density_per_mm2=round(rho, 3),
                thresholds=dict(small_n=small_n, p75=p75, median=med, size_cap=cap, size_prior=sp["area_mm2"]), patches_sheet_graph=st, split_stats=st4, patches_4a=int(len(sizes)), unassigned_after_4a=int((new < 0).sum()),
                patch_size=dict(median=int(np.median(sizes)), p90=int(np.percentile(sizes, 90)), max=int(sizes.max())), planarity=dict(l3_over_l1_median=(round(float(np.median(plan[:, 0])), 4) if len(plan) else None),
                l3_over_l1_p90=(round(float(np.percentile(plan[:, 0], 90)), 4) if len(plan) else None), l2_over_l1_median=(round(float(np.median(plan[:, 1])), 4) if len(plan) else None)),
                multi_sheet_patches=int(sum(multi_patch)), multi_sheet_patch_share=(round(float(np.mean(multi_patch)), 4) if multi_patch else None), seconds=round(time.time() - t0))
    w = cell(iid) / "work"
    np.savez_compressed(w / "geometry.npz", rows=pl["rows"], blade=blade, few=few, r21=r21, r31=r31, b=b, normal=S["normal"].astype(np.float32), n_sheets=S["n_sheets"], index=S["index"], gap=S["gap"], patch_sheet=patch, patch_4a=new)
    json.dump(summ, open(cell(iid) / "geometry.json", "w", encoding="utf-8"), indent=1, default=float); mark(iid, "geometry", dict(patches=int(len(sizes)), t_sheet=round(t_sheet, 3))); log(iid, "geometry done"); return summ


# ---------------------------------------------------------------- Roboflow lane (R): boxes only, key in memory, cached per photo
def step_rf(iid):
    from PIL import Image
    import cv2
    from tomato3d.leafseg import rfdet
    from tomato3d.organs import auto_proposers as AP
    key = AP._rf_key(); d = cell(iid); di = json.load(open(d / "detect_input.json", encoding="utf-8")); missing = []; done_n = 0
    if not key:
        mark(iid, "rf", dict(note="no Roboflow key in memory: layer not used", photos=0)); return
    for v in di["views"]:
        stem = Path(v).stem; info = di["photos"].get(v, {})
        if "error" in info or not (d / "photos" / v).exists():
            continue
        cache = d / "rf_detections" / f"{stem}.json"; tcache = d / "rf_detections_tiled"
        for attempt in range(3):
            try:
                rgb = np.asarray(Image.open(d / "photos" / v).convert("RGB")); c = info.get("crop") or dict(x0=0, y0=0, x1=rgb.shape[1], y1=rgb.shape[0], scale=1.0)
                img = rgb[c["y0"]:c["y1"], c["x0"]:c["x1"]]; img = cv2.resize(img, (int(img.shape[1] * c["scale"]), int(img.shape[0] * c["scale"])), interpolation=cv2.INTER_CUBIC) if c["scale"] > 1 else img
                pred = rfdet.fetch(d / "photos" / v, img, cache, key)
                if len(pred.get("predictions", [])) >= AP.RF_CAP:
                    rfdet.tiled_dets(d / "photos" / v, img, tcache, stem, key)
                done_n += 1; break
            except Exception as e:
                msg = AP._scrub(f"{type(e).__name__}: {e}", key)
                if attempt == 2:
                    missing.append(dict(view=v, error=msg)); log(iid, f"rf {stem}: {msg}")
                else:
                    time.sleep(20)
    mark(iid, "rf", dict(photos=done_n, missing=missing)); log(iid, f"rf done {done_n} photos, {len(missing)} missing")


def write_rf_inputs(iid):
    """Amendment A3: the detector input image of every prepared photo (crop + upsample as in step_rf, JPEG q95) for the local Roboflow lane; written inside the `pre` kernel."""
    from PIL import Image
    import cv2
    d = cell(iid); di = json.load(open(d / "detect_input.json", encoding="utf-8")); out = d / "rf_inputs"; out.mkdir(exist_ok=True); n = 0
    for v in di["views"]:
        info = di["photos"].get(v, {}); f = out / f"{Path(v).stem}.jpg"
        if "error" in info or not (d / "photos" / v).exists() or f.exists():
            continue
        rgb = np.asarray(Image.open(d / "photos" / v).convert("RGB")); c = info.get("crop") or dict(x0=0, y0=0, x1=rgb.shape[1], y1=rgb.shape[0], scale=1.0)
        img = rgb[c["y0"]:c["y1"], c["x0"]:c["x1"]]; img = cv2.resize(img, (int(img.shape[1] * c["scale"]), int(img.shape[0] * c["scale"])), interpolation=cv2.INTER_CUBIC) if c["scale"] > 1 else img
        Image.fromarray(img).save(f, quality=95); n += 1
    return n


STEPS = dict(inputs=step_inputs, prep=step_prep, geometry=step_geometry, rf=step_rf)


HEAVY = {"size_prior", "inputs", "prep", "geometry"}                  # Amendment A3: these run inside a Kaggle kernel only (local worker count 0)


def on_kaggle():
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--id"); ap.add_argument("--step", required=True); a = ap.parse_args(argv)
    if a.step in HEAVY and not on_kaggle():
        raise SystemExit(f"{a.step}: Amendment A3 - heavy steps run on Kaggle only (tomato3d/organs/all3d_ext.py); not started locally")
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "size_prior":
        print(json.dumps(step_size_prior(), indent=1)); return
    lock = cell(a.id) / ("RUNNING_rf.lock" if a.step == "rf" else "RUNNING.lock"); lock.write_text(f"pid {os.getpid()} step {a.step} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        if a.step in state(a.id, a.step):
            log(a.id, f"{a.step}: already done, skipped"); return
        STEPS[a.step](a.id)
    except Exception as e:
        log(a.id, f"{a.step}: FAILED {type(e).__name__}: {e}"); raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
