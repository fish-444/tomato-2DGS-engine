"""microtom_recon one-zip v0.2.2 — `--mask auto+stem`: the mask steps of our R3-1b reconstruction on top of the `auto` masks.

Order, as R3-1b: base training on the `auto` masks -> that model's renders of the TRAINING views -> mask steps -> initial points again
-> final training. Steps (tomato3d/recon/r31_masks.py, functions imported unchanged; the per-view driver below is its `main`, adapted):
  stage a  M2  the stem step run again on the current mask (ridge seeds -> SAM prompts; added region thin, green-gated, touching)
           M3  soil as its own class inside the pot-top polygon  (needs the pot: --pot-top-mm, --pot-side-mm)
  stage b  M1  the previous view's mask lifted through the base model's rendered depth -> SAM box + point prompt
           M5  base render alpha >= 0.8 where the mask is empty and M3 says plant -> SAM prompts
           M4  soft matte + 2 px band (as in `auto`)
Held-out views keep their `auto` masks (R3-1 built these masks for training views only; held-out masks are scoring references).
Not included: the monocular-depth initial points R3-1b also had. Which of the two made R3-1b's stem better was never separated.
SAM weights: our leaflet-tuned SAM 2.1 tiny (v5), NOT in the zip — read from --weights, sha256 checked, never downloaded.
Label of every output: mask_auto_plus_stem_r31b_steps. Checked on one plant only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

WEIGHTS_NAME = "sam2.1_hiera_tiny_leaflet_v5_4plants.pt"
WEIGHTS_SHA256 = "161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d"
LABEL = "mask_auto_plus_stem_r31b_steps"
MARK = "stem_masks.json"
BASE = "base_auto"                      # the `auto` run is kept here: masks, bands, soft, sheet, summary, points, surfels, training folder


def find_weights(a, Stop) -> Path:
    """One clear stop when the weights are not there. No fallback to `auto`, no download."""
    if not a.weights:
        raise Stop(f"--mask auto+stem 에는 --weights <폴더> 가 필요합니다({WEIGHTS_NAME} 이 든 폴더; 비공개 Dataset 으로 붙이세요). 가중치는 zip 에 없고 내려받지 않습니다.")
    root = Path(a.weights); hit = [root] if root.is_file() else sorted(root.rglob(WEIGHTS_NAME)) if root.is_dir() else []
    if not hit:
        raise Stop(f"--weights: {WEIGHTS_NAME} 을 찾지 못했습니다: {root}. auto+stem 은 실행하지 않습니다(auto 로 바꿔 실행하지도 않습니다).")
    h = hashlib.sha256(hit[0].read_bytes()).hexdigest()
    if h != WEIGHTS_SHA256:
        raise Stop(f"--weights: {hit[0].name} 의 sha256 이 기대값과 다릅니다({h[:16]}… / 기대 {WEIGHTS_SHA256[:16]}…). auto+stem 은 실행하지 않습니다.")
    return hit[0]


def check_args(a, Stop) -> Path:
    if a.pot_top_mm is None or a.pot_side_mm is None:
        raise Stop("--mask auto+stem 에는 화분 값이 필요합니다: --pot-top-mm <보드 면에서 화분 윗면(흙 면)까지 높이 mm> --pot-side-mm <화분 윗면 한 변 mm>. "
                   "흙과 식물을 가르는 단계(M3)가 이 값을 씁니다. 기본값은 없습니다.")
    if not (5 <= a.pot_top_mm <= 400 and 10 <= a.pot_side_mm <= 400):
        raise Stop("--pot-top-mm / --pot-side-mm 값이 범위를 벗어났습니다(mm 단위인지 확인하세요).")
    return find_weights(a, Stop)


def _sheet(out: Path, views, held, R, path):
    import cv2, numpy as np
    from PIL import Image
    po = json.load(open(out / "poses" / "poses.json")); ud = out / "poses" / f"undist_{po['undistorted_res']}"; tiles = []
    for v in views[:12]:
        o = cv2.resize(np.asarray(Image.open(ud / v).convert("RGB")), (R, R), interpolation=cv2.INTER_AREA).astype(np.float32)
        new = np.asarray(Image.open(out / "masks" / f"{Path(v).stem}.png")) > 0; old = np.asarray(Image.open(out / BASE / "masks" / f"{Path(v).stem}.png")) > 0
        o[new & old] = o[new & old] * 0.5 + np.array([255, 0, 190]) * 0.5; o[new & ~old] = o[new & ~old] * 0.3 + np.array([0, 255, 255]) * 0.7      # cyan = added by the stem steps
        o[old & ~new] = o[old & ~new] * 0.3 + np.array([255, 40, 40]) * 0.7                                                                    # red = removed
        t = cv2.resize(o.astype(np.uint8), (300, 300)); cv2.putText(t, f"{Path(v).stem[-12:]} {'H' if v in held else 'T'}", (4, 18), 0, 0.5, (255, 255, 0), 2); tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    Image.fromarray(np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)])).save(path, quality=85)


def stem_steps(a, out: Path, work: Path, run: str, log, ck: Path) -> dict:
    """r31_masks.main, stages a then b, on this run's folders. Parameters are the ones inside r31_masks / r31_contract / wholeplant_masks, unchanged."""
    import cv2, numpy as np, torch
    from PIL import Image
    from tomato3d.recon import charuco_a3 as ca
    from tomato3d.recon import r31_masks as RM
    from tomato3d.recon import wholeplant_masks as WM
    from tomato3d.recon.r31_contract import M as METHODS
    import run_recon as RR                                                                  # SAM_CFG and the pinned sam2 install of the `auto` stage
    os.environ.setdefault("SAM2_BUILD_CUDA", "0"); RR.ensure("hydra", RR.PINS["hydra"], log); RR.ensure("iopath", RR.PINS["iopath"], log)
    RR.ensure("sam2", RR.PINS["sam2"], log, extra=("--no-deps", "--no-build-isolation"))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    MASK_RES = RM.MASK_RES; dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = SAM2ImagePredictor(build_sam2(RR.SAM_CFG, str(ck), device=dev, apply_postprocessing=False), max_hole_area=0, max_sprinkle_area=0)

    class _Sam:                                                                             # r31_masks.sam_from_region expects `.predictor`
        predictor = pred
    sam = _Sam()
    po = json.load(open(out / "poses" / "poses.json")); sp = json.load(open(out / "split.json")); train, held = sp["train"], sp["heldout"]; R0 = po["undistorted_res"]
    K = ca.scaled_K(np.array(po["K_undist"]), R0, MASK_RES); hull = json.load(open(out / BASE / "masks_summary.json"))["_hull"]; c = (np.array(hull["lo"]) + np.array(hull["hi"])) / 2
    z_top, side = a.pot_top_mm / 1000.0, a.pot_side_mm / 1000.0; z = np.load(out / BASE / "train" / "out" / f"{run}_views.npz")
    cams = {v: (np.array(po["views"][v]["R_cw"]), np.array(po["views"][v]["t_cw"]), K) for v in train}
    qc = {r["file"]: r for r in csv.DictReader(open(out / "poses" / "qc.csv"))}
    ring_of = lambda v: "top" if float(qc[v]["elevation_deg"]) > 75 else "upper" if float(qc[v]["elevation_deg"]) > 58 else "middle" if float(qc[v]["elevation_deg"]) > 45 else "low"  # noqa: E731
    order = sorted(train, key=lambda v: (ring_of(v), float(qc[v]["azimuth_deg"]))); summ = {}
    dirs = {s: {k: out / "_stem" / f"{k}_{s}" for k in ("masks", "bands", "soft")} for s in ("a", "b")}
    for s in dirs.values():
        for d in s.values():
            d.mkdir(parents=True, exist_ok=True)
    for stage in ("a", "b"):
        rng = np.random.default_rng(0); prev = None; base_dir = out / BASE / "masks" if stage == "a" else dirs["a"]["masks"]; md, bd, sd = (dirs[stage][k] for k in ("masks", "bands", "soft"))
        for i, v in enumerate(order):
            st = Path(v).stem; rgb3 = np.asarray(Image.open(out / "poses" / f"undist_{R0}" / v).convert("RGB"))
            if rgb3.shape[0] != WM.RES:
                rgb3 = cv2.resize(rgb3, (WM.RES, WM.RES), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(rgb3, (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA)
            lab0 = np.asarray(Image.open(base_dir / f"{st}.png")); plant0 = np.isin(lab0, (1, 2)); plant = plant0.copy(); rec = summ.setdefault(v, {}); rec[f"{stage}_base_px"] = int(plant0.sum())
            R, t, _ = cams[v]
            with torch.inference_mode():
                if stage == "a":
                    if plant.sum() >= 500:
                        # M2: thin-structure prompts (the A2 stem step on the current mask)
                        pos = WM.seeds(rgb3, cv2.resize(plant.astype(np.uint8), (WM.RES, WM.RES), interpolation=cv2.INTER_NEAREST) > 0, rng)
                        ys, xs = np.nonzero(plant); pad = 0.06 * max(np.ptp(xs), np.ptp(ys)); s3 = WM.RES / MASK_RES
                        box = np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], np.float32) * s3
                        dt = cv2.distanceTransform(plant.astype(np.uint8), cv2.DIST_L2, 5); py, px = np.unravel_index(int(np.argmax(dt)), dt.shape)
                        pos = np.vstack([[[px * s3, py * s3]], pos]) if len(pos) else np.array([[px * s3, py * s3]], np.float32)
                        neg = WM.negatives(R, t, ca.scaled_K(np.array(po["K_undist"]), R0, WM.RES), c, WM.RES)
                        pred.set_image(rgb3); pts = np.vstack([pos, neg]) if len(neg) else pos
                        masks, ious, _ = pred.predict(box=box, point_coords=pts, point_labels=np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32), multimask_output=True)
                        best = max(zip(masks, ious), key=lambda m: float(m[1]))[0] > 0
                        big = cv2.resize(plant.astype(np.uint8), (WM.RES, WM.RES), interpolation=cv2.INTER_NEAREST) > 0
                        lb = cv2.cvtColor(rgb3, cv2.COLOR_RGB2LAB).astype(np.float32); gate = ((lb[..., 1] - 128) <= WM.ADD_A_MAX) & (lb[..., 0] * 100 / 255 >= WM.ADD_L_MIN)
                        add3 = WM.thin_parts(best & ~big & gate, big); add = cv2.resize(add3.astype(np.uint8), (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST) > 0
                        plant |= add; rec["m2_added_px"] = int(add.sum())
                    # M3: soil class inside the pot polygon
                    poly = RM.pot_polygon(c, R, t, K, MASK_RES, side=side, z=z_top); pin, soil = RM.soil_split(rgb, plant, poly)
                    removed = plant & poly & ~pin; added3 = pin & ~plant
                    added3 = cv2.morphologyEx(added3.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
                    plant = (plant & ~removed) | added3; rec.update(m3_removed_px=int(removed.sum()), m3_added_px=int(added3.sum()), m3_soil_px=int(soil.sum()))
                else:
                    # M1: propagate from the previous view in ring order through the base model's rendered depth
                    if prev is not None and f"d_{prev}" in z:
                        lifted = RM.lift_mask(np.isin(np.asarray(Image.open(dirs["a"]["masks"] / f"{Path(prev).stem}.png")), (1, 2)), z[f"d_{prev}"].astype(np.float32), cams[prev], cams[v], MASK_RES)
                        pred.set_image(rgb3); m1 = RM.sam_from_region(sam, rgb3, lifted & ~plant, rng=rng)
                        ys, xs = np.nonzero(cv2.dilate(plant.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0)
                        region = np.zeros_like(plant)
                        if len(ys):
                            region[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                        lb = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); gate = ((lb[..., 1] - 128) <= WM.ADD_A_MAX) & (lb[..., 0] * 100 / 255 >= WM.ADD_L_MIN)
                        add = m1 & region & gate & ~plant; plant |= add; rec["m1_added_px"] = int(add.sum())
                    else:
                        pred.set_image(rgb3); rec["m1_added_px"] = 0
                    # M5: one 3D->2D round from the base model's alpha (training views only)
                    if f"a_{v}" in z:
                        al = z[f"a_{v}"].astype(np.float32); miss = (al >= METHODS["M5"]["alpha_min"]) & ~plant
                        poly = RM.pot_polygon(c, R, t, K, MASK_RES, side=side, z=z_top); pin, _ = RM.soil_split(rgb, plant, poly | miss)
                        cand = miss & (pin | ~poly)
                        m5 = RM.sam_from_region(sam, rgb3, cand, rng=rng) if cand.sum() >= 50 else np.zeros_like(plant)
                        add = m5 & cv2.dilate(cand.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~plant; plant |= add
                        rec.update(m5_candidate_px=int(cand.sum()), m5_added_px=int(add.sum()))
            # M4: matte + band
            plant = cv2.morphologyEx(plant.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            alpha, band = RM.matte(rgb, plant)
            lab = np.zeros((MASK_RES, MASK_RES), np.uint8); lab[lab0 == 3] = 3; lab[plant & (lab0 == 1)] = 1; lab[plant & (lab0 != 1)] = 2
            Image.fromarray(lab).save(md / f"{st}.png"); Image.fromarray(band.astype(np.uint8)).save(bd / f"{st}.png"); Image.fromarray((alpha * 255).round().astype(np.uint8)).save(sd / f"{st}.png")
            rec[f"{stage}_final_px"] = int(plant.sum()); prev = v
            if i % 10 == 0:
                log(f"stem masks {stage}: {i + 1}/{len(order)} {v} " + " ".join(f"{k}={rec[k]}" for k in rec if k.startswith(("m1", "m2", "m3", "m5"))))
    del pred
    if dev == "cuda":
        torch.cuda.empty_cache()
    for v in train:                                                                         # training views take the stage-b masks; held-out views keep `auto`
        for k in ("masks", "bands", "soft"):
            shutil.copy(dirs["b"][k] / f"{Path(v).stem}.png", out / k / f"{Path(v).stem}.png")
    med = lambda k: float(np.median([r.get(k, 0) for r in summ.values()]))                # noqa: E731
    return dict(views=summ, median=dict(auto_px=med("a_base_px"), m2_added_px=med("m2_added_px"), m3_removed_px=med("m3_removed_px"), m3_added_px=med("m3_added_px"),
                                        m1_added_px=med("m1_added_px"), m5_added_px=med("m5_added_px"), final_px=med("b_final_px")), train_views=len(train), heldout_keep_auto=held, mask_res=MASK_RES)


def run(a, out: Path, work: Path, run_id: str, log, stage_points, stage_train, Stop):
    """Called by run_recon.py after the base (`auto`) training. Skipped when already done."""
    mark = out / MARK
    if mark.exists() and json.load(open(mark)).get("done"):
        log("stem masks: skip (output exists)"); return
    ck = check_args(a, Stop); t0 = time.time(); b = out / BASE
    if not (b / "surfels_full.ply").exists():                                               # keep the `auto` run before anything is overwritten
        b.mkdir(exist_ok=True)
        for d in ("masks", "bands", "soft"):
            shutil.copytree(out / d, b / d, dirs_exist_ok=True)
        for f in ("masks_sheet.jpg", "masks_summary.json", "plant_points.ply", "plant_points_dense.ply", "sparse_dense_checks.json"):
            if (out / f).exists():
                shutil.copy2(out / f, b / f)
        shutil.copytree(out / "train", b / "train", dirs_exist_ok=True); shutil.copy2(out / "surfels_full.ply", b / "surfels_full.ply")
    log(f"stem masks: start (weights {ck.name}, sha256 {WEIGHTS_SHA256[:16]}; pot top {a.pot_top_mm:g} mm, side {a.pot_side_mm:g} mm) [{LABEL}]")
    res = stem_steps(a, out, work, run_id, log, ck); t1 = time.time()
    sp = json.load(open(out / "split.json")); _sheet(out, sp["heldout"] + sp["train"][::max(1, len(sp["train"]) // 10)], sp["heldout"], 1512, out / "masks_sheet.jpg")
    ms = json.load(open(b / "masks_summary.json")); auto_method = ms["mask_method"]
    ms.update(mask_method=auto_method + " -> base training -> R3-1b steps M2 (stem step again), M3 (soil class), M1 (propagation through the base render), M5 (base alpha round), M4 matte; "
                          "training views only, held-out views keep the auto masks; no monocular-depth points", mask_label=LABEL, mask_mode="sam2+stem",
              stem=dict(weights=ck.name, weights_sha256=WEIGHTS_SHA256, pot_top_mm=a.pot_top_mm, pot_side_mm=a.pot_side_mm, median=res["median"], train_views=res["train_views"],
                        heldout_keep_auto=res["heldout_keep_auto"], base_run=BASE + "/", checked="pilot plant only"))
    ms["plant_px_median"] = res["median"]["final_px"]; json.dump(ms, open(out / "masks_summary.json", "w"), indent=1)
    json.dump(dict(done=False, steps=res), open(mark, "w"), indent=1)
    log(f"stem masks: done {t1 - t0:.0f} s; median px per training view auto {res['median']['auto_px']:.0f} -> {res['median']['final_px']:.0f} "
        f"(M2 +{res['median']['m2_added_px']:.0f}, M3 -{res['median']['m3_removed_px']:.0f}/+{res['median']['m3_added_px']:.0f}, M1 +{res['median']['m1_added_px']:.0f}, M5 +{res['median']['m5_added_px']:.0f})")
    log.stage(4, "points (stem masks)", lambda: False, lambda: stage_points(out, work, log))
    shutil.rmtree(out / "train", ignore_errors=True); (out / "surfels_full.ply").unlink()
    log.stage(5, "train (stem masks)", lambda: False, lambda: stage_train(a, out, work, run_id, log))
    json.dump(dict(done=True, label=LABEL, seconds_stem_masks=round(t1 - t0), steps=res), open(mark, "w"), indent=1); shutil.rmtree(out / "_stem", ignore_errors=True)
