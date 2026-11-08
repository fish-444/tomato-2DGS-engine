"""Flower / young leaf / mature leaf by calibrated colour (contract build/p1_flower_youngleaf_colour_20261101_01/CONTRACT.md).
2D first: ChArUco colour calibration per photo, a seeded 40-photo labelling sheet with loose candidate regions, a local labelling page (the USER labels, never the session),
then (after labels exist) a depth <= 3 tree fitted once on the fit plants, one scoring run on the test plants, and a per-surfel flag for the classes that pass.
Light task: photos are decoded one at a time, local memory <= 1 GB. No SAM, no 3D input before the transfer step.

usage: python -m tomato3d.organs.colour_organ --step select|calibrate|sheet|page|fit|score|transfer
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tomato3d.organs import light_organs as LO
from tomato3d.recon import charuco_a3 as ca

ROOT = LO.ROOT
OUT = ROOT / "build/p1_flower_youngleaf_colour_20261101_01"
ALLB = ROOT / "build/p1_recon_all_20261025_01"
POSES_CSV = ROOT / "build/p1_autolabel_20261026_01/poses"
R31B_PHOTOS = ROOT / "build/p1_charuco_a3_20261021_01/undist_3024"
R31B_MASKS = ROOT / "build/p1_recon_charuco_20261021_01/masks"
PLANTS = ["R3-1b", "n_01", "n_02", "n_03", "n_04", "n_05", "n_06", "t2_02", "t2_04", "t2_05"]
LABELS = dict(flag="colour_organ_flag", scale=LO.SCALE_TAG)
CLASSES = ["flower", "bud", "young leaf", "mature leaf", "senescent-yellow", "other"]
CAL = dict(white_ring_mm=(15.5, 17.5), black_half_mm=10.0, step_mm=1.0, clip=250, neutral_ratio=1.35, top_share=0.40, min_squares=8, ref_white=0.85, max_gain_clip=1.2)
SEL = dict(seed=20261101, n_per_plant=4, top_el=75.0, side_el=(30.0, 70.0), side_corners=20, daz_deg=90.0, low_corners=8, max_low=4)
RULE = dict(yellow=dict(h=(35, 70), s=0.35, v=0.40, area=25, cap=4), light_green=dict(h=(70, 100), s=0.25, v=0.45, area=40, cap=4, top_share=0.20),
            control=dict(h=(90, 130), s=0.20, v_max=0.60, area=80, cap=3), bbox_margin=0.15, grid=1536, overlap=0.30)
MID_Z = 0.12            # px/mm is taken at this plane height (mid-canopy approximation, +-6 % area)
GATE = dict(flower_p=0.80, flower_r=0.70, young_p=0.70, min_n=10)


# ---------------------------------------------------------------- helpers
def srgb_to_lin(u8):
    x = u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


# ---------------------------------------------------------------- photo inventory and loading
def inventory(plant):
    rows = [r for r in csv.DictReader(open(POSES_CSV / f"{plant}.csv", encoding="utf-8")) if r.get("elevation_deg") and r.get("flag") != "pose_board_only"]
    return [dict(plant=plant, view=r["view"], elevation=float(r["elevation_deg"]), azimuth=float(r["azimuth_deg"]), corners=int(float(r["corners"] or 0)), flag=r["flag"]) for r in rows]


def pose_file(plant):
    from tomato3d.organs import all_compound as AC
    return LO.DEFAULTS["poses"] if plant == "R3-1b" else AC.POSES[plant]


_HEIC = {}


def heic_path(view):
    from tomato3d.organs import all_compound as AC
    if not _HEIC:
        _HEIC.update(AC.heic_index())
    return _HEIC[view[:-4] + ".HEIC"][0]


def load_photo(plant, view):
    """Undistorted 3024 RGB (uint8), K (3024), R_cw, t_cw, plant mask (1536 bool), capture time string."""
    pz = json.load(open(pose_file(plant), encoding="utf-8"))
    if plant == "R3-1b":
        rgb = np.asarray(Image.open(R31B_PHOTOS / view).convert("RGB")); K = np.array(pz["K_undist"]); v = pz["views"][view]
        mask = np.asarray(Image.open(R31B_MASKS / (Path(view).stem + ".png")).resize((1536, 1536), Image.NEAREST)) == 1
        m = re.match(r"\d{8}_(\d{2})(\d{2})(\d{2})", view); t = f"2026-10-21 {m.group(1)}:{m.group(2)}:{m.group(3)}" if m else ""
    else:
        import pillow_heif
        pillow_heif.register_heif_opener(); v = pz["views"][view]; K = np.array(v["K"]); dist = np.array(v["dist"]); f = heic_path(view)
        im = Image.open(f); exif = im.getexif(); t = str(exif.get(36867) or exif.get(306) or "")
        try:
            t = str(im.getexif().get_ifd(0x8769).get(36867, t))
        except Exception:
            pass
        rgb = cv2.undistort(np.asarray(im.convert("RGB")), K, dist, None, K)
        mask = np.asarray(Image.open(ALLB / "k3a" / plant / "masks" / (Path(view).stem + ".png")).resize((1536, 1536), Image.NEAREST)) == 1
    return rgb, K, np.array(v["R_cw"]), np.array(v["t_cw"]), mask, t


# ---------------------------------------------------------------- calibration
def board_squares():
    b = ca.board(); ids = b.getIds().ravel(); objs = b.getObjPoints(); w = set()
    for k, pts in zip(ids, objs):
        c = np.mean(pts, axis=0); w.add((int(c[0] // ca.SQUARE_M), int(c[1] // ca.SQUARE_M)))
    nx, ny = ca.SQUARES; white = sorted(w); black = [(i, j) for i in range(nx) for j in range(ny) if (i, j) not in w]
    return white, black


def project(K, R, t, Pw):
    xc = Pw @ R.T + t; z = xc[:, 2]; uv = (xc / np.where(z > 1e-9, z, 1e-9)[:, None]) @ K.T; return uv[:, 0], uv[:, 1], z


def calibrate_photo(rgb, K, R, t):
    """Per-photo white / black reference from the board squares projected with the reconstruction pose. Returns dict with gains or status not_calibrated."""
    white, black = board_squares(); sq = ca.SQUARE_M; h, w = rgb.shape[:2]; lin_cache = {}
    g = np.arange(-sq / 2, sq / 2 + 1e-9, CAL["step_mm"] / 1000)
    U, V = np.meshgrid(g, g); U, V = U.ravel(), V.ravel(); a = np.maximum(np.abs(U), np.abs(V)); b = np.minimum(np.abs(U), np.abs(V))
    ring = (a >= CAL["white_ring_mm"][0] / 1000) & (a <= CAL["white_ring_mm"][1] / 1000) & (b <= CAL["white_ring_mm"][1] / 1000)
    core = (np.abs(U) <= CAL["black_half_mm"] / 1000) & (np.abs(V) <= CAL["black_half_mm"] / 1000)
    def sample(squares, sel):
        meds, n_clip = [], 0
        for (i, j) in squares:
            cx, cy = (i + 0.5) * sq, (j + 0.5) * sq; Pb = np.c_[cx + U[sel], cy + V[sel], np.zeros(sel.sum())]; Pw = (ca.A_WB @ Pb.T).T + ca.B_WB
            u, v, z = project(K, R, t, Pw); ui, vi = np.round(u).astype(int), np.round(v).astype(int); ok = (z > 0) & (ui >= 0) & (vi >= 0) & (ui < w) & (vi < h)
            if ok.sum() < 20:
                continue
            px = rgb[vi[ok], ui[ok]]; meds.append((px.max(), np.median(srgb_to_lin(px), axis=0), np.median(px, axis=0)))
        return meds
    wm = sample(white, ring); bm = sample(black, core); clipped = 0; wq = []
    for mx, lin, raw in wm:
        if raw.max() >= CAL["clip"]:
            clipped += 1; continue
        if raw.max() / max(raw.min(), 1e-3) > CAL["neutral_ratio"]:
            continue
        wq.append(lin)
    out = dict(white_squares_seen=len(wm), white_usable=len(wq), white_clipped=clipped, black_squares_seen=len(bm))
    if len(wq) < CAL["min_squares"] or len(bm) < CAL["min_squares"]:
        out.update(status="not_calibrated", reason="fewer than %d usable white or black squares" % CAL["min_squares"]); return out
    wl = np.array(sorted(wq, key=lambda x: -x.mean())[:max(1, int(round(CAL["top_share"] * len(wq))))]); bl = np.array(sorted([m[1] for m in bm], key=lambda x: x.mean())[:max(1, int(round(CAL["top_share"] * len(bm))))])
    out.update(status="calibrated", white_lin=np.median(wl, axis=0).tolist(), black_lin=np.median(bl, axis=0).tolist())
    out["gain"] = (CAL["ref_white"] / np.maximum(np.array(out["white_lin"]) - np.array(out["black_lin"]), 1e-4)).tolist(); return out


def apply_cal(img_u8, cal):
    """Calibrated sRGB uint8 image (same size). Not calibrated -> the original image unchanged."""
    if cal.get("status") != "calibrated":
        return img_u8
    lin = srgb_to_lin(img_u8); wl = np.array(cal["white_lin"], np.float32); bl = np.array(cal["black_lin"], np.float32)
    lin = np.clip((lin - bl) / np.maximum(wl - bl, 1e-4), 0, CAL["max_gain_clip"]) * CAL["ref_white"]
    return (lin_to_srgb(lin) * 255 + 0.5).astype(np.uint8)


def to_lab(cal_u8):
    lin = srgb_to_lin(cal_u8); M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32); xyz = lin @ M.T
    xyz = xyz / (CAL["ref_white"] * np.array([0.9505, 1.0, 1.089], np.float32)); f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def to_hsv(cal_u8):
    hsv = cv2.cvtColor(cal_u8, cv2.COLOR_RGB2HSV); return np.stack([hsv[..., 0].astype(np.float32) * 2, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0], -1)


# ---------------------------------------------------------------- selection (seeded, fixed)
def select_photos():
    rng = np.random.default_rng(SEL["seed"]); chosen = []; low_pool = []
    for plant in PLANTS:
        inv = inventory(plant); used = set(); pick = []
        top = max(inv, key=lambda r: (r["elevation"], r["corners"])); pick.append(dict(top, why="highest elevation")); used.add(top["view"])
        side = [r for r in inv if SEL["side_el"][0] <= r["elevation"] <= SEL["side_el"][1] and r["corners"] >= SEL["side_corners"] and r["view"] not in used]
        if side:
            s1 = side[int(rng.integers(len(side)))]; pick.append(dict(s1, why="side view 1")); used.add(s1["view"])
            far = [r for r in side if r["view"] not in used and abs((r["azimuth"] - s1["azimuth"] + 180) % 360 - 180) >= SEL["daz_deg"]] or [r for r in side if r["view"] not in used]
            if far:
                s2 = far[int(rng.integers(len(far)))]; pick.append(dict(s2, why="side view 2 (azimuth apart)")); used.add(s2["view"])
        low = [r for r in inv if r["corners"] < SEL["low_corners"] and r["view"] not in used]; low_pool.append((plant, low))
        rest = [r for r in inv if r["view"] not in used]; chosen.append((plant, pick, used, rest))
    out = []; n_low = 0
    for (plant, pick, used, rest), (_, low) in zip(chosen, low_pool):
        if len(pick) < SEL["n_per_plant"] and low and n_low < SEL["max_low"]:
            r = low[int(rng.integers(len(low)))]; pick.append(dict(r, why="few board corners (not_calibrated row)")); used.add(r["view"]); n_low += 1
        while len(pick) < SEL["n_per_plant"]:
            rest = [r for r in rest if r["view"] not in used]
            if not rest:
                break
            r = rest[int(rng.integers(len(rest)))]; pick.append(dict(r, why="remaining set")); used.add(r["view"])
        out += pick
    return out


def step_select():
    rows = select_photos(); OUT.mkdir(exist_ok=True)
    with open(OUT / "selected_photos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["photo_id", "plant", "view", "elevation", "azimuth", "corners", "flag", "why"]); w.writeheader()
        for r in rows:
            w.writerow(dict(photo_id=f"{r['plant']}__{Path(r['view']).stem}", plant=r["plant"], view=r["view"], elevation=r["elevation"], azimuth=r["azimuth"], corners=r["corners"], flag=r["flag"], why=r["why"]))
    print(len(rows), "photos"); return rows


def selected():
    return list(csv.DictReader(open(OUT / "selected_photos.csv", encoding="utf-8")))


# ---------------------------------------------------------------- calibration table + sheet
def ppm_1536(K, R, t, mask):
    """px/mm of the 1536 grid at the plane z = MID_Z through the plant-mask centroid ray."""
    ys, xs = np.nonzero(mask); Ki = np.linalg.inv(ca.scaled_K(K, 3024, 1536)); d = np.array([xs.mean(), ys.mean(), 1.0]) @ Ki.T; d = R.T @ d; o = -R.T @ t
    s = (MID_Z - o[2]) / d[2]; X = o + s * d; zc = (R @ X + t)[2]; return float(ca.scaled_K(K, 3024, 1536)[0, 0] * 0.001 / zc)


def step_calibrate():
    cal_path = OUT / "calibration.csv"; rows = []; (OUT / "cal").mkdir(exist_ok=True)
    for r in selected():
        rgb, K, R, t, mask, when = load_photo(r["plant"], r["view"]); c = calibrate_photo(rgb, K, R, t); c.update(photo_id=r["photo_id"], plant=r["plant"], capture_time=when, ppm_1536=round(ppm_1536(K, R, t, mask), 3) if mask.any() else None)
        json.dump(c, open(OUT / "cal" / f"{r['photo_id']}.json", "w", encoding="utf-8"), default=float)
        rows.append(dict(photo_id=r["photo_id"], plant=r["plant"], view=r["view"], capture_time=when, status=c["status"], white_usable=c["white_usable"], white_clipped=c["white_clipped"], black_seen=c["black_squares_seen"],
                         white_lin=json.dumps([round(x, 4) for x in c.get("white_lin", [])]), black_lin=json.dumps([round(x, 4) for x in c.get("black_lin", [])]), gain=json.dumps([round(x, 3) for x in c.get("gain", [])]), ppm_1536=c["ppm_1536"]))
        print(r["photo_id"], c["status"], c["white_usable"], c["white_clipped"], c["black_squares_seen"], flush=True)
    with open(cal_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def components(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8); return [(lab == i, int(st[i, cv2.CC_STAT_AREA])) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def propose(cal1536, plant, rng):
    """Loose candidate regions (rule section 2). Returns list of dict(kind, mask)."""
    hsv = to_hsv(cal1536); H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]; ys, xs = np.nonzero(plant); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    mx, my = int(RULE["bbox_margin"] * (x1 - x0)), int(RULE["bbox_margin"] * (y1 - y0)); box = np.zeros_like(plant); box[max(0, y0 - my):y1 + my + 1, max(0, x0 - mx):x1 + mx + 1] = True
    rows = np.arange(plant.shape[0])[:, None] * np.ones((1, plant.shape[1])); upper = rows <= y0 + RULE["light_green"]["top_share"] * (y1 - y0); k3 = np.ones((3, 3), np.uint8); out = []
    def add(kind, m, area, cap, rank_key=lambda c: -c[1]):
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0; cs = sorted(components(m, area), key=rank_key)[:cap]
        for c, a in cs:
            if all((c & o["mask"]).sum() / max((c | o["mask"]).sum(), 1) <= RULE["overlap"] for o in out):
                out.append(dict(kind=kind, mask=c, area=a))
    y = RULE["yellow"]; add("yellow", box & (H >= y["h"][0]) & (H <= y["h"][1]) & (S >= y["s"]) & (V >= y["v"]), y["area"], y["cap"])
    g = RULE["light_green"]; add("light_green", plant & upper & (H > g["h"][0]) & (H <= g["h"][1]) & (S >= g["s"]) & (V >= g["v"]), g["area"], g["cap"])
    c = RULE["control"]; m = plant & ~upper & (H >= c["h"][0]) & (H <= c["h"][1]) & (S >= c["s"]) & (V <= c["v_max"]); m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
    cs = components(m, c["area"]); idx = rng.permutation(len(cs))[:c["cap"]]
    for i in idx:
        cc, a = cs[i]
        if all((cc & o["mask"]).sum() / max((cc | o["mask"]).sum(), 1) <= RULE["overlap"] for o in out):
            out.append(dict(kind="dark_green_control", mask=cc, area=a))
    return out


def rle(m):
    from tomato3d.organs import seed2d as S2
    return dict(size=[1536, 1536], counts=S2.rle_encode(m))


def unrle(r):
    from tomato3d.leafseg import pixels as px
    return px.rle_decode(r)


def step_sheet():
    rng = np.random.default_rng(SEL["seed"] + 1); photos, regions = [], []; d = OUT / "sheet"; (d / "crops").mkdir(parents=True, exist_ok=True); (d / "photos").mkdir(exist_ok=True)
    cal_rows = {r["photo_id"]: r for r in csv.DictReader(open(OUT / "calibration.csv", encoding="utf-8"))}
    for r in selected():
        pid = r["photo_id"]; rgb, K, R, t, mask, when = load_photo(r["plant"], r["view"]); cal = json.load(open(OUT / "cal" / f"{pid}.json", encoding="utf-8"))
        small = cv2.resize(rgb, (1536, 1536), interpolation=cv2.INTER_AREA); calimg = apply_cal(small, cal); regs = propose(calimg, mask, rng); over = calimg.copy()
        for k, g in enumerate(regs):
            rid = f"{pid}_r{k + 1:02d}"; ys, xs = np.nonzero(g["mask"]); x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            pad = max(12, int(0.6 * max(x1 - x0, y1 - y0))); cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(1536, x1 + pad), min(1536, y1 + pad)
            def crop(img):
                c = img[cy0:cy1, cx0:cx1].copy(); cv2.rectangle(c, (x0 - cx0, y0 - cy0), (x1 - cx0 - 1, y1 - cy0 - 1), (255, 0, 255), 1); s = 200 / max(c.shape[:2]); return cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
            a, b = crop(small), crop(calimg); hh = max(a.shape[0], b.shape[0]); pad_ = lambda im: np.pad(im, ((0, hh - im.shape[0]), (0, 0), (0, 0)), constant_values=40)
            Image.fromarray(np.hstack([pad_(a), pad_(b)])).save(d / "crops" / f"{rid}.jpg", quality=88)
            cv2.rectangle(over, (x0, y0), (x1, y1), (255, 0, 255), 2); cv2.putText(over, str(k + 1), (x0, max(14, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            regions.append(dict(region_id=rid, photo_id=pid, n=k + 1, kind=g["kind"], bbox=[x0, y0, x1, y1], area_px=g["area"], mask=rle(g["mask"])))
        Image.fromarray(cv2.resize(over, (1100, 1100), interpolation=cv2.INTER_AREA)).save(d / "photos" / f"{pid}.jpg", quality=85)
        photos.append(dict(photo_id=pid, plant=r["plant"], view=r["view"], elevation=float(r["elevation"]), status=cal["status"], regions=[x["region_id"] for x in regions if x["photo_id"] == pid], note=r["why"]))
        print(pid, cal["status"], len(photos[-1]["regions"]), "regions", flush=True)
    json.dump(dict(photos=photos, regions=regions, classes=CLASSES, labels=LABELS), open(d / "sheet.json", "w", encoding="utf-8"))
    return photos, regions


# ---------------------------------------------------------------- labelling page (the user labels)
PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>색 라벨</title>
<style>body{margin:0;background:#161616;color:#eee;font:14px system-ui,sans-serif}#top{position:sticky;top:0;background:#222;padding:8px 12px;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:5px 10px;cursor:pointer;font:inherit}button.on{background:#2d6a3e;border-color:#7fd48c}
#main{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:12px;padding:10px}@media(max-width:1000px){#main{grid-template-columns:1fr}}img{max-width:100%;display:block}
.card{background:#1f1f1f;border:1px solid #333;border-radius:8px;padding:6px;margin-bottom:8px}.card .btns{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}small{opacity:.7}</style></head><body>
<div id="top"><b>꽃 / 어린잎 / 성숙잎 색 라벨</b><span id="nav"></span><button id="prev">◀</button><button id="next">▶</button><span id="prog"></span><span id="msg"></span></div>
<div id="main"><div><img id="ov"><div id="info"></div></div><div id="cards"></div></div>
<script>let D=null,idx=0,L={};const CL=['flower','bud','young leaf','mature leaf','senescent-yellow','other'];const KO={'flower':'꽃','bud':'꽃눈','young leaf':'어린잎','mature leaf':'성숙잎','senescent-yellow':'노랗게 늙은 잎','other':'기타'};
async function post(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json();}
function prog(){const tot=D.regions.length,n=Object.keys(L).length;document.getElementById('prog').textContent=`라벨 ${n}/${tot} · 남은 약 ${Math.max(0,Math.round((tot-n)*4/60))}분`;}
function show(){const p=D.photos[idx];document.getElementById('nav').textContent=`사진 ${idx+1}/${D.photos.length} · ${p.photo_id}`;document.getElementById('ov').src='/img/photos/'+p.photo_id+'.jpg';
 document.getElementById('info').innerHTML=`<small>${p.plant} · 고도 ${p.elevation.toFixed(0)}° · ${p.status==='calibrated'?'색 보정됨':'<b>색 보정 안 됨</b> (원본 색)'} · 오른쪽 그림: 왼쪽 원본 | 오른쪽 보정 · 자주색 네모가 후보, 번호는 왼쪽 사진과 같음</small>`;
 const c=document.getElementById('cards');c.innerHTML='';for(const rid of p.regions){const r=D.regions.find(x=>x.region_id===rid);const el=document.createElement('div');el.className='card';
  el.innerHTML=`<div><b>${r.n}</b> <small>${rid}</small></div><img src="/img/crops/${rid}.jpg"><div class="btns"></div>`;const bs=el.querySelector('.btns');
  CL.forEach(k=>{const b=document.createElement('button');b.textContent=KO[k];b.className=(L[rid]===k?'on':'');b.onclick=async()=>{await post('/api/label',{region_id:rid,photo_id:p.photo_id,cls:k});L[rid]=k;show();prog();};bs.appendChild(b);});c.appendChild(el);}
 if(!p.regions.length)c.innerHTML='<div class="card">이 사진에는 후보가 없습니다.</div>';}
document.getElementById('prev').onclick=()=>{idx=(idx+D.photos.length-1)%D.photos.length;show();};document.getElementById('next').onclick=()=>{idx=(idx+1)%D.photos.length;show();};
(async()=>{D=await (await fetch('/api/sheet')).json();L=await (await fetch('/api/labels')).json();show();prog();})();</script></body></html>"""


def serve(port=8840):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    sheet = json.load(open(OUT / "sheet" / "sheet.json", encoding="utf-8")); raw = OUT / "labels_raw.jsonl"

    def latest():
        L = {}
        if raw.exists():
            for line in raw.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line); L[r["region_id"]] = r["cls"]
        return L

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/sheet":
                return self._send(json.dumps(dict(photos=sheet["photos"], regions=[{k: v for k, v in r.items() if k != "mask"} for r in sheet["regions"]])).encode(), "application/json")
            if self.path == "/api/labels":
                return self._send(json.dumps(latest()).encode(), "application/json")
            if self.path.startswith("/img/"):
                f = (OUT / "sheet" / self.path[5:]).resolve()
                if str(f).startswith(str((OUT / "sheet").resolve())) and f.exists():
                    return self._send(f.read_bytes(), "image/jpeg")
            self.send_response(404); self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); b = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/label" and b.get("cls") in CLASSES:
                with open(raw, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(region_id=b["region_id"], photo_id=b["photo_id"], cls=b["cls"], user="user", provenance="human_click", at=time.strftime("%Y-%m-%dT%H:%M:%S")), ensure_ascii=False) + "\n")
                return self._send(b'{"ok":true}', "application/json")
            self.send_response(400); self.end_headers()

        def log_message(self, *a):
            pass
    print(f"serving http://127.0.0.1:{port}/", flush=True); ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def export_labels():
    sheet = json.load(open(OUT / "sheet" / "sheet.json", encoding="utf-8")); raw = OUT / "labels_raw.jsonl"; L = {}
    if raw.exists():
        for line in raw.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); L[r["region_id"]] = r
    with open(OUT / "labels_colour.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["photo", "region_id", "bbox_x0y0x1y1", "proposed_kind", "class", "user", "at"])
        for r in sheet["regions"]:
            if r["region_id"] in L:
                w.writerow([r["photo_id"], r["region_id"], " ".join(map(str, r["bbox"])), r["kind"], L[r["region_id"]]["cls"], L[r["region_id"]]["user"], L[r["region_id"]]["at"]])
    return len(L)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--port", type=int, default=8840); a = ap.parse_args(argv); OUT.mkdir(exist_ok=True)
    {"select": step_select, "calibrate": step_calibrate, "sheet": step_sheet}.get(a.step, lambda: None)()
    if a.step == "page":
        serve(a.port)
    if a.step == "export":
        print(export_labels(), "labels")


if __name__ == "__main__":
    main()
