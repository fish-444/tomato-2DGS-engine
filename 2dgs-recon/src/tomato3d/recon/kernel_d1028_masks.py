# Adapted from tomato3d/recon/kernel_all_masks.py for CONTRACT p1_recon_d1028_20261028_01 (individual list and photo folder prefix only; nothing else changed).
# Kaggle GPU kernel — K3a for CONTRACT_ALL.md: ONE kernel builds whole-plant masks for all 11 individuals.
# C3 U-Net (trained once on the 4 R3-1 human whole-plant masks, 30 epochs, CONTRACT_MASK_METHOD.md) is applied to every accepted view,
# intersected with a visual hull carved from the TRAINING views' own C3 masks, then M4 soft matte + 2 px band (R3-1). For t2_02 the C2
# masks (SAM 2 propagation from the user's masks) replace C3 where they exist. Each individual holds out 2 views (seed 20261023).
# Output per individual: <iid>/{masks,bands,soft}/<view>.png (1512), split.json, hull.ply, stats. Nothing runs locally (Rule 0).
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif", "segmentation-models-pytorch"], check=True)
import cv2, numpy as np, pillow_heif, torch
from PIL import Image
pillow_heif.register_heif_opener()
T0 = time.time(); DEV = "cuda" if torch.cuda.is_available() else "cpu"
def find(pat):
    r = glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
    if not r:                                                                                   # folder uploaded as a zip and not extracted
        import zipfile
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            zipfile.ZipFile(z).extractall("/kaggle/tmp/zips")
        r = glob.glob(f"/kaggle/tmp/zips/**/{pat}", recursive=True)
    assert r, pat; print("found", pat, "->", r[0], flush=True); return r[0]
CODE = os.path.dirname(find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt")); JOB = os.path.dirname(find("r31_human_masks.json"))
sys.path.insert(0, CODE)
from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import FINE_M, HULL_DILATE_PX, MASK_RES, carve, project
from tomato3d.recon.r31_masks import matte
from tomato3d.recon.r31_score import rle_decode
from tomato3d.recon.wholeplant_contract import ring
import segmentation_models_pytorch as smp
R = MASK_RES; W = "/kaggle/working"; SEED = 20261023
PHOTO = {}
for ds in glob.glob("/kaggle/input/**/census.csv", recursive=True):
    d = os.path.dirname(ds)
    for sub in os.listdir(d):
        if os.path.isdir(f"{d}/{sub}") and sub.startswith("e_"): PHOTO[sub] = f"{d}/{sub}"
print("photo folders", sorted(PHOTO), flush=True)
IDS = ["e_01", "e_02", "e_03", "e_04", "e_05", "e_06", "e_07"]

# ------------------------------------------------------------------ C3: train once on the 4 R3-1 human masks
h31 = json.load(open(f"{JOB}/r31_human_masks.json")); data = []
for v, d in h31.items():
    img = cv2.resize(np.asarray(Image.open(f"{JOB}/r31/{v}").convert("RGB")), (R, R), interpolation=cv2.INTER_AREA)
    m = rle_decode(d["mask"], R * R).reshape(R, R).astype(np.float32); ign = rle_decode(d.get("band", [R * R]), R * R).reshape(R, R) > 0
    data.append((img, m, (~ign).astype(np.float32)))
torch.manual_seed(0); np.random.seed(0)
net = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
def norm(x): return torch.from_numpy(((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))
rng = np.random.default_rng(0); C = 512
for ep in range(30):
    net.train(); tot = 0
    for _ in range(64 * len(data) // 8):
        xb, yb, wb = [], [], []
        for _ in range(8):
            img, m, w = data[int(rng.integers(len(data)))]; y0, x0 = rng.integers(0, R - C, 2)
            a, b, c_ = img[y0:y0 + C, x0:x0 + C].copy(), m[y0:y0 + C, x0:x0 + C].copy(), w[y0:y0 + C, x0:x0 + C].copy()
            if rng.random() < 0.5: a, b, c_ = a[:, ::-1], b[:, ::-1], c_[:, ::-1]
            if rng.random() < 0.5: a, b, c_ = a[::-1], b[::-1], c_[::-1]
            a = np.clip(a.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
            xb.append(norm(np.ascontiguousarray(a))); yb.append(torch.from_numpy(np.ascontiguousarray(b))[None]); wb.append(torch.from_numpy(np.ascontiguousarray(c_))[None])
        x, y, w = torch.stack(xb).to(DEV), torch.stack(yb).to(DEV), torch.stack(wb).to(DEV)
        lo = net(x); p = torch.sigmoid(lo)
        bce = (torch.nn.functional.binary_cross_entropy_with_logits(lo, y, reduction="none") * w).sum() / w.sum()
        dice = 1 - (2 * (p * y * w).sum() + 1) / ((p * w).sum() + (y * w).sum() + 1); loss = bce + dice
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    if ep % 5 == 4: print("C3 epoch", ep, round(tot, 3), f"{time.time() - T0:.0f}s", flush=True)
net.eval()
def infer(img):
    acc_ = np.zeros((R, R), np.float32); cnt = np.zeros((R, R), np.float32); st = 384; pos = list(range(0, R - C, st)) + [R - C]
    with torch.no_grad():
        for y0 in pos:
            for x0 in pos:
                p = torch.sigmoid(net(norm(img[y0:y0 + C, x0:x0 + C])[None].to(DEV)))[0, 0].cpu().numpy(); acc_[y0:y0 + C, x0:x0 + C] += p; cnt[y0:y0 + C, x0:x0 + C] += 1
    return acc_ / cnt

def az_el(c):
    d = np.array(c) - np.array([0.19, 0.133, 0.0]); return float(np.degrees(np.arctan2(d[1], d[0])) % 360), float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
def pick_heldout(views, pz, seed):
    """2 held-out views: one azimuth in [0,180), one in [180,360); ring preference low then upper (R3-1 ring rule), uniform within the cell."""
    r_ = np.random.default_rng(seed); out = []
    for lo, hi in ((0, 180), (180, 360)):
        cand = [v for v in views if lo <= az_el(pz["views"][v]["center_world"])[0] < hi]
        for want in ("low", "upper", "middle", "top"):
            c = [v for v in cand if ring(az_el(pz["views"][v]["center_world"])[1]) == want]
            if c:
                out.append(sorted(c)[int(r_.integers(len(c)))]); break
    return out

summary = {}
for iid in IDS:
    pz = json.load(open(f"{JOB}/{iid}/poses.json")); acc = pz["accepted"]; held = pick_heldout(acc, pz, SEED)
    if iid == "t2_02":                                                                         # pilot split of CONTRACT_TOMATO2 Amendment 2 (the user's masks cover these)
        held = ["IMG_9357.jpg", "IMG_9310.jpg", "IMG_9352.jpg", "IMG_9353.jpg"]; assert all(v in acc for v in held)
    train = [v for v in acc if v not in held]
    for d in ("masks", "bands", "soft"):
        os.makedirs(f"{W}/{iid}/{d}", exist_ok=True)
    cache, raw = {}, {}
    c2dir = f"{JOB}/{iid}/masks_C2"; c2 = {Path(p).name for p in glob.glob(f"{c2dir}/*.png")} if os.path.isdir(c2dir) else set()
    cams = {}
    for v in acc:
        pv = pz["views"][v]; K = np.array(pv["K"]); dist = np.array(pv["dist"]); Wd = pv["size"][0]
        im = np.asarray(Image.open(f"{PHOTO[iid]}/{v[:-4]}.HEIC").convert("RGB")); assert im.shape[1] == Wd, (iid, v, im.shape, Wd)
        rgb = cv2.resize(cv2.undistort(im, K, dist, None, K), (R, R), interpolation=cv2.INTER_AREA); cache[v] = rgb
        cams[v] = (pv["R_cw"], pv["t_cw"], ca.scaled_K(K, Wd, R))
        m = infer(rgb) >= 0.5
        if f"{Path(v).stem}.png" in c2:                                                      # t2_02: the user's-seed C2 mask where it exists
            a = np.asarray(Image.open(f"{c2dir}/{Path(v).stem}.png")); m = (a > 0) if a.shape[0] == R else cv2.resize((a > 0).astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST) > 0
        raw[v] = m
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_DILATE_PX + 1,) * 2); dil = [cv2.dilate(raw[v].astype(np.uint8), kd) for v in train]
    Wb, Hb = ca.SQUARES[0] * ca.SQUARE_M, ca.SQUARES[1] * ca.SQUARE_M; tc = [cams[v] for v in train]
    P = carve((0, 0, 0.0), (Wb, Hb, 0.40), 0.004, tc, dil, R); assert len(P), f"{iid}: coarse hull empty"
    lo = np.maximum(P.min(0) - 0.012, [0, 0, 0]); hi = np.minimum(P.max(0) + 0.012, [Wb, Hb, 0.40]); Pf = carve(lo, hi, FINE_M, tc, dil, R)
    kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)); stats = {}
    for v in acc:
        Rm, t, Kk = cams[v]; u, w_, z = project(Pf, Rm, t, Kk); ok = (z > 0) & (u >= 0) & (w_ >= 0) & (u < R) & (w_ < R)
        hp = np.zeros((R, R), np.uint8); hp[w_[ok].astype(int), u[ok].astype(int)] = 1; r = max(1, int(round(FINE_M * Kk[0, 0] / np.median(z[ok])))) if ok.any() else 1
        hp = cv2.dilate(cv2.dilate(hp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)), kf) > 0
        plant = raw[v] & hp; alpha, band = matte(cache[v], plant)
        lab = np.zeros((R, R), np.uint8); lab[plant] = 1
        Image.fromarray(lab).save(f"{W}/{iid}/masks/{Path(v).stem}.png"); Image.fromarray(band.astype(np.uint8)).save(f"{W}/{iid}/bands/{Path(v).stem}.png")
        Image.fromarray((alpha * 255).astype(np.uint8)).save(f"{W}/{iid}/soft/{Path(v).stem}.png")
        stats[v] = dict(split="heldout" if v in held else "train", raw_px=int(raw[v].sum()), plant_px=int(plant.sum()), hull_removed=round(1 - plant.sum() / max(raw[v].sum(), 1), 4),
                        source="C2" if f"{Path(v).stem}.png" in c2 else "C3")
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Pf)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    open(f"{W}/{iid}/visual_hull.ply", "wb").write(hdr.encode() + Pf.astype("<f4").tobytes())
    json.dump(dict(individual=iid, heldout=held, train=train, views=stats, hull=dict(carved_from=len(train), voxels=int(len(Pf)), lo=Pf.min(0).tolist(), hi=Pf.max(0).tolist())), open(f"{W}/{iid}/split.json", "w"), indent=1)
    tiles = []
    for v in (held + train[::max(1, len(train) // 10)])[:12]:
        o = cache[v].astype(np.float32); lab = np.asarray(Image.open(f"{W}/{iid}/masks/{Path(v).stem}.png")) == 1
        o[lab] = o[lab] * 0.5 + np.array([255, 0, 190]) * 0.5; t = cv2.resize(o.astype(np.uint8), (300, 300)); cv2.putText(t, f"{iid} {v[:-4]} {'H' if v in held else 'T'}", (4, 18), 0, 0.5, (255, 255, 0), 2); tiles.append(t)
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    Image.fromarray(np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)])).save(f"{W}/{iid}/sheet.jpg", quality=85)
    summary[iid] = dict(accepted=len(acc), heldout=held, hull_voxels=int(len(Pf)), plant_px_median=float(np.median([s["plant_px"] for s in stats.values()])),
                        hull_removed_median=float(np.median([s["hull_removed"] for s in stats.values()])))
    print(iid, summary[iid], f"{time.time() - T0:.0f}s", flush=True); json.dump(summary, open(f"{W}/masks_summary.json", "w"), indent=1)
print("done", f"{time.time() - T0:.0f}s", flush=True)
