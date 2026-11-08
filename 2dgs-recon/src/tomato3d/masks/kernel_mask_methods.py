# Kaggle GPU kernel — mask-method comparison (CONTRACT_MASK_METHOD.md + Amendments 1–2) on t2_02: C1 (M1 + SAM v5 + union + thin parts),
# C2 (SAM 2.1 tiny v5 video propagation from the user's development masks, leave-one-view-out for scoring), C3 (U-Net trained on the
# 4 R3-1 human whole-plant masks). Masks only; held-out views are never prompts or training data. Nothing runs locally (Rule 0).
import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif", "hydra-core", "iopath", "ultralytics", "segmentation-models-pytorch"], check=True)
import cv2, numpy as np, pillow_heif, torch
from PIL import Image
pillow_heif.register_heif_opener()
T0 = time.time(); DEV = "cuda" if torch.cuda.is_available() else "cpu"
def find(pat):
    r = glob.glob(f"/kaggle/input/**/{pat}", recursive=True); assert r, pat; return r[0]
CODE = os.path.dirname(find("sam2.1_hiera_tiny_leaflet_v5_4plants.pt")); PHOTO = os.path.dirname(find("census.csv"))
MI = os.path.dirname(find("M1_s0_best.pt")); JOB = os.path.dirname(find("job/contract.json")); R31 = os.path.dirname(find("r31/human_masks.json"))
sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/sam2_vendor")
CKPT = f"{CODE}/sam2.1_hiera_tiny_leaflet_v5_4plants.pt"
assert hashlib.sha256(open(CKPT, "rb").read()).hexdigest() == "161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d"
assert hashlib.sha256(open(f"{MI}/M1_s0_best.pt", "rb").read()).hexdigest() == "43e78232378092b03bd9118c7c331af845329fcec09b259f275c41f16e84bd3f"
from tomato3d.recon import wholeplant_masks as wm
from tomato3d.recon.r31_score import rle_decode
from tomato3d.recon.r31_maskpage import rle
from tomato3d.recon.wholeplant_contract import ring
print("GPU", torch.cuda.get_device_name(0) if DEV == "cuda" else "none", flush=True)

R, RF = 1512, 3024; W = "/kaggle/working"; TMP = "/kaggle/tmp"
con = json.load(open(f"{JOB}/contract.json")); pz = json.load(open(f"{JOB}/poses.json")); user = json.load(open(f"{JOB}/human_masks.json"))
held = set(con["heldout"]); dev = [v for v in con["gap_views"] if v in user]; acc = pz["accepted"]
assert not held & set(dev)
v0 = pz["views"][acc[0]]; K = np.array(v0["K"]); dist = np.array(v0["dist"])
qc = {r["file"]: r for r in __import__("csv").DictReader(open(glob.glob(f"/kaggle/input/**/job/qc.csv", recursive=True)[0]))} if glob.glob("/kaggle/input/**/job/qc.csv", recursive=True) else {}
meta = {h["file"]: h for h in con["heldout_detail"] + con["gap_detail"]}
def az_el(v):
    c = np.array(pz["views"][v]["center_world"]); ctr = np.array([0.19, 0.133, 0.0]); d = c - ctr
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360), float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
os.makedirs(f"{TMP}/ud", exist_ok=True)
def rgb_full(v):
    p = f"{TMP}/ud/{v}"
    if not os.path.exists(p):
        im = np.asarray(Image.open(f"{PHOTO}/t2_02/{v[:-4]}.HEIC").convert("RGB")); Image.fromarray(cv2.undistort(im, K, dist, None, K)).save(p, quality=95)
    return np.asarray(Image.open(p).convert("RGB"))
for d in ("masks_C1", "masks_C2", "masks_C2_loo", "masks_C3", "leaflet_instances"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
def save(d, v, m):
    Image.fromarray((m.astype(np.uint8))).save(f"{W}/{d}/{Path(v).stem}.png")
log = dict(dev=dev, heldout=sorted(held))

# ------------------------------------------------------------------ C1
from ultralytics import YOLO
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
TILE, STRIDE, IMGSZ, M1_SCORE, TILE_NMS, PAD = 1024, 819, 1024, 0.15, 0.5, 0.15
def _iou_box(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1)
def _crop_iou(a, b):
    x0, y0, x1, y1 = min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]), max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])
    ma = np.zeros((y1 - y0, x1 - x0), bool); mb = ma.copy()
    ma[a["box"][1] - y0:a["box"][3] - y0, a["box"][0] - x0:a["box"][2] - x0] = a["crop"]; mb[b["box"][1] - y0:b["box"][3] - y0, b["box"][0] - x0:b["box"][2] - x0] = b["crop"]
    return (ma & mb).sum() / max((ma | mb).sum(), 1)
def m1_detect(model, rgb):                                   # verbatim from tomato3d/leafseg/hybrid.py::m1_detect (device only)
    H, Wd = rgb.shape[:2]; ys = list(range(0, max(H - TILE, 0) + 1, STRIDE)); xs = list(range(0, max(Wd - TILE, 0) + 1, STRIDE))
    if ys[-1] + TILE < H: ys.append(H - TILE)
    if xs[-1] + TILE < Wd: xs.append(Wd - TILE)
    cands = []
    for y0 in ys:
        for x0 in xs:
            tile = np.ascontiguousarray(rgb[y0:y0 + TILE, x0:x0 + TILE])
            r = model.predict(tile[:, :, ::-1], imgsz=IMGSZ, conf=M1_SCORE, iou=0.7, max_det=300, retina_masks=True, verbose=False, device=0 if DEV == "cuda" else "cpu")[0]
            if r.masks is None or not len(r.masks): continue
            pm = r.masks.data.cpu().numpy() > 0.5; ps = r.boxes.conf.cpu().numpy()
            for m, s in zip(pm, ps):
                ys_, xs_ = np.nonzero(m)
                if not len(xs_): continue
                by0, by1, bx0, bx1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
                cands.append(dict(score=float(s), box=[int(x0 + bx0), int(y0 + by0), int(x0 + bx1), int(y0 + by1)], crop=m[by0:by1, bx0:bx1].copy()))
    cands.sort(key=lambda c: -c["score"]); kept = []
    for c in cands:
        if all(_iou_box(c["box"], k["box"]) == 0 or _crop_iou(c, k) <= TILE_NMS for k in kept): kept.append(c)
    return kept
def solidity(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return 0.0
    hull = cv2.contourArea(cv2.convexHull(np.vstack(cs))); return float(m.sum() / hull) if hull else 0.0
yolo = YOLO(f"{MI}/M1_s0_best.pt"); sam = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", CKPT, device=DEV, apply_postprocessing=False), max_hole_area=0, max_sprinkle_area=0)
class _S: predictor = sam
c1log = {}; rng = np.random.default_rng(0)
for v in acc:
    rgb = rgb_full(v); kept = m1_detect(yolo, rgb); sam.set_image(rgb); union = np.zeros(rgb.shape[:2], bool); insts = []
    for k, c in enumerate(kept):
        x0, y0, x1, y1 = c["box"]; w, h = x1 - x0, y1 - y0; pb = np.array([max(0, x0 - PAD * w), max(0, y0 - PAD * h), min(RF, x1 + PAD * w), min(RF, y1 + PAD * h)], np.float32)
        ms, qs, _ = sam.predict(box=pb, point_coords=None, point_labels=None, multimask_output=True)
        best = max(zip(ms, qs), key=lambda t: float(t[1]) * solidity(t[0] > 0)); m = best[0] > 0
        if not m.any(): continue
        union |= m; small = cv2.resize(m.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST) > 0
        insts.append(dict(id=f"{Path(v).stem}_L{k:03d}", view=v, rle=rle(small), score=round(c["score"], 4), source="H2(M1)-box"))
    plant = union.copy(); added = 0
    if plant.sum() > 500:
        pos = wm.seeds(rgb, plant, rng); ys, xs = np.nonzero(plant); pad = 0.06 * max(np.ptp(xs), np.ptp(ys))
        box = np.array([xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad], np.float32)
        dt = cv2.distanceTransform(plant.astype(np.uint8), cv2.DIST_L2, 5); py, px_ = np.unravel_index(int(np.argmax(dt)), dt.shape)
        pts = np.vstack([[[px_, py]], pos]) if len(pos) else np.array([[px_, py]], np.float32)
        ms, qs, _ = sam.predict(box=box, point_coords=pts.astype(np.float32), point_labels=np.ones(len(pts), np.int32), multimask_output=True)
        b = ms[int(np.argmax(qs))] > 0; lb = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        gate = ((lb[..., 1] - 128) <= wm.ADD_A_MAX) & (lb[..., 0] * 100 / 255 >= wm.ADD_L_MIN)    # rejection term only
        add = wm.thin_parts(b & ~plant & gate, plant); plant |= add; added = int(add.sum())
    save("masks_C1", v, cv2.resize(plant.astype(np.uint8), (R, R), interpolation=cv2.INTER_NEAREST) > 0)
    json.dump(dict(view=v, note="2D leaflet instances (M1 + SAM v5 box), not organs; no compound-leaf membership, rank or attachment", instances=insts),
              open(f"{W}/leaflet_instances/{Path(v).stem}.json", "w"))
    c1log[v] = dict(m1=len(kept), instances=len(insts), union_px=int(union.sum()), thin_added_px=added); print("C1", v, c1log[v], f"{time.time() - T0:.0f}s", flush=True)
log["C1"] = c1log; del yolo; torch.cuda.empty_cache()

# ------------------------------------------------------------------ C2
vp = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", CKPT, device=DEV)
rings = {}
for v in acc:
    az, el = az_el(v); rings.setdefault(ring(el), []).append((az, v))
def user_mask(v):
    return rle_decode(user[v]["mask"], R * R).reshape(R, R) > 0
def propagate(frames, seeds):
    d = f"{TMP}/vid"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for i, v in enumerate(frames):
        Image.fromarray(cv2.resize(rgb_full(v), (R, R), interpolation=cv2.INTER_AREA)).save(f"{d}/{i:05d}.jpg", quality=95)
    st = vp.init_state(video_path=d); out = {}
    for s in seeds:
        vp.add_new_mask(st, frame_idx=frames.index(s), obj_id=1, mask=user_mask(s))
    for rev in (False, True):
        for fi, ids, logits in vp.propagate_in_video(st, reverse=rev):
            m = (logits[0, 0] > 0).cpu().numpy(); out[fi] = out.get(fi, np.zeros_like(m)) | m
    vp.reset_state(st); return {frames[i]: m for i, m in out.items()}
c2log = {}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
    for rg, items in rings.items():
        frames = [v for _, v in sorted(items)]; seeds = [v for v in frames if v in dev]
        c2log[rg] = dict(frames=len(frames), seeds=seeds)
        if not seeds:
            print("C2 ring", rg, "no seed -> no mask", flush=True); continue
        full = propagate(frames, seeds)
        for v, m in full.items():
            save("masks_C2", v, m)
        for d_ in seeds:                                                       # leave-one-view-out for development scoring
            rest = [s for s in seeds if s != d_]
            if not rest:
                c2log[rg].setdefault("loo_missing", []).append(d_); continue
            save("masks_C2_loo", d_, propagate(frames, rest)[d_])
        print("C2 ring", rg, c2log[rg], f"{time.time() - T0:.0f}s", flush=True)
log["C2"] = c2log; del vp; torch.cuda.empty_cache()

# ------------------------------------------------------------------ C3
import segmentation_models_pytorch as smp
torch.manual_seed(0); np.random.seed(0)
h31 = json.load(open(f"{R31}/human_masks.json")); data = []
for v, d in h31.items():
    img = cv2.resize(np.asarray(Image.open(f"{R31}/{v}").convert("RGB")), (R, R), interpolation=cv2.INTER_AREA)
    m = rle_decode(d["mask"], R * R).reshape(R, R).astype(np.float32); ign = rle_decode(d.get("band", [R * R]), R * R).reshape(R, R) > 0
    data.append((img, m, (~ign).astype(np.float32)))
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
    print("C3 epoch", ep, round(tot, 3), f"{time.time() - T0:.0f}s", flush=True)
net.eval()
def infer(img):
    acc_ = np.zeros((R, R), np.float32); cnt = np.zeros((R, R), np.float32); st = 384
    pos = list(range(0, R - C, st)) + [R - C]
    with torch.no_grad():
        for y0 in pos:
            for x0 in pos:
                p = torch.sigmoid(net(norm(img[y0:y0 + C, x0:x0 + C])[None].to(DEV)))[0, 0].cpu().numpy(); acc_[y0:y0 + C, x0:x0 + C] += p; cnt[y0:y0 + C, x0:x0 + C] += 1
    return acc_ / cnt
for v in acc:
    save("masks_C3", v, infer(cv2.resize(rgb_full(v), (R, R), interpolation=cv2.INTER_AREA)) >= 0.5)
log["C3"] = dict(train_images=list(h31), epochs=30); log["seconds"] = round(time.time() - T0, 1)
json.dump(log, open(f"{W}/mask_methods_log.json", "w"), indent=1, default=str)
shutil.rmtree(TMP, ignore_errors=True); print("done", f"{time.time() - T0:.0f}s", flush=True)
