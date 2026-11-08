# Kaggle CPU kernel — 토마토2 Stage 3, part K2 (D10): per-individual pixel classifier fitted on the user's masks of the GAP views
# (training views only), leave-one-gap-view-out check, masks for every accepted view, visual hull from the training views' new masks,
# intersection with the hull projection. Held-out views and their human masks take no part. Nothing runs locally (Rule 0).
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path

IID = "t2_02"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif"], check=True)
import cv2, numpy as np, pillow_heif
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from skimage.filters import sato
pillow_heif.register_heif_opener()
T0 = time.time()
CODE = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/sam2.1_hiera_tiny_leaflet_v5_4plants.pt", recursive=True))))
PHOTO = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/census.csv", recursive=True))))
_hm = glob.glob("/kaggle/input/**/human_masks.json", recursive=True)
if not _hm:                                                                                    # folder uploaded as a zip and not extracted
    import zipfile
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        zipfile.ZipFile(z).extractall("/kaggle/tmp/jobzip")
    _hm = glob.glob("/kaggle/tmp/jobzip/**/human_masks.json", recursive=True)
print("job files", _hm, sorted(glob.glob("/kaggle/input/*/*"))[:20], flush=True)
JOB = os.path.dirname(next(p_ for p_ in _hm if IID in p_))
sys.path.insert(0, CODE)
from tomato3d.recon import charuco_a3 as ca
from tomato3d.recon.charuco_masks import FINE_M, HULL_DILATE_PX, HULL_KEEP, HULL_MIN_VIEWS, MASK_RES, carve, project
from tomato3d.recon.r31_score import rle_decode
from tomato3d.recon.wholeplant_masks import band

R = MASK_RES; N_SAMPLE, P_MIN, MIN_CC, MAX_HOLE, ROI_PAD = 300_000, 0.5, 300, 400, 0.03
con = json.load(open(f"{JOB}/contract.json")); pz = json.load(open(f"{JOB}/poses.json")); human = json.load(open(f"{JOB}/human_masks.json"))
r0 = json.load(open(f"{JOB}/r0_masks_summary.json"))["hull"]
acc = pz["accepted"]; v0 = pz["views"][acc[0]]; K = np.array(v0["K"]); dist = np.array(v0["dist"]); Wd = v0["size"][0]
Km = ca.scaled_K(K, Wd, R); held = set(con["heldout"]); gap = [v for v in con["gap_views"] if v in human]
assert not (set(gap) & held), "a held-out view among the classifier's training views"
W = f"/kaggle/working/{IID}"; os.makedirs(f"{W}/masks", exist_ok=True); os.makedirs(f"{W}/bands", exist_ok=True)


def rgb_of(v):
    im = np.asarray(Image.open(f"{PHOTO}/{IID}/{v[:-4]}.HEIC").convert("RGB"))
    return cv2.resize(cv2.undistort(im, K, dist, None, K), (R, R), interpolation=cv2.INTER_AREA)


def roi(v):
    lo = np.array(r0["lo"]) - ROI_PAD; hi = np.array(r0["hi"]) + ROI_PAD; lo[2] = 0.0
    P = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    u, w, z = project(P, pz["views"][v]["R_cw"], pz["views"][v]["t_cw"], Km); m = np.zeros((R, R), np.uint8)
    if (z > 0).all():
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.c_[u, w]).astype(np.int32)), 1)
    return m > 0


def features(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); f = [lab[..., i] for i in range(3)]
    for s in (3, 9):
        b = cv2.GaussianBlur(lab, (0, 0), s); f += [b[..., i] for i in range(3)]
    L = lab[..., 0]; mu = cv2.GaussianBlur(L, (0, 0), 5); f.append(np.sqrt(np.maximum(cv2.GaussianBlur(L * L, (0, 0), 5) - mu * mu, 0)))
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1); f.append(np.hypot(gx, gy))
    small = cv2.resize(L, (R // 2, R // 2), interpolation=cv2.INTER_AREA) / 255
    f.append(cv2.resize(sato(small, sigmas=[1.5, 3, 5], black_ridges=False).astype(np.float32), (R, R)))
    return np.stack(f, -1)


def clean(p, rmask):
    m = ((p >= P_MIN) & rmask).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.r_[False, st[1:, cv2.CC_STAT_AREA] >= MIN_CC]; m = keep[lab]
    inv = (~m).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    small = np.r_[False, st[1:, cv2.CC_STAT_AREA] < MAX_HOLE]; return m | small[lab]


cache = {}
def feat(v):
    if v not in cache:
        rgb = rgb_of(v); cache[v] = (rgb, features(rgb), roi(v))
    return cache[v]


def fit(views, seed=0):
    rng = np.random.default_rng(seed); X, y = [], []
    for v in views:
        _, F, rm = feat(v); hm = rle_decode(human[v]["mask"], R * R).reshape(R, R).astype(bool)
        hb = rle_decode(human[v].get("band", [R * R]), R * R).reshape(R, R).astype(bool); ok = rm & ~hb
        pos, neg = np.argwhere(ok & hm), np.argwhere(ok & ~hm); k = N_SAMPLE // (2 * len(views))
        for P_, lab in ((pos, 1), (neg, 0)):
            s = P_[rng.choice(len(P_), min(k, len(P_)), replace=False)]; X.append(F[s[:, 0], s[:, 1]]); y.append(np.full(len(s), lab))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed); clf.fit(np.concatenate(X), np.concatenate(y)); return clf


def predict(clf, v):
    _, F, rm = feat(v); p = np.zeros(R * R, np.float32); idx = np.flatnonzero(rm.ravel())
    p[idx] = clf.predict_proba(F.reshape(-1, F.shape[-1])[idx])[:, 1]; return clean(p.reshape(R, R), rm)


# --- leave-one-gap-view-out check (reported, nothing tuned on it)
lovo = {}
for v in gap:
    clf = fit([g for g in gap if g != v]); m = predict(clf, v); hm = rle_decode(human[v]["mask"], R * R).reshape(R, R).astype(bool)
    lovo[v] = dict(iou=round(float((m & hm).sum() / max((m | hm).sum(), 1)), 4), pred_px=int(m.sum()), human_px=int(hm.sum()))
    print("LOVO", v, lovo[v], f"{time.time() - T0:.0f}s", flush=True)
# --- final classifier on all gap views; masks for every accepted view
clf = fit(gap); raw = {}
for v in acc:
    raw[v] = predict(clf, v); cache.pop(v, None) if v not in gap else None
    print("pred", v, int(raw[v].sum()), flush=True)
# --- visual hull from the TRAINING views' new masks, then intersect (R0 rules)
kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HULL_DILATE_PX + 1,) * 2); cams = {v: (pz["views"][v]["R_cw"], pz["views"][v]["t_cw"], Km) for v in acc}
tr = con["train"]; Wb, Hb = ca.SQUARES[0] * ca.SQUARE_M, ca.SQUARES[1] * ca.SQUARE_M
dil = [cv2.dilate(raw[v].astype(np.uint8), kd) for v in tr]
P = carve((0, 0, 0.0), (Wb, Hb, 0.40), 0.004, [cams[v] for v in tr], dil, R); assert len(P), "coarse hull empty"
lo = np.maximum(P.min(0) - 0.012, [0, 0, 0]); hi = np.minimum(P.max(0) + 0.012, [Wb, Hb, 0.40]); Pf = carve(lo, hi, FINE_M, [cams[v] for v in tr], dil, R)
kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)); board = np.array([[0, 0, 0], [Wb, 0, 0], [Wb, Hb, 0], [0, Hb, 0]], np.float32); stats = {}
for v in acc:
    Rm, t, Kk = cams[v]; u, w, z = project(Pf, Rm, t, Kk); ok = (z > 0) & (u >= 0) & (w >= 0) & (u < R) & (w < R)
    hp = np.zeros((R, R), np.uint8); hp[w[ok].astype(int), u[ok].astype(int)] = 1
    r = max(1, int(round(FINE_M * Kk[0, 0] / np.median(z[ok])))) if ok.any() else 1
    hp = cv2.dilate(cv2.dilate(hp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)), kf) > 0
    plant = raw[v] & hp; bu, bv, _ = project(board, Rm, t, Kk); bm = np.zeros((R, R), np.uint8); cv2.fillConvexPoly(bm, np.round(np.c_[bu, bv]).astype(np.int32), 1)
    lab = np.zeros((R, R), np.uint8); lab[bm > 0] = 3; lab[plant] = 1
    Image.fromarray(lab).save(f"{W}/masks/{Path(v).stem}.png"); Image.fromarray(band(plant, 2).astype(np.uint8)).save(f"{W}/bands/{Path(v).stem}.png")
    st = dict(split="heldout" if v in held else "train", gap=v in gap, raw_px=int(raw[v].sum()), plant_px=int(plant.sum()), hull_removed=round(1 - plant.sum() / max(raw[v].sum(), 1), 4))
    if v in human:
        hm = rle_decode(human[v]["mask"], R * R).reshape(R, R).astype(bool); st["iou_vs_user"] = round(float((plant & hm).sum() / max((plant | hm).sum(), 1)), 4)
    stats[v] = st
hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Pf)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
open(f"{W}/visual_hull.ply", "wb").write(hdr.encode() + Pf.astype("<f4").tobytes())
# contact sheet: gap + held-out (secondary) + 8 training views; pink = new mask, cyan outline = user mask where saved
tiles = []
for v in gap + sorted(held) + [x for x in tr if x not in gap][::4][:8]:
    rgb = rgb_of(v).astype(np.float32); lab = np.asarray(Image.open(f"{W}/masks/{Path(v).stem}.png")); o = rgb.copy(); o[lab == 1] = o[lab == 1] * 0.5 + np.array([255, 0, 190]) * 0.5
    if v in human:
        hm = rle_decode(human[v]["mask"], R * R).reshape(R, R).astype(np.uint8); o[band(hm > 0, 2)] = [0, 240, 255]
    t = cv2.resize(o.astype(np.uint8), (378, 378)); cv2.putText(t, f"{v[:-4]} {'H' if v in held else 'G' if v in gap else 'T'} {stats[v].get('iou_vs_user', '')}", (6, 20), 0, 0.5, (255, 255, 0), 2); tiles.append(t)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
Image.fromarray(np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)])).save(f"{W}/k2_sheet.jpg", quality=85)
json.dump(dict(individual=IID, deviation="D10", gap_views_used=gap, lovo=lovo, lovo_median=float(np.median([x["iou"] for x in lovo.values()])),
               hull=dict(carved_from=len(tr), voxels=int(len(Pf)), lo=Pf.min(0).tolist(), hi=Pf.max(0).tolist()), views=stats,
               params=dict(n_sample=N_SAMPLE, p_min=P_MIN, min_cc=MIN_CC, max_hole=MAX_HOLE, roi_pad_m=ROI_PAD), seconds=round(time.time() - T0, 1)),
          open(f"{W}/k2_summary.json", "w"), indent=1)
print("done", f"{time.time() - T0:.0f}s", flush=True)
