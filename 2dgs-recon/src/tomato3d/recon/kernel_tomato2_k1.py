# Kaggle GPU kernel — 토마토2 Stage 3, part K1 for one individual: decode + undistort the accepted views, R0 colour-in-hull masks and
# visual hull (charuco_masks), A2 whole-plant masks with SAM v5 (wholeplant_masks), and the correction-page data (proposals + SAM
# click-add candidates for the 12 page views). Nothing runs locally (Rule 0). Inputs: dolgfish/tomato-recon-tomato2-input (HEIC),
# dolgfish/tomato-recon-code (tomato3d, sam2 vendor, v5 checkpoint, job/<iid>/{contract.json, poses.json, qc.csv}).
import base64, glob, hashlib, io, json, os, shutil, subprocess, sys, time, types
from pathlib import Path

IID = "t2_02"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow-heif", "hydra-core", "iopath"], check=True)
import cv2, numpy as np, pillow_heif, torch
from PIL import Image
pillow_heif.register_heif_opener()
T0 = time.time()
CODE = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/sam2.1_hiera_tiny_leaflet_v5_4plants.pt", recursive=True))))
PHOTO = os.path.dirname(next(iter(glob.glob("/kaggle/input/**/census.csv", recursive=True))))
sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/sam2_vendor")
CKPT, CKPT_SHA = f"{CODE}/sam2.1_hiera_tiny_leaflet_v5_4plants.pt", "161b9c6caaf3a8a31df43e5f1aecc672f02409d2ff4b161515d7cd25bd02e73d"
assert hashlib.sha256(open(CKPT, "rb").read()).hexdigest() == CKPT_SHA, "SAM v5 checkpoint hash differs"
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)

# --- SAM v5 loader injected where the mask code imports it (same weights/prompts as the local loader; device cuda: deviation D8)
class _Seg:
    def __init__(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", CKPT, device=dev, apply_postprocessing=False),
                                            max_hole_area=0, max_sprinkle_area=0)
_SAM = None
def load_sam(_cfg=None):
    global _SAM
    if _SAM is None:
        _SAM = _Seg()
    return _SAM
m1 = types.ModuleType("tomato3d.topview.measure2d"); m1.load_sam = load_sam
m2 = types.ModuleType("tomato3d.leafseg.run"); m2.SAMCFG = Path("kaggle:sam2_v5")
import tomato3d.topview, tomato3d.leafseg                                                      # noqa: E401  (package objects only)
sys.modules["tomato3d.topview.measure2d"] = m1; sys.modules["tomato3d.leafseg.run"] = m2
tomato3d.topview.measure2d = m1; tomato3d.leafseg.run = m2
from tomato3d.recon import charuco_a3 as ca, charuco_masks as cm, wholeplant_masks as wm
from tomato3d.recon.r31_maskpage import candidates, rle
from tomato3d.recon.charuco_masks import MASK_RES

JOB = f"{CODE}/job/{IID}"; W = f"/kaggle/working/{IID}"; SRC = f"/kaggle/tmp/{IID}/src"; R0D = f"/kaggle/tmp/{IID}/r0"; A2D = f"{W}/a2"
for d in (W, SRC, R0D, A2D):
    os.makedirs(d, exist_ok=True)
con = json.load(open(f"{JOB}/contract.json")); pz = json.load(open(f"{JOB}/poses.json"))
acc = pz["accepted"]; v0 = pz["views"][acc[0]]; K = np.array(v0["K"]); dist = np.array(v0["dist"]); Wd, Hd = v0["size"]
assert all(np.allclose(pz["views"][v]["K"], K) for v in acc), "one camera expected for this individual"
RES = Wd                                                                                     # square frame (FRAME_GEOMETRY.md): 3024
# --- A: decode HEIC -> undistort (pinhole K, same size) -> undist_<RES>/<name>.jpg ; poses.json in the charuco_a3 layout
ud = f"{SRC}/undist_{RES}"; os.makedirs(ud, exist_ok=True)
for v in acc:
    p = f"{ud}/{v}"
    if not os.path.exists(p):
        im = np.asarray(Image.open(f"{PHOTO}/{IID}/{v[:-4]}.HEIC").convert("RGB"))
        assert im.shape[:2] == (Hd, Wd), f"{v}: decoded {im.shape[:2]} != {(Hd, Wd)}"
        Image.fromarray(cv2.undistort(im, K, dist, None, K)).save(p, quality=95)
views = {v: dict(R_cw=pz["views"][v]["R_cw"], t_cw=pz["views"][v]["t_cw"], center_world=pz["views"][v]["center_world"], rms_px=pz["views"][v]["rms_px"], accepted=1) for v in acc}
json.dump(dict(frame="board metres, z up (charuco_a3)", full_size=[Wd, Hd], K_full=K.tolist(), dist=dist.tolist(), undistorted_res=RES, K_undist=K.tolist(),
               views=views, accepted=acc), open(f"{SRC}/poses.json", "w"))
shutil.copy(f"{JOB}/qc.csv", f"{SRC}/qc.csv"); print("A undistort", len(acc), f"{time.time() - T0:.0f}s", flush=True)
# --- B: R0 colour-in-hull masks + visual hull (training views only carve)
shutil.copy(f"{JOB}/contract.json", f"{R0D}/contract.json"); cm.main(["--poses", SRC, "--out", R0D]); print("B r0 (a*<-4)", f"{time.time() - T0:.0f}s", flush=True)
# D9: refit A_MAX per individual = Otsu of a* inside the hull projections of the TRAINING views, clamped to [-4, +1]; rerun R0
hdr = open(f"{R0D}/visual_hull.ply", "rb").read(); Pf = np.frombuffer(hdr[hdr.index(b"end_header" + bytes([10])) + 11:], "<f4").reshape(-1, 3)
Km = ca.scaled_K(K, RES, MASK_RES); avals = []
for v in con["train"]:
    u, w_, z = cm.project(Pf, pz["views"][v]["R_cw"], pz["views"][v]["t_cw"], Km); ok = (z > 0) & (u >= 0) & (w_ >= 0) & (u < MASK_RES) & (w_ < MASK_RES)
    hp = np.zeros((MASK_RES, MASK_RES), np.uint8); hp[w_[ok].astype(int), u[ok].astype(int)] = 1; hp = cv2.dilate(hp, np.ones((9, 9), np.uint8)) > 0
    lab = cv2.cvtColor(cv2.resize(np.asarray(Image.open(f"{ud}/{v}").convert("RGB")), (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
    avals.append(lab[..., 1][hp])
av = np.concatenate(avals); thr8, _ = cv2.threshold(av.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
A_FIT = float(np.clip(float(thr8) - 128.0, -4.0, 1.0)); D9 = dict(a_max_default=-4.0, otsu_a=float(thr8) - 128.0, a_max_used=A_FIT, pixels=int(av.size),
                                                                   a_star_pct={str(q): float(np.percentile(av.astype(np.float32) - 128, q)) for q in (5, 25, 50, 75, 95)})
print("D9", D9, flush=True); cm.A_MAX = A_FIT
shutil.rmtree(R0D); os.makedirs(R0D); shutil.copy(f"{JOB}/contract.json", f"{R0D}/contract.json"); cm.main(["--poses", SRC, "--out", R0D]); print("B r0 refit", f"{time.time() - T0:.0f}s", flush=True)
# --- C: A2 whole-plant masks (SAM v5, ridge seeds, thin parts, hull pass 2)
shutil.copy(f"{JOB}/contract.json", f"{A2D}/contract.json"); wm.main(["--poses", SRC, "--run01", R0D, "--out", A2D]); print("C a2", f"{time.time() - T0:.0f}s", flush=True)
shutil.copy(f"{R0D}/masks_summary.json", f"{W}/r0_masks_summary.json"); shutil.copy(f"{R0D}/visual_hull.ply", f"{W}/r0_visual_hull.ply")
shutil.rmtree(f"{A2D}/_raw", ignore_errors=True)
# --- D: correction-page data for the 12 page views (proposal = A2 mask, candidates = SAM v5 grid)
sam = load_sam(); page = {}
for v in con["page_views"]:
    rgb = cv2.resize(np.asarray(Image.open(f"{ud}/{v}").convert("RGB")), (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA)
    plant = np.isin(np.asarray(Image.open(f"{A2D}/masks/{Path(v).stem}.png")), (1, 2)); cands = candidates(sam, rgb, plant)
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=86)
    det = next((h for h in con["heldout_detail"] + con["gap_detail"] if h["file"] == v), {})
    page[v] = dict(jpg=base64.b64encode(buf.getvalue()).decode(), proposal=rle(plant), ring=det.get("ring"), az=det.get("azimuth_deg"), el=det.get("elevation_deg"),
                   role="heldout" if v in con["heldout"] else "gap", candidates=[dict(rle=rle(c["m"]), area=c["area"]) for c in sorted(cands, key=lambda c: c["area"])])
    print("D", v, "proposal px", int(plant.sum()), "candidates", len(cands), flush=True)
json.dump(dict(res=MASK_RES, cand_res=756, views=page, note=f"{IID}: 4 held-out + 8 gap views; proposals = A2 masks"), open(f"{W}/page_data.json", "w"))
# --- E: contact sheet (12 page views + 8 more training views): photo with A2 plant (pink) / added stem (cyan)
tiles = []
for v in con["page_views"] + [x for x in con["train"] if x not in con["page_views"]][::4][:8]:
    rgb = cv2.resize(np.asarray(Image.open(f"{ud}/{v}").convert("RGB")), (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA).astype(np.float32)
    lab = np.asarray(Image.open(f"{A2D}/masks/{Path(v).stem}.png")); o = rgb.copy()
    o[lab == 1] = o[lab == 1] * 0.55 + np.array([255, 0, 190]) * 0.45; o[lab == 2] = o[lab == 2] * 0.2 + np.array([0, 240, 255]) * 0.8
    t = cv2.resize(o.astype(np.uint8), (378, 378)); cv2.putText(t, f"{v[:-4]} {'H' if v in con['heldout'] else 'G' if v in con['gap_views'] else 'T'}", (6, 20), 0, 0.55, (255, 255, 0), 2); tiles.append(t)
while len(tiles) % 5:
    tiles.append(np.zeros_like(tiles[0]))
Image.fromarray(np.vstack([np.hstack(tiles[k:k + 5]) for k in range(0, len(tiles), 5)])).save(f"{W}/a2_sheet.jpg", quality=85)
summ = json.load(open(f"{A2D}/masks_summary.json")); fin = [x for k, x in summ.items() if not k.startswith("_") and "added_px" in x]
json.dump(dict(individual=IID, views=len(acc), res=RES, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", seconds=round(time.time() - T0, 1),
               a2_added_px_median=float(np.median([x["added_px"] for x in fin])), a2_plant_px_median=float(np.median([x["plant_px_final"] for x in fin])),
               hull=summ.get("_hull"), d9=D9), open(f"{W}/k1_summary.json", "w"), indent=1)
shutil.rmtree(f"/kaggle/tmp/{IID}", ignore_errors=True)
print("done", f"{time.time() - T0:.0f}s", flush=True)
