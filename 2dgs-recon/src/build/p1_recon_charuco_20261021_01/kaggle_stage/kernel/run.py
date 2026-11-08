# Kaggle T4 kernel: 2DGS (gsplat 1.5.3 rasterization_2dgs) on the A3 ChArUco capture with FIXED board poses.
# Contract: build/p1_recon_charuco_<date>_01/CONTRACT.md. Training loop and losses as the row-31 trainer
# (build/p1_2dgs_full_20261018_01/kernel/run.py, sha256 eb9a91eb...) with these declared differences:
#   - alpha target = plant foreground only (label 1); board/pot/background are black and alpha 0;
#   - world = board metres (z up); scene scale H = plant hull height; init = triangulated plant points + visual-hull voxels;
#   - held-out views are never rendered during training; after training they are rendered once for the acceptance metrics.
import glob, hashlib, json, math, os, subprocess, sys, time, zipfile
WORK = "/kaggle/working"; OUT = f"{WORK}/out"; os.makedirs(OUT, exist_ok=True)
LOG = open(f"{WORK}/run_log.txt", "a")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def sh(cmd):
    log("$", cmd); r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode: log(r.stdout[-3000:], r.stderr[-3000:]); raise RuntimeError(cmd)
    return r.stdout.strip()


log(sh("nvidia-smi --query-gpu=name,memory.total --format=csv"))
import torch
log("torch", torch.__version__, torch.version.cuda)
sh(f"{sys.executable} -m pip install -q gsplat==1.5.3")
import gsplat
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy
import numpy as np, torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

PROV = dict(gsplat=gsplat.__version__, torch=torch.__version__, trainer_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall(f"/tmp/in/{os.path.basename(z)[:-4]}")
R = ["/kaggle/input", "/tmp/in"]
find = lambda pat: sorted(p for r in R for p in glob.glob(f"{r}/**/{pat}", recursive=True))  # noqa: E731
idx = [p for p in find("index.json") if json.load(open(p)).get("schema") == "recon-charuco-1"]; assert idx, "input dataset not found"
ROOT = os.path.dirname(idx[0]); IDX = json.load(open(idx[0]))
RES, ITERS, N_INIT, SEED = IDX["res"], IDX["iters"], IDX["n_init"], IDX["seed"]
L_SSIM, L_ALPHA, L_NORMAL, L_DIST, NORMAL_FROM, DIST_FROM = 0.2, 0.1, 0.05, 100.0, 2000, 1500
LR = dict(means=1.6e-4, scales=5e-3, quats=1e-3, opacities=5e-2, colors=2.5e-3)
PROV["params"] = dict(res=RES, iters=ITERS, n_init=N_INIT, seed=SEED, l_ssim=L_SSIM, l_alpha=L_ALPHA, l_normal=L_NORMAL, l_dist=L_DIST,
                      normal_from=NORMAL_FROM, dist_from=DIST_FROM, lr=LR, strategy=dict(grow_grad2d=2e-4, refine_start=500, refine_stop=5000,
                      refine_every=100, reset_every=3000, prune_opa=0.05, key_for_gradient="gradient_2dgs"))
dev = torch.device("cuda")


def read_ply(p):
    b = open(p, "rb").read(); i = b.index(b"end_header\n") + 11
    n = int([l for l in b[:i].decode().split("\n") if l.startswith("element vertex")][0].split()[-1])
    r = np.frombuffer(b[i:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return np.c_[r["x"], r["y"], r["z"]].astype(np.float32), np.c_[r["r"], r["g"], r["b"]].astype(np.float32) / 255


def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    g = torch.tensor([math.exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)], device=x.device); g = (g / g.sum())
    w = (g[:, None] * g[None]).expand(3, 1, 11, 11)
    mu_x = F.conv2d(x, w, padding=5, groups=3); mu_y = F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2; syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    return ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))


def ssim(x, y):
    return ssim_map(x, y).mean()


def quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def write_ply_surfels_full(path, means, quats, Rm, scales, opac, cols, fdc):
    n = len(means)
    keys = ("x", "y", "z", "qw", "qx", "qy", "qz", "tux", "tuy", "tuz", "tvx", "tvy", "tvz", "nx", "ny", "nz", "s0", "s1", "opacity")
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment 2DGS surfels (gsplat 1.5.3 rasterization_2dgs); R = quat(qw,qx,qy,qz); tu,tv,n = columns 0,1,2 of R\n"
           "comment s0,s1 = exp(scales[:2]) = std of the 2D Gaussian along tu,tv; opacity = sigmoid; f_dc = trained colour logits, rgb = sigmoid(f_dc), NOT SH\n"
           "comment board metres, z up (A3 ChArUco board frame); shape-only use, no accuracy claim\n"
           f"element vertex {n}\n" + "".join(f"property float {k}\n" for k in keys) +
           "property uchar red\nproperty uchar green\nproperty uchar blue\n" + "".join(f"property float f_dc_{j}\n" for j in range(3)) + "end_header\n").encode()
    rec = np.zeros(n, dtype=[(k, "<f4") for k in keys] + [("r", "u1"), ("g", "u1"), ("b", "u1")] + [(f"f_dc_{j}", "<f4") for j in range(3)])
    for k, a in zip(("x", "y", "z"), means.T): rec[k] = a
    for k, a in zip(("qw", "qx", "qy", "qz"), quats.T): rec[k] = a
    for j, ax in enumerate(("tu", "tv", "n")):
        for c, comp in enumerate("xyz"):
            rec[ax + comp] = Rm[:, c, j]
    rec["s0"], rec["s1"], rec["opacity"] = scales[:, 0], scales[:, 1], opac
    for k, a in zip(("r", "g", "b"), (np.clip(cols, 0, 1) * 255).astype(np.uint8).T): rec[k] = a
    for j in range(3): rec[f"f_dc_{j}"] = fdc[:, j]
    open(path, "wb").write(hdr + rec.tobytes())


def load_view(v):
    im = np.asarray(Image.open(f"{ROOT}/images/{v}").convert("RGB")).astype(np.float32) / 255
    lab = np.asarray(Image.open(f"{ROOT}/masks/{os.path.splitext(v)[0]}.png"))
    assert im.shape[:2] == (RES, RES) and lab.shape == (RES, RES)
    a = (lab == 1).astype(np.float32)
    return im * a[..., None], a, im
cams = IDX["cams"]


def KV(v):
    K = torch.tensor(cams[v]["K"], dtype=torch.float32); V = torch.eye(4); V[:3, :3] = torch.tensor(cams[v]["R_cw"]); V[:3, 3] = torch.tensor(cams[v]["t_cw"])
    return K, V


def render(params, K, V, **kw):
    return rasterization_2dgs(means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]), opacities=torch.sigmoid(params["opacities"]),
                              colors=torch.sigmoid(params["colors"])[None], viewmats=V[None], Ks=K[None], width=RES, height=RES,
                              render_mode="RGB+ED", depth_mode="median", packed=False, absgrad=False, **kw)


def main():
    if os.path.exists(f"{OUT}/done.json"):
        log("done already"); return
    train, held = IDX["train"], IDX["heldout"]; assert not set(train) & set(held)
    imgs, alphas, Ks, Vs = [], [], [], []
    for v in train:
        im, a, _ = load_view(v); imgs.append(torch.from_numpy(im)); alphas.append(torch.from_numpy(a)); K, V = KV(v); Ks.append(K); Vs.append(V)
    imgs = torch.stack(imgs).to(dev); alphas = torch.stack(alphas).to(dev); Ks = torch.stack(Ks).to(dev); Vs = torch.stack(Vs).to(dev)
    H = float(IDX["scene_scale_m"])
    P1, C1 = read_ply(f"{ROOT}/init/plant_points.ply"); P2, C2 = read_ply(f"{ROOT}/init/hull_points.ply")
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    P = np.vstack([P1, P2]); C = np.vstack([C1, C2]); sel = rng.choice(len(P), min(N_INIT, len(P)), replace=False); P, C = P[sel], C[sel]
    d3 = cKDTree(P).query(P, k=4)[0][:, 1:].mean(1); N = len(P)
    params = torch.nn.ParameterDict(dict(
        means=torch.nn.Parameter(torch.from_numpy(P).to(dev)),
        scales=torch.nn.Parameter(torch.log(torch.from_numpy(np.repeat(np.maximum(d3, 1e-5)[:, None], 3, 1).astype(np.float32))).to(dev)),
        quats=torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * N, device=dev)),
        opacities=torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        colors=torch.nn.Parameter(torch.logit(torch.from_numpy(np.clip(C, 0.02, 0.98)).to(dev)))))
    opts = {k: torch.optim.Adam([params[k]], lr=LR[k] * (H if k == "means" else 1.0), eps=1e-15) for k in params}
    strategy = DefaultStrategy(verbose=False, grow_grad2d=2e-4, refine_start_iter=500, refine_stop_iter=5000, refine_every=100,
                               reset_every=3000, prune_opa=0.05, key_for_gradient="gradient_2dgs", absgrad=False)
    strategy.check_sanity(params, opts); state = strategy.initialize_state(scene_scale=H)
    t0 = time.time(); C_ = len(train); log("train views", C_, "held-out", held, "init", N, "H", H)
    for it in range(1, ITERS + 1):
        i = int(rng.integers(C_))
        rc, ra, rn, sn, rd, rm, info = render(params, Ks[i], Vs[i], distloss=it >= DIST_FROM)
        rgb = rc[0, ..., :3]; alpha = ra[0, ..., 0]
        strategy.step_pre_backward(params, opts, state, it, info)
        gt = imgs[i]; l1 = (rgb - gt).abs().mean()
        ls = 1 - ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1 - L_SSIM) * l1 + L_SSIM * ls + L_ALPHA * (alpha - alphas[i]).abs().mean()
        if it >= NORMAL_FROM:
            loss = loss + L_NORMAL * ((1 - (rn[0] * sn[0]).sum(-1)) * alphas[i]).mean()
        if it >= DIST_FROM:
            loss = loss + L_DIST * rd[0, ..., 0].mean()
        loss.backward()
        strategy.step_post_backward(params, opts, state, it, info, packed=False)
        for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
        if it % 500 == 0 or it == 1:
            log(it, "loss %.4f l1 %.4f n=%d" % (loss.item(), l1.item(), len(params["means"])), "%.0fs" % (time.time() - t0),
                "mem %.1fG" % (torch.cuda.max_memory_allocated() / 1e9))
    train_s = time.time() - t0
    with torch.no_grad():
        means = params["means"].cpu().numpy(); qn = F.normalize(params["quats"], dim=-1); Rm = quat_to_rot(qn).cpu().numpy()
        scales = torch.exp(params["scales"])[:, :2].cpu().numpy(); opac = torch.sigmoid(params["opacities"]).cpu().numpy(); cols = torch.sigmoid(params["colors"]).cpu().numpy()
        write_ply_surfels_full(f"{OUT}/surfels_full.ply", means, qn.cpu().numpy(), Rm, scales, opac, cols, params["colors"].cpu().numpy())
        torch.save({k: v.detach().cpu() for k, v in params.items()}, f"{OUT}/params.pt")
        # per-view renders: training views (check) and held-out views (acceptance)
        Pts = read_ply(f"{ROOT}/init/plant_points.ply")[0]
        metrics, depth_npz, sheet = {}, {}, []
        for v in train + held:
            gt_m, a_gt, gt_full = load_view(v); K, V = KV(v); K, V = K.to(dev), V.to(dev)
            rc, ra, rn, sn, rd, rm, info = render(params, K, V)
            r = rc[0, ..., :3].clamp(0, 1); a = ra[0, ..., 0]; dm = rm[0, ..., 0]
            A = torch.from_numpy(a_gt).to(dev) > 0.5; Ar = a >= 0.5
            iou = float((A & Ar).sum() / max(int((A | Ar).sum()), 1))
            gtm = torch.from_numpy(gt_m).to(dev)
            reg = A | Ar; ys, xs = torch.nonzero(reg, as_tuple=True)
            if len(ys):
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                rr, gg = r[y0:y1, x0:x1], gtm[y0:y1, x0:x1]; mm = A[y0:y1, x0:x1]
                mse = float(((rr - gg) ** 2)[mm].mean()) if mm.any() else float("nan")
                sm = ssim_map(rr.permute(2, 0, 1)[None], gg.permute(2, 0, 1)[None])[0].mean(0)
                ss = float(sm[mm].mean()) if mm.any() else float("nan")
            else:
                mse, ss = float("nan"), float("nan")
            psnr = -10 * math.log10(max(mse, 1e-12))
            # depth check: triangulated plant points (training views only) projected into this view; compare their camera
            # depth with the rendered median depth where the render is opaque; robust (median) deviation in metres
            Pt = torch.from_numpy(Pts).to(dev); xc = Pt @ V[:3, :3].T + V[:3, 3]; z = xc[:, 2]
            u = (K[0, 0] * xc[:, 0] / z + K[0, 2]).round().long(); w = (K[1, 1] * xc[:, 1] / z + K[1, 2]).round().long()
            ok = (z > 0) & (u >= 0) & (w >= 0) & (u < RES) & (w < RES)
            dz = (z[ok] - dm[w[ok], u[ok]])[a[w[ok], u[ok]] >= 0.5]
            dev_m = float(dz.abs().median()) if len(dz) else float("nan")
            front = float((dz.abs() <= 0.005).float().mean()) if len(dz) else float("nan")
            metrics[v] = dict(split="heldout" if v in held else "train", silhouette_iou=iou, psnr_plant=psnr, ssim_plant=ss,
                              depth_mad_m=dev_m, depth_points=int(len(dz)), depth_share_within_5mm=front,
                              alpha_gt_px=int(A.sum()), alpha_render_px=int(Ar.sum()))
            depth_npz[f"d_{v}"] = dm.cpu().numpy().astype(np.float32); depth_npz[f"a_{v}"] = a.cpu().numpy().astype(np.float16)
            depth_npz[f"n_{v}"] = rn[0].cpu().numpy().astype(np.float16)
            if v in held:
                Image.fromarray(np.hstack([(gt_full * 255).astype(np.uint8), (r.cpu().numpy() * 255).astype(np.uint8),
                                           np.dstack([(a_gt * 255).astype(np.uint8), (a.cpu().numpy() * 255).astype(np.uint8), np.zeros((RES, RES), np.uint8)])])[::2, ::2]
                                ).save(f"{OUT}/heldout_{os.path.splitext(v)[0]}.jpg", quality=85)
            else:
                sheet.append((r.cpu().numpy() * 255).astype(np.uint8)[::4, ::4])
            log(v, metrics[v])
        np.savez_compressed(f"{OUT}/views.npz", views=np.array(train + held), **depth_npz)
        cols_ = 8; sheet += [np.zeros_like(sheet[0])] * ((-len(sheet)) % cols_)
        Image.fromarray(np.vstack([np.hstack(sheet[k:k + cols_]) for k in range(0, len(sheet), cols_)])).save(f"{OUT}/train_render_sheet.jpg", quality=80)
        hm = [metrics[v] for v in held]
        summary = dict(heldout=dict(views=held, silhouette_iou=[m["silhouette_iou"] for m in hm], silhouette_iou_min=min(m["silhouette_iou"] for m in hm),
                                    psnr_plant=[m["psnr_plant"] for m in hm], ssim_plant=[m["ssim_plant"] for m in hm], depth_mad_m=[m["depth_mad_m"] for m in hm]),
                       train_mean=dict(silhouette_iou=float(np.mean([metrics[v]["silhouette_iou"] for v in train])),
                                       psnr_plant=float(np.mean([metrics[v]["psnr_plant"] for v in train]))))
        json.dump(dict(prov=PROV, n_surfels=int(len(means)), train_seconds=train_s, summary=summary, views=metrics,
                       cameras={v: dict(K=cams[v]["K"], R_cw=cams[v]["R_cw"], t_cw=cams[v]["t_cw"]) for v in train + held}),
                  open(f"{OUT}/done.json", "w"), indent=1)
        log("SUMMARY", json.dumps(summary))


try:
    main()
except Exception:
    import traceback
    log(traceback.format_exc()); raise
