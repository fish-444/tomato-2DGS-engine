# dwt_03 OSS addendum C1: 2D Gaussian Splatting with gsplat 1.5.3 (Apache-2.0), fixed MASt3R-member poses, plant+pot alpha.
# Contract: build/p1_dwt03_oss_20261015_01/CONTRACT.md. Sets: scene_0002 (renders from kernel dense-a) then dwt_03.
# 2026-10-18 re-run (contract build/p1_2dgs_full_20261018_01/CONTRACT.md): EXPORT-ONLY changes after the training loop --
# full surfels with rotation, raw params, cameras, per-view render L1/SSIM. Training code below is unchanged.
import glob, hashlib, json, math, os, subprocess, sys, time, traceback, zipfile
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
PROV = dict(gsplat=gsplat.__version__, torch=torch.__version__, trainer_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
import numpy as np, torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall(f"/tmp/in/{os.path.basename(z)[:-4]}")
R = ["/kaggle/input", "/tmp/in"]
find = lambda pat: sorted(p for r in R for p in glob.glob(f"{r}/**/{pat}", recursive=True))  # noqa: E731
idx = [p for p in find("index.json") if json.load(open(p)).get("schema") == "oss-input-1"]; assert idx
ROOT = os.path.dirname(idx[0])
ren = find("renders_scene_0002_3024.zip"); assert ren, "kernel dense-a renders missing"
zipfile.ZipFile(ren[0]).extractall("/tmp/s3024")
SETS = {"scene_0002": lambda v: f"/tmp/s3024/{v}.png", "dwt_03": lambda v: f"{ROOT}/dwt_03/dwt_03_{v}.jpg"}
# declared hyper-parameters
RES, ITERS, N_INIT = 1512, 7000, 150000
L_SSIM, L_ALPHA, L_NORMAL, L_DIST, NORMAL_FROM, DIST_FROM = 0.2, 0.1, 0.05, 100.0, 2000, 1500
LR = dict(means=1.6e-4, scales=5e-3, quats=1e-3, opacities=5e-2, colors=2.5e-3)
PROV["params"] = dict(res=RES, iters=ITERS, n_init=N_INIT, l_ssim=L_SSIM, l_alpha=L_ALPHA, l_normal=L_NORMAL, l_dist=L_DIST,
                      normal_from=NORMAL_FROM, dist_from=DIST_FROM, lr=LR, strategy=dict(grow_grad2d=2e-4, refine_start=500, refine_stop=5000,
                      refine_every=100, reset_every=3000, prune_opa=0.05, key_for_gradient="gradient_2dgs"))
dev = torch.device("cuda")


def read_ply(p):
    b = open(p, "rb").read(); i = b.index(b"end_header\n") + 11
    n = int([l for l in b[:i].decode().split("\n") if l.startswith("element vertex")][0].split()[-1])
    r = np.frombuffer(b[i:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return np.c_[r["x"], r["y"], r["z"]].astype(np.float32), np.c_[r["r"], r["g"], r["b"]].astype(np.float32) / 255


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    g = torch.tensor([math.exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)], device=dev); g = (g / g.sum())
    w = (g[:, None] * g[None]).expand(3, 1, 11, 11)
    mu_x = F.conv2d(x, w, padding=5, groups=3); mu_y = F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2; syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    return (((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))).mean()


def quat_to_rot(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)


def write_ply_surfels(path, means, normals, scales, opac, cols):
    n = len(means)
    hdr = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n" + "".join(f"property float {k}\n" for k in
           ("x", "y", "z", "nx", "ny", "nz", "s0", "s1", "opacity")) + "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    rec = np.zeros(n, dtype=[(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz", "s0", "s1", "opacity")] + [("r", "u1"), ("g", "u1"), ("b", "u1")])
    for k, a in zip(("x", "y", "z"), means.T): rec[k] = a
    for k, a in zip(("nx", "ny", "nz"), normals.T): rec[k] = a
    rec["s0"], rec["s1"], rec["opacity"] = scales[:, 0], scales[:, 1], opac
    for k, a in zip(("r", "g", "b"), (np.clip(cols, 0, 1) * 255).astype(np.uint8).T): rec[k] = a
    open(path, "wb").write(hdr + rec.tobytes())


def write_ply_surfels_full(path, means, quats, Rm, scales, opac, cols, fdc):
    n = len(means)
    keys = ("x", "y", "z", "qw", "qx", "qy", "qz", "tux", "tuy", "tuz", "tvx", "tvy", "tvz", "nx", "ny", "nz", "s0", "s1", "opacity")
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment 2DGS surfels (gsplat 1.5.3 rasterization_2dgs); R = quat(qw,qx,qy,qz); tu,tv,n = columns 0,1,2 of R\n"
           "comment s0,s1 = exp(scales[:2]) = std of the 2D Gaussian along tu,tv; opacity = sigmoid; f_dc = trained colour logits, rgb = sigmoid(f_dc), NOT SH\n"
           "comment relative R1 VGGT frame units (no metric scale)\n"
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


def run_set(name):
    if os.path.exists(f"{OUT}/{name}_done.json"):
        log("skip", name); return
    cam = json.load(open(f"{ROOT}/{name}_cams.json")); views = cam["views"]
    s = RES / 3024.0
    imgs, alphas, pmasks, Ks, Vs = [], [], [], [], []
    for v in views:
        im = np.asarray(Image.open(SETS[name](v)).convert("RGB").resize((RES, RES), Image.BILINEAR)).astype(np.float32) / 255
        lab = np.asarray(Image.open(f"{ROOT}/masks/{name}/{v}.png").resize((RES, RES), Image.NEAREST))
        a = ((lab == 1) | (lab == 2)).astype(np.float32); im = im * a[..., None]
        imgs.append(torch.from_numpy(im)); alphas.append(torch.from_numpy(a)); pmasks.append(lab == 1)
        c = cam["cams"][v]; K = np.array(c["K"], float).copy(); K[0, 0] *= s; K[1, 1] *= s; K[0, 2] = (K[0, 2] + 0.5) * s - 0.5; K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        V = np.eye(4); V[:3, :3] = np.array(c["R_cw"]); V[:3, 3] = np.array(c["t_cw"])
        Ks.append(torch.tensor(K, dtype=torch.float32)); Vs.append(torch.tensor(V, dtype=torch.float32))
    imgs = torch.stack(imgs).to(dev); alphas = torch.stack(alphas).to(dev); Ks = torch.stack(Ks).to(dev); Vs = torch.stack(Vs).to(dev)
    H = json.load(open(f"{ROOT}/init/{name}_results.json"))["H"]
    P1, C1 = read_ply(f"{ROOT}/init/{name}_plant_r1.ply"); P2, C2 = read_ply(f"{ROOT}/init/{name}_plant_dense_vox.ply")
    P = np.vstack([P1, P2]); C = np.vstack([C1, C2])
    rng = np.random.default_rng(0); sel = rng.choice(len(P), min(N_INIT, len(P)), replace=False); P, C = P[sel], C[sel]
    if name == "scene_0002": C = np.full_like(C, 0.35)
    d3 = cKDTree(P).query(P, k=4)[0][:, 1:].mean(1)
    N = len(P)
    params = torch.nn.ParameterDict(dict(
        means=torch.nn.Parameter(torch.from_numpy(P).to(dev)),
        scales=torch.nn.Parameter(torch.log(torch.from_numpy(np.repeat(d3[:, None], 3, 1).astype(np.float32))).to(dev)),
        quats=torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * N, device=dev)),
        opacities=torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        colors=torch.nn.Parameter(torch.logit(torch.from_numpy(np.clip(C, 0.02, 0.98)).to(dev)))))
    opts = {k: torch.optim.Adam([params[k]], lr=LR[k] * (H if k == "means" else 1.0), eps=1e-15) for k in params}
    strategy = DefaultStrategy(verbose=False, grow_grad2d=2e-4, refine_start_iter=500, refine_stop_iter=5000, refine_every=100,
                               reset_every=3000, prune_opa=0.05, key_for_gradient="gradient_2dgs", absgrad=False)
    strategy.check_sanity(params, opts); state = strategy.initialize_state(scene_scale=H)
    t0 = time.time(); C_ = len(views)
    for it in range(1, ITERS + 1):
        i = int(rng.integers(C_))
        out = rasterization_2dgs(means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]), opacities=torch.sigmoid(params["opacities"]),
                                 colors=torch.sigmoid(params["colors"])[None], viewmats=Vs[i:i + 1], Ks=Ks[i:i + 1], width=RES, height=RES,
                                 render_mode="RGB+ED", distloss=it >= DIST_FROM, depth_mode="median", packed=False, absgrad=False)
        rc, ra, rn, sn, rd, rm, info = out
        rgb = rc[0, ..., :3]; alpha = ra[0, ..., 0]
        strategy.step_pre_backward(params, opts, state, it, info)
        gt = imgs[i]
        l1 = (rgb - gt).abs().mean()
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
            log(name, it, "loss %.4f l1 %.4f n=%d" % (loss.item(), l1.item(), len(params["means"])), "%.0fs" % (time.time() - t0),
                "mem %.1fG" % (torch.cuda.max_memory_allocated() / 1e9))
    train_s = time.time() - t0
    # ---- export: surfels
    with torch.no_grad():
        means = params["means"].cpu().numpy(); Rm = quat_to_rot(F.normalize(params["quats"], dim=-1)); normals = Rm[:, :, 2].cpu().numpy()
        scales = torch.exp(params["scales"])[:, :2].cpu().numpy(); opac = torch.sigmoid(params["opacities"]).cpu().numpy(); cols = torch.sigmoid(params["colors"]).cpu().numpy()
        write_ply_surfels(f"{OUT}/{name}_surfels.ply", means, normals, scales, opac, cols)
        # ---- 2026-10-18 export additions (no effect on training)
        qn = F.normalize(params["quats"], dim=-1).cpu().numpy(); Rn = Rm.cpu().numpy()
        write_ply_surfels_full(f"{OUT}/{name}_surfels_full.ply", means, qn, Rn, scales, opac, cols, params["colors"].cpu().numpy())
        torch.save({k: v.detach().cpu() for k, v in params.items()}, f"{OUT}/{name}_params.pt")
        json.dump(dict(frame="R1 VGGT frame (relative units)", res=RES, convention="x_cam = viewmat[:3,:3] @ x_world + viewmat[:3,3]; pixel = K @ x_cam / z",
                       views={v: dict(K_1512=Ks[i].cpu().numpy().tolist(), viewmat=Vs[i].cpu().numpy().tolist(), K_3024=cam["cams"][v]["K"],
                                      R_cw=cam["cams"][v]["R_cw"], t_cw=cam["cams"][v]["t_cw"]) for i, v in enumerate(views)}),
                  open(f"{OUT}/{name}_cameras.json", "w"), indent=1)
        view_metrics = {}
        # per-view median depth, alpha, normals; visibility of surfel centres (for C2) at RES
        depth, vis = {}, np.zeros((C_, len(means)), bool)
        rgbs = []
        for i, v in enumerate(views):
            rc, ra, rn, sn, rd, rm, info = rasterization_2dgs(means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                                                               opacities=torch.sigmoid(params["opacities"]), colors=torch.sigmoid(params["colors"])[None],
                                                               viewmats=Vs[i:i + 1], Ks=Ks[i:i + 1], width=RES, height=RES, render_mode="RGB+ED", depth_mode="median")
            dm = rm[0, ..., 0].cpu().numpy(); a = ra[0, ..., 0].cpu().numpy()
            depth[v] = dict(median=dm.astype(np.float32), alpha=a.astype(np.float32), normal=rn[0].cpu().numpy().astype(np.float16))
            rgbs.append((rc[0, ..., :3].cpu().numpy() * 255).astype(np.uint8))
            _r = rc[0, ..., :3].clamp(0, 1); view_metrics[v] = dict(l1=float((_r - imgs[i]).abs().mean()), ssim=float(ssim(_r.permute(2, 0, 1)[None], imgs[i].permute(2, 0, 1)[None])))
            xc = (Vs[i, :3, :3] @ params["means"].T).T + Vs[i, :3, 3]; z = xc[:, 2]
            u = (Ks[i, 0, 0] * xc[:, 0] / z + Ks[i, 0, 2]).round().long(); w = (Ks[i, 1, 1] * xc[:, 1] / z + Ks[i, 1, 2]).round().long()
            ok = (z > 0) & (u >= 0) & (w >= 0) & (u < RES) & (w < RES)
            dmt = torch.from_numpy(dm).to(dev); at = torch.from_numpy(a).to(dev)
            dz = dmt[w[ok], u[ok]]; az = at[w[ok], u[ok]]
            good = (az >= 0.5) & ((z[ok] - dz).abs() / dz.clamp_min(1e-9) <= 0.02)
            vis[i, ok.cpu().numpy()] = good.cpu().numpy()
        np.savez_compressed(f"{OUT}/{name}_views.npz", views=np.array(views), **{f"d_{v}": depth[v]["median"] for v in views},
                            **{f"a_{v}": depth[v]["alpha"] for v in views}, **{f"n_{v}": depth[v]["normal"] for v in views}, vis=vis)
        sheet = np.vstack([np.hstack(rgbs[k:k + 6] + [np.zeros_like(rgbs[0])] * (6 - len(rgbs[k:k + 6]))) for k in range(0, len(rgbs), 6)])
        Image.fromarray(sheet[::4, ::4]).save(f"{OUT}/{name}_render_sheet.jpg", quality=80)
        # masked median depth -> points per view, fused by >= 2 other views within 1 %
        pts, src = [], []
        for i, v in enumerate(views):
            dm, a = depth[v]["median"], depth[v]["alpha"]; m = pmasks[i] & (a >= 0.5) & (dm > 0)
            rows, cols_ = np.nonzero(m); K = Ks[i].cpu().numpy(); V = Vs[i].cpu().numpy()
            xc = (np.linalg.inv(K) @ np.vstack([cols_, rows, np.ones_like(cols_)])) * dm[rows, cols_]
            pts.append((V[:3, :3].T @ (xc - V[:3, 3:4])).T); src.append(np.full(len(rows), v))
        Pall = np.vstack(pts).astype(np.float64); Sall = np.concatenate(src)
        agree = np.zeros(len(Pall), int)
        for i, v in enumerate(views):
            K = Ks[i].cpu().numpy(); V = Vs[i].cpu().numpy(); xc = Pall @ V[:3, :3].T + V[:3, 3]; z = xc[:, 2]
            with np.errstate(all="ignore"):
                u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
            ok = (z > 0) & (u >= 0) & (w >= 0) & (u < RES) & (w < RES) & (Sall != v)
            dm = depth[v]["median"]; a = depth[v]["alpha"]
            dz = dm[w[ok], u[ok]]; az = a[w[ok], u[ok]]
            agree[np.nonzero(ok)[0][(az >= 0.5) & (dz > 0) & (np.abs(z[ok] - dz) / np.maximum(dz, 1e-9) <= 0.01)]] += 1
        keep = agree >= 2
        np.savez_compressed(f"{OUT}/{name}_fused.npz", xyz=Pall[keep].astype(np.float32), ref_view=Sall[keep], candidates=len(Pall))
    json.dump(dict(set=name, views=views, n_surfels=int(len(means)), train_seconds=round(train_s, 1), peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                   fused_points=int(keep.sum()), fused_candidates=int(len(Pall)), render_metrics_1512=view_metrics), open(f"{OUT}/{name}_done.json", "w"), indent=1)
    log(name, "done surfels", len(means), "fused", int(keep.sum()), "/", len(Pall), "train %.0fs" % train_s)
    torch.cuda.empty_cache()


for name in ["scene_0002", "dwt_03"]:
    try:
        run_set(name)
    except Exception as e:
        log("FAIL", name, repr(e), traceback.format_exc()[-3000:]); json.dump(dict(set=name, error=repr(e)), open(f"{OUT}/{name}_error.json", "w"))
    json.dump(PROV, open(f"{OUT}/provenance.json", "w"), indent=1)
log("ALL DONE")
