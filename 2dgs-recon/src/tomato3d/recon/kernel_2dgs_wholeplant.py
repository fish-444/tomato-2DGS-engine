# Kaggle T4 kernel: 2DGS runs R0 / R2 / R3 of the whole-plant contract, one after another in one job.
# Contract: build/p1_recon_wholeplant_20261022_01/CONTRACT.md. Training loop and losses as the row-31 trainer
# (build/p1_2dgs_full_20261018_01/kernel/run.py) with the declared differences:
#   - alpha target = the run's foreground mask; for R2/R3 that is the whole-plant mask (leaves + stem), one class;
#   - R2/R3 exclude the declared 2 px boundary band from the ALPHA loss only (the RGB terms are unchanged);
#   - world = board metres (z up); held-out views are never rendered during training.
# Results are accumulated per run, so a re-push resumes with the runs already finished.
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
import cv2
from PIL import Image
from scipy.spatial import cKDTree

PROV = dict(gsplat=gsplat.__version__, torch=torch.__version__, trainer_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
    zipfile.ZipFile(z).extractall(f"/tmp/in/{os.path.basename(z)[:-4]}")
R = ["/kaggle/input", "/tmp/in"]
find = lambda pat: sorted(p for r in R for p in glob.glob(f"{r}/**/{pat}", recursive=True))  # noqa: E731
idx = [p for p in find("index.json") if json.load(open(p)).get("schema") == "wholeplant-input-1"]; assert idx, "input dataset not found"
ROOT = os.path.dirname(idx[0]); IDX = json.load(open(idx[0]))
ITERS, N_INIT, SEED = IDX["iters"], IDX["n_init"], IDX["seed"]
L_SSIM, L_ALPHA, L_NORMAL, L_DIST, NORMAL_FROM, DIST_FROM = 0.2, 0.1, 0.05, 100.0, 2000, 1500
LR = dict(means=1.6e-4, scales=5e-3, quats=1e-3, opacities=5e-2, colors=2.5e-3)
PROV["params"] = dict(iters=ITERS, n_init=N_INIT, seed=SEED, l_ssim=L_SSIM, l_alpha=L_ALPHA, l_normal=L_NORMAL, l_dist=L_DIST,
                      normal_from=NORMAL_FROM, dist_from=DIST_FROM, lr=LR)
dev = torch.device("cuda")
RES_FILES = {int(d.split("_")[-1]): d for d in glob.glob(f"{ROOT}/images_*")}


def read_ply(p):
    b = open(p, "rb").read(); i = b.index(b"end_header\n") + 11
    n = int([l for l in b[:i].decode().split("\n") if l.startswith("element vertex")][0].split()[-1])
    r = np.frombuffer(b[i:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return np.c_[r["x"], r["y"], r["z"]].astype(np.float32), np.c_[r["r"], r["g"], r["b"]].astype(np.float32) / 255


def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    g = torch.tensor([math.exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)], device=x.device); g = (g / g.sum())
    w = (g[:, None] * g[None]).expand(3, 1, 11, 11)
    mx = F.conv2d(x, w, padding=5, groups=3); my = F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mx ** 2; syy = F.conv2d(y * y, w, padding=5, groups=3) - my ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mx * my
    return ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))


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
    hdr = ("ply\nformat binary_little_endian 1.0\ncomment 2DGS surfels (gsplat 1.5.3); R = quat(qw,qx,qy,qz); tu,tv,n = columns 0,1,2 of R\n"
           "comment board metres, z up; whole plant, no organ split; shape-only, no accuracy claim\n"
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


def edge_f(gt, pr, t=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1,) * 2)
    gd = cv2.dilate(gt.astype(np.uint8), k) > 0; rd = cv2.dilate(pr.astype(np.uint8), k) > 0
    if not pr.any() or not gt.any():
        return 0.0
    p = (pr & gd).sum() / pr.sum(); r = (gt & rd).sum() / gt.sum()
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def load_view(v, cfg, res):
    im = np.asarray(Image.open(f"{RES_FILES[res]}/{v}").convert("RGB")).astype(np.float32) / 255
    lab = np.asarray(Image.open(f"{ROOT}/{cfg['masks']}/{os.path.splitext(v)[0]}.png"))
    if lab.shape[0] != res:
        lab = cv2.resize(lab, (res, res), interpolation=cv2.INTER_NEAREST)
    fg = np.isin(lab, cfg["fg_labels"]).astype(np.float32)
    stem = (lab == 2)
    if cfg.get("band"):
        bd = np.asarray(Image.open(f"{ROOT}/{cfg['band']}/{os.path.splitext(v)[0]}.png")) > 0
        if bd.shape[0] != res:
            bd = cv2.resize(bd.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0
    else:
        bd = np.zeros((res, res), bool)
    return im * fg[..., None], fg, bd, stem, im


def render(params, K, V, res, **kw):
    return rasterization_2dgs(means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                              opacities=torch.sigmoid(params["opacities"]), colors=torch.sigmoid(params["colors"])[None],
                              viewmats=V[None], Ks=K[None], width=res, height=res, render_mode="RGB+ED",
                              depth_mode="median", packed=False, absgrad=False, **kw)


def run_one(rid, cfg, results):
    if rid in results:
        log("skip", rid); return
    res = cfg["res"]; cams = IDX["cams"][str(res)]
    train, held, held01 = IDX["train"], IDX["heldout_new"], IDX["heldout_run01"]
    train = [v for v in train if v not in held]
    log(rid, cfg["note"], "res", res, "train", len(train), "held-out", len(held))
    imgs, fgs, bds, Ks, Vs = [], [], [], [], []
    for v in train:
        im, fg, bd, _, _ = load_view(v, cfg, res)
        imgs.append(torch.from_numpy(im)); fgs.append(torch.from_numpy(fg)); bds.append(torch.from_numpy((~bd).astype(np.float32)))
        K = torch.tensor(cams[v]["K"], dtype=torch.float32); V = torch.eye(4)
        V[:3, :3] = torch.tensor(cams[v]["R_cw"]); V[:3, 3] = torch.tensor(cams[v]["t_cw"]); Ks.append(K); Vs.append(V)
    imgs = torch.stack(imgs).to(dev); fgs = torch.stack(fgs).to(dev); bds = torch.stack(bds).to(dev)
    Ks = torch.stack(Ks).to(dev); Vs = torch.stack(Vs).to(dev)
    H = float(IDX["scene_scale_m"])
    P1, C1 = read_ply(f"{ROOT}/{cfg['init']}"); P2, C2 = read_ply(f"{ROOT}/{cfg['init_hull']}")
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    P = np.vstack([P1, P2]); C = np.vstack([C1, C2])
    sel = rng.choice(len(P), min(N_INIT, len(P)), replace=False); P, C = P[sel], C[sel]
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
    t0 = time.time(); C_ = len(train)
    for it in range(1, ITERS + 1):
        i = int(rng.integers(C_))
        rc, ra, rn, sn, rd, rm, info = render(params, Ks[i], Vs[i], res, distloss=it >= DIST_FROM)
        rgb = rc[0, ..., :3]; alpha = ra[0, ..., 0]
        strategy.step_pre_backward(params, opts, state, it, info)
        gt = imgs[i]; l1 = (rgb - gt).abs().mean()
        ls = 1 - ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        wa = bds[i]                                           # 1 outside the band, 0 inside it (R0: all ones)
        loss = (1 - L_SSIM) * l1 + L_SSIM * ls + L_ALPHA * (((alpha - fgs[i]).abs() * wa).sum() / wa.sum().clamp_min(1))
        if it >= NORMAL_FROM:
            loss = loss + L_NORMAL * ((1 - (rn[0] * sn[0]).sum(-1)) * fgs[i]).mean()
        if it >= DIST_FROM:
            loss = loss + L_DIST * rd[0, ..., 0].mean()
        loss.backward()
        strategy.step_post_backward(params, opts, state, it, info, packed=False)
        for o in opts.values(): o.step(); o.zero_grad(set_to_none=True)
        if it % 1000 == 0 or it == 1:
            log(rid, it, "loss %.4f l1 %.4f n=%d" % (loss.item(), l1.item(), len(params["means"])), "%.0fs" % (time.time() - t0))
    train_s = time.time() - t0
    with torch.no_grad():
        means = params["means"].cpu().numpy(); qn = F.normalize(params["quats"], dim=-1); Rm = quat_to_rot(qn).cpu().numpy()
        scales = torch.exp(params["scales"])[:, :2].cpu().numpy(); opac = torch.sigmoid(params["opacities"]).cpu().numpy()
        cols = torch.sigmoid(params["colors"]).cpu().numpy()
        write_ply_surfels_full(f"{OUT}/{rid}_surfels_full.ply", means, qn.cpu().numpy(), Rm, scales, opac, cols, params["colors"].cpu().numpy())
        Pts = read_ply(f"{ROOT}/{cfg['init']}")[0]
        metrics, npz, sheet = {}, {}, []
        for v in train + held + [x for x in held01 if x not in train and x not in held]:
            split = "heldout_new" if v in held else ("train" if v in train else "heldout_run01")
            gt_m, fg, bd, stem, full = load_view(v, cfg, res)
            K = torch.tensor(cams[v]["K"], dtype=torch.float32).to(dev); V = torch.eye(4)
            V[:3, :3] = torch.tensor(cams[v]["R_cw"]); V[:3, 3] = torch.tensor(cams[v]["t_cw"]); V = V.to(dev)
            rc, ra, rn, sn, rd, rm, info = render(params, K, V, res)
            r = rc[0, ..., :3].clamp(0, 1); al = ra[0, ..., 0]; dm = rm[0, ..., 0]
            A = fg > 0.5; Ar = (al >= 0.5).cpu().numpy()
            iou = float((A & Ar).sum() / max((A | Ar).sum(), 1))
            nb = ~bd
            iou_nb = float(((A & Ar) & nb).sum() / max(((A | Ar) & nb).sum(), 1))
            ef = edge_f(A, Ar, 2)
            stem_cov = float((stem & Ar).sum() / stem.sum()) if stem.sum() else None
            gtm = torch.from_numpy(gt_m).to(dev); At = torch.from_numpy(A).to(dev)
            mse = float(((r - gtm) ** 2)[At].mean()) if A.any() else float("nan")
            Pt = torch.from_numpy(Pts).to(dev); xc = Pt @ V[:3, :3].T + V[:3, 3]; z = xc[:, 2]
            u = (K[0, 0] * xc[:, 0] / z + K[0, 2]).round().long(); w = (K[1, 1] * xc[:, 1] / z + K[1, 2]).round().long()
            ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res)
            dz = (z[ok] - dm[w[ok], u[ok]])[al[w[ok], u[ok]] >= 0.5]
            metrics[v] = dict(split=split, iou=iou, iou_no_band=iou_nb, edge_f2=ef, stem_coverage=stem_cov,
                              psnr_plant=-10 * math.log10(max(mse, 1e-12)), depth_mad_m=float(dz.abs().median()) if len(dz) else None)
            if split != "train":
                npz[f"a_{v}"] = al.cpu().numpy().astype(np.float16); npz[f"d_{v}"] = dm.cpu().numpy().astype(np.float32)
                Image.fromarray(np.hstack([(full * 255).astype(np.uint8), (r.cpu().numpy() * 255).astype(np.uint8),
                                           np.dstack([(A * 255).astype(np.uint8), (Ar * 255).astype(np.uint8), (stem * 255).astype(np.uint8)])])[::3, ::3]
                                ).save(f"{OUT}/{rid}_{split}_{os.path.splitext(v)[0][9:18]}.jpg", quality=85)
            elif len(sheet) < 24:
                sheet.append((r.cpu().numpy() * 255).astype(np.uint8)[::6, ::6])
        np.savez_compressed(f"{OUT}/{rid}_views.npz", **npz)
        cols_ = 6; sheet += [np.zeros_like(sheet[0])] * ((-len(sheet)) % cols_)
        Image.fromarray(np.vstack([np.hstack(sheet[k:k + cols_]) for k in range(0, len(sheet), cols_)])).save(f"{OUT}/{rid}_train_sheet.jpg", quality=80)

        def agg(split, key):
            xs = [m[key] for m in metrics.values() if m["split"] == split and m[key] is not None]
            return float(np.median(xs)) if xs else None
        summary = dict(run=rid, note=cfg["note"], res=res, n_surfels=int(len(means)), train_seconds=train_s,
                       init_points=int(N), train_views=len(train))
        for sp in ("train", "heldout_new", "heldout_run01"):
            summary[sp] = {k: agg(sp, k) for k in ("iou", "iou_no_band", "edge_f2", "stem_coverage", "psnr_plant", "depth_mad_m")}
            summary[sp]["per_view"] = {v: {k: m[k] for k in ("iou", "iou_no_band", "edge_f2", "stem_coverage")}
                                       for v, m in metrics.items() if m["split"] == sp and sp != "train"}
        summary["gap_iou"] = (summary["heldout_new"]["iou"] - summary["train"]["iou"]) if summary["train"]["iou"] else None
        summary["gap_iou_no_band"] = (summary["heldout_new"]["iou_no_band"] - summary["train"]["iou_no_band"]) if summary["train"]["iou_no_band"] else None
        results[rid] = summary
        json.dump(dict(prov=PROV, runs=results), open(f"{OUT}/results.json", "w"), indent=1)
        log("SUMMARY", rid, json.dumps({k: v for k, v in summary.items() if k not in ("train", "heldout_new", "heldout_run01")}))
        log("  heldout_new", json.dumps({k: v for k, v in summary["heldout_new"].items() if k != "per_view"}))


def main():
    rp = f"{OUT}/results.json"
    results = json.load(open(rp))["runs"] if os.path.exists(rp) else {}
    for rid, cfg in IDX["runs"].items():
        run_one(rid, cfg, results)
    log("ALL DONE", list(results))


try:
    main()
except Exception:
    import traceback
    log(traceback.format_exc()); raise
