"""CONTRACT §4 + A1/A3: synthetic plants -> surfels, labels, clicks -> the same fit path -> errors vs truth.
python synth.py [--smoke]"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_plant as FP  # noqa: E402
from fit_lib import W, BX, resample, point_seg_dist  # noqa: E402
import dynamic_rules as DR  # noqa: E402
import mtg_spec as MS  # noqa: E402
from jrandom import u_of  # noqa: E402
from tomato3d.plant_tree import obj as PO  # noqa: E402

RC = json.load(open(HERE / "synthetic/real_density_noise.json")); RHO, SIG, ASSIGN, WRONG = RC["density_per_mm2"], RC["sigma_mm"], RC["assigned_share"], 0.25
ROWS = list(csv.DictReader(open(FP.ROOT / "build/traits_base_angle_v4_20261102_01/kaggle_out/out/real_click_stats.csv", encoding="utf-8")))
POOL_OFF = [float(x) for r in ROWS for x in json.loads(r["thin_offset"]) if x is not None]          # null = click not near a thin structure
SEED = 20261103; TPL = MS.TPL


def make_plant(m):
    DR.SEED0 = SEED; ib = DR.internode_beta(); sp = DR.drawn_spec(m, ib, "beta"); pid = f"S{m + 1:02d}"; seed = SEED + m
    sp["side_axes"] = []; sp["trusses"] = []; sp["plant_id"] = pid
    amp = [6 * u_of(seed, pid, f"STEM_BEND_A{i}") for i in (1, 2)]; per = [40 + 80 * u_of(seed, pid, f"STEM_BEND_P{i}") for i in (1, 2)]; ph = [2 * math.pi * u_of(seed, pid, f"STEM_BEND_F{i}") for i in (1, 2)]
    ztop = sp["nodes"][-1]["s_mm"]

    def C(z):
        return np.array([amp[0] * (math.sin(2 * math.pi * z / per[0] + ph[0]) - math.sin(ph[0])), amp[1] * (math.sin(2 * math.pi * z / per[1] + ph[1]) - math.sin(ph[1])), z])
    zs = list(np.arange(0, ztop, 1.0)) + [ztop]; true_axis = np.array([C(z) for z in zs])
    for n in sp["nodes"]:
        n["xyz_mm"] = C(n["s_mm"]).tolist()
    for lf in sp["leaves"]:
        lf["base_mm"] = C(sp["nodes"][lf["node_k"] - 1]["s_mm"]).tolist(); lf["geometry"] = "observed"
    sp["kind"] = "observed"; sp["axis"] = dict(curve_mm=true_axis.tolist(), base_mm=[0, 0, 0], source="synthetic truth")
    return sp, W.spec_to_tree(sp), true_axis, seed, pid


def surfels(tree, seed):
    rng = np.random.default_rng(seed); P, leaf = [], []; kmap = {lf["id"]: lf["rank"] for lf in tree["leaves"]}
    for o in PO.organ_objects(tree):
        V = np.asarray(o["V"], float); F = np.asarray(o["F"], int)
        if not len(F):
            continue
        T = V[F]; cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); ar = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / np.maximum(2 * ar[:, None], 1e-12)
        n = rng.poisson(ar * RHO)
        for t, c, nr in zip(T[n > 0], n[n > 0], nrm[n > 0]):
            u = rng.random(c); v = rng.random(c); fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
            P.append(t[0] + u[:, None] * (t[1] - t[0]) + v[:, None] * (t[2] - t[0]) + nr[None] * rng.normal(0, SIG, c)[:, None])
            leaf += [kmap.get(o["leaf"], 0) if o["cat"] != "stem" else 0] * c
    return np.vstack(P), np.asarray(leaf)


def labels(X, true_k, seed):
    rng = np.random.default_rng(seed + 7); lab = np.zeros(len(X), int); leafy = np.nonzero(true_k > 0)[0]
    assigned = leafy[rng.random(len(leafy)) < ASSIGN]; wrong = assigned[rng.random(len(assigned)) < WRONG]; lab[assigned] = true_k[assigned]
    for k in np.unique(true_k[wrong]):
        w = wrong[true_k[wrong] == k]; other = np.nonzero((true_k > 0) & (true_k != k))[0]
        if len(other):
            lab[w] = true_k[other[cKDTree(X[other]).query(X[w])[1]]]
    return lab


def clicks(tree, seed, pid):
    out = {}
    for lf in tree["leaves"]:
        pl = np.asarray(lf["rachis"]["polyline"], float); R, s = resample(pl, 0.25); L = s[-1]; k = lf["rank"]; idk = f"{pid}/L{k}"
        r = ROWS[min(len(ROWS) - 1, int(u_of(seed, idk, "CLICK_ROW") * len(ROWS)))]
        s0 = float(np.clip(float(r["s_along_new"]), 0, 0.5 * L)); n = int(float(r["n_clicks"])); sp = float(r["spacing_median"]); offs = [x for x in json.loads(r["thin_offset"]) if x is not None] or [float(np.median(POOL_OFF))]
        pts = []
        for i in range(n):
            si = s0 + i * sp
            if si > L:
                break
            j = int(np.searchsorted(s, si)); p = R[min(j, len(R) - 1)]; t = R[min(j + 1, len(R) - 1)] - R[max(j - 1, 0)]; t = t / (np.linalg.norm(t) or 1)
            a = 2 * math.pi * u_of(seed, idk, f"CLICK_DIR_{i}"); e1 = np.cross(t, [0, 0, 1.0]); e1 = e1 / (np.linalg.norm(e1) or 1); e2 = np.cross(t, e1)
            pts.append(p + offs[i % len(offs)] * (math.cos(a) * e1 + math.sin(a) * e2))
        out[f"{pid}-L{k:02d}"] = np.asarray(pts if len(pts) >= 2 else [R[0], R[-1]])
    return out


def run(smoke=False):
    res, stems = [], []; m = 0; main = 0
    while (smoke and m < 1) or (not smoke and main < 100):
        sp, tree, true_axis, seed, pid = make_plant(m); X, tk = surfels(tree, seed); lab = labels(X, tk, seed); cl = clicks(tree, seed, pid)
        zz = np.arange(0, true_axis[-1, 2] + 1e-9, 10.0)
        ax = [true_axis[0]]
        for i, z in enumerate(zz[1:], 1):
            a = 2 * math.pi * u_of(seed, pid, f"AXIS_DIR_{i}"); o = POOL_OFF[min(len(POOL_OFF) - 1, int(u_of(seed, pid, f"AXIS_OFF_{i}") * len(POOL_OFF)))]
            ax.append(true_axis[min(int(round(z)), len(true_axis) - 1)] + o * np.array([math.cos(a), math.sin(a), 0.0]))
        ax = np.asarray(ax)
        sk = dict(schema="skeleton_v1.1", plant_id=pid, labels=["synthetic"], leaves=[dict(leaf_uid=u, rank_hint=int(u.split("L")[-1]), status="active", polyline_xyz=(p / 1000).tolist(), n_clicks=len(p), attrs={}) for u, p in cl.items()],
                  axes=[dict(type="main", status="active", fit=dict(curve_mm=ax.tolist()), segments=[dict(s0_mm=0, s1_mm=1e6, provenance="observed")], flags=[])])
        skf = HERE / f"synthetic/{pid}_skeleton.json"; skf.write_text(json.dumps(sk), encoding="utf-8")
        truth = {f"{pid}-L{lf['k']:02d}": lf for lf in sp["leaves"]}
        obs = BX.load_observed(pid, skf, {u: truth[u]["area_mm2"] for u in truth}, "synthetic truth area", age_dag=60)
        dth = TPL.dtheta()
        spec = dict(plant_id=pid, kind="observed", labels=["synthetic"], skeleton_file=str(skf.relative_to(FP.ROOT)).replace("\\", "/"), axis=dict(curve_mm=obs["axis_curve_mm"], base_mm=obs["stem_base_mm"], source=obs["axis_source"]),
                    age_dag=60, age_source="synthetic", flags=obs["flags"], nodes=[], leaves=[], side_axes=[], trusses=[], notes=[])
        for nd, lf in zip(obs["nodes"], obs["leaves"]):
            k = lf["model_rank"]; th = TPL.theta0(k if k <= 9 else "all")
            spec["nodes"].append(dict(k=k, s_mm=nd["s_mm"], xyz_mm=nd["xyz_mm"], internode_mm=nd["internode_mm"], internode_source="synthetic", stem_width_mm=4.0, width_source="synthetic"))
            spec["leaves"].append(dict(k=k, uid=lf["leaf_uid"], node_k=k, base_mm=lf["base_mm"], azimuth_deg=lf["azimuth_deg"], azimuth_source="synthetic click", skeleton_elevation_deg=lf["elevation_deg"], theta0_deg=th[0], theta0_source=th[2],
                                       dtheta_deg=dth[0], dtheta_source=dth[2], length_mm=lf["length_mm"], length_source="synthetic click path", leaflets=TPL.leaflets_mode(k if k <= 9 else "all"), leaflets_source="hand mode",
                                       area_mm2=lf["area_b_mm2"], area_source="truth", flags=lf["flags"], geometry=lf["geometry"]))
        uids = [lf["uid"] for lf in spec["leaves"]]; kof = {u: int(u.split("L")[-1]) for u in uids}
        D = dict(spec=spec, X=X, leaf_idx={u: np.nonzero(lab == kof[u])[0] for u in uids}, leaf_mask=lab > 0, stem_mask=np.zeros(len(X), bool), clicks={u: cl[u] for u in uids}, label_source="synthetic")
        fitted, stem, rows = FP.fit_plant(D, MS.pooled_areas())
        fc = np.asarray(stem["curve"], float)
        stems.append(dict(plant=pid, seed=seed, stem_offset_median_mm=float(np.median(point_seg_dist(resample(fc, 1.0)[0], true_axis))),
                          start_offset_median_mm=float(np.median(point_seg_dist(resample(np.asarray(spec["axis"]["curve_mm"], float), 1.0)[0], true_axis))), flags=";".join(stem["flags"]), candidates=stem["candidates"]))
        for r in rows:
            t = truth[r["uid"]]; r.update(true_azimuth=t["azimuth_deg"], true_theta0=t["theta0_deg"], true_dtheta=t["dtheta_deg"], true_length=t["length_mm"])
            if "azimuth_fit" in r:
                r.update(err_azimuth=abs((r["azimuth_fit"] - t["azimuth_deg"] + 180) % 360 - 180), err_theta0=abs(r["theta0_fit"] - t["theta0_deg"]), err_dtheta=abs(r["dtheta_fit"] - t["dtheta_deg"]),
                         err_length_rel=abs(r["length_fit"] / t["length_mm"] - 1), err_azimuth_start=abs((r["azimuth_prev"] - t["azimuth_deg"] + 180) % 360 - 180), err_theta0_start=abs(r["theta0_prev"] - t["theta0_deg"]))
            r["flags"] = ";".join(r["flags"]); res.append(r)
        main += len(sp["leaves"]); m += 1; print(pid, "leaves", len(sp["leaves"]), "surfels", len(X), flush=True)
    tag = "smoke_discarded" if smoke else "synthetic_check"
    keys = []
    for r in res:
        keys += [k for k in r if k not in keys]
    with open(HERE / f"synthetic/{tag}_leaves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
    with open(HERE / f"synthetic/{tag}_stems.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stems[0])); w.writeheader(); w.writerows(stems)
    fit = [r for r in res if "err_azimuth" in r]
    if not fit:
        print("no leaf optimised"); return
    S = dict(plants=m, main_leaves=main, leaves_optimised=len(fit), gate1_pass=sum(bool(r.get("gate1")) for r in fit), not_fitted=sum(str(r["status"]).startswith("not_fitted") for r in res),
             median_err_azimuth=float(np.median([r["err_azimuth"] for r in fit])), median_err_theta0=float(np.median([r["err_theta0"] for r in fit])), median_err_dtheta=float(np.median([r["err_dtheta"] for r in fit])),
             median_err_length_rel=float(np.median([r["err_length_rel"] for r in fit])), median_stem_offset_mm=float(np.median([s["stem_offset_median_mm"] for s in stems])),
             start_median_err_azimuth=float(np.median([r["err_azimuth_start"] for r in fit])), start_median_err_theta0=float(np.median([r["err_theta0_start"] for r in fit])),
             start_median_stem_offset_mm=float(np.median([s["start_offset_median_mm"] for s in stems])))
    S["acceptable"] = dict(theta0=S["median_err_theta0"] <= 8, dtheta=S["median_err_dtheta"] <= 8, azimuth=S["median_err_azimuth"] <= 10, length=S["median_err_length_rel"] <= 0.10, stem=S["median_stem_offset_mm"] <= 3)
    S["all_acceptable"] = all(S["acceptable"].values())
    json.dump(S, open(HERE / f"synthetic/{tag}_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
