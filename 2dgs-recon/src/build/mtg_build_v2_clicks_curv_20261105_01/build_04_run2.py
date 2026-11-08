"""Set _04 (leaflet surfaces = quadric fits of each plant's own surfels, area (b)) + reconstruction check + leaflet curvature table.
python -X utf8 build_04.py"""
import copy, csv, hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; LEAF = ROOT / "build/mtg_build_v2_clicks_leaflets_20261105_01"
V03 = ROOT / "build/mtg_build_v2_clicks_20261105_03"; OUT = ROOT / "build/mtg_build_v2_clicks_20261105_04"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(LEAF))
import curv_lib as CL  # noqa: E402
import run_split as RS  # noqa: E402  (surfel loader, writers; read only)
from run_split import PE, PO, W, RR, validate  # noqa: E402

PLANTS = RS.PLANTS
LABELS = ["leaf_curvature_from_reconstruction", "mtg_follows_reconstruction", "skeleton_labels_ungraded", "conditional_on_unvalidated_scale", "not_validation", "split_amendment5_count_fixed_to_prior"]
DENS = 9.73; SIG = 0.124; DENSE = 40.0
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ARR = {(r["plant"], r["uid"], r["leaflet"]): r for r in csv.DictReader(open(LEAF / "leaflets.csv", encoding="utf-8"))}


def ndev_of(P, Nn, A=None):
    A = np.ones(len(P)) if A is None else A
    Ns, _ = CL.smooth_normals(P, Nn); nb = (Ns * A[:, None]).sum(0); nb /= np.linalg.norm(nb) or 1
    return float(np.median(np.degrees(np.arccos(np.clip(Ns @ nb, -1, 1))))), float(np.percentile(np.degrees(np.arccos(np.clip(Ns @ nb, -1, 1))), 90))


def run_plant(pid, rng):
    D = RS.FP.load_real(pid); X, A, N, rows, _ = RS.surfel_data(pid, D); S = np.load(LEAF / f"split_{pid}.npz", allow_pickle=True)
    assert np.array_equal(np.asarray(S["rows"]), rows), pid
    leaf_of = S["leaf"].astype(str); lt_of = S["leaflet"].astype(str)
    tree = json.load(open(V03 / f"static/mtg_{pid}_engine.plant_tree_v1.json", encoding="utf-8")); spec = json.load(open(V03 / f"static/mtg_{pid}_spec.json", encoding="utf-8"))
    H = float(spec["height_mm"]); lid_of = {lf["attrs"].get("uid"): lf["id"] for lf in tree["leaves"]}; lts = {t["id"]: t for t in tree["leaflets"]}
    curv_rows, leaf_rows = [], []; surf_scaled, surf_unscaled, pts_idx = [], [], []
    for u in sorted(set(leaf_of) - {""}):
        lid = lid_of[u]; tot = 0.0; target = 0.0
        for code in sorted(set(lt_of[leaf_of == u]) - {""}):
            m = np.nonzero((leaf_of == u) & (lt_of == code))[0]; t = lts[f"{lid}__{code}"]
            if t["surface"]["kind"] != "mesh":
                continue
            V0 = t["surface"]["vertices"]; F = t["surface"]["faces"]; f = CL.fit(X[m], A[m], N[m], t["base"])
            ar = ARR.get((pid, u, code), {})
            row = dict(plant=pid, uid=u, k=int(ar.get("k", 0)), leaflet=code, position="T" if code == "T" else code[:2], side=t["side"], rank_class="main", area_b_mm2=t["area_mm2"])
            if f is None:      # amendment 1: flat alpha-mesh in its PCA plane, scaled about the base to area (b); counted in area / distance only
                Vf = np.asarray(V0, float); c0 = Vf.mean(0); w_, v_ = np.linalg.eigh(np.cov((Vf - c0).T)); pn = v_[:, 0]; Vf = Vf - np.outer((Vf - c0) @ pn, pn)
                b0 = np.asarray(t["base"]); lo, hi = 0.05, 3.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi); Vt = b0 + mid * (Vf - b0)
                    lo, hi = (mid, hi) if CL.mesh_area(Vt, F) < t["area_mm2"] else (lo, mid)
                k = 0.5 * (lo + hi); Vs = b0 + k * (Vf - b0)
                t["surface"] = dict(kind="mesh", vertices=np.round(Vs, 4).tolist(), faces=F); t["attrs"].update(outline_scale_k=k, split_flags=list(t["attrs"].get("split_flags", [])) + ["too_few_surfels_flat"])
                row.update(status="too_few_surfels_flat", scale_k=k, curved_area_mm2=CL.mesh_area(Vs, F)); curv_rows.append(row)
                tot += CL.mesh_area(Vs, F); target += t["area_mm2"]; surf_scaled.append((Vs, np.asarray(F), code, m, False)); surf_unscaled.append((np.asarray(V0, float), np.asarray(F), code, m, False)); continue
            Vs, k = CL.quadric_surface(V0, F, f, t["base"], target_area=t["area_mm2"]); Vu, _ = CL.quadric_surface(V0, F, f, t["base"])
            pr = CL.proj_ratio(V0, F)
            row.update(status="fitted", **{q: f[q] for q in ("kappa_l", "kappa_w", "twist_c", "r2", "n", "length_mm", "width_mm", "dalpha_l_deg", "dalpha_w_deg", "ndev_median_deg", "ndev_p90_deg", "rms_residual_mm")},
                       proj_ratio=pr, scale_k=k, alpha_mesh_area_mm2=CL.mesh_area(V0, F), curved_area_mm2=CL.mesh_area(Vs, F))
            curv_rows.append(row)
            # node values + surface
            t["surface"] = dict(kind="mesh", vertices=np.round(Vs, 4).tolist(), faces=F)
            t["attrs"].update(curvature=dict(kappa_l=f["kappa_l"], kappa_w=f["kappa_w"], twist_c=f["twist_c"], r2=f["r2"], a0=f["a0"], a1=f["a1"], a2=f["a2"], dalpha_l_deg=f["dalpha_l_deg"], dalpha_w_deg=f["dalpha_w_deg"]),
                              leaflet_frame=dict(centroid=f["centroid"], e_s=f["e_s"], e_d=f["e_d"], n_bar=f["n_bar"]), outline_scale_k=k, surface_rule="alpha-mesh (s,d) in the leaflet frame x k about the base, z = fitted quadric; area = area (b)",
                              spread_phi_deg=float(ar["spread_phi_deg"]) if ar else None, tilt_deg=float(ar["tilt_deg"]) if ar else None, roll_sign=int(float(ar["roll_sign"])) if ar else None,
                              petiolule_offset_mm=float(ar["petiolule_offset_mm"]) if ar else None, elev_deg=float(ar["elev_deg"]) if ar else None, az_deg=float(ar["az_deg"]) if ar else None,
                              split_flags=list(t["attrs"].get("split_flags", [])) + ["leaflet_geometry_from_own_surfels"])
            t["normal"] = f["n_bar"]
            tot += CL.mesh_area(Vs, F); target += t["area_mm2"]
            surf_scaled.append((Vs, np.asarray(F), code, m, True)); surf_unscaled.append((Vu, np.asarray(F), code, m, True))
        sp_area = next(lf["area_mm2"] for lf in spec["leaves"] if lf["uid"] == u)
        leaf_rows.append(dict(plant=pid, uid=u, area_b_mm2=sp_area, mtg_leaflet_surface_mm2=tot, area_rel_err=(tot - sp_area) / sp_area if sp_area else float("nan")))
    tree["plant"]["attrs"]["labels"] = list(dict.fromkeys(tree["plant"]["attrs"]["labels"] + LABELS))
    tree["plant"]["attrs"]["leaflet_geometry"] = "build/mtg_build_v2_clicks_curv_20261105_01: leaflet surfaces = quadric fits of each leaflet's own surfels over its alpha outline, area (b)"
    tree = validate(tree)
    # checks
    res = {}
    for nm, surfs in (("areab", surf_scaled), ("unscaled", surf_unscaled)):
        P_all, NN_all, own_all = [], [], []
        dense = []
        for j, (V, F, code, m, fitted) in enumerate(surfs):
            p, n = CL.sample_mesh(V, F, DENSE, 0.0, rng); dense.append(p)
            q, qn = CL.sample_mesh(V, F, DENS, SIG, rng); P_all.append(q); NN_all.append(qn); own_all.append(np.full(len(q), j))
        Dn = np.vstack(dense); Pm = np.vstack(P_all); own_m = np.concatenate(own_all)
        sidx = np.concatenate([m for _, _, _, m, _ in surfs]); Xs = X[sidx]
        d1 = cKDTree(Dn).query(Xs)[0]; d2 = cKDTree(Xs).query(Pm)[0]
        diffs = []
        for j, (V, F, code, m, fitted) in enumerate(surfs):
            mm = own_m == j
            if not fitted or mm.sum() < CL.MIN_N or len(m) < CL.MIN_N:
                continue
            a, ap = ndev_of(Pm[mm], NN_all[j]); b, bp = ndev_of(X[m], N[m], A[m]); diffs.append((a - b, ap - bp))
        diffs = np.array(diffs)
        res[nm] = dict(p50_s2m_pctH=100 * float(np.median(d1)) / H, p95_s2m_pctH=100 * float(np.percentile(d1, 95)) / H, p50_m2s_pctH=100 * float(np.median(d2)) / H, p95_m2s_pctH=100 * float(np.percentile(d2, 95)) / H,
                       p50_s2m_mm=float(np.median(d1)), ndev_median_diff_deg=float(np.median(diffs[:, 0])), ndev_p90_diff_deg=float(np.median(diffs[:, 1])), n_leaflets=len(diffs))
    worst_area = max(abs(r["area_rel_err"]) for r in leaf_rows)
    a = res["areab"]; passed = dict(area=worst_area <= 0.05, distance=a["p50_s2m_pctH"] <= 1.5, normals=abs(a["ndev_median_diff_deg"]) <= 5.0)
    lab_idx = np.nonzero(leaf_of != "")[0]; H_s = float(X[lab_idx, 2].max() - X[lab_idx, 2].min())
    prow = dict(plant=pid, H_mm=H, H_surfel_mm=H_s, p50_s2m_pct_Hsurfel_report=100 * a["p50_s2m_mm"] / H_s, pass_distance_Hsurfel_report=100 * a["p50_s2m_mm"] / H_s <= 1.5, leaves_checked=len(leaf_rows), worst_leaf_area_rel_err=worst_area, **{f"{k}": v for k, v in a.items()}, **{f"unscaled_{k}": v for k, v in res["unscaled"].items()},
                pass_area=passed["area"], pass_distance=passed["distance"], pass_normals=passed["normals"], passed=all(passed.values()))
    return tree, spec, curv_rows, leaf_rows, prow, X, surf_scaled


def write_set(spec, tree, pid):
    d = OUT / "static"; name = f"mtg_{pid}"
    (d / f"{name}_engine.plant_tree_v1.json").write_text(json.dumps(tree, indent=1, default=float), encoding="utf-8")
    PE.to_engine_structure(tree, d / f"{name}_engine", plant_instance_id="p1", source_plant_id=f"mtg_{pid}", fixture=False, capture_id=f"mtg_build_{pid}")
    (d / f"{name}_engine.obj").write_text(PO.to_obj(tree, header=f"mtg static _04 {pid} (leaflet surfaces from own surfels, curved); mm, z up"), encoding="utf-8")
    bsha, info = W.spec_to_bmodel(spec, d / f"{name}_bmodel.csv")
    return RR.engine_round_trip(tree, d / f"{name}_engine"), bsha, info


def overlay(pid, X, surfs, H):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5)); allX = X; c = allX[:, :2].mean(0); cols = plt.cm.tab20.colors
    for a, (i, j, lab) in zip(ax, ((0, 1, "top view (x, y)"), (0, 2, "side view (x, z)"))):
        a.scatter(allX[:, i] - (c[0] if i == 0 else 0), allX[:, j] - (c[1] if j == 1 else 0), s=0.2, c="0.75", label="surfels (all)")
        for q, (V, F, code, m, _f) in enumerate(surfs):
            P = V[:, [i, j]] - np.array([c[0] if i == 0 else 0, c[1] if j == 1 else 0])
            a.triplot(P[:, 0], P[:, 1], F, color=cols[q % 20], lw=0.15)
        a.set_aspect("equal"); a.set_title(f"{pid} {lab}: MTG _04 leaflet surfaces over surfels (mm, plant-centred)"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / f"figures/overlay_{pid}.png", dpi=90); plt.close(fig)


def stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0, mean="", sd="", median="", min="", p1="", p10="", p90="", p99="", max="")
    p = np.percentile(v, [1, 10, 50, 90, 99])
    return dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else 0.0, median=p[2], min=v.min(), p1=p[0], p10=p[1], p90=p[3], p99=p[4], max=v.max())


def main():
    (OUT / "static").mkdir(parents=True, exist_ok=True); (HERE / "figures").mkdir(exist_ok=True); man = dict(copied={})
    for p in sorted(V03.rglob("*")):
        rel = p.relative_to(V03)
        if p.is_dir() or "__pycache__" in p.parts or rel.name in ("FINAL_HASHES.json", "MANIFEST.json", "WITHHELD_README_amendment3_superseded.txt", "roundtrip_engine_03.csv"):
            continue
        if rel.parts[0] == "static" and any(f"mtg_{q}_" in rel.name or rel.parts[1:2] == (f"mtg_{q}_engine",) for q in PLANTS):
            continue
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, OUT / rel); man["copied"][str(rel).replace("\\", "/")] = sha(p) == sha(OUT / rel)
    rng = np.random.default_rng(0); C, LR, PR, RT = [], [], [], []
    for pid in PLANTS:
        tree, spec, cr, lr, prow, X, surfs = run_plant(pid, rng)
        spec4 = copy.deepcopy(spec); spec4["labels"] = list(dict.fromkeys(spec4["labels"] + LABELS)); spec4["mtg_set"] = dict(version="static_v2_click_skeleton_20261105_04", source_spec=f"build/mtg_build_v2_clicks_20261105_03/static/mtg_{pid}_spec.json", curvature="build/mtg_build_v2_clicks_curv_20261105_01")
        json.dump(spec4, open(OUT / f"static/mtg_{pid}_spec.json", "w", encoding="utf-8"), indent=1, default=float)
        rt, bsha, info = write_set(spec4, tree, pid); same = sha(V03 / f"static/mtg_{pid}_bmodel.csv") == sha(OUT / f"static/mtg_{pid}_bmodel.csv")
        prow.update(engine_roundtrip=rt["passed"], bmodel_identical_to_03=same); overlay(pid, X, surfs, prow["H_mm"])
        C += cr; LR += lr; PR.append(prow); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in prow.items() if not k.startswith("unscaled")}, flush=True)
    for fn, data in (("leaflet_curvature.csv", C), ("mtg_geometry_check_leaves.csv", LR), ("mtg_geometry_check.csv", PR)):
        keys = []
        for r in data:
            keys += [k for k in r if k not in keys]
        with open(HERE / fn, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)
    shutil.copy2(HERE / "mtg_geometry_check.csv", OUT / "mtg_geometry_check.csv")
    # curvature traits file
    NF = list(csv.DictReader(open(HERE / "noise_floor.csv", encoding="utf-8")))
    Q = ["kappa_l", "kappa_w", "twist_c", "dalpha_l_deg", "dalpha_w_deg", "r2", "ndev_median_deg", "ndev_p90_deg", "proj_ratio"]
    rows = []; fit = [r for r in C if r["status"] == "fitted"]
    for subset, src, sel in (("all_11", "reconstruction_ungraded", lambda r: True), ("pilot_only", "reconstruction_ungraded_pilot_only", lambda r: r["plant"] == "R3-1b"), ("noise_floor", "synthetic_engine_flat_leaflets", None)):
        for pos in ("P1", "P2", "P3", "T"):
            for q in Q:
                vals = [float(r[q]) for r in NF if r["position"] == pos] if sel is None else [float(r[q]) for r in fit if sel(r) and r["position"] == pos]
                rows.append(dict(subset=subset, rank_class="main", position=pos, quantity=q, **stats(vals), source=src))
        for q in Q:
            rows.append(dict(subset=subset, rank_class="side", position="all", quantity=q, **stats([]), source=src + " (no observed side-axis leaves)" if sel is not None else src))
    TR = ROOT / "build/traits_recon_20261101_01"; final = TR / "MTG_LEAF_CURVATURE_v1.csv"; outf = HERE / "MTG_LEAF_CURVATURE_v1_rerun.csv"   # amendment 1: compare with the frozen file
    lab = ";".join(LABELS)
    with open(outf, "w", newline="", encoding="utf-8") as f:
        f.write(f"# MTG_LEAF_CURVATURE_v1; per-leaflet quadric z = a0 + a1 s + a2 d + k_l s^2/2 + k_w d^2/2 + c s d on smoothed (r 2 mm) normals' frame; from build/mtg_build_v2_clicks_curv_20261105_01 (_03 split, amendment 5); labels {lab}; read real values net of the noise_floor rows; for the dynamic engine's draws only (static MTG uses each leaflet's own fit)\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    h = sha(outf); same_as_frozen = final.exists() and sha(final) == h
    if not final.exists():
        shutil.copy2(outf, final); (TR / "MTG_LEAF_CURVATURE_v1.sha256").write_text(f"{h}  MTG_LEAF_CURVATURE_v1.csv\n", encoding="utf-8")
    man["checks"] = dict(copied_identical=all(man["copied"].values()), plants_passed=sum(p["passed"] for p in PR), engine_roundtrip=sum(p["engine_roundtrip"] for p in PR), bmodel_identical=sum(p["bmodel_identical_to_03"] for p in PR),
                         leaflets_fitted=len(fit), leaflets_not_fitted=len(C) - len(fit), curvature_file=str(final.relative_to(ROOT)).replace("\\", "/"), curvature_sha256=sha(final), rerun_table_identical_to_frozen=same_as_frozen)
    json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), indent=1); print(json.dumps(man["checks"], indent=1))


if __name__ == "__main__":
    main()
