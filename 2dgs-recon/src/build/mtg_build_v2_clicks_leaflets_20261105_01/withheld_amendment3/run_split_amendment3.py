"""Real plants: split (amendment 3) -> leaflets.csv, leaflet_split_summary.csv, per-plant npz; then set _03 (trees with split leaflets).
python -X utf8 run_split.py"""
import copy, csv, hashlib, json, math, shutil, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
V02 = ROOT / "build/mtg_build_v2_clicks_20261103_02"; OUT = ROOT / "build/mtg_build_v2_clicks_20261105_03"; FIT = ROOT / "build/mtg_fit_surfels_20261103_01"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(FIT)); sys.path.insert(0, str(ROOT / "build/mtg_build_20261102_01"))
import split_lib as SL  # noqa: E402
import fit_plant as FP  # noqa: E402
import run_real as RR  # noqa: E402
import writers as W  # noqa: E402
import mtg_spec as MS  # noqa: E402
from tomato3d.plant_model import areas as AR, inputs as I  # noqa: E402
from tomato3d.plant_tree import engine as PE, obj as PO  # noqa: E402
from tomato3d.plant_tree.schema import validate  # noqa: E402

PLANTS = ["R3-1b", "n_01", "n_02", "n_03", "n_04", "n_05", "n_06", "e_01", "e_04", "e_05", "e_07"]
LABELS = ["leaflet_split_from_surfels", "skeleton_labels_ungraded", "conditional_on_unvalidated_scale", "not_validation", "user_accepted_despite_count_gate_fail"]
E_RATIO = {"e_01": 0.062968, "e_04": 0.060958, "e_05": 0.056927, "e_07": 0.061757}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
RASTER = 0.25


def surfel_data(pid, D):
    """X (mm, label order), area (b) mm2, normals, aligned to D['X']."""
    if pid == "R3-1b":
        z = np.load(ROOT / "build/engine_coplanar_sep_20261101_01/out_k1/per_surfel_D.npz"); sf = I.sio.read_surfels_full(ROOT / "build/p1_recon_r3-1_contam_20261022_01/outputs/R3-1b_surfels_full.ply")
        pr = np.asarray(z["ply_row"]); X = np.asarray(z["xyz"], float) * 1000.0; assert np.allclose(X, D["X"]); assert np.allclose(sf["xyz"][pr] * 1000, X, atol=1e-3)
        return X, np.asarray(z["area_b_mm2"], float), np.asarray(sf["normal"][pr], float), pr, "per_surfel_D.npz area_b_mm2"
    L = np.load(MS.DZ / f"labels_{MS.ROUTE[pid]}_{pid}_{MS.SNAP[pid]}.npz"); rows = np.asarray(L["rows"])
    sf = I.sio.read_surfels_full(next((I.ALL_OUT / pid).glob("*_surfels_full.ply"))); X = sf["xyz"][rows] * 1000.0; assert np.allclose(X, D["X"], atol=1e-3)
    if pid.startswith("n_"):
        z = np.load(ROOT / f"build/engine_static_nine_20261028_01/kaggle_out/{pid}/per_surfel_light.npz", allow_pickle=True); assert np.array_equal(np.asarray(z["ply_row"]), rows), pid
        A = np.asarray(z["area_b_silhouette_mm2"], float); src = "per_surfel_light.npz area_b_silhouette_mm2"
    else:
        op = np.nonzero(sf["opacity"] >= I.OPACITY_MIN)[0]; assert np.array_equal(op, rows), pid
        A = AR.disc_area_m2(sf["scale"][rows]) * 1e6 * E_RATIO[pid]; src = f"disc_area x r {E_RATIO[pid]} (run_area_e.py method)"
    return X, A, np.asarray(sf["normal"][rows], float), rows, src


def curve_of(lf):
    for k, src in (("rachis_curve_from_clicks_mm", "clicks"), ("rachis_polyline_level2_mm", "level2_surfel_fit")):
        c = lf.get(k)
        if c and len(c) >= 2:
            return np.asarray(c, float), src
    return None, None


def raster_overlap(leaflets):
    """top-view (xy) overlap of the alpha meshes: (sum of areas - union) / sum."""
    polys = [(np.asarray(l["mesh"]["vertices"], float), np.asarray(l["mesh"]["faces"], int)) for l in leaflets if len(l["mesh"]["faces"])]
    if len(polys) < 2:
        return float("nan"), 0.0
    allv = np.vstack([v for v, _ in polys]); lo = allv[:, :2].min(0) - 1; hi = allv[:, :2].max(0) + 1; nx, ny = (np.ceil((hi - lo) / RASTER).astype(int) + 1)
    cnt = np.zeros((ny, nx), np.int16); tot = 0
    for V, F in polys:
        im = Image.new("1", (int(nx), int(ny)), 0); dr = ImageDraw.Draw(im); P = (V[:, :2] - lo) / RASTER
        for f in F:
            dr.polygon([tuple(P[i]) for i in f], fill=1)
        m = np.array(im, dtype=bool); cnt += m; tot += int(m.sum())
    union = int((cnt > 0).sum()); return (tot - union) / tot if tot else float("nan"), tot * RASTER ** 2


def run_plant(pid):
    D = FP.load_real(pid); spec = json.load(open(V02 / f"static/mtg_{pid}_spec.json", encoding="utf-8")); X, A, N, rows, asrc = surfel_data(pid, D)   # _02 spec (load_real reads the first build)
    assert [lf["uid"] for lf in spec["leaves"]] == [lf["uid"] for lf in D["spec"]["leaves"]], pid
    todo = []
    for lf in spec["leaves"]:
        idx = D["leaf_idx"].get(lf["uid"], np.zeros(0, int)); c, csrc = curve_of(lf)
        todo.append((lf, idx, c, csrc))
    split = lambda lf, idx, c, ext: SL.split_leaf(X[idx], A[idx], N[idx], c, lf["leaflets"], ext=ext)
    ok = [(lf, idx, c, s) for lf, idx, c, s in todo if c is not None and len(idx) >= SL.MIN_SURFELS]
    lat = [l["length_mm"] for lf, idx, c, s in ok for l in split(lf, idx, c, SL.EXT)["leaflets"] if l["type"] == "P"]
    ext = 1.5 * float(np.median(lat)) if lat else SL.EXT
    res = {}; summ = []; lrows = []; lab_leaf = np.full(len(X), "", object); lab_lt = np.full(len(X), "", object)
    for lf, idx, c, csrc in todo:
        u = lf["uid"]; base = dict(plant=pid, uid=u, k=lf["k"], n_surfels=len(idx), curve_source=csrc or "none", prior_leaflets=lf["leaflets"], spec_area_mm2=lf.get("area_mm2"), extension_mm=round(ext, 3))
        if c is None or len(idx) < SL.MIN_SURFELS:
            base.update(status="split_not_possible_kept_helpers"); summ.append(base); continue
        r = split(lf, idx, c, ext); S = r["summary"]; res[u] = (r, idx, c)
        fr = r["frame"]; Q, sQ = fr["Q"], fr["sQ"]; Tq = SL.tangents(Q)
        # compound plane (PCA of all leaf surfels), normal +z
        Y = X[idx] - X[idx].mean(0); w, v = np.linalg.eigh(np.cov(Y.T)); nc = v[:, 0] * (1 if v[2, 0] >= 0 else -1)
        own_tot = sum(l["area_own"] for l in r["leaflets"]); ov, proj = raster_overlap(r["leaflets"])
        for l in r["leaflets"]:
            j = int(np.clip(np.searchsorted(sQ, l["base_s"]), 0, len(Q) - 1)); t = Tq[j]
            d = np.asarray(l["d"]); n_l = np.asarray(l["normal"])
            tp = t - nc * (t @ nc); dp = d - nc * (d @ nc); tp /= np.linalg.norm(tp) or 1; dp /= np.linalg.norm(dp) or 1
            phi = float(np.degrees(np.arccos(np.clip(tp @ dp, -1, 1))))
            tilt = float(np.degrees(np.arccos(np.clip(abs(n_l @ nc), -1, 1)))); sgn = 1 if l["side"] in ("L", "C") else -1
            roll = int(np.sign(n_l @ np.cross(t, nc)) * sgn)
            lrows.append(dict(plant=pid, uid=u, k=lf["k"], rank_class="main", leaflet=f"{l['type']}{l['pair'] if l['type'] == 'P' else ''}{l['side'] if l['type'] == 'P' else ''}", type=l["type"], pair=l["pair"], side=l["side"],
                              L_mm=S["L"], base_s_mm=l["base_s"], s_over_L=l["base_s"] / S["L"], s_mean_mm=l["s_mean"], spread_phi_deg=phi, petiolule_offset_mm=l["base_r"], tilt_deg=tilt, roll_sign=roll,
                              length_mm=l["length_mm"], width_mm=l["width_mm"], area_own_mm2=l["area_own"], area_share=l["area_own"] / own_tot if own_tot else float("nan"), mesh_area_mm2=l["mesh_area"],
                              elev_deg=l["elev_deg"], az_deg=l["az_deg"], n_surfels=l["n"], within_leaf_overlap=ov))
        for q, l in enumerate(r["leaflets"]):
            nm = f"{l['type']}{l['pair'] if l['type'] == 'P' else ''}{l['side'] if l['type'] == 'P' else ''}"; lab_lt[idx[l["idx"]]] = nm
        lab_leaf[idx] = u
        base.update(status="split", L_mm=S["L"], m_data_L=S["m_data_L"], m_chosen_L=S["m_chosen_L"], m_data_R=S["m_data_R"], m_chosen_R=S["m_chosen_R"], n_leaflets=S["n_leaflets"],
                    terminal_found=S["terminal_present"], missing=";".join(S["missing"]), rachis_area_mm2=S["rachis_area"], rachis_share=S["rachis_share"], surfel_area_mm2=S["total_area"],
                    leaflet_area_mm2=S["leaflet_area"], conservation_err=S["conservation_err"], spec_minus_surfel_area=(lf["area_mm2"] or 0) - S["total_area"], orphans=S["n_orphan_reassigned"],
                    within_leaf_overlap=ov, bic_L=json.dumps(S["bic_L"]), bic_R=json.dumps(S["bic_R"]))
        summ.append(base)
    np.savez_compressed(HERE / f"split_{pid}.npz", rows=rows, leaf=lab_leaf.astype(str), leaflet=lab_lt.astype(str), area_b=A, area_source=asrc)
    return D, spec, res, summ, lrows, ext


def build_tree(spec, res, X):
    queue = [np.asarray(lf["rachis_polyline_level2_mm"], float) if lf.get("rachis_polyline_level2_mm") else None for lf in spec["leaves"]]
    orig = W.leaf_polyline; it = iter(queue)
    def patched(base, az, th, dt, L, m):
        q = next(it, None)
        return orig(base, az, th, dt, L, m) if q is None else ([p.tolist() for p in q], None)
    W.leaf_polyline = patched
    try:
        tree = W.spec_to_tree(spec)
    finally:
        W.leaf_polyline = orig
    tree["plant"]["attrs"]["labels"] = list(dict.fromkeys(tree["plant"]["attrs"]["labels"] + LABELS))
    tree["plant"]["attrs"]["leaflet_split"] = "build/mtg_build_v2_clicks_leaflets_20261105_01 (amendment 3); split leaves carry reconstructed leaflets (mesh surfaces); others keep the _02 helper leaflets"
    byuid = {lf["attrs"].get("uid"): lf for lf in tree["leaves"]}; sp = {lf["uid"]: lf for lf in spec["leaves"]}
    for u, (r, idx, c) in res.items():
        lf = byuid[u]; lid = lf["id"]; tree["leaflets"] = [t for t in tree["leaflets"] if t["leaf"] != lid]
        S = r["summary"]; L = S["L"]; Qr, sr, _ = SL.resample(c); P = r["leaflets"]
        pairs = sorted({l["pair"] for l in P if l["type"] == "P"})
        att_s = []
        for k in pairs:
            s = float(np.mean([l["base_s"] for l in P if l["type"] == "P" and l["pair"] == k])); att_s.append(min(max(s, 0.01), L))
        for i in range(len(att_s)):          # strictly increasing, below L
            if i:
                att_s[i] = max(att_s[i], att_s[i - 1] + 0.01)
        hi = L - 0.01 * np.arange(len(att_s), 0, -1); att_s = [min(a, h) for a, h in zip(att_s, hi)]
        pl = [Qr[0].tolist()] + [SL.point_at(Qr, sr, s).tolist() for s in att_s] + [Qr[-1].tolist()]
        lf["rachis"].update(polyline=pl, radius=[0.6] * len(pl)); lf["petiole"].update(polyline=[pl[0], pl[1]])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.asarray(pl), axis=0), axis=1))]      # attachment s along the tree polyline (chords), curve s kept in attrs
        att_curve = list(att_s); att_s = [float(x) for x in cum[1:1 + len(att_curve)]]; L_poly = float(cum[-1])
        spec_area = sp[u]["area_mm2"] or 0.0; own = np.array([l["area_own"] for l in P]); tot_own = own.sum()
        with_r = own + S["rachis_area"] * own / tot_own if tot_own else own; scale = spec_area / with_r.sum() if with_r.sum() else 0.0
        atts = {}
        for l, a_r in zip(P, with_r):
            name = f"P{l['pair']}{l['side']}" if l["type"] == "P" else "T"
            d = np.asarray(l["d"]); n = np.asarray(l["normal"]); n = n - d * (n @ d); n /= np.linalg.norm(n) or 1; b = np.cross(n, d)
            s_att = att_s[pairs.index(l["pair"])] if l["type"] == "P" else L_poly
            V = np.round(np.asarray(l["mesh"]["vertices"]), 4).tolist(); F = l["mesh"]["faces"]
            surf = dict(kind="mesh", vertices=V, faces=F) if len(F) else dict(kind="param_blade", base=l["base"], d=d.tolist(), n=n.tolist(), L=l["length_mm"], W=l["width_mm"], cup=0.0, bend=0.0, nu=12, nv=7)
            lt = dict(id=f"{lid}__{name}", leaf=lid, type=l["type"], side=l["side"], pair=l["pair"], attachment_s_mm=float(s_att), base=l["base"], frame=dict(d=d.tolist(), n=n.tolist(), b=b.tolist()),
                      length_mm=l["length_mm"], width_mm=l["width_mm"], area_mm2=float(a_r * scale), normal=n.tolist(), surface=surf, petiolule=None, age_d=None, expansion=None,
                      flags=[],
                      attrs=dict(split_flags=["leaflet_split_from_surfels"] + ([] if len(F) else ["mesh_empty_param_blade_stand_in"]), area_own_mm2=l["area_own"], rachis_share_added_mm2=float(a_r - l["area_own"]), scale_to_spec_area=scale, base_s_mm=l["base_s"], attachment_s_on_curve_mm=(att_curve[pairs.index(l["pair"])] if l["type"] == "P" else L), s_mean_mm=l["s_mean"], n_surfels=l["n"], mesh_area_mm2=l["mesh_area"]))
            tree["leaflets"].append(lt); atts.setdefault((l["type"], l["pair"]), dict(s_mm=float(s_att), leaflets=[]))["leaflets"].append(lt["id"])
        lf["attachments"] = sorted(atts.values(), key=lambda a: a["s_mm"])
        lf["flags"] = [f for f in lf["flags"] if f != "intercalary_as_extra_leaflet"]
        lf["attrs"].update(split_flags=["leaflet_split_from_surfels"] + (["no_terminal_surfels"] if not S["terminal_present"] else []), pairs=len(pairs), intercalary=0, leaflets_split=S["n_leaflets"], leaflets_geometry="reconstructed leaflets from the labelled surfels (amendment 3)", rachis_surfel_area_mm2=S["rachis_area"],
                           split_missing=S["missing"], m_chosen=[S["m_chosen_L"], S["m_chosen_R"]], m_data=[S["m_data_L"], S["m_data_R"]])
    return validate(tree)


def write_set(spec, tree, name):
    d = OUT / "static"; (d / f"{name}_engine.plant_tree_v1.json").write_text(json.dumps(tree, indent=1, default=float), encoding="utf-8")
    PE.to_engine_structure(tree, d / f"{name}_engine", plant_instance_id="p1", source_plant_id=f"mtg_{spec['plant_id']}", fixture=False, capture_id=f"mtg_build_{spec['plant_id']}")
    (d / f"{name}_engine.obj").write_text(PO.to_obj(tree, header=f"mtg static _03 {spec['plant_id']} (leaflets split from surfels); mm, z up"), encoding="utf-8")
    bsha, info = W.spec_to_bmodel(spec, d / f"{name}_bmodel.csv")
    return RR.engine_round_trip(tree, d / f"{name}_engine"), bsha, info


def main():
    (OUT / "static").mkdir(parents=True, exist_ok=True); man = dict(inputs={}, copied={}, plants={})
    for p in sorted(V02.rglob("*")):        # copy _02 except observed-plant static files and bookkeeping
        rel = p.relative_to(V02)
        if p.is_dir() or "__pycache__" in p.parts or rel.parts[0] == "kaggle" or rel.name.startswith("RUNNING.lock") or rel.name in ("FINAL_HASHES.json", "MANIFEST.json", "CONTRACT.md", "package_02.py", "roundtrip_bmodel_02.py", "roundtrip_bmodel_02.csv", "roundtrip_engine_e.csv"):
            continue
        if rel.parts[0] == "static" and any(f"mtg_{q}_" in rel.name or rel.parts[1:2] == (f"mtg_{q}_engine",) for q in PLANTS):
            continue
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, OUT / rel); man["copied"][str(rel).replace("\\", "/")] = sha(p) == sha(OUT / rel)
    S_all, L_all, rt_rows = [], [], []
    for pid in PLANTS:
        D, spec, res, summ, lrows, ext = run_plant(pid)
        spec3 = copy.deepcopy(spec); spec3["labels"] = list(dict.fromkeys(spec3["labels"] + LABELS)); spec3["mtg_set"] = dict(version="static_v2_click_skeleton_20261105_03", source_spec=f"build/mtg_build_v2_clicks_20261103_02/static/mtg_{pid}_spec.json", leaflet_split="build/mtg_build_v2_clicks_leaflets_20261105_01")
        for lf in spec3["leaves"]:
            if lf["uid"] in res:
                S = res[lf["uid"]][0]["summary"]; lf["leaflets_split"] = S["n_leaflets"]; lf["leaflets_split_source"] = "leaflet_split_from_surfels (amendment 3); spec 'leaflets' (prior, hand mode by rank) unchanged"
        json.dump(spec3, open(OUT / f"static/mtg_{pid}_spec.json", "w", encoding="utf-8"), indent=1, default=float)
        tree = build_tree(spec3, res, D["X"]); rt, bsha, info = write_set(spec3, tree, f"mtg_{pid}")
        old_b = V02 / f"static/mtg_{pid}_bmodel.csv"; same = bsha is not None and old_b.exists() and sha(old_b) == sha(OUT / f"static/mtg_{pid}_bmodel.csv")
        rt_rows.append(dict(plant=pid, leaves=len(spec["leaves"]), split=len(res), engine_roundtrip=rt["passed"], **{k: v for k, v in rt.items() if k != "passed"}, bmodel=info["status"], bmodel_identical_to_02=same, extension_mm=round(ext, 2)))
        S_all += summ; L_all += lrows; print(rt_rows[-1], flush=True)
    for fn, data in (("leaflet_split_summary.csv", S_all), ("leaflets.csv", L_all), ("roundtrip_03.csv", rt_rows)):
        keys = []
        for r in data:
            keys += [k for k in r if k not in keys]
        with open(HERE / fn, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)
    shutil.copy2(HERE / "roundtrip_03.csv", OUT / "roundtrip_engine_03.csv")
    man["checks"] = dict(copied_identical=all(man["copied"].values()), copied_n=len(man["copied"]), engine_roundtrip=sum(r["engine_roundtrip"] for r in rt_rows), bmodel_identical=sum(r["bmodel_identical_to_02"] for r in rt_rows),
                         max_conservation_err=max(abs(r.get("conservation_err", 0)) for r in S_all), split_leaves=sum(r["status"] == "split" for r in S_all), kept_helpers=sum(r["status"] != "split" for r in S_all))
    json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), indent=1); print(json.dumps(man["checks"]))


if __name__ == "__main__":
    main()
