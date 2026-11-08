"""Part A: area (b) for e_* with the unchanged tomato3d/plant_model code (CONTRACT.md). usage: python -X utf8 run_area_e.py {hashes|run}"""
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from tomato3d.plant_model import areas as AR
from tomato3d.plant_model import inputs as I

OUT = Path(__file__).resolve().parent
SKEL = {"e_01": "26f27427b73b", "e_04": "73fdd1ffeead", "e_05": "524f23114149", "e_07": "16690f4f8d25"}
LAB = ROOT / "build/p1_dz10_labels_20261101_01"
OLD = ROOT / "build/p1_plant_model_20261026_01"
TAGS = "conditional_on_unvalidated_scale; skeleton_labels_ungraded; R3-1b-style_unaccepted"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).relative_to(ROOT)).replace("\\", "/")


def inputs_list():
    fs = []
    for e, sk in SKEL.items():
        fs += [next((I.ALL_OUT / e).glob("*_surfels_full.ply")), I.stage2_poses_path(e), LAB / f"labels_tprox_{e}_{sk}.npz"]
    fs += [ROOT / "build/p1_recon_all_20261025_01/census_new/census_d1028.csv", OLD / "summary.csv", OLD / "plants/n_03/static_surfels.csv",
           I.stage2_poses_path("n_03"), next((I.ALL_OUT / "n_03").glob("*_surfels_full.ply"))]
    fs += sorted((ROOT / "tomato3d/plant_model").glob("*.py"))
    return fs


def poses_of(pid):
    pz = json.load(open(I.stage2_poses_path(pid), encoding="utf-8")); acc = pz["accepted"]; v0 = pz["views"][acc[0]]
    # identical to inputs.photo_source(...)["poses"] (photos are not needed by the area)
    return dict(full_size=v0["size"], K_full=v0["K"], dist=v0["dist"], undistorted_res=v0["size"][0], K_undist=v0["K"],
                views={v: dict(R_cw=pz["views"][v]["R_cw"], t_cw=pz["views"][v]["t_cw"]) for v in acc})


def area_of(pid):
    sf = I.sio.read_surfels_full(next((I.ALL_OUT / pid).glob("*_surfels_full.ply")))
    rows = np.nonzero(sf["opacity"] >= I.OPACITY_MIN)[0]
    xyz, tu, tv, sc = (sf[k][rows] for k in ("xyz", "tu", "tv", "scale"))
    po = poses_of(pid); views = AR.pick_views(po)
    s = AR.silhouette_ratio(xyz, tu, tv, sc, po, views)
    a = AR.disc_area_m2(sc) * 1e6
    return rows, a, s, views, len(po["views"])


def main(step):
    if step == "hashes":
        json.dump({rel(p): sha(p) for p in inputs_list()}, open(OUT / "CONTRACT_HASHES.json", "w"), indent=1); return
    ch = json.load(open(OUT / "CONTRACT_HASHES.json"))
    bad = [k for k, v in ch.items() if sha(ROOT / k) != v]
    if bad:
        raise SystemExit(f"input changed since contract: {bad}")
    t = time.time()
    # gate: n_03
    rows, a, s, views, nacc = area_of("n_03")
    old = {r["id"]: r for r in I.read_commented_csv(OLD / "summary.csv")[1]}["n_03"]
    _, ss = I.read_commented_csv(OLD / "plants/n_03/static_surfels.csv")
    ob = np.array([float(r["area_b_mm2"]) for r in ss]); orow = np.array([int(r["surfel_row"]) for r in ss])
    nb = a * s["ratio"]
    gate = dict(r=s["ratio"], r_old=float(old["canopy_silhouette_ratio"]), area_b=float(nb.sum()), area_b_old=float(old["canopy_area_b_mm2"]),
                rows_equal=bool(np.array_equal(orow, rows)), max_abs_surfel_diff=float(np.max(np.abs(ob - nb))) if np.array_equal(orow, rows) else None)
    gate["pass"] = bool(gate["rows_equal"] and gate["max_abs_surfel_diff"] <= 1e-5 and round(gate["r"], 4) == gate["r_old"]
                        and round(gate["area_b"]) == gate["area_b_old"])
    json.dump(gate, open(OUT / "GATE_n03.json", "w"), indent=1)
    print("gate", gate, f"{time.time() - t:.0f}s", flush=True)
    if not gate["pass"]:
        raise SystemExit("reproduction gate FAILED; no e_* output")
    summ, perview = [], {}
    for e, sk in SKEL.items():
        rows, a, s, views, nacc = area_of(e)
        z = np.load(LAB / f"labels_tprox_{e}_{sk}.npz")
        if not np.array_equal(z["rows"], rows):
            raise SystemExit(f"{e}: label rows != opacity rows")
        b = a * s["ratio"]; lab = z["label"].astype(int); st = z["status"].astype(int); uids = [str(u) for u in z["leaf_uids"]]
        out = []
        for k, u in enumerate(uids):
            m = (lab == k + 1) & (st == 1)
            out.append(dict(leaf_uid=u, area_b_mm2=f"{b[m].sum():.3f}", n_surfels=int(m.sum()), source="recon_area_b_ungraded_labels"))
        with open(OUT / f"area_b_e_{e[2:]}.csv", "w", encoding="utf-8", newline="") as f:
            f.write(f"# {e}; {TAGS}; area_b = area_a x canopy silhouette ratio r={s['ratio']:.6f} (8 views); per leaf = sum over status-1 surfels (lower bound); labels {rel(LAB / f'labels_tprox_{e}_{sk}.npz')}\n")
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
        asg = (st == 1) & (lab > 0)
        summ.append(dict(id=e, set="e_new", n_surfels=len(rows), n_accepted_views=nacc, n_views_silhouette=s["n_views"], canopy_area_a_mm2=round(float(a.sum()), 1),
                         silhouette_ratio=round(s["ratio"], 6), ratio_min=round(s["ratio_min"], 6), ratio_max=round(s["ratio_max"], 6),
                         canopy_area_b_mm2=round(float(b.sum()), 1), assigned_area_b_mm2=round(float(b[asg].sum()), 1),
                         unassigned_area_b_mm2=round(float(b[~asg].sum()), 1), n_leaves=len(uids), n_leaves_empty=sum(1 for o in out if o["n_surfels"] == 0),
                         source="this_run"))
        perview[e] = dict(views=views, **{k: v for k, v in s.items()})
        print(summ[-1], f"{time.time() - t:.0f}s", flush=True)
    for r in I.read_commented_csv(OLD / "summary.csv")[1]:
        summ.append(dict(id=r["id"], set="pilot" if r["id"] == "R3-1b" else "nine_10-26", n_surfels=int(r["n_surfels"]), n_accepted_views="", n_views_silhouette=8,
                         canopy_area_a_mm2=r["canopy_area_a_mm2"], silhouette_ratio=r["canopy_silhouette_ratio"], ratio_min="", ratio_max="",
                         canopy_area_b_mm2=r["canopy_area_b_mm2"], assigned_area_b_mm2="", unassigned_area_b_mm2="", n_leaves="", n_leaves_empty="",
                         source="copied build/p1_plant_model_20261026_01/summary.csv (unchanged)"))
    with open(OUT / "area_summary.csv", "w", encoding="utf-8", newline="") as f:
        f.write(f"# {TAGS}; per-plant canopy silhouette ratio r and area (b); R3-1b pilot leaves r 0.0572 (canopy); n_*/t2_*/R3-1b rows copied, not recomputed\n")
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    json.dump(perview, open(OUT / "per_view_ratios.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1])
