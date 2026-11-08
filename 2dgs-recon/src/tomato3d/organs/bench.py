"""Step 1 of CONTRACT_ORGAN_TOPOLOGY.md: the synthetic bench from microtom_synth_v2 (plant_tree_v1 truth).

DEV 12 (master 20261024) / TEST 6 (master 20261025); observation-like clouds (OBJ surfaces sampled at 0.36 mm^2, placed at the
R3 pot, visibility from the 61 R3 training views at 504 px, noise 0.25 mm, axis organs thinned to 64 %), declared perturbations
(sparse_base, stem_gap, low_lateral) and the touching-leaflet count. Everything is synthetic; nothing is a measurement.

usage: python -m tomato3d.organs.bench --split dev|test
Writes <run>/bench/<split>/<name>.npz + tree.json + plant.obj and <run>/bench_summary_<split>.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import contract as C
from tomato3d.plant_tree import obj as po
from tomato3d.plant_tree.schema import validate
from tomato3d.plant_tree.summary import summary
from tomato3d.recon import charuco_a3 as ca
from tomato3d.synth import microtom_v2 as v2
from tomato3d.synth.microtom_v2_batch import draw_v2

SPLITS = dict(dev=dict(master=20261024, n=12, prefix="dev"), test=dict(master=20261025, n=6, prefix="test"))
BENCH = dict(sample_mm2=0.36, vis_res=504, vis_tol_m=0.0015, n_vis_min=1, noise_mm=0.25, keep_run_mm=(7.0, 18.0), gap_mm=(4.0, 10.0),
             short_organ_mm=4.0, short_drop_p=0.36, sparse_base_mm=15.0, sparse_base_keep=0.15, stem_gap_mm=(12.0, 16.0),
             low_lateral_node=2, first_two_leaves_mm=4.0, touching_mm=1.5, lift_mm=0.4)
AXIS_FAMILIES = ("stem", "side_stem", "rachis", "petiolule", "peduncle", "pedicel")
BLADE_FAMILIES = ("blade", "cotyledon")
NAME = dict(rachis=re.compile(r"^rachis_(.+)$"), petiolule=re.compile(r"^petiolule_(.+?)__(.+)$"),
            blade=re.compile(r"^((?:S\d+_)?L\d+)__(P\d+[LR]|T|I\d+)$"), truss=re.compile(r"^(T\d+)_(peduncle|pedicel\d+|flower\d+)$"),
            coty=re.compile(r"^cotyledon_(C\d+)$"), side=re.compile(r"^side_stem(?:_(S\d+))?$"))


def perturbations(i: int, split: str, has_side: bool) -> list[str]:
    p = []
    if i % 4 == 0 or (split == "dev" and i == 11):
        p.append("sparse_base")
    if i % 4 == 1 or (split == "dev" and i == 11):
        p.append("stem_gap")
    if i % 4 == 2 and has_side:
        p.append("low_lateral")
    return p


# ------------------------------------------------------------------ low_lateral: translate one side axis to a low node
def _shift(p, d):
    return [p[0] + d[0], p[1] + d[1], p[2] + d[2]]


def low_lateral(tree: dict) -> dict | None:
    sides = [a for a in tree["axes"] if a["kind"] == "side"]
    if not sides:
        return None
    nleaves = {a["id"]: sum(lf["axis"] == a["id"] for lf in tree["leaves"]) for a in sides}
    ax = min(sides, key=lambda a: (nleaves[a["id"]], a["id"]))
    nodes = {n["id"]: n for n in tree["nodes"]}; main = sorted([n for n in tree["nodes"] if n["axis"] == "M"], key=lambda n: n["index"])
    target = main[1]
    if len(main) > 2 and abs(main[1]["s_mm"] - main[0]["s_mm"]) <= BENCH["first_two_leaves_mm"]:
        target = main[2]
    old = nodes[ax["parent_node"]]; d = np.subtract(target["position"], old["position"]).tolist()
    ax["tube"]["polyline"] = [_shift(p, d) for p in ax["tube"]["polyline"]]; ax["parent_node"] = target["id"]
    moved_leaves = [lf for lf in tree["leaves"] if lf["axis"] == ax["id"]]; lids = {lf["id"] for lf in moved_leaves}
    for n in tree["nodes"]:
        if n["axis"] == ax["id"]:
            n["position"] = _shift(n["position"], d)
    for lf in moved_leaves:
        lf["rachis"]["polyline"] = [_shift(p, d) for p in lf["rachis"]["polyline"]]
    for lt in tree["leaflets"]:
        if lt["leaf"] in lids:
            lt["base"] = _shift(lt["base"], d); lt["surface"]["base"] = _shift(lt["surface"]["base"], d)
            if lt.get("petiolule"):
                lt["petiolule"]["polyline"] = [_shift(p, d) for p in lt["petiolule"]["polyline"]]
    for t in tree["trusses"]:
        if t.get("axis") == ax["id"]:
            t["peduncle"]["polyline"] = [_shift(p, d) for p in t["peduncle"]["polyline"]]
            for fl in t["flowers"]:
                fl["position"] = _shift(fl["position"], d)
                if fl.get("mesh"):
                    fl["mesh"]["vertices"] = [_shift(p, d) for p in fl["mesh"]["vertices"]]
                if fl.get("pedicel"):
                    fl["pedicel"]["polyline"] = [_shift(p, d) for p in fl["pedicel"]["polyline"]]

    def lift(p):
        if p[2] < BENCH["lift_mm"]:
            p[2] = BENCH["lift_mm"]
    for lt in tree["leaflets"]:
        if lt["leaf"] in lids:
            lift(lt["base"]); lift(lt["surface"]["base"]); [lift(p) for p in lt["petiolule"]["polyline"]]
    for lf in moved_leaves:
        [lift(p) for p in lf["rachis"]["polyline"]]
    tree["plant"]["flags"] = tree["plant"].get("flags", [])
    tree["notes"].append(f"bench perturbation low_lateral: axis {ax['id']} translated from {old['id']} to {target['id']} (CONTRACT_ORGAN_TOPOLOGY 1.2)")
    validate(tree)
    return dict(axis=ax["id"], from_node=old["id"], to_node=target["id"], shift_mm=[round(x, 2) for x in d])


# ------------------------------------------------------------------ sampling
def sample_obj(objs: list[dict], rng: np.random.Generator, mm2: float):
    xyz, nrm, oid = [], [], []
    for k, o in enumerate(objs):
        if not o["F"]:
            continue
        V = np.asarray(o["V"], float); T = V[np.asarray(o["F"], int)]
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]; cr = np.cross(e1, e2); area = 0.5 * np.linalg.norm(cr, axis=1)
        n = np.floor(area / mm2 + rng.uniform(0, 1, len(area))).astype(int); tot = int(n.sum())
        if tot == 0:
            continue
        ti = np.repeat(np.arange(len(T)), n); u = rng.uniform(0, 1, tot); v = rng.uniform(0, 1, tot); flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        xyz.append(T[ti, 0] + u[:, None] * e1[ti] + v[:, None] * e2[ti])
        nn = cr[ti] / np.maximum(np.linalg.norm(cr[ti], axis=1, keepdims=True), 1e-12); nrm.append(nn); oid.append(np.full(tot, k))
    return np.vstack(xyz), np.vstack(nrm), np.concatenate(oid)


def classify(name: str, tree: dict) -> tuple[str, str, str, str]:
    """-> (family, organ id, leaf id, axis id)."""
    leaf_axis = {lf["id"]: lf["axis"] for lf in tree["leaves"]}; truss_axis = {t["id"]: (t.get("axis") or next(n["axis"] for n in tree["nodes"] if n["id"] == t["node"])) for t in tree["trusses"]}
    if name == "stem":
        return "stem", "M", "", "M"
    m = NAME["side"].match(name)
    if m:
        s = m.group(1) or "S1"; return "side_stem", s, "", s
    m = NAME["rachis"].match(name)
    if m:
        return "rachis", m.group(1), m.group(1), leaf_axis.get(m.group(1), "")
    m = NAME["petiolule"].match(name)
    if m:
        return "petiolule", f"{m.group(1)}__{m.group(2)}", m.group(1), leaf_axis.get(m.group(1), "")
    m = NAME["blade"].match(name)
    if m:
        return "blade", name, m.group(1), leaf_axis.get(m.group(1), "")
    m = NAME["truss"].match(name)
    if m:
        fam = "peduncle" if m.group(2) == "peduncle" else ("pedicel" if m.group(2).startswith("pedicel") else "flower")
        return fam, m.group(1), "", truss_axis.get(m.group(1), "")
    m = NAME["coty"].match(name)
    if m:
        return "cotyledon", m.group(1), "", "M"
    return "other", name, "", ""


def centreline_of(name: str, tree: dict):
    fam, organ, leaf, _ = classify(name, tree)
    if fam == "stem":
        return next(a for a in tree["axes"] if a["kind"] == "main")["tube"]["polyline"]
    if fam == "side_stem":
        return next(a["tube"]["polyline"] for a in tree["axes"] if a["id"] == organ)
    if fam == "rachis":
        return next(lf["rachis"]["polyline"] for lf in tree["leaves"] if lf["id"] == organ)
    if fam == "petiolule":
        return next(lt["petiolule"]["polyline"] for lt in tree["leaflets"] if lt["id"] == organ)
    if fam == "peduncle":
        return next(t["peduncle"]["polyline"] for t in tree["trusses"] if t["id"] == organ)
    if fam == "pedicel":
        k = int(re.search(r"pedicel(\d+)$", name).group(1)); t = next(t for t in tree["trusses"] if t["id"] == organ)
        return t["flowers"][k - 1]["pedicel"]["polyline"]
    return None


def arc_positions(poly, P: np.ndarray) -> np.ndarray:
    """Vectorised arclength_of_nearest: arc position (same units) of the nearest polyline point for every row of P."""
    Q = np.asarray(poly, float); seg = np.diff(Q, axis=0); ln = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(ln)]
    best_d = np.full(len(P), np.inf); best_s = np.zeros(len(P))
    for k in range(len(seg)):
        if ln[k] == 0:
            continue
        t = np.clip((P - Q[k]) @ seg[k] / ln[k] ** 2, 0, 1); d = np.linalg.norm(Q[k] + t[:, None] * seg[k] - P, axis=1)
        m = d < best_d; best_d[m] = d[m]; best_s[m] = cum[k] + t[m] * ln[k]
    return best_s


def visibility(xyz_m: np.ndarray, po_: dict, views: list[str], res: int, tol: float) -> np.ndarray:
    R0 = po_["undistorted_res"]; K = ca.scaled_K(np.array(po_["K_undist"]), R0, res); n_vis = np.zeros(len(xyz_m), int)
    for v in views:
        R = np.array(po_["views"][v]["R_cw"]); t = np.array(po_["views"][v]["t_cw"])
        xc = xyz_m @ R.T + t; z = xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(K[0, 0] * xc[:, 0] / z + K[0, 2]).astype(int); w = np.round(K[1, 1] * xc[:, 1] / z + K[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (w >= 0) & (u < res) & (w < res); idx = np.nonzero(ok)[0]
        zb = np.full((res, res), np.inf); order = idx[np.argsort(-z[idx])]; zb[w[order], u[order]] = z[order]
        vis = idx[z[idx] <= zb[w[idx], u[idx]] + tol]; n_vis[vis] += 1
    return n_vis


def thin_axis(xyz_mm: np.ndarray, names: np.ndarray, tree: dict, rng: np.random.Generator, perts: list[str]) -> tuple[np.ndarray, dict]:
    """Keep mask for axis organs: run/gap thinning per organ centreline, short organs kept or dropped whole, sparse_base / stem_gap."""
    keep = np.ones(len(xyz_mm), bool); rec = dict(organs=0, dropped_short=0)
    main = next(a for a in tree["axes"] if a["kind"] == "main")["tube"]["polyline"]
    for name in np.unique(names):
        fam = classify(str(name), tree)[0]
        if fam not in AXIS_FAMILIES:
            continue
        idx = np.nonzero(names == name)[0]; poly = centreline_of(str(name), tree); L = float(np.linalg.norm(np.diff(np.asarray(poly), axis=0), axis=1).sum())
        rec["organs"] += 1
        if L < BENCH["short_organ_mm"]:
            if rng.uniform() < BENCH["short_drop_p"]:
                keep[idx] = False; rec["dropped_short"] += 1
            continue
        s = arc_positions(poly, xyz_mm[idx]); k = np.zeros(len(idx), bool)
        pos = -rng.uniform(0, sum(BENCH["keep_run_mm"]) / 2 + sum(BENCH["gap_mm"]) / 2)
        while pos < L:
            run = rng.uniform(*BENCH["keep_run_mm"]); k |= (s >= pos) & (s < pos + run); pos += run + rng.uniform(*BENCH["gap_mm"])
        if fam == "stem":
            if "sparse_base" in perts:
                base = s < BENCH["sparse_base_mm"]; k[base] = rng.uniform(size=int(base.sum())) < BENCH["sparse_base_keep"]
                rec["sparse_base_kept"] = int(k[base].sum()); rec["sparse_base_total"] = int(base.sum())
            if "stem_gap" in perts:
                mn = sorted([n for n in tree["nodes"] if n["axis"] == "M"], key=lambda n: n["index"])
                lo, hi = mn[1]["s_mm"], mn[-2]["s_mm"] if len(mn) > 2 else mn[-1]["s_mm"]
                c = rng.uniform(lo, hi); gl = rng.uniform(*BENCH["stem_gap_mm"]); k &= ~(np.abs(s - c) < gl / 2)
                rec["stem_gap"] = dict(centre_s_mm=round(float(c), 2), length_mm=round(float(gl), 2))
        keep[idx] = k
    return keep, rec


def touching_pairs(xyz_mm, names, families, leaves) -> int:
    m = families == "blade"; idx = np.nonzero(m)[0]
    if len(idx) < 2:
        return 0
    pairs = cKDTree(xyz_mm[idx]).query_pairs(BENCH["touching_mm"], output_type="ndarray")
    a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]; diff = leaves[a] != leaves[b]
    return len({tuple(sorted((str(names[i]), str(names[j])))) for i, j in zip(a[diff], b[diff])})


# ------------------------------------------------------------------ one plant
def build_plant(split: str, i: int, out: Path, po_: dict, views: list[str]) -> dict:
    sp = SPLITS[split]; name = f"{sp['prefix']}_{i:02d}"; P, rec = draw_v2(sp["master"], f"draw{i}")
    tree = v2.generate(P); has_side = any(a["kind"] == "side" for a in tree["axes"]); perts = perturbations(i, split, has_side); moved = None
    if "low_lateral" in perts:
        moved = low_lateral(tree)
    d = out / name; d.mkdir(parents=True, exist_ok=True)
    (d / "tree.json").write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")
    obj_text = po.to_obj(tree, header=f"# SYNTHETIC bench plant {name} - microtom_synth_v2 seed {P['seed']} - mm, z up, pot top centre")
    (d / "plant.obj").write_text("# synthetic: true\n" + obj_text, encoding="utf-8")
    rng = np.random.default_rng(sp["master"] + i)
    objs = po.parse_obj(obj_text); xyz, nrm, oid = sample_obj(objs, rng, BENCH["sample_mm2"])
    names = np.array([objs[k]["name"] for k in oid]); cls = {o["name"]: classify(o["name"], tree) for o in objs}
    fam = np.array([cls[n][0] for n in names]); organ = np.array([cls[n][1] for n in names]); leaf = np.array([cls[n][2] for n in names]); axis = np.array([cls[n][3] for n in names])
    n_sampled = len(xyz)
    xyz_m = xyz / 1000 + np.array(C.POT_CENTRE_M)
    n_vis = visibility(xyz_m, po_, views, BENCH["vis_res"], BENCH["vis_tol_m"]); vis = n_vis >= BENCH["n_vis_min"]
    touching = touching_pairs(xyz[vis], names[vis], fam[vis], leaf[vis])
    keep, trec = thin_axis(xyz, names, tree, rng, perts); keep &= vis
    xyz_m = xyz_m + nrm * rng.normal(0, BENCH["noise_mm"] / 1000, (len(xyz), 1))
    k = np.nonzero(keep)[0]
    axis_label = np.where(np.isin(fam, AXIS_FAMILIES), 1, np.where(np.isin(fam, BLADE_FAMILIES), 0, -1)).astype(np.int8)
    np.savez_compressed(d / "cloud.npz", xyz=xyz_m[k].astype(np.float32), normal=nrm[k].astype(np.float32), family=fam[k], obj=names[k], leaf=leaf[k],
                        organ=organ[k], axis=axis[k], axis_label=axis_label[k], n_vis=n_vis[k].astype(np.int16))
    s = summary(tree); mn = sorted([n for n in tree["nodes"] if n["axis"] == "M"], key=lambda n: n["index"])
    lats = []
    for a in tree["axes"]:
        if a["kind"] != "side":
            continue
        pn = next(n for n in tree["nodes"] if n["id"] == a["parent_node"])
        lats.append(dict(id=a["id"], parent_node=a["parent_node"], parent_node_index=pn["index"], parent_z_mm=round(pn["position"][2], 2),
                         leaves=[lf["id"] for lf in tree["leaves"] if lf["axis"] == a["id"]], trusses=[t["id"] for t in tree["trusses"] if t.get("axis") == a["id"]],
                         side_stem_len_mm=round(float(np.linalg.norm(np.diff(np.asarray(a["tube"]["polyline"]), axis=0), axis=1).sum()), 2),
                         side_stem_points_present=int(((organ == a["id"]) & (fam == "side_stem") & keep).sum())))
    ax_all = int((axis_label == 1).sum()); ax_kept = int((axis_label[k] == 1).sum())
    return dict(name=name, split=split, index=i, seed=P["seed"], draw=rec, perturbations=perts, low_lateral=moved, thinning=trec,
                params=dict(n_truss=P["n_truss"], visible=P["visible"], side_leaves=P["side_leaves"], internode=P["internode"], lf_len=P["lf_len"]),
                counts=dict(points_sampled=n_sampled, points_visible=int(vis.sum()), points_kept=int(len(k)), axis_points_visible=int(((axis_label == 1) & vis).sum()),
                            axis_points_kept=ax_kept, axis_kept_share=round(ax_kept / max(ax_all, 1), 3), blade_points_kept=int((axis_label[k] == 0).sum()),
                            touching_blade_pairs=touching, n_vis_median=float(np.median(n_vis[k]))),
                truth=dict(summary_counts=s["counts"], per_axis=s["per_axis"], main_ranks=s["counts"]["main_ranks"], main_nodes=len(mn),
                           main_node_z_mm=[round(n["position"][2], 2) for n in mn], main_node_s_mm=[round(n["s_mm"], 2) for n in mn], main_node_rank=[n["rank"] for n in mn],
                           top_main_node_z_mm=round(mn[-1]["position"][2], 2), stem_top_z_mm=round(max(p[2] for p in next(a for a in tree["axes"] if a["kind"] == "main")["tube"]["polyline"]), 2),
                           laterals=lats, leaves=len(tree["leaves"]), leaflets=len(tree["leaflets"]), trusses=len(tree["trusses"])))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=("dev", "test"), required=True); a = ap.parse_args(argv)
    run = C.TOPO_RUN; out = run / "bench" / a.split; out.mkdir(parents=True, exist_ok=True)
    po_ = json.load(open(C.R3["poses"])); views = json.load(open(C.R3["run_dir"] / "contract.json"))["train"]
    rows = [build_plant(a.split, i, out, po_, views) for i in range(SPLITS[a.split]["n"])]
    for r in rows:
        print(f"{r['name']} seed {r['seed']} n_truss {r['params']['n_truss']} side_leaves {r['params']['side_leaves']} perts {r['perturbations']} "
              f"kept {r['counts']['points_kept']} axis_share {r['counts']['axis_kept_share']} laterals {[(l['id'], l['parent_node'], l['side_stem_points_present']) for l in r['truth']['laterals']]} "
              f"touching {r['counts']['touching_blade_pairs']}", flush=True)
    comp = dict(split=a.split, master_seed=SPLITS[a.split]["master"], n=len(rows), bench=BENCH, views=len(views),
                plants_per_perturbation={p: sum(p in r["perturbations"] for r in rows) for p in ("sparse_base", "stem_gap", "low_lateral")},
                plants_with_laterals=sum(bool(r["truth"]["laterals"]) for r in rows), laterals=sum(len(r["truth"]["laterals"]) for r in rows),
                laterals_side_stem_present=sum(l["side_stem_points_present"] >= 5 for r in rows for l in r["truth"]["laterals"]),
                touching_pairs_total=sum(r["counts"]["touching_blade_pairs"] for r in rows), plants_with_touching=sum(r["counts"]["touching_blade_pairs"] > 0 for r in rows),
                synthetic=True, note="synthetic bench; nothing here is a plant measurement")
    json.dump(dict(composition=comp, plants=rows), open(run / f"bench_summary_{a.split}.json", "w"), indent=1, default=float)
    print(json.dumps(comp, indent=1, default=float))


if __name__ == "__main__":
    main()
