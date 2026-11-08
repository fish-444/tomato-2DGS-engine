"""E8-1 - synthetic-scene test of E1-E6 (SYNTHETIC performance only; never reported as real-photo performance).

Test set: generator scenes with truth (``build/synthetic_dwarf_tomato_20261008_07/O-mature/scenes``), each in three
variants of increasing occlusion - ``sparse`` (compounds kept greedily so that no two compounds overlap in the top
view), ``half`` (every other compound removed), ``full``. The occlusion actually reached is MEASURED (1 - visible /
amodal leaflet area, top view) and binned: zero <= 2 %, low <= 15 %, high > 15 %.

Metrics, split by occlusion bin x rank position (lower / middle / upper third of the registry ranks, Xin 2024 frame):
node detection P/R/F1 (side-view attachment within the tolerance), compound and leaflet area MAPE (plain top
projection vs the E flow), elevation MAE; occluded-compound comparison (Itakura 2018 frame: plain projection vs E,
success = |error| < 10 %, a candidate threshold); coverage of the truth by the N completion samples; E1 relation
accuracy; E6 contradiction counts; reconstructed_prior area share (E7).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tomato3d.recon.stage1 import assemble, attach, complete, height_order, priors as pr, refine, tilt, views
from tomato3d.state import fixture_truth as ft
from tomato3d.state.topraster import OrthoCamera, render_top

SCENES = Path("build/synthetic_dwarf_tomato_20261008_07/O-mature/scenes")


def occlusion_bin(frac: float) -> str:
    return "zero" if frac <= 0.02 else "low" if frac <= 0.15 else "high"


def variants(plant: ft.FixturePlant) -> dict[str, ft.FixturePlant]:
    out = {"full": plant}
    out["half"] = ft.FixturePlant(plant.source_plant_id, plant.stem_points_mm,
                                  [c for c in plant.compounds if c.serial % 2 == 1], dict(plant.provenance))
    meshes = plant.leaflet_meshes()
    allv = np.vstack([v for v, _ in meshes.values()])
    cam = OrthoCamera(int(np.ceil(2 * (np.abs(allv[:, :2]).max() + 10) / 0.4)), 0.4)
    tr = render_top(meshes, cam)
    comp_mask = {}
    for lid, m in tr.amodal.items():
        cid = lid.rsplit("/", 1)[0]
        comp_mask[cid] = comp_mask.get(cid, np.zeros_like(m)) | m
    kept, union = [], np.zeros_like(next(iter(comp_mask.values())))
    for c in sorted(plant.compounds, key=lambda c: -c.gen_rank_from_bottom):
        m = comp_mask[plant.compound_vertex_id(c)]
        if not (m & union).any():
            kept.append(c)
            union |= m
    out["sparse"] = ft.FixturePlant(plant.source_plant_id, plant.stem_points_mm, kept, dict(plant.provenance))
    return out


def position(rank: int, ranks: list[int]) -> str:
    q = np.argsort(np.argsort(ranks))[ranks.index(rank)] / max(len(ranks) - 1, 1)
    return "lower" if q < 1 / 3 else "middle" if q <= 2 / 3 else "upper"


def run_plant(plant: ft.FixturePlant, P: pr.Priors, n_samples: int, seed: int, out_dir: Path | None) -> dict:
    top, sides, truth = views.build(plant, 0.2, seed=seed)
    registry = truth.registry()   # U5 rank + node index (A7)
    h = height_order.run(top, sides)
    a = attach.run(top, sides, registry, P)
    t = tilt.run(top, sides, a["rows"], P)
    c = complete.run(top, t, h, P, n_samples, seed)
    refined, structures = [], []
    for n in range(n_samples):
        asm = assemble.build(top, a["rows"], t, c, P, n, seed)
        refined += refine.run(top, sides, asm, c)
        structures.append(asm)
    # apply E6 refinements to the per-sample completed areas
    adj = defaultdict(float)
    for r in refined:
        adj[(r["compound"], r["sample"])] += r["area_after_mm2"] - r["area_before_mm2"]
    for comp in c["compounds"]:
        comp["area_completed_refined_mm2"] = [None if v is None else v + adj[(comp["compound"], n)]
                                              for n, v in enumerate(comp["area_completed_mm2"])]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for asm in structures:
            np.savez_compressed(out_dir / f"points_sample{asm['sample']:02d}.npz", points=asm["points"].astype(np.float32),
                                elements=np.array(json.dumps(asm["elements"])))
        (out_dir / "structure.json").write_text(json.dumps(dict(
            height_order=h["rows"], attachments=a["rows"], rank_height_inversions=a["rank_height_inversions"],
            phyllotaxis_flags=a["phyllotaxis_flags"], tilt=t["compounds"], completion=c["compounds"],
            elements=c["elements"], unavailable=c["unavailable"], refinement=refined, seed=seed, n_samples=n_samples),
            default=_jsonable, indent=0), encoding="utf-8")
    return dict(top=top, truth=truth, h=h, a=a, t=t, c=c, refined=refined)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def score(res: dict, P: pr.Priors) -> dict:
    top, truth, h, a, t, c = res["top"], res["truth"], res["h"], res["a"], res["t"], res["c"]
    tol = float(P.value("evaluation", "node_tolerance_mm"))
    # position along the axis = generator node index (U5 ranks share rank 1 on the first two nodes, A7); the written
    # ``rank`` stays the U5 rank
    node = truth.compound_node_index or truth.compound_rank
    ranks = sorted(node.values())
    vis_px = defaultdict(float)
    for lab, info in top.labels.items():
        if info["kind"] == "leaflet":
            vis_px[info["compound"]] += float((top.id_map == lab).sum()) * top.mm2_per_px()
    amodal_frac = {}
    for cid in truth.compound_rank:
        tags = [k for k in truth.visible_fraction if k.rsplit("/", 1)[0] == cid]
        w = [truth.leaflet_area_mm2[k] for k in tags]
        amodal_frac[cid] = float(np.average([truth.visible_fraction[k] for k in tags], weights=w)) if tags else 0.0
    rows = []
    elev = {r["compound"]: r for r in t["compounds"]}
    comp = {r["compound"]: r for r in c["compounds"]}
    att = {r["compound"]: r for r in a["rows"]}
    for cid, rank in truth.compound_rank.items():
        z = att[cid]["attachment_z_mm"]
        true_area = truth.compound_area_mm2[cid]
        cc = comp.get(cid, {})
        comp_n = cc.get("area_completed_refined_mm2", [])
        vis_n = cc.get("area_visible_samples_mm2", [])
        dist = [a + b for a, b in zip(vis_n, comp_n) if a is not None and b is not None]
        samples = [b for b in comp_n if b is not None]
        e_area = float(np.median(dist)) if dist else None
        tilt_only = cc.get("area_visible_mm2")
        rows.append(dict(
            compound=cid, rank=rank, node_index=node[cid], position=position(node[cid], ranks), visible_fraction=amodal_frac[cid],
            node="TP" if z is not None and abs(z - truth.attachment_z_mm[cid]) <= tol else ("FP" if z is not None else "FN"),
            area_true=true_area, area_simple=vis_px[cid], area_e=e_area,
            ape_simple=abs(vis_px[cid] - true_area) / true_area, ape_e=None if e_area is None else abs(e_area - true_area) / true_area,
            err_simple=(vis_px[cid] - true_area) / true_area, err_e=None if e_area is None else (e_area - true_area) / true_area,
            area_tilt=tilt_only, ape_tilt=None if tilt_only is None else abs(tilt_only - true_area) / true_area,
            err_tilt=None if tilt_only is None else (tilt_only - true_area) / true_area,
            node_error_mm=None if z is None else z - truth.attachment_z_mm[cid],
            covered_minmax=bool(dist) and min(dist) <= true_area <= max(dist),
            covered_p5_p95=bool(dist) and np.percentile(dist, 5) <= true_area <= np.percentile(dist, 95),
            elev_err=None if elev[cid]["elevation_deg"] is None else abs(elev[cid]["elevation_deg"] - truth.elevation_deg[cid]),
            prior_share=None if e_area is None or not samples else float(np.median(samples)) / e_area))
    # leaflet-level (observed leaflets)
    leaf = []
    cut_add = defaultdict(list)
    for el in c["elements"]:
        if el["kind"] == "cut_completion":
            cut_add[el["element"].split("#")[0]].append(el["area_mm2"])
    for r in t["leaflets"]:
        tag = truth.leaflet_tag[r["obs_id"]]
        tv = truth.leaflet_area_mm2[tag]
        e = None if r["area_corrected_mm2"] is None else r["area_corrected_mm2"] + (float(np.median(cut_add[r["obs_id"]])) if cut_add[r["obs_id"]] else 0.0)
        leaf.append(dict(position=position(node[r["compound"]], ranks),
                         ape_simple=abs(r["area_top_mm2"] - tv) / tv, ape_e=None if e is None else abs(e - tv) / tv))
    rel = defaultdict(int)
    for x in h["relations"]:
        if x["source"] in ("occlusion", "side_view"):
            ta, tb = truth.leaflet_tag[top.labels[x["a"]]["obs_id"]], truth.leaflet_tag[top.labels[x["b"]]["obs_id"]]
            t_ab, t_ba = truth.above_pairs.get((ta, tb), 0), truth.above_pairs.get((tb, ta), 0)
            if t_ab != t_ba:
                rel[f"{x['source']}|{'ok' if t_ab > t_ba else 'wrong'}"] += 1
    # 'higher-ranked compound on top' agreement between leaflets of different compounds (REPORTED ONLY, not used:
    # generator-derived, circular - E2 prompt item 4)
    rank_cue = dict(agree=0, disagree=0)
    for (ta, tb), n_ab in truth.above_pairs.items():
        ca, cb = ta.rsplit("/", 1)[0], tb.rsplit("/", 1)[0]
        if ca == cb or n_ab <= truth.above_pairs.get((tb, ta), 0):
            continue
        rank_cue["agree" if node[ca] > node[cb] else "disagree"] += 1
    contra = dict(before=sum(r["contradicts_photo_before"] for r in res["refined"]),
                  after=sum(r["contradicts_photo"] for r in res["refined"]), n=len(res["refined"]))
    return dict(compounds=rows, leaflets=leaf, relations=dict(rel), contradictions=contra, rank_cue=rank_cue,
                unavailable=c["unavailable"], height_unavailable=sum(r["height_rank"] == "unavailable" for r in h["rows"]),
                n_leaflets=len(h["rows"]))


def run_all(out: Path, n_scenes: int = 10, n_samples: int = 20, seed: int = 20261014) -> dict:
    P = pr.load()
    summary = []
    for sc in range(n_scenes):
        plant = ft.load_plant(SCENES / f"scene_{sc:04d}", f"syn{sc:02d}")
        for name, var in variants(plant).items():
            if not var.compounds:
                continue
            res = run_plant(var, P, n_samples, seed + sc, out / f"scene_{sc:04d}_{name}")
            s = score(res, P)
            vf = np.array(list(res["truth"].visible_fraction.values()))
            am = np.array([res["truth"].leaflet_area_mm2[k] for k in res["truth"].visible_fraction])
            occl = 1 - float(np.sum(vf * am) / np.sum(am))
            s.update(scene=sc, variant=name, occlusion=occl, bin=occlusion_bin(occl))
            summary.append(s)
    (out / "scores.json").write_text(json.dumps(summary, default=_jsonable, indent=0), encoding="utf-8")
    return dict(summary=summary, seed=seed, n_samples=n_samples)


def _mape(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)) * 100.0, len(v)) if v else (None, 0)


def tables(summary: list[dict], success_rel: float) -> dict:
    """Aggregate per occlusion bin x rank position (+ all)."""
    cells: dict[tuple[str, str], dict] = {}
    for s in summary:
        for pos in ("lower", "middle", "upper", "all"):
            key = (s["bin"], pos)
            cell = cells.setdefault(key, defaultdict(list))
            comps = [r for r in s["compounds"] if pos == "all" or r["position"] == pos]
            leafs = [r for r in s["leaflets"] if pos == "all" or r["position"] == pos]
            for r in comps:
                cell["node"].append(r["node"])
                cell["ape_simple"].append(r["ape_simple"])
                cell["ape_e"].append(r["ape_e"])
                cell["ape_tilt"].append(r.get("ape_tilt"))
                cell["err_simple"].append(r.get("err_simple"))
                cell["err_tilt"].append(r.get("err_tilt"))
                cell["err_e"].append(r.get("err_e"))
                cell["prior_share"].append(r.get("prior_share"))
                cell["elev"].append(r["elev_err"])
                cell["cov_minmax"].append(r["covered_minmax"] if r["area_e"] is not None else None)
                cell["cov_p5_95"].append(r["covered_p5_p95"] if r["area_e"] is not None else None)
            for r in leafs:
                cell["leaf_simple"].append(r["ape_simple"])
                cell["leaf_e"].append(r["ape_e"])
    out = []
    for (b, pos), c in sorted(cells.items()):
        tp, fp, fn = (sum(x == k for x in c["node"]) for k in ("TP", "FP", "FN"))
        pr_ = tp / (tp + fp) if tp + fp else None
        re_ = tp / (tp + fp + fn) if tp + fp + fn else None   # FP = detected at the wrong height = also a missed node
        f1 = 2 * pr_ * re_ / (pr_ + re_) if pr_ and re_ else None
        el = [x for x in c["elev"] if x is not None]
        cov1 = [x for x in c["cov_minmax"] if x is not None]
        cov2 = [x for x in c["cov_p5_95"] if x is not None]
        out.append(dict(bin=b, position=pos, n_compounds=len(c["node"]), node_precision=pr_, node_recall=re_, node_f1=f1,
                        compound_mape_simple=_mape(c["ape_simple"])[0], compound_mape_e=_mape(c["ape_e"])[0],
                        compound_mape_tilt=_mape(c["ape_tilt"])[0],
                        signed_simple=_mape(c["err_simple"])[0], signed_tilt=_mape(c["err_tilt"])[0],
                        signed_e=_mape(c["err_e"])[0],
                        prior_share_median=float(np.median([x for x in c["prior_share"] if x is not None]))
                        if any(x is not None for x in c["prior_share"]) else None,
                        compound_e_n=_mape(c["ape_e"])[1], leaflet_mape_simple=_mape(c["leaf_simple"])[0],
                        leaflet_mape_e=_mape(c["leaf_e"])[0], leaflet_n=_mape(c["leaf_simple"])[1],
                        elevation_mae_deg=float(np.mean(el)) if el else None, elevation_n=len(el),
                        coverage_minmax=float(np.mean(cov1)) if cov1 else None,
                        coverage_p5_p95=float(np.mean(cov2)) if cov2 else None))
    # occluded compounds (visible fraction < 0.9): plain projection vs E flow (Itakura 2018 frame)
    occl = [r for s in summary for r in s["compounds"] if r["visible_fraction"] < 0.9 and r["ape_e"] is not None]
    pairs = dict(n=len(occl), mape_simple=_mape([r["ape_simple"] for r in occl])[0], mape_e=_mape([r["ape_e"] for r in occl])[0],
                 success_simple=float(np.mean([r["ape_simple"] < success_rel for r in occl])) if occl else None,
                 success_e=float(np.mean([r["ape_e"] < success_rel for r in occl])) if occl else None)
    rel = defaultdict(int)
    for s in summary:
        for k, v in s["relations"].items():
            rel[k] += v
    contra = {k: sum(s["contradictions"][k] for s in summary) for k in ("before", "after", "n")}
    share = [r["prior_share"] for s in summary for r in s["compounds"] if r["prior_share"] is not None]
    bins = defaultdict(list)
    for s in summary:
        bins[s["bin"]].append(round(s["occlusion"], 3))
    node_fail = dict(TP=0, FP_wrong_height=0, FN_not_detected=0)
    above, below = 0, 0
    for s in summary:
        for r in s["compounds"]:
            node_fail[{"TP": "TP", "FP": "FP_wrong_height", "FN": "FN_not_detected"}[r["node"]]] += 1
            if r["node"] == "FP" and r.get("node_error_mm") is not None:
                above += r["node_error_mm"] > 0
                below += r["node_error_mm"] < 0
    node_fail.update(wrong_height_above_truth=above, wrong_height_below_truth=below)
    rc = dict(agree=sum(s.get("rank_cue", {}).get("agree", 0) for s in summary),
              disagree=sum(s.get("rank_cue", {}).get("disagree", 0) for s in summary))
    return dict(cells=out, occluded_compounds=pairs, relations=dict(rel), contradictions=contra, node_failures=node_fail,
                rank_cue_reported_only=rc,
                prior_share_median=float(np.median(share)) if share else None,
                prior_share_p90=float(np.percentile(share, 90)) if share else None, bins=dict(bins),
                height_unavailable=sum(s["height_unavailable"] for s in summary), leaflets=sum(s["n_leaflets"] for s in summary))


# --------------------------------------------------------------------------------------
# Session E3: truth stagger x reconstruction stagger grid (synthetic only)
# --------------------------------------------------------------------------------------

TRUTHS = ("simultaneous", "low", "base", "high")
RECONS = ("mature_E2", "low", "base", "high")


def _grid_job(args: tuple) -> list[dict]:
    """One scene under one truth: observe once (E1-E3), reconstruct with every prior (E4-E6), score."""
    sc, truth_name, n_samples, seed, out = args[:5]
    age_mode = args[5] if len(args) > 5 else "age"
    from tomato3d.recon.stage1 import age as age_mod, stagger

    P = pr.load()
    par = age_mod.parameters(P)
    scene_dir = SCENES / f"scene_{sc:04d}"
    plant = ft.load_plant(scene_dir, f"syn{sc:02d}")
    if truth_name != "simultaneous":
        plant = stagger.stagger_plant(plant, scene_dir, par["stagger_dd"][truth_name], par)
    rows = []
    for name, var in variants(plant).items():
        if not var.compounds:
            continue
        top, sides, truth = views.build(var, 0.2, seed=seed + sc)
        registry = truth.registry()   # U5 rank + node index (A7)
        node = truth.compound_node_index or truth.compound_rank
        h = height_order.run(top, sides)
        a = attach.run(top, sides, registry, P)
        t = tilt.run(top, sides, a["rows"], P)
        vf = np.array(list(truth.visible_fraction.values()))
        am = np.array([truth.leaflet_area_mm2[k] for k in truth.visible_fraction])
        occl = 1 - float(np.sum(vf * am) / np.sum(am))
        for recon in RECONS:
            st = None if recon == "mature_E2" else par["stagger_dd"][recon]
            c = complete.run(top, t, h, P, n_samples, seed + sc, registry=registry, stagger_dd=st, age_mode=age_mode)
            refined = []
            for n in range(n_samples):
                refined += refine.run(top, sides, assemble.build(top, a["rows"], t, c, P, n, seed + sc), c)
            adj = defaultdict(float)
            for r in refined:
                adj[(r["compound"], r["sample"])] += r["area_after_mm2"] - r["area_before_mm2"]
            for comp in c["compounds"]:
                comp["area_completed_refined_mm2"] = [None if v is None else v + adj[(comp["compound"], k)]
                                                      for k, v in enumerate(comp["area_completed_mm2"])]
            s = score(dict(top=top, truth=truth, h=h, a=a, t=t, c=c, refined=refined), P)
            s.update(scene=sc, variant=name, occlusion=occl, bin=occlusion_bin(occl), truth=truth_name, recon=recon)
            for r in s["compounds"]:
                cc = next((x for x in c["compounds"] if x["compound"] == r["compound"]), {})
                r.update(age_dd=cc.get("age_dd"), size_rule=cc.get("size_rule"))
            rows.append(s)
            if out is not None:
                d = Path(out) / f"truth_{truth_name}" / f"recon_{recon}" / f"scene_{sc:04d}_{name}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "structure.json").write_text(json.dumps(dict(
                    completion=c["compounds"], unavailable=c["unavailable"], refinement_summary=dict(
                        n=len(refined), contradict_after=sum(r["contradicts_photo"] for r in refined)),
                    seed=seed + sc, n_samples=n_samples, truth_stagger=truth_name, recon_stagger=recon,
                    age_mode=age_mode, restored_age_factors=[dict(
                        compound=e["compound"], position=position(node[e["compound"]], sorted(node.values())),
                        kind=e["kind"], sample=e["sample"], pair_order=e.get("pair_order"),
                        age_dd=next((x["age_dd"] for x in c["compounds"] if x["compound"] == e["compound"]), None),
                        n_groups=next((x.get("n_groups") for x in c["compounds"] if x["compound"] == e["compound"]), None),
                        length_e2_mm=e.get("length_e2_mm"), length_age_mm=e.get("length_age_mm"))
                        for e in c["elements"] if e["source_kind"] == "reconstructed_prior"
                        and e.get("length_e2_mm") not in (None, 0)]),
                    default=_jsonable), encoding="utf-8")
    return rows


def run_grid(out: Path, n_scenes: int = 10, n_samples: int = 20, seed: int = 20261014, workers: int = 8,
             age_mode: str = "age") -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(sc, tn, n_samples, seed, str(out), age_mode) for tn in TRUTHS for sc in range(n_scenes)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_grid_job, jobs):
            rows.extend(part)
    (Path(out) / "scores_grid.json").write_text(json.dumps(rows, default=_jsonable), encoding="utf-8")
    return rows


def grid_tables(rows: list[dict]) -> list[dict]:
    """Per truth x recon x occlusion bin x rank third: MAPE and signed error of simple / tilt-only / E, coverage."""
    cells: dict[tuple, dict] = {}
    for s in rows:
        for r in s["compounds"]:
            if r["area_e"] is None:
                continue
            for pos in (r["position"], "all"):
                for b in (s["bin"], "all"):
                    c = cells.setdefault((s["truth"], s["recon"], b, pos), defaultdict(list))
                    for k in ("simple", "tilt", "e"):
                        v = r.get(f"err_{k}")
                        if v is not None:
                            c[k].append(v)
                    c["cov"].append(r["covered_p5_p95"])
                    c["node"].append(r.get("node_error_mm"))
                    if r.get("prior_share") is not None:
                        c["share"].append(r["prior_share"])
    out = []
    for (tn, rc, b, pos), c in sorted(cells.items()):
        row = dict(truth=tn, recon=rc, bin=b, position=pos, n=len(c["e"]))
        for k in ("simple", "tilt", "e"):
            v = np.array(c[k])
            row[f"mape_{k}"] = float(np.mean(np.abs(v)) * 100) if len(v) else None
            row[f"signed_{k}"] = float(np.mean(v) * 100) if len(v) else None
        row["coverage_p5_p95"] = float(np.mean(c["cov"])) if c["cov"] else None
        nodes = [x for x in c["node"] if x is not None]
        row["node_error_mean_mm"] = float(np.mean(nodes)) if nodes else None
        row["prior_share_median"] = float(np.median(c["share"])) if c["share"] else None
        out.append(row)
    return out
