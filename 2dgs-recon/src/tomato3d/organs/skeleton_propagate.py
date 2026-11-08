"""skeleton_v1 -> surfel labels (contract build/p1_skeleton_layer_20261101_01/CONTRACT.md, sections 3-4).

`distance_baseline` = the frozen distance rule of `skeleton_assign` (25 mm, 3 mm margin, 3D-first groups whole), fed from the skeleton instead of the raw click log.
The main route (R8_pc) runs only where its frozen inputs exist (R3-1b); for the other individuals it is reported as not runnable (missing axis_kind and stage1), never approximated.
Labels are always re-derived from the skeleton; nothing is edited by hand. Previous result files are kept (timestamped).

usage: python -m tomato3d.organs.skeleton_propagate --step convert|distance|check --id n_04
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tomato3d.organs import skeleton_assign as SK
from tomato3d.organs import skeleton_io as SIO

ROOT = SK.ROOT
ALL3 = SK.OUT                                                       # build/p1_all_3dfirst_20261027_01
LIGHT_NINE = ROOT / "build/engine_static_nine_20261028_01/kaggle_out"
PROP_LABELS = SIO.LABELS + ["R3-1b-style_unaccepted", "no_reference"]


def capture_date(pid):
    f = ROOT / "build/p1_recon_all_20261025_01/outputs" / pid / "recon_meta.json"
    return json.load(open(f, encoding="utf-8"))["session_date"] if f.exists() else "unknown"


def reconstruction_id(pid):
    ply = ROOT / f"build/p1_recon_all_20261025_01/outputs/{pid}/R3-1b-style_{pid}_surfels_full.ply"
    return f"build/p1_recon_all_20261025_01/outputs/{pid}#{SIO.sha_file(ply)[:16]}" if ply.exists() else f"unknown/{pid}"


def groups(pid):
    z = np.load(ALL3 / pid / "merge" / "groups.npz"); return z


# ---------------------------------------------------------------- convert
def step_convert(pid):
    sk = SIO.from_click_log(ALL3 / "user_base_click.jsonl", pid, capture_date(pid), reconstruction_id(pid))
    z = groups(pid); flags = SIO.validate(sk, z["xyz"].astype(float))
    f = SIO.save(sk, ROOT); live = [l for l in sk["leaves"]]
    print(json.dumps(dict(file=str(f.relative_to(ROOT)), leaves=len(live), clicks=sum(l["n_clicks"] for l in live), flags=[(a, b, c) for a, b, c in flags]), ensure_ascii=False, indent=1)); return sk, flags


# ---------------------------------------------------------------- distance route
def route_distance(sk, z):
    """Returns label (leaf index 1..K in `uids` order, 0 none), status (0 none / 1 assigned / 2 contested), uids."""
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new")]
    uids = [l["leaf_uid"] for l in live]
    xyz = z["xyz"].astype(float) * 1000.0; grp = z["group"].astype(int); n = len(xyz)
    br = [SK.densify(np.asarray(l["polyline_xyz"], float) * 1000.0) for l in live]
    lab = np.zeros(n, int); st = np.zeros(n, int)
    if not br:
        return lab, st, uids
    D = np.stack([cKDTree(b).query(xyz)[0] for b in br], 1); gids = np.unique(grp[grp >= 0])
    if len(gids):
        order = np.argsort(grp); gs = grp[order]; starts = np.searchsorted(gs, gids); ends = np.searchsorted(gs, gids, side="right")
        med = np.stack([np.median(D[order[a:b]], 0) for a, b in zip(starts, ends)]); gl, gst = SK.decide(med)
        for l_, s_, a, b in zip(gl, gst, starts, ends):
            lab[order[a:b]] = l_; st[order[a:b]] = s_
    rest = grp < 0; l2, s2 = SK.decide(D[rest]); lab[rest] = l2; st[rest] = s2
    return lab, st, uids


def contested_light(pid, rows, status):
    f = LIGHT_NINE / pid / "per_surfel_light.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    if not np.array_equal(z["ply_row"], rows):
        return dict(note="per-surfel light rows differ from the grouping rows; not used")
    a = z["absorbed_ru_mm2"].astype(float); tot = float(a.sum())
    return dict(contested=round(float(a[status == 2].sum() / tot), 4), none=round(float(a[status == 0].sum() / tot), 4), assigned=round(float(a[status == 1].sum() / tot), 4), source=str(f.relative_to(ROOT)))


def step_distance(pid, sk_file=None):
    d = SIO.skeleton_dir(ROOT, pid, capture_date(pid)); sk = SIO.load(sk_file or d / "skeleton_v1.json"); z = groups(pid)
    lab, st, uids = route_distance(sk, z); n = len(lab); stamp = time.strftime("%Y%m%dT%H%M%S")
    leaf_of = lambda i: (uids[lab[i] - 1] if (st[i] == 1 and lab[i] > 0) else None)
    ho = SIO.heldout_agreement(sk.get("heldout", []), leaf_of) if sk.get("heldout") else None
    per_leaf_light = None
    if ho and ho["grade"] in ("A", "B"):                                                       # A4: per-leaf light only for grade A / B, re-aggregated from the frozen per-surfel light (no new ray tracing)
        f = LIGHT_NINE / pid / "per_surfel_light.npz"
        if f.exists():
            L = np.load(f, allow_pickle=True)
            if np.array_equal(L["ply_row"], z["rows"]):
                per_leaf_light = {u: dict(absorbed_ru_mm2=float(L["absorbed_ru_mm2"][(lab == k + 1) & (st == 1)].sum()), area_b_mm2=float(L["area_b_silhouette_mm2"][(lab == k + 1) & (st == 1)].sum()),
                                          n_surfels=int(((lab == k + 1) & (st == 1)).sum())) for k, u in enumerate(uids)}
    summ = dict(labels=PROP_LABELS + ["distance_route"], plant_id=pid, capture_date=sk["capture_date"], route="distance_route (operating route, A4)", rule=SK.R, per_leaf_light=(per_leaf_light or "not reported: grade " + (ho["grade"] if ho else "pending")),
                grade_ko=(f"held-out {ho['agree']}/{ho['n']} 일치 (Wilson 95 % {ho['wilson95'][0]:.2f}–{ho['wilson95'][1]:.2f}) → 등급 {ho['grade']}" + ("" if ho["grade"] in ("A", "B") else " (C: 잎별 광 미보고)") if ho else "확인 점 미완료"), skeleton_file=str((d / "skeleton_v1.json").relative_to(ROOT)),
                skeleton_sha256=SIO.sha_file(d / "skeleton_v1.json"), leaves=len(uids), assigned=round(float((st == 1).mean()), 4), contested=round(float((st == 2).mean()), 4), none=round(float((st == 0).mean()), 4),
                surfels_per_leaf={u: int(((lab == k + 1) & (st == 1)).sum()) for k, u in enumerate(uids)}, heldout=ho if ho else "pending: no held-out clicks yet",
                contested_light_share=contested_light(pid, z["rows"], st), main_route_r8pc=("runnable" if pid == "R3-1b" else "not_runnable_inputs_missing: axis_kind (Q1) and stage1 (R7 labels from the V-P base) exist for R3-1b only"),
                main_route_status="failed_gate (adapter, A2/A3); distance route is the operating route (A4)", at=stamp)
    np.savez_compressed(d / f"labels_skeleton_distance.{stamp}.npz", rows=z["rows"], label=lab.astype(np.int16), status=st.astype(np.int8), leaf_uids=np.array(uids))
    np.savez_compressed(d / "labels_skeleton_distance.npz", rows=z["rows"], label=lab.astype(np.int16), status=st.astype(np.int8), leaf_uids=np.array(uids))
    json.dump(summ, open(d / f"summary.{stamp}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float); json.dump(summ, open(d / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("rule",)}, ensure_ascii=False, indent=1, default=float)); return summ, lab, st, uids


def step_check(pid):
    """Plumbing check: the skeleton route must reproduce the labels of the existing skeleton_assign run from the same clicks."""
    s, zl = SK.latest(pid)
    if s is None:
        print("no earlier skeleton_assign run"); return None
    d = SIO.skeleton_dir(ROOT, pid, capture_date(pid)); sk = SIO.load(d / "skeleton_v1.json"); z = groups(pid); lab, st, uids = route_distance(sk, z)
    same = bool(np.array_equal(zl["label"], lab) and np.array_equal(zl["status"], st)); print(json.dumps(dict(earlier_file=s["file"], identical_labels_and_status=same, branches_earlier=int(s["branches"]), leaves_now=len(uids)))); return same


# ---------------------------------------------------------------- skeleton -> plant_tree_v1 (section 4)
def step_tree(pid):
    from tomato3d.organs import skeleton_tree as ST
    d = SIO.skeleton_dir(ROOT, pid, capture_date(pid)); sk = SIO.load(d / "skeleton_v1.json"); lab = np.load(d / "labels_skeleton_distance.npz", allow_pickle=True)
    uids = [str(u) for u in lab["leaf_uids"]]; areas, src = None, None; f = LIGHT_NINE / pid / "per_surfel_light.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if np.array_equal(z["ply_row"], lab["rows"]):
            areas = ST.leaf_areas_from_labels(lab["label"], lab["status"], z["area_b_silhouette_mm2"], uids); src = f"{f.relative_to(ROOT)} (area_b_silhouette_mm2 of surfels assigned by the distance_baseline route)"
    tree = ST.skeleton_to_plant_tree(sk, areas, src); import json as _j
    (d / "plant_tree_v1.json").write_text(_j.dumps(tree, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    out = ROOT / "build/p1_skeleton_layer_20261101_01" / f"engine_{pid}"; rt = ST.engine_round_trip(tree, out, pid)
    res = dict(labels=PROP_LABELS, plant_id=pid, leaves=len(tree["leaves"]), tree_file=str((d / "plant_tree_v1.json").relative_to(ROOT)), areas=("from distance_baseline labels (lower bound)" if areas else "unavailable"),
               engine_round_trip=rt, one_click_leaves=[l["leaf_uid"] for l in sk["leaves"] if l["n_clicks"] < 2])
    json.dump(res, open(ROOT / "build/p1_skeleton_layer_20261101_01" / f"tree_{pid}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float); print(json.dumps(res, indent=1, ensure_ascii=False, default=float)); return res


# ---------------------------------------------------------------- R3-1b simulated check (contract section 5 + Amendment A1)
def step_r31b(pid="R3-1b"):
    """Skeleton from the 10 human rachis traces -> back to frozen trace arrays -> compare with r8pc_bundle.npz. Only light inputs are read (trace JSONs, matrix_T.csv, the bundle)."""
    from tomato3d.organs import compound_cluster as CC
    from tomato3d.organs import local_diffusion as LD
    from tomato3d.organs import trace_reconcile as TR
    traces = CC.load_traces(TR.TRACES); tmap = LD.trace_map_vp()
    for t in traces:
        t["leaf"] = tmap[t["key"]]["leaf"]                                              # the frozen "vp" numbering of the bundle
    zb = np.load(ROOT / "build/p1_pc_absorb_20261027_01/kaggle_data/r8pc_bundle.npz"); ply = ROOT / "build/p1_recon_r3-1_contam_20261022_01/outputs/R3-1b_surfels_full.ply"
    ori = json.load(open(ROOT / "build/p1_compound_2dvote_20261025_01/traces_oriented.json", encoding="utf-8")); flip = [t["key"] for t in ori["traces"] if t["flipped"]]      # base = end nearer the pot-centre vertical
    sk = SIO.from_traces(traces, "R3-1b", "2026-10-21", f"build/p1_recon_r3-1_contam_20261022_01/outputs/R3-1b_surfels_full.ply#{SIO.sha_file(ply)[:16]}", source="simulated_from_human_traces", flip_keys=flip)
    flags = SIO.validate(sk, zb["xyz"].astype(float), pot_xy=CC.POT_CENTRE)
    f = SIO.save(sk, ROOT); back = SIO.to_traces(sk); pts, lf, ids = [], [], []
    for t in back:
        if t["leaf"] <= 0:
            continue
        d = CC.densify(t["pts"]); pts.append(d); lf.append(np.full(len(d), t["leaf"])); ids.append(np.full(len(d), t["index"]))
    P, L, I = np.vstack(pts), np.concatenate(lf), np.concatenate(ids)
    eq = dict(trace_pts=bool(np.array_equal(P, zb["trace_pts"])), trace_leaf=bool(np.array_equal(L, zb["trace_leaf"])), trace_id=bool(np.array_equal(I, zb["trace_id"])),
              max_dev_mm=(float(np.abs(P - zb["trace_pts"]).max() * 1000) if P.shape == zb["trace_pts"].shape else None), shapes=[list(P.shape), list(zb["trace_pts"].shape)])
    same = all(eq[k] for k in ("trace_pts", "trace_leaf", "trace_id"))
    res = dict(labels=PROP_LABELS + ["simulated_from_human_traces"], skeleton_file=str(f.relative_to(ROOT)), leaves=len(sk["leaves"]), mapped_to_pipeline_leaf=sum(1 for t in back if t["leaf"] > 0),
               unmapped=[t["leaf_uid"] for t in back if t["leaf"] <= 0], validator_flags=[list(x) for x in flags], bundle_arrays_identical=eq, plumbing_pass=same,
               frozen_results_used=("the frozen R8_pc / R8_pc_rachis result files are the output of this route (same input arrays, same frozen code); no rerun (contract Amendment A1)" if same else "arrays differ: Kaggle rerun required"),
               frozen_numbers=dict(R8_pc=dict(report_half_accuracy=0.8326, coverage=0.9012), R8_pc_rachis=dict(pooled_accuracy=0.9362, coverage=0.9805, weakest="leaf 7 -> leaf 6 (0.849)")), human_leaves_used_for="scoring only (frozen numbers)")
    out = ROOT / "build/p1_skeleton_layer_20261101_01" / "r31b_simulated_check.json"; json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float)); return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--id", required=True); a = ap.parse_args(argv)
    {"convert": step_convert, "distance": step_distance, "check": step_check, "r31b": step_r31b, "tree": step_tree}[a.step](a.id)


if __name__ == "__main__":
    main()
