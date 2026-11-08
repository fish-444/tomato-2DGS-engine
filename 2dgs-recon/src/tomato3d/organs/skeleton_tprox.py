"""T-prox as the operating route on the other individuals (contract build/p1_tprox_others_20261101_01/CONTRACT.md). Reuses skeleton_r8pc unchanged with RUNS pointed at the new folder.

usage: python -m tomato3d.organs.skeleton_tprox --step prep|push|finish|r31b_conf|summary --id <pid>
  prep    copy axis labels, build the prox bundle
  push    Kaggle dataset + kernel (frozen pc_absorb.py)
  finish  poll -> download -> set A -> frozen rachis -> label files, diagnostics, figure, plant tree
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np

from tomato3d.organs import skeleton_io as SIO
from tomato3d.organs import skeleton_propagate as SP
from tomato3d.organs import skeleton_r8pc as R8

ROOT = SP.ROOT
OUT = ROOT / "build/p1_tprox_others_20261101_01"
R8.RUNS = OUT / "runs"
SET = "A"
SEED_MIN = 50
LABELS = ["route_tprox", "tprox_unverified_until_heldout", "conditional_on_unvalidated_scale", "user_skeleton_seeds", "frozen_code_reuse"]
ORDER = ["n_04", "t2_02", "e_01", "e_04", "e_05", "n_01", "n_02", "n_03", "n_05", "n_06", "t2_05"]
NOT_RUN = {"e_02": "no 3D-first groups", "e_03": "no 3D-first groups", "e_06": "no 3D-first groups", "e_07": "no skeleton", "t2_04": "no skeleton"}


def skel_dir(pid):
    return SIO.skeleton_dir(ROOT, pid, SP.capture_date(pid))


def step_prep(pid):
    src = ROOT / "build/p1_skeleton_layer_20261101_01/r8pc" / pid; d = R8.RUNS / pid; d.mkdir(parents=True, exist_ok=True)
    for f in ("axis_labels.npz", "axis.json"):
        if not (d / f).exists():
            shutil.copy2(src / f, d / f)
    return R8.step_bundle(pid, mode="prox")


def step_push(pid):
    R8.step_push(pid)


def download_set(pid):
    """poll, download, and place the declared set's labels at R8_pc/labels.npz (kernel output kept as is in kaggle_out/)."""
    d = R8.run_dir(pid); st = R8.step_poll(pid)
    if st != "COMPLETE":
        raise SystemExit(f"{pid}: kernel {st}")
    kj = json.load(open(d / "kaggle.json")); got = d / "kaggle_out"; got.mkdir(exist_ok=True)
    for _ in range(6):
        R8.kg("kernels", "output", "dolgfish/" + kj["kernel"], "-p", str(got), "-q")
        if list(got.rglob("labels.npz")):
            break
        time.sleep(60)
    res = json.load(open(next(got.rglob("results.json"))))
    chosen = res["choice"]["chosen"]; f = got / ("R8_pc" if chosen == SET else f"set_{SET}") / "labels.npz"
    z = np.load(f); assert str(z["set_name"]) == SET, (str(z["set_name"]), f)
    (d / "R8_pc").mkdir(exist_ok=True); shutil.copy2(f, d / "R8_pc" / "labels.npz"); shutil.copy2(next(got.rglob("results.json")), d / "R8_pc" / "results.json")
    return d, dict(in_kernel_choice=chosen, in_kernel_gate=res["choice"]["gate_pass"], used_set=SET, seconds=res.get("seconds"))


def q(x):
    return {p: round(float(np.percentile(x, p)), 4) for p in (5, 25, 50, 75, 95)} if len(x) else None


def step_finish(pid):
    d, kinfo = download_set(pid); json.dump(kinfo, open(d / "kernel_info.json", "w"), indent=1); R8.step_rachis(pid); return step_post(pid)


def provenance(r8, zr):
    """A1 classes: 1 seed (R8_pc stage1), 2 absorbed (R8_pc rounds), 3 rachis_step (rachis-file trace_seed / rounds), 0 none."""
    c = np.zeros(len(zr["label"]), np.int8); s8 = r8["source"]; sr = zr["source"]
    c[s8 == 1] = 1; c[(s8 >= 2) & (s8 <= 6)] = 2; c[(sr >= 2) & (sr <= 7)] = 3; c[zr["label"] <= 0] = 0; return c


def step_post(pid):
    """Everything after the frozen rachis step (no Kaggle, no rerun of any route)."""
    d = R8.run_dir(pid)
    if not (d / "kernel_info.json").exists():                                                                  # finished by the pre-A1 code: rebuild from the kernel's results.json
        res = json.load(open(next((d / "kaggle_out").rglob("results.json")))); json.dump(dict(in_kernel_choice=res["choice"]["chosen"], in_kernel_gate=res["choice"]["gate_pass"], used_set=SET, seconds=res.get("seconds")), open(d / "kernel_info.json", "w"), indent=1)
    kinfo = json.load(open(d / "kernel_info.json"))
    meta = json.load(open(d / "bundle_meta.json", encoding="utf-8")); lm = {int(k): v for k, v in meta["leaf_map"].items()}
    z = np.load(d / "R8_pc_rachis" / "labels_rachis.npz"); rows = z["rows"]; lab_t = z["label"].astype(int); r8 = np.load(d / "R8_pc" / "labels.npz"); assert np.array_equal(r8["rows"], rows)
    p1 = r8["p1"]; p2 = r8["p2"]; prov = provenance(r8, z)                                                       # A1: margin from R8_pc, meaningful for absorbed only
    dist_f = skel_dir(pid) / "labels_skeleton_distance.npz"; zd = np.load(dist_f, allow_pickle=True); assert np.array_equal(zd["rows"], rows)
    uids = [str(u) for u in zd["leaf_uids"]]; ui = {u: i + 1 for i, u in enumerate(uids)}
    lab = np.array([ui[lm[k]] if k > 0 else 0 for k in range(max(lm) + 1)])[lab_t]                    # T-prox leaf k -> index in the distance file's leaf_uids
    tags = LABELS + (["legacy_split_20mm_unreviewed"] if any(f in l.get("flags", []) for f in ("legacy_split_20mm", "legacy_split_dz10mm") for l in SIO.load(skel_dir(pid) / "skeleton_v1.json")["leaves"]) else [])
    np.savez_compressed(OUT / f"labels_tprox_{pid}.npz", rows=rows, label=lab.astype(np.int16), status=(lab > 0).astype(np.int8), leaf_uids=np.array(uids), p1=p1, p2=p2, margin=np.where(prov == 2, p1 - p2, np.nan).astype(np.float32),
                        provenance=prov, provenance_names=np.array(["0:none", "1:seed", "2:absorbed", "3:rachis_step"]), labels=np.array(tags))
    shutil.copy2(dist_f, OUT / f"labels_distance_{pid}.npz")
    ld = zd["label"].astype(int); both = (lab > 0) & (ld > 0)
    M = np.zeros((len(uids) + 1, len(uids) + 1), int); np.add.at(M, (lab, ld), 1)
    st1 = np.load(d / "kaggle_data" / "r8pc_bundle.npz")["stage1"].astype(int)
    sk = SIO.load(skel_dir(pid) / "skeleton_v1.json"); live = [l for l in sk["leaves"] if l["status"] in ("active", "new")]
    a = lab > 0; ab = prov == 2; mg = (p1 - p2)[ab]
    diag = dict(labels=tags, plant_id=pid, run=str(d.relative_to(ROOT)), kernel=kinfo, lines_live=len(live), leaves_tprox=len(lm), line_too_short=[l["leaf_uid"] for l in live if len(l["polyline_xyz"]) < 2],
                seeds=meta["seeds"], seed_surfels_per_leaf={lm[k]: int((st1 == k).sum()) for k in lm}, leaves_seed_below_min=[lm[k] for k in lm if (st1 == k).sum() < SEED_MIN], seed_min=SEED_MIN,
                assigned=round(float(a.mean()), 4), contested_or_unlabelled=round(float((~a).mean()), 4), provenance={nm: int((prov == i).sum()) for i, nm in enumerate(("none", "seed", "absorbed", "rachis_step"))}, p1_quantiles_absorbed=q(p1[ab]), margin_quantiles_absorbed=q(mg),
                distance=dict(assigned=round(float((ld > 0).mean()), 4)), both_assigned=int(both.sum()), changed_leaf_share_among_both=round(float((lab[both] != ld[both]).mean()), 4) if both.any() else None,
                tprox_only=int(((lab > 0) & (ld == 0)).sum()), distance_only=int(((lab == 0) & (ld > 0)).sum()), agreement_matrix=dict(row="tprox (0=none)", col="distance (0=none)", leaf_uids=uids, counts=M.tolist()),
                axis=json.load(open(R8.RUNS / pid / "axis.json", encoding="utf-8")), grade="ungraded (no held-out picks)" if not sk.get("heldout") else "see heldout",
                light=("per_surfel_light present; per-leaf light not reported: ungraded" if (SP.LIGHT_NINE / pid / "per_surfel_light.npz").exists() else "light_not_available"))
    diag["axis_share_unusual"] = diag["axis"]["axis_share_unusual"]
    if sk.get("heldout"):
        diag["heldout"] = heldout(pid, sk, rows, lab, ld, uids)
    json.dump(diag, open(OUT / f"diag_{pid}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    tree(pid, sk, rows, lab, uids); figure(pid, rows, lab, ld)
    print(json.dumps({k: diag[k] for k in ("plant_id", "leaves_tprox", "assigned", "changed_leaf_share_among_both", "leaves_seed_below_min", "kernel")}, ensure_ascii=False, default=float)); return diag


def heldout(pid, sk, rows, lab, ld, uids):
    pos = {int(r): i for i, r in enumerate(rows)}; out = {}
    for name, L in (("tprox", lab), ("distance", ld)):
        out[name] = SIO.heldout_agreement(sk["heldout"], lambda s, L=L: (uids[L[pos[int(s)]] - 1] if int(s) in pos and L[pos[int(s)]] > 0 else None))
    t, dd = out["tprox"], out["distance"]; better = "tprox" if (t["rate"] or 0) >= (dd["rate"] or 0) else "distance"      # heldout_agreement() names the share "rate" (was "agreement": KeyError 2026-11-02 n_01)
    out["operating_route"] = better; out["grade"] = out[better]["grade"]; out["grade_from"] = better; return out


def tree(pid, sk, rows, lab, uids):
    from tomato3d.organs import skeleton_tree as ST
    f = SP.LIGHT_NINE / pid / "per_surfel_light.npz"; areas = src = None
    if f.exists():
        L = np.load(f, allow_pickle=True)
        if np.array_equal(L["ply_row"], rows):
            areas = ST.leaf_areas_from_labels(lab, (lab > 0).astype(int), L["area_b_silhouette_mm2"], uids); src = f"{f.relative_to(ROOT)} (area_b_silhouette_mm2 of surfels assigned by route_tprox; tprox_unverified_until_heldout)"
    t = ST.skeleton_to_plant_tree(sk, areas, src); (OUT / f"plant_tree_tprox_{pid}.json").write_text(json.dumps(t, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def figure(pid, rows, lab, ld):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    g = np.load(SP.ALL3 / pid / "merge" / "groups.npz"); assert np.array_equal(g["rows"], rows); X = g["xyz"] * 1000
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    ims = sorted((SP.ALL3 / pid / "rf_inputs").glob("*.jpg"))
    if ims:
        ax[0].imshow(Image.open(ims[0])); ax[0].set_title(f"photo {ims[0].name} (prepared crop)")
    ax[0].axis("off")
    cm = plt.get_cmap("tab20"); n = lab > 0
    ax[1].scatter(X[~n, 0], X[~n, 1], s=0.3, c="#cccccc"); ax[1].scatter(X[n, 0], X[n, 1], s=0.3, c=cm((lab[n] - 1) % 20)); ax[1].set_title(f"{pid} T-prox leaves (top view, mm); grey = none")
    dis = (lab != ld); ax[2].scatter(X[:, 0], X[:, 1], s=0.3, c="#dddddd")
    for sel, col, nm in ((dis & (lab > 0) & (ld > 0), "#d73027", "different leaf"), (dis & (lab > 0) & (ld == 0), "#1a9850", "T-prox only"), (dis & (lab == 0) & (ld > 0), "#4575b4", "distance only")):
        ax[2].scatter(X[sel, 0], X[sel, 1], s=0.6, c=col, label=f"{nm} ({int(sel.sum())})")
    ax[2].legend(markerscale=8, fontsize=8); ax[2].set_title("T-prox vs distance disagreement")
    for a in ax[1:]:
        a.set_aspect("equal")
    fig.suptitle(f"{pid}  route_tprox  tprox_unverified_until_heldout  conditional_on_unvalidated_scale"); fig.tight_layout(); fig.savefig(OUT / f"fig_{pid}.png", dpi=90); plt.close(fig)


def step_r31b_conf():
    """Contract section 5: high-confidence wrong assignments of the passing R3-1b T-prox run."""
    from tomato3d.organs import oversegment as OS
    from tomato3d.organs import skeleton_r31b_eval as EV
    run = ROOT / "build/p1_skeleton_layer_20261101_01/r8pc/R3-1b/20261101T133038"; z = np.load(run / "R8_pc_rachis/labels_rachis.npz"); r8 = np.load(run / "R8_pc/labels.npz"); Bs = OS.load_base(); si, hl = EV.human_projections(Bs)
    pos = {int(r): i for i, r in enumerate(z["rows"])}; brow = Bs["pl"]["rows"][Bs["b"]]; bi = np.array([pos[int(r)] for r in brow])[si]
    lab = z["label"].astype(int)[bi]; prov = provenance(r8, z)[bi]; mg = (r8["p1"] - r8["p2"])[bi]; names = ("none", "seed", "absorbed", "rachis_step")

    def block(sel_leaf):
        a = sel_leaf & (lab > 0); wrong = a & (lab != hl); out = dict(assigned=int(a.sum()), wrong=int(wrong.sum()), accuracy=round(float(1 - wrong.sum() / max(a.sum(), 1)), 4) if a.any() else None)
        for i in (1, 2, 3):
            m = a & (prov == i); out[names[i]] = dict(n=int(m.sum()), wrong=int((m & wrong).sum()), wrong_rate=round(float((m & wrong).sum() / max(m.sum(), 1)), 4), share_of_wrong=round(float((m & wrong).sum() / max(wrong.sum(), 1)), 4))
        return out, a, wrong

    rep = np.isin(hl, (2, 4, 5, 7)); R, a, wrong = block(rep); ab = a & (prov == 2); q75 = float(np.percentile(mg[ab], 75)) if ab.any() else None
    top = ab & (mg >= q75); R["absorbed_margin"] = dict(q75=round(q75, 4), wrong_share_with_margin_ge_q75=round(float((wrong & top).sum() / max((wrong & ab).sum(), 1)), 4),
                                                        wrong_rate_top_quartile=round(float((wrong & top).sum() / max(top.sum(), 1)), 4), wrong_rate_below_q75=round(float((wrong & ab & ~top).sum() / max((ab & ~top).sum(), 1)), 4))
    per = {str(L): dict(report_half=L in (2, 4, 5, 7), scored=int((hl == L).sum()), **block(hl == L)[0]) for L in range(1, 9)}
    res = dict(labels=["route_tprox", "reference_only_scoring", "sim_skeleton_completed", "amendment_A1"], run="20261101T133038", report_half=R, per_leaf=per,
               note="leaf 6 ~0.02 in every skeleton route: known failure, not a route property; seed = forced (stage1, never weighed against other leaves)")
    json.dump(res, open(OUT / "r31b_tprox_confidence.json", "w", encoding="utf-8"), indent=1); print(json.dumps(res, indent=1)); return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--id"); a = ap.parse_args(argv)
    dict(prep=lambda: step_prep(a.id), push=lambda: step_push(a.id), finish=lambda: step_finish(a.id), post=lambda: step_post(a.id), r31b_conf=step_r31b_conf)[a.step]()


if __name__ == "__main__":
    main()
