"""v2 batch with the v1 batch's sampling contract (same master seed and draw order: tomato3d.synth.microtom_batch.draw), a
v1 / v2 contact sheet for the same seeds, a range check (not a fit) and the compare() self-test. Synthetic only.

usage: python -m tomato3d.synth.microtom_v2_batch --out build/p1_synth_v2_20261022_01 --v1 build/p1_synth_microtom_20261022_01 [--n 40]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from tomato3d.plant_tree import obj as po
from tomato3d.plant_tree.summary import compare, summary
from tomato3d.synth import microtom_batch as b1
from tomato3d.synth import microtom_v2 as v2

COL = dict(stem="#c98a2b", rachis="#7d9a4a", blade_P="#3f8a45", blade_T="#2f7552", blade_I="#9fbc3a", petiolule="#8aa362", flower="#e3c22c", truss="#7d9a4a", cotyledon="#88b06a")


def draw_v2(master: int, key: str) -> tuple[dict, dict]:
    P1, rec = b1.draw(master, key); P = v2.params_for(None, **P1)
    rec = dict(rec, v2_extra={k: P[k] for k, _, _ in v2.V2_EXTRA}, v2_rule="v1 draw, v2 keys at their defaults")
    return P, rec


def obj_tris(text: str):
    tris, cols = [], []
    for o in po.parse_obj(text):
        V = np.asarray(o["V"], float)
        for f in o["F"]:
            tris.append(V[f]); cols.append(COL.get(o["cat"], "#888"))
    return np.array(tris), np.array(cols)


def contact_sheet(pairs: list[tuple[str, str, str]], path: Path, cols: int = 5) -> None:
    """pairs = (name, v1 obj path, v2 obj path); each item = 2 columns (v1 | v2), rows top / side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    rows = math.ceil(len(pairs) / cols); fig, axs = plt.subplots(rows * 2, cols * 2, figsize=(cols * 3.6, rows * 3.8)); axs = np.atleast_2d(axs)
    for k, (name, p1, p2) in enumerate(pairs):
        r, c = divmod(k, cols)
        for col_off, (label, path_) in enumerate((("v1", p1), ("v2", p2))):
            T, C = obj_tris(Path(path_).read_text(encoding="utf-8"))
            for row_off, (a, b, dep, title) in enumerate(((0, 1, 2, "top"), (0, 2, 1, "side"))):
                ax = axs[r * 2 + row_off, c * 2 + col_off]
                order = np.argsort(T[:, :, dep].mean(1)) if title == "top" else np.argsort(-T[:, :, dep].mean(1))
                ax.add_collection(PolyCollection(T[order][:, :, [a, b]], facecolors=C[order], edgecolors="none"))
                ax.set_xlim(-90, 90); ax.set_ylim(-90, 90) if title == "top" else ax.set_ylim(-5, 175)
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{name} {label} {title}", fontsize=6)
    for k in range(len(pairs), rows * cols):
        r, c = divmod(k, cols)
        for i in range(2):
            for j in range(2):
                axs[r * 2 + i, c * 2 + j].axis("off")
    fig.suptitle("SYNTHETIC Micro-Tom: v1 (left pair) vs v2 (right pair), same seeds - top / side, mm grid", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--v1", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--master", type=int, default=20261022); a = ap.parse_args(argv); out = Path(a.out); v1dir = Path(a.v1)
    summ, leaf_rows, pairs, arch = [], [], [], []
    for i in range(a.n):
        P, rec = draw_v2(a.master, f"draw{i}"); name = f"ind_{i:03d}"
        tree = v2.write_individual(P, out / "individuals" / name, None, synthetic_sampling=rec); s = summary(tree)
        main_lts = [lt for lt in tree["leaflets"] if lt["leaf"].startswith("L")]
        big = max(lt["length_mm"] for lt in main_lts if lt["type"] != "I")
        summ.append(dict(individual=name, synthetic=True, seed=P["seed"], n_truss=P["n_truss"], visible=P["visible"], lf_len_param=P["lf_len"], lw_ratio_param=P["lw_ratio"],
                         side_leaves=P["side_leaves"], side_axes=s["counts"]["side_axes"], compound_leaves=s["counts"]["leaves"], leaflets=s["counts"]["leaflets"],
                         intercalary=s["counts"]["leaflets_I"], trusses=s["counts"]["trusses"], flowers=s["counts"]["flowers"], largest_main_leaflet_mm=round(big, 2),
                         clipped=sum("clipped_to_measured_max" in lt["flags"] for lt in tree["leaflets"]), plant_age_d=round(tree["plant"]["attrs"]["plant_age_d"], 1),
                         note="synthetic geometry, not a measurement"))
        leaf_rows += [dict(individual=name, object=lt["id"], leaf=lt["leaf"], kind=lt["type"], length_mm=lt["length_mm"], width_mm=lt["width_mm"], expansion=lt["expansion"]) for lt in tree["leaflets"]]
        # architecture record (change 3)
        main_tr = [t for t in tree["trusses"] if t["axis"] == "M"]; side = [ax for ax in tree["axes"] if ax["kind"] == "side"]
        arch.append(dict(individual=name, flowered=tree["plant"]["attrs"]["flowered"], main_trusses=len(main_tr), side_axes=len(side),
                         side_leaves_max=max([sum(lf["axis"] == ax["id"] for lf in tree["leaves"]) for ax in side] or [0]),
                         side_trusses=sum(t["axis"] != "M" for t in tree["trusses"]), side_branches=sum(ax.get("parent_axis") not in (None, "M") for ax in tree["axes"])))
        if (v1dir / "individuals" / name / "plant.obj").exists():
            pairs.append((name, v1dir / "individuals" / name / "plant.obj", out / "individuals" / name / "plant.obj"))
    with open(out / "batch_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    with open(out / "batch_leaflets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(leaf_rows[0])); w.writeheader(); w.writerows(leaf_rows)
    json.dump(arch, open(out / "architecture_check.json", "w"), indent=1)
    contact_sheet(pairs[:20], out / "contact_sheet_v1_v2.png")
    big = np.array([s["largest_main_leaflet_mm"] for s in summ]); lw = np.array([x["length_mm"] / x["width_mm"] for x in leaf_rows]); nt = [s["n_truss"] for s in summ]
    rc = dict(note="range check against values quoted in the v1 page / prompt; not a fit",
              largest_leaflet_mm=dict(synthetic_mean=float(big.mean()), synthetic_sd=float(big.std(ddof=1)), synthetic_min=float(big.min()), synthetic_max=float(big.max()),
                                      measured="18.6 +- 3.1 (14.2-22.5)", share_within_measured_range=float(((big >= 14.2) & (big <= 22.5)).mean()), clipped_leaflets=int(sum(s["clipped"] for s in summ))),
              leaflet_lw_ratio=dict(synthetic_p05=float(np.percentile(lw, 5)), synthetic_median=float(np.median(lw)), synthetic_p95=float(np.percentile(lw, 95)), measured_range="1.0-2.2",
                                    share_within_measured_range=float(((lw >= 1.0) & (lw <= 2.2)).mean()), n=int(len(lw))),
              true_leaves_to_first_truss=dict(synthetic_counts={str(k): nt.count(k) for k in sorted(set(nt))}, measured_counts={"6": 2, "7": 1, "8": 2}),
              architecture=dict(all_flowered_main_two_trusses=all(x["main_trusses"] == 2 for x in arch if x["flowered"]),
                                side_leaves_max=max(x["side_leaves_max"] for x in arch), side_axes_without_truss=sum(x["side_axes"] - x["side_trusses"] for x in arch),
                                side_branches=sum(x["side_branches"] for x in arch)))
    json.dump(rc, open(out / "range_check.json", "w"), indent=1, ensure_ascii=False)
    # compare() self-test on the first individual
    t = json.loads((out / "individuals" / "ind_000" / "tree.json").read_text(encoding="utf-8")); self_ = compare(t, t)
    import copy
    t2 = copy.deepcopy(t); rm = t2["leaves"].pop(2)["id"]; t2["leaflets"] = [lt for lt in t2["leaflets"] if lt["leaf"] != rm]
    for n in t2["nodes"]:
        n["leaves"] = [x for x in n["leaves"] if x != rm]
    mv = next(lt for lt in t2["leaflets"] if lt["type"] == "P"); src = mv["leaf"]; dst = next(lf["id"] for lf in t2["leaves"] if lf["id"] != src)
    for lf in t2["leaves"]:
        for at in lf["attachments"]:
            if mv["id"] in at["leaflets"]:
                at["leaflets"].remove(mv["id"])
    mv["leaf"] = dst; next(lf for lf in t2["leaves"] if lf["id"] == dst)["attachments"][0]["leaflets"].append(mv["id"])
    pert = compare(t, t2)
    json.dump(dict(self=self_, one_leaf_removed_one_leaflet_moved=pert), open(out / "compare_selftest.json", "w"), indent=1)
    print(json.dumps(rc, indent=1, ensure_ascii=False)); print("self:", self_["node_count"], self_["matched_leaves"], self_["leaflet_membership_agreement"])
    print("perturbed:", pert["node_count"], "matched", pert["matched_leaves"], "unmatched_a", pert["unmatched_a"], "membership", round(pert["leaflet_membership_agreement"], 3))


if __name__ == "__main__":
    main()
