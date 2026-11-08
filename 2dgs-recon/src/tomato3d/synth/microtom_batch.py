"""Batch of Micro-Tom synthetic individuals (contract build/p1_synth_microtom_20261022_01/CONTRACT.md). Synthetic only.

usage: python -m tomato3d.synth.microtom_batch --out build/p1_synth_microtom_20261022_01 [--n 40] [--master 20261022]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from tomato3d.synth import microtom as mt

STAGE_N = 10
LIT_TRUSS = (5, 7, 9)
LIT_EACH = 3


def draw(master: int, key: str, n_truss: int | None = None) -> tuple[dict, dict]:
    """The page's "새 개체 뽑기" with the Date.now() key replaced by a declared key; same draw order."""
    r = mt.rng_for(master, key); P = mt.params_for(None)
    P["seed"] = 1 + math.floor(r() * 9998)
    u = r(); P["n_truss"] = 6 if u < 0.4 else (7 if u < 0.6 else 8)
    if n_truss is not None:
        P["n_truss"] = n_truss                                   # literature arm: n_truss fixed, draw kept for order
    P["visible"] = P["n_truss"]
    P["internode"] = float(mt.js_fixed(mt.clamp(7 + 1.0 * mt.gauss(r), 5, 10), 1))
    P["lf_len"] = float(mt.js_fixed(mt.clamp(18.6 + 3.1 * mt.gauss(r), 12, 26), 1))
    P["lw_ratio"] = float(mt.js_fixed(mt.clamp(1.5 + 0.15 * mt.gauss(r), 1.1, 2.0), 2))
    P["side_leaves"] = math.floor(r() * 4)
    rec = dict(master_seed=master, draw_key=key, rule="page '새 개체 뽑기' draw order; Date.now() key replaced by draw_key",
               tiers={"n_truss": "user_measured" if n_truss is None else "literature", "internode": "user_measured", "lf_len": "user_measured",
                      "lw_ratio": "user_measured", "side_leaves": "literature", "visible": "assumption", "seed": "assumption", "others": "page defaults"})
    return P, rec


def summarise(name: str, P: dict, res: dict, arm: str) -> dict:
    s = res["stats"]
    main = [o for o in res["objs"] if o["c"].startswith("blade_") and not o["leaf"].startswith("S1_")]
    lf = [o["info"]["length_mm"] for o in main]
    return dict(individual=name, arm=arm, synthetic=True, seed=P["seed"], n_truss=P["n_truss"], visible=P["visible"], internode=P["internode"],
                lf_len_param=P["lf_len"], lw_ratio_param=P["lw_ratio"], side_leaves=P["side_leaves"], compound_leaves=s["leaves"],
                leaflets=s["leaflets"], flowers=s["flowers"], height_mm=round(s["height"], 2), width_mm=round(s["width"], 2),
                leaflet_area_one_side_mm2=round(s["area"], 1), largest_main_leaflet_mm=round(max(lf), 2) if lf else None,
                note="synthetic geometry, not a measurement")


def leaflet_rows(res: dict) -> list[dict]:
    return [dict(object=o["n"], leaf=o["leaf"], kind=o["info"]["kind"], length_mm=o["info"]["length_mm"], width_mm=o["info"]["width_mm"])
            for o in res["objs"] if o["c"].startswith("blade_")]


def contact_sheet(items: list[tuple[str, dict]], path: Path, cols: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    COL = dict(stem="#c98a2b", rachis="#7d9a4a", blade_P="#3f8a45", blade_T="#2f7552", blade_I="#9fbc3a", petiolule="#8aa362",
               flower="#e3c22c", truss="#7d9a4a", cotyledon="#88b06a")
    rows = math.ceil(len(items) / cols); fig, axs = plt.subplots(rows * 2, cols, figsize=(cols * 1.9, rows * 3.8))
    axs = np.atleast_2d(axs)
    for k, (name, res) in enumerate(items):
        r, c = divmod(k, cols)
        tris, cols_, depth_top, depth_side = [], [], [], []
        for o in res["objs"]:
            V = np.array(o["v"])
            X = np.c_[V[:, 0], -V[:, 2], V[:, 1]]              # OBJ frame: z up
            for f in o["f"]:
                tris.append(X[f]); cols_.append(COL.get(o["c"], "#888"))
        T = np.array(tris); C = np.array(cols_)
        for row_off, (a, b, dep, title) in enumerate(((0, 1, 2, "top"), (0, 2, 1, "side"))):
            ax = axs[r * 2 + row_off, c]
            order = np.argsort(T[:, :, dep].mean(1)) if title == "top" else np.argsort(-T[:, :, dep].mean(1))
            ax.add_collection(PolyCollection(T[order][:, :, [a, b]], facecolors=C[order], edgecolors="none"))
            ax.set_xlim(-90, 90); ax.set_ylim(-90, 90) if title == "top" else ax.set_ylim(-5, 175)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{name} {title}", fontsize=6)
    for k in range(len(items), rows * cols):
        r, c = divmod(k, cols); axs[r * 2, c].axis("off"); axs[r * 2 + 1, c].axis("off")
    fig.suptitle("SYNTHETIC Micro-Tom individuals (microtom_synth_v1) - top / side, mm grid +-90 x 0-175", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--master", type=int, default=20261022)
    a = ap.parse_args(argv); out = Path(a.out); summ, leaf_rows, sheet = [], [], []
    batch = []
    for i in range(a.n):
        P, rec = draw(a.master, f"draw{i}"); name = f"ind_{i:03d}"
        res = mt.write_individual(P, out / "individuals" / name, None, synthetic_sampling=rec)
        summ.append(summarise(name, P, res, "measured_batch")); leaf_rows += [dict(individual=name, arm="measured_batch", **x) for x in leaflet_rows(res)]
        sheet.append((name, res)); batch.append(P)
    for i, P0 in enumerate(batch[:STAGE_N]):
        for v in range(2, P0["n_truss"] + 1):
            P = dict(P0, visible=v); name = f"ind_{i:03d}_v{v}"
            res = mt.write_individual(P, out / "stage" / name, None, synthetic_sampling=dict(stage_of=f"ind_{i:03d}", visible=v))
            summ.append(summarise(name, P, res, "stage_series"))
    for n in LIT_TRUSS:
        for i in range(LIT_EACH):
            P, rec = draw(a.master, f"lit{n}_{i}", n_truss=n); name = f"lit_{n}_{i}"
            res = mt.write_individual(P, out / "literature" / name, None, synthetic_sampling=rec)
            summ.append(summarise(name, P, res, "literature_arm"))
    with open(out / "batch_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    with open(out / "batch_leaflets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(leaf_rows[0])); w.writeheader(); w.writerows(leaf_rows)
    contact_sheet(sheet, out / "contact_sheet.png")
    # ---- range check vs measured values as quoted in the page / prompt (not a fit)
    B = [s for s in summ if s["arm"] == "measured_batch"]; big = np.array([s["largest_main_leaflet_mm"] for s in B])
    lw = np.array([x["length_mm"] / x["width_mm"] for x in leaf_rows]); nt = [s["n_truss"] for s in B]
    rc = dict(source="values quoted in the reference page / prompt; source docs 소엽형상_기준세트 and 실측_주경길이_복엽수 not on disk",
              largest_leaflet_mm=dict(synthetic_mean=float(big.mean()), synthetic_sd=float(big.std(ddof=1)), synthetic_min=float(big.min()), synthetic_max=float(big.max()),
                                      measured="18.6 +- 3.1 (14.2-22.5)", share_within_measured_range=float(((big >= 14.2) & (big <= 22.5)).mean())),
              leaflet_lw_ratio=dict(synthetic_p05=float(np.percentile(lw, 5)), synthetic_median=float(np.median(lw)), synthetic_p95=float(np.percentile(lw, 95)),
                                    synthetic_min=float(lw.min()), synthetic_max=float(lw.max()), measured_range="1.0-2.2",
                                    share_within_measured_range=float(((lw >= 1.0) & (lw <= 2.2)).mean()), n=int(len(lw))),
              true_leaves_to_first_truss=dict(synthetic_counts={str(k): nt.count(k) for k in sorted(set(nt))}, synthetic_mean=float(np.mean(nt)),
                                              measured_counts={"6": 2, "7": 1, "8": 2}, measured_mean=7.0))
    json.dump(rc, open(out / "range_check.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rc, indent=1, ensure_ascii=False))
    print("individuals", len(summ))


if __name__ == "__main__":
    main()
