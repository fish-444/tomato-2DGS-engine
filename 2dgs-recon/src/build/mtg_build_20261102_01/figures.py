"""Figures (main prompt §6): each static MTG plant drawn over its surfels (2 views), the template-mean plant, 4 check plants, internode Beta."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as Beta

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import mtg_spec as MS  # noqa: E402


def surfels(pid):
    if pid == "R3-1b":
        z = np.load(ROOT / "build/engine_coplanar_sep_20261101_01/out_k1/per_surfel_D.npz"); return np.asarray(z["xyz"], float) * (1000 if np.abs(z["xyz"]).max() < 10 else 1)
    g = np.load(ROOT / f"build/p1_all_3dfirst_20261027_01/{pid}/merge/groups.npz"); X = np.asarray(g["xyz"], float); return X * (1000 if np.abs(X).max() < 10 else 1)


def draw_tree(ax, tree, proj, lw=1.2):
    i, j = proj
    for a in tree["axes"]:
        P = np.asarray(a["tube"]["polyline"], float); ax.plot(P[:, i], P[:, j], "-", color="saddlebrown" if a["kind"] == "main" else "peru", lw=2.2)
    for lf in tree["leaves"]:
        P = np.asarray(lf["rachis"]["polyline"], float); ax.plot(P[:, i], P[:, j], "-", color="darkgreen", lw=lw)
        ax.text(P[-1, i], P[-1, j], lf["id"].replace("L0", "L").replace("_L", "/"), fontsize=5, color="k")
    for lt in tree["leaflets"]:
        b = np.asarray(lt["base"], float); d = np.asarray(lt["frame"]["d"], float) * lt["length_mm"]; bb = np.asarray(lt["frame"]["b"], float) * lt["width_mm"] / 2
        poly = np.array([b, b + d * 0.5 + bb, b + d, b + d * 0.5 - bb]); ax.fill(poly[:, i], poly[:, j], color="limegreen", alpha=0.35, lw=0)


def plant_figure(name, tree, S, title):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, proj, lab in ((axs[0], (0, 2), "x-z (side)"), (axs[1], (0, 1), "x-y (top)")):
        if S is not None:
            ax.scatter(S[:, proj[0]], S[:, proj[1]], s=0.3, c="lightgray", alpha=0.5, rasterized=True)
        draw_tree(ax, tree, proj); ax.set_aspect("equal"); ax.set_title(f"{title} — {lab}", fontsize=9); ax.set_xlabel("mm")
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=130); plt.close(fig)


def main():
    for pid in MS.PLANTS:
        tree = json.load(open(HERE / f"static/mtg_{pid}_engine.plant_tree_v1.json", encoding="utf-8"))
        try:
            S = surfels(pid)
        except Exception as e:  # noqa: BLE001
            S = None; print(pid, "no surfels", e)
        plant_figure(f"static_{pid}_over_surfels", tree, S, f"{pid} static MTG (theta0 hand by rank, dtheta 32.8 deg) over surfels; grade {MS.GRADE[pid]}")
    tree = json.load(open(HERE / "static/mtg_template_mean_v2draft_engine.plant_tree_v1.json", encoding="utf-8")); plant_figure("template_mean_v2draft", tree, None, "template-mean plant (v2 draft, no randomness)")
    for m in range(1, 5):
        tree = json.load(open(HERE / f"dynamic/check_plants/P{m:02d}_engine.plant_tree_v1.json", encoding="utf-8")); plant_figure(f"check_plant_P{m:02d}", tree, None, f"check plant P{m:02d} (seed {20261102 + m - 1}, rules v1)")
    ib = next(csv.DictReader(open(HERE / "dynamic/internode_range_setting.csv", encoding="utf-8"))); a, b = float(ib["a"]), float(ib["b"])
    x = np.linspace(2, 4, 400); pdf = Beta.pdf((x - 2) / 2, a, b) / 2
    dr = [float(r["value"]) for r in csv.DictReader(open(HERE / "dynamic/random_draws.csv", encoding="utf-8")) if r["key"] == "INTERNODE_MM"]
    fig, ax = plt.subplots(figsize=(6, 3.6)); ax.plot(x, pdf, "k-", label=f"scaled Beta a={a:.3f}, b={b:.3f} on [2,4]"); ax.hist(dr, bins=20, range=(2, 4), density=True, alpha=0.4, label=f"{len(dr)} keyed draws (16 check plants)")
    ax.axvline(3.75, color="r", ls="--", label="median 3.75 (user setting)"); ax.axvline(float(ib["mean_mm"]), color="b", ls=":", label=f"mean {float(ib['mean_mm']):.3f}")
    ax.set_xlabel("main-axis internode length (mm)"); ax.set_ylabel("density"); ax.legend(fontsize=7); ax.set_title("internode setting: user_set_range_2.00_4.00_median_3.75 (not measured)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "internode_beta.png", dpi=130); plt.close(fig)
    print("figures", len(list(FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
