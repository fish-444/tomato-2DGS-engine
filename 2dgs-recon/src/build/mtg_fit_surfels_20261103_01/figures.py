"""Eye-check figures (CONTRACT §5): per plant, side x-z and top x-y over grey surfels; before = original spec geometry (dashed), after = fitted (solid);
stem brown, leaf labels L1... Leaf geometry drawn with the old build's writer (spec_to_tree) for both, Level-2 polylines where accepted."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fit_plant as FP  # noqa: E402
from fit_lib import W  # noqa: E402

PLANTS = ["R3-1b", "n_01", "n_02", "n_03", "n_04", "n_05", "n_06", "e_01", "e_04", "e_05", "e_07"]


def tree_of(spec):
    q = [np.asarray(lf["rachis_polyline_level2_mm"], float) if lf.get("rachis_polyline_level2_mm") else None for lf in spec["leaves"]]; it = iter(q); orig = W.leaf_polyline
    def pat(base, az, th, dt, L, m):
        x = next(it, None); return orig(base, az, th, dt, L, m) if x is None else ([p.tolist() for p in x], None)
    W.leaf_polyline = pat
    try:
        return W.spec_to_tree(spec)
    finally:
        W.leaf_polyline = orig


def draw(ax, tree, proj, ls, alpha):
    i, j = proj
    for a in tree["axes"]:
        P = np.asarray(a["tube"]["polyline"], float); ax.plot(P[:, i], P[:, j], ls, color="saddlebrown", lw=2.4, alpha=alpha)
    for lf in tree["leaves"]:
        P = np.asarray(lf["rachis"]["polyline"], float); ax.plot(P[:, i], P[:, j], ls, color="darkgreen", lw=1.3, alpha=alpha)
        if ls == "-":
            ax.text(P[-1, i], P[-1, j], "L" + str(int(lf["id"][1:])), fontsize=6)


def main():
    for pid in PLANTS:
        old = json.load(open(FP.OLD / f"static/mtg_{pid}_spec.json", encoding="utf-8")); new = json.load(open(HERE / f"fitted/mtg_{pid}_spec_fitted.json", encoding="utf-8"))
        X = FP.load_real(pid)["X"]; X = X[np.random.default_rng(0).choice(len(X), min(40000, len(X)), replace=False)]
        t0, t1 = tree_of(old), tree_of(new)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        for ax, proj, lab in ((axs[0], (0, 2), "x-z (side)"), (axs[1], (0, 1), "x-y (top)")):
            ax.scatter(X[:, proj[0]], X[:, proj[1]], s=0.3, c="lightgray", rasterized=True); draw(ax, t0, proj, "--", 0.6); draw(ax, t1, proj, "-", 1.0)
            ax.set_aspect("equal"); ax.set_xlabel("mm"); ax.set_title(f"{pid}: before (dashed) / fitted (solid) — {lab}", fontsize=9)
        st = new["fit"]["stem"]; n_fit = sum(1 for lf in new["leaves"] if lf.get("fit_status") == "fitted"); n_fail = sum(1 for lf in new["leaves"] if lf.get("fit_status") == "fit_failed_kept_original")
        fig.suptitle(f"{pid} fitted_to_surfels (ungraded labels; geometry, not traits) - leaves fitted {n_fit}, kept {n_fail}, stem shift median {st.get('shift_median', 0):.1f} mm"
                     + (" - DISPLAY ONLY" if "display_only_synthetic_check_not_met" in new["labels"] else ""), fontsize=9)
        fig.tight_layout(); fig.savefig(HERE / f"figures/fit_{pid}.png", dpi=130); plt.close(fig)
    print("figures", len(list((HERE / "figures").glob("fit_*.png"))))


if __name__ == "__main__":
    main()
