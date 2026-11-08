"""Eye check: where do the rachis clicks end relative to the leaf's surfels? Top view (leaf frame) of several leaves, surfels coloured by split leaflet, clicks + end point."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run_split as RS, split_lib as SL  # noqa: E402

CASES = [("n_01", 0), ("n_01", 1), ("n_01", 2), ("n_01", 3), ("n_04", 5), ("e_05", 6), ("n_05", 4), ("e_01", 5)]
cache = {}
fig, axes = plt.subplots(2, 4, figsize=(20, 10)); cols = plt.cm.tab10.colors
for ax, (pid, i) in zip(axes.ravel(), CASES):
    if pid not in cache:
        D = RS.FP.load_real(pid); spec = json.load(open(RS.V02 / f"static/mtg_{pid}_spec.json", encoding="utf-8")); X, A, N, rows, _ = RS.surfel_data(pid, D); cache[pid] = (D, spec, X, A, N)
    D, spec, X, A, N = cache[pid]; lf = spec["leaves"][i]; idx = D["leaf_idx"][lf["uid"]]; c, _ = RS.curve_of(lf)
    r = SL.split_leaf(X[idx], A[idx], N[idx], c, lf["leaflets"])
    P = X[idx]; ctr = P.mean(0); w, v = np.linalg.eigh(np.cov((P - ctr).T)); e1, e2 = v[:, 2], v[:, 1]     # leaf plane view
    pr = lambda Z: np.c_[(Z - ctr) @ e1, (Z - ctr) @ e2]
    lab = r["labels"]; Pp = pr(P)
    ax.scatter(Pp[lab < 0, 0], Pp[lab < 0, 1], s=1, c="0.6", label="rachis band")
    for q, l in enumerate(r["leaflets"]):
        m = lab == q; nm = f"{l['type']}{l['pair'] if l['type'] == 'P' else ''}{l['side'] if l['type'] == 'P' else ''}"
        ax.scatter(Pp[m, 0], Pp[m, 1], s=1, color=cols[q % 10], label=f"{nm} {l['area_own']:.0f} mm²")
    Cp = pr(np.asarray(lf.get("rachis_curve_from_clicks_mm") or c)); ax.plot(Cp[:, 0], Cp[:, 1], "k-", lw=1.5)
    ax.plot(Cp[-1, 0], Cp[-1, 1], "r*", ms=14, label="last click (P_end)", zorder=5)
    ax.set_aspect("equal"); ax.set_title(f"{lf['uid']}: L {r['summary']['L']:.0f} mm, end leaflet {100 * sum(l["area_own"] for l in r["leaflets"] if l["type"] == "T") / r["summary"]["total_area"]:.0f} % of area", fontsize=10)
    lg = ax.legend(fontsize=7, loc="best"); [h.set_sizes([20]) for h in lg.legend_handles if hasattr(h, "set_sizes")]; [h.set_markersize(8) for h in lg.legend_handles if hasattr(h, "set_markersize")]; ax.grid(alpha=0.2)
fig.suptitle("Leaflet split, amendment 5 (clicks reach the end-leaflet tip; count = prior). Leaf-plane view, mm. Black = clicked rachis, red star = last click.", fontsize=12)
fig.tight_layout(); (HERE / "figures").mkdir(exist_ok=True); fig.savefig(HERE / "figures/split_a5_check.png", dpi=100); print("ok")
