"""Part A: MTG_LEAFLET_ARRANGEMENT_v1.csv from the _03 split (leaflets.csv). python -X utf8 arrangement.py"""
import csv, hashlib, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; TR = ROOT / "build/traits_recon_20261101_01"
OUTF = TR / "MTG_LEAFLET_ARRANGEMENT_v1.csv"
L = list(csv.DictReader(open(HERE / "leaflets.csv", encoding="utf-8"))); S = [r for r in csv.DictReader(open(HERE / "leaflet_split_summary.csv", encoding="utf-8")) if r["status"] == "split"]
LAB = "leaflet_arrangement_from_reconstruction;leaflet_split_from_surfels;skeleton_labels_ungraded;conditional_on_unvalidated_scale;not_validation;split_amendment5_count_fixed_to_prior"
PHI = json.load(open(HERE / "synthetic_phi_check.json", encoding="utf-8"))
def quality(pos, q):
    if q == "spread_phi_deg" and pos == "T":
        return "not_reliable: real end leaflets lie across the clicked line (median ~97 deg) unlike every synthetic end leaflet (split vs truth |diff| 1.7 deg); axis direction base->centroid is ill-defined when the rachis runs through the leaflet"
    if q == "spread_phi_deg":
        return f"synthetic check (engine leaflets): split phi - true phi median {PHI['diff_median']:.1f} deg, |diff| median {PHI['abs_diff_median']:.1f}, p90 {PHI['abs_diff_p90']:.0f} (axis flips)"
    if q == "s_over_L" and pos == "T":
        return "end-leaflet base s = median s of its 5 % surfels nearest the clicked line (which runs through its midrib) -> > 1 by construction; use 1.0 for the attachment"
    return "ok"
Q = [("s_over_L", "base s / rachis length L (L = node -> end-leaflet base)"), ("spread_phi_deg", "angle leaflet long axis vs rachis tangent at base, in the compound-leaf plane"),
     ("petiolule_offset_mm", "distance leaflet base -> rachis curve"), ("tilt_deg", "angle leaflet normal vs compound-leaf plane normal"), ("roll_sign", "sign of roll (+1 / -1), mean = balance"),
     ("length_mm", "leaflet length (p1-p99 along long axis)"), ("width_mm", "leaflet width (p1-p99 across)"), ("area_share", "own area / sum of own leaflet areas in the compound")]


def stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0, mean="", sd="", median="", p1="", p10="", p25="", p75="", p90="", p99="")
    p = np.percentile(v, [1, 10, 25, 50, 75, 90, 99])
    return dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else 0.0, median=p[3], p1=p[0], p10=p[1], p25=p[2], p75=p[4], p90=p[5], p99=p[6])


rows = []
for subset, src, sel in (("all_11", "reconstruction_ungraded", lambda r: True), ("pilot_only", "reconstruction_ungraded_pilot_only", lambda r: r["plant"] == "R3-1b")):
    LL = [r for r in L if sel(r)]; SS = [r for r in S if sel(r)]
    for pos in ("P1", "P2", "P3", "T"):
        for side in (("L", "R", "both") if pos != "T" else ("C",)):
            g = [r for r in LL if (r["type"] == "T" if pos == "T" else (r["type"] == "P" and r["pair"] == pos[1] and (side == "both" or r["side"] == side)))]
            for q, desc in Q:
                rows.append(dict(subset=subset, rank_class="main", position=pos, side=side, quantity=q, **stats([float(r[q]) for r in g]), description=desc, quality=quality(pos, q), source=src, labels=LAB))
    rows.append(dict(subset=subset, rank_class="main", position="leaf", side="", quantity="within_leaf_overlap", **stats([float(r["within_leaf_overlap"]) for r in SS]),
                    description="top-view (xy) overlap of the leaflets' alpha meshes / their summed projected area, 0.25 mm raster", quality="ok", source=src, labels=LAB))
    rows.append(dict(subset=subset, rank_class="main", position="leaf", side="", quantity="rachis_L_mm", **stats([float(r["L_mm"]) for r in SS]), description="rachis length L_r = node -> end-leaflet base s_T (all ranks)", quality="ok", source=src, labels=LAB))
    for k in sorted({int(r["k"]) for r in SS}):
        rows.append(dict(subset=subset, rank_class="main", position=f"leaf_rank_{k}", side="", quantity="rachis_L_mm", **stats([float(r["L_mm"]) for r in SS if int(r["k"]) == k]), description=f"rachis length L_r, leaf rank {k}", quality="ok", source=src, labels=LAB))
    rows.append(dict(subset=subset, rank_class="main", position="leaf", side="", quantity="leaflets_per_compound_split", **stats([float(r["n_leaflets"]) for r in SS]), description="leaflets per compound in the split (amendment 5: fixed to the prior = hand mode by rank; not a measurement)", quality="not_a_measurement", source=src, labels=LAB))
    for q, _ in Q:   # side axes: no observed side-axis leaves
        rows.append(dict(subset=subset, rank_class="side", position="all", side="", quantity=q, **stats([]), description="no observed side-axis compound leaves in the 11 plants", quality="no_data", source=src, labels=LAB))
assert not OUTF.exists(), "arrangement v1 already exists"
with open(OUTF, "w", newline="", encoding="utf-8") as f:
    f.write(f"# MTG_LEAFLET_ARRANGEMENT_v1; from build/mtg_build_v2_clicks_leaflets_20261105_01 (_03 split, amendment 5: clicks reach the end-leaflet tip, leaflets per side fixed to the prior, synthetic gate passed 86 % / 13.3 %); {len(L)} leaflets / {len(S)} compound leaves / 11 plants; labels {LAB}; for the dynamic engine's draws only (static MTG uses each plant's own leaflets)\n")
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
h = hashlib.sha256(OUTF.read_bytes()).hexdigest(); (TR / "MTG_LEAFLET_ARRANGEMENT_v1.sha256").write_text(f"{h}  MTG_LEAFLET_ARRANGEMENT_v1.csv\n", encoding="utf-8")
# figures
(HERE / "figures").mkdir(exist_ok=True)
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for a, (q, desc) in zip(ax.ravel(), Q):
    data = [[float(r[q]) for r in L if (r["type"] == "T" if p == "T" else (r["type"] == "P" and r["pair"] == p[1]))] for p in ("P1", "P2", "P3", "T")]
    a.boxplot(data, tick_labels=["P1", "P2", "P3", "T"], showfliers=True); a.set_title(q, fontsize=10); a.grid(alpha=0.3)
fig.suptitle(f"MTG_LEAFLET_ARRANGEMENT_v1: {len(L)} leaflets, {len(S)} compound leaves, 11 plants (reconstruction_ungraded)"); fig.tight_layout(); fig.savefig(HERE / "figures/arrangement_distributions.png", dpi=110)
print(json.dumps(dict(file=str(OUTF.relative_to(ROOT)).replace("\\", "/"), sha256=h, rows=len(rows), leaflets=len(L), leaves=len(S)), indent=1))
for r in rows:
    if r["subset"] == "all_11" and r["side"] in ("both", "C", "") and r["rank_class"] == "main" and r["quantity"] in ("s_over_L", "spread_phi_deg", "area_share", "within_leaf_overlap", "petiolule_offset_mm", "tilt_deg"):
        print(r["position"], r["quantity"], r["n"], round(r["median"], 3) if r["n"] else "")
