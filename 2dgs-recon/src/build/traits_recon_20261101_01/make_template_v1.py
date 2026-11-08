"""MTG_TEMPLATE_v1 (long format, frozen): hand-measured MTG-building data (Amendments 1-2d) + reconstruction traits that passed section 3.
Exclusions while unanswered: t2_02/t2_04/t2_05 (validation membership), no hand-vs-recon per-leaf pooling (ID map)."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
HL = "hand_measured;user_confirmed_2026-11-01;mtg_building_not_validation"
RL = "recon_trait;conditional_on_unvalidated_scale;after_checkpoint_v1;age_dag=60;user_stated"
leaf = pd.read_csv(D / "traits_per_leaf.csv"); th = pd.read_csv(D / "hand_thickness.csv"); sa = pd.read_csv(D / "hand_side_axes.csv")
sl = pd.read_csv(D / "hand_side_leaves.csv"); tr = pd.read_csv(D / "hand_trusses.csv"); fr = pd.read_csv(D / "hand_fruits.csv")
plant = pd.read_csv(D / "traits_per_plant.csv"); gate = json.load(open(D / "kaggle_out/out/gate.json"))["gate"]["primary"]
R = []
def add(organ, rank, param, unit, value, sd=None, n_plants=None, n_items=None, prov="hand_measured", labels=HL, note=""):
    R.append(dict(organ=organ, rank=rank, parameter=param, unit=unit, value=value, sd=sd, n_plants=n_plants, n_items=n_items, provenance=prov, labels=labels, note=note))

for k, s in leaf.groupby("rank"):
    np_, ni = s.plant.nunique(), len(s); note = "rank 1 kept separate" if k == 1 else ""
    add("main_compound_leaf", int(k), "theta0_base_above_horizontal", "deg", round(s.theta0_deg.mean(), 2), round(s.theta0_deg.std(), 2), np_, ni, note=note)
    add("main_compound_leaf", int(k), "insertion_from_vertical (engine COMPOUND_INSERTION_DEG = 90 - theta0)", "deg", round(90 - s.theta0_deg.mean(), 2), round(s.theta0_deg.std(), 2), np_, ni)
    for m in (5, 6, 7):
        add("main_compound_leaf", int(k), f"p_leaflets_{m}", "fraction", round(float((s.n_leaflets == m).mean()), 3), None, np_, ni)
    t = th[th.main_rank == k].mm
    if len(t):
        add("main_stem", int(k), "stem_thickness", "mm", round(t.mean(), 2), round(t.std(), 2) if len(t) > 1 else None, th[th.main_rank == k].plant.nunique(), len(t), note="incl. realigned cells (rule 6)")
add("main_compound_leaf", "all", "theta0_base_above_horizontal", "deg", round(leaf.theta0_deg.mean(), 2), round(leaf.theta0_deg.std(), 2), leaf.plant.nunique(), len(leaf))
for m in (5, 6, 7):
    add("main_compound_leaf", "all", f"p_leaflets_{m}", "fraction", round(float((leaf.n_leaflets == m).mean()), 3), None, leaf.plant.nunique(), len(leaf))
npl = leaf.groupby("plant").size()
add("main_stem", "all", "compound_leaves_per_main_stem", "count", round(npl.mean(), 2), round(npl.std(), 2), len(npl), int(npl.sum()), note="per plant " + ", ".join(f"{p} {v}" for p, v in npl.items()) + "; dt04 leaf 8 not measured")
add("side_axis", "all", "side_axes_per_plant", "count", round(sa.groupby("plant").size().reindex(npl.index, fill_value=0).mean(), 2), None, len(npl), len(sa))
add("side_axis", "all", "attach_main_rank_counts", "json", json.dumps({str(int(k)): int(v) for k, v in sa.main_rank.dropna().value_counts().sort_index().items()}), None, sa.plant.nunique(), len(sa), note=f"{int(sa.main_rank.isna().sum())} attach_node_unknown")
add("side_axis", "all", "angle_from_horizontal_values", "deg", json.dumps(sa.angle.dropna().tolist()), None, int(sa[sa.angle.notna()].plant.nunique()), int(sa.angle.notna().sum()), note="only 2 recorded; sparse")
add("side_axis_compound_leaf", "all", "theta0_base_above_horizontal", "deg", round(sl.theta.mean(), 2), round(sl.theta.std(), 2), sl.plant.nunique(), len(sl))
for m in (5, 7):
    add("side_axis_compound_leaf", "all", f"p_leaflets_{m}", "fraction", round(float((sl.n == m).mean()), 3), None, sl.plant.nunique(), len(sl))
add("side_axis_compound_leaf", "all", "leaves_per_side_axis_max", "count", int(sl.groupby(["plant", "k"]).size().max()), None, sl.plant.nunique(), len(sl))
fl = [int(f) for n_, f in zip(tr.n_truss, tr.flowers_each) for _ in range(int(n_))]
add("truss", "all", "flowers_per_truss", "count", round(float(np.mean(fl)), 2), round(float(np.std(fl, ddof=1)), 2), tr.plant.nunique(), len(fl), note="node position not confirmed")
add("truss", "all", "trusses_per_plant", "count", round(tr.groupby("plant").n_truss.sum().reindex(npl.index, fill_value=0).mean(), 2), None, len(npl), int(tr.n_truss.sum()))
add("fruit", "all", "fruits_per_plant", "count", round(fr.groupby("plant").n_fruit.sum().reindex(npl.index, fill_value=0).mean(), 2), None, len(npl), int(fr.n_fruit.sum()), note="node position not confirmed")
add("plant", "all", "age_at_photographing", "DAG", 60, None, None, None, prov="user_stated", labels="user_stated")

use = plant[~plant.plant.isin(["t2_02", "t2_04", "t2_05"])]
for col, key, unit in [("A2_height_p995_mm", "height", "mm"), ("A2_width_x_mm", "width_x", "mm"), ("A2_width_y_mm", "width_y", "mm")]:
    ok = gate[key]["passed"]; n = len(use)
    add("plant", "all", "recon_" + col, unit, round(use[col].mean(), 2), round(use[col].std(), 2), n, n,
        prov=("recon_measured" if n >= 3 else "recon_sparse") if ok else "trait_unreliable", labels=RL,
        note=f"section 3 rel bias {gate[key]['rel_bias']:+.3f}; plants {', '.join(use.plant)}" + ("; h_from_z_min (base = lowest surfel)" if key == "height" else "") + "; t2_* excluded (validation unanswered)")
for param, why in [("A1 LAD / A3 azimuth / C1 curl / C3 inclination by rank", "trait_unreliable (section 3: normal noise 42.9 deg)"),
                   ("A2 projected canopy area", "trait_unreliable (section 3: +13 %)"),
                   ("B1 divergence, B2 total bend dtheta / insertion from skeleton, B3 internode / leaf length", "missing: extractors passed section 3 but real skeletons await 8835 confirmation"),
                   ("hand vs recon theta0 pooling", "not_possible: dt01-dt06 -> recon ID map not given")]:
    add("not_in_template", "all", param, "", "", prov=why.split(" ")[0].rstrip(":"), labels="", note=why)
T = pd.DataFrame(R)
out = D / "MTG_TEMPLATE_v1.csv"; T.to_csv(out, index=False)
h = hashlib.sha256(out.read_bytes()).hexdigest()
(D / "MTG_TEMPLATE_v1.sha256").write_text(f"{h}  MTG_TEMPLATE_v1.csv\n")
print(len(T), h)
