"""Amendment 2: re-read the irregular hand cells by the user's rules 1-8 and rebuild the hand-only MTG template draft.
Every cell decision is listed explicitly below (excel row, column, raw value as asserted against the workbook copy)."""
import hashlib, json
from pathlib import Path
import openpyxl
import pandas as pd

D = Path(__file__).resolve().parent
ws = openpyxl.load_workbook(D / "inputs/hand_leaf_measurements_t1_dt01-06_2026-11-01.xlsx", data_only=True).active
cell = lambda r, c: ws[f"{c}{r}"].value
LAB = "hand_measured;user_confirmed_2026-11-01"

# (row, col, raw, kind, plant, main_rank, payload, rule, flags)
C = [
    (10, "E", 1, "side_axis", "dt01", 2, dict(k=1), "1", "angle_not_recorded"),
    (15, "E", 2, "side_axis", "dt01", 7, dict(k=2), "1", "angle_not_recorded"),
    (17, "E", 3, "side_axis", "dt01", 9, dict(k=3), "1", "angle_not_recorded"),
    (11, "G", 3, "truss", "dt01", 3, dict(n_truss=1, flowers_each=3), "answer_23:16", "single_number_in_truss_column=flowers|position_not_confirmed"),
    *[(r, "I", v, "thickness", "dt01", k, dict(mm=v), "header", "header_aligned") for r, k, v in [(9, 1, 9.7), (10, 2, 9.1), (11, 3, 9), (12, 4, 8.8)]],
    (19, "E", "1,20", "side_axis", "dt02", 1, dict(k=1, angle=20), "1", ""),
    (19, "F", "1,5,20", "side_leaf", "dt02", 1, dict(k=1, n=5, theta=20), "3", ""),
    (19, "G", "2,5,20", "side_leaf", "dt02", 1, dict(k=2, n=5, theta=20), "3", "side_axis_2_not_in_same_row"),
    (19, "H", 3, "pending", "dt02", 1, {}, "-", "held_meaning_not_stated_by_user"),
    *[(r, "I", v, "thickness", "dt02", k, dict(mm=v), "header", "header_aligned") for r, k, v in [(19, 1, 9), (20, 2, 6.8), (21, 3, 4.5), (22, 4, 4.5), (23, 5, 4.3), (24, 6, 3.3)]],
    (22, "E", "1,60", "side_axis", "dt02", 4, dict(k=1, angle=60), "1", "k_repeated_in_plant"),
    (25, "E", 1, "side_axis", "dt02", 7, dict(k=1), "1", "angle_not_recorded|k_repeated_in_plant"),
    (26, "H", 3, "pending", "dt02", 8, {}, "-", "held_meaning_not_stated_by_user"),
    (27, "E", 2, "side_axis", "dt02", 9, dict(k=2), "1", "angle_not_recorded"),
    (29, "H", 8.1, "thickness", "dt03", 1, dict(mm=8.1), "6", "realigned"),
    (30, "H", 5.4, "thickness", "dt03", 2, dict(mm=5.4), "6", "realigned"),
    (33, "H", 4.3, "thickness", "dt03", 5, dict(mm=4.3), "6", "realigned"),
    (33, "E", 1, "side_axis", "dt03", 5, dict(k=1), "1", "angle_not_recorded"),
    (33, "F", "7,5", "side_leaf", "dt03", 5, dict(k=1, n=7, theta=5), "2", "assigned_by_row_order"),
    (34, "F", "7,5", "side_leaf", "dt03", 6, dict(k=1, n=7, theta=5), "2", "assigned_by_row_order"),
    (35, "F", "5,-5", "side_leaf", "dt03", 7, dict(k=1, n=5, theta=-5), "2", "assigned_by_row_order"),
    (36, "E", 2, "side_axis", "dt03", None, dict(k=2), "1", "angle_not_recorded|attach_node_unknown"),
    (36, "F", "5,10", "side_leaf", "dt03", None, dict(k=2, n=5, theta=10), "2", "assigned_by_row_order"),
    (36, "H", "4,4", "truss", "dt03", None, dict(n_truss=2, flowers_each=4), "4", "position_not_confirmed"),
    (37, "F", "5,10", "side_leaf", "dt03", None, dict(k=2, n=5, theta=10), "2", "assigned_by_row_order"),
    (40, "E", 1, "side_axis", "dt04", 2, dict(k=1), "1", "angle_not_recorded"),
    (40, "F", "5,10", "side_leaf", "dt04", 2, dict(k=1, n=5, theta=10), "2", "assigned_by_row_order"),
    (41, "E", 4.2, "thickness", "dt04", 3, dict(mm=4.2), "6", "realigned|two_values_same_row"),
    (41, "F", "5,10", "side_leaf", "dt04", 3, dict(k=1, n=5, theta=10), "2", "assigned_by_row_order"),
    (41, "G", 1, "truss", "dt04", 3, dict(n_truss=1, flowers_each=1), "answer_23:16", "single_number_in_truss_column=flowers|position_not_confirmed"),
    (41, "H", 4, "pending", "dt04", 3, {}, "-", "held_meaning_not_stated_by_user"),
    (41, "J", 8.7, "thickness", "dt04", 3, dict(mm=8.7), "6", "realigned|two_values_same_row"),
    (42, "F", "5,30", "side_leaf", "dt04", 4, dict(k=1, n=5, theta=30), "2", "assigned_by_row_order"),
    (42, "J", 8.3, "thickness", "dt04", 4, dict(mm=8.3), "6", "realigned"),
    (43, "F", "5,30", "side_leaf", "dt04", 5, dict(k=1, n=5, theta=30), "2", "assigned_by_row_order"),
    (44, "H", 4, "pending", "dt04", 6, {}, "-", "held_meaning_not_stated_by_user"),
    (44, "J", 5.3, "thickness", "dt04", 6, dict(mm=5.3), "6", "realigned"),
    (45, "H", 4, "pending", "dt04", 7, {}, "-", "held_meaning_not_stated_by_user"),
    (45, "J", 4.2, "thickness", "dt04", 7, dict(mm=4.2), "6", "realigned"),
    (46, "C", 8, "omitted", "dt04", 8, {}, "7", "not_measured_left_empty"),
    (53, "E", 1, "side_axis", "dt05", 5, dict(k=1), "1", "angle_not_recorded"),
    (53, "G", "5,30", "side_leaf", "dt05", 5, dict(k=1, ordinal=1, n=5, theta=30), "2", "ordinal_from_F"),
    (54, "G", "5,50", "side_leaf", "dt05", 6, dict(k=1, ordinal=2, n=5, theta=50), "2", "ordinal_from_F|assigned_by_row_order"),
    (55, "G", "7,60", "side_leaf", "dt05", 7, dict(k=1, ordinal=3, n=7, theta=60), "2", "ordinal_from_F|assigned_by_row_order"),
    (55, "I", "4,4", "truss", "dt05", 7, dict(n_truss=2, flowers_each=4), "4", "position_not_confirmed"),
    (57, "H", 3, "pending", "dt05", 9, {}, "-", "held_meaning_not_stated_by_user"),
    (59, "E", 1, "side_axis", "dt06", None, dict(k=1), "1", "angle_not_recorded|attach_node_unknown"),
    (59, "G", "5,10", "side_leaf", "dt06", None, dict(k=1, ordinal=1, n=5, theta=10), "2", "ordinal_from_F"),
    (59, "J", 8.1, "thickness", "dt06", None, dict(mm=8.1), "6", "realigned|rank_unknown"),
    (60, "G", "7,0", "side_leaf", "dt06", 1, dict(k=1, ordinal=2, n=7, theta=0), "2", "ordinal_from_F|assigned_by_row_order"),
    (60, "I", "4,4", "truss", "dt06", 1, dict(n_truss=2, flowers_each=4), "4", "position_not_confirmed"),
    (60, "J", 6, "thickness", "dt06", 1, dict(mm=6), "6", "realigned|two_values_same_row"),
    (60, "K", 3.7, "thickness", "dt06", 1, dict(mm=3.7), "6", "realigned|two_values_same_row"),
    (61, "G", "7,30", "side_leaf", "dt06", 2, dict(k=1, ordinal=3, n=7, theta=30), "2", "ordinal_from_F|assigned_by_row_order"),
    (61, "I", 4, "pending", "dt06", 2, {}, "-", "held_meaning_not_stated_by_user"),
    (62, "G", "5,45", "side_leaf", "dt06", 3, dict(k=1, ordinal=4, n=5, theta=45), "2", "ordinal_from_F|assigned_by_row_order"),
]
ORD = [(53, "F", 1), (54, "F", 2), (55, "F", 3), (59, "F", 1), (60, "F", 2), (61, "F", 3), (62, "F", 4)]
for r, c, raw, *_ in C:
    assert str(cell(r, c)).strip() == str(raw), (r, c, cell(r, c), raw)
for r, c, raw in ORD:
    assert cell(r, c) == raw, (r, c)
# every non-empty irregular cell is covered exactly once (main-stem C/D cells excluded)
irr = json.load(open(D / "hand_irregular_rows.json", encoding="utf-8"))["rows"]
want = {(x["excel_row"], col) for x in irr for col in x["cells"] if col not in ("C", "D")} | {(46, "C")}
have = {(r, c) for r, c, *_ in C} | {(r, c) for r, c, _ in ORD}
assert want == have, (sorted(want - have), sorted(have - want))

rows = [dict(excel_row=r, col=c, raw=str(raw), kind=k, plant=p, main_rank=mr, rule=rule, flags=fl, labels=LAB, confirmation="user_confirmed_2026-11-01 (rules 22:16, structural readings 23:14, truss numbers 23:16)" if k != "pending" else "held", **pl) for r, c, raw, k, p, mr, pl, rule, fl in C]
df = pd.DataFrame(rows)
df.to_csv(D / "hand_reinterpreted_cells.csv", index=False)
sa = df[df.kind == "side_axis"][["plant", "main_rank", "k", "angle", "excel_row", "flags"]]
sl = df[df.kind == "side_leaf"][["plant", "main_rank", "k", "ordinal", "n", "theta", "excel_row", "flags"]]
tr = df[df.kind == "truss"][["plant", "main_rank", "n_truss", "flowers_each", "excel_row", "flags"]]
th = df[df.kind == "thickness"][["plant", "main_rank", "mm", "excel_row", "flags"]]
for name, t in (("hand_side_axes.csv", sa), ("hand_side_leaves.csv", sl), ("hand_trusses.csv", tr), ("hand_thickness.csv", th)):
    t.assign(labels=LAB).to_csv(D / name, index=False)

# template draft v0c (hand only; main-stem part unchanged from v0)
main = pd.read_csv(D / "MTG_TEMPLATE_v0_hand_draft.csv")
thr = th.dropna(subset=["main_rank"]).groupby("main_rank").mm.agg(["count", "mean", "std"]).round(2)
main["stem_thickness_n"] = main["rank"].map(thr["count"]).fillna(0).astype(int)
main["stem_thickness_mean_mm"] = main["rank"].map(thr["mean"]); main["stem_thickness_sd_mm"] = main["rank"].map(thr["std"])
main["age_dag"] = 60; main["age_label"] = "user_stated"
main.to_csv(D / "MTG_TEMPLATE_v0c_hand_draft.csv", index=False)
side = dict(
    labels=[LAB, "draft_not_frozen", "excludes t2_02/t2_04/t2_05 until validation membership is answered (not used: hand only)"],
    side_axis=dict(n=int(len(sa)), n_with_angle=int(sa.angle.notna().sum()), angles_deg=sa.angle.dropna().tolist(),
                   attach_main_rank_counts={str(int(k)): int(v) for k, v in sa.main_rank.dropna().value_counts().sort_index().items()},
                   attach_node_unknown=int(sa.main_rank.isna().sum()), per_plant={p: int(v) for p, v in sa.plant.value_counts().sort_index().items()}),
    side_axis_leaf=dict(n=int(len(sl)), leaflets={str(int(k)): int(v) for k, v in sl.n.value_counts().sort_index().items()},
                        theta0_mean_deg=round(float(sl.theta.mean()), 2), theta0_sd_deg=round(float(sl.theta.std()), 2), theta0_median_deg=float(sl.theta.median()),
                        theta0_min=float(sl.theta.min()), theta0_max=float(sl.theta.max()), per_side_axis_max=int(sl.groupby(["plant", "k"]).size().max())),
    truss=dict(cells=int(len(tr)), n_truss=int(tr.n_truss.sum()), flowers_per_truss=[int(f) for n_, f in zip(tr.n_truss, tr.flowers_each) for _ in range(int(n_))], n_flowers=int((tr.n_truss * tr.flowers_each).sum()), position="not_confirmed"),
    thickness=dict(placed=int(len(th)), header_aligned=int(th["flags"].str.contains("header_aligned").sum()), realigned=int(th["flags"].str.contains("realigned").sum()),
                   rank_unknown=int(th.main_rank.isna().sum())),
    pending_user=int((df.kind == "pending").sum()), omitted=int((df.kind == "omitted").sum()), age=dict(age_dag=60, label="user_stated"),
)
json.dump(side, open(D / "MTG_TEMPLATE_v0c_side_organs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
H = {f: hashlib.sha256((D / f).read_bytes()).hexdigest() for f in ["hand_reinterpreted_cells.csv", "hand_side_axes.csv", "hand_side_leaves.csv", "hand_trusses.csv", "hand_thickness.csv",
                                                                     "MTG_TEMPLATE_v0c_hand_draft.csv", "MTG_TEMPLATE_v0c_side_organs.json"]}
json.dump(H, open(D / "output_hashes_amendment2b.json", "w"), indent=1)
print(json.dumps(side, ensure_ascii=False))
print(df.kind.value_counts().to_dict())
