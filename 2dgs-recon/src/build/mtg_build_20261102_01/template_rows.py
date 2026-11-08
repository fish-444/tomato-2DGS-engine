"""Template v2 draft (+ v1 baseline) -> the values the MTG build may use, with their draft labels (CONTRACT §3; addendum §2.1-2.6).
Writes template_rows_used.csv (row by row: value, draft label, which output uses it), template_v1_vs_v2draft_used.csv, excluded.csv.
Rules: any row labelled display_only / superseded_display_only / *_not_used / trait_unreliable / not_possible is metadata only;
skeleton_internode row is replaced by the user setting (16:22); divergence = literature range row."""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
V1 = ROOT / "build/traits_recon_20261101_01/MTG_TEMPLATE_v1.csv"
V2 = ROOT / "build/traits_recon_20261101_01/MTG_TEMPLATE_v2_ungraded_draft.csv"
INTERNODE_SETTING = dict(lo=2.0, hi=4.0, median=3.75, label="user_set_range_2.00_4.00_median_3.75", concentration=6.0)
NOT_VALUE = ("display_only", "superseded_display_only", "not_used", "trait_unreliable", "not_possible", "resolved", "pending_user")


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def rid(r):
    return f"{r['organ']}|{r['rank']}|{r['parameter'].split(' (')[0]}"


def classify(r):
    """-> (use, reason). use in {value, metadata, replaced, not_used}."""
    p = r["parameter"]; prov = r["provenance"]; lab = r["labels"]
    if r["organ"] == "not_in_template":
        return "metadata", f"not_in_template ({prov})"
    if p == "fruits_per_plant":
        return "not_used", "fruit row pending_user (main prompt §2)"
    if prov.endswith("_not_used") or "not_used" in lab:
        return "not_used", "draft label not_used (theta0 = hand only, user 16:13)"
    if p == "skeleton_internode":
        return "replaced", "user setting 16:22: range 2.00-4.00 mm, median 3.75 (scaled Beta), label " + INTERNODE_SETTING["label"]
    if p in ("skeleton_curvature", "skeleton_tip_drop", "skeleton_s_apex_rel", "skeleton_arc_better_share"):
        return "metadata", "derived shape rows: geometry follows dtheta and L (CONTRACT §3); reported, not an input"
    if p.startswith("recon_A2_"):
        return "check_target", "reconstruction size: check target only (not drawn, not written)"
    if any(t in lab for t in ("display_only",)):
        return "not_used", "display_only"
    return "value", prov


def uses_of(r):
    p = r["parameter"]; o = r["organ"]
    if p.startswith("theta0") or p.startswith("insertion") or p.startswith("p_leaflets") or p == "stem_thickness":
        return "static_observed(by rank);static_template_mean;dynamic_rules"
    if p in ("compound_leaves_per_main_stem", "side_axes_per_plant", "attach_main_rank_counts", "angle_from_horizontal_values", "leaves_per_side_axis_max", "flowers_per_truss", "trusses_per_plant"):
        return "static_template_mean;dynamic_rules"
    if p in ("skeleton_dtheta_total_bend", "skeleton_leaf_length_L"):
        return "static_observed(dtheta: all leaves; L: template only);static_template_mean;dynamic_rules"
    if p == "divergence_deg_range":
        return "static_template_mean(midpoint 137.5);dynamic_rules(range)"
    if p == "age_at_photographing":
        return "all"
    return ""


def load():
    v1 = {rid(r): r for r in rows(V1)}; v2 = {rid(r): r for r in rows(V2)}
    used, excluded, diff = [], [], []
    for k, r in v2.items():
        use, why = classify(r)
        rec = dict(row_id=k, organ=r["organ"], rank=r["rank"], parameter=r["parameter"], unit=r["unit"], value=r["value"], sd=r["sd"], n_plants=r["n_plants"], n_items=r["n_items"],
                   provenance=r["provenance"], labels=r["labels"], template_status=r["template_status"], use=use, reason=why, used_by=uses_of(r) if use in ("value", "replaced") else "")
        used.append(rec)
        if use in ("not_used", "metadata", "check_target"):
            excluded.append(dict(row_id=k, value=r["value"], label=r["labels"] or r["provenance"], why=why))
        a = v1.get(k)
        if a is None:
            diff.append(dict(row_id=k, in_v1="no", in_v2draft="yes", v1_value="", v2_value=r["value"], v1_label="", v2_label=r["labels"] or r["provenance"], differs="new_in_v2",
                             version_used_here="v2draft" if use in ("value", "replaced") else "none(metadata)", note=why))
        else:
            same = (a["value"] == r["value"] and a["sd"] == r["sd"] and a["provenance"] == r["provenance"])
            diff.append(dict(row_id=k, in_v1="yes", in_v2draft="yes", v1_value=a["value"], v2_value=r["value"], v1_label=a["labels"] or a["provenance"], v2_label=r["labels"] or r["provenance"],
                             differs="no" if same else "yes", version_used_here=("v2draft(=v1)" if same else "v2draft") if use in ("value", "replaced") else "none(metadata)", note=why))
    for k, a in v1.items():
        if k not in v2:
            diff.append(dict(row_id=k, in_v1="yes", in_v2draft="no", v1_value=a["value"], v2_value="", v1_label=a["labels"] or a["provenance"], v2_label="", differs="removed_in_v2",
                             version_used_here="v1(carried unchanged)" if a["organ"] != "not_in_template" else "none", note="row only in v1: carried over unchanged (addendum §2.1)"))
            used.append(dict(row_id=k, organ=a["organ"], rank=a["rank"], parameter=a["parameter"], unit=a["unit"], value=a["value"], sd=a["sd"], n_plants=a["n_plants"], n_items=a["n_items"],
                             provenance=a["provenance"], labels=a["labels"], template_status="v1_frozen_carried", use="value", reason="only in v1", used_by=uses_of(a)))
    return v1, v2, used, excluded, diff


def write_tables():
    v1, v2, used, excluded, diff = load()
    for name, data in (("template_rows_used.csv", used), ("excluded_template_rows.csv", excluded), ("template_v1_vs_v2draft_used.csv", diff)):
        with open(HERE / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    return used


class Template:
    """Typed access to the usable rows."""
    def __init__(self):
        self.v1, self.v2, self.used, _, _ = load()
        self.R = self.v2

    def row(self, organ, rank, param_prefix):
        for k, r in self.R.items():
            if r["organ"] == organ and str(r["rank"]) == str(rank) and r["parameter"].startswith(param_prefix):
                return r
        return None

    def theta0(self, rank):
        r = self.row("main_compound_leaf", rank, "theta0") or self.row("main_compound_leaf", "all", "theta0")
        return float(r["value"]), float(r["sd"] or 0), ("hand_measured_by_rank" if r["rank"] != "all" else "hand_measured_all_ranks(rank>9)"), r

    def leaflet_shares(self, rank):
        out = {}
        for n in (5, 6, 7):
            r = self.row("main_compound_leaf", rank, f"p_leaflets_{n}") or self.row("main_compound_leaf", "all", f"p_leaflets_{n}")
            out[n] = float(r["value"]) if r else 0.0
        s = sum(out.values()) or 1.0
        return {k: v / s for k, v in out.items()}

    def leaflets_mode(self, rank):
        sh = self.leaflet_shares(rank); return max(sorted(sh), key=lambda n: sh[n])

    def stem_thickness(self, rank):
        r = self.row("main_stem", rank, "stem_thickness")
        if r is None:
            r = self.row("main_stem", 7, "stem_thickness"); return float(r["value"]), float(r["sd"] or 0), "hand_measured_rank7_reused(rank>7)"
        return float(r["value"]), float(r["sd"] or 0), "hand_measured_by_rank"

    def dtheta(self):
        r = self.row("main_compound_leaf", "all", "skeleton_dtheta_total_bend"); return float(r["value"]), float(r["sd"]), "skeleton_traits_ungraded", r["note"]

    def leaf_length(self):
        r = self.row("main_compound_leaf", "all", "skeleton_leaf_length_L"); return float(r["value"]), float(r["sd"]), "skeleton_traits_ungraded"

    def divergence_range(self):
        r = self.row("main_compound_leaf", "all", "divergence_deg_range"); lo, hi = (float(x) for x in r["value"].split("-")); return lo, hi, "literature_range"

    def main_leaves(self):
        r = self.row("main_stem", "all", "compound_leaves_per_main_stem"); return float(r["value"]), float(r["sd"])

    def side(self):
        cnt = json.loads(self.row("side_axis", "all", "attach_main_rank_counts")["value"]); ang = json.loads(self.row("side_axis", "all", "angle_from_horizontal_values")["value"])
        t = self.row("side_axis_compound_leaf", "all", "theta0"); p5 = self.row("side_axis_compound_leaf", "all", "p_leaflets_5"); p7 = self.row("side_axis_compound_leaf", "all", "p_leaflets_7")
        return dict(axes_per_plant=float(self.row("side_axis", "all", "side_axes_per_plant")["value"]), attach_counts={int(k): v for k, v in cnt.items()}, angles=ang,
                    theta0=float(t["value"]), theta0_sd=float(t["sd"]), n_side_leaves=float(t["n_items"]), n_axes_items=float(self.row("side_axis", "all", "side_axes_per_plant")["n_items"]),
                    shares={5: float(p5["value"]), 7: float(p7["value"])}, leaves_max=int(float(self.row("side_axis_compound_leaf", "all", "leaves_per_side_axis_max")["value"])))

    def trusses(self):
        f = self.row("truss", "all", "flowers_per_truss"); t = self.row("truss", "all", "trusses_per_plant")
        return dict(flowers_mean=float(f["value"]), flowers_sd=float(f["sd"]), trusses_mean=float(t["value"]))

    def recon_size(self):
        h = self.row("plant", "all", "recon_A2_height"); x = self.row("plant", "all", "recon_A2_width_x"); y = self.row("plant", "all", "recon_A2_width_y")
        return dict(height=(float(h["value"]), float(h["sd"])), width_x=(float(x["value"]), float(x["sd"])), width_y=(float(y["value"]), float(y["sd"])))


if __name__ == "__main__":
    u = write_tables(); print(len(u), "rows;", sum(r["use"] == "value" for r in u), "value rows")
