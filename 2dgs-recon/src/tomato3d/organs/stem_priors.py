"""P1 stem gap-fill: stem geometry priors from the user's clicks on the close-up stem photos (CONTRACT.md + Revision 1).

Reads `stem_clicks.jsonl` (append-only; `remove` records undo) and the proposal JSONs; writes `stem_priors_from_photos.csv`,
`stem_priors_summary.json` and `qc_centreline.json`. Ratios only unless a mm-per-px scale is given. The photos are generic reference
material: nothing here refers to an individual.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

QC = {"IMG_0736", "IMG_0727", "IMG_0731", "IMG_0740", "IMG_0753"}


def active(recs):
    gone = {r["target"] for r in recs if r["kind"] == "remove"}
    return [r for r in recs if r["kind"] != "remove" and r["id"] not in gone]


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def polyline_dist(p, line):
    """Perpendicular distance of point p from a polyline (list of [x, y])."""
    L = np.asarray(line, float); p = np.asarray(p, float); best = np.inf
    for a, b in zip(L[:-1], L[1:]):
        d = b - a; t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(a + t * d - p)))
    return best


def per_photo(name, A, prop):
    flags = {r["value"] for r in A if r["kind"] == "flag"}
    cen = sorted([r["pts"][0] for r in A if r["kind"] == "centre"], key=lambda p: -p[1])  # bottom -> top (Rev 3: y order, hand + accepted mixed)
    rims = [r["pts"] for r in A if r["kind"] == "rim"]
    grids = [dist(*r["pts"]) for r in A if r["kind"] == "grid"]
    grid_px = float(np.median(grids)) if grids else None
    rim_w = float(np.median([dist(*pr) for pr in rims])) if rims else None
    ys = [p[1] for p in cen]
    y_top = min(ys) if ys else None
    # Rev 3: without a rim click the base is the lowest centre click (tag base_from_lowest_centre_click)
    base_src = "rim" if rims else ("lowest_centre_click" if ys else None)
    y_base = float(np.mean([p[1] for pr in rims for p in pr])) if rims else (max(ys) if ys else None)
    span = float(sum(dist(a, b) for a, b in zip(cen[:-1], cen[1:]))) if len(cen) >= 2 else None
    rows = []

    def hclass(y):
        if y_base is not None and y_top is not None and y_base > y_top:
            h = (y_base - y) / (y_base - y_top)
            return ("lower" if h < 1 / 3 else "mid" if h < 2 / 3 else "upper"), round(h, 3)
        return "unclassed", None

    def axis_at(y):
        """Unit upward stem direction of the centre polyline near image row y."""
        if len(cen) < 2: return np.array([0.0, -1.0])
        a, b = cen[0], cen[1]
        for a, b in zip(cen[:-1], cen[1:]):
            if b[1] <= y <= a[1]: break
        v = np.array([b[0] - a[0], b[1] - a[1]], float); n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([0.0, -1.0])

    widths = sorted([(float(np.mean([p[1] for p in r["pts"]])), dist(*r["pts"]), r) for r in A if r["kind"] == "width"], key=lambda a: -a[0])
    w_base = widths[0][1] if widths else None
    for y, w, r in widths:
        c, h = hclass(y)
        rows.append(dict(photo=name, kind="stem_width", cls=c, h_rel=h, y_px=round(y, 1), value_px=round(w, 1),
                         w_rim=(round(w / rim_w, 4) if rim_w else None), w_grid=(round(w / grid_px, 4) if grid_px else None),
                         w_rel=(round(w / w_base, 4) if len(widths) >= 2 else None), click_ids=r["id"]))
    nodes = sorted([r for r in A if r["kind"] == "node"], key=lambda r: -r["pts"][0][1])

    def width_at(y):
        if not widths: return None
        return min(widths, key=lambda a: abs(a[0] - y))[1]
    for k, r in enumerate(nodes):
        (nx, ny), (px, py) = r["pts"]
        c, h = hclass(ny)
        u = np.array([px - nx, py - ny], float); u = u / max(np.linalg.norm(u), 1e-9)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(u, axis_at(ny)), -1, 1))))  # Rev 3: node -> leaf point vs. upward stem axis
        wn = width_at(ny)
        rows.append(dict(photo=name, kind="node_to_leaf_angle_deg", cls=c, h_rel=h, y_px=round(ny, 1), value_px=round(ang, 1), click_ids=r["id"]))
        if k + 1 < len(nodes):
            il = dist(r["pts"][0], nodes[k + 1]["pts"][0])
            rows.append(dict(photo=name, kind="internode_upper_bound", cls=c, h_rel=h, y_px=round(ny, 1), value_px=round(il, 1),
                             ilen_w=(round(il / wn, 3) if wn else None), w_rim=(round(il / rim_w, 4) if rim_w else None),
                             ilen_span=(round(il / span, 4) if span else None), w_grid=(round(il / grid_px, 4) if grid_px else None),
                             click_ids=f'{r["id"]}+{nodes[k + 1]["id"]}'))
    for r in [r for r in A if r["kind"] == "petw"]:
        y = float(np.mean([p[1] for p in r["pts"]])); c, h = hclass(y); wn = width_at(y); pw = dist(*r["pts"])
        rows.append(dict(photo=name, kind="petiole_base_width", cls=c, h_rel=h, y_px=round(y, 1), value_px=round(pw, 1),
                         pet_w=(round(pw / wn, 3) if wn else None), click_ids=r["id"]))
    for r in [r for r in A if r["kind"] == "branch"]:
        y = float(np.mean([p[1] for p in r["pts"]])); c, h = hclass(y); wn = width_at(y); bw = dist(*r["pts"])
        rows.append(dict(photo=name, kind="branch_width", cls=c, h_rel=h, y_px=round(y, 1), value_px=round(bw, 1),
                         w_rel=(round(bw / wn, 3) if wn else None), click_ids=r["id"]))
    info = dict(photo=name, flags=sorted(flags), n_centre=len(cen), n_nodes_lower_bound=len(nodes), rim_w_px=rim_w, grid_px=grid_px,
                base_from=base_src, y_base=y_base, y_top=y_top, stem_span_px=(round(span, 1) if span else None),
                n_width=len(widths), qc=name in QC, centre=cen)
    return rows, info


def stat(v):
    v = [x for x in v if x is not None]
    if not v: return dict(n=0)
    return dict(n=len(v), median=round(float(np.median(v)), 4), min=round(float(min(v)), 4), max=round(float(max(v)), 4))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(argv)
    D = Path(a.dir)
    recs = [json.loads(l) for l in open(D / "stem_clicks.jsonl", encoding="utf8")] if (D / "stem_clicks.jsonl").exists() else []
    A = [r for r in active(recs) if r["kind"] != "cand_verdict"]; names = sorted({r["photo"] for r in A})
    rows, infos, qc = [], [], []
    for n in names:
        pf = D / "stem_masks" / f"{n}.json"; prop = json.load(open(pf)) if pf.exists() else None
        r, info = per_photo(n, [x for x in A if x["photo"] == n], prop); infos.append(info)
        excluded = "hand_occluded" in info["flags"] or "blurred" in info["flags"]
        if n in QC:
            # Revision 2: the automatic proposal is the geometric candidate path (stem_cands); the user's polyline = hand-clicked centre points only
            cf = D / "stem_cands" / f"{n}.json"; cpath = (json.load(open(cf)).get("path") if cf.exists() else None) or (prop or {}).get("centreline")
            man = sorted([r["pts"][0] for r in A if r["photo"] == n and r["kind"] == "centre" and r.get("source") != "auto"], key=lambda p: -p[1])
            if cpath and len(man) >= 3:
                ys = [p[1] for p in man]
                d = [polyline_dist(p, man) for p in cpath if min(ys) <= p[1] <= max(ys)]
                wpx = [x["value_px"] for x in r if x["kind"] == "stem_width"] or ([prop["width_px_median"]] if (prop or {}).get("width_px_median") else [])
                qc.append(dict(photo=n, n_rows=len(d), mean_px=(round(float(np.mean(d)), 1) if d else None),
                               mean_stem_widths=(round(float(np.mean(d)) / float(np.median(wpx)), 2) if d and wpx else None)))
            else:
                qc.append(dict(photo=n, n_rows=0, mean_px=None, reason="no candidate path or < 3 hand-clicked centre points"))
            continue  # QC photos are excluded from the priors
        for x in r: x["excluded"] = excluded
        rows += r
    keys = ["photo", "kind", "cls", "h_rel", "y_px", "value_px", "w_rim", "w_grid", "w_rel", "ilen_w", "ilen_span", "pet_w", "excluded", "click_ids"]
    with open(D / "stem_priors_from_photos.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); [w.writerow(x) for x in rows]
    use = [x for x in rows if not x["excluded"]]
    summ = {"units": "ratios (no mm scale: no rim click, grid mesh size not given); conditional_on_unvalidated_scale", "revision": 3, "reference_material_only": True,
            "n_photos_with_clicks": len(names), "n_photos_used": len({x["photo"] for x in use})}
    for cls in ("lower", "mid", "upper"):
        u = [x for x in use if x["cls"] == cls]
        summ[cls] = {"w_rim": stat([x.get("w_rim") for x in u if x["kind"] == "stem_width"]),
                     "w_rel": stat([x.get("w_rel") for x in u if x["kind"] == "stem_width"]),
                     "w_grid": stat([x.get("w_grid") for x in u if x["kind"] == "stem_width"]),
                     "internode_upper_bound_ilen_span": stat([x.get("ilen_span") for x in u if x["kind"] == "internode_upper_bound"]),
                     "internode_upper_bound_ilen_w": stat([x.get("ilen_w") for x in u if x["kind"] == "internode_upper_bound"]),
                     "node_to_leaf_angle_deg": stat([x["value_px"] for x in u if x["kind"] == "node_to_leaf_angle_deg"]),
                     "petiole_base_pet_w": stat([x.get("pet_w") for x in u if x["kind"] == "petiole_base_width"]),
                     "branch_w_rel": stat([x.get("w_rel") for x in u if x["kind"] == "branch_width"])}
    # P1: lower-class w_rim, n = photos (median per photo), range within +-40 % of the median
    per = {}
    for x in use:
        if x["cls"] == "lower" and x["kind"] == "stem_width" and x.get("w_rim") is not None: per.setdefault(x["photo"], []).append(x["w_rim"])
    pm = [float(np.median(v)) for v in per.values()]
    if len(pm) >= 5:
        med = float(np.median(pm)); ok = (max(pm) - med) / med <= 0.40 and (med - min(pm)) / med <= 0.40
        summ["P1"] = dict(n_photos=len(pm), median=round(med, 4), min=round(min(pm), 4), max=round(max(pm), 4), verdict="accepted" if ok else "weak_prior")
    else:
        summ["P1"] = dict(n_photos=len(pm), verdict="weak_prior", reason="n < 5 photos with lower-stem width and rim clicks")
    summ["all_heights_pooled_labelled"] = {
        "internode_upper_bound_ilen_span": stat([x.get("ilen_span") for x in use if x["kind"] == "internode_upper_bound"]),
        "node_to_leaf_angle_deg": stat([x["value_px"] for x in use if x["kind"] == "node_to_leaf_angle_deg"]),
        "nodes_per_photo_lower_bound": stat([i["n_nodes_lower_bound"] for i in infos if i["photo"] not in QC])}
    summ["qc"] = qc
    summ["photos"] = [{k: v for k, v in i.items() if k != "centre"} for i in infos]
    json.dump(summ, open(D / "stem_priors_summary.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: summ[k] for k in ("n_photos_with_clicks", "n_photos_used", "P1")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
