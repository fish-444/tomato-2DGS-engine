"""Observed plant -> B-model MTG CSV (Micbut/FSP-model-dwarf-tomato @ 2d727a3). build/bmodel_mtg_20261101_01, CONTRACT.md §4, ADAPTERS.md.

Input: our skeleton (skeleton_v1 / v1.1, read only) + per-leaf area (b) from the operating label route; output: one scenario CSV in the
B-model's own format = the published parameter block (static 1_50: first 76 lines; dynamic base_plant_dynamic_50: first 120 lines),
changed only where ADAPTERS.md says, then a plant row and MTG rows. Every observed organ: sd 0, probability 1. A side file
`<name>.rows.json` keeps, per row, our ids and the provenance of every number (observed / model_completed / model_derived / assumed_age).

Conventions (ADAPTERS.md "Converter conventions"; verified by the one-leaf test kernel before plant runs):
* organ azimuths are absolute in the plant frame (the importer undoes each internode's RH/RL); orientation = (our azimuth + ORI_OFFSET) mod 360,
  inclination = 90 - elevation (RL from the stem axis, 90 = horizontal). Our azimuth/elevation = first 10 mm of the leaf line (skeleton_tree).
* `readMTGData` deletes every '-' -> no negative number is written.
* static leaf columns: 14 petiole length (m), 16 petiole diameter (m), 17 blade area (m2), 30 rachis(+petiolule) dry weight (g, the importer
  turns it into rachis length 17.142*(1000*dw)^0.29607/1000 - 0.01 m). Observed leaf length L (skeleton path) = petiole + rachis + last
  petiolule (0.01 m) + terminal leaflet extent; petiole:rachis split in the ratio of the model's own allometric lengths for that area.
* dynamic leaf columns: 17 dry biomass mg = area (b) / SLA_model / blade fraction (model_derived); 19/20/21 = the scenario's own per-rank-class
  potential max / growth duration / age at max growth; 22 = leaf age in degree days (assumed_age).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["bmodel_micbut_2d727a3", "observed_plant_input", "conditional_on_unvalidated_scale", "after_checkpoint_v1"]
DIR_MM = 10.0
NCOL = 33
STATIC_HEAD, DYNAMIC_HEAD = 76, 120
ORI_OFFSET = 90.0               # model ori = our azimuth + 90: verified bm-geo4 (ori 0 -> 270 deg, 90 -> 0, 180 -> 90); inclination 90 = horizontal, 45 -> elevation 45
SHARED_NODE_MM = 0.5         # CONTRACT A1
I0_COMPLETED_MM = 8.865065    # published 1_50 I0 length (m 0.008865065): stem below the lowest leaf base when no axis exists (model_completed)
GENERIC_PETIOLULE_M = 0.01      # static importer: genericLengthPetiolules = SA_factor (1) * 0.01 m
# terminal-leaflet extent per sqrt(area) for the model's leaflet meshes (rank 1 class / other classes); measured from the model's own mesh
# in the one-leaf test (ADAPTERS.md); None until then -> the converter refuses static leaf rows.
LEAFLET_EXTENT_PER_SQRT_AREA = {"first": 1.088, "other": 1.538}    # measured, bm-geo4 one-leaf tests t1-t6 (results/one_leaf_tests.json)


# ------------------------------------------------------------------ parameter block of a published scenario
def read_block(csv_path, head):
    raw = Path(csv_path).read_bytes()
    lines = raw.split(b"\n")
    return lines[:head], raw


def block_value(lines, name):
    for ln in lines:
        c = ln.decode("utf-8-sig", "replace").split(",")
        if c and c[0].strip() == name:
            return c[1]
    raise KeyError(name)


def set_block_value(lines, name, value):
    out, hit = [], 0
    for ln in lines:
        c = ln.decode("utf-8-sig" if ln.startswith(b"\xef\xbb\xbf") else "utf-8", "replace").split(",")
        if c and c[0].strip() == name:
            c[1] = value; hit += 1
            s = ",".join(c).encode("utf-8"); out.append((b"\xef\xbb\xbf" + s) if ln.startswith(b"\xef\xbb\xbf") else s)
        else:
            out.append(ln)
    assert hit == 1, f"{name}: {hit} lines"
    return out


# ------------------------------------------------------------------ geometry helpers (same rule as tomato3d.organs.skeleton_tree)
def first_direction(P_mm, length_mm=DIR_MM):
    P = np.asarray(P_mm, float)
    if len(P) < 2:
        return None
    acc = 0.0
    for i in range(1, len(P)):
        acc += float(np.linalg.norm(P[i] - P[i - 1]))
        if acc >= length_mm:
            break
    v = P[i] - P[0]; n = np.linalg.norm(v)
    return v / n if n > 0 else None


def az_el(d):
    return float(math.degrees(math.atan2(d[1], d[0])) % 360.0), float(math.degrees(math.asin(max(-1.0, min(1.0, d[2])))))


def path_len(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def nearest_on_curve(C, p):
    """(s along C, distance) of the point of polyline C nearest to p."""
    C = np.asarray(C, float); p = np.asarray(p, float); best = (None, 1e18); s0 = 0.0
    for a, b in zip(C[:-1], C[1:]):
        ab = b - a; L = float(np.linalg.norm(ab))
        t = 0.0 if L == 0 else float(np.clip(np.dot(p - a, ab) / L ** 2, 0, 1))
        d = float(np.linalg.norm(a + t * ab - p))
        if d < best[1]:
            best = (s0 + t * L, d)
        s0 += L
    return best


def point_at(C, s):
    C = np.asarray(C, float); acc = 0.0
    for a, b in zip(C[:-1], C[1:]):
        L = float(np.linalg.norm(b - a))
        if acc + L >= s:
            return a + (b - a) * ((s - acc) / L if L else 0.0)
        acc += L
    return C[-1]


def ori_inc(v):
    """Model orientation / inclination (deg, both >= 0) of a 3D vector: inclination from vertical, orientation from our azimuth."""
    v = np.asarray(v, float); n = np.linalg.norm(v)
    if n == 0:
        return 0.0, 0.0
    v = v / n; inc = math.degrees(math.acos(max(-1.0, min(1.0, v[2]))))
    az = math.degrees(math.atan2(v[1], v[0])) % 360.0 if math.hypot(v[0], v[1]) > 1e-9 else 0.0
    return (az + ORI_OFFSET) % 360.0, inc


# ------------------------------------------------------------------ observed plant
def load_observed(pid, skeleton_file, areas, area_source, age_dag=None):
    """Ordered nodes + leaves from the skeleton. areas: {leaf_uid: area_b_mm2}. Returns a dict (all mm, board frame)."""
    sk = json.loads(Path(skeleton_file).read_text(encoding="utf-8"))
    live = [l for l in sk["leaves"] if l["status"] in ("active", "new", "candidate_lost")]
    main = next((a for a in sk.get("axes", []) if a.get("type") == "main" and a.get("status") == "active" and a.get("fit")), None)
    flags = []
    if main is not None:                         # the user's axis, oriented bottom -> top (the editor stores the clicking order)
        C = np.asarray(main["fit"]["curve_mm"], float); axis_src = "user_skeleton_axis"; Lc = path_len(C)
        prov_seg = [(sg["s0_mm"], sg["s1_mm"], sg["provenance"]) for sg in main["segments"]]
        if C[0][2] > C[-1][2]:
            C = C[::-1].copy(); prov_seg = [(Lc - b, Lc - a, p) for a, b, p in prov_seg]; flags.append("axis_reversed_to_bottom_up")
        flags += [f"axis_{f}" for f in main.get("flags", [])]
    else:                                        # no axis: vertical line through the median xy of the leaf bases, from the lowest base minus
        B = np.asarray([np.asarray(l["polyline_xyz"][0], float) * 1000 for l in live])        # the scenario's I0 length (model_completed)
        xy = np.median(B[:, :2], axis=0); z0 = B[:, 2].min() - I0_COMPLETED_MM
        C = np.array([[xy[0], xy[1], z0], [xy[0], xy[1], B[:, 2].max()]]); axis_src = "vertical_proxy_through_median_leaf_base (no user axis)"; prov_seg = []
        flags.append("no_user_axis")
    leaves = []
    for l in live:
        P = np.asarray(l["polyline_xyz"], float) * 1000.0; s, dist = nearest_on_curve(C, P[0])
        d = first_direction(P); one = len(P) < 2 or d is None
        rec = dict(leaf_uid=l["leaf_uid"], rank_hint=l["rank_hint"], base_mm=P[0].tolist(), s_mm=s, base_axis_dist_mm=dist, n_clicks=l["n_clicks"],
                   length_mm=None if one else path_len(P), straight_mm=None if one else float(np.linalg.norm(P[-1] - P[0])),
                   azimuth_deg=None if one else az_el(d)[0], elevation_deg=None if one else az_el(d)[1],
                   area_b_mm2=areas.get(l["leaf_uid"]), flags=(["one_click_no_direction"] if one else []) + (["base_off_axis"] if dist > 10 else []))
        leaves.append(rec)
    leaves.sort(key=lambda r: (r["s_mm"], r["base_mm"][2]))
    obsL = [r["length_mm"] for r in leaves if r["length_mm"]]
    for k, r in enumerate(leaves, 1):
        r["model_rank"] = k
        if r["length_mm"] is None:              # CONTRACT A1: one-click leaf -> model_completed direction and length
            h = np.asarray(r["base_mm"], float) - point_at(C, r["s_mm"]); h[2] = 0.0
            r.update(length_mm=float(np.median(obsL)) if obsL else 30.0, elevation_deg=0.0,
                     azimuth_deg=float(math.degrees(math.atan2(h[1], h[0])) % 360.0) if np.linalg.norm(h) > 0 else 0.0, geometry="model_completed")
        else:
            r["geometry"] = "observed"
    def prov(s0, s1):
        if not prov_seg:
            return "model_completed"
        cov = sum(max(0.0, min(s1, b) - max(s0, a)) for a, b, p in prov_seg if p == "observed")
        return "observed" if s1 > s0 and cov >= 0.999 * (s1 - s0) else "model_completed"
    nodes, prev = [], 0.0
    for r in leaves:
        nodes.append(dict(leaf_uid=r["leaf_uid"], s_mm=r["s_mm"], xyz_mm=point_at(C, r["s_mm"]).tolist(), internode_mm=r["s_mm"] - prev, internode_provenance=prov(prev, r["s_mm"])))
        prev = r["s_mm"]
    base = C[0]; tops = [np.asarray(r["base_mm"]) for r in leaves]
    for l in live:
        tops += [np.asarray(p, float) * 1000 for p in l["polyline_xyz"]]
    height = float(max(p[2] for p in tops) - base[2])
    return dict(plant_id=pid, skeleton_file=str(skeleton_file), skeleton_sha256=hashlib.sha256(Path(skeleton_file).read_bytes()).hexdigest(),
                skeleton_schema=sk["schema"], skeleton_labels=sk.get("labels", []), axis_source=axis_src, axis_curve_mm=C.tolist(), stem_base_mm=base.tolist(),
                leaves=leaves, nodes=nodes, height_mm=height, area_source=area_source, age_dag=age_dag, age_status="recorded" if age_dag else "assumed_age",
                flags=flags, labels=LABELS)


# ------------------------------------------------------------------ model allometry (the scenario's own values)
class Allo:
    def __init__(self, block):
        g = lambda n: float(block_value(block, n))
        self.spl1, self.spl2 = g("SPECIFIC_PETIOLE_LENGTH_ FACTOR 1"), g("SPECIFIC_PETIOLE_LENGTH_ FACTOR 2")
        self.spw1, self.spw2 = g("SPECIFIC_PETIOLE_WIDTH FACTO 1"), g("SPECIFIC_PETIOLE_WIDTH FACTO 2")
        self.sil = (g("SPECIFIC_INTERNODE_LENGTH_FACTOR_1"), g("SPECIFIC_INTERNODE_LENGTH _FACTOR 2"))
        self.sil_thin = (g("SPECIFIC_INTERNODE_> 5 LENGTH FACTOR 1"), g("SPECIFIC_INTERNODE_> 5LENGTH FACTOR 2"))
        self.siw = (g("SPECIFIC_INTERNODE_WIDTH_FACTOR 1"), g("SPECIFIC_INTERNODE_WIDTH_FACTOR 2"))

    def internode_width_m(self, length_m, rank):
        f1, f2 = self.sil_thin if rank > 5 else self.sil
        mass = ((length_m * 1000) / f1) ** (1 / f2)        # the importer's own length -> mass rule (organs.rgg Internode.mtgConstructor)
        return self.siw[0] * mass ** self.siw[1] / 1000      # width from the scenario's internode width factors (model_derived)


def leaf_class(rank):
    return "first" if rank == 1 else ("mid" if rank in (2, 3) else "last")


# ------------------------------------------------------------------ rows
def _row(**cols):
    r = [""] * NCOL
    for k, v in cols.items():
        r[int(k[1:])] = v
    return r


def _f(x):
    """Fixed-point text: readMTGData deletes every '-', which would also break exponents like 1e-05."""
    x = float(x)
    assert x >= 0 and math.isfinite(x), f"{x}: negative / non-finite numbers are mangled by readMTGData"
    s = format(x, ".12f").rstrip("0").rstrip(".")
    return s or "0"


def static_rows(obs, block, sla_mm2_mg, only_leaf=None, short_mode=None):
    """MTG rows for a static scenario + provenance records. only_leaf = model rank: every other leaf gets probability 0 (round-trip
    geometry: one leaf per export; 0 = stem only)."""
    ext = LEAFLET_EXTENT_PER_SQRT_AREA
    assert ext["first"] and ext["other"], "terminal-leaflet extents not measured yet (one-leaf test, ADAPTERS.md)"
    A = Allo(block); rows, prov = [], []
    lam = lambda k: block_value(block, k)
    props = {"first": [float(x) for x in lam("PROPORTION_LEAFLETS_FIRST_LEAVES").split(";")],
             "mid": [float(x) for x in lam("PROPORTION_LEAFLETS_MID_LEAVES").split(";")],
             "last": [float(x) for x in lam("PROPORTION_LEAFLETS_LAST_LEAVES").split(";")]}
    seg_props = {"mid": [float(lam("PROPORTION_SEGMENTS_MID_LEAVES"))], "last": [float(x) for x in lam("PROPORTION_SEGMENTS_LAST_LEAVES").split(";")]}
    rows.append(["1"] + [""] * (NCOL - 1)); prov.append(dict(kind="plant", plant_id=obs["plant_id"]))
    C = np.asarray(obs["axis_curve_mm"], float); prev_xyz = np.asarray(obs["stem_base_mm"], float)
    for i, (nd, lf) in enumerate(zip(obs["nodes"], obs["leaves"])):
        rank = lf["model_rank"]; xyz = np.asarray(nd["xyz_mm"], float); v = xyz - prev_xyz; L_int = float(np.linalg.norm(v)) / 1000
        o, inc = ori_inc(v) if L_int > 0 else (0.0, 0.0)
        name = "I0" if i == 0 else f"I{rank}"; irank = 0 if i == 0 else rank
        if i > 0 and nd["internode_mm"] < SHARED_NODE_MM:          # CONTRACT A1: leaves on one node, no internode row
            lf["flags"].append("shared_node")
        else:
          rows.append(_row(c1=name, c7="0", c8=str(irank), c9="1", c10=_f(o), c11="0", c12=_f(inc), c13="0", c14=_f(max(L_int, 1e-6)), c15="0",
                         c16=_f(A.internode_width_m(max(L_int, 1e-6), irank)), c17="0", c31="0"))
          prov.append(dict(kind="internode", rank=irank, length=nd["internode_provenance"], width="model_derived", direction=nd["internode_provenance"]))
          prev_xyz = xyz
        cls = leaf_class(rank); n_bl = len(props[cls]); area = lf["area_b_mm2"]; aprov = "observed_area_b" if area else "model_completed"
        if not area:
            area = None
        rec = dict(kind="leaf", leaf_uid=lf["leaf_uid"], rank=rank, cls=cls, area=aprov, flags=list(lf["flags"]))
        if area is None:
            rec["skipped_reason"] = "no area"; rows.append(None); prov.append(rec); continue
        A_m2 = area / 1e6
        t_share = props[cls][-1]; t_ext = (ext["first"] if cls == "first" else ext["other"]) * math.sqrt(A_m2 * t_share)
        dw = A_m2 * 1e6 / sla_mm2_mg / 0.75 if cls != "first" else A_m2 * 1e6 / sla_mm2_mg / 0.81      # leaf DW from blade area (model_derived)
        p_alo = A.spl1 * (0.9 * 0.1 * dw) ** A.spl2 / 1000
        r_alo = sum(17.142 * (0.9 * 0.145 * dw * sp) ** 0.29607 / 1000 for sp in seg_props[cls]) if cls != "first" else 0.0
        R = lf["length_mm"] / 1000 - t_ext - GENERIC_PETIOLULE_M
        if R <= 0.002:
            rec["flags"].append("length_short"); R = 0.002
        k = R / (p_alo + r_alo); pet = k * p_alo; rach = k * r_alo
        if "length_short" in rec["flags"] and short_mode == "length_priority":     # CONTRACT A5
            c = ext["first"] if cls == "first" else ext["other"]
            pet, rach = (0.001, 0.0) if cls == "first" else (0.0005, 0.0005)
            t = lf["length_mm"] / 1000 - GENERIC_PETIOLULE_M - 0.001
            if t < 0.0005:
                t = 0.0005; rec["flags"].append("below_model_minimum_length")
            A_new = (t / c) ** 2 / t_share
            rec.update(mapping="length_priority_short", area_input_mm2=area, area_written_mm2=A_new * 1e6, area_not_representable_mm2=area - A_new * 1e6)
            A_m2 = A_new; t_ext = t
        rach_dw_g = ((rach + GENERIC_PETIOLULE_M) * 1000 / 17.142) ** (1 / 0.29607) / 1000 if cls != "first" else 0.015869
        o, _ = ((lf["azimuth_deg"] + ORI_OFFSET) % 360.0, None); inc = 90.0 - lf["elevation_deg"]
        diam = A.spw1 * (0.9 * 0.1 * dw) ** A.spw2 / 1000
        rows.append(_row(c2=f"xL{rank}", c7="0", c8=str(rank), c9="1" if only_leaf in (None, rank) else "0", c10=_f(o), c11="0", c12=_f(inc), c13="0", c14=_f(pet), c15="0", c16=_f(diam),
                         c17=_f(A_m2), c30=_f(rach_dw_g), c31="0"))
        g = lf["geometry"]
        rec.update(orientation=g, inclination=g, length=g + " (split petiole:rachis by model allometry)", petiole_m=pet, rachis_m=rach,
                   terminal_extent_m=t_ext, diameter="model_derived", rachis_dw_g=rach_dw_g)
        prov.append(rec)
    keep = [(r, p) for r, p in zip(rows, prov) if r is not None]
    return [r for r, _ in keep], prov


def write_scenario(out_csv, block_lines, rows, prov, meta):
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    body = b"\n".join(block_lines) + b"\n" + b"".join((",".join(r) + "\n").encode("utf-8") for r in rows)
    out_csv.write_bytes(body)
    side = dict(meta, labels=LABELS, csv=out_csv.name, csv_sha256=hashlib.sha256(body).hexdigest(), rows=prov)
    out_csv.with_suffix(".rows.json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return side["csv_sha256"]


# ------------------------------------------------------------------ dynamic scenario (mass-based import)
def appearance_dd(rank, block):
    """Plant thermal time at which main-stem rank `rank` appears, from the scenario's own phytomer curve
    (-f1 x^2 + f2 x - f3 = rank, larger root as in solveForX). None beyond the curve's maximum rank."""
    f1 = float(block_value(block, "phytomer_appearance_rate_main_stem_factor_1")); f2 = float(block_value(block, "phytomer_appearance_rate_main_stem_factor_2"))
    f3 = float(block_value(block, "phytomer_appearance_rate_main_stem_factor_3"))
    a, b, c = -f1, f2, -f3 - rank; disc = b * b - 4 * a * c
    return None if disc < 0 else (-b + math.sqrt(disc)) / (2 * a)


def plant_dd(age_days, temps, base_t):
    """Thermal time of a plant age (days) with the model's own daily rule age += T - BASE_TEMPERATURE, temperatures by day index."""
    n = int(math.floor(age_days)); s = sum(temps[d] - base_t for d in range(n))
    return s + (age_days - n) * (temps[n] - base_t)


def dynamic_rows(obs, block, plant_age_dd):
    """MTG rows for a dynamic scenario. The dynamic importer is mass based: internode length and every leaf dimension are recomputed by the model
    from dry mass, so observed lengths enter only through mass (internode: inverse of the model's own length rule; leaf: area (b) / SLA);
    orientation and inclination enter directly."""
    g = lambda n: float(block_value(block, n))
    sla = (g("SLA FACTOR_1") * g("PLANT_DENSITY") + g("SLA FACTOR_2"))          # mm2/mg, Leaf.initiate rule (DENSITY = scenario value)
    f1, f2 = g("SPECIFIC_INTERNODE_LENGTH_FACTOR_1"), g("SPECIFIC_INTERNODE_LENGTH _FACTOR 2")
    t1, t2 = g("SPECIFIC_INTERNODE_> 5 LENGTH FACTOR 1"), g("SPECIFIC_INTERNODE_> 5LENGTH FACTOR 2")
    imax, igd, iam = g("DRY_BIOMASS_POTENTIAL_MAX_INTERNODE_MG"), g("GROWTH_DURATION_INTERNODE_DD"), g("AGE_MAXIMUM_GROWTH_INTERNODE_DD")
    lmax = {"first": g("DRY_BIOMASS_POTENTIAL_first_MG"), "other": g("DRY_BIOMASS_POTENTIAL_MAX_MID_AND_LAST_LEAVES_MG")}
    lgd = {"first": 298.0, "other": g("GROWTH_DURATION_MID_AND_LAST_LEAVES_DD")}       # rank-1 gd/am: the published MTG rows (298 / 15)
    lam = {"first": 15.0, "other": g("AGE_MAXIMUM_GROWTH_MID_AND_LAST_LEAVES_DD")}
    rows = [_row(c0="1", c22=_f(plant_age_dd))]; prov = [dict(kind="plant", plant_id=obs["plant_id"], degree_days=plant_age_dd, age="assumed_age")]
    prev_xyz = np.asarray(obs["stem_base_mm"], float); top = 0
    for i, (nd, lf) in enumerate(zip(obs["nodes"], obs["leaves"])):
        rank = lf["model_rank"]; top = rank; xyz = np.asarray(nd["xyz_mm"], float); v = xyz - prev_xyz; L_mm = float(np.linalg.norm(v))
        app = appearance_dd(rank, block); age = max(plant_age_dd - app, 0.0) if app is not None else 0.0
        if i == 0 or nd["internode_mm"] >= SHARED_NODE_MM:
            irank = 0 if i == 0 else rank; F1, F2 = (t1, t2) if irank > 5 else (f1, f2)
            dw = (max(L_mm, 0.01) / F1) ** (1 / F2) / 0.9                                   # structural = 0.9 dw (importer); model_derived
            o, inc = ori_inc(v) if L_mm > 0 else (0.0, 0.0)
            rows.append(_row(c1="I0" if i == 0 else f"I{rank}", c7="0", c8=str(irank), c9="1", c10=_f(o), c11="0", c12=_f(inc), c13="0",
                             c17=_f(dw), c18="0", c19=_f(max(imax, dw)), c20=_f(igd), c21=_f(iam), c22=_f(age)))
            prov.append(dict(kind="internode", rank=irank, length=nd["internode_provenance"], dry_mass="model_derived", age="assumed_age"))
            prev_xyz = xyz
        else:
            lf["flags"].append("shared_node")
        cls = "first" if rank == 1 else "other"; area = lf["area_b_mm2"] or 0.0
        dw = area / sla / (0.9 * (0.81 if cls == "first" else 0.75))                    # blade area = 0.9 * frac_blades * dw * SLA (Leaf.calcDimensions)
        fl = list(lf["flags"]) + (["appearance_beyond_model_curve"] if app is None else []) + (["dw_above_potential_max"] if dw > lmax[cls] else [])
        rows.append(_row(c2=f"xL{rank}", c7="0", c8=str(rank), c9="1", c10=_f((lf["azimuth_deg"] + ORI_OFFSET) % 360.0), c11="0",
                         c12=_f(90.0 - lf["elevation_deg"]), c13="0", c17=_f(dw), c18="0", c19=_f(max(lmax[cls], dw)), c20=_f(lgd[cls]), c21=_f(lam[cls]), c22=_f(age)))
        prov.append(dict(kind="leaf", leaf_uid=lf["leaf_uid"], rank=rank, orientation=lf["geometry"], inclination=lf["geometry"], dry_mass="model_derived (area (b) / SLA)",
                         length="not representable (mass-based import)", age="assumed_age", flags=fl))
    rows.append(_row(c1=f"M{top + 1}", c7="0", c8=str(top + 1), c9="1", c17="0", c22="0"))
    prov.append(dict(kind="main_meristem", rank=top + 1))
    return rows, prov


# ------------------------------------------------------------------ hand-data template (OPTION_C step 5)
def published_main_rows(csv_path, head=STATIC_HEAD):
    """Main-stem rows of a published static scenario: {rank: dict(internode=row, leaf=row)} (lists of 33 strings)."""
    lines = Path(csv_path).read_bytes().decode("utf-8-sig").split("\n")[head + 1:]
    out = {}
    for ln in lines:
        c = (ln.split(",") + [""] * NCOL)[:NCOL]
        if c[1].startswith("I") and c[7] == "0":
            out.setdefault(int(c[8]), {})["internode"] = c
        elif c[2].startswith("xL") and c[7] == "0":
            out.setdefault(int(c[8]), {})["leaf"] = c
    return out


def template_static_rows(template_rows, published_csv, divergence_deg=137.0):
    """Static MTG of a template plant: main-stem ranks of the hand template. Per rank: leaf inclination = hand insertion from vertical
    (mean, sd = hand theta0 sd), orientation = rank x divergence (the scenario's own COUNTER_CLOCKWISE_ORIENTATION 137 deg, model_default),
    internode width = hand stem thickness where measured; petiole length, petiole width, blade area, rachis dw and internode lengths /
    tilts = the published 1_50 main-stem row of the same rank (model_published; the hand data have no lengths or areas)."""
    pub = published_main_rows(published_csv); rows = [["1"] + [""] * (NCOL - 1)]; prov = [dict(kind="plant", source="hand_template")]
    for t in template_rows:
        r = int(t["rank"]); p = pub.get(r) or pub[max(k for k in pub if "leaf" in pub[k])]
        it = list(p.get("internode") or pub[r - 1]["internode"]); it[1] = f"I{r}"; it[8] = str(r); it[9] = "1"   # the hand data record this node: present
        th = t.get("stem_thickness_mean_mm")
        if th not in (None, ""):
            it[16] = _f(float(th) / 1000)
        rows.append(it); prov.append(dict(kind="internode", rank=r, length="model_published", width="hand_measured" if th not in (None, "") else "model_published"))
        lf = list(p["leaf"]); lf[2] = f"xL{r}"; lf[8] = str(r)
        lf[10] = _f((r * divergence_deg) % 360.0); lf[11] = "0"
        lf[12] = _f(float(t["insertion_from_vertical_mean_deg"])); lf[13] = _f(float(t["theta0_sd_deg"] or 0))
        rows.append(lf); prov.append(dict(kind="leaf", rank=r, inclination="hand_measured (insertion from vertical)", inclination_sd="hand_measured (theta0 sd)",
                                          orientation="model_default (rank x 137 deg)", size="model_published (1_50 rank row)",
                                          leaflets_hand=dict(p5=t["p_leaflets_5"], p6=t["p_leaflets_6"], p7=t["p_leaflets_7"]),
                                          leaflets_model="by rank class (first 3 / mid 5 / last 7 leaflets): hand counts not representable per leaf"))
    return rows, prov


def published_branch_rows(csv_path, head=STATIC_HEAD):
    """First side branch of the published static plant: (branch internode rows by branch rank, branch leaf rows by branch rank)."""
    lines = Path(csv_path).read_bytes().decode("utf-8-sig").split("\n")[head + 1:]; bi, bl = {}, {}
    for ln in lines:
        c = (ln.split(",") + [""] * NCOL)[:NCOL]
        if c[7] == "1" and (c[2].startswith("xI") or c[2].startswith("I")):
            bi.setdefault(int(c[8]), c)
        elif c[7] == "1" and c[3].startswith("xL"):
            bl.setdefault(int(c[8]), c)
    return bi, bl


def template_v1_static_rows(v1_csv, published_csv, divergence_deg=137.0):
    """Template-v1 plant (A3; declared rule): main compound leaves = round(compound_leaves_per_main_stem) ranks 1..n, inclination = 90 - theta0
    (rank mean, sd); side axes = round(side_axes_per_plant) at the most frequent attach ranks <= n (ties -> lowest ranks), axis inclination
    = 90 - mean(angle_from_horizontal_values), azimuth = the attached main leaf's; leaves per side axis = round(n_items / n_axes) (18 / 12),
    inclination = 90 - side theta0; orientation = rank x 137 deg (model_default); sizes = published 1_50 rows (model_published); stem
    thickness -> internode width (hand_measured). Trusses / flowers / fruits: no row type in the static importer -> not_representable."""
    V = list(csv.DictReader(open(v1_csv, encoding="utf-8")))
    get = lambda o, r, p: next((x for x in V if x["organ"] == o and x["rank"] == str(r) and x["parameter"].startswith(p)), None)
    pub = published_main_rows(published_csv); bi, bl = published_branch_rows(published_csv)
    n_main = int(round(float(get("main_stem", "all", "compound_leaves_per_main_stem")["value"])))
    n_ax = int(round(float(get("side_axis", "all", "side_axes_per_plant")["value"])))
    cnt = json.loads(get("side_axis", "all", "attach_main_rank_counts")["value"])
    ranks_ax = sorted(sorted((int(k) for k in cnt if int(k) <= n_main), key=lambda k: (-cnt[str(k)], k))[:n_ax])
    ang = json.loads(get("side_axis", "all", "angle_from_horizontal_values")["value"]); ax_inc = 90.0 - float(np.mean(ang))
    sl = get("side_axis_compound_leaf", "all", "theta0_base_above_horizontal"); n_sl = int(round(float(sl["n_items"]) / float(get("side_axis", "all", "side_axes_per_plant")["n_items"])))
    rows = [["1"] + [""] * (NCOL - 1)]; prov = [dict(kind="plant", source="MTG_TEMPLATE_v1", sha256=hashlib.sha256(Path(v1_csv).read_bytes()).hexdigest(), n_main=n_main,
                                                     side_axes_at=ranks_ax, side_axis_inclination=ax_inc, side_leaves_per_axis=n_sl,
                                                     not_representable=["trusses", "flowers_per_truss", "fruits_per_plant (pending_user anyway)"])]
    for r in range(1, n_main + 1):
        p = pub.get(r) or pub[max(pub)]
        it = list(p.get("internode") or pub[r - 1]["internode"]); it[1] = "I0" if r == 1 else f"I{r}"; it[8] = "0" if r == 1 else str(r); it[9] = "1"
        th = get("main_stem", r, "stem_thickness")
        if th and th["value"]:
            it[16] = _f(float(th["value"]) / 1000)
        rows.append(it); prov.append(dict(kind="internode", rank=r, length="model_published", width="hand_measured" if th and th["value"] else "model_published"))
        t0 = get("main_compound_leaf", r, "theta0_base_above_horizontal"); ori = (r * divergence_deg) % 360.0
        lf = list(p["leaf"]); lf[2] = f"xL{r}"; lf[8] = str(r); lf[9] = "1"; lf[10] = _f(ori); lf[11] = "0"
        lf[12] = _f(90.0 - float(t0["value"])); lf[13] = _f(float(t0["sd"] or 0))
        rows.append(lf); prov.append(dict(kind="leaf", order=0, rank=r, inclination="hand_measured (90 - theta0)", orientation="model_default", size="model_published"))
        if r in ranks_ax:
            for k in range(1, n_sl + 1):
                b = list(bi.get(k) or bi[max(bi)]); b[2] = "xI" if k == 1 else f"I{k}"; b[8] = str(k); b[9] = "1"
                b[10] = _f(ori); b[12] = _f(ax_inc if k == 1 else float(b[12] or 0))
                rows.append(b); prov.append(dict(kind="side_internode", attach_rank=r, rank=k, inclination="hand_measured (90 - axis angle)" if k == 1 else "model_published"))
                q = list(bl.get(k) or bl[max(bl)]); q[3] = "xL"; q[8] = str(k); q[9] = "1"; q[12] = _f(90.0 - float(sl["value"])); q[13] = _f(float(sl["sd"] or 0))
                rows.append(q); prov.append(dict(kind="side_leaf", attach_rank=r, rank=k, inclination="hand_measured (90 - side theta0)", size="model_published"))
    return rows, prov
