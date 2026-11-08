"""microtom_synth_v2 — Micro-Tom synthetic individuals as a plant_tree_v1 first, meshes from the tree.

Changes from v1 (CHANGES_v1_to_v2.md): (1) `lf_len` is the realised largest leaflet (terminal factor 1.0, sizes rescaled so the
plant's largest leaflet equals lf_len x its jitter; lengths clipped at LF_MAX_MM = 22.5, user_measured); (2) leaf size = final
size(rank) x expansion(age): the rank profile is v1's smooth `prof` (assumption), expansion is the engine's logistic in thermal
time (K, MID_DD of the engine's leaflet kind) applied as sqrt (length ~ sqrt(mass)); the visible stage comes from ages, not an
ad-hoc youngest-leaf table; (3) Micro-Tom architecture (Vicente 2015, literature): the main stem ends in two successive trusses,
side axes grow from the upper axils with 1-3 leaves then a truss, no further branching; (4) rachis as polyline + attachment arc
lengths; (5) leaflet shape option A (v1 parametric outline, default) or B (outline library: 28 dissected leaflets, tag
user_measured_shape); (6) every parameter keeps a provenance tag; (7) v1 seeding (rng_for(seed, key) per organ) and draw order
for the organs that exist in v1.

v1 (tomato3d/synth/microtom.py) is imported for its RNG, jitter and helpers and is NOT changed. Everything here is synthetic.
usage: python -m tomato3d.synth.microtom_v2 --out dir/ [--params p.json] [--preset S02] [--seed N] [--shape library]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tomato3d.plant_tree import geometry as g
from tomato3d.plant_tree import obj as po
from tomato3d.plant_tree.schema import new_tree, validate
from tomato3d.synth import microtom as v1
from tomato3d.synth.microtom import D2R, DOWN, UP, add, clamp, cross, dot, gauss, js_round, mul, norm, rng_for, rot_around, smooth, sub

SCHEMA = "microtom_synth_v2"
LIBRARY = Path("handoff/leaflet_shape_20261014/microtom_leaflet_contours_20261014.json")
# ---- v2 parameters: v1 DEF + new keys (each with a provenance tag); v1 defaults unchanged unless a numbered change says so
V2_EXTRA = [("terminal_factor", 1.0, "a"), ("lf_max_mm", 22.5, "m"), ("phyllochron_dd", 37.9, "l"), ("expand_k", 0.025, "a"), ("expand_mid_dd", 150.0, "a"),
            ("dd_per_day", 20.0, "a"), ("side_axes", 2, "l"), ("shape_source", "param", "a")]
DEF = dict(v1.DEF); DEF.update({k: d for k, d, _ in V2_EXTRA})
PROVENANCE = dict(v1.PROVENANCE); PROVENANCE.update({k: v1.TAG_PROVENANCE[t] for k, _, t in V2_EXTRA})
PROVENANCE_NOTE = {"terminal_factor": "change 1: terminal no longer 1.08 x the pairs", "lf_max_mm": "measured maximum leaflet length (leaflet_shape_20261014)",
                   "phyllochron_dd": "engine PHYLLOCHRON_DD (literature)", "expand_k": "engine K[Leaflet] = 0.025 /dd (engine default, read not changed)",
                   "expand_mid_dd": "engine MID_DD[Leaflet] = 150 dd (engine default)", "dd_per_day": "20 dd per day at the growth-room temperature (assumption)",
                   "side_axes": "Vicente 2015: lateral shoots from the upper axils after the main stem terminates", "shape_source": "param (v1 outline) or library"}


def params_for(preset: str | None = None, **over) -> dict:
    P = dict(DEF)
    if preset:
        for k, v in v1.PRESETS[preset].items():
            if k in DEF:
                P[k] = v
    for k, v in over.items():
        if k in DEF:
            P[k] = v
    return P


# ------------------------------------------------------------------ change 2: expansion in thermal time (engine form)
def expansion(age_dd: float, k: float, mid: float) -> float:
    """sqrt of the engine's logistic mass fraction ((s - s0) / (1 - s0)); length ~ sqrt(mass). 0 at age 0, -> 1."""
    s0 = 1 / (1 + math.exp(k * mid)); s = 1 / (1 + math.exp(-k * (age_dd - mid)))
    return math.sqrt(max(0.0, min(1.0, (s - s0) / (1 - s0))))


def final_size(k: int, n_truss: int) -> float:
    """v1's rank profile `prof`, unchanged (assumption): rises over the first 60 % of the ranks, eases 15 % near the truss."""
    return (0.45 + 0.55 * smooth(0, 0.6 * n_truss, k - 1)) * (1 - 0.15 * smooth(0.7 * n_truss, n_truss, k))


# ------------------------------------------------------------------ outline library (change 5, option B)
_LIB = None


def library() -> list[dict]:
    global _LIB
    if _LIB is None:
        d = json.loads(LIBRARY.read_text(encoding="utf-8"))
        _LIB = [dict(g.normalise_contour(x["contour_mm"]), id=x["id"], tag="user_measured_shape") for x in d["leaflets"]]
    return _LIB


def z(p):
    return g.build_to_zup(p)


# ------------------------------------------------------------------ generator
def generate(P: dict) -> dict:
    """plant_tree_v1 (units mm, pot top centre, z up) from v2 parameters; geometry built in v1's y-up frame then stored z up."""
    seed = P["seed"]; cv = P["cv"]; nT = P["n_truss"]; V = min(P["visible"], nT); flowered = V >= nT
    tree = new_tree("synthetic", "mm", "pot top centre, z up", f"{SCHEMA} seed {seed}", plant_id=f"synth_{seed}", synthetic=True)
    tree["params"] = dict(schema=SCHEMA, params={k: P[k] for k in DEF}, provenance=dict(PROVENANCE), provenance_note=PROVENANCE_NOTE)
    tree["plant"]["attrs"].update(organ_split="synthetic_truth", n_truss=nT, visible=V, flowered=flowered)
    tree["provenance"].update(geometry="assumption", length_mm="assumption", lf_len="user_measured", lw_ratio="user_measured")

    def jit(key, sd):
        return 1 + cv * sd * gauss(rng_for(seed, key))

    # ---- main axis (as v1)
    leanAz = rng_for(seed, "lean")() * 2 * math.pi; lean = P["lean"] * D2R
    axis = norm(add(UP, [math.sin(lean) * math.cos(leanAz), 0, math.sin(lean) * math.sin(leanAz)]))
    y = P["hypocotyl"] * jit("hyp", 0.6); cur = mul(axis, y); stemPts = [[0, 0, 0], list(cur)]; nodes = []
    for k in range(1, V + 1):
        il = P["internode"] * jit("int" + str(k), 1) * (0.35 if k == 2 else 1) * (0.4 if k == 1 else 1)
        if k == V and V < nT:
            il *= 0.6
        wob = rng_for(seed, "wob" + str(k)); gx = 0.06 * gauss(wob); gz = 0.06 * gauss(wob)
        cur = add(cur, mul(norm(add(axis, [gx, 0, gz])), il)); nodes.append(list(cur)); stemPts.append(list(cur))
    topNode = list(cur)
    if flowered:
        cur = add(cur, mul(axis, P["internode"] * 0.6)); stemPts.append(list(cur))
    rads = [1.9 - 0.9 * i / max(1, len(stemPts) - 1) for i in range(len(stemPts))]
    tree["axes"].append(dict(id="M", kind="main", parent_axis=None, parent_node=None, tube=dict(kind="tube", polyline=[z(p) for p in stemPts], radius=rads, segs=8), flags=[]))
    tree["params"]["hypocotyl_mm"] = y
    # ---- cotyledons (as v1; assumption)
    if P["cotyledons"]:
        cn = mul(axis, y); caz = rng_for(seed, "coty")() * math.pi
        for c in range(2):
            az = caz + c * math.pi; dd = norm([math.cos(az), 0.25, math.sin(az)]); nn = norm(sub(UP, mul(dd, dot(UP, dd))))
            base = add(cn, mul(dd, 1.5)); L = 15 * jit("coty" + str(c), 0.5)
            tree["cotyledons"].append(dict(id=f"cotyledon_C{c + 1}", surface=dict(kind="param_blade", base=z(base), d=z(dd), n=z(nn), L=L, W=4.2, cup=0.1, bend=0.2, nu=12, nv=7), flags=[]))
    # ---- ages (change 2): plant thermal age from the phyllochron; leaf k emerged (k-1) phyllochrons after the first
    ph = P["phyllochron_dd"]; plant_age = (V - 1) * ph + (2 * ph if flowered else 0.6 * ph)
    tree["plant"]["attrs"].update(plant_age_dd=plant_age, plant_age_d=plant_age / P["dd_per_day"])
    # ---- leaves (geometry as v1's compound_leaf; sizes by change 1-2; rachis stored as polyline + attachments)
    az0 = rng_for(seed, "az0")() * 2 * math.pi
    lib = library() if P["shape_source"] == "library" else None
    pending = []                                                  # leaflet size records, rescaled after all are drawn (change 1)

    def compound_leaf(id_, node_pos, az, elev, scale, nPairs, age_dd, axis_id, node_id, rank):
        r = rng_for(seed, id_); L = P["leaf_len"] * scale * jit(id_ + ":len", 1)
        h = [math.cos(az), 0, math.sin(az)]; d0 = norm(add(mul(h, math.sin(elev)), mul(UP, math.cos(elev)))); droop = P["droop"] * (0.7 + 0.6 * r())

        def at(s): return add(add(node_pos, mul(d0, s * L)), mul(DOWN, droop * s * s * L * 0.45))
        def tan(s): return norm(add(mul(d0, L), mul(DOWN, droop * 2 * s * L * 0.45)))
        pts = [at(i / 14) for i in range(15)]; rad = [(0.75 - 0.4 * (i / 14)) * max(0.6, scale) for i in range(15)]
        pl = [z(p) for p in pts]
        leaf = dict(id=id_, node=node_id, axis=axis_id, rank=rank, petiole=None, rachis=dict(kind="tube", polyline=pl, radius=rad, segs=6), attachments=[],
                    length_mm=L, azimuth_deg=float((math.degrees(math.atan2(-h[2], h[0])) + 360) % 360), elevation_deg=float(90 - elev / D2R),
                    age_d=age_dd / P["dd_per_day"], age_dd=age_dd, expansion=expansion(age_dd, P["expand_k"], P["expand_mid_dd"]), final_size=scale, flags=[])
        atts = {}

        def leaflet(name, s, side, posF, kind, pair):
            rr = rng_for(seed, id_ + "__" + name); p = at(s); t = tan(s)
            sd = cross(t, UP); sd = h if math.hypot(sd[0], sd[1], sd[2]) < 1e-3 else norm(sd)
            alpha = (P["pair_angle"] + 8 * cv / 0.12 * gauss(rr)) * D2R
            d = t if side == 0 else norm(add(mul(t, math.cos(alpha)), mul(sd, side * math.sin(alpha))))
            d = norm(add(d, mul(DOWN, 0.12 * P["bend"]))); n0 = norm(sub(UP, mul(d, dot(UP, d)))); n0 = rot_around(n0, d, (side * 10 + 12 * gauss(rr) * cv / 0.12) * D2R)
            raw = posF * math.sqrt(scale) * jit(id_ + "__" + name + ":lf", 1) * (0.32 if kind == "I" else 1.0)
            ratio = clamp(P["lw_ratio"] + 0.25 * gauss(rr) * (1 if cv > 0 else 0), 1.0, 2.2)
            pl_ = max(1.2, 0.12 * raw * P["lf_len"]); pe = add(p, mul(d, pl_))
            lid = id_ + "__" + name
            outline = None
            if lib is not None:
                outline = lib[int(rng_for(seed, lid + ":shape")() * len(lib))]
            lt = dict(id=lid, leaf=id_, type=kind, side="C" if side == 0 else ("L" if side > 0 else "R"), pair=pair, attachment_s_mm=g.polyline_length(pl[: int(round(s * 14)) + 1]) if s < 1 else g.polyline_length(pl),
                      base=z(pe), frame=dict(d=z(d), n=z(n0), b=z(cross(n0, d))), length_mm=None, width_mm=None, area_mm2=None, normal=z(n0),
                      surface=dict(kind="param_blade", base=z(pe), d=z(d), n=z(n0), L=None, W=None, cup=P["cup"], bend=P["bend"], nu=12, nv=7, outline=outline),
                      petiolule=dict(kind="tube", polyline=[z(p), z(pe)], radius=[0.35, 0.3], segs=5), petiolule_mm=pl_, age_d=leaf["age_d"], expansion=leaf["expansion"],
                      shape_id=outline["id"] if outline else None, flags=[])
            pending.append((lt, raw, ratio)); tree["leaflets"].append(lt)
            key = (pair, kind) if kind == "P" else (lid,)
            atts.setdefault(key, dict(s_mm=lt["attachment_s_mm"], leaflets=[]))["leaflets"].append(lid)
        p0 = 0.32
        for j in range(1, nPairs + 1):
            s = p0 + (0.92 - p0) * (j - 1) / max(1, nPairs - 0.4); posF = 0.72 + 0.28 * (j / nPairs)
            leaflet("P" + str(j) + "L", s, 1, posF, "P", j); leaflet("P" + str(j) + "R", s + 0.015, -1, posF, "P", j)
            if j < nPairs and rng_for(seed, id_ + ":I" + str(j))() < P["inter_p"]:
                sI = s + 0.5 * (0.92 - p0) / max(1, nPairs - 0.4); leaflet("I" + str(j), sI, (1 if j % 2 else -1), 0.5, "I", j)
        leaflet("T", 1.0, 0, P["terminal_factor"], "T", 0)
        leaf["attachments"] = sorted(atts.values(), key=lambda a: a["s_mm"]); tree["leaves"].append(leaf)
        return leaf

    for k in range(1, V + 1):
        az = az0 if k == 1 else (az0 + math.pi * 0.92 if k == 2 else az0 + math.pi * 0.92 + (k - 2) * P["phyllo"] * D2R)
        az += 8 * D2R * gauss(rng_for(seed, "az" + str(k))) * cv / 0.12
        rel = V - k; age = plant_age - (k - 1) * ph
        scale = final_size(k, nT) * expansion(age, P["expand_k"], P["expand_mid_dd"])
        fr = (k - 1) / max(1, nT - 1)
        elev = (P["elev_low"] + (P["elev_high"] - P["elev_low"]) * fr - 18 * ((2 - rel) / 2 if rel < 2 else 0) * (1 if V < nT else 0.5)) * D2R
        elev += 5 * D2R * gauss(rng_for(seed, "el" + str(k))) * cv / 0.12
        pairs = 1 if k <= 2 else (min(2, P["max_pairs"]) if k == 3 else P["max_pairs"])
        rank = 1 if k <= 2 else k - 1; nid = f"N{k:02d}"; lid = "L" + str(k).rjust(2, "0")
        s_node = g.polyline_length([z(p) for p in stemPts[: k + 1]])
        tree["nodes"].append(dict(id=nid, axis="M", index=k, rank=rank, s_mm=s_node, position=z(nodes[k - 1]), leaves=[lid], trusses=[], flags=[]))
        compound_leaf(lid, add(nodes[k - 1], mul([math.cos(az), 0, math.sin(az)], 1.4)), az, elev, scale, pairs, age, "M", nid, rank)
    # ---- change 1: lf_len = realised largest leaflet of the plant (main stem), then the measured clip
    raw_max = max((raw for lt, raw, _ in pending if lt["type"] != "I" and not lt["leaf"].startswith("S")), default=1.0)
    for lt, raw, ratio in pending:
        Lf = P["lf_len"] * raw / raw_max
        if Lf > P["lf_max_mm"]:
            Lf = P["lf_max_mm"]; lt["flags"].append("clipped_to_measured_max")
        Wf = Lf / ratio; s = lt["surface"]; s["L"], s["W"] = Lf, Wf
        V_, F_, area = g.blade_mesh(s["base"], s["d"], s["n"], Lf, Wf, s["cup"], s["bend"], 12, 7, s.get("outline"))
        lt.update(length_mm=Lf, width_mm=Wf, area_mm2=area)
    # ---- change 3: two successive trusses end the main stem; side axes from the upper axils (1-3 leaves, then a truss)
    fl_count = [0]

    def truss(id_, base, dir_, nFl, axis_id):
        r = rng_for(seed, id_); e1 = add(base, mul(dir_, 10)); g1 = 0.3 * gauss(r); g2 = 0.3 * gauss(r)
        e2 = add(e1, mul(norm(add(dir_, [g1, -0.2, g2])), 8)); flowers = []
        for i in range(nFl):
            s = i / max(1, nFl - 1); p = add(e1, mul(sub(e2, e1), s)); fa = r() * 2 * math.pi; fd = norm([math.cos(fa), 0.6 + 0.4 * r(), math.sin(fa)]); fc = add(p, mul(fd, 4))
            nrm = norm(fd); bx = norm(cross(nrm, UP if abs(nrm[1]) < 0.9 else [1, 0, 0])); by = cross(nrm, bx); Vf = [fc]; Ff = []
            for pt in range(5):
                a = pt / 5 * 2 * math.pi + r(); dirp = add(mul(bx, math.cos(a)), mul(by, math.sin(a))); tip = add(add(fc, mul(dirp, 4.2)), mul(nrm, -0.8))
                l_ = add(fc, mul(add(mul(bx, math.cos(a - 0.35)), mul(by, math.sin(a - 0.35))), 1.6)); rr = add(fc, mul(add(mul(bx, math.cos(a + 0.35)), mul(by, math.sin(a + 0.35))), 1.6))
                b0 = len(Vf); Vf.extend([l_, tip, rr]); Ff.append([0, b0, b0 + 1]); Ff.append([0, b0 + 1, b0 + 2])
            flowers.append(dict(id=f"{id_}_flower{i + 1}", position=z(fc), mesh=dict(vertices=[z(v) for v in Vf], faces=Ff), pedicel=dict(kind="tube", polyline=[z(p), z(fc)], radius=[0.3, 0.25], segs=4)))
            fl_count[0] += 1
        tree["trusses"].append(dict(id=id_, node=None, axis=axis_id, peduncle=dict(kind="tube", polyline=[z(base), z(e1), z(e2)], radius=[0.7, 0.55, 0.45], segs=5),
                                    flowers=flowers, flower_count=nFl, stage="flowering", flags=[]))

    def flower_count(key):
        r = rng_for(seed, key); return clamp(js_round(P["flowers"] + 1.2 * gauss(r)), 4, 8)

    if flowered:
        truss("T1", topNode, norm(add(axis, [0.5, 0, 0.3])), flower_count("T1"), "M")
        if P["truss2"]:
            truss("T2", cur, norm(add(axis, [-0.4, 0.2, -0.35])), flower_count("T2"), "M")
        n_side = int(P["side_axes"]) if P["side_leaves"] > 0 else 0; t_index = 3
        for j in range(1, n_side + 1):
            kn = V - j                                           # upper axils: node V-1, V-2, ...
            if kn < 1 or kn > len(nodes):
                break
            sn = nodes[kn - 1]; sid_ax = f"S{j}"
            saz = az0 + math.pi * 0.92 + (kn - 2) * P["phyllo"] * D2R + 0.35 + (j - 1) * 0.9
            sdir = norm(add(mul(axis, 0.85), mul([math.cos(saz), 0, math.sin(saz)], 0.55)))
            n_leaves = max(1, min(3, int(P["side_leaves"]) - (j - 1)))
            spts = [sn]; sc = list(sn)
            for s in range(1, n_leaves + 1):
                sc = add(sc, mul(sdir, P["internode"] * 0.75 * jit(("sint" if j == 1 else f"s{j}int") + str(s), 1))); spts.append(list(sc))
            tree["axes"].append(dict(id=sid_ax, kind="side", parent_axis="M", parent_node=f"N{kn:02d}",
                                     tube=dict(kind="tube", polyline=[z(p) for p in spts], radius=[1.1 - 0.4 * i / len(spts) for i in range(len(spts))], segs=6), flags=[]))
            for s in range(1, n_leaves + 1):
                a2 = saz + s * P["phyllo"] * D2R; lid = f"{sid_ax}_L" + str(s).rjust(2, "0"); nid = f"{sid_ax}_N{s:02d}"
                age = max(0.0, (plant_age - (V - 1) * ph) - (s - 1) * ph)       # side leaves start when the main stem has terminated
                scale = 0.62 * final_size(min(nT, 3 + s), nT) / final_size(3, nT) * expansion(age + 0.5 * ph, P["expand_k"], P["expand_mid_dd"])
                tree["nodes"].append(dict(id=nid, axis=sid_ax, index=s, rank=s, s_mm=g.polyline_length([z(p) for p in spts[: s + 1]]), position=z(spts[s]), leaves=[lid], trusses=[], flags=[]))
                compound_leaf(lid, spts[s], a2, (P["elev_high"] + 10) * D2R, scale, min(2, P["max_pairs"]), age, sid_ax, nid, s)
            # side-leaf sizes were not in `pending` at rescale time: size them now with the same rule
            for lt, raw, ratio in pending:
                if lt["leaf"].startswith(sid_ax) and lt["length_mm"] is None:
                    Lf = min(P["lf_len"] * raw / raw_max, P["lf_max_mm"]); Wf = Lf / ratio; sf = lt["surface"]; sf["L"], sf["W"] = Lf, Wf
                    _, _, area = g.blade_mesh(sf["base"], sf["d"], sf["n"], Lf, Wf, sf["cup"], sf["bend"], 12, 7, sf.get("outline")); lt.update(length_mm=Lf, width_mm=Wf, area_mm2=area)
            truss(f"T{t_index}", sc, sdir, flower_count(f"T{t_index}"), sid_ax); t_index += 1
    # leaves that droop below the pot top rest on it (as v1): applied to stored geometry
    def lift(p):
        if p[2] < 0.4:
            p[2] = 0.4
    for lt in tree["leaflets"]:
        lift(lt["base"]); lift(lt["surface"]["base"]); [lift(p) for p in lt["petiolule"]["polyline"]]
    for lf in tree["leaves"]:
        [lift(p) for p in lf["rachis"]["polyline"]]
    return validate(tree)


# ------------------------------------------------------------------ outputs
def organs_csv(tree: dict) -> str:
    rows = ["object,category,leaf,length_mm,width_mm,area_mm2,age_d,expansion"]
    for lf in tree["leaves"]:
        rows.append(",".join(["rachis_" + lf["id"], "rachis", lf["id"], f"{lf['length_mm']:.2f}", "", "", f"{lf['age_d']:.2f}", f"{lf['expansion']:.3f}"]))
    for lt in tree["leaflets"]:
        rows.append(",".join([lt["id"], "blade_" + lt["type"], lt["leaf"], f"{lt['length_mm']:.2f}", f"{lt['width_mm']:.2f}", f"{lt['area_mm2']:.1f}", f"{lt['age_d']:.2f}", f"{lt['expansion']:.3f}"]))
    return "\n".join(rows)


def write_individual(P: dict, out: Path, preset: str | None = None, **extra) -> dict:
    out.mkdir(parents=True, exist_ok=True); tree = generate(P)
    (out / "tree.json").write_text(json.dumps(tree, ensure_ascii=False), encoding="utf-8")
    (out / "plant.obj").write_text("# synthetic: true\n" + po.to_obj(tree, header=f"# Micro-Tom synthetic individual · {SCHEMA} · units mm · z up · seed {P['seed']}"), encoding="utf-8")
    (out / "organs.csv").write_text(organs_csv(tree) + "\n", encoding="utf-8")
    pj = dict(schema=SCHEMA, units="mm", frame="pot top centre, z up", generator="tomato3d.synth.microtom_v2", params={k: P[k] for k in DEF},
              provenance=dict(PROVENANCE), provenance_note=PROVENANCE_NOTE, preset=preset, synthetic=True, tree="tree.json (plant_tree_v1)", **extra)
    (out / "params.json").write_text(json.dumps(pj, ensure_ascii=False, indent=2), encoding="utf-8")
    return tree


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--params"); ap.add_argument("--preset", choices=sorted(v1.PRESETS)); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int); ap.add_argument("--shape", choices=("param", "library")); a = ap.parse_args(argv)
    if a.params:
        d = json.loads(Path(a.params).read_text(encoding="utf-8")); src = d.get("params", d); P = params_for(None, **{k: v for k, v in src.items() if k in DEF}); preset = d.get("preset")
    else:
        P, preset = params_for(a.preset), a.preset
    if a.seed is not None:
        P["seed"] = a.seed
    if a.shape:
        P["shape_source"] = a.shape
    from tomato3d.plant_tree.summary import summary
    tree = write_individual(P, Path(a.out), preset); print(json.dumps(dict(out=a.out, **summary(tree)["counts"])))


if __name__ == "__main__":
    main()
