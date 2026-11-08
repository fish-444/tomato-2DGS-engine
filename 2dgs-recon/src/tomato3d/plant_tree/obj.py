"""OBJ in both directions. `to_obj` writes the v1 object names (stem, rachis_L01, L01__P1L, petiolule_L01__P1L, L01__T,
S1_L01…, T1_peduncle, T1_pedicel1, T1_flower1, cotyledon_C1) with an optional `<plant>__` prefix; param surfaces and
tubes are regenerated with the v1 maths, stored meshes are written verbatim. `from_obj_v1` rebuilds a tree from any v1
synthetic OBJ using its object names; meshes are kept verbatim (so the round trip is exact) and descriptors
(centrelines, lengths, widths, normals, outlines) are derived from them with provenance `derived_from_mesh`.
"""

from __future__ import annotations

import re

import numpy as np

from tomato3d.plant_tree import geometry as g
from tomato3d.plant_tree.schema import new_tree, validate

CAT = dict(stem="stem", rachis="rachis", petiolule="petiolule", P="blade_P", T="blade_T", I="blade_I", truss="truss", flower="flower", cotyledon="cotyledon")
SEGS = dict(stem=8, side_stem=6, rachis=6, petiolule=5, peduncle=5, pedicel=4)
BLADE_NU, BLADE_NV = 12, 7


def _fmt(x: float) -> str:
    s = f"{x:.3f}"
    return "0.000" if s in ("-0.000",) else s


# ------------------------------------------------------------------ meshes of one organ
def surface_mesh(s: dict):
    k = s["kind"]
    if k == "param_blade":
        V, F, _ = g.blade_mesh(s["base"], s["d"], s["n"], s["L"], s["W"], s["cup"], s["bend"], s.get("nu", BLADE_NU), s.get("nv", BLADE_NV), s.get("outline"))
        return V, F
    if k == "grid":
        nu, nv = s["nu"], s["nv"]; F = []
        for i in range(nu):
            for j in range(nv):
                a0 = i * (nv + 1) + j; a1 = a0 + 1; b0 = a0 + nv + 1; b1 = b0 + 1
                F.append([a0, b0, b1]); F.append([a0, b1, a1])
        return s["vertices"], F
    if k == "mesh":
        return s["vertices"], s["faces"]
    raise ValueError(f"surface kind {k!r} has no mesh (outline_polygon / ref need a param_blade to draw)")


def tube_mesh(t: dict):
    if t["kind"] == "tube":
        return g.tube_mesh(t["polyline"], t["radius"], t.get("segs", 6))
    return t["vertices"], t["faces"]


def organ_objects(tree: dict) -> list[dict]:
    """Ordered list of dict(name, cat, leaf, V, F) for every organ, in the tree's stored order if it has one."""
    out = []
    axes = {a["id"]: a for a in tree["axes"]}
    for a in tree["axes"]:
        V, F = tube_mesh(a["tube"])
        name = "stem" if a["kind"] == "main" else ("side_stem" if a["id"] in ("S1", "side_stem") else f"side_stem_{a['id']}")
        out.append(dict(name=a.get("obj_name", name), cat="stem", leaf="", V=V, F=F))
    for c in tree.get("cotyledons", []):
        V, F = surface_mesh(c["surface"]); out.append(dict(name=c.get("obj_name", c["id"]), cat="cotyledon", leaf="", V=V, F=F))
    lf_by = {lf["id"]: lf for lf in tree["leaves"]}; lt_by = {lt["id"]: lt for lt in tree["leaflets"]}
    for lf in tree["leaves"]:
        V, F = tube_mesh(lf["rachis"]); out.append(dict(name=lf.get("obj_name", "rachis_" + lf["id"]), cat="rachis", leaf=lf["id"], V=V, F=F))
        seen = []
        for at in lf["attachments"]:
            seen += at["leaflets"]
        for li in seen:
            lt = lt_by[li]
            if lt.get("petiolule"):
                V, F = tube_mesh(lt["petiolule"]); out.append(dict(name=lt.get("petiolule_obj_name", "petiolule_" + lt["id"]), cat="petiolule", leaf=lf["id"], V=V, F=F))
            V, F = surface_mesh(lt["surface"]); out.append(dict(name=lt.get("obj_name", lt["id"]), cat=CAT[lt["type"]], leaf=lf["id"], V=V, F=F))
    for t in tree.get("trusses", []):
        V, F = tube_mesh(t["peduncle"]); out.append(dict(name=t.get("obj_name", t["id"] + "_peduncle"), cat="truss", leaf="", V=V, F=F))
        for k, fl in enumerate(t["flowers"], 1):
            if fl.get("pedicel"):
                V, F = tube_mesh(fl["pedicel"]); out.append(dict(name=fl.get("pedicel_obj_name", f"{t['id']}_pedicel{k}"), cat="truss", leaf="", V=V, F=F))
            if fl.get("mesh"):
                out.append(dict(name=fl.get("obj_name", f"{t['id']}_flower{k}"), cat="flower", leaf="", V=fl["mesh"]["vertices"], F=fl["mesh"]["faces"]))
    order = tree.get("params", {}).get("_obj_order")
    if order:
        rank = {n: i for i, n in enumerate(order)}; out.sort(key=lambda o: rank.get(o["name"], 10 ** 6))
    return out


def to_obj(tree: dict, plant_prefix: str | None = None, header: str | None = None) -> str:
    """OBJ text (z up, mm). `plant_prefix` prepends '<prefix>__' to every object name (the old diagnostics' convention)."""
    pre = f"{plant_prefix}__" if plant_prefix else ""
    lines = [header or f"# plant_tree_v1 · source {tree['source']} · units {tree['units']} · frame {tree['frame']} · synthetic {str(tree['synthetic']).lower()}"]
    off = 1
    for o in organ_objects(tree):
        if not o["F"]:
            continue
        lines += [f"o {pre}{o['name']}", f"usemtl {o['cat']}"]
        lines += ["v " + " ".join(_fmt(float(c)) for c in p) for p in o["V"]]
        lines += ["f %d %d %d" % (f[0] + off, f[1] + off, f[2] + off) for f in o["F"]]
        off += len(o["V"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ parse
def parse_obj(text: str) -> list[dict]:
    objs, cur, voff = [], None, 0
    for line in text.splitlines():
        if line.startswith("o "):
            if cur:
                objs.append(cur)
            cur = dict(name=line[2:].strip(), cat="", V=[], F=[], voff=voff)
        elif line.startswith("usemtl ") and cur:
            cur["cat"] = line[7:].strip()
        elif line.startswith("v ") and cur:
            cur["V"].append([float(x) for x in line.split()[1:4]]); voff += 1
        elif line.startswith("f ") and cur:
            cur["F"].append([int(x.split("/")[0]) - 1 - cur["voff"] for x in line.split()[1:4]])
    if cur:
        objs.append(cur)
    return objs


NAME = dict(rachis=re.compile(r"^rachis_(.+)$"), petiolule=re.compile(r"^petiolule_(.+?)__(.+)$"), blade=re.compile(r"^((?:S\d+_)?L\d+)__(P(\d+)([LR])|T|I(\d+))$"),
            truss=re.compile(r"^(T\d+)_(peduncle|pedicel(\d+)|flower(\d+))$"), coty=re.compile(r"^cotyledon_(C\d+)$"))


def from_obj_v1(text: str, plant_id: str = "plant", input_status: str = "v1_synthetic") -> dict:
    """Rebuild a plant_tree_v1 from a v1 synthetic OBJ. Geometry is kept verbatim; descriptors are derived from it."""
    objs = parse_obj(text); prefix = None
    names = [o["name"] for o in objs]
    m = [n for n in names if n.endswith("__stem") or n == "stem"]
    if m and m[0] != "stem":
        prefix = m[0][:-len("__stem")]
    if prefix:
        for o in objs:
            if o["name"].startswith(prefix + "__"):
                o["name"] = o["name"][len(prefix) + 2:]
    by = {o["name"]: o for o in objs}
    hdr = text.splitlines()[0] if text else ""
    tree = new_tree("synthetic", "mm", "pot top centre, z up", input_status, plant_id=plant_id, synthetic=True)
    tree["notes"].append("rebuilt from a microtom_synth_v1 OBJ by object names; meshes verbatim, descriptors derived_from_mesh")
    tree["provenance"].update(geometry="derived_from_mesh", length_mm="derived_from_mesh", width_mm="derived_from_mesh", normal="derived_from_mesh")
    tree["params"]["_obj_order"] = names if not prefix else [n[len(prefix) + 2:] if n.startswith(prefix + "__") else n for n in names]
    tree["params"]["_obj_header"] = hdr

    def tube_entry(o, segs):
        ctr, rad = g.ring_centreline(o["V"], segs)
        return dict(kind="mesh", vertices=o["V"], faces=o["F"], centreline=ctr, radius=rad, segs=segs)

    # axes
    if "stem" in by:
        t = tube_entry(by["stem"], SEGS["stem"]); tree["axes"].append(dict(id="M", kind="main", parent_axis=None, parent_node=None, tube=t, flags=[]))
    nodes_by_leaf = {}
    # leaves and leaflets
    leaf_ids = [NAME["rachis"].match(n).group(1) for n in names if NAME["rachis"].match(n)]
    blades = {}
    for n in names:
        mb = NAME["blade"].match(n)
        if mb and by[n]["cat"].startswith("blade"):
            blades.setdefault(mb.group(1), []).append(n)
    for lid in leaf_ids:
        ra = by["rachis_" + lid]; t = tube_entry(ra, SEGS["rachis"]); pl = t["centreline"]
        side_axis = lid.split("_")[0] if lid.startswith("S") else None
        lf = dict(id=lid, node=None, axis=side_axis or "M", rank=None, petiole=None, rachis=t, attachments=[], length_mm=g.polyline_length(pl),
                  azimuth_deg=None, elevation_deg=None, age_d=None, expansion=None, flags=[])
        d0 = np.subtract(pl[min(3, len(pl) - 1)], pl[0]); lf["azimuth_deg"] = float(np.degrees(np.arctan2(d0[1], d0[0])) % 360)
        lf["elevation_deg"] = float(np.degrees(np.arctan2(d0[2], np.hypot(d0[0], d0[1]))))
        lf["curvature_deg_per_mm"] = g.polyline_turning_deg(pl) / (lf["length_mm"] or 1)
        atts = {}
        for bn in blades.get(lid, []):
            mb = NAME["blade"].match(bn); tag = mb.group(2); o = by[bn]
            typ = "T" if tag == "T" else ("I" if tag.startswith("I") else "P")
            side = "C" if typ == "T" else ("L" if tag.endswith("L") else ("R" if tag.endswith("R") else ("L" if int(mb.group(5)) % 2 else "R")))
            pair = 0 if typ == "T" else int(mb.group(3) or mb.group(5))
            nv = BLADE_NV if len(o["V"]) == (BLADE_NU + 1) * (BLADE_NV + 1) else None
            if nv is not None:
                desc = g.grid_descriptors(o["V"], BLADE_NU, BLADE_NV)
                surf = dict(kind="grid", nu=BLADE_NU, nv=BLADE_NV, vertices=o["V"])
                outline = g.grid_outline(o["V"], BLADE_NU, BLADE_NV, desc["base"], desc["d"], desc["b"])
            else:
                V = np.asarray(o["V"]); desc = dict(base=V[0].tolist(), tip=V[-1].tolist(), d=[0, 0, 1], n=[0, 0, 1], b=[1, 0, 0], length_mm=None, width_mm=None); surf = dict(kind="mesh", vertices=o["V"], faces=o["F"]); outline = None
            pet = by.get("petiolule_" + lid + "__" + tag)
            petiolule = tube_entry(pet, SEGS["petiolule"]) if pet else None
            att_pt = petiolule["centreline"][0] if petiolule else desc["base"]
            s_mm = g.arclength_of_nearest(pl, att_pt)
            lt = dict(id=bn, leaf=lid, type=typ, side=side, pair=pair, attachment_s_mm=s_mm, base=desc["base"], frame=dict(d=desc["d"], n=desc["n"], b=desc["b"]),
                      length_mm=desc["length_mm"], width_mm=desc["width_mm"], area_mm2=None, normal=desc["n"], surface=surf, outline=outline,
                      petiolule=petiolule, petiolule_mm=g.polyline_length(petiolule["centreline"]) if petiolule else None, age_d=None, expansion=None, flags=[])
            tree["leaflets"].append(lt)
            key = round(s_mm, 3) if typ != "P" else (pair, "P")
            atts.setdefault(key, dict(s_mm=s_mm, leaflets=[]))["leaflets"].append(bn)
        lf["attachments"] = sorted(atts.values(), key=lambda a: a["s_mm"])
        tree["leaves"].append(lf)
    # nodes: one per main-stem leaf at the rachis start projected on the stem centreline; side-axis leaves on the side stem
    main_pl = tree["axes"][0]["tube"]["centreline"] if tree["axes"] else None
    side_names = [n for n in names if n == "side_stem" or n.startswith("side_stem_")]       # v1: side_stem (= S1); v2: side_stem_S2, ...
    side_pos = {}
    for n in sorted(side_names, key=lambda x: (x != "side_stem", x)):
        sid = "S1" if n == "side_stem" else n[len("side_stem_"):]
        t = tube_entry(by[n], SEGS["side_stem"]); side_pos[sid] = t["centreline"][0]
        tree["axes"].append(dict(id=sid, kind="side", parent_axis="M", parent_node=None, tube=t, obj_name=n, flags=[]))
    node_idx = {a["id"]: 0 for a in tree["axes"]}
    for lf in sorted(tree["leaves"], key=lambda x: (x["axis"] != "M", x["axis"], int(re.search(r"L(\d+)$", x["id"]).group(1)))):
        ax = lf["axis"]; pl = next(a for a in tree["axes"] if a["id"] == ax)["tube"]["centreline"]
        start = lf["rachis"]["centreline"][0]; s = g.arclength_of_nearest(pl, start); pos = g.point_at_arclength(pl, s)
        node_idx[ax] += 1; k = node_idx[ax]
        nid = f"N{k:02d}" if ax == "M" else f"{ax}_N{k:02d}"
        rank = max(1, k - 1) if ax == "M" else k
        tree["nodes"].append(dict(id=nid, axis=ax, index=k, rank=rank, s_mm=s, position=pos, leaves=[lf["id"]], trusses=[], flags=[]))
        lf["node"] = nid; lf["rank"] = rank
    for a in tree["axes"]:
        if a["kind"] == "side" and main_pl is not None:
            s = g.arclength_of_nearest(main_pl, side_pos[a["id"]])
            kk = min(range(len(tree["nodes"])), key=lambda i: abs(tree["nodes"][i]["s_mm"] - s) if tree["nodes"][i]["axis"] == "M" else 1e9)
            a["parent_node"] = tree["nodes"][kk]["id"]
    # trusses
    tr = {}
    for n in names:
        mt = NAME["truss"].match(n)
        if mt:
            tr.setdefault(mt.group(1), {})[mt.group(2)] = by[n]
    for tid, parts in sorted(tr.items()):
        ped = parts.get("peduncle")
        if not ped:
            continue
        pt = tube_entry(ped, SEGS["peduncle"]); flowers = []
        ks = sorted(int(k[6:]) for k in parts if k.startswith("flower"))
        for k in ks:
            fo = parts[f"flower{k}"]; pe = parts.get(f"pedicel{k}")
            flowers.append(dict(id=f"{tid}_flower{k}", position=fo["V"][0], mesh=dict(vertices=fo["V"], faces=fo["F"]),
                                pedicel=tube_entry(pe, SEGS["pedicel"]) if pe else None))
        base = pt["centreline"][0]
        # attach to the nearest axis end / node
        best = None
        for a in tree["axes"]:
            pl = a["tube"]["centreline"]; d = float(np.linalg.norm(np.subtract(pl[-1], base)))
            if best is None or d < best[0]:
                best = (d, a["id"])
        tree["trusses"].append(dict(id=tid, node=None, axis=best[1], peduncle=pt, flowers=flowers, flower_count=len(flowers), stage="flowering", flags=[]))
    for n in names:
        mc = NAME["coty"].match(n)
        if mc:
            o = by[n]; tree["cotyledons"].append(dict(id=n, surface=dict(kind="grid", nu=BLADE_NU, nv=BLADE_NV, vertices=o["V"]) if len(o["V"]) == (BLADE_NU + 1) * (BLADE_NV + 1) else dict(kind="mesh", vertices=o["V"], faces=o["F"]), flags=[]))
    return validate(tree)
