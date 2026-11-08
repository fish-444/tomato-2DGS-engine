"""Micro-Tom synthetic individual generator — exact Python port of the chat artifact "Micro-Tom 합성 개체 생성기"
(https://claude.ai/artifact/6SwdtC9PWAp142tq4KtoWK; reference copy build/p1_synth_microtom_20261022_01/reference/,
sha256 b6f0cac5...). Schema ``microtom_synth_v1``.

The port is line for line: same parameter keys and defaults, same order of random draws, same per-organ RNG keys
(``rng_for(seed, key)``), ``hash_str`` / mulberry32 with JavaScript 32-bit integer semantics (``Math.imul``, ``>>>``,
``|0``) and Box-Muller ``gauss`` as written. The model is NOT changed here; model changes are future declared versions
(``microtom_synth_v2``). Geometry is built y-up in mm (as in the page); the OBJ export writes (x, -z, y) = z up,
origin at the pot top centre. Everything produced here is SYNTHETIC geometry, never a measurement.

usage: python -m tomato3d.synth.microtom --params p.json --out dir/ [--seed N] [--preset S01]
"""

from __future__ import annotations

import argparse
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SCHEMA = "microtom_synth_v1"
REFERENCE_SHA256 = "b6f0cac57355cc04f8b5341131766dce312f2dd3821f0c68fb2cf610e31bc37d"
TAG_PROVENANCE = {"m": "user_measured", "l": "literature", "a": "assumption"}

# ---------------------------------------------------------------- parameters (GROUPS order = page DEF order)
GROUPS = [
    ("구조", [("n_truss", 7, "m"), ("visible", 7, "a"), ("hypocotyl", 14, "a"), ("internode", 7, "m"), ("phyllo", 137.5, "a"), ("lean", 4, "a")]),
    ("복엽", [("leaf_len", 72, "a"), ("max_pairs", 3, "l"), ("elev_low", 72, "a"), ("elev_high", 42, "a"), ("droop", 0.35, "a"), ("inter_p", 0.25, "l")]),
    ("소엽", [("lf_len", 19, "m"), ("lw_ratio", 1.5, "m"), ("pair_angle", 62, "a"), ("cup", 0.25, "a"), ("bend", 0.3, "a")]),
    ("생식·분지", [("flowers", 6, "l"), ("side_leaves", 2, "l"), ("truss2", True, "l"), ("cotyledons", True, "a")]),
    ("개체 변이", [("cv", 0.12, "a"), ("seed", 20261022, "a")]),
]
DEF = {k: d for _, items in GROUPS for k, d, _ in items}
PROVENANCE = {k: TAG_PROVENANCE[t] for _, items in GROUPS for k, _, t in items}
PRESETS = {
    "S01": dict(n_truss=6, visible=4, leaf_len=58, lf_len=15, cv=0.10, side_leaves=0, seed=101, label="어린 개체"),
    "S02": dict(n_truss=7, visible=7, leaf_len=72, lf_len=19, cv=0.12, side_leaves=2, seed=202, label="첫 화방 (실측 평균)"),
    "S03": dict(n_truss=8, visible=8, leaf_len=84, lf_len=22, cv=0.14, side_leaves=3, droop=0.5, seed=303, label="큰 개체"),
    "S04": dict(n_truss=9, visible=9, leaf_len=66, lf_len=17, cv=0.16, side_leaves=1, elev_low=85, seed=404, label="문헌 상한 9"),
}


def params_for(preset: str | None = None, **over) -> dict:
    """Page semantics: P = Object.assign({}, DEF); preset keys that exist in DEF overwrite (key order = DEF order)."""
    P = dict(DEF)
    if preset:
        for k, v in PRESETS[preset].items():
            if k in DEF:
                P[k] = v
    for k, v in over.items():
        if k in DEF:
            P[k] = v
    return P


# ---------------------------------------------------------------- JavaScript number semantics
def _i32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def _to_int32(x) -> int:
    """ECMAScript ToInt32 for a finite number."""
    if isinstance(x, float):
        x = int(math.trunc(x))
    return _i32(x)


def _imul(a, b) -> int:
    return _i32(_to_int32(a) * _to_int32(b))


def _ushr(a, n: int) -> int:
    return _u32(_to_int32(a)) >> n


def js_str(x) -> str:
    """String(number) for the values used as RNG keys (integers print without a decimal point)."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float) and x.is_integer() and abs(x) < 1e21:
        return str(int(x))
    if isinstance(x, float):
        return repr(x)
    return str(x)


def hash_str(s: str) -> int:
    units = s.encode("utf-16-le"); codes = [units[i] | (units[i + 1] << 8) for i in range(0, len(units), 2)]
    n = len(codes)
    h1 = _to_int32(0xDEADBEEF) ^ n
    h2 = 0x41C6CE57 ^ n
    for c in codes:
        h1 = _imul(h1 ^ c, 2654435761)
        h2 = _imul(h2 ^ c, 1597334677)
    h1 = _imul(h1 ^ _ushr(h1, 16), 2246822507) ^ _imul(h2 ^ _ushr(h2, 13), 3266489909)
    h1 = _i32(h1)
    h2 = _imul(h2 ^ _ushr(h2, 16), 2246822507) ^ _imul(h1 ^ _ushr(h1, 13), 3266489909)
    h2 = _i32(h2)
    return _i32(_u32(h2) ^ _u32(h1))          # (h2>>>0)^(h1>>>0): the ^ operator returns a signed int32


def rng_for(seed, key):
    a = [_u32(hash_str(js_str(seed) + ":" + js_str(key)))]

    def r():
        x = _to_int32(a[0])
        x = _i32(x + 0x6D2B79F5)
        a[0] = x
        t = _imul(x ^ _ushr(x, 15), 1 | x)
        t = _to_int32((t + _imul(t ^ _ushr(t, 7), 61 | t))) ^ t   # (t + imul(...)) ^ t
        t = _i32(t)
        return _u32(t ^ _ushr(t, 14)) / 4294967296
    return r


def gauss(r) -> float:
    u = max(1e-9, r()); v = r()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


def clamp(x, a, b):
    return max(a, min(b, x))


def smooth(a, b, x):
    t = clamp((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def js_fixed(x: float, d: int) -> str:
    """Number.prototype.toFixed: exact decimal value, ties to the larger n; -0 prints as 0."""
    if x == 0:
        x = 0.0
    neg = x < 0
    q = Decimal(-x if neg else x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:.{d}f}"
    return ("-" + s) if neg else s


D2R = math.pi / 180


# ---------------------------------------------------------------- vector helpers (y up while building)
def add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def norm(a):
    ln = math.hypot(a[0], a[1], a[2]) or 1
    return [a[0] / ln, a[1] / ln, a[2] / ln]


def rot_around(p, axis, ang):
    c, s, k = math.cos(ang), math.sin(ang), axis
    return add(add(mul(p, c), mul(cross(k, p), s)), mul(k, dot(k, p) * (1 - c)))


UP, DOWN = [0, 1, 0], [0, -1, 0]


def tri_area(a, b, c):
    x = cross(sub(b, a), sub(c, a)); return 0.5 * math.hypot(x[0], x[1], x[2])


# ---------------------------------------------------------------- generator
def generate(P: dict) -> dict:
    objs = []; seed = P["seed"]; cv = P["cv"]

    def obj(name, cat, leaf=""):
        o = dict(n=name, c=cat, leaf=leaf or "", v=[], f=[], info={}); objs.append(o); return o

    def jit(key, sd):
        return 1 + cv * sd * gauss(rng_for(seed, key))

    def tube(o, pts, radii, segs=None):
        segs = segs or 6; base = len(o["v"]); prevN = None
        for i in range(len(pts)):
            t = norm(sub(pts[min(i + 1, len(pts) - 1)], pts[max(i - 1, 0)]))
            n = norm(sub(prevN, mul(t, dot(prevN, t)))) if prevN else norm(cross(t, UP) if abs(t[1]) < 0.9 else cross(t, [1, 0, 0]))
            prevN = n; b = cross(t, n)
            for j in range(segs):
                a = j / segs * 2 * math.pi
                o["v"].append(add(pts[i], add(mul(n, math.cos(a) * radii[i]), mul(b, math.sin(a) * radii[i]))))
        for i in range(len(pts) - 1):
            for j in range(segs):
                a0 = base + i * segs + j; a1 = base + i * segs + (j + 1) % segs; b0 = a0 + segs; b1 = a1 + segs
                o["f"].append([a0, a1, b1]); o["f"].append([a0, b1, b0])

    def blade(o, base, d, nrm, L, W, cup, bend):
        b = norm(cross(nrm, d)); NU, NV = 12, 7; start = len(o["v"])
        for i in range(NU + 1):
            u = i / NU
            hw = W / 2 * math.pow(math.sin(math.pi * math.pow(u, 0.82)), 0.85) * (1 - 0.18 * u) / 0.93
            for j in range(NV + 1):
                s = j / NV * 2 - 1
                p = add(base, mul(d, u * L))
                p = add(p, mul(b, s * hw))
                p = add(p, mul(nrm, cup * W * 0.28 * s * s - bend * L * 0.22 * u * u))
                o["v"].append(p)
        area = 0
        for i in range(NU):
            for j in range(NV):
                a0 = start + i * (NV + 1) + j; a1 = a0 + 1; b0 = a0 + NV + 1; b1 = b0 + 1
                o["f"].append([a0, b0, b1]); o["f"].append([a0, b1, a1])
                area += tri_area(o["v"][a0], o["v"][b0], o["v"][b1]) + tri_area(o["v"][a0], o["v"][b1], o["v"][a1])
        return area

    # ---- one compound leaf
    def compound_leaf(id_, node, az, elev, scale, nPairs, lfMax):
        r = rng_for(seed, id_)
        L = P["leaf_len"] * scale * jit(id_ + ":len", 1)
        h = [math.cos(az), 0, math.sin(az)]
        d0 = norm(add(mul(h, math.sin(elev)), mul(UP, math.cos(elev))))
        droop = P["droop"] * (0.7 + 0.6 * r())

        def at(s): return add(add(node, mul(d0, s * L)), mul(DOWN, droop * s * s * L * 0.45))
        def tan(s): return norm(add(mul(d0, L), mul(DOWN, droop * 2 * s * L * 0.45)))
        ra = obj("rachis_" + id_, "rachis", id_); pts, rad = [], []
        for i in range(15):
            s = i / 14; pts.append(at(s)); rad.append((0.75 - 0.4 * s) * max(0.6, scale))
        tube(ra, pts, rad, 6); ra["info"] = dict(length_mm=L)
        acc = dict(area=0.0, count=0)

        def leaflet(name, s, side, posF, kind):
            rr = rng_for(seed, id_ + "__" + name)
            p = at(s); t = tan(s)
            sd = cross(t, UP); sd = h if math.hypot(sd[0], sd[1], sd[2]) < 1e-3 else norm(sd)
            alpha = (P["pair_angle"] + 8 * cv / 0.12 * gauss(rr)) * D2R
            d = t if side == 0 else norm(add(mul(t, math.cos(alpha)), mul(sd, side * math.sin(alpha))))
            d = norm(add(d, mul(DOWN, 0.12 * P["bend"])))
            n0 = norm(sub(UP, mul(d, dot(UP, d))))
            n0 = rot_around(n0, d, (side * 10 + 12 * gauss(rr) * cv / 0.12) * D2R)
            Lf = lfMax * posF * math.sqrt(scale) * jit(id_ + "__" + name + ":lf", 1)
            if kind == "I":
                Lf *= 0.32
            ratio = clamp(P["lw_ratio"] + 0.25 * gauss(rr) * (1 if cv > 0 else 0), 1.0, 2.2)
            Wf = Lf / ratio
            pl = max(1.2, 0.12 * Lf)
            po = obj("petiolule_" + id_ + "__" + name, "petiolule", id_)
            tube(po, [p, add(p, mul(d, pl))], [0.35, 0.3], 5)
            cat = "blade_T" if kind == "T" else ("blade_I" if kind == "I" else "blade_P")
            bo = obj(id_ + "__" + name, cat, id_)
            a = blade(bo, add(p, mul(d, pl)), d, n0, Lf, Wf, P["cup"], P["bend"])
            bo["info"] = dict(length_mm=Lf, width_mm=Wf, area_mm2=a, kind=kind)
            acc["area"] += a; acc["count"] += 1
        p0 = 0.32
        for j in range(1, nPairs + 1):
            s = p0 + (0.92 - p0) * (j - 1) / max(1, nPairs - 0.4)
            posF = 0.72 + 0.28 * (j / nPairs)
            leaflet("P" + str(j) + "L", s, 1, posF, "P"); leaflet("P" + str(j) + "R", s + 0.015, -1, posF, "P")
            if j < nPairs and rng_for(seed, id_ + ":I" + str(j))() < P["inter_p"]:
                sI = s + 0.5 * (0.92 - p0) / max(1, nPairs - 0.4)
                leaflet("I" + str(j), sI, (1 if j % 2 else -1), 0.5, "I")
        leaflet("T", 1.0, 0, 1.08, "T")
        return dict(area=acc["area"], n=acc["count"], length=L)

    # ---- main stem
    nT = P["n_truss"]; V = min(P["visible"], nT)
    leanAz = rng_for(seed, "lean")() * 2 * math.pi; lean = P["lean"] * D2R
    leanDir = [math.sin(lean) * math.cos(leanAz), 0, math.sin(lean) * math.sin(leanAz)]
    axis = norm(add(UP, leanDir))
    y = P["hypocotyl"] * jit("hyp", 0.6)
    nodes = []; cur = mul(axis, y)
    stemPts = [[0, 0, 0], list(cur)]
    for k in range(1, V + 1):
        il = P["internode"] * jit("int" + str(k), 1) * (0.35 if k == 2 else 1) * (0.4 if k == 1 else 1)
        if k == V and V < nT:
            il *= 0.6
        wob = rng_for(seed, "wob" + str(k))
        gx = 0.06 * gauss(wob); gz = 0.06 * gauss(wob)
        cur = add(cur, mul(norm(add(axis, [gx, 0, gz])), il))
        nodes.append(list(cur)); stemPts.append(list(cur))
    topNode = list(cur)
    if V >= nT:
        cur = add(cur, mul(axis, P["internode"] * 0.6)); stemPts.append(list(cur))
    stem = obj("stem", "stem", ""); rads = [1.9 - 0.9 * i / max(1, len(stemPts) - 1) for i in range(len(stemPts))]
    tube(stem, stemPts, rads, 8); stem["info"] = dict(nodes=V)

    # cotyledons
    if P["cotyledons"]:
        cn = mul(axis, y); caz = rng_for(seed, "coty")() * math.pi
        for c in range(2):
            az = caz + c * math.pi; dd = norm([math.cos(az), 0.25, math.sin(az)])
            co = obj("cotyledon_C" + str(c + 1), "cotyledon", ""); nn = norm(sub(UP, mul(dd, dot(UP, dd))))
            a = blade(co, add(cn, mul(dd, 1.5)), dd, nn, 15 * jit("coty" + str(c), 0.5), 4.2, 0.1, 0.2)
            co["info"] = dict(length_mm=15, area_mm2=a)

    # leaves on the main stem
    az0 = rng_for(seed, "az0")() * 2 * math.pi; leaves = []; totA = 0; totLf = 0
    for k in range(1, V + 1):
        az = az0 if k == 1 else (az0 + math.pi * 0.92 if k == 2 else az0 + math.pi * 0.92 + (k - 2) * P["phyllo"] * D2R)
        az += 8 * D2R * gauss(rng_for(seed, "az" + str(k))) * cv / 0.12
        rel = V - k
        tab = [0.55, 0.8, 0.95] if V >= nT else [0.3, 0.6, 0.85]
        expand = (tab[rel] if rel < len(tab) else None) or 1
        prof = (0.45 + 0.55 * smooth(0, 0.6 * nT, k - 1)) * (1 - 0.15 * smooth(0.7 * nT, nT, k))
        scale = prof * expand
        fr = (k - 1) / max(1, nT - 1)
        elev = (P["elev_low"] + (P["elev_high"] - P["elev_low"]) * fr - 18 * ((2 - rel) / 2 if rel < 2 else 0) * (1 if V < nT else 0.5)) * D2R
        elev += 5 * D2R * gauss(rng_for(seed, "el" + str(k))) * cv / 0.12
        pairs = 1 if k <= 2 else (min(2, P["max_pairs"]) if k == 3 else P["max_pairs"])
        id_ = "L" + str(k).rjust(2, "0")
        res = compound_leaf(id_, add(nodes[k - 1], mul([math.cos(az), 0, math.sin(az)], 1.4)), az, elev, scale, pairs, P["lf_len"])
        leaves.append(dict(id=id_, rank=(1 if k <= 2 else k - 1), scale=scale, leaflets=res["n"], area=res["area"], length=res["length"]))
        totA += res["area"]; totLf += res["n"]

    # trusses
    fl_count = [0]

    def truss(id_, base, dir_, nFl):
        r = rng_for(seed, id_); pd = obj(id_ + "_peduncle", "truss", "")
        e1 = add(base, mul(dir_, 10)); g1 = 0.3 * gauss(r); g2 = 0.3 * gauss(r)
        e2 = add(e1, mul(norm(add(dir_, [g1, -0.2, g2])), 8))
        tube(pd, [base, e1, e2], [0.7, 0.55, 0.45], 5)
        for i in range(nFl):
            s = i / max(1, nFl - 1)
            p = add(add(e1, mul(sub(e2, e1), s)), [0, 0, 0])
            fa = r() * 2 * math.pi; fd = norm([math.cos(fa), 0.6 + 0.4 * r(), math.sin(fa)])
            pe = obj(id_ + "_pedicel" + str(i + 1), "truss", ""); fc = add(p, mul(fd, 4))
            tube(pe, [p, fc], [0.3, 0.25], 4)
            fl = obj(id_ + "_flower" + str(i + 1), "flower", "")
            nrm = norm(fd); bx = norm(cross(nrm, UP if abs(nrm[1]) < 0.9 else [1, 0, 0])); by = cross(nrm, bx)
            ci = len(fl["v"]); fl["v"].append(fc)
            for pt in range(5):
                a = pt / 5 * 2 * math.pi + r()
                dirp = add(mul(bx, math.cos(a)), mul(by, math.sin(a)))
                tip = add(add(fc, mul(dirp, 4.2)), mul(nrm, -0.8))
                l_ = add(fc, mul(add(mul(bx, math.cos(a - 0.35)), mul(by, math.sin(a - 0.35))), 1.6))
                rr = add(fc, mul(add(mul(bx, math.cos(a + 0.35)), mul(by, math.sin(a + 0.35))), 1.6))
                b0 = len(fl["v"]); fl["v"].extend([l_, tip, rr]); fl["f"].append([ci, b0, b0 + 1]); fl["f"].append([ci, b0 + 1, b0 + 2])
            fl_count[0] += 1

    def flower_count(key):
        r = rng_for(seed, key); return clamp(js_round(P["flowers"] + 1.2 * gauss(r)), 4, 8)

    if V >= nT:
        truss("T1", topNode, norm(add(axis, [0.5, 0, 0.3])), flower_count("T1"))
        if P["truss2"]:
            truss("T2", cur, norm(add(axis, [-0.4, 0.2, -0.35])), flower_count("T2"))
        # side shoot from the uppermost leaf axil
        if P["side_leaves"] > 0 and len(nodes) >= 2:
            sn = nodes[len(nodes) - 2]; saz = az0 + math.pi * 0.92 + (V - 3) * P["phyllo"] * D2R + 0.35
            sdir = norm(add(mul(axis, 0.85), mul([math.cos(saz), 0, math.sin(saz)], 0.55)))
            spts = [sn]; sc = list(sn)
            for s in range(1, P["side_leaves"] + 1):
                sc = add(sc, mul(sdir, P["internode"] * 0.75 * jit("sint" + str(s), 1))); spts.append(list(sc))
            ss = obj("side_stem", "stem", ""); tube(ss, spts, [1.1 - 0.4 * i / len(spts) for i in range(len(spts))], 6)
            for s in range(1, P["side_leaves"] + 1):
                a2 = saz + s * P["phyllo"] * D2R; sid = "S1_L" + str(s).rjust(2, "0")
                rs = compound_leaf(sid, spts[s], a2, (P["elev_high"] + 10) * D2R, 0.62 * (0.7 if s == P["side_leaves"] else 1), min(2, P["max_pairs"]), P["lf_len"])
                leaves.append(dict(id=sid, rank="side", scale=0.62, leaflets=rs["n"], area=rs["area"], length=rs["length"]))
                totA += rs["area"]; totLf += rs["n"]
            if P["side_leaves"] == 3:
                truss("T3", sc, sdir, flower_count("T3"))
    # leaves that droop below the pot top rest on it
    for o in objs:
        if o["c"] == "stem":
            continue
        for p in o["v"]:
            if p[1] < 0.4:
                p[1] = 0.4
    mn = [1e9, 1e9, 1e9]; mx = [-1e9, -1e9, -1e9]
    for o in objs:
        for p in o["v"]:
            for i in range(3):
                mn[i] = min(mn[i], p[i]); mx[i] = max(mx[i], p[i])
    return dict(objs=objs, leaves=leaves, stats=dict(leaves=len(leaves), leaflets=totLf, area=totA, flowers=fl_count[0],
                                                     height=mx[1], width=max(mx[0] - mn[0], mx[2] - mn[2]), truss=V >= nT))


# ---------------------------------------------------------------- exports (same text as the page buttons)
def _js_json(x) -> str:
    """JSON.stringify for P (numbers with JS formatting)."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if x is None:
        return "null"
    if isinstance(x, (int, float)):
        return js_str(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + _js_json(v) for k, v in x.items()) + "}"
    raise TypeError(type(x))


def obj_text(P: dict, res: dict) -> str:
    lines = ["# Micro-Tom synthetic individual · microtom_synth_v1 · units mm · z up · seed " + js_str(P["seed"]), "# params " + _js_json(P)]
    off = 1
    for o in res["objs"]:
        if not o["f"]:
            continue
        lines += ["o " + o["n"], "usemtl " + o["c"]]
        lines += ["v " + js_fixed(p[0], 3) + " " + js_fixed(-p[2], 3) + " " + js_fixed(p[1], 3) for p in o["v"]]
        lines += ["f %d %d %d" % (f[0] + off, f[1] + off, f[2] + off) for f in o["f"]]
        off += len(o["v"])
    return "\n".join(lines)


def organs_csv(res: dict) -> str:
    rows = ["object,category,leaf,length_mm,width_mm,area_mm2"]
    for o in res["objs"]:
        i = o["info"] or {}
        if not (i.get("length_mm") or i.get("area_mm2")):
            continue
        f = lambda key, d: js_fixed(i[key], d) if i.get(key) else ""  # noqa: E731
        rows.append(",".join([o["n"], o["c"], o["leaf"], f("length_mm", 2), f("width_mm", 2), f("area_mm2", 1)]))
    return "\n".join(rows)


def param_dict(P: dict, preset: str | None = None, **extra) -> dict:
    o = dict(schema=SCHEMA, units="mm", frame="pot top centre, z up", generator="Micro-Tom 합성 개체 생성기 (artifact)",
             params={k: P[k] for k in DEF}, provenance=dict(PROVENANCE), preset=preset)
    o.update(extra)
    return o


def write_individual(P: dict, out: Path, preset: str | None = None, **extra) -> dict:
    """OBJ + organs.csv + params.json; every output carries synthetic: true."""
    out.mkdir(parents=True, exist_ok=True); res = generate(P)
    (out / "plant.obj").write_text("# synthetic: true\n" + obj_text(P, res) + "\n", encoding="utf-8")
    (out / "organs.csv").write_text(organs_csv(res) + "\n", encoding="utf-8")
    pj = param_dict(P, preset, synthetic=True, port="tomato3d.synth.microtom", reference_sha256=REFERENCE_SHA256, **extra)
    (out / "params.json").write_text(json.dumps(pj, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def load_params(path: Path) -> tuple[dict, str | None]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    src = d.get("params", d)
    P = params_for(None, **{k: v for k, v in src.items() if k in DEF})
    return P, d.get("preset")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--params"); ap.add_argument("--preset", choices=sorted(PRESETS)); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int); a = ap.parse_args(argv)
    if a.params:
        P, preset = load_params(Path(a.params))
    else:
        P, preset = params_for(a.preset), a.preset
    if a.seed is not None:
        P["seed"] = a.seed
    res = write_individual(P, Path(a.out), preset)
    print(json.dumps(dict(out=a.out, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["stats"].items()})))


if __name__ == "__main__":
    main()
