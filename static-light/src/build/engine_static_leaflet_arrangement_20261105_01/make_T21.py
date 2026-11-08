"""Part C, C-1 (CONTRACT amendment 2): T21 = engine v2.1 leaflet laws (C4L spread + C4B bend) re-implemented verbatim and applied to the unchanged R3-1b crosswalk
of engine_disc_vs_tri_20261101_01 (same compound ids p1/C000k, same rachis polylines, same blade areas). Local, light.
Writes T21_triangles.csv / T21.obj / T21.ply / T21_meta.json and port_check.json (checks i-iii)."""
import csv, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; B = HERE.parents[0]; DVT = B / 'engine_disc_vs_tri_20261101_01'; W = B / 'engine_v2_calib_20261104_01/work'
MODEL = W / 'Model_v2_1.rgg'; UVF = W / 'leaflet_outline_uv.csv'; M00 = B / 'engine_v2_calib_20261104_01/kout/ev-v21c-k0-20261104/out/V21Bs5.85_m00/geometry_day71.npz'
sys.path.insert(0, str(DVT)); import make_T as OLD                                # old T helpers (interval_point, az_el, diamond, write format)
TAGS = ['disc_vs_triangle', 'relative_light', 'conditional_on_unvalidated_scale', 'leaflet_arrangement_from_reconstruction', 'R3-1b_unaccepted', 'engine_v21_leaflet_laws_ported']
TXT = MODEL.read_text(encoding='utf-8')
M64 = (1 << 48) - 1


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jhash(s):
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= 1 << 31 else h


class JRandom:
    """java.util.Random, exactly (seed scramble, next(bits), nextDouble, nextInt(bound), nextGaussian)."""
    def __init__(self, seed):
        self.s = (seed ^ 0x5DEECE66D) & M64; self.g = None
    def next(self, bits):
        self.s = (self.s * 0x5DEECE66D + 0xB) & M64; r = self.s >> (48 - bits)
        return r - (1 << bits) if bits == 32 and r >= 1 << 31 else r
    def nextDouble(self):
        return ((self.next(26) << 27) + self.next(27)) * (1.0 / (1 << 53))
    def nextInt(self, bound):
        if bound & (bound - 1) == 0:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31); val = bits % bound
            if bits - val + (bound - 1) < (1 << 31):
                return val
    def nextGaussian(self):
        if self.g is not None:
            g, self.g = self.g, None; return g
        while True:
            v1 = 2 * self.nextDouble() - 1; v2 = 2 * self.nextDouble() - 1; s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        mul = math.sqrt(-2 * math.log(s) / s)   # StrictMath.log / sqrt: same IEEE results for these inputs in practice
        self.g = v2 * mul; return v1 * mul


def arr(name, depth):
    if depth == 1:                                                               # 1-D arrays may share a declaration line (outlineLengths, outlineWidths)
        m = re.search(rf'\b{name}=new double\[\]\{{([^}}]*)\}}', TXT); assert m, name
        return [float(x) for x in m.group(1).split(',')]
    m = re.search(rf'double{"\\[\\]" * depth} {name}=(new double[^;]*);', TXT); assert m, name
    s = re.sub(r'new double(\[\])+', '', m.group(1)).replace('{', '[').replace('}', ']')
    return json.loads(s)


RP_SEED = int(re.search(r'long rpSeed=(\d+)L;', TXT).group(1)); SEED = int(re.search(r'int n=\d+,seed=(\d+)', TXT).group(1))
qU = arr('v2qU', 1); qS = arr('v2qS', 3); qOff = arr('v2qOff', 3); qTilt = arr('v2qTilt', 3); PHI = arr('v2phi', 3); pROLL = arr('v2pRoll', 2); SHARE = arr('v2Share', 1)
CHAIN_MED = float(re.search(r'double v2ChainMedian=([0-9.eE-]+);', TXT).group(1)); qCu = arr('v2qCu', 1)
KL = arr('v2qLEAFLET_KAPPA_L', 2); KW = arr('v2qLEAFLET_KAPPA_W', 2); KC = arr('v2qLEAFLET_TWIST_C', 2); LREF = arr('v2Lref', 1)
OUT_LEN = arr('outlineLengths', 1); OUT_WID = arr('outlineWidths', 1)
OUT_SIZES = [int(x) for x in re.search(r'int\[\] outlineSizes=new int\[\]\{([^}]*)\}', TXT).group(1).split(',')]
PETIOLE_M = float(re.search(r'PETIOLE_M = ([0-9.]+);', TXT).group(1)); RACHIS_M = float(re.search(r'RACHIS_M = ([0-9.]+);', TXT).group(1)); RACHIS_W = float(re.search(r'RACHIS_WIDTH_M = ([0-9.]+);', TXT).group(1))
LC_TABLE = {}   # v2LeafletCount, main axis (controller 0)
for rk, vals, sh in re.findall(r'if\(rank==(\d+)\)return v2Cat\(id,"LEAFLET_COUNT",new int\[\]\{([0-9,]+)\},new double\[\]\{([0-9.,eE-]+)\}\);', TXT):
    LC_TABLE[int(rk)] = ([int(v) for v in vals.split(',')], [float(v) for v in sh.split(',')])
_d = re.search(r'return v2Cat\(id,"LEAFLET_COUNT",new int\[\]\{([0-9,]+)\},new double\[\]\{([0-9.,eE-]+)\}\);\n\}', TXT)
LC_DEFAULT = ([int(v) for v in _d.group(1).split(',')], [float(v) for v in _d.group(2).split(',')])
for law in ('if(o.kind!=9 || o.imported || o.emerged<0 || o.area<=0 || o.v2Comp<0)continue;', 'double off=o.v2Off*Lc/v2ChainMedian;', 'o.v2vx=cs*v0x+sn*nx;',
            'double kl=o.v2Kl*1000*f,kw=o.v2Kw*1000*f,kc=o.v2Kc*1000*f;', 'o.len=Math.sqrt(o.area)*outlineLengths[oi];'):
    assert law in TXT, law
UV = [np.zeros(n) for n in OUT_SIZES]
with open(UVF, encoding='utf-8') as fh:
    next(fh)
    for line in fh:
        a, b, c = line.strip().split(','); UV[int(a)][int(b)] = float(c)


def rpU(i, key):
    r = JRandom(RP_SEED + jhash(f'{i}:{key}')); r.nextDouble(); return r.nextDouble()


def gauss(i, key):
    r = JRandom(RP_SEED + jhash(f'{i}:{key}')); r.nextDouble(); return r.nextGaussian()


def quant(i, key, q, n, grid):
    u = rpU(i, key)
    if u <= grid[0]:
        return q[0]
    if u >= grid[n - 1]:
        return q[n - 1]
    k = 1
    while k < n - 1 and u > grid[k]:
        k += 1
    f = (u - grid[k - 1]) / (grid[k] - grid[k - 1]); return q[k - 1] + f * (q[k] - q[k - 1])


def cat(i, key, vals, shares):
    u = rpU(i, key); acc = 0.0; tot = sum(shares)
    for v, s in zip(vals, shares):
        acc += s / tot
        if u < acc:
            return v
    return vals[-1]


def leaflet_count(cid, rank):
    vals, sh = LC_TABLE.get(rank, LC_DEFAULT); return cat(cid, 'LEAFLET_COUNT', vals, sh)


def outline_index(oid):
    return JRandom(SEED + jhash(oid)).nextInt(len(OUT_SIZES))


def draws(lid, pair, side):
    pos = 3 if side == 0 else min(pair, 3) - 1; sx = 0 if side < 0 else (1 if side > 0 else 0); d = dict(pos=pos)
    if side == 0:
        d.update(S=1.0, phi=0.0, off=0.0)
    else:
        d['S'] = max(0.0, min(1.0, quant(lid, 'LEAFLET_S_OVER_L', qS[pos][sx], 6, qU)))
        ph = PHI[pos][sx]; d['phi'] = max(ph[2], min(ph[3], ph[0] + ph[1] * gauss(lid, 'LEAFLET_SPREAD_PHI_DEG')))
        d['off'] = quant(lid, 'LEAFLET_PETIOLULE_MM', qOff[pos][sx], 6, qU) / 1000.0
    d['tilt'] = quant(lid, 'LEAFLET_TILT_DEG', qTilt[pos][sx], 7, qU)
    d['roll'] = 1.0 if rpU(lid, 'LEAFLET_ROLL_SIGN') < pROLL[pos][sx] else -1.0
    d['kl'] = quant(lid, 'LEAFLET_KAPPA_L', KL[pos], len(KL[pos]), qCu); d['kw'] = quant(lid, 'LEAFLET_KAPPA_W', KW[pos], len(KW[pos]), qCu); d['kc'] = quant(lid, 'LEAFLET_TWIST_C', KC[pos], len(KC[pos]), qCu)
    return d


def bend(uv, base, scale, u, v, dr, length):
    """v2BendInto (metres): quadric bend along the leaflet normal, area conserved by the lam iteration."""
    n = np.cross(u, v); n /= np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    f = LREF[dr['pos']] / length if length > 0 else 1.0; kl, kw, kc = dr['kl'] * 1000 * f, dr['kw'] * 1000 * f, dr['kc'] * 1000 * f
    T = uv.reshape(-1, 3, 2); a2 = 0.5 * np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
    cen = (T.mean(1) * a2[:, None]).sum(0) / a2.sum(); uc, vc = cen; target = a2.sum() * scale * scale
    a, b = uv[0::2], uv[1::2]; lam = 1.0
    for _ in range(60):
        s0, d0 = -lam * scale * uc, -lam * scale * vc; wb = 0.5 * kl * s0 * s0 + 0.5 * kw * d0 * d0 + kc * s0 * d0
        s, d = lam * scale * (a - uc), lam * scale * (b - vc); w = 0.5 * kl * s * s + 0.5 * kw * d * d + kc * s * d - wb
        P = base + lam * scale * (a[:, None] * u + b[:, None] * v) + w[:, None] * n
        Tr = P.reshape(-1, 3, 3); A = 0.5 * np.linalg.norm(np.cross(Tr[:, 1] - Tr[:, 0], Tr[:, 2] - Tr[:, 0]), axis=1).sum()
        if A <= 0 or abs(A / target - 1) < 1e-13:
            break
        lam *= math.sqrt(target / A)
    return P


def leaf_T21(P_mm, rank, area_mm2, cid, tag, flat=False):
    P = P_mm / 1000.0; cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]); Lp = cum[-1]
    count = leaflet_count(cid, rank); m = count // 2; omit = lambda i, side: count == 6 and i == m and side < 0
    s_pet = Lp * PETIOLE_M / (PETIOLE_M + RACHIS_M); step = (Lp - s_pet) / m; marks = [0.0, s_pet] + [s_pet + i * step for i in range(1, m + 1)]
    pts = [OLD.interval_point(P, cum, s) for s in marks]; seg = [(pts[k], pts[k + 1]) for k in range(m + 1)]; segL = [np.linalg.norm(b - a) for a, b in seg]; Lc = sum(segL)
    out = []
    for k, (a, b) in enumerate(seg):                                             # petiole / rachis: slender diamonds (unchanged from the old T)
        u = (b - a) * 1000
        if np.linalg.norm(u) <= 0:
            continue
        az, _ = OLD.az_el(u); wd = 1000 * RACHIS_W; vv = np.array([-wd * math.sin(az) / 2, wd * math.cos(az) / 2, 0.0])
        for t in OLD.diamond(a * 1000, u, vv):
            out.append((f'{tag}/{"P" if k == 0 else f"R{k}"}', 'Petiole' if k == 0 else 'RachisSegment', 0, 0, t))
    total = SHARE[3] + sum((1 if omit(i, -1) else 2) * SHARE[min(i, 3) - 1] for i in range(1, m + 1))
    leaflets = [(i, s) for i in range(1, m + 1) for s in (-1, 1) if not omit(i, s)] + [(m + 1, 0)]
    info = []
    for pair, side in leaflets:
        nm = 'T' if side == 0 else f'P{pair}{"L" if side < 0 else "R"}'; lid = f'{cid}/{nm}'; dr = draws(lid, pair, side)
        if flat:
            dr.update(kl=0.0, kw=0.0, kc=0.0)                                    # T21_flat: C4L placement, no C4B bend
        share = SHARE[dr['pos'] if side != 0 else 3] / total; A = area_mm2 * 1e-6 * share; oi = outline_index(lid); length = math.sqrt(A) * OUT_LEN[oi]
        target = dr['S'] * Lc; acc = 0.0; k = 0
        for k in range(len(seg) - 1):
            if acc + segL[k] >= target:
                break
            acc += segL[k]
        else:
            k = len(seg) - 1
        if k == len(seg) - 1 and acc + segL[k] < target - 1e-15:
            pass
        qa, qb = seg[k]; fr = max(0.0, min(1.0, (target - acc) / segL[k])) if segL[k] > 0 else 1.0
        base = qa + fr * (qb - qa); pa, pe = OLD.az_el(qb - qa)
        t = np.array([math.cos(pe) * math.cos(pa), math.cos(pe) * math.sin(pa), math.sin(pe)]); h = np.array([-math.sin(pa), math.cos(pa), 0.0]); nn = np.array([-math.sin(pe) * math.cos(pa), -math.sin(pe) * math.sin(pa), math.cos(pe)])
        ph = math.radians(dr['phi']); u = t.copy() if side == 0 else math.cos(ph) * t + side * math.sin(ph) * h
        base = base + side * (dr['off'] * Lc / CHAIN_MED) * h
        v0 = np.cross(nn, u); ang = math.radians(dr['roll'] * dr['tilt']); v = math.cos(ang) * v0 + math.sin(ang) * nn
        el = math.asin(max(-1.0, min(1.0, u[2]))); az = math.atan2(u[1], u[0]); u = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])   # o.az / o.elev round trip
        Pv = bend(UV[oi], base, math.sqrt(A), u, v, dr, length) * 1000.0
        for tri in Pv.reshape(-1, 3, 3):
            out.append((f'{tag}/{nm}', 'Leaflet', pair, side, tri))
        info.append(dict(leaflet=lid, outline=oi, share=share, area_mm2=A * 1e6, **{k2: v2 for k2, v2 in dr.items()}))
    return out, count, info


def main():
    meta_old = json.load(open(DVT / 'T_meta.json', encoding='utf-8')); tree = json.load(open(DVT / 'crosswalk/plant_tree_v1.json', encoding='utf-8'))
    lines = {l['leaf_uid']: l for l in meta_old['lines']}; tris = []; rows_info = []; checks = {}
    for lf in tree['leaves']:
        uid = lf['attrs']['leaf_uid']; L = lines[uid]; P = np.asarray(lf['rachis']['polyline'], float)
        t, count, info = leaf_T21(P, L['rank'], L['area_b_mm2'], L['engine_organ_id'], f'R3-1b/{uid}')
        for organ, cls, pair, side, tri in t:
            tris.append(dict(organ=organ, cls=cls, leaf=L['leaf_R8pc'], line=uid, pair=pair, side=side, tri=tri))
        rows_info.append(dict(leaf_uid=uid, compound=L['engine_organ_id'], rank=L['rank'], leaflet_count=count, area_b_mm2=L['area_b_mm2'], leaflets=info))
    files = OLD.write_files.__wrapped__(tris, 'T21', True) if hasattr(OLD.write_files, '__wrapped__') else None
    # write with the old writer but into this folder
    OLD.HERE = HERE; OLD.TAGS = TAGS; files = OLD.write_files(tris, 'T21', True)
    V = np.array([r['tri'] for r in tris]); ar = 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1); isl = np.array([r['cls'] == 'Leaflet' for r in tris])
    # check (i): area per line
    ci = []
    for ri in rows_info:
        a = ar[isl & np.array([r['line'] == ri['leaf_uid'] for r in tris])].sum(); ci.append(abs(a / ri['area_b_mm2'] - 1))
    checks['i_area_max_rel'] = float(max(ci)); checks['i_pass'] = bool(max(ci) < 1e-6)
    # check (ii): leaflet counts / outline triangle counts vs the engine's day-71 m00 plant (same ids)
    z = np.load(M00); own = z['owner']; cls = z['cls']; comp = sorted({o.rsplit('/', 1)[0] for o, c in zip(own, cls) if c == 'Leaflet' and o.count('/') == 2})
    rank_of = lambda cid: int(cid.split('/C')[1]); cnt_ok = tri_ok = n = 0; mism = []
    for cid in comp:
        lfs = sorted({o for o, c in zip(own, cls) if c == 'Leaflet' and o.startswith(cid + '/')}); n += 1
        if len(lfs) == leaflet_count(cid, rank_of(cid)):
            cnt_ok += 1
        else:
            mism.append(dict(compound=cid, engine=len(lfs), port=leaflet_count(cid, rank_of(cid))))
        for lid in lfs:
            k = int((own == lid).sum()); tri_ok += int(k == OUT_SIZES[outline_index(lid)] // 6)
    nl = sum(len({o for o, c in zip(own, cls) if c == 'Leaflet' and o.startswith(cid + '/')}) for cid in comp)
    checks['ii_compounds'] = n; checks['ii_leaflet_count_match'] = cnt_ok; checks['ii_leaflets'] = nl; checks['ii_outline_triangle_count_match'] = tri_ok; checks['ii_mismatch'] = mism
    checks['ii_pass'] = bool(n >= 10 and cnt_ok == n and tri_ok == nl)
    json.dump(dict(labels=TAGS, engine_model=str(MODEL.relative_to(B.parent)), engine_sha256=sha(MODEL), outline_csv_sha256=sha(UVF), rpSeed=RP_SEED, seed=SEED, crosswalk_from=str(DVT.relative_to(B.parent)),
                   lines=rows_info, triangles=len(tris), leaflet_triangles=int(isl.sum()), files={f: sha(HERE / f) for f in files}, checks=checks), open(HERE / 'T21_meta.json', 'w'), indent=1)
    tf = []
    for lf in tree['leaves']:
        uid = lf['attrs']['leaf_uid']; L = lines[uid]; t, _, _ = leaf_T21(np.asarray(lf['rachis']['polyline'], float), L['rank'], L['area_b_mm2'], L['engine_organ_id'], f'R3-1b/{uid}', flat=True)
        tf += [dict(organ=o, cls=c, leaf=L['leaf_R8pc'], line=uid, pair=pr, side=sd, tri=tr) for o, c, pr, sd, tr in t]
    OLD.write_files(tf, 'T21_flat', False)
    json.dump(checks, open(HERE / 'port_check.json', 'w'), indent=1); print(json.dumps(checks)); [print(r['leaf_uid'], r['compound'], 'rank', r['rank'], 'leaflets', r['leaflet_count']) for r in rows_info]


if __name__ == '__main__':
    main()
