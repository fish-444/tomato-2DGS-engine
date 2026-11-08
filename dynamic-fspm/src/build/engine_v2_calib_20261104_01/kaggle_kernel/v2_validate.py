"""Engine v2 output validator (build/engine_v2_calib_20261104_01/CONTRACT.md §1, ADAPTERS.md V1-V8).
= the frozen tools.vtc_development_io.validate_outputs source with only the replay rules the v2 laws change replaced; every replacement is asserted
to occur exactly once in the frozen source. install(cfg) puts the v2 function into tools.vtc_development_io (the bridge imports it inside run()).
cfg = v2_config.json of the case (level flags, SLA base, rule tables). Levels below C1 never call install (B0 uses the frozen validator)."""
from __future__ import annotations

import inspect
import math
import re

CFG: dict = {}


class V2:
    C4 = False; SLA = 0.025; RM = []

    @staticmethod
    def rmult(s):                                               # V12: C4A per-rank rachis multiplier (main axis only)
        if re.match(r'^[^/]+/S[0-9]+/', str(s['axis_id'])): return 1.0
        r = int(s['true_leaf_order']); return V2.RM[r - 1] if 1 <= r <= len(V2.RM) else 1.0

    @staticmethod
    def internode_cap(rpd, pid, oid, par):                      # V3: per-internode Beta draw (mm) for organs born in the run
        v = rpd.values.get((oid, 'INTERNODE_MM')) if rpd.on else None
        return v / 1000.0 if v is not None else rpd.plant(pid, 'INTERNODE_CAP_M', par['INTERNODE_CAP_M'])

    @staticmethod
    def compound_inventory(rpd, oid, m, blades, rach, require):  # V2: per-compound leaflet count; n = 6 -> left leaflet of pair m omitted
        n = rpd.values.get((oid, 'LEAFLET_COUNT')) if rpd.on else None
        if n is None:
            require(len(blades) == 2 * m + 1 and len(rach) == m, 'rank-indexed compound inventory')
            want = {(i, s) for i in range(1, m + 1) for s in (-1, 1)} | {(m + 1, 0)}
        else:
            n = int(n); require(m == n // 2, 'v2 pair count / leaflet draw mismatch')
            want = {(i, s) for i in range(1, m + 1) for s in (-1, 1)} | {(m + 1, 0)}
            if n == 6: want.discard((m, -1))
            require(len(blades) == n and len(rach) == m, 'v2 per-compound inventory')
        require({(int(c['pair_index']), int(c['side'])) for c in blades} == want, 'terminal/pair identity mismatch')

    @staticmethod
    def controller(organ_id: str) -> str:
        parts = organ_id.split('/'); return parts[1] if len(parts) > 1 and re.fullmatch(r'S[0-9]+', parts[1]) else 'M'

    @staticmethod
    def same_controller(a: str, b: str) -> bool:                # V6: a determinate stop closes its own controller only
        return (not V2.C4) or V2.controller(a) == V2.controller(b)

    @staticmethod
    def cat_value(u: float, table: dict) -> int:
        ks = sorted(table, key=float); tot = sum(float(table[k]) for k in ks); acc = 0.0
        for k in ks:
            acc += float(table[k]) / tot
            if u < acc: return int(k)
        return int(ks[-1])

    @staticmethod
    def schedule_check(req, pkg, par, rpd, fptd, days, n, stage, interval_seconds, require):
        """V7: independent v2 developmental schedule per controller (main targets N, T-N, 0; side created at first eligibility, clock from 0,
        first leaf after BUD_DELAY + one phyllochron, targets 2, S-2, 0). Compares compound / truss counts per controller with the states."""
        require(not stage.enabled(req, 'process_base_temperatures') and not req.get('engine_hooks', {}).get('development_temperature_curve'), 'v2 schedule: base-temperature hooks unsupported')
        C = CFG; raw = {}
        import csv
        for pid in {p['plant_instance_id'] for p in pkg['tables']['plants.csv']}:
            phyl = par['PHYLLOCHRON_DD'] * rpd.plant(pid, 'PHYLLOCHRON_FACTOR')
            def d(i, k): v = rpd.values.get((i, k)); require(v is not None, f'v2 draw missing {i}:{k}'); return int(v)
            N, T = d(pid, 'MAIN_TRUSS_NODE'), d(pid, 'MAIN_TOTAL_LEAVES')
            require(N in C['main_truss_node_int'] and T in C['main_total_int'], 'v2 main draw outside its declared values')
            ctl = {'M': dict(first=N, rest=T - N, germ=par['GERMINATION_DD'], created=True, attach=0)}
            ns = d(pid, 'SIDE_AXES_N') if C.get('c4t') else C['n_side']
            if C.get('c4t'):
                require(ns in {int(k) for k in C['c4t_pmf']['side_axes']}, 'v2 side-axis count outside its declared values'); require(T >= N + 2, 'C4T coupling T >= N + 2 violated')
            for j in range(1, ns + 1):
                S, r = d(f'{pid}/S{j}', 'SIDE_TOTAL_LEAVES'), d(f'{pid}/S{j}', 'SIDE_ATTACH_RANK')
                require(S in C['side_total_int'] and r in C['side_attach_int'], 'v2 side draw outside its declared values')
                ctl[f'S{j}'] = dict(first=C['side_first_truss'], rest=S - C['side_first_truss'], germ=C['bud_delay_dd'] + ((j - 1) * phyl if C.get('c4t') else 0.0), created=False, attach=r)
            ranks = [q['attach'] for k_, q in ctl.items() if k_ != 'M']; require(len(set(ranks)) == len(ranks), 'v2 attach ranks drawn without replacement')
            for q in ctl.values(): q.update(clock=0.0, produced=0, leaves=0, trusses=0, axis_leaves=0, stopped=False, created_day=None)
            for day in range(1, n + 1):
                dt_days = interval_seconds(req, day - 1) / 86400; e = pkg['tables']['environment.csv'][day - 1]; h = float(e['light_hours'])
                dd = max(0, (float(e['temp_day_c']) * h + float(e['temp_night_c']) * (24 * dt_days - h)) / (24 * dt_days) - par['BASE_C']) * dt_days
                for q in ctl.values():
                    if q['created']: q['clock'] += dd
                for name, q in ctl.items():
                    if req['freeze_at'] >= 0 and day - 1 >= req['freeze_at']: break
                    if not q['created']:
                        M = ctl['M']
                        if M['trusses'] >= 1 and M['leaves'] >= q['attach']:
                            q['created'] = True; q['created_day'] = day; q['clock'] = 0.0
                        continue
                    if q['clock'] < q['germ']: continue
                    due = math.floor((q['clock'] - q['germ']) / phyl)
                    while q['produced'] < due and not q['stopped']:
                        q['produced'] += 1; target = q['first'] if q['trusses'] == 0 else (q['rest'] if q['trusses'] == 1 else 0)
                        if q['axis_leaves'] < target: q['leaves'] += 1; q['axis_leaves'] += 1
                        else: q['trusses'] += 1; q['stopped'] = q['trusses'] > 1 and target == 0; q['axis_leaves'] = 0
                ps = [s for s in days[day].values() if s['plant_instance_id'] == pid]
                for name, q in ctl.items():
                    mine = [s for s in ps if V2.controller(s['organ_id']) == name]
                    require(sum(s['class'] == 'CompoundLeaf' for s in mine) == q['leaves'], f'v2 compound schedule mismatch {name} day {day}')
                    require(sum(s['class'] == 'TrussAxis' for s in mine) == q['trusses'], f'v2 truss schedule mismatch {name} day {day}')
                    if name != 'M':
                        ax = [s for s in mine if s['organ_id'] == f'{pid}/{name}/A0']
                        require(bool(ax) == q['created'] and (not ax or int(ax[0]['birth_day']) == q['created_day']), f'v2 side axis creation day {name}')
                require(sum(s['class'] in ('Flower', 'Fruit') for s in ps) == sum(fptd.count(t['organ_id']) for t in ps if t['class'] == 'TrussAxis'), 'flower/fruit unit inventory mismatch')
                require(sum(s['class'] == 'Root' for s in ps) == 1, 'root compartment inventory')
            raw[pid] = {k: dict(first=v['first'], rest=v['rest'], attach=v['attach'], created_day=v['created_day'], leaves=v['leaves'], trusses=v['trusses'], stopped=v['stopped']) for k, v in ctl.items()}
        CFG['_schedule'] = raw


REPLACEMENTS = [
    ("                    close(f(s,'length_m'),(par['PETIOLE_M'] if kind==7 else par['RACHIS_M']/m)*fraction,'leaf-pair funded rachis/petiole length')\n",
     "                    close(f(s,'length_m'),(par['PETIOLE_M'] if kind==7 else par['RACHIS_M']/m)*fraction,'leaf-pair funded rachis/petiole length')\n"
     "                elif engine_hook_enabled(req,'leaf_development') and hooks['LEAFLET_GROWTH_ORDER']['value']=='simultaneous':   # V12 (Amendment 17)\n"
     "                    m=rpd.pair_count(pid,int(s['true_leaf_order']),oid in imported)\n"
     "                    close(f(s,'length_m'),((par['PETIOLE_M'] if kind==7 else par['RACHIS_M']/m*V2.rmult(s))*(f(s,'mass_gDM')/f(s,'max_mass_gDM'))**(1/3)) if f(s,'max_mass_gDM')>0 else 0.,'own-funded rachis/petiole length (V12)')\n"),
    ("close(f(s,'length_m'),rpd.plant(pid,'INTERNODE_CAP_M',par['INTERNODE_CAP_M'])*(f(s,'mass_gDM')/f(s,'max_mass_gDM'))**(1/3),'funded internode')",
     "close(f(s,'length_m'),V2.internode_cap(rpd,pid,oid,par)*(f(s,'mass_gDM')/f(s,'max_mass_gDM'))**(1/3),'funded internode')"),
    ("else par['SLA_M2_G']*rpd.plant(pid,'SLA_FACTOR')", "else V2.SLA*rpd.plant(pid,'SLA_FACTOR')"),
    ("            require(len(blades)==2*m+1 and len(rach)==m,'rank-indexed compound inventory')\n"
     "            require({(int(c['pair_index']),int(c['side'])) for c in blades}=={(i,side) for i in range(1,m+1) for side in (-1,1)}|{(m+1,0)},'terminal/pair identity mismatch')\n",
     "            V2.compound_inventory(rpd,oid,m,blades,rach,require)\n"),
    ("if s['plant_instance_id']==pid and (not enabled(req) or s['axis_id']==axis)),'new organ after declared axis stop')",
     "if s['plant_instance_id']==pid and (not enabled(req) or s['axis_id']==axis) and V2.same_controller(s['organ_id'],e['organ_id'])),'new organ after declared axis stop')"),
    ("    if req['initialization_route']=='seed':\n        # Independent developmental clock",
     "    if req['initialization_route']=='seed' and V2.C4:\n        V2.schedule_check(req,pkg,par,rpd,fptd,days,n,stage,interval_seconds,require)\n"
     "    if req['initialization_route']=='seed' and not V2.C4:\n        # Independent developmental clock"),
]


C4L_CHAIN_MEDIAN_M = 0.0382264   # c4l_patch.CHAIN_MEDIAN_MM / 1000


def c4l_frame(state, states, draws):
    """V8: replay of the C4L leaflet frame (c4l_patch.java_helpers.v2PlaceLeaflets) from the recorded draws and the chain organs' states.
    Returns (u, v, base): long axis, in-plane second axis, base position (metres). Asserts it equals the exported state (az/el, x/y/z)."""
    import numpy as np, math
    from tools.vtc_growth_io import require
    oid = state['organ_id']; side = int(state['side']); vals = draws.values
    comp = state['parent_id']
    while states[comp]['class'] != 'CompoundLeaf':
        comp = states[comp]['parent_id']
    cache = c4l_frame.__dict__.setdefault('_cache', {})
    key = id(states)
    if cache.get('key') != key:
        cache.clear(); cache['key'] = key; chains = {}
        for r in states.values():
            if r['class'] in ('Petiole', 'RachisSegment'):
                c = r['parent_id']
                while states[c]['class'] != 'CompoundLeaf':
                    c = states[c]['parent_id']
                chains.setdefault(c, []).append(r)
        for c in chains:
            chains[c].sort(key=lambda r: (0, 0) if r['class'] == 'Petiole' else (1, int(r['pair_index'])))
        cache['chains'] = chains
    ch = cache['chains'][comp]; f = lambda r, k: float(r[k])
    Lc = sum(f(r, 'length_m') for r in ch)
    q = ch[-1]; fr = 1.0
    sfrac = 1.0 if side == 0 else vals[(oid, 'LEAFLET_S_OVER_L')]
    if Lc > 0:
        target = sfrac * Lc; acc = 0.0; k = 0
        while k < len(ch) - 1:
            L = f(ch[k], 'length_m')
            if acc + L >= target: break
            acc += L; k += 1
        q = ch[k]; L = f(q, 'length_m'); fr = max(0.0, min(1.0, (target - acc) / L)) if L > 0 else 1.0
    base = np.array([f(q, 'x_m') + fr * (f(q, 'tip_x_m') - f(q, 'x_m')), f(q, 'y_m') + fr * (f(q, 'tip_y_m') - f(q, 'y_m')), f(q, 'z_m') + fr * (f(q, 'tip_z_m') - f(q, 'z_m'))])
    pa, pe = math.radians(f(q, 'azimuth_deg')), math.radians(f(q, 'elevation_deg'))
    t = np.array([math.cos(pe) * math.cos(pa), math.cos(pe) * math.sin(pa), math.sin(pe)]); h = np.array([-math.sin(pa), math.cos(pa), 0.0])
    n0 = np.array([-math.sin(pe) * math.cos(pa), -math.sin(pe) * math.sin(pa), math.cos(pe)])
    phi = 0.0 if side == 0 else math.radians(vals[(oid, 'LEAFLET_SPREAD_PHI_DEG')])
    u = t if side == 0 else math.cos(phi) * t + side * math.sin(phi) * h
    off = 0.0 if side == 0 else vals[(oid, 'LEAFLET_PETIOLULE_MM')] / 1000.0 * Lc / C4L_CHAIN_MEDIAN_M
    base = base + side * off * h
    v0 = np.cross(n0, u); ang = math.radians(vals[(oid, 'LEAFLET_ROLL_SIGN')] * vals[(oid, 'LEAFLET_TILT_DEG')])
    v = math.cos(ang) * v0 + math.sin(ang) * n0
    az, el = np.radians([f(state, 'azimuth_deg'), f(state, 'elevation_deg')])
    u_state = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    require(np.allclose(u, u_state, atol=1e-9), 'C4L leaflet axis mismatch (draws / chain replay vs exported azimuth, elevation)')
    require(np.allclose(base, [f(state, 'x_m'), f(state, 'y_m'), f(state, 'z_m')], atol=1e-9), 'C4L leaflet base mismatch (s / offset replay vs exported position)')
    return u, v, base



def c4b_expected(state, uv, u, v, base, area, draws):
    """V9: replay of c4b_patch v2BendInto (quadric bend in the leaflet frame, similarity-scaled, area-conserving lambda iteration)."""
    import numpy as np, math
    oid = state['organ_id']; vals = draws.values; side = int(state['side'])
    pos = 3 if side == 0 else min(int(state['pair_index']), 3) - 1
    lref = [CFG['c4b_lref_mm'][k] / 1000 for k in ('P1', 'P2', 'P3', 'T')][pos]
    L = float(state['length_m']); f = lref / L if L > 0 else 1.0
    kl = vals[(oid, 'LEAFLET_KAPPA_L')] * 1000 * f; kw = vals[(oid, 'LEAFLET_KAPPA_W')] * 1000 * f; kc = vals[(oid, 'LEAFLET_TWIST_C')] * 1000 * f
    n = np.cross(u, v); n = n / np.linalg.norm(n)
    if n[2] < 0: n = -n
    fl = np.asarray(uv, float).reshape(-1, 2); tri = fl.reshape(-1, 3, 2)
    a2 = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    A2 = a2.sum(); uc = float((a2 * tri[:, :, 0].mean(1)).sum() / A2); vc = float((a2 * tri[:, :, 1].mean(1)).sum() / A2)
    scale = math.sqrt(area); target = A2 * scale * scale; lam = 1.0
    for _ in range(60):
        s0 = -lam * scale * uc; d0 = -lam * scale * vc; wb = 0.5 * kl * s0 * s0 + 0.5 * kw * d0 * d0 + kc * s0 * d0
        s = lam * scale * (fl[:, 0] - uc); d = lam * scale * (fl[:, 1] - vc); w = 0.5 * kl * s * s + 0.5 * kw * d * d + kc * s * d - wb
        P = base + lam * scale * (fl[:, :1] * u + fl[:, 1:] * v) + w[:, None] * n
        T = P.reshape(-1, 3, 3); A = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum()
        if A <= 0 or abs(A / target - 1) < 1e-13: break
        lam *= math.sqrt(target / A)
    return P.reshape(-1, 3, 3)

def install_c4l(c4b=False):
    import re, inspect
    import tools.fsp_leaflet_outline as lo
    src = inspect.getsource(lo.check_leaf)
    pat = re.compile(r"    parent=states\[state\['parent_id'\]\];pa,pe=.*?-math\.sin\(beta\)\*math\.sin\(pe\)\]\)\n", re.S)
    assert len(pat.findall(src)) == 1, 'check_leaf frame block not found'
    src = pat.sub(lambda _m: "    u,v,_c4l_base=V2.c4l_frame(state,states,draws)\n", src)
    if c4b:
        old = "    expected=base+math.sqrt(area)*(uv[:,:,0,None]*u+uv[:,:,1,None]*v)\n"
        assert src.count(old) == 1, 'check_leaf expected line not found'
        src = src.replace(old, "    expected=V2.c4b_expected(state,uv,u,v,base,area,draws)\n")
        old2 = "require(minimum_angle>=3.,'placed outline angle below hard floor')"
        assert src.count(old2) == 1, 'angle floor line not found'
        src = src.replace(old2, "require(minimum_angle>=1.,'placed outline angle below hard floor (C4B bent leaflet: 1 deg, CONTRACT Amendment 13)')")   # V10
    ns = dict(lo.__dict__); ns['V2'] = type('V2', (), dict(c4l_frame=staticmethod(c4l_frame), c4b_expected=staticmethod(c4b_expected)))
    exec(compile(src, 'v2_validate<check_leaf_c4l>', 'exec'), ns)
    lo.check_leaf = ns['check_leaf']; return True


def install(cfg: dict) -> dict:
    """Install the v2 validator and the two draw-reader overrides (V1 per-compound pairs by unique rank; V5 per-truss flowers)."""
    import tools.vtc_development_io as io
    from tools import fsp_random_parameters as rp, fsp_flowers_per_truss as fpt
    CFG.clear(); CFG.update(cfg)
    CFG['main_truss_node_int'] = {int(k) for k in cfg['main_truss_node']}; CFG['main_total_int'] = {int(k) for k in cfg['main_total']}
    CFG['side_total_int'] = {int(k) for k in cfg['side_total']}; CFG['side_attach_int'] = {int(k) for k in cfg['side_attach']}
    V2.C4 = bool(cfg['v2_c4']); V2.SLA = float(cfg['sla_m2_g']); V2.RM = [float(x) for x in cfg.get('c4a_rachis_main', [])]
    src = inspect.getsource(io.validate_outputs); applied = []
    for old, new in REPLACEMENTS:
        assert src.count(old) == 1, ('validator replacement point', src.count(old), old[:80]); src = src.replace(old, new); applied.append(old[:60])
    ns = dict(io.__dict__); ns['V2'] = V2; exec(compile(src, 'v2_validate<validate_outputs>', 'exec'), ns)
    v2_fn = ns['validate_outputs']

    def validate_outputs(pkg, native, binding, checkpoint=False):           # V4b: every replay (incl. S1 birth-frozen SLA) sees the v2 SLA base
        import copy
        pkg = dict(pkg); req = copy.deepcopy(pkg['request']); req['parameters']['SLA_M2_G']['value'] = V2.SLA; pkg['request'] = req
        if CFG.get('c4c'):                                                     # V11 (CONTRACT Amendment 14): C4C chain 40 mm
            req['parameters']['PETIOLE_M']['value'] = float(CFG['c4c']['petiole_m']); req['parameters']['RACHIS_M']['value'] = float(CFG['c4c']['rachis_m'])
        return v2_fn(pkg, native, binding, checkpoint)
    io.validate_outputs = validate_outputs

    orig_pairs = rp.Draws.pairs

    def pairs(self, plant_id, imported):                                         # V1
        if imported or not self.on: return orig_pairs(self, plant_id, imported)
        cache = self.__dict__.setdefault('_v2_pairs', {})
        if plant_id not in cache:
            base = orig_pairs(self, plant_id, imported); per = {}
            for (i, k), v in self.values.items():
                if k == 'LEAFLET_COUNT' and i.startswith(plant_id + '/'):
                    mt = re.search(r'C([0-9]{4})$', i); per[int(mt.group(1))] = int(v) // 2
            if per:
                out = [base[min(r - 1, len(base) - 1)] for r in range(1, max(per) + 1)]
                for r, m in per.items(): out[r - 1] = m
                cache[plant_id] = out
            else:
                cache[plant_id] = base
        return cache[plant_id]
    rp.Draws.pairs = pairs

    orig_init = fpt.Draws.__init__

    def init(self, req, working):                                                # V5
        orig_init(self, req, working)
        from pathlib import Path
        import csv
        p = Path(working) / 'random_draws.csv' if working is not None else None
        if p is not None and p.exists():
            got = {r['id']: int(float(r['value'])) for r in csv.DictReader(open(p, encoding='utf-8')) if r['key'] == 'FLOWERS_PER_TRUSS'}
            if got: self.on = True; self.counts.update(got)
    fpt.Draws.__init__ = init
    c4l = install_c4l(bool(cfg.get('c4b'))) if cfg.get('c4l') else False
    return dict(applied=applied, c4=V2.C4, sla=V2.SLA, c4l=c4l, c4c=CFG.get('c4c'))
