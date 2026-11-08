"""Engine hook `random_parameters` (H prompt H2/H3, REVISION 2+3; request handoff/HOOK_REQUEST_RANDOM_PARAMETERS_2026-10-20.md).

Bounded per-plant / per-organ draws of the group 1-4 parameters, physics unchanged. Default off: the native template is returned byte for byte.
On: a native fragment is injected and the constants are replaced by draws made inside the engine, keyed by (seed, id, key) with java.util.Random, so
the same plant / organ id draws the same value in every scenario (common random numbers). Plant-level: PHYLLOCHRON, AMAX, ALPHA, SLA factors,
INTERNODE_CAP_M, PAIRS choice, handedness. Organ-level at birth: MAX_MASS factor, MIDDLE_RAMP, DIVERGENCE_DEG (per leaf, cumulative azimuth),
COMPOUND_INSERTION_DEG, TOTAL_BEND_DEG, BASAL_LEAFLET_ANGLE_DEG. Imported organs keep their stored values. Every draw is written to
working/random_draws.csv (id, key, u, value). Bounds come from tools/microtom_parameters.csv rows (uncertainty_kind bounded_random) and are carried
in the request as hook parameter records; not drawn: reserve cap, lamp, stagger, LEAFLET_MID_DD/K, bud-break (own draw), FIRST_LEAVES (own draw),
outline (own draw), FLOWERS (separate flowers_per_truss hook), S1 fruit/truss sink parameters."""
from __future__ import annotations

import json
from pathlib import Path

from tools.vtc_growth_io import require

SWITCH = 'random_parameters'
SEED_KEY = 'RANDOM_PARAMETERS_SEED'
PAIRS_KEY = 'RP_PAIRS'
# key -> (unit, physical lower limit, physical upper limit, draw unit, group)
BOUND_KEYS = {
    'RP_PHYLLOCHRON_FACTOR': ('factor', 0.0, None, 'plant', 1),
    'RP_AMAX_FACTOR': ('factor', 0.0, None, 'plant', 3),
    'RP_ALPHA_FACTOR': ('factor', 0.0, None, 'plant', 3),
    'RP_SLA_FACTOR': ('factor', 0.0, None, 'plant', 3),
    'RP_INTERNODE_CAP_M': ('m', 0.0, None, 'plant', 4),
    'RP_MIDDLE_RAMP': ('dimensionless', 0.0, None, 'compound', 2),
    'RP_MAX_MASS_FACTOR': ('factor', 0.0, None, 'organ', 3),
    'RP_DIVERGENCE_DEG': ('deg', 0.0, 180.0, 'leaf', 4),
    'RP_COMPOUND_INSERTION_DEG': ('deg', 0.0, 90.0, 'compound', 4),
    'RP_TOTAL_BEND_DEG': ('deg', 0.0, 90.0, 'compound', 4),
    'RP_BASAL_LEAFLET_ANGLE_DEG': ('deg', 0.0, 90.0, 'compound', 4),
}
PARAMETERS = {SEED_KEY, PAIRS_KEY, *BOUND_KEYS}
SOURCES = ('user_assumed', 'literature', 'literature_other_cultivar', 'literature_wild_species', 'user_stated')
NATIVE = Path(__file__).with_name('fsp_random_parameters_native.rgg')


def enabled(req: dict) -> bool:
    return req.get('engine_hooks', {}).get(SWITCH, False) is True


active = enabled


def _records(req: dict) -> dict:
    return req.get('engine_hooks', {}).get('parameters', {})


def validate(req: dict) -> None:
    params = _records(req)
    present = set(params) & PARAMETERS
    if not enabled(req):
        require(not present, 'random_parameters records require the random_parameters switch')
        return
    require(present == PARAMETERS, f'random_parameters needs every record: missing {sorted(PARAMETERS - present)}')
    from tools import fsp_temperature
    require(not any(req['engine_hooks'].get(s, False) for s in fsp_temperature.SWITCHES), 'random_parameters: S2-T switches must be off (the photosynthesis line is shared)')
    require(not req.get('extensions'), 'random_parameters: optical extensions (bend mixture) must be off; TOTAL_BEND_DEG is drawn here')
    require(req.get('initialization_route') != 'seed', 'random_parameters: fixture-host runs only (H3.8); the seed-route replay keeps one clock for all plants')
    for key in PARAMETERS:
        rec = params[key]
        require(all(rec.get(k) for k in ('unit', 'source_ref', 'cultivar', 'uncertainty')), f'{key}: provenance incomplete')
        require(rec.get('source_kind') in SOURCES, f'{key}: source_kind not accepted')
    seed = params[SEED_KEY]['value']
    require(type(seed) is int and not isinstance(seed, bool) and 0 <= seed < 2 ** 62, 'RANDOM_PARAMETERS_SEED must be a non-negative integer')
    require(params[SEED_KEY]['unit'] == 'seed', 'RANDOM_PARAMETERS_SEED unit')
    for key, (unit, lo_lim, hi_lim, _, _) in BOUND_KEYS.items():
        rec = params[key]; v = rec['value']
        require(rec['unit'] == unit, f'{key}: unit must be {unit}')
        require(isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v), f'{key}: value must be [lo, hi]')
        lo, hi = float(v[0]), float(v[1])
        require(lo < hi, f'{key}: lo < hi required')
        require(lo >= lo_lim and (hi_lim is None or hi <= hi_lim), f'{key}: bounds outside the physical range')
    choices = params[PAIRS_KEY]['value']
    require(params[PAIRS_KEY]['unit'] == 'choice', 'RP_PAIRS unit must be choice')
    require(isinstance(choices, list) and len(choices) >= 2 and all(isinstance(c, list) and c and all(type(x) is int and x >= 1 for x in c) for c in choices), 'RP_PAIRS: >= 2 integer pair lists')


def bounds(req: dict) -> dict:
    p = _records(req)
    return {k: [float(p[k]['value'][0]), float(p[k]['value'][1])] for k in BOUND_KEYS}


def metadata(req: dict) -> dict:
    p = _records(req)
    return dict(switch=True, seed=p[SEED_KEY]['value'], bounds=bounds(req), pairs_choices=p[PAIRS_KEY]['value'],
                draw_units={k: BOUND_KEYS[k][3] for k in BOUND_KEYS}, provenance='model_drawn (every drawn value); bounds as recorded per key',
                mechanism='java.util.Random(seed + hash(id:key)), first output skipped; plant level once at run start, organ level once at birth; imported organs unchanged',
                not_drawn=['reserve cap', 'cap floor', 'lamp', 'LEAFLET_STAGGER_DD', 'LEAFLET_MID_DD/K', 'BUD_BREAK_* (own draw)', 'FIRST_LEAVES (own draw)', 'outline (own draw)', 'FLOWERS', 'S1 fruit/truss sink'],
                output='working/random_draws.csv', statement='between-plant variation is assumed, not measured')


def registry_records(registry: list[dict]) -> dict:
    """Hook parameter records for the request from the registry's bounded_random rows (candidates = bounds / choices)."""
    rows = {r['key']: r for r in registry if r['uncertainty_kind'] == 'bounded_random'}
    missing = (set(BOUND_KEYS) | {PAIRS_KEY}) - set(rows)
    require(not missing, f'registry lacks bounded_random rows: {sorted(missing)}')
    out = {}
    for key in (*BOUND_KEYS, PAIRS_KEY):
        r = rows[key]
        out[key] = dict(value=r['candidates'], unit=r['unit'], source_kind=r['source_kind'], source_ref=f'{key}: {r["source_ref"]} [{r["note"]}]',
                        cultivar=r['cultivar'], uncertainty=r['uncertainty'])
    return out


def seed_record(seed: int, note: str) -> dict:
    return dict(value=int(seed), unit='seed', source_kind='user_assumed', source_ref=f'RANDOM_PARAMETERS_SEED: {note}', cultivar='Micro-Tom_unfitted_hypothesis',
                uncertainty='seed of a declared random-parameter member; not a plant value')


def transform_native(template: str, req: dict) -> str:
    """Inject the draw fragment and replace the constants; disabled requests return the exact input bytes."""
    validate(req)
    if not enabled(req):
        return template
    p = _records(req); b = bounds(req)
    code = NATIVE.read_text(encoding='utf-8').replace('@RP_SEED@', f'{int(p[SEED_KEY]["value"])}L')
    for key, (lo, hi) in b.items():
        code = code.replace(f'@{key}_LO@', repr(lo)).replace(f'@{key}_HI@', repr(hi))
    code = code.replace('@RP_PAIRS_CHOICES@', 'new int[][]{' + ','.join('new int[]{' + ','.join(str(x) for x in c) + '}' for c in p[PAIRS_KEY]['value']) + '}')
    require('@RP_' not in code, 'unresolved random_parameters token')

    def replace(old: str, new: str, count: int = 1) -> None:
        nonlocal template
        require(template.count(old) == count, f'random_parameters insertion point changed ({template.count(old)} != {count}): ' + old[:70])
        template = template.replace(old, new)

    replace('float[] shape=null;', 'float[] shape=null; double rpAz=0,rpInsertion=0,rpBasal=0,rpRamp=0;')
    replace('@AXIS_CODE@', '@AXIS_CODE@\n' + code)
    # organ birth: MAX_MASS factor for organs born in the run (events is null during initialisation and the observation import)
    replace('o.maxMass=MAX_MASS[kind];', 'o.maxMass=MAX_MASS[kind];if(events!=null)o.maxMass*=rpMaxMass(o.id);')
    # compound birth: per-compound draws, per-plant PAIRS, cumulative azimuth from per-leaf divergence with the plant's handedness
    replace('    c.bendAmplitude=mixtureAmplitude(c.id);\n    int m=pairs(rank);',
            '    c.rpInsertion=rpDraw(c.id,"COMPOUND_INSERTION_DEG",rpInsertionLo,rpInsertionHi);c.rpBasal=rpDraw(c.id,"BASAL_LEAFLET_ANGLE_DEG",rpBasalLo,rpBasalHi);c.rpRamp=rpDraw(c.id,"MIDDLE_RAMP",rpRampLo,rpRampHi);\n'
            '    if(!rpStarted[node.controller]){rpStarted[node.controller]=true;rpAzimuth[node.controller]=((rank-1)*DIVERGENCE_DEG)%360;}\n'
            '    else rpAzimuth[node.controller]+=rpHand[p]*rpDraw(c.id,"DIVERGENCE_DEG",rpDivergenceLo,rpDivergenceHi);\n'
            '    c.rpAz=rpAzimuth[node.controller];\n'
            '    c.bendAmplitude=rpDraw(c.id,"TOTAL_BEND_DEG",rpBendLo,rpBendHi);\n    int m=pairs(p,rank);')
    replace('for(int i=1;i<=m;i++) total+=2*pairWeight(i,m);', 'for(int i=1;i<=m;i++) total+=2*pairWeight(i,m,c.rpRamp);')
    replace('l.pair=i;l.side=side;l.share=pairWeight(i,m)/total;l.maxMass=MAX_MASS[9]*l.share;',
            'l.pair=i;l.side=side;l.rpBasal=c.rpBasal;l.share=pairWeight(i,m,c.rpRamp)/total;l.maxMass=MAX_MASS[9]*l.share*rpMaxMass(l.id);')
    replace('r.pair=i;r.maxMass=MAX_MASS[8]/m;', 'r.pair=i;r.maxMass=MAX_MASS[8]/m*rpMaxMass(r.id);')
    replace('terminal.pair=m+1;terminal.share=1/total;terminal.maxMass=MAX_MASS[9]/total;',
            'terminal.pair=m+1;terminal.rpBasal=c.rpBasal;terminal.share=1/total;terminal.maxMass=MAX_MASS[9]/total*rpMaxMass(terminal.id);')
    replace('double pairWeight(int i,int m) { return m==1?1:1+MIDDLE_RAMP*', 'double pairWeight(int i,int m,double ramp) { return m==1?1:1+ramp*')
    # pairs per plant for organs born in the run; imported organs keep PAIRS
    n = template.count('pairs(o.rank)'); require(n >= 3, 'pairs(o.rank) call sites changed')
    template = template.replace('pairs(o.rank)', 'pairsOf(o)')
    # plant-level rates
    replace('/PHYLLOCHRON_DD);', '/(PHYLLOCHRON_DD*rpPhyllochron[p]));')
    replace('calculatePS_Thornley(o.photons/o.area,AMAX,ALPHA,THETA)', 'calculatePS_Thornley(o.photons/o.area,AMAX*rpAmax[o.owner],ALPHA*rpAlpha[o.owner],THETA)')
    if 'o.area=o.mass*s1LeafSla(o);' in template:
        replace('o.area=o.mass*s1LeafSla(o);', 'o.area=o.mass*s1LeafSla(o)*rpSla[o.owner];')
    else:
        replace('o.area=o.mass*SLA_M2_G;', 'o.area=o.mass*SLA_M2_G*rpSla[o.owner];')
    replace('o.len=INTERNODE_CAP_M*fraction;', 'o.len=rpInternode[o.owner]*fraction;')
    replace('o.az=((o.rank-1)*DIVERGENCE_DEG)%360;o.elev=90-COMPOUND_INSERTION_DEG;', 'o.az=((o.rpAz%360)+360)%360;o.elev=90-o.rpInsertion;')
    nb = template.count('o.side*BASAL_LEAFLET_ANGLE_DEG*'); require(nb >= 2, 'basal angle sites changed')   # native + outline fragment (outlineVertices)
    template = template.replace('o.side*BASAL_LEAFLET_ANGLE_DEG*', 'o.side*rpBasalOf(o)*')
    # provenance file and the plant-level draws, right after the writers open and before the first geometry update
    replace('edges.println("day,parent_id,child_id");', 'edges.println("day,parent_id,child_id");\n    rpDraws=new PrintWriter(new FileWriter("random_draws.csv"));rpDraws.println("id,key,u,value");rpInitPlants();')
    replace('edges.close();', 'edges.close();rpDraws.close();')
    return template


def validate_draws(path: Path, req: dict) -> dict:
    """Independent check of working/random_draws.csv: every row inside its declared bounds, u in [0, 1), one row per (id, key)."""
    import csv
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    require(bool(rows), 'random_draws.csv empty')
    b = bounds(req); limits = {k.replace('RP_', ''): v for k, v in b.items()}
    seen = set(); bad = []
    for r in rows:
        k = (r['id'], r['key']); require(k not in seen, f'duplicate draw {k}'); seen.add(k)
        u = float(r['u']); v = float(r['value']); require(0 <= u < 1, f'u outside [0,1): {k}')
        if r['key'] in limits:
            lo, hi = limits[r['key']]
            if not (lo <= v <= hi): bad.append(k)
        elif r['key'] == 'PAIRS': require(0 <= int(float(r['value'])) < len(_records(req)[PAIRS_KEY]['value']), f'PAIRS choice out of range: {k}')
        elif r['key'] == 'HANDEDNESS': require(v in (-1., 1.), f'handedness: {k}')
        else: require(False, f'unknown draw key {r["key"]}')
    require(not bad, f'{len(bad)} draws outside bounds: {bad[:5]}')
    keys = {}
    for r in rows: keys[r['key']] = keys.get(r['key'], 0) + 1
    return dict(rows=len(rows), per_key=keys)


class Draws:
    """The run's recorded draws (working/random_draws.csv) for the independent checkers: plant factors / values, per-plant PAIRS, per-compound values.
    Inactive requests (or a missing file) return the request constants, so every checker keeps its old behaviour when the hook is off."""

    def __init__(self, req: dict, working: Path | None):
        self.req = req; self.on = enabled(req); self.values = {}
        self.choices = _records(req)[PAIRS_KEY]['value'] if self.on else None
        if self.on:
            import csv
            path = Path(working) / 'random_draws.csv' if working is not None else None
            require(path is not None and path.exists(), 'random_parameters on but working/random_draws.csv missing')
            for r in csv.DictReader(open(path, encoding='utf-8')):
                self.values[(r['id'], r['key'])] = float(r['value'])

    def plant(self, plant_id: str, key: str, default: float = 1.0) -> float:
        if not self.on: return default
        v = self.values.get((plant_id, key)); require(v is not None, f'draw missing: {plant_id}:{key}'); return v

    def pairs(self, plant_id: str, imported: bool) -> list[int]:
        base = self.req['parameters']['PAIRS']['value']
        if not self.on or imported: return base
        return [int(x) for x in self.choices[int(self.plant(plant_id, 'PAIRS'))]]

    def pair_count(self, plant_id: str, rank: int, imported: bool) -> int:
        p = self.pairs(plant_id, imported); return p[min(max(0, rank - 1), len(p) - 1)]

    def compound(self, compound_id: str, key: str, default: float) -> float:
        if not self.on: return default
        v = self.values.get((compound_id, key)); require(v is not None, f'draw missing: {compound_id}:{key}'); return v


def compound_of(organ_id: str) -> str:
    """Compound id of a leaflet / petiole / rachis id ('fx_om0/C0003/P1L' -> 'fx_om0/C0003'; axis-prefixed ids keep their prefix)."""
    return organ_id.rsplit('/', 1)[0]
