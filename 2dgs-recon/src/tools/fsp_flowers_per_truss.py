"""Engine hook `flowers_per_truss` (DEV 2026-10-20; handoff/CLI_PROMPT_ENGINE_DEV_SESSION_2026-10-20.md Hook 1, H prompt REVISION 3 arm H_f48).

Flower count per truss drawn once at truss creation from an inclusive integer range (registry row FLOWERS_PER_TRUSS_RANGE, default [4, 8]).
Default off: every truss gets FLOWERS (4) and the native template is returned byte for byte. On: a native fragment is injected; the draw is keyed by
(seed, truss id) with java.util.Random(seed + (truss_id + ":FLOWERS").hashCode()), first output skipped - the same mechanism as the
random_parameters hook, so the same truss id draws the same count in every scenario (common random numbers) independent of run order.
Imported trusses keep their flowers (the draw happens only in develop()). Every draw is written to working/flower_draws.csv
(truss_id, u, n_flowers). The independent checkers read that file through `Draws`; `java_u` is a pure-Python replica of the native draw so the
recorded draws can be re-derived without the JVM. Separate from random_parameters: its own switch, seed and provenance file; both may be on."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from tools.vtc_growth_io import require

SWITCH = 'flowers_per_truss'
SEED_KEY = 'FLOWERS_PER_TRUSS_SEED'
RANGE_KEY = 'FLOWERS_PER_TRUSS_RANGE'
PARAMETERS = {SEED_KEY, RANGE_KEY}
DRAW_KEY = 'FLOWERS'
OUTPUT = 'flower_draws.csv'
SOURCES = ('user_assumed', 'literature', 'literature_microtom', 'literature_other_cultivar', 'literature_wild_species', 'user_stated')
NATIVE = Path(__file__).with_name('fsp_flowers_per_truss_native.rgg')
_MULT, _MASK = 0x5DEECE66D, (1 << 48) - 1


def enabled(req: dict) -> bool:
    return req.get('engine_hooks', {}).get(SWITCH, False) is True


active = enabled


def _records(req: dict) -> dict:
    return req.get('engine_hooks', {}).get('parameters', {})


def validate(req: dict) -> None:
    params = _records(req)
    present = set(params) & PARAMETERS
    if not enabled(req):
        require(not present, 'flowers_per_truss records require the flowers_per_truss switch')
        return
    require(present == PARAMETERS, f'flowers_per_truss needs every record: missing {sorted(PARAMETERS - present)}')
    for key in PARAMETERS:
        rec = params[key]
        require(all(rec.get(k) for k in ('unit', 'source_ref', 'cultivar', 'uncertainty')), f'{key}: provenance incomplete')
        require(rec.get('source_kind') in SOURCES, f'{key}: source_kind not accepted')
    seed = params[SEED_KEY]['value']
    require(type(seed) is int and 0 <= seed < 2 ** 62, 'FLOWERS_PER_TRUSS_SEED must be a non-negative integer')
    require(params[SEED_KEY]['unit'] == 'seed', 'FLOWERS_PER_TRUSS_SEED unit')
    rec = params[RANGE_KEY]; v = rec['value']
    require(rec['unit'] == 'count', 'FLOWERS_PER_TRUSS_RANGE unit must be count')
    require(isinstance(v, list) and len(v) == 2 and all(type(x) is int for x in v), 'FLOWERS_PER_TRUSS_RANGE: value must be [lo, hi] integers')
    require(1 <= v[0] < v[1], 'FLOWERS_PER_TRUSS_RANGE: 1 <= lo < hi required')


def bounds(req: dict) -> tuple[int, int]:
    v = _records(req)[RANGE_KEY]['value']
    return int(v[0]), int(v[1])


def seed(req: dict) -> int:
    return int(_records(req)[SEED_KEY]['value'])


def metadata(req: dict) -> dict:
    lo, hi = bounds(req)
    return dict(switch=True, seed=seed(req), range=[lo, hi], draw_unit='truss', provenance='model_drawn (every drawn count); range as recorded',
                mechanism='java.util.Random(seed + hash(truss_id:FLOWERS)), first output skipped; n = lo + floor(u * (hi - lo + 1)) capped at hi; once at truss creation; imported trusses unchanged',
                default_when_off='FLOWERS (registry) on every truss', output='working/' + OUTPUT, statement='between-truss variation is assumed, not measured')


def registry_record(registry: list[dict]) -> dict:
    """Hook parameter record for the request from the registry's FLOWERS_PER_TRUSS_RANGE row (candidates = inclusive bounds)."""
    row = next((r for r in registry if r['key'] == RANGE_KEY), None)
    require(row is not None and row['uncertainty_kind'] == 'bounded_random', f'registry lacks the bounded_random row {RANGE_KEY}')
    return {RANGE_KEY: dict(value=[int(x) for x in row['candidates']], unit=row['unit'], source_kind=row['source_kind'],
                            source_ref=f'{RANGE_KEY}: {row["source_ref"]} [{row["note"]}]', cultivar=row['cultivar'], uncertainty=row['uncertainty'])}


def seed_record(value: int, note: str) -> dict:
    return dict(value=int(value), unit='seed', source_kind='user_assumed', source_ref=f'{SEED_KEY}: {note}', cultivar='Micro-Tom_unfitted_hypothesis',
                uncertainty='seed of a declared random-flower-count member; not a plant value')


# ---- pure-Python replica of the native draw (java.util.Random, String.hashCode) for the independent checkers
def java_hash(s: str) -> int:
    h = 0
    units = s.encode('utf-16-be')   # Java hashes UTF-16 code units
    for i in range(0, len(units), 2):
        h = (31 * h + ((units[i] << 8) | units[i + 1])) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h


class JavaRandom:
    def __init__(self, s: int):
        self.s = (s ^ _MULT) & _MASK

    def _next(self, bits: int) -> int:
        self.s = (self.s * _MULT + 0xB) & _MASK
        return self.s >> (48 - bits)

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) * (1.0 / (1 << 53))


def java_u(seed_value: int, truss_id: str, key: str = DRAW_KEY) -> float:
    r = JavaRandom(int(seed_value) + java_hash(f'{truss_id}:{key}')); r.next_double()
    return r.next_double()


def count_from_u(u: float, lo: int, hi: int) -> int:
    return lo + min(hi - lo, int(math.floor(u * (hi - lo + 1))))


def draw(seed_value: int, truss_id: str, lo: int, hi: int) -> tuple[float, int]:
    u = java_u(seed_value, truss_id); return u, count_from_u(u, lo, hi)


def transform_native(template: str, req: dict) -> str:
    """Inject the draw fragment and replace the two FLOWERS uses; disabled requests return the exact input bytes."""
    validate(req)
    if not enabled(req):
        return template
    lo, hi = bounds(req)
    code = NATIVE.read_text(encoding='utf-8').replace('@FPT_SEED@', f'{seed(req)}L').replace('@FPT_LO@', str(lo)).replace('@FPT_HI@', str(hi))
    require('@FPT_' not in code, 'unresolved flowers_per_truss token')

    def replace(old: str, new: str) -> None:
        nonlocal template
        require(template.count(old) == 1, f'flowers_per_truss insertion point changed ({template.count(old)} != 1): ' + old[:70])
        template = template.replace(old, new)

    replace('float[] shape=null;', 'float[] shape=null; int fptFlowers=0;')
    replace('@AXIS_CODE@', '@AXIS_CODE@\n' + code)
    # truss creation: one draw per truss born in the run, stored on the truss for the flower azimuth
    replace('for(int f=1;f<=FLOWERS;f++) { Organ flower=add(p,truss.idx,axis,11,leafOrders[controller],day,prefix+"Truss"+count+"/F"+f);flower.pair=f; }',
            'truss.fptFlowers=fptCount(truss.id);for(int f=1;f<=truss.fptFlowers;f++) { Organ flower=add(p,truss.idx,axis,11,leafOrders[controller],day,prefix+"Truss"+count+"/F"+f);flower.pair=f; }')
    replace('o.az+=o.pair*360.0/FLOWERS;', 'o.az+=o.pair*360.0/fptFlowersOf(o);')
    replace('edges.println("day,parent_id,child_id");', 'edges.println("day,parent_id,child_id");\n    fptDraws=new PrintWriter(new FileWriter("' + OUTPUT + '"));fptDraws.println("truss_id,u,n_flowers");')
    replace('edges.close();', 'edges.close();fptDraws.close();')
    return template


def validate_draws(path: Path, req: dict) -> dict:
    """Independent check of working/flower_draws.csv: one row per truss, u in [0, 1) equal to the Python replica, n = lo + floor(u (hi-lo+1)) in [lo, hi]."""
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    lo, hi = bounds(req); s = seed(req); seen = set(); counts = {}
    for r in rows:
        tid = r['truss_id']; require(tid not in seen, f'duplicate flower draw {tid}'); seen.add(tid)
        u = float(r['u']); n = int(r['n_flowers'])
        require(0 <= u < 1, f'u outside [0,1): {tid}')
        require(lo <= n <= hi, f'flower count outside [{lo},{hi}]: {tid} -> {n}')
        eu, en = draw(s, tid, lo, hi)
        require(math.isclose(u, eu, rel_tol=0, abs_tol=1e-15) and n == en, f'flower draw not reproducible from seed and truss id: {tid} ({u},{n}) != ({eu},{en})')
        counts[n] = counts.get(n, 0) + 1
    return dict(rows=len(rows), per_count={str(k): counts[k] for k in sorted(counts)}, range=[lo, hi], seed=s)


class Draws:
    """The run's recorded flower draws (working/flower_draws.csv) for the independent checkers. Inactive requests (or a missing working folder)
    return the request constant FLOWERS for every truss, so every checker keeps its old behaviour when the hook is off."""

    def __init__(self, req: dict, working: Path | None):
        self.req = req; self.on = enabled(req); self.default = int(req['parameters']['FLOWERS']['value']); self.counts = {}
        if self.on:
            path = Path(working) / OUTPUT if working is not None else None
            require(path is not None and path.exists(), 'flowers_per_truss on but working/' + OUTPUT + ' missing')
            for r in csv.DictReader(open(path, encoding='utf-8')):
                self.counts[r['truss_id']] = int(r['n_flowers'])

    def count(self, truss_id: str, imported: bool = False) -> int:
        if not self.on or imported: return self.default
        n = self.counts.get(truss_id); require(n is not None, f'flower draw missing: {truss_id}'); return n
