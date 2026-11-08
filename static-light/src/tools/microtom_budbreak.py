"""Session F (2026-10-14) — new side-shoot emergence (axillary bud outgrowth) as a stochastic event: LAYER planner + F5 estimators.

python -X utf8 -m tools.microtom_budbreak plan --run <committed pipeline run dir> --label base --members 4 --seed 11
python -X utf8 -m tools.microtom_budbreak estimate --side-registry <side_shoot_registry.csv> --truss-registry <truss_registry.csv>

Why a planner and not a simulation. The native engine cannot create an axis during a run: its per-axis controller arrays
(`clocks`, `produced`, `activeApex`, ... in `tools/vtc_development_native.rgg`) are sized once at prepare time from the declared
axes (`tools/fsp_axes.py::native_setup`), and the sympodial successor reuses its parent's controller. An emerged shoot therefore
cannot grow, take carbon or intercept light in this engine until hook H7 (LOG 2026-10-14) exists. This module does everything that
does not need the engine: it decides, per member and from declared scenario values, WHICH axils of a committed host trajectory
break and WHEN (the emergence plan the hook would consume), and it defines the F5 estimators that will replace the assumed values
once the side-shoot / truss registries hold observations.

Rules (work order F1/F2/F4):
- candidates are compound-leaf axils on every axis present in the host; the axil of an axis's END node (the node bearing its first
  truss, U6) is excluded because its continuation belongs to the existing sympodial mechanism (no double creation), and an axil that
  already bears a declared axis is excluded;
- eligibility `after_parent_axis_end` (default, apical dominance released at the parent's first truss) or `at_node_formation`
  (comparison only); emergence = the day the cumulative thermal time since eligibility reaches `BUD_BREAK_DELAY_DD` (the `axis_shoot`
  day: two leaves visible); one Bernoulli draw per bud per member with `BUD_BREAK_PROB`, seeded and recorded;
- position `all_nodes` (default); `upper_n_plus_basal` needs an `n` nobody has measured -> unavailable, never guessed;
- new axis: `order` = parent + 1, `origin = axillary`, `parent_axis`, `parent_rank` = rank of the subtending leaf, dotted id (`S<k>`,
  `S1.<k>`, A6), `provenance = model_generated`, `emergence_rule = BUD_BREAK_* scenario label`, forecast id in the `F:` namespace;
- no carbon gate on emergence and no cap on the number of new axes (`MAX_NEW_AXES = none`); order >= 2 needs leaves on the new axes,
  i.e. the engine, so it is `unavailable` here;
- every value is `user_assumed` with no basis (F2): a plan is never a Micro-Tom property and never mixed into observation records.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from tools.microtom_environment import thermal_time_per_day
from tools.microtom_forecast_io import now_iso
from tools.vtc_growth_io import ROOT, ContractError, fresh_build, read_json, require, rows, write_csv, write_json

ELIGIBILITY_OPTIONS = ('after_parent_axis_end', 'at_node_formation')
POSITION_OPTIONS = ('all_nodes', 'upper_n_plus_basal', 'upper_weighted')
UPPER_WEIGHT_CLIP = 0.95  # F4 Stage 1: per-node probability is clipped here; the realised mean is reported
DEFAULTS = dict(eligibility='after_parent_axis_end', position='all_nodes')
NO_CAP = 'none'
STATEMENT = ('New side-shoot emergence is generated from ASSUMED probabilities and delays with no Micro-Tom basis (user decision '
             '2026-10-14, F2); it is not a property of Micro-Tom. Planned emergences are not simulated growth: the engine cannot create '
             'an axis during a run until hook H7 exists.')
PLAN_COLUMNS = ('scenario_id', 'member_index', 'seed', 'plant_instance_id', 'parent_axis', 'parent_order', 'parent_rank', 'subtending_leaf_id',
                'candidate', 'exclusion_reason', 'eligible_day', 'eligibility', 'delay_dd', 'emergence_day', 'probability', 'draw_u', 'emerged',
                'new_axis_id', 'order', 'origin', 'forecast_axis_id', 'provenance', 'emergence_rule', 'position', 'status',
                'position_weight', 'draw_key')
TABLE1_COLUMNS = ('scenario_id', 'member_index', 'plant_instance_id', 'day', 'host_axes', 'new_axes_order_1', 'new_axes_order_2_plus', 'axes_total',
                  'host_compound_leaves', 'host_leaflets', 'new_axis_leaves', 'new_axis_leaflets', 'source')
GROUPS = ('tray', 'pruned_sep')


# ---------------------------------------------------------------------------------------------------------------
# host trajectory (committed checkpoints only)
# ---------------------------------------------------------------------------------------------------------------

def _logical_axis(state: dict, axis_mode: bool) -> str:
    """fsp-axes-1 state rows carry logical M/S<k>; a legacy run has one controller per plant = the main stem."""
    return state['axis_id'] if axis_mode else 'M'


def host_trajectory(run_dir: Path, days: int) -> dict:
    """Per plant: compound leaves (axis, rank, birth day), declared axes, first-truss (= axis end, U6) day and rank per axis."""
    run_dir = Path(run_dir)
    scen = read_json(run_dir / 'scenario.json')
    axis_mode = bool(scen.get('axis_contract'))
    axil_nodes = declared_axil_node_indices(run_dir, scen) if axis_mode else {}
    states = {0: rows(run_dir / 'native_inputs' / 'initial_state.csv')}
    for k in range(1, days + 1):
        p = run_dir / 'committed' / f'step_{k:04d}' / 'state.csv'
        require(p.is_file(), f'committed step {k} missing in {run_dir}')
        states[k] = rows(p)
    env = scen['environment']
    plants: dict[str, dict] = {}
    for d in range(0, days + 1):
        for s in states[d]:
            pid = s['plant_instance_id']
            pl = plants.setdefault(pid, dict(compounds={}, axes=set(), declared_side=set(), first_truss={}, leaflets_by_day={}, compounds_by_day={}))
            ax = _logical_axis(s, axis_mode)
            pl['axes'].add(ax)
            if s['class'] == 'CompoundLeaf' and s['organ_id'] not in pl['compounds']:
                birth = int(s['birth_day']) if s.get('birth_day') not in (None, '', 'unavailable') else d
                successor = (not axis_mode) and not s['axis_id'].endswith('/A0')  # legacy host: leaf on an engine sympodial successor unit
                pl['compounds'][s['organ_id']] = dict(organ_id=s['organ_id'], axis=ax, rank=int(s['true_leaf_order']), birth_day=max(0, birth) if d > 0 else 0,
                                                      successor_unit=successor)
            if s['class'] == 'TrussAxis':
                birth = int(s['birth_day']) if s.get('birth_day') not in (None, '', 'unavailable') else d
                birth = 0 if d == 0 else max(0, birth)
                prev = pl['first_truss'].get(ax)
                if prev is None or birth < prev['day']:
                    pl['first_truss'][ax] = dict(day=birth, rank=int(s['true_leaf_order']), organ_id=s['organ_id'])
            if axis_mode and ax != 'M' and s.get('parent_rank') not in (None, '', 'unavailable'):
                # an occupied axil on M, as a NODE index (candidates below are compared by true_leaf_order = node index). A declared side axis
                # carries the producer's MTG rank in parent_rank (fsp-axes-1); revision-9 packages give its node index explicitly.
                pl['declared_side'].add(('M', axil_nodes.get((pid, ax), int(s['parent_rank']))))
        for pid, pl in plants.items():
            ps = [s for s in states[d] if s['plant_instance_id'] == pid]
            pl['leaflets_by_day'][d] = sum(s['class'] == 'Leaflet' for s in ps)
            pl['compounds_by_day'][d] = sum(s['class'] == 'CompoundLeaf' for s in ps)
    return dict(run_dir=str(run_dir), axis_mode=axis_mode, days=days, plants=plants,
                thermal_per_day=[thermal_time_per_day(e['temp_day_c'], e['temp_night_c'], e['light_hours'], float(scen['parameters']['BASE_C']['value'])) for e in env[:days]],
                parameters={k: v['value'] for k, v in scen['parameters'].items() if k in ('FIRST_LEAVES', 'CADENCE', 'PHYLLOCHRON_DD', 'BASE_C')})


def retime_host(host: dict, axis_end_day: dict, source: str) -> dict:
    """F4 addendum 4: a copy of `host` whose parent-axis ends (first-truss day) are replaced by those read from a run with another phyllochron.
    axis_end_day: {(plant_instance_id, axis): day, or None when that axis did not end within the horizon}. Only the day changes - the truss rank,
    candidates and thermal series stay - so plan_emergence() on the copy keeps every bud's draw, probability and weight (common random numbers)."""
    out = copy.deepcopy(host)
    for (pid, ax), day in axis_end_day.items():
        pl = out['plants'].get(pid)
        require(pl is not None and ax in pl['first_truss'], f'no first truss of {pid}/{ax} in the host to retime')
        if day is None:
            # keep the rank (the end node stays excluded and upper_weighted weights unchanged); a missing day reads 'never_within_horizon'
            pl['first_truss'][ax] = dict(pl['first_truss'][ax], day=None)
            continue
        pl['first_truss'][ax] = dict(pl['first_truss'][ax], day=int(day))
    out['retimed'] = dict(source=source, axis_end_day={f'{p}/{a}': d for (p, a), d in sorted(axis_end_day.items())})
    return out


def declared_axil_node_indices(run_dir: Path, scen: dict) -> dict:
    """{(plant_instance_id, axis_id): parent_node_index} of declared side axes, from the run's own P1 copy (`<run>/p1/axes.csv`). Pre-A7
    packages (revision < 9) return {} - their parent_rank already is the node index. A revision-9 package without a node index for a declared
    side axis is refused: a U5 rank is never used as a node index."""
    from tools.microtom_axes import rank_basis, RANK_BASIS_PRE_A7
    p1 = Path(run_dir) / 'p1'
    if not (p1 / 'axes.csv').is_file() or rank_basis(p1) == RANK_BASIS_PRE_A7:
        return {}
    caps = scen.get('captures_by_instance', {})
    out = {}
    for r in rows(p1 / 'axes.csv'):
        if caps.get(r['plant_instance_id']) != r['capture_id'] or r['axis_id'] == 'M':
            continue
        n = r.get('parent_node_index', 'unavailable')
        require(n not in ('', 'unavailable', None), f"{r['plant_instance_id']}/{r['axis_id']}: revision-9 axes.csv without parent_node_index; a U5 rank is never used as a node index")
        out[(r['plant_instance_id'], r['axis_id'])] = int(n)
    return out


def emergence_day(eligible_day: int, delay_dd: float, thermal_per_day: list[float]) -> int | None:
    """First day whose cumulative thermal time since `eligible_day` reaches `delay_dd`; None beyond the host horizon."""
    acc = 0.0
    for d in range(eligible_day, len(thermal_per_day)):
        acc += thermal_per_day[d]
        if acc >= delay_dd - 1e-9:
            return d + 1
    return None


def _axis_order(axis: str) -> int:
    return 0 if axis == 'M' else axis.count('.') + 1


def _child_id(parent: str, k: int) -> str:
    return f'S{k}' if parent == 'M' else f'{parent}.{k}'


def plan_emergence(host: dict, params: dict, label: str, member: int, seed: int, run_id: str = 'host') -> list[dict]:
    """One member's emergence plan: one row per candidate axil (emerged or not) and one row per excluded axil (with the reason)."""
    prob, delay = float(params['prob']), float(params['delay_dd'])
    eligibility, position = params.get('eligibility', DEFAULTS['eligibility']), params.get('position', DEFAULTS['position'])
    require(0.0 <= prob <= 1.0, 'BUD_BREAK_PROB must be a probability')
    require(delay >= 0, 'BUD_BREAK_DELAY_DD must be >= 0')
    require(eligibility in ELIGIBILITY_OPTIONS, f'eligibility must be one of {ELIGIBILITY_OPTIONS}')
    require(position in POSITION_OPTIONS, f'position must be one of {POSITION_OPTIONS}')
    rule = f'BUD_BREAK_*:{label}'
    # F4: common random numbers - cells that share a draw key share every bud's uniform draw (default: the label, as in Session F)
    draw_key = params.get('draw_key', label)
    out = []
    for pid, pl in sorted(host['plants'].items()):
        if position == 'upper_n_plus_basal':
            out.append(dict(scenario_id=label, member_index=member, seed=seed, plant_instance_id=pid, parent_axis='*', parent_order='', parent_rank='',
                            subtending_leaf_id='', candidate=False, exclusion_reason='position option upper_n_plus_basal needs n (unmeasured): unavailable',
                            eligible_day='', eligibility=eligibility, delay_dd=delay, emergence_day='', probability=prob, draw_u='', emerged=False,
                            new_axis_id='', order='', origin='axillary', forecast_axis_id='', provenance='model_generated', emergence_rule=rule,
                            position=position, status='unavailable', position_weight='', draw_key=draw_key))
            continue
        used = defaultdict(int)  # next child number per parent axis (declared S axes keep their numbers)
        for ax in pl['axes']:
            if ax != 'M':
                parent = ax.rsplit('.', 1)[0] if '.' in ax else 'M'
                used[parent] = max(used[parent], int(ax.rsplit('.', 1)[-1].lstrip('S')))
        cands = sorted(pl['compounds'].values(), key=lambda c: (c['axis'], c['rank']))
        weights = upper_weights(pl, cands) if position == 'upper_weighted' else {}
        emerged_rows = []
        for c in cands:
            ax, rank = c['axis'], c['rank']
            end = pl['first_truss'].get(ax)
            base = dict(scenario_id=label, member_index=member, seed=seed, plant_instance_id=pid, parent_axis=ax, parent_order=_axis_order(ax), parent_rank=rank,
                        subtending_leaf_id=c['organ_id'], eligibility=eligibility, delay_dd=delay, probability=prob, origin='axillary',
                        provenance='model_generated', emergence_rule=rule, position=position, position_weight='', draw_key=draw_key)
            if end is not None and rank == end['rank']:
                out.append(dict(base, candidate=False, exclusion_reason='axis end node (first truss): continuation is the existing sympodial mechanism',
                                eligible_day='', emergence_day='', draw_u='', emerged=False, new_axis_id='', order='', forecast_axis_id='', status='excluded'))
                continue
            if c.get('successor_unit'):
                out.append(dict(base, candidate=False, exclusion_reason='leaf on an engine sympodial successor unit: under A6 a post_first_truss axis that this legacy host does not represent -> unavailable',
                                eligible_day='', emergence_day='', draw_u='', emerged=False, new_axis_id='', order='', forecast_axis_id='', status='unavailable'))
                continue
            if (ax, rank) in pl['declared_side']:
                out.append(dict(base, candidate=False, exclusion_reason='axil already bears a declared axis', eligible_day='', emergence_day='', draw_u='',
                                emerged=False, new_axis_id='', order='', forecast_axis_id='', status='excluded'))
                continue
            if eligibility == 'after_parent_axis_end':
                eligible = end['day'] if end is not None else None
            else:
                eligible = c['birth_day']
            u = random.Random(f'{seed}:{member}:budbreak:{draw_key}:{pid}:{ax}:{rank}').random()
            if position == 'upper_weighted':
                w, mean_w = weights[(ax, rank)]
                p_node = min(UPPER_WEIGHT_CLIP, prob * w / mean_w)
                base = dict(base, probability=round(p_node, 6), position_weight=round(w, 6))
            else:
                p_node = prob
                base = dict(base, position_weight=1.0)
            if eligible is None:
                out.append(dict(base, candidate=True, exclusion_reason='', eligible_day='never_within_horizon (parent axis did not end)', emergence_day='',
                                draw_u=round(u, 6), emerged=False, new_axis_id='', order='', forecast_axis_id='', status='not_eligible_within_horizon'))
                continue
            day = emergence_day(max(eligible, c['birth_day']), delay, host['thermal_per_day'])
            breaks = u < p_node
            row = dict(base, candidate=True, exclusion_reason='', eligible_day=max(eligible, c['birth_day']), emergence_day=day if day is not None else 'beyond_horizon',
                       draw_u=round(u, 6), emerged=bool(breaks and day is not None), new_axis_id='', order='', forecast_axis_id='',
                       status='emerged' if (breaks and day is not None) else ('breaks_beyond_horizon' if breaks else 'stays_dormant'))
            out.append(row)
            if row['emerged']:
                emerged_rows.append(row)
        # A6 numbering: per parent axis in emergence order, lower node first on the same day
        for row in sorted(emerged_rows, key=lambda r: (r['parent_axis'], r['emergence_day'], r['parent_rank'])):
            used[row['parent_axis']] += 1
            nid = _child_id(row['parent_axis'], used[row['parent_axis']])
            row.update(new_axis_id=nid, order=row['parent_order'] + 1, forecast_axis_id=f'F:{run_id}:{label}:m{member:02d}:{pid}/{nid}/Axis')
    return out


def upper_weights(pl: dict, cands: list[dict]) -> dict:
    """F4 Stage 1 `upper_weighted`: per parent axis, the n eligible nodes ranked 1..n from the base get w_r = 0.5 + (r-1)/(n-1).
    `r` (and every `rank` / `parent_rank` in this planner) is the engine NODE INDEX (`true_leaf_order`), not an MTG rank: under U5 the
    first two main-stem leaves share MTG rank 1, so an MTG rank is converted to a node index by the P1 producer (`node_index_from_base`)
    before it reaches the engine (microtom_axes.axis_attachment_node_index).
    (0.5x at the lowest, 1.5x at the highest; mean 1 by construction); n = 1 gives w = 1. Direction from Mol Hortic 2022 (WT Micro-Tom
    bud outgrowth increases from basal to apical nodes, T2 qualitative): the slope is user_assumed."""
    by_axis = defaultdict(list)
    for c in cands:
        end = pl['first_truss'].get(c['axis'])
        if (end is not None and c['rank'] == end['rank']) or c.get('successor_unit') or (c['axis'], c['rank']) in pl['declared_side']:
            continue
        by_axis[c['axis']].append(c['rank'])
    out = {}
    for ax, ranks in by_axis.items():
        ranks = sorted(ranks); n = len(ranks)
        ws = [0.5 + (i / (n - 1)) if n > 1 else 1.0 for i in range(n)]
        mean_w = sum(ws) / n
        for r, w in zip(ranks, ws):
            out[(ax, r)] = (w, mean_w)
    return out


def table1(host: dict, plan: list[dict], days_report: list[int]) -> list[dict]:
    """Axes / leaves / leaflets / new shoots by order per plant and day; leaves on NEW axes need the engine -> unavailable."""
    out = []
    keys = sorted({(r['scenario_id'], r['member_index']) for r in plan})
    for label, member in keys:
        for pid, pl in sorted(host['plants'].items()):
            for d in days_report:
                if d > host['days']:
                    continue
                em = [r for r in plan if r['scenario_id'] == label and r['member_index'] == member and r['plant_instance_id'] == pid and r['emerged']
                      and int(r['emergence_day']) <= d]
                host_axes = len(pl['axes'])
                n1 = sum(1 for r in em if r['order'] == 1)
                out.append(dict(scenario_id=label, member_index=member, plant_instance_id=pid, day=d, host_axes=host_axes, new_axes_order_1=n1,
                                new_axes_order_2_plus='unavailable (needs leaves on new axes: engine hook H7)', axes_total=host_axes + len(em),
                                host_compound_leaves=pl['compounds_by_day'][d], host_leaflets=pl['leaflets_by_day'][d],
                                new_axis_leaves='unavailable (engine hook H7)' if em else 0, new_axis_leaflets='unavailable (engine hook H7)' if em else 0,
                                source=f'host committed checkpoints {Path(host["run_dir"]).name}; planned emergences (not simulated)'))
    return out


def run_plan(run_dir: Path, scenarios: dict[str, dict], members: int, seed: int, days: int, out: Path, days_report: list[int]) -> Path:
    """Plan every scenario x member on one host trajectory; write plan.csv, table1.csv and a manifest (not a forecast)."""
    out = fresh_build(Path(out))
    host = host_trajectory(run_dir, days)
    plan = []
    for label, params in scenarios.items():
        for m in range(members):
            plan += plan_emergence(host, params, label, m, seed, out.name)
    write_csv(out / 'plan.csv', [{k: r[k] for k in PLAN_COLUMNS} for r in plan])
    t1 = table1(host, plan, days_report)
    write_csv(out / 'table1.csv', [{k: r[k] for k in TABLE1_COLUMNS} for r in t1])
    counts = defaultdict(list)
    for (label, m), _ in {(r['scenario_id'], r['member_index']): 1 for r in plan}.items():
        for pid in host['plants']:
            counts[(label, pid)].append(sum(1 for r in plan if r['scenario_id'] == label and r['member_index'] == m and r['plant_instance_id'] == pid and r['emerged']))
    realised = {}
    for label in scenarios:
        ps = [float(r['probability']) for r in plan if r['scenario_id'] == label and r['candidate'] and r['member_index'] == 0]
        realised[label] = round(sum(ps) / len(ps), 6) if ps else 'unavailable'
    write_json(out / 'manifest.json', dict(realised_mean_probability_per_candidate=realised,
        draw_keys={label: s.get('draw_key', label) for label, s in scenarios.items()},
        schema='microtom-budbreak-plan-1', created_at=now_iso(), host_run=str(Path(run_dir)), host_days=days, host_axis_mode=host['axis_mode'],
        host_parameters=host['parameters'], scenarios=scenarios, members=members, seed=seed, statement=STATEMENT,
        emergence_status='planned_not_simulated (engine hook H7 absent: no growth, carbon or light for new axes)',
        new_axes_per_plant={f'{label}|{pid}': v for (label, pid), v in counts.items()}, max_new_axes=NO_CAP,
        carbon_gate='none (work order F2: no carbon rule on emergence)', pruning_response='not_implemented (U1, out of scope)',
        provenance='model_generated', origin='axillary', id_namespace='F:<plan>:<scenario>:m<NN>:<plant>/<axis>/Axis',
        reuse='a plan is not an observation and never enters the side-shoot registry'))
    return out


# ---------------------------------------------------------------------------------------------------------------
# F5 — estimators from the observation registries (computed only when rows exist)
# ---------------------------------------------------------------------------------------------------------------

def _without_examples(path: Path) -> tuple[Path, int]:
    """Copy of a registry without the template's EXAMPLE rows (they are instructions, not observations)."""
    with Path(path).open(encoding='utf-8', newline='') as fh:
        reader = csv.DictReader(fh); header = reader.fieldnames; keep = [r for r in reader if not (r.get('note') or '').startswith('EXAMPLE ROW')]
    tmp = Path(tempfile.mkdtemp()) / Path(path).name
    with tmp.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=header); w.writeheader(); w.writerows(keep)
    return tmp, len(keep)


def group_of(plant_id: str) -> str:
    """Tray experimental group vs the separately grown pruned group (`sep_` ids); never pooled."""
    return 'pruned_sep' if plant_id.startswith('sep_') else 'tray'


def estimate(side_registry: Path, truss_registry: Path, thermal_per_day: float | None = None) -> dict:
    """The four F5 quantities per group, or `unavailable` with the reason. Reads the registries through their own validators."""
    from tomato3d.state import axes as sa
    from tomato3d.state import trusses as st
    side_path, n_side = _without_examples(side_registry)
    truss_path, n_truss = _without_examples(truss_registry)
    side = sa.read_registry(side_path) if n_side else []
    truss = st.read_registry(truss_path) if n_truss else []
    tpd = thermal_per_day if thermal_per_day is not None else thermal_time_per_day(26, 22, 14, 5.7)
    out = {}
    for g in GROUPS:
        sg = [e for e in side if group_of(e.plant_id) == g and e.origin == 'axillary']
        tg = [e for e in truss if group_of(e.plant_id) == g]
        res = {}
        # BUD_BREAK_PROB: axillary shoots per eligible node of their parent axis (eligible = nodes below the parent's first-truss node)
        ends = {(e.plant_id, e.axis): e for e in tg if e.is_first_sympodial_branch or (e.a6 and e.node_rank is not None)}
        parents = {(e.plant_id, e.parent_axis) for e in sg}
        nodes = {k: ends[k].node_rank - 1 for k in parents if k in ends and ends[k].node_rank}
        if sg and nodes and len(nodes) == len(parents):
            res['BUD_BREAK_PROB'] = dict(value=round(len(sg) / sum(nodes.values()), 4), n_shoots=len(sg), n_eligible_nodes=sum(nodes.values()), source_kind='measured_derived')
        else:
            res['BUD_BREAK_PROB'] = dict(value='unavailable', reason='needs axillary shoots AND the first-truss node rank of every parent axis' if sg else 'no axillary shoot rows')
        # BUD_BREAK_DELAY_DD: parent first-truss observation -> child first observation, days x set-point thermal time
        delays, before_end = [], 0
        for e in sg:
            end = ends.get((e.plant_id, e.parent_axis))
            if end is None:
                continue
            dd = (e.first_observed - end.first_observed).days
            if dd < 0:
                before_end += 1
            else:
                delays.append(dd * tpd)
        res['BUD_BREAK_DELAY_DD'] = (dict(value=round(sum(delays) / len(delays), 1), n=len(delays), observed_before_parent_end=before_end,
                                          thermal_basis='days x set-point thermal time (user_stated climate), not a logged temperature', source_kind='measured_derived')
                                     if delays else dict(value='unavailable', reason='needs axillary shoots and the first truss of their parent axis, both dated',
                                                         observed_before_parent_end=before_end))
        # BUD_BREAK_POSITION: where on the parent axis shoots appear (rank, relative to the parent's end node; basal = rank <= 1)
        pos = [(e.parent_rank, nodes.get((e.plant_id, e.parent_axis))) for e in sg if e.parent_rank is not None]
        res['BUD_BREAK_POSITION'] = (dict(value=dict(parent_ranks=sorted(p for p, _ in pos), basal_rank_le_1=sum(1 for p, _ in pos if p <= 1),
                                                     relative_position=[round(p / n, 3) for p, n in pos if n]), n=len(pos), source_kind='measured')
                                     if pos else dict(value='unavailable', reason='no axillary shoot with a recorded parent_rank'))
        # AXIS_LEAVES_TO_TRUSS_NEW_AXIS: ranks before the first truss on order >= 1 axes (true leaves = ranks + 1 reported beside)
        sub = [e for e in tg if s_order(e.axis) >= 1 and e.ranks_before_truss is not None]
        res['AXIS_LEAVES_TO_TRUSS_NEW_AXIS'] = (dict(value=dict(ranks=sorted(e.ranks_before_truss for e in sub), true_leaves=sorted(e.leaves_before_truss_true for e in sub)),
                                                     n=len(sub), source_kind='measured')
                                                if sub else dict(value='unavailable', reason='no order >= 1 truss row with ranks_before_truss'))
        out[g] = res
    return dict(groups=out, pooled=False, side_rows=n_side, truss_rows=n_truss, side_registry=str(side_registry), truss_registry=str(truss_registry),
                rule='tray and pruned (sep_) groups are computed separately and never pooled; template EXAMPLE rows are ignored')


def s_order(axis: str) -> int:
    return _axis_order(axis)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('plan', help='plan emergences on one committed host run (all registry BUD_BREAK scenarios)')
    p.add_argument('--run', type=Path, required=True); p.add_argument('--days', type=int, required=True)
    p.add_argument('--members', type=int, default=4); p.add_argument('--seed', type=int, default=11)
    p.add_argument('--eligibility', choices=ELIGIBILITY_OPTIONS, default=None); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--report-days', type=int, nargs='+', default=[0, 10, 18, 24, 30])
    p = sub.add_parser('estimate', help='F5: compute the four quantities from the observation registries (unavailable without rows)')
    p.add_argument('--side-registry', type=Path, required=True); p.add_argument('--truss-registry', type=Path, required=True)
    a = parser.parse_args(argv)
    try:
        if a.op == 'plan':
            scen = scenarios_from_registry(eligibility=a.eligibility)
            folder = run_plan(a.run, scen, a.members, a.seed, a.days, a.out, a.report_days)
            print(json.dumps(dict(plan=str(folder), new_axes_per_plant=read_json(folder / 'manifest.json')['new_axes_per_plant']), indent=2))
        else:
            print(json.dumps(estimate(a.side_registry, a.truss_registry), indent=2, default=str))
    except (ContractError, ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


def scenarios_from_registry(registry_path: Path | None = None, eligibility: str | None = None) -> dict[str, dict]:
    """The declared BUD_BREAK scenario set: label L applies each row's L value jointly (labels name the declared values, not an effect direction)."""
    from tools import microtom_parameters as mp
    reg = mp.load_registry(registry_path) if registry_path else mp.load_registry()
    by = {r['key']: r for r in reg}
    require('BUD_BREAK_PROB' in by and 'BUD_BREAK_DELAY_DD' in by, 'registry lacks the BUD_BREAK rows')
    prob, delay = by['BUD_BREAK_PROB']['candidates'], by['BUD_BREAK_DELAY_DD']['candidates']
    require(list(prob) == list(delay), 'BUD_BREAK_PROB and BUD_BREAK_DELAY_DD must declare the same labels')
    elig = eligibility or by['BUD_BREAK_ELIGIBILITY']['value']
    pos = by['BUD_BREAK_POSITION']['value']
    return {label: dict(prob=prob[label], delay_dd=delay[label], eligibility=elig, position=pos) for label in prob}


if __name__ == '__main__':
    main()
