"""Axis-, truss- and segment-aware input/output helpers for the Micro-Tom forecast layer (P3 V3, 2026-10-12).

Reads a P1 package's `axes.csv` / `trusses.csv` and builds the fsp-axes-1 request records the engine consumes
(`tools.fsp_axes.validate_records` / `select_axes`), keeping the two `parent_rank` meanings and the two segment
counters apart:

- `axil_parent_rank`  — on an AXIS row: the main-stem leaf rank at whose axil the side shoot attaches (MTG A4 §15.2).
- `truss_above_rank`  — on a TRUSS row: the rank, on the truss's own bearing axis, of the leaf directly below it (A5 §16.4).
- `organ_segment_index` — an organ's IMMUTABLE segment (engine `organ_inventory.csv` / `state.csv segment_index`).
- `axis_segment_index`  — an axis's CURRENT counter, 1 + trusses so far (engine `axis_inventory.csv` / `apices.csv axis_segment_index`).

`segment_index` is never derived here: the producer's value is copied, `unavailable` stays `unavailable`.
Truss geometry is a declared supplement (zero-extent bookkeeping points that cast no shadow), never an observed location.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.vtc_growth_io import ContractError, read_json, require, rows

AXIS_CONTRACT = 'fsp-axes-1'
AXIS_REQUIRED = ('plant_instance_id', 'capture_id', 'axis_id', 'axis_vertex_id', 'order', 'parent_axis_id', 'parent_node_id', 'parent_rank',
                 'stage', 'source_kind', 'source_ref', 'segment_index', 'truss_count')
TRUSS_REQUIRED = ('plant_instance_id', 'capture_id', 'truss_vertex_id', 'axis_id', 'parent_rank', 'segment_index', 'source_kind', 'source_ref')
UNAVAILABLE = 'unavailable'
ZERO_EXTENT_LABEL = 'zero_extent_bookkeeping_point_casts_no_shadow'
TRUSS_CLASSES = ('Truss', 'TrussAxis')


def has_axes(package: Path) -> bool:
    return (Path(package) / 'axes.csv').is_file()


NODE_INDEX_REVISION = 9  # p1_state_v1 revision that carries node_index_from_base / parent_node_index / above_node_index (A7 follow-through)
RANK_BASIS_NODE_INDEX = 'node_index_from_base'
RANK_BASIS_PRE_A7 = 'node_index_pre_A7'


def package_revision(package: Path) -> int:
    manifest = Path(package) / 'p1_manifest.json'
    rev = read_json(manifest).get('schema_revision', 0) if manifest.is_file() else 0
    return int(rev) if str(rev).isdigit() else 0


def rank_basis(package: Path) -> str:
    """How this consumer reads a producer `rank`: a node index (pre-A7 packages, revision < 9) or an MTG rank beside an explicit
    node index (revision >= 9). Never a +1 guess: a revision-9 package without the node index refuses (see leaf_node_indices)."""
    return RANK_BASIS_NODE_INDEX if package_revision(package) >= NODE_INDEX_REVISION else RANK_BASIS_PRE_A7


def leaf_node_indices(leaves: list[dict], basis: str) -> list[int]:
    """Per-axis node indices of CompoundLeaf vertices. Pre-A7: the producer rank IS the node index. Revision 9: `node_index_from_base`
    is required for every ranked leaf (an `unavailable` node index is refused, not guessed)."""
    if basis == RANK_BASIS_PRE_A7:
        return [int(v['rank']) for v in leaves if v.get('rank') not in ('', UNAVAILABLE, None)]
    out = []
    for v in leaves:
        if v.get('rank') in ('', UNAVAILABLE, None):
            continue
        n = v.get('node_index_from_base', UNAVAILABLE)
        require(n not in ('', UNAVAILABLE, None), f"{v.get('vertex_id')}: rank {v['rank']} without node_index_from_base (revision-9 package); a node index is never guessed from a U5 rank")
        out.append(int(n))
    return out


def axis_leaf_order(leaves: list[dict], axis_row: dict, basis: str) -> int:
    """Highest node index bearing a leaf on the axis (engine `leaf_order`). Revision 9: the producer's `node_count_incl_scars` when it is a
    number, else max(node_index_from_base). Pre-A7: max(rank) (the rank was a node index). Never max(rank) on a U5 package."""
    if basis != RANK_BASIS_PRE_A7:
        n = axis_row.get('node_count_incl_scars', UNAVAILABLE)
        if n not in ('', UNAVAILABLE, None):
            return int(n)
    idx = leaf_node_indices(leaves, basis)
    return max(idx) if idx else 0


def _int_or_unavailable(value: str) -> int | str:
    return UNAVAILABLE if value in ('', UNAVAILABLE, None) else int(value)


def read_axis_tables(package: Path, captures_by_instance: dict[str, str]) -> dict:
    """Selected-capture rows of axes.csv / trusses.csv with the two parent_rank meanings renamed apart.

    Axis rows get `row_kind = axis`, `axil_parent_rank`; truss rows get `row_kind = truss`, `truss_above_rank`.
    The ambiguous producer key `parent_rank` is removed from both so it cannot be read by accident.
    """
    package = Path(package)
    require(has_axes(package), f'{package.name} has no axes.csv (legacy single-axis package)')
    axes = []
    for r in rows(package / 'axes.csv'):
        if captures_by_instance.get(r['plant_instance_id']) != r['capture_id']:
            continue
        require(set(AXIS_REQUIRED) <= set(r), 'axes.csv lacks required columns')
        require('axis' not in r or r['axis'] == r['axis_id'], 'producer axis aliases disagree')
        require('axis_state' not in r or r['axis_state'] == r['stage'], 'producer axis-state aliases disagree')
        row = {k: v for k, v in r.items() if k != 'parent_rank'}
        row.update(row_kind='axis', axil_parent_rank=int(r['parent_rank']), segment_index=_int_or_unavailable(r['segment_index']),
                   truss_count=int(r['truss_count']), order=int(r['order']),
                   axil_parent_node_index=_int_or_unavailable(r.get('parent_node_index', UNAVAILABLE)))  # revision 9: the node index beside parent_rank
        axes.append(row)
    trusses = []
    if (package / 'trusses.csv').is_file():
        for r in rows(package / 'trusses.csv'):
            if captures_by_instance.get(r['plant_instance_id']) != r['capture_id']:
                continue
            require(set(TRUSS_REQUIRED) <= set(r), 'trusses.csv lacks required columns')
            row = {k: v for k, v in r.items() if k != 'parent_rank'}
            row.update(row_kind='truss', truss_above_rank=_int_or_unavailable(r['parent_rank']), segment_index=_int_or_unavailable(r['segment_index']),
                       truss_above_node_index=_int_or_unavailable(r.get('above_node_index', UNAVAILABLE)))
            trusses.append(row)
    require(bool(axes), 'no axis rows for the selected captures')
    return dict(axes=axes, trusses=trusses)


def axis_attachment_rank(row: dict) -> int:
    """The main-stem axil rank of a side AXIS. Refuses truss rows: a truss's parent_rank is its above-rank, not an attachment."""
    require(row.get('row_kind') == 'axis', 'axil_parent_rank is defined for axis rows only (a truss row carries truss_above_rank)')
    return int(row['axil_parent_rank'])


def axis_attachment_node_index(row: dict, basis: str) -> int:
    """The engine's parent index of a side AXIS = the NODE index of its axil on the parent axis (the engine compares it with
    `true_leaf_order`, a node index). Revision 9: `parent_node_index` (refused when unavailable); pre-A7: the rank was the node index."""
    require(row.get('row_kind') == 'axis', 'axil_parent_node_index is defined for axis rows only')
    if basis == RANK_BASIS_PRE_A7:
        return int(row['axil_parent_rank'])
    n = row.get('axil_parent_node_index', UNAVAILABLE)
    require(n not in ('', UNAVAILABLE, None), f"{row.get('axis_id')}: parent_rank {row['axil_parent_rank']} without parent_node_index (revision-9 package); never guessed from a U5 rank")
    return int(n)


def truss_above_rank(row: dict) -> int | str:
    """The rank on the bearing axis of the leaf directly below a TRUSS. Refuses axis rows."""
    require(row.get('row_kind') == 'truss', 'truss_above_rank is defined for truss rows only (an axis row carries axil_parent_rank)')
    return row['truss_above_rank']


def selected_vertices(package: Path, captures_by_instance: dict[str, str], state_source: str) -> list[dict]:
    """The MTG vertices the pipeline will select (truth table for fixture_truth, else the observed MTG)."""
    package = Path(package)
    if state_source == 'fixture_truth':
        caps = {r['capture_id']: r for r in rows(package / 'captures.csv') if r['plant_instance_id'] in captures_by_instance}
        out = []
        for pid, cid in captures_by_instance.items():
            source = next(r['source_plant_id'] for r in rows(package / 'captures.csv') if r['plant_instance_id'] == pid and r['capture_id'] == cid)
            out.extend(dict(v, plant_instance_id=pid) for v in rows(package / 'truth' / 'truth_mtg_vertices.csv')
                       if v['plant_instance_id'] == source and v['capture_id'] == cid)
        return out
    return [v for v in rows(package / 'mtg_vertices.csv') if captures_by_instance.get(v['plant_instance_id']) == v['snapshot_capture_id']]


def build_axis_records(package: Path, captures_by_instance: dict[str, str], vertices: list[dict], axis_parameters: dict[str, dict],
                       hidden: dict, thermal: dict) -> tuple[list[dict], dict]:
    """fsp-axes-1 axis records from the producer tables; producer values verbatim, hidden counters labelled assumptions.

    `axis_parameters`: the registry's per-axis entries (SUBSTEM_SINK_SCALE, APPLY_DETERMINATE_STOP). `hidden`: declared
    side-axis hidden-state assumptions {stem_mass_gDM, stem_width_m, source_ref}. `thermal`: {GERMINATION_DD, PHYLLOCHRON_DD}
    for the explicit synthetic clock (same construction Codex used for the actual-P1 run; an assumption, not an age record).
    Returns (records, provenance notes).
    """
    tables = read_axis_tables(package, captures_by_instance)
    basis = rank_basis(package)  # A7 follow-through: ranks are node indices only in pre-revision-9 packages
    leaves_by_axis: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in vertices:
        if v['class'] == 'CompoundLeaf' and v.get('axis') not in ('', UNAVAILABLE, None):
            leaves_by_axis[(v['plant_instance_id'], v['axis'])].append(v)
    records, notes = [], {}
    for a in tables['axes']:
        pid, aid = a['plant_instance_id'], a['axis_id']
        main = aid == 'M'
        leaves = sorted(leaves_by_axis.get((pid, aid), []), key=lambda v: int(v['rank']) if v['rank'] not in ('', UNAVAILABLE) else 0)
        ranked = leaf_node_indices(leaves, basis)   # node indices (pre-A7: = producer ranks); never max(rank) on a U5 package
        leaf_order = axis_leaf_order(leaves, a, basis)
        seg = a['segment_index']
        axis_leaf_count = sum(1 for v in leaves if _int_or_unavailable(v.get('segment_index', UNAVAILABLE)) == seg) if seg != UNAVAILABLE else len(ranked)
        produced = leaf_order + a['truss_count']
        params = {k: dict(v) for k, v in axis_parameters.items()}
        if main:
            params['SUBSTEM_SINK_SCALE'] = dict(params['SUBSTEM_SINK_SCALE'], value=1.0, source_ref=params['SUBSTEM_SINK_SCALE']['source_ref'] + ' [M/root reference = 1 by definition]')
        record = dict(plant_instance_id=pid, axis_id=aid, axis_vertex_id=a['axis_vertex_id'], order=a['order'],
                      parent_axis_id='' if main else a['parent_axis_id'], parent_node_id='' if main else a['parent_node_id'],
                      parent_rank=0 if main else axis_attachment_rank(a),  # fsp-axes-1 contract: the producer's AXIL MTG rank (fsp_axes checks it against axes.csv and the parent vertex rank; natively a label). The axil NODE index is in the notes.
                      stage=a['stage'], source_kind='synthetic_assumed' if a['source_kind'] in ('fixture_declared', 'derived', 'synthetic_truth') else a['source_kind'],
                      source_ref=f'P1 axes.csv ({a["source_kind"]}: {a["source_ref"]}); registration/hidden counters are consumer assumptions',
                      registration_leaf_ids=[] if main else [v['vertex_id'] for v in leaves[:2]],
                      parameters=params,
                      initialization=dict(thermal_dd=round(thermal['GERMINATION_DD'] + thermal['PHYLLOCHRON_DD'] * (produced + .1), 6), produced=produced,
                                          leaf_order=leaf_order, truss_count=a['truss_count'], axis_leaf_count=axis_leaf_count, apex_stopped=False,
                                          stem_mass_gDM=hidden['stem_mass_gDM'], stem_width_m=hidden['stem_width_m'],
                                          source_kind='user_assumed',
                                          source_ref=f'{hidden["source_ref"]}; clock = GERMINATION_DD + PHYLLOCHRON_DD x (leaf_order + truss_count + 0.1); apex_stopped=false supplied as an assumption, not inferred from leaves'))
        if seg != UNAVAILABLE:
            record['segment_index'] = seg  # producer value copied; never derived
        records.append(record)
        notes[f'{pid}/{aid}'] = dict(axil_parent_rank=None if main else axis_attachment_rank(a), axil_parent_node_index=None if main else axis_attachment_node_index(a, basis),
                                     producer_segment_index=seg, producer_truss_count=a['truss_count'],
                                     leaf_node_indices=ranked, leaf_order=leaf_order, rank_basis=basis, registration_leaf_ids=record['registration_leaf_ids'])
    notes['rank_basis'] = basis
    notes['trusses'] = [dict(truss_vertex_id=t['truss_vertex_id'], axis_id=t['axis_id'], truss_above_rank=truss_above_rank(t), segment_index=t['segment_index'],
                             source_kind=t['source_kind'], registry_ref=t.get('registry_ref', '')) for t in tables['trusses']]
    return records, notes


def truss_geometry_supplement(organ_initialization: dict, vertices: list[dict], thermal_dd: float = 300.0) -> list[str]:
    """Explicit hidden state + zero-extent geometry for every selected Truss vertex (the engine requires a supplement).

    Values are declared assumptions (`user_assumed`), identical to the fixture convention of the engine owner; a zero-extent
    point casts no shadow and is NOT an observed truss location. Returns the organ ids supplemented.
    """
    done = []
    for v in vertices:
        if v['class'] not in TRUSS_CLASSES:
            continue
        oid = f'{v["plant_instance_id"]}|{v["vertex_id"]}'
        entry = organ_initialization.get(oid, {})
        entry.update(mass_gDM=.001, max_mass_gDM=.002, thermal_age_dd=thermal_dd, true_leaf_order=0, width_m=0,
                     units={'mass': 'gDM', 'age': 'degree_day'},
                     provenance='user_assumed: explicit synthetic truss hidden state (mass/age not measured)',
                     base_xyz_m=[0.0, 0.0, 0.0], tip_xyz_m=[0.0, 0.0, 0.0], geometry_source_kind='user_assumed',
                     geometry_source_ref=f'{ZERO_EXTENT_LABEL}; not an observed truss location (A5: trusses carry no geometry)')
        organ_initialization[oid] = entry
        done.append(oid)
    return done


def axis_controls(records: list[dict]) -> dict[str, dict]:
    """Per-axis controller state for the eligibility check (`microtom_parameters.eligibility`)."""
    out = {}
    for a in records:
        h = a['initialization']
        can_truss = (not h['apex_stopped']) and a.get('segment_index', UNAVAILABLE) != UNAVAILABLE and a['parameters']['APPLY_DETERMINATE_STOP']['value'] is True
        out[f'{a["plant_instance_id"]}/{a["axis_id"]}'] = dict(thermal_dd=h['thermal_dd'], produced=h['produced'], leaf_order=h['leaf_order'],
                                                             truss_count=h['truss_count'], axis_leaf_count=h['axis_leaf_count'], stopped=h['apex_stopped'],
                                                             is_substem=a['stage'] == 'substem', can_truss=can_truss, axis_id=a['axis_id'])
    return out


def committed_axis_inventory(run_dir: Path, boundary: int) -> dict[tuple[str, str], dict]:
    """`axis_inventory.csv` of one committed checkpoint keyed by (plant, axis): current axis_segment_index, truss_rule_status, ..."""
    path = Path(run_dir) / 'committed' / f'step_{boundary:04d}' / 'axis_inventory.csv'
    if not path.is_file():
        return {}
    return {(r['plant_instance_id'], r['axis_id']): r for r in rows(path)}


def axis_sums_check(run_dir: Path) -> dict:
    """Per committed step: summed axis_flux columns vs the plant carbon ledger (gross, maintenance, structural, growth_resp)."""
    run_dir = Path(run_dir)
    steps = sorted((run_dir / 'committed').glob('step_*'))
    worst = 0.0; checked = 0
    for step in steps:
        flux_path, ledger_path = step / 'axis_flux.csv', step / 'carbon_ledger.csv'
        if not (flux_path.is_file() and ledger_path.is_file()):
            continue
        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in rows(flux_path):
            for k in ('gross_gC', 'maintenance_gC', 'structural_gC', 'growth_resp_gC'):
                sums[r['plant_instance_id']][k] += float(r[k])
        for l in rows(ledger_path):
            for k in ('gross_gC', 'maintenance_gC', 'structural_gC', 'growth_resp_gC'):
                worst = max(worst, abs(sums[l['plant_instance_id']][k] - float(l[k]))); checked += 1
    return dict(steps=len(steps), comparisons=checked, max_abs_difference_gC=worst, status='axis_sums_match_plant_ledger' if worst <= 1e-12 else 'axis_sums_differ')


def truss_provenance(state: dict, states_day0: dict) -> str:
    """registry_truss for an imported (observed/fixture) truss present at the issue boundary; generated_truss otherwise; n/a for non-trusses."""
    if state['class'] not in TRUSS_CLASSES:
        return 'n/a'
    if state['organ_id'] in states_day0 and state.get('provenance') in ('observation_fixture', 'observed', 'conditional_hidden_state'):
        return 'registry_truss'
    return 'generated_truss'
