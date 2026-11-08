"""Forecast folder contract: write-once `build/forecast_<YYYYMMDD>_<NN>/`, own id namespace, tables, hashes.

Nothing here touches biological state. A forecast is issued once; re-issuing creates a new folder.
`forecast_map.csv` is the one append-only file (observation links); every other file is frozen by the
manifest's artifact hashes.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.vtc_growth_io import ROOT, ContractError, digest, read_json, require, rows, sha, write_json

SCHEMA = 'microtom-forecast-1'
FOLDER_RE = re.compile(r'^forecast_(\d{8})_(\d{2})$')
RUN_STATUS = 'assumption_conditioned'
EVALUATION_STATUS = 'evaluated_unaccepted'
FORBIDDEN_STATUS = 'accepted_for_domain'
VALUE_STATUS = 'generated_candidate'
TOPOLOGY_SCOPE = 'provided_axes'
APPEND_ONLY = ('forecast_map.csv',)
ID_PREFIX = 'F'
# F10: per-interval plant carbon budget of every pipeline member (committed carbon_ledger + axis_flux), rays beside every number
TRAJECTORY_COLUMNS = ('forecast_run_id', 'scenario_id', 'scenario_value', 'member_index', 'day', 'plant_instance_id', 'rays', 'reserve_start_gC', 'gross_gC',
                      'maintenance_gC', 'budget_gC', 'demand_gC', 'allocated_gC', 'funded_fraction', 'cap_binding', 'reserve_end_gC', 'deficit_gC', 'residual_gC',
                      'turning_point', 'events_this_day')
# F9: a stress fixture writes this mark into the engine scenario's `initialization_provenance` (and so into native `plants.csv`);
# the forecast layer refuses any run carrying it as a member, and no forecast table is ever derived from it.
NO_REUSE_MARK = 'reuse_as_prediction=forbidden'

MTG_COLUMNS = ('forecast_run_id', 'scenario_id', 'scenario_value', 'trajectory_id', 'forecast_id', 'rooted_at_vertex_id', 'root_relation', 'source_organ_id',
               'class', 'parent_forecast_id', 'plant_instance_id', 'true_leaf_order', 'pair_index', 'side', 'horizon_day',
               'boundary_day', 'days_after_sowing', 'frame', 'base_x_mm', 'base_y_mm', 'base_z_mm', 'tip_x_mm', 'tip_y_mm',
               'tip_z_mm', 'length_mm', 'width_mm', 'area_mm2', 'area_basis', 'surface_role', 'azimuth_deg', 'elevation_deg',
               'rachis_bend_alpha_deg', 'direction_mechanism', 'status', 'source_kind',
               'axis_id', 'axil_parent_rank', 'organ_segment_index', 'axis_segment_index', 'developmental_axis_id', 'truss_provenance')
TRUSS_PROVENANCE = ('registry_truss', 'generated_truss', 'n/a')
INTERVAL_COLUMNS = ('forecast_run_id', 'scenario_id', 'scenario_value', 'forecast_id', 'plant_instance_id', 'axis_id', 'quantity', 'unit', 'horizon_day', 'days_after_sowing',
                    'point', 'interval_low', 'interval_high', 'nominal_level', 'interval_kind', 'calibrated', 'n_members',
                    'n_members_present', 'presence_probability', 'status', 'source_kind')
EVENT_COLUMNS = ('forecast_run_id', 'scenario_id', 'scenario_value', 'plant_instance_id', 'scope_id', 'event_type', 'horizon_day', 'days_after_sowing', 'probability',
                 'probability_basis', 'count_median', 'count_low', 'count_high', 'timing_day_median', 'timing_day_low', 'timing_day_high',
                 'n_members', 'probability_kind', 'status', 'source_kind', 'axis_id', 'truss_provenance')
PROBABILITY_BASIS = ('rule_deterministic', 'scenario_fraction', 'unavailable')
ASSUMPTION_COLUMNS = ('symbol', 'value', 'unit', 'source_kind', 'source_ref', 'cultivar', 'uncertainty', 'engine_key',
                      'engine_status', 'origin')
MAP_COLUMNS = ('forecast_run_id', 'trajectory_id', 'forecast_id', 'observed_vertex_id', 'observed_capture_id', 'observed_at',
               'linked_at', 'link_source', 'axis')
UNRESOLVED_COLUMNS = ('quantity', 'scope', 'reason', 'status', 'spec_reference')
MEMBER_COLUMNS = ('scenario_id', 'scenario_value', 'member_index', 'trajectory_id', 'member_values', 'run_id', 'inputs_dir', 'run_dir', 'results_dir',
                  'results_manifest_sha256', 'member_source', 'engine', 'instance_ids', 'checkpoint_hashes')


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def new_forecast_dir(root: Path = ROOT / 'build', date: str | None = None) -> Path:
    """Next free `forecast_<YYYYMMDD>_<NN>` under `root`; created write-once (mkdir fails if it appears meanwhile)."""
    root = Path(root)
    require(root.resolve().is_relative_to((ROOT / 'build').resolve()), 'forecast folders live below project build/')
    date = date or datetime.now().strftime('%Y%m%d')
    require(re.fullmatch(r'\d{8}', date) is not None, 'date must be YYYYMMDD')
    taken = set()
    if root.exists():
        for p in root.iterdir():
            m = FOLDER_RE.match(p.name)
            if m and m.group(1) == date:
                taken.add(int(m.group(2)))
    nn = next(i for i in range(1, 100) if i not in taken)
    folder = root / f'forecast_{date}_{nn:02d}'
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def forecast_identifier(run_id: str, organ_key: str, scenario_id: str | None = None) -> str:
    """Own namespace: `F:<run>:<organ key>` or `F:<run>:<scenario>:<organ key>` (F8). Observed vertex ids never start with `F:`."""
    require(bool(organ_key) and not any(c in organ_key for c in ',"\\\r\n\t'), f'unsafe organ key {organ_key!r}')
    return f'{ID_PREFIX}:{run_id}:{scenario_id}:{organ_key}' if scenario_id else f'{ID_PREFIX}:{run_id}:{organ_key}'


def write_table(path: Path, table: list[dict], columns: tuple[str, ...]) -> None:
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction='raise')
        writer.writeheader()
        for row in table:
            missing = set(columns) - set(row)
            require(not missing, f'row lacks columns {sorted(missing)} for {path.name}')
            writer.writerow(row)


def read_table(path: Path, columns: tuple[str, ...] | None = None) -> list[dict]:
    return rows(Path(path), ','.join(columns) if columns else None)


def artifact_hashes(folder: Path, exclude: tuple[str, ...] = APPEND_ONLY + ('manifest.json',)) -> dict[str, str]:
    folder = Path(folder)
    out = {}
    for p in sorted(folder.rglob('*')):
        if p.is_file():
            rel = p.relative_to(folder).as_posix()
            if rel in exclude:
                continue
            out[rel] = sha(p)
    return out


def finalize(folder: Path, manifest: dict) -> dict:
    """Write `manifest.json` last with hashes of every frozen file; refuse to finalize twice."""
    folder = Path(folder)
    require(not (folder / 'manifest.json').exists(), 'forecast already issued; re-issue creates a new folder')
    for key in ('forecast_run_id', 'issued_at', 'input_snapshot_sha256', 'code_sha256', 'parameter_sha256'):
        require(bool(manifest.get(key)), f'manifest needs {key}')
    require(manifest.get('run_status') == RUN_STATUS and manifest.get('evaluation_status') == EVALUATION_STATUS,
            'forecast status labels fixed by the spec')
    require(FORBIDDEN_STATUS not in str(manifest), f'{FORBIDDEN_STATUS} may not appear')
    manifest = dict(manifest, schema=SCHEMA, append_only_files=list(APPEND_ONLY), artifact_hashes=artifact_hashes(folder))
    manifest['artifact_digest'] = digest(manifest['artifact_hashes'])
    write_json(folder / 'manifest.json', manifest)
    return manifest


def verify_forecast(folder: Path) -> dict:
    """Recompute the frozen-file hashes and compare with the manifest; returns the manifest."""
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    require(manifest.get('schema') == SCHEMA, 'not a forecast folder')
    actual = artifact_hashes(folder, exclude=tuple(manifest['append_only_files']) + ('manifest.json',))
    require(actual == manifest['artifact_hashes'], 'forecast folder changed after issue')
    return manifest


def snapshot_digest(paths: list[Path]) -> str:
    """Digest of {relative name: sha256} for the input snapshot files, independent of folder location."""
    return digest({Path(p).name: sha(Path(p)) for p in paths})


def code_digest(paths: list[Path]) -> str:
    return digest({Path(p).name: sha(Path(p)) for p in paths})


def list_forecasts(root: Path = ROOT / 'build') -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and FOLDER_RE.match(p.name)) if root.exists() else []
