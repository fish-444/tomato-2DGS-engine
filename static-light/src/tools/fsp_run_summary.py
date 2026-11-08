"""Run-level summary of a committed pipeline trajectory (F11 C4): daily_summary.csv, RUN_SUMMARY.md, run_manifest.json.

Everything here is aggregated from committed/step_*/ (the evidence) and the run's own manifests. Nothing is
recomputed from the engine, nothing is imputed: a day without a committed ledger is reported as unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.vtc_growth_io import ROOT, require, sha, read_json, write_json, rows, write_csv, relative_to_root

SCHEMA = 'fsp-run-summary-1'
CARBON_COLUMNS = ('reserve_start_gC', 'gross_gC', 'maintenance_gC', 'structural_gC', 'growth_resp_gC', 'reserve_end_gC', 'deficit_gC')
UNAVAILABLE = 'unavailable'


def step_number(path: Path) -> int:
    return int(path.name.split('_')[1])


def committed_steps(folder: Path) -> list[Path]:
    return sorted((p for p in (folder / 'committed').glob('step_*') if (p / 'checkpoint.json').is_file()), key=step_number)


def folder_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def organ_summary(state_rows: list[dict]) -> dict:
    leaflets = [s for s in state_rows if s['class'] == 'Leaflet']
    return dict(organ_records=len(state_rows), leaflet_records=len(leaflets),
                organ_mass_sum_gDM=sum(float(s['mass_gDM']) for s in state_rows),
                leaf_area_sum_m2=sum(float(s['area_m2']) for s in leaflets))


def daily_rows(folder: Path) -> tuple[list[dict], list[dict]]:
    """One row per committed day (plus day 0 from the frozen initial state); per-plant rows alongside."""
    daily: list[dict] = []
    per_plant: list[dict] = []
    initial = folder / 'native_inputs/initial_state.csv'
    if initial.is_file():
        s0 = rows(initial)
        daily.append(dict(day=0, **organ_summary(s0), **{c: UNAVAILABLE for c in CARBON_COLUMNS}, source_kind='frozen_initial_state'))
        for pid in sorted({s['plant_instance_id'] for s in s0}):
            per_plant.append(dict(day=0, plant_instance_id=pid, **organ_summary([s for s in s0 if s['plant_instance_id'] == pid]),
                                  **{c: UNAVAILABLE for c in CARBON_COLUMNS}, source_kind='frozen_initial_state'))
    for step in committed_steps(folder):
        day = step_number(step)
        state = rows(step / 'state.csv')
        ledger = rows(step / 'carbon_ledger.csv')
        require(all(int(r['interval']) == day - 1 for r in ledger), f'ledger interval mismatch in {step.name}')
        totals = {c: sum(float(r[c]) for r in ledger) for c in CARBON_COLUMNS}
        daily.append(dict(day=day, **organ_summary(state), **totals, source_kind='committed_native_trajectory'))
        for r in ledger:
            pid = r['plant_instance_id']
            per_plant.append(dict(day=day, plant_instance_id=pid, **organ_summary([s for s in state if s['plant_instance_id'] == pid]),
                                  **{c: float(r[c]) for c in CARBON_COLUMNS}, source_kind='committed_native_trajectory'))
    return daily, per_plant


def cap_binding(per_plant: list[dict], horizon_days: int) -> dict:
    """First committed day on which any plant's reserve could not fund structural demand (deficit_gC > 0)."""
    binding = sorted((r['day'], r['plant_instance_id']) for r in per_plant if r['deficit_gC'] != UNAVAILABLE and float(r['deficit_gC']) > 0)
    committed_days = sorted({r['day'] for r in per_plant if r['deficit_gC'] != UNAVAILABLE})
    covered = bool(committed_days) and committed_days[-1] == horizon_days
    if binding:
        value = binding[0][0]
        status = 'binds'
    elif covered:
        value, status = 'never_within_horizon', 'never_within_horizon'
    else:
        value, status = UNAVAILABLE, 'horizon_not_fully_committed'
    return dict(value=value, status=status, horizon_days=horizon_days, committed_days=len(committed_days),
                first_binding_plant=binding[0][1] if binding else UNAVAILABLE, source_kind='model_property',
                meaning='model property of this run under its declared reserve and parameters; never a plant measurement')


def lineage_verified(folder: Path) -> dict:
    """Re-check every committed step's file hashes and the previous-checkpoint chain (no engine access)."""
    from tools.vtc_growth_io import digest
    steps = committed_steps(folder)
    checked = 0
    for step in steps:
        meta = read_json(step / 'checkpoint.json')
        for name, h in meta['files'].items():
            require(sha(step / name) == h, f'committed file changed: {step.name}/{name}')
        n = step_number(step)
        if n == 1:
            expected = digest(dict(initialization=sha(folder / 'initialization.json'), prepared=sha(folder / 'prepared.json')))
        else:
            expected = sha(folder / 'committed' / f'step_{n - 1:04d}' / 'checkpoint.json')
        require(meta['lineage']['previous_checkpoint_sha256'] == expected, f'lineage break at {step.name}')
        checked += 1
    return dict(status='verified', steps=checked)


def write_summary(folder: Path, retention: dict | None = None, viewer: dict | None = None, compaction: dict | None = None) -> dict:
    folder = Path(folder).resolve()
    scenario = read_json(folder / 'scenario.json')
    contract = read_json(folder / 'contract.json') if (folder / 'contract.json').is_file() else {}
    horizon = int(scenario['intervals'])
    daily, per_plant = daily_rows(folder)
    columns = ['day', 'organ_records', 'leaflet_records', 'organ_mass_sum_gDM', 'leaf_area_sum_m2', *CARBON_COLUMNS, 'source_kind']
    write_csv(folder / 'daily_summary.csv', daily, columns)
    write_csv(folder / 'plant_daily_summary.csv', per_plant, ['day', 'plant_instance_id', *columns[1:]])
    cap = cap_binding(per_plant, horizon)
    lineage = lineage_verified(folder)
    sizes = {name: folder_size_bytes(folder / name) for name in ('committed', 'native', 'native_inputs', 'p1') if (folder / name).exists()}
    sizes['native/working'] = folder_size_bytes(folder / 'native/working') if (folder / 'native/working').exists() else 0
    sizes['native/results'] = folder_size_bytes(folder / 'native/results') if (folder / 'native/results').exists() else 0
    sizes['total'] = folder_size_bytes(folder)
    collection = read_json(folder / 'collection.json') if (folder / 'collection.json').is_file() else {}
    last = daily[-1] if daily else None
    manifest = dict(schema_version=SCHEMA, run=relative_to_root(folder), horizon_days=horizon, dt_s=scenario.get('dt_s', 86400),
                    plant_count=len(scenario.get('captures_by_instance', {})), collection_status=collection.get('status', UNAVAILABLE),
                    committed_steps=lineage['steps'], lineage=lineage, cap_binding_day=cap,
                    checkpoint_retention=retention or read_json(folder / 'retention_policy.json') if (folder / 'retention_policy.json').is_file() or retention else UNAVAILABLE,
                    compaction=compaction or (read_json(folder / 'native/working/compaction.json') if (folder / 'native/working/compaction.json').is_file() else dict(applied=False)),
                    viewer=viewer or (read_json(folder / 'native/results/manifest.json').get('viewer', UNAVAILABLE) if (folder / 'native/results/manifest.json').is_file() else UNAVAILABLE),
                    sizes_bytes=sizes, sizes_mb={k: round(v / 1e6, 1) for k, v in sizes.items()},
                    final_state=dict(day=last['day'], organ_records=last['organ_records'], leaflet_records=last['leaflet_records'],
                                     organ_mass_sum_gDM=last['organ_mass_sum_gDM'], leaf_area_sum_m2=last['leaf_area_sum_m2']) if last else UNAVAILABLE,
                    inactive_responses=dict(temperature_physiology=contract.get('temperature_physiology', UNAVAILABLE), co2_physiology=contract.get('co2_physiology', UNAVAILABLE),
                                            nutrient_feedback_enabled=contract.get('nutrient_feedback_enabled', UNAVAILABLE), fruit_sink='absent_by_declared_scope'),
                    biological_validation='not_established')
    write_json(folder / 'run_manifest.json', manifest)
    (folder / 'RUN_SUMMARY.md').write_text(summary_markdown(manifest, daily), encoding='utf-8')
    return manifest


def summary_markdown(manifest: dict, daily: list[dict]) -> str:
    cap = manifest['cap_binding_day']
    if cap['status'] == 'binds':
        cap_sentence = (f"The reserve-carbon cap first binds on day {cap['value']} (plant {cap['first_binding_plant']}): "
                        f"structural demand exceeded the reserve on that day. / 상한은 {cap['value']}일차에 처음 물린다.")
    elif cap['status'] == 'never_within_horizon':
        cap_sentence = (f"The reserve-carbon cap never binds within the {cap['horizon_days']}-day horizon: deficit_gC is 0 on every committed day. "
                        f"/ 상한은 {cap['horizon_days']}일 지평 안에서 물리지 않는다.")
    else:
        cap_sentence = f"Cap binding is unavailable: only {cap['committed_days']} of {cap['horizon_days']} days are committed. / 지평이 완주되지 않아 판단 불가."
    lines = [f"# RUN_SUMMARY ({manifest['run']['relative']})", '',
             f"Horizon {manifest['horizon_days']} days, {manifest['plant_count']} plant instances, collection status `{manifest['collection_status']}`, "
             f"committed steps {manifest['committed_steps']} (lineage {manifest['lineage']['status']}).", '',
             f"**Cap binding (source_kind = model_property):** {cap_sentence}", '',
             f"Checkpoint retention: `{manifest['checkpoint_retention']['value'] if isinstance(manifest['checkpoint_retention'], dict) else manifest['checkpoint_retention']}`; "
             f"folder total {manifest['sizes_mb']['total']} MB (committed {manifest['sizes_mb'].get('committed', 0)} MB, native/working {manifest['sizes_mb']['native/working']} MB, native/results {manifest['sizes_mb']['native/results']} MB).", '',
             f"Viewer: {json.dumps(manifest['viewer'], ensure_ascii=False)}", '',
             'Inactive responses (declared, not defects): ' + ', '.join(f'{k}={v}' for k, v in manifest['inactive_responses'].items()) + '. Biological validation: not established.', '',
             '| day | organs | leaflets | mass gDM | leaf area m2 | reserve_start gC | gross gC | structural gC | reserve_end gC | deficit gC |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in daily:
        def f(v):
            return v if v == UNAVAILABLE else f'{float(v):.4f}'
        lines.append(f"| {r['day']} | {r['organ_records']} | {r['leaflet_records']} | {float(r['organ_mass_sum_gDM']):.4f} | {float(r['leaf_area_sum_m2']):.4f} | "
                     f"{f(r['reserve_start_gC'])} | {f(r['gross_gC'])} | {f(r['structural_gC'])} | {f(r['reserve_end_gC'])} | {f(r['deficit_gC'])} |")
    lines.append('')
    lines.append('Columns are sums over all plant instances per committed day; per-plant rows are in plant_daily_summary.csv. Day 0 carbon columns are unavailable (no interval yet).')
    return '\n'.join(lines) + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(write_summary(a.run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
