"""Rolling-origin evaluation, observation linkage and baseline preconditions for issued Micro-Tom forecasts.

python -X utf8 -m tools.microtom_forecast_eval --help

Rules: a forecast is compared only with observations taken AFTER its issue boundary, at the same photoperiod
phase (`time_since_lights_on_min`), through the append-only correspondence table `forecast_map.csv`; ids are
never renamed. Baselines are reached only through the public dispatcher `tools.compound_growth.forecast`, and
only when the six preconditions of the forecast spec §11 hold; otherwise each one is reported, nothing is run.
Every result stays `evaluation_status = evaluated_unaccepted`; growth accuracy is not testable today.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import timedelta
from pathlib import Path
from typing import Any

from tools.microtom_forecast_io import (EVALUATION_STATUS, EVENT_COLUMNS, FORBIDDEN_STATUS, ID_PREFIX, INTERVAL_COLUMNS, MAP_COLUMNS,
                                        MTG_COLUMNS, now_iso, read_table, verify_forecast, write_table)
from tools.vtc_growth_io import ROOT, ContractError, aware, fresh_build, number, read_json, require, rows, write_json

OBSERVATION_COLUMNS = ('observed_vertex_id', 'plant_id', 'capture_id', 'observed_at', 'time_since_lights_on_min', 'photoperiod_phase',
                       'quantity', 'unit', 'value', 'reference_status', 'role', 'axis', 'truss_source')
TRUSS_SOURCES = ('registry', 'generated', 'n/a', '')
REFERENCE_STATUS = ('independent_observation', 'synthetic_truth')
ROLES = ('input', 'target')
EVAL_COLUMNS = ('forecast_run_id', 'forecast_id', 'observed_vertex_id', 'quantity', 'horizon_day', 'observed_at', 'metric', 'observed_value',
                'forecast_point', 'forecast_low', 'forecast_high', 'value', 'status', 'reason', 'reference_status')
# The six §11 preconditions with the file:function that imposes each one (verified 2026-10-11).
PRECONDITIONS = (
    ('P1_frame_board_mm', 'tools/compound_growth.py::extract_compound_state (line 124: coordinate_frame must be board_mm)', 'CR-2 human decision'),
    ('P2_millimetre_scale', 'tools/compound_growth.py::validate_history (lines 183-195: rachis_points_mm + rachis_arc_length_mm)', 'scale_status unavailable for every plant'),
    ('P3_identity_confirmed', 'tools/compound_growth.py::extract_compound_state (line 94: identity_status == confirmed)', 'phynotype2 links are proposed until review'),
    ('P4_diurnal_phase', 'tools/compound_growth.py::forecast_recent_change (lines 221-233: comparison_reason diurnal gate)', 'lights-on clock time is null (CR-6)'),
    ('P5_prediction_interval', 'tools/compound_growth.py::forecast (line 296: no interval output; uncertainty uncalibrated)', 'coverage unscoreable for the baseline column'),
    ('P6_exact_timestamp_targets', 'tools/compound_growth.py::evaluate_compound_forecasts (lines 346-355: exact observed_at match)', 'plan targets must copy observed_at verbatim'),
)


def check_preconditions(descriptor: dict) -> list[dict]:
    """Evaluate the six §11 preconditions against a snapshot descriptor; never silently defaults."""
    history = descriptor.get('history')
    checks = {
        'P1_frame_board_mm': descriptor.get('frame') == 'board_mm',
        'P2_millimetre_scale': descriptor.get('scale_status') in ('solved', 'measured') and bool(history),
        'P3_identity_confirmed': descriptor.get('identity_status') == 'confirmed',
        'P4_diurnal_phase': descriptor.get('lights_on_clock_time') is not None and int(descriptor.get('repeated_observations') or 0) >= 2,
        'P5_prediction_interval': False,  # structural: compound_growth produces no interval
        'P6_exact_timestamp_targets': bool(history) and len({s['observed_at'] for s in history}) == len(history),
    }
    out = []
    for pid, where, consequence in PRECONDITIONS:
        ok = checks[pid]
        # P5 is structural (the baseline has no interval): it limits what can be scored, it does not block the point baseline.
        status = 'pass' if ok else ('limitation' if pid == 'P5_prediction_interval' else 'blocked')
        out.append(dict(precondition=pid, status=status, reference=where,
                        reason='' if ok else consequence, descriptor=json.dumps({k: v for k, v in descriptor.items() if k != 'history'}, default=str)))
    return out


def run_baselines(out_dir: Path, descriptor: dict, horizons_days: list[float] | None = None,
                  model_parameters: dict | None = None) -> dict:
    """Report preconditions; call `tools.compound_growth.forecast` (the public dispatcher) only if all six pass."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    checks = check_preconditions(descriptor)
    write_table(out_dir / 'preconditions.csv', checks, ('precondition', 'status', 'reference', 'reason', 'descriptor'))
    blocked = [c['precondition'] for c in checks if c['status'] == 'blocked']
    result: dict[str, Any] = dict(status='', preconditions=checks, dispatcher='tools.compound_growth.forecast', models={})
    if blocked:
        result['status'] = 'blocked_by_preconditions: ' + ', '.join(blocked)
        write_json(out_dir / 'baselines.json', dict(result, note='no baseline value produced; see preconditions.csv'))
        return result
    from tools.compound_growth import forecast  # public dispatcher only (spec §2)
    params = model_parameters or {}
    for model in ('persistence', 'recent_change'):
        p = dict(params, model=model)
        require('max_horizon_days' in p and 'provenance' in p, 'baseline model_parameters need max_horizon_days and provenance')
        result['models'][model] = forecast(descriptor['history'], horizons_days or [1], None, p)
    result['status'] = 'run_via_compound_growth.forecast (P5: no interval -> coverage unscoreable for this column)'
    write_json(out_dir / 'baselines.json', result)
    return result


def load_observations(path: Path) -> list[dict]:
    table = rows(Path(path), ','.join(OBSERVATION_COLUMNS))
    for r in table:
        require(not r['observed_vertex_id'].startswith(ID_PREFIX + ':'), f'observed id in the forecast namespace: {r["observed_vertex_id"]}')
        try:
            aware(r['observed_at'])
        except ValueError as exc:
            raise ContractError(f'observed_at must be aware ISO8601: {r["observed_at"]!r}') from exc
        require(r['reference_status'] in REFERENCE_STATUS, f'reference_status invalid: {r["reference_status"]}')
        require(r['role'] in ROLES, f'role invalid: {r["role"]}')
        require(r['truss_source'] in TRUSS_SOURCES, f'truss_source invalid: {r["truss_source"]}')
    return table


def link_observation(folder: Path, links: list[dict], link_source: str) -> int:
    """Append forecast_id <-> observed_vertex_id rows to forecast_map.csv. Never rewrites or renames."""
    folder = Path(folder)
    manifest = verify_forecast(folder)
    known = {r['forecast_id'] for r in read_table(folder / 'forecast_mtg.csv', MTG_COLUMNS)}
    known |= {r['forecast_id'] for r in read_table(folder / 'forecast_intervals.csv', INTERVAL_COLUMNS)}
    existing = read_table(folder / 'forecast_map.csv', MAP_COLUMNS)
    before = (folder / 'forecast_map.csv').read_bytes()
    taken = {(r['forecast_id'], r['observed_vertex_id']) for r in existing}
    linked_at = now_iso()
    new_rows = []
    for link in links:
        fid, vid = link['forecast_id'], link['observed_vertex_id']
        require(fid in known, f'unknown forecast_id {fid}')
        require(not vid.startswith(ID_PREFIX + ':') and vid != fid, 'observed id must stay outside the forecast namespace')
        aware(link['observed_at'])
        if (fid, vid) in taken:
            continue
        taken.add((fid, vid))
        new_rows.append(dict(forecast_run_id=manifest['forecast_run_id'], trajectory_id=link.get('trajectory_id', 'ensemble'), forecast_id=fid,
                             observed_vertex_id=vid, observed_capture_id=link.get('observed_capture_id', ''), observed_at=link['observed_at'],
                             linked_at=linked_at, link_source=link_source, axis=link.get('axis', '')))
    with (folder / 'forecast_map.csv').open('a', encoding='utf-8', newline='') as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=list(MAP_COLUMNS))
        writer.writerows(new_rows)
    after = (folder / 'forecast_map.csv').read_bytes()
    require(after.startswith(before), 'append-only violated')
    return len(new_rows)


def boundary_time(manifest: dict, horizon_day: int):
    return aware(manifest['scenario']['start_time']) + timedelta(days=manifest['issue_boundary_day'] + horizon_day)


def leakage_check(manifest: dict, observations: list[dict]) -> dict:
    """No observation later than the issue boundary may be an input; targets must be strictly later."""
    issue_time = boundary_time(manifest, 0)
    issued_at = aware(manifest['issued_at'])
    independent = any(r['reference_status'] == 'independent_observation' for r in observations)
    synthetic_clock = issue_time > issued_at
    # A real observation can only post-date an origin that lies in the past; a synthetic-truth set may run on the scenario clock.
    require(not (synthetic_clock and independent), 'issue boundary lies after the wall-clock issue time; independent observations cannot be evaluated against a future origin')
    leaked = [r for r in observations if r['role'] == 'input' and aware(r['observed_at']) > issue_time]
    targets_after = [r for r in observations if r['role'] == 'target' and aware(r['observed_at']) > issue_time]
    targets_before = [r for r in observations if r['role'] == 'target' and aware(r['observed_at']) <= issue_time]
    return dict(issue_boundary_time=issue_time.isoformat(), issued_at=issued_at.isoformat(), synthetic_clock=synthetic_clock, leaked_inputs=len(leaked),
                leaked_ids=[r['observed_vertex_id'] for r in leaked], targets_after_issue=len(targets_after),
                targets_excluded_not_after_issue=len(targets_before), passed=not leaked)


def evaluate(folder: Path, observations_csv: Path, out_dir: Path, tolerance_min: int = 0) -> dict:
    """Rolling-origin comparison of one issued forecast with later, phase-matched, linked observations."""
    folder = Path(folder)
    manifest = verify_forecast(folder)
    observations = load_observations(observations_csv)
    out_dir = fresh_build(Path(out_dir))
    leak = leakage_check(manifest, observations)
    require(leak['passed'], f'future information in forecast inputs: {leak["leaked_ids"]}')
    links = {r['observed_vertex_id']: r['forecast_id'] for r in read_table(folder / 'forecast_map.csv', MAP_COLUMNS)}
    intervals = read_table(folder / 'forecast_intervals.csv', INTERVAL_COLUMNS)
    events = read_table(folder / 'forecast_events.csv', EVENT_COLUMNS)
    by_iv = {(r['forecast_id'], r['quantity'], int(r['horizon_day'])): r for r in intervals}
    by_ev = {(r['scope_id'], r['event_type'], int(r['horizon_day'])): r for r in events}
    phase = manifest['photoperiod_phase']['time_since_lights_on_min_at_boundaries']
    issue_time = boundary_time(manifest, 0)
    horizons = set(manifest['horizons_days'])
    results = []
    for o in observations:
        if o['role'] != 'target':
            continue
        base = dict(forecast_run_id=manifest['forecast_run_id'], forecast_id=links.get(o['observed_vertex_id'], ''), observed_vertex_id=o['observed_vertex_id'],
                    quantity=o['quantity'], horizon_day='', observed_at=o['observed_at'], metric='', observed_value=o['value'], forecast_point='', forecast_low='',
                    forecast_high='', value='', status='unavailable', reason='', reference_status=o['reference_status'])
        t = aware(o['observed_at'])
        if t <= issue_time:
            results.append(dict(base, reason='observation not after issue boundary')); continue
        if not base['forecast_id']:
            results.append(dict(base, reason='no correspondence row in forecast_map.csv')); continue
        delta = t - issue_time
        days, rem = divmod(delta.total_seconds(), 86400)
        rem_min = min(rem, 86400 - rem) / 60
        if rem_min > tolerance_min or int(days + (1 if rem > 43200 else 0)) not in horizons:
            results.append(dict(base, reason='observation time is not a forecast boundary horizon')); continue
        h = int(days + (1 if rem > 43200 else 0))
        base['horizon_day'] = h
        if phase == 'unavailable' or o['time_since_lights_on_min'] in ('', 'unavailable'):
            results.append(dict(base, reason='unavailable_diurnal_confound: lights-on time unknown on one side')); continue
        if abs(number(o['time_since_lights_on_min'], 'time_since_lights_on_min') - float(phase)) > tolerance_min:
            results.append(dict(base, reason='unavailable_diurnal_confound: photoperiod phase differs')); continue
        q = o['quantity']
        if q.startswith('event_time:') or q.startswith('event_count:'):
            kind, ev = q.split(':', 1)
            row = by_ev.get((base['forecast_id'], ev, h))
            if row is None:
                results.append(dict(base, reason='no forecast event row for this scope/horizon')); continue
            value = number(o['value'], 'value')
            truss_event = ev in ('new_truss', 'new_truss_generated')
            if truss_event and o['truss_source'] != 'registry':
                results.append(dict(base, reason='truss timing is compared only against registry trusses (truss_source must be registry)')); continue
            if kind == 'event_time':
                metric = 'truss_timing_error_days' if truss_event else 'event_timing_error_days'
                if row['timing_day_median'] == 'unavailable':
                    results.append(dict(base, metric=metric, forecast_point=row['timing_day_median'], reason='event never occurred in any member')); continue
                err = value - float(row['timing_day_median'])
                results.append(dict(base, metric=metric, forecast_point=row['timing_day_median'], forecast_low=row['timing_day_low'],
                                    forecast_high=row['timing_day_high'], value=err, status='evaluated_unaccepted', reason=''))
            else:
                err = value - float(row['count_median'])
                results.append(dict(base, metric='count_error', forecast_point=row['count_median'], forecast_low=row['count_low'], forecast_high=row['count_high'],
                                    value=err, status='evaluated_unaccepted', reason=''))
            continue
        row = by_iv.get((base['forecast_id'], q, h))
        if row is None:
            results.append(dict(base, reason='no forecast interval row for this quantity/horizon')); continue
        if row['unit'] != o['unit']:
            results.append(dict(base, reason=f'unit mismatch {row["unit"]} vs {o["unit"]}')); continue
        value = number(o['value'], 'value')
        lo, hi, pt = float(row['interval_low']), float(row['interval_high']), float(row['point'])
        common = dict(base, forecast_point=pt, forecast_low=lo, forecast_high=hi, status='evaluated_unaccepted', reason='')
        results.append(dict(common, metric='interval_coverage', value=1 if lo <= value <= hi else 0))
        if q == 'axis_leaf_count' and row['axis_id'] != 'M':
            results.append(dict(common, metric='substem_leaf_count_error', value=value - pt))
        else:
            results.append(dict(common, metric='area_error_mm2' if q.endswith('_mm2') else 'abs_error_' + row['unit'], value=abs(value - pt)))
    write_table(out_dir / 'evaluation.csv', results, EVAL_COLUMNS)
    scored = [r for r in results if r['status'] == 'evaluated_unaccepted']
    def mean(metric: str):
        vals = [float(r['value']) for r in scored if r['metric'] == metric]
        return dict(n=len(vals), mean=statistics.fmean(vals)) if vals else dict(n=0, mean='unavailable')
    refs = {r['reference_status'] for r in scored}
    summary = dict(schema='microtom-forecast-evaluation-1', forecast_run_id=manifest['forecast_run_id'], evaluated_at=now_iso(), leakage=leak,
                   n_target_rows=sum(r['role'] == 'target' for r in observations), n_scored=len(scored),
                   n_unavailable=len(results) - len(scored), unavailable_reasons=sorted({r['reason'] for r in results if r['status'] == 'unavailable'}),
                   interval_coverage=mean('interval_coverage'), event_timing_error_days=mean('event_timing_error_days'), count_error=mean('count_error'),
                   area_error_mm2=mean('area_error_mm2'), substem_leaf_count_error=mean('substem_leaf_count_error'),
                   truss_timing_error_days=mean('truss_timing_error_days'), truss_timing_rule='registry trusses only; generated trusses are never a reference',
                   evaluation_status=EVALUATION_STATUS,
                   growth_accuracy_status='software_check_only_synthetic_truth' if refs <= {'synthetic_truth'} else 'evaluated_unaccepted_independent_observations',
                   tolerance_min=tolerance_min)
    require(FORBIDDEN_STATUS not in json.dumps(summary), 'forbidden status label')
    write_json(out_dir / 'evaluation_summary.json', summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('link', help='append forecast_id <-> observed_vertex_id rows (JSON list) to forecast_map.csv')
    p.add_argument('--forecast', type=Path, required=True); p.add_argument('--links', type=Path, required=True); p.add_argument('--source', required=True)
    p = sub.add_parser('evaluate', help='rolling-origin evaluation against an observations CSV')
    p.add_argument('--forecast', type=Path, required=True); p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--tolerance-min', type=int, default=0)
    p = sub.add_parser('preconditions', help='report the six baseline preconditions for a descriptor JSON')
    p.add_argument('--descriptor', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(argv)
    try:
        if a.op == 'link':
            n = link_observation(a.forecast, json.loads(a.links.read_text(encoding='utf-8-sig')), a.source)
            print(json.dumps(dict(appended=n), indent=2))
        elif a.op == 'evaluate':
            print(json.dumps(evaluate(a.forecast, a.observations, a.out, a.tolerance_min), ensure_ascii=False, indent=2))
        else:
            result = run_baselines(a.out, read_json(a.descriptor))
            print(json.dumps(dict(status=result['status']), indent=2))
    except (ContractError, ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
