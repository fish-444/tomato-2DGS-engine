"""Climate set-points, reference light, slot-ratio pointer and 16-cell light validation for the Micro-Tom runs.

python -X utf8 -m tools.microtom_environment --help

Rules carried from the work order: set-points are `user_stated`, never logs; absolute PPFD/DLI come only
from `light_input.csv` with its own `source_kind`; phone ratios are relative (value / 16-cell mean, divided by
the recorded tray-base factor when the reference is defined over the three trays) and are never multiplied
into ray-traced light; the pointer file is read, never edited by code.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tools.vtc_development_io import TABLES
from tools.vtc_growth_io import ContractError, require, rows, sha, write_csv, write_json, number, aware, ratio_table

ENVIRONMENT_HEADER = TABLES['environment.csv'].split(',')
LIGHT_INPUT_HEADER = TABLES['light_input.csv'].split(',')
SLOT_HEADER_REQUIRED = ('slot_id', 'ratio_now')
GRID_HEADER_REQUIRED = ('grid_row', 'grid_col', 'ratio_to_mean')
SENSOR_HEADER = ('grid_row', 'grid_col', 'absorbed_umol_s', 'area_m2')
LIGHT_SOURCE_KINDS = ('measured', 'user_assumed', 'synthetic_assumed')
SECONDS_PER_HOUR = 3600  # unit conversion, not a coefficient
UMOL_PER_MOL = 1e6       # unit conversion, not a coefficient


def thermal_time_per_day(day_c: float, night_c: float, photoperiod_h: float, t_base_c: float) -> float:
    """Same expression V3 uses (vtc_development_io.validate_outputs): time-weighted mean minus base, floored at 0."""
    return max(0.0, (day_c * photoperiod_h + night_c * (24 - photoperiod_h)) / 24 - t_base_c)


def make_climate_csv(out_dir: Path, start_time: str, days: int, day_c: float = 26, night_c: float = 22,
                     photoperiod_h: float = 14, rh: float = 60, co2_ppm: float = 400,
                     source_kind: str = 'user_stated', co2_source_kind: str = 'user_assumed',
                     t_base_c: float | None = None) -> dict:
    """Write `environment.csv` in V3's exact header plus a `climate_setpoints.json` sidecar (RH has no V3 column).

    `start_time` must be timezone-aware ISO8601 (V3 requires aware timestamps). Returns the sidecar dict.
    """
    out_dir = Path(out_dir)
    require(type(days) is int and 1 <= days <= 365, 'days must be 1..365 (V3 interval range)')
    require(0 < photoperiod_h <= 24, 'photoperiod must be in (0, 24]')
    start = aware(start_time)
    table = [dict(interval=i, start_time=(start + timedelta(days=i)).isoformat(), light_hours=photoperiod_h,
                  temp_day_c=day_c, temp_night_c=night_c, co2_ppm=co2_ppm, source_kind=source_kind) for i in range(days)]
    write_csv(out_dir / 'environment.csv', table, ENVIRONMENT_HEADER)
    sidecar = dict(schema='microtom-climate-setpoints-1', kind='set_points_not_logs', source_kind=source_kind,
                   start_time=start.isoformat(), days=days, temp_day_c=day_c, temp_night_c=night_c,
                   photoperiod_h=photoperiod_h, rh_pct=rh, co2_ppm=co2_ppm, co2_source_kind=co2_source_kind,
                   lights_on_clock_time=None, dli_factor_mol_per_umol_s=photoperiod_h * SECONDS_PER_HOUR / UMOL_PER_MOL)
    if t_base_c is not None:
        sidecar['thermal_time_dd_per_day'] = thermal_time_per_day(day_c, night_c, photoperiod_h, t_base_c)
        sidecar['t_base_c'] = t_base_c
    write_json(out_dir / 'climate_setpoints.json', sidecar)
    return sidecar


def dli_from_ppfd(ppfd_umol_m2_s: float, photoperiod_h: float) -> float:
    """DLI [mol m-2 d-1] = PPFD x 3600 x photoperiod_h / 1e6 (0.0504 x PPFD at 14 h). Derived, not a fixed coefficient."""
    return number(ppfd_umol_m2_s, 'reference_ppfd', 0) * SECONDS_PER_HOUR * photoperiod_h / UMOL_PER_MOL


def load_light_input(path: Path, photoperiod_h: float = 14) -> dict:
    """Read `light_input.csv` (V3 header) and derive the DLI; anything but `measured` is conditional."""
    table = rows(Path(path), TABLES['light_input.csv'])
    require(len(table) == 1, 'light_input.csv must hold exactly one reference row')
    row = table[0]
    require(row['source_kind'] in LIGHT_SOURCE_KINDS, f'light source_kind invalid: {row["source_kind"]!r}')
    if row['source_kind'] == 'measured':
        require(bool(row['measured_at']), 'measured light needs measured_at (aware ISO8601)')
        try:
            aware(row['measured_at'])
        except ValueError as exc:
            raise ContractError(f'measured_at is not an aware ISO8601 timestamp: {row["measured_at"]!r}') from exc
    ppfd = number(row['reference_ppfd_umol_m2_s'], 'reference_ppfd_umol_m2_s', 0)
    return dict(reference_ppfd_umol_m2_s=ppfd, source_kind=row['source_kind'], measured_at=row['measured_at'],
                photoperiod_h=photoperiod_h, dli_mol_m2_d=dli_from_ppfd(ppfd, photoperiod_h),
                conditional=row['source_kind'] != 'measured',
                absolute_status='measured' if row['source_kind'] == 'measured' else 'conditional_on_assumed_reference',
                sha256=sha(Path(path)))


def resolve_ratio_pointer(pointer_txt: Path) -> dict:
    """Read `light_ratio_latest.txt` (one relative path), validate the target header, record its hash.

    Slot schema needs `slot_id, ratio_now`; grid schema needs `grid_row, grid_col, ratio_to_mean`
    (the two schemas `vtc_growth_io.ratio_table` accepts). Anything else aborts with ContractError.
    """
    pointer_txt = Path(pointer_txt)
    require(pointer_txt.is_file(), f'pointer file missing: {pointer_txt}')
    text = pointer_txt.read_text(encoding='utf-8-sig').strip()
    require(bool(text) and '\n' not in text and '\r' not in text, 'pointer must hold exactly one line')
    require(not Path(text).is_absolute(), 'pointer must be a relative path')
    target = (pointer_txt.parent / text).resolve()
    require(target.is_file(), f'pointer target missing: {target}')
    table = rows(target)
    header = set(table[0]) if table else set()
    if set(SLOT_HEADER_REQUIRED) <= header:
        schema = 'slot'
    elif set(GRID_HEADER_REQUIRED) <= header:
        schema = 'grid'
    else:
        raise ContractError(f'ratio file header must contain {SLOT_HEADER_REQUIRED} or {GRID_HEADER_REQUIRED}: {target}')
    ratio_table(target)  # V3's own positivity/uniqueness/normalisation checks
    return dict(pointer=str(pointer_txt), pointer_text=text, target=str(target), target_sha256=sha(target),
                schema=schema, n_rows=len(table), role='comparison_only')


def slot_relative_targets(resolved: dict, tray_base_factor: float | None = None) -> list[dict]:
    """Per-slot RELATIVE light targets (ratio only). Divides by `tray_base_factor` when the reference is the 3 trays.

    Absolute µmol values are deliberately not produced here; they exist only as reference x ratio when the
    reference in `light_input.csv` is `measured`, and even then never feed ray-traced photons.
    """
    require(resolved['schema'] == 'slot', 'slot targets need the slot schema')
    factor = 1.0 if tray_base_factor is None else number(tray_base_factor, 'tray_base_factor', 0)
    out = []
    for r in rows(Path(resolved['target'])):
        ratio = number(r['ratio_now'], 'ratio_now', 0)
        lo = number(r['ratio_now_lo'], 'ratio_now_lo', 0) if r.get('ratio_now_lo') else ratio
        hi = number(r['ratio_now_hi'], 'ratio_now_hi', 0) if r.get('ratio_now_hi') else ratio
        out.append(dict(slot_id=r['slot_id'], relative_target=ratio / factor, relative_low=lo / factor,
                        relative_high=hi / factor, base='three_trays' if tray_base_factor else 'sixteen_cell_mean',
                        source_kind='app_ratio', ratio_file_sha256=resolved['target_sha256']))
    return out


def _normalised(values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    require(len(values) == 16 and set(values) == {(str(r), str(c)) for r in range(1, 5) for c in range(1, 5)},
            'exactly the 16 canonical grid cells are required on each side')
    mean = sum(values.values()) / 16
    require(mean > 0, 'cannot normalise a zero mean')
    return {k: v / mean for k, v in values.items()}


def pearson_r(x: list[float], y: list[float]) -> float | str:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return 'unavailable'  # a constant side has no correlation
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def light_validation(sim_cells_csv: Path, measured_cells_csv: Path, out_csv: Path | None = None,
                     tray_base_factor: float | None = None) -> dict:
    """Compare V3 `sensors.csv` (16 virtual sensors) with the measured 16-cell ratio map, both / their mean.

    Writes `light_validation.csv` (one row per cell) and returns the summary (ratio RMSE, Pearson r, n).
    Shape check only: no absolute calibration is implied and the sensor height stays a flagged parameter.
    """
    sim_rows = rows(Path(sim_cells_csv))
    require(set(SENSOR_HEADER) <= set(sim_rows[0]) if sim_rows else False, 'simulated cells need V3 sensors.csv columns')
    sim = {}
    for r in sim_rows:
        area = number(r['area_m2'], 'area_m2', 0)
        require(area > 0, 'sensor area must be positive')
        sim[(r['grid_row'], r['grid_col'])] = number(r['absorbed_umol_s'], 'absorbed_umol_s', 0) / area
    meas_rows = rows(Path(measured_cells_csv))
    require(set(GRID_HEADER_REQUIRED) <= set(meas_rows[0]) if meas_rows else False, 'measured cells need grid_row, grid_col, ratio_to_mean')
    meas = {(r['grid_row'], r['grid_col']): number(r['ratio_to_mean'], 'ratio_to_mean', 0) for r in meas_rows}
    sim_n, meas_n = _normalised(sim), _normalised(meas)
    factor = 1.0 if tray_base_factor is None else number(tray_base_factor, 'tray_base_factor', 0)
    keys = sorted(sim_n, key=lambda k: (int(k[0]), int(k[1])))
    table = [dict(grid_row=k[0], grid_col=k[1], simulated_per_area=sim[k], simulated_ratio=sim_n[k],
                  measured_ratio=meas_n[k] / factor, difference=sim_n[k] - meas_n[k] / factor,
                  physical_validation=False, sensor_height_status='parameter_unrecorded',
                  measured_source_kind='app_ratio') for k in keys]
    diffs = [t['difference'] for t in table]
    summary = dict(schema='microtom-light-validation-1', n_cells=16,
                   ratio_rmse=math.sqrt(sum(d * d for d in diffs) / 16),
                   pearson_r=pearson_r([sim_n[k] for k in keys], [meas_n[k] / factor for k in keys]),
                   tray_base_factor=factor, normalisation='each side divided by its own 16-cell mean',
                   absolute_calibration=False, lamp_placement_status='declared_guess',
                   simulated_sha256=sha(Path(sim_cells_csv)), measured_sha256=sha(Path(measured_cells_csv)))
    if out_csv is not None:
        write_csv(Path(out_csv), table)
        write_json(Path(out_csv).with_suffix('.summary.json'), summary)
    summary['cells'] = table
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('climate', help='write environment.csv + climate_setpoints.json')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--start', required=True, help='aware ISO8601, e.g. 2026-10-11T06:00:00+09:00')
    p.add_argument('--days', type=int, required=True); p.add_argument('--day-c', type=float, default=26); p.add_argument('--night-c', type=float, default=22)
    p.add_argument('--photoperiod-h', type=float, default=14); p.add_argument('--rh', type=float, default=60); p.add_argument('--co2-ppm', type=float, default=400)
    p.add_argument('--t-base-c', type=float, default=None)
    p = sub.add_parser('light-input', help='read light_input.csv and derive the DLI')
    p.add_argument('--path', type=Path, required=True); p.add_argument('--photoperiod-h', type=float, default=14)
    p = sub.add_parser('pointer', help='resolve light_ratio_latest.txt and print target + hash')
    p.add_argument('--pointer', type=Path, required=True)
    p = sub.add_parser('validate-light', help='16-cell shape comparison: V3 sensors.csv vs measured ratio map')
    p.add_argument('--sim', type=Path, required=True); p.add_argument('--measured', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--tray-base-factor', type=float, default=None)
    a = parser.parse_args(argv)
    try:
        if a.op == 'climate':
            a.out.mkdir(parents=True, exist_ok=True)
            result: Any = make_climate_csv(a.out, a.start, a.days, a.day_c, a.night_c, a.photoperiod_h, a.rh, a.co2_ppm, t_base_c=a.t_base_c)
        elif a.op == 'light-input':
            result = load_light_input(a.path, a.photoperiod_h)
        elif a.op == 'pointer':
            result = resolve_ratio_pointer(a.pointer)
        else:
            result = light_validation(a.sim, a.measured, a.out, a.tray_base_factor)
            result.pop('cells')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ContractError, ValueError, OSError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
