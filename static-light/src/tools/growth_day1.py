"""Read-only Day-1 protocol ingestion for P3; recorded counts are not inferred ranks."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import date
from pathlib import Path

from tools.compound_growth import aware_time
from tools.leaf_light.common import StudyError, file_hash, output_dir, write_json

COLUMNS=('plant_id','capture_session_id','captured_at','capture_phase','node_count','node_count_source_ref',
         'truss_observable','truss_visible_bud_count','sowing_date','transplant_date','notes')
PHASE_MEANINGS={'early_photoperiod':'2 h after actual lights-on',
                'late_photoperiod':'2 h before actual lights-off'}


def load_day1_protocol(path: str | Path) -> dict:
    source=Path(path).resolve(); sha=file_hash(source)
    with source.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if tuple(reader.fieldnames or ())!=COLUMNS: raise StudyError('Day-1 header must match the declared protocol columns')
        raw=list(reader)
    records=[]
    for index,item in enumerate(raw,2):
        if not any(item.values()): continue
        if None in item: raise StudyError('Day-1 row has extra columns')
        row={key:(value if value!='' else None) for key,value in item.items()}
        issues=[]
        for key in ('plant_id','capture_session_id','captured_at','capture_phase'):
            if not row[key]: issues.append('day1_'+key+'_missing')
        if row['captured_at']: aware_time(row['captured_at'])
        for key in ('node_count','truss_visible_bud_count'):
            if row[key] is not None:
                try: value=int(row[key])
                except ValueError as exc: raise StudyError('Day-1 counts must be nonnegative integers') from exc
                if value<0 or str(value)!=row[key].strip(): raise StudyError('Day-1 counts must be nonnegative integers')
                row[key]=value
        if row['node_count'] is not None and not row['node_count_source_ref']: issues.append('day1_node_count_source_missing')
        if not row['truss_observable']: issues.append('day1_truss_observable_missing')
        elif row['truss_observable']!='visible_bud': issues.append('day1_truss_observable_incompatible')
        for key in ('sowing_date','transplant_date'):
            if row[key] is not None:
                try: date.fromisoformat(row[key])
                except ValueError as exc: raise StudyError('Day-1 date must be ISO YYYY-MM-DD') from exc
        row.update(source_path=str(source),source_sha256=sha,source_row=index,row_id=f'{sha}:{index}',
            availability=issues,status='unavailable' if issues else 'available',
            phase_meaning=PHASE_MEANINGS.get(row['capture_phase']),organ_age_inferred=False)
        records.append(row)
    if file_hash(source)!=sha: raise StudyError('Day-1 source changed during read')
    return dict(status='header_only' if not records else ('partially_available' if any(r['availability'] for r in records) else 'available'),
        records=records,source_hashes={str(source):sha},phase_semantics=PHASE_MEANINGS,
        truss_count_is_node_rank=False,organ_ages_inferred=False)


def backfill_origin(origin: dict, result: dict, axis_id: str | None) -> dict:
    """Only an explicit matching capture can supply a sourced Day-0 count."""
    out=deepcopy(origin); out['day1_records']=deepcopy(result['records'])
    out['source_hashes'].update(result['source_hashes'])
    selected=[r for r in result['records'] if r['plant_id']==origin['plant_id'] and
        r['capture_session_id']==origin['session_id'] and r['captured_at'] and
        aware_time(r['captured_at'])==aware_time(origin['observed_at'])]
    if len(selected)>1: raise StudyError('Ambiguous Day-1 capture records')
    if selected:
        row=selected[0]
        if row['capture_phase']: out['capture_phase']=row['capture_phase']
        if axis_id is not None:
            if axis_id not in out['axes']: raise StudyError('Unknown explicit Day-1 axis mapping')
            if row['node_count'] is not None and row['node_count_source_ref']:
                axis=out['axes'][axis_id]
                if axis.get('observed_node_count') is not None and axis['observed_node_count']!=row['node_count']:
                    raise StudyError('Day-1 node count conflicts with reviewed structure')
                axis.update(observed_node_count=row['node_count'],node_count_source_ref=row['node_count_source_ref'],
                    day1_source_row=row['row_id'])
        elif row['node_count'] is not None:
            out['availability'].append(dict(component='day1',reason_code='day1_axis_mapping_missing'))
    return out


def main(argv: list[str] | None=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',required=True,type=Path); parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv); result=load_day1_protocol(args.records)
    folder=output_dir(args.out); write_json(folder/'day1_import.json',result)
    write_json(folder/'manifest.json',dict(source_hashes=result['source_hashes'],output_hashes={'day1_import.json':file_hash(folder/'day1_import.json')}))


if __name__=='__main__': main()
