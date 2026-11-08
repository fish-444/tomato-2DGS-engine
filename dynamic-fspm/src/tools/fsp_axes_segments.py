"""Explicit producer segments and same-axis truss history; no imputation."""
from copy import deepcopy
from collections import defaultdict
from tools.vtc_growth_io import require

UNAVAILABLE = 'unavailable'
TRUSS_CLASSES = {'Truss', 'TrussAxis'}


def validate_truss_rows(events, vertices, membership):
    by={(v['plant_instance_id'],v['vertex_id']):v for v in vertices}
    seen=set()
    for r in events:
        require('node_id' not in r or 'truss_vertex_id' not in r or r['node_id']==r['truss_vertex_id'],'truss node aliases disagree')
        oid=r.get('truss_vertex_id',r.get('node_id'))
        key=r['plant_instance_id'],oid
        require(key not in seen,'duplicate truss event');seen.add(key)
        v=by.get(key)
        require(v is not None and v['class'] in TRUSS_CLASSES and membership[key]==r['axis_id'],'truss event has no matching same-axis node')
        require(segment(r)==segment(v),'truss event/node segment disagreement')
    require(seen=={key for key,v in by.items() if v['class'] in TRUSS_CLASSES},'truss table/node coverage mismatch')


def segment(row):
    values = [row[k] for k in ('segment_index', 'sympodial_segment_index') if k in row]
    def parse(v):
        if v in (None, '', UNAVAILABLE): return UNAVAILABLE
        require(not isinstance(v, bool) and str(v).isdigit() and int(v) >= 1, 'segment_index must be positive integer or unavailable')
        return int(v)
    parsed = [parse(v) for v in values]
    require(not parsed or all(v == parsed[0] for v in parsed), 'segment aliases disagree')
    return parsed[0] if parsed else UNAVAILABLE


def prepare_segments(records, axis_rows, vertices, membership):
    """Copy metadata into the request; never infer an absent index from trusses."""
    result = deepcopy(records)
    producer = {(r['plant_instance_id'], r['axis_id']): r for r in axis_rows}
    by = {(v['plant_instance_id'], v['vertex_id']): v for v in vertices}
    for a in result:
        key = a['plant_instance_id'], a['axis_id']; index = segment(producer[key])
        if 'segment_index' in a or 'sympodial_segment_index' in a:
            require(segment(a) == index, 'request/producer segment disagreement')
        a['segment_index'] = index
        owned = [v for v in vertices if (v['plant_instance_id'], membership[v['plant_instance_id'], v['vertex_id']]) == key]
        axis_node = by[a['plant_instance_id'], a['axis_vertex_id']]
        require(segment(axis_node) == index, 'axis/MTG segment disagreement')
        leaves = sorted((v for v in owned if v['class'] == 'CompoundLeaf'), key=lambda v: int(v['rank']))
        known = [segment(v) for v in leaves if segment(v) != UNAVAILABLE]
        require(known == sorted(known), 'decreasing segment_index along axis leaf ranks')
        history = [dict(node_id=v['vertex_id'], segment_index=segment(v)) for v in owned if v['class'] in TRUSS_CLASSES]
        history.sort(key=lambda r: r['segment_index'] if r['segment_index'] != UNAVAILABLE else 10**9)
        if 'truss_history' in a:
            declared = a['truss_history']
            require(len({e['node_id'] for e in declared}) == len(declared), 'duplicate truss history node')
            for e in declared:
                v = by.get((a['plant_instance_id'], e['node_id']))
                require(v is not None and v['class'] in TRUSS_CLASSES and membership[a['plant_instance_id'], e['node_id']] == a['axis_id'], 'truss event has no matching same-axis node')
                require(segment(e) == segment(v), 'truss event/node segment disagreement')
            require(declared == history, 'truss history decreasing or incomplete')
        a['truss_history'] = history
        complete = index != UNAVAILABLE and all(segment(v) != UNAVAILABLE for v in owned if v['class'] != 'Plant')
        if complete:
            numbers = [e['segment_index'] for e in history]
            require(numbers == list(range(1, len(numbers)+1)), 'truss segment history must increase from 1 without gaps')
            require(a['initialization']['truss_count'] == len(history), 'hidden truss count disagrees with supplied history')
            expected = len(history)+1
            require(index == expected, 'current segment disagrees with truss history')
            require(all(segment(v) <= index for v in owned if v['class'] != 'Plant'), 'node segment exceeds axis current segment')
        a['truss_rule_status'] = 'available' if complete else UNAVAILABLE
        a['segment_source'] = dict(source_kind='producer_supplied', source_ref='axes.csv + selected MTG nodes; missing values retained')
    return result


def validate_segments(pkg, data, n):
    """Independently audit immutable organ segments and truss/successor increments."""
    axes = {(a['plant_instance_id'], a['axis_id']): a for a in pkg['request']['axes']}
    states = {(int(s['day']), s['organ_id']): s for s in data['states.csv']}
    initial = {s['organ_id']: s for s in pkg['tables']['initial_state.csv']}
    for (day, oid), s in states.items():
        value = segment(s); a = axes[s['plant_instance_id'], s['axis_id']]
        if oid in initial:
            require(value == segment(initial[oid]), 'imported segment changed')
        else:
            require(a['truss_rule_status'] == 'available' and value != UNAVAILABLE, 'birth with unavailable truss history')
            require(value >= a['segment_index'], 'generated segment decreased')
            first = states[int(s['birth_day']), oid]
            require(value == segment(first), 'organ segment changed after birth')
    for table, clock in (('events.csv','day'), ('meshes.csv','state'), ('consumed.csv','state'), ('flux.csv','interval')):
        for r in data[table]:
            day = int(r[clock]) + (table == 'flux.csv')
            require(segment(r) == segment(states[day, r['organ_id']]), 'segment export disagreement: '+table)
    for day in range(n+1):
        for key, a in axes.items():
            ss = [s for (d,_),s in states.items() if d == day and (s['plant_instance_id'],s['axis_id']) == key]
            new = [s for s in ss if s['organ_id'] not in initial]
            if a['truss_rule_status'] == UNAVAILABLE or a['initialization']['apex_stopped']:
                require(not new, 'unavailable/stopped axis initiated organs')
            if a['truss_rule_status'] == UNAVAILABLE: continue
            trusses = [s for s in ss if s['class'] == 'TrussAxis']
            numbers = sorted(segment(s) for s in trusses)
            require(numbers == list(range(1,len(numbers)+1)), 'native truss history decreasing/gapped')
            for s in new:
                value = segment(s)
                if value > a['segment_index']:
                    require(any(segment(t) == value-1 and int(t['birth_day']) <= int(s['birth_day']) for t in trusses), 'segment increment lacks truss node')
    for e in data['events.csv']:
        if e['event'] != 'determinate_stop': continue
        day = int(e['day']); s = states[day,e['organ_id']]; idx = segment(s)
        ss = [v for (d,_),v in states.items() if d==day and v['plant_instance_id']==s['plant_instance_id'] and v['axis_id']==s['axis_id']]
        require(idx != UNAVAILABLE and idx > 1 and {idx-1,idx} <= {segment(v) for v in ss if v['class']=='TrussAxis'}, 'stop has no two same-axis truss nodes')
        require(not any(v['class']=='CompoundLeaf' and segment(v)==idx for v in ss), 'stop segment contains leaves')


def axis_inventory(request, states, apex_rows=()):
    result=[]
    for a in request['axes']:
        organs=[s for s in states if s['plant_instance_id']==a['plant_instance_id'] and s['axis_id']==a['axis_id']]
        apices=[s for s in organs if s['class']=='Apex']
        known=[segment(s) for s in apices if segment(s)!=UNAVAILABLE]
        ids={s['organ_id'] for s in apices}
        counters={r['axis_segment_index'] for r in apex_rows if r['organ_id'] in ids and 'axis_segment_index' in r}
        require(len(counters)<=1,'axis segment counters disagree')
        result.append(dict(plant_instance_id=a['plant_instance_id'],axis_id=a['axis_id'],parent_rank=a['parent_rank'],order=a['order'],
            axis_state=a.get('axis_state',a.get('stage')),segment_index=next(iter(counters)) if counters else (max(known) if known else UNAVAILABLE),
            truss_rule_status=a['truss_rule_status'],truss_count=sum(s['class']=='TrussAxis' for s in organs),
            topology_scope='provided_axes',topology_completeness=UNAVAILABLE))
        from tools.fsp_engine_hooks import enabled,planned
        if enabled(request,'planned_axes'):
            result[-1].update(emergence_state=('active' if organs else 'dormant') if planned(a) else 'declared', forecast_axis_id=a.get('forecast_axis_id',UNAVAILABLE))
    return result
