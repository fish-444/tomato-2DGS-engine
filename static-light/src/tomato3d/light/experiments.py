"""Paired repeat statistics and separate Monte Carlo precision diagnostics."""
from collections import defaultdict
import json
import numpy as np


def signature(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def paired_statistics(records: list[dict], baseline: str, bootstrap_seed: int,
                      draws: int) -> tuple[list[dict], list[dict]]:
    """Resample whole paired repetitions, never treat leaflets as independent repeats."""
    if type(draws) is not int or draws < 100 or draws > 100000:
        raise ValueError('bootstrap draws must be 100..100000')
    by = {}
    for r in records:
        key = tuple(r[k] for k in ('ensemble', 'rays', 'rep_id', 'scenario', 'level', 'entity_id'))
        if key in by:
            raise ValueError('duplicate experiment result')
        by[key] = r
    deltas = []
    for key, r in by.items():
        if r['scenario'] == baseline:
            continue
        base_key = key[:3]+(baseline,)+key[4:]
        if base_key not in by:
            raise ValueError('unmatched paired identity')
        b = by[base_key]
        if signature(r['controls']) != signature(b['controls']):
            raise ValueError('pair changed ray/rotation/outline/member controls')
        valid = b['absorbed_umol_s'] is not None and r['absorbed_umol_s'] is not None
        delta = r['absorbed_umol_s']-b['absorbed_umol_s'] if valid else None
        deltas.append({k:r[k] for k in ('ensemble','rays','rep_id','scenario','level','entity_id','rank','position')} |
            dict(delta_umol_s=delta, relative_delta=delta/b['absorbed_umol_s'] if valid and b['absorbed_umol_s']>0 else None,
                 baseline_umol_s=b['absorbed_umol_s'], candidate_umol_s=r['absorbed_umol_s'],
                 status='computed' if valid else 'unavailable'))
    grouped = defaultdict(list)
    for row in deltas:
        grouped[tuple(row[k] for k in ('ensemble','rays','scenario','level','entity_id'))].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda r: r['rep_id'])
        values = [r['delta_umol_s'] for r in group]
        valid = len(values)>1 and all(v is not None for v in values)
        summary = dict(zip(('ensemble','rays','scenario','level','entity_id'), key), n_pairs=len(values),
            mean_delta_umol_s=None, median_delta_umol_s=None, ci_low_umol_s=None, ci_high_umol_s=None,
            method='paired_repeat_percentile_bootstrap_95_pointwise', status='unavailable')
        if valid:
            a = np.asarray(values)
            # Identical indices across entities preserve repeat clustering, but CIs are pointwise.
            rng = np.random.default_rng(bootstrap_seed)
            boot = np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(draws)])
            summary.update(mean_delta_umol_s=float(a.mean()), median_delta_umol_s=float(np.median(a)),
                ci_low_umol_s=float(np.quantile(boot,.025)),ci_high_umol_s=float(np.quantile(boot,.975)),status='estimated')
        summaries.append(summary)
    return deltas, summaries


def convergence(records: list[dict], baseline: str, target_cv: float | None) -> tuple[list[dict], dict]:
    """CV is repeat SD(ddof=1)/mean, not SE/mean. Zero mean is unavailable."""
    if target_cv is not None and (not np.isfinite(target_cv) or not 0 < target_cv < 1):
        raise ValueError('invalid CV target')
    selected = [r for r in records if r['ensemble']=='ray_noise' and r['scenario']==baseline]
    groups = defaultdict(list)
    for r in selected:
        groups[(r['rays'],r['level'],r['entity_id'])].append(r)
    table = []
    geometry_by_entity = defaultdict(set)
    for (rays,level,entity), group in sorted(groups.items()):
        controls = [{k:v for k,v in r['controls'].items() if k != 'ray_seed'} for r in group]
        if len({signature(c) for c in controls}) != 1 or len({r['controls']['ray_seed'] for r in group}) != len(group):
            raise ValueError('ray noise ensemble must vary only unique ray seeds')
        geometry_by_entity[(level,entity)].update(r['geometry_hash'] for r in group)
        vals = [r['absorbed_umol_s'] for r in group]
        valid = len(vals)>1 and all(v is not None for v in vals)
        mean = float(np.mean(vals)) if valid else None
        sd = float(np.std(vals, ddof=1)) if valid else None
        cv = sd/mean if valid and mean>0 else None
        table.append(dict(rays=rays,level=level,entity_id=entity,n_repeats=len(vals),mean_umol_s=mean,
            sd_umol_s=sd,cv=cv,below_target=cv<=target_cv if cv is not None and target_cv is not None else None,
            target_cv=target_cv,target_source_kind='user_assumed' if target_cv is not None else 'unavailable',
            zero_mean=(mean==0) if valid else None))
    if any(len(v)!=1 for v in geometry_by_entity.values()):
        raise ValueError('convergence changed geometry across ray counts/seeds')
    minimum = {}
    for level in ('leaflet','leaf'):
        counts = sorted({r['rays'] for r in table if r['level']==level})
        minimum[level] = next((n for n in counts if all(r['below_target'] is True for r in table
            if r['level']==level and r['rays']>=n)), None)
    return table, dict(minimum_tested_rays=minimum, target_cv=target_cv,
        criterion='all_entities_at_this_and_every_higher_tested_count',
        interpretation='sample_precision_only_not_bias_accuracy_or_received_ray_count')
