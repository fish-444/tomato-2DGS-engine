"""Strict, provenance-preserving per-run development parameter overrides."""
from __future__ import annotations

from copy import deepcopy

from tools.vtc_growth_io import require, number

OVERRIDES = {
    'LEAFLET_MID_DD': ('MID_DD', 9, 'degree_day'),
    'LEAFLET_K': ('K', 9, '1/degree_day'),
    'PHYLLOCHRON_DD': ('PHYLLOCHRON_DD', None, 'degree_day'),
}
SOURCES = {'user_assumed', 'synthetic_assumed', 'literature_other_cultivar',
           'literature', 'literature_generic', 'derived_low_confidence'}


def validate_overrides(request: dict) -> None:
    params = request.get('engine_hooks', {}).get('parameters', {})
    for key, (_, _, unit) in OVERRIDES.items():
        if key not in params:
            continue
        p = params[key]
        require(isinstance(p, dict), 'development override record required')
        require(p.get('source_kind') in SOURCES, 'development override source_kind invalid')
        require(all(p.get(k) for k in ('source_ref', 'cultivar', 'uncertainty')),
                'development override provenance incomplete')
        require(p.get('unit') == unit, 'development override unit: ' + key)
        require(type(p.get('value')) in (float, int) and number(p['value'], key) > 0,
                'positive numeric development override required: ' + key)


def effective_parameters(request: dict) -> dict:
    """Never alter the raw request or assign scalar provenance to other organs."""
    validate_overrides(request)
    registry = deepcopy(request['parameters'])
    params = request.get('engine_hooks', {}).get('parameters', {})
    for key, (target, index, _) in OVERRIDES.items():
        if key not in params:
            continue
        p = deepcopy(params[key])
        if index is None:
            registry[target] = p
        else:
            registry[target]['value'][index] = p['value']
            registry[target].setdefault('component_overrides', {})[str(index)] = p
    return registry
