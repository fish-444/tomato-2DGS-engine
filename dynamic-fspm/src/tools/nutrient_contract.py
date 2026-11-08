"""Strict contracts for conditional nutrient accounting, not crop response laws."""
from __future__ import annotations

from datetime import datetime
import json
import math
from typing import Any

from tools.leaf_light.common import StudyError, digest

SCHEMA = 'fsp-nutrients-1'
# Conventional atomic weights: unit conversion constants, not fitted parameters.
ATOMIC_MASS = {'N': 14.007, 'P': 30.973761998, 'K': 39.0983, 'Ca': 40.078,
               'Mg': 24.305, 'S': 32.06, 'Fe': 55.845, 'B': 10.81,
               'Mn': 54.938044, 'Zn': 65.38, 'Cu': 63.546, 'Mo': 95.95,
               'Na': 22.98976928, 'Cl': 35.45, 'O': 15.999}
SPECIES_ELEMENT = {s: s for s in ATOMIC_MASS if s not in {'N', 'O'}}
SPECIES_ELEMENT.update(NO3_N='N', NH4_N='N', unspecified_N='N')
EVIDENCE_KINDS = {'measured', 'assumed', 'borrowed', 'fitted'}
SCIENCE_DEPENDENCIES = (
    'substrate_water_transport', 'ion_retention_release', 'root_nutrient_uptake',
    'plant_nutrient_demand_allocation', 'nutrient_growth_response',
    'salinity_and_ion_interactions', 'carbon_nutrient_growth_controller')


def require_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise StudyError('Finite JSON-compatible input/output required') from exc


class Unavailable(Exception):
    """Required scientific input is absent; malformed input is StudyError."""

    def __init__(self, code: str, detail: str):
        self.code, self.detail = code, detail
        super().__init__(detail)


def obj(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise StudyError(f'{label}: object required')
    return value


def array(value: Any, label: str) -> list:
    if not isinstance(value, list):
        raise StudyError(f'{label}: array required')
    return value


def text(value: Any, label: str) -> str:
    if value is None or value == '':
        raise Unavailable('input_missing', label)
    if not isinstance(value, str) or not value.strip():
        raise StudyError(f'{label}: nonempty string required')
    return value


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if value is None:
        raise Unavailable('input_missing', label)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise StudyError(f'{label}: finite numeric value required')
    if value < 0 or (positive and value == 0):
        raise StudyError(f'{label}: invalid nonnegative/positive value')
    return float(value)


def evidence(value: Any, label: str) -> dict:
    if value is None:
        raise Unavailable('evidence_missing', label)
    data = obj(value, label)
    if data.get('kind') not in EVIDENCE_KINDS:
        raise StudyError(f'{label}: unsupported evidence kind')
    text(data.get('source_ref'), label + '.source_ref')
    if data['kind'] == 'assumed':
        text(data.get('assumption'), label + '.assumption')
    return data


def time(value: Any, label: str) -> datetime:
    value = text(value, label)
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StudyError(f'{label}: ISO8601 timestamp required') from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise StudyError(f'{label}: timezone required')
    return result


def species(value: Any) -> list[str]:
    values = array(value, 'tracked_species')
    if not values or any(not isinstance(v, str) or v not in SPECIES_ELEMENT for v in values):
        raise StudyError('tracked_species: nonempty list of supported element/form names required')
    if len(values) != len(set(values)):
        raise StudyError('tracked_species: duplicates')
    return values


def vector(value: Any, scope: list[str], label: str) -> dict[str, float]:
    if value is None:
        raise Unavailable('nutrient_pool_missing', label)
    data = obj(value, label)
    if set(data) - set(scope):
        raise StudyError(f'{label}: species outside tracked scope')
    return {s: number(data.get(s), f'{label}.{s}') for s in scope}


def zeros(scope: list[str]) -> dict[str, float]:
    """Accounting accumulator only; never fills unknown physical inputs."""
    return dict.fromkeys(scope, 0.0)


def unavailable(exc: Unavailable, request: Any) -> dict:
    return dict(schema_version=SCHEMA, status='unavailable', reason=exc.code,
                detail=exc.detail, input_hash=digest(request), validated_nutrient_model=False)


def science_readiness() -> dict:
    return dict(status='unavailable', reason='nutrient_science_contract_pending',
                missing_contracts=list(SCIENCE_DEPENDENCIES),
                uptake_prediction='not_implemented', growth_feedback='not_implemented',
                direction_response='out_of_scope', validated_nutrient_model=False)
