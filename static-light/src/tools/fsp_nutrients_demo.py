"""Invented fixtures for nutrient accounting. No biological parameter defaults."""
from __future__ import annotations

from tools.nutrient_contract import SCHEMA, zeros
from tools.nutrient_balance import PROCESS

DEMO_SPECIES = ['NO3_N', 'NH4_N', 'unspecified_N', 'P', 'K', 'Ca', 'Mg', 'S']


def assumption(detail: str = 'Invented arithmetic fixture, not a crop measurement') -> dict:
    return dict(kind='assumed', source_ref='synthetic_nutrient_fixture', assumption=detail)


def synthetic_recipe(*, dose_g: float = 0.1, p_percent: float = 10.0) -> dict:
    return dict(batch_id='batch1', tracked_species=DEMO_SPECIES.copy(), final_volume_L=1.0,
        source_water_volume_L=1.0, source_water_mg_L=zeros(DEMO_SPECIES),
        evidence=assumption(), source_water_evidence=assumption('Zero tracked nutrients in invented source water'),
        products=[dict(product_id='fictional_powder', label_basis='mass_percent',
            unlisted_tracked_species='declared_absent', evidence=assumption(),
            labels=[dict(analyte='total_N', value=10.0), dict(analyte='NO3_N', value=8.0),
                    dict(analyte='NH4_N', value=2.0), dict(analyte='P2O5', value=p_percent),
                    dict(analyte='K2O', value=20.0), dict(analyte='Ca', value=3.0),
                    dict(analyte='Mg', value=2.0), dict(analyte='S', value=4.0)])],
        doses=[dict(dose_id='dose1', product_id='fictional_powder', amount=dose_g, unit='g', evidence=assumption())])


def synthetic_scenario() -> dict:
    scope = DEMO_SPECIES.copy()
    # Values below are deliberately invented tests, not estimates for this user's plants.
    layers = [dict(layer_id=name, material=material, height_mm=height,
        water_ml=water, water_capacity_ml=capacity, capacity_evidence=assumption(),
        dissolved_mg={s: 1.0 for s in scope}, retained_mg={s: .2 for s in scope}, evidence=assumption())
        for name, material, height, water, capacity in
        [('upper', 'vermiculite', 50.0, 60.0, 100.0), ('lower', 'perlite', 10.0, 10.0, 25.0)]]
    events = []

    def event(kind: str, **values: object) -> None:
        index = len(events) + 1
        events.append(dict(event_id=f'e{index}', at=f'2026-10-04T{index:02d}:00:00+09:00',
                           kind=kind, evidence=assumption(), **values))

    event('irrigate', layer_id='upper', batch_id='batch1', water_ml=20.0)
    event('transfer', from_layer_id='upper', to_layer_id='lower', water_ml=10.0)
    event('drain', layer_id='lower', water_ml=5.0)
    event('reuse', layer_id='upper', water_ml=2.0)
    event('evaporate', layer_id='upper', water_ml=2.0)
    event('transpire', layer_id='upper', water_ml=3.0)
    event('retain', layer_id='upper', nutrient_mg={s: .05 for s in scope})
    event('release', layer_id='upper', nutrient_mg={s: .02 for s in scope})
    event('uptake', layer_id='upper', nutrient_mg={s: .1 for s in scope})
    event('allocate', organ_id='leaf', nutrient_mg={s: .05 for s in scope})
    return dict(schema_version=SCHEMA, scenario_id='constant_fixture',
        plant_instance_id='plant1', source_plant_id='source1',
        t_start='2026-10-04T00:00:00+09:00', t_end='2026-10-05T00:00:00+09:00',
        tracked_species=scope, evidence=assumption(), mixing_model='well_mixed_each_layer',
        mixing_evidence=assumption('Within-layer complete mixing for prescribed water aliquots'),
        drainage_mode='retain_for_explicit_reuse',
        geometry=dict(width_mm=60.0, depth_mm=60.0, dimension_basis='internal_filled',
                      evidence=assumption('User dimensions treated as internal only for this synthetic fixture')),
        layers=layers, plant_initial=dict(labile_mg=zeros(scope), organs={'leaf': zeros(scope)}, evidence=assumption()),
        drainage_initial=dict(water_ml=0.0, dissolved_mg=zeros(scope), evidence=assumption()),
        processes={p: dict(mode='prescribed_events', evidence=assumption()) for p in sorted(set(PROCESS.values()))},
        recipes=[synthetic_recipe()], events=events, observations=[],
        numerics=dict(water_abs_tol_ml=1e-10, mass_abs_tol_mg=1e-10))


def blank_template() -> dict:
    """Known dimensions are metadata; physical/evidence fields stay blank."""
    result = synthetic_scenario()
    for field in ('scenario_id', 'plant_instance_id', 'source_plant_id', 't_start', 't_end',
                  'evidence', 'mixing_model', 'mixing_evidence', 'drainage_mode'):
        result[field] = None
    result['geometry']['dimension_basis'] = None
    result['geometry']['evidence'] = dict(kind='assumed', source_ref='user_capture_context',
        assumption='Reported 6x6 cm and 5 cm vermiculite over 1 cm perlite; internal filling dimensions unconfirmed')
    for row in result['layers']:
        for key in ('water_ml', 'water_capacity_ml', 'capacity_evidence', 'evidence'):
            row[key] = None
        for key in ('dissolved_mg', 'retained_mg'):
            row[key] = dict.fromkeys(DEMO_SPECIES)
    result['plant_initial'] = dict(labile_mg=dict.fromkeys(DEMO_SPECIES), organs={}, evidence=None)
    result['drainage_initial'] = dict(water_ml=None, dissolved_mg=dict.fromkeys(DEMO_SPECIES), evidence=None)
    result['processes'] = {p: dict(mode=None, evidence=None) for p in sorted(set(PROCESS.values()))}
    recipe = synthetic_recipe()
    recipe.update(batch_id=None, final_volume_L=None, source_water_volume_L=None,
                  source_water_mg_L=dict.fromkeys(DEMO_SPECIES), source_water_evidence=None,
                  evidence=None, products=[], doses=[])
    result['recipes'] = [recipe]
    result['events'] = []
    result['observations'] = []
    result['numerics'] = dict(water_abs_tol_ml=None, mass_abs_tol_mg=None)
    return result
