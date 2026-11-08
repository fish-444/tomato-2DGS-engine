"""Soluble fertilizer declarations to element mass; EC is never inverted."""
from __future__ import annotations

from copy import deepcopy
import math

from tools.leaf_light.common import StudyError, digest
from tools.nutrient_contract import (ATOMIC_MASS, SCHEMA, SPECIES_ELEMENT, Unavailable,
    array, evidence, number, obj, require_json, species, text, unavailable, vector)

OXIDES = {
    'P2O5': ('P', 2 * ATOMIC_MASS['P'] / (2 * ATOMIC_MASS['P'] + 5 * ATOMIC_MASS['O'])),
    'K2O': ('K', 2 * ATOMIC_MASS['K'] / (2 * ATOMIC_MASS['K'] + ATOMIC_MASS['O'])),
}


def product_composition(product: dict, scope: list[str]) -> dict:
    """Return mg element per g or mL product, retaining N-form uncertainty."""
    product = obj(product, 'product')
    identity = text(product.get('product_id'), 'product_id')
    evidence(product.get('evidence'), identity)
    basis = product.get('label_basis')
    if basis not in {'mass_percent', 'g_per_L_product'}:
        raise StudyError('label_basis must be mass_percent or g_per_L_product')
    # g/L equals mg/mL; percent by mass equals 10 mg/g per percentage point.
    factor, per_unit = (10.0, 'g') if basis == 'mass_percent' else (1.0, 'mL')
    labels = array(product.get('labels'), 'labels')
    out, seen = {}, set()
    total_n = None
    for row in labels:
        row = obj(row, 'label')
        analyte = text(row.get('analyte'), 'analyte')
        value = number(row.get('value'), 'label value')
        if basis == 'mass_percent' and value > 100:
            raise StudyError('Mass percent exceeds 100')
        if analyte in seen:
            raise StudyError('Duplicate label analyte')
        seen.add(analyte)
        if analyte == 'total_N':
            total_n = value * factor
            continue
        destination, conversion = OXIDES.get(analyte, (analyte, 1.0))
        if destination not in SPECIES_ELEMENT:
            raise StudyError(f'Unsupported analyte: {analyte}')
        if destination in out:
            raise StudyError('Duplicate elemental and oxide declaration')
        out[destination] = value * factor * conversion
    if total_n is not None:
        forms = math.fsum(out.get(s, 0.0) for s in ('NO3_N', 'NH4_N', 'unspecified_N'))
        if forms > total_n:
            raise StudyError('Declared N forms exceed total N')
        if 'unspecified_N' in out and forms != total_n:
            raise StudyError('Explicit N forms do not close declared total N')
        if 'unspecified_N' not in out:
            out['unspecified_N'] = total_n - forms
    if basis == 'mass_percent' and math.fsum(out.values()) > 1000:
        raise StudyError('Element mass exceeds product mass')
    if any(v > 0 for s, v in out.items() if s not in scope):
        raise StudyError('Positive product constituent outside tracked scope')
    coverage = product.get('unlisted_tracked_species')
    if coverage not in {'declared_absent', 'unknown'}:
        raise Unavailable('composition_coverage_missing', identity)
    missing = [s for s in scope if s not in out]
    if missing and coverage == 'unknown':
        raise Unavailable('composition_incomplete', f'{identity}: {missing}')
    return dict(product_id=identity, mg_per_unit={s: out.get(s, 0.0) for s in scope},
                per_unit=per_unit, evidence=deepcopy(product['evidence']),
                composition_scope='declared_tracked_elements_only', product_hash=digest(product))


def _dose_in_basis(dose: dict, product: dict, per_unit: str) -> float:
    amount = number(dose.get('amount'), 'dose.amount')
    unit = dose.get('unit')
    if unit not in {'g', 'mL'}:
        raise StudyError('Dose unit must be g or mL')
    if unit == per_unit:
        return amount
    density = number(product.get('density_g_ml'), 'density_g_ml', positive=True)
    evidence(product.get('density_evidence'), 'density_evidence')
    return amount * density if unit == 'mL' else amount / density


def _formulate(recipe: dict) -> dict:
    recipe = obj(recipe, 'recipe')
    scope = species(recipe.get('tracked_species'))
    batch = text(recipe.get('batch_id'), 'batch_id')
    evidence(recipe.get('evidence'), 'recipe')
    volume = number(recipe.get('final_volume_L'), 'final_volume_L', positive=True)
    water_volume = number(recipe.get('source_water_volume_L'), 'source_water_volume_L')
    if water_volume > volume:
        raise StudyError('Source water exceeds declared final solution volume')
    evidence(recipe.get('source_water_evidence'), 'source water')
    water = vector(recipe.get('source_water_mg_L'), scope, 'source_water_mg_L')
    mass = {s: water[s] * water_volume for s in scope}
    products = {}
    for product in array(recipe.get('products'), 'products'):
        identity = text(obj(product, 'product').get('product_id'), 'product_id')
        if identity in products:
            raise StudyError('Duplicate product_id')
        products[identity] = product
    rows, dose_ids = [], set()
    for dose in array(recipe.get('doses'), 'doses'):
        dose = obj(dose, 'dose')
        dose_id = text(dose.get('dose_id'), 'dose_id')
        if dose_id in dose_ids:
            raise StudyError('Duplicate dose_id')
        dose_ids.add(dose_id)
        evidence(dose.get('evidence'), 'dose')
        product = products.get(dose.get('product_id'))
        if product is None:
            raise StudyError('Unknown dose product_id')
        composition = product_composition(product, scope)
        amount = _dose_in_basis(dose, product, composition['per_unit'])
        added = {s: composition['mg_per_unit'][s] * amount for s in scope}
        for s in scope:
            mass[s] += added[s]
        rows.append(dict(dose_id=dose_id, **composition, element_mass_mg=added,
                         dose_evidence=deepcopy(dose['evidence'])))
    concentrations = {s: mass[s] / volume for s in scope}
    molar = {s: concentrations[s] / ATOMIC_MASS[SPECIES_ELEMENT[s]] for s in scope}
    n_molar = math.fsum(v for s, v in molar.items() if SPECIES_ELEMENT[s] == 'N')
    # Ratios are only defined if the complete N/P/K scope was declared.
    complete_npk = {'NO3_N', 'NH4_N', 'unspecified_N', 'P', 'K'}.issubset(scope)
    ratio = ({'N': 1.0, 'P': molar['P'] / n_molar, 'K': molar['K'] / n_molar}
             if complete_npk and n_molar > 0 else None)
    return dict(schema_version=SCHEMA, status='available', batch_id=batch,
                quantity_basis='mg_element', tracked_species=scope, final_volume_L=volume,
                element_mass_mg=mass, concentration_mg_L=concentrations,
                concentration_mmol_element_L=molar, npk_molar_ratio_N1=ratio,
                npk_ratio_status='available' if ratio else 'undefined_or_incomplete_scope',
                dose_components=rows, input_hash=digest(recipe), evidence=deepcopy(recipe['evidence']),
                source_water_evidence=deepcopy(recipe['source_water_evidence']),
                interpretation='nominal_formulated_composition_not_measured_root_zone',
                ec_prediction=None, validated_nutrient_model=False)


def formulate(recipe: dict) -> dict:
    """Pure formulation. Absent scientific inputs return structured unavailable."""
    require_json(recipe)
    try:
        result = _formulate(recipe)
        require_json(result)
        return result
    except Unavailable as exc:
        return unavailable(exc, recipe)
