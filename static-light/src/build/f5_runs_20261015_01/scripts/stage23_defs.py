"""Declared Stage-2/3 case definitions (LOG CLAIM extension, rules S2-1..S3-4). Imported by the run and prepare-check scripts."""
from pathlib import Path
ROOT = Path(r'<PROJECT_ROOT>')
PLAN = ROOT / 'build' / 'budbreak_plan_20261015_f4' / 'plan.csv'
OUT2 = ROOT / 'build' / 'f4_stage2_20261015_01'
OUT3 = ROOT / 'build' / 'f4_stage3_20261015_01'
CELL = 'p0.3_d300_upper'
CMP = 'handoff/F3_LITERATURE_COMPARISON_2026-10-15.md'

LIT_EXPANSION = {
    'LEAFLET_MID_DD': dict(value=250, unit='degree_day', source_kind='derived_low_confidence',
                           source_ref=f'{CMP} proposal 3 (Twiggy Red beta tm 271 / te 464 degC d, Butturini 2026, T3; Tbase 5.7 identical) - F4 Stage 3 declared expansion scenario',
                           cultivar='Twiggy Red (other cultivar), converted to a logistic midpoint',
                           uncertainty='derived low confidence; not fitted to Micro-Tom; scenario only'),
    'LEAFLET_K': dict(value=0.014, unit='1/degree_day', source_kind='derived_low_confidence',
                      source_ref=f'{CMP} proposal 3 (95 % at about 460 degC d, Twiggy Red, Butturini 2026, T3) - F4 Stage 3 declared expansion scenario',
                      cultivar='Twiggy Red (other cultivar), converted to a logistic rate',
                      uncertainty='derived low confidence; not fitted to Micro-Tom; scenario only'),
}

def phyllochron(v):
    return {'PHYLLOCHRON_DD': dict(value=v, unit='degree_day', source_kind='derived_low_confidence',
                                   source_ref=f"{CMP} row 'leaf appearance speed' (rough Micro-Tom greenhouse estimate 77-135 degC d per leaf, Mol Hortic 2022 Fig. S3 text) - F4 Stage 3 declared sensitivity level",
                                   cultivar='Micro-Tom greenhouse rough estimate (not this host, not measured here)',
                                   uncertainty='derived low confidence; sensitivity level only; does not retime the host state or the bud plan')}

CAP = {'reserve_cap': True}
OFF = {'reserve_cap': False}
STAGE2 = ([('control_capon', None, None, CAP)] + [(f'{CELL}_m{m:02d}_capon', CELL, m, CAP) for m in range(4)]
          + [('control_capoff', None, None, OFF)] + [(f'{CELL}_m{m:02d}_capoff', CELL, m, OFF) for m in range(4)])  # S2-2: hashes differ -> cap-off re-run
STAGE2_KEEP = f'{CELL}_m00_capon'

STAGE3 = []
for exp in ('cur', 'lit'):
    for ph in (37.9, 77, 135):
        if exp == 'cur' and ph == 37.9:
            continue  # IS Stage-2 cap-on m00/m01 (identical request), reused
        ov = {}
        if exp == 'lit': ov.update(LIT_EXPANSION)
        if ph != 37.9: ov.update(phyllochron(ph))
        for m in range(2):
            STAGE3.append((f'{CELL}_m{m:02d}_capon_exp{exp}_ph{ph:g}', CELL, m, dict(reserve_cap=True, overrides=ov)))
STAGE3_KEEP = f'{CELL}_m00_capon_explit_ph77'
