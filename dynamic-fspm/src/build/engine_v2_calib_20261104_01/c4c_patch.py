"""Engine v2 block C4C (CONTRACT.md Amendment 14, user request 2026-11-06): compound leaf 40 mm and equal leaflets within a compound.
* fully funded petiole + rachis = 40 mm, split 20 : 35 as before (PETIOLE_M, RACHIS_M in Model.rgg and parametersMicroTom.rgg);
* every leaflet of a compound: area share 1/n and maximum mass = mean of the compound's drawn leaflet maxima (draws still made and recorded).
Applied after v2_patch.apply (needs C4L, which defines v2Share). Every replacement must occur exactly once."""
from __future__ import annotations

CHAIN_M = 0.040
PETIOLE_M = CHAIN_M * 20 / 55
RACHIS_M = CHAIN_M * 35 / 55
OLD_PET = 'public static final double PETIOLE_M = 0.02;'
OLD_RAC = 'public static final double RACHIS_M = 0.035;'
OLD_SHARE = 'double[] v2Share=new double[]{0.108926,0.146984,0.13449,0.191132};'
TERMINAL_TAIL = 'terminal.v2Comp=c.idx;v2LeafletDraws(terminal);v2CurvDraws(terminal);\n'
EQUALISE = ('    {double v2mm=0;int v2n=0;for(int j=pet.idx;j<=terminal.idx;j++){Organ q=org(j);if(q.kind==9 && q.v2Comp==c.idx){v2mm+=q.maxMass;v2n++;}}\n'
            '     v2mm/=v2n;for(int j=pet.idx;j<=terminal.idx;j++){Organ q=org(j);if(q.kind==9 && q.v2Comp==c.idx)q.maxMass=v2mm;}}   // C4C: equal leaflets\n')


def _rep(text, old, new):
    assert text.count(old) == 1, ('c4c patch point', text.count(old), old[:90])
    return text.replace(old, new)


def apply_c4c(text: str) -> str:
    text = _rep(text, OLD_PET, f'public static final double PETIOLE_M = {PETIOLE_M!r};')
    text = _rep(text, OLD_RAC, f'public static final double RACHIS_M = {RACHIS_M!r};')
    text = _rep(text, OLD_SHARE, 'double[] v2Share=new double[]{0.25,0.25,0.25,0.25};')
    text = _rep(text, TERMINAL_TAIL, TERMINAL_TAIL + EQUALISE)
    return text


def params(text: str) -> str:
    text = _rep(text, OLD_PET, f'public static final double PETIOLE_M = {PETIOLE_M!r};')
    return _rep(text, OLD_RAC, f'public static final double RACHIS_M = {RACHIS_M!r};')


def config_extra() -> dict:
    return dict(c4c=dict(chain_m=CHAIN_M, petiole_m=PETIOLE_M, rachis_m=RACHIS_M, equal_leaflets=True))
