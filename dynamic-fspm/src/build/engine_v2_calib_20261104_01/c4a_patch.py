"""Engine v2 block C4A Step C (CONTRACT.md Amendment 17): rachis base length multiplier per main-axis rank class.
apply_rachis(text, mult) on a V22b-type Model.rgg (C4C applied, simultaneous leaflet growth): rachis segment length
= RACHIS_M / m * v2RMult(o) * own funded fraction; main controller ranks 1..len(mult) use mult[rank-1], every other organ 1.0. Petiole unchanged."""
from __future__ import annotations

OLD = 'int m=pairsOf(o);o.len=(o.kind==7?PETIOLE_M:RACHIS_M/m)*fraction;o.wid=RACHIS_WIDTH_M*fraction;'
NEW = 'int m=pairsOf(o);o.len=(o.kind==7?PETIOLE_M:RACHIS_M/m*v2RMult(o))*fraction;o.wid=RACHIS_WIDTH_M*fraction;'
ANCHOR = 'double pairWeight(int i,int m,double ramp)'


def apply_rachis(text: str, mult: list[float]) -> str:
    assert text.count(OLD) == 1, ('c4a patch point', text.count(OLD))
    assert text.count(ANCHOR) == 1
    arr = ','.join(repr(float(x)) for x in mult)
    fn = ('double[] v2RMultMain=new double[]{' + arr + '};\n'
          'double v2RMult(Organ o) { if(o.controller!=0)return 1.0; int r=o.rank; return (r>=1 && r<=v2RMultMain.length)?v2RMultMain[r-1]:1.0; }   // C4A Step C\n')
    return text.replace(OLD, NEW).replace(ANCHOR, fn + ANCHOR)


def config_extra(mult: list[float]) -> dict:
    return dict(c4a_rachis_main=[float(x) for x in mult])
