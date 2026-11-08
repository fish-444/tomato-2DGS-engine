"""Engine v2 block C4T (CONTRACT.md Amendment 9): template v2.4 counts (discrete normal, sigma = range/3) and 2-4 side axes per plant.
apply_c4t(text) works on a single-plant v2-patched Model.rgg at level C4 or higher (3 controllers -> 5). Every replacement must occur the stated number of times."""
from __future__ import annotations

import json
import math
from pathlib import Path

H = Path(__file__).resolve().parent
NSIDE_MAX = 4
SIDE_ATTACH = {1: .1, 2: .2, 4: .1, 5: .2, 7: .2, 9: .2}


def pmf(lo, hi, div=3):
    mu = (lo + hi) / 2; s = (hi - lo) / div; w = [math.exp(-0.5 * ((k - mu) / s) ** 2) for k in range(lo, hi + 1)]; t = sum(w)
    return {k: x / t for k, x in zip(range(lo, hi + 1), w)}


PMF = dict(main_truss_node=pmf(5, 9), main_total=pmf(10, 13), side_axes=pmf(2, 4), side_total=pmf(5, 7))


def jarr(d):
    ks = sorted(d); return 'new int[]{' + ','.join(str(k) for k in ks) + '}', 'new double[]{' + ','.join(repr(float(d[k])) for k in ks) + '}'


def apply_c4t(text: str, divergence=None) -> str:
    def rep(old, new, count=1):
        nonlocal text
        assert text.count(old) == count, ('c4t patch point', text.count(old), old[:90])
        text = text.replace(old, new)
    nc = 1 + NSIDE_MAX
    rep('new int[3]', f'new int[{nc}]', 10); rep('new double[3]', f'new double[{nc}]', 4); rep('new boolean[3]', f'new boolean[{nc}]', 3)
    rep('String[] logicalAxisIds=new String[]{"M","S1","S2"};', 'String[] logicalAxisIds=new String[]{"M",' + ','.join(f'"S{j}"' for j in range(1, NSIDE_MAX + 1)) + '};')
    rep('{0,0,0}', '{' + ','.join(['0'] * nc) + '}', 2); rep('{1,1,1}', '{' + ','.join(['1'] * nc) + '}')
    rep('{true,true,true}', '{' + ','.join(['true'] * nc) + '}', 2); rep('{1.0,1.0,1.0}', '{' + ','.join(['1.0'] * nc) + '}')
    i0 = text.index('void v2InitPlant(int p) {'); i1 = text.index('double v2Germ(int c)')
    init = init_text()
    text = text[:i0] + init + text[i1:]
    rep('double v2Germ(int c) { return c==0?GERMINATION_DD:v2BudDelay; }',
        'double v2Germ(int c) { return c==0?GERMINATION_DD:v2BudDelay+(c-1)*PHYLLOCHRON_DD*rpPhyllochron[controllerOwners[c]]; }')
    rep('    int p=controllerOwners[c];int r=v2Attach[c];\n', '    int p=controllerOwners[c];int r=v2Attach[c];if(r<=0)return false;\n')
    if divergence is not None:
        rep('double rpDivergenceLo=119.3,rpDivergenceHi=154.7;', f'double rpDivergenceLo={float(divergence[0])!r},rpDivergenceHi={float(divergence[1])!r};')
    return text


def init_text() -> str:
    vN, sN = jarr(PMF['main_truss_node']); vT, sT = jarr(PMF['main_total']); vS, sS = jarr(PMF['side_total']); vA, sA = jarr(SIDE_ATTACH); vX, sX = jarr(PMF['side_axes'])
    init = f'''int[] v2NSide=new int[1];
int v2CatMin(String id,String key,int[] vals,double[] shares,int min) {{
    java.util.Random r=new java.util.Random(rpSeed+(long)(id+":"+key).hashCode());r.nextDouble();
    double tot=0;for(int k=0;k<shares.length;k++)tot+=shares[k];
    int redraws=0;double u=0;int v=vals[vals.length-1];
    while(true){{
        u=r.nextDouble();double acc=0;v=vals[vals.length-1];
        for(int k=0;k<vals.length;k++){{acc+=shares[k]/tot;if(u<acc){{v=vals[k];break;}}}}
        if(v>=min || redraws>=1000)break;redraws++;
    }}
    rpRecord(id,key,u,v);rpRecord(id,key+"_REDRAWS",0,redraws);return v;
}}
void v2InitPlant(int p) {{
    String id=plantIds[p];
    int n=v2Cat(id,"MAIN_TRUSS_NODE",{vN},{sN});int t=v2CatMin(id,"MAIN_TOTAL_LEAVES",{vT},{sT},n+2);
    v2First[0]=n;v2Rest[0]=t-n;v2Created[0]=true;
    int ns=v2Cat(id,"SIDE_AXES_N",{vX},{sX});v2NSide[p]=ns;
    int[] av={vA};double[] sh={sA};
    for(int j=1;j<={NSIDE_MAX};j++){{
        if(j>ns){{v2Attach[j]=0;continue;}}
        int s=v2Cat(id+"/S"+j,"SIDE_TOTAL_LEAVES",{vS},{sS});v2First[j]=2;v2Rest[j]=s-2;
        v2Attach[j]=v2Cat(id+"/S"+j,"SIDE_ATTACH_RANK",av,sh);
        for(int k=0;k<av.length;k++)if(av[k]==v2Attach[j])sh[k]=0;
    }}
}}
'''
    return init



def config_extra() -> dict:
    return dict(c4t=True, c4t_pmf={k: {str(a): b for a, b in v.items()} for k, v in PMF.items()}, n_side_max=NSIDE_MAX)


if __name__ == '__main__':
    print(json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in PMF.items()}))
