"""Engine v2 patch (CONTRACT.md §2): cumulative levels B0 < C1 < C2 < C3 < C4 < C5 applied to a prepared v1 Model.rgg (single plant, 1 controller).
Every replacement must occur exactly once (else AssertionError). All values come from rules v2.2 (build/mtg_template_v2_2_20261104_01).
apply(model_text, level, sla_factor=1.0, test=None) -> new text;  params(param_text, level, sla_factor) -> new parametersMicroTom text.
test = ('p1/C0003', 30.0) forces one compound's theta0 (C2 one-leaf test arm C2T)."""
from __future__ import annotations

import json

LEVELS = ['B0', 'C1', 'C2', 'C3', 'C4', 'C4U', 'C5']
# ---- rules v2.2 values (read once, written here verbatim; v2_config.json records them with the rules hash)
LEAFLET_MAIN = {1: {5: .333, 6: 0.0, 7: .667}, 2: {5: 0.0, 6: 0.0, 7: 1.0}, 3: {5: 0.6663336663336663, 6: 0.16683316683316682, 7: 0.16683316683316682},
                4: {5: .4, 6: 0.0, 7: .6}, 5: {5: .2, 6: 0.0, 7: .8}, 6: {5: .2, 6: .2, 7: .6}, 7: {5: .4, 6: 0.0, 7: .6}, 8: {5: .667, 6: 0.0, 7: .333},
                9: {5: 1.0, 6: 0.0, 7: 0.0}}
LEAFLET_FALLBACK = {5: 0.3863863863863864, 6: 0.04504504504504504, 7: 0.5685685685685685}
LEAFLET_SIDE = {5: .722, 7: .278}
THETA_MAIN = {1: (3.33, 6.06), 2: (26.67, 23.38), 3: (37.5, 28.24), 4: (15.0, 10.0), 5: (38.0, 14.83), 6: (28.0, 20.49), 7: (36.0, 15.17), 8: (26.67, 15.28), 9: (26.67, 15.28)}
THETA_FALLBACK = (26.14, 20.14); THETA_SIDE = (20.56, 17.98); THETA_CLIP = (-5.0, 80.0)
BEND = (32.764, 10.602); BEND_CLIP = (0.0, 90.0)
BETA = dict(a=5.024584763293008, b=0.9754152367069917, lnB=-1.5576497286764441, lo_mm=2.0, hi_mm=4.0)
MAIN_TRUSS_NODE = {5: 1 / 9, 6: 2 / 9, 7: 3 / 9, 8: 2 / 9, 9: 1 / 9}
MAIN_TOTAL = {10: .25, 11: .25, 12: .25, 13: .25}
SIDE_TOTAL = {5: 1 / 3, 6: 1 / 3, 7: 1 / 3}
SIDE_ATTACH = {1: .1, 2: .2, 4: .1, 5: .2, 7: .2, 9: .2}
SIDE_FIRST_TRUSS = 2; SIDE_ELEV_DEG = 40.0; BUD_DELAY_DD = 150.0; N_SIDE = 2
FLOWERS = {1: 1 / 9, 3: 1 / 9, 4: 7 / 9}
SLA_BASE = 0.025
LOW_LEAF_SCALE = 0.5; INTERNODE_SCALE = 1.25; INTERNODE_SCALE_FROM_RANK = 5   # C4U, user_stated_2026-11-05 (CONTRACT Amendment 3)


def config(level: str, sla_factor: float = 1.0, low_leaf: float = LOW_LEAF_SCALE, inter_all: float = 1.0, flowers_fixed: int = 0, c4l: bool = False, c4t: bool = False, divergence=None, c4b: bool = False) -> dict:
    return dict(level=level, sla_m2_g=SLA_BASE * sla_factor if LEVELS.index(level) >= LEVELS.index('C5') else SLA_BASE, sla_factor=sla_factor,
                leaflet_main={str(k): v for k, v in LEAFLET_MAIN.items()}, leaflet_fallback=LEAFLET_FALLBACK, leaflet_side=LEAFLET_SIDE,
                theta_main={str(k): v for k, v in THETA_MAIN.items()}, theta_fallback=THETA_FALLBACK, theta_side=THETA_SIDE, theta_clip=THETA_CLIP,
                bend=BEND, bend_clip=BEND_CLIP, beta=BETA, main_truss_node=MAIN_TRUSS_NODE, main_total=MAIN_TOTAL, side_total=SIDE_TOTAL,
                side_attach=SIDE_ATTACH, side_first_truss=SIDE_FIRST_TRUSS, side_elev_deg=SIDE_ELEV_DEG, bud_delay_dd=BUD_DELAY_DD, n_side=N_SIDE,
                flowers=FLOWERS, c4u=LEVELS.index(level) >= LEVELS.index('C4U'), low_leaf_scale=low_leaf, internode_scale_all=inter_all, flowers_fixed=flowers_fixed, c4l=c4l, divergence=divergence, **(__import__('c4b_patch').config_extra() if c4b else {}), **(__import__('c4t_patch').config_extra() if c4t else {}), internode_scale=INTERNODE_SCALE, internode_scale_from_rank=INTERNODE_SCALE_FROM_RANK, v2_c1=LEVELS.index(level) >= 1, v2_c2=LEVELS.index(level) >= 2, v2_c3=LEVELS.index(level) >= 3, v2_c4=LEVELS.index(level) >= 4)


def _jarr(d: dict) -> tuple[str, str]:
    ks = sorted(d); return 'new int[]{' + ','.join(str(k) for k in ks) + '}', 'new double[]{' + ','.join(repr(float(d[k])) for k in ks) + '}'


def helpers(level: str, test, low_leaf: float = LOW_LEAF_SCALE, inter_all: float = 1.0, flowers_fixed: int = 0) -> str:
    i = LEVELS.index(level); out = ['// ===== Engine v2 (build/engine_v2_calib_20261104_01, CONTRACT.md) level ' + level + ' =====']
    out.append('''int v2Cat(String id,String key,int[] vals,double[] shares) {
    double u=rpU(id,key),acc=0,tot=0;for(int k=0;k<shares.length;k++)tot+=shares[k];
    for(int k=0;k<vals.length;k++){acc+=shares[k]/tot;if(u<acc){rpRecord(id,key,u,vals[k]);return vals[k];}}
    rpRecord(id,key,u,vals[vals.length-1]);return vals[vals.length-1];
}
double v2Normal(String id,String key,double mean,double sd,double lo,double hi) {
    java.util.Random r=new java.util.Random(rpSeed+(long)(id+":"+key).hashCode());r.nextDouble();double z=r.nextGaussian();
    double v=Math.max(lo,Math.min(hi,mean+sd*z));rpRecord(id,key,z,v);return v;
}
int v2PairsOf(Organ o) { Organ c=o;while(c.kind!=4 && c.parent>=0)c=org(c.parent);return c.kind==4 && c.v2Pairs>0?c.v2Pairs:pairs(o.owner,o.rank); }
boolean v2Omit(Organ c,int i,int side) { return c.v2Leaflets==6 && i==c.v2Pairs && side<0; }''')
    rows = []
    for r in range(1, 10):
        v, s = _jarr(LEAFLET_MAIN[r]); rows.append(f'    if(rank=={r})return v2Cat(id,"LEAFLET_COUNT",{v},{s});')
    vf, sf = _jarr(LEAFLET_FALLBACK); vs, ss = _jarr(LEAFLET_SIDE)
    out.append('int v2LeafletCount(String id,int controller,int rank) {\n    if(controller>0)return v2Cat(id,"LEAFLET_COUNT",' + vs + ',' + ss + ');\n' + '\n'.join(rows) +
               f'\n    return v2Cat(id,"LEAFLET_COUNT",{vf},{sf});\n}}')
    if i >= 2:
        rows = [f'    if(rank=={r}){{m={THETA_MAIN[r][0]!r};s={THETA_MAIN[r][1]!r};}}' for r in range(1, 10)]
        tid, tval = (test if test else ('', 0.0))
        out.append(f'String v2TestId="{tid}";double v2TestTheta={float(tval)!r};\n'
                   'double v2Theta0(String id,int controller,int rank) {\n'
                   f'    double m={THETA_FALLBACK[0]!r},s={THETA_FALLBACK[1]!r};\n    if(controller>0){{m={THETA_SIDE[0]!r};s={THETA_SIDE[1]!r};}}\n    else {{\n' + '\n'.join(rows) + '\n    }\n'
                   f'    double v=v2Normal(id,"THETA0_DEG",m,s,{THETA_CLIP[0]!r},{THETA_CLIP[1]!r});\n'
                   '    if(id.equals(v2TestId)){v=v2TestTheta;rpRecord(id,"THETA0_TEST_OVERRIDE",0,v);}\n    return v;\n}')
    if i >= 3:
        out.append(f'double v2Bend(String id) {{ return v2Normal(id,"TOTAL_BEND_DEG",{BEND[0]!r},{BEND[1]!r},{BEND_CLIP[0]!r},{BEND_CLIP[1]!r}); }}')
    if i >= 4:
        vN, sN = _jarr(MAIN_TRUSS_NODE); vT, sT = _jarr(MAIN_TOTAL); vS, sS = _jarr(SIDE_TOTAL); vA, sA = _jarr(SIDE_ATTACH); vF, sF = _jarr(FLOWERS)
        out.append(f'''double v2BetaA={BETA['a']!r},v2BetaB={BETA['b']!r},v2LnB={BETA['lnB']!r};
double v2Betacf(double a,double b,double x) {{
    double qab=a+b,qap=a+1,qam=a-1,c=1,d=1-qab*x/qap;if(Math.abs(d)<1e-300)d=1e-300;d=1/d;double h=d;
    for(int m=1;m<=300;m++){{int m2=2*m;double aa=m*(b-m)*x/((qam+m2)*(a+m2));
        d=1+aa*d;if(Math.abs(d)<1e-300)d=1e-300;c=1+aa/c;if(Math.abs(c)<1e-300)c=1e-300;d=1/d;h*=d*c;
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
        d=1+aa*d;if(Math.abs(d)<1e-300)d=1e-300;c=1+aa/c;if(Math.abs(c)<1e-300)c=1e-300;d=1/d;double de=d*c;h*=de;
        if(Math.abs(de-1)<3e-16)break;}}
    return h;
}}
double v2Ibeta(double x) {{
    if(x<=0)return 0;if(x>=1)return 1;
    double bt=Math.exp(v2BetaA*Math.log(x)+v2BetaB*Math.log(1-x)-v2LnB);
    return x<(v2BetaA+1)/(v2BetaA+v2BetaB+2)?bt*v2Betacf(v2BetaA,v2BetaB,x)/v2BetaA:1-bt*v2Betacf(v2BetaB,v2BetaA,1-x)/v2BetaB;
}}
double v2Internode(String id,double scale) {{
    double u=rpU(id,"INTERNODE_MM"),lo=0,hi=1;
    for(int k=0;k<80;k++){{double mid=0.5*(lo+hi);if(v2Ibeta(mid)<u)lo=mid;else hi=mid;}}
    double mm=scale*({BETA['lo_mm']!r}+({BETA['hi_mm']!r}-{BETA['lo_mm']!r})*0.5*(lo+hi));rpRecord(id,"INTERNODE_MM",u,mm);return mm/1000.0;
}}
int[] v2First=new int[3],v2Rest=new int[3],v2Attach=new int[3];boolean[] v2Created=new boolean[3];double[] v2BranchAz=new double[3];
double v2SideElev={SIDE_ELEV_DEG!r},v2BudDelay={BUD_DELAY_DD!r};
double v2LowLeaf={(low_leaf if i >= 5 else 1.0)!r},v2InterScale={(INTERNODE_SCALE if i >= 5 else 1.0)!r};int v2InterFrom={INTERNODE_SCALE_FROM_RANK};
double v2LeafScale(int controller,int rank) {{ return controller==0 && rank==1?v2LowLeaf:1.0; }}
double v2InterScaleOf(int controller,int rank) {{ return {('' if inter_all == 1.0 else repr(float(inter_all)) + '*(')}controller==0 && rank>=v2InterFrom?v2InterScale:1.0{('' if inter_all == 1.0 else ')')}; }}
int v2Flowers(String trussId) {{ return {('v2Cat(trussId,"FLOWERS_PER_TRUSS",' + vF + ',' + sF + ')') if not flowers_fixed else str(int(flowers_fixed))}; }}
void v2InitPlant(int p) {{
    String id=plantIds[p];
    int n=v2Cat(id,"MAIN_TRUSS_NODE",{vN},{sN});int t=v2Cat(id,"MAIN_TOTAL_LEAVES",{vT},{sT});
    v2First[0]=n;v2Rest[0]=t-n;v2Created[0]=true;
    int[] av={vA};double[] as={sA};
    for(int j=1;j<=2;j++){{
        int s=v2Cat(id+"/S"+j,"SIDE_TOTAL_LEAVES",{vS},{sS});v2First[j]={SIDE_FIRST_TRUSS};v2Rest[j]=s-{SIDE_FIRST_TRUSS};
        double[] sh=new double[as.length];for(int k=0;k<as.length;k++)sh[k]=(j==2 && av[k]==v2Attach[1])?0:as[k];
        v2Attach[j]=v2Cat(id+"/S"+j,"SIDE_ATTACH_RANK",av,sh);
    }}
}}
double v2Germ(int c) {{ return c==0?GERMINATION_DD:v2BudDelay; }}
boolean v2TryCreate(int c,int day) {{
    int p=controllerOwners[c];int r=v2Attach[c];
    if(trussCounts[0]<1 || leafOrders[0]<r)return false;
    Organ node=null,leaf=null;
    for(int j=0;j<organs.size();j++){{Organ o=org(j);if(o.owner==p && o.controller==0 && o.rank==r){{if(o.kind==3)node=o;if(o.kind==4)leaf=o;}}}}
    if(node==null || leaf==null)throw new IllegalStateException("v2 side axis attach node missing");
    Organ ax=add(p,node.idx,-1,1,0,day,"S"+c+"/A0");ax.axis=ax.idx;ax.controller=c;ax.segment=currentSegments[c];
    Organ ap=add(p,ax.idx,ax.idx,2,0,day,"S"+c+"/A0/apex");
    activeAxis[c]=ax.idx;activeApex[c]=ap.idx;lastStem[c]=ax.idx;parentRanks[c]=r;
    v2BranchAz[c]=((leaf.rpAz%360)+360)%360;v2Created[c]=true;clocks[c]=0;
    return true;
}}''')
    return '\n'.join(out) + '\n'


def apply(text: str, level: str, sla_factor: float = 1.0, test=None, low_leaf: float = LOW_LEAF_SCALE, inter_all: float = 1.0, flowers_fixed: int = 0, c4l: bool = False, c4t: bool = False, divergence=None, c4b: bool = False) -> str:
    i = LEVELS.index(level)
    if i == 0:
        return text

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, ('patch point', text.count(old), old[:90])
        text = text.replace(old, new)
    rep('    boolean imported=false;\n', '    boolean imported=false;\n    int v2Pairs=0,v2Leaflets=0; double v2Len=-1;\n')
    anchor = 'double rpBasalOf(Organ o) { return o.imported?BASAL_LEAFLET_ANGLE_DEG:o.rpBasal; }\n'
    rep(anchor, anchor + helpers(level, test, low_leaf, inter_all, flowers_fixed))
    # C1 leaflets per compound
    rep('int pairsOf(Organ o) { if(o.imported)return PAIRS[Math.min(o.rank-1,PAIRS.length-1)];return pairs(o.owner,o.rank); }',
        'int pairsOf(Organ o) { if(o.imported)return PAIRS[Math.min(o.rank-1,PAIRS.length-1)];return v2PairsOf(o); }')
    rep('    int m=pairs(p,rank);\n', '    c.v2Leaflets=v2LeafletCount(c.id,node.controller,rank);int m=c.v2Leaflets/2;c.v2Pairs=m;\n')
    rep('    for(int i=1;i<=m;i++) total+=2*pairWeight(i,m,c.rpRamp);\n', '    for(int i=1;i<=m;i++) total+=(v2Omit(c,i,-1)?1:2)*pairWeight(i,m,c.rpRamp);\n')
    rep('        for(int side=-1;side<=1;side+=2) {\n            Organ l=add(', '        for(int side=-1;side<=1;side+=2) {\n            if(v2Omit(c,i,side))continue;\n            Organ l=add(')
    if i >= 2:
        rep('c.rpInsertion=rpDraw(c.id,"COMPOUND_INSERTION_DEG",rpInsertionLo,rpInsertionHi);', 'c.rpInsertion=90-v2Theta0(c.id,node.controller,rank);')
    if i >= 3:
        rep('c.bendAmplitude=rpDraw(c.id,"TOTAL_BEND_DEG",rpBendLo,rpBendHi);', 'c.bendAmplitude=v2Bend(c.id);')
    if i >= 4:
        for old, new in [('double[] clocks=new double[1];', 'double[] clocks=new double[3];'),
                         ('int[] produced=new int[1],leafOrders=new int[1],trussCounts=new int[1],axisLeafCounts=new int[1];', 'int[] produced=new int[3],leafOrders=new int[3],trussCounts=new int[3],axisLeafCounts=new int[3];'),
                         ('int[] activeApex=new int[1],plantRoots=new int[1],activeAxis=new int[1],lastStem=new int[1];', 'int[] activeApex=new int[3],plantRoots=new int[1],activeAxis=new int[3],lastStem=new int[3];'),
                         ('int[] currentSegments=new int[]{1};', 'int[] currentSegments=new int[]{1,1,1};'),
                         ('boolean[] trussHistoryKnown=new boolean[]{true};', 'boolean[] trussHistoryKnown=new boolean[]{true,true,true};'),
                         ('int[] controllerOwners=new int[]{0};', 'int[] controllerOwners=new int[]{0,0,0};'),
                         ('String[] logicalAxisIds=new String[]{"M"};', 'String[] logicalAxisIds=new String[]{"M","S1","S2"};'),
                         ('int[] parentRanks=new int[]{0};', 'int[] parentRanks=new int[]{0,0,0};'),
                         ('double[] sinkScales=new double[]{1.0};', 'double[] sinkScales=new double[]{1.0,1.0,1.0};'),
                         ('boolean[] axisPrograms=new boolean[]{true};', 'boolean[] axisPrograms=new boolean[]{true,true,true};'),
                         ('double[] rpAzimuth=new double[1];', 'double[] rpAzimuth=new double[3];'),
                         ('boolean[] rpStarted=new boolean[1];', 'boolean[] rpStarted=new boolean[3];'),
                         ('boolean[] s1NewAxes=new boolean[1];', 'boolean[] s1NewAxes=new boolean[3];'),
                         ('double[] s1GermClocks=new double[1];', 'double[] s1GermClocks=new double[3];')]:
            rep(old, new)
        rep('String axisPrefix(int controller) {return axesEnabled && !logicalAxisIds[controller].equals("M")?logicalAxisIds[controller]+"/":"";}',
            'String axisPrefix(int controller) {return !logicalAxisIds[controller].equals("M")?logicalAxisIds[controller]+"/":"";}')
        rep('        double h=rpU(id,"HANDEDNESS");rpHand[p]=h<0.5?-1:1;rpRecord(id,"HANDEDNESS",h,rpHand[p]);\n',
            '        double h=rpU(id,"HANDEDNESS");rpHand[p]=h<0.5?-1:1;rpRecord(id,"HANDEDNESS",h,rpHand[p]);\n        v2InitPlant(p);\n')
        rep('    int p=controllerOwners[controller];\n    Organ apex=org(activeApex[controller]);\n',
            '    int p=controllerOwners[controller];\n    if(controller>0 && !v2Created[controller] && !v2TryCreate(controller,day))return;\n    Organ apex=org(activeApex[controller]);\n')
        rep('    if(clocks[controller]<GERMINATION_DD)return;\n    int due=(int)Math.floor((clocks[controller]-GERMINATION_DD)/(PHYLLOCHRON_DD*rpPhyllochron[p]));',
            '    if(clocks[controller]<v2Germ(controller))return;\n    int due=(int)Math.floor((clocks[controller]-v2Germ(controller))/(PHYLLOCHRON_DD*rpPhyllochron[p]));')
        rep('        int target=trussCounts[controller]==0?FIRST_LEAVES:CADENCE[Math.min(trussCounts[controller]-1,CADENCE.length-1)];',
            '        int target=trussCounts[controller]==0?v2First[controller]:(trussCounts[controller]==1?v2Rest[controller]:0);')
        rep('            int rank=++leafOrders[controller];\n', '            int rank=(controller>0?100*controller:0)+(++leafOrders[controller]);\n')
        rep('            Organ stem=add(p,lastStem[controller],axis,6,rank,day,prefix+"I"+rank);lastStem[controller]=stem.idx;\n',
            '            Organ stem=add(p,lastStem[controller],axis,6,rank,day,prefix+"I"+rank);lastStem[controller]=stem.idx;stem.v2Len=v2Internode(stem.id,v2InterScaleOf(controller,rank));\n')
        rep('            for(int f=1;f<=FLOWERS;f++) { Organ flower=', '            int v2nf=v2Flowers(truss.id);for(int f=1;f<=v2nf;f++) { Organ flower=')
        rep('if(!rpStarted[node.controller]){rpStarted[node.controller]=true;rpAzimuth[node.controller]=((rank-1)*DIVERGENCE_DEG)%360;}',
            'if(!rpStarted[node.controller]){rpStarted[node.controller]=true;rpAzimuth[node.controller]=node.controller>0?v2BranchAz[node.controller]:((rank-1)*DIVERGENCE_DEG)%360;}')
        rep('        if(o.kind==6) {o.len=rpInternode[o.owner]*fraction;o.wid=STEM_WIDTH_M*fraction;o.elev=90;}',
            '        if(o.kind==6) {o.len=(o.v2Len>0?o.v2Len:rpInternode[o.owner])*fraction;o.wid=STEM_WIDTH_M*fraction;o.elev=o.controller>0?v2SideElev:90;if(o.controller>0)o.az=v2BranchAz[o.controller];}')
        rep('            for(int c=0;c<controllerOwners.length;c++)if(controllerOwners[c]==p)clocks[c]+=dd;',
            '            for(int c=0;c<controllerOwners.length;c++)if(controllerOwners[c]==p && v2Created[c])clocks[c]+=dd;')
    if i >= 4:
        rep('l.pair=i;l.side=side;l.rpBasal=c.rpBasal;l.share=pairWeight(i,m,c.rpRamp)/total;l.maxMass=MAX_MASS[9]*l.share*rpMaxMass(l.id);',
            'l.pair=i;l.side=side;l.rpBasal=c.rpBasal;l.share=pairWeight(i,m,c.rpRamp)/total;l.maxMass=MAX_MASS[9]*l.share*rpMaxMass(l.id)*v2LeafScale(node.controller,rank);')
        rep('terminal.maxMass=MAX_MASS[9]/total*rpMaxMass(terminal.id);', 'terminal.maxMass=MAX_MASS[9]/total*rpMaxMass(terminal.id)*v2LeafScale(node.controller,rank);')
    if i >= LEVELS.index('C5'):
        rep('public static final double SLA_M2_G = 0.025;', f'public static final double SLA_M2_G = {SLA_BASE * sla_factor!r};')
    if c4t:
        import c4t_patch
        text = c4t_patch.apply_c4t(text, divergence)
    if c4l:
        import c4l_patch
        text = c4l_patch.apply_c4l(text)
    if c4b:
        assert c4l, 'C4B needs C4L'
        import c4b_patch
        text = c4b_patch.apply_c4b(text)
    return text


def params(text: str, level: str, sla_factor: float = 1.0) -> str:
    """parametersMicroTom_v2.rgg: same constants as Model_v2.rgg (only SLA changes as a constant; the rule changes live in Model.rgg)."""
    if LEVELS.index(level) >= LEVELS.index('C5'):
        assert text.count('public static final double SLA_M2_G = 0.025;') == 1
        text = text.replace('public static final double SLA_M2_G = 0.025;', f'public static final double SLA_M2_G = {SLA_BASE * sla_factor!r};')
    return text


if __name__ == '__main__':
    import sys
    from pathlib import Path
    src = Path(sys.argv[1]).read_text(encoding='utf-8')
    for lv in LEVELS:
        t = apply(src, lv, 1.0, ('p1/C0003', 30.0) if lv == 'C2' else None); print(lv, len(t), t.count('v2'))
    print(json.dumps(config('C5', 1.0))[:200])
