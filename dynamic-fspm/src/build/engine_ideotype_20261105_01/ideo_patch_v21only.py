"""Ideotype patch (CONTRACT.md §1-§6 + Amendment 1): the frozen Engine **v2.1** rules (build/engine_v2_calib_20261104_01: v2_patch level C5,
SLA x1.15, lowest leaf x0.7, internodes x5.85, 4 flowers, C4T template v2.4 with up to 4 side axes, divergence 90-100, C4L, C4B) generalised to
n plants, plus per-plant ideotype multipliers with allometric cost and the flat luminaire. Every replacement must occur exactly once.
apply_multi(text, n, ideo=None, luminaire=None) -> new Model.rgg text
  ideo = dict(plants=[0-based plant indices], IN=k_internode, LS=k_rachis, PE=k_petiole, beta=0.5)  (k = 1.0 means unchanged)
  luminaire = dict(z=0.40, length=0.60, width=0.10, power=1200.0)  (None keeps the prepared point lamp)
Controllers: main c = p (0..n-1); side axis j (1..4) of plant p: c = n + 4p + (j - 1)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1] / 'engine_v2_calib_20261104_01'
sys.path.insert(0, str(V2))
import v2_patch as P  # noqa: E402
import c4t_patch as CT  # noqa: E402
import c4l_patch as CL  # noqa: E402
import c4b_patch as CB  # noqa: E402

V21 = dict(level='C5', sla_factor=1.15, low_leaf=0.7, inter_all=5.85, flowers_fixed=4, divergence=(90.0, 100.0))
NS = CT.NSIDE_MAX   # 4 side axes max per plant


def _helpers(n: int) -> str:
    h = P.helpers(V21['level'], None, V21['low_leaf'], V21['inter_all'], V21['flowers_fixed'])
    nc = (1 + NS) * n

    def rep(old, new, count=1):
        nonlocal h
        assert h.count(old) == count, ('helper point', h.count(old), old[:80]); h = h.replace(old, new)
    rep('    if(controller>0)return v2Cat(id,"LEAFLET_COUNT",', '    if(v2IsSide(controller))return v2Cat(id,"LEAFLET_COUNT",')
    rep('    if(controller>0){m=', '    if(v2IsSide(controller)){m=')
    rep('int[] v2First=new int[3],v2Rest=new int[3],v2Attach=new int[3];boolean[] v2Created=new boolean[3];double[] v2BranchAz=new double[3];',
        f'int v2Np={n};boolean v2IsSide(int c) {{ return c>=v2Np; }} int v2J(int c) {{ return c<v2Np?0:(c-v2Np)%{NS}+1; }} int v2SideC(int p,int j) {{ return v2Np+{NS}*p+(j-1); }}\n'
        f'int[] v2First=new int[{nc}],v2Rest=new int[{nc}],v2Attach=new int[{nc}];boolean[] v2Created=new boolean[{nc}];double[] v2BranchAz=new double[{nc}];')
    rep('double v2LeafScale(int controller,int rank) { return controller==0 && rank==1?v2LowLeaf:1.0; }',
        'double v2LeafScale(int controller,int rank) { return !v2IsSide(controller) && rank==1?v2LowLeaf:1.0; }')
    s = repr(float(V21['inter_all']))
    rep(f'double v2InterScaleOf(int controller,int rank) {{ return {s}*(controller==0 && rank>=v2InterFrom?v2InterScale:1.0); }}',
        f'double v2InterScaleOf(int controller,int rank) {{ return {s}*(!v2IsSide(controller) && rank>=v2InterFrom?v2InterScale:1.0); }}')
    # C4T plant init (n plants): take the single-plant init text and map the controller indices
    init = CT.init_text()
    for old, new in [('int[] v2NSide=new int[1];', f'int[] v2NSide=new int[{n}];'),
                     ('    v2First[0]=n;v2Rest[0]=t-n;v2Created[0]=true;', '    v2First[p]=n;v2Rest[p]=t-n;v2Created[p]=true;'),
                     ('        if(j>ns){v2Attach[j]=0;continue;}', '        if(j>ns){v2Attach[v2SideC(p,j)]=0;continue;}'),
                     ('v2First[j]=2;v2Rest[j]=s-2;', 'v2First[v2SideC(p,j)]=2;v2Rest[v2SideC(p,j)]=s-2;'),
                     ('        v2Attach[j]=v2Cat(', '        v2Attach[v2SideC(p,j)]=v2Cat('),
                     ('if(av[k]==v2Attach[j])sh[k]=0;', 'if(av[k]==v2Attach[v2SideC(p,j)])sh[k]=0;')]:
        assert init.count(old) == 1, ('init point', old); init = init.replace(old, new)
    i0 = h.index('void v2InitPlant(int p) {'); i1 = h.index('double v2Germ(int c)')
    h = h[:i0] + init + h[i1:]
    rep('double v2Germ(int c) { return c==0?GERMINATION_DD:v2BudDelay; }',
        'double v2Germ(int c) { return !v2IsSide(c)?GERMINATION_DD:v2BudDelay+(v2J(c)-1)*PHYLLOCHRON_DD*rpPhyllochron[controllerOwners[c]]; }')
    rep('    int p=controllerOwners[c];int r=v2Attach[c];\n', '    int p=controllerOwners[c];int r=v2Attach[c];if(r<=0)return false;\n')
    rep('    if(trussCounts[0]<1 || leafOrders[0]<r)return false;', '    if(trussCounts[p]<1 || leafOrders[p]<r)return false;')
    rep('if(o.owner==p && o.controller==0 && o.rank==r)', 'if(o.owner==p && o.controller==p && o.rank==r)')
    rep('    Organ ax=add(p,node.idx,-1,1,0,day,"S"+c+"/A0");', '    Organ ax=add(p,node.idx,-1,1,0,day,"S"+v2J(c)+"/A0");')
    rep('    Organ ap=add(p,ax.idx,ax.idx,2,0,day,"S"+c+"/A0/apex");', '    Organ ap=add(p,ax.idx,ax.idx,2,0,day,"S"+v2J(c)+"/A0/apex");')
    return h


def _ideo_code(n: int, ideo) -> str:
    plants = set(ideo['plants']) if ideo else set()
    k = lambda key: (float(ideo.get(key, 1.0)) if ideo else 1.0)
    beta = float(ideo.get('beta', 0.5)) if ideo else 0.5
    arr = lambda key: 'new double[]{' + ','.join(repr(k(key) if p in plants else 1.0) for p in range(n)) + '}'
    return (f'// ideotype (build/engine_ideotype_20261105_01 CONTRACT §3-4): per-plant multipliers; allometry d = d0*k^beta, max mass x k^(1+2 beta)\n'
            f'double[] ioIN={arr("IN")},ioLS={arr("LS")},ioPE={arr("PE")};double ioBeta={beta!r};\n'
            'double ioMass(double k) { return k==1.0?1.0:Math.pow(k,1+2*ioBeta); }\n'
            'double ioWid(double k) { return k==1.0?1.0:Math.pow(k,ioBeta); }\n')


def apply_multi(text: str, n: int, ideo=None, luminaire=None) -> str:
    nc = (1 + NS) * n

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, ('patch point', text.count(old), old[:90])
        text = text.replace(old, new)

    def rep_re(pat, new):
        nonlocal text
        got = re.findall(pat, text); assert len(got) == 1, ('patch regex', len(got), pat[:90]); text = re.sub(pat, new, text)
    zeros = lambda v: ','.join([v] * n)
    rep('    boolean imported=false;\n', '    boolean imported=false;\n    int v2Pairs=0,v2Leaflets=0; double v2Len=-1,ioWidScale=1.0;\n')
    anchor = 'double rpBasalOf(Organ o) { return o.imported?BASAL_LEAFLET_ANGLE_DEG:o.rpBasal; }\n'
    rep(anchor, anchor + _helpers(n) + _ideo_code(n, ideo))
    # C1
    rep('int pairsOf(Organ o) { if(o.imported)return PAIRS[Math.min(o.rank-1,PAIRS.length-1)];return pairs(o.owner,o.rank); }',
        'int pairsOf(Organ o) { if(o.imported)return PAIRS[Math.min(o.rank-1,PAIRS.length-1)];return v2PairsOf(o); }')
    rep('    int m=pairs(p,rank);\n', '    c.v2Leaflets=v2LeafletCount(c.id,node.controller,rank);int m=c.v2Leaflets/2;c.v2Pairs=m;\n')
    rep('    for(int i=1;i<=m;i++) total+=2*pairWeight(i,m,c.rpRamp);\n', '    for(int i=1;i<=m;i++) total+=(v2Omit(c,i,-1)?1:2)*pairWeight(i,m,c.rpRamp);\n')
    rep('        for(int side=-1;side<=1;side+=2) {\n            Organ l=add(', '        for(int side=-1;side<=1;side+=2) {\n            if(v2Omit(c,i,side))continue;\n            Organ l=add(')
    # C2, C3
    rep('c.rpInsertion=rpDraw(c.id,"COMPOUND_INSERTION_DEG",rpInsertionLo,rpInsertionHi);', 'c.rpInsertion=90-v2Theta0(c.id,node.controller,rank);')
    rep('c.bendAmplitude=rpDraw(c.id,"TOTAL_BEND_DEG",rpBendLo,rpBendHi);', 'c.bendAmplitude=v2Bend(c.id);')
    # controller arrays (n plants -> (1 + 4) n controllers)
    rep_re(r'double\[\] clocks=new double\[\d+\];', f'double[] clocks=new double[{nc}];')
    rep_re(r'int\[\] produced=new int\[\d+\],leafOrders=new int\[\d+\],trussCounts=new int\[\d+\],axisLeafCounts=new int\[\d+\];',
           f'int[] produced=new int[{nc}],leafOrders=new int[{nc}],trussCounts=new int[{nc}],axisLeafCounts=new int[{nc}];')
    rep_re(r'int\[\] activeApex=new int\[\d+\],plantRoots=new int\[\d+\],activeAxis=new int\[\d+\],lastStem=new int\[\d+\];',
           f'int[] activeApex=new int[{nc}],plantRoots=new int[{n}],activeAxis=new int[{nc}],lastStem=new int[{nc}];')
    rep('int[] currentSegments=new int[]{' + zeros('1') + '};', 'int[] currentSegments=new int[]{' + ','.join(['1'] * nc) + '};')
    rep('boolean[] trussHistoryKnown=new boolean[]{' + zeros('true') + '};', 'boolean[] trussHistoryKnown=new boolean[]{' + ','.join(['true'] * nc) + '};')
    rep('int[] controllerOwners=new int[]{' + ','.join(str(p) for p in range(n)) + '};',
        'int[] controllerOwners=new int[]{' + ','.join([str(p) for p in range(n)] + [str(p) for p in range(n) for _ in range(NS)]) + '};')
    rep('String[] logicalAxisIds=new String[]{' + zeros('"M"') + '};', 'String[] logicalAxisIds=new String[]{' + ','.join(['"M"'] * n + [f'"S{j}"' for j in range(1, NS + 1)] * n) + '};')
    rep('int[] parentRanks=new int[]{' + zeros('0') + '};', 'int[] parentRanks=new int[]{' + ','.join(['0'] * nc) + '};')
    rep('double[] sinkScales=new double[]{' + zeros('1.0') + '};', 'double[] sinkScales=new double[]{' + ','.join(['1.0'] * nc) + '};')
    rep('boolean[] axisPrograms=new boolean[]{' + zeros('true') + '};', 'boolean[] axisPrograms=new boolean[]{' + ','.join(['true'] * nc) + '};')
    for name, typ in (('rpAzimuth', 'double'), ('rpStarted', 'boolean'), ('s1NewAxes', 'boolean'), ('s1GermClocks', 'double')):
        rep_re(typ + r'\[\] ' + name + r'=new ' + typ + r'\[\d+\];', f'{typ}[] {name}=new {typ}[{nc}];')
    rep('String axisPrefix(int controller) {return axesEnabled && !logicalAxisIds[controller].equals("M")?logicalAxisIds[controller]+"/":"";}',
        'String axisPrefix(int controller) {return !logicalAxisIds[controller].equals("M")?logicalAxisIds[controller]+"/":"";}')
    rep('        double h=rpU(id,"HANDEDNESS");rpHand[p]=h<0.5?-1:1;rpRecord(id,"HANDEDNESS",h,rpHand[p]);\n',
        '        double h=rpU(id,"HANDEDNESS");rpHand[p]=h<0.5?-1:1;rpRecord(id,"HANDEDNESS",h,rpHand[p]);\n        v2InitPlant(p);\n')
    rep('    int p=controllerOwners[controller];\n    Organ apex=org(activeApex[controller]);\n',
        '    int p=controllerOwners[controller];\n    if(v2IsSide(controller) && !v2Created[controller] && !v2TryCreate(controller,day))return;\n    Organ apex=org(activeApex[controller]);\n')
    rep('    if(clocks[controller]<GERMINATION_DD)return;\n    int due=(int)Math.floor((clocks[controller]-GERMINATION_DD)/(PHYLLOCHRON_DD*rpPhyllochron[p]));',
        '    if(clocks[controller]<v2Germ(controller))return;\n    int due=(int)Math.floor((clocks[controller]-v2Germ(controller))/(PHYLLOCHRON_DD*rpPhyllochron[p]));')
    rep('        int target=trussCounts[controller]==0?FIRST_LEAVES:CADENCE[Math.min(trussCounts[controller]-1,CADENCE.length-1)];',
        '        int target=trussCounts[controller]==0?v2First[controller]:(trussCounts[controller]==1?v2Rest[controller]:0);')
    rep('            int rank=++leafOrders[controller];\n', '            int rank=100*v2J(controller)+(++leafOrders[controller]);\n')
    rep('            Organ stem=add(p,lastStem[controller],axis,6,rank,day,prefix+"I"+rank);lastStem[controller]=stem.idx;\n',
        '            Organ stem=add(p,lastStem[controller],axis,6,rank,day,prefix+"I"+rank);lastStem[controller]=stem.idx;'
        'stem.v2Len=v2Internode(stem.id,v2InterScaleOf(controller,rank)*ioIN[p]);if(ioIN[p]!=1.0){stem.maxMass*=ioMass(ioIN[p]);stem.ioWidScale=ioWid(ioIN[p]);}\n')
    rep('            for(int f=1;f<=FLOWERS;f++) { Organ flower=', '            int v2nf=v2Flowers(truss.id);for(int f=1;f<=v2nf;f++) { Organ flower=')
    rep('if(!rpStarted[node.controller]){rpStarted[node.controller]=true;rpAzimuth[node.controller]=((rank-1)*DIVERGENCE_DEG)%360;}',
        'if(!rpStarted[node.controller]){rpStarted[node.controller]=true;rpAzimuth[node.controller]=v2IsSide(node.controller)?v2BranchAz[node.controller]:((rank-1)*DIVERGENCE_DEG)%360;}')
    rep('        if(o.kind==6) {o.len=rpInternode[o.owner]*fraction;o.wid=STEM_WIDTH_M*fraction;o.elev=90;}',
        '        if(o.kind==6) {o.len=(o.v2Len>0?o.v2Len:rpInternode[o.owner])*fraction;o.wid=STEM_WIDTH_M*o.ioWidScale*fraction;o.elev=v2IsSide(o.controller)?v2SideElev:90;if(v2IsSide(o.controller))o.az=v2BranchAz[o.controller];}')
    rep('            for(int c=0;c<controllerOwners.length;c++)if(controllerOwners[c]==p)clocks[c]+=dd;',
        '            for(int c=0;c<controllerOwners.length;c++)if(controllerOwners[c]==p && v2Created[c])clocks[c]+=dd;')
    # C4V lowest leaf (C4L replaces these share lines afterwards, keeping the leaf scale)
    rep('l.pair=i;l.side=side;l.rpBasal=c.rpBasal;l.share=pairWeight(i,m,c.rpRamp)/total;l.maxMass=MAX_MASS[9]*l.share*rpMaxMass(l.id);',
        'l.pair=i;l.side=side;l.rpBasal=c.rpBasal;l.share=pairWeight(i,m,c.rpRamp)/total;l.maxMass=MAX_MASS[9]*l.share*rpMaxMass(l.id)*v2LeafScale(node.controller,rank);')
    rep('terminal.maxMass=MAX_MASS[9]/total*rpMaxMass(terminal.id);', 'terminal.maxMass=MAX_MASS[9]/total*rpMaxMass(terminal.id)*v2LeafScale(node.controller,rank);')
    # ideotype: rachis segments (LS) and petiole (PE): max mass x k^(1+2b), length x k, width x k^b
    rep('Organ r=add(p,previous,node.axis,8,rank,day,key+"/R"+i);r.pair=i;r.maxMass=MAX_MASS[8]/m*rpMaxMass(r.id);',
        'Organ r=add(p,previous,node.axis,8,rank,day,key+"/R"+i);r.pair=i;r.maxMass=MAX_MASS[8]/m*rpMaxMass(r.id)*ioMass(ioLS[p]);r.ioWidScale=ioWid(ioLS[p]);')
    rep('    Organ pet=add(p,c.idx,node.axis,7,rank,day,key+"/P");\n', '    Organ pet=add(p,c.idx,node.axis,7,rank,day,key+"/P");if(ioPE[p]!=1.0){pet.maxMass*=ioMass(ioPE[p]);pet.ioWidScale=ioWid(ioPE[p]);}\n')
    rep('            int m=pairsOf(o);o.len=(o.kind==7?PETIOLE_M:RACHIS_M/m)*leafPairFraction(o);o.wid=RACHIS_WIDTH_M*fraction;',
        '            int m=pairsOf(o);o.len=(o.kind==7?PETIOLE_M*ioPE[o.owner]:RACHIS_M/m*ioLS[o.owner])*leafPairFraction(o);o.wid=RACHIS_WIDTH_M*o.ioWidScale*fraction;')
    # SLA, divergence
    rep('public static final double SLA_M2_G = 0.025;', f'public static final double SLA_M2_G = {P.SLA_BASE * V21["sla_factor"]!r};')
    rep('double rpDivergenceLo=119.3,rpDivergenceHi=154.7;', f'double rpDivergenceLo={V21["divergence"][0]!r},rpDivergenceHi={V21["divergence"][1]!r};')
    # C4L, C4B (organ rules, plant-agnostic)
    text = CL.apply_c4l(text)
    text = CB.apply_c4b(text)
    if luminaire:
        old = re.findall(r'\[ Axiom ==> World \[ Translate\(0\.0f,0\.0f,[0-9.]+f\) Lamp\(0\)\.\(setLight\(new PointLight\(\)\.\(setPower\([0-9.]+f\)\)\)\) \]; \]', text)
        assert len(old) == 1, old
        L, W, z, pw = luminaire['length'], luminaire['width'], luminaire['z'], luminaire['power']
        mat = [1, 0, 0, 0, 0, 0, 1, -L / 2, 0, -1, 0, z, 0, 0, 0, 1]   # local x -> world x, local +y (emitting) -> world -z, local z -> world +y
        node = ('[ Axiom ==> World [ Null().(setTransform(new javax.vecmath.Matrix4d(' + ','.join(repr(float(v)) for v in mat) + '))) '
                f'Parallelogram({L!r}f,{W!r}f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower({pw!r}f),setExponent(0.0f)))) ]; ]')
        text = text.replace(old[0], node)
    return text


def params(text: str) -> str:
    return P.params(text, V21['level'], V21['sla_factor'])


def config(n, ideo=None, luminaire=None) -> dict:
    c = P.config(V21['level'], V21['sla_factor'], V21['low_leaf'], V21['inter_all'], V21['flowers_fixed'], True, True, V21['divergence'], True)
    c.update(n_plants=n, ideo=ideo, luminaire=luminaire); return c
