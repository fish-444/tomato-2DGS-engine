"""Engine v2 block C4L (CONTRACT.md Amendment 8): leaflet arrangement from MTG_LEAFLET_ARRANGEMENT_v1.csv.
apply_c4l(text) adds the rules on top of an already v2-patched Model.rgg (level C4U or higher). Every replacement must occur exactly once."""
from __future__ import annotations

import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'build' / 'traits_recon_20261101_01' / 'MTG_LEAFLET_ARRANGEMENT_v1.csv'
EXPECT_SHA = 'faa67d5e926003814d7a6105b31b002aebf6b8e3d93e6245c8d11b2a8cd1daba'
POS = ['P1', 'P2', 'P3', 'T']
QKEYS = ['p1', 'p10', 'p25', 'median', 'p75', 'p90', 'p99']
QU = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
CHAIN_MEDIAN_MM = 38.2264


def load():
    import hashlib
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == EXPECT_SHA, 'MTG_LEAFLET_ARRANGEMENT_v1.csv changed'
    body = [l for l in CSV.read_text(encoding='utf-8').splitlines() if not l.startswith('#')]
    return list(csv.DictReader(io.StringIO('\n'.join(body))))


def table(rows):
    """dict[(quantity, pos, side)] -> row of floats; side L/R for P1-P3, C for T."""
    t = {}
    for r in rows:
        if r['subset'] == 'all_11' and r['rank_class'] == 'main' and r['position'] in POS:
            t[(r['quantity'], r['position'], r['side'])] = {k: float(r[k]) for k in ['mean', 'sd', 'median', 'p1', 'p10', 'p25', 'p75', 'p90', 'p99']}
    return t


def jarr(v):
    return 'new double[]{' + ','.join(repr(float(x)) for x in v) + '}'


def java_helpers(t, rows):
    ptype = lambda q, pos: [(pos, s) for s in (('C',) if pos == 'T' else ('L', 'R'))]
    lines = ['// ===== C4L leaflet arrangement (build/engine_v2_calib_20261104_01 CONTRACT Amendment 8; table = MTG_LEAFLET_ARRANGEMENT_v1.csv ' + EXPECT_SHA[:12] + ') =====']
    # quantile tables indexed [pos 0..3][side 0..1] (T uses side 0)
    def qtab(q, upto):
        out = []
        for pos in POS:
            row = []
            for s in (('C', 'C') if pos == 'T' else ('L', 'R')):
                r = t[(q, pos, s)]; row.append(jarr([r[k] for k in QKEYS[:upto]]))
            out.append('new double[][]{' + ','.join(row) + '}')
        return 'new double[][][]{' + ','.join(out) + '}'
    lines.append('double[] v2qU=' + jarr(QU) + ';')
    lines.append('double[][][] v2qS=' + qtab('s_over_L', 6) + ';')
    lines.append('double[][][] v2qOff=' + qtab('petiolule_offset_mm', 6) + ';')
    lines.append('double[][][] v2qTilt=' + qtab('tilt_deg', 7) + ';')
    phi = [[[t[('spread_phi_deg', pos, s)][k] for k in ('mean', 'sd', 'p1', 'p99')] for s in (('C', 'C') if pos == 'T' else ('L', 'R'))] for pos in POS]
    lines.append('double[][][] v2phi=new double[][][]{' + ','.join('new double[][]{' + ','.join(jarr(x) for x in side) + '}' for side in phi) + '};')
    roll = [[(1 + t[('roll_sign', pos, s)]['mean']) / 2 for s in (('C', 'C') if pos == 'T' else ('L', 'R'))] for pos in POS]
    lines.append('double[][] v2pRoll=new double[][]{' + ','.join(jarr(x) for x in roll) + '};')
    share = {}
    for r in rows:
        if r['subset'] == 'all_11' and r['rank_class'] == 'main' and r['quantity'] == 'area_share' and r['side'] in ('both', 'C') and r['position'] in POS:
            share[r['position']] = float(r['median'])
    lines.append('double[] v2Share=' + jarr([share[p] for p in POS]) + f';double v2ChainMedian={CHAIN_MEDIAN_MM / 1000!r};')
    lines.append('''double v2Quant(String id,String key,double[] q,int n) {
    double u=rpU(id,key),v;
    if(u<=v2qU[0])v=q[0];else if(u>=v2qU[n-1])v=q[n-1];
    else {int k=1;while(k<n-1 && u>v2qU[k])k++;double f=(u-v2qU[k-1])/(v2qU[k]-v2qU[k-1]);v=q[k-1]+f*(q[k]-q[k-1]);}
    rpRecord(id,key,u,v);return v;
}
int v2PosOf(Organ o) { return o.side==0?3:Math.min(o.pair,3)-1; }
int v2SideIx(Organ o) { return o.side<0?0:(o.side>0?1:0); }
double v2ShareOfPos(int pos) { return v2Share[pos]; }
double v2ShareTotal(Organ c,int m) {
    double t=v2Share[3];
    for(int i=1;i<=m;i++)t+=(v2Omit(c,i,-1)?1:2)*v2Share[Math.min(i,3)-1];
    return t;
}
void v2LeafletDraws(Organ l) {
    int pos=v2PosOf(l),sx=v2SideIx(l);
    if(l.side==0) {
        l.v2S=1.0;l.v2Phi=0;l.v2Off=0;
    } else {
        l.v2S=Math.max(0,Math.min(1,v2Quant(l.id,"LEAFLET_S_OVER_L",v2qS[pos][sx],6)));
        double[] ph=v2phi[pos][sx];java.util.Random r=new java.util.Random(rpSeed+(long)(l.id+":LEAFLET_SPREAD_PHI_DEG").hashCode());r.nextDouble();double z=r.nextGaussian();
        l.v2Phi=Math.max(ph[2],Math.min(ph[3],ph[0]+ph[1]*z));rpRecord(l.id,"LEAFLET_SPREAD_PHI_DEG",z,l.v2Phi);
        l.v2Off=v2Quant(l.id,"LEAFLET_PETIOLULE_MM",v2qOff[pos][sx],6)/1000.0;
    }
    l.v2Tilt=v2Quant(l.id,"LEAFLET_TILT_DEG",v2qTilt[pos][sx],7);
    double ur=rpU(l.id,"LEAFLET_ROLL_SIGN");l.v2Roll=ur<v2pRoll[pos][sx]?1.0:-1.0;rpRecord(l.id,"LEAFLET_ROLL_SIGN",ur,l.v2Roll);
}
void v2PlaceLeaflets() {
    for(int j=0;j<organs.size();j++){
        Organ o=org(j);
        if(o.kind!=9 || o.imported || o.emerged<0 || o.area<=0 || o.v2Comp<0)continue;
        Organ c=org(o.v2Comp);int[] ch=c.v2Chain;double Lc=0;
        for(int k=0;k<ch.length;k++)Lc+=org(ch[k]).len;
        Organ q=org(ch[ch.length-1]);double fr=1.0;
        if(Lc>0){
            double target=o.v2S*Lc,acc=0;int k=0;
            for(;k<ch.length-1;k++){double L=org(ch[k]).len;if(acc+L>=target)break;acc+=L;}
            q=org(ch[k]);double L=q.len;fr=L>0?Math.max(0,Math.min(1,(target-acc)/L)):1.0;
        }
        double bx=q.x+fr*(q.tx-q.x),by=q.y+fr*(q.ty-q.y),bz=q.z+fr*(q.tz-q.z);
        double pa=Math.toRadians(q.az),pe=Math.toRadians(q.elev);
        double tx=Math.cos(pe)*Math.cos(pa),ty=Math.cos(pe)*Math.sin(pa),tz=Math.sin(pe);
        double hx=-Math.sin(pa),hy=Math.cos(pa),hz=0;
        double nx=-Math.sin(pe)*Math.cos(pa),ny=-Math.sin(pe)*Math.sin(pa),nz=Math.cos(pe);
        double ph=Math.toRadians(o.v2Phi),sd=o.side;
        double ux,uy,uz;
        if(o.side==0){ux=tx;uy=ty;uz=tz;}
        else {ux=Math.cos(ph)*tx+sd*Math.sin(ph)*hx;uy=Math.cos(ph)*ty+sd*Math.sin(ph)*hy;uz=Math.cos(ph)*tz+sd*Math.sin(ph)*hz;}
        double off=o.v2Off*Lc/v2ChainMedian;
        bx+=sd*off*hx;by+=sd*off*hy;bz+=sd*off*hz;
        double v0x=ny*uz-nz*uy,v0y=nz*ux-nx*uz,v0z=nx*uy-ny*ux;
        double ang=Math.toRadians(o.v2Roll*o.v2Tilt),cs=Math.cos(ang),sn=Math.sin(ang);
        o.v2vx=cs*v0x+sn*nx;o.v2vy=cs*v0y+sn*ny;o.v2vz=cs*v0z+sn*nz;
        o.x=bx;o.y=by;o.z=bz;
        o.az=(Math.toDegrees(Math.atan2(uy,ux))+360)%360;o.elev=Math.toDegrees(Math.asin(Math.max(-1,Math.min(1,uz))));
        double az=Math.toRadians(o.az),el=Math.toRadians(o.elev);
        o.tx=o.x+o.len*Math.cos(el)*Math.cos(az);o.ty=o.y+o.len*Math.cos(el)*Math.sin(az);o.tz=o.z+o.len*Math.sin(el);
    }
}''')
    return '\n'.join(lines) + '\n'


def apply_c4l(text: str) -> str:
    rows = load(); t = table(rows)

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, ('c4l patch point', text.count(old), old[:90])
        text = text.replace(old, new)
    rep('    int v2Pairs=0,v2Leaflets=0; double v2Len=-1', '    int[] v2Chain=null; int v2Comp=-1; double v2vx=0,v2vy=0,v2vz=0,v2S=1,v2Phi=0,v2Off=0,v2Tilt=0,v2Roll=1;\n    int v2Pairs=0,v2Leaflets=0; double v2Len=-1')
    anchor = 'boolean v2Omit(Organ c,int i,int side) { return c.v2Leaflets==6 && i==c.v2Pairs && side<0; }\n'
    rep(anchor, anchor + java_helpers(t, rows))
    rep('    int previous=pet.idx;\n    double total=1;\n    for(int i=1;i<=m;i++) total+=(v2Omit(c,i,-1)?1:2)*pairWeight(i,m,c.rpRamp);\n',
        '    int previous=pet.idx;java.util.ArrayList v2ch=new java.util.ArrayList();v2ch.add(Integer.valueOf(pet.idx));\n    double total=v2ShareTotal(c,m);\n')
    rep('            l.pair=i;l.side=side;l.rpBasal=c.rpBasal;l.share=pairWeight(i,m,c.rpRamp)/total;l.maxMass=MAX_MASS[9]*l.share*rpMaxMass(l.id)*v2LeafScale(node.controller,rank);\n',
        '            l.pair=i;l.side=side;l.rpBasal=c.rpBasal;l.share=v2ShareOfPos(Math.min(i,3)-1)/total;l.maxMass=MAX_MASS[9]*l.share*rpMaxMass(l.id)*v2LeafScale(node.controller,rank);l.v2Comp=c.idx;v2LeafletDraws(l);\n')
    rep('        previous=r.idx;\n    }\n', '        previous=r.idx;v2ch.add(Integer.valueOf(r.idx));\n    }\n')
    rep('terminal.share=1/total;terminal.maxMass=MAX_MASS[9]/total*rpMaxMass(terminal.id)*v2LeafScale(node.controller,rank);\n}\n',
        'terminal.share=v2ShareOfPos(3)/total;terminal.maxMass=MAX_MASS[9]*terminal.share*rpMaxMass(terminal.id)*v2LeafScale(node.controller,rank);terminal.v2Comp=c.idx;v2LeafletDraws(terminal);\n'
        '    c.v2Chain=new int[v2ch.size()];for(int q=0;q<v2ch.size();q++)c.v2Chain[q]=((Integer)v2ch.get(q)).intValue();\n}\n')
    rep('\n    }\n}\nFloatList vertices(Organ o) {', '\n    }\n    v2PlaceLeaflets();\n}\nFloatList vertices(Organ o) {')
    rep('    Organ par=org(o.parent);double pa=Math.toRadians(par.az),pe=Math.toRadians(par.elev);\n    int m=pairsOf(o);double beta=Math.toRadians(o.side*rpBasalOf(o)*(m+1-o.pair)/m);\n    double vx=-Math.sin(beta)*Math.cos(pe)*Math.cos(pa)-Math.cos(beta)*Math.sin(pa);\n    double vy=-Math.sin(beta)*Math.cos(pe)*Math.sin(pa)+Math.cos(beta)*Math.cos(pa);\n    double vz=-Math.sin(beta)*Math.sin(pe);\n',
        '    double vx=o.v2vx,vy=o.v2vy,vz=o.v2vz;\n')
    return text
