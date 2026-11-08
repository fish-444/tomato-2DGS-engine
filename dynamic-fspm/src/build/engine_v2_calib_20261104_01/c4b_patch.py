"""Engine v2 block C4B (CONTRACT.md Amendment 11): leaflet-surface curvature from MTG_LEAF_CURVATURE_v1.csv, applied on top of C4L.
apply_c4b(text) needs the C4L patch already applied (outlineVertices uses o.v2vx/v2vy/v2vz). Every replacement must occur exactly once."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / 'build' / 'traits_recon_20261101_01' / 'MTG_LEAF_CURVATURE_v1.csv'
EXPECT_SHA = '8cdf7ff764e65440a1854aa8500ee9b1caa69b96313376cb9b74c413f7370119'
POS = ['P1', 'P2', 'P3', 'T']
LREF_MM = {'P1': 17.0926, 'P2': 14.5765, 'P3': 14.3016, 'T': 19.4931}   # MTG_LEAFLET_ARRANGEMENT_v1 length_mm medians (both / C)
QU = [0.01, 0.10, 0.50, 0.90, 0.99]
QK = ['p1', 'p10', 'median', 'p90', 'p99']
KEYS = [('kappa_l', 'LEAFLET_KAPPA_L'), ('kappa_w', 'LEAFLET_KAPPA_W'), ('twist_c', 'LEAFLET_TWIST_C')]


def table():
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == EXPECT_SHA, 'MTG_LEAF_CURVATURE_v1.csv changed'
    body = [l for l in CSV.read_text(encoding='utf-8').splitlines() if not l.startswith('#')]
    t = {}
    for r in csv.DictReader(io.StringIO('\n'.join(body))):
        if r['subset'] == 'all_11' and r['rank_class'] == 'main' and r['position'] in POS:
            t[(r['quantity'], r['position'])] = [float(r[k]) for k in QK]
    return t


def jarr(v):
    return 'new double[]{' + ','.join(repr(float(x)) for x in v) + '}'


def config_extra():
    t = table()
    return dict(c4b=True, c4b_q={f'{q}|{p}': t[(q, p)] for q, _ in KEYS for p in POS}, c4b_qu=QU, c4b_lref_mm=LREF_MM)


def apply_c4b(text: str) -> str:
    t = table()

    def rep(old, new):
        nonlocal text
        assert text.count(old) == 1, ('c4b patch point', text.count(old), old[:90])
        text = text.replace(old, new)
    rep(',v2S=1,', ',v2Kl=0,v2Kw=0,v2Kc=0,v2S=1,')
    tabs = '\n'.join(f'double[][] v2q{name}=new double[][]{{' + ','.join(jarr(t[(q, p)]) for p in POS) + '};' for q, name in KEYS)
    helpers = f'''// ===== C4B leaflet curvature (CONTRACT Amendment 11; table = MTG_LEAF_CURVATURE_v1.csv {EXPECT_SHA[:12]}) =====
double[] v2qCu={jarr(QU)};
{tabs}
double[] v2Lref={jarr([LREF_MM[p] / 1000 for p in POS])};
double v2QuantC(String id,String key,double[] q) {{
    double u=rpU(id,key),v;int n=q.length;
    if(u<=v2qCu[0])v=q[0];else if(u>=v2qCu[n-1])v=q[n-1];
    else {{int k=1;while(k<n-1 && u>v2qCu[k])k++;double f=(u-v2qCu[k-1])/(v2qCu[k]-v2qCu[k-1]);v=q[k-1]+f*(q[k]-q[k-1]);}}
    rpRecord(id,key,u,v);return v;
}}
void v2CurvDraws(Organ l) {{
    int pos=v2PosOf(l);
    l.v2Kl=v2QuantC(l.id,"LEAFLET_KAPPA_L",v2qLEAFLET_KAPPA_L[pos]);l.v2Kw=v2QuantC(l.id,"LEAFLET_KAPPA_W",v2qLEAFLET_KAPPA_W[pos]);l.v2Kc=v2QuantC(l.id,"LEAFLET_TWIST_C",v2qLEAFLET_TWIST_C[pos]);
}}
double[][] v2Centroid=null;
double[] v2OutlineCentroid(int oi,double[] uv) {{
    if(v2Centroid==null)v2Centroid=new double[outlineUV.length][];
    if(v2Centroid[oi]!=null)return v2Centroid[oi];
    double A=0,cx=0,cy=0;
    for(int i=0;i<uv.length;i+=6){{double a=0.5*Math.abs((uv[i+2]-uv[i])*(uv[i+5]-uv[i+1])-(uv[i+4]-uv[i])*(uv[i+3]-uv[i+1]));A+=a;cx+=a*(uv[i]+uv[i+2]+uv[i+4])/3;cy+=a*(uv[i+1]+uv[i+3]+uv[i+5])/3;}}
    v2Centroid[oi]=new double[]{{cx/A,cy/A}};return v2Centroid[oi];
}}
void v2BendInto(Organ o,double[] uv,float[] result,double scale,double ux,double uy,double uz,double vx,double vy,double vz) {{
    double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx,nn=Math.sqrt(nx*nx+ny*ny+nz*nz);nx/=nn;ny/=nn;nz/=nn;
    if(nz<0){{nx=-nx;ny=-ny;nz=-nz;}}
    int pos=v2PosOf(o);double f=o.len>0?v2Lref[pos]/o.len:1.0;
    double kl=o.v2Kl*1000*f,kw=o.v2Kw*1000*f,kc=o.v2Kc*1000*f;
    double[] cc=v2OutlineCentroid(outlineIndex(o),uv);double uc=cc[0],vc=cc[1];
    int nv=uv.length/2;double[] px=new double[nv],py=new double[nv],pz=new double[nv];
    double target=0;for(int i=0;i<uv.length;i+=6){{target+=0.5*Math.abs((uv[i+2]-uv[i])*(uv[i+5]-uv[i+1])-(uv[i+4]-uv[i])*(uv[i+3]-uv[i+1]));}}target*=scale*scale;
    double lam=1;
    for(int it=0;it<60;it++){{
        double s0=-lam*scale*uc,d0=-lam*scale*vc,wb=0.5*kl*s0*s0+0.5*kw*d0*d0+kc*s0*d0;
        for(int i=0;i<nv;i++){{
            double a=uv[2*i],b=uv[2*i+1],s=lam*scale*(a-uc),d=lam*scale*(b-vc),w=0.5*kl*s*s+0.5*kw*d*d+kc*s*d-wb;
            px[i]=o.x+lam*scale*(a*ux+b*vx)+w*nx;py[i]=o.y+lam*scale*(a*uy+b*vy)+w*ny;pz[i]=o.z+lam*scale*(a*uz+b*vz)+w*nz;
        }}
        double A=0;
        for(int i=0;i<nv;i+=3){{double ax=px[i+1]-px[i],ay=py[i+1]-py[i],az=pz[i+1]-pz[i],bx=px[i+2]-px[i],by=py[i+2]-py[i],bz=pz[i+2]-pz[i];
            double cx=ay*bz-az*by,cy=az*bx-ax*bz,cz=ax*by-ay*bx;A+=0.5*Math.sqrt(cx*cx+cy*cy+cz*cz);}}
        if(A<=0 || Math.abs(A/target-1)<1e-13)break;
        lam*=Math.sqrt(target/A);
    }}
    for(int i=0;i<nv;i++){{result[3*i]=(float)px[i];result[3*i+1]=(float)py[i];result[3*i+2]=(float)pz[i];}}
}}
'''
    anchor = 'void v2PlaceLeaflets() {\n'
    rep(anchor, helpers + anchor)
    rep('l.v2Comp=c.idx;v2LeafletDraws(l);', 'l.v2Comp=c.idx;v2LeafletDraws(l);v2CurvDraws(l);')
    rep('terminal.v2Comp=c.idx;v2LeafletDraws(terminal);', 'terminal.v2Comp=c.idx;v2LeafletDraws(terminal);v2CurvDraws(terminal);')
    rep('''    for(int i=0;i<uv.length;i+=2){int j=i/2*3;
        result[j]=(float)(o.x+scale*(uv[i]*ux+uv[i+1]*vx));
        result[j+1]=(float)(o.y+scale*(uv[i]*uy+uv[i+1]*vy));
        result[j+2]=(float)(o.z+scale*(uv[i]*uz+uv[i+1]*vz));
    }
''', '    v2BendInto(o,uv,result,scale,ux,uy,uz,vx,vy,vz);\n')
    return text
